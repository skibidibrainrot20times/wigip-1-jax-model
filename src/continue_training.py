#the given .pkl file contains all the optimizer states and stuff out there. So you can contunue training.
#This training scipt is made for TPU (model was trained on TPU v5e-8) so you are recommended to use that.
# Assuming you are doing in cloud or using Jupyter Notebooks
!pip install -U pyarrow datasets
!pip install tqdm

# === SCRIPT TO CONTINUE TRAINING FOR 7 HOURS ===

import jax, jax.numpy as jnp, flax.linen as nn, optax, numpy as np, pickle, os, time, glob, gzip
from jax import random
from flax.training import train_state
from functools import partial
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
# Model Parameters
N_LAYER = 24; N_EMBD = 1280; N_HEAD = 16; BLOCK_SIZE = 256
BATCH_SIZE = 16
# --- This Session's Training Goal ---
HOURS_TO_RUN = 8.0
STEPS_PER_HOUR = 38000 # Based on your previous performance
SAVE_INTERVAL_HOURS = 3.0 # Save a backup every 3 hours
# --- Overall Project Goal & Hyperparameters ---
TOTAL_STEPS_FOR_LR_SCHEDULE = 610000 # The original ~2.5B token goal
LEARNING_RATE = 3e-4; WARMUP_STEPS = 200
# Precision
PARAM_DTYPE = jnp.float32; COMPUTE_DTYPE = jnp.bfloat16
CHECKPOINT_PREFIX = '/kaggle/working/training_state_step_'

# --- DATA & VOCAB ---
print("Loading dataset..."); dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
print("Building vocabulary..."); processed_text_sample = ""
for i, example in enumerate(dataset.take(1000)): processed_text_sample += example['text']
chars = sorted(list(set(processed_text_sample))); vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}; encode = lambda s: [stoi.get(c, 0) for c in s]
print(f"Vocabulary size: {vocab_size}")

# --- MODEL DEFINITION ---
class CausalSelfAttention(nn.Module):
    n_head:int; n_embd:int; param_dtype:jnp.dtype; dtype:jnp.dtype
    @nn.compact
    def __call__(self,x):
        B,T,C=x.shape; qkv=nn.Dense(features=3*self.n_embd,use_bias=False,param_dtype=self.param_dtype,dtype=self.dtype)(x); q,k,v=jnp.array_split(qkv,3,axis=-1); q=q.reshape(B,T,self.n_head,C//self.n_head).transpose(0,2,1,3); k=k.reshape(B,T,self.n_head,C//self.n_head).transpose(0,2,1,3); v=v.reshape(B,T,self.n_head,C//self.n_head).transpose(0,2,1,3); tril=jnp.tril(jnp.ones((T,T),dtype=jnp.bool_)).reshape(1,1,T,T); att=(q@k.transpose(0,1,3,2))/jnp.sqrt(k.shape[-1]); att=jnp.where(tril,att,-jnp.inf); att=nn.softmax(att,axis=-1).astype(self.dtype); y=(att@v).transpose(0,2,1,3).reshape(B,T,C); return nn.Dense(features=C,param_dtype=self.param_dtype,dtype=self.dtype)(y)
class Block(nn.Module):
    n_head:int; n_embd:int; param_dtype:jnp.dtype; dtype:jnp.dtype
    @nn.compact
    def __call__(self,x):
        x=x+CausalSelfAttention(n_head=self.n_head,n_embd=self.n_embd,param_dtype=self.param_dtype,dtype=self.dtype)(nn.LayerNorm(dtype=self.dtype,use_scale=False,use_bias=False)(x)); ffwd=nn.Sequential([nn.Dense(4*self.n_embd,param_dtype=self.param_dtype,dtype=self.dtype),nn.relu,nn.Dense(self.n_embd,param_dtype=self.param_dtype,dtype=self.dtype)]); x=x+ffwd(nn.LayerNorm(dtype=self.dtype,use_scale=False,use_bias=False)(x)); return x
class Transformer(nn.Module):
    n_layer:int; n_head:int; n_embd:int; block_size:int; vocab_size:int
    param_dtype:jnp.dtype=PARAM_DTYPE; dtype:jnp.dtype=COMPUTE_DTYPE
    @nn.compact
    def __call__(self,idx):
        B,T=idx.shape; tok_emb=nn.Embed(num_embeddings=self.vocab_size,features=self.n_embd,param_dtype=self.param_dtype)(idx); pos_emb=self.param('pos_emb',nn.initializers.normal(stddev=0.02),(1,self.block_size,self.n_embd),self.param_dtype); x=tok_emb+pos_emb[:,:T,:];
        for _ in range(self.n_layer): x=Block(n_head=self.n_head,n_embd=self.n_embd,param_dtype=self.param_dtype,dtype=self.dtype)(x)
        x=nn.LayerNorm(dtype=self.dtype,use_scale=False,use_bias=False)(x); return nn.Dense(features=self.vocab_size,param_dtype=self.param_dtype,dtype=jnp.float32)(x)

# --- TRAINING LOGIC ---
def get_stream_batches(streaming_dataset):
    buffer = [];
    for example in streaming_dataset:
        tokens = encode(example['text']); buffer.extend(tokens)
        while len(buffer) >= (BATCH_SIZE * BLOCK_SIZE) + 1:
            all_chunks = np.array(buffer[:BATCH_SIZE * BLOCK_SIZE + 1]); x = all_chunks[:-1].reshape(BATCH_SIZE, BLOCK_SIZE); y = all_chunks[1:].reshape(BATCH_SIZE, BLOCK_SIZE); buffer = buffer[BATCH_SIZE * BLOCK_SIZE:]; yield x, y
class TrainState(train_state.TrainState):pass
def create_train_state(rng, total_steps_for_lr_schedule):
    model=Transformer(N_LAYER,N_HEAD,N_EMBD,BLOCK_SIZE,vocab_size); params=model.init(rng,jnp.zeros((BATCH_SIZE,BLOCK_SIZE),dtype=jnp.int32))['params']
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params)); print(f"Model parameter count: {param_count / 1e6:.2f}M")
    lr_schedule=optax.warmup_cosine_decay_schedule(init_value=0.0,peak_value=LEARNING_RATE,warmup_steps=WARMUP_STEPS,decay_steps=total_steps_for_lr_schedule-WARMUP_STEPS,end_value=LEARNING_RATE/10)
    tx=optax.chain(optax.clip_by_global_norm(1.0),optax.adafactor(learning_rate=lr_schedule)); return TrainState.create(apply_fn=model.apply,params=params,tx=tx)
@jax.jit
def train_step(state,x,y):
    def loss_fn(params):
        logits=state.apply_fn({'params':params},x.astype(jnp.int32)); return optax.softmax_cross_entropy_with_integer_labels(logits,y.astype(jnp.int32)).mean()
    loss,grads=jax.value_and_grad(loss_fn)(state.params); return state.apply_gradients(grads=grads),loss

# --- RUN TRAINING ---
state = create_train_state(random.PRNGKey(0), TOTAL_STEPS_FOR_LR_SCHEDULE)
start_step = 0

existing_checkpoints = glob.glob(f"{CHECKPOINT_PREFIX}*.pkl.gz")
if existing_checkpoints:
    latest_checkpoint = max(existing_checkpoints, key=lambda p: int(p.split('_')[-1].split('.')[0]))
    print(f"Resuming training from {latest_checkpoint}...")
    with gzip.open(latest_checkpoint, 'rb') as f:
        saved_data = pickle.load(f)
    state = state.replace(params=saved_data['params'], opt_state=saved_data['opt_state'], step=saved_data['step'])
    start_step = int(state.step)
    print(f"Successfully restored state. Resuming from step {start_step}.")
else:
    print("No checkpoint found. Starting new training run from step 0.")

# Calculate the target step for this session
steps_to_run_this_session = int(HOURS_TO_RUN * STEPS_PER_HOUR)
target_step = start_step + steps_to_run_this_session

batch_generator=get_stream_batches(dataset.shuffle(buffer_size=10000, seed=int(time.time())))
last_save_time = time.time()

print(f"\nStarting training from step {start_step} and running until step {target_step} (~{HOURS_TO_RUN} hours)...")
with tqdm(total=target_step, desc="Training Steps", initial=start_step) as progress_bar:
    for step in range(start_step, target_step):
        try:
            x, y = next(batch_generator)
            state, loss = train_step(state, x, y)
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{float(loss):.4f}")

            # Time-based safety checkpoint
            current_time = time.time()
            if (current_time - last_save_time) / 3600 >= SAVE_INTERVAL_HOURS:
                print(f"\n{SAVE_INTERVAL_HOURS} hours have passed. Saving safety checkpoint at step {step+1}...")
                save_path = f"{CHECKPOINT_PREFIX}{step+1}.pkl.gz"
                data_to_save = {'params': state.params, 'opt_state': state.opt_state, 'step': state.step}
                with gzip.open(save_path, 'wb') as f: pickle.dump(data_to_save, f)
                print(f"State saved to {save_path}")
                last_save_time = current_time
            
        except StopIteration:
            print("Dataset stream finished. Stopping training.")
            break

# Final save at the end of the session
print(f"\n{HOURS_TO_RUN}-hour session complete. Performing final save at step {state.step}...")
final_save_path = f"{CHECKPOINT_PREFIX}{int(state.step)}.pkl.gz"
data_to_save = {'params': state.params, 'opt_state': state.opt_state, 'step': state.step}
with gzip.open(final_save_path, 'wb') as f:
    pickle.dump(data_to_save, f)
print(f"Final state saved to {final_save_path}. Don't forget to download it!")

