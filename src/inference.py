# === WIGIP-1 INFERENCE SCRIPT (CPU-ONLY VERSION) ===
# NOTE: This will be VERY slow. It is for demonstration purposes.

# First, install the necessary libraries if you don't have them
# You might need to run this from your terminal:
# pip install -q jax flax optax datasets gradio

import jax, jax.numpy as jnp, flax.linen as nn, pickle, os, time, glob
from jax import random
from functools import partial
from datasets import load_dataset
import gradio as gr

# --- CONFIGURATION ---
N_LAYER = 24; N_EMBD = 1280; N_HEAD = 16; BLOCK_SIZE = 256
PARAM_DTYPE = jnp.float32; COMPUTE_DTYPE = jnp.bfloat16

# --- YOUR LOCAL FILE PATH ---
# You need to download your .pkl file and place it in the same folder as this script.
load_path = '/kaggle/input/training-model/training_state_step_533082.pkl' 

# --- DATA & VOCAB ---
print("--- Building vocabulary... ---");
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
processed_text_sample = ""
for i, example in enumerate(dataset.take(1000)): processed_text_sample += example['text']
chars = sorted(list(set(processed_text_sample))); vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}; itos = {i: ch for ch, i in stoi.items()} 
encode = lambda s: [stoi.get(c, 0) for c in s];
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
        
# --- LOAD PARAMETERS ---
model=Transformer(N_LAYER,N_HEAD,N_EMBD,BLOCK_SIZE,vocab_size)
print(f"\n--- Loading checkpoint from {load_path}... ---")
with open(load_path, 'rb') as f:
    saved_data = pickle.load(f)
params = saved_data['params']
# We do NOT use jax.device_put(), so the model stays on the CPU
print("Load complete.")

# --- GENERATION LOGIC ---
# We still JIT compile, but JAX will target the CPU
@partial(jax.jit, static_argnames='apply_fn')
def model_step(apply_fn, params, current_tokens, key):
    current_tokens_cond = current_tokens[:, -BLOCK_SIZE:]; logits = apply_fn({'params': params}, current_tokens_cond)
    last_token_logits = logits[:, -1, :]; return random.categorical(key, last_token_logits)

# --- CHATBOT LOGIC ---
key = random.PRNGKey(42)
is_first_run = True

def chat_with_wigip(user_prompt, max_new_tokens=200):
    global key, is_first_run
    prompt_tokens = jnp.array([encode(user_prompt)])
    current_tokens = prompt_tokens
    response_string = ""

    # Compile on the first run
    if is_first_run:
        print("\n--- Performing one-time model compilation for CPU (warm-up)... ---")
        start_time = time.time()
        _ = model_step(model.apply, params, warmup_tokens, key)
        _.block_until_ready()
        end_time = time.time()
        print(f"CPU compilation complete. (Took {end_time - start_time:.2f} seconds)")
        is_first_run = False

    for _ in range(max_new_tokens):
        key, subkey = random.split(key)
        next_token_array = model_step(model.apply, params, current_tokens, subkey)
        next_token_id = next_token_array.item()
        next_char = itos.get(next_token_id, '')
        if next_char == '\n': break
        response_string += next_char
        current_tokens = jnp.concatenate([current_tokens, next_token_array[None, ...]], axis=1)
        yield response_string

# --- LAUNCH UI ---
print("\n--- Launching chatbot interface... ---")
# Dummy values for the first run compilation
warmup_tokens = jnp.array([[0]])

iface = gr.Interface(
    fn=chat_with_wigip,
    inputs=gr.Textbox(lines=2, label="Your Prompt", placeholder="Type your message to Wigip-1 here..."),
    outputs=gr.Textbox(label="Wigip-1's Response"),
    title="Wigip-1: A 473M Parameter Model (Running on CPU)",
    description="This is a live interface for the Wigip-1 language model. By Prathamesh Dere.",
    allow_flagging="never"
)

iface.launch(share=True, debug=True)
