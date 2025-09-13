import dataclasses
import enum
import os
from pathlib import Path
from pprint import pprint
from typing import List, Optional

import numpy as np
import tvm
from tvm import dlight, relax, te, tir
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax import op as R
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache, TIRPagedKVCache
from tvm.runtime import ShapeTuple

@dataclasses.dataclass
class VisionTransformerEncoderDecoderConfig:
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_attention_heads: int = 32
    num_hidden_layers: int = 22
    rms_norm_eps: float = 1e-05
    vocab_size: int = 32000
    rope_theta: int = 10000
    context_window_size: int = 2048
    prefill_chunk_size: int = 2048
    num_key_value_heads: int = 4
    head_dim: int = 64


dev = tvm.device("cuda", 0)
target = tvm.target.Target.from_device(dev)

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch, in_chans, embed_dim):
        super().__init__()
        self.h = img_size // patch
        self.w = img_size // patch
        self.N = self.h * self.w
        self.proj = nn.Linear(in_chans * patch * patch, embed_dim, bias=False)

        self.pos = nn.Embedding(self.N, embed_dim)

        self.patch = patch
        self.img_size = img_size

    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        p = self.patch
        x = op.reshape(x, (B, C, self.h, p, self.w, p))
        x = op.permute_dims(x, (0, 2, 4, 1, 3, 5))
        x = op.reshape(x, (B, self.N, C * p * p))
        x = self.proj(x)
        pe = self.pos.weight
        pe = op.reshape(pe, (1, self.N, pe.shape[-1]))
        return x + pe

class FFN(nn.Module):
    def __init__(self, config: VisionTransformerEncoderDecoderConfig):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, x):
        x = self.dense_1(x)
        x = op.silu(x)
        x = self.dense_2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config: VisionTransformerEncoderDecoderConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.query = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim)
        self.key = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim)
        self.value = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, q, k, v, attention_mask=None, key_padding_mask=None, causal_mask=None):
        batch_size, query_len, hidden_size = q.shape
        _, kv_len, _ = k.shape
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        q = op.reshape(q, (batch_size, query_len, self.num_attention_heads, self.head_dim))
        k = op.reshape(k, (batch_size, kv_len, self.num_attention_heads, self.head_dim))
        v = op.reshape(v, (batch_size, kv_len, self.num_attention_heads, self.head_dim))

        kt = op.permute_dims(k, (0, 2, 3, 1))
        qh = op.permute_dims(q, (0, 2, 1, 3))
        scale = tvm.tir.const(self.head_dim ** 0.5, "float16")
        scores = op.matmul(qh, kt) / scale

        if key_padding_mask is not None:
            keep = op.astype(key_padding_mask, "float16")
            scores = scores + (op.astype(1.0, "float16") - keep) * tvm.tir.const(-1e4, "float16")

        if attention_mask is not None:
            scores = scores + op.astype(attention_mask, "float16")

        if causal_mask and (query_len == kv_len):
            tri = op.triu(op.full((query_len, query_len), 1.0, dtype="float16"), 1)
            neg_inf = tvm.tir.const(-1e4, "float16")
            scores = scores + op.reshape(tri * neg_inf, (1, 1, query_len, query_len))

        attention = op.softmax(scores, axis=-1)

        context = op.matmul(attention, op.permute_dims(v, (0, 2, 1, 3)))
        context = op.permute_dims(context, (0, 2, 1, 3))
        context = op.reshape(context, (batch_size, query_len, hidden_size))
        out = self.out(context)

        return out

class EncoderBlock(nn.Module):
    def __init__(self, config: VisionTransformerEncoderDecoderConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.ffn = FFN(config)

    def forward(self, x: Tensor):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, config: VisionTransformerEncoderDecoderConfig,
                 img_size=224, patch=16, in_chans=3):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch, in_chans, config.hidden_size)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, img: Tensor):
        x = self.patch(img)
        for block in self.blocks:
            x = block(x)
        return self.ln(x)

class DecoderBlock(nn.Module):
    def __init__(self, config: VisionTransformerEncoderDecoderConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.self_attn = MultiHeadAttention(config)
        self.ln2 = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.cross_attn = MultiHeadAttention(config)
        self.ln3 = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.ffn = FFN(config)

    def forward(self, x: Tensor, mem: Tensor):
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), causal_mask=True)
        x = x + self.cross_attn(self.ln2(x), mem, mem)
        x = x + self.ffn(self.ln3(x))
        return x

class CaptionDecoder(nn.Module):
    def __init__(self, config: VisionTransformerEncoderDecoderConfig, max_len=128):
        super().__init__()
        self.tok = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos = nn.Embedding(max_len, config.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_len = max_len
        self.hidden_size = config.hidden_size

    def embed_tokens(self, ids: Tensor):
        B, T = ids.shape
        tok = self.tok(ids)

        pe = op.reshape(self.pos.weight, (1, self.max_len, self.hidden_size))
        pe = op.astype(pe, tok.dtype)

        return tok + pe

    def forward(self, ids: Tensor, mem: Tensor):
        x = self.embed_tokens(ids)
        for blk in self.blocks:
            x = blk(x, mem)
        x = self.ln(x)
        logits = self.head(x)
        return logits

class ViT2Text(nn.Module):
    def __init__(self, config: VisionTransformerEncoderDecoderConfig, img_size=224, patch=16, in_chans=3, max_len=128, dtype="float16"):
        super().__init__()
        self.enc = ViTEncoder(config, img_size, patch, in_chans)
        self.dec = CaptionDecoder(config, max_len=max_len)
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.dtype = dtype

    def to(self, dtype = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def encode(self, image):
        return self.enc(image)

    def prefill(self, token_ids, mem):
        logits = self.dec(token_ids, mem)
        return logits, mem

    def decode(self, token_ids, mem):
        logits = self.dec(token_ids, mem)
        return logits, mem

    def get_default_spec(self):
        N = (224//16) * (224//16)
        return nn.spec.ModuleSpec.from_raw(
            {
                "encode": {"image": nn.spec.Tensor([1, 3, 224, 224], self.dtype),
                           "$": {"param_mode": "packed", "effect_mode": "none"}},
                "prefill": {"token_ids": nn.spec.Tensor([1, self.dec.max_len], "int32"),
                            "mem": nn.spec.Tensor([1, N, self.hidden_size], self.dtype),
                            "$": {"param_mode": "packed", "effect_mode": "none"}},
                "decode": {"token_ids": nn.spec.Tensor([1, 1], "int32"),
                           "mem": nn.spec.Tensor([1, N, self.hidden_size], self.dtype),
                           "$": {"param_mode": "packed", "effect_mode": "none"}},
            },
            self,
        )

cfg = VisionTransformerEncoderDecoderConfig()
model = ViT2Text(cfg, dtype="float16")
model.to("float16")
mod, named_params = model.export_tvm(spec=model.get_default_spec())

@register_pipeline("baseline_3060")
def _baseline():
    @tvm.transform.module_pass(opt_level=0)
    def _p(mod, _):
        seq = tvm.transform.Sequential([
            relax.transform.LegalizeOps(),
            relax.transform.FoldConstant(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
            tir.transform.DefaultGPUSchedule(),
            relax.transform.RewriteDataflowReshape(),
            relax.transform.ToNonDataflow(),
            relax.transform.RemovePurityChecking(),
            relax.transform.CallTIRRewrite(),
            relax.transform.StaticPlanBlockMemory(),
            relax.transform.RewriteCUDAGraph(),
            relax.transform.LowerAllocTensor(),
            relax.transform.KillAfterLastUse(),
            relax.transform.LowerRuntimeBuiltin(),
            relax.transform.VMShapeLower(),
            relax.transform.AttachGlobalSymbol(),
        ])
        return seq(mod)
    return _p

@register_pipeline("opt_vit2txt")
def _pipeline():
    @tvm.transform.module_pass(opt_level=0)
    def _p(mod, _):
        seq = tvm.transform.Sequential([
            relax.transform.FuseTransposeMatmul(),
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FoldConstant(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
            relax.transform.DeadCodeElimination(),
            dlight.ApplyDefaultSchedule(
                dlight.gpu.Matmul(), dlight.gpu.GEMV(), dlight.gpu.Reduction(),
                dlight.gpu.GeneralReduction(), dlight.gpu.Fallback(),
            ),
            relax.transform.RewriteDataflowReshape(),
            relax.transform.ToNonDataflow(),
            relax.transform.RemovePurityChecking(),
            relax.transform.CallTIRRewrite(),
            relax.transform.StaticPlanBlockMemory(),
            relax.transform.RewriteCUDAGraph(),
            relax.transform.LowerAllocTensor(),
            relax.transform.KillAfterLastUse(),
            relax.transform.LowerRuntimeBuiltin(),
            relax.transform.VMShapeLower(),
            relax.transform.AttachGlobalSymbol(),
        ])
        return seq(mod)
    return _p

def build_vm(mod, target, pipeline_name, dev):
    with target:
        ex = tvm.compile(mod, target, relax_pipeline=relax.get_pipeline(pipeline_name))
    return relax.VirtualMachine(ex, dev)

def _rand_weight(shape, dtype_str):
    if "float" in dtype_str:
        scale = 0.02
        return (np.random.randn(*shape) * scale).astype(np.float16)
    return np.zeros(shape, dtype=np.int32)

vm_base = build_vm(mod, target, "baseline_3060", dev)
vm_opt  = build_vm(mod, target, "opt_vit2txt",   dev)

params = []
for name, spec in named_params:
    shape = tuple(int(s) for s in spec.shape)
    dtype = str(spec.dtype)
    arr = _rand_weight(shape, dtype)
    params.append(tvm.runtime.tensor(arr, device=dev))
img_np = (np.random.randn(1, 3, 224, 224) * 0.1).astype("float16")
img = tvm.runtime.tensor(img_np, device=dev)

_ = vm_base["encode"](img, params)
_ = vm_opt["encode"](img, params)

mem_base = vm_base["encode"](img, params)
mem_opt  = vm_opt["encode"](img, params)

tok = tvm.runtime.tensor(np.array([[1]], dtype=np.int32), device=dev)

def bench_encode(vm, label):
    te = vm.time_evaluator("encode", dev, number=10, repeat=5)
    res = te(img, params)
    mean_ms = res.mean * 1000.0
    std_ms  = res.std * 1000.0
    print(f"[{label}] encode: {mean_ms:.3f} ms ± {std_ms:.3f} ms")
    return mean_ms

def bench_decode(vm, label, mem_tensor):
    te = vm.time_evaluator("decode", dev, number=50, repeat=5)
    res = te(tok, mem_tensor, params)
    mean_ms = res.mean * 1000.0
    std_ms  = res.std * 1000.0
    tok_per_s = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    print(f"[{label}] decode (1 token): {mean_ms:.3f} ms ± {std_ms:.3f} ms  |  ~{tok_per_s:.1f} tok/s")
    return mean_ms, tok_per_s

print("\n=== Microbenchmarks (CUDA) ===")
enc_base_ms = bench_encode(vm_base, "baseline")
enc_opt_ms  = bench_encode(vm_opt,  "optimized")

dec_base_ms, dec_base_tps = bench_decode(vm_base, "baseline", mem_base)
dec_opt_ms,  dec_opt_tps  = bench_decode(vm_opt,  "optimized", mem_opt)

print("\n=== Summary ===")
def pct_better(old, new):
    return (old - new) / old * 100.0 if old > 0 else float("inf")

print(f"Encode speedup:   {enc_base_ms:.3f} ms  ->  {enc_opt_ms:.3f} ms  ({pct_better(enc_base_ms, enc_opt_ms):.1f}% faster)")
print(f"Decode speedup:   {dec_base_ms:.3f} ms  ->  {dec_opt_ms:.3f} ms  ({pct_better(dec_base_ms, dec_opt_ms):.1f}% faster)")
print(f"Tokens/sec:       {dec_base_tps:.1f}  ->  {dec_opt_tps:.1f}")

def time_full_decode(vm, label):
    import time
    mem_local = vm["encode"](img, params)
    last = 1
    steps = 16
    start = time.time()
    for _ in range(steps):
        token_nd = tvm.runtime.tensor(np.array([[last]], dtype=np.int32), device=dev)
        logits, mem_local = vm["decode"](token_nd, mem_local, params)
        last = int(np.argmax(logits.numpy()[0, 0]))
    dur_ms = (time.time() - start) * 1000.0
    print(f"[{label}] end-to-end decode loop ({steps} steps): {dur_ms:.2f} ms  (~{steps * 1000.0 / dur_ms:.1f} tok/s)")

print("\n=== End-to-end (optional) ===")
time_full_decode(vm_base, "baseline")
time_full_decode(vm_opt,  "optimized")
