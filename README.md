# TVM Relax: ViT Encoder to Transformer Decoder (multimodal architecture)

This is a benchmark that builds a Vision Transformer encoder and a transformer decoder in **TVM Relax**. It compiles two pipelines (baseline and optimized), and times **encode** and **1-token decode** on an RTX 3060 in float16.

It’s not a full model just enough to show end-to-end IRModule export, compilation, and perf deltas.
This is *not* a trained captioning model since weights are random. It is just to show end to end IRModule export and compilation process.

---

## Requirements

- NVIDIA GPU with CUDA (this was done on **RTX 3060**)
- CUDA 11+
- Python 3.10+
- TVM built **from source** with CUDA enabled

---

## Build TVM (quick notes)

```bash
git clone --recursive https://github.com/apache/tvm.git
cd tvm
mkdir build && cp cmake/config.cmake build/config.cmake

sed -i 's/USE_CUDA OFF/USE_CUDA ON/' build/config.cmake
sed -i 's/USE_CUBLAS OFF/USE_CUBLAS ON/' build/config.cmake

cd build
cmake ..
make -j$(nproc)

export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

---

## Python env

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy
```

---

## Run

From the repo root:

```bash
python3 tvm_project/multimodal.py
```

You should see two pipelines:

- `baseline_3060` – basic legalization/fusion/default GPU schedule
- `opt_vit2txt` – adds dlight scheduling for matmul/reductions, etc.

Sample output (what I got on RTX 3060):

```
=== Microbenchmarks (CUDA) ===
[baseline]  encode: ~666.5 ms
[optimized] encode: ~24.4  ms
[baseline]  decode (1 token): ~744.5 ms  | ~1.3 tok/s
[optimized] decode (1 token): ~22.6  ms  | ~44.2 tok/s

=== Summary ===
Encode speedup:  ~96%
Decode speedup:  ~97%
Tokens/sec:      ~1.3 -> ~44
```

---
