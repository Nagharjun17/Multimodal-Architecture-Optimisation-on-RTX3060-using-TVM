import numpy as np
import tvm
from tvm import te

n = 1024
A = te.placeholder((n,), "float32", name="A")
B = te.placeholder((n,), "float32", name="B")
C = te.compute((n,), lambda i: A[i] + B[i], name="C")

prim = te.create_prim_func([A, B, C])
f_cpu = tvm.build(prim, target="llvm")

sch = tvm.tir.Schedule(prim)
blk = sch.get_block("C")
(i,) = sch.get_loops(blk)
i0, i1 = sch.split(i, factors=[None, 256])
sch.bind(i0, "blockIdx.x")
sch.bind(i1, "threadIdx.x")
f_cuda = tvm.build(sch.mod, target="cuda")

def to_tensor(np_arr, device):
    return tvm.runtime.tensor(np_arr, device=device)

def empty(shape, dtype, device):
    return tvm.runtime.tensor(np.empty(shape, dtype=dtype), device=device)

dev_cpu = tvm.cpu()
a_np = np.random.rand(n).astype("float32")
b_np = np.random.rand(n).astype("float32")
a = to_tensor(a_np, dev_cpu)
b = to_tensor(b_np, dev_cpu)
c = empty((n,), "float32", dev_cpu)
cpu_out = c.numpy()
f_cpu(a, b, c)
np.testing.assert_allclose(c.numpy(), a_np + b_np, rtol=1e-5)

dev_gpu = tvm.cuda(0)
ag = to_tensor(a_np, dev_gpu)
bg = to_tensor(b_np, dev_gpu)
cg = empty((n,), "float32", dev_gpu)
gpu_out = cg.numpy()
f_cuda(ag, bg, cg)
np.testing.assert_allclose(cg.numpy(), a_np + b_np, rtol=1e-5)

print("Hello TVM ✔ (CPU + CUDA)")
print("A[:8]      =", a_np[:8])
print("B[:8]      =", b_np[:8])
print("CPU C[:8]  =", cpu_out[:8])
print("GPU C[:8]  =", gpu_out[:8])
print("max|CPU-GPU| =", float(np.max(np.abs(cpu_out - gpu_out))))
