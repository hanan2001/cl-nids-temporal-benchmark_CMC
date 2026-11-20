import sys, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
from cl_methods import MLP
nt = int(sys.argv[1]); torch.set_num_threads(nt)
m = MLP(78, [512, 256, 128], 15, 0.3); opt = torch.optim.Adam(m.parameters(), 1e-4)
X = torch.randn(512, 78); y = torch.randint(0, 15, (512,)); Xr = torch.randn(256, 78); yr = torch.randint(0, 15, (256,))
def step():
    loss = F.cross_entropy(m(X), y) + 0.5 * F.cross_entropy(m(Xr), yr)
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
for _ in range(20): step()
t = time.perf_counter(); n = 300
for _ in range(n): step()
dt = (time.perf_counter() - t) / n
Xv = torch.randn(100000, 78)
t = time.perf_counter()
with torch.no_grad():
    for i in range(0, len(Xv), 8192): m(Xv[i:i+8192])
print(f"threads={nt}: {1000*dt:.2f} ms/step, eval 100k={time.perf_counter()-t:.2f}s")
