"""GPU diagnostic suite.

Replaces ad-hoc TFLOPS tests with a proper bottleneck-discriminating set:
  1. FP32 GEMM at multiple sizes (peak compute)
  2. FP16 GEMM at multiple sizes (does FP16 work? does it speed up?)
  3. Kernel launch overhead (tiny op timing)
  4. Conv2d at IMPALA-CNN dimensions (forward + backward, FP32 + FP16)
  5. Real ActorCritic forward + backward at training batch size
  6. Memory bandwidth (large tensor copy)

Each test runs warmup iterations, then timed iterations with cuda.synchronize.
Output is human-readable; we want to *see* what's slow.
"""
import sys
import time
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda")
sys.path.insert(0, "/home/steve/Thesis-RL-Sweep/src")


def bench(fn, warmup=10, iters=50, label=""):
    """Time fn() in ms with proper sync. Returns (mean_ms, min_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return sum(times) / len(times), times[0]


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def test_gemm(dtype, sizes=(1024, 2048, 4096, 8192)):
    section(f"GEMM {dtype}")
    print(f"{'size':>6}  {'mean ms':>10}  {'min ms':>10}  {'TFLOPS (mean)':>15}  {'TFLOPS (peak)':>15}")
    for n in sizes:
        try:
            a = torch.randn(n, n, device=DEVICE, dtype=dtype)
            b = torch.randn(n, n, device=DEVICE, dtype=dtype)
            mean_ms, min_ms = bench(lambda: a @ b, warmup=5, iters=20)
            flops = 2 * n ** 3
            tflops_mean = flops / (mean_ms / 1000) / 1e12
            tflops_peak = flops / (min_ms / 1000) / 1e12
            print(f"{n:>6}  {mean_ms:>10.2f}  {min_ms:>10.2f}  {tflops_mean:>15.2f}  {tflops_peak:>15.2f}")
            del a, b
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{n:>6}  FAILED: {e}")


def test_launch_overhead():
    section("Kernel launch overhead (small op timing)")
    print("Lower-bound = pure dispatch cost. Healthy native ROCm: ~10-30us.")
    a = torch.randn(8, 8, device=DEVICE)
    mean_ms, min_ms = bench(lambda: a + 1, warmup=20, iters=200)
    print(f"  add (8x8):           mean={mean_ms*1000:.1f}us  min={min_ms*1000:.1f}us")
    mean_ms, min_ms = bench(lambda: a * 2, warmup=20, iters=200)
    print(f"  mul (8x8):           mean={mean_ms*1000:.1f}us  min={min_ms*1000:.1f}us")
    # cudaEventRecord pair to isolate launch from sync overhead
    e_start = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)
    n_iters = 1000
    for _ in range(20):  # warmup
        _ = a + 1
    torch.cuda.synchronize()
    e_start.record()
    for _ in range(n_iters):
        _ = a + 1
    e_end.record()
    torch.cuda.synchronize()
    elapsed_ms = e_start.elapsed_time(e_end)
    print(f"  add (8x8) x{n_iters} (CUDA events):  total={elapsed_ms:.1f}ms  per-launch={elapsed_ms*1000/n_iters:.1f}us")


def test_conv2d(dtype):
    section(f"Conv2d (IMPALA-like blocks) {dtype}")
    # IMPALA-CNN first block: 3->16 channels at 64x64
    # second:                16->32 channels at 32x32
    # third:                 32->32 channels at 16x16
    cases = [
        ("3->16 64x64",  3, 16, 64),
        ("16->32 32x32", 16, 32, 32),
        ("32->32 16x16", 32, 32, 16),
        ("32->32 8x8",   32, 32, 8),
    ]
    batch = 64 * 256 // 8  # ppo minibatch with default config
    print(f"batch={batch}")
    print(f"{'shape':>20}  {'fwd ms':>10}  {'bwd ms':>10}  {'bwd/fwd':>10}")
    for label, c_in, c_out, hw in cases:
        try:
            x = torch.randn(batch, c_in, hw, hw, device=DEVICE, dtype=dtype, requires_grad=True)
            w = torch.randn(c_out, c_in, 3, 3, device=DEVICE, dtype=dtype, requires_grad=True)

            def fwd():
                y = F.conv2d(x, w, padding=1)
                return y

            mean_fwd, _ = bench(fwd, warmup=5, iters=20)

            def fwdbwd():
                y = F.conv2d(x, w, padding=1)
                y.sum().backward()
                x.grad = None
                w.grad = None

            mean_total, _ = bench(fwdbwd, warmup=5, iters=20)
            mean_bwd = mean_total - mean_fwd
            print(f"{label:>20}  {mean_fwd:>10.2f}  {mean_bwd:>10.2f}  {mean_bwd / max(mean_fwd, 1e-3):>10.2f}")
            del x, w
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{label:>20}  FAILED: {e}")


def test_actorcritic(dtype):
    section(f"ActorCritic full forward+backward {dtype}")
    from networks import ActorCritic

    model = ActorCritic(encoder="impala", num_actions=15).to(DEVICE)

    batch = 64 * 256 // 8  # 2048
    obs = torch.randint(0, 256, (batch, 64, 64, 3), dtype=torch.uint8, device=DEVICE)
    acts = torch.randint(0, 15, (batch,), device=DEVICE)

    use_amp = (dtype == torch.float16)

    def fwd():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, logp, ent, val = model.get_action_and_value(obs, acts)
                return (logp + ent + val).sum()
        else:
            _, logp, ent, val = model.get_action_and_value(obs, acts)
            return (logp + ent + val).sum()

    mean_fwd, min_fwd = bench(fwd, warmup=5, iters=20)

    def fwdbwd():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, logp, ent, val = model.get_action_and_value(obs, acts)
                loss = (logp + ent + val).sum()
        else:
            _, logp, ent, val = model.get_action_and_value(obs, acts)
            loss = (logp + ent + val).sum()
        loss.backward()
        for p in model.parameters():
            p.grad = None

    mean_tot, min_tot = bench(fwdbwd, warmup=5, iters=20)
    mean_bwd = mean_tot - mean_fwd
    print(f"  batch={batch}")
    print(f"  forward:        mean={mean_fwd:.2f}ms  min={min_fwd:.2f}ms")
    print(f"  fwd+bwd:        mean={mean_tot:.2f}ms  min={min_tot:.2f}ms")
    print(f"  backward only:  ~{mean_bwd:.2f}ms  (bwd/fwd ratio = {mean_bwd / max(mean_fwd, 1e-3):.2f})")
    print()
    # Project to PPO update: 24 minibatches per update
    ppo_grad_estimate = mean_tot * 24
    print(f"  Projected ppo_grad (mean_tot * 24 minibatches) = {ppo_grad_estimate:.0f}ms")


def test_memcopy():
    section("Memory bandwidth (D2D copy)")
    for n in (16, 64, 256, 1024):
        bytes_ = n * 1024 * 1024
        a = torch.empty(n * 1024 * 1024 // 4, device=DEVICE, dtype=torch.float32)
        b = torch.empty_like(a)
        mean_ms, min_ms = bench(lambda: b.copy_(a), warmup=5, iters=20)
        gbs = bytes_ / (mean_ms / 1000) / 1e9
        print(f"  {n:>4} MB:  mean={mean_ms:.2f}ms  GB/s={gbs:.0f}")
        del a, b
        torch.cuda.empty_cache()


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    p = torch.cuda.get_device_properties(0)
    print(f"Device: {p.name}  ({p.gcnArchName})  {p.total_memory // (1024**2)}MB")
    print()

    test_gemm(torch.float32)
    try:
        test_gemm(torch.float16)
    except Exception as e:
        print(f"FP16 GEMM FAILED: {e}")
    test_launch_overhead()
    test_conv2d(torch.float32)
    try:
        test_conv2d(torch.float16)
    except Exception as e:
        print(f"FP16 Conv2d FAILED: {e}")
    test_actorcritic(torch.float32)
    try:
        test_actorcritic(torch.float16)
    except Exception as e:
        print(f"FP16 ActorCritic FAILED: {e}")
    test_memcopy()


if __name__ == "__main__":
    main()
