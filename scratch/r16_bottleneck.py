#!/usr/bin/env python3
"""
R16: is prefill read-bound or compute-bound?

R14 asserts prefill is read-bound and blames it on the drive's "random vs
ordered" scheduling.  But 165 tok/s of IQ3_XXS prefill moves 6.5 MiB/token =
1075 MB/s, while scratch/r15_qd measured this drive doing **6102 MB/s** on
ordered O_DIRECT reads.  So either the engine is leaving 5 GB/s on the table
for a reason, or the read rate was never the binding constraint.

This decides it the direct way: run a long prefill and sample GPU utilization,
DRAM/NVMe throughput, and the engine's own tier counters at 200 ms.  A card
pinned at 99% with the drive at 200 MB/s is compute-bound, and every
"drive scheduling" idea is then dead on arrival - the lever is the kernel.

Usage: python3 scratch/r16_bottleneck.py [port] [prompt_tokens]
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
stop = False
samples = []


def sampler():
    while not stop:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,"
                 "memory.used,clocks.sm,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            f = [x.strip() for x in out.split(",")]
            d = {"gpu": float(f[0]), "memu": float(f[1]), "vram": float(f[2]),
                 "clk": float(f[3]), "w": float(f[4])}
            with open("/proc/diskstats") as fh:
                for ln in fh:
                    p = ln.split()
                    if p[2] == "nvme0n1":
                        d["rsekt"] = int(p[5])
                        break
            samples.append(d)
        except Exception:
            pass
        time.sleep(0.2)


def post(payload, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


para = ("The lighthouse keeper surveyed the horizon: lens mechanics, tide tables, "
        "and the logbook of storms. He adjusted the brass gears, checked the mercury "
        "bath, and recorded the barometric pressure at dusk. ")
prompt = para * (NTOK // 46 + 1)

th = threading.Thread(target=sampler, daemon=True)
th.start()
t0 = time.time()
r = post({"model": "x",
          "messages": [{"role": "user",
                        "content": prompt + "\n\nSummarize the keeper's routine in two sentences."}],
          "max_tokens": 4, "temperature": 0})
t_done = time.time() - t0          # includes 4 decode tokens; prefill dominates
stop = True
th.join(timeout=3)
dt = time.time() - t0

u = r.get("usage", {})
pt = u.get("prompt_tokens", 0)
if not samples:
    print("no samples")
    sys.exit(1)
a, b = samples[0], samples[-1]
elapsed = (len(samples) - 1) * 0.2
mbs = (b["rsekt"] - a["rsekt"]) * 512 / elapsed / 1e6
gpus = sorted(s["gpu"] for s in samples)
n = len(gpus)
print(f"prompt={pt} tok  elapsed={dt:.1f}s  prefill~={pt / max(dt - 0.3, 0.1):.0f} tok/s")
print(f"NVMe read rate during run : {mbs:.0f} MB/s")
print(f"GPU util  min/med/p90/max : {gpus[0]:.0f} / {gpus[n // 2]:.0f} / {gpus[int(n * .9)]:.0f} / {gpus[-1]:.0f} %")
print(f"VRAM used                 : {b['vram'] / 1024:.2f} GiB   SM {b['clk']:.0f} MHz  {b['w']:.0f} W")
mbps = mbs * 1e6 / (pt / max(dt - 0.3, 0.1)) / (1024 * 1024)
print(f"\nMB read per prefill token : {mbps:.2f} MiB")
print("DRIVE CEILING (r15_qd, ordered 2 MiB O_DIRECT) = 6102 MB/s")
if gpus[n // 2] >= 90:
    print("VERDICT: GPU is SATURATED -> prefill is COMPUTE-bound. Drive scheduling is a")
    print("         dead end; the lever is the expert kernel / batch efficiency.")
elif mbs < 1000:
    print("VERDICT: GPU idle AND drive idle -> prefill is STALL/DEPTH-bound (pipeline")
    print("         depth, not bandwidth or flops). Raise in-flight reads + chunk size.")
else:
    print("VERDICT: drive is the constraint at these depths.")
