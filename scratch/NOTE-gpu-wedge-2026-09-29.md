# GPU wedge 2026-09-29 11:12 — findings + recovery (handoff note)

**Symptom:** the engine (or any CUDA program) dies at startup with
`native embedding: cannot pin 644 MiB (cuda: CUDA-capable device(s) is/are busy or unavailable)`.
The desktop works normally.

**Root cause (kernel log, `sudo dmesg`):**
1. `11:12:20` — global OOM: `Killed process 1801618 (strata) total-vm:98.9 GB, anon-rss:25.2 GB`
   (the running engine; the box has 30 GiB RAM).
2. `11:17:07+` — `NVRM: GPU0 rpcRmApiAlloc_GSP: GspRmAlloc failed ... status=0x63`
   (`NV_ERR_STATE_IN_USE`) for every new client. The OOM-killed CUDA context left a stuck
   channel in the kernel driver; nothing can create a new CUDA context since.
   Display (kwin/chrome) is unaffected, which is why the box "isn't wedged" from the desktop.

**Proof it is driver-side, not engine-side:** `scratch/cuda_probe` (20-line CUDA program,
`nvcc`-built from `scratchpad`) fails `cudaSetDevice` for every process.

**Recovery:** `sudo bash scratch/gpu-recover.sh` — stops display-manager, tears down any
surviving session, `modprobe -r`/`modprobe` nvidia_drm/nvidia_modeset/nvidia_uvm/nvidia,
re-enables persistence mode, verifies with `scratch/cuda_probe`, restarts sddm.
Log: `logs/gpu-recover.log`.

**Then launch:** `cd ~/models/strata && bash srv.sh strata-r19-best.json 8123`
(r19-best = the current best config per REPORT.md §0.9; the Pi harness points at :8123.)

**Prevention:** the box OOM'd at the engine's normal 25.2 GB footprint with a heavier
desktop — same marginal-OOM mode as R14 §0.7.2. Levers: keep Chrome lighter, or drop
`--hot-ram-gib` 24.0 -> ~23.0 in the config (each GiB ~0.8% coverage).
