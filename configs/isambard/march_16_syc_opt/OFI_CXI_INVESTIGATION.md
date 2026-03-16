# OFI/CXI NCCL Transport Failure on Isambard GH200

**Date:** 2026-03-16
**Investigated by:** Puria (with Claude Code)
**Jobs:** 2895379 (2-node, worked), 2896705 (8-node, failed)

## Background

Isambard AI uses HPE Slingshot interconnect with the CXI (Cassini eXtended Interface) provider. For NCCL to use native Slingshot transport (instead of falling back to TCP sockets), you need the **aws-ofi-nccl** plugin, which bridges NCCL's network plugin API to libfabric's OFI interface, enabling GPU Direct RDMA (GDR) over Slingshot.

## What's installed

- **aws-ofi-nccl 1.8.1** (`brics/aws-ofi-nccl/1.8.1`) — built against **NCCL 2.26.6**
- Our venv uses **NCCL 2.27.5** (required by PyTorch 2.9.1, loaded via `LD_PRELOAD`)
- The NCCL net plugin API is ABI-stable across these versions — the plugin loads fine and NCCL successfully selects it as `NET/AWS Libfabric`

## What works (2-node / 7B)

At 2-node scale (5 NCCL ranks: 1 learner rank-0 + 4 vLLM engines), OFI works perfectly:
- All inter-node channels use `NET/AWS Libfabric/0/GDRDMA`
- Zero RC:107 errors
- GPU Direct RDMA enabled for 31 GPU/HCA pairs, disabled for 7 (topology distance > 8)
- Training runs normally

However, **no performance benefit over sockets** at this scale — the 7B weight sync is only ~0.3s either way.

## What fails (8-node / 32B)

At 8-node scale (25 NCCL ranks: 8 learners + 12 vLLM engines across TP=2), the OFI transport **fails to complete NCCL connection setup**:

**The error:**
```
ofi_process_cq:196 NCCL WARN NET/OFI Request 0x4010bfae3b98 completed with error.
RC: 107. Error: Inappropriate ioctl for device.
Completed length: 0, Request: { dev: 2, size: 0, state: CREATED, direction: SEND }
```
This is `ENOTTY` (errno 107) from the CXI provider during completion queue processing — a zero-length SEND request completes with an error.

**Statistics from the 8-node run:**
- 8 RC:107 errors (with Ray log dedup, actually ~41x across cluster)
- 477 IPC register failures (`failed to IPC register userbuff`)
- GDRDMA enabled for 114 GPU/HCA pairs, disabled for 16 (distance > 8)
- Mixed transport: 10 channels via `NET/AWS Libfabric/GDRDMA`, 15 via `NET/AWS Libfabric` (no RDMA)
- Connection setup never completed — spent 25 minutes in P2P connect phase, then timed out
- `RAS idle timeout (60s)` errors appeared as connections dropped mid-setup

**The job never completed a single weight sync or training step.** After 25 min stuck in init, the training loop timed out and initiated graceful shutdown. Hit SLURM wall time at 30 min.

## Why it scales poorly

The failure is in the CXI provider's completion queue (`ofi_process_cq`), not in NCCL itself. At 2 nodes / 5 ranks, there are ~20 inter-node channels to set up. At 8 nodes / 25 ranks, there are hundreds of channels across 7 inter-node links, and the CXI provider hits a race condition or resource limit during the connection storm.

There's a known workaround for a *similar* CXI race during NCCL init: setting `NCCL_DEBUG=INFO` to stdout adds enough I/O overhead to slow down the init sequence and avoid the race. **This workaround is active but insufficient for the OFI transport** — the RC:107 errors originate inside the OFI plugin's CXI codepath, not in NCCL's built-in socket transport init.

## GDR topology issue

Even when OFI connections succeed, GPU Direct RDMA is **selectively disabled** based on GPU-to-NIC topology distance:
```
GPU Direct RDMA Enabled for GPU 8 / HCA 3 (distance 3 <= 8)
GPU Direct RDMA Disabled for GPU 1 / HCA 3 (distance 9 > 8)
```
On GH200 (Grace-Hopper), GPUs connect to NICs via C2C links. Some GPU/HCA pairs exceed the distance threshold (default 8), falling back to non-RDMA OFI transport. This means even with working OFI, not all channels would use GDRDMA.

## What would fix it

1. **Rebuild aws-ofi-nccl against NCCL 2.27.5** — the current 1.8.1 build was compiled against 2.26.6. While the net plugin ABI is stable, there may be subtle behavioral differences in how NCCL 2.27.5 drives the plugin (new features like `ncclCollNetPlugin_v6`, `ncclTunerPlugin_v3` that the plugin doesn't implement, causing fallback paths)

2. **Upgrade to aws-ofi-nccl 1.12+** — newer versions have significant CXI bug fixes, including better handling of completion queue errors, retry logic, and GDR topology awareness. The 1.8.1 version is from September 2025.

3. **CXI firmware/driver update on Isambard** — the RC:107 ENOTTY may be a known CXI provider bug that's fixed in newer libfabric/CXI versions

4. **Tune `NCCL_NET_GDR_LEVEL`** — currently set to `PHB`, could try `SYS` or `LOC` to change which GPU/HCA pairs attempt RDMA, potentially reducing the connection count that triggers the race

5. **Stagger NCCL init across ranks** — the connection storm of 25 ranks all calling `init_process_group` simultaneously may overwhelm the CXI provider. A custom init sequence with delays between groups of ranks could help.

## Impact

Without OFI, all inter-node NCCL traffic uses TCP sockets over the `hsn` Slingshot interface. For 32B training, weight sync (broadcasting ~32B parameters from learner rank-0 to 12 vLLM engines) takes **65–240 seconds per step** over sockets. This is the dominant bottleneck — actual training forward/backward is 480-620s on step 1 but will be much less on steady-state steps. Native CXI transport with GDRDMA could significantly reduce this.

## Experiment configs

All configs in `configs/isambard/march_16_syc_opt/`:
- `grpo_rlzero_ofi.sbatch` — sbatch with aws-ofi-nccl loaded
- `ray_node_setup_ofi.sh` — worker node setup with OFI env vars
- `grpo_benchmark_32b_ofi_flash.yaml` — 32B config used for the failed 8-node run
- `grpo_benchmark_7b.yaml` — 7B config used for the successful 2-node run
