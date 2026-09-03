# GPU run. Provisioning and cost control

The GPU-only conditions of two experiments, on one instance, for a few dollars.

## What this run produces

| experiment | conditions | what it settles |
| --- | --- | --- |
| [regime-discrimination](../experiments/regime-discrimination/) | `fp16`, `gpu_tf32_on`, `gpu_tf32_off`, `gpu_bf16` | The retrieval half of the **TF32 audit gap**. `gpu-determinism` reports that a model audited at fp32/CPU and served at fp32/GPU disagrees on 12–46% of inputs, and asserts twice that this is "invisible to cosine/retrieval" — without measuring a retrieval metric. These conditions measure it. |
| [decoding-reproducibility](../experiments/decoding-reproducibility/) | `fp16`, `tf32_on`, `tf32_off` on Qwen2.5-3B | Moves ARI-D off TinyLlama onto a model people serve, and exercises the **chunked SEMQ path** (151,936-token vocabulary vs SEMQ's 65,536 `max_dim` cap). |

## Cost

The compute is cheap. The way this gets expensive is an instance nobody
terminated, so the guards target that rather than the run.

| item | estimate |
| --- | --- |
| g5.xlarge (A10G, 24 GB), on-demand, us-east-1 | ~$1.01/hr |
| expected work incl. setup and debugging | 3–5 hrs |
| **expected total** | **$3–5** |
| 200 GB gp3 volume, for one day | ~$0.55 |

Three independent guards, in order of how much they can be trusted:

1. **`shutdown -h +300` armed in user-data**, before any step that can hang.
2. **`instance-initiated-shutdown-behavior=terminate`**, so that shutdown
   destroys the instance rather than leaving a stopped one with an EBS volume.
3. **A budget alarm** from `budget.sh`. Useful, but AWS cost data lags by
   hours, so treat it as a backstop and not as protection.

Verify current pricing before launching. The figures above are approximate.

## Instance sizing

Sizing is driven by the **fp32 reference**, not by inference. Every condition
is compared against a full-precision baseline, so that baseline has to fit.

| model | fp32 weights | fits 24 GB (g5.xlarge)? |
| --- | ---: | :--- |
| Qwen2.5-3B-Instruct | ~12 GB | yes — **the default** |
| Llama-3.2-3B | ~12 GB | yes |
| Qwen2.5-7B-Instruct | ~28 GB | **no** — use `g6e.xlarge` (L40S, 48 GB, ~$1.86/hr) |

For a 7B run:

```bash
INSTANCE_TYPE=g6e.xlarge ./launch.sh
# then, on the instance:
ARI_D_MODEL=Qwen/Qwen2.5-7B-Instruct bash ~/semq-ari-benchmark/infra/run_gpu.sh
```

## Procedure

```bash
# 0. Once per account, before the first launch.
./budget.sh you@example.com 100

# 1. Check what would happen without touching anything.
./launch.sh --dry-run

# 2. Provision. Prints the SSH command and the instance id.
./launch.sh

# 3. Wait for setup, which runs in the background on boot.
ssh -i ~/.ssh/semq-ari-gpu.pem ubuntu@<ip> 'tail -f /var/log/semq-bootstrap.log'

# 4. Run both experiments.
ssh -i ~/.ssh/semq-ari-gpu.pem ubuntu@<ip> 'bash ~/semq-ari-benchmark/infra/run_gpu.sh'

# 5. Pull results back. Skips the multi-GB caches.
./fetch_results.sh <ip>

# 6. Terminate. Do this yourself. Do not wait for the timer.
aws ec2 terminate-instances --instance-ids $(cat .last-instance) --region us-east-2
```

## Reading the output

**A GPU run that reproduces the CPU numbers exactly means the GPU conditions
did not execute.** `run_gpu.sh` prints the detected device and free VRAM before
starting, and exits if CUDA is missing, but check the device line in the output
rather than trusting that the run "completed".

Two conditions are expected to be skipped and are not failures:

- `int8` in ARI-D. Torch dynamic quantisation is CPU-only. The CPU result
  already exists.
- Any condition marked `needs_gpu` when CUDA is absent.

The interesting comparison in regime-discrimination is `gpu_tf32_on` against
`gpu_tf32_off`. If Recall@10 is unchanged between them while SEMQ HER moves,
the audit-gap argument is closed with both halves measured. If Recall@10 *does*
move, that is also a result. It means retrieval detects TF32 and the claim
needs narrowing again, the way the CPU run already narrowed it.

## If the repository is private

`bootstrap.sh` clones over HTTPS and will fail on a private repository. Copy
the tree up instead:

```bash
rsync -avz --exclude='.git' --exclude='**/results/cache' \
  -e "ssh -i ~/.ssh/semq-ari-gpu.pem" \
  ../ ubuntu@<ip>:semq-ari-benchmark/
```
