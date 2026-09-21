# Matched-budget detector study

Compares the ARI code against simpler detectors at equal storage.
[PROTOCOL.md](PROTOCOL.md) is the frozen protocol (v2). [PROTOCOL-v1.md](PROTOCOL-v1.md)
scored the pilot. [INVENTORY.md](INVENTORY.md) records which artifacts exist and what
the confirmatory run still needs. No confirmatory episode exists; the committed pilot
is development data.

## Run the confirmatory collection

Targets, instance and model choices are in INVENTORY.md sections 3.1 and 3.2.
Complete the items marked **gap** in INVENTORY.md section 3.6 before the first episode.

### Collection commands

Run from a laptop with AWS credentials. Angle brackets mark operator values.

```bash
cd infra
cp operator.env.example operator.env         # fill CA_* and bucket names
MAX_MINUTES=1440 ./launch.sh                 # default 300 min is too short, see the estimate below
./push_tree.sh <ip>                          # private repo: rsync the tree
ssh -i ~/.ssh/semq-ari-gpu-us-east-1.pem ubuntu@<ip> 'tail -f /var/log/semq-bootstrap.log'
```

On the instance, after "bootstrap complete":

```bash
cd ~/semq-ari-benchmark/experiments/matched-budget-detectors
RUN=mbd/$(date -u +%Y%m%dT%H%M%SZ)
for ep in $(seq 0 299); do
  for cond in control tf32_on batch_1 batch_256 fp16 bf16 threads_1; do
    ~/venv/bin/python collect_embeddings.py --condition "$cond" --episode "$ep" \
      --inputs ../../data/ari-bench-v0.1.jsonl --n 200 \
      --out ~/mbd_collect --publish-s3 "s3://<episodes-bucket>/$RUN" \
      --publish-region us-east-2 || exit 1
  done
done
```

The inner loop cycles every condition at each episode index, which is the
interleaving `collect_embeddings.py:7-9` requires. `|| exit 1` stops the run
at the first failed publication. Rerunning the same loop resumes: complete
episodes are validated and re-uploaded, not re-encoded
(`collect_embeddings.py:54-78`, `135-145`).

### Durability

`collect_embeddings.py` publishes each episode as it is written. The rationale
is the comment at `collect_embeddings.py:203-207`: a full collection was lost
on 2026-09-09 because the driver wrote only to local disk.

- Each episode is uploaded as it is written, both `.npz` and `.json`
  (`collect_embeddings.py:208-211`).
- `publish_files` runs `aws s3 cp` per file with a 600 s timeout and raises
  `RuntimeError` if any upload fails (`collect_embeddings.py:81-97`). Tests:
  `tests/test_collect_embeddings.py:50-74`.
- The reuse path also uploads, so a retry after a failed upload still
  publishes (`collect_embeddings.py:135-145`; test at `117-138`).
- Existing output is reused only after manifest and checksum validation;
  partial or corrupt output raises (`collect_embeddings.py:54-78`; tests at
  `77-115`).
- The destination prefix is `<publish-s3>/<condition>/epNNN.*`. No such
  prefix exists in either bucket today (INVENTORY.md section 6).

### Wall-clock and cost estimate

Assumptions, each stated so it can be replaced:

1. One episode is one fresh Python process. Torch import, CUDA init and
   loading all-MiniLM-L6-v2 from the local HF cache take 8-15 s.
2. Encoding 200 texts of at most 512 characters on an A10G takes under 1 s.
3. Two `aws s3 cp` calls of about 300 KB and 1 KB take 1-3 s.
4. Per-episode total: 15-30 s. Midpoint 20 s.
5. `g5.xlarge` on-demand is about $1.01/h (`infra/README.md:19`). A 200 GB gp3
   volume is about $0.55/day (`infra/README.md:22`); the default 1,000 MB/s
   throughput and 16,000 IOPS (`infra/launch.sh:70-71`) add a few dollars per
   day (`infra/launch.sh:67-69`).

Arithmetic:

- Episodes: 2,100.
- Wall clock: 2,100 x 20 s = 42,000 s = 11.7 h. Range 2,100 x 15 s = 8.8 h
  to 2,100 x 30 s = 17.5 h. Add about 15 min for bootstrap.
- Compute: 11.7 h x $1.01/h = $11.8. Range $8.9 to $17.7.
- Volume: one day, about $1 to $4 with the default throughput settings.
- S3: 2,100 x (307,200 B + about 1 KB) = about 650 MB. Storage cost is under
  one cent per month.
- **Expected total: about $15. Range $10 to $22.**

`launch.sh` arms `shutdown -h +MAX_MINUTES` with `MAX_MINUTES` default 300
(`infra/launch.sh:48`, `infra/bootstrap.sh:14`) and
`instance-initiated-shutdown-behavior terminate` (`infra/launch.sh:241`). Five
hours is shorter than the run. Set `MAX_MINUTES=1440`. If the timer fires
early, every published episode survives and the loop resumes on a new
instance.
