# Replay instructions — L3_09_agentic_compounding / 2026-07-07_113131_20260707-112517

Captured at (UTC): 2026-07-07T11:32:01+00:00
Execution mode: **aws-ec2**

## Canonical replay

- Container image: `127348475353.dkr.ecr.us-east-2.amazonaws.com/semq-research-cpu:20260707-112517`
- Image digest: `sha256:f67a3d281a01232db2a30aaa3d4353d17f4defe0fca0a3f6c08ebd357a312930`
- Git SHA: `<unknown>`
- AWS instance type: `c7i.2xlarge`
- AWS AMI: `ami-0a9d57e25bb9c8b0e`
- AWS region: `us-east-2`

### Steps

1. Pull the same container image:

    ```bash
    docker pull 127348475353.dkr.ecr.us-east-2.amazonaws.com/semq-research-cpu:20260707-112517@sha256:f67a3d281a01232db2a30aaa3d4353d17f4defe0fca0a3f6c08ebd357a312930
    ```

2. Run on the same host class:

    ```bash
    docker run --rm \
      --platform linux/amd64 \
      -v $(pwd)/results:/workspace/results \
      127348475353.dkr.ecr.us-east-2.amazonaws.com/semq-research-cpu:20260707-112517@sha256:f67a3d281a01232db2a30aaa3d4353d17f4defe0fca0a3f6c08ebd357a312930 \
      run L3_09_agentic_compounding
    ```

3. Compare your `analysis.json` against the archived one in this bundle.
    All recall / delta values should match within float rounding.

## Bundle integrity

All file hashes are recorded in `MANIFEST.sha256`. To verify:

```bash
cd <bundle_dir> && sha256sum -c MANIFEST.sha256
```

## Full environment

See `environment.json` for the complete machine-readable snapshot 
(OS, packages, hardware, GPU, AWS metadata, git revision).
