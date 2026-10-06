#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Self-hosted CPU panel, day 0: same/proc reports, the bf16 prec diagnostic, the
# `time` baselines, and the CPU fp32 reference captures for the GPU `mach` compare.
# Run cpu_panel_time.sh on this same machine >=24h later.
export LABEL=${LABEL:-cpu}
source "$(dirname "$0")/common.sh"
for m in "${MODELS8[@]}"; do
  slug=${m//\//_}
  step "report $m" bash -c "$VP/python ari/tools/run_report.py --agent bge --model '$m' --inputs '$INPUTS' && cp 'leaderboard/submissions/$slug.json' '$OUT/report_$slug.json'"
  step "prec $m" $VP/python ari/tools/selfhosted_prec.py --model "$m" --inputs "$INPUTS" --out "$OUT/prec_$slug.json"
  step "time-baseline $m" $VP/python ari/tools/selfhosted_conc_time.py --mode baseline --model "$m" --inputs "$INPUTS"
  step "mach-ref $m" $VP/python ari/tools/gpu_capture.py capture --model "$m" --device cpu --dtype fp32 --tf32 off \
    --inputs "$INPUTS" --n 512 --outdir "$OUT/mach_ref/${slug}__cpu_fp32" --tag "${slug}__cpu_fp32" --conditions same
done
step "copy baselines" cp -r "$HOME/ari_selfhosted_conc_time" "$OUT/time_baselines"
publish
