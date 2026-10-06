#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Self-hosted CPU panel, day >=1: `conc` and `time` against the day-0 baselines.
export LABEL=${LABEL:-cpu_time}
source "$(dirname "$0")/common.sh"
for m in "${MODELS8[@]}"; do
  $VP/python ari/tools/selfhosted_conc_time.py --mode measure --model "$m" --inputs "$INPUTS"
done
cp -r "$HOME/ari_selfhosted_conc_time" "$OUT/conc_time"
publish
