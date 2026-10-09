#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Self-hosted CPU panel, day >=1, for google/embeddinggemma-2: `conc` and `time`
# against the day-0 baseline from cpu_panel_gemma.sh, on the same host.
export LABEL=${LABEL:-cpu_gemma_time}
source "$(dirname "$0")/common.sh"
VG=$HOME/venvgemma/bin
m=google/embeddinggemma-2; slug=${m//\//_}
step "conc-time $m" $VG/python ari/tools/selfhosted_conc_time.py --mode measure --model "$m" --inputs "$INPUTS"
step "copy results" cp -r "$HOME/ari_selfhosted_conc_time/$slug" "$OUT/conc_time_$slug"
publish
