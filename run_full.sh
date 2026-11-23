#!/usr/bin/env bash
set -euo pipefail

cache=${1:?Usage: bash run_full.sh ABSOLUTE_CACHE_DIR ABSOLUTE_NEW_RUN_DIR}
output=${2:?Missing run directory}
raw=${DBMF_RAW:-/root/data1/folsom_multimodal/data/raw}
py=${DBMF_PYTHON:-/root/data1/miniconda3/envs/folsom-lab/bin/python}
cd "$(dirname "$0")"
for year in 2014 2015 2016; do
    test -s "$cache/$year/complete.json"
done
mkdir -p "$(dirname "$output")"
mkdir "$output"
exec > >(tee "$output/run.log") 2>&1

date -u +%FT%TZ
if "$py" -u dbmf.py --raw "$raw" \
    --image-cache "$cache/2014" "$cache/2015" "$cache/2016" \
    --image-clock mtime --issue-clock image --max-history-gap 2 \
    --output "$output/result" --epochs 50 --seed 42 --workers 4 --device cuda; then
    status=0
else
    status=$?
fi
date -u +%FT%TZ
printf '%s\n' "$status" > "$output/exit_code"
exit "$status"
