#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

raw=${DBMF_RAW:-/root/data1/folsom_multimodal/data/raw}
py=${DBMF_PYTHON:-/root/data1/miniconda3/envs/folsom-lab/bin/python}
output=${1:?Usage: bash run_smoke.sh NEW_RUN_DIRECTORY}
mkdir -p "$(dirname "$output")"
mkdir "$output"
exec > >(tee "$output/run.log") 2>&1

date -u +%FT%TZ
for year in 2014 2015 2016; do
    "$py" prepare_images.py "$raw/Folsom_sky_images_${year}.tar.bz2" \
        "$output/images/$year" --limit 2500
done

"$py" -u dbmf.py --raw "$raw" \
    --images "$output/images/2014" "$output/images/2015" "$output/images/2016" \
    --output "$output/result" --epochs 1 --limit-per-year 1024 --device cuda
date -u +%FT%TZ
