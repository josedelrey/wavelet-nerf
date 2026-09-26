#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Based on bmild/nerf's download_example_data.sh; upstream notice: LICENSE.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dataset_dir="$script_dir/datasets"
archive_url="https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip"
scenes=(lego fern)
missing_scenes=()

for scene in "${scenes[@]}"; do
    if [[ -e "$dataset_dir/$scene" || -L "$dataset_dir/$scene" ]]; then
        printf 'Keeping existing dataset path: %s\n' "$dataset_dir/$scene"
    else
        missing_scenes+=("$scene")
    fi
done

if (( ${#missing_scenes[@]} == 0 )); then
    printf 'Both dataset paths already exist; nothing to download.\n'
    exit 0
fi

for command in wget unzip; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$command" >&2
        exit 1
    fi
done

mkdir -p -- "$dataset_dir"
temp_dir="$(mktemp -d "$dataset_dir/.nerf-download.XXXXXX")"
trap 'rm -rf -- "$temp_dir"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

archive_path="$temp_dir/nerf_example_data.zip"
extract_dir="$temp_dir/extracted"
wget --tries=3 --timeout=30 --progress=dot:giga --output-document="$archive_path" "$archive_url"
unzip -q "$archive_path" 'nerf_synthetic/lego/*' 'nerf_llff_data/fern/*' -d "$extract_dir"

lego_source="$extract_dir/nerf_synthetic/lego"
fern_source="$extract_dir/nerf_llff_data/fern"

# Validate both scenes before installing either one.
for split in train val test; do
    if [[ ! -f "$lego_source/transforms_$split.json" || ! -d "$lego_source/$split" ]]; then
        printf 'Archive is missing the Lego %s split.\n' "$split" >&2
        exit 1
    fi
done

if [[ ! -f "$fern_source/poses_bounds.npy" || ! -d "$fern_source/images" ]]; then
    printf 'Archive is missing Fern camera metadata or images.\n' >&2
    exit 1
fi

for scene in "${missing_scenes[@]}"; do
    if [[ -e "$dataset_dir/$scene" || -L "$dataset_dir/$scene" ]]; then
        printf 'Keeping dataset path created during download: %s\n' "$dataset_dir/$scene"
        continue
    fi
    if [[ "$scene" == lego ]]; then
        source_dir="$lego_source"
    else
        source_dir="$fern_source"
    fi
    mv -n -- "$source_dir" "$dataset_dir/"
    printf 'Dataset available at: %s\n' "$dataset_dir/$scene"
done
