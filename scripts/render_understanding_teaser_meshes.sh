#!/bin/bash
set -euo pipefail

DATA_ROOT="scene_understanding/teaser_material"
DATA_DIR="data/${DATA_ROOT}"
scene_names=(0568_00)
scene_names=(0568_00 0000_00 0011_01 0095_00 0221_00 0354_00 0435_00)

if [ ! -d "$DATA_DIR" ]; then
  echo "Data directory not found: $DATA_DIR" >&2
  exit 1
fi

for scene_name in "${scene_names[@]}"; do
  scene_file_name="scene${scene_name}_vh_clean_2"
  task_name="${DATA_ROOT}/${scene_name}"
  echo "Rendering $scene_name"

  python examples/render_sgmesh.py \
    --task_name "$task_name" \
    --scene_name "$scene_file_name"

  python examples/render_sgmesh.py \
    --task_name "$task_name" \
    --scene_name "$scene_file_name" \
    --render_seg

  python examples/render_sgmesh.py \
    --task_name "$task_name" \
    --scene_name "$scene_file_name" \
    --render-graph

  python examples/render_sgmesh.py \
    --task_name "$task_name" \
    --scene_name "$scene_file_name" \
    --render_seg \
    --render-graph
done
