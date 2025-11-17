#! /bin/bash
set -e
base_dir=data/gen3r/
base_output_dir=outputs/gen3r/
scene_type=re10k # dl3dv, re10k, object-centric
for scene in $(ls $base_dir/$scene_type); do
    echo processing: $(basename $scene)
    python examples/render_gen3r.py \
        --scene_name $(basename $scene) \
        --task_name $scene_type \
        --base_data_dir $base_dir \
        --base_output_dir $base_output_dir
done