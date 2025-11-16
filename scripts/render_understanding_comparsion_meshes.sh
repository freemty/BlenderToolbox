#!/bin/bash
set -euo pipefail
SCENE_NAME="0435_00"
python examples/render_sgmesh.py --scene_name=scene${SCENE_NAME}_vh_clean_2 --task_name=scene_understanding/comparsion_material/openscene/${SCENE_NAME} 
python examples/render_sgmesh.py --render_seg --scene_name=scene${SCENE_NAME}_vh_clean_2 --task_name=scene_understanding/comparsion_material/openscene/${SCENE_NAME} 
python examples/render_sgmesh.py --scene_name=scene${SCENE_NAME}_vh_clean_2 --task_name=scene_understanding/comparsion_material/ours/${SCENE_NAME} 
python examples/render_sgmesh.py --render_seg --scene_name=scene${SCENE_NAME}_vh_clean_2 --task_name=scene_understanding/comparsion_material/ours/${SCENE_NAME} 
SCENE_NAME="0406_00"
python examples/render_sgmesh.py --scene_name=scene${SCENE_NAME}_vh_clean_2 --task_name=scene_understanding/comparsion_material/ours/${SCENE_NAME} 
python examples/render_sgmesh.py --render_seg --scene_name=scene${SCENE_NAME}_vh_clean_2 --task_name=scene_understanding/comparsion_material/ours/${SCENE_NAME} 
python examples/render_sgmesh.py --scene_name=scene${SCENE_NAME}_vh_clean_2 --task_name=scene_understanding/comparsion_material/openscene/${SCENE_NAME} 
python examples/render_sgmesh.py --render_seg --scene_name=scene${SCENE_NAME}_vh_clean_2 --task_name=scene_understanding/comparsion_material/openscene/${SCENE_NAME} 