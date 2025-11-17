"""
This script creates a point cloud visualization in Blender using the blendertoolbox library.
It loads a PLY file, processes the point cloud data, and renders the scene.
"""
import argparse
from ast import main
import os
import random
import numpy as np
import plyfile
from tqdm import tqdm
import bpy # pylint: disable=import-error
import blendertoolbox as bt

def render_gen3r(args):
    CWD = os.getcwd()
    SCENE_NAME = args.scene_name
    TASK_NAME = args.task_name
    BASE_DATA_DIR = args.base_data_dir
    BASE_OUTPUT_DIR = args.base_output_dir
    # data/gen3r/object-centric/30
    COSTUMIZED_POSE = True
    #RE10k
    # SCENE_NAME = "25", "90", "85", "436", "328", "400"
    # SCENE_NAME = "113", "135", "150", "135", "147", "158", "164", "169", "181"
    # TASK_NAME = "gen3r/dl3dv"
    # SCENE_NAME = "35"
    PCD_PATH = os.path.join(BASE_DATA_DIR, TASK_NAME, SCENE_NAME, "pcds.ply")
    BASE_OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, TASK_NAME)
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    OUTPUT_PATH = os.path.join(BASE_OUTPUT_PATH, f'{SCENE_NAME}.png')  # make it abs path for windows

    # Initialize blender
    IMG_RES_X = 1024
    IMG_RES_Y = 1024
    NUM_SAMPLES = 100 
    EXPOSURE = 1.5
    bt.blenderInit(IMG_RES_X, IMG_RES_Y, NUM_SAMPLES, EXPOSURE)
    # PCD_PATH = "data/volume.ply"

    SAMPLE_RATE = 1
    # Read mesh
    POINT_SIZE = 0.0035
    SCALE = (0.5, 0.5, 0.5)
    # Set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
    CAM_LOCATION = (3, 0, 2)
    LOOK_AT_LOCATION = (0, 0, 0.5)
    FOCAL_LENGTH = 50  # (UI: click camera > Object Data > Focal Length)
    LOCATION = (2.75, 0.02, 1.81)
    ROTATION = (-108, 3, 92)
    #TODO set light location and rotation
    if TASK_NAME in ["co3d"]:
        POINT_SIZE = 0.01
    else:
        POINT_SIZE = 0.0035
    if SCENE_NAME in ["436"] and COSTUMIZED_POSE:
            LOCATION = (2.63, 0.14, 1.87)
            ROTATION = (-123, 6, 105)
    elif SCENE_NAME in ["328"] and COSTUMIZED_POSE:
            LOCATION = (2.57, -0.17, 1.76)
            ROTATION = (-109, -1, 74)
    elif SCENE_NAME in ["25", "85", "90"] and COSTUMIZED_POSE:
            LOCATION = (2.75, 0.02, 1.81)
            ROTATION = (-108, 3, 92)

    pcd = plyfile.PlyData.read(PCD_PATH)
    x = pcd['vertex']['x']
    y = pcd['vertex']['y']
    z = pcd['vertex']['z']
    P = np.array([x, y, z], dtype=np.float32).T 
    red = pcd['vertex']['red']
    green = pcd['vertex']['green']
    blue = pcd['vertex']['blue']
    PC = np.array([red, green, blue], dtype=np.float32).T / 255.0
    pt_color = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
    P = P[::SAMPLE_RATE]
    PC = PC[::SAMPLE_RATE]

    NUM_GROUPS = 1
    group_indices = np.random.choice(NUM_GROUPS, size=len(P))
    # Split the point cloud into parts
    mesh_list = []
    for i in tqdm(range(NUM_GROUPS)):
        mask = group_indices == i
        group_points = P[mask]
        group_colors = PC[mask]
        mesh_list.append(bt.readNumpyPoints(group_points, LOCATION, ROTATION, SCALE))
        mesh_list[i] = bt.setPointColors(mesh_list[i], group_colors)
        # point_size = random.uniform(0.01, 0.2)
        point_size = POINT_SIZE
        # Set material pt_color = (vertex_RGBA, H, S, V, Bright, Contrast)
        bt.setMat_pointCloudColored(mesh_list[i], pt_color, point_size)

    # Set invisible plane (shadow catcher)
    bt.invisibleGround(location=(0,0,-5), shadowBrightness=0.9)

    cam = bt.setCamera(CAM_LOCATION, LOOK_AT_LOCATION, FOCAL_LENGTH)

    # Set light
    LIGHT_ANGLE = (64, 0, 90)
    LIGHT_LOCATION = CAM_LOCATION
    STRENGTH = 2
    SHADOW_SOFTNESS = 0.3
    sun = bt.setLight_sun(LIGHT_ANGLE, STRENGTH, LIGHT_LOCATION, SHADOW_SOFTNESS)

    # Set ambient light
    bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))

    # Set gray shadow to completely white with a threshold
    bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')

    # Save blender file so that you can adjust parameters in the UI
    bpy.ops.wm.save_mainfile(filepath=os.path.join(BASE_OUTPUT_PATH, f'{SCENE_NAME}.blend'))

    # Save rendering
    bt.renderImage(OUTPUT_PATH, cam)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="35")
    parser.add_argument("--task_name", type=str, default="dl3dv")
    parser.add_argument("--base_data_dir", type=str, default=os.path.join("data", "gen3r"))
    parser.add_argument("--base_output_dir", type=str, default=os.path.join("outputs", "gen3r"))
    args = parser.parse_args()
    render_gen3r(args)