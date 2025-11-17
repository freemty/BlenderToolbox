"""
This script creates a point cloud visualization in Blender using the blendertoolbox library.
It loads a PLY file, processes the point cloud data, and renders the scene.
"""
import os
import random
import numpy as np
import plyfile
from tqdm import tqdm
import bpy # pylint: disable=import-error
import blendertoolbox as bt


CWD = os.getcwd()
COSTUMIZED_POSE = False
TASK_NAME = "scene_understanding"
# Teaser scene list
# SCENE_NAME = "scene0000_01_vh_clean_2."
# SCENE_NAME = "scene0568_00_vh_clean_2"
# SCENE_NAME = "scene0011_00_vh_clean_2"
# SCENE_NAME = "scene0144_00_vh_clean_2"
# SCENE_NAME = "scene0131_00_vh_clean_2"
# SCENE_NAME = "scene0046_00_vh_clean_2"
# SCENE_NAME = "scene0100_00_vh_clean_2"

SCENE_NAME = "scene0231_00_vh_clean_2.labels"
PCD_PATH = os.path.join(CWD, "data", TASK_NAME, SCENE_NAME + ".ply")
BASE_OUTPUT_PATH = os.path.join(CWD, "outputs", TASK_NAME)
os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
OUTPUT_PATH = os.path.join(BASE_OUTPUT_PATH, f'{SCENE_NAME}.png')  # make it abs path for windows


# Read mesh

# by default
POINT_SIZE = 0.07
LOCATION = (2.2, 0.13, 1.3)
ROTATION = (-155, 227, 333)
SCALE = (0.05, 0.05, 0.05)
if COSTUMIZED_POSE and "scene0081_00_vh_clean_2" in SCENE_NAME:
    LOCATION = (2.48, 0.2, 1.65)
    ROTATION = (-181, 186, 373)
elif COSTUMIZED_POSE and "scene0144_00_vh_clean_2" in SCENE_NAME:
    LOCATION = (2.59, 0.1, 1.67)
    ROTATION = (-174, 227, 346)
elif COSTUMIZED_POSE and "scene0131_00_vh_clean_2" in SCENE_NAME:
    LOCATION = (2.3, -0.24, 1.55)
    ROTATION = (-152, 164, 244)
elif COSTUMIZED_POSE and "scene0046_00_vh_clean_2" in SCENE_NAME:
    LOCATION = (2.35, -0.07, 1.77)
    ROTATION = (-190, 163, 523)
elif COSTUMIZED_POSE and "scene0100_00_vh_clean_2" in SCENE_NAME:
    LOCATION = (2.59, -0.14, 1.88)
    ROTATION = (190, 145, 210)

# Set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
CAM_LOCATION = (3, 0, 2)
LOOK_AT_LOCATION = (0, 0, 0.5)
FOCAL_LENGTH = 70  # (UI: click camera > Object Data > Focal Length)
# Set light
LIGHT_ANGLE = (64, 0, 90)
# LIGHT_ANGLE = (6, -30, -155)
STRENGTH = 2
LIGHT_LOCATION = CAM_LOCATION
SHADOW_SOFTNESS = 0.3

# Initialize blender
IMG_RES_X = 1024
IMG_RES_Y = 1024
NUM_SAMPLES = 100
EXPOSURE = 1.5
bt.blenderInit(IMG_RES_X, IMG_RES_Y, NUM_SAMPLES, EXPOSURE)
# PCD_PATH = "data/volume.ply"

SAMPLE_RATE = 1

# XYZ_SCALE = np.array([1, 1, 1])
# XYZ_SHIFT = np.array([0.0, 0.0, 0.0])

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
bt.invisibleGround(location=(0,0,0.2), shadowBrightness=0.9)

cam = bt.setCamera(CAM_LOCATION, LOOK_AT_LOCATION, FOCAL_LENGTH)


sun = bt.setLight_sun(LIGHT_ANGLE, STRENGTH, LIGHT_LOCATION, SHADOW_SOFTNESS)

# Set ambient light
bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))

# Set gray shadow to completely white with a threshold
bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')

# Save blender file so that you can adjust parameters in the UI
bpy.ops.wm.save_mainfile(filepath=os.path.join(BASE_OUTPUT_PATH, f'{SCENE_NAME}.blend'))

# Save rendering
bt.renderImage(OUTPUT_PATH, cam)
