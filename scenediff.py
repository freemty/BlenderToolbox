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

OUTPUT_PATH = os.path.join(CWD, './scenediff_pcd.png')  # make it abs path for windows

# Initialize blender
IMG_RES_X = 1024
IMG_RES_Y = 1024
NUM_SAMPLES = 100
EXPOSURE = 1.5
bt.blenderInit(IMG_RES_X, IMG_RES_Y, NUM_SAMPLES, EXPOSURE)
PCD_PATH = "data/volume.ply"
SAMPLE_RATE = 3

# Read mesh
LOCATION = (0.7, -0.02, 0.75)
ROTATION = (78, 182, 268)
SCALE = (0.05, 0.05, 0.05)
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
    point_size = random.uniform(0.01, 0.2)
    # Set material pt_color = (vertex_RGBA, H, S, V, Bright, Contrast)
    bt.setMat_pointCloudColored(mesh_list[i], pt_color, point_size)

# Set invisible plane (shadow catcher)
bt.invisibleGround(location=(0,0,0.5), shadowBrightness=0.9)

# Set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
CAM_LOCATION = (3, 0, 2)
LOOK_AT_LOCATION = (0, 0, 0.5)
FOCAL_LENGTH = 45  # (UI: click camera > Object Data > Focal Length)
cam = bt.setCamera(CAM_LOCATION, LOOK_AT_LOCATION, FOCAL_LENGTH)

# Set light
LIGHT_ANGLE = (6, -30, -155)
STRENGTH = 2
SHADOW_SOFTNESS = 0.3
sun = bt.setLight_sun(LIGHT_ANGLE, STRENGTH, SHADOW_SOFTNESS)

# Set ambient light
bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))

# Set gray shadow to completely white with a threshold
bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')

# Save blender file so that you can adjust parameters in the UI
bpy.ops.wm.save_mainfile(filepath=os.path.join(CWD, 'test.blend'))

# Save rendering
bt.renderImage(OUTPUT_PATH, cam)
