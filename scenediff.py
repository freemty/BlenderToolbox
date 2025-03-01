# # if you want to call the toolbox the old way with `blender -b -P demo_XXX.py`, then uncomment these two lines
# import sys, os
# sys.path.append("../../BlenderToolbox/")
import blendertoolbox as bt 
import bpy
import os
import random
import numpy as np
import plyfile
from tqdm import tqdm


cwd = os.getcwd()

outputPath = os.path.join(cwd, './scenediff_pcd.png') # make it abs path for windows

## initialize blender
imgRes_x = 1024 
imgRes_y = 1024 
numSamples = 100 
exposure = 1.5 
bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
pcd_path = "data/volume.ply"
## read mesh 
location = (0.7,-0.02,0.75)
rotation = (78,182,268) 
scale = (.05,.05,.05)
pcd = plyfile.PlyData.read(pcd_path)
x = pcd['vertex']['x']
y = pcd['vertex']['y']
z = pcd['vertex']['z']
P = np.array([x, y, z], dtype=np.float32).T
red = pcd['vertex']['red']
green = pcd['vertex']['green']
blue = pcd['vertex']['blue']
PC = np.array([red, green, blue], dtype=np.float32).T / 255.0
ptColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
P = P[::3]
PC = PC[::3]

num_groups = 20
group_indices = np.random.choice(num_groups, size=len(P))
# split the point cloud into len(PC) parts
mesh_list = []
for i in tqdm(range(num_groups)):
    mask = (group_indices == i)
    group_points = P[mask]
    group_colors = PC[mask]
    mesh_list.append(bt.readNumpyPoints(group_points,location,rotation,scale))
    mesh_list[i] = bt.setPointColors(mesh_list[i], group_colors)
    point_size = random.uniform(0.01, 0.2)
    ## set material ptColor = (vertex_RGBA, H, S, V, Bright, Contrast)
    bt.setMat_pointCloudColored(mesh_list[i], ptColor, point_size)



## set invisible plane (shadow catcher)
bt.invisibleGround(shadowBrightness=0.9)

## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
camLocation = (3, 0, 2)
lookAtLocation = (0,0,0.5)
focalLength = 45 # (UI: click camera > Object Data > Focal Length)
cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

## set light
lightAngle = (6, -30, -155) 
strength = 2
shadowSoftness = 0.3
sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

## set ambient light
bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

## set gray shadow to completely white with a threshold 
bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

## save blender file so that you can adjust parameters in the UI
bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')

## save rendering
bt.renderImage(outputPath, cam)