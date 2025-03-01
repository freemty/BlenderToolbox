import os
import math
import random
import plyfile
import numpy as np
import bpy # pylint: disable=import-error
from mathutils import Vector # pylint: disable=import-error
from tqdm import tqdm

import blendertoolbox as bt


PCD_PATH = "data/volume.ply"
OUTPUT_PATH = os.path.join(os.getcwd(), 'ellipsoid.png')

IMG_RES_X = 1024
IMG_RES_Y = 1024
NUM_SAMPLES = 100
EXPOSURE = 1.5

CAM_LOCATION = (3, 0, 2)
LOOK_AT_LOCATION = (0, 0, 0.5)
FOCAL_LENGTH = 45  # (UI: click camera > Object Data > Focal Length)

LIGHT_ANGLE = (6, -30, -155)
STRENGTH = 2
SHADOW_SOFTNESS = 0.3

# Clear scene (delete default objects)
def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge()  # Clean up residual data


# Create ellipsoid function
def create_ellipsoid(name="Ellipsoid",
                    major_axis=3.0,
                    middle_axis=2.0,
                    minor_axis=1.5,
                    location=(0, 0, 0),
                    rotation=(0, 0, 0),
                    subdivisions=3):
    """
    Parameters:
    major_axis: Length of major axis (X direction)
    middle_axis: Length of middle axis (Y direction)
    minor_axis: Length of minor axis (Z direction)
    subdivisions: Subdivision level (improves model quality)
    """
    # Create base sphere
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1.0,
        location=location,
        segments=64,
        ring_count=32
    )
    obj = bpy.context.active_object
    obj.name = name

    # Apply scaling to create ellipsoid shape
    obj.scale = (major_axis, middle_axis, minor_axis)
    bpy.ops.object.transform_apply(scale=True)

    # Set auto smooth
    mesh = obj.data
    mesh.use_auto_smooth = False
    mesh.auto_smooth_angle = 1.0472  # 60 degrees

    # Add subdivision surface modifier
    subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.levels = subdivisions
    subsurf.render_levels = subdivisions

    # Set rotation
    obj.rotation_euler = Vector(rotation)

    return obj


# Main program
if __name__ == "__main__":
    bt.blenderInit(IMG_RES_X, IMG_RES_Y, NUM_SAMPLES, EXPOSURE)

    pcd = plyfile.PlyData.read(PCD_PATH)
    x = pcd['vertex']['x'] + 0.7
    y = pcd['vertex']['y'] - 0.02
    z = pcd['vertex']['z'] + 0.75
    P = np.array([x, y, z], dtype=np.float32).T
    red = pcd['vertex']['red']
    green = pcd['vertex']['green']
    blue = pcd['vertex']['blue']
    PC = np.array([red, green, blue], dtype=np.float32).T / 255.0
    PC = np.concatenate((PC, np.ones((PC.shape[0], 1))), axis=1)
    LOCATION = (0.7, -0.02, 0.75)
    ROTATION = tuple((np.array([78, 182, 268]) * 1.0 / 180.0 * np.pi).tolist())
    SCALE = (0.05, 0.05, 0.05)

    # clean_scene()

    for i in tqdm(range(50)): # pylint: disable=C0200
        # Ellipsoid parameters
        ellipsoid_params = {
            "name": "MyEllipsoid",
            "major_axis": random.uniform(0.01, 0.2),
            "middle_axis": random.uniform(0.01, 0.2),
            "minor_axis": random.uniform(0.01, 0.2),
            "location": (P[i][0], P[i][1], P[i][2]),
            "rotation": (math.radians(30), 0, math.radians(45)),
            "subdivisions": 3
        }

        # Create ellipsoid
        ellipsoid = create_ellipsoid(**ellipsoid_params)
        ellipsoid.location = LOCATION
        ellipsoid.rotation_euler = Vector(ROTATION)
        ellipsoid.scale = SCALE

        mat = bpy.data.materials.new(name=f"PointMat_{i}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for node in nodes:
            nodes.remove(node)
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = PC[i]
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 1.0

        output = nodes.new('ShaderNodeOutputMaterial')
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        mat.blend_method = 'OPAQUE'
        mat.shadow_method = 'OPAQUE'

        ellipsoid.data.materials.append(mat)

    # bt.invisibleGround(shadowBrightness=0.9)
    cam = bt.setCamera(CAM_LOCATION, LOOK_AT_LOCATION, FOCAL_LENGTH)
    sun = bt.setLight_sun(LIGHT_ANGLE, STRENGTH, SHADOW_SOFTNESS)
    bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))
    bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')

    # Save project file
    save_path = os.getcwd() + '/test.blend'
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    print(f"Ellipsoid created and saved to: {save_path}")

    # Save rendering
    bt.renderImage(OUTPUT_PATH, cam)
