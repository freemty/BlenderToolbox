import os
import random
import plyfile
import numpy as np
import bpy # pylint: disable=import-error
from mathutils import Vector # pylint: disable=import-error
from tqdm import tqdm

import blendertoolbox as bt

# Basic Configuration
PCD_PATH = "data/volume.ply"         # Point cloud file path
SAMPLE_RATE = 10                      # Point cloud sampling rate (higher value means fewer ellipsoids)
ELLIPSE_SCALE = (0.02, 0.02, 0.02)   # Base scaling ratio
BASE_SUBDIV = 2                      # Base subdivision level (0-5)

# Optimization Mode Selection
USE_VERTEX_COLOR = True              # Use vertex colors (True) or materials (False)
USE_GEOMETRY_NODES = False           # Whether to use geometry nodes instancing (Blender 3.0+)
MERGE_THRESHOLD = 5000               # Auto-merge batches when ellipsoid count exceeds this value

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

def clean_scene():
    """Clean the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    bpy.ops.outliner.orphans_purge()


def create_base_sphere():
    """Create base sphere template (low poly)"""
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1.0,
        segments=16,
        ring_count=8
    )
    base_sphere = bpy.context.active_object
    base_sphere.name = "BaseSphereTemplate"
    base_sphere.hide_render = True
    base_sphere.hide_viewport = True

    # Apply subdivision modifier
    subsurf = base_sphere.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.levels = BASE_SUBDIV
    subsurf.render_levels = BASE_SUBDIV
    bpy.context.view_layer.update()
    bpy.ops.object.modifier_apply(modifier=subsurf.name)

    return base_sphere.data


def create_merged_ellipsoids(points, colors):
    """Batch merge ellipsoids into a single mesh"""
    # Get base sphere data
    base_mesh = create_base_sphere()
    base_verts = [v.co for v in base_mesh.vertices]
    base_faces = [p.vertices[:] for p in base_mesh.polygons]

    # Create merge container
    merged_mesh = bpy.data.meshes.new("MergedEllipsoids")
    merged_obj = bpy.data.objects.new("MergedEllipsoids", merged_mesh)
    bpy.context.collection.objects.link(merged_obj)

    all_verts = []
    all_faces = []
    color_data = []

    # Material setup
    if not USE_VERTEX_COLOR:
        material = bpy.data.materials.new(name="EllipsoidMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        output = nodes.new('ShaderNodeOutputMaterial')
        material.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        merged_obj.data.materials.append(material)

    # Batch process each ellipsoid
    for idx, (pos, color) in tqdm(enumerate(zip(points, colors)),
                                total=len(points),
                                desc="Merging"):
        # Random scaling
        scale = (
            random.uniform(1, 10) * ELLIPSE_SCALE[0],
            random.uniform(1, 10) * ELLIPSE_SCALE[1],
            random.uniform(1, 10) * ELLIPSE_SCALE[2]
        )

        # Transform vertices
        vert_offset = len(all_verts)
        transformed_verts = [
            (
                pos[0] + v.x * scale[0],
                pos[1] + v.y * scale[1],
                pos[2] + v.z * scale[2]
            ) for v in base_verts
        ]
        all_verts.extend(transformed_verts)

        # Build faces
        all_faces.extend([[v + vert_offset for v in f] for f in base_faces])

        # Record color data
        if USE_VERTEX_COLOR:
            color_data.extend([color] * len(transformed_verts))

    # Build mesh
    merged_mesh.from_pydata(all_verts, [], all_faces)

    # Add vertex colors
    if USE_VERTEX_COLOR:
        color_layer = merged_mesh.vertex_colors.new()
        color_data_np = np.array(color_data, dtype=np.float32)
        alpha = np.ones((color_data_np.shape[0], 1), dtype=np.float32)
        color_with_alpha = np.hstack([color_data_np, alpha])
        vertex_indices = np.array([loop.vertex_index for loop in merged_mesh.loops], dtype=np.int32)
        loop_colors = color_with_alpha[vertex_indices].flatten()
        color_layer.data.foreach_set("color", loop_colors)

        # Configure material
        mat = bpy.data.materials.new(name="VertexColorMat")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()

        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Roughness'].default_value = 1.0

        attr = nodes.new('ShaderNodeAttribute')
        attr.attribute_name = "Col"

        output = nodes.new('ShaderNodeOutputMaterial')
        mat.node_tree.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        merged_obj.data.materials.append(mat)

    return merged_obj


if __name__ == "__main__":
    bt.blenderInit(IMG_RES_X, IMG_RES_Y, NUM_SAMPLES, EXPOSURE)

    # Load point cloud data
    ply = plyfile.PlyData.read(PCD_PATH)
    vertices = ply['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0

    # Data sampling
    points = points[::SAMPLE_RATE]
    colors = colors[::SAMPLE_RATE]

    # Automatically select optimization mode based on data volume
    if USE_GEOMETRY_NODES and len(points) > MERGE_THRESHOLD:
        # Geometry nodes mode (requires Blender 3.0+)
        raise NotImplementedError("Geometry nodes mode requires manual configuration in Blender interface")
    else:
        # Merged mesh mode
        merged_obj = create_merged_ellipsoids(points, colors)
        merged_obj.location = (0.7, -0.02, 0.75)
        merged_obj.rotation_euler = tuple((np.array([78, 182, 268]) * 1.0 / 180.0 * np.pi).tolist())
        merged_obj.scale = (0.05, 0.05, 0.05)

    # setup scene
    cam = bt.setCamera(CAM_LOCATION, LOOK_AT_LOCATION, FOCAL_LENGTH)
    sun = bt.setLight_sun(LIGHT_ANGLE, STRENGTH, SHADOW_SOFTNESS)
    bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))
    bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')

    print("Saving scene blend...")
    save_path = os.path.join(os.getcwd(), "optimized_ellipsoids.blend")
    bpy.ops.wm.save_as_mainfile(filepath=save_path)

    print("Starting render...")
    render_path = os.path.join(os.getcwd(), "render_result.png")
    bt.renderImage(render_path, cam)

    print(f"Scene saved to: {save_path}")
    print(f"Render result saved to: {render_path}")
