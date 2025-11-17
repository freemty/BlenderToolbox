"""
Render a ScanNet mesh in Blender using the blendertoolbox library.
The script imports the mesh, preserves vertex colors, and outputs a still image.

python examples/render_sgmesh.py --scene_name=data_example/scene0406_00_vh_clean_2 --task_name=scene_understanding/comparsion_material/ours/0406_00 
python examples/render_sgmesh.py --render_seg --scene_name=data_example/scene0406_00_vh_clean_2 --task_name=scene_understanding/comparsion_material/ours/0406_00 
python examples/render_sgmesh.py --scene_name=data_example/scene0406_00_vh_clean_2 --task_name=scene_understanding/comparsion_material/openscene/0406_00 
python examples/render_sgmesh.py --render_seg --scene_name=data_example/scene0406_00_vh_clean_2 --task_name=scene_understanding/comparsion_material/openscene/0406_00 
"""
import argparse
import os
import json
import math

import numpy as np
import plyfile
import torch
import bpy  # pylint: disable=import-error
from mathutils import Euler, Vector, Matrix
import blendertoolbox as bt
try:
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


COSTUMIZED_POSE = False
# Teaser scene list
# scene0000_01_vh_clean_2
# scene0568_00_vh_clean_2
# scene0011_00_vh_clean_2
# scene0144_00_vh_clean_2
# scene0131_00_vh_clean_2
# scene0046_00_vh_clean_2
# scene0100_00_vh_clean_2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a ScanNet mesh using BlenderToolbox."
    )
    parser.add_argument(
        "--scene_name",
        # nargs="?",
        default="scene0011_01_vh_clean_2",
        help="Scene basename (without extension) to render.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="data/scene_understanding/teaser_material/0011_01",
        help="Optional path to a scene-graph JSON (nodes, edges, optional colors).",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Optional path to a scene-graph JSON (nodes, edges, optional colors).",
    )
    # parser.add_argument(
    #     "",
    #     type=str,
    #     default=None,
    #     help="Optional path to a scene-graph JSON (nodes, edges, optional colors).",
    # )
    parser.add_argument(
        "--graph-space",
        choices=["mesh", "world"],
        default="mesh",
        help="Whether graph node positions are in mesh(local) space or already in world space.",
    )
    parser.add_argument(
        "--node-radius",
        type=float,
        default=0.06,
        help="Sphere radius for graph nodes.",
    )
    parser.add_argument(
        "--edge-radius",
        type=float,
        default=0.01,
        help="Cylinder radius for graph edges.",
    )
    parser.add_argument(
        "--edge-emission",
        type=float,
        default=2.0,
        help="Emission strength for edges. Set 0 to disable emission.",
    )
    parser.add_argument(
        "--graph-join",
        action="store_true",
        default=False,
        help="Join all scene graph parts (spheres/cylinders) into one mesh object.",
    )
    parser.add_argument(
        "--render-graph",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="If set, load scene_name_graph.pt/json from the mesh folder (or from the provided path) and render scene graph. Usage: --render-graph [path/to/graph.pt]",
    )
    parser.add_argument(
        "--render_seg",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="If set, load scene_name_seg.ply from the mesh folder (or from the provided path) and override mesh vertex RGB colors. Usage: --render_seg [path/to/seg.ply]",
    )
    parser.add_argument(
        "--render-feat",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="If set, load scene_name_feature.pt from the mesh folder (or from the provided path) and use first 3 channels as RGB colors. Usage: --render_feat [path/to/feature.pt]",
    )
    parser.add_argument(
        "--node-colormap",
        type=str,
        default=None,
        help="Colormap name for node colors (e.g., 'plasma', 'viridis', 'spectral', 'tab10'). If not specified, uses default palette. Requires matplotlib.",
    )
    return parser.parse_args()


def _as_xyz_tuple(p):
    a = np.asarray(p, dtype=np.float32).reshape(-1)
    if a.size < 3:
        raise ValueError("Point has fewer than 3 elements")
    return float(a[0]), float(a[1]), float(a[2])


def find_auxiliary_file(scene_name, task_name, pcd_path, file_type, custom_path=None):
    """
    Find auxiliary files (graph, seg, feature) with unified naming convention.
    
    Args:
        scene_name: Scene name (e.g., "scene0568_00_vh_clean_2")
        task_name: Task name (e.g., "scene_understanding/0568_00")
        pcd_path: Path to the main PLY file
        file_type: One of "graph", "seg", "feature"
        custom_path: Optional custom path provided by user
    
    Returns:
        Path to the file if found, None otherwise
    """
    base_name = scene_name.replace('.labels', '')
    
    # File naming convention: {scene_name}_{type}.{ext}
    file_patterns = {
        "graph": [f"{base_name}_graph.pt", f"{base_name}_graph.json"],
        "seg": [f"{base_name}_seg.ply"],
        "feature": [f"{base_name}_feature.pt"],
    }
    
    if file_type not in file_patterns:
        raise ValueError(f"Unknown file type: {file_type}")
    
    # If custom path provided and exists, use it
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    # Try to find in the same folder as the mesh
    search_paths = []
    for pattern in file_patterns[file_type]:
        search_paths.append(os.path.join(os.path.dirname(pcd_path), pattern))
    
    # Also try in the task folder (for backward compatibility)
    if task_name:
        base_root = task_name.split('/')[0] if '/' in task_name else task_name
        base_name_clean = scene_name.replace('.labels', '')
        id_part = base_name_clean.split('_vh_clean_2')[0] if '_vh_clean_2' in base_name_clean else base_name_clean
        folder_id = id_part.replace('scene', '')
        for pattern in file_patterns[file_type]:
            search_paths.append(os.path.join(base_root, folder_id, pattern))
    
    # Return first existing path
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    return None


def get_colormap_colors(colormap_name, num_colors):
    """
    Generate colors from a matplotlib colormap.
    
    Args:
        colormap_name: Name of the colormap (e.g., 'plasma', 'viridis', 'spectral')
        num_colors: Number of colors to generate
    
    Returns:
        List of RGBA tuples, each in range [0, 1]
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ValueError("matplotlib is required for colormap support. Install it with: pip install matplotlib")
    
    # Try different ways to get colormap (compatible with different matplotlib versions)
    cmap = None
    try:
        # Try matplotlib 3.5+ API
        if hasattr(cm, 'get_cmap'):
            cmap = cm.get_cmap(colormap_name)
        elif hasattr(plt.cm, 'get_cmap'):
            cmap = plt.cm.get_cmap(colormap_name)
        else:
            # Fallback: direct access
            cmap = getattr(plt.cm, colormap_name, None)
            if cmap is None:
                cmap = getattr(cm, colormap_name, None)
    except (ValueError, AttributeError) as e:
        # List available colormaps
        try:
            available = sorted([name for name in plt.cm.datad.keys() if not name.endswith('_r')])
        except:
            available = ['plasma', 'viridis', 'spectral', 'tab10', 'Set1', 'Set2', 'Set3']
        raise ValueError(f"Unknown colormap: {colormap_name}. Available colormaps include: {', '.join(available[:10])}...")
    
    if cmap is None:
        raise ValueError(f"Could not find colormap: {colormap_name}")
    
    # Generate evenly spaced values in [0, 1]
    values = np.linspace(0, 1, num_colors)
    colors = []
    for v in values:
        rgba = cmap(v)
        # Handle both tuple and array returns
        if isinstance(rgba, (tuple, list, np.ndarray)):
            r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
            a = float(rgba[3]) if len(rgba) > 3 else 1.0
        else:
            # Handle scalar colormaps (unlikely but safe)
            r = g = b = float(rgba)
            a = 1.0
        colors.append((r, g, b, a))
    return colors


def transform_points(points_np, location, rotation_deg, scale_xyz):
    R = Euler(tuple(np.deg2rad(rotation_deg)), 'XYZ').to_matrix()
    S = Matrix.Diagonal((scale_xyz[0], scale_xyz[1], scale_xyz[2]))
    M3 = R @ S
    t = Vector(location)
    out = []
    for p in np.asarray(points_np, dtype=np.float32):
        x, y, z = _as_xyz_tuple(p)
        v = Vector((x, y, z))
        w = M3 @ v + t
        out.append((w.x, w.y, w.z))
    return np.asarray(out, dtype=np.float32)


def transform_points_by_object(points_np, obj):
    M = obj.matrix_world
    out = []
    for p in np.asarray(points_np, dtype=np.float32):
        x, y, z = _as_xyz_tuple(p)
        w = M @ Vector((x, y, z, 1.0))
        out.append((float(w.x), float(w.y), float(w.z)))
    return np.asarray(out, dtype=np.float32)


def draw_edges_with_emission(P1, P2, radius, colors_rgba=None, emission_strength=2.0, target_collection=None, parent=None):
    import bpy
    import blendertoolbox as bt
    n = len(P1)
    for i in range(n):
        a = P1[i]
        b = P2[i]
        dx, dy, dz = (b[0]-a[0]), (b[1]-a[1]), (b[2]-a[2])
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=radius,
            depth=dist,
            location=(a[0]+dx/2.0, a[1]+dy/2.0, a[2]+dz/2.0)
        )
        obj = bpy.context.object
        phi = math.atan2(dy, dx)
        theta = math.acos(dz / dist) if dist > 1e-8 else 0.0
        obj.rotation_euler[1] = theta
        obj.rotation_euler[2] = phi
        if parent is not None:
            obj.parent = parent
        if target_collection is not None:
            # move object into the target collection
            for c in list(obj.users_collection):
                try:
                    c.objects.unlink(obj)
                except Exception:
                    pass
            target_collection.objects.link(obj)
        if emission_strength > 0.0:
            if colors_rgba is not None:
                c = colors_rgba[i]
                rgba = (float(c[0]), float(c[1]), float(c[2]), float(c[3] if len(c) > 3 else 1.0))
            else:
                rgba = (1.0, 0.6, 0.0, 1.0)
            edge_color = bt.colorObj(rgba, 0.5, 1.0, 1.0, 0.0, 0.0)
            bt.setMat_emission(obj, edge_color, emission_strength)


def load_graph_pt(pt_path):
    G = torch.load(pt_path, map_location='cpu')
    nodes_list = G.get('nodes', [])
    edges_list = G.get('edges', [])
    nodes = []
    for n in nodes_list:
        cp = n.get('central_point', None)
        if cp is None:
            bbox = n.get('bbox', None)
            if bbox is not None:
                try:
                    cp = (np.asarray(bbox[0], dtype=np.float32).reshape(-1)[:3] +
                          np.asarray(bbox[1], dtype=np.float32).reshape(-1)[:3]) / 2.0
                except Exception:
                    cp = None
        if cp is not None:
            x, y, z = _as_xyz_tuple(cp)
            nodes.append([x, y, z])
    nodes = np.asarray(nodes, dtype=np.float32) if len(nodes) > 0 else np.zeros((0,3), dtype=np.float32)
    edges = np.asarray(edges_list, dtype=np.int64) if len(edges_list) > 0 else np.zeros((0,2), dtype=np.int64)
    return nodes, edges


def ensure_collection(name: str):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def ensure_empty(name: str, collection):
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != 'EMPTY':
        obj = bpy.data.objects.new(name, None)
        obj.empty_display_type = 'PLAIN_AXES'
        obj.empty_display_size = 0.2
        collection.objects.link(obj)
    elif collection not in obj.users_collection:
        collection.objects.link(obj)
    return obj


def link_to_collection(obj, collection):
    for c in list(obj.users_collection):
        try:
            c.objects.unlink(obj)
        except Exception:
            pass
    collection.objects.link(obj)


def join_collection_meshes(collection, new_name, parent=None):
    meshes = [o for o in collection.objects if o.type == 'MESH']
    if len(meshes) == 0:
        return None
    if len(meshes) == 1:
        obj = meshes[0]
        obj.name = new_name
        if parent is not None:
            obj.parent = parent
        return obj
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = new_name
    if parent is not None:
        obj.parent = parent
    return obj


def render_scene(args):
    scene_name = args.scene_name
    task_name = args.task_name
    # cwd = os.getcwd()
    pcd_path = os.path.join("data", task_name, scene_name + ".ply")
    base_output_path = os.path.join("outputs", task_name)
    os.makedirs(base_output_path, exist_ok=True)
    # Add suffix based on render options (in alphabetical order for consistency)
    output_suffix_parts = []
    if args.render_graph is not None:
        output_suffix_parts.append("_graph")
    if args.render_seg is not None:
        output_suffix_parts.append("_seg")
    if args.render_feat is not None:
        output_suffix_parts.append("_feat")
    output_suffix = "".join(output_suffix_parts)
    output_path = os.path.join(base_output_path, f"{scene_name}{output_suffix}.png")

    # Configure mesh transform (tweak per scene)
    # METHOD
    # location = (2.35, -0.2, 1.48)
    # rotation = (-172, 174, 252)
    # BEV
    location = (2.2, 0.13, 1.3)
    rotation = (-155, 227, 333)
    scale = (0.05, 0.05, 0.05)
    if COSTUMIZED_POSE and "scene0081_00_vh_clean_2" in scene_name:
        location = (2.48, 0.2, 1.65)
        rotation = (-181, 186, 373)
    elif COSTUMIZED_POSE and "scene0144_00_vh_clean_2" in scene_name:
        location = (2.59, 0.1, 1.67)
        rotation = (-174, 227, 346)
    elif COSTUMIZED_POSE and "scene0131_00_vh_clean_2" in scene_name:
        location = (2.3, -0.24, 1.55)
        rotation = (-152, 164, 244)
    elif COSTUMIZED_POSE and "scene0046_00_vh_clean_2" in scene_name:
        location = (2.35, -0.07, 1.77)
        rotation = (-190, 163, 523)
    elif COSTUMIZED_POSE and "scene0100_00_vh_clean_2" in scene_name:
        location = (2.59, -0.14, 1.88)
        rotation = (190, 145, 210)
    elif COSTUMIZED_POSE and "scene0568_00_vh_clean_2" in scene_name:
        location = (2.35, -0.2, 1.48)
        rotation = (-172, 174, 252)

    # Set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
    cam_location = (3, 0, 2)
    look_at_location = (0, 0, 0.5)
    focal_length = 70  # (UI: click camera > Object Data > Focal Length)
    # Set light
    # light_angle = (6, -30, -155)
    light_angle = (63, 0, 90)
    strength = 2
    light_location = cam_location
    shadow_softness = 0.3

    # Initialize blender
    img_res_x = 1024
    img_res_y = 1024
    num_samples = 100
    exposure = 1.5
    bt.blenderInit(img_res_x, img_res_y, num_samples, exposure)

    # Load mesh and ensure vertex colors
    mesh = bt.readMesh(pcd_path, location, rotation, scale)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.shade_smooth()
    bt.recalculateNormals(mesh)

    color_layers = None
    if hasattr(mesh.data, "color_attributes"):
        color_layers = mesh.data.color_attributes
    elif hasattr(mesh.data, "vertex_colors"):
        color_layers = mesh.data.vertex_colors

    if color_layers is not None and len(color_layers) > 0:
        existing_names = [layer.name for layer in color_layers]
        if "Col" not in existing_names:
            color_layers[0].name = "Col"
    else:
        ply_data = plyfile.PlyData.read(pcd_path)
        colors = np.stack(
            (
                ply_data["vertex"]["red"],
                ply_data["vertex"]["green"],
                ply_data["vertex"]["blue"],
            ),
            axis=1,
        ).astype(np.float32) / 255.0
        bt.setMeshColors(mesh, colors, type="vertex")

    # Optional override: segmentation colors from scene_name_seg.ply
    if args.render_seg is not None:
        seg_path = find_auxiliary_file(scene_name, task_name, pcd_path, "seg", args.render_seg)
        if seg_path:
            print(f"[render_seg] Loading segmentation colors from: {seg_path}")
            seg_ply = plyfile.PlyData.read(seg_path)
            seg_colors = np.stack(
                (
                    seg_ply["vertex"]["red"],
                    seg_ply["vertex"]["green"],
                    seg_ply["vertex"]["blue"],
                ),
                axis=1,
            ).astype(np.float32) / 255.0
            mesh_vertex_count = len(mesh.data.vertices)
            seg_vertex_count = seg_colors.shape[0]
            print(f"[render_seg] Mesh vertices: {mesh_vertex_count}, Segmentation vertices: {seg_vertex_count}")
            if seg_vertex_count == mesh_vertex_count:
                # Remove existing 'Col' color layer if it exists to avoid conflicts
                # Try color_attributes first (Blender 2.92+)
                if hasattr(mesh.data, "color_attributes") and "Col" in mesh.data.color_attributes:
                    mesh.data.color_attributes.remove(mesh.data.color_attributes["Col"])
                # Fallback to vertex_colors (older Blender versions)
                elif hasattr(mesh.data, "vertex_colors") and "Col" in mesh.data.vertex_colors:
                    mesh.data.vertex_colors.remove(mesh.data.vertex_colors["Col"])
                # Apply segmentation colors
                bt.setMeshColors(mesh, seg_colors, type="vertex")
                # Force update the mesh
                mesh.data.update()
                print(f"[render_seg] Successfully applied segmentation colors to mesh")
            else:
                print(f"[render_seg] WARNING: Vertex count mismatch! Colors not applied.")
        else:
            base_name = scene_name.replace('.labels', '')
            print(f"[render_seg] WARNING: Segmentation file not found. Searched for: {base_name}_seg.ply")

    # Optional override: feature colors from scene_name_feature.pt
    if args.render_feat is not None:
        feat_path = find_auxiliary_file(scene_name, task_name, pcd_path, "feature", args.render_feat)
        if feat_path:
            print(f"[render_feat] Loading feature colors from: {feat_path}")
            try:
                feat_data = torch.load(feat_path, map_location='cpu')
                print(f"[render_feat] Loaded data type: {type(feat_data)}")
                
                # Handle different data formats
                if isinstance(feat_data, torch.Tensor):
                    features = feat_data.numpy()
                    print(f"[render_feat] Tensor shape: {features.shape}")
                elif isinstance(feat_data, dict):
                    print(f"[render_feat] Dict keys: {list(feat_data.keys())}")
                    # Try common keys
                    features = None
                    for key in ['features', 'feat', 'feature', 'data']:
                        if key in feat_data:
                            feat_tensor = feat_data[key]
                            if isinstance(feat_tensor, torch.Tensor):
                                features = feat_tensor.numpy()
                                print(f"[render_feat] Found feature data in key '{key}', shape: {features.shape}")
                                break
                    if features is None:
                        raise ValueError(f"Could not find feature data in {feat_path}. Available keys: {list(feat_data.keys())}")
                else:
                    features = np.asarray(feat_data)
                    print(f"[render_feat] Converted to array, shape: {features.shape}")
                
                # Ensure features is 2D array
                if features.ndim == 1:
                    features = features.reshape(-1, features.shape[0])
                    print(f"[render_feat] Reshaped 1D to 2D: {features.shape}")
                elif features.ndim > 2:
                    original_shape = features.shape
                    features = features.reshape(features.shape[0], -1)
                    print(f"[render_feat] Reshaped {original_shape} to {features.shape}")
                
                # Extract first 3 channels
                if features.shape[1] < 3:
                    raise ValueError(f"Feature dimension ({features.shape[1]}) is less than 3. Cannot extract RGB channels.")
                feat_rgb = features[:, :3].astype(np.float32)
                print(f"[render_feat] Extracted RGB channels, shape: {feat_rgb.shape}")
                print(f"[render_feat] RGB value range: min={feat_rgb.min(axis=0)}, max={feat_rgb.max(axis=0)}")
                
                # Normalize to [0, 1] range
                feat_min = feat_rgb.min(axis=0, keepdims=True)
                feat_max = feat_rgb.max(axis=0, keepdims=True)
                feat_range = feat_max - feat_min
                # Avoid division by zero
                feat_range = np.where(feat_range < 1e-8, 1.0, feat_range)
                feat_colors = (feat_rgb - feat_min) / feat_range
                feat_colors = np.clip(feat_colors, 0.0, 1.0)
                print(f"[render_feat] Normalized colors range: min={feat_colors.min(axis=0)}, max={feat_colors.max(axis=0)}")
                print(f"[render_feat] Sample colors (first 5): {feat_colors[:5]}")
                
                mesh_vertex_count = len(mesh.data.vertices)
                feat_vertex_count = feat_colors.shape[0]
                print(f"[render_feat] Mesh vertices: {mesh_vertex_count}, Feature vertices: {feat_vertex_count}")
                if feat_vertex_count == mesh_vertex_count:
                    # Remove existing 'Col' color layer if it exists to avoid conflicts
                    # Try color_attributes first (Blender 2.92+)
                    if hasattr(mesh.data, "color_attributes") and "Col" in mesh.data.color_attributes:
                        mesh.data.color_attributes.remove(mesh.data.color_attributes["Col"])
                        print(f"[render_feat] Removed existing color_attributes 'Col'")
                    # Fallback to vertex_colors (older Blender versions)
                    elif hasattr(mesh.data, "vertex_colors") and "Col" in mesh.data.vertex_colors:
                        mesh.data.vertex_colors.remove(mesh.data.vertex_colors["Col"])
                        print(f"[render_feat] Removed existing vertex_colors 'Col'")
                    # Apply feature colors
                    bt.setMeshColors(mesh, feat_colors, type="vertex")
                    # Force update the mesh
                    mesh.data.update()
                    print(f"[render_feat] Successfully applied feature colors to mesh")
                else:
                    print(f"[render_feat] WARNING: Vertex count mismatch! Colors not applied.")
            except Exception as e:
                print(f"[render_feat] ERROR: Failed to load or process feature file: {e}")
                import traceback
                traceback.print_exc()
        else:
            base_name = scene_name.replace('.labels', '')
            print(f"[render_feat] WARNING: Feature file not found. Searched for: {base_name}_feature.pt")

    mesh_vcolor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
    bt.setMat_VColor(mesh, mesh_vcolor)

    # Optional: overlay scene graph (.json or .pt)
    graph_path = args.graph
    if args.render_graph is not None:
        # If --render-graph is enabled, try to find graph file automatically
        # Support both --graph and --render-graph for specifying path
        custom_graph_path = args.render_graph if args.render_graph and args.render_graph != "" else None
        if custom_graph_path:
            # Use custom path from --render-graph
            graph_path = custom_graph_path
            if not os.path.exists(graph_path):
                print(f"[render-graph] Provided path not found: {graph_path}, trying auto-find...")
                graph_path = find_auxiliary_file(scene_name, task_name, pcd_path, "graph", None)
                if graph_path:
                    print(f"[render-graph] Using auto-found graph file: {graph_path}")
        elif graph_path is None:
            # Auto-find graph file
            graph_path = find_auxiliary_file(scene_name, task_name, pcd_path, "graph", None)
            if graph_path:
                print(f"[render-graph] Auto-found graph file: {graph_path}")
        elif not os.path.exists(graph_path):
            # If --graph path doesn't exist, try auto-find
            auto_path = find_auxiliary_file(scene_name, task_name, pcd_path, "graph", None)
            if auto_path:
                graph_path = auto_path
                print(f"[render-graph] Provided path not found, using auto-found: {graph_path}")
    
    # Only render graph if --render-graph is enabled or graph_path is explicitly provided
    if (args.render_graph is not None or args.graph is not None) and graph_path is not None and os.path.exists(graph_path):
        nodes = np.zeros((0,3), dtype=np.float32)
        edges = np.zeros((0,2), dtype=np.int64)
        node_colors = None
        edge_colors = None
        if graph_path.lower().endswith('.pt'):
            nodes, edges = load_graph_pt(graph_path)
        else:
            with open(graph_path, 'r') as f:
                G = json.load(f)
            nodes = np.asarray(G.get('nodes', []), dtype=np.float32)
            edges = np.asarray(G.get('edges', []), dtype=np.int64)
            node_colors = G.get('node_colors', None)
            if node_colors is not None:
                node_colors = np.asarray(node_colors, dtype=np.float32)
                if node_colors.shape[1] == 3:
                    node_colors = np.concatenate([node_colors, np.ones((node_colors.shape[0],1), dtype=np.float32)], axis=1)
            edge_colors = G.get('edge_colors', None)
            if edge_colors is not None:
                edge_colors = np.asarray(edge_colors, dtype=np.float32)
                if edge_colors.shape[1] == 3:
                    edge_colors = np.concatenate([edge_colors, np.ones((edge_colors.shape[0],1), dtype=np.float32)], axis=1)

        # coerce nodes shape to (N,3)
        if nodes.size > 0:
            nodes = np.asarray(nodes, dtype=np.float32)
            if nodes.ndim == 3 and nodes.shape[1] == 1 and nodes.shape[2] >= 3:
                nodes = nodes.reshape(nodes.shape[0], nodes.shape[2])
            if nodes.ndim == 2 and nodes.shape[1] > 3:
                nodes = nodes[:, :3]
            if nodes.ndim == 1 and nodes.size % 3 == 0:
                nodes = nodes.reshape(-1, 3)

        if nodes.size > 0 and edges.size > 0:
            nodes_world = transform_points_by_object(nodes, mesh) if args.graph_space == 'mesh' else nodes
            num_nodes = len(nodes_world)
            
            # Determine color palette
            default_palette = [
                (0.95,0.35,0.35,1.0), (0.95,0.70,0.25,1.0), (0.95,0.90,0.25,1.0),
                (0.40,0.80,0.40,1.0), (0.25,0.70,0.95,1.0), (0.55,0.45,0.95,1.0),
            ]
            
            # Generate colors from colormap if specified
            if args.node_colormap is not None:
                try:
                    colormap_palette = get_colormap_colors(args.node_colormap, num_nodes)
                    print(f"[node-colormap] Using colormap '{args.node_colormap}' for {num_nodes} nodes")
                except Exception as e:
                    print(f"[node-colormap] WARNING: Failed to use colormap '{args.node_colormap}': {e}")
                    print(f"[node-colormap] Falling back to default palette")
                    colormap_palette = None
            else:
                colormap_palette = None
            
            sf = float(np.mean(np.abs(np.array(mesh.scale))))
            r_node_w = float(args.node_radius) * sf if args.graph_space == 'mesh' else float(args.node_radius)

            # create/manage collection and root empty for scene graph
            graph_coll = ensure_collection(f"SceneGraph_{scene_name}")
            graph_root = ensure_empty(f"SceneGraph_{scene_name}_root", graph_coll)
            for i, p in enumerate(nodes_world):
                if node_colors is not None and i < len(node_colors):
                    # Use colors from JSON if provided
                    rgba = tuple(map(float, node_colors[i]))
                elif colormap_palette is not None:
                    # Use colormap colors
                    rgba = colormap_palette[i]
                else:
                    # Use default palette
                    rgba = default_palette[i % len(default_palette)]
                ptColor = bt.colorObj(rgba, 0.5, 1.0, 1.0, 0.0, 0.0)
                sphere = bt.drawSphere(r_node_w, ptColor, ptLoc=(float(p[0]), float(p[1]), float(p[2])))
                sphere.parent = graph_root
                link_to_collection(sphere, graph_coll)

            valid = (edges[:,0] >= 0) & (edges[:,0] < len(nodes_world)) & (edges[:,1] >= 0) & (edges[:,1] < len(nodes_world))
            edges = edges[valid]
            if len(edges) > 0:
                P1 = nodes_world[edges[:,0]]
                P2 = nodes_world[edges[:,1]]
                r_edge_w = float(args.edge_radius) * sf if args.graph_space == 'mesh' else float(args.edge_radius)
                draw_edges_with_emission(
                    P1, P2, r_edge_w,
                    colors_rgba=edge_colors,
                    emission_strength=float(args.edge_emission),
                    target_collection=graph_coll,
                    parent=graph_root,
                )
            if args.graph_join:
                join_collection_meshes(graph_coll, f"SceneGraph_{scene_name}_mesh", graph_root)
    elif args.render_graph is not None:
        # Graph file not found
        base_name = scene_name.replace('.labels', '')
        print(f"[render-graph] WARNING: Graph file not found. Searched for: {base_name}_graph.pt or {base_name}_graph.json")

    # Set invisible plane (shadow catcher)
    bt.invisibleGround(location=(0, 0, 0.2), shadowBrightness=0.9)

    cam = bt.setCamera(cam_location, look_at_location, focal_length)
    bt.setLight_sun(light_angle, strength, light_location, shadow_softness)

    # Set ambient light
    bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))

    # Set gray shadow to completely white with a threshold
    bt.shadowThreshold(alphaThreshold=0.05, interpolationMode="CARDINAL")

    # Save blender file so that you can adjust parameters in the UI
    bpy.ops.wm.save_mainfile(filepath=os.path.join(base_output_path, f"{scene_name}{output_suffix}.blend"))

    # Save rendering
    bt.renderImage(output_path, cam)


def main():
    args = parse_args()
    render_scene(args)


if __name__ == "__main__":
    main()
