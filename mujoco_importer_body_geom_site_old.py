import bpy
import xml.etree.ElementTree as ET
import numpy as np
import os
import mujoco
from mathutils import Quaternion, Matrix, Vector, Euler
from tqdm import tqdm

def transform_matrix_from_pos_quat(pos, quat_wxyz):
    """Convert MuJoCo position and quaternion (w,x,y,z) to 4x4 transform matrix"""
    # Convert quaternion to rotation matrix
    w, x, y, z = quat_wxyz
    R = Matrix((
        (1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0),
        (2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x, 0),
        (2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y, 0),
        (0, 0, 0, 1)
    ))
    
    # Create translation matrix
    T = Matrix.Translation(pos)
    
    # Combine rotation and translation
    return T @ R

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Convert each line to a numpy array and stack them
    data = np.stack([np.fromstring(line.strip(), sep=',') for line in lines])
    return data

def import_mesh(filepath):
    """Import mesh file (STL or OBJ) based on file extension"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.stl':
        bpy.ops.wm.stl_import(filepath=filepath)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=filepath)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")
    return bpy.context.object

def create_path_curve(model_path):
    """
    Create a path curve with given points
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    
    # Create curve data
    for tendon_idx in range(model.ntendon):
        curve_data = bpy.data.curves.new(name=model.tendon(tendon_idx).name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 12
        curve_data.bevel_depth = 0.0001
        curve_obj = bpy.data.objects.new(model.tendon(tendon_idx).name, curve_data)
        bpy.context.scene.collection.objects.link(curve_obj)
        curve_data.splines.new(type='POLY')
    
    return curve_obj

def import_mujoco_xml(filepath, mode='body', site_radius=0):
    """
    Import MuJoCo XML file and import meshes
    mode: 'body' or 'geom' - determines whether to create parent body objects
    """
    # First handle any include tags by combining XML files
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Find mesh directory
    mesh_dir = ""
    for c in root.findall("compiler"):
        mesh_dir = c.get("meshdir", "")
    mesh_dir_full = os.path.join(os.path.dirname(filepath), mesh_dir)

    # Find all mesh files
    mesh_files = {}
    for a in root.findall("asset"):
        for m in a.findall("mesh"):
            filename = m.get("file", None)
            if filename is not None:
                # basename = os.path.basename(filename)
                mesh_path = os.path.join(mesh_dir_full, filename)
                if os.path.exists(mesh_path):
                    mesh_files[m.get('name')] = mesh_path
    
    # Define recursive function to process bodies and import meshes
    def process_body(body, parent_transform=Matrix.Identity(4)):
        # Get body position and orientation
        body_pos = np.array([float(x) for x in body.get("pos", "0 0 0").split()])
        
        # Handle either euler or quaternion orientation
        if "euler" in body.attrib:
            # Convert euler angles (in radians) to quaternion
            euler = np.array([float(x) for x in body.get("euler").split()])
            # Create Euler object and convert to quaternion
            euler_obj = Euler(euler, 'ZYX') 
            body_wxyz = np.array(euler_obj.to_quaternion())
        else:
            # Use quaternion directly if specified
            body_wxyz = np.array([float(x) for x in body.get("quat", "1 0 0 0").split()])
            
        # Calculate body transform relative to parent
        local_transform = transform_matrix_from_pos_quat(body_pos, body_wxyz)
        world_transform = parent_transform @ local_transform
        
        # Create empty object for the body only in body mode
        body_empty = None
        if mode == 'body':
            body_name = body.get("name", "unnamed_body")
            body_empty = bpy.data.objects.new(body_name, None)
            bpy.context.scene.collection.objects.link(body_empty)
            body_empty.empty_display_type = 'ARROWS'
            body_empty.matrix_world = world_transform

        # Import meshes for this body
        for geom in body.findall("geom"):
            geom_type = geom.get("type", "unknown")
            if geom_type == "mesh":
                geom_meshfile = geom.get("mesh", None)
                if geom_meshfile is not None:
                    mesh_path = mesh_files.get(os.path.splitext(geom_meshfile)[0])
                    if mesh_path and os.path.exists(mesh_path):
                        # Import mesh
                        imported_mesh = import_mesh(mesh_path)
                        imported_mesh.name = geom.get("name", "unnamed_geom")
                        
                        # get geom position and orientation
                        geom_pos = np.array([float(x) for x in geom.get("pos", "0 0 0").split()])
                        if "euler" in geom.attrib:
                            euler = np.array([float(x) for x in geom.get("euler").split()])
                            euler_obj = Euler(euler, 'ZYX') 
                            geom_wxyz = np.array(euler_obj.to_quaternion())
                        else:
                            geom_wxyz = np.array([float(x) for x in geom.get("quat", "1 0 0 0").split()])
                        # Set world transform
                        geom_transform = transform_matrix_from_pos_quat(geom_pos, geom_wxyz)
                        
                        if mode == 'body':
                            # Parent the geom to the body empty
                            imported_mesh.parent = body_empty
                        imported_mesh.matrix_world = world_transform @ geom_transform
        # Add sites
        if site_radius>0:
            for site in body.findall("site"):
                site_name = site.get("name", "unnamed_site")
                site_pos = np.array([float(x) for x in site.get("pos", "0 0 0").split()])
                site_quat = np.array([float(x) for x in site.get("quat", "1 0 0 0").split()])
                site_transform = transform_matrix_from_pos_quat(site_pos, site_quat)
                
                # Create UV sphere
                bpy.ops.object.metaball_add(type='BALL', radius=0.01)
                # bpy.ops.mesh.primitive_uv_sphere_add(segments=2, ring_count=2, radius=0.01, calc_uvs=False)
                uv_sphere_obj = bpy.context.active_object
                uv_sphere_obj.name = site_name
                
                # Set transform
                if mode == 'body':
                    uv_sphere_obj.parent = body_empty
                uv_sphere_obj.matrix_world = world_transform @ site_transform
                
        # Process child bodies
        for child_body in body.findall("body"):
            process_body(child_body, world_transform)

    # Process all bodies starting from worldbody
    for worldbody in root.findall("worldbody"):
        process_body(worldbody)

def import_animation(model_path, qpos_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    qpos = read_txt_file(qpos_path)
    print(qpos)
    time_step = 1  # Number of frames to skip between keyframes
    
    for time_idx in tqdm(range(0, qpos.shape[0], time_step), desc="Processing timesteps"):
        data.qpos = qpos[time_idx, :]
        mujoco.mj_fwdPosition(model, data)
        
        # Set keyframes for each body
        for body_idx in range(model.nbody):
            body_name = data.body(body_idx).name
            if body_name not in bpy.data.objects:
                continue
                
            blender_obj = bpy.data.objects[body_name]
            
            # Set position keyframe
            blender_obj.location = data.body(body_idx).xpos
            blender_obj.keyframe_insert("location", frame=time_idx//time_step)
            
            # Set rotation keyframe
            blender_obj.rotation_mode = "QUATERNION"
            # MuJoCo uses wxyz quaternions, convert to Blender's wxyz format
            quat = data.body(body_idx).xquat
            blender_obj.rotation_quaternion = Quaternion((quat[0], quat[1], quat[2], quat[3]))
            blender_obj.keyframe_insert("rotation_quaternion", frame=time_idx//time_step)
        
    # Find max number of waypoints for each tendon across timesteps
    max_waypoints_per_tendon = np.zeros(model.ntendon, dtype=int)
    for time_idx in tqdm(range(0, qpos.shape[0], time_step), desc="Finding max waypoints"):
        data.qpos = qpos[time_idx, :]
        mujoco.mj_fwdPosition(model, data)
        max_waypoints_per_tendon = np.maximum(max_waypoints_per_tendon, data.ten_wrapnum)

    # Initialize curves with max number of points
    for tendon_idx in range(model.ntendon):
        curve_data = bpy.data.curves.new(name=model.tendon(tendon_idx).name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 12
        curve_data.bevel_depth = 0.002
        curve_obj = bpy.data.objects.new(model.tendon(tendon_idx).name, curve_data)
        bpy.context.scene.collection.objects.link(curve_obj)
        curve_data.splines.new(type='POLY')
        curve_obj.data.splines[0].points.add(max_waypoints_per_tendon[tendon_idx] - 1)
            
    for time_idx in tqdm(range(0, qpos.shape[0], time_step), desc="Processing timesteps"):
        data.qpos = qpos[time_idx, :]
        mujoco.mj_fwdPosition(model, data)
        
        ten_wrapadr = data.ten_wrapadr
        ten_wrapnum = data.ten_wrapnum
        wrap_xpos = data.wrap_xpos.reshape(-1,3)
        for tendon_idx in range(model.ntendon):
            start = ten_wrapadr[tendon_idx]
            end = start + ten_wrapnum[tendon_idx]
            waypoint_xpos = wrap_xpos[start:end,:]
            curve_name = model.tendon(tendon_idx).name
            curve_obj = bpy.data.objects[curve_name].data.splines[0]
            
            for curve_point_idx in range(len(curve_obj.points)):
                if curve_point_idx < waypoint_xpos.shape[0]:
                    if np.allclose(waypoint_xpos[curve_point_idx], [0,0,0]):
                        # Find nearest non-zero point to curve_point_idx
                        min_dist = float('inf')
                        nearest_point = None
                        for way_point_idx in range(waypoint_xpos.shape[0]):
                            if not np.allclose(waypoint_xpos[way_point_idx], [0,0,0]):
                                dist = abs(way_point_idx - curve_point_idx)
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_point = waypoint_xpos[way_point_idx]
                        if nearest_point is not None:
                            waypoint_xpos[curve_point_idx] = nearest_point
                    x, y, z = waypoint_xpos[curve_point_idx]
                    curve_obj.points[curve_point_idx].co = (x, y, z, 1)
                    curve_obj.points[curve_point_idx].keyframe_insert("co", frame=time_idx//time_step)
                else:
                    if np.allclose(waypoint_xpos[-1], [0,0,0]):
                        # Find nearest non-zero point to curve_point_idx
                        min_dist = float('inf')
                        nearest_point = None
                        for way_point_idx in range(waypoint_xpos.shape[0]):
                            if not np.allclose(waypoint_xpos[way_point_idx], [0,0,0]):
                                dist = abs(way_point_idx - curve_point_idx)
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_point = waypoint_xpos[way_point_idx]
                        if nearest_point is not None:
                            waypoint_xpos[curve_point_idx] = nearest_point
                    curve_obj.points[curve_point_idx].co = np.concatenate([waypoint_xpos[-1], [1]])
                    curve_obj.points[curve_point_idx].keyframe_insert("co", frame=time_idx//time_step)
                    
        

if True:
    # xml_file_path = "/home/mct/Desktop/Projects/blender_mujoco/MS_Human_700_Release/mjmodel.xml"
    xml_file_path = "/home/micha/Desktop/mj_blender/mjmodel.xml"
    import_mujoco_xml(xml_file_path, mode='body', site_radius=0)  # Creates hierarchy with body empties

    # qpos_path = "/home/mct/Desktop/Projects/blender_mujoco/extracted_qpos_sequences.txt"
    qpos_path = "/home/micha/Desktop/mj_blender/test.txt"
    # create_path_curve(model_path)
    import_animation(xml_file_path, qpos_path)
    # qpos = np.loadtxt(qpos_path)
    
