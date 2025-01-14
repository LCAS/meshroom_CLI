import sys
import os
import shutil
import open3d as o3d
import numpy as np

import json

import time

import trimesh

import random

import trimesh.exchange
import trimesh.exchange.export
import utils

def convert_to_dae(input_mesh, output_mesh):
    """
    Converts the input mesh file to a .dae (COLLADA) format using trimesh.
    
    Parameters:
    - input_mesh (str): Path to the input mesh file.
    - output_mesh (str): Path to save the converted .dae file.
    
    Raises:
    - FileNotFoundError: If the input file does not exist.
    - ValueError: If the output file extension is not '.dae'.
    - Exception: For other unexpected errors during the conversion process.
    """
    try:
        # Validate input mesh file
        if not os.path.isfile(input_mesh):
            raise FileNotFoundError(f"Input mesh file does not exist: {input_mesh}")
        
        # Validate output file extension
        if not output_mesh.lower().endswith('.dae'):
            raise ValueError(f"Output file must have a '.dae' extension: {output_mesh}")
        
        # Load the mesh using trimesh
        print(f"Loading input mesh from: {input_mesh}")
        mesh = trimesh.load(input_mesh)
        if mesh.is_empty:
            print("Error: The mesh is empty!")
        else:
            print("Mesh successfully loaded.")
        
        # Export the mesh to .dae
        print(f"Exporting mesh to: {output_mesh}")
        mesh.export(output_mesh)
        print(f"Mesh successfully converted and saved to: {output_mesh}")
    
    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except ValueError as val_error:
        print(f"Error: {val_error}")
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")

def convert_mesh_to_format(input_mesh, output_mesh, output_format="dae"):
    """
    Converts the mesh into a specified format.
    """
    if not os.path.exists(input_mesh):
        raise FileNotFoundError(f"Input mesh file not found: {input_mesh}")
    
    if output_format == "dae":
        convert_to_dae(input_mesh, output_mesh)
    elif output_format == "stl":
        mesh = o3d.io.read_triangle_mesh(input_mesh)
        print(f"Input mesh {input_mesh} is read as {mesh}")
        mesh.compute_vertex_normals()
        print(f"Mesh vertex computed")
        o3d.io.write_triangle_mesh(output_mesh, mesh)
    else:
        raise ValueError(f"Unsupported format: {output_format}")
    
    print(f"Mesh converted to {output_format}: {output_mesh}")

def generate_gazebo_model(output_dir, mesh_file, texture_file, texture_img, model_name="generated_model", output_format="dae"):
    """
    Creates a Gazebo model folder with the required structure and files.
    """
    model_dir = os.path.join(output_dir, model_name)
    mesh_dir = os.path.join(model_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    material_dir = os.path.join(model_dir, "materials")
    os.makedirs(material_dir, exist_ok=True)

    texture_dir = os.path.join(material_dir, "textures")
    os.makedirs(texture_dir, exist_ok=True)

    script_dir = os.path.join(material_dir, "scripts")
    os.makedirs(script_dir, exist_ok=True)

    # Copy the mesh file to the meshes directory
    mesh_filename = os.path.basename(mesh_file)
    shutil.copy(mesh_file, os.path.join(mesh_dir, mesh_filename))

    texture_filename = os.path.basename(texture_file)
    shutil.copy(texture_file, os.path.join(texture_dir, texture_filename))

    texture_imgname = os.path.basename(texture_img)
    shutil.copy(texture_img, os.path.join(texture_dir, texture_imgname))

    material_name = "material_1001"

    # Create model.config
    config_content = f"""<?xml version="1.0"?>
<model>
    <name>{model_name}</name>
    <version>1.0</version>
    <sdf version="1.6">model.sdf</sdf>
    <author>
        <name>Abdurrahman Yilmaz</name>
        <email>ayilmaz@lincoln.ac.uk</email>
    </author>
    <description>A model generated from a 3D mesh</description>
</model>
"""
    with open(os.path.join(model_dir, "model.config"), "w") as f:
        f.write(config_content)

    # Create model.sdf
    if output_format == "dae":
        sdf_content = f"""<?xml version="1.0" ?>
<sdf version="1.6">
    <model name="{model_name}">
        <static>true</static>
        <link name="link">
            <visual name="visual">
                <geometry>
                    <mesh>
                        <uri>model://{model_name}/meshes/{mesh_filename}</uri>
                    </mesh>
                </geometry>
                <material>
                    <script>
                        <uri>model://{model_name}/materials/scripts/</uri>
		                <uri>model://{model_name}/materials/textures/</uri>
		                <name>{material_name}</name>
                    </script>
                </material>
            </visual>
             <collision name="collision">
                <geometry>
                    <mesh>
                        <uri>model://{model_name}/meshes/{mesh_filename}</uri>
                    </mesh>
                </geometry>
            </collision>
        </link>
    </model>
</sdf>"""
    elif output_format == "stl":
        sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.6">
    <model name="{model_name}">
        <static>true</static>
        <link name="link">
            <visual name="visual">
                <geometry>
                    <mesh>
                        <uri>model://{model_name}/meshes/{mesh_filename}</uri>
                    </mesh>
                </geometry>
            </visual>
            <collision name="collision">
                <geometry>
                    <mesh>
                        <uri>model://{model_name}/meshes/{mesh_filename}</uri>
                    </mesh>
                </geometry>
            </collision>
        </link>
    </model>
</sdf>
"""
    with open(os.path.join(model_dir, "model.sdf"), "w") as f:
        f.write(sdf_content)
    
    # Create texturedMesh.material file
    material_content = f"""material {material_name}
{{
  technique
  {{
    pass
    {{
      texture_unit
      {{
        texture {texture_imgname}
      }}
    }}
  }}
}}
"""
    with open(os.path.join(script_dir, "texturedMesh.material"), "w") as f:
        f.write(material_content)

    print(f"Gazebo model generated at: {model_dir}")

def crop_object(input_obj):

    mesh = trimesh.load(input_obj)

    # Get the bounding box of the mesh
    bounds_min, bounds_max = mesh.bounds  # bounds is a tuple: (min_point, max_point)

    # Extract min and max values for x, y, and z axes
    min_x, min_y, min_z = bounds_min
    max_x, max_y, max_z = bounds_max

    # Choose a crop plane, for example, cropping at a point in the x direction
    crop_x_position = (min_x + max_x) / 2  # Example: crop at the center in x direction

    # Define the plane for cropping
    plane_origin = [crop_x_position, 0, 0]  # Change y, z if cropping along other axes
    plane_normal = [1, 0, 0]  # Normal vector for a vertical plane cutting in x-direction

    # Perform the crop
    cropped_obj = mesh.slice_plane(plane_origin=plane_origin, plane_normal=plane_normal)
    
    return cropped_obj

import open3d as o3d
import numpy as np
import json

def mesh_crop(mesh):
    """
    Crop a mesh object based on user-selected points.
    """
    print('mesh_crop: Cropping the mesh based on selected plot area.')
    
    # Visualize mesh and pick points
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window('Pick Points on Mesh to Define Crop Area')
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()
    points_list = vis.get_picked_points()
    
    if not points_list:
        print("No points selected, returning the original mesh.")
        return mesh
    
    vertices = np.asarray(mesh.vertices)
    bound_box = vertices[points_list].tolist()
    print(f"Selected bounding polygon: {bound_box}")

    # Define cropping parameters
    x_max, y_max, z_max = mesh.get_max_bound()
    x_min, y_min, z_min = mesh.get_min_bound()
    
    dictionary = {
        "axis_max": z_max,
        "axis_min": z_min,
        "bounding_polygon": bound_box,
        "class_name": "SelectionPolygonVolume",
        "orthogonal_axis": "Z",
        "version_major": 1,
        "version_minor": 0
    }
    
    # Write JSON configuration
    json_object = json.dumps(dictionary, indent=4)
    with open("mesh_crop_sample.json", "w") as outfile:
        outfile.write(json_object)
    
    # Load selection volume from JSON
    vol = o3d.visualization.read_selection_polygon_volume('mesh_crop_sample.json')
    
    # Crop the mesh (vol.crop_triangle_mesh method)
    cropped_mesh = vol.crop_triangle_mesh(mesh)
    
    # Visualize the cropped mesh
    o3d.visualization.draw_geometries([cropped_mesh])
    
    return cropped_mesh

def pcd_crop(pcd):

    print('pcd_crop: croping the pcd plot area')
    # Visualize cloud and pick 4 points
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window('Pick 4 Points on Quadrant to Crop Plot Area')
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    points_list = vis.get_picked_points()
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    x_max, y_max, z_max = pcd.get_max_bound()
    x_min, y_min, z_min = pcd.get_min_bound()
    bound_box = points[points_list].tolist()
    print(bound_box)
    #Wrtie json file
    dictionary = {
        "axis_max" : z_max,
        "axis_min" : z_min,
        "bounding_polygon" : bound_box,
        "class_name" : "SelectionPolygonVolume",
        "orthogonal_axis" : "Z",
        "version_major" : 1,
        "version_minor" : 0
    } 
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)
    
    vol = o3d.visualization.read_selection_polygon_volume('sample.json')
    #print(vol)
    #print(vol.bounding_polygon)
    crop_pcd = vol.crop_point_cloud(pcd)
    #print(crop_pcd)
    o3d.visualization.draw_geometries([crop_pcd])
    
    return crop_pcd

def mesh_to_point_cloud(mesh, density=100000):
    """
    Convert a mesh to a sampled point cloud for interactive selection.

    Parameters:
    - mesh: The input mesh (open3d.geometry.TriangleMesh).
    - density: The number of points to sample from the mesh (default is 100,000).

    Returns:
    - point_cloud: A PointCloud object with sampled points from the mesh surface.
    """
    print(f"Sampling {density} points from mesh using Poisson Disk sampling.")
    
    # Sample points from the mesh surface
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=density)
    
    # Ensure color is assigned to the points (here we use white for visibility)
    colors = np.zeros((len(point_cloud.points), 3))  # RGB = [0, 0, 0] (white)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return point_cloud


def crop_mesh_with_selected_points(mesh, selected_points):
    """
    Crop the mesh using selected points as a bounding box.
    """
    min_bound = np.min(selected_points, axis=0)
    max_bound = np.max(selected_points, axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    cropped_mesh = mesh.crop(bbox)
    return cropped_mesh


def sample_random_points_in_volume(min_bound, max_bound, num_points=1000):
    """
    Sample random points uniformly in the bounding volume defined by min_bound and max_bound.
    
    Parameters:
    - min_bound: The minimum bounding coordinates (3D point).
    - max_bound: The maximum bounding coordinates (3D point).
    - num_points: The number of random points to sample (default is 1000).

    Returns:
    - sampled_points: List of sampled random points inside the bounding box.
    """
    sampled_points = []
    for _ in range(num_points):
        random_point = [
            random.uniform(min_bound[i], max_bound[i]) for i in range(3)
        ]
        sampled_points.append(random_point)
    
    return np.array(sampled_points)


def interactive_mesh_crop(mesh):
    """
    Allow vertex-based cropping by selecting points on a point cloud derived from the mesh,
    along with sampling random points in the bounding volume.
    """
    # Get the point cloud from the mesh
    point_cloud = mesh_to_point_cloud(mesh)
    
    # Get the bounding box of the mesh to sample random points within it
    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()

    # Sample random points within the bounding volume of the mesh
    num_random_points = 1000  # You can adjust this number
    random_points = sample_random_points_in_volume(min_bound, max_bound, num_random_points)

    # Convert the random points to a point cloud
    random_pc = o3d.geometry.PointCloud()
    random_pc.points = o3d.utility.Vector3dVector(random_points)
    
    # Color the random points (e.g., red for visibility)
    random_colors = np.zeros((len(random_points), 3))  # Black (RGB = [0, 0, 0])
    random_colors[:, 0] = 1  # Set the red channel to 1 (making the points red)
    random_pc.colors = o3d.utility.Vector3dVector(random_colors)

    # Combine point clouds and colors into one
    combined_points = np.concatenate([np.asarray(point_cloud.points), np.asarray(random_pc.points)], axis=0)
    combined_colors = np.concatenate([np.asarray(point_cloud.colors), np.asarray(random_pc.colors)], axis=0)

    # Create a new point cloud for the combined points and colors
    combined_pc = o3d.geometry.PointCloud()
    combined_pc.points = o3d.utility.Vector3dVector(combined_points)
    combined_pc.colors = o3d.utility.Vector3dVector(combined_colors)

    # Visualize and pick points
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window('Pick Points on Point Cloud to Define Crop Area')
    vis.add_geometry(combined_pc)
    vis.run()
    vis.destroy_window()

    # Get the picked points from the visualization window
    picked_indices = vis.get_picked_points()
    if not picked_indices:
        print("No points selected. Returning original mesh.")
        return mesh

    vertices = np.asarray(mesh.vertices)
    selected_points = vertices[picked_indices]
    print(f"Selected points: {selected_points}")
    
    # Crop the mesh using the selected points
    cropped_mesh = crop_mesh_with_selected_points(mesh, selected_points)
    
    # Visualize the cropped mesh
    o3d.visualization.draw_geometries([cropped_mesh])
    
    return cropped_mesh

def gazebo_model(base_dir, mesh_dir, output_dir, output_format="dae", model_name="reconstructed_model"):
    """
    Final step to convert the mesh to Gazebo format and generate the model files.
    """
    print(f"baseDir: {base_dir}")
    print(f"output_dir: {output_dir}")
    print(f"output_format: {output_format}")
    print(f"model_name: {model_name}")
    input_mesh = os.path.join(base_dir, mesh_dir, "mesh.obj")  # Path to the output mesh
    #input_mesh = os.path.join(base_dir, mesh_dir, "texturedMesh.obj")  # Path to the output mesh
    input_texture = os.path.join(base_dir, mesh_dir, "mesh.mtl")
    #input_texture = os.path.join(base_dir, mesh_dir, "texturedMesh.mtl")
    input_texture_img = os.path.join(base_dir, mesh_dir, "texture_1001.png")
    print(f"Input Mesh object is located at {input_mesh}")
    converted_mesh = os.path.join(base_dir, f"gazebo_mesh.{output_format}") # Desired mesh format
    print(f"Converted mesh will be saved to {converted_mesh}")

    input_ply_mesh = os.path.join(base_dir, mesh_dir, "Tomato01_polytunnel_full_mesh.ply") 
    output_obj = os.path.join(base_dir, mesh_dir, "mesh.obj")
    output_mtl = os.path.join(base_dir, mesh_dir, "mesh.mtl")
    output_texture = os.path.join(base_dir, mesh_dir, "material0000.png") 

    # Example usage
    utils.convert_ply_to_obj_with_texture(
        input_ply=input_ply_mesh,
        output_obj=output_obj,
        output_mtl=output_mtl,
        output_texture=output_texture
    )
    print(f"Input PLY Mesh converted to object and texture")

    '''print(f"Mesh crop started")
    mesh = o3d.io.read_triangle_mesh(input_mesh)
    print(f"Original mesh vertices count: {len(np.asarray(mesh.vertices))}")
    cropped_mesh = interactive_mesh_crop(mesh)
    #o3d.io.write_triangle_mesh("cropped_mesh.obj", cropped_mesh, write_triangle_uvs=True)
    o3d.io.write_triangle_mesh(
        "cropped_mesh.obj", 
        cropped_mesh, 
        write_vertex_normals=True, 
        write_vertex_colors=True, 
        write_triangle_uvs=True
    )
    saved_mesh = o3d.io.read_triangle_mesh("cropped_mesh.obj")
    print(f"Saved mesh vertices count: {len(np.asarray(saved_mesh.vertices))}")
    print(f"Mesh crop ended") 

    print(f"PC crop started")
    input_pc = os.path.join(base_dir, mesh_dir, "densePointCloud.ply")
    pcd = o3d.io.read_point_cloud(input_pc)
    cropped_pcd = pcd_crop(pcd)
    o3d.io.write_point_cloud("densePointCloud_crop.ply", cropped_pcd)
    print(f"PC crop ended")

    print(f"Mesh crop started")
    mesh = o3d.io.read_triangle_mesh(input_mesh)
    cropped_mesh = mesh_crop(mesh)
    o3d.io.write_triangle_mesh("cropped_mesh.obj", cropped_mesh)
    print(f"Mesh crop ended")'''

    ## Convert the mesh
    #convert_mesh_to_format(input_mesh, converted_mesh, output_format)

    ## Generate Gazebo model
    #generate_gazebo_model(output_dir, converted_mesh, input_texture, input_texture_img, model_name, output_format)

def main():

    #mesh_dir = "10_MeshFiltering"
    mesh_dir = "13_Texturing"

    # Pass the arguments of the function as parameters in the command line code
    baseDir = sys.argv[1]               ##  --> name of the Folder containing the process (a new folder will be created)
    output_dir = sys.argv[2]            ##  --> path of the folder where the gazebo model to be saved
    if len(sys.argv) > 2:
        output_format = sys.argv[3]         ##  --> Output Gazebo model format
        if len(sys.argv) > 3:
            model_name = sys.argv[4]            ##  --> Output Gazebo model name    

    try:

        startTime = time.time()
        gazebo_model(baseDir, mesh_dir, output_dir, output_format, model_name)
        endTime = time.time()
        hours, rem = divmod(endTime-startTime, 3600)
        minutes, seconds = divmod(rem, 60)
        print("time elapsed: "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")

if __name__ == "__main__":
    main()