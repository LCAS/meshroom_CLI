import open3d as o3d
import numpy as np
from PIL import Image
import os

def convert_ply_to_obj_with_texture(input_ply, output_obj, output_mtl, output_texture):
    # Load the PLY file
    ply_mesh = o3d.io.read_triangle_mesh(input_ply)
    if not ply_mesh.has_vertex_colors() or not ply_mesh.has_triangles():
        raise ValueError("The PLY file must contain vertex colors and faces.")

    # Extract vertex positions and colors
    vertices = np.asarray(ply_mesh.vertices)
    colors = (np.asarray(ply_mesh.vertex_colors) * 255).astype(np.uint8)
    triangles = np.asarray(ply_mesh.triangles)
    
    # Create a simple texture from vertex colors (UV mapping one color per vertex)
    texture_size = 4096  # Example texture size
    texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    step = max(1, len(colors) // (texture_size ** 2))
    
    for i, color in enumerate(colors[::step]):
        x = (i % texture_size)
        y = (i // texture_size) % texture_size
        texture[y, x] = color

    # Save the texture as an image
    Image.fromarray(texture).save(output_texture)

    # Write the OBJ file
    with open(output_obj, 'w') as obj_file:
        obj_file.write(f"mtllib {os.path.basename(output_mtl)}\n")
        for v, color in zip(vertices, colors):
            obj_file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for v, color in zip(vertices, colors):
            obj_file.write(f"vt {v[0] % 1} {v[1] % 1}\n") 
        for tri in triangles:
            obj_file.write(f"f {tri[0] + 1}/{tri[0] + 1} {tri[1] + 1}/{tri[1] + 1} {tri[2] + 1}/{tri[2] + 1}\n")

    material_name = "material_0000"
    # Write the MTL file
    with open(output_mtl, 'w') as mtl:
        mtl.write(f"newmtl {material_name}\n")
        mtl.write("Ka 1.000000 1.000000 1.000000\n")
        mtl.write("Kd 1.000000 1.000000 1.000000\n")
        mtl.write("Ks 0.000000 0.000000 0.000000\n")
        mtl.write("d 1.0\n")
        mtl.write("illum 2\n")
        mtl.write(f"map_Kd {os.path.basename(output_texture)}\n")


