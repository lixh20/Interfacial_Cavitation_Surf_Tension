"""
Mesh generation for interfacial cavatiation problem with surface tension;

@author Xuanhe Li lixh20@mit.edu

Spring 2025
"""


"""
Create the mesh with pygmsh 
"""

import pygmsh

# Geometry parameters
L = 50.0             # Size of the computation domain (x direction)
H = 50.0             # Size of the computation domain (y direction)
a = 0.5              # Raidus of the defect region
b = 0.1              # Size of the refined mesh region
resolution_fine = a/1000     # mesh resolution on the right side of defect edge
resolution_edge = 0.01       # mesh resolution on the left side of defect edge
resolution_mid = 0.05        # mesh resolution in the refined mesh domain
resolution_row = 5           # mesh resolution in the far field

# Create an empty geometry
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# Add points and specify the local mesh resolution
points = [
          model.add_point((0,H), mesh_size=resolution_row),
          model.add_point((0,b), mesh_size=resolution_mid),
          model.add_point((a,0), mesh_size=resolution_fine),
          model.add_point((L,0), mesh_size=resolution_row),
          model.add_point((L,H), mesh_size=resolution_row),
          model.add_point((0,0), mesh_size=resolution_edge)          
          ]

# Add lines
line1  = model.add_line(points[0], points[1])
line2  = model.add_line(points[2], points[3])
line3  = model.add_line(points[3], points[4])
line4  = model.add_line(points[4], points[0])
line5  = model.add_line(points[5],points[2])
line6  = model.add_line(points[1],points[5])

# Add imaginary boundary for the refined mesh domain near the defect
circle = model.add_ellipse_arc(points[1], points[5], points[2],points[2])

# Create the mesh domain
plate_whole = model.add_curve_loop([line5,line2,line3,line4,line1,line6])
plane_surface = model.add_plane_surface(plate_whole)
# Introduce the refined mesh domain boundary to the mesh
model.in_surface(circle,plane_surface)

# Call gmsh before adding physical entities
model.synchronize()
model.add_physical(plane_surface, "surface")


"""
Write the mesh to file using gmsh
"""
import gmsh
geometry.generate_mesh(dim=2)
gmsh.write("meshes/CAVA_mesh.msh")
gmsh.clear()
geometry.__exit__()


"""
Convert the mesh to.xdmf  format using meshio
"""
import meshio
mesh_from_file = meshio.read("meshes/CAVA_mesh.msh")


"""
Extract cells and boundary data and save in 'XDMF format'
"""
import numpy
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells},\
                           cell_data={"name_to_read":[cell_data]})
    return out_mesh


volume_mesh = create_mesh(mesh_from_file, "triangle",True)
meshio.write("meshes/CAVA_mesh.xdmf", volume_mesh)