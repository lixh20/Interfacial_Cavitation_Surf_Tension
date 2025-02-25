"""
Code for interfacial cavatiation problem with surface tension;

@author Xuanhe Li lixh20@mit.edu

Spring 2025
"""

# Fenics-related packages
from dolfin import *
# Numerical array package
import numpy as np
# Current time package
from datetime import datetime

# Set compiler parameters
parameters["form_compiler"]["quadrature_degree"] = 2
parameters["form_compiler"]["cpp_optimize"] = True


'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''
# Read mesh file (Run "CAVA_MESH.py" to generate the mesh at first)
mesh = Mesh()
with XDMFFile("meshes/CAVA_mesh.xdmf") as infile:
    infile.read(mesh)

# Extract initial mesh coords
x = SpatialCoordinate(mesh)

# Geometry parameters (need to be consistent with the setting in the generated mesh)
Le = 50.0             # Size of the computation domain (x direction)
H = 50.0             # Size of the computation domain (y direction)
a = 0.5              # Raidus of the defect region

# Identify  the boundary entities of the created mesh
tol=1e-4   # tolerance for identification of boundary
class Defect(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0,tol) and x[0] <= a and on_boundary
class Down(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0,tol) and x[0] >= a and on_boundary
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0,tol) and on_boundary    
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],Le,tol) and on_boundary
class Up(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],H,tol) and on_boundary

# Mark boundary subdomians
facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
DomainBoundary().mark(facets, 0)  # mark all boundaries with common index
# Mark sepcific boundaries
Defect().mark(facets, 1)
Down().mark(facets, 2)
Left().mark(facets,3)
Right().mark(facets,4)
Up().mark(facets, 5)

# Define the normal direction and the tangent direction of the boundary
Facet_normal=FacetNormal(mesh)
Facent_tangent = as_tensor([-Facet_normal[1],Facet_normal[0]])


'''''''''''''''''''''
Set up function spaces
'''''''''''''''''''''
# Define function space, both vectorial displacement and the scalar pressure
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # Quadratic interpolation for displacement
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # Linear interpolation for pressure

TH = MixedElement([U2, P1])  # Mixed element
ME = FunctionSpace(mesh, TH) # Total space for all DOFs

# Define test functions in weak form
w_test = TestFunction(ME)       # Test function
u_test, p_test = split(w_test)  # Split the test function

dw = TrialFunction(ME)          # Trial functions used for automatic differentiation                           

# Define actual functions with the required DOFs
w = Function(ME)
u, p = split(w)    # displacement, pressure

# Function spaces for visualization
W2 = FunctionSpace(mesh, U2) # Vector space for visulization  
W  = FunctionSpace(mesh,P1)  # Scalar space for visulization  


'''''''''''''''''''''
Subroutines for weak form
'''''''''''''''''''''
# Deformation gradient under axis-symmetric kinematic
def F_3D_calc(u):
    F = Identity(2)+grad(u)
    return as_tensor([[F[0,0], 0,F[0,1]],
                    [0, u[0]/x[0]+1, 0],
                    [F[1,0], 0, F[1,1]]])

# Calculate normal direction of the defect region in the deformed configuration
# using Nanson's formula
# output in the 2D geometry (axis symmetry setup)
def defect_normal_2D(u):
    n_R = as_tensor([0,0,-1])
    n = (inv(F_3D_calc(u))).T*n_R
    norm = (n[0]**2+n[1]**2+n[2]**2)**0.5
    return as_tensor([n[0]/norm,n[2]/norm])

# Calculate the tangent direction in the deformed configuration
def surf_tangent(u):
    F = Identity(2)+grad(u)
    return F*Facent_tangent


'''''''''''''''''''''
Construction of Weak form
'''''''''''''''''''''

# Material parameters
Gshear  = 1              # Shear modulus
Kbulk   = 1E12*Gshear    # Bulk modulus (choose a large number to ensure incompressibility)
gamma   = 1.0            # Surface tension (normalized)

# Kinematics
F_3D    = variable(F_3D_calc(u))            # Deformation gradient
C_3D    = variable(F_3D.T*F_3D)             # Right Cauchy tensor
Cdis_3D = variable(det(C_3D)**(-1/3)*C_3D)  # exclude volume change effect
Eg_3D   = variable(0.5*(C_3D-Identity(3)))  # Green strain tensor
# Invariants
I1 = tr(Cdis_3D)
J  = det(C_3D)**(0.5)

# Elastic free energy
psi = Gshear/2*(I1-3) +  p*(J-1) -(Kbulk/2.0)*p**2.0
psi_main =Gshear/2*(I1-3)

# Calculate second Piola stress using automatic differentiation
S = 2.0*diff(psi,C_3D)

#Caculate the  directional derivative  of the Green strain 
D_E = derivative(Eg_3D, u, u_test)


# Calculation of the deformed area of the defect region
Surf_2D = variable(sqrt(dot(surf_tangent(u),surf_tangent(u)))*(u[0]/x[0]+1))
# Use automatic differentiation to calculate the derivative of the deformed surface area
D_surf = derivative(Surf_2D,u,u_test)


# Specify the degree of Gauss integration
dsm= ds(metadata={"quadrature_degree": 2, "quadrature_scheme": "default"})
dxm = dx(metadata={"quadrature_degree": 2, "quadrature_scheme": "default"})

# Weak form (no body force and no traction bc conditions):
# First term: elastic contrubtion
# Second term: Incompressible condition
# Third term: surface tension on the defect region
L = inner(S,D_E)*x[0]*dxm + inner(((J-1.0) - p/Kbulk), p_test)*x[0]*dxm + gamma*D_surf*x[0]*dsm(1)
#  Calculate the Jacobian using automatic differentiation:
a = derivative(L, w, dw)


'''''''''''''''''''''''
BOUNDARY CONDITIONS
'''''''''''''''''''''''

# Loading condition (displacement control)
t    = 0.0        # start time
T    = 1000        # total simulation time 
dt   = 1.0        # (fixed) step size

# Cavity volume
V_tot  = 3     # normalized (V/d^3)
disp_tot = V_tot/(pi*Le**2)   # displacement of the top surface to match the cavity volume
# Define dispacement control at each time step
disp = Expression(("disp_tot*(t/T)"),
                    disp_tot = disp_tot,t = 0.0, T=T, degree=1)

# Boundary condition definitions
bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 2)  # u1 fix - bottom
bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 2)  # u2 fix - bottom
bcs_3 = DirichletBC(ME.sub(0).sub(0), 0, facets, 3)  # u1 fix  - left
bcs_4 = DirichletBC(ME.sub(0).sub(0), 0, facets, 4)  # u1 fix - right
bcs_5 = DirichletBC(ME.sub(0).sub(0), 0, facets, 5)  # u1 fix - top
bcs_6 = DirichletBC(ME.sub(0).sub(1), disp , facets, 5)  # u2 controlled - top

bcs = [bcs_1, bcs_2,bcs_3,bcs_4,bcs_5,bcs_6]



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Set up the non-linear variational problem and the solver
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''  

# Set up the  variational problem
stressProblem = NonlinearVariationalProblem(L, w, bcs, J=a)
 
# Set up the non-linear solver
solver  = NonlinearVariationalSolver(stressProblem)

# Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "lu" #'lu'   #'petsc'   #'gmres'
prm['newton_solver']['absolute_tolerance'] = 1.e-6
prm['newton_solver']['relative_tolerance'] = 1.e-6
prm['newton_solver']['maximum_iterations'] = 20
set_log_level(30)


'''''''''''''''''''''
Run the simulation
'''''''''''''''''''''
print("------------------------------------")
print("Simulation Start")
print("------------------------------------")

# First set up the file for writing results
#
file_results = XDMFFile("results/CAVA_result.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

#  Determine the start time for the analysis
#
startTime = datetime.now()

# Determine the number of time increments in the analysis
#
totSteps = int(np.floor(T/dt)) + 1

# Store result at each time step for output
Re = np.zeros([totSteps,5])

# A subroutine for writing results at time t to the XDMF file
def writeResults(t):
        # Need this to see deformation
        w.rename("disp", "Displacement")
        file_results.write(w.sub(0), t) 
        
        # Variable casting and renaming
        _w_1, _w_2 = w.split()            
        _w_1a,_w_1b = _w_1.split()
        
        # Project stress and pressure fields for visualization
        F_Vis = F_3D_calc(_w_1)
        T_Vis = (F_3D*S*F_3D.T)/det(F_Vis)
        sig11_Viz = project(T_Vis[0,0],W)
        sig11_Viz.rename("sig11","sig11")
        sig22_Viz = project(T_Vis[1,1],W)
        sig22_Viz.rename("sig22","sig22")       
        
        p_Vis = _w_2
        p_Viz = project(p_Vis,W)
        p_Viz.rename("pressure","Pressure")
        
        pre_Viz   = project(1/3*(T_Vis[0,0] + T_Vis[1,1]+T_Vis[2,2])+p_Vis,W)
        pre_Viz.rename("pressure_real","pressure_real")
        
        # Project effective stretch field for visualization
        C_Vis = F_Vis.T*F_Vis
        lambdaBar_Vis = sqrt(tr(C_Vis)/3.0)
        lambdaBar_Viz = project(lambdaBar_Vis,W)
        lambdaBar_Viz.rename("LambdaBar","Effective stretch.")
        
        # Calculate  and project J for visualzation
        detF_Vis =  det(F_Vis)
        detF_Viz = project(detF_Vis,W)
        detF_Viz.rename("J","J")

        # Write field quantities of interest
        file_results.write(p_Viz, t)
        file_results.write(sig11_Viz, t)
        file_results.write(sig22_Viz, t) 
        file_results.write(lambdaBar_Viz, t)
        file_results.write(detF_Viz, t)
        file_results.write(pre_Viz, t)


# Write the initial state to the results file
writeResults(t=0.0)

# Time-stepping solution procedure
ii =0 # step indicitor
while (t < T):

    # increment time
    t += dt
    ii += 1
    # update time variable in time-dependent displacement BC
    disp.t = t
    # store current displacement
    Re[ii,0] = disp_tot*(t/T)
    # Solve the problem
    (iter, converged) = solver.solve()
    # Store the results for visualization (every ten time steps)
    if ii%10 <0.1:
        writeResults(t)


    # Calculate the Force and the volume
    F        = F_3D_calc(u)                # deformation gradient
    P        = F*S                         # First piola stress        
    Sigma    = F*S*F.T                     # Cauchy stress
    Force    = 2*pi*P[2,2]*x[0]*dsm(5)     # Force applied on the top surface
    Volume   = pi*(x[0]+u[0])**2*F[2,0]*dsm(1)   # Volume of cavity
    Area     = 2*pi*Surf_2D*x[0]*dsm(1)          # Deformed area of the defection region
    Energy_e  = 2*pi*psi_main*x[0]*dxm           # Total elastic energy

    Re[ii,1] = assemble(Volume)      # store cavity volume
    Re[ii,2]  =assemble(Force)       # store applied force
    Re[ii,3] = assemble(Area)        # store deformed area
    Re[ii,4] = assemble(Energy_e)    # store total elastic energy

    
    # Print progress of calculation, newton iterations
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Step: Stretch   |   Simulation Time: {}    |     Wallclock Time: {}".format(t, current_time))
    print("Newton iterations: {}".format(iter))
    print()
    
# End analysis
print("End computation")     
# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("------------------------------------")


'''''''''''''''''''''
Data output
'''''''''''''''''''''
np.savetxt("RESULT_CAVA.csv",Re,delimiter=",")
