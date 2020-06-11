# Copyright (C) 2019 Stéphane T. Urcun
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

from fenics import *
from mshr import *
from ufl import nabla_div

##############################
# Terzaghi's 1D problem
##############################

T = 5.1282                  #final time
num_steps =1000        #number of time steps
dt = T / (1.0*num_steps) #time step size
rho_l = 1000
E=5000 #Pa
nu=0.4
k = 1.82e-15 	#intrinsic permeability
mu_l = 1e-2 	#dynamic viscosity of IF

lambda_=(E*nu)/((1+nu)*(1-2*nu))
mu =E/(2*(1+nu))
poro=0.2
Kf=2.2e9	#fluid compressibility
Ks=1e10		#solid compressibility
S=(poro/Kf)+(1-poro)/Ks

# Create mesh and define expression
mesh=RectangleMesh(Point(0.0,0.0), Point(1e-5, 1e-4),8,20,'crossed')
top =  CompiledSubDomain("near(x[1], side)", side = 1e-4)
bottom =  CompiledSubDomain("near(x[1], side)", side = 0.0)
left_right = CompiledSubDomain("(near(x[0], 0.0) )|| (near(x[0], 1e-5) )")

boundary = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
boundary.set_all(1)
top.mark(boundary, 2)
ds = Measure('ds', domain = mesh, subdomain_data = boundary)

T  = Expression(('0.0', '-100'),degree=1)  # Load on the boundary

#Define Mixed Space (R2,R) -> (u,p)
V = VectorElement("CG", mesh.ufl_cell(), 2)
W = FiniteElement("CG", mesh.ufl_cell(), 1)
L = FunctionSpace(mesh,W)
MS = dolfin.FunctionSpace(mesh, MixedElement([V,W]))

# Define boundary condition
bcu1 = DirichletBC(MS.sub(0).sub(0), 0.0, left_right)  # slip condition
bcu2 = DirichletBC(MS.sub(0).sub(1), 0.0, bottom)      # slip condition
bcp = DirichletBC(MS.sub(1), 0.0, top) # drained condition
bc=[bcu1,bcu2,bcp]

# Define strain
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
   return lambda_*tr(epsilon(u))*Identity(2) + 2.0*mu*epsilon(u)

# Define variational problem and initial condition
X0 = Function(MS)
B = TestFunction(MS)

e_u0 = Expression(('0.0', '0.0'), degree=1)
e_p0 = Expression('100.0', degree=1)

u0 = interpolate(e_u0, MS.sub(0).collapse())
p0 = interpolate(e_p0, MS.sub(1).collapse())

Xn = Function(MS)
assign(Xn, [u0, p0])

(u,p)=split(X0)
(u_n,p_n)=split(Xn)
(v,q)=split(B)

F = (1/dt)*nabla_div(u-u_n)*q*dx + (k/(mu_l))*dot(grad(p),grad(q))*dx  + ( S/dt )*(p-p_n)*q*dx
F += inner(2*mu*epsilon(u),epsilon(v))*dx + lambda_*nabla_div(u)*nabla_div(v)*dx -p*nabla_div(v)*dx- dot(T,v)*ds(2)

#solver tuning
dX0 = TrialFunction(MS)
J = derivative(F, X0, dX0)
Problem = NonlinearVariationalProblem(F, X0, J = J, bcs = bc)
Solver  = NonlinearVariationalSolver(Problem)
Solver.parameters['newton_solver']['convergence_criterion'] = 'incremental'
Solver.parameters['newton_solver']['relative_tolerance'] = 1.e-11
Solver.parameters['newton_solver']['absolute_tolerance'] = 5.e-10

#if you want to store the results
#vtkfile_u = File('Terzaghi_CG/u.pvd')
#vtkfile_p = File('Terzaghi_CG/p.pvd')

t = 0

for n in range(num_steps):
	t += dt
	Solver.solve()    
	assign(Xn,X0)

	if n%10 == 0:
		(_u,_p)=X0.split()
		#vtkfile_u << (_u,t)
		#vtkfile_p << (_p,t)

