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

from __future__ import print_function
from dolfin import *

if __name__ == "__main__":

	mesh = Mesh()
	with XDMFFile("Mesh_free/capsule.xdmf") as infile:
		infile.read(mesh)
	mvc = MeshValueCollection("size_t", mesh, 1)
	with XDMFFile("Mesh_free/mf.xdmf") as infile:
		infile.read(mvc, "name_to_read")
	mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
	mvc = MeshValueCollection("size_t", mesh, 2)
	with XDMFFile("Mesh_free/cf.xdmf") as infile:
		infile.read(mvc, "name_to_read")
	cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

	dx = Measure('dx', domain = mesh, subdomain_data = cf)
	ds = Measure('ds', domain = mesh, subdomain_data = mf)
	n = FacetNormal(mesh)

	#Cylindrical coordinates
	x = SpatialCoordinate(mesh)
	r=abs(x[0])

	c = Expression(("0.0"), degree=1)

	#################
	# Parameters
	#################

	num_steps =514
	dT = Constant(1.0)
	t=0.0
	#ECM
	poro_0_E=0.8	
	E_E=1000 #Pa
	nu_E=0.4
	k_E = 1.8e-15 
	#IF
	rho_l = 1000	#density
	mu_l = 1e-2 	#dynamic viscosity
	#TC
	rho_t=1000
	gam_tn = 1e-2
	mu_t=36  
	w_crit = 1e-6
	w_env = 4.2e-6 
	#Nutrient
	D_0 = 3.2e-9
	delta = 2

	#Constitutive optimized parameters
	a=890
	gam_tg = 3.33e-2
	gam_nl_g = 4e-4
	gam_nl_0 = 6.65e-4
	p1=1432			#begin inhibition
	p_crit=5944		#total inhibition

	# Lame coef
	lambda_E=(E_E*nu_E)/((1+nu_E)*(1-2*nu_E))
	mu_E =E_E/(2*(1+nu_E))

	#########################################
	# Function spaces, BC, Experimental data
	#########################################

	#Define Mixed Space (R2,R,R,R) -> (u,pl,ptl,wnl)
	V = VectorElement("CG", mesh.ufl_cell(), 3)
	W = FiniteElement("CG", mesh.ufl_cell(), 2)
	V2 = FunctionSpace(mesh, V)
	W1= FunctionSpace(mesh, W)
	MS = FunctionSpace(mesh, MixedElement([V,W,W,W]))


	# Define boundary condition
	bcu10 = DirichletBC(MS.sub(0).sub(1), c, mf, 10)  	# condition appui simple
	bcu20 = DirichletBC(MS.sub(0).sub(0), c, mf, 20)  	# condition appui simple
	bcu30 = DirichletBC(MS.sub(0), (0.0, 0.0), mf, 30)    # condition appui simple
	bcpl = DirichletBC(MS.sub(1), c, mf, 30) 			# drained condition
	bcpt = DirichletBC(MS.sub(2), c, mf, 30) 			# drained condition
	bcw = DirichletBC(MS.sub(3), 4.2e-6, mf, 30) 		# nutrient boundary
	bc=[bcu10,bcu20,bcu30,bcpl,bcpt,bcw]

	# Vectorial div in cyl
	def axi_div_2D(u):
	    return u[0]/r+u[0].dx(0)+u[1].dx(1)

	#quadrature degree
	q_degree=10
	dx = dx(metadata={'quadrature_degree': q_degree})


	################################
	# Constitutive relationships
	################################
	poro=Function(W1)
	poro_n=Function(W1)

	# Define strain
	def epsilon(v):
	    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],[0, v[0]/r, 0],[v[1].dx(0),0, v[1].dx(1)]]))

	# Define sigma
	def sigma_E(u):
	   return lambda_E*tr(epsilon(u))*Identity(3) + 2.0*mu_E*epsilon(u)

	# Define saturation of IF and TC and derivative
	def Sl(ptl):
		cond=conditional(lt((2/pi)*atan(ptl/(a)),0.0),1.0,1-(2/pi)*atan(ptl/(a)))
		return cond

	def St(ptl):
	    return 1-Sl(ptl)

	def dSldpl(ptl):
	    return -(2/((a)*pi))*( 1/( 1+pow((ptl/(a)),2) ) )

	# Define solid pressure
	def ps(ptl,pl):
	    return pl+(2/pi)*atan(ptl/(a))*ptl


	# Define relative permeabilities
	def k_rl(ptl):
		cond=conditional(lt(pow(Sl(ptl),6),1e-3),1e-3 ,pow(Sl(ptl),6))
		return cond


	def k_rt_E(ptl):
		cond=conditional(lt(pow(St(ptl),2),1e-3),1e-3 ,pow(St(ptl),2))
		return cond

	# Define TC growth
	def M_l_t(wnl,ptl,pl):
		#H1
		cond = conditional(lt(w_crit,wnl),1.0 ,0.0)
		cond2 = conditional(lt(wnl,w_env),0.5-0.5*cos(pi*( (wnl-w_crit)/(w_env-w_crit) ) ),1.0)
		#Hp
		cond3 = conditional(lt((p1),ptl+pl),1.0,0.0)
		cond4 = conditional(lt(ptl+pl,(p_crit)),0.5-0.5*cos(pi*( (ptl+pl-p1)/((p_crit)-(p1)) ) ),1.0)
		return (gam_tg)*cond*cond2*(1-cond3*cond4)*(1-t_wnecro)*poro*(1.0-Sl(ptl))

	# Define nutrient consumption
	def M_nl_t(wnl,ptl,pl):
		#H1
		cond = conditional(lt(w_crit,wnl),1.0 ,0.0)
		cond2 = conditional(lt(wnl,w_env),0.5-0.5*cos(pi*( (wnl-w_crit)/(w_env-w_crit) ) ),1.0)
		#H0
		cond0 = conditional(lt(wnl,w_env),0.5-0.5*cos(pi*( wnl/w_env ) ),1.0)
		#Hp
		cond3 = conditional(lt((p1),ptl+pl),1.0,0.0)
		cond4 = conditional(lt(ptl+pl,(p_crit)),0.5-0.5*cos(pi*( (ptl+pl-p1)/((p_crit)-(p1)) ) ),1.0)
		return ( gam_nl_g*cond*cond2*(1-cond3*cond4) + gam_nl_0*cond0 )*(1-t_wnecro)*poro*(1.0-Sl(ptl))

	# Define nutrient diffusion
	def D_nl(ptl):
	    return D_0*pow(poro*Sl(ptl),(delta))

	# Define positivity of ptl
	def pos_ptl(ptl):
	    cond = conditional(lt(ptl,1e-2),0.0,ptl)
	    return cond

	# Define positivity of wnecro
	def pos_wnecro(wnecro):
	    cond = conditional(lt(wnecro,1e-4),0.0,wnecro)
	    return cond


	#####################
	# Initial conditions
	#####################

	# Define variational problem and initial condition
	X0 = Function(MS)
	Xn = Function(MS)
	B = TestFunction(MS)

	#Pressure: IF and TC
	e_pl0 = Expression('0.0', degree=1)
	e_ptl0=Expression(('sqrt( pow(x[0],2) + pow(x[1],2) )<3e-5 ? 120 : 0.0'), degree=3) #St=?

	#Nutrient
	e_w0 = Expression('4.2e-6', degree=1)

	#Strain
	e_u0 = Expression(('0.0', '0.0'), degree=1)

	re_u0 = interpolate(e_u0, MS.sub(0).collapse())
	pl0 = interpolate(e_pl0, MS.sub(1).collapse())
	ptl0 = interpolate(e_ptl0, MS.sub(2).collapse())
	w0 = interpolate(e_w0, MS.sub(3).collapse())
	assign(Xn, [re_u0, pl0, ptl0,w0])

	(u,pl,ptl,wnl)=split(X0)
	(u_n,pl_n,ptl_n,wnl_n)=split(Xn)
	(v,ql,qtl,om)=split(B)


	#########################
	# Internal variables
	#########################

	# Define variation of porosity
	varporo_0= Expression('0.8', degree=1)
	poro_n=project(varporo_0,W1)

	def var_poro(u,u_n,poro_n):
		return (axi_div_2D(u-u_n)+poro_n)/(axi_div_2D(u-u_n)+1)
	poro=project(var_poro(u,u_n,poro_n),W1)


	#Necrotic core problem
	wnecro=TrialFunction(W1)
	wn=TestFunction(W1)
	test_necro_n=Function(W1)
	t_t_ptl=Function(W1)

	bcnecro = DirichletBC(W1, c, mf, 30)

	t_ptl= interpolate(e_ptl0, W1)
	assign(t_t_ptl,t_ptl)
	wnecro_n= interpolate(e_pl0, W1)
	t_wnecro= interpolate(e_pl0, W1)

	(_u,_pl,_ptl,_wnl)=X0.split()

	# Define TC necrosis
	def rate_necro(wnecro,wnl):
		cond = conditional(lt(wnl,w_crit),0.5-0.5*cos( pi*(wnl/w_crit) ),1.0)
		return gam_tn*(1-cond)*(1-wnecro)

	#####################
	# Weak formulation 
	#####################

	#Tumor Growth
	F = 2*pi*(1/dT)*poro*Sl(ptl)*(wnl-wnl_n)*om*r*dx(1) \
	+ pi*poro*Sl(ptl)*D_nl(ptl)*dot( grad(wnl),grad(om) )*r*dx(1) \
	- pi*(1/rho_l)*( wnl*M_l_t(wnl,ptl,pl)-M_nl_t(wnl,ptl,pl) )*om*r*dx(1)\
	- pi*k_rl(ptl)*(k_E/(mu_l))*dot(grad(pl),grad(wnl))*om*r*dx(1) \
	+ pi*poro*Sl(ptl_n)*D_nl(ptl_n)*dot( grad(wnl_n),grad(om) )*r*dx(1) \
	- pi*(1/rho_l)*( wnl_n*M_l_t(wnl_n,ptl_n,pl_n)-M_nl_t(wnl_n,ptl_n,pl_n) )*om*r*dx(1)\
	- pi*k_rl(ptl_n)*(k_E/(mu_l))*dot(grad(pl_n),grad(wnl_n))*om*r*dx(1) \


	F += 2*pi*(1/dT)*Sl(ptl)*axi_div_2D(u-u_n)*ql*r*dx(1) + 2*pi*( 1/dT )*poro*dSldpl(ptl)*(ptl-ptl_n)*ql*r*dx(1) \
	+ pi*(1/rho_l)*M_l_t(wnl,ptl,pl)*ql*r*dx(1) \
	+ pi*k_rl(ptl)*(k_E/(mu_l))*dot(grad(pl),grad(ql))*r*dx(1) \
	+ pi*(1/rho_l)*M_l_t(wnl_n,ptl_n,pl_n)*ql*r*dx(1) \
	+ pi*k_rl(ptl_n)*(k_E/(mu_l))*dot(grad(pl_n),grad(ql))*r*dx(1) \


	F += 2*pi*(1/dT)*(1-Sl(ptl))*axi_div_2D(u-u_n)*qtl*r*dx(1)-2*pi*( 1/dT)*poro*dSldpl(ptl)*(ptl-ptl_n)*qtl*r*dx(1)\
	- pi*(1/rho_t)*M_l_t(wnl,ptl,pl)*qtl*r*dx(1)\
	+ pi*k_rt_E(ptl)*(k_E/(mu_t))*dot(grad(ptl+pl),grad(qtl))*r*dx(1) \
	- pi*(1/rho_t)*M_l_t(wnl_n,ptl_n,pl_n)*qtl*r*dx(1)\
	+ pi*k_rt_E(ptl_n)*(k_E/(mu_t))*dot(grad(ptl_n+pl_n),grad(qtl))*r*dx(1) \


	F += 2*pi*(1/dT)*( (inner(sigma_E(u),epsilon(v))*r*dx(1)- ps(ptl,pl)*axi_div_2D(v)*r*dx(1)) \
	- (inner(sigma_E(u_n),epsilon(v))*r*dx(1) - ps(ptl_n,pl_n)*axi_div_2D(v)*r*dx(1)) )


	#Necrotic Core
	F2= 2*pi*poro*(1-Sl(ptl))*(1/dT)*(wnecro-wnecro_n)*wn*r*dx(1)\
	- 2*pi*(poro*(1-Sl(ptl))/rho_t)*( rate_necro(wnecro,wnl) - wnecro*M_l_t(wnl,ptl,pl) )*wn*r*dx(1)\
	- 2*pi*k_rt_E(ptl)*(k_E/(mu_t))*dot(grad(ptl+pl),grad(wnecro))*wn*r*dx(1) \


	a0, L = lhs(F2), rhs(F2)

	#vtkfile_u = File('output_free/u.pvd')
	#vtkfile_pl = File('output_free/pl.pvd')
	#vtkfile_pt = File('output_free/ptl.pvd')
	#vtkfile_ps = File('output_free/ps.pvd')
	#vtkfile_wnl = File('output_free/wnl.pvd')
	#vtkfile_poro = File('output_free/poro.pvd')
	#vtkfile_wnecro = File('output_free/wnecro.pvd')

	#solver tuning
	dX0 = TrialFunction(MS)
	J = derivative(F, X0, dX0)
	Problem = NonlinearVariationalProblem(F, X0, J = J, bcs = bc)
	Solver  = NonlinearVariationalSolver(Problem)
	Solver.parameters['newton_solver']['convergence_criterion'] = 'incremental'
	Solver.parameters['newton_solver']['linear_solver'] = 'mumps'
	Solver.parameters['newton_solver']['relative_tolerance'] = 1.e-15
	Solver.parameters['newton_solver']['absolute_tolerance'] = 2.1e-11
	Solver.parameters['newton_solver']['maximum_iterations'] = 100

	wnecro=Function(W1)
	val_ps=Function(W1)
	abs_ptl=Function(W1)
	Sat_X=Function(MS)
	(sat_u,sat_pl,sat_ptl,sat_wnl)=split(Sat_X)

	for n in range(num_steps):
		t += float(dT)
		Solver.solve()
		(u,pl,ptl,wnl)=X0.split()
		abs_ptl=project(pos_ptl(ptl),W1)

		#update of internal variables
		t_ptl= interpolate(abs_ptl, W1)
		assign(t_t_ptl,t_ptl)
	
		solve(a0 == L, wnecro, bcnecro)
		
		poro_n=project(poro,W1) 
		poro=project(var_poro(u,u_n,poro_n),W1)
		abs_wnecro=project(pos_wnecro(wnecro),W1)
		t_wnecro.assign(abs_wnecro)
		wnecro_n.assign(t_wnecro)
		M_l_t.t_wnecro=t_wnecro
		M_nl_t.t_wnecro=t_wnecro

		if n == 19:
			dT.assign(10.0)
			print('dt=',float(dT))
		if n == 23:
			dT.assign(60.0)
			print('dt=',float(dT))
		if n == 27:
			dT.assign(300.0)
			print('dt=',float(dT))
		if n == 30:
			dT.assign(2400.0)
			print('dt=',float(dT))
		if n == 150:
			print('dt=',float(dT))
			Solver.parameters['newton_solver']['absolute_tolerance'] = 1e-10

		val_ps=project(ps(t_t_ptl,pl),W1)

		#assign(Sat_X, [u, val_ps, poro, t_wnecro])
		#(_sat_u,_val_ps,_poro,_wnecro)=Sat_X.split()
		#if (n%10==0):
		#	vtkfile_u << (u,t)
		#	vtkfile_pl << (pl,t)
		#	vtkfile_pt << (t_t_ptl,t)
		#	vtkfile_ps << (_val_ps,t)
		#	vtkfile_wnl << (wnl,t)
		#	vtkfile_poro << (_poro,t)
		#	vtkfile_wnecro << (_wnecro,t)

		assign(Xn, [u, pl, t_t_ptl, wnl])
