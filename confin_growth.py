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

##################################
# 		 Encapsulated MCTS
# 		incremental version
##################################


#################
# Parameters
#################

num_steps =2000
dT = Constant(1.0)
t=0.0
#Alginate
poro_0_A=0.8	#initial porosity
nu_A=0.4 		#poisson modulus
k_A = 1e-17 	# 1/500 of ECM intrinsic permeability
a2=40000		# network thinness (extremely thin) 
E_A=68000 		#young modulus of the Alginate
#ECM
poro_0_E=0.8	
E_E=1000 #Pa
nu_E=0.4
k_E = 1.8e-15 
#Outside (fictive matrix)
poro_0_L=0.8	
E_L=600 #Pa
nu_L=0.4
k_L = 1.8e-15
#IF
rho_l = 1000	#density
mu_l = 1e-2 	#dynamic viscosity
#TC
rho_t=1000 
gam_tn = 1e-2 
#Nutrient
D_0 = 3.2e-9
delta = 2
w_crit = 1e-6
w_env = 4.2e-6

#Constitutive optimized parameters
a=890
mu_t=36
gam_tg = 3.33e-2
gam_nl_g = 4e-4
gam_nl_0 = 6.65e-4
p1=1432			#begin inhibition
p_crit=5944		#total inhibition


# Elastic sub-domains
lambda_E=(E_E*nu_E)/((1+nu_E)*(1-2*nu_E))
mu_E =E_E/(2*(1+nu_E))

lambda_A=(E_A*nu_A)/((1+nu_A)*(1-2*nu_A))
mu_A =E_A/(2*(1+nu_A))

lambda_L=(E_L*nu_L)/((1+nu_L)*(1-2*nu_L))
mu_L =E_L/(2*(1+nu_L))

######################################
# Mesh, Domain, Subdomains, Boundaries
######################################
mesh = Mesh()
with XDMFFile("Mesh_confin/capsule_fig.xdmf") as infile:
	infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("Mesh_confin/mf_fig.xdmf") as infile:
	infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("Mesh_confin/cf_fig.xdmf") as infile:
	infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

#Cylindrical coordinates
x = SpatialCoordinate(mesh)
r=abs(x[0])

c = Expression(("0.0"), degree=1)

#########################################
# Function spaces, BC, Experimental data
#########################################

#Define Mixed Space (R2,R,R,R) -> (u,pl,ptl,wnl)
d=2
V = VectorElement("CG", mesh.ufl_cell(), 3)
W = FiniteElement("CG", mesh.ufl_cell(), 2)
V2 = FunctionSpace(mesh, V)
W1= FunctionSpace(mesh, W)
W2 = FunctionSpace(mesh, W)
MS = FunctionSpace(mesh, MixedElement([V,W,W,W]))


# Define boundary condition
bcu10 = DirichletBC(MS.sub(0).sub(1), c, mf, 10)  	# condition appui simple
bcu12 = DirichletBC(MS.sub(0).sub(1), c, mf, 12)  	# condition appui simple
bcu13 = DirichletBC(MS.sub(0).sub(1), c, mf, 13)  	# condition appui simple
bcu20 = DirichletBC(MS.sub(0).sub(0), c, mf, 20)  	# condition appui simple
bcu22 = DirichletBC(MS.sub(0).sub(0), c, mf, 22)  	# condition appui simple
bcu23 = DirichletBC(MS.sub(0).sub(0), c, mf, 23)  	# condition appui simple
bcu33 = DirichletBC(MS.sub(0), (0.0, 0.0), mf, 33)    # condition appui simple
bcpl = DirichletBC(MS.sub(1), c, mf, 33) 			# drained condition
bcpt = DirichletBC(MS.sub(2), c, mf, 33) 			# drained condition
bcw = DirichletBC(MS.sub(3), c, mf, 33) 		# nutrient boundary
bc=[bcu10,bcu12,bcu13,bcu20,bcu22,bcu23,bcu33,bcpl,bcpt,bcw]

# Div vect en cyl
def axi_div_2D(u):
    return u[0]/r+u[0].dx(0)+u[1].dx(1)

#quadrature degree
q_degree=10

dx = Measure('dx', domain = mesh, subdomain_data = cf)
ds = Measure('ds', domain = mesh, subdomain_data = mf)
dx = dx(metadata={'quadrature_degree': q_degree})

################################
# Constitutive relationships
################################


# Define strain
def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],[0, v[0]/r, 0],[v[1].dx(0),0, v[1].dx(1)]]))

I = Identity(3)             # Identity tensor

def sigma_E(u):
	return lambda_E*tr(epsilon(u))*I + 2.0*mu_E*epsilon(u)
def sigma_A(u):
	return lambda_A*tr(epsilon(u))*I + 2.0*mu_A*epsilon(u)
def sigma_L(u):
	return lambda_L*tr(epsilon(u))*I + 2.0*mu_L*epsilon(u)

# Define saturation of IF and TC and derivative

def Sl(ptl_n):
	cond=conditional(lt((2/pi)*atan(ptl_n/(a)),0.01),1.0,1-(2/pi)*atan(ptl_n/(a*(1-t_wnecro)+1e-12)))
	return cond

def St(ptl_n):
    return 1-Sl(ptl_n)

def dSldpl(ptl_n):
	return -(2/((a*(1-t_wnecro)+1e-12)*pi))*( 1/( 1+pow((ptl_n/(a*(1-t_wnecro)+1e-12)),2) ) )

# for alginate
def Sl_a(ptl_n):
	cond=conditional(lt((2/pi)*atan(ptl_n/(a2)),0.01),1.0,1-(2/pi)*atan(ptl_n/(a2*(1-t_wnecro)+1e-12)))
	return cond

def St_a(ptl_n):
    return 1-Sl_a(ptl_n)

def dSldpl_a(ptl_n):
	return -(2/((a2*(1-t_wnecro)+1e-12)*pi))*( 1/( 1+pow((ptl_n/(a2*(1-t_wnecro)+1e-12)),2) ) )

# Define solid pressure
def ps(ptl_n,pl_n):
    return pl_n+(2/pi)*atan(ptl_n/(a*(1-t_wnecro)+1e-12))*ptl_n

# for alginate
def ps_a(ptl_n,pl_n):
    return pl_n+(1-Sl_a(ptl_n))*ptl_n

# Define relative permeabilities
def k_rl(ptl_n):
	cond=conditional(lt(pow(Sl(ptl_n),6),1e-3),1e-3 ,pow(Sl(ptl_n),6))
	return cond


def k_rt_E(ptl_n):
	cond=conditional(lt(pow(St(ptl_n),2),1e-3),1e-3 ,pow(St(ptl_n),2))
	return cond

# for alginate
def k_rt_A(ptl_n):
	cond=conditional(lt(pow(St_a(ptl_n),6),1e-10),1e-10,pow(St_a(ptl_n),6))
	return cond

def k_rt_L(ptl_n):
	cond=conditional(lt(pow(St(ptl_n),2),1e-3),1e-3,pow(St(ptl_n),2))
	return cond



# Define TC growth
def M_l_t(wnl_n,ptl_n,pl_n):
	#H1
	cond = conditional(lt(w_crit,wnl_n),1.0 ,0.0)
	cond2 = conditional(lt(wnl_n,w_env),0.5-0.5*cos(pi*( (wnl_n-w_crit)/(w_env-w_crit) ) ),1.0)
	#Hp
	cond3 = conditional(lt((p1),ptl_n+pl_n),1.0,0.0)
	cond4 = conditional(lt(ptl_n+pl_n,(p_crit)),pow(( abs(ptl_n+pl_n-p1)  / (p_crit-p1)),0.5),1.0)
	return (gam_tg)*cond*cond2*(1-cond3*cond4)*(1-t_wnecro)*t_poro*(1.0-Sl(ptl_n))


# for alginate
def M_l_t_a(wnl_n,ptl_n,pl_n):
	#H1
	cond = conditional(lt(w_crit,wnl_n),1.0 ,0.0)
	cond2 = conditional(lt(wnl_n,w_env),0.5-0.5*cos(pi*( (wnl_n-w_crit)/(w_env-w_crit) ) ),1.0)
	#Hp
	cond3 = conditional(lt((p1),ptl_n+pl_n),1.0,0.0)
	cond4 = conditional(lt(ptl_n+pl_n,(p_crit)),pow(( abs(ptl_n+pl_n-p1)  / (p_crit-p1)),0.5),1.0)
	return (gam_tg)*cond*cond2*(1-cond3*cond4)*(1-t_wnecro)*t_poro*(1.0-Sl_a(ptl_n))

# Define nutrient consumption
def M_nl_t(wnl_n,ptl_n,pl_n):
	#H1
	cond = conditional(lt(w_crit,wnl_n),1.0 ,0.0)
	cond2 = conditional(lt(wnl_n,w_env),0.5-0.5*cos(pi*( (wnl_n-w_crit)/(w_env-w_crit) ) ),1.0)
	#H0
	cond0 = conditional(lt(wnl_n,w_env),0.5-0.5*cos(pi*( wnl_n/w_env ) ),1.0)
	#Hp
	cond3 = conditional(lt((p1),ptl_n+pl_n),1.0,0.0)
	cond4 = conditional(lt(ptl_n+pl_n,(p_crit)),pow(( abs(ptl_n+pl_n-p1)  / (p_crit-p1)),0.5),1.0)
	return ( gam_nl_g*cond*cond2*(1-cond3*cond4) + gam_nl_0*cond0 )*(1-t_wnecro)*t_poro*(1.0-Sl(ptl_n))

# for alginate
def M_nl_t_a(wnl_n,ptl_n,pl_n):
	#H1
	cond = conditional(lt(w_crit,wnl_n),1.0 ,0.0)
	cond2 = conditional(lt(wnl_n,w_env),0.5-0.5*cos(pi*( (wnl_n-w_crit)/(w_env-w_crit) ) ),1.0)
	#H0
	cond0 = conditional(lt(wnl_n,w_env),0.5-0.5*cos(pi*( wnl_n/w_env ) ),1.0)
	#Hp
	cond3 = conditional(lt((p1),ptl_n+pl_n),1.0,0.0)
	cond4 = conditional(lt(ptl_n+pl_n,(p_crit)),pow(( abs(ptl_n+pl_n-p1)  / (p_crit-p1)),0.5),1.0)
	return ( gam_nl_g*cond*cond2*(1-cond3*cond4) + gam_nl_0*cond0 )*(1-t_wnecro)*t_poro*(1.0-Sl_a(ptl_n))

# Define nutrient diffusion
def D_nl(ptl_n):
    return D_0*pow(t_poro*Sl(ptl_n),(delta))

# Define positivity of ptl
def pos_ptl(ptl_n):
    cond = conditional(lt(ptl_n,1e-2),0.0,ptl_n)
    return cond


# Define positivity of wnecro
def pos_wnecro(wnecro_n):
    cond = conditional(lt(wnecro_n,1e-4),0.0,wnecro_n)
    return cond

#####################
# Initial conditions
#####################

# Define variational problem and initial condition
#X0 = Function(MS)
d_X0 = Function(MS)
Xn = Function(MS)
B = TestFunction(MS)

#Pressure: IF and TC
e_pl0 = Expression('0.0', degree=1)
e_ptl0=Expression(('sqrt( pow(x[0],2) + pow(x[1],2) )<5e-5 ? 120 : 0.0'), degree=3)

#Nutrient
e_w0 = Expression('4.2e-6', degree=1)

#Strain
e_u0 = Expression(('0.0', '0.0'), degree=1)

re_u0 = interpolate(e_u0, MS.sub(0).collapse())
pl0 = interpolate(e_pl0, MS.sub(1).collapse())
ptl0 = interpolate(e_ptl0, MS.sub(2).collapse())
w0 = interpolate(e_w0, MS.sub(3).collapse())
assign(Xn, [re_u0, pl0, ptl0,w0])

(d_u,d_pl,d_ptl,d_wnl)=split(d_X0)
(u_n,pl_n,ptl_n,wnl_n)=Xn.split()
(v,ql,qtl,om)=split(B)


def expr_u(u_n,d_u):
	return u_n+d_u
def expr_pl(pl_n,d_pl):
	return pl_n+d_pl
def expr_ptl(ptl_n,d_ptl):
	return ptl_n+d_ptl
def expr_wnl(wnl_n,d_wnl):
	return wnl_n+d_wnl


#########################
# Internal variables
#########################
poro=TrialFunction(W2)
pr=TestFunction(W2)
poro_n=Function(W2)

# Define variation of porosity
varporo_0= Expression('0.8', degree=1)
poro_n=project(varporo_0,W2)

def var_poro(d_u,poro_n):
	return (axi_div_2D(d_u)+poro_n)/(axi_div_2D(d_u)+1)

t_poro=interpolate(poro_n,W2)

bcporo = DirichletBC(W2, 0.8, mf, 33)

#Necrotic core problem
wnecro=TrialFunction(W2)
wn=TestFunction(W2)
test_necro_n=Function(W2)
t_t_ptl=Function(W2)
bcnecro = DirichletBC(W2, c, mf, 33)

t_ptl= interpolate(e_ptl0, W2)
assign(t_t_ptl,t_ptl)
wnecro_n= interpolate(e_pl0, W2)
t_wnecro= interpolate(e_pl0, W2)

# Define TC necrosis
def rate_necro(wnecro_n,wnl_n):
	cond = conditional(lt(wnl_n,w_crit),0.5-0.5*cos( pi*(wnl_n/w_crit) ),1.0)
	return gam_tn*(1-cond)*(1-wnecro)

zero_wnecro=Expression(('sqrt( pow(x[0],2) + pow(x[1],2) )>4e-4 ? 0.0 : t_wnecro'),t_wnecro=t_wnecro,degree=1)

#####################
# Weak formulation 
#####################

#Tumor Growth, 4 subdomains, implicit euler scheme incremental
F = 2*pi*(1/dT)*t_poro*Sl(ptl_n+d_ptl)*(d_wnl)*om*r*dx(1) \
+ 2*pi*(1/dT)*t_poro*Sl_a(ptl_n+d_ptl)*(d_wnl)*om*r*dx(2) \
+ 2*pi*(1/dT)*t_poro*Sl(ptl_n+d_ptl)*(d_wnl)*om*r*dx(3) \
+ 2*pi*t_poro*Sl(ptl_n+d_ptl)*D_nl(ptl_n+d_ptl)*dot( grad(wnl_n + d_wnl),grad(om) )*r*dx(1) \
+ 2*pi*t_poro*Sl_a(ptl_n+d_ptl)*D_nl(ptl_n+d_ptl)*dot( grad(wnl_n + d_wnl),grad(om) )*r*dx(2) \
+ 2*pi*t_poro*Sl(ptl_n+d_ptl)*D_nl(ptl_n+d_ptl)*dot( grad(wnl_n + d_wnl),grad(om) )*r*dx(3) \
- 2*pi*(1/rho_l)*((wnl_n+d_wnl)*M_l_t(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl)-M_nl_t(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl) )*om*r*dx(1)\
- 2*pi*(1/rho_l)*((wnl_n+d_wnl)*M_l_t_a(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl)-M_nl_t_a(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl) )*om*r*dx(2)\
- 2*pi*(1/rho_l)*((wnl_n+d_wnl)*M_l_t(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl)-M_nl_t(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl) )*om*r*dx(3)\
- 2*pi*k_rl(ptl_n+d_ptl)*(k_E/(mu_l))*dot(grad(pl_n+d_pl),grad(wnl_n + d_wnl))*om*r*dx(1) \
- 2*pi*k_rl(ptl_n+d_ptl)*(k_A/(mu_l))*dot(grad(pl_n+d_pl),grad(wnl_n + d_wnl))*om*r*dx(2) \
- 2*pi*k_rl(ptl_n+d_ptl)*(k_L/(mu_l))*dot(grad(pl_n+d_pl),grad(wnl_n + d_wnl))*om*r*dx(3) \


F += 2*pi*(1/dT)*Sl(ptl_n+d_ptl)*axi_div_2D(d_u)*ql*r*dx(1) + 2*pi*( 1/dT )*t_poro*dSldpl(ptl_n+d_ptl)*(d_ptl)*ql*r*dx(1) \
+ 2*pi*(1/dT)*Sl_a(ptl_n+d_ptl)*axi_div_2D(d_u)*ql*r*dx(2) + 2*pi*( 1/dT )*t_poro*dSldpl_a(ptl_n+d_ptl)*(d_ptl)*ql*r*dx(2) \
+ 2*pi*(1/dT)*Sl(ptl_n+d_ptl)*axi_div_2D(d_u)*ql*r*dx(3) + 2*pi*( 1/dT )*t_poro*dSldpl(ptl_n+d_ptl)*(d_ptl)*ql*r*dx(3) \
+ 2*pi*(1/rho_l)*M_l_t(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl)*ql*r*dx(1) \
+ 2*pi*(1/rho_l)*M_l_t_a(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl)*ql*r*dx(2) \
+ 2*pi*(1/rho_l)*M_l_t(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl)*ql*r*dx(3) \
+ 2*pi*k_rl(ptl_n+d_ptl)*(k_E/(mu_l))*dot(grad(pl_n+d_pl),grad(ql))*r*dx(1) \
+ 2*pi*k_rl(ptl_n+d_ptl)*(k_A/(mu_l))*dot(grad(pl_n+d_pl),grad(ql))*r*dx(2) \
+ 2*pi*k_rl(ptl_n+d_ptl)*(k_L/(mu_l))*dot(grad(pl_n+d_pl),grad(ql))*r*dx(3) \


F += 2*pi*(1/dT)*(1-Sl(ptl_n+d_ptl))*axi_div_2D(d_u)*qtl*r*dx(1)-2*pi*( 1/dT)*t_poro*dSldpl(ptl_n+d_ptl)*(d_ptl)*qtl*r*dx(1)\
+ 2*pi*(1/dT)*(1-Sl_a(ptl_n+d_ptl))*axi_div_2D(d_u)*qtl*r*dx(2)-2*pi*( 1/dT)*t_poro*dSldpl_a(ptl_n+d_ptl)*(d_ptl)*qtl*r*dx(2)\
+ 2*pi*(1/dT)*(1-Sl(ptl_n+d_ptl))*axi_div_2D(d_u)*qtl*r*dx(3)-2*pi*( 1/dT)*t_poro*dSldpl(ptl_n+d_ptl)*(d_ptl)*qtl*r*dx(3)\
- 2*pi*(1/rho_t)*M_l_t(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl)*qtl*r*dx(1)\
- 2*pi*(1/rho_t)*M_l_t_a(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl)*qtl*r*dx(2)\
- 2*pi*(1/rho_t)*M_l_t(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl)*qtl*r*dx(3)\
+ 2*pi*k_rt_E(ptl_n+d_ptl)*(k_E/(mu_t))*dot(grad(ptl_n+d_ptl+pl_n+d_pl),grad(qtl))*r*dx(1) \
+ 2*pi*k_rt_A(ptl_n+d_ptl)*(k_A/(mu_t))*dot(grad(ptl_n+d_ptl+pl_n+d_pl),grad(qtl))*r*dx(2) \
+ 2*pi*k_rt_L(ptl_n+d_ptl)*(k_L/(mu_t))*dot(grad(ptl_n+d_ptl+pl_n+d_pl),grad(qtl))*r*dx(3) \


F += 2*pi*( inner(sigma_E(u_n+d_u),epsilon(v))*r*dx(1)- ps(ptl_n+d_ptl,pl_n+d_pl)*axi_div_2D(v)*r*dx(1) )
F += 2*pi*( inner(sigma_A(u_n+d_u),epsilon(v))*r*dx(2)- ps_a(ptl_n+d_ptl,pl_n+d_pl)*axi_div_2D(v)*r*dx(2) )
F += 2*pi*( inner(sigma_L(u_n+d_u),epsilon(v))*r*dx(3)- ps(ptl_n+d_ptl,pl_n+d_pl)*axi_div_2D(v)*r*dx(3) )


#Necrotic Core
F2= 2*pi*t_poro*(1-Sl(ptl_n+d_ptl))*(1/dT)*(wnecro-wnecro_n)*wn*r*dx(1)\
+ 2*pi*t_poro*(1-Sl(ptl_n+d_ptl))*(1/dT)*(wnecro-wnecro_n)*wn*r*dx(4)\
+ 2*pi*t_poro*(1-Sl_a(ptl_n+d_ptl))*(1/dT)*(wnecro-wnecro_n)*wn*r*dx(2)\
+ 2*pi*t_poro*(1-Sl(ptl_n+d_ptl))*(1/dT)*(wnecro-wnecro_n)*wn*r*dx(3)\
- 2*pi*(t_poro*(1-Sl(ptl_n+d_ptl))/rho_t)*( rate_necro(wnecro,wnl_n + d_wnl) - wnecro*M_l_t(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl) ) *wn*r*dx(1)\
- 2*pi*(t_poro*(1-Sl(ptl_n+d_ptl))/rho_t)*( rate_necro(wnecro,wnl_n + d_wnl) - wnecro*M_l_t(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl) ) *wn*r*dx(4)\
- 2*pi*(t_poro*(1-Sl_a(ptl_n+d_ptl))/rho_t)*( rate_necro(wnecro,wnl_n + d_wnl) - wnecro* M_l_t_a(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl) ) *wn*r*dx(2)\
- 2*pi*(t_poro*(1-Sl(ptl_n+d_ptl))/rho_t)*( rate_necro(wnecro,wnl_n + d_wnl) - wnecro*M_l_t(wnl_n + d_wnl,ptl_n+d_ptl,pl_n+d_pl) )*wn*r*dx(3)\
- 2*pi*k_rt_E(ptl_n+d_ptl)*(k_E/(mu_t))*dot(grad(ptl_n+d_ptl+pl_n+d_pl),grad(wnecro))*wn*r*dx(1) \
- 2*pi*k_rt_E(ptl_n+d_ptl)*(k_E/(mu_t))*dot(grad(ptl_n+d_ptl+pl_n+d_pl),grad(wnecro))*wn*r*dx(4) \
- 2*pi*k_rt_A(ptl_n+d_ptl)*(k_A/(mu_t))*dot(grad(ptl_n+d_ptl+pl_n+d_pl),grad(wnecro))*wn*r*dx(2) \
- 2*pi*k_rt_L(ptl_n+d_ptl)*(k_L/(mu_t))*dot(grad(ptl_n+d_ptl+pl_n+d_pl),grad(wnecro))*wn*r*dx(3)


a0, L = lhs(F2), rhs(F2)

#Porosity
F3=2*pi*(1/dT)*(poro-poro_n)*pr*r*dx-2*pi*(1/dT)*(1-poro)*axi_div_2D(d_u)*pr*r*dx

a3, L3 = lhs(F3), rhs(F3)

#if you want to store the results
comm='''
vtkfile_u = File('result_av_47/u.pvd')
vtkfile_pl = File('result_av_47/pl.pvd')
vtkfile_ptl = File('result_av_47/ptl.pvd')
vtkfile_wnl = File('result_av_47/wnl.pvd')
vtkfile_ps = File('result_av_47/ps.pvd')
vtkfile_wnecro = File('result_av_47/wnecro.pvd')
vtkfile_sat_t = File('result_av_60/sat_t.pvd')
#'''


#to represent saturations
sat_l=Function(W2)
sat_t=Function(W2)
abs_ptl=Function(W2)

Sat_X=Function(MS)
(sat_u,sat_pl,sat_ptl,sat_wnl)=split(Sat_X)
Sat_X2=Function(MS)
(sat_u2,sat_pl2,sat_ptl2,sat_wnl2)=split(Sat_X2)

#solver tuning
dd_X0 = TrialFunction(MS)
J = derivative(F, d_X0, dd_X0)
Problem = NonlinearVariationalProblem(F, d_X0, J = J, bcs = bc)
Solver  = NonlinearVariationalSolver(Problem)
Solver.parameters['newton_solver']['convergence_criterion'] = 'incremental'
Solver.parameters['newton_solver']['linear_solver'] = 'mumps'
Solver.parameters['newton_solver']['relative_tolerance'] = 1.e-15
Solver.parameters['newton_solver']['absolute_tolerance'] = 4.e-11
Solver.parameters['newton_solver']['maximum_iterations'] = 30

wnecro=Function(W2)
poro=Function(W2)
val_ps=Function(W2)
val_u=Function(V2)
val_pl=Function(W2)
val_ptl=Function(W2)
val_wnl=Function(W2)
abs_ptl=Function(W2)
val_sat_t=Function(W2)

for n in range(num_steps):
	t += float(dT)
	Solver.solve()
	(d_u,d_pl,d_ptl,d_wnl)=d_X0.split()
	
	assign(abs_ptl,d_X0.sub(2))
	expr_ptl0 = Expression(('abs_ptl<1e-4 ? 0.0 : abs_ptl'),abs_ptl=abs_ptl,degree=1)
	truc=interpolate(expr_ptl0,W2)
	assign(t_t_ptl,truc)

	solve(a0 == L, wnecro, bcnecro)

	solve(a3 == L3, poro, bcporo)

	poro_n.assign(poro)
	t_poro.assign(poro)

	expr_necro = Expression(('wnecro<1e-4 ? 0.0 : wnecro'),wnecro=wnecro,degree=1)
	t_wnecro.assign(expr_necro)
	zero_wnecro.t_wnecro=t_wnecro
	z_wnecro=interpolate(zero_wnecro,W1)
	t_wnecro.assign(z_wnecro)
	wnecro_n.assign(t_wnecro)
	M_l_t.t_wnecro=t_wnecro
	M_nl_t.t_wnecro=t_wnecro
	M_l_t_a.t_wnecro=t_wnecro
	M_nl_t_a.t_wnecro=t_wnecro
	M_l_t.t_poro=t_poro
	M_nl_t.t_poro=t_poro
	M_l_t_a.t_poro=t_poro
	M_nl_t_a.t_poro=t_poro

	if n == 9:
		dT.assign(10.0)
	if n == 23:
		dT.assign(60.0)
	if n == 27:
		dT.assign(300.0)
	if n == 30:
		dT.assign(600.0)
	if n == 200:
		Solver.parameters['newton_solver']['absolute_tolerance'] = 6e-11
	if n == 300:
		Solver.parameters['newton_solver']['absolute_tolerance'] = 8e-11
	if n == 500:
		dT.assign(300.0)
		Solver.parameters['newton_solver']['absolute_tolerance'] = 1e-10
	if n == 600:
		Solver.parameters['newton_solver']['absolute_tolerance'] = 1.5e-10
	if n == 800:
		Solver.parameters['newton_solver']['absolute_tolerance'] = 2e-10
	if n == 1200:
		Solver.parameters['newton_solver']['absolute_tolerance'] = 2.5e-10
	if n == 1800:
		dT.assign(150.0)
		Solver.parameters['newton_solver']['absolute_tolerance'] = 3e-10


	val_ps=project(ps(ptl_n+t_t_ptl,pl_n+d_pl),W2)
	val_u=project(expr_u(u_n,d_u),V2)
	val_pl=project(expr_pl(pl_n,d_pl),W2)
	val_ptl=project(expr_ptl(ptl_n,t_t_ptl),W2)
	val_wnl=project(expr_wnl(wnl_n,d_wnl),W2)
	val_sat_t=project(St(ptl_n+t_t_ptl),W2)

	assign(Sat_X, [val_u, val_ps, val_sat_t, wnecro_n])
	(_sat_u,_val_ps,_val_sat_t,_wnecro)=Sat_X.split()
	assign(Sat_X2, [val_u, val_pl, val_ptl, val_wnl])
	(_sat_u2,_pl,_ptl,_wnl)=Sat_X2.split()

	#if you want to store the results
	comm='''
	vtkfile_u << (_sat_u,t)
	vtkfile_ps << (_val_ps,t)
	vtkfile_pl << (_pl,t)
	vtkfile_ptl << (_ptl,t)
	vtkfile_wnl << (_wnl,t)
	vtkfile_wnecro << (_wnecro,t)
	vtkfile_sat_t << (_val_sat_t,t)
	#'''

	assign(Xn, [val_u,val_pl,val_ptl,val_wnl])
	X = SpatialCoordinate(mesh)
	x = X + d_u
	r=abs(x[0])

