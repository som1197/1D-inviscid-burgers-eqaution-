# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 22:43:03 2020

@author: Saumya Dholakia
"""
#PART 1 
#Input parameters
import numpy as np
L=1
c = 0.5
n=1
tb= L/(np.pi)
tend = tb/2
dx= 0.025
umax = 1
x = np.arange(0,L+dx,dx)
nx = np.size(x)

#The Maccormick algorithm
class maccormick():
    def solve(self,x,uinitial,c,umax,tend):
        import numpy as np
        dx = x[1]-x[0]
        dt = (c*dx)/umax
        nx = np.size(x) 
        uold = np.zeros(nx)
        ucorr = np.zeros(nx)
        upred = np.zeros(nx)
        t = np.arange(0,tend,dt)
        uold = np.copy(uinitial)
        for j in t:
            for i in range(1,nx-1):
                upred[i] =uold[i]-(dt/(2*dx))*((uold[i+1]**2)-(uold[i]**2))
                ucorr[i] = 0.5*(uold[i]+upred[i]-((dt/(2*dx))*((upred[i]**2)-(upred[i-1]**2))))
            
            #Boundary conditions
            
            upred[0] =uold[0]-(dt/(2*dx))*((uold[1]**2)-(uold[0]**2))
            upred[nx-1] = upred[0]
            ucorr[0] = 0.5*(uold[0]+upred[0]-((dt/(2*dx))*((upred[0]**2)-(upred[nx-2]**2))))
            ucorr[nx-1] = ucorr[0]
            for i in range(nx):
                uold[i]=ucorr[i]
        return ucorr

#The Godunov algorithm
class godunov():
    def solve(self,x,uinitial,c,umax,tend):
        import numpy as np
        dx = x[1]-x[0]
        dt = (c*dx)/umax
        nx = np.size(x) 
        uold = np.zeros(nx)
        unew = np.zeros(nx)
        t = np.arange(0,tend,dt)
        uold = np.copy(uinitial)
        for j in t:
            for i in range(1,nx-1):
                s = (uold[i] + uold[i+1])/2
                
                if uold[i] >= uold[i+1]:
                    if s > 0:
                        Fright = (uold[i]**2)/2
                    else:
                        Fright = (uold[i+1]**2)/2
                elif uold[i] < uold[i+1]:
                    if uold[i] > 0:
                        Fright = (uold[i]**2)/2
                    elif uold[i+1] < 0:
                        Fright = (uold[i+1]**2)/2
                    elif uold[i] <= 0 <= uold[i+1]:
                        Fright = 0
                        
                if uold[i-1] >= uold[i]:
                    if s > 0:
                        Fleft = (uold[i-1]**2)/2
                    else:
                        Fleft = (uold[i]**2)/2
                elif uold[i-1] < uold[i]:
                    if uold[i-1] > 0:
                        Fleft = (uold[i-1]**2)/2
                    elif uold[i] < 0:
                        Fleft = (uold[i]**2)/2
                    elif uold[i-1] <= 0 <= uold[i]:
                        Fleft = 0
                        
                 #Boundary conditions  
                 
                if uold[0] >= uold[1]:
                    if s > 0:
                        Fright0 = (uold[0]**2)/2
                    else:
                        Fright0 = (uold[1]**2)/2
                elif uold[0] < uold[1]:
                    if uold[0] > 0:
                        Fright0 = (uold[0]**2)/2
                    elif uold[1] < 0:
                        Fright0 = (uold[1]**2)/2
                    elif uold[0] <= 0 <= uold[1]:
                        Fright0 = 0
                        
                if uold[nx-2] >= uold[0]:
                    if s > 0:
                        Fleft0 = (uold[nx-2]**2)/2
                    else:
                        Fleft0 = (uold[0]**2)/2
                elif uold[nx-2] < uold[0]:
                    if uold[nx-2] > 0:
                        Fleft0 = (uold[nx-2]**2)/2
                    elif uold[0] < 0:
                        Fleft0 = (uold[0]**2)/2
                    elif uold[nx-2] <= 0 <= uold[0]:
                        Fleft0 = 0
                        
                unew[i] =uold[i]-((Fright-Fleft)*(dt/dx))
            unew[0] =uold[0]-((Fright0-Fleft0)*(dt/dx))
            unew[nx-1] = unew[0]
            for i in range(nx):
                uold[i]=unew[i]
        return unew
    
#The Roe algorithm   
class roe():
    def solve(self,x,uinitial,c,umax,tend):
        import numpy as np
        dx = x[1]-x[0]
        dt = c*dx/umax
        nx = np.size(x) 
        uold = np.zeros(nx)
        unew = np.zeros(nx)
        t = np.arange(0,tend,dt)
        uold = np.copy(uinitial)
        for j in t:
            for i in range(1,nx-1):
                Fleft = ((uold[i]**2) + (uold[i-1]**2))/4 - ((1/4)*np.abs(uold[i]+uold[i-1])*(uold[i]-uold[i-1]))
                Fright = ((uold[i]**2) + (uold[i+1]**2))/4 - ((1/4)*np.abs(uold[i]+uold[i+1])*(uold[i+1]-uold[i]))
                unew[i] =uold[i]-(Fright-Fleft)*(dt/dx)
                
            #Boundary conditions
            
            Fleft0 = ((uold[0]**2) + (uold[nx-2]**2))/4 - ((1/4)*np.abs(uold[0]+uold[nx-2])*(uold[0]-uold[nx-2]))
            Fright0 = ((uold[0]**2) + (uold[1]**2))/4 - ((1/4)*np.abs(uold[0]+uold[1])*(uold[1]-uold[0]))
            unew[0] =uold[0]-(Fright0-Fleft0)*(dt/dx)
            unew[nx-1] = unew[0]
            for i in range(nx):
                uold[i]=unew[i]
        return unew
    

#Initial condition and Exact solution for the sine function
uinitial = (1 + np.sin(2.*np.pi*x))/2

#PART 1A
from scipy.optimize import brentq
nx=np.size(x)
uexact=np.zeros(nx)
def bisectu(u,x,tend):
    zeta=x-u*tend
    return u-(1+np.sin(2*np.pi*zeta))/2
for i in range(nx):
    uexact[i]=brentq(bisectu,0.,1.,args=(x[i],tend),rtol=1.e-10)

#Solvers
solver1 = maccormick()
solver2 = godunov()
solver3 = roe()
u_maccormick_sine = solver1.solve(x,uinitial,c,umax,tend)
u_godunov_sine = solver2.solve(x,uinitial,c,umax,tend)
u_roe_sine = solver3.solve(x,uinitial,c,umax,tend)
    
import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
ax.plot(x,u_maccormick_sine,label='Maccormack',color='red')
ax.plot(x,u_godunov_sine,label='Godunov',color='blue')
ax.plot(x,u_roe_sine,label='Roe',color='orange')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Q1a.png',bbox_inches='tight')

#PART 1B
tend = tb

from scipy.optimize import brentq
nx=np.size(x)
uexact=np.zeros(nx)
def bisectu(u,x,tend):
    zeta=x-u*tend
    return u-(1+np.sin(2*np.pi*zeta))/2
for i in range(nx):
    uexact[i]=brentq(bisectu,0.,1.,args=(x[i],tend),rtol=1.e-10)

#Solvers
solver1 = maccormick()
solver2 = godunov()
solver3 = roe()
u_maccormick_sine = solver1.solve(x,uinitial,c,umax,tend)
u_godunov_sine = solver2.solve(x,uinitial,c,umax,tend)
u_roe_sine = solver3.solve(x,uinitial,c,umax,tend)

import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
ax.plot(x,u_maccormick_sine,label='Maccormack',color='red')
ax.plot(x,u_godunov_sine,label='Godunov',color='blue')
ax.plot(x,u_roe_sine,label='Roe',color='orange')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Q1b.png',bbox_inches='tight')

#PART 1C
tend = 2*tb

#from scipy.optimize import brentq
#nx=np.size(x)
#uexact=np.zeros(nx)
#def bisectu(u,x,tend):
#    zeta=x-u*tend
#    return u-(1+np.sin(2*np.pi*zeta))/2
#for i in range(nx):
#    uexact[i]=brentq(bisectu,0.,1.,args=(x[i],tend),rtol=1.e-10)

#Solvers
solver1 = maccormick()
solver2 = godunov()
solver3 = roe()
u_maccormick_sine = solver1.solve(x,uinitial,c,umax,tend)
u_godunov_sine = solver2.solve(x,uinitial,c,umax,tend)
u_roe_sine = solver3.solve(x,uinitial,c,umax,tend)

import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
#ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
ax.plot(x,u_maccormick_sine,label='Maccormack',color='red')
ax.plot(x,u_godunov_sine,label='Godunov',color='blue')
ax.plot(x,u_roe_sine,label='Roe',color='orange')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Q1c.png',bbox_inches='tight')

#PART2
#PART 2A
tend = tb/2

from scipy.optimize import brentq
nx=np.size(x)
uexact=np.zeros(nx)
def bisectu(u,x,tend):
    zeta=x-u*tend
    return u-(1+np.sin(2*np.pi*zeta))/2
for i in range(nx):
    uexact[i]=brentq(bisectu,0.,1.,args=(x[i],tend),rtol=1.e-10)

#Solvers
solver1 = maccormick()
solver2 = godunov()
solver3 = roe()
u_maccormick_sine = solver1.solve(x,uinitial,c,umax,tend)
u_godunov_sine = solver2.solve(x,uinitial,c,umax,tend)
u_roe_sine = solver3.solve(x,uinitial,c,umax,tend)

import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,u_maccormick_sine - uexact,label='Maccormack',color='red')
ax.plot(x,u_godunov_sine - uexact,label='Godunov',color='blue')
ax.plot(x,u_roe_sine - uexact,label='Roe',color='orange')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$\epsilon = u-u_{exact}$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Q2a.png',bbox_inches='tight')

#PART 2B
tend = tb

from scipy.optimize import brentq
nx=np.size(x)
uexact=np.zeros(nx)
def bisectu(u,x,tend):
    zeta=x-u*tend
    return u-(1+np.sin(2*np.pi*zeta))/2
for i in range(nx):
    uexact[i]=brentq(bisectu,0.,1.,args=(x[i],tend),rtol=1.e-10)

#Solvers
solver1 = maccormick()
solver2 = godunov()
solver3 = roe()
u_maccormick_sine = solver1.solve(x,uinitial,c,umax,tend)
u_godunov_sine = solver2.solve(x,uinitial,c,umax,tend)
u_roe_sine = solver3.solve(x,uinitial,c,umax,tend)

import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,u_maccormick_sine - uexact,label='Maccormack',color='red')
ax.plot(x,u_godunov_sine - uexact,label='Godunov',color='blue')
ax.plot(x,u_roe_sine - uexact,label='Roe',color='orange')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$\epsilon = u-u_{exact}$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Q2b.png',bbox_inches='tight')

#PART 3A
tend = tb/2
c = 0.9

from scipy.optimize import brentq
nx=np.size(x)
uexact=np.zeros(nx)
def bisectu(u,x,tend):
    zeta=x-u*tend
    return u-(1+np.sin(2*np.pi*zeta))/2
for i in range(nx):
    uexact[i]=brentq(bisectu,0.,1.,args=(x[i],tend),rtol=1.e-10)

#Solvers
solver1 = maccormick()
solver2 = godunov()
solver3 = roe()
u_maccormick_sine = solver1.solve(x,uinitial,c,umax,tend)
u_godunov_sine = solver2.solve(x,uinitial,c,umax,tend)
u_roe_sine = solver3.solve(x,uinitial,c,umax,tend)

import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,u_maccormick_sine - uexact,label='Maccormack',color='red')
ax.plot(x,u_godunov_sine - uexact,label='Godunov',color='blue')
ax.plot(x,u_roe_sine - uexact,label='Roe',color='orange')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$\epsilon = u-u_{exact}$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Q3a.png',bbox_inches='tight')


#PART 3B
tend = tb
c = 0.9

from scipy.optimize import brentq
nx=np.size(x)
uexact=np.zeros(nx)
def bisectu(u,x,tend):
    zeta=x-u*tend
    return u-(1+np.sin(2*np.pi*zeta))/2
for i in range(nx):
    uexact[i]=brentq(bisectu,0.,1.,args=(x[i],tend),rtol=1.e-10)

#Solvers
solver1 = maccormick()
solver2 = godunov()
solver3 = roe()
u_maccormick_sine = solver1.solve(x,uinitial,c,umax,tend)
u_godunov_sine = solver2.solve(x,uinitial,c,umax,tend)
u_roe_sine = solver3.solve(x,uinitial,c,umax,tend)

import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,u_maccormick_sine - uexact,label='Maccormack',color='red')
ax.plot(x,u_godunov_sine - uexact,label='Godunov',color='blue')
ax.plot(x,u_roe_sine - uexact,label='Roe',color='orange')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$\epsilon = u-u_{exact}$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Q3b.png',bbox_inches='tight')


#PART 4A
dx= 0.0125
c = 0.5
tend = tb/2

from scipy.optimize import brentq
nx=np.size(x)
uexact=np.zeros(nx)
def bisectu(u,x,tend):
    zeta=x-u*tend
    return u-(1+np.sin(2*np.pi*zeta))/2
for i in range(nx):
    uexact[i]=brentq(bisectu,0.,1.,args=(x[i],tend),rtol=1.e-10)

#Solvers
solver1 = maccormick()
solver2 = godunov()
solver3 = roe()
u_maccormick_sine = solver1.solve(x,uinitial,c,umax,tend)
u_godunov_sine = solver2.solve(x,uinitial,c,umax,tend)
u_roe_sine = solver3.solve(x,uinitial,c,umax,tend)

import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,u_maccormick_sine - uexact,label='Maccormack',color='red')
ax.plot(x,u_godunov_sine - uexact,label='Godunov',color='blue')
ax.plot(x,u_roe_sine - uexact,label='Roe',color='orange')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$\epsilon = u-u_{exact}$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Q4a.png',bbox_inches='tight')


#PART 4B
dx= 0.0125
c = 0.5
tend = tb
from scipy.optimize import brentq
nx=np.size(x)
uexact=np.zeros(nx)
def bisectu(u,x,tend):
    zeta=x-u*tend
    return u-(1+np.sin(2*np.pi*zeta))/2
for i in range(nx):
    uexact[i]=brentq(bisectu,0.,1.,args=(x[i],tend),rtol=1.e-10)

#Solvers
solver1 = maccormick()
solver2 = godunov()
solver3 = roe()
u_maccormick_sine = solver1.solve(x,uinitial,c,umax,tend)
u_godunov_sine = solver2.solve(x,uinitial,c,umax,tend)
u_roe_sine = solver3.solve(x,uinitial,c,umax,tend)

import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,u_maccormick_sine - uexact,label='Maccormack',color='red')
ax.plot(x,u_godunov_sine - uexact,label='Godunov',color='blue')
ax.plot(x,u_roe_sine - uexact,label='Roe',color='orange')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$\epsilon = u-u_{exact}$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Q4b.png',bbox_inches='tight')

#L2 Norm
import numpy as np
ndx = 7
Ldx = np.arange(1,ndx + 1)
dxstart = 0.1
dx = np.zeros(ndx)
dxinv = np.zeros(ndx)
dxsq = np.zeros(ndx)
eps1 = np.zeros(np.size(Ldx))
eps2 = np.zeros(np.size(Ldx))
eps3 = np.zeros(np.size(Ldx)) 
for kk in Ldx:
    dx[kk-1] = dxstart/(2**(kk-1))
    dxinv[kk-1] = 1/dx[kk-1]
    dxsq[kk-1] = dx[kk-1]**2
    x = np.arange(0,L+dx[kk-1],dx[kk-1])
    uinitial = (1 + np.sin(2.*np.pi*x))/2
    nx = np.size(x)
    err = np.zeros(nx)
    from scipy.optimize import brentq      
    uexact=np.zeros(nx)
    def bisectu(u,x,tend):
        zeta=x-u*tend
        return u-(1+np.sin(2*np.pi*zeta))/2
    for i in range(nx):
        uexact[i]=brentq(bisectu,0.,1.,args=(x[i],tend),rtol=1.e-10)
        solver1 = maccormick()
        solver2 = godunov()
        solver3 = roe()
    u_maccormick_sine = solver1.solve(x,uinitial,c,umax,tend)
    u_godunov_sine = solver2.solve(x,uinitial,c,umax,tend)
    u_roe_sine = solver3.solve(x,uinitial,c,umax,tend)
    eps1[kk-1] = ((np.sum((u_maccormick_sine - uexact)**2))**0.5)/nx
    eps2[kk-1] = ((np.sum((u_godunov_sine - uexact)**2))**0.5)/nx
    eps3[kk-1] = ((np.sum((u_roe_sine - uexact)**2))**0.5)/nx
import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.loglog(dxinv,eps1,label='Maccormick',color='blue')
ax.loglog(dxinv,eps2,label='Godunov',color='orange')
ax.loglog(dxinv,eps3,label='Roe',color='brown')
ax.loglog(dxinv,dx/0.1,label='First order convergence',color='black',linestyle='--')
ax.loglog(dxinv,dxsq/0.01,label='Second order convergence',color='black',linestyle='-')
ax.set_xlabel('$1/Deltax$',size=20)
ax.set_ylabel('$L2error$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.show()





    





