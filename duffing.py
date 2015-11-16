# This code simulates the duffing oscillator:
# Damped driven harmonic oscillator in a double well potential.
# F = -gamma*dx/dt + 2*a*x - 4*b*x^3 + F_0*cos(omega*t)
# Second order nonlinear differential equation numerically solved by Taylor expansion.

# For the current set of parameters the motion is chaotic, i.e.
# the motion is strongly sensitive to the initial conditions. Additionally
# no fixed period of motion is observed. The poincare plot is a fractal.

import numpy as np
import matplotlib.pyplot as plt
import time
t1 = time.time() #times the computation

# parameters (mass = 1)
a = 0.5
b = 1/16.0
F_0 = 2.5
omega = 2.0
gamma = 0.1
h = 1e-1 # time step
period = 2*np.pi/(1.0*omega)
# length of the simulation
T = 30000
t = np.arange(0,T,h)

def x_2(x,v):
    '''
    second derivative term for Taylor series
    '''
    return -gamma*v + 2.0*a*x - 4.0*b*x*x*x

def x_3(x2,x,v):
    '''
    third derivative term for Taylor series
    '''
    return -gamma*x2 + 2.0*a*v -12.0*b*x*x*v

def x_4(x3,x2,x,v):
    '''
    fourth derivative term for Taylor series
    '''
    return -gamma*x3 + 2.0*a*x2 -12.0*b*x*x*x2 - 24.0*b*v*v*x

def x_5(x4,x3,x2,x,v):
    '''
    fifth derivative term for Taylor series
    '''
    return -gamma*x4 + 2*a*x3 -12.0*b*(x*x*x3 + 2.0*x2*x*v) -24.0*b*(v*v*v+2*x*v*x2)

# Trigonometric terms in derivatives. Evaluate before the loop
x2F = F_0*np.cos(omega*t)
x3F = -F_0*omega*np.sin(omega*t)
x4F = -F_0*omega*omega*np.cos(omega*t)
x5F = F_0*omega*omega*omega*np.sin(omega*t)

# coefficients in front of Taylor series expansion
# Evaluate before the loop
coef1 = 0.5*h**2.0
coef2 = 1.0/6.0*h**3.0
coef3 = 1.0/24.0*h**4.0
coef4 = 1.0/120.0*h**5.0

# initial conditions
v = 0.0
x = 0.5

position = np.zeros(len(t))
velocity = np.zeros(len(t))
position[0] = x

for i in range(1,len(t)):
    d2 = x_2(x,v) + x2F[i]
    d3 = x_3(d2,x,v) + x3F[i]
    d4 = x_4(d3,d2,x,v) + x4F[i]
    d5 = x_5(d4,d3,d2,x,v) + x5F[i]
    # Taylor series expansion for x,v. Order h^5
    x += v*h + coef1*d2 + coef2*d3 + coef3*d4 + coef4*d5
    v += d2*h + coef1*d3 + coef2*d4 + coef3*d5
    position[i] = x
    velocity[i] = v

##f = open('data_duffing_pos_vel.txt','w')
##for i in range(len(t)):
##    f.write('%f %f' %(position[i], velocity[i]))
##f.close()

# obtain phase space points at integer multiples of the period for Poincare plot
strange_attractor = np.zeros([int(T/period),2])
k = 1
for i in range(len(t)):
    if abs(t[i]-k*period)<h:
        strange_attractor[k-1,0] = position[i]
        strange_attractor[k-1,1] = velocity[i]
        k+=1

t2 = time.time()
print 'computation takes ',t2-t1,' seconds.'

plt.figure(1)
plt.plot(t[-3000:],position[-3000:],'g-',linewidth=4.0)
plt.title('Trajectory of the oscillator',{'fontsize':24})
plt.xlabel('time',{'fontsize':24})
plt.ylabel('Position',{'fontsize':24})
plt.tick_params(axis='both',labelsize=24)

plt.figure(2)
plt.plot(position[-3000:],velocity[-3000:],'r-',linewidth=4.0)
plt.title('Phase space',{'fontsize':24})
plt.xlim([-4.5,4.5])
plt.xlabel('Position',{'fontsize':24})
plt.ylabel('Momentum',{'fontsize':24})
plt.tick_params(axis='both',labelsize=24)

plt.figure(3)
plt.scatter(strange_attractor[:,0],strange_attractor[:,1])
plt.xlabel('Position',{'fontsize':24})
plt.ylabel('Momentum',{'fontsize':24})
plt.title(r'Poincare Plot (Phase space at time = $\frac{2\pi N}{\omega}$, N = 1,2,3...)',{'fontsize':24})
plt.tick_params(axis='both',labelsize=24)

plt.show((1,2,3))
