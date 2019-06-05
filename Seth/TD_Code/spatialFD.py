import numpy as np
import math
import scipy

def du_dx_midpoint(u,dx):
    du       = np.ones(len(u))
    du[0]    = u[1] - u[-1]
    du[-1]   = u[0] - u[-2]
    du[1:-1] = u[2:] - u[:-2]

    return du/(2*dx)

def du_dx_forward_1stO(u,dx):
    du       = np.ones(len(u))
    du[-1]   = u[0] - u[-1]
    du[:-1]  = u[1:] - u[:-1]

    return du/dx

def du_dx_forward_2ndO(u,dx):
    du       = np.ones(len(u))
    du[-2]   = -u[0] + 4*u[-1] - 3*u[-2]
    du[-1]   = -u[1] + 4*u[0] - 3*u[-1]
    du[:-2]  = -u[2:] + 4*u[1:-1] - 3*u[:-2]

    return du/(2*dx)

def du_dx_backward_2ndO(u,dx):
    du      = np.ones(len(u))
    du[1]   = u[-1] - 4*u[0] + 3*u[1]
    du[0]   = u[-2] - 4*u[-1] + 3*u[0]
    du[2:]  = u[:-2] - 4*u[1:-1] + 3*u[2:]

    return du/(2*dx)

def du_dx_4th_order(u,dx):
    u_new = np.ones(len(u))
    u_new[1]    = - u[3]  + 8.0*u[2]    - 8.0*u[0]    + u[-1]
    u_new[0]    = - u[2]  + 8.0*u[1]    - 8.0*u[-1]   + u[-2]
    u_new[-1]   = - u[1]  + 8.0*u[0]    - 8.0*u[-2]   + u[-3]
    u_new[-2]   = - u[0]  + 8.0*u[-1]   - 8.0*u[-3]   + u[-4]
    u_new[2:-2] = - u[4:] + 8.0*u[3:-1] - 8.0*u[1:-3] + u[:-4]

    return u_new/(12.0*dx)
