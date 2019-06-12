import numpy as np
import math
import scipy

# def FE_step(u,dt,dx,du_dt_rhs):
#     return u + dt * du_dt_rhs(u,dx)
#
# def RK2_step(u,dt,dx,du_dt_rhs):
#     w1 = du_dt_rhs(u,dx)
#     w2 = du_dt_rhs(u + 0.5*dt*w1,dx)
#     return u + dt * w2
#
# def RK4_step(u,dt,dx,du_dt_rhs):
#     w1 = du_dt_rhs(u, dx)
#     w2 = du_dt_rhs(u + 0.5*dt*w1, dx)
#     w3 = du_dt_rhs(u + 0.5*dt*w2, dx)
#     w4 = du_dt_rhs(u + dt*w3, dx)
#     return u + dt * (w1 + 2.0*w2 + 2.0*w3 + w4)/6.0
#
# def evolve(u, dx_min, t_current, t_final, CF, du_dt_rhs, t_stepper):
#     dt_temp = dx_min * CF
#     t_steps = math.ceil((t_final - t_current)/dt_temp)
#     dt = (t_final - t_current)/t_steps
#
#     for n in range(t_steps):
#         u = t_stepper(u,dt,dx_min,du_dt_rhs)
#
#     return u


def FE_step(u,dt,du_dt_rhs,dict):
    return u + dt * du_dt_rhs(u,dict)

def RK2_step(u,dt,du_dt_rhs,dict):
    w1 = du_dt_rhs(u,dict)
    w2 = du_dt_rhs(u + 0.5*dt*w1,dict)
    return u + dt * w2

def RK4_step(u,dt,du_dt_rhs,dict):
    w1 = du_dt_rhs(u, dict)
    w2 = du_dt_rhs(u + 0.5*dt*w1, dict)
    w3 = du_dt_rhs(u + 0.5*dt*w2, dict)
    w4 = du_dt_rhs(u + dt*w3, dict)
    return u + dt * (w1 + 2.0*w2 + 2.0*w3 + w4)/6.0

def evolve(u, t_current, t_final, du_dt_rhs, t_stepper, dict):
    dt_temp = dict['dx_min'] * dict['CF']
    t_steps = math.ceil((t_final - t_current)/dt_temp)
    dt = (t_final - t_current)/t_steps

    for n in range(t_steps):
        u = t_stepper(u,dt,du_dt_rhs,dict)

    return u
