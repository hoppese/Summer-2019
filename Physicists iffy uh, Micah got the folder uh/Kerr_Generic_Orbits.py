import numpy as np
import scipy
from scipy import fftpack
import yaml
import sys
import os
import math
import collections
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def signa(a):
    if a > 0:
        eq = 1
    elif a < 0:
        eq = -1
    else:
        eq = 0
    return eq

def abar(M,a):
    eq = a/M
    return eq

def zmin(Thetamin):
    eq = np.cos(Thetamin)
    return eq

def f(var,M,Thetamin,a):
    a_bar = abar(M,a)
    varbar = var/M
    deltabar = varbar**2 - 2 * varbar + a_bar**2
    eq = varbar**4 + a_bar**2 * (varbar * (varbar + 2) + zmin(Thetamin)**2 * deltabar)
    return eq

def g(var,M,a):
    varbar = var/M
    eq = 2 * abar(M,a) * varbar
    return eq

def h(var,M,Thetamin,a):
    varbar = var/M
    z_min = zmin(Thetamin)
    deltabar = varbar**2 - 2 * varbar + abar(M,a)**2
    eq = varbar * (varbar - 2) + ((z_min**2)/(1 - z_min**2)) * deltabar
    return eq

def d(var,M,Thetamin,a):
    a_bar = abar(M,a)
    varbar = var/M
    deltabar = varbar**2 - 2 * varbar + a_bar**2
    eq = (varbar**2 + a_bar**2 * zmin(Thetamin)**2) * deltabar
    return eq

def fprime(var,M,Thetamin,a):
    varbar = var/M
    z_min = zmin(Thetamin)
    eq = 4 * varbar**3 + 2 * abar(M,a)**2 * ((1 + z_min**2) * varbar + (1 - z_min**2))
    return eq

def gprime(var,M,a):
    varbar = var/M
    eq = 2 * abar(M,a)
    return eq

def hprime(var,M,Thetamin):
    varbar = var/M
    eq = ((2 * (varbar - 1))/(1 - zmin(Thetamin)**2))
    return eq

def dprime(var,M,Thetamin,a):
    varbar = var/M
    z_min = zmin(Thetamin)
    eq = 2 * (2 * varbar - 3) * varbar**2 + 2 * abar(M,a)**2 * ((1 + z_min**2) * varbar - z_min**2)
    return eq

def r_p(p,e,M):
    eq = ((p * M)/(1 + e))
    return eq

def r_a(p,e,M):
    eq = ((p * M)/(1 - e))
    return eq

def r_naught(p,M):
    eq = p * M
    return eq

def rbar_p(p,e):
    eq = (p/(1 + e))
    return eq

def rbar_a(p,e):
    eq = (p/(1 - e))
    return eq

def f1(p,e,M,Thetamin,a):
    if e > 0:
        eq = f(rbar_p(p,e),M,Thetamin,a)
    elif e == 0:
        eq = f(p,M,Thetamin,a)
    return eq

def f2(p,e,M,Thetamin,a):
    if e > 0:
        eq = f(rbar_a(p,e),M,Thetamin,a)
    elif e == 0:
        eq = f(p,M,Thetamin,a)
    return eq

def g1(p,e,M,a):
    if e > 0:
        eq = g(rbar_p(p,e),M,a)
    elif e == 0:
        eq = g(p,M,a)
    return eq

def g2(p,e,M,a):
    if e > 0:
        eq = g(rbar_a(p,e),M,a)
    elif e == 0:
        eq = g(p,M,a)
    return eq

def h1(p,e,M,Thetamin,a):
    if e > 0:
        eq = h(rbar_p(p,e),M,Thetamin,a)
    elif e == 0:
        eq = h(p,M,Thetamin,a)
    return eq

def h2(p,e,M,Thetamin,a):
    if e > 0:
        eq = h(rbar_a(p,e),M,Thetamin,a)
    elif e == 0:
        eq = h(p,M,Thetamin,a)
    return eq

def d1(p,e,M,Thetamin,a):
    if e > 0:
        eq = d(rbar_p(p,e),M,Thetamin,a)
    elif e == 0:
        eq = d(p,M,Thetamin,a)
    return eq

def d2(p,e,M,Thetamin,a):
    if e > 0:
        eq = d(rbar_a(p,e),M,Thetamin,a)
    elif e == 0:
        eq = d(p,M,Thetamin,a)
    return eq

def kappa(p,e,M,Thetamin,a):
    eq = (d1(p,e,M,Thetamin,a) * h2(p,e,M,Thetamin,a)) - (h1(p,e,M,Thetamin,a) * d2(p,e,M,Thetamin,a))
    return eq

def epsilon(p,e,M,Thetamin,a):
    eq = (d1(p,e,M,Thetamin,a) * g2(p,e,M,a)) - (g1(p,e,M,a) * d2(p,e,M,Thetamin,a))
    return eq

def rho(p,e,M,Thetamin,a):
    eq = (f1(p,e,M,Thetamin,a) * h2(p,e,M,Thetamin,a)) - (h1(p,e,M,Thetamin,a) * f2(p,e,M,Thetamin,a))
    return eq

def eta(p,e,M,Thetamin,a):
    eq = (f1(p,e,M,Thetamin,a) * g2(p,e,M,a)) - (g1(p,e,M,a) * f2(p,e,M,Thetamin,a))
    return eq

def sigma(p,e,M,Thetamin,a):
    eq = (g1(p,e,M,a) * h2(p,e,M,Thetamin,a)) - (h1(p,e,M,Thetamin,a) * g2(p,e,M,a))
    return eq

def Ebar(p,e,M,Thetamin,a,plmi):
    kappa_ = kappa(p,e,M,Thetamin,a)
    rho_ = rho(p,e,M,Thetamin,a)
    epsilon_ = epsilon(p,e,M,Thetamin,a)
    sigma_ = sigma(p,e,M,Thetamin,a)
    eta_ = eta(p,e,M,Thetamin,a)
    if plmi == 1:
        eq = np.sqrt((kappa_ * rho_ + 2 * epsilon_ * sigma_ + 2 * np.sqrt(sigma_ * (sigma_ * epsilon_**2 + rho_ * epsilon_ * kappa_ - eta_ * kappa_**2)))/(rho_**2 + 4 * eta_ * sigma_))
    if plmi == -1:
        eq = np.sqrt((kappa_ * rho_ + 2 * epsilon_ * sigma_ - 2 * np.sqrt(sigma_ * (sigma_ * epsilon_**2 + rho_ * epsilon_ * kappa_ - eta_ * kappa_**2)))/(rho_**2 + 4 * eta_ * sigma_))
    return eq

def Lbar_z(p,e,M,Thetamin,a,plmi):
    E_bar = Ebar(p,e,M,Thetamin,a,plmi)
    g_1 = g1(p,e,M,a)
    h_1 = h1(p,e,M,Thetamin,a)
    if plmi == 1:
        eq = ((2 * g_1 * E_bar) + np.sqrt(4 * g_1**2 * E_bar**2 + 4 * h_1 * (f1(p,e,M,Thetamin,a) * E_bar**2 - d1(p,e,M,Thetamin,a))))/(-2 * h_1)
    if plmi == -1:
        eq = ((2 * g_1 * E_bar) - np.sqrt(4 * g_1**2 * E_bar**2 + 4 * h_1 * (f1(p,e,M,Thetamin,a) * E_bar**2 - d1(p,e,M,Thetamin,a))))/(-2 * h_1)
    return eq

def Qbar(p,e,M,Thetamin,a,plmi):
    z_min = zmin(Thetamin)
    eq = z_min**2 * (abar(M,a)**2 * (1 - Ebar(p,e,M,Thetamin,a,plmi)**2) + ((Lbar_z(p,e,M,Thetamin,a,plmi)**2)/(1 - z_min**2)))
    return eq

def E(p,e,M,mu,Thetamin,a,plmi):
    eq = Ebar(p,e,M,Thetamin,a,plmi) * mu
    return eq

def L_z(p,e,M,mu,Thetamin,a,plmi):
    eq = Lbar_z(p,e,M,Thetamin,a,plmi) * mu * M
    return eq

def Q(p,e,M,mu,Thetamin,a,plmi):
    eq = Qbar(p,e,M,Thetamin,a,plmi) * mu**2 * M**2
    return eq

def R(rvar,p,e,M,mu,Thetamin,a,plmi):
    E_ = E(p,e,M,mu,Thetamin,a,plmi)
    L_z_ = L_z(p,e,M,mu,Thetamin,a,plmi)
    Delta = rvar**2 - 2 * M * rvar + a**2
    eq = ((rvar**2 + a**2) * E_ - a * L_z_)**2 - Delta * (mu**2 * rvar**2 + (L_z_ - a * E_)**2 + Q(p,e,M,mu,Thetamin,a,plmi))
    return eq

def Theta(thvar,p,e,M,mu,Thetamin,a,plmi):
    eq = Q(p,e,M,mu,Thetamin,a,plmi) - ((mu**2 - E(p,e,M,mu,Thetamin,a,plmi)**2) * a**2 + ((L_z(p,e,M,mu,Thetamin,a,plmi)**2)/(np.sin(thvar)**2))) * np.cos(thvar)**2
    return eq

def LEP(f,xmin,xmax,n,p,e,M,mu,Thetamin,a,plmi):
    if xmax == xmin:
        return 0
    else:
        dx = (xmax - xmin)/n
        xs = np.arange(xmin,xmax,dx)
        res = 0
        for i in range(n):
            res = res + dx * f(xs[i],p,e,M,mu,Thetamin,a,plmi)
        return res

def p3(p,e,M,mu,Thetamin,a,plmi):
    qua = np.zeros(5)
    E_ = E(p,e,M,mu,Thetamin,a,plmi)
    L_z_ = L_z(p,e,M,mu,Thetamin,a,plmi)
    Q_ = Q(p,e,M,mu,Thetamin,a,plmi)
    qua[0] = (E_**2 -1)
    qua[1] = 2 * M
    qua[2] = (a**2 * (E_**2 - 1) - L_z_**2 - Q_)
    qua[3] = 2 * M * (Q_ + (a * E_ - L_z_)**2)
    qua[4] = -a**2 * Q_
    rts = np.sort(np.roots(qua))
    eq = (rts[1] * (1 - e))/M
    return eq

def p4(p,e,M,mu,Thetamin,a,plmi):
    qua = np.zeros(5)
    E_ = E(p,e,M,mu,Thetamin,a,plmi)
    L_z_ = L_z(p,e,M,mu,Thetamin,a,plmi)
    Q_ = Q(p,e,M,mu,Thetamin,a,plmi)
    qua[0] = (E_**2 -1)
    qua[1] = 2 * M
    qua[2] = (a**2 * (E_**2 - 1) - L_z_**2 - Q_)
    qua[3] = 2 * M * (Q_ + (a * E_ - L_z_)**2)
    qua[4] = -a**2 * Q_
    rts = np.sort(np.roots(qua))
    eq = (rts[0] * (1 + e))/M
    return eq

def invP(var,p,e,M,mu,Thetamin,a,plmi):
    p_3 = p3(p,e,M,mu,Thetamin,a,plmi)
    p_4 = p4(p,e,M,mu,Thetamin,a,plmi)
    eq = (M * np.sqrt(1 - E(p,e,M,mu,Thetamin,a,plmi)**2) * np.sqrt(((p - p_3) - e * (p + p_3 * np.cos(var)))) * np.sqrt(((p - p_4) + e * (p - p_4 * np.cos(var)))))/(1 - e**2)
    inveq = (1/eq)
    return inveq

def wzmin(p,e,M,mu,Thetamin,a,plmi):
    E_ = E(p,e,M,mu,Thetamin,a,plmi)
    Q_ = Q(p,e,M,mu,Thetamin,a,plmi)
    quad = np.zeros(3)
    quad[0] = a**2 * (1 - E_**2)
    quad[1] = -(Q_ + L_z(p,e,M,mu,Thetamin,a,plmi)**2 + a**2 * (1 - E_**2))
    quad[2] = Q_
    rts = np.sort(np.roots(quad))
    eq = rts[0]
    return eq

def wzmax(p,e,M,mu,Thetamin,a,plmi):
    E_ = E(p,e,M,mu,Thetamin,a,plmi)
    Q_ = Q(p,e,M,mu,Thetamin,a,plmi)
    quad = np.zeros(3)
    quad[0] = a**2 * (1 - E_**2)
    quad[1] = -(Q_ + L_z(p,e,M,mu,Thetamin,a,plmi)**2 + a**2 * (1 - E_**2))
    quad[2] = Q_
    rts = np.sort(np.roots(quad))
    eq = rts[1]
    return eq

def beta(p,e,M,mu,Thetamin,a,plmi):
    eq = a**2 * (1 - E(p,e,M,mu,Thetamin,a,plmi)**2)
    return eq

def Lambda_theta(p,e,M,mu,Thetamin,a,plmi):
    wzmax_ = wzmax(p,e,M,mu,Thetamin,a,plmi)
    eq = (4/(np.sqrt(beta(p,e,M,mu,Thetamin,a,plmi) * wzmax_))) * scipy.special.ellipk((wzmin(p,e,M,mu,Thetamin,a,plmi)/wzmax_))
    return eq

def Upsilon_r(p,e,M,mu,Thetamin,a,plmi):
    E_ = E(p,e,M,mu,Thetamin,a,plmi)
    L_z_ = L_z(p,e,M,mu,Thetamin,a,plmi)
    Q_ = Q(p,e,M,mu,Thetamin,a,plmi)
    qua = np.zeros(5)
    qua[0] = (E_**2 -1)
    qua[1] = 2 * M
    qua[2] = (a**2 * (E_**2 - 1) - L_z_**2 - Q_)
    qua[3] = 2 * M * (Q_ + (a * E_ - L_z_)**2)
    qua[4] = -a**2 * Q_
    rts = np.sort(np.roots(qua))
    r1 = rts[3]
    r2 = rts[2]
    r3 = rts[1]
    r4 = rts[0]
    krsqu = ((r1 - r2)/(r1 - r3)) * ((r3 - r4)/(r2 - r4))
    eq = ((np.pi * np.sqrt((1 - E_**2) * (r1 - r3) * (r2 - r4)))/(2 * scipy.special.ellipk(krsqu)))
    return eq

def Lambda_r(p,e,M,mu,Thetamin,a,plmi):
    eq = (2 * np.pi)/Upsilon_r(p,e,M,mu,Thetamin,a,plmi)
    return eq

def Upsilon_theta(p,e,M,mu,Thetamin,a,plmi):
    eq = (2 * np.pi)/Lambda_theta(p,e,M,mu,Thetamin,a,plmi)
    return eq

def T(rvar,thetavar,p,e,M,mu,Thetamin,a,plmi):
    eq = E(p,e,M,mu,Thetamin,a,plmi) * ((((rvar**2 + a**2)**2)/(rvar**2 - 2 * M * rvar + a**2)) - a**2 * np.sin(thetavar)**2) + a * L_z(p,e,M,mu,Thetamin,a,plmi) * (1 - ((rvar**2 + a**2)/(rvar**2 - 2 * M * rvar + a**2)))
    return eq

def Tmod(rvar,zvar,p,e,M,mu,Thetamin,a,plmi):
    eq = E(p,e,M,mu,Thetamin,a,plmi) * ((((rvar**2 + a**2)**2)/(rvar**2 - 2 * M * rvar + a**2)) - a**2 * (1 - zvar)) + a * L_z(p,e,M,mu,Thetamin,a,plmi) * (1 - ((rvar**2 + a**2)/(rvar**2 - 2 * M * rvar + a**2)))
    return eq

def Phi(rvar,thetavar,p,e,M,mu,Thetamin,a,plmi):
    L_z_ = L_z(p,e,M,mu,Thetamin,a,plmi)
    eq = (1/np.sin(thetavar))**2 * L_z_ + a * E(p,e,M,mu,Thetamin,a,plmi) * (((rvar**2 + a**2)/(rvar**2 - 2 * M * rvar + a**2)) - 1) - ((a**2 * L_z_)/(rvar**2 - 2 * M * rvar + a**2))
    return eq

def Phimod(rvar,zvar,p,e,M,mu,Thetamin,a,plmi):
    L_z_ = L_z(p,e,M,mu,Thetamin,a,plmi)
    eq = (1/(1-zvar)) * L_z_ + a * E(p,e,M,mu,Thetamin,a,plmi) * (((rvar**2 + a**2)/(rvar**2 - 2 * M * rvar + a**2)) - 1) - ((a**2 * L_z_)/(rvar**2 - 2 * M * rvar + a**2))
    return eq

def mino_of_chi(chivar,p,e,M,mu,Thetamin,a,plmi):
    beta_ = beta(p,e,M,mu,Thetamin,a,plmi)
    wzmax_ = wzmax(p,e,M,mu,Thetamin,a,plmi)
    wzmin_ = wzmin(p,e,M,mu,Thetamin,a,plmi)
    if chivar < (np.pi/2):
        eq = (1/(np.sqrt(beta_ * wzmax_))) * ((scipy.special.ellipk((wzmin_/wzmax_))) - scipy.special.ellipkinc(((np.pi/2) - chivar),(wzmin_/wzmax_)))
    elif chivar < (np.pi):
        eq = (2/(np.sqrt(beta_ * wzmax_))) * ((scipy.special.ellipk((wzmin_/wzmax_)))) - ((1/(np.sqrt(beta_ * wzmax_))) * ((scipy.special.ellipk((wzmin_/wzmax_))) - scipy.special.ellipkinc(((np.pi/2) - (np.pi - chivar)),(wzmin_/wzmax_))))
    elif chivar < ((3 * np.pi)/2):
        eq = (2/(np.sqrt(beta_ * wzmax_))) * ((scipy.special.ellipk((wzmin_/wzmax_)))) + ((1/(np.sqrt(beta_ * wzmax_))) * ((scipy.special.ellipk((wzmin_/wzmax_))) - scipy.special.ellipkinc(((np.pi/2) - (chivar - np.pi)),(wzmin_/wzmax_))))
    else:
        eq = (4/(np.sqrt(beta_ * wzmax_))) * ((scipy.special.ellipk((wzmin_/wzmax_)))) - ((1/(np.sqrt(beta_ * wzmax_))) * ((scipy.special.ellipk((wzmin_/wzmax_))) - scipy.special.ellipkinc(((np.pi/2) - (2 * np.pi - chivar)),(wzmin_/wzmax_))))
    return eq

def mino_of_psi(psivar,p,e,M,mu,Thetamin,a,plmi,disc):
    eq = LEP(invP,0,psivar,disc,p,e,M,mu,Thetamin,a,plmi)
    return eq

def r_fn(psivar,p,e,M):
    eq = ((p * M)/(1 + e * np.cos(psivar)))
    return eq

def z_of_chi(chivar,p,e,M,mu,Thetamin,a,plmi):
    eq = wzmin(p,e,M,mu,Thetamin,a,plmi) * np.cos(chivar)**2
    return eq

def Tkn(p,e,M,mu,Thetamin,a,plmi,disc):
    Upsilon_r_ = Upsilon_r(p,e,M,mu,Thetamin,a,plmi)
    Upsilon_theta_ = Upsilon_theta(p,e,M,mu,Thetamin,a,plmi)
    Lambda_r_ = Lambda_r(p,e,M,mu,Thetamin,a,plmi)
    Lambda_theta_ = Lambda_theta(p,e,M,mu,Thetamin,a,plmi)
    beta_ = beta(p,e,M,mu,Thetamin,a,plmi)
    wzmax_ = wzmax(p,e,M,mu,Thetamin,a,plmi)
    wzmin_ = wzmin(p,e,M,mu,Thetamin,a,plmi)

    values = np.zeros(1 + 2 * disc, dtype=complex)
    psischis = np.arange(0,2 * np.pi,((2 * np.pi)/disc))
    minochis = [mino_of_chi(psischis[i],p,e,M,mu,Thetamin,a,plmi) for i in range(len(psischis))]
    minopsis = [mino_of_psi(psischis[i],p,e,M,mu,Thetamin,a,plmi,disc) for i in range(len(psischis))]
    intervalues = np.zeros((len(psischis),len(psischis)))

    for i in range(len(psischis)):
        for j in range(len(psischis)):
            intervalues[i][j] = Tmod(r_fn(Upsilon_r_ * minopsis[i],p,e,M),z_of_chi(Upsilon_theta_ * minochis[j],p,e,M,mu,Thetamin,a,plmi),p,e,M,mu,Thetamin,a,plmi) * ((2 * np.pi)/Lambda_r_) * invP(psischis[i],p,e,M,mu,Thetamin,a,plmi) * ((2 * np.pi)/Lambda_theta_) * 1/(np.sqrt(beta_ * (wzmax_ - wzmin_ * (np.cos(psischis[j])**2))))

    def fn(k,n):
        res = 0
        for i in range(disc):
            for j in range(disc):
                res = res + (np.exp(1j * (k * (Upsilon_theta_ * minochis[j]) + n * (Upsilon_r_ * minopsis[i]))) * intervalues[i][j] * ((2 * np.pi)/disc)**2)
        eq = (1/((2 * np.pi)**2)) * res
        return eq

    values[0] = fn(0,0)
    for i in range(1,disc+1):
        values[i] = fn(i,0)
    for i in range(1,disc+1):
        values[i + disc] = fn(0,i)

    return(values)

def Phikn(p,e,M,mu,Thetamin,a,plmi,disc):
    Upsilon_r_ = Upsilon_r(p,e,M,mu,Thetamin,a,plmi)
    Upsilon_theta_ = Upsilon_theta(p,e,M,mu,Thetamin,a,plmi)
    Lambda_r_ = Lambda_r(p,e,M,mu,Thetamin,a,plmi)
    Lambda_theta_ = Lambda_theta(p,e,M,mu,Thetamin,a,plmi)
    beta_ = beta(p,e,M,mu,Thetamin,a,plmi)
    wzmax_ = wzmax(p,e,M,mu,Thetamin,a,plmi)
    wzmin_ = wzmin(p,e,M,mu,Thetamin,a,plmi)

    values = np.zeros(1 + 2 * disc, dtype=complex)
    psischis = np.arange(0,2 * np.pi,((2 * np.pi)/disc))
    minochis = [mino_of_chi(psischis[i],p,e,M,mu,Thetamin,a,plmi) for i in range(len(psischis))]
    minopsis = [mino_of_psi(psischis[i],p,e,M,mu,Thetamin,a,plmi,disc) for i in range(len(psischis))]
    intervalues = np.zeros((len(psischis),len(psischis)))

    for i in range(len(psischis)):
        for j in range(len(psischis)):
            intervalues[i][j] = Phimod(r_fn(Upsilon_r_ * minopsis[i],p,e,M),z_of_chi(Upsilon_theta_ * minochis[j],p,e,M,mu,Thetamin,a,plmi),p,e,M,mu,Thetamin,a,plmi) * ((2 * np.pi)/Lambda_r_) * invP(psischis[i],p,e,M,mu,Thetamin,a,plmi) * ((2 * np.pi)/Lambda_theta_) * 1/(np.sqrt(beta_ * (wzmax_ - wzmin_ * (np.cos(psischis[j])**2))))

    def fn(k,n):
        res = 0
        for i in range(disc):
            for j in range(disc):
                res = res + (np.exp(1j * (k * (Upsilon_theta_ * minochis[j]) + n * (Upsilon_r_ * minopsis[i]))) * intervalues[i][j] * ((2 * np.pi)/disc)**2)
        eq = (1/((2 * np.pi)**2)) * res
        return eq

    values[0] = fn(0,0)
    for i in range(1,disc+1):
        values[i] = fn(i,0)
    for i in range(1,disc+1):
        values[i + disc] = fn(0,i)

    return(values)

def convtest(f,p,e,M,mu,Thetamin,a,plmi):
    values = np.zeros(5)
    xvalues = np.array([0,1,2,3,4])
    di = 4
    for i in range(5):
        values[i] = np.real(f(p,e,M,mu,Thetamin,a,plmi,di)[0])
        di = di * 2
    testval = np.real(f(p,e,M,mu,Thetamin,a,plmi,di)[0])
    comp = [np.log10(np.abs(values[i] - testval)) for i in range(5)]
    plt.plot(xvalues,comp)
    plt.show()
