import numpy as np
import scipy
import mpmath
from scipy import fftpack
import yaml
import sys
import os
import math
import collections
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def Sigma(r,theta,a):
    return r**2 + a**2 * np.cos(theta)**2

def Delta(r,a,M):
    return r**2 - 2 * M * r + a**2

def sidebeta(r,a):
    return np.sqrt(r**2 + a**2)

def signa(a):
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0

def abar(M,a):
    return a/M

def zmin(Thetamin):
    return np.cos(Thetamin)

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
    if e != 0:
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
    else:
        return 0

def Lbar_z(p,e,M,Thetamin,a,plmi):
    if e != 0:
        E_bar = Ebar(p,e,M,Thetamin,a,plmi)
        g_1 = g1(p,e,M,a)
        h_1 = h1(p,e,M,Thetamin,a)
        if plmi == 1:
            eq = ((2 * g_1 * E_bar) + np.sqrt(4 * g_1**2 * E_bar**2 + 4 * h_1 * (f1(p,e,M,Thetamin,a) * E_bar**2 - d1(p,e,M,Thetamin,a))))/(-2 * h_1)
        if plmi == -1:
            eq = ((2 * g_1 * E_bar) - np.sqrt(4 * g_1**2 * E_bar**2 + 4 * h_1 * (f1(p,e,M,Thetamin,a) * E_bar**2 - d1(p,e,M,Thetamin,a))))/(-2 * h_1)
        return eq
    else:
        return 0

def Qbar(p,e,M,Thetamin,a,plmi):
    z_min = zmin(Thetamin)
    eq = z_min**2 * (abar(M,a)**2 * (1 - Ebar(p,e,M,Thetamin,a,plmi)**2) + ((Lbar_z(p,e,M,Thetamin,a,plmi)**2)/(1 - z_min**2)))
    return eq

def cursiveE(p,e,M,mu,Thetamin,a,plmi):
    return Ebar(p,e,M,Thetamin,a,plmi) * mu

def cursiveL_z(p,e,M,mu,Thetamin,a,plmi):
    return Lbar_z(p,e,M,Thetamin,a,plmi) * mu * M

def cursiveQ(p,e,M,mu,Thetamin,a,plmi):
    return Qbar(p,e,M,Thetamin,a,plmi) * mu**2 * M**2

def Psir(r,p,e,M,mu,Thetamin,a,plmi):
    D = Delta(r,a,M)
    return a * cursiveE(p,e,M,mu,Thetamin,a,plmi) * (((sidebeta(r,a)**2)/D) - 1) - ((a**2 * cursiveL_z(p,e,M,mu,Thetamin,a,plmi))/D)

def Psitheta(theta,p,e,M,mu,Thetamin,a,plmi):
    return cursiveL_z(p,e,M,mu,Thetamin,a,plmi) * (1/(np.sin(theta)**2))

def Psicostheta(costheta,p,e,M,mu,Thetamin,a,plmi):
    return cursiveL_z(p,e,M,mu,Thetamin,a,plmi) * (1/(1 - costheta**2))

def Tr(r,p,e,M,mu,Thetamin,a,plmi):
    sb = sidebeta(r,a)
    D = Delta(r,a,M)
    return cursiveE(p,e,M,mu,Thetamin,a,plmi) * ((sb**4)/D) + a * cursiveL_z(p,e,M,mu,Thetamin,a,plmi) * (1 - ((sb**2)/D))

def Ttheta(theta,p,e,M,mu,Thetamin,a,plmi):
    return -a**2 * cursiveE(p,e,M,mu,Thetamin,a,plmi) * np.sin(theta)**2

def Tcostheta(costheta,p,e,M,mu,Thetamin,a,plmi):
    return -a**2 * cursiveE(p,e,M,mu,Thetamin,a,plmi) * (1 - costheta**2)

def r1(p,e,M):
    return ((p * M)/(1 - e))

def r2(p,e,M):
    return ((p * M)/(1 + e))

def p3(p,e,M,mu,Thetamin,a,plmi):
    if e != 0:
        qua = np.zeros(5)
        E = cursiveE(p,e,M,mu,Thetamin,a,plmi)
        L_z = cursiveL_z(p,e,M,mu,Thetamin,a,plmi)
        Q = cursiveQ(p,e,M,mu,Thetamin,a,plmi)
        qua[0] = (E**2 -1)
        qua[1] = 2 * M
        qua[2] = (a**2 * (E**2 - 1) - L_z**2 - Q)
        qua[3] = 2 * M * (Q + (a * E - L_z)**2)
        qua[4] = -a**2 * Q
        rts = np.sort(np.roots(qua))
        return ((rts[1] * (1 - e))/M)
    else:
        return ((r1(p,e,M) * (1 - e))/M)

def p4(p,e,M,mu,Thetamin,a,plmi):
    if e != 0:
        qua = np.zeros(5)
        E = cursiveE(p,e,M,mu,Thetamin,a,plmi)
        L_z = cursiveL_z(p,e,M,mu,Thetamin,a,plmi)
        Q = cursiveQ(p,e,M,mu,Thetamin,a,plmi)
        qua[0] = (E**2 -1)
        qua[1] = 2 * M
        qua[2] = (a**2 * (E**2 - 1) - L_z**2 - Q)
        qua[3] = 2 * M * (Q + (a * E - L_z)**2)
        qua[4] = -a**2 * Q
        rts = np.sort(np.roots(qua))
        return ((rts[0] * (1 + e))/M)
    else:
        return ((r1(p,e,M) * (1 + e))/M)

def rp(psi,p,e,M):
    return (p * M)/(1 + e * np.cos(psi))

def costhetap(chi,p,e,M,mu,Thetamin,a,plmi):
    return np.sqrt(zminus(p,e,M,mu,Thetamin,a,plmi)) * np.cos(chi)

def zminus(p,e,M,mu,Thetamin,a,plmi):
    L_z = cursiveL_z(p,e,M,mu,Thetamin,a,plmi)
    Q = cursiveQ(p,e,M,mu,Thetamin,a,plmi)
    Beta = a**2 * (1 - cursiveE(p,e,M,mu,Thetamin,a,plmi)**2)
    return (L_z**2 + Q + Beta - np.sqrt((L_z**2 + Q + Beta)**2 - 4 * Q * Beta))/(2 * Beta)

def zplus(p,e,M,mu,Thetamin,a,plmi):
    L_z = cursiveL_z(p,e,M,mu,Thetamin,a,plmi)
    Q = cursiveQ(p,e,M,mu,Thetamin,a,plmi)
    Beta = a**2 * (1 - cursiveE(p,e,M,mu,Thetamin,a,plmi)**2)
    return (L_z**2 + Q + Beta + np.sqrt((L_z**2 + Q + Beta)**2 - 4 * Q * Beta))/(2 * Beta)

def Pr_of_psi(psi,p,e,M,mu,Thetamin,a,plmi):
    Beta = a**2 * (1 - cursiveE(p,e,M,mu,Thetamin,a,plmi)**2)
    p_3 = p3(p,e,M,mu,Thetamin,a,plmi)
    p_4 = p4(p,e,M,mu,Thetamin,a,plmi)
    topline = ((p - p_4) + e * (p - p_4 * np.cos(psi)))**(-0.5) * a * (1 - e**2)
    botline = ((p - p_3) - e * (p + p_3 * np.cos(psi)))**(1/2) * M * Beta**(1/2)
    return (topline/botline)

def Ptheta_of_chi(chi,p,e,M,mu,Thetamin,a,plmi):
    Beta = a**2 * (1 - cursiveE(p,e,M,mu,Thetamin,a,plmi)**2)
    return ((Beta * (zplus(p,e,M,mu,Thetamin,a,plmi) - zminus(p,e,M,mu,Thetamin,a,plmi) * (np.cos(chi)**2)))**(-1/2))

def cursivePr_n(p,e,M,mu,Thetamin,a,plmi,disc):
    dpsi = (np.pi)/(disc - 1)
    psis_ = np.arange(0,np.pi + dpsi,dpsi)
    if len(psis_) > (disc):
        psis = np.zeros(disc)
        for i in range(disc):
            psis[i] = psis_[i]
    else:
        psis = psis_

    f = np.zeros(disc)
    f[:] = Pr_of_psi(psis[:],p,e,M,mu,Thetamin,a,plmi)
    return scipy.fftpack.dct(f,type=1) * (1/2) * np.sqrt(2/(disc-1))
#     return dct(Pr_of_phi,p,e,M,mu,Thetamin,a,plmi,disc)

def cursivePtheta_n(p,e,M,mu,Thetamin,a,plmi,disc):
    dchi = (np.pi)/(disc - 1)
    chis_ = np.arange(0,np.pi + dchi,dchi)
    if len(chis_) > (disc):
        chis = np.zeros(disc)
        for i in range(disc):
            chis[i] = chis_[i]
    else:
        chis = chis_

    f = np.zeros(disc)
    f[:] = Ptheta_of_chi(chis[:],p,e,M,mu,Thetamin,a,plmi)
    return scipy.fftpack.dct(f,type=1) * (1/2) * np.sqrt(2/(disc-1))
#     return dct(Ptheta_of_chi,p,e,M,mu,Thetamin,a,plmi,disc)

def lambdar_of_psi(psi,p,e,M,mu,Thetamin,a,plmi,disc):
    coefs = cursivePr_n(p,e,M,mu,Thetamin,a,plmi,disc)
    sum = 0
    for i in range(1,disc - 1):
        sum = sum + coefs[i] * np.sin((i * psi)/i)
    return np.sqrt(2/(disc-1)) * (.5 * psi * coefs[0] + .5 * coefs[disc - 1] * ((np.sin((disc - 1) * psi))/(disc - 1)) + sum)

def lambdatheta_of_chi(chi,p,e,M,mu,Thetamin,a,plmi,disc):
    coefs = cursivePtheta_n(p,e,M,mu,Thetamin,a,plmi,disc)
    sum = 0
    for i in range(1,disc - 1):
        sum = sum + coefs[i] * np.sin((i * chi)/i)
    return np.sqrt(2/(disc-1)) * (.5 * chi * coefs[0] + .5 * coefs[disc - 1] * ((np.sin((disc - 1) * chi))/(disc - 1)) + sum)

def Lambdar(p,e,M,mu,Thetamin,a,plmi,disc):
    coefs = cursivePr_n(p,e,M,mu,Thetamin,a,plmi,disc)
    return np.pi * coefs[0] * np.sqrt(2/(disc-1))

def Lambdatheta(p,e,M,mu,Thetamin,a,plmi,disc):
    coefs = cursivePtheta_n(p,e,M,mu,Thetamin,a,plmi,disc)
    return np.pi * coefs[0] * np.sqrt(2/(disc-1))

def Upsilonr(p,e,M,mu,Thetamin,a,plmi,disc):
    return (2 * np.pi)/Lambdar(p,e,M,mu,Thetamin,a,plmi,disc)

def Upsilontheta(p,e,M,mu,Thetamin,a,plmi,disc):
    return (2 * np.pi)/Lambdatheta(p,e,M,mu,Thetamin,a,plmi,disc)

def cursiveT_n(n,p,e,M,mu,Thetamin,a,plmi,disc):
    dpsi = ((2*np.pi)/disc)
    psis = np.arange(0,(2*np.pi),dpsi)
    res = 0
    for i in range(disc):
        res = res + dpsi * Tr(rp(psis[i],p,e,M),p,e,M,mu,Thetamin,a,plmi) * Pr_of_psi(psis[i],p,e,M,mu,Thetamin,a,plmi) * np.exp(1j * n * Upsilonr(p,e,M,mu,Thetamin,a,plmi,disc) * lambdar_of_psi(psis[i],p,e,M,mu,Thetamin,a,plmi,disc))
    return res * (1/Lambdar(p,e,M,mu,Thetamin,a,plmi,disc))


def cursiveT_k(k,p,e,M,mu,Thetamin,a,plmi,disc):

    dchi = ((2*np.pi)/disc)
    chis = np.arange(0,(2*np.pi),dchi)
    res = 0
    for i in range(disc):
        res = res + dchi * Tcostheta(costhetap(chis[i],p,e,M,mu,Thetamin,a,plmi),p,e,M,mu,Thetamin,a,plmi) * Ptheta_of_chi(chis[i],p,e,M,mu,Thetamin,a,plmi) * np.exp(1j * k * Upsilontheta(p,e,M,mu,Thetamin,a,plmi,disc) * lambdatheta_of_chi(chis[i],p,e,M,mu,Thetamin,a,plmi,disc))
    return res * (1/Lambdatheta(p,e,M,mu,Thetamin,a,plmi,disc))

def cursiveo_n(n,p,e,M,mu,Thetamin,a,plmi,disc):
    dpsi = ((2*np.pi)/disc)
    psis = np.arange(0,(2*np.pi),dpsi)
    res = 0
    for i in range(disc):
        res = res + dpsi * Psir(rp(psis[i],p,e,M),p,e,M,mu,Thetamin,a,plmi) * Pr_of_psi(psis[i],p,e,M,mu,Thetamin,a,plmi) * np.exp(1j * n * Upsilonr(p,e,M,mu,Thetamin,a,plmi,disc) * lambdar_of_psi(psis[i],p,e,M,mu,Thetamin,a,plmi,disc))
    return res * (1/Lambdar(p,e,M,mu,Thetamin,a,plmi,disc))

def cursiveo_k(k,p,e,M,mu,Thetamin,a,plmi,disc):
    dchi = ((2*np.pi)/disc)
    chis = np.arange(0,(2*np.pi),dchi)
    res = 0
    for i in range(disc):
        res = res + dchi * Psicostheta(costhetap(chis[i],p,e,M,mu,Thetamin,a,plmi),p,e,M,mu,Thetamin,a,plmi) * Ptheta_of_chi(chis[i],p,e,M,mu,Thetamin,a,plmi) * np.exp(1j * k * Upsilontheta(p,e,M,mu,Thetamin,a,plmi,disc) * lambdatheta_of_chi(chis[i],p,e,M,mu,Thetamin,a,plmi,disc))
    return res * (1/Lambdatheta(p,e,M,mu,Thetamin,a,plmi,disc))

def deltr(mino,p,e,M,mu,Thetamin,a,plmi,disc):
    if disc % 2 == 1:
        print("N_r must be even")
        raise ValueError
    else:
        Upsilonr_ = Upsilonr(p,e,M,mu,Thetamin,a,plmi,disc)
        res = 0
        for i in range(1,(disc//2) + 1):
            res = res + ((1j * cursiveT_n(i,p,e,M,mu,Thetamin,a,plmi,disc))/(i * Upsilonr_)) * np.exp(-1 * 1j * i * Upsilonr_ * mino)
        return 2 * np.real(res)

def delttheta(mino,p,e,M,mu,Thetamin,a,plmi,disc):
    if disc % 2 == 1:
        print("N_r must be even")
        raise ValueError
    else:
        Upsilont_ = Upsilontheta(p,e,M,mu,Thetamin,a,plmi,disc)
        res = 0
        for i in range(1,(disc//2) + 1):
            res = res + ((1j * cursiveT_k(i,p,e,M,mu,Thetamin,a,plmi,disc))/(i * Upsilont_)) * np.exp(-1 * 1j * i * Upsilont_ * mino)
        return 2 * np.real(res)

def delphir(mino,p,e,M,mu,Thetamin,a,plmi,disc):
    if disc % 2 == 1:
        print("N_r must be even")
        raise ValueError
    else:
        Upsilonr_ = Upsilonr(p,e,M,mu,Thetamin,a,plmi,disc)
        res = 0
        for i in range(1,(disc//2) + 1):
            res = res + ((1j * cursiveo_n(i,p,e,M,mu,Thetamin,a,plmi,disc))/(i * Upsilonr_)) * np.exp(-1 * 1j * i * Upsilonr_ * mino)
        return 2 * np.real(res)

def delphitheta(mino,p,e,M,mu,Thetamin,a,plmi,disc):
    if disc % 2 == 1:
        print("N_r must be even")
        raise ValueError
    else:
        Upsilont_ = Upsilontheta(p,e,M,mu,Thetamin,a,plmi,disc)
        res = 0
        for i in range(1,(disc//2) + 1):
            res = res + ((1j * cursiveo_k(i,p,e,M,mu,Thetamin,a,plmi,disc))/(i * Upsilont_)) * np.exp(-1 * 1j * i * Upsilont_ * mino)
        return 2 * np.real(res)

def t_of_mino(mino,t_0,p,e,M,mu,Thetamin,a,plmi,disc):
    Gamma = np.real(cursiveT_n(0,p,e,M,mu,Thetamin,a,plmi,disc) + cursiveT_k(0,p,e,M,mu,Thetamin,a,plmi,disc))
    return Gamma * mino + deltr(mino,p,e,M,mu,Thetamin,a,plmi,disc) + delttheta(mino,p,e,M,mu,Thetamin,a,plmi,disc) + t_0

def phi_of_mino(mino,phi_0,p,e,M,mu,Thetamin,a,plmi,disc):
    Upsilonphi = np.real(cursiveo_n(0,p,e,M,mu,Thetamin,a,plmi,disc) + cursiveo_k(0,p,e,M,mu,Thetamin,a,plmi,disc))
    return Upsilonphi * mino + delphir(mino,p,e,M,mu,Thetamin,a,plmi,disc) + delphitheta(mino,p,e,M,mu,Thetamin,a,plmi,disc) + phi_0
