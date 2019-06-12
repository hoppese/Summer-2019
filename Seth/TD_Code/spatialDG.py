import numpy as np
import math
import scipy

def ReferenceElement(N):
    """Given the polynomial order N, initialize quantites of a reference-element
    in logical coordinate -1<=r<=1."""

    # Legendre polynomials are Jacobi(alpha,beta) polynomials
    alpha=0.; beta=0;

    # Gauss-Lobatto quadrature points for Legendre polynomials
    # According to Hesthaven+Warburton (p. 448, JacobGL.m) these
    # are the order N-2 Gauss quadrature points of Jacobi polynomials
    # with different alpha, beta, enlarged by end-points -1, +1.
    #
    # note len(r)=N, i.e. matrices defined below (V, Vr, etc) are square
    if(N==1):
        r=np.array([-1.,1.])
    else:
        # (N-2)-th order quadrature points are roots of (N-1)-st Jacobi polynomial
        inner_roots, inner_weights = scipy.special.roots_jacobi(N-1, alpha+1, beta+1)
        r = np.concatenate([ [-1.], inner_roots, [1.] ])

    # Vandermonde matrix for Legendre polynomials
    # V[i,j] = P_j(r_i),  j=0,...,N,  i=0,...,len(r)-1
    V = np.zeros( (len(r), N+1))
    for j in range(N+1):
        # scipy normalization determined by trial and error.
        # For **LAGRANGE POLY** ONLY, not general alpha, beta.
        # This makes the returned polynomials orthonormal
        normalization = np.sqrt((1.+2.*j)/2.)
        V[:,j] = scipy.special.eval_jacobi(j, alpha, beta, r)*normalization
        # or V[:,j] = scipy.special.legendre(j)(r)

        # check normalization
        # tmp_r, tmp_w = scipy.special.roots_jacobi(j+1, alpha, beta)
        # tmp_L=scipy.special.eval_jacobi(j, alpha, beta, tmp_r)*normalization
        # L_dot_L = sum(tmp_w*tmp_L*tmp_L)
        # print("j={}, (L,L)={}".format(j, L_dot_L))


    Vinv=np.linalg.inv(V)

    # derivatives of Legendre polynomials, evaluated at quadrature points
    # Vr[i,j] = dP_j/dr(r_i),  j=0,...,N,  i=0,...,len(r)-1
    #   use dP_j/dr = sqrt(j(j+1)) J^{alpha+1,beta+1}_{j-1}  (H+W, Eq A2)
    #
    Vr = np.zeros( (len(r), N+1))
    for j in range(1,N+1):
        # scipy normalization determined by trial and error.
        # For **LAGRANGE POLY** ONLY, not general alpha, beta.
        # This makes the returned polynomials orthonormal, conforming
        # to H+W conventions
        scipy_normalization=np.sqrt((1.+2.*j)*(j+1.)/(8.*j))
        normed_J = scipy.special.jacobi(j-1, alpha+1, beta+1)(r)*scipy_normalization
        Vr[:,j] = np.sqrt(j*(j+alpha+beta+1.))*normed_J  # H+W Eq. A2

        # - check normalization
        # - integrate by Legendre quadrature, to explicitly show weight-function in orthogonality
        # tmp_r, tmp_w = scipy.special.roots_jacobi(j+4, alpha, beta)
        # tmp_L=scipy.special.eval_jacobi(j-1, alpha+1, beta+1, tmp_r)*scipy_normalization
        # - evaluate orthogonality; note weight function (1-r)(1+r)
        # L_dot_L = sum(tmp_w*tmp_L*tmp_L*(1-tmp_r)*(1+tmp_r))
        # print("j={}, (L,L)={}".format(j, L_dot_L))


    # derivatives of Lagrange interpolating polynomials
    #    Dr(i,j) = dl_j/dr(r=r_i),
    # where  l_j(r_i) = delta_{ij}
    # compute using P_j(r) = V[i,j]*l_i(r) =>  V[i,j] dl_i/dr = dP_j/dr     (*)
    #     => V^{-T} V^T[j,i] dl_i/dr = V^{-T} dP_j/dr
    Dr = np.matmul(Vr,Vinv)

    # inverse of mass-matrix
    # Using (*), one can show  M = (V V^T)^(-1)
    # Turns out that the inverse of M is used in the DG algorithm,
    # and so we can directly compute M-inverse, without computing
    # matrix-inverses:
    Minv = np.matmul(V, V.transpose())

    # finally, M^{-1}S = Dr, and since we need S only multiplied by M^{-1},
    # we can just return Dr
    MinvS=Dr

    return r, Minv, MinvS

def du_dx_DG(u, xs, K, Np, Dr, Minv):
    if K * Np != len(xs) or K * Np != len(u):
        print('In du_dx_DG, inconsistent lengths of u, xs, Np, and K. Aborting.')
        exit(7)

    if Np != len(Dr) or Np != len(Minv):
        print('In du_dx_DG, matrices Dr and Minv must be have Np columns. Aborting.')
        exit(8)

    uks    = [u[k*Np:Np*(k+1)] for k in range(K)]
    xks    = [xs[k*Np:Np*(k+1)] for k in range(K)]

    uk_rs  = [u[(k+1)*Np] if k != K-1 else u[0] for k in range(K)]
    uk_ls  = [u[k*Np - 1] if k != 0 else u[-1] for k in range(K)]

    hs     = [xks[k][-1] - xks[k][0] for k in range(K)]

    Minvks = [2*Minv/hs[k] for k in range(K)]
    Drks   = [2*Dr/hs[k] for k in range(K)]

    return np.concatenate([deriv_DG_element(uks[k], uk_rs[k], uk_ls[k], Drks[k], Minvks[k]) for k in range(K)])

def du_dx_DG_dict(u, dict):
    return du_dx_DG(u, dict['xs'], dict['K'], dict['Np'], dict['Dr'], dict['Minv'])


def deriv_DG_element(uk, uk_r, uk_l, Drk, Minvk):
    u_star_l = numerical_flux(uk[0], uk_l, 1, 1/2)
    u_star_r = numerical_flux(uk_r, uk[-1], 1, 1/2)

    return np.dot(Drk,uk) - Minvk[:,-1]*(uk[-1] - u_star_r) + Minvk[:,0]*(uk[0] - u_star_l)


def numerical_flux(ur, ul, a, alpha):
    u_diff = -ul + ur
    u_avg  = (ul + ur)/2

    return a * u_avg + np.abs(a)*(1-alpha)*u_diff/2


def xs_tot(x_min,x_max,K,rs):
    dx = (x_max - x_min)/K
    return np.concatenate([xs_from_rs(rs,x_min+dx*k,x_min+(k+1)*dx) for k in range(K)])


def xs_from_rs(rs,x_min,x_max):
    return x_min + (1+rs)*(x_max - x_min)/2
