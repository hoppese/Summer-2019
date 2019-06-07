import numpy as np
import scipy
from scipy import special
from matplotlib import pyplot as plt
%matplotlib inline

# Midpoint Derivative
def centerdudx(u, info):
    du = np.zeros(len(u))
    du[1:-1] = u[2:] - u[0:-2]
    du[0] = u[1] - u[-1]
    du[-1] = u[0] - u[-2]

    return du / (2* info['dx'])


# Fourier Transform Derivative
def fourierderivative(u, info):
    fouru = np.fft.fft(u * 2 / (len(u)))
    fouru = np.conjugate(fouru)

    fouru = -2 * np.pi * 1j * fouru

    klist = []
    for i in range(int((len(u)+1)/2)):
        klist.append(i)
    for i in range(int(len(u)/2), 0, -1):
        klist.append(-1 * i)
    derfouru = fouru * np.array(klist)

    der = np.fft.ifft(np.conjugate(derfouru) / 2 / info['dx'])
    return -1 * der


# Discontinuous Galerkin Derivative
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

#     inverse of mass-matrix
#     Using (*), one can show  M = (V V^T)^(-1)
#     Turns out that the inverse of M is used in the DG algorithm,
#     and so we can directly compute M-inverse, without computing
#     matrix-inverses:
    Minv = np.matmul(V, V.transpose())

    # finally, M^{-1}S = Dr, and since we need S only multiplied by M^{-1},
    # we can just return Dr
    MinvS=Dr

    return r, Minv, MinvS

def derive_DG_elem(uk, ur, ul, MkinvSk, Mkinv, a, alpha):

    MkinvSkuk = np.dot(MkinvSk, uk)

    duk_dx = MkinvSkuk\
        - Mkinv[:, -1] * (uk[-1] - (a * (uk[-1] + ul) / 2 + abs(a) * ((1 - alpha) / 2) * (uk[-1] - ul)))\
        + Mkinv[:, 0] * (uk[0] - (a * (uk[0] + ur) / 2 + abs(a) * ((1 - alpha) / 2) * (ur - uk[0])))

    return duk_dx


def dudtDG(inarray, info):
    return -1 * dudxDG(inarray, info)

def dudxDG(inarray, info):

    MkinvSk = info['MinvS_ref'][:] * 2 / ((info['maxx'] - info['minx']) / info['nk'])
    Mkinv = info['Minv_ref'][:] * 2 / ((info['maxx'] - info['minx']) / info['nk'])

    u_k = np.zeros((info['nk'],info['npe']))
    for i in range(info['nk']):
        u_k[i,:] = inarray[i * info['npe']:(i + 1) * info['npe']]

    MkinvSkuk = np.zeros((info['nk'],info['npe']))
    for i in range(len(u_k)):
        MkinvSkuk[i] = np.dot(MkinvSk, u_k[i])
    dudx = np.zeros((info['nk'], info['npe']))

    dudx[0,:] = derive_DG_elem(u_k[0], u_k[-1, -1], u_k[1,0], MkinvSk, Mkinv, info['a'], info['alpha'])
    for i in range(1, info['nk'] -1):
        dudx[i,:] = derive_DG_elem(u_k[i], u_k[i-1, -1], u_k[i+1, 0], MkinvSk, Mkinv, info['a'], info['alpha'])

    dudx[-1, :] = derive_DG_elem(u_k[-1], u_k[-2,-1], u_k[0,0], MkinvSk, Mkinv, info['a'], info['alpha'])

    dudxlist = [dudx[i,j] for i in range(info['nk']) for j in range(info['npe'])]
    dudxarray = np.array(dudxlist)
    return dudxarray
