def rk4step(array, dt, xmin, xmax, nk, npe, a, alpha, dudtrhs):
    w1 = dudtrhs(array, xmin, xmax, nk, npe, a, alpha)
    w2 = dudtrhs(array + 0.5 * dt * w1, xmin, xmax, nk, npe, a, alpha)
    w3 = dudtrhs(array + 0.5 * dt * w2, xmin, xmax, nk, npe, a, alpha)
    w4 = dudtrhs(array + dt * w3, xmin, xmax, nk, npe, a, alpha)

    return array + (dt / 6) * (w1 + 2 * w2 + 2 * w3 + w4)

def fe_step(array, dt, dx, dudtrhs):
    newarray = array + dt * dudtrhs(array, dx)
    return newarray

def rk2step(array, dt, dx, dudtrhs):
    w1 = dudtrhs(array, dx)
    w2 = dudtrhs(array + 0.5 * dt * w1, dx)
    return array + dt * w2
