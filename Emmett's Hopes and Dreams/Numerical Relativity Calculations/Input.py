default_info = {'minx':0, 'maxx': 1, 'xvalues':[1], 'dx': 0, 'nk':0, 'npe':0, 'a':1, 'alpha':1/2, 'r': [1], 'Minv_ref':[1], 'MinvS_ref':[1]}


def DG_dict(default_info, minx, maxx, nk ,npe, a, alpha):
    info = default_info
    r , Minv_ref, MinvS_ref = ReferenceElement(npe - 1)

    info['minx'] = minx
    info['maxx'] = maxx
    info['nk']   = nk
    info['npe']  = npe
    info['a']    = a
    info['alpha'] = alpha
    info['r'] = r
    info['Minv_ref'] = Minv_ref
    info['MinvS_ref'] = MinvS_ref

    xvalues = []
    for i in range(0, nk):
        for j in ((r+1) * ((maxx - minx) / nk) / 2 + (i * (maxx - minx)) / nk):
            xvalues.append(j)

    info['xvalues'] = np.array(xvalues)
    return info

def FE_Spectral_dict(default_info, minx, maxx, nx):
    info = default_info

    info['dx']   = (maxx - minx) / nx
    info['minx'] = minx
    info['maxx'] = maxx
    info['xvalues'] = np.linspace(minx, maxx, nx, endpoint = False)
    return info
