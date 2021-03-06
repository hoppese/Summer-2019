import yaml
import sys
import os
from scipy.integrate import odeint

def setup_run(argv):
    if not(len(argv) == 2):
    	print('Usage: TD_driver param_file.yaml')
    	exit(1)

    param_file = argv[1]
    param_file_base, param_file_ext = os.path.splitext(param_file)

    if param_file_ext != '.yaml':
    	print(param_file + ' not a parfile')
    	exit(2)

    params_import = import_params(param_file)
    params = dict(default_params(), **params_import)

    return params

def default_params():
    return {'t_0': 0,
            'phi_0': 0,
            'p': 10,
            'e': 0.7,
            'M': 1,
            'mu': 1,
            'Thetamin': .5,
            'a': 0.1,
            'plmi': 1,
            'disc': 20}

def import_params(file):
    with open(file, 'r') as yamlfile:
        return yaml.load(yamlfile)
