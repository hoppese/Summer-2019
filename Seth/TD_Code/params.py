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

    if params['output_directory'] == 'CWD':
        params['output_directory'] = os.getcwd() + '/' + param_file_base

    if os.path.isdir(params['output_directory']):
    	print('Directory ' + params['output_directory'] + ' already exists')
    	exit(3)

    params['file'] = param_file
    params['directory'] = param_file_base
    os.makedirs(params['output_directory'])
    os.makedirs(params['output_directory']+'/Runlogs/')
    os.makedirs(params['output_directory']+'/Data/')
    os.makedirs(params['output_directory']+'/Source/')

    params['runlog'] = open(params['output_directory']+'/Runlogs/output.out','w')

    # output_source_code(params)

    return params

def default_params():
    return {'x_min': 0,
            'x_max': 1,
            'nx': 100,
            'K': 4,
            'Np': 5,
            'CF': 1e-5,
            't_init': 0,
            't_final': .1,
            'output_data': True,
            'equation': 'advec_l2r',
            'method': 'FD-C4',
            'output_directory': 'CWD'}

def import_params(file):
    with open(file, 'r') as yamlfile:
        return yaml.load(yamlfile)

# def output_source_code(params):
