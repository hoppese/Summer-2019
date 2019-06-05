import sys
import os
from datetime import datetime

def print_message(params, message):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    params['runlog'].write(time + ': ' + message + '\n')

def output_data_1D(params, filename, xs, u):

    if(len(u) != len(xs)):
        print_message(params, 'Arrays of data passed to output_data_1D are of inconsisent size.')
        exit(6)

    file = open(params['output_directory']+'/Data/'+filename,'w')
    print_message(params, 'Beginning outputting data to ' + filename)

    for i in range(len(xs)):
        file.write(str(xs[i]) + '\t' + str(u[i]) + '\n')

    file.close()
    print_message(params, 'Finished outputting data to ' + filename)

    return

    
