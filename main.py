import os
from configuration import code_path

# list_opt = ['ABC', 'ABO', 'ACO', 'ACS', 'ALO', 'ASO', 'BA', 'BOA', 'CS', 'DE', 'EPO',
            #  'FA', 'FOA', 'FPA', 'GA', 'GNDO', 'GWO', 'HGSO', 'HHO', 'HLO',
            #    'JA', 'MBO', 'MFO', 'MPA', 'MRFO', 'MVO', 'PFA', 'PRO', 'PSO', 'SA', 'SBO',
            #      'SCA', 'SMA', 'SOS', 'SSA', 'TGA', 'TSA', 'WOA', 'WSA']

list_opt = ['SCA', 'SMA','SOS', 'SSA', 'TGA', 'TSA', 'WOA']
for opt in list_opt:
    try:
        exec(open(code_path+"/"+opt+"_optimization.py").read())
        print(f""+opt+"_optimization executed successfully.\n")
    except Exception as e:
        print(opt+"_optimization")
        print(f"Error executing above optimization: {e}\n")
