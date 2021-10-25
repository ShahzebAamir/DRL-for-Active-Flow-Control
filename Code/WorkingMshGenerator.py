import os, subprocess
from numpy import deg2rad
from printind.printind_function import printiv


def generate_mesh(args, template='geometry_2d.template_geo', dim=2):
    '''Modify template according args and make gmsh generate the mesh'''
    assert os.path.exists(template)
    args = args.copy()

    printiv(template)

    with open(template, 'r') as f: old = f.readlines()
    # Chop the file to replace the jet positions
    split = list(map(lambda s: s.startswith('DefineConstant'), old)).index(True)
    #jet_positions = deg2rad(map(float, args.pop('jet_positions')))
    jet_positions = map(lambda x: deg2rad(x), args.pop('jet_positions'))
    jet_positions = 'jet_positions[] = {%s};\n' % (', '.join(map(str, jet_positions)))
    print(jet_positions)
    body = ''.join([jet_positions] + old[split:])

    output = args.pop('output')
    printiv(output)

    if not output:
        output = template
    assert os.path.splitext(output)[1] == '.geo'

    with open(output, 'w') as f: f.write(body)
        
    args['jet_width'] = deg2rad(args['jet_width'])

    scale = args.pop('clscale')

    cmd = 'gmsh -0 %s ' % output

    list_geometric_parameters = ['width', 'jet_radius', 'jet_width', 'box_size', 'length',
                                 'bottom_distance', 'cylinder_size', 'front_distance'
                                ]

    constants = " "

    for crrt_param in list_geometric_parameters:
        constants = constants + " -setnumber " + crrt_param + " " + str(args[crrt_param])

    # Unrolled model
    subprocess.call(cmd + constants, shell=True)

    unrolled = '_'.join([output, 'unrolled'])
    #assert os.path.exists(unrolled)

    return subprocess.call(['gmsh -%d -clscale %g %s' % (dim, scale, unrolled)], shell=True)