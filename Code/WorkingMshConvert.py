import subprocess, os
from dolfin import Mesh, HDF5File, MeshFunction


def convert(msh_file, h5_file):
    '''Convert msh file to h5_file'''
    root, _ = os.path.splitext(msh_file)
    assert os.path.splitext(msh_file)[1] == '.msh'
    assert os.path.splitext(h5_file)[1] == '.h5'

    # Get the xml mesh
    xml_file = '.'.join([root, 'xml'])
    os.system('dolfin-convert %s %s' % (msh_file, xml_file))
    #subprocess.call(['dolfin-convert %s %s' % (msh_file, xml_file)], shell=True)
    # Success?
    assert os.path.exists(xml_file)

    mesh = Mesh(xml_file)
    out = HDF5File(mesh.mpi_comm(), h5_file, 'w')
    out.write(mesh, 'mesh')
                           
    for region in ('facet_region.xml', ):
        name, _ = region.split('_')
        r_xml_file = '_'.join([root, region])
        
        f = MeshFunction('size_t', mesh, r_xml_file)
        out.write(f, name)

    # Sucess?
    assert os.path.exists(h5_file)

    return mesh
    

def cleanup(files=None, exts=()):
    '''Get rid of xml'''
    if files is not None:
        return map(os.remove, files)
    else:
        files = filter(lambda f: any(map(f.endswith, exts)), os.listdir('.'))
        print('Removing', files)
        return cleanup(files)
                    