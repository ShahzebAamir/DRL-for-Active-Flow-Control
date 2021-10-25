import numpy as np
import dolfin
import iUFL.compilation


def eval_grad_foo(foo):
    '''Exact gradient of function'''

    def evaluate(x, foo=foo):
        x = np.fromiter(x, dtype=float)

        V = foo.function_space()
        el = V.element()
        mesh = V.mesh()
        # Find the cell with point
        x_point = dolfin.Point(*x) 
        cell_id = mesh.bounding_box_tree().compute_first_entity_collision(x_point)
        cell = dolfin.Cell(mesh, cell_id)
        coordinate_dofs = cell.get_vertex_coordinates()

        shape = dolfin.grad(foo).ufl_shape
        # grad grad grad ... ntime is stored in
        nslots = np.prod(shape)

        # Array for values with derivatives of all basis functions
        values = np.zeros(nslots*el.space_dimension(), dtype=float)
        # Compute all derivatives
        el.evaluate_basis_derivatives_all(1, values, x, coordinate_dofs, cell.orientation())
        # Reshape such that colums are [d/dxx, d/dxy, d/dyx, d/dyy, ...]
        values = values.reshape((-1, nslots))

        # Get expansion coefs on cell. Alternative to this is f.restrict(...)
        dofs = V.dofmap().cell_dofs(cell_id)
        dofs = foo.vector()[dofs]

        # Perform reduction on each colum - you get that deriv of foo
        values = np.array([np.inner(row, dofs) for row in values.T])
        values = values.reshape(shape)

        return values
    return evaluate


def eval_grad_expr(foo, mesh):
    '''Gradient of polyn fit of foo'''

    assert mesh is not None, 'Compiling requires mesh'
    
    # The idea is to compute df  as sum c_k dphi_k    
    def evaluate(x, foo=foo, mesh=mesh):
        x = np.fromiter(x, dtype=float)
        ufl_element = compilation.get_element(foo, mesh)

        # NOTE: this is here to make evaluating dofs simpler. Too lazy
        # now to handle Hdiv and what not
        assert ufl_element.family() in ('Lagrange', 'Discontinuous Lagrange')

        V = dolfin.FunctionSpace(mesh, ufl_element)
        
        el = V.element()
        mesh = V.mesh()
        # Find the cell with point
        x_point = dolfin.Point(*x) 
        cell_id = mesh.bounding_box_tree().compute_first_entity_collision(x_point)
        cell = dolfin.Cell(mesh, cell_id)
        coordinate_dofs = cell.get_vertex_coordinates()

        shape = dolfin.grad(foo).ufl_shape
        # grad grad grad ... ntime is stored in
        nslots = np.prod(shape)

        # Array for values with derivatives of all basis functions
        values = np.zeros(nslots*el.space_dimension(), dtype=float)
        # Compute all derivatives
        el.evaluate_basis_derivatives_all(1, values, x, coordinate_dofs, cell.orientation())
        # Reshape such that colums are [d/dxx, d/dxy, d/dyx, d/dyy...]
        values = values.reshape((-1, nslots))

        # Get expansion coefs on cell.
        indices = list(V.dofmap().cell_dofs(cell_id))
        dofs_x = V.tabulate_dof_coordinates().reshape((V.dim(), -1))[indices]

        # Sclar spaces
        if V.num_sub_spaces() == 0:
            dofs = np.array([foo(xj) for xj in dofs_x])
        # Not scalar spaces are filled by components
        else:
            dofs = np.zeros(len(indices), dtype=float)

            for comp in range(V.num_sub_spaces()):
                # Global
                comp_indices = list(V.sub(comp).dofmap().cell_dofs(cell_id))
                # Local w.r.t to all dofs
                local = [indices.index(comp_id) for comp_id in comp_indices]
                for lid in local:
                    dofs[lid] = foo(dofs_x[lid])[comp]

        # Perform reduction on each colum - you get that deriv of foo
        values = np.array([np.inner(row, dofs) for row in values.T])
        values = values.reshape(shape)

        return values
    return evaluate    


def eval_curl(arg, mesh):
    '''Curl of simple (i.e. not UFL composite)'''
    # This is not the most efficient, but curl is defined in terms of curl
    grad = compilation.icompile(dolfin.grad(arg), mesh)
    if arg.ufl_shape == ():
        # scalar -> R grad (expr) = vector
        return lambda x, g=grad: (lambda G: np.array([G[1], -G[0]]))(g(x))

    if arg.ufl_shape == (2, ):
        # vector -> div(R expr) = scalar
        # NOTE: I don't want to reshape grad to 2x2 and index then
        # so this is vector indexing
        return lambda x, g=grad: (lambda G: G[2]-G[1])(g(x))

    if arg.ufl_shape == (3, ):
        # The usual stuff
        return lambda x, g=grad: (lambda G: np.array([G[7]-G[5], G[2]-G[6], G[3]-G[1]]))(g(x))
    
    assert False

    
def eval_div(arg, mesh):
    '''Div of simple (i.e. not UFL composite)'''
    # This is not the most efficient, but curl is defined in terms of curl
    grad_ = dolfin.grad(arg)
    grad = compilation.icompile(grad_, mesh)

    # Consistency with UFL behavior
    if len(arg.ufl_shape) == 0:
        return lambda x, g=grad: g(x)
    # Vector
    if len(arg.ufl_shape) == 1:
        n = arg.ufl_shape[0]
        assert n in (2, 3)

        if n == 2:
            return lambda x, g=grad: (lambda G: G[0]+G[3])(g(x))
        else:
            return lambda x, g=grad: (lambda G: G[0]+G[4]+G[8])(g(x))
    # Tensors, rank > 1
    assert len(grad_.ufl_shape) == len(arg.ufl_shape) + 1
    # I think the general definition of div is tr grad which mounts to
    # grad : I with identity over the last two indices
    out_shape, id_shape = grad_.ufl_shape[:-2], grad_.ufl_shape[-2:]
    # Build the identity 'matrix'
    identity = [1 if i == j else 0 for i in range(id_shape[0]) for j in range(id_shape[1])]

    return lambda x, g=grad, I=identity, N=np.prod(out_shape):(
        lambda G, I=I: np.array([np.inner(Gi, I) for Gi in G])
        )(g(x).reshape((N, -1)))
