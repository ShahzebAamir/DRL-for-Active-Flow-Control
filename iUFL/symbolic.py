import sympy
import ufl


x, y = sympy.symbols('x, y')
A, b = sympy.MatrixSymbol('A', 'n', 'n'), sympy.MatrixSymbol('b', 'n', 'n')


UFL_EXPR = {type(x+y): ufl.algebra.Sum,
            type(x/y): ufl.algebra.Division,
            type(x*y): ufl.algebra.Product,
            type(x**y): ufl.algebra.Power,
            type(sympy.det(A)): ufl.tensoralgebra.Determinant,
            type(A.inverse()): ufl.tensoralgebra.Inverse,
            type(A.transpose()): ufl.tensoralgebra.Transposed,
            type(sympy.trace(A)): ufl.tensoralgebra.Trace,
            type(A*A): ufl.tensoralgebra.Dot}


def to_ufl(expr, substitutions):
    '''UFL expression from Sympy expression'''
    # A terminal
    if not expr.args:
        return substitutions[expr]
    # A node (with two arguments?)
    else:
        return UFL_EXPR[expr.func](*map(lambda e: to_ufl(e, substitutions), expr.args))

    
# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    import iufl

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)

    u = interpolate(Constant(1), V)
    v = interpolate(Constant(2), V)

    uu, vv = sympy.symbols('u, v')

    X = to_ufl(uu*vv, {uu: u, vv: v})
    print iufl.icompile(X)(0.2, 0.2)
