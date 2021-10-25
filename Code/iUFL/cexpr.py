from dolfin import UserExpression


def build_cexpr(element, shape, body):
    '''
    An Expression instance with element and which uses body to compute 
    it's eval method.
    '''

    def f(values, x, body=body): values[:] = body(x).flatten()

    def closure_f(values, x, f=f): return f(values, x)
    
    return type('CExpr',
                (UserExpression, ),
                {'value_shape': lambda self, shape=shape: shape,
                 'eval': lambda self, values, x: closure_f(values, x),
                 'is_CExpr': True})(element=element)
