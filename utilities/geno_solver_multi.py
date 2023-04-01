"""
Sample code automatically generated on 2021-07-20 08:20:53

by geno from www.geno-project.org

from input

parameters
  matrix A
  matrix B
  matrix C
variables
  vector x
min
  -sum(tanh(tanh(A*x)+tanh(B*x)+tanh(C*x)))+norm1(x)
st
  sum(x) >= 2
  sum(x) <= 10
  0 <= x
  1 >= x


Original problem has been transformed into

parameters
  matrix A
  matrix B
  matrix C
variables
  vector x
  vector tmp000
min
  sum(tmp000)-sum(tanh(tanh(A*x)+tanh(B*x)+tanh(C*x)))
st
  2-sum(x) <= 0
  sum(x)-10 <= 0
  x-tmp000 <= vector(0)
  -(x+tmp000) <= vector(0)


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf
from timeit import default_timer as timer
import numpy as np
from typing import List


try:
    from genosolver import minimize, check_version
    USE_GENO_SOLVER = True
except ImportError:
    from scipy.optimize import minimize
    USE_GENO_SOLVER = False
    WRN = 'WARNING: GENO solver not installed. Using SciPy solver instead.\n' + \
          'Run:     pip install genosolver'
    print('*' * 63)
    print(WRN)
    print('*' * 63)

class GenoNLP:
    def __init__(self, A, B, C, tmp000Init):
        self.A = A
        self.B = B
        self.C = C
        self.tmp000Init = tmp000Init
        assert isinstance(A, np.ndarray)
        dim = A.shape
        assert len(dim) == 2
        self.A_rows = dim[0]
        self.A_cols = dim[1]
        assert isinstance(B, np.ndarray)
        dim = B.shape
        assert len(dim) == 2
        self.B_rows = dim[0]
        self.B_cols = dim[1]
        assert isinstance(C, np.ndarray)
        dim = C.shape
        assert len(dim) == 2
        self.C_rows = dim[0]
        self.C_cols = dim[1]
        assert isinstance(tmp000Init, np.ndarray)
        dim = tmp000Init.shape
        assert len(dim) == 1
        self.tmp000_rows = dim[0]
        self.tmp000_cols = 1
        self.x_rows = self.A_cols
        self.x_cols = 1
        self.x_size = self.x_rows * self.x_cols
        self.tmp000_size = self.tmp000_rows * self.tmp000_cols
        # the following dim assertions need to hold for this problem
        assert self.A_rows == self.B_rows == self.C_rows
        assert self.B_cols == self.x_rows == self.C_cols == self.A_cols
        assert self.x_rows == self.tmp000_rows

    def getBounds(self):
        bounds = []
        bounds += [(0, 1)] * self.x_size
        bounds += [(0, inf)] * self.tmp000_size
        return bounds

    def getStartingPoint(self):
        self.xInit = np.random.randn(self.x_rows, self.x_cols)
        return np.hstack((self.xInit.reshape(-1), self.tmp000Init.reshape(-1)))

    def variables(self, _x):
        x = _x[0 : 0 + self.x_size]
        tmp000 = _x[0 + self.x_size : 0 + self.x_size + self.tmp000_size]
        return x, tmp000

    def fAndG(self, _x):
        x, tmp000 = self.variables(_x)
        t_0 = np.tanh((self.A).dot(x))
        t_1 = np.tanh((self.B).dot(x))
        t_2 = np.tanh((self.C).dot(x))
        t_3 = np.tanh(((t_0 + t_1) + t_2))
        t_4 = (np.ones(self.A_rows) - (t_3 ** 2))
        f_ = (np.sum(tmp000) - np.sum(t_3))
        g_0 = -(((self.A.T).dot((t_4 * (np.ones(self.A_rows) - (t_0 ** 2)))) + (self.B.T).dot((t_4 * (np.ones(self.A_rows) - (t_1 ** 2))))) + (self.C.T).dot((t_4 * (np.ones(self.A_rows) - (t_2 ** 2)))))
        g_1 = np.ones(self.tmp000_rows)
        g_ = np.hstack((g_0, g_1))
        return f_, g_

    def functionValueIneqConstraint000(self, _x):
        x, tmp000 = self.variables(_x)
        f = (2 - np.sum(x))
        return f

    def gradientIneqConstraint000(self, _x):
        x, tmp000 = self.variables(_x)
        g_0 = (-np.ones(self.x_rows))
        g_1 = (np.ones(self.tmp000_rows) * 0)
        g_ = np.hstack((g_0, g_1))
        return g_

    def jacProdIneqConstraint000(self, _x, _v):
        x, tmp000 = self.variables(_x)
        gv_0 = (-(_v * np.ones(self.x_rows)))
        gv_1 = (np.ones(self.tmp000_rows) * 0)
        gv_ = np.hstack((gv_0, gv_1))
        return gv_

    def functionValueIneqConstraint001(self, _x):
        x, tmp000 = self.variables(_x)
        f = (np.sum(x) - 10)
        return f

    def gradientIneqConstraint001(self, _x):
        x, tmp000 = self.variables(_x)
        g_0 = (np.ones(self.x_rows))
        g_1 = (np.ones(self.tmp000_rows) * 0)
        g_ = np.hstack((g_0, g_1))
        return g_

    def jacProdIneqConstraint001(self, _x, _v):
        x, tmp000 = self.variables(_x)
        gv_0 = ((_v * np.ones(self.x_rows)))
        gv_1 = (np.ones(self.tmp000_rows) * 0)
        gv_ = np.hstack((gv_0, gv_1))
        return gv_

    def functionValueIneqConstraint002(self, _x):
        x, tmp000 = self.variables(_x)
        f = (x - tmp000)
        return f

    def gradientIneqConstraint002(self, _x):
        x, tmp000 = self.variables(_x)
        g_0 = (np.eye(self.x_rows, self.x_rows))
        g_1 = (-np.eye(self.x_rows, self.tmp000_rows))
        g_ = np.hstack((g_0, g_1))
        return g_

    def jacProdIneqConstraint002(self, _x, _v):
        x, tmp000 = self.variables(_x)
        gv_0 = (_v)
        gv_1 = (-_v)
        gv_ = np.hstack((gv_0, gv_1))
        return gv_

    def functionValueIneqConstraint003(self, _x):
        x, tmp000 = self.variables(_x)
        f = -(x + tmp000)
        return f

    def gradientIneqConstraint003(self, _x):
        x, tmp000 = self.variables(_x)
        g_0 = (-np.eye(self.x_rows, self.x_rows))
        g_1 = (-np.eye(self.x_rows, self.tmp000_rows))
        g_ = np.hstack((g_0, g_1))
        return g_

    def jacProdIneqConstraint003(self, _x, _v):
        x, tmp000 = self.variables(_x)
        gv_0 = (-_v)
        gv_1 = (-_v)
        gv_ = np.hstack((gv_0, gv_1))
        return gv_

def toArray(v):
    return np.ascontiguousarray(v, dtype=np.float64).reshape(-1)

def solve(A, B, C, tmp000Init):
    start = timer()
    NLP = GenoNLP(A, B, C, tmp000Init)
    x0 = NLP.getStartingPoint()
    bnds = NLP.getBounds()
    tol = 1E-6
    # These are the standard GENO solver options, they can be omitted.
    options = {'tol' : tol,
               'constraintsTol' : 1E-4,
               'maxiter' : 1000,
               'verbosity' : 1  # Set it to 0 to fully mute it.
              }

    if USE_GENO_SOLVER:
        # Check if installed GENO solver version is sufficient.
        check_version('0.0.3')
        constraints = ({'type' : 'ineq',
                        'fun' : NLP.functionValueIneqConstraint000,
                        'jacprod' : NLP.jacProdIneqConstraint000},
                       {'type' : 'ineq',
                        'fun' : NLP.functionValueIneqConstraint001,
                        'jacprod' : NLP.jacProdIneqConstraint001},
                       {'type' : 'ineq',
                        'fun' : NLP.functionValueIneqConstraint002,
                        'jacprod' : NLP.jacProdIneqConstraint002},
                       {'type' : 'ineq',
                        'fun' : NLP.functionValueIneqConstraint003,
                        'jacprod' : NLP.jacProdIneqConstraint003})
        result = minimize(NLP.fAndG, x0,
                          bounds=bnds, options=options,
                          constraints=constraints)
    else:
        # SciPy: for inequality constraints need to change sign f(x) <= 0 -> f(x) >= 0
        constraints = ({'type' : 'ineq',
                        'fun' : lambda x: -NLP.functionValueIneqConstraint000(x),
                        'jac' : lambda x: -NLP.gradientIneqConstraint000(x)},
                       {'type' : 'ineq',
                        'fun' : lambda x: -NLP.functionValueIneqConstraint001(x),
                        'jac' : lambda x: -NLP.gradientIneqConstraint001(x)},
                       {'type' : 'ineq',
                        'fun' : lambda x: -NLP.functionValueIneqConstraint002(x),
                        'jac' : lambda x: -NLP.gradientIneqConstraint002(x)},
                       {'type' : 'ineq',
                        'fun' : lambda x: -NLP.functionValueIneqConstraint003(x),
                        'jac' : lambda x: -NLP.gradientIneqConstraint003(x)})
        result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                          bounds=bnds,
                          constraints=constraints)

    # assemble solution and map back to original problem
    x = result.x
    ineqConstraint000 = np.asarray(NLP.functionValueIneqConstraint000(x))
    ineqConstraint000[ineqConstraint000 < 0] = 0
    ineqConstraint001 = np.asarray(NLP.functionValueIneqConstraint001(x))
    ineqConstraint001[ineqConstraint001 < 0] = 0
    ineqConstraint002 = np.asarray(NLP.functionValueIneqConstraint002(x))
    ineqConstraint002[ineqConstraint002 < 0] = 0
    ineqConstraint003 = np.asarray(NLP.functionValueIneqConstraint003(x))
    ineqConstraint003[ineqConstraint003 < 0] = 0
    x, tmp000 = NLP.variables(x)
    solution = {}
    solution['success'] = result.success
    solution['message'] = result.message
    solution['fun'] = result.fun
    solution['grad'] = result.jac
    if USE_GENO_SOLVER:
        solution['slack'] = result.slack
    solution['x'] = x
    solution['tmp000'] = tmp000
    solution['ineqConstraint000'] = toArray(ineqConstraint000)
    solution['ineqConstraint001'] = toArray(ineqConstraint001)
    solution['ineqConstraint002'] = toArray(ineqConstraint002)
    solution['ineqConstraint003'] = toArray(ineqConstraint003)
    solution['elapsed'] = timer() - start
    return solution

def generateRandomData():
    np.random.seed(0)
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)
    C = np.random.randn(3, 3)
    tmp000Init = np.random.randn(3)
    return A, B, C, tmp000Init


def run(tokens: List[str], matrixs:List[List[List[float]]], max_k: int, min_k: int):
    """Call geno solver to return min_k and max_k tokens"""
    A, B, C = matrixs
    assert len(tokens) == len(A)
    A = np.array(A).transpose(1, 0)
    B = np.array(B).transpose(1, 0)
    C = np.array(C).transpose(1, 0)
    #matrix = np.array(matrix).transpose(1, 0)
    np.random.seed(0)
    tmp000Init = np.random.randn(len(tokens))
    solution = solve(A, B, C, tmp000Init)
    solved = solution['x']
    sigmoid_index = np.where(solved > 0.5)[0]       # only keep the rows where the value is above 0.5.
    sorted_index = np.argsort(-solved)[:max_k]   # descreasing order
    
    picked_index = []
    for i in sorted_index:
        if i in sigmoid_index or len(picked_index) < min_k:
            picked_index.append(i)
    
    A_picked = A[:, np.array(picked_index)]
    B_picked = B[:, np.array(picked_index)]
    C_picked = C[:, np.array(picked_index)]
    A_utility = (np.sum(A_picked, axis = 1) >= 0).sum() / float(A_picked.shape[0])
    B_utility = (np.sum(B_picked, axis = 1) >= 0).sum() / float(B_picked.shape[0])
    C_utility = (np.sum(C_picked, axis = 1) >= 0).sum() / float(C_picked.shape[0])
    expansion = [tokens[i] for i in picked_index]
    utility = max(A_utility, B_utility, C_utility)
    return expansion, utility


if __name__ == '__main__':
    print('\ngenerating random instance')
    A, B, C, tmp000Init = generateRandomData()
    print(A, B, C)
    print('solving ...')
    solution = solve(A, B, C, tmp000Init)
    print('*'*5, 'solution', '*'*5)
    print(solution['message'])
    if solution['success']:
        print('optimal function value   = ', solution['fun'])
        print('norm of the gradient     = ',
              np.linalg.norm(solution['grad'], np.inf))
        if USE_GENO_SOLVER:
            print('maximal compl. slackness = ', solution['slack'])
        print('optimal variable x = ', solution['x'])
        print('optimal variable tmp000 = ', solution['tmp000'])
        print('norm of the 1st inequality constraint violation ',
              np.linalg.norm(solution['ineqConstraint000'], np.inf))
        print('norm of the 2nd inequality constraint violation ',
              np.linalg.norm(solution['ineqConstraint001'], np.inf))
        print('norm of the 3rd inequality constraint violation ',
              np.linalg.norm(solution['ineqConstraint002'], np.inf))
        print('norm of the 4th inequality constraint violation ',
              np.linalg.norm(solution['ineqConstraint003'], np.inf))
        print('solving took %.3f sec' % solution['elapsed'])


    # confirm
    x = np.array([0, 0, 1])
    print(np.tanh(A@x))
    tanh= np.tanh((np.tanh(A @x) + np.tanh(B@x) + np.tanh(C@x)))
    y = -tanh.sum() + x.sum()
    print(f'replicate: {tanh}\n{tanh.sum()}\n{y}')