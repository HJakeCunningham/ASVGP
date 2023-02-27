import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from bandgp.utils import repeat_and_pad_32

class SplineBasis(ABC):
    """ 
    Parent class associated with B Spline basis
    """

    def __init__(self, a, b, m):
        self.a = a
        self.b = b
        self.m = m
        self.mesh = tf.cast(tf.linspace(a, b, m - (self.order-1)), dtype=tf.float64)
        self.delta = self.mesh[1] - self.mesh[0]

    def _make_matrix(self, diags):
        """ Computes dense banded matrix
        """
        diags = tf.cast(diags, dtype=tf.float64)
        K = tf.linalg.diag(tf.repeat(diags[0], self.m), k=0)
        for i in range(len(diags)-1):
            j = i + 1
            K += tf.linalg.diag(tf.repeat(diags[j], self.m-j), k=j)
            K += tf.linalg.diag(tf.repeat(diags[j], self.m-j), k=-j)
        return K

    def _make_banded_matrix(self, diags, pad='right'):
        """ Computes sparse banded matrix taking into account edge effects of B Splines
        """
        bands = []
        for i, diag in enumerate(diags):
            lhs = tf.math.cumsum(diag)
            mid = tf.repeat(tf.math.reduce_sum(diag), self.m - 2*(diag.shape[0]) - i)
            rhs = lhs[::-1]
            padding = tf.zeros((i,), dtype=tf.float64)
            if pad == 'right':
                bands.append(tf.concat([lhs, mid, rhs, padding], axis=0))
            if pad == 'left':
                bands.append(tf.concat([padding, lhs, mid, rhs], axis=0))
        K = tf.stack(bands, axis=0)
        return K

    @abstractmethod
    def _evaluate(self, neighbour_value: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        return None

    def evaluate_basis(self, X, dx=0, sparse=True):
        """ Evaluations of B Spline basis functions returned as (m,n) matrix
        """
        n, m = X.shape[0], self.m
        X = tf.cast(tf.reshape(X, (n,)), dtype=tf.float64)

        # Index the nearest left mesh point
        neighbour_index = tf.cast(tf.nn.relu((tf.searchsorted(self.mesh, X) - 1)), tf.int64)
        neighbour_value = tf.gather(self.mesh, neighbour_index)

        if dx == 0:
            data = self._evaluate(neighbour_value, X)
        elif dx == 1:
            data = self._evaluate_grad(neighbour_value, X)
        elif dx == 2:
            data = self._evaluate_ggrad(neighbour_value, X)
        elif dx == 3:
            data = self._evaluate_gggrad(neighbour_value, X)
        else:
            raise NotImplementedError

        rows = tf.cast(tf.concat([neighbour_index + self.order - i for i in range(self.order+1)], axis=0), dtype=tf.int64)
        cols = tf.cast(tf.tile(tf.range(n), tf.constant([self.order+1])), dtype=tf.int64)

        if sparse:
            return csr_matrix((data, (rows, cols)), shape=(m,n))

        if not sparse:
            indices = tf.stack([rows, cols], axis=1)
            return tf.scatter_nd(tf.cast(indices, dtype=tf.int64), data, tf.cast([m, n], dtype=tf.int64))

    def make_boundary_conditions(self, dx=0, pad='right'):
        """ Computes Kuu boundary condition products
        """
        if dx == 0:
            lhs = self.evaluate_basis(tf.cast([self.a], dtype=tf.float64), dx=0.0, sparse=False)
            rhs = lhs
        if dx == 1:
            lhs = self.evaluate_basis(tf.cast([self.a], dtype=tf.float64), dx=1.0, sparse=False)
            rhs = lhs
        if dx == 2:
            lhs = self.evaluate_basis(tf.cast([self.a], dtype=tf.float64), dx=2.0, sparse=False)
            rhs = lhs
        if dx == 3:
            lhs = self.evaluate_basis(tf.cast([self.a], dtype=tf.float64), dx=2.0, sparse=False)
            rhs = self.evaluate_basis(tf.cast([self.b], dtype=tf.float64), dx=0.0, sparse=False)
        if dx == 4:
            lhs = self.evaluate_basis(tf.cast([self.a], dtype=tf.float64), dx=0.0, sparse=False)
            rhs = self.evaluate_basis(tf.cast([self.b], dtype=tf.float64), dx=2.0, sparse=False)

        # Compute outer product
        lhs_mat = lhs[:self.order] @ tf.transpose(rhs[:self.order])

        bands = []
        for i in range(lhs_mat.shape[0]):
            l = tf.linalg.diag_part(lhs_mat, k=i)
            zero_fill = tf.zeros((self.m - 2 * l.shape[0] - i), dtype=tf.float64)
            zero_pad = tf.zeros(i, dtype=tf.float64)
            if pad=='right':
                bands.append(tf.concat([l, zero_fill, l, zero_pad], axis=0))
            if pad=='left':
                bands.append(tf.concat([zero_pad, l, zero_fill, l], axis=0))
        bands.append(tf.zeros(self.m, dtype=tf.float64))
        return tf.stack(bands, axis=0)

    
class B1Spline(SplineBasis):
    """
    Class to compute inner product matrices corresponding to B1 Spline basis functions
    """

    def __init__(self, a, b, m):
        self.order = 1
        super().__init__(a, b, m)

        # Precompute static matrices
        self.A = self.l2_inner_product()
        self.B = self.l2_grad_inner_product()

        # Boundary condition matrices
        self.BC = self.make_boundary_conditions(dx=0)

    def _evaluate(self, u, X):
        b1 = (X - u) / self.delta
        b2 = (u + self.delta - X) / self.delta
        return tf.concat([b1, b2], axis=0)

    def _evaluate_grad(self, u, X):
        b1 = 1/self.delta
        b2 = -1/self.delta
        return tf.concat([b1, b2], axis=0)

    def l2_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i(x) ϕ_j(x) dx for the B1 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = 2 * self.delta / 3
        # d1 = self.delta / 6
        d0 = tf.cast([1/3*self.delta,1/3*self.delta], dtype=tf.float64)
        d1 = tf.cast([1/6*self.delta], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1])

    def l2_grad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i'(x) ϕ_j'(x) dx for the B1 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = 2 / self.delta
        # d1 = -1 / self.delta
        d0 = tf.cast([1/self.delta,1/self.delta], dtype=tf.float64)
        d1 = tf.cast([-1/self.delta], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1])


class B2Spline(SplineBasis):
    """
    Class to compute inner product matrices corresponding to B1 Spline basis functions
    """

    def __init__(self, a, b, m):
        self.order = 2
        super().__init__(a, b, m)

        # Precompute Static Matrices
        self.A = self.l2_inner_product()
        self.B = self.l2_grad_inner_product()
        self.C = self.l2_ggrad_inner_product()

        # Boundary condition matrices
        self.BC = self.make_boundary_conditions(dx=0)
        self.BC_grad = self.make_boundary_conditions(dx=1)

    def _evaluate(self, u, X):
        b1 = (X - u)**2 / (2 * self.delta**2)
        b2 = ((X - u + self.delta) * (u + self.delta - X) + (u + 2*self.delta - X) * (X - u)) / (2 * self.delta**2)
        b3 = (u + self.delta - X)**2 / (2 * self.delta**2)
        return tf.concat([b1, b2, b3], axis=0)

    def _evaluate_grad(self, u, X):
        b1 = -(u - X)/self.delta**2
        b2 = 1/2*(2*self.delta + u - X)/self.delta**2 + 1/2*(self.delta + u - X)/self.delta**2 - 1/2*(self.delta - u + X)/self.delta**2 + 1/2*(u - X)/self.delta**2
        b3 = -(self.delta + u - X)/self.delta**2
        return tf.concat([b1, b2, b3], axis=0)

    def _evaluate_ggrad(self, u, X):
        b1 = self.delta**(-2)
        b2 = -2/self.delta**2
        b3 = self.delta**(-2)
        return tf.concat([b1, b2, b3], axis=0)

    def l2_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i(x) ϕ_j(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = self.delta * 11/20
        # d1 = self.delta * 13/60
        # d2 = self.delta * 1/120
        d0 = tf.cast([1/20*self.delta,9/20*self.delta,1/20*self.delta], dtype=tf.float64)
        d1 = tf.cast([13/120*self.delta,13/120*self.delta], dtype=tf.float64)
        d2 = tf.cast([1/120*self.delta], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2])

    def l2_grad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i'(x) ϕ_j'(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta
        # d1 = 1/self.delta * -1/3
        # d2 = 1/self.delta * -1/6
        d0 = tf.cast([1/3/self.delta,1/3/self.delta,1/3/self.delta], dtype=tf.float64)
        d1 = tf.cast([-1/6/self.delta,-1/6/self.delta], dtype=tf.float64)
        d2 = tf.cast([-1/6/self.delta], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2])

    def l2_ggrad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i''(x) ϕ_j''(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = 6 / self.delta**3
        # d1 = -4 / self.delta**3
        # d2 = 1 / self.delta**3 
        d0 = tf.cast([self.delta**(-3),4/self.delta**3,self.delta**(-3)], dtype=tf.float64)
        d1 = tf.cast([-2/self.delta**3,-2/self.delta**3], dtype=tf.float64)
        d2 = tf.cast([self.delta**(-3)], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2])


class B3Spline(SplineBasis):
    """
    Class to compute inner product matrices corresponding to B1 Spline basis functions
    """

    def __init__(self, a, b, m):
        self.order = 3
        super().__init__(a, b, m)

        # Precompute Static Matrices
        self.A = self.l2_inner_product()
        self.B = self.l2_grad_inner_product()
        self.C = self.l2_ggrad_inner_product()
        self.D = self.l2_gggrad_inner_product()

        # Boundary condition matrices
        self.BC = self.make_boundary_conditions(dx=0)
        self.BC_grad = self.make_boundary_conditions(dx=1)
        self.BC_ggrad = self.make_boundary_conditions(dx=2)
        self.BC_ggrad_none = self.make_boundary_conditions(dx=3)
        self.BC_none_ggrad = self.make_boundary_conditions(dx=4)

    def _evaluate(self, neighbour_value, X):
        u = neighbour_value
        b1 = 1 / (6*self.delta**3) * (X - u)**3 
        b2 = 1 / (6*self.delta**3) * ( ((X - u + self.delta)**2)*(u+self.delta-X) + (X-u+self.delta)*(u+2*self.delta-X)*(X-u) + (u+3*self.delta-X)*((X-u)**2) )
        b3 = 1 / (6*self.delta**3) * ( (X - u + 2*self.delta)*((u+self.delta-X)**2) + (X-u+self.delta)*(u+self.delta-X)*(u+2*self.delta-X) + ((u+2*self.delta-X)**2)*(X-u) )
        b4 = 1 / (6*self.delta**3) * (u + self.delta - X)**3
        return tf.concat([b1, b2, b3, b4], axis=0)

    def _evaluate_grad(self, u, X):
        b1 = 1/2*(u - X)**2/self.delta**3
        b2 = 1/6*(self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 1/6*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 1/3*(3*self.delta + u - X)*(u - X)/self.delta**3 - 1/6*(u - X)**2/self.delta**3
        b3 = 1/6*(2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - 1/6*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 1/3*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + 1/6*(self.delta + u - X)**2/self.delta**3
        b4 = -1/2*(self.delta + u - X)**2/self.delta**3
        return tf.concat([b1, b2, b3, b4], axis=0)

    def _evaluate_ggrad(self, u, X):
        b1 = -(u - X)/self.delta**3
        b2 = 1/3*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 1/3*(3*self.delta + u - X)/self.delta**3 - 2/3*(self.delta - u + X)/self.delta**3 + 2/3*(u - X)/self.delta**3
        b3 = -1/3*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - 2/3*(2*self.delta + u - X)/self.delta**3 + 1/3*(2*self.delta - u + X)/self.delta**3 - 2/3*(self.delta + u - X)/self.delta**3
        b4 = (self.delta + u - X)/self.delta**3
        return tf.concat([b1, b2, b3, b4], axis=0)

    def _evaluate_gggrad(self, u, X):
        b1 = self.delta**(-3) * tf.ones(X.shape, dtype=tf.float64)
        b2 = -3/self.delta**3 * tf.ones(X.shape, dtype=tf.float64)
        b3 = 3/self.delta**3 * tf.ones(X.shape, dtype=tf.float64)
        b4 = -1/self.delta**3 * tf.ones(X.shape, dtype=tf.float64)
        return tf.concat([b1, b2, b3, b4], axis=0)

    def l2_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i(x) ϕ_j(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = 151/315 * self.delta
        # d1 = 397/1680 * self.delta
        # d2 = 1/42 * self.delta
        # d3 = 1/5040 * self.delta
        d0 = tf.cast([1/252*self.delta,33/140*self.delta,33/140*self.delta,1/252*self.delta], dtype=tf.float64)
        d1 = tf.cast([43/1680*self.delta,311/1680*self.delta,43/1680*self.delta], dtype=tf.float64)
        d2 = tf.cast([1/84*self.delta,1/84*self.delta], dtype=tf.float64)
        d3 = tf.cast([1/5040*self.delta], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3])

    def l2_grad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i'(x) ϕ_j'(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta * 2/3
        # d1 = 1/self.delta * -1/8
        # d2 = 1/self.delta * -1/5
        # d3 = 1/self.delta * -1/120
        d0 = tf.cast([1/20/self.delta,17/60/self.delta,17/60/self.delta,1/20/self.delta], dtype=tf.float64)
        d1 = tf.cast([7/120/self.delta,-29/120/self.delta,7/120/self.delta], dtype=tf.float64)
        d2 = tf.cast([-1/10/self.delta,-1/10/self.delta], dtype=tf.float64)
        d3 = tf.cast([-1/120/self.delta], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3])

    def l2_ggrad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i''(x) ϕ_j''(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            di: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta**3 * 8/3
        # d1 = 1/self.delta**3 * -3/2
        # d2 = tf.cast(0.0, dtype=tf.float64)
        # d3 = 1/self.delta**3 * 1/6
        d0 = tf.cast([1/3/self.delta**3,self.delta**(-3),self.delta**(-3),1/3/self.delta**3], dtype=tf.float64)
        d1 = tf.cast([-1/2/self.delta**3,-1/2/self.delta**3,-1/2/self.delta**3], dtype=tf.float64)
        d2 = tf.cast([0,0], dtype=tf.float64)
        d3 = tf.cast([1/6/self.delta**3], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3])

    def l2_gggrad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i'''(x) ϕ_j'''(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            di: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta**5 * 20.0
        # d1 = 1/self.delta**5 * -15.0
        # d2 = 1/self.delta**5 * 6.0
        # d3 = 1/self.delta**5 * -1.0
        d0 = tf.cast([self.delta**(-5),9/self.delta**5,9/self.delta**5,self.delta**(-5)], dtype=tf.float64)
        d1 = tf.cast([-3/self.delta**5,-9/self.delta**5,-3/self.delta**5], dtype=tf.float64)
        d2 = tf.cast([3/self.delta**5,3/self.delta**5], dtype=tf.float64)
        d3 = tf.cast([-1/self.delta**5], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3])


class B4Spline(SplineBasis):
    """
    Class to compute inner product matrices corresponding to B4 Spline basis functions
    """

    def __init__(self, a, b, m):
        self.order = 4
        if m < 12:
            raise NameError('Not enough basis functions m >= 12')

        super().__init__(a, b, m)

        # Precompute Static Matrices
        self.A = self.l2_inner_product()
        self.B = self.l2_grad_inner_product()
        self.C = self.l2_ggrad_inner_product()
        self.D = self.l2_gggrad_inner_product()

        # Boundary condition matrices
        self.BC = self.make_boundary_conditions(dx=0)
        self.BC_grad = self.make_boundary_conditions(dx=1)
        self.BC_ggrad = self.make_boundary_conditions(dx=2)
        self.BC_ggrad_none = self.make_boundary_conditions(dx=3)
        self.BC_none_ggrad = self.make_boundary_conditions(dx=4)

    def _evaluate(self, u, X):
        b1 = 1/24*(u - X)**4/self.delta**4
        b2 = 1/24*(self.delta - u + X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta - 1/24*(4*self.delta + u - X)*(u - X)**3/self.delta**4
        b3 = 1/24*(2*self.delta - u + X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + 1/24*(3*self.delta + u - X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta
        b4 = 1/24*(2*self.delta + u - X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + 1/24*(3*self.delta - u + X)*(self.delta + u - X)**3/self.delta**4
        b5 = 1/24*(self.delta + u - X)**4/self.delta**4
        return tf.concat([b1, b2, b3, b4, b5], axis=0)

    def _evaluate_grad(self, u, X):
        b1 = -1/6*(u - X)**3/self.delta**4
        b2 = 1/24*(self.delta - u + X)*((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta + 1/24*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta + 1/8*(4*self.delta + u - X)*(u - X)**2/self.delta**4 + 1/24*(u - X)**3/self.delta**4
        b3 = 1/24*(2*self.delta - u + X)*((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta + 1/24*(3*self.delta + u - X)*((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta + 1/24*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta - 1/24*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta
        b4 = 1/24*(2*self.delta + u - X)*((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - 1/24*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta - 1/8*(3*self.delta - u + X)*(self.delta + u - X)**2/self.delta**4 + 1/24*(self.delta + u - X)**3/self.delta**4
        b5 = -1/6*(self.delta + u - X)**3/self.delta**4
        return tf.concat([b1, b2, b3, b4, b5], axis=0)

    def _evaluate_ggrad(self, u, X):
        b1 = 1/2*(u - X)**2/self.delta**4
        b2 = 1/12*(self.delta - u + X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta + 1/12*((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta - 1/4*(4*self.delta + u - X)*(u - X)/self.delta**4 - 1/4*(u - X)**2/self.delta**4
        b3 = 1/12*(3*self.delta + u - X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta - 1/12*(2*self.delta - u + X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta + 1/12*((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - 1/12*((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta
        b4 = -1/12*(2*self.delta + u - X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta - 1/12*((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta + 1/4*(3*self.delta - u + X)*(self.delta + u - X)/self.delta**4 - 1/4*(self.delta + u - X)**2/self.delta**4
        b5 = 1/2*(self.delta + u - X)**2/self.delta**4
        return tf.concat([b1, b2, b3, b4, b5], axis=0)

    def _evaluate_gggrad(self, u, X):
        b1 = -(u - X)/self.delta**4
        b2 = 1/4*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta + 1/4*(4*self.delta + u - X)/self.delta**4 - 3/4*(self.delta - u + X)/self.delta**4 + 3/4*(u - X)/self.delta**4
        b3 = -1/4*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta - 1/4*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta - 3/4*(3*self.delta + u - X)/self.delta**4 + 3/4*(2*self.delta - u + X)/self.delta**4
        b4 = 1/4*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta - 1/4*(3*self.delta - u + X)/self.delta**4 + 3/4*(2*self.delta + u - X)/self.delta**4 + 3/4*(self.delta + u - X)/self.delta**4
        b5 = -(self.delta + u - X)/self.delta**4
        return tf.concat([b1, b2, b3, b4, b5], axis=0)

    def l2_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i(x) ϕ_j(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = 15619/36288 * self.delta
        # d1 = 44117/181440 * self.delta
        # d2 = 913/22680 * self.delta
        # d3 = 251/181440 * self.delta
        # d4 = 1/362880 * self.delta
        d0 = tf.cast([1/5184*self.delta,2281/36288*self.delta,409/1344*self.delta,2281/36288*self.delta,1/5184*self.delta], dtype=tf.float64)
        d1 = tf.cast([1121/362880*self.delta,3583/30240*self.delta,3583/30240*self.delta,1121/362880*self.delta], dtype=tf.float64)
        d2 = tf.cast([527/120960*self.delta,5723/181440*self.delta,527/120960*self.delta], dtype=tf.float64)
        d3 = tf.cast([251/362880*self.delta,251/362880*self.delta], dtype=tf.float64)
        d4 = tf.cast([1/362880*self.delta], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3, d4])

    def l2_grad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i'(x) ϕ_j'(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta * 35/72
        # d1 = 1/self.delta * -11/360
        # d2 = 1/self.delta * -17/90
        # d3 = 1/self.delta * -59/2520
        # d4 = 1/self.delta * -1/5040
        d0 = tf.cast([1/252/self.delta,95/504/self.delta,17/168/self.delta,95/504/self.delta,1/252/self.delta], dtype=tf.float64)
        d1 = tf.cast([109/5040/self.delta,-31/840/self.delta,-31/840/self.delta,109/5040/self.delta], dtype=tf.float64)
        d2 = tf.cast([-23/1680/self.delta,-407/2520/self.delta,-23/1680/self.delta], dtype=tf.float64)
        d3 = tf.cast([-59/5040/self.delta,-59/5040/self.delta], dtype=tf.float64)
        d4 = tf.cast([-1/5040/self.delta], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3, d4])

    def l2_ggrad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i''(x) ϕ_j''(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            di: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta**3 * 19/12
        # d1 = 1/self.delta**3 * -43/60
        # d2 = 1/self.delta**3 * -4/15
        # d3 = 1/self.delta**3 * 11/60
        # d4 = 1/self.delta**3 * 1/120
        d0 = tf.cast([1/20/self.delta**3,13/60/self.delta**3,21/20/self.delta**3,13/60/self.delta**3,1/20/self.delta**3], dtype=tf.float64)
        d1 = tf.cast([1/120/self.delta**3,-11/30/self.delta**3,-11/30/self.delta**3,1/120/self.delta**3], dtype=tf.float64)
        d2 = tf.cast([-19/120/self.delta**3,1/20/self.delta**3,-19/120/self.delta**3], dtype=tf.float64)
        d3 = tf.cast([11/120/self.delta**3,11/120/self.delta**3], dtype=tf.float64)
        d4 = tf.cast([1/120/self.delta**3], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3, d4])

    def l2_gggrad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i'''(x) ϕ_j'''(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            di: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta**5 * 25/3
        # d1 = 1/self.delta**5 * -17/3
        # d2 = 1/self.delta**5 * 4/3
        # d3 = 1/self.delta**5 * 1/3
        # d4 = 1/self.delta**5 * -1/6
        d0 = tf.cast([1/3/self.delta**5,7/3/self.delta**5,3/self.delta**5,7/3/self.delta**5,1/3/self.delta**5], dtype=tf.float64)
        d1 = tf.cast([-5/6/self.delta**5,-2/self.delta**5,-2/self.delta**5,-5/6/self.delta**5], dtype=tf.float64)
        d2 = tf.cast([1/2/self.delta**5,1/3/self.delta**5,1/2/self.delta**5], dtype=tf.float64)
        d3 = tf.cast([1/6/self.delta**5,1/6/self.delta**5], dtype=tf.float64)
        d4 = tf.cast([-1/6/self.delta**5], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3, d4])


class B5Spline(SplineBasis):
    """
    Class to compute inner product matrices corresponding to B4 Spline basis functions
    """

    def __init__(self, a, b, m):
        self.order = 5
        super().__init__(a, b, m)

        # Precompute Static Matrices
        self.A = self.l2_inner_product()
        self.B = self.l2_grad_inner_product()
        self.C = self.l2_ggrad_inner_product()
        self.D = self.l2_gggrad_inner_product()

        # Boundary condition matrices
        self.BC = self.make_boundary_conditions(dx=0)
        self.BC_grad = self.make_boundary_conditions(dx=1)
        self.BC_ggrad = self.make_boundary_conditions(dx=2)
        self.BC_ggrad_none = self.make_boundary_conditions(dx=3)
        self.BC_none_ggrad = self.make_boundary_conditions(dx=4)

    def _evaluate(self, u, X):
        b1 = -1/120*(u - X)**5/self.delta**5
        b2 = 1/120*(self.delta - u + X)*((self.delta - u + X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta - (4*self.delta + u - X)*(u - X)**3/self.delta**4)/self.delta + 1/120*(5*self.delta + u - X)*(u - X)**4/self.delta**5
        b3 = 1/120*(2*self.delta - u + X)*((2*self.delta - u + X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta + u - X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta)/self.delta + 1/120*(4*self.delta + u - X)*((self.delta - u + X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta - (4*self.delta + u - X)*(u - X)**3/self.delta**4)/self.delta
        b4 = 1/120*(3*self.delta - u + X)*((2*self.delta + u - X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta - u + X)*(self.delta + u - X)**3/self.delta**4)/self.delta + 1/120*(3*self.delta + u - X)*((2*self.delta - u + X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta + u - X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta)/self.delta
        b5 = 1/120*(2*self.delta + u - X)*((2*self.delta + u - X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta - u + X)*(self.delta + u - X)**3/self.delta**4)/self.delta + 1/120*(4*self.delta - u + X)*(self.delta + u - X)**4/self.delta**5
        b6 = 1/120*(self.delta + u - X)**5/self.delta**5
        return tf.concat([b1, b2, b3, b4, b5, b6], axis=0)

    def _evaluate_grad(self, u, X):
        b1 = 1/24*(u - X)**4/self.delta**5
        b2 = 1/120*(self.delta - u + X)*((self.delta - u + X)*((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta + ((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta + 3*(4*self.delta + u - X)*(u - X)**2/self.delta**4 + (u - X)**3/self.delta**4)/self.delta + 1/120*((self.delta - u + X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta - (4*self.delta + u - X)*(u - X)**3/self.delta**4)/self.delta - 1/30*(5*self.delta + u - X)*(u - X)**3/self.delta**5 - 1/120*(u - X)**4/self.delta**5
        b3 = 1/120*(2*self.delta - u + X)*((2*self.delta - u + X)*((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta + u - X)*((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta + ((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta - ((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta)/self.delta + 1/120*(4*self.delta + u - X)*((self.delta - u + X)*((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta + ((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta + 3*(4*self.delta + u - X)*(u - X)**2/self.delta**4 + (u - X)**3/self.delta**4)/self.delta + 1/120*((2*self.delta - u + X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta + u - X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta)/self.delta - 1/120*((self.delta - u + X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta - (4*self.delta + u - X)*(u - X)**3/self.delta**4)/self.delta
        b4 = 1/120*(3*self.delta - u + X)*((2*self.delta + u - X)*((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - ((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta - 3*(3*self.delta - u + X)*(self.delta + u - X)**2/self.delta**4 + (self.delta + u - X)**3/self.delta**4)/self.delta + 1/120*(3*self.delta + u - X)*((2*self.delta - u + X)*((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta + u - X)*((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta + ((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta - ((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta)/self.delta + 1/120*((2*self.delta + u - X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta - u + X)*(self.delta + u - X)**3/self.delta**4)/self.delta - 1/120*((2*self.delta - u + X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta + u - X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta)/self.delta
        b5 = 1/120*(2*self.delta + u - X)*((2*self.delta + u - X)*((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - ((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta - 3*(3*self.delta - u + X)*(self.delta + u - X)**2/self.delta**4 + (self.delta + u - X)**3/self.delta**4)/self.delta - 1/120*((2*self.delta + u - X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta - u + X)*(self.delta + u - X)**3/self.delta**4)/self.delta - 1/30*(4*self.delta - u + X)*(self.delta + u - X)**3/self.delta**5 + 1/120*(self.delta + u - X)**4/self.delta**5
        b6 = -1/24*(self.delta + u - X)**4/self.delta**5
        return tf.concat([b1, b2, b3, b4, b5, b6], axis=0)

    def _evaluate_ggrad(self, u, X):
        b1 = -1/6*(u - X)**3/self.delta**5
        b2 = 1/60*(self.delta - u + X)*((self.delta - u + X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta + ((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta - 3*(4*self.delta + u - X)*(u - X)/self.delta**4 - 3*(u - X)**2/self.delta**4)/self.delta + 1/60*((self.delta - u + X)*((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta + ((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta + 3*(4*self.delta + u - X)*(u - X)**2/self.delta**4 + (u - X)**3/self.delta**4)/self.delta + 1/10*(5*self.delta + u - X)*(u - X)**2/self.delta**5 + 1/15*(u - X)**3/self.delta**5
        b3 = 1/60*(2*self.delta - u + X)*((3*self.delta + u - X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta - (2*self.delta - u + X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta + ((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - ((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta)/self.delta + 1/60*(4*self.delta + u - X)*((self.delta - u + X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta + ((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta - 3*(4*self.delta + u - X)*(u - X)/self.delta**4 - 3*(u - X)**2/self.delta**4)/self.delta + 1/60*((2*self.delta - u + X)*((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta + u - X)*((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta + ((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta - ((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta)/self.delta - 1/60*((self.delta - u + X)*((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta + ((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta + 3*(4*self.delta + u - X)*(u - X)**2/self.delta**4 + (u - X)**3/self.delta**4)/self.delta
        b4 = 1/60*(3*self.delta + u - X)*((3*self.delta + u - X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta - (2*self.delta - u + X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta + ((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - ((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta)/self.delta - 1/60*(3*self.delta - u + X)*((2*self.delta + u - X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta + ((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - 3*(3*self.delta - u + X)*(self.delta + u - X)/self.delta**4 + 3*(self.delta + u - X)**2/self.delta**4)/self.delta + 1/60*((2*self.delta + u - X)*((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - ((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta - 3*(3*self.delta - u + X)*(self.delta + u - X)**2/self.delta**4 + (self.delta + u - X)**3/self.delta**4)/self.delta - 1/60*((2*self.delta - u + X)*((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta + u - X)*((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta + ((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta - ((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta)/self.delta
        b5 = -1/60*(2*self.delta + u - X)*((2*self.delta + u - X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta + ((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - 3*(3*self.delta - u + X)*(self.delta + u - X)/self.delta**4 + 3*(self.delta + u - X)**2/self.delta**4)/self.delta - 1/60*((2*self.delta + u - X)*((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - ((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta - 3*(3*self.delta - u + X)*(self.delta + u - X)**2/self.delta**4 + (self.delta + u - X)**3/self.delta**4)/self.delta + 1/10*(4*self.delta - u + X)*(self.delta + u - X)**2/self.delta**5 - 1/15*(self.delta + u - X)**3/self.delta**5
        b6 = 1/6*(self.delta + u - X)**3/self.delta**5
        return tf.concat([b1, b2, b3, b4, b5, b6], axis=0)

    def _evaluate_gggrad(self, u, X):
        b1 = 1/2*(u - X)**2/self.delta**5
        b2 = 1/20*(self.delta - u + X)*((((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta + (4*self.delta + u - X)/self.delta**4 - 3*(self.delta - u + X)/self.delta**4 + 3*(u - X)/self.delta**4)/self.delta + 1/20*((self.delta - u + X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta + ((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta - 3*(4*self.delta + u - X)*(u - X)/self.delta**4 - 3*(u - X)**2/self.delta**4)/self.delta - 1/5*(5*self.delta + u - X)*(u - X)/self.delta**5 - 3/10*(u - X)**2/self.delta**5
        b3 = -1/20*(2*self.delta - u + X)*((((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta + (((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta + 3*(3*self.delta + u - X)/self.delta**4 - 3*(2*self.delta - u + X)/self.delta**4)/self.delta + 1/20*(4*self.delta + u - X)*((((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta + (4*self.delta + u - X)/self.delta**4 - 3*(self.delta - u + X)/self.delta**4 + 3*(u - X)/self.delta**4)/self.delta + 1/20*((3*self.delta + u - X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta - (2*self.delta - u + X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta + ((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - ((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta)/self.delta - 1/20*((self.delta - u + X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta + ((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta - 3*(4*self.delta + u - X)*(u - X)/self.delta**4 - 3*(u - X)**2/self.delta**4)/self.delta
        b4 = -1/20*(3*self.delta + u - X)*((((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta + (((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta + 3*(3*self.delta + u - X)/self.delta**4 - 3*(2*self.delta - u + X)/self.delta**4)/self.delta + 1/20*(3*self.delta - u + X)*((((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta - (3*self.delta - u + X)/self.delta**4 + 3*(2*self.delta + u - X)/self.delta**4 + 3*(self.delta + u - X)/self.delta**4)/self.delta - 1/20*((3*self.delta + u - X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)/self.delta**3 - 2*(self.delta - u + X)/self.delta**3 + 2*(u - X)/self.delta**3)/self.delta - (2*self.delta - u + X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta + ((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - ((self.delta - u + X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(3*self.delta + u - X)*(u - X)/self.delta**3 - (u - X)**2/self.delta**3)/self.delta)/self.delta - 1/20*((2*self.delta + u - X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta + ((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - 3*(3*self.delta - u + X)*(self.delta + u - X)/self.delta**4 + 3*(self.delta + u - X)**2/self.delta**4)/self.delta
        b5 = 1/20*(2*self.delta + u - X)*((((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta - (3*self.delta - u + X)/self.delta**4 + 3*(2*self.delta + u - X)/self.delta**4 + 3*(self.delta + u - X)/self.delta**4)/self.delta + 1/20*((2*self.delta + u - X)*(((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta + 2*(2*self.delta + u - X)/self.delta**3 - (2*self.delta - u + X)/self.delta**3 + 2*(self.delta + u - X)/self.delta**3)/self.delta + ((2*self.delta + u - X)*((2*self.delta + u - X)/self.delta**2 + (self.delta + u - X)/self.delta**2 - (self.delta - u + X)/self.delta**2 + (u - X)/self.delta**2)/self.delta - ((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta - 2*(2*self.delta - u + X)*(self.delta + u - X)/self.delta**3 + (self.delta + u - X)**2/self.delta**3)/self.delta - 3*(3*self.delta - u + X)*(self.delta + u - X)/self.delta**4 + 3*(self.delta + u - X)**2/self.delta**4)/self.delta - 1/5*(4*self.delta - u + X)*(self.delta + u - X)/self.delta**5 + 3/10*(self.delta + u - X)**2/self.delta**5
        b6 = -1/2*(self.delta + u - X)**2/self.delta**5
        return tf.concat([b1, b2, b3, b4, b5, b6], axis=0)

    def l2_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i(x) ϕ_j(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = 655177/1663200 * self.delta
        # d1 = 1623019/6652800 * self.delta
        # d2 = 1093/19800 * self.delta
        # d3 = 50879/13305600 * self.delta
        # d4 = 509/9979200 * self.delta
        # d5 = 1/39916800 * self.delta
        d0 = tf.cast([1/158400*self.delta,16559/1663200*self.delta,103673/554400*self.delta,103673/554400*self.delta,16559/1663200*self.delta,1/158400*self.delta], dtype=tf.float64)
        d1 = tf.cast([9113/39916800*self.delta,779353/19958400*self.delta,1650619/9979200*self.delta,779353/19958400*self.delta,9113/39916800*self.delta], dtype=tf.float64)
        d2 = tf.cast([14779/19958400*self.delta,536093/19958400*self.delta,536093/19958400*self.delta,14779/19958400*self.delta], dtype=tf.float64)
        d3 = tf.cast([41/105600*self.delta,40547/13305600*self.delta,41/105600*self.delta], dtype=tf.float64)
        d4 = tf.cast([509/19958400*self.delta,509/19958400*self.delta], dtype=tf.float64)
        d5 = tf.cast([1/39916800*self.delta], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3, d4, d5])

    def l2_grad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i'(x) ϕ_j'(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta * 809/2160
        # d1 = 1/self.delta * 1/64
        # d2 = 1/self.delta * -31/189
        # d3 = 1/self.delta * -907/24192
        # d4 = 1/self.delta * -25/18144
        # d5 = 1/self.delta * -1/362880
        d0 = tf.cast([1/5184/self.delta,10319/181440/self.delta,2953/22680/self.delta,2953/22680/self.delta,10319/181440/self.delta,1/5184/self.delta], dtype=tf.float64)
        d1 = tf.cast([1051/362880/self.delta,1409/25920/self.delta,-8971/90720/self.delta,1409/25920/self.delta,1051/362880/self.delta], dtype=tf.float64)
        d2 = tf.cast([23/18144/self.delta,-1511/18144/self.delta,-1511/18144/self.delta,23/18144/self.delta], dtype=tf.float64)
        d3 = tf.cast([-19/5184/self.delta,-2189/72576/self.delta,-19/5184/self.delta], dtype=tf.float64)
        d4 = tf.cast([-25/36288/self.delta,-25/36288/self.delta], dtype=tf.float64)
        d5 = tf.cast([-1/362880/self.delta], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3, d4, d5])

    def l2_ggrad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i''(x) ϕ_j''(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            di: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta**3 * 31/30
        # d1 = 1/self.delta**3 * -43/120
        # d2 = 1/self.delta**3 * -34/105
        # d3 = 1/self.delta**3 * 239/1680
        # d4 = 1/self.delta**3 * 29/1260
        # d5 = 1/self.delta**3 * 1/5040
        d0 = tf.cast([1/252/self.delta**3,47/315/self.delta**3,229/630/self.delta**3,229/630/self.delta**3,47/315/self.delta**3,1/252/self.delta**3], dtype=tf.float64)
        d1 = tf.cast([89/5040/self.delta**3,-479/2520/self.delta**3,-17/1260/self.delta**3,-479/2520/self.delta**3,89/5040/self.delta**3], dtype=tf.float64)
        d2 = tf.cast([-89/2520/self.delta**3,-319/2520/self.delta**3,-319/2520/self.delta**3,-89/2520/self.delta**3], dtype=tf.float64)
        d3 = tf.cast([1/504/self.delta**3,697/5040/self.delta**3,1/504/self.delta**3], dtype=tf.float64)
        d4 = tf.cast([29/2520/self.delta**3,29/2520/self.delta**3], dtype=tf.float64)
        d5 = tf.cast([1/5040/self.delta**3], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3, d4, d5])

    def l2_gggrad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i'''(x) ϕ_j'''(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            di: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta**5 * 23/5
        # d1 = 1/self.delta**5 * -11/4
        # d2 = tf.cast(0.0, dtype=tf.float64)
        # d3 = 1/self.delta**5 * 5/8
        # d4 = 1/self.delta**5 * -1/6
        # d5 = 1/self.delta**5 * -1/120
        d0 = tf.cast([1/20/self.delta**5,1/4/self.delta**5,2/self.delta**5,2/self.delta**5,1/4/self.delta**5,1/20/self.delta**5], dtype=tf.float64)
        d1 = tf.cast([-1/24/self.delta**5,-5/12/self.delta**5,-11/6/self.delta**5,-5/12/self.delta**5,-1/24/self.delta**5], dtype=tf.float64)
        d2 = tf.cast([-1/6/self.delta**5,1/6/self.delta**5,1/6/self.delta**5,-1/6/self.delta**5], dtype=tf.float64)
        d3 = tf.cast([1/4/self.delta**5,1/8/self.delta**5,1/4/self.delta**5], dtype=tf.float64)
        d4 = tf.cast([-1/12/self.delta**5,-1/12/self.delta**5], dtype=tf.float64)
        d5 = tf.cast([-1/120/self.delta**5], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3, d4, d5])


class B6Spline(SplineBasis):
    """
    Class to compute inner product matrices corresponding to B4 Spline basis functions
    """

    def __init__(self, a, b, m):
        self.order = 6
        super().__init__(a, b, m)

        # Precompute Static Matrices
        self.A = self.l2_inner_product()
        self.B = self.l2_grad_inner_product()
        self.C = self.l2_ggrad_inner_product()
        self.D = self.l2_gggrad_inner_product()

        # Boundary condition matrices
        self.BC = self.make_boundary_conditions(dx=0)
        self.BC_grad = self.make_boundary_conditions(dx=1)

    def _evaluate(self, u, X):
        b1 = 1/720*(u - X)**6/self.delta**6
        b2 = 1/720*(self.delta - u + X)*((self.delta - u + X)*((self.delta - u + X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta - (4*self.delta + u - X)*(u - X)**3/self.delta**4)/self.delta + (5*self.delta + u - X)*(u - X)**4/self.delta**5)/self.delta - 1/720*(6*self.delta + u - X)*(u - X)**5/self.delta**6
        b3 = 1/720*(2*self.delta - u + X)*((2*self.delta - u + X)*((2*self.delta - u + X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta + u - X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta)/self.delta + (4*self.delta + u - X)*((self.delta - u + X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta - (4*self.delta + u - X)*(u - X)**3/self.delta**4)/self.delta)/self.delta + 1/720*(5*self.delta + u - X)*((self.delta - u + X)*((self.delta - u + X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta - (4*self.delta + u - X)*(u - X)**3/self.delta**4)/self.delta + (5*self.delta + u - X)*(u - X)**4/self.delta**5)/self.delta
        b4 = 1/720*(3*self.delta - u + X)*((3*self.delta - u + X)*((2*self.delta + u - X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta - u + X)*(self.delta + u - X)**3/self.delta**4)/self.delta + (3*self.delta + u - X)*((2*self.delta - u + X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta + u - X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta)/self.delta)/self.delta + 1/720*(4*self.delta + u - X)*((2*self.delta - u + X)*((2*self.delta - u + X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta + u - X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta)/self.delta + (4*self.delta + u - X)*((self.delta - u + X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta - (4*self.delta + u - X)*(u - X)**3/self.delta**4)/self.delta)/self.delta
        b5 = 1/720*(3*self.delta + u - X)*((3*self.delta - u + X)*((2*self.delta + u - X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta - u + X)*(self.delta + u - X)**3/self.delta**4)/self.delta + (3*self.delta + u - X)*((2*self.delta - u + X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta + u - X)*((self.delta - u + X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (3*self.delta + u - X)*(u - X)**2/self.delta**3)/self.delta)/self.delta)/self.delta + 1/720*(4*self.delta - u + X)*((2*self.delta + u - X)*((2*self.delta + u - X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta - u + X)*(self.delta + u - X)**3/self.delta**4)/self.delta + (4*self.delta - u + X)*(self.delta + u - X)**4/self.delta**5)/self.delta
        b6 = 1/720*(2*self.delta + u - X)*((2*self.delta + u - X)*((2*self.delta + u - X)*((2*self.delta + u - X)*((self.delta + u - X)*(self.delta - u + X)/self.delta**2 - (2*self.delta + u - X)*(u - X)/self.delta**2)/self.delta + (2*self.delta - u + X)*(self.delta + u - X)**2/self.delta**3)/self.delta + (3*self.delta - u + X)*(self.delta + u - X)**3/self.delta**4)/self.delta + (4*self.delta - u + X)*(self.delta + u - X)**4/self.delta**5)/self.delta + 1/720*(5*self.delta - u + X)*(self.delta + u - X)**5/self.delta**6
        b7 = 1/720*(self.delta + u - X)**6/self.delta**6
        return tf.concat([b1, b2, b3, b4, b5, b6, b7], axis=0)

    def _evaluate_grad(self, u, X):
        b1 = -1/120*(u**5 - 5*u**4*X + 10*u**3*X**2 - 10*u**2*X**3 + 5*u*X**4 - X**5)/self.delta**6
        b2 = 1/120*(self.delta**5 - 5*self.delta**4*u + 10*self.delta**3*u**2 - 10*self.delta**2*u**3 + 5*self.delta*u**4 + 6*u**5 + 5*(self.delta + 6*u)*X**4 - 6*X**5 + 10*(self.delta**2 - 2*self.delta*u - 6*u**2)*X**3 + 10*(self.delta**3 - 3*self.delta**2*u + 3*self.delta*u**2 + 6*u**3)*X**2 + 5*(self.delta**4 - 4*self.delta**3*u + 6*self.delta**2*u**2 - 4*self.delta*u**3 - 6*u**4)*X)/self.delta**6
        b3 = 1/24*(5*self.delta**5 - 9*self.delta**4*u + 2*self.delta**3*u**2 + 6*self.delta**2*u**3 - 5*self.delta*u**4 - 3*u**5 - 5*(self.delta + 3*u)*X**4 + 3*X**5 - 2*(3*self.delta**2 - 10*self.delta*u - 15*u**2)*X**3 + 2*(self.delta**3 + 9*self.delta**2*u - 15*self.delta*u**2 - 15*u**3)*X**2 + (9*self.delta**4 - 4*self.delta**3*u - 18*self.delta**2*u**2 + 20*self.delta*u**3 + 15*u**4)*X)/self.delta**6
        b4 = 1/12*(4*self.delta**5 + 5*self.delta**4*u - 8*self.delta**3*u**2 - 2*self.delta**2*u**3 + 5*self.delta*u**4 + 2*u**5 + 5*(self.delta + 2*u)*X**4 - 2*X**5 + 2*(self.delta**2 - 10*self.delta*u - 10*u**2)*X**3 - 2*(4*self.delta**3 + 3*self.delta**2*u - 15*self.delta*u**2 - 10*u**3)*X**2 - (5*self.delta**4 - 16*self.delta**3*u - 6*self.delta**2*u**2 + 20*self.delta*u**3 + 10*u**4)*X)/self.delta**6
        b5 = -1/24*(8*self.delta**5 - 10*self.delta**4*u - 16*self.delta**3*u**2 + 4*self.delta**2*u**3 + 10*self.delta*u**4 + 3*u**5 + 5*(2*self.delta + 3*u)*X**4 - 3*X**5 - 2*(2*self.delta**2 + 20*self.delta*u + 15*u**2)*X**3 - 2*(8*self.delta**3 - 6*self.delta**2*u - 30*self.delta*u**2 - 15*u**3)*X**2 + (10*self.delta**4 + 32*self.delta**3*u - 12*self.delta**2*u**2 - 40*self.delta*u**3 - 15*u**4)*X)/self.delta**6
        b6 = -1/120*(25*self.delta**5 + 45*self.delta**4*u + 10*self.delta**3*u**2 - 30*self.delta**2*u**3 - 25*self.delta*u**4 - 6*u**5 - 5*(5*self.delta + 6*u)*X**4 + 6*X**5 + 10*(3*self.delta**2 + 10*self.delta*u + 6*u**2)*X**3 + 10*(self.delta**3 - 9*self.delta**2*u - 15*self.delta*u**2 - 6*u**3)*X**2 - 5*(9*self.delta**4 + 4*self.delta**3*u - 18*self.delta**2*u**2 - 20*self.delta*u**3 - 6*u**4)*X)/self.delta**6
        b7 = -1/120*(self.delta**5 + 5*self.delta**4*u + 10*self.delta**3*u**2 + 10*self.delta**2*u**3 + 5*self.delta*u**4 + u**5 + 5*(self.delta + u)*X**4 - X**5 - 10*(self.delta**2 + 2*self.delta*u + u**2)*X**3 + 10*(self.delta**3 + 3*self.delta**2*u + 3*self.delta*u**2 + u**3)*X**2 - 5*(self.delta**4 + 4*self.delta**3*u + 6*self.delta**2*u**2 + 4*self.delta*u**3 + u**4)*X)/self.delta**6
        return tf.concat([b1, b2, b3, b4, b5, b6, b7], axis=0)

    def _evaluate_ggrad(self, u, X):
        b1 = 1/24*(u**4 - 4*u**3*X + 6*u**2*X**2 - 4*u*X**3 + X**4)/self.delta**6
        b2 = 1/24*(self.delta**4 - 4*self.delta**3*u + 6*self.delta**2*u**2 - 4*self.delta*u**3 - 6*u**4 + 4*(self.delta + 6*u)*X**3 - 6*X**4 + 6*(self.delta**2 - 2*self.delta*u - 6*u**2)*X**2 + 4*(self.delta**3 - 3*self.delta**2*u + 3*self.delta*u**2 + 6*u**3)*X)/self.delta**6
        b3 = 1/24*(9*self.delta**4 - 4*self.delta**3*u - 18*self.delta**2*u**2 + 20*self.delta*u**3 + 15*u**4 - 20*(self.delta + 3*u)*X**3 + 15*X**4 - 6*(3*self.delta**2 - 10*self.delta*u - 15*u**2)*X**2 + 4*(self.delta**3 + 9*self.delta**2*u - 15*self.delta*u**2 - 15*u**3)*X)/self.delta**6
        b4 = -1/12*(5*self.delta**4 - 16*self.delta**3*u - 6*self.delta**2*u**2 + 20*self.delta*u**3 + 10*u**4 - 20*(self.delta + 2*u)*X**3 + 10*X**4 - 6*(self.delta**2 - 10*self.delta*u - 10*u**2)*X**2 + 4*(4*self.delta**3 + 3*self.delta**2*u - 15*self.delta*u**2 - 10*u**3)*X)/self.delta**6
        b5 = -1/24*(10*self.delta**4 + 32*self.delta**3*u - 12*self.delta**2*u**2 - 40*self.delta*u**3 - 15*u**4 + 20*(2*self.delta + 3*u)*X**3 - 15*X**4 - 6*(2*self.delta**2 + 20*self.delta*u + 15*u**2)*X**2 - 4*(8*self.delta**3 - 6*self.delta**2*u - 30*self.delta*u**2 - 15*u**3)*X)/self.delta**6
        b6 = 1/24*(9*self.delta**4 + 4*self.delta**3*u - 18*self.delta**2*u**2 - 20*self.delta*u**3 - 6*u**4 + 4*(5*self.delta + 6*u)*X**3 - 6*X**4 - 6*(3*self.delta**2 + 10*self.delta*u + 6*u**2)*X**2 - 4*(self.delta**3 - 9*self.delta**2*u - 15*self.delta*u**2 - 6*u**3)*X)/self.delta**6
        b7 = 1/24*(self.delta**4 + 4*self.delta**3*u + 6*self.delta**2*u**2 + 4*self.delta*u**3 + u**4 - 4*(self.delta + u)*X**3 + X**4 + 6*(self.delta**2 + 2*self.delta*u + u**2)*X**2 - 4*(self.delta**3 + 3*self.delta**2*u + 3*self.delta*u**2 + u**3)*X)/self.delta**6
        return tf.concat([b1, b2, b3, b4, b5, b6, b7], axis=0)

    def _evaluate_gggrad(self, u, X):
        b1 = -1/6*(u**3 - 3*u**2*X + 3*u*X**2 - X**3)/self.delta**6
        b2 = 1/6*(self.delta**3 - 3*self.delta**2*u + 3*self.delta*u**2 + 6*u**3 + 3*(self.delta + 6*u)*X**2 - 6*X**3 + 3*(self.delta**2 - 2*self.delta*u - 6*u**2)*X)/self.delta**6
        b3 = 1/6*(self.delta**3 + 9*self.delta**2*u - 15*self.delta*u**2 - 15*u**3 - 15*(self.delta + 3*u)*X**2 + 15*X**3 - 3*(3*self.delta**2 - 10*self.delta*u - 15*u**2)*X)/self.delta**6
        b4 = -1/3*(4*self.delta**3 + 3*self.delta**2*u - 15*self.delta*u**2 - 10*u**3 - 15*(self.delta + 2*u)*X**2 + 10*X**3 - 3*(self.delta**2 - 10*self.delta*u - 10*u**2)*X)/self.delta**6
        b5 = 1/6*(8*self.delta**3 - 6*self.delta**2*u - 30*self.delta*u**2 - 15*u**3 - 15*(2*self.delta + 3*u)*X**2 + 15*X**3 + 3*(2*self.delta**2 + 20*self.delta*u + 15*u**2)*X)/self.delta**6
        b6 = -1/6*(self.delta**3 - 9*self.delta**2*u - 15*self.delta*u**2 - 6*u**3 - 3*(5*self.delta + 6*u)*X**2 + 6*X**3 + 3*(3*self.delta**2 + 10*self.delta*u + 6*u**2)*X)/self.delta**6
        b7 = -1/6*(self.delta**3 + 3*self.delta**2*u + 3*self.delta*u**2 + u**3 + 3*(self.delta + u)*X**2 - X**3 - 3*(self.delta**2 + 2*self.delta*u + u**2)*X)/self.delta**6
        return tf.concat([b1, b2, b3, b4, b5, b6, b7], axis=0)

    def l2_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i(x) ϕ_j(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = 27085381/74131200 * self.delta
        # d1 = 125468459/518918400 * self.delta
        # d2 = 28218769/415134720 * self.delta
        # d3 = 910669/124540416 * self.delta
        # d4 = 82207/345945600 * self.delta
        # d5 = 1363/1037836800 * self.delta
        # d6 = 1/6227020800 * self.delta
        d0 = tf.cast([1/6739200*self.delta,465761/444787200*self.delta,14758057/222393600*self.delta,25637101/111196800*self.delta,14758057/222393600*self.delta,465761/444787200*self.delta,1/6739200*self.delta], dtype=tf.float64)
        d1 = tf.cast([71611/6227020800*self.delta,48441667/6227020800*self.delta,176074369/1556755200*self.delta,176074369/1556755200*self.delta,48441667/6227020800*self.delta,71611/6227020800*self.delta], dtype=tf.float64)
        d2 = tf.cast([153977/2075673600*self.delta,17452829/1556755200*self.delta,282735041/6227020800*self.delta,17452829/1556755200*self.delta,153977/2075673600*self.delta], dtype=tf.float64)
        d3 = tf.cast([28027/311351040*self.delta,4441237/1245404160*self.delta,4441237/1245404160*self.delta,28027/311351040*self.delta], dtype=tf.float64)
        d4 = tf.cast([17053/778377600*self.delta,603439/3113510400*self.delta,17053/778377600*self.delta], dtype=tf.float64)
        d5 = tf.cast([1363/2075673600*self.delta,1363/2075673600*self.delta], dtype=tf.float64)
        d6 = tf.cast([1/6227020800*self.delta], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3, d4, d5, d6])

    def l2_grad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i'(x) ϕ_j'(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            d1: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta * 4319/14400
        # d1 = 1/self.delta * 11731/302400
        # d2 = 1/self.delta * -6647/48384
        # d3 = 1/self.delta * -3455/72576
        # d4 = 1/self.delta * -2251/604800
        # d5 = 1/self.delta * -113/2217600
        # d6 = 1/self.delta * -1/39916800
        d0 = tf.cast([1/158400/self.delta,27103/2851200/self.delta,33889/285120/self.delta,6157/142560/self.delta,33889/285120/self.delta,27103/2851200/self.delta,1/158400/self.delta], dtype=tf.float64)
        d1 = tf.cast([8861/39916800/self.delta,228169/7983360/self.delta,-18773/1995840/self.delta,-18773/1995840/self.delta,228169/7983360/self.delta,8861/39916800/self.delta], dtype=tf.float64)
        d2 = tf.cast([1363/2661120/self.delta,-23623/1995840/self.delta,-915949/7983360/self.delta,-23623/1995840/self.delta,1363/2661120/self.delta], dtype=tf.float64)
        d3 = tf.cast([-703/1995840/self.delta,-187213/7983360/self.delta,-187213/7983360/self.delta,-703/1995840/self.delta], dtype=tf.float64)
        d4 = tf.cast([-181/498960/self.delta,-59803/19958400/self.delta,-181/498960/self.delta], dtype=tf.float64)
        d5 = tf.cast([-113/4435200/self.delta,-113/4435200/self.delta], dtype=tf.float64)
        d6 = tf.cast([-1/39916800/self.delta], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3, d4, d5, d6])

    def l2_ggrad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i''(x) ϕ_j''(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            di: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta**3 * 3101/4320
        # d1 = 1/self.delta**3 * -1807/10080
        # d2 = 1/self.delta**3 * -823/2688
        # d3 = 1/self.delta**3 * 3281/36288
        # d4 = 1/self.delta**3 * 2101/60480
        # d5 = 1/self.delta**3 * 83/60480
        # d6 = 1/self.delta**3 * 1/362880
        d0 = tf.cast([1/5184/self.delta**3,443/8640/self.delta**3,677/8640/self.delta**3,2969/6480/self.delta**3,677/8640/self.delta**3,443/8640/self.delta**3,1/5184/self.delta**3], dtype=tf.float64)
        d1 = tf.cast([109/40320/self.delta**3,-107/120960/self.delta**3,-5531/60480/self.delta**3,-5531/60480/self.delta**3,-107/120960/self.delta**3,109/40320/self.delta**3], dtype=tf.float64)
        d2 = tf.cast([-197/120960/self.delta**3,-4013/30240/self.delta**3,-4537/120960/self.delta**3,-4013/30240/self.delta**3,-197/120960/self.delta**3], dtype=tf.float64)
        d3 = tf.cast([-179/36288/self.delta**3,1213/24192/self.delta**3,1213/24192/self.delta**3,-179/36288/self.delta**3], dtype=tf.float64)
        d4 = tf.cast([1/336/self.delta**3,1741/60480/self.delta**3,1/336/self.delta**3], dtype=tf.float64)
        d5 = tf.cast([83/120960/self.delta**3,83/120960/self.delta**3], dtype=tf.float64)
        d6 = tf.cast([1/362880/self.delta**3], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3, d4, d5, d6])

    def l2_gggrad_inner_product(self):
        """
        Computes the L2 inner proudct ∫ ϕ_i'''(x) ϕ_j'''(x) dx for the B2 Spline basis
        Returns:
            d0: main diagonal for Kuu
            di: sub_diagonal for Kuu
        """
        # d0 = 1/self.delta**5 * 167/60
        # d1 = 1/self.delta**5 * -599/420
        # d2 = 1/self.delta**5 * -145/336
        # d3 = 1/self.delta**5 * 295/504
        # d4 = 1/self.delta**5 * -27/280
        # d5 = 1/self.delta**5 * -19/840
        # d6 = 1/self.delta**5 * -1/5040
        d0 = tf.cast([1/252/self.delta**5,33/280/self.delta**5,25/28/self.delta**5,95/126/self.delta**5,25/28/self.delta**5,33/280/self.delta**5,1/252/self.delta**5], dtype=tf.float64)
        d1 = tf.cast([23/1680/self.delta**5,-481/1680/self.delta**5,-37/84/self.delta**5,-37/84/self.delta**5,-481/1680/self.delta**5,23/1680/self.delta**5], dtype=tf.float64)
        d2 = tf.cast([-89/1680/self.delta**5,11/420/self.delta**5,-127/336/self.delta**5,11/420/self.delta**5,-89/1680/self.delta**5], dtype=tf.float64)
        d3 = tf.cast([47/1260/self.delta**5,143/560/self.delta**5,143/560/self.delta**5,47/1260/self.delta**5], dtype=tf.float64)
        d4 = tf.cast([1/105/self.delta**5,-97/840/self.delta**5,1/105/self.delta**5], dtype=tf.float64)
        d5 = tf.cast([-19/1680/self.delta**5,-19/1680/self.delta**5], dtype=tf.float64)
        d6 = tf.cast([-1/5040/self.delta**5], dtype=tf.float64)
        return self._make_banded_matrix([d0, d1, d2, d3, d4, d5, d6])


