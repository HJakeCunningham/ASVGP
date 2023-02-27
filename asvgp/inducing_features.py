import numpy as np
import gpflow
import tensorflow as tf
import asvgp.basis as basis

class SplineFeatures1D():

    def __init__(self, kernel, basis):
        self.kernel = kernel
        self.basis = basis

    def make_Kuu(self, kernel):
        """ Returns a banded Kuu matrix given a kernel function
        """

        if isinstance(kernel, gpflow.kernels.Matern12):
            A = 1 / (2 * kernel.lengthscales * kernel.variance) * self.basis.A
            B = kernel.lengthscales / (2 * kernel.variance) * self.basis.B
            BC = 1 / (2 * kernel.variance) * self.basis.BC
            return tf.cast(A + B + BC, tf.float64)

        elif isinstance(kernel, gpflow.kernels.Matern32):
            A = np.sqrt(3) / (4 * kernel.lengthscales * kernel.variance) * self.basis.A
            B = kernel.lengthscales / (2 * np.sqrt(3.0) * kernel.variance) * self.basis.B
            C = kernel.lengthscales**3 / (12 * np.sqrt(3.0) * kernel.variance) * self.basis.C

            BC1 = 1 / (2 * kernel.variance) * self.basis.BC
            BC2 = kernel.lengthscales**2 / (2 * kernel.variance) * self.basis.BC_grad

            return tf.cast(A + B + C + BC1 + BC2, tf.float64)

        elif isinstance(kernel, gpflow.kernels.Matern52):
            A = (3 * np.sqrt(5)) / (16 * kernel.lengthscales * kernel.variance) * self.basis.A
            B = (9 * kernel.lengthscales) / (16 * np.sqrt(5.0) * kernel.variance) * self.basis.B
            C = (9 * kernel.lengthscales**3) / (80 * np.sqrt(5.0) * kernel.variance) * self.basis.C
            D = (3 * kernel.lengthscales**5) / (400 * np.sqrt(5.0) * kernel.variance) * self.basis.D

            BC1 = 9 / (16 * kernel.variance) * self.basis.BC
            BC2 = (3 * kernel.lengthscales**2) / (10 * kernel.variance) * self.basis.BC_grad
            BC3 = (9 * kernel.lengthscales**4) / (400 * kernel.variance) * self.basis.BC_ggrad
            BC4 = (3 * kernel.lengthscales**2) / (80 * kernel.variance) * self.basis.BC_ggrad_none
            BC5 = (3 * kernel.lengthscales**2) / (80 * kernel.variance) * self.basis.BC_none_ggrad

            return tf.cast(A + B + C + D + BC1 + BC2 + BC3 + BC4 + BC5, tf.float64)


    def make_Kuf(self, X, sparse=True):
        return self.basis.evaluate_basis(X, dx=0, sparse=True)

        