import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from functools import reduce
import numpy as np
import gpflow
import tensorflow as tf
from banded_matrices import banded
import scipy.sparse as sparse
from sksparse.cholmod import cholesky as sparse_cholesky

from asvgp.inducing_features import SplineFeatures1D
import asvgp.utils as utils
import asvgp.kronecker as kron



class GPR_1d(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(self, data, kernel, basis):

        # Check inputs
        assert isinstance(kernel, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52))
        assert data[0].shape[1] == 1
        self.X, self.y = data
        assert np.all(self.X > basis.a)
        assert np.all(self.X < basis.b)

        # Init model
        likelihood = gpflow.likelihoods.Gaussian()
        mean_function = gpflow.mean_functions.Zero()
        num_latent_gps = self.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)
        self.basis = basis
        self.inducing_features = SplineFeatures1D(kernel, basis)

        # Bandwidth
        self.bandwidth = self.basis.order

        # Precompute static quantities
        Kuf = self.inducing_features.make_Kuf(self.X)
        self.Kuf_y = tf.constant(Kuf @ self.y, dtype=tf.float64)
        self.KufKfu_sparse = Kuf @ Kuf.T
        self.KufKfu = tf.constant(utils.sparse_to_band(self.KufKfu_sparse, self.bandwidth), dtype=tf.float64)
        self.tr_yTy = tf.constant(np.sum(np.square(self.y)))

    def maximum_log_likelihood_objective(self):
        return tf.reduce_sum(self.elbo())

    def elbo(self):
        """ Variational bound on the log marginal likelihood
        """
        K_diag = self.kernel.K_diag(self.X)
        sigma2 = self.likelihood.variance

        Kuu = self.inducing_features.make_Kuu(self.kernel)
        L_Kuu = banded.cholesky_band(Kuu)
        log_det_Kuu = tf.reduce_sum(tf.math.log(tf.square(L_Kuu[0,:])))

        Kuu_inv = utils.symmetrise_banded(banded.inverse_from_cholesky_band(L_Kuu))
        Kuu_inv_KufKfu = banded.product_band_band(
            Kuu_inv,
            banded.symmetrise_band(self.KufKfu, self.bandwidth),
            left_lower_bandwidth=int(self.bandwidth),
            left_upper_bandwidth=int(self.bandwidth),
            right_lower_bandwidth=int(self.bandwidth),
            right_upper_bandwidth=int(self.bandwidth),
            result_lower_bandwidth=0,
            result_upper_bandwidth=0
        )
        trace_term = tf.reduce_sum(Kuu_inv_KufKfu)

        P = self.KufKfu / sigma2 + Kuu
        L_P = banded.cholesky_band(P)
        log_det_P = tf.reduce_sum(tf.math.log(tf.square(L_P[0,:])))
        # c = tf.linalg.banded_triangular_solve(utils.band_to_tfband(L_P), self.Kuf_y) / sigma2
        c = banded.solve_triang_mat(L_P, self.Kuf_y) / sigma2

        # Compute bound on log marginal likelihood
        ND = tf.cast(tf.size(self.y), tf.float64)
        D = tf.cast(tf.shape(self.y)[1], tf.float64)

        elbo =- 0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        elbo -= 0.5 * D * log_det_P
        elbo += 0.5 * D * log_det_Kuu
        elbo -= 0.5 * self.tr_yTy / sigma2
        elbo += 0.5 * tf.reduce_sum(tf.square(c))
        elbo -= 0.5 * tf.reduce_sum(K_diag) / sigma2
        elbo += 0.5 * trace_term / sigma2

        return elbo

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False, batch=False):
        # assert not full_output_cov

        # Kuu = self.inducing_features.make_Kuu(self.kernel)
        # L_Kuu = banded.cholesky_band(Kuu)
        # sigma2 = self.likelihood.variance

        # P = self.KufKfu / sigma2 + Kuu
        # L_P = banded.cholesky_band(P)
        # c = tf.linalg.banded_triangular_solve(utils.band_to_tfband(L_P), self.Kuf_y) / sigma2

        # Kus = self.inducing_features.make_Kuf(Xnew).todense()
        # tmp = tf.linalg.banded_triangular_solve(utils.band_to_tfband(L_P), Kus)
        # mean = tf.matmul(tf.transpose(tmp), c)
        # KiKus = banded.chol_solve_band_mat(L_Kuu, Kus)

        # if full_cov:
        #     var = self.kernel(Xnew)
        #     var += tf.matmul(tf.transpose(tmp), tmp)
        #     var -= tf.matmul(tf.transpose(KiKus), Kus)
        #     # shape = tf.stack([1, 1, tf.shape(self.y)[1]])
        #     # var = tf.tile(tf.expand_dims(var, 2), shape)
        # else:
        #     var = self.kernel.K_diag(Xnew)
        #     var += tf.reduce_sum(tf.square(tmp), 0)
        #     var -= tf.reduce_sum(KiKus * Kus, 0)
        #     shape = tf.stack([1, tf.shape(self.Kuf_y)[1]])
        #     var = tf.tile(tf.expand_dims(var, 1), shape)
        assert not full_output_cov

        def predict_f_per_batch(Xnew):

            Kuu = self.inducing_features.make_Kuu(self.kernel)
            Kuu = utils.band_to_sparse(Kuu)
            L_Kuu = sparse_cholesky(Kuu.tocsc(), ordering_method="natural")
            sigma2 = self.likelihood.variance * 1

            P = self.KufKfu_sparse / sigma2.numpy() + Kuu
            L_P = sparse_cholesky(P.tocsc(), ordering_method="natural")
            c = L_P.solve_L(self.Kuf_y, use_LDLt_decomposition=False) / sigma2

            Kus = self.inducing_features.make_Kuf(Xnew).tocsc()
            tmp = L_P.solve_L(Kus, use_LDLt_decomposition=False)
            mean = tmp.T @ c
            KiKus = L_Kuu(Kus)

            K_diag = np.repeat(self.kernel.variance, Xnew.shape[0])

            if full_cov:
                raise NotImplementedError
            else:
                var = K_diag
                var += np.sum(np.square(tmp.toarray()), axis=0).reshape(-1)
                var -= np.sum(Kus.multiply(KiKus).toarray(), axis=0).reshape(-1)
                var = var.reshape(-1,1)

            return mean, var

        if not batch:
            return predict_f_per_batch(Xnew)

        if batch:
            num_test = Xnew.shape[0]
            mean = np.zeros((num_test, 1))
            var = np.zeros((num_test, 1))

            for i in range(int(num_test / 10_000)):
                x = Xnew[i*10_000:(i+1)*10_000,:]
                mean_, var_ = predict_f_per_batch(x)
                mean[i*10_000:(i+1)*10_000,:] = mean_
                var[i*10_000:(i+1)*10_000,:] = var_

            return mean, var


class GPR_additive(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(self, data, kernels, bases):

        self.X, self.y = data
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]

        # Check dimensionality of inputs
        assert len(kernels) == len(bases) == self.d
        assert self.y.shape[1] == 1

        # Check valid kernels
        for kernel in kernels:
            assert isinstance(kernel, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52))

        likelihood = gpflow.likelihoods.Gaussian()
        super().__init__(kernel, likelihood, num_latent_gps=1)

        self.bases = bases
        self.kernels = kernels
        self.inducing_features = [SplineFeatures1D(self.kernels[i], self.bases[i]) for i in range(self.d)]

        # Bandwidth
        bandwidths = [basis.order for basis in self.bases]
        assert all(x==bandwidths[0] for x in bandwidths)
        self.bandwidth = self.bases[0].order

       # Precompute static quantities
        t = time.time()
        self.tr_yTy = tf.constant(np.sum(np.square(self.y)))
        Kuf = [self.inducing_features[i].make_Kuf(self.X[:, i : i+1]) for i in range(self.d)]
        Kuf = sparse.vstack(Kuf)
        self.KufKfu_sparse = (Kuf @ Kuf.T)
        self.KufKfu = tf.constant(self.KufKfu_sparse.todense())
        self.Kuf_y = tf.constant((Kuf @ self.y))

    def maximum_log_likelihood_objective(self):
        return tf.reduce_sum(self.elbo())

    def elbo(self):
        num_data = tf.shape(self.y)[0]
        output_dim = tf.shape(self.y)[1]

        total_variance = reduce(tf.add, [kernel.variance for kernel in self.kernels])
        Kuu_band = [self.inducing_features[i].make_Kuu(self.kernels[i]) for i in range(self.d)]
        Kuu_sym = [utils.symmetrise_banded(K) for K in Kuu_band]
        Kuu =[banded.unpack_banded_matrix_to_dense(K, self.bandwidth, self.bandwidth) for K in Kuu_sym]
        ops = [tf.linalg.LinearOperatorFullMatrix(K) for K in Kuu]
        Kuu = tf.linalg.LinearOperatorBlockDiag(ops)
        log_det_Kuu = Kuu.log_abs_determinant()
        sigma2 = self.likelihood.variance
        
        # Compute intermediate matrices
        P = Kuu.to_dense() + self.KufKfu / sigma2
        L = tf.linalg.cholesky(P)
        log_det_P = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L))))
        c = tf.linalg.triangular_solve(L, self.Kuf_y) / sigma2

        # compute log marginal bound
        ND = tf.cast(num_data * output_dim, tf.float64)
        D = tf.cast(output_dim, tf.float64)
        
        elbo = -0.5 * ND * tf.cast(tf.math.log(2 * np.pi * sigma2), tf.float64)
        elbo += -0.5 * D * log_det_P
        elbo += 0.5 * D * log_det_Kuu
        elbo += -0.5 * self.tr_yTy / sigma2
        elbo += 0.5 * tf.reduce_sum(tf.square(c))
        elbo += -0.5 * ND * total_variance / sigma2
        elbo += 0.5 * D * tf.linalg.trace(Kuu.solve(self.KufKfu)) / sigma2
        return elbo

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        
        total_variance = reduce(tf.add, [kernel.variance for kernel in self.kernels])
        Kuu_band = [self.inducing_features[i].make_Kuu(self.kernels[i]) for i in range(self.d)]
        Kuu_sym = [utils.symmetrise_banded(K) for K in Kuu_band]
        Kuu =[banded.unpack_banded_matrix_to_dense(K, self.bandwidth, self.bandwidth) for K in Kuu_sym]
        ops = [tf.linalg.LinearOperatorFullMatrix(K) for K in Kuu]
        Kuu = tf.linalg.LinearOperatorBlockDiag(ops)
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.to_dense()
        L = tf.linalg.cholesky(P)
        c = tf.linalg.triangular_solve(L, self.Kuf_y) / sigma2

        Kus = sparse.vstack([self.inducing_features[i].make_Kuf(Xnew[:, i:i+1]) for i in range(self.d)])
        tmp = tf.linalg.triangular_solve(L, Kus.todense())
        mean = tf.matmul(tf.transpose(tmp), c)
        KuuInv_Kus = Kuu.solve(Kus.todense())

        var = reduce(tf.add, [kernel.K_diag(Xnew[:, i:i+1]) for i, kernel in enumerate(self.kernels)])
        var += tf.reduce_sum(tf.square(tmp), 0)
        var -= tf.reduce_sum(KuuInv_Kus * Kus.todense(), 0)
        shape = tf.stack([1, tf.shape(self.y)[1]])
        var = tf.tile(tf.expand_dims(var, 1), shape)

        return mean, var


class GPR_kron(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(self, data, kernels, bases):

        self.X, self.y = data
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]

        # Check dimensionality of inputs
        assert len(kernels) == len(bases) == self.d
        assert self.y.shape[1] == 1

        # Check valid kernels
        for kernel in kernels:
            assert isinstance(kernel, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52))

        likelihood = gpflow.likelihoods.Gaussian()
        super().__init__(kernel, likelihood, num_latent_gps=1)
        self.bases = bases
        self.kernels = kernels

        # Bandwidth
        self.m = self.bases[0].m
        self.order = self.bases[0].order
        self.bandwidth = int((self.m**(self.d) - 1) * self.order / (self.m-1))

        # Initialise inducing features
        self.inducing_features = [SplineFeatures1D(self.kernels[i], self.bases[i]) for i in range(self.d)]

        # Precompute static quantities
        Kuf = [self.inducing_features[i].make_Kuf(self.X[:, i : i+1]) for i in range(self.d)]
        self.Kuf = kron.make_kvs_sparse(Kuf)
        self.Kuf_y = tf.constant(self.Kuf @ self.y, dtype=tf.float64)
        self.KufKfu_sparse = (self.Kuf @ self.Kuf.T)
        self.KufKfu_dense = tf.constant(self.KufKfu_sparse.todense(), tf.float64)
        self.KufKfu_band = utils.sparse_to_band(self.KufKfu_sparse, self.bandwidth)
        self.tr_yTy = np.sum(np.square(self.y))

        memory = self.Kuf.data.nbytes + self.Kuf.indptr.nbytes + self.Kuf.indices.nbytes
        print('Kuf num bytes = ',memory)

    def maximum_log_likelihood_objective(self):
        return tf.reduce_sum(self.elbo())

    def elbo(self):

        K_diag = reduce(tf.multiply, [kernel.K_diag(self.X[:,i:i+1]) for i, kernel in enumerate(self.kernels)])
        sigma2 = self.likelihood.variance

        Kuu_mats = [self.inducing_features[i].make_Kuu(self.kernels[i]) for i in range(self.d)]
        Kuu, L_Kuu = utils.bands_to_kron_cholesky(Kuu_mats, self.order)
        log_det_Kuu = tf.reduce_sum(tf.math.log(tf.math.square(tf.linalg.diag_part(L_Kuu))))

        # Compute intermeidate matrices
        P = Kuu + self.KufKfu_dense / sigma2
        L_P = tf.linalg.cholesky(P)
        log_det_P = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L_P))))
        c = tf.linalg.triangular_solve(L_P, self.Kuf_y) / sigma2

        # Compute log marginal bound
        ND = tf.cast(tf.size(self.y), tf.float64)
        D = tf.cast(tf.shape(self.y)[1], tf.float64)

        elbo =- 0.5 * ND * tf.cast(tf.math.log(2 * np.pi * sigma2), dtype=tf.float64)
        elbo -= 0.5 * D * log_det_P
        elbo += 0.5 * D * log_det_Kuu
        elbo -= 0.5 * self.tr_yTy / sigma2
        elbo += 0.5 * tf.reduce_sum(tf.square(c))
        elbo -= 0.5 * tf.reduce_sum(K_diag) / sigma2
        elbo += 0.5 * tf.linalg.trace(tf.linalg.cholesky_solve(L_Kuu, self.KufKfu_dense)) / sigma2
        return elbo

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        
        sigma2 = self.likelihood.variance

        Kuu_mats = [self.inducing_features[i].make_Kuu(self.kernels[i]) for i in range(self.d)]
        Kuu, L_Kuu = utils.bands_to_kron_cholesky(Kuu_mats, self.order)

        # Compute intermediate matrices
        P = Kuu + self.KufKfu_dense / sigma2
        L_P = tf.linalg.cholesky(P)
        c = tf.linalg.triangular_solve(L_P, self.Kuf_y) / sigma2

        Kus = [self.inducing_features[i].make_Kuf(Xnew[:, i : i + 1]) for i in range(self.d)]
        Kus = kron.make_kvs_sparse(Kus).todense()
        tmp = tf.linalg.triangular_solve(L_P, Kus)
        mean = tf.matmul(tf.transpose(tmp), c)
        KInv_Kus = tf.linalg.cholesky_solve(L_Kuu, Kus)

        var = reduce(tf.multiply, [kernel.K_diag(Xnew[:, i : i + 1]) for i, kernel in enumerate(self.kernels)])
        var += tf.reduce_sum(tf.square(tmp), 0)
        var -= tf.reduce_sum(KInv_Kus * Kus, 0)
        shape = tf.stack([1, tf.shape(self.y)[1]])
        var = tf.tile(tf.expand_dims(var, 1), shape)

        return mean, var

    def predict_f_sparse(self, Xnew, full_cov=False, full_output_cov=False):

        sigma2 = self.likelihood.variance.numpy()

        Kuu = [self.inducing_features[i].make_Kuu(self.kernels[i]) for i in range(self.d)]
        Kuu = utils.bands_to_sparse(Kuu, self.order)
        L_Kuu = sparse_cholesky(Kuu.tocsc(), ordering_method="natural")
        
        P = self.KufKfu_sparse / sigma2 + Kuu
        L_P = sparse_cholesky(P.tocsc(), ordering_method="natural")
        c = L_P.solve_L(self.Kuf_y, use_LDLt_decomposition=False) / sigma2

        Kus = [self.inducing_features[i].make_Kuf(Xnew[:, i : i + 1]) for i in range(self.d)]
        Kus = kron.make_kvs_sparse(Kus)
        tmp = L_P.solve_L(Kus.tocsc(), use_LDLt_decomposition=False)
        mean = tmp.T @ c
        KInv_Kus = L_Kuu(Kus.tocsc())

        var = reduce(tf.multiply, [kernel.K_diag(Xnew[:, i : i + 1]) for i, kernel in enumerate(self.kernels)])
        var += np.sum(np.square(tmp.toarray()), axis=0).reshape(-1)
        var -= np.sum(Kus.multiply(KInv_Kus).toarray(), axis=0).reshape(-1)
        var = tf.expand_dims(var, 1)

        return mean, var


        



if __name__ == "__main__":

    import asvgp.basis as basis
    import matplotlib.pyplot as plt

    dim = 2
    kernels = [gpflow.kernels.Matern32() for d in range(dim)]
    bases = [basis.B3Spline(0, 1, 2000) for d in range(dim)]

    X = np.random.uniform(0, 1, (1_000_000, 2))
    Xnew = np.random.uniform(0, 1, (10_000, 2))
    y = np.random.uniform(0, 1, (1_000_000, 1))

    print('made it')

    t = time.time()
    # model = GPR_kron((X, y), kernels, bases)
    model = GPR_additive((X,y), kernels, bases)
    print('Precompute = {}'.format(time.time()-t))

    mean, var = model.predict_f(Xnew)

    

    # fig = plt.figure(figsize=(8,5))
    # plt.imshow(model.KufKfu.todense())
    # plt.savefig('KufKfu.png', dpi=300)




        