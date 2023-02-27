import tensorflow as tf
import numpy as np
import gpflow
import banded_matrices.banded as banded
import scipy.sparse as sparse

def symmetrise_banded(K_lower):
    K_upper = banded.transpose_band(K_lower, K_lower.shape[0]-1, 0)
    return tf.concat([K_upper[:-1,:], K_lower], axis=0)

# def sparse_to_band(kernel, K_sparse):

#     if isinstance(kernel, gpflow.kernels.Matern12):
#         d0 = K_sparse.diagonal().reshape(1,-1)
#         d1 = np.pad(K_sparse.diagonal(k=-1), (0,1)).reshape(1,-1)
#         return np.concatenate([d0, d1], axis=0)

#     if isinstance(kernel, gpflow.kernels.Matern32):
#         d0 = K_sparse.diagonal().reshape(1,-1)
#         d1 = np.pad(K_sparse.diagonal(k=-1), (0,1)).reshape(1,-1)
#         d2 = np.pad(K_sparse.diagonal(k=-2), (0,2)).reshape(1,-1)
#         return np.concatenate([d0, d1, d2], axis=0)

def sparse_to_band(K_sparse, bandwidth):
    diags = [K_sparse.diagonal().reshape(1,-1)]
    for i in range(1, bandwidth+1):
        diag = tf.cast(K_sparse.diagonal(k=-i).reshape(1,-1), tf.float64)
        pad = tf.zeros([1,i], dtype=tf.float64)
        diags.append(tf.concat([diag, pad], axis=1))
    return tf.concat(diags, axis=0)

def band_to_sparse(K_lower):
    return sparse.spdiags(K_lower, np.arange(0,-(K_lower.shape[0]), -1), K_lower.shape[1], K_lower.shape[1])

def band_to_tfband(K_lower):
    lower_bandwidth = K_lower.shape[0] - 1
    return tf.reverse(banded.transpose_band(K_lower, lower_bandwidth, 0), axis=[0])

def band_to_kron_band(K_sparse, mat_bandwidth):
    K_dense = [banded.unpack_banded_matrix_to_dense(k, mat_bandwidth, 0) for k in K_sparse]
    operators = [tf.linalg.LinearOperatorFullMatrix(k) for k in K_dense]
    Kuu = tf.linalg.LinearOperatorKronecker(operators)
    return banded.pack_dense_matrix_to_banded(Kuu.to_dense(), mat_bandwidth, 0)

def bands_to_kron_cholesky(K_sparse, mat_bandwidth):
    K_sym = [symmetrise_banded(k) for k in K_sparse]
    K_dense = [banded.unpack_banded_matrix_to_dense(k, mat_bandwidth, mat_bandwidth) for k in K_sym]
    K_ops = [tf.linalg.LinearOperatorFullMatrix(K) for K in K_dense]
    Ls = [tf.linalg.cholesky(K) for K in K_dense]
    Ls_ops = [tf.linalg.LinearOperatorFullMatrix(L) for L in Ls]
    return tf.linalg.LinearOperatorKronecker(K_ops).to_dense(), tf.linalg.LinearOperatorKronecker(Ls_ops).to_dense()

def bands_to_sparse(K_bands, mat_bandwidth):
    K_sym = [symmetrise_banded(k) for k in K_bands]
    K_np = [banded.unpack_banded_matrix_to_dense(k, mat_bandwidth, mat_bandwidth).numpy() for k in K_sym]
    K_sparse = [sparse.csc_matrix(k) for k in K_np]
    return sparse.kron(K_sparse[0], K_sparse[1])









