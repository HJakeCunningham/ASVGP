from functools import reduce
import numpy as np
import tensorflow as tf
import scipy.sparse as sparse
import banded_matrices.banded as banded

def sparse_repeats(A, repeats):
    A = A.tocoo()
    n = A.row.shape[0]
    data = np.tile(A.data, repeats)
    row = np.zeros(data.shape)
    for i in range(0, repeats):
        row[i*n:(i+1)*n] = i + A.row*repeats
    col = np.tile(A.col, repeats)
    return sparse.csr_matrix((data, (row, col)), shape=(repeats * A.shape[0], A.shape[1]))

def sparse_tile(A, repeats):
    A = A.tocoo()
    n = A.row.shape[0]
    data = np.tile(A.data, repeats)
    row = np.zeros(data.shape)
    for i in range(0, repeats):
        row[i*n:(i+1)*n] = A.row + i*repeats
    col = np.tile(A.col, repeats)
    return sparse.csr_matrix((data, (row, col)), shape=(A.shape[0] * repeats, A.shape[1]))

def make_kvs_two_sparse(A, B):
    M1 = sparse_repeats(A, B.shape[0])
    M2 = sparse_tile(B, A.shape[0])
    return M1.multiply(M2)

def make_kvs_sparse(A_list):
    return reduce(make_kvs_two_sparse, A_list)

def kron_log_determinant(Kuu, M, d):
    M = tf.ones(d) * M
    L = [banded.cholesky_band(kuu) for kuu in Kuu]
    logdets = [tf.reduce_sum(tf.math.log(tf.square(l[0,:]))) for l in L]
    N = [np.prod(M) / m for m in M]
    return reduce(tf.add, [N * logdet for n, logdet in zip(N, logdets)])