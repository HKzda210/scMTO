import os
import scipy
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import scipy.sparse

def get_distance(x):
    X = np.mat(x)
    M = np.sum(np.multiply(X, X), 1)
    dist = M + M.T - 2 * X * X.T
    dist[dist < 0] = 0
    dist = np.sqrt(dist)
    dist = np.asarray(np.maximum(dist, dist.T))
    return dist

def get_M(S, t=1):
    M = np.zeros_like(S)
    step = t
    while step > 0:
        M = M + S ** step
        step = step - 1
    M = M / t
    return M

def graph_Laplacian(X, n_k):
    S = np.zeros([X.shape[0], X.shape[0]])
    dist = get_distance(X)
    gaussian_similarity_function = np.exp(- dist**2 / np.max(dist)**1.5)
    for i in range(X.shape[0]):
        cur_dist = dist[i, :]
        cur_sorted = np.argsort(cur_dist)[1: n_k+1]
        S[i, cur_sorted] = gaussian_similarity_function[i, cur_sorted]
    S = np.maximum(S, S.T)
    S = get_M(S, t=2)
    Degree = np.diag(np.sum(S, axis=0))
    Laplacian = Degree - S
    return Laplacian, S, Degree

def get_distance_sparse(x, k=10):
    X = np.mat(x)
    M = np.sum(np.multiply(X, X), 1)
    n_samples = X.shape[0]
    row_indices = []
    col_indices = []
    distances = []
    for i in range(n_samples):
        print(f'Calculating the distances: cell {i}')
        dist_i = M[i] + M.T - 2 * X[i, :] * X.T
        dist_i = np.array(dist_i).flatten()
        dist_i[dist_i < 0] = 0
        dist_i = np.sqrt(dist_i)
        knn_indices = np.argsort(dist_i)[1:k+1]
        for j in knn_indices:
            row_indices.append(i)
            col_indices.append(j)
            distances.append(dist_i[j])
    dist_sparse = csr_matrix((distances, (row_indices, col_indices)), shape=(n_samples, n_samples))
    dist_sparse = dist_sparse.maximum(dist_sparse.T)
    return dist_sparse

def get_M_sparse(S, t=1):
    M = sp.lil_matrix(S.shape)
    step = t
    while step > 0:
        M = M + (S ** step)
        step = step - 1
    M = M / t
    return M

def graph_Laplacian_sparse(X, n_k):
    S = sp.lil_matrix((X.shape[0], X.shape[0]))
    if not os.path.exists('dist_matrix.npz'): 
        print('Begin: calculating the distance matrix')
        dist = get_distance_sparse(X)
        scipy.sparse.save_npz('dist_matrix.npz', dist)
        print('End: calculating the distance matrix')
    else:
        dist = scipy.sparse.load_npz('dist_matrix.npz')
        print('Read dist_matrix done')
    dist_max = dist.max() ** 1.5
    gaussian_similarity_function = dist.copy()
    gaussian_similarity_function.data = np.exp(-gaussian_similarity_function.data ** 2 / dist_max)
    for i in range(X.shape[0]):
        cur_neighbors = gaussian_similarity_function[i].nonzero()[1]
        if len(cur_neighbors) > n_k:
            cur_neighbors = cur_neighbors[:n_k]
        S[i, cur_neighbors] = gaussian_similarity_function[i, cur_neighbors]
    S = S.maximum(S.T)
    S = get_M_sparse(S, t=2)
    Degree = S.sum(axis=0).A1 
    Degree_matrix = sp.lil_matrix((S.shape[0], S.shape[0]))  
    Degree_matrix.setdiag(Degree) 
    Laplacian = Degree_matrix - S
    return Laplacian, S, Degree_matrix

