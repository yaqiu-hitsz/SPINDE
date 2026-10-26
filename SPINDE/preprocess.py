"""
# Author: Yaqiu Wang
# File Name: preprocess.py
# Description:  preprocess tools of SPINDE
"""

import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix, csr_matrix, issparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
    
def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3): 

    if sp.sparse.issparse(adata_omics1.obsm['feat']):
        adata_omics1.obsm['feat'] = adata_omics1.obsm['feat'].toarray()
    if sp.sparse.issparse(adata_omics2.obsm['feat']):
        adata_omics2.obsm['feat'] = adata_omics2.obsm['feat'].toarray()
    
    """
    Construct neighbor graphs, including feature graph and spatial graph. 
    Feature graph is based expression data while spatial graph is based on cell/spot spatial coordinates.


    Args: n_neighbors, Number of neighbors.

    Returns: data : dict, AnnData objects with preprossed data for different omics.

    """

    # construct spatial neighbor graphs
    ################# spatial graph #################
    if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
       n_neighbors=6 
    

    # omics1
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1
    
    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2
    
    ################# feature graph #################
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2
    
    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}
    
    return data

def pca(adata, use_reps=None, n_comps=10):
    
    """
    Performs principal component analysis (PCA) for dimensionality reduction on the input AnnData object.

    Args:
        adata: AnnData object containing the data matrix (.X) or a specified representation (.obsm).
        use_reps: Optional string key. If provided, uses adata.obsm[use_reps] as input for PCA; otherwise uses adata.X.
        n_comps: Number of principal components to compute (int).

    Returns:
        feat_pca: Numpy array of shape (n_samples, n_comps), the reduced feature matrix after PCA transformation.
    """

    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca

def clr_normalize_each_cell(adata, inplace=True):
    
    """
    Performs centered log-ratio (CLR) normalization on each cell (row) in the AnnData object.

    Args:
        adata: AnnData object containing the count matrix (.X).
        inplace: If True, modifies adata.X in place; if False, returns a copy of the AnnData object.

    Returns:
        adata: The AnnData object with CLR-normalized .X matrix (dense numpy array).
    """

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if sp.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     



def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode= "connectivity", metric="correlation", include_self=False):
    
    """
    Constructs feature-based neighbor graphs for two omics datasets using k-nearest neighbors.

    Args:
        adata_omics1: AnnData object for omics1, with features in .obsm['feat'].
        adata_omics2: AnnData object for omics2, with features in .obsm['feat'].
        k: Number of neighbors for each node (int).
        mode: Type of graph to construct ('connectivity' or 'distance').
        metric: Similarity metric to use for neighbor search (e.g., 'correlation').
        include_self: Whether to include self-loops in the graph (bool).

    Returns:
        feature_graph_omics1: Sparse matrix representing the feature neighbor graph for omics1.
        feature_graph_omics2: Sparse matrix representing the feature neighbor graph for omics2.
    """
    
    feature_graph_omics1=kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    feature_graph_omics2=kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)

    return feature_graph_omics1, feature_graph_omics2

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    #print('n_neighbor:', n_neighbors)
    """Constructing spatial neighbor graph according to spatial coordinates."""
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)  
    _ , indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj

def transform_adjacent_matrix(adjacent):
    
    """
    Converts a pandas DataFrame of edge list (adjacent) into a scipy sparse adjacency matrix.

    Args:
        adjacent: DataFrame with columns ['x', 'y', 'value'], representing edges from node x to node y with given value.

    Returns:
        adj: Scipy sparse COO matrix of shape (n_spot, n_spot), where n_spot is the maximum node index + 1.
    """

    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    """
    Converts a scipy sparse matrix to a PyTorch sparse tensor.

    Args:
        sparse_mx: Scipy sparse matrix (e.g., COO, CSR, CSC format).

    Returns:
        torch_sparse_tensor: PyTorch sparse FloatTensor with the same shape and values as the input matrix.
    """
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# ====== Graph preprocessing
def preprocess_graph(adj):
    
    """
    Preprocesses an adjacency matrix for graph neural networks by applying symmetric normalization.

    Args:
        adj: Input adjacency matrix (numpy array or scipy sparse matrix).

    Returns:
        torch_sparse_tensor: PyTorch sparse FloatTensor representing the normalized adjacency matrix.
    """

    adj = sp.sparse.coo_matrix(adj)        
    adj_ = adj + sp.sparse.eye(adj.shape[0])       
    rowsum = np.array(adj_.sum(1))      
    degree_mat_inv_sqrt = sp.sparse.diags(np.power(rowsum, -0.5).flatten())        
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo() 
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)     

def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    
    """
    Converts dense adjacency matrices for spatial and feature graphs in two omics datasets 
    into normalized sparse matrices suitable for graph neural networks.

    Args:
        adata_omics1: AnnData object for omics1, with spatial and feature adjacency info.
        adata_omics2: AnnData object for omics2, with spatial and feature adjacency info.

    Returns:
        adj: Dictionary containing normalized sparse adjacency matrices for both spatial and feature graphs:
            - 'adj_spatial_omics1'
            - 'adj_spatial_omics2'
            - 'adj_feature_omics1'
            - 'adj_feature_omics2'
    """
    
    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)
    
    adj_spatial_omics1 = adj_spatial_omics1.toarray()   # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()
    
    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1>1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2>1, 1, adj_spatial_omics2)
    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1) # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)
    
    ######################################## construct feature graph ########################################
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())
    
    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1>1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2>1, 1, adj_feature_omics2)
    
    # convert dense matrix to sparse matrix
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1) # sparse adjacent matrix corresponding to feature graph
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2)
    
    adj = {'adj_spatial_omics1': adj_spatial_omics1,
           'adj_spatial_omics2': adj_spatial_omics2,
           'adj_feature_omics1': adj_feature_omics1,
           'adj_feature_omics2': adj_feature_omics2,
           }
    
    return adj

def lsi(
        adata, n_components = 20,
        use_highly_variable = None, **kwargs
       ):
    
    """
    Performs Latent Semantic Indexing (LSI) dimensionality reduction on the input AnnData object, 
    following the Seurat v3 approach.

    Args:
        adata: AnnData object containing the count matrix (.X).
        n_components: Number of LSI components to compute (int).
        use_highly_variable: Whether to use highly variable features (bool or None).
        **kwargs: Additional arguments passed to sklearn.utils.extmath.randomized_svd.

    Returns:
        None. The LSI components are stored in adata.obsm["X_lsi"].
    """

    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):

    """
    Applies TF-IDF normalization to the input count matrix, following the Seurat v3 approach.

    For each feature (column), computes the inverse document frequency (IDF).
    For each cell (row), computes term frequency (TF) as the normalized counts.
    Returns the element-wise product TF * IDF for dense or sparse matrices.

    Args:
        X: Input count matrix (numpy array or scipy sparse matrix), shape (n_cells, n_features).

    Returns:
        X_tfidf: TF-IDF normalized matrix of the same shape as X.
    """

    if scipy.sparse.issparse(X):
        X = X.tocsr()
        row_sum = np.asarray(X.sum(axis=1)).reshape(-1)
        col_sum = np.asarray(X.sum(axis=0)).reshape(-1)
        inv_row_sum = np.divide(
            1.0,
            row_sum,
            out=np.zeros_like(row_sum, dtype=np.float64),
            where=row_sum > 0,
        )
        idf = np.divide(
            X.shape[0],
            col_sum,
            out=np.zeros_like(col_sum, dtype=np.float64),
            where=col_sum > 0,
        )
        tf = X.multiply(inv_row_sum[:, None])
        X_tfidf = tf.multiply(idf[None, :])
        X_tfidf.data = np.nan_to_num(X_tfidf.data, nan=0.0, posinf=0.0, neginf=0.0)
        return X_tfidf
    else:
        X = np.asarray(X)
        row_sum = X.sum(axis=1, keepdims=True)
        col_sum = X.sum(axis=0, keepdims=True)
        tf = np.divide(
            X,
            row_sum,
            out=np.zeros_like(X, dtype=np.float64),
            where=row_sum > 0,
        )
        idf = np.divide(
            X.shape[0],
            col_sum,
            out=np.zeros_like(col_sum, dtype=np.float64),
            where=col_sum > 0,
        )
        return np.nan_to_num(tf * idf, nan=0.0, posinf=0.0, neginf=0.0)
    
def fix_seed(seed):

    """
    Sets random seeds for Python, NumPy, PyTorch, and CUDA to ensure reproducible results.

    Args:
        seed: Integer seed value to set for all random number generators.

    Returns:
        None.
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def _compute_size_factors(counts, normalize_by_median=True, min_value=1e-8):
    """
    Compute positive exposure factors from per-cell / per-spot total counts.

    These size factors are stored in ``adata.obs['size_factors']`` and later used
    by count decoders as multiplicative exposure terms.
    """
    counts = np.asarray(counts, dtype=np.float32).reshape(-1)
    if counts.size == 0:
        return counts

    counts = np.clip(counts, a_min=0.0, a_max=None)
    if normalize_by_median:
        positive = counts[counts > 0]
        denom = np.median(positive) if positive.size > 0 else 1.0
        denom = max(float(denom), min_value)
        size_factors = counts / denom
    else:
        size_factors = counts

    return np.clip(size_factors, a_min=min_value, a_max=None).astype(np.float32)


def RNA_preproces(adata, min_cells=10, n_top_genes=3000, n_comps=100):
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    
    adata.obsm['raw']=adata.X.copy()
    raw_counts_per_cell = np.asarray(adata.X.sum(axis=1)).reshape(-1)
    sc.pp.normalize_per_cell(adata)
    # RNA count decoders use per-cell sequencing depth as exposure.
    adata.obs['size_factors'] = _compute_size_factors(raw_counts_per_cell, normalize_by_median=True)

    sc.pp.log1p(adata)

    adata.obsm['feat']=adata.X

    sc.pp.scale(adata)
    
    adata.obsm['raw']=adata.obsm['raw'][:, adata.var['highly_variable']]
    adata.obsm['feat'] = adata.obsm['feat'][:, adata.var['highly_variable']]
    adata= adata[:, adata.var['highly_variable']]

    if issparse(adata.X):
        adata.X=adata.X.toarray()
    if issparse(adata.obsm['raw']):
        adata.obsm['raw']=adata.obsm['raw'].toarray()
    if issparse(adata.obsm['feat']):
        adata.obsm['feat']=adata.obsm['feat'].toarray() 
    
    adata.obsm['feat'] =PCA(n_components=n_comps, random_state=42).fit_transform(adata.X)

    return adata


def RNA_preprocess(adata, min_cells=10, n_top_genes=3000, n_comps=100):
    """Compatibility wrapper for the corrected RNA preprocessing name."""
    return RNA_preproces(
        adata,
        min_cells=min_cells,
        n_top_genes=n_top_genes,
        n_comps=n_comps,
    )



def ADT_preprocess(adata):
    adata.obsm['raw']=adata.X

    if issparse(adata.X):
        adata.X=adata.X.toarray()
    if issparse(adata.obsm['raw']):
        adata.obsm['raw']=adata.obsm['raw'].toarray()
    
    raw_n_counts=adata.X.sum(axis=1)
    # Keep an exposure-like scale for count decoders even though ADT may also use MSE.
    adata.obs['size_factors'] = _compute_size_factors(raw_n_counts, normalize_by_median=True)
    adata= clr_normalize_each_cell(adata)
    adata.obsm['feat'] = np.asarray(adata.X, dtype=np.float32)

    return adata


def ATAC_preprocess(adata, n_comps=100):
    adata.obsm['raw']=adata.X
    raw_n_counts = np.asarray(adata.X.sum(axis=1)).reshape(-1)

    if issparse(adata.X):
        adata.X=adata.X.toarray()
    if issparse(adata.obsm['raw']):
        adata.obsm['raw']=adata.obsm['raw'].toarray()
    
    if 'X_lsi' not in adata.obsm.keys():
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        adata.obsm['raw'] = adata.X[:, adata.var['highly_variable']]
        lsi(adata, use_highly_variable=False, n_components=n_comps)

    # For sparse ATAC count decoders, especially Hurdle-Poisson, the exposure term should
    # come from each spot/cell's total accessibility load. We use raw total counts and
    # normalize by the median for numerical stability.
    adata.obs['size_factors'] = _compute_size_factors(raw_n_counts, normalize_by_median=True)
    adata.obsm['feat'] = adata.obsm['X_lsi'].copy()

    return adata
