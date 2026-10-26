"""
# Author: Yaqiu Wang
# File Name: utils.py
# Description: utilities for workflow of SPINDE
"""


import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from .preprocess import pca
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


_MCLUST_MODEL_TO_COVARIANCE = {
    # mclust modelNames -> sklearn covariance_type approximation.
    # EEE is the common setting used by spatial omics tutorials: all clusters
    # share one full covariance matrix, equivalent to sklearn's "tied".
    "EEE": "tied",
    "VVV": "full",
    "EEI": "diag",
    "VEI": "diag",
    "EVI": "diag",
    "VVI": "diag",
    "EII": "spherical",
    "VII": "spherical",
}

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def mclust_python(
    adata,
    num_cluster,
    modelNames='EEE',
    used_obsm='emb_pca',
    random_seed=2020,
    max_iter=1000,
    n_init=10,
    reg_covar=1e-6,
):
    """
    Python approximation of R package mclust clustering.

    R mclust fits Gaussian mixture models by EM and can choose among covariance
    parameterizations. This implementation uses sklearn GaussianMixture with a
    mapping for the most common mclust modelNames. If multiple modelNames are
    provided, the model with the lowest sklearn BIC is selected.

    Args:
        adata: AnnData object.
        num_cluster: Number of mixture components.
        modelNames: String or list of mclust covariance model names. Supported
            approximations: EEE, VVV, EEI, VEI, EVI, VVI, EII, VII.
        used_obsm: Key in adata.obsm used as clustering input.
        random_seed: Random seed passed to sklearn GaussianMixture.
        max_iter: Maximum EM iterations.
        n_init: Number of EM initializations.
        reg_covar: Non-negative regularization added to covariance diagonal.

    Returns:
        AnnData with 1-indexed cluster labels stored in adata.obs['mclust'].
    """

    if used_obsm not in adata.obsm_keys():
        raise KeyError(f"Expected adata.obsm['{used_obsm}'] to exist.")
    if num_cluster <= 0:
        raise ValueError(f"num_cluster must be positive, got {num_cluster}.")

    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"adata.obsm['{used_obsm}'] must be 2-dimensional, got shape {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError(f"adata.obsm['{used_obsm}'] contains NaN or Inf.")
    if x.shape[0] < num_cluster:
        raise ValueError(
            f"num_cluster={num_cluster} cannot exceed number of samples={x.shape[0]}."
        )

    if isinstance(modelNames, str):
        model_names = [modelNames]
    else:
        model_names = list(modelNames)

    best_model = None
    best_bic = np.inf
    best_name = None
    errors = {}

    for model_name in model_names:
        model_key = str(model_name).upper()
        if model_key not in _MCLUST_MODEL_TO_COVARIANCE:
            errors[model_key] = "unsupported modelName"
            continue

        covariance_type = _MCLUST_MODEL_TO_COVARIANCE[model_key]
        try:
            gm = GaussianMixture(
                n_components=num_cluster,
                covariance_type=covariance_type,
                random_state=random_seed,
                max_iter=max_iter,
                n_init=n_init,
                reg_covar=reg_covar,
                init_params="kmeans",
            )
            gm.fit(x)
            bic = gm.bic(x)
        except Exception as exc:
            errors[model_key] = str(exc)
            continue

        if bic < best_bic:
            best_model = gm
            best_bic = bic
            best_name = model_key

    if best_model is None:
        raise RuntimeError(
            "mclust_python failed to fit any model. "
            f"Tried modelNames={model_names}; errors={errors}."
        )

    labels = best_model.predict(x) + 1
    adata.obs['mclust'] = labels.astype(int)
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    adata.uns['mclust_python'] = {
        'modelName': best_name,
        'covariance_type': best_model.covariance_type,
        'bic': float(best_bic),
        'converged': bool(best_model.converged_),
        'n_iter': int(best_model.n_iter_),
    }
    return adata


def clustering(adata, n_clusters=7, key='emb', add_key='SPINDE', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20):

    """
    Cluster cells in AnnData object using specified method.

    adata : AnnData
        Input AnnData object containing cell embeddings.
    n_clusters : int, optional
        Target number of clusters (default: 7).
    key : str, optional
        Key in adata.obsm for embeddings to use (default: 'emb').
    add_key : str, optional
        Key in adata.obs to store clustering result (default: 'SPINDE').
    method : str, optional
        Clustering method: 'mclust', 'leiden', or 'louvain' (default: 'mclust').
    start, end, increment : float, optional
        Range and step for searching resolution (used for leiden/louvain).
    use_pca : bool, optional
        Whether to perform PCA on embeddings before clustering (default: False).
    n_comps : int, optional
        Number of PCA components if use_pca is True (default: 20).

    Returns: None
        Clustering labels are added to adata.obs[add_key].
    """
    
    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    
    if method == 'mclust':
       if use_pca: 
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'mclust_python':
       if use_pca:
          adata = mclust_python(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
       else:
          adata = mclust_python(adata, used_obsm=key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']             #saved in adata.obs[add_key]
    elif method == 'louvain':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']
       
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    
    '''
    Searching corresponding resolution according to given cluster number
    
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns: res, float Resolution.
        
    '''
    
    print('Searching resolution...')
    label = 0
    
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    
    #binary search
    maybe_res = sorted(list(np.arange(start, end, increment)), reverse=False)
    i = 0
    j = len(maybe_res) - 1
    middle = (i + j) // 2
    while(i <= j):
        res = maybe_res[middle]
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break
        elif count_unique > n_clusters:       
            j = middle - 1
            middle = (i + j) // 2
        else:
           i = middle + 1
           middle = (i + j) // 2

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res     

def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
  
  """
  Plotting weight values
  
  """  
  
  df = pd.DataFrame(columns=[modality1, modality2, 'label'])  
  df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label_SPINDE', 'Modality', 'Weight value']
  ax = sns.violinplot(data=df, x='label_SPINDE', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False)
  ax.set_title(modality1 + ' vs ' + modality2) 

  plt.tight_layout(w_pad=0.05)
  plt.show()     

