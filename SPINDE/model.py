"""
# Author: Yaqiu Wang
# File Name: model.py
# Description:  network of SPINDE
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


def add_noise_to_features(features, noise_std=0.1, noise_type="gaussian", t=None):
    """
    Add Gaussian or diffusion-style noise to low-dimensional PCA/LSI features.
    """
    if noise_std <= 0:
        return features

    noise = torch.randn_like(features)
    if noise_type == "gaussian":
        return features + noise_std * noise
    if noise_type in {"diffusion", "diffusion-style"}:
        if t is None:
            t = torch.rand(features.size(0), 1, device=features.device, dtype=features.dtype)
        elif not torch.is_tensor(t):
            t = torch.tensor(t, device=features.device, dtype=features.dtype)
        if t.dim() == 0:
            t = t.view(1, 1).expand(features.size(0), 1)
        elif t.dim() == 1:
            t = t.view(-1, 1)
        alpha_bar = torch.clamp(1.0 - t, min=1e-4, max=1.0)
        return torch.sqrt(alpha_bar) * features + torch.sqrt(1.0 - alpha_bar) * noise_std * noise
    raise ValueError(f"Unsupported noise_type='{noise_type}'. Expected 'gaussian' or 'diffusion'.")


class Network(Module):
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, 
                 dim_out_feat_omics2, dropout=0.0, act=F.relu,
                 use_contrastive=True):
        super(Network, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act
        self.use_contrastive = use_contrastive
        
        # four encoders with two modalities and two views
        self.encoder_omics1_spatial = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.encoder_omics1_feature = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.encoder_omics2_spatial = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.encoder_omics2_feature = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)

        # four decoders with two modalities and two views
        self.decoder_omics1_spatial = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.decoder_omics1_feature = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.decoder_omics2_spatial = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        self.decoder_omics2_feature = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        
        # attention layers
        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1 , self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_cross = AttentionLayer(
            self.dim_out_feat_omics1,  # spatial + feature concatenated dimension
            self.dim_out_feat_omics2   # spatial + feature concatenated dimension
        )
        
        # contrastive learning related parameters
        self.contrastive_temp = 0.1
        self.k_spatial = 5
        self.k_feature = 5
        self.max_negative_samples = 500  # limit negative samples
        self.contrastive_update_freq = 10  # update contrastive pairs every 10 epochs
        
        # cache variables
        self.cached_spatial_pairs = None
        self.cached_feature_pairs_omics1 = None
        self.cached_feature_pairs_omics2 = None
        self.cached_negative_pairs_omics1 = None
        self.cached_negative_pairs_omics2 = None
        self.last_cache_epoch = -1

    def precompute_spatial_pairs(self, spatial_coords):
        
        """
        Computes spatial-based positive sample pairs for contrastive learning.

        Args:
            spatial_coords: Tensor of shape (N, C), spatial coordinates for each sample.

        Returns:
            pairs: Tensor of shape (M, 2), each row contains indices (i, j) representing a positive sample pair
                   where sample j is among the top-k nearest neighbors of sample i in spatial space.
                   If input is None or empty, returns an empty tensor.
        """

        if spatial_coords is None:
            return torch.empty(0, 2, dtype=torch.long, device=spatial_coords.device if spatial_coords is not None else torch.device('cpu'))
        
        coords_np = spatial_coords.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=min(self.k_spatial + 1, len(coords_np))).fit(coords_np)
        distances, indices = nbrs.kneighbors(coords_np)
        
        pairs = []
        for i in range(len(coords_np)):
            for j in range(1, min(self.k_spatial + 1, indices.shape[1])):
                pairs.append([i, indices[i, j]])
        
        return torch.tensor(pairs, dtype=torch.long, device=spatial_coords.device)

    def compute_feature_pairs_gpu(self, features):
        
        """
        Computes feature-based positive sample pairs for contrastive learning using GPU.

        Args:
            features: Tensor of shape (N, D), feature matrix for one omics modality.

        Returns:
            pairs: Tensor of shape (M, 2), each row contains indices (i, j) representing a positive sample pair
                   where sample j is among the top-k most similar to sample i in feature space.
                   If input is empty, returns an empty tensor.
        """

        if len(features) == 0:
            return torch.empty(0, 2, dtype=torch.long, device=features.device)
        
        # normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # calculate similarity matrix
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        # find top-k samples
        _, top_k_indices = torch.topk(similarity_matrix, k=min(self.k_feature + 1, similarity_matrix.size(1)), dim=1)
        
        pairs = []
        for i in range(similarity_matrix.size(0)):
            for j in range(1, min(self.k_feature + 1, top_k_indices.size(1))):  # skip self
                pairs.append([i, top_k_indices[i, j].item()])
        
        return torch.tensor(pairs, dtype=torch.long, device=features.device)

    def sample_negative_pairs(self, positive_pairs, batch_size, max_samples=None):
        
        """
        Randomly samples negative pairs for contrastive learning by excluding all positive pairs and self-loops.

        Args:
            positive_pairs: Tensor of shape (M, 2), each row contains indices (i, j) representing a positive sample pair.
            batch_size: Number of samples in the batch (int).
            max_samples: Maximum number of negative pairs to sample (int). If None, uses self.max_negative_samples.

        Returns:
            neg_pairs: Tensor of shape (K, 2), each row contains indices (i, j) representing a negative sample pair,
                       where (i, j) is not a positive pair and i ≠ j. If no pairs found, returns an empty tensor.
        """   

        if max_samples is None:
            max_samples = self.max_negative_samples
            
        # positive sample set
        pos_set = set()
        for pair in positive_pairs:
            i, j = pair[0].item(), pair[1].item()
            pos_set.add((i, j))
            pos_set.add((j, i))
        
        # self loop
        for i in range(batch_size):
            pos_set.add((i, i))
        
        # random sample negative pairs
        neg_pairs = []
        attempts = 0
        max_attempts = max_samples * 3  # avoid infinite loop
        
        while len(neg_pairs) < max_samples and attempts < max_attempts:
            i = np.random.randint(0, batch_size)
            j = np.random.randint(0, batch_size)
            if (i, j) not in pos_set:
                neg_pairs.append([i, j])
            attempts += 1
        
        if len(neg_pairs) == 0:
            return torch.empty(0, 2, dtype=torch.long, device=positive_pairs.device)
        
        return torch.tensor(neg_pairs, dtype=torch.long, device=positive_pairs.device)

    def update_contrastive_pairs_cache(self, features_omics1, features_omics2, spatial_coords, current_epoch):
        
        """
        Updates and caches the positive and negative sample pairs for contrastive learning.

        Args:
            features_omics1: Tensor of shape (N, D1), feature matrix for omics1 (e.g., RNA).
            features_omics2: Tensor of shape (N, D2), feature matrix for omics2 (e.g., protein/ADT).
            spatial_coords: Tensor of shape (N, C), spatial coordinates for each sample.
            current_epoch: Current training epoch (int), used to determine when to update the cache.

        Returns:
            None. Updates internal cached variables for spatial pairs, feature pairs, and negative pairs.
        """
        
        if (current_epoch % self.contrastive_update_freq == 0 or 
            self.last_cache_epoch == -1):
            
            # pre-calculated negtive samples
            if self.cached_spatial_pairs is None:
                self.cached_spatial_pairs = self.precompute_spatial_pairs(spatial_coords)
            
            # feature positive samples
            self.cached_feature_pairs_omics1 = self.compute_feature_pairs_gpu(features_omics1)
            self.cached_feature_pairs_omics2 = self.compute_feature_pairs_gpu(features_omics2)
            
            # combined postive samples
            if len(self.cached_spatial_pairs) > 0 and len(self.cached_feature_pairs_omics1) > 0:
                combined_pos_omics1 = torch.unique(torch.cat([self.cached_spatial_pairs, self.cached_feature_pairs_omics1], dim=0), dim=0)
            elif len(self.cached_spatial_pairs) > 0:
                combined_pos_omics1 = self.cached_spatial_pairs
            else:
                combined_pos_omics1 = self.cached_feature_pairs_omics1
                
            if len(self.cached_spatial_pairs) > 0 and len(self.cached_feature_pairs_omics2) > 0:
                combined_pos_omics2 = torch.unique(torch.cat([self.cached_spatial_pairs, self.cached_feature_pairs_omics2], dim=0), dim=0)
            elif len(self.cached_spatial_pairs) > 0:
                combined_pos_omics2 = self.cached_spatial_pairs
            else:
                combined_pos_omics2 = self.cached_feature_pairs_omics2
            
            # negative samples
            self.cached_negative_pairs_omics1 = self.sample_negative_pairs(combined_pos_omics1, features_omics1.size(0))
            self.cached_negative_pairs_omics2 = self.sample_negative_pairs(combined_pos_omics2, features_omics2.size(0))
            
            self.last_cache_epoch = current_epoch
    

    @staticmethod
    def approx_quantile_spatial_threshold_chunked(
        spatial_coords: torch.Tensor,
        q: float = 0.75,
        chunk_size: int = 2048,
        bins: int = 2048,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        通过分块 + 直方图CDF近似计算 pairwise spatial distances 的分位数阈值。
        不构造 N×N 的完整距离矩阵，不对 N² 元素调用 torch.quantile。

        返回：标量 tensor (device 与 spatial_coords 一致)
        """
        device = spatial_coords.device
        dtype = spatial_coords.dtype

        n = spatial_coords.size(0)
        if n <= 1:
            return torch.zeros((), device=device, dtype=dtype)

        # 先做一次粗略的 min/max（只统计上三角 i<j，避免自环）
        d_min = None
        d_max = None

        for i0 in range(0, n, chunk_size):
            xi = spatial_coords[i0:i0 + chunk_size]  # (ci, C)
            for j0 in range(i0, n, chunk_size):
                xj = spatial_coords[j0:j0 + chunk_size]  # (cj, C)

                # torch.cdist: (ci, cj)
                d = torch.cdist(xi, xj)

                # 去掉对角块内的对角线（自环距离=0）
                if i0 == j0:
                    diag_len = min(d.size(0), d.size(1))
                    idx = torch.arange(diag_len, device=device)
                    d[idx, idx] = float('nan')

                # 只取有效值
                d_valid = d[torch.isfinite(d)]
                if d_valid.numel() == 0:
                    continue

                cur_min = d_valid.min()
                cur_max = d_valid.max()
                d_min = cur_min if d_min is None else torch.minimum(d_min, cur_min)
                d_max = cur_max if d_max is None else torch.maximum(d_max, cur_max)

        if d_min is None or d_max is None:
            return torch.zeros((), device=device, dtype=dtype)

        # 避免所有距离相同导致 bins=0 宽度
        d_min = d_min.to(dtype)
        d_max = d_max.to(dtype)
        if (d_max - d_min).abs() < eps:
            return d_max

        # 第二遍：累积直方图
        hist = torch.zeros(bins, device=device, dtype=torch.float64)
        total = 0

        for i0 in range(0, n, chunk_size):
            xi = spatial_coords[i0:i0 + chunk_size]
            for j0 in range(i0, n, chunk_size):
                xj = spatial_coords[j0:j0 + chunk_size]
                d = torch.cdist(xi, xj)

                if i0 == j0:
                    diag_len = min(d.size(0), d.size(1))
                    idx = torch.arange(diag_len, device=device)
                    d[idx, idx] = float('nan')

                d_valid = d[torch.isfinite(d)]
                if d_valid.numel() == 0:
                    continue

                # 归一化到 [0, bins-1] 的 bin index
                # idx = floor((d - min) / (max-min) * bins)
                t = (d_valid - d_min) / (d_max - d_min)
                bin_idx = torch.clamp((t * bins).long(), 0, bins - 1)

                hist.scatter_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.float64))
                total += d_valid.numel()

        if total == 0:
            return torch.zeros((), device=device, dtype=dtype)

        # CDF -> 分位数所在 bin
        cdf = torch.cumsum(hist, dim=0)
        target = q * total
        k = int(torch.searchsorted(cdf, torch.tensor(target, device=device, dtype=cdf.dtype)).clamp(0, bins - 1).item())

        # 将 bin 中心映射回距离值（近似）
        # center = (k + 0.5) / bins
        center = (k + 0.5) / bins
        thresh = d_min + (d_max - d_min) * torch.tensor(center, device=device, dtype=dtype)
        return thresh

    def build_hard_negative_pairs_efficient(self, embeddings, spatial_coords, current_epoch, total_epochs):
        
        """
        Efficiently constructs hard negative sample pairs for contrastive learning.

        Args:
            embeddings: Tensor of shape (N, D), latent representations of samples.
            spatial_coords: Tensor of shape (N, C), spatial coordinates for each sample.
            current_epoch: Current training epoch (int), used to control when hard negatives are used.
            total_epochs: Total number of training epochs (int), used to control when hard negatives are used.

        Returns:
            hard_indices: Tensor of shape (M, 2), each row contains indices (i, j) of a hard negative sample pair.
                         If not enough pairs are found or not in the right epoch, returns an empty tensor.
        """

        if current_epoch < total_epochs * 0.5:  # not use on early epochs
            return torch.empty(0, 2, dtype=torch.long, device=embeddings.device)
        
        if spatial_coords is None:
            return torch.empty(0, 2, dtype=torch.long, device=embeddings.device)
        
        # restrict calculate density
        if current_epoch % (self.contrastive_update_freq * 2) != 0:
            return torch.empty(0, 2, dtype=torch.long, device=embeddings.device)
        
        batch_size = embeddings.size(0)
        max_hard_samples = min(100, batch_size)  # restrict hard negtive samples
        
        # calculate embedding similarities
        if embeddings.size(0) < 5000:  # avoid large memory usage
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            emb_similarity = torch.mm(embeddings_norm, embeddings_norm.t())
            
            # calculate spatial distances
            spatial_coords_expanded_i = spatial_coords.unsqueeze(1).expand(-1, batch_size, -1)
            spatial_coords_expanded_j = spatial_coords.unsqueeze(0).expand(batch_size, -1, -1)
            spatial_distances = torch.norm(spatial_coords_expanded_i - spatial_coords_expanded_j, dim=2)
            
            # spatial quantile threshold
            spatial_threshold = torch.quantile(spatial_distances.flatten(), 0.75)
            # spatial_threshold = self.safe_quantile(spatial_distances, 0.75)
            
            # find embeddings similar in embedding dimension but with long distances
            similarity_mask = emb_similarity > 0.5  # similar embeddings
            distance_mask = spatial_distances > spatial_threshold  # long distances
            hard_mask = similarity_mask & distance_mask
            
            # remove diagonals
            hard_mask.fill_diagonal_(False)
            
            # get indices of hard negative pairs
            hard_indices = torch.nonzero(hard_mask, as_tuple=False)
            
            # Random sampling
            if len(hard_indices) > max_hard_samples:
                perm = torch.randperm(len(hard_indices))[:max_hard_samples]
                hard_indices = hard_indices[perm]
            
            return hard_indices
        else:
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            emb_similarity = torch.mm(embeddings_norm, embeddings_norm.t())

            # 计算 spatial_threshold：改为分块近似分位数，避免 quantile 输入太大
            spatial_threshold = self.approx_quantile_spatial_threshold_chunked(
                spatial_coords,
                q=0.75,
                chunk_size=2048,  # 可根据显存/内存调小，例如 512/1024
                bins=2048,
            )

            # 下面 distance_mask 仍需要 spatial_distances
            # 为了避免构造 N×N×C，这里也用 torch.cdist 直接得到 N×N 距离矩阵（仍是 N×N）
            # 如果 N 很大，建议进一步把 hard negative 构造也改成基于采样/邻域候选，避免 N×N。
            spatial_distances = torch.cdist(spatial_coords, spatial_coords)

            # find embeddings similar in embedding dimension but with long distances
            similarity_mask = emb_similarity > 0.5  # similar embeddings
            distance_mask = spatial_distances > spatial_threshold  # long distances
            hard_mask = similarity_mask & distance_mask

            # remove diagonals
            hard_mask.fill_diagonal_(False)

            hard_indices = torch.nonzero(hard_mask, as_tuple=False)

            # Random sampling
            batch_size = embeddings.size(0)
            max_hard_samples = min(100, batch_size)
            if len(hard_indices) > max_hard_samples:
                perm = torch.randperm(len(hard_indices), device=hard_indices.device)[:max_hard_samples]
                hard_indices = hard_indices[perm]

            return hard_indices

    def contrastive_loss_efficient(self, embeddings, pairs_dict, current_epoch, total_epochs):
        
        """
        Efficient computation of contrastive loss for a batch of embeddings using provided sample pairs.

        Args:
            embeddings: Tensor of shape (N, D), where N is the number of samples and D is the embedding dimension.
            pairs_dict: Dictionary containing different types of sample pairs (e.g., positive, negative, hard negative), 
                        each as a tensor of shape (M, 2) with indices of sample pairs.
            current_epoch: Current training epoch (int), used for dynamic weighting of loss components.
            total_epochs: Total number of training epochs (int), used for dynamic weighting of loss components.

        Returns:
            loss: Scalar tensor representing the average contrastive loss over all provided pairs, 
                  with dynamic weighting for easy and hard samples.
        """

        if not self.use_contrastive:
            return torch.tensor(0.0, device=embeddings.device)
        
        total_loss = 0.0
        loss_count = 0
        
        # Simplified dynamic weights
        epoch_ratio = current_epoch / total_epochs
        easy_weight = max(0.3, 1.0 - epoch_ratio)
        hard_weight = min(1.0, epoch_ratio * 2)
        
        # Only compute main positive and negative pairs to reduce computation
        for pair_type, pairs in pairs_dict.items():
            if len(pairs) == 0:
                continue
                
            # Limit the number of pairs per type
            max_pairs = 200
            if len(pairs) > max_pairs:
                indices = torch.randperm(len(pairs))[:max_pairs]
                pairs = pairs[indices]
            
            if 'pos' in pair_type:
                loss = self._compute_pair_loss_efficient(embeddings, pairs, is_positive=True)
                total_loss += loss * easy_weight
                loss_count += 1
            elif 'neg' in pair_type:
                weight = hard_weight if 'hard' in pair_type else easy_weight
                loss = self._compute_pair_loss_efficient(embeddings, pairs, is_positive=False)
                total_loss += loss * weight
                loss_count += 1
        
        return total_loss / max(loss_count, 1)  # average loss

    def _compute_pair_loss_efficient(self, embeddings, pairs, is_positive=True):
        
        """
        Efficient computation of pairwise loss for contrastive learning.
        
        Args:
            embeddings: Tensor of shape (N, D) where N is the number of samples and D is the embedding dimension
            pairs:  Tensor of shape (M, 2), each row contains indices (i, j) representing a sample pair.
            is_positive: Boolean indicating whether the pairs are positive or negative samples.

        Returns: Scalar tensor representing the average contrastive loss over the provided pairs.
        """

        if len(pairs) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Batch fetch embeddings
        emb_i = embeddings[pairs[:, 0]]
        emb_j = embeddings[pairs[:, 1]]
        
        # Use more efficient similarity calculation
        similarity = F.cosine_similarity(emb_i, emb_j, dim=1) / self.contrastive_temp
        
        if is_positive:
            # Positive samples: use more stable loss function
            loss = -F.logsigmoid(similarity).mean()
        else:
            # Negative samples: use more stable loss function
            loss = -F.logsigmoid(-similarity).mean()
        
        return loss
    
    def safe_quantile(tensor, q, max_elements=1e7):

        """
        Safe quantile calculation to avoid issues with large tensors
        
        Args:
            tensor: Input tensor, can be any shape
            q: Quantile value (float), between 0 and 1, e.g., 0.75 for the 75th percentile
            max_elements: Maximum number of elements allowed for quantile calculation; if exceeded, random sampling is used
        
        Returns:
            quantile_value: The computed quantile value (float); for large tensors, computed on a random subset for efficiency and memory safety
        """

        flattened = tensor.flatten()
        
        if flattened.numel() > max_elements:
            # random sampling
            sample_size = int(max_elements)
            indices = torch.randperm(flattened.numel(), device=flattened.device)[:sample_size]
            sampled_tensor = flattened[indices]
            return torch.quantile(sampled_tensor, q)
        else:
            return torch.quantile(flattened, q)

    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1,
                adj_spatial_omics2, adj_feature_omics2, spatial_coords=None, current_epoch=0, total_epochs=1000,
                clean_features_omics1=None, clean_features_omics2=None):
        if clean_features_omics1 is None:
            clean_features_omics1 = features_omics1
        if clean_features_omics2 is None:
            clean_features_omics2 = features_omics2

        # four encoders
        emb_latent_spatial_omics1_raw = self.encoder_omics1_spatial(features_omics1, adj_spatial_omics1)
        emb_latent_feature_omics1_raw = self.encoder_omics1_feature(features_omics1, adj_feature_omics1)
        emb_latent_spatial_omics2 = self.encoder_omics2_spatial(features_omics2, adj_spatial_omics2)
        emb_latent_feature_omics2 = self.encoder_omics2_feature(features_omics2, adj_feature_omics2)

        emb_latent_spatial_omics1 = emb_latent_spatial_omics1_raw
        emb_latent_feature_omics1 = emb_latent_feature_omics1_raw
        emb_latent_spatial_omics1_mu = None
        emb_latent_spatial_omics1_logvar = None
        emb_latent_feature_omics1_mu = None
        emb_latent_feature_omics1_logvar = None

        # attention fusion
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)
        # directly concatenate spatial and feature embeddings as an alternative to attention fusion
        # emb_latent_omics1 = torch.cat([emb_latent_spatial_omics1, emb_latent_feature_omics1], dim=1)
        # emb_latent_omics2 = torch.cat([emb_latent_spatial_omics2, emb_latent_feature_omics2], dim=1)

        # cross-modal attention fusion
        emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_latent_omics1, emb_latent_omics2)

        # use four decoders for reconstruction
        emb_recon_spatial_omics1 = self.decoder_omics1_spatial(emb_latent_combined, adj_spatial_omics1)
        emb_recon_feature_omics1 = self.decoder_omics1_feature(emb_latent_combined, adj_feature_omics1)
        emb_recon_spatial_omics2 = self.decoder_omics2_spatial(emb_latent_combined, adj_spatial_omics2)
        emb_recon_feature_omics2 = self.decoder_omics2_feature(emb_latent_combined, adj_feature_omics2)

        ############################################################################################
        # cross-modal reconstruction
        emb_latent_omics1_across_recon = self.encoder_omics2_spatial(
            self.decoder_omics2_spatial(emb_latent_omics1, adj_spatial_omics2), adj_spatial_omics2)
        emb_latent_omics2_to_omics1_raw = self.encoder_omics1_spatial(
            self.decoder_omics1_spatial(emb_latent_omics2, adj_spatial_omics1), adj_spatial_omics1)
        emb_latent_omics2_across_recon = emb_latent_omics2_to_omics1_raw
        ############################################################################################

        denoising_loss_omics1 = 0.5 * (
            F.mse_loss(emb_recon_spatial_omics1, clean_features_omics1) +
            F.mse_loss(emb_recon_feature_omics1, clean_features_omics1)
        )
        denoising_loss_omics2 = 0.5 * (
            F.mse_loss(emb_recon_spatial_omics2, clean_features_omics2) +
            F.mse_loss(emb_recon_feature_omics2, clean_features_omics2)
        )

        contrastive_loss_omics1 = torch.tensor(0.0, device=features_omics1.device)
        contrastive_loss_omics2 = torch.tensor(0.0, device=features_omics2.device)

        # contrastive learning
        if self.use_contrastive:
            # update cache
            self.update_contrastive_pairs_cache(features_omics1, features_omics2, spatial_coords, current_epoch)

            # construct efficient hard negative pairs
            hard_neg_pairs_omics1 = self.build_hard_negative_pairs_efficient(
                emb_latent_omics1, spatial_coords, current_epoch, total_epochs)
            hard_neg_pairs_omics2 = self.build_hard_negative_pairs_efficient(
                emb_latent_omics2, spatial_coords, current_epoch, total_epochs)

            # construct sample dict
            contrastive_pairs_omics1 = {
                'spatial_pos': self.cached_spatial_pairs if self.cached_spatial_pairs is not None else torch.empty(0, 2, dtype=torch.long, device=features_omics1.device),
                'feature_pos_omics1': self.cached_feature_pairs_omics1 if self.cached_feature_pairs_omics1 is not None else torch.empty(0, 2, dtype=torch.long, device=features_omics1.device),
                'neg_omics1': self.cached_negative_pairs_omics1 if self.cached_negative_pairs_omics1 is not None else torch.empty(0, 2, dtype=torch.long, device=features_omics1.device),
                'hard_neg_omics1': hard_neg_pairs_omics1
            }

            contrastive_pairs_omics2 = {
                'spatial_pos': self.cached_spatial_pairs if self.cached_spatial_pairs is not None else torch.empty(0, 2, dtype=torch.long, device=features_omics2.device),
                'feature_pos_omics2': self.cached_feature_pairs_omics2 if self.cached_feature_pairs_omics2 is not None else torch.empty(0, 2, dtype=torch.long, device=features_omics2.device),
                'neg_omics2': self.cached_negative_pairs_omics2 if self.cached_negative_pairs_omics2 is not None else torch.empty(0, 2, dtype=torch.long, device=features_omics2.device),
                'hard_neg_omics2': hard_neg_pairs_omics2
            }

            # calculate contrastive loss
            contrastive_loss_omics1 = self.contrastive_loss_efficient(
                emb_latent_omics1, contrastive_pairs_omics1, current_epoch, total_epochs)
            contrastive_loss_omics2 = self.contrastive_loss_efficient(
                emb_latent_omics2, contrastive_pairs_omics2, current_epoch, total_epochs)

        results = {
            'emb_latent_omics1': emb_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined,
            'emb_latent_spatial_omics1_mu': emb_latent_spatial_omics1_mu,
            'emb_latent_spatial_omics1_logvar': emb_latent_spatial_omics1_logvar,
            'emb_latent_feature_omics1_mu': emb_latent_feature_omics1_mu,
            'emb_latent_feature_omics1_logvar': emb_latent_feature_omics1_logvar,
            'emb_recon_spatial_omics1': emb_recon_spatial_omics1,
            'emb_recon_feature_omics1': emb_recon_feature_omics1,
            'emb_recon_spatial_omics2': emb_recon_spatial_omics2,
            'emb_recon_feature_omics2': emb_recon_feature_omics2,
            'denoised_omics1': emb_recon_spatial_omics1,
            'denoised_omics2': emb_recon_spatial_omics2,
            'denoising_loss_omics1': denoising_loss_omics1,
            'denoising_loss_omics2': denoising_loss_omics2,
            'emb_latent_omics1_across_recon': emb_latent_omics1_across_recon,
            'emb_latent_omics2_across_recon': emb_latent_omics2_across_recon,
            'alpha_omics1': alpha_omics1,
            'alpha_omics2': alpha_omics2,
            'alpha': alpha_omics_1_2,
            'contrastive_loss_omics1': contrastive_loss_omics1,
            'contrastive_loss_omics2': contrastive_loss_omics2
        }

        return results


class Encoder(Module): 

    """
    Two-Layer GCN Encoder

    Args:
        in_feat (int): Input feature dimension
        out_feat (int): Output feature dimension
        dropout (float): Dropout rate
        act (function): Activation function

    Returns:
        x (Tensor): Encoded features
    """

    def __init__(self, in_feat, out_feat, hidden_feat=None, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        if hidden_feat is None:
            self.hidden_feat = (in_feat + out_feat) // 2
        else:
            self.hidden_feat = hidden_feat

        self.weight1 = Parameter(torch.FloatTensor(self.in_feat, self.hidden_feat))
        self.weight2 = Parameter(torch.FloatTensor(self.hidden_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
                                                        
    def forward(self, feat, adj):                       
        x = torch.mm(feat, self.weight1)
        x = torch.spmm(adj, x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = torch.mm(x, self.weight2)
        x = torch.spmm(adj, x)
        
        return x
class Decoder(Module):

    """
    Two-Layer GCN Decoder

    Args:
        in_feat (int): Input feature dimension
        out_feat (int): Output feature dimension
        hidden_feat (int): Hidden feature dimension
        dropout (float): Dropout rate
        act (function): Activation function
    
    Returns:
        x (Tensor): Reconstructed features
    """

    def __init__(self, in_feat, out_feat, hidden_feat=None, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        if hidden_feat is None:
            self.hidden_feat = (in_feat + out_feat) // 2
        else:
            self.hidden_feat = hidden_feat

        self.weight1 = Parameter(torch.FloatTensor(self.in_feat, self.hidden_feat))
        self.weight2 = Parameter(torch.FloatTensor(self.hidden_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        
    def forward(self, feat, adj):       
        x = torch.mm(feat, self.weight1)
        x = torch.spmm(adj, x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = torch.mm(x, self.weight2)
        x = torch.spmm(adj, x)
        
        return x


class AttentionLayer(Module):
    """
    Attention Layer for combining two embeddings

    Args:
        in_feat (int): Input feature dimension
        out_feat (int): Output feature dimension
        dropout (float): Dropout rate
        act (function): Activation function
    
    Returns:
        emb_combined (Tensor): Combined embedding
        alpha (Tensor): Attention weights
    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        # Keep batch dimension even when batch_size == 1.
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)

        # (N, 2, D)
        self.emb = torch.stack([emb1, emb2], dim=1)

        # (N, 2, out_feat) -> (N, 2, 1)
        self.v = torch.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)

        # Softmax over the modality dimension (2).
        self.alpha = F.softmax(self.vu.squeeze(-1) + 1e-6, dim=1)

        # (N, D, 1)
        emb_combined = torch.bmm(self.emb.transpose(1, 2), self.alpha.unsqueeze(-1))
        return emb_combined.squeeze(-1), self.alpha
