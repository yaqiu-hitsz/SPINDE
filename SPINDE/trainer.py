"""
# Author: Yaqiu Wang
# File Name: trainer.py
# Description: Trainer for SPINDE model
"""

import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import Network, add_noise_to_features
from .preprocess import adjacent_matrix_preprocessing
import numpy as np
import scipy.sparse as sp

class SPINDE_Trainer:

    @staticmethod
    def _obsm_to_tensor(adata, key, device, dtype=torch.float32):
        if key not in adata.obsm_keys():
            raise KeyError(f"Expected adata.obsm['{key}'] to exist.")

        matrix = adata.obsm[key]
        if sp.issparse(matrix):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float32)
        return torch.as_tensor(matrix, dtype=dtype, device=device)

    def __init__(self, 
        data,
        device = torch.device('cpu'),
        random_seed = 2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=1500, 
        dim_input=3000,
        dim_output=64,
        weight_factors = [1, 5, 1, 10],
        spatial_bias = 0.5,
        use_contrastive=True,
        contrastive_weight=1.0,
        spatial_reg_weight=0.0,                 # 新增：空间正则项权重（默认关闭）
        spatial_reg_acceleration=True,          # 新增：是否用采样加速
        spatial_reg_edge_subset_sz=200000,      # 新增：采样 pair 数量（按显存/速度调）
        spatial_reg_use_combined=True,          # 新增：用 combined embedding 做正则（否则用 omics1/2 均值）
        denoising_weight=1.0,                   # 新增：低维特征去噪重建损失权重
        noise_std=0.1,                          # 新增：高斯/扩散式噪声强度
        noise_type="gaussian",                  # 新增："gaussian" or "diffusion"
        grad_clip_norm=None,                     # 新增：可选梯度裁剪（例如 5.0）
        fail_fast_on_nan=True                    # 新增：遇到 NaN/Inf 立即报错定位
        ):
        
        """Initialize SPINDE trainer"""

        
        self.data = data.copy()
        self.device = device
        self.random_seed = random_seed
        self.epochs = epochs
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.spatial_bias = spatial_bias
        self.spatial_reg_weight = float(spatial_reg_weight)
        self.spatial_reg_acceleration = bool(spatial_reg_acceleration)
        self.spatial_reg_edge_subset_sz = int(spatial_reg_edge_subset_sz)
        self.spatial_reg_use_combined = bool(spatial_reg_use_combined)
        self.denoising_weight = float(denoising_weight)
        self.noise_std = float(noise_std)
        self.noise_type = str(noise_type).lower().strip()
        self.grad_clip_norm = grad_clip_norm
        self.fail_fast_on_nan = bool(fail_fast_on_nan)

        if self.spatial_reg_weight < 0.0:
            raise ValueError(f"spatial_reg_weight must be non-negative, got {self.spatial_reg_weight}")
        if self.denoising_weight < 0.0:
            raise ValueError(f"denoising_weight must be non-negative, got {self.denoising_weight}")
        if self.noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")
        if self.noise_type not in {"gaussian", "diffusion", "diffusion-style"}:
            raise ValueError(f"noise_type must be 'gaussian' or 'diffusion', got {noise_type}")
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        
        # feature matrices from obsm
        self.features_omics1 = self._obsm_to_tensor(self.adata_omics1, 'feat', self.device)
        self.features_omics2 = self._obsm_to_tensor(self.adata_omics2, 'feat', self.device)
        if not torch.isfinite(self.features_omics1).all():
            raise ValueError("adata_omics1.obsm['feat'] contains NaN/Inf.")
        if not torch.isfinite(self.features_omics2).all():
            raise ValueError("adata_omics2.obsm['feat'] contains NaN/Inf.")
        
        # get spatial coords
        if 'spatial' in self.adata_omics1.obsm.keys():
            self.spatial_coords = torch.FloatTensor(self.adata_omics1.obsm['spatial'].copy()).to(self.device)
        elif 'X_spatial' in self.adata_omics1.obsm.keys():
            self.spatial_coords = torch.FloatTensor(self.adata_omics1.obsm['X_spatial'].copy()).to(self.device)
        else:
            # if spatial coords not found in obsm, try to find in obs
            try:
                spatial_cols = [col for col in self.adata_omics1.obs.columns if 'x' in col.lower() or 'y' in col.lower()]
                if len(spatial_cols) >= 2:
                    coords = self.adata_omics1.obs[spatial_cols[:2]].values
                    self.spatial_coords = torch.FloatTensor(coords).to(self.device)
                else:
                    self.spatial_coords = None
                    print("Warning: No spatial coordinates found. Contrastive learning will not use spatial information.")
            except:
                self.spatial_coords = None
                print("Warning: No spatial coordinates found. Contrastive learning will not use spatial information.")
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        
        # set random seed
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
    
    def _spatial_regularization_loss(self, z: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        spatial penalty computation:
        penalty = mean( (1 - norm_dist(z_i,z_j)) * norm_dist(coord_i,coord_j) )

        avoid O(N^2).
        """
        if self.spatial_reg_weight <= 0:
            return torch.tensor(0.0, device=z.device)

        if coords is None or z is None or z.size(0) <= 1:
            return torch.tensor(0.0, device=z.device)

        n = z.size(0)
        pdist = torch.nn.PairwiseDistance(p=2)

        # 加速：采样 pair
        if self.spatial_reg_acceleration or n > 5000:
            m = min(self.spatial_reg_edge_subset_sz, n * n)
            idx1 = torch.randint(0, n, (m,), device=z.device)
            idx2 = torch.randint(0, n, (m,), device=z.device)

            z1 = z.index_select(0, idx1)
            z2 = z.index_select(0, idx2)
            c1 = coords.index_select(0, idx1)
            c2 = coords.index_select(0, idx2)

            z_dists = pdist(z1, z2)
            sp_dists = pdist(c1, c2)

            # 归一化，避免尺度影响；加 eps 防止除 0
            z_dists = z_dists / (torch.max(z_dists) + 1e-12)
            sp_dists = sp_dists / (torch.max(sp_dists) + 1e-12)

            penalty = torch.mean((1.0 - z_dists) * sp_dists)
            return penalty

        # 不加速：全量 N×N（N小才建议）
        z_dists = torch.cdist(z, z, p=2)
        sp_dists = torch.cdist(coords, coords, p=2)

        z_dists = z_dists / (torch.max(z_dists) + 1e-12)
        sp_dists = sp_dists / (torch.max(sp_dists) + 1e-12)

        penalty = torch.mean((1.0 - z_dists) * sp_dists)
        return penalty

    def train(self):

        """train SPINDE model"""

        self.model = Network(
            self.dim_input1, self.dim_output1, 
            self.dim_input2, self.dim_output2, 
            use_contrastive=self.use_contrastive,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.model.train()
        
        # record loss
        if self.spatial_reg_weight > 0.0:
            loss_history = {
                'total_loss': [],
                'recon_loss': [],
                'denoising_loss': [],
                'corr_loss': [],
                'contrastive_loss': [],
                'spatial_reg_loss': []
            }
        else:
            loss_history = {
                'total_loss': [],
                'recon_loss': [],
                'denoising_loss': [],
                'corr_loss': [],
                'contrastive_loss': [],
            }
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            noisy_features_omics1 = add_noise_to_features(
                self.features_omics1,
                noise_std=self.noise_std,
                noise_type=self.noise_type,
            )
            noisy_features_omics2 = add_noise_to_features(
                self.features_omics2,
                noise_std=self.noise_std,
                noise_type=self.noise_type,
            )

            results = self.model(
                noisy_features_omics1, noisy_features_omics2, 
                self.adj_spatial_omics1, self.adj_feature_omics1, 
                self.adj_spatial_omics2, self.adj_feature_omics2,
                spatial_coords=self.spatial_coords,
                current_epoch=epoch,
                total_epochs=self.epochs,
                clean_features_omics1=self.features_omics1,
                clean_features_omics2=self.features_omics2,
            )
            
            # Denoising reconstruction loss: noisy PCA/LSI features -> clean PCA/LSI features.
            recon_spatial_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_spatial_omics1'])
            recon_feature_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_feature_omics1'])
            self.loss_recon_omics1 = self.spatial_bias * recon_spatial_omics1 + (1 - self.spatial_bias) * recon_feature_omics1

            recon_spatial_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_spatial_omics2'])
            recon_feature_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_feature_omics2'])
            self.loss_recon_omics2 = self.spatial_bias * recon_spatial_omics2 + (1 - self.spatial_bias) * recon_feature_omics2
            
            # corr loss
            self.loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], 
                                              results['emb_latent_omics1_across_recon'])
            self.loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], 
                                              results['emb_latent_omics2_across_recon'])
            
            # contrastice learning loss
            if self.use_contrastive:
                contrastive_loss = results['contrastive_loss_omics1'] + results['contrastive_loss_omics2']
            else:
                contrastive_loss = torch.tensor(0.0, device=self.device)
            
            # total_loss
            reconstruction_loss = self.denoising_weight * (
                self.weight_factors[0]*self.loss_recon_omics1 + self.weight_factors[1]*self.loss_recon_omics2
            )
            correspondence_loss = self.weight_factors[2]*self.loss_corr_omics1 + self.weight_factors[3]*self.loss_corr_omics2
            
            # spatial regularization loss
            if self.spatial_reg_use_combined:
                z_for_reg = results['emb_latent_combined']
            else:
                z_for_reg = 0.5 * (results['emb_latent_omics1'] + results['emb_latent_omics2'])
            
            spatial_reg_loss = self._spatial_regularization_loss(z_for_reg, self.spatial_coords)

            loss = reconstruction_loss + correspondence_loss + self.contrastive_weight * contrastive_loss + self.spatial_reg_weight * spatial_reg_loss

            if self.fail_fast_on_nan and (not torch.isfinite(loss)):
                parts = {
                    'reconstruction_loss': reconstruction_loss,
                    'correspondence_loss': correspondence_loss,
                    'contrastive_loss': contrastive_loss,
                    'spatial_reg_loss': spatial_reg_loss,
                    'loss_recon_omics1': self.loss_recon_omics1,
                    'loss_recon_omics2': self.loss_recon_omics2,
                    'loss_corr_omics1': self.loss_corr_omics1,
                    'loss_corr_omics2': self.loss_corr_omics2,
                }
                bad = {k: float(v.detach().cpu().item()) for k, v in parts.items() if not torch.isfinite(v)}
                raise FloatingPointError(
                    f"Non-finite loss detected at epoch {epoch + 1}. Non-finite parts: {bad}. "
                    "This is usually caused by invalid raw counts in obsm['raw'] or exploding parameters (try lowering lr / enabling grad_clip_norm)."
                )
            
            # record loss
            loss_history['total_loss'].append(loss.item())
            loss_history['recon_loss'].append((reconstruction_loss).item())
            loss_history['denoising_loss'].append((reconstruction_loss).item())
            loss_history['corr_loss'].append((correspondence_loss).item())
            loss_history['contrastive_loss'].append(contrastive_loss.item())
            if self.spatial_reg_weight > 0.0:
                loss_history['spatial_reg_loss'].append(spatial_reg_loss.item())
            
            self.optimizer.zero_grad()
            loss.backward() 

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))

            self.optimizer.step()
            
            # print loss info 
            if (epoch + 1) % 100 == 0 or epoch == 0:
                if self.spatial_reg_weight >= 0.0:
                    print(f"Epoch {epoch+1}/{self.epochs}: "
                          f"Total Loss: {loss.item():.4f}, "
                          f"Denoising Loss: {reconstruction_loss.item():.4f}, "
                          f"Corr Loss: {correspondence_loss.item():.4f}, "
                          f"Contrastive Loss: {contrastive_loss.item():.4f}, "
                          f"Spatial Reg Loss: {spatial_reg_loss.item():.4f}")
                else:
                    print(f"Epoch {epoch+1}/{self.epochs}: "
                          f"Total Loss: {loss.item():.4f}, "
                          f"Denoising Loss: {reconstruction_loss.item():.4f}, "
                          f"Corr Loss: {correspondence_loss.item():.4f}, "
                          f"Contrastive Loss: {contrastive_loss.item():.4f}")
        
        print("Model training finished!\n")    
        
        # record history
        self.loss_history = loss_history
    
        # eval mode
        with torch.no_grad():
            self.model.eval()
            results = self.model(
                self.features_omics1, self.features_omics2, 
                self.adj_spatial_omics1, self.adj_feature_omics1, 
                self.adj_spatial_omics2, self.adj_feature_omics2,
                spatial_coords=self.spatial_coords,
                current_epoch=self.epochs-1,
                total_epochs=self.epochs,
                clean_features_omics1=self.features_omics1,
                clean_features_omics2=self.features_omics2,
            )
 
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        output = {
            'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),           #Y1
            'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),           #Y2
            'SPINDE': emb_combined.detach().cpu().numpy(),                #Z
            'emb_recon_omics1': results['emb_recon_spatial_omics1'].detach().cpu().numpy(),   #X1重构
            'emb_recon_omics2': results['emb_recon_spatial_omics2'].detach().cpu().numpy(),   #X2重构
            'denoised_omics1': results['denoised_omics1'].detach().cpu().numpy(),
            'denoised_omics2': results['denoised_omics2'].detach().cpu().numpy(),
            'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),   #模态1内注意力分数
            'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),   #模态2内注意力分数
            'alpha': results['alpha'].detach().cpu().numpy(),                 #模态间注意力分数
            'loss_history': self.loss_history,                               #训练损失历史
            'contrastive_pairs': results.get('contrastive_pairs', None)       #对比学习样本对信息
        }
        
        return output
    
    def get_loss_history(self):
        """get loss history"""
        if hasattr(self, 'loss_history'):
            return self.loss_history
        else:
            print("Model has not been trained yet.")
            return None
    
    def plot_loss_history(self):
        """plot loss history"""
        if not hasattr(self, 'loss_history'):
            print("Model has not been trained yet.")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 总损失
            axes[0, 0].plot(self.loss_history['total_loss'])
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            
            # 重构损失
            axes[0, 1].plot(self.loss_history['recon_loss'])
            axes[0, 1].set_title('Denoising Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            
            # 对应关系损失
            axes[1, 0].plot(self.loss_history['corr_loss'])
            axes[1, 0].set_title('Correspondence Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            
            # 对比学习损失
            axes[1, 1].plot(self.loss_history['contrastive_loss'])
            axes[1, 1].set_title('Contrastive Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available. Cannot plot loss history.")





