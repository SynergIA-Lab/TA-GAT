import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class TopologyAwareGATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super().__init__()
        # GATv2 para atención dinámica (mucho mejor que GCN para redes densas)
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, edge_dim=1)
        # Las cabezas se concatenan, así que la entrada a cond_mu es hidden_channels * heads
        self.conv_mu = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, edge_dim=1)
        self.conv_logstd = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, edge_dim=1)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None:
            x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
            return self.conv_mu(x, edge_index, edge_attr=edge_attr), self.conv_logstd(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv1(x, edge_index).relu()
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class TAGAT(nn.Module):
    """
    Topology-Aware Graph Attention Autoencoder (TA-GAT)
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, *args, **kwargs):
        # Retorna el embedding determinista z (mu) + reparametrización
        mu, logstd = self.encoder(*args, **kwargs)
        self.__mu__ = mu
        # Clamping fuerte para evitar explosión de la divergencia KL en época 1
        self.__logstd__ = logstd.clamp(min=-10, max=2)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd
        # logstd <= 2 -> exp(2) = 7.3 -> exp(2)^2 = 54
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1)
        )

# --- Funciones de Pérdida (Loss) ---

def scale_free_loss_v2(adj_pred: torch.Tensor, gamma: float = 2.5) -> torch.Tensor:
    """
    Pérdida Topológica Scale-Free v2 (Diferenciable en log-log space).
    
    Estrategia: En una red scale-free, la relación log(grado) vs log(rango) es lineal
    con pendiente negativa. Optimizamos directamente el R² de esta regresión lineal
    en espacio logarítmico, más un término de heterogeneidad de grado.
    
    Componentes:
    1. Linearity Loss: (1 - R²) en espacio log-log → fuerza ajuste power-law
    2. Slope Penalty: penaliza pendiente positiva (debe ser negativa)
    3. Heterogeneity: coeficiente de variación del grado → fomenta hubs
    """
    # Grado suave (soft degree) de cada nodo
    degrees = torch.sum(adj_pred, dim=1)
    degrees = torch.clamp(degrees, min=0.1)
    
    # Ordenar descendente (hubs primero)
    sorted_deg, _ = torch.sort(degrees, descending=True)
    
    N = adj_pred.size(0)
    ranks = torch.arange(1, N + 1, device=adj_pred.device, dtype=torch.float32)
    
    # --- Regresión lineal diferenciable en log-log ---
    log_ranks = torch.log(ranks)
    log_degrees = torch.log(sorted_deg + 1e-6)
    
    mean_x = log_ranks.mean()
    mean_y = log_degrees.mean()
    
    dx = log_ranks - mean_x
    dy = log_degrees - mean_y
    
    ss_xy = (dx * dy).sum()
    ss_xx = (dx ** 2).sum()
    ss_yy = (dy ** 2).sum()
    
    # R² diferenciable
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy + 1e-9)
    linearity_loss = 1.0 - r_squared
    
    # Pendiente: debe ser negativa (grado decrece con rango)
    slope = ss_xy / (ss_xx + 1e-9)
    slope_penalty = F.relu(slope + 0.3)  # Penalizar si slope > -0.3
    
    # Heterogeneidad: coeficiente de variación alto = red con hubs
    # CV bajo → red uniforme/aleatoria → penalizar
    degree_cv = degrees.std() / (degrees.mean() + 1e-9)
    uniformity_penalty = 1.0 / (degree_cv + 0.1)
    
    return linearity_loss + 0.5 * slope_penalty + 0.05 * uniformity_penalty


def targeted_sparsity_loss(adj_pred: torch.Tensor) -> torch.Tensor:
    """
    Sparsity dirigida compatible con scale-free (normalizada por N).
    
    En lugar de penalizar la media global de adj (que aplana hubs), penalizamos:
    1. El grado mínimo → fuerza que existan nodos hoja (grado bajo)
    2. La mediana del grado → controla la densidad sin destruir hubs
    
    Normalizado por N para que la escala sea comparable a las otras losses (~0-1).
    """
    degrees = torch.sum(adj_pred, dim=1)
    N = adj_pred.size(0)
    
    # Normalizar por N para escala comparable
    norm_degrees = degrees / N
    
    # Penalizar grado mínimo alto (queremos nodos hoja)
    min_degree_penalty = torch.min(norm_degrees)
    
    # Penalizar mediana de grado alta (controla densidad general)
    sorted_deg, _ = torch.sort(norm_degrees)
    median_idx = len(sorted_deg) // 2
    median_degree_penalty = sorted_deg[median_idx]
    
    return 0.5 * min_degree_penalty + 0.5 * median_degree_penalty


def ta_gat_loss(z: torch.Tensor, 
                pos_edge_index: torch.Tensor, 
                neg_edge_index: torch.Tensor, 
                lambda_sf: float = 5.0,
                pos_edge_weights: torch.Tensor = None) -> tuple:
    """
    Pérdida combinada de TA-GAT v2:
    1. Reconstrucción BCE Ponderada (Edges iniciales dictados por correlación)
    2. Scale-Free Loss v2 (R² en log-log + heterogeneidad de grado)
    3. Targeted Sparsity (mediana + mínimo de grado, compatible con hubs)
    """
    EPS = 1e-15
    
    # --- 1. Reconstrucción (BCE Básica Vectorizada) ---
    pos_pred = torch.sigmoid(torch.sum(z[pos_edge_index[0]] * z[pos_edge_index[1]], dim=1))
    neg_pred = torch.sigmoid(torch.sum(z[neg_edge_index[0]] * z[neg_edge_index[1]], dim=1))
    
    pos_loss = -torch.log(pos_pred + EPS).mean()
    neg_loss = -torch.log(1 - neg_pred + EPS).mean()
    recon_loss = pos_loss + neg_loss
    
    # --- 2. Scale-Free Regularization v2 ---
    z_norm = F.normalize(z, p=2, dim=1)
    adj_pred_dense = torch.sigmoid(torch.matmul(z_norm, z_norm.t()))
    mask = ~torch.eye(adj_pred_dense.size(0), dtype=torch.bool, device=adj_pred_dense.device)
    adj_pred_dense = adj_pred_dense * mask
    
    sf_loss = scale_free_loss_v2(adj_pred_dense)
    sparsity_loss = targeted_sparsity_loss(adj_pred_dense)
    
    return recon_loss, sf_loss, sparsity_loss
