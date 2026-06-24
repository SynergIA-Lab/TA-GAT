import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

##################################################
# PHASE 1: GATV2 ENCODER NETWORK ARCHITECTURE   #
##################################################

class TopologyAwareGATEncoder(nn.Module):
    """
    Graph Attention Network (GATv2) encoder for gene expression and topology.

    This encoder projects input node features and edge weights into a latent 
    space, outputting the parameters (mean and log variance) of a variational 
    distribution. It utilizes GATv2 layers to capture dynamic, cell-context-specific 
    regulatory relationships.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 2):
        """
        Initializes the TopologyAwareGATEncoder.

        Args:
            in_channels (int): Dimension of the input node features.
            hidden_channels (int): Dimension of the hidden attention channels.
            out_channels (int): Dimension of the output latent space.
            heads (int, optional): Number of attention heads. Defaults to 2.
        """
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, edge_dim=1)
        self.conv_mu = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, edge_dim=1)
        self.conv_logstd = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, edge_dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the forward pass of the encoder.

        Args:
            x (torch.Tensor): Node feature matrix of shape (num_nodes, in_channels).
            edge_index (torch.Tensor): Graph edge indices of shape (2, num_edges).
            edge_attr (torch.Tensor, optional): Edge weights/attributes of shape (num_edges, 1). Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - mu (torch.Tensor): Mean vector of the latent distribution of shape (num_nodes, out_channels).
                - logstd (torch.Tensor): Log standard deviation vector of shape (num_nodes, out_channels).
        """
        if edge_attr is not None:
            x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
            return self.conv_mu(x, edge_index, edge_attr=edge_attr), self.conv_logstd(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv1(x, edge_index).relu()
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

##################################################
# PHASE 2: VARIATIONAL AUTOENCODER CONTAINER    #
##################################################

class TAGAT(nn.Module):
    """
    Topology-Aware Graph Attention Autoencoder (TA-GAT) model.

    This model integrates a GATv2 encoder within a variational autoencoder (VAE) 
    framework to reconstruct gene regulatory networks from gene expression data 
    and topological baselines.
    """
    def __init__(self, encoder: nn.Module):
        """
        Initializes the TAGAT model.

        Args:
            encoder (nn.Module): The GATv2-based encoder module.
        """
        super().__init__()
        self.encoder = encoder

    def encode(self, *args, **kwargs) -> torch.Tensor:
        """
        Encodes input features and extracts the latent representation.

        During training, this method applies the reparameterization trick using 
        mu and clamped logstd. During evaluation, it returns the deterministic mu.

        Args:
            *args: Positional arguments passed to the encoder.
            **kwargs: Keyword arguments passed to the encoder.

        Returns:
            torch.Tensor: Latent representation z of shape (num_nodes, out_channels).
        """
        mu, logstd = self.encoder(*args, **kwargs)
        self.__mu__ = mu
        self.__logstd__ = logstd.clamp(min=-10, max=2)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        """
        Applies the reparameterization trick for variational inference.

        Args:
            mu (torch.Tensor): Mean vector of the latent distribution.
            logstd (torch.Tensor): Log standard deviation of the latent distribution.

        Returns:
            torch.Tensor: Sampled latent representation z.
        """
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu: torch.Tensor = None, logstd: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the Kullback-Leibler (KL) divergence loss.

        This regularizes the latent space by minimizing the divergence between 
        the predicted variational distribution and a standard normal prior.

        Args:
            mu (torch.Tensor, optional): Latent mean vector. Defaults to cached __mu__.
            logstd (torch.Tensor, optional): Latent log standard deviation vector. Defaults to cached __logstd__.

        Returns:
            torch.Tensor: Scalar KL divergence loss.
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1)
        )

##################################################
# PHASE 3: DIFFERENTIABLE TOPOLOGICAL LOSSES     #
##################################################

def scale_free_loss_v2(adj_pred: torch.Tensor, gamma: float = 2.5) -> torch.Tensor:
    """
    Computes the differentiable scale-free topology loss.

    This loss forces the predicted network to exhibit a power-law degree distribution, 
    which is characteristic of real biological networks. It performs a differentiable 
    linear regression in log-log space (log degree vs log rank), minimizing the 
    residual variance (1 - R^2), penalizing positive slopes, and maximizing 
    degree heterogeneity using the coefficient of variation.

    Args:
        adj_pred (torch.Tensor): Predicted soft adjacency matrix of shape (num_nodes, num_nodes).
        gamma (float, optional): Power-law exponent target parameter. Defaults to 2.5.

    Returns:
        torch.Tensor: Scalar scale-free regularization loss.
    """
    degrees = torch.sum(adj_pred, dim=1)
    degrees = torch.clamp(degrees, min=0.1)
    
    sorted_deg, _ = torch.sort(degrees, descending=True)
    
    N = adj_pred.size(0)
    ranks = torch.arange(1, N + 1, device=adj_pred.device, dtype=torch.float32)
    
    log_ranks = torch.log(ranks)
    log_degrees = torch.log(sorted_deg + 1e-6)
    
    mean_x = log_ranks.mean()
    mean_y = log_degrees.mean()
    
    dx = log_ranks - mean_x
    dy = log_degrees - mean_y
    
    ss_xy = (dx * dy).sum()
    ss_xx = (dx ** 2).sum()
    ss_yy = (dy ** 2).sum()
    
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy + 1e-9)
    linearity_loss = 1.0 - r_squared
    
    slope = ss_xy / (ss_xx + 1e-9)
    slope_penalty = F.relu(slope + 0.3)
    
    degree_cv = degrees.std() / (degrees.mean() + 1e-9)
    uniformity_penalty = 1.0 / (degree_cv + 0.1)
    
    return linearity_loss + 0.5 * slope_penalty + 0.05 * uniformity_penalty

def targeted_sparsity_loss(adj_pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the targeted sparsity loss.

    This loss penalizes high minimum and median node degrees in the predicted network, 
    enforcing a sparse periphery and allowing a few central hub genes to remain highly 
    connected. It controls overall network density without destroying biological hubs.

    Args:
        adj_pred (torch.Tensor): Predicted soft adjacency matrix of shape (num_nodes, num_nodes).

    Returns:
        torch.Tensor: Scalar targeted sparsity loss.
    """
    degrees = torch.sum(adj_pred, dim=1)
    N = adj_pred.size(0)
    
    norm_degrees = degrees / N
    
    min_degree_penalty = torch.min(norm_degrees)
    
    sorted_deg, _ = torch.sort(norm_degrees)
    median_idx = len(sorted_deg) // 2
    median_degree_penalty = sorted_deg[median_idx]
    
    return 0.5 * min_degree_penalty + 0.5 * median_degree_penalty

def ta_gat_loss(z: torch.Tensor, 
                pos_edge_index: torch.Tensor, 
                neg_edge_index: torch.Tensor, 
                lambda_sf: float = 5.0,
                pos_edge_weights: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the combined loss terms for the TA-GAT training step.

    Integrates standard binary cross-entropy reconstruction loss over positive 
    and negative sampled edges, scale-free topological regularization, and 
    targeted sparsity loss.

    Args:
        z (torch.Tensor): Latent node representations of shape (num_nodes, out_channels).
        pos_edge_index (torch.Tensor): Positive edge indices of shape (2, num_pos_edges).
        neg_edge_index (torch.Tensor): Sampled negative edge indices of shape (2, num_neg_edges).
        lambda_sf (float, optional): Weight of the scale-free regularization loss. Defaults to 5.0.
        pos_edge_weights (torch.Tensor, optional): Edge weights for positive edges. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - recon_loss (torch.Tensor): Scalar edge reconstruction loss (BCE).
            - sf_loss (torch.Tensor): Scalar scale-free topological regularization loss.
            - sparsity_loss (torch.Tensor): Scalar targeted network sparsity loss.
    """
    EPS = 1e-15
    
    pos_pred = torch.sigmoid(torch.sum(z[pos_edge_index[0]] * z[pos_edge_index[1]], dim=1))
    neg_pred = torch.sigmoid(torch.sum(z[neg_edge_index[0]] * z[neg_edge_index[1]], dim=1))
    
    pos_loss = -torch.log(pos_pred + EPS).mean()
    neg_loss = -torch.log(1 - neg_pred + EPS).mean()
    recon_loss = pos_loss + neg_loss
    
    z_norm = F.normalize(z, p=2, dim=1)
    adj_pred_dense = torch.sigmoid(torch.matmul(z_norm, z_norm.t()))
    mask = ~torch.eye(adj_pred_dense.size(0), dtype=torch.bool, device=adj_pred_dense.device)
    adj_pred_dense = adj_pred_dense * mask
    
    sf_loss = scale_free_loss_v2(adj_pred_dense)
    sparsity_loss = targeted_sparsity_loss(adj_pred_dense)
    
    return recon_loss, sf_loss, sparsity_loss
