from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv


class PGCN(torch.nn.Module):
    """
    Graph convolutional network for disease–gene link prediction.

    Embeds genes and diseases in a shared latent space via two layers of
    heterogeneous graph convolution and predicts associations using a
    bilinear decoder. Supports dropout and Xavier initialization.

    Args:
        hidden_channels (int): Dimensionality of the intermediate embeddings.
        out_channels (int): Dimensionality of the final embeddings.
        dropout (float): Dropout rate applied after each GCN layer.

    Attributes:
        conv_layer1 (HeteroConv): First heterogeneous graph convolution layer.
        conv_layer2 (HeteroConv): Second heterogeneous graph convolution layer.
        bilinear_decoder (torch.nn.Bilinear): Bilinear layer for link prediction.
        dropout_layer (torch.nn.Dropout): Dropout module.
    """

    def __init__(
        self, hidden_channels: int = 64, out_channels: int = 32, dropout: float = 0.1
    ) -> None:
        """
        Initialize PGCN layers with dropout and Xavier initialization.

        Args:
            hidden_channels (int): Dimensionality of the hidden embeddings.
            out_channels (int): Dimensionality of the output embeddings.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.dropout_layer = torch.nn.Dropout(p=dropout)

        # First heterogeneous graph convolution
        conv_layers1 = {
            ("gene", "interacts", "gene"): GCNConv(-1, hidden_channels),
            ("disease", "similar", "disease"): GCNConv(-1, hidden_channels),
            ("gene", "assoc", "disease"): GCNConv(-1, hidden_channels),
            ("disease", "rev_assoc", "gene"): GCNConv(-1, hidden_channels),
        }
        self.conv_layer1 = HeteroConv(conv_layers1, aggr="sum")

        # Second heterogeneous graph convolution
        conv_layers2 = {
            ("gene", "interacts", "gene"): GCNConv(hidden_channels, out_channels),
            ("disease", "similar", "disease"): GCNConv(hidden_channels, out_channels),
            ("gene", "assoc", "disease"): GCNConv(hidden_channels, out_channels),
            ("disease", "rev_assoc", "gene"): GCNConv(hidden_channels, out_channels),
        }
        self.conv_layer2 = HeteroConv(conv_layers2, aggr="sum")

        # Bilinear decoder for computing link scores
        self.bilinear_decoder = torch.nn.Bilinear(out_channels, out_channels, 1)

        # Xavier initialization for convolution and decoder weights
        for module in self.modules():
            if isinstance(module, GCNConv):
                torch.nn.init.xavier_uniform_(module.lin.weight)
            if isinstance(module, torch.nn.Bilinear):
                torch.nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute node embeddings via two heterogeneous GCN layers.

        Args:
            x_dict (Dict[str, torch.Tensor]):
                Input features per node type, e.g. {'gene': Tensor, 'disease': Tensor}.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]):
                Edge indices keyed by (src_node, relation, dst_node).

        Returns:
            Dict[str, torch.Tensor]: Updated node embeddings per type.
        """
        # First convolution + ReLU
        hidden_dict = self.conv_layer1(x_dict, edge_index_dict)
        hidden_dict = {
            node_type: F.relu(feats) for node_type, feats in hidden_dict.items()
        }

        # Apply dropout
        hidden_dict = {
            nt: self.dropout_layer(feats) for nt, feats in hidden_dict.items()
        }

        # Second convolution + ReLU
        out_dict = self.conv_layer2(hidden_dict, edge_index_dict)
        out_dict = {node_type: F.relu(feats) for node_type, feats in out_dict.items()}

        return out_dict

    def decode(
        self, embeddings: Dict[str, torch.Tensor], edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict link probabilities for gene–disease pairs.

        Args:
            embeddings (Dict[str, torch.Tensor]):
                Node embeddings keyed by type.
            edge_label_index (torch.Tensor):
                Indices of candidate edges, shape [2, num_edges].

        Returns:
            torch.Tensor: Sigmoid probabilities per edge.
        """
        gene_embeddings = embeddings["gene"]
        disease_embeddings = embeddings["disease"]
        gene_indices, disease_indices = edge_label_index

        # Compute bilinear score and apply sigmoid
        logits = self.bilinear_decoder(
            gene_embeddings[gene_indices], disease_embeddings[disease_indices]
        ).view(-1)

        return torch.sigmoid(logits)
