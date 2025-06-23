from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData


def build_hetero_data(
    gene_features: torch.Tensor,
    disease_features: torch.Tensor,
    edge_index_gene_gene: torch.Tensor,
    edge_index_dis_dis: torch.Tensor,
    assoc_gene_dis: torch.Tensor,
) -> HeteroData:
    """
    Build a heterogeneous graph for PGCN training.

    Constructs a HeteroData object containing gene and disease nodes,
    along with gene–gene, disease–disease, and gene–disease edges.

    Args:
        gene_features (torch.Tensor):
            Node feature matrix for genes of shape [num_genes, gene_feat_dim].
        disease_features (torch.Tensor):
            Node feature matrix for diseases of shape [num_diseases, disease_feat_dim].
        edge_index_gene_gene (torch.Tensor):
            Edge index for gene–gene interactions, shape [2, num_edges_gene_gene].
        edge_index_dis_dis (torch.Tensor):
            Edge index for disease–disease similarities, shape [2, num_edges_dis_dis].
        assoc_gene_dis (torch.Tensor):
            Edge index for gene–disease associations, shape [2, num_edges_assoc].

    Returns:
        HeteroData: A populated heterogeneous graph with fields:
            - data['gene'].x: Gene node features.
            - data['disease'].x: Disease node features.
            - data['gene','interacts','gene'].edge_index: Gene–gene edges.
            - data['disease','similar','disease'].edge_index: Disease–disease edges.
            - data['gene','assoc','disease'].edge_index: Gene–disease edges.
            - data['disease','rev_assoc','gene'].edge_index: Reversed gene–disease edges.
    """
    data = HeteroData()

    # Assign node feature matrices
    data["gene"].x = gene_features
    data["disease"].x = disease_features

    # Assign edges for each relation
    data["gene", "interacts", "gene"].edge_index = edge_index_gene_gene
    data["disease", "similar", "disease"].edge_index = edge_index_dis_dis
    data["gene", "assoc", "disease"].edge_index = assoc_gene_dis

    # Reverse the gene–disease associations for bidirectional message passing
    data["disease", "rev_assoc", "gene"].edge_index = assoc_gene_dis.flip(0)

    return data


def split_gene_disease_edges(
    assoc_edge_index: torch.Tensor, test_ratio: float = 0.1, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split gene–disease association edges into training and test sets.

    Args:
        assoc_edge_index (torch.Tensor):
            Tensor of shape [2, num_edges] containing all gene–disease edges.
        test_ratio (float, optional):
            Fraction of edges to reserve for testing. Defaults to 0.1.
        seed (int, optional):
            Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            A tuple (train_edge_index, test_edge_index), each of shape [2, n_edges].
    """
    num_edges = assoc_edge_index.size(1)
    num_test = int(num_edges * test_ratio)

    # Generate a reproducible random permutation of edge indices
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_edges, generator=rng)

    test_idx = perm[:num_test]
    train_idx = perm[num_test:]

    train_edge_index = assoc_edge_index[:, train_idx]
    test_edge_index = assoc_edge_index[:, test_idx]
    return train_edge_index, test_edge_index


def build_train_test_data(
    gene_features: torch.Tensor,
    disease_features: torch.Tensor,
    edge_index_gene_gene: torch.Tensor,
    edge_index_dis_dis: torch.Tensor,
    assoc_gene_dis: torch.Tensor,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[HeteroData, torch.Tensor]:
    """
    Build training HeteroData object and extract test edges.

    This function splits the provided gene–disease associations into
    train and test sets, then constructs a HeteroData graph using only
    the training associations.

    Args:
        gene_features (torch.Tensor):
            Feature matrix for gene nodes.
        disease_features (torch.Tensor):
            Feature matrix for disease nodes.
        edge_index_gene_gene (torch.Tensor):
            Gene–gene edge indices.
        edge_index_dis_dis (torch.Tensor):
            Disease–disease edge indices.
        assoc_gene_dis (torch.Tensor):
            Full set of gene–disease association edges.
        test_ratio (float, optional):
            Fraction of associations to hold out for testing. Defaults to 0.1.
        seed (int, optional):
            Random seed for splitting. Defaults to 42.

    Returns:
        Tuple[HeteroData, torch.Tensor]:
            - train_data (HeteroData): Graph built with only training associations.
            - test_pos_edge_index (torch.Tensor): Held-out test associations.
    """
    train_edges, test_edges = split_gene_disease_edges(
        assoc_gene_dis, test_ratio=test_ratio, seed=seed
    )
    train_data = build_hetero_data(
        gene_features,
        disease_features,
        edge_index_gene_gene,
        edge_index_dis_dis,
        train_edges,
    )
    return train_data, test_edges


class LinkPredictionDataset(Dataset):
    """
    PyTorch Dataset for link prediction on gene–disease associations.

    Iterates over positive edges and generates negative samples on the fly.

    Attributes:
        edge_index (torch.Tensor): Positive edge indices [2, num_pos_edges].
        num_genes (int): Number of gene nodes.
        num_diseases (int): Number of disease nodes.
        num_neg_samples (int): Number of negative samples per positive edge.
        rng (torch.Generator): Random generator for reproducibility.
        pos_set (set): Set of existing (gene, disease) tuples for filtering.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_genes: int,
        num_diseases: int,
        num_neg_samples: int = 1,
        seed: int = 42,
    ) -> None:
        """
        Initialize the link prediction dataset.

        Args:
            edge_index (torch.Tensor):
                Positive gene–disease edge indices, shape [2, num_edges].
            num_genes (int):
                Total number of gene nodes.
            num_diseases (int):
                Total number of disease nodes.
            num_neg_samples (int, optional):
                Number of negative samples per positive edge. Defaults to 1.
            seed (int, optional):
                Random seed for negative sampling. Defaults to 42.
        """
        self.edge_index = edge_index
        self.num_genes = num_genes
        self.num_diseases = num_diseases
        self.num_neg_samples = num_neg_samples
        self.rng = torch.Generator().manual_seed(seed)

        # Precompute set of positive edges for O(1) negative sampling checks
        self.pos_set = {
            (int(edge_index[0, i]), int(edge_index[1, i]))
            for i in range(edge_index.size(1))
        }

    def __len__(self) -> int:
        """Return the total number of positive edges."""
        return self.edge_index.size(1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a positive edge and corresponding negative samples.

        Args:
            idx (int): Index of the positive edge.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - pos_edge: Tensor of shape [2] (gene_idx, disease_idx).
                - neg_edges: Tensor of shape [num_neg_samples, 2].
        """
        gene_idx = int(self.edge_index[0, idx])
        dis_idx = int(self.edge_index[1, idx])

        # Sample negatives for this gene
        negs = []
        for _ in range(self.num_neg_samples):
            while True:
                neg_dis = int(
                    torch.randint(0, self.num_diseases, (1,), generator=self.rng).item()
                )
                if (gene_idx, neg_dis) not in self.pos_set:
                    negs.append([gene_idx, neg_dis])
                    break

        pos_edge = torch.tensor([gene_idx, dis_idx], dtype=torch.long)
        neg_edges = torch.tensor(negs, dtype=torch.long)
        return pos_edge, neg_edges


def _collate_link_pred(
    batch: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to batch link prediction samples.

    Args:
        batch: Sequence of tuples (pos_edge [2], neg_edges [num_neg, 2]).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - batched_pos: Tensor of shape [2, batch_size].
            - batched_neg: Tensor of shape [2, batch_size * num_neg_samples].
    """
    pos_list, neg_list = zip(*batch)
    # Stack positive edges: [2, batch_size]
    batched_pos = torch.stack(pos_list, dim=1)

    # Stack negative edges: [num_neg, batch_size, 2]
    neg = torch.stack(neg_list, dim=1)
    num_neg, batch_size, _ = neg.size()
    # Rearrange to [2, batch_size * num_neg]
    batched_neg = neg.permute(2, 1, 0).reshape(2, batch_size * num_neg)
    return batched_pos, batched_neg


def create_link_pred_dataloader(
    data: HeteroData,
    batch_size: int = 512,
    num_neg_samples: int = 1,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader for mini-batch link prediction training.

    Args:
        data (HeteroData):
            Heterogeneous graph containing
            data['gene','assoc','disease'].edge_index.
        batch_size (int, optional):
            Number of positive edges per batch. Defaults to 512.
        num_neg_samples (int, optional):
            Number of negative samples per positive edge. Defaults to 1.
        shuffle (bool, optional):
            Whether to shuffle positive edges each epoch. Defaults to True.
        seed (int, optional):
            Random seed for negative sampling. Defaults to 42.

    Returns:
        DataLoader: Yields tuples (pos_edge_index [2, B], neg_edge_index [2, B * num_neg_samples]).
    """
    pos_edge_index = data["gene", "assoc", "disease"].edge_index
    num_genes = data["gene"].x.size(0)
    num_diseases = data["disease"].x.size(0)

    dataset = LinkPredictionDataset(
        edge_index=pos_edge_index,
        num_genes=num_genes,
        num_diseases=num_diseases,
        num_neg_samples=num_neg_samples,
        seed=seed,
    )

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_link_pred
    )
