# train_test.py

import math
from typing import Optional

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData
from tqdm import tqdm

from pgcn.dataloader import create_link_pred_dataloader
from pgcn.model import PGCN


def train(
    model: PGCN,
    data: HeteroData,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 512,
    num_neg_samples: int = 1,
    log_interval: int = 50,
    seed: int = 42,
    log_dir: str = "runs",
) -> None:
    """
    Train a PGCN model using mini-batches of edges and negative sampling via DataLoader,
    while logging training metrics to TensorBoard.

    Args:
        model (PGCN): Initialized PGCN model (with dropout and Xavier init).
        data (HeteroData): Heterogeneous graph containing training edges.
        epochs (int): Number of epochs to train (default: 300).
        lr (float): Learning rate for Adam optimizer (default: 1e-3).
        weight_decay (float): Weight decay (L2 regularization) (default: 1e-5).
        batch_size (int): Number of positive edges per mini-batch (default: 512).
        num_neg_samples (int): Number of negative samples per positive edge (default: 1).
        log_interval (int): Epoch interval for logging to console (default: every 50 epochs).
        seed (int): Random seed for reproducibility (default: 42).
        log_dir (str): Directory where TensorBoard logs will be written.
    """
    device = next(model.parameters()).device
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Prepare DataLoader for link prediction
    train_loader = create_link_pred_dataloader(
        data,
        batch_size=batch_size,
        num_neg_samples=num_neg_samples,
        shuffle=True,
        seed=seed,
    )

    num_pos = data["gene", "assoc", "disease"].edge_index.size(1)

    # Log hyperparameters
    writer.add_hparams(
        {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "num_neg_samples": num_neg_samples,
            "dropout": model.dropout,
        },
        {"hparam/avg_loss": 0},  # placeholder for scalar
    )

    for epoch in tqdm(range(1, epochs + 1), total=epochs, desc="Training"):
        epoch_loss = 0.0

        for pos_batch, neg_batch in train_loader:
            pos_batch = pos_batch.to(device)  # [2, B]
            neg_batch = neg_batch.to(device)  # [2, B * num_neg_samples]

            optimizer.zero_grad()

            # Forward: compute all embeddings once per batch
            z_dict = model(data.x_dict, data.edge_index_dict)

            # Decode scores
            pos_scores = model.decode(z_dict, pos_batch)  # [B]
            neg_scores = model.decode(z_dict, neg_batch)  # [B * num_neg_samples]

            # Compute losses
            pos_loss = -torch.log(pos_scores + 1e-15).mean()
            neg_loss = -torch.log(1.0 - neg_scores + 1e-15).mean()
            loss = pos_loss + neg_loss

            # Backprop + optimize
            loss.backward()
            optimizer.step()

            # Accumulate weighted by number of positives in this batch
            epoch_loss += loss.item() * pos_batch.size(1)

        # Compute average loss per positive edge
        avg_loss = epoch_loss / num_pos

        # Log to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        # Console logging
        if epoch % log_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs}, Avg Loss: {avg_loss:.4f}")

    # Close the TensorBoard writer
    writer.close()


def test(
    model: PGCN,
    data: HeteroData,
    edge_label_index: torch.Tensor,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute predicted probabilities for given gene–disease pairs.

    Args:
        model (PGCN): Trained PGCN model.
        data (HeteroData): Heterogeneous graph data.
        edge_label_index (torch.Tensor): Shape [2, num_pairs], pairs to score.
        batch_size (Optional[int]): If provided, split scoring into mini-batches.

    Returns:
        torch.Tensor: Predicted probabilities for each pair, shape [num_pairs].
    """
    model.eval()
    device = next(model.parameters()).device
    edge_label_index = edge_label_index.to(device)

    with torch.no_grad():
        # Compute embeddings once
        z_dict = model(data.x_dict, data.edge_index_dict)

        # Score all at once or in batches
        if batch_size is None:
            return model.decode(z_dict, edge_label_index)

        scores = []
        num_pairs = edge_label_index.size(1)
        for start in tqdm(
            range(0, num_pairs, batch_size),
            total=math.ceil(num_pairs / batch_size),
            desc="Testing",
        ):
            end = min(start + batch_size, num_pairs)
            batch_idx = edge_label_index[:, start:end]
            scores.append(model.decode(z_dict, batch_idx))
        return torch.cat(scores, dim=0)
