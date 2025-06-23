#!/usr/bin/env python3
"""
Script to train and evaluate a PGCN model for disease–gene link prediction.

This module parses command-line arguments, loads feature and edge data,
constructs train/test splits, initializes the PGCN model, performs training,
and computes prediction scores on held-out associations.
"""

import argparse
import logging
from pathlib import Path

import torch

from pgcn.dataloader import build_train_test_data
from pgcn.model import PGCN
from pgcn.train_test import test, train


def parse() -> argparse.Namespace:
    """
    Parse command-line arguments for PGCN training and evaluation.

    Returns:
        argparse.Namespace: Parsed arguments, including paths to input tensors,
        training hyperparameters, and device selection.
    """
    parser = argparse.ArgumentParser(
        description="Train and test PGCN for disease–gene link prediction."
    )
    parser.add_argument(
        "--gene_features",
        type=Path,
        required=True,
        help="Path to .pt file with gene_features tensor",
    )
    parser.add_argument(
        "--disease_features",
        type=Path,
        required=True,
        help="Path to .pt file with disease_features tensor",
    )
    parser.add_argument(
        "--edge_gene_gene",
        type=Path,
        required=True,
        help="Path to .pt file with edge_index_gene_gene tensor",
    )
    parser.add_argument(
        "--edge_dis_dis",
        type=Path,
        required=True,
        help="Path to .pt file with edge_index_dis_dis tensor",
    )
    parser.add_argument(
        "--assoc_gene_dis",
        type=Path,
        required=True,
        help="Path to .pt file with assoc_gene_dis tensor",
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Positive edges per mini-batch"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay (L2 penalty)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on ('cpu' or 'cuda')",
    )
    args = parser.parse_args()

    # Verify that all input files exist
    for path_attr in (
        "gene_features",
        "disease_features",
        "edge_gene_gene",
        "edge_dis_dis",
        "assoc_gene_dis",
    ):
        path: Path = getattr(args, path_attr)
        if not path.exists():
            parser.error(f"File not found: {path}")

    return args


def load(args: argparse.Namespace):
    """
    Load feature tensors and build train/test heterogeneous data.

    Args:
        args (argparse.Namespace): Parsed command-line arguments with file paths
            and device selection.

    Returns:
        tuple:
            - train_data (HeteroData): Training graph with gene and disease nodes
              and training associations only, moved to the specified device.
            - test_edges (torch.Tensor): Held-out test association edges of shape [2, num_test].
            - gene_feats (torch.Tensor): Loaded gene feature matrix.
            - dis_feats (torch.Tensor): Loaded disease feature matrix.
    """
    # Load all tensors
    gene_feats = torch.load(args.gene_features)
    dis_feats = torch.load(args.disease_features)
    edge_gg = torch.load(args.edge_gene_gene)
    edge_dd = torch.load(args.edge_dis_dis)
    assoc_gd = torch.load(args.assoc_gene_dis)

    # Build train/test data
    train_data, test_edges = build_train_test_data(
        gene_feats, dis_feats, edge_gg, edge_dd, assoc_gd, test_ratio=0.1, seed=42
    )
    train_data = train_data.to(args.device)

    return train_data, test_edges, gene_feats, dis_feats


def main() -> None:
    """
    Main entry point: parse args, load data, initialize model, train and test.

    Workflow:
        1. Parse command-line arguments.
        2. Load feature matrices and build train/test splits.
        3. Initialize PGCN model.
        4. Train the model on training data.
        5. Evaluate and print scores on held-out test edges.
    """
    # --- configure logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("pgcn-main")

    args = parse()
    logger.info("Arguments parsed successfully")

    train_data, test_edges, gene_feats, dis_feats = load(args)
    logger.info(
        "Data loaded: %d genes, %d diseases, %d training edges",
        train_data["gene"].num_nodes,
        train_data["disease"].num_nodes,
        train_data["gene", "assoc", "disease"].edge_index.size(1),
    )

    # Initialize model
    model = PGCN(
        num_gene_features=gene_feats.size(1),
        num_disease_features=dis_feats.size(1),
        hidden_channels=64,
        out_channels=32,
        dropout=0.1,
    ).to(args.device)
    logger.info("Model initialized on %s", args.device)

    # Train
    train(
        model,
        train_data,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )

    # Test on held-out edges
    scores = test(model, train_data, test_edges, batch_size=args.batch_size)
    logger.info("Test positive-edge scores: %s", scores.tolist())


if __name__ == "__main__":
    main()
