# PGCN Link Prediction

Implementation of the PGCN method for disease–gene prioritization via heterogeneous graph convolutional networks described *Li, Yu & Kuwahara, Hiroyuki & Yang, Peng & Song, Le & Gao, Xin. (2019). PGCN: Disease gene prioritization by disease and gene embedding through graph convolutional neural networks. 10.1101/532226.*

---

## Installation

```bash
pip install torch torch-geometric
``` 

Ensure dependencies for PyTorch Geometric are satisfied: see https://pytorch-geometric.readthedocs.io/

---

## Data Preparation

Prepare three matrices:

1. **Gene features**: `[num_genes, gene_feat_dim]`
2. **Disease features**: `[num_diseases, dis_feat_dim]`
3. **Associations**: binary adjacency `[num_genes, num_diseases]` or edge index tensor of shape `[2, num_edges]`

Use `build_hetero_data(...)` in `data_loader.py` to construct a `HeteroData` object.

---

## Usage Example

```python
from data_loader import build_hetero_data
from model import PGCN
from train_test import train, test

# Load feature matrices and edge indices (as torch tensors)
data = build_hetero_data(
    gene_features,
    disease_features,
    assoc_edge_index
)
model = PGCN(hidden_channels=64, out_channels=32, dropout=0.1)
train(model, data)

# Evaluate on new pairs
eval_scores = test(model, data, edge_label_index)
```

---

## Project Structure

```
├── model.py         # PGCN class (2-layer heterogeneous GCN + bilinear decoder)
├── data_loader.py   # build_hetero_data helper
├── train_test.py    # train() and test() routines
└── README.md        # this documentation
```

---

## Mathematical Formulation

Let $G=(V,E)$ be a heterogeneous graph with node sets $V_{g}$ (genes) and $V_{d}$ (diseases), and edge sets:
- $E_{gg}$: gene–gene interactions
- $E_{dd}$: disease–disease similarities
- $E_{gd}$: known gene–disease associations
- $E_{dg}$: reverse associations for symmetry

Feature matrices:
- $X_{g}\in\mathbb{R}^{|V_g|\times F_g}$
- $X_{d}\in\mathbb{R}^{|V_d|\times F_d}$

### Graph Convolutional Encoder
At layer $k$, each node $i$'s hidden representation $\mathbf{h}_i^{(k)}$ updates to:

$$
\mathbf{h}_i^{(k+1)} = \phi\Bigl( \sum_{l\in\{gg,dd,gd,dg\}} \sum_{j\in N_i^l} \frac{1}{\sqrt{|N_i^l|\,|N_j^l|}} W_l^{(k)} \mathbf{h}_j^{(k)} + W_{self}^{(k)}\mathbf{h}_i^{(k)}\Bigr),
$$

- $N_i^l$: neighbors via relation $l$
- $W_l^{(k)}$: relation-specific weight
- $W_{self}^{(k)}$: self-loop weight
- $\phi=\mathrm{ReLU}$
- $\mathbf{h}_i^{(0)}$ initialized from node features

### Decoder
Final embeddings $\mathbf{z}_g,\mathbf{z}_d$ yield link score:

$$
s_{gd} = \sigma\bigl(\mathbf{z}_g^T W_{dec} \mathbf{z}_d\bigr),
$$

with $W_{dec}\in\mathbb{R}^{D\times D}$ and sigmoid $\sigma(x)$.

### Loss
Binary cross-entropy with negative sampling:

$$
\mathcal{L} = -\sum_{(g,d)\in E_{gd}} \Bigl[\log s_{gd} + \mathbb{E}_{d'\sim U}\log(1 - s_{gd'})\Bigr].
$$

Negative samples $d'$ drawn uniformly over diseases.

---

## License
MIT
