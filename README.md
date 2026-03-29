# GSoC 2026 — ML4SCI: Non-local GNNs for Jet Classification

**Applicant:** Sanjana Soni · [github.com/isha822](https://github.com/isha822)  
**Organisation:** ML4SCI · **Project:** Non-local GNNs for Jet Classification  
**Mentors:** Sergei Gleyzer (Univ. Alabama) · Ali Hariri (EPFL) · Tom Magorsch (TUM)

---

## Repository Structure

```
GSoC-2026-ML4SCI-Jet-Classification/
├── ML4SCI-GSoC-2026/
│   ├── Notebooks/
│   │   ├── Common_task_1.ipynb                          ← Task 1: Autoencoder
│   │   ├── ML4SCI_Tasks_2_and_4.ipynb                   ← Task 2 + Specific Task 4
│   │   ├── 01_Jet_Classification_Evaluation.ipynb       ← Pre-proposal: Local vs Non-local GNN
│   │   ├── 01_1_jet_classification_efficiency_evaluation.ipynb  ← Pre-proposal: Efficiency experiments
│   │   └── 02_NON_LOCAL_GNN.ipynb                       ← Pre-proposal: Non-local GNN implementation
│   └── model_weights/
│       ├── edgeconv_baseline.pt
│       ├── nonlocal_best_task4.pt
│       ├── fastlinear_best_task4.pt
│       └── local_gnn_weights.pth
└── README.md
```

---

## Evaluation Tasks

### Common Task 1 — Physics-Aware Autoencoder
**Notebook:** `Notebooks/Common_task_1.ipynb`

U-Net convolutional autoencoder trained to reconstruct high-granularity Quark/Gluon jet images (125×125×3 — ECAL, HCAL, Tracks).

| Metric | Result |
|---|---|
| ECAL MSE (test) | 0.000010 |
| HCAL MSE (test) | 0.000013 |
| Tracks MSE (test) | 0.000002 |
| Mean Relative Energy Error | 0.2695 |

Key design decisions: skip connections to prevent mean collapse, bilinear upsampling to eliminate checkerboard artifacts, physics-weighted MSE loss with channel weights [1, 2, 10].

---

### Common Task 2 — Baseline GNN for Jet Classification
**Notebook:** `Notebooks/ML4SCI_Tasks_2_and_4.ipynb`

Local EdgeConv GNN for binary Quark/Gluon classification. Images converted to sparse point clouds via chunked HDF5 loader (50k jets, 5k/chunk).

| Metric | Result |
|---|---|
| Architecture | EdgeConv GNN, k=7 |
| Complexity | O(N·k) |
| Val AUC | 0.796 |

---

### Specific Task 4 — Non-Local GNNs and the O(N²) Efficiency Bottleneck
**Notebook:** `Notebooks/ML4SCI_Tasks_2_and_4.ipynb`

Comparison of local, non-local (O(N²)), and linear attention (O(N)) architectures on the Quark/Gluon dataset.

| Model | Complexity | AUC | Notes |
|---|---|---|---|
| EdgeConv baseline | O(N·k) | 0.796 | Full uncapped dataset |
| NonLocalGNN (TransformerConv) | O(N²) | 0.788 | Top-100 particles/jet |
| FastLinearGNN (Performer-style) | O(N) | 0.778 | Top-100 particles/jet |

FastLinearGNN matches NonLocalGNN within 1% AUC while eliminating graph construction entirely — validating the O(N) linear attention approach proposed in NL-ParticleNet.

---

## Pre-Proposal Experiments

Structured experiments on the **Top Tagging Reference Dataset** (Zenodo 2603256, 200k jets) establishing the architectural and efficiency baselines for the GSoC proposal.

### Experiment Results Summary

| Model | AUC | Latency | Memory | Complexity |
|---|---|---|---|---|
| Local GNN (EdgeConv, k=7) | 0.96290 | 44.6 ms | 794 MB | O(N·k) |
| Hybrid GNN (negative result) | 0.96495 | 254.6 ms | 3,105 MB | O(Nk+N²) |
| Non-local GNN (TransformerConv) | 0.97495 | 260.7 ms | 1,589 MB | O(N²) |
| FastLinear GNN (Performer) | 0.97155 | 21.3 ms | 564 MB | O(N) |

**Key finding:** FastLinearGNN retains 99.6% of Non-local GNN accuracy at 12.45× speedup and 2.84× memory reduction.

**Notebooks:**
- `01_Jet_Classification_Evaluation.ipynb` — Local vs Non-local GNN, sequence baselines, O(N²) profiling
- `01_1_jet_classification_efficiency_evaluation.ipynb` — Hybrid GNN and FastLinear efficiency experiments
- `02_NON_LOCAL_GNN.ipynb` — Non-local GNN implementation and LinearAttentionGNN

---

## Dataset

- **Common Tasks:** [Quark/Gluon Dataset](https://drive.google.com/file/d/1W02K-SfU2dntGU48b3IYBp9Rh7rtTYEr) — 139,306 jets, 125×125×3 images
- **Pre-proposal:** [Top Tagging Reference Dataset](https://zenodo.org/record/2603256) — Zenodo 2603256

---

## Setup

```bash
pip install torch torch-geometric
pip install torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-{version}+cu{cuda}.html
pip install h5py scikit-learn matplotlib tqdm
```

All notebooks are self-contained and runnable on Google Colab with a GPU runtime.
