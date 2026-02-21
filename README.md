# Non-local GNNs for Jet Classification (GSoC 2026 Evaluation)

This repository contains the evaluation tasks and proof-of-concept models for the *ML4SCI GENIE Project: Non-local GNNs for Jet Classification*. 

The primary objective is to capture long-range dependencies inherent in jet morphology—specifically focusing on the 3-prong decay structure of Top quarks—which standard Message Passing Neural Networks (MPNNs) struggle to aggregate.

## 📊 Benchmark Results (Top Tagging Reference Dataset)
Evaluated on 200,000 jets using physical 4-momenta ($p_T, \eta, \phi, E$) as node features.

| Model Architecture | Graph Topology | ROC-AUC |
| :--- | :--- | :--- |
| *Non-local GNN (Graph Transformer)* | *Fully Connected* | *0.9750* |
| Local GNN (EdgeConv) | $k$-NN ($k=7$) | 0.9629 |
| Ensemble (GRU + Transformer) | Sequence | 0.9418 |
| Bi-directional GRU | Sequence | 0.9410 |

Note: The Non-local GNN achieves a significant +1.2% AUC improvement over the local baseline by utilizing a fully connected attention mechanism.

## 📁 Repository Structure
* /notebooks: Contains the primary evaluation submission and the raw training code for the GNNs.
* /model_weights: Includes the trained .pth state dictionaries for rapid inference and reproducibility.
* /results: Contains the generated ROC curves and benchmark visualizations.

## 🚀 Next Steps (GSoC Proposal Focus)
While the fully connected Graph Transformer successfully captures non-local dependencies, it scales at $O(N^2)$. The formal GSoC proposal will focus on optimizing this memory bottleneck via *Linear Attention mechanisms (e.g., Performers)* and exploring *Simplicial Complexes* to handle higher particle multiplicities on larger datasets (e.g., JetClass).
