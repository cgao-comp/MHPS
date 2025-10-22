## MHPS: Enhancing Information Diffusion Prediction via Multiple Granularity Hypergraphs and Position-aware Sequence Model


This paper introduces MHPS, a method that proposes the multi-granularity hypergraphs and position-aware sequence model for information diffusion prediction.



## Abstract

With the rise of social media, accurately predicting information diffusion has become crucial for a wide range of applications. Existing methods usually employ sequential hypergraphs to model users’latent interaction preferences and use self-attention mechanisms to capture dependencies with users. However, they typically focus on a single temporal scale and lack the ability to effectively model temporal influence, which limits their performance in diffusion prediction tasks. To address these limitations, we propose a novel method (MHPS) to enhance information diffusion prediction via multiple granularity hypergraphs and a position-aware sequence model. Specifically, MHPS constructs hypergraph sequences of different granularities by grouping user interactions according to various time intervals. Additionally, to further enhance the modeling of temporal influence, two types of cross-attention mechanisms, namely next-step positional cross-attention and source influence cross-attention, are introduced within the cascade representation. The next-step positional cross-attention captures target position awareness, while the source influence cross-attention focuses on the impact of the initial source. Then, gating mechanisms and GRUs are employed to fuse the different attention outputs and predict the next target user. Extensive experiments on real-world datasets demonstrate that MHPS achieves competitive performance against state-of-the-art methods. The average improvements are up to 7.82% in terms of Hits@10 and 5.60% in terms of MAP@100.



## Project Structure
The project comprises the following key components:

- **DataConstruct.py**: Utilities for loading datasets, preprocessing data, and preparing input for the model.
- **GraphBuilder.py**: Utilities for constructing various types of graphs (e.g., heterogeneous graphs, hypergraphs).
- **Metrics.py**: Utilities for computing evaluation metrics to assess model performance.
- **MHPS.py**: Implementation of the main MHPS model, including graph encoders and sequence models.
- **Decoder.py,GraphEncoder.py,Merger.py,TransformerBlock.py**: Custom layers such as lightgcn, fusion layer and decoder.
- **run.py**: Training scripts, including dataset handling, loss calculation, and optimization steps.
- **Constants.py**: Helper functions for preprocessing, evaluation, and logging.
- **dani.py**: Utilities for constructing network inference graphs, with a configurable number of edges.




## Installation
To set up the environment for SIGKAN:

1. **Clone the repository**:
   ```sh
   git clone at https://github.com/cgao-comp/MHPS
   cd Mhps
   ```

2. **Create a virtual environment** (recommended):
   ```sh
   python3.9 -m venv mhps_env
   source mhps_env/bin/activate
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```


## Usage
To train the MHPS model, use the provided `run.py` script:

```sh
python run.py ---data=christianity
```




### Parameters
Customize the training process by modifying arguments in 'graphEncoder.py' or 'run.py'. Key parameters include:
- **Dataset selection**: Twitter, Christianity, Android, Douban
- **Batch size**, **learning rate**, and other hyperparameters
- **LightGCN layer number**: Adjustable in graphEncoder.py (line 101).
                            Use 6 layers for Christianity
                            Use 4 layers for Android
                            Use 3 layers for Douban and Twitter

## Model Architecture
MHPS introduces several novel components:

- **Graph Encoder**: Accurately captures user representations through a multi-granularity hypergraph structure combined with LightGCN layers, effectively modeling complex social relationships and multi-level interaction dependencies.
- **Sequence Model**: Captures contextual relationships from multiple perspectives using source attention, self-attention, and next-step position-aware attention, enabling the model to jointly learn temporal dependencies and user interaction dynamics.
## Datasets
- **christianity**
- **Twitter**
- **Android**
- **Douban**


The datasets are preprocessed using `DataConstruct.py`, which handles feature extraction, normalization, and graph construction.


## Evaluation
MHPS has been evaluated using several key metrics to measure prediction accuracy and generalization:
- **Hits@K**, **Map@K** are used to evaluate the model’s prediction accuracy, where K is set to 10, 50, and 100.


## Results
MHPS significantly outperforms existing baseline methods on the christianity, twitter, douban, and android datasets in predicting information diffusion accuracy.


## License
This project is licensed under the MIT License.

## Citation
If you use MHPS in your research, please cite our paper:

```
@article{CIKM2025,
  title={Enhancing Information Diffusion Prediction via Multiple Granularity Hypergraphs and Position-aware Sequence Model},
  author={Weikai Jing, Yuchen Wang,Haotong Du, Songxin Wang,Xiaoyu Li and Chao Gao},
  journal={In Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  year={2025}
}
```


