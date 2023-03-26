# fairness-aware-classification
Skoltech ML Course project "Boosting for Fairness-Aware Classification"

The list of compared methods:

| Method  | Description |
| ------------- | ------------- |
| SMOTE + LogisticRegression  | [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813) |
| SMOTE + RandomForest | [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813) |
| SMOTEBoost  | [SMOTEBoost: Improving Prediction of the Minority Class in Boosting](https://link.springer.com/chapter/10.1007/978-3-540-39804-2_12)  |
| RUSBoost  | [RUSBoost: A Hybrid Approach to Alleviating Class Imbalance](https://ieeexplore.ieee.org/document/5299216)  |
| AdaBoost  | [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://www.sciencedirect.com/science/article/pii/S002200009791504X)  |
| AdaFair  | [AdaFair: Cumulative Fairness Adaptive Boosting](https://arxiv.org/abs/1909.08982)  |
| AttentiveGradientBoosting | Our novel approach |

# How to run

1. Firstly you need to download all datasets. To do it you should just run script `download.sh` from working directory. Note, that `kaggle` `pip` library should be installed to download some datasets. As a result four datasets will be downloaded and folders `adult`, `bank`, `compass`, `kdd` will be created.
2. To run experiments you can just run notebook `experiments.ipynb`. As a result plots will be saved in folder `plots`.

# Description of the structure

- `download.sh`: script for downloading datasets
- `download.ipynb`: notebook for downloading datasets (similar to script)
- `experiments.ipynb`: notebook to run experiments
- `algo.py`: implementation of all algorithms are written here
- `utils.py`: functions for loading and preprocessing datasets are written here
- `metrics.py`: implementations of all required metrics are written here

- `presentation.pdf`: presentation of the project
- `report.pdf`: report of the project
- `plots`: folder with plots created after the experiment
