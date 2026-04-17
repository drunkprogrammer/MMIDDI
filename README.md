# MMI-DDI: Multimodal Mutual Influence Framework for Drug–Drug Interaction Prediction Under Drug Cold Start 

![](https://github.com/drunkprogrammer/MMI-DDI/blob/main/Framework.jpg)

## Contents

- [Overiew](#overview)
- [Obtain Datasets](#obtain-datasets)
- [Enviornment](#docking-and-scoring)
- [Run Command](#post-processing)

## Overview

Accurate prediction of drug-drug interactions (DDIs) is essential for safe polypharmacy and pharmacovigilance, yet traditional experimental approaches remain resource-intensive. While deep learning has advanced DDI prediction, existing frameworks face two fundamental limitations: (1) they predominantly focus on molecular structures while treating other critical pharmacological modalities (e.g., targets, enzymes) as independent features, neglecting the asymmetric mutual influences both within a single drug and between drugs; and (2) prevalent dataset partitioning strategies introduce evaluation bias by failing to distinguish drug cold start from label cold start, thereby obscuring true model generalizability. To address these challenges, we propose the Multimodal Mutual Influence Drug-Drug Interaction (MMI-DDI) framework. MMI-DDI explicitly mines and fuses asymmetric multimodal mutual influences at two levels: within a single drug (capturing intra-modal dependencies and inter-modal interactions within one drug) and between drugs (modeling bidirectional asymmetric influence propagation across modalities between drug pairs). These comprehensively derived representations are then fed into a custom-designed multi-type DDI predictor. In addition, we construct a new benchmark dataset and develop a label-aware and drug-aware partitioning algorithm to enable rigorous evaluation under both standard and cold-start scenarios. Extensive experiments on three tasks (standard prediction, single-drug cold start, and double-drug cold start) across four datasets—including three randomly split datasets and our dataset constructed using the automated partitioning algorithm —demonstrate that MMI-DDI consistently outperforms representative baselines across key metrics such as accuracy, macro precision, and macro recall. Furthermore, case studies provide interpretable pharmacological insights, while error analysis identifies directions for future improvement.

## Obtain Datasets

The datasets and the prepared benchmarking dataset are available at https://zenodo.org/records/18907274.

## Enviornment

```
conda create -n MMI-DDI
conda activate MMI-DDI
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pandas
conda install seaborn
conda install conda-forge::curl
conda install -c conda-forge spacy
conda install -c conda-forge cupy
pip install requests
pip install openpyxl --upgrade
pip install transformers
pip install torch_geometric
pip install scikit-learn
pip install scikit-fingerprints
pip install matplotlib
pip3 install chardet
python -m spacy download en_core_web_sm
```


## Run Command
**Standard Task 1 (Event_special Dataset)**
```
python main.py -d event_special -t task1 -e 100 -o 0.3 -a 0.0
```


**Single-drug Cold Start Task 2 (Event_special Dataset)**
```
python main.py -d event_special -t task2_3 -e 100 -o 0.3 -a 0.2
```

**Double-drug Cold Start Task 3 (Event_special Dataset)**
```
python main.py -d event_special -t task2_3 -e 100 -o 0.2 -a 0.2
```
