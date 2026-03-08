# MMIDDI: Multimodal Mutual Influence Framework for Drug–drug Interaction Under Drug Cold Start 

![]([https://github.com/drunkprogrammer/MMIDDI/blob/Figure1.jpg)

## Contents

- [Overiew](#overview)
- [Obtain Datasets](#obtain-datasets)
- [Enviornment](#docking-and-scoring)
- [Run Command](#post-processing)

## Overview

Accurate prediction of drug-drug interactions (DDIs) is crucial for safe drug administration and essential to pharmacovigilance. Traditional drug-drug prediction costs huge time and financial resources. In recent years, deep learning has facilitated the prediction of drug-drug interactions. However, most deep learning models focus on drug molecular structure and consider the crucial pharmacological modalities (such as targets and enzymes) independently. Moreover, standard evaluation paradigms often neglect the critical scenarios of single-drug and double-drug cold start, which are essential for real-world applicability. Consequently, predictions are compromised not only by the neglect of multimodal drug-drug interactions but also by dataset shortcomings that cause biased evaluation and muddle the core tasks. To address these challenges, we propose the Multimodal Mutual Influence Drug-Drug Interaction (MMIDDI) framework and contribute a new dataset with a novel label-aware and drug-aware partitioning algorithm. MMIDDI captures and fuses multimodal mutual influences for both intra-drug and inter-drug multimodal mutual influences. The intra-drug multimodal mutual influences include intra-modal dependencies (within a single modality) and inter-modal interactions (across different modalities). The inter-drug multimodal mutual influences represent the bidirectional drug pair influences, which mutual influences cross the modalities of each drug. These comprehensively mined and fused mutual influences are then leveraged by the multi-type DDI predictor to accurately predict the interaction type. Extensive experiments on three tasks (standard, single-drug cold start, and double-drug cold start) demonstrate that MMIDDI significantly outperforms baseline models on key metrics (accuracy, macro precision, and macro recall) while providing deeper insights into pharmacology mechanisms.

## Obtain Datasets

The datasets and the prepared benchmarking dataset are available at https://zenodo.org/records/18907274.

## Enviornment

```
conda create -n MMIDDI
conda activate MMIDDI
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


#### Run Command
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
