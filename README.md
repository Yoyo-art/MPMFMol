# MPMFMol

## 🚀 Getting Started

### 1️⃣ Data Preprocessing

Generate labeled data from raw SMILES

```bash
python chemprop/data/data_process.py
```

This will output a file:
zinc15_250K_labels.csv

### 2️⃣ Pretraining
Train the molecular encoder in a self-supervised fashion:

```bash
python pretrain.py
```
This will save the pretrained model checkpoint as:

original_MoleculeModel.pkl 
### 3️⃣ Downstream Prediction
Use the pretrained model for downstream prediction tasks:

```bash
python predict.py
```
