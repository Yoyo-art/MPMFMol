import csv

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm


def generate_fingerprints(smiles_list, fp_type='maccs'):
    """生成分子指纹（MACCS或PubChem）"""
    fingerprints = []
    valid_indices = []

    for idx, smi in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Generating fingerprints"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            if fp_type == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif fp_type == 'pubchem':
                fp = AllChem.GetPubChemFingerprint(mol)
            fingerprints.append(fp)
            valid_indices.append(idx)

    return np.array(fingerprints), valid_indices


def cluster_and_label(fingerprints, k_values=[100, 1000, 10000]):
    """对指纹进行K均值聚类并生成标签"""
    labels = {}
    for k in tqdm(enumerate(k_values), total=len(k_values), desc="Generating labels"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels[f'label_k{k}'] = kmeans.fit_predict(fingerprints)
    return labels


def main(input_smiles_file, output_csv_file, fp_type='maccs'):
    # 读取SMILES文件（假设每行一个SMILES）
    with open(input_smiles_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        smiles_list = [line[0] for line in reader]

    # 生成指纹
    fingerprints, valid_indices = generate_fingerprints(smiles_list, fp_type)
    valid_smiles = [smiles_list[i] for i in valid_indices]

    # 聚类并生成标签
    labels_dict = cluster_and_label(fingerprints)

    # 构建结果DataFrame
    results = pd.DataFrame({'smiles': valid_smiles})
    for k, labels in labels_dict.items():
        results[k] = labels

    # 保存结果
    results.to_csv(output_csv_file, index=False)
    print(f"结果已保存至 {output_csv_file}，有效分子数：{len(valid_smiles)}")


if __name__ == "__main__":
    # 示例用法
    input_file = "../../data/zinc15_250K.csv"  # 输入SMILES文件路径
    output_file = "../../data/zinc15_250K_labels.csv"  # 输出CSV文件路径

    # 选择指纹类型：'maccs' 或 'pubchem'
    fingerprint_type = 'maccs'

    main(input_file, output_file, fingerprint_type)