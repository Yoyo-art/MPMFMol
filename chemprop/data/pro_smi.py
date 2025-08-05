import json
import re
from rdkit import Chem
import torch

# 加载词汇表
with open('./chemprop/data/vocab.json', 'r') as f:
    vocab = json.load(f)

# 反转词汇表（ID -> token）
id2token = {v: k for k, v in vocab.items()}

# 需要特殊处理的元素（两个字母的元素符号）
double_letter_elements = {'Br', 'Cl', 'Si', 'Se', 'As', 'Te', 'Li', 'Na', 'K', 'Rb',
                          'Cs', 'Fr', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra', 'Sc', 'Ti',
                          'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
                          'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                          'In', 'Sn', 'Sb', 'I', 'Xe', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                          'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                          'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                          'Pb', 'Bi', 'Po', 'At', 'Rn'}
def standardize_smiles(smiles):
    """标准化 SMILES：确保两个字母的元素符号（如 Cl, Br）用 [] 包裹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    atom_list = []
    atom_index = []
    for atom in mol.GetAtoms():
        atom_list.append(atom.GetSymbol())
        atom_index.append(atom.GetIdx())
    present_elements = set(double_letter_elements).intersection(set(atom_list))
    # 1. 使用正则表达式找出所有被[]包起来的部分
    bracketed_tokens = re.findall(r'\[(.*?)\]', smiles)
    # 2. 加上[]还原完整token形式
    full_tokens = [f"[{token}]" for token in bracketed_tokens]
    ele = list(present_elements) + full_tokens
    return smiles, ele, atom_list


def tokenize_smiles(smiles):
    """将单个SMILES字符串转换为token ID序列"""
    # 1. 标准化SMILES并提取需要特殊处理的元素
    standardized, ele, _ = standardize_smiles(smiles)

    # 2. 特殊处理双字母元素（确保已被[]包裹）
    tokens = []
    i = 0
    n = len(standardized)

    while i < n:
        # 处理[]包裹的token（如[Cl]、[NH+]等）
        if standardized[i] == '[':
            j = standardized.find(']', i)
            if j == -1:
                raise ValueError(f"Unmatched '[' in SMILES: {standardized}")
            token = standardized[i:j + 1]
            tokens.append(token)
            i = j + 1
        # 处理普通字符
        else:
            # 检查是否是双字母元素（如Cl、Br等）
            found = False
            for elem in double_letter_elements:
                if standardized.startswith(elem, i):
                    tokens.append(f"[{elem}]")
                    i += len(elem)
                    found = True
                    break
            if not found:
                tokens.append(standardized[i])
                i += 1

    # 3. 转换为ID序列
    ids = []
    for token in tokens:
        # 查找token在词汇表中的ID，若不存在则使用UNK
        if token in vocab:
            ids.append(vocab[token])
        elif token.upper() in vocab:  # 尝试大写形式
            ids.append(vocab[token.upper()])
        elif token.lower() in vocab:  # 尝试小写形式
            ids.append(vocab[token.lower()])
        else:
            ids.append(vocab['[UNK]'])  # 未知token

    return ids


def pad_sequences(sequences, max_len=None, pad_value=0):
    """填充序列至批次最大长度"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [pad_value] * (max_len - len(seq)))
        else:
            padded.append(seq[:max_len])

    return torch.LongTensor(padded)


def batch_encode_smiles(smiles_list):
    """批量编码SMILES为填充后的张量"""
    cls_token = '[CLS]'
    cls_id = vocab[cls_token]

    # 1. 对每个SMILES进行tokenize
    tokenized = [tokenize_smiles(smiles) for smiles in smiles_list]

    # 2. 计算原始SMILES的最大长度（不含CLS）
    max_len = max(len(seq) for seq in tokenized)

    # 3. 初始化结果张量（注意长度+1是为了CLS token）
    batch_size = len(tokenized)
    encoded_tensor = torch.zeros((batch_size, max_len + 1), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len + 1), dtype=torch.bool)

    # 4. 填充数据
    for i, seq in enumerate(tokenized):
        # 在最前面添加CLS token
        encoded_tensor[i, 0] = cls_id
        encoded_tensor[i, 1:len(seq) + 1] = torch.tensor(seq)

        # 设置attention mask（CLS位置和有效token位置为True）
        attention_mask[i, 0] = True  # CLS token位置
        attention_mask[i, 1:len(seq) + 1] = True  # 有效token位置

    return encoded_tensor, attention_mask, max_len


# 示例用法
if __name__ == "__main__":
    # 示例批次SMILES
    batch_smiles = [
        "CCO",  # 乙醇
        "C=O",  # 甲醛
        "ClC(Cl)(Cl)Cl",  # 四氯化碳
        "[Na+].[Cl-]",  # 氯化钠
        "C1=CC=CC=C1"  # 苯
    ]

    # 编码并填充
    encoded_batch, a, _ = batch_encode_smiles(batch_smiles)

    print("Encoded and padded batch:")
    print(encoded_batch)
    print("Shape:", encoded_batch.shape)