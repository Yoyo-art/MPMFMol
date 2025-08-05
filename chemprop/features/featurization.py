import os
from argparse import Namespace
from typing import List, Tuple, Union

import dgl
from rdkit import Chem
import torch
import numpy as np
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.BRICS import BreakBRICSBonds, FindBRICSBonds
from torch import nn, LongTensor, FloatTensor

# Atom feature sizes
MAX_ATOMIC_NUM = 110
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}
PHARM_FEATURE_SIZE= 182
declist = Descriptors.descList
calc = {}
for (i,j) in declist:
    calc[i] = j

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}
REACT_FDIM = 34
def get_react_fdim() -> int:
    return REACT_FDIM

def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim() -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim() -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str, args: Namespace):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.bonds = []
        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()

        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)

                if args.atom_messages:
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                else:
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2
                self.bonds.append(np.array([a1, a2]))
        # rectify a2b

class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim() + (not args.atom_messages) * self.atom_fdim  # * 2

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        bonds = [[0, 0]]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])  # if b!=-1 else 0

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
                bonds.append([b2a[-1],
                              self.n_atoms + mol_graph.b2a[mol_graph.b2revb[b]]])
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        bonds = np.array(bonds).transpose(1, 0)

        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor(
            [a2b[a][:self.max_num_bonds] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.bonds = torch.LongTensor(bonds)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> tuple[
        FloatTensor, FloatTensor, LongTensor, LongTensor, LongTensor, list[tuple[int, int]], list[
            tuple[int, int]], LongTensor]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.bonds

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(smiles_batch: List[str],
              args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_GRAPH:
            mol_graph = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles, args)
            if not args.no_cache:
                SMILES_TO_GRAPH[smiles] = mol_graph
        mol_graphs.append(mol_graph)

    return BatchMolGraph(mol_graphs, args)


CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, "*": 65}
def smiles_to_tensor(smiles_batch: List[str]) -> (torch.Tensor, torch.Tensor):
    """
    将SMILES批次转换为填充后的索引Tensor和掩码

    参数:
        smiles_batch: SMILES字符串列表
        char_to_idx: 字符到索引的映射字典

    返回:
        (encoded_tensor, mask_tensor)
        encoded_tensor: 形状为 (batch_size, max_len) 的LongTensor
        mask_tensor: 形状为 (batch_size, max_len) 的BoolTensor (1表示有效字符)
    """
    # 1. 计算最大长度
    max_len = max(len(smiles) for smiles in smiles_batch)

    # 2. 初始化结果张量
    batch_size = len(smiles_batch)
    encoded_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask_tensor = torch.zeros((batch_size, max_len), dtype=torch.bool)

    # 3. 填充每个SMILES
    for i, smiles in enumerate(smiles_batch):
        # 字符级编码
        # encoded = [CHARISOSMISET[char] for char in smiles]
        encoded_tensor[i, :len(smiles)] = torch.tensor([CHARISOSMISET[char] for char in smiles])
        mask_tensor[i, :len(smiles)] = True  # 有效位置标记为True

    # cls_id = 65
    #
    # # 创建 shape 为 [256, 1] 的 CLS token 矩阵
    # cls_column = torch.full((batch_size, 1), cls_id, dtype=encoded_tensor.dtype, device=encoded_tensor.device)
    #
    # # 拼接在前面
    # output_tensor = torch.cat([cls_column, encoded_tensor], dim=1)
    #
    # if mask_tensor.dim() == 2:
    #     attention_mask = mask_tensor.unsqueeze(1)  # [B, L] → [B, 1, L]
    #
    # cls_mask = torch.ones((batch_size, 1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
    # new_attention_mask = torch.cat([cls_mask, attention_mask], dim=2)  # [B, 1, L+1]

    # return encoded_tensor, attention_mask.squeeze(1), max_len
    return encoded_tensor, mask_tensor, max_len

func_path = os.path.abspath("./chemprop/data/funcgroup.txt")
with open(func_path, "r") as f:
    funcgroups = f.read().strip().split('\n')
    name = [i.split()[0] for i in funcgroups]
    smart = [Chem.MolFromSmarts(i.split()[1]) for i in funcgroups]#	list of Mol，SMARTS 对象列表
    smart2name = dict(zip(smart, name))#字典key:mol对象，value：名称
    func2index = {smart2name[sm]:i for i,sm in enumerate(smart)}#字典key:官能团名字，value：ID

def match_group(mol):
    mapping = []#包含的功能团编号
    func2atom,mathch_smart2name = [],[]#匹配到的原子索引，匹配到的功能团名字
    for sm in smart:
        if mol.HasSubstructMatch(sm):
            atom_indices = mol.GetSubstructMatch(sm)
            mapping.extend([func2index[smart2name[sm]]])
            for atom_lst in [atom_indices]:
                atom_l = [x+1 for x in list(atom_lst)]
                if len(atom_l)>15:#官能团包含的原子个数为15
                    atom_l = atom_l[:15]
                else:
                    atom_l = atom_l+[0]*(15-len(atom_l))
                func2atom.extend([atom_l])
                mathch_smart2name.append(smart2name[sm])
    return mapping,func2atom,mathch_smart2name

def smiles_batch_to_func(smiles_batch: List[str]) -> tuple[
    list[tuple[int, int]], list[tuple[int, int]], LongTensor, list[tuple[int, int]], LongTensor]:
    """
    处理一个SMILES批次，提取功能团嵌入及索引信息

    参数：
        smiles_batch: SMILES字符串列表

    返回：
        a_scope: 原子起始范围（每个分子）
        mapping_scope: 功能团编号范围（每个分子）
        func2atom: 所有功能团到原子索引的映射（固定长度15）
        func2atom_scope: 每个功能团在func2atom中的范围
    """
    a_scope = []            # 每个分子的原子范围 (start_idx, length)
    mapping_scope = []      # 每个分子的功能团编号范围
    func2atom = []          # 所有官能团到原子的映射
    func2atom_scope = []    # 每个功能团对应在func2atom中的位置
    all_mapping = []
    total_atoms = 1         #number of atoms (start at 1 b/c need index 0 as padding)
    total_groups = 0
    total_f2a = 0

    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")

        num_atoms = mol.GetNumAtoms()
        a_scope.append((total_atoms, num_atoms))
        total_atoms += num_atoms

        # 调用你定义的 match_group 方法
        mapping, f2a, _ = match_group(mol)

        # 累积 mapping & func2atom
        all_mapping.extend(mapping)
        func2atom.extend(f2a)

        num_groups = len(mapping)
        mapping_scope.append((total_groups, num_groups))
        n_func2atom = len(f2a)
        func2atom_scope.append((total_f2a,n_func2atom))  # 每个官能团1个列表
        total_f2a += n_func2atom
        total_groups += num_groups
    all_mapping = torch.LongTensor(all_mapping)
    func2atom = torch.LongTensor(func2atom)

    return a_scope, mapping_scope, func2atom, func2atom_scope, all_mapping

def get_PharmElement(mol_pharm):
    atom_symbol={'C':0,'H':0,'O':0,'N':0,'P':0,
                 'S':0,'F':0,'CL':0,'Br':0,'other':0,}
    feat_pharm_element =[]
    # [C,H,O,N,P,S,F,CL,Br,other]
    for atom in mol_pharm.GetAtoms():
        if atom.GetSymbol() in atom_symbol.keys():
            atom_symbol[atom.GetSymbol()]+=1
        else:
            atom_symbol['other']+=1
    for key,value in atom_symbol.items():
        feat_pharm_element+=[value]
    return feat_pharm_element

def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))

def get_pharm_fdim() -> int:
    return PHARM_FEATURE_SIZE

def GetBRICSFeature(mol_pharm):
    try:
        pharm_feat= [calc['TPSA'](mol_pharm)*0.01]+[calc['MolLogP'](mol_pharm)]+\
                                [calc['HeavyAtomMolWt'](mol_pharm)*0.01]+[1 if mol_pharm.GetRingInfo().NumRings()>0 else 0]+\
                                [mol_pharm.GetRingInfo().NumRings()]+\
                                get_PharmElement(mol_pharm)+maccskeys_emb(mol_pharm)
    except:
        pharm_feat = [0]*get_pharm_fdim()
    return pharm_feat

def brics_features(mol, pretrain=False):
    fragsmiles = [Chem.MolToSmiles(x, True) for x in Chem.GetMolFrags(BreakBRICSBonds(mol), asMols=True)]
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0], i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol, break_bonds, addDummies=False)
    frags_idx_lst = Chem.GetMolFrags(tmp)
    pharm_feats = {}
    atom2pharmid = {}
    for idx, frag_idx in enumerate(frags_idx_lst):
        for atom_idx in frag_idx:
            atom2pharmid[atom_idx] = idx
        try:
            frag_pharm = fragsmiles[idx]
            mol_pharm = Chem.MolFromSmiles(frag_pharm)
            pharm_feat = GetBRICSFeature(mol_pharm)
        except:
            print(f'generate Pharm feature make a error in {Chem.MolToSmiles(mol)}')
            pharm_feat = [0] * PHARM_FEATURE_SIZE
        pharm_feats[idx] = pharm_feat

    return pharm_feats, atom2pharmid, frags_idx_lst

def GetBricsBonds(mol):
    bonds_tmp = FindBRICSBonds(mol)
    bonds = [b for b in bonds_tmp]
    result = {}
    for item in bonds:# item[0] is atom, item[1] is brics type
        result.update({(int(item[0][0]), int(item[0][1])):[int(item[1][0]), int(item[1][1])]})
        result.update({(int(item[0][1]), int(item[0][0])):[int(item[1][1]), int(item[1][0])]})
    return result

def GetBRICSBondFeature_Hetero(react_1,react_2):
    result = []
    start_action_bond = int(react_1) if (react_1 !='7a' and react_1 !='7b') else 7
    end_action_bond = int(react_2) if (react_2 !='7a' and react_2 !='7b') else 7
    # return react_features(start_action_bond,end_action_bond)
    emb_0 = [0 for i in range(17)]
    emb_1 = [0 for i in range(17)]
    emb_0[start_action_bond] = 1
    emb_1[end_action_bond] = 1
    result = emb_0 + emb_1
    return result

def Mol2HeteroGraph(mol, args: Namespace):
    # build graphs
    edge_types = [('a', 'b', 'a'), ('p', 'r', 'p'), ('a', 'j', 'p'), ('p', 'j', 'a')]
    pharm_feats, atom2pharmid, frags_idx_lst = brics_features(mol, pretrain=False)
    result = GetBricsBonds(mol)
    edges = {k: [] for k in edge_types}
    for bond in mol.GetBonds():
        edges[('a', 'b', 'a')].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges[('a', 'b', 'a')].append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    for (a1, a2) in result.keys():
        edges[('p', 'r', 'p')].append([atom2pharmid[a1], atom2pharmid[a2]])

    for k, v in atom2pharmid.items():
        edges[('a', 'j', 'p')].append([k, v])
        edges[('p', 'j', 'a')].append([v, k])
    g = dgl.heterograph(edges)
    # atom view
    f_atom = []
    for idx in g.nodes('a'):
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))

    f_bond = []
    src, dst = g.edges(etype=('a', 'b', 'a'))
    for i in range(g.num_edges(etype=('a', 'b', 'a'))):
        if not args.atom_messages:
            f_bond.append(f_atom[src[i].item()] + bond_features(mol.GetBondBetweenAtoms(src[i].item(), dst[i].item())))
        else:
            f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(), dst[i].item())))
    g.edges[('a', 'b', 'a')].data['x'] = torch.FloatTensor(f_bond)
    f_atom = torch.FloatTensor(f_atom)
    g.nodes['a'].data['f'] = f_atom
    dim_atom = len(f_atom[0])

    # pharm view
    f_pharm = []
    for k, v in pharm_feats.items():
        f_pharm.append(v)

    f_reac = []
    src, dst = g.edges(etype=('p', 'r', 'p'))
    for idx in range(g.num_edges(etype=('p', 'r', 'p'))):
        p1 = src[idx].item()
        p2 = dst[idx].item()
        for k, v in result.items():
            if p1 == atom2pharmid[k[0]] and p2 == atom2pharmid[k[1]]:
                if not args.atom_messages:
                    f_reac.append(f_pharm[p1] + GetBRICSBondFeature_Hetero(v[0], v[1]))
                else:
                    f_reac.append(GetBRICSBondFeature_Hetero(v[0], v[1]))

    g.edges[('p', 'r', 'p')].data['x'] = torch.FloatTensor(f_reac)
    g.nodes['p'].data['f'] = torch.FloatTensor(f_pharm)
    dim_pharm = len(f_pharm[0])

    dim_atom_padding = g.nodes['a'].data['f'].size()[0]  # 原子个数
    dim_pharm_padding = g.nodes['p'].data['f'].size()[0]  # 药效团个数
    # junction view
    g.nodes['a'].data['f_junc'] = torch.cat([g.nodes['a'].data['f'], torch.zeros(dim_atom_padding, dim_pharm)], 1)
    g.nodes['p'].data['f_junc'] = torch.cat([torch.zeros(dim_pharm_padding, dim_atom), g.nodes['p'].data['f']], 1)

    return g

def create_dgl_batch(smiles_batch: List[str], args: Namespace = None, pretrain: bool = None):
    graph,smiles = [],[]
    for smile in smiles_batch:
        mol = Chem.MolFromSmiles(smile)
        g = Mol2HeteroGraph(mol,args)
        graph.append(g)
        smiles.append(smile)
    return dgl.batch(graph),smiles