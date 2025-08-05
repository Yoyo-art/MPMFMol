import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple
import numpy as np

from chemprop.train.run_training import run_training
# from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.torchlight import initialize_exp


def run_stat(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    save_dir = args.save_dir
    # task_names = get_task_names(args.data_path)
    info(f'Run scaffold {args.runs}')
    args.save_dir = os.path.join(save_dir, f'run_{args.seed}')
    makedirs(args.save_dir)
    model_scores = run_training(args, args.pretrain, logger)
    info(f'{args.runs}-times runs')
    info(f'Scaffold {args.runs} ==> test {args.metric} = {model_scores:.6f}')

    return model_scores


if __name__ == '__main__':
    dataset_configs = {
        "bace": {"metric": "auc", "type": "classification"},
        "bbbp": {"metric": "auc", "type": "classification"},
        "clintox": {"metric": "auc", "type": "classification"},
        "esol": {"metric": "rmse", "type": "regression"},
        "freesolv": {"metric": "rmse", "type": "regression"},
        "lipo": {"metric": "rmse", "type": "regression"},
        # "sider": {"metric": "auc", "type": "classification"},
        "tox21": {"metric": "auc", "type": "classification"},
        "toxcast": {"metric": "auc", "type": "classification"},
    }
    for dataset, config in dataset_configs.items():
        args = parse_train_args()
        args.dataset = dataset
        args.exp_id = dataset
        args.root_path = "./data"
        args.data_path = os.path.join(args.root_path, f"{dataset}.csv")
        args.metric = config["metric"]
        args.dataset_type = config["type"]
        args.split_type = "scaffold_balanced"  # scaffold_balanced
        args.runs = 2
        args.encoder = False
        args.exp_name = "finetune"
        args.checkpoint_path = "./ckpt/original_MoleculeModel_0727_0817_55th_epoch.pkl"
        args.gpu = 0
        args.epochs = 100
        args.pretrain = False
        args.atom_messages =  True
        args.smi_encoder = True
        args.increase_parm = 1
        args.init_lr = 1e-4
        args.max_lr = 1e-3
        args.final_lr = 1e-4
        args.warmup_epochs =2
        args.hidden_size = 300
        args.ffn_hidden_size = 300
        args.add_reactive = False
        args.add_step = 'concat_mol_frag_attention'
        args.step = 'func_prompt'
        args.early_stop = False
        args.patience = 30
        args.last_early_stop = 0
        args.batch_size = 256
        args.ffn_num_layers = 2
        args.encoder_name = "CMPNN"
        args.dropout = 0.1
        args.l2_norm = 0
        args.depth = 3
        args.num_attention = 2
        modify_train_args(args)
        logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
        model_scores = run_stat(args, logger)
        print(f'Scaffold-{args.runs} Results: {model_scores:.5f}')

    # args = parse_train_args()
    # args.data_path = "./data/bbbp.csv"
    # args.dataset = "bbbp"
    # args.exp_id = "bbbp"
    # args.root_path = "./data"
    # args.metric = "auc"#rmse,auc
    # args.dataset_type = "classification"#regression,classification
    # args.split_type = "scaffold_balanced"# scaffold_balanced
    # args.runs = 2
    # args.encoder = False
    # args.exp_name = "finetune"
    # # args.checkpoint_path = "./ckpt/original_MoleculeModel.pkl"
    # args.gpu = 0
    # args.epochs = 100
    # args.pretrain = False
    # args.atom_messages =  True
    # args.smi_encoder = True
    # args.increase_parm = 1
    # args.init_lr = 1e-4
    # args.max_lr = 1e-3
    # args.final_lr = 1e-4
    # args.warmup_epochs =2
    # args.hidden_size = 300
    # args.ffn_hidden_size = 300
    # args.add_reactive = False
    # args.add_step = 'concat_mol_frag_attention'
    # args.step = 'func_prompt'
    # args.early_stop = False
    # args.patience = 30
    # args.last_early_stop = 0
    # args.batch_size = 256
    # args.ffn_num_layers = 2
    # args.encoder_name = "CMPNN"
    # args.dropout = 0.1
    # args.l2_norm = 0
    # args.depth = 3
    # args.num_attention = 2
    # modify_train_args(args)
    # logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    # model_scores = run_stat(args, logger)
    # print(f'Scaffold-{args.runs} Results: {model_scores:.5f}')
