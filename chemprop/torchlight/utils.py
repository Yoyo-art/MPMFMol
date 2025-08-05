import os
from datetime import datetime

import torch
from torch.nn .parallel import DataParallel, DistributedDataParallel


def get_code_version(short_sha=True):
    from subprocess import check_output, STDOUT, CalledProcessError
    try:
        sha = check_output('git rev-parse HEAD', stderr=STDOUT,
                         shell=True, encoding='utf-8')
        if short_sha:
            sha = sha[:7]
        return sha
    except CalledProcessError:
        # There was an error - command exited with non-zero code
        pwd = check_output('pwd', stderr=STDOUT, shell=True, encoding='utf-8')
        pwd = os.path.abspath(pwd).strip()
        print(f'Working dir {pwd} is not a git repo.')



def snapshot(model, epoch, save_path, name):
    """
    Saving model w/ its params.
        Get rid of the ONNX Protocal.
    F-string feature new in Python 3.6+ is used.
    """
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    save_path = os.path.join(save_path, f'{name}_{type(model).__name__}_{timestamp}_{epoch}th_epoch.pkl')
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)