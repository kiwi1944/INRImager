import torch
import os
import logging
import json
import numpy as np
import random


def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_dirs(config):
    if not os.path.exists(config.logs_dir) and config.wandb_flag:
        os.makedirs(config.logs_dir)
    if not os.path.exists(config.ckpt_dir) and config.wandb_flag:
        os.makedirs(config.ckpt_dir)


def save_config(config, filename):
    if config.wandb_flag:
        device = config.device
        config.device = str(config.device)
        with open(filename, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        config.device = device


def create_grid(h, w):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h), torch.linspace(0, 1, steps=w)], indexing="ij")
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid


def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, ckpt_dir, B, save_flag):
    para_dict = {}
    para_dict['epoch'] = epoch
    para_dict['model_state_dict'] = model_state_dict
    para_dict['optimizer_state_dict'] = optimizer_state_dict
    para_dict['B'] = B
    if save_flag == 1:
        ckpt_path = os.path.join(ckpt_dir, 'epoch{}_ckpt.pth'.format(epoch))
        torch.save(para_dict, ckpt_path)
        logging.info(f'Checkpoint {epoch} saved!')
    elif save_flag == 2:
        ckpt_path = os.path.join(ckpt_dir, 'model_best.pth')
        torch.save(para_dict, ckpt_path)


def load_config(args, config_path):
    with open(config_path) as f:
        config_from_json = json.load(fp=f)
    args_as_dict = vars(args)
    args_as_dict.update(config_from_json)
    return args

