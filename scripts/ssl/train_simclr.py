# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml
import torch
from archai.networks_ssl.simclr import ModelSimCLR
from archai.common.trainer_ssl import TrainerSimClr
from archai.common.config import Config
from archai.common.common import common_init
from archai.datasets import data
from archai.common.checkpoint import CheckPoint

def train_test(conf_eval:Config):
    conf_loader = conf_eval['loader']
    conf_trainer = conf_eval['trainer']
    conf_dataset = conf_loader['dataset']
    conf_checkpoint = conf['common']['checkpoint']
    with open('confs/algos/simclr_resnets.yaml', 'r') as f:
        conf_models = yaml.load(f, Loader=yaml.Loader)
    conf_model = conf_models[conf_trainer['model']]

    # create model
    model = ModelSimCLR(conf_dataset['name'], conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
        conf_model['compress'], conf_model['hidden_dim'], conf_model['out_features'], groups = conf_model['groups'],
        width_per_group=conf_model['width_per_group'])
    model = model.to(torch.device('cuda', 0))
    print('Number of trainable params: {:.2f}M'
          .format(sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6))
    # print(conf_dataset['name'])
    # exit()
    # get data
    data_loaders = data.get_data_ssl(conf_loader)

    # train!
    ckpt = CheckPoint(conf_checkpoint, load_existing=False)
    trainer = TrainerSimClr(conf_trainer, model, ckpt)
    trainer.fit(data_loaders)


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/algos/simclr.yaml')
    

    train_test(conf)


