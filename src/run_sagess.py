import os

from pytorch_lightning import Trainer

import wandb
import torch
import utils

from analysis.spectre_utils import LargeGraphSamplingMetrics
from datasets.large_graph_datasets import LargeGraphModule,LargeGraphDatasetInfos
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from analysis.visualization import NonMolecularVisualization
from diffusion.extra_features import DummyExtraFeatures
from pytorch_lightning.callbacks import ModelCheckpoint
import networkx as nx

import hydra
import omegaconf
from omegaconf import DictConfig

torch.manual_seed(120)
def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = cfg.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': 'sagess', 'project': f'sagess_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=False), 'reinit': True, 
              'mode': cfg.general.wandb,
            #   , 'entity': ''
              }
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(version_base='1.1', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    datamodule = LargeGraphModule(cfg)
    sampling_metrics = LargeGraphSamplingMetrics(datamodule.dataloaders)

    dataset_infos = LargeGraphDatasetInfos(datamodule, dataset_config)
    train_metrics = TrainAbstractMetricsDiscrete()
    visualization_tools = NonMolecularVisualization()
    extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)
    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics, 'visualization_tools': visualization_tools,
                    'sampling_metrics': sampling_metrics, 'extra_features': extra_features, 'domain_features': domain_features }
    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)
    print(cfg)

    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)

    name = cfg.general.name
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      devices=cfg.general.gpus if torch.cuda.is_available() and cfg.general.gpus > 0 else None,
                      limit_train_batches=20 if name == 'test' else None,
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      logger=[])

    #trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
    
    
if __name__ == '__main__':
    main()
