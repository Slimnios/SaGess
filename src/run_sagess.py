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
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 
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
    device = torch.device('cuda')
    model.to(device)
    dir = cfg.dataset.directed if hasattr(cfg.dataset, 'directed') else False

    for i in range(10):
        torch.cuda.empty_cache()
        samples_left_to_generate = cfg.general.final_model_samples_to_generate
        samples_left_to_save = 1    #cfg.general.final_model_samples_to_save
        number_chain_steps = cfg.general.number_chain_steps
        chains_left_to_save = cfg.general.final_model_chains_to_save
        samples = []
        id = 0
        edge_list_samples = []
        number_of_edges = []
        n_edges = 0

        while samples_left_to_generate > 0:# or n_edges < 5:
            print(f'Samples left to generate: {samples_left_to_generate}/ '
                  f'{cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = cfg.dataset.batch_size
            to_generate = min(samples_left_to_generate,cfg.dataset.batch_size)
            to_save = min(samples_left_to_save, bs)
            #print('to_save: ' +str(to_save))
            #print('to_generate: '+str(to_generate))
            chains_save = 0#min(chains_left_to_save, bs)
            temp_samples = model.sample_batch(id, to_generate, num_nodes=None, save_final=to_save, keep_chain=chains_save, number_chain_steps=number_chain_steps)
            #print('chains_save: '+str(chains_save))
            #samples.extend(model.sample_batch(id, to_generate, num_nodes=None, save_final=to_save, keep_chain=chains_save, number_chain_steps=number_chain_steps))
            for sample in temp_samples:
                temp_indexes = torch.nonzero(torch.tensor(sample[1]))
                edge_list_samples += [((int(sample[0][i].cpu())), int(sample[0][j].cpu())) for i, j in temp_indexes]
                if dir == True:
                    g = nx.from_edgelist(edge_list_samples, create_using=nx.DiGraph())
                else:
                    g = nx.from_edgelist(edge_list_samples, create_using=nx.Graph())
                n_edges = len(list(g.edges()))
                number_of_edges.append(n_edges)
                if n_edges > cfg.dataset.edge_count: #73312:#12761:#27755:#16706:#int(i*16706/10):#34812:#77943:#81492:#10556:
                    break

            print('number of edges sampled: '+str(n_edges))
            if n_edges > cfg.dataset.edge_count: #73312:#12761:#27755:#16706:#int(i*16706/10):#34812:#77943:#81492:#10556:
                break
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        # for sample in samples:
        #     temp_indexes = torch.nonzero(sample[1])
        #     edge_list_samples += [((int(sample[0][i].cpu())), int(sample[0][j].cpu())) for i, j in temp_indexes]

        import pickle
        with open('./' + str(i) + f'_{cfg.dataset.name}_edge_list.pkl', 'wb') as f:
            pickle.dump(edge_list_samples, f)
        with open('./' + str(i) + f'_{cfg.dataset.name}_edge_list_numbers_sampled.pkl', 'wb') as f:
            pickle.dump(number_of_edges, f)
    
    
if __name__ == '__main__':
    main()
#
#
# base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
# graphs = EmailEUCore(base_path)
#
# edge_list = graphs.data.edge_index
#
# n_nodes = graphs.data.num_nodes
# subgraph_size = 100
#
#
# n_samples = int(n_nodes**2*math.log(n_nodes)/100)
# dataset_node_lists = [list(random.choices(range(n_nodes),k=100)) for i in range(n_samples)]
#
# print(f'We need to sample {n_samples} subgraphs')
# dataset_samples_initialids = [(dataset_node_lists[i],
#                 subgraph(torch.tensor(dataset_node_lists[i]),edge_list)[0])
#                 for i in range(n_samples)]
# dict_maps= [{dataset_samples_initialids[j][0][i]: i for i in range(subgraph_size)} for j in range(n_samples)]
# dataset_samples_wnmaps = [(torch.tensor([[x] for x in dataset_samples_initialids[i][0]]),
#                            dataset_samples_initialids[i][1].apply_(lambda x: dict_maps[i][x])) for i in range(n_samples)]
#
#
# DataSet = [Data(x = dataset_samples_wnmaps[i][0], edge_index = dataset_samples_wnmaps[i][1], n_nodes = subgraph_size) for i in range(n_samples)]
#
#
#
#
#
# edge_list = []
# f = open('email-Eu-core.txt','r')
# for edge in f.readlines():
#     tmp = edge.strip().split()
#     edge_list.append([int(tmp[0]), int(tmp[1])])
#
#
# nx_graph = nx.from_edgelist(edge_list, create_using=nx.DiGraph())
#
# sparce_graph = nx.to_scipy_sparse_array(nx_graph)
#
# sample = GraphSampler(sparce_graph,np.array(range(1003)),5,{})
#
