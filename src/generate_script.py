import torch 
import pickle
import networkx as nx

from src.analysis.spectre_utils import LargeGraphSamplingMetrics
from src.datasets.large_graph_datasets import LargeGraphModule,LargeGraphDatasetInfos
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.analysis.visualization import NonMolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures
import hydra
from omegaconf import DictConfig
import os 

models_dir = 'saved_models'
gen_graphs_save_dir = 'generated_graphs'
no_of_graphs_to_generate = 10

@hydra.main(version_base='1.1', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    
    root_path = os.getcwd().split('outputs')[0]

    os.makedirs(os.path.join(root_path, gen_graphs_save_dir, cfg.dataset.name), exist_ok=True)

    datamodule = LargeGraphModule(cfg, sample=False)
    # sampling_metrics = LargeGraphSamplingMetrics(datamodule.dataloaders)

    dataset_infos = LargeGraphDatasetInfos(datamodule, cfg["dataset"])
    train_metrics = TrainAbstractMetricsDiscrete()
    visualization_tools = NonMolecularVisualization()
    extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)

    model_kwargs = {'dataset_infos': dataset_infos, 
                    'train_metrics': train_metrics, 
                    'visualization_tools': visualization_tools,
                    'sampling_metrics': None, 
                    'extra_features': extra_features, 
                    'domain_features': domain_features }

    model_path = None 
    import pdb; pdb.set_trace()
    for file in os.listdir(os.path.join(root_path, models_dir)):
        if file.endswith('.ckpt') and cfg.dataset.name in file:
            model_path = file
            break

    print('using model : ', model_path)

    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs).load_from_checkpoint(model_path)

    device = torch.device('cuda')
    model.to(device)
    dir = cfg.dataset.directed if hasattr(cfg.dataset, 'directed') else False

    for i in range(no_of_graphs_to_generate):
        torch.cuda.empty_cache()
        samples_left_to_generate = cfg.general.final_model_samples_to_generate
        samples_left_to_save = 1    #cfg.general.final_model_samples_to_save
        number_chain_steps = cfg.general.number_chain_steps
        chains_left_to_save = cfg.general.final_model_chains_to_save
        id = 0
        edge_list_samples = []
        number_of_edges = []
        n_edges = 0

        while samples_left_to_generate > 0:
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
                if n_edges > cfg.dataset.edge_count:
                    break

            print('number of edges sampled: '+str(n_edges))
            if n_edges > cfg.dataset.edge_count:
                break
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        with open(os.path.join(gen_graphs_save_dir, cfg.dataset.name, + str(i) + f'_{cfg.dataset.name}_edge_list.pkl'), 'wb') as f:
            pickle.dump(edge_list_samples, f)
        with open(os.path.join(gen_graphs_save_dir, cfg.dataset.name, + str(i) + f'_{cfg.dataset.name}_edge_list_numbers_sampled.pkl'), 'wb') as f:
            pickle.dump(number_of_edges, f)



if __name__ == '__main__':
    main()
