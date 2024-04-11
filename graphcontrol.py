import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np


from utils.random import reset_random_seed
from utils.args import Arguments
from utils.sampling import collect_subgraphs
from utils.transforms import process_attributes, obtain_attributes
from models import load_model
from datasets import NodeDataset
from optimizers import create_optimizer

import loralib as lora


def preprocess(config, dataset_obj, device):
    kwargs = {'batch_size': config.batch_size, 'num_workers': 4, 'persistent_workers': True, 'pin_memory': True}
    
    print('generating subgraphs....')
    
    train_idx = dataset_obj.data.train_mask.nonzero().squeeze()
    test_idx = dataset_obj.data.test_mask.nonzero().squeeze()
    # 随机游走子图
    train_graphs = collect_subgraphs(train_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart)
    test_graphs = collect_subgraphs(test_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart)

    [process_attributes(g, use_adj=config.use_adj, threshold=config.threshold, num_dim=config.num_dim) for g in train_graphs]
    [process_attributes(g, use_adj=config.use_adj, threshold=config.threshold, num_dim=config.num_dim) for g in test_graphs]
    
        
    train_loader = DataLoader(train_graphs, shuffle=True, **kwargs)
    test_loader = DataLoader(test_graphs, **kwargs)

    return train_loader, test_loader


def finetune(config, model, train_loader, device, full_x_sim, test_loader):
    # print(model.named_parameters())
    # freeze the pre-trained encoder (left branch)
    # for k, v in model.named_parameters():
    #     if 'encoder' in k:
    #         v.requires_grad = False
    lora.mark_only_lora_as_trainable(model)
            
    model.reset_classifier()
    eval_steps = 3
    patience = 15
    count = 0
    best_acc = 0

    params  = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = create_optimizer(name=config.optimizer, parameters=params, lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    process_bar = tqdm(range(config.epochs))

    for epoch in process_bar:
        for data in train_loader:
            optimizer.zero_grad()
            model.train()

            data = data.to(device)
            
            if not hasattr(data, 'root_n_id'):
                data.root_n_id = data.root_n_index
            #from gcc sign flip, because the sign of eigen-vectors can be filpped randomly (annotate this operate if we conduct eigen-decomposition on full graph)
            sign_flip = torch.rand(data.x.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            x = data.x * sign_flip.unsqueeze(0)
            
            x_sim = full_x_sim[data.original_idx]
            preds = model.forward_subgraph(x, x_sim, data.edge_index, data.batch, data.root_n_id, frozen=True)
                
            loss = criterion(preds, data.y)
            # element 0 of tensors does not require grad and does not have a grad_fn
            # loss.requires_grad = True
            loss.backward()
            optimizer.step()
    
        if epoch % eval_steps == 0:
            acc = eval_subgraph(config, model, test_loader, device, full_x_sim)
            process_bar.set_postfix({"Epoch": epoch, "Accuracy": f"{acc:.4f}"})
            if best_acc < acc:
                best_acc = acc
                count = 0
            else:
                count += 1

        if count == patience:
            break

    return best_acc


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    # 转无向图，1：9分train/test，seed为参数输入，数据集为节点分类
    dataset_obj = NodeDataset(config.dataset, n_seeds=config.seeds)
    dataset_obj.print_statistics()
    
    # For large graph, we use cpu to preprocess it rather than gpu because of OOM problem.
    if dataset_obj.num_nodes < 30000:
        dataset_obj.to(device)
    torch.set_printoptions(profile="full")
    print(dataset_obj.data.x[0])
    torch.set_printoptions(profile="default") 
    
    num_node_features = config.num_dim
    x_sim = obtain_attributes(dataset_obj.data, use_adj=False, threshold=config.threshold, num_dim=config.num_dim).to(device)
    print(x_sim[0])
    
    dataset_obj.to('cpu') # Otherwise the deepcopy will raise an error

    train_masks = dataset_obj.data.train_mask
    test_masks = dataset_obj.data.test_mask

    acc_list = []

    for i, seed in enumerate(config.seeds):
        reset_random_seed(seed)
        if dataset_obj.random_split:
            dataset_obj.data.train_mask = train_masks[:, seed]
            dataset_obj.data.test_mask = test_masks[:, seed]
        
        train_loader, test_loader = preprocess(config, dataset_obj, device)
        
        model = load_model(num_node_features, dataset_obj.num_classes, config)
        # lora.mark_only_lora_as_trainable(model)
        model = model.to(device)

        # finetuning model
        best_acc = finetune(config, model, train_loader, device, x_sim, test_loader)
        
        acc_list.append(best_acc)
        print(f'Seed: {seed}, Accuracy: {best_acc:.4f}')

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")


def eval_subgraph(config, model, test_loader, device, full_x_sim):
    model.eval()
    
    correct = 0
    total_num = 0
    for batch in test_loader:
        batch = batch.to(device)
        if not hasattr(batch, 'root_n_id'):
            batch.root_n_id = batch.root_n_index
        x_sim = full_x_sim[batch.original_idx]
        preds = model.forward_subgraph(batch.x, x_sim, batch.edge_index, batch.batch, batch.root_n_id, frozen=True).argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total_num += batch.y.shape[0]
    acc = correct / total_num
    return acc

if __name__ == '__main__':
    config = Arguments().parse_args()
    
    main(config)