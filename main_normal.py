from contextlib import nullcontext
import numpy as np
import argparse
import wandb
import os
import os.path as osp
import random
import nni
import time

import torch
from torch._C import wait
from torch_geometric.utils import dropout_adj, degree, to_undirected

from simple_param.sp import SimpleParam
from pHNGCL.model import Discriminator, Encoder, HNGCL, ADNet, Dis
from pHNGCL.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pHNGCL.eval import log_regression, MulticlassEvaluator
from pHNGCL.utils import common_loss, generate_feature_graph_edge_index, get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality, loss_dependence
from pHNGCL.dataset import get_dataset

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def train_normal(drop_weights1):
    ADNet.requires_grad_(False)
    model.requires_grad_(True)
    model.train()
    model_optimizer.zero_grad()

    def drop_edge(idx: int, edge_index, drop_weights):
        if config['drop_scheme'] == 'uniform':
            return dropout_adj(edge_index, p=config[f'drop_edge_rate_{idx}'])[0]
        elif config['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(edge_index, drop_weights, p=config[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {config["drop_scheme"]}')

    edge_index_1 = drop_edge(1, data.edge_index, drop_weights1)
    edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0]
    if config['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, config['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, config['drop_feature_rate_2'])
    else:
        x_1 = drop_feature(data.x, config['drop_feature_rate_1'])
        x_2 = drop_feature(data.x, config['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2)
    loss.backward(retain_graph=True)
    model_optimizer.step()

    return loss.item()

def train(drop_weights1, weight):
    ADNet.requires_grad_(False)
    model.requires_grad_(True)
    model.train()
    model_optimizer.zero_grad()

    def drop_edge(idx: int, edge_index, drop_weights):
        if config['drop_scheme'] == 'uniform':
            return dropout_adj(edge_index, p=config[f'drop_edge_rate_{idx}'])[0]
        elif config['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(edge_index, drop_weights, p=config[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {config["drop_scheme"]}')

    edge_index_1 = drop_edge(1, data.edge_index, drop_weights1)
    edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0]
    if config['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, config['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, config['drop_feature_rate_2'])
    else:
        x_1 = drop_feature(data.x, config['drop_feature_rate_1'])
        x_2 = drop_feature(data.x, config['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    z3 = ADNet(x_1, edge_index_1)
    z3 = ADNet.Generate_hard(z1, z3)
    loss = model.loss_neg(z1, z2, z3, weight=weight)

    loss.backward(retain_graph=True)
    model_optimizer.step()

    return loss.item()

# def train(weight):
#     model.requires_grad_(True)
#     ADNet.requires_grad_(False)
#     model.train()
#     model_optimizer.zero_grad()
#
#     def drop_edge(idx: int, edge_index, drop_weights):
#         if config['drop_scheme'] == 'uniform':
#             return dropout_adj(edge_index, p=config[f'drop_edge_rate_{idx}'])[0]
#         elif config['drop_scheme'] in ['degree', 'evc', 'pr']:
#             return drop_edge_weighted(edge_index, drop_weights, p=config[f'drop_edge_rate_{idx}'], threshold=0.7)
#         else:
#             raise Exception(f'undefined drop scheme: {config["drop_scheme"]}')
#
#     edge_index_1 = drop_edge(1, data.edge_index, drop_weights1)
#     edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0]
#     if config['drop_scheme'] in ['pr', 'degree', 'evc']:
#         x_1 = drop_feature_weighted_2(data.x, feature_weights, config['drop_feature_rate_1'])
#         x_2 = drop_feature_weighted_2(data.x, feature_weights, config['drop_feature_rate_2'])
#     else:
#         x_1 = drop_feature(data.x, config['drop_feature_rate_1'])
#         x_2 = drop_feature(data.x, config['drop_feature_rate_2'])
#
#     z1 = model(x_1, edge_index_1)
#     z2 = model(x_2, edge_index_2)
#
#     z3 = ADNet(x_1, edge_index_1)
#     z3 = ADNet.Generate_hard(z1, z3)
#
#     loss = model.loss(z1, z2, z3, batch_size=32, weight=weight)
#     loss.backward(retain_graph=True)
#     model_optimizer.step()
#
#     return loss.item()


def train_hard(drop_weights1, AD_True: int, AD_hard: int, SE: int, True_gap=1, False_gap=1):
    def drop_edge(idx: int, edge_index, drop_weights):
        if config['drop_scheme'] == 'uniform':
            return dropout_adj(edge_index, p=config[f'drop_edge_rate_{idx}'])[0]
        elif config['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(edge_index, drop_weights, p=config[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {config["drop_scheme"]}')

    edge_index_1 = drop_edge(1, data.edge_index, drop_weights1)
    edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0]
    if config['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, config['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, config['drop_feature_rate_2'])
    else:
        x_1 = drop_feature(data.x, config['drop_feature_rate_1'])
        x_2 = drop_feature(data.x, config['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    # 先训练ADNet, 生成比较真实的
    for i in range(AD_True):
        for i in range(SE):  # SE: self enhencment 自我提升的轮数
            discriminator.requires_grad_(False)
            model.requires_grad_(False)
            ADNet.requires_grad_(True)
            ADNet.train()
            # ADNet.requires_grad=True
            ADNet_optimizer.zero_grad()
            z3 = ADNet(x_1, edge_index_1)
            z3 = ADNet.Generate_hard(z1, z3)
            # z1 11701,128
            loss = Dis(discriminator, z1, z3)
            loss.backward(retain_graph=True)
            ADNet_optimizer.step()
            if Dis(discriminator, z1, z3) < True_gap:  # True_gap 多真实就退出， 数值越小，意味要求越像
                break

        # 训练好的ADNet生成困难负样本，来让判别器判别，并使得判别器提升判别能力
        z3 = ADNet(x_1, edge_index_1)
        z3 = ADNet.Generate_hard(z1, z3)

        for i in range(SE):
            discriminator.requires_grad_(True)
            ADNet.requires_grad_(False)
            discriminator.train()
            # ADNet.requires_grad=True
            discriminator_optimizer.zero_grad()
            loss_Dis = 1 - Dis(discriminator, z1, z3)
            loss_Dis.backward(retain_graph=True)
            discriminator_optimizer.step()
            if loss_Dis.item() < False_gap:
                break
        import gc
        del z3
        gc.collect()


    for i in range(AD_hard):  # 使生成的真实的样本 变得困难
        discriminator.requires_grad_(False)
        model.requires_grad_(False)
        ADNet.requires_grad_(True)
        ADNet.train()
        ADNet_optimizer.zero_grad()
        z3 = ADNet(x_1, edge_index_1)
        z3 = ADNet.Generate_hard(z1, z3)
        loss = - model.loss_neg(z1, z2, z3) #+ Dis(discriminator, z1, z3) #困难 且 真实;
        loss.backward(retain_graph=True)
        ADNet_optimizer.step()

    return loss.item()


def test(final=False):
    model.eval()
    ADNet.eval()
    z = model(data.x, data.edge_index)

    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    # if final and use_nni:
    #     nni.report_final_result(acc)
    # elif use_nni:
    #     nni.report_intermediate_result(acc)

    return acc


def save_embedding():
    model.eval()
    z = model(data.x, data.edge_index)
    z = z.detach().cpu().numpy()
    path = osp.expanduser('~/HNGCL-Experiment/result')
    embedding_path = osp.join(path, "visulization", "embeddings", args.dataset)
    file_name = osp.join(embedding_path, args.dataset.lower() + "_" + str(config['k']) + "nn")
    check_dir(file_name)
    np.save(file_name, z)


def save_labels(labels):
    path = osp.expanduser('~/HNGCL-Experiment/result')
    labels_path = osp.join(path, "visulization", "labels", args.dataset.lower())
    check_dir(labels_path)
    np.save(labels_path, labels)


def plot_embedding(labels):
    path = osp.expanduser('~/HNGCL-Experiment/result')
    embedding_path = osp.join(path, "visulization", "embeddings", args.dataset)
    figure_path = osp.join(path, "visulization", "figures", args.dataset)
    check_dir(embedding_path)
    check_dir(figure_path)

    embeddings = np.load(osp.join(embedding_path, args.dataset.lower() + "_" + str(config['k']) + "nn.npy"))
    tsne = TSNE(init='pca', random_state=0)

    tsne_features = tsne.fit_transform(embeddings)

    xs = tsne_features[:, 0]
    ys = tsne_features[:, 1]

    plt.scatter(xs, ys, c=labels)
    figure_name = osp.join(figure_path, args.dataset.lower() + "_" + str(config['k']) + "nn.pdf")
    check_dir(figure_name)
    plt.savefig(figure_name)


def check_dir(file_name=None):
    dir_name = osp.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def record_hyper_parameter(result_file, param):
    fb = open(result_file, 'a+', encoding='utf-8')
    fb.write('\n'*5)
    fb.write('-'*30 + ' ' * 5 + 'Hyper parameters in training' + ' ' * 5 + '-'*30 + '\n\n')
    fb.write("total training epoches: {}\n".format(args.num_epochs))
    fb.write("learning rate: {}\n".format(param['learning_rate']))
    fb.write("hidden num: {}\n".format(param['num_hidden']))
    fb.write("projection hidden num: {}\n".format(param['num_proj_hidden']))
    fb.write("activation function: {}\n".format(param['activation']))
    fb.write("drop edge rate1: {}\n".format(param['drop_edge_rate_1']))
    fb.write("drop edge rate2: {}\n".format(param['drop_edge_rate_2']))
    fb.write("drop feature rate1: {}\n".format(param['drop_feature_rate_1']))
    fb.write("drop feature rate2: {}\n".format(param['drop_feature_rate_2']))
    fb.write("temperature coefficient tau: {}\n".format(param['tau']))
    fb.write("alpha: {}\n".format(param['alpha']))
    fb.write("AD_true:{}\n".format(param['AD_True']))
    fb.write("AD_hard:{}\n".format(param['AD_hard']))
    fb.write("hard_num:{}\n".format(param['hard_num']))
    fb.write("True_gap:{}\n".format(param['True_gap']))
    fb.write("False_gap:{}\n".format(param['False_gap']))
    fb.write("hard negatives Weight:{}\n".format(param['weight']))
    fb.write("Self enhencment :{}\n".format(param['SE']))
    fb.write("Stop :{}\n".format(param['stop']))
    fb.write("weight :{}\n".format(param['weight']))
    fb.write('\n' + '-'*30 + ' ' * 5 + 'Hyper parameters in training' + ' ' * 5 + '-'*30 + '\n')
    fb.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Coauthor-Phy')
    # parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--seed', type=int, default=120456)#120546
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--num_epochs', type=int, default='1500')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--theta', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--activation', type=str, default='rrelu')
    parser.add_argument('--base_model', type=str, default= 'GCNConv')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--num_proj_hidden', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.4)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.1)
    parser.add_argument('--drop_feature_rate_1', type=float, default=0.1)
    parser.add_argument('--drop_feature_rate_2', type=float, default=0.4)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--drop_scheme', type=str, default='degree')
    parser.add_argument('--hard_num', type=int, default=32)
    parser.add_argument('--SE', type=int, default=30)
    parser.add_argument('--True_gap', type=float, default=0.1)
    parser.add_argument('--False_gap', type=float, default=0.1)
    parser.add_argument('--AD_True', type=int, default=30)
    parser.add_argument('--AD_hard', type=int, default=30)
    parser.add_argument('--weight', type=int, default=10)
    parser.add_argument('--stop', type=int, default=1500)
    parser.add_argument('--AD_rate', type=float, default=0.01)
    parser.add_argument('--sum_number', type=int, default=10)
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--hard', type=str, default='True')

    args = parser.parse_args()
    # ! Wandb settings
    wandb.init(config=args)
    config = wandb.config
    args = config

    # sp = SimpleParam(default=config)
    # param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    # for key in param_keys:
    #     if getattr(args, key) is not None:
    #         param[key] = getattr(args, key)
    # # use_nni = args.param == 'nni'
    # if use_nni and args.device != 'cpu':
    #     args.device = 'cuda:1'

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)
    device = torch.device(args.device)

    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    data = data.to(device)

    # feature_graph_edge_index = generate_feature_graph_edge_index(data.x, config['k']).to(device)

    # generate split
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)
    if args.save_split:
        torch.save(split, args.save_split)
    elif args.load_split:
        split = torch.load(args.load_split)

    encoder = Encoder(dataset.num_features, config['num_hidden'], get_activation(config['activation']),
                      base_model=get_base_model(config['base_model']), k=config['num_layers']).to(device)
    #


    discriminator = Discriminator(config['num_hidden'], config['num_proj_hidden'], 1).to(device)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )



    ADNet = ADNet(dataset.num_features, config['hard_num'], get_activation(config['activation']),
                  base_model=get_base_model(config['base_model']), k=config['num_layers']).to(device)
    ADNet_optimizer = torch.optim.Adam(
        ADNet.parameters(),
        lr=config['AD_rate'],
        weight_decay=config['weight_decay']
    )

    model = HNGCL(encoder, config['num_hidden'], config['num_proj_hidden'], config['tau'], config['alpha']).to(device)
    # 256;32
    model_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    if config['drop_scheme'] == 'degree':
        drop_weights1 = degree_drop_weights(data.edge_index).to(device)
        drop_weights2 = degree_drop_weights(data.edge_index).to(device)
    elif config['drop_scheme'] == 'pr':
        drop_weights1 = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
        drop_weights2 = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif config['drop_scheme'] == 'evc':
        drop_weights1 = evc_drop_weights(data).to(device)
        drop_weights2 = drop_weights1[:data.edge_index.size()[1]]
    else:
        drop_weights1 = None
        drop_weights2 = None

    if config['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif config['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif config['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']

    log = args.verbose.split(',')


    result_file_path = osp.expanduser('~/HNGCL-Experiment/result')
    result_file = osp.join(result_file_path, args.dataset, "{}_epoches_{}NN_on_{}_result.txt".format(config["num_epochs"], config['k'], args.dataset))
    check_dir(result_file)

    record_hyper_parameter(result_file, config)

    with open(result_file, 'a+', encoding='utf-8') as fb:
        best_acc = 0.0
        best_epoch = 0
        wait_times = 0
        if config['hard'] == True:
            print('warmup phase!')
            loss_hard = train_hard(drop_weights1, config['AD_True'], config['AD_hard'], config['SE'], config['True_gap'], config['False_gap'])
            print('warmup phase final loss:', loss_hard)

            for epoch in range(config["num_epochs"]):
                if config['mode'] == 'normal':
                    loss = train(drop_weights1, config['weight'])
                    # if epoch < config['stop']:  # 参数
                    _ = train_hard(drop_weights1, 1, 1, 1)
                if epoch <= 1200 and epoch % 100 == 0:
                    acc = test()
                    if acc > best_acc:
                        best_acc = acc
                        best_epoch = epoch
                        wait_times = 0

                    else:
                        wait_times += 1
                        if wait_times > args.patience:
                            break
                    print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}\tBest epoch={best_epoch:04d}, best_acc = {best_acc:.4f}')
                    line_str = '(T) | Epoch={:04d}, loss={:.2f}\t(E) | Epoch={:04d}, avg_acc = {:.4f}\tBest epoch={:04d}, best_acc = {:.4f}\n'
                    fb.write(line_str.format(epoch, loss, epoch, acc, best_epoch, best_acc))
                    metrics = {"Acc": acc, "Loss": loss, "Best_acc": best_acc}
                    wandb.log(metrics)
                elif epoch > 1200 and epoch % 50 == 0:
                    acc = test()
                    if acc > best_acc:
                        best_acc = acc
                        best_epoch = epoch
                        wait_times = 0

                    else:
                        wait_times += 1
                        # if wait_times > args.patience:
                        #     break
                    print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}\tBest epoch={best_epoch:04d}, best_acc = {best_acc:.4f}')
                    line_str = '(T) | Epoch={:04d}, loss={:.2f}\t(E) | Epoch={:04d}, avg_acc = {:.4f}\tBest epoch={:04d}, best_acc = {:.4f}\n'
                    fb.write(line_str.format(epoch, loss, epoch, acc, best_epoch, best_acc))
                    metrics = {"Acc": acc, "Loss": loss, "Best_acc": best_acc}
                    wandb.log(metrics)
        else:
            for epoch in range(config["num_epochs"]):  # 1, param['num_epochs'] + 1
                loss = train_normal(drop_weights1)
                if epoch <= 1200 and epoch % 100 == 0:
                    acc = test()
                    if acc > best_acc:
                        best_acc = acc
                        best_epoch = epoch
                        wait_times = 0

                    else:
                        wait_times += 1
                        if wait_times > args.patience:
                            break
                    print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}\tBest epoch={best_epoch:04d}, best_acc = {best_acc:.4f}')
                    line_str = '(T) | Epoch={:04d}, loss={:.2f}\t(E) | Epoch={:04d}, avg_acc = {:.4f}\tBest epoch={:04d}, best_acc = {:.4f}\n'
                    fb.write(line_str.format(epoch, loss, epoch, acc, best_epoch, best_acc))
                    metrics = {"Acc": acc, "Loss": loss, "Best_acc": best_acc}
                    wandb.log(metrics)
                elif epoch > 1200 and epoch % 50 == 0:
                    acc = test()
                    if acc > best_acc:
                        best_acc = acc
                        best_epoch = epoch
                        wait_times = 0

                    else:
                        wait_times += 1
                        # if wait_times > args.patience:
                        #     break
                    print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}\tBest epoch={best_epoch:04d}, best_acc = {best_acc:.4f}')
                    line_str = '(T) | Epoch={:04d}, loss={:.2f}\t(E) | Epoch={:04d}, avg_acc = {:.4f}\tBest epoch={:04d}, best_acc = {:.4f}\n'
                    fb.write(line_str.format(epoch, loss, epoch, acc, best_epoch, best_acc))
                    metrics = {"Acc": acc, "Loss": loss, "Best_acc": best_acc}
                    wandb.log(metrics)

        acc = test(final=True)

        if 'final' in log:
            print(f'{acc}')
        fb.write("final result: {:.4f}".format(acc))
        # save_embedding()
        # save_labels(dataset[0].y.view(-1))
        # plot_embedding(dataset[0].y.view(-1))
    fb.close()

    # # for epoch in range(1, config['num_warmup'] + 1):
    # print('warmup phase!')
    # loss_hard = train_hard(feature_graph_edge_index, drop_weights1, drop_weights2, config['AD_True'], config['AD_hard'], param['SE'], param['True_gap'], param['False_gap'])
    # #print(f'(T) | Epoch={epoch:04d}, loss_hard={loss_hard:.4f}')
    #
    #
    # for epoch in range(1, param['num_epochs'] + 1):
    #     time_start = time.time()
    #     loss = train(feature_graph_edge_index, drop_weights1, drop_weights2)
    #
    #     loss_hard_1 = train_hard(feature_graph_edge_index, drop_weights1, drop_weights2, 1, 1, 1)  # 正常训练时 一次就行
    #
    #     time_end = time.time()
    #     time_c = time_end - time_start
    #     print('time cost', time_c, 's')
    #     if 'train' in log:
    #         print(f'(T) | Epoch={epoch:04d}, loss={loss:.4f}, loss_hard={loss_hard_1:.4f}')
    #         # wandb.log(f'(T) | Epoch={epoch:04d}, loss={loss:.4f}, loss_hard={loss_hard:.4f}')
    #
    # acc = test(final=True)
    #
    # if 'final' in log:
    #     print(f'acc:{acc}')

    # save_embedding()
    # save_labels(dataset[0].y.view(-1))
    # plot_embedding(dataset[0].y.view(-1))