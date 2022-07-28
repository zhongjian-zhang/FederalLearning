import argparse
import json
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from Models import CNN
import numpy as np
import torch
import torch.utils.data
from tqdm import trange

from experiments.pfedhn.node import BaseNodes
from experiments.utils import get_device, set_logger, set_seed, str2bool


def evaluate(net, global_parameters, testDataLoader, dev):
    net.load_state_dict(global_parameters, strict=True)
    running_correct = 0
    running_samples = 0
    net.eval()
    # 载入测试集
    for data, label in testDataLoader:
        data, label = data.to(dev), label.to(dev)
        pred = net(data)
        running_correct += pred.argmax(1).eq(label).sum().item()
        running_samples += len(label)
    print("\t" + 'accuracy: %.2f' % (running_correct / running_samples), end="")


def local_upload(train_data_set, local_epoch, net, loss_fun, opt, global_parameters, dev):
    # 加载当前通信中最新全局参数
    net.load_state_dict(global_parameters, strict=True)
    # 设置迭代次数
    net.train()
    for epoch in range(local_epoch):
        for data, label in train_data_set:
            data, label = data.to(dev), label.to(dev)
            # 模型上传入数据
            predict = net(data)
            loss = loss_fun(predict, label)
            # 反向传播
            loss.backward()
            # 计算梯度，并更新梯度
            opt.step()
            # 将梯度归零，初始化梯度
            opt.zero_grad()
    # 返回当前Client基于自己的数据训练得到的新的模型参数
    return net.state_dict()


def train(data_name: str, data_path: str, classes_per_node: int, num_nodes: int,
          steps: int, node_iter: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          seed: int) -> None:
    ###############################
    # init nodes, hnet, local net #
    ###############################
    steps = 5
    node_iter = 5
    nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=classes_per_node,
                      batch_size=bs)
    net = CNN(n_kernels=n_kernels)
    # hnet = hnet.to(device)
    net = net.to(device)

    ##################
    # init optimizer #
    ##################
    # embed_lr = embed_lr if embed_lr is not None else lr
    optimizer = torch.optim.SGD(
        net.parameters(), lr=inner_lr, momentum=.9, weight_decay=inner_wd
    )
    criteria = torch.nn.CrossEntropyLoss()

    ################
    # init metrics #
    ################
    # step_iter = trange(steps)
    step_iter = range(steps)
    # train process
    # record  the global parameters
    global_parameters = {}
    for key, parameter in net.state_dict().items():
        global_parameters[key] = parameter.clone()
    for step in step_iter:

        local_parameters_list = {}
        # 需要训练的node数目
        for i in range(node_iter):
            # 随机选择一个客户端
            node_id = random.choice(range(num_nodes))
            # 用全局模型参数训练当前客户端
            local_parameters = local_upload(nodes.train_loaders[node_id], 5, net, criteria, optimizer,
                                            global_parameters, dev='cpu')
            print("\nEpoch: {}, Node Count: {}, Node ID: {}".format(step + 1, i + 1, node_id), end="")
            evaluate(net, local_parameters, nodes.val_loaders[node_id], 'cpu')
            local_parameters_list[i] = local_parameters

        # 更新当前轮次模型的参数
        sum_parameters = None
        for node_id, parameters in local_parameters_list.items():
            if sum_parameters is None:
                sum_parameters = parameters
            else:
                for key in parameters.keys():
                    sum_parameters[key] += parameters[key]
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / node_iter)
    # test
    net.load_state_dict(global_parameters, strict=True)
    net.eval()
    for data_set in nodes.test_loaders:
        running_correct = 0
        running_samples = 0
        for data, label in data_set:
            pred = net(data)
            running_correct += pred.argmax(1).eq(label).sum().item()
            running_samples += len(label)
        print("\t" + 'accuracy: %.2f' % (running_correct / running_samples), end="")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Hypernetwork with Lookahead experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="cifar10", choices=['cifar10', 'cifar100'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
    parser.add_argument("--num-nodes", type=int, default=50, help="number of simulated nodes")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inner-steps", type=int, default=50, help="number of inner steps")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=0.15, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-5, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="pfedhn_hetro_res", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    if args.data_name == 'cifar10':
        args.classes_per_node = 2
    else:
        args.classes_per_node = 10

    train(
        data_name=args.data_name,
        data_path=args.data_path,
        classes_per_node=args.classes_per_node,
        num_nodes=args.num_nodes,
        steps=args.num_steps,
        node_iter=args.inner_steps,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed
    )
