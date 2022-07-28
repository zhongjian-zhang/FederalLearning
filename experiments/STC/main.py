from models import *
from utils import *
import time
import random


def train():
    hp = {
        "communication_rounds": 20,
        "dataset": "mnist",
        "n_clients": 50,
        "classes_per_client": 10,
        "local_iterations": 1,
        "weight_decay": 0.0,
        "optimizer": "SGD",
        "log_frequency": -100,
        "count_bits": False,
        "participation_rate": 1.0,
        "balancedness": 1.0,
        "compression_up": ["stc", {"p": 0.001}],
        "compression_down": ["stc", {"p": 0.002}],
        "accumulation_up": True,
        "accumulation_down": True,
        "aggregation": "mean",
        'type': 'CNN', 'lr': 0.04,
        'batch_size': 100,
        'lr_decay': ['LambdaLR', {'lr_lambda': lambda epoch: 1.0}],
        'momentum': 0.0,
    }
    xp = {
        "iterations": 100,
        "participation_rate": 0.5,
        "momentum": 0.9,
        "compression": [
            "stc_updown",
            {
                "p_up": 0.001,
                "p_down": 0.002
            }
        ],
        "log_frequency": 30,
        "log_path": "results/trash/"
    }
    # 加载数据集并根据客户端来进行划分
    client_loaders, train_loader, test_loader, stats = get_data_loaders(hp)
    # 初始化服务器与客户端的神经网络模型
    net = logistic()
    clients = [Client(loader, net, hp, xp, id_num=i) for i, loader in enumerate(client_loaders)]
    server = Server(test_loader, net, hp, xp, stats)
    # 开始训练
    print("Start Distributed Training..\n")
    t1 = time.time()
    for c_round in range(1, hp['communication_rounds'] + 1):
        # 随机选择一定的客户端来训练
        participating_clients = random.sample(clients, int(len(clients) * hp['participation_rate']))
        # 客户端
        for client in participating_clients:
            client.synchronize_with_server(server)  # 加载当前全局模型参数
            client.compute_weight_update(hp['local_iterations'])  # 权重更性
            client.compress_weight_update_up(compression=hp['compression_up'], accumulate=hp['accumulation_up'],
                                             count_bits=hp["count_bits"])  # 超参数压缩，联邦通信优化

        # 服务端
        server.aggregate_weight_updates(participating_clients, aggregation=hp['aggregation'])  # 聚集客户端的权重
        server.compress_weight_update_down(compression=hp['compression_down'], accumulate=hp['accumulation_down'],
                                           count_bits=hp["count_bits"])  # 超参数压缩，联邦通信优化
        # 全局模型评估
        print("Evaluate...")
        results_train = server.evaluate(max_samples=5000, loader=train_loader)
        results_test = server.evaluate(max_samples=10000)
        # 日志情况
        print({'communication_round': c_round, 'lr': clients[0].optimizer.__dict__['param_groups'][0]['lr'],
                'epoch': clients[0].epoch, 'iteration': c_round * hp['local_iterations']})
        print({'client{}_loss'.format(client.id): client.train_loss for client in clients})

        print({key + '_train': value for key, value in results_train.items()})
        print({key + '_test': value for key, value in results_test.items()})

        print({'time': time.time() - t1})
        total_time = time.time() - t1
        avrg_time_per_c_round = (total_time) / c_round
        e = int(avrg_time_per_c_round * (hp['communication_rounds'] - c_round))
        print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60),
              "[{:.2f}%]\n".format(c_round / hp['communication_rounds'] * 100))


if __name__ == '__main__':
    train()
