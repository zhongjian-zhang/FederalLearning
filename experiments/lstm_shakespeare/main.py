from models import *
import numpy as np
import torch
import os
from collections import defaultdict
import json


def read_dir(data_dir):
    clients = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, data


def read_data(train_data_dir, test_data_dir):
    train_clients, train_data = read_dir(train_data_dir)
    test_clients, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients

    return train_clients, train_data, test_data


def create_clients(users, train_data, test_data, model):
    clients = [Client(u, train_data[u], test_data[u], model) for u in users]
    return clients


def setup_clients(model=None, use_val_set=False):
    eval_set = 'test' if not use_val_set else 'test'
    train_data_dir = os.path.join('.', 'data', 'train')
    test_data_dir = os.path.join('.', 'data', eval_set)
    users, train_data, test_data = read_data(train_data_dir, test_data_dir)

    clients = create_clients(users, train_data, test_data, model)

    return clients


def main():
    seed = 1
    random.seed(1 + seed)
    np.random.seed(12 + seed)
    torch.manual_seed(123 + seed)
    torch.manual_seed(123 + seed)
    lr = 0.0003
    seq_len = 80
    num_classes = 80
    n_hidden = 256
    num_rounds = 20
    eval_every = 1
    clients_per_round = 2
    num_epochs = 1
    batch_size = 10
    minibatch = None
    use_val_set = 'test'
    # 全局模型(服务端)
    global_model = LSTMModel(seed, lr, seq_len, num_classes, n_hidden)
    # 服务端
    server = Serves(global_model)
    # 客户端
    client_model = LSTMModel(seed, lr, seq_len, num_classes, n_hidden)
    clients = setup_clients(client_model, use_val_set)
    client_ids, client_num_samples = server.get_clients_info(clients)
    print(('Clients in Total: %d' % len(clients)))
    print('--- Random Initialization ---')
    # Simulate training
    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        # Select clients to train this round
        server.select_clients(i, clients, num_clients=clients_per_round)
        c_ids, c_num_samples = server.get_clients_info(server.selected_clients)

        # Simulate server model training on selected clients' data
        sys_metrics = server.train_model(num_epochs=num_epochs, batch_size=batch_size,
                                         minibatch=minibatch)
        print(sys_metrics)
        # sys_writer_fn(i + 1, c_ids, sys_metrics, c_num_samples)
        metrics = server.test_model()
        print(metrics)

        # Update server model
        server.update_model()

        # Test model


if __name__ == '__main__':
    main()
