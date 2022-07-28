import random
from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim
from utils import batch_data, assess_fun, word_to_indices, letter_to_vec
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, seed, lr, optimizer=None):
        super().__init__()
        self.lr = lr
        self.seed = seed
        self.optimizer = optimizer
        self.flops = 0
        self.size = 0

    def get_params(self):
        return self.state_dict()

    def set_params(self, state_dict):
        self.load_state_dict(state_dict)

    def __post_init__(self):
        if self.optimizer is None:
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    def train_model(self, data, num_epochs=1, batch_size=10):
        self.train()
        for batch in range(num_epochs):
            for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
                self.optimizer.zero_grad()
                input_data = self.process_x(batched_x)
                target_data = self.process_y(batched_y)
                logits, loss = self.forward(input_data, target_data)
                loss.backward()
                self.optimizer.step()
        update = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size
        return comp, update

    def test_model(self, data):
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        self.eval()
        with torch.no_grad():
            logits, loss = self.forward(x_vecs, labels)
            acc = assess_fun(labels, logits)
        return {"accuracy": acc.detach().cpu().numpy(), 'loss': loss.detach().cpu().numpy()}


class LSTMModel(Model):
    def __init__(self, seed, lr, seq_len, num_classes, n_hidden):
        super().__init__(seed, lr)
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.word_embedding = nn.Embedding(self.num_classes, 8)
        self.lstm = nn.LSTM(input_size=8, hidden_size=self.n_hidden, num_layers=2, batch_first=True)
        self.pred = nn.Linear(self.n_hidden * 2, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        super().__post_init__()

    def forward(self, features, labels):
        emb = self.word_embedding(features)
        output, (h_n, c_n) = self.lstm(emb)
        h_n = h_n.transpose(0, 1).reshape(-1, 2 * self.n_hidden)
        logits = self.pred(h_n)
        loss = self.loss_fn(logits, labels)
        return logits, loss

    def process_x(self, raw_x_batch):
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = torch.LongTensor(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [letter_to_vec(c) for c in raw_y_batch]
        y_batch = torch.LongTensor(y_batch)
        return y_batch


class Client:
    def __init__(self, client_id, train_data, eval_data, model=None):
        self._model = model
        self.id = client_id
        self.train_data = train_data if train_data is not None else {'x': [], 'y': []}
        self.eval_data = eval_data if eval_data is not None else {'x': [], 'y': []}

    @property
    def model(self):
        return self._model

    @property
    def num_test_samples(self):
        if self.eval_data is None:
            return 0
        else:
            return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        if self.train_data is None:
            return 0
        else:
            return len(self.train_data['y'])

    @property
    def num_samples(self):
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])
        eval_size = 0
        if self.eval_data is not None:
            eval_size = len(self.eval_data['y'])
        return train_size + eval_size

    def train(self, num_epochs=1, batch_size=128, minibatch=None):
        if minibatch is None:
            data = self.train_data
            comp, update = self.model.train_model(data, num_epochs, batch_size)
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac * len(self.train_data['y'])))
            xs, xy = zip(*random.sample(list(zip(self.train_data['x'], self.train_data['y'])), num_data))
            data = {
                'x': xs,
                'y': xy
            }
            num_epochs = 1
            comp, update = self.model.train_model(data, num_epochs, num_data)
        num_train_samples = len(data['y'])
        return comp, num_train_samples, update

    def test(self, set_to_use='test'):
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        else:
            data = self.eval_data
        return self.model.test_model(data)


class Serves:
    def __init__(self, global_model):
        self.global_model = global_model
        self.model = global_model.get_params()
        self.selected_clients = []
        self.update = []

    def select_clients(self, my_round, possible_clients, num_clients=20):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        # return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None):
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {"bytes_written": 0,
                   "bytes_read": 0,
                   "local_computations": 0} for c in clients}
        for c in clients:
            c.model.set_params(self.model)
            comp, num_samples, update = c.train(num_epochs, batch_size, minibatch)
            sys_metrics[c.id]["bytes_read"] += c.model.size
            sys_metrics[c.id]["bytes_written"] += c.model.size
            sys_metrics[c.id]["local_computations"] = comp

            self.update.append((num_samples, update))
        return sys_metrics

    def aggregate(self, updates):
        avg_param = OrderedDict()
        total_weight = 0.
        for (client_samples, client_model) in updates:
            total_weight += client_samples
            for name, param in client_model.items():
                if name not in avg_param:
                    avg_param[name] = client_samples * param
                else:
                    avg_param[name] += client_samples * param

        for name in avg_param:
            avg_param[name] = avg_param[name] / total_weight
        return avg_param

    def update_model(self):
        avg_param = self.aggregate(self.update)
        self.model = avg_param
        self.global_model.load_state_dict(self.model)
        self.update = []

    def test_model(self, clients_to_test=None, set_to_use='test'):
        metrics = {}
        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in tqdm(clients_to_test):
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics

        return metrics

    def get_clients_info(self, clients):
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        return torch.save({"model_state_dict": self.model}, path)
