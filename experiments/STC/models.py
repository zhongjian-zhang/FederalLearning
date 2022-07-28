import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = 'cpu'


class logistic(nn.Module):
    """
    logistic模型，用于MINIST图片分类预测
    """

    def __init__(self, in_size=32 * 32 * 1, num_classes=10):
        super(logistic, self).__init__()
        self.linear = nn.Linear(in_size, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear(out)
        return out


class DistributedTrainingDevice(object):
    '''
    分布式培训设备类（客户端或服务器）
    dataloader: 由数据点（x，y）组成的pytorch数据集
    model: pytorch神经网络
    hyperparameters：包含所有超参数的python dict
    experiment: 实验类型
    '''

    def __init__(self, dataloader, model, hyperparameters, experiment):
        self.hp = hyperparameters
        self.xp = experiment
        self.loader = dataloader
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def copy(self, target, source):
        """拷贝超参数，结果保存在target中"""
        for name in target:
            target[name].data = source[name].data.clone()

    def add(self, target, source):
        """超参数做加法，结果保存在target中"""
        for name in target:
            target[name].data += source[name].data.clone()

    def subtract(self, target, source):
        """超参数做减法，结果保存在target中"""
        for name in target:
            target[name].data -= source[name].data.clone()

    def subtract_(self, target, minuend, subtrahend):
        """超参数做减法(minuend-subtrahend)，结果保存在target中"""
        for name in target:
            target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()

    def approx_v(self, T, p, frac):
        if frac < 1.0:
            n_elements = T.numel()
            n_sample = min(int(max(np.ceil(n_elements * frac), np.ceil(100 / p))), n_elements)
            n_top = int(np.ceil(n_sample * p))

            if n_elements == n_sample:
                i = 0
            else:
                i = np.random.randint(n_elements - n_sample)

            topk, _ = torch.topk(T.flatten()[i:i + n_sample], n_top)
            if topk[-1] == 0.0 or topk[-1] == T.max():
                return self.approx_v(T, p, 1.0)
        else:
            n_elements = T.numel()
            n_top = int(np.ceil(n_elements * p))
            topk, _ = torch.topk(T.flatten(), n_top)  # 返回列表中最大的n_top个值

        return topk[-1], topk

    def stc(self, T, hp):
        """稀疏三元组压缩算法"""
        hp_ = {'p': 0.001, 'approx': 1.0}
        hp_.update(hp)

        T_abs = torch.abs(T)

        v, topk = self.approx_v(T_abs, hp_["p"], hp_["approx"])
        mean = torch.mean(topk)  # 前n_top的均值

        out_ = torch.where(T >= v, mean, torch.Tensor([0.0]).to(device))  # 大于均值的重新赋值为均值，小于自己的赋值为0
        out = torch.where(T <= -v, -mean, out_)  # 小于副的均值的赋值为-v，大于的赋值为out_对应索引值

        return out

    def compress(self, target, source):
        '''
        分别对每一个超参数进行稀疏三元压缩
        '''
        for name in target:
            target[name].data = self.stc(source[name].data.clone(), self.hp)


class Client(DistributedTrainingDevice):
    """
    客户端类，继承分布式培训设备类
    """

    def __init__(self, dataloader, model, hyperparameters, experiment, id_num=0):
        super().__init__(dataloader, model, hyperparameters, experiment)

        self.id = id_num

        # 超参数
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW_compressed = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.n_params = sum([T.numel() for T in self.W.values()])
        self.bits_sent = []

        optimizer_object = getattr(optim, self.hp['optimizer'])
        optimizer_parameters = {k: v for k, v in self.hp.items() if k in optimizer_object.__init__.__code__.co_varnames}

        self.optimizer = optimizer_object(self.model.parameters(), **optimizer_parameters)

        # 学习率动态变化
        self.scheduler = getattr(optim.lr_scheduler, self.hp['lr_decay'][0])(self.optimizer, **self.hp['lr_decay'][1])

        # 状态记录
        self.epoch = 0
        self.train_loss = 0.0

    def synchronize_with_server(self, server):
        # W_client = W_server
        self.copy(target=self.W, source=server.W)

    def train_cnn(self, iterations):

        running_loss = 0.0
        for i in range(iterations):

            try:  # Load new batch of data
                x, y = next(self.epoch_loader)
            except:  # Next epoch
                self.epoch_loader = iter(self.loader)
                self.epoch += 1

                # 动态调整lr
                if isinstance(self.scheduler, optim.lr_scheduler.LambdaLR):
                    self.scheduler.step()
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) and 'loss_test' in self.xp.results:
                    self.scheduler.step(self.xp.results['loss_test'][-1])

                x, y = next(self.epoch_loader)

            x, y = x.to(device), y.to(device)

            self.optimizer.zero_grad()

            y_ = self.model(x)

            loss = self.loss_fn(y_, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / iterations

    def compute_weight_update(self, iterations=1):

        # 设置为训练模式
        self.model.train()

        # W_old = W
        self.copy(target=self.W_old, source=self.W)

        # W = SGD(W, D)
        self.train_loss = self.train_cnn(iterations)

        # dW = W - W_old
        self.subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

    def compress_weight_update_up(self, compression=None, accumulate=False, count_bits=False):

        if accumulate and compression[0] != "none":
            # 超参数压缩，联邦通信优化
            self.add(target=self.A, source=self.dW)
            self.compress(target=self.dW_compressed, source=self.A)
            self.subtract(target=self.A, source=self.dW_compressed)

        else:
            # 没有任何压缩措施
            self.compress(target=self.dW_compressed, source=self.dW, )


class Server(DistributedTrainingDevice):
    """
    服务端类，继承分布式培训设备类
    """

    def __init__(self, dataloader, model, hyperparameters, experiment, stats):
        super().__init__(dataloader, model, hyperparameters, experiment)

        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_compressed = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.n_params = sum([T.numel() for T in self.W.values()])
        self.bits_sent = []

        self.client_sizes = torch.Tensor(stats["split"])

    def average(self, target, sources):
        """求超参数平均函数，平均值赋值在target中"""
        for name in target:
            target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()

    def aggregate_weight_updates(self, clients, aggregation="mean"):
        # dW = aggregate(dW_i, i=1,..,n)
        self.average(target=self.dW, sources=[client.dW_compressed for client in clients])

    def compress_weight_update_down(self, compression=None, accumulate=False, count_bits=False):
        if accumulate and compression[0] != "none":
            # 对超参数进行稀疏三元压缩
            self.add(target=self.A, source=self.dW)
            self.compress(target=self.dW_compressed, source=self.A)
            self.subtract(target=self.A, source=self.dW_compressed)

        else:
            self.compress(target=self.dW_compressed, source=self.dW)

        self.add(target=self.W, source=self.dW_compressed)

    def evaluate(self, loader=None, max_samples=50000, verbose=True):
        """评估服务端全局模型的训练效果"""
        self.model.eval()

        eval_loss, correct, samples, iters = 0.0, 0, 0, 0
        if not loader:
            loader = self.loader
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):

                x, y = x.to(device), y.to(device)
                y_ = self.model(x)
                _, predicted = torch.max(y_.data, 1)
                eval_loss += self.loss_fn(y_, y).item()
                correct += (predicted == y).sum().item()
                samples += y_.shape[0]
                iters += 1

                if samples >= max_samples:
                    break
            if verbose:
                print("Evaluated on {} samples ({} batches)".format(samples, iters))

            results_dict = {'loss': eval_loss / iters, 'accuracy': correct / samples}

        return results_dict
