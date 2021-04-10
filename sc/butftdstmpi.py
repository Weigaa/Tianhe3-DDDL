import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """ Network architecture. """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        sfm = F.log_softmax(x,dim=1)
        return sfm
    # #和非分布式一致的网络
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 32, 3, 1)
    #     self.conv2 = nn.Conv2d(32, 64, 3, 1)
    #     self.dropout1 = nn.Dropout2d(0.25)
    #     self.dropout2 = nn.Dropout2d(0.5)
    #     self.fc1 = nn.Linear(9216, 128)
    #     self.fc2 = nn.Linear(128, 10)
    #
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = F.relu(x)
    #     x = self.conv2(x)
    #     x = F.relu(x)
    #     x = F.max_pool2d(x, 2)
    #     x = self.dropout1(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     x = self.dropout2(x)
    #     x = self.fc2(x)
    #     output = F.log_softmax(x, dim=1)
    #     return output


def partition_dataset():
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    print("进程的个数",size)
    print("每一批每个进程分得的训练集个数", bsz)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=int(bsz), shuffle=True)
    return train_set, bsz


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        # dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=0)
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def butterflyallreduce(model):
    g1 = dist.new_group([0, 1])
    g2 = dist.new_group([2, 3])
    g3 = dist.new_group([1, 2])
    for param in model.parameters():
        # dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=0)
        dist.all_reduce(param.grad.data,op=dist.reduce_op.SUM,group=g1)
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=g2)
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=g3)
        dist.broadcast(tensor=param.grad.data, src=1, group=g1)
        dist.broadcast(tensor=param.grad.data,src=2,group=g2)
        param.grad.data /= 4

def allreduce(model):
   rank = dist.get_rank()
   size = dist.get_world_size()
   for param in model.parameters():
        send_buff = param.grad.data.clone()
        recv_buff = param.grad.data.clone()
        accum = param.grad.data.clone()
        left = ((rank - 1) + size) % size
        right = (rank + 1) % size
        for i in range(size - 1):
            if i % 2 == 0:
                # Send send_buff
                send_req = dist.isend(send_buff, right)
                dist.recv(recv_buff, left)
                accum += recv_buff
            else:
                # Send recv_buff
                send_req = dist.isend(recv_buff, right)
                dist.recv(send_buff, left)
                accum += send_buff
            send_req.wait()
        param.grad.data = accum / size

def allreduce2(model):
    rank = dist.get_rank()
    size = dist.get_world_size()
    for param in model.parameters():
        send_buff = param.grad.data.clone()
        recv_buff = param.grad.data.clone()
        accum = param.grad.data.clone()
        left = ((rank - 1) + size) % size
        right = (rank + 1) % size
        if rank % 2 == 0:
            # Send send_buff
            dist.isend(send_buff, right)
        else:
            # Send recv_buff
            dist.recv(recv_buff, left)
            accum += recv_buff
        if rank == 1:
            dist.isend(accum, 3)
            dist.recv(recv_buff, 3)
            accum += recv_buff
            dist.isend(accum, 0)
        if rank == 3:
            dist.isend(accum, 1)
            dist.recv(recv_buff, 1)
            accum += recv_buff
            dist.isend(accum, 2)
        if rank == 0:
            dist.recv(recv_buff, 1)
            accum = recv_buff
        if rank == 2:
            dist.recv(recv_buff, 3)
            accum = recv_buff
        param.grad.data = accum / size

def allreduce3(model):
    rank = dist.get_rank()
    size = dist.get_world_size()
    for param in model.parameters():
        send_buff = param.grad.data.clone()
        recv_buff = param.grad.data.clone()
        accum = param.grad.data.clone()
        left = ((rank - 1) + size) % size
        right = (rank + 1) % size
        if rank % 2 == 0:
            # Send send_buff
           dist.isend(send_buff, right)
        else:
            # Send recv_buff
            dist.recv(recv_buff, left)
            accum += recv_buff
        if rank == 1:
            dist.isend(accum,3)
            dist.recv(recv_buff,3)
            accum = recv_buff
            dist.isend(accum,0)
        if rank == 3:
            dist.recv(recv_buff,1)
            accum += recv_buff
            dist.isend(accum,1)
            dist.isend(accum,2)
        if rank == 0:
            dist.recv(recv_buff,1)
            accum = recv_buff
        if rank == 2:
            dist.recv(recv_buff,3)
            accum = recv_buff
        param.grad.data = accum / size

def run(rank, size):
    """ Distributed Synchronous SGD Example """
    totalbegin = time.clock()
    totalbegin2 = time.time()
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    model = model
#    model = model.cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    #lr代表学习率，momentum表示动量因子
    # v = mu * v - learning_rate * dw
    # w = w + v
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    print("训练集总个数",len(train_set.dataset))
    print("batch大小", num_batches)
    totalallreuduce = 0
    for epoch in range(10):
        epoch_loss = 0.0
        begin = time.clock()
        begin2 = time.time()
        #记录1个epoch执行了多少次all_reducue操作
        reducetime = 0
        #记录1个epoch执行all_reducue时间
        rt = 0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            rtbegin = time.time()
            allreduce2(model)
            rtend = time.time()
            rt += (rtend-rtbegin)/60
            reducetime += 1
            optimizer.step()
        end = time.clock()
        end2 = time.time()
        spendtime = (end-begin)/60
        spendtime2 = (end2 - begin2) / 60
        totalallreuduce += rt
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches,', spend process time: ',spendtime,
              'spend real time: ', spendtime2,'all-reduce times: ',reducetime,
              'all-reduce spend time: ',rt)
        # print('Rank ',
        #       dist.get_rank(), ', epoch ', epoch, ': ',
        #       epoch_loss / num_batches,', spend process time: ',spendtime,
        #       'spend real time: ', spendtime2,file=f)
    totalend =time.clock()
    totalend2 = time.time()
    print("total processrun time: ",(totalend-totalbegin)/60)
    print("total realrun time: ", (totalend2 - totalbegin2) / 60)
    print("total allreduce time: ", totalallreuduce)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data',
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.1307,
                         ),
                        (0.3081,
                         ))])),
        batch_size=32,
        shuffle=True,
    )
    get_accuracy(test_loader, model)

def get_accuracy(test_loader, model):
    model.eval()
    correct_sum = 0
    # Use GPU to evaluate if possible
    device = torch.device("cpu")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum = correct_sum + correct
    print("Accuracy:", correct_sum / len(test_loader.dataset))


def init_processes(rank, size, fn, backend='mpi'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    print("程序开始执行")
    print("all process BEGINtime is : ", time.strftime('%Y-%m-%d %H:%M:%S'))
    init_processes(0, 0, run, backend='mpi')
    print("all process ENDtime is : ", time.strftime('%Y-%m-%d %H:%M:%S'))

