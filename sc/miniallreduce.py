#!/usr/bin/env python
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


def average_gradients(sharetensor,group,mygroup1):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    dist.all_reduce(sharetensor, op=dist.reduce_op.SUM,group=group)
    # print("sharetensor",sharetensor.item(),"size", size)
    if dist.get_rank() in mygroup1:
         sharetensor /= size
    print("sharetensor",sharetensor.item())

def butterflyallreduce(sharetensor):
    # g1 = dist.new_group([0, 1])
    # g2 = dist.new_group([2, 3])
    # g3 = dist.new_group([1, 2])
    # dist.all_reduce(sharetensor,op=dist.reduce_op.SUM,group=g1)
    # dist.all_reduce(sharetensor, op=dist.reduce_op.SUM, group=g2)
    # dist.all_reduce(sharetensor, op=dist.reduce_op.SUM, group=g3)
    # dist.broadcast(tensor=sharetensor,src=1,group=g1)
    # dist.broadcast(tensor=sharetensor,src=2,group=g2)
    # sharetensor /= 4
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = sharetensor.clone()
    recv_buff = sharetensor.clone()
    accum = sharetensor.clone()
    left = ((rank - 1) + size) % size
    right = (rank + 1) % size
    if rank % 2 == 0:
        # Send send_buff
        dist.isend(send_buff, right)
        dist.recv(recv_buff, right)
        accum += recv_buff
        # print("rank: ",dist.get_rank(),"first accum is",accum.tolist())
    else:
        # Send recv_buff
        dist.isend(send_buff,left)
        dist.recv(recv_buff, left)
        accum += recv_buff
        # print("rank: ",dist.get_rank(),"first accum is",accum.tolist())
    if rank == 1:
        dist.isend(accum,3)
        dist.recv(recv_buff,3)
        accum += recv_buff
        # print("rank: ",dist.get_rank(),"second accum is",accum.tolist())
    if rank == 3:
        dist.isend(accum,1)
        dist.recv(recv_buff, 1)
        accum += recv_buff
        # print("rank: ",dist.get_rank(),"second accum is",accum.tolist())
    if rank == 0:
        dist.isend(accum, 2)
        dist.recv(recv_buff,2)
        accum += recv_buff
        # print("rank: ",dist.get_rank(),"second accum is",accum.tolist())
    if rank == 2:
        dist.recv(recv_buff,0)
        dist.isend(accum, 0)
        accum += recv_buff
        # print("rank: ",dist.get_rank(),"second accum is",accum.tolist())
    sharetensor = accum/4
    # print("rank: ", dist.get_rank(), "last accum is", sharetensor.tolist())
    return sharetensor

def getaverage(sharetensor):
    dist.all_reduce(sharetensor,op=dist.reduce_op.SUM)
    sharetensor /= 4

def allreduce(sharetensor):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = sharetensor.clone()
    recv_buff = sharetensor.clone()
    accum = sharetensor.clone()
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
    sharetensor = accum / size
    return sharetensor

def allreduce2(sharetensor):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = sharetensor.clone()
    recv_buff = sharetensor.clone()
    accum = sharetensor.clone()
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
        accum += recv_buff
        dist.isend(accum,0)
    if rank == 3:
        dist.isend(accum,1)
        dist.recv(recv_buff,1)
        accum += recv_buff
        dist.isend(accum,2)
    if rank == 0:
        dist.recv(recv_buff,1)
        accum = recv_buff
    if rank == 2:
        dist.recv(recv_buff,3)
        accum = recv_buff
    sharetensor = accum/4
    return sharetensor

def allreduce3(sharetensor):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = sharetensor.clone()
    recv_buff = sharetensor.clone()
    accum = sharetensor.clone()
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
    sharetensor = accum/4
    return sharetensor

def run(rank, size):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    # mygroup1 = [0, 1, 2, 3]
    # group = dist.new_group([0, 1, 2, 3])
    sharetensor = torch.FloatTensor([dist.get_rank()])
    # sharetensor = torch.rand(1000,1000)
    # print(sharetensor.tolist())
    begin = time.time()
    sharetensor = butterflyallreduce(sharetensor)
    end = time.time()
    print("method 1 answer is : ",sharetensor.tolist())
    print("this is rank ",dist.get_rank())
    print("method 1 spend time is",(end-begin)/60)
    sharetensor = torch.FloatTensor([dist.get_rank()])
    begin = time.time()
    getaverage(sharetensor)
    end = time.time()
    print("method 2 answer is : ",sharetensor.tolist())
    print("method 2 spend time is",(end-begin)/60)
    sharetensor = torch.FloatTensor([dist.get_rank()])
    begin = time.time()
    sharetensor = allreduce(sharetensor)
    end = time.time()
    # print("method 3 answer is : ",sharetensor.tolist())
    print("method 3 spend time is",(end-begin)/60)
    sharetensor = torch.FloatTensor([dist.get_rank()])
    begin = time.time()
    sharetensor = allreduce2(sharetensor)
    end = time.time()
    print("method 4 answer is : ",sharetensor.tolist())
    print("method 4 spend time is",(end-begin)/60)
    sharetensor = torch.FloatTensor([dist.get_rank()])
    begin = time.time()
    sharetensor = allreduce3(sharetensor)
    end = time.time()
    print("method 5 answer is : ",sharetensor.tolist())
    print("method 5 spend time is",(end-begin)/60)




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
