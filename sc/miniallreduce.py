#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math

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

# def butterflyallreduce(sharetensor):
#     rank = dist.get_rank()
#     size = dist.get_world_size()
#     send_buff = sharetensor.clone()
#     recv_buff = sharetensor.clone()
#     accum = sharetensor.clone()
#     left = ((rank - 1) + size) % size
#     right = (rank + 1) % size
#     if rank % 2 == 0:
#         # Send send_buff
#         dist.isend(send_buff, right)
#         dist.recv(recv_buff, right)
#         accum += recv_buff
#         # print("rank: ",dist.get_rank(),"first accum is",accum.tolist())
#     else:
#         # Send recv_buff
#         dist.isend(send_buff,left)
#         dist.recv(recv_buff, left)
#         accum += recv_buff
#         # print("rank: ",dist.get_rank(),"first accum is",accum.tolist())
#     if rank == 1:
#         dist.isend(accum,3)
#         dist.recv(recv_buff,3)
#         accum += recv_buff
#         # print("rank: ",dist.get_rank(),"second accum is",accum.tolist())
#     if rank == 3:
#         dist.isend(accum,1)
#         dist.recv(recv_buff, 1)
#         accum += recv_buff
#         # print("rank: ",dist.get_rank(),"second accum is",accum.tolist())
#     if rank == 0:
#         dist.isend(accum, 2)
#         dist.recv(recv_buff,2)
#         accum += recv_buff
#         # print("rank: ",dist.get_rank(),"second accum is",accum.tolist())
#     if rank == 2:
#         dist.recv(recv_buff,0)
#         dist.isend(accum, 0)
#         accum += recv_buff
#         # print("rank: ",dist.get_rank(),"second accum is",accum.tolist())
#     sharetensor = accum/4
#     # print("rank: ", dist.get_rank(), "last accum is", sharetensor.tolist())
#     return sharetensor

def butterflyallreduce(sharetensor,btflist):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = sharetensor.clone()
    recv_buff = sharetensor.clone()
    time = int(math.log2(size))
    for i in range(time):
        # Send send_buff
        a = dist.isend(send_buff, btflist[i][rank])
        dist.recv(recv_buff,  btflist[i][rank])
        send_buff = send_buff + recv_buff
        a.wait()
    # for i in range(time):
    #     for j in range(size):
    #         if j % 2**(i+1) == 0:
    #             for k in range(2**i):
    #                 if dist.get_rank() == (j+k):
    #                     dist.isend(send_buff, j+2**i+k)
    #                     dist.recv(recv_buff, +2**i+k)
    #                 if dist.get_rank() == (j+2**i+k):
    #                     dist.isend(send_buff, j+k)
    #                     dist.recv(recv_buff, j+k)
    send_buff /= size
            # print("rank ",rank,"data ",send_buff.tolist())
    # print("rank: ", dist.get_rank(), "last result is", send_buff.tolist())
    return send_buff


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

# def allreduce2(sharetensor):
#     rank = dist.get_rank()
#     size = dist.get_world_size()
#     send_buff = sharetensor.clone()
#     recv_buff = sharetensor.clone()
#     accum = sharetensor.clone()
#     left = ((rank - 1) + size) % size
#     right = (rank + 1) % size
#     if rank % 2 == 0:
#         # Send send_buff
#        dist.isend(send_buff, right)
#     else:
#         # Send recv_buff
#         dist.recv(recv_buff, left)
#         accum += recv_buff
#     if rank == 1:
#         dist.isend(accum,3)
#         dist.recv(recv_buff,3)
#         accum += recv_buff
#         dist.isend(accum,0)
#     if rank == 3:
#         dist.isend(accum,1)
#         dist.recv(recv_buff,1)
#         accum += recv_buff
#         dist.isend(accum,2)
#     if rank == 0:
#         dist.recv(recv_buff,1)
#         accum = recv_buff
#     if rank == 2:
#         dist.recv(recv_buff,3)
#         accum = recv_buff
#     sharetensor = accum/4
#     return sharetensor

# def allreduce3(sharetensor):
#     # print("555")
#     rank = dist.get_rank()
#     size = dist.get_world_size()
#     send_buff = sharetensor.clone()
#     recv_buff = sharetensor.clone()
#     accum = sharetensor.clone()
#     left = ((rank - 1) + size) % size
#     right = (rank + 1) % size
#     if rank % 2 == 0:
#         # Send send_buff
#         send_req = dist.isend(send_buff, right)
#         send_req.wait()
#     else:
#         # Send recv_buff
#         dist.recv(recv_buff, left)
#         accum += recv_buff
#     if rank == 1:
#         dist.isend(accum,3)
#         dist.recv(recv_buff,3)
#         accum = recv_buff
#         dist.isend(accum,0)
#     if rank == 3:
#         dist.recv(recv_buff,1)
#         accum += recv_buff
#         dist.isend(accum,1)
#         dist.isend(accum,2)
#     if rank == 0:
#         dist.recv(recv_buff,1)
#         accum = recv_buff
#     if rank == 2:
#         dist.recv(recv_buff,3)
#         accum = recv_buff
#     sharetensor = accum/4
#     # return sharetensor

def allreduce3(sharetensor):
    # print("555")
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = sharetensor.clone()
    recv_buff = sharetensor.clone()
    time = int(math.log2(size))
    for i in range(time):
        if rank % 2**(i+1) == 2**i - 1:
            # Send send_buff
            a = dist.isend(send_buff, rank + 2**i)
            a.wait()
        if rank % 2**(i+1) == 2**(i+1) - 1:
            dist.recv(recv_buff,rank - 2**i)
            send_buff += recv_buff
            # print("rank ",rank,"data ",send_buff.tolist())
    for i in range(time-1,-1,-1):
        if rank % 2**(i+1) == 2**(i+1) - 1:
            a = dist.isend(send_buff, rank - 2**i)
            a.wait()
        if rank % 2**(i+1) == 2**i - 1:
            # Send send_buff
            dist.recv(recv_buff,rank + 2**i)
            send_buff = recv_buff
    send_buff /= size
    # print("rank ",rank,"data ",send_buff.tolist())

def parameterserver(sharetensor):
    # print("555")
    rank = dist.get_rank()
    size = dist.get_world_size()
    if rank != 0:
        dist.isend(sharetensor,0)
    else:
        recv_buff = sharetensor.clone()
        for i in range(1,size):
            dist.recv(recv_buff,i)
            sharetensor += recv_buff
        sharetensor /= size
    dist.broadcast(sharetensor,src=0)
    # print("rank ",rank,"data ",sharetensor.tolist())







def run(rank, size):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    btflist= []
    n = dist.get_world_size()
    for i in range(int(math.log2(n))):
        D = {}
        for j in range(n):
            if j % 2**(i+1) == 0:
                for k in range(2**i):
                    D[j+k] = j+2**i+k
                    D[j+2**i+k] = j+k
        btflist.append(D)
    # print(str(btflist))
    # mygroup1 = [0, 1, 2, 3]
    # group = dist.new_group([0, 1, 2, 3])
    numlist = []
    # sharetensor = []
    # i = 21
    # a = 268435440
    a=240
    for i in range(21):
        numlist.append(a)
        a = a * 2 + 16
        # sharetensor = torch.rand(1000,1000)
        sharetensor = torch.rand(numlist[i],1)
        # # sharetensor = torch.FloatTensor([dist.get_rank()])
        begin = time.time()
        butterflyallreduce(sharetensor,btflist)
        end = time.time()
        # print("butterfly Allreduce answer is : ",sharetensor.tolist())
        # print("this is rank ",dist.get_rank())
        print("total number is ",dist.get_world_size(),"rank ",dist.get_rank(),"datasize is ",2**i,"kB ","butterfly Allreduce spend time is ",(end-begin)/60)

        # sharetensor = torch.rand(a, 1)
        sharetensor = torch.rand(numlist[i],1)
        begin = time.time()
        getaverage(sharetensor)
        end = time.time()
        # print("method 2 answer is : ",sharetensor.tolist())
        print("total number is ",dist.get_world_size(),"rank ",dist.get_rank(),"datasize is ",2**i,"kB ","Original Allreduce spend time is ",(end-begin)/60)


        # sharetensor = torch.rand(a,1)
        sharetensor = torch.rand(numlist[i], 1)
        begin = time.time()
        sharetensor = allreduce(sharetensor)
        end = time.time()
        # print("method 3 answer is : ",sharetensor.tolist())
        print("total number is ",dist.get_world_size(),"rank ",dist.get_rank(),"datasize is ",2**i,"kB ","Ring Allreduce spend time is ",(end-begin)/60)
        #
        # # sharetensor = torch.rand(a,1)
        # # begin = time.time()
        # # sharetensor = allreduce2(sharetensor)
        # # end = time.time()
        # # print("method 4 answer is : ",sharetensor.tolist())
        # # print("method 4 spend time is",(end-begin)/60)
        # # sharetensor = torch.FloatTensor([dist.get_rank()])



        sharetensor = torch.rand(numlist[i], 1)
        # sharetensor = torch.rand(a,1)
        # sharetensor = torch.FloatTensor([dist.get_rank()])
        begin = time.time()
        # sharetensor = allreduce3(sharetensor)
        # mynew = allreduce3(sharetensor)
        allreduce3(sharetensor)
        end = time.time()
        # print("method 5 answer is : ",sharetensor.tolist())
        # print("Recursive Allreduce spend time is ",(end-begin)/60)
        print("total number is ",dist.get_world_size(),"rank ",dist.get_rank(),"datasize is ",2**i,"kB ","Recursive Allreduce spend time is ",2*(end-begin)/60)

        sharetensor = torch.rand(numlist[i], 1)
        # sharetensor = torch.rand(a,1)
        # sharetensor = torch.FloatTensor([dist.get_rank()])
        begin = time.time()
        # sharetensor = allreduce3(sharetensor)
        # mynew = allreduce3(sharetensor)
        parameterserver(sharetensor)
        end = time.time()
        # print("method 5 answer is : ",sharetensor.tolist())
        # print("Recursive Allreduce spend time is ",(end-begin)/60)
        print("total number is ",dist.get_world_size(),"rank ",dist.get_rank(),"datasize is ",2**i,"kB ","Parameter Server spend time is ",(end-begin)/60)





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
