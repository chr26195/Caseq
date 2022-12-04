import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import pickle

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def pop_neg(l, r, s, item_pop):
    t = np.random.choice(np.arange(l, r), p = item_pop)
    while t in s:
        t = np.random.choice(np.arange(l, r), p = item_pop)
    return t

def pop_neg_seq(itemnum, item_pop, maxlen, rated):
    candidates = np.arange(1, itemnum+1)
    temp_smpl = np.random.choice(candidates,  maxlen + 50, replace=True, p=item_pop)
    samples = [t for t in temp_smpl if t not in rated]
    while len(samples) < maxlen:
        temp_smpl = np.random.choice(candidates, maxlen + 50, replace=True, p=item_pop)
        samples.extend([t for t in temp_smpl if t not in rated and t not in samples])
    samples = samples[:maxlen]
    return np.array(samples, dtype=np.int32)

def uni_neg_seq(itemnum, item_pop, maxlen, rated):
    candidates = np.arange(1, itemnum+1)
    temp_smpl = np.random.choice(candidates,  maxlen + 50, replace=True)
    samples = [t for t in temp_smpl if t not in rated]
    while len(samples) < maxlen:
        temp_smpl = np.random.choice(candidates, maxlen + 50, replace=True)
        samples.extend([t for t in temp_smpl if t not in rated and t not in samples])
    samples = samples[:maxlen]
    return np.array(samples, dtype=np.int32)
    
# sample in training
def sample_function(user_train, usernum, itemnum, item_pop, batch_size, maxlen, result_queue, SEED, train_mode):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        # neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1
        ts = set(user_train[user])
        
        if train_mode == 'pop':
            neg = pop_neg_seq(itemnum, item_pop, maxlen, ts)
        elif train_mode == 'uni':
            neg = uni_neg_seq(itemnum, item_pop, maxlen, ts)

        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt == 0:
                neg[idx] = 0
            
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, item_pop, batch_size=64, maxlen=10, n_workers=1, train_mode='uni'):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      item_pop,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      train_mode
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()