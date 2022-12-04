import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import pickle

# train/val/test data generation
def data_partition(fname, lapse):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_train2 = {}
    user_valid = {}
    user_valid2 = {}
    user_test = {}
    user_test2 = {}
    f = open('/home/yangchenxiao/sas_data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3 + lapse:
            user_train[user] = User[user]
            user_train2[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
            user_valid2[user] = []
            user_test2[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_train2[user] = User[user][:-(2+lapse)]
            
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            
            user_valid2[user] = []
            user_valid2[user].append(User[user][-(2+lapse)])
            
            user_test[user] = []
            user_test[user].append(User[user][-1])
            
            user_test2[user] = []
            user_test2[user].append(User[user][-(1+lapse)])
    return [user_train, user_valid, user_test, usernum, itemnum], [user_train2,user_valid2, user_test2, usernum, itemnum]


def load_item_pop(fname):
    with open('/home/yangchenxiao/sas_data/%s-ihis.pkl' % fname, 'rb') as f:
        item_his = pickle.load(f)
    item_rated_num = {}
    for i in item_his.keys():
        item_rated_num[i+1] = len(item_his[i])
    return item_rated_num

# TODO: unify test and val
# evaluate on test set
def evaluate(model, item_pop, dataset, users, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        
        candidates = np.arange(1, itemnum+1)
        temp_smpl = np.random.choice(candidates, 150, replace=False, p=item_pop)
        samples = [t for t in temp_smpl if t not in rated]
        while len(samples) < 100:
            temp_smpl = np.random.choice(candidates, 150, replace=False, p=item_pop)
            samples.extend([t for t in temp_smpl if t not in rated and t not in samples])
        samples = samples[:100]
        item_idx.extend(samples)
        
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]]) # user_ids, log_seqs, item_indices
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, item_pop, dataset, users, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]

        candidates = np.arange(1, itemnum+1)
        temp_smpl = np.random.choice(candidates, 150, replace=False, p=item_pop)
        samples = [t for t in temp_smpl if t not in rated]
        while len(samples) < 100:
            temp_smpl = np.random.choice(candidates, 150, replace=False, p=item_pop)
            samples.extend([t for t in temp_smpl if t not in rated and t not in samples])
        samples = samples[:100]
        item_idx.extend(samples)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user