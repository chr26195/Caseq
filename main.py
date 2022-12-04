import os
import time
import torch
import argparse

from model import Casseq
from tqdm import tqdm
from utils import *
from sampler import *
import pickle

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', default='default', type=str)
parser.add_argument('--num_epochs', default=301, type=int)
parser.add_argument('--train_mode', default='uni', type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--eval_every', default=10, type=int)
parser.add_argument('--save_every', default=20, type=int)
parser.add_argument('--test_limit', default=1000, type=int)
parser.add_argument('--save_epoches', default=[50], type=int, nargs='+')
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--lapse', default=30, type=int)

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=0, type=float)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)

parser.add_argument('--user_D', default=25, type=int) # only for ssept
parser.add_argument('--sse_prob', default=0.9, type=float) # only for ssept
parser.add_argument('--num_context', default=3, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--kl', default=0.0, type=float)
parser.add_argument('--N', default=10, type=int) # number of psuedo sequences
parser.add_argument('--backbone', default='att', type=str)
parser.add_argument('--gumbel', action='store_true')

# parser.add_argument('--personalized_gate', default=False, type=str2bool)
# parser.add_argument('--att_independence', default=False, type=str2bool)
# parser.add_argument('--sse_prob', default=0.5, type=float)
# parser.add_argument('--inde_threshol', default=0.3, type=float)

args = parser.parse_args()
fix_seed(args.seed)


if not os.path.isdir('results/' + args.dataset + '_' + args.train_dir):
    os.makedirs('results/' + args.dataset + '_' + args.train_dir)
with open(os.path.join('results/' + args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()
print('using device {}'.format(args.device))

dataset, dataset2 = data_partition(args.dataset, args.lapse)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
[user_train2, user_valid2, user_test2, usernum, itemnum] = dataset2
num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)

item_rated_num = load_item_pop(args.dataset)
item_pop = np.array(list(item_rated_num.values()), dtype=np.float)
item_pop /= np.sum(item_pop)

cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join('results/' + args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

sampler = WarpSampler(user_train2, usernum, itemnum, item_pop, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, train_mode=args.train_mode)
model = Casseq(usernum, itemnum, args).to(args.device) 

for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except:
        pass # just ignore those failed init layers

model.train() # enable model training

epoch_start_idx = 1
if args.state_dict_path is not None:
    try:
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1
    except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
        print('failed loading state_dicts, pls check file path: ', end="")
        print(args.state_dict_path)
        print('pdb enabled for your quick check, pls type exit() if you do not need it')
        import pdb; pdb.set_trace()

bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')
adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))

T = 0.0
t0 = time.time()
valid_bestNDCG, valid_bestHR = 0., 0.

bestNDCG, bestHR = 0., 0.
bestNDCG2, bestHR2 = 0., 0.

for epoch in range(epoch_start_idx, args.num_epochs + 1):
    
    if args.inference_only: break # just to decrease identition
    for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        u, seq, pos, neg = sampler.next_batch() 
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        pos_logits, neg_logits, weights = model(u, seq, pos, neg)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
        
        adam_optimizer.zero_grad()
        
        indices = np.where(pos != 0)
        loss_pred = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss_pred += bce_criterion(neg_logits[indices], neg_labels[indices])
        loss_kl = model.kl_loss(weights, indices) * args.kl
        
        if step == num_batch-1:
            if epoch % args.eval_every == 0:
                f.write(str(weights[0][-1])+'\n')

        # if epoch == args.num_epochs:
        #     with open('weights-{}-{}.pkl'.format(args.dataset,args.backbone),'wb') as f:
        #         pickle.dump(weights.cpu().detach().numpy(), f)
        
        if args.kl == 0.0: loss = loss_pred
        else: loss = loss_pred + loss_kl
        loss.backward()
        
        adam_optimizer.step()
        if step == num_batch-1:
            print("loss in epoch {} iteration {}: loss_pred {}".format(epoch, step, loss_pred.item()))
            # print("loss in epoch {} iteration {}: loss_pred {}, loss_kl {}".format(epoch, step, loss_pred.item(), loss_kl.item()))
            
    if epoch % args.eval_every == 0:
        
        if usernum>args.test_limit:
            users = random.sample(range(1, usernum + 1), args.test_limit)
        else:
            users = range(1, usernum + 1)
            
        model.eval()
        print('Evaluating', end='')
        
        t_test = evaluate(model, item_pop, dataset, users, args)
        t_test2 = evaluate(model, item_pop, dataset2, users, args)
        t_valid = evaluate_valid(model, item_pop, dataset2, users, args)
        
        if t_valid[0] > valid_bestNDCG:
            valid_bestNDCG = t_valid[0]
            bestNDCG = t_test[0]
            bestNDCG2 = t_test2[0]
            bestHR = t_test[1]
            bestHR2 = t_test2[1]

        t1 = time.time() - t0
        T += t1
        
        print('\nepoch:%d, time: %f(s), valid (NDCG: %.4f, HR: %.4f), test_wo_lap (NDCG: %.4f, HR: %.4f), test_lap (NDCG: %.4f, HR: %.4f)'
                % (epoch, T, t_valid[0], t_valid[1], t_test2[0], t_test2[1], t_test[0], t_test[1]))
        
        f.write('epoch:%d, time: %f(s), valid (NDCG: %.4f, HR: %.4f), test_wo_lap (NDCG: %.4f, HR: %.4f), test_lap (NDCG: %.4f, HR: %.4f)\n'
                % (epoch, T, t_valid[0], t_valid[1], t_test2[0], t_test2[1], t_test[0], t_test[1]))
        
        f.flush()
        t0 = time.time()
        model.train()

    # if epoch in args.save_epoches or epoch == args.num_epochs:
    #     folder = 'results/' + args.dataset + '_' + args.train_dir
    #     fname = 'Casseq.epoch={}.pth'
    #     fname = fname.format(epoch)
    #     torch.save(model.state_dict(), os.path.join(folder, fname))

f.write('lapse: best NDCG: %.4f, best HR: %.4f \n' % (bestNDCG, bestHR))
f.write('no lapse: best NDCG: %.4f, best HR: %.4f \n' % (bestNDCG2, bestHR2))
f.write('dec: NDCG: %.2f, HR: %.2f \n' % ((bestNDCG2-bestNDCG)/bestNDCG2*100, (bestHR2-bestHR)/bestHR2*100))

f.close()
sampler.close()
print("Done")
