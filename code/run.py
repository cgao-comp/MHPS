from utils.Constants import Logger, Options
from model.MHPS import MHPS
from utils.Optim import ScheduledOptim
from utils.Metrics import Metrics
from utils.DataConstruct import DataConstruct
import utils.Constants as Constants
import torch.nn as nn
import torch
import argparse
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm
import os
from torch_geometric.data import Data

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
np.set_printoptions(threshold=np.inf)


def best_result(scores, k_list=[10,50,100], weights=[1,1,1,1,1,1]):
    result = 0.0
    i=0
    for k in k_list:
        #s = scores["hits@"+str(k)]
        #w = weights[i]
        result += weights[i]*scores["hits@"+str(k)]
        result += weights[i+1] * scores["map@" + str(k)]
        i+=2
    return result


def SeedEverything(SEED):

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(False)
    print(f"Init_every_seed with {SEED}")


root_path = './'
parser = argparse.ArgumentParser()
## Training Hypers
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument("--data", type=str, default="douban", metavar='dataname', help='dataset name')
parser.add_argument('--batch_size', type=int, default=16,metavar='BS', help='batch size')
parser.add_argument('--dropout', type=float, default=0.15, metavar='dropout',help='dropout rate')
parser.add_argument('--smooth', type=float, default=0.1, help='Lable Smooth rate for the model')
parser.add_argument('--seed', type=int, default=2023, help='random state seed')
parser.add_argument('--warmup', type=int, default=10)  # warmup epochs
parser.add_argument('--n_warmup_steps', type=int, default=1000, metavar='LR',
                    help='the warmup steps in the model')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2',
                    help='L2 regularization weight')
parser.add_argument('--norm', type=bool, default=True, metavar='Norm',
                    help='Need norm')

## Module Hypers
# time representation
parser.add_argument('--time_encoder_type', type=str, default="Interval",
                    choices=['Interval', 'Neural', 'Slide', 'None'],
                    help='The decoder of the model') # time Encoder

parser.add_argument('--time_encoder_type_gru', type=str, default="Interval",
                    choices=['Interval', 'Neural', 'Slide', 'None'],
                    help='The decoder of the model') # time Encoder

## Model Hypers 
parser.add_argument('--d_model', type=int, default=64, metavar='inputF',
                    help='dimension of initial features.')
parser.add_argument('--time_dim', type=int, default=8, metavar='time', help='The dim of the time encoder')
parser.add_argument('--pos_dim', type=int, default=8, metavar='pos', help='The dim of the positional embedding')
parser.add_argument('--heads', type=int, default=1,help='number of heads in transformer')
parser.add_argument('--time_interval', type=int, default=10000,
                    help='the time interval for each time slice')
parser.add_argument('--graph_type', default="social+diffusion+item",
                    choices=['social', 'diffusion', 'item', 'social+diffusion', 'social+item', 'diffusion+item', 'social+diffusion+item'],
                    help='set the edges in the heterogenerous graph type.')
# parser.add_argument('--graph_type', default="diffusion+item", help='set the edges in the heterogenerous graph type.')

# Logging Hypers
parser.add_argument('--save_path', default=None)
parser.add_argument('--save_mode', type=str,
                    choices=['all', 'best'], default='best')
parser.add_argument('--notes', default="",help='lets take some notes for the model.')
parser.add_argument('--export_log', type=bool, default=True,help='copy the log to a certain file.')


opt = parser.parse_args()
opt.d_word_vec = opt.d_model
# 去掉时间维度
opt.transformer_dim = opt.d_model + opt.pos_dim
opt.notes = "MHPS"
if opt.save_path is None:
    opt.save_path = root_path + f"checkpoints/MHPS_{opt.data}_{int(time.time())}.pt"
#opt.save_path = root_path + f"checkpoints/MHPS_android.pt"
print(opt)
SeedEverything(opt.seed)

metric = Metrics()
data_path = opt.data

def get_performance(crit, pred, gold):
    ''' Apply label smoothing 
    if needed '''
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    # print ("get performance, ", gold.data, pred.data)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()

    true_set = set()
    for items in gold.cpu().numpy().tolist():
        true_set.add(items)
    pre_set = set()
    for item in pred.cpu().numpy().tolist():
        if item in true_set:
            pre_set.add(item)

    return loss, n_correct, len(pre_set), len(true_set)



def train_epoch(model, training_data, loss_func, optimizer,epoch):
    ''' Epoch operation in training phase'''
    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    total_same_user = 0.0
    n_total_uniq_user = 0.0
    batch_num = 0.0

    #for i, batch in tqdm(
    #        enumerate(training_data), mininterval=2,
    #        desc='  - (Training)   ', leave=False):

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
    #for i, batch in enumerate(
            #training_data):  # tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # prepare data
        tgt, tgt_timestamp, tgt_id = batch
        tgt.cuda(3)
        tgt_timestamp.cuda(3)
        user_gold = tgt[:, 1:].cuda(3)
        time_gold = tgt_timestamp[:, 1:].cuda(3)

        # start_time = time.time()

        n_words = user_gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        batch_num += tgt.size(0)

        optimizer.zero_grad()

        user_pred, diff_loss = model(tgt, tgt_timestamp, tgt_id,train=True)

        # backward
        loss, n_correct, same_user, input_users = get_performance(loss_func, user_pred, user_gold)
        loss = loss + diff_loss
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate(epoch)

        # note keeping
        n_total_correct += n_correct
        total_loss = total_loss + loss.item()

        total_same_user += same_user
        n_total_uniq_user += input_users

        #print("Training batch ", i, " loss: ", loss.item(), " acc:", (n_correct.item() / len(user_pred)), f"\t\toutput_users:{(same_user)}/{(input_users)}={same_user / input_users}", )

    return total_loss / n_total_words, n_total_correct / n_total_words, total_same_user / n_total_uniq_user


def test_epoch(model, validation_data, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    for batch in tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
        #print("Validation batch ", i)
        # prepare data
        # print(batch)

        tgt, tgt_timestamp, tgt_id = batch
        tgt.cuda(3)
        tgt_timestamp.cuda(3)


        user_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

        user_pred, diff_loss = model(tgt, tgt_timestamp, tgt_id,train=False)

        user_pred = user_pred.detach().cpu().numpy()

        scores_batch, scores_len,MRR = metric.compute_metric(user_pred, user_gold, k_list)

        n_total_words += scores_len
        for k in k_list:
            scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
            scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores, MRR


'''
Part of this function is derived from
https://github.com/slingling/MS-HGAT
'''

def CascadeHypergraph(cascades):
    # cascades = cascades.tolist()
    hyper_edge_list = []
    hyper_edge_indice = []
    counter = 0
    for cascade in cascades:
        cascade = set(cascade)
        cascade.discard(0)
        if len(cascade) > 1:
            hyper_edge_list += list(cascade)
            hyper_edge_indice += [counter]*len(cascade)
            counter += 1

    hyper_edge = torch.tensor([hyper_edge_list, hyper_edge_indice])
    #cascade_hypergraph = dhg.Hypergraph(user_size, edge_list, device=device)
    cascade_hypergraph = Data(edge_index= hyper_edge)
    return cascade_hypergraph

def DynamicCasHypergraph(examples, examples_times, step_split=2):
    '''
    :param examples: 级联（用户）
    :param examples_times: 级联时间戳（用户参与级联的时间）
    :param user_size: 数据集中的所有用户
    :param device: 所在设备
    :param step_split: 划分几个超图
    :return: 超图序列
    '''

    hypergraph_list = []
    time_sorted = []
    for time in examples_times:
        time_sorted += time[:-1]
    time_sorted = sorted(time_sorted)   # 将所有时间戳升序排列
    split_length = len(time_sorted) // step_split    # 一个时间段包含的时间戳个数
    start_time = 0
    end_time = 0

    for x in range(split_length, split_length * step_split, split_length):
        # if x == split_length:
        #     end_time = time_sorted[x]
        # else:
        #     end_time = time_sorted[x]
        start_time = end_time
        end_time = time_sorted[x]

        selected_examples = []
        for i in range(len(examples)):
            example = examples[i]
            example_times = examples_times[i]
            if isinstance(example, list):
                example = torch.tensor(example)
                example_times = torch.tensor(example_times, dtype=torch.float64)
            selected_example = torch.where((example_times < end_time) & (example_times > start_time), example, torch.zeros_like(example))
            selected_examples.append(selected_example.numpy().tolist())

        sub_hypergraph = CascadeHypergraph(selected_examples)
        # print(sub_hypergraph)
        hypergraph_list.append(sub_hypergraph)

    # =============== 最后一张超图 ===============
    start_time = end_time
    selected_examples = []
    for i in range(len(examples)):
        example = examples[i]
        example_times = examples_times[i]
        if isinstance(example, list):
            example = torch.tensor(example)
            example_times = torch.tensor(example_times, dtype=torch.float64)
        selected_example = torch.where(example_times > start_time, example, torch.zeros_like(example))
        # print(selected_example)
        selected_examples.append(selected_example.numpy().tolist())
    hypergraph_list.append(CascadeHypergraph(selected_examples))
    print("超图数量：", len(hypergraph_list))

    return hypergraph_list

def DynamicCasHypergraph_micro(examples, examples_times, step_split=8):
    '''
    :param examples: 级联（用户）
    :param examples_times: 级联时间戳（用户参与级联的时间）
    :param user_size: 数据集中的所有用户
    :param device: 所在设备
    :param step_split: 划分几个超图
    :return: 超图序列
    '''

    hypergraph_list = []
    time_sorted = []
    for time in examples_times:
        time_sorted += time[:-1]
    time_sorted = sorted(time_sorted)   # 将所有时间戳升序排列
    split_length = len(time_sorted) // step_split    # 一个时间段包含的时间戳个数
    start_time = 0
    end_time = 0

    for x in range(split_length, split_length * step_split, split_length):
        # if x == split_length:
        #     end_time = time_sorted[x]
        # else:
        #     end_time = time_sorted[x]
        start_time = end_time
        end_time = time_sorted[x]

        selected_examples = []
        for i in range(len(examples)):
            example = examples[i]
            example_times = examples_times[i]
            if isinstance(example, list):
                example = torch.tensor(example)
                example_times = torch.tensor(example_times, dtype=torch.float64)
            selected_example = torch.where((example_times < end_time) & (example_times > start_time), example, torch.zeros_like(example))
            selected_examples.append(selected_example.numpy().tolist())

        sub_hypergraph = CascadeHypergraph(selected_examples)
        # print(sub_hypergraph)
        hypergraph_list.append(sub_hypergraph)
    # =============== 最后一张超图 ===============
    start_time = end_time
    selected_examples = []
    for i in range(len(examples)):
        example = examples[i]
        example_times = examples_times[i]
        if isinstance(example, list):
            example = torch.tensor(example)
            example_times = torch.tensor(example_times, dtype=torch.float64)
        selected_example = torch.where(example_times > start_time, example, torch.zeros_like(example))
        # print(selected_example)
        selected_examples.append(selected_example.numpy().tolist())
    hypergraph_list.append(CascadeHypergraph(selected_examples))
    print("超图数量：", len(hypergraph_list))

    return hypergraph_list







def train_model(data_path):
    # ========= Preparing DataLoader =========#

    train_data = DataConstruct(data_path, data=0, load_dict=True, batch_size=opt.batch_size, cuda=False, seed=opt.seed)
    valid_data = DataConstruct(data_path, data=1, batch_size=opt.batch_size, cuda=False, seed=opt.seed)  # torch.cuda.is_available()
    test_data = DataConstruct(data_path, data=2, batch_size=opt.batch_size, cuda=False, seed=opt.seed)

    total_cascades = train_data._train_cascades + valid_data._valid_cascades + test_data._test_cascades
    total_time_stamps  = train_data._train_cascades_timestamp + valid_data._valid_cascades_timestamp + test_data._test_cascades_timestamp

    hypergraph_list = DynamicCasHypergraph(total_cascades, total_time_stamps, step_split=5)
    hypergraph_list_micro = DynamicCasHypergraph_micro(total_cascades, total_time_stamps, step_split=10)


    opt.user_size = train_data.user_size
    opt.ntoken = train_data.ntoken

    # ========= Preparing Model =========#
    opt.data_path = data_path
    opt.norm = train_data.need_norm
    model = MHPS(opt, hypergraph_list = hypergraph_list, hypergraph_list_micro = hypergraph_list_micro)

    #print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

   
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizerAdam = torch.optim.Adam(params, betas=(
        0.9, 0.98), eps=1e-09, weight_decay=opt.l2)  # weight_decay is l2 regularization
    optimizer = ScheduledOptim(
        optimizerAdam, opt.d_model, opt.n_warmup_steps, data_path)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerAdam, 'min', factor=0.4, patience=7, verbose=True)

    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)

    if torch.cuda.is_available():
        model = model.cuda(3)
        loss_func = loss_func.cuda(3)

    validation_history = 0.0
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_accu, train_pred = train_epoch(model, train_data, loss_func, optimizer,epoch_i)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, predected:{pred:3.3f} %' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu, pred=100 * train_pred,
            elapse=(time.time() - start) / 60), flush=True)

        if epoch_i >= 0:
            start = time.time()
            scores,MRR = test_epoch(model, valid_data)
            print('  - ( Validation )) ')
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            print("Validation use time: ", (time.time() - start) / 60, "min")
            print(f"MRR: {MRR}")

            print('  - (Test) ')
            scores,MRR = test_epoch(model, test_data)
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            if validation_history <= best_result(scores):
                print("Best Validation at Epoch:{}".format(epoch_i))
                validation_history = best_result(scores)
                #print("Save best model!!!")
                torch.save(model.state_dict(), opt.save_path)
                print(f"MRR: {MRR}")            
        scheduler.step(validation_history)

    

def test_model(data_path):
    train_data = DataConstruct(data_path, data=0, load_dict=True, batch_size=opt.batch_size, cuda=False)
    valid_data = DataConstruct(data_path, data=1, batch_size=opt.batch_size, cuda=torch.cuda.is_available())
    test_data = DataConstruct(data_path, data=2, batch_size=opt.batch_size, cuda=False)

    total_cascades = train_data._train_cascades + valid_data._valid_cascades + test_data._test_cascades
    total_time_stamps = train_data._train_cascades_timestamp + valid_data._valid_cascades_timestamp + test_data._test_cascades_timestamp
    hypergraph_list = DynamicCasHypergraph(total_cascades, total_time_stamps, step_split=5)
    hypergraph_list_micro = DynamicCasHypergraph_micro(total_cascades, total_time_stamps, step_split=10)

    opt.user_size = train_data.user_size
    opt.ntoken = train_data.ntoken

    # ========= Preparing Model =========#
    opt.data_path = data_path
    opt.norm = train_data.need_norm
    model = MHPS(opt, hypergraph_list = hypergraph_list, hypergraph_list_micro = hypergraph_list_micro)
    opt.user_size = train_data.user_size
    opt.ntoken = train_data.ntoken
    opt.data_path = data_path
    opt.norm = train_data.need_norm

    model.load_state_dict(torch.load(opt.save_path), strict=False)
    model.cuda(3)
    scores,MRR= test_epoch(model, test_data)
    print('  - (Test) ')
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))
    print(f"MRR: {MRR}")
    return scores







if __name__ == "__main__":


    hits10 = []
    hits50 = []
    hits100 = []
    map10 = []
    map50 = []
    map100 = []

    for i in range(1):
        train_model(data_path)
        scores = test_model(data_path)
        hits10.append(scores['hits@10'])
        hits50.append(scores['hits@50'])
        hits100.append(scores['hits@100'])
        map10.append(scores['map@10'])
        map50.append(scores['map@50'])
        map100.append(scores['map@100'])

    print('Hits10', np.mean(hits10), np.std(hits10))
    print('Hits50', np.mean(hits50), np.std(hits50))
    print('Hits100', np.mean(hits100), np.std(hits100))
    print('MAP10', np.mean(map10), np.std(map10))
    print('MAP50', np.mean(map50), np.std(map50))
    print('MAP100', np.mean(map100), np.std(map100))
