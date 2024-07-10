import os
import sys
import dill
import time
import logging
import argparse
import copy
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
from torch.optim import Adam

sys.path.append("..")
sys.path.append("../..")

from utils.util import create_log_id, logging_config, multi_label_metric, get_n_params
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from models.GBert import BertConfig, GBERT_Pretrain


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default='pretrain', help="User notes")
    parser.add_argument('--model_name', type=str, default='GBert', help="model name")
    parser.add_argument('--dataset', type=str, default='mimic-iii', help='dataset')

    parser.add_argument("--batch_size", default=64, type=int, help="Total batch size for training.")

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--therhold", default=0.3, type=float, help="therhold.")

    parser.add_argument('--early_stop', type=int, default=10, help='early stop after this many epochs without improvement')
    parser.add_argument('--cuda', type=int, default=0 , help='which cuda')  #TODO 记得改cuda选项
    parser.add_argument("--num_train_epochs", default=100.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=1203, help="random seed for initialization")
    parser.add_argument('--acc_name', default='jaccard', help="evaluation metric")
    
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of node embedding(randomly initialize)')

    args = parser.parse_args()
    return args

def random_mask_word(seq, vocab):
    mask_idx = vocab.word2idx['[MASK]']
    for i, _ in enumerate(seq):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                seq[i] = mask_idx
            # 10% randomly change token to random token
            elif prob < 0.9:
                seq[i] = random.choice(list(vocab.word2idx.items()))[1]
            else:
                pass
        else:
            pass
    return seq

def add_sentence(voc, sentence):
    for word in sentence:
        # if word not in voc.word2idx:
        voc.idx2word[len(voc.word2idx)] = word
        voc.word2idx[word] = len(voc.word2idx)

def fill_to_max(l, voc, seq_len):
    pad_idx = voc.word2idx['[PAD]']
    while len(l) < seq_len:
        l.append(pad_idx)
    return l

class EHRDataset(Dataset):
    def __init__(self, data, voc, voc_size, max_seq_len):
        self.data = data
        self.voc = voc
        self.diag_voc, self.med_voc = voc['diag_voc'], voc['med_voc']
        self.voc_size = voc_size
        self.max_seq_len = max_seq_len
    
    def __getitem__(self, index):
        adm = copy.deepcopy(self.data[index]) # 只用有一次visit的病人pretrain
        y_dx = np.zeros(self.voc_size[0])
        y_rx = np.zeros(self.voc_size[2])
        y_dx[adm[0]] = 1
        y_rx[adm[2]] = 1
        adm[0] = random_mask_word(adm[0], self.diag_voc)
        adm[2] = random_mask_word(adm[2], self.med_voc)
        input_ids = []
        input_ids.extend([self.diag_voc.word2idx['[CLS]']] + fill_to_max(adm[0], self.diag_voc, self.max_seq_len-1))
        input_ids.extend([self.med_voc.word2idx['[CLS]']] + fill_to_max(adm[2], self.med_voc, self.max_seq_len-1))
        cur_tensors = (torch.tensor(input_ids, dtype=torch.long).view(-1, self.max_seq_len), # [2, max_seq_len]
                        torch.tensor(y_dx, dtype=torch.float), # [diag_voc_size] one-hot
                        torch.tensor(y_rx, dtype=torch.float)) # [med_voc_size] one-hot
        return cur_tensors

    def __len__(self):
        return len(self.data)

def t2n(x):
    return x.detach().cpu().numpy()

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def main(args):
    # set logger
    log_directory_path = os.path.join('../log', args.dataset, args.model_name)
    # os.makedirs(log_directory_path, exist_ok=True)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log'+str(log_save_id)+'_'+args.note)
    os.makedirs(save_dir, exist_ok=True)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    logging.info(f'model_name={args.model_name}, dataset={args.dataset}, lr={args.lr}, '
    f'therhold={args.therhold}, save_dir={log_save_id}')
    
    # load data
    data_path = f'../../data/output/{args.dataset}' + '/records_final.pkl'
    voc_path = f'../../data/output/{args.dataset}' + '/voc_final.pkl'
    
    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    
    special_tokens=["[PAD]", "[CLS]", "[MASK]"]
    add_sentence(diag_voc, special_tokens)
    add_sentence(pro_voc, special_tokens)
    add_sentence(med_voc, special_tokens)
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    # 统计
    max_diag_len = 0
    max_med_len = 0
    for idx, patient in enumerate(data):
        for adm in patient:
            if len(adm[0]) > max_diag_len:
                max_diag_len = len(adm[0])
            if len(adm[2]) > max_med_len:
                max_med_len = len(adm[2])
    print(max_diag_len)  # [39, 32, 52]
    print(max_med_len) 
    # print(len(data)) # 635
    # print(len(single_idx)) # 908
    
    max_seq_len = max([max_diag_len, max_med_len]) + 2
    print(max_seq_len) # 54
    
    data = [visit for patient in data for visit in patient]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_val = data[split_point:split_point + eval_len]
    data_test = data[split_point+eval_len:]

    data_train = EHRDataset(data_train, voc, voc_size, max_seq_len)
    data_val = EHRDataset(data_val, voc, voc_size, max_seq_len)
    data_test = EHRDataset(data_test, voc, voc_size, max_seq_len)

    train_dataloader = DataLoader(data_train,
                                  sampler=RandomSampler(data_train),
                                  batch_size=args.batch_size)
    eval_dataloader = DataLoader(data_val,
                                 sampler=SequentialSampler(data_val),
                                 batch_size=args.batch_size)
    test_dataloader = DataLoader(data_test,
                                 sampler=SequentialSampler(data_test),
                                 batch_size=args.batch_size)
    
    config = BertConfig(vocab_size_or_config_json_file=sum(voc_size))
    model = GBERT_Pretrain(config, diag_voc, med_voc) 
    logging.info(model)

    model.to(device)
    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = Adam(model.parameters(), lr=args.lr)
    print('parameters', get_n_params(model))

    EPOCH = int(args.num_train_epochs)

    logging.info("Num train samples = %d", len(data_train))
    
    acc_name = args.acc_name
    best_epoch = 0
    dx_acc_best, rx_acc_best = 0, 0
    for epoch in range(EPOCH):
        epoch += 1
        print(f'\nepoch {epoch} --------------------------model_name={args.model_name}, dataset={args.dataset}, logger={log_save_id}')
        model.train()

        tr_loss = 0
        nb_tr_steps = 0
        prog_iter = tqdm(train_dataloader, ncols=80, leave = False, desc="Training")
        tic = time.time()
        for _, batch in enumerate(prog_iter):
            batch = tuple(t.to(device) for t in batch)
            input_ids, dx_labels, rx_labels = batch
            loss, dx2dx, rx2dx, dx2rx, rx2rx = model(
                input_ids, dx_labels, rx_labels)
            loss.backward()
            tr_loss += loss.item()
            nb_tr_steps += 1
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))
            optimizer.step()
            optimizer.zero_grad()
        tic2 = time.time()
        logging.info('training time: {:.1f}'.format(tic2 - tic))

        # evaluation
        model.eval()
        dx2dx_y_preds = []
        rx2dx_y_preds = []
        dx_y_trues = []

        dx2rx_y_preds = []
        rx2rx_y_preds = []
        rx_y_trues = []
        for batch in tqdm(eval_dataloader, ncols=80, leave=False, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, dx_labels, rx_labels = batch
            with torch.no_grad():
                dx2dx, rx2dx, dx2rx, rx2rx = model(input_ids)
                dx2dx_y_preds.append(t2n(dx2dx))
                rx2dx_y_preds.append(t2n(rx2dx))
                dx2rx_y_preds.append(t2n(dx2rx))
                rx2rx_y_preds.append(t2n(rx2rx))

                dx_y_trues.append(
                    t2n(dx_labels))
                rx_y_trues.append(
                    t2n(rx_labels))
        logging.info('rx2rx')
        rx2rx_acc_container = metric_report(
            np.concatenate(rx2rx_y_preds, axis=0), np.concatenate(rx_y_trues, axis=0), args.therhold)
        
        if rx2rx_acc_container[acc_name] > rx_acc_best:
            best_epoch = epoch
            rx_acc_best = rx2rx_acc_container[acc_name]
            # save model
            with open(os.path.join(save_dir, 'bert_config.json'), 'w', encoding='utf-8') as fout:
                fout.write(model.config.to_json_string())
            best_model_state = deepcopy(model.state_dict()) 
        logging.info('best_epoch: {}, best_rx_acc: {:.4f}'.format(best_epoch, rx_acc_best))
        if epoch - best_epoch > args.early_stop:   # n个epoch内，验证集性能不上升之后就停
            break
        logging.info('eval time: {:.1f}\n'.format(time.time() - tic2))
    # save best model
    logging.info('Train finished')
    torch.save(best_model_state, open(os.path.join(save_dir, \
                'Epoch_{}_{}_{:.4}.model'.format(best_epoch, acc_name, rx_acc_best)), 'wb'))  

def metric_report(y_pred, y_true, therhold=0.5):
    y_prob = y_pred.copy()
    y_pred[y_pred > therhold] = 1
    y_pred[y_pred <= therhold] = 0

    acc_container = {}
    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(
        y_true, y_pred, y_prob)
    acc_container['jaccard'] = ja
    acc_container['f1'] = avg_f1
    acc_container['prauc'] = prauc

    logging.info(f"ja: {acc_container['jaccard']:.4f}, f1: {acc_container['f1']:.4f}, prauc: {acc_container['prauc']:.4f} ")

    return acc_container


if __name__ == '__main__':
    torch.manual_seed(1203)
    np.random.seed(2048)
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
        