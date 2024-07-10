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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

sys.path.append("..")
sys.path.append("../..")

from models.GBert import GBERT_Predict
from utils.util import multi_label_metric, ddi_rate_score, get_n_params, create_log_id, logging_config, get_model_path, get_grouped_metrics

# Training settings
def get_args():
    model_name = 'GBert'

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default=' ', help="User notes")
    parser.add_argument('-t', '--Test', action='store_true', default=False, help="test mode")
    parser.add_argument('--model_name', type=str, default=model_name, help="model name")
    parser.add_argument('--dataset', type=str, default='mimic-iii', help='dataset')
    parser.add_argument("--use_pretrain", default=True, action='store_false', help="is use pretrain")
    parser.add_argument('-p', "--pretrain_prefix", type=int, default=0, help="pretrain_prefix")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default="log1", help='log dir prefix like "log0", for model test')
    parser.add_argument("--batch_size", default=1, type=int, help="Total batch size for training.")

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--therhold", default=0.3, type=float, help="therhold.")

    parser.add_argument('--early_stop', type=int, default=10, help='early stop after this many epochs without improvement')
    parser.add_argument('--cuda', type=int, default=0 , help='which cuda')  #TODO 记得改cuda选项
    parser.add_argument("--num_train_epochs", default=20.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=1203, help="random seed for initialization")

    args = parser.parse_args()
    return args


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

class BertDataset(Dataset):
    def __init__(self, data, voc, voc_size, max_seq_len):
        self.data = data
        self.voc = voc
        self.diag_voc, self.med_voc = voc['diag_voc'], voc['med_voc']
        self.voc_size = voc_size
        self.max_seq_len = max_seq_len
    
    def __getitem__(self, index):
        patient = copy.deepcopy(self.data[index]) 
        input_ids = []
        output_dx_labels = []  # (adm-1, dx_voc_size)
        output_rx_labels = []  # (adm-1, rx_voc_size)
        patient_weights = []
        for idx, adm in enumerate(patient):
            input_ids.extend([self.diag_voc.word2idx['[CLS]']] + fill_to_max(adm[0], self.diag_voc, self.max_seq_len-1))
            input_ids.extend([self.med_voc.word2idx['[CLS]']] + fill_to_max(adm[2], self.med_voc, self.max_seq_len-1))
            patient_weights.append(adm[3])
            if idx != 0:
                y_dx = np.zeros(self.voc_size[0])
                y_rx = np.zeros(self.voc_size[2])
                y_dx[adm[0]] = 1
                y_rx[adm[2]] = 1
                output_dx_labels.append(y_dx)
                output_rx_labels.append(y_rx)
        cur_tensors = (torch.tensor(input_ids, dtype=torch.long).view(-1, self.max_seq_len), # [2*adm, max_seq_len]
                        torch.tensor(output_dx_labels, dtype=torch.float), # [adm-1, diag_voc_size] one-hot
                        torch.tensor(output_rx_labels, dtype=torch.float), # [adm-1, med_voc_size] one-hot
                        np.mean(patient_weights))  # [adm, 1] 
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
    if args.Test:
        args.note = f'test of {args.log_dir_prefix}'
    else:
        args.note = f'finetune of {args.pretrain_prefix}'
    log_directory_path = os.path.join('../log', args.dataset, args.model_name)
    # os.makedirs(log_directory_path, exist_ok=True)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log'+str(log_save_id)+'_'+args.note)
    os.makedirs(save_dir, exist_ok=True)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    logging.info(f'model_name={args.model_name}, dataset={args.dataset}, lr={args.lr}, '
    f'use_pretrain={args.use_pretrain}, num_train_epochs={args.num_train_epochs}, '
    f'therhold={args.therhold}, save_dir={log_save_id}')
    
    # load data
    data_path = f'../../data/output/{args.dataset}' + '/records_final.pkl'
    voc_path = f'../../data/output/{args.dataset}' + '/voc_final.pkl'
    ddi_adj_path = f'../../data/output/{args.dataset}' + '/ddi_A_final.pkl'
    
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
    max_proc_len = 0
    max_med_len = 0
    for idx, patient in enumerate(data):
        for adm in patient:
            if len(adm[0]) > max_diag_len:
                max_diag_len = len(adm[0])
            if len(adm[1]) > max_proc_len:
                max_proc_len = len(adm[1])
            if len(adm[2]) > max_med_len:
                max_med_len = len(adm[2])
    # print(max_diag_len)  # [39, 32, 52]
    # print(max_proc_len) 
    # print(max_med_len) 
    # print(len(data)) # 635
    # print(len(single_idx)) # 908
    
    max_seq_len = max([max_diag_len, max_proc_len, max_med_len]) + 2

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_val = data[split_point:split_point + eval_len]
    data_test = data[split_point+eval_len:]

    data_train = BertDataset(data_train, voc, voc_size, max_seq_len)
    data_val = BertDataset(data_val, voc, voc_size, max_seq_len)
    data_test = BertDataset(data_test, voc, voc_size, max_seq_len)

    train_dataloader = DataLoader(data_train,
                                  sampler=RandomSampler(data_train),
                                  batch_size=1)
    eval_dataloader = DataLoader(data_val,
                                 sampler=SequentialSampler(data_val),
                                 batch_size=1)
    test_dataloader = DataLoader(data_test,
                                 sampler=SequentialSampler(data_test),
                                 batch_size=1)
    
    logging.info("Use Pretraining model")
    pretrain_dir = get_model_path(log_directory_path, 'log'+str(args.pretrain_prefix))
    logging.info(f'pretrained_weight_path={pretrain_dir}')
    model = GBERT_Predict.from_pretrained(pretrain_dir, rx_voc = med_voc, device=device)

    # model initialization
    if args.Test:
        model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        model.load_state_dict(torch.load(open(model_path, 'rb'))) # , map_location='cuda:0' , map_location = {'cuda:1': 'cuda:0'}
        # model = GBERT_Predict.from_pretrained(model_path, rx_voc = med_voc, device=device)
        model.to(device=device)
        rec_results_path = save_dir + '/' + 'rec_results'
        evaluator(args, model, test_dataloader, ddi_adj_path, device, rec_results_path, mode='Test')
        return
    
    # train and validation
    model.to(device=device)
    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = Adam(model.parameters(), lr=args.lr)
    print('parameters', get_n_params(model))

    EPOCH = int(args.num_train_epochs)

    best_epoch, best_ja = 0, 0
    for epoch in range(EPOCH):
        epoch += 1
        logging.info(f'\nepoch {epoch} --------------------------model_name={args.model_name}, dataset={args.dataset}, logger={log_save_id}')
        model.train()

        tr_loss, eval_loss = 0, 0
        tic = time.time()
        for _, batch in enumerate(tqdm(train_dataloader, ncols=80, desc="Training")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, dx_labels, rx_labels, _ = batch
            input_ids, dx_labels, rx_labels = input_ids.squeeze(
                dim=0), dx_labels.squeeze(dim=0), rx_labels.squeeze(dim=0)
            if rx_labels.size(0) == 0:
                continue
            loss, rx_logits = model(input_ids, dx_labels=dx_labels, rx_labels=rx_labels, epoch=epoch)
            loss.backward()

            tr_loss += loss.mean().item()
            optimizer.step()
            optimizer.zero_grad()
        tic2 = time.time()

        ddi_rate, ja, prauc, avg_f1, avg_med = evaluator(
            args, model, eval_dataloader, ddi_adj_path, device, mode='Val')

        # save best epoch
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja, best_prauc, best_ddi_rate, best_avg_med = ja, prauc, ddi_rate, avg_med
            best_model_state = deepcopy(model.state_dict()) 
        logging.info('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))

        if epoch - best_epoch > args.early_stop:   # n个epoch内，验证集性能不上升之后就停
            break

    # save best model
    logging.info('Train finished')
    torch.save(best_model_state, open(os.path.join(save_dir, \
                'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best_epoch, best_ja, ddi_rate)), 'wb'))  


def evaluator(args, model, eval_dataloader, ddi_adj_path, device, rec_results_path=None, mode='train'):
    model.eval()    

    eval_loss = 0
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    visit_weights = []
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    rec_results = [] 
    ja_visit = [[] for _ in range(5)]

    for eval_input in tqdm(eval_dataloader, ncols=80, desc="Evaluating"):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        # eval_input = tuple(t.to(device) for t in eval_input[:3])
        input_ids, dx_labels, rx_labels, patient_weight = eval_input
        input_ids, dx_labels, rx_labels = input_ids.to(device), dx_labels.to(device), rx_labels.to(device)
        input_ids, dx_labels, rx_labels = input_ids.squeeze(
        ), dx_labels.squeeze(), rx_labels.squeeze(dim=0)
        if rx_labels.size(0) == 0:
            continue
        visit_weights.append(t2n(patient_weight)[0])
        with torch.no_grad():
            loss, rx_logits = model(
                input_ids, dx_labels=dx_labels, rx_labels=rx_labels)
            eval_loss += loss.mean().item()

        for i in range(rx_labels.size(0)):
            y_gt.append(t2n(rx_labels[i])[:-3]) # correct medicine
            # true label
            y_gt_label_tmp = np.where(t2n(rx_labels[i]) == 1)[0]
            # y_gt_label.append(sorted(y_gt_label_tmp))
            # prediction prod
            y_pred_prob_tmp = t2n(torch.sigmoid(rx_logits[i]))
            y_pred_prob_tmp = y_pred_prob_tmp[:-3]
            y_pred_prob.append(y_pred_prob_tmp)
            # prediction med set
            y_pred_tmp = y_pred_prob_tmp.copy()

            y_pred_tmp[y_pred_tmp > args.therhold] = 1
            y_pred_tmp[y_pred_tmp <= args.therhold] = 0
            y_pred.append(y_pred_tmp)
            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        # ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(np.concatenate(y_gt, axis=0), np.concatenate(y_pred, axis=0), np.concatenate(y_pred_prob, axis=0))
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
                np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        # if mode == "Test":
        #     records = [y_gt_label_tmp, y_pred_label_tmp]
        #     rec_results.append(records)
        if mode == "Test":
            if rx_labels.size(0) < 5:
                ja_visit[rx_labels.size(0)-1].append(adm_ja)
            else:
                ja_visit[4].append(adm_ja)
            records = [y_gt_label_tmp, y_pred_label, t2n(patient_weight)[0], [adm_ja]]
            rec_results.append(records)
        smm_record.append(y_pred_label) # 所有patient的y prediction
        
        
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
    if args.Test:
        os.makedirs(rec_results_path, exist_ok=True)
        rec_results_file = rec_results_path + '/' + 'rec_results.pkl'
        dill.dump(rec_results, open(rec_results_file, 'wb'))
        # plot_path = rec_results_path + '/' + 'pred_prob.jpg'
        # plot_hist(all_pred_prob, plot_path)
        ja_result_file = rec_results_path + '/' + 'ja_result.pkl'
        dill.dump(ja_visit, open(ja_result_file, 'wb'))
        for i in range(5):
            logging.info(str(i+1) + f'visit\t mean: {np.mean(ja_visit[i]):.4}, std: {np.std(ja_visit[i]):.4}, se: {np.std(ja_visit[i])/np.sqrt(len(ja_visit[i])):.4}')
        
    get_grouped_metrics(ja, visit_weights)

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path) # if (epoch != 0) | (mode=='Test')  else 0.0

    logging.info(f'Jaccard: {np.mean(ja):.4}, DDI Rate: {ddi_rate:.4}, PRAUC: {np.mean(prauc):.4}, AVG_F1: {np.mean(avg_f1):.4}, AVG_MED: {med_cnt / visit_cnt:.4}')  

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), med_cnt / visit_cnt


if __name__ == '__main__':
    torch.manual_seed(1203)
    np.random.seed(2048)
    args = get_args()
    random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
        