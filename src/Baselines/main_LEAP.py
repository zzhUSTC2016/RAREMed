import argparse
from copy import deepcopy
import torch
import numpy as np
import dill
from tqdm import tqdm
import logging
from torch.optim import Adam
import os
import torch.nn.functional as F
import random

import sys
sys.path.append("..")
sys.path.append("../..")
from models.LEAP import Leap
from utils.util import multi_label_metric, ddi_rate_score, get_n_params, create_log_id, logging_config, get_grouped_metrics, pop_metric, get_model_path, get_pretrained_model_path, sequence_output_process

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default='', help="User notes")
    parser.add_argument('--model_name', type=str, default='Leap', help="model name")
    parser.add_argument('--dataset', type=str, default='mimic-iii', help='dataset')
    parser.add_argument('-t', '--test', action='store_true', help="test mode")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default="log0", help='log dir prefix like "log0", for model test')
    parser.add_argument('--cuda', type=int, default=5, help='which cuda')

    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate') 
    parser.add_argument('--lr_finetune', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=30, help='number of epoches')          # epoch增大容易过拟合
    args = parser.parse_args()
    return args

def eval(model, data_eval, voc_size, epoch, ddi_adj_path):
    # evaluate
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    visit_weights = []
    records = []
    med_cnt = 0
    visit_cnt = 0
    for input in tqdm(data_eval, ncols=80, total=len(data_eval), desc="Evaluation"):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        visit_weights_patient = []
        for adm in input:
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)
            visit_weights_patient.append(adm[3])

            output_logits = model(adm)
            output_logits = output_logits.detach().cpu().numpy()

            out_list, sorted_predict = sequence_output_process(output_logits, [voc_size[2], voc_size[2]+1])

            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(output_logits[:, :-2], axis=0))

            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            visit_cnt += 1
            med_cnt += len(sorted_predict)
        records.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        visit_weights.append(np.mean(visit_weights_patient))

    get_grouped_metrics(ja, visit_weights)

    # ddi rate
    ddi_rate = ddi_rate_score(records, path=ddi_adj_path)
    logging.info(f'''Epoch {epoch:03d}, Jaccard: {np.mean(ja):.4}, DDI Rate: {ddi_rate:.4}, PRAUC: {np.mean(prauc):.4}, AVG_F1: {np.mean(avg_f1):.4}, AVG_PRC: {np.mean(avg_p):.4f}, AVG_RECALL: {np.mean(avg_r):.4f}, AVG_MED: {med_cnt / visit_cnt:.4}''')  
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)

def main():
    # set logger
    if args.test:
        args.note = 'test of ' + args.log_dir_prefix
    log_directory_path = os.path.join('../log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log'+str(log_save_id)+'_'+args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    data_path = f'../../data/output/{args.dataset}' + '/records_final.pkl'
    voc_path = f'../../data/output/{args.dataset}' + '/voc_final.pkl'
    ddi_adj_path = f'../../data/output/{args.dataset}' + '/ddi_A_final.pkl'

    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']


    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point:split_point + eval_len]
    data_test = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    END_TOKEN = voc_size[2] + 1

    model = Leap(voc_size, device=device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.test:
        model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        model.load_state_dict(torch.load(open(model_path, 'rb')))
        model.to(device=device)
        logging.info("load model from %s", model_path)
        eval(model, data_test, voc_size, 0, ddi_adj_path)
        return

    model.to(device=device)
    print('parameters', get_n_params(model))

    best_epoch, best_ja = 0, 0
    for epoch in range(args.epoch):
        epoch += 1
        model.train()
        for input in tqdm(data_train, ncols=80, total=len(data_train), desc="Training"):
            for adm in input:
                loss_target = adm[2] + [END_TOKEN]
                output_logits = model(adm)
                loss = F.cross_entropy(output_logits, torch.LongTensor(loss_target).to(device))

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch, ddi_adj_path)

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            best_model_state = deepcopy(model.state_dict()) 
        logging.info(f'best epoch: {best_epoch}, best_ja: {best_ja:.4}')
    
    logging.info('Train finished')
    torch.save(best_model_state, open(os.path.join(save_dir, \
                'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best_epoch, best_ja, ddi_rate)), 'wb'))  
    

def fine_tune():
    # set logger
    args.note = 'finetune of ' + args.log_dir_prefix
    log_directory_path = os.path.join('log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log'+str(log_save_id)+'_'+args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)
    
    data_path = f'../data/output/{args.dataset}' + '/records_final.pkl'
    voc_path = f'../data/output/{args.dataset}' + '/voc_final.pkl'
    ddi_adj_path = f'../data/output/{args.dataset}' + '/ddi_A_final.pkl'
    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ddi_A = dill.load(open(ddi_adj_path, 'rb'))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = Leap(voc_size, device=device)
    model_path = get_model_path(log_directory_path, args.log_dir_prefix)
    logging.info("load model from %s", model_path)
    model.load_state_dict(torch.load(open(model_path, 'rb')))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr_finetune)
    for epoch in range(1):
        random_train_set = [random.choice(data_train) for i in range(len(data_train))]
        for input in tqdm(random_train_set, ncols=80, total=len(random_train_set), desc="fine-tuning"):
            model.train()
            K_flag = False
            for adm in input:
                target = adm[2]
                output_logits = model(adm)
                out_list, sorted_predict = sequence_output_process(output_logits.detach().cpu().numpy(), [voc_size[2], voc_size[2] + 1])

                inter = set(out_list) & set(target)
                union = set(out_list) | set(target)
                jaccard = 0 if union == 0 else len(inter) / len(union)
                K = 0
                for i in out_list:
                    if K == 1:
                        K_flag = True
                        break
                    for j in out_list:
                        if ddi_A[i][j] == 1:
                            K = 1
                            break

                loss = -jaccard * K * torch.mean(F.log_softmax(output_logits, dim=-1))

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

    if K_flag:
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_test, voc_size, epoch, ddi_adj_path)

    logging.info('Train finished')
    torch.save(model.state_dict(), open(os.path.join(save_dir, \
                'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, ja, ddi_rate)), 'wb'))  
    

if __name__ == '__main__':
    sys.path.append("..")
    torch.manual_seed(1203)

    args = get_args()

    main()
    # fine_tune()