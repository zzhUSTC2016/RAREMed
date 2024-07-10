import os
import sys
import dill
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
sys.path.append("../..")

from utils.data_loader import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace
from models.COGNet import COGNet
from utils.util import llprint, get_n_params, output_flatten, create_log_id, logging_config, get_model_path
from utils.recommend import eval

# Training settings
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default=' ', help="User notes")
    parser.add_argument('-t', '--test', action='store_true', help="test mode")
    parser.add_argument('-s', '--single', action='store_true', default=False, help="single visit")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default="log0", help='log dir prefix like "log0"')
    parser.add_argument('--model_name', type=str, default="COGNet", help="model name")
    parser.add_argument('--dataset', type=str, default="mimic-iii", help='dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--beam_size', type=int, default=4, help='max num of sentences in beam searching')
    parser.add_argument('--max_len', type=int, default=45, help='maximum prediction medication sequence')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop after this many epochs without improvement')
    parser.add_argument('--cuda', type=int, default=2 , help='which cuda')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of node embedding(randomly initialize)')

    args = parser.parse_args()
    return args

def main(args):
    # set logger
    if args.test:
        args.note = f'test of {args.log_dir_prefix}'
    log_directory_path = os.path.join('../log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log'+str(log_save_id)+'_'+args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    # load data
    data_path = f'../../data/output/{args.dataset}' + '/records_final.pkl'
    voc_path = f'../../data/output/{args.dataset}' + '/voc_final.pkl'
    ddi_adj_path = f'../../data/output/{args.dataset}' + '/ddi_A_final.pkl'
    ehr_adj_path = f'../../data/output/{args.dataset}' + '/ehr_adj_final.pkl'
    ddi_mask_path = f'../../data/output/{args.dataset}' + '/ddi_mask_H.pkl'

    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))

    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    # frequency statistic
    med_count = defaultdict(int)
    for patient in data:
        for adm in patient:
            for med in adm[2]:
                med_count[med] += 1
    
    ## rare first
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur_medications = sorted(data[i][j][2], key=lambda x:med_count[x])
            data[i][j][2] = cur_medications


    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    val_len = int(len(data[split_point:]) / 2)
    data_val = data[split_point:split_point + val_len]
    data_test = data[split_point+val_len:]
    if args.single:
        data_train = [[visit] for patient in data_train for visit in patient]
        data_val = [[visit] for patient in data_val for visit in patient]
        data_test = [[visit] for patient in data_test for visit in patient]

    train_dataset = mimic_data(data_train)
    eval_dataset = mimic_data(data_val)
    test_dataset = mimic_data(data_test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch_v2_train, shuffle=False, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=False, pin_memory=True)
    
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    # model initialization
    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    model = COGNet(args, voc_size, ehr_adj, ddi_adj, ddi_mask_H)
    logging.info(model)
         
    # test
    if args.test:
        model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        model.load_state_dict(torch.load(open(model_path, 'rb')))
        model.to(device=device)

        eval(args, 0, model, test_dataloader, voc_size, ddi_adj_path, save_dir+'/rec_results.pkl')
        return
    else:
        writer = SummaryWriter(save_dir) # 自动生成log文件夹

    # train and validation
    model.to(device=device)
    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = Adam(model.parameters(), lr=args.lr)
    print('parameters', get_n_params(model))
    # tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch=0)


    EPOCH = 100
    best_epoch, best_ja = 0, 0
    for epoch in range(EPOCH):
        epoch += 1
        print(f'\nepoch {epoch} --------------------------model_name={args.model_name}, lr={args.lr}, '
            f'batch_size={args.batch_size}, beam_size={args.beam_size}, max_med_len={args.max_len}, logger={log_save_id}')
        model.train()
        tic = time.time()
        loss_train, loss_val = 0, 0
        for idx, data in enumerate(train_dataloader):

            diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data

            diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
            procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
            dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
            stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
            dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
            stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
            medications = medications.to(device)
            m_mask_matrix = m_mask_matrix.to(device)
            d_mask_matrix = d_mask_matrix.to(device)
            p_mask_matrix = p_mask_matrix.to(device)
            dec_disease_mask = dec_disease_mask.to(device)
            stay_disease_mask = stay_disease_mask.to(device)
            dec_proc_mask = dec_proc_mask.to(device)
            stay_proc_mask = stay_proc_mask.to(device)
            output_logits = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
                dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)
            labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2] + 2, END_TOKEN, device, max_len=args.max_len)

            loss = F.nll_loss(predictions, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()/len(train_dataloader)
            llprint('\rtraining step: {} / {}'.format(idx, len(train_dataloader)))

        with torch.no_grad():
            for idx, data in tqdm(enumerate(eval_dataloader), ncols=60, desc="Val loss", total=len(eval_dataloader)):  # every patient
                diseases, procedures, medications, visit_weights_patient, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data
            diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
            procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
            dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
            stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
            dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
            stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
            medications = medications.to(device)
            m_mask_matrix = m_mask_matrix.to(device)
            d_mask_matrix = d_mask_matrix.to(device)
            p_mask_matrix = p_mask_matrix.to(device)
            dec_disease_mask = dec_disease_mask.to(device)
            stay_disease_mask = stay_disease_mask.to(device)
            dec_proc_mask = dec_proc_mask.to(device)
            stay_proc_mask = stay_proc_mask.to(device)
            output_logits = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
                dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)
            labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2] + 2, END_TOKEN, device, max_len=args.max_len)
            loss = F.nll_loss(predictions, labels.long())
            loss_val += loss.item()/len(eval_dataloader)

        # logging.info(f'loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}')
        # evaluation

        tic2 = time.time()
        ddi_rate, ja, prauc, avg_f1, avg_med  = eval(args, epoch, model, eval_dataloader, voc_size, ddi_adj_path)
        logging.info('training time: {:.1f}, test time: {:.1f}'.format(time.time() - tic, time.time() - tic2))
        # print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))
        tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                          loss_train, loss_val)

        # save best epoch
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja, best_prauc, best_ddi_rate, best_avg_med = ja, prauc, ddi_rate, avg_med
            best_model_state = deepcopy(model.state_dict()) 
        logging.info('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))
        # print ('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))

        if epoch - best_epoch > args.early_stop:   # n个epoch内，验证集性能不上升之后就停
            break

    # save best model
    logging.info('Train finished')
    torch.save(best_model_state, open(os.path.join(save_dir, \
                'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best_epoch, best_ja, ddi_rate)), 'wb'))  
    


def tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                        loss_train=0, loss_val=0):
    if epoch > 0:
        writer.add_scalar('Loss/Train', loss_train, epoch)
        writer.add_scalar('Loss/Val', loss_val, epoch)

    writer.add_scalar('Metrics/Jaccard', ja, epoch)
    writer.add_scalar('Metrics/prauc', prauc, epoch)
    writer.add_scalar('Metrics/DDI', ddi_rate, epoch)
    writer.add_scalar('Metrics/Med_count', avg_med, epoch)

if __name__ == '__main__':
    torch.manual_seed(1203)
    np.random.seed(2048)
    args = get_args()
    main(args)
