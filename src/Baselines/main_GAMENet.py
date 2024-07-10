import os
import sys
import dill
import logging
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
sys.path.append("../..")

from models.GAMENet import GAMENet
from utils.util import multi_label_metric, ddi_rate_score, get_n_params, create_log_id, logging_config, get_grouped_metrics, get_model_path

import matplotlib.pyplot as plt

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default=' ', help="User notes")
    parser.add_argument('-t', '--test', action='store_true', default=False, help="test mode")
    parser.add_argument('-s', '--single', action='store_true', default=False, help="single visit")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default='log0', help='log dir prefix like "log0"')
    parser.add_argument('--model_name', type=str, default='GAMENet', help="model name")
    parser.add_argument('--dataset', type=str, default='mimic-iii', help='dataset')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop after this many epochs without improvement')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--target_ddi', type=float, default=0.05, help='target ddi')
    parser.add_argument('--T', type=float, default=0.5, help='T')   # T越小，使用DDI loss的概率越小
    parser.add_argument('--weight_decay', type=float, default=0.85, help="decay weight")
    # parser.add_argument('--weight_multi', type=float, default=0.01, help='weight of multilabel_margin_loss')
    parser.add_argument('--weight_multi', type=float, default=0.005, help='weight of multilabel_margin_loss')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of node embedding(randomly initialize)')
    parser.add_argument('--cuda', type=int, default=7 , help='which cuda')

    args = parser.parse_args()
    return args

# evaluate
@torch.no_grad()
def evaluator(args, model, data_val, voc_size, epoch, 
              ddi_adj_path,
              rec_results_path=None, mode='Train'):
    model.eval()

    ja_visit = [[] for _ in range(5)]
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    visit_weights = []
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    recommended_drugs = set()
    rec_results = [] 
    all_pred_prob = []

    for patient in tqdm(data_val, ncols=60, total=len(
        data_val), desc=f"Evaluation"):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        visit_weights_patient = []
        if mode == "Test":
            all_diseases = []
            all_procedures = []
            all_medications = []

        for adm_idx, adm in enumerate(patient): # every admission
            if mode == "Test":
                diseases = adm[0]
                procedures = adm[1]
                medications = adm[2]
                all_diseases.append(diseases)
                all_procedures.append(procedures)
                all_medications.append(medications)
            target_output = model(patient[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp) # correct medicine
            visit_weights_patient.append(adm[3])

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            all_pred_prob.append(list(y_pred_tmp))
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            recommended_drugs = set(y_pred_label_tmp) | recommended_drugs
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        
        smm_record.append(y_pred_label) # 所有patient的y prediction
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        if mode == "Test":
            if len(patient) < 5:
                ja_visit[len(patient)-1].append(adm_ja)
            else:
                ja_visit[4].append(adm_ja)
            records = [all_diseases, all_procedures, all_medications, y_pred_label, visit_weights_patient, [adm_ja]]
            rec_results.append(records)

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        visit_weights.append(np.max(visit_weights_patient))

    if mode == "Test":
        os.makedirs(rec_results_path, exist_ok=True)
        rec_results_file = rec_results_path + '/' + 'rec_results.pkl'
        dill.dump(rec_results, open(rec_results_file, 'wb'))
        plot_path = rec_results_path + '/' + 'pred_prob.jpg'
        plot_hist(all_pred_prob, plot_path)
        ja_result_file = rec_results_path + '/' + 'ja_result.pkl'
        dill.dump(ja_visit, open(ja_result_file, 'wb'))
        for i in range(5):
            logging.info(str(i+1) + f'visit\t mean: {np.mean(ja_visit[i]):.4}, std: {np.std(ja_visit[i]):.4}, se: {np.std(ja_visit[i]) / np.sqrt(len(ja_visit[i])):.4}')
        

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path) # if (epoch != 0) | (mode=='test')  else 0.0

    get_grouped_metrics(ja, visit_weights)

    logging.info(f'''Epoch {epoch:03d}, Jaccard: {np.mean(ja):.4}, DDI Rate: {ddi_rate:.4}, PRAUC: {np.mean(prauc):.4}, AVG_F1: {np.mean(avg_f1):.4},  AVG_PRC: {np.mean(avg_p):.4f}, AVG_RECALL: {np.mean(avg_r):.4f}, AVG_MED: {med_cnt / visit_cnt:.4}''')  

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), med_cnt / visit_cnt

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
    
    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    val_len = int(len(data[split_point:]) / 2)
    data_val = data[split_point:split_point + val_len]
    data_test = data[split_point+val_len:]
    if args.single:
        data_train = [[visit] for patient in data_train for visit in patient]
        data_val = [[visit] for patient in data_val for visit in patient]
        data_test = [[visit] for patient in data_test for visit in patient]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    # model initialization
    model = GAMENet(args, voc_size, ehr_adj, ddi_adj)
    logging.info(model)

    # test
    if args.test:
        rec_results_path = save_dir + '/' + 'rec_results'

        model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        model.load_state_dict(torch.load(open(model_path, 'rb'))) # , map_location='cuda:0' , map_location = {'cuda:1': 'cuda:0'}
        model.to(device=device)
        logging.info("load model from {}".format(model_path))
        evaluator(args, model, data_test, voc_size, 0, ddi_adj_path, rec_results_path, "Test")
        return 
    else:
        writer = SummaryWriter(save_dir) # 自动生成log文件夹

    # train and validation
    model.to(device=device)
    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = Adam(model.parameters(), lr=args.lr)
    print('parameters', get_n_params(model))
    

    EPOCH = 200
    best_epoch, best_ja = 0, 0
    for epoch in range(EPOCH):
        epoch += 1
        print(f'\nepoch {epoch} --------------------------model_name={args.model_name}, dataset={args.dataset}, logger={log_save_id}')
        model.train()
        prediction_loss_cnt, neg_loss_cnt = 0, 0
        loss_train, loss_val = 0, 0
        for step, patient in tqdm(enumerate(data_train), ncols=60, desc="Training", total=len(data_train)):  # every patient
            for idx, adm in enumerate(patient):
                seq_input = patient[:idx+1]
                result, loss_ddi = model(seq_input) # result = target_output1
                prediction_loss, neg_loss,loss = loss_func(voc_size, adm, result, loss_ddi, ddi_adj_path, device, args.weight_multi)
                prediction_loss_cnt += prediction_loss
                neg_loss_cnt += neg_loss
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                loss_train += loss.item()/len(data_train)  # 用于记录每个epoch的总loss
        
        with torch.no_grad():
            for step, patient in tqdm(enumerate(data_val), ncols=60, desc="Val loss", total=len(data_val)):  # every patient
                for idx, adm in enumerate(patient):   # every admission
                    seq_input = patient[:idx+1] # 前T次数据输入
                    result, loss_ddi = model(seq_input)
                    _, _, loss = loss_func(voc_size, adm, result, loss_ddi, ddi_adj_path, device, args.weight_multi)
                    loss_val += loss.item()/len(data_val)  # 用于记录每个epoch的总loss

        logging.info(f'loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}')
        # evaluation
        args.T *= args.weight_decay

        ddi_rate, ja, prauc, f1,  avg_med = evaluator(
            args, model, data_val, voc_size, epoch, ddi_adj_path) 
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
    

def loss_func(voc_size, adm, result, loss_ddi, ddi_adj_path, device, weight_multi):
    loss_bce_target = np.zeros((1, voc_size[2]))
    loss_bce_target[:, adm[2]] = 1

    loss_multi_target = np.full((1, voc_size[2]), -1)
    for idx, item in enumerate(adm[2]):
        loss_multi_target[0][idx] = item

    prediction_loss_cnt, neg_loss_cnt = 0, 0
    loss_bce = F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device))
    loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device))
    result = F.sigmoid(result).detach().cpu().numpy()[0]
    result[result >= 0.5] = 1
    result[result < 0.5] = 0
    y_label = np.where(result == 1)[0]
    current_ddi_rate = ddi_rate_score([[y_label]], path = ddi_adj_path)
    # print("current_ddi_rate:{}".format(current_ddi_rate))
    
    if current_ddi_rate <= args.target_ddi:
        loss = (1 - args.weight_multi) * loss_bce + args.weight_multi * loss_multi
        prediction_loss_cnt += 1
        # print("current_ddi_rate<=args.target_ddi")
    else:
        rnd = np.exp((args.target_ddi - current_ddi_rate) / args.T)
        # print("rnd:{}".format(rnd))
        if np.random.rand(1) < rnd:
            # print("np.random.rand(1) < rnd:")
            loss = loss_ddi
            neg_loss_cnt += 1
        else:
            loss = (1 - args.weight_multi) * loss_bce + args.weight_multi * loss_multi
            prediction_loss_cnt += 1
    return prediction_loss_cnt, neg_loss_cnt, loss

def tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                        loss_train=0, loss_val=0):
    if epoch > 0:
        writer.add_scalar('Loss/Train', loss_train, epoch)
        writer.add_scalar('Loss/Val', loss_val, epoch)

    writer.add_scalar('Metrics/Jaccard', ja, epoch)
    writer.add_scalar('Metrics/prauc', prauc, epoch)
    writer.add_scalar('Metrics/DDI', ddi_rate, epoch)
    writer.add_scalar('Metrics/Med_count', avg_med, epoch)


def plot_hist(all_y_pred, save_path):
    y_pred =  [item for sublist in all_y_pred for item in sublist]
    hist, _ = np.histogram(y_pred, bins = 10, range = (0, 1))
    # f = open('hist.txt', 'a')
    logging.info(f'hist: {hist}')
    plt.hist(y_pred)
    plt.savefig(save_path)

if __name__ == '__main__':
    torch.manual_seed(1203)
    np.random.seed(2048)
    args = get_args()
    main(args)
        