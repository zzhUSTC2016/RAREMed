import torch
import numpy as np
import pandas as pd
import dill
import logging
import os
import logging
from scipy.stats import linregress

from utils.data_loader import pad_num_replace
from utils.beam import Beam

import sys
sys.path.append("..")
# from models import Leap #, CopyDrug_batch, CopyDrug_tranformer, CopyDrug_generate_prob, CopyDrug_diag_proc_encode
# from COGNet_model import COGNet
from utils.util import llprint, sequence_output_process, ddi_rate_score, output_flatten,\
      multi_label_metric, pop_metric, get_grouped_metrics

torch.manual_seed(1203)

'''# 读取disease跟proc的英文名
icd_diag_path = '../data/D_ICD_DIAGNOSES.csv'
icd_proc_path = '../data/D_ICD_PROCEDURES.csv'
code2diag = {}
code2proc = {}

with open(icd_diag_path, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        line = line.strip().split(',"')
        if line[-1] == '': line = line[:-1]
        _, icd_code, _, title = line
        code2diag[icd_code[:-1]] = title

with open(icd_proc_path, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        _, icd_code, _, title = line.strip().split(',"')
        code2proc[icd_code[:-1]] = title'''



def eval_recommend_batch(model, batch_data, device, TOKENS, args):
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS

    diseases, procedures, medications, visit_weights_patient, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = batch_data
    # continue
    # 根据vocab对padding数值进行替换
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

    batch_size = medications.size(0)
    max_visit_num = medications.size(1)

    input_disease_embdding, input_proc_embedding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory = model.encode(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, 
        seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20)

    partial_input_medication = torch.full((batch_size, max_visit_num, 1), SOS_TOKEN).to(device)
    parital_logits = None


    for i in range(args.max_len):
        partial_input_med_num = partial_input_medication.size(2)
        partial_m_mask_matrix = torch.zeros((batch_size, max_visit_num, partial_input_med_num), device=device).float()
        # print('val', i, partial_m_mask_matrix.size())

        parital_logits = model.decode(partial_input_medication, input_disease_embdding, input_proc_embedding, encoded_medication, last_seq_medication, cross_visit_scores,
            d_mask_matrix, p_mask_matrix, partial_m_mask_matrix, last_m_mask, drug_memory)
        _, next_medication = torch.topk(parital_logits[:, :, -1, :], 1, dim=-1)
        partial_input_medication = torch.cat([partial_input_medication, next_medication], dim=-1)

    return parital_logits


def test_recommend_batch(model, batch_data, device, TOKENS, ddi_adj, args):
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS

    diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = batch_data
    # continue
    # 根据vocab对padding数值进行替换
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

    batch_size = medications.size(0)
    visit_num = medications.size(1)

    input_disease_embdding, input_proc_embedding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory = model.encode(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, 
        seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20)

    # partial_input_medication = torch.full((batch_size, visit_num, 1), SOS_TOKEN).to(device)
    # parital_logits = None

     # 为每一个样本声明一个beam
    # 这里为了方便实现，写死batch_size必须为1
    assert batch_size == 1
    # visit_num个batch
    beams = [Beam(args.beam_size, MED_PAD_TOKEN, SOS_TOKEN, END_TOKEN, ddi_adj, device) for _ in range(visit_num)]

    # 构建decode输入，每一个visit上需要重复beam_size次数据
    input_disease_embdding = input_disease_embdding.repeat_interleave(args.beam_size, dim=0)
    input_proc_embedding = input_proc_embedding.repeat_interleave(args.beam_size, dim=0)
    encoded_medication = encoded_medication.repeat_interleave(args.beam_size, dim=0)
    last_seq_medication = last_seq_medication.repeat_interleave(args.beam_size, dim=0)
    cross_visit_scores = cross_visit_scores.repeat_interleave(args.beam_size, dim=0)
    # cross_visit_scores = cross_visit_scores.repeat_interleave(args.beam_size, dim=2)
    d_mask_matrix = d_mask_matrix.repeat_interleave(args.beam_size, dim=0)
    p_mask_matrix = p_mask_matrix.repeat_interleave(args.beam_size, dim=0)
    last_m_mask = last_m_mask.repeat_interleave(args.beam_size, dim=0)

    for i in range(args.max_len):
        len_dec_seq = i + 1
        # b.get_current_state(): (beam_size, len_dec_seq) --> (beam_size, 1, len_dec_seq)
        # dec_partial_inputs: (beam_size, visit_num, len_dec_seq)
        dec_partial_inputs = torch.cat([b.get_current_state().unsqueeze(dim=1) for b in beams], dim=1).to(device)
        # dec_partial_inputs = dec_partial_inputs.view(args.beam_size, visit_num, len_dec_seq)

        partial_m_mask_matrix = torch.zeros((args.beam_size, visit_num, len_dec_seq), device=device).float().to(device)
        # print('val', i, partial_m_mask_matrix.size())

        # parital_logits: (beam_size, visit_sum, len_dec_seq, all_med_num)
        parital_logits = model.decode(dec_partial_inputs, input_disease_embdding, input_proc_embedding, encoded_medication, last_seq_medication, cross_visit_scores,
            d_mask_matrix, p_mask_matrix, partial_m_mask_matrix, last_m_mask, drug_memory)

        # word_lk: (beam_size, visit_sum, all_med_num)
        word_lk = parital_logits[:, :, -1, :]

        active_beam_idx_list = []   # 记录目前仍然active的beam
        for beam_idx in range(visit_num):
            # # 如果当前beam完成了，则跳过，这里beams的size应该是不变的
            # if beams[beam_idx].done: continue
            # inst_idx = beam_inst_idx_map[beam_idx]  # 该beam所对应的adm下标
            # 更新beam，同时返回当前beam是否完成，如果未完成则表示active
            if not beams[beam_idx].advance(word_lk[:, beam_idx, :]):
                active_beam_idx_list.append(beam_idx)

        # 如果没有active的beam，则全部样本预测完毕
        if not active_beam_idx_list: break

    # Return useful information
    all_hyp = []
    all_prob = []
    for beam_idx in range(visit_num):
        scores, tail_idxs = beams[beam_idx].sort_scores()   # 每个beam按照score排序，找出最优的生成
        hyps = beams[beam_idx].get_hypothesis(tail_idxs[0])
        probs = beams[beam_idx].get_prob_list(tail_idxs[0])
        all_hyp += [hyps]    # 注意这里只关注最优解，否则写法上要修改
        all_prob += [probs]

    return all_hyp, all_prob


# evaluate
def eval(args, epoch, model, eval_dataloader, voc_size, ddi_adj_path, rec_results_path = None):
    device = torch.device('cuda:{}'.format(args.cuda))
    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]
    
    model.eval()
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    visit_weights = []
    ja_ips, avg_p_ips, avg_r_ips, avg_pop = [[] for _ in range(4)]
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    recommended_drugs = set()

    ja_visit = [[] for _ in range(5)]
    rec_results = []
    for idx, data in enumerate(eval_dataloader):
        diseases, procedures, medications, visit_weights_patient, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data
        visit_cnt += seq_length.sum().item()

        output_logits = eval_recommend_batch(model, data, device, TOKENS, args)

        # 每一个med上的预测结果
        labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2], END_TOKEN, device, training=False, testing=False, max_len=args.max_len)

        y_gt = []       # groud truth 表示正确的label   0-1序列
        y_pred = []     # 预测的结果    0-1序列
        y_pred_prob = []    # 预测的每一个药物的平均概率，非0-1序列
        y_pred_label = []   # 预测的结果，非0-1序列
        # 针对每一个admission的预测结果

        for label, prediction in zip(labels, predictions):
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[label] = 1    # 01序列，表示正确的label
            y_gt.append(y_gt_tmp)

            # label: med set
            # prediction: [med_num, probability]
            out_list, sorted_predict = sequence_output_process(prediction, [voc_size[2], voc_size[2]+1])

            recommended_drugs = set(sorted_predict) | recommended_drugs
            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(prediction[:, :-2], axis=0))

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            med_cnt += len(sorted_predict)


        smm_record.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        visit_weights.append(torch.max(visit_weights_patient, dim=1)[0].item())
        if args.test:
            diseases = diseases.tolist()[0]
            procedures = procedures.tolist()[0]
            medications = medications.tolist()[0]
            visit_weights_patient = visit_weights_patient.tolist()
            if len(diseases) < 5:
                ja_visit[len(diseases)-1].append(adm_ja)
            else:
                ja_visit[4].append(adm_ja)
            rec_results.append([diseases, procedures, medications, y_pred_label, visit_weights_patient, [adm_ja]])
    
        llprint('\rtest step: {} / {}'.format(idx, len(eval_dataloader)))
    if args.test:
        os.makedirs(rec_results_path, exist_ok=True)
        rec_results_file = rec_results_path + '/' + 'rec_results.pkl'
        dill.dump(rec_results, open(rec_results_file, 'wb'))
        # plot_path = rec_results_path + '/' + 'pred_prob.jpg'
        # plot_hist(all_pred_prob, plot_path)
        ja_result_file = rec_results_path + '/' + 'ja_result.pkl'
        dill.dump(ja_visit, open(ja_result_file, 'wb'))
        for i in range(5):
            logging.info(str(i+1) + f'visit\t mean: {np.mean(ja_visit[i]):.4}, std: {np.std(ja_visit[i]):.4}, se: {np.std(ja_visit[i])/np.sqrt(len(ja_visit[i])):.4}')
        
    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path) # if (epoch != 0) | (mode=='test')  else 0.0

    get_grouped_metrics(ja, visit_weights)
    # # weighted jaccard
    # weighted_jaccard = np.average(ja, weights=visit_weights)

    # # create a dataframe with visit_weights and jaccard
    # visit_weights_df = pd.DataFrame({'visit_weights': visit_weights, 'jaccard': ja})
    # visit_weights_df.sort_values(by='visit_weights', inplace=True)
    # visit_weights_df.reset_index(drop=True, inplace=True)

    # sorted_jaccard = visit_weights_df['jaccard'].values

    # K=int(len(sorted_jaccard)/5)+1
    # grouped_mean_jac = [sorted_jaccard[i:i+K].mean() for i in range(0,int(len(sorted_jaccard)),K)]
    # grouped_mean_jac = [round(i, 4) for i in grouped_mean_jac]
    # # calculate the correlation between grouped_mean_jac and x
    # corr = -np.corrcoef(grouped_mean_jac, np.arange(len(grouped_mean_jac)))[0, 1]
    # slope_corr = -linregress(np.arange(len(grouped_mean_jac)), grouped_mean_jac)[0]

    logging.info(f'''Epoch {epoch:03d}, Jaccard: {np.mean(ja):.4}, DDI Rate: {ddi_rate:.4}, PRAUC: {np.mean(prauc):.4},  AVG_PRC: {np.mean(avg_p):.4f}, AVG_RECALL: {np.mean(avg_r):.4f}, AVG_F1: {np.mean(avg_f1):.4}, AVG_MED: {med_cnt / visit_cnt:.4}''')  
    # logging.info(f'''Epoch {epoch:03d}, weighted_Jaccard: {weighted_jaccard:.4}, corr: {corr:.4}, slope_corr: {slope_corr:.4}''')
    # logging.info(f'''Epoch {epoch:03d}, grouped_mean_jac: {grouped_mean_jac}''')

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), med_cnt / visit_cnt


# test 
def test(args, model, test_dataloader, voc_size, ddi_adj, rec_results_path, ddi_adj_path, medicine_pop_path):
    device = torch.device('cuda:{}'.format(args.cuda))

    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    model.eval()
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    ja_ips, avg_p_ips, avg_r_ips, avg_pop = [[] for _ in range(4)]
    med_cnt_list = []
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    recommended_drugs = set()

    all_pred_list = []
    all_label_list = []
    rec_results = [] # added

    ja_by_visit = [[] for _ in range(5)]
    auc_by_visit = [[] for _ in range(5)]
    pre_by_visit = [[] for _ in range(5)]
    recall_by_visit = [[] for _ in range(5)]
    f1_by_visit = [[] for _ in range(5)]
    smm_record_by_visit = [[] for _ in range(5)]

    for idx, data in enumerate(test_dataloader):
        diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data
        visit_cnt += seq_length.sum().item()

        output_logits, output_probs = test_recommend_batch(model, data, device, TOKENS, ddi_adj, args)

        labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2], END_TOKEN, device, training=False, testing=True, max_len=args.max_len)
        _, probs = output_flatten(medications, output_probs, seq_length, m_length_matrix, voc_size[2], END_TOKEN, device, training=False, testing=True, max_len=args.max_len)
        y_gt = []       
        y_pred = []    
        y_pred_label = [] 
        y_pred_prob = [] 

        label_hisory = []
        label_hisory_list = []
        pred_list = []
        jaccard_list = []
        def cal_jaccard(set1, set2):
            if not set1 or not set2:
                return 0
            set1 = set(set1)
            set2 = set(set2)
            a, b = len(set1 & set2), len(set1 | set2)
            return a/b
        def cal_overlap_num(set1, set2):
            count = 0
            for d in set1:
                if d in set2:
                    count += 1
            return count

        # 针对每一个admission的预测结果
        for label, prediction, prob_list in zip(labels, predictions, probs):
            label_hisory += label.tolist()  ### case study

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[label] = 1    # 01序列，表示正确的label
            y_gt.append(y_gt_tmp)

            out_list = []
            out_prob_list = []
            for med, prob in zip(prediction, prob_list):
                if med in [voc_size[2], voc_size[2]+1]:
                    break
                out_list.append(med)
                out_prob_list.append(prob[:-2]) # 去掉SOS与EOS符号

            ## case study
            if label_hisory:
                jaccard_list.append(cal_jaccard(prediction, label_hisory))
            pred_list.append(out_list)
            label_hisory_list.append(label.tolist())

            # 对于没预测的药物，取每个位置上平均的概率，否则直接取对应的概率
            # pred_out_prob_list = np.mean(out_prob_list, axis=0)
            if len(out_prob_list) > 0:
                pred_out_prob_list = np.max(out_prob_list, axis=0)
            # pred_out_prob_list = np.min(out_prob_list, axis=0)
            for i in range(904):
                if i in out_list:
                    pred_out_prob_list[i] = out_prob_list[out_list.index(i)][i]

            y_pred_prob.append(pred_out_prob_list)
            y_pred_label.append(out_list)
            recommended_drugs = set(out_list) | recommended_drugs

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            med_cnt += len(prediction)
            med_cnt_list.append(len(prediction))


        smm_record.append(y_pred_label)
        for i in range(min(len(labels), 5)):
            # single_ja, single_p, single_r, single_f1 = sequence_metric_v2(np.array(y_gt[i:i+1]), np.array(y_pred[i:i+1]), np.array(y_pred_label[i:i+1]))
            single_ja, single_auc, single_p, single_r, single_f1 = multi_label_metric(np.array([y_gt[i]]), np.array([y_pred[i]]), np.array([y_pred_prob[i]]))
            ja_by_visit[i].append(single_ja)
            auc_by_visit[i].append(single_auc)
            pre_by_visit[i].append(single_p)
            recall_by_visit[i].append(single_r)
            f1_by_visit[i].append(single_f1)
            smm_record_by_visit[i].append(y_pred_label[i:i+1])

        # 存储所有预测结果
        all_pred_list.append(pred_list)
        all_label_list.append(labels)
        records = [diseases[0].tolist(), procedures[0].tolist(), medications[0].tolist(), pred_list]
        
        rec_results.append(records) #added

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(idx, len(test_dataloader)))

        # 统计不同visit的指标
        if idx%200==0:
            print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
            print('count:', [len(buf) for buf in ja_by_visit])
            print('jaccard:', [np.mean(buf) for buf in ja_by_visit])
            print('auc:', [np.mean(buf) for buf in auc_by_visit])
            print('precision:', [np.mean(buf) for buf in pre_by_visit])
            print('recall:', [np.mean(buf) for buf in recall_by_visit])
            print('f1:', [np.mean(buf) for buf in f1_by_visit])
            print('DDI:', [ddi_rate_score(buf, ddi_adj_path) for buf in smm_record_by_visit])
    
    ## added for rec result
    os.makedirs(rec_results_path, exist_ok=True)
    rec_results_file = rec_results_path + '/' + 'rec_results.pkl'
    dill.dump(rec_results, open(rec_results_file, 'wb'))

    logging.info('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
    logging.info(f'count: {[len(buf) for buf in ja_by_visit]}')
    logging.info(f'jaccard: {[np.mean(buf) for buf in ja_by_visit]}')
    logging.info(f'auc: {[np.mean(buf) for buf in auc_by_visit]}')
    logging.info(f'precision: {[np.mean(buf) for buf in pre_by_visit]}')
    logging.info(f'recall: {[np.mean(buf) for buf in recall_by_visit]}')
    logging.info(f'f1: {[np.mean(buf) for buf in f1_by_visit]}')
    logging.info(f'DDI: {[ddi_rate_score(buf, ddi_adj_path) for buf in smm_record_by_visit]}')
    # pickle.dump(all_pred_list, open('out_list.pkl', 'wb'))
    # pickle.dump(all_label_list, open('out_list_gt.pkl', 'wb'))

    return smm_record, ja, prauc, avg_p, avg_r, avg_f1, med_cnt_list
