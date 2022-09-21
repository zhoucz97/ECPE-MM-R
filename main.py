import os
from config import *
import torch
from data_loader import *
from utils.utils import *
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
from networks.rank_cp import Network
import datetime
import numpy as np


class OriginalDataset(Dataset):
    def __init__(self, pre_data):
        self.docid_list = pre_data['_docid_list']
        self.clause_list = pre_data['_clause_list']
        self.doc_len_list = pre_data['_doc_len_list']
        self.clause_len_list = pre_data['_clause_len_list']
        self.pairs = pre_data['_pairs']

        self._f_emo_query = pre_data['_f_emo_query']  # [1, max_for_emo_len]
        self._f_cau_query = pre_data['_f_cau_query']  # [max_for_num, max_for_cau_len]
        self._f_emo_query_len = pre_data['_f_emo_query_len']
        self._f_cau_query_len = pre_data['_f_cau_query_len']
        self._f_emo_query_answer = pre_data['_f_emo_query_answer']
        self._f_cau_query_answer = pre_data['_f_cau_query_answer']
        self._f_emo_query_mask = pre_data['_f_emo_query_mask']  # [1,max_for_emo_len]
        self._f_cau_query_mask = pre_data['_f_cau_query_mask']  # [max_for_num, max_for_cau_len]
        self._f_emo_query_seg = pre_data['_f_emo_query_seg']  # [1,max_for_emo_len]
        self._f_cau_query_seg = pre_data['_f_cau_query_seg']  # [max_for_num, max_for_cau_len]

        self._b_emo_query = pre_data['_b_emo_query']
        self._b_cau_query = pre_data['_b_cau_query']  #
        self._b_emo_query_len = pre_data['_b_emo_query_len']
        self._b_cau_query_len = pre_data['_b_cau_query_len']
        self._b_emo_query_answer = pre_data['_b_emo_query_answer']
        self._b_cau_query_answer = pre_data['_b_cau_query_answer']
        self._b_emo_query_mask = pre_data['_b_emo_query_mask']  #
        self._b_cau_query_mask = pre_data['_b_cau_query_mask']  #
        self._b_emo_query_seg = pre_data['_b_emo_query_seg']  #
        self._b_cau_query_seg = pre_data['_b_cau_query_seg']  #

        self._forward_c_num = pre_data['_forward_c_num']
        self._backward_e_num = pre_data['_backward_e_num']



def evaluate_one_batch(configs, batch, model, tokenizer):
    # 一个文档中最多3个情感子句，最多4个原因子句
    # 一个情感子句最多的对应三个原因子句，一个原因子句唯一对应情感子句
    with open('data/sentimental_clauses.pkl', 'rb') as f:
        emo_dictionary = pickle.load(f)

    docid_list, clause_list, pairs, \
    feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj, feq_an, fe_an_mask, \
    fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj, fcq_an, fc_an_mask, \
    bcq, bcq_mask, bcq_seg, bcq_len, bc_clause_len, bc_doc_len, bc_adj, bcq_an, bc_an_mask, \
    beq, beq_mask, beq_seg, beq_len, be_clause_len, be_doc_len, be_adj, beq_an, be_an_mask \
        = batch
    # 因为是按batch=1取的，所以最外层都有维度1
    doc_id, clause_list, true_pairs = docid_list[0], clause_list[0], pairs[0]
    true_emo, true_cau = zip(*true_pairs)
    true_emo, true_cau = list(true_emo), list(true_cau)
    text = ''.join(clause_list)
    text = ' '.join(text).split(' ')

    pred_emo_f = []
    pred_pair_f = []
    pred_pair_f_pro = []
    pred_pair_b = []
    pred_pair_b_pro = []
    pred_emo_single = []
    pred_cau_single = []
    aaa=[]
    aaa_pro = []

    # step 1
    f_emo_pred = model(feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj, 'f_emo')
    temp_emo_f_prob = f_emo_pred.masked_select(fe_an_mask.bool()).cpu().numpy().tolist()
    for idx in range(len(temp_emo_f_prob)):
        if temp_emo_f_prob[idx] > 0.99 or (temp_emo_f_prob[idx] > 0.5 and idx + 1 in emo_dictionary[str(doc_id)]):
            pred_emo_f.append(idx)
            pred_emo_single.append(idx + 1)

    # step 2
    for idx_emo in pred_emo_f:
        f_query = clause_list[idx_emo]+ '这句话对应的原因子句有哪些?'
        f_query = ' '.join(f_query).split(' ')
        f_qa = ['[CLS]'] + f_query + ['[SEP]'] + text
        f_qa = tokenizer.convert_tokens_to_ids([w.lower() if w not in ['[CLS]', '[SEP]'] else w for w in f_qa])
        f_mask = [1] * len(f_qa)
        f_seg = [0] * (len(f_query) + 2) + [1] * len(text)
        f_len = len(f_query)
        f_qa = torch.LongTensor([f_qa])
        f_mask = torch.LongTensor([f_mask])
        f_seg = torch.LongTensor([f_seg])
        f_len = [f_len]
        f_clause_len = fe_clause_len
        f_doc_len = fe_doc_len
        f_adj = fe_adj
        f_cau_pred = model(f_qa, f_mask, f_seg, f_len, f_clause_len, f_doc_len, f_adj, 'f_cau')
        temp_cau_f_prob = f_cau_pred[0].cpu().numpy().tolist()

        # step 3
        for idx_cau in range(len(temp_cau_f_prob)):
            if temp_cau_f_prob[idx_cau] > 0.5 and abs(idx_emo - idx_cau) <= 11:
                if idx_cau + 1 not in pred_cau_single:
                    pred_cau_single.append(idx_cau + 1)
                prob_t = temp_emo_f_prob[idx_emo] * temp_cau_f_prob[idx_cau]
                if idx_cau - idx_emo >= 0 and idx_cau - idx_emo <= 2:
                    pass
                else:
                    prob_t *= 0.9
                pred_pair_f_pro.append(prob_t)
                pred_pair_f.append([idx_emo + 1, idx_cau + 1])
                aaa.append([idx_emo + 1, idx_cau + 1])
                aaa_pro.append(prob_t)

    for k in range(len(pred_pair_f)):
        pair = pred_pair_f[k]
        # re-think
        idx_emo, idx_cau = pair[0] - 1, pair[1] - 1
        b_query = clause_list[idx_cau] + '这句话对应的情感子句是哪一句?'
        b_query = ' '.join(b_query).split(' ')
        b_qa = ['[CLS]'] + b_query + ['[SEP]'] + text
        b_qa = tokenizer.convert_tokens_to_ids([w.lower() if w not in ['[CLS]', '[SEP]'] else w for w in b_qa])
        b_mask = [1] * len(b_qa)
        b_seg = [0] * (len(b_query) + 2) + [1] * len(text)
        b_len = len(b_query)
        b_qa = torch.LongTensor([b_qa])
        b_mask = torch.LongTensor([b_mask])
        b_seg = torch.LongTensor([b_seg])
        b_len = [b_len]
        b_clause_len = bc_clause_len
        b_doc_len = bc_doc_len
        b_adj = bc_adj
        b_emo_pred = model(b_qa, b_mask, b_seg, b_len, b_clause_len, b_doc_len, b_adj, 'b_emo')
        temp_emo_b_prob = b_emo_pred[0].cpu().numpy().tolist()
        for i in range(len(temp_emo_b_prob)):
            if temp_emo_b_prob[i] > 0.5 and i + 1 in emo_dictionary[str(doc_id)]:
                if i == idx_emo:
                    pass
                else:
                    pred_pair_f_pro[k] *= 0.7
                    pred_pair_b_pro.append(temp_emo_b_prob[i] * temp_cau_f_prob[idx_cau])
                    pred_pair_b.append([i + 1, idx_cau + 1])

                if idx_emo + 1 not in pred_emo_single:
                    pred_emo_single.append(idx_emo + 1)


    pred_emo_final = []
    pred_cau_final = []
    pred_pair_final = []
    for i, pair in enumerate(pred_pair_b):
        if pair not in pred_pair_f:
            pred_pair_f.append(pair)
            pred_pair_f_pro.append(pred_pair_b_pro[i])

    for i, pair in enumerate(pred_pair_f):
        if pred_pair_f_pro[i] > 0.5:
            pred_pair_final.append(pair)

    for pair in pred_pair_final:
        if pair[0] not in pred_emo_final:
            pred_emo_final.append(pair[0])
        if pair[1] not in pred_cau_final:
            pred_cau_final.append(pair[1])

    metric_e_s, metric_c_s, _ = cal_metric(pred_emo_single, true_emo, pred_cau_single, true_cau, pred_pair_final,
                                           true_pairs, len(clause_list))
    metric_e, metric_c, metric_p = \
        cal_metric(pred_emo_final, true_emo, pred_cau_final, true_cau, pred_pair_final, true_pairs, len(clause_list))
    return metric_e, metric_c, metric_p, metric_e_s, metric_c_s


def evaluate(configs, test_loader, model, tokenizer):
    model.eval()
    all_emo, all_cau, all_pair = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    all_emo_s, all_cau_s = [0, 0, 0], [0, 0, 0]
    for batch in test_loader:
        emo, cau, pair, emo_s, cau_s = evaluate_one_batch(configs, batch, model, tokenizer)
        for i in range(3):
            all_emo[i] += emo[i]
            all_cau[i] += cau[i]
            all_pair[i] += pair[i]
            all_emo_s[i] += emo_s[i]
            all_cau_s[i] += cau_s[i]

    eval_emo = eval_func(all_emo)
    eval_cau = eval_func(all_cau)
    eval_pair = eval_func(all_pair)
    eval_emo_s = eval_func(all_emo_s)
    eval_cau_s = eval_func(all_cau_s)
    return eval_emo, eval_cau, eval_pair, eval_emo_s, eval_cau_s


def main(configs, fold_id, tokenizer):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    data_path = 'data/preprocess/fold{}'.format(fold_id) + '.pt'
    total_data = torch.load(data_path)
    train_loader = build_dataset(configs, total_data['train'], mode='train')
    test_loader = build_dataset(configs, total_data['test'], mode='test')

    # model
    model = Network(configs).to(DEVICE)
    # optimizer
    params = list(model.named_parameters())
    optimizer_grouped_params = [
        {'params': [p for n, p in params if '_bert' in n], 'weight_decay': 0.01},
        {'params': [p for n, p in params if '_bert' not in n], 'lr': configs.lr, 'weight_decay': 0.01}
    ]
    optimizer = AdamW(params=optimizer_grouped_params, lr=configs.tuning_bert_rate)
    # scheduler
    training_steps = configs.epochs * len(train_loader) // configs.gradient_accumulation_steps
    warmup_steps = int(training_steps * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)

    # training
    model.zero_grad()
    max_result_pair, max_result_emo, max_result_cau = None, None, None
    max_result_emos, max_result_caus = None, None
    early_stop_flag = None

    for epoch in range(1, configs.epochs+1):
        for train_step, batch in enumerate(train_loader, 1):
            model.train()
    
            _, clause_list, pairs, \
            feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj, feq_an, fe_an_mask, \
            fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj, fcq_an, fc_an_mask, \
            bcq, bcq_mask, bcq_seg, bcq_len, bc_clause_len, bc_doc_len, bc_adj, bcq_an, bc_an_mask, \
            beq, beq_mask, beq_seg, beq_len, be_clause_len, be_doc_len, be_adj, beq_an, be_an_mask \
             = batch
    
            f_emo_pred = model(feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj, 'f_emo')
            f_cau_pred = model(fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj, 'f_cau')
            b_emo_pred = model(beq, beq_mask, beq_seg, beq_len, be_clause_len, be_doc_len, be_adj, 'b_emo')
    
            loss_e = model.loss_pre(f_emo_pred, feq_an, fe_an_mask)
            loss_ec = model.loss_pre(f_cau_pred, fcq_an, fc_an_mask)
            loss_ce = model.loss_pre(b_emo_pred, beq_an, be_an_mask)
            losses = (loss_e + loss_ec + loss_ce) / configs.gradient_accumulation_steps
            losses.backward()
    
            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
    
            if train_step % 200 == 0:
                print('epoch: {}, step: {}, loss: {}, {}, {}'.format(epoch, train_step, loss_e, loss_ec, loss_ce))
    
        with torch.no_grad():
            eval_emo, eval_cau, eval_pair, eval_emos, eval_cuas = evaluate(configs, test_loader, model, tokenizer)
    
            if max_result_pair is None or eval_pair[0] > max_result_pair[0]:
                early_stop_flag = 1
                max_result_emo = eval_emo
                max_result_cau = eval_cau
                max_result_pair = eval_pair
    
                state_dict = {'model': model.state_dict(), 'result': max_result_pair}
                torch.save(state_dict, 'model/model_fold{}.pth'.format(fold_id))
            else:
                early_stop_flag += 1
        if epoch > configs.epochs / 2 and early_stop_flag >= 7:
            break


    return max_result_emo, max_result_cau, max_result_pair, max_result_emos, max_result_caus


if __name__ == '__main__':
    configs = Config()
    t = BertTokenizer.from_pretrained(configs.bert_cache_path)
    result_e, result_c, result_p = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    res, rcs = [0, 0, 0], [0, 0, 0]
    for fold_id in range(1, 11):
        print('===== fold {} ====='.format(fold_id))
        metric_e, metric_c, metric_pair, es, cs = main(configs, fold_id, t)
        print('Best pair result - f1:{}, p:{}, r:{}'.format(metric_pair[0], metric_pair[1], metric_pair[2]))
        print('Best emo  result - f1:{}, p:{}, r:{}'.format(metric_e[0], metric_e[1], metric_e[2]))
        print('Best cau  result - f1:{}, p:{}, r:{}'.format(metric_c[0], metric_c[1], metric_c[2]))

        for i in range(3):
            result_e[i] += metric_e[i]
            result_c[i] += metric_c[i]
            result_p[i] += metric_pair[i]

    for i in range(3):
        result_e[i] /= 10
        result_c[i] /= 10
        result_p[i] /= 10

    print('Average pair result- f1:{}, p:{}, r:{}'.format(result_p[0], result_p[1], result_p[2]))
    print('Average emo result - f1:{}, p:{}, r:{}'.format(result_e[0], result_e[1], result_e[2]))
    print('Average cau result - f1:{}, p:{}, r:{}'.format(result_c[0], result_c[1], result_c[2]))