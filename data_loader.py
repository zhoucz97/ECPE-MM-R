import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def build_dataset(configs, dataset, mode='train'):
    dataset = MyDataSet(dataset)
    if mode == 'train':
        data_loader = DataLoader(dataset=dataset, batch_size=configs.batch_size, shuffle=True,
                             collate_fn=bert_batch_preprocessing)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                                 collate_fn=bert_batch_preprocessing)
    return data_loader


def bert_batch_preprocessing(batch):
    docid_list, clause_list, doc_len_list, clause_len_list, pairs, \
    feq, feq_len, feq_an, feq_mask, feq_seg, fcq, fcq_len, fcq_an, fcq_mask, fcq_seg, \
    bcq, bcq_len, bcq_an, bcq_mask, bcq_seg, beq, beq_len, beq_an, beq_mask, beq_seg, \
    fc_num, be_num = zip(*batch)

    # query, query_mask, query_seg, answer已经在makeData_dual中padding好了，还需要设置answer mask
    feq_an, fe_an_mask = get_answer_pad_mask(feq_an)
    bcq_an, bc_an_mask = get_answer_pad_mask(bcq_an)
    fcq_an, fc_an_mask = get_answer_pad_mask(fcq_an)
    beq_an, be_an_mask = get_answer_pad_mask(beq_an)

    feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj \
        = convert_batch(feq, feq_mask, feq_seg, feq_len, clause_len_list, doc_len_list)
    fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj \
        = convert_batch(fcq, fcq_mask, fcq_seg, fcq_len, clause_len_list, doc_len_list)
    bcq, bcq_mask, bcq_seg, bcq_len, bc_clause_len, bc_doc_len, bc_adj \
        = convert_batch(bcq, bcq_mask, bcq_seg, bcq_len, clause_len_list, doc_len_list)
    beq, beq_mask, beq_seg, beq_len, be_clause_len, be_doc_len, be_adj \
        = convert_batch(beq, beq_mask, beq_seg, beq_len, clause_len_list, doc_len_list)

    return docid_list, clause_list, pairs, \
           feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj, feq_an, fe_an_mask, \
           fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj, fcq_an, fc_an_mask, \
           bcq, bcq_mask, bcq_seg, bcq_len, bc_clause_len, bc_doc_len, bc_adj, bcq_an, bc_an_mask, \
           beq, beq_mask, beq_seg, beq_len, be_clause_len, be_doc_len, be_adj, beq_an, be_an_mask


class MyDataSet(Dataset):
    def __init__(self, pre_data):
        self.docid_list = pre_data.docid_list
        self.clause_list = pre_data.clause_list
        self.doc_len_list = pre_data.doc_len_list
        self.clause_len_list = pre_data.clause_len_list
        self.pairs = pre_data.pairs

        self._f_emo_query = pre_data._f_emo_query  # [1, max_for_emo_len]
        self._f_emo_query_len = pre_data._f_emo_query_len
        self._f_emo_query_answer = pre_data._f_emo_query_answer
        self._f_emo_query_mask = pre_data._f_emo_query_mask  # [1,max_for_emo_len]
        self._f_emo_query_seg = pre_data._f_emo_query_seg  # [1,max_for_emo_len]
        self._f_cau_query = pre_data._f_cau_query  # [max_for_num, max_for_cau_len]
        self._f_cau_query_len = pre_data._f_cau_query_len
        self._f_cau_query_answer = pre_data._f_cau_query_answer
        self._f_cau_query_mask = pre_data._f_cau_query_mask  # [max_for_num, max_for_cau_len]
        self._f_cau_query_seg = pre_data._f_cau_query_seg  # [max_for_num, max_for_cau_len]

        self._b_cau_query = pre_data._b_cau_query  #
        self._b_cau_query_len = pre_data._b_cau_query_len
        self._b_cau_query_answer = pre_data._b_cau_query_answer
        self._b_cau_query_mask = pre_data._b_cau_query_mask  #
        self._b_cau_query_seg = pre_data._b_cau_query_seg  #
        self._b_emo_query = pre_data._b_emo_query
        self._b_emo_query_len = pre_data._b_emo_query_len
        self._b_emo_query_answer = pre_data._b_emo_query_answer
        self._b_emo_query_mask = pre_data._b_emo_query_mask  #
        self._b_emo_query_seg = pre_data._b_emo_query_seg  #

        self._forward_c_num = pre_data._forward_c_num
        self._backward_e_num = pre_data._backward_e_num

        # print(self.doc_len_list)
        # print(self.clause_len_list)
        # print(self.pairs)
        # print(self._f_query_len_list)
        # print(self._b_query_len_list)
        # print(self._f_cau_query[0])
        # print(self._f_cau_query_len[0])
        # print(self._f_cau_query_answer[0])
        # print(self._f_cau_query_mask[0])
        # print(self._f_cau_query_seg[0])
        # exit(0)

        
        # print(self._f_cau_query)
        # print(self._f_emo_query_answer)
        # print(self._f_cau_query_answer)
        # print(self._f_emo_query_mask)
        # print(self._f_cau_query_mask)
        # print(self._f_emo_query_seg)
        # print(self._f_cau_query_seg)
        # print(self._b_emo_query)
        # print(self._b_cau_query)
        # print(self._b_emo_query_answer)
        # print(self._b_cau_query_answer)
        # print(self._b_emo_query_mask)
        # print(self._b_cau_query_mask)
        # print(self._b_emo_query_seg)
        # print(self._b_cau_query_seg)
        # print(self._forward_num)
        # print(self._backward_num)

    def __len__(self):
        return len(self.doc_len_list)

    def __getitem__(self, i):
        docid_list, clause_list, doc_len_list, clause_len_list, pairs, \
        feq, feq_len, feq_an, feq_mask, feq_seg, fcq, fcq_len, fcq_an, fcq_mask, fcq_seg, \
        bcq, bcq_len, bcq_an, bcq_mask, bcq_seg, beq, beq_len, beq_an, beq_mask, beq_seg, \
        fc_num, be_num  = \
            self.docid_list[i], self.clause_list[i], self.doc_len_list[i], self.clause_len_list[i], self.pairs[i], \
            self._f_emo_query[i], self._f_emo_query_len[i], self._f_emo_query_answer[i], self._f_emo_query_mask[i], self._f_emo_query_seg[i], \
            self._f_cau_query[i], self._f_cau_query_len[i], self._f_cau_query_answer[i], self._f_cau_query_mask[i], self._f_cau_query_seg[i], \
            self._b_cau_query[i], self._b_cau_query_len[i], self._b_cau_query_answer[i], self._b_cau_query_mask[i], self._b_cau_query_seg[i], \
            self._b_emo_query[i], self._b_emo_query_len[i], self._b_emo_query_answer[i], self._b_emo_query_mask[i], self._b_emo_query_seg[i], \
            self._forward_c_num[i], self._backward_e_num[i]
        return docid_list, clause_list, doc_len_list, clause_len_list, pairs, \
        feq, feq_len, feq_an, feq_mask, feq_seg, fcq, fcq_len, fcq_an, fcq_mask, fcq_seg, \
        bcq, bcq_len, bcq_an, bcq_mask, bcq_seg, beq, beq_len, beq_an, beq_mask, beq_seg, \
        fc_num, be_num



def get_answer_pad_mask(answer):
    new_answer = []
    for batch_answer in answer:
        for qa_answer in batch_answer:
            new_answer.append(torch.tensor(qa_answer))
    answer = pad_sequence(new_answer, padding_value=-1).transpose(0, 1)
    mask = torch.where(answer != -1, 1, 0)
    assert mask.shape == answer.shape
    return answer, mask


def convert_batch(query, query_mask, query_seg, query_len, seq_len, doc_len):
    query_list, query_mask_list, query_seg_list = [], [], []
    new_query_len, new_seq_len, new_doc_len = [], [], []
    for i in range(len(query_len)):
        for j in range(len(query_len[i])):
            query_list.append(query[i][j])
            query_mask_list.append(query_mask[i][j])
            query_seg_list.append(query_seg[i][j])
            new_seq_len.append(seq_len[i])
            new_doc_len.append(doc_len[i])
            new_query_len.append(query_len[i][j])

    query = torch.LongTensor(query_list)
    query_mask = torch.LongTensor(query_mask_list)
    query_seg = torch.LongTensor(query_seg_list)
    adj = pad_matrices(new_doc_len)
    return query, query_mask, query_seg, new_query_len, new_seq_len, new_doc_len, adj



def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad.extend(pad_mark_list)
    return element_list_pad

def pad_docs(doc_len_b, answer):
    max_doc_len = max(doc_len_b)
    y_mask_b, ans_b = [], []
    for ans in answer:
        ans_ = pad_list(ans, max_doc_len, -1)
        y_mask = list(map(lambda x: 0 if x == -1 else 1, ans_))
        y_mask_b.append(y_mask)
        ans_b.append(ans_)
    return y_mask_b, ans_b

def pad_matrices(doc_len_b):
    N = max(doc_len_b)
    adj_b = []
    for doc_len in doc_len_b:
        adj = np.ones((doc_len, doc_len))
        adj = sp.coo_matrix(adj)
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                            shape=(N, N), dtype=np.float32)
        adj_b.append(adj.toarray())
    return adj_b


if __name__ == "__main__":
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained('../bert-base-chinese')
    print(tok.encode('情感子句'))