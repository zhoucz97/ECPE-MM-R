import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class dual_sample(object):
    def __init__(self, doc_id, doc_len, text, clause_list, clause_len_list, f_query_len_list, b_query_len_list,
                 emotion_category_list, emotion_token_list, pairs,
                 forward_query_list, backward_query_list, sentiment_query_list,
                 f_e_query_answer, f_c_query_answer_list, b_c_query_answer, b_e_query_answer_list,
                 sentiment_answer_list, forward_query_seg=None, backward_query_seg=None):
        # 最后两个放到makeData_dual.py文件中设置
        # print(text)
        # print(clause_list)
        # print(clause_len_list)
        # print(f_query_len_list)
        # print(b_query_len_list)
        # print(emotion_category_list)
        # print(emotion_token_list)
        # print(pairs)
        # print(forward_query_list)
        # print(backward_query_list)
        # print(sentiment_query_list)
        # print(f_e_query_answer)
        # print(f_c_query_answer_list)
        # print(b_c_query_answer)
        # print(b_e_query_answer_list)
        # print(sentiment_answer_list)
        # print('--------------')
        self.doc_id = doc_id
        self.doc_len = doc_len
        self.text = text
        self.clause_list = clause_list
        self.clause_len_list = clause_len_list
        self.emotion_category_list = emotion_category_list
        self.emotion_token_list = emotion_token_list
        self.pairs = pairs

        self.f_query_list = forward_query_list
        self.f_query_len_list = f_query_len_list
        self.f_e_query_answer = f_e_query_answer
        self.f_c_query_answer = f_c_query_answer_list
        self.f_query_seg = forward_query_seg

        self.b_query_list = backward_query_list
        self.b_query_len_list = b_query_len_list
        self.b_c_query_answer = b_c_query_answer
        self.b_e_query_answer = b_e_query_answer_list
        self.b_query_seg = backward_query_seg

        self.sentiment_query = sentiment_query_list  # 暂时是空
        self.sentiment_answer = sentiment_answer_list


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


def pre_processing(sample_list, max_len_dict):
    tokenizer = BertTokenizer.from_pretrained('pretrained_model/bert-base-chinese')
    _docid_list = []
    _clause_list = []
    _doc_len_list = []
    _clause_len_list = []
    _pairs = []

    _f_emo_query = []
    _f_cau_query = []
    _f_emo_query_len = []
    _f_cau_query_len = []
    _f_emo_query_answer = []
    _f_cau_query_answer = []
    _f_emo_query_mask = []
    _f_cau_query_mask = []
    _f_emo_query_seg = []
    _f_cau_query_seg = []
    _b_emo_query = []
    _b_cau_query = []
    _b_emo_query_len = []
    _b_cau_query_len = []
    _b_emo_query_answer = []
    _b_cau_query_answer = []
    _b_emo_query_mask = []
    _b_cau_query_mask = []
    _b_emo_query_seg = []
    _b_cau_query_seg = []

    _forward_c_num = []
    _backward_e_num = []

    for instance in sample_list:  # 对每一个文档
        _docid_list.append(instance.doc_id)
        _clause_list.append(instance.clause_list)
        _doc_len_list.append(instance.doc_len)
        _clause_len_list.append(instance.clause_len_list)
        _pairs.append(instance.pairs)
        #_f_emo_query_answer.append(instance.f_e_query_answer)
        #_f_cau_query_answer.append(instance.f_c_query_answer)
        _f_emo_query_len.append(instance.f_query_len_list[0:1])
        _f_cau_query_len.append(instance.f_query_len_list[1:])
        #_b_cau_query_answer.append(instance.b_c_query_answer)
        #_b_emo_query_answer.append(instance.b_e_query_answer)
        _b_cau_query_len.append(instance.b_query_len_list[0:1])
        _b_emo_query_len.append(instance.b_query_len_list[1:])

        f_query_list = instance.f_query_list
        f_query_seg_list = instance.f_query_seg
        b_query_list = instance.b_query_list
        b_query_seg_list = instance.b_query_seg
        _forward_c_num.append(len(f_query_list) - 1)
        _backward_e_num.append(len(b_query_list) - 1)

        # step1   对前向第一个问题：哪一个是情感子句？
        f_single_emotion_query = []
        f_single_emotion_query_mask = []
        f_single_emotion_query_seg = []
        f_emo_pad_num = max_len_dict['max_f_emo_len'] - len(f_query_list[0])
        # query
        f_single_emotion_query.append(tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_query_list[0]])
                                  + [0] * f_emo_pad_num)
        # mask
        f_single_emotion_query_mask.append([1] * len(f_query_list[0]) + [0] * f_emo_pad_num)
        # segment
        f_single_emotion_query_seg.append(f_query_seg_list[0] + [1] * f_emo_pad_num)
        # answer
        # for i in range(len(instance.f_e_query_answer)):
        #     instance.f_e_query_answer[i].extend([-1] * (max_len_dict['max_doc_len'] - len(instance.f_e_query_answer[i])))
        _f_emo_query_answer.append(instance.f_e_query_answer)
        _f_emo_query.append(f_single_emotion_query)
        _f_emo_query_seg.append(f_single_emotion_query_seg)
        _f_emo_query_mask.append(f_single_emotion_query_mask)

        # 对后面的前向限制性问题
        f_single_cause_query = []
        f_single_cause_query_mask = []
        f_single_cause_query_seg = []
        for i in range(1, len(f_query_list)):
            pad_num = max_len_dict['max_f_cau_len'] - len(f_query_list[i])
            # query
            f_single_cause_query.append(tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_query_list[i]])
                                      + [0] * pad_num)
            # mask
            f_single_cause_query_mask.append([1] * len(f_query_list[i]) + [0] * pad_num)
            # segment
            f_single_cause_query_seg.append(f_query_seg_list[i] + [1] * pad_num)
            assert len(f_single_cause_query[-1]) == len(f_single_cause_query_seg[-1]) == len(f_single_cause_query_mask[-1])
        # PAD: max_f_num
        _f_cau_query.append(f_single_cause_query)
        _f_cau_query[-1].extend([[0] * max_len_dict['max_f_cau_len']] * (max_len_dict['max_f_c_num'] - _forward_c_num[-1]))
        _f_cau_query_mask.append(f_single_cause_query_mask)
        _f_cau_query_mask[-1].extend([[0] * max_len_dict['max_f_cau_len']] * (max_len_dict['max_f_c_num'] - _forward_c_num[-1]))
        _f_cau_query_seg.append(f_single_cause_query_seg)
        _f_cau_query_seg[-1].extend([[0] * max_len_dict['max_f_cau_len']] * (max_len_dict['max_f_c_num'] - _forward_c_num[-1]))
        # answer
        # for i in range(len(instance.f_c_query_answer)):
        #     instance.f_c_query_answer[i].extend([-1] * (max_len_dict['max_doc_len'] - len(instance.f_c_query_answer[i])))
        # instance.f_c_query_answer.extend([[-1] * (max_len_dict['max_doc_len'])] * (max_len_dict['max_f_c_num'] - len(instance.f_c_query_answer)))
        _f_cau_query_answer.append(instance.f_c_query_answer)


        # step2   对后向第一个问题：哪一个是原因子句？
        b_single_cause_query = []
        b_single_cause_query_mask = []
        b_single_cause_query_seg = []
        b_cau_pad_num = max_len_dict['max_b_cau_len'] - len(b_query_list[0])
        b_single_cause_query.append(tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in b_query_list[0]])
                                  + [0] * b_cau_pad_num)
        # mask
        b_single_cause_query_mask.append([1] * len(b_query_list[0]) + [0] * b_cau_pad_num)
        # segment
        b_single_cause_query_seg.append(b_query_seg_list[0] + [1] * b_cau_pad_num)
        _b_cau_query.append(b_single_cause_query)
        _b_cau_query_mask.append(b_single_cause_query_mask)
        _b_cau_query_seg.append(b_single_cause_query_seg)
        # answer
        # for i in range(len(instance.b_c_query_answer)):
        #     instance.b_c_query_answer[i].extend([-1] * (max_len_dict['max_doc_len'] - len(instance.b_c_query_answer[i])))
        _b_cau_query_answer.append(instance.b_c_query_answer)

        # 对后面的限制性问题
        b_single_emotion_query = []
        b_single_emotion_query_mask = []
        b_single_emotion_query_seg = []
        for i in range(1, len(b_query_list)):
            pad_num = max_len_dict['max_b_emo_len'] - len(b_query_list[i])
            # query
            b_single_emotion_query.append(tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in b_query_list[i]])
                                      + [0] * pad_num)
            # mask
            b_single_emotion_query_mask.append([1] * len(b_query_list[i]) + [0] * pad_num)
            # segment
            b_single_emotion_query_seg.append(b_query_seg_list[i] + [1] * pad_num)
        # PAD: max_for_num
        _b_emo_query.append(b_single_emotion_query)
        _b_emo_query[-1].extend([[0] * max_len_dict['max_b_emo_len']] * (max_len_dict['max_b_e_num'] - _backward_e_num[-1]))
        _b_emo_query_mask.append(b_single_emotion_query_mask)
        _b_emo_query_mask[-1].extend([[0] * max_len_dict['max_b_emo_len']] * (max_len_dict['max_b_e_num'] - _backward_e_num[-1]))
        _b_emo_query_seg.append(b_single_emotion_query_seg)
        _b_emo_query_seg[-1].extend([[0] * max_len_dict['max_b_emo_len']] * (max_len_dict['max_b_e_num'] - _backward_e_num[-1]))
        # for i in range(len(instance.b_e_query_answer)):
        #     instance.b_e_query_answer[i].extend([-1] * (max_len_dict['max_doc_len'] - len(instance.b_e_query_answer[i])))
        # instance.b_e_query_answer.extend([[-1] * (max_len_dict['max_doc_len'])] * (max_len_dict['max_b_e_num'] - len(instance.b_e_query_answer)))
        _b_emo_query_answer.append(instance.b_e_query_answer)


        # if instance.doc_id == 8:
        #     print(_doc_len_list[-1])
        #     print(_clause_len_list[-1])
        #     print(_pairs[-1])
        #     print(_f_emo_query[-1])
        #     print(_f_emo_query_len[-1])
        #     print(_f_emo_query_answer[-1])
        #     print(_f_emo_query_mask[-1])
        #     print(_f_emo_query_seg[-1])
        #     print(_f_cau_query[-1])
        #     print(_f_cau_query_len[-1])
        #     print(_f_cau_query_answer[-1])
        #     print(_f_cau_query_mask[-1])
        #     print(_f_cau_query_seg[-1])
        #     print(_forward_c_num)
        #     print(_backward_e_num)
        #
        #     assert 1 == 2
    result =  {'_docid_list': _docid_list, '_clause_list': _clause_list, '_doc_len_list': _doc_len_list,
               '_clause_len_list': _clause_len_list, '_pairs': _pairs,
               '_f_emo_query': _f_emo_query, '_f_cau_query': _f_cau_query,
               '_f_emo_query_len': _f_emo_query_len, '_f_cau_query_len': _f_cau_query_len,
               '_f_emo_query_answer': _f_emo_query_answer, '_f_cau_query_answer': _f_cau_query_answer,
               "_f_emo_query_mask": _f_emo_query_mask, "_f_cau_query_mask": _f_cau_query_mask,
               "_f_emo_query_seg": _f_emo_query_seg, "_f_cau_query_seg": _f_cau_query_seg,
               "_b_emo_query": _b_emo_query, "_b_cau_query": _b_cau_query,
               '_b_emo_query_len': _b_emo_query_len, '_b_cau_query_len': _b_cau_query_len,
               '_b_emo_query_answer': _b_emo_query_answer, '_b_cau_query_answer': _b_cau_query_answer,
               "_b_emo_query_mask": _b_emo_query_mask, "_b_cau_query_mask": _b_cau_query_mask,
               "_b_emo_query_seg": _b_emo_query_seg, "_b_cau_query_seg": _b_cau_query_seg,
               "_forward_c_num": _forward_c_num, "_backward_e_num": _backward_e_num}
    return OriginalDataset(result)


def tokenized_data(data):
    max_f_emo_query_len = 0
    max_f_cau_query_len = 0
    max_b_emo_query_len = 0
    max_b_cau_query_len = 0
    max_f_c_querys_num = 0
    max_b_e_querys_num = 0
    max_doc_len = 0
    tokenized_sample_list = []
    for sample in data:  # 对每一个文档
        doc_len = sample.doc_len
        max_doc_len = max(max_doc_len, doc_len)
        f_querys, f_answers, f_querys_seg, b_querys, b_answers, b_querys_seg = [], [], [], [], [], []

        # 计算除了非限制性问题外，前向和后向的最大问题个数，便于后面做padding
        max_f_c_querys_num = max(max_f_c_querys_num, len(sample.f_query_list) - 1)
        max_b_e_querys_num = max(max_b_e_querys_num, len(sample.b_query_list) - 1)

        # forward questions
        for i in range(len(sample.f_query_list)):  # 对每个前向问题
            temp_query = sample.f_query_list[i]  # 是个长度为1的列表
            # ['啥是情感子句'] --->   ['啥', '是', '情', ....]
            temp_query = ' '.join(temp_query[0]).split(' ')
            temp_text = sample.text
            # text也做同样的操作
            temp_text = ' '.join(temp_text).split(' ')
            temp_qa = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_qa_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)
            assert len(temp_qa) == len(temp_qa_seg)

            if i == 0:  # 非限制性问题
                max_f_emo_query_len = max(max_f_emo_query_len, len(temp_qa))
            else:
                max_f_cau_query_len = max(max_f_cau_query_len, len(temp_qa))
            f_querys.append(temp_qa)
            f_querys_seg.append(temp_qa_seg)

        # backward questions
        for i in range(len(sample.b_query_list)):  # 对每个后向问题
            temp_query = sample.b_query_list[i]  # 是个长度为1的列表
            # ['啥是原因子句'] --->   ['啥', '是', '原', ....]
            temp_query = ' '.join(temp_query[0]).split(' ')
            temp_text = sample.text
            # text也做同样的操作
            temp_text = ' '.join(temp_text).split(' ')
            temp_qa = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_qa_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)
            assert len(temp_qa) == len(temp_qa_seg)

            if i == 0:
                max_b_cau_query_len = max(max_b_cau_query_len, len(temp_qa))
            else:
                max_b_emo_query_len = max(max_b_emo_query_len, len(temp_qa))
            b_querys.append(temp_qa)
            b_querys_seg.append(temp_qa_seg)

        sample.f_query_list = f_querys
        sample.b_query_list = b_querys
        sample.f_query_seg = f_querys_seg
        sample.b_query_seg = b_querys_seg
        tokenized_sample_list.append(sample)
        # if sample.doc_id == 8:
        #     print(sample.doc_len)
        #     print(sample.text)
        #     print(sample.clause_list)
        #     print(sample.clause_len_list)
        #     print(sample.emotion_category_list)
        #     print(sample.emotion_token_list)
        #     print(sample.pairs)
        #     print(sample.f_query_list)
        #     print(sample.f_query_len_list)
        #     print(sample.f_e_query_answer)
        #     print(sample.f_c_query_answer)
        #     print(sample.f_query_seg)
        #
        #     print(sample.b_query_list)
        #     print(sample.b_query_len_list)
        #     print(sample.b_c_query_answer)
        #     print(sample.b_e_query_answer)
        #     print(sample.b_query_seg)
        #
        #     print(sample.sentiment_query)
        #     print(sample.sentiment_answer)
        #     assert 1 == 2

    return tokenized_sample_list, {'max_f_emo_len': max_f_emo_query_len,
                                   'max_f_cau_len': max_f_cau_query_len,
                                   'max_b_emo_len': max_b_emo_query_len,
                                   'max_b_cau_len': max_b_cau_query_len,
                                   'max_f_c_num': max_f_c_querys_num,
                                   'max_b_e_num': max_b_e_querys_num,
                                   'max_doc_len': max_doc_len}


if __name__ == '__main__':
    dataset_name_list = []
    for i in range(1, 11):
        dataset_name_list.append('fold{}'.format(i))
    for dataset_name in dataset_name_list:
        output_path = 'data/preprocess/' + dataset_name + '.pt'
        train_data = torch.load('data/preprocess/' + dataset_name + '_train_dual.pt')
        test_data = torch.load('data/preprocess/' + dataset_name + '_test_dual.pt')

        train_tokenized, train_max_len = tokenized_data(train_data)
        test_tokenized, test_max_len = tokenized_data(test_data)

        train_preprocess = pre_processing(train_tokenized, train_max_len)
        test_preprocess = pre_processing(test_tokenized, test_max_len)
        print(output_path, ' finish.')
        torch.save({'train': train_preprocess, 'test': test_preprocess}, output_path)