import json
import torch

sentiment_category = {'null': 0, 'happiness': 1, 'sadness': 2, 'disgust': 3, 'surprise': 4, 'fear': 5, 'anger': 6}

def data_bert_trunk(doc_len, doc_couples, clauses, emotion_categorys, emotion_tokens):
    sum_len = 0
    for clause in clauses:
        sum_len += (2 + len(clause))  # 2 是 [CLS], [SEP]的长度
    if sum_len > 375:  # trunk
        # 375是根据数据集的特点事先算好的，只要总文档长度不大于375，那么加上query就能小于512
        pair = doc_couples[0]
        half_len = doc_len // 2
        if pair[0] <= half_len and pair[1] <= half_len:  # 取文档前半部分
            doc_len = half_len
            clauses = clauses[: half_len]
            emotion_categorys = emotion_categorys[: half_len]
            emotion_tokens = emotion_tokens[: half_len]
        else:   # 取文档后半部分
            doc_len = doc_len - half_len
            for i in range(len(doc_couples)):
                doc_couples[i][0] -= half_len
                doc_couples[i][1] -= half_len
            clauses = clauses[half_len: ]
            emotion_categorys = emotion_categorys[half_len: ]
            emotion_tokens = emotion_tokens[half_len:]
    assert doc_len == len(clauses) == len(emotion_categorys) == len(emotion_tokens)
    return doc_len, doc_couples, clauses, emotion_categorys, emotion_tokens


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



if __name__ == '__main__':
    dataset_name_list = []
    for i in range(1, 11):
        dataset_name_list.append('fold{}'.format(i))
    dataset_type_list = ['train', 'test']
    for dataset_name in dataset_name_list:
        for dataset_type in dataset_type_list:  # 对每个文件
            output_path = 'data/preprocess/' + dataset_name + '_' + dataset_type + '_dual.pt'
            input_path = 'data/split10/' + dataset_name + '_' + dataset_type + '.json'
            sample_list = []
            with open(input_path, 'r', encoding='utf-8') as file:
                dataset = json.load(file)
            for doc in dataset:  # 对每个文档
                doc_id = int(doc['doc_id'])
                doc_len = int(doc['doc_len'])
                doc_couples = doc['pairs']
                doc_clauses = doc['clauses']
                clause_list = []
                emotion_categorys = []
                emotion_tokens = []
                for i in range(len(doc_clauses)):
                    clause_list.append(doc_clauses[i]['clause'])
                    emotion_category = doc_clauses[i]['emotion_category']
                    if '&' in emotion_category:
                        emotion_category = emotion_category.split('&')[0]
                    emotion_categorys.append(emotion_category)
                    emotion_tokens.append(doc_clauses[i]['emotion_token'])
                doc_len, doc_couples, clause_list, emotion_categorys, emotion_tokens = \
                    data_bert_trunk(doc_len, doc_couples, clause_list, emotion_categorys, emotion_tokens)
                emotion_list, cause_list = zip(*doc_couples)
                emotion_list = list(set(emotion_list))
                cause_list = list(set(cause_list))
                clause_len_list = [len(clause) for clause in clause_list]
                assert  len(clause_list) == len(clause_len_list) == len(emotion_categorys) == len(emotion_tokens)
                text = ''.join(clause_list)
                # 上面代码得到doc_id, doc_len, doc_couples, emotion_list, cause_list,
                # text, clause_list, clause_len_list, emotion_categorys, emotion_tokens

                # 开始构造
                forward_query_list = []
                backward_query_list = []
                f_e_query_answer = []
                b_e_query_answer = []
                f_c_query_answer = []
                b_c_query_answer = []
                f_query_len_list = []
                b_query_len_list = []
                sentiment_query_list = []  # 暂时无用
                sentiment_answer_list = [] # 暂时无用

                forward_query_list.append(["这是情感子句吗?"])
                backward_query_list.append(["这是原因子句吗?"])
                f_query_len_list.append(8)
                b_query_len_list.append(8)
                #  给出这两个问题的答案
                f_e_query_answer = [[0] * doc_len]
                b_c_query_answer = [[0] * doc_len]
                for emo_idx in emotion_list:
                    f_e_query_answer[0][emo_idx - 1] = 1
                    sc = sentiment_category[emotion_categorys[emo_idx - 1]]
                    sentiment_answer_list.append(sc)
                for cau_idx in cause_list:
                    b_c_query_answer[0][cau_idx - 1] = 1

                # 构造限制性问题及答案
                temp_emotion = set()
                for pair in doc_couples:  # 注意，一个情感子句可能对应多个原因子句，所以会有[[10, 8], [10,9]]的情况
                    emotion_idx = pair[0]
                    cause_idx = pair[1]
                    if emotion_idx not in temp_emotion:
                        causes = []
                        for e, c in doc_couples:
                            if e == emotion_idx:
                                causes.append(c)
                        query_f = clause_list[emotion_idx - 1] + '这句话对应的原因子句有哪些?'
                        forward_query_list.append([query_f])
                        f_query_len_list.append(len(query_f))
                        f_query2_answer = [0] * doc_len
                        for c_idx in causes:
                            f_query2_answer[c_idx - 1] = 1
                        f_c_query_answer.append(f_query2_answer)
                        temp_emotion.add(emotion_idx)
                    query_b = clause_list[cause_idx - 1] + '这句话对应的情感子句是哪一句?'
                    backward_query_list.append([query_b])
                    b_query_len_list.append(len(query_b))
                    b_query2_answer = [0] * doc_len
                    b_query2_answer[emotion_idx - 1] = 1
                    b_e_query_answer.append(b_query2_answer)

                # if doc_id == 204:
                #     print(doc_id)
                #     print(doc_len)
                #     print(text)
                #     print(clause_list)
                #     print(clause_len_list)
                #     print(f_query_len_list)
                #     print(b_query_len_list)
                #     print(emotion_categorys)
                #     print(emotion_tokens)
                #     print(doc_couples)
                #     print(forward_query_list)
                #     print(backward_query_list)
                #     print(sentiment_query_list)
                #     print(f_e_query_answer)
                #     print(f_c_query_answer)
                #     print(b_c_query_answer)
                #     print(b_e_query_answer)
                #     print(sentiment_answer_list)
                #     print('--------------')
                #     assert 1 == 2
                temp_sample = dual_sample(doc_id, doc_len, text, clause_list, clause_len_list, f_query_len_list,
                                          b_query_len_list, emotion_categorys, emotion_tokens, doc_couples,
                                          forward_query_list, backward_query_list, sentiment_query_list,
                                          f_e_query_answer, f_c_query_answer, b_c_query_answer, b_e_query_answer,
                                          sentiment_answer_list, None, None)
                sample_list.append(temp_sample)
            torch.save(sample_list, output_path)
            print(output_path, ' build finish!')