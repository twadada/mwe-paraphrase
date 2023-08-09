# -*- coding: utf-8 -*-
import time
import torch
import os
import string
import numpy as np
import pickle
import torch.nn.functional  as F
import argparse
from utils import tokenise_phrase, Encode_BERT_PAD,  Get_model,Identify_Key_Indices
from tqdm import tqdm

def T5_generate_paraprhase(sentences_raw, model,tokenizer,mask_ids,all_special_ids,punctuation_ids,num_beams,max_num_word):
    #sentences_raw: N_sent
    bs = 4
    mask_ids_replace = [tokenizer.mask_token_id] #Always Use one mask
    sentences_maskedList = []
    all_candidates = []
    all_candidates_ids= []
    prediction_emb_List = []
    for j in range(0, len(sentences_raw), bs):
        # NOTE: Attention mask for PADDING?
        sentences_tmp = tokenizer(sentences_raw[j:j + bs], padding=True)["input_ids"]
        sentences = []
        for x in sentences_tmp:
            assert len(x) < 512
            sentences.append(x)
        sentences_masked, mask_row_idx, mask_col_idx, \
        _, _, _ = \
            Identify_Key_Indices(sentences, mask_ids, mask_ids_replace, 0, tokenizer,
                                 [], "span_two", None)
        # gold_sent = " ".join(tokenizer.convert_ids_to_tokens(sentences_masked[0]))
        sentences_masked = torch.LongTensor(sentences_masked).to(model.device)
        sentences_maskedList.append(sentences_masked)
        outputs = model.generate(sentences_masked,
                                 do_sample=False,
                                 num_beams=num_beams,
                                 num_return_sequences=num_beams,
                                 max_new_tokens = 10)
        # outputs: bs*num_beams, max_len
        outputs = outputs.view(len(sentences_masked), num_beams, -1)
        # bs, num_beams, max_len
        outputs = outputs.data.cpu().numpy()
        # out_set = set([])
        assert len(outputs)==len(sentences_tmp)
        for batch_idx in range(len(sentences_tmp)):  # for each sent
            prediction_emb = torch.zeros(model.Output_layer.size()[-1]).to(model.device)
            out = []
            out_ids = []
            outputs_batch = outputs[batch_idx] #n_beam
            assert len(outputs_batch)==num_beams
            for idx in range(num_beams):  # for each beam
                out_k = outputs_batch[idx]
                flag = False
                # PAD <mask_1> A B C <mask_2>
                if len(out_k) > 3 and out_k[0] == tokenizer.pad_token_id \
                        and out_k[1] == tokenizer.mask1_token_id:
                    word_idx = 2
                    all_words = []
                    while word_idx < len(out_k):
                        word_id = out_k[word_idx]
                        word_idx += 1
                        if word_id == tokenizer.mask2_token_id: #EOS
                            flag = True
                            break
                        elif (word_id in all_special_ids):  # skip special tokens
                            continue
                        elif (word_id in punctuation_ids):  # skip punct
                            continue
                        # Or, Remove candidates containing them? (e.g. all_words =[]; break)
                        all_words.append(word_id)
                    if flag and len(all_words):
                        out_str = tokenizer.convert_ids_to_tokens(all_words)
                        out_str = "".join(out_str).replace("▁", " ").lstrip(' ').rstrip(' ')
                        if len(out_str.split()) <= max_num_word:
                        # if len(all_words) <= max_num_word:
                            # if len(all_words) ==1 and tokenizer.convert_ids_to_tokens(all_words)[0].replace("▁", "").lower() in stop_words:
                            #     continue #omit stop words
                            prediction_emb += model.Output_layer[all_words].sum(0)
                            # if out_str not in out_set:
                            out_ids.append(tuple(all_words))
                            out.append(out_str)  # unique candidates within beams
                            # out_set.add(out_str)
            all_candidates.append(out)
            all_candidates_ids.append(out_ids)  # out_all_batch: bs, N_cand
            prediction_emb_List.append(F.normalize(prediction_emb, dim=-1))
    return all_candidates_ids,all_candidates, prediction_emb_List,sentences_maskedList


def BERT_Prob(model, MWE_ffn, discard_indices, pred_subword):
    #MWE_ffn: bs, n_mask, dim
    MWE_ffn = MWE_ffn.mean(0).unsqueeze(0)    #bs -> 1
    prediction_scores = model.Output_layer.mm(
        MWE_ffn.view(-1, model.Output_layer.size()[-1]).T)  # V, bs*n_mask
    prediction_scores = prediction_scores + model.Output_layer_bias.expand_as(prediction_scores)  # V, bs*N_mask
    prediction_scores = prediction_scores.view(len(model.Output_layer), -1, MWE_ffn.size()[1])  # V, bs, n_mask
    orig_masking = torch.zeros(prediction_scores.shape).to('cuda')
    orig_masking[discard_indices] = float("-inf")
    if prediction_scores.size()[-1] == 2 and pred_subword:
        orig_masking[WORD_INDICES,:,1] = float("-inf") #generate subwords only
    probability = F.softmax((prediction_scores + orig_masking), dim=0)  # V, bs, n_mask
    probability = probability.mean(1)  # V, n_mask (mean over batch)
    return torch.log(probability)

def BERT_decode_2masks(model, tokenizer,  LogProb_mean_MASK, MASK_idx, num_beams, sentences_masked,
                       mask_col_idx, max_tokens):
    # cand_log_prob_given_Nmasks: scalar
    # LogProb_mean/LogProb_mean_NoMASK: V, n_mask
    #num_beams[0]: see the num_beams[0] words given 2 masks
    #num_beams[1]: see the num_beams[1] words given 1 mask and cand
    top_idx = LogProb_mean_MASK.topk(num_beams[0], dim=0)  # num_beams, n_mask
    cand_ids2score = {}
    assert len(MASK_idx)==2
    cand_phrase_ids = [None, None]
    for j in MASK_idx:  # for each in remaining MASKs
        Log_prob_given_2masks_List = top_idx[0][:, j].data.cpu().numpy()
        candidateList = top_idx[1][:, j].data.cpu().numpy()
        last_idx = MASK_idx[MASK_idx != j]
        assert len(last_idx) == 1
        last_idx = last_idx[0]
        for Log_prob_given_2masks, cand in zip(Log_prob_given_2masks_List, candidateList):
            cand_phrase_ids_tmp = cand_phrase_ids.copy()
            assert cand_phrase_ids_tmp[j] == None
            assert cand_phrase_ids_tmp[last_idx] == None
            cand_phrase_ids_tmp[j] = cand
            sentences_masked_copy = []
            for sid, sent in enumerate(sentences_masked):
                sent_copy = sent.copy()
                assert sent_copy[mask_col_idx[sid, j]] == tokenizer.mask_token_id, sent_copy
                assert sent_copy[mask_col_idx[sid, last_idx]] == tokenizer.mask_token_id, sent_copy
                sent_copy[mask_col_idx[sid, j]] = cand  # fill mask with the prediction
                sentences_masked_copy.append(sent_copy)

            _, MWE_ffn, _, _ ,_= Encode_BERT_PAD(
                tokenizer, model, sentences_masked_copy, mask_col_idx,
                max_tokens=max_tokens, layers=[0])
            # print(LogProb_mean_1mask)
            #MWE_ffn: n_mask, counter.most_freq
            LogProb_mean_tmp = BERT_Prob(model, MWE_ffn, PUNCTUATION_IDS, opt.pred_subword)  # V, n_mask
            LogProb_mean_topk = LogProb_mean_tmp.topk(num_beams[1], dim=0)[1]  # k, n_mask
            for idx in range(len(LogProb_mean_topk)):
                pred_idx = LogProb_mean_topk[idx, last_idx].item()
                log_prob_given_1mask = LogProb_mean_tmp[pred_idx, last_idx].item()
                cand_phrase_ids_out = cand_phrase_ids_tmp.copy()
                assert cand_phrase_ids_out[last_idx] == None
                cand_phrase_ids_out[last_idx] = pred_idx
                assert None not in cand_phrase_ids_out
                cand_phrase_ids_out = tuple(cand_phrase_ids_out)
                score = log_prob_given_1mask + Log_prob_given_2masks  # Multiply prob
                # P(MASK1=x|c, MASK1, MASK2) * P(MASK2 = y|c, x, MASK2)
                if cand_phrase_ids_out in cand_ids2score:
                    cand_ids2score[cand_phrase_ids_out] = max(cand_ids2score[cand_phrase_ids_out], score)
                else:
                    cand_ids2score[cand_phrase_ids_out] = score
    return cand_ids2score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model',
        help='model')
    parser.add_argument(
        '-clustered_sents',
        required=True)
    parser.add_argument(
        '-n_mask',
        type=int,
        default=1)
    parser.add_argument(
        '-window',
        type=int,
        default=0)
    parser.add_argument(
        '-save')
    parser.add_argument(
        '-folder')
    parser.add_argument(
        '-start_from',
        type= int,
        default=-1)
    parser.add_argument(
        '-multiply_prob',
        action='store_true')
    parser.add_argument(
        '-pre_ave',
        action='store_true')
    parser.add_argument(
        '-normalise_prob',
        action='store_true')
    parser.add_argument(
        '-pred_subword',
        action='store_true')
    parser.add_argument(
        '-end_at',
        type= int,
        default=-1)
    parser.add_argument(
        '-max_num_word',
        type= int,
        default=100)
    parser.add_argument(
        '-num_beams',
        type= int,
        default=-1)
    parser.add_argument(
        '-skip_words',
        type=str,
        default=None)
    parser.add_argument(
        '-max_tokens',
        type=int,
        default=8192)
    opt = parser.parse_args()
    folder = None
    model_path = opt.model
    folder = opt.folder
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        print("Directory ", folder, " already exists")
    model, tokenizer = Get_model(model_path, torch.cuda.is_available())
    assert opt.n_mask >= 1
    stop_words_ids = []
    target_sentences = []
    embeddings = []
    punctuations = list(string.punctuation + "“" + "”" + "-" + "’" + "‘" + "…")
    punctuations = punctuations + ["▁" + x for x in punctuations]
    punctuations = punctuations + ["Ġ" + x for x in punctuations]
    # stop_words_ids = set()
    punctuation_ids = set([])
    for w in punctuations:
        if w in tokenizer.vocab:
            w_id = tokenizer.convert_tokens_to_ids([w])[0]
            punctuation_ids.add(w_id)
        else:
            print(w)
    PUNCTUATION_IDS = list(punctuation_ids)
    if model.model_name in ["bert", 'jbert','spanbert']:
        non_alnum_vocab = []
        BERT_tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer.vocab))))
        WORD_INDICES =[]
        for i, w in enumerate(BERT_tokens):
            if not w.startswith("##"):
                WORD_INDICES.append(i)
            if not w.isalnum() and model.model_name != 'jbert':
                non_alnum_vocab.append(i)
    elif model.model_name in ["roberta","deberta"]:
        non_alnum_vocab = []
        alnum_vocab = []
        BERT_tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer.vocab))))
        WORD_INDICES =[]
        for i, w in enumerate(BERT_tokens):
            if w.startswith("Ġ"):
                WORD_INDICES.append(i)
            else:
                non_alnum_vocab.append(i)
    elif model.model_name in ['t5','mt5']:
        all_special_ids = set(tokenizer.all_special_ids)
    n_mask = opt.n_mask
    assert n_mask in [1,2]
    print(str(n_mask)+"-MASKED")
    vocab = dict()
    count = -1

    with open(opt.clustered_sents, 'rb') as f:
        phrase2sent = pickle.load(f)

    with open(opt.folder + "/options.pkl", 'wb') as f:
        pickle.dump(opt, f)  # N_idiom, N_sample, s_len, dim

    v = list(phrase2sent.keys())
    V_size = len(v)
    if opt.end_at == -1:
        opt.end_at = float("inf")
    start = time.time()
    N_not_found = 0
    MWE2cand_loss = {}
    MWE2cand_prob = {}
    with torch.no_grad():
        for i in tqdm(range(len(v))):
            phrase = v[i]
            # print(phrase)
            assert phrase[0] != " "
            # phrase = " " + phrase #Add space
            count += 1
            if opt.start_from > count:
                continue
            if count >= opt.end_at:
                break
            # if count%1000 ==0:
                # print(count)
            sentences_raw_all = phrase2sent[phrase]
            phrase = " " + phrase  # add space
            phrase_tokenised_ids = tokenise_phrase(model, tokenizer, phrase)
            mask_ids = phrase_tokenised_ids
            mask_ids_replace = [tokenizer.mask_token_id for _ in range(n_mask)]
            #sentences_raw_all: #K, N_sent
            assert isinstance(sentences_raw_all, list)
            if 'dbscan' in opt.clustered_sents and len(sentences_raw_all)>1:
                sentences_raw_all = sentences_raw_all[1:] #remove the noise cluster
            K = len(sentences_raw_all)
            # sent_class = MWE2sentclass[phrase]
            # K = len(set(sent_class))
            for k_class in range(K): #for each cluster
                all_candidates = []
                sentences_masked_all = []
                all_candidates_ids = []
                # sentences_raw = [x for idx, x in enumerate(sentences_raw_all) if sent_class[idx] == k_class]
                sentences_raw = sentences_raw_all[k_class]
                if model.model_name in ['bert','spanbert','xlnet','roberta','albert']:
                    # NOTE: Attention mask for PADDING?
                    sentences_tmp = tokenizer(sentences_raw)["input_ids"]
                    # sentences_tmp = sentences_raw
                    sentences = []
                    for x in sentences_tmp:
                        assert len(x) < 512
                        sentences.append(x)
                    sentences_masked, mask_row_idx, mask_col_idx, \
                    around_idx_list, gold_ids_list, padding_list = \
                        Identify_Key_Indices(sentences, mask_ids, mask_ids_replace, opt.window, tokenizer,
                                             stop_words_ids, "span_two", None)

                    # print("N_sent: " + str(len(sentences_masked)))
                    for x in range(len(sentences_masked[:3])):
                        gold_sent = " ".join(tokenizer.convert_ids_to_tokens(sentences_masked[x]))
                        # print(gold_sent)

                    _, MWE_ffn, _, _,_ = Encode_BERT_PAD(
                        tokenizer, model, sentences_masked, mask_col_idx,
                        max_tokens= opt.max_tokens, layers=[0],  around_idx_list=around_idx_list, padding_list=padding_list)
                    #MWE_ffn: bs, n_mask, dim
                    LogProb_mean = BERT_Prob(model, MWE_ffn, PUNCTUATION_IDS, opt.pred_subword)
                    #LogProb_mean: V, n_mask
                    if phrase.lstrip().rstrip() in MWE2cand_prob:
                        MWE2cand_prob[phrase.lstrip().rstrip()].append(LogProb_mean.data.cpu().numpy())
                    else:
                        MWE2cand_prob[phrase.lstrip().rstrip()] = [LogProb_mean.data.cpu().numpy()]
                    #topword_ids:
                    if n_mask == 1: #single mask
                        assert LogProb_mean.shape == (len(LogProb_mean),1)
                        top_idx = LogProb_mean.view(-1).topk(3*opt.num_beams, dim=0)  # 50
                        scores = top_idx[0].data.cpu().numpy()
                        widList = top_idx[1].data.cpu().numpy()
                        id_word_score = []
                        for j in range(3*opt.num_beams):
                            wid = widList[j]
                            word = tokenizer.convert_ids_to_tokens([wid])
                            assert len(word)==1
                            word = word[0]
                            if word != tokenizer.unk_token:
                                if model.model_name in ['xlnet','deberta-v2','albert']:
                                    word = word.replace("▁", " ").lstrip(' ').rstrip(' ')
                                elif model.model_name in ['roberta','deberta']:
                                    word = word.replace("Ġ", " ").lstrip(' ').rstrip(' ')
                                # if Levenshtein.distance(word.lower(), phrase.lower().lstrip(" ").rstrip(" "))  > opt.lev:
                                if word.lower() not in [phrase.lstrip(" ").rstrip(" ").lower(), "".join(phrase.split()).lower()]:
                                    id_word_score.append([tuple([wid]), word, scores[j]])
                            if len(id_word_score)==opt.num_beams:
                                break
                        assert len(id_word_score)==opt.num_beams
                        if phrase.lstrip().rstrip() in MWE2cand_loss:
                            MWE2cand_loss[phrase.lstrip().rstrip()].append(id_word_score)
                        else:
                            MWE2cand_loss[phrase.lstrip().rstrip()] = [id_word_score]
                        # print(MWE2cand_loss[phrase.lstrip().rstrip()][-1])
                    else: #multilple masks
                        assert n_mask == 2
                        top_idx = LogProb_mean.topk(opt.num_beams, dim=0)  # 50, n_mask
                        pred_words = np.array(
                            tokenizer.convert_ids_to_tokens(
                                top_idx[1].data.cpu().numpy().reshape(-1))).reshape(opt.num_beams, n_mask)
                        cand_ids2score =  BERT_decode_2masks(model, tokenizer,
                                           LogProb_mean, np.array([0,1]), [opt.num_beams, opt.num_beams], sentences_masked,
                                           mask_col_idx, opt.max_tokens)

                        id_word_score = []
                        for wids, value in sorted(cand_ids2score.items(), key=lambda item: -1 * item[1]):
                            if model.model_name in ['bert', 'spanbert']:
                                pred_phrase = " ".join(tokenizer.convert_ids_to_tokens(wids)).replace(" ##", "")
                            elif model.model_name in ['xlnet','deberta-v2','albert']:
                                pred_phrase = "".join(tokenizer.convert_ids_to_tokens(wids)).replace("▁", " ").lstrip(' ').rstrip(' ')
                            elif model.model_name in ['roberta','deberta']:
                                pred_phrase = "".join(tokenizer.convert_ids_to_tokens(wids)).replace("Ġ", " ").lstrip(' ').rstrip(' ')
                            if pred_phrase.lower() not in [phrase.lstrip(" ").rstrip(" ").lower(), "".join(phrase.split()).lower()]:
                                id_word_score.append([tuple(wids), pred_phrase, value])
                        assert len(id_word_score)>=2*opt.num_beams

                        if phrase.lstrip().rstrip() in MWE2cand_loss:
                            MWE2cand_loss[phrase.lstrip().rstrip()].append(id_word_score)
                        else:
                            MWE2cand_loss[phrase.lstrip().rstrip()] = [id_word_score]
                elif model.model_name in ['t5']: #T5 generation
                    all_special_ids = set(tokenizer.all_special_ids)
                    max_num_word = 2
                    num_beams = opt.num_beams
                    all_candidates_ids, all_candidates_str, prediction_emb_List,sentences_maskedList=\
                        T5_generate_paraprhase(sentences_raw, model,tokenizer, mask_ids, all_special_ids, punctuation_ids, num_beams, max_num_word)
                    # all_candidates_ids: N_sen, N_cand
                    # all_candidates: N_sent, N_cand
                    assert len(all_candidates_ids) == len(sentences_raw)
                    wstr2id = {}
                    str2count = {}
                    for sid in range(len(sentences_raw)):
                        for wids in all_candidates_ids[sid]:
                            out_str = tokenizer.convert_ids_to_tokens(wids)
                            out_str = "".join(out_str).replace("▁", " ").lstrip(' ').rstrip(' ')
                            if len(out_str)==0:
                                continue
                            if out_str not in wstr2id:
                                wstr2id[out_str] = wids
                                str2count[out_str] = 1
                            else:
                                str2count[out_str] += 1
                                if len(wids) < len(wstr2id[out_str]):
                                    wstr2id[out_str] = wids
                    id_word_score = []
                    for wstr, value in sorted(str2count.items(), key=lambda item: -1 * item[1]):
                        if wstr.lower() not in [phrase.lstrip(" ").rstrip(" ").lower(),
                                               "".join(phrase.split()).lower()]:
                            id_word_score.append([tuple(wstr2id[wstr]), wstr, value])
                    if phrase.lstrip().rstrip() in MWE2cand_loss:
                        MWE2cand_loss[phrase.lstrip().rstrip()].append(id_word_score)
                    else:
                        MWE2cand_loss[phrase.lstrip().rstrip()] = [id_word_score]

    elapsed_time = time.time() - start
    print("Train elapsed_time: "+str(elapsed_time))
    with open(opt.folder + "/"+str(opt.n_mask) + "MASKs_candidates2inner_score.pkl", 'wb') as f:
        pickle.dump(MWE2cand_loss, f)  # N_idiom, N_sample, s_len, dim

