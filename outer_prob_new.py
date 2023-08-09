import time
import torch
import os
import string
import numpy as np
import pickle
import torch.nn.functional  as F
import argparse
import sys
from utils import Identify_Key_Indices, tokenise_phrase, Get_Attention,Get_model, Encode_BERT_PAD
from tqdm import tqdm

def Get_Maskids(model,tokenizer, sentences_masked, MWE_col_idx, context_maskid_window, stop_word_ids, mask_opt, window,  attentions= None, MWE_col_idx_noMASK= None):
    context_maskidList = []
    if model.model_name in ['t5','mt5']:
        assert mask_opt.startswith("span") or mask_opt.startswith("Rspan")
    if mask_opt.startswith("rand"):
        mask_opt = mask_opt.split("_")
        assert len(mask_opt) == 3
        assert mask_opt[1].startswith("Nmask")
        assert mask_opt[2].startswith("Nsplit")
        mask_size = int(mask_opt[1][5:])
        N_split = int(mask_opt[2][6:])
        assert N_split in [-1, 1, 2]
        # weightList = []
        for sid, sent in enumerate(sentences_masked):  # for each sent
            context_maskid = []
            rand_idx = np.random.permutation(len(sent))
            if mask_size == -1:
                mask_size_tmp = max(len(sent) * 12 // 100, 1)
            else:
                mask_size_tmp = mask_size
            for idx in rand_idx:
                if idx not in MWE_col_idx[sid] and sent[idx] not in stop_word_ids:
                    context_maskid.append(idx)
                if len(context_maskid) == mask_size_tmp:
                    break
            context_maskid = np.array(context_maskid)
            # if N_sample == 1:
            if len(context_maskid) % 2 == 1:
                context_maskid = np.append(context_maskid, [-100])
            if N_split == -1:
                context_maskid = context_maskid.reshape(-1, 2)
            else:
                context_maskid = context_maskid.reshape(N_split, -1)

            context_maskidList.append(context_maskid)

    elif mask_opt.startswith("Rspan"):
        mask_opt = mask_opt.split("_")
        assert len(mask_opt) == 3
        assert mask_opt[1].startswith("Nmask")
        assert mask_opt[2].startswith("Nsplit")
        mask_size = int(mask_opt[1][5:])
        N_split = int(mask_opt[2][6:])
        assert N_split in [-1, 1, 2]
        assert window == 0
        assert mask_size == 5
        for sid, sent in enumerate(sentences_masked):  # for each sent
            context_maskid = []
            rand_idx = np.random.permutation(list(range(3,len(sent)-3)))
            for idx in rand_idx:
                stop_count = 0
                span = [idx - 2, idx - 1, idx, idx + 1, idx + 2]
                for k_ in span:
                    if k_ in MWE_col_idx[sid] or sent[k_] in tokenizer.all_special_ids:
                        stop_count = 100
                    elif sent[k_] in stop_word_ids:
                        stop_count += 1
                if stop_count >=2:
                    continue
                else:
                    context_maskid = span
                    break
            context_maskid = np.array(context_maskid)
            # if N_sample == 1:
            if len(context_maskid) % 2 == 1:
                context_maskid = np.append(context_maskid, [-100])
            if N_split == -1:
                context_maskid = context_maskid.reshape(-1, 2)
            else:
                context_maskid = context_maskid.reshape(N_split, -1)

            context_maskidList.append(context_maskid)

    elif mask_opt.startswith("attn"):
        ### attn_Nmask-1_Nsplit1_attnL4_norm1
        mask_opt = mask_opt.split("_")
        assert len(mask_opt) == 5
        assert mask_opt[1].startswith("Nmask")
        assert mask_opt[2].startswith("Nsplit")
        assert mask_opt[3].startswith("attnL")
        assert mask_opt[4].startswith("norm")
        mask_size = int(mask_opt[1][5:])
        N_split = int(mask_opt[2][6:])
        assert N_split in [-1, 1, 2]
        # weightList = []
        for sid, sent in enumerate(sentences_masked):  # for each sent
            if mask_size < 0:
                mask_size_tmp = -1 * mask_size * len(sent) * 12 // 100
            else:
                mask_size_tmp = mask_size
            attn_weight = attentions[sid]  # N_mask, N_words
            if len(attn_weight) != len(sent) and attn_weight[len(sent)] > 1e-40:
                raise Exception
            # Assert the weight at last has a non-zero value (not padding)
            assert attn_weight[len(sent) - 1] != 0, attn_weight[len(sent) - 1]
            attn_weight = attn_weight[:len(sent)]
            attn_weight[0] = -0.00001  # CLS
            attn_weight[-1] = -0.00001  # SEP
            # weightList.append(attn_weight)
            important_indices = np.argsort(-1 * attn_weight)
            context_maskid = []
            for idx in important_indices:
                if idx not in MWE_col_idx[sid] and sent[idx] not in stop_word_ids:
                    context_maskid.append(idx)
                if len(context_maskid) == mask_size_tmp:
                    break
            context_maskid = np.array(context_maskid)
            if len(context_maskid) % 2 == 1:
                context_maskid = np.append(context_maskid, [-100])
            if N_split == -1:
                ###2masks at once inference###
                context_maskid = context_maskid.reshape(-1, 2)
            else:
                context_maskid = context_maskid.reshape(N_split, -1)
            context_maskidList.append(context_maskid)

    return context_maskidList

def Calc_OuterProb(tokenizer, model, sent, MWE_col_idx, widList,
                   prepost_mask_idx, max_tokens):
    # sentences_masked: N_words (1 sentence)
    # MWE_col_idx: N_indices (idx for MWE)
    # prepost_mask_idx: N_SPAN, N_mask
    # prepost_goldidx: N_SPAN, N_MASK
    # padding_idx: N_SPAN, N_MASK
    sent_tmp = sent.copy()
    assert tokenizer.mask_token_id in sent_tmp
    sent_paraphrased = np.repeat([sent_tmp], len(widList), axis=0)  # N_cand, seq_len
    assert len(MWE_col_idx) ==1 or all(np.diff(MWE_col_idx) ==1) #continous MWE
    sent_paraphrased[:, MWE_col_idx] = widList
    # sent_tmp: bs, sent_tmp
    if [tokenizer.mask_token_id for _ in range(len(MWE_col_idx))] not in widList:
        assert tokenizer.mask_token_id not in sent_paraphrased
    logprob_all = []
    # two spans:
    bs = len(sent_paraphrased)
    # print(attn_weight)
    all_masked_idx = []
    for ith_span, mask_col_idx in enumerate(prepost_mask_idx): #for each span (Nspan, n_mask)
        #mask_col_idx: mask_idx
        if len(mask_col_idx):
            assert mask_col_idx[-1] < sent_paraphrased.shape[1]
            assert len(set(MWE_col_idx).intersection(mask_col_idx)) == 0
            # print(mask_col_idx)
            sent_paraphrased_new = sent_paraphrased.copy()
            padding_idx = [idx != -100 for idx in mask_col_idx]
            mask_col_idx = mask_col_idx[padding_idx] #remove padding
            if len(mask_col_idx):
                goldidx_tmp = []
                for k, idx in enumerate(mask_col_idx):
                    assert idx >= 0
                    goldidx_tmp.append(sent_tmp[idx])
                    assert sent_tmp[idx] not in (set(tokenizer.all_special_ids)-set([tokenizer.unk_token_id]))
                all_masked_idx.extend(mask_col_idx)
                if model.model_name in ['bert', 'spanbert','electra','albert']:
                    for idx in mask_col_idx:
                        if idx not in MWE_col_idx:
                            sent_paraphrased_new[:,idx] = tokenizer.mask_token_id
                    # print(sent_paraphrased_new[0])
                    mask_col_idx_repeat = np.repeat([mask_col_idx], bs, axis=0) #same index for all sents
                    _, MWE_ffn, _, _, _ = Encode_BERT_PAD(
                            tokenizer, model, sent_paraphrased_new.tolist(), mask_col_idx_repeat,
                            max_tokens = max_tokens, layers=[0])
                    # attentions: #N_layer, bs, n_mask, N_words
                    prediction_scores = model.Output_layer.mm(
                        MWE_ffn.view(-1, model.Output_layer.size()[-1]).T)  # V, bs*n_mask
                    prediction_scores = prediction_scores + model.Output_layer_bias.expand_as(
                        prediction_scores)  # V, bs*N_mask
                    prediction_scores = prediction_scores.view(len(model.Output_layer), -1,
                                                               MWE_ffn.size()[1])  # V, bs, n_mask
                    probability = F.softmax((prediction_scores), dim=0)  # V, bs, n_mask
                    probability = probability.data.cpu().numpy()

                    # LogProb_mean: V, n_mask (mean over batch)
                    # if SBO:
                    #     pre = mask_col_idx[:, 0] - 1 #bs,
                    #     post = mask_col_idx[:, -1] + 1
                    #     pairs_idx = np.stack([pre, post], axis=-1)
                    #     # _, _, prediction_scores = SpanBERT_SBO(model, last_layer, pairs_idx, N_mask, remove_bias, ith_span)
                    pred_prob = [probability[goldidx_tmp[k], :, k] for k in range(len(goldidx_tmp))]
                    # pred_prob: N_Mask, N_cand
                    logprob_all.append(np.log(pred_prob))  # N_span, N_Mask, N_cand
                elif model.model_name in ['t5', 'mt5']:
                    assert all(np.diff(mask_col_idx) == 1), str(mask_col_idx) + " is not span-masking"
                    sent_paraphrased_new[:, mask_col_idx] = tokenizer.mask1_token_id
                    # remove duplicated mask tokens
                    # print("before")
                    # print(sent_paraphrased_new)
                    # span masking
                    sent_paraphrased_new = np.concatenate(
                        [sent_paraphrased_new[:, :np.min(mask_col_idx)], sent_paraphrased_new[:, np.max(mask_col_idx):]], axis=1)
                    assert np.sum(sent_paraphrased_new == tokenizer.mask1_token_id) == len(sent_paraphrased_new)
                    labels = [tokenizer.mask1_token_id] + goldidx_tmp + [tokenizer.mask2_token_id]
                    labels = torch.LongTensor([labels]).to(model.device)  # 1, N_words
                    # print(tokenizer.convert_ids_to_tokens(labels[0]))
                    labels = labels.repeat(len(sent_paraphrased_new), 1)  # bs, N_words
                    # the forward function automatically creates the correct decoder_input_ids
                    logits = model(input_ids=torch.LongTensor(sent_paraphrased_new).to(model.device),
                                   labels=labels).logits
                    logits = F.softmax(logits, dim=-1).view(-1, logits.size()[-1])  # (bs* N_words + 2), V
                    # CROSS-ENTROPY
                    prob = logits[np.arange(len(logits)), labels.view(-1)]
                    prob = prob.view(len(labels), -1)  # bs, N_MWE + 2
                    # print(prob)
                    # loss = F.cross_entropy(logits.view(-1, logits.size()[-1]), labels_tmp.view(-1), reduction='none')
                    # loss = loss.view(len(labels_tmp), -1) #bs_mini, N_MWE + 2
                    assert prob.shape == labels.shape
                    # mean loss for the candidate words + loss for <extra_id_1> (loss for <extra_id_0> is the same for all sentences_tok)
                    eps = 0.00001
                    prob = np.log(prob.data.cpu().numpy()+eps)  # bs, N_MWE + 2
                    # prob_EOS = prob[:, -1].reshape(1,-1) #omit EOS (1, bs)
                    prob = prob[:, 1:] # omit BOS: # bs, N_MWE+1
                    # assert len(prob[0]) <= len(target_idx) + 1
                    # prob = prob[:,target_idx] # remove stop words
                    # logprob_all.append(prob_EOS)  # N_span, 1, N_cand
                    logprob_all.append(prob.T)  # N_span, N_MWE+1, N_cand
    if len(logprob_all):
        logprob_all = np.concatenate(logprob_all, axis=0)  # N_Mask_all, N_cand
        return logprob_all
    else:
        return None  # N_cand *(average logprob over N_MASK_all)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model',
        help='model')
    parser.add_argument(
        '-candidates',
        required=True,
        nargs='+')
    parser.add_argument(
        '-clustered_sents',
        required=True)
    parser.add_argument(
        '-mask_opt',
        required=True)
    parser.add_argument(
        '-n_mask',
        type=int,
        default=1,
        help='N_mask')
    parser.add_argument(
        '-sent_class',
        help='N_mask')
    parser.add_argument(
        '-window',
        type=int,
        default=0)
    parser.add_argument(
        '-save')
    parser.add_argument(
        '-word_list',
        default=None)
    parser.add_argument(
        '-folder')
    # parser.add_argument(
    #     '-use_stopwords',
    #     action='store_true',
    #     help='save_name')
    parser.add_argument(
        '-prob_filter',
        action='store_true')
    parser.add_argument(
        '-swords',
        action='store_true')
    parser.add_argument(
        '-component_pred',
        action='store_true')
    parser.add_argument(
        '-mean_prob',
        action='store_true')
    parser.add_argument(
        '-multiply_prob',
        action='store_true')
    parser.add_argument(
        '-normalise_prob',
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
        '-lowercase',
        action='store_true')
    parser.add_argument(
        '-t5',
        action='store_true')
    parser.add_argument(
        '-tokenize',
        action='store_true')
    parser.add_argument(
        '-prob_type')
    parser.add_argument(
        '-max_tokens',
        type=int,
        default=8192)
    opt = parser.parse_args()
    folder = None
    print(opt.mask_opt.split("_"))
    model_path = opt.model
    folder = opt.folder
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        print("Directory ", folder, " already exists")
    model, tokenizer = Get_model(model_path, torch.cuda.is_available())
    if model.model_name in ["t5", "mt5"]:
        assert opt.n_mask == 1
        assert opt.mask_opt.startswith('span') or opt.mask_opt.startswith('Rspan')
    else:
        assert opt.n_mask >= 1
    input_ids = torch.cuda.LongTensor([[1, 2]])
    if model.model_name == "spanbert":
        N_layer = 25
    elif model.model_name in ['deberta-v3', "bert", 'electra', 'sbert-bert','albert']:
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        N_layer = len(outputs["hidden_states"])  # 13 or 25
    elif model.model_name in ['xlnet']:
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        N_layer = len(outputs["hidden_states"])*2  # 13 or 25
    target_sentences = []
    embeddings = []
    punctuations = list(string.punctuation + "“" + "”" + "-" + "’" + "‘" + "…")
    punctuations = punctuations + ["▁" + x for x in punctuations]
    punctuations = punctuations + ["Ġ" + x for x in punctuations]
    punctuation_ids = set([])
    for w in punctuations:
        if w in tokenizer.vocab:
            w_id = tokenizer.convert_tokens_to_ids([w])[0]
            punctuation_ids.add(w_id)
        # else:
        #     print(w)
    punctuation_ids.update(tokenizer.all_special_ids)
    stop_word_ids = list(punctuation_ids)
    if model.model_name in ["bert",'spanbert','electra']:
        non_alnum_vocab = []
        BERT_tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer.vocab))))
        for i, w in enumerate(BERT_tokens):
            if not w.isalnum():
                non_alnum_vocab.append(i)
        stop_word_ids = stop_word_ids + non_alnum_vocab #avoid subwords

    elif model.model_name in ["albert"]:
        non_alnum_vocab = []
        BERT_tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer.vocab))))
        for i, w in enumerate(BERT_tokens):
            if not w.startswith("▁"):
                non_alnum_vocab.append(i)
        stop_word_ids = stop_word_ids + non_alnum_vocab #avoid subwords

    elif model.model_name in ["roberta"]:
        non_word_vocab = []
        BERT_tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer.vocab))))
        for i, w in enumerate(BERT_tokens):
            if not w.startswith("Ġ"):
                non_word_vocab.append(w)
        stop_word_ids = stop_word_ids + non_word_vocab #avoid subwords

    n_mask = opt.n_mask
    print(str(n_mask)+"-MASKED")
    vocab = dict()
    count = -1

    with open(opt.clustered_sents, 'rb') as f:
        phrase2sent_1B = pickle.load(f)

    candidatesList = []
    for i in range(len(opt.candidates)):
        with open(opt.candidates[i], 'rb') as f:
            candidates = pickle.load(f)
            MWEList = set(candidates.keys())
            candidatesList.append(candidates)

    MWEList = list(MWEList)
    if opt.lowercase:
        MWEList_lower = []
        for i in range(len(MWEList)):
            if MWEList[i]==MWEList[i].lower():
                MWEList_lower.append(MWEList[i])
        MWEList = MWEList_lower
    MWEList = sorted(MWEList)
    V_size = len(MWEList)
    print(str(V_size) + "words")
    start = time.time()
    N_not_found = 0
    MWE2cand_score = {}
    with torch.no_grad():
        for i in tqdm(range(len(MWEList))):
            phrase = MWEList[i]
            print(phrase)
            assert phrase[0] != " "
            sentences_raw_all = phrase2sent_1B[phrase] #K, N_sent
            assert isinstance(sentences_raw_all, list)
            if 'dbscan' in opt.clustered_sents and len(sentences_raw_all) > 1:
                sentences_raw_all = sentences_raw_all[1:]  # remove the noise cluster
            K = len(sentences_raw_all)
            phrase = " " + phrase  # add space
            if model.model_name == 'spanbert':
                phrase_tokenised_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(phrase.lstrip(" ")))
            else:
                phrase_tokenised_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]  # remove CLS/SEP
            # mask_ids_replace = [tokenizer.mask_token_id for _ in range(n_mask)]
            for k_class in range(K): #for each cluster
                all_candidates = []
                sentences_masked_all = []
                all_candidates_ids = []
                sentences_raw = sentences_raw_all[k_class]
                candList = []
                if opt.t5:
                    N_cand_max = 20
                else:
                    N_cand_max = 10
                for candidates in candidatesList:
                    if opt.t5:
                        cand_tmp = candidates[phrase.lstrip().rstrip()][k_class]
                        cand_tmp_new = []
                        value_list = []
                        for elem in cand_tmp:  # para candidates
                            assert len(elem)==3
                            # [tuple(wstr2id[wstr]), wstr, value]
                            # wid = elem[0]  # word id
                            cand_str = elem[1]  # string
                            val = elem[2]
                            assert len(cand_str.split()) <= 2
                            if len(cand_str.split()) == 2:
                                val = val*2
                            value_list.append(-1*val)
                            if len(cand_str.split()) <= 2:
                                cand_tmp_new.append(elem)
                        idx = np.argsort(value_list)[:N_cand_max]
                        candList = [cand_tmp_new[k] for k in idx]
                    else:
                        candList = candList + candidates[phrase.lstrip().rstrip()][k_class][:N_cand_max]

                len2words = {}
                for elem in candList:  # para candidates
                    cand_str = elem[1]  #string
                    if opt.tokenize:
                        cand_ids = tokenise_phrase(model, tokenizer, " " + cand_str)
                    else:
                        cand_ids = list(elem[0])  # tuple->list
                    if len(cand_ids) == 0:
                        continue
                    inner_score = elem[2]  #string
                    if len(cand_ids) not in len2words:
                        len2words[len(cand_ids)] = [(cand_ids, cand_str, inner_score)]
                    else:
                        len2words[len(cand_ids)].append((cand_ids, cand_str, inner_score))

                sentences_tmp = tokenizer(sentences_raw)["input_ids"]
                sentences_tok = []
                for x in sentences_tmp:
                    assert len(x) < 512
                    sentences_tok.append(x)
                outer_scoreList = []
                attentions = []
                cossim = 0
                if opt.mask_opt.startswith("attn"):
                    N_placeholder = 2 #replace MWEs with two [MASK]s
                    mask_ids_replace = [tokenizer.mask_token_id for _ in range(N_placeholder)]  # one MASK
                    sentences_masked, mask_row_idx, MWE_col_idx, \
                    context_maskid_window, _, _ = \
                        Identify_Key_Indices(sentences_tok, phrase_tokenised_ids, mask_ids_replace, opt.window,
                                             tokenizer,
                                             stop_word_ids,
                                             "span_two", None)
                else:
                    N_placeholder = len(phrase_tokenised_ids)
                    mask_ids_replace = phrase_tokenised_ids
                    sentences_masked, mask_row_idx, MWE_col_idx, \
                    context_maskid_window, _, _ = \
                        Identify_Key_Indices(sentences_tok, phrase_tokenised_ids, mask_ids_replace, opt.window, tokenizer,
                                             stop_word_ids,
                                             "span_two", None)
                attentions = None
                MWE_col_idx_noMASK = None
                if opt.mask_opt.startswith("attn"):
                    mask_opt = opt.mask_opt.split("_")
                    attnL = int(mask_opt[3][5:])
                    norm_attn = bool(int(mask_opt[4][4:]))
                    layers = [N_layer - 2 - xx for xx in range(attnL)]
                    # Get attention weights
                    _, _, attentions = Get_Attention(
                        tokenizer, model, sentences_masked, MWE_col_idx,
                        max_tokens=opt.max_tokens, layers=layers, norm_attn=norm_attn)
                    # attentions: bs, n_mask, max_sent_len
                    attentions = attentions.data.cpu().numpy()  # last four layers
                    attentions = attentions.mean(1)  # bs, max_sent_len

                context_maskidList = Get_Maskids(model,tokenizer, sentences_masked, MWE_col_idx, context_maskid_window, stop_word_ids,
                            opt.mask_opt, opt.window,  attentions, MWE_col_idx_noMASK)

                masked_tokenList_prev = None
                for wlen in sorted(len2words.keys()):
                    # print(wlen)
                    masked_tokenList = []
                    mask_ids_replace = [tokenizer.mask_token_id for _ in range(wlen)]  # placeholder
                    sentences_masked, mask_row_idx, mask_col_idx,\
                    _, _, _ = \
                        Identify_Key_Indices(sentences_tok, phrase_tokenised_ids, mask_ids_replace, 0, tokenizer,
                                             stop_word_ids,
                                             opt.mask_opt, None)
                    candidateList_wlen = len2words[wlen] #[(cand_ids, cand_str)] * N_cand
                    widList = [x[0] for x in candidateList_wlen] #N_cand, N_word
                    wstrList = [x[1] for x in candidateList_wlen] #N_cand, N_word
                    gold_sent = " ".join(tokenizer.convert_ids_to_tokens(sentences_masked[0]))
                    logprob_List = [] #aggregare logprob over all sents
                    for sid, sent in enumerate(sentences_masked): #for each sent
                        sent_tmp = sent.copy() #[1,2,3,[4,5],6,7]/[1 2 3, [4], 5, 6]
                        context_maskid = context_maskidList[sid].copy()
                        prob_weight = None
                        if wlen != N_placeholder: #IMPORTANT: adjust the context mask index
                            context_maskid[context_maskid > mask_col_idx[sid][0]] += (wlen-N_placeholder)
                        if opt.component_pred:
                            assert len(context_maskid)==1
                            for widx, wid in enumerate(sent_tmp):
                                if wid in phrase_tokenised_ids and widx not in mask_col_idx[sid] and widx not in context_maskid[0]:
                                    context_maskid = np.append(context_maskid[0], widx).reshape(1,-1)
                            # comp_idx = [x for x in sent_tmp if x in phrase_tokenised_ids]
                        masked_tokens_tmp = [sent_tmp[i] for i in context_maskid.reshape(-1) if i != -100]
                        masked_tokenList.extend(masked_tokens_tmp)
                        logprob = Calc_OuterProb(tokenizer, model, sent_tmp, mask_col_idx[sid], widList, context_maskid, opt.max_tokens)
                        if logprob is not None:
                            if model.model_name in ['t5', 'mt5']:
                                assert len(masked_tokens_tmp) + 1 == len(logprob), str(
                                    len(masked_tokens_tmp)) + " " + str(len(logprob))
                            else:
                                assert len(masked_tokens_tmp) == len(logprob)
                            if opt.mean_prob:
                                logprob = np.exp(logprob) #loglog -> prob
                            logprob_List.append(logprob)  # sum over sent

                    if masked_tokenList_prev == None:
                        masked_tokenList_prev = masked_tokenList
                    else:
                        # make sure the same words are masked
                        assert masked_tokenList_prev == masked_tokenList

                    logprob_List = np.concatenate(logprob_List, axis=0)  # N_Mask_all, N_cand
                    # print(" ".join(tokenizer.convert_ids_to_tokens(np.array(masked_tokenList)[:10])))
                    logprob_List = logprob_List.mean(axis=0) #N_cand; average probs over all masked words
                    assert len(logprob_List)==len(candidateList_wlen)
                    for k in range(len(candidateList_wlen)):
                        inner_score = candidateList_wlen[k][2]
                        outer_score = logprob_List[k]
                        outer_scoreList.append([candidateList_wlen[k][0],candidateList_wlen[k][1], outer_score, inner_score])
                    #MWE_ffn: bs, n_mask, dim
                outer_scoreList = sorted(outer_scoreList, key=lambda item: -1* float(item[2])) #sort by the outer prob
                if phrase.lstrip().rstrip() in MWE2cand_score:
                    MWE2cand_score[phrase.lstrip().rstrip()].append(outer_scoreList)
                else:
                    MWE2cand_score[phrase.lstrip().rstrip()] = [outer_scoreList]

    elapsed_time = time.time() - start
    print("Train elapsed_time: "+str(elapsed_time))
    with open(opt.folder + "/candidates2outer_score_model_"+ opt.model.split("/")[-1] + "_"+opt.mask_opt+".pkl", 'wb') as f:
        pickle.dump(MWE2cand_score, f)  # N_idiom, N_sample, s_len, dim

