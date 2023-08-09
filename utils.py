import random
from transformers.models.bart.modeling_bart import shift_tokens_right as mono_shift_tokens_right
import sys
import torch
from collections import Counter
import numpy as np
import pickle
import torch.nn.functional  as F
from transformers import DebertaV2ForMaskedLM,DebertaForMaskedLM,BertJapaneseTokenizer, XLNetLMHeadModel, XLNetTokenizer,ElectraTokenizer,ElectraModel,ElectraForPreTraining,MBart50TokenizerFast,MBartForConditionalGeneration,T5Tokenizer, BartTokenizer,AlbertTokenizer, GPT2Tokenizer, OpenAIGPTTokenizer,OpenAIGPTLMHeadModel, GPT2LMHeadModel, AlbertForMaskedLM, BartForConditionalGeneration, XLMWithLMHeadModel, CamembertForMaskedLM, BertForMaskedLM, BertTokenizer, BertModel, AutoTokenizer, AutoModel, RobertaForMaskedLM, T5ForConditionalGeneration, T5EncoderModel, M2M100ForConditionalGeneration,M2M100Tokenizer,MPNetTokenizer,MPNetForMaskedLM, PegasusTokenizer,PegasusForConditionalGeneration


def Identify_Key_Indices(tokenised_sentence_ids, phrase, mask_ids, window, tokenizer, stop_words, mask_opt, MWE_index):
    # tokenised_sentences_list: bs, N_sent
    # tokenised_sentences_list
    # phrase: ids for MWE [12,43,55]
    phrase_len = len(phrase)
    assert phrase_len != 0
    phrase_tmp = phrase.copy()
    idiom_col_idx = []
    out = []
    mask_idx_list = []
    if MWE_index is not None:
        assert len(MWE_index) == len(tokenised_sentence_ids), str(len(MWE_index)) + "|"+str(len(tokenised_sentence_ids))
    else:
        MWE_index = [0 for _ in range(len(tokenised_sentence_ids))]
    phrase_additional = []
    for sid, sent in enumerate(tokenised_sentence_ids):
        flag = False
        sent_tmp = sent.copy()
        MWE_count = 0
        is_bbb = False
        for i in range(len(sent_tmp)-len(phrase_tmp)+1):
            if len(phrase_additional):
                for e in range(len(phrase_additional)):
                    if all([sent_tmp[i + j] == phrase_additional[e][j] for j in range(len(phrase_additional[e]))]):
                        is_bbb = True
                        matched = phrase_additional[e]
                        break
            if is_bbb or all([sent_tmp[i + j] == phrase_tmp[j] for j in range(len(phrase_tmp))]):
                if MWE_count !=  MWE_index[sid]:
                    MWE_count += 1
                else:
                    flag = True
                    idiom_idx = [i + j for j in range(len(mask_ids))]
                    idiom_col_idx.append(idiom_idx)
                    if is_bbb:
                        for j in range(len(matched)):  # remove
                            sent_tmp.pop(i)
                    else:
                        for j in range(len(phrase_tmp)):  # remove
                            sent_tmp.pop(i)
                    for j in range(len(mask_ids)):  # remove
                        sent_tmp.insert(i+j, mask_ids[j])
                    if window > 0:
                        assert mask_opt in ["span_one","span_two","one","two"]
                        # print(tokenizer.convert_ids_to_tokens(sent))
                        # print(tokenizer.convert_ids_to_tokens(sent_tmp))
                        # print(tokenizer.convert_ids_to_tokens(phrase))
                        # print(idiom_idx)
                        mask_idx = Pre_Post_idx(sent_tmp, idiom_idx, window, tokenizer, stop_words, mask_opt)
                        mask_idx_list.append(mask_idx)
                        # 2*window
                    break
        if not flag:
            print(tokenizer.convert_ids_to_tokens(sent))
            print(tokenizer.convert_ids_to_tokens(phrase))
            raise Exception
        out.append(sent_tmp)
    idiom_col_idx = np.array(idiom_col_idx)  # bs, N_mask
    idiom_row_idx = np.arange(len(idiom_col_idx))[:, None]  # bs, 1
    # window, bs, NMASK
    if window > 0:
        mask_idx_list = np.array(mask_idx_list) #bs, N_span, n_mask
        # gold_ids_list = np.array(gold_ids_list) #bs, N_span, n_mask
        # padding_list = np.array(padding_list) #bs, N_span, n_mask
        return out, idiom_row_idx, idiom_col_idx, \
               mask_idx_list, None, None  # N_gold, N_sample, N_sent
    else:
        return out, idiom_row_idx, idiom_col_idx, None, None, None
        # N_gold, N_sample, N_sent

def Identify_Indices(tokenised_sentence_ids, phrase_ids, mask_ids, tokenizer):
    phrase_len = len(phrase_ids)
    assert phrase_len != 0
    phrase_tmp = phrase_ids.copy()
    phrase_col_idx = []
    out = []
    is_xlnet = tokenizer.name_or_path == "xlnet-large-cased"
    phrase_additional = []
    if is_xlnet:
        #XLNet tokeniser can produce different tokens depending on the surrounding context.
        if tokenizer.unk_token_id not in phrase_ids: #NO <unk>
            phrase_str = tokenizer.convert_ids_to_tokens(phrase_ids)
            phrase_str = "".join(phrase_str).replace("▁","")
            phrase_tmp = tokenizer(phrase_str, add_special_tokens=False)["input_ids"]
            assert phrase_tmp == phrase_ids, " ".join(tokenizer.convert_ids_to_tokens(phrase_tmp)) + " " + phrase_str
            phrase2 = tokenizer("a "+phrase_str,add_special_tokens=False)["input_ids"][1:]
            phrase3 = tokenizer("aa "+phrase_str,add_special_tokens=False)["input_ids"][2:]
            phrase4 = tokenizer(phrase_str +" a",add_special_tokens=False)["input_ids"][:-1]
            phrase5 = tokenizer(phrase_str +" aa",add_special_tokens=False)["input_ids"][:-2]
            phrase_set = set([tuple(phrase_tmp), tuple(phrase2),tuple(phrase3),tuple(phrase4),tuple(phrase5)])
            if len(phrase_set) == 1: #all segmentations are the same
                pass
            else:
                phrase_additional_all = phrase_set - set([tuple(phrase_tmp)])
                phrase_additional_all = list(phrase_additional_all) #[(tuple),(tuple)]
                for x in phrase_additional_all: #for each tuple
                    phrase_additional.append(list(x))
    for sid, sent in enumerate(tokenised_sentence_ids):
        flag = False
        sent_tmp = sent.copy()
        other_tok_found = False
        for i in range(len(sent_tmp)-len(phrase_tmp)+1):
            if len(phrase_additional):
                for e in range(len(phrase_additional)):
                    if all([sent_tmp[i + j] == phrase_additional[e][j] for j in range(len(phrase_additional[e]))]):
                        other_tok_found = True
                        to_fill = phrase_additional[e]
                        break
            if all([sent_tmp[i + j] == phrase_tmp[j] for j in range(len(phrase_tmp))])\
            or other_tok_found:
                flag = True
                phrase_idx = [i + j for j in range(len(mask_ids))]
                phrase_col_idx.append(phrase_idx)
                for j in range(phrase_len):  # remove
                    sent_tmp.pop(i)
                if other_tok_found:
                    assert mask_ids == phrase_ids
                    for j in range(len(to_fill)):  # remove
                        sent_tmp.insert(i + j, to_fill[j])
                else:
                    for j in range(len(mask_ids)):  # remove
                        sent_tmp.insert(i+j, mask_ids[j])
                break
        if not flag:
            print(tokenizer.convert_ids_to_tokens(sent))
            print(tokenizer.convert_ids_to_tokens(phrase_ids))
            print("NOT FOUND")
            raise Exception
        out.append(sent_tmp)
    phrase_col_idx = np.array(phrase_col_idx)  # bs, N_mask
    phrase_row_idx = np.arange(len(phrase_col_idx))[:, None]  # bs, 1
    return out, phrase_row_idx, phrase_col_idx

def tokenise_phrase(model, tokenizer, phrase):
    assert phrase[0]== " "
    if model.model_name == 'spanbert':
        premask_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(phrase.lstrip(" ")))
    else:
        if model.model_name in ['deberta-v2','deberta-v3',"bert","electra", 'fbmbart','fbmbart_MT', 'albert', 'mpnet', 'sbert-bert', 'sbert-mpnet']:
            premask_ids = tokenizer(phrase.lstrip(" "))["input_ids"][1:-1]  # remove CLS/SEP
        elif model.model_name in ['deberta','fbbart', 'roberta', 'sbert-roberta','deberta']:
            premask_ids = tokenizer(phrase)["input_ids"][1:-1]  # remove CLS/SEP
        elif model.model_name in ['gpt2', 'xlnet','pegasus','marian','t5']:
            premask_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]  # remove CLS/SEP
        premask_ids_ = tokenizer(phrase, add_special_tokens=False)["input_ids"]  # remove CLS/SEP
        assert premask_ids_ == premask_ids
    return premask_ids

def _create_perm_mask_and_target_map(seq_len, target_ids, device, mask_tgt = False):
    """
    Generates permutation mask and target mapping.
    If `self.masked` is true then there is no word that sees target word through attention.
    If it is false then only target word doesn't see itself.
    Args:
        seq_len: length of the sequence (context)
        target_ids: target word indexes
    Returns:
        two `torch.Tensor`s: permutation mask and target mapping
    """
    # assert isinstance(target_ids[0], int), "One target per sentence"
    # assert isinstance(target_ids[0], int), "One target per sentence"
    batch_size = len(target_ids)
    pred_len = len(target_ids[0])
    perm_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float)
    # target_mapping = torch.zeros((batch_size, 1, seq_len))
    target_mapping = torch.zeros((batch_size, pred_len, seq_len))
    for idx in range(batch_size):
        target_id = target_ids[idx]
        for k, each in enumerate(target_id):
            target_mapping[idx, k, each] = 1.0
        for k, each in enumerate(target_id):
            if mask_tgt:
                perm_mask[idx, :, each] = 1.0
            else:
                perm_mask[idx, each, each] = 1.0
    perm_mask = perm_mask.to(device)
    target_mapping = target_mapping.to(device)
    return perm_mask, target_mapping



def load_w2v(file, word_list = None, N_words= None):
    word2vec = {}
    if word_list is not None:
        word_list = set(word_list)
    with open(file, 'r', errors='ignore') as f:
        first_line = f.readline()
        first_line=first_line.split(' ')
        assert len(first_line)==2
        dim = int(first_line[1])
        for line in f:
            line = line.rstrip('\n')
            line = line.rstrip(' ')
            w = line.split(' ')[0]
            vec = line.split(' ')[1:]
            if word_list:
                if w in word_list:
                    word2vec[w] = [float(x) for x in vec]
            else:
                word2vec[w] = [float(x) for x in vec]
            if N_words and len(word2vec) == N_words:
                break
    return word2vec


def Get_pre_postindex(mask_col_idx, window, random_mask, input_ids, sentences_masked, tokenizer):
    #bs, N_mask
    SEP_idx_list = [len(x)-1 for x in sentences_masked]
    goldids = []
    prepost_mask_idx = []
    prepost_padding_idx = []
    for j, indices in enumerate(mask_col_idx): #bs, n_mask
        #indices: [13,14] #n_mask
        SEP_idx = SEP_idx_list[j]  #
        sentids = input_ids[j]
        padding = []
        non_mask_idx = []
        goldids_tmp = []
        # if random_mask:
        #     for k in range(1, window + 1):
        #         init_idx = max(1, indices[0]-10) #<CLS>
        #         last_idx = min(indices[-1]+10, SEP_idx)
        #         out_maskidx1 = random.sample(range(init_idx, indices[0]-1), 2)
        #         out_maskidx2 = random.sample(range(indices[-1]+1, last_idx), 2)
        #         non_mask_idx.append(out_maskidx1+out_maskidx2)  # omit CLS, SEP and PAD
        #         padding.append([True if (k >= 1 and k < SEP_idx)
        #                         else False for k in non_mask_idx[-1]])  # omit CLS, SEP and PAD
        #         goldids_tmp.append([sentids[idx] for idx in non_mask_idx[-1]])
        #         for special_token_id in [tokenizer.pad_token_id, tokenizer.mask_token_id,tokenizer.cls_token_id,tokenizer.sep_token_id]:
        #             assert special_token_id not in goldids_tmp[-1]
        # else:
        Nmask_out = 1
        if False:
            for k in range(1, window + 1):
                non_mask_idx.append([max(1, indices[0] - k),
                                     min(indices[-1] + k, SEP_idx)])
                padding.append([True if (u >= 1 and u < SEP_idx)
                                else False for u in non_mask_idx[-1]])  # omit CLS, SEP and PAD
                goldids_tmp.append([sentids[idx] for idx in non_mask_idx[-1]])
                for special_token_id in [tokenizer.pad_token_id, tokenizer.mask_token_id,tokenizer.cls_token_id,tokenizer.sep_token_id]:
                    assert special_token_id not in goldids_tmp[-1]
        elif False:
            for k in range(1, window + 1):
                non_mask_idx.append([max(1,indices[0] - k)])#,
                                     #max(0,indices[0] - k - 1),
                                     #max(0,indices[0] - k - 2)]) # omit CLS, SEP and P
                padding.append([True if (u < SEP_idx and u >= 1)
                                else False for u in non_mask_idx[-1]])  # omit CLS, SEP and
                goldids_tmp.append([sentids[idx] for idx in non_mask_idx[-1]])
                for special_token_id in [tokenizer.pad_token_id, tokenizer.mask_token_id,tokenizer.cls_token_id,tokenizer.sep_token_id]:
                    assert special_token_id not in goldids_tmp[-1]
                non_mask_idx.append([min(indices[-1] + k, SEP_idx)])#,
                                     # min(indices[-1] + k + 1, sent_len),  # omit CLS, SEP and P
                                     # min(indices[-1] + k + 2, sent_len)])  # omit CLS, SEP and P
                padding.append([True if (u < SEP_idx and u >= 1)
                                else False for u in non_mask_idx[-1]])  # omit CLS, SEP and
                goldids_tmp.append([sentids[idx] for idx in non_mask_idx[-1]])
                for special_token_id in [tokenizer.pad_token_id, tokenizer.mask_token_id,tokenizer.cls_token_id,tokenizer.sep_token_id]:
                    assert special_token_id not in goldids_tmp[-1]
        else:
            #*******span masking*******#
            non_mask_idx.append([max(1, indices[0] - k) for k in reversed(range(1, window + 1))])
            padding.append([True if (indices[0] - k >= 1)
                            else False for k in range(1, window + 1)])  # omit CLS, SEP and
            goldids_tmp.append([sentids[idx] for idx in non_mask_idx[-1]])
            for special_token_id in [tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.cls_token_id,tokenizer.sep_token_id]:
                assert special_token_id not in goldids_tmp[-1], [special_token_id] + tokenizer.convert_ids_to_tokens(goldids_tmp[-1])
            non_mask_idx.append([min(indices[-1] + k, SEP_idx - 1) for k in range(1, window + 1)])
            padding.append([True if (indices[-1] + k <= SEP_idx - 1)
                            else False for k in range(1, window + 1)])  # omit CLS, SEP and
            goldids_tmp.append([sentids[idx] for idx in non_mask_idx[-1]])
            for special_token_id in [tokenizer.pad_token_id, tokenizer.mask_token_id,tokenizer.cls_token_id,tokenizer.sep_token_id]:
                assert special_token_id not in goldids_tmp[-1], tokenizer.convert_ids_to_tokens(goldids_tmp[-1])
            prepost_mask_idx.append(non_mask_idx)  # TODO is it OK non_mask_idx has duplicate elements?
            prepost_padding_idx.append(padding)
            goldids.append(goldids_tmp)  # TODO is it OK non_mask_idx has duplicate elements?
            #non_mask_idx: [[2],[4],[1],[5]]
    prepost_mask_idx = np.array(prepost_mask_idx).transpose(1, 0, 2)  # window, bs, NMASK
    prepost_padding_idx = np.array(prepost_padding_idx).transpose(1, 0, 2)  # window, bs, NMASK
    goldids = np.array(goldids).transpose(1, 0, 2)  # window, bs, NMASK
    assert prepost_mask_idx.shape == prepost_padding_idx.shape
    assert prepost_mask_idx.shape == goldids.shape
    return prepost_mask_idx, prepost_padding_idx, goldids


def MASK_Idiom(tokenised_sentences_list, phrase,
               mask_token_id, n_mask, pre_mask, post_mask):
    #tokenised_sentences_list: N_gold, N_sent
    #phrase: ids for MWE [12,43,55]
    phrase_len = len(phrase)
    phrase_tmp = phrase.copy()
    mask_col_idx = []
    out = []
    for sent in tokenised_sentences_list:
        sent_len = len(sent)
        x = False
        sent_tmp = sent.copy()
        for i, word in enumerate(sent):
            if all([sent[i+j] == phrase_tmp[j] for j in range(len(phrase_tmp))]):
                for j in range(pre_mask):
                    idx = i - 1 - j  #before idiom (starting at i)
                    if idx >= 0:
                        sent_tmp[idx] = mask_token_id
                for j in range(post_mask):
                    idx = i + phrase_len + j #after idiom (ending at i+phrase_len)
                    if idx < sent_len - 1: #<SEP>
                        sent_tmp[idx] = mask_token_id
                if n_mask != 0:
                    for j in range(phrase_len): #remove
                        sent_tmp.pop(i)
                    for _ in range(n_mask):
                        sent_tmp.insert(i, mask_token_id)
                    mask_idx = [i + j for j in range(n_mask)]
                else:
                    mask_idx = [i + j for j in range(phrase_len)]
                mask_col_idx.append(mask_idx)
                x = True
                break
        assert x #contains the idiom
        out.append(sent_tmp)
    mask_col_idx = np.array(mask_col_idx)  # bs, N_mask
    mask_row_idx = np.arange(len(mask_col_idx))[:, None]  # bs, 1
    return out, mask_row_idx, mask_col_idx #N_gold, N_sample, N_sent


def Find_Idiom_and_MASK(raw_sentences, n_mask, tokenizer):
    #tokenised_sentences_list: N_gold, N_sent
    #phrase: [old, flame] or [med, ##ics]
    mask_col_idx = []
    out = []
    for sent in raw_sentences:
        x = False
        sent_tmp = sent.copy() #I look[SEP_WORD]forward[SEP_WORD]to seeing you.
        for i, word in enumerate(sent):
            if "[SEP_WORD]" in word: #look[SEP_WORD]forward[SEP_WORD]to
                phrase = " ".join(word.split("[SEP_WORD]"))
                if i != 0 :
                    phrase = " " + phrase #add Ġ for the unicode tokenizer
                phrase = tokenizer.tokenize(phrase)
                pre = tokenizer(sent[:i])
                post = tokenizer(sent[i+1:])
                MWE_start_idx = len(pre)
                mask  = [tokenizer.mask_token for _ in range(n_mask)]
                if n_mask != 0:
                    sent_tok = pre + mask + post
                    mask_idx = [MWE_start_idx + j for j in range(n_mask)]
                else:
                    sent_tok = pre + phrase + post
                    mask_idx = [MWE_start_idx + j for j in range(len(phrase))]
                mask_col_idx.append(mask_idx)
                x = True
                break
        assert x #contains the idiom
        assert "[SEP_WORD]" not in sent_tmp
        out.append(sent_tok)
    mask_col_idx = np.array(mask_col_idx)  # bs, N_mask
    mask_row_idx = np.arange(len(mask_col_idx))[:, None]  # bs, 1
    return out, mask_row_idx, mask_col_idx, phrase #N_gold, N_sample, N_sent

def Get_index(input_ids, mask_token_id_list, token_length, pre_postmask, opt):
    mask_col_idx = []
    prepost_mask_idx = []
    prepost_goldidx = []
    prepost_padding_idx = []
    for j, ids in enumerate(input_ids.data.numpy()):
        flag = 0
        for i, word in enumerate(ids):
            if all([ids[i + k] == mask_token_id_list[k] for k in range(len(mask_token_id_list))]):
                mask_idx = [i + k for k in range(len(mask_token_id_list))]
                mask_col_idx.append(mask_idx)
                flag = 1
                if pre_postmask:
                    sent_len = token_length[j]  # sent_len w.o PAD
                    padding = []
                    non_mask_idx = []
                    if opt.random_mask:
                        for k in range(1, opt.window + 1):
                            # out_maskidx = list(range(init_id, len(ids) + init_id, 15))
                            init_idx = max(1, mask_idx[0]-10)
                            last_idx = min(mask_idx[-1]+10, len(ids))
                            out_maskidx1 = random.sample(range(init_idx, mask_idx[0]-1), 2)
                            out_maskidx2 = random.sample(range(mask_idx[-1]+1, last_idx), 2)
                            # out_maskidx = random.sample(range(idx1, idx2), 8)
                            non_mask_idx.append(out_maskidx1+out_maskidx2)  # omit CLS, SEP and PAD
                            padding.append([True if (k <= sent_len - 2 and k >= 1 and k not in mask_idx)
                                            else False for k in non_mask_idx[-1]])  # omit CLS, SEP and PAD
                    else:
                        Nmask_out = 1
                        if Nmask_out == 2:
                            for k in range(1, opt.window + 1):
                                non_mask_idx.append([max(0, mask_idx[0] - k),
                                                     min(mask_idx[-1] + k, sent_len - 1)])  # omit CLS, SEP and P
                                padding.append([True if (u <= sent_len - 2 and u >= 1 and u not in mask_idx)
                                                else False for u in non_mask_idx[-1]])  # omit CLS, SEP and PAD
                        else:
                            for k in range(1, opt.window + 1):
                                non_mask_idx.append([max(0, mask_idx[0] - k - 2)])#,
                                                     #max(0, mask_idx[0] - k - 1),
                                                     #max(0, mask_idx[0] - k)])  # omit CLS, SEP and P
                                padding.append([True if (u <= sent_len - 2 and u >= 1 and u not in mask_idx)
                                                else False for u in non_mask_idx[-1]])  # omit CLS, SEP and

                                non_mask_idx.append([min(mask_idx[-1] + k, sent_len - 1)])#,
                                                     #min(mask_idx[-1] + k + 1, sent_len - 1),  # omit CLS, SEP and P
                                                     #min(mask_idx[-1] + k + 2, sent_len - 1)])  # omit CLS, SEP and P
                                padding.append([True if (u <= sent_len - 2 and u >= 1 and u not in mask_idx)
                                                else False for u in non_mask_idx[-1]])  # omit CLS, SEP and
                    prepost_padding_idx.append(padding)
                    prepost_mask_idx.append(non_mask_idx)  # TODO is it OK non_mask_idx has duplicate elements?
                    prepost_goldidx.append([ids[idx] for idx in non_mask_idx])
                break
        assert flag == 1  ##contains [MASK]

    mask_col_idx = np.array(mask_col_idx)  # bs, N_mask
    mask_row_idx = np.arange(len(mask_col_idx))[:, None]  # bs, 1
    if pre_postmask:
        prepost_goldidx = np.array(prepost_goldidx).transpose(1, 0, 2)  # window, bs, NMASK
        prepost_mask_idx = np.array(prepost_mask_idx).transpose(1, 0, 2)  # window, bs, NMASK
        prepost_padding_idx = np.array(prepost_padding_idx).transpose(1, 0, 2)  # window, bs, NMASK

    return mask_col_idx, mask_row_idx, prepost_goldidx,prepost_mask_idx, prepost_padding_idx

def _Encode_model(model, input_ids, token_ids, attn, col_idx):
    attentions = None
    # self.emb_dropout = nn.Dropout(p=dr_rate)
    if model.model_name in ["bert","jbert"]:
        outputs = model.bert(input_ids = input_ids,
                             # inputs_embeds=input_embbeddings,
                             attention_mask= attn,
                             token_type_ids= token_ids,
                             output_hidden_states=True,
                             output_attentions = True)
        last_layer_tmp = outputs[0].detach()  # bs, seq_len, dim
        all_L_tmp = outputs['hidden_states'] #
        ffn_tmp = model.cls.predictions.transform(last_layer_tmp)#[row_idx, col_idx]
        attentions = torch.stack(outputs['attentions'], dim = 0) #L, bs, N_head, seq_len, seq_len
        attentions = attentions.mean(2) #L, bs, seq_len, seq_len

    elif model.model_name in ["spanbert"]:
        outputs = model.bert(input_ids = input_ids,
                             attention_mask= attn,
                             token_type_ids= token_ids,
                             output_hidden_states=True)
                             # output_attentions = True)
        last_layer_tmp = outputs[0].detach()  # bs, seq_len, dim
        all_L_tmp = outputs[-2]
        assert len(all_L_tmp)==25
        attentions = torch.stack(outputs[-1], dim=0)  # L, bs, N_head, seq_len, seq_len
        assert len(attentions) == 24
        attentions = attentions.mean(2)
        ffn_tmp = model.cls.predictions.transform(last_layer_tmp)#[row_idx, col_idx]

    elif model.model_name == 'xlnet':
        seq_len = input_ids.size()[1]
        perm_mask, target_mapping = _create_perm_mask_and_target_map(seq_len, col_idx, input_ids.device, True)
        outputs = model.transformer(
            input_ids = input_ids,
            perm_mask= perm_mask,
            target_mapping = target_mapping,
            attention_mask = attn,
            output_hidden_states = True,
            output_attentions=True
        )
        last_layer_tmp = outputs["last_hidden_state"] # g embeddings
        # print(outputs[-1])
        # last_layer_tmp = last_layer_tmp.repeat(1,N_MWE,1)
        all_L_tmp = outputs["hidden_states"]
        ffn_tmp = last_layer_tmp
        attn_h = torch.stack([x[0] for x in outputs['attentions']], dim=0).mean(2)
        attn_g = torch.stack([x[1] for x in outputs['attentions']], dim=0).mean(2)
        assert len(attn_h)==len(attn_g)
        attentions = []
        for i in range(len(attn_h)):
            attentions.append(attn_h[i])
            attentions.append(attn_g[i])
        attentions = torch.stack(attentions)
        #L, bs, seq_len, seq_len
        # predictions = predictions[:, 0, :]
    elif model.model_name == 'electra':
        # emb_tmp = input_embbeddings[i:i + bs]
        outputs = model.electra(input_ids=input_ids,
                                attention_mask=attn,
                                token_type_ids=token_ids,
                                output_hidden_states=True)
        last_layer_tmp = outputs[0].detach()  # bs, seq_len, dim
        all_L_tmp = outputs[-1]
        ffn_tmp = last_layer_tmp
        # ffn_tmp = model.discriminator_predictions.dense(last_layer_tmp)
        # ffn_tmp = electra_get_activation(model.discriminator_predictions.config.hidden_act)(ffn_tmp)
        # logits = model.discriminator_predictions.dense_prediction(ffn_tmp).squeeze(-1)

    elif model.model_name == 'deberta-v3':
        outputs = model(input_ids=input_ids,
                        attention_mask=attn,
                        token_type_ids=token_ids,
                        output_hidden_states=True,
                        output_attentions = True)
        last_layer_tmp = outputs["last_hidden_state"].detach()  # bs, seq_len, dim
        all_L_tmp = outputs["hidden_states"]
        ffn_tmp = last_layer_tmp
        attentions = torch.stack(outputs['attentions'], dim=0)  # L, bs, N_head, seq_len, seq_len
        attentions = attentions.mean(2)  # L, bs, seq_len, seq_len
        # logits = model.discriminator_predictions.dense_prediction(ffn_tmp).squeeze(-1)

    elif model.model_name in ['deberta','deberta-v2']:
        outputs = model.deberta(input_ids=input_ids,
                        attention_mask=attn,
                        token_type_ids=token_ids,
                        output_hidden_states=True,
                        output_attentions = True)
        last_layer_tmp = outputs["last_hidden_state"].detach()  # bs, seq_len, dim
        all_L_tmp = outputs["hidden_states"]
        ffn_tmp = last_layer_tmp
        #NOTE there is a bug in transform
        # ffn_tmp = model.cls.predictions.transform(last_layer_tmp)  # [row_idx, col_idx]
        attentions = torch.stack(outputs['attentions'], dim=0)  # L, bs, N_head, seq_len, seq_len
        attentions = attentions.mean(2)  # L, bs, seq_len, seq_len
        # logits = model.discriminator_predictions.dense_prediction(ffn_tmp).squeeze(-1)

    elif model.model_name.startswith("sbert"):
        # No ML head
        # emb_tmp = input_embbeddings[i:i + bs]
        outputs = model(input_ids=input_ids,
                        attention_mask=attn,
                        output_hidden_states=True)
        last_layer_tmp = outputs[0].detach()  # bs, seq_len, dim
        all_L_tmp = outputs[-1]
        ffn_tmp = last_layer_tmp#[row_idx, col_idx]

    elif model.model_name in ["albert"]:
        # emb_tmp = input_embbeddings[i:i + bs]
        outputs = model.albert(input_ids=input_ids,
                               attention_mask=attn,
                               token_type_ids=token_ids,
                               output_hidden_states=True,
                               output_attentions = True)
        last_layer_tmp = outputs[0]  # bs, seq_len, dim
        ffn_tmp = model.predictions.dense(last_layer_tmp)
        ffn_tmp = model.predictions.activation(ffn_tmp)
        ffn_tmp = model.predictions.LayerNorm(ffn_tmp)
        all_L_tmp = outputs[-2]
        assert len(all_L_tmp) == 25
        attentions = torch.stack(outputs[-1], dim=0)  # L, bs, N_head, seq_len, seq_len
        assert len(attentions) == 24
        attentions = attentions.mean(2)

    elif model.model_name in ["marian"]:
        outputs = model.model.encoder(input_ids=input_ids,
                                      attention_mask=attn,
                                      output_hidden_states=True)
        last_layer_tmp = outputs[0]  # bs, seq_len, dim
        ffn_tmp = last_layer_tmp #[row_idx, col_idx]  # bs, N_mask, dim
        all_L_tmp = outputs['hidden_states'] #7
    elif model.model_name in ["fbmbart_MT"]:
        #
        # outputs = model.model(input_ids = input_ids,
        #                       attention_mask = attn,
        #                       output_hidden_states=True)
        outputs = model.model.encoder(input_ids = input_ids,
                              attention_mask = attn,
                              output_hidden_states=True)
        # print("enc")
        last_layer_tmp = outputs["last_hidden_state"] #bs, seq_len, dim
        ffn_tmp = last_layer_tmp  # [row_idx, col_idx]
        all_L_tmp = outputs["hidden_states"]  # N_layer, bs, seq_len, dim

        # input: lang_id, w1,...wn, EOS
        # dec_input_tmp = input_ids.clone()
        # dec_input_tmp[:, 0] = tokenizer.tgt_lang_id
        # # dec_input_tmp[row_idx, col_idx] = unk_token_id
        # dec_input_tmp = mbart_shift_tokens_right(dec_input_tmp, model.model.config.pad_token_id)
        # # dec_input: EOS, LANG_id, w1,...,wn
        # # dec_output: LANG_id, w1,..., EOS
        # outputs = model.model(input_ids=input_ids,
        #                       decoder_input_ids=dec_input_tmp,
        #                       attention_mask=attn,
        #                       output_hidden_states=True)
        # last_layer_tmp = outputs[0]
        # ffn_tmp = last_layer_tmp #[row_idx, col_idx]
        # all_L_tmp = outputs["encoder_hidden_states"]  # N_layer, bs, seq_len, dim
        # all_L_tmp = all_L_tmp + outputs["decoder_hidden_states"]  # decoder =h_t-1
        # all_L_tmp = all_L_tmp + tuple([x[:, 1:] for x in outputs["decoder_hidden_states"]])  # h_t
    elif model.model_name in ["pegasus"]:
        outputs = model.model.encoder(input_ids = input_ids,
                              attention_mask = attn,
                              output_hidden_states=True)
        # print("enc")
        last_layer_tmp = outputs["last_hidden_state"] #bs, seq_len, dim
        ffn_tmp = last_layer_tmp  # [row_idx, col_idx]
        all_L_tmp = outputs["hidden_states"]  # N_layer, bs, seq_len, dim

    elif model.model_name in ["t5"]:
        outputs = model.encoder(input_ids = input_ids,
                              attention_mask = attn,
                              output_hidden_states=True)
        # print("enc")
        last_layer_tmp = outputs["last_hidden_state"] #bs, seq_len, dim
        ffn_tmp = last_layer_tmp  # [row_idx, col_idx]
        all_L_tmp = outputs["hidden_states"]  # N_layer, bs, seq_len, dim

    elif model.model_name in ["fbbart"]:
        dec_input_tmp = input_ids.clone()
        # dec_input_tmp[row_idx, col_idx] = unk_token_id
        dec_input_tmp = mono_shift_tokens_right(dec_input_tmp, model.model.config.pad_token_id,
                                                model.model.config.decoder_start_token_id)
        outputs = model.model(input_ids=input_ids,
                              decoder_input_ids=dec_input_tmp,
                              attention_mask=attn,
                              output_hidden_states=True)
        last_layer_tmp = outputs[0]
        ffn_tmp = last_layer_tmp #[row_idx, col_idx]
        all_L_tmp = outputs["encoder_hidden_states"]  # N_layer, bs, seq_len, dim
        # print(all_L_tmp[-1][5])
        all_L_tmp = all_L_tmp + outputs["decoder_hidden_states"]  # decoder =h_t-1
        all_L_tmp = all_L_tmp + tuple([x[:, 1:] for x in outputs["decoder_hidden_states"]])  # h_t
        # print(all_L_tmp[-1][5])

    elif model.model_name in ["gpt2"]:
        outputs = model.transformer(input_ids=input_ids,
                                    attention_mask=attn,
                                    output_hidden_states=True,
                                    use_cache=True)
        last_layer_tmp = outputs[0]
        ffn_tmp = last_layer_tmp #[row_idx, col_idx]  # bs, n_MWE, dim
        all_L_tmp = outputs["hidden_states"]  # N_layer, bs, seq_len, dim
        dummpy = torch.zeros((last_layer_tmp.size(0), 1, last_layer_tmp.size(0)))
        all_L_tmp = all_L_tmp + tuple([torch.cat([dummpy, x], dim=1) for x in outputs["hidden_states"]])  # h_t-1
    else:
        raise Exception
    # ffn_tmp = ffn_tmp[row_idx, col_idx]
    return all_L_tmp, ffn_tmp, attentions

def Get_Attention(tokenizer, model, sent_list, mask_col_idx,
                    max_tokens=8192, layers = None, norm_attn = False, from_attn = False):
    #around_idx_list: 2, bs, window
    sorted_idx = np.argsort([-1 * len(x) for x in sent_list])
    original_idx = np.argsort(sorted_idx)
    input_ids_tmp = []
    MWE_all_states = []
    attentionList = []
    MWE_ffn = []
    max_len = None
    max_len_all = len(sent_list[sorted_idx[0]])
    idx_list_tmp=[]
    for i in sorted_idx: #sort by sent len
        sent = sent_list[i].copy()
        idx_list_tmp.append(i)
        if max_len == None:
            max_len = len(sent)
        assert isinstance(sent, list)
        input_ids_tmp.append(sent + [tokenizer.pad_token_id] * (max_len - len(sent)))
        if len(input_ids_tmp) * max_len>=max_tokens or i == sorted_idx[-1]:
            col_idx = mask_col_idx[idx_list_tmp] #bs, idx
            row_idx = np.arange(len(col_idx))[:, None]
            if torch.cuda.is_available():
                input_ids_tmp = torch.cuda.LongTensor(input_ids_tmp)
            else:
                input_ids_tmp = torch.LongTensor(input_ids_tmp)
            N_sent, N_token = input_ids_tmp.size()
            device = input_ids_tmp.device
            token_tmp = torch.LongTensor([0] * (N_sent * N_token)).view(N_sent, N_token).to(device)
            attn_mask_tmp = input_ids_tmp != tokenizer.pad_token_id
            attn_mask_tmp = attn_mask_tmp.long()
            all_L_tmp, last_layer_tmp, attentions = \
                _Encode_model(model, input_ids_tmp, token_tmp, attn_mask_tmp, col_idx)
            assert layers is not None
            ffn_tmp = last_layer_tmp[row_idx,col_idx]
            if from_attn:
                #attentions: L, bs, seq_len, seq_len
                attentions = attentions.transpose(2,3)
                attentions = torch.stack([attentions[k][row_idx, col_idx].detach() for k in layers])
            else:
                if model.model_name == "xlnet":
                    attentions = torch.stack([attentions[k][row_idx, col_idx] if k % 2 == 0 else all_L_tmp[k] for k in layers])
                else:
                    attentions = torch.stack([attentions[k][row_idx, col_idx].detach() for k in layers])
                if norm_attn:
                    # attentions: L, bs, n_mask, seq_len
                    # h_norm: L, bs, seq_len
                    # all_L_tmp: L, bs, seq_len, dim
                    h_norm = torch.stack([all_L_tmp[k].norm(dim=-1) for k in layers])
                    h_norm = F.softmax(h_norm, dim=-1)
                    # print(h_norm[0])
                    # print(h_norm[-1])
                    # print(attentions[0])
                    # print(attentions[-1])
                    attentions = attentions * h_norm.unsqueeze(2).expand_as(attentions)
            attentions = attentions.mean(0) #bs, n_mask, seq_len
            #bs, n_mask, N_words
            if model.model_name == "xlnet":
                MWE_all_states_tmp = torch.stack([all_L_tmp[k][row_idx, col_idx] if k%2==0 else all_L_tmp[k] for k in layers])
            else:
                MWE_all_states_tmp = torch.stack([all_L_tmp[k][row_idx, col_idx].detach() for k in layers])
            MWE_ffn.append(ffn_tmp)
            MWE_all_states.append(MWE_all_states_tmp)
            bs, n_mask, tmp = attentions.size()
            assert tmp == max_len
            padding = torch.zeros((bs, n_mask, max_len_all-max_len)).to(attentions.device)
            attentions = torch.cat([attentions, padding],dim=-1) #set the attn size same
            attentionList.append(attentions)
            max_len = None
            input_ids_tmp = []
            idx_list_tmp = []

    MWE_ffn = torch.cat(MWE_ffn, dim=0) #bs, n_mask, dim
    MWE_all_states = torch.cat(MWE_all_states, dim=1) #N_layer, bs, n_mask, dim
    MWE_ffn = MWE_ffn[original_idx]
    MWE_all_states = MWE_all_states[:, original_idx]
    attentionList = torch.cat(attentionList, dim=0)  # bs, n_mask, N_words
    attentionList = attentionList[original_idx]
    return MWE_all_states, MWE_ffn, attentionList





def Encode_BERT_PAD(tokenizer, model, sent_list, mask_col_idx,
                    max_tokens=8192, layers = None,
                    around_idx_list = None, padding_list=None, bert_ls = False, dropout= [0,0]):
    #around_idx_list: 2, bs, window
    if padding_list is not None:
        assert around_idx_list is not None
        if torch.cuda.is_available():
            padding_list = torch.cuda.FloatTensor(padding_list)
        else:
            padding_list = torch.FloatTensor(padding_list)
    sorted_idx = np.argsort([-1 * len(x) for x in sent_list])
    original_idx = np.argsort(sorted_idx)
    input_ids_tmp = []
    MWE_all_states = []
    attentionList = []
    sent_states = []
    around_states = []
    MWE_ffn = []
    # max_len = np.max([len(x) for x in sent_list])
    max_len = None
    max_len_all = len(sent_list[sorted_idx[0]])
    idx_list_tmp=[]
    for i in sorted_idx: #sort by sent len
        sent = sent_list[i].copy()
        idx_list_tmp.append(i)
        if max_len == None:
            max_len = len(sent)
        assert isinstance(sent, list)
        # assert len(sent) <= max_len
        input_ids_tmp.append(sent + [tokenizer.pad_token_id] * (max_len - len(sent)))
        if len(input_ids_tmp) * max_len>=max_tokens or i == sorted_idx[-1]:
            col_idx = mask_col_idx[idx_list_tmp] #bs, idx
            col_idx_sur = np.array([[min(col_idx[i])-1,max(col_idx[i])+1] for i in range(len(col_idx))])
            row_idx = np.arange(len(col_idx))[:, None]
            if padding_list is not None:
                around_idx = around_idx_list[:,idx_list_tmp]  #2, bs, window
                padding_idx = padding_list[:,idx_list_tmp]  #2, bs, window
            if torch.cuda.is_available():
                input_ids_tmp = torch.cuda.LongTensor(input_ids_tmp)
            else:
                input_ids_tmp = torch.LongTensor(input_ids_tmp)
            N_sent, N_token = input_ids_tmp.size()
            # print(input_ids_tmp)
            # print(N_sent)
            device = input_ids_tmp.device
            token_tmp = torch.LongTensor([0] * (N_sent * N_token)).view(N_sent, N_token).to(device)
            attn_tmp = input_ids_tmp != tokenizer.pad_token_id
            attn_tmp = attn_tmp.long()
            all_L_tmp, last_layer_tmp, attentions = \
                _Encode_model(model, input_ids_tmp, token_tmp,attn_tmp, col_idx)
            assert layers is not None
            if model.model_name == "xlnet":
                ffn_tmp = last_layer_tmp
            else:
                ffn_tmp = last_layer_tmp[row_idx,col_idx]
            if attentions is not None:
                #all_L_tmp: L, bs, seq_len, dim
                #attentions: L, bs, seq_len, seq_len
                # attn form i to j
                # attentions = attentions.transpose(1,2) # attn form j to i
                attentions = torch.stack([attentions[k][row_idx, col_idx].detach() for k in layers])
                # h_norm = torch.stack([all_L_tmp[k].norm(dim=-1) for k in layers])
                #attentions: L, bs, n_mask, seq_len
                #h_norm: L, bs, seq_len
                # attentions = attentions * h_norm.unsqueeze(2).expand_as(attentions)
                attentions = attentions.mean(0) #bs, n_mask, seq_len
                #bs, n_mask, N_words
            if model.model_name == "xlnet":
                MWE_all_states_tmp = torch.stack([all_L_tmp[k][row_idx, col_idx] if k%2==0 else all_L_tmp[k] for k in layers])
            else:
                MWE_all_states_tmp = torch.stack([all_L_tmp[k][row_idx, col_idx].detach() for k in layers])
            MWE_ffn.append(ffn_tmp)
            MWE_all_states.append(MWE_all_states_tmp)
            if attentions is not None:
                bs, n_mask, tmp = attentions.size()
                assert tmp == max_len
                padding = torch.zeros((bs, n_mask, max_len_all-max_len)).to(attentions.device)
                attentions = torch.cat([attentions, padding],dim=-1) #set the attn size same
                attentionList.append(attentions)
            max_len = None
            input_ids_tmp = []
            idx_list_tmp = []

    MWE_ffn = torch.cat(MWE_ffn, dim=0) #bs, n_mask, dim
    MWE_all_states = torch.cat(MWE_all_states, dim=1) #N_layer, bs, n_mask, dim
    MWE_ffn = MWE_ffn[original_idx]
    MWE_all_states = MWE_all_states[:, original_idx]
    if attentions is not None:
        attentionList = torch.cat(attentionList, dim=0)  # bs, n_mask, N_words
        attentionList = attentionList[original_idx]
    return MWE_all_states, MWE_ffn, around_states, sent_states, attentionList

def SpanBERT_SBO(model, last_layer, pairs_idx, n_mask, remove_bias, i):
    pairs_idx = np.array(pairs_idx)  # bs, 2; pre and post idx
    pairs = torch.LongTensor(pairs_idx.reshape(-1, 1, 2))  # bs, 1, 2
    if torch.cuda.is_available():
        pairs = pairs.to("cuda")
    ##BertPairTargetPredictionHead.forward()###
    _, num_pairs, _ = pairs.size()
    bs, seq_len, dim = last_layer.size() #last layer
    # pair indices: (bs, num_pairs)
    left, right = pairs[:, :, 0], pairs[:, :, 1] #bs, 1
    # (bs, num_pairs)
    left_hidden = torch.gather(last_layer, 1, left.unsqueeze(2).repeat(1, 1, dim))
    # pair states: bs * num_pairs, max_targets, dim
    left_hidden = left_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, model.cls.pair_target_predictions.max_targets, 1)

    right_hidden = torch.gather(last_layer, 1, right.unsqueeze(2).repeat(1, 1, dim))
    # bs * num_pairs, max_targets, dim
    right_hidden = right_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, model.cls.pair_target_predictions.max_targets,1)
    # (max_targets, dim)
    position_embeddings = model.cls.pair_target_predictions.position_embeddings.weight
    if i ==0:
        left_hidden *=0
    elif i ==1:
        right_hidden *=0
    else:
        raise Exception
    hidden_states = model.cls.pair_target_predictions.mlp_layer_norm(
        torch.cat((left_hidden, right_hidden, position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1)),-1))
    # target scores : bs, max_targets, dim
    hidden_states = hidden_states[:,:n_mask,:] #bs, N_mask, dim
    prediction_scores = model.Output_layer.mm(hidden_states.view(-1, model.Output_layer.size()[-1]).T)  # V, bs*N_mask
    if not remove_bias:
        prediction_scores = prediction_scores + model.Output_layer_bias.expand_as(prediction_scores)  # V, N_mask

    return hidden_states, hidden_states.mean(dim=0), prediction_scores.view(len(model.Output_layer), -1, n_mask) #V, bs, n_mask
# K = len(probability)
    # mask_size = mask_col_idx.shape[1]
    # top_idx = probability.topk(K, dim=0)  # K,N_MASK
    # prob_topk_sum = top_idx[0].sum(0)  # N_MASK
    # average_emb = model.Output_layer.view(V, 1, embdim).expand((V, mask_size, embdim)) * probability.view(V, mask_size,1).expand(
    #     (V, mask_size, embdim)) #V,mask_size, dim
    # average_emb = torch.stack(
    #     [average_emb[top_idx[1][:, j], j].sum(dim=0) / prob_topk_sum[j] for j in range(mask_size)])  # N_mask, dim
    # return last_layer, MWE_mean, average_emb, top_idx

def Mean_Probability(probability, meantype):
    #mean probability over dim = 1
    if meantype == "geomean":
        probability = torch.exp(torch.log(probability).mean(dim=1))  # V, N_mask
    elif meantype.startswith("geomean_smooth"):
        alpha= float(meantype.split("_")[-1])
        uniform = 1/len(probability) #uniform distribution
        probability = alpha*probability +(1-alpha)*uniform
        probability = probability / torch.sum(probability, dim=0, keepdim=True)
        probability = torch.exp(torch.log(probability).mean(dim=1))  # V, N_mask
    elif meantype == "average":
        probability = probability.mean(1)
    elif meantype == "None":
        probability = probability.view(len(probability), -1)
    elif meantype == "max":
        #probability:
        maxidx = probability.argmax(dim=0).cpu().numpy()  # bs, n_mask
        # V, bs, n_mask
        ########advanced indexing#########
        one_hot = np.zeros(probability.shape)
        bs, n_mask = probability.shape[1:]
        I, J = np.ogrid[:bs, :n_mask]
        one_hot[maxidx,I,J] = 1
        one_hot = np.mean(one_hot,axis = 1) #V, n_mask
        probability = torch.FloatTensor(one_hot).to(probability.device)
    # elif meantype == "topk_average":
    #     probability = torch.zeros(V, len(MWE_mean)).to(prediction_scores.device)  # V, n_mask
    #     for i in range(batch_size):
    #         for k in range(n_mask):
    #             prob = top_prob[:, i, k]  # 5
    #             idx = topwords[:, i, k]  # 5
    #             probability[idx, k] += prob
    #     probability = probability / torch.sum(probability, dim=0, keepdim=True)
    else:
        raise Exception

    return probability


def Count_Predwords(probability,topk = 10):
    # the most frequently predicted words in the topk
    _, topwords = probability.topk(topk, dim=0) #10, bs, n_mask
    Most_Pred_words = list(zip(*topwords[0].T.data.cpu().tolist())) #bs; [(4,7),(23,4),...]
    Most_Pred_words = Counter(Most_Pred_words).most_common() #
    Most_Common_Count = []  # K
    Most_Common_MWEidx = []  # K*n_mask
    V, bs, n_mask = probability.size()
    for i in range(len(Most_Pred_words)):
        jointprob = [Most_Pred_words[i][1]/bs for _ in range(n_mask)]
        Most_Common_MWEidx.append(list(Most_Pred_words[i][0]))#K, n_mask
        Most_Common_Count.append(jointprob) #K, n_mask

    Most_Common_MWEidx = np.array(Most_Common_MWEidx)
    Most_Common_Count = np.array(Most_Common_Count)
    # pred_words = np.array(tokenizer.convert_ids_to_tokens(Most_Common_MWEidx.reshape(-1))).reshape(-1, n_mask)
    # print(Most_Common_Count)
    # print(pred_words)
    Most_Common_MWEidx = torch.LongTensor(Most_Common_MWEidx).to(probability.device)
    Most_Common_Count = torch.FloatTensor(Most_Common_Count).to(probability.device)

    return (Most_Common_Count, Most_Common_MWEidx)

# def Analyse_Embedding(average_emb, phrase_wordemb_normalised, phrase_subwordemb, phrase_subwordemb_flat):
#     #phrase_subwordemb_flat: ALL_tokens, dim
#     #phrase_wordemb_normalised:  dim
#     #phrase_subwordemb: N_mask, N_tokens_i dim
#     w1 = F.normalize(phrase_subwordemb[0], dim=-1)  # N_token1,dim
#     w2 = F.normalize(phrase_subwordemb[-1], dim=-1)  # N_token2,dim
#     phrase_flat_normalised = F.normalize(phrase_subwordemb_flat, dim=-1)  # N_token2,dim
#     phrase_normalised_sum =  F.normalize(phrase_flat_normalised.sum(dim=0), dim=-1)
#     average_emb_normalised = F.normalize(average_emb, dim=-1).T  # dim, N_mask
#     average_sumemb_normalised = F.normalize(average_emb.sum(0), dim=-1)# dim, N_mask
#     average_normalised_sumemb = F.normalize(F.normalize(average_emb, dim=-1).sum(0),dim = -1)# dim, N_mask
#     inner_cossim = (average_emb_normalised[:, 0] * average_emb_normalised[:, -1]).sum()
#     inner_cossim = np.round(inner_cossim.data.cpu().numpy(),2)
#     sum_compositionality = (phrase_wordemb_normalised * average_sumemb_normalised).sum()
#     sum_compositionality = np.round(sum_compositionality.data.cpu().numpy(),2)
#     normsum_compositionality = (phrase_normalised_sum * average_normalised_sumemb).sum()
#     normsum_compositionality = np.round(normsum_compositionality.data.cpu().numpy(),2)
#     cossim_MWE_phrase2_1 = w1.mm(average_emb_normalised[:, 0].view(-1, 1)).max()  #
#     cossim_MWE_phrase2_2 = w2.mm(average_emb_normalised[:, -1].view(-1, 1)).max()  #
#     cossim_Phrase_MWE = phrase_flat_normalised.mm(average_emb_normalised)  # N_tokens, N_MASK
#     cossim_Phrase_MWE = cossim_Phrase_MWE.max(dim=0)[0] #Min_for_MASK(Max_for_tokens)
#     cossim_Phrase_MWE = np.round(cossim_Phrase_MWE.data.cpu().numpy(), 2)
#     x = Output_layer_norm.mm(average_emb_normalised)  # V,N_MASK
#     # print("cossim(sum(phrase),sum(MASK))")
#     # print(sum_compositionality)
#     # print("min_mask_max_token(cossim(token,MWE))")
#     # print(np.min(cossim_Phrase_MWE))
#     # print("max_max_token(cossim(token,MWE))")
#     # print(np.max(cossim_Phrase_MWE))
#     # print("min{cossim(phrase0,MASK0),cossim(phrase1,MASK1))")
#     # print(torch.min(cossim_MWE_phrase2_1, cossim_MWE_phrase2_2))
#     # print("top NN words")
#     NN_Words = np.array(tokenizer.convert_ids_to_tokens(x.topk(5, dim=0)[1].cpu().numpy().reshape(-1))).reshape(5, mask_size)
#     # print(NN_Words.T)
#     return NN_Words, inner_cossim, sum_compositionality, normsum_compositionality, cossim_Phrase_MWE

def Expected_embedding(probability, model):
    #probability: V, n_mask
    probability_tmp = probability.clone()
    V = len(probability_tmp)
    embdim = model.Output_layer.size()[-1]
    K = len(probability_tmp)
    # tokens = phrase_first_tokens
    # if any(bias_flag) and alpha > 1.0 or beta > 0:
    #     for i in range(len(tokens)):
    #         if bias_flag[i]:
    #             probability_tmp[tokens[i], i] *= alpha
    #             probability_tmp[tokens[i], i] += beta
    #     probability_tmp = probability_tmp / probability_tmp.sum(dim=0, keepdims=True).expand_as(probability_tmp)
    mask_size = probability_tmp.shape[1]
    top_idx = probability_tmp.topk(K, dim=0)  # K,N_MASK
    prob_topk_sum = top_idx[0].sum(0)  # N_MASK
    average_emb = model.Output_layer.view(V, 1, embdim).expand((V, mask_size, embdim)) * probability_tmp.view(V, mask_size,1).expand((V, mask_size, embdim)) #V,mask_size, dim
    average_emb = torch.stack(
        [average_emb[top_idx[1][:, j], j].sum(dim=0) / prob_topk_sum[j] for j in range(mask_size)])  # N_mask, dim

    return average_emb, top_idx

def Get_model(model_path, is_cuda):
    if is_cuda:
        device = "cuda"
    else:
        device = "cpu"
    if model_path == "camembert-base":
        model = CamembertForMaskedLM.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = model.lm_head.decoder.weight.data
        model.model_name ="roberta"
    elif model_path.startswith("cl-tohoku/bert"):
        tokenizer = BertJapaneseTokenizer.from_pretrained(model_path)
        model = BertForMaskedLM.from_pretrained(model_path)
        model.model_name = "jbert"
        model.to(device)
        model.Output_layer = model.cls.predictions.decoder.weight.data
        model.Output_layer_bias = model.cls.predictions.bias.data
    elif model_path.startswith("princeton-nlp/sup-simcse-bert-large-uncased"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = BertForMaskedLM.from_pretrained(model_path)
        model.model_name = "bert"
        model.to(device)
        model.Output_layer = model.cls.predictions.decoder.weight.data
        model.Output_layer_bias = model.cls.predictions.bias.data
    elif model_path.startswith("dvilares/bertinho-gl-base-cased"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = BertForMaskedLM.from_pretrained(model_path)
        model.model_name = "bert"
        model.to(device)
        model.Output_layer = model.cls.predictions.decoder.weight.data
        model.Output_layer_bias = model.cls.predictions.bias.data
    elif model_path.startswith("dbmdz/bert"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = BertForMaskedLM.from_pretrained(model_path)
        model.model_name = "bert"
        model.to(device)
        model.Output_layer = model.cls.predictions.decoder.weight.data
        model.Output_layer_bias = model.cls.predictions.bias.data
    elif model_path.startswith("microsoft/mpnet"):
        tokenizer = MPNetTokenizer.from_pretrained(model_path)
        model = MPNetForMaskedLM.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = model.lm_head.decoder.weight.data
        model.Output_layer_bias = model.lm_head.bias.data
        model.model_name ="mpnet"
    elif model_path.startswith("sentence-transformers"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = None
        model.Output_layer_bias = None
        # model.Output_layer = model.lm_head.weight.data
        # model.Output_layer_bias = torch.zeros(len(model.Output_layer))
        if model_path=="sentence-transformers/all-roberta-large-v1":
            model.model_name = "sbert-roberta"
        elif model_path in ["sentence-transformers/bert-large-nli-stsb-mean-tokens", "sentence-transformers/bert-large-nli-mean-tokens"]:
            model.model_name = "sbert-bert"
        elif model_path in ["sentence-transformers/all-mpnet-base-v2","sentence-transformers/paraphrase-mpnet-base-v2"]:
            model.model_name = "sbert-mpnet"

    elif model_path.startswith("google/electra") or model_path.startswith("dbmdz/electra"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = ElectraForPreTraining.from_pretrained(model_path)
        model.to(device)
        model.Output_layer_bias = None
        model.Output_layer = None
        model.model_name = "electra"

    elif model_path == 'facebook/m2m100_418M':
        model = M2M100ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = M2M100Tokenizer.from_pretrained(model_path, src_lang="en",
                                                    tgt_lang="en")
        model.eval()
        model.model_name='m2m100_418M'
    elif model_path.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.add_tokens(["<pad>"], special_tokens=True)
        tokenizer.add_tokens(["<mask>"], special_tokens=True)
        tokenizer.pad_token = '<pad>'
        tokenizer.mask_token = '<mask>'
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = model.lm_head.weight.data
        model.Output_layer_bias = torch.zeros(len(model.Output_layer))
        model.model_name = "gpt2"
        tokenizer.vocab = tokenizer.get_vocab()
    elif model_path.startswith("xlnet"):
        tokenizer = XLNetTokenizer.from_pretrained(model_path)
        model = XLNetLMHeadModel.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = model.lm_loss.weight.data
        model.Output_layer_bias = model.lm_loss.bias.data
        model.model_name = "xlnet"
        tokenizer.vocab = tokenizer.get_vocab()
    elif model_path.startswith("t5") or model_path.startswith("google/t5"):
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.mask_token = "<extra_id_0>"
        # mask_token_id is automatically assigned
        tokenizer.mask1_token = "<extra_id_0>"
        tokenizer.mask1_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask1_token)
        tokenizer.mask2_token= "<extra_id_1>"
        tokenizer.mask2_token_id = tokenizer.convert_tokens_to_ids("<extra_id_1>")
        assert tokenizer.mask_token_id == tokenizer.mask1_token_id
        model.to(device)
        model.Output_layer = model.lm_head.weight.data
        model.Output_layer_bias = torch.zeros(len(model.Output_layer))
        model.model_name = "t5"
    elif model_path.startswith('openai-gpt'):
        model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        model.to(device)
        model.Output_layer = model.lm_head.weight.data
        model.Output_layer_bias = torch.zeros(len(model.Output_layer))
        model.model_name = "gpt-1"
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_path)

    elif model_path.startswith("albert"):
        tokenizer = AlbertTokenizer.from_pretrained(model_path)
        model = AlbertForMaskedLM.from_pretrained(model_path)
        model.model_name = "albert"
        model.to(device)
        model.Output_layer = model.predictions.decoder.weight.data
        model.Output_layer_bias = model.predictions.bias.data
        tokenizer.vocab = tokenizer.get_vocab()
    elif model_path.startswith("bert") or model_path.startswith("neuralmind/bert"):
        space_as_token = False
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = BertForMaskedLM.from_pretrained(model_path)
        model.model_name = "bert"
        model.to(device)
        model.Output_layer = model.cls.predictions.decoder.weight.data
        model.Output_layer_bias = model.cls.predictions.bias.data

    elif model_path.startswith("roberta"):
        space_as_token = True
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = RobertaForMaskedLM.from_pretrained(model_path)
        model.model_name = "roberta"
        model.to(device)
        model.Output_layer = model.lm_head.decoder.weight.data
        model.Output_layer_bias = model.lm_head.decoder.bias.data
    elif model_path.startswith("google/pegasus"):
        tokenizer = PegasusTokenizer.from_pretrained(model_path)
        model = PegasusForConditionalGeneration.from_pretrained(model_path)
        model.model_name = "pegasus"
        model.to(device)
        ####The paper says MLM is not used during training###
        # tokenizer.mask_token = tokenizer.mask_token_sent
        # tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        model.Output_layer = model.lm_head.weight.data
        model.Output_layer_bias = torch.zeros(len(model.Output_layer)).to(device)
    elif model_path.startswith("xlm"):
        model = XLMWithLMHeadModel.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = model.pred_layer.proj.weight.data
        model.model_name = "xlm"
    elif model_path =="microsoft/deberta-v3-large":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.model_name = "deberta-v3"
        model.to(device)
        model.Output_layer = None
        model.Output_layer_bias = None
    elif model_path.startswith("microsoft/deberta-v2"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = DebertaV2ForMaskedLM.from_pretrained(model_path)
        model.model_name = "deberta-v2"
        model.to(device)
        model.Output_layer = model.cls.predictions.decoder.weight.data
        model.Output_layer_bias = model.cls.predictions.bias.data
        print(model.Output_layer.sum())
        print(model.Output_layer_bias.sum())
    elif model_path.startswith("microsoft/deberta"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = DebertaForMaskedLM.from_pretrained(model_path)
        model.model_name = "deberta"
        model.to(device)
        model.Output_layer = model.cls.predictions.decoder.weight.data
        model.Output_layer_bias = model.cls.predictions.bias.data
    elif model_path=="Helsinki-NLP/opus-mt-en-de":
        from transformers import MarianTokenizer
        '''        
        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        src_texts = ["I am a small frog.", "Tom asked his teacher for advice."]
        tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
        inputs = tokenizer(src_texts, return_tensors="pt", padding=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(tgt_texts, return_tensors="pt", padding=True)
        inputs["labels"] = labels["input_ids"]
        # keys  [input_ids, attention_mask, labels].        
        outputs = model(**inputs)
        encoder_outputs = model.model.encoder(**inputs,output_hidden_states=True)
        '''
        from transformers import MarianMTModel, MarianTokenizer
        model = MarianMTModel.from_pretrained(model_path)
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = None
        model.Output_layer = None
        model.Output_layer_bias = None
        model.model_name = "marian"
    elif model_path.startswith("facebook/bart"):
        space_as_token = True
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
        model.model_name ="fbbart"
        tokenizer.vocab = tokenizer.get_vocab()
        if model_path == "facebook/bart-base":
            # pass
            bart = torch.hub.load('pytorch/fairseq', 'bart.base')
            tokenid = bart.task.source_dictionary.indices["<mask>"]
            mask_emb = bart.model.encoder.embed_tokens.weight[tokenid].data
            model.model.encoder.embed_tokens.weight[tokenizer.mask_token_id] = mask_emb
        model.to(device)
        model.Output_layer = model.lm_head.weight.data
        model.Output_layer_bias = torch.zeros(len(model.Output_layer)).to(device)
    elif model_path =="facebook/mbart-large-50-one-to-many-mmt":
        tokenizer = MBart50TokenizerFast.from_pretrained(model_path, src_lang="en_XX")
        # tokenizer.src_lang = "en_XX"
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        model.model_name ="fbmbart_MT"
        # tokenizer.tgt_lang_id = tokenizer.convert_tokens_to_ids("en_XX")
        # assert tokenizer.convert_ids_to_tokens(tokenizer.tgt_lang_id)=="en_XX"
        # tokenizer.vocab = tokenizer.get_vocab()
        model.to(device)
        model.Output_layer = None
        model.Output_layer_bias = None
    if model.Output_layer_bias is not None:
        assert len(model.Output_layer_bias.size())==1
        assert len(model.Output_layer.size())==2
        assert len(model.Output_layer)==len(model.Output_layer_bias)
        model.Output_layer_bias = model.Output_layer_bias.view(-1, 1).to(device)
    model.eval()
    if model.model_name == "spanbert":
        pass
    elif model.model_name in ["gpt2", "xlnet","pegasus",'marian','t5','albert']:
        you_id = tokenizer('you', add_special_tokens=False)['input_ids']
        assert len(you_id) == 1
        assert tokenizer.convert_ids_to_tokens(you_id) == tokenizer.tokenize('you')
    else:
        w_id = tokenizer('.')['input_ids'][1:-1]
        assert len(w_id)==1
        assert tokenizer.convert_ids_to_tokens(w_id)==tokenizer.tokenize('.')
    return model, tokenizer

def Calc_InnerProb(model, input_embbeddings, attention_mask, token_type_ids, mask_row_idx, mask_col_idx, MWEidx, mask_token_id, opt):
    batch_size, _, embdim = input_embbeddings.size()
    pred_prob = []
    input_embbeddings_NoMASK = input_embbeddings.clone()
    MWEemb = model.Output_layer[MWEidx].unsqueeze(0).clone()  # 1, n_mask, emb
    maskemb = model.Output_layer[mask_token_id].clone().unsqueeze(0).unsqueeze(0)  # 1, 1, emb
    input_embbeddings_NoMASK[mask_row_idx, mask_col_idx] = MWEemb.expand((batch_size, len(MWEidx), embdim))
    for j in range(len(MWEidx)): #N_mask
        input_embbeddings_1MASK = input_embbeddings_NoMASK.clone()
        input_embbeddings_1MASK[mask_row_idx, mask_col_idx[:, j:j+1]] = maskemb.expand((batch_size, 1, embdim))
        _, _, _,_, prediction_scores = Encode_BERT(None, model, None,input_embbeddings_1MASK, attention_mask,
                                                 token_type_ids, mask_row_idx, mask_col_idx,
                                                  opt.remove_bias)
        probability = F.softmax(prediction_scores / opt.tau, dim=0) # V, bs, N_mask
        probability_mean = Mean_Probability(probability, opt.meantype)

        prob = probability_mean.data.cpu().numpy()[MWEidx[j],j]
        pred_prob.append(prob)
    pred_prob = np.mean(pred_prob)
    return pred_prob

def Calc_OuterProb(model, input_embbeddings, attention_mask, token_type_ids,
                   mask_row_idx, mask_col_idx, MWEidx, mask_token_id,
                   prepost_mask_idx, prepost_goldidx, prepost_padding_idx, remove_bias,
                   tau, SBO):
    #prepost_goldidx: bs, N_MASK(OUT)
    batch_size, sent_len, embdim = input_embbeddings.size()
    MWEemb = model.Output_layer[MWEidx].unsqueeze(0).clone()  # 1, n_mask, emb
    maskemb = model.Output_layer[mask_token_id].unsqueeze(0).unsqueeze(0).clone()  # 1, 1, emb
    pred_prob_list = []
    prob_all = []
    for i, prepost_mask_idx_tmp in enumerate(prepost_mask_idx): #for each window (bs, n_mask)
        #prepost_mask_idx_tmp: bs, N_mask
        N_mask = prepost_mask_idx_tmp.shape[-1] #N_mask
        input_embbeddings_prepostMASK = input_embbeddings.clone()  # bs, N_len, dim
        input_embbeddings_prepostMASK[mask_row_idx, prepost_mask_idx_tmp] = maskemb.expand((batch_size, N_mask, embdim))
        if mask_col_idx is not None:
            input_embbeddings_prepostMASK[mask_row_idx, mask_col_idx] = MWEemb.expand((batch_size, len(MWEidx), embdim))
        MWE_all_states, last_layer, _, _, prediction_scores = Encode_BERT(None,model,None ,input_embbeddings_prepostMASK, attention_mask,
                                                                            token_type_ids, mask_row_idx, prepost_mask_idx_tmp,
                                                                            remove_bias)
        if SBO:
            pre = prepost_mask_idx_tmp[:, 0] - 1 #bs,
            post = prepost_mask_idx_tmp[:, -1] + 1
            pairs_idx = np.stack([pre, post], axis=-1)
            _, _, prediction_scores = SpanBERT_SBO(model, last_layer, pairs_idx, N_mask, remove_bias, i)

        probability = F.softmax(prediction_scores/ tau, dim=0) # V, bs, N_mask
        #probability_mean = Mean_Probability(probability, opt.meantype)
        # probability: V, bs, N_prepostMASK
        # prepost_goldidx: bs,  N_prepostMASK
        # padding_idx: bs,  N_prepostMASK
        # bs = len(prepost_goldidx)
        prepost_goldidx_tmp = prepost_goldidx[i].reshape(-1) # bs*N_prepostMASK
        padding_idx =  prepost_padding_idx[i].reshape(-1) # bs*N_prepostMASK
        probability = probability.view(len(probability), -1) #V, bs* N_prepostMASK
        alpha = 0.99
        uniform = 1 / len(probability)  # add a uniform distribution
        probability = alpha * probability + (1 - alpha) * uniform
        pred_prob = [probability[prepost_goldidx_tmp[k],k]
                     for k in range(len(prepost_goldidx_tmp)) if padding_idx[k]==1]
        pred_prob_list.append(pred_prob) #
        prob_all.extend(pred_prob) #
        #TODO: pred_prob = pred_prob.mean() Or, pred_prob = torch.exp(torch.log(pred_prob))  # geomean
    log_prob_all = (torch.log(torch.stack(prob_all)).mean()).data.cpu().numpy()
    return pred_prob_list, log_prob_all

def Prepare_BERT_input(tokenised_sent_id, tokenizer,is_cuda):
    input_ids = []
    attention_mask = []
    max_len = max([len(x) for x in tokenised_sent_id])
    for sent in tokenised_sent_id:
        attention_mask.append([1] * len(sent) + [0] * (max_len - len(sent)))
        input_ids.append(tokenizer.convert_tokens_to_ids(sent) + [tokenizer.pad_token_id] * (max_len - len(sent)))
    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    token_type_ids = torch.LongTensor([0] * (len(input_ids) * max_len)).view(len(input_ids), max_len)
    if is_cuda:
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        token_type_ids = token_type_ids.to("cuda")
    return input_ids, attention_mask, token_type_ids


def BERT_2MASK_BS(model, candidate_idx, input_embbeddings, attention_mask,
                     token_type_ids, mask_row_idx,
                     mask_col_idx, tokenizer,
                     prepost_mask_idx, prepost_goldidx, prepost_padding_idx, opt):
    #top_idx: 2, topk, n_mask
    batch_size, _, embdim = input_embbeddings.size()
    best_inner_prob = -1
    best_outer_prob = -1
    best_MWE = None
    candidates2score = dict()
    for j in range(2):  # [MASK] [MASK]
        for i in range(len(candidate_idx[j])):
            idx = candidate_idx[j][i]
            input_embbeddings_1MASK = input_embbeddings.clone()
            MWEemb = model.Output_layer[idx].unsqueeze(0).unsqueeze(0)  # 1, 1, emb
            input_embbeddings_1MASK[mask_row_idx, mask_col_idx[:, j:j + 1]] = MWEemb.expand((batch_size, 1, embdim))
            meantype = opt.meantype
            _, _, _, _,prediction_scores = Encode_BERT(tokenizer,model, None,input_embbeddings_1MASK, attention_mask,
                                                    token_type_ids, mask_row_idx, mask_col_idx,
                                                    opt.remove_bias)
            probability = F.softmax(prediction_scores / opt.tau, dim=0)  # V, bs, N_mask
            probability_mean = Mean_Probability(probability, opt.meantype)

            #probability_mean: V, n_mask
            if j==0:
                pred_idx = probability_mean[:, 1].argmax()
                MWEidx = torch.stack([idx, pred_idx])
            else:
                pred_idx = probability_mean[:, 0].argmax()
                MWEidx = torch.stack([pred_idx, idx])
            if opt.pred_subword:
                if tuple(MWEidx.data.cpu().tolist()) not in tokenizer.rareword_ids: #not a word
                    continue
            if tuple(MWEidx.data.cpu().tolist()) not in candidates2score:
                pred_prob_list, outer_prob = Calc_OuterProb(model, input_embbeddings, attention_mask, token_type_ids,
                                            mask_row_idx, mask_col_idx,
                                            MWEidx, tokenizer.mask_token_id, prepost_mask_idx, prepost_goldidx,prepost_padding_idx,
                                            opt.remove_bias, opt.tau, opt.SBO)
                candidates2score[tuple(MWEidx.data.cpu().tolist())] = outer_prob
                #MWE_candidates.add(tuple(MWEidx))
                if best_outer_prob < outer_prob:
                    best_MWE = MWEidx
                    best_outer_prob = outer_prob
                    # best_inner_prob = Calc_InnerProb(input_embbeddings, attention_mask, token_type_ids, Output_layer,
                    #                                  mask_row_idx,
                    #                                  mask_col_idx, best_MWE, mask_token_id, opt)
    if best_MWE is None:
        return None, np.array(0), dict()
    else:
        return best_MWE.cpu().numpy(), best_outer_prob, candidates2score

def BERT_inner_2MASK(model, candidate_idx, input_embbeddings, attention_mask,
                     token_type_ids, mask_row_idx,
                     mask_col_idx, tokenizer,
                     prepost_mask_idx, prepost_goldidx, prepost_padding_idx, opt):
    #top_idx: 2, topk, n_mask
    batch_size, _, embdim = input_embbeddings.size()
    best_inner_prob = -1
    best_outer_prob = -1
    best_MWE = None
    candidates2score = dict()
    for j in range(2):  # [MASK] [MASK]
        for i in range(len(candidate_idx[j])):
            idx = candidate_idx[j][i]
            input_embbeddings_1MASK = input_embbeddings.clone()
            MWEemb = model.Output_layer[idx].unsqueeze(0).unsqueeze(0)  # 1, 1, emb
            input_embbeddings_1MASK[mask_row_idx, mask_col_idx[:, j:j + 1]] = MWEemb.expand((batch_size, 1, embdim))
            meantype = opt.meantype
            _, _, _, _,prediction_scores = Encode_BERT(tokenizer, model, None,input_embbeddings_1MASK, attention_mask,
                                                    token_type_ids, mask_row_idx, mask_col_idx,
                                                    opt.remove_bias)
            probability = F.softmax(prediction_scores / opt.tau, dim=0)  # V, bs, N_mask
            probability_mean = Mean_Probability(probability, opt.meantype)

            #probability_mean: V, n_mask
            if j==0:
                pred_idx = probability_mean[:, 1].argmax()
                MWEidx = torch.stack([idx, pred_idx])
            else:
                pred_idx = probability_mean[:, 0].argmax()
                MWEidx = torch.stack([pred_idx, idx])
            if opt.pred_subword:
                if tuple(MWEidx.data.cpu().tolist()) not in tokenizer.rareword_ids: #not a word
                    continue
            if tuple(MWEidx.data.cpu().tolist()) not in candidates2score:
                pred_prob_list, outer_prob = Calc_OuterProb(model, input_embbeddings, attention_mask, token_type_ids,
                                            mask_row_idx, mask_col_idx,
                                            MWEidx, tokenizer.mask_token_id, prepost_mask_idx, prepost_goldidx,prepost_padding_idx,
                                            opt.remove_bias, opt.tau, opt.SBO)
                candidates2score[tuple(MWEidx.data.cpu().tolist())] = outer_prob
                #MWE_candidates.add(tuple(MWEidx))
                if best_outer_prob < outer_prob:
                    best_MWE = MWEidx
                    best_outer_prob = outer_prob
                    # best_inner_prob = Calc_InnerProb(input_embbeddings, attention_mask, token_type_ids, Output_layer,
                    #                                  mask_row_idx,
                    #                                  mask_col_idx, best_MWE, mask_token_id, opt)
    if best_MWE is None:
        return None, np.array(0), dict()
    else:
        return best_MWE.cpu().numpy(), best_outer_prob, candidates2score


def BERT_1MASK_BS(model, candidate_idx, input_embbeddings, attention_mask,
                     token_type_ids, mask_row_idx,
                     mask_col_idx, tokenizer,
                     prepost_mask_idx, prepost_goldidx, prepost_padding_idx, opt):
    #top_idx: 2, topk, n_mask (idx)
    MWE_candidates = set([])
    #inner_prob = top_idx[0][:,0].cpu().numpy()  #k
    best_prob_list = None
    best_outer_prob = -1
    best_MWE = None
    candidates2score = dict()
    candidate_idx  = candidate_idx[0] #1, beam_size -> beam_size
    for i in range(len(candidate_idx)):
        pred_prob_list, outer_prob = Calc_OuterProb(model,input_embbeddings, attention_mask, token_type_ids,                                     mask_row_idx,
                                    mask_col_idx, [candidate_idx[i]], tokenizer.mask_token_id,
                                    prepost_mask_idx,
                                    prepost_goldidx, prepost_padding_idx,
                                    opt.remove_bias, opt.tau, opt.SBO)
        candidates2score[tuple([candidate_idx[i].item()])] = outer_prob
        # if best_prob_list is None:
        #     best_prob_list = pred_prob_list
        #     best_MWE = candidate_idx[i].unsqueeze(0)  # 1, 1
        #     best_outer_prob = outer_prob
        # elif len(pred_prob_list)//2 < torch.sum(best_prob_list < pred_prob_list):
        #     best_prob_list = pred_prob_list
        #     best_MWE = candidate_idx[i].unsqueeze(0)  # 1, 1
        #     best_outer_prob = outer_prob
        #
        if best_outer_prob < outer_prob:
            best_outer_prob = outer_prob
            best_MWE =  candidate_idx[i].unsqueeze(0) #1, 1

    return best_MWE.cpu().numpy(), best_outer_prob, candidates2score

def BERT_Beam_Search(model, candidate_idx, input_embbeddings, attention_mask,
                     token_type_ids, mask_row_idx,
                     mask_col_idx, tokenizer ,
                     prepost_mask_idx, prepost_goldidx, prepost_padding_idx, opt):

    n_mask = len(candidate_idx)
    # best_MWE = None
    # best_inner_prob = -1
    # best_outer_prob = -1
    if n_mask==1:
        best_MWE_tmp, best_outer_prob_tmp, candidates2score =\
        BERT_1MASK_BS(model, candidate_idx, input_embbeddings, attention_mask,
                     token_type_ids, mask_row_idx,
                     mask_col_idx, tokenizer,
                     prepost_mask_idx, prepost_goldidx,prepost_padding_idx,  opt)

    elif n_mask==2:
            best_MWE_tmp, best_outer_prob_tmp, candidates2score =\
        BERT_2MASK_BS(model, candidate_idx, input_embbeddings, attention_mask,
                     token_type_ids, mask_row_idx,
                     mask_col_idx, tokenizer,
                     prepost_mask_idx, prepost_goldidx, prepost_padding_idx, opt)
    else:
        raise Exception
    if best_MWE_tmp is None:
        return None, np.array(0), np.array(0), dict()
    else:
        best_inner_prob_tmp = Calc_InnerProb(model, input_embbeddings, attention_mask, token_type_ids,
                                             mask_row_idx, mask_col_idx,
                                             best_MWE_tmp, tokenizer.mask_token_id, opt)
        return best_MWE_tmp, best_inner_prob_tmp, best_outer_prob_tmp, candidates2score

def Mix_sentences(silver_sent):
    with open(silver_sent[0], 'rb') as f:
        phrase2sent_1B_X = pickle.load(f)
    with open(silver_sent[1], 'rb') as f:
        phrase2sent_1B_Y = pickle.load(f)
    phrase2sent_1B = {}
    for MWE in phrase2sent_1B_X.keys():
        phrase2sent_1B[MWE] = []
        assert len(phrase2sent_1B_X[MWE]) == len(phrase2sent_1B_Y[MWE])
        for nth_gold in range(len(phrase2sent_1B_X[MWE])):
            assert phrase2sent_1B_X[MWE][nth_gold][0] == phrase2sent_1B_Y[MWE][nth_gold][0]
            phrase2sent_1B[MWE].append(phrase2sent_1B_X[MWE][nth_gold][:26] + phrase2sent_1B_Y[MWE][nth_gold][26:])

    return phrase2sent_1B