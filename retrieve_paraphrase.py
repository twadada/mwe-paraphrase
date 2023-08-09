import os
import string
import numpy as np
import pickle
import torch.nn.functional  as F
import argparse
from utils import Encode_BERT_PAD, Get_model, Identify_Key_Indices

def Read_tgtsent(target_sent, lowercase, model):
    target_sentences=[]
    for line in open(target_sent, encoding="utf8"):
        if lowercase:
            line = line.lower()
        line = line.rstrip("\n").split("\t")  # MWE, gold, silver*N
        assert len(line)>=4
        _ = int(line[1])
        if model.model_name == 'gpt2':
            for i in range(1,len(line)):
                line[2] = "<|endoftext|>"+line[2]
        target_sentences.append(line)
    return target_sentences
#
# def Read_vecfiles_multi(vec_folders, vec_type, word_list = None, N_words = -1):
#     # emb_list: L, K_vec
#     Vocab = None
#     emb_list = []
#     for i in range(len(vec_folders)):
#         emb = []
#         Vocab_tmp = []
#         folder = vec_folders[i] #bert-large-uncased_k4/K4/K0
#         file = folder+"/"+vec_type + "0.txt" #bert-large-uncased_k4/K4/K0/vec${l}.txt
#         vec_tmp = load_w2v(file, word_list, N_words)
#         for w in vec_tmp.keys():
#             Vocab_tmp.append(w)
#             emb.append(vec_tmp[w])
#         if Vocab is None:
#             Vocab = np.array(Vocab_tmp)
#         else:
#             assert all(Vocab == np.array(Vocab_tmp))
#         emb_list.append(emb) #K,V, dim
#     word2id = {}
#     for w in Vocab:
#         word2id[w] = len(word2id)
#     print("V: ",len(Vocab))
#     return Vocab, word2id, emb_list

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model',
        help='model')
    parser.add_argument(
        '-target_sent',
        required=True,
        help='target_sent_path')
    parser.add_argument(
        '-n_mask',
        type=int,
        default=1,
        help='N_mask')
    parser.add_argument(
        '-threshold',
        type=float,
        help='N_mask')
    parser.add_argument(
        '-mask_prob',
        help='N_mask')
    parser.add_argument(
        '-pre_mask_token',
        default="<mask>",
        type= str,
        help='N_mask')
    parser.add_argument(
        '-tgt_layer',
        type=int,
        default=None,
        help='N_mask')
    parser.add_argument(
        '-out_cossim',
        action='store_true',
        help='N_mask')
    parser.add_argument(
        '-V_size',
        type=int,
        help='V')
    parser.add_argument(
        '-save',
        help='V')
    parser.add_argument(
        '-folder',
        help='save_name')
    parser.add_argument(
        '-debug',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-rand_j',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-rand_k',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-use_points',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-verbose',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-back_trans',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-pred_subword',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-lowercase',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-pred_MWE',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-remove_bias',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-lev',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-count',  ##there are some issures about subwords
        type=int,
        default=-1,
        help='save_name'
    )
    parser.add_argument(
        '-MWE_para',  ##there are some issures about subwords
        nargs='+',
        help='save_name'
    )
    parser.add_argument(
        '-vec',
        type=str,
        help='save_name')
    parser.add_argument(
        '-vec_MWE',
        type=str,
        default=[],
        nargs='+',
        help='save_name')
    parser.add_argument(
        '-Lvec',
        type=str,
        default=[],
        nargs='+',
        help='save_name')

    parser.add_argument(
        '-Rvec',
        type=str,
        default=[],
        nargs='+',
        help='save_name')


    opt = parser.parse_args()

    if opt.vec:
        labels = []
        word2vec = {}
        for line in open(opt.vec,encoding="utf8"):
            line = line.split(" ||| ")
            vector_list=[]
            word = None
            print(len(line))
            for k in range(len(line)):
                if k==0:
                    line_tmp = line[k].split() #word vec
                    word = line_tmp[0]
                    vectors = line_tmp[1:]
                else:
                    vectors = line[k].split() #vec
                if len(vectors)!=0:
                    if len(vectors) in [1024*2,768*2]:
                        vectors = [float(vectors[2*j].split("tensor(")[1].split(",")[0]) for j in range(len(vectors)//2)]
                    elif len(vectors) in [1024,768,128]:
                        vectors = [float(vectors[j]) for j in range(len(vectors))]
                    else:
                        print(len(vectors))
                        raise Exception
                    vector_list.append(np.array(vectors)/np.linalg.norm(vectors))
            assert word is not None
            word not in word2vec
            word2vec[word] = np.array(vector_list)

    folder = None
    model_path = opt.model
    folder = opt.folder
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        print("Directory ", folder, " already exists")
    model, tokenizer = Get_model(model_path, torch.cuda.is_available())

    for i in range(len(opt.MWE_para[-4:])):
        assert opt.MWE_para[i][-4:]==".pkl"

    if len(opt.MWE_para[-4:])==1:
        with open(opt.MWE_para[0], 'rb') as f:
            MWE2cand_loss = pickle.load(f)
    else:
        with open(opt.MWE_para[0], 'rb') as f:
            MWE2cand_loss1 = pickle.load(f)
        with open(opt.MWE_para[1], 'rb') as f:
            MWE2cand_loss2 = pickle.load(f)
        keywords = set(MWE2cand_loss1.keys())
        assert keywords == set(MWE2cand_loss2.keys())
        MWE2cand_loss= {}
        N_cand_max = 10
        for w in keywords:
            cand1 = MWE2cand_loss1[w]
            cand2 = MWE2cand_loss2[w]
            assert len(cand1)==len(cand2) #custer size
            MWE2cand_loss[w] = []
            for k_class in range(len(cand1)):
                candList = cand1[k_class][:N_cand_max] + cand2[k_class][:N_cand_max]
                MWE2cand_loss[w].append(candList)
    if opt.mask_prob:
        with open(opt.mask_prob, 'rb') as f:
            mask_probDict = pickle.load(f)

    non_alnum_vocab = []
    if model.model_name in ["bert",'jbert']:
        BERT_tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer.vocab))))
        for i, w in enumerate(BERT_tokens):
            if not w.isalnum() and model.model_name!='jbert':
                non_alnum_vocab.append(i)
            elif model_path == "dbmdz/bert-base-italian-xxl-uncased" and w.startswith("unused"):
                print(w)
                non_alnum_vocab.append(i)
    # if opt.vec_MWE:
    #     MWE_Vocab, MWE2id, MWEemb_list = Read_vecfiles_multi(opt.vec_MWE,  "vec", None, opt.V_size)
    #     MWEemb_list = torch.FloatTensor(np.array(MWEemb_list)).to(model.device)  # K, V, dim
    candidate_output = []
    # stop_word_ids = Get_stopword_ids(tokenizer)
    punctuation_ids = set([])
    punctuations = list(string.punctuation + "“" + "”" + "-" + "’" + "‘" + "…")
    punctuations = punctuations + ["▁" + x for x in punctuations]
    for w in punctuations:
        if w in tokenizer.vocab:
            w_id = tokenizer.convert_tokens_to_ids([w])[0]
            punctuation_ids.add(w_id)
        else:
            print(w)
    punctuation_ids.update(tokenizer.all_special_ids)
    stop_word_ids = list(punctuation_ids)
    target_sentences = Read_tgtsent(opt.target_sent, opt.lowercase, model)
    count = -1
    MWE_cluster_idxList = []
    with torch.no_grad():
        for s_id, line in enumerate(target_sentences):
            count+=1
            if count == opt.count:
                break
            phrase = line[0]
            MWE_index = int(line[1])
            if model.model_name in ['t5'] and line[2].startswith("<mask> is less dense than salt water"):
                MWE_index = 0
            phrase = phrase.replace("-"," ")
            phrase_joined = "▁".join(phrase.lstrip(" ").split())
            sentences_raw = [line[2]]
            # assert len(sentences_raw)==1
            if model.model_name in ["t5","mt5"]:#'<mask>' and ' <mask>' both become '▁', '<unk>', 'mas', 'k', '>', '</s>'
                assert opt.n_mask == 1
                sentences_raw = [x.replace("<mask>", "<extra_id_0>") for x in sentences_raw]
                if " <extra_id_0>" in sentences_raw[0]:
                    token = " <extra_id_0>"
                else:
                    token = "<extra_id_0>"
            else:
                if " <mask>" in sentences_raw[0]:
                    token = " <mask>"
                else:
                    token = "<mask>"
            sentences = tokenizer(sentences_raw)["input_ids"]
            mask_ids = tokenizer(token, add_special_tokens=False)["input_ids"]  # remove CLS/SEP

            if len(sentences[0]) > 512:
                print(sentences[0])
                raise Exception

            phrase_tokenised_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(phrase)) #remove CLS/SEP
            phrase_tokenised_str = tokenizer.convert_ids_to_tokens(phrase_tokenised_ids)

            count+=1
            paraphrases = []
            assert opt.n_mask >0
            if opt.n_mask == -1: #Adaptive or zero masking
                n_mask = len(phrase_tokenised_ids) #
            else:
                n_mask = opt.n_mask
            mask_ids_replace = [tokenizer.mask_token_id for _ in range(n_mask)]
            sentences_masked, mask_row_idx, mask_col_idx,\
            around_idx_list, gold_ids_list, padding_list =\
                Identify_Key_Indices(sentences, mask_ids, mask_ids_replace, 0, tokenizer, stop_word_ids, "span_two", [MWE_index])
            gold_sent = " ".join(tokenizer.convert_ids_to_tokens(sentences_masked[0]))
            # input_ids, attention_mask, token_type_ids = Pad_sentence_ids(sentences_masked, tokenizer.pad_token_id,torch.cuda.is_available())
            batch_size = len(sentences_masked)
            dictkey = phrase.lstrip().rstrip()
            if dictkey in MWE2cand_loss:
                phrase2cand_loss = MWE2cand_loss[dictkey]
                if opt.mask_prob:
                    phrase2mask_prob = mask_probDict[dictkey]
            elif dictkey.lower() in MWE2cand_loss:
                phrase2cand_loss = MWE2cand_loss[dictkey.lower()]
            elif dictkey[:-1] in MWE2cand_loss: #bad apples -> bad apple
                phrase2cand_loss = MWE2cand_loss[dictkey[:-1]]
            else:
                print(dictkey)
                print(MWE2cand_loss.keys())
                raise Exception

            _, MWE_ffn, _,_,_ =\
                Encode_BERT_PAD(tokenizer, model, sentences_masked, mask_col_idx,
                                max_tokens=8192, layers = [1])
            if model.model_name in ['bert','xlnet','albert']:
                # MWE_ffn :         bs, n_mask,      dim
                prob = model.Output_layer.mm(
                    MWE_ffn.view(-1, model.Output_layer.size()[-1]).T)  # V, n_mask
                prob = prob + model.Output_layer_bias.expand_as(prob)  # V, n_mask
                prob = F.softmax(prob, dim=0)  # V, n_mask
            elif model.model_name in ['t5','mt5']:
                sentences_masked = torch.LongTensor(sentences_masked).to(model.device)

            dictkey = "▁".join(phrase.lstrip(" ").split(" "))
            if len(phrase2cand_loss)==1: #one cluster
                MWE_cluster_idx = 0
            else: #retrieve the closest cluster
                if dictkey in word2vec:
                    MASK_vectors = word2vec[dictkey]  # K, vec
                elif dictkey.lower() in word2vec:
                    MASK_vectors = word2vec[dictkey.lower()]  # K, vec
                elif dictkey[:-1] in word2vec:  # bad apples -> bad apple
                    MASK_vectors = word2vec[dictkey[:-1]]  # K, vec
                if "dbscan" in opt.vec and len(MASK_vectors) > 1:
                    print("remove noise cluster")
                    MASK_vectors = MASK_vectors[1:].copy()
                assert len(MASK_vectors) == len(phrase2cand_loss), str(len(MASK_vectors)) + "_" + str(
                    len(phrase2cand_loss))
                MWE_ffn_normalised = F.normalize(MWE_ffn.mean(1), dim=-1).repeat(len(MASK_vectors),1).data.cpu().numpy() #K, dim
                cossim = np.sum(MASK_vectors * MWE_ffn_normalised, axis= -1) #k
                MWE_cluster_idx = np.argmax(cossim)
                print(MWE_cluster_idx)
            MWE_cluster_idxList.append(MWE_cluster_idx)
            candList = phrase2cand_loss[MWE_cluster_idx]
            if opt.verbose:
                print("MWE_cluster_idx", MWE_cluster_idx)
                for K_size in range(len(phrase2cand_loss)):
                    print([x[1] for x in phrase2cand_loss[K_size][:5]])
                print("&&&&&&&&&")
                print([x[1] for x in candList])
            # MWE_ffn: 1, n_mask, dim
            len2words = {}
            cand2score_sample_dict = {}
            cand2innerscore_dict = {}
            for cand_ids_str_score in candList:
                cand_ids =  cand_ids_str_score[0] #n_words
                cand_str =  cand_ids_str_score[1] #n_words
                scores  =  cand_ids_str_score[2:] #score ([outer, inner])
                if cand_str in cand2score_sample_dict: #differnt ids, but same word (with different subwords)
                    prev_score = cand2score_sample_dict[cand_str][0]
                    if scores[0] > prev_score[0]: #compare the first score
                        cand2score_sample_dict[cand_str] = [scores, cand_ids]
                else:
                    cand2score_sample_dict[cand_str] = [scores, cand_ids]
                if len(cand_ids) not in len2words:
                    len2words[len(cand_ids)] = [(cand_ids, cand_str)]
                else:
                    len2words[len(cand_ids)].append((cand_ids, cand_str))

            candidate_output.append([cand2score_sample_dict])

    if not opt.debug:
        with open(opt.save + "_" + opt.target_sent.split("/")[-1] + ".pkl", 'wb') as f:
            pickle.dump(candidate_output, f) # N_idiom, N_sample, s_len, dim

        with open(opt.save + "_" + opt.target_sent.split("/")[-1] + "_clusters.txt", "w") as f:
            for i in MWE_cluster_idxList:
                f.write(str(i) + "\n")

