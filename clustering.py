from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture

from sklearn.decomposition import PCA

import time
import torch
from sklearn.cluster import KMeans
from collections import Counter
import os
import numpy as np
import pickle
import torch.nn.functional  as F
import argparse
import sys
sys.path.insert(1, '../')
from utils import tokenise_phrase, Encode_BERT_PAD,  Get_model,Identify_Key_Indices
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model',
        help='model')
    parser.add_argument(
        '-silver_sent',
        required=True)
    parser.add_argument(
        '-n_mask',
        type=int,
        default=1)
    parser.add_argument(
        '-N_sample',
        type=int,
        default=1000)
    parser.add_argument(
        '-window',
        type=int,
        default=0)
    parser.add_argument(
        '-weight',
        type=float,
        default=0)
    parser.add_argument(
        '-lowercase',
        action='store_true')

    parser.add_argument(
        '-mask_emb',
        action='store_true')
    parser.add_argument(
        '-mean_first',
        action='store_true')

    parser.add_argument(
        '-POS',
        action='store_true')

    parser.add_argument(
        '-plot',
        action='store_true')

    parser.add_argument(
        '-save',
        help='save_name')
    parser.add_argument(
        '-folder',
        help='save_name')
    parser.add_argument(
        '-debug',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-start_from',
        type= int,
        default=-1,
        help='save_name')
    parser.add_argument(
        '-end_at',
        type= int,
        default=-1,
        help='save_name')
    parser.add_argument(
        '-skip_words',
        type=str,
        default=None,
        help='save_name')
    parser.add_argument(
        '-clustering',
        help='save_name')
    parser.add_argument(
        '-max_tokens',
        type=int,
        default=8192,
        help='save_name')
    parser.add_argument(
        '-MWE_list',
        default=None,
        help='save_name')

    opt = parser.parse_args()
    folder = None
    model_path = opt.model
    folder = opt.folder
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        print("Directory ", folder, " already exists")
    model, tokenizer = Get_model(model_path, torch.cuda.is_available())
    stop_words_ids = []
    Word_embs = model.Output_layer
    target_sentences = []
    embeddings = []
    n_mask = opt.n_mask
    assert n_mask == 1
    if n_mask>=1:
        print(str(n_mask)+"-MASKED")
    else:
        print("NON-MASKED")
    input_ids = torch.cuda.LongTensor([[1, 2]])
    skip_words = []
    vocab = dict()
    count = -1

    colorlist = ["r", "b", "k", "m", "y", "g","c","w"]
    with open(opt.silver_sent, 'rb') as f:
        phrase2sent_1B = pickle.load(f)

    v = list(phrase2sent_1B.keys())

    if opt.MWE_list is not None:
        import os.path
        if os.path.isfile(opt.MWE_list):
            MWE_list = []
            for line in open(opt.MWE_list):
                line = line.rstrip('\n')
                if line not in MWE_list:
                    MWE_list.append(line)
        else:
            MWE_list = opt.MWE_list.split("|")
        N_NOT_found = 0
        v_found = []
        for w in MWE_list:
            if w in v:
                v_found.append(w)
            else:
                # print(w)
                N_NOT_found+=1
                if w.lower() not in MWE_list:
                    print(w +" NOT found")
        v = v_found
        print("N_NOT_found",N_NOT_found)

    # print("N_MWE", len(v))
    V_size = len(v)
    start = time.time()

    if 'dbscan' in opt.clustering or 'xmeans' in opt.clustering:
        K_list = [1]
        if 'dbscan' in opt.clustering:
            elem = opt.clustering.split("_")
            assert len(elem)== 3
            assert elem[0] in ['dbscan']
            _ = float(elem[1]) #assert float val
            val = float(elem[2]) #assert int val
            assert val<1
    else:
        if opt.clustering in ["kmeans2","kmeans3","kmeans4"]:
            K_list = [int(opt.clustering[-1])]
        else:
            assert opt.clustering =="none"
            K_list = [1]

    sent_clusteredListList = [{} for _ in range(len(K_list))]
    with open(opt.folder + "/count.txt", "w") as f_count:
        with open(opt.folder + "/vec.txt", "w") as f_vec:
            with torch.no_grad():
                for i in tqdm(range(len(v))):
                    phrase = v[i]
                    # print(phrase)
                    assert phrase[0] != " "
                    count+=1
                    sentences_raw = phrase2sent_1B[phrase]
                    assert isinstance(sentences_raw, list)
                    if len(sentences_raw)>=1:
                        sentences_raw = sentences_raw[:opt.N_sample]
                        if model.model_name == 'gpt2':
                            for j in range(len(sentences_raw)):
                                sentences_raw[j] = "<|endoftext|>" + sentences_raw[j]
                        else:
                            sentences = tokenizer(sentences_raw)["input_ids"]
                        for x in sentences:
                            assert len(x) < 512 #max token length
                        phrase = " " + phrase #add space
                        if model_path.startswith("cl-tohoku/bert"):
                            #phrase: 年 ##明け (pre-tokenised)
                            phrase_tokenised_ids = tokenizer.convert_tokens_to_ids(phrase.lstrip().split())
                        else:
                            phrase_tokenised_ids = tokenise_phrase(model, tokenizer, phrase)

                        paraphrases = []
                        mask_ids = phrase_tokenised_ids
                        if opt.n_mask == 0:
                            n_mask = len(phrase_tokenised_ids)
                            mask_ids_replace = phrase_tokenised_ids
                        else: #not mask
                            if opt.n_mask == -1: #Adaptive or zero masking
                                n_mask  = len(phrase_tokenised_ids) #
                            mask_ids_replace = [tokenizer.mask_token_id for _ in range(n_mask)]

                        sentences_masked, mask_row_idx, mask_col_idx,\
                        _, _, _ =\
                            Identify_Key_Indices(sentences, mask_ids, mask_ids_replace, opt.window, tokenizer, stop_words_ids, "span_two", None)
                        gold_sent = " ".join(tokenizer.convert_ids_to_tokens(sentences_masked[0]))
                        if opt.clustering == "none" and len(sentences_masked)>4:
                            sentences_raw_tmp = [sentences_raw[idx] for idx in range(len(sentences_raw))]
                            sent_clusteredListList[0][phrase.lstrip(" ")] = [sentences_raw_tmp] #1, N
                        elif len(sentences_masked)>4:
                            _, MWE_ffn, around_states, _, _ = Encode_BERT_PAD(
                                tokenizer, model, sentences_masked, mask_col_idx, max_tokens= opt.max_tokens, layers = [1])
                            veckey = "▁".join(phrase.lstrip(" ").split(" "))
                            f_vec.write(veckey + " ")
                            prediction_scores = model.Output_layer.mm(
                                MWE_ffn.view(-1, model.Output_layer.size()[-1]).T)  # V, bs*n_mask
                            prediction_scores = prediction_scores + model.Output_layer_bias.expand_as(
                                prediction_scores)  # V, bs*N_mask
                            prediction_scores = prediction_scores.view(len(model.Output_layer), -1,MWE_ffn.size()[1])  # V, bs, n_mask
                            probability = F.softmax((prediction_scores), dim=0)  # V, bs, n_mask
                            top_prob_words = probability.topk(10,dim=0)  # Ks_cluster, bs, n_mask
                            top_prob = top_prob_words[0].data.cpu().numpy()
                            top_words = top_prob_words[1].data.cpu().numpy()  # Ks_cluster, bs, n_mask
                            BS_MWE_states = F.normalize(MWE_ffn.mean(1), dim=-1)  # bs, dim

                            assert len(BS_MWE_states.shape)==2
                            BS_MWE_states = BS_MWE_states.data.cpu().numpy()
                            if opt.plot:
                                pca = PCA()
                                new_values = BS_MWE_states
                                # new_values = pca.fit_transform(BS_MWE_states)
                                # new_values = new_values[:,:50]
                                tsne_model = TSNE(perplexity=30, n_components=2, init='pca', random_state=10)
                                new_values = tsne_model.fit_transform(new_values) #N, 2
                                # new_values = tsne_model.fit_transform(BS_MWE_states.data.cpu().numpy()) #N, 2
                                x_val = []
                                y_val = []
                                for value in new_values:
                                    x_val.append(value[0])
                                    y_val.append(value[1])

                            if 'dbscan' in opt.clustering or 'xmeans' in opt.clustering:
                                #adaptive K
                                if 'dbscan' in opt.clustering:
                                    ###DBSCAN###
                                    vals = opt.clustering.split("_")
                                    assert vals[0] =='dbscan'
                                    param1 = float(vals[1])
                                    param2 = float(vals[2])
                                    min_samples = max(3, round(len(BS_MWE_states) * param2))  # alpha
                                    assert min_samples<len(BS_MWE_states)
                                    MWE_class = cluster.DBSCAN(eps = param1, min_samples = min_samples, metric = 'cosine').fit_predict(BS_MWE_states)
                                    # MWE_class = cluster.DBSCAN(eps = 0.1, min_samples = min_samples, metric='cosine').fit_predict(BS_MWE_states)
                                    if -1 in MWE_class: #if there is a noise cluster
                                        MWE_class += 1
                                else:##X-MEANS###
                                    amount_initial_centers = 1
                                    initial_centers = kmeans_plusplus_initializer(BS_MWE_states, amount_initial_centers).initialize()
                                    xmeans_instance = xmeans(BS_MWE_states, initial_centers, kmax = 4)
                                    xmeans_instance.process()
                                    clusters = xmeans_instance.get_clusters()
                                    MWE_class = np.array([-1 for _ in range(len(BS_MWE_states))])
                                    for k in range(len(clusters)):
                                        for sid in clusters[k]:
                                            MWE_class[sid] = k
                                    assert -1 not in MWE_class

                                MWE_class_counter = Counter(MWE_class)
                                # print("N_cluster: ", MWE_class_counter)
                                K_list = [len(MWE_class_counter)] #ONE CLASS
                                k_class_list = [MWE_class]
                                if opt.plot:
                                    plt.figure(figsize=(16, 16))
                                    for sid in range(len(MWE_class)):
                                        plt.scatter(x_val[sid], y_val[sid], c=colorlist[MWE_class[sid]])
                                        # plt.annotate(words[sid], (x[sid], y[sid]))
                                    plt.savefig(folder+"/"+phrase.lstrip().replace(" ","_") + str(len(MWE_class_counter))+'.png')
                            else: #K-MEANS
                                k_class_list = []
                                for y in K_list:
                                    km = KMeans(n_clusters = y)
                                    MWE_class = km.fit_predict(BS_MWE_states)
                                    # silhouette_avg = silhouette_score(BS_MWE_states, MWE_class)
                                    k_class_list.append(MWE_class)
                                    # silhouette_avgList.append(silhouette_avg)
                                    if opt.plot:
                                        plt.figure(figsize=(16, 16))
                                        for sid in range(len(MWE_class)):
                                            plt.scatter(x_val[sid], y_val[sid], c=colorlist[MWE_class[sid]])
                                        plt.savefig(opt.folder+"/"+phrase.lstrip().replace(" ","_") + str(y)+'.png')

                            assert len(K_list)==len(k_class_list)
                            K_list_size = len(K_list)
                            for K_list_idx in range(K_list_size):
                                MWE_class = k_class_list[K_list_idx] #N
                                f_count.write(veckey + " " + str(len(sentences)) + " " + " ".join(
                                    [str(e) for e in MWE_class]) + "\n")
                                assert len(MWE_class) == len(sentences)
                                sent_clusteredList = []
                                MWE_states_K = [MWE_ffn[MWE_class == k].data.cpu().numpy().mean(0) for k in range(K_list[K_list_idx])]
                                #MWE_states_K: K, N_MASK, dim
                                N_sent = 0
                                for class_id in range(K_list[K_list_idx]):
                                    sentences_raw_tmp = [sentences_raw[idx] for idx in range(len(sentences_raw)) if MWE_class[idx] == class_id]
                                    N_sent+=len(sentences_raw_tmp)
                                    sent_clusteredList.append(sentences_raw_tmp) #K, N_sent
                                    f_vec.write(" ".join([str(x) for x in MWE_states_K[class_id].reshape(-1)]) + " ||| ")
                                assert N_sent == len(sentences_raw)
                                assert phrase.lstrip(" ") not in sent_clusteredListList[K_list_idx]
                                sent_clusteredListList[K_list_idx][phrase.lstrip(" ")] = sent_clusteredList
                            f_vec.write("\n")


    elapsed_time = time.time() - start
    print("Train elapsed_time: "+str(elapsed_time))

    if 'dbscan' in opt.clustering or 'xmeans' in opt.clustering:
        assert len(sent_clusteredListList)==1
        with open(opt.folder + "/sents_by_cluster_"+opt.clustering+".pkl", 'wb') as f:
            pickle.dump(sent_clusteredListList[0], f)  # N_idiom, N_sample, s_len, dim
    else:
        for i in range(len(K_list)):
            with open(opt.folder + "/sents_by_cluster"+str(K_list[i])+".pkl", 'wb') as f:
                pickle.dump(sent_clusteredListList[i], f)  # N_idiom, N_sample, s_len, dim