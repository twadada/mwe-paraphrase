###collect sentences and filter them###
N_words=2
MWEfile=SEMEVAL_B_MWE.unique.train.EN.txt
monofile=path_to_monolingual_data
folder=folder_path
mkdir $folder
python extract_sentence.py -MWEfile ${MWEfile} -monofile ${monofile} -N_words ${N_words} -folder $folder

model=bert-base-uncased
silver_sent=${folder}/$(basename "${MWEfile}")_silversent.pkl
python remove_duplicates.py -silver_sent ${silver_sent} -model bert-base-uncased

sent=${folder}/$(basename "${MWEfile}")_silversent_cleaned.pkl
mindist=0.4
min_sample=0.03
clustering=dbscan_${mindist}_${min_sample}
cluster_model=bert-base-uncased
echo $clustering
n_mask=1
N_sample=300
cluster_folder=$(basename "$cluster_model")_${n_mask}_MASK_${clustering}
echo "clustering"
# CUDA_VISIBLE_DEVICES=0 python clustering.py -clustering ${clustering} -N_sample ${N_sample}  -folder $cluster_folder -model ${cluster_model} -silver_sent ${sent} -n_mask $n_mask


clustered_sents="${cluster_folder}/sents_by_cluster_dbscan_0.4_0.03.pkl"
folder=Result
model=bert-base-uncased
n_mask=1
beam_size=10
echo "1 masks"
# CUDA_VISIBLE_DEVICES=0 python bert_generate.py  -clustered_sents ${clustered_sents} -folder ${folder} -model ${model} -n_mask $n_mask -num_beams ${beam_size}


n_mask=2
beam_size=5
echo "2 masks"
# CUDA_VISIBLE_DEVICES=0 python bert_generate.py  -clustered_sents ${clustered_sents} -folder ${folder} -model ${model} -n_mask $n_mask -num_beams ${beam_size}




# reranking

echo "reranking"
MWE_para="${folder}/2MASKs_candidates2inner_score.pkl ${folder}/1MASKs_candidates2inner_score.pkl"
mask_opt=attn_Nmask5_Nsplit1_attnL1_norm0
# CUDA_VISIBLE_DEVICES=0 python outer_prob_new.py -clustered_sents ${clustered_sents} -folder $folder -mask_opt ${mask_opt} -candidates ${MWE_para} -model ${model}

echo "retrieval"

MWE_para_outer=${folder}/candidates2outer_score_model_${model}_${mask_opt}.pkl
vec=${cluster_folder}/vec.txt
model=${cluster_model}
target_sent=SEMEVAL_B_MWE_sim1_sent.train.EN.txt
save=${folder}/reranked_${mask_opt}
CUDA_VISIBLE_DEVICES=0 python retrieve_paraphrase.py -vec ${vec} -save ${save} -folder ${folder} -MWE_para ${MWE_para_outer} -model ${model} -target_sent ${target_sent}


candidates=${folder}/reranked_attn_Nmask5_Nsplit1_attnL1_norm0_SEMEVAL_B_MWE_sim1_sent.train.EN.txt.pkl
gold=gold_paraphrases.train.EN.txt
target_sent=SEMEVAL_B_MWE_sim1_sent.train.EN.txt
save_file=matching_result.txt
python eval_matching.py -remove_space -len_normalise -candidates ${candidates} -save ${save_file} -gold_labels ${gold} -sent_list ${target_sent}


