#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -q gpgpudeeplearn
module load anaconda3
module load gcc/9.2.0
pip install -r requirements.txt

sent=/data/projects/punim0478/twada/Para_MWE/English_SEMEVAL_AB_MWE_Sents_eval/SEMEVAL_B_MWE.unique.eval.EN.txt_silversent_cleaned.pkl
sent=SEMEVAL_AB_MWE.unique.train_dev_testEN.txt_silversent_cleaned.pkl
mindist=0.4
min_sample=0.03
clustering=dbscan_${mindist}_${min_sample}
cluster_model=bert-base-uncased
echo $clustering
n_mask=1
N_sample=300
cluster_folder=$(basename "$cluster_model")_${n_mask}_MASK_${clustering}_train_dev_testEN
echo "clustering"
# CUDA_VISIBLE_DEVICES=0 python clustering.py -clustering ${clustering} -N_sample ${N_sample}  -folder $cluster_folder -model ${cluster_model} -silver_sent ${sent} -n_mask $n_mask

clustered_sents="${cluster_folder}/sents_by_cluster_dbscan_0.4_0.03.pkl"
folder=Result2
model=bert-base-uncased
n_mask=1
beam_size=10
echo "1 masks"
# CUDA_VISIBLE_DEVICES=0 python bert_generate.py  -clustered_sents ${clustered_sents} -folder ${folder} -model ${model} -n_mask $n_mask -num_beams ${beam_size}

echo "2 masks"
n_mask=2
beam_size=5
# CUDA_VISIBLE_DEVICES=0 python bert_generate.py  -clustered_sents ${clustered_sents} -folder ${folder} -model ${model} -n_mask $n_mask -num_beams ${beam_size}


echo "reranking"
MWE_para="${folder}/2MASKs_candidates2inner_score.pkl ${folder}/1MASKs_candidates2inner_score.pkl"
mask_opt=attn_Nmask5_Nsplit1_attnL1_norm0
# CUDA_VISIBLE_DEVICES=0 python outer_prob_new.py  -clustered_sents ${clustered_sents} -folder $folder -mask_opt ${mask_opt} -candidates ${MWE_para} -model ${model}

echo "retrieval"
MWE_para_outer=${folder}/candidates2outer_score_model_${model}_${mask_opt}.pkl
vec=${cluster_folder}/vec.txt
model=${cluster_model}
target_sent=/data/projects/punim0478/twada/Para_MWE/SEMEVAL_B_masked.train.EN.txt
target_sent=/data/projects/punim0478/twada/Para_MWE/SEMEVAL_B_MWE_sim1_sent.train.EN.txt
save=${folder}/reranked_${mask_opt}
CUDA_VISIBLE_DEVICES=0 python retrieve_paraphrase.py -vec ${vec} -save ${save} -folder ${folder} -MWE_para ${MWE_para_outer} -model ${model} -target_sent ${target_sent}