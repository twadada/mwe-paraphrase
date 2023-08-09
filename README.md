# "Unsupervised Paraphrasing of Multiword Expressions"
Code for ["Unsupervised Paraphrasing of Multiword Expressions" (Findings of ACL 2023)](https://aclanthology.org/2023.findings-acl.290)

If you use the code or embeddings, please cite the following paper:

```
@inproceedings{wada-etal-2023-unsupervised,
    title = "Unsupervised Paraphrasing of Multiword Expressions",
    author = "Wada, Takashi  and
      Matsumoto, Yuji  and
      Baldwin, Timothy  and
      Lau, Jey Han",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.290",
    doi = "10.18653/v1/2023.findings-acl.290",
    pages = "4732--4746",
    abstract = "We propose an unsupervised approach to paraphrasing multiword expressions (MWEs) in context. Our model employs only monolingual corpus data and pre-trained language models (without fine-tuning), and does not make use of any external resources such as dictionaries. We evaluate our method on the SemEval 2022 idiomatic semantic text similarity task, and show that it outperforms all unsupervised systems and rivals supervised systems.",
}
```

For lexical substitution, see also [this repository](https://github.com/twadada/lexsub_decontextualised).

## Reproduce Experiments for paraphrasing MWEs using BERT

1. Clone the [SemEval-22 Task2 repository](https://github.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity) at this folder

2. Run the following code

```
python preprocess_B.py -tgtlang EN -data_split train -sim_1 
```

This will produce "SEMEVAL_B_MWE.unique.train.EN.txt" (A list of MWEs used in the train split of the SemEval data), "SEMEVAL_B_MWE_sim1_sent.train.EN.txt" (MWEs and sentences), and "gold_paraphrases.train.EN.txt" (gold paraprhases). Replace EN with PT for preparing the Portuguese data.

3. Collect sentences

```
N_words=2
MWEfile=SEMEVAL_B_MWE.unique.train.EN.txt
# MWEfile = a list of target MWEs containing ${N_words} (one MWE for each line)
monofile=path_to_large_monolingual_data (we used OSCAR)
folder=folder_path
mkdir $folder
python extract_sentence.py -MWEfile ${MWEfile} -monofile ${monofile} -N_words ${N_words} -folder $folder
```

4. Discard sentences with similar local contexts
```
silver_sent=${folder}/$(basename "${MWEfile}")_silversent.pkl
python remove_duplicates.py -silver_sent ${silver_sent} -model bert-base-uncased
```

5. Cluster the collected contexts

You may skip the 3rd and 4th steps and prepare the sampled sentences ($sent) by yourself. The data format is dict{"MWE": List(sentences containing the MWE)} in a pickled file.

```
sent=${folder}/$(basename "${MWEfile}")_silversent_cleaned.pkl
mindist=0.4
# mindist = ϵ; this value needs to be tuned for each LM in the range of [0.1, 0.5]
min_sample=0.03
clustering=dbscan_${mindist}_${min_sample}
cluster_model=bert-base-uncased
echo $clustering
n_mask=1
N_sample=300
cluster_folder=$(basename "$cluster_model")_${n_mask}_MASK_${clustering}
echo "clustering"
CUDA_VISIBLE_DEVICES=0 python clustering.py -clustering ${clustering} -N_sample ${N_sample}  -folder $cluster_folder -model ${cluster_model} -silver_sent ${sent} -n_mask $n_mask
```

Replace cluster_model=bert-base-uncased with google/t5-v1_1-large for clustering sentences using T5. That said, BERT might perform better for clustering.

6. Generate paraphrases for each cluster
   
```
clustered_sents="${cluster_folder}/sents_by_cluster_dbscan_0.4_0.03.pkl"
folder=Result
model=bert-base-uncased
n_mask=1
beam_size=10
echo "Generate 1-word paraphrases"
CUDA_VISIBLE_DEVICES=0 python generate.py  -clustered_sents ${clustered_sents} -folder ${folder} -model ${model} -n_mask $n_mask -num_beams ${beam_size}

n_mask=2
beam_size=5
echo "Generate 2-word paraphrases"
CUDA_VISIBLE_DEVICES=0 python generate.py  -clustered_sents ${clustered_sents} -folder ${folder} -model ${model} -n_mask $n_mask -num_beams ${beam_size}
```

Run the following command for using T5
```
clustered_sents="${cluster_folder}/sents_by_cluster_dbscan_0.4_0.03.pkl"
folder=Result
model=google/t5-v1_1-large
n_mask=1
beam_size=20
CUDA_VISIBLE_DEVICES=0 python generate.py -t5  -clustered_sents ${clustered_sents} -folder ${folder} -model ${model} -n_mask $n_mask -num_beams ${beam_size}
```

7. Perform reranking

```
model=bert-base-uncased
MWE_para="${folder}/2MASKs_candidates2inner_score.pkl ${folder}/1MASKs_candidates2inner_score.pkl"
mask_opt=attn_Nmask5_Nsplit1_attnL1_norm0
CUDA_VISIBLE_DEVICES=0 python outer_prob_new.py -clustered_sents ${clustered_sents} -folder $folder -mask_opt ${mask_opt} -candidates ${MWE_para} -model ${model}
```

Run the following command for using T5

```
MWE_para="${folder}/1MASKs_candidates2inner_score.pkl"
mask_opt=Rspan_Nmask5_Nsplit1
model=google/t5-v1_1-large
CUDA_VISIBLE_DEVICES=0 python outer_prob_new.py -clustered_sents ${clustered_sents} -folder $folder -mask_opt ${mask_opt} -candidates ${MWE_para} -model ${model}
```

8. Retrieve relevant paraphrases given target sentences

```
MWE_para_outer=${folder}/candidates2outer_score_model_${model}_${mask_opt}.pkl
vec=${cluster_folder}/vec.txt
model=${cluster_model}
target_sent=SEMEVAL_B_MWE_sim1_sent.train.EN.txt
save=${folder}/reranked_${mask_opt}
CUDA_VISIBLE_DEVICES=0 python retrieve_paraphrase.py -vec ${vec} -save ${save} -folder ${folder} -MWE_para ${MWE_para_outer} -model ${model} -target_sent ${target_sent}
```

This will output ${folder}/reranked_attn_Nmask5_Nsplit1_attnL1_norm0_SEMEVAL_B_MWE_sim1_sent.train.EN.txt.pkl. This file pickles a Python list that consists of a list of their paraphrases with scores ([outer_probability, mask-filling probability]) for each target sentence.

8. Evaluate MWE Paraphrases

```
candidates=${folder}/reranked_attn_Nmask5_Nsplit1_attnL1_norm0_SEMEVAL_B_MWE_sim1_sent.train.EN.txt.pkl
gold=gold_paraphrases.train.EN.txt
target_sent=SEMEVAL_B_MWE_sim1_sent.train.EN.txt
save_file=matching_result.txt
in_out_w=1
python eval_matching.py -remove_space -len_normalise -candidates ${candidates} -save ${save_file} -gold_labels ${gold} -sent_list ${target_sent} -in_out_w ${in_out_w}
```

(If you set in_out_w to 0.5, it sorts paraphrases based on the rankings of the outer probability and mask-filling probability, which may work better on some data.)

