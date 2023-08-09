import argparse
import pickle
import numpy as np
from transformers import BertTokenizer
import string

# stop_words = stopwords.words('english')
punctuations = list(string.punctuation + "“" + "”" + "-" + "’" + "‘" + "…")
stop_words = punctuations
stop_words = set(stop_words)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-silver_sent',
    help='save_name')
parser.add_argument(
    '-model',
    help='save_name')
opt = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(opt.model)

assert opt.silver_sent[-4:] == ".pkl"
with open(opt.silver_sent, 'rb') as f:
    phrase2sent_1B = pickle.load(f)

v = list(phrase2sent_1B.keys())
phrase2sent_new = dict()
threshold = 0.5
for phrase in phrase2sent_1B.keys():
    phrase_split = " ".join(tokenizer.tokenize(phrase)).replace(" ##","").lower().split()
    sentenceList = phrase2sent_1B[phrase]
    sentenceList = np.random.permutation(sentenceList)
    wordList_clean = []
    contextList = []
    sentenceList_clean = []
    for sent in sentenceList:
        words = sent.split()
        if sent and sent[-2:] != ".." and (sent[-1] in [".", "?", "!"] or sent[-2:] in ['."', '?"', '!"', '.\'', '?\'', '!\'']) and max([len(x) for x in words])<=20:
            sent_tok = tokenizer.tokenize(sent)
            if len(sent_tok)<256:
                wordList = " ".join(sent_tok).replace(" ##","").lower().split()
                MWE_count = 0
                window = 3
                for i in range(len(wordList)-len(phrase_split)):
                    if all([wordList[i+j] == phrase_split[j] for j in range(len(phrase_split))]):
                        MWE_count += 1
                        context = set(wordList[max(0,i-window):i] + wordList[i+len(phrase_split):i+len(phrase_split)+window])
                        context = context - stop_words
                assert MWE_count>0, wordList + phrase_split
                if MWE_count>1:
                    pass
                    # print(sent)
                else:
                    max_overal = []
                    max_val = 0
                    N_context = len(context)
                    if len(contextList)>0 and N_context>0:
                        for prev_context in contextList:
                            N_common_words = len(context.intersection(prev_context))
                            N_common_words = N_common_words/min(len(prev_context),N_context)
                            max_overal.append(N_common_words)
                        idx = np.argmax(max_overal)
                        max_val = max_overal[idx]
                    if max_val <= threshold and N_context>0:
                        contextList.append(set(context))
                        sentenceList_clean.append(sent)
                        wordList_clean.append(wordList)
                    else:
                        pass
                        # print(phrase)
                        # print(max_val)
                        # print(N_context)
                        # print(" ".join(wordList))
                        # print(" ".join(wordList_clean[idx]))
    print(len(sentenceList), len(sentenceList_clean))
    phrase2sent_new[phrase] = sentenceList_clean

with open(opt.silver_sent[:-4] + "_cleaned.pkl", 'wb') as f:
    pickle.dump(phrase2sent_new, f)  # N_idiom, N_sample, s_len, dim
