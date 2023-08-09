
import pickle
import timeit
import argparse

def Extract_Sentences(phrasekey_list, file, save, N_words, min_Nword, detok, lowercase, para = False):
    phrasekey_list_tmp = phrasekey_list.copy()
    phrase2sent ={}
    for sentence in open(file):
        sentence = sentence.strip('\n')
        if para:
            sentece_src_tgt = sentence.split(" ||| ")
            assert len(sentece_src_tgt)==2
            sentence_src = sentece_src_tgt[0]
        else:
            sentence_src = sentence
        if detok:
            sentence_src = sentence_src.replace(subword_prefix, '')
        if lowercase:
            sentence_src = sentence_src.lower()
        word_list = sentence_src.split(" ")
        if len(word_list)< 200 and len(word_list)>min_Nword:
            for i in range(1, len(word_list) - N_words+1):
                #NOTE: Exclude the first word
                if tuple(word_list[i:i + N_words]) in phrasekey_list_tmp:
                    phrase = " ".join(word_list[i:i + N_words])
                    if phrase not in phrase2sent:
                        phrase2sent[phrase] = set([sentence])
                    elif len(phrase2sent[phrase])<2000:
                        phrase2sent[phrase].add(sentence)
                    else:
                        if phrase in phrasekey_list_tmp:
                            phrasekey_list_tmp.remove(phrase)
        if len(phrasekey_list_tmp) ==0:
            break
    for x in phrase2sent.keys():
        phrase2sent[x] = list(phrase2sent[x])
    with open(save + ".pkl", 'wb') as f:
        pickle.dump(phrase2sent, f, pickle.HIGHEST_PROTOCOL)
    return phrase2sent


parser = argparse.ArgumentParser()



parser.add_argument(
    '-MWEfile',
    required=True,
    help='silver_sent_path')

parser.add_argument(
    '-monofile',
    required=True,
    help='silver_sent_path')

parser.add_argument(
    '-parafile',
    help='silver_sent_path')

parser.add_argument(
    '-N_words',
    type=int,
    required=True,
    help='silver_sent_path')
parser.add_argument(
    '-min_Nword',
    type=int,
    default=15)
parser.add_argument(
    '-folder',
    required=True,
    help='silver_sent_path')
parser.add_argument(
        '-lowercase',
        action='store_true',
        help='save_name')

parser.add_argument(
        '-detok',
        action='store_true',
        help='save_name')

opt = parser.parse_args()
MWEfile = opt.MWEfile

N_words = opt.N_words
folder = opt.folder
subword_prefix = " ##"

phrase2sent  = {}
phrasekey_list = set([])
for line in open(MWEfile):
    line = line.strip('\n')
    phrase = line.split() #["old, flame"]
    assert len(phrase) == N_words, phrase
    phrasekey_list.add(tuple(phrase))

print("#phrase, ",len(phrasekey_list))

N_count = 0
save = folder + "/"  + MWEfile.split("/")[-1]

if opt.parafile is not None:
    # Check_file(opt.parafile + suffix)
    start = timeit.default_timer()
    phrases2sent = Extract_Sentences(phrasekey_list, opt.parafile, save + "_parasent", N_words, opt.min_Nword, opt.detok, opt.lowercase,para= True)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print("#phrase, ",len(phrases2sent))

start = timeit.default_timer()
phrases2sent = Extract_Sentences(phrasekey_list, opt.monofile, save+"_silversent", N_words, opt.min_Nword, opt.detok, opt.lowercase)
stop = timeit.default_timer()
print('Time: ', stop - start)
print("#phrase, ", len(phrases2sent))

