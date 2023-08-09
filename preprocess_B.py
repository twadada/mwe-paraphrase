import csv
import re
import string
import numpy as np
import argparse

def Find_MWE(sent_tmp, phraseList, lang):
    #sent_tmp: List of words (str)
    #phraseList: List of phrases (list)
    #lang: target lang (str)
    MWE_indexList=[]
    punct = string.punctuation + "“" + "”" + "-" + "’" + "‘" + "…" + "»" + "«" + "–"
    correct_phraseList = []
    puntsList =[]
    # assert len(set([len(x) for x in phraseList]))==1
    for i, phrase_tmp in enumerate(phraseList):
        phrase_len = len(phraseList[i]) #N words
        for i in range(len(sent_tmp) - phrase_len + 1):
            hit = 0
            before_panct = ""
            after_panct = ""
            phrase_matched = []
            for j in range(len(phrase_tmp)):  # for each word
                re_pattern = re.compile(re.escape(phrase_tmp[j]))
                w = sent_tmp[i + j]
                found = re_pattern.search(w.lower())
                if not found:
                    break #break the loop over the words in phrase
                else:
                    # print(w)
                    init, end = found.span()
                    if lang.lower() == "en":
                        has_apos_s = w[end:] == "'s" or w[end:] == "’s" or w[end:] == "'s'" or w[end:] == '\'s"'
                    else:
                        has_apos_s = False
                    # print(w[end:])
                    # print(has_apos_s)
                    has_punct_before = init != 0 and (all([c in punct for c in list(w[:init])]))
                    has_hyphen_before = init != 0 and w[init-1] in ["-","/","—"] #Diddy-Dirty Money, office/research lab
                    has_punct_before = has_punct_before or has_hyphen_before
                    has_punct_after = end != len(w) and all([c in punct for c in list(w[end:])])
                    has_hyphen_after = end != len(w) and w[end] in ["-","/","—"]
                    special_cases= w[end:].lower() == "…moito" or w[end:].lower() == ",(o"
                    has_punct_after = has_punct_after or has_apos_s or has_hyphen_after or special_cases
                    if (init == 0 or has_punct_before) and (end == len(w) or has_punct_after):
                        hit += 1
                        phrase_matched.append(w[init:end])
                        if init == 0 and end == len(w): #no punct
                            pass
                        else:
                            if has_punct_before and j == 0: #punct before the first word of MWE
                                before_panct = w[:init]
                            if has_punct_after and j == len(phrase_tmp)-1: #punct after last word
                                after_panct = w[end:]
                            if has_punct_before and j != 0:  # punct before the second word of MWE "sala pós-cirúrgica"
                                assert w =="pós-cirúrgica"
                                break
                            # if has_punct_after and  j !=  len(phrase_tmp)-1:  # punct before the first word of MWE
                            #     print(sent_tmp)
                            #     print(w)
                            #     print(j)
                            #     print(len(phrase_tmp)-1)
                            #     print(has_punct_after)
                            #     raise Exception
                            #     # print(w)
                            #     break
                        if hit == len(phrase_tmp): #found the whole phrase
                            # assert not (j == 0 and has_punct_after)
                            # assert not (j == len(phrase_tmp) - 1 and has_punct_before)
                            assert " ".join(phrase_matched).lower() == " ".join(phrase_tmp)
                            correct_phrase = phrase_matched #w.o punct
                            MWE_indexList.append(i)
                            correct_phraseList.append(correct_phrase)
                            puntsList.append([before_panct, after_panct])
                            # out.append([i, correct_phrase, before_panct, after_panct])
                            break
    #[47,25]
    #[bad apple, bad apples]
    idx = np.argsort(MWE_indexList)
    MWE_indexList = [MWE_indexList[i] for i in idx]
    correct_phraseList = [correct_phraseList[i] for i in idx]
    puntsList = [puntsList[i] for i in idx]
    return MWE_indexList, correct_phraseList, puntsList


def load_csv( path ) :
  header = None
  data   = list()
  with open( path, encoding='utf-8') as csvfile:
    reader = csv.reader( csvfile )
    for row in reader :
      if header is None :
        header = row
        continue
      data.append( row )
  return header, data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '-tgtlang',
      required=True,
    type=str
    )
    parser.add_argument(
      '-data_split',
      required=True,
        type=str)
    parser.add_argument(
      '-sim_1',
    action = 'store_true')

    opt = parser.parse_args()
    header, data = load_csv("SemEval_2022_Task2-idiomaticity/SubTaskB/EvaluationData/dev.gold.csv")
    DevID2sim = {}
    otherID = set([])
    for i in range(len(data)):
        SID = data[i][0]
        sim = data[i][3]
        assert SID not in DevID2sim
        DevID2sim[SID] = sim
        otherID.add(data[i][4])

    MWEs_set = []
    if opt.data_split=="train":
        file = ["SemEval_2022_Task2-idiomaticity/SubTaskB/TrainData/train_data.csv"]
    elif opt.data_split=="dev":
        file = ["SemEval_2022_Task2-idiomaticity/SubTaskB/EvaluationData/dev.csv"]
    elif opt.data_split=="eval":
        file = ["SemEval_2022_Task2-idiomaticity/SubTaskB/EvaluationData/eval.csv"]
    elif opt.data_split=="test":
        file = ["SemEval_2022_Task2-idiomaticity/SubTaskB/TestData/test.csv"]
    elif opt.data_split=="all":
        file = ["SemEval_2022_Task2-idiomaticity/SubTaskB/TestData/test.csv",
                "SemEval_2022_Task2-idiomaticity/SubTaskB/EvaluationData/dev.csv",
                "SemEval_2022_Task2-idiomaticity/SubTaskB/TrainData/train_data.csv"]


    header, data = load_csv("SemEval_2022_Task2-idiomaticity/SubTaskB/TestData/test.csv")
    x = set([])
    for elem in data:
        mwe1 = elem[header.index('MWE1')]
        mwe2 = elem[header.index('MWE2')]
        assert mwe2 == "None"
        if mwe1 != "None":
            x.add(mwe1)
    # header_submit, data_submit = load_csv("SemEval_2022_Task2-idiomaticity/SubTaskB/TestData/test_submission_format.csv")
    x = set([])
    for f in file:
        header, data = load_csv(f)
        # header, data = load_csv("SemEval_2022_Task2-idiomaticity/SubTaskB/TrainData/train_data.csv")
        # header, data = load_csv("SemEval_2022_Task2-idiomaticity/SubTaskB/EvaluationData/dev.csv")
        # header, data = load_csv("SemEval_2022_Task2-idiomaticity/SubTaskB/TestData/test.csv")
        mwe1_list = []
        en_sent1_list = []
        en_sent2_list = []
        general_sts_sim = []
        general_sts_sent1 = []
        general_sts_sent2 = []
        punct = string.punctuation+"“"+"”"+"-"+"’" +"‘"+"…"+"»"+"«"+"–"
        out_masked_sent = []
        MWE_sent_idx = []
        gold_paraphraseList = []
        count = -1
        two_MWEs = 0
        for elem in data:
            # ID = elem[header.index('ID')]
            # assert data_submit[count][0]==ID
            lang = elem[header.index('Language')]
            if 'sentence1' in header:
                sentence1 = elem[header.index('sentence1')]
                sentence2 = elem[header.index('sentence2')]
            else:
                sentence1 = elem[header.index('sentence_1')]
                sentence2 = elem[header.index('sentence_2')]
            if lang.lower() == opt.tgtlang.lower():
                count += 1
                if "position--would" in sentence1:
                    sentence1 = sentence1.replace("position--would", "position-- would")
                    sentence2 = sentence2.replace("position--would", "position-- would")
                en_sent1_list.append(sentence1)
                en_sent2_list.append(sentence2)
                mwe1 = elem[header.index('MWE1')]
                mwe2 = elem[header.index('MWE2')]
                assert mwe2 == "None"
                if mwe1 == "None" or sentence1 =='The university distributed a smaller portion of CARES Act grants to graduate students because they could qualify for greater amounts of aid and loans in comparison to undergraduate students, according to a university spokesperson.':# in train data
                    #graduate students -> understudents??
                    # sentID  = elem[header.index('ID')]
                    # if "sim" not in header and sentID not in otherID: #dev
                    #     sim = DevID2sim[sentID]
                    #     assert sim not in ["1",""] #sts
                    #     _  = float(sim)
                    #     general_sts_sim.append(sim)
                    #     general_sts_sent1.append(sentence1)
                    #     general_sts_sent2.append(sentence2)
                    pass
                else: #MWE in the sentence1
                    x.add(mwe1)
                    if opt.sim_1:
                        if "sim" in header:
                            sim = elem[header.index('sim')]
                            assert sim in ["1", "None"]
                        else:
                            sim = DevID2sim[elem[header.index('ID')]]
                            assert sim in ["1",""]
                    if not opt.sim_1 or (opt.sim_1 and sim =="1"):
                        if "sim" in header and opt.sim_1:
                            assert elem[header.index('alternative_1')]=="",elem[header.index('alternative_1')]
                        # if sim!="":
                        #     assert sim == "1"
                        MWE_sent_idx.append(count)
                        # phrase_tmp = mwe1.lower().split()
                        phrase_tmp = mwe1.split()
                        phrase_len = len(phrase_tmp)
                        sent_tmp = sentence1.split().copy()
                        flag = False
                        phrase_tmp_s = phrase_tmp.copy()
                        phrase_tmp_s[-1] += 's'
                        phrase_tmp_es = phrase_tmp.copy()
                        phrase_tmp_es[-1] += 'es'
                        phrase_tmp_ing = phrase_tmp.copy()
                        phrase_tmp_ing[-1] += 'ing'
                        # if lang=="EN":
                        phraseList = [phrase_tmp, phrase_tmp_s, phrase_tmp_es, phrase_tmp_ing]
                        # else:
                        #     phraseList = [phrase_tmp]
                        # for phrase_tmp in x:
                        diff_len = 0
                        masked_sent1 = []
                        masked_sent2 = []
                        paraphraseList = []
                        MWE_indexList, correct_phraseList, puntsList \
                            = Find_MWE(sent_tmp, phraseList, lang)
                        # if ["sala", "cirúrgica"] in phraseList and "pós-cirúrgica" in sent_tmp:
                        #     MWE_indexList = MWE_indexList[:1]
                        #     correct_phraseList = correct_phraseList[:1]
                        #     puntsList = puntsList[:1]
                        assert len(MWE_indexList)>0, sent_tmp+phraseList
                        # MWE_indexList = [MWE_indexList[0]]
                        # correct_phraseList = [correct_phraseList[0]]
                        # puntsList = MWE_indexList[0]
                        # print(sent_tmp)
                        # print(MWE_indexList)
                        # print(correct_phraseList)
                        for N in range(len(MWE_indexList)):
                            sent1_MWE_idx = MWE_indexList[N]
                            correct_phrase = correct_phraseList[N]
                            before_panct, after_panct = puntsList[N]
                            ####
                            sent_tmp2 = sentence2.split().copy()
                            if N == 0:
                                # print(sent_tmp[:sent1_MWE_idx] )
                                # print(sent_tmp2[:sent1_MWE_idx] )
                                assert sent_tmp[:sent1_MWE_idx] == sent_tmp2[:sent1_MWE_idx]
                                masked_sent1 = sent_tmp[:sent1_MWE_idx].copy()
                                # masked_sent2 = masked_sent1.copy()
                            Next_word_S1_idx = sent1_MWE_idx + len(correct_phraseList[N])
                            word_after_MWE = sent_tmp[Next_word_S1_idx:Next_word_S1_idx + 3].copy()
                            sent2_MWE_idx = sent1_MWE_idx + diff_len
                            if len(word_after_MWE) ==0: #MWE in the last sent
                                assert N == len(MWE_indexList)-1
                                assert Next_word_S1_idx == len(sent_tmp)
                                paraphrase = sent_tmp2[sent2_MWE_idx:].copy() #the last words
                            else:
                                paraphrase = []
                                para_len = 0
                                for k in range(len(sent_tmp2)-sent1_MWE_idx):
                                    if sent_tmp2[sent2_MWE_idx+k:sent2_MWE_idx+k+3] == word_after_MWE:
                                        break
                                    elif phrase_tmp ==  ['number', 'crunching'] and sent_tmp2[sent2_MWE_idx+k:sent2_MWE_idx+k+3] in [['(although', 'the','calculations'],['(although', 'the','crunching']]:
                                        #when MWE appears twice
                                        # print(sent_tmp)
                                        # print(sent_tmp2)
                                        break
                                    elif phrase_tmp ==  ['double', 'cross'] and sent_tmp2[sent2_MWE_idx+k:sent2_MWE_idx+k+3] in [['of', 'the', 'betrayal'],['of', 'the', 'two']]:
                                        # when MWE appears twice
                                        # print(sent_tmp)
                                        # print(sent_tmp2)
                                        break
                                    elif phrase_tmp ==  ['inner', 'product'] and sent_tmp2[sent2_MWE_idx+k:sent2_MWE_idx+k+3] in [['and', 'the', 'inner'],['and', 'the', 'algebraic']]:
                                        # when MWE appears twice
                                        # print(sent_tmp)
                                        # print(sent_tmp2)
                                        break
                                    else:
                                        paraphrase.append(sent_tmp2[sent2_MWE_idx+k]) #append words (paraphrase) until we have the same right context
                            # assert len(paraphrase)<=4
                            if before_panct != "":
                                assert paraphrase[0][:len(before_panct)] == before_panct
                                paraphrase[0] = paraphrase[0][len(before_panct):] #omit punct
                                before_panct = before_panct + " " #add space
                            if after_panct != "":
                                assert paraphrase[-1][-len(after_panct):] == after_panct
                                paraphrase[-1] = paraphrase[-1][:-len(after_panct)] #omit punct
                                after_panct =  " " + after_panct #add space
                            para_len = len(paraphrase)
                            Next_word_S2_idx = sent2_MWE_idx + para_len
                            diff_len += para_len - phrase_len #adjust index for sent_2
                            # assert from MWE to next id before MWE
                            if N == len(MWE_indexList)-1: #this is the last MWE
                                assert sent_tmp[Next_word_S1_idx:] == sent_tmp2[Next_word_S2_idx:]  # two MWEs
                                masked_sent1 = masked_sent1 + [before_panct + "<mask>" + after_panct] + sent_tmp[Next_word_S1_idx:].copy()
                                # masked_sent2 = masked_sent2 + [before_panct+"<mask>"+after_panct]+sent_tmp2[Next_word_S2_idx:MWE_indexList[N+1]+diff_len] #two MWEs
                            else: #if there is another MWE
                                assert sent_tmp[Next_word_S1_idx:MWE_indexList[N + 1]] == sent_tmp2[Next_word_S2_idx:MWE_indexList[N + 1] + diff_len]  # two MWEs
                                masked_sent1 = masked_sent1 + [before_panct + "<mask>" + after_panct] + sent_tmp[Next_word_S1_idx:MWE_indexList[N + 1]].copy()
                                # masked_sent2 = masked_sent2 + [before_panct+"<mask>"+after_panct] + sent_tmp2[Next_word_S2_idx:]  # two MWEs
                            paraphraseList.append(paraphrase)
                            # if before_panct!="" or after_panct!="":
                            #     print("N: ",N)
                            #     print(sentence1)
                            #     print(sentence2)
                            #     print(masked_sent1)
                            #     print(masked_sent2)
                            #     print(correct_phrase, paraphrase)
                            # print("^^^^")
                            # print(correct_phrase)
                            # print(out)
                        # reconst = " ".join(masked_sent1)
                        # for N in range(len(MWE_indexList)):
                        #     before_panct, after_panct = puntsList[N]
                        #     if before_panct == "" and after_panct == "":
                        #         reconst = reconst.replace( "<mask>" , " ".join(correct_phraseList[N]), 1)
                        #     else:
                        #         if before_panct != "" and after_panct != "":
                        #             reconst = reconst.replace(" <mask> ", " ".join(correct_phraseList[N]), 1)
                        #         elif before_panct != "":
                        #             reconst = reconst.replace(" <mask>", " ".join(correct_phraseList[N]), 1)
                        #         elif after_panct != "":
                        #             reconst = reconst.replace("<mask> ", " ".join(correct_phraseList[N]), 1)
                        #
                        #
                        reconst_sent1 = " ".join(masked_sent1)
                        reconst_sent2 = " ".join(masked_sent1)
                        for N in range(len(MWE_indexList)):
                            before_panct, after_panct = puntsList[N]
                            if before_panct == "" and after_panct == "":
                                reconst_sent1 = reconst_sent1.replace( "<mask>" , " ".join(correct_phraseList[N]), 1)
                                reconst_sent2 = reconst_sent2.replace( "<mask>" , " ".join(paraphraseList[N]), 1)
                            else:
                                if before_panct != "" and after_panct != "":
                                    reconst_sent1 = reconst_sent1.replace(" <mask> ", " ".join(correct_phraseList[N]), 1)
                                    reconst_sent2 = reconst_sent2.replace(" <mask> ", " ".join(paraphraseList[N]), 1)
                                elif before_panct != "":
                                    reconst_sent1 = reconst_sent1.replace(" <mask>", " ".join(correct_phraseList[N]), 1)
                                    reconst_sent2 = reconst_sent2.replace(" <mask>", " ".join(paraphraseList[N]), 1)
                                elif after_panct != "":
                                    reconst_sent1 = reconst_sent1.replace("<mask> ", " ".join(correct_phraseList[N]), 1)
                                    reconst_sent2 = reconst_sent2.replace("<mask> ", " ".join(paraphraseList[N]), 1)
                                else:
                                    raise Exception
                        assert reconst_sent1 == " ".join(sent_tmp), reconst_sent1+"|"+" ".join(sent_tmp)#make sure the original sent can be reconstucted
                        assert reconst_sent2 == " ".join(sent_tmp2) #make sure the original sent can be reconstucted
                        reconst = " ".join(masked_sent1)
                        assert "<mask>" in reconst
                        keep_mask_idx = 0
                        for hit_idx in range(len(correct_phraseList)):
                            if correct_phraseList[hit_idx] == phraseList[0]:
                                keep_mask_idx = hit_idx
                                break
                        MWE = " ".join(correct_phraseList[keep_mask_idx])
                        gold_para = " ".join(paraphraseList[keep_mask_idx])
                        gold_paraphraseList.append(gold_para)
                        if len(MWE) == 1 and len(MWE.split("-"))==2:
                            MWE = " ".join(MWE.split("-"))
                        out = MWE  + "\t" + str(keep_mask_idx) + "\t"+reconst+"\t"+reconst_sent1
                        for MWE_tmp in correct_phraseList:
                            if len(MWE_tmp) == 1 and len(MWE_tmp[0].split("-")) == 2:
                                MWE_tmp = " ".join(MWE_tmp[0].split("-"))
                            else:
                                MWE_tmp = " ".join(MWE_tmp)
                            if MWE_tmp not in MWEs_set:
                                MWEs_set.append(MWE_tmp)
                        out_masked_sent.append(out)


    with open("SEMEVAL_B_MWE.unique."+opt.data_split+"."+opt.tgtlang+".txt", "w") as f:
        for w in MWEs_set:
            f.write(w + "\n")
    #
    if opt.sim_1:
        with open("SEMEVAL_B_MWE_sim1_sent."+opt.data_split+"."+opt.tgtlang+".txt", "w") as f:
            for w in out_masked_sent:
                f.write(str(w) + "\n")
        with open("gold_paraphrases."+opt.data_split+"."+opt.tgtlang+".txt", "w") as f:
            for w in gold_paraphraseList:
                f.write(str(w) + "\n")
    else:
        with open("SEMEVAL_B_MWE_sent_idx"+opt.data_split+"."+opt.tgtlang+".txt", "w") as f:
            for w in MWE_sent_idx:
                f.write(str(w) + "\n")
        with open("SEMEVAL_B_Src_sent."+opt.data_split+"."+opt.tgtlang+"txt", "w") as f:
            for w in en_sent1_list:
                f.write(w + "\n")
        with open("SEMEVAL_B_Tgt_sent."+opt.data_split+"."+opt.tgtlang+".txt", "w") as f:
            for w in en_sent2_list:
                f.write(w + "\n")
        with open("SEMEVAL_B_masked."+opt.data_split+"."+opt.tgtlang+".txt", "w") as f:
            for w in out_masked_sent:
                f.write(w + "\n")
