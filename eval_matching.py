import argparse
import pickle
import Levenshtein

def get_final_score(cand2samplescore_tgt, MWE, in_out_w, lev, lower = True, len_normalise = True):
    final_scoreDict1 = {}
    final_scoreDict2 = {}
    for w in cand2samplescore_tgt.keys():
        if len(w.strip())==0 or w.lower() in [MWE.lower(), "".join(MWE.split()).lower()]:
            continue
        else:
            if lev > 0:
                denominator = max(len(w), len(MWE.lstrip(" ").rstrip(" ")))
                if Levenshtein.distance(w.lower(), MWE.lower().lstrip(" ").rstrip(" ")) / denominator < lev:
                    # print("Skipped")
                    continue
            scores, w_ids = cand2samplescore_tgt[w]
            assert len(scores) == 2  # [inner, outer]
            outer = scores[0] #outer mask-filling score
            inner = scores[1] #MWE mask-filling score
            if len_normalise:
                inner = inner/len(w_ids)
            if lower:
                w = w.lower()
            each_word = w.split()
            if len(each_word) > 1 and len(set(each_word)) == 1:  # duplicated words, e.g. written written
                w = each_word[0]
            if w in final_scoreDict1:
                final_scoreDict1[w] = max(outer, final_scoreDict1[w])
                final_scoreDict2[w] = max(inner, final_scoreDict2[w])
            else:
                final_scoreDict1[w] = outer
                final_scoreDict2[w] = inner
    final_scoreDict1 = {word_score[0]: rank+1 for rank, word_score in
                   enumerate(sorted(final_scoreDict1.items(), key=lambda item: -1 * item[1]))}
    final_scoreDict2 = {word_score[0]: rank+1 for rank, word_score in
                   enumerate(sorted(final_scoreDict2.items(), key=lambda item: -1 * item[1]))}

    #mix rankings of inner and outer mask-filling scores
    final_scoreDict = {}
    for w in final_scoreDict1:
        final_scoreDict[w] = in_out_w*final_scoreDict1[w] + (1-in_out_w) *final_scoreDict2[w]

    final_scoreDict = [[word, rank] for word, rank in
                   sorted(final_scoreDict.items(), key=lambda item: item[1])]

    return final_scoreDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sent_list',
        required=True)
    parser.add_argument(
        '-gold_labels',
        required=True)
    parser.add_argument(
        '-candidates',
        required=True)
    parser.add_argument(
        '-lev',
        default=0.2,
        type=float)
    parser.add_argument(
        '-verbose',
        action='store_true')
    parser.add_argument(
        '-save',
        required=True)
    parser.add_argument(
        '-remove_space',
        action='store_true')

    parser.add_argument(
        '-in_out_w',
        default=1.0,
        type=float)

    opt = parser.parse_args()

    sentencesList = []
    MWEList = []

    for line in open(opt.sent_list, encoding="utf8"):
        line = line.rstrip("\n").split("\t")
        assert len(line)==4
        MWEList.append(line[0])
        sentencesList.append(line[2])

    gold_candList=[]
    for line in open(opt.gold_labels, encoding="utf8"):
        line = line.rstrip("\n")
        gold_candList.append(line)

    with open(opt.candidates, 'rb') as f:
        candidates_scoresList = pickle.load(f)


    assert len(gold_candList) == len(candidates_scoresList),str(len(gold_candList)) + " " + str(len(candidates_scoresList))
    assert len(sentencesList) == len(gold_candList),str(len(sentencesList)) + " " + str(len(gold_candList))

    glove_vectors =[]

    acc_1 = 0
    acc_5 = 0
    acc_10 = 0
    acc_15 = 0
    count = 0
    p1_matched_MWEs= set([])
    MWE_para_bestpred = set([])
    misspell_seen = set([])
    for i in range(len(gold_candList)):
        sent  = sentencesList[i]
        MWE  = MWEList[i]
        gold_para  = gold_candList[i]
        count += 1
        cand_scores = candidates_scoresList[i]
        cand2samplescore_tgt = cand_scores[0] # dict(str, val)
        # print(cand2samplescore_tgt)
        final_score = get_final_score(cand2samplescore_tgt, MWE, opt.in_out_w, opt.lev)
        #final_score: Sorted_List([paraphrase, score])
        assert "<mask>" in sent
        if len(final_score)<10:
            print("Warning: less than 10 candidates for " + MWE )
        elif len(final_score)<15:
            print("Warning: less than 20 candidates for " + MWE )
        elif len(final_score)>20:
            raise Exception

        if len(final_score):
            MWE_para_bestpred.add((MWE, gold_para, final_score[0][0]))

        if gold_para == "children's storys":
            gold_para = "children's stories"
            misspell_seen.add(gold_para)
        elif gold_para == "make feel guiltys":
            gold_para = "make feel guilty"
            misspell_seen.add(gold_para)
        elif gold_para=="movie industrys":
            gold_para = "movie industries"
            misspell_seen.add(gold_para)
        elif gold_para=="toxic trashs":
            gold_para = "toxic trash"
            misspell_seen.add(gold_para)
        elif gold_para=="unrealistics":
            gold_para = "unrealistic"
            misspell_seen.add(gold_para)
        elif gold_para=="showss":
            gold_para = "shows"
            misspell_seen.add(gold_para)
        elif gold_para=="crime levelss":
            gold_para = "crime levels"
            misspell_seen.add(gold_para)
        elif gold_para=="hiearchical order":
            gold_para = "hierarchical order"
            misspell_seen.add(gold_para)
        elif gold_para =="practical acadamic work":
            gold_para = "practical academic work"
            misspell_seen.add(gold_para)
        elif gold_para =="analgesicss":
            gold_para = "analgesics"
            misspell_seen.add(gold_para)
        elif gold_para =="disapointment":
            gold_para = "disappointment"
            misspell_seen.add(gold_para)

        gold_para_lower = gold_para.lower()

        idx = -1
        # if len(final_score) < 15:
        #     acc_15 += 1
        # else:
        for k in range(len(final_score)):
            best_word = final_score[k][0].lower()
            if best_word == gold_para_lower or best_word.replace(" ","") == gold_para.replace(" ",""): #case insensitive
                if k==0:
                    p1_matched_MWEs.add((MWE,best_word))
                    acc_1+=1
                    acc_5+=1
                    acc_10+=1
                    acc_15 += 1
                elif k<5:
                    acc_5+=1
                    acc_10+=1
                    acc_15+=1
                elif k < 10:
                    acc_10+=1
                    acc_15+=1
                elif k < 15:
                    acc_15+=1
                break

    out = []
    out.append("P@1: " + str(round(acc_1/count,3)))
    out.append("P@5: " + str(round(acc_5/count,3)))
    out.append("P@10: " + str(round(acc_10/count,3)))
    out.append("P@15: " + str(round(acc_15/count,3)))

    with open(opt.save, "w") as f:
        for i in range(len(out)):
            f.write(out[i]+"\n")
            print(out[i])

    print("p@1_matched_MWEs")
    print(p1_matched_MWEs)
    # assert len(misspell_seen)==11
