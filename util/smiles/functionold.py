from time import time
import numpy as np
import torch

from guacamol.utils.chemistry import canonicalize

from joblib import Parallel, delayed

from tqdm import tqdm


def process_smis(
    smis,
    scoring_function,
    pool,
    canonicalization,
    duplicate_removal,
    scoring_parallelization,
    max_smi_len=100,
):
    if canonicalization:
        smis = pool(
            delayed(lambda smi: canonicalize(smi, include_stereocenters=False))(smi) for smi in smis
        )
        smis = list(filter(lambda smi: (smi is not None) and (len(smi) < max_smi_len), smis))

    if duplicate_removal:
        smis = list(set(smis))

    if scoring_function is None:
        return smis

    if scoring_parallelization:
        scores = pool(delayed(scoring_function)(smi) for smi in smis)
    else:
        scores = [scoring_function(smi) for smi in smis]

    smis, scores = filter_by_score(smis, scores, -1e-8)

    return smis, scores


def smis_to_actions(char_dict, smis):
    max_seq_length = char_dict.max_smi_len + 1
    enc_smis = list(map(lambda smi: char_dict.encode(smi) + char_dict.END, smis))
    actions = np.zeros((len(smis), max_seq_length), dtype=np.int32)
    seq_lengths = np.zeros((len(smis),), dtype=np.long)

    for i, enc_smi in list(enumerate(enc_smis)):
        for c in range(len(enc_smi)):
            try:
                actions[i, c] = char_dict.char_idx[enc_smi[c]]
            except:
                print(char_dict.char_idx)
                print(enc_smi)
                print(enc_smi[c])
                assert False

        seq_lengths[i] = len(enc_smi)

    return actions, seq_lengths


def filter_by_score(smis, scores, score_thr):
    filtered_smis_and_scores = list(filter(lambda elem: elem[1] > score_thr, zip(smis, scores)))
    filtered_smis, filtered_scores = map(list, zip(*filtered_smis_and_scores))
    return filtered_smis, filtered_scores





















from time import time
import numpy as np
import torch

from guacamol.utils.chemistry import canonicalize

from joblib import Parallel, delayed

from tqdm import tqdm
import pandas as pd
from subword_nmt.apply_bpe import BPE
import codecs
#add
vocab_path = '/home/gcn/code/genetic-expert-guided-learning/util/smiles/ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('/home/gcn/code/genetic-expert-guided-learning/util/smiles/ESPF/subword_units_map_chembl.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

def drug2emb_encoder(x):
    max_d = 50
    # max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)









def process_smis(
    smis,
    scoring_function,
    pool,
    canonicalization,
    duplicate_removal,
    scoring_parallelization,
    max_smi_len=100,
):
    if canonicalization:
        smis = pool(
            delayed(lambda smi: canonicalize(smi, include_stereocenters=False))(smi) for smi in smis
        )
        smis = list(filter(lambda smi: (smi is not None) and (len(smi) < max_smi_len), smis))

    if duplicate_removal:
        smis = list(set(smis))

    if scoring_function is None:
        return smis

    if scoring_parallelization:
        scores = pool(delayed(scoring_function)(smi) for smi in smis)
    else:
        scores = [scoring_function(smi) for smi in smis]

    smis, scores = filter_by_score(smis, scores, -1e-8)

    return smis, scores


def smis_to_actions(char_dict, smis):
    print(len(smis))
    max_seq_length = char_dict.max_smi_len + 1
    enc_smis = list(map(lambda smi: char_dict.encode(smi) + char_dict.END, smis))
    actions = np.zeros((len(smis), max_seq_length), dtype=np.int32)
    seq_lengths = np.zeros((len(smis),), dtype=np.long)
    ac = []
    for i, enc_smi in list(enumerate(enc_smis)):
        a ,_=drug2emb_encoder(enc_smi)
        #print("find a")
        #print(a)
        ac.append(a)
        print(i)
        arrayac = np.array(ac)
        # for c in range(len(enc_smi)):
        #     try:
        #         actions[i, c] = a
        #     except:
        #         print(char_dict.char_idx)
        #         print(enc_smi)
        #         print(enc_smi[c])
        #         assert False

        seq_lengths[i] = len(enc_smi)
    print("arrayac")
    print(arrayac)
    print(arrayac.shape)
    return arrayac, seq_lengths


def filter_by_score(smis, scores, score_thr):
    filtered_smis_and_scores = list(filter(lambda elem: elem[1] > score_thr, zip(smis, scores)))
    filtered_smis, filtered_scores = map(list, zip(*filtered_smis_and_scores))
    return filtered_smis, filtered_scores