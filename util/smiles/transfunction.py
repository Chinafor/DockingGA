from time import time
import numpy as np
import torch
import pandas as pd
from guacamol.utils.chemistry import canonicalize
from joblib import Parallel, delayed
from tqdm import tqdm
from subword_nmt.apply_bpe import BPE
import codecs
from time import time
import numpy as np
import torch
from guacamol.utils.chemistry import canonicalize
from joblib import Parallel, delayed
import pandas as pd
from subword_nmt.apply_bpe import BPE
import codecs

vocab_path = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')

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
def smis_to_actions(self,char_dict, smis):
    max_seq_length = char_dict.max_smi_len + 1
    enc_smis = list(map(lambda smi: char_dict.encode(smi) + char_dict.END, smis))
    print("debug")
    print(enc_smis)
    print(type(enc_smis))
    #enc_smis,_ = drug2emb_encoder(smis)
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


x = "O=C(N[C@@H]1[C@@H]2CCO[C@@H]2C12CCC2)c1cnc([C@H]2CCCO2)s1"
a,b = drug2emb_encoder(x)
print(a)
print(a.shape)#(50,)
print("****0000")
print(b)
print(b.shape)#(50,)
