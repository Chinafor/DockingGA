import moses
import pandas as pd
import csv
import numpy
from rdkit.Chem import *
from rdkit import Chem
from rdkit.Chem import AllChem
import moses
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors

import math
import pickle
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import os
import os.path as op

# get_sa_score start
_fscores = None


def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(os.getcwd(), name)
        # name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open('%s.pkl.gz' % name))
    #data = pickle.load(gzip.open('./fpscores.pkl.gz'))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                               2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    # macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


#
# if __name__ == "__main__":
#     a = Chem.MolFromSmiles('CN(C)CCC=C1C2=CC=CC=C2CCC2=CC=CC=C12')
#     b = Chem.MolFromSmiles('[H][C@@]12CC3=CNC4=CC=CC(=C34)[C@@]1([H])C[C@H](CN2CC=C)C(=O)N(CCCN(C)C)C(=O)NCC')
#     c = Chem.MolFromSmiles('CCOC1=NC(NC(=O)CC2=CC(OC)=C(Br)C=C2OC)=CC(N)=C1C#N')
#     d = Chem.MolFromSmiles('OC(=O)C1=CC=CC=C1O')
#     x = [a, b, c, d]
#     sa_score = my_score(x)
#






# #write
# morgan_train = pd.read_csv('500k.csv')
# print(len(morgan_train))
# outfile = open('chembl_all_500k.csv', 'w')
# csv_writer = csv.writer(outfile)
# csv_writer.writerow(["", "smiles", "logP", "qed", "SAS"])
# #outfile.write('smiles,logP,qed,SAS\n')
# #x = []
# for i in range(len(morgan_train)):
#     temp = morgan_train['smiles'][i]
#     #print(temp)
#     mol = Chem.MolFromSmiles(temp)
#     num_atoms = mol.GetNumAtoms()
#     if num_atoms<=38:
#         logp = Crippen.MolLogP(Chem.MolFromSmiles(temp))
#         #print(type(logp))
#         qed = QED.default(Chem.MolFromSmiles(temp))
#         #print(qed)
#         x = [Chem.MolFromSmiles(temp)]
#         SAS = my_score(x)
#         #print(SAS)
#         #print('****')
#         #print(type(SAS))
#         csv_writer.writerow([i,temp,format(logp,'.4f'),qed,SAS])
#         print('finish'+str(i))
#     else:
#         print('finish'+str(i)+'too large'+str(num_atoms))
# print('finish')
#
#
