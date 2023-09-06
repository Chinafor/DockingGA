from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
reference_smi = 'COC1=C(C(=C(C(=C1)C2=CC=C(C=C2)O)OC)O)C3=CC=C(C=C3)OCC=C'
path = './similarity.csv'
data = pd.read_csv(path)
smi = list(data['smiles'].values)
print(len(smi))



mol1 = Chem.MolFromSmiles('COC1=C(C(=C(C(=C1)C2=CC=C(C=C2)O)OC)O)C3=CC=C(C=C3)OCC=C')

#mol2 = Chem.MolFromSmiles('COc1ccc(-c2ccc(-c3ccc(OC)cc3)cc2)cc1')


for i in range(len(smi)):
    mol2 = Chem.MolFromSmiles(smi[i])
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)  
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)


    tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)

    print(tanimoto)
