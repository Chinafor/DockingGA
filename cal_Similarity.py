from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
reference_smi = 'COC1=C(C(=C(C(=C1)C2=CC=C(C=C2)O)OC)O)C3=CC=C(C=C3)OCC=C'
path = './similarity.csv'
data = pd.read_csv(path)
smi = list(data['smiles'].values)
print(len(smi))


# 分子的SMILES表示
mol1 = Chem.MolFromSmiles('COC1=C(C(=C(C(=C1)C2=CC=C(C=C2)O)OC)O)C3=CC=C(C=C3)OCC=C')#海洋分子

#mol2 = Chem.MolFromSmiles('COc1ccc(-c2ccc(-c3ccc(OC)cc3)cc2)cc1')

# 将分子转化为分子指纹（Fingerprint）
for i in range(len(smi)):
    mol2 = Chem.MolFromSmiles(smi[i])
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)  # 半径为2的Morgan分子指纹
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)

    # 计算两个分子指纹的谷本系数
    tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)

    print(tanimoto)  # 输出相似度系数
