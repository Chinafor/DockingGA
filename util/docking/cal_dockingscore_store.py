import pyscreener as ps
import ray
ray.init()
#ignore_reinit_error=True
def cal_docking(smi):
    ray.init(ignore_reinit_error=True)
    metadata = ps.build_metadata("vina")
    virtual_screen = ps.virtual_screen("vina",receptors=["2rgp.pdb"],center=(19.496, 35.001, 89.27),size=(44.0, 49.0, 57.0),metadata_template=metadata,ncpu=6)
    #virtual_screen = ps.virtual_screen("vina",receptors=["7vih.pdb"],center=(120.713, 118.886, 131.755),size=(75.0, 75.0, 75.0),metadata_template=metadata,ncpu=6)
    docking_scores = virtual_screen(smi)
    #print(docking_scores[0])
    return docking_scores[0]

with open('../../read_molecules/selfies_ga.txt', 'r') as f:
    lines = f.readlines()
    result = [lines[i] for i in range(0, len(lines))]
if __name__ == "__main__":
    #smiles = 'CC(C)(N)c1cc(C(=O)CCCCc2ccc(Oc3ccnc4c3CCC(=O)N4)cc2)cc(C(F)(F)F)c1'
    print('docking..selfies_ga')
    docking_score = []
    
    for i in range(len(result)):
        try:
            aaa = cal_docking(result[i])
            print(aaa)
            docking_score.append(aaa)
        except Exception as e:
            print('error')
            continue
    file = open("../../read_molecules/ds_selfies_ga.txt", "w")
    for item in docking_score:
        file.write(str(item) + '\n')
    file.close()

    print('finish')
    