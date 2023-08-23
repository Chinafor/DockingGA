import pyscreener as ps
import ray
ray.init()
#ignore_reinit_error=True
def cal_docking(smi):
    ray.init(ignore_reinit_error=True)
    metadata = ps.build_metadata("vina")
    #virtual_screen = ps.virtual_screen("vina",receptors=["2rgp.pdb"],center=(19.496, 35.001, 89.27),size=(44.0, 49.0, 57.0),metadata_template=metadata,ncpu=6)
    virtual_screen = ps.virtual_screen("vina",receptors=["7vih.pdb"],center=(120.713, 118.886, 131.755),size=(75.0, 75.0, 75.0),metadata_template=metadata,ncpu=6)
    docking_scores = virtual_screen(smi)
    #print(docking_scores[0])
    return docking_scores[0]

if __name__ == "__main__":
    smiles = 'CC(C)(N)c1cc(C(=O)CCCCc2ccc(Oc3ccnc4c3CCC(=O)N4)cc2)cc(C(F)(F)F)c1'
    aaa = cal_docking(smiles)
    print(aaa)