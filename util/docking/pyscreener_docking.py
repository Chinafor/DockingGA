import subprocess
import pyscreener as ps
import ray


ray.init()
smi = 'O=C(NCc1ccc(F)cc1)Nc1cc(-c2cc(=O)n3ccccc3n2)ccc1F'
smi_gen = 'Cc1cccc(C)c1NC(=O)c1ccc(-c2ccc3ncnc(Nc4ccc(OCc5ccc(F)c(F)c5)c(Cl)c4)c3c2)s1'
metadata = ps.build_metadata("vina")
virtual_screen = ps.virtual_screen("vina",receptors=["2rgp.pdb"],center=(19.496, 35.001, 89.27),size=(44.0, 49.0, 57.0),metadata_template=metadata,ncpu=4)
scores111 = virtual_screen(smi_gen)
print(scores111[0])
