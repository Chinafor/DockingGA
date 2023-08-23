import subprocess
import pyscreener as ps
import ray

#####
#smi = 'c1(ccc(NC(C(C)(C)C)=O)cc1)C(F)(F)c1ccc(-c2ccc3c(c2)ccc(C(N)=O)n3)cc1'
# metadata = ps.build_metadata("vina")
# #metadata = ps.build_metadata("vina", dict(software="qvina"))
# virtual_screen = ps.virtual_screen("vina",receptors=["2rgp.pdb"],center=(19.496, 35.001, 89.27),size=(44.0, 49.0, 57.0),metadata_template=metadata,ncpu=6)
# scores = virtual_screen(smi)
# print(scores)
# print('finish')
#####
ray.init()
smi = 'O=C(NCc1ccc(F)cc1)Nc1cc(-c2cc(=O)n3ccccc3n2)ccc1F'
smi_gen = 'Cc1cccc(C)c1NC(=O)c1ccc(-c2ccc3ncnc(Nc4ccc(OCc5ccc(F)c(F)c5)c(Cl)c4)c3c2)s1'
metadata = ps.build_metadata("vina")
#metadata = ps.build_metadata("vina", dict(software="qvina"))
#virtual_screen = ps.virtual_screen("vina", ["/home/ubuntu/gcn/genetic-expert-guided-learning/util/docking/pyscreener/integration-tests/inputs/5WIU.pdb"], (-18.2, 14.4, -16.1), (15.4, 13.9, 14.5), metadata, ncpu=4)
virtual_screen = ps.virtual_screen("vina",receptors=["2rgp.pdb"],center=(19.496, 35.001, 89.27),size=(44.0, 49.0, 57.0),metadata_template=metadata,ncpu=4)
#virtual_screen_2qd9 = ps.virtual_screen("vina", ["/home/ubuntu/gcn/genetic-expert-guided-learning/util/docking/pyscreener/integration-tests/inputs/2qd9.pdb"], (6.635, 3.548, 15.841), (72.45, 72.45, 72.45), metadata, ncpu=4)
scores111 = virtual_screen(smi_gen)
#scores = virtual_screen_2qd9(smi)
print('****')
print(scores111[0])
print('****')
#/home/ubuntu/gcn/genetic-expert-guided-learning/util/docking
#/home/ubuntu/gcn/genetic-expert-guided-learning/util/docking/pyscreener/integration-tests/inputs

#/home/ubuntu/gcn/genetic-expert-guided-learning/util/docking/ADFRsuite_x86_64Linux_1.0/myFolder/bin/prepare_receptor