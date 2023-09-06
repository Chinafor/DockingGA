# DockingGA
The code of article: 

DockingGA: Enhancing Targeted Molecule Generation using Transformer Neural Network and Genetic Algorithm with Docking Simulation

1.Setting up the environment

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

conda install -c dglteam dgl-cuda10.1

conda install -c rdkit rdkit

pip install neptune-client

pip install tqdm

pip install psutil

2.neptune initialization (https://app.neptune.ai/)

You need to complete the neptune initialization here:

neptune.init(project_qualified_name="",api_token='',)

3.Run pre-train or generate

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python xx.py

4.Dataset

All data is stored here:

./resource/data

5.pyscreener

For pyscreener installation please refer to the following link:

https://github.com/coleygroup/pyscreener


