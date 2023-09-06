import gc
import random
from typing import List

import numpy as np
import torch
from joblib import Parallel, delayed
from rdkit import Chem
import sys
sys.path.append("../")
from model.genetic_operator.crossover import crossover
from model.genetic_operator.mutate import mutate
from model.genetic_operator.selfies_crossover import selfies_crossover
from model.genetic_operator.selfies_mutate import selfies_mutate


def reproduce(parent_a, parent_b, mutation_rate):
    # print(parent_a)
    # print(parent_b)
    #parent_a, parent_b = Chem.MolFromSmiles(parent_a), Chem.MolFromSmiles(parent_b)
    new_child = selfies_crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = selfies_mutate(new_child, mutation_rate)

    smis = Chem.MolToSmiles(new_child, isomericSmiles=True) if new_child is not None else None

    return smis
class GeneticOperatorHandler:
    def __init__(
        self,
        crossover_type: str,
        mutation_type: str,
        mutation_initial_rate: float,
    ) -> None:
        self.mutation_initial_rate = mutation_initial_rate
        self.mutation_rate = mutation_initial_rate

        if crossover_type == "SMILES":
            self.crossover_func = crossover
        elif crossover_type == "SELFIES":
            self.crossover_func = selfies_crossover
        else:
            raise ValueError(f"'crossover_type' {crossover_type} is invalid")

        if mutation_type == "SMILES":
            self.mutate_func = mutate
        elif mutation_type == "SELFIES":
            self.mutate_func = selfies_mutate
        else:
            raise ValueError(f"'mutation_type' {mutation_type} is invalid")


    def query(self, query_size, mating_pool, pool):
        ###old
        smis = random.choices(mating_pool, k=query_size * 2)
        smi0s, smi1s = smis[:query_size], smis[query_size:]
        #print(smis)
        smis = pool(
            delayed(reproduce)(smi0, smi1, self.mutation_rate) for smi0, smi1 in zip(smi0s, smi1s)
        )
        smis = list(filter(lambda smi: smi is not None, smis))
        gc.collect()
        return smis


    def reproduce_mols(
        self, parent_a: str, parent_b: str, mutation_rate: float
    ) -> List[str]:
        new_child = self.crossover_func(parent_a, parent_b)
        if new_child is not None:
            new_child = self.mutate_func(new_child, mutation_rate)

        smiles = (
            Chem.MolToSmiles(new_child, isomericSmiles=True)
            if new_child is not None
            else None
        )
        return smiles

    def reproduce_frags(self, smiles_list: List[str], mutation_rate: float) -> str:
        num_fragments = np.random.randint(2, 6)
        fragments = np.random.choice(smiles_list, num_fragments, replace=True).tolist()

        fragments_mol = [Chem.MolFromSmiles(frag) for frag in fragments]
        new_child = self.crossover_func(fragments_mol)  # type: ignore
        if new_child is not None:
            new_child = self.mutate_func(new_child, mutation_rate)

        smiles = (
            Chem.MolToSmiles(new_child, isomericSmiles=True)
            if new_child is not None
            else None
        )
        return smiles

