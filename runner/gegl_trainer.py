import random
import numpy as np
import torch
import neptune
from typing import List, Optional, Tuple
from joblib import delayed
from joblib import Parallel
from guacamol.utils.chemistry import canonicalize
#from genetic_expert import GeneticOperatorHandler
from guacamol.scoring_function import ScoringFunction
from util.selfiesutil.smiles import canonicalize_and_score_smiles
from model.genetic_operator import *
class GeneticExpertGuidedLearningTrainer:
    def __init__(
        self,
        apprentice_storage,
        expert_storage,
        apprentice_handler,
        expert_handler,
        char_dict,
        num_keep,
        apprentice_sampling_batch_size,
        expert_sampling_batch_size,
        apprentice_training_batch_size,
        num_apprentice_training_steps,
        init_smis,
    ):
        self.apprentice_storage = apprentice_storage
        self.expert_storage = expert_storage

        self.apprentice_handler = apprentice_handler
        self.expert_handler = expert_handler

        self.char_dict = char_dict

        self.num_keep = num_keep
        self.apprentice_sampling_batch_size = apprentice_sampling_batch_size
        self.expert_sampling_batch_size = expert_sampling_batch_size
        self.apprentice_training_batch_size = apprentice_training_batch_size
        self.num_apprentice_training_steps = num_apprentice_training_steps

        self.init_smis = init_smis
        #add
        num_experts = len(self.expert_handler)
        self.num_experts = num_experts
        self.apprentice_mean_similarity = 1.0
        self.partial_query_sizes = (
                [  # Initialize the query sizes to uniform distribution at the beginning
                    int(self.expert_sampling_batch_size / self.num_experts)
                ]
                * self.num_experts
        )

    def init(self, scoring_function, device, pool):
        if len(self.init_smis) > 0:
            smis, scores = self.canonicalize_and_score_smiles(
                smis=self.init_smis, scoring_function=scoring_function, pool=pool
            )

            self.apprentice_storage.add_list(smis=smis, scores=scores)
            self.expert_storage.add_list(smis=smis, scores=scores)

    def step(self, scoring_function, device, pool: Parallel):
        apprentice_smis, apprentice_scores = self.update_storage_by_apprentice(
            scoring_function, device, pool
        )
        expert_smis, expert_scores = self.update_storage_by_expert(scoring_function, pool)
        loss, fit_size = self.train_apprentice_step(device)

        neptune.log_metric("apprentice_loss", loss)
        neptune.log_metric("fit_size", fit_size)

        return apprentice_smis + expert_smis, apprentice_scores + expert_scores

    def update_storage_by_apprentice(self, scoring_function, device, pool: Parallel):
        with torch.no_grad():
            self.apprentice_handler.model.eval()
            smis, _, _, _ = self.apprentice_handler.sample(
                num_samples=self.apprentice_sampling_batch_size, device=device
            )

        smis, scores = self.canonicalize_and_score_smiles(
            smis=smis, scoring_function=scoring_function, pool=pool
        )

        self.apprentice_storage.add_list(smis=smis, scores=scores)
        self.apprentice_storage.squeeze_by_kth(k=self.num_keep)

        return smis, scores
    #add
    # def update_storage_by_expert(self, scoring_function: ScoringFunction, pool: Parallel):
    #     apprentice_smiles, _ = self.apprentice_storage.sample_batch(self.expert_sampling_batch_size)
    #     canon_smiles: List[str] = []
    #     canon_scores: List[float] = []
    #
    #     for expert_idx in range(self.num_experts):
    #         print(expert_idx)
    #         expert = self.expert_handler[expert_idx]
    #         mating_pool = apprentice_smiles
    #         query_smiles = expert.query(
    #             query_size=self.partial_query_sizes[expert_idx],
    #             #apprentice_mean_similarity=self.apprentice_mean_similarity,
    #             mating_pool=mating_pool,
    #             pool=pool,
    #         )
    #         print("*****"+query_smiles)
    #
    #         partial_smiles, partial_scores = canonicalize_and_score_smiles(
    #             smiles=query_smiles,
    #             scoring_function=scoring_function,
    #             #char_dict=self.char_dict,
    #             pool=pool,
    #         )
    #
    #         self.expert_storage.add_list(
    #             smiles=partial_smiles, scores=partial_scores, expert_id=expert_idx
    #         )
    #         canon_smiles += partial_smiles
    #         canon_scores += partial_scores
    #
    #     self.logger.log_metric("mutation_rate", self.expert_handlers[0].mutation_rate)
    #     self.expert_storage.squeeze_by_rank(top_k=self.num_keep)
    #
    #     expert_ratios = [
    #         query_size / self.expert_sampling_batch_size
    #         for query_size in self.partial_query_sizes
    #     ]
    #     self.logger.log_values("expert_ratios", expert_ratios)
    #
    #     # Update the partial query sizes for the next round
    #     _, _, expert_ids = self.expert_storage.get_elements()
    #     self.partial_query_sizes = self.sampling_handler.calculate_partial_query_size(
    #         expert_ids
    #     )
    #
    #     return canon_smiles, canon_scores


    def update_storage_by_expert(self, scoring_function, pool):
        expert_smis, expert_scores = self.apprentice_storage.sample_batch(
            self.expert_sampling_batch_size
        )
        for expert_idx in range(self.num_experts):
            expert = self.expert_handler[expert_idx]
            smis = expert.query(
                query_size=self.expert_sampling_batch_size, mating_pool=expert_smis, pool=pool
            )
            smis, scores = self.canonicalize_and_score_smiles(
                smis=smis, scoring_function=scoring_function, pool=pool
            )

            self.expert_storage.add_list(smis=smis, scores=scores)
            self.expert_storage.squeeze_by_kth(k=self.num_keep)

            return smis, scores

    def train_apprentice_step(self, device):
        avg_loss = 0.0
        apprentice_smis, _ = self.apprentice_storage.get_elems()
        expert_smis, _ = self.expert_storage.get_elems()
        total_smis = list(set(apprentice_smis + expert_smis))

        self.apprentice_handler.model.train()
        for _ in range(self.num_apprentice_training_steps):
            smis = random.choices(population=total_smis, k=self.apprentice_training_batch_size)
            loss = self.apprentice_handler.train_on_batch(smis=smis, device=device)

            avg_loss += loss / self.num_apprentice_training_steps

        fit_size = len(total_smis)

        return avg_loss, fit_size

    def canonicalize_and_score_smiles(self, smis, scoring_function, pool):
        smis = pool(
            delayed(lambda smi: canonicalize(smi, include_stereocenters=False))(smi) for smi in smis
        )
        smis = list(filter(lambda smi: (smi is not None) and self.char_dict.allowed(smi), smis))
        scores = pool(delayed(scoring_function.score)(smi) for smi in smis)
        # scores = [0.0 for smi in smis]

        filtered_smis_and_scores = list(
            filter(
                lambda smi_and_score: smi_and_score[1]
                > scoring_function.scoring_function.corrupt_score,
                zip(smis, scores),
            )
        )

        smis, scores = (
            map(list, zip(*filtered_smis_and_scores))
            if len(filtered_smis_and_scores) > 0
            else ([], [])
        )
        return smis, scores
