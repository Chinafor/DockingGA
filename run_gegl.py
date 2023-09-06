import random
import json
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from runner.gegl_trainer import GeneticExpertGuidedLearningTrainer
from runner.guacamol_generator import GeneticExpertGuidedLearningGenerator
from model.neural_apprentice import SmilesGenerator, SmilesGeneratorHandler
from model.genetic_expert import GeneticOperatorHandler
from util.storage.priority_queue import MaxRewardPriorityQueue
from util.storage.recorder import Recorder
from util.chemistry.benchmarks import load_benchmark
from util.smiles.char_dict import SmilesCharDictionary
from util.selfiesutil.load_function import load_genetic_experts
import neptune

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark_id", type=int, default=28)
    parser.add_argument("--dataset", type=str, default="zinc")
    parser.add_argument("--max_smiles_length", type=int, default=81)#100
    parser.add_argument("--apprentice_load_dir", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--sample_batch_size", type=int, default=512)
    parser.add_argument("--optimize_batch_size", type=int, default=256)
    parser.add_argument("--mutation_rate", type=float, default=0.01)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--num_keep", type=int, default=1024)
    parser.add_argument("--max_sampling_batch_size", type=int, default=64)
    parser.add_argument("--apprentice_sampling_batch_size", type=int, default=8192)
    parser.add_argument("--expert_sampling_batch_size", type=int, default=8192)
    parser.add_argument("--apprentice_training_batch_size", type=int, default=256)
    parser.add_argument("--num_apprentice_training_steps", type=int, default=8)
    parser.add_argument("--num_jobs", type=int, default=8)
    parser.add_argument("--record_filtered", action="store_true")
    parser.add_argument("--genetic_experts", type=str, nargs="+", default=["SELFIES"])
    args = parser.parse_args()

    # Prepare CUDA device
    device = torch.device(0)

    neptune.init(project_qualified_name="",
                 api_token='',
                 )
    experiment = neptune.create_experiment(name="", params=vars(args))
    neptune.append_tag(args.benchmark_id)

    benchmark, scoring_num_list = load_benchmark(args.benchmark_id)

    char_dict = SmilesCharDictionary(dataset=args.dataset, max_smi_len=args.max_smiles_length)

    apprentice_storage = MaxRewardPriorityQueue()
    expert_storage = MaxRewardPriorityQueue()

    apprentice = SmilesGenerator.load(load_dir=args.apprentice_load_dir)
    apprentice = apprentice.to(device)
    apprentice_optimizer = Adam(apprentice.parameters(), lr=args.learning_rate)
    apprentice_handler = SmilesGeneratorHandler(
        model=apprentice,
        optimizer=apprentice_optimizer,
        char_dict=char_dict,
        max_sampling_batch_size=args.max_sampling_batch_size,
    )
    apprentice.train()

    expert_handler = load_genetic_experts(
        args.genetic_experts,
        args=args,
    )



    trainer = GeneticExpertGuidedLearningTrainer(
        apprentice_storage=apprentice_storage,
        expert_storage=expert_storage,
        apprentice_handler=apprentice_handler,
        expert_handler=expert_handler,
        char_dict=char_dict,
        num_keep=args.num_keep,
        apprentice_sampling_batch_size=args.apprentice_sampling_batch_size,
        expert_sampling_batch_size=args.expert_sampling_batch_size,
        apprentice_training_batch_size=args.apprentice_training_batch_size,
        num_apprentice_training_steps=args.num_apprentice_training_steps,
        init_smis=[],
    )

    recorder = Recorder(scoring_num_list=scoring_num_list, record_filtered=args.record_filtered)

    guacamol_generator = GeneticExpertGuidedLearningGenerator(
        trainer=trainer,
        recorder=recorder,
        num_steps=args.num_steps,
        device=device,
        scoring_num_list=scoring_num_list,
        num_jobs=args.num_jobs,
    )

    result = benchmark.assess_model(guacamol_generator)

    neptune.set_property("benchmark_score", result.score)
    
