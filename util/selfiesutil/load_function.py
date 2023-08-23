import logging
from typing import List, Optional, Union

import torch
import sys
sys.path.append("../../")

from model.genetic_expert import GeneticOperatorHandler



def load_genetic_experts(
    expert_types: List[str],
    args,
) -> List[GeneticOperatorHandler]:
    experts = []
    for ge_type in expert_types:
        expert_handler = GeneticOperatorHandler(
            crossover_type=ge_type,
            mutation_type=ge_type,
            mutation_initial_rate=args.mutation_rate,
        )
        experts.append(expert_handler)

    return experts