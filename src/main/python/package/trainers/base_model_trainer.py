from ..model.base_hyperparameters import BaseHyperparameters
from ..data.base_dataset_transformer import BaseDatasetTransformer

import torch


class BaseModelTrainer:

    def __init__(self, dataset_transformer):
        pass

    def set_hyperparameters(self, hyperparameters: BaseHyperparameters):
        pass

    def set_source_dataset(self, source_dataset):
        pass

    def set_target_dataset(self, target_dataset):
        pass

    def load_trainer(self):
        pass

    def run_trainer(self, model: torch.nn.Module, dataset_transformer: BaseDatasetTransformer):
        pass

