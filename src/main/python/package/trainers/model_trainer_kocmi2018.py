
import torch

from ..data.base_dataset_transformer import BaseDatasetTransformer
from ..model.base_hyperparameters import BaseHyperparameters
from .base_model_trainer import BaseModelTrainer


class ModelTrainerKocmi2018(BaseModelTrainer):

    def __init__(self):
        pass

    def set_hyperparameters(self, hyperparameters: BaseHyperparameters):
        pass

    def load_trainer(self):
        pass

    def run_trainer(self, model: torch.nn.Module, dataset_transformer: BaseDatasetTransformer):
        output = model.forward(
            dataset_transformer.source_encodings[0],
            dataset_transformer.target_encodings[0]
        )
        return output
