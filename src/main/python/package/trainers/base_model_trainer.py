from ..data.dataset_holder import DatasetHolder

import torch


class BaseModelTrainer:

    def __init__(self):
        self.dataset_holder: DatasetHolder = None
        self.model: torch.nn.Module = None

    def get_dataset_holder(self):
        return self.dataset_holder

    def set_dataset_holder(self, dataset_holder):
        self.dataset_holder = dataset_holder

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def run_trainer(self):
        pass

