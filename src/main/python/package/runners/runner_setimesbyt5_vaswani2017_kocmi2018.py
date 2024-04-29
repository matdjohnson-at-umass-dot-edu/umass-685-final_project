import sys
import importlib

from .base_runner import BaseRunner
from ..data.dataset_transformer_setimesbyt5 import DatasetTransformerSetimesByt5
from ..model.setimesbyt5_vaswani2017_kocmi2018_ver0_hyperparameters import Setimesbyt5Vaswani2017Kocmi2018Ver0Hyperparameters
from ..trainers.model_trainer_kocmi2018 import ModelTrainerKocmi2018


class RunnerSetimesbyt5Vaswani2017Kocmi2018(BaseRunner):

    def __init__(self, training_run_number: int=0):
        super().__init__(
            dataset_transformer=DatasetTransformerSetimesByt5(),
            model_name='transformer_vaswani2017',
            trainer=ModelTrainerKocmi2018(),
            hyperparameters=Setimesbyt5Vaswani2017Kocmi2018Ver0Hyperparameters(),
            training_run_number=training_run_number
        )

    def load_model(self):
        # load from string, hyperparameter values, and previous stored parameters
        module = importlib.import_module('package.model.' + self.model_name)
        _model_class_ = getattr(module, self.model_name)
        self.model = _model_class_(
            src_vocab_size=len(self.dataset_transformer.source_vocab),
            tgt_vocab_size=len(self.dataset_transformer.target_vocab)
        )
