
from ..data.dataset_holder import DatasetHolder
import torch
import os

from ..utils import Utils


class Runner:

    def __init__(self,
                 model_parameter_directory="src/main/resources/model_parameters",
                 trainer_parameter_directory="src/main/resources/trainer_parameters",
                 runner_hyperparameters_name="SETimesByT5Vaswani2017Kocmi2018_0"):
        self.model_parameter_directory = model_parameter_directory
        self.trainer_parameter_directory = trainer_parameter_directory
        self.runner_hyperparameters_name = runner_hyperparameters_name
        self.runner_hyperparameters = Utils.load_python_object(
            'package.runners.runner_hyperparameters',
            runner_hyperparameters_name
        )
        self.dataset_holder: DatasetHolder = None
        self.model = None
        self.trainer = None
        print(f"Initialized runner {runner_hyperparameters_name} with parameters {self.runner_hyperparameters}")

    def load_dataset(self):
        dataset_transformer_name = self.runner_hyperparameters.get('dataset_transformer_name')
        dataset_hyperparameters = self.runner_hyperparameters.get('dataset_transformer_hyperparameters')
        _transformer_class_ = Utils.load_python_object('package.data.' + dataset_transformer_name, dataset_transformer_name)
        dataset_transformer = _transformer_class_(dataset_hyperparameters=dataset_hyperparameters)
        self.dataset_holder = dataset_transformer.read_dataset()
        self.dataset_holder = dataset_transformer.encode_dataset(self.dataset_holder)

    def load_model(self):
        model_name = self.runner_hyperparameters.get('model_name')
        model_hyperparameters = self.runner_hyperparameters.get('model_hyperparameters')
        model_hyperparameters['src_vocab_size'] = len(self.dataset_holder.get_source_vocab())
        model_hyperparameters['tgt_vocab_size'] = len(self.dataset_holder.get_target_vocab())
        model_hyperparameters['max_seq_len'] = self.dataset_holder.get_projected_max_seq_len()
        model_parameter_filepath = "{}/{}-{}-model.params".format(
            self.model_parameter_directory,
            model_name,
            self.runner_hyperparameters_name
        )
        _model_class_ = Utils.load_python_object('package.model.' + model_name, model_name)
        self.model = _model_class_(model_hyperparameters=model_hyperparameters)
        if os.path.exists(model_parameter_filepath):
            model_parameters = torch.load(model_parameter_filepath)
            self.model.load_state_dict(model_parameters)

    def load_trainer(self):
        trainer_name = self.runner_hyperparameters.get('trainer_name')
        trainer_hyperparameters = self.runner_hyperparameters.get('trainer_hyperparameters')
        _trainer_class_ = Utils.load_python_object('package.trainers.' + trainer_name, trainer_name)
        self.trainer = _trainer_class_(
            trainer_hyperparameters=trainer_hyperparameters,
            model_parameter_directory=self.model_parameter_directory,
            trainer_parameter_directory=self.trainer_parameter_directory,
            runner_hyperparameters_name=self.runner_hyperparameters_name
        )

    def run_trainer(self):
        # load from string, hyperparameter values, and previous stored parameters
        self.trainer.set_dataset_holder(self.dataset_holder)
        self.trainer.set_model(self.model)
        self.trainer.run_trainer()

