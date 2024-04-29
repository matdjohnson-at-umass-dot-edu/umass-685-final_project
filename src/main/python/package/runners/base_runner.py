
from ..data.base_dataset_transformer import BaseDatasetTransformer
from ..model.base_hyperparameters import BaseHyperparameters
from ..trainers.base_model_trainer import BaseModelTrainer


class BaseRunner:

    def __init__(self,
                 dataset_transformer: BaseDatasetTransformer=None,
                 model_name: str=None,
                 trainer: BaseModelTrainer=None,
                 hyperparameters: BaseHyperparameters=None,
                 training_run_number: int=0):
        self.dataset_transformer = dataset_transformer
        self.model_name = model_name
        self.trainer = trainer
        self.hyperparameters = hyperparameters
        self.model = None

    def load_dataset(self):
        self.dataset_transformer.read_dataset()
        self.dataset_transformer.encode_dataset()

    def load_model(self):
        # load from hyperparameter values and previous stored parameters
        module = __import__('package.model.' + self.model_name)
        _model_class_ = getattr(module, self.model_name)
        self.model = _model_class_()

    def run_trainer(self):
        # load from string, hyperparameter values, and previous stored parameters
        self.trainer.set_hyperparameters(self.hyperparameters)
        self.trainer.set_source_dataset(self.dataset_transformer.get_source_encodings())
        self.trainer.set_target_dataset(self.dataset_transformer.get_target_encodings())
        self.trainer.load_trainer()
        self.trainer.run_trainer(self.model, self.dataset_transformer)

