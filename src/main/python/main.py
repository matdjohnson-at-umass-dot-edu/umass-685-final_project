import numpy
import torch

from src.main.python.package.data.dataset_transformer_setimesbyt5 import dataset_transformer_setimesbyt5
from src.main.python.package.data.dataset_utils import DatasetUtils
from src.main.python.package.runners.runner import Runner


# runner = Runner(runner_hyperparameters_name="SETimesByT5Vaswani2017Kocmi2018_0")
#
# runner.load_dataset()
# runner.load_model()
# runner.load_trainer()
# runner.run_trainer()

dataset_transformer = dataset_transformer_setimesbyt5(dataset_hyperparameters={"sentence_length_max_percentile": 95})

dataset_holder = dataset_transformer.read_dataset()

dataset_holder = DatasetUtils.create_dataset_segments(dataset_holder)

dataset_transformer.write_dataset_to_disk(dataset_holder)

