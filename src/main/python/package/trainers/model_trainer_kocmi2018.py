import time
import torch
import numpy as np

from .base_model_trainer import BaseModelTrainer
from ..data.dataset_utils import DatasetUtils
from ..utils import Utils


class model_trainer_kocmi2018(BaseModelTrainer):

    def __init__(self,
                 trainer_hyperparameters=None,
                 model_parameter_directory=None,
                 trainer_parameter_directory=None,
                 runner_hyperparameters_name=None):
        self.trainer_hyperparameters = trainer_hyperparameters
        self.model_parameter_directory = model_parameter_directory
        self.trainer_parameter_directory = trainer_parameter_directory
        self.runner_hyperparameters_name = runner_hyperparameters_name

    # pretraining is not used for monolingual english as described in Xue 2021 - ByT5 - Sec 3.1
    def run_trainer(self):
        optimizer_name = self.trainer_hyperparameters.get('optimizer_name')
        _optimizer_class_ = Utils.load_python_object('torch.optim', optimizer_name)
        initial_lr = self.trainer_hyperparameters.get('initial_lr')
        optimizer = _optimizer_class_(self.model.parameters(), lr=initial_lr)

        lr_scheduler_name = self.trainer_hyperparameters.get('lr_scheduler_name')
        _lr_scheduler_class_ = Utils.load_python_object('torch.optim.lr_scheduler', lr_scheduler_name)
        # constructor call assumes that the scheduler is the ExponentialLR scheduler
        lr_scheduler = _lr_scheduler_class_(optimizer, np.reciprocal(np.e))

        epochs = self.trainer_hyperparameters.get('epochs')
        batch_size = self.trainer_hyperparameters.get('batch_size')

        loss_fcn = torch.nn.NLLLoss()

        parameter_count = 0
        bytes_consumed = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter_count = parameter_count + np.prod(parameter.data.shape)
                bytes_consumed = bytes_consumed + parameter.data.nbytes
        gb_consumed = bytes_consumed / 1024 / 1024 / 1024

        print(f"Beginning training of model with parameter count {parameter_count} and parameter memory use {gb_consumed} GB")
        for i in range(0, epochs):
            epoch_start = time.time()
            print(f"Beginning epoch {i} of {epochs}")
            DatasetUtils.shuffle_dataset(self.dataset_holder)
            source_encoding_batches, target_encoding_batches = (
                DatasetUtils.prepare_batches(
                    self.dataset_holder,
                    batch_size
                )
            )
            assert len(source_encoding_batches) == len(target_encoding_batches)
            batch_ct = len(source_encoding_batches)
            label_values_to_add_for_batch_elements = torch.eye(batch_size)
            for j in range(0, batch_ct):
                batch_sequence_length = source_encoding_batches[j].shape[1]
                for k in range(1, batch_sequence_length-1):
                    print(f"Beginning step:{k}/{batch_sequence_length-1}, batch:{j}/{batch_ct}, epoch:{i}")
                    step_start = time.time()
                    target_batch_slices = torch.tensor_split(target_encoding_batches[j], [k], dim=1)
                    self.model.zero_grad()
                    output_logits = self.model.forward(
                        source_encoding_batches[j],
                        target_batch_slices[0]
                    )
                    next_word_indices = target_batch_slices[1][:, 1]
                    loss = loss_fcn(output_logits, next_word_indices)
                    loss.backward()
                    optimizer.step()
                    step_end = time.time()
                    print(f"Completed step {k}/{batch_sequence_length} in :{(step_end-step_start)}s")
                    print(f"epoch:{i+1}, batch:{j+1}/{batch_ct+1}, step:{k}/{batch_sequence_length} loss:{loss}")
            lr_scheduler.step()
            epoch_end = time.time()
            print(f"Completed epoch {i+1}/{epochs+1} in {(epoch_end - epoch_start) / 60 / 60}h")
            print(f"epoch:{i+1}, batch:{j+1}/{batch_ct+1}, loss:{loss}")
            torch.save(
                self.model.state_dict(),
                self.model_parameter_directory + "/" + self.runner_hyperparameters_name + "-model.params")
            torch.save(
                lr_scheduler.state_dict(),
                self.trainer_parameter_directory + "/" + self.runner_hyperparameters_name + "-trainer.params"
            )

