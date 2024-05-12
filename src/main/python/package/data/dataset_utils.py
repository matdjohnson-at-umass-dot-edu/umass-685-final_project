import random

import numpy as np
import torch

from src.main.python.package.data.dataset_holder import DatasetHolder


class DatasetUtils:

    @staticmethod
    def shuffle_dataset(dataset_holder: DatasetHolder):
        new_source_encodings, new_target_encodings = (
            DatasetUtils.shuffle_encodings(
                dataset_holder.get_source_encodings(),
                dataset_holder.get_target_encodings()
            )
        )
        dataset_holder.set_source_encodings(new_source_encodings)
        dataset_holder.set_target_encodings(new_target_encodings)
        return dataset_holder

    @staticmethod
    def shuffle_training_dataset(dataset_holder: DatasetHolder):
        new_source_encodings, new_target_encodings = (
            DatasetUtils.shuffle_encodings(
                dataset_holder.get_source_encodings_validate(),
                dataset_holder.get_target_encodings_validate()
            )
        )
        dataset_holder.set_source_encodings(new_source_encodings)
        dataset_holder.set_target_encodings(new_target_encodings)
        return dataset_holder

    @staticmethod
    def shuffle_validation_dataset(dataset_holder: DatasetHolder):
        new_source_encodings, new_target_encodings = (
            DatasetUtils.shuffle_encodings(
                dataset_holder.get_source_encodings_validate(),
                dataset_holder.get_target_encodings_validate()
            )
        )
        dataset_holder.set_source_encodings(new_source_encodings)
        dataset_holder.set_target_encodings(new_target_encodings)
        return dataset_holder

    @staticmethod
    def shuffle_test_dataset(dataset_holder: DatasetHolder):
        new_source_encodings, new_target_encodings = (
            DatasetUtils.shuffle_encodings(
                dataset_holder.get_source_encodings_test(),
                dataset_holder.get_target_encodings_test()
            )
        )
        dataset_holder.set_source_encodings(new_source_encodings)
        dataset_holder.set_target_encodings(new_target_encodings)
        return dataset_holder

    @staticmethod
    def shuffle_encodings(source_encodings, target_encodings):
        assert len(source_encodings) == len(target_encodings)
        dataset_element_shuffle_indices = list(range(0, len(source_encodings)))
        random.shuffle(dataset_element_shuffle_indices)
        new_source_encodings = list()
        new_target_encodings = list()
        for i in dataset_element_shuffle_indices:
            new_source_encodings.append(source_encodings)
            new_target_encodings.append(target_encodings)
        assert (len(new_source_encodings) == len(new_target_encodings)
                == len(source_encodings) == len(target_encodings))
        return new_source_encodings, new_target_encodings

    @staticmethod
    def create_dataset_segments(dataset_holder: DatasetHolder):
        split_with_even_target_distribution = False
        iteration = 1
        best_split_target_encodings = None
        best_split_source_encodings = None
        best_split_deviation_from_desired = 1
        split_size = len(dataset_holder.get_target_encodings()) // 10
        train_size = split_size * 8
        while not split_with_even_target_distribution and iteration < 1000:
            dataset_holder = DatasetUtils.shuffle_dataset(dataset_holder)
            target_encodings = dataset_holder.get_target_encodings()
            source_encodings = dataset_holder.get_source_encodings()
            split_size = len(target_encodings) // 10
            train_size = split_size * 8
            train_set_target_enc = target_encodings[0:train_size]
            validation_set_target_enc = target_encodings[train_size:train_size+split_size]
            test_set_target_enc = target_encodings[train_size+split_size:]
            numpy_encodings = list()
            for encoding in train_set_target_enc:
                numpy_encodings.append(encoding.flatten().numpy())
            train_set_target_enc_cts = np.bincount(np.concatenate(numpy_encodings))
            numpy_encodings = list()
            for encoding in validation_set_target_enc:
                numpy_encodings.append(encoding.flatten().numpy())
            validation_set_target_enc_cts = np.bincount(np.concatenate(numpy_encodings))
            numpy_encodings = list()
            for encoding in test_set_target_enc:
                numpy_encodings.append(encoding.flatten().numpy())
            test_set_target_enc_cts = np.bincount(np.concatenate(numpy_encodings))
            # terms with probability ~ 1%
            total_5 = train_set_target_enc_cts[5] + validation_set_target_enc_cts[5] + test_set_target_enc_cts[5]
            total_40 = train_set_target_enc_cts[40] + validation_set_target_enc_cts[40] + test_set_target_enc_cts[40]
            total_42 = train_set_target_enc_cts[42] + validation_set_target_enc_cts[42] + test_set_target_enc_cts[42]
            # top 3 terms
            total_7 = train_set_target_enc_cts[7] + validation_set_target_enc_cts[7] + test_set_target_enc_cts[7]
            total_15 = train_set_target_enc_cts[15] + validation_set_target_enc_cts[15] + test_set_target_enc_cts[15]
            total_12 = train_set_target_enc_cts[12] + validation_set_target_enc_cts[12] + test_set_target_enc_cts[12]
            deviation_from_desired = (
                    np.abs(.8 - (train_set_target_enc_cts[5] / total_5)) +
                    np.abs(.8 - (train_set_target_enc_cts[40] / total_40)) +
                    np.abs(.8 - (train_set_target_enc_cts[42] / total_42)) +
                    np.abs(.8 - (train_set_target_enc_cts[7] / total_7)) +
                    np.abs(.8 - (train_set_target_enc_cts[15] / total_15)) +
                    np.abs(.8 - (train_set_target_enc_cts[12] / total_12)) +
                    np.abs(.1 - (validation_set_target_enc_cts[5] / total_5)) +
                    np.abs(.1 - (validation_set_target_enc_cts[40] / total_40)) +
                    np.abs(.1 - (validation_set_target_enc_cts[42] / total_42)) +
                    np.abs(.1 - (validation_set_target_enc_cts[7] / total_7)) +
                    np.abs(.1 - (validation_set_target_enc_cts[15] / total_15)) +
                    np.abs(.1 - (validation_set_target_enc_cts[12] / total_12)) +
                    np.abs(.1 - (test_set_target_enc_cts[5] / total_5)) +
                    np.abs(.1 - (test_set_target_enc_cts[40] / total_40)) +
                    np.abs(.1 - (test_set_target_enc_cts[42] / total_42)) +
                    np.abs(.1 - (test_set_target_enc_cts[7] / total_7)) +
                    np.abs(.1 - (test_set_target_enc_cts[15] / total_15)) +
                    np.abs(.1 - (test_set_target_enc_cts[12] / total_12))
            )
            if deviation_from_desired <= 18 * 0.0001:
                split_with_even_target_distribution = True
                best_split_target_encodings = target_encodings
                best_split_source_encodings = source_encodings
                print(f"Found dataset split within tolerance for deviation from uniform distribution over characters")
            if deviation_from_desired < best_split_deviation_from_desired:
                best_split_target_encodings = target_encodings
                best_split_source_encodings = source_encodings
                best_split_deviation_from_desired = deviation_from_desired
            print(f"Completed data split attempt. "
                  f"iteration:{iteration} "
                  f"best_split_deviation_from_desired:{best_split_deviation_from_desired}")
            iteration = iteration + 1
        dataset_holder.set_target_encodings(best_split_target_encodings)
        dataset_holder.set_target_encodings_train(best_split_target_encodings[0:train_size])
        dataset_holder.set_target_encodings_validate(best_split_target_encodings[train_size:train_size+split_size])
        dataset_holder.set_target_encodings_test(best_split_target_encodings[train_size+split_size:])
        dataset_holder.set_target_encodings(best_split_source_encodings)
        dataset_holder.set_source_encodings_train(best_split_source_encodings[0:train_size])
        dataset_holder.set_source_encodings_validate(best_split_source_encodings[train_size:train_size+split_size])
        dataset_holder.set_source_encodings_test(best_split_source_encodings[train_size+split_size:])
        return dataset_holder

    # use a dedicated padding token to pad batches as in Xue 2021 - ByT5 - Sec 3.1
    @staticmethod
    def prepare_batches(
            source_encodings,
            target_encodings,
            source_vocab,
            target_vocab,
            batch_size: int,
            pad_length: int,
            padding_value):
        assert len(source_encodings) == len(target_encodings)
        total_elements = len(source_encodings)
        batch_ct = (total_elements // batch_size) + 1
        source_encodings_batches = list()
        target_encodings_batches = list()
        for i in range(0, batch_ct):
            source_encodings_tensors = list()
            target_encodings_tensors = list()
            batch_range = None
            if i < batch_ct - 1:
                batch_range = range(i*batch_size, (i+1)*batch_size)
            elif i == batch_ct - 1:
                batch_range = range(i*batch_size, total_elements)
            for j in batch_range:
                source_encoding = source_encodings[j]
                target_encoding = target_encodings[j]
                source_encodings_tensors.append(
                    torch.nn.functional.pad(
                        source_encoding,
                        (0, pad_length - len(source_encoding)),
                        value=source_vocab.index(padding_value)
                    )
                )
                target_encodings_tensors.append(
                    torch.nn.functional.pad(
                        target_encoding,
                        (0, pad_length - len(target_encoding)),
                        value=target_vocab.index(padding_value)
                    )
                )
            source_encodings_batches.append(torch.stack(source_encodings_tensors))
            target_encodings_batches.append(torch.stack(target_encodings_tensors))
        return source_encodings_batches, target_encodings_batches

    @staticmethod
    def prepare_training_batches(
            dataset_holder: DatasetHolder,
            batch_size: int):
        source_encodings_batches, target_encodings_batches = DatasetUtils.prepare_batches(
            dataset_holder.get_source_encodings_train(),
            dataset_holder.get_target_encodings_train(),
            dataset_holder.get_source_vocab(),
            dataset_holder.get_target_vocab(),
            batch_size,
            dataset_holder.get_max_seq_obs(),
            dataset_holder.get_padding_type()
        )
        return source_encodings_batches, target_encodings_batches

