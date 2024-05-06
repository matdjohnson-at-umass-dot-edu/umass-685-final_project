import random
import torch

from src.main.python.package.data.dataset_holder import DatasetHolder


class DatasetUtils:

    @staticmethod
    def shuffle_dataset(dataset_holder: DatasetHolder):
        assert (len(dataset_holder.get_source_encodings()) == len(dataset_holder.get_target_encodings())
                == len(dataset_holder.get_source_sentences()) == len(dataset_holder.get_target_sentences()))
        dataset_element_shuffle_indices = list(range(0, len(dataset_holder.get_source_encodings())))
        random.shuffle(dataset_element_shuffle_indices)
        new_source_encodings = list()
        new_target_encodings = list()
        new_source_sentences = list()
        new_target_sentences = list()
        for i in dataset_element_shuffle_indices:
            new_source_encodings.append(dataset_holder.get_source_encodings()[i])
            new_target_encodings.append(dataset_holder.get_target_encodings()[i])
            new_source_sentences.append(dataset_holder.get_source_sentences()[i])
            new_target_sentences.append(dataset_holder.get_target_sentences()[i])
        assert (len(new_source_encodings) == len(new_target_encodings) == len(new_source_sentences)
                == len(new_target_sentences) == len(dataset_holder.get_source_encodings())
                == len(dataset_holder.get_target_encodings()) == len(dataset_holder.get_source_sentences())
                == len(dataset_holder.get_target_sentences()))
        dataset_holder.set_source_encodings(new_source_encodings)
        dataset_holder.set_target_encodings(new_target_encodings)
        dataset_holder.set_source_sentences(new_source_sentences)
        dataset_holder.set_target_sentences(new_target_sentences)
        return dataset_holder

    # use a dedicated padding token to pad batches as in Xue 2021 - ByT5 - Sec 3.1
    @staticmethod
    def prepare_batches(dataset_holder: DatasetHolder, batch_size: int):
        assert (len(dataset_holder.get_source_encodings()) == len(dataset_holder.get_target_encodings())
                == len(dataset_holder.get_source_sentences()) == len(dataset_holder.get_target_sentences()))
        total_elements = len(dataset_holder.get_source_encodings())
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
                source_encoding = dataset_holder.get_source_encodings()[j]
                target_encoding = dataset_holder.get_target_encodings()[j]
                source_encodings_tensors.append(
                    torch.nn.functional.pad(
                        source_encoding,
                        (0, dataset_holder.get_projected_max_seq_len() - len(source_encoding)),
                        value=dataset_holder.get_source_vocab().index(dataset_holder.get_padding_type())
                    )
                )
                target_encodings_tensors.append(
                    torch.nn.functional.pad(
                        target_encoding,
                        (0, dataset_holder.get_projected_max_seq_len() - len(target_encoding)),
                        value=dataset_holder.get_target_vocab().index(dataset_holder.get_padding_type())
                    )
                )
            source_encodings_batches.append(torch.stack(source_encodings_tensors))
            target_encodings_batches.append(torch.stack(target_encodings_tensors))
        return source_encodings_batches, target_encodings_batches

