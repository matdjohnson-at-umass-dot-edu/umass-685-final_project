import torch
import os
import numpy as np
import random
from typing import Optional
import time


root_filepath = "src/main/"
is_remote_execution = False

SETimesByT5Vaswani2017Kocmi2018_0 = {
    'dataset_transformer_name': 'dataset_transformer_setimesbyt5',
    'model_name': 'transformer_vaswani2017',
    'trainer_name': 'model_trainer_kocmi2018',
    # corresponds to dictionary 'get' calls in the dataset_loader constructor
    'dataset_transformer_hyperparameters': {},
    # corresponds to dictionary 'get' calls in the model constructor
    'model_hyperparameters': {
        'd_model': 512,
        'nhead': 8,
        # number of encoders is 3 times that of decoders, following Xue 2021 - ByT5 - Sec 3.1
        'num_encoder_layers': 9,
        'num_decoder_layers': 3,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': torch.nn.functional.relu,
        'custom_encoder': None,
        'custom_decoder': None,
        'layer_norm_eps': 1e-5,
        'batch_first': True,
        'norm_first': False,
        'bias': True,
        'device': None,
        'dtype': None
    },
    # corresponds to dictionary 'get' calls in the trainer constructor
    'trainer_hyperparameters': {
        # optimization and lr schedule following Kocmi 2018 - Trivial TL - Sec 3
        'optimizer_name': 'Adam',
        'lr_scheduler_name': 'ExponentialLR',
        'initial_lr': 0.2,
        'epochs': 1,
        'batch_size': 10
    }
}

SETimesByT5Vaswani2017Kocmi2018_1 = {
    'dataset_transformer_name': 'dataset_transformer_setimesbyt5',
    'model_name': 'transformer_vaswani2017',
    'trainer_name': 'model_trainer_kocmi2018',
    # corresponds to dictionary 'get' calls in the dataset_loader constructor
    'dataset_transformer_hyperparameters': {
        'sentence_length_max_percentile': 95
    },
    # corresponds to dictionary 'get' calls in the model constructor
    'model_hyperparameters': {
        'd_model': 256,
        'nhead': 8,
        # number of encoders is 3 times that of decoders, following Xue 2021 - ByT5 - Sec 3.1
        'num_encoder_layers': 9,
        'num_decoder_layers': 3,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'activation': torch.nn.functional.relu,
        'custom_encoder': None,
        'custom_decoder': None,
        'layer_norm_eps': 1e-5,
        'batch_first': True,
        'norm_first': False,
        'bias': True,
        'device': None,
        'dtype': None
    },
    # corresponds to dictionary 'get' calls in the trainer constructor
    'trainer_hyperparameters': {
        # optimization and lr schedule following Kocmi 2018 - Trivial TL - Sec 3
        'optimizer_name': 'Adam',
        'lr_scheduler_name': 'ExponentialLR',
        'initial_lr': 0.2,
        'epochs': 5,
        'batch_size': 500
    }
}


class DatasetHolder:

    def __init__(self):
        self.unknown_vocabulary_type = None
        self.padding_vocabulary_type = None
        self.end_of_sequence_type = None
        self.target_vocab = None
        self.target_vocab_tensor = None
        self.source_vocab = None
        self.source_vocab_tensor = None
        self.target_encodings = None
        self.target_encodings_train = None
        self.target_encodings_validate = None
        self.target_encodings_test = None
        self.source_encodings = None
        self.source_encodings_train = None
        self.source_encodings_validate = None
        self.source_encodings_test = None
        self.max_seq_obs = 0

    def get_unknown_vocabulary_type(self):
        return self.unknown_vocabulary_type

    def set_unknown_vocabulary_type(self, unknown_vocabulary_type):
        self.unknown_vocabulary_type = unknown_vocabulary_type

    def get_padding_vocabulary_type(self):
        return self.padding_vocabulary_type

    def set_padding_vocabulary_type(self, padding_vocabulary_type):
        self.padding_vocabulary_type = padding_vocabulary_type

    def get_end_of_sequence_vocabulary_type(self):
        return self.end_of_sequence_type

    def set_end_of_sequence_vocabulary_type(self, end_of_sequence_type):
        self.end_of_sequence_type = end_of_sequence_type

    def get_target_vocab(self):
        return self.target_vocab

    def get_target_vocab_numpy(self):
        if self.target_vocab_tensor is None:
            self.target_vocab_tensor = np.array(self.target_vocab)
        return self.target_vocab_tensor

    def set_target_vocab(self, target_vocab):
        self.target_vocab = target_vocab

    def get_source_vocab(self):
        return self.source_vocab

    def get_source_vocab_numpy(self):
        if self.source_vocab_tensor is None:
            self.source_vocab_tensor = np.array(self.source_vocab)
        return self.source_vocab_tensor

    def set_source_vocab(self, source_vocab):
        self.source_vocab = source_vocab

    def get_target_encodings(self):
        return self.target_encodings

    def set_target_encodings(self, target_encodings):
        self.target_encodings = target_encodings

    def get_source_encodings(self):
        return self.source_encodings

    def set_source_encodings(self, source_encodings):
        self.source_encodings = source_encodings

    def get_target_encodings_train(self):
        return self.target_encodings_train

    def set_target_encodings_train(self, target_encodings_train):
        self.target_encodings_train = target_encodings_train

    def get_source_encodings_train(self):
        return self.source_encodings_train

    def set_source_encodings_train(self, source_encodings_train):
        self.source_encodings_train = source_encodings_train

    def get_target_encodings_validate(self):
        return self.target_encodings_validate

    def set_target_encodings_validate(self, target_encodings_validate):
        self.target_encodings_validate = target_encodings_validate

    def get_source_encodings_validate(self):
        return self.source_encodings_validate

    def set_source_encodings_validate(self, source_encodings_validate):
        self.source_encodings_validate = source_encodings_validate

    def get_target_encodings_test(self):
        return self.target_encodings_test

    def set_target_encodings_test(self, target_encodings_test):
        self.target_encodings_test = target_encodings_test

    def get_source_encodings_test(self):
        return self.source_encodings_test

    def set_source_encodings_test(self, source_encodings_test):
        self.source_encodings_test = source_encodings_test

    def get_max_seq_obs(self):
        return self.max_seq_obs

    def set_max_seq_obs(self, max_seq_obs):
        self.max_seq_obs = max_seq_obs


# class name matches file name
class dataset_transformer_setimesbyt5():

    def __init__(self,
                 datasets_directory=root_filepath+"resources",
                 raw_dataset_directory="raw_datasets/setimes",
                 parsed_dataset_directory="parsed_datasets/setimes",
                 ids_filename='SETIMES.en-tr.ids',
                 en_filename='SETIMES.en-tr.en',
                 tr_filename='SETIMES.en-tr.tr',
                 dataset_hyperparameters=None):
        self.datasets_directory = datasets_directory
        self.raw_dataset_directory = raw_dataset_directory
        self.parsed_dataset_directory = parsed_dataset_directory
        self.ids_filename = ids_filename
        self.en_filename = en_filename
        self.tr_filename = tr_filename
        self.parsed_dataset_filename = None
        if 'parsed_dataset_filename' in dataset_hyperparameters:
            self.parsed_dataset_filename = dataset_hyperparameters['parsed_dataset_filename']
        self.sentence_length_max_percentile = None
        if 'sentence_length_max_percentile' in dataset_hyperparameters:
            self.sentence_length_max_percentile = dataset_hyperparameters['sentence_length_max_percentile']

    def read_dataset(self):
        dataset_holder = None
        if self.parsed_dataset_filename is not None:
            dataset_holder = torch.load(self.datasets_directory + "/"
                                        + self.parsed_dataset_directory + "/"
                                        + self.parsed_dataset_filename)
        else:
            target_sentences = list()
            source_sentences = list()
            index_file = open(self.datasets_directory + "/" + self.raw_dataset_directory + "/" + self.ids_filename)
            en_file = open(self.datasets_directory + "/" + self.raw_dataset_directory + "/" + self.en_filename)
            tr_file = open(self.datasets_directory + "/" + self.raw_dataset_directory + "/" + self.tr_filename)
            indices = list()
            en_sentences = list()
            tr_sentences = list()
            line_number = 1
            for line in index_file:
                line_segments = line.strip().split()
                if len(line_segments) != 4:
                    print("Line segmentation error on line " + str(line_number))
                    print("Content: " + line)
                    continue
                if line_segments[0].startswith("en") and line_segments[1].startswith("tr"):
                    indices.append((int(line_segments[2]), int(line_segments[3])))
                elif line_segments[0].startswith("tr") and line_segments[1].startswith("en"):
                    indices.append((int(line_segments[3]), int(line_segments[2])))
                else:
                    print("Index parsing error on line " + str(line_number))
                    print("Content: " + line)
                    continue
                line_number = line_number + 1
            for line in en_file:
                en_sentences.append(line.strip())
            for line in tr_file:
                tr_sentences.append(line.strip())
            for index in indices:
                target_sentences.append(en_sentences[index[0] - 1])
                source_sentences.append(tr_sentences[index[1] - 1])
            target_sentence_lengths = list()
            for sentence in target_sentences:
                target_sentence_lengths.append(len(sentence))
            source_sentence_lengths = list()
            for sentence in source_sentences:
                source_sentence_lengths.append(len(sentence))
            target_sentences_length_limited = list()
            source_sentences_length_limited = list()
            target_max_len = int(np.percentile(sorted(target_sentence_lengths), self.sentence_length_max_percentile))
            source_max_len = int(np.percentile(sorted(source_sentence_lengths), self.sentence_length_max_percentile))
            max_seq_obs = 0
            for i in range(0, len(target_sentences)):
                if len(target_sentences[i]) <= target_max_len and len(source_sentences[i]) <= source_max_len:
                    if len(target_sentences[i]) > max_seq_obs:
                        max_seq_obs = len(target_sentences[i])
                    target_sentences_length_limited.append(target_sentences[i])
                    source_sentences_length_limited.append(source_sentences[i])
            dataset_holder = DatasetHolder()
            dataset_holder.set_max_seq_obs(max_seq_obs)
            # encode to Pytorch tensors as raw UTF-8 character vocabulary
            # method replicated from Xue 2021 - ByT5 - Introduction, sec 3.1
            unknown_vocabulary_type = '<unk>'
            padding_vocabulary_type = '<pad>'
            end_of_sequence_vocabulary_type = '<eos>'
            dataset_holder.set_unknown_vocabulary_type(unknown_vocabulary_type)
            dataset_holder.set_padding_vocabulary_type(padding_vocabulary_type)
            dataset_holder.set_end_of_sequence_vocabulary_type(end_of_sequence_vocabulary_type)
            target_vocab = list([unknown_vocabulary_type, padding_vocabulary_type, end_of_sequence_vocabulary_type])
            source_vocab = list([unknown_vocabulary_type, padding_vocabulary_type, end_of_sequence_vocabulary_type])
            target_encodings = list()
            source_encodings = list()
            for entry in target_sentences_length_limited:
                encoding = list()
                for character in entry:
                    if character not in target_vocab:
                        target_vocab.append(character)
                    encoding.append(target_vocab.index(character))
                encoding.append(target_vocab.index(end_of_sequence_vocabulary_type))
                target_encodings.append(torch.tensor(encoding))
            for entry in source_sentences_length_limited:
                encoding = list()
                for character in entry:
                    if character not in source_vocab:
                        source_vocab.append(character)
                    encoding.append(source_vocab.index(character))
                encoding.append(target_vocab.index(end_of_sequence_vocabulary_type))
                source_encodings.append(torch.tensor(encoding))
            # fix vocabulary indices using tuple type
            dataset_holder.set_target_vocab(tuple(target_vocab))
            dataset_holder.set_target_encodings(target_encodings)
            dataset_holder.set_source_vocab(tuple(source_vocab))
            dataset_holder.set_source_encodings(source_encodings)
            dataset_holder = DatasetUtils.create_dataset_segments(dataset_holder)
        return dataset_holder

    def write_dataset_to_disk(self, dataset_holder: DatasetHolder):
        torch.save(dataset_holder,
                   self.datasets_directory + "/" +
                   self.parsed_dataset_directory + "/" +
                   "setimes_parsed-" + str(int(time.time())))


class Utils:

    def __init__(self):
        pass

    @staticmethod
    def load_python_object(object_path: str, object_attribute: str):
        path_segments = object_path.split('.')
        module = __import__(object_path)
        for segment in path_segments[1:]:
            module = getattr(module, segment)
        return getattr(module, object_attribute)


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
            new_source_encodings.append(source_encodings[i])
            new_target_encodings.append(target_encodings[i])
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
            segments = 20
            split_size = len(target_encodings) // segments
            train_size = split_size * (segments - 2)
            train_set_target_enc = target_encodings[0:train_size]
            validation_set_target_enc = target_encodings[train_size:train_size+split_size]
            test_set_target_enc = target_encodings[train_size+split_size:]
            numpy_encodings = list()
            for encoding in train_set_target_enc:
                numpy_encodings.append(encoding.flatten().numpy())
            train_set_target_enc_cts = np.bincount(
                np.concatenate([
                    np.concatenate(numpy_encodings),
                    np.arange(0, 170)
                ])
            )
            numpy_encodings = list()
            for encoding in validation_set_target_enc:
                numpy_encodings.append(encoding.flatten().numpy())
            validation_set_target_enc_cts = np.bincount(
                np.concatenate([
                    np.concatenate(numpy_encodings),
                    np.arange(0, 170)
                ])
            )
            numpy_encodings = list()
            for encoding in test_set_target_enc:
                numpy_encodings.append(encoding.flatten().numpy())
            test_set_target_enc_cts = np.bincount(
                np.concatenate([
                    np.concatenate(numpy_encodings),
                    np.arange(0, 170)
                ])
            )
            # terms with probability ~ 1%
            total_5 = train_set_target_enc_cts[5] + validation_set_target_enc_cts[5] + test_set_target_enc_cts[5]
            total_40 = train_set_target_enc_cts[40] + validation_set_target_enc_cts[40] + test_set_target_enc_cts[40]
            total_42 = train_set_target_enc_cts[42] + validation_set_target_enc_cts[42] + test_set_target_enc_cts[42]
            # top 3 terms
            total_7 = train_set_target_enc_cts[7] + validation_set_target_enc_cts[7] + test_set_target_enc_cts[7]
            total_15 = train_set_target_enc_cts[15] + validation_set_target_enc_cts[15] + test_set_target_enc_cts[15]
            total_12 = train_set_target_enc_cts[12] + validation_set_target_enc_cts[12] + test_set_target_enc_cts[12]
            train_dist_goal = (segments - 2)/segments
            val_test_dist_goal = (1 / segments)
            deviation_from_desired = (
                    np.abs(((segments - 2)/segments) - (train_set_target_enc_cts[5] / total_5)) +
                    np.abs(train_dist_goal - (train_set_target_enc_cts[40] / total_40)) +
                    np.abs(train_dist_goal - (train_set_target_enc_cts[42] / total_42)) +
                    np.abs(train_dist_goal - (train_set_target_enc_cts[7] / total_7)) +
                    np.abs(train_dist_goal - (train_set_target_enc_cts[15] / total_15)) +
                    np.abs(train_dist_goal - (train_set_target_enc_cts[12] / total_12)) +
                    np.abs(val_test_dist_goal - (validation_set_target_enc_cts[5] / total_5)) +
                    np.abs(val_test_dist_goal - (validation_set_target_enc_cts[40] / total_40)) +
                    np.abs(val_test_dist_goal - (validation_set_target_enc_cts[42] / total_42)) +
                    np.abs(val_test_dist_goal - (validation_set_target_enc_cts[7] / total_7)) +
                    np.abs(val_test_dist_goal - (validation_set_target_enc_cts[15] / total_15)) +
                    np.abs(val_test_dist_goal - (validation_set_target_enc_cts[12] / total_12)) +
                    np.abs(val_test_dist_goal - (test_set_target_enc_cts[5] / total_5)) +
                    np.abs(val_test_dist_goal - (test_set_target_enc_cts[40] / total_40)) +
                    np.abs(val_test_dist_goal - (test_set_target_enc_cts[42] / total_42)) +
                    np.abs(val_test_dist_goal - (test_set_target_enc_cts[7] / total_7)) +
                    np.abs(val_test_dist_goal - (test_set_target_enc_cts[15] / total_15)) +
                    np.abs(val_test_dist_goal - (test_set_target_enc_cts[12] / total_12))
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
            dataset_holder.get_padding_vocabulary_type()
        )
        return source_encodings_batches, target_encodings_batches

    @staticmethod
    def decode_target_tensor(dataset_holder: DatasetHolder, tensor_to_decode):
        vocab = dataset_holder.get_target_vocab_numpy()
        decoded_tensor = np.take(vocab, tensor_to_decode.flatten().numpy())
        return "".join(decoded_tensor.tolist())

    @staticmethod
    def decode_source_tensor(dataset_holder: DatasetHolder, tensor_to_decode):
        vocab = dataset_holder.get_source_vocab_numpy()
        decoded_tensor = np.take(vocab, tensor_to_decode.flatten().numpy())
        return "".join(decoded_tensor.tolist())


# class name matches file name
class transformer_vaswani2017(torch.nn.Transformer):

    def __init__(self,
                 model_hyperparameters):
        super().__init__(
            d_model=model_hyperparameters['d_model'],
            nhead=model_hyperparameters['nhead'],
            num_encoder_layers=model_hyperparameters['num_encoder_layers'],
            num_decoder_layers=model_hyperparameters['num_decoder_layers'],
            dim_feedforward=model_hyperparameters['dim_feedforward'],
            dropout=model_hyperparameters['dropout'],
            activation=model_hyperparameters['activation'],
            custom_encoder=model_hyperparameters['custom_encoder'],
            custom_decoder=model_hyperparameters['custom_decoder'],
            layer_norm_eps=model_hyperparameters['layer_norm_eps'],
            batch_first=model_hyperparameters['batch_first'],
            norm_first=model_hyperparameters['norm_first'],
            bias=model_hyperparameters['bias'],
            device=model_hyperparameters['device'],
            dtype=model_hyperparameters['dtype']
        )
        self.max_seq_len = model_hyperparameters['max_seq_len']
        self.src_embeddings = torch.nn.Embedding(
            model_hyperparameters['src_vocab_size'],
            model_hyperparameters['d_model']
        )
        self.tgt_embeddings = torch.nn.Embedding(
            model_hyperparameters['tgt_vocab_size'],
            model_hyperparameters['d_model']
        )
        self.linear_output_projection_1 = torch.nn.Linear(
            self.max_seq_len,
            1,
            bias=False
        )
        self.linear_output_projection_2 = torch.nn.Linear(
            model_hyperparameters['d_model'],
            model_hyperparameters['tgt_vocab_size'],
            bias=False
        )
        self.logsoftmax_output = torch.nn.LogSoftmax(dim=1)
        self.model_hyperparameters = model_hyperparameters
        self.tgt_mask_cache = {}

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                src_is_causal: Optional[bool] = None,
                tgt_is_causal: Optional[bool] = True,
                memory_is_causal: bool = False) -> torch.Tensor:
        src_embedding = self.src_embeddings(src)
        tgt_embedding = self.tgt_embeddings(tgt)
        tgt_mask = self.get_tgt_mask(tgt.shape[1])
        transformer_output = super().forward(src_embedding, tgt_embedding, src_mask,
                                             tgt_mask, memory_mask, src_key_padding_mask,
                                             tgt_key_padding_mask, memory_key_padding_mask,
                                             src_is_causal, tgt_is_causal, memory_is_causal)
        transformer_output = torch.swapaxes(transformer_output, -1, -2)
        transformer_output = torch.nn.functional.pad(
            transformer_output,
            (0, self.max_seq_len - transformer_output.shape[2]),
            value=0
        )
        output = self.logsoftmax_output(
            self.linear_output_projection_2(
                torch.squeeze(
                    self.linear_output_projection_1(
                        transformer_output
                    )
                )
            )
        )
        del transformer_output
        return output

    def set_tgt_mask_cache(self, tgt_mask_cache):
        self.tgt_mask_cache = tgt_mask_cache

    def get_tgt_mask(self, mask_len):
        return self.tgt_mask_cache[mask_len]


class model_trainer_kocmi2018():

    def __init__(self,
                 trainer_hyperparameters=None,
                 model_parameter_directory=None,
                 trainer_parameter_directory=None,
                 runner_hyperparameters_name=None):
        self.trainer_hyperparameters = trainer_hyperparameters
        self.optimizer_name = self.trainer_hyperparameters['optimizer_name']
        self.initial_lr = self.trainer_hyperparameters['initial_lr']
        self.lr_scheduler_name = self.trainer_hyperparameters['lr_scheduler_name']
        self.epochs = self.trainer_hyperparameters['epochs']
        self.batch_size = self.trainer_hyperparameters['batch_size']
        self.model_parameter_directory = model_parameter_directory
        self.trainer_parameter_directory = trainer_parameter_directory
        self.runner_hyperparameters_name = runner_hyperparameters_name
        self.dataset_holder = None
        self.model = None

    # pretraining is not used for monolingual english as described in Xue 2021 - ByT5 - Sec 3.1
    def run_trainer(self):
        if is_remote_execution:
            torch.cuda.empty_cache()
            self.model.cuda()
        tgt_mask_cache = {}
        for i in range(1, self.model.max_seq_len + 1):
            if is_remote_execution:
                tgt_mask_cache[i] = self.model.generate_square_subsequent_mask(i, device="cuda:0")
            else:
                tgt_mask_cache[i] = self.model.generate_square_subsequent_mask(i)
        self.model.set_tgt_mask_cache(tgt_mask_cache)
        _optimizer_class_ = Utils.load_python_object('torch.optim', self.optimizer_name)
        optimizer = _optimizer_class_(self.model.parameters(), lr=self.initial_lr)
        _lr_scheduler_class_ = Utils.load_python_object('torch.optim.lr_scheduler', self.lr_scheduler_name)
        # constructor call assumes that the scheduler is the ExponentialLR scheduler
        lr_scheduler = _lr_scheduler_class_(optimizer, np.reciprocal(np.e))
        loss_fcn = torch.nn.NLLLoss()

        parameter_count = 0
        bytes_consumed = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter_count = parameter_count + np.prod(parameter.data.shape)
                bytes_consumed = bytes_consumed + parameter.data.nbytes
        gb_consumed = bytes_consumed / 1024 / 1024 / 1024

        print(f"Beginning training of model with parameter count {parameter_count} "
              f"and parameter memory use {gb_consumed} GB")
        for i in range(0, self.epochs):
            epoch_start = time.time()
            print(f"Beginning epoch {i+1} of {self.epochs}")
            self.dataset_holder = DatasetUtils.shuffle_training_dataset(self.dataset_holder)
            source_encoding_batches, target_encoding_batches = (
                DatasetUtils.prepare_training_batches(
                    self.dataset_holder,
                    self.batch_size
                )
            )
            if is_remote_execution:
                source_batches = list()
                target_batches = list()
                for batch in source_encoding_batches:
                    source_batches.append(batch.to(device="cuda:0"))
                    del batch
                del source_encoding_batches
                for batch in target_encoding_batches:
                    target_batches.append(batch.to(device="cuda:0"))
                    del batch
                del target_encoding_batches
            else:
                source_batches = source_encoding_batches
                target_batches = target_encoding_batches
            assert len(source_batches) == len(target_batches)
            batch_ct = len(source_batches)
            for j in range(0, batch_ct):
                batch_start = time.time()
                batch_sequence_length = source_batches[j].shape[1]
                for k in range(1, batch_sequence_length-1):
                    step_start = time.time()
                    target_batch_slices = torch.tensor_split(target_batches[j], [k], dim=1)
                    self.model.zero_grad()
                    output_logits = self.model.forward(
                        source_batches[j],
                        target_batch_slices[0]
                    )
                    next_word_indices = target_batch_slices[1][:, 0]
                    loss = loss_fcn(output_logits, next_word_indices)
                    loss.backward()
                    optimizer.step()
                    step_end = time.time()
                    if k % 100 == 0:
                        print(f"Completed step.")
                        print(f"epoch:{i+1}/{self.epochs+1}, batch:{j+1}/{batch_ct}, step:{k}/{batch_sequence_length}, "
                              f"loss:{loss}, time_for_individual_step:{step_end-step_start}")
                        full_sequence = DatasetUtils.decode_target_tensor(self.dataset_holder, target_batches[j][0])
                        prefix_sequence = DatasetUtils.decode_target_tensor(
                            self.dataset_holder,
                            target_batch_slices[0][0]
                        )
                        next_token = DatasetUtils.decode_target_tensor(
                            self.dataset_holder,
                            next_word_indices[0]
                        )
                        predicted_token = DatasetUtils.decode_target_tensor(
                            self.dataset_holder,
                            torch.argmax(output_logits[0])
                        )
                        print("Attempted to predict next token for step:")
                        print(f"full seq: {full_sequence}")
                        print(f"pref seq: {prefix_sequence}")
                        print(f"next tok: {next_token.ljust(k, ' ')}")
                        print(f"pred tok: {predicted_token.ljust(k, ' ')}")
                batch_end = time.time()
                print(f"Completed batch.")
                print(f"epoch:{i+1}/{self.epochs+1} batch:{j+1}/{batch_ct} time:{(batch_end-batch_start) / 60 }m")
                if is_remote_execution:
                    print(f"Memory usage summary:")
                    print(f"{torch.cuda.memory_summary()}")
                    torch.cuda.empty_cache()
                param_filename_tag = str(int(time.time()))
                torch.save(
                    self.model.state_dict(),
                    self.model_parameter_directory + "/" + self.runner_hyperparameters_name + "-" + param_filename_tag + "-model.params"
                )
                torch.save(
                    lr_scheduler.state_dict(),
                    self.trainer_parameter_directory + "/" + self.runner_hyperparameters_name + "-" + param_filename_tag + "-scheduler.params"
                )
                torch.save(
                    f"epoch:{i+1}/{self.epochs+1} batch:{j+1}/{batch_ct}",
                    self.trainer_parameter_directory + "/" + self.runner_hyperparameters_name + "-" + param_filename_tag + "-trainer.params"
                )
            lr_scheduler.step()
            epoch_end = time.time()
            print(f"Completed epoch {i+1}/{self.epochs+1} in {(epoch_end - epoch_start) / 60 }m")
            print(f"epoch:{i+1}, batch:{j+1}/{batch_ct+1}, loss:{loss}")

    def get_dataset_holder(self):
        return self.dataset_holder

    def set_dataset_holder(self, dataset_holder):
        self.dataset_holder = dataset_holder

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model


class Runner:

    def __init__(self,
                 model_parameter_directory=root_filepath+"resources/model_parameters",
                 trainer_parameter_directory=root_filepath+"resources/trainer_parameters",
                 runner_hyperparameters_name="SETimesByT5Vaswani2017Kocmi2018_1"):
        self.model_parameter_directory = model_parameter_directory
        self.trainer_parameter_directory = trainer_parameter_directory
        self.runner_hyperparameters_name = runner_hyperparameters_name
        self.runner_hyperparameters = SETimesByT5Vaswani2017Kocmi2018_1
        self.dataset_holder: DatasetHolder = None
        self.model = None
        self.trainer = None
        print(f"Initialized runner {runner_hyperparameters_name} with parameters {self.runner_hyperparameters}")

    def load_dataset(self):
        dataset_transformer_name = self.runner_hyperparameters.get('dataset_transformer_name')
        dataset_hyperparameters = self.runner_hyperparameters.get('dataset_transformer_hyperparameters')
        dataset_transformer = dataset_transformer_setimesbyt5(dataset_hyperparameters=dataset_hyperparameters)
        self.dataset_holder = dataset_transformer.read_dataset()

    def load_model(self):
        model_name = self.runner_hyperparameters.get('model_name')
        model_hyperparameters = self.runner_hyperparameters.get('model_hyperparameters')
        model_hyperparameters['src_vocab_size'] = len(self.dataset_holder.get_source_vocab())
        model_hyperparameters['tgt_vocab_size'] = len(self.dataset_holder.get_target_vocab())
        model_hyperparameters['max_seq_len'] = self.dataset_holder.get_max_seq_obs()
        model_parameter_filepath = root_filepath+"{}/{}-{}-model.params".format(
            self.model_parameter_directory,
            model_name,
            self.runner_hyperparameters_name
        )
        self.model = transformer_vaswani2017(model_hyperparameters=model_hyperparameters)
        if os.path.exists(model_parameter_filepath):
            model_parameters = torch.load(model_parameter_filepath)
            self.model.load_state_dict(model_parameters)

    def load_trainer(self):
        trainer_name = self.runner_hyperparameters.get('trainer_name')
        trainer_hyperparameters = self.runner_hyperparameters.get('trainer_hyperparameters')
        self.trainer = model_trainer_kocmi2018(
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


runner = Runner(runner_hyperparameters_name="SETimesByT5Vaswani2017Kocmi2018_1")

runner.load_dataset()
runner.load_model()
runner.load_trainer()
runner.run_trainer()
