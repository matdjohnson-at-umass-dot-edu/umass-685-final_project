import time

import numpy as np
import torch

from .dataset_holder import DatasetHolder
from .dataset_utils import DatasetUtils


# class name matches file name
class dataset_transformer_setimesbyt5():

    def __init__(self,
                 datasets_directory="src/main/resources",
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
            target_vocab = list([unknown_vocabulary_type, padding_vocabulary_type])
            source_vocab = list([unknown_vocabulary_type, padding_vocabulary_type])
            target_encodings = list()
            source_encodings = list()
            for entry in target_sentences_length_limited:
                encoding = list()
                for character in entry:
                    if character not in target_vocab:
                        target_vocab.append(character)
                    encoding.append(target_vocab.index(character))
                target_encodings.append(torch.tensor(encoding))
            for entry in source_sentences_length_limited:
                encoding = list()
                for character in entry:
                    if character not in source_vocab:
                        source_vocab.append(character)
                    encoding.append(source_vocab.index(character))
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
                   "setimes_parsed-" + str(time.time()))

    def decode_target_tensor(self, dataset_holder: DatasetHolder, tensor_to_decode):
        vocab = dataset_holder.get_target_vocab()
        decoded_tensor = torch.select(vocab, 0, tensor_to_decode)
        return "".join(decoded_tensor.tolist()[0])

    def decode_source_tensor(self, dataset_holder: DatasetHolder, tensor_to_decode):
        vocab = dataset_holder.get_target_vocab()
        decoded_tensor = torch.select(vocab, 0, tensor_to_decode)
        return "".join(decoded_tensor.tolist()[0])
