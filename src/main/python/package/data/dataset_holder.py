import numpy as np


class DatasetHolder:

    def __init__(self):
        self.unknown_vocabulary_type = '<unk>'
        self.padding_vocabulary_type = '<pad>'
        self.target_sentences = None
        self.source_sentences = None
        self.target_vocab = list([self.unknown_vocabulary_type, self.padding_vocabulary_type])
        self.source_vocab = list([self.unknown_vocabulary_type, self.padding_vocabulary_type])
        self.target_encodings = None
        self.source_encodings = None
        self.max_seq_obs = 0

    def get_unknown_type(self):
        return self.unknown_vocabulary_type

    def get_padding_type(self):
        return self.padding_vocabulary_type

    def get_target_sentences(self):
        return self.target_sentences

    def set_target_sentences(self, target_sentences):
        self.target_sentences = target_sentences

    def get_source_sentences(self):
        return self.source_sentences

    def set_source_sentences(self, source_sentences):
        self.source_sentences = source_sentences

    def get_target_vocab(self):
        return self.target_vocab

    def set_target_vocab(self, target_vocab):
        self.target_vocab = target_vocab

    def get_source_vocab(self):
        return self.source_vocab

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

    def get_max_seq_obs(self):
        return self.max_seq_obs

    def set_max_seq_obs(self, max_seq_obs):
        self.max_seq_obs = max_seq_obs

    def get_projected_max_seq_len(self):
        return int(np.max([
            np.floor(self.get_max_seq_obs() * 1.1),
            self.get_max_seq_obs() + 100
        ]))

