import numpy as np


class DatasetHolder:

    def __init__(self):
        self.target_vocab = None
        self.source_vocab = None
        self.target_encodings = None
        self.source_encodings = None
        self.max_seq_obs = 0

    def get_unknown_type(self):
        return self.unknown_vocabulary_type

    def get_padding_type(self):
        return self.padding_vocabulary_type

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

    def get_target_encodings_train(self):
        return self.target_encodings

    def set_target_encodings_train(self, target_encodings):
        self.target_encodings = target_encodings

    def get_source_encodings_train(self):
        return self.source_encodings

    def set_source_encodings_train(self, source_encodings):
        self.source_encodings = source_encodings

    def get_target_encodings_validate(self):
        return self.target_encodings

    def set_target_encodings_validate(self, target_encodings):
        self.target_encodings = target_encodings

    def get_source_encodings_validate(self):
        return self.source_encodings

    def set_source_encodings_validate(self, source_encodings):
        self.source_encodings = source_encodings

    def get_target_encodings_test(self):
        return self.target_encodings

    def set_target_encodings_test(self, target_encodings):
        self.target_encodings = target_encodings

    def get_source_encodings_test(self):
        return self.source_encodings

    def set_source_encodings_test(self, source_encodings):
        self.source_encodings = source_encodings

    def get_max_seq_obs(self):
        return self.max_seq_obs

    def set_max_seq_obs(self, max_seq_obs):
        self.max_seq_obs = max_seq_obs

