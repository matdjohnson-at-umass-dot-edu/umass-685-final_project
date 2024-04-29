

class BaseDatasetTransformer:

    def __init__(self):
        self.source_vocab = list()
        self.source_sentences = list()
        self.source_encodings = list()
        self.target_vocab = list()
        self.target_sentences = list()
        self.target_encodings = list()

    def initialize_with_values(self, hyperparameters):
        pass

    # todo - add return object with embeddings
    def read_dataset(self):
        pass

    def encode_dataset(self, dataset):
        pass

    def get_source_vocab(self):
        return self.source_vocab

    def get_source_sentences(self):
        return self.source_sentences

    def get_source_encodings(self):
        return self.source_encodings

    def get_target_vocab(self):
        return self.target_vocab

    def get_target_sentences(self):
        return self.target_sentences

    def get_target_encodings(self):
        return self.target_encodings

    def get_name(self):
        pass
