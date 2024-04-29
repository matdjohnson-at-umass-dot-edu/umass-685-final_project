from .base_dataset_transformer import BaseDatasetTransformer
import torch


class DatasetTransformerSetimesByt5(BaseDatasetTransformer):

    def __init__(self,
                 filepath="src/main/resources/raw_datasets/setimes",
                 ids_filename="SETIMES.en-tr.ids",
                 en_filename="SETIMES.en-tr.en",
                 tr_filename="SETIMES.en-tr.tr"):
        self.filepath = filepath
        self.ids_filename = ids_filename
        self.en_filename = en_filename
        self.tr_filename = tr_filename
        self.target_sentences = list()
        self.source_sentences = list()
        self.target_vocab = list('<unk>')
        self.source_vocab = list('<unk>')
        self.target_encodings = list()
        self.source_encodings = list()

    def read_dataset(self):
        index_file = open(self.filepath + "/" + self.ids_filename)
        en_file = open(self.filepath + "/" + self.en_filename)
        tr_file = open(self.filepath + "/" + self.tr_filename)
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
            self.target_sentences.append(en_sentences[index[0] - 1])
            self.source_sentences.append(tr_sentences[index[1] - 1])

    # encode to Pytorch tensors
    def encode_dataset(self):
        for entry in self.target_sentences:
            encoding = list()
            for character in entry:
                if character not in self.target_vocab:
                    self.target_vocab.append(character)
                encoding.append(self.target_vocab.index(character))
            self.target_encodings.append(torch.tensor(encoding))
        for entry in self.source_sentences:
            encoding = list()
            for character in entry:
                if character not in self.source_vocab:
                    self.source_vocab.append(character)
                encoding.append(self.source_vocab.index(character))
            self.source_encodings.append(torch.tensor(encoding))

