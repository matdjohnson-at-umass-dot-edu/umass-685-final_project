import json
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np

root_filepath = "src/main/"
is_remote_execution = False
torch.set_printoptions(threshold=100000, edgeitems=10000, linewidth=100000)


class DatasetHolder:

    def __init__(self):
        self.unknown_vocabulary_type = None
        self.padding_vocabulary_type = None
        self.end_of_sequence_type = None
        self.target_vocab = None
        self.target_vocab_array = None
        self.target_vocab_counts = None
        self.source_vocab = None
        self.source_vocab_array = None
        self.source_vocab_counts = None
        self.target_encodings = None
        self.target_encodings_train = None
        self.target_encodings_test = None
        self.source_encodings = None
        self.source_encodings_train = None
        self.source_encodings_test = None
        self.max_src_seq_obs = 0
        self.max_tgt_seq_obs = 0

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

    def set_target_vocab(self, target_vocab):
        self.target_vocab = target_vocab

    def get_target_vocab_numpy(self):
        if self.target_vocab_array is None:
            self.target_vocab_array = np.array(self.target_vocab)
        return self.target_vocab_array

    def get_target_vocab_counts(self):
        return self.target_vocab_counts

    def set_target_vocab_counts(self, target_vocab_counts):
        self.target_vocab_counts = target_vocab_counts

    def get_source_vocab(self):
        return self.source_vocab

    def set_source_vocab(self, source_vocab):
        self.source_vocab = source_vocab

    def get_source_vocab_numpy(self):
        if self.source_vocab_array is None:
            self.source_vocab_array = np.array(self.source_vocab)
        return self.source_vocab_array

    def get_source_vocab_counts(self):
        return self.source_vocab_counts

    def set_source_vocab_counts(self, source_vocab_counts):
        self.source_vocab_counts = source_vocab_counts

    def get_target_encodings(self):
        return self.target_encodings

    def set_target_encodings(self, target_encodings):
        del self.target_encodings
        self.target_encodings = target_encodings
        if is_remote_execution:
            torch.cuda.empty_cache()

    def get_source_encodings(self):
        return self.source_encodings

    def set_source_encodings(self, source_encodings):
        del self.source_encodings
        self.source_encodings = source_encodings
        if is_remote_execution:
            torch.cuda.empty_cache()

    def get_target_encodings_train(self):
        return self.target_encodings_train

    def set_target_encodings_train(self, target_encodings_train):
        del self.target_encodings_train
        self.target_encodings_train = target_encodings_train
        if is_remote_execution:
            torch.cuda.empty_cache()

    def get_source_encodings_train(self):
        return self.source_encodings_train

    def set_source_encodings_train(self, source_encodings_train):
        del self.source_encodings_train
        self.source_encodings_train = source_encodings_train
        if is_remote_execution:
            torch.cuda.empty_cache()

    def get_target_encodings_test(self):
        return self.target_encodings_test

    def set_target_encodings_test(self, target_encodings_test):
        del self.target_encodings_test
        self.target_encodings_test = target_encodings_test
        if is_remote_execution:
            torch.cuda.empty_cache()

    def get_source_encodings_test(self):
        return self.source_encodings_test

    def set_source_encodings_test(self, source_encodings_test):
        del self.source_encodings_test
        self.source_encodings_test = source_encodings_test
        if is_remote_execution:
            torch.cuda.empty_cache()

    def get_max_src_seq_obs(self):
        return self.max_src_seq_obs

    def set_max_src_seq_obs(self, max_src_seq_obs):
        self.max_src_seq_obs = max_src_seq_obs

    def get_max_tgt_seq_obs(self):
        return self.max_tgt_seq_obs

    def set_max_tgt_seq_obs(self, max_tgt_seq_obs):
        self.max_tgt_seq_obs = max_tgt_seq_obs



dataset_holder: DatasetHolder = torch.load(
    root_filepath + "resources/parsed_datasets/setimes/setimes_parsed-1715586293")

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="kaz_Cyrl")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("cuda").eval()

source_sentences = list()
target_sentences = list()
outputs = list()
for encoding in dataset_holder.get_source_encodings_test():
    decoded_tensor = np.take(dataset_holder.get_source_vocab_numpy(), encoding.detach().to(device="cpu").flatten().numpy())
    source_sentences.append("".join(decoded_tensor)[:-1])
for encoding in dataset_holder.get_target_encodings_test():
    decoded_tensor = np.take(dataset_holder.get_target_vocab_numpy(), encoding.detach().to(device="cpu").flatten().numpy())
    target_sentences.append("".join(decoded_tensor)[:-1])
assert len(source_sentences) == len(target_sentences)
for i in range(0, len(source_sentences)):
    tokenization = tokenizer(source_sentences[i], return_tensors="pt").to("cuda")
    generated_output = model.generate(**tokenization, forced_bos_token_id=tokenizer.lang_code_to_id['eng_Latn'])
    del tokenization
    torch.cuda.empty_cache()
    decoded_output = tokenizer.batch_decode(generated_output)[0].replace('</s>eng_Latn', '').replace('</s>', '')
    outputs.append({'source': source_sentences[i], 'target': target_sentences[i], 'baseline': decoded_output})
    print(f"completed processing {i+1} of {len(source_sentences)} at {time.time()}")
    if is_remote_execution and i % 100 == 0:
        print(f"Memory usage summary:")
        print(f"{torch.cuda.memory_summary()}")
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_peak_memory_stats()

output_file = open(
    root_filepath + "resources/baseline_translations/setimes/setimes_parsed-1715586293-NLLB.json", "w+")

output_file.write(json.dumps(outputs))

output_file.close()

