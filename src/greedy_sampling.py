import json
import random
import time
from typing import Optional

import torch
import numpy as np

root_filepath = "src/main/"
is_remote_execution = False
torch.set_printoptions(threshold=100000, edgeitems=10000, linewidth=100000)


# 16M Turkish
SETimesByT5Vaswani2017Kocmi2018_0 = {
    'runner_hyperparameters_name': 'SETimesByT5Vaswani2017Kocmi2018_0',
    'model_parameter_filepath': root_filepath + "resources/saved_parameters/SETimesByT5Vaswani2017Kocmi2018_0-1715937924-model.params",
    'datasets_filepath': root_filepath + "resources/parsed_datasets/setimes/setimes_parsed-1715586293",
    'output_filepath': root_filepath + "resources/model_translations/SETimesByT5Vaswani2017Kocmi2018_0.json",
    'model_hyperparameters': {
        'src_vocab_size': 0,
        'tgt_vocab_size': 81,
        'max_src_seq_len': 0,
        'max_tgt_seq_len': 256,
        'd_model': 256,
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
    }
}

# 15M Turkish
SETimesByT5Vaswani2017Kocmi2018_1 = {
    'runner_hyperparameters_name': 'SETimesByT5Vaswani2017Kocmi2018_1',
    'model_parameter_filepath': root_filepath + "resources/saved_parameters/SETimesByT5Vaswani2017Kocmi2018_1-1715937961-model.params",
    'datasets_filepath': root_filepath + "resources/parsed_datasets/setimes/setimes_parsed-1715586974",
    'output_filepath': root_filepath + "resources/model_translations/SETimesByT5Vaswani2017Kocmi2018_1.json",
    'model_hyperparameters': {
        'src_vocab_size': 0,
        'tgt_vocab_size': 81,
        'max_src_seq_len': 0,
        'max_tgt_seq_len': 256,
        'd_model': 256,
        'nhead': 8,
        # number of encoders is 3 times that of decoders, following Xue 2021 - ByT5 - Sec 3.1
        'num_encoder_layers': 9,
        'num_decoder_layers': 3,
        'dim_feedforward': 1796,
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
    }
}

# 4M Turkish
SETimesByT5Vaswani2017Kocmi2018_2 = {
    'runner_hyperparameters_name': 'SETimesByT5Vaswani2017Kocmi2018_2',
    'model_parameter_filepath': root_filepath + "resources/saved_parameters/SETimesByT5Vaswani2017Kocmi2018_2-1715936459-model.params",
    'datasets_filepath': root_filepath + "resources/parsed_datasets/setimes/setimes_parsed-1715586361",
    'output_filepath': root_filepath + "resources/model_translations/SETimesByT5Vaswani2017Kocmi2018_2.json",
    'model_hyperparameters': {
        'src_vocab_size': 0,
        'tgt_vocab_size': 81,
        'max_src_seq_len': 0,
        'max_tgt_seq_len': 256,
        'd_model': 128,
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
    }
}

# 16M Kazakh
NewsCommentaryByT5Vaswani2017Kocmi2018_0 = {
    'runner_hyperparameters_name': 'NewsCommentaryByT5Vaswani2017Kocmi2018_0',
    'model_parameter_filepath': root_filepath + "resources/saved_parameters/NewsCommentaryByT5Vaswani2017Kocmi2018_0-1715965639-model.params",
    'datasets_filepath': root_filepath + "resources/parsed_datasets/SMTNewsCommentary/SMTNewsCommentary_parsed-1715949808",
    'output_filepath': root_filepath + "resources/model_translations/NewsCommentaryByT5Vaswani2017Kocmi2018_0.json",
    'model_hyperparameters': {
        'src_vocab_size': 150,
        'tgt_vocab_size': 81,
        'max_src_seq_len': 266,
        'max_tgt_seq_len': 256,
        'd_model': 256,
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
    }
}

# 15M Kazakh
NewsCommentaryByT5Vaswani2017Kocmi2018_1 = {
    'runner_hyperparameters_name': 'NewsCommentaryByT5Vaswani2017Kocmi2018_1',
    'model_parameter_filepath': root_filepath + "resources/saved_parameters/NewsCommentaryByT5Vaswani2017Kocmi2018_1-1715966141-model.params",
    'datasets_filepath': root_filepath + "resources/parsed_datasets/SMTNewsCommentary/SMTNewsCommentary_parsed-1715949808",
    'output_filepath': root_filepath + "resources/model_translations/NewsCommentaryByT5Vaswani2017Kocmi2018_1.json",
    'model_hyperparameters': {
        'src_vocab_size': 0,
        'tgt_vocab_size': 81,
        'max_src_seq_len': 0,
        'max_tgt_seq_len': 256,
        'd_model': 256,
        'nhead': 8,
        # number of encoders is 3 times that of decoders, following Xue 2021 - ByT5 - Sec 3.1
        'num_encoder_layers': 9,
        'num_decoder_layers': 3,
        'dim_feedforward': 1796,
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
    }
}

# 4M Kazakh
NewsCommentaryByT5Vaswani2017Kocmi2018_2 = {
    'runner_hyperparameters_name': 'NewsCommentaryByT5Vaswani2017Kocmi2018_2',
    'model_parameter_filepath': root_filepath + "resources/saved_parameters/NewsCommentaryByT5Vaswani2017Kocmi2018_2-1715965581-model.params",
    'datasets_filepath': root_filepath + "resources/parsed_datasets/SMTNewsCommentary/SMTNewsCommentary_parsed-1715949808",
    'output_filepath': root_filepath + "resources/model_translations/NewsCommentaryByT5Vaswani2017Kocmi2018_2.json",
    'model_hyperparameters': {
        'src_vocab_size': 0,
        'tgt_vocab_size': 81,
        'max_src_seq_len': 0,
        'max_tgt_seq_len': 256,
        'd_model': 128,
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
    }
}


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
        self.model_hyperparameters = model_hyperparameters
        # add one for the end of sequence token
        self.max_src_seq_len = model_hyperparameters['max_src_seq_len'] + 1
        self.max_tgt_seq_len = model_hyperparameters['max_tgt_seq_len'] + 1
        self.src_embeddings = torch.nn.Embedding(
            model_hyperparameters['src_vocab_size'],
            model_hyperparameters['d_model']
        )
        self.tgt_embeddings = torch.nn.Embedding(
            model_hyperparameters['tgt_vocab_size'],
            model_hyperparameters['d_model']
        )
        self.src_pos_enc = torch.nn.Embedding(
            self.max_src_seq_len,
            model_hyperparameters['d_model']
        ).requires_grad_(False)
        self.tgt_pos_enc = torch.nn.Embedding(
            self.max_tgt_seq_len,
            model_hyperparameters['d_model']
        ).requires_grad_(False)
        self.linear_output_projection_1 = torch.nn.Linear(
            self.max_tgt_seq_len,
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
        for i in range(0, self.max_tgt_seq_len + 1):
            tgt_mask = self.generate_square_subsequent_mask(i)
            if is_remote_execution:
                self.tgt_mask_cache[i] = tgt_mask.to(device="cuda")
            self.tgt_mask_cache[i] = tgt_mask
        self.indices_cache = {}
        for i in range(0, max([self.max_src_seq_len, self.max_tgt_seq_len]) + 1):
            indices = torch.tensor(np.arange(0, i), dtype=torch.long)
            if is_remote_execution:
                indices = indices.to(device="cuda")
            self.indices_cache[i] = indices

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
        if len(src.shape) == 1:
            src = src.unsqueeze(dim=0)
        if len(tgt.shape) == 1:
            tgt = tgt.unsqueeze(dim=0)
        src_embedding_pos_enc = self.src_embeddings(src) + self.src_pos_enc(self.indices_cache[src.shape[1]])
        tgt_embedding_pos_enc = self.tgt_embeddings(tgt) + self.tgt_pos_enc(self.indices_cache[tgt.shape[1]])
        tgt_mask = self.tgt_mask_cache[tgt.shape[1]]
        transformer_output = super().forward(src_embedding_pos_enc, tgt_embedding_pos_enc, src_mask,
                                             tgt_mask, memory_mask, src_key_padding_mask,
                                             tgt_key_padding_mask, memory_key_padding_mask,
                                             src_is_causal, tgt_is_causal, memory_is_causal)
        transformer_output = torch.swapaxes(transformer_output, -1, -2)
        transformer_output = torch.nn.functional.pad(
            transformer_output,
            (0, self.max_tgt_seq_len - transformer_output.shape[2]),
            value=0
        )
        output = self.logsoftmax_output(
            self.linear_output_projection_2(
                torch.squeeze(
                    self.linear_output_projection_1(
                        transformer_output
                    ),
                    dim=2
                )
            )
        )
        del src_embedding_pos_enc
        del tgt_embedding_pos_enc
        del transformer_output
        return output

    def freeze_target_embeddings(self):
        del self.tgt_embeddings
        self.tgt_embeddings = torch.nn.Embedding(
            self.model_hyperparameters['tgt_vocab_size'],
            self.model_hyperparameters['d_model']
        ).requires_grad_(False)
        del self.linear_output_projection_1
        self.linear_output_projection_1 = torch.nn.Linear(
            self.max_tgt_seq_len,
            1,
            bias=False
        ).requires_grad_(False)
        del self.linear_output_projection_2
        self.linear_output_projection_2 = torch.nn.Linear(
            self.model_hyperparameters['d_model'],
            self.model_hyperparameters['tgt_vocab_size'],
            bias=False
        ).requires_grad_(False)

    def set_source_embeddings_for_transfer_learning(self, source_embeddings_dim):
        del self.src_embeddings
        self.src_embeddings = torch.nn.Embedding(
            source_embeddings_dim,
            self.model_hyperparameters['d_model']
        )


class Runner:

    def __init__(self):
        self.runner_hyperparameters = NewsCommentaryByT5Vaswani2017Kocmi2018_0
        self.model_hyperparameters = self.runner_hyperparameters['model_hyperparameters']
        self.runner_hyperparameters_name = self.runner_hyperparameters['runner_hyperparameters_name']
        self.model_parameter_filepath = self.runner_hyperparameters['model_parameter_filepath']
        self.datasets_filepath = self.runner_hyperparameters['datasets_filepath']
        self.output_filepath = self.runner_hyperparameters['output_filepath']
        self.max_target_sequence_length = self.runner_hyperparameters['model_hyperparameters']['max_tgt_seq_len']
        self.dataset_holder: DatasetHolder = None
        self.model = None
        print(f"Initialized runner {self.runner_hyperparameters_name} with parameters {self.runner_hyperparameters}")

    def load_dataset(self):
        self.dataset_holder = torch.load(self.datasets_filepath)

    def load_model(self):
        self.model = transformer_vaswani2017(model_hyperparameters=self.model_hyperparameters)
        self.model.freeze_target_embeddings()
        if is_remote_execution:
            model_parameters = torch.load(self.model_parameter_filepath, map_location=torch.device('cuda'))
        else:
            model_parameters = torch.load(self.model_parameter_filepath, map_location=torch.device('cpu'))
        self.model.load_state_dict(model_parameters)
        self.model.eval()

    def evaluate_model(self):
        outputs = list()
        source_vocab_numpy = self.dataset_holder.get_source_vocab_numpy()
        target_vocab_numpy = self.dataset_holder.get_target_vocab_numpy()
        assert len(self.dataset_holder.get_source_encodings_test()) == len(self.dataset_holder.get_target_encodings_test())
        for i in range(0, len(self.dataset_holder.get_source_encodings_test())):
            if is_remote_execution:
                source_encoding = self.dataset_holder.get_source_encodings_test()[i].to(device="cuda")
                target_encoding = self.dataset_holder.get_target_encodings_test()[i].to(device="cuda")
            else:
                source_encoding = self.dataset_holder.get_source_encodings_test()[i].to(device="cpu")
                target_encoding = self.dataset_holder.get_target_encodings_test()[i].to(device="cpu")
            j = 0
            end_of_sequence = False
            while j < self.model_hyperparameters['max_tgt_seq_len'] and end_of_sequence == False:
                target_encoding_slice = torch.tensor_split(target_encoding, [j], dim=0)
                output_logits = self.model.forward(
                    source_encoding,
                    target_encoding_slice[0]
                ).detach().to(device="cpu").flatten().numpy()
                output_logits_sort_pairs = list()
                for k in range(0, len(output_logits)):
                    output_logits_sort_pairs.append([output_logits[k], k])
                output_logits_sorted = sorted(output_logits_sort_pairs, key=lambda logit_pair: logit_pair[0], reverse=True)
                output_sort_index = 0
                # don't pick mark up characters or whitespace if first term
                if j == 0:
                    while output_logits_sorted[output_sort_index][1] in (0, 1, 2, 7):
                        output_sort_index = output_sort_index + 1
                else:
                    # don't repeat unknown, padding, or spaces
                    if (prediction_encoding[j-1] == 0
                            or prediction_encoding[j-1] == 1
                            or prediction_encoding[j-1] == 7):
                        while (output_logits_sorted[output_sort_index][1] == 0
                               or output_logits_sorted[output_sort_index][1] == 1
                               or output_logits_sorted[output_sort_index][1] == 7):
                            output_sort_index = output_sort_index + 1
                output_vocab_index = output_logits_sorted[output_sort_index][1]
                # index = output_vocab_index # always outputs fixed sequence
                index = min(random.randint(output_vocab_index, output_vocab_index + 3), len(output_logits_sorted))
                if j == 0:
                    prediction_encoding = torch.tensor([output_logits_sorted[index][1]], dtype=torch.float)
                else:
                    prediction_encoding = torch.cat([prediction_encoding, torch.tensor([output_logits_sorted[index][1]], dtype=torch.float)])
                if prediction_encoding[j] == self.dataset_holder.get_target_vocab().index(
                        self.dataset_holder.get_end_of_sequence_vocabulary_type()):
                    end_of_sequence = True
                j = j + 1
            decoded_source = np.take(source_vocab_numpy, source_encoding.detach().to(device="cpu").flatten().numpy())
            decoded_target = np.take(target_vocab_numpy, target_encoding.detach().to(device="cpu").flatten().numpy())
            decoded_translation = np.take(target_vocab_numpy, prediction_encoding.detach().to(device="cpu", dtype=torch.int32).numpy())
            outputs.append({
                "source": "".join(decoded_source.tolist()),
                "target": "".join(decoded_target.tolist()),
                "translation": "".join(decoded_translation.tolist())
            })
            del source_encoding
            del target_encoding
            del prediction_encoding
            del target_encoding_slice
            del decoded_source
            del decoded_target
            del decoded_translation
            print(f"Completed translation {i} of {len(self.dataset_holder.get_source_encodings_test())} at {time.time()}")
            if is_remote_execution:
                torch.cuda.empty_cache()
                if i % 100 == 0:
                    print(f"Translation{i}: {outputs[i]}")
                    print(f"Memory usage summary:")
                    print(f"{torch.cuda.memory_summary()}")
                    torch.cuda.reset_max_memory_allocated()
                    torch.cuda.reset_max_memory_cached()
                    torch.cuda.reset_peak_memory_stats()
        output_file = open(self.output_filepath, "w+")
        output_file.write(json.dumps(outputs))
        output_file.close()


runner = Runner()
runner.load_dataset()
runner.load_model()
runner.evaluate_model()
