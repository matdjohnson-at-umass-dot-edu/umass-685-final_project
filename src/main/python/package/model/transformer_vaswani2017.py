from typing import Optional, Union, Callable
import torch


class transformer_vaswani2017(torch.nn.Transformer):

    def __init__(self, model_hyperparameters):
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
        self.model_hyperparameters = model_hyperparameters

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
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1])
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
        output = torch.softmax(
            self.linear_output_projection_2(
                torch.squeeze(
                    self.linear_output_projection_1(
                        transformer_output
                    )
                )
            ),
            dim=1
        )
        del transformer_output
        return output
