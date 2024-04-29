from typing import Optional, Any, Union, Callable
import torch


class transformer_vaswani2017(torch.nn.Transformer):

    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.functional.relu,
                 custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = True,
                 norm_first: bool = False,
                 bias: bool = True,
                 device=None,
                 dtype=None):
        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                         dropout, activation, custom_encoder, custom_decoder, layer_norm_eps,
                         batch_first, norm_first, bias, device, dtype)
        self.src_embeddings = torch.nn.Embedding(src_vocab_size, d_model)
        self.tgt_embeddings = torch.nn.Embedding(tgt_vocab_size, d_model)

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
        tgt_mask = self.generate_square_subsequent_mask(len(tgt_embedding))
        output = super().forward(src_embedding, tgt_embedding, src_mask, tgt_mask, memory_mask, src_key_padding_mask,
                        tgt_key_padding_mask, memory_key_padding_mask, src_is_causal, tgt_is_causal, memory_is_causal)
        return output
