import math
import pytorch_lightning as pl

from torch import nn

import torch.nn.functional as F

import torch

import EfficcientTransformer


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


class TransformerBase(pl.LightningModule):
    def __init__(self, config, len_dataset, vocab, lr, batch_size):
        super().__init__()
        self.vocab = vocab
        self.pitch_vocab = torch.Tensor([t_id for t_name, t_id in vocab.items() if "Pitch_" in t_name])
        self.onset_vocab = torch.Tensor([t_id for t_name, t_id in vocab.items() if "Position_" in t_name])
        self.duration_vocab = torch.Tensor([t_id for t_name, t_id in vocab.items() if "Duration_" in t_name])
        self.transformer = self._get_transformer(config, len_dataset)
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        self.save_hyperparameters(config_dict)
        self.loss = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
        self.batch_size = batch_size
        self.train_iters_per_epoch = len_dataset // self.batch_size
        self.max_epochs = config.max_epochs
        self.lr = lr

        self.exclude_ctx = config.exclude_ctx

    def generate(
            self,
            input_decoder,
            tgt,
            conditioning_mask
    ):
        t = input_decoder.shape[0]
        input_decoder, tgt = input_decoder.unsqueeze(dim=0), tgt.unsqueeze(dim=0)
        out = input_decoder.clone()
        conditioning_mask = conditioning_mask.unsqueeze(dim=0).bool()
        enc = self.transformer.inference_enc(tgt=tgt, conditioning_mask=conditioning_mask)

        out = self.transformer.decoder.generate(out, enc, labels=tgt, conditioning_mask=conditioning_mask)
        return out

    def generate_batch(
            self,
            input_decoder,
            tgt,
            conditioning_mask,
    ):
        enc = self.transformer.inference_enc(tgt=tgt, conditioning_mask=conditioning_mask)
        out = self.transformer.decoder.generate_batch(input_decoder, enc=enc, labels=tgt, conditioning_mask=conditioning_mask)
        return out


class InpaintingBase(nn.Module):
    def __init__(self, dim, enc_num_tokens, enc_depth, enc_heads, dec_num_tokens, dec_depth, dec_heads, max_seq_len,
                 tie_token_emb):
        super(InpaintingBase, self).__init__()
        # Token Embedding
        self.enc_token_emb = nn.Embedding(enc_num_tokens, dim)
        self.dec_token_emb = nn.Embedding(dec_num_tokens, dim)
        # Positional Embedding
        self.position_emb = nn.Embedding(max_seq_len, dim)
        # Optionally tie embeddings of encoder and decoder
        if tie_token_emb:
            self.dec_token_emb = self.enc_token_emb
        # Create encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=enc_heads,
            dim_feedforward=dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_depth)
        # Create decoder
        self.decoder = EfficcientTransformer.EfficientDecoder(
            num_layers=dec_depth, d_model=dim, h=dec_heads, vocab_size=dec_num_tokens, max_length=max_seq_len,
            dropout=0.1, dec_token_emb=self.dec_token_emb, position_emb=self.position_emb, is_half=False
        )
        self.max_seq_len = max_seq_len

    def anticausal_mask(self, sz, device):
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer.generate_square_subsequent_mask
        # The masked positions are filled with float(‘-inf’). Unmasked positions are filled with float(0.0).
        mask = torch.tril(torch.ones(sz, sz), diagonal=-1)
        # pdb.set_trace()
        return mask.float().masked_fill(mask == 0, float(0)).masked_fill(mask == 1, float('-inf')).to(device)

    def inference_enc(self, tgt, conditioning_mask=None):
        device = tgt.device
        max_seq_len = self.max_seq_len
        anti_causal_mask = self.anticausal_mask(max_seq_len, device)
        # encode tgt masked
        PAD_TOKEN_ID = 0
        tgt = tgt.masked_fill(~conditioning_mask, PAD_TOKEN_ID)
        enc_embedded = self.enc_token_emb(tgt) + self.position_emb(torch.arange(tgt.shape[-1], device=device))
        # pdb.set_trace()
        enc = self.encoder(enc_embedded, mask=anti_causal_mask)
        return enc

    def inference_dec(self, src, idx, enc):
        device = src.device
        identity_mask = self.identity_mask(idx + 1, device)
        causal_mask = self.causal_mask(idx + 1, device)
        dec_embedded = self.dec_token_emb(src[:, :idx + 1]) + self.position_emb(
            torch.arange(src[:, :idx + 1].shape[1], device=src.device))
        # Decode
        out = self.decoder(dec_embedded, enc[:, :idx + 1], tgt_mask=causal_mask, memory_mask=identity_mask)
        # Convert to logits
        logits = self.to_logits(out)
        return logits[:, - 1]

    def inference(self, src, tgt, enc, idx=None, conditioning_mask=None):
        if enc is None:
            ans = self.inference_enc(tgt, conditioning_mask)
        else:
            ans = self.inference_dec(src, idx, enc)
        return ans


class TransformerInpainting(TransformerBase):
    def __init__(self, config, len_dataset, vocab, lr, batch_size):
        super().__init__(config, len_dataset, vocab, lr, batch_size)
        print("Inpainting transformer")

    def _get_transformer(self, config, len_dataset):
        return InpaintingBase(
            dim=512,
            enc_num_tokens=config.num_tokens,
            enc_depth=4,
            enc_heads=8,
            dec_num_tokens=config.num_tokens,
            dec_depth=8,
            dec_heads=8,
            max_seq_len=config.max_seq_len,
            tie_token_emb=True
        )

    def inference(self, x, tgt, conditioning_mask, idx, enc=None):
        output = self.transformer.inference(x, tgt=tgt, idx=idx, conditioning_mask=conditioning_mask, enc=enc)
        return output







