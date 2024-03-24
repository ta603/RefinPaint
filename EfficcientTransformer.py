import math
import pdb

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def top_p_sampling(logits, thres=0.9):
    """
    Performs top-p sampling from logits.
    :param logits: The logits from the model's output of shape [1, 1, vocab_size]
    :param p: The threshold for the cumulative probability
    :return: The sampled word index of shape [batch_size, seq_len]
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # Corrected scatter operation
    sampled_logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

    return sampled_logits


class EfficientDecoder(nn.Module):
    def __init__(self, num_layers,d_model, max_length, h, dropout, vocab_size, dec_token_emb, position_emb, is_half=True):
        super(EfficientDecoder, self).__init__()

        self.max_length = max_length
        self.pad_idx = 0
        self.decoder = Decoder(num_layers, d_model, vocab_size, h, dropout)

        self.dec_token_emb = dec_token_emb
        self.position_emb = position_emb
        self.is_half = is_half

    def generate(self, decoder_input, enc, labels, conditioning_mask, temperature=1.0):
        B = decoder_input.size(0)
        L = decoder_input.size(1)
        DEV = decoder_input.device

        decoder_input = decoder_input[:, :self.max_length].unsqueeze(0)

        int_tensor = conditioning_mask.int()
        # Find the index of the first False
        first_false_idx = (int_tensor == 0).nonzero(as_tuple=True)[1][0].item()
        # Find the index of the last False
        last_false_idx = (int_tensor == 0).nonzero(as_tuple=True)[1][-1].item()

        tgt_mask = self.causal_mask(L, DEV)
        src_mask = self.identity_mask(L, DEV)

        predicted_sequence = torch.empty(0, dtype=torch.long, device=DEV)  # Create an empty tensor on the device

        # primer part
        if first_false_idx != 0:
            x = (self.dec_token_emb(decoder_input[:, :, :first_false_idx-1]) +
                 self.position_emb(torch.arange(first_false_idx-1, device=DEV))).squeeze(0)
            # pdb.set_trace()
            log_prob, prev_states = self.decoder.incremental_primer_forward(x, enc, src_mask[:first_false_idx-1],
                                                                            tgt_mask[:first_false_idx-1, :first_false_idx-1])
            predicted_sequence = torch.cat((predicted_sequence, labels[0, :first_false_idx-1]), dim=0)

            # first token after primer
            next_token = (self.dec_token_emb(decoder_input[:, :, first_false_idx-1]) +
                          self.position_emb(torch.arange(first_false_idx-1, first_false_idx, device=DEV)))
        else:
            next_token = (self.dec_token_emb(decoder_input[:, :, 0]) +
                          self.position_emb(torch.arange(0, 1, device=DEV)))

        # Cache the position embeddings
        pos_emb_cache = self.position_emb(torch.arange(0, L, device=DEV))

        # user selected fragment
        for k in tqdm(range(first_false_idx-1, last_false_idx - 1)):
            log_prob, prev_states = self.decoder.incremental_forward(
                next_token, enc,
                src_mask[k:k + 1], tgt_mask[k:k + 1, :k + 1],
                temperature,
                prev_states
            )

            # predict next_id
            if conditioning_mask[0][k] == 1:
                next_id = decoder_input[:, :, k + 1]
            else:
                log_prob = top_p_sampling(log_prob.squeeze(0), 0.05)
                log_prob = F.softmax(log_prob / temperature, dim=-1)
                next_id = torch.multinomial(log_prob, 1)
                # next_id = log_prob.argmax(-1)
            # next token embedding using the cached embeddings
            next_token = self.dec_token_emb(next_id) + pos_emb_cache[k + 1]
            predicted_sequence = torch.cat((predicted_sequence, next_id.squeeze(0)), dim=0)

        # last part not selected by the user
        predicted_sequence = torch.cat((predicted_sequence, labels[0, last_false_idx - 1:]), dim=0)

        return predicted_sequence

    def generate_tokencritic(self, decoder_input, enc, labels, conditioning_mask, clf, temperature=1.0):
        B = decoder_input.size(0)
        L = decoder_input.size(1)
        DEV = decoder_input.device
        decoder_input = decoder_input[:, :self.max_length].unsqueeze(0)
        int_tensor = conditioning_mask.int()
        # Find the index of the first False
        first_false_idx = (int_tensor == 0).nonzero(as_tuple=True)[1][0].item()
        # Find the index of the last False
        last_false_idx = (int_tensor == 0).nonzero(as_tuple=True)[1][-1].item()

        tgt_mask = self.causal_mask(L, DEV)
        src_mask = self.identity_mask(L, DEV)

        predicted_sequence = torch.empty(0, dtype=torch.long, device=DEV)  # Create an empty tensor on the device
        LOGITS = []

        # primer part
        if first_false_idx != 0:
            x = (self.dec_token_emb(decoder_input[:, :, :first_false_idx-1]) +
                 self.position_emb(torch.arange(first_false_idx-1, device=DEV))).squeeze(0)
            # pdb.set_trace()
            log_prob, prev_states = self.decoder.incremental_primer_forward(x, enc, src_mask[:first_false_idx-1],
                                                                            tgt_mask[:first_false_idx-1, :first_false_idx-1])
            predicted_sequence = torch.cat((predicted_sequence, labels[0, :first_false_idx-1]), dim=0)

            # first token after primer
            next_token = (self.dec_token_emb(decoder_input[:, :, first_false_idx-1]) +
                          self.position_emb(torch.arange(first_false_idx-1, first_false_idx, device=DEV)))
            LOGITS.extend(decoder_input[:, :, :first_false_idx-1].tolist()[0][0])
        else:
            next_token = (self.dec_token_emb(decoder_input[:, :, 0]) +
                          self.position_emb(torch.arange(0, 1, device=DEV)))

        # Cache the position embeddings
        pos_emb_cache = self.position_emb(torch.arange(0, L, device=DEV))

        # user selected fragment
        for k in tqdm(range(first_false_idx-1, last_false_idx - 1)):
            log_prob, prev_states = self.decoder.incremental_forward(
                next_token, enc,
                src_mask[k:k + 1], tgt_mask[k:k + 1, :k + 1],
                temperature,
                prev_states
            )

            # predict next_id
            if conditioning_mask[0][k] == 1:
                next_id = decoder_input[:, :, k + 1]
                LOGITS.append(next_id.item())
            else:
                log_prob = clf(log_prob.squeeze(0))
                next_id = (F.sigmoid(log_prob) > 0.5).int()
                LOGITS.append(F.sigmoid(log_prob).item())
            # next token embedding using the cached embeddings
            next_token = self.dec_token_emb(next_id) + pos_emb_cache[k + 1]
            predicted_sequence = torch.cat((predicted_sequence, next_id.squeeze(0)), dim=0)

        # last part not selected by the user
        predicted_sequence = torch.cat((predicted_sequence, labels[0, last_false_idx - 1:]), dim=0)
        LOGITS.extend(decoder_input[:, :, last_false_idx - 1:].tolist()[0][0])
        return torch.tensor(LOGITS).to(decoder_input.device)

    def find_global_false_indices(self, conditioning_mask: torch.Tensor) -> (int, int):
        """
        Finds the global minimum for the first False index and the global maximum for the last False index
        across all batches in the given conditioning_mask tensor.

        Parameters:
            conditioning_mask (torch.Tensor): A 2D tensor of shape (B, L) with values of True or False.

        Returns:
            (int, int): Tuple containing the global minimum first False index and the global maximum last False index.
        """
        int_tensor = conditioning_mask.int()  # Convert tensor to int type (False -> 0, True -> 1)

        global_min_first = float('inf')  # Initialized to positive infinity
        global_max_last = float('-inf')  # Initialized to negative infinity

        for i in range(int_tensor.size(0)):  # Iterate over all batches
            false_indices = (int_tensor[i] == 0).nonzero(as_tuple=True)[0]

            if len(false_indices) == 0:
                continue  # If no False values for this batch, move to the next batch

            first_false_idx = false_indices[0].item()
            last_false_idx = false_indices[-1].item()

            # Update the global minimum and maximum
            global_min_first = min(global_min_first, first_false_idx)
            global_max_last = max(global_max_last, last_false_idx)

        return global_min_first, global_max_last

    def generate_batch(self, decoder_input, enc, labels, conditioning_mask, temperature=1.0):
        B = decoder_input.size(0)
        L = decoder_input.size(1)
        DEV = decoder_input.device

        decoder_input = decoder_input[:, :self.max_length]

        # Find the index of the first False
        first_false_idx, last_false_idx = self.find_global_false_indices(conditioning_mask)
        print("first", first_false_idx)
        print("last", last_false_idx)

        tgt_mask = self.causal_mask(L, DEV)
        src_mask = self.identity_mask(L, DEV)

        predicted_sequence = torch.empty(0, dtype=torch.long, device=DEV)  # Create an empty tensor on the device
        if first_false_idx != 0:
            # primer part
            x = (self.dec_token_emb(decoder_input[:, :first_false_idx]) +
                 self.position_emb(torch.arange(first_false_idx, device=DEV)))
            log_prob, prev_states = self.decoder.incremental_primer_forward(x, enc, src_mask[:first_false_idx],
                                                                            tgt_mask[:first_false_idx, :first_false_idx])
            predicted_sequence = torch.cat((predicted_sequence, labels[:, :first_false_idx]), dim=0)

            # first token after primer
            next_token = (self.dec_token_emb(decoder_input[:, first_false_idx]) +
                          self.position_emb(torch.arange(first_false_idx-1, first_false_idx, device=DEV))).unsqueeze(1)
        else:
            next_token = (self.dec_token_emb(decoder_input[:, 0]) +
                          self.position_emb(torch.arange(0, 1, device=DEV))).unsqueeze(1)
            prev_states = None

        # Cache the position embeddings
        pos_emb_cache = self.position_emb(torch.arange(0, L, device=DEV))

        # user selected fragment
        for k in tqdm(range(first_false_idx, last_false_idx - 1)):
            log_prob, prev_states = self.decoder.incremental_forward(
                next_token, enc,
                src_mask[k:k + 1], tgt_mask[k:k + 1, :k + 1],
                temperature,
                prev_states
            )
            # Determine which elements to get from decoder_input using the conditioning mask
            next_id = labels[:, k]

            # For elements where conditioning_mask is not 1, compute next_id using top_p_sampling
            generation_mask = conditioning_mask[:, k] == 0
            log_probs_masked = log_prob[generation_mask]
            log_probs_masked = top_p_sampling(log_probs_masked.squeeze(1), 0.05)
            log_probs_masked = F.softmax(log_probs_masked / temperature, dim=-1)
            next_id_generated = torch.multinomial(log_probs_masked, 1).squeeze()

            # Merge results
            next_id[generation_mask] = next_id_generated
            # next token embedding using the cached embeddings
            next_token = (self.dec_token_emb(next_id) + pos_emb_cache[k + 1]).unsqueeze(1)
            predicted_sequence = torch.cat((predicted_sequence, next_id.unsqueeze(1)), dim=1)

        # last part not selected by the user
        # pdb.set_trace()
        predicted_sequence = torch.cat((predicted_sequence, labels[:, last_false_idx - 1:]), dim=1)

        return predicted_sequence

    def generate_slow(self, decoder_input, enc, conditioning_mask, temperature=1.0):
        B = decoder_input.size(0)
        L = decoder_input.size(1)
        DEV = decoder_input.device

        decoder_input = decoder_input[:, :self.max_length].unsqueeze(0)

        int_tensor = conditioning_mask.int()
        # Find the index of the first False
        first_false_idx = (int_tensor == 0).nonzero(as_tuple=True)[1][0].item()
        # Find the index of the last False
        last_false_idx = (int_tensor == 0).nonzero(as_tuple=True)[1][-1].item()

        tgt_mask = self.causal_mask(L, DEV)
        src_mask = self.identity_mask(L, DEV)

        next_token = self.dec_token_emb(decoder_input[:, :, 0]) + self.position_emb(torch.arange(1, device=DEV))
        predicted_sequence = []
        prev_states = None

        # log_prob, prev_states = self.incremental_primer_forward()
        for k in tqdm(range(0, last_false_idx-1 )):
            log_prob, prev_states = self.decoder.incremental_forward(
                next_token, enc,
                src_mask[k:k+1], tgt_mask[k:k+1, :k+1],
                temperature,
                prev_states
            )

            # predict next_id
            if conditioning_mask[0][k] == 1:
                next_id = decoder_input[:, :, k + 1]
            else:

                log_prob = top_p_sampling(log_prob.squeeze(0), 0.05)
                log_prob = F.softmax(log_prob / temperature, dim=-1)
                next_id = torch.multinomial(log_prob, 1)
                # next_id = log_prob.argmax(-1)

            # next token embedding
            next_token = (self.dec_token_emb(next_id) + self.position_emb(torch.tensor(k+1, device=DEV)))
            predicted_sequence.append(next_id.item())
        predicted_sequence.extend(labels[last_false_idx-1:].tolist())
        return predicted_sequence

    def causal_mask(self, sz, device):
        c_mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        # pdb.set_trace()
        mask = c_mask.float().masked_fill(c_mask == 0, float(0.0)).masked_fill(c_mask == 1, float('-inf')).to(
            device)
        return mask.half() if self.is_half else mask

    def identity_mask(self, sz, device):
        mask = (-float('inf') * torch.ones(sz, sz)).to(device)
        mask.fill_diagonal_(0)
        return mask.half() if self.is_half else mask

    def forward(self, decoder_input, enc, ctx_embedded=None, temperature=1.0):
        B = decoder_input.size(0)
        L = decoder_input.size(1)
        DEV = decoder_input.device

        tgt_mask = self.causal_mask(L, DEV)
        src_mask = self.identity_mask(L, DEV)

        dec_embedded = self.dec_token_emb(decoder_input) + self.position_emb(torch.arange(L, device=DEV))
        if ctx_embedded is not None:
            dec_embedded += ctx_embedded

        # x, enc, src_mask, tgt_mask, temperature
        log_probs = self.decoder(
            x=dec_embedded, memory=enc if enc is not None else dec_embedded,
            src_mask=src_mask, tgt_mask=tgt_mask,
            temperature=temperature
        )
        return log_probs


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, vocab_size, h, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
        self.generator = Generator(d_model, vocab_size)

    def forward(self, x, memory, src_mask, tgt_mask, temperature):
        y = x

        assert y.size(1) == tgt_mask.size(-1)

        for layer in self.layers:
            y = layer(y, memory, src_mask, tgt_mask)

        return self.generator(self.norm(y), temperature)

    def incremental_forward(self, x, memory, src_mask, tgt_mask, temperature, prev_states=None):
        y = x

        new_states = []
        for i, layer in enumerate(self.layers):
            y, new_sub_states = layer.incremental_forward(
                y, memory, src_mask, tgt_mask,
                prev_states[i] if prev_states else None
            )
            new_states.append(new_sub_states)

        new_states.append(torch.cat((prev_states[-1], y), 1) if prev_states else y)
        y = self.norm(new_states[-1])[:, -1:]

        return self.generator(y, temperature), new_states


    def incremental_primer_forward(self, x, memory, src_mask, tgt_mask):
        y = x
        new_states = []
        for i, layer in enumerate(self.layers):
            y, new_sub_states = layer.incremental_primer_forward(
                y, memory, src_mask, tgt_mask
            )
            new_states.append(new_sub_states)

        new_states.append(y)
        y = self.norm(new_states[-1])[:, -1:]
        return self.generator(y, 1), new_states


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, temperature):
        return self.proj(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.pw_ffn)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.src_attn = MultiHeadAttention(d_model, h, dropout)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.pw_ffn)

    def incremental_forward(self, x, memory, src_mask, tgt_mask, prev_states=None):
        new_states = []
        # x is 150
        m = memory
        x = torch.cat((prev_states[0], x), 1) if prev_states else x
        new_states.append(x)
        if 513 in x.shape:
            print()
        x = self.sublayer[0].incremental_forward(x, lambda x: self.self_attn(x[:, -1:], x, x, tgt_mask))
        x = torch.cat((prev_states[1], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[1].incremental_forward(x, lambda x: self.src_attn(x[:, -1:], m, m, src_mask))
        x = torch.cat((prev_states[2], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[2].incremental_forward(x, lambda x: self.pw_ffn(x[:, -1:]))
        return x, new_states

    def incremental_primer_forward(self, x, memory, src_mask, tgt_mask):
        new_states = []
        m = memory
        new_states.append(x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        new_states.append(x)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        new_states.append(x)
        x = self.sublayer[2](x, self.pw_ffn)
        return x, new_states


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.head_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x, l in zip((query, key, value), self.head_projs)]

        # attn_feature, _ = scaled_attention(query, key, value, mask)
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=True):
             attn_feature = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False)
        attn_concated = attn_feature.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.fc(attn_concated)


def scaled_attention(query, key, value, mask):
    d_k = query.size(-1)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
    scores += mask
    attn_weight = F.softmax(scores, -1)
    attn_feature = attn_weight.matmul(value)
    return attn_feature, attn_weight


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.mlp = nn.Sequential(
            Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.mlp(x)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x + self.dropout(y)

    def incremental_forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x[:, -1:] + self.dropout(y)


def Linear(in_features, out_features, bias=True, uniform=True):
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim, eps=1e-6):
    m = nn.LayerNorm(embedding_dim, eps)
    return m