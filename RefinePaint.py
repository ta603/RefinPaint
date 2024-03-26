import gzip
import json
import math
from math import cos, pi
from time import sleep
from zipfile import ZipFile

import numpy as np
import torch
from matplotlib import pyplot as plt

import sys

from numpy import convolve
from tqdm import tqdm

from MidiTok_modified.miditok.tokenizations.remi_wvelocity import REMIwVEL

from visual import plot_original, create_MIDI, plot_tokenization, save_midi_piano, wait_for_file_modification, \
    get_mask_saved_file

from InpaintingModel import TransformerInpainting
from FeedbackModel import TransformerClassifier


import os
import pickle
import random


import dataclasses


def save_model_in_parts(model, path, parts=5):
    # Get the state dictionary from the model
    state_dict = model.state_dict()

    # Split the state dictionary into parts
    items = list(state_dict.items())
    split_size = math.ceil(len(items) / parts)
    for i in range(parts):
        part_dict = dict(items[i * split_size:(i + 1) * split_size])

        # Save each part of the state dictionary separately
        part_path = f"{path}_part{i + 1}.pth"
        torch.save(part_dict, part_path)
        print(f"Saved {part_path}")


def load_model_state_from_parts(path, parts=5):
    # Reconstruct the state dictionary from parts
    state_dict = {}
    for i in range(parts):
        part_path = f"{path}_part{i + 1}.pth"
        part_dict = torch.load(part_path)
        state_dict.update(part_dict)

    return state_dict


@dataclasses.dataclass
class Config:
    max_seq_len: int
    pad_token_id: int
    batch_size: int
    max_epochs: int
    num_tokens: int
    exclude_ctx: bool
    pretrained_path: str = None
    beta: float = 0.1




def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_binary_data(data_path):
    with gzip.open(data_path + '.tar', 'rb') as f:
        return pickle.load(f)


def save_piece_data(song_ids, labels, mask, ctx, path):
    data = {
        "generated": song_ids,
        "labels": labels,
        "mask": mask,
        "ctx": ctx
    }
    #save json
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def expand_window(tensor):
    B, L = tensor.shape
    expanded = torch.zeros(B, L, dtype=torch.bool).to(tensor.device)

    for i in range(B):
        for j in range(1, L - 1):
            if tensor[i, j] == False:
                expanded[i, j - 1] = False
                expanded[i, j] = False
                expanded[i, j + 1] = False
            else:
                # Keep the value if it hasn't been set to False by its neighbors
                expanded[i, j] = expanded[i, j] or tensor[i, j]

        # Handle edge cases separately
        if tensor[i, 0] == False:
            expanded[i, 0] = False
            expanded[i, 1] = False
        else:
            expanded[i, 0] = expanded[i, 0] or tensor[i, 0]

        if tensor[i, L - 1] == False:
            expanded[i, L - 1] = False
            expanded[i, L - 2] = False
        else:
            expanded[i, L - 1] = expanded[i, L - 1] or tensor[i, L - 1]

    return expanded


tokenizer = REMIwVEL(
        pitch_range=range(21, 109),
        beat_res={(0, 3): 8, (4, 12): 4},
        nb_velocities=32,
        additional_tokens={
            'Chord': False,
            'Program': False,
            'Rest': False,
            'Tempo': False,
            'TimeSignature': False,
            'chord_tokens': False,
            'chord_tokens_with_root_note': False,
            'chord_unknown': False,
            'nb_tempos': 0,
            'rest_range': (2, 8),
            'tempo_range': (40, 250),
            'time_signature_range': (8, 2)
        },
        special_tokens=['PAD', 'BOS', 'EOS', 'MASK'],
        params=None
    )


good_ones = [
    4365, 4337, 1338, 4852, 4486, 1568, 1087, 3473, 2115, 2638, 2893, 5222, 1877, 7576, 2099, 2704, 203, 1494,
    5997, 5451, 7468, 17, 943, 5189, 3104, 6442, 4811, 3650, 1381, 6618, 761, 4987, 6741, 1375, 6587, 3659,
    6092, 7131, 2980, 1588, 4343, 5598, 2243, 6778, 4232, 3171, 6885, 6999, 2466, 85, 7132, 7281, 4778, 4146,
    949, 4969, 4375, 5882, 1254, 1216, 2684, 3785, 5701, 5923, 5874, 7441, 3778, 6669, 6138, 580, 7517, 2617,
    379, 2109, 2932, 909, 3056, 4673, 4052, 1088, 6975, 1076, 3866, 2574, 5741, 3623, 4402, 7328, 15, 6025,
    3003, 945, 6311, 3873, 4954, 2374, 6414, 3186, 6955, 2025
]


def plot_song_metrics(data, show, path_save=None):
    x = sorted(list(data.keys()), reverse=True)

    metrics = list(data[next(iter(data))].keys())

    plt.figure(figsize=(10, 20))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(len(metrics), 1, i)
        y = [sub_dict[metric] for key, sub_dict in sorted(data.items(), key=lambda item: item[0], reverse=False)]
        plt.plot(x, y, label=metric, marker='o')
        plt.ylabel('Value')
        plt.title(metric)
        plt.grid(True)
        plt.tight_layout()

    plt.xlabel('Time Point')
    plt.tight_layout()

    if path_save is not None:
        plt.savefig(path_save)

    if show:
        plt.show()



    plt.close()
    return path_save


# def get_experiment(device):
#     config = Config(max_seq_len=512, pad_token_id=0, batch_size=1, max_epochs=10, num_tokens=181, exclude_ctx=True)
#     state_dict = load_model_state_from_parts('checkpoints/inpainting', parts=5)
#     generator = TransformerInpainting(config=config, len_dataset=1, vocab={},
#                                                     lr=0, batch_size=1)
#     generator.load_state_dict(state_dict)
#
#     state_dict = load_model_state_from_parts('checkpoints/feedback', parts=5)
#     # state_dict = {k.replace("transformer.", ""): v for k, v in state_dict.items()}
#     feedback = TransformerClassifier(embedding_dim=512, layers=6, dropout=0.1, ctx_tokens=True, weighted=True,
#                                      max_seq_len=512, clf_size=181, num_tokens=181)
#     feedback.load_state_dict(state_dict)
#
#     generator.to(device)
#     generator.eval()
#
#     feedback.to(device)
#     feedback.eval()
#     return generator, feedback


def get_experiment(device):
    config = Config(max_seq_len=512, pad_token_id=0, batch_size=1, max_epochs=10, num_tokens=181, exclude_ctx=True)
    checkpoint_path = 'checkpoints/PIA_piano_efficient_v2/best-checkpoint.ckpt'
    generator = TransformerInpainting.load_from_checkpoint(checkpoint_path, config=config, len_dataset=1, vocab={},
                                                    lr=0)
    checkpoint_path = 'checkpoints/piano_5_million/noisyreal/best-checkpoint.ckpt'
    tokencritic = TransformerClassifier.load_from_checkpoint(
        checkpoint_path, max_epochs=0, lr=0, len_dataset=1, batch_size=1, embedding_dim=512, layers=6,
        dropout=0.1, ctx_tokens=True, weighted=True, max_seq_len=512, clf_size=181, num_tokens=181)
    save_model_in_parts(generator, 'checkpoints/inpainting', parts=5)
    save_model_in_parts(tokencritic, 'checkpoints/feedback', parts=5)



def save_json(data, path):
    with open(path, 'w') as fp:
        json.dump(data, fp)

def countRealLabels(y, kernel_size=3):
    y = y.detach().cpu().numpy()
    kernel = np.ones(kernel_size)
    pad_size = kernel_size // 2
    y_padded = np.pad(y, (pad_size, pad_size), mode='edge')
    conv_result = convolve(y_padded, kernel, mode='valid')
    labeled_result = conv_result == kernel_size
    return labeled_result.sum()


def sampleOneRealSegment(y_hat, previous_s, kernel_size, r):
    pad_size = kernel_size // 2
    s_padded = np.pad(previous_s.detach().cpu().numpy(), (pad_size, pad_size), mode='edge')
    y_hat[previous_s] = float('-inf')
    _, top_k_indices = torch.topk(y_hat, r)
    i = top_k_indices[-1]
    s_padded[i:i + kernel_size] = True
    s = s_padded[pad_size:-pad_size]  # trim the padded positions
    return s


def feedback_avg_decoding(generated, ctx, real_rate, feedback, avg_pooling, length_section):
    random_mask = None
    s = ctx.clone()
    print("masking rate ", real_rate, " length section ", length_section)

    y_hat = feedback(generated.unsqueeze(0), ctx.unsqueeze(0)).squeeze()

    r = 1
    while random_mask is None:
        s_temp = sampleOneRealSegment(y_hat, s, avg_pooling, r)
        s_temp = torch.tensor(s_temp, device=ctx.device)
        c = countRealLabels(s_temp[~ctx], kernel_size=avg_pooling) / length_section
        if c + avg_pooling/512 < real_rate:
            s = s_temp
        elif c == real_rate:
            s = s_temp
            random_mask = s
        else:
            if c != 1.0 and random.uniform(0, 1) < 0.5:
                s = s_temp
            random_mask = s
        r += 1
    print("La mascara tiene ", torch.sum(random_mask[~ctx]).item(), " reales de ", length_section)
    return random_mask, y_hat

def mask_top_k(to_rank, num_top_tokens, ctx ):
    # Select top tokens based on scores
    to_rank[ctx] = float('-inf')
    _, top_k_indices = torch.topk(to_rank, num_top_tokens)

    # Create a selection mask for top tokens, including context tokens
    selection_mask = torch.zeros_like(to_rank, dtype=torch.bool)
    selection_mask[top_k_indices] = True
    selection_mask[ctx] = True
    return selection_mask


def avg_topk(generated_ids, ctx, real_rate, feedback, avg_pooling, length_section):
    print("Masking rate: ", real_rate, " Length section: ", length_section)

    if real_rate == 1.0:
        real_rate = real_rate - (1/512)


    device = generated_ids.device

    with torch.no_grad():  # Ensure all operations are non-learnable
        # Evaluate tokens using the feedback model
        scores = feedback(generated_ids.unsqueeze(0), ctx.unsqueeze(0)).squeeze().to(device)

        # Calculate padding for 'same' mode
        padding = (avg_pooling - 1) // 2

        # Apply average pooling to smooth the scores
        kernel = torch.ones(avg_pooling, device=device) / avg_pooling
        smoothed_scores = torch.conv1d(scores.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=padding).squeeze()

        # Exclude context tokens from being selected
        smoothed_scores[ctx] = float('-inf')

        # Calculate the number of top tokens to select
        num_top_tokens = min(int(real_rate * length_section), scores.numel())

        # Select top tokens based on scores
        _, top_k_indices = torch.topk(smoothed_scores, num_top_tokens)

        # Create a selection mask for top tokens, including context tokens
        selection_mask = torch.zeros_like(smoothed_scores, dtype=torch.bool)
        selection_mask[top_k_indices] = True
        selection_mask[ctx] = True

    return selection_mask, smoothed_scores



def RefinPaint(song_ids_0, song_labels_0, song_mask_0, ctx_0, show=True, light=True,
               verbose=True, only1=False, avg_pooling=1, decoding="segment", iterations_to_run=1,
               human_in_the_loop=False, only_human=False):
    # Get a random index
    randint = random.randint(0, 1000000)
    experiment = "experiment"

    vocab = load_json('tokenizer_piano.json')['_vocab_base']
    song_metrics = {}
    device = "cuda:0"
    ids2tokens = {v: k for k, v in vocab.items()}
    zip_files = []

    if not light:
        song_ids_save = song_ids_0.clone()
        mask_bars_infilling = song_ids_save == 4
        song_ids_save[~np.concatenate([np.array([True]), ctx_0[:-1]])] = 0
        song_ids_save[mask_bars_infilling] = 4
        zip_files.append(
            save_midi_piano(
                song_ids_save.tolist(),
                tokenizer,
                f"proofreading/piece_to_infilling_{randint}.mid"
            )
        )
    algorithm1 = None
    if verbose:
        print("REAL TOKENS ORIGINAL", song_mask_0.sum())
    if show:
        plot_original(song_ids_0, tokenizer, f"proofreading/original_0_{randint}")
    if not light:
        zip_files.append(save_midi_piano(song_ids_0.tolist(), tokenizer, f"proofreading/original_0_{randint}.mid"))
    prev_mask = song_mask_0.clone()
    generator, feedback = get_experiment(device)
    generated_ids, tgt, song_mask, ctx = \
        song_ids_0.clone().to(device).long(), song_labels_0.to(device), song_mask_0.to(device), torch.from_numpy(ctx_0).to(device)
    if show:
        plot_tokenization(generated_ids.tolist(), song_mask, vocab)
    # other parameters
    max_seq_len = 512
    N = max_seq_len - torch.sum(song_mask).item()
    T = 11
    best_real = 0
    if verbose:
        print("ctx", ctx)
    with (torch.no_grad()):
        for it in reversed(range(1, T-iterations_to_run)):
            if verbose:
                print("Iteration ", it, "of ", T)
                print(" ")
            # Analyse
            gamma = lambda r: np.cos(r * np.pi / 2)
            k = math.ceil(gamma(it / T) * N)

            if decoding == "segment":
                song_mask, y_hat = feedback_avg_decoding(generated_ids, ctx, k / N,
                                                            feedback, avg_pooling, (~ctx).sum().item())
            elif decoding == "average_topk":
                song_mask, y_hat = avg_topk(generated_ids, ctx, k / N,
                                            feedback, avg_pooling, (~ctx).sum().item())

            if human_in_the_loop:
                if only_human:
                    song_mask = torch.tensor([True] * 512, device=song_mask.device)
                create_MIDI(
                    tokenizer, f"proofreading/current_mask_{it}.mid", randint, generated_ids.tolist(),
                    song_mask, ids2tokens, it
                )
                wait_for_file_modification(f"proofreading/current_mask_{it}.musicxml")
                sleep(1)
                song_mask = get_mask_saved_file(f"proofreading/current_mask_{it}.musicxml",
                                                generated_ids, song_mask, tokenizer)
                print("HUMAN IN THE LOOP")
                plot_tokenization(generated_ids.tolist(), song_mask, vocab)
                song_mask = torch.cat((song_mask[1:], torch.tensor([False], device=song_mask.device)))

            # generation
            print("song_mask", song_mask.sum().item())
            generated_ids = generator.generate(
                generated_ids.to(device),
                conditioning_mask=song_mask,
                tgt=tgt,
            )
            prev_mask = song_mask.clone()
            if it == T - 1 and algorithm1 is None:
                algorithm1 = generated_ids.clone()
            elif it == T - 1 and algorithm1 is not None:
                generated_ids = algorithm1.clone()

            y_pred_probs = torch.sigmoid(y_hat)
            real_score = torch.sum(y_pred_probs[~ctx]).item()
            if verbose:
                print("REAL SCORE feedback", real_score)
            if real_score > best_real:
                best_real = real_score
                best_generation = generated_ids.clone()
            tokens_real = (y_pred_probs[~ctx] > 0.5)
            real_count = torch.sum(tokens_real.int()).item()
            if verbose:
                print("REALS TOKENS", real_count, tokens_real.shape)

            if show:
                plot_tokenization(generated_ids.tolist(), song_mask, vocab)


            if it == T - 1:
                path_save = f"proofreading/algorithm1piano_{it}_{randint}.mid"
            elif it == 1:
                path_save = f"proofreading/algorithm3piano_{it}_{randint}.mid"
            else:
                path_save = f"proofreading/algorithm_intermediate_{it}_{randint}.mid"
            # if it == 1:
            #     generated_ids = best_generation
            if True:
                create_MIDI(
                    tokenizer, path_save, randint, generated_ids.tolist(),
                    prev_mask, ids2tokens, it
                )
            if not light:
                zip_files.append(
                    save_midi_piano(
                        generated_ids.tolist(),
                        tokenizer,
                        path_save.replace(".mid", f"_{experiment}.mid")
                    )
                )

            zip_files.append(
                save_piece_data(
                    generated_ids.tolist(),
                    song_labels_0.tolist(),
                    song_mask_0.tolist(),
                    ctx_0.tolist(),
                    path_save.replace(".mid", f"_{experiment}.json")
                )
            )

            metrics = {}
            metrics["real_score"] = real_score
            metrics["real_count"] = real_count
            song_metrics[it] = metrics
            tgt = generated_ids
            generated_ids = torch.concatenate((torch.tensor([1], device=generated_ids.device), generated_ids[:-1]))
            it -= 1
            if only1:
                break
        if verbose:
            print(song_metrics)
        save_json(song_metrics, f"proofreading/metrics_{experiment}_{randint}.json")
        zip_files.append(f"proofreading/metrics_{experiment}_{randint}.json")
        if not light:
            zip_files.append(plot_song_metrics(song_metrics, show, path_save=f"proofreading/metrics_{experiment}_{randint}.png"))

    with ZipFile(f"proofreading/{randint}.zip", "w") as zip_file:
        for file in zip_files:
            if os.path.exists(file):
                filename_only = os.path.basename(file)
                zip_file.write(file, arcname=filename_only)

def get_bar_index(new_ids, tokenizer):
    bar_id = tokenizer.vocab['Bar_None']
    bar_index = []
    for i, token in enumerate(new_ids):
        if token == bar_id:
            bar_index.append(i)
    return bar_index

def load_midi(path, bar_begin, bar_end):

    toks = tokenizer(path)[0]
    data_ids = toks.ids
    new_ids, new_mask = data_ids, np.ones_like(data_ids, dtype=bool)
    bar_index = get_bar_index(new_ids, tokenizer)
    # between bars 4 and 5 everything to True
    new_mask[bar_index[bar_begin]:bar_index[bar_end]] = False
    new_mask[bar_index] = True
    # zero padding up to 512
    new_labels = np.concatenate([new_ids, np.zeros(512 - len(new_ids), dtype=int)])
    new_mask = np.concatenate([new_mask, np.ones(512 - len(new_mask), dtype=bool)])
    # new labels is new ids adding start token (1)
    new_ids = np.concatenate([np.array([1]), new_labels[:-1]])
    return np.array(new_ids), new_labels, new_mask


import mido
from mido import MidiFile, MidiTrack, merge_tracks


def merge_midi_tracks_to_single_track(input_file_path, output_file_path):
    # Load the MIDI file
    midi = MidiFile(input_file_path)

    # Merge all tracks into one
    single_track = merge_tracks(midi.tracks)

    # Create a new MIDI file with a single track
    new_midi = MidiFile()
    new_track = MidiTrack()
    new_midi.tracks.append(new_track)

    # Copy all messages from the merged track to the new track
    for msg in single_track:
        new_track.append(msg)

    # Save the new MIDI file
    new_midi.save(output_file_path)



import argparse

def parse_arguments():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some parameters for music composition.")

    # Add arguments with default values
    parser.add_argument("--path", type=str, default="example.mid", help="Path to the MIDI file")
    parser.add_argument("--bar_begin", type=int, default=3, help="Starting bar number")
    parser.add_argument("--bar_end", type=int, default=5, help="Ending bar number")
    parser.add_argument("--confidence_about_your_composition", type=int, default=6, help="Confidence level about the composition")
    parser.add_argument("--human_in_the_loop", action='store_true', default=False, help="Include human in the loop processing")
    parser.add_argument("--only_human", action='store_true', default=False, help="Use only human-generated compositions")

    # Parse the arguments
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = parse_arguments()
    path, bar_begin, bar_end, confidence_about_your_composition, human_in_the_loop, only_human = \
        args.path, args.bar_begin, args.bar_end, args.confidence_about_your_composition, args.human_in_the_loop, args.only_human

    merged_path = path.replace(".mid", "_merged.mid")
    merge_midi_tracks_to_single_track(path, merged_path)
    new_ids, new_labels, new_mask = load_midi(merged_path, bar_begin=bar_begin, bar_end=bar_end)

    RefinPaint(
        torch.from_numpy(new_ids),
        torch.from_numpy(new_labels),
        torch.from_numpy(new_mask),
        new_mask,
        show=True,
        avg_pooling=1,
        light=False,
        verbose=True,
        only1=False,
        decoding="average_topk",
        iterations_to_run=confidence_about_your_composition,
        human_in_the_loop=human_in_the_loop,
        only_human=only_human
    )

