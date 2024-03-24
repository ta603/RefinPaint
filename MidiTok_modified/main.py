import json
import os

from MidiTok_modified.miditok import REMI, REMIPlusPlus, REMIPlus
from miditoolkit import MidiFile

from MidiTok_modified.miditok.tokenizations.remi_plus_five_regularly import REMIPlusFiveRegularly
from MidiTok_modified.miditok.tokenizations.remi_plus_pcp import REMIPlusPCP
from compute_embellish import get_pcp
from compute_tokenization import save_mask_conditioning, order_melody_first

# Creates the tokenizer and loads a MIDI
tokenizer = REMIPlusPCP(
        pitch_range=range(21, 109),
        beat_res={(0, 3): 8, (4, 12): 4},
        nb_velocities=32,
        additional_tokens={
            'Chord': True,
            'Program': False,
            'Rest': False,
            'Tempo': False,
            'TimeSignature': False,
            'chord_maps': {
                '7aug': (0, 4, 8, 11),
                '7dim': (0, 3, 6, 9),
                '7dom': (0, 4, 7, 10),
                '7halfdim': (0, 3, 6, 10),
                '7maj': (0, 4, 7, 11),
                '7min': (0, 3, 7, 10),
                '9maj': (0, 4, 7, 10, 14),
                '9min': (0, 4, 7, 10, 13),
                'aug': (0, 4, 8),
                'dim': (0, 3, 6),
                'maj': (0, 4, 7),
                'min': (0, 3, 7),
                'sus2': (0, 2, 7),
                'sus4': (0, 5, 7)},
            'chord_tokens': True,
            'chord_tokens_with_root_note': True,
            'chord_unknown': False,
            'nb_tempos': 32,
            'programs': [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                         25,
                         26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                         50,
                         51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
                         75,
                         76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                         100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                         119,
                         120, 121, 122, 123, 124, 125, 126, 127],
            'rest_range': (2, 8),
            'tempo_range': (40, 250),
            'time_signature_range': (8, 2)
        },
        special_tokens=['PAD', 'BOS', 'EOS', 'MASK', 'Chord'],
        params=None,
        max_bar_embedding=None)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


midi_path = '../lmd_separated/f/fe1a2ee4f92c538d0fc44eaed25da775.mid'
melody_program = load_json('../lmd_separated/f/program_result.json')['lmd_full/f/fe1a2ee4f92c538d0fc44eaed25da775.mid']
chords_file = '../lmd_chord/f/fe1a2ee4f92c538d0fc44eaed25da775.txt'

print(f"Melody program: {melody_program}")

# Converts MIDI to tokens, and back to a MIDI
toks = tokenizer(midi_path, midi_miner=melody_program, chords_file=chords_file)  # calling it will automatically detect MIDIs, paths and tokens before the conversion
data_ids = toks.ids
print(toks.events[:100])
# for en, (ii) in enumerate(data_ids):
#     print(en, ii, tokenizer[ii])

mask, mask_melody = save_mask_conditioning(tokenizer, data_ids, melody_program["melody"])

def plot_tokenization(data_ids, data_mask, melody_mask=None, vocab=None):
    GREEN = "\033[92m"
    END = "\033[0m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    if melody_mask is not None:
        zip_iter = zip(data_ids, data_mask, melody_mask)
    else:
        zip_iter = zip(data_ids, data_mask)
    for idx, batch in enumerate(zip_iter):
        if melody_mask is not None:
            ii, mm, mm_melody = batch
        else:
            ii, mm = batch
        if melody_mask is not None and mm_melody:
            is_bold = BOLD
        else:
            is_bold = ""
        if "Chord" in vocab[ii]:
            print(is_bold + YELLOW + vocab[ii] + END, end=" ")
        elif "Bar" in vocab[ii]:
            print(is_bold + RED + vocab[ii] + END, end=" ")
        elif mm:
            print(is_bold + GREEN + vocab[ii] + END, end=" ")
            # print(GREEN, idx, ii, vocab[ii], mm, END)
        else:
            print(vocab[ii], end=" ")
    print()

# plot_tokenization(data_ids, mask, mask_melody, {k: v for v, k in tokenizer.vocab.items()})

# for en, (ii, mm) in enumerate(zip(data_ids, mask_melody)):
#     print(en, ii, tokenizer[ii], mm)

# new_ids, new_mask = order_melody_first(data_ids, mask, mask_melody)
new_ids, new_mask = data_ids, mask
print(get_pcp(data={'ids': new_ids}, tokenizer={v: k for v, k in tokenizer.vocab.items()})['pcps'])
plot_tokenization(new_ids, new_mask, vocab={k: v for v, k in tokenizer.vocab.items()})
# assert len(data_ids) == len(new_ids), "Error in mask melody proccess"
#
# for en, (ii, mm) in enumerate(zip(new_ids, new_mask)):
#     print(en, ii, tokenizer[ii], mm)

# for ii, idx in enumerate(data["ids"]):
#     print( idx, tokenizer[idx])

