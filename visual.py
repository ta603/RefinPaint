import os
import torch
from miditoolkit import MidiFile
from music21 import converter

from MidiTok_modified.miditok import TokSequence

import time


def wait_for_file_modification(filename):
    """
    Block until the file specified by filename is modified.
    :param filename: the name of the file to monitor.
    :param check_interval: interval between checks in seconds.
    """
    last_mod_time = os.path.getmtime(filename)

    while True:
        time.sleep(1)
        current_mod_time = os.path.getmtime(filename)
        if current_mod_time != last_mod_time:
            break


def get_instrument_info(path):
    # Load the MIDI file
    mid = MidiFile(path)

    # List to store instruments
    instruments = []

    # Iterate through each track in the MIDI file
    for track in mid.tracks:
        for msg in track:
            # Check if the message is a 'program change' which signals an instrument change
            if msg.type == 'program_change':
                instruments.append(msg.program)

    # Remove duplicates
    unique_instruments = list(set(instruments))

    # Print the instruments
    for instrument in unique_instruments:
        print(f"Instrument Program Number: {instrument}")


def plot_tokenization(data_ids, data_mask, vocab_custom):
    GREEN = "\033[92m"
    END = "\033[0m"
    YELLOW = "\033[93m"
    vocab_custom = {v: k for k, v in vocab_custom.items()}
    for idx, (ii, mm) in enumerate(zip(data_ids, data_mask)):
        if "Chord" in vocab_custom[ii]:
            print(YELLOW + vocab_custom[ii] + END, end=" ")
        elif mm:
            print(GREEN + vocab_custom[ii] + END, end=" ")
        else:
            print(vocab_custom[ii], end=" ")
    print()


def is_more_or_less_red(color_hex):
    # Ensure the input is in the correct format
    if not isinstance(color_hex, str) or not color_hex.startswith('#') or len(color_hex) != 7:
        raise ValueError("Color must be a string in the format #RRGGBB")

    # Convert hex to RGB
    red = int(color_hex[1:3], 16)
    green = int(color_hex[3:5], 16)
    blue = int(color_hex[5:7], 16)

    # Define conditions for "more or less red"
    red_dominant = red > 200  # Red is considered dominant if it's very high
    other_colors_low = green < 50 and blue < 50  # Other colors are low if below this threshold

    return red_dominant and other_colors_low

def get_mask_saved_file(path, song_ids, previous_mask, ids2tokens):
    mscz_path = path.replace(".musicxml", ".mscz")
    # check red notes
    os.system(f"cp {path} {mscz_path}")
    os.system(f"mscore3 {mscz_path} -o {path}")

    sc = converter.parse(path)
    om = strm2map(sc)
    pitch_masked = []
    song_mask = torch.ones_like(song_ids, dtype=torch.bool)
    for oo in om:
        if is_more_or_less_red(oo['element'].style.color):
            pitch_masked.append(True)
        else:
            pitch_masked.append(False)
    # map red notes into tokenization
    ii_pitch_mask = 0
    song_ids_list = song_ids.tolist()
    for ii, gg in enumerate(song_ids_list):
        if "Pitch" in ids2tokens[gg]:
            if ii_pitch_mask < len(pitch_masked) and pitch_masked[ii_pitch_mask]:
                song_mask[ii] = False
                if ii != 0 and "Position" in ids2tokens[song_ids_list[ii - 1]]:
                    song_mask[ii - 1] = False
                if ii + 1 != len(song_mask):
                    song_mask[ii + 1] = False
            ii_pitch_mask += 1
    # add red notes to previous mask
    song_mask = song_mask & previous_mask
    return song_mask


def plot_original(song_ids, tokenizer, path, multi=False):

    if not multi:
        remiplusplus_tokens = TokSequence(ids=song_ids.tolist())
        instrument, t_changes = tokenizer.tokens_to_track(remiplusplus_tokens)
        midi = MidiFile(ticks_per_beat=384)

        midi.instruments = []
        midi.instruments.append(instrument)

        midi.tempo_changes = t_changes
        # midi.time_signature_changes = time_signature_changes
        midi.max_tick = max(
            [
                max([note.end for note in track.notes] + [0]) if len(track.notes) > 0 else 0
                for track in midi.instruments
            ]
        )
        midi.dump(path + ".mid")
    else:
        song_ids = remove_drum_from_sequence(song_ids.tolist(), tokenizer)
        remiplusplus_tokens = TokSequence(ids=song_ids)
        midi = tokenizer.tokens_to_midi(remiplusplus_tokens)
        midi.dump(path + ".mid")

    os.system(f"mscore3 {path}.mid -o {path}.mscz")
    os.system(f"mscore3 {path}.mscz")


def remove_drum_from_sequence(sequence, tokenizer):
    token_drum = 181
    position_ids = [k for k, v in tokenizer.vocab.items() if "Position" in k]
    for ii, _ in enumerate(sequence):
        if sequence[ii] == token_drum:
            if ii - 1 < len(sequence) and sequence[ii - 1] in position_ids:
                sequence[ii - 1] = 0
            sequence[ii] = 0
            if ii + 1 < len(sequence) < ii + 1:
                sequence[ii + 1] = 0
            if ii + 2 < len(sequence):
                sequence[ii + 2] = 0
    return sequence



def create_MIDI(tokenizer, path, randint, generated_ids, song_mask, ids2tokens, it=0, multi=False, show=True):
    if multi:
        generated_ids = remove_drum_from_sequence(generated_ids, tokenizer)
        remiplusplus_tokens = TokSequence(ids=generated_ids)
        midi = tokenizer.tokens_to_midi(remiplusplus_tokens)
    else:
        remiplusplus_tokens = TokSequence(ids=generated_ids)
        instrument, t_changes = tokenizer.tokens_to_track(remiplusplus_tokens)
        midi = MidiFile(ticks_per_beat=384)

        midi.instruments = []
        midi.instruments.append(instrument)

        midi.tempo_changes = t_changes
        # midi.time_signature_changes = time_signature_changes
        midi.max_tick = max(
            [
                max([note.end for note in track.notes] + [0]) if len(track.notes) > 0 else 0
                for track in midi.instruments
            ]
        )

    midi.dump(path)
    # os.system(f"mscore3 {path}")
    drum_program_token = 181 # tokenizer.vocab["Program_-1"]
    # # how much notes are going to be inpainted
    pitches_generated = []
    for jj, gg in enumerate(generated_ids):
        if "Pitch" in ids2tokens[gg] and generated_ids[jj - 1] != drum_program_token:
            have_position = "Position" in ids2tokens[generated_ids[jj - 1]]
            if (((have_position and song_mask[jj - 1]) or not have_position) and song_mask[jj] and jj + 1 != len(generated_ids) and
                    song_mask[jj + 1]):
                pitches_generated.append("not_generated")

            elif not song_mask[jj - 1] and not song_mask[jj] and jj + 1 != len(generated_ids) and not song_mask[jj + 1]:
                pitches_generated.append("full_generated")
            else:
                pitches_generated.append("partial_generated")

    # os.system(f"mscore3 output/{it}_algorithm3_{randint}.mid")
    musicxml_path = path.replace(".mid", ".musicxml")
    os.system(f"mscore3 {path} -o {musicxml_path}")

    from music21 import converter
    sc = converter.parse(musicxml_path)

    om = strm2map(sc)

    # color all notes of red
    # for oo in om:
    #     oo['element'].style.color = 'red'

    for oo, pp in zip(om, pitches_generated):
        if pp == "not_generated":
            oo['element'].style.color = 'green'
        elif pp == "partial_generated":
            oo['element'].style.color = 'yellow'
        elif pp == "full_generated":
            oo['element'].style.color = '#FF0000'

    sc.write('musicxml', fp=musicxml_path)

    if show:
        os.system(f"mscore3 {musicxml_path}")


def remove_drum(path):
    # Load the MIDI file
    from mido import MidiFile
    mid = MidiFile(path)

    # Create a new MIDI file without the drum track
    new_mid = MidiFile()

    for i, track in enumerate(mid.tracks):
        # Check if the current track is the 10th one
        # Note: In programming, lists are zero-based, so track 10 is index 9
        if i != 9:
            new_mid.tracks.append(track)

    # Save the new MIDI file without the drum track
    path = path.replace(".mid", "_nodrum.mid")
    new_mid.save(path)


def save_midi_piano(song_ids, tokenizer, path):
    try:
        remiplusplus_tokens = TokSequence(ids=song_ids)
        instrument, t_changes = tokenizer.tokens_to_track(remiplusplus_tokens)
        midi = MidiFile(ticks_per_beat=384)

        midi.instruments = []
        midi.instruments.append(instrument)

        midi.tempo_changes = t_changes
        # midi.time_signature_changes = time_signature_changes
        midi.max_tick = max(
            [
                max([note.end for note in track.notes] + [0]) if len(track.notes) > 0 else 0
                for track in midi.instruments
            ]
        )
        midi.dump(path)
        return path
    except:
        return None


def save_midi_multi(song_ids, tokenizer, path):
    remiplusplus_tokens = TokSequence(ids=song_ids)
    midi = tokenizer.tokens_to_midi(remiplusplus_tokens)
    midi.dump(path)
    os.system(f"mscore3 {path}")
    return path

def strm2map(strm):
    import music21
    converted = []
    om = []
    chordID = 0
    for o in strm.flatten().secondsMap:
        # if hasattr(o['element'], 'getInstrument'):
        #     print(o['element'].getInstrument())
        if music21.note.Note in o['element'].classSet and hasattr(o['element'], 'getInstrument'):
            o['chordID'] = chordID
            om.append(o)
            chordID += 1
        elif music21.chord.Chord in o['element'].classSet and hasattr(o['element'], 'getInstrument'):
            om_chord = [
                {
                    'element': oc,
                    'offsetSeconds': o['offsetSeconds'],
                    'endTimeSeconds': o['endTimeSeconds'],
                    'chord': o['element'],
                    'chordID': chordID,
                }
                for oc in zip(sorted(o['element'].notes, key=lambda a: a.pitch))
            ]
            om.extend(om_chord)
            chordID += 1
        elif type(o['element']) == music21.note.Unpitched:
            # print(type(o["element"]))
            pass
    om_filtered = []
    for o in om:
        if type(o['element']) == tuple:
            o['element'] = o['element'][0]

        if not (o['element'].tie and (o['element'].tie.type == 'continue' or o['element'].tie.type == 'stop')) and \
                not ((hasattr(o['element'], 'tie') and o['element'].tie
                      and (o['element'].tie.type == 'continue' or o['element'].tie.type == 'stop'))) and \
                not (o['element'].duration.quarterLength == 0):
            om_filtered.append(o)

    return sorted(om_filtered, key=lambda a: (a['offsetSeconds'], a['element'].pitch))


if __name__ == '__main__':
    midi_path = "/home/pedro/PycharmProjects/LAKH-MuseNet-MIDI-Dataset/Output/f/f782ee623347e2a86e8b408c9726f0b9.mid"
    xml_path = "output.musicxml"
    print("coonvert to mscz")
    os.system(f"mscore3 {midi_path} -o {xml_path}")

    sc = converter.parse(xml_path)
    om = strm2map(sc)
    for idx, oo in enumerate(om):
        if idx % 4 == 0:
            oo['element'].style.color = 'red'
        elif idx % 4 == 1:
            oo['element'].style.color = 'blue'
        elif idx % 4 == 2:
            oo['element'].style.color = 'green'
        elif idx % 4 == 3:
            oo['element'].style.color = 'yellow'
    sc.show()
