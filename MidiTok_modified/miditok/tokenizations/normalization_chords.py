import os
from collections import Counter

# chords = list()
# count = 0
# for dir in os.listdir('lmd_chord'):
#     for p in os.listdir(f'lmd_chord/{dir}'):
#         if p.endswith('.txt'):
#             count += 1
#             # read tsv
#             with open(f'lmd_chord/{dir}/{p}', 'r') as f:
#                 lines = f.readlines()
#                 for l in lines:
#                     chords.append(l[:-1].split('\t')[-1])
# print(count)
#
# # order set by first letter
# chord_types = sorted(set(chords), key=lambda x: x[0])
# print(len(chord_types))
# print(chord_types)
#
# # plot histogram of chords
# print(Counter(chords).most_common(300))

# root = set([l.split(':')[-1] for l in lst if l != 'N'])
# print(root)


large_vocabulary = ['Ab:maj6', 'Ab:hdim7', 'Ab:min7/b7', 'A:minmaj7', 'Ab:maj9', 'Ab:sus4', 'A:min(11)', 'A:sus4(b7,9)', 'Ab:7/5', 'A:dim', 'A:maj(9)', 'Ab:maj7/7', 'Ab:maj7', 'A:maj9(11)', 'Ab:maj9(11)', 'A:7/3', 'A:7/5', 'A:min/5', 'Ab:maj13', 'Ab:min/5', 'A:9', 'Ab:maj/5', 'Ab:sus4(b7,9)', 'Ab:maj/3', 'A:maj/5', 'A:min9', 'Ab:7/3', 'Ab:7/b7', 'Ab:min7/5', 'A:min/b3', 'Ab:aug', 'A:11', 'A:maj/3', 'Ab:maj', 'A:sus4(b7)', 'Ab:7(#9)', 'Ab:min', 'A:dim7', 'Ab:min(11)', 'A:min7', 'A:min13', 'Ab:sus4(b7)', 'Ab:maj6(9)', 'A:aug', 'Ab:min/b3', 'Ab:dim', 'A:maj6', 'A:min6', 'Ab:11', 'A:maj7/3', 'Ab:7', 'A:maj7/7', 'A:maj7/5', 'A:sus2', 'A:maj(11)', 'A:hdim7', 'Ab:min9', 'Ab:min(9)', 'A:min(9)', 'A:min7/b7', 'A:sus4', 'Ab:min6', 'A:min11', 'A:maj7', 'Ab:dim7', 'A:maj9', 'Ab:sus2', 'Ab:min6(9)', 'A:7/b7', 'Ab:min11', 'A:maj', 'A:min6(9)', 'Ab:min13', 'A:min', 'A:min7/5', 'Ab:min7', 'A:maj13', 'Ab:maj(9)', 'A:maj6(9)', 'Ab:9', 'Ab:minmaj7', 'Ab:maj7/3', 'Ab:13', 'A:7(#9)', 'Ab:maj7/5', 'A:7', 'A:13', 'Ab:maj(11)', 'B:min(11)', 'B:sus4', 'Bb:maj7/5', 'B:min13', 'B:maj/3', 'Bb:7', 'Bb:7/b7', 'B:7/3', 'Bb:7(#9)', 'B:dim7', 'Bb:sus4(b7)', 'Bb:maj7/3', 'B:sus4(b7,9)', 'Bb:min13', 'B:maj', 'Bb:min6', 'Bb:maj(9)', 'B:maj(9)', 'B:9', 'B:maj(11)', 'B:7/5', 'B:7/b7', 'Bb:min7', 'Bb:7/3', 'Bb:maj(11)', 'Bb:hdim7', 'B:maj7/3', 'B:maj6(9)', 'B:13', 'B:7', 'B:maj/5', 'B:min6', 'B:min/5', 'Bb:maj6', 'Bb:maj13', 'B:min', 'Bb:sus4', 'B:hdim7', 'Bb:min11', 'Bb:aug', 'Bb:maj7/7', 'Bb:maj7', 'B:min9', 'Bb:9', 'B:min7', 'Bb:dim', 'Bb:sus4(b7,9)', 'Bb:maj6(9)', 'Bb:maj9', 'Bb:13', 'B:sus4(b7)', 'B:min6(9)', 'Bb:min7/b7', 'B:min7/5', 'B:sus2', 'B:min/b3', 'Bb:maj', 'B:maj9', 'B:min(9)', 'Bb:min(9)', 'B:maj7/5', 'Bb:maj9(11)', 'Bb:min(11)', 'B:aug', 'B:11', 'B:maj9(11)', 'B:maj6', 'Bb:min9', 'Bb:sus2', 'B:maj7', 'B:maj13', 'Bb:min', 'B:dim', 'Bb:11', 'Bb:min7/5', 'Bb:min6(9)', 'B:min7/b7', 'Bb:min/b3', 'Bb:7/5', 'Bb:maj/3', 'B:maj7/7', 'Bb:min/5', 'Bb:minmaj7', 'Bb:maj/5', 'Bb:dim7', 'B:7(#9)', 'B:minmaj7', 'B:min11', 'C:minmaj7', 'C#:sus4', 'C#:maj13', 'C:min7/b7', 'C#:7(#9)', 'C#:min(11)', 'C:9', 'C#:min11', 'C:min', 'C:7', 'C#:min7/5', 'C:min7', 'C:7/5', 'C#:7', 'C#:aug', 'C:min6(9)', 'C:min9', 'C#:dim7', 'C:sus4', 'C#:min7', 'C:maj/3', 'C:11', 'C#:sus2', 'C#:7/3', 'C#:11', 'C#:maj/5', 'C#:maj(11)', 'C:maj7', 'C:maj(11)', 'C:sus4(b7)', 'C#:maj6(9)', 'C#:min9', 'C#:min(9)', 'C:maj6(9)', 'C#:min/b3', 'C:maj7/3', 'C:maj6', 'C:dim7', 'C:min/b3', 'C:maj13', 'C#:maj9(11)', 'C#:maj(9)', 'C#:maj7/5', 'C:maj7/5', 'C:maj9', 'C:hdim7', 'C#:maj/3', 'C:min(9)', 'C#:9', 'C#:13', 'C:min13', 'C#:7/b7', 'C:maj', 'C:7(#9)', 'C#:min7/b7', 'C:min/5', 'C#:maj7/7', 'C:min11', 'C:min6', 'C:sus2', 'C#:minmaj7', 'C#:maj9', 'C#:sus4(b7)', 'C#:maj7', 'C#:min', 'C#:min6(9)', 'C:7/b7', 'C#:maj6', 'C:min7/5', 'C:13', 'C#:maj', 'C#:maj7/3', 'C:sus4(b7,9)', 'C:maj(9)', 'C#:dim', 'C#:min6', 'C#:sus4(b7,9)', 'C#:hdim7', 'C#:min13', 'C:dim', 'C#:min/5', 'C:min(11)', 'C:maj/5', 'C:maj7/7', 'C:7/3', 'C:maj9(11)', 'C:aug', 'C#:7/5', 'D:maj7/7', 'D:sus4(b7,9)', 'D:7', 'D:min7/b7', 'D:dim7', 'D:min/b3', 'D:min', 'D:sus4(b7)', 'D:maj7', 'D:7/5', 'D:min6', 'D:min9', 'D:min(9)', 'D:7(#9)', 'D:min7/5', 'D:maj9', 'D:maj/5', 'D:maj6', 'D:min6(9)', 'D:maj/3', 'D:9', 'D:maj(9)', 'D:min11', 'D:hdim7', 'D:13', 'D:min13', 'D:maj7/5', 'D:7/3', 'D:sus4', 'D:maj9(11)', 'D:11', 'D:maj7/3', 'D:minmaj7', 'D:maj(11)', 'D:dim', 'D:aug', 'D:maj', 'D:sus2', 'D:maj13', 'D:maj6(9)', 'D:min(11)', 'D:min/5', 'D:min7', 'D:7/b7', 'Eb:7', 'E:7', 'Eb:maj', 'E:min7/5', 'Eb:7(#9)', 'E:aug', 'Eb:maj7/3', 'E:min7/b7', 'Eb:dim7', 'E:min7', 'Eb:maj/3', 'E:maj(11)', 'E:sus4', 'E:min', 'Eb:hdim7', 'E:maj7', 'Eb:min7/5', 'Eb:maj9', 'E:min/5', 'E:maj6', 'Eb:maj7', 'E:maj6(9)', 'E:13', 'Eb:minmaj7', 'E:maj9(11)', 'E:11', 'E:sus4(b7)', 'Eb:11', 'E:dim', 'E:maj/5', 'Eb:min13', 'Eb:min6', 'Eb:7/b7', 'E:maj(9)', 'Eb:aug', 'E:sus2', 'E:maj13', 'E:min13', 'E:maj9', 'Eb:13', 'E:min11', 'E:dim7', 'Eb:min6(9)', 'Eb:maj7/5', 'Eb:dim', 'Eb:min/5', 'E:min/b3', 'Eb:maj/5', 'Eb:min7', 'Eb:7/3', 'E:maj7/7', 'Eb:min11', 'Eb:maj(11)', 'E:maj7/3', 'E:min6(9)', 'Eb:9', 'E:minmaj7', 'E:min(9)', 'Eb:maj6(9)', 'Eb:sus2', 'E:maj/3', 'E:7/5', 'Eb:min(11)', 'Eb:sus4', 'E:hdim7', 'Eb:min7/b7', 'Eb:sus4(b7,9)', 'E:9', 'Eb:maj9(11)', 'E:maj', 'E:maj7/5', 'Eb:min/b3', 'Eb:min9', 'E:min9', 'Eb:sus4(b7)', 'Eb:maj7/7', 'E:7/b7', 'E:min6', 'Eb:min', 'Eb:maj13', 'Eb:maj6', 'E:sus4(b7,9)', 'E:7/3', 'E:7(#9)', 'Eb:maj(9)', 'Eb:7/5', 'E:min(11)', 'Eb:min(9)', 'F#:dim7', 'F:maj', 'F#:min9', 'F#:13', 'F#:maj7/5', 'F#:min6', 'F:maj/3', 'F#:maj7/3', 'F#:minmaj7', 'F:maj(11)', 'F:7(#9)', 'F:maj7', 'F:min/b3', 'F:7/3', 'F:sus4(b7,9)', 'F#:maj(9)', 'F#:min/5', 'F:9', 'F#:maj/3', 'F:maj9(11)', 'F:min13', 'F:maj(9)', 'F#:maj/5', 'F#:maj6', 'F:maj6', 'F#:sus4', 'F:min(9)', 'F:7/5', 'F#:min(9)', 'F#:maj7/7', 'F#:min7/b7', 'F#:sus4(b7)', 'F#:min7', 'F#:7/5', 'F#:min6(9)', 'F:hdim7', 'F:min6', 'F:maj7/5', 'F#:maj9(11)', 'F:11', 'F:min11', 'F#:aug', 'F:maj13', 'F#:maj(11)', 'F#:min(11)', 'F#:maj7', 'F#:sus2', 'F#:maj13', 'F:13', 'F#:maj9', 'F:dim7', 'F:maj9', 'F:min7', 'F:maj7/3', 'F#:dim', 'F:min9', 'F#:11', 'F:min', 'F#:min7/5', 'F:7', 'F#:min13', 'F:min(11)', 'F:min7/5', 'F:maj/5', 'F:sus2', 'F:min6(9)', 'F#:min11', 'F#:9', 'F:min7/b7', 'F:sus4(b7)', 'F:maj6(9)', 'F#:maj6(9)', 'F#:min/b3', 'F:min/5', 'F#:min', 'F#:hdim7', 'F#:7/b7', 'F#:maj', 'F:dim', 'F:sus4', 'F:minmaj7', 'F#:7(#9)', 'F#:7/3', 'F:aug', 'F#:7', 'F:maj7/7', 'F:7/b7', 'F#:sus4(b7,9)', 'G:7', 'G:maj', 'G:min(11)', 'G:min7/b7', 'G:maj(11)', 'G:maj13', 'G:minmaj7', 'G:sus4(b7,9)', 'G:min6(9)', 'G:maj9(11)', 'G:maj/3', 'G:maj7/5', 'G:maj/5', 'G:aug', 'G:maj6(9)', 'G:maj7/7', 'G:hdim7', 'G:min(9)', 'G:min/5', 'G:dim', 'G:7/3', 'G:13', 'G:7/5', 'G:sus4', 'G:maj6', 'G:min13', 'G:9', 'G:min/b3', 'G:maj(9)', 'G:min11', 'G:dim7', 'G:min9', 'G:min7', 'G:maj7/3', 'G:7/b7', 'G:sus4(b7)', 'G:maj7', 'G:min', 'G:sus2', 'G:maj9', 'G:7(#9)', 'G:min6', 'G:min7/5', 'G:11', 'N']

short_vocabulary = {
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
    'sus4': (0, 5, 7)
}


map_function = {
    '13': '7dom',  # Not perfect: 13th notes reduced to 7th
    'maj': 'maj',
    '7': '7dom',
    'min11': '7min',  # Not perfect: 11th notes reduced to 7th
    'maj(9)': '9maj',
    'min7/5': '7min',  # Not perfect: slash notation typically indicates an inversion or a specific bass note
    'min7/b7': '7min',  # Not perfect: slash notation typically indicates an inversion or a specific bass note
    'aug': 'aug',
    'min7': '7min',
    'maj(11)': '7maj',  # Not perfect: 11th notes reduced to 7th
    'maj6(9)': '9maj',  # Not perfect: 6th note reduced
    'min(9)': '9min',
    'min/5': 'min',  # Not perfect: slash notation typically indicates an inversion or a specific bass note
    '7/3': '7dom',  # Not perfect: slash notation typically indicates an inversion or a specific bass note
    '7/b7': '7dom',  # Not perfect: slash notation typically indicates an inversion or a specific bass note
    'maj9': '9maj',
    'maj7/7': '7maj',  # Not perfect: slash notation typically indicates an inversion or a specific bass note
    'dim': 'dim',
    'maj/3': 'maj',  # Not perfect: slash notation typically indicates an inversion or a specific bass note
    'minmaj7': '7maj',  # Not perfect: this chord is a minor with a major 7th, doesn't fit well into short_vocabulary
    'maj/5': 'maj',  # Not perfect: slash notation typically indicates an inversion or a specific bass note
    'maj9(11)': '9maj',  # Not perfect: 11th notes reduced
    'min13': '7min',  # Not perfect: 13th notes reduced to 7th
    'maj13': '7maj',  # Not perfect: 13th notes reduced to 7th
    'maj6': 'maj',  # Not perfect: 6th note ignored
    'min9': '9min',
    '11': '7dom',  # Not perfect: 11th notes reduced to 7th
    'maj7': '7maj',
    'sus4': 'sus4',
    'maj7/5': '7maj',  # Not perfect: slash notation typically indicates an inversion or a specific bass note
    'min/b3': 'min',  # Not perfect: slash notation typically indicates an inversion or a specific bass note
    'sus4(b7)': '7dom',  # Not perfect: sus4 chord reduced to a dominant 7th chord
    '9': '9maj',
    'maj7/3': '7maj',  # Not perfect: slash notation typically indicates an inversion or a specific bass note
    'min': 'min',
    'dim7': '7dim',
    'sus4(b7,9)': '9maj',  # Not perfect: sus4 chord reduced to a major 9th chord
    'hdim7': '7halfdim',
    '7/5': '7dom',  # Not perfect: slash notation typically indicates an inversion or a specific bass note
    'min(11)': '7min',  # Not perfect: 11th notes reduced to 7th
    'min6(9)': '9min',  # Not perfect: 6th note reduced
    'min6': 'min',  # Not perfect: 6th note ignored
    'sus2': 'sus2',
    '7(#9)': '9maj'  # Not perfect: the #9 (altered note) is treated as a regular 9th
}

root_map = {
    'C': 'C',
    'C#': 'C#',
    'Db': 'C#',
    'D': 'D',
    'D#': 'D#',
    'Eb': 'D#',
    'E': 'E',
    'F': 'F',
    'F#': 'F#',
    'Gb': 'F#',
    'G': 'G',
    'G#': 'G#',
    'Ab': 'G#',
    'A': 'A',
    'A#': 'A#',
    'Bb': 'A#',
    'B': 'B'
}


def normalization_chord(ch):
    """
    Returns a dictionary with the chord names as keys and the normalized chord names as values.
    """
    ans = ""
    if ch == 'N':
        return 'None'
    else:
        root, quality = ch.split(':')
        return f"{root_map[root]}:{map_function[quality]}"


