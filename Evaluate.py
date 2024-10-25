import pretty_midi
import muspy
from collections import Counter

def scale_to_index(scale):
    major_scales = {
        'C major': 0, 'C# major': 1, 'Db major':1, 'D major': 2, 'D# major': 3, 'Eb major': 3,
        'E major': 4, 'F major': 5, 'F# major': 6,'Gb major': 6, 'G major': 7,
        'G# major': 8, 'Ab major': 8, 'A major': 9, 'A# major': 10, 'Bb major': 10, 'B major': 11
    }

    minor_scales = {
        'C minor': 3, 'C# minor': 4,'Db minor': 4, 'D minor': 5, 'D# minor': 6, 'Eb minor': 6,
        'E minor': 7, 'F minor': 8, 'F# minor': 9, 'Gb minor': 9, 'G minor': 10,
        'G# minor': 11, 'Ab minor': 11, 'A minor': 0, 'A# minor': 1, 'Bb minor': 1, 'B minor': 2
    }

    if scale in major_scales:
        return major_scales[scale]
    elif scale in minor_scales:
        return minor_scales[scale]
    else:
        raise ValueError("未知的音阶名称: " + scale)

midi_path = "/home/kinnryuu/ダウンロード/ICASSP_2025_Model/polyffusion/exp/3.mid"

midi = pretty_midi.PrettyMIDI(midi_path)

pitch_counter = Counter()

# 遍历所有乐器中的所有音符
for instrument in midi.instruments:
    for note in instrument.notes:
        # 将音高归一化到一个八度（0到11表示C到B）
        normalized_pitch = note.pitch % 12
        pitch_counter[normalized_pitch] += 1

# 针对12个可能的调号，计算其与音高分布的相似性
major_key_profiles = {
    'C major': [0, 2, 4, 5, 7, 9, 11],
    'G major': [7, 9, 11, 0, 2, 4, 6],
    'D major': [2, 4, 6, 7, 9, 11, 1],
    'A major': [9, 11, 1, 2, 4, 6, 8],
    'E major': [4, 6, 8, 9, 11, 1, 3],
    'B major': [11, 1, 3, 4, 6, 8, 10],
    'F major': [5, 7, 9, 10, 0, 2, 4],
    'Bb major': [10, 0, 2, 3, 5, 7, 9],
    'Eb major': [3, 5, 7, 8, 10, 0, 2],
    'Ab major': [8, 10, 0, 1, 3, 5, 7],
    'Db major': [1, 3, 5, 6, 8, 10, 0],
    'Gb major': [6, 8, 10, 11, 1, 3, 5],
}

# 计算每个调号的匹配得分
def score_key(pitch_counter, key_profile):
    score = 0
    for pitch in key_profile:
        score += pitch_counter[pitch]
    return score

# 找到最佳匹配的调号
best_key = max(major_key_profiles, key=lambda k: score_key(pitch_counter, major_key_profiles[k]))


key = scale_to_index(best_key)
ts = midi.time_signature_changes[0].numerator/midi.time_signature_changes[0].denominator
tempo = midi.get_tempo_changes()[1][0]

print(best_key)
print(ts)
print(tempo)