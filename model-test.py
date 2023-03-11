
import os
import numpy as np
import matplotlib.pyplot as plt

from pydub import AudioSegment
from scipy.stats import gaussian_kde, pearsonr
from model import Model, TRF, Tool

def test_spectrogram_generate(test_wave_file='./assets/FDRW0-SA2.wav'):
    frame_len, time_const, shift = 5, 8, 0
    audio_chunk_raw = AudioSegment.from_file(test_wave_file)
    audio_chunk = np.array(audio_chunk_raw.get_array_of_samples())
    spectrogram, freq_scale = Model.compute_spectrogram(audio_chunk, params=[frame_len, time_const, shift])
    plt.figure(figsize=(audio_chunk_raw.duration_seconds*4, 4), dpi=300)
    plt.pcolor(np.arange(len(spectrogram))*frame_len/1000, freq_scale[:len(spectrogram[0])], 20*np.log10(spectrogram.T), shading='auto', cmap='gray_r')
    plt.savefig('spectrogram.png', dpi=300)
    plt.close()
    return spectrogram, freq_scale, frame_len

def test_spectrum_generate(bandwidth='broad', test_wave_file='./assets/FDRW0-SA2.wav'):
    spectrogram, _, frame_len = test_spectrogram_generate(test_wave_file)
    ms_output, freq = Model.compute_spectrum(spectrogram, params=[bandwidth, 1000//frame_len, 'log'])
    spectrum = ms_output[bandwidth]
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(freq, spectrum/max(spectrum))
    plt.xlim(0, 16)
    plt.ylim(0, 1.05)
    plt.xticks([0, 4, 8, 12, 16])
    plt.yticks([0, 0.5, 1])
    plt.savefig(f'spectrum-{bandwidth}.png', dpi=300)
    plt.close()
    return freq[np.argmax(spectrum)]

def test_rate_of_syllable_generate(rate_type='sr', test_textgrid_file='./assets/FDRW0-SA2.TextGrid', sequence_flag=False):
    if rate_type not in ['sr', 'ar', 'mode']: return

    silence_syllable = ['', 'sp', 'sil']
    textgrid_data = Tool.read_textgrid(test_textgrid_file)
    syllable_tier = textgrid_data['tiers']['syllables']
    syllables = [abs(item[1]-item[0]) for item in syllable_tier if item[2].lower() not in silence_syllable]
    if sequence_flag:
        return [item for item in syllable_tier if item[2].lower() not in silence_syllable]

    if rate_type=='sr':
        rate_of_syllable = len(syllables)/(textgrid_data['xmax']-textgrid_data['xmin'])
    elif rate_type=='ar':
        rate_of_syllable = 1/np.mean(syllables)
    elif rate_type=='mode':
        x_point, y_fit = np.arange(0, 1600)/100, gaussian_kde([1/it for it in syllables], bw_method="scott")
        rate_of_syllable = x_point[np.argmax(np.array(y_fit(x_point)))]
    print(f'The {rate_type} of syllables is: {rate_of_syllable:.2f}')
    return rate_of_syllable

def demo_of_modeling_relationship_between_syllable_rate_and_modulation_spectrum(data_path='', data_list_file=''):
    assert data_path, 'input the data dir containing the wave and TextGrid files'
    if data_list_file:
        data_list = [it.strip().split('.')[0] for it in open(data_list_file).readlines() if it.strip()]
    else:
        data_list = list(set([item.split('.')[0] for item in os.listdir(data_path)]))
    peak_freq_list, syll_rate_list = [], []
    for adata in data_list:
        wave_file, textgrid_file = f'{data_path}/{adata}.wav', f'{data_path}/{adata}.TextGrid'
        if os.path.exists(wave_file) and os.path.exists(textgrid_file):
            peak_freq = test_spectrum_generate(bandwidth='broad', test_wave_file=wave_file)
            syll_rate = test_rate_of_syllable_generate(rate_type='ar', test_textgrid_file=textgrid_file)
            peak_freq_list.append(peak_freq)
            syll_rate_list.append(syll_rate)
    r, p = pearsonr(peak_freq_list, syll_rate_list)
    print(f'The pearson correlation: r = {r:.02f}, p = {p:.02f}')

def demo_of_modeling_relationship_between_syllable_onset_and_broadband_envelope(data_path='', data_list_file=''):
    assert data_path, 'input the data dir containing the wave and TextGrid files'
    if data_list_file:
        data_list = [it.strip().split('.')[0] for it in open(data_list_file).readlines() if it.strip()]
    else:
        data_list = list(set([item.split('.')[0] for item in os.listdir(data_path)]))
    broadband_envelope, syllable_onset_sequence = np.array([]), np.array([])
    for adata in data_list:
        wave_file, textgrid_file = f'{data_path}/{adata}.wav', f'{data_path}/{adata}.TextGrid'
        if os.path.exists(wave_file) and os.path.exists(textgrid_file):
            spectrogram, _, frame_len = test_spectrogram_generate(test_wave_file=wave_file)
            broadband_envelope = np.append(broadband_envelope, np.mean(spectrogram, axis=1))
            syllables = test_rate_of_syllable_generate(sequence_flag=True, test_textgrid_file=textgrid_file)
            a_syllable_onset = np.zeros(len(spectrogram))
            for a_syllable in syllables:
                a_syllable_onset[int(a_syllable[0]*1000/frame_len)] = 1
            syllable_onset_sequence = np.append(syllable_onset_sequence, a_syllable_onset)

    order, regularization = 200, 0.0001
    syllable_onset_sequence, broadband_envelope = np.append(syllable_onset_sequence, np.zeros(order//2)), np.append(np.zeros(order//2), broadband_envelope)
    trf_h, trf_score = TRF.estimate(np.array([syllable_onset_sequence]), np.array([broadband_envelope/max(broadband_envelope)]), order, regularization)
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(trf_h[:, 0, 0], label=f'score: {trf_score[0]}')
    plt.xlim(0, 200)
    plt.xticks([0, 50, 100, 150, 200], [-0.5, -0.25, 0, 0.25, 0.5])
    plt.savefig('trf.png')
    plt.close()

if __name__ == '__main__':
    # _ = test_spectrogram_generate()
    # peak_freq = test_spectrum_generate(bandwidth='broad')
    # peak_freq = test_spectrum_generate(bandwidth='narrow')

    # syll_rate = test_rate_of_syllable_generate(rate_type='sr')
    # syll_rate = test_rate_of_syllable_generate(rate_type='ar')
    # syll_rate = test_rate_of_syllable_generate(rate_type='mode')

    # demo_of_modeling_relationship_between_syllable_rate_and_modulation_spectrum(data_path='./assets')
    # demo_of_modeling_relationship_between_syllable_onset_and_broadband_envelope(data_path='./assets')
    pass
