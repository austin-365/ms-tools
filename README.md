# MS-Tools

This repository contains the code for analyzing the modulation spectrum.

## Prepare

1. Download the audio file and its transcription
2. Convert audio file to .wav format with ffmpeg by setting -ac to 1, -ar to 16000
3. Align and obtain the .TextGrid file with .wav & .txt file by [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)

## Modulation spectrogram and spectrum

The modulation spectrogram and spectrum can be calculated by the three function below in the `model-test.py` script.

```python
# modulation spectrogram
_ = test_spectrogram_generate()
# broadband spectrum
peak_freq = test_spectrum_generate(bandwidth='broad')
# narrowband spectrum
peak_freq = test_spectrum_generate(bandwidth='narrow')
```

## Rate of syllable

Three kinds of rate of syllable can be calculated by the three function below in the `model-test.py` script.

```python
# syllable rate
syll_rate = test_rate_of_syllable_generate(rate_type='sr')
# articulation rate
syll_rate = test_rate_of_syllable_generate(rate_type='ar')
# syllable mode
syll_rate = test_rate_of_syllable_generate(rate_type='mode')
```

## Correlation

```python
# correlation
demo_of_modeling_relationship_between_syllable_rate_and_modulation_spectrum(data_path='./assets')
```

## Temporal responce function (TRF)

```python
# TRF
demo_of_modeling_relationship_between_syllable_onset_and_broadband_envelope(data_path='./assets')
```
