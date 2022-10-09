# python script

## Modulation spectrogram and spectrum

```python
from model import Model
frame_len, time_const, shift = 5, 8, 0
input_chunk = ...
spectrogram, freq_scale = Model.compute_spectrogram(input_chunk, params=[frame_len, time_const, shift])
plt.pcolor(np.arange(len(spectrogram))*frame_len/1000, freq_scale[:len(spectrogram[0)], 20*np.log10(spectrogram.T), shading='auto', cmap='gray_r')
plt.show()

ms_output, freq = Model.compute_spectrum(spectrogram, params=['broad', 200, 'log'])
ms_narrow = ms_output['narrow']
plt.plot(freq, ms_narrow/max(ms_narrow))
plt.show()
```

## Temporal responce function (TRF)

```python
import json
from model import TRF

order = 200
signal_x, signal_y = prepare_xxx()
trf_h, trf_score = TRF.estimate(signal_x, signal_y, order, 0.0001)
for response_dim in range(trf_h.shape[2]):
    plt.plot(trf_h[:,:,response_dim].T, label=response_dim)
    plt.label('scorei is {}'.format(trf_score[response_dim]))
plt.show()
```
