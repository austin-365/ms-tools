import re
import json
import logging
import numpy as np
import multiprocessing

from scipy.signal import lfilter

class Model():
    def __init__(self) -> None:
        pass

    @classmethod
    def compute_spectrogram(cls, input_chunk, params=[5, 8, 0]):
        r"""
            Compute the auditory spectrogram of sound signal.
            Warnning: Zero-padding will greatly increase the computation time.

            Args:
                input_chunk [np.array(float)]: the acoustic input.
                params [frame_len, time_const, shift]:
                    frame_len [int; 4, 8, 16, 2^Z]: frame length.
                    time_const [int; 4, 16, 64, etc]: time const.
                    shift [int; 0, 1, etc]: shifted by octave.
            Built-in:
                COCHBA [cochead, cochfil]:
                    M [int]: highest frequency channel.
                    cochead [f + CF*i]: 1-by-M filter length vector.
                        f [real(cochead)]: M, filter order.
                        CF [imag(cochead)]: characteristic frequency.
                    cochfil [B + A*i]: [M]-channel filterbank matrix.
                        B [real(cochfil)]: MA (Moving Average) coefficients.
                        A [imag(cochfil)]: AR (AutoRegressive) coefficients.
            Returns:
                spectrogram [np.array(_, _)]: the auditory spectrogram. (N-by-(M-1))
                freq_scale [np.array(_)]: the frequency scale. (band 180 - 7246 Hz)

            Examples:
            >>> spectrogram, freq_scale = compute_spectrogram(input_chunk, params=[frame_len, time_const, shift])
        """
        cochba=np.array(json.load(open('/home/zhang/acoustic_theory/examples/spectrum_cochba.json')))

        freq_scale = 440*2**(np.arange(-31, 98)/24)
        freq_scale = np.round(freq_scale/10)*10

        column_num, input_chunk_len = cochba.shape[1], len(input_chunk)
        shift = pow(2, 4+params[2])
        alph = np.exp(-1/(params[1]*shift))
        frame_length = round(params[0]*shift)
        N = int(np.ceil(input_chunk_len/frame_length))
        logging.debug('Spectrogram - Params init Done.')

        input_chunk[-1] = 0
        spectrogram = np.zeros([N, column_num-1])
        for column_index in range(column_num-1, -1, -1):
            chunk_p = cochba[0, column_index][0]
            chunk_B = [item[0] for item in cochba[1:int(chunk_p)+2, column_index]]
            chunk_A = [item[1] for item in cochba[1:int(chunk_p)+2, column_index]]
            filtered_chunk = lfilter(chunk_B, chunk_A, input_chunk)

            if column_index==column_num-1:
                filtered_chunk_previous = filtered_chunk
            else:
                chunk_difference = np.maximum(filtered_chunk-filtered_chunk_previous, 0)
                filtered_chunk_previous = filtered_chunk
                filtered_chunk_difference = lfilter(np.array([1]), np.array([1, -alph]), chunk_difference)
                try:
                    spectrogram[:, column_index] = np.append(filtered_chunk_difference[frame_length-1:frame_length*N:frame_length], np.array(filtered_chunk_difference[-1]))
                except:
                    spectrogram[:, column_index] = filtered_chunk_difference[frame_length-1:frame_length*N:frame_length]
        logging.debug('Spectrogram - Computation Done.')
        return spectrogram, freq_scale

    @classmethod
    def compute_spectrum(cls, spectrogram, params=['narrow', 200, 'log']):
        r"""
            Compute the modulation spectrum of auditory spectrogram.

            Args:
                spectrogram [np.array(_, _)]: the output of the compute_spectrogram()
                params : [bandwidth, sample_rate, fscale]
                    bandwidth ['broad', 'narrow', 'all']: default 'narrow'.
                    sample_rate [int]: default 200. Equal to 1/frame_len(in compute_spectrogram).
                    fscale ['log', 'lin', float]: default 'log'. Aim to correct the 1/f trend.
            Returns:
                ms_output [Dict(np.array(_))]: use the key, i.e., 'broad', 'narrow', to get modulation spectrum.
                freq [np.array(_)]: the frequency scale. (0 - sample_rate Hz)

            Examples
            --------
            >>> ms_output, freq = compute_spectrum(spectrogram, params=['broad', 200, 'log'])
        """
        spectrograms = { 'narrow': spectrogram, 'broad': np.mean(spectrogram, 1, keepdims=True) }

        (bandwidth, sample_rate, fscale) = params
        bandwidth = ['broad', 'narrow'] if bandwidth=='all' else [bandwidth]
        fscale = {'log': 0.5, 'lin': 0}[fscale] if fscale in ['log', 'lin'] else float(fscale)

        freq = np.arange(spectrogram.shape[0]/2)
        freq = sample_rate * freq.T / spectrogram.shape[0]
        logging.debug('Spectrum - Params init Done.')

        ms_output = {}
        for bw_item in bandwidth:
            spectrum_temp = np.abs(np.fft.fft(spectrograms[bw_item]**2, axis=0))
            spectrum = np.sqrt(np.mean(spectrum_temp**2, 1))
            spectrum = np.append(spectrum, np.zeros(1)) if spectrum.shape[0]%2 else spectrum
            ms_output[bw_item] = freq**fscale * (spectrum.T.reshape(2,-1)[0])
        logging.debug('Spectrum - Computation Done.')

        return ms_output, freq

class TRF():
    def __init__(self) -> None:
        r"""
            Estimate the temporal responce function (TRF) with stimulus & responce signal.

            Args:
                signals_x [np.array(stimulus_dim, signal_len)]: stimulus signal
                signals_y [np.array(response_dim, signal_len)]: response signal
                order [int]: the order of the system, and the length of the output, i.e., trf_h
                tikf [float]: regularization parameter
            Built-in:
                seg_num [int]: number for cross-validation
            Returns:
                trf_h [np.array(_)]: the trf
                trf_score [float]: the predictive power

            Examples
            --------
            >>> trf = TRF()
            >>> trf.xxx()
        """

    @classmethod
    def estimate(cls, signals_x, signals_y, order, tikf):
        '''
            Estimate the TRF.

            Args:
                see __init__()

            Examples
            --------
            >>> trf_h, trf_score = TRF.estimate(signals_x, signals_y, order, tikf)
        '''
        input_size, output_size = len(signals_x), len(signals_y)
        h = np.zeros((order, input_size, output_size, 10))
        cr_test = np.zeros((output_size, 10))

        for seg_num in range(10):
            temp_h, temp_s = cls().model([signals_x, signals_y, seg_num, order, tikf])
            h[:,:,:,seg_num], cr_test[:,seg_num] = np.real(temp_h), temp_s
        return np.nanmean(h, -1), np.nanmean(cr_test, -1)

    @classmethod
    def estimate_in_parallel(cls, signals_x, signals_y, order, tikf):
        '''
            Estimate the TRF in parallel.

            Args:
                see __init__()

            Examples
            --------
            >>> trf_h, trf_score = TRF.estimate_in_parallel(signals_x, signals_y, order, tikf)
        '''
        pool = multiprocessing.Pool(10)
        all_pairs = pool.map(cls().model, [[signals_x, signals_y, seg_num, order, tikf] for seg_num in range(10)])
        pool.close()
        pool.join()

        h, cr_test = [np.real(item[0]) for item in all_pairs], [item[1] for item in all_pairs]
        return np.mean(np.array(h), 0), np.mean(np.array(cr_test), 0)

    def model(self, params):
        '''
            Args:
                see __init__()
        '''
        signals_x, signals_y, seg_num, order, tikf = params

        testing_range = np.arange(len(signals_x[0])//10)+len(signals_x[0])*seg_num//10
        training_range = np.setdiff1d(np.arange(len(signals_x[0])), testing_range)
        x_test = signals_x[:,testing_range]
        y_test = signals_y[:,testing_range]
        input_signal = signals_x[:,training_range]
        output_signal = signals_y[:,training_range]

        if np.sum(y_test)==0 or np.sum(x_test)==0:
            return np.array([[[np.nan]]]*order), np.nan

        logging.debug(x_test.shape, y_test.shape, input_signal.shape, output_signal.shape)
        dimx = np.size(input_signal, 0)
        lenx = np.size(input_signal, 1)
        chhn = np.size(output_signal, 0)

        Y = output_signal[:,(order-1):lenx]
        X = np.zeros((dimx*order, lenx-order+1))

        for ind1 in range(dimx):
            for ind2 in range(order):
                X[ind1*order+ind2,:] = input_signal[ind1,ind2:(lenx-order+1+ind2)]

        if tikf==np.inf:
            h = np.dot(X, Y.T)
        else:
            Rx = np.dot(X, X.T)
            d, _ = np.linalg.eig(Rx)
            d = d[np.argsort(d)]
            tikf = tikf*d[-1]
            h = np.linalg.solve(Rx+tikf*np.eye(Rx.shape[0]), np.dot(X, Y.T))

        h = np.transpose(h.reshape(dimx, order, chhn), (1, 0, 2))
        h = h[-1::-1,:,:]

        cr_test = np.zeros((np.size(output_signal, 0), 1))
        y_pred = y_test*0
        for chh in range(chhn):
            for ind1 in range(np.size(h, 1)):
                y_pred[chh,:] = y_pred[chh,:] + np.real(lfilter(h[:, ind1, chh], 1, x_test[ind1,:]))
            cr_test[chh,:] = np.corrcoef(y_test[chh,:].T, y_pred[chh,:].T)[0][1]

        return h, cr_test.reshape(np.size(output_signal, 0))

class Tool():
    def __init__(self) -> None:
        pass

    ''' TextGrid Part '''
    def parse_line(self, line, to_round=4):
        line = line.strip()
        if '"' in line:
            key, value = re.findall(r'(.+?) = "(.*)"', line)[0]
            return [ key, value ]
        else:
            key, value = re.findall(r'(.+?) = (.*)', line)[0]
            return [ key, int(value) if 'size' in key else round(float(value), to_round) ]

    @classmethod
    def read_textgrid(cls, filepath):
        '''
        Read the tiers contained in the Praat-formatted TextGrid file.
        Adapted form python package 'textgrid'.
        '''
        temp_data, textgrid_data = [], {}
        with open(filepath, 'r') as source:
            temp_data += [cls().parse_line(source.readline())]
            temp_data += [cls().parse_line(source.readline())]
            source.readline()
            assert temp_data[0][1].startswith('ooTextFile') and temp_data[1][1]=='TextGrid'
            temp_data += [cls().parse_line(source.readline())]
            temp_data += [cls().parse_line(source.readline())]
            source.readline()
            temp_data += [cls().parse_line(source.readline())]
            source.readline()
            temp_data += [['tiers', {}]]
            textgrid_data = {item[0]: item[1] for item in temp_data}

            for _ in range(textgrid_data['size']):
                temp_data, tier_data = [], {}
                source.readline()
                temp_data += [cls().parse_line(source.readline())]
                temp_data += [cls().parse_line(source.readline())]
                temp_data += [cls().parse_line(source.readline())]
                temp_data += [cls().parse_line(source.readline())]
                temp_data += [cls().parse_line(source.readline())]
                tier_data = {item[0]: item[1] for item in temp_data}

                for _ in range(tier_data['intervals: size']):
                    temp_data = []
                    source.readline()
                    temp_data += [cls().parse_line(source.readline())]
                    temp_data += [cls().parse_line(source.readline())]
                    temp_data += [cls().parse_line(source.readline())]
                    tier_data['intervals'] = tier_data.get('intervals', []) + [[item[1] for item in temp_data]]
                textgrid_data['tiers'][tier_data['name']] = tier_data['intervals']
        return textgrid_data

    @classmethod
    def write_textgrid(cls, textgrid_data, filepath):
        """
        Write the current state into a Praat-format TextGrid file.
        Adapted form python package 'textgrid'.
        format:
            textgrid_data:
                {
                    'xmin': 0.0,
                    'xmax': 3.78,
                    'tiers': {
                        'words': [[0.0, 0.16, ''], ...],
                        'phones': [[0.0, 0.16, 'sil'], ...]
                    }
                }
        """
        with open(filepath, 'w') as target:
            target.write('File type = "ooTextFile"\n')
            target.write('Object class = "TextGrid"\n\n')
            target.write('xmin = {}\n'.format(textgrid_data['xmin']))
            target.write('xmax = {}\n'.format(textgrid_data['xmax']))
            target.write('tiers? <exists>\n')
            target.write('size = {}\n'.format(len(textgrid_data['tiers'])))
            target.write('item []:\n')
            for (i, name) in enumerate(textgrid_data['tiers'], 1):
                target.write('\titem [{}]:\n'.format(i))
                target.write('\t\tclass = "IntervalTier"\n')
                target.write('\t\tname = "{}"\n'.format(name))
                target.write('\t\txmin = {}\n'.format(textgrid_data['xmin']))
                target.write('\t\txmax = {}\n'.format(textgrid_data['xmax']))
                target.write('\t\tintervals: size = {}\n'.format(len(textgrid_data['tiers'][name])))
                for (j, interval) in enumerate(textgrid_data['tiers'][name], 1):
                    target.write('\t\t\tintervals [{}]:\n'.format(j))
                    target.write('\t\t\t\txmin = {}\n'.format(interval[0]))
                    target.write('\t\t\t\txmax = {}\n'.format(interval[1]))
                    target.write('\t\t\t\ttext = "{}"\n'.format(interval[2]))
