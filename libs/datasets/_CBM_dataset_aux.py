import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from python_speech_features import mfcc, logfbank, delta
# import librosa
from scipy import signal
from scipy.io import loadmat
from torchvision.transforms import transforms


CBM_DATASET_ATTR = ['material', 'subclass', 'class', 'instance', # Class part
                    'macroImage', 'frictionForce', 'displayImage', 'temperatureData', 'metalData',
                    'reflectance', 'accelDFT', 'accelDFTSteelTooltip', 'sound', 'soundSteelTooltip', 'accelTap',
                    'accelTapSteelTooltip', 'soundTap', 'soundTapSteelTooltip', 'pressure', 'pressureFolding',
                    'foldingAngle', 'normalForce', 'frictionForce', 'materialID', 'illuminatedImage', 'labelFineP',
                    'labelFinePX', 'processing', 'extra', 'magneticData', 'pressureRaw', 'pressureFoldingRaw',
                    'foldingAngleRaw', 'ambientTemperature'
                    ]

"""
C3_S1_M4 Lack of accelTap
"""

NOT_RECORDED_YET = ['C2_S1_M5', 'C2_S1_M6',
                    'C3_S1_M13', 'C3_S1_M14', 'C3_S1_M15', 'C3_S1_M16', 'C3_S1_M17', 'C3_S1_M18', 'C3_S1_M20', 'C3_S1_M22', 'C3_S1_M23', 'C3_S1_M24', 'C3_S1_M25'
                    'C3_S2_M1', 'C3_S2_M3', 'C3_S2_M5',
                    'C4_S1_M4', 'C4_S1_M5', 'C4_S1_M6',
                    'C4_S3_M2',
                    'C5_S1_M2', 'C5_S1_M3', 'C5_S1_M4', 'C5_S1_M5', 'C5_S1_M6', 'C5_S1_M7',
                    'C5_S2_M1',
                    'C6_S1_M4', # Lack of Image.
                    'C6_S1_M5',
                    'C6_S2_M4',# Lack of Image.
                    'C6_S3_M8', 'C6_S3_M9', 'C6_S3_M10', 'C6_S3_M11', 'C6_S3_M12',
                    'C6_S4_M2',
                    'C6_S5_M1', 'C6_S5_M2', 'C6_S5_M3',
                    'C6_S6_M1', 'C6_S6_M2',
                    'C7_S2_M5', 'C7_S2_M6', 'C7_S2_M7', 'C7_S2_M8',
                    'C7_S3_M4', 'C7_S3_M5', 'C7_S3_M6',
                    'C7_S5_M1',
                    'C8_S3_M2', # Lack of Image and sound
                    'C8_S4_M2']

ONLY_ONE_IN_SUBCLASS = []


def cleanup_data(data_paths):
    result = []
    for data_path in data_paths:
        dir_name, file_name = os.path.split(data_path)
        material_name = '_'.join(dir_name.split("/")[-3:])
        if material_name in NOT_RECORDED_YET:
            continue
        else:
            if "tmp" in file_name:
                pass
            else:
                result.append(data_path)

    return result


def check_mat_file(mat_file_path):
    try:
        _ = loadmat(mat_file_path)
        return True
    except TypeError:
        print(mat_file_path + " is not a valid mat file path")
        return False


def modality_preprocessor(name, modality, cfg, spectrogram=False, mel_spectrogram=False, raw_data=False, fig_name=None):
    x = modality.copy()
    if cfg.MODALITY.TS_NOISE_SNR <= 0:
        pass
    else:
        if "Image" in name:
            pass
        else:
            x = _add_noise(x, cfg.MODALITY.TS_NOISE_SNR)
            print(x.shape)
    assert ~(raw_data | spectrogram)

    if raw_data:
        if "Image" in name:
            return _image_processor(x, cfg)
        else:
            # print(torch.FloatTensor(x.T).shape)
            return torch.FloatTensor(x.T)

    if mel_spectrogram:
        if "Image" in name:
            x = Image.fromarray(x)
        else:
            if "sound" in name:
                sample_rate = cfg.MODALITY.SOUND.SAMPLERATE
                hop_len = 512
            elif "accel" in name:
                sample_rate = cfg.MODALITY.ACCEL.SAMPLERATE
                hop_len = 35
            elif "Force" in name:
                sample_rate = cfg.MODALITY.FORCE.SAMPLERATE
                hop_len = 35
            else:
                raise NotImplementedError
            melspec = librosa.feature.melspectrogram(x.squeeze(), sample_rate, n_fft=1024, hop_length=hop_len, n_mels=60)
            logmelspec = librosa.amplitude_to_db(melspec)
            logmelspec_delta = librosa.feature.delta(logmelspec)

            logmelspec = np.expand_dims(logmelspec, axis=0)
            logmelspec_delta = np.expand_dims(logmelspec_delta, axis=0)
            logmelspec_plus_delta = np.concatenate([logmelspec, logmelspec_delta], axis=0)

            return torch.FloatTensor(logmelspec_plus_delta)

    if spectrogram:

        if "Image" in name:
            x = Image.fromarray(x)
        else:
            if "sound" in name:
                sample_rate = cfg.MODALITY.SOUND.SAMPLERATE
            elif "accel" in name:
                sample_rate = cfg.MODALITY.ACCEL.SAMPLERATE
            elif "Force" in name:
                sample_rate = cfg.MODALITY.FORCE.SAMPLERATE
            else:
                raise NotImplementedError

            x = x.flatten()
            fmin = 0  # Hz
            fmax = 2000  # Hz
            # b, a = signal.butter(4, 2000, 'low', fs=sample_rate)
            # x = signal.filtfilt(b, a, x)
            f, t, Sxx = signal.spectrogram(x, sample_rate)
            # print(f.shape, t.shape, Sxx.shape)
            freq_slice = np.where((f >= fmin) & (f <= fmax))
            # print(freq_slice.shape)
            f = f[freq_slice]
            Sxx = Sxx[freq_slice, :][0]
            # print(f.shape, t.shape, Sxx.shape)
            plt.pcolormesh(t, f, Sxx, shading='gouraud')
            plt.show()
            plt.savefig(fig_name + "_{:}.png".format(name))


            plt.close()  # Avoid ploting all infos in one image.
            x = Image.open(fig_name + "_{:}.png".format(name))
            x = x.convert("RGB")

            # print(fig_name + "_{:}.png".format(name))

            # if os.path.exists(fig_name + "_{:}.png".format(name)):
            #     pass
            # else:
            #     starts, spec = create_spectrogram(x, 256, 85)
            #     plot_spectrogram(spec, fig_name + "_{:}.png".format(name))

            # starts, spec = create_spectrogram(x, 256, 85)
            # plot_spectrogram(spec, fig_name=fig_name + "_{:}.png".format(name))
            # x = Image.open(fig_name + "_{:}.png".format(name))
            # x = x.convert("RGB")

            # x = Image.open(fig_name + "_{:}.png".format(name))
            # x = x.convert("RGB")

        return transforms.Compose([transforms.Resize((cfg.MODALITY.IMAGE.SIZE, cfg.MODALITY.IMAGE.SIZE)),
                                   transforms.ToTensor()])(x)
    else:
        if "Image" in name:
            return _image_processor(x, cfg)
        elif "sound" in name:
            return _sound_processor(x, cfg)
        elif "accel" in name:
            return _accel_processor(x, cfg)
        elif "Force" in name:
            return _force_processor(x, cfg)
        else:
            raise NotImplementedError


def modality_visualize(name, modality):
    x = modality.copy()
    if "Image" in name:
        _image_visualize(x)
    elif "sound" in name:
        _sound_visualize(x)
    elif "accel" in name:
        _accel_visualize(x)
    elif "Force" in name:
        _force_visualize(x)
    else:
        raise NotImplementedError


def _image_processor(x, cfg):
    x = Image.fromarray(x)
    return transforms.Compose([transforms.Resize((cfg.MODALITY.IMAGE.SIZE, cfg.MODALITY.IMAGE.SIZE)),
                               transforms.ToTensor()])(x)


def _image_visualize(x):
    img = Image.fromarray(x)
    img.show()


def _sound_processor(x, cfg):
    if cfg.MODALITY.TS_METHOD == "PSD":
        _, ts_result = signal.periodogram(x, fs=cfg.MODALITY.SOUND.SAMPLERATE, nfft=cfg.MODALITY.SOUND.PSD.NFFT, axis=0)
        return torch.FloatTensor(ts_result)
    elif cfg.MODALITY.TS_METHOD == "MFCCDelta":
        ts_result = mfcc(x, cfg.MODALITY.SOUND.SAMPLERATE,
                         numcep=cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                         nfft=cfg.MODALITY.SOUND.XVECTOR.NFFT)
        delta_1 = delta(ts_result, 1)
        delta_2 = delta(ts_result, 2)

        result = np.concatenate([ts_result, delta_1, delta_2], 1)
        # print(result.shape)
        return torch.FloatTensor(result.T)
    elif cfg.MODALITY.TS_METHOD == "xvector":
        ts_result = mfcc(x, cfg.MODALITY.SOUND.SAMPLERATE,
                         numcep=cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                         nfft=cfg.MODALITY.SOUND.XVECTOR.NFFT,
                         )
        return torch.FloatTensor(ts_result.T)
    elif cfg.MODALITY.TS_METHOD == "fbank":
        ts_result = logfbank(x, cfg.MODALITY.SOUND.SAMPLERATE,
                             nfilt=cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                             nfft=cfg.MODALITY.SOUND.XVECTOR.NFFT)
        return torch.FloatTensor(ts_result.T)
    else:
        raise NotImplementedError


def _sound_visualize(x):
    pass


def _accel_processor(x, cfg):
    if cfg.MODALITY.TS_METHOD == "PSD":
        _, ts_result = signal.periodogram(x, fs=cfg.MODALITY.ACCEL.SAMPLERATE, nfft=cfg.MODALITY.ACCEL.PSD.NFFT, axis=0)
        return torch.FloatTensor(ts_result)
    elif cfg.MODALITY.TS_METHOD == "MFCCDelta":
        ts_result = mfcc(x, cfg.MODALITY.ACCEL.SAMPLERATE,
                         numcep=cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                         nfft=cfg.MODALITY.ACCEL.XVECTOR.NFFT)
        delta_1 = delta(ts_result, 1)
        delta_2 = delta(ts_result, 2)

        result = np.concatenate([ts_result, delta_1, delta_2], 1)
        # print(result.shape)
        return torch.FloatTensor(result.T)

    elif cfg.MODALITY.TS_METHOD == "xvector":
        ts_result = mfcc(x, cfg.MODALITY.ACCEL.SAMPLERATE,
                         numcep=cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                         nfft=cfg.MODALITY.ACCEL.XVECTOR.NFFT,)
        return torch.FloatTensor(ts_result.T)
    elif cfg.MODALITY.TS_METHOD == "fbank":
        ts_result = logfbank(x, cfg.MODALITY.ACCEL.SAMPLERATE,
                             nfilt=cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                             nfft=cfg.MODALITY.ACCEL.XVECTOR.NFFT)
        return torch.FloatTensor(ts_result.T)
    else:
        raise NotImplementedError


def _accel_visualize(x):
    pass


def _force_processor(x, cfg):
    if cfg.MODALITY.TS_METHOD == "PSD":
        _, ts_result = signal.periodogram(x, fs=cfg.MODALITY.FORCE.SAMPLERATE, nfft=cfg.MODALITY.FORCE.PSD.NFFT, axis=0)
        return torch.FloatTensor(ts_result)
    elif cfg.MODALITY.TS_METHOD == "MFCCDelta":
        ts_result = mfcc(x, cfg.MODALITY.FORCE.SAMPLERATE,
                         numcep=cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                         nfft=cfg.MODALITY.FORCE.XVECTOR.NFFT)
        delta_1 = delta(ts_result, 1)
        delta_2 = delta(ts_result, 2)

        result = np.concatenate([ts_result, delta_1, delta_2], 1)
        # print(result.shape)
        return torch.FloatTensor(result.T)
    elif cfg.MODALITY.TS_METHOD == "xvector":
        ts_result = mfcc(x, cfg.MODALITY.FORCE.SAMPLERATE,
                         numcep=cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                         nfft=cfg.MODALITY.FORCE.XVECTOR.NFFT,)
        return torch.FloatTensor(ts_result.T)
    elif cfg.MODALITY.TS_METHOD == "fbank":
        ts_result = logfbank(x, cfg.MODALITY.FORCE.SAMPLERATE,
                             nfilt=cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                             nfft=cfg.MODALITY.FORCE.XVECTOR.NFFT)
        return torch.FloatTensor(ts_result.T)
    else:
        raise NotImplementedError


def _force_visualize(x):
    pass


def _add_noise(signal, SNR=5):
    noise = np.random.randn(*signal.shape)
    noise = noise - np.mean(noise)
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
    noise_variance = signal_power / np.power(10, (SNR / 10))

    noise = (np.sqrt(noise_variance)/np.std(noise))*noise

    return noise + signal


def get_xn(Xs,n):
    '''
    calculate the Fourier coefficient X_n of
    Discrete Fourier Transform (DFT)
    '''
    L  = len(Xs)
    ks = np.arange(0,L,1)
    xn = np.sum(Xs*np.exp((1j*2*np.pi*ks*n)/L))/L
    return(xn)

def get_xns(ts):
    '''
    Compute Fourier coefficients only up to the Nyquest Limit Xn, n=1,...,L/2
    and multiply the absolute value of the Fourier coefficients by 2,
    to account for the symetry of the Fourier coefficients above the Nyquest Limit.
    '''
    mag = []
    L = len(ts)
    for n in range(int(L/2)): # Nyquest Limit
        mag.append(np.abs(get_xn(ts,n))*2)
    return(mag)

def get_Hz_scale_vec(ks, sample_rate, Npoints):
    freq_Hz = ks*sample_rate/Npoints
    freq_Hz  = [int(i) for i in freq_Hz ]
    return(freq_Hz )


def create_spectrogram(ts, NFFT, noverlap = None):
    '''
          ts: original time series
        NFFT: The number of data points used in each block for the DFT.
          Fs: the number of points sampled per second, so called sample_rate
    noverlap: The number of points of overlap between blocks. The default value is 128.
    '''
    if noverlap is None:
        noverlap = NFFT/2
    noverlap = int(noverlap)
    starts  = np.arange(0,len(ts), NFFT-noverlap,dtype=int)
    # remove any window with less than NFFT sample size
    starts  = starts[starts + NFFT < len(ts)]
    xns = []
    for start in starts:
        # short term discrete fourier transform
        ts_window = get_xns(ts[start:start + NFFT])
        xns.append(ts_window)
    specX = np.array(xns).T
    # rescale the absolute value of the spectrogram as rescaling is standard
    spec = 10*np.log10(specX)
    assert spec.shape[1] == len(starts)
    return(starts,spec)


def plot_spectrogram(spec, stop_hz=None, fig_name=None):
    # total_ts_sec = len(ts)/sample_rate
    plt.figure(figsize=(3, 3))
    if stop_hz is None:
        stop_hz = spec.shape[0]
    plt.imshow(spec[:stop_hz, :], origin='lower')

    ## create ylim
    # Nyticks = 10
    # ks      = np.linspace(0,spec.shape[0],Nyticks)
    # ksHz    = get_Hz_scale_vec(ks,sample_rate,len(ts))
    # plt.yticks(ks,ksHz)
    # # plt.ylabel("Frequency (Hz)")
    #
    # ## create xlim
    # Nxticks = 10
    # ts_spec = np.linspace(0,spec.shape[1],Nxticks)
    # ts_spec_sec  = ["{:4.2f}".format(i) for i in np.linspace(0, total_ts_sec*starts[-1]/len(ts), Nxticks)]
    # plt.xticks(ts_spec,ts_spec_sec)
    # plt.xlabel("Time (sec)")

    # plt.title("Spectrogram L={} Spectrogram.shape={}".format(L,spec.shape))
    # plt.colorbar(None, use_gridspec=True)
    plt.savefig(fig_name)
    plt.close()
    # return(plt_spec)