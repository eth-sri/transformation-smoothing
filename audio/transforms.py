import io
import soundfile as sf
import librosa


def round(y, sr):
    bo = io.BytesIO()
    bo.name = "foo.wav"
    sf.write(bo, y, sr, 'PCM_16')
    bo.seek(0)
    y, sr = librosa.load(bo, sr=None)
    return y, sr


def scale_volume(data, sr, f, do_round=True):
    """
    Parameters
    ----------
    data : actually sampled data
    sr : sample
    rate f : scaling factor

    Returns
    -------
    data_out : scaled data
    sr : sample rate
    """
    data_out = 10**(f / 20) * data
    if do_round:
        data_out, sr = round(data_out, sr)
    return data_out, sr


def shift(data, sr, s, do_round=True):
    """
    Parameters
    ----------
    data : actually sampled data
    sr : sample
    s : shift (in fractional half-steps)

    Returns
    -------
    data_out : scaled data
    sr : sample rate
    """
  
    data_out, sr = librosa.effects.pitch_shift(data, sr, s), sr
    if do_round:
        data_out, sr = round(data_out, sr)
    return data_out, sr
