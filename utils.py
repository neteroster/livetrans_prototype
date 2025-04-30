import soundfile as sf
import numpy as np
import io

def np_to_wav(audio: np.ndarray, sampling_rate: int):
    is_audio_range_valid = np.all(np.logical_and(-1.0 <= audio, audio <= 1.0))

    if not is_audio_range_valid:
        raise ValueError("Audio range must be in [-1, 1]")
    
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio, sampling_rate, format="WAV")

    return wav_buffer.getvalue()