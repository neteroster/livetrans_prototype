from faster_whisper import WhisperModel
from silero_vad import read_audio, save_audio


model = WhisperModel("large-v2", download_root="./whisper_cache/")

audio = read_audio("demo.wav", sampling_rate=16000)

audio = audio[:16000*40].detach().numpy()

result = model.transcribe(audio, initial_prompt="", beam_size=5, word_timestamps=True, condition_on_previous_text=True)

print(list(result[0]))