from faster_whisper import WhisperModel
import numpy as np
from typing import List, Tuple, Callable


class FasterWhisperBlockTranscriber:
    def __init__(self, whisper_model_config):
        self.model = WhisperModel(**whisper_model_config)

    def transcribe(
            self,
            audio, 
            prompt,
            segment_max_no_speech_prob,
            segments_merge_fn,
            language=None
        ):
        # segment_merge_fn: List[str] -> T and this function return T

        transribe_result, transcription_info = self.model.transcribe(audio, initial_prompt=prompt, language=language)

        segments = [segment.text for segment in transribe_result if segment.no_speech_prob < segment_max_no_speech_prob]

        return segments_merge_fn(segments)


##########################################
################## TODO ##################
##########################################

def longest_common_prefix(x: List, y: List, fn_same: Callable) -> List:
    raise NotImplementedError()

class WhisperTranscribeStream:
    def __init__(self, whisper_model_config, max_prompt_cache_len=200):
        self.model = WhisperModel(**whisper_model_config)
        self.max_prompt_cache_len = max_prompt_cache_len

        self.reset_states()

    def reset_states(self):
        """
        |---------------------|-----------------------|
        ^buffer begin         ^confirmed end          ^buffer end
        """

        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        
        self.confirmed_end: int = 0 # 0 is the begin of audio buffer

        self.text_buffer: List[Tuple[str, float, float]] = [] # list of (text: str, begin: float, end: float)
        self.text_confirmed_end: int = 0

        self.prompt_cache: str = "" # cache the text before audio buffer to provide to the model

        # We need to save the segment points to trim the audio buffer when necessary.
        # t = 0 -> the begin of the audio buffer
        self.segment_points: List[int] = []

    def submit_audio(self, audio: np.ndarray):
        self.audio_buffer = np.concatenate([self.audio_buffer, audio])

    def do_whisper_iter(self):
        # trim the prompt cache
        if len(self.prompt_cache) > self.max_prompt_cache_len:
            self.prompt_cache = self.prompt_cache[-self.max_prompt_cache_len:]

        # immediately transcribe the whole audio buffer
        transribe_result, transcribe_info = self.model.transcribe(
            self.audio_buffer,
            initial_prompt=self.prompt_cache,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True
        )

        transribe_transformed = []
        for segment in transribe_result:
            if segment.no_speech_prob > 0.9: continue  # TODO: make this threshold configurable

            segment_start_sample = segment.start * 16000
            self.segment_points.append(segment_start_sample)

            for word in segment.words:
                transribe_transformed.append((word.word, word.start, word.end))

