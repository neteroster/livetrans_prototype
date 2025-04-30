from openai import AsyncOpenAI
from openai.types.audio.transcription_verbose import TranscriptionVerbose
from utils import np_to_wav

class OpenAIWhisperBlockTranscriber:
    def __init__(self, base_url, api_key):
        self.openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def transcribe(
            self,
            audio, 
            model,
            prompt,
            segment_min_no_speech_prob,
            segments_merge_fn,
            language=None
        ):

        target_params = {
            "file": np_to_wav(audio, 16000),
            "model": model,
            "prompt": prompt,
            "language": language,
            "response_format": "verbose_json"
        }

        target_params_none_wrapped = {
            k: v for k, v in target_params.items() if v is not None
        }

        transribe_result: TranscriptionVerbose = await self.openai_client.audio.transcriptions.create(
            **target_params_none_wrapped
        )

        segments = [segment.text for segment in transribe_result.segments if segment.no_speech_prob < segment_min_no_speech_prob]

        return segments_merge_fn(segments)