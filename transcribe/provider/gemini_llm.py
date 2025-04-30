from google import genai
from google.genai import types as genai_types
from utils import np_to_wav

class GeminiBlockTranscriber:
    def __init__(self, api_key):
        self.genai_client = genai.Client(
            api_key=api_key
        )

    @staticmethod
    def build_prompt(
        ctx=None,
        language=None
    ):
        base_prompt = "Transcribe the given audio into text."

        additional_info = "<Additional Info>\n\n"

        if ctx:
            additional_info += f"**Transcription Context** (For Reference Only): {ctx}\n"

        if language:
            additional_info += f"**Target Language**: {language}\n"

        additional_info += "\n</Additional Info>"

        transribe_prefix = "Transcript: "

        if ctx or language:
            return f"{base_prompt}\n\n{additional_info}" + "\n\n" + transribe_prefix
        
        return base_prompt + "\n\n" + transribe_prefix

    async def transcribe(
            self,
            audio, 
            model,
            ctx=None,
            language=None,
            temperature=0.0,
        ):

        prompt = self.build_prompt(ctx=ctx, language=language)

        response = await self.genai_client.aio.models.generate_content(
            model=model,
            contents=[
                prompt,
                genai_types.Part.from_bytes(
                    data=np_to_wav(audio, 16000),
                    mime_type="audio/wav"
                )
            ],
            config=genai_types.GenerateContentConfig(temperature=temperature)
        )

        return response.text
