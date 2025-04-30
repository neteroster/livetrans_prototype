from openai import AsyncOpenAI

class OpenAICompatibleLLMProvider:
    def __init__(
            self,
            base_url: str,
            api_key: str,
            model: str,
            system_prompt: str=None,
            temperature: float=0.5
        ) -> None:
        self.openai = AsyncOpenAI(base_url=base_url, api_key=api_key)

        self.translate_prompt = """你正在翻译一个在线直播，请将下面文本翻译为中文，仅输出翻译结果：{src_text}"""

        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature

    async def translate(self, src_text: str) -> str:
        prompt = self.translate_prompt.format(src_text=src_text)

        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        result = (await self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )).choices[0].message.content

        return result