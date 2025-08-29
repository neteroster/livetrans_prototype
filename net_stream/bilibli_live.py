import asyncio
import requests
import numpy as np

class BilibiliLive:
    def __init__(self, room_id, platform: str = "web", quality: int | None = None, qn: int | None = None, prefer_lowest: bool = True):
        self.room_id = room_id
        # Streaming options
        self.platform = platform  # 'web' (http-flv) or 'h5' (hls)
        self.quality = quality    # 2: 流畅, 3: 高清, 4: 原画 (API 'quality' param)
        self.qn = qn              # 80, 150, 400, 10000, 20000, 30000 (API 'qn' param)
        self.prefer_lowest = prefer_lowest

        def default_read_audio():
            raise RuntimeError("You should call spin_ffmpeg first.")

        self.default_read_audio = default_read_audio
        self.read_audio = default_read_audio

        self.process = None
        self.reader_task = None

        self.ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
        self.stream_play_url = "http://api.live.bilibili.com/room/v1/Room/playUrl"

    def get_stream_url(self):
        headers = {
            "User-Agent": self.ua
        }

        base_params = {
            "cid": self.room_id,
            "platform": self.platform,
        }

        # If explicit quality is provided, request directly.
        if self.qn is not None or self.quality is not None:
            direct_params = base_params.copy()
            if self.qn is not None:
                direct_params["qn"] = self.qn
            if self.quality is not None:
                direct_params["quality"] = self.quality
            response = requests.get(self.stream_play_url, params=direct_params, headers=headers)
            return response.json()["data"]["durl"][0]["url"]

        # Default behavior: choose the lowest available quality
        response = requests.get(self.stream_play_url, params=base_params, headers=headers)
        data = response.json().get("data", {})

        chosen_quality: int | None = None
        chosen_qn: int | None = None
        accept_quality = data.get("accept_quality")
        if isinstance(accept_quality, list) and accept_quality:
            try:
                codes = [int(x) for x in accept_quality]
                m = min(codes)
                # Heuristic: small numbers (<=10) correspond to 'quality' codes like 2/3/4,
                # otherwise they are 'qn' codes like 80/150/400/10000.
                if m <= 10:
                    chosen_quality = m
                else:
                    chosen_qn = m
            except Exception:
                chosen_quality = None
                chosen_qn = None

        # Fallback if accept_quality is not present or parsing failed
        if chosen_quality is None and chosen_qn is None:
            # Default to commonly lowest qn is 80 (流畅)
            chosen_qn = 80

        # Request the URL for the chosen lowest quality
        direct_params = base_params.copy()
        if chosen_quality is not None:
            direct_params["quality"] = chosen_quality
        else:
            direct_params["qn"] = chosen_qn
        response2 = requests.get(self.stream_play_url, params=direct_params, headers=headers)
        return response2.json()["data"]["durl"][0]["url"]

    async def spin_ffmpeg(self, ffmpeg_path: str, sampling_rate=16000, samples_per_chunk=512):
        command = [
            ffmpeg_path,
            "-headers", "User-Agent: " + self.ua,
            "-i", self.get_stream_url(),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sampling_rate),
            "-ac", "1",
            "-f", "s16le",
            "pipe:1"
        ]

        self.audio_buffer = asyncio.Queue()
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        bytes_per_chunk = 2 * samples_per_chunk

        async def reader_worker():
            while True:
                buf = bytearray()

                while len(buf) < bytes_per_chunk:
                    piece = await self.process.stdout.read(bytes_per_chunk - len(buf))
                    if not piece: break
                    buf.extend(piece)

                if not buf: break

                await self.audio_buffer.put(bytes(buf))

        self.reader_task = asyncio.create_task(reader_worker())

        async def read_audio(n_chunk=1):
            chunks = [await self.audio_buffer.get() for _ in range(n_chunk)]
            arrays = [np.frombuffer(c, dtype=np.int16) for c in chunks]
            return np.concatenate(arrays).astype(np.float32) / 32768.0

        self.read_audio = read_audio

    async def stop_ffmpeg(self):
        if self.process is not None:
            self.process.terminate()
            await self.process.wait()
            self.process = None
        if self.reader_task is not None:
            await self.reader_task
            self.reader_task = None

        self.read_audio = self.default_read_audio

