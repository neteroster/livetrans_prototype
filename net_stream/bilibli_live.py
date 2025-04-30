import asyncio
import requests
import numpy as np
from queue import SimpleQueue

class BilibiliLive:
    def __init__(self, room_id):
        self.room_id = room_id
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

        params = {
            "cid": self.room_id,            
        }

        response = requests.get(self.stream_play_url, params=params, headers=headers)

        return response.json()["data"]["durl"][0]["url"] # TODO: Maybe do durl selection to ensure low bandwidth / low latency

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

