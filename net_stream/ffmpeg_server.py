import asyncio
import requests
import numpy as np
from queue import SimpleQueue

class FFmpegServer:
    def __init__(self, bind_ip: str, bind_port: int):
        self.ip = bind_ip
        self.port = bind_port
        def default_read_audio():
            raise RuntimeError("You should call spin_ffmpeg first.")
        
        self.default_read_audio = default_read_audio
        self.read_audio = default_read_audio

        self.process = None
        self.reader_task = None

    async def spin_ffmpeg(self, ffmpeg_path: str, sampling_rate=16000, samples_per_chunk=512):
        command = [
            ffmpeg_path,
            "-i", f"srt://{self.ip}:{self.port}?mode=listener",
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

