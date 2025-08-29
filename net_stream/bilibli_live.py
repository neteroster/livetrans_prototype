import asyncio
import requests
import numpy as np
from typing import Optional, Dict, Any


class BilibiliLive:
    """Bilibili live stream helper with quality selection.

    By default, it selects the lowest available quality to reduce bandwidth and
    latency. You can override by passing qn or quality.
    """
    def __init__(self, room_id: int, *, platform: str = "web", qn: Optional[int] = None, quality: Optional[int] = None):
        self.room_id = room_id
        self.platform = platform
        self.force_qn = qn
        self.force_quality = quality
        def default_read_audio():
            raise RuntimeError("You should call spin_ffmpeg first.")

        self.default_read_audio = default_read_audio
        self.read_audio = default_read_audio

        self.process = None
        self.reader_task = None

        self.ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
        self.stream_play_url = "https://api.live.bilibili.com/room/v1/Room/playUrl"

    def _headers(self) -> Dict[str, str]:
        return {"User-Agent": self.ua}

    def _request(self, params: Dict[str, Any]):
        resp = requests.get(self.stream_play_url, params=params, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def _choose_lowest_params(self) -> Dict[str, Any]:
        base = {"cid": self.room_id, "platform": self.platform}
        try:
            data = self._request(base).get("data", {})
        except Exception:
            return base
        qlist = []
        accept_q = data.get("accept_quality") or []
        try:
            qlist = [int(x) for x in accept_q]
        except Exception:
            qlist = []
        if not qlist:
            desc = data.get("quality_description") or []
            try:
                qlist = [int(item.get("qn")) for item in desc if item.get("qn") is not None]
            except Exception:
                qlist = []
        if not qlist:
            return base
        lowest = min(qlist)
        if lowest >= 80:
            return {**base, "qn": lowest}
        else:
            return {**base, "quality": lowest}

    def get_stream_url(self):
        params: Dict[str, Any] = {"cid": self.room_id, "platform": self.platform}
        if self.force_qn is not None:
            params["qn"] = self.force_qn
        elif self.force_quality is not None:
            params["quality"] = self.force_quality
        else:
            params = self._choose_lowest_params()

        response = self._request(params)
        try:
            return response["data"]["durl"][0]["url"]
        except Exception:
            fallback = self._request({"cid": self.room_id, "platform": self.platform})
            return fallback["data"]["durl"][0]["url"]

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

