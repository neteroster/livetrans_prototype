import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from net_stream.bilibli_live import BilibiliLive


class FakeResponse:
    def __init__(self, json_data):
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        return None


def test_selects_lowest_quality_default(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None):
        calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        p = params or {}
        if ("quality" in p) or ("qn" in p):
            # Final URL fetch with chosen quality
            data = {
                "code": 0,
                "message": "0",
                "ttl": 1,
                "data": {"durl": [{"url": "url_low_q2"}]},
            }
        else:
            # Discovery: provide available qualities (small numbers -> `quality`)
            data = {
                "code": 0,
                "message": "0",
                "ttl": 1,
                "data": {
                    "accept_quality": ["4", "3", "2"],
                    "quality_description": [
                        {"qn": 4, "desc": "原画"},
                        {"qn": 3, "desc": "高清"},
                        {"qn": 2, "desc": "流畅"},
                    ],
                    "durl": [{"url": "url_default"}],
                },
            }
        return FakeResponse(data)

    monkeypatch.setattr("net_stream.bilibli_live.requests.get", fake_get)

    live = BilibiliLive(room_id=123456)
    url = live.get_stream_url()

    assert url == "url_low_q2"
    assert len(calls) == 2
    # First call: discovery with platform=web
    assert calls[0]["params"].get("platform") == "web"
    # Second call should include the chosen lowest quality=2
    assert calls[1]["params"].get("platform") == "web"
    assert int(calls[1]["params"].get("quality")) == 2


def test_respects_qn_override(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None):
        calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        p = params or {}
        if "qn" in p:
            data = {
                "code": 0,
                "message": "0",
                "ttl": 1,
                "data": {"durl": [{"url": "url_forced_qn"}]},
            }
        else:
            # Should not be called in this test; provide a safe default if it is
            data = {
                "code": 0,
                "message": "0",
                "ttl": 1,
                "data": {"durl": [{"url": "url_default"}]},
            }
        return FakeResponse(data)

    monkeypatch.setattr("net_stream.bilibli_live.requests.get", fake_get)

    live = BilibiliLive(room_id=654321, qn=150)
    url = live.get_stream_url()

    assert url == "url_forced_qn"
    assert len(calls) == 1
    assert calls[0]["params"].get("platform") == "web"
    assert int(calls[0]["params"].get("qn")) == 150
