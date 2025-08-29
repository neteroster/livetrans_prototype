import types
import pytest
import os, sys

# Ensure project root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import net_stream.bilibli_live as bl


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_default_selects_lowest_quality(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None):
        calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        params = params or {}
        # If asking for a concrete quality, return a URL indicating that quality
        if "quality" in params:
            q = params["quality"]
            return DummyResponse({
                "code": 0,
                "message": "0",
                "ttl": 1,
                "data": {
                    "durl": [{"url": f"http://example.com/stream_q{q}.flv"}]
                }
            })
        # Initial probe without quality: return accept_quality list (strings)
        return DummyResponse({
            "code": 0,
            "message": "0",
            "ttl": 1,
            "data": {
                "accept_quality": ["4", "3", "2"],
                "durl": [{"url": "http://example.com/base.flv"}],
            }
        })

    monkeypatch.setattr(bl.requests, "get", fake_get)

    live = bl.BilibiliLive(room_id=14073662)
    url = live.get_stream_url()

    # It should make two requests: probe then fetch with the lowest quality=2
    assert len(calls) == 2
    assert calls[0]["params"].get("cid") == 14073662
    assert calls[0]["params"].get("platform") == "web"
    assert "quality" not in calls[0]["params"] and "qn" not in calls[0]["params"]

    assert calls[1]["params"].get("quality") == 2
    assert url.endswith("stream_q2.flv")


def test_explicit_quality_is_respected(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None):
        calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        params = params or {}
        if "quality" in params:
            q = params["quality"]
            return DummyResponse({
                "code": 0,
                "message": "0",
                "ttl": 1,
                "data": {
                    "durl": [{"url": f"http://example.com/stream_q{q}.flv"}]
                }
            })
        # Should not reach here in this test
        raise AssertionError("Unexpected probe without explicit quality")

    monkeypatch.setattr(bl.requests, "get", fake_get)

    live = bl.BilibiliLive(room_id=1, quality=3)
    url = live.get_stream_url()

    assert len(calls) == 1
    assert calls[0]["params"].get("quality") == 3
    assert url.endswith("stream_q3.flv")


def test_fallback_uses_qn_80_when_accept_quality_missing(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None):
        calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        params = params or {}
        if "qn" in params:
            qn = params["qn"]
            return DummyResponse({
                "code": 0,
                "message": "0",
                "ttl": 1,
                "data": {
                    "durl": [{"url": f"http://example.com/stream_qn{qn}.flv"}]
                }
            })
        # First probe returns no accept_quality
        return DummyResponse({
            "code": 0,
            "message": "0",
            "ttl": 1,
            "data": {
                # intentionally missing 'accept_quality'
                "durl": [{"url": "http://example.com/base.flv"}],
            }
        })

    monkeypatch.setattr(bl.requests, "get", fake_get)

    live = bl.BilibiliLive(room_id=2)
    url = live.get_stream_url()

    assert len(calls) == 2
    # Second call should include qn=80 as fallback
    assert calls[1]["params"].get("qn") == 80
    assert url.endswith("stream_qn80.flv")
