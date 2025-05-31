import time
import json
import pytest

from common.observability.loki_logger import log_to_loki


class DummyResponse:
    def __init__(self, status_code=200, text="OK"):
        self.status_code = status_code
        self.text = text


@pytest.fixture(autouse=True)
def fake_post(monkeypatch):
    """
    Monkeypatch requests.post so that we can inspect the args
    instead of actually sending to Loki.
    """
    calls = {}

    def fake_post_func(url, data, headers):
        # Record the arguments for later assertions
        calls['url'] = url
        calls['data'] = data
        calls['headers'] = headers
        # Return a dummy response that mimics requests.Response
        return DummyResponse(status_code=202, text='{"status": "success"}')

    monkeypatch.setattr("requests.post", fake_post_func)
    return calls


def test_log_to_loki_posts_to_correct_url_and_headers(fake_post, capsys):
    """
    Ensure that log_to_loki constructs the correct JSON payload,
    posts it to the expected Loki endpoint, and prints status/text.
    """
    label = "test_app"
    message = "Test message for Loki"

    # Call the function under test
    log_to_loki(label, message)

    # 1) Verify the URL
    assert fake_post["url"] == "http://localhost:3100/loki/api/v1/push"

    # 2) Verify headers
    expected_headers = {"Content-Type": "application/json"}
    assert fake_post["headers"] == expected_headers

    # 3) Verify payload structure
    payload = json.loads(fake_post["data"])
    assert isinstance(payload, dict)
    assert "streams" in payload
    streams = payload["streams"]
    assert isinstance(streams, list) and len(streams) == 1

    stream0 = streams[0]
    assert "stream" in stream0
    assert "values" in stream0

    # The "app" label must match
    assert stream0["stream"]["app"] == label

    # The timestamp in nanoseconds should be a string of digits
    values = stream0["values"]
    assert (
        isinstance(values, list)
        and len(values) == 1
        and isinstance(values[0], list)
        and len(values[0]) == 2
    )
    ts_ns_str, msg = values[0]
    assert msg == message
    assert ts_ns_str.isdigit()

    # Because we returned a DummyResponse with status_code 202,
    # the function prints “Status: 202” etc.
    captured = capsys.readouterr()
    assert "Status: 202" in captured.out
    assert '"status": "success"' in captured.out  # substring of the dummy response text


@pytest.mark.parametrize("status_code, resp_text", [
    (400, "Bad Request"),
    (500, "Internal Server Error"),
])
def test_log_to_loki_prints_error_status(monkeypatch, capsys, status_code, resp_text):
    """
    If the server replies with a non‐2xx status, log_to_loki will still print
    that status code and response text. We verify that behavior here.
    """
    calls = {}

    def fake_post_error(url, data, headers):
        calls["url"] = url
        calls["data"] = data
        calls["headers"] = headers
        return DummyResponse(status_code=status_code, text=resp_text)

    monkeypatch.setattr("requests.post", fake_post_error)

    log_to_loki("myapp", "Something went wrong")

    # It should still attempt to POST to the same endpoint:
    assert calls["url"] == "http://localhost:3100/loki/api/v1/push"
    captured = capsys.readouterr()
    assert f"Status: {status_code}" in captured.out
    assert f"Response: {resp_text}" in captured.out
