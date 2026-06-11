from src.stomp_client import _encode_frame, _parse_frames


def test_stomp_frame_round_trip() -> None:
    encoded = _encode_frame(
        "SEND",
        {"destination": "/sim/events", "category": "client", "name": "initialized"},
        '{"category":"client","name":"initialized","data":{}}',
    )

    frames = _parse_frames(encoded)

    assert len(frames) == 1
    assert frames[0].command == "SEND"
    assert frames[0].headers["destination"] == "/sim/events"
    assert frames[0].headers["category"] == "client"
    assert frames[0].headers["name"] == "initialized"
    assert frames[0].body == '{"category":"client","name":"initialized","data":{}}'
