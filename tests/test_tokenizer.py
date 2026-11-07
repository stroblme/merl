from merl import tokenizer


def test_roundtrip():
    text = "Setup a scene for dinner — 21.5°C"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bytes_pass_through():
    data = bytes([tokenizer.SUF, 0x41, tokenizer.PRE, 0xC3, 0xA9, tokenizer.MID])
    assert bytes(tokenizer.encode(data)) == data


def test_sentinels_are_not_utf8():
    for sentinel in (tokenizer.SUF, tokenizer.PRE, tokenizer.MID):
        try:
            bytes([sentinel]).decode("utf-8")
        except UnicodeDecodeError:
            continue
        raise AssertionError(f"{sentinel:#x} decodes, so it could clash with text")
