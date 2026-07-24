####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str, /) -> str:
            return payload.upper()

        def dumps(self, obj: str, /) -> str:
            return obj.lower()

    serializer = StringSerializer()
    assert serializer.loads("hello") == "HELLO"

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return payload[::-1]

        def dumps(self, obj: bytes, /) -> bytes:
            return obj[::-1]

    serializer = BytesSerializer()
    assert serializer.loads(b"hello") == b"olleh"

    # Test with a mixed serializer (accepts both str and bytes)
    class MixedSerializer:
        def loads(self, payload: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return payload.upper()
            return payload[::-1]

        def dumps(self, obj: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return obj.lower()
            return obj[::-1]

    serializer = MixedSerializer()
    assert serializer.loads("hello") == "HELLO"
    assert serializer.loads(b"hello") == b"olleh"


# LLM-generated content at query #2
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def loads(self, payload: str) -> dict:
            return {"custom": payload}

        def dumps(self, obj: dict) -> str:
            return obj["custom"]

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b'{"custom": "test"}'
    assert serializer.load_payload(payload) == {"custom": "test"}

    # Test with custom bytes serializer
    class CustomBytesSerializer:
        def loads(self, payload: bytes) -> dict:
            return {"custom": payload.decode("utf-8")}

        def dumps(self, obj: dict) -> bytes:
            return obj["custom"].encode("utf-8")

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b'test'
    assert serializer.load_payload(payload) == {"custom": "test"}

    # Test with BadPayload exception
    class FailingSerializer:
        def loads(self, payload: str) -> dict:
            raise ValueError("Test error")

        def dumps(self, obj: dict) -> str:
            return str(obj)

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b'{"key": "value"}'
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)


# LLM-generated content at query #3
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer
    class TextSerializer:
        def loads(self, payload: str, /) -> str:
            return f"loaded_{payload}"

        def dumps(self, obj: str, /) -> str:
            return f"dumped_{obj}"

    serializer = TextSerializer()
    assert serializer.loads("test") == "loaded_test"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return b"loaded_" + payload

        def dumps(self, obj: bytes, /) -> bytes:
            return b"dumped_" + obj

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == b"loaded_test"


# LLM-generated content at query #4
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json)
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    assert isinstance(signed, str)
    assert serializer.loads(signed) == data

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')
        def loads(self, data):
            return eval(data.decode('utf-8'))

    serializer_bytes = Serializer("secret-key", serializer=BytesSerializer())
    signed_bytes = serializer_bytes.dumps(data)
    assert isinstance(signed_bytes, bytes)
    assert serializer_bytes.loads(signed_bytes) == data

    # Test with custom salt
    serializer_salt = Serializer("secret-key", salt="custom-salt")
    signed_salt = serializer_salt.dumps(data)
    assert serializer_salt.loads(signed_salt) == data

    # Test with key rotation
    serializer_keys = Serializer(["old-key", "new-key"])
    signed_keys = serializer_keys.dumps(data)
    assert serializer_keys.loads(signed_keys) == data

    # Test with different data types
    assert serializer.dumps(42) == serializer.dumps(42)
    assert serializer.dumps("string") == serializer.dumps("string")
    assert serializer.dumps([1, 2, 3]) == serializer.dumps([1, 2, 3])


# LLM-generated content at query #5
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json) and text output
    s = Serializer("secret-key")
    data = {"a": 1, "b": 2}
    signed = s.dumps(data)
    assert isinstance(signed, str)
    assert s.loads(signed) == data

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")

        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))

    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    signed_bytes = s_bytes.dumps(data)
    assert isinstance(signed_bytes, bytes)
    assert s_bytes.loads(signed_bytes) == data

    # Test with custom salt
    s_salt = Serializer("secret-key", salt="custom-salt")
    signed_salt = s_salt.dumps(data)
    assert s_salt.loads(signed_salt) == data

    # Test with key rotation
    s_rotated = Serializer(["old-key", "new-key"])
    signed_rotated = s_rotated.dumps(data)
    assert s_rotated.loads(signed_rotated) == data

    # Test with different data types
    assert s.dumps("string") == s.loads(s.dumps("string"))
    assert s.dumps(123) == s.loads(s.dumps(123))
    assert s.dumps([1, 2, 3]) == s.loads(s.dumps([1, 2, 3]))


# LLM-generated content at query #6
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            return int(payload)

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"42"
    assert serializer.load_payload(payload) == 42

    # Test with custom bytes serializer
    class CustomBytesSerializer:
        def dumps(self, obj):
            return bytes(obj, 'utf-8')

        def loads(self, payload):
            return payload.decode('utf-8')

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b"hello"
    assert serializer.load_payload(payload) == "hello"

    # Test with invalid payload
    serializer = Serializer("secret-key")
    payload = b"invalid json"
    with pytest.raises(BadPayload):
        serializer.load_payload(payload)

    # Test with custom serializer that raises exception
    class FailingSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            raise ValueError("Custom error")

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b"anything"
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(payload)
    assert "Custom error" in str(exc_info.value.original_error)


# LLM-generated content at query #7
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a text serializer (str)
    class TextSerializer:
        def loads(self, payload: str, /) -> str:
            return f"loaded_{payload}"

        def dumps(self, obj: str, /) -> str:
            return f"dumped_{obj}"

    serializer = TextSerializer()
    assert serializer.loads("test_payload") == "loaded_test_payload"

    # Test with a bytes serializer (bytes)
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return b"loaded_" + payload

        def dumps(self, obj: bytes, /) -> bytes:
            return b"dumped_" + obj

    serializer = BytesSerializer()
    assert serializer.loads(b"test_payload") == b"loaded_test_payload"


# LLM-generated content at query #8
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str, /) -> str:
            return f"loaded_{payload}"

        def dumps(self, obj: str, /) -> str:
            return f"dumped_{obj}"

    serializer = StringSerializer()
    assert serializer.loads("test") == "loaded_test"

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return b"loaded_" + payload

        def dumps(self, obj: bytes, /) -> bytes:
            return b"dumped_" + obj

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == b"loaded_test"

    # Test with a mixed serializer (str and bytes)
    class MixedSerializer:
        def loads(self, payload: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return f"loaded_{payload}"
            return b"loaded_" + payload

        def dumps(self, obj: str | bytes, /) -> str | bytes:
            if isinstance(obj, str):
                return f"dumped_{obj}"
            return b"dumped_" + obj

    serializer = MixedSerializer()
    assert serializer.loads("test") == "loaded_test"
    assert serializer.loads(b"test") == b"loaded_test"


# LLM-generated content at query #9
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default parameters
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == ["secret-key"]
    assert signers[0].salt == b"itsdangerous"

    # Test with custom salt
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"

    # Test with fallback signers
    fallback_signers = [
        {"digest_method": "sha256"},
        (Signer, {"digest_method": "sha512"}),
        Signer,
    ]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4  # 1 default + 3 fallbacks
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == ["secret-key"]
    assert signers[1].secret_keys == ["secret-key"]
    assert signers[1].digest_method.name == "sha256"
    assert signers[2].secret_keys == ["secret-key"]
    assert signers[2].digest_method.name == "sha512"
    assert signers[3].secret_keys == ["secret-key"]

    # Test with key rotation
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == ["old-key", "new-key"]

    # Test with key rotation and fallback signers
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"digest_method": "sha256"}],
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2  # 1 default + 1 fallback
    assert signers[0].secret_keys == ["old-key", "new-key"]
    assert signers[1].secret_keys == ["old-key"]
    assert signers[1].digest_method.name == "sha256"
    assert signers[1].secret_keys == ["new-key"]
    assert signers[1].digest_method.name == "sha256"


# LLM-generated content at query #10
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer (JSON)
    text_serializer = json
    payload = '{"key": "value"}'
    assert text_serializer.loads(payload) == {"key": "value"}

    # Test with a simple bytes serializer (pickle-like)
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return {"deserialized": payload}

        def dumps(self, obj: t.Any, /) -> bytes:
            return b"serialized"

    bytes_serializer = BytesSerializer()
    payload_bytes = b"test_payload"
    assert bytes_serializer.loads(payload_bytes) == {"deserialized": b"test_payload"}

    # Test with a custom serializer that doesn't match the strict protocol
    class CustomSerializer:
        def loads(self, payload: str, extra_arg=None, /) -> t.Any:
            return {"custom": payload}

        def dumps(self, obj: t.Any, extra_arg=None, /) -> str:
            return "custom_serialized"

    custom_serializer = CustomSerializer()
    payload_custom = "custom_payload"
    assert custom_serializer.loads(payload_custom) == {"custom": "custom_payload"}


# LLM-generated content at query #11
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def loads(self, s):
            return f"custom_{s}"

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"test_data"
    result = serializer.load_payload(payload)
    assert result == "custom_test_data"

    # Test with bytes serializer
    class CustomBytesSerializer:
        def loads(self, b):
            return f"bytes_{b.decode()}"

        def dumps(self, obj):
            return str(obj).encode()

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b"test_bytes"
    result = serializer.load_payload(payload)
    assert result == "bytes_test_bytes"

    # Test with BadPayload exception
    class FailingSerializer:
        def loads(self, s):
            raise ValueError("Serialization failed")

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b"test_data"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)


# LLM-generated content at query #12
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def loads(self, s):
            return f"loaded: {s}"

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"test_data"
    assert serializer.load_payload(payload) == "loaded: test_data"

    # Test with bytes serializer
    class CustomBytesSerializer:
        def loads(self, data):
            return f"bytes_loaded: {data}"

        def dumps(self, obj):
            return bytes(obj, 'utf-8')

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b"test_bytes"
    assert serializer.load_payload(payload) == "bytes_loaded: b'test_bytes'"

    # Test with invalid payload (should raise BadPayload)
    serializer = Serializer("secret-key")
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass

    # Test with custom serializer that raises exception
    class FailingSerializer:
        def loads(self, data):
            raise ValueError("Custom error")

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b"test"
    try:
        serializer.load_payload(payload)
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)


# LLM-generated content at query #13
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str, /) -> str:
            return payload.upper()

        def dumps(self, obj: str, /) -> str:
            return obj.lower()

    serializer = StringSerializer()
    assert serializer.loads("hello") == "HELLO"

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return payload[::-1]

        def dumps(self, obj: bytes, /) -> bytes:
            return obj[::-1]

    serializer = BytesSerializer()
    assert serializer.loads(b"hello") == b"olleh"

    # Test with a JSON-like serializer
    class JSONLikeSerializer:
        def loads(self, payload: str, /) -> dict:
            return {"data": payload}

        def dumps(self, obj: dict, /) -> str:
            return obj["data"]

    serializer = JSONLikeSerializer()
    assert serializer.loads("test") == {"data": "test"}


# LLM-generated content at query #14
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert result == '{"key": "value"}'
    assert isinstance(result, str)


# LLM-generated content at query #15
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON serializer
    json_serializer = json
    data = {"key": "value"}
    serialized = json_serializer.dumps(data)
    assert json_serializer.loads(serialized) == data

    # Test with a custom serializer that returns bytes
    class CustomBytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

        def loads(self, payload):
            return payload.decode('utf-8')

    bytes_serializer = CustomBytesSerializer()
    data = "test_string"
    serialized = bytes_serializer.dumps(data)
    assert bytes_serializer.loads(serialized) == data

    # Test with a custom serializer that returns str
    class CustomStrSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            return payload

    str_serializer = CustomStrSerializer()
    data = "test_string"
    serialized = str_serializer.dumps(data)
    assert str_serializer.loads(serialized) == data


# LLM-generated content at query #16
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def loads(self, payload):
            return payload

        def dumps(self, obj):
            return obj

    serializer = TestSerializer()
    assert serializer.dumps({"key": "value"}) == {"key": "value"}


# LLM-generated content at query #17
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str, /) -> str:
            return payload.upper()

        def dumps(self, obj: str, /) -> str:
            return obj.lower()

    serializer = StringSerializer()
    assert serializer.loads("hello") == "HELLO"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return payload * 2

        def dumps(self, obj: bytes, /) -> bytes:
            return obj[::-1]

    serializer = BytesSerializer()
    assert serializer.loads(b"abc") == b"abcabc"

    # Test with a more complex serializer (e.g., JSON-like)
    class DictSerializer:
        def loads(self, payload: str, /) -> dict:
            return {"data": payload}

        def dumps(self, obj: dict, /) -> str:
            return obj["data"]

    serializer = DictSerializer()
    assert serializer.loads("test") == {"data": "test"}


# LLM-generated content at query #18
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def loads(self, s):
            return f"custom_{s}"

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"test_payload"
    assert serializer.load_payload(payload) == "custom_test_payload"

    # Test with custom bytes serializer
    class CustomBytesSerializer:
        def loads(self, s):
            return f"custom_{s.decode()}"

        def dumps(self, obj):
            return str(obj).encode()

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b"test_payload"
    assert serializer.load_payload(payload) == "custom_test_payload"

    # Test with invalid payload
    serializer = Serializer("secret-key")
    payload = b'invalid_json'
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass

    # Test with custom serializer that raises exception
    class FailingSerializer:
        def loads(self, s):
            raise ValueError("Custom error")

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b"test_payload"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert str(e) == "Could not load the payload because an exception occurred on unserializing the data."
        assert isinstance(e.original_error, ValueError)
        assert str(e.original_error) == "Custom error"


# LLM-generated content at query #19
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer
    class TextSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return f"loaded: {payload}"

        def dumps(self, obj: t.Any, /) -> str:
            return f"dumped: {obj}"

    serializer = TextSerializer()
    assert serializer.loads("test") == "loaded: test"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return f"loaded: {payload.decode()}"

        def dumps(self, obj: t.Any, /) -> bytes:
            return f"dumped: {obj}".encode()

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == "loaded: test"

    # Test with a serializer that doesn't match the strict protocol
    class NonStrictSerializer:
        def loads(self, payload: str, extra_arg=None) -> t.Any:
            return f"loaded: {payload}"

        def dumps(self, obj: t.Any, extra_arg=None) -> str:
            return f"dumped: {obj}"

    serializer = NonStrictSerializer()
    assert serializer.loads("test") == "loaded: test"


# LLM-generated content at query #20
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str, /) -> str:
            return payload.upper()

        def dumps(self, obj: str, /) -> str:
            return obj.lower()

    serializer = StringSerializer()
    assert serializer.loads("hello") == "HELLO"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return payload[::-1]

        def dumps(self, obj: bytes, /) -> bytes:
            return obj[::-1]

    serializer = BytesSerializer()
    assert serializer.loads(b"hello") == b"olleh"

    # Test with a serializer that handles both str and bytes
    class FlexibleSerializer:
        def loads(self, payload: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return payload.upper()
            return payload[::-1]

        def dumps(self, obj: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return obj.lower()
            return obj[::-1]

    serializer = FlexibleSerializer()
    assert serializer.loads("hello") == "HELLO"
    assert serializer.loads(b"hello") == b"olleh"


# LLM-generated content at query #21
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json) and text output
    s = Serializer("secret-key")
    data = {"key": "value"}
    signed = s.dumps(data)
    assert isinstance(signed, str)
    assert s.loads(signed) == data

    # Test with default serializer (json) and bytes output
    s_bytes = Serializer("secret-key", serializer=json)
    signed_bytes = s_bytes.dumps(data)
    assert isinstance(signed_bytes, bytes)
    assert s_bytes.loads(signed_bytes) == data

    # Test with custom serializer (text)
    class CustomTextSerializer:
        def dumps(self, obj):
            return f"custom-{obj['key']}"

        def loads(self, s):
            return {"key": s.split("-")[1]}

    s_custom_text = Serializer("secret-key", serializer=CustomTextSerializer())
    signed_custom_text = s_custom_text.dumps(data)
    assert isinstance(signed_custom_text, str)
    assert s_custom_text.loads(signed_custom_text) == data

    # Test with custom serializer (bytes)
    class CustomBytesSerializer:
        def dumps(self, obj):
            return f"custom-{obj['key']}".encode("utf-8")

        def loads(self, s):
            return {"key": s.decode("utf-8").split("-")[1]}

    s_custom_bytes = Serializer("secret-key", serializer=CustomBytesSerializer())
    signed_custom_bytes = s_custom_bytes.dumps(data)
    assert isinstance(signed_custom_bytes, bytes)
    assert s_custom_bytes.loads(signed_custom_bytes) == data

    # Test with salt parameter
    s_salt = Serializer("secret-key", salt="custom-salt")
    signed_salt = s_salt.dumps(data)
    assert s_salt.loads(signed_salt) == data

    # Test with key rotation
    s_rotated = Serializer(["old-key", "new-key"])
    signed_rotated = s_rotated.dumps(data)
    assert s_rotated.loads(signed_rotated) == data


# LLM-generated content at query #22
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer (str)
    class TextSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return f"loaded: {payload}"

        def dumps(self, obj: t.Any, /) -> str:
            return str(obj)

    serializer = TextSerializer()
    assert serializer.loads("test") == "loaded: test"

    # Test with a simple binary serializer (bytes)
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return f"loaded: {payload.decode()}"

        def dumps(self, obj: t.Any, /) -> bytes:
            return str(obj).encode()

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == "loaded: test"

    # Test with a serializer that doesn't match the strict protocol
    class NonStrictSerializer:
        def loads(self, payload, **kwargs) -> t.Any:
            return f"loaded: {payload}"

        def dumps(self, obj, **kwargs) -> str:
            return str(obj)

    serializer = NonStrictSerializer()
    assert serializer.loads("test") == "loaded: test"


# LLM-generated content at query #23
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json) and text output
    s = Serializer("secret-key")
    data = {"key": "value"}
    signed = s.dumps(data)
    assert isinstance(signed, str)
    assert s.loads(signed) == data

    # Test with default serializer (json) and bytes output
    s_bytes = Serializer("secret-key", serializer=json)
    signed_bytes = s_bytes.dumps(data)
    assert isinstance(signed_bytes, bytes)
    assert s_bytes.loads(signed_bytes) == data

    # Test with custom serializer (text)
    class CustomTextSerializer:
        def dumps(self, obj):
            return f"custom-{obj['key']}"

        def loads(self, s):
            return {"key": s.split("-")[1]}

    s_custom_text = Serializer("secret-key", serializer=CustomTextSerializer())
    signed_custom_text = s_custom_text.dumps(data)
    assert isinstance(signed_custom_text, str)
    assert s_custom_text.loads(signed_custom_text) == data

    # Test with custom serializer (bytes)
    class CustomBytesSerializer:
        def dumps(self, obj):
            return f"custom-{obj['key']}".encode()

        def loads(self, s):
            return {"key": s.decode().split("-")[1]}

    s_custom_bytes = Serializer("secret-key", serializer=CustomBytesSerializer())
    signed_custom_bytes = s_custom_bytes.dumps(data)
    assert isinstance(signed_custom_bytes, bytes)
    assert s_custom_bytes.loads(signed_custom_bytes) == data

    # Test with different salt
    s_salt = Serializer("secret-key", salt="custom-salt")
    signed_salt = s_salt.dumps(data)
    assert s_salt.loads(signed_salt) == data

    # Test with key rotation
    s_rotated = Serializer(["old-key", "new-key"])
    signed_rotated = s_rotated.dumps(data)
    assert s_rotated.loads(signed_rotated) == data


# LLM-generated content at query #24
#--------------------------

```python
def test_Serializer_dumps():
    serializer = Serializer("secret-key")
    obj = {"key": "value"}
    signed_data = serializer.dumps(obj)
    assert isinstance(signed_data, str)
    assert serializer.loads(signed_data) == obj

    # Test with bytes serializer
    class BytesSerializer:
        @staticmethod
        def dumps(data):
            return json.dumps(data).encode("utf-8")

        @staticmethod
        def loads(data):
            return json.loads(data.decode("utf-8"))

    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    signed_bytes = bytes_serializer.dumps(obj)
    assert isinstance(signed_bytes, bytes)
    assert bytes_serializer.loads(signed_bytes) == obj

    # Test with custom salt
    custom_salt_serializer = Serializer("secret-key", salt="custom-salt")
    signed_custom = custom_salt_serializer.dumps(obj)
    assert isinstance(signed_custom, str)
    assert custom_salt_serializer.loads(signed_custom) == obj

    # Test with key rotation
    keys = ["old-key", "new-key"]
    key_rotation_serializer = Serializer(keys)
    signed_rotated = key_rotation_serializer.dumps(obj)
    assert isinstance(signed_rotated, str)
    assert key_rotation_serializer.loads(signed_rotated) == obj


# LLM-generated content at query #25
#--------------------------

```python
def test__PDataSerializer_loads():
    class TestSerializer:
        def loads(self, payload: str, /) -> str:
            return f"loaded_{payload}"

        def dumps(self, obj: str, /) -> str:
            return f"dumped_{obj}"

    serializer = TestSerializer()
    assert serializer.loads("test_payload") == "loaded_test_payload"


# LLM-generated content at query #26
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def loads(self, s):
            return f"loaded: {s}"

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"test_data"
    assert serializer.load_payload(payload) == "loaded: test_data"

    # Test with custom bytes serializer
    class CustomBytesSerializer:
        def loads(self, s):
            return f"loaded: {s.decode('utf-8')}"

        def dumps(self, obj):
            return bytes(obj, 'utf-8')

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b"test_data"
    assert serializer.load_payload(payload) == "loaded: test_data"

    # Test with BadPayload exception
    class FailingSerializer:
        def loads(self, s):
            raise ValueError("Test error")

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b"test_data"
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(payload)
    assert "Could not load the payload" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, ValueError)


# LLM-generated content at query #27
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def dumps(self, obj, /) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert result == '{"key": "value"}'


# LLM-generated content at query #28
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json)
    s = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = s.dumps(data)
    assert isinstance(signed_data, str)
    assert s.loads(signed_data) == data

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")

        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))

    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    signed_data_bytes = s_bytes.dumps(data)
    assert isinstance(signed_data_bytes, bytes)
    assert s_bytes.loads(signed_data_bytes) == data

    # Test with custom salt
    s_salt = Serializer("secret-key", salt="custom-salt")
    signed_data_salt = s_salt.dumps(data)
    assert s_salt.loads(signed_data_salt) == data

    # Test with key rotation
    s_rotation = Serializer(["old-key", "new-key"])
    signed_data_rotation = s_rotation.dumps(data)
    assert s_rotation.loads(signed_data_rotation) == data

    # Test with different data types
    assert s.dumps(None) is not None
    assert s.dumps(123) is not None
    assert s.dumps([1, 2, 3]) is not None


# LLM-generated content at query #29
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns str
    class StrSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = StrSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"

    # Test with a simple serializer that returns bytes
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"


# LLM-generated content at query #30
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def loads(self, s):
            return f"custom_{s}"

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"test_data"
    assert serializer.load_payload(payload) == "custom_test_data"

    # Test with bytes serializer
    class CustomBytesSerializer:
        def loads(self, b):
            return f"bytes_{b.decode()}"

        def dumps(self, obj):
            return str(obj).encode()

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b"test_data"
    assert serializer.load_payload(payload) == "bytes_test_data"

    # Test with BadPayload exception
    class FailingSerializer:
        def loads(self, s):
            raise ValueError("Test error")

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b"test_data"
    try:
        serializer.load_payload(payload)
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)


# LLM-generated content at query #31
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns str
    class StrSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = StrSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"
    assert isinstance(serializer.dumps({"key": "value"}), str)

    # Test with a simple serializer that returns bytes
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"
    assert isinstance(serializer.dumps({"key": "value"}), bytes)

    # Test with a more complex object
    class ComplexSerializer:
        def dumps(self, obj):
            if isinstance(obj, dict):
                return json.dumps(obj)
            return str(obj)

    serializer = ComplexSerializer()
    assert serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert isinstance(serializer.dumps({"key": "value"}), str)


# LLM-generated content at query #32
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns str
    class StrSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = StrSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"

    # Test with a serializer that returns bytes
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"


# LLM-generated content at query #33
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple object
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert "key" in signed_data

    # Test with a different salt
    serializer_with_salt = Serializer("secret-key", salt="custom-salt")
    signed_data_salt = serializer_with_salt.dumps(data)
    assert signed_data_salt != signed_data

    # Test with a bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))

    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    signed_bytes = bytes_serializer.dumps(data)
    assert isinstance(signed_bytes, bytes)
    assert b"key" in signed_bytes

    # Test with a custom signer
    class CustomSigner(Signer):
        pass

    custom_serializer = Serializer("secret-key", signer=CustomSigner)
    signed_custom = custom_serializer.dumps(data)
    assert isinstance(signed_custom, str)
    assert "key" in signed_custom

    # Test with key rotation
    serializer_rotated = Serializer(["old-key", "new-key"])
    signed_rotated = serializer_rotated.dumps(data)
    assert isinstance(signed_rotated, str)
    assert "key" in signed_rotated


# LLM-generated content at query #34
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a text serializer (JSON)
    text_serializer = json
    payload = '{"key": "value"}'
    assert text_serializer.loads(payload) == {"key": "value"}

    # Test with a bytes serializer (custom)
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode('utf-8')

        def dumps(self, obj: t.Any) -> bytes:
            return obj.encode('utf-8')

    bytes_serializer = BytesSerializer()
    payload_bytes = b'{"key": "value"}'
    assert bytes_serializer.loads(payload_bytes) == '{"key": "value"}'

    # Test with a non-standard serializer (returns a different type)
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()

        def dumps(self, obj: t.Any) -> str:
            return str(obj).lower()

    custom_serializer = CustomSerializer()
    payload_custom = "hello"
    assert custom_serializer.loads(payload_custom) == "HELLO"


# LLM-generated content at query #35
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple object
    serializer = json
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert result == '{"key": "value"}'

    # Test with a list
    obj = [1, 2, 3]
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert result == '[1, 2, 3]'

    # Test with a nested object
    obj = {"a": {"b": [1, 2, 3]}}
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert result == '{"a": {"b": [1, 2, 3]}}'

    # Test with a string
    obj = "hello"
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert result == '"hello"'

    # Test with an integer
    obj = 42
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert result == '42'


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer (JSON)
    text_serializer = json
    assert text_serializer.loads('{"key": "value"}') == {"key": "value"}

    # Test with a bytes serializer (custom)
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode('utf-8')

        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode('utf-8')

    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == '{"key": "value"}'

    # Test with a custom serializer that returns a different type
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()

        def dumps(self, obj: t.Any) -> str:
            return str(obj).lower()

    custom_serializer = CustomSerializer()
    assert custom_serializer.loads("hello") == "HELLO"


# LLM-generated content at query #2
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer
    text_serializer = json
    assert text_serializer.loads('{"key": "value"}') == {"key": "value"}

    # Test with a simple bytes serializer (assuming a custom serializer)
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))

        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')

    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}

    # Test with a custom serializer that doesn't match the strict protocol
    class CustomSerializer:
        def loads(self, payload: str | bytes, extra_arg=None) -> t.Any:
            if isinstance(payload, bytes):
                payload = payload.decode('utf-8')
            return json.loads(payload)

        def dumps(self, obj: t.Any, extra_arg=None) -> str:
            return json.dumps(obj)

    custom_serializer = CustomSerializer()
    assert custom_serializer.loads('{"key": "value"}') == {"key": "value"}
    assert custom_serializer.loads(b'{"key": "value"}') == {"key": "value"}


# LLM-generated content at query #3
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer
    class TextSerializer:
        def loads(self, payload: str, /) -> str:
            return f"loaded: {payload}"

        def dumps(self, obj: str, /) -> str:
            return f"dumped: {obj}"

    serializer = TextSerializer()
    assert serializer.loads("test") == "loaded: test"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return b"loaded: " + payload

        def dumps(self, obj: bytes, /) -> bytes:
            return b"dumped: " + obj

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == b"loaded: test"

    # Test with a serializer that matches the protocol but has additional arguments
    class FlexibleSerializer:
        def loads(self, payload: str | bytes, encoding: str = "utf-8", /) -> str:
            if isinstance(payload, bytes):
                return f"loaded: {payload.decode(encoding)}"
            return f"loaded: {payload}"

        def dumps(self, obj: str, encoding: str = "utf-8", /) -> str:
            return f"dumped: {obj}"

    serializer = FlexibleSerializer()
    assert serializer.loads("test") == "loaded: test"
    assert serializer.loads(b"test") == "loaded: test"


# LLM-generated content at query #4
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str, /) -> str:
            return f"loaded_{payload}"

        def dumps(self, obj: str, /) -> str:
            return f"dumped_{obj}"

    serializer = StringSerializer()
    assert serializer.loads("test") == "loaded_test"

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return b"loaded_" + payload

        def dumps(self, obj: bytes, /) -> bytes:
            return b"dumped_" + obj

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == b"loaded_test"

    # Test with a mixed serializer (str and bytes)
    class MixedSerializer:
        def loads(self, payload: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return f"loaded_{payload}"
            return b"loaded_" + payload

        def dumps(self, obj: str | bytes, /) -> str | bytes:
            if isinstance(obj, str):
                return f"dumped_{obj}"
            return b"dumped_" + obj

    serializer = MixedSerializer()
    assert serializer.loads("test") == "loaded_test"
    assert serializer.loads(b"test") == b"loaded_test"


# LLM-generated content at query #5
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default parameters
    serializer = Serializer("secret-key")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"secret-key"]
    assert unsigners[0].salt == b"itsdangerous"

    # Test with custom salt
    serializer = Serializer("secret-key", salt="custom-salt")
    unsigners = list(serializer.iter_unsigners())
    assert unsigners[0].salt == b"custom-salt"

    # Test with fallback signers as dict
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"digest_method": "sha256"}]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert unsigners[1].digest_method.name == "sha256"

    # Test with fallback signers as tuple
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(Signer, {"digest_method": "sha256"})]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert unsigners[1].digest_method.name == "sha256"

    # Test with fallback signers as Signer class
    serializer = Serializer(
        "secret-key",
        fallback_signers=[Signer]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[1], Signer)

    # Test with multiple fallback signers
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"digest_method": "sha256"},
            (Signer, {"digest_method": "sha512"})
        ]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 3
    assert unsigners[1].digest_method.name == "sha256"
    assert unsigners[2].digest_method.name == "sha512"

    # Test with key rotation
    serializer = Serializer(["old-key", "new-key"])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert unsigners[0].secret_keys == [b"old-key", b"new-key"]

    # Test with custom salt in iter_unsigners
    serializer = Serializer("secret-key")
    unsigners = list(serializer.iter_unsigners(salt="custom-salt"))
    assert unsigners[0].salt == b"custom-salt"


# LLM-generated content at query #6
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a text serializer
    class TextSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return f"loaded: {payload}"

        def dumps(self, obj: t.Any, /) -> str:
            return str(obj)

    serializer = TextSerializer()
    assert serializer.loads("test") == "loaded: test"

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return f"loaded: {payload.decode()}"

        def dumps(self, obj: t.Any, /) -> bytes:
            return str(obj).encode()

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == "loaded: test"


# LLM-generated content at query #7
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default parameters
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[0].salt == b"itsdangerous"

    # Test with custom salt
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"

    # Test with custom salt in iter_unsigners
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners(salt="another-salt"))
    assert len(signers) == 1
    assert signers[0].salt == b"another-salt"

    # Test with key rotation
    serializer = Serializer(["key1", "key2", "key3"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"key1", b"key2", b"key3"]

    # Test with fallback signers as dict
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].digest_method == Signer.digest_method
    assert signers[1].digest_method == "sha256"

    # Test with fallback signers as tuple
    from itsdangerous.signer import Signer as CustomSigner
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(CustomSigner, {"digest_method": "sha512"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], CustomSigner)
    assert signers[1].digest_method == "sha512"

    # Test with fallback signers as class
    serializer = Serializer(
        "secret-key",
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], CustomSigner)

    # Test with multiple fallback signers
    serializer = Serializer(
        ["key1", "key2"],
        fallback_signers=[
            {"digest_method": "sha256"},
            (CustomSigner, {"digest_method": "sha512"}),
            CustomSigner
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1 + 3 * 2  # 1 default + 3 fallbacks * 2 keys
    assert signers[0].secret_keys == [b"key1", b"key2"]
    assert signers[1].secret_keys == [b"key1"]
    assert signers[2].secret_keys == [b"key2"]
    assert signers[3].secret_keys == [b"key1"]
    assert signers[4].secret_keys == [b"key2"]
    assert signers[5].secret_keys == [b"key1"]
    assert signers[6].secret_keys == [b"key2"]


# LLM-generated content at query #8
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert result == '{"key": "value"}'


# LLM-generated content at query #9
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer(_PDataSerializer[str]):
        def loads(self, payload: str, /) -> t.Any:
            return json.loads(payload)

        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    data = {"key": "value"}
    dumped = serializer.dumps(data)
    assert dumped == '{"key": "value"}'
    assert isinstance(dumped, str)

    class TestBytesSerializer(_PDataSerializer[bytes]):
        def loads(self, payload: bytes, /) -> t.Any:
            return json.loads(payload.decode('utf-8'))

        def dumps(self, obj: t.Any, /) -> bytes:
            return json.dumps(obj).encode('utf-8')

    bytes_serializer = TestBytesSerializer()
    dumped_bytes = bytes_serializer.dumps(data)
    assert dumped_bytes == b'{"key": "value"}'
    assert isinstance(dumped_bytes, bytes)


# LLM-generated content at query #10
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple dictionary
    serializer = json
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert result == '{"key": "value"}'

    # Test with a list
    obj = [1, 2, 3]
    result = serializer.dumps(obj)
    assert result == "[1, 2, 3]"

    # Test with a string
    obj = "test string"
    result = serializer.dumps(obj)
    assert result == '"test string"'

    # Test with a number
    obj = 42
    result = serializer.dumps(obj)
    assert result == "42"

    # Test with a boolean
    obj = True
    result = serializer.dumps(obj)
    assert result == "true"

    # Test with None
    obj = None
    result = serializer.dumps(obj)
    assert result == "null"


# LLM-generated content at query #11
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple dict serializer
    class DictSerializer:
        def loads(self, payload):
            return {"data": payload}

        def dumps(self, obj):
            return str(obj["data"])

    serializer = DictSerializer()
    payload = "test_data"
    result = serializer.loads(payload)
    assert result == {"data": "test_data"}

    # Test with a JSON serializer
    json_payload = '{"key": "value"}'
    result = json.loads(json_payload)
    assert result == {"key": "value"}

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return payload.decode("utf-8")

        def dumps(self, obj):
            return obj.encode("utf-8")

    bytes_serializer = BytesSerializer()
    bytes_payload = b"test_data"
    result = bytes_serializer.loads(bytes_payload)
    assert result == "test_data"


# LLM-generated content at query #12
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with default JSON serializer
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert "key" in signed_data

    # Test with custom text serializer
    class CustomTextSerializer:
        def dumps(self, obj):
            return f"custom-{obj['key']}"

        def loads(self, payload):
            return {"key": payload.split("-")[1]}

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert "custom-value" in signed_data

    # Test with bytes serializer
    class CustomBytesSerializer:
        def dumps(self, obj):
            return f"custom-{obj['key']}".encode()

        def loads(self, payload):
            return {"key": payload.decode().split("-")[1]}

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, bytes)
    assert b"custom-value" in signed_data


# LLM-generated content at query #13
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert result == '{"key": "value"}'


# LLM-generated content at query #14
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer (json)
    serializer = json
    payload = '{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}

    # Test with a simple bytes serializer (pickle)
    import pickle
    serializer = pickle
    payload = pickle.dumps({"key": "value"})
    result = serializer.loads(payload)
    assert result == {"key": "value"}

    # Test with a custom text serializer
    class CustomTextSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return payload.upper()

        def dumps(self, obj: t.Any, /) -> str:
            return str(obj).lower()

    serializer = CustomTextSerializer()
    payload = "hello"
    result = serializer.loads(payload)
    assert result == "HELLO"

    # Test with a custom bytes serializer
    class CustomBytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return payload.upper()

        def dumps(self, obj: t.Any, /) -> bytes:
            return bytes(str(obj).lower(), 'utf-8')

    serializer = CustomBytesSerializer()
    payload = b"hello"
    result = serializer.loads(payload)
    assert result == b"HELLO"


# LLM-generated content at query #15
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def loads(self, s):
            return f"loaded: {s}"

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"test_data"
    assert serializer.load_payload(payload) == "loaded: test_data"

    # Test with bytes serializer
    class CustomBytesSerializer:
        def loads(self, b):
            return f"bytes_loaded: {b}"

        def dumps(self, obj):
            return bytes(obj, 'utf-8')

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b"test_bytes"
    assert serializer.load_payload(payload) == "bytes_loaded: b'test_bytes'"

    # Test with BadPayload exception
    class FailingSerializer:
        def loads(self, s):
            raise ValueError("Serialization failed")

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b"test_data"

    try:
        serializer.load_payload(payload)
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)


# LLM-generated content at query #16
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def loads(self, payload):
            return payload

        def dumps(self, obj):
            return obj

    serializer = TestSerializer()
    assert serializer.dumps({"key": "value"}) == {"key": "value"}


# LLM-generated content at query #17
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = TestSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps("test") == "test"


# LLM-generated content at query #18
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer
    class TextSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return f"loaded_{payload}"

        def dumps(self, obj: t.Any, /) -> str:
            return f"dumped_{obj}"

    serializer = TextSerializer()
    assert serializer.loads("test") == "loaded_test"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return f"loaded_{payload.decode()}"

        def dumps(self, obj: t.Any, /) -> bytes:
            return f"dumped_{obj}".encode()

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == "loaded_test"

    # Test with a serializer that doesn't match the strict protocol
    class CustomSerializer:
        def loads(self, payload, **kwargs) -> t.Any:
            return f"loaded_{payload}"

        def dumps(self, obj, **kwargs) -> str:
            return f"dumped_{obj}"

    serializer = CustomSerializer()
    assert serializer.loads("test") == "loaded_test"


# LLM-generated content at query #19
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer
    class TextSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return f"loaded_{payload}"

        def dumps(self, obj: t.Any, /) -> str:
            return f"dumped_{obj}"

    serializer = TextSerializer()
    assert serializer.loads("test") == "loaded_test"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return f"loaded_{payload.decode()}"

        def dumps(self, obj: t.Any, /) -> bytes:
            return f"dumped_{obj}".encode()

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == "loaded_test"


# LLM-generated content at query #20
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple dictionary
    serializer = json
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert result == '{"key": "value"}'

    # Test with a list
    obj = [1, 2, 3]
    result = serializer.dumps(obj)
    assert result == "[1, 2, 3]"

    # Test with a string
    obj = "test string"
    result = serializer.dumps(obj)
    assert result == '"test string"'

    # Test with a number
    obj = 42
    result = serializer.dumps(obj)
    assert result == "42"

    # Test with a boolean
    obj = True
    result = serializer.dumps(obj)
    assert result == "true"

    # Test with None
    obj = None
    result = serializer.dumps(obj)
    assert result == "null"


# LLM-generated content at query #21
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert result == '{"key": "value"}'


# LLM-generated content at query #22
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer(_PDataSerializer[str]):
        def loads(self, payload: str, /) -> t.Any:
            return json.loads(payload)

        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    data = {"key": "value"}
    dumped = serializer.dumps(data)
    assert dumped == '{"key": "value"}'
    assert isinstance(dumped, str)


# LLM-generated content at query #23
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer(_PDataSerializer[str]):
        def loads(self, payload: str, /) -> t.Any:
            return json.loads(payload)

        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    data = {"key": "value"}
    dumped = serializer.dumps(data)
    assert dumped == '{"key": "value"}'
    assert isinstance(dumped, str)


# LLM-generated content at query #24
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer(_PDataSerializer[str]):
        def loads(self, payload: str, /) -> t.Any:
            return json.loads(payload)

        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    data = {"key": "value"}
    dumped = serializer.dumps(data)
    assert dumped == '{"key": "value"}'
    assert isinstance(dumped, str)


# LLM-generated content at query #25
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a mock serializer that returns str
    class MockStrSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = MockStrSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"

    # Test with a mock serializer that returns bytes
    class MockBytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

    serializer = MockBytesSerializer()
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"


# LLM-generated content at query #26
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def loads(self, payload):
            return payload

        def dumps(self, obj):
            return obj

    serializer = TestSerializer()
    assert serializer.dumps({"key": "value"}) == {"key": "value"}


# LLM-generated content at query #27
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def loads(self, payload):
            return payload

        def dumps(self, obj):
            return obj

    serializer = TestSerializer()
    assert serializer.dumps({"key": "value"}) == {"key": "value"}


# LLM-generated content at query #28
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text)
    serializer_text = Serializer("secret-key", serializer=json)
    payload_text = b'{"key": "value"}'
    assert serializer_text.load_payload(payload_text) == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            return payload

    serializer_custom_text = Serializer("secret-key", serializer=CustomTextSerializer())
    payload_custom_text = b"custom_payload"
    assert serializer_custom_text.load_payload(payload_custom_text) == "custom_payload"

    # Test with bytes serializer
    class CustomBytesSerializer:
        def dumps(self, obj):
            return obj.encode() if isinstance(obj, str) else obj

        def loads(self, payload):
            return payload.decode() if isinstance(payload, bytes) else payload

    serializer_bytes = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload_bytes = b'{"key": "value"}'
    assert serializer_bytes.load_payload(payload_bytes) == '{"key": "value"}'

    # Test with BadPayload exception
    class FailingSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            raise ValueError("Deserialization failed")

    serializer_fail = Serializer("secret-key", serializer=FailingSerializer())
    payload_fail = b"invalid_payload"

    with pytest.raises(BadPayload) as exc_info:
        serializer_fail.load_payload(payload_fail)

    assert "Could not load the payload" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, ValueError)


# LLM-generated content at query #29
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return f"custom_{payload}"

        def dumps(self, obj: t.Any, /) -> str:
            return f"{obj}_custom"

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"custom_data"
    assert serializer.load_payload(payload) == "custom_custom_data"

    # Test with bytes serializer
    class CustomBytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return f"bytes_{payload.decode()}"

        def dumps(self, obj: t.Any, /) -> bytes:
            return f"{obj}_bytes".encode()

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b"bytes_data"
    assert serializer.load_payload(payload) == "bytes_bytes_data"

    # Test with invalid payload (should raise BadPayload)
    serializer = Serializer("secret-key")
    payload = b"invalid_json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass

    # Test with custom serializer that raises exception
    class FailingSerializer:
        def loads(self, payload: str, /) -> t.Any:
            raise ValueError("Custom error")

        def dumps(self, obj: t.Any, /) -> str:
            return str(obj)

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b'{"key": "value"}'
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Custom error" in str(e.original_error)


# LLM-generated content at query #30
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json)
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert serializer.loads(signed_data) == data

    # Test with bytes serializer
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode('utf-8')

        @staticmethod
        def loads(data):
            return json.loads(data.decode('utf-8'))

    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    signed_bytes = bytes_serializer.dumps(data)
    assert isinstance(signed_bytes, bytes)
    assert bytes_serializer.loads(signed_bytes) == data

    # Test with custom salt
    custom_salt_serializer = Serializer("secret-key", salt="custom-salt")
    signed_custom = custom_salt_serializer.dumps(data)
    assert custom_salt_serializer.loads(signed_custom) == data

    # Test with key rotation
    keys = ["old-key", "new-key"]
    key_rotation_serializer = Serializer(keys)
    signed_rotated = key_rotation_serializer.dumps(data)
    assert key_rotation_serializer.loads(signed_rotated) == data

    # Test with different data types
    test_data = [
        None,
        123,
        45.67,
        "string",
        [1, 2, 3],
        {"a": 1, "b": 2},
        True,
        False
    ]
    for item in test_data:
        signed_item = serializer.dumps(item)
        assert serializer.loads(signed_item) == item

    # Test that different data produces different signatures
    data1 = {"key": "value1"}
    data2 = {"key": "value2"}
    signed1 = serializer.dumps(data1)
    signed2 = serializer.dumps(data2)
    assert signed1 != signed2


# LLM-generated content at query #31
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert result == '{"key": "value"}'


# LLM-generated content at query #32
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert result == '{"key": "value"}'


# LLM-generated content at query #33
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns str
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = TextSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"

    # Test with a serializer that returns bytes
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"


# LLM-generated content at query #34
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            return f"loaded-{payload}"

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"test-payload"
    assert serializer.load_payload(payload) == "loaded-test-payload"

    # Test with custom bytes serializer
    class CustomBytesSerializer:
        def dumps(self, obj):
            return bytes(obj, 'utf-8')

        def loads(self, payload):
            return f"loaded-{payload.decode('utf-8')}"

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b"test-payload"
    assert serializer.load_payload(payload) == "loaded-test-payload"

    # Test with BadPayload exception
    class FailingSerializer:
        def dumps(self, obj):
            return b""

        def loads(self, payload):
            raise ValueError("Test error")

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b"test-payload"
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(payload)
    assert "Could not load the payload" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, ValueError)


# LLM-generated content at query #35
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def loads(self, payload):
            return f"custom_{payload}"

        def dumps(self, obj):
            return str(obj)

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"test_payload"
    assert serializer.load_payload(payload) == "custom_test_payload"

    # Test with custom bytes serializer
    class CustomBytesSerializer:
        def loads(self, payload):
            return f"custom_{payload.decode()}"

        def dumps(self, obj):
            return str(obj).encode()

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b"test_payload"
    assert serializer.load_payload(payload) == "custom_test_payload"

    # Test with invalid payload
    serializer = Serializer("secret-key")
    payload = b"invalid_json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #36
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json)
    s = Serializer("secret-key")
    data = {"key": "value"}
    signed = s.dumps(data)
    assert isinstance(signed, str)
    assert s.loads(signed) == data

    # Test with custom serializer (bytes)
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode()

        def loads(self, payload):
            return int(payload.decode())

    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    data_bytes = 42
    signed_bytes = s_bytes.dumps(data_bytes)
    assert isinstance(signed_bytes, bytes)
    assert s_bytes.loads(signed_bytes) == data_bytes

    # Test with salt
    s_salt = Serializer("secret-key", salt="custom-salt")
    signed_salt = s_salt.dumps(data)
    assert s_salt.loads(signed_salt) == data

    # Test with key rotation
    s_rotated = Serializer(["old-key", "new-key"])
    signed_rotated = s_rotated.dumps(data)
    assert s_rotated.loads(signed_rotated) == data

    # Test with custom signer
    class CustomSigner(Signer):
        pass

    s_custom = Serializer("secret-key", signer=CustomSigner)
    signed_custom = s_custom.dumps(data)
    assert s_custom.loads(signed_custom) == data


# LLM-generated content at query #37
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert result == '{"key": "value"}'


# LLM-generated content at query #38
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str, /) -> str:
            return payload.upper()

        def dumps(self, obj: str, /) -> str:
            return obj.lower()

    serializer = StringSerializer()
    assert serializer.loads("hello") == "HELLO"

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return payload[::-1]

        def dumps(self, obj: bytes, /) -> bytes:
            return obj[::-1]

    serializer = BytesSerializer()
    assert serializer.loads(b"hello") == b"olleh"

    # Test with a serializer that handles both str and bytes
    class MixedSerializer:
        def loads(self, payload: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return payload.upper()
            return payload[::-1]

        def dumps(self, obj: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return obj.lower()
            return obj[::-1]

    serializer = MixedSerializer()
    assert serializer.loads("hello") == "HELLO"
    assert serializer.loads(b"hello") == b"olleh"


# LLM-generated content at query #39
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with default JSON serializer
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    loaded_data = serializer.loads(signed_data)
    assert loaded_data == data

    # Test with custom text serializer
    class CustomTextSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            return payload

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    data = "test-data"
    signed_data = serializer.dumps(data)
    loaded_data = serializer.loads(signed_data)
    assert loaded_data == data

    # Test with custom bytes serializer
    class CustomBytesSerializer:
        def dumps(self, obj):
            return bytes(obj, 'utf-8')

        def loads(self, payload):
            return payload.decode('utf-8')

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    data = "test-data"
    signed_data = serializer.dumps(data)
    loaded_data = serializer.loads(signed_data)
    assert loaded_data == data

    # Test with invalid signature
    serializer = Serializer("secret-key")
    try:
        serializer.loads("invalid-signature")
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

    # Test with invalid payload
    serializer = Serializer("secret-key")
    try:
        serializer.loads("valid-signature-but-invalid-payload")
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass


# LLM-generated content at query #40
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple dictionary
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert "key" in signed_data

    # Test with a list
    data = [1, 2, 3]
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert "1" in signed_data

    # Test with a string
    data = "test string"
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert "test string" in signed_data

    # Test with a custom salt
    signed_data = serializer.dumps(data, salt="custom-salt")
    assert isinstance(signed_data, str)
    assert "test string" in signed_data

    # Test with a bytes serializer
    bytes_serializer = Serializer("secret-key", serializer=json)
    data = {"key": "value"}
    signed_data = bytes_serializer.dumps(data)
    assert isinstance(signed_data, bytes)
    assert b"key" in signed_data


# LLM-generated content at query #41
#--------------------------

```python
def test__PDataSerializer_loads():
    serializer = json
    payload = '{"key": "value"}'
    assert serializer.loads(payload) == {"key": "value"}

    payload = b'{"key": "value"}'
    assert serializer.loads(payload) == {"key": "value"}

    payload = '{"key": "value"}'
    assert serializer.loads(payload) == {"key": "value"}

    payload = b'{"key": "value"}'
    assert serializer.loads(payload) == {"key": "value"}


# LLM-generated content at query #42
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a text serializer (JSON)
    text_serializer = json
    data = {"key": "value"}
    serialized = text_serializer.dumps(data)
    assert isinstance(serialized, str)
    deserialized = text_serializer.loads(serialized)
    assert deserialized == data

    # Test with a bytes serializer (custom)
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode('utf-8')

        def loads(self, payload):
            return json.loads(payload.decode('utf-8'))

    bytes_serializer = BytesSerializer()
    serialized_bytes = bytes_serializer.dumps(data)
    assert isinstance(serialized_bytes, bytes)
    deserialized_bytes = bytes_serializer.loads(serialized_bytes)
    assert deserialized_bytes == data

    # Test with a custom serializer that doesn't match the strict protocol
    class CustomSerializer:
        def dumps(self, obj, **kwargs):
            return str(obj).encode('utf-8')

        def loads(self, payload, **kwargs):
            return int(payload.decode('utf-8'))

    custom_serializer = CustomSerializer()
    serialized_custom = custom_serializer.dumps(42)
    assert isinstance(serialized_custom, bytes)
    deserialized_custom = custom_serializer.loads(serialized_custom)
    assert deserialized_custom == 42


# LLM-generated content at query #43
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns str
    class StrSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = StrSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"

    # Test with a serializer that returns bytes
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"


# LLM-generated content at query #44
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer (str)
    class TextSerializer:
        def loads(self, payload: str, /) -> str:
            return payload.upper()

        def dumps(self, obj: str, /) -> str:
            return obj.lower()

    serializer = TextSerializer()
    assert serializer.loads("hello") == "HELLO"

    # Test with a simple bytes serializer (bytes)
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return payload[::-1]  # reverse the bytes

        def dumps(self, obj: bytes, /) -> bytes:
            return obj[::-1]  # reverse the bytes

    serializer = BytesSerializer()
    assert serializer.loads(b"hello") == b"olleh"

    # Test with a serializer that handles both str and bytes (union)
    class UnionSerializer:
        def loads(self, payload: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return payload.upper()
            return payload[::-1]

        def dumps(self, obj: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return obj.lower()
            return obj[::-1]

    serializer = UnionSerializer()
    assert serializer.loads("hello") == "HELLO"
    assert serializer.loads(b"hello") == b"olleh"


# LLM-generated content at query #45
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple dictionary
    serializer = json
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a list
    data = [1, 2, 3]
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with None
    data = None
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) is None

    # Test with a string
    data = "test string"
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a number
    data = 42
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data


