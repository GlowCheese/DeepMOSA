####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
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
            return str(obj).encode()

        @staticmethod
        def loads(data):
            return data.decode()

    serializer = Serializer("secret-key", serializer=BytesSerializer())
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, bytes)
    assert serializer.loads(signed_data) == data

    # Test with custom salt
    serializer = Serializer("secret-key", salt="custom-salt")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert serializer.loads(signed_data) == data

    # Test with different secret keys
    serializer = Serializer(["old-key", "new-key"])
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert serializer.loads(signed_data) == data

    # Test with custom signer
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert serializer.loads(signed_data) == data


# LLM-generated content at query #2
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple string serializer
    class StringSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            return payload

    serializer = StringSerializer()
    assert serializer.dumps("test") == "test"
    assert serializer.dumps(123) == "123"

    # Test with a bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

        def loads(self, payload):
            return payload.decode('utf-8')

    bytes_serializer = BytesSerializer()
    assert bytes_serializer.dumps("test") == b"test"
    assert bytes_serializer.dumps(123) == b"123"

    # Test with a JSON serializer
    json_serializer = json
    assert json_serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert json_serializer.dumps([1, 2, 3]) == "[1, 2, 3]"

    # Test with a custom object
    class CustomObject:
        def __init__(self, value):
            self.value = value

    class CustomSerializer:
        def dumps(self, obj):
            if isinstance(obj, CustomObject):
                return f"CustomObject({obj.value})"
            return str(obj)

        def loads(self, payload):
            if payload.startswith("CustomObject(") and payload.endswith(")"):
                value = payload[len("CustomObject("):-1]
                return CustomObject(value)
            return payload

    custom_serializer = CustomSerializer()
    obj = CustomObject("test_value")
    assert custom_serializer.dumps(obj) == "CustomObject(test_value)"
    assert custom_serializer.loads("CustomObject(test_value)").value == "test_value"


# LLM-generated content at query #3
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
    class CustomSerializer:
        def loads(self, payload: str | bytes, **kwargs) -> t.Any:
            return f"loaded: {payload}"

        def dumps(self, obj: t.Any, **kwargs) -> str | bytes:
            return f"dumped: {obj}"

    serializer = CustomSerializer()
    assert serializer.loads("test") == "loaded: test"
    assert serializer.loads(b"test") == "loaded: b'test'"


# LLM-generated content at query #4
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


# LLM-generated content at query #5
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple string serializer
    class StringSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            return payload

    serializer = StringSerializer()
    assert serializer.dumps("test") == "test"
    assert serializer.dumps(123) == "123"

    # Test with a bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

        def loads(self, payload):
            return payload.decode('utf-8')

    bytes_serializer = BytesSerializer()
    assert bytes_serializer.dumps("test") == b"test"
    assert bytes_serializer.dumps(123) == b"123"

    # Test with a JSON serializer
    assert json.dumps({"key": "value"}) == '{"key": "value"}'
    assert json.dumps([1, 2, 3]) == "[1, 2, 3]"

    # Test with a custom object
    class CustomObject:
        def __init__(self, value):
            self.value = value

    class CustomSerializer:
        def dumps(self, obj):
            if isinstance(obj, CustomObject):
                return f"CustomObject({obj.value})"
            return str(obj)

        def loads(self, payload):
            if payload.startswith("CustomObject(") and payload.endswith(")"):
                value = payload[len("CustomObject("):-1]
                return CustomObject(value)
            return payload

    custom_serializer = CustomSerializer()
    obj = CustomObject("test")
    assert custom_serializer.dumps(obj) == "CustomObject(test)"
    assert custom_serializer.dumps(123) == "123"


# LLM-generated content at query #6
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple text serializer
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            return payload

    serializer = TextSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

        def loads(self, payload):
            return payload.decode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == b"[1, 2, 3]"

    # Test with a more complex object
    class ComplexSerializer:
        def dumps(self, obj):
            if isinstance(obj, dict):
                return json.dumps(obj)
            elif isinstance(obj, list):
                return json.dumps(obj)
            else:
                return str(obj)

        def loads(self, payload):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return payload

    serializer = ComplexSerializer()
    assert serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert serializer.dumps([1, 2, 3]) == '[1, 2, 3]'
    assert serializer.dumps("simple string") == "simple string"


# LLM-generated content at query #7
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text)
    s_text = Serializer("secret-key", salt="test-salt")
    data = {"key": "value"}
    signed = s_text.dumps(data)
    assert isinstance(signed, str)
    assert s_text.loads(signed) == data

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")

        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))

    s_bytes = Serializer("secret-key", salt="test-salt", serializer=BytesSerializer())
    signed_bytes = s_bytes.dumps(data)
    assert isinstance(signed_bytes, bytes)
    assert s_bytes.loads(signed_bytes) == data

    # Test with custom salt
    custom_salt = "custom-salt"
    s_custom = Serializer("secret-key", salt=custom_salt)
    signed_custom = s_custom.dumps(data)
    assert s_custom.loads(signed_custom) == data

    # Test with key rotation
    keys = ["old-key", "new-key"]
    s_rotated = Serializer(keys, salt="rotation-salt")
    signed_rotated = s_rotated.dumps(data)
    assert s_rotated.loads(signed_rotated) == data

    # Test with different data types
    test_cases = [
        None,
        123,
        45.67,
        "string",
        [1, 2, 3],
        {"a": 1, "b": 2},
        True,
        False,
    ]
    for case in test_cases:
        signed_case = s_text.dumps(case)
        assert s_text.loads(signed_case) == case


# LLM-generated content at query #8
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default signer and no fallback signers
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
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(Signer, {"digest_method": "sha256"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].digest_method == Signer.digest_method
    assert signers[1].digest_method == "sha256"

    # Test with fallback signers as Signer class
    serializer = Serializer(
        "secret-key",
        fallback_signers=[Signer]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert all(isinstance(s, Signer) for s in signers)

    # Test with key rotation (multiple secret keys)
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-key", b"new-key"]

    # Test with key rotation and fallback signers
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_keys == [b"old-key"]
    assert signers[1].digest_method == "sha256"


# LLM-generated content at query #9
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer (str)
    class TextSerializer:
        def loads(self, payload: str, /) -> str:
            return f"loaded_{payload}"

        def dumps(self, obj: str, /) -> str:
            return f"dumped_{obj}"

    serializer = TextSerializer()
    assert serializer.loads("test") == "loaded_test"

    # Test with a simple bytes serializer (bytes)
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return b"loaded_" + payload

        def dumps(self, obj: bytes, /) -> bytes:
            return b"dumped_" + obj

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == b"loaded_test"

    # Test with a serializer that handles both str and bytes
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


# LLM-generated content at query #10
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


# LLM-generated content at query #11
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (JSON)
    serializer = Serializer("secret-key", salt="test-salt")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert "key" in signed_data
    assert "value" in signed_data

    # Test with bytes serializer
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode("utf-8")

        @staticmethod
        def loads(data):
            return data.decode("utf-8")

    serializer_bytes = Serializer("secret-key", salt="test-salt", serializer=BytesSerializer())
    signed_data_bytes = serializer_bytes.dumps(data)
    assert isinstance(signed_data_bytes, bytes)
    assert b"key" in signed_data_bytes
    assert b"value" in signed_data_bytes

    # Test with custom salt
    custom_salt_serializer = Serializer("secret-key", salt="custom-salt")
    signed_data_custom_salt = custom_salt_serializer.dumps(data)
    assert isinstance(signed_data_custom_salt, str)
    assert signed_data_custom_salt != signed_data

    # Test with key rotation
    serializer_rotated = Serializer(["old-key", "new-key"], salt="test-salt")
    signed_data_rotated = serializer_rotated.dumps(data)
    assert isinstance(signed_data_rotated, str)
    assert "key" in signed_data_rotated
    assert "value" in signed_data_rotated


# LLM-generated content at query #12
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json) and text output
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert serializer.loads(signed_data) == data

    # Test with custom serializer that outputs bytes
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
    custom_salt_serializer = Serializer("secret-key", salt="custom-salt")
    signed_custom = custom_salt_serializer.dumps(data)
    assert custom_salt_serializer.loads(signed_custom) == data

    # Test with key rotation
    keys = ["old-key", "new-key"]
    rotator = Serializer(keys)
    signed_rotated = rotator.dumps(data)
    assert rotator.loads(signed_rotated) == data

    # Test with different data types
    test_cases = [
        None,
        123,
        45.67,
        "string",
        [1, 2, 3],
        {"a": 1, "b": 2},
        True,
        False
    ]
    for case in test_cases:
        signed = serializer.dumps(case)
        assert serializer.loads(signed) == case

    # Test that different data produces different signatures
    data1 = {"data": 1}
    data2 = {"data": 2}
    assert serializer.dumps(data1) != serializer.dumps(data2)


# LLM-generated content at query #13
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple text serializer
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            return payload

    serializer = TextSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"

    # Test with a bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

        def loads(self, payload):
            return payload.decode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == b"[1, 2, 3]"


# LLM-generated content at query #14
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json) and text output
    s = Serializer("secret-key")
    data = {"key": "value"}
    signed = s.dumps(data)
    assert isinstance(signed, str)
    assert s.loads(signed) == data

    # Test with bytes output
    s_bytes = Serializer("secret-key", serializer=json, salt=b"salt")
    signed_bytes = s_bytes.dumps(data)
    assert isinstance(signed_bytes, bytes)
    assert s_bytes.loads(signed_bytes) == data

    # Test with custom salt
    s_salt = Serializer("secret-key", salt="custom-salt")
    signed_salt = s_salt.dumps(data)
    assert isinstance(signed_salt, str)
    assert s_salt.loads(signed_salt) == data

    # Test with key rotation
    s_rotated = Serializer(["old-key", "new-key"])
    signed_rotated = s_rotated.dumps(data)
    assert isinstance(signed_rotated, str)
    assert s_rotated.loads(signed_rotated) == data

    # Test with custom serializer (bytes)
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(data):
            return json.loads(data.decode("utf-8"))

    s_custom_bytes = Serializer("secret-key", serializer=CustomBytesSerializer())
    signed_custom_bytes = s_custom_bytes.dumps(data)
    assert isinstance(signed_custom_bytes, bytes)
    assert s_custom_bytes.loads(signed_custom_bytes) == data


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json)
    s = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = s.dumps(data)
    assert isinstance(signed_data, str)
    assert s.loads(signed_data) == data

    # Test with custom serializer (bytes)
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode('utf-8')

        @staticmethod
        def loads(data):
            return int(data.decode('utf-8'))

    s_bytes = Serializer("secret-key", serializer=CustomBytesSerializer())
    data_bytes = 42
    signed_data_bytes = s_bytes.dumps(data_bytes)
    assert isinstance(signed_data_bytes, bytes)
    assert s_bytes.loads(signed_data_bytes) == data_bytes

    # Test with custom serializer (str)
    class CustomStrSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(data):
            return float(data)

    s_str = Serializer("secret-key", serializer=CustomStrSerializer())
    data_str = 3.14
    signed_data_str = s_str.dumps(data_str)
    assert isinstance(signed_data_str, str)
    assert s_str.loads(signed_data_str) == data_str

    # Test with salt parameter
    s_salt = Serializer("secret-key", salt="custom-salt")
    signed_data_salt = s_salt.dumps(data)
    assert isinstance(signed_data_salt, str)
    assert s_salt.loads(signed_data_salt) == data

    # Test with key rotation
    s_rotated = Serializer(["old-key", "new-key"])
    signed_data_rotated = s_rotated.dumps(data)
    assert isinstance(signed_data_rotated, str)
    assert s_rotated.loads(signed_data_rotated) == data


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


# LLM-generated content at query #18
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

    # Test with a serializer that doesn't match the strict protocol
    class NonStrictSerializer:
        def loads(self, payload: str, extra_arg=None) -> str:
            return f"loaded_{payload}"

        def dumps(self, obj: str, extra_arg=None) -> str:
            return f"dumped_{obj}"

    serializer = NonStrictSerializer()
    assert serializer.loads("test") == "loaded_test"


# LLM-generated content at query #19
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with valid JSON data
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    loaded_data = serializer.loads(signed_data)
    assert loaded_data == data

    # Test with invalid signature
    invalid_signed_data = signed_data[:-1] + b"x"
    try:
        serializer.loads(invalid_signed_data)
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

    # Test with corrupted payload
    corrupted_data = b"corrupted"
    try:
        serializer.loads(corrupted_data)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass

    # Test with custom serializer
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()

        @staticmethod
        def loads(data):
            return int(data.decode())

    custom_serializer = Serializer("secret-key", serializer=CustomSerializer())
    signed_int = custom_serializer.dumps(42)
    loaded_int = custom_serializer.loads(signed_int)
    assert loaded_int == 42

    # Test with key rotation
    keys = ["old-key", "new-key"]
    serializer_rotated = Serializer(keys)
    signed_with_old = serializer_rotated.dumps(data)
    loaded_with_new = serializer_rotated.loads(signed_with_old)
    assert loaded_with_new == data


# LLM-generated content at query #20
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns str
    class StrSerializer:
        def loads(self, payload: str, /) -> str:
            return f"loaded_{payload}"

        def dumps(self, obj: str, /) -> str:
            return f"dumped_{obj}"

    serializer = StrSerializer()
    assert serializer.loads("test") == "loaded_test"

    # Test with a simple serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return b"loaded_" + payload

        def dumps(self, obj: bytes, /) -> bytes:
            return b"dumped_" + obj

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == b"loaded_test"

    # Test with a serializer that raises an exception
    class ExceptionSerializer:
        def loads(self, payload: str, /) -> str:
            raise ValueError("Test exception")

        def dumps(self, obj: str, /) -> str:
            return f"dumped_{obj}"

    serializer = ExceptionSerializer()
    try:
        serializer.loads("test")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Test exception"


# LLM-generated content at query #21
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple dictionary
    data = {"key": "value"}
    serializer = json
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a list
    data = [1, 2, 3]
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a string
    data = "test string"
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with an integer
    data = 42
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a float
    data = 3.14
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a boolean
    data = True
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with None
    data = None
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) is None


# LLM-generated content at query #22
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer
    class TextSerializer:
        def loads(self, payload: str, /) -> str:
            return payload.upper()

        def dumps(self, obj: str, /) -> str:
            return obj.lower()

    serializer = TextSerializer()
    assert serializer.loads("hello") == "HELLO"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return payload[::-1]

        def dumps(self, obj: bytes, /) -> bytes:
            return obj[::-1]

    serializer = BytesSerializer()
    assert serializer.loads(b"hello") == b"olleh"

    # Test with a serializer that returns a different type
    class JsonSerializer:
        def loads(self, payload: str, /) -> dict:
            return json.loads(payload)

        def dumps(self, obj: dict, /) -> str:
            return json.dumps(obj)

    serializer = JsonSerializer()
    assert serializer.loads('{"key": "value"}') == {"key": "value"}

    # Test with a serializer that raises an exception
    class ExceptionSerializer:
        def loads(self, payload: str, /) -> str:
            raise ValueError("Test exception")

        def dumps(self, obj: str, /) -> str:
            return obj

    serializer = ExceptionSerializer()
    try:
        serializer.loads("test")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #23
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
    assert isinstance(dumped, str)
    assert json.loads(dumped) == data


# LLM-generated content at query #25
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

    # Test with a custom serializer that returns a different type
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()

        def dumps(self, obj: t.Any) -> str:
            return str(obj).lower()

    custom_serializer = CustomSerializer()
    assert custom_serializer.loads("hello") == "HELLO"


# LLM-generated content at query #26
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple text serializer
    class TextSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return payload

        def dumps(self, obj: t.Any, /) -> str:
            return str(obj)

    serializer = TextSerializer()
    assert serializer.dumps("test") == "test"
    assert serializer.dumps(123) == "123"

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return payload

        def dumps(self, obj: t.Any, /) -> bytes:
            return str(obj).encode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps("test") == b"test"
    assert serializer.dumps(123) == b"123"

    # Test with a custom object
    class CustomObject:
        def __str__(self):
            return "custom"

    serializer = TextSerializer()
    assert serializer.dumps(CustomObject()) == "custom"

    serializer = BytesSerializer()
    assert serializer.dumps(CustomObject()) == b"custom"


# LLM-generated content at query #27
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


# LLM-generated content at query #28
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple dictionary
    serializer = json
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    assert serializer.loads(serialized) == data

    # Test with a list
    data = [1, 2, 3]
    serialized = serializer.dumps(data)
    assert serializer.loads(serialized) == data

    # Test with a string
    data = "test string"
    serialized = serializer.dumps(data)
    assert serializer.loads(serialized) == data

    # Test with an integer
    data = 42
    serialized = serializer.dumps(data)
    assert serializer.loads(serialized) == data

    # Test with a float
    data = 3.14
    serialized = serializer.dumps(data)
    assert serializer.loads(serialized) == data

    # Test with a boolean
    data = True
    serialized = serializer.dumps(data)
    assert serializer.loads(serialized) == data

    # Test with None
    data = None
    serialized = serializer.dumps(data)
    assert serializer.loads(serialized) == data


# LLM-generated content at query #29
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON serializer
    class JSONSerializer:
        def loads(self, payload: str | bytes, /) -> t.Any:
            return json.loads(payload)

        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    serializer = JSONSerializer()
    assert serializer.loads('{"key": "value"}') == {"key": "value"}

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return pickle.loads(payload)

        def dumps(self, obj: t.Any, /) -> bytes:
            return pickle.dumps(obj)

    serializer = BytesSerializer()
    data = {"key": "value"}
    dumped = serializer.dumps(data)
    assert serializer.loads(dumped) == data


# LLM-generated content at query #30
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

    # Test with a string
    data = "test string"
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with an integer
    data = 42
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a boolean
    data = True
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with None
    data = None
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) is None


# LLM-generated content at query #31
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str, /) -> str:
            return payload

        def dumps(self, obj: str, /) -> str:
            return obj

    serializer = StringSerializer()
    assert serializer.loads("test") == "test"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return payload

        def dumps(self, obj: bytes, /) -> bytes:
            return obj

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == b"test"

    # Test with a JSON serializer
    assert json.loads('{"key": "value"}') == {"key": "value"}
    assert json.loads(b'{"key": "value"}') == {"key": "value"}


# LLM-generated content at query #32
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = TestSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"


# LLM-generated content at query #33
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


# LLM-generated content at query #34
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple dictionary
    data = {"key": "value"}
    serializer = json
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a list
    data = [1, 2, 3]
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a string
    data = "test string"
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with an integer
    data = 42
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a float
    data = 3.14
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a boolean
    data = True
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with None
    data = None
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) is None


# LLM-generated content at query #35
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple string serializer
    class StringSerializer:
        def dumps(self, obj: t.Any, /) -> str:
            return str(obj)

    serializer = StringSerializer()
    assert serializer.dumps("test") == "test"
    assert serializer.dumps(123) == "123"
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"

    # Test with a bytes serializer
    class BytesSerializer:
        def dumps(self, obj: t.Any, /) -> bytes:
            return str(obj).encode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps("test") == b"test"
    assert serializer.dumps(123) == b"123"
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"


# LLM-generated content at query #36
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a valid payload
    serializer = json
    payload = '{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}

    # Test with an invalid payload
    invalid_payload = '{"key": "value"'
    try:
        serializer.loads(invalid_payload)
        assert False, "Expected JSONDecodeError"
    except json.JSONDecodeError:
        pass

    # Test with a different serializer (e.g., a custom one)
    class CustomSerializer:
        @staticmethod
        def loads(data):
            return data.upper()

    custom_serializer = CustomSerializer()
    custom_payload = "test"
    custom_result = custom_serializer.loads(custom_payload)
    assert custom_result == "TEST"


# LLM-generated content at query #37
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

    # Test with key rotation
    serializer = Serializer(["key1", "key2", "key3"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"key1", b"key2", b"key3"]

    # Test with fallback signers (dict)
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[1].digest_method.name == "sha256"

    # Test with fallback signers (Signer class)
    serializer = Serializer(
        "secret-key",
        fallback_signers=[Signer]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], Signer)

    # Test with fallback signers (tuple)
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(Signer, {"digest_method": "sha256"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[1].digest_method.name == "sha256"

    # Test with multiple fallback signers
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"digest_method": "sha256"},
            Signer,
            (Signer, {"digest_method": "sha512"})
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    assert signers[1].digest_method.name == "sha256"
    assert isinstance(signers[2], Signer)
    assert signers[3].digest_method.name == "sha512"

    # Test with custom salt in iter_unsigners
    serializer = Serializer("secret-key", salt="default-salt")
    signers = list(serializer.iter_unsigners(salt="custom-salt"))
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"


# LLM-generated content at query #38
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON serializer
    serializer = json
    payload = '{"key": "value"}'
    expected = {"key": "value"}
    assert serializer.loads(payload) == expected

    # Test with a custom serializer that returns bytes
    class CustomBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode('utf-8').upper()

        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode('utf-8')

    custom_serializer = CustomBytesSerializer()
    payload_bytes = b'hello'
    expected_bytes = 'HELLO'
    assert custom_serializer.loads(payload_bytes) == expected_bytes

    # Test with a custom serializer that returns str
    class CustomStrSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.split(',')

        def dumps(self, obj: t.Any) -> str:
            return ','.join(obj)

    custom_str_serializer = CustomStrSerializer()
    payload_str = 'a,b,c'
    expected_str = ['a', 'b', 'c']
    assert custom_str_serializer.loads(payload_str) == expected_str


# LLM-generated content at query #39
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

    # Test with a string
    data = "test string"
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with an integer
    data = 42
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a float
    data = 3.14
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a boolean
    data = True
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with None
    data = None
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data


# LLM-generated content at query #40
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with default JSON serializer
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert serializer.loads(signed_data) == data

    # Test with custom text serializer
    class CustomTextSerializer:
        def dumps(self, obj):
            return f"custom-{json.dumps(obj)}"

        def loads(self, payload):
            return json.loads(payload[7:])

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert serializer.loads(signed_data) == data

    # Test with bytes serializer
    class CustomBytesSerializer:
        def dumps(self, obj):
            return f"custom-{json.dumps(obj)}".encode("utf-8")

        def loads(self, payload):
            return json.loads(payload[7:].decode("utf-8"))

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, bytes)
    assert serializer.loads(signed_data) == data

    # Test with different data types
    data_list = [1, 2, 3]
    signed_data = serializer.dumps(data_list)
    assert isinstance(signed_data, bytes)
    assert serializer.loads(signed_data) == data_list

    data_str = "test string"
    signed_data = serializer.dumps(data_str)
    assert isinstance(signed_data, bytes)
    assert serializer.loads(signed_data) == data_str

    # Test with salt
    serializer = Serializer("secret-key", salt="custom-salt")
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, bytes)
    assert serializer.loads(signed_data) == data


# LLM-generated content at query #41
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple text serializer
    class TextSerializer:
        def dumps(self, obj):
            return json.dumps(obj)

        def loads(self, payload):
            return json.loads(payload)

    serializer = TextSerializer()
    assert serializer.dumps({"key": "value"}) == '{"key": "value"}'

    # Test with a simple bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode('utf-8')

        def loads(self, payload):
            return json.loads(payload.decode('utf-8'))

    serializer = BytesSerializer()
    assert serializer.dumps({"key": "value"}) == b'{"key": "value"}'

    # Test with a custom object
    class CustomObject:
        def __init__(self, value):
            self.value = value

    class CustomSerializer:
        def dumps(self, obj):
            if isinstance(obj, CustomObject):
                return f"CustomObject({obj.value})"
            return str(obj)

        def loads(self, payload):
            if payload.startswith("CustomObject(") and payload.endswith(")"):
                value = payload[len("CustomObject("):-1]
                return CustomObject(value)
            return payload

    serializer = CustomSerializer()
    obj = CustomObject("test")
    assert serializer.dumps(obj) == "CustomObject(test)"


# LLM-generated content at query #42
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns str
    class StrSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = StrSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"

    # Test with a simple serializer that returns bytes
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == b"[1, 2, 3]"

    # Test with a more complex object
    class ComplexSerializer:
        def dumps(self, obj):
            if isinstance(obj, dict):
                return json.dumps(obj)
            elif isinstance(obj, list):
                return json.dumps(obj)
            else:
                return str(obj)

    serializer = ComplexSerializer()
    assert serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert serializer.dumps([1, 2, 3]) == '[1, 2, 3]'
    assert serializer.dumps("test") == "test"


# LLM-generated content at query #43
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with default JSON serializer
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert serializer.loads(signed_data) == data

    # Test with custom text serializer
    class CustomTextSerializer:
        def dumps(self, obj):
            return f"custom-{json.dumps(obj)}"

        def loads(self, payload):
            return json.loads(payload[7:])

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert signed_data.startswith("custom-")
    assert serializer.loads(signed_data) == data

    # Test with bytes serializer
    class CustomBytesSerializer:
        def dumps(self, obj):
            return f"custom-{json.dumps(obj)}".encode("utf-8")

        def loads(self, payload):
            return json.loads(payload[7:].decode("utf-8"))

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, bytes)
    assert signed_data.startswith(b"custom-")
    assert serializer.loads(signed_data) == data

    # Test with key rotation
    serializer = Serializer(["old-key", "new-key"])
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert serializer.loads(signed_data) == data

    # Test with different salt
    serializer = Serializer("secret-key", salt="custom-salt")
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert serializer.loads(signed_data) == data


# LLM-generated content at query #44
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

    # Test with a mixed serializer (str or bytes)
    class MixedSerializer:
        def loads(self, payload: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return payload.upper()
            return payload[::-1]

        def dumps(self, obj: str | bytes, /) -> str | bytes:
            if isinstance(obj, str):
                return obj.lower()
            return obj[::-1]

    serializer = MixedSerializer()
    assert serializer.loads("hello") == "HELLO"
    assert serializer.loads(b"hello") == b"olleh"


# LLM-generated content at query #45
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple dictionary
    data = {"key": "value"}
    serializer = json
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a list
    data = [1, 2, 3]
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a string
    data = "test string"
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with an integer
    data = 42
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a boolean
    data = True
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with None
    data = None
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data


# LLM-generated content at query #46
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple text serializer
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, payload):
            return payload

    serializer = TextSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

        def loads(self, payload):
            return payload.decode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == b"[1, 2, 3]"


# LLM-generated content at query #47
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple string serializer
    class StringSerializer:
        def dumps(self, obj: t.Any, /) -> str:
            return str(obj)

        def loads(self, payload: str, /) -> t.Any:
            return payload

    serializer = StringSerializer()
    assert serializer.dumps("test") == "test"
    assert serializer.dumps(123) == "123"
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"

    # Test with a bytes serializer
    class BytesSerializer:
        def dumps(self, obj: t.Any, /) -> bytes:
            return str(obj).encode('utf-8')

        def loads(self, payload: bytes, /) -> t.Any:
            return payload.decode('utf-8')

    bytes_serializer = BytesSerializer()
    assert bytes_serializer.dumps("test") == b"test"
    assert bytes_serializer.dumps(123) == b"123"
    assert bytes_serializer.dumps({"key": "value"}) == b"{'key': 'value'}"


# LLM-generated content at query #48
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
    class DictSerializer:
        def loads(self, payload: str, /) -> dict:
            return json.loads(payload)

        def dumps(self, obj: dict, /) -> str:
            return json.dumps(obj)

    serializer = DictSerializer()
    assert serializer.loads('{"a": 1}') == {"a": 1}


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Serializer():
    # Test with default parameters
    s = Serializer("secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

    # Test with custom parameters
    s = Serializer(
        secret_key=["key1", "key2"],
        salt="custom-salt",
        serializer=json,
        serializer_kwargs={"indent": 2},
        signer=Signer,
        signer_kwargs={"sep": "|"},
        fallback_signers=[{"sep": ":"}]
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"custom-salt"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {"sep": "|"}
    assert s.fallback_signers == [{"sep": ":"}]
    assert s.serializer_kwargs == {"indent": 2}

    # Test with None salt
    s = Serializer("secret-key", salt=None)
    assert s.salt is None

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode("utf-8")

        def loads(self, payload):
            return int(payload.decode("utf-8"))

    s = Serializer("secret-key", serializer=BytesSerializer())
    assert s.is_text_serializer is False

    # Test with fallback signers as tuples
    s = Serializer(
        "secret-key",
        fallback_signers=[(Signer, {"sep": ":"})]
    )
    assert s.fallback_signers == [(Signer, {"sep": ":"})]

    # Test with fallback signers as Signer classes
    s = Serializer(
        "secret-key",
        fallback_signers=[Signer]
    )
    assert s.fallback_signers == [Signer]

    # Test with secret_key as bytes
    s = Serializer(b"secret-key")
    assert s.secret_keys == [b"secret-key"]

    # Test with secret_key as iterable of bytes
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

    # Test with secret_key as iterable of strings
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

    # Test with salt as bytes
    s = Serializer("secret-key", salt=b"custom-salt")
    assert s.salt == b"custom-salt"

    # Test with salt as string
    s = Serializer("secret-key", salt="custom-salt")
    assert s.salt == b"custom-salt"


# LLM-generated content at query #2
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default settings
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[0].salt == b"itsdangerous"

    # Test with custom salt
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert signers[0].salt == b"custom-salt"

    # Test with custom signer class
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    signers = list(serializer.iter_unsigners())
    assert isinstance(signers[0], CustomSigner)

    # Test with custom signer_kwargs
    serializer = Serializer("secret-key", signer_kwargs={"digest_method": "sha256"})
    signers = list(serializer.iter_unsigners())
    assert signers[0].digest_name == "sha256"

    # Test with fallback_signers as dict
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].digest_name == "sha512"  # default
    assert signers[1].digest_name == "sha256"

    # Test with fallback_signers as tuple
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(Signer, {"digest_method": "sha256"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[1].digest_name == "sha256"

    # Test with fallback_signers as Signer class
    serializer = Serializer(
        "secret-key",
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], CustomSigner)

    # Test with multiple fallback_signers
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"digest_method": "sha256"},
            (CustomSigner, {"digest_method": "sha384"})
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].digest_name == "sha512"
    assert signers[1].digest_name == "sha256"
    assert isinstance(signers[2], CustomSigner)
    assert signers[2].digest_name == "sha384"

    # Test with key rotation (multiple secret_keys)
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-key", b"new-key"]

    # Test with key rotation and fallback_signers
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3  # 1 default + 2 from fallback (one per key)
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_keys == [b"old-key"]
    assert signers[2].secret_keys == [b"new-key"]

    # Test with custom salt in iter_unsigners
    serializer = Serializer("secret-key", salt="default-salt")
    signers = list(serializer.iter_unsigners(salt="custom-salt"))
    assert signers[0].salt == b"custom-salt"


# LLM-generated content at query #3
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

    # Test with a boolean
    data = True
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with None
    data = None
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) is None


# LLM-generated content at query #4
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
            return str(obj).encode('utf-8')

        @staticmethod
        def loads(data):
            return data.decode('utf-8')

    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    signed_bytes = bytes_serializer.dumps(data)
    assert isinstance(signed_bytes, bytes)
    assert bytes_serializer.loads(signed_bytes) == str(data)

    # Test with custom salt
    custom_salt_serializer = Serializer("secret-key", salt="custom-salt")
    signed_custom = custom_salt_serializer.dumps(data)
    assert isinstance(signed_custom, str)
    assert custom_salt_serializer.loads(signed_custom) == data

    # Test with key rotation
    keys = ["old-key", "new-key"]
    key_rotation_serializer = Serializer(keys)
    signed_rotation = key_rotation_serializer.dumps(data)
    assert isinstance(signed_rotation, str)
    assert key_rotation_serializer.loads(signed_rotation) == data

    # Test with different data types
    test_cases = [
        None,
        123,
        45.67,
        "string",
        [1, 2, 3],
        {"a": 1, "b": 2},
        True,
        False,
    ]
    for test_data in test_cases:
        signed = serializer.dumps(test_data)
        assert serializer.loads(signed) == test_data


# LLM-generated content at query #5
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer (JSON)
    text_serializer = json
    data = {"key": "value"}
    serialized = text_serializer.dumps(data)
    assert text_serializer.loads(serialized) == data

    # Test with a simple bytes serializer (pickle-like)
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))

        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')

    bytes_serializer = BytesSerializer()
    data = {"key": "value"}
    serialized = bytes_serializer.dumps(data)
    assert bytes_serializer.loads(serialized) == data

    # Test with a custom serializer that doesn't match the strict protocol
    class CustomSerializer:
        def loads(self, payload: str | bytes, encoding: str = 'utf-8') -> t.Any:
            if isinstance(payload, bytes):
                payload = payload.decode(encoding)
            return json.loads(payload)

        def dumps(self, obj: t.Any, encoding: str = 'utf-8') -> str:
            return json.dumps(obj)

    custom_serializer = CustomSerializer()
    data = {"key": "value"}
    serialized = custom_serializer.dumps(data)
    assert custom_serializer.loads(serialized) == data


# LLM-generated content at query #6
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json) and text output
    serializer = Serializer("secret-key", salt="test-salt")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert "key" in signed_data
    assert "value" in signed_data

    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode("utf-8")

        def loads(self, payload):
            return eval(payload.decode("utf-8"))

    serializer_bytes = Serializer("secret-key", salt="test-salt", serializer=BytesSerializer())
    signed_data_bytes = serializer_bytes.dumps(data)
    assert isinstance(signed_data_bytes, bytes)
    assert b"key" in signed_data_bytes
    assert b"value" in signed_data_bytes

    # Test with custom salt
    serializer_custom_salt = Serializer("secret-key", salt="custom-salt")
    signed_data_custom_salt = serializer_custom_salt.dumps(data)
    assert isinstance(signed_data_custom_salt, str)
    assert "key" in signed_data_custom_salt
    assert "value" in signed_data_custom_salt

    # Test with key rotation
    serializer_key_rotation = Serializer(["old-key", "new-key"], salt="test-salt")
    signed_data_key_rotation = serializer_key_rotation.dumps(data)
    assert isinstance(signed_data_key_rotation, str)
    assert "key" in signed_data_key_rotation
    assert "value" in signed_data_key_rotation


# LLM-generated content at query #7
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer
    class TextSerializer:
        def loads(self, payload: str, /) -> str:
            return f"loaded_{payload}"

        def dumps(self, obj: str, /) -> str:
            return obj

    serializer = TextSerializer()
    assert serializer.loads("test") == "loaded_test"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return b"loaded_" + payload

        def dumps(self, obj: bytes, /) -> bytes:
            return obj

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == b"loaded_test"


# LLM-generated content at query #8
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str, /) -> str:
            return payload
        def dumps(self, obj: str, /) -> str:
            return obj

    serializer = StringSerializer()
    assert serializer.loads("test") == "test"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return payload
        def dumps(self, obj: bytes, /) -> bytes:
            return obj

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == b"test"

    # Test with a JSON serializer
    assert json.loads('{"key": "value"}') == {"key": "value"}
    assert json.loads(b'{"key": "value"}') == {"key": "value"}

    # Test with a custom serializer that handles both str and bytes
    class CustomSerializer:
        def loads(self, payload: str | bytes, /) -> str:
            if isinstance(payload, bytes):
                return payload.decode('utf-8')
            return payload
        def dumps(self, obj: str, /) -> str:
            return obj

    serializer = CustomSerializer()
    assert serializer.loads("test") == "test"
    assert serializer.loads(b"test") == "test"


# LLM-generated content at query #9
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json)
    serializer = Serializer("secret-key", salt="test-salt")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert "key" in signed_data
    assert "value" in signed_data

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode("utf-8")

        def loads(self, payload):
            return eval(payload.decode("utf-8"))

    bytes_serializer = Serializer("secret-key", salt="test-salt", serializer=BytesSerializer())
    signed_bytes = bytes_serializer.dumps(data)
    assert isinstance(signed_bytes, bytes)
    assert b"key" in signed_bytes
    assert b"value" in signed_bytes

    # Test with custom salt
    custom_salt_serializer = Serializer("secret-key", salt="custom-salt")
    signed_custom = custom_salt_serializer.dumps(data)
    assert signed_custom != signed_data

    # Test with key rotation
    keys = ["old-key", "new-key"]
    key_rotation_serializer = Serializer(keys, salt="test-salt")
    signed_rotation = key_rotation_serializer.dumps(data)
    assert isinstance(signed_rotation, str)
    assert "key" in signed_rotation
    assert "value" in signed_rotation

    # Test with different data types
    data_list = [1, 2, 3]
    signed_list = serializer.dumps(data_list)
    assert isinstance(signed_list, str)
    assert "1" in signed_list
    assert "2" in signed_list
    assert "3" in signed_list

    # Test with empty data
    empty_data = {}
    signed_empty = serializer.dumps(empty_data)
    assert isinstance(signed_empty, str)


# LLM-generated content at query #10
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default signer and no fallback signers
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
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(Signer, {"digest_method": "sha256"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].digest_method == Signer.digest_method
    assert signers[1].digest_method == "sha256"

    # Test with fallback signers as Signer class
    serializer = Serializer(
        "secret-key",
        fallback_signers=[Signer]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert all(isinstance(s, Signer) for s in signers)

    # Test with multiple fallback signers
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"digest_method": "sha256"},
            (Signer, {"digest_method": "sha512"}),
            Signer
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    assert signers[0].digest_method == Signer.digest_method
    assert signers[1].digest_method == "sha256"
    assert signers[2].digest_method == "sha512"
    assert signers[3].digest_method == Signer.digest_method

    # Test with custom salt in fallback signers
    serializer = Serializer(
        "secret-key",
        salt="main-salt",
        fallback_signers=[{"salt": "fallback-salt"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].salt == b"main-salt"
    assert signers[1].salt == b"fallback-salt"

    # Test with custom salt parameter in iter_unsigners
    serializer = Serializer("secret-key", salt="default-salt")
    signers = list(serializer.iter_unsigners(salt="custom-salt"))
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str, /) -> str:
            return payload

        def dumps(self, obj: str, /) -> str:
            return obj

    serializer = StringSerializer()
    assert serializer.loads("test") == "test"

    # Test with a simple bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return payload

        def dumps(self, obj: bytes, /) -> bytes:
            return obj

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == b"test"

    # Test with a JSON serializer
    assert json.loads('{"key": "value"}') == {"key": "value"}
    assert json.loads(b'{"key": "value"}') == {"key": "value"}

    # Test with a custom serializer that handles both str and bytes
    class CustomSerializer:
        def loads(self, payload: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return payload.upper()
            return payload.upper()

        def dumps(self, obj: str | bytes, /) -> str | bytes:
            if isinstance(obj, str):
                return obj.lower()
            return obj.lower()

    serializer = CustomSerializer()
    assert serializer.loads("test") == "TEST"
    assert serializer.loads(b"test") == b"TEST"


# LLM-generated content at query #13
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

    # Test with a serializer that matches the strict protocol
    class StrictSerializer:
        def loads(self, payload: str | bytes, /) -> t.Any:
            if isinstance(payload, str):
                return f"loaded_{payload}"
            return f"loaded_{payload.decode()}"

        def dumps(self, obj: t.Any, /) -> str | bytes:
            if isinstance(obj, str):
                return f"dumped_{obj}".encode()
            return f"dumped_{obj}"

    serializer = StrictSerializer()
    assert serializer.loads("test") == "loaded_test"
    assert serializer.loads(b"test") == "loaded_test"


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
def test_Serializer_iter_unsigners():
    secret_key = "secret"
    salt = "test-salt"
    serializer = Serializer(secret_key, salt)

    # Test with default signer
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].salt == salt.encode()
    assert signers[0].secret_keys == [secret_key.encode()]

    # Test with fallback signers
    fallback_signers = [{"digest_method": "md5"}, (Signer, {"digest_method": "sha1"})]
    serializer_with_fallback = Serializer(secret_key, salt, fallback_signers=fallback_signers)
    signers = list(serializer_with_fallback.iter_unsigners())
    assert len(signers) == 3
    assert isinstance(signers[0], Signer)
    assert signers[0].salt == salt.encode()
    assert signers[0].secret_keys == [secret_key.encode()]
    assert isinstance(signers[1], Signer)
    assert signers[1].digest_method == "md5"
    assert isinstance(signers[2], Signer)
    assert signers[2].digest_method == "sha1"

    # Test with different salt
    new_salt = "new-salt"
    signers = list(serializer.iter_unsigners(new_salt))
    assert len(signers) == 1
    assert signers[0].salt == new_salt.encode()

    # Test with key rotation
    secret_keys = ["old-secret", "new-secret"]
    serializer_with_rotation = Serializer(secret_keys, salt)
    signers = list(serializer_with_rotation.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-secret", b"new-secret"]

    # Test with key rotation and fallback signers
    serializer_with_rotation_and_fallback = Serializer(secret_keys, salt, fallback_signers=fallback_signers)
    signers = list(serializer_with_rotation_and_fallback.iter_unsigners())
    assert len(signers) == 5  # 1 default + 2 fallbacks * 2 keys
    assert signers[0].secret_keys == [b"old-secret", b"new-secret"]
    assert signers[1].secret_keys == [b"old-secret"]
    assert signers[2].secret_keys == [b"new-secret"]
    assert signers[3].secret_keys == [b"old-secret"]
    assert signers[4].secret_keys == [b"new-secret"]


# LLM-generated content at query #16
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


# LLM-generated content at query #17
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string payload
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    loaded_data = serializer.loads(signed_data)
    assert loaded_data == data

    # Test with bytes payload
    serializer_bytes = Serializer("secret-key", serializer=json)
    data_bytes = {"key": "value"}
    signed_data_bytes = serializer_bytes.dumps(data_bytes)
    loaded_data_bytes = serializer_bytes.loads(signed_data_bytes)
    assert loaded_data_bytes == data_bytes

    # Test with invalid signature
    serializer_invalid = Serializer("secret-key")
    invalid_signed_data = serializer_invalid.dumps({"key": "value"}) + b"invalid"
    try:
        serializer_invalid.loads(invalid_signed_data)
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

    # Test with invalid payload
    serializer_invalid_payload = Serializer("secret-key")
    invalid_payload = b"invalid_payload"
    try:
        serializer_invalid_payload.loads(invalid_payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass

    # Test with custom salt
    serializer_salt = Serializer("secret-key", salt="custom-salt")
    data_salt = {"key": "value"}
    signed_data_salt = serializer_salt.dumps(data_salt)
    loaded_data_salt = serializer_salt.loads(signed_data_salt)
    assert loaded_data_salt == data_salt

    # Test with key rotation
    serializer_rotation = Serializer(["old-key", "new-key"])
    data_rotation = {"key": "value"}
    signed_data_rotation = serializer_rotation.dumps(data_rotation)
    loaded_data_rotation = serializer_rotation.loads(signed_data_rotation)
    assert loaded_data_rotation == data_rotation


# LLM-generated content at query #18
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple dictionary
    data = {"key": "value"}
    serializer = json
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a list
    data = [1, 2, 3]
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a string
    data = "hello world"
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with an integer
    data = 42
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a float
    data = 3.14
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with a boolean
    data = True
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data

    # Test with None
    data = None
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert json.loads(result) == data


# LLM-generated content at query #19
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text)
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
    payload = b"test_payload"
    assert serializer.load_payload(payload) == {"custom": "test_payload"}

    # Test with bytes serializer
    class CustomBytesSerializer:
        def loads(self, payload: bytes) -> dict:
            return {"bytes": payload.decode()}

        def dumps(self, obj: dict) -> bytes:
            return obj["bytes"].encode()

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b"test_payload"
    assert serializer.load_payload(payload) == {"bytes": "test_payload"}

    # Test with BadPayload exception
    class FailingSerializer:
        def loads(self, payload: str) -> None:
            raise ValueError("Serialization failed")

        def dumps(self, obj: None) -> str:
            return ""

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b"test_payload"
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(payload)
    assert "Could not load the payload" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, ValueError)


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


# LLM-generated content at query #21
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


# LLM-generated content at query #22
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple string serializer
    class StringSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = StringSerializer()
    assert serializer.dumps("test") == "test"
    assert serializer.dumps(123) == "123"

    # Test with a bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps("test") == b"test"
    assert serializer.dumps(123) == b"123"

    # Test with a JSON serializer
    class JSONSerializer:
        def dumps(self, obj):
            return json.dumps(obj)

    serializer = JSONSerializer()
    assert serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"

    # Test with a custom object
    class CustomObject:
        def __init__(self, value):
            self.value = value

    class CustomSerializer:
        def dumps(self, obj):
            if isinstance(obj, CustomObject):
                return f"CustomObject({obj.value})"
            return str(obj)

    serializer = CustomSerializer()
    obj = CustomObject(42)
    assert serializer.dumps(obj) == "CustomObject(42)"
    assert serializer.dumps("test") == "test"


# LLM-generated content at query #23
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns str
    class StrSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = StrSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"

    # Test with a serializer that returns bytes
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == b"[1, 2, 3]"

    # Test with a more complex object
    class ComplexSerializer:
        def dumps(self, obj):
            if isinstance(obj, dict):
                return json.dumps(obj)
            elif isinstance(obj, list):
                return str(obj)
            else:
                return str(obj)

    serializer = ComplexSerializer()
    assert serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps("test") == "test"


# LLM-generated content at query #24
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns str
    class StrSerializer:
        def dumps(self, obj: t.Any, /) -> str:
            return str(obj)

    serializer = StrSerializer()
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"

    # Test with a simple serializer that returns bytes
    class BytesSerializer:
        def dumps(self, obj: t.Any, /) -> bytes:
            return str(obj).encode('utf-8')

    serializer = BytesSerializer()
    assert serializer.dumps({"key": "value"}) == b"{'key': 'value'}"
    assert serializer.dumps([1, 2, 3]) == b"[1, 2, 3]"

    # Test with a serializer that raises an exception
    class ExceptionSerializer:
        def dumps(self, obj: t.Any, /) -> str:
            raise ValueError("Serialization error")

    serializer = ExceptionSerializer()
    with pytest.raises(ValueError):
        serializer.dumps({"key": "value"})


# LLM-generated content at query #25
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

    # Test with custom text serializer
    class CustomTextSerializer:
        def loads(self, payload: str) -> str:
            return f"loaded: {payload}"

        def dumps(self, obj: str) -> str:
            return obj

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"test_data"
    assert serializer.load_payload(payload) == "loaded: test_data"

    # Test with bytes serializer
    class CustomBytesSerializer:
        def loads(self, payload: bytes) -> bytes:
            return b"loaded: " + payload

        def dumps(self, obj: bytes) -> bytes:
            return obj

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b"test_data"
    assert serializer.load_payload(payload) == b"loaded: test_data"

    # Test with invalid payload
    serializer = Serializer("secret-key")
    payload = b"invalid_json"
    with pytest.raises(BadPayload):
        serializer.load_payload(payload)

    # Test with custom serializer that raises exception
    class FailingSerializer:
        def loads(self, payload: str) -> str:
            raise ValueError("Custom error")

        def dumps(self, obj: str) -> str:
            return obj

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b"test_data"
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(payload)
    assert "Could not load the payload" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, ValueError)


# LLM-generated content at query #26
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json)
    serializer = Serializer("secret-key")
    obj = {"key": "value"}
    signed = serializer.dumps(obj)
    assert isinstance(signed, str)
    assert serializer.loads(signed) == obj

    # Test with custom text serializer
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, s):
            return s

    serializer = Serializer("secret-key", serializer=TextSerializer())
    signed = serializer.dumps(obj)
    assert isinstance(signed, str)
    assert serializer.loads(signed) == str(obj)

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode()

        def loads(self, b):
            return b.decode()

    serializer = Serializer("secret-key", serializer=BytesSerializer())
    signed = serializer.dumps(obj)
    assert isinstance(signed, bytes)
    assert serializer.loads(signed) == str(obj)

    # Test with salt
    serializer = Serializer("secret-key", salt="custom-salt")
    signed = serializer.dumps(obj)
    assert serializer.loads(signed, salt="custom-salt") == obj

    # Test with key rotation
    serializer = Serializer(["old-key", "new-key"])
    signed = serializer.dumps(obj)
    assert serializer.loads(signed) == obj


# LLM-generated content at query #27
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple dict
    serializer = json
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert json.loads(result) == obj

    # Test with a list
    obj = [1, 2, 3]
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert json.loads(result) == obj

    # Test with a string
    obj = "test string"
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert json.loads(result) == obj

    # Test with an integer
    obj = 42
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert json.loads(result) == obj

    # Test with a float
    obj = 3.14
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert json.loads(result) == obj

    # Test with a boolean
    obj = True
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert json.loads(result) == obj

    # Test with None
    obj = None
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert json.loads(result) is None


# LLM-generated content at query #28
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with JSON serializer (default)
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
            return int(payload)

    serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    data = 42
    signed_data = serializer.dumps(data)
    loaded_data = serializer.loads(signed_data)
    assert loaded_data == data

    # Test with custom bytes serializer
    class CustomBytesSerializer:
        def dumps(self, obj):
            return bytes(str(obj), 'utf-8')

        def loads(self, payload):
            return int(payload.decode('utf-8'))

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    data = 42
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
        serializer.loads(serializer.dumps({"key": "value"}) + b"corrupted")
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass


# LLM-generated content at query #29
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

    # Test with a complex object serializer
    class ComplexSerializer:
        def loads(self, payload: str, /) -> dict:
            return json.loads(payload)

        def dumps(self, obj: dict, /) -> str:
            return json.dumps(obj)

    serializer = ComplexSerializer()
    assert serializer.loads('{"key": "value"}') == {"key": "value"}

    # Test with a serializer that raises an exception
    class ExceptionSerializer:
        def loads(self, payload: str, /) -> str:
            raise ValueError("Test exception")

        def dumps(self, obj: str, /) -> str:
            return obj

    serializer = ExceptionSerializer()
    try:
        serializer.loads("test")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Test exception"


# LLM-generated content at query #30
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
    assert isinstance(dumped, str)
    assert json.loads(dumped) == data


# LLM-generated content at query #31
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text)
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
    payload = b'data'
    assert serializer.load_payload(payload) == {"custom": "data"}

    # Test with bytes serializer
    class CustomBytesSerializer:
        def loads(self, payload: bytes) -> dict:
            return {"bytes": payload.decode()}

        def dumps(self, obj: dict) -> bytes:
            return obj["bytes"].encode()

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b'bytes_data'
    assert serializer.load_payload(payload) == {"bytes": "bytes_data"}

    # Test with invalid payload
    serializer = Serializer("secret-key")
    payload = b'invalid_json'
    with pytest.raises(BadPayload):
        serializer.load_payload(payload)

    # Test with custom serializer that raises an exception
    class FailingSerializer:
        def loads(self, payload: str) -> None:
            raise ValueError("Custom error")

        def dumps(self, obj: None) -> str:
            return ""

    serializer = Serializer("secret-key", serializer=FailingSerializer())
    payload = b'data'
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(payload)
    assert "Custom error" in str(exc_info.value.original_error)


# LLM-generated content at query #32
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

    # Test with a serializer that doesn't match the strict protocol
    class CustomSerializer:
        def loads(self, payload: str, extra_arg: str = "default") -> str:
            return f"loaded_{payload}_{extra_arg}"

        def dumps(self, obj: str, extra_arg: str = "default") -> str:
            return f"dumped_{obj}_{extra_arg}"

    serializer = CustomSerializer()
    assert serializer.loads("test") == "loaded_test_default"
    assert serializer.loads("test", "custom") == "loaded_test_custom"


# LLM-generated content at query #33
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
    class DictSerializer:
        def loads(self, payload: str, /) -> dict:
            return {"data": payload}

        def dumps(self, obj: dict, /) -> str:
            return obj["data"]

    serializer = DictSerializer()
    assert serializer.loads("test") == {"data": "test"}


# LLM-generated content at query #34
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json) and text output
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert "key" in signed_data
    assert "value" in signed_data

    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode("utf-8")

        def loads(self, payload):
            return eval(payload.decode("utf-8"))

    serializer_bytes = Serializer("secret-key", serializer=BytesSerializer())
    signed_data_bytes = serializer_bytes.dumps(data)
    assert isinstance(signed_data_bytes, bytes)
    assert b"key" in signed_data_bytes
    assert b"value" in signed_data_bytes

    # Test with custom salt
    serializer_salt = Serializer("secret-key", salt="custom-salt")
    signed_data_salt = serializer_salt.dumps(data)
    assert isinstance(signed_data_salt, str)
    assert "key" in signed_data_salt
    assert "value" in signed_data_salt

    # Test with key rotation
    serializer_keys = Serializer(["old-key", "new-key"])
    signed_data_keys = serializer_keys.dumps(data)
    assert isinstance(signed_data_keys, str)
    assert "key" in signed_data_keys
    assert "value" in signed_data_keys

    # Test with custom signer
    from .signer import Signer
    class CustomSigner(Signer):
        pass

    serializer_signer = Serializer("secret-key", signer=CustomSigner)
    signed_data_signer = serializer_signer.dumps(data)
    assert isinstance(signed_data_signer, str)
    assert "key" in signed_data_signer
    assert "value" in signed_data_signer


# LLM-generated content at query #35
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default serializer (json)
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
    payload = b'custom_data'
    assert serializer.load_payload(payload) == {"custom": "custom_data"}

    # Test with custom bytes serializer
    class CustomBytesSerializer:
        def loads(self, payload: bytes) -> dict:
            return {"custom": payload.decode()}

        def dumps(self, obj: dict) -> bytes:
            return obj["custom"].encode()

    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b'custom_data'
    assert serializer.load_payload(payload) == {"custom": "custom_data"}

    # Test with serializer override
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload, json) == {"key": "value"}

    # Test with invalid payload
    serializer = Serializer("secret-key")
    payload = b'invalid_json'
    with pytest.raises(BadPayload):
        serializer.load_payload(payload)


# LLM-generated content at query #36
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple text serializer (str)
    class TextSerializer:
        def loads(self, payload: str, /) -> str:
            return f"loaded_{payload}"

        def dumps(self, obj: str, /) -> str:
            return f"dumped_{obj}"

    serializer = TextSerializer()
    assert serializer.loads("test") == "loaded_test"

    # Test with a simple bytes serializer (bytes)
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> bytes:
            return b"loaded_" + payload

        def dumps(self, obj: bytes, /) -> bytes:
            return b"dumped_" + obj

    serializer = BytesSerializer()
    assert serializer.loads(b"test") == b"loaded_test"

    # Test with a serializer that doesn't match the strict protocol
    class CustomSerializer:
        def loads(self, payload: str | bytes, /) -> str | bytes:
            if isinstance(payload, str):
                return f"loaded_{payload}"
            return b"loaded_" + payload

        def dumps(self, obj: str | bytes, /) -> str | bytes:
            if isinstance(obj, str):
                return f"dumped_{obj}"
            return b"dumped_" + obj

    serializer = CustomSerializer()
    assert serializer.loads("test") == "loaded_test"
    assert serializer.loads(b"test") == b"loaded_test"


