####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a concrete implementation that matches the protocol
    class TestSerializer:
        def loads(self, payload):
            if isinstance(payload, str):
                return {"data": payload}
            return {"data": payload.decode()}
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with string payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"data": '{"key": "value"}'}
    
    # Test with bytes payload
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"data": '{"key": "value"}'}
    
    # Test with empty string
    result = serializer.loads("")
    assert result == {"data": ""}
    
    # Test with numeric string
    result = serializer.loads("123")
    assert result == {"data": "123"}
    
    # Test with special characters
    result = serializer.loads("hello world!")
    assert result == {"data": "hello world!"}
    
    # Verify protocol compatibility
    import typing as t
    assert isinstance(serializer, _PDataSerializer)
```


# LLM-generated content at query #2
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
        def loads(self, payload):
            return {"custom": payload}
        def dumps(self, obj):
            return "test"

    custom_serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"test data"
    result = custom_serializer.load_payload(payload)
    assert result == {"custom": "test data"}

    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return {"bytes": payload}
        def dumps(self, obj):
            return b"test"

    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    payload = b"binary data"
    result = bytes_serializer.load_payload(payload)
    assert result == {"bytes": b"binary data"}

    # Test with override serializer parameter
    serializer = Serializer("secret-key")
    class OverrideSerializer:
        def loads(self, payload):
            return {"override": payload}
        def dumps(self, obj):
            return "test"
    
    result = serializer.load_payload(b"data", serializer=OverrideSerializer())
    assert result == {"override": b"data"}

    # Test with invalid payload that raises BadPayload
    serializer = Serializer("secret-key")
    try:
        serializer.load_payload(b"invalid json")
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass

    # Test with text serializer that raises exception
    class BadTextSerializer:
        def loads(self, payload):
            raise ValueError("Load error")
        def dumps(self, obj):
            return "test"

    bad_serializer = Serializer("secret-key", serializer=BadTextSerializer())
    try:
        bad_serializer.load_payload(b"test")
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert isinstance(e.original_error, ValueError)

    # Test with bytes serializer that raises exception
    class BadBytesSerializer:
        def loads(self, payload):
            raise TypeError("Type error")
        def dumps(self, obj):
            return b"test"

    bad_bytes_serializer = Serializer("secret-key", serializer=BadBytesSerializer())
    try:
        bad_bytes_serializer.load_payload(b"test")
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert isinstance(e.original_error, TypeError)

    # Test with empty payload
    serializer = Serializer("secret-key")
    try:
        serializer.load_payload(b"")
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass

    # Test with payload that decodes to valid JSON but empty
    serializer = Serializer("secret-key")
    result = serializer.load_payload(b"null")
    assert result is None

    result = serializer.load_payload(b"[]")
    assert result == []

    result = serializer.load_payload(b"{}")
    assert result == {}
```


# LLM-generated content at query #3
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a text serializer (json)
    serializer = _PDataSerializer[str]
    
    # Create a mock serializer that implements the protocol
    class MockTextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    mock_serializer = MockTextSerializer()
    payload = '{"key": "value"}'
    result = mock_serializer.loads(payload)
    assert result == {"key": "value"}
    
    # Test with a bytes serializer
    class MockBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    mock_bytes_serializer = MockBytesSerializer()
    payload_bytes = b'{"key": "value"}'
    result_bytes = mock_bytes_serializer.loads(payload_bytes)
    assert result_bytes == {"key": "value"}
    
    # Test with integer payload
    payload_int_str = '42'
    result_int = mock_serializer.loads(payload_int_str)
    assert result_int == 42
    
    # Test with list payload
    payload_list_str = '[1, 2, 3]'
    result_list = mock_serializer.loads(payload_list_str)
    assert result_list == [1, 2, 3]
    
    # Test with null payload
    payload_null_str = 'null'
    result_null = mock_serializer.loads(payload_null_str)
    assert result_null is None
    
    # Test with boolean payload
    payload_bool_str = 'true'
    result_bool = mock_serializer.loads(payload_bool_str)
    assert result_bool is True
    
    # Test with float payload
    payload_float_str = '3.14'
    result_float = mock_serializer.loads(payload_float_str)
    assert result_float == 3.14
```


# LLM-generated content at query #4
#--------------------------

```python
def test_Serializer_dumps():
    serializer = Serializer("secret-key")
    
    # Test dumps with default JSON serializer (text)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert serializer.loads(result) == {"key": "value"}
    
    # Test dumps with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert result_with_salt != result
    assert serializer.loads(result_with_salt, salt="custom-salt") == {"key": "value"}
    
    # Test dumps produces different signatures for different data
    result1 = serializer.dumps("data1")
    result2 = serializer.dumps("data2")
    assert result1 != result2
    
    # Test dumps with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            import json
            return json.loads(payload.decode())
        def dumps(self, obj):
            import json
            return json.dumps(obj).encode()
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert bytes_serializer.loads(result_bytes) == {"key": "value"}


# LLM-generated content at query #5
#--------------------------

```python
def test_Serializer_dumps():
    """Test that dumps correctly signs and serializes data."""
    # Test with default JSON serializer (text)
    serializer = Serializer("test-secret")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Should contain separator
    
    # Test that different data produces different output
    result2 = serializer.dumps({"key": "different"})
    assert result != result2
    
    # Test with binary serializer
    class BinarySerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode()
        def loads(self, payload):
            return json.loads(payload)
    
    binary_serializer = Serializer("test-secret", serializer=BinarySerializer())
    binary_result = binary_serializer.dumps({"key": "value"})
    assert isinstance(binary_result, bytes)
    
    # Test with salt parameter
    salted_result = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert salted_result != result
    
    # Test with serializer_kwargs
    class CustomSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj, **kwargs)
        def loads(self, payload):
            return json.loads(payload)
    
    custom_serializer = Serializer(
        "test-secret",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    custom_result = custom_serializer.dumps({"b": 2, "a": 1})
    assert custom_result is not None
    
    # Test that dumps works with various data types
    for data in [None, True, 42, 3.14, "string", [1, 2, 3], {"nested": {"key": "value"}}]:
        result = serializer.dumps(data)
        assert isinstance(result, str)
        assert len(result) > 0


# LLM-generated content at query #6
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns str
    class StrSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    serializer = StrSerializer()
    result = serializer.loads("hello")
    assert result == "HELLO"
    
    # Test with a JSON serializer
    json_serializer = json
    result = json_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode("utf-8").split(",")
        
        def dumps(self, obj: t.Any) -> bytes:
            return ",".join(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b"a,b,c")
    assert result == ["a", "b", "c"]
    
    # Test with integer payload
    class IntSerializer:
        def loads(self, payload: str) -> t.Any:
            return int(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    int_serializer = IntSerializer()
    result = int_serializer.loads("42")
    assert result == 42
    assert isinstance(result, int)
    
    # Test with None payload
    class NoneSerializer:
        def loads(self, payload: str) -> t.Any:
            return None
        
        def dumps(self, obj: t.Any) -> str:
            return "null"
    
    none_serializer = NoneSerializer()
    result = none_serializer.loads("anything")
    assert result is None
```


# LLM-generated content at query #7
#--------------------------

```python
def test_Serializer_iter_unsigners():
    """Test that iter_unsigners yields the main signer first, then fallback signers."""
    # Setup: Create a Serializer with a known secret key and salt
    secret_key = b"test-secret-key"
    salt = b"test-salt"
    
    # Create a custom signer class to track instances
    class TestSigner(Signer):
        def __init__(self, secret_key, salt=None, **kwargs):
            self._secret_key = secret_key
            self._salt = salt
            super().__init__(secret_key, salt=salt, **kwargs)
    
    # Create a fallback signer with different parameters
    fallback_signer = TestSigner
    fallback_kwargs = {"digest_method": "sha256"}
    
    # Create Serializer with fallback signers
    serializer = Serializer(
        secret_key=secret_key,
        salt=salt,
        signer=TestSigner,
        fallback_signers=[
            {"digest_method": "sha256"},  # dict fallback
            (fallback_signer, {"digest_method": "sha512"}),  # tuple fallback
            fallback_signer,  # class fallback
        ]
    )
    
    # Call iter_unsigners
    signers = list(serializer.iter_unsigners())
    
    # Verify we got the right number of signers
    # Main signer + 3 fallback signers
    assert len(signers) == 4
    
    # Check main signer is yielded first
    main_signer = signers[0]
    assert isinstance(main_signer, TestSigner)
    assert main_signer._salt == salt
    
    # Check dict fallback (uses default signer class with provided kwargs)
    dict_fallback = signers[1]
    assert isinstance(dict_fallback, TestSigner)
    
    # Check tuple fallback (uses specified signer class with provided kwargs)
    tuple_fallback = signers[2]
    assert isinstance(tuple_fallback, TestSigner)
    
    # Check class fallback (uses specified signer class with default kwargs)
    class_fallback = signers[3]
    assert isinstance(class_fallback, TestSigner)
    
    # Test with custom salt parameter
    custom_salt = b"custom-salt"
    custom_signers = list(serializer.iter_unsigners(salt=custom_salt))
    for signer in custom_signers:
        assert signer._salt == custom_salt
    
    # Test with no fallback signers
    serializer_no_fallback = Serializer(
        secret_key=secret_key,
        salt=salt,
        fallback_signers=[]
    )
    no_fallback_signers = list(serializer_no_fallback.iter_unsigners())
    assert len(no_fallback_signers) == 1
    assert isinstance(no_fallback_signers[0], Signer)
    
    # Test with multiple secret keys (key rotation scenario)
    multi_key_signer = Serializer(
        secret_key=[b"old-key", b"newer-key", b"current-key"],
        salt=salt,
        fallback_signers=[{"digest_method": "sha256"}]
    )
    multi_signers = list(multi_key_signer.iter_unsigners())
    # Main signer + 3 fallback signers (one for each secret key)
    assert len(multi_signers) == 4
    
    # Verify each fallback signer uses a different secret key
    assert multi_signers[0].secret_key == b"current-key"  # main signer uses newest key
    assert multi_signers[1].secret_key == b"old-key"  # first fallback with oldest key
    assert multi_signers[2].secret_key == b"newer-key"  # second fallback
    assert multi_signers[3].secret_key == b"current-key"  # third fallback with newest key
```


# LLM-generated content at query #8
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple json serializer
    serializer = json
    
    # Test basic JSON payload
    payload = '{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}
    
    # Test with integer
    payload = "42"
    result = serializer.loads(payload)
    assert result == 42
    
    # Test with list
    payload = "[1, 2, 3]"
    result = serializer.loads(payload)
    assert result == [1, 2, 3]
    
    # Test with nested structure
    payload = '{"nested": {"inner": "data"}}'
    result = serializer.loads(payload)
    assert result == {"nested": {"inner": "data"}}
    
    # Test with boolean
    payload = "true"
    result = serializer.loads(payload)
    assert result is True
    
    payload = "false"
    result = serializer.loads(payload)
    assert result is False
    
    # Test with null
    payload = "null"
    result = serializer.loads(payload)
    assert result is None
    
    # Test with empty string
    payload = '""'
    result = serializer.loads(payload)
    assert result == ""
    
    # Test with float
    payload = "3.14"
    result = serializer.loads(payload)
    assert result == 3.14
    
    # Test with empty object
    payload = "{}"
    result = serializer.loads(payload)
    assert result == {}
    
    # Test with empty array
    payload = "[]"
    result = serializer.loads(payload)
    assert result == []
```


# LLM-generated content at query #9
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default signer only (no fallback signers)
    serializer = Serializer("secret-key", fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret-key"]

    # Test with custom salt
    serializer = Serializer("secret-key", salt=b"custom-salt", fallback_signers=[])
    signers = list(serializer.iter_unsigners(salt=b"override-salt"))
    assert len(signers) == 1
    assert signers[0].salt == b"override-salt"

    # Test with fallback signers as dict
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"key_derivation": "none"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].key_derivation == "none"

    # Test with fallback signers as tuple
    class CustomSigner(Signer):
        pass
    
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(CustomSigner, {"key_derivation": "hmac"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], CustomSigner)
    assert signers[1].key_derivation == "hmac"

    # Test with fallback signers as Signer class
    serializer = Serializer(
        "secret-key",
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], CustomSigner)

    # Test with multiple secret keys
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"key_derivation": "none"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3  # 1 default + 2 from fallback (one for each secret key)
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_keys == [b"old-key"]
    assert signers[2].secret_keys == [b"new-key"]

    # Test that default fallback signers are used when none provided
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1  # Only default signer, no fallbacks

    # Test with mixed fallback signers
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"key_derivation": "none"},
            (CustomSigner, {"digest_method": "sha256"}),
            CustomSigner
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].key_derivation == "none"
    assert isinstance(signers[2], CustomSigner)
    assert signers[2].digest_method == "sha256"
    assert isinstance(signers[3], CustomSigner)

    # Test that salt is passed correctly to fallback signers
    serializer = Serializer(
        "secret-key",
        salt=b"base-salt",
        fallback_signers=[{"key_derivation": "none"}]
    )
    signers = list(serializer.iter_unsigners())
    for signer in signers:
        assert signer.salt == b"base-salt"

    # Test with salt override
    signers = list(serializer.iter_unsigners(salt=b"override"))
    for signer in signers:
        assert signer.salt == b"override"
```


# LLM-generated content at query #10
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()

    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'

    # Test with a simple serializer that returns str
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)

    text_serializer = TextSerializer()
    result = text_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'

    # Test with empty dict
    result = bytes_serializer.dumps({})
    assert result == b'{}'

    # Test with None
    result = bytes_serializer.dumps(None)
    assert result == b'null'

    # Test with list
    result = bytes_serializer.dumps([1, 2, 3])
    assert result == b'[1, 2, 3]'

    # Test with int
    result = bytes_serializer.dumps(42)
    assert result == b'42'

    # Test with string
    result = bytes_serializer.dumps("test")
    assert result == b'"test"'


# LLM-generated content at query #11
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer.dumps returns the expected type."""
    serializer = _PDataSerializer()
    assert isinstance(serializer.dumps({"test": "data"}), str)

    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload

        def dumps(self, obj: t.Any) -> bytes:
            return b"test_bytes"

    bytes_serializer = BytesSerializer()
    assert isinstance(bytes_serializer.dumps({"test": "data"}), bytes)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_Serializer_load_payload():
    """Test load_payload method with various scenarios."""
    
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("test-secret")
    
    # Test loading valid JSON payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading valid JSON payload with explicit serializer parameter
    result = serializer.load_payload(payload, serializer=json)
    assert result == {"key": "value"}
    
    # Test loading invalid JSON payload raises BadPayload
    invalid_payload = b"not valid json"
    try:
        serializer.load_payload(invalid_payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None
    
    # Test with bytes serializer (custom serializer that returns bytes)
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return {"data": payload.decode()}
        
        def dumps(self, obj: t.Any) -> bytes:
            return b"test"
    
    bytes_serializer = Serializer("test-secret", serializer=BytesSerializer())
    
    # Test loading with bytes serializer
    payload = b"hello"
    result = bytes_serializer.load_payload(payload)
    assert result == {"data": "hello"}
    
    # Test loading with bytes serializer using explicit serializer parameter
    result = bytes_serializer.load_payload(payload, serializer=BytesSerializer())
    assert result == {"data": "hello"}
    
    # Test loading empty payload
    try:
        serializer.load_payload(b"")
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
    
    # Test loading payload that causes serializer to raise an exception
    class FailingSerializer:
        def loads(self, payload: bytes) -> t.Any:
            raise ValueError("Serializer failed")
        
        def dumps(self, obj: t.Any) -> bytes:
            return b""
    
    failing_serializer = Serializer("test-secret", serializer=FailingSerializer())
    
    try:
        failing_serializer.load_payload(b"test")
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)
        assert str(e.original_error) == "Serializer failed"
    
    # Test with text serializer that returns string from dumps
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"text": payload}
        
        def dumps(self, obj: t.Any) -> str:
            return "test"
    
    text_serializer = Serializer("test-secret", serializer=TextSerializer())
    
    # Test loading with text serializer (payload should be decoded)
    payload = b"hello"
    result = text_serializer.load_payload(payload)
    assert result == {"text": "hello"}
    
    # Test that text serializer properly decodes UTF-8
    payload = b"\xc3\xa9"  # é in UTF-8
    result = text_serializer.load_payload(payload)
    assert result == {"text": "é"}
    
    # Test with non-UTF-8 bytes raises UnicodeDecodeError and then BadPayload
    payload = b"\xff\xfe"  # Invalid UTF-8
    try:
        text_serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
    
    # Test load_payload with None payload
    try:
        serializer.load_payload(None)  # type: ignore[arg-type]
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
    
    # Test load_payload with integer payload (should fail)
    try:
        serializer.load_payload(123)  # type: ignore[arg-type]
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
```


# LLM-generated content at query #13
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer dumps method works correctly."""
    # Create a simple serializer that implements the protocol
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
            
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    test_data = {"key": "value"}
    result = serializer.dumps(test_data)
    assert result == '{"key": "value"}'
    assert isinstance(result, str)
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
            
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    bytes_result = bytes_serializer.dumps(test_data)
    assert bytes_result == b'{"key": "value"}'
    assert isinstance(bytes_result, bytes)


# LLM-generated content at query #14
#--------------------------

```python
def test_Serializer_load_payload():
    """Test load_payload method of Serializer class."""
    # Setup
    serializer = Serializer("test-secret-key")
    
    # Test 1: Successful loading with default json serializer
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test 2: Successful loading with text serializer
    text_serializer = type('TextSerializer', (), {
        'dumps': lambda self, x: '{"text": "data"}',
        'loads': lambda self, x: {"text": "data"}
    })()
    serializer_with_text = Serializer("test-secret-key", serializer=text_serializer)
    result = serializer_with_text.load_payload(b'{"text": "data"}')
    assert result == {"text": "data"}
    
    # Test 3: Loading with custom serializer parameter
    custom_serializer = type('CustomSerializer', (), {
        'dumps': lambda self, x: '{"custom": "data"}',
        'loads': lambda self, x: {"custom": "data_from_custom"}
    })()
    result = serializer.load_payload(b'{"custom": "data"}', serializer=custom_serializer)
    assert result == {"custom": "data_from_custom"}
    
    # Test 4: Raises BadPayload on invalid JSON
    try:
        serializer.load_payload(b'invalid json')
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None
    
    # Test 5: Raises BadPayload on empty payload
    try:
        serializer.load_payload(b'')
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
    
    # Test 6: Loading with bytes serializer
    bytes_serializer = type('BytesSerializer', (), {
        'dumps': lambda self, x: b'{"bytes": "data"}',
        'loads': lambda self, x: {"bytes": "data_from_bytes"}
    })()
    
    # Override is_text_serializer for testing
    bytes_serializer_instance = Serializer(
        "test-secret-key", 
        serializer=bytes_serializer
    )
    result = bytes_serializer_instance.load_payload(b'{"bytes": "data"}')
    assert result == {"bytes": "data_from_bytes"}
    
    # Test 7: Loading complex nested data
    nested_payload = b'{"level1": {"level2": [1, 2, 3]}}'
    result = serializer.load_payload(nested_payload)
    assert result == {"level1": {"level2": [1, 2, 3]}}
    
    # Test 8: Loading list data
    list_payload = b'[1, 2, 3, "test"]'
    result = serializer.load_payload(list_payload)
    assert result == [1, 2, 3, "test"]
    
    # Test 9: Loading with None serializer (should use default)
    result = serializer.load_payload(b'{"test": "value"}', serializer=None)
    assert result == {"test": "value"}
    
    # Test 10: Custom serializer that raises an exception
    error_serializer = type('ErrorSerializer', (), {
        'dumps': lambda self, x: b'data',
        'loads': lambda self, x: (_ for _ in ()).throw(ValueError("Custom error"))
    })()
    
    try:
        serializer.load_payload(b'some data', serializer=error_serializer)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert isinstance(e.original_error, ValueError)
        assert "Custom error" in str(e.original_error)
```


# LLM-generated content at query #15
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def dumps(self, obj):
            return json.dumps(obj)
        
        def loads(self, payload):
            return json.loads(payload)
    
    serializer = TestSerializer()
    data = {"key": "value", "number": 42}
    result = serializer.dumps(data)
    expected = json.dumps(data)
    assert result == expected
    assert isinstance(result, str)
```


# LLM-generated content at query #16
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol requires dumps method."""
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer: _PDataSerializer[str] = MockSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == '{"key": "value"}'
    assert isinstance(result, str)


# LLM-generated content at query #17
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol requires a dumps method."""
    # Create a mock serializer that conforms to the protocol
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with various data types
    test_data = [
        {"key": "value"},
        [1, 2, 3],
        "string",
        42,
        3.14,
        True,
        None,
        {"nested": {"list": [1, 2, 3]}}
    ]
    
    for data in test_data:
        result = serializer.dumps(data)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert serializer.loads(result) == data, f"Roundtrip failed for {data}"
    
    # Test that the protocol is structural (duck typing)
    class MinimalSerializer:
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
        
        def loads(self, payload: str) -> t.Any:
            return payload
    
    minimal = MinimalSerializer()
    result = minimal.dumps(42)
    assert isinstance(result, str)
    assert result == "42"


# LLM-generated content at query #18
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test loading valid JSON payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading with custom serializer
    class CustomSerializer:
        def loads(self, payload: bytes) -> dict:
            return {"custom": payload.decode()}
        
        def dumps(self, obj: dict) -> bytes:
            return b"test"
    
    custom_serializer = CustomSerializer()
    result = serializer.load_payload(b"test_data", serializer=custom_serializer)
    assert result == {"custom": "test_data"}
    
    # Test with bytes serializer (non-text)
    class BytesSerializer:
        def loads(self, payload: bytes) -> dict:
            return {"data": payload}
        
        def dumps(self, obj: dict) -> bytes:
            return obj["data"]
    
    bytes_serializer = BytesSerializer()
    bytes_ser = Serializer("secret-key", serializer=bytes_serializer)
    result = bytes_ser.load_payload(b"binary_data")
    assert result == {"data": b"binary_data"}
    
    # Test with invalid payload (raises BadPayload)
    import pytest
    with pytest.raises(BadPayload):
        serializer.load_payload(b"invalid json")
    
    # Test with empty payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test with non-UTF-8 bytes for text serializer
    with pytest.raises(BadPayload):
        serializer.load_payload(b"\xff\xfe\x00\x01")
    
    # Test with custom serializer that raises exception
    class FailingSerializer:
        def loads(self, payload):
            raise ValueError("Custom error")
        
        def dumps(self, obj):
            return b"test"
    
    with pytest.raises(BadPayload):
        serializer.load_payload(b"test", serializer=FailingSerializer())
```


# LLM-generated content at query #19
#--------------------------

```python
def test_Serializer_dumps():
    serializer = Serializer("test-secret-key")
    
    # Test with a simple dictionary
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert serializer.loads(result) == data
    
    # Test with list
    data_list = [1, 2, 3, "test"]
    result = serializer.dumps(data_list)
    assert serializer.loads(result) == data_list
    
    # Test with None
    result = serializer.dumps(None)
    assert serializer.loads(result) is None
    
    # Test with custom salt
    result_with_salt = serializer.dumps(data, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert serializer.loads(result_with_salt, salt="custom-salt") == data
    
    # Test that different salt produces different signature
    result_different_salt = serializer.dumps(data, salt="different-salt")
    assert result_with_salt != result_different_salt
    
    # Test with bytes serializer
    bytes_serializer = Serializer("test-key", serializer=BytesSerializer())
    data_bytes = {"test": "data"}
    result_bytes = bytes_serializer.dumps(data_bytes)
    assert isinstance(result_bytes, bytes)
    assert bytes_serializer.loads(result_bytes) == data_bytes
    
    # Test with empty data
    result_empty = serializer.dumps({})
    assert serializer.loads(result_empty) == {}
    
    # Test with nested data
    nested_data = {"level1": {"level2": [1, 2, 3]}}
    result_nested = serializer.dumps(nested_data)
    assert serializer.loads(result_nested) == nested_data
    
    # Test that dumps produces a valid signed payload
    result = serializer.dumps("test")
    # The result should contain the separator
    assert "." in result

class BytesSerializer:
    """Simple bytes serializer for testing"""
    def dumps(self, obj):
        return json.dumps(obj).encode("utf-8")
    
    def loads(self, payload):
        return json.loads(payload.decode("utf-8"))
```


# LLM-generated content at query #20
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer.dumps returns the expected serialized type."""
    # Create a serializer that returns str
    class TextSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)
    
    # Create a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any, /) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    # Test with text serializer
    text_serializer = TextSerializer()
    text_result = text_serializer.dumps({"key": "value"})
    assert isinstance(text_result, str), "Text serializer should return str"
    assert text_result == '{"key": "value"}'
    
    # Test with bytes serializer
    bytes_serializer = BytesSerializer()
    bytes_result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(bytes_result, bytes), "Bytes serializer should return bytes"
    assert bytes_result == b'{"key": "value"}'
    
    # Test with default json serializer
    json_result = _PDataSerializer[json].dumps(json, {"test": 123})
    # Note: json module's dumps returns str by default
    assert isinstance(json_result, str)
    assert json_result == '{"test": 123}'


# LLM-generated content at query #21
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    serializer = BytesSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with a simple serializer that returns str
    class StrSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StrSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with None value
    class NoneSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return None
        
        def dumps(self, obj: t.Any) -> bytes:
            return b'null'
    
    serializer = NoneSerializer()
    result = serializer.dumps(None)
    assert isinstance(result, bytes)
    assert result == b'null'
    
    # Test with empty dict
    serializer = BytesSerializer()
    result = serializer.dumps({})
    assert isinstance(result, bytes)
    assert result == b'{}'
    
    # Test with list
    serializer = BytesSerializer()
    result = serializer.dumps([1, 2, 3])
    assert isinstance(result, bytes)
    assert result == b'[1, 2, 3]'
    
    # Test with string value
    serializer = StrSerializer()
    result = serializer.dumps("test_string")
    assert isinstance(result, str)
    assert result == '"test_string"'
    
    # Test with integer value
    serializer = BytesSerializer()
    result = serializer.dumps(42)
    assert isinstance(result, bytes)
    assert result == b'42'
    
    # Test with float value
    serializer = StrSerializer()
    result = serializer.dumps(3.14)
    assert isinstance(result, str)
    assert result == '3.14'
    
    # Test with boolean values
    serializer = BytesSerializer()
    result = serializer.dumps(True)
    assert isinstance(result, bytes)
    assert result == b'true'
    
    result = serializer.dumps(False)
    assert isinstance(result, bytes)
    assert result == b'false'


# LLM-generated content at query #22
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test successful loading with text serializer
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return b"dumped:" + str(obj).encode()
        
        def loads(self, payload):
            return payload.decode()
    
    bytes_serializer = BytesSerializer()
    serializer_bytes = Serializer("secret-key", serializer=bytes_serializer)
    payload_bytes = b"test:data"
    result = serializer_bytes.load_payload(payload_bytes)
    assert result == "test:data"
    
    # Test with custom text serializer
    class CustomTextSerializer:
        def dumps(self, obj):
            return str(obj)
        
        def loads(self, payload):
            return payload.upper()
    
    text_serializer = CustomTextSerializer()
    serializer_custom = Serializer("secret-key", serializer=text_serializer)
    payload_text = b"hello"
    result = serializer_custom.load_payload(payload_text)
    assert result == "HELLO"
    
    # Test BadPayload raised for invalid payload
    import json
    try:
        serializer.load_payload(b"invalid json")
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None
    
    # Test with serializer parameter override
    result = serializer.load_payload(b'{"test": 123}', serializer=json)
    assert result == {"test": 123}
```


# LLM-generated content at query #23
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps works correctly with different implementations."""

    # Test with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)

        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)

    str_serializer = StringSerializer()
    assert isinstance(str_serializer, _PDataSerializer)
    result = str_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)

        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")

    bytes_serializer = BytesSerializer()
    assert isinstance(bytes_serializer, _PDataSerializer)
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'

    # Test with various data types
    assert str_serializer.dumps(None) == "null"
    assert str_serializer.dumps(True) == "true"
    assert str_serializer.dumps(False) == "false"
    assert str_serializer.dumps(42) == "42"
    assert str_serializer.dumps(3.14) == "3.14"
    assert str_serializer.dumps("hello") == '"hello"'
    assert str_serializer.dumps([1, 2, 3]) == "[1, 2, 3]"

    # Test with bytes serializer
    assert bytes_serializer.dumps(None) == b"null"
    assert bytes_serializer.dumps(True) == b"true"
    assert bytes_serializer.dumps(42) == b"42"
    assert bytes_serializer.dumps([1, 2, 3]) == b"[1, 2, 3]"

    # Test that the dumps method is callable (protocol requirement)
    assert callable(str_serializer.dumps)
    assert callable(bytes_serializer.dumps)


# LLM-generated content at query #24
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method is properly defined."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}, "Should parse valid JSON"
    
    # Test with simple types
    assert serializer.loads('"string"') == "string"
    assert serializer.loads('123') == 123
    assert serializer.loads('true') == True
    assert serializer.loads('null') is None
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test with empty structures
    assert serializer.loads('{}') == {}
    assert serializer.loads('[]') == []
    
    # Test with nested structures
    nested = '{"a": {"b": [1, 2, {"c": "d"}]}}'
    expected = {"a": {"b": [1, 2, {"c": "d"}]}}
    assert serializer.loads(nested) == expected
    
    # Test that invalid JSON raises an error
    import json as json_module
    with pytest.raises(json_module.JSONDecodeError):
        serializer.loads("invalid json")
    
    # Test with empty string
    with pytest.raises(json_module.JSONDecodeError):
        serializer.loads("")
```


# LLM-generated content at query #25
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a concrete implementation of _PDataSerializer for testing
    class TestSerializer:
        def loads(self, payload):
            if isinstance(payload, str):
                return json.loads(payload)
            elif isinstance(payload, bytes):
                return json.loads(payload.decode('utf-8'))
            return payload

        def dumps(self, obj):
            return json.dumps(obj)

    serializer = TestSerializer()
    
    # Test with string input
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes input
    result = serializer.loads(b'{"number": 42}')
    assert result == {"number": 42}
    
    # Test with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with primitive values
    result = serializer.loads('"string"')
    assert result == "string"
    
    result = serializer.loads('123')
    assert result == 123
    
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('null')
    assert result is None
    
    # Test with empty inputs
    result = serializer.loads('{}')
    assert result == {}
    
    result = serializer.loads('[]')
    assert result == []
```


# LLM-generated content at query #26
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.count(".") == 2  # payload, timestamp, signature
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode()
        def loads(self, payload):
            return json.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result.count(b".") == 2
    
    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert result_with_salt != result  # Different salt produces different signature
    
    # Test with serializer_kwargs
    class CustomSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj, **kwargs)
        def loads(self, payload):
            return json.loads(payload)
    
    custom_serializer = Serializer(
        "secret-key",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result = custom_serializer.dumps({"b": 1, "a": 2})
    assert isinstance(result, str)
    
    # Test with multiple secret keys (key rotation)
    multi_key_serializer = Serializer(["old-key", "new-key"])
    result = multi_key_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    # The newest key should be used for signing
    signer = multi_key_serializer.make_signer()
    payload = multi_key_serializer.dump_payload({"key": "value"})
    assert signer.sign(payload).decode() in result
    
    # Test empty payload
    result = serializer.dumps({})
    assert isinstance(result, str)
    assert result.count(".") == 2
    
    # Test None payload
    result = serializer.dumps(None)
    assert isinstance(result, str)
    assert result.count(".") == 2
    
    # Test list payload
    result = serializer.dumps([1, 2, 3])
    assert isinstance(result, str)
    assert result.count(".") == 2
    
    # Test that dumps returns bytes when serializer returns bytes
    class BytesOnlySerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode()
        def loads(self, payload):
            return json.loads(payload)
    
    bytes_ser = Serializer("secret", serializer=BytesOnlySerializer())
    result = bytes_ser.dumps({"test": "data"})
    assert isinstance(result, bytes)
    
    # Test that dumps returns str when serializer returns str (default JSON)
    str_ser = Serializer("secret")
    result = str_ser.dumps({"test": "data"})
    assert isinstance(result, str)


# LLM-generated content at query #27
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works with various types."""
    # Create a concrete class that implements _PDataSerializer for str
    class StrSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    # Create a concrete class that implements _PDataSerializer for bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode("utf-8").upper()
        
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode("utf-8")
    
    # Test with str serializer
    str_serializer = StrSerializer()
    assert str_serializer.loads("hello") == "HELLO"
    assert str_serializer.loads("test") == "TEST"
    assert str_serializer.loads("") == ""
    
    # Test with bytes serializer
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b"hello") == "HELLO"
    assert bytes_serializer.loads(b"test") == "TEST"
    assert bytes_serializer.loads(b"") == ""
    
    # Test that the protocol type checking works at runtime
    assert is_text_serializer(str_serializer) == True
    assert is_text_serializer(bytes_serializer) == False
```


# LLM-generated content at query #28
#--------------------------

```python
def test_Serializer_dumps():
    """Test that dumps returns correctly signed and serialized data."""
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    
    result = serializer.dumps(data)
    
    # Should return a string (not bytes) since json serializer is text
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    bytes_result = bytes_serializer.dumps(data)
    
    # Should return bytes since serializer returns bytes
    assert isinstance(bytes_result, bytes)
    
    # Test that dumps works with different data types
    for test_data in [None, True, 42, 3.14, "string", [1, 2, 3], {"nested": {"a": 1}}]:
        result = serializer.dumps(test_data)
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Verify we can load it back
        loaded = serializer.loads(result)
        assert loaded == test_data
    
    # Test with custom serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "indent": 2}
    )
    result_with_kwargs = serializer_with_kwargs.dumps(data)
    assert isinstance(result_with_kwargs, str)
    
    # Verify the kwargs were applied (should have newlines due to indent)
    assert "\n" in result_with_kwargs


# LLM-generated content at query #29
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    serializer = StringSerializer()
    result = serializer.loads("hello")
    assert result == "HELLO"
    
    # Test with JSON serializer
    json_serializer = json
    result = json_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode("utf-8").upper()
        
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b"hello")
    assert result == "HELLO"
    
    # Test that loads returns any type
    class AnySerializer:
        def loads(self, payload: str) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    any_serializer = AnySerializer()
    assert any_serializer.loads("test") == "test"
    assert any_serializer.loads("123") == "123"
    
    # Test that loads uses positional-only parameter
    class PositionalSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any, /) -> str:
            return str(obj)
    
    pos_serializer = PositionalSerializer()
    result = pos_serializer.loads("data")
    assert result == "data"
```


# LLM-generated content at query #30
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any, /) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    serializer = BytesSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with a simple serializer that returns str
    class StrSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)
    
    serializer = StrSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with None value
    result = serializer.dumps(None)
    assert result == 'null'
    
    # Test with list value
    result = serializer.dumps([1, 2, 3])
    assert result == '[1, 2, 3]'
    
    # Test with integer value
    result = serializer.dumps(42)
    assert result == '42'
```


# LLM-generated content at query #31
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    
    # Should return a string (not bytes) since JSON is text serializer
    assert isinstance(result, str)
    
    # Verify the result contains the serialized payload
    # The result should be a signed JSON string
    assert "." in result  # Contains separator
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    
    # Should return bytes since BytesSerializer returns bytes
    assert isinstance(result_bytes, bytes)
    
    # Test with custom serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_kwargs = serializer_with_kwargs.dumps({"b": 1, "a": 2})
    assert isinstance(result_kwargs, str)
    
    # Test with salt parameter
    serializer_salt = Serializer("secret-key", salt="custom-salt")
    result_salt = serializer_salt.dumps({"key": "value"})
    assert isinstance(result_salt, str)
    
    # Test that different keys produce different signatures
    serializer1 = Serializer("key1")
    serializer2 = Serializer("key2")
    result1 = serializer1.dumps({"data": "test"})
    result2 = serializer2.dumps({"data": "test"})
    assert result1 != result2
    
    # Test with empty object
    result_empty = serializer.dumps({})
    assert isinstance(result_empty, str)
    
    # Test with list
    result_list = serializer.dumps([1, 2, 3])
    assert isinstance(result_list, str)
    
    # Test with None
    result_none = serializer.dumps(None)
    assert isinstance(result_none, str)
    
    # Test with string
    result_string = serializer.dumps("test string")
    assert isinstance(result_string, str)
    
    # Test with integer
    result_int = serializer.dumps(42)
    assert isinstance(result_int, str)
    
    # Test that the result can be loaded back correctly
    loaded = serializer.loads(result)
    assert loaded == {"key": "value"}
    
    # Verify that dumps and loads are inverses for different data types
    test_data = [
        {"nested": {"data": True}},
        [1, "two", 3.0],
        "simple string",
        12345,
        None,
        [{"complex": [1, 2, 3]}]
    ]
    
    for data in test_data:
        dumped = serializer.dumps(data)
        loaded = serializer.loads(dumped)
        assert loaded == data


# LLM-generated content at query #32
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol requires dumps method."""
    class TestSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return str(obj)
    
    serializer = TestSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == '{"key": "value"}'
    assert isinstance(result, str)


# LLM-generated content at query #33
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol requires dumps method."""
    # Test with a valid serializer that implements dumps
    class ValidSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = ValidSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == '{"key": "value"}'
    
    # Test that dumps returns the expected type (str in this case)
    assert isinstance(result, str)
    
    # Test with a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert result_bytes == b'{"key": "value"}'
    assert isinstance(result_bytes, bytes)
    
    # Test empty object serialization
    empty_result = serializer.dumps({})
    assert empty_result == '{}'
    
    # Test nested object serialization
    nested_result = serializer.dumps({"a": [1, 2, 3], "b": {"c": "d"}})
    assert nested_result == '{"a": [1, 2, 3], "b": {"c": "d"}}'
    
    # Test that the protocol is structural (duck typing)
    assert hasattr(serializer, 'dumps')
    assert callable(serializer.dumps)


# LLM-generated content at query #34
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test _PDataSerializer protocol loads method"""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str | bytes, /) -> t.Any:
            if isinstance(payload, str):
                return json.loads(payload)
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any, /) -> str | bytes:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with string payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes payload
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer
    result = serializer.loads('42')
    assert result == 42
    
    # Test with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with None
    result = serializer.loads('null')
    assert result is None
    
    # Test with boolean
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('false')
    assert result is False
    
    # Test with empty string/bytes
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('')
    
    with pytest.raises(json.JSONDecodeError):
        serializer.loads(b'')

    # Test protocol conformance - verify it's a _PDataSerializer
    assert isinstance(serializer, _PDataSerializer)
    
    # Test that loads is callable with positional argument only
    # (this should work since / marks positional-only parameters)
    result = serializer.loads('{"test": "data"}')
    assert result == {"test": "data"}
    
    # Verify that loads doesn't accept keyword arguments
    with pytest.raises(TypeError):
        serializer.loads(payload='{}')  # type: ignore[call-arg]
```


# LLM-generated content at query #35
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns strings
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StringSerializer()
    serializer_protocol: _PDataSerializer[str] = serializer
    
    # Test loading a JSON string
    result = serializer_protocol.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = serializer_protocol.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading a simple value
    result = serializer_protocol.loads('"hello"')
    assert result == "hello"
    
    # Test loading a number
    result = serializer_protocol.loads('42')
    assert result == 42
    
    # Test loading null
    result = serializer_protocol.loads('null')
    assert result is None
    
    # Test loading boolean
    result = serializer_protocol.loads('true')
    assert result is True
    
    result = serializer_protocol.loads('false')
    assert result is False
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    bytes_protocol: _PDataSerializer[bytes] = bytes_serializer
    
    # Test loading bytes payload
    result = bytes_protocol.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    result = bytes_protocol.loads(b'[1, 2, 3]')
    assert result == [1, 2, 3]
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a concrete implementation of _PDataSerializer for testing
    class TestSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with simple types
    assert serializer.loads('42') == 42
    assert serializer.loads('"hello"') == "hello"
    assert serializer.loads('true') == True
    assert serializer.loads('null') is None
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test with empty JSON
    assert serializer.loads('{}') == {}
    assert serializer.loads('[]') == []
    
    # Test with nested structures
    nested = {"a": [1, 2, {"b": "c"}]}
    assert serializer.loads(json.dumps(nested)) == nested
    
    # Test with invalid JSON should raise appropriate exception
    import pytest
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
    
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('')
    
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('{invalid}')
    
    # Test with bytes input (should still work if loads handles bytes)
    class BytesSerializer:
        def loads(self, payload):
            if isinstance(payload, bytes):
                payload = payload.decode('utf-8')
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'42') == 42
```


# LLM-generated content at query #2
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StringSerializer()
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with simple types
    class SimpleSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    simple_serializer = SimpleSerializer()
    result = simple_serializer.loads("test_string")
    assert result == "test_string"
    
    # Test with integer
    result = simple_serializer.loads("42")
    assert result == "42"
```


# LLM-generated content at query #3
#--------------------------

Here's a unit test for the `loads` method of the `_PDataSerializer` protocol:

```python
def test__PDataSerializer_loads():
    # Test with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    serializer = StringSerializer()
    result = serializer.loads("hello")
    assert result == "HELLO"
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode("utf-8").upper()
        
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b"world")
    assert result == "WORLD"
    
    # Test with JSON serializer
    import json
    json_serializer = json
    result = json_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that loads returns Any type
    result = serializer.loads("test")
    assert isinstance(result, str)
```


# LLM-generated content at query #4
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            if isinstance(payload, str):
                return json.loads(payload)
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with string payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes payload
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with different data types
    result = serializer.loads('123')
    assert result == 123
    
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test that it raises appropriate exception for invalid payload
    import pytest
    with pytest.raises(Exception):
        serializer.loads('invalid json')
    
    # Test with empty list
    result = serializer.loads('[]')
    assert result == []
    
    # Test with null
    result = serializer.loads('null')
    assert result is None
```


# LLM-generated content at query #5
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default signer and no fallback signers
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)

    # Test with custom salt
    custom_salt = b"custom-salt"
    signers = list(serializer.iter_unsigners(salt=custom_salt))
    assert len(signers) == 1
    assert signers[0].salt == custom_salt

    # Test with fallback signers as dict
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"key_derivation": "none"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2  # default + fallback
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

    # Test with fallback signers as tuple (signer class, kwargs)
    from itsdangerous.signer import HMACAlgorithm
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(custom_signer, {"digest_method": "sha256"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], custom_signer)

    # Test with multiple secret keys
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1  # only the main signer, not one per key

    # Test with fallback signers that are Signer classes
    serializer = Serializer(
        "secret-key",
        fallback_signers=[Signer]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert all(isinstance(s, Signer) for s in signers)

    # Test with multiple keys and fallback signers
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"key_derivation": "none"}]
    )
    signers = list(serializer.iter_unsigners())
    # 1 main signer + 2 fallback signers (one per key)
    assert len(signers) == 3
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert isinstance(signers[2], Signer)

    # Test that salt is passed correctly to all signers
    test_salt = b"test-salt"
    serializer = Serializer(
        "secret-key",
        fallback_signers=[Signer]
    )
    signers = list(serializer.iter_unsigners(salt=test_salt))
    assert len(signers) == 2
    for signer in signers:
        assert signer.salt == test_salt

    # Test with empty fallback signers list
    serializer = Serializer(
        "secret-key",
        fallback_signers=[]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1

    # Test with fallback signer that has different kwargs
    serializer = Serializer(
        "secret-key",
        salt=b"default-salt",
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].key_derivation == "hmac"  # default signer
    assert signers[1].key_derivation == "none"   # fallback signer
```


# LLM-generated content at query #6
#--------------------------

```python
def test_Serializer_load_payload():
    """Test Serializer.load_payload method."""
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test loading valid JSON payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading valid JSON payload with explicit serializer
    result = serializer.load_payload(payload, serializer=json)
    assert result == {"key": "value"}
    
    # Test loading with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return obj
    
    bytes_serializer = BytesSerializer()
    bytes_ser = Serializer("secret-key", serializer=bytes_serializer)
    payload = b"test bytes payload"
    result = bytes_ser.load_payload(payload)
    assert result == payload
    
    # Test loading with bytes serializer using explicit serializer parameter
    result = bytes_ser.load_payload(payload, serializer=bytes_serializer)
    assert result == payload
    
    # Test loading invalid payload raises BadPayload
    import pytest
    from .exc import BadPayload
    
    invalid_payload = b"not valid json"
    with pytest.raises(BadPayload):
        serializer.load_payload(invalid_payload)
    
    # Test loading empty payload
    empty_payload = b""
    with pytest.raises(BadPayload):
        serializer.load_payload(empty_payload)
    
    # Test loading with custom text serializer that returns something specific
    class CustomTextSerializer:
        def loads(self, payload):
            if payload == "custom":
                return {"custom": "data"}
            raise ValueError("Invalid")
        
        def dumps(self, obj):
            return "custom"
    
    custom_serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    result = custom_serializer.load_payload(b"custom")
    assert result == {"custom": "data"}
    
    # Test that BadPayload preserves original error
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"invalid")
    assert exc_info.value.original_error is not None
    
    # Test loading with non-text serializer that raises exception
    class ErrorSerializer:
        def loads(self, payload):
            raise RuntimeError("Test error")
        
        def dumps(self, obj):
            return b"test"
    
    error_ser = Serializer("secret-key", serializer=ErrorSerializer())
    with pytest.raises(BadPayload) as exc_info:
        error_ser.load_payload(b"test")
    assert isinstance(exc_info.value.original_error, RuntimeError)
```


# LLM-generated content at query #7
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns strings
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StringSerializer()
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer payload
    result = serializer.loads('42')
    assert result == 42
    
    # Test with list payload
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with None payload
    result = serializer.loads('null')
    assert result is None
    
    # Test with boolean payload
    result = serializer.loads('true')
    assert result is True
    
    # Test with a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with invalid JSON
    import json
    try:
        serializer.loads('invalid json')
        assert False, "Should have raised JSONDecodeError"
    except json.JSONDecodeError:
        pass
    
    # Test with empty string
    try:
        serializer.loads('')
        assert False, "Should have raised JSONDecodeError"
    except json.JSONDecodeError:
        pass
```


# LLM-generated content at query #8
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default json serializer and text payload
    serializer = Serializer("test-secret")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes):
            return {"data": payload.decode()}
        
        def dumps(self, obj):
            return b"test"

    bytes_serializer = Serializer("test-secret", serializer=BytesSerializer())
    bytes_payload = b"hello"
    result = bytes_serializer.load_payload(bytes_payload)
    assert result == {"data": "hello"}

    # Test with custom serializer that returns text
    class TextSerializer:
        def loads(self, payload: str):
            return {"data": payload}
        
        def dumps(self, obj):
            return "test"

    text_serializer = Serializer("test-secret", serializer=TextSerializer())
    text_payload = b"world"
    result = text_serializer.load_payload(text_payload)
    assert result == {"data": "world"}

    # Test with custom serializer passed as parameter
    class CustomSerializer:
        def loads(self, payload: bytes):
            return {"custom": payload.decode()}
        
        def dumps(self, obj):
            return b"test"

    serializer = Serializer("test-secret")
    custom_serializer = CustomSerializer()
    result = serializer.load_payload(b"test", serializer=custom_serializer)
    assert result == {"custom": "test"}

    # Test with invalid payload that raises BadPayload
    import pytest
    serializer = Serializer("test-secret")
    with pytest.raises(BadPayload):
        serializer.load_payload(b"invalid json")

    # Test with empty payload
    serializer = Serializer("test-secret")
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")

    # Test with non-UTF8 bytes for text serializer
    serializer = Serializer("test-secret")
    with pytest.raises(BadPayload):
        serializer.load_payload(b"\xff\xfe")

    # Test that BadPayload has original_error attribute
    try:
        serializer.load_payload(b"invalid")
    except BadPayload as e:
        assert e.original_error is not None
```


# LLM-generated content at query #9
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol supports dumps method."""
    # Create a concrete implementation that follows the _PDataSerializer protocol
    class TestSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test basic dumps functionality
    test_data = {"key": "value", "number": 42}
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    assert result == json.dumps(test_data)
    
    # Test with list data
    list_data = [1, 2, 3, "test"]
    result = serializer.dumps(list_data)
    assert isinstance(result, str)
    assert result == json.dumps(list_data)
    
    # Test with simple data types
    assert serializer.dumps("string") == json.dumps("string")
    assert serializer.dumps(123) == json.dumps(123)
    assert serializer.dumps(True) == json.dumps(True)
    assert serializer.dumps(None) == json.dumps(None)
    
    # Test that dumps returns bytes when using bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any, /) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"data": "test"})
    assert isinstance(result, bytes)
    assert result == b'{"data": "test"}' 


# LLM-generated content at query #10
#--------------------------

```python
def test_Serializer_iter_unsigners():
    secret_key = b"secret-key"
    serializer = Serializer(secret_key)

    # Test with default salt
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_key == b"secret-key"
    assert signers[0].salt == b"itsdangerous"

    # Test with custom salt
    custom_salt = b"custom-salt"
    signers = list(serializer.iter_unsigners(salt=custom_salt))
    assert len(signers) == 1
    assert signers[0].salt == custom_salt

    # Test with fallback signers as dict
    serializer_with_fallback = Serializer(
        secret_key,
        fallback_signers=[{"key": b"fallback-key"}]
    )
    signers = list(serializer_with_fallback.iter_unsigners())
    assert len(signers) == 2  # default + fallback
    assert signers[0].secret_key == b"secret-key"
    assert signers[1].secret_key == b"fallback-key"

    # Test with fallback signers as tuple
    serializer_with_fallback_tuple = Serializer(
        secret_key,
        fallback_signers=[(Signer, {"key": b"fallback-key"})]
    )
    signers = list(serializer_with_fallback_tuple.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].secret_key == b"secret-key"
    assert signers[1].secret_key == b"fallback-key"

    # Test with fallback signers as Signer class directly (not a dict/tuple)
    serializer_with_fallback_class = Serializer(
        secret_key,
        fallback_signers=[Signer]
    )
    signers = list(serializer_with_fallback_class.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].secret_key == b"secret-key"
    assert signers[1].secret_key == b"secret-key"  # uses same secret key

    # Test with multiple secret keys
    multiple_keys_serializer = Serializer(
        [b"old-key", b"new-key"],
        fallback_signers=[{"key": b"fallback-key"}]
    )
    signers = list(multiple_keys_serializer.iter_unsigners())
    assert len(signers) == 3  # default + 2 keys from fallback
    assert signers[0].secret_key == b"new-key"
    assert signers[1].secret_key == b"old-key"
    assert signers[2].secret_key == b"fallback-key"

    # Test that salt is passed correctly to fallback signers
    custom_salt_fallback = Serializer(
        secret_key,
        salt=b"custom-salt",
        fallback_signers=[{"key": b"fallback-key"}]
    )
    signers = list(custom_salt_fallback.iter_unsigners(salt=b"override-salt"))
    for signer in signers:
        assert signer.salt == b"override-salt"
```


# LLM-generated content at query #11
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("test-secret")
    
    # Test successful load with text serializer
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with custom bytes serializer
    class BytesSerializer:
        @staticmethod
        def loads(payload):
            return payload
        
        @staticmethod
        def dumps(obj):
            return obj
    
    bytes_serializer = Serializer(
        "test-secret",
        serializer=BytesSerializer()
    )
    payload = b"test_data"
    result = bytes_serializer.load_payload(payload)
    assert result == b"test_data"
    
    # Test with explicit serializer parameter
    class CustomSerializer:
        @staticmethod
        def loads(payload):
            return f"custom_{payload}"
        
        @staticmethod
        def dumps(obj):
            return str(obj)
    
    custom_serializer = CustomSerializer()
    payload = b"hello"
    result = serializer.load_payload(payload, serializer=custom_serializer)
    assert result == "custom_hello"
    
    # Test that BadPayload is raised on invalid data
    with pytest.raises(BadPayload):
        serializer.load_payload(b"invalid json")
    
    # Test that original_error is set
    try:
        serializer.load_payload(b"invalid json")
    except BadPayload as e:
        assert isinstance(e.original_error, (json.JSONDecodeError, Exception))
    
    # Test with empty payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test with bytes payload that decodes to valid JSON
    payload = b'{"number": 42, "list": [1, 2, 3]}'
    result = serializer.load_payload(payload)
    assert result == {"number": 42, "list": [1, 2, 3]}
```


# LLM-generated content at query #12
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    payload_data = {"key": "value"}
    json_payload = json.dumps(payload_data).encode("utf-8")
    result = serializer.load_payload(json_payload)
    assert result == payload_data

    # Test with custom bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return payload

        def dumps(self, obj):
            return obj

    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    bytes_payload = b"test bytes data"
    result = bytes_serializer.load_payload(bytes_payload)
    assert result == bytes_payload

    # Test with custom text serializer
    class TextSerializer:
        def loads(self, payload):
            return payload.upper()

        def dumps(self, obj):
            return obj.lower()

    text_serializer = Serializer("secret-key", serializer=TextSerializer())
    text_payload = "hello".encode("utf-8")
    result = text_serializer.load_payload(text_payload)
    assert result == "HELLO"

    # Test that BadPayload is raised for invalid data
    import pytest
    from itsdangerous.exc import BadPayload

    with pytest.raises(BadPayload):
        serializer.load_payload(b"invalid json")

    # Test with explicit serializer parameter
    class ExplicitSerializer:
        def loads(self, payload):
            return f"explicit: {payload}"

        def dumps(self, obj):
            return obj

    explicit_result = serializer.load_payload(
        b"test", serializer=ExplicitSerializer()
    )
    assert explicit_result == "explicit: test"
```


# LLM-generated content at query #13
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StringSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps(data)
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with empty data
    empty_result = serializer.dumps({})
    assert empty_result == "{}"
    
    # Test with nested data
    nested_data = {"a": [1, 2, 3], "b": {"c": "d"}}
    nested_result = serializer.dumps(nested_data)
    assert nested_result == '{"a": [1, 2, 3], "b": {"c": "d"}}'
    
    # Test with simple types
    int_result = serializer.dumps(42)
    assert int_result == "42"
    
    str_result = serializer.dumps("hello")
    assert str_result == '"hello"'
    
    list_result = serializer.dumps([1, "two", 3.0])
    assert list_result == '[1, "two", 3.0]'
```


# LLM-generated content at query #14
#--------------------------

```python
def test_Serializer_dumps():
    serializer = Serializer("test-secret-key")
    
    # Test dumps with a simple JSON-serializable object
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Should contain separator between payload and signature
    
    # Test dumps returns bytes when using binary serializer
    class BinarySerializer:
        def dumps(self, obj):
            import pickle
            return pickle.dumps(obj)
        def loads(self, payload):
            import pickle
            return pickle.loads(payload)
    
    bin_serializer = Serializer("test-secret-key", serializer=BinarySerializer())
    bin_result = bin_serializer.dumps({"key": "value"})
    assert isinstance(bin_result, bytes)
    
    # Test dumps with custom salt
    result_with_salt = serializer.dumps("test", salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert result_with_salt != serializer.dumps("test")  # Different salt should produce different output
    
    # Test that dumps produces consistent results with same input
    result1 = serializer.dumps("test")
    result2 = serializer.dumps("test")
    assert result1 == result2
    
    # Test dumps with various data types
    assert isinstance(serializer.dumps(123), str)
    assert isinstance(serializer.dumps([1, 2, 3]), str)
    assert isinstance(serializer.dumps(None), str)
    assert isinstance(serializer.dumps(True), str)
```


# LLM-generated content at query #15
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            if isinstance(payload, str):
                return {"data": payload}
            raise TypeError("Expected string payload")
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with string payload
    result = serializer.loads('{"data": "test"}')
    assert result == {"data": '{"data": "test"}'}
    
    # Test with different string payload
    result = serializer.loads("hello")
    assert result == {"data": "hello"}
    
    # Test with empty string
    result = serializer.loads("")
    assert result == {"data": ""}
```


# LLM-generated content at query #16
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer loads method works correctly."""
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test loading valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading primitive types
    assert serializer.loads('"hello"') == "hello"
    assert serializer.loads('42') == 42
    assert serializer.loads('true') is True
    assert serializer.loads('null') is None
    
    # Test that it raises exception for invalid JSON
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
    
    # Test that it works with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Verify the protocol is satisfied (type checking would pass)
    assert isinstance(serializer, _PDataSerializer)
```


# LLM-generated content at query #17
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer loads method handles payload correctly."""
    # Create a concrete implementation of _PDataSerializer
    class MockSerializer:
        def loads(self, payload):
            if payload == b'{"key": "value"}':
                return {"key": "value"}
            raise ValueError("Invalid payload")
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test successful loads
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with different payload
    result = serializer.loads(b'{"a": 1, "b": 2}')
    assert result == {"a": 1, "b": 2}
    
    # Test that it raises exception for invalid payload
    import pytest
    with pytest.raises(ValueError, match="Invalid payload"):
        serializer.loads(b'invalid')
```


# LLM-generated content at query #18
#--------------------------

```python
def test_Serializer_load_payload():
    # Create serializer instances for testing
    serializer = Serializer("test-secret")
    
    # Test 1: Basic payload loading with default JSON serializer
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test 2: Payload with nested structures
    payload = b'{"numbers": [1, 2, 3], "nested": {"a": 1}}'
    result = serializer.load_payload(payload)
    assert result == {"numbers": [1, 2, 3], "nested": {"a": 1}}
    
    # Test 3: Payload with simple data types
    payload = b'"string"'
    result = serializer.load_payload(payload)
    assert result == "string"
    
    payload = b'42'
    result = serializer.load_payload(payload)
    assert result == 42
    
    payload = b'true'
    result = serializer.load_payload(payload)
    assert result == True
    
    payload = b'null'
    result = serializer.load_payload(payload)
    assert result is None
    
    # Test 4: Empty payload
    payload = b'{}'
    result = serializer.load_payload(payload)
    assert result == {}
    
    # Test 5: Payload with list
    payload = b'[1, 2, 3]'
    result = serializer.load_payload(payload)
    assert result == [1, 2, 3]
    
    # Test 6: Invalid JSON payload raises BadPayload
    payload = b'invalid json'
    with pytest.raises(BadPayload):
        serializer.load_payload(payload)
    
    # Test 7: Empty bytes payload raises BadPayload
    payload = b''
    with pytest.raises(BadPayload):
        serializer.load_payload(payload)
    
    # Test 8: Invalid UTF-8 bytes payload
    payload = b'\xff\xfe\x00\x00'
    with pytest.raises(BadPayload):
        serializer.load_payload(payload)
    
    # Test 9: Custom serializer that returns text
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)
        
        def loads(self, payload):
            return eval(payload)
    
    text_serializer = Serializer("test-secret", serializer=TextSerializer())
    
    # Test with text serializer (converts bytes to string first)
    payload = b'[1, 2, 3]'
    result = text_serializer.load_payload(payload)
    assert result == [1, 2, 3]
    
    # Test 10: Custom serializer that returns bytes
    class BytesSerializer:
        def dumps(self, obj):
            return bytes(obj)
        
        def loads(self, payload):
            return list(payload)
    
    bytes_serializer = Serializer("test-secret", serializer=BytesSerializer())
    
    # Test with bytes serializer
    payload = b'\x01\x02\x03'
    result = bytes_serializer.load_payload(payload)
    assert result == [1, 2, 3]
    
    # Test 11: Override serializer parameter in load_payload
    payload = b'{"different": "serializer"}'
    result = serializer.load_payload(payload, serializer=json)
    assert result == {"different": "serializer"}
    
    # Test 12: Override with custom serializer that fails
    class FailingSerializer:
        def dumps(self, obj):
            return "test"
        
        def loads(self, payload):
            raise ValueError("Serialization error")
    
    failing_serializer = Serializer("test-secret", serializer=FailingSerializer())
    payload = b'some data'
    with pytest.raises(BadPayload, match="Could not load the payload"):
        failing_serializer.load_payload(payload)
    
    # Test 13: Verify original_error is preserved in BadPayload
    try:
        serializer.load_payload(b'invalid')
    except BadPayload as e:
        assert isinstance(e.original_error, (json.JSONDecodeError, Exception))
    
    # Test 14: Unicode payload with special characters
    payload = '{"unicode": "测试"}'.encode('utf-8')
    result = serializer.load_payload(payload)
    assert result == {"unicode": "测试"}
```


# LLM-generated content at query #19
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer.loads properly handles the payload parameter."""
    # Create a concrete implementation that satisfies the protocol
    class TestSerializer:
        def loads(self, payload: str | bytes) -> t.Any:
            if isinstance(payload, str):
                return json.loads(payload)
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> str | bytes:
            return json.dumps(obj)

    serializer = TestSerializer()
    
    # Test with string payload
    result = serializer.loads('{"key": "value"}')
    assert isinstance(result, dict)
    assert result["key"] == "value"
    
    # Test with bytes payload
    result = serializer.loads(b'{"key": "value"}')
    assert isinstance(result, dict)
    assert result["key"] == "value"
    
    # Test with integer payload
    result = serializer.loads("42")
    assert result == 42
    
    # Test with list payload
    result = serializer.loads("[1, 2, 3]")
    assert result == [1, 2, 3]
    
    # Test with null payload
    result = serializer.loads("null")
    assert result is None
    
    # Test with boolean payload
    result = serializer.loads("true")
    assert result is True
    
    # Test with invalid JSON payload
    with pytest.raises(json.JSONDecodeError):
        serializer.loads("invalid json")
```


# LLM-generated content at query #20
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol defines dumps method correctly."""
    import json
    
    # Create a concrete serializer that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test that dumps returns a string
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with different data types
    assert serializer.dumps(123) == "123"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps(None) == "null"
    assert serializer.dumps(True) == "true"
    
    # Test with empty data
    assert serializer.dumps({}) == "{}"
    assert serializer.dumps([]) == "[]"
    
    # Test that dumps is callable with just one argument (positional only)
    result = serializer.dumps("test")
    assert result == '"test"'


# LLM-generated content at query #21
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert serializer.loads(result) == {"key": "value"}

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        
        def loads(self, payload):
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            return json.loads(payload)

    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert bytes_serializer.loads(result) == {"key": "value"}

    # Test with custom salt
    serializer_with_salt = Serializer("secret-key", salt="custom-salt")
    result1 = serializer.dumps({"data": 1}, salt="custom-salt")
    result2 = serializer_with_salt.dumps({"data": 1})
    assert result1 == result2

    # Test with empty data
    result = serializer.dumps({})
    assert serializer.loads(result) == {}

    # Test with None value
    result = serializer.dumps(None)
    assert serializer.loads(result) is None

    # Test with list data
    result = serializer.dumps([1, 2, 3])
    assert serializer.loads(result) == [1, 2, 3]

    # Test with nested data
    result = serializer.dumps({"nested": {"list": [1, 2, 3]}})
    assert serializer.loads(result) == {"nested": {"list": [1, 2, 3]}}

    # Test that dumps is deterministic
    result1 = serializer.dumps({"test": "data"})
    result2 = serializer.dumps({"test": "data"})
    assert result1 == result2

    # Test that different keys produce different signatures
    serializer2 = Serializer("different-secret-key")
    result1 = serializer.dumps({"test": "data"})
    result2 = serializer2.dumps({"test": "data"})
    assert result1 != result2

    # Test with serializer_kwargs
    class SerializerWithKwargs:
        def dumps(self, obj, **kwargs):
            if kwargs.get("sort_keys"):
                return json.dumps(obj, sort_keys=True)
            return json.dumps(obj)
        
        def loads(self, payload):
            return json.loads(payload)

    kwargs_serializer = Serializer(
        "secret-key",
        serializer=SerializerWithKwargs(),
        serializer_kwargs={"sort_keys": True}
    )
    result = kwargs_serializer.dumps({"b": 1, "a": 2})
    assert b'"a"' in result if isinstance(result, bytes) else '"a"' in result
    assert kwargs_serializer.loads(result) == {"b": 1, "a": 2}
```


# LLM-generated content at query #22
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol has a dumps method that works correctly."""
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    test_data = {"key": "value", "number": 42}
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    assert json.loads(result) == test_data
    
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    result_bytes = bytes_serializer.dumps(test_data)
    assert isinstance(result_bytes, bytes)
    assert json.loads(result_bytes) == test_data


# LLM-generated content at query #23
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a simple serializer that implements the _PDataSerializer protocol
    class TestSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with various data types
    test_cases = [
        {"key": "value"},
        [1, 2, 3],
        "string",
        42,
        None,
        {"nested": {"data": [1, 2, 3]}}
    ]
    
    for obj in test_cases:
        result = serializer.dumps(obj)
        assert isinstance(result, str), f"dumps should return str, got {type(result)}"
        assert serializer.loads(result) == obj, f"Round-trip failed for {obj}"
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj):
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    
    for obj in test_cases:
        result = bytes_serializer.dumps(obj)
        assert isinstance(result, bytes), f"dumps should return bytes, got {type(result)}"
        assert bytes_serializer.loads(result) == obj, f"Round-trip failed for {obj}"
    
    # Verify is_text_serializer works correctly
    assert is_text_serializer(TestSerializer()) == True
    assert is_text_serializer(BytesSerializer()) == False


# LLM-generated content at query #24
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a simple serializer that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test basic serialization
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with various data types
    assert serializer.dumps(None) == "null"
    assert serializer.dumps(True) == "true"
    assert serializer.dumps(42) == "42"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    
    # Test roundtrip
    original = {"a": 1, "b": [2, 3], "c": {"d": "e"}}
    dumped = serializer.dumps(original)
    loaded = serializer.loads(dumped)
    assert loaded == original


# LLM-generated content at query #25
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a mock serializer that implements the protocol
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with simple values
    assert serializer.loads('"test"') == "test"
    assert serializer.loads('123') == 123
    assert serializer.loads('true') == True
    assert serializer.loads('null') is None
    
    # Test with array
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test with empty objects
    assert serializer.loads('{}') == {}
    assert serializer.loads('[]') == []
    
    # Test that invalid JSON raises exception
    with pytest.raises(Exception):
        serializer.loads('invalid json')
    
    # Test with different serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    
    # Test type checking - loads should accept _TSerialized type
    assert isinstance(serializer.loads('"test"'), str)
```


# LLM-generated content at query #26
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation that matches the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test basic JSON loading
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading primitive types
    result = serializer.loads('"string"')
    assert result == "string"
    
    result = serializer.loads('42')
    assert result == 42
    
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('null')
    assert result is None
    
    # Test with bytes protocol implementation
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that invalid JSON raises appropriate error
    import pytest
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
```


# LLM-generated content at query #27
#--------------------------

```python
def test_Serializer_dumps():
    """Test that dumps returns a signed string serialized with the internal serializer."""
    # Test with default json serializer (text serializer)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
    # Verify we can load it back
    assert s.loads(result) == {"key": "value"}

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))

    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = s_bytes.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert s_bytes.loads(result_bytes) == {"key": "value"}

    # Test with custom salt
    s_custom_salt = Serializer("secret-key", salt="custom-salt")
    result_custom = s_custom_salt.dumps({"key": "value"})
    assert isinstance(result_custom, str)
    assert s_custom_salt.loads(result_custom, salt="custom-salt") == {"key": "value"}

    # Test with serializer_kwargs
    s_kwargs = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    result_kwargs = s_kwargs.dumps({"b": 2, "a": 1})
    assert isinstance(result_kwargs, str)
    # The sorted version should have "a" before "b"
    assert s_kwargs.loads(result_kwargs) == {"b": 2, "a": 1}

    # Test with empty data
    s_empty = Serializer("secret-key")
    result_empty = s_empty.dumps({})
    assert isinstance(result_empty, str)
    assert s_empty.loads(result_empty) == {}

    # Test with list data
    s_list = Serializer("secret-key")
    result_list = s_list.dumps([1, 2, 3])
    assert isinstance(result_list, str)
    assert s_list.loads(result_list) == [1, 2, 3]

    # Test with string data
    s_string = Serializer("secret-key")
    result_string = s_string.dumps("test string")
    assert isinstance(result_string, str)
    assert s_string.loads(result_string) == "test string"

    # Test with numeric data
    s_num = Serializer("secret-key")
    result_num = s_num.dumps(42)
    assert isinstance(result_num, str)
    assert s_num.loads(result_num) == 42

    # Test with None data
    s_none = Serializer("secret-key")
    result_none = s_none.dumps(None)
    assert isinstance(result_none, str)
    assert s_none.loads(result_none) is None

    # Test with boolean data
    s_bool = Serializer("secret-key")
    result_true = s_bool.dumps(True)
    result_false = s_bool.dumps(False)
    assert s_bool.loads(result_true) is True
    assert s_bool.loads(result_false) is False

    # Test with multiple secret keys (key rotation)
    s_rotation = Serializer(["old-key", "new-key"])
    result_rotation = s_rotation.dumps({"key": "value"})
    assert isinstance(result_rotation, str)
    # Should be able to unsign with the same serializer
    assert s_rotation.loads(result_rotation) == {"key": "value"}
    # Should also be able to unsign with just the new key
    s_new_only = Serializer("new-key")
    assert s_new_only.loads(result_rotation) == {"key": "value"}
```


# LLM-generated content at query #28
#--------------------------

```python
def test__PDataSerializer_dumps():
    class MockSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return str(obj)
    
    serializer = MockSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == "{'key': 'value'}"
```


# LLM-generated content at query #29
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol loads method works correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any, /) -> bytes:
            return json.dumps(obj).encode()

    serializer = TestSerializer()
    
    # Test with valid JSON data
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with list data
    result = serializer.loads(b'[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with primitive types
    result = serializer.loads(b'"string"')
    assert result == "string"
    
    result = serializer.loads(b'42')
    assert result == 42
    
    result = serializer.loads(b'null')
    assert result is None
    
    result = serializer.loads(b'true')
    assert result is True
    
    result = serializer.loads(b'false')
    assert result is False
    
    # Test with empty data
    result = serializer.loads(b'{}')
    assert result == {}
    
    result = serializer.loads(b'[]')
    assert result == []
    
    # Test with nested data
    result = serializer.loads(b'{"a": {"b": [1, 2, 3]}}')
    assert result == {"a": {"b": [1, 2, 3]}}
    
    # Test with bytes payload (not str)
    result = serializer.loads(b'"unicode text"')
    assert result == "unicode text"
    
    # Test that invalid JSON raises appropriate exception
    with pytest.raises(json.JSONDecodeError):
        serializer.loads(b'invalid json')
    
    with pytest.raises(json.JSONDecodeError):
        serializer.loads(b'')
    
    # Test with a serializer that returns str
    class TextSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)

    text_serializer = TextSerializer()
    result = text_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that the method signature accepts only positional argument (no keyword)
    with pytest.raises(TypeError):
        serializer.loads(payload=b'test')
```


# LLM-generated content at query #30
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test successful load with text serializer
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes):
            return {"key": "value"}
        def dumps(self, obj):
            return b'{"key": "value"}'
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    
    # Test successful load with bytes serializer
    result = bytes_serializer.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with override serializer parameter
    class CustomSerializer:
        def loads(self, payload: str):
            return {"custom": True}
        def dumps(self, obj):
            return '{"custom": true}'
    
    custom_serializer = CustomSerializer()
    result = serializer.load_payload(b'{"test": "data"}', serializer=custom_serializer)
    assert result == {"custom": True}
    
    # Test BadPayload exception with invalid json
    import pytest
    from itsdangerous.exc import BadPayload
    
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"invalid json")
    assert "Could not load the payload" in str(exc_info.value)
    assert exc_info.value.original_error is not None
    
    # Test BadPayload exception with empty payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test BadPayload exception with non-utf8 bytes for text serializer
    with pytest.raises(BadPayload):
        serializer.load_payload(b"\xff\xfe\x00\x00")
    
    # Test with bytes serializer and invalid data
    with pytest.raises(BadPayload):
        bytes_serializer.load_payload(b"\xff\xfe")
    
    # Test that original_error is set correctly
    try:
        serializer.load_payload(b"not json")
    except BadPayload as e:
        assert isinstance(e.original_error, json.JSONDecodeError)
```


# LLM-generated content at query #31
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    string_serializer = StringSerializer()
    test_data = {"key": "value", "number": 42}
    result = string_serializer.dumps(test_data)
    assert isinstance(result, str)
    assert result == json.dumps(test_data)
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    test_data = {"key": "value", "number": 42}
    result = bytes_serializer.dumps(test_data)
    assert isinstance(result, bytes)
    assert result == json.dumps(test_data).encode("utf-8")
    
    # Test with a serializer that returns different types
    class IntSerializer:
        def loads(self, payload: int) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any) -> int:
            return int(obj)
    
    int_serializer = IntSerializer()
    result = int_serializer.dumps(42)
    assert isinstance(result, int)
    assert result == 42
    
    # Test with default JSON serializer
    json_serializer = json
    result = json_serializer.dumps({"test": "data"})
    assert isinstance(result, str)
    assert result == '{"test": "data"}'


# LLM-generated content at query #32
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test _PDataSerializer protocol's dumps method."""
    
    # Create a concrete implementation that follows the _PDataSerializer protocol
    class TestSerializer:
        def loads(self, payload: str | bytes, /) -> t.Any:
            return json.loads(payload) if isinstance(payload, str) else json.loads(payload.decode())
        
        def dumps(self, obj: t.Any, /) -> str | bytes:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with various data types
    test_cases = [
        {"key": "value"},
        [1, 2, 3],
        "simple string",
        42,
        None,
        {"nested": {"data": True}},
    ]
    
    for data in test_cases:
        result = serializer.dumps(data)
        assert isinstance(result, (str, bytes)), f"dumps should return str or bytes, got {type(result)}"
        # Verify round-trip
        loaded = serializer.loads(result)
        assert loaded == data, f"Round-trip failed for {data}: got {loaded}"
    
    # Test with empty data
    empty_cases = [
        {},
        [],
        "",
        0,
        False,
    ]
    
    for data in empty_cases:
        result = serializer.dumps(data)
        assert isinstance(result, (str, bytes)), f"dumps should return str or bytes for empty data, got {type(result)}"
        loaded = serializer.loads(result)
        assert loaded == data, f"Round-trip failed for empty data {data}: got {loaded}"
    
    # Test that dumps is callable and returns expected type
    result = serializer.dumps({"test": "data"})
    assert callable(serializer.dumps), "dumps should be callable"
    assert isinstance(result, str), "JSON dumps should return str by default"
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: str | bytes, /) -> t.Any:
            return json.loads(payload if isinstance(payload, str) else payload.decode())
        
        def dumps(self, obj: t.Any, /) -> str | bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    bytes_result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(bytes_result, bytes), "Bytes serializer should return bytes"
    loaded = bytes_serializer.loads(bytes_result)
    assert loaded == {"key": "value"}, "Round-trip failed for bytes serializer"


# LLM-generated content at query #33
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns str
    class TestStrSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestStrSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with a serializer that returns bytes
    class TestBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    serializer = TestBytesSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with None value
    result = TestStrSerializer().dumps(None)
    assert result == "null"
    
    # Test with list
    result = TestStrSerializer().dumps([1, 2, 3])
    assert result == "[1, 2, 3]"
    
    # Test with integer
    result = TestStrSerializer().dumps(42)
    assert result == "42"
    
    # Test with string
    result = TestStrSerializer().dumps("hello")
    assert result == '"hello"'
```


# LLM-generated content at query #34
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON serializer
    serializer = json
    pdata_serializer = _PDataSerializer[str]()
    
    # Test that the protocol accepts json as a valid implementation
    json_payload = '{"key": "value"}'
    result = serializer.loads(json_payload)
    assert result == {"key": "value"}
    
    # Test with a custom serializer that implements the protocol
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"custom": True, "data": payload}
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    custom_serializer = CustomSerializer()
    custom_result = custom_serializer.loads("test_data")
    assert custom_result == {"custom": True, "data": "test_data"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return {"bytes_data": payload}
        
        def dumps(self, obj: t.Any) -> bytes:
            return b"test"
    
    bytes_serializer = BytesSerializer()
    bytes_result = bytes_serializer.loads(b"binary_data")
    assert bytes_result == {"bytes_data": b"binary_data"}
```


# LLM-generated content at query #35
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test successful payload loading
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload):
            return {"from_bytes": payload}
        def dumps(self, obj):
            return b'{"test": "data"}'
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    payload = b'some bytes payload'
    result = bytes_serializer.load_payload(payload)
    assert result == {"from_bytes": payload}
    
    # Test with invalid payload (raises BadPayload)
    invalid_payload = b'invalid json'
    try:
        serializer.load_payload(invalid_payload)
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None
    
    # Test with override serializer parameter
    class OverrideSerializer:
        def loads(self, payload):
            return {"from_override": payload}
        def dumps(self, obj):
            return b'{"test": "data"}'
    
    override_serializer = OverrideSerializer()
    result = serializer.load_payload(payload, serializer=override_serializer)
    assert result == {"from_override": payload}
    
    # Test with empty payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test with text serializer that returns str
    class TextSerializer:
        def loads(self, payload: str):
            return {"from_text": payload}
        def dumps(self, obj):
            return '{"test": "data"}'
    
    text_serializer = Serializer("secret-key", serializer=TextSerializer())
    payload = b'{"key": "value"}'
    result = text_serializer.load_payload(payload)
    assert result == {"from_text": '{"key": "value"}'}
```


# LLM-generated content at query #36
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol correctly defines loads method."""
    # Create a concrete implementation that matches the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with simple types
    assert serializer.loads('42') == 42
    assert serializer.loads('"hello"') == "hello"
    assert serializer.loads('true') is True
    assert serializer.loads('null') is None
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test with empty data
    assert serializer.loads('{}') == {}
    assert serializer.loads('[]') == []
    
    # Test that it raises appropriate exception for invalid input
    import pytest
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
```


# LLM-generated content at query #37
#--------------------------

```python
def test_Serializer_load_payload():
    # Create a serializer instance for testing
    serializer = Serializer("test-secret-key")
    
    # Test 1: Successfully load a valid JSON payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test 2: Load with custom serializer that returns text
    class TextSerializer:
        def loads(self, payload: str):
            return json.loads(payload)
        def dumps(self, obj):
            return json.dumps(obj)
    
    text_serializer = TextSerializer()
    payload = b'{"number": 42}'
    result = serializer.load_payload(payload, serializer=text_serializer)
    assert result == {"number": 42}
    
    # Test 3: Load with custom serializer that works with bytes directly
    class BytesSerializer:
        def loads(self, payload: bytes):
            return json.loads(payload.decode('utf-8'))
        def dumps(self, obj):
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    payload = b'{"list": [1, 2, 3]}'
    result = serializer.load_payload(payload, serializer=bytes_serializer)
    assert result == {"list": [1, 2, 3]}
    
    # Test 4: Raise BadPayload for invalid JSON
    import pytest
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b'invalid json')
    assert "Could not load the payload" in str(exc_info.value)
    assert exc_info.value.original_error is not None
    
    # Test 5: Raise BadPayload for empty payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b'')
    
    # Test 6: Raise BadPayload for None payload (if applicable)
    with pytest.raises(BadPayload):
        serializer.load_payload(b'null')
    
    # Test 7: Serializer with text serializer that returns str
    serializer_text = Serializer("test-secret-key", serializer=json)
    payload = b'{"text": "hello"}'
    result = serializer_text.load_payload(payload)
    assert result == {"text": "hello"}
    
    # Test 8: Verify that the original_error attribute contains the underlying exception
    try:
        serializer.load_payload(b'{"broken": }')
    except BadPayload as e:
        assert isinstance(e.original_error, (json.JSONDecodeError, ValueError))
    
    # Test 9: Verify that load_payload works with nested data structures
    complex_data = {"nested": {"list": [1, 2, 3], "dict": {"a": 1}}}
    payload = json.dumps(complex_data).encode('utf-8')
    result = serializer.load_payload(payload)
    assert result == complex_data
    
    # Test 10: Verify that load_payload works with various data types
    test_cases = [
        (b'"string"', "string"),
        (b'123', 123),
        (b'true', True),
        (b'false', False),
        (b'null', None),
        (b'[1, "two", 3.0]', [1, "two", 3.0]),
    ]
    
    for payload, expected in test_cases:
        result = serializer.load_payload(payload)
        assert result == expected, f"Failed for payload {payload}"
```


# LLM-generated content at query #38
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            if payload == b'{"key": "value"}':
                return {"key": "value"}
            elif payload == b'42':
                return 42
            elif payload == b'null':
                return None
            elif payload == b'"string"':
                return "string"
            elif payload == b'invalid':
                raise ValueError("Invalid JSON")
            raise ValueError("Unknown payload")
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test loading a dict
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading an integer
    result = serializer.loads(b'42')
    assert result == 42
    
    # Test loading null
    result = serializer.loads(b'null')
    assert result is None
    
    # Test loading a string
    result = serializer.loads(b'"string"')
    assert result == "string"
    
    # Test that loads raises ValueError for invalid payload
    with pytest.raises(ValueError, match="Invalid JSON"):
        serializer.loads(b'invalid')
```


# LLM-generated content at query #39
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            if payload == b'{"key": "value"}':
                return {"key": "value"}
            elif payload == b'42':
                return 42
            elif payload == b'true':
                return True
            elif payload == '{"key": "value"}':
                return {"key": "value"}
            elif payload == '42':
                return 42
            elif payload == 'true':
                return True
            else:
                raise ValueError("Invalid payload")
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with bytes payload
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer payload
    result = serializer.loads(b'42')
    assert result == 42
    
    # Test with boolean payload
    result = serializer.loads(b'true')
    assert result is True
    
    # Test with string payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with string integer payload
    result = serializer.loads('42')
    assert result == 42
    
    # Test with string boolean payload
    result = serializer.loads('true')
    assert result is True
    
    # Test with invalid payload raises exception
    import pytest
    with pytest.raises(ValueError, match="Invalid payload"):
        serializer.loads(b'invalid')
    
    # Test with empty payload
    with pytest.raises(ValueError, match="Invalid payload"):
        serializer.loads(b'')
```


# LLM-generated content at query #40
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation that satisfies the protocol
    class TestSerializer:
        def loads(self, payload: str | bytes, /) -> t.Any:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            return json.loads(payload)
        
        def dumps(self, obj: t.Any, /) -> str | bytes:
            return json.dumps(obj)

    serializer = TestSerializer()
    
    # Test with string payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes payload
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer
    result = serializer.loads("42")
    assert result == 42
    
    # Test with list
    result = serializer.loads("[1, 2, 3]")
    assert result == [1, 2, 3]
    
    # Test with None
    result = serializer.loads("null")
    assert result is None
    
    # Test with boolean
    result = serializer.loads("true")
    assert result is True
    
    # Test with float
    result = serializer.loads("3.14")
    assert result == 3.14
    
    # Test that it raises appropriate error for invalid input
    with pytest.raises(Exception):
        serializer.loads("invalid json")
```


