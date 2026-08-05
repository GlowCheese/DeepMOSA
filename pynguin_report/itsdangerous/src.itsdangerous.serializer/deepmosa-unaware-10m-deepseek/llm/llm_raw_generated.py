####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
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
    s_salt = Serializer("secret-key", salt="custom-salt")
    result_salt = s_salt.dumps({"key": "value"})
    assert s_salt.loads(result_salt) == {"key": "value"}

    # Test that different salts produce different signatures
    s_salt2 = Serializer("secret-key", salt="other-salt")
    result_salt2 = s_salt2.dumps({"key": "value"})
    assert result_salt != result_salt2

    # Test with serializer_kwargs
    s_kwargs = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    result_kwargs = s_kwargs.dumps({"b": 1, "a": 2})
    assert s_kwargs.loads(result_kwargs) == {"b": 1, "a": 2}

    # Test with empty data
    s_empty = Serializer("secret-key")
    result_empty = s_empty.dumps({})
    assert s_empty.loads(result_empty) == {}

    # Test with None
    s_none = Serializer("secret-key")
    result_none = s_none.dumps(None)
    assert s_none.loads(result_none) is None

    # Test with list
    s_list = Serializer("secret-key")
    result_list = s_list.dumps([1, 2, 3])
    assert s_list.loads(result_list) == [1, 2, 3]

    # Test with multiple secret keys (key rotation)
    s_rotation = Serializer(["old-key", "new-key"])
    result_rotation = s_rotation.dumps({"key": "value"})
    # Should still verify with old key
    s_rotation_old = Serializer(["old-key"])
    assert s_rotation_old.loads(result_rotation) == {"key": "value"}
    # Should verify with new key
    s_rotation_new = Serializer(["new-key"])
    assert s_rotation_new.loads(result_rotation) == {"key": "value"}
```


# LLM-generated content at query #2
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method is properly defined."""
    # Create a simple serializer that conforms to _PDataSerializer[str]
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)

    # Create a simple serializer that conforms to _PDataSerializer[bytes]
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')

    text_serializer = TextSerializer()
    bytes_serializer = BytesSerializer()

    # Test that dumps returns str for text serializer
    result = text_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'

    # Test that dumps returns bytes for bytes serializer
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'

    # Test with different data types
    assert text_serializer.dumps(42) == '42'
    assert text_serializer.dumps([1, 2, 3]) == '[1, 2, 3]'
    assert text_serializer.dumps(None) == 'null'
    assert text_serializer.dumps(True) == 'true'

    # Test empty data
    assert text_serializer.dumps({}) == '{}'
    assert text_serializer.dumps([]) == '[]'
```


# LLM-generated content at query #3
#--------------------------

Here's a unit test for the `loads` method of `_PDataSerializer`:

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that conforms to _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            if payload == b'{"key": "value"}':
                return {"key": "value"}
            elif payload == b'[1, 2, 3]':
                return [1, 2, 3]
            elif payload == b'"hello"':
                return "hello"
            elif payload == b'42':
                return 42
            elif payload == b'null':
                return None
            elif payload == b'invalid json':
                raise ValueError("Invalid JSON")
            return payload
        
        def dumps(self, obj):
            return str(obj).encode()
    
    serializer = MockSerializer()
    
    # Test with valid JSON payload
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with array payload
    result = serializer.loads(b'[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with string payload
    result = serializer.loads(b'"hello"')
    assert result == "hello"
    
    # Test with number payload
    result = serializer.loads(b'42')
    assert result == 42
    
    # Test with null payload
    result = serializer.loads(b'null')
    assert result is None
    
    # Test with invalid payload
    try:
        serializer.loads(b'invalid json')
        assert False, "Expected ValueError"
    except ValueError:
        pass
```


# LLM-generated content at query #4
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with a simple object
    test_obj = {"key": "value"}
    result = serializer.dumps(test_obj)
    assert result == '{"key": "value"}'
    
    # Test with a list
    test_list = [1, 2, 3]
    result = serializer.dumps(test_list)
    assert result == "[1, 2, 3]"
    
    # Test with a string
    test_string = "hello"
    result = serializer.dumps(test_string)
    assert result == '"hello"'
    
    # Test with None
    result = serializer.dumps(None)
    assert result == "null"
    
    # Test with integer
    result = serializer.dumps(42)
    assert result == "42"
    
    # Test that the return type is string (as specified by the protocol for str serializer)
    assert isinstance(result, str)```


# LLM-generated content at query #5
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns str
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TextSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    serializer = BytesSerializer()
    result = serializer.dumps([1, 2, 3])
    assert isinstance(result, bytes)
    assert result == b'[1, 2, 3]'
    
    # Test with None value
    result = serializer.dumps(None)
    assert result == b'null'
    
    # Test with empty dict
    result = serializer.dumps({})
    assert result == b'{}'


# LLM-generated content at query #6
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return b'{"key":"value"}'
        def loads(self, payload):
            return json.loads(payload)
    
    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = s_bytes.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    
    # Test roundtrip
    s = Serializer("secret-key")
    data = {"test": "data", "number": 42}
    signed = s.dumps(data)
    loaded = s.loads(signed)
    assert loaded == data


# LLM-generated content at query #7
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("test-secret")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    # Verify it can be loaded back
    loaded = serializer.loads(result)
    assert loaded == {"key": "value"}

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))

    bytes_serializer = Serializer("test-secret", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    loaded = bytes_serializer.loads(result)
    assert loaded == {"key": "value"}

    # Test with custom salt
    serializer_with_salt = Serializer("test-secret", salt="custom-salt")
    result_default = serializer.dumps({"key": "value"})
    result_with_salt = serializer_with_salt.dumps({"key": "value"})
    assert result_default != result_with_salt

    # Test with serializer_kwargs
    class CustomJSONSerializer:
        def dumps(self, obj, **kwargs):
            if kwargs.get("sort_keys"):
                return json.dumps(obj, sort_keys=True)
            return json.dumps(obj)
        def loads(self, payload):
            return json.loads(payload)

    serializer_with_kwargs = Serializer(
        "test-secret",
        serializer=CustomJSONSerializer(),
        serializer_kwargs={"sort_keys": True}
    )
    result_sorted = serializer_with_kwargs.dumps({"b": 1, "a": 2})
    assert result_sorted is not None
    # The sorted version should have keys in alphabetical order
    loaded_sorted = serializer_with_kwargs.loads(result_sorted)
    assert loaded_sorted == {"a": 2, "b": 1}

    # Test with key rotation (multiple secret keys)
    serializer_rotation = Serializer(["old-key", "new-key"])
    result_rotation = serializer_rotation.dumps("test-data")
    # Should be signed with the newest key
    loaded_rotation = serializer_rotation.loads(result_rotation)
    assert loaded_rotation == "test-data"

    # Test that dumps returns consistent results for same input
    result1 = serializer.dumps("same-data")
    result2 = serializer.dumps("same-data")
    assert result1 != result2  # Different signatures due to timestamp/nonce
```


# LLM-generated content at query #8
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
    # Verify it's a valid signed JSON string (contains separator)
    assert "." in result

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            import pickle
            return pickle.dumps(obj)
        def loads(self, payload):
            import pickle
            return pickle.loads(payload)

    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = s_bytes.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes

    # Test with custom serializer_kwargs
    s_kwargs = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    result_kwargs = s_kwargs.dumps({"b": 2, "a": 1})
    # Should have sorted keys in the payload
    assert result_kwargs.count("a") == 1
    assert result_kwargs.count("b") == 1

    # Test with salt parameter
    s_salt = Serializer("secret-key")
    result_default_salt = s_salt.dumps("test")
    result_custom_salt = s_salt.dumps("test", salt="custom-salt")
    assert result_default_salt != result_custom_salt

    # Test with key rotation (list of keys)
    s_rotation = Serializer(["old-key", "new-key"])
    result_rotation = s_rotation.dumps("test")
    assert isinstance(result_rotation, str)

    # Test that dumps with bytes serializer returns bytes
    s_bytes2 = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes2 = s_bytes2.dumps("test")
    assert isinstance(result_bytes2, bytes)

    # Test that dumps with text serializer returns str
    s_text = Serializer("secret-key")
    result_text = s_text.dumps("test")
    assert isinstance(result_text, str)

    # Test dumps with None salt
    s_no_salt = Serializer("secret-key", salt=None)
    result_no_salt = s_no_salt.dumps("test")
    assert isinstance(result_no_salt, str)
    assert "." in result_no_salt

    # Test dumps with various data types
    test_data = [
        None,
        True,
        42,
        3.14,
        "string",
        [1, 2, 3],
        {"nested": {"data": "value"}},
    ]
    for data in test_data:
        result = s_text.dumps(data)
        assert isinstance(result, str)
        # Verify we can load it back
        loaded = s_text.loads(result)
        assert loaded == data
```


# LLM-generated content at query #9
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
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with a serializer that returns str
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    text_serializer = TextSerializer()
    result = text_serializer.dumps(data)
    assert isinstance(result, str)
    assert result == '{"key": "value"}'


# LLM-generated content at query #10
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test successful load with text serializer
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with custom bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return {"key": "value"}
        def dumps(self, obj):
            return b'{"key": "value"}'
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test BadPayload exception with invalid JSON
    import pytest
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"invalid json")
    assert "Could not load the payload" in str(exc_info.value)
    assert exc_info.value.original_error is not None
    
    # Test with override serializer parameter
    class OverrideSerializer:
        def loads(self, payload):
            return {"overridden": True}
        def dumps(self, obj):
            return b'test'
    
    result = serializer.load_payload(b'some payload', serializer=OverrideSerializer())
    assert result == {"overridden": True}
    
    # Test with binary serializer that returns bytes
    class BinarySerializer:
        def loads(self, payload):
            return payload
        def dumps(self, obj):
            return obj
    
    binary_serializer = Serializer("secret-key", serializer=BinarySerializer())
    result = binary_serializer.load_payload(b"binary data")
    assert result == b"binary data"
    
    # Test with text serializer that returns string
    class TextSerializer:
        def loads(self, payload):
            return payload
        def dumps(self, obj):
            return obj
    
    text_serializer = Serializer("secret-key", serializer=TextSerializer())
    result = text_serializer.load_payload("text data".encode("utf-8"))
    assert result == "text data"


# LLM-generated content at query #11
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol has dumps method."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test basic dumps functionality
    result = serializer.dumps({"key": "value"})
    assert result == '{"key": "value"}'
    
    # Test with different data types
    assert serializer.dumps(42) == "42"
    assert serializer.dumps("hello") == '"hello"'
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps(None) == "null"
```


# LLM-generated content at query #12
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a simple serializer that implements the _PDataSerializer protocol
    class TestSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer
    result = serializer.loads('42')
    assert result == 42
    
    # Test with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test with null
    result = serializer.loads('null')
    assert result is None
    
    # Test with boolean
    result = serializer.loads('true')
    assert result is True
    
    # Test with invalid JSON
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
    
    # Test with empty string
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('')
    
    # Test with bytes input (should work if serializer handles it)
    class BytesSerializer:
        def loads(self, payload):
            if isinstance(payload, bytes):
                return json.loads(payload.decode('utf-8'))
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
```


# LLM-generated content at query #13
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a serializer that returns bytes
    bytes_serializer = type("BytesSerializer", (), {
        "loads": lambda self, payload: payload,
        "dumps": lambda self, obj: want_bytes(json.dumps(obj))
    })()
    
    # Create a serializer that returns str
    str_serializer = type("StrSerializer", (), {
        "loads": lambda self, payload: payload,
        "dumps": lambda self, obj: json.dumps(obj)
    })()
    
    # Test bytes serializer
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert result_bytes == b'{"key": "value"}'
    
    # Test str serializer
    result_str = str_serializer.dumps({"key": "value"})
    assert isinstance(result_str, str)
    assert result_str == '{"key": "value"}'
    
    # Verify the protocol works with isinstance check
    assert is_text_serializer(str_serializer) == True
    assert is_text_serializer(bytes_serializer) == False
```


# LLM-generated content at query #14
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON serializer
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test loading a valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading primitive types
    assert serializer.loads('"string"') == "string"
    assert serializer.loads('42') == 42
    assert serializer.loads('true') == True
    assert serializer.loads('null') is None
    
    # Test loading invalid JSON raises exception
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
    
    # Test with a custom serializer that returns different types
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    custom_serializer = CustomSerializer()
    result = custom_serializer.loads("hello")
    assert result == "HELLO"
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that loads is called with correct argument type
    class TrackingSerializer:
        def __init__(self):
            self.last_payload = None
            
        def loads(self, payload: str) -> t.Any:
            self.last_payload = payload
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    tracking_serializer = TrackingSerializer()
    tracking_serializer.loads('{"test": 123}')
    assert tracking_serializer.last_payload == '{"test": 123}'
    
    # Test that loads returns any type (not just specific types)
    class AnyReturnSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload  # Returns the string itself
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    any_return = AnyReturnSerializer()
    result = any_return.loads("test payload")
    assert result == "test payload"
    assert isinstance(result, str)
```


# LLM-generated content at query #15
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer
    result = serializer.loads('42')
    assert result == 42
    
    # Test with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with null
    result = serializer.loads('null')
    assert result is None
    
    # Test with boolean
    result = serializer.loads('true')
    assert result is True
    
    # Test with string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test that invalid JSON raises appropriate error
    import json
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
```


# LLM-generated content at query #16
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
    result = string_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with various data types
    assert string_serializer.dumps(123) == "123"
    assert string_serializer.dumps("test") == '"test"'
    assert string_serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert string_serializer.dumps(None) == "null"
    assert string_serializer.dumps(True) == "true"
    assert string_serializer.dumps({"a": 1, "b": 2}) == '{"a": 1, "b": 2}'```


# LLM-generated content at query #17
#--------------------------

```python
def test_Serializer_dumps():
    """Test Serializer.dumps method with various configurations."""
    
    # Test with default JSON serializer (text)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
    assert s.loads(result) == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            import json
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            import json
            return json.loads(payload.decode("utf-8"))
    
    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = s_bytes.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert s_bytes.loads(result_bytes) == {"key": "value"}
    
    # Test with custom salt
    s_salt = Serializer("secret-key", salt="custom-salt")
    result_salt = s_salt.dumps({"key": "value"})
    assert isinstance(result_salt, str)
    assert s_salt.loads(result_salt) == {"key": "value"}
    
    # Verify different salts produce different signatures
    s_salt2 = Serializer("secret-key", salt="different-salt")
    result_salt2 = s_salt2.dumps({"key": "value"})
    assert result_salt != result_salt2
    
    # Test with serializer_kwargs
    class CustomSerializer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def dumps(self, obj, **kwargs):
            import json
            combined_kwargs = {**self.kwargs, **kwargs}
            return json.dumps(obj, **combined_kwargs)
        def loads(self, payload):
            import json
            return json.loads(payload)
    
    s_kwargs = Serializer("secret-key", serializer=CustomSerializer(), serializer_kwargs={"sort_keys": True})
    result_kwargs = s_kwargs.dumps({"b": 2, "a": 1})
    assert result_kwargs.count('"a"') == 1
    assert result_kwargs.count('"b"') == 1
    # Sort keys should place "a" before "b"
    assert result_kwargs.find('"a"') < result_kwargs.find('"b"')
    
    # Test with different payload types
    test_cases = [
        None,
        True,
        False,
        42,
        3.14,
        "string",
        [1, 2, 3],
        {"nested": {"key": "value"}},
    ]
    
    for obj in test_cases:
        s = Serializer("secret-key")
        result = s.dumps(obj)
        assert s.loads(result) == obj
    
    # Test that dumps produces different output for different secret keys
    s1 = Serializer("secret-key-1")
    s2 = Serializer("secret-key-2")
    obj = {"data": "test"}
    result1 = s1.dumps(obj)
    result2 = s2.dumps(obj)
    assert result1 != result2
    
    # Test with key rotation (list of keys)
    s_rotation = Serializer(["old-key", "new-key"])
    result_rotation = s_rotation.dumps({"key": "value"})
    assert isinstance(result_rotation, str)
    assert s_rotation.loads(result_rotation) == {"key": "value"}  # Newest key signs
    
    # Verify old key can't unsign (signed with newest key)
    s_old = Serializer(["old-key"])
    with pytest.raises(BadSignature):
        s_old.loads(result_rotation)  # Should fail since signed with "new-key"```


# LLM-generated content at query #18
#--------------------------

```python
def test_Serializer_dumps():
    """Test that dumps returns the expected serialized and signed output."""
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("test-secret-key")
    data = {"key": "value"}
    result = serializer.dumps(data)
    
    # Result should be a string with the payload and signature separated by '.'
    assert isinstance(result, str)
    assert "." in result
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("test-secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps(data)
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    # Test with custom serializer_kwargs
    serializer_with_kwargs = Serializer(
        "test-secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_kwargs = serializer_with_kwargs.dumps({"b": 1, "a": 2})
    # The payload should be sorted and compact
    assert result_kwargs.count(":") == 2  # Two key-value pairs
    
    # Test that the output can be verified back
    signed = serializer.dumps(data)
    assert serializer.loads(signed) == data
    
    # Test with different salt
    result_with_salt = serializer.dumps(data, salt="custom-salt")
    assert serializer.loads(result_with_salt, salt="custom-salt") == data
    
    # Test with bytes payload (should still work)
    bytes_data = {"test": b"bytes value"}
    result_bytes_data = serializer.dumps(bytes_data)
    assert serializer.loads(result_bytes_data) == bytes_data
    
    # Test with empty dict
    result_empty = serializer.dumps({})
    assert serializer.loads(result_empty) == {}
    
    # Test with list
    result_list = serializer.dumps([1, 2, 3])
    assert serializer.loads(result_list) == [1, 2, 3]
    
    # Test that the signature is different for different keys
    serializer2 = Serializer("different-secret-key")
    result1 = serializer.dumps(data)
    result2 = serializer2.dumps(data)
    assert result1 != result2
    
    # Test that the signature is different for different data
    result3 = serializer.dumps({"different": "data"})
    assert result1 != result3
    
    # Test with None values
    result_none = serializer.dumps(None)
    assert serializer.loads(result_none) is None
    
    # Test that dumps returns bytes when is_text_serializer is False
    class TextSerializer:
        def dumps(self, obj):
            return json.dumps(obj)
        def loads(self, payload):
            return json.loads(payload)
    
    text_serializer = Serializer("test-key", serializer=TextSerializer())
    result_text = text_serializer.dumps(data)
    assert isinstance(result_text, str)
    
    # Test that the method doesn't raise exceptions for valid inputs
    serializer.dumps("simple string")
    serializer.dumps(123)
    serializer.dumps(True)
    serializer.dumps([{"nested": "value"}])


# LLM-generated content at query #19
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
    result = string_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'

    # Test with empty object
    result = string_serializer.dumps({})
    assert result == "{}"

    # Test with None value
    result = string_serializer.dumps(None)
    assert result == "null"

    # Test with list
    result = string_serializer.dumps([1, 2, 3])
    assert result == "[1, 2, 3]"
```


# LLM-generated content at query #20
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Test with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    string_serializer = StringSerializer()
    assert string_serializer.loads('{"key": "value"}') == {"key": "value"}
    assert string_serializer.loads('"test"') == "test"
    assert string_serializer.loads('123') == 123
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'"test"') == "test"
    assert bytes_serializer.loads(b'123') == 123
    
    # Test that loads raises appropriate exceptions
    with pytest.raises(json.JSONDecodeError):
        string_serializer.loads("invalid json")
    
    with pytest.raises(json.JSONDecodeError):
        bytes_serializer.loads(b"invalid json")
```


# LLM-generated content at query #21
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a basic serializer that returns a dict
    class DictSerializer:
        def loads(self, payload: str) -> dict:
            return {"data": payload}
        
        def dumps(self, obj: dict) -> str:
            return str(obj)
    
    serializer = DictSerializer()
    result = serializer.loads("test_payload")
    assert result == {"data": "test_payload"}
    
    # Test with JSON serializer
    json_serializer = type('JSONSerializer', (), {
        'loads': staticmethod(json.loads),
        'dumps': staticmethod(json.dumps)
    })()
    
    result = json_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> dict:
            return json.loads(payload.decode())
        
        def dumps(self, obj: dict) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"number": 42}')
    assert result == {"number": 42}
    
    # Test that the protocol is followed correctly
    assert callable(serializer.loads)
    assert callable(json_serializer.loads)
    assert callable(bytes_serializer.loads)


# LLM-generated content at query #22
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default settings
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)

    # Test with custom salt
    signers_with_salt = list(serializer.iter_unsigners(salt=b"custom-salt"))
    assert len(signers_with_salt) == 1
    assert isinstance(signers_with_salt[0], Signer)

    # Test with fallback signers as dict
    serializer_with_fallback = Serializer(
        "secret-key",
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer_with_fallback.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

    # Test with fallback signers as tuple
    class CustomSigner(Signer):
        pass
    
    serializer_with_tuple_fallback = Serializer(
        "secret-key",
        fallback_signers=[(CustomSigner, {"digest_method": "sha512"})]
    )
    signers = list(serializer_with_tuple_fallback.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], CustomSigner)

    # Test with fallback signers as Signer class
    serializer_with_class_fallback = Serializer(
        "secret-key",
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer_with_class_fallback.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], CustomSigner)

    # Test with multiple secret keys for key rotation
    serializer_with_keys = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer_with_keys.iter_unsigners())
    # First signer uses all keys, then for each fallback we get one signer per key
    assert len(signers) == 3  # 1 default + 2 fallback (one per key)
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert isinstance(signers[2], Signer)

    # Test with fallback signers that are Signer class and multiple keys
    serializer_with_class_and_keys = Serializer(
        ["key1", "key2"],
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer_with_class_and_keys.iter_unsigners())
    assert len(signers) == 3  # 1 default + 2 fallback (one per key)
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], CustomSigner)
    assert isinstance(signers[2], CustomSigner)

    # Test iter_unsigners returns an iterator (not a list)
    serializer_simple = Serializer("secret-key")
    iterator = serializer_simple.iter_unsigners()
    assert hasattr(iterator, '__next__')
    assert hasattr(iterator, '__iter__')
```


# LLM-generated content at query #23
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default settings (no fallback signers)
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_key == b"secret-key"
    
    # Test with dict fallback signer
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2  # main signer + fallback
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    
    # Test with tuple fallback signer (signer class + kwargs)
    class CustomSigner(Signer):
        pass
    
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(CustomSigner, {"key_derivation": "hmac"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], CustomSigner)
    
    # Test with Signer class as fallback
    serializer = Serializer(
        "secret-key",
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], CustomSigner)
    
    # Test with multiple secret keys (key rotation)
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer.iter_unsigners())
    # main signer + 2 fallbacks (one for each secret key)
    assert len(signers) == 3
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_key == b"old-key"
    assert signers[2].secret_key == b"new-key"
    
    # Test with custom salt
    serializer = Serializer("secret-key", salt=b"custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"
    
    # Test with salt parameter override
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners(salt=b"override-salt"))
    assert signers[0].salt == b"override-salt"
    
    # Test with no fallback signers (empty list)
    serializer = Serializer("secret-key", fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    
    # Test with multiple fallback signers
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"digest_method": "sha256"},
            (CustomSigner, {"key_derivation": "hmac"}),
            CustomSigner
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4  # main + 3 fallbacks
```


# LLM-generated content at query #24
#--------------------------

```python
def test_Serializer_load_payload():
    """Test load_payload method of Serializer class."""
    # Test with default JSON serializer (text-based)
    serializer = Serializer("test-secret")
    
    # Test successful load with bytes payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload):
            return {"data": payload}
        def dumps(self, obj):
            return b"test"
    
    bytes_serializer = Serializer("test-secret", serializer=BytesSerializer())
    payload = b"test_bytes"
    result = bytes_serializer.load_payload(payload)
    assert result == {"data": b"test_bytes"}
    
    # Test with custom serializer that returns text
    class TextSerializer:
        def loads(self, payload):
            return {"data": payload}
        def dumps(self, obj):
            return "test"
    
    text_serializer = Serializer("test-secret", serializer=TextSerializer())
    payload = b"test_text"
    result = text_serializer.load_payload(payload)
    assert result == {"data": "test_text"}
    
    # Test raising BadPayload on invalid JSON
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"invalid json")
    assert "Could not load the payload" in str(exc_info.value)
    assert exc_info.value.original_error is not None
    
    # Test raising BadPayload on empty bytes
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test with override serializer parameter
    class OverrideSerializer:
        def loads(self, payload):
            return {"overridden": payload}
        def dumps(self, obj):
            return b"test"
    
    payload = b"override_test"
    result = serializer.load_payload(payload, serializer=OverrideSerializer())
    assert result == {"overridden": b"override_test"}
    
    # Test with text-based override serializer
    class OverrideTextSerializer:
        def loads(self, payload):
            return {"overridden": payload}
        def dumps(self, obj):
            return "test"
    
    payload = b"override_text_test"
    result = serializer.load_payload(payload, serializer=OverrideTextSerializer())
    assert result == {"overridden": "override_text_test"}
    
    # Test with JSON serializer and complex data
    import json
    data = {"numbers": [1, 2, 3], "nested": {"key": "value"}}
    json_payload = json.dumps(data).encode("utf-8")
    result = serializer.load_payload(json_payload)
    assert result == data
    
    # Test that original_error is set correctly for JSON decode errors
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"\x80\x81\x82")
    assert exc_info.value.original_error is not None
```


# LLM-generated content at query #25
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test that _PDataSerializer is a Protocol and cannot be instantiated directly
    with pytest.raises(TypeError):
        _PDataSerializer()
    
    # Test that a class implementing the Protocol can be used
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test loads with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loads with integer
    result = serializer.loads('42')
    assert result == 42
    
    # Test loads with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loads with None
    result = serializer.loads('null')
    assert result is None
    
    # Test loads with boolean
    result = serializer.loads('true')
    assert result is True
    
    # Test loads with string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test loads with invalid JSON
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
    
    # Test that the protocol accepts bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
```


# LLM-generated content at query #26
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol defines loads method correctly."""
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
    assert serializer.loads('"string"') == "string"
    assert serializer.loads('42') == 42
    assert serializer.loads('true') is True
    assert serializer.loads('null') is None
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test that protocol accepts bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}

```


# LLM-generated content at query #27
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer dumps method returns correct type."""
    # Create a concrete implementation that matches the protocol
    class TestSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return str(obj)
    
    serializer = TestSerializer()
    
    # Test dumps returns string for text serializer
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == "{'key': 'value'}"
    
    # Test dumps returns bytes for bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return bytes(str(obj), 'utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b"{'key': 'value'}"
```


# LLM-generated content at query #28
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockTextSerializer:
        def loads(self, payload: str) -> t.Any:
            if payload == '{"key": "value"}':
                return {"key": "value"}
            raise ValueError("Invalid payload")
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    class MockBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            if payload == b'{"key": "value"}':
                return {"key": "value"}
            raise ValueError("Invalid payload")
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    # Test with text serializer
    text_serializer = MockTextSerializer()
    result = text_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    bytes_serializer = MockBytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with invalid payload
    try:
        text_serializer.loads("invalid json")
        assert False, "Should have raised an exception"
    except ValueError:
        pass
    
    try:
        bytes_serializer.loads(b"invalid json")
        assert False, "Should have raised an exception"
    except ValueError:
        pass
    
    # Test with empty payload
    try:
        text_serializer.loads("")
        assert False, "Should have raised an exception"
    except ValueError:
        pass
    
    # Test with complex nested data
    complex_data = {"nested": {"list": [1, 2, 3], "bool": True, "null": None}}
    serialized = json.dumps(complex_data)
    result = text_serializer.loads(serialized)
    assert result == complex_data
    
    # Test with bytes serializer and complex data
    serialized_bytes = json.dumps(complex_data).encode()
    result = bytes_serializer.loads(serialized_bytes)
    assert result == complex_data
```


# LLM-generated content at query #29
#--------------------------

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
    json_serializer = json
    result = json_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with custom serializer that returns a dict
    class DictSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"data": payload}
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    dict_serializer = DictSerializer()
    result = dict_serializer.loads("test")
    assert result == {"data": "test"}
    
    # Test with empty string
    empty_serializer = StringSerializer()
    result = empty_serializer.loads("")
    assert result == ""
    
    # Test with numeric string
    result = StringSerializer().loads("123")
    assert result == "123"
```


# LLM-generated content at query #30
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns str
    class StrSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StrSerializer()
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer payload
    class IntSerializer:
        def loads(self, payload: str) -> t.Any:
            return int(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    int_serializer = IntSerializer()
    result = int_serializer.loads("42")
    assert result == 42
    
    # Test with list payload
    result = int_serializer.loads("123")
    assert result == 123
    
    # Test with None payload
    result = int_serializer.loads("0")
    assert result == 0
```


# LLM-generated content at query #31
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a text serializer (str return type)
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TextSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'

    # Test with a bytes serializer (bytes return type)
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'

    # Test with None value
    result = serializer.dumps(None)
    assert result == "null"

    # Test with list value
    result = serializer.dumps([1, 2, 3])
    assert result == "[1, 2, 3]"

    # Test with string value
    result = serializer.dumps("test")
    assert result == '"test"'

    # Test with integer value
    result = serializer.dumps(42)
    assert result == "42"


# LLM-generated content at query #32
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with various data types
    test_data = [
        {"key": "value"},
        [1, 2, 3],
        "string_data",
        42,
        True,
        None,
        {"nested": {"list": [1, 2]}}
    ]
    
    for data in test_data:
        result = serializer.dumps(data)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        # Verify it can be loaded back correctly
        assert serializer.loads(result) == data
    
    # Test that it returns str (not bytes)
    assert isinstance(serializer.dumps({}), str), "dumps should return str for text serializer"


# LLM-generated content at query #33
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test valid payload with JSON
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return payload.decode("utf-8")
        def dumps(self, obj):
            return obj.encode("utf-8")
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    payload = b"test data"
    result = bytes_serializer.load_payload(payload)
    assert result == "test data"
    
    # Test with custom serializer parameter
    class CustomSerializer:
        def loads(self, payload):
            return payload.upper()
        def dumps(self, obj):
            return obj.lower()
    
    custom_ser = CustomSerializer()
    payload = b"hello"
    result = serializer.load_payload(payload, serializer=custom_ser)
    assert result == "HELLO"
    
    # Test with invalid payload (should raise BadPayload)
    import json
    invalid_payload = b"not valid json"
    try:
        serializer.load_payload(invalid_payload)
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test with empty payload
    try:
        serializer.load_payload(b"")
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test with text serializer that returns str
    class TextSerializer:
        def loads(self, payload):
            return payload
        def dumps(self, obj):
            return str(obj)
    
    text_ser = TextSerializer()
    payload = b"unicode text"
    result = serializer.load_payload(payload, serializer=text_ser)
    assert result == "unicode text"
    
    # Test with bytes payload that contains non-UTF8 characters
    non_utf8_payload = b"\xff\xfe"
    try:
        serializer.load_payload(non_utf8_payload)
        assert False, "Should have raised BadPayload"
    except (BadPayload, UnicodeDecodeError):
        pass
```


# LLM-generated content at query #34
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test _PDataSerializer loads method protocol."""
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            return {"result": payload}
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test loading a string payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}, "Should deserialize JSON string to dict"
    
    # Test loading bytes payload
    result = serializer.loads(b'{"number": 42}')
    assert result == {"number": 42}, "Should deserialize bytes to dict"
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3], "Should deserialize JSON array to list"
    
    # Test loading with custom implementation
    class CustomSerializer:
        def loads(self, payload):
            if isinstance(payload, str):
                return payload.upper()
            return payload.decode().upper()
        
        def dumps(self, obj):
            return str(obj)
    
    custom_serializer = CustomSerializer()
    result = custom_serializer.loads("hello")
    assert result == "HELLO", "Custom loads should transform input"
    
    # Test that loads returns Any type
    result = serializer.loads("null")
    assert result is None, "Should handle null values"
    
    result = serializer.loads("true")
    assert result is True, "Should handle boolean values"
    
    # Test with integer payload
    result = serializer.loads("42")
    assert result == 42, "Should handle numeric values"
```


# LLM-generated content at query #35
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StringSerializer()
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with an integer payload
    class IntSerializer:
        def loads(self, payload: str) -> t.Any:
            return int(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    int_serializer = IntSerializer()
    result = int_serializer.loads("42")
    assert result == 42
```


# LLM-generated content at query #36
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method can be implemented."""
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
            
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'


# LLM-generated content at query #37
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol defines dumps method correctly."""
    # Create a mock serializer that conforms to the protocol
    class MockSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return str(obj)
    
    serializer = MockSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == "{'key': 'value'}"
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return str(obj).encode()
    
    bytes_serializer = BytesSerializer()
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert result_bytes == b"{'key': 'value'}"
    
    # Test that dumps returns correct type based on protocol
    assert is_text_serializer(MockSerializer()) == True
    assert is_text_serializer(BytesSerializer()) == False


# LLM-generated content at query #38
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol loads method works correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            if payload == '{"key": "value"}':
                return {"key": "value"}
            elif payload == "invalid":
                raise ValueError("Invalid payload")
            return None

        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)

    serializer = TestSerializer()
    
    # Test successful loading
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that it raises on invalid input
    with pytest.raises(ValueError, match="Invalid payload"):
        serializer.loads("invalid")
    
    # Test with bytes payload (should work at runtime)
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
```


# LLM-generated content at query #39
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.startswith("eyJrZXkiOiAidmFsdWUifQ")  # base64 of JSON payload
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            import pickle
            return pickle.dumps(obj)
        def loads(self, payload):
            import pickle
            return pickle.loads(payload)
    
    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    result = s_bytes.dumps({"key": "value"})
    assert isinstance(result, bytes)
    
    # Test with custom serializer_kwargs
    s = Serializer("secret-key", serializer_kwargs={"sort_keys": True, "separators": (",", ":")})
    result = s.dumps({"b": 2, "a": 1})
    assert isinstance(result, str)
    assert '{"a":1,"b":2}' in result  # sorted keys, compact format
    
    # Test with salt parameter
    s = Serializer("secret-key")
    result1 = s.dumps({"key": "value"}, salt="custom-salt")
    result2 = s.dumps({"key": "value"}, salt="different-salt")
    assert result1 != result2  # Different salts produce different signatures
    
    # Test that the payload is properly serialized and signed
    s = Serializer("secret-key")
    result = s.dumps(42)
    assert isinstance(result, str)
    # Verify it can be loaded back
    loaded = s.loads(result)
    assert loaded == 42
    
    # Test with empty dict
    s = Serializer("secret-key")
    result = s.dumps({})
    assert isinstance(result, str)
    assert s.loads(result) == {}
    
    # Test with list
    s = Serializer("secret-key")
    result = s.dumps([1, 2, 3])
    assert isinstance(result, str)
    assert s.loads(result) == [1, 2, 3]
    
    # Test with None
    s = Serializer("secret-key")
    result = s.dumps(None)
    assert isinstance(result, str)
    assert s.loads(result) is None
    
    # Test with multiple secret keys (key rotation)
    s = Serializer(["old-key", "new-key"])
    result = s.dumps("test")
    assert isinstance(result, str)
    # Should be signed with the newest key
    assert s.loads(result) == "test"
```


# LLM-generated content at query #40
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method is properly defined."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return obj
    
    serializer = TestSerializer()
    test_data = {"key": "value"}
    
    # Test that dumps works correctly
    result = serializer.dumps(test_data)
    assert result == test_data
    
    # Test with different data types
    assert serializer.dumps("string") == "string"
    assert serializer.dumps(123) == 123
    assert serializer.dumps([1, 2, 3]) == [1, 2, 3]
    
    # Test that dumps returns the correct type based on protocol
    text_serializer = json
    assert isinstance(text_serializer.dumps({}), str)
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    assert isinstance(bytes_serializer.dumps({}), bytes)
```


# LLM-generated content at query #41
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer loads method works correctly with various serializers."""
    # Create a simple serializer that implements the protocol
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    # Test with text serializer
    text_serializer = TextSerializer()
    assert text_serializer.loads('{"key": "value"}') == {"key": "value"}
    assert text_serializer.loads("42") == 42
    assert text_serializer.loads("null") is None
    assert text_serializer.loads("true") is True
    assert text_serializer.loads('"hello"') == "hello"
    
    # Test with bytes serializer
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b"42") == 42
    assert bytes_serializer.loads(b"null") is None
    assert bytes_serializer.loads(b"true") is True
    assert bytes_serializer.loads(b'"hello"') == "hello"
    
    # Test with actual json module (which is a valid _PDataSerializer)
    assert json.loads('{"key": "value"}') == {"key": "value"}
    assert json.loads(b'{"key": "value"}') == {"key": "value"}
    
    # Test that loads raises appropriate exceptions
    with pytest.raises(json.JSONDecodeError):
        text_serializer.loads("invalid json")
    
    with pytest.raises(json.JSONDecodeError):
        bytes_serializer.loads(b"invalid json")
```


# LLM-generated content at query #42
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json) - returns str
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return b'{"key":"value"}'
        def loads(self, payload):
            return json.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    
    # Test with custom serializer_kwargs
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    result = serializer.dumps({"b": 2, "a": 1})
    assert result.startswith('{"a":1,"b":2}')
    
    # Test with salt
    serializer1 = Serializer("secret-key", salt=b"salt1")
    serializer2 = Serializer("secret-key", salt=b"salt2")
    result1 = serializer1.dumps({"key": "value"})
    result2 = serializer2.dumps({"key": "value"})
    assert result1 != result2
    
    # Test with multiple secret keys
    serializer = Serializer(["old-key", "new-key"])
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    
    # Verify dumps creates a valid signed payload
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    # Should contain the serialized payload and signature
    assert "." in result
    
    # Test with bytes input
    serializer = Serializer(b"secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
```


# LLM-generated content at query #43
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation of _PDataSerializer protocol
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
    assert serializer.loads('"string"') == "string"
    assert serializer.loads('123') == 123
    assert serializer.loads('true') == True
    assert serializer.loads('null') is None
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test with empty data
    assert serializer.loads('{}') == {}
    assert serializer.loads('[]') == []
    
    # Test with nested structures
    nested = '{"a": {"b": [1, 2, {"c": "d"}]}}'
    result = serializer.loads(nested)
    assert result == {"a": {"b": [1, 2, {"c": "d"}]}}
    
    # Test with special characters in strings
    special = '{"text": "hello\\nworld\\t\\u00e9"}'
    result = serializer.loads(special)
    assert result == {"text": "hello\nworld\té"}
    
    # Test that it raises appropriate error for invalid input
    with pytest.raises(json.JSONDecodeError):
        serializer.loads("invalid json")
    
    with pytest.raises(json.JSONDecodeError):
        serializer.loads("{broken}")
```


# LLM-generated content at query #44
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with simple values
    assert serializer.loads('42') == 42
    assert serializer.loads('"hello"') == "hello"
    assert serializer.loads('true') == True
    assert serializer.loads('null') is None
    
    # Test with list
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test that it raises appropriate exception for invalid input
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')


# LLM-generated content at query #45
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
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
    s_salt = Serializer("secret-key", salt="custom-salt")
    result_salt = s_salt.dumps({"key": "value"})
    assert isinstance(result_salt, str)
    assert s_salt.loads(result_salt) == {"key": "value"}
    # Different salt should produce different signature
    s_other_salt = Serializer("secret-key", salt="other-salt")
    result_other_salt = s_other_salt.dumps({"key": "value"})
    assert result_salt != result_other_salt

    # Test that dumps returns different values for different data
    result1 = s.dumps({"a": 1})
    result2 = s.dumps({"b": 2})
    assert result1 != result2

    # Test with serializer_kwargs
    s_kwargs = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    result_kwargs = s_kwargs.dumps({"b": 1, "a": 2})
    assert isinstance(result_kwargs, str)
    # The sorted JSON should be '{"a":2,"b":1}'
    assert s_kwargs.loads(result_kwargs) == {"a": 2, "b": 1}

    # Test with key rotation (list of keys)
    s_keys = Serializer(["old-key", "new-key"])
    result_keys = s_keys.dumps({"key": "value"})
    assert isinstance(result_keys, str)
    assert s_keys.loads(result_keys) == {"key": "value"}
    # The newest key should be used for signing
    assert s_keys.secret_key == b"new-key"

    # Test that dumps produces valid output that can be verified
    s_verify = Serializer("verify-key")
    payload = {"user": "test"}
    signed = s_verify.dumps(payload)
    # Should be able to load it back
    assert s_verify.loads(signed) == payload
```


# LLM-generated content at query #46
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with string serializer
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
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    result_bytes = bytes_serializer.dumps(data)
    assert isinstance(result_bytes, bytes)
    assert result_bytes == b'{"key": "value"}'
    
    # Test with various data types
    serializer2 = StringSerializer()
    assert serializer2.dumps(123) == '123'
    assert serializer2.dumps([1, 2, 3]) == '[1, 2, 3]'
    assert serializer2.dumps(None) == 'null'
    assert serializer2.dumps(True) == 'true'


# LLM-generated content at query #47
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test that loads method works with a simple string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StringSerializer()
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that loads method works with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that loads method raises appropriate exception for invalid input
    class StrictSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    strict_serializer = StrictSerializer()
    import pytest
    with pytest.raises(json.JSONDecodeError):
        strict_serializer.loads("invalid json")
```


# LLM-generated content at query #48
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test that dumps returns the expected type based on serializer
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    text_serializer = TextSerializer()
    bytes_serializer = BytesSerializer()
    
    # Test text serializer
    result = text_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test bytes serializer
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test that dumps handles different data types
    assert text_serializer.dumps(None) == 'null'
    assert text_serializer.dumps(42) == '42'
    assert text_serializer.dumps([1, 2, 3]) == '[1, 2, 3]'
    
    # Test that bytes serializer works with complex data
    data = {"nested": {"list": [1, 2, 3], "bool": True}}
    result = bytes_serializer.dumps(data)
    assert isinstance(result, bytes)
    assert json.loads(result.decode()) == data
```


# LLM-generated content at query #49
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
    
    # Test basic loads functionality
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with primitive types
    result = serializer.loads('"text"')
    assert result == "text"
    
    result = serializer.loads('42')
    assert result == 42
    
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('null')
    assert result is None
    
    # Test with empty payload
    result = serializer.loads('{}')
    assert result == {}
    
    # Test that the protocol accepts only str input
    serializer_loads = serializer.loads
    assert callable(serializer_loads)
    
    # Test that loads returns t.Any (can be any type)
    result = serializer.loads('{"nested": {"list": [1, 2, 3]}}')
    assert isinstance(result, dict)
    assert result["nested"]["list"] == [1, 2, 3]


# LLM-generated content at query #50
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert result == serializer.dumps(data)  # Deterministic
    
    # Test round trip
    loaded = serializer.loads(result)
    assert loaded == data
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        
        def loads(self, payload):
            return json.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    data = [1, 2, 3]
    result = bytes_serializer.dumps(data)
    assert isinstance(result, bytes)
    
    # Test with custom salt
    result_with_salt = serializer.dumps(data, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert result_with_salt != serializer.dumps(data)
    
    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    data = {"b": 2, "a": 1}
    result = serializer_with_kwargs.dumps(data)
    assert b'"a"' in result.encode() or '"a"' in result
    
    # Test with key rotation
    serializer_rotated = Serializer(["old-key", "new-key"])
    data = "test"
    result = serializer_rotated.dumps(data)
    assert isinstance(result, str)
    loaded = serializer_rotated.loads(result)
    assert loaded == data
    
    # Test with empty data
    serializer = Serializer("secret-key")
    result = serializer.dumps({})
    assert isinstance(result, str)
    loaded = serializer.loads(result)
    assert loaded == {}
    
    # Test with None data
    result = serializer.dumps(None)
    loaded = serializer.loads(result)
    assert loaded is None
    
    # Test with list data
    data = [1, "two", 3.0]
    result = serializer.dumps(data)
    loaded = serializer.loads(result)
    assert loaded == data
    
    # Test with nested data
    data = {"nested": {"list": [1, 2, 3]}}
    result = serializer.dumps(data)
    loaded = serializer.loads(result)
    assert loaded == data
    
    # Test that dumps returns bytes when serializer returns bytes
    class BytesSerializer2:
        def dumps(self, obj):
            return json.dumps(obj).encode()
        
        def loads(self, payload):
            return json.loads(payload.decode())
    
    serializer_bytes = Serializer("key", serializer=BytesSerializer2())
    result = serializer_bytes.dumps({"test": "data"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #51
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer loads method works correctly with different serializers."""
    # Test with JSON serializer
    json_serializer = json
    assert json_serializer.loads('{"key": "value"}') == {"key": "value"}
    assert json_serializer.loads('[1, 2, 3]') == [1, 2, 3]
    assert json_serializer.loads('"string"') == "string"
    assert json_serializer.loads('null') is None
    assert json_serializer.loads('true') is True
    assert json_serializer.loads('42') == 42
    
    # Test with custom text serializer
    class CustomTextSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj).lower()
    
    custom_serializer = CustomTextSerializer()
    assert custom_serializer.loads("hello") == "HELLO"
    assert custom_serializer.loads("test") == "TEST"
    
    # Test with custom bytes serializer
    class CustomBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode("utf-8").split(",")
        
        def dumps(self, obj: t.Any) -> bytes:
            return ",".join(obj).encode("utf-8")
    
    bytes_serializer = CustomBytesSerializer()
    assert bytes_serializer.loads(b"a,b,c") == ["a", "b", "c"]
    assert bytes_serializer.loads(b"x,y,z") == ["x", "y", "z"]
    
    # Test that loads works with different payload types
    class TypePreservingSerializer:
        def loads(self, payload: t.Any) -> t.Any:
            if isinstance(payload, bytes):
                return payload.hex()
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    type_serializer = TypePreservingSerializer()
    assert type_serializer.loads("hello") == "hello"
    assert type_serializer.loads(b"\x00\x01\x02") == "000102"
    
    # Test that loads is called with the correct argument
    call_args = []
    
    class TrackingSerializer:
        def loads(self, payload: t.Any) -> t.Any:
            call_args.append(payload)
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    tracking_serializer = TrackingSerializer()
    tracking_serializer.loads("test_payload")
    assert call_args == ["test_payload"] 


# LLM-generated content at query #52
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation that follows the protocol
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
    assert serializer.loads('true') == True
    assert serializer.loads('null') is None
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test with empty structures
    assert serializer.loads('{}') == {}
    assert serializer.loads('[]') == []
    
    # Test that it raises appropriate error for invalid input
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
    
    # Test that the method accepts only positional argument (per protocol)
    with pytest.raises(TypeError):
        serializer.loads(payload='test')  # type: ignore[call-overload]
```


# LLM-generated content at query #53
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    
    # Verify the result can be loaded back
    loaded = serializer.loads(result)
    assert loaded == data
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    data = {"key": "value"}
    result = bytes_serializer.dumps(data)
    assert isinstance(result, bytes)
    
    # Verify bytes result can be loaded back
    loaded = bytes_serializer.loads(result)
    assert loaded == data
    
    # Test with custom salt
    serializer_with_salt = Serializer("secret-key", salt="custom-salt")
    result_with_salt = serializer_with_salt.dumps(data)
    assert isinstance(result_with_salt, str)
    
    # Results with different salts should be different
    assert result != result_with_salt
    
    # Test with serializer_kwargs
    serializer_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_kwargs = serializer_kwargs.dumps(data)
    assert isinstance(result_kwargs, str)
    loaded_kwargs = serializer_kwargs.loads(result_kwargs)
    assert loaded_kwargs == data
    
    # Test with empty data
    empty_result = serializer.dumps({})
    assert isinstance(empty_result, str)
    loaded_empty = serializer.loads(empty_result)
    assert loaded_empty == {}
    
    # Test with list data
    list_data = [1, 2, 3]
    list_result = serializer.dumps(list_data)
    assert isinstance(list_result, str)
    loaded_list = serializer.loads(list_result)
    assert loaded_list == list_data
    
    # Test with string data
    str_data = "test string"
    str_result = serializer.dumps(str_data)
    assert isinstance(str_result, str)
    loaded_str = serializer.loads(str_result)
    assert loaded_str == str_data
    
    # Test with numeric data
    num_data = 42
    num_result = serializer.dumps(num_data)
    assert isinstance(num_result, str)
    loaded_num = serializer.loads(num_result)
    assert loaded_num == num_data
    
    # Test with None data
    none_result = serializer.dumps(None)
    assert isinstance(none_result, str)
    loaded_none = serializer.loads(none_result)
    assert loaded_none is None
    
    # Test with boolean data
    bool_result = serializer.dumps(True)
    assert isinstance(bool_result, str)
    loaded_bool = serializer.loads(bool_result)
    assert loaded_bool is True
    
    # Test that dumps produces deterministic output for same input
    result1 = serializer.dumps(data)
    result2 = serializer.dumps(data)
    assert result1 == result2
    
    # Test with different secret keys produce different results
    serializer2 = Serializer("different-secret-key")
    result_diff = serializer2.dumps(data)
    assert result != result_diff
    
    # Test with multiple secret keys (key rotation)
    serializer_rotation = Serializer(["old-key", "new-key"])
    result_rotation = serializer_rotation.dumps(data)
    assert isinstance(result_rotation, str)
    loaded_rotation = serializer_rotation.loads(result_rotation)
    assert loaded_rotation == data
    
    # Verify old keys can still verify
    loaded_old = serializer_rotation.loads(result_rotation)
    assert loaded_old == data
```


# LLM-generated content at query #54
#--------------------------

```python
def test_Serializer_dumps():
    serializer = Serializer("secret-key")
    
    # Test dumps with a simple object
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Contains signature separator
    
    # Test dumps with bytes serializer
    bytes_serializer = Serializer("secret-key", serializer=Serializer.default_serializer)
    bytes_serializer.is_text_serializer = False
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    
    # Test dumps with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert result != result_with_salt  # Different salt produces different signature
    
    # Test dumps with integer
    result_int = serializer.dumps(42)
    assert isinstance(result_int, str)
    
    # Test dumps with list
    result_list = serializer.dumps([1, 2, 3])
    assert isinstance(result_list, str)
    
    # Test dumps with None
    result_none = serializer.dumps(None)
    assert isinstance(result_none, str)
    
    # Verify the result can be loaded back
    loaded = serializer.loads(result)
    assert loaded == {"key": "value"}


# LLM-generated content at query #55
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # contains signature separator
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert b"." in result
    
    # Test with salt parameter
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert result_with_salt != result  # different salt produces different signature
    
    # Test that dumps returns a valid signed payload that can be verified
    loaded = serializer.loads(result)
    assert loaded == {"key": "value"}
    
    # Test with serializer kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    assert isinstance(result, str)
    loaded = serializer_with_kwargs.loads(result)
    assert loaded == {"b": 2, "a": 1}
    
    # Test with different data types
    assert serializer.loads(serializer.dumps(42)) == 42
    assert serializer.loads(serializer.dumps("string")) == "string"
    assert serializer.loads(serializer.dumps([1, 2, 3])) == [1, 2, 3]
    assert serializer.loads(serializer.dumps(None)) is None
```


# LLM-generated content at query #56
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps returns the expected type."""
    # Test with a text serializer (str)
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    text_serializer = TextSerializer()
    result = text_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'

    # Test with various data types
    assert text_serializer.dumps(None) == "null"
    assert text_serializer.dumps(42) == "42"
    assert text_serializer.dumps("hello") == '"hello"'
    assert text_serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
```


# LLM-generated content at query #57
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test loading valid JSON payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading valid JSON payload with explicit serializer
    result = serializer.load_payload(payload, serializer=json)
    assert result == {"key": "value"}
    
    # Test loading with binary serializer
    class BinarySerializer:
        def dumps(self, obj):
            return b"dummy"
        
        def loads(self, payload):
            return {"from": "binary"}
    
    binary_serializer = BinarySerializer()
    serializer_binary = Serializer("secret-key", serializer=binary_serializer)
    result = serializer_binary.load_payload(b"some_bytes")
    assert result == {"from": "binary"}
    
    # Test loading invalid payload raises BadPayload
    try:
        serializer.load_payload(b"invalid json")
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None
    
    # Test loading with custom serializer that raises exception
    class FailingSerializer:
        def dumps(self, obj):
            return b"test"
        
        def loads(self, payload):
            raise ValueError("Custom error")
    
    failing_serializer = FailingSerializer()
    try:
        serializer.load_payload(b"test", serializer=failing_serializer)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert isinstance(e.original_error, ValueError)
    
    # Test loading with text serializer (returns string)
    serializer_text = Serializer("secret-key", serializer=json)
    payload_text = b'{"text": "hello"}'
    result = serializer_text.load_payload(payload_text)
    assert result == {"text": "hello"}
    
    # Test loading empty payload
    try:
        serializer.load_payload(b"")
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
    
    # Test loading with explicit serializer that is text
    class CustomTextSerializer:
        def dumps(self, obj):
            return '{"custom": "data"}'
        
        def loads(self, payload):
            return {"custom": "data"}
    
    custom_serializer = CustomTextSerializer()
    result = serializer.load_payload(b'{"custom": "data"}', serializer=custom_serializer)
    assert result == {"custom": "data"}
```


# LLM-generated content at query #58
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a serializer that returns bytes for JSON dumps
    serializer = _PDataSerializer()
    
    # Test loading a payload
    test_data = {"key": "value"}
    dumped = serializer.dumps(test_data)
    loaded = serializer.loads(dumped)
    assert loaded == test_data
    
    # Test loading a bytes payload
    bytes_payload = b'{"key": "value"}'
    loaded_from_bytes = serializer.loads(bytes_payload)
    assert loaded_from_bytes == test_data
    
    # Test loading a string payload
    str_payload = '{"key": "value"}'
    loaded_from_str = serializer.loads(str_payload)
    assert loaded_from_str == test_data
    
    # Test loading empty data
    assert serializer.loads("{}") == {}
    assert serializer.loads(b"{}") == {}
    
    # Test loading list data
    list_data = [1, 2, 3]
    dumped_list = serializer.dumps(list_data)
    loaded_list = serializer.loads(dumped_list)
    assert loaded_list == list_data
    
    # Test loading None
    assert serializer.loads("null") is None
    assert serializer.loads(b"null") is None
    
    # Test loading with special characters
    special_data = {"text": "hello\nworld\t!"}
    dumped_special = serializer.dumps(special_data)
    loaded_special = serializer.loads(dumped_special)
    assert loaded_special == special_data
    
    # Test loading with unicode characters
    unicode_data = {"unicode": "üñîçødé"}
    dumped_unicode = serializer.dumps(unicode_data)
    loaded_unicode = serializer.loads(dumped_unicode)
    assert loaded_unicode == unicode_data
```


# LLM-generated content at query #59
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a serializer that returns str
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TextSerializer()
    assert serializer.loads('{"key": "value"}') == {"key": "value"}
    assert serializer.loads('123') == 123
    assert serializer.loads('"string"') == "string"
    assert serializer.loads('null') is None
    assert serializer.loads('true') is True
    assert serializer.loads('false') is False
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test with a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'123') == 123
    assert bytes_serializer.loads(b'"string"') == "string"
    assert bytes_serializer.loads(b'null') is None
    assert bytes_serializer.loads(b'true') is True
    assert bytes_serializer.loads(b'false') is False
    assert bytes_serializer.loads(b'[1, 2, 3]') == [1, 2, 3]
    
    # Test that loads raises appropriate exception for invalid payload
    import pytest
    with pytest.raises(json.JSONDecodeError):
        serializer.loads("invalid json")
    
    with pytest.raises(json.JSONDecodeError):
        bytes_serializer.loads(b"invalid json")


# LLM-generated content at query #60
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a concrete implementation of _PDataSerializer protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with simple values
    assert serializer.loads('"string"') == "string"
    assert serializer.loads('123') == 123
    assert serializer.loads('true') == True
    assert serializer.loads('null') is None
    
    # Test with list
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test with empty dict
    assert serializer.loads('{}') == {}
    
    # Test with nested structures
    nested = '{"a": {"b": [1, 2, {"c": 3}]}}'
    expected = {"a": {"b": [1, 2, {"c": 3}]}}
    assert serializer.loads(nested) == expected
    
    # Test that it raises appropriate exception for invalid input
    import json as json_module
    try:
        serializer.loads("invalid json")
        assert False, "Should have raised exception"
    except json_module.JSONDecodeError:
        pass
```


# LLM-generated content at query #61
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol provides proper loads method signature."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON string
    test_payload = '{"key": "value", "num": 42}'
    result = serializer.loads(test_payload)
    assert result == {"key": "value", "num": 42}
    
    # Test with list
    list_payload = '[1, 2, 3]'
    result = serializer.loads(list_payload)
    assert result == [1, 2, 3]
    
    # Test with simple types
    assert serializer.loads('"hello"') == "hello"
    assert serializer.loads('true') is True
    assert serializer.loads('null') is None
    assert serializer.loads('42') == 42
    
    # Test that it raises appropriate exception for invalid input
    with pytest.raises(json.JSONDecodeError):
        serializer.loads("{invalid json}")
```


# LLM-generated content at query #62
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
    assert string_serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert isinstance(string_serializer.dumps({}), str)
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.dumps({"key": "value"}) == b'{"key": "value"}'
    assert isinstance(bytes_serializer.dumps({}), bytes)
    
    # Test with different data types
    assert string_serializer.dumps(None) == "null"
    assert string_serializer.dumps(123) == "123"
    assert string_serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    
    # Test that dumps returns the correct type
    assert isinstance(string_serializer.dumps("test"), str)
    assert isinstance(bytes_serializer.dumps("test"), bytes)
```


# LLM-generated content at query #63
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
    assert s.loads(result) == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode()
        def loads(self, payload):
            return json.loads(payload)
    
    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = s_bytes.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert s_bytes.loads(result_bytes) == {"key": "value"}
    
    # Test with custom salt
    s_salt = Serializer("secret-key", salt=b"custom_salt")
    result_salt = s_salt.dumps({"key": "value"})
    assert s_salt.loads(result_salt) == {"key": "value"}
    
    # Test with salt parameter in dumps call
    s2 = Serializer("secret-key")
    result_salt_param = s2.dumps({"key": "value"}, salt=b"call_salt")
    assert s2.loads(result_salt_param, salt=b"call_salt") == {"key": "value"}
    
    # Verify that different salts produce different signatures
    result_default = s2.dumps({"key": "value"})
    assert result_default != result_salt_param
    
    # Test that dumps produces consistent results for same input
    result1 = s.dumps({"a": 1})
    result2 = s.dumps({"a": 1})
    assert result1 == result2
    
    # Test with empty dict
    result_empty = s.dumps({})
    assert s.loads(result_empty) == {}


# LLM-generated content at query #64
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test that _PDataSerializer protocol works with different serializer implementations
    class JSONSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = JSONSerializer()
    
    # Test basic JSON deserialization
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test list deserialization
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test primitive types
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    result = serializer.loads('42')
    assert result == 42
    
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('null')
    assert result is None

```


# LLM-generated content at query #65
#--------------------------

```python
def test_Serializer_load_payload():
    """Test load_payload method of Serializer class."""
    serializer = Serializer("test-secret-key")
    
    # Test with valid JSON payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with text serializer (JSON returns str)
    text_serializer = Serializer("test-secret-key", serializer=json)
    text_payload = b'{"hello": "world"}'
    result = text_serializer.load_payload(text_payload)
    assert result == {"hello": "world"}
    
    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload):
            return {"data": payload.decode()}
        def dumps(self, obj):
            return b"test"
    
    bytes_serializer = Serializer("test-secret-key", serializer=BytesSerializer())
    bytes_payload = b"custom_data"
    result = bytes_serializer.load_payload(bytes_payload)
    assert result == {"data": "custom_data"}
    
    # Test with custom serializer passed as parameter
    custom_serializer = Serializer("test-secret-key")
    custom_payload = b'[1, 2, 3]'
    result = custom_serializer.load_payload(custom_payload, serializer=json)
    assert result == [1, 2, 3]
    
    # Test BadPayload exception with invalid JSON
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"invalid json")
    assert "Could not load the payload" in str(exc_info.value)
    assert exc_info.value.original_error is not None
    
    # Test BadPayload with non-decodable bytes for text serializer
    with pytest.raises(BadPayload):
        serializer.load_payload(b"\xff\xfe\xff")
    
    # Test with empty bytes
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test with bytes serializer that expects bytes
    class BytesOnlySerializer:
        def loads(self, payload):
            if not isinstance(payload, bytes):
                raise TypeError("Expected bytes")
            return payload
        def dumps(self, obj):
            return b""
    
    bytes_only = Serializer("test-secret-key", serializer=BytesOnlySerializer())
    result = bytes_only.load_payload(b"hello bytes")
    assert result == b"hello bytes"
    
    # Test is_text_serializer flag usage
    text_s = Serializer("test-secret-key")
    assert text_s.is_text_serializer is True
    
    # Test with a bytes serializer that returns non-text
    class NonTextSerializer:
        def loads(self, payload):
            return payload
        def dumps(self, obj):
            return b"test"
    
    non_text = Serializer("test-secret-key", serializer=NonTextSerializer())
    assert non_text.is_text_serializer is False
    
    # Verify the exception chain
    try:
        serializer.load_payload(b"not json")
    except BadPayload as e:
        assert isinstance(e.original_error, json.JSONDecodeError)
```


# LLM-generated content at query #66
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test successful load with bytes payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test successful load with text serializer (payload decoded from UTF-8)
    payload = b'{"number": 42}'
    result = serializer.load_payload(payload)
    assert result == {"number": 42}
    
    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload):
            return {"data": payload}
        def dumps(self, obj):
            return b"test"
    
    bytes_serializer = BytesSerializer()
    serializer_bytes = Serializer("secret-key", serializer=bytes_serializer)
    
    # Test successful load with bytes serializer
    payload = b"raw bytes data"
    result = serializer_bytes.load_payload(payload)
    assert result == {"data": b"raw bytes data"}
    
    # Test with custom serializer that returns text
    class TextSerializer:
        def loads(self, payload):
            return {"text": payload}
        def dumps(self, obj):
            return "test"
    
    text_serializer = TextSerializer()
    serializer_text = Serializer("secret-key", serializer=text_serializer)
    
    # Test successful load with text serializer
    payload = b"hello world"
    result = serializer_text.load_payload(payload)
    assert result == {"text": "hello world"}
    
    # Test with explicit serializer parameter overriding stored serializer
    class OverrideSerializer:
        def loads(self, payload):
            return {"overridden": payload}
        def dumps(self, obj):
            return "test"
    
    override_serializer = OverrideSerializer()
    result = serializer.load_payload(b"test", serializer=override_serializer)
    assert result == {"overridden": b"test"}
    
    # Test BadPayload exception with invalid JSON
    try:
        serializer.load_payload(b"invalid json")
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None
    
    # Test BadPayload exception with empty payload
    try:
        serializer.load_payload(b"")
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test BadPayload exception with non-decodable bytes for text serializer
    try:
        serializer.load_payload(b"\xff\xfe\x00\x01")
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test with serializer that raises arbitrary exception
    class FailingSerializer:
        def loads(self, payload):
            raise ValueError("Custom error")
        def dumps(self, obj):
            return "test"
    
    failing_serializer = FailingSerializer()
    serializer_fail = Serializer("secret-key", serializer=failing_serializer)
    
    try:
        serializer_fail.load_payload(b"test")
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)
        assert str(e.original_error) == "Custom error"
```


# LLM-generated content at query #67
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            if isinstance(payload, str):
                return {"data": payload}
            return {"data": payload.decode("utf-8")}
        
        def dumps(self, obj):
            return json.dumps(obj)

    serializer = MockSerializer()
    
    # Test with string payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes payload - _PDataSerializer protocol expects str or bytes
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with different data types
    result = serializer.loads("42")
    assert result == {"data": "42"}
    
    result = serializer.loads("true")
    assert result == {"data": "true"}
    
    # Test with complex nested data
    complex_data = '{"nested": {"list": [1, 2, 3]}, "value": "test"}'
    result = serializer.loads(complex_data)
    assert result == {"nested": {"list": [1, 2, 3]}, "value": "test"}
    
    # Test that the method is callable and returns something
    assert callable(serializer.loads)
    
    # Verify the return type is Any (can be any type)
    result = serializer.loads("null")
    assert result is None or result == {"data": "null"}
```


# LLM-generated content at query #68
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a mock serializer that implements the protocol
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    test_data = {"key": "value", "number": 42}
    
    result = serializer.dumps(test_data)
    
    assert isinstance(result, str)
    assert result == '{"key": "value", "number": 42}'
    assert json.loads(result) == test_data

    # Test with different data types
    result_list = serializer.dumps([1, 2, 3])
    assert result_list == "[1, 2, 3]"
    
    result_string = serializer.dumps("hello")
    assert result_string == '"hello"'
    
    result_none = serializer.dumps(None)
    assert result_none == "null"
```


# LLM-generated content at query #69
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns dicts
    class TestSerializer:
        def loads(self, payload: str) -> dict:
            return {"parsed": payload}

        def dumps(self, obj) -> str:
            return str(obj)

    serializer = TestSerializer()
    result = serializer.loads("test_data")
    assert result == {"parsed": "test_data"}

    # Test with a serializer that handles bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> list:
            return [1, 2, 3]

        def dumps(self, obj) -> bytes:
            return b"test"

    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b"some_bytes")
    assert result == [1, 2, 3]

    # Test with a serializer that raises exception
    class FailingSerializer:
        def loads(self, payload: str) -> None:
            raise ValueError("Failed to load")

        def dumps(self, obj) -> str:
            return str(obj)

    failing_serializer = FailingSerializer()
    try:
        failing_serializer.loads("test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Failed to load"
```


# LLM-generated content at query #70
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a simple serializer that implements the protocol
    class TestTextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    class TestBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    # Test text serializer
    text_serializer = TestTextSerializer()
    result = text_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test bytes serializer
    bytes_serializer = TestBytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with different data types
    assert text_serializer.dumps(42) == '42'
    assert text_serializer.dumps([1, 2, 3]) == '[1, 2, 3]'
    assert text_serializer.dumps(None) == 'null'
    
    # Test empty data
    assert text_serializer.dumps({}) == '{}'
    assert text_serializer.dumps([]) == '[]'
    
    # Test that dumps returns string type for text serializer
    assert is_text_serializer(text_serializer)
    assert not is_text_serializer(bytes_serializer)


# LLM-generated content at query #71
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
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
    s2 = Serializer("secret-key")
    result_with_salt = s2.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert s2.loads(result_with_salt, salt="custom-salt") == {"key": "value"}
    
    # Test with serializer_kwargs
    s3 = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    result3 = s3.dumps({"b": 2, "a": 1})
    assert isinstance(result3, str)
    assert s3.loads(result3) == {"b": 2, "a": 1}
    
    # Test that dumps produces different results for different secret keys
    s4 = Serializer("secret-key-1")
    s5 = Serializer("secret-key-2")
    data = {"test": "data"}
    result4 = s4.dumps(data)
    result5 = s5.dumps(data)
    assert result4 != result5
    
    # Test that dumps produces different results for different salts
    s6 = Serializer("secret-key")
    result6 = s6.dumps(data, salt="salt-1")
    result7 = s6.dumps(data, salt="salt-2")
    assert result6 != result7
    
    # Test with empty data
    s7 = Serializer("secret-key")
    result_empty = s7.dumps({})
    assert isinstance(result_empty, str)
    assert s7.loads(result_empty) == {}
    
    # Test with list of secret keys (key rotation)
    s8 = Serializer(["old-key", "new-key"])
    result8 = s8.dumps({"data": "test"})
    assert isinstance(result8, str)
    assert s8.loads(result8) == {"data": "test"}
    # Verify it was signed with the newest key
    assert s8.secret_keys[-1] == b"new-key"


# LLM-generated content at query #72
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # contains signature separator
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    
    # Test that dumps returns signable payload
    signed = serializer.dumps("test")
    assert signed != "test"
    assert isinstance(signed, str)
    
    # Test with custom salt
    result_with_salt = serializer.dumps("test", salt="custom-salt")
    result_default = serializer.dumps("test")
    assert result_with_salt != result_default
    
    # Test that dumps output can be loaded back
    data = {"nested": {"list": [1, 2, 3]}, "bool": True}
    signed_data = serializer.dumps(data)
    loaded = serializer.loads(signed_data)
    assert loaded == data
    
    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_kwargs = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    # Should produce compact JSON with sorted keys
    assert ",:," not in result_kwargs  # no spaces
    # Verify it can be loaded back
    loaded_kwargs = serializer_with_kwargs.loads(result_kwargs)
    assert loaded_kwargs == {"a": 1, "b": 2}  # order doesn't matter for dict
    
    # Test with multiple secret keys
    multi_key_serializer = Serializer(["old-key", "new-key"])
    signed_multi = multi_key_serializer.dumps("test")
    # Should verify with any key
    loaded_multi = multi_key_serializer.loads(signed_multi)
    assert loaded_multi == "test"


# LLM-generated content at query #73
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Test with str serializer
    class StrSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StrSerializer()
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
    
    # Test with custom serializer
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            parts = payload.split("|")
            return {"a": int(parts[0]), "b": int(parts[1])}
        
        def dumps(self, obj: t.Any) -> str:
            return f"{obj['a']}|{obj['b']}"
    
    custom_serializer = CustomSerializer()
    result = custom_serializer.loads("1|2")
    assert result == {"a": 1, "b": 2}
    
    # Test type annotation compliance - the loads method should accept _TSerialized
    # and return Any
    assert callable(getattr(serializer, 'loads', None))
```


# LLM-generated content at query #74
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol handles loads method correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with primitive types
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    result = serializer.loads('42')
    assert result == 42
    
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('null')
    assert result is None
    
    # Test with empty JSON
    result = serializer.loads('{}')
    assert result == {}
    
    # Test that it conforms to the protocol
    assert isinstance(serializer, _PDataSerializer)
```


# LLM-generated content at query #75
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default configuration (no fallback signers)
    serializer = Serializer("secret-key")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"secret-key"]

    # Test with custom salt
    serializer = Serializer("secret-key", salt=b"custom-salt")
    unsigners = list(serializer.iter_unsigners())
    assert unsigners[0].salt == b"custom-salt"

    # Test with fallback signers as dict
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"digest_method": "sha256"}]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2  # main signer + fallback
    assert isinstance(unsigners[1], Signer)

    # Test with fallback signers as tuple
    class CustomSigner(Signer):
        pass
    
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(CustomSigner, {"key_derivation": "hmac"})]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[1], CustomSigner)

    # Test with fallback signers as Signer class
    serializer = Serializer(
        "secret-key",
        fallback_signers=[CustomSigner]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[1], CustomSigner)

    # Test with multiple secret keys (key rotation)
    serializer = Serializer(["old-key", "new-key"])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert unsigners[0].secret_keys == [b"old-key", b"new-key"]

    # Test with multiple secret keys and fallback signers
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"digest_method": "sha256"}]
    )
    unsigners = list(serializer.iter_unsigners())
    # Should yield: 1 main signer + 2 fallback signers (one per secret key)
    assert len(unsigners) == 3
    assert isinstance(unsigners[0], Signer)  # main signer
    assert isinstance(unsigners[1], Signer)  # fallback with old-key
    assert isinstance(unsigners[2], Signer)  # fallback with new-key

    # Test with overridden salt parameter
    serializer = Serializer("secret-key", salt=b"default-salt")
    unsigners = list(serializer.iter_unsigners(salt=b"override-salt"))
    assert unsigners[0].salt == b"override-salt"

    # Test with empty fallback signers list
    serializer = Serializer("secret-key", fallback_signers=[])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1

    # Test with None salt (should use the signer's default)
    serializer = Serializer("secret-key", salt=None)
    unsigners = list(serializer.iter_unsigners())
    assert unsigners[0].salt is None

    # Test that fallback signers are iterated correctly with multiple keys
    serializer = Serializer(
        ["key1", "key2", "key3"],
        fallback_signers=[{"digest_method": "sha256"}, CustomSigner]
    )
    unsigners = list(serializer.iter_unsigners())
    # 1 main signer + 3 fallback (dict) + 3 fallback (class) = 7
    assert len(unsigners) == 7
    # First is main signer
    assert isinstance(unsigners[0], Signer)
    # Next three are from dict fallback (one per key)
    assert all(isinstance(u, Signer) for u in unsigners[1:4])
    # Last three are from CustomSigner fallback (one per key)
    assert all(isinstance(u, CustomSigner) for u in unsigners[4:7])
```


# LLM-generated content at query #76
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol requires a dumps method."""
    # Create a minimal serializer that conforms to the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == '{"key": "value"}'
    assert isinstance(result, str)

    # Test with bytes serializer
    class TestBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = TestBytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert result == b'{"key": "value"}'
    assert isinstance(result, bytes)


# LLM-generated content at query #77
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with str serializer
    class StrSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StrSerializer()
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
    
    # Test with different data types
    result = serializer.loads('123')
    assert result == 123
    
    result = serializer.loads('"string"')
    assert result == "string"
    
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('null')
    assert result is None
    
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]


# LLM-generated content at query #78
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    
    # Test with a concrete implementation that matches the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
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
    
    # Test with string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test with empty string should raise an error
    with pytest.raises(Exception):
        serializer.loads('')
    
    # Test with invalid JSON should raise an error
    with pytest.raises(Exception):
        serializer.loads('{invalid}')
```


# LLM-generated content at query #79
#--------------------------

```python
def test_Serializer_dumps():
    serializer = Serializer("secret-key")
    
    # Test with default json serializer (text)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Contains signature
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return b'{"key":"value"}'
        def loads(self, payload):
            return json.loads(payload.decode())
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    # Test with custom serializer_kwargs
    serializer_kwargs = {"sort_keys": True, "separators": (",", ":")}
    custom_serializer = Serializer("secret-key", serializer_kwargs=serializer_kwargs)
    result_custom = custom_serializer.dumps({"b": 1, "a": 2})
    assert result_custom.count(":") == 2  # No spaces after separators
    
    # Test with salt parameter
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert result_with_salt != result  # Different salt produces different signature
    
    # Test with different payload types
    assert isinstance(serializer.dumps(None), str)
    assert isinstance(serializer.dumps(123), str)
    assert isinstance(serializer.dumps("string"), str)
    assert isinstance(serializer.dumps([1, 2, 3]), str)```


# LLM-generated content at query #80
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test that the protocol is structural and can be satisfied by various implementations
    import json
    from itsdangerous.serializer import _PDataSerializer
    
    # Test with json serializer (standard text serializer)
    json_serializer: _PDataSerializer[str] = json
    result = json_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> dict:
            return {"data": payload.decode("utf-8")}
        
        def dumps(self, obj: dict) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer: _PDataSerializer[bytes] = BytesSerializer()
    result = bytes_serializer.loads(b'{"data": "test"}')
    assert result == {"data": "test"}
    
    # Test with custom serializer that returns different types
    class IntSerializer:
        def loads(self, payload: str) -> int:
            return int(payload)
        
        def dumps(self, obj: int) -> str:
            return str(obj)
    
    int_serializer: _PDataSerializer[str] = IntSerializer()
    result = int_serializer.loads("42")
    assert result == 42
    assert isinstance(result, int)
    
    # Test that loads raises appropriate exceptions for malformed data
    with pytest.raises(json.JSONDecodeError):
        json_serializer.loads("invalid json")
    
    # Test with empty payload
    result = json_serializer.loads("null")
    assert result is None
    
    result = json_serializer.loads("[]")
    assert result == []
    
    result = json_serializer.loads("{}")
    assert result == {}
    
    # Test with various JSON types
    result = json_serializer.loads('"string"')
    assert result == "string"
    
    result = json_serializer.loads("123")
    assert result == 123
    
    result = json_serializer.loads("true")
    assert result is True
    
    result = json_serializer.loads("false")
    assert result is False
    
    # Test with whitespace in payload
    result = json_serializer.loads('  {"key": "value"}  ')
    assert result == {"key": "value"}
```


# LLM-generated content at query #81
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns str
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TextSerializer()
    assert serializer.loads('{"key": "value"}') == {"key": "value"}
    assert serializer.loads('42') == 42
    assert serializer.loads('"test"') == "test"
    assert serializer.loads('null') is None
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'42') == 42

    # Test with custom serializer that handles special types
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            if payload.startswith('int:'):
                return int(payload[4:])
            elif payload.startswith('float:'):
                return float(payload[6:])
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            if isinstance(obj, int):
                return f'int:{obj}'
            elif isinstance(obj, float):
                return f'float:{obj}'
            return str(obj)
    
    custom_serializer = CustomSerializer()
    assert custom_serializer.loads('int:42') == 42
    assert custom_serializer.loads('float:3.14') == 3.14
    assert custom_serializer.loads('plain text') == 'plain text'

    # Test that loads raises appropriate exceptions for invalid input
    class StrictSerializer:
        def loads(self, payload: str) -> t.Any:
            if not payload:
                raise ValueError("Empty payload")
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    strict_serializer = StrictSerializer()
    
    # Test with invalid JSON
    try:
        strict_serializer.loads('invalid json')
        assert False, "Expected exception"
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Test with empty payload
    try:
        strict_serializer.loads('')
        assert False, "Expected exception"
    except ValueError:
        pass

    # Test that _PDataSerializer protocol is satisfied
    protocol_check: _PDataSerializer[str] = TextSerializer()
    assert isinstance(protocol_check, _PDataSerializer)
    assert protocol_check.loads('"test"') == "test"
```


# LLM-generated content at query #82
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer loads method works correctly."""
    # Create a concrete implementation of _PDataSerializer
    class TestSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON payload
    payload = '{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}
    
    # Test with integer payload
    payload = "42"
    result = serializer.loads(payload)
    assert result == 42
    
    # Test with list payload
    payload = "[1, 2, 3]"
    result = serializer.loads(payload)
    assert result == [1, 2, 3]
    
    # Test with string payload
    payload = '"hello"'
    result = serializer.loads(payload)
    assert result == "hello"
    
    # Test with boolean payload
    payload = "true"
    result = serializer.loads(payload)
    assert result is True
    
    # Test with null payload
    payload = "null"
    result = serializer.loads(payload)
    assert result is None
    
    # Test with nested payload
    payload = '{"outer": {"inner": [1, 2, 3]}}'
    result = serializer.loads(payload)
    assert result == {"outer": {"inner": [1, 2, 3]}}
    
    # Test that it raises on invalid JSON
    import pytest
    with pytest.raises(Exception):
        serializer.loads("invalid json")
```


# LLM-generated content at query #83
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test loading valid payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading with custom serializer
    class CustomSerializer:
        def loads(self, payload):
            return {"custom": payload}
        def dumps(self, obj):
            return str(obj)
    
    custom_serializer = Serializer("secret-key", serializer=CustomSerializer())
    result = custom_serializer.load_payload(b"test_data")
    assert result == {"custom": b"test_data"}
    
    # Test loading invalid payload raises BadPayload
    invalid_payload = b"invalid json"
    try:
        serializer.load_payload(invalid_payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None
    
    # Test loading bytes payload with text serializer that fails
    class FailingSerializer:
        def loads(self, payload):
            raise ValueError("Load failed")
        def dumps(self, obj):
            return str(obj)
    
    failing_serializer = Serializer("secret-key", serializer=FailingSerializer())
    try:
        failing_serializer.load_payload(b"test")
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert isinstance(e.original_error, ValueError)
    
    # Test with explicit serializer parameter
    class ExplicitSerializer:
        def loads(self, payload):
            return {"explicit": payload}
        def dumps(self, obj):
            return str(obj)
    
    result = serializer.load_payload(b'{"test": 1}', serializer=ExplicitSerializer())
    assert result == {"explicit": b'{"test": 1}'}
    
    # Test with binary serializer (returns bytes)
    class BytesSerializer:
        def loads(self, payload):
            return payload
        def dumps(self, obj):
            return obj
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.load_payload(b"binary_data")
    assert result == b"binary_data"
```


# LLM-generated content at query #84
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test successful load with valid payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return payload.decode("utf-8")
        def dumps(self, obj):
            return obj.encode("utf-8")
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    bytes_payload = b"test data"
    result = bytes_serializer.load_payload(bytes_payload)
    assert result == "test data"
    
    # Test with custom serializer passed as parameter
    class CustomSerializer:
        def loads(self, payload):
            return {"custom": payload.decode("utf-8")}
        def dumps(self, obj):
            return str(obj).encode("utf-8")
    
    custom_serializer = CustomSerializer()
    result = serializer.load_payload(b"test", serializer=custom_serializer)
    assert result == {"custom": "test"}
    
    # Test BadPayload raised for invalid json
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"invalid json")
    assert "Could not load the payload" in str(exc_info.value)
    assert exc_info.value.original_error is not None
    
    # Test BadPayload raised for empty payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test BadPayload raised for payload with utf-8 decode error
    with pytest.raises(BadPayload):
        serializer.load_payload(b"\xff\xfe")
    
    # Test with text serializer that returns str
    class TextSerializer:
        def loads(self, payload):
            return payload.upper()
        def dumps(self, obj):
            return obj.lower()
    
    text_serializer = Serializer("secret-key", serializer=TextSerializer())
    result = text_serializer.load_payload(b"hello")
    assert result == "HELLO"
    
    # Test with bytes serializer that returns bytes
    class BytesSerializer2:
        def loads(self, payload):
            return payload
        def dumps(self, obj):
            return obj if isinstance(obj, bytes) else obj.encode()
    
    bytes_serializer2 = Serializer("secret-key", serializer=BytesSerializer2())
    result = bytes_serializer2.load_payload(b"\x00\x01\x02")
    assert result == b"\x00\x01\x02"
```


# LLM-generated content at query #85
#--------------------------

```python
def test_Serializer_iter_unsigners():
    """Test iter_unsigners yields signers in correct order and configuration."""
    secret_key = b"test-secret-key"
    salt = b"test-salt"
    fallback_signer = Signer
    fallback_kwargs = {"digest_method": "sha256"}
    
    # Test with no fallback signers
    serializer = Serializer(secret_key, salt=salt)
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [secret_key]
    assert signers[0].salt == salt
    
    # Test with fallback signers as dict
    serializer = Serializer(
        secret_key,
        salt=salt,
        fallback_signers=[fallback_kwargs]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].salt == salt
    assert signers[1].secret_keys == [secret_key]
    assert signers[1].salt == salt
    
    # Test with fallback signers as tuple (signer class, kwargs)
    serializer = Serializer(
        secret_key,
        salt=salt,
        fallback_signers=[(fallback_signer, fallback_kwargs)]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], Signer)
    assert signers[1].secret_keys == [secret_key]
    assert signers[1].salt == salt
    
    # Test with fallback signers as signer class
    serializer = Serializer(
        secret_key,
        salt=salt,
        fallback_signers=[fallback_signer]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], Signer)
    assert signers[1].secret_keys == [secret_key]
    assert signers[1].salt == salt
    
    # Test with multiple fallback signers
    serializer = Serializer(
        secret_key,
        salt=salt,
        fallback_signers=[
            fallback_kwargs,
            (fallback_signer, fallback_kwargs),
            fallback_signer
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4  # 1 default + 3 fallbacks
    
    # Test with custom salt parameter
    custom_salt = b"custom-salt"
    signers = list(serializer.iter_unsigners(salt=custom_salt))
    assert all(s.salt == custom_salt for s in signers)
    
    # Test with multiple secret keys
    secret_keys = [b"old-key", b"newer-key", b"newest-key"]
    serializer = Serializer(secret_keys, salt=salt)
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == secret_keys
    
    # Test that iter_unsigners yields a generator
    serializer = Serializer(secret_key, salt=salt)
    result = serializer.iter_unsigners()
    from collections.abc import Iterator
    assert isinstance(result, Iterator)
```


# LLM-generated content at query #86
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert serializer.loads(result) == {"key": "value"}

    # Test with binary serializer
    class BinarySerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    binary_serializer = Serializer("secret-key", serializer=BinarySerializer())
    binary_result = binary_serializer.dumps({"key": "value"})
    assert isinstance(binary_result, bytes)
    assert binary_serializer.loads(binary_result) == {"key": "value"}

    # Test with custom salt
    custom_salt_result = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(custom_salt_result, str)
    assert serializer.loads(custom_salt_result, salt="custom-salt") == {"key": "value"}

    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_with_kwargs = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    assert isinstance(result_with_kwargs, str)
    assert serializer_with_kwargs.loads(result_with_kwargs) == {"a": 1, "b": 2}

    # Test that dumps returns consistent format
    first_result = serializer.dumps({"test": "data"})
    second_result = serializer.dumps({"test": "data"})
    assert first_result != second_result  # Signatures should differ due to timestamp
```


# LLM-generated content at query #87
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol correctly defines the dumps method."""
    import pytest
    
    # Create a concrete serializer that implements the protocol
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
    
    # Test with list
    result = serializer.dumps([1, 2, 3])
    assert result == '[1, 2, 3]'
    
    # Test with simple types
    result = serializer.dumps("test")
    assert result == '"test"'
    
    result = serializer.dumps(123)
    assert result == '123'
    
    result = serializer.dumps(None)
    assert result == 'null'
    
    # Test that it returns str (protocol requirement)
    assert isinstance(serializer.dumps({}), str)
```


# LLM-generated content at query #88
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a simple serializer that implements the protocol
    class TestSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return str(obj)
    
    serializer = TestSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == "{'key': 'value'}"
    assert isinstance(result, str)
```


# LLM-generated content at query #89
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test that _PDataSerializer protocol is satisfied by json module
    serializer = json
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'

    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()

    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'

    # Test with custom serializer that returns str
    class StrSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)

    str_serializer = StrSerializer()
    result = str_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'

    # Test dumps with different data types
    assert serializer.dumps(123) == '123'
    assert serializer.dumps([1, 2, 3]) == '[1, 2, 3]'
    assert serializer.dumps(None) == 'null'
    assert serializer.dumps(True) == 'true'
```


# LLM-generated content at query #90
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a simple serializer that implements the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    test_data = {"key": "value", "number": 42}
    
    # Test that dumps returns a string
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    
    # Test that the result is valid JSON
    assert json.loads(result) == test_data
    
    # Test with different data types
    assert serializer.dumps("string") == '"string"'
    assert serializer.dumps(123) == "123"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps(None) == "null"
    
    # Test empty data
    assert serializer.dumps({}) == "{}"
    assert serializer.dumps([]) == "[]"
```


# LLM-generated content at query #91
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol is properly implemented by json serializer."""
    # Test with json serializer (which is the default)
    serializer = json
    
    # Test dumps with a simple dict
    data = {"key": "value"}
    result = serializer.dumps(data)
    
    # Verify the result is a string
    assert isinstance(result, str)
    
    # Verify the result can be loaded back
    assert json.loads(result) == data
    
    # Test dumps with a list
    data_list = [1, 2, 3]
    result_list = serializer.dumps(data_list)
    assert isinstance(result_list, str)
    assert json.loads(result_list) == data_list
    
    # Test dumps with None
    result_none = serializer.dumps(None)
    assert isinstance(result_none, str)
    assert json.loads(result_none) is None
    
    # Test dumps with empty dict
    result_empty = serializer.dumps({})
    assert isinstance(result_empty, str)
    assert json.loads(result_empty) == {}


# LLM-generated content at query #92
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns a known value
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"test": "data"}
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    result = serializer.loads('{"test": "data"}')
    assert result == {"test": "data"}
    
    # Test with a bytes serializer
    class TestBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = TestBytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer payload
    class TestIntSerializer:
        def loads(self, payload: str) -> t.Any:
            return int(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    int_serializer = TestIntSerializer()
    result = int_serializer.loads("42")
    assert result == 42
    
    # Test with list payload
    class TestListSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    list_serializer = TestListSerializer()
    result = list_serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with None payload
    class TestNoneSerializer:
        def loads(self, payload: str) -> t.Any:
            return None
        
        def dumps(self, obj: t.Any) -> str:
            return "null"
    
    none_serializer = TestNoneSerializer()
    result = none_serializer.loads("null")
    assert result is None
    
    # Test that loads raises appropriate exception for invalid data
    class TestErrorSerializer:
        def loads(self, payload: str) -> t.Any:
            raise ValueError("Invalid data")
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    error_serializer = TestErrorSerializer()
    try:
        error_serializer.loads("invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass  # Expected exception
```


# LLM-generated content at query #93
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns a string
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    serializer = StringSerializer()
    result = serializer.loads("hello")
    assert result == "HELLO"
    
    # Test with a JSON serializer
    json_serializer = json
    result = json_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode("utf-8")
        
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b"test data")
    assert result == "test data"
```


# LLM-generated content at query #94
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol defines dumps method correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str | bytes) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any) -> str | bytes:
            if isinstance(obj, str):
                return obj.encode('utf-8')
            return obj
    
    serializer = TestSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, (str, bytes))
```


# LLM-generated content at query #95
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test that the protocol method signature is callable
    serializer = _PDataSerializer()
    
    # Mock the loads method
    def mock_loads(payload):
        return {"key": "value"}
    serializer.loads = mock_loads
    
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with string payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with different payload types
    result = serializer.loads(b'123')
    assert result == {"key": "value"}
    
    result = serializer.loads('123')
    assert result == {"key": "value"}
```


# LLM-generated content at query #96
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a simple serializer that implements the protocol
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
    
    # Test serialization of various types
    assert serializer.dumps(42) == "42"
    assert serializer.dumps("hello") == '"hello"'
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps(None) == "null"
```


# LLM-generated content at query #97
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a concrete implementation of _PDataSerializer for testing
    class TestSerializer:
        def loads(self, payload: str | bytes) -> t.Any:
            if isinstance(payload, bytes):
                return payload.decode("utf-8")
            return payload
        
        def dumps(self, obj: t.Any) -> str | bytes:
            if isinstance(obj, str):
                return obj
            return str(obj)
    
    serializer = TestSerializer()
    
    # Test with string payload
    result = serializer.loads("hello")
    assert result == "hello"
    
    # Test with bytes payload
    result = serializer.loads(b"world")
    assert result == "world"
    
    # Test with JSON-like string
    result = serializer.loads('{"key": "value"}')
    assert result == '{"key": "value"}'
    
    # Test with numeric string
    result = serializer.loads("12345")
    assert result == "12345"
    
    # Test with empty string
    result = serializer.loads("")
    assert result == ""
    
    # Test with empty bytes
    result = serializer.loads(b"")
    assert result == ""
    
    # Test with special characters
    result = serializer.loads("héllo wörld")
    assert result == "héllo wörld"
    
    # Test with unicode bytes
    result = serializer.loads(b"\xc3\xa9")
    assert result == "é"
    
    # Test with None payload (should work if serializer handles it)
    try:
        result = serializer.loads(None)
        assert False, "Expected TypeError"
    except TypeError:
        pass
    except Exception:
        pass
```


# LLM-generated content at query #98
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol requires a dumps method."""
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer: _PDataSerializer[str] = MockSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == '{"key": "value"}'
    assert isinstance(result, str)


# LLM-generated content at query #99
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            if payload == "valid":
                return {"key": "value"}
            elif payload == "number":
                return 42
            elif payload == "list":
                return [1, 2, 3]
            elif payload == "none":
                return None
            elif payload == "empty":
                return ""
            raise ValueError("Invalid payload")
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON-like string
    result = serializer.loads("valid")
    assert result == {"key": "value"}
    
    # Test with number string
    result = serializer.loads("number")
    assert result == 42
    
    # Test with list string
    result = serializer.loads("list")
    assert result == [1, 2, 3]
    
    # Test with None string
    result = serializer.loads("none")
    assert result is None
    
    # Test with empty string
    result = serializer.loads("empty")
    assert result == ""
    
    # Test that the protocol accepts bytes payload
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode("utf-8")
        
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b"test")
    assert result == "test"
    
    # Test that loads can handle complex nested structures
    class ComplexSerializer:
        def loads(self, payload: str) -> t.Any:
            import json
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            import json
            return json.dumps(obj)
    
    complex_serializer = ComplexSerializer()
    data = '{"name": "test", "values": [1, 2, 3], "nested": {"a": 1}}'
    result = complex_serializer.loads(data)
    assert result == {"name": "test", "values": [1, 2, 3], "nested": {"a": 1}}
    
    # Test error handling for invalid input
    with pytest.raises(ValueError, match="Invalid payload"):
        serializer.loads("invalid")
```


# LLM-generated content at query #100
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON-like serializer
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"data": payload}
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    serializer = MockSerializer()
    
    # Test basic loading
    result = serializer.loads('{"key": "value"}')
    assert result == {"data": '{"key": "value"}'}
    
    # Test with integer payload
    result = serializer.loads("123")
    assert result == {"data": "123"}
    
    # Test with empty string
    result = serializer.loads("")
    assert result == {"data": ""}
    
    # Test with special characters
    result = serializer.loads("test!@#$%^&*()")
    assert result == {"data": "test!@#$%^&*()"}
    
    # Test with unicode
    result = serializer.loads("héllo wörld")
    assert result == {"data": "héllo wörld"}
```


# LLM-generated content at query #101
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Create a mock serializer that conforms to _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with various data types
    test_cases = [
        {"key": "value"},
        [1, 2, 3],
        "string",
        123,
        None,
        True,
        {"nested": {"data": [1, 2, 3]}}
    ]
    
    for obj in test_cases:
        # Call dumps method
        result = serializer.dumps(obj)
        
        # Verify result is a string
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        
        # Verify we can load it back
        loaded = serializer.loads(result)
        assert loaded == obj, f"Round trip failed for {obj}: got {loaded}"
    
    # Test that dumps raises appropriate error for non-serializable objects
    class NonSerializable:
        pass
    
    with pytest.raises(TypeError):
        serializer.dumps(NonSerializable())
```


# LLM-generated content at query #102
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON-like serializer
    class MockSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test loading a valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading a simple value
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test loading an integer
    result = serializer.loads('42')
    assert result == 42
    
    # Test loading None
    result = serializer.loads('null')
    assert result is None
    
    # Test loading a boolean
    result = serializer.loads('true')
    assert result is True
    
    # Test loading bytes (should still work if bytes contains valid JSON)
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}

    # Test that loads returns Any type
    import typing as t
    # Verify the protocol is satisfied
    assert isinstance(serializer, _PDataSerializer) is True
```


# LLM-generated content at query #103
#--------------------------

```python
def test_Serializer_dumps():
    """Test Serializer.dumps method."""
    # Test with default JSON serializer (text)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # contains separator between payload and signature
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes):
            return json.loads(payload.decode("utf-8"))
        def dumps(self, obj) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = s_bytes.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    # Test with custom serializer kwargs
    s_kwargs = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    result_kwargs = s_kwargs.dumps({"b": 2, "a": 1})
    assert result_kwargs is not None
    
    # Test with salt parameter
    s_salt = Serializer("secret-key")
    result1 = s_salt.dumps({"key": "value"}, salt="custom-salt")
    result2 = s_salt.dumps({"key": "value"}, salt="different-salt")
    assert result1 != result2  # different salts produce different signatures
    
    # Test that payload is serialized correctly
    s = Serializer("secret-key")
    result = s.dumps([1, 2, 3])
    assert result is not None
    
    # Test with empty dict
    result_empty = s.dumps({})
    assert result_empty is not None
    
    # Test round-trip works
    s = Serializer("secret-key")
    original = {"test": "data", "number": 42}
    dumped = s.dumps(original)
    assert s.loads(dumped) == original
```


# LLM-generated content at query #104
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
    assert serializer.loads('{"key": "value"}') == {"key": "value"}
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    assert serializer.loads('"text"') == "text"
    assert serializer.loads('42') == 42
    assert serializer.loads('null') is None
    assert serializer.loads('true') is True
    assert serializer.loads('false') is False

    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'[1, 2, 3]') == [1, 2, 3]
    assert bytes_serializer.loads(b'"text"') == "text"

    # Test with custom object
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            if payload.startswith("custom:"):
                return payload[7:]
            raise ValueError("Invalid format")
        
        def dumps(self, obj: t.Any) -> str:
            return f"custom:{obj}"
    
    custom_serializer = CustomSerializer()
    assert custom_serializer.loads("custom:hello") == "hello"
    assert custom_serializer.loads("custom:") == ""
    
    # Test with complex data
    class PickleLikeSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode('utf-8').split('|')
        
        def dumps(self, obj: t.Any) -> bytes:
            return '|'.join(obj).encode('utf-8')
    
    pickle_serializer = PickleLikeSerializer()
    assert pickle_serializer.loads(b'a|b|c') == ['a', 'b', 'c']
    assert pickle_serializer.loads(b'') == ['']
    assert pickle_serializer.loads(b'single') == ['single']


# LLM-generated content at query #105
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer
    result = serializer.loads('42')
    assert result == 42
    
    # Test with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test with null
    result = serializer.loads('null')
    assert result is None
    
    # Test with boolean
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('false')
    assert result is False
    
    # Test with empty dict
    result = serializer.loads('{}')
    assert result == {}
    
    # Test with empty list
    result = serializer.loads('[]')
    assert result == []
    
    # Test with nested structures
    result = serializer.loads('{"a": {"b": [1, 2, 3]}}')
    assert result == {"a": {"b": [1, 2, 3]}}
```


# LLM-generated content at query #106
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default json serializer
    serializer = Serializer("secret-key")
    
    # Test successful loading with text serializer
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with custom serializer passed as parameter
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"custom": payload}
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    custom_serializer = Serializer("secret-key")
    result = custom_serializer.load_payload(b"test", serializer=CustomSerializer())
    assert result == {"custom": "test"}
    
    # Test BadPayload exception on invalid data
    with pytest.raises(BadPayload):
        serializer.load_payload(b"invalid json")
    
    # Test BadPayload exception on empty payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test with text serializer that returns string
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    text_serializer = Serializer("secret-key", serializer=TextSerializer())
    result = text_serializer.load_payload(b'{"text": "data"}')
    assert result == {"text": "data"}
    
    # Test BadPayload with original_error
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"not json")
    
    assert exc_info.value.original_error is not None
    assert isinstance(exc_info.value.original_error, json.JSONDecodeError)
```


# LLM-generated content at query #107
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    serializer = StringSerializer()
    result = serializer.dumps(42)
    assert result == "42"
    assert isinstance(result, str)
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode()
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps("hello")
    assert result == b"hello"
    assert isinstance(result, bytes)
    
    # Test with complex object
    result = serializer.dumps({"key": "value"})
    assert result == "{'key': 'value'}"
    
    # Test with None
    result = serializer.dumps(None)
    assert result == "None"


# LLM-generated content at query #108
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Has signature separator
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    # Test with custom serializer kwargs
    custom_serializer = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_custom = custom_serializer.dumps({"b": 1, "a": 2})
    # Should have sorted keys and no spaces
    assert '{"a":2,"b":1}' in result_custom
    
    # Test with salt parameter
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert result_with_salt != result  # Different salt produces different signature
    
    # Test with different secret keys
    serializer2 = Serializer("different-secret")
    result2 = serializer2.dumps({"key": "value"})
    assert result2 != result  # Different secret produces different signature
    
    # Test empty object
    result_empty = serializer.dumps({})
    assert isinstance(result_empty, str)
    
    # Test with None value
    result_none = serializer.dumps(None)
    assert isinstance(result_none, str)
    
    # Test with list
    result_list = serializer.dumps([1, 2, 3])
    assert isinstance(result_list, str)
    
    # Test with nested structure
    result_nested = serializer.dumps({"a": {"b": [1, 2, 3]}})
    assert isinstance(result_nested, str)
```


# LLM-generated content at query #109
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            if payload == '{"key": "value"}':
                return {"key": "value"}
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test basic JSON loading
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with different data types
    result = serializer.loads('{"number": 42, "list": [1, 2, 3]}')
    assert result == {"number": 42, "list": [1, 2, 3]}
    
    # Test with empty JSON
    result = serializer.loads('{}')
    assert result == {}
    
    # Test with string
    result = serializer.loads('"test string"')
    assert result == "test string"
    
    # Test with number
    result = serializer.loads('42')
    assert result == 42
    
    # Test with boolean
    result = serializer.loads('true')
    assert result is True
    
    # Test with null
    result = serializer.loads('null')
    assert result is None
```


# LLM-generated content at query #110
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    serializer = BytesSerializer()
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with a text serializer
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    text_serializer = TextSerializer()
    result = text_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Verify it works with the default json serializer
    result = json.loads(b'{"key": "value"}'.decode('utf-8'))
    assert result == {"key": "value"}


# LLM-generated content at query #111
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a concrete serializer that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with various data types
    test_cases = [
        {"key": "value"},
        [1, 2, 3],
        "string",
        42,
        None,
        {"nested": {"list": [1, 2]}},
    ]
    
    for obj in test_cases:
        result = serializer.dumps(obj)
        assert isinstance(result, str)
        assert serializer.loads(result) == obj
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    
    for obj in test_cases:
        result = bytes_serializer.dumps(obj)
        assert isinstance(result, bytes)
        assert bytes_serializer.loads(result) == obj


# LLM-generated content at query #112
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer dumps method works correctly."""
    # Create a concrete implementation of _PDataSerializer for testing
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with various data types
    assert serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps("string") == '"string"'
    assert serializer.dumps(42) == "42"
    assert serializer.dumps(None) == "null"
    assert serializer.dumps(True) == "true"


# LLM-generated content at query #113
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    serializer = BytesSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with a custom serializer that returns str
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    text_serializer = TextSerializer()
    result = text_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with various Python objects
    test_cases = [
        ({"a": 1, "b": 2}, '{"a": 1, "b": 2}'),
        ([1, 2, 3], "[1, 2, 3]"),
        ("hello", '"hello"'),
        (42, "42"),
        (3.14, "3.14"),
        (True, "true"),
        (None, "null"),
    ]
    
    for obj, expected in test_cases:
        result = text_serializer.dumps(obj)
        assert result == expected, f"Failed for {obj}: expected {expected}, got {result}"
```


# LLM-generated content at query #114
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test the loads method of _PDataSerializer protocol."""
    # Create a concrete implementation that follows the _PDataSerializer protocol
    class TestSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer
    result = serializer.loads('42')
    assert result == 42
    
    # Test with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test with null
    result = serializer.loads('null')
    assert result is None
    
    # Test with boolean
    result = serializer.loads('true')
    assert result is True
    result = serializer.loads('false')
    assert result is False
    
    # Test with empty JSON
    result = serializer.loads('{}')
    assert result == {}
    
    # Test with nested structure
    result = serializer.loads('{"a": {"b": [1, 2, 3]}}')
    assert result == {"a": {"b": [1, 2, 3]}}
    
    # Test with bytes payload (should work if serializer handles bytes)
    class BytesSerializer:
        def loads(self, payload):
            if isinstance(payload, bytes):
                payload = payload.decode('utf-8')
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}```


# LLM-generated content at query #115
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # contains separator between payload and signature
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            import json
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            import json
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert b"." in result
    
    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert result_with_salt != result
    
    # Test roundtrip: dumps then loads should return original data
    data = {"test": 123, "nested": {"list": [1, 2, 3]}}
    signed = serializer.dumps(data)
    assert serializer.loads(signed) == data
    
    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True}
    )
    result_with_kwargs = serializer_with_kwargs.dumps({"b": 1, "a": 2})
    assert result_with_kwargs != serializer.dumps({"b": 1, "a": 2})
    
    # Test with different key types
    for key in ["string-key", b"bytes-key", ["key1", "key2"]]:
        s = Serializer(key)
        result = s.dumps("test")
        assert isinstance(result, (str, bytes))
        assert s.loads(result) == "test"
    
    # Test serializing various data types
    test_data = [
        None,
        True,
        False,
        42,
        3.14,
        "string",
        [1, 2, 3],
        {"a": 1},
        (1, 2),  # tuple becomes list in JSON
    ]
    for data in test_data:
        signed = serializer.dumps(data)
        assert serializer.loads(signed) == data


# LLM-generated content at query #116
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
    assert serializer.loads('{"key": "value"}') == {"key": "value"}
    assert serializer.loads('123') == 123
    assert serializer.loads('"text"') == "text"
    assert serializer.loads('null') is None
    assert serializer.loads('true') is True
    assert serializer.loads('false') is False
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'42') == 42
    assert bytes_serializer.loads(b'"hello"') == "hello"

    # Test that loads raises appropriate exceptions for invalid data
    import pytest
    with pytest.raises(json.JSONDecodeError):
        serializer.loads("invalid json")
    
    with pytest.raises(json.JSONDecodeError):
        bytes_serializer.loads(b"invalid json")
```


# LLM-generated content at query #117
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer loads method can be called with various payload types."""
    # Create a concrete implementation that matches the protocol
    class MockSerializer:
        def loads(self, payload):
            if isinstance(payload, str):
                return json.loads(payload)
            elif isinstance(payload, bytes):
                return json.loads(payload.decode('utf-8'))
            raise ValueError("Invalid payload type")
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with string payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes payload
    result = serializer.loads(b'{"number": 42}')
    assert result == {"number": 42}
    
    # Test with empty dict
    result = serializer.loads('{}')
    assert result == {}
    
    # Test with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with nested data
    result = serializer.loads('{"nested": {"a": 1, "b": [1, 2]}}')
    assert result == {"nested": {"a": 1, "b": [1, 2]}}
    
    # Test with None value
    result = serializer.loads('{"value": null}')
    assert result == {"value": None}
    
    # Test with boolean values
    result = serializer.loads('{"flag": true, "active": false}')
    assert result == {"flag": True, "active": False}
```


# LLM-generated content at query #118
#--------------------------

```python
def test_Serializer_dumps():
    """Test Serializer.dumps method."""
    # Test with default JSON serializer (text serializer)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Contains signature separator
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode()
        def loads(self, payload):
            return json.loads(payload.decode())
    
    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = s_bytes.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    # Test with custom salt
    s2 = Serializer("secret-key", salt="custom-salt")
    result2 = s2.dumps({"key": "value"}, salt="other-salt")
    assert result != result2  # Different salt produces different signature
    
    # Test with serializer_kwargs
    s3 = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    result3 = s3.dumps({"b": 2, "a": 1})
    # Verify sorted keys were used in serialization
    assert result3.count(b"a") == 1 if isinstance(result3, bytes) else result3.count("a") == 1
    
    # Test that dumps returns consistent results for same input
    result4 = s.dumps({"key": "value"})
    assert result != result4  # Different salt (default vs None)
    
    # Test with key rotation
    s5 = Serializer(["old-key", "new-key"])
    result5 = s5.dumps({"key": "value"})
    assert isinstance(result5, str)


# LLM-generated content at query #119
#--------------------------

```python
def test_Serializer_dumps():
    """Test that dumps returns a signed, serialized string."""
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    result = serializer.dumps(data)
    
    # Should return a string (not bytes) since JSON is a text serializer
    assert isinstance(result, str)
    # Should contain the serialized payload and signature
    assert "." in result
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps(data)
    
    # Should return bytes for a bytes serializer
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    # Verify that the result can be unsigned back to the original data
    unsigned = serializer.loads(result)
    assert unsigned == data
    
    # Test with custom salt
    result_with_salt = serializer.dumps(data, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert "." in result_with_salt
    # Different salt should produce different signature
    assert result_with_salt != result
    
    # Test with empty data
    result_empty = serializer.dumps({})
    assert isinstance(result_empty, str)
    assert "." in result_empty


# LLM-generated content at query #120
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol loads method works with different serializer implementations."""
    
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            if payload == "null":
                return None
            if payload == "true":
                return True
            if payload == "false":
                return False
            if payload == "42":
                return 42
            if payload == "3.14":
                return 3.14
            if payload == '"hello"':
                return "hello"
            if payload == '{"key": "value"}':
                return {"key": "value"}
            if payload == "[1, 2, 3]":
                return [1, 2, 3]
            raise ValueError(f"Unexpected payload: {payload}")
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)

    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            if payload == b"null":
                return None
            if payload == b"true":
                return True
            if payload == b"false":
                return False
            if payload == b"42":
                return 42
            if payload == b"3.14":
                return 3.14
            if payload == b'"hello"':
                return "hello"
            if payload == b'{"key": "value"}':
                return {"key": "value"}
            if payload == b"[1, 2, 3]":
                return [1, 2, 3]
            raise ValueError(f"Unexpected payload: {payload}")
        
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode()

    # Test with text serializer
    text_serializer = TextSerializer()
    assert text_serializer.loads("null") is None
    assert text_serializer.loads("true") is True
    assert text_serializer.loads("false") is False
    assert text_serializer.loads("42") == 42
    assert text_serializer.loads("3.14") == 3.14
    assert text_serializer.loads('"hello"') == "hello"
    assert text_serializer.loads('{"key": "value"}') == {"key": "value"}
    assert text_serializer.loads("[1, 2, 3]") == [1, 2, 3]
    
    # Test with bytes serializer
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b"null") is None
    assert bytes_serializer.loads(b"true") is True
    assert bytes_serializer.loads(b"false") is False
    assert bytes_serializer.loads(b"42") == 42
    assert bytes_serializer.loads(b"3.14") == 3.14
    assert bytes_serializer.loads(b'"hello"') == "hello"
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b"[1, 2, 3]") == [1, 2, 3]
    
    # Test that loads is a method (not a static function)
    assert callable(text_serializer.loads)
    assert callable(bytes_serializer.loads)
    
    # Test that the protocol requires loads to accept exactly one argument
    import inspect
    text_sig = inspect.signature(text_serializer.loads)
    bytes_sig = inspect.signature(bytes_serializer.loads)
    assert len(text_sig.parameters) == 1  # self is already bound
    assert len(bytes_sig.parameters) == 1  # self is already bound
    
    # Test with edge cases
    class EdgeCaseSerializer:
        def loads(self, payload: str) -> t.Any:
            if payload == "":
                return ""
            if payload == " ":
                return " "
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    edge_serializer = EdgeCaseSerializer()
    assert edge_serializer.loads("") == ""
    assert edge_serializer.loads(" ") == " "
    assert edge_serializer.loads("special_chars_!@#$%") == "special_chars_!@#$%"
    
    # Test that loads can handle complex nested structures
    class ComplexSerializer:
        def loads(self, payload: str) -> t.Any:
            import json
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    complex_serializer = ComplexSerializer()
    assert complex_serializer.loads('{"a": [1, 2, {"b": 3}]}') == {"a": [1, 2, {"b": 3}]}
    assert complex_serializer.loads('[[1, 2], [3, 4]]') == [[1, 2], [3, 4]]
    
    # Verify that the protocol type is satisfied
    from typing import Protocol
    assert isinstance(text_serializer, Protocol)
    assert isinstance(bytes_serializer, Protocol)
    assert isinstance(edge_serializer, Protocol)
    assert isinstance(complex_serializer, Protocol)
```


# LLM-generated content at query #121
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a simple serializer that implements the _PDataSerializer protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test loading valid JSON
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading a string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test loading a number
    result = serializer.loads('42')
    assert result == 42
    
    # Test loading null
    result = serializer.loads('null')
    assert result is None
    
    # Test loading boolean
    result = serializer.loads('true')
    assert result is True
    
    # Test that it raises an exception for invalid input
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
```


# LLM-generated content at query #122
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns parsed data
    class TestSerializer:
        def loads(self, payload: str) -> dict:
            return {"data": payload}
        
        def dumps(self, obj: dict) -> str:
            return str(obj)
    
    serializer = TestSerializer()
    result = serializer.loads('{"key": "value"}')
    assert result == {"data": '{"key": "value"}'}
    
    # Test with integer payload
    class IntSerializer:
        def loads(self, payload: str) -> int:
            return int(payload)
        
        def dumps(self, obj: int) -> str:
            return str(obj)
    
    int_serializer = IntSerializer()
    result = int_serializer.loads("42")
    assert result == 42
    
    # Test with list payload
    class ListSerializer:
        def loads(self, payload: str) -> list:
            return [payload]
        
        def dumps(self, obj: list) -> str:
            return str(obj)
    
    list_serializer = ListSerializer()
    result = list_serializer.loads("test")
    assert result == ["test"]
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> dict:
            return {"data": payload.decode()}
        
        def dumps(self, obj: dict) -> bytes:
            return bytes(str(obj), "utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"data": '{"key": "value"}'}
    
    # Test with None payload
    class NoneSerializer:
        def loads(self, payload: str) -> None:
            return None
        
        def dumps(self, obj: None) -> str:
            return "None"
    
    none_serializer = NoneSerializer()
    result = none_serializer.loads("anything")
    assert result is None
    
    # Test with empty string payload
    class EmptySerializer:
        def loads(self, payload: str) -> str:
            return payload
        
        def dumps(self, obj: str) -> str:
            return obj
    
    empty_serializer = EmptySerializer()
    result = empty_serializer.loads("")
    assert result == ""
```


# LLM-generated content at query #123
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Create a mock serializer that implements the protocol
    class MockSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with a simple object
    test_obj = {"key": "value"}
    result = serializer.dumps(test_obj)
    assert result == '{"key": "value"}'
    
    # Test with a list
    test_list = [1, 2, 3]
    result = serializer.dumps(test_list)
    assert result == "[1, 2, 3]"
    
    # Test with a string
    test_string = "hello"
    result = serializer.dumps(test_string)
    assert result == '"hello"'
    
    # Test with None
    result = serializer.dumps(None)
    assert result == "null"
    
    # Test with an integer
    result = serializer.dumps(42)
    assert result == "42"
    
    # Test with a boolean
    result = serializer.dumps(True)
    assert result == "true"
    
    # Test with a nested object
    test_nested = {"a": [1, 2, {"b": "c"}]}
    result = serializer.dumps(test_nested)
    assert result == '{"a": [1, 2, {"b": "c"}]}'
```


# LLM-generated content at query #124
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON serializer
    serializer = json
    
    # Test loading a valid JSON string
    data = '{"key": "value"}'
    result = serializer.loads(data)
    assert result == {"key": "value"}
    
    # Test loading a JSON array
    data = '[1, 2, 3]'
    result = serializer.loads(data)
    assert result == [1, 2, 3]
    
    # Test loading a simple value
    data = '"hello"'
    result = serializer.loads(data)
    assert result == "hello"
    
    # Test loading a number
    data = '42'
    result = serializer.loads(data)
    assert result == 42
    
    # Test loading with a custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes):
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj):
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    data = b'{"key": "value"}'
    result = bytes_serializer.loads(data)
    assert result == {"key": "value"}
    
    # Test loading with invalid JSON should raise exception
    data = '{invalid json}'
    try:
        serializer.loads(data)
        assert False, "Should have raised exception"
    except Exception:
        pass
    
    # Test loading None
    data = 'null'
    result = serializer.loads(data)
    assert result is None
    
    # Test loading boolean
    data = 'true'
    result = serializer.loads(data)
    assert result is True
    
    data = 'false'
    result = serializer.loads(data)
    assert result is False
```


# LLM-generated content at query #125
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer
    serializer = Serializer("secret-key")
    
    # Test loading valid JSON payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading payload with text serializer (JSON returns str)
    payload_str = '{"key": "value"}'
    result = serializer.load_payload(payload_str.encode("utf-8"))
    assert result == {"key": "value"}
    
    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload):
            return {"data": payload}
        def dumps(self, obj):
            return b"test"
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    
    # Test loading bytes payload with bytes serializer
    payload = b"test_data"
    result = bytes_serializer.load_payload(payload)
    assert result == {"data": b"test_data"}
    
    # Test that BadPayload is raised for invalid JSON
    import json
    invalid_payload = b"invalid json"
    try:
        serializer.load_payload(invalid_payload)
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test that BadPayload preserves original error
    try:
        serializer.load_payload(invalid_payload)
    except BadPayload as e:
        assert isinstance(e.original_error, json.JSONDecodeError)
    
    # Test with override serializer parameter
    class OverrideSerializer:
        def loads(self, payload):
            return {"override": payload.decode()}
        def dumps(self, obj):
            return b"test"
    
    result = serializer.load_payload(b"test", serializer=OverrideSerializer())
    assert result == {"override": "test"}
    
    # Test with custom text serializer
    class CustomTextSerializer:
        def loads(self, payload):
            return {"text": payload}
        def dumps(self, obj):
            return "test"
    
    text_serializer = Serializer("secret-key", serializer=CustomTextSerializer())
    payload = b"hello"
    result = text_serializer.load_payload(payload)
    assert result == {"text": "hello"}
```


# LLM-generated content at query #126
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol requires a dumps method."""
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer: _PDataSerializer[str] = MockSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == '{"key": "value"}'
    assert isinstance(result, str)


# LLM-generated content at query #127
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a mock serializer that implements the protocol
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with various data types
    test_cases = [
        {"key": "value"},
        [1, 2, 3],
        "string",
        42,
        True,
        None
    ]
    
    for obj in test_cases:
        result = serializer.dumps(obj)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert serializer.loads(result) == obj, f"Round-trip failed for {obj}"
    
    # Test with bytes serializer
    class MockBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = MockBytesSerializer()
    result = bytes_serializer.dumps({"test": "data"})
    assert isinstance(result, bytes), f"Expected bytes, got {type(result)}"
    assert bytes_serializer.loads(result) == {"test": "data"} 


# LLM-generated content at query #128
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str | bytes) -> t.Any:
            if isinstance(payload, bytes):
                payload = payload.decode('utf-8')
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str | bytes:
            return json.dumps(obj)

    serializer = TestSerializer()
    
    # Test with string payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes payload
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with different data types
    result = serializer.loads('123')
    assert result == 123
    
    result = serializer.loads('"string"')
    assert result == "string"
    
    result = serializer.loads('null')
    assert result is None
    
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3] 


# LLM-generated content at query #129
#--------------------------

```python
def test_Serializer_dumps():
    """Test that dumps produces a signed serialized string that can be verified."""
    # Test with default JSON serializer (text serializer)
    secret_key = b"secret-key"
    serializer = Serializer(secret_key)
    
    # Test with simple data
    data = {"key": "value"}
    result = serializer.dumps(data)
    
    # Result should be a string (since JSON is text serializer)
    assert isinstance(result, str)
    
    # Result should contain the serialized payload and signature
    assert "." in result
    
    # Verify we can unsign and get back original data
    loaded = serializer.loads(result)
    assert loaded == data
    
    # Test with bytes data
    data2 = ["list", 123, True]
    result2 = serializer.dumps(data2)
    loaded2 = serializer.loads(result2)
    assert loaded2 == data2
    
    # Test with empty data
    result3 = serializer.dumps(None)
    loaded3 = serializer.loads(result3)
    assert loaded3 is None
    
    # Test with custom salt
    custom_salt = b"custom-salt"
    result4 = serializer.dumps(data, salt=custom_salt)
    loaded4 = serializer.loads(result4, salt=custom_salt)
    assert loaded4 == data
    
    # Test that different salts produce different signatures
    result5 = serializer.dumps(data)
    assert result4 != result5  # Different salt should produce different output
    
    # Test with bytes serializer
    bytes_serializer = Serializer(secret_key, serializer=lambda: None)
    # Override serializer with custom bytes serializer for testing
    class BytesSerializer:
        def dumps(self, obj):
            import pickle
            return pickle.dumps(obj)
        def loads(self, payload):
            import pickle
            return pickle.loads(payload)
    
    bytes_ser = Serializer(secret_key, serializer=BytesSerializer())
    data_bytes = {"test": "bytes"}
    result_bytes = bytes_ser.dumps(data_bytes)
    assert isinstance(result_bytes, bytes)
    loaded_bytes = bytes_ser.loads(result_bytes)
    assert loaded_bytes == data_bytes
    
    # Test with key rotation (multiple secret keys)
    keys = [b"old-key", b"newer-key", b"newest-key"]
    serializer_rotation = Serializer(keys)
    data_rotation = {"rotated": True}
    result_rotation = serializer_rotation.dumps(data_rotation)
    loaded_rotation = serializer_rotation.loads(result_rotation)
    assert loaded_rotation == data_rotation
    
    # Test that dumps uses the newest key for signing
    assert serializer_rotation.secret_key == keys[-1]


# LLM-generated content at query #130
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Create a mock serializer that conforms to _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with simple data
    result = serializer.dumps({"key": "value"})
    assert result == '{"key": "value"}'
    
    # Test with list
    result = serializer.dumps([1, 2, 3])
    assert result == '[1, 2, 3]'
    
    # Test with string
    result = serializer.dumps("test")
    assert result == '"test"'
    
    # Test with None
    result = serializer.dumps(None)
    assert result == 'null'
    
    # Test with integer
    result = serializer.dumps(42)
    assert result == '42'
    
    # Test with boolean
    result = serializer.dumps(True)
    assert result == 'true'
    
    # Test with nested structure
    nested = {"a": [1, 2, {"b": 3}], "c": "d"}
    result = serializer.dumps(nested)
    assert result == json.dumps(nested)  # Compare with standard json.dumps
    
    # Test that dumps returns a string (not bytes)
    assert isinstance(result, str)
    
    # Test with a bytes serializer
    class BytesMockSerializer:
        def loads(self, payload):
            return json.loads(payload.decode())
        
        def dumps(self, obj):
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesMockSerializer()
    result = bytes_serializer.dumps("test")
    assert result == b'"test"'
    assert isinstance(result, bytes)
```


# LLM-generated content at query #131
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
    
    # Test with simple string
    result = serializer.loads('"test"')
    assert result == "test"
    
    # Test with number
    result = serializer.loads('42')
    assert result == 42
    
    # Test with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with boolean
    result = serializer.loads('true')
    assert result is True
    
    # Test with null
    result = serializer.loads('null')
    assert result is None
    
    # Test with bytes serializer
    class MockBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = MockBytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that invalid JSON raises appropriate error
    import pytest
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json') 


# LLM-generated content at query #132
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with a serializer that returns string
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    string_serializer = StringSerializer()
    result = string_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with empty dict
    result = string_serializer.dumps({})
    assert isinstance(result, str)
    assert result == '{}'
    
    # Test with list
    result = string_serializer.dumps([1, 2, 3])
    assert isinstance(result, str)
    assert result == '[1, 2, 3]'
    
    # Test with None
    result = string_serializer.dumps(None)
    assert isinstance(result, str)
    assert result == 'null'


# LLM-generated content at query #133
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Contains separator between payload and signature
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert result_with_salt != result
    
    # Test with empty data
    result_empty = serializer.dumps({})
    assert isinstance(result_empty, str)
    assert "." in result_empty
    
    # Test that different keys produce different signatures
    serializer2 = Serializer("different-secret-key")
    result2 = serializer2.dumps({"key": "value"})
    assert result != result2
    
    # Test that the result can be loaded back
    loaded = serializer.loads(result)
    assert loaded == {"key": "value"}
```


# LLM-generated content at query #134
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns a dict
    class TestSerializer:
        def loads(self, payload: str) -> dict:
            return {"data": payload}
        
        def dumps(self, obj: dict) -> str:
            return str(obj)
    
    serializer = TestSerializer()
    result = serializer.loads("test_payload")
    assert result == {"data": "test_payload"}
    
    # Test with a JSON serializer
    result = json.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes input
    class BinarySerializer:
        def loads(self, payload: bytes) -> dict:
            return {"data": payload.decode()}
        
        def dumps(self, obj: dict) -> bytes:
            return str(obj).encode()
    
    binary_serializer = BinarySerializer()
    result = binary_serializer.loads(b"binary_payload")
    assert result == {"data": "binary_payload"}
    
    # Test with integer input
    class IntSerializer:
        def loads(self, payload: int) -> int:
            return payload * 2
        
        def dumps(self, obj: int) -> int:
            return obj // 2
    
    int_serializer = IntSerializer()
    result = int_serializer.loads(5)
    assert result == 10
    
    # Test with list input
    class ListSerializer:
        def loads(self, payload: list) -> list:
            return [x * 2 for x in payload]
        
        def dumps(self, obj: list) -> list:
            return [x // 2 for x in obj]
    
    list_serializer = ListSerializer()
    result = list_serializer.loads([1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with empty string
    class EmptySerializer:
        def loads(self, payload: str) -> str:
            return payload
        
        def dumps(self, obj: str) -> str:
            return obj
    
    empty_serializer = EmptySerializer()
    result = empty_serializer.loads("")
    assert result == ""```


# LLM-generated content at query #135
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a concrete implementation that matches the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with various data types
    test_cases = [
        {"key": "value"},
        [1, 2, 3],
        "string",
        42,
        None,
        {"nested": {"data": True}},
    ]
    
    for obj in test_cases:
        result = serializer.dumps(obj)
        assert isinstance(result, str), f"dumps should return str, got {type(result)}"
        # Verify we can load it back
        loaded = serializer.loads(result)
        assert loaded == obj, f"Round-trip failed for {obj}: got {loaded}"


# LLM-generated content at query #136
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol is structural and can be used with any object
    that has dumps and loads methods."""
    
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = CustomSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    
    assert isinstance(result, str)
    assert json.loads(result) == data
    assert result == '{"key": "value"}'
    
    # Test with simple data types
    assert serializer.dumps(123) == "123"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps(None) == "null"
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    bytes_result = bytes_serializer.dumps(data)
    
    assert isinstance(bytes_result, bytes)
    assert json.loads(bytes_result.decode('utf-8')) == data
```


# LLM-generated content at query #137
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
    
    # Test with numeric payload
    result = serializer.loads('42')
    assert result == 42
    
    # Test with list payload
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with null payload
    result = serializer.loads('null')
    assert result is None
    
    # Test with boolean payload
    result = serializer.loads('true')
    assert result is True
    result = serializer.loads('false')
    assert result is False
    
    # Test that it raises appropriate exception for invalid JSON
    import pytest
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
    
    # Test with empty payload
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('')
    
    # Test with a different serializer implementation
    class CustomSerializer:
        def loads(self, payload):
            return payload.upper()
        
        def dumps(self, obj):
            return str(obj)
    
    custom_serializer = CustomSerializer()
    result = custom_serializer.loads("hello")
    assert result == "HELLO"
    
    # Test that loads only takes one positional argument
    with pytest.raises(TypeError):
        custom_serializer.loads("test", "extra_arg")
```


# LLM-generated content at query #138
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol requires dumps method."""
    # Create a mock serializer that implements the protocol
    class MockSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return str(obj)
    
    serializer = MockSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == "{'key': 'value'}"
    
    # Test with a different object type
    result = serializer.dumps([1, 2, 3])
    assert result == "[1, 2, 3]"
```


# LLM-generated content at query #139
#--------------------------

```python
def test_Serializer_load_payload():
    """Test load_payload method with various scenarios."""
    import pytest
    
    # Test with default JSON serializer and bytes payload
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with custom serializer that returns text
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()
        def dumps(self, obj: t.Any) -> str:
            return str(obj).lower()
    
    text_serializer = TextSerializer()
    serializer_text = Serializer("secret-key", serializer=text_serializer)
    payload_text = b"hello"
    result_text = serializer_text.load_payload(payload_text)
    assert result_text == "HELLO"
    
    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload + b"_decoded"
        def dumps(self, obj: t.Any) -> bytes:
            return bytes(obj) + b"_encoded"
    
    bytes_serializer = BytesSerializer()
    serializer_bytes = Serializer("secret-key", serializer=bytes_serializer)
    payload_bytes = b"data"
    result_bytes = serializer_bytes.load_payload(payload_bytes)
    assert result_bytes == b"data_decoded"
    
    # Test with explicit serializer parameter (override)
    class OverrideSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"from": "override"}
        def dumps(self, obj: t.Any) -> str:
            return "override"
    
    override_serializer = OverrideSerializer()
    result_override = serializer.load_payload(b"{}", serializer=override_serializer)
    assert result_override == {"from": "override"}
    
    # Test with invalid payload (raises BadPayload)
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"invalid json")
    assert "Could not load the payload" in str(exc_info.value)
    assert exc_info.value.original_error is not None
    
    # Test with empty payload (raises BadPayload)
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test with payload that causes UnicodeDecodeError in text serializer
    class ErrorSerializer:
        def loads(self, payload: str) -> t.Any:
            raise ValueError("Custom error")
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    error_serializer = ErrorSerializer()
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"test", serializer=error_serializer)
    assert exc_info.value.original_error is not None
    assert isinstance(exc_info.value.original_error, ValueError)
```


# LLM-generated content at query #140
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StringSerializer()
    assert serializer.loads('{"key": "value"}') == {"key": "value"}
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    assert serializer.loads('"hello"') == "hello"
    assert serializer.loads('42') == 42
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'[1, 2, 3]') == [1, 2, 3]
    
    # Test with invalid payload
    class InvalidSerializer:
        def loads(self, payload: str) -> t.Any:
            raise ValueError("Invalid payload")
        
        def dumps(self, obj: t.Any) -> str:
            return ""
    
    invalid_serializer = InvalidSerializer()
    with pytest.raises(ValueError, match="Invalid payload"):
        invalid_serializer.loads("bad data")
```


# LLM-generated content at query #141
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON serializer
    serializer = json
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with primitive types
    assert serializer.loads('"hello"') == "hello"
    assert serializer.loads('42') == 42
    assert serializer.loads('true') == True
    assert serializer.loads('null') is None
    
    # Test with empty JSON
    assert serializer.loads('{}') == {}
    assert serializer.loads('[]') == []
    
    # Test with nested structures
    nested_json = '{"outer": {"inner": [1, 2, 3]}}'
    result = serializer.loads(nested_json)
    assert result == {"outer": {"inner": [1, 2, 3]}} 


# LLM-generated content at query #142
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a simple serializer that implements the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test basic serialization
    result = serializer.dumps({"key": "value"})
    assert result == '{"key": "value"}'
    assert isinstance(result, str)
    
    # Test with list
    result = serializer.dumps([1, 2, 3])
    assert result == "[1, 2, 3]"
    
    # Test with None
    result = serializer.dumps(None)
    assert result == "null"
    
    # Test with int
    result = serializer.dumps(42)
    assert result == "42"
    
    # Test with boolean
    result = serializer.dumps(True)
    assert result == "true"
```


# LLM-generated content at query #143
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a simple text serializer
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TextSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    serializer = BytesSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with None value
    result = TextSerializer().dumps(None)
    assert result == "null"
    
    # Test with list value
    result = TextSerializer().dumps([1, 2, 3])
    assert result == "[1, 2, 3]"
    
    # Test with integer value
    result = TextSerializer().dumps(42)
    assert result == "42"
    
    # Test with boolean value
    result = TextSerializer().dumps(True)
    assert result == "true"
```


# LLM-generated content at query #144
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol defines loads method properly."""
    # Create a concrete implementation that satisfies the protocol
    class TestSerializer:
        def loads(self, payload: str | bytes, /) -> t.Any:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            return json.loads(payload)

        def dumps(self, obj: t.Any, /) -> str | bytes:
            return json.dumps(obj)

    serializer = TestSerializer()
    
    # Test with str payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes payload
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with different data types
    result = serializer.loads('42')
    assert result == 42
    
    result = serializer.loads('"string"')
    assert result == "string"
    
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test that it works as a Protocol (structural subtyping)
    def process_serializer(s: _PDataSerializer[t.Any]) -> t.Any:
        return s.loads('{"test": "data"}')
    
    assert process_serializer(serializer) == {"test": "data"}
```


# LLM-generated content at query #145
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a serializer that returns str
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    text_serializer = TextSerializer()
    assert text_serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert isinstance(text_serializer.dumps({}), str)
    assert text_serializer.dumps(42) == "42"
    assert text_serializer.dumps(None) == "null"

    # Test with a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.dumps({"key": "value"}) == b'{"key": "value"}'
    assert isinstance(bytes_serializer.dumps({}), bytes)
    assert bytes_serializer.dumps(42) == b"42"
    assert bytes_serializer.dumps(None) == b"null"

    # Test that dumps is callable and returns a value
    class MinimalSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    minimal_serializer = MinimalSerializer()
    result = minimal_serializer.dumps("test_data")
    assert result == "test_data"
    assert isinstance(result, str)
```


# LLM-generated content at query #146
#--------------------------

```python
def test__PDataSerializer_dumps():
    class TestSerializer:
        def dumps(self, obj):
            return str(obj)
        
        def loads(self, payload):
            return eval(payload)
    
    serializer = TestSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == "{'key': 'value'}"
```


# LLM-generated content at query #147
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a simple serializer that matches the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    test_data = {"key": "value", "number": 42}
    
    # Test dumps returns expected serialized string
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    assert result == json.dumps(test_data)
    
    # Test with different data types
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps("hello") == '"hello"'
    assert serializer.dumps(None) == "null"
    
    # Test roundtrip: dumps then loads returns original data
    assert serializer.loads(serializer.dumps(test_data)) == test_data


# LLM-generated content at query #148
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test that _PDataSerializer protocol is structural and can be used with any object
    # that has a dumps method
    
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj)
        
        def loads(self, payload):
            return eval(payload)
    
    serializer = CustomSerializer()
    
    # Verify dumps returns string for various inputs
    assert serializer.dumps(42) == "42"
    assert serializer.dumps("hello") == "hello"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps({"a": 1}) == "{'a': 1}"
    
    # Test that the result type matches _PDataSerializer[str]
    result = serializer.dumps({})
    assert isinstance(result, str)
```


# LLM-generated content at query #149
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a serializer that returns str
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    text_serializer = TextSerializer()
    result = text_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with a serializer that returns bytes
    class BinarySerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    binary_serializer = BinarySerializer()
    result = binary_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with empty dict
    result = text_serializer.dumps({})
    assert result == "{}"
    
    # Test with list
    result = text_serializer.dumps([1, 2, 3])
    assert result == "[1, 2, 3]"
    
    # Test with None
    result = text_serializer.dumps(None)
    assert result == "null"
    
    # Test with string
    result = text_serializer.dumps("hello")
    assert result == '"hello"'```


# LLM-generated content at query #150
#--------------------------

```python
def test__PDataSerializer_dumps():
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = CustomSerializer()
    test_data = {"key": "value", "number": 42}
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    assert result == '{"key": "value", "number": 42}'
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    bytes_result = bytes_serializer.dumps(test_data)
    assert isinstance(bytes_result, bytes)
    assert bytes_result == b'{"key": "value", "number": 42}'


# LLM-generated content at query #151
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # contains signature separator
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    
    # Test with salt parameter
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert result_with_salt != result  # different salt produces different signature
    
    # Test that dumps produces consistent results with same input
    result1 = serializer.dumps("test data")
    result2 = serializer.dumps("test data")
    assert result1 == result2
    
    # Test with different data types
    assert isinstance(serializer.dumps(123), str)
    assert isinstance(serializer.dumps([1, 2, 3]), str)
    assert isinstance(serializer.dumps(None), str)
    
    # Test that the output can be verified with loads
    data = {"nested": {"list": [1, 2, 3]}}
    signed = serializer.dumps(data)
    loaded = serializer.loads(signed)
    assert loaded == data
    
    # Test with serializer_kwargs
    class CustomSerializer:
        def __init__(self):
            self.dumps_called_with = None
        
        def dumps(self, obj, **kwargs):
            self.dumps_called_with = kwargs
            return json.dumps(obj)
        
        def loads(self, payload):
            return json.loads(payload)
    
    custom_ser = CustomSerializer()
    custom_serializer = Serializer("key", serializer=custom_ser, serializer_kwargs={"indent": 2})
    custom_serializer.dumps({"test": "data"})
    assert custom_ser.dumps_called_with == {"indent": 2}


# LLM-generated content at query #152
#--------------------------

```python
def test_Serializer_dumps():
    """Test Serializer.dumps method."""
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    # Verify it can be loaded back
    loaded = serializer.loads(result)
    assert loaded == {"key": "value"}

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    loaded_bytes = bytes_serializer.loads(result_bytes)
    assert loaded_bytes == {"key": "value"}

    # Test with custom serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_kwargs = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    # Should have sorted keys and no spaces
    assert "a" in result_kwargs and "b" in result_kwargs
    loaded_kwargs = serializer_with_kwargs.loads(result_kwargs)
    assert loaded_kwargs == {"b": 2, "a": 1}

    # Test with custom salt
    serializer_salt = Serializer("secret-key", salt="custom-salt")
    result_salt = serializer_salt.dumps({"key": "value"})
    loaded_salt = serializer_salt.loads(result_salt)
    assert loaded_salt == {"key": "value"}

    # Test dumps with different salt parameter
    result_different_salt = serializer.dumps({"key": "value"}, salt="different-salt")
    # Should fail to load with default salt
    import pytest
    from itsdangerous.exc import BadSignature
    with pytest.raises(BadSignature):
        serializer.loads(result_different_salt)
    # But should load with the correct salt
    loaded_different = serializer.loads(result_different_salt, salt="different-salt")
    assert loaded_different == {"key": "value"}


# LLM-generated content at query #153
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a serializer that returns str
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    text_serializer = TextSerializer()
    assert text_serializer.loads('{"key": "value"}') == {"key": "value"}
    assert text_serializer.loads('[1, 2, 3]') == [1, 2, 3]
    assert text_serializer.loads('"string"') == "string"
    
    # Test with a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'[1, 2, 3]') == [1, 2, 3]
    assert bytes_serializer.loads(b'"string"') == "string"
    
    # Test with a custom serializer that handles complex types
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.split(",")
        
        def dumps(self, obj: t.Any) -> str:
            return ",".join(obj)
    
    custom_serializer = CustomSerializer()
    assert custom_serializer.loads("a,b,c") == ["a", "b", "c"]
    assert custom_serializer.loads("single") == ["single"]
    
    # Test with empty payload
    assert text_serializer.loads("null") is None
    assert bytes_serializer.loads(b"null") is None
    
    # Test Protocol conformance
    def accepts_serializer(serializer: _PDataSerializer[t.Any]) -> None:
        result = serializer.loads('"test"')
        assert result == "test"
    
    accepts_serializer(text_serializer)
    accepts_serializer(bytes_serializer)
```


# LLM-generated content at query #154
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test basic JSON payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test integer payload
    result = serializer.loads('42')
    assert result == 42
    
    # Test list payload
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test string payload
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test boolean payload
    result = serializer.loads('true')
    assert result is True
    
    # Test null payload
    result = serializer.loads('null')
    assert result is None
    
    # Test empty object
    result = serializer.loads('{}')
    assert result == {}
    
    # Test nested structure
    result = serializer.loads('{"a": {"b": [1, 2, 3]}}')
    assert result == {"a": {"b": [1, 2, 3]}}
    
    # Test with a custom serializer that returns different types
    class CustomSerializer:
        def loads(self, payload):
            if payload == b"special":
                return "special_value"
            return payload
        
        def dumps(self, obj):
            return str(obj)
    
    custom_serializer = CustomSerializer()
    
    # Test with bytes payload
    result = custom_serializer.loads(b"test_bytes")
    assert result == b"test_bytes"
    
    # Test with special payload
    result = custom_serializer.loads(b"special")
    assert result == "special_value"
    
    # Test with string payload
    result = custom_serializer.loads("test_string")
    assert result == "test_string"


# LLM-generated content at query #155
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a simple serializer that implements the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with different input types
    test_cases = [
        {"key": "value"},
        [1, 2, 3],
        "string",
        42,
        None,
        {"nested": {"inner": [1, 2, 3]}},
        {"unicode": "日本語"},
    ]
    
    for obj in test_cases:
        result = serializer.dumps(obj)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert json.loads(result) == obj, f"Round-trip failed for {obj}"
    
    # Test that the result is indeed a string
    assert isinstance(serializer.dumps({}), str)
    
    # Test with a bytes serializer to verify the protocol works with bytes too
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes), f"Expected bytes, got {type(result)}"
```


# LLM-generated content at query #156
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a simple serializer that matches the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with various data types
    test_cases = [
        {"key": "value"},
        [1, 2, 3],
        "string",
        42,
        3.14,
        True,
        None,
        {"nested": {"data": [1, 2, 3]}}
    ]
    
    for test_data in test_cases:
        result = serializer.dumps(test_data)
        assert isinstance(result, str)
        assert json.loads(result) == test_data
    
    # Test that the serializer is identified as a text serializer
    assert is_text_serializer(serializer) is True
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    
    # Test that bytes serializer returns bytes
    bytes_result = bytes_serializer.dumps({"test": "data"})
    assert isinstance(bytes_result, bytes)
    assert json.loads(bytes_result.decode('utf-8')) == {"test": "data"}
    
    # Verify is_text_serializer returns False for bytes serializer
    assert is_text_serializer(bytes_serializer) is False
```


# LLM-generated content at query #157
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol works with loads method."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            if payload == '{"key": "value"}':
                return {"key": "value"}
            elif payload == "42":
                return 42
            elif payload == "null":
                return None
            elif payload == "[1, 2, 3]":
                return [1, 2, 3]
            elif payload == "invalid":
                raise ValueError("Invalid JSON")
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test basic JSON deserialization
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test integer deserialization
    result = serializer.loads("42")
    assert result == 42
    
    # Test null deserialization
    result = serializer.loads("null")
    assert result is None
    
    # Test list deserialization
    result = serializer.loads("[1, 2, 3]")
    assert result == [1, 2, 3]
    
    # Test with empty string
    try:
        serializer.loads("")
        assert False, "Should have raised an exception"
    except (ValueError, json.JSONDecodeError):
        pass
    
    # Test with invalid payload
    try:
        serializer.loads("invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test with complex nested JSON
    complex_json = '{"name": "test", "items": [1, 2, {"nested": True}]}'
    result = serializer.loads(complex_json)
    assert result == {"name": "test", "items": [1, 2, {"nested": True}]}
    
    # Test that the protocol is properly typed
    from typing import Protocol
    assert isinstance(serializer, _PDataSerializer)
    assert hasattr(serializer, 'loads')
    assert callable(serializer.loads)```


# LLM-generated content at query #158
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test loading valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a simple string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test loading a number
    result = serializer.loads('42')
    assert result == 42
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading null
    result = serializer.loads('null')
    assert result is None
    
    # Test loading boolean
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('false')
    assert result is False
```


# LLM-generated content at query #159
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a basic serializer that returns str
    class StrSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StrSerializer()
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with simple types
    class SimpleSerializer:
        def loads(self, payload: str) -> t.Any:
            if payload == "true":
                return True
            elif payload == "false":
                return False
            elif payload == "null":
                return None
            elif payload.startswith('"') and payload.endswith('"'):
                return payload[1:-1]
            elif "." in payload:
                return float(payload)
            else:
                try:
                    return int(payload)
                except ValueError:
                    return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    simple_serializer = SimpleSerializer()
    assert simple_serializer.loads("true") is True
    assert simple_serializer.loads("false") is False
    assert simple_serializer.loads("null") is None
    assert simple_serializer.loads('"hello"') == "hello"
    assert simple_serializer.loads("123") == 123
    assert simple_serializer.loads("3.14") == 3.14
    
    # Test with list payload
    class ListSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    list_serializer = ListSerializer()
    result = list_serializer.loads("[1, 2, 3]")
    assert result == [1, 2, 3]
    
    # Test with nested structure
    result = list_serializer.loads('{"a": {"b": [1, 2, 3]}}')
    assert result == {"a": {"b": [1, 2, 3]}}
    
    # Test with empty payload
    result = list_serializer.loads("{}")
    assert result == {}
    
    # Test with None/empty values
    result = list_serializer.loads('{"key": null}')
    assert result == {"key": None}
    
    # Test with special characters
    result = list_serializer.loads('{"key": "value with \\"quotes\\""}')
    assert result == {"key": 'value with "quotes"'}
```


# LLM-generated content at query #160
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StringSerializer()
    assert serializer.loads('{"key": "value"}') == {"key": "value"}
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    assert serializer.loads('"hello"') == "hello"
    assert serializer.loads('42') == 42
    assert serializer.loads('null') is None
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'[1, 2, 3]') == [1, 2, 3]
    assert bytes_serializer.loads(b'"hello"') == "hello"
    assert bytes_serializer.loads(b'42') == 42
    assert bytes_serializer.loads(b'null') is None
    
    # Test Protocol compliance
    assert isinstance(serializer, _PDataSerializer)
    assert isinstance(bytes_serializer, _PDataSerializer)


# LLM-generated content at query #161
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test that a serializer implementing the protocol works correctly
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert result == '{"key": "value"}'
    assert isinstance(result, str)
    
    # Test with different data types
    assert serializer.dumps(123) == "123"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps(None) == "null"
    
    # Verify roundtrip
    original = {"test": 42, "nested": {"a": "b"}}
    dumped = serializer.dumps(original)
    loaded = serializer.loads(dumped)
    assert loaded == original


# LLM-generated content at query #162
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol supports dumps method."""
    # Create a simple serializer that implements the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with basic types
    assert serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps("string") == '"string"'
    assert serializer.dumps(42) == "42"
    assert serializer.dumps(None) == "null"
    
    # Test that it returns str type
    result = serializer.dumps({})
    assert isinstance(result, str)


# LLM-generated content at query #163
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Create a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return b'{"test": "data"}'
    
    serializer = BytesSerializer()
    result = serializer.dumps({"test": "data"})
    assert isinstance(result, bytes)
    assert result == b'{"test": "data"}'
    
    # Create a serializer that returns str
    class StrSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return '{"test": "data"}'
    
    serializer = StrSerializer()
    result = serializer.dumps({"test": "data"})
    assert isinstance(result, str)
    assert result == '{"test": "data"}'
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Serializer():
    # Test with default parameters
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

    # Test with bytes secret key
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

    # Test with multiple secret keys for rotation
    serializer = Serializer(["key1", "key2", "key3"])
    assert serializer.secret_keys == [b"key1", b"key2", b"key3"]
    assert serializer.secret_key == b"key3"

    # Test with custom salt
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

    # Test with None salt
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

    # Test with custom serializer
    custom_serializer = type('CustomSerializer', (), {
        'dumps': lambda self, obj: str(obj),
        'loads': lambda self, s: int(s)
    })()
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True

    # Test with bytes serializer
    bytes_serializer = type('BytesSerializer', (), {
        'dumps': lambda self, obj: str(obj).encode(),
        'loads': lambda self, s: int(s.decode())
    })()
    serializer = Serializer("secret-key", serializer=bytes_serializer)
    assert serializer.is_text_serializer is False

    # Test with custom signer class
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

    # Test with signer kwargs
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

    # Test with fallback signers
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

    # Test with serializer kwargs
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

    # Test that default_fallback_signers is used when fallback_signers is None
    serializer = Serializer("secret-key")
    assert serializer.fallback_signers == []

    # Test that default_fallback_signers is copied, not referenced
    original_fallback = Serializer.default_fallback_signers
    serializer = Serializer("secret-key")
    assert serializer.fallback_signers is not original_fallback
```


# LLM-generated content at query #2
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default configuration
    serializer = Serializer("test-secret")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_key == b"test-secret"
    assert unsigners[0].salt == b"itsdangerous"

    # Test with custom salt
    serializer = Serializer("test-secret", salt="custom-salt")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert unsigners[0].salt == b"custom-salt"

    # Test with fallback signers as dict
    serializer = Serializer(
        "test-secret",
        fallback_signers=[{"digest_method": "sha256"}]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].salt == b"itsdangerous"

    # Test with fallback signers as tuple
    from itsdangerous.signer import HMACAlgorithm
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer(
        "test-secret",
        fallback_signers=[(custom_signer, {"digest_method": "sha256"})]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[1], custom_signer)

    # Test with fallback signers as class
    serializer = Serializer(
        "test-secret",
        fallback_signers=[custom_signer]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[1], custom_signer)

    # Test with multiple secret keys
    serializer = Serializer(["old-key", "new-key"])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert unsigners[0].secret_keys == [b"old-key", b"new-key"]

    # Test with multiple secret keys and fallback signers
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"digest_method": "sha256"}]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 3  # 1 default + 2 fallbacks (one per secret key)
    assert unsigners[0].secret_keys == [b"old-key", b"new-key"]
    assert unsigners[1].secret_key == b"old-key"
    assert unsigners[2].secret_key == b"new-key"

    # Test with None salt (uses signer default)
    serializer = Serializer("test-secret", salt=None)
    unsigners = list(serializer.iter_unsigners(None))
    assert len(unsigners) == 1
    assert unsigners[0].salt is None

    # Test overriding salt parameter
    serializer = Serializer("test-secret")
    unsigners = list(serializer.iter_unsigners(salt="override-salt"))
    assert len(unsigners) == 1
    assert unsigners[0].salt == b"override-salt"
```


# LLM-generated content at query #3
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Create a simple protocol implementation for testing
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    test_data = {"key": "value", "number": 42}
    
    result = serializer.dumps(test_data)
    
    assert isinstance(result, str)
    assert result == '{"key": "value", "number": 42}'
    
    # Test with different data types
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps("hello") == '"hello"'
    assert serializer.dumps(None) == "null"
    assert serializer.dumps(True) == "true"


# LLM-generated content at query #4
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            import pickle
            return pickle.dumps(obj)
        def loads(self, payload):
            import pickle
            return pickle.loads(payload)
    
    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = s_bytes.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    
    # Test with custom salt
    result_custom_salt = s.dumps({"key": "value"}, salt=b"custom-salt")
    assert isinstance(result_custom_salt, str)
    assert result_custom_salt != result
    
    # Test roundtrip
    original = {"key": "value", "number": 42}
    dumped = s.dumps(original)
    loaded = s.loads(dumped)
    assert loaded == original
    
    # Test with different data types
    complex_data = {"list": [1, 2, 3], "nested": {"a": "b"}, "bool": True, "none": None}
    dumped_complex = s.dumps(complex_data)
    loaded_complex = s.loads(dumped_complex)
    assert loaded_complex == complex_data
    
    # Test with key rotation (list of keys)
    s_rotation = Serializer(["old-key", "new-key"])
    result_rotation = s_rotation.dumps({"key": "value"})
    assert isinstance(result_rotation, str)
    # Should be signed with the newest key
    loaded_rotation = s_rotation.loads(result_rotation)
    assert loaded_rotation == {"key": "value"}


# LLM-generated content at query #5
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation that follows the protocol
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
    
    # Test with list
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test with nested structure
    assert serializer.loads('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}
    
    # Test with invalid JSON should raise exception
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')


# LLM-generated content at query #6
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Create a simple serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    test_data = {"key": "value"}
    result = serializer.dumps(test_data)
    
    assert isinstance(result, str)
    assert json.loads(result) == test_data
    
    # Test with different data types
    assert serializer.dumps(None) == "null"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps("string") == '"string"'
```


# LLM-generated content at query #7
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Contains separator between payload and signature
    
    # Test dumps and loads roundtrip
    data = {"user_id": 1, "name": "Alice"}
    signed = serializer.dumps(data)
    loaded = serializer.loads(signed)
    assert loaded == data
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    # Test roundtrip with bytes serializer
    data_bytes = {"number": 42}
    signed_bytes = bytes_serializer.dumps(data_bytes)
    loaded_bytes = bytes_serializer.loads(signed_bytes)
    assert loaded_bytes == data_bytes
    
    # Test with custom salt
    serializer_with_salt = Serializer("secret-key", salt="custom-salt")
    result_salt = serializer_with_salt.dumps({"key": "value"})
    assert isinstance(result_salt, str)
    
    # Test that different salts produce different signatures
    serializer_salt1 = Serializer("secret-key", salt="salt1")
    serializer_salt2 = Serializer("secret-key", salt="salt2")
    data_salt = {"test": "data"}
    signed1 = serializer_salt1.dumps(data_salt)
    signed2 = serializer_salt2.dumps(data_salt)
    assert signed1 != signed2
    
    # Test with serializer_kwargs
    serializer_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "indent": 2}
    )
    result_kwargs = serializer_kwargs.dumps({"b": 2, "a": 1})
    # The payload should have sorted keys and indentation
    payload_part = result_kwargs.rsplit(".", 1)[0]
    import base64
    decoded = base64.urlsafe_b64decode(payload_part + "==")
    assert b'"a": 1' in decoded
    assert b'"b": 2' in decoded
    
    # Test with key rotation (multiple secret keys)
    serializer_rotation = Serializer(["old-key", "new-key"])
    data_rotation = {"version": 1}
    signed_rotation = serializer_rotation.dumps(data_rotation)
    # Should be able to unsign with the same serializer (which tries newest key first)
    loaded_rotation = serializer_rotation.loads(signed_rotation)
    assert loaded_rotation == data_rotation
    
    # Test dumps returns correct type for text serializer
    text_serializer = Serializer("secret")
    text_result = text_serializer.dumps("test")
    assert isinstance(text_result, str)
    
    # Test dumps returns correct type for bytes serializer
    class SimpleBytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_ser = Serializer("secret", serializer=SimpleBytesSerializer())
    bytes_result = bytes_ser.dumps("test")
    assert isinstance(bytes_result, bytes)


# LLM-generated content at query #8
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test successful loading of valid payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return b"test"
        def loads(self, payload):
            return payload.decode()
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    
    # Test successful loading with bytes serializer
    result = bytes_serializer.load_payload(b"hello")
    assert result == "hello"
    
    # Test with custom serializer via parameter
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return int(payload)
    
    custom_serializer = Serializer("secret-key", serializer=CustomSerializer())
    result = custom_serializer.load_payload(b"42")
    assert result == 42
    
    # Test BadPayload exception on invalid data
    import pytest
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"invalid json")
    assert "Could not load the payload" in str(exc_info.value)
    assert exc_info.value.original_error is not None
    
    # Test BadPayload exception on empty payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test with custom serializer passed as parameter overriding instance serializer
    json_serializer = Serializer("secret-key", serializer=CustomSerializer())
    result = json_serializer.load_payload(b'{"key": "value"}', serializer=json)
    assert result == {"key": "value"}
    
    # Test with text serializer that returns non-string from dumps
    class TextSerializer:
        def dumps(self, obj):
            return "test"
        def loads(self, payload):
            return payload.upper()
    
    text_serializer = Serializer("secret-key", serializer=TextSerializer())
    result = text_serializer.load_payload(b"hello")
    assert result == "HELLO"
```


# LLM-generated content at query #9
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a concrete implementation that matches the protocol
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
    
    # Test with different data types
    assert serializer.dumps(42) == "42"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps(None) == "null"
    
    # Verify it's a text serializer (returns str)
    assert isinstance(serializer.dumps({}), str)
```


# LLM-generated content at query #10
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON serializer
    serializer = _PDataSerializer()
    serializer.loads = lambda payload: json.loads(payload)
    
    # Test loading a valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading a simple string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test loading a number
    result = serializer.loads('42')
    assert result == 42
    
    # Test loading null
    result = serializer.loads('null')
    assert result is None
    
    # Test loading a boolean
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('false')
    assert result is False
    
    # Test with a custom serializer that returns different types
    class CustomSerializer:
        def loads(self, payload):
            return payload.upper()
    
    custom_serializer = _PDataSerializer()
    custom_serializer.loads = CustomSerializer().loads
    
    result = custom_serializer.loads("hello")
    assert result == "HELLO"
    
    # Test with bytes payload
    serializer_bytes = _PDataSerializer()
    serializer_bytes.loads = lambda payload: json.loads(payload.decode('utf-8'))
    
    result = serializer_bytes.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
```


# LLM-generated content at query #11
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a simple serializer that implements the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    test_data = {"key": "value", "number": 42}
    
    # Test basic serialization
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    assert result == '{"key": "value", "number": 42}'
    
    # Test with different data types
    assert serializer.dumps(None) == 'null'
    assert serializer.dumps(True) == 'true'
    assert serializer.dumps([1, 2, 3]) == '[1, 2, 3]'
    
    # Verify the result can be deserialized back
    assert serializer.loads(result) == test_data
    
    # Test with empty data
    assert serializer.dumps({}) == '{}'
    assert serializer.dumps([]) == '[]'
```


# LLM-generated content at query #12
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.count(".") == 2  # payload, timestamp, signature
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert result_bytes.count(b".") == 2
    
    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert result_with_salt != result
    assert isinstance(result_with_salt, str)
    
    # Test with different data types
    result_str = serializer.dumps("string data")
    assert isinstance(result_str, str)
    
    result_int = serializer.dumps(42)
    assert isinstance(result_int, str)
    
    # Test that the output can be verified with loads
    original = {"test": "data", "number": 123}
    signed = serializer.dumps(original)
    loaded = serializer.loads(signed)
    assert loaded == original
    
    # Test with multiple secret keys (key rotation)
    multi_serializer = Serializer(["old-key", "new-key"])
    result_multi = multi_serializer.dumps({"key": "value"})
    assert isinstance(result_multi, str)
    # Should verify with the newest key
    loaded_multi = multi_serializer.loads(result_multi)
    assert loaded_multi == {"key": "value"}
    
    # Test with serializer_kwargs
    class CustomSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj, **kwargs)
        def loads(self, payload):
            return json.loads(payload)
    
    kwargs_serializer = Serializer(
        "secret-key",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True}
    )
    result_kwargs = kwargs_serializer.dumps({"b": 2, "a": 1})
    assert isinstance(result_kwargs, str)
    loaded_kwargs = kwargs_serializer.loads(result_kwargs)
    assert loaded_kwargs == {"a": 1, "b": 2}  # sorted order in serialization
```


# LLM-generated content at query #13
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert serializer.loads(result) == {"key": "value"}

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))

    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert bytes_serializer.loads(result_bytes) == {"key": "value"}

    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert serializer.loads(result_with_salt) == {"key": "value"}
    # Should fail with different salt
    with pytest.raises(BadSignature):
        serializer.loads(result_with_salt, salt="wrong-salt")

    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_kwargs = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    assert serializer_with_kwargs.loads(result_kwargs) == {"a": 1, "b": 2}

    # Test with key rotation
    serializer_rotation = Serializer(["old-key", "new-key"])
    result_rotation = serializer_rotation.dumps({"data": "test"})
    # Should unsign with both keys
    assert serializer_rotation.loads(result_rotation) == {"data": "test"}

    # Test with empty dict
    result_empty = serializer.dumps({})
    assert serializer.loads(result_empty) == {}

    # Test with None value
    result_none = serializer.dumps(None)
    assert serializer.loads(result_none) is None

    # Test with list
    result_list = serializer.dumps([1, 2, 3])
    assert serializer.loads(result_list) == [1, 2, 3]

    # Test with custom signer
    class CustomSigner(Signer):
        pass

    custom_signer_serializer = Serializer("secret-key", signer=CustomSigner)
    result_custom = custom_signer_serializer.dumps({"key": "value"})
    assert custom_signer_serializer.loads(result_custom) == {"key": "value"}

    # Test with signer_kwargs
    serializer_signer_kwargs = Serializer(
        "secret-key",
        signer_kwargs={"key_derivation": "hmac"}
    )
    result_signer_kwargs = serializer_signer_kwargs.dumps({"key": "value"})
    assert serializer_signer_kwargs.loads(result_signer_kwargs) == {"key": "value"}


# LLM-generated content at query #14
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer.dumps can be called with an object and returns the expected type."""
    # Create a concrete implementation of _PDataSerializer for testing
    class TestSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = TestSerializer()
    test_data = {"key": "value", "number": 42}
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    assert result == '{"key": "value", "number": 42}'
```


# LLM-generated content at query #15
#--------------------------

```python
def test_Serializer_load_payload():
    """Test load_payload method of Serializer class."""
    # Test with default json serializer (text serializer)
    serializer = Serializer("test-secret-key")
    
    # Test successful loading
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload):
            return payload.decode() if isinstance(payload, bytes) else payload
        
        def dumps(self, obj):
            return obj.encode() if isinstance(obj, str) else obj
    
    bytes_serializer = Serializer("test-secret-key", serializer=BytesSerializer())
    bytes_payload = b'hello'
    assert bytes_serializer.load_payload(bytes_payload) == 'hello'
    
    # Test with custom serializer that returns text
    class TextSerializer:
        def loads(self, payload):
            return payload.upper()
        
        def dumps(self, obj):
            return obj.lower()
    
    text_serializer = Serializer("test-secret-key", serializer=TextSerializer())
    text_payload = b'hello'
    assert text_serializer.load_payload(text_payload) == 'HELLO'
    
    # Test with explicit serializer parameter
    class ExplicitSerializer:
        def loads(self, payload):
            return payload[::-1]
        
        def dumps(self, obj):
            return obj
    
    explicit_serializer = Serializer("test-secret-key", serializer=ExplicitSerializer())
    explicit_payload = b'hello'
    result = explicit_serializer.load_payload(explicit_payload, serializer=ExplicitSerializer())
    assert result == 'olleh'
    
    # Test with invalid payload (raises BadPayload)
    invalid_serializer = Serializer("test-secret-key")
    try:
        invalid_serializer.load_payload(b'invalid json')
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None
    
    # Test with empty payload
    try:
        serializer.load_payload(b'')
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
    
    # Test with binary serializer that doesn't decode
    class BinarySerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return obj
    
    binary_serializer = Serializer("test-secret-key", serializer=BinarySerializer())
    binary_payload = b'\x00\x01\x02'
    assert binary_serializer.load_payload(binary_payload) == binary_payload
```


# LLM-generated content at query #16
#--------------------------

```python
def test_Serializer_load_payload():
    """Test the load_payload method of Serializer."""
    
    def test_json_serializer_loads_text_payload():
        """Test loading a JSON payload from text."""
        serializer = Serializer("secret-key")
        payload = b'{"key": "value"}'
        result = serializer.load_payload(payload)
        assert result == {"key": "value"}
    
    def test_json_serializer_loads_binary_payload_with_text_serializer():
        """Test loading a JSON payload from binary with text serializer."""
        serializer = Serializer("secret-key")
        payload = b'{"number": 42}'
        result = serializer.load_payload(payload)
        assert result == {"number": 42}
    
    def test_custom_serializer_loads_bytes():
        """Test loading with a custom bytes serializer."""
        class BytesSerializer:
            def loads(self, payload):
                return payload.decode("utf-8")
        
        serializer = Serializer("secret-key", serializer=BytesSerializer())
        payload = b"hello world"
        result = serializer.load_payload(payload)
        assert result == "hello world"
    
    def test_load_payload_with_custom_serializer_param():
        """Test loading payload with an explicit serializer parameter."""
        class CustomSerializer:
            def loads(self, payload):
                return {"custom": payload}
        
        serializer = Serializer("secret-key")
        payload = b"test data"
        result = serializer.load_payload(payload, serializer=CustomSerializer())
        assert result == {"custom": "test data"}
    
    def test_load_payload_raises_bad_payload_on_invalid_json():
        """Test that invalid JSON raises BadPayload."""
        serializer = Serializer("secret-key")
        payload = b"invalid json"
        
        with pytest.raises(BadPayload) as exc_info:
            serializer.load_payload(payload)
        assert "Could not load the payload" in str(exc_info.value)
        assert exc_info.value.original_error is not None
    
    def test_load_payload_raises_bad_payload_on_empty_payload():
        """Test that empty payload raises BadPayload."""
        serializer = Serializer("secret-key")
        payload = b""
        
        with pytest.raises(BadPayload):
            serializer.load_payload(payload)
    
    def test_load_payload_with_text_serializer_utf8_encoded():
        """Test that text serializer decodes UTF-8 properly."""
        serializer = Serializer("secret-key")
        payload = '{"message": "héllo"}'.encode("utf-8")
        result = serializer.load_payload(payload)
        assert result == {"message": "héllo"}
    
    def test_load_payload_with_text_serializer_non_utf8_raises_error():
        """Test that non-UTF-8 bytes with text serializer raises BadPayload."""
        serializer = Serializer("secret-key")
        payload = b'\xff\xfe\x00\x00'
        
        with pytest.raises(BadPayload):
            serializer.load_payload(payload)
    
    def test_load_payload_preserves_original_error():
        """Test that the original error is preserved in BadPayload."""
        serializer = Serializer("secret-key")
        payload = b"not json"
        
        try:
            serializer.load_payload(payload)
        except BadPayload as e:
            assert e.original_error is not None
            assert isinstance(e.original_error, json.JSONDecodeError)
    
    def test_load_payload_with_custom_serializer_raises_bad_payload():
        """Test that custom serializer errors are wrapped in BadPayload."""
        class FailingSerializer:
            def loads(self, payload):
                raise ValueError("Custom error")
        
        serializer = Serializer("secret-key", serializer=FailingSerializer())
        payload = b"some data"
        
        with pytest.raises(BadPayload) as exc_info:
            serializer.load_payload(payload)
        assert isinstance(exc_info.value.original_error, ValueError)
        assert str(exc_info.value.original_error) == "Custom error"
    
    def test_load_payload_with_bytes_serializer():
        """Test loading with a bytes serializer (non-text)."""
        class BytesSerializer:
            def dumps(self, obj):
                return obj.encode("utf-8")
            def loads(self, payload):
                return payload
        
        serializer = Serializer("secret-key", serializer=BytesSerializer())
        payload = b"test payload"
        result = serializer.load_payload(payload)
        assert result == b"test payload"
    
    def test_load_payload_with_none_serializer_uses_default():
        """Test that None serializer parameter uses the instance's serializer."""
        serializer = Serializer("secret-key")
        payload = b'{"key": "value"}'
        result = serializer.load_payload(payload)
        assert result == {"key": "value"}
    
    def test_load_payload_concurrent_calls():
        """Test that multiple calls to load_payload work correctly."""
        serializer = Serializer("secret-key")
        payload1 = b'{"a": 1}'
        payload2 = b'{"b": 2}'
        
        result1 = serializer.load_payload(payload1)
        result2 = serializer.load_payload(payload2)
        
        assert result1 == {"a": 1}
        assert result2 == {"b": 2}
    
    # Run all test functions
    test_json_serializer_loads_text_payload()
    test_json_serializer_loads_binary_payload_with_text_serializer()
    test_custom_serializer_loads_bytes()
    test_load_payload_with_custom_serializer_param()
    test_load_payload_raises_bad_payload_on_invalid_json()
    test_load_payload_raises_bad_payload_on_empty_payload()
    test_load_payload_with_text_serializer_utf8_encoded()
    test_load_payload_with_text_serializer_non_utf8_raises_error()
    test_load_payload_preserves_original_error()
    test_load_payload_with_custom_serializer_raises_bad_payload()
    test_load_payload_with_bytes_serializer()
    test_load_payload_with_none_serializer_uses_default()
    test_load_payload_concurrent_calls()
```


# LLM-generated content at query #17
#--------------------------

```python
def test_Serializer_dumps():
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result
    assert len(result.split(".")) == 3

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))

    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    bytes_result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(bytes_result, bytes)
    assert b"." in bytes_result

    # Test with custom salt
    custom_salt_result = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert custom_salt_result != result

    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True}
    )
    kwargs_result = serializer_with_kwargs.dumps({"b": 1, "a": 2})
    expected_payload = json.dumps({"a": 2, "b": 1}, sort_keys=True)
    assert expected_payload.encode("utf-8") in want_bytes(kwargs_result)

    # Test that dumps returns consistent results for same input
    result1 = serializer.dumps("test")
    result2 = serializer.dumps("test")
    assert result1 == result2

    # Test with fallback signers configuration
    fallback_serializer = Serializer(
        "secret-key",
        fallback_signers=[{"digest_method": "sha256"}]
    )
    fallback_result = fallback_serializer.dumps("test")
    assert isinstance(fallback_result, str)
```


# LLM-generated content at query #18
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that dumps method works correctly for protocol compliance."""
    # Create a mock serializer that conforms to _PDataSerializer
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


# LLM-generated content at query #19
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert serializer.loads(result) == {"key": "value"}

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert bytes_serializer.loads(result) == {"key": "value"}

    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert serializer.loads(result_with_salt, salt="custom-salt") == {"key": "value"}

    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    assert result.count('"a"') == 1
    assert result.count('"b"') == 1

    # Test that dumps produces consistent results with same input
    result1 = serializer.dumps({"key": "value"})
    result2 = serializer.dumps({"key": "value"})
    # Note: Different salts or timestamps may cause different results
    assert isinstance(result1, str) and isinstance(result2, str)

    # Test with empty dict
    result = serializer.dumps({})
    assert isinstance(result, str)
    assert serializer.loads(result) == {}

    # Test with list
    result = serializer.dumps([1, 2, 3])
    assert isinstance(result, str)
    assert serializer.loads(result) == [1, 2, 3]

    # Test with None
    result = serializer.dumps(None)
    assert isinstance(result, str)
    assert serializer.loads(result) is None
```


# LLM-generated content at query #20
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)

    serializer = StringSerializer()
    assert serializer.loads('{"key": "value"}') == {"key": "value"}
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    assert serializer.loads('"string"') == "string"
    assert serializer.loads('null') is None

    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')

    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'[1, 2, 3]') == [1, 2, 3]

    # Test with a custom serializer that returns int
    class IntSerializer:
        def loads(self, payload: str) -> t.Any:
            return int(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)

    int_serializer = IntSerializer()
    assert int_serializer.loads("42") == 42
    assert int_serializer.loads("-10") == -10

    # Test with complex nested data
    complex_data = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
    result = serializer.loads(complex_data)
    assert result == {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
    assert len(result["users"]) == 2
```


# LLM-generated content at query #21
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert serializer.loads(result) == data
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return json.loads(payload)
        def dumps(self, obj):
            return json.dumps(obj).encode()
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    bytes_result = bytes_serializer.dumps(data)
    assert isinstance(bytes_result, bytes)
    assert bytes_serializer.loads(bytes_result) == data
    
    # Test with custom salt
    result_with_salt = serializer.dumps(data, salt="custom-salt")
    assert serializer.loads(result_with_salt, salt="custom-salt") == data
    
    # Test with empty dict
    empty_result = serializer.dumps({})
    assert serializer.loads(empty_result) == {}
    
    # Test with list
    list_result = serializer.dumps([1, 2, 3])
    assert serializer.loads(list_result) == [1, 2, 3]
    
    # Test with string
    str_result = serializer.dumps("test")
    assert serializer.loads(str_result) == "test"


# LLM-generated content at query #22
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that loads method works with string input."""
    serializer = json
    payload = '{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}

    # Test with bytes payload
    serializer = json
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}

    # Test with custom serializer that returns string
    class CustomSerializer:
        def loads(self, payload):
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            return eval(payload)
        
        def dumps(self, obj):
            return str(obj)
    
    serializer = CustomSerializer()
    payload = '{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}

    # Test with integer payload
    serializer = json
    payload = "42"
    result = serializer.loads(payload)
    assert result == 42

    # Test with list payload
    serializer = json
    payload = "[1, 2, 3]"
    result = serializer.loads(payload)
    assert result == [1, 2, 3]
```


# LLM-generated content at query #23
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test basic JSON loading
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading a string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test loading a number
    result = serializer.loads('42')
    assert result == 42
    
    # Test loading null
    result = serializer.loads('null')
    assert result is None
    
    # Test loading boolean
    result = serializer.loads('true')
    assert result is True
    
    # Test that it raises on invalid input
    with pytest.raises(Exception):
        serializer.loads('invalid json')
```


# LLM-generated content at query #24
#--------------------------

```python
def test__PDataSerializer_loads():
    import json
    
    # Create a concrete implementation that matches the protocol
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    # Test with text serializer
    text_serializer = TextSerializer()
    assert text_serializer.loads('{"key": "value"}') == {"key": "value"}
    assert text_serializer.loads('42') == 42
    assert text_serializer.loads('"hello"') == "hello"
    assert text_serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test with bytes serializer
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'42') == 42
    assert bytes_serializer.loads(b'"hello"') == "hello"
    assert bytes_serializer.loads(b'[1, 2, 3]') == [1, 2, 3]
    
    # Test with the default json module (which is a valid _PDataSerializer)
    assert json.loads('{"a": 1}') == {"a": 1}
    assert json.loads(b'{"a": 1}') == {"a": 1}
```


# LLM-generated content at query #25
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method is callable with appropriate types."""
    # Create a mock serializer that conforms to _PDataSerializer[str]
    class MockStringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)

    # Create a mock serializer that conforms to _PDataSerializer[bytes]
    class MockBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')

    # Test with string serializer
    str_serializer = MockStringSerializer()
    result = str_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'

    # Test with bytes serializer
    bytes_serializer = MockBytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'

    # Test with default json serializer
    json_serializer = json
    result = json_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
```


# LLM-generated content at query #26
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer
    serializer = Serializer("test-secret")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

    # Test with custom text serializer
    class TextSerializer:
        def loads(self, payload):
            return payload.upper()
        def dumps(self, obj):
            return str(obj)
    
    text_serializer = Serializer("test-secret", serializer=TextSerializer())
    text_payload = b"hello"
    text_result = text_serializer.load_payload(text_payload)
    assert text_result == "HELLO"

    # Test with custom bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return payload
        def dumps(self, obj):
            return obj
    
    bytes_serializer = Serializer("test-secret", serializer=BytesSerializer())
    bytes_payload = b"test"
    bytes_result = bytes_serializer.load_payload(bytes_payload)
    assert bytes_result == b"test"

    # Test with explicit serializer parameter
    serializer_explicit = Serializer("test-secret")
    explicit_payload = b'{"a": 1}'
    explicit_result = serializer_explicit.load_payload(explicit_payload, serializer=json)
    assert explicit_result == {"a": 1}

    # Test with invalid JSON payload for default serializer
    invalid_payload = b"invalid json"
    with pytest.raises(BadPayload):
        serializer.load_payload(invalid_payload)

    # Test with empty payload
    empty_payload = b""
    with pytest.raises(BadPayload):
        serializer.load_payload(empty_payload)

    # Test with payload that causes custom serializer to raise exception
    class FailingSerializer:
        def loads(self, payload):
            raise ValueError("Failed to load")
        def dumps(self, obj):
            return str(obj)
    
    failing_serializer = Serializer("test-secret", serializer=FailingSerializer())
    with pytest.raises(BadPayload):
        failing_serializer.load_payload(b"test")

    # Test with text serializer that returns non-string payload
    class NonStringTextSerializer:
        def loads(self, payload):
            return 42
        def dumps(self, obj):
            return str(obj)
    
    non_string_serializer = Serializer("test-secret", serializer=NonStringTextSerializer())
    non_string_result = non_string_serializer.load_payload(b"test")
    assert non_string_result == 42
```


# LLM-generated content at query #27
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test loading valid JSON payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading invalid JSON payload
    invalid_payload = b"not valid json"
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(invalid_payload)
    assert "Could not load the payload" in str(exc_info.value)
    assert exc_info.value.original_error is not None
    
    # Test with custom bytes serializer
    class BytesSerializer:
        @staticmethod
        def loads(payload: bytes) -> t.Any:
            return {"data": payload}
        
        @staticmethod
        def dumps(obj: t.Any) -> bytes:
            return b"test"
    
    bytes_serializer = BytesSerializer()
    serializer_bytes = Serializer("secret-key", serializer=bytes_serializer)
    
    # Test loading with bytes serializer
    payload_bytes = b"some bytes data"
    result = serializer_bytes.load_payload(payload_bytes)
    assert result == {"data": b"some bytes data"}
    
    # Test with text serializer that returns string
    class TextSerializer:
        @staticmethod
        def loads(payload: str) -> t.Any:
            return {"text": payload}
        
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return "test"
    
    text_serializer = TextSerializer()
    serializer_text = Serializer("secret-key", serializer=text_serializer)
    
    # Test loading with text serializer
    payload_text = b"hello world"
    result = serializer_text.load_payload(payload_text)
    assert result == {"text": "hello world"}
    
    # Test loading with custom serializer that raises exception
    class FailingSerializer:
        @staticmethod
        def loads(payload: t.Any) -> t.Any:
            raise ValueError("Custom error")
        
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return "test"
    
    failing_serializer = FailingSerializer()
    serializer_fail = Serializer("secret-key", serializer=failing_serializer)
    
    with pytest.raises(BadPayload) as exc_info:
        serializer_fail.load_payload(b"any data")
    assert "Could not load the payload" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, ValueError)
    
    # Test with explicit serializer parameter
    explicit_result = serializer.load_payload(payload, serializer=json)
    assert explicit_result == {"key": "value"}
    
    # Test with explicit serializer that is text serializer
    explicit_text_result = serializer_text.load_payload(
        b"hello world", 
        serializer=TextSerializer()
    )
    assert explicit_text_result == {"text": "hello world"}
```


# LLM-generated content at query #28
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a text serializer (json)
    serializer = json
    pdata = _PDataSerializer()
    pdata.loads = serializer.loads
    pdata.dumps = serializer.dumps
    
    # Test loading valid JSON data
    result = pdata.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = pdata.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading a string
    result = pdata.loads('"hello"')
    assert result == "hello"
    
    # Test loading a number
    result = pdata.loads('42')
    assert result == 42
    
    # Test loading null
    result = pdata.loads('null')
    assert result is None
    
    # Test loading boolean
    result = pdata.loads('true')
    assert result is True
    
    # Test with a custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    pdata_bytes = _PDataSerializer()
    pdata_bytes.loads = bytes_serializer.loads
    pdata_bytes.dumps = bytes_serializer.dumps
    
    result = pdata_bytes.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading invalid JSON should raise an exception
    with pytest.raises(Exception):
        pdata.loads('{invalid json}')
    
    # Test loading empty string
    with pytest.raises(Exception):
        pdata.loads('')
```


# LLM-generated content at query #29
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test loading valid json payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading valid json list payload
    payload = b'[1, 2, 3]'
    result = serializer.load_payload(payload)
    assert result == [1, 2, 3]
    
    # Test loading valid json string payload
    payload = b'"hello"'
    result = serializer.load_payload(payload)
    assert result == "hello"
    
    # Test loading valid json number payload
    payload = b'42'
    result = serializer.load_payload(payload)
    assert result == 42
    
    # Test loading valid json null payload
    payload = b'null'
    result = serializer.load_payload(payload)
    assert result is None
    
    # Test loading valid json boolean payload
    payload = b'true'
    result = serializer.load_payload(payload)
    assert result is True
    
    # Test loading invalid json payload raises BadPayload
    payload = b'{invalid json}'
    with pytest.raises(BadPayload):
        serializer.load_payload(payload)
    
    # Test loading empty payload raises BadPayload
    payload = b''
    with pytest.raises(BadPayload):
        serializer.load_payload(payload)
    
    # Test loading non-utf8 bytes with text serializer raises BadPayload
    payload = b'\xff\xfe'
    with pytest.raises(BadPayload):
        serializer.load_payload(payload)
    
    # Test with custom serializer that returns text
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"custom": payload}
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    text_serializer = TextSerializer()
    serializer_text = Serializer("secret-key", serializer=text_serializer)
    payload = b"test data"
    result = serializer_text.load_payload(payload)
    assert result == {"custom": "test data"}
    
    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return {"custom": payload.decode()}
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode()
    
    bytes_serializer = BytesSerializer()
    serializer_bytes = Serializer("secret-key", serializer=bytes_serializer)
    payload = b"test data"
    result = serializer_bytes.load_payload(payload)
    assert result == {"custom": "test data"}
    
    # Test with custom serializer that raises exception
    class FailingSerializer:
        def loads(self, payload: t.Any) -> t.Any:
            raise ValueError("Custom error")
        def dumps(self, obj: t.Any) -> t.Any:
            return str(obj)
    
    failing_serializer = FailingSerializer()
    serializer_fail = Serializer("secret-key", serializer=failing_serializer)
    payload = b"any data"
    with pytest.raises(BadPayload) as exc_info:
        serializer_fail.load_payload(payload)
    assert "Could not load the payload" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, ValueError)
    
    # Test with override serializer parameter
    override_serializer = BytesSerializer()
    payload = b"override test"
    result = serializer.load_payload(payload, serializer=override_serializer)
    assert result == {"custom": "override test"}
    
    # Test that original error is preserved in BadPayload exception
    class ExceptionSerializer:
        def loads(self, payload: t.Any) -> t.Any:
            raise RuntimeError("Original error")
        def dumps(self, obj: t.Any) -> t.Any:
            return str(obj)
    
    exception_serializer = ExceptionSerializer()
    serializer_exc = Serializer("secret-key", serializer=exception_serializer)
    payload = b"data"
    with pytest.raises(BadPayload) as exc_info:
        serializer_exc.load_payload(payload)
    assert isinstance(exc_info.value.original_error, RuntimeError)
    assert str(exc_info.value.original_error) == "Original error"
```


# LLM-generated content at query #30
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a simple serializer that implements the protocol
    class JSONSerializer:
        def loads(self, payload: str | bytes) -> t.Any:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str | bytes:
            return json.dumps(obj)
    
    serializer = JSONSerializer()
    
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
    
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('null')
    assert result is None
    
    # Test with empty payload
    result = serializer.loads('{}')
    assert result == {}
    
    result = serializer.loads('[]')
    assert result == []
```


# LLM-generated content at query #31
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test the loads method of _PDataSerializer protocol."""
    # Create a mock serializer that implements the protocol
    class MockSerializer:
        def loads(self, payload):
            return {"test": payload}
        
        def dumps(self, obj):
            return str(obj)
    
    serializer = MockSerializer()
    
    # Test with string payload
    result = serializer.loads("hello")
    assert result == {"test": "hello"}
    
    # Test with bytes payload
    result = serializer.loads(b"world")
    assert result == {"test": b"world"}
    
    # Test with integer payload
    result = serializer.loads(42)
    assert result == {"test": 42}
    
    # Test with dict payload
    result = serializer.loads({"key": "value"})
    assert result == {"test": {"key": "value"}}
    
    # Test with list payload
    result = serializer.loads([1, 2, 3])
    assert result == {"test": [1, 2, 3]}
    
    # Test with None payload
    result = serializer.loads(None)
    assert result == {"test": None}
    
    # Test that loads method exists and is callable
    assert callable(getattr(serializer, "loads", None))
```


# LLM-generated content at query #32
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
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with simple types
    assert serializer.loads('42') == 42
    assert serializer.loads('"hello"') == "hello"
    assert serializer.loads('null') is None
    assert serializer.loads('true') is True
    
    # Test with list
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test that it raises on invalid input
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
```


# LLM-generated content at query #33
#--------------------------

```python
def test_Serializer_load_payload():
    serializer = Serializer("secret-key")
    
    # Test with default json serializer
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with custom text serializer
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"custom": payload}
        def dumps(self, obj: t.Any) -> str:
            return "dummy"
    
    text_serializer = TextSerializer()
    serializer2 = Serializer("secret-key", serializer=text_serializer)
    payload2 = b"test payload"
    result2 = serializer2.load_payload(payload2)
    assert result2 == {"custom": "test payload"}
    
    # Test with custom bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return {"custom": payload.decode("utf-8")}
        def dumps(self, obj: t.Any) -> bytes:
            return b"dummy"
    
    bytes_serializer = BytesSerializer()
    serializer3 = Serializer("secret-key", serializer=bytes_serializer)
    payload3 = b"test bytes"
    result3 = serializer3.load_payload(payload3)
    assert result3 == {"custom": "test bytes"}
    
    # Test with override serializer parameter
    class OverrideSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"override": payload}
        def dumps(self, obj: t.Any) -> str:
            return "dummy"
    
    override_serializer = OverrideSerializer()
    payload4 = b"override payload"
    result4 = serializer.load_payload(payload4, serializer=override_serializer)
    assert result4 == {"override": "override payload"}
    
    # Test with invalid payload that raises BadPayload
    invalid_payload = b"invalid json"
    try:
        serializer.load_payload(invalid_payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None
    
    # Test with empty bytes payload
    empty_payload = b""
    try:
        serializer.load_payload(empty_payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
    
    # Test with None serializer parameter (uses default)
    payload5 = b'{"test": 123}'
    result5 = serializer.load_payload(payload5, serializer=None)
    assert result5 == {"test": 123}
```


# LLM-generated content at query #34
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol defines dumps method correctly."""
    # Test with a serializer that returns str
    class StrSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any, /) -> str:
            return json.dumps(obj)
    
    str_serializer: _PDataSerializer[str] = StrSerializer()
    result = str_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes, /) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any, /) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer: _PDataSerializer[bytes] = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test that dumps preserves type annotation
    import inspect
    sig = inspect.signature(_PDataSerializer[str].dumps)
    assert list(sig.parameters.keys()) == ['payload']
    assert sig.parameters['payload'].kind == inspect.Parameter.POSITIONAL_ONLY
    
    # Test protocol structural compatibility
    class MinimalSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any, /) -> str:
            return str(obj)
    
    minimal: _PDataSerializer[str] = MinimalSerializer()
    assert minimal.dumps(42) == "42" 


# LLM-generated content at query #35
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Contains signature separator
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode()
        def loads(self, payload):
            return json.loads(payload.decode())
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert b"." in result  # Contains signature separator
    
    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert result_with_salt != result  # Different salt produces different result
    
    # Test that dumps produces a valid signed payload that can be verified
    loaded = serializer.loads(result)
    assert loaded == {"key": "value"}
    
    # Test with different secret key produces different result
    other_serializer = Serializer("different-secret")
    other_result = other_serializer.dumps({"key": "value"})
    assert other_result != result
    
    # Test with key rotation (list of keys)
    rotation_serializer = Serializer(["old-key", "new-key"])
    result_rotation = rotation_serializer.dumps({"key": "value"})
    assert isinstance(result_rotation, str)
    
    # Verify it can be loaded with any of the keys
    loaded_rotation = rotation_serializer.loads(result_rotation)
    assert loaded_rotation == {"key": "value"}


# LLM-generated content at query #36
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test the loads method of _PDataSerializer protocol."""
    
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"test": payload}
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    serializer = TestSerializer()
    
    # Test basic string payload
    result = serializer.loads("hello")
    assert result == {"test": "hello"}
    
    # Test with empty string
    result = serializer.loads("")
    assert result == {"test": ""}
    
    # Test with JSON-like string
    result = serializer.loads('{"key": "value"}')
    assert result == {"test": '{"key": "value"}'}
    
    # Test with numeric string
    result = serializer.loads("123")
    assert result == {"test": "123"}
    
    # Test with special characters
    result = serializer.loads("!@#$%^&*()")
    assert result == {"test": "!@#$%^&*()"}
    
    # Test with Unicode string
    result = serializer.loads("héllo wörld")
    assert result == {"test": "héllo wörld"}
    
    # Test that the method is callable with only payload argument
    # (no additional positional or keyword arguments)
    class StrictSerializer:
        def loads(self, payload: str, /) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    strict_serializer = StrictSerializer()
    result = strict_serializer.loads("test")
    assert result == "test"  # type: ignore[comparison-overlap]
    
    # Verify that loads returns Any type (can return different types)
    class VariedSerializer:
        def loads(self, payload: str, /) -> t.Any:
            if payload == "int":
                return 42
            elif payload == "list":
                return [1, 2, 3]
            elif payload == "dict":
                return {"key": "value"}
            elif payload == "none":
                return None
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    varied = VariedSerializer()
    assert varied.loads("int") == 42
    assert varied.loads("list") == [1, 2, 3]
    assert varied.loads("dict") == {"key": "value"}
    assert varied.loads("none") is None
    assert varied.loads("string") == "string"  # type: ignore[comparison-overlap]
```


# LLM-generated content at query #37
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test loading a valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a simple string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test loading a number
    result = serializer.loads('42')
    assert result == 42
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading a boolean
    result = serializer.loads('true')
    assert result is True
    
    # Test loading null
    result = serializer.loads('null')
    assert result is None
    
    # Test loading empty dict
    result = serializer.loads('{}')
    assert result == {}
    
    # Test loading empty list
    result = serializer.loads('[]')
    assert result == []
    
    # Test loading nested structure
    result = serializer.loads('{"a": {"b": [1, 2, 3]}}')
    assert result == {"a": {"b": [1, 2, 3]}}
    
    # Test with bytes input (should still work with the protocol)
    class BytesMockSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesMockSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
```


# LLM-generated content at query #38
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a string serializer
    str_serializer = _PDataSerializer[str]()
    
    # Mock the loads method
    def mock_str_loads(payload: str) -> t.Any:
        if payload == '{"key": "value"}':
            return {"key": "value"}
        raise ValueError("Invalid JSON")
    
    str_serializer.loads = mock_str_loads
    
    # Test successful load
    result = str_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    bytes_serializer = _PDataSerializer[bytes]()
    
    def mock_bytes_loads(payload: bytes) -> t.Any:
        if payload == b'{"key": "value"}':
            return {"key": "value"}
        raise ValueError("Invalid JSON")
    
    bytes_serializer.loads = mock_bytes_loads
    
    # Test successful load with bytes
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that loads is callable
    assert callable(str_serializer.loads)
    assert callable(bytes_serializer.loads)
    
    # Test that loads accepts only one positional argument (the payload)
    # This is enforced by the protocol signature
    try:
        str_serializer.loads("test", "extra_arg")
        assert False, "loads should raise TypeError with extra arguments"
    except TypeError:
        pass
    
    # Test with custom serializer class
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj).lower()
    
    custom_serializer = _PDataSerializer[str]()
    custom_serializer.loads = CustomSerializer().loads
    
    result = custom_serializer.loads("hello")
    assert result == "HELLO"
```


# LLM-generated content at query #39
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a concrete implementation of _PDataSerializer
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test dumps with various data types
    assert serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps("string") == '"string"'
    assert serializer.dumps(42) == "42"
    assert serializer.dumps(None) == "null"
    assert serializer.dumps(True) == "true"
    assert serializer.dumps(False) == "false"
    
    # Test that dumps returns a string
    assert isinstance(serializer.dumps({}), str)


# LLM-generated content at query #40
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol is properly duck-typed."""
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'


# LLM-generated content at query #41
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (JSON)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
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
    s_custom_salt = Serializer("secret-key", salt=b"custom-salt")
    result_salt = s_custom_salt.dumps({"key": "value"})
    assert s_custom_salt.loads(result_salt) == {"key": "value"}

    # Test with different salt parameter
    result_diff_salt = s.dumps({"key": "value"}, salt="different-salt")
    assert s.loads(result_diff_salt, salt="different-salt") == {"key": "value"}

    # Test that dumps creates a signed payload that fails with wrong key
    s2 = Serializer("different-secret")
    try:
        s2.loads(result)
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

    # Test with empty object
    result_empty = s.dumps({})
    assert s.loads(result_empty) == {}

    # Test with list
    result_list = s.dumps([1, 2, 3])
    assert s.loads(result_list) == [1, 2, 3]

    # Test with nested structure
    nested = {"a": [1, {"b": "c"}], "d": None}
    result_nested = s.dumps(nested)
    assert s.loads(result_nested) == nested

    # Test with serializer_kwargs
    s_kwargs = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    result_kwargs = s_kwargs.dumps({"b": 1, "a": 2})
    assert b'"a"' in result_kwargs.encode() or '"a"' in result_kwargs

    # Test with multiple secret keys
    s_multi = Serializer(["old-key", "new-key"])
    result_multi = s_multi.dumps("test")
    assert s_multi.loads(result_multi) == "test"

    # Test return type matches serializer type
    assert isinstance(s.dumps("test"), str)
    assert isinstance(s_bytes.dumps("test"), bytes) 
```


# LLM-generated content at query #42
#--------------------------

```python
def test_Serializer_load_payload():
    serializer = Serializer("secret-key")
    
    # Test loading valid JSON payload as bytes
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading valid JSON payload with text serializer
    text_serializer = Serializer("secret-key", serializer=json)
    result = text_serializer.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading with custom serializer
    class CustomSerializer:
        def loads(self, payload):
            return {"custom": payload}
        
        def dumps(self, obj):
            return str(obj)
    
    custom_serializer = CustomSerializer()
    result = serializer.load_payload(b"test", serializer=custom_serializer)
    assert result == {"custom": b"test"}
    
    # Test BadPayload exception on invalid payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b"invalid json")
    
    # Test BadPayload exception on empty payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test BadPayload exception with original error
    try:
        serializer.load_payload(b"{invalid}")
    except BadPayload as e:
        assert e.original_error is not None
        assert isinstance(e.original_error, json.JSONDecodeError)
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            if not isinstance(payload, bytes):
                raise TypeError("Expected bytes")
            return payload.decode("utf-8")
        
        def dumps(self, obj):
            return obj.encode("utf-8")
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.load_payload(b"hello")
    assert result == "hello"
    
    # Test load_payload passes serializer_kwargs
    class KwargsSerializer:
        def __init__(self):
            self.called_with = None
        
        def loads(self, payload, **kwargs):
            self.called_with = kwargs
            return payload
        
        def dumps(self, obj, **kwargs):
            return obj
    
    kwargs_serializer = KwargsSerializer()
    test_serializer = Serializer("secret-key", serializer=kwargs_serializer, serializer_kwargs={"extra": "arg"})
    # load_payload doesn't use serializer_kwargs, so this tests the direct call
    result = test_serializer.load_payload(b"test", serializer=kwargs_serializer)
    assert result == b"test"
```


# LLM-generated content at query #43
#--------------------------

```python
def test_Serializer_dumps():
    """Test Serializer.dumps method with various scenarios."""
    
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert serializer.loads(result) == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert bytes_serializer.loads(result) == {"key": "value"}
    
    # Test with custom salt
    serializer_with_salt = Serializer("secret-key", salt="custom-salt")
    result1 = serializer_with_salt.dumps({"data": 1})
    result2 = Serializer("secret-key").dumps({"data": 1})
    assert result1 != result2
    
    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    assert result.endswith('{"a":1,"b":2}')
    
    # Test dumping and loading roundtrip
    original_data = {"nested": {"list": [1, 2, 3], "bool": True}, "number": 42}
    serialized = serializer.dumps(original_data)
    deserialized = serializer.loads(serialized)
    assert deserialized == original_data
    
    # Test with empty data
    empty_data = {}
    serialized_empty = serializer.dumps(empty_data)
    deserialized_empty = serializer.loads(serialized_empty)
    assert deserialized_empty == empty_data
    
    # Test with None value
    none_data = {"value": None}
    serialized_none = serializer.dumps(none_data)
    deserialized_none = serializer.loads(serialized_none)
    assert deserialized_none == none_data
    
    # Test with list data
    list_data = [1, "two", 3.0]
    serialized_list = serializer.dumps(list_data)
    deserialized_list = serializer.loads(serialized_list)
    assert deserialized_list == list_data
    
    # Test that dumps returns different values for different secret keys
    serializer1 = Serializer("secret1")
    serializer2 = Serializer("secret2")
    data = {"test": "data"}
    assert serializer1.dumps(data) != serializer2.dumps(data)
    
    # Test with key rotation (list of secret keys)
    multi_key_serializer = Serializer(["old_key", "new_key"])
    result_multi = multi_key_serializer.dumps({"data": "test"})
    assert multi_key_serializer.loads(result_multi) == {"data": "test"}
    
    # Verify the return type matches serializer type
    text_serializer = Serializer("key")
    assert isinstance(text_serializer.dumps("test"), str)
    
    class CustomTextSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return payload
    
    custom_text_serializer = Serializer("key", serializer=CustomTextSerializer())
    result_custom = custom_text_serializer.dumps("test_string")
    assert isinstance(result_custom, str)
    assert custom_text_serializer.loads(result_custom) == "test_string"


# LLM-generated content at query #44
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")

    class StrSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)

    # Test with bytes serializer
    bytes_serializer = BytesSerializer()
    assert isinstance(bytes_serializer.dumps({"key": "value"}), bytes)
    assert bytes_serializer.dumps({"key": "value"}) == b'{"key": "value"}'

    # Test with str serializer
    str_serializer = StrSerializer()
    assert isinstance(str_serializer.dumps({"key": "value"}), str)
    assert str_serializer.dumps({"key": "value"}) == '{"key": "value"}'

    # Test with json module directly
    json_serializer = json
    assert isinstance(json_serializer.dumps({"key": "value"}), str)
    assert json_serializer.dumps({"key": "value"}) == '{"key": "value"}'
```


# LLM-generated content at query #45
#--------------------------

```python
def test_Serializer_dumps():
    """Test Serializer.dumps method with various scenarios."""
    # Test 1: Basic dumps with default json serializer (text serializer)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
    # Verify it contains the signature (starts with payload followed by '.')
    assert "." in result
    payload_part = result.rsplit(".", 1)[0]
    # The payload should be base64 encoded json
    import base64
    decoded = base64.urlsafe_b64decode(payload_part + "==")
    assert json.loads(decoded) == {"key": "value"}

    # Test 2: Dumps with custom salt
    s = Serializer("secret-key")
    result1 = s.dumps("data", salt="custom-salt")
    result2 = s.dumps("data", salt="different-salt")
    assert result1 != result2  # Different salts should produce different signatures

    # Test 3: Dumps with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    s = Serializer("secret-key", serializer=BytesSerializer())
    result = s.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert b"." in result

    # Test 4: Dumps with serializer_kwargs
    s = Serializer("secret-key", serializer_kwargs={"sort_keys": True, "separators": (",", ":")})
    result = s.dumps({"b": 2, "a": 1})
    assert result is not None
    # Verify sort_keys was applied by checking payload contains "a" before "b"
    payload = result.rsplit(".", 1)[0]
    decoded = base64.urlsafe_b64decode(payload + "==").decode("utf-8")
    assert decoded.index('"a"') < decoded.index('"b"')

    # Test 5: Dumps with multiple keys (key rotation)
    s = Serializer(["old-key", "new-key"])
    result = s.dumps("test")
    # The signature should use the newest key
    # We can verify by trying to unsign with both keys
    signer = Signer(s.secret_keys[-1], salt=s.salt)
    signed_bytes = signer.sign(want_bytes(s.dump_payload("test")))
    if s.is_text_serializer:
        assert result == signed_bytes.decode("utf-8")
    else:
        assert result == signed_bytes

    # Test 6: Dumps with custom signer class
    class CustomSigner(Signer):
        pass
    
    s = Serializer("secret-key", signer=CustomSigner, signer_kwargs={"key_derivation": "none"})
    result = s.dumps("test")
    assert isinstance(result, (str, bytes))

    # Test 7: Dumps returns text when using text serializer
    s = Serializer("secret-key")
    result = s.dumps("test")
    assert isinstance(result, str)

    # Test 8: Dumps returns bytes when using bytes serializer
    class BytesSerializer2:
        def dumps(self, obj):
            if isinstance(obj, str):
                return obj.encode("utf-8")
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload)
    
    s = Serializer("secret-key", serializer=BytesSerializer2())
    result = s.dumps("test")
    assert isinstance(result, bytes)

    # Test 9: Dumps with no salt (None)
    s = Serializer("secret-key", salt=None)
    result = s.dumps("test")
    assert isinstance(result, (str, bytes))


# LLM-generated content at query #46
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation that matches the protocol
    class TestSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()

    serializer = TestSerializer()
    
    # Test with valid JSON bytes
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer
    result = serializer.loads(b'42')
    assert result == 42
    
    # Test with list
    result = serializer.loads(b'[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with string
    result = serializer.loads(b'"hello"')
    assert result == "hello"
    
    # Test with null
    result = serializer.loads(b'null')
    assert result is None
    
    # Test with boolean
    result = serializer.loads(b'true')
    assert result is True
    
    result = serializer.loads(b'false')
    assert result is False
    
    # Test with float
    result = serializer.loads(b'3.14')
    assert result == 3.14
    
    # Test that it raises appropriate exception for invalid input
    with pytest.raises(json.JSONDecodeError):
        serializer.loads(b'invalid json')

    # Test with str payload (since protocol allows str | bytes)
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)

    text_serializer = TextSerializer()
    result = text_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
```


# LLM-generated content at query #47
#--------------------------

```python
def test_Serializer_iter_unsigners():
    """Test iter_unsigners method of Serializer class."""
    # Test with default signer only
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    
    # Test with fallback signers as dict
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2  # default + fallback
    assert all(isinstance(s, Signer) for s in signers)
    
    # Test with fallback signers as tuple
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(Signer, {"key_derivation": "none"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert all(isinstance(s, Signer) for s in signers)
    
    # Test with fallback signers as Signer class
    serializer = Serializer(
        "secret-key",
        fallback_signers=[Signer]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert all(isinstance(s, Signer) for s in signers)
    
    # Test with multiple secret keys
    serializer = Serializer(
        ["old-secret", "new-secret"],
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    signers = list(serializer.iter_unsigners())
    # 1 default signer (uses all keys) + 2 fallback signers (one per key)
    assert len(signers) == 3
    
    # Test with custom salt
    serializer = Serializer("secret-key", salt=b"custom-salt")
    signers = list(serializer.iter_unsigners())
    signer = signers[0]
    assert signer.salt == b"custom-salt"
    
    # Test with override salt parameter
    serializer = Serializer("secret-key", salt=b"default-salt")
    signers = list(serializer.iter_unsigners(salt=b"override-salt"))
    signer = signers[0]
    assert signer.salt == b"override-salt"
    
    # Test with no fallback signers (empty list)
    serializer = Serializer("secret-key", fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    
    # Test with multiple fallback signers of different types
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"key_derivation": "hmac"},
            (Signer, {"key_derivation": "none"}),
            Signer
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4  # default + 3 fallbacks
    
    # Verify that signers are actually functional
    serializer = Serializer("secret-key")
    data = "test data"
    signed = serializer.make_signer().sign(data.encode())
    for signer in serializer.iter_unsigners():
        unsigned = signer.unsign(signed)
        assert unsigned == data.encode()
        break  # Just test the first one
```


# LLM-generated content at query #48
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Contains signature separator
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    # Test with salt parameter
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert result_with_salt != result  # Different salt produces different signature
    
    # Test that dumps returns a serialized payload with signature
    parts = result.rsplit(".", 1)
    assert len(parts) == 2
    payload, signature = parts
    assert len(signature) > 0  # Signature is present
    
    # Test with empty dict
    empty_result = serializer.dumps({})
    assert isinstance(empty_result, str)
    assert "." in empty_result
    
    # Test with list
    list_result = serializer.dumps([1, 2, 3])
    assert isinstance(list_result, str)
    assert "." in list_result
    
    # Test with simple string
    string_result = serializer.dumps("test")
    assert isinstance(string_result, str)
    assert "." in string_result
    
    # Test that serialized value can be loaded back
    from itsdangerous.signer import Signer
    signer = Signer("secret-key")
    unsigned = signer.unsign(result.encode("utf-8"))
    assert json.loads(unsigned) == {"key": "value"}
```


# LLM-generated content at query #49
#--------------------------

```python
def test_Serializer_iter_unsigners():
    """Test iter_unsigners returns the correct signers in order."""
    # Test 1: Basic case with no fallback signers
    serializer = Serializer("test-secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"test-secret-key"]

    # Test 2: With fallback signers as dict
    serializer = Serializer(
        "test-secret-key",
        fallback_signers=[{"key_derivation": "none"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].key_derivation == "none"

    # Test 3: With fallback signers as tuple
    class CustomSigner(Signer):
        pass
    
    serializer = Serializer(
        "test-secret-key",
        fallback_signers=[(CustomSigner, {"key_derivation": "hmac"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], CustomSigner)

    # Test 4: With fallback signers as Signer class
    serializer = Serializer(
        "test-secret-key",
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], CustomSigner)

    # Test 5: With custom salt
    serializer = Serializer("test-secret-key")
    signers = list(serializer.iter_unsigners(salt=b"custom-salt"))
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"

    # Test 6: With multiple secret keys (key rotation)
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-key", b"new-key"]

    # Test 7: With fallback signers and multiple secret keys
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"key_derivation": "none"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3  # 1 main signer + 2 fallback signers (one for each key)
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_keys == [b"old-key"]  # fallback uses individual keys
    assert signers[2].secret_keys == [b"new-key"]

    # Test 8: With salt=None (should use default salt from signer)
    serializer = Serializer("test-secret-key", salt=None)
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt is None or signers[0].salt == Signer.default_salt

    # Test 9: Verify that the first signer is always the main signer
    serializer = Serializer(
        "test-secret-key",
        signer_kwargs={"key_derivation": "none"},
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    signers = list(serializer.iter_unsigners())
    assert signers[0].key_derivation == "none"  # main signer uses signer_kwargs
    assert signers[1].key_derivation == "hmac"  # fallback signer uses its own kwargs
```


# LLM-generated content at query #50
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test loading valid payload
    result = serializer.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading with custom bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return b"test"
        
        def loads(self, payload):
            return {"loaded": payload}
    
    bytes_serializer = BytesSerializer()
    serializer_bytes = Serializer("secret-key", serializer=bytes_serializer)
    result = serializer_bytes.load_payload(b"some bytes")
    assert result == {"loaded": b"some bytes"}
    
    # Test with custom serializer passed as parameter
    class CustomSerializer:
        def dumps(self, obj):
            return "custom"
        
        def loads(self, payload):
            return {"custom": payload}
    
    custom_serializer = CustomSerializer()
    result = serializer.load_payload(b"test", serializer=custom_serializer)
    assert result == {"custom": "test"}
    
    # Test BadPayload raised on invalid json
    try:
        serializer.load_payload(b"invalid json")
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test BadPayload raised on invalid payload for non-json serializer
    class FailingSerializer:
        def dumps(self, obj):
            return "test"
        
        def loads(self, payload):
            raise ValueError("Cannot load")
    
    failing_serializer = FailingSerializer()
    serializer_fail = Serializer("secret-key", serializer=failing_serializer)
    try:
        serializer_fail.load_payload(b"test")
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test that original_error is set in BadPayload
    try:
        serializer.load_payload(b"not json")
    except BadPayload as e:
        assert e.original_error is not None
        assert isinstance(e.original_error, ValueError)
```


# LLM-generated content at query #51
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a concrete implementation of _PDataSerializer for testing
    class MockSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test loading valid JSON
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading a string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test loading a number
    result = serializer.loads('42')
    assert result == 42
    
    # Test loading null
    result = serializer.loads('null')
    assert result is None
    
    # Test loading boolean
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('false')
    assert result is False
```


# LLM-generated content at query #52
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("test-secret")
    
    # Test successful loading of valid payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return {"loaded": payload}
        def dumps(self, obj: t.Any) -> bytes:
            return b"test"
    
    bytes_serializer = Serializer("test-secret", serializer=BytesSerializer())
    payload = b"test-payload"
    result = bytes_serializer.load_payload(payload)
    assert result == {"loaded": b"test-payload"}
    
    # Test with custom serializer passed as parameter
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"custom": payload}
        def dumps(self, obj: t.Any) -> str:
            return "test"
    
    serializer = Serializer("test-secret")
    payload = b'{"test": "data"}'
    result = serializer.load_payload(payload, serializer=CustomSerializer())
    assert result == {"custom": '{"test": "data"}'}
    
    # Test with invalid JSON payload
    from itsdangerous.exc import BadPayload
    try:
        serializer.load_payload(b"invalid json")
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None
    
    # Test with empty payload
    try:
        serializer.load_payload(b"")
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test with text serializer and non-UTF-8 bytes
    try:
        serializer.load_payload(b"\xff\xfe")
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
```


# LLM-generated content at query #53
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Create a mock serializer that returns a specific value
    class MockSerializer:
        def dumps(self, obj):
            return "serialized_data"
        
        def loads(self, payload):
            return {"key": "value"}
    
    serializer = MockSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == "serialized_data"
```


# LLM-generated content at query #54
#--------------------------

```python
def test__PDataSerializer_loads():
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
    
    # Test loading a simple string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test loading a number
    result = serializer.loads('42')
    assert result == 42
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading None
    result = serializer.loads('null')
    assert result is None
    
    # Test loading boolean
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('false')
    assert result is False
    
    # Test with bytes serializer
    class MockBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = MockBytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
```


# LLM-generated content at query #55
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    serializer = BytesSerializer()
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}
    
    # Test with a text serializer
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    text_serializer = TextSerializer()
    text_payload = '{"key": "value"}'
    text_result = text_serializer.loads(text_payload)
    assert text_result == {"key": "value"}
    
    # Test that it works with the protocol type hint
    typed_serializer: _PDataSerializer[bytes] = BytesSerializer()
    typed_result = typed_serializer.loads(b'[1, 2, 3]')
    assert typed_result == [1, 2, 3]
    
    # Test with empty payload
    empty_result = serializer.loads(b'{}')
    assert empty_result == {}
    
    # Test with nested structures
    nested_payload = b'{"a": [1, 2, {"b": 3}]}'
    nested_result = serializer.loads(nested_payload)
    assert nested_result == {"a": [1, 2, {"b": 3}]}
```


# LLM-generated content at query #56
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a serializer that returns str (JSON-like)
    class StrSerializer:
        def loads(self, payload: str):
            if payload == '{"key": "value"}':
                return {"key": "value"}
            raise ValueError("Invalid payload")
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = StrSerializer()
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes):
            if payload == b'{"key": "value"}':
                return {"key": "value"}
            raise ValueError("Invalid payload")
        
        def dumps(self, obj):
            return json.dumps(obj).encode()
    
    serializer = BytesSerializer()
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that loads raises exception for invalid payload
    import pytest
    with pytest.raises(ValueError):
        serializer = StrSerializer()
        serializer.loads("invalid payload")


# LLM-generated content at query #57
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns a dict
    class TestSerializer:
        def loads(self, payload: str) -> dict:
            return {"data": payload}
        
        def dumps(self, obj: dict) -> str:
            return str(obj)
    
    serializer = TestSerializer()
    result = serializer.loads("test_payload")
    assert result == {"data": "test_payload"}
    
    # Test with integer payload
    class IntSerializer:
        def loads(self, payload: str) -> int:
            return int(payload)
        
        def dumps(self, obj: int) -> str:
            return str(obj)
    
    int_serializer = IntSerializer()
    result = int_serializer.loads("42")
    assert result == 42
    
    # Test with list payload
    class ListSerializer:
        def loads(self, payload: str) -> list:
            return list(payload)
        
        def dumps(self, obj: list) -> str:
            return "".join(obj)
    
    list_serializer = ListSerializer()
    result = list_serializer.loads("abc")
    assert result == ["a", "b", "c"]
    
    # Test with bytes payload
    class BytesSerializer:
        def loads(self, payload: bytes) -> str:
            return payload.decode("utf-8")
        
        def dumps(self, obj: str) -> bytes:
            return obj.encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b"hello")
    assert result == "hello"
    
    # Test with None payload
    class NoneSerializer:
        def loads(self, payload: str) -> None:
            return None
        
        def dumps(self, obj: None) -> str:
            return "null"
    
    none_serializer = NoneSerializer()
    result = none_serializer.loads("null")
    assert result is None
```


# LLM-generated content at query #58
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol supports loads method"""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer
    result = serializer.loads('42')
    assert result == 42
    
    # Test with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test with null
    result = serializer.loads('null')
    assert result is None
    
    # Test with boolean
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('false')
    assert result is False
    
    # Test with empty JSON
    result = serializer.loads('{}')
    assert result == {}
    
    # Test with nested structure
    result = serializer.loads('{"a": {"b": [1, 2, 3]}}')
    assert result == {"a": {"b": [1, 2, 3]}}
```


# LLM-generated content at query #59
#--------------------------

```python
def test__PDataSerializer_dumps():
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
    
    # Test with a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with empty dict
    result = serializer.dumps({})
    assert result == "{}"
    
    # Test with None
    result = serializer.dumps(None)
    assert result == "null"
    
    # Test with list
    result = serializer.dumps([1, 2, 3])
    assert result == "[1, 2, 3]"
```


# LLM-generated content at query #60
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer.loads works correctly with different payload types."""
    # Test with string payload
    serializer_str = type("TestSerializer", (), {
        "loads": lambda self, payload: json.loads(payload),
        "dumps": lambda self, obj: json.dumps(obj)
    })()
    
    result = serializer_str.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes payload
    serializer_bytes = type("TestSerializer", (), {
        "loads": lambda self, payload: json.loads(payload.decode()),
        "dumps": lambda self, obj: json.dumps(obj).encode()
    })()
    
    result = serializer_bytes.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer payload
    serializer_int = type("TestSerializer", (), {
        "loads": lambda self, payload: int(payload),
        "dumps": lambda self, obj: str(obj)
    })()
    
    result = serializer_int.loads("42")
    assert result == 42
    
    # Test with list payload
    result = serializer_str.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with None payload
    result = serializer_str.loads('null')
    assert result is None
```


# LLM-generated content at query #61
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"key": "value"}
        
        def dumps(self, obj: t.Any) -> str:
            return '{"key": "value"}'
    
    serializer = MockSerializer()
    
    # Test that loads returns the expected deserialized data
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with different payload
    result = serializer.loads('{"number": 42}')
    assert result == {"number": 42}
    
    # Test with empty payload
    result = serializer.loads("{}")
    assert result == {}
    
    # Test with simple string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test with integer
    result = serializer.loads("123")
    assert result == 123
```


# LLM-generated content at query #62
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test loading valid JSON payload
    data = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(data)
    assert result == {"key": "value"}
    
    # Test loading payload with custom serializer
    class CustomSerializer:
        def loads(self, payload):
            return {"custom": payload.decode()}
        
        def dumps(self, obj):
            return b"test"
    
    custom_serializer = CustomSerializer()
    custom_data = b"hello"
    result = serializer.load_payload(custom_data, serializer=custom_serializer)
    assert result == {"custom": "hello"}
    
    # Test loading invalid payload raises BadPayload
    try:
        serializer.load_payload(b"invalid json")
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test loading bytes payload with text serializer
    valid_json_bytes = b'{"test": 123}'
    result = serializer.load_payload(valid_json_bytes)
    assert result == {"test": 123}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return bytes(obj)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    bytes_data = bytes_serializer.dump_payload(b"raw bytes")
    result = bytes_serializer.load_payload(bytes_data)
    assert result == b"raw bytes"
    
    # Test BadPayload includes original error
    try:
        serializer.load_payload(b"not json")
    except BadPayload as e:
        assert e.original_error is not None
        assert isinstance(e.original_error, json.JSONDecodeError)
```


# LLM-generated content at query #63
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with valid JSON payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer payload
    result = serializer.loads('42')
    assert result == 42
    
    # Test with list payload
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with string payload
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test with boolean payload
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('false')
    assert result is False
    
    # Test with null payload
    result = serializer.loads('null')
    assert result is None
    
    # Test with empty payload
    result = serializer.loads('{}')
    assert result == {}
    
    result = serializer.loads('[]')
    assert result == []```


# LLM-generated content at query #64
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a text serializer (e.g., json)
    serializer = json
    data = '{"key": "value"}'
    result = serializer.loads(data)
    assert result == {"key": "value"}
    
    # Test with a custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    bytes_data = b'{"key": "value"}'
    result = bytes_serializer.loads(bytes_data)
    assert result == {"key": "value"}
    
    # Test with a custom serializer that returns str
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    text_serializer = TextSerializer()
    text_data = '{"key": "value"}'
    result = text_serializer.loads(text_data)
    assert result == {"key": "value"}
    
    # Test that it handles complex nested data
    complex_data = '{"numbers": [1, 2, 3], "nested": {"a": 1}}'
    result = serializer.loads(complex_data)
    assert result == {"numbers": [1, 2, 3], "nested": {"a": 1}}
    
    # Test with empty data
    empty_data = '{}'
    result = serializer.loads(empty_data)
    assert result == {}
    
    # Test with list data
    list_data = '[1, 2, 3]'
    result = serializer.loads(list_data)
    assert result == [1, 2, 3]
```


# LLM-generated content at query #65
#--------------------------

```python
def test_Serializer_load_payload():
    # Create a serializer with JSON serializer (text serializer)
    serializer = Serializer("test-secret-key")
    
    # Test with valid JSON payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with valid payload using custom serializer parameter
    custom_serializer = json
    result = serializer.load_payload(payload, serializer=custom_serializer)
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return {"data": payload}
        
        def dumps(self, obj):
            return b"test"
    
    bytes_serializer = Serializer("test-secret-key", salt=b"salt", serializer=BytesSerializer())
    
    # Test with bytes payload using bytes serializer
    bytes_payload = b"test_bytes"
    result = bytes_serializer.load_payload(bytes_payload)
    assert result == {"data": b"test_bytes"}
    
    # Test with invalid payload (raises BadPayload)
    import json
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"invalid json")
    assert "Could not load the payload" in str(exc_info.value)
    
    # Test with empty bytes payload (raises BadPayload for JSON)
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test with None payload (raises BadPayload)
    with pytest.raises(BadPayload):
        serializer.load_payload(b"null")
    
    # Test with integer payload (valid JSON)
    result = serializer.load_payload(b"42")
    assert result == 42
    
    # Test with list payload (valid JSON)
    result = serializer.load_payload(b'[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with boolean payload (valid JSON)
    result = serializer.load_payload(b"true")
    assert result is True
    
    # Test with null payload (valid JSON)
    result = serializer.load_payload(b"null")
    assert result is None
    
    # Test with string payload (valid JSON)
    result = serializer.load_payload(b'"hello"')
    assert result == "hello"
```


# LLM-generated content at query #66
#--------------------------

```python
def test_Serializer_load_payload():
    """Test load_payload method with various scenarios."""
    serializer = Serializer("test-secret-key")
    
    # Test loading valid JSON payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading valid JSON payload as text
    text_serializer = Serializer("test-secret-key", serializer=json)
    payload_text = b'{"key": "value"}'
    result = text_serializer.load_payload(payload_text)
    assert result == {"key": "value"}
    
    # Test with custom serializer
    class CustomSerializer:
        def loads(self, payload):
            return {"custom": payload}
        
        def dumps(self, obj):
            return "custom"
    
    custom_serializer = Serializer("test-secret-key", serializer=CustomSerializer())
    payload = b"test_data"
    result = custom_serializer.load_payload(payload)
    assert result == {"custom": b"test_data"}
    
    # Test with custom serializer passed as parameter
    class AnotherSerializer:
        def loads(self, payload):
            return {"from_param": payload}
        
        def dumps(self, obj):
            return "another"
    
    result = serializer.load_payload(b"test", serializer=AnotherSerializer())
    assert result == {"from_param": b"test"}
    
    # Test with invalid payload (not JSON)
    try:
        serializer.load_payload(b"invalid json")
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test with empty payload
    try:
        serializer.load_payload(b"")
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            if isinstance(payload, bytes):
                return {"bytes": payload}
            raise TypeError("Expected bytes")
        
        def dumps(self, obj):
            return b"bytes"
    
    bytes_serializer = Serializer("test-secret-key", serializer=BytesSerializer())
    payload = b"test_bytes"
    result = bytes_serializer.load_payload(payload)
    assert result == {"bytes": b"test_bytes"}
    
    # Test BadPayload exception contains original error
    try:
        serializer.load_payload(b"not json")
    except BadPayload as e:
        assert e.original_error is not None
        assert isinstance(e.original_error, json.JSONDecodeError)
    
    # Test with text serializer that returns string
    class TextSerializer:
        def loads(self, payload):
            if isinstance(payload, str):
                return {"text": payload}
            raise TypeError("Expected string")
        
        def dumps(self, obj):
            return "text"
    
    # This serializer returns string, so is_text_serializer should be True
    text_serializer = Serializer("test-secret-key", serializer=TextSerializer())
    # Payload needs to be bytes, but serializer expects string, so it will decode
    payload = b"hello"
    result = text_serializer.load_payload(payload)
    assert result == {"text": "hello"}  # Actually {"text": "hello"} since payload is decoded
    
    # Test BinarySerializer that expects bytes
    class BinarySerializer:
        def loads(self, payload):
            return {"binary": payload}
        
        def dumps(self, obj):
            return b"binary"
    
    binary_serializer = Serializer("test-secret-key", serializer=BinarySerializer())
    payload = b"binary_data"
    result = binary_serializer.load_payload(payload)
    assert result == {"binary": b"binary_data"}
    
    # Test with exception in loads
    class FailingSerializer:
        def loads(self, payload):
            raise ValueError("Custom error")
        
        def dumps(self, obj):
            return "fail"
    
    failing_serializer = Serializer("test-secret-key", serializer=FailingSerializer())
    try:
        failing_serializer.load_payload(b"test")
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert str(e.original_error) == "Custom error"
```


# LLM-generated content at query #67
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test that _PDataSerializer protocol is properly defined
    # Since it's a protocol, we can test that objects conforming to it work correctly
    
    class TestSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = TestSerializer()
    test_data = {"key": "value", "number": 42}
    
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    assert result == '{"key": "value", "number": 42}'
    
    # Test roundtrip
    loaded = serializer.loads(result)
    assert loaded == test_data
    
    # Test with different data types
    test_list = [1, 2, 3]
    result = serializer.dumps(test_list)
    assert result == '[1, 2, 3]'
    
    test_string = "hello"
    result = serializer.dumps(test_string)
    assert result == '"hello"'
```


# LLM-generated content at query #68
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return b'{"key": "value"}'
        def loads(self, payload):
            return json.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    
    # Test with custom salt
    serializer = Serializer("secret-key")
    result1 = serializer.dumps({"key": "value"}, salt="custom-salt")
    result2 = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert result1 == result2
    
    # Test different data produces different signatures
    serializer = Serializer("secret-key")
    result1 = serializer.dumps({"key": "value1"})
    result2 = serializer.dumps({"key": "value2"})
    assert result1 != result2
    
    # Verify that dumps output can be loaded back
    serializer = Serializer("secret-key")
    original = {"test": "data", "number": 42}
    dumped = serializer.dumps(original)
    loaded = serializer.loads(dumped)
    assert loaded == original
    
    # Test with serializer_kwargs
    class CustomSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj, **kwargs)
        def loads(self, payload):
            return json.loads(payload)
    
    serializer = Serializer("secret-key", serializer=CustomSerializer(), serializer_kwargs={"indent": 2})
    result = serializer.dumps({"key": "value"})
    assert "\n" in result  # indented JSON contains newlines
    
    # Test with empty data
    serializer = Serializer("secret-key")
    result = serializer.dumps({})
    assert isinstance(result, str)
    
    # Test with list data
    serializer = Serializer("secret-key")
    result = serializer.dumps([1, 2, 3])
    assert isinstance(result, str)


# LLM-generated content at query #69
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol defines loads method correctly."""
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
    
    # Test with list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test that it raises appropriate exception for invalid input
    with pytest.raises(json.JSONDecodeError):
        serializer.loads('invalid json')
```


# LLM-generated content at query #70
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Test with a text serializer
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    text_serializer = TextSerializer()
    assert text_serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert isinstance(text_serializer.dumps({}), str)
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.dumps({"key": "value"}) == b'{"key": "value"}'
    assert isinstance(bytes_serializer.dumps({}), bytes)
    
    # Test that the protocol is satisfied by different return types
    assert is_text_serializer(text_serializer) == True
    assert is_text_serializer(bytes_serializer) == False
```


# LLM-generated content at query #71
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test _PDataSerializer loads method protocol conformance."""
    
    # Test with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    string_serializer = StringSerializer()
    assert string_serializer.loads("hello") == "HELLO"
    assert isinstance(string_serializer.loads("test"), str)
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode("utf-8").upper()
        
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b"hello") == "HELLO"
    assert isinstance(bytes_serializer.loads(b"test"), str)
    
    # Test with a serializer that returns different types
    class MixedSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    mixed_serializer = MixedSerializer()
    result = mixed_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    assert isinstance(result, dict)
    
    # Test that loads accepts only the payload argument positionally
    class PositionalOnlySerializer:
        def loads(self, payload: str, /) -> t.Any:
            return payload
        
    positional_serializer = PositionalOnlySerializer()
    assert positional_serializer.loads("test") == "test"
    
    # Verify type conformance by checking dumps returns the expected type
    assert is_text_serializer(StringSerializer()) == True
    assert is_text_serializer(BytesSerializer()) == False
    assert is_text_serializer(MixedSerializer()) == True
```


# LLM-generated content at query #72
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Should contain signature separator
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode()
        def loads(self, payload):
            return json.loads(payload.decode())
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert b"." in result
    
    # Test dumps with salt parameter
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert result != result_with_salt  # Different salt should produce different result
    
    # Test that dumps produces a verifiable result
    signed = serializer.dumps("test data")
    loaded = serializer.loads(signed)
    assert loaded == "test data"
    
    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_with_kwargs = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    assert isinstance(result_with_kwargs, str)
    # Should be compact JSON with sorted keys
    assert '"a":1' in result_with_kwargs
    assert '"b":2' in result_with_kwargs
    assert " " not in result_with_kwargs  # No spaces in compact JSON
    
    # Test with non-dict payload
    result_list = serializer.dumps([1, 2, 3])
    loaded_list = serializer.loads(result_list)
    assert loaded_list == [1, 2, 3]
    
    result_string = serializer.dumps("simple string")
    loaded_string = serializer.loads(result_string)
    assert loaded_string == "simple string"
    
    result_none = serializer.dumps(None)
    loaded_none = serializer.loads(result_none)
    assert loaded_none is None
    
    # Test with integer payload
    result_int = serializer.dumps(42)
    loaded_int = serializer.loads(result_int)
    assert loaded_int == 42
    
    # Test with boolean payload
    result_bool = serializer.dumps(True)
    loaded_bool = serializer.loads(result_bool)
    assert loaded_bool is True
    
    # Test that dumps returns consistent type for same serializer
    text_serializer = Serializer("secret-key")
    assert isinstance(text_serializer.dumps("test"), str)
    assert isinstance(text_serializer.dumps(123), str)
    assert isinstance(text_serializer.dumps(None), str)
    
    # Test with key rotation (multiple secret keys)
    multi_key_serializer = Serializer(["old-key", "new-key"])
    signed_multi = multi_key_serializer.dumps("test")
    # Should be verifiable with any key in the list
    loaded_multi = multi_key_serializer.loads(signed_multi)
    assert loaded_multi == "test"
```


# LLM-generated content at query #73
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test with a basic serializer that returns str
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TextSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    serializer = BytesSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with various data types
    serializer = TextSerializer()
    assert serializer.dumps(None) == 'null'
    assert serializer.dumps(123) == '123'
    assert serializer.dumps([1, 2, 3]) == '[1, 2, 3]'
    assert serializer.dumps({"a": 1, "b": 2}) == '{"a": 1, "b": 2}'```


# LLM-generated content at query #74
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test loads method of _PDataSerializer protocol."""
    # Test with a JSON serializer
    json_serializer = json
    serializer = _PDataSerializer[json_serializer.dumps({}).__class__]()
    
    # Test loading JSON string
    payload_str = '{"key": "value"}'
    result = serializer.loads(payload_str)
    assert result == {"key": "value"}
    
    # Test loading JSON bytes
    payload_bytes = b'{"key": "value"}'
    result = serializer.loads(payload_bytes)
    assert result == {"key": "value"}
    
    # Test loading with different data types
    payload_int = "42"
    result = serializer.loads(payload_int)
    assert result == 42
    
    payload_list = '["a", "b", "c"]'
    result = serializer.loads(payload_list)
    assert result == ["a", "b", "c"]
    
    payload_none = "null"
    result = serializer.loads(payload_none)
    assert result is None
    
    # Test loading invalid JSON
    invalid_payload = "{invalid: json}"
    try:
        serializer.loads(invalid_payload)
        assert False, "Should have raised an exception"
    except Exception:
        pass
    
    # Test loading empty payload
    empty_payload = ""
    try:
        serializer.loads(empty_payload)
        assert False, "Should have raised an exception"
    except Exception:
        pass
    
    # Test with a custom serializer that returns bytes
    class CustomBytesSerializer:
        def loads(self, payload):
            if isinstance(payload, str):
                payload = payload.encode('utf-8')
            return payload.decode('utf-8').upper()
    
    bytes_serializer = CustomBytesSerializer()
    result = bytes_serializer.loads("hello")
    assert result == "HELLO"
    
    # Test with a custom serializer that returns int
    class CustomIntSerializer:
        def loads(self, payload):
            return len(payload)
    
    int_serializer = CustomIntSerializer()
    result = int_serializer.loads("test")
    assert result == 4
    
    # Test that loads method accepts both str and bytes
    str_serializer = _PDataSerializer[str]()
    str_serializer.loads("test")  # Should work with str
    
    bytes_serializer = _PDataSerializer[bytes]()
    bytes_serializer.loads(b"test")  # Should work with bytes
```


# LLM-generated content at query #75
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol loads method works correctly."""
    # Create a concrete implementation that follows the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload: str) -> dict:
            if payload == '{"key": "value"}':
                return {"key": "value"}
            raise ValueError("Invalid payload")
        
        def dumps(self, obj: dict) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test successful load
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that it raises appropriate exception for invalid payload
    with pytest.raises(ValueError, match="Invalid payload"):
        serializer.loads("invalid json")
```


# LLM-generated content at query #76
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer's loads method protocol is properly implemented."""
    # Create a simple serializer that implements the _PDataSerializer protocol
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode("utf-8").upper()
        
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode("utf-8")
    
    # Test with string serializer
    str_serializer = StringSerializer()
    result = str_serializer.loads("hello")
    assert result == "HELLO"
    
    # Test with bytes serializer
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b"hello")
    assert result == "HELLO"
    
    # Test that the protocol works with different return types
    class DictSerializer:
        def loads(self, payload: str) -> t.Any:
            import json
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            import json
            return json.dumps(obj)
    
    dict_serializer = DictSerializer()
    result = dict_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that loads can handle various input types
    class FlexibleSerializer:
        def loads(self, payload: str) -> t.Any:
            if payload.startswith("int:"):
                return int(payload[4:])
            elif payload.startswith("float:"):
                return float(payload[6:])
            elif payload.startswith("list:"):
                return payload[5:].split(",")
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    flexible = FlexibleSerializer()
    assert flexible.loads("int:42") == 42
    assert flexible.loads("float:3.14") == 3.14
    assert flexible.loads("list:a,b,c") == ["a", "b", "c"]
    assert flexible.loads("plain string") == "plain string"
```


# LLM-generated content at query #77
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that handles bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode("utf-8")
        
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode("utf-8")
    
    serializer = BytesSerializer()
    result = serializer.loads(b"test payload")
    assert result == "test payload"
    
    # Test with a JSON serializer
    json_serializer = json
    result = json_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with a custom serializer that returns complex types
    class ComplexSerializer:
        def loads(self, payload: bytes) -> t.Any:
            data = payload.decode("utf-8")
            parts = data.split(",")
            return {"first": parts[0], "second": parts[1]}
        
        def dumps(self, obj: t.Any) -> bytes:
            return f"{obj['first']},{obj['second']}".encode("utf-8")
    
    complex_serializer = ComplexSerializer()
    result = complex_serializer.loads(b"hello,world")
    assert result == {"first": "hello", "second": "world"}
```


# LLM-generated content at query #78
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
        def loads(self, payload):
            return {"key": "value"}
        def dumps(self, obj):
            return b'{"key": "value"}'
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with custom serializer parameter
    class CustomSerializer:
        def loads(self, payload):
            return {"custom": True}
        def dumps(self, obj):
            return '{"custom": true}'
    
    custom_serializer = CustomSerializer()
    result = serializer.load_payload(b'{}', serializer=custom_serializer)
    assert result == {"custom": True}
    
    # Test BadPayload exception with invalid data
    import json
    try:
        serializer.load_payload(b'invalid json')
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None
    
    # Test BadPayload exception with bytes serializer and invalid data
    class FailingSerializer:
        def loads(self, payload):
            raise ValueError("Failed to load")
        def dumps(self, obj):
            return b'test'
    
    failing_serializer = Serializer("secret-key", serializer=FailingSerializer())
    try:
        failing_serializer.load_payload(b'test')
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)
    
    # Test with empty payload
    try:
        serializer.load_payload(b'')
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test with None payload
    try:
        serializer.load_payload(None)
        assert False, "Should have raised BadPayload"
    except (BadPayload, TypeError):
        pass
```


# LLM-generated content at query #79
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StringSerializer()
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
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
    
    # Test with list payload
    class ListSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    list_serializer = ListSerializer()
    result = list_serializer.loads("[1, 2, 3]")
    assert result == [1, 2, 3]
    
    # Test with None payload
    class NoneSerializer:
        def loads(self, payload: str) -> t.Any:
            return None if payload == "null" else json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return "null" if obj is None else json.dumps(obj)
    
    none_serializer = NoneSerializer()
    result = none_serializer.loads("null")
    assert result is None
```


# LLM-generated content at query #80
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
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert bytes_serializer.loads(result_bytes) == {"key": "value"}

    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt=b"custom-salt")
    assert isinstance(result_with_salt, str)
    assert serializer.loads(result_with_salt, salt=b"custom-salt") == {"key": "value"}

    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True}
    )
    result_kwargs = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    assert isinstance(result_kwargs, str)
    assert serializer_with_kwargs.loads(result_kwargs) == {"a": 1, "b": 2}

    # Test with multiple secret keys (key rotation)
    serializer_rotation = Serializer(["old-key", "new-key"])
    result_rotation = serializer_rotation.dumps({"data": "test"})
    assert isinstance(result_rotation, str)
    assert serializer_rotation.loads(result_rotation) == {"data": "test"}
    assert serializer_rotation.secret_key == b"new-key"

    # Test with empty dict payload
    result_empty = serializer.dumps({})
    assert isinstance(result_empty, str)
    assert serializer.loads(result_empty) == {}

    # Test with list payload
    result_list = serializer.dumps([1, 2, 3])
    assert isinstance(result_list, str)
    assert serializer.loads(result_list) == [1, 2, 3]
```


# LLM-generated content at query #81
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a valid payload
    serializer = json
    data_serializer = _PDataSerializer()
    # Since _PDataSerializer is a protocol, we test through a concrete implementation
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}

    # Test with bytes payload
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}

    # Test with invalid JSON
    with pytest.raises(json.JSONDecodeError):
        serializer.loads("invalid json")

    # Test with empty object
    result = serializer.loads("{}")
    assert result == {}

    # Test with array
    result = serializer.loads("[1, 2, 3]")
    assert result == [1, 2, 3]

    # Test with null
    result = serializer.loads("null")
    assert result is None

    # Test with boolean
    result = serializer.loads("true")
    assert result is True
    result = serializer.loads("false")
    assert result is False

    # Test with number
    result = serializer.loads("42")
    assert result == 42
    result = serializer.loads("3.14")
    assert result == 3.14

    # Test with string
    result = serializer.loads('"hello"')
    assert result == "hello"

    # Test with unicode
    result = serializer.loads('"\\u0048\\u0065\\u006c\\u006c\\u006f"')
    assert result == "Hello"

    # Test with nested structures
    result = serializer.loads('{"a": {"b": 1}, "c": [2, 3]}')
    assert result == {"a": {"b": 1}, "c": [2, 3]}

    # Test with special float values
    result = serializer.loads("Infinity")
    assert result == float('inf')
    result = serializer.loads("-Infinity")
    assert result == float('-inf')
    result = serializer.loads("NaN")
    assert result != result  # NaN is not equal to itself

    # Test with empty string
    with pytest.raises(json.JSONDecodeError):
        serializer.loads("")

    # Test with whitespace
    result = serializer.loads("  42  ")
    assert result == 42

    # Test with custom object hook
    class CustomDecoder(json.JSONDecoder):
        def decode(self, s, **kwargs):
            return super().decode(s, **kwargs)

    custom_serializer = CustomDecoder()
    result = custom_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
```


# LLM-generated content at query #82
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default signer and no fallback signers
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    
    # Test with salt parameter
    salt = b"custom-salt"
    signers = list(serializer.iter_unsigners(salt))
    assert len(signers) == 1
    assert signers[0].salt == salt
    
    # Test with fallback signers as dict
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2  # default + fallback
    
    # Test with fallback signers as tuple
    from itsdangerous.signer import Signer as SignerClass
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(SignerClass, {"digest_method": "sha256"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    
    # Test with fallback signers as Signer class
    serializer = Serializer(
        "secret-key",
        fallback_signers=[SignerClass]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    
    # Test with multiple secret keys
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1  # default signer uses all keys
    
    # Test with multiple secret keys and fallback signers
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3  # 1 default + 2 fallbacks (one per secret key)
    
    # Verify signer properties
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert signers[0].salt == b"itsdangerous"
    assert isinstance(signers[0].secret_key, bytes)
```


# LLM-generated content at query #83
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    import json
    import pickle
    
    # Test with JSON serializer (text-based)
    json_serializer = json
    test_data = {"key": "value", "number": 42}
    serialized = json_serializer.dumps(test_data)
    result = json_serializer.loads(serialized)
    assert result == test_data
    
    # Test with pickle serializer (binary-based)
    pickle_serializer = pickle
    test_data_bytes = {"key": "value", "number": 42}
    serialized_bytes = pickle_serializer.dumps(test_data_bytes)
    result_bytes = pickle_serializer.loads(serialized_bytes)
    assert result_bytes == test_data_bytes
    
    # Test with simple string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    string_serializer = StringSerializer()
    result_str = string_serializer.loads("test_string")
    assert result_str == "test_string"
    
    # Test that loads returns Any type
    result_any = json_serializer.loads('{"test": 123}')
    assert isinstance(result_any, dict)
    assert result_any == {"test": 123}
    
    # Test with invalid payload
    with pytest.raises(json.JSONDecodeError):
        json_serializer.loads("invalid json")
```


# LLM-generated content at query #84
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a simple serializer that implements the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert result == '{"key": "value"}'
    assert isinstance(result, str)


# LLM-generated content at query #85
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol is properly duck-typed for loads"""
    
    # Create a mock serializer that implements loads and dumps
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"key": "value"}
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    # Verify it works with the protocol
    serializer: _PDataSerializer[str] = MockSerializer()
    
    # Test basic loads functionality
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with different payload types
    assert serializer.loads('{"number": 42}') == {"number": 42}
    assert serializer.loads('{"nested": {"inner": "data"}}') == {"nested": {"inner": "data"}}
    
    # Test that loads is called with the correct argument
    class TrackingSerializer:
        def __init__(self):
            self.last_payload = None
            
        def loads(self, payload: str) -> t.Any:
            self.last_payload = payload
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    tracker = TrackingSerializer()
    tracker.loads('{"test": true}')
    assert tracker.last_payload == '{"test": true}'
    
    # Test that it works with bytes serializer as well
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer: _PDataSerializer[bytes] = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    
    # Test that loads raises appropriate errors
    class ErrorSerializer:
        def loads(self, payload: str) -> t.Any:
            raise ValueError("Invalid data")
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    error_serializer = ErrorSerializer()
    import pytest
    with pytest.raises(ValueError, match="Invalid data"):
        error_serializer.loads("invalid")
```


# LLM-generated content at query #86
#--------------------------

```python
def test_Serializer_load_payload():
    """Test load_payload method with various scenarios."""
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("test-secret")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes):
            return {"key": "value"}
        def dumps(self, obj):
            return b'{"key": "value"}'

    bytes_serializer = Serializer("test-secret", serializer=BytesSerializer())
    payload = b'{"key": "value"}'
    result = bytes_serializer.load_payload(payload)
    assert result == {"key": "value"}

    # Test with custom serializer passed as parameter
    class CustomSerializer:
        def loads(self, payload: str):
            return {"custom": "data"}
        def dumps(self, obj):
            return '{"custom": "data"}'

    serializer = Serializer("test-secret")
    custom_ser = CustomSerializer()
    payload = b'{"custom": "data"}'
    result = serializer.load_payload(payload, serializer=custom_ser)
    assert result == {"custom": "data"}

    # Test with invalid payload that raises BadPayload
    serializer = Serializer("test-secret")
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None

    # Test with empty payload
    serializer = Serializer("test-secret")
    payload = b""
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass

    # Test with None payload
    serializer = Serializer("test-secret")
    try:
        serializer.load_payload(None)  # type: ignore
        assert False, "Expected BadPayload exception"
    except (BadPayload, TypeError):
        pass

    # Test with text serializer that returns complex data
    serializer = Serializer("test-secret")
    payload = b'{"list": [1, 2, 3], "nested": {"a": 1}}'
    result = serializer.load_payload(payload)
    assert result == {"list": [1, 2, 3], "nested": {"a": 1}}

    # Test with bytes containing UTF-8 encoded text
    serializer = Serializer("test-secret")
    payload = '{"message": "héllo"}'.encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"message": "héllo"}

    # Test with custom serializer parameter that is a bytes serializer
    class BytesOnlySerializer:
        def loads(self, payload: bytes):
            return payload
        def dumps(self, obj):
            return obj

    serializer = Serializer("test-secret")
    payload = b"raw bytes data"
    result = serializer.load_payload(payload, serializer=BytesOnlySerializer())
    assert result == b"raw bytes data"

    # Test with serializer that raises exception during loads
    class FailingSerializer:
        def loads(self, payload):
            raise ValueError("Serialization failed")
        def dumps(self, obj):
            return "{}"

    serializer = Serializer("test-secret")
    payload = b"{}"
    try:
        serializer.load_payload(payload, serializer=FailingSerializer())
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)

    # Test with serializer that returns non-dict data
    serializer = Serializer("test-secret")
    payload = b'[1, 2, 3]'
    result = serializer.load_payload(payload)
    assert result == [1, 2, 3]

    # Test with serializer that returns primitive types
    serializer = Serializer("test-secret")
    payload = b'"string"'
    result = serializer.load_payload(payload)
    assert result == "string"

    payload = b'42'
    result = serializer.load_payload(payload)
    assert result == 42

    payload = b'true'
    result = serializer.load_payload(payload)
    assert result is True

    payload = b'null'
    result = serializer.load_payload(payload)
    assert result is None
```


# LLM-generated content at query #87
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a simple serializer that implements the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    test_data = {"key": "value", "number": 42}
    
    result = serializer.dumps(test_data)
    
    assert isinstance(result, str)
    assert result == '{"key": "value", "number": 42}'


# LLM-generated content at query #88
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
            return json.dumps(obj).encode()
        def loads(self, payload):
            return json.loads(payload.decode())
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert bytes_serializer.loads(result) == {"key": "value"}
    
    # Test with custom salt
    serializer_with_salt = Serializer("secret-key", salt="custom-salt")
    result1 = serializer_with_salt.dumps({"key": "value"})
    result2 = serializer.dumps({"key": "value"})
    assert result1 != result2  # Different salts produce different signatures
    
    # Test with serializer_kwargs
    class KwargsSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj, **kwargs)
        def loads(self, payload):
            return json.loads(payload)
    
    kwargs_serializer = Serializer(
        "secret-key", 
        serializer=KwargsSerializer(),
        serializer_kwargs={"indent": 2}
    )
    result = kwargs_serializer.dumps({"key": "value"})
    assert "  " in result  # Indented JSON
    
    # Test dumps with multiple secret keys (key rotation)
    rotated_serializer = Serializer(["old-key", "new-key"])
    result = rotated_serializer.dumps({"test": "data"})
    assert rotated_serializer.loads(result) == {"test": "data"}
    # Should verify with both keys
    assert rotated_serializer.loads(result, salt=rotated_serializer.salt) == {"test": "data"}
    
    # Test dumps returns correct type for text serializer
    assert isinstance(serializer.dumps(123), str)
    
    # Test dumps returns correct type for bytes serializer  
    assert isinstance(bytes_serializer.dumps(123), bytes)
    
    # Test that dumps with empty data works
    assert serializer.dumps({}) is not None
    assert bytes_serializer.dumps({}) is not None
```


# LLM-generated content at query #89
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON serializer
    serializer = json
    pdata = _PDataSerializer()
    
    # Test loading a JSON string
    payload = '{"key": "value"}'
    result = pdata.loads(payload)
    assert result == {"key": "value"}
    
    # Test loading a list
    payload = '[1, 2, 3]'
    result = pdata.loads(payload)
    assert result == [1, 2, 3]
    
    # Test loading a number
    payload = '42'
    result = pdata.loads(payload)
    assert result == 42
    
    # Test loading a string
    payload = '"hello"'
    result = pdata.loads(payload)
    assert result == "hello"
    
    # Test loading with custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    payload = b'{"key": "value"}'
    result = pdata.loads(payload)
    assert result == {"key": "value"}
    
    # Test loading with invalid JSON
    import json
    try:
        pdata.loads('invalid json')
        assert False, "Should have raised an exception"
    except json.JSONDecodeError:
        pass
    
    # Test loading None
    try:
        pdata.loads(None)  # type: ignore
        assert False, "Should have raised an exception"
    except (TypeError, json.JSONDecodeError):
        pass
    
    # Test loading empty string
    try:
        pdata.loads('')
        assert False, "Should have raised an exception"
    except json.JSONDecodeError:
        pass
```


# LLM-generated content at query #90
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Create a simple serializer that implements the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with various data types
    test_data = [
        {"key": "value"},
        [1, 2, 3],
        "string",
        42,
        None,
        True,
    ]
    
    for data in test_data:
        result = serializer.dumps(data)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert serializer.loads(result) == data, f"Roundtrip failed for {data}"
    
    # Test that it returns str type (not bytes)
    assert isinstance(serializer.dumps({}), str), "Should return str type"


# LLM-generated content at query #91
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
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with a serializer that returns str
    class StrSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    str_serializer = StrSerializer()
    result = str_serializer.dumps(data)
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with edge cases
    assert serializer.dumps(None) == b'null'
    assert serializer.dumps(42) == b'42'
    assert serializer.dumps([1, 2, 3]) == b'[1, 2, 3]'
    
    # Test that dumps returns the correct type based on protocol
    assert isinstance(serializer.dumps({}), bytes)
    assert isinstance(str_serializer.dumps({}), str)


# LLM-generated content at query #92
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol correctly defines dumps method."""
    # Create a concrete implementation that matches the protocol
    class TestSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return str(obj)
    
    serializer = TestSerializer()
    
    # Test that dumps returns the expected type and value
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == "{'key': 'value'}"
    
    # Test with different types
    assert serializer.dumps(123) == "123"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    
    # Verify it conforms to _PDataSerializer protocol
    from typing import Protocol
    assert isinstance(serializer, _PDataSerializer) or hasattr(serializer, 'dumps')
```


# LLM-generated content at query #93
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple string serializer
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
    
    # Test with null payload
    result = serializer.loads('null')
    assert result is None
    
    # Test with boolean payload
    result = serializer.loads('true')
    assert result is True
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with custom object serializer
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            # Simple custom format: "key:value"
            parts = payload.split(":")
            return {parts[0]: parts[1]}
        
        def dumps(self, obj: t.Any) -> str:
            key, value = next(iter(obj.items()))
            return f"{key}:{value}"
    
    custom_serializer = CustomSerializer()
    result = custom_serializer.loads("name:John")
    assert result == {"name": "John"}
    
    # Test with empty payload
    class EmptySerializer:
        def loads(self, payload: str) -> t.Any:
            return payload
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    empty_serializer = EmptySerializer()
    result = empty_serializer.loads("")
    assert result == ""
    
    # Test protocol conformance - verify the loads method signature
    import inspect
    sig = inspect.signature(StringSerializer.loads)
    params = list(sig.parameters.values())
    assert len(params) == 2  # self and payload
    assert params[1].name == 'payload'
```


# LLM-generated content at query #94
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer.dumps serializes data correctly."""
    # Create a concrete implementation of _PDataSerializer
    class JSONSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = JSONSerializer()
    
    # Test basic serialization
    test_data = {"key": "value", "number": 42}
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    assert json.loads(result) == test_data
    
    # Test with list
    test_list = [1, 2, 3]
    result = serializer.dumps(test_list)
    assert json.loads(result) == test_list
    
    # Test with simple types
    assert serializer.dumps("hello") == '"hello"'
    assert serializer.dumps(123) == "123"
    assert serializer.dumps(True) == "true"
    assert serializer.dumps(None) == "null"


# LLM-generated content at query #95
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    # Verify it contains the payload and signature
    assert "." in result
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert b"." in result
    
    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert result_with_salt != result
    
    # Test with empty payload
    result_empty = serializer.dumps({})
    assert isinstance(result_empty, str)
    assert "." in result_empty
    
    # Test with list payload
    result_list = serializer.dumps([1, 2, 3])
    assert isinstance(result_list, str)
    assert "." in result_list
    
    # Verify the result can be loaded back
    loaded = serializer.loads(result)
    assert loaded == {"key": "value"}
    
    # Test with multiple secret keys (key rotation)
    serializer_multi = Serializer(["old-key", "new-key"])
    result_multi = serializer_multi.dumps({"test": "data"})
    assert isinstance(result_multi, str)
    # Should be signed with the newest key
    loaded_multi = serializer_multi.loads(result_multi)
    assert loaded_multi == {"test": "data"}
    
    # Test with serializer_kwargs
    serializer_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_kwargs = serializer_kwargs.dumps({"b": 2, "a": 1})
    assert isinstance(result_kwargs, str)
    # Verify the kwargs were used (sorted keys, no spaces)
    payload_part = result_kwargs.rsplit(".", 1)[0]
    # The payload after dumps is base64 encoded, so we can't directly check JSON format
    # But we can verify it loads correctly
    loaded_kwargs = serializer_kwargs.loads(result_kwargs)
    assert loaded_kwargs == {"b": 2, "a": 1}
    
    # Test with signer_kwargs
    serializer_signer = Serializer(
        "secret-key",
        signer_kwargs={"key_derivation": "hmac"}
    )
    result_signer = serializer_signer.dumps({"test": "data"})
    assert isinstance(result_signer, str)
    loaded_signer = serializer_signer.loads(result_signer)
    assert loaded_signer == {"test": "data"}
    
    # Test with custom signer class
    class CustomSigner(Signer):
        pass
    
    serializer_custom = Serializer("secret-key", signer=CustomSigner)
    result_custom = serializer_custom.dumps({"test": "data"})
    assert isinstance(result_custom, str)
    loaded_custom = serializer_custom.loads(result_custom)
    assert loaded_custom == {"test": "data"}
```


# LLM-generated content at query #96
#--------------------------

```python
def test_Serializer_load_payload():
    """Test Serializer.load_payload method."""
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test loading valid payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading payload with custom serializer (bytes)
    class BytesSerializer:
        def loads(self, payload):
            return {"data": payload}
        def dumps(self, obj):
            return b"test"
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    payload_bytes = b"test_payload"
    result = bytes_serializer.load_payload(payload_bytes)
    assert result == {"data": b"test_payload"}
    
    # Test with explicit serializer parameter
    class CustomSerializer:
        def loads(self, payload):
            return {"custom": payload}
        def dumps(self, obj):
            return b"custom"
    
    custom_serializer = Serializer("secret-key")
    result = custom_serializer.load_payload(b'test', serializer=CustomSerializer())
    assert result == {"custom": b"test"}
    
    # Test that BadPayload is raised for invalid JSON
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"invalid json")
    assert "Could not load the payload" in str(exc_info.value)
    assert exc_info.value.original_error is not None
    
    # Test with text serializer that returns string
    class TextSerializer:
        def loads(self, payload):
            return {"text": payload}
        def dumps(self, obj):
            return "test"
    
    text_serializer = Serializer("secret-key", serializer=TextSerializer())
    result = text_serializer.load_payload(b"hello world")
    assert result == {"text": "hello world"}
    
    # Test that exception is properly wrapped
    class FailingSerializer:
        def loads(self, payload):
            raise ValueError("Custom error")
        def dumps(self, obj):
            return "test"
    
    failing_serializer = Serializer("secret-key", serializer=FailingSerializer())
    with pytest.raises(BadPayload) as exc_info:
        failing_serializer.load_payload(b"test")
    assert isinstance(exc_info.value.original_error, ValueError)
    assert str(exc_info.value.original_error) == "Custom error"
```


# LLM-generated content at query #97
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default configuration (no fallback signers)
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_key == b"secret-key"

    # Test with fallback signers as dict
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"key_derivation": "none"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2  # main signer + fallback
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].key_derivation == "none"

    # Test with fallback signers as tuple
    class CustomSigner(Signer):
        pass
    
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(CustomSigner, {"key_derivation": "none"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], CustomSigner)
    assert signers[1].key_derivation == "none"

    # Test with fallback signers as Signer class
    serializer = Serializer(
        "secret-key",
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], CustomSigner)

    # Test with custom salt
    serializer = Serializer("secret-key", salt=b"custom-salt")
    signers = list(serializer.iter_unsigners())
    assert signers[0].salt == b"custom-salt"

    # Test with salt argument passed to iter_unsigners
    serializer = Serializer("secret-key", salt=b"default-salt")
    signers = list(serializer.iter_unsigners(salt=b"override-salt"))
    assert signers[0].salt == b"override-salt"

    # Test with multiple secret keys
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-key", b"new-key"]

    # Test with fallback signers and multiple secret keys
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"key_derivation": "none"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3  # main signer + 2 fallbacks (one for each key)
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_key == b"old-key"
    assert signers[2].secret_key == b"new-key"
    assert signers[1].key_derivation == "none"
    assert signers[2].key_derivation == "none"

    # Test with multiple secret keys and tuple fallback
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[(CustomSigner, {"key_derivation": "hmac"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert isinstance(signers[1], CustomSigner)
    assert isinstance(signers[2], CustomSigner)
    assert signers[1].secret_key == b"old-key"
    assert signers[2].secret_key == b"new-key"
    assert signers[1].key_derivation == "hmac"
    assert signers[2].key_derivation == "hmac"

    # Test with multiple secret keys and class fallback
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert isinstance(signers[1], CustomSigner)
    assert isinstance(signers[2], CustomSigner)
    assert signers[1].secret_key == b"old-key"
    assert signers[2].secret_key == b"new-key"

    # Test with empty fallback signers list
    serializer = Serializer("secret-key", fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1

    # Test with no salt
    serializer = Serializer("secret-key", salt=None)
    signers = list(serializer.iter_unsigners())
    assert signers[0].salt is None or signers[0].salt == b"itsdangerous"  # Signer default

    # Test that yield order is correct
    serializer = Serializer(
        "main-key",
        fallback_signers=[
            {"key_derivation": "none"},
            CustomSigner,
            (CustomSigner, {"key_derivation": "hmac"})
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    assert signers[0].secret_key == b"main-key"
    assert signers[0].key_derivation == "django-context"  # default
    assert signers[1].key_derivation == "none"
    assert isinstance(signers[2], CustomSigner)
    assert isinstance(signers[3], CustomSigner)
    assert signers[3].key_derivation == "hmac"

    # Test that secret_keys are properly passed to fallback signers
    serializer = Serializer(
        ["key1", "key2", "key3"],
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4  # main + 3 fallbacks (one for each key)
    assert signers[0].secret_keys == [b"key1", b"key2", b"key3"]
    assert signers[1].secret_key == b"key1"
    assert signers[2].secret_key == b"key2"
    assert signers[3].secret_key == b"key3"

    # Test that signer_kwargs are passed to class-based fallback signers
    serializer = Serializer(
        "secret-key",
        signer_kwargs={"key_derivation": "none"},
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer.iter_unsigners())
    assert signers[0].key_derivation == "none"
    assert signers[1].key_derivation == "none"  # should inherit from signer_kwargs

    # Test that dict fallback kwargs override signer_kwargs
    serializer = Serializer(
        "secret-key",
        signer_kwargs={"key_derivation": "none"},
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    signers = list(serializer.iter_unsigners())
    assert signers[0].key_derivation == "none"
    assert signers[1].key_derivation == "hmac"  # overridden by fallback kwargs
```


# LLM-generated content at query #98
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default configuration (no fallback signers)
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_key == b"secret-key"
    assert signers[0].salt == b"itsdangerous"

    # Test with custom salt
    serializer = Serializer("secret-key", salt=b"custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"

    # Test with fallback signers as dict
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"key_derivation": "none"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].key_derivation == "hmac"  # default
    assert signers[1].key_derivation == "none"

    # Test with fallback signers as tuple (signer class, kwargs)
    class CustomSigner(Signer):
        pass
    
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(CustomSigner, {"key_derivation": "none"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], CustomSigner)
    assert signers[1].key_derivation == "none"

    # Test with fallback signers as signer class
    serializer = Serializer(
        "secret-key",
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], CustomSigner)
    assert signers[1].key_derivation == "hmac"  # default from Signer

    # Test with multiple secret keys (key rotation)
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1  # only one signer with all keys
    assert signers[0].secret_keys == [b"old-key", b"new-key"]

    # Test with multiple secret keys and fallback signers
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"key_derivation": "none"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3  # 1 main + 2 fallback (one per secret key)
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_keys == [b"old-key"]
    assert signers[2].secret_keys == [b"new-key"]

    # Test with salt parameter passed to iter_unsigners
    serializer = Serializer("secret-key", salt=b"default-salt")
    signers = list(serializer.iter_unsigners(salt=b"override-salt"))
    assert len(signers) == 1
    assert signers[0].salt == b"override-salt"
```


# LLM-generated content at query #99
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a concrete implementation that matches the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with various data types
    assert serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps("test") == '"test"'
    assert serializer.dumps(42) == "42"
    assert serializer.dumps(None) == "null"
    assert serializer.dumps(True) == "true"
    
    # Test that the return type is str (not bytes)
    result = serializer.dumps({"a": 1})
    assert isinstance(result, str), "dumps should return str type"


# LLM-generated content at query #100
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test dumps with various Python objects
    assert serializer.dumps({"key": "value"}) == '{"key": "value"}'
    assert serializer.dumps([1, 2, 3]) == '[1, 2, 3]'
    assert serializer.dumps("test") == '"test"'
    assert serializer.dumps(42) == '42'
    assert serializer.dumps(None) == 'null'
    assert serializer.dumps(True) == 'true'
    assert serializer.dumps(False) == 'false'
    
    # Verify the return type is str as per the protocol
    result = serializer.dumps({"test": "data"})
    assert isinstance(result, str)
    
    # Test with complex nested data
    complex_data = {
        "string": "hello",
        "number": 123,
        "list": [1, 2, 3],
        "nested": {"key": "value"}
    }
    dumped = serializer.dumps(complex_data)
    assert json.loads(dumped) == complex_data
```


# LLM-generated content at query #101
#--------------------------

```python
def test_Serializer_dumps():
    """Test that dumps returns correct type and can be loaded back."""
    # Test with default JSON serializer (text)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
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
    s_salt = Serializer("secret-key", salt="custom-salt")
    result_salt = s_salt.dumps({"key": "value"})
    assert s_salt.loads(result_salt) == {"key": "value"}

    # Test with different data types
    for data in [None, True, 42, 3.14, "string", [1, 2, 3]]:
        result = s.dumps(data)
        assert s.loads(result) == data

    # Test that dumps with different keys produce different signatures
    s1 = Serializer("key1")
    s2 = Serializer("key2")
    payload = {"test": "data"}
    assert s1.dumps(payload) != s2.dumps(payload)```


# LLM-generated content at query #102
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer.dumps works correctly."""
    # Create a concrete implementation that matches the protocol
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with a simple dictionary
    test_data = {"key": "value", "number": 42}
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    assert json.loads(result) == test_data
    
    # Test with a list
    test_list = [1, 2, 3, "a", "b"]
    result = serializer.dumps(test_list)
    assert isinstance(result, str)
    assert json.loads(result) == test_list
    
    # Test with primitive types
    test_int = 123
    result = serializer.dumps(test_int)
    assert isinstance(result, str)
    assert json.loads(result) == test_int
    
    # Test with None
    result = serializer.dumps(None)
    assert isinstance(result, str)
    assert json.loads(result) is None
    
    # Test with boolean
    result = serializer.dumps(True)
    assert isinstance(result, str)
    assert json.loads(result) is True
    
    # Test with nested structures
    nested_data = {
        "outer": {
            "inner": [1, 2, 3],
            "value": "test"
        },
        "list": [{"a": 1}, {"b": 2}]
    }
    result = serializer.dumps(nested_data)
    assert isinstance(result, str)
    assert json.loads(result) == nested_data
```


# LLM-generated content at query #103
#--------------------------

```python
def test_Serializer_load_payload():
    """Test load_payload method of Serializer class."""
    # Test with default json serializer (text serializer)
    serializer = Serializer("test-secret-key")
    
    # Test loading valid JSON payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading valid JSON payload with non-ASCII characters
    payload = b'{"name": "\u00e9l\u00e8ve"}'
    result = serializer.load_payload(payload)
    assert result == {"name": "élève"}
    
    # Test loading valid JSON payload with numbers
    payload = b'{"count": 42, "price": 19.99}'
    result = serializer.load_payload(payload)
    assert result == {"count": 42, "price": 19.99}
    
    # Test loading valid JSON payload with list
    payload = b'[1, 2, 3, "test"]'
    result = serializer.load_payload(payload)
    assert result == [1, 2, 3, "test"]
    
    # Test loading valid JSON payload with boolean values
    payload = b'{"active": true, "completed": false}'
    result = serializer.load_payload(payload)
    assert result == {"active": True, "completed": False}
    
    # Test loading valid JSON payload with null
    payload = b'{"data": null}'
    result = serializer.load_payload(payload)
    assert result == {"data": None}
    
    # Test loading empty JSON object
    payload = b'{}'
    result = serializer.load_payload(payload)
    assert result == {}
    
    # Test loading empty JSON array
    payload = b'[]'
    result = serializer.load_payload(payload)
    assert result == []
    
    # Test that BadPayload is raised for invalid JSON
    payload = b'invalid json'
    with pytest.raises(BadPayload):
        serializer.load_payload(payload)
    
    # Test that BadPayload is raised for empty payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b'')
    
    # Test that BadPayload is raised for payload with trailing characters
    payload = b'{"key": "value"}extra'
    with pytest.raises(BadPayload):
        serializer.load_payload(payload)
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return {"data": payload}
        
        def dumps(self, obj: t.Any) -> bytes:
            return b"test"
    
    bytes_serializer = BytesSerializer()
    serializer_bytes = Serializer("test-secret-key", serializer=bytes_serializer)
    
    # Test loading with bytes serializer
    payload = b"test bytes payload"
    result = serializer_bytes.load_payload(payload)
    assert result == {"data": b"test bytes payload"}
    
    # Test with custom serializer provided as parameter
    class CustomSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"custom": payload}
        
        def dumps(self, obj: t.Any) -> str:
            return "custom"
    
    custom_serializer = CustomSerializer()
    
    # Test loading with custom serializer as parameter
    payload = b"custom data"
    result = serializer.load_payload(payload, serializer=custom_serializer)
    assert result == {"custom": "custom data"}
    
    # Test that BadPayload preserves original error message
    class FailingSerializer:
        def loads(self, payload: bytes) -> t.Any:
            raise ValueError("Custom error message")
        
        def dumps(self, obj: t.Any) -> bytes:
            return b"test"
    
    failing_serializer = FailingSerializer()
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"test", serializer=failing_serializer)
    assert "Could not load the payload" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, ValueError)
    assert str(exc_info.value.original_error) == "Custom error message"
    
    # Test with nested JSON structures
    payload = b'{"outer": {"inner": "value"}, "list": [1, {"nested": True}]}'
    result = serializer.load_payload(payload)
    assert result == {"outer": {"inner": "value"}, "list": [1, {"nested": True}]}
    
    # Test with JSON containing special characters
    payload = b'{"text": "Line 1\\nLine 2\\tTabbed"}'
    result = serializer.load_payload(payload)
    assert result == {"text": "Line 1\nLine 2\tTabbed"}
    
    # Test with JSON containing unicode escape sequences
    payload = b'{"unicode": "\\u0048\\u0065\\u006c\\u006c\\u006f"}'
    result = serializer.load_payload(payload)
    assert result == {"unicode": "Hello"}```


# LLM-generated content at query #104
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer.loads correctly deserializes data."""
    # Create a concrete implementation of _PDataSerializer
    class TestSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with valid JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with simple types
    assert serializer.loads('123') == 123
    assert serializer.loads('"hello"') == "hello"
    assert serializer.loads('null') is None
    assert serializer.loads('true') is True
    assert serializer.loads('[1, 2, 3]') == [1, 2, 3]
    
    # Test with nested structures
    nested = '{"a": [1, 2, {"b": 3}], "c": "d"}'
    result = serializer.loads(nested)
    assert result == {"a": [1, 2, {"b": 3}], "c": "d"}
```


# LLM-generated content at query #105
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a JSON serializer
    serializer = json
    pds = _PDataSerializer[str]()
    pds.loads = serializer.loads
    pds.dumps = serializer.dumps
    
    # Test loading valid JSON
    result = pds.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = pds.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading a string
    result = pds.loads('"hello"')
    assert result == "hello"
    
    # Test loading a number
    result = pds.loads('42')
    assert result == 42
    
    # Test loading null
    result = pds.loads('null')
    assert result is None
    
    # Test loading boolean
    result = pds.loads('true')
    assert result is True
    
    # Test loading with a custom serializer
    class CustomSerializer:
        def loads(self, payload):
            return payload.upper()
        def dumps(self, obj):
            return obj.lower()
    
    custom_serializer = CustomSerializer()
    pds2 = _PDataSerializer[str]()
    pds2.loads = custom_serializer.loads
    pds2.dumps = custom_serializer.dumps
    
    result = pds2.loads("hello")
    assert result == "HELLO"
```


# LLM-generated content at query #106
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Create a mock serializer that implements the _PDataSerializer protocol
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
    
    # Test with different data types
    assert serializer.dumps("string") == '"string"'
    assert serializer.dumps(123) == "123"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    assert serializer.dumps(None) == "null"
```


# LLM-generated content at query #107
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol accepts valid serializer implementations."""
    
    class ValidTextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    class ValidBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    # Test with text serializer
    text_serializer = ValidTextSerializer()
    assert text_serializer.loads('{"key": "value"}') == {"key": "value"}
    
    # Test with bytes serializer
    bytes_serializer = ValidBytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    
    # Test that loads method accepts only positional argument
    serializer = ValidTextSerializer()
    result = serializer.loads('{"test": 123}')
    assert result == {"test": 123}
    
    # Test with various data types
    assert text_serializer.loads('null') is None
    assert text_serializer.loads('true') is True
    assert text_serializer.loads('42') == 42
    assert text_serializer.loads('"string"') == "string"
    assert text_serializer.loads('[1, 2, 3]') == [1, 2, 3]
```


# LLM-generated content at query #108
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple dict payload
    serializer = json
    data = '{"key": "value"}'
    result = serializer.loads(data)
    assert result == {"key": "value"}

    # Test with a list payload
    data = '[1, 2, 3]'
    result = serializer.loads(data)
    assert result == [1, 2, 3]

    # Test with a string payload
    data = '"hello"'
    result = serializer.loads(data)
    assert result == "hello"

    # Test with a number payload
    data = '42'
    result = serializer.loads(data)
    assert result == 42

    # Test with null payload
    data = 'null'
    result = serializer.loads(data)
    assert result is None

    # Test with boolean payload
    data = 'true'
    result = serializer.loads(data)
    assert result is True

    data = 'false'
    result = serializer.loads(data)
    assert result is False

    # Test with nested structure
    data = '{"a": [1, 2, {"b": 3}]}'
    result = serializer.loads(data)
    assert result == {"a": [1, 2, {"b": 3}]}

    # Test with empty dict
    data = '{}'
    result = serializer.loads(data)
    assert result == {}

    # Test with empty list
    data = '[]'
    result = serializer.loads(data)
    assert result == []
```


# LLM-generated content at query #109
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload):
            if isinstance(payload, str):
                return json.loads(payload)
            elif isinstance(payload, bytes):
                return json.loads(payload.decode('utf-8'))
            raise TypeError("Payload must be str or bytes")
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with string payload
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes payload
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with different data types
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test with null
    result = serializer.loads('null')
    assert result is None
    
    # Test with number
    result = serializer.loads('42')
    assert result == 42
    
    # Test with boolean
    result = serializer.loads('true')
    assert result is True
    
    # Test with empty JSON
    result = serializer.loads('{}')
    assert result == {}
```


# LLM-generated content at query #110
#--------------------------

```python
def test_Serializer_dumps():
    """Test that dumps returns a signed string serialized with the internal serializer."""
    # Test with default JSON serializer (text serializer)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Should contain a separator
    
    # Test that the result can be loaded back
    loaded = s.loads(result)
    assert loaded == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            import json
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            import json
            return json.loads(payload.decode("utf-8"))
    
    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = s_bytes.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    # Test with salt parameter
    s_salt = Serializer("secret-key")
    result_with_salt = s_salt.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert result_with_salt != result  # Different salt produces different signature
    
    # Test that different objects produce different results
    result2 = s.dumps({"key": "other-value"})
    assert result != result2
    
    # Test with empty dict
    result_empty = s.dumps({})
    assert isinstance(result_empty, str)
    assert s.loads(result_empty) == {}
    
    # Test with list
    result_list = s.dumps([1, 2, 3])
    assert isinstance(result_list, str)
    assert s.loads(result_list) == [1, 2, 3]
    
    # Test with None
    result_none = s.dumps(None)
    assert isinstance(result_none, str)
    assert s.loads(result_none) is None
```


# LLM-generated content at query #111
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol implementers can dumps data."""
    # Test with JSON serializer (text-based)
    json_serializer = json
    result = json_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'

    # Test with custom bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))

        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")

    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'

    # Test with custom text serializer
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)

        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)

    text_serializer = TextSerializer()
    result = text_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
```


# LLM-generated content at query #112
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with default json serializer (text serializer)
    serializer = Serializer("test-secret-key")
    
    # Test valid payload
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes):
            return {"data": payload}
        
        def dumps(self, obj):
            return b"test"
    
    bytes_serializer = Serializer("test-secret-key", serializer=BytesSerializer())
    payload_bytes = b"test-data"
    assert bytes_serializer.load_payload(payload_bytes) == {"data": payload_bytes}
    
    # Test with custom serializer
    class CustomSerializer:
        def loads(self, payload: str):
            return {"custom": payload}
        
        def dumps(self, obj):
            return "test"
    
    custom_serializer = Serializer("test-secret-key", serializer=CustomSerializer())
    payload_str = b"custom-data"
    assert custom_serializer.load_payload(payload_str) == {"custom": "custom-data"}
    
    # Test with override serializer parameter
    class OverrideSerializer:
        def loads(self, payload: str):
            return {"override": payload}
        
        def dumps(self, obj):
            return "test"
    
    override_serializer = OverrideSerializer()
    result = serializer.load_payload(b'{"key": "value"}', serializer=override_serializer)
    assert result == {"override": '{"key": "value"}'}
    
    # Test invalid payload raises BadPayload
    import pytest
    with pytest.raises(BadPayload):
        serializer.load_payload(b"invalid json")
    
    # Test empty payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test with None payload (should raise BadPayload due to exception)
    with pytest.raises(BadPayload):
        serializer.load_payload(None)
    
    # Test payload with special characters
    special_payload = b'{"special": "\u00e9\u00f1\u00fc"}'
    result = serializer.load_payload(special_payload)
    assert result == {"special": "éñü"}
    
    # Test payload with numbers
    number_payload = b'{"num": 42}'
    assert serializer.load_payload(number_payload) == {"num": 42}
    
    # Test payload with boolean
    bool_payload = b'{"flag": true}'
    assert serializer.load_payload(bool_payload) == {"flag": True}
    
    # Test payload with null
    null_payload = b'{"value": null}'
    assert serializer.load_payload(null_payload) == {"value": None}
    
    # Test payload with nested structures
    nested_payload = b'{"nested": {"inner": "value"}}'
    assert serializer.load_payload(nested_payload) == {"nested": {"inner": "value"}}
    
    # Test payload with list
    list_payload = b'{"items": [1, 2, 3]}'
    assert serializer.load_payload(list_payload) == {"items": [1, 2, 3]}
```


# LLM-generated content at query #113
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test _PDataSerializer protocol's loads method behavior."""
    # Test with a JSON serializer (text-based)
    json_serializer = json
    serializer_instance = _PDataSerializer[str]()
    
    # Test that loads works with valid JSON string
    payload = '{"key": "value"}'
    result = json_serializer.loads(payload)
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> dict:
            return json.loads(payload.decode())
        
        def dumps(self, obj: dict) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    bytes_payload = b'{"key": "value"}'
    result = bytes_serializer.loads(bytes_payload)
    assert result == {"key": "value"}
    
    # Test with custom serializer
    class CustomSerializer:
        def loads(self, payload: str) -> list:
            return payload.split(",")
        
        def dumps(self, obj: list) -> str:
            return ",".join(obj)
    
    custom_serializer = CustomSerializer()
    result = custom_serializer.loads("a,b,c")
    assert result == ["a", "b", "c"]
    
    # Test that loads raises appropriate exception for invalid data
    try:
        json_serializer.loads("invalid json")
        assert False, "Should have raised exception"
    except json.JSONDecodeError:
        assert True
```


# LLM-generated content at query #114
#--------------------------

```python
def test_Serializer_iter_unsigners():
    """Test iter_unsigners method of Serializer class."""
    # Test with default settings
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    
    # Test with fallback signers as dict
    fallback_dict = {"digest_method": "sha256"}
    serializer = Serializer("secret-key", fallback_signers=[fallback_dict])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2  # default + fallback
    assert all(isinstance(s, Signer) for s in signers)
    
    # Test with fallback signers as tuple
    fallback_tuple = (Serializer.default_signer, {"key_derivation": "hmac"})
    serializer = Serializer("secret-key", fallback_signers=[fallback_tuple])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert all(isinstance(s, Signer) for s in signers)
    
    # Test with fallback signers as Signer class
    serializer = Serializer("secret-key", fallback_signers=[Serializer.default_signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert all(isinstance(s, Signer) for s in signers)
    
    # Test with multiple secret keys
    serializer = Serializer(["key1", "key2", "key3"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1  # only one signer for multiple keys
    
    # Test with multiple secret keys and fallback signers
    serializer = Serializer(["key1", "key2"], fallback_signers=[{"digest_method": "sha256"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3  # 1 default + 2 fallbacks (one for each key)
    
    # Test with custom salt
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners(salt=b"custom-salt"))
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    
    # Test with empty fallback signers
    serializer = Serializer("secret-key", fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    
    # Test generator behavior (not a list)
    serializer = Serializer("secret-key", fallback_signers=[{"digest_method": "sha256"}])
    gen = serializer.iter_unsigners()
    assert hasattr(gen, "__next__")
    assert hasattr(gen, "__iter__")
    
    # Verify fallback signers use correct secret keys
    serializer = Serializer(["key1", "key2", "key3"], fallback_signers=[{"digest_method": "sha256"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4  # 1 default + 3 fallbacks (one for each key)
    assert signers[0].secret_keys == [b"key3"]  # default uses newest
    assert signers[1].secret_keys == [b"key1"]  # fallback iterates in order
    assert signers[2].secret_keys == [b"key2"]
    assert signers[3].secret_keys == [b"key3"]
```


# LLM-generated content at query #115
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text)
    s = Serializer("secret-key")
    result = s.dumps({"key": "value"})
    assert isinstance(result, str)
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
    s_salt = Serializer("secret-key", salt="custom-salt")
    result_salt = s_salt.dumps({"key": "value"})
    assert s_salt.loads(result_salt, salt="custom-salt") == {"key": "value"}

    # Test with serializer_kwargs
    s_kwargs = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    result_kwargs = s_kwargs.dumps({"b": 2, "a": 1})
    assert s_kwargs.loads(result_kwargs) == {"a": 1, "b": 2}

    # Test that different keys produce different signatures
    s1 = Serializer("key1")
    s2 = Serializer("key2")
    data = {"test": "data"}
    assert s1.dumps(data) != s2.dumps(data)

    # Test with empty dict
    s = Serializer("secret-key")
    result_empty = s.dumps({})
    assert s.loads(result_empty) == {}

    # Test with list data
    result_list = s.dumps([1, 2, 3])
    assert s.loads(result_list) == [1, 2, 3]

    # Test with string data
    result_str = s.dumps("hello")
    assert s.loads(result_str) == "hello"

    # Test that dumps returns consistent results with same inputs
    result1 = s.dumps({"key": "value"})
    result2 = s.dumps({"key": "value"})
    assert result1 == result2  # Same inputs should produce same output

    # Test with multiple secret keys
    s_multi = Serializer(["old-key", "new-key"])
    result_multi = s_multi.dumps({"key": "value"})
    # Should be signed with the newest key
    assert s_multi.loads(result_multi) == {"key": "value"}
    # Old key should still be able to verify
    s_old = Serializer(["old-key"])
    assert s_old.loads(result_multi) == {"key": "value"}  # This should fail since signed with new key
```


# LLM-generated content at query #116
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works with different payload types."""
    # Test with str serializer
    str_serializer = type("StrSerializer", (), {
        "loads": lambda self, payload: json.loads(payload),
        "dumps": lambda self, obj: json.dumps(obj)
    })()
    
    result = str_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    bytes_serializer = type("BytesSerializer", (), {
        "loads": lambda self, payload: json.loads(payload.decode()),
        "dumps": lambda self, obj: json.dumps(obj).encode()
    })()
    
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test that loads returns Any (can be various types)
    assert isinstance(str_serializer.loads('"string"'), str)
    assert isinstance(str_serializer.loads('123'), int)
    assert isinstance(str_serializer.loads('null'), type(None))
```


# LLM-generated content at query #117
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key", salt=b"test-salt")
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    # Verify we can decode it back
    assert serializer.loads(result) == data

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")

        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))

    bytes_serializer = Serializer(
        "secret-key", salt=b"test-salt", serializer=BytesSerializer()
    )
    result_bytes = bytes_serializer.dumps(data)
    assert isinstance(result_bytes, bytes)
    assert bytes_serializer.loads(result_bytes) == data

    # Test with custom salt
    result_custom_salt = serializer.dumps(data, salt=b"custom-salt")
    assert isinstance(result_custom_salt, str)
    assert serializer.loads(result_custom_salt, salt=b"custom-salt") == data

    # Test that different salts produce different results
    result_salt1 = serializer.dumps(data, salt=b"salt1")
    result_salt2 = serializer.dumps(data, salt=b"salt2")
    assert result_salt1 != result_salt2

    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        salt=b"test-salt",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_kwargs = serializer_with_kwargs.dumps(data)
    assert isinstance(result_kwargs, str)
    assert serializer_with_kwargs.loads(result_kwargs) == data

    # Test round trip with various data types
    test_cases = [
        {"key": "value"},
        [1, 2, 3],
        "simple string",
        42,
        None,
        {"nested": {"a": 1, "b": [1, 2, 3]}},
    ]
    for test_data in test_cases:
        serialized = serializer.dumps(test_data)
        deserialized = serializer.loads(serialized)
        assert deserialized == test_data, f"Failed for data: {test_data}"

    # Test that dumps returns consistent results for same input
    result1 = serializer.dumps(data)
    result2 = serializer.dumps(data)
    assert result1 == result2, "Dumps should be deterministic for same input"

    # Test with empty data
    empty_result = serializer.dumps({})
    assert isinstance(empty_result, str)
    assert serializer.loads(empty_result) == {}
```


# LLM-generated content at query #118
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    # Test with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = StringSerializer()
    assert isinstance(serializer, _PDataSerializer)
    
    # Test dumps returns expected string
    result = serializer.dumps({"key": "value"})
    assert result == '{"key": "value"}'
    assert isinstance(result, str)
    
    # Test with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    assert isinstance(bytes_serializer, _PDataSerializer)
    
    # Test dumps returns expected bytes
    result = bytes_serializer.dumps({"key": "value"})
    assert result == b'{"key": "value"}'
    assert isinstance(result, bytes)


# LLM-generated content at query #119
#--------------------------

```python
def test_Serializer_dumps():
    """Test Serializer.dumps method with various configurations."""
    
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("test-secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Contains signature separator
    
    # Test that dumps produces a valid signed payload
    payload = serializer.dumps("test-data")
    assert isinstance(payload, str)
    assert serializer.loads(payload) == "test-data"
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("test-key", serializer=BytesSerializer())
    bytes_result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(bytes_result, bytes)
    assert bytes_serializer.loads(bytes_result) == {"key": "value"}
    
    # Test with custom salt
    custom_salt = b"custom-salt"
    serializer_with_salt = Serializer("test-key", salt=custom_salt)
    result_with_salt = serializer_with_salt.dumps("test-data")
    assert isinstance(result_with_salt, str)
    assert serializer_with_salt.loads(result_with_salt, salt=custom_salt) == "test-data"
    
    # Test that different keys produce different signatures
    serializer1 = Serializer("key1")
    serializer2 = Serializer("key2")
    result1 = serializer1.dumps("test")
    result2 = serializer2.dumps("test")
    assert result1 != result2
    
    # Test with serializer_kwargs
    class CustomSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj, **kwargs)
        def loads(self, payload):
            return json.loads(payload)
    
    custom_serializer = Serializer(
        "test-key",
        serializer=CustomSerializer(),
        serializer_kwargs={"indent": 2}
    )
    result = custom_serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert custom_serializer.loads(result) == {"key": "value"}
    
    # Test that dumps handles complex nested data
    complex_data = {
        "string": "hello",
        "number": 42,
        "list": [1, 2, 3],
        "nested": {"a": 1, "b": 2}
    }
    serialized = serializer.dumps(complex_data)
    assert serializer.loads(serialized) == complex_data
    
    # Test with empty data
    empty_result = serializer.dumps({})
    assert serializer.loads(empty_result) == {}
    
    # Test with None value
    none_result = serializer.dumps(None)
    assert serializer.loads(none_result) is None
    
    # Test that dumps returns consistent length for same data
    result_a = serializer.dumps("same-data")
    result_b = serializer.dumps("same-data")
    assert len(result_a) == len(result_b)
    assert result_a == result_b  # Same key and salt should produce same result```


# LLM-generated content at query #120
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple dict payload
    serializer = json
    pd_serializer: _PDataSerializer[t.Any] = serializer
    
    payload = '{"key": "value"}'
    result = pd_serializer.loads(payload)
    assert result == {"key": "value"}
    
    # Test with a list payload
    payload = "[1, 2, 3]"
    result = pd_serializer.loads(payload)
    assert result == [1, 2, 3]
    
    # Test with primitive types
    payload = "42"
    result = pd_serializer.loads(payload)
    assert result == 42
    
    payload = '"hello"'
    result = pd_serializer.loads(payload)
    assert result == "hello"
    
    payload = "true"
    result = pd_serializer.loads(payload)
    assert result is True
    
    payload = "null"
    result = pd_serializer.loads(payload)
    assert result is None
    
    # Test with bytes payload (should still work with json)
    payload = b'{"key": "value"}'
    result = pd_serializer.loads(payload)
    assert result == {"key": "value"}
    
    # Test with custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode())
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode()
    
    bytes_serializer = BytesSerializer()
    pd_bytes_serializer: _PDataSerializer[t.Any] = bytes_serializer
    
    payload = b'{"key": "value"}'
    result = pd_bytes_serializer.loads(payload)
    assert result == {"key": "value"}
    
    # Test error handling - invalid JSON should raise an exception
    with pytest.raises(json.JSONDecodeError):
        pd_serializer.loads("invalid json")
    
    # Test with empty payload
    with pytest.raises(json.JSONDecodeError):
        pd_serializer.loads("")
```


# LLM-generated content at query #121
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol dumps method works correctly."""
    
    # Create a concrete implementation of _PDataSerializer
    class TestSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test basic serialization
    test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    assert json.loads(result) == test_data
    
    # Test serialization of simple types
    assert serializer.dumps("hello") == '"hello"'
    assert serializer.dumps(123) == "123"
    assert serializer.dumps(True) == "true"
    assert serializer.dumps(None) == "null"
    
    # Test serialization of empty data
    assert serializer.dumps({}) == "{}"
    assert serializer.dumps([]) == "[]"
    
    # Verify that dumps is callable with any object
    class CustomObject:
        pass
    
    # Should raise TypeError for non-serializable objects
    try:
        serializer.dumps(CustomObject())
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #122
#--------------------------

```python
def test_Serializer_load_payload():
    # Test with text serializer (json)
    serializer = Serializer("secret", serializer=json)
    
    # Test successful load with text serializer
    data = b'{"key": "value"}'
    result = serializer.load_payload(data)
    assert result == {"key": "value"}
    
    # Test successful load with text serializer and explicit serializer parameter
    result = serializer.load_payload(data, serializer=json)
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    serializer_bytes = Serializer("secret", serializer=bytes_serializer)
    
    # Test successful load with bytes serializer
    result = serializer_bytes.load_payload(data)
    assert result == {"key": "value"}
    
    # Test BadPayload exception for invalid JSON
    invalid_data = b"invalid json"
    with pytest.raises(BadPayload):
        serializer.load_payload(invalid_data)
    
    # Test with custom serializer that raises Exception
    class FailingSerializer:
        def loads(self, payload: str) -> t.Any:
            raise ValueError("Custom error")
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    failing_serializer = FailingSerializer()
    serializer_fail = Serializer("secret", serializer=failing_serializer)
    
    with pytest.raises(BadPayload) as exc_info:
        serializer_fail.load_payload(data)
    assert "Could not load the payload" in str(exc_info.value)
    assert "original_error" in str(exc_info.value)
    
    # Test with empty payload
    empty_data = b""
    with pytest.raises(BadPayload):
        serializer.load_payload(empty_data)
    
    # Test with bytes payload using explicit text serializer parameter
    result = serializer.load_payload(data, serializer=json)
    assert result == {"key": "value"}
    
    # Test with bytes payload using explicit bytes serializer parameter
    result = serializer_bytes.load_payload(data, serializer=bytes_serializer)
    assert result == {"key": "value"}
```


# LLM-generated content at query #123
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a serializer that returns str
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    text_serializer = TextSerializer()
    result = text_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with a serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with integer payload
    class IntSerializer:
        def loads(self, payload: str) -> t.Any:
            return int(payload)
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    int_serializer = IntSerializer()
    result = int_serializer.loads("42")
    assert result == 42
    
    # Test with list payload
    class ListSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    list_serializer = ListSerializer()
    result = list_serializer.loads("[1, 2, 3]")
    assert result == [1, 2, 3]
```


# LLM-generated content at query #124
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert serializer.loads(result) == data

    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            import pickle
            return pickle.dumps(obj)
        def loads(self, payload):
            import pickle
            return pickle.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    data = [1, 2, 3]
    result = bytes_serializer.dumps(data)
    assert isinstance(result, bytes)
    assert bytes_serializer.loads(result) == data

    # Test with salt parameter
    serializer_with_salt = Serializer("secret-key", salt="custom-salt")
    data = "test data"
    result = serializer_with_salt.dumps(data)
    assert serializer_with_salt.loads(result) == data

    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    data = {"b": 2, "a": 1}
    result = serializer_with_kwargs.dumps(data)
    assert serializer_with_kwargs.loads(result) == data

    # Test with different secret key types
    serializer_with_bytes_key = Serializer(b"bytes-key")
    data = "test"
    result = serializer_with_bytes_key.dumps(data)
    assert serializer_with_bytes_key.loads(result) == data

    # Test with multiple secret keys (key rotation)
    serializer_multi_keys = Serializer(["old-key", "new-key"])
    data = "test data"
    result = serializer_multi_keys.dumps(data)
    assert serializer_multi_keys.loads(result) == data

    # Test that dumps produces different results for different data
    serializer1 = Serializer("secret")
    result1 = serializer1.dumps({"a": 1})
    result2 = serializer1.dumps({"a": 2})
    assert result1 != result2

    # Test that dumps with different salts produces different results
    serializer_salt1 = Serializer("secret", salt="salt1")
    serializer_salt2 = Serializer("secret", salt="salt2")
    data = {"test": "data"}
    result_salt1 = serializer_salt1.dumps(data)
    result_salt2 = serializer_salt2.dumps(data)
    assert result_salt1 != result_salt2
    assert serializer_salt1.loads(result_salt1) == data
    assert serializer_salt2.loads(result_salt2) == data
```


# LLM-generated content at query #125
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test that _PDataSerializer protocol requires dumps method
    class MockSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return str(obj)
    
    serializer = MockSerializer()
    result = serializer.dumps({"test": "data"})
    assert result == "{'test': 'data'}"
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return str(obj).encode()
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"test": "data"})
    assert isinstance(result, bytes)
    assert result == b"{'test': 'data'}"
```


# LLM-generated content at query #126
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Test that _PDataSerializer protocol works with a string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload
            
        def dumps(self, obj: t.Any) -> str:
            return str(obj)
    
    serializer = StringSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == "{'key': 'value'}"
    
    # Test that _PDataSerializer protocol works with a bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload
            
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode()
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b"{'key': 'value'}"
    
    # Test is_text_serializer function
    assert is_text_serializer(StringSerializer()) == True
    assert is_text_serializer(BytesSerializer()) == False
```


# LLM-generated content at query #127
#--------------------------

```python
def test_Serializer_dumps():
    serializer = Serializer("secret-key")
    
    # Test with default json serializer
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.count(".") == 2  # payload.salt.signature format
    
    # Verify the result can be loaded back
    loaded = serializer.loads(result)
    assert loaded == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode()
        def loads(self, payload):
            return json.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    loaded_bytes = bytes_serializer.loads(result_bytes)
    assert loaded_bytes == {"key": "value"}
    
    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    loaded_with_salt = serializer.loads(result_with_salt, salt="custom-salt")
    assert loaded_with_salt == {"key": "value"}
    
    # Test with different data types
    assert serializer.dumps(123) is not None
    assert serializer.dumps("string") is not None
    assert serializer.dumps([1, 2, 3]) is not None
    assert serializer.dumps(None) is not None
    
    # Test with serializer_kwargs
    custom_serializer = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True}
    )
    result_sorted = custom_serializer.dumps({"b": 1, "a": 2})
    assert isinstance(result_sorted, str)
    
    # Test with key rotation (list of keys)
    key_rotation_serializer = Serializer(["old-key", "new-key"])
    result_rotated = key_rotation_serializer.dumps({"test": "data"})
    assert isinstance(result_rotated, str)
    loaded_rotated = key_rotation_serializer.loads(result_rotated)
    assert loaded_rotated == {"test": "data"}

```


# LLM-generated content at query #128
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a simple serializer that implements the protocol
    class TestSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return str(obj)
    
    serializer = TestSerializer()
    
    # Test with various types
    assert serializer.dumps({"key": "value"}) == "{'key': 'value'}"
    assert serializer.dumps(42) == "42"
    assert serializer.dumps("hello") == "hello"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
    
    # Test that the return type matches _TSerialized (str in this case)
    result = serializer.dumps({"test": "data"})
    assert isinstance(result, str)


# LLM-generated content at query #129
#--------------------------

```python
def test_Serializer_dumps():
    """Test Serializer.dumps method with various configurations."""
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.startswith("eyJrZXkiOiAidmFsdWUifQ")  # base64 of JSON
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode()
        def loads(self, data):
            return json.loads(data.decode())
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    
    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert result_with_salt != result  # Different due to different salt
    
    # Verify that dumps produces consistent results
    result1 = serializer.dumps({"data": "test"})
    result2 = serializer.dumps({"data": "test"})
    assert result1 == result2  # Same input should produce same output
    
    # Test with serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    # With sort_keys=True and separator=",:", this should be compact and sorted
    assert isinstance(result, str)
    assert ',"' not in result  # No spaces after commas
    
    # Test with simple data types
    assert isinstance(serializer.dumps("string"), str)
    assert isinstance(serializer.dumps(123), str)
    assert isinstance(serializer.dumps([1, 2, 3]), str)
    assert isinstance(serializer.dumps(None), str)


# LLM-generated content at query #130
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol implementations can dumps objects."""
    # Test with json serializer (str based)
    json_serializer = json
    assert isinstance(json_serializer.dumps({"key": "value"}), str)
    
    # Test with custom bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    assert isinstance(bytes_serializer.dumps({"key": "value"}), bytes)
    
    # Verify protocol conformance
    assert isinstance(json_serializer, _PDataSerializer)
    assert isinstance(bytes_serializer, _PDataSerializer)
```


# LLM-generated content at query #131
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns a dictionary
    class TestSerializer:
        def loads(self, payload):
            return {"key": "value"}
        
        def dumps(self, obj):
            return "dummy"
    
    serializer = TestSerializer()
    result = serializer.loads("test payload")
    assert result == {"key": "value"}
    
    # Test with a serializer that handles JSON
    import json
    json_serializer = json
    payload = '{"name": "test", "value": 42}'
    result = json_serializer.loads(payload)
    assert result == {"name": "test", "value": 42}
    
    # Test with bytes payload
    class BytesSerializer:
        def loads(self, payload):
            return payload.decode("utf-8") if isinstance(payload, bytes) else payload
        
        def dumps(self, obj):
            return str(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b"hello bytes")
    assert result == "hello bytes"
    
    # Test with integer payload
    class IntSerializer:
        def loads(self, payload):
            return int(payload)
        
        def dumps(self, obj):
            return str(obj)
    
    int_serializer = IntSerializer()
    result = int_serializer.loads("123")
    assert result == 123
    
    # Test with None payload (should raise exception in most cases)
    class NoneSerializer:
        def loads(self, payload):
            if payload is None:
                raise ValueError("Cannot load None payload")
            return payload
        
        def dumps(self, obj):
            return str(obj)
    
    none_serializer = NoneSerializer()
    try:
        none_serializer.loads(None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test with empty payload
    class EmptySerializer:
        def loads(self, payload):
            if not payload:
                return None
            return payload
        
        def dumps(self, obj):
            return str(obj)
    
    empty_serializer = EmptySerializer()
    result = empty_serializer.loads("")
    assert result is None
    result = empty_serializer.loads("non-empty")
    assert result == "non-empty"
```


# LLM-generated content at query #132
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple serializer that returns JSON
    serializer = json
    pdata_serializer: _PDataSerializer = serializer
    
    # Test loading a valid JSON string
    payload = '{"key": "value"}'
    result = pdata_serializer.loads(payload)
    assert result == {"key": "value"}
    
    # Test loading a JSON array
    payload = '[1, 2, 3]'
    result = pdata_serializer.loads(payload)
    assert result == [1, 2, 3]
    
    # Test loading a JSON number
    payload = '42'
    result = pdata_serializer.loads(payload)
    assert result == 42
    
    # Test loading a JSON boolean
    payload = 'true'
    result = pdata_serializer.loads(payload)
    assert result is True
    
    # Test loading a JSON null
    payload = 'null'
    result = pdata_serializer.loads(payload)
    assert result is None
    
    # Test loading a JSON string
    payload = '"hello"'
    result = pdata_serializer.loads(payload)
    assert result == "hello"
    
    # Test with a custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode('utf-8').split(',')
        
        def dumps(self, obj: t.Any) -> bytes:
            return ','.join(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    pdata_serializer_bytes: _PDataSerializer = bytes_serializer
    
    payload = b"apple,banana,cherry"
    result = pdata_serializer_bytes.loads(payload)
    assert result == ["apple", "banana", "cherry"]
    
    # Test loading an empty payload
    payload = '{}'
    result = pdata_serializer.loads(payload)
    assert result == {}
    
    # Test loading a nested structure
    payload = '{"a": {"b": [1, 2, 3]}}'
    result = pdata_serializer.loads(payload)
    assert result == {"a": {"b": [1, 2, 3]}}
```


# LLM-generated content at query #133
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload: str | bytes) -> t.Any:
            if isinstance(payload, str):
                return {"data": payload.upper()}
            elif isinstance(payload, bytes):
                return {"data": payload.decode("utf-8").upper()}
            raise ValueError("Invalid payload type")
        
        def dumps(self, obj: t.Any) -> str | bytes:
            return json.dumps(obj)

    serializer = MockSerializer()
    
    # Test with string payload
    result = serializer.loads("hello")
    assert result == {"data": "HELLO"}
    
    # Test with bytes payload
    result = serializer.loads(b"world")
    assert result == {"data": "WORLD"}
    
    # Test with JSON string payload
    json_serializer = _PDataSerializer[json]
    result = json.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with empty payload
    result = serializer.loads("")
    assert result == {"data": ""}
    
    result = serializer.loads(b"")
    assert result == {"data": ""}
```


# LLM-generated content at query #134
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test loading a simple JSON payload
    payload = '{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}
    
    # Test loading a list
    payload = '[1, 2, 3]'
    result = serializer.loads(payload)
    assert result == [1, 2, 3]
    
    # Test loading a string
    payload = '"hello"'
    result = serializer.loads(payload)
    assert result == "hello"
    
    # Test loading a number
    payload = '42'
    result = serializer.loads(payload)
    assert result == 42
    
    # Test loading None
    payload = 'null'
    result = serializer.loads(payload)
    assert result is None
    
    # Test loading boolean
    payload = 'true'
    result = serializer.loads(payload)
    assert result is True
    
    # Test that it raises an error for invalid JSON
    with pytest.raises(Exception):
        serializer.loads('invalid json')
```


# LLM-generated content at query #135
#--------------------------

```python
def test_Serializer_load_payload():
    """Test Serializer.load_payload method."""
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test loading valid JSON payload
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading with bytes serializer
    class BytesSerializer:
        def loads(self, payload):
            return {"data": payload}
        def dumps(self, obj):
            return b"test"
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    bytes_payload = b"some bytes data"
    result = bytes_serializer.load_payload(bytes_payload)
    assert result == {"data": b"some bytes data"}
    
    # Test with custom serializer passed as parameter
    class CustomSerializer:
        def loads(self, payload):
            return {"custom": payload}
        def dumps(self, obj):
            return b"custom"
    
    custom_serializer = Serializer("secret-key")
    custom_payload = b"custom data"
    result = custom_serializer.load_payload(custom_payload, serializer=CustomSerializer())
    assert result == {"custom": b"custom data"}
    
    # Test BadPayload exception for invalid payload
    import pytest
    from itsdangerous.exc import BadPayload
    
    with pytest.raises(BadPayload):
        serializer.load_payload(b"invalid json")
    
    # Test with empty payload
    with pytest.raises(BadPayload):
        serializer.load_payload(b"")
    
    # Test with None payload (should fail since want_bytes expects bytes)
    with pytest.raises(BadPayload):
        serializer.load_payload(None)
    
    # Test with text serializer that requires UTF-8 decoding
    class TextSerializer:
        def loads(self, payload):
            return {"text": payload}
        def dumps(self, obj):
            return "text"
    
    text_serializer = Serializer("secret-key", serializer=TextSerializer())
    text_payload = b"hello world"
    result = text_serializer.load_payload(text_payload)
    assert result == {"text": "hello world"}
```


# LLM-generated content at query #136
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.count(".") == 2  # Has payload and signature parts
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
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
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    assert isinstance(result, str)
    
    # Verify the signed payload can be loaded back
    loaded = serializer.loads(result)
    assert loaded == {"key": "value"}
    
    # Test with empty dict
    result_empty = serializer.dumps({})
    assert isinstance(result_empty, str)
    loaded_empty = serializer.loads(result_empty)
    assert loaded_empty == {}


# LLM-generated content at query #137
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a concrete implementation that follows the protocol
    class TestSerializer:
        def loads(self, payload):
            return payload
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with a simple dict
    test_data = {"key": "value"}
    result = serializer.dumps(test_data)
    assert result == '{"key": "value"}'
    assert isinstance(result, str)
    
    # Test with a list
    test_list = [1, 2, 3]
    result = serializer.dumps(test_list)
    assert result == "[1, 2, 3]"
    assert isinstance(result, str)
    
    # Test with None
    result = serializer.dumps(None)
    assert result == "null"
    assert isinstance(result, str)
    
    # Test with a string
    result = serializer.dumps("hello")
    assert result == '"hello"'
    assert isinstance(result, str)
    
    # Test roundtrip
    original = {"nested": {"a": 1, "b": [1, 2, 3]}, "value": None}
    dumped = serializer.dumps(original)
    loaded = serializer.loads(dumped)
    assert loaded == original
```


# LLM-generated content at query #138
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple json serializer
    serializer = json
    pdata_serializer = _PDataSerializer()
    
    # Test loading a valid payload
    payload = '{"key": "value"}'
    result = pdata_serializer.loads(payload)
    assert result == {"key": "value"}
    
    # Test loading an integer
    payload = "42"
    result = pdata_serializer.loads(payload)
    assert result == 42
    
    # Test loading a list
    payload = "[1, 2, 3]"
    result = pdata_serializer.loads(payload)
    assert result == [1, 2, 3]
    
    # Test loading None
    payload = "null"
    result = pdata_serializer.loads(payload)
    assert result is None
    
    # Test loading boolean
    payload = "true"
    result = pdata_serializer.loads(payload)
    assert result is True
    
    # Test loading float
    payload = "3.14"
    result = pdata_serializer.loads(payload)
    assert result == 3.14
    
    # Test loading empty string
    payload = '""'
    result = pdata_serializer.loads(payload)
    assert result == ""
```


# LLM-generated content at query #139
#--------------------------

```python
def test_Serializer_load_payload():
    """Test load_payload method with various scenarios."""
    
    # Test with default json serializer (text)
    s = Serializer("secret-key")
    
    # Test loading valid payload
    payload = b'{"key": "value"}'
    result = s.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test loading with explicit serializer parameter
    import json
    result = s.load_payload(payload, serializer=json)
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            import pickle
            return pickle.dumps(obj)
        def loads(self, payload):
            import pickle
            return pickle.loads(payload)
    
    s_bytes = Serializer("secret-key", serializer=BytesSerializer())
    payload_bytes = BytesSerializer().dumps({"key": "value"})
    result = s_bytes.load_payload(payload_bytes)
    assert result == {"key": "value"}
    
    # Test with invalid payload - should raise BadPayload
    import pytest
    from itsdangerous.exc import BadPayload
    
    with pytest.raises(BadPayload) as excinfo:
        s.load_payload(b"invalid json")
    assert "Could not load the payload" in str(excinfo.value)
    assert excinfo.value.original_error is not None
    
    # Test with empty payload
    with pytest.raises(BadPayload):
        s.load_payload(b"")
    
    # Test with text serializer returning unicode
    class TextSerializer:
        def dumps(self, obj):
            import json
            return json.dumps(obj)
        def loads(self, payload):
            import json
            return json.loads(payload)
    
    s_text = Serializer("secret-key", serializer=TextSerializer())
    result = s_text.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with custom serializer that might raise different exceptions
    class FailingSerializer:
        def dumps(self, obj):
            return "test"
        def loads(self, payload):
            raise ValueError("Custom error")
    
    s_fail = Serializer("secret-key", serializer=FailingSerializer())
    with pytest.raises(BadPayload) as excinfo:
        s_fail.load_payload(b"test")
    assert "Custom error" in str(excinfo.value.original_error)
    
    # Test that payload is always bytes
    with pytest.raises(AttributeError):
        s.load_payload("not bytes")  # type: ignore
    
    # Test with None serializer parameter (should use instance's serializer)
    result = s.load_payload(b'{"a": 1}', serializer=None)
    assert result == {"a": 1}
```


# LLM-generated content at query #140
#--------------------------

```python
def test_Serializer_iter_unsigners():
    secret_key = b"secret-key"
    salt = b"my-salt"
    
    # Test with no fallback signers
    serializer = Serializer(secret_key, salt=salt)
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [secret_key]
    assert unsigners[0].salt == salt
    
    # Test with fallback signers as dict
    class CustomSigner(Signer):
        pass
    
    serializer = Serializer(
        secret_key,
        salt=salt,
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert unsigners[0].key_derivation == "hmac"  # default signer uses default kwargs
    assert unsigners[1].key_derivation == "hmac"
    
    # Test with fallback signers as tuple of (Signer class, kwargs)
    serializer = Serializer(
        secret_key,
        salt=salt,
        fallback_signers=[(CustomSigner, {"digest_method": "sha256"})]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], CustomSigner)
    
    # Test with fallback signers as Signer class
    serializer = Serializer(
        secret_key,
        salt=salt,
        fallback_signers=[CustomSigner]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], CustomSigner)
    
    # Test with custom salt parameter
    custom_salt = b"custom-salt"
    unsigners = list(serializer.iter_unsigners(salt=custom_salt))
    for unsigner in unsigners:
        assert unsigner.salt == custom_salt
    
    # Test with multiple secret keys (key rotation)
    secret_keys = [b"old-key", b"newer-key", b"newest-key"]
    serializer = Serializer(secret_keys, salt=salt)
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 3  # one per secret key
    assert unsigners[0].secret_keys == secret_keys


# LLM-generated content at query #141
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol conforming objects can dump data."""
    # Create a simple serializer that conforms to _PDataSerializer[str]
    class TextSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TextSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result == '{"key": "value"}'
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode("utf-8"))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode("utf-8")
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == b'{"key": "value"}'
    
    # Test with various data types
    assert serializer.dumps(None) == "null"
    assert serializer.dumps(42) == "42"
    assert serializer.dumps([1, 2, 3]) == "[1, 2, 3]"
```


# LLM-generated content at query #142
#--------------------------

```python
def test_Serializer_iter_unsigners():
    # Test with default signer and no fallbacks
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret-key"]
    
    # Test with custom salt
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"
    
    # Test with fallback signers as dict
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].key_derivation == "hmac"
    
    # Test with fallback signers as tuple (signer class, kwargs)
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
    
    # Test with fallback signers as signer class
    serializer = Serializer(
        "secret-key",
        fallback_signers=[CustomSigner]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], CustomSigner)
    
    # Test with multiple secret keys
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    
    # Test with multiple secret keys and fallback signers as dict
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3  # 1 default + 2 fallbacks (one for each secret key)
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].key_derivation == "hmac"
    assert signers[2].key_derivation == "hmac"
    
    # Test with custom salt passed as argument
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners(salt="custom-salt"))
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"
    
    # Test with no salt (default salt from serializer)
    serializer = Serializer("secret-key", salt=None)
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    
    # Test with empty fallback signers
    serializer = Serializer("secret-key", fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
```


# LLM-generated content at query #143
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test basic serialization
    result = serializer.dumps({"key": "value"})
    assert result == '{"key": "value"}'
    
    # Test with list
    result = serializer.dumps([1, 2, 3])
    assert result == '[1, 2, 3]'
    
    # Test with simple types
    result = serializer.dumps("test")
    assert result == '"test"'
    
    result = serializer.dumps(42)
    assert result == '42'
    
    result = serializer.dumps(True)
    assert result == 'true'
    
    result = serializer.dumps(None)
    assert result == 'null'
    
    # Test empty structures
    result = serializer.dumps({})
    assert result == '{}'
    
    result = serializer.dumps([])
    assert result == '[]'
    
    # Test nested structures
    result = serializer.dumps({"a": {"b": [1, 2, 3]}})
    assert result == '{"a": {"b": [1, 2, 3]}}'
```


# LLM-generated content at query #144
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # contains separator between payload and signature
    
    # Test that dumps produces a valid signed payload
    payload = serializer.dumps("test_data")
    # Verify we can loads it back
    assert serializer.loads(payload) == "test_data"
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            import json
            return json.dumps(obj).encode()
        def loads(self, payload):
            import json
            return json.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    # Test with custom serializer_kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_kwargs = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    assert result_kwargs.count(":") == 2  # compact separators
    
    # Test that different data produces different signatures
    result1 = serializer.dumps("data1")
    result2 = serializer.dumps("data2")
    assert result1 != result2
    
    # Test with salt parameter
    result_with_salt = serializer.dumps("test", salt="custom_salt")
    assert isinstance(result_with_salt, str)
    # Should not be verifiable with default salt
    import pytest
    with pytest.raises(BadSignature):
        serializer.loads(result_with_salt)  # uses default salt
    
    # Test with key rotation (list of keys)
    serializer_rotation = Serializer(["old_key", "new_key"])
    result_rotation = serializer_rotation.dumps("test_data")
    assert serializer_rotation.loads(result_rotation) == "test_data"  # signed with newest key
    
    # Test that dumps returns correct type based on serializer
    assert isinstance(serializer.dumps(123), str)  # text serializer
    assert isinstance(bytes_serializer.dumps(123), bytes)  # bytes serializer
    
    # Test with empty data
    empty_result = serializer.dumps({})
    assert isinstance(empty_result, str)
    assert serializer.loads(empty_result) == {}
    
    # Test with None
    none_result = serializer.dumps(None)
    assert serializer.loads(none_result) is None
```


# LLM-generated content at query #145
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert "." in result  # Contains signature separator
    
    # Verify the result can be loaded back correctly
    loaded = serializer.loads(result)
    assert loaded == data
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    data = {"key": "value"}
    result = bytes_serializer.dumps(data)
    assert isinstance(result, bytes)
    assert b"." in result
    
    # Verify bytes result can be loaded back
    loaded = bytes_serializer.loads(result)
    assert loaded == data
    
    # Test with custom salt
    salt = b"custom-salt"
    result_with_salt = serializer.dumps(data, salt=salt)
    assert result_with_salt != result  # Different salt produces different signature
    
    # Test with serializer_kwargs
    class JSONSerializerWithKwargs:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj, **kwargs)
        def loads(self, payload):
            return json.loads(payload)
    
    serializer_kwargs = Serializer(
        "secret-key",
        serializer=JSONSerializerWithKwargs(),
        serializer_kwargs={"sort_keys": True}
    )
    data = {"b": 2, "a": 1}
    result = serializer_kwargs.dumps(data)
    loaded = serializer_kwargs.loads(result)
    assert loaded == data
    
    # Test empty data
    result_empty = serializer.dumps({})
    assert isinstance(result_empty, str)
    loaded_empty = serializer.loads(result_empty)
    assert loaded_empty == {}
```


# LLM-generated content at query #146
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a text serializer (json)
    text_serializer = _PDataSerializer[str]()
    text_serializer.loads = lambda x: json.loads(x)
    
    # Test with valid JSON payload
    assert text_serializer.loads('{"key": "value"}') == {"key": "value"}
    assert text_serializer.loads('[1, 2, 3]') == [1, 2, 3]
    assert text_serializer.loads('"string"') == "string"
    assert text_serializer.loads('42') == 42
    assert text_serializer.loads('true') == True
    assert text_serializer.loads('null') is None
    
    # Test with a bytes serializer
    bytes_serializer = _PDataSerializer[bytes]()
    bytes_serializer.loads = lambda x: json.loads(x.decode('utf-8'))
    
    # Test with valid bytes payload
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'[1, 2, 3]') == [1, 2, 3]
    
    # Test with custom serializer protocol
    class CustomSerializer:
        def loads(self, payload):
            return payload.upper()
            
    custom_serializer = _PDataSerializer[str]()
    custom_serializer.loads = CustomSerializer().loads
    assert custom_serializer.loads("hello") == "HELLO"
    
    # Test protocol compliance - verify it accepts both str and bytes based on type parameter
    assert _PDataSerializer[str].__bound__ == str | bytes
    assert _PDataSerializer[bytes].__bound__ == str | bytes
```


# LLM-generated content at query #147
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a basic serializer that returns a simple value
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test loading valid JSON
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading a string
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test loading a number
    result = serializer.loads('42')
    assert result == 42
    
    # Test loading null
    result = serializer.loads('null')
    assert result is None
    
    # Test loading True/False
    result = serializer.loads('true')
    assert result is True
    
    result = serializer.loads('false')
    assert result is False
    
    # Test with a custom serializer that returns bytes
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    result = bytes_serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with a serializer that returns a custom type
    class CustomType:
        def __init__(self, value):
            self.value = value
    
    class CustomSerializer:
        def loads(self, payload: str) -> CustomType:
            return CustomType(payload)
        
        def dumps(self, obj: CustomType) -> str:
            return str(obj.value)
    
    custom_serializer = CustomSerializer()
    result = custom_serializer.loads("test_data")
    assert isinstance(result, CustomType)
    assert result.value == "test_data"

```


# LLM-generated content at query #148
#--------------------------

```python
def test__PDataSerializer_dumps():
    """Test that _PDataSerializer protocol's dumps method works correctly."""
    # Create a concrete implementation of _PDataSerializer for testing
    class TestSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = TestSerializer()
    
    # Test with simple data
    test_data = {"key": "value", "number": 42}
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    assert result == '{"key": "value", "number": 42}'
    
    # Test with list data
    test_list = [1, 2, 3, "test"]
    result = serializer.dumps(test_list)
    assert isinstance(result, str)
    assert result == '[1, 2, 3, "test"]'
    
    # Test with primitive data
    test_primitive = "hello"
    result = serializer.dumps(test_primitive)
    assert isinstance(result, str)
    assert result == '"hello"'
    
    # Test with None
    result = serializer.dumps(None)
    assert isinstance(result, str)
    assert result == 'null'
```


# LLM-generated content at query #149
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    
    # Test with salt parameter
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert result_with_salt != result  # Different salt should produce different signature
    
    # Test with different secret keys
    serializer2 = Serializer("different-secret")
    result2 = serializer2.dumps({"key": "value"})
    assert result != result2  # Different keys should produce different signatures
    
    # Test serialization of various data types
    assert isinstance(serializer.dumps(123), str)
    assert isinstance(serializer.dumps("string"), str)
    assert isinstance(serializer.dumps([1, 2, 3]), str)
    assert isinstance(serializer.dumps(None), str)


# LLM-generated content at query #150
#--------------------------

```python
def test_Serializer_iter_unsigners():
    """Test iter_unsigners method returns correct signers."""
    # Setup
    secret_key = b"test-secret-key"
    salt = b"test-salt"
    signer_kwargs = {"digest_method": "sha256"}
    
    # Create serializer with fallback signers
    fallback_signers = [
        {"digest_method": "sha512"},
        (Signer, {"digest_method": "sha384"}),
        Signer,
    ]
    
    serializer = Serializer(
        secret_key=secret_key,
        salt=salt,
        signer_kwargs=signer_kwargs,
        fallback_signers=fallback_signers,
    )
    
    # Test iter_unsigners yields correct signers
    signers = list(serializer.iter_unsigners())
    
    # First signer should be the main signer
    assert len(signers) == 1 + len(serializer.secret_keys) * len(fallback_signers)
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == serializer.secret_keys
    assert signers[0].salt == salt
    assert signers[0].digest_method == "sha256"
    
    # Test with custom salt
    custom_salt = b"custom-salt"
    signers_with_custom_salt = list(serializer.iter_unsigners(salt=custom_salt))
    assert signers_with_custom_salt[0].salt == custom_salt
    
    # Test fallback signers are yielded correctly
    for i, signer in enumerate(signers[1:], 1):
        assert isinstance(signer, Signer)
        # Each fallback signer should use one of the secret keys
        assert signer.secret_keys == [serializer.secret_keys[i % len(serializer.secret_keys)]]
    
    # Test with no fallback signers
    serializer_no_fallback = Serializer(
        secret_key=secret_key,
        salt=salt,
        fallback_signers=[],
    )
    signers_no_fallback = list(serializer_no_fallback.iter_unsigners())
    assert len(signers_no_fallback) == 1
    
    # Test with default fallback signers
    serializer_default_fallback = Serializer(
        secret_key=secret_key,
        salt=salt,
    )
    signers_default = list(serializer_default_fallback.iter_unsigners())
    assert len(signers_default) == 1 + len(serializer_default_fallback.secret_keys) * len(serializer_default_fallback.default_fallback_signers)
```


# LLM-generated content at query #151
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON serializer
    serializer = json
    data_serializer = _PDataSerializer()
    data_serializer.loads = serializer.loads
    data_serializer.dumps = serializer.dumps
    
    # Test loading valid JSON
    payload = '{"key": "value"}'
    result = data_serializer.loads(payload)
    assert result == {"key": "value"}
    
    # Test loading a list
    payload = '[1, 2, 3]'
    result = data_serializer.loads(payload)
    assert result == [1, 2, 3]
    
    # Test loading a string
    payload = '"hello"'
    result = data_serializer.loads(payload)
    assert result == "hello"
    
    # Test loading a number
    payload = '42'
    result = data_serializer.loads(payload)
    assert result == 42
    
    # Test loading null
    payload = 'null'
    result = data_serializer.loads(payload)
    assert result is None
    
    # Test loading boolean
    payload = 'true'
    result = data_serializer.loads(payload)
    assert result is True
    
    # Test loading empty object
    payload = '{}'
    result = data_serializer.loads(payload)
    assert result == {}
    
    # Test loading with a custom serializer that returns different types
    class CustomSerializer:
        def loads(self, payload):
            return payload.upper()
        def dumps(self, obj):
            return str(obj)
    
    custom = CustomSerializer()
    data_serializer.loads = custom.loads
    result = data_serializer.loads("hello")
    assert result == "HELLO"
```


# LLM-generated content at query #152
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default serializer (json) and bytes output
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    
    # Test that dumps produces a string that can be loaded back
    loaded = serializer.loads(result)
    assert loaded == {"key": "value"}
    
    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result_with_salt, str)
    assert result != result_with_salt  # Different salt should produce different signature
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            import json
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            import json
            return json.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    loaded_bytes = bytes_serializer.loads(result_bytes)
    assert loaded_bytes == {"key": "value"}
    
    # Test empty payload
    result_empty = serializer.dumps({})
    assert isinstance(result_empty, str)
    loaded_empty = serializer.loads(result_empty)
    assert loaded_empty == {}
    
    # Test with list payload
    result_list = serializer.dumps([1, 2, 3])
    assert isinstance(result_list, str)
    loaded_list = serializer.loads(result_list)
    assert loaded_list == [1, 2, 3]
    
    # Test with None payload
    result_none = serializer.dumps(None)
    assert isinstance(result_none, str)
    loaded_none = serializer.loads(result_none)
    assert loaded_none is None
    
    # Test with custom serializer_kwargs
    class CustomJSONSerializer:
        def dumps(self, obj, **kwargs):
            import json
            return json.dumps(obj, **kwargs)
        def loads(self, payload):
            import json
            return json.loads(payload)
    
    custom_serializer = Serializer("secret-key", serializer=CustomJSONSerializer(), serializer_kwargs={"indent": 2})
    result_custom = custom_serializer.dumps({"key": "value"})
    assert isinstance(result_custom, str)
    # Should contain newlines due to indent
    assert "\n" in result_custom
    
    # Verify the payload part is indented before the signature
    assert '  "key"' in result_custom


# LLM-generated content at query #153
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default JSON serializer (text serializer)
    serializer = Serializer("secret-key")
    data = {"key": "value"}
    result = serializer.dumps(data)
    
    # Should return a string (text serializer)
    assert isinstance(result, str)
    # Should contain the serialized payload
    assert "." in result  # Signed format contains dots
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps(data)
    
    # Should return bytes
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    # Test with None salt
    result_no_salt = serializer.dumps(data, salt=None)
    assert isinstance(result_no_salt, str)
    
    # Test with custom salt
    result_custom_salt = serializer.dumps(data, salt="custom-salt")
    assert isinstance(result_custom_salt, str)
    
    # Different salt should produce different signature
    assert result_no_salt != result_custom_salt
    
    # Test dumps and loads roundtrip
    signed_data = serializer.dumps(data)
    loaded_data = serializer.loads(signed_data)
    assert loaded_data == data
    
    # Test with different serializer kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True}
    )
    result_kwargs = serializer_with_kwargs.dumps({"b": 1, "a": 2})
    assert isinstance(result_kwargs, str)
    # Should have sorted keys in payload
    loaded = serializer_with_kwargs.loads(result_kwargs)
    assert loaded == {"a": 2, "b": 1}


# LLM-generated content at query #154
#--------------------------

```python
def test__PDataSerializer_dumps():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test with a simple dictionary
    test_data = {"key": "value", "number": 42}
    result = serializer.dumps(test_data)
    assert isinstance(result, str)
    assert result == '{"key": "value", "number": 42}'
    
    # Test with a list
    test_list = [1, 2, 3, "hello"]
    result = serializer.dumps(test_list)
    assert isinstance(result, str)
    assert result == '[1, 2, 3, "hello"]'
    
    # Test with None
    result = serializer.dumps(None)
    assert isinstance(result, str)
    assert result == "null"
    
    # Test with a string
    result = serializer.dumps("test_string")
    assert isinstance(result, str)
    assert result == '"test_string"'
    
    # Test with an integer
    result = serializer.dumps(123)
    assert isinstance(result, str)
    assert result == "123"
    
    # Test with a boolean
    result = serializer.dumps(True)
    assert isinstance(result, str)
    assert result == "true"


# LLM-generated content at query #155
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer protocol's loads method works with different implementations."""
    # Test with JSON serializer (text serializer)
    json_serializer = json
    result = json_serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test with a custom text serializer that implements the protocol
    class CustomTextSerializer:
        def loads(self, payload: str) -> t.Any:
            return payload.upper()
        
        def dumps(self, obj: t.Any) -> str:
            return str(obj).lower()
    
    custom_text = CustomTextSerializer()
    result = custom_text.loads("hello")
    assert result == "HELLO"
    
    # Test with a custom bytes serializer that implements the protocol
    class CustomBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return payload.decode("utf-8").upper()
        
        def dumps(self, obj: t.Any) -> bytes:
            return str(obj).encode("utf-8")
    
    custom_bytes = CustomBytesSerializer()
    result = custom_bytes.loads(b"hello")
    assert result == "HELLO"
    
    # Test that loads handles various data types
    # Integer
    assert json_serializer.loads("42") == 42
    
    # List
    assert json_serializer.loads("[1, 2, 3]") == [1, 2, 3]
    
    # Nested structure
    nested = json_serializer.loads('{"a": {"b": [1, 2]}}')
    assert nested == {"a": {"b": [1, 2]}}
    
    # None
    assert json_serializer.loads("null") is None
    
    # Boolean
    assert json_serializer.loads("true") is True
    assert json_serializer.loads("false") is False
    
    # Test that loads raises appropriate exceptions for invalid input
    import pytest
    with pytest.raises(json.JSONDecodeError):
        json_serializer.loads("invalid json")
    
    with pytest.raises(json.JSONDecodeError):
        json_serializer.loads("{broken}")
```


# LLM-generated content at query #156
#--------------------------

```python
def test_Serializer_dumps():
    # Test with default json serializer (text serializer)
    serializer = Serializer("secret-key")
    
    # Test basic serialization
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "." in result  # Should contain signature separator
    
    # Test that dumps output can be loaded back
    loaded = serializer.loads(result)
    assert loaded == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode("utf-8")
        
        def loads(self, payload):
            return json.loads(payload)
    
    bytes_serializer = Serializer("secret-key", serializer=BytesSerializer())
    result_bytes = bytes_serializer.dumps({"key": "value"})
    assert isinstance(result_bytes, bytes)
    assert b"." in result_bytes
    
    loaded_bytes = bytes_serializer.loads(result_bytes)
    assert loaded_bytes == {"key": "value"}
    
    # Test with custom salt
    result_with_salt = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert result_with_salt != result
    
    # Test with serializer kwargs
    serializer_with_kwargs = Serializer(
        "secret-key",
        serializer_kwargs={"sort_keys": True, "separators": (",", ":")}
    )
    result_kwargs = serializer_with_kwargs.dumps({"b": 2, "a": 1})
    assert result_kwargs is not None
    # Verify compact JSON output (no spaces)
    assert ", " not in result_kwargs
    
    # Test with list of secret keys (key rotation)
    serializer_rotation = Serializer(["old-key", "new-key"])
    result_rotation = serializer_rotation.dumps("test data")
    
    # Should be able to verify with both keys
    loaded_rotation = serializer_rotation.loads(result_rotation)
    assert loaded_rotation == "test data"
    
    # Test that dumps returns correct type for text serializer
    assert isinstance(serializer.dumps(42), str)
    
    # Test serializing various data types
    assert serializer.loads(serializer.dumps(None)) is None
    assert serializer.loads(serializer.dumps([1, 2, 3])) == [1, 2, 3]
    assert serializer.loads(serializer.dumps("hello")) == "hello"
    
    # Test that different secret keys produce different signatures
    s1 = Serializer("key1")
    s2 = Serializer("key2")
    data = {"test": True}
    assert s1.dumps(data) != s2.dumps(data)
```


# LLM-generated content at query #157
#--------------------------

```python
def test_Serializer_load_payload():
    """Test load_payload method of Serializer class."""
    
    # Test with default JSON serializer and text input
    serializer = Serializer("test-secret")
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
    
    # Test with custom serializer that returns text
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)
        
        def loads(self, payload):
            return eval(payload)
    
    text_serializer = TextSerializer()
    serializer_text = Serializer("test-secret", serializer=text_serializer)
    payload = "{'key': 'value'}"
    result = serializer_text.load_payload(payload.encode("utf-8"))
    assert result == {"key": "value"}
    
    # Test with bytes serializer
    class BytesSerializer:
        def dumps(self, obj):
            return obj.encode() if isinstance(obj, str) else str(obj).encode()
        
        def loads(self, payload):
            return payload.decode()
    
    bytes_serializer = BytesSerializer()
    serializer_bytes = Serializer("test-secret", serializer=bytes_serializer)
    payload = b"hello"
    result = serializer_bytes.load_payload(payload)
    assert result == "hello"
    
    # Test with custom serializer passed as parameter
    class CustomSerializer:
        def dumps(self, obj):
            return f"custom:{obj}"
        
        def loads(self, payload):
            return payload.replace("custom:", "")
    
    custom_serializer = CustomSerializer()
    serializer = Serializer("test-secret")
    payload = b"custom:data"
    result = serializer.load_payload(payload, serializer=custom_serializer)
    assert result == "data"
    
    # Test with BadPayload exception for invalid payload
    serializer = Serializer("test-secret")
    try:
        serializer.load_payload(b"invalid json")
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert e.original_error is not None
    
    # Test with empty payload
    serializer = Serializer("test-secret")
    try:
        serializer.load_payload(b"")
        assert False, "Should have raised BadPayload"
    except BadPayload:
        pass
    
    # Test with integer payload (should work with JSON serializer)
    serializer = Serializer("test-secret")
    payload = json.dumps(42).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == 42
    
    # Test with list payload
    serializer = Serializer("test-secret")
    payload = json.dumps([1, 2, 3]).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == [1, 2, 3]
    
    # Test with None payload
    serializer = Serializer("test-secret")
    payload = json.dumps(None).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result is None
    
    # Test with boolean payload
    serializer = Serializer("test-secret")
    payload = json.dumps(True).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result is True
    
    # Test with nested objects
    serializer = Serializer("test-secret")
    nested = {"a": {"b": [1, 2, {"c": "d"}]}}
    payload = json.dumps(nested).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == nested
    
    # Test with special characters
    serializer = Serializer("test-secret")
    special = {"text": "héllo wörld 🎉"}
    payload = json.dumps(special).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == special
    
    # Test with very large payload
    serializer = Serializer("test-secret")
    large_data = {"key": "x" * 10000}
    payload = json.dumps(large_data).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == large_data
    
    # Test with unicode characters in payload
    serializer = Serializer("test-secret")
    unicode_data = {"message": "こんにちは世界"}
    payload = json.dumps(unicode_data).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == unicode_data
    
    # Test that is_text_serializer flag affects behavior
    class FakeSerializer:
        def dumps(self, obj):
            return "text"
        
        def loads(self, payload):
            return f"loaded: {payload}"
    
    fake_serializer = FakeSerializer()
    serializer = Serializer("test-secret", serializer=fake_serializer)
    result = serializer.load_payload(b"test payload")
    assert result == "loaded: test payload"
```


# LLM-generated content at query #158
#--------------------------

```python
def test__PDataSerializer_loads():
    # Create a mock serializer that implements the _PDataSerializer protocol
    class MockSerializer:
        def loads(self, payload):
            return json.loads(payload)
        
        def dumps(self, obj):
            return json.dumps(obj)
    
    serializer = MockSerializer()
    
    # Test loading a JSON string
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a list
    result = serializer.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading a simple value
    result = serializer.loads('"hello"')
    assert result == "hello"
    
    # Test loading an integer
    result = serializer.loads('42')
    assert result == 42
    
    # Test loading with bytes input
    result = serializer.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading empty JSON
    result = serializer.loads('null')
    assert result is None
    
    # Test loading invalid JSON raises exception
    import pytest
    with pytest.raises(Exception):
        serializer.loads('invalid json')
```


# LLM-generated content at query #159
#--------------------------

```python
def test__PDataSerializer_loads():
    """Test that _PDataSerializer loads method works correctly with string and bytes payloads."""
    # Test with string serializer
    class StringSerializer:
        def loads(self, payload: str) -> t.Any:
            return json.loads(payload)
        
        def dumps(self, obj: t.Any) -> str:
            return json.dumps(obj)
    
    string_serializer = StringSerializer()
    assert string_serializer.loads('{"key": "value"}') == {"key": "value"}
    assert string_serializer.loads('42') == 42
    assert string_serializer.loads('null') is None
    
    # Test with bytes serializer
    class BytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return json.loads(payload.decode('utf-8'))
        
        def dumps(self, obj: t.Any) -> bytes:
            return json.dumps(obj).encode('utf-8')
    
    bytes_serializer = BytesSerializer()
    assert bytes_serializer.loads(b'{"key": "value"}') == {"key": "value"}
    assert bytes_serializer.loads(b'42') == 42
    assert bytes_serializer.loads(b'null') is None
    
    # Test that loads is callable and returns correct type
    assert callable(string_serializer.loads)
    assert callable(bytes_serializer.loads)
    
    # Test with empty payloads
    assert string_serializer.loads('{}') == {}
    assert bytes_serializer.loads(b'{}') == {}
    
    # Test with complex nested structures
    complex_payload = {"nested": {"list": [1, 2, 3], "bool": True}}
    assert string_serializer.loads(json.dumps(complex_payload)) == complex_payload
    assert bytes_serializer.loads(json.dumps(complex_payload).encode('utf-8')) == complex_payload
```


# LLM-generated content at query #160
#--------------------------

```python
def test__PDataSerializer_loads():
    # Test with a simple JSON serializer
    serializer = json
    pdata = _PDataSerializer()
    
    # Test loading a JSON string
    result = pdata.loads('{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading a simple value
    result = pdata.loads('42')
    assert result == 42
    
    # Test loading a list
    result = pdata.loads('[1, 2, 3]')
    assert result == [1, 2, 3]
    
    # Test loading with bytes input
    result = pdata.loads(b'{"key": "value"}')
    assert result == {"key": "value"}
    
    # Test loading empty JSON
    result = pdata.loads('null')
    assert result is None
    
    # Test loading boolean
    result = pdata.loads('true')
    assert result is True
    
    # Test loading with custom serializer
    class CustomSerializer:
        def loads(self, payload):
            return payload.upper()
    
    custom_serializer = CustomSerializer()
    result = pdata.loads("hello", serializer=custom_serializer)
    assert result == "HELLO"
    
    # Test that BadPayload is raised for invalid JSON
    with pytest.raises(BadPayload):
        pdata.loads('invalid json')
```


