####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_dump_payload_without_compression():
    serializer = URLSafeSerializerMixin()
    payload = serializer.dump_payload({"key": "value"})
    assert isinstance(payload, bytes)
    assert not payload.startswith(b".")

def test_dump_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    large_data = "x" * 1000
    payload = serializer.dump_payload({"data": large_data})
    assert isinstance(payload, bytes)
    assert payload.startswith(b".")

def test_dump_payload_compression_threshold():
    serializer = URLSafeSerializerMixin()
    small_data = "x"
    payload = serializer.dump_payload({"data": small_data})
    assert isinstance(payload, bytes)
    assert not payload.startswith(b".")


# LLM-generated content at query #2
#--------------------------

```python
def test_load_payload_uncompressed_valid():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_compressed_valid():
    serializer = URLSafeSerializerMixin()
    compressed = zlib.compress(b'{"key":"value"}')
    payload = b"." + base64_encode(compressed)
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64_raises_badpayload():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid_base64")
        assert False
    except BadPayload:
        pass

def test_load_payload_compressed_corrupt_zlib_raises_badpayload():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not_zlib_compressed")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_empty_string():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b"")
    result = serializer.load_payload(payload)
    assert result is None

def test_load_payload_none_serializer():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b"null")
    result = serializer.load_payload(payload)
    assert result is None
```


# LLM-generated content at query #3
#--------------------------

```python
def test_dump_payload_compression_triggered():
    serializer = URLSafeSerializerMixin()
    obj = "a" * 1000
    result = serializer.dump_payload(obj)
    assert result.startswith(b".")```


# LLM-generated content at query #4
#--------------------------

def test_load_payload_non_compressed_non_decompressed_payload():
    serializer = URLSafeSerializerMixin()
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)


# LLM-generated content at query #5
#--------------------------

```python
def test_base64_decode_does_not_raise_exception_for_valid_payload():
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    serializer = URLSafeSerializerMixin()
    payload = b"eyJhIjogMX0"  # valid base64 encoded JSON
    result = serializer.load_payload(payload)
    assert result == {"a": 1}  # predicate at line 16 evaluates to False for valid payload
```


# LLM-generated content at query #6
#--------------------------

def test_load_payload_normal_base64_no_compression():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key": "value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_compressed_base64():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(zlib.compress(b'{"key": "value"}'))
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_invalid_compressed_data():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not compressed")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_empty_payload():
    serializer = URLSafeSerializerMixin()
    result = serializer.load_payload(base64_encode(b""))
    assert result is None or result == b""


# LLM-generated content at query #7
#--------------------------

def test_dump_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    payload = serializer.dump_payload({"key": "value"})
    assert payload.startswith(b"ey")  # base64 encoded json without compression

def test_dump_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    long_data = "x" * 1000
    payload = serializer.dump_payload({"data": long_data})
    assert payload.startswith(b".")  # compressed data should have dot prefix

def test_dump_payload_compressed_length_check():
    serializer = URLSafeSerializerMixin()
    short_data = "short"
    payload = serializer.dump_payload({"data": short_data})
    assert not payload.startswith(b".")  # no compression for short data


# LLM-generated content at query #8
#--------------------------

def test_load_payload_no_compression():
    mock_serializer = type("MockSerializer", (object,), {"load_payload": lambda self, payload, *args, **kwargs: {"key": "value"}})()
    url_safe_mixin = URLSafeSerializerMixin()
    url_safe_mixin.serializer = mock_serializer
    payload = base64_encode(b'{"key": "value"}')
    result = url_safe_mixin.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    mock_serializer = type("MockSerializer", (object,), {"load_payload": lambda self, payload, *args, **kwargs: {"key": "value"}})()
    url_safe_mixin = URLSafeSerializerMixin()
    url_safe_mixin.serializer = mock_serializer
    compressed = zlib.compress(b'{"key": "value"}')
    payload = b"." + base64_encode(compressed)
    result = url_safe_mixin.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    mock_serializer = type("MockSerializer", (object,), {"load_payload": lambda self, payload, *args, **kwargs: {}})()
    url_safe_mixin = URLSafeSerializerMixin()
    url_safe_mixin.serializer = mock_serializer
    try:
        url_safe_mixin.load_payload(b"invalid!")
        assert False
    except BadPayload:
        pass

def test_load_payload_bad_compression():
    mock_serializer = type("MockSerializer", (object,), {"load_payload": lambda self, payload, *args, **kwargs: {}})()
    url_safe_mixin = URLSafeSerializerMixin()
    url_safe_mixin.serializer = mock_serializer
    payload = b"." + base64_encode(b"not compressed")
    try:
        url_safe_mixin.load_payload(payload)
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #9
#--------------------------

def test_dump_payload_compression_triggered():
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import want_bytes
    import zlib

    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass

    serializer = TestSerializer("secret")
    obj = "a" * 1000
    payload = serializer.dump_payload(obj)
    assert payload.startswith(b".")


# LLM-generated content at query #10
#--------------------------

def test_dump_payload_compression_triggered():
    serializer = URLSafeSerializerMixin(secret_key="test")
    obj = "a" * 1000
    result = serializer.dump_payload(obj)
    assert result.startswith(b".")


# LLM-generated content at query #11
#--------------------------

def test_load_payload_does_not_raise_bad_payload_on_valid_base64():
    serializer = URLSafeSerializerMixin()
    serializer.load_payload(b"eyJhIjoiYiJ9")


# LLM-generated content at query #12
#--------------------------

```python
def test_load_payload_with_valid_compressed_payload_does_not_raise():
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import base64_encode
    import zlib
    
    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass
    
    serializer = TestSerializer("secret")
    obj = {"test": "data"}
    payload = serializer.dump_payload(obj)
    # payload starts with b'.' and is compressed and base64 encoded
    # This should not raise BadPayload at line 25
    result = serializer.load_payload(payload)
```


# LLM-generated content at query #13
#--------------------------

def test_load_payload_compressed_invalid_zlib_does_not_raise():
    serializer = URLSafeSerializerMixin("secret")
    payload = b".aW52YWxpZA=="
    try:
        result = serializer.load_payload(payload)
        assert False  # Should not reach here
    except BadPayload:
        pass  # Expected, the predicate at line 25 should evaluate to False because decompress is True but zlib.decompress raises


# LLM-generated content at query #14
#--------------------------

def test_load_payload_compressed_with_invalid_zlib_data_raises_badpayload():
    import zlib
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.exc import BadPayload
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import base64_encode

    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass

    serializer = TestSerializer("secret")
    payload = b"." + base64_encode(b"invalid-zlib-data")
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass


# LLM-generated content at query #15
#--------------------------

def test_load_payload_base64_decode_success():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"a":1}')
    result = serializer.load_payload(payload)
    assert result == {"a": 1}


# LLM-generated content at query #16
#--------------------------

def test_load_payload_normal_base64_no_decompress():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"a":1}')
    result = serializer.load_payload(payload)
    assert result == {"a": 1}

def test_load_payload_compressed_with_dot_prefix():
    serializer = URLSafeSerializerMixin()
    json_data = b'{"a":1}'
    compressed = zlib.compress(json_data)
    base64d = base64_encode(compressed)
    payload = b"." + base64d
    result = serializer.load_payload(payload)
    assert result == {"a": 1}

def test_load_payload_invalid_base64_raises_badpayload():
    serializer = URLSafeSerializerMixin()
    payload = b"invalid!!!"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload"
    except BadPayload:
        pass

def test_load_payload_corrupted_compressed_data_raises_badpayload():
    serializer = URLSafeSerializerMixin()
    compressed = zlib.compress(b'{"a":1}')
    base64d = base64_encode(compressed)
    corrupted_base64d = base64d[:-1]
    payload = b"." + corrupted_base64d
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload"
    except BadPayload:
        pass

def test_load_payload_empty_payload_raises_badpayload():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"")
        assert False, "Expected BadPayload"
    except BadPayload:
        pass


# LLM-generated content at query #17
#--------------------------

def test_load_payload_with_valid_base64_no_compression_does_not_raise_bad_payload():
    serializer = URLSafeSerializerMixin("secret")
    payload = serializer.dump_payload({"key": "value"})
    if payload.startswith(b"."):
        payload = payload[1:]
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #18
#--------------------------

def test_load_payload_no_decompression_on_valid_compressed_payload():
    serializer = URLSafeSerializerMixin("secret")
    payload = serializer.dump_payload({"key": "value"})
    # Ensure payload starts with b"." indicating compression
    assert payload.startswith(b".")
    # Remove the dot to simulate a payload that was compressed but without decompression flag
    payload_without_dot = payload[1:]
    # This should not raise BadPayload decompression error
    result = serializer.load_payload(payload_without_dot)
    assert result == {"key": "value"}


# LLM-generated content at query #19
#--------------------------

def test_load_payload_no_decompress_valid_base64():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key": "value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_decompress_valid_base64():
    serializer = URLSafeSerializerMixin()
    json_data = b'{"key": "value"}'
    compressed = zlib.compress(json_data)
    payload = b"." + base64_encode(compressed)
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64_raises_bad_payload():
    serializer = URLSafeSerializerMixin()
    payload = b"invalid_base64"
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_decompress_fails_raises_bad_payload():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"invalid_compressed_data")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #20
#--------------------------

def test_load_payload_no_decompress_after_base64_decode():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)


# LLM-generated content at query #21
#--------------------------

def test_load_payload_normal_base64():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_compressed():
    serializer = URLSafeSerializerMixin()
    compressed = zlib.compress(b'{"key":"value"}')
    payload = b"." + base64_encode(compressed)
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    import pytest
    try:
        serializer.load_payload(b"invalid!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_invalid_compressed():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not compressed")
    import pytest
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_load_payload_without_compression():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    compressed = zlib.compress(b'{"key":"value"}')
    payload = b"." + base64_encode(compressed)
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid!")
    except BadPayload:
        pass

def test_load_payload_invalid_compression():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not compressed data")
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass

def test_load_payload_empty_payload():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"")
    except BadPayload:
        pass


# LLM-generated content at query #2
#--------------------------

def test_dump_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    serializer.dump_payload = lambda obj: super(URLSafeSerializerMixin, serializer).dump_payload(obj)
    result = serializer.dump_payload("test")
    assert isinstance(result, bytes)
    assert result == base64_encode(serializer.default_serializer.dumps("test"))

def test_dump_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    long_string = "x" * 1000
    result = serializer.dump_payload(long_string)
    assert isinstance(result, bytes)
    assert result.startswith(b".")
    assert base64_decode(result[1:]) == zlib.compress(serializer.default_serializer.dumps(long_string))

def test_dump_payload_compression_not_applied_when_not_beneficial():
    serializer = URLSafeSerializerMixin()
    short_string = "short"
    result = serializer.dump_payload(short_string)
    assert isinstance(result, bytes)
    assert not result.startswith(b".")
    assert result == base64_encode(serializer.default_serializer.dumps(short_string))

def test_dump_payload_compression_boundary():
    serializer = URLSafeSerializerMixin()
    json_data = serializer.default_serializer.dumps("test")
    compressed = zlib.compress(json_data)
    assert len(compressed) >= len(json_data) - 1
    result = serializer.dump_payload("test")
    assert not result.startswith(b".")
    assert result == base64_encode(json_data)


# LLM-generated content at query #3
#--------------------------

def test_load_payload_without_compression():
    mock_serializer = type('MockSerializer', (), {'load_payload': lambda self, payload, *args, **kwargs: 'decoded'} )()
    ussm = URLSafeSerializerMixin()
    ussm.load_payload = lambda payload, *args, serializer=None, **kwargs: URLSafeSerializerMixin.load_payload(ussm, payload, *args, serializer=serializer, **kwargs)
    ussm.serializer = mock_serializer
    result = ussm.load_payload(b"eyJhIjoiYiJ9")
    assert result == 'decoded'

def test_load_payload_with_compression():
    import zlib
    mock_serializer = type('MockSerializer', (), {'load_payload': lambda self, payload, *args, **kwargs: 'decoded'} )()
    ussm = URLSafeSerializerMixin()
    ussm.load_payload = lambda payload, *args, serializer=None, **kwargs: URLSafeSerializerMixin.load_payload(ussm, payload, *args, serializer=serializer, **kwargs)
    ussm.serializer = mock_serializer
    compressed = zlib.compress(b'{"a":"b"}')
    base64_compressed = base64_encode(compressed)
    result = ussm.load_payload(b"." + base64_compressed)
    assert result == 'decoded'

def test_load_payload_invalid_base64():
    ussm = URLSafeSerializerMixin()
    try:
        ussm.load_payload(b"invalid!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_corrupted_compressed():
    ussm = URLSafeSerializerMixin()
    try:
        ussm.load_payload(b".invalidbase64")
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #4
#--------------------------

def test_load_payload_normal_base64_only():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(zlib.compress(b'{"key":"value"}'))
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64_raises_badpayload():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid!base64")
        assert False, "Expected BadPayload"
    except BadPayload:
        pass

def test_load_payload_corrupted_compressed_data_raises_badpayload():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not_valid_zlib")
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload"
    except BadPayload:
        pass

def test_load_payload_empty_bytes():
    serializer = URLSafeSerializerMixin()
    payload = b""
    result = serializer.load_payload(payload)
    assert result is None


# LLM-generated content at query #5
#--------------------------

def test_load_payload_base64_decode_no_exception():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key": "value"}')
    result = serializer.load_payload(payload)


# LLM-generated content at query #6
#--------------------------

```python
def test_load_payload_without_exception_on_base64_decode():
    serializer = URLSafeSerializerMixin()
    payload = serializer.dump_payload({"key": "value"})
    serializer.load_payload(payload)


# LLM-generated content at query #7
#--------------------------

def test_load_payload_with_compressed_data_triggers_decompress():
    import zlib
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import base64_encode

    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass

    serializer = TestSerializer("secret")
    obj = {"key": "value"}
    compressed = zlib.compress(serializer.dump_payload(obj))
    base64d = base64_encode(compressed)
    payload = b"." + base64d
    result = serializer.load_payload(payload)
    assert result == obj


# LLM-generated content at query #8
#--------------------------

def test_load_payload_without_compression():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    original = base64_encode(zlib.compress(b'{"key":"value"}'))
    payload = b"." + original
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid!!!")
    except BadPayload:
        pass

def test_load_payload_invalid_compression():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not_compressed")
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass

def test_load_payload_empty_string():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"")
    except BadPayload:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_load_payload_does_not_raise_zlib_exception_for_non_compressed_base64_data():
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import base64_encode
    import zlib

    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #10
#--------------------------

Here's a unit test for the predicate at line 25:

```python
def test_predicate_at_line_25_evaluates_to_true():
    serializer = URLSafeSerializerMixin()
    invalid_payload = b"." + b"aGVsbG8="
    try:
        serializer.load_payload(invalid_payload)
    except BadPayload:
        pass
```


# LLM-generated content at query #11
#--------------------------

```python
def test_load_payload_with_compressed_flag_but_invalid_data_does_not_raise_zlib_error():
    import zlib
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer

    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass

    serializer = TestSerializer(secret_key="test")
    # Create payload that starts with b"." but has valid base64 content that is not zlib compressed
    valid_json = b'{"key":"value"}'
    compressed = zlib.compress(valid_json)
    base64_encoded = b"eJxLys9PUbC1AwAADgACAQ=="  # base64 of compressed data
    payload = b"." + base64_encoded
    # This should not raise BadPayload from zlib decompression
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}```


# LLM-generated content at query #12
#--------------------------

def test_load_payload_with_compressed_flag_but_no_compression():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b'{"a":1}')
    result = serializer.load_payload(payload)


# LLM-generated content at query #13
#--------------------------

def test_load_payload_normal_payload():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_compressed_payload():
    serializer = URLSafeSerializerMixin()
    compressed = zlib.compress(b'{"key":"value"}')
    payload = b"." + base64_encode(compressed)
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_corrupted_compressed():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not compressed data")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #14
#--------------------------

def test_load_payload_raises_bad_payload_on_zlib_decompress_failure():
    import zlib
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.exc import BadPayload
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import base64_encode

    serializer_instance = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"invalid_zlib_data")
    try:
        serializer_instance.load_payload(payload)
    except BadPayload as e:
        assert "Could not zlib decompress the payload before decoding the payload" in str(e)
        assert isinstance(e.original_error, zlib.error)


# LLM-generated content at query #15
#--------------------------

def test_load_payload_without_compression():
    serializer = URLSafeSerializerMixin(Serializer(secret_key="test"))
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin(Serializer(secret_key="test"))
    large_data = {"key": "x" * 1000}
    payload = serializer.dump_payload(large_data)
    assert payload.startswith(b".")
    result = serializer.load_payload(payload)
    assert result == large_data

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin(Serializer(secret_key="test"))
    try:
        serializer.load_payload(b"invalid_base64")
        assert False
    except BadPayload:
        pass

def test_load_payload_invalid_compressed_data():
    serializer = URLSafeSerializerMixin(Serializer(secret_key="test"))
    try:
        serializer.load_payload(b".invalid_base64")
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #16
#--------------------------

def test_load_payload_raises_bad_payload_on_zlib_decompress_failure():
    serializer = URLSafeSerializerMixin("secret")
    payload = b"." + base64_encode(b"not valid zlib compressed data")
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload"


# LLM-generated content at query #17
#--------------------------

```python
def test_load_payload_does_not_raise_exception_on_valid_base64():
    serializer = URLSafeSerializerMixin(secret_key="test")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)


# LLM-generated content at query #18
#--------------------------

def test_load_payload_with_dot_prefix_but_valid_base64_and_regular_compressed_data_does_not_raise_bad_payload_at_line_25():
    import zlib
    import base64
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer

    class ConcreteSerializer(URLSafeSerializerMixin, Serializer):
        pass

    serializer = ConcreteSerializer(secret_key="test")
    original_data = {"key": "value"}
    payload = serializer.dumps(original_data)
    # Force a scenario where payload starts with b"." and base64_decode succeeds
    # but zlib decompress fails (e.g., invalid compressed data)
    # We'll craft a payload that starts with b"." and has valid base64 but invalid zlib
    invalid_zlib_data = b"not_zlib_compressed"
    base64_encoded_invalid = base64.urlsafe_b64encode(invalid_zlib_data).rstrip(b"=")
    crafted_payload = b"." + base64_encoded_invalid
    try:
        serializer.load_payload(crafted_payload)
    except Exception:
        pass


# LLM-generated content at query #19
#--------------------------

```python
def test_load_payload_base64_decode_succeeds(self, mocker):
    url_safe_serializer = URLSafeSerializerMixin()
    mocker.patch.object(url_safe_serializer, 'load_payload', return_value={})
    valid_payload = base64_encode(b'{"key": "value"}')
    url_safe_serializer.load_payload(valid_payload)


# LLM-generated content at query #20
#--------------------------

```python
def test_load_payload_with_compressed_flag_and_non_compressed_data_does_not_raise_zlib_error():
    serializer = URLSafeSerializerMixin()
    # Create a payload that starts with '.' (indicating compression) but is not actually compressed
    # base64_encode of some valid JSON without compression, then prepend '.'
    original_json = b'{"key": "value"}'
    base64d = base64_encode(original_json)
    payload = b'.' + base64d
    # This should not raise BadPayload from zlib decompression because the data is not compressed
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
```


# LLM-generated content at query #21
#--------------------------

```python
def test_load_payload_without_decompression_does_not_raise_bad_payload():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"eyJmb28iOiAiYmFyIn0")
    except BadPayload as e:
        assert False, "Unexpected BadPayload exception"
    except Exception:
        pass
```


# LLM-generated content at query #22
#--------------------------

```python
def test_load_payload_base64_decode_does_not_raise_exception():
    serializer = URLSafeSerializerMixin("secret")
    payload = base64_encode(b'{"a":1}')
    result = serializer.load_payload(payload)
    assert result == {"a": 1}


# LLM-generated content at query #23
#--------------------------

def test_load_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    json_data = b'{"key":"value"}'
    compressed = zlib.compress(json_data)
    payload = b"." + base64_encode(compressed)
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid_base64!")
        assert False
    except BadPayload:
        pass

def test_load_payload_corrupted_compression():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not_compressed_data")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_empty_payload():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b"null")
    result = serializer.load_payload(payload)
    assert result is None


# LLM-generated content at query #24
#--------------------------

def test_load_payload_base64_decode_exception_raises_bad_payload():
    serializer = URLSafeSerializerMixin()
    invalid_payload = b"!!!invalid base64!!!"
    try:
        serializer.load_payload(invalid_payload)
        assert False
    except BadPayload as e:
        assert str(e) == "Could not base64 decode the payload because of an exception"


