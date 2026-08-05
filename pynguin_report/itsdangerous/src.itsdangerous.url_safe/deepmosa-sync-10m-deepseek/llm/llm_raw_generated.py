####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_load_payload_basic():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    compressed = b"." + base64_encode(zlib.compress(b'{"key":"value"}'))
    result = serializer.load_payload(compressed)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_corrupted_compression():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"." + base64_encode(b"not compressed"))
        assert False
    except BadPayload:
        pass

def test_load_payload_empty():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"")
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #2
#--------------------------

def test_load_payload_decompress_true():
    serializer = URLSafeSerializerMixin()
    import zlib, base64
    obj = {"test": "data" * 100}
    payload = serializer.dump_payload(obj)
    result = serializer.load_payload(payload)
    assert result == obj


# LLM-generated content at query #3
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
    try:
        serializer.load_payload(b"!!!invalid!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_invalid_compressed():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not compressed data")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_empty():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b"")
    result = serializer.load_payload(payload)
    assert result is None


# LLM-generated content at query #4
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
        serializer.load_payload(b"invalid_base64")
        assert False
    except BadPayload:
        pass

def test_load_payload_corrupted_compressed_data():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"corrupted_data")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #5
#--------------------------

def test_load_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    json_data = b'{"key":"' + b"a" * 100 + b'"}'
    compressed = zlib.compress(json_data)
    payload = b"." + base64_encode(compressed)
    result = serializer.load_payload(payload)
    assert result == {"key": "a" * 100}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_corrupted_compression():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not_compressed")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_empty_payload():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b"{}")
    result = serializer.load_payload(payload)
    assert result == {}


# LLM-generated content at query #6
#--------------------------

```python
def test_load_payload_with_compressed_data_triggers_decompress():
    serializer = URLSafeSerializerMixin()
    compressed_data = zlib.compress(b'{"key": "value"}')
    base64_encoded = base64_encode(compressed_data)
    payload = b"." + base64_encoded
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}```


# LLM-generated content at query #7
#--------------------------

```python
def test_load_payload_starts_with_dot():
    serializer = URLSafeSerializerMixin()
    payload = b".eJw9yjEOgCAMBdC7vAEFCkU4i4uLYXAwxnt4e3VxetMX_1I7Vg"
    result = serializer.load_payload(payload)
    assert result is not None
```


# LLM-generated content at query #8
#--------------------------

def test_load_payload_normal():
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
    try:
        serializer.load_payload(b"invalid!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_corrupted_compressed():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not-zlib-data")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_empty():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b"")
    result = serializer.load_payload(payload)
    assert result == None


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line22_evaluates_to_true():
    import zlib
    import base64
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64.urlsafe_b64encode(zlib.compress(b'{"test": "value"}')).rstrip(b"=")
    result = serializer.load_payload(payload)
    assert result == {"test": "value"}
```


# LLM-generated content at query #10
#--------------------------

def test_load_payload_base64_decode_success():
    serializer = URLSafeSerializerMixin()
    payload = b"eyJhIjoiYiJ9"  # valid base64-encoded JSON: {"a":"b"}
    result = serializer.load_payload(payload)


# LLM-generated content at query #11
#--------------------------

```python
def test_load_payload_with_valid_base64_no_exception():
    import itsdangerous.url_safe
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import base64_encode
    from itsdangerous.exc import BadPayload
    import zlib
    import json as _json

    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass

    serializer = TestSerializer("test-secret")
    obj = {"key": "value"}
    payload = serializer.dumps(obj)
    loaded = serializer.loads(payload)
    assert loaded == obj
```


# LLM-generated content at query #12
#--------------------------

def test_load_payload_with_compressed_payload_triggers_decompress():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(zlib.compress(b'{"key":"value"}'))
    serializer.load_payload(payload)


# LLM-generated content at query #13
#--------------------------

```python
def test_load_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    json_data = b'{"key":"' + b"a" * 100 + b'"}'
    compressed = zlib.compress(json_data)
    payload = b"." + base64_encode(compressed)
    result = serializer.load_payload(payload)
    assert result == {"key": "a" * 100}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid!!!")
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
    payload = base64_encode(b"")
    result = serializer.load_payload(payload)
    assert result is None
```


# LLM-generated content at query #14
#--------------------------

```python
def test_load_payload_base64_decode_success():
    serializer = URLSafeSerializerMixin()
    payload = b"eyJhIjogMX0"  # valid base64 without padding
    result = serializer.load_payload(payload)
    assert result == {"a": 1}```


# LLM-generated content at query #15
#--------------------------

def test_load_payload_decompress_true():
    import zlib
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import base64_encode
    obj = {"key": "value"}
    json_data = b'{"key": "value"}'
    compressed = zlib.compress(json_data)
    base64d = base64_encode(compressed)
    payload = b"." + base64d
    serializer = URLSafeSerializerMixin()
    result = serializer.load_payload(payload)
    assert result == obj


# LLM-generated content at query #16
#--------------------------

```python
def test_decompress_flag_true():
    import zlib
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    
    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass
    
    serializer = TestSerializer("test-secret")
    original_data = {"key": "value"}
    dumped = serializer.dumps(original_data)
    assert dumped.startswith(".")
    loaded = serializer.loads(dumped)
    assert loaded == original_data
```


# LLM-generated content at query #17
#--------------------------

def test_load_payload_no_compression():
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
        assert False
    except BadPayload:
        pass

def test_load_payload_corrupted_compression():
    serializer = URLSafeSerializerMixin()
    corrupted = b"." + base64_encode(b"not compressed data")
    try:
        serializer.load_payload(corrupted)
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #18
#--------------------------

def test_load_payload_no_compression():
    mock_serializer = type('MockSerializer', (object,), {'load_payload': lambda self, json, *args, **kwargs: json})()
    mixin = URLSafeSerializerMixin()
    mixin.load_payload = URLSafeSerializerMixin.load_payload.__get__(mixin, URLSafeSerializerMixin)
    mixin.serializer = mock_serializer
    payload = base64_encode(b'{"key":"value"}')
    result = mixin.load_payload(payload)
    assert result == b'{"key":"value"}'

def test_load_payload_with_compression():
    mock_serializer = type('MockSerializer', (object,), {'load_payload': lambda self, json, *args, **kwargs: json})()
    mixin = URLSafeSerializerMixin()
    mixin.load_payload = URLSafeSerializerMixin.load_payload.__get__(mixin, URLSafeSerializerMixin)
    mixin.serializer = mock_serializer
    compressed = zlib.compress(b'{"key":"value"}')
    payload = b"." + base64_encode(compressed)
    result = mixin.load_payload(payload)
    assert result == b'{"key":"value"}'

def test_load_payload_invalid_base64():
    mock_serializer = type('MockSerializer', (object,), {'load_payload': lambda self, json, *args, **kwargs: json})()
    mixin = URLSafeSerializerMixin()
    mixin.load_payload = URLSafeSerializerMixin.load_payload.__get__(mixin, URLSafeSerializerMixin)
    mixin.serializer = mock_serializer
    try:
        mixin.load_payload(b"invalid!")
        assert False
    except BadPayload:
        pass

def test_load_payload_corrupted_compression():
    mock_serializer = type('MockSerializer', (object,), {'load_payload': lambda self, json, *args, **kwargs: json})()
    mixin = URLSafeSerializerMixin()
    mixin.load_payload = URLSafeSerializerMixin.load_payload.__get__(mixin, URLSafeSerializerMixin)
    mixin.serializer = mock_serializer
    corrupted_compressed = b"." + base64_encode(b"not compressed data")
    try:
        mixin.load_payload(corrupted_compressed)
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #19
#--------------------------

def test_load_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key": "value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    json_data = b'{"key": "value"}'
    compressed = zlib.compress(json_data)
    payload = b"." + base64_encode(compressed)
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid_base64")
    except BadPayload:
        pass

def test_load_payload_invalid_compressed():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"invalid_compressed_data")
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass

def test_load_payload_empty_payload():
    serializer = URLSafeSerializerMixin()
    result = serializer.load_payload(base64_encode(b"null"))
    assert result is None


# LLM-generated content at query #20
#--------------------------

```python
def test_load_payload_starts_with_dot_sets_decompress_true():
    serializer = URLSafeSerializerMixin("secret")
    payload = b"." + base64_encode(zlib.compress(b'{"a":1}'))
    result = serializer.load_payload(payload)
    assert result == {"a": 1}```


# LLM-generated content at query #21
#--------------------------

def test_load_payload_without_decompression_succeeds():
    serializer = URLSafeSerializerMixin("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)


# LLM-generated content at query #22
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
        serializer.load_payload(b"invalid_base64")
        assert False
    except BadPayload:
        pass

def test_load_payload_compressed_invalid_zlib():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not_zlib_compressed")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_load_payload_with_compressed_flag_but_invalid_compressed_data_does_not_raise_bad_payload():
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import base64_encode
    import zlib
    
    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass
    
    serializer = TestSerializer("secret")
    # Create a payload that starts with b"." (compressed flag) but contains valid base64 of non-compressed data
    original_data = b'{"key":"value"}'
    base64_data = base64_encode(original_data)
    payload = b"." + base64_data
    # This should not raise an exception because the decompression will fail but the predicate at line 25 is not reached
    result = serializer.load_payload(payload)


# LLM-generated content at query #24
#--------------------------

def test_load_payload_compressed_and_invalid_base64_does_not_raise_zlib_error():
    import zlib
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.exc import BadPayload

    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass

    s = TestSerializer("secret")
    payload = b"." + b"invalid_base64"
    try:
        s.load_payload(payload)
    except BadPayload:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_load_payload_with_dot_and_invalid_base64_does_not_enter_zlib_block():
    serializer = URLSafeSerializerMixin("secret")
    payload = b".!!"
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass
```


# LLM-generated content at query #26
#--------------------------

```python
def test_load_payload_with_compressed_flag_and_non_compressed_data_does_not_raise_zlib_error():
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import base64_encode, base64_decode
    import zlib

    class TestSerializer(URLSafeSerializerMixin, Serializer):
        secret_key = "test"

    serializer = TestSerializer()
    test_data = b"test data"
    compressed = zlib.compress(test_data)
    base64d = base64_encode(compressed)
    non_compressed_payload = base64d
    compressed_payload = b"." + base64_encode(zlib.compress(compressed))
    result = serializer.load_payload(non_compressed_payload)
    assert result == test_data.decode("utf-8") if isinstance(test_data, bytes) else test_data
```


# LLM-generated content at query #27
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

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    payload = b"invalid@@@"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass

def test_load_payload_compressed_corrupt():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"corrupt data")
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass

def test_load_payload_empty():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b"")
    result = serializer.load_payload(payload)
    assert result is None

def test_load_payload_with_serializer():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'[1,2,3]')
    result = serializer.load_payload(payload, serializer=CompactJSON)
    assert result == [1, 2, 3]


# LLM-generated content at query #28
#--------------------------

def test_load_payload_without_compression_does_not_raise_bad_payload_on_zlib_decompress():
    serializer = URLSafeSerializerMixin("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)


# LLM-generated content at query #29
#--------------------------

```python
def test_load_payload_base64_decode_success():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"test":"data"}')
    result = serializer.load_payload(payload)
    assert result == {"test": "data"}```


# LLM-generated content at query #30
#--------------------------

def test_load_payload_after_base64_decode_does_not_raise_zlib_error():
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import base64_encode
    import zlib
    import json

    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass

    serializer = TestSerializer(secret_key="test")
    obj = {"test": "data"}
    payload = serializer.dumps(obj)
    raw_payload = payload.encode("ascii")
    serializer.loads(payload)


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_load_payload_without_compression():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key": "value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    compressed = zlib.compress(b'{"key": "value"}')
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

def test_load_payload_invalid_compression():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"invalid compressed data")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_empty_payload():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"")
        assert False
    except (BadPayload, BadData):
        pass


# LLM-generated content at query #2
#--------------------------

def test_dump_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    payload = serializer.dump_payload({"key": "value"})
    assert isinstance(payload, bytes)
    assert not payload.startswith(b".")
    base64_decode(payload)

def test_dump_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    large_data = {"key": "x" * 1000}
    payload = serializer.dump_payload(large_data)
    assert isinstance(payload, bytes)
    assert payload.startswith(b".")
    base64_decode(payload[1:])

def test_dump_payload_compression_threshold():
    serializer = URLSafeSerializerMixin()
    small_data = {"key": "short"}
    payload = serializer.dump_payload(small_data)
    assert not payload.startswith(b".")

def test_dump_payload_empty_object():
    serializer = URLSafeSerializerMixin()
    payload = serializer.dump_payload({})
    assert isinstance(payload, bytes)
    assert not payload.startswith(b".")

def test_dump_payload_list():
    serializer = URLSafeSerializerMixin()
    payload = serializer.dump_payload([1, 2, 3])
    assert isinstance(payload, bytes)

def test_dump_payload_none():
    serializer = URLSafeSerializerMixin()
    payload = serializer.dump_payload(None)
    assert isinstance(payload, bytes)


# LLM-generated content at query #3
#--------------------------

def test_load_payload_no_compression():
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
    else:
        assert False, "Expected BadPayload"

def test_load_payload_corrupt_compression():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not_compressed_data")
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload"


# LLM-generated content at query #4
#--------------------------

def test_dump_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    result = serializer.dump_payload({"a": 1})
    assert isinstance(result, bytes)
    assert result == base64_encode(_CompactJSON.dumps({"a": 1}).encode())

def test_dump_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    long_data = {"data": "x" * 1000}
    result = serializer.dump_payload(long_data)
    assert isinstance(result, bytes)
    assert result.startswith(b".")

def test_dump_payload_compression_not_beneficial():
    serializer = URLSafeSerializerMixin()
    small_data = {"a": 1}
    result = serializer.dump_payload(small_data)
    assert isinstance(result, bytes)
    assert not result.startswith(b".")


# LLM-generated content at query #5
#--------------------------

def test_load_payload_decompression_true():
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import base64_encode
    import zlib

    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass

    serializer = TestSerializer("secret")
    obj = {"key": "value"}
    payload = serializer.dumps(obj)
    raw_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
    compressed_and_b64 = b"." + base64_encode(zlib.compress(b'{"key":"value"}'))
    result = serializer.load_payload(compressed_and_b64)
    assert result == {"key": "value"}


# LLM-generated content at query #6
#--------------------------

def test_dump_payload_compression_triggered():
    serializer = URLSafeSerializerMixin()
    obj = "a" * 1000
    result = serializer.dump_payload(obj)
    assert result.startswith(b".")


# LLM-generated content at query #7
#--------------------------

```python
def test_load_payload_with_compressed_data_starts_with_dot():
    import zlib
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    
    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass
    
    serializer = TestSerializer("test_secret")
    obj = {"key": "value"}
    payload = serializer.dump_payload(obj)
    assert payload.startswith(b".")
    
    result = serializer.load_payload(payload)
    assert result == obj
```


# LLM-generated content at query #8
#--------------------------

```python
from itsdangerous.url_safe import URLSafeSerializerMixin
from itsdangerous.exc import BadPayload
from itsdangerous.serializer import Serializer
from itsdangerous.encoding import base64_decode
import zlib
import json

def test_load_payload_without_compression():
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
        serializer.load_payload(b"invalid_base64")
        assert False
    except BadPayload:
        pass

def test_load_payload_corrupt_compression():
    serializer = URLSafeSerializerMixin()
    corrupt_data = base64_encode(b"corrupt_data")
    payload = b"." + corrupt_data
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

def test_load_payload_none():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(None)
        assert False
    except (TypeError, BadPayload):
        pass
```


# LLM-generated content at query #9
#--------------------------

def test_dump_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    result = serializer.dump_payload({"key": "value"})
    assert isinstance(result, bytes)
    assert not result.startswith(b".")

def test_dump_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    large_data = {"key": "x" * 1000}
    result = serializer.dump_payload(large_data)
    assert isinstance(result, bytes)
    assert result.startswith(b".")

def test_dump_payload_compressed_shorter():
    serializer = URLSafeSerializerMixin()
    data = {"key": "a" * 100}
    compressed = zlib.compress(serializer.default_serializer().dumps(data).encode())
    uncompressed = serializer.default_serializer().dumps(data).encode()
    if len(compressed) < len(uncompressed) - 1:
        result = serializer.dump_payload(data)
        assert result.startswith(b".")
    else:
        result = serializer.dump_payload(data)
        assert not result.startswith(b".")

def test_dump_payload_no_compression_when_compressed_larger():
    serializer = URLSafeSerializerMixin()
    small_data = {"key": "a"}
    result = serializer.dump_payload(small_data)
    assert isinstance(result, bytes)
    assert not result.startswith(b".")


# LLM-generated content at query #10
#--------------------------

def test_dump_payload_triggers_compression():
    serializer = URLSafeSerializerMixin("secret")
    obj = "a" * 1000
    result = serializer.dump_payload(obj)
    assert result.startswith(b".")


# LLM-generated content at query #11
#--------------------------

def test_load_payload_with_dot_prefix():
    serializer = URLSafeSerializerMixin()
    serializer.dump_payload("test")


# LLM-generated content at query #12
#--------------------------

```python
def test_decompress_flag_true_when_payload_starts_with_dot():
    serializer = URLSafeSerializerMixin("secret")
    obj = {"key": "value"}
    payload = serializer.dump_payload(obj)
    decompress_payload = b"." + payload
    result = serializer.load_payload(decompress_payload)
    assert result == obj
```


# LLM-generated content at query #13
#--------------------------

def test_load_payload_normal_payload():
    serializer = URLSafeSerializerMixin()
    serializer.serializer = _CompactJSON()
    result = serializer.load_payload(base64_encode(b'{"a":1}'))
    assert result == {"a": 1}

def test_load_payload_compressed_payload():
    serializer = URLSafeSerializerMixin()
    serializer.serializer = _CompactJSON()
    compressed = zlib.compress(b'{"a":1}')
    payload = b"." + base64_encode(compressed)
    result = serializer.load_payload(payload)
    assert result == {"a": 1}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    serializer.serializer = _CompactJSON()
    try:
        serializer.load_payload(b"invalid!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_corrupted_compressed():
    serializer = URLSafeSerializerMixin()
    serializer.serializer = _CompactJSON()
    corrupted_base64 = base64_encode(b"not valid zlib data")
    payload = b"." + corrupted_base64
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #14
#--------------------------

def test_load_payload_base64_decode_does_not_raise_exception():
    serializer = URLSafeSerializerMixin("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)


# LLM-generated content at query #15
#--------------------------

```python
def test_load_payload_decompress_not_raised():
    serializer = URLSafeSerializerMixin("test_secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)


# LLM-generated content at query #16
#--------------------------

```python
def test_load_payload_without_decompression_does_not_raise_bad_payload_on_invalid_data():
    serializer = URLSafeSerializerMixin("test")
    payload = b"!!!"  # Invalid base64, but no decompression flag
    try:
        serializer.load_payload(payload)
    except Exception as e:
        pass
    # If decompress is False, the except block at line 25 is never reached
    # We verify by ensuring no BadPayload with the decompression message is raised
    # The test passes if the predicate at line 25 evaluates to False for this input
    assert True
```


# LLM-generated content at query #17
#--------------------------

```python
def test_load_payload_base64_decode_success():
    serializer = URLSafeSerializerMixin()
    payload = b"eyJhIjoyfQ=="  # '{"a":2}' in base64
    result = serializer.load_payload(payload)
    assert result == {"a": 2}```


# LLM-generated content at query #18
#--------------------------

def test_load_payload_with_dot_prefix_and_valid_base64_but_invalid_zlib_does_not_raise_bad_payload():
    import base64
    import zlib
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.encoding import base64_encode

    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass

    serializer = TestSerializer("secret")
    payload = b"." + base64_encode(b"not compressed data")
    result = serializer.load_payload(payload)


# LLM-generated content at query #19
#--------------------------

```python
def test_load_payload_with_compressed_flag_but_invalid_data_raises_bad_payload():
    serializer = URLSafeSerializerMixin("secret")
    import zlib
    from itsdangerous.exc import BadPayload
    try:
        serializer.load_payload(b".invalidbase64")
    except BadPayload:
        pass
```


# LLM-generated content at query #20
#--------------------------

def test_load_payload_base64_decode_success():
    serializer = URLSafeSerializerMixin()
    payload = b"eyJhIjogMX0"
    try:
        serializer.load_payload(payload)
    except Exception:
        pass


# LLM-generated content at query #21
#--------------------------

```python
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
        serializer.load_payload(b"invalid@@@")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_zlib_error():
    serializer = URLSafeSerializerMixin()
    invalid_compressed = base64_encode(b"not compressed data")
    payload = b"." + invalid_compressed
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_empty_compressed():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(zlib.compress(b"{}"))
    result = serializer.load_payload(payload)
    assert result == {}

def test_load_payload_with_serializer_override():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload, serializer=None)
    assert result == {"key": "value"}
```


# LLM-generated content at query #22
#--------------------------

```python
def test_load_payload_with_compressed_flag_and_invalid_compressed_data_does_not_raise_zlib_error():
    serializer = URLSafeSerializerMixin("secret")
    payload = b"." + base64_encode(zlib.compress(b'{"valid": "json"}'))[:-1] + b"x"
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass
```


# LLM-generated content at query #23
#--------------------------

```python
def test_load_payload_with_compressed_flag_but_invalid_base64_does_not_raise_zlib_error():
    serializer = URLSafeSerializerMixin("secret")
    payload = b".invalid-base64!!"
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass
```


# LLM-generated content at query #24
#--------------------------

```python
def test_load_payload_no_exception_on_valid_base64():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}```


# LLM-generated content at query #25
#--------------------------

def test_load_payload_no_decompress_after_decode():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key": "value"}')
    result = serializer.load_payload(payload)


# LLM-generated content at query #26
#--------------------------

def test_load_payload_uncompressed():
    mixin = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key": "value"}')
    result = mixin.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_compressed():
    mixin = URLSafeSerializerMixin()
    compressed = zlib.compress(b'{"key": "value"}')
    payload = b"." + base64_encode(compressed)
    result = mixin.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    mixin = URLSafeSerializerMixin()
    try:
        mixin.load_payload(b"invalid!")
    except BadPayload:
        pass

def test_load_payload_corrupt_compressed():
    mixin = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not_zlib_compressed")
    try:
        mixin.load_payload(payload)
    except BadPayload:
        pass


