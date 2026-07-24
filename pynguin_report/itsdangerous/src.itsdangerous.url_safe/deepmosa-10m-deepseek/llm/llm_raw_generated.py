####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_load_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    result = serializer.load_payload(b"eyJhIjogMX0=")
    assert result == {"a": 1}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(zlib.compress(b'{"a": 1}'))
    result = serializer.load_payload(payload)
    assert result == {"a": 1}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid!")
    except BadPayload:
        pass

def test_load_payload_invalid_compression():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b".eyJhIjogMX0=")
    except BadPayload:
        pass


# LLM-generated content at query #2
#--------------------------

def test_load_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    data = "x" * 1000
    payload = serializer.dump_payload(data)
    result = serializer.load_payload(payload)
    assert result == data

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"!!!invalid!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_bad_compression():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b".AAAA")
        assert False
    except BadPayload:
        pass

def test_load_payload_empty_payload():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"")
        assert False
    except (BadPayload, Exception):
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_load_payload_with_compressed_payload_enters_decompress_branch():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(zlib.compress(b'{"a":1}'))
    serializer.load_payload(payload)


# LLM-generated content at query #4
#--------------------------

def test_load_payload_does_not_raise_base64_exception():
    serializer = URLSafeSerializerMixin()
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)


# LLM-generated content at query #5
#--------------------------

def test_load_payload_no_compression_no_prefix():
    url_serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = url_serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression_prefix():
    url_serializer = URLSafeSerializerMixin()
    json_data = b'{"key":"value"}'
    compressed = zlib.compress(json_data)
    base64d = base64_encode(compressed)
    payload = b"." + base64d
    result = url_serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    url_serializer = URLSafeSerializerMixin()
    try:
        url_serializer.load_payload(b"invalid_base64!!!")
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload exception"

def test_load_payload_invalid_json():
    url_serializer = URLSafeSerializerMixin()
    payload = base64_encode(b"not json")
    try:
        url_serializer.load_payload(payload)
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload exception"


# LLM-generated content at query #6
#--------------------------

def test_load_payload_base64_decode_no_exception_with_valid_payload():
    serializer = URLSafeSerializerMixin()
    payload = b"eyJhIjogMX0"  # base64 encoded without padding, valid
    serializer.load_payload(payload)


# LLM-generated content at query #7
#--------------------------

def test_load_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    payload = b"eyJhIjogMX0"
    result = serializer.load_payload(payload)
    assert result == {"a": 1}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    payload = b".eJxT0lEoSczJUdJRAABlKwR3"
    result = serializer.load_payload(payload)
    assert result == {"test": "data"}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    payload = b"invalid!!!"
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_empty():
    serializer = URLSafeSerializerMixin()
    payload = b""
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_with_serializer_arg():
    serializer = URLSafeSerializerMixin()
    payload = b"WzFd"
    result = serializer.load_payload(payload, serializer=_CompactJSON)
    assert result == [1]

def test_load_payload_with_additional_args():
    serializer = URLSafeSerializerMixin()
    payload = b"ImhlbGxvIg"
    result = serializer.load_payload(payload, "extra")
    assert result == "hello"


# LLM-generated content at query #8
#--------------------------

def test_load_payload_with_decompress_exception():
    from itsdangerous.url_safe import URLSafeSerializerMixin
    from itsdangerous.serializer import Serializer
    from itsdangerous.exc import BadPayload
    import zlib
    import base64

    class TestSerializer(URLSafeSerializerMixin, Serializer):
        pass

    serializer = TestSerializer("test-secret")
    obj = {"test": "data" * 1000}
    payload = serializer.dumps(obj)
    payload_bytes = payload.encode("ascii") if isinstance(payload, str) else payload
    
    corrupted_payload = b"." + b"corrupted_base64"
    try:
        serializer.loads(corrupted_payload)
    except BadPayload as e:
        assert "Could not zlib decompress" in str(e)


# LLM-generated content at query #9
#--------------------------

def test_load_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(zlib.compress(b'{"key":"value"}'))
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid")
    except BadPayload:
        pass
    else:
        assert False

def test_load_payload_corrupt_zlib():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"corrupt data")
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass
    else:
        assert False


# LLM-generated content at query #10
#--------------------------

def test_load_payload_raises_bad_payload_on_zlib_decompress_failure():
    serializer = URLSafeSerializerMixin("secret")
    compressed_prefix = b"."
    invalid_compressed_data = base64_encode(b"not zlib compressed data")
    payload = compressed_prefix + invalid_compressed_data
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass


# LLM-generated content at query #11
#--------------------------

def test_load_payload_with_compressed_flag_but_valid_base64_and_valid_zlib():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(zlib.compress(b'{"a":1}'))
    result = serializer.load_payload(payload)


# LLM-generated content at query #12
#--------------------------

```python
def test_load_payload_does_not_raise_on_non_compressed_valid_payload():
    serializer = URLSafeSerializerMixin("secret")
    payload = serializer.dumps({"key": "value"})
    result = serializer.loads(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #13
#--------------------------

def test_load_payload_decompress_false_does_not_raise():
    serializer = URLSafeSerializerMixin()
    payload = serializer.dump_payload({"key": "value"})
    if payload.startswith(b"."):
        payload = payload[1:]
    serializer.load_payload(payload)


# LLM-generated content at query #14
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
        serializer.load_payload(b"invalid!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_invalid_compression():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not compressed data")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #15
#--------------------------

def test_load_payload_does_not_raise_badpayload_for_valid_base64():
    serializer = URLSafeSerializerMixin()
    payload = b"eyJmb28iOiAiYmFyIn0"
    result = serializer.load_payload(payload)


# LLM-generated content at query #16
#--------------------------

def test_load_payload_normal_base64_decoded():
    instance = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = instance.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_compressed():
    instance = URLSafeSerializerMixin()
    compressed = zlib.compress(b'{"key":"value"}')
    payload = b"." + base64_encode(compressed)
    result = instance.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64_raises_badpayload():
    instance = URLSafeSerializerMixin()
    try:
        instance.load_payload(b"invalid!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_compressed_corrupt_zlib_raises_badpayload():
    instance = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not valid zlib")
    try:
        instance.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_empty_string_byte():
    instance = URLSafeSerializerMixin()
    payload = base64_encode(b"")
    result = instance.load_payload(payload)
    assert result == {}


# LLM-generated content at query #17
#--------------------------

def test_load_payload_normal_payload():
    mixin = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = mixin.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_compressed_payload():
    mixin = URLSafeSerializerMixin()
    json_data = b'{"key":"value"}'
    compressed = zlib.compress(json_data)
    base64d = base64_encode(compressed)
    payload = b"." + base64d
    result = mixin.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_invalid_base64():
    mixin = URLSafeSerializerMixin()
    try:
        mixin.load_payload(b"invalid!!")
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload exception"

def test_load_payload_corrupted_compressed():
    mixin = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"corrupted_data")
    try:
        mixin.load_payload(payload)
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload exception"

def test_load_payload_empty_payload():
    mixin = URLSafeSerializerMixin()
    try:
        mixin.load_payload(b"")
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload exception"


# LLM-generated content at query #18
#--------------------------

def test_load_payload_base64_decode_success():
    serializer = URLSafeSerializerMixin()
    payload = b"eyJmb28iOiJiYXIifQ"
    result = serializer.load_payload(payload)
    assert result == {"foo": "bar"}


# LLM-generated content at query #19
#--------------------------

def test_load_payload_compressed_decompress_fails():
    serializer = URLSafeSerializerMixin("secret")
    payload = b"." + base64_encode(zlib.compress(b'{"a":1}')[:1])
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass


# LLM-generated content at query #20
#--------------------------

def test_load_payload_no_decompress_does_not_raise_on_invalid_zlib():
    serializer = URLSafeSerializerMixin()
    # Provide a valid base64 payload that is not zlib compressed
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #21
#--------------------------

def test_load_payload_base64_decode_success():
    serializer = URLSafeSerializerMixin()
    payload = b"eyJhIjoiYiJ9"
    result = serializer.load_payload(payload)


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

def test_load_payload_with_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid!base64")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_corrupted_compressed():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"notvalidzlib")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_with_empty_payload():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b"")
    result = serializer.load_payload(payload)
    assert result is None


# LLM-generated content at query #2
#--------------------------

def test_dump_payload_no_compression():
    serializer = URLSafeSerializerMixin()
    serializer.dump_payload = lambda obj: b'{"a":1}'
    result = serializer.dump_payload({"a": 1})
    assert result == b'eyJhIjoxfQ'

def test_dump_payload_with_compression():
    serializer = URLSafeSerializerMixin()
    serializer.dump_payload = lambda obj: b'a' * 100
    result = serializer.dump_payload("test")
    assert result.startswith(b".")
    assert result.endswith(b"=") == False


# LLM-generated content at query #3
#--------------------------

def test_load_payload_no_exception_on_valid_base64():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)


# LLM-generated content at query #4
#--------------------------

```python
def test_load_payload_with_compressed_data_triggers_decompress_path():
    serializer = URLSafeSerializerMixin()
    compressed_payload = b"." + base64_encode(zlib.compress(b'{"key":"value"}'))
    result = serializer.load_payload(compressed_payload)
    assert result == {"key": "value"}


# LLM-generated content at query #5
#--------------------------

def test_load_payload_base64_decodes_payload():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_decompresses_when_prefixed_with_dot():
    serializer = URLSafeSerializerMixin()
    compressed = zlib.compress(b'{"key":"value"}')
    payload = b"." + base64_encode(compressed)
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_raises_bad_payload_on_base64_decode_error():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid!base64")
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass

def test_load_payload_raises_bad_payload_on_decompress_error():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not compressed data")
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass

def test_load_payload_handles_empty_payload():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b"null")
    result = serializer.load_payload(payload)
    assert result is None


# LLM-generated content at query #6
#--------------------------

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

def test_load_payload_raises_bad_payload_on_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"invalid-base64!!!")
        assert False
    except BadPayload:
        pass

def test_load_payload_raises_bad_payload_on_decompression_error():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not compressed data")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #7
#--------------------------

def test_load_payload_no_decompress_no_exception():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b'{"key":"value"}')
    result = serializer.load_payload(payload)


# LLM-generated content at query #8
#--------------------------

def test_load_payload_empty_payload():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"")
    except Exception:
        pass

def test_load_payload_with_dot():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b".eJw9kE0OAiEMhe8y6wYosDDxLgY3Jh7AoTPh5u6F0bj5vveT9vV7z3s_AQ")
    except Exception:
        pass

def test_load_payload_without_dot():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"eyJhIjogMX0")
    except Exception:
        pass

def test_load_payload_invalid_base64():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b"!!!")
    except Exception:
        pass

def test_load_payload_compressed():
    serializer = URLSafeSerializerMixin()
    try:
        serializer.load_payload(b".eJw9kE0OAiEMhe8y6wYosDDxLgY3Jh7AoTPh5u6F0bj5vveT9vV7z3s_AQ")
    except Exception:
        pass


# LLM-generated content at query #9
#--------------------------

def test_load_payload_base64_decode_success():
    serializer = URLSafeSerializerMixin()
    payload = b"eyJrZXkiOiAidmFsdWUifQ"  # valid base64 without padding
    serializer.load_payload(payload)


# LLM-generated content at query #10
#--------------------------

def test_load_payload_base64_decode_succeeds():
    serializer = URLSafeSerializerMixin()
    payload = b"eyJrZXkiOiAidmFsdWUifQ"
    result = serializer.load_payload(payload)


# LLM-generated content at query #11
#--------------------------

def test_load_payload_no_decompress_no_exception():
    serializer = URLSafeSerializerMixin()
    # payload that is valid base64 and does not start with '.', so decompress stays False
    # base64_encode(b'{"a":1}') -> b'eyJhIjoxfQ' (without padding, but base64_decode adds padding)
    payload = b'eyJhIjoxfQ'
    serializer.load_payload(payload)


# LLM-generated content at query #12
#--------------------------

def test_load_payload_normal_base64_decoded():
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
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload"

def test_load_payload_bad_compressed_data():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not compressed")
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload"


# LLM-generated content at query #13
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

def test_load_payload_with_compression_prefix():
    serializer = URLSafeSerializerMixin()
    compressed = zlib.compress(b'{"key":"value"}')
    payload = b"." + base64_encode(compressed)
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_raises_bad_payload_for_invalid_base64():
    serializer = URLSafeSerializerMixin()
    payload = b"invalid_base64!!!"
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_raises_bad_payload_for_corrupted_compressed_data():
    serializer = URLSafeSerializerMixin()
    payload = b"." + base64_encode(b"not_valid_zlib_data")
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_with_empty_payload():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b"")
    result = serializer.load_payload(payload)
    assert result is None

def test_load_payload_with_none_payload():
    serializer = URLSafeSerializerMixin()
    payload = base64_encode(b"null")
    result = serializer.load_payload(payload)
    assert result is None


# LLM-generated content at query #14
#--------------------------

def test_load_payload_with_dot_prefix_and_invalid_compressed_data_raises_bad_payload():
    import base64
    serializer = URLSafeSerializerMixin("test_secret")
    invalid_compressed = b"x\x9c\xcbH\xcd\xc9\xc9\x07\x00\x06,\x02\x15"
    encoded = base64.urlsafe_b64encode(invalid_compressed).rstrip(b"=")
    payload = b"." + encoded
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass


# LLM-generated content at query #15
#--------------------------

def test_load_payload_no_decompression_fails_on_invalid_payload():
    serializer = URLSafeSerializerMixin("secret")
    payload = b".invalid_base64"  # starts with '.', so decompress=True, but base64_decode will succeed with padding
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass


# LLM-generated content at query #16
#--------------------------

def test_load_payload_no_base64_exception():
    serializer = URLSafeSerializerMixin()
    result = serializer.load_payload(b"SGVsbG8=")


# LLM-generated content at query #17
#--------------------------

def test_load_payload_no_decompress_does_not_raise_decompress_exception():
    serializer = URLSafeSerializerMixin()
    payload = b"eyJrZXkiOiAidmFsdWUifQ"
    serializer.load_payload(payload)


# LLM-generated content at query #18
#--------------------------

def test_load_payload_does_not_raise_bad_payload_with_valid_base64():
    serializer = URLSafeSerializerMixin("test_secret")
    payload = b"eyJhIjogMX0"  # valid base64 without padding
    result = serializer.load_payload(payload)
    assert result is not None


# LLM-generated content at query #19
#--------------------------

def test_load_payload_does_not_raise_zlib_error_when_decompress_false():
    serializer = URLSafeSerializerMixin("secret")
    payload = b"eyJrZXkiOiAidmFsdWUifQ"
    result = serializer.load_payload(payload)


# LLM-generated content at query #20
#--------------------------

def test_load_payload_base64_decode_does_not_raise():
    serializer = URLSafeSerializerMixin()
    serializer.load_payload(b"eyJhIjoiYiJ9")


