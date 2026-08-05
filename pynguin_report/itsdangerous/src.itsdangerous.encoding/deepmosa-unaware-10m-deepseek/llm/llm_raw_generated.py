####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test single character
    assert base64_decode(base64_encode(b"a")) == b"a"
    
    # Test padding cases
    assert base64_decode(base64_encode(b"ab")) == b"ab"
    assert base64_decode(base64_encode(b"abc")) == b"abc"
    
    # Test with special characters
    assert base64_decode(base64_encode(b"\x00\x01\x02")) == b"\x00\x01\x02"
    
    # Test with text input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes that have extra padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with no padding in input
    assert base64_decode(b"aGVsbG8") == b"hello"
```


# LLM-generated content at query #2
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    assert base64_decode("Pj4-Pg") == b">>>"  # Without padding
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with special characters
    assert base64_decode("AAECAw==") == b"\x00\x01\x02\x03"
    assert base64_decode("AAECAw") == b"\x00\x01\x02\x03"  # Without padding
    
    # Test that invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("ABC")  # Invalid length without proper padding
    
    # Test with very long data
    long_data = b"x" * 10000
    encoded = base64_encode(long_data)
    assert base64_decode(encoded) == long_data
```


# LLM-generated content at query #3
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special URL-safe characters
    assert base64_decode("Pj4-Pz8_") == b">>>??"
    
    # Test with binary data
    binary = bytes(range(256))
    encoded = base64_encode(binary)
    assert base64_decode(encoded) == binary
    
    # Test BadData exception for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode("$$$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #4
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello")
    decoded = base64_decode(encoded)
    assert decoded == b"hello"
    
    # Test with string input
    decoded = base64_decode(base64_encode("test"))
    assert decoded == b"test"
    
    # Test empty input
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test with padding
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test without padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test with special characters
    encoded = base64_encode(b"\x00\x01\x02")
    decoded = base64_decode(encoded)
    assert decoded == b"\x00\x01\x02"
    
    # Test with max length
    encoded = base64_encode(b"a" * 1000)
    decoded = base64_decode(encoded)
    assert decoded == b"a" * 1000
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in string input
    decoded = base64_decode("aGVsbG8=")
    assert decoded == b"hello"
```


# LLM-generated content at query #5
#--------------------------

```python
def test_base64_decode():
    # Test normal valid base64 string
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test URL-safe base64 (no padding)
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test with bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with invalid characters
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with incorrect length (should still work with padding)
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test special characters in URL-safe base64
    result = base64_decode("Pz8_Pz8")
    assert result == b"???\x3f\x3f"
    
    # Test with string input with non-ASCII characters (should be ignored)
    result = base64_decode("SGVsbG8gV29ybGQ=\xc3")
    assert result == b"Hello World"
```


# LLM-generated content at query #6
#--------------------------

```python
def test_base64_decode():
    # Test with valid URL-safe base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test with bytes input
    assert base64_decode(b"VGVzdA==") == b"Test"
    assert base64_decode(b"VGVzdA") == b"Test"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with special characters
    assert base64_decode("_-xq") == b"\xff\xbe"
    
    # Test with padding variations
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test raising BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 2)
```


# LLM-generated content at query #7
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with string input instead of bytes
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test special URL-safe characters
    assert base64_decode(b"Pj4-Pg==") == b">>>"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid character encoding
    try:
        base64_decode(b"\xff\xfe")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #8
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test data")
    decoded = base64_decode(encoded)
    assert decoded == b"test data"
    
    # Test with already decoded bytes
    decoded = base64_decode(b"aGVsbG8=")
    assert decoded == b"hello"
    
    # Test with padding
    decoded = base64_decode(b"YQ==")
    assert decoded == b"a"
    
    # Test with no padding
    decoded = base64_decode(b"YQ")
    assert decoded == b"a"
    
    # Test empty string
    decoded = base64_decode(b"")
    assert decoded == b""
    
    # Test invalid base64 data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special URL-safe characters
    decoded = base64_decode(b"a-_w")
    assert decoded == b"k\xef"
    
    # Test with non-ASCII characters in input (should be ignored)
    decoded = base64_decode(b"aGVsbG8=\xff")
    assert decoded == b"hello"
```


# LLM-generated content at query #9
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test decoding without padding
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test with string input
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with special characters (URL-safe)
    assert base64_decode(b"aGVsbG8gd29ybGQ") == b"hello world"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with non-ASCII characters in string input
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test long string
    long_input = b"a" * 100
    encoded = base64.b64encode(long_input).rstrip(b"=")
    assert base64_decode(encoded) == long_input
```


# LLM-generated content at query #10
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"YQ==") == b"a"
    
    # Test without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode(b"YQ") == b"a"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with empty input
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with special URL-safe characters
    encoded_special = base64_encode(b"\xff\xfe\xfd\xfc")
    decoded_special = base64_decode(encoded_special)
    assert decoded_special == b"\xff\xfe\xfd\xfc"
    
    # Test BadData exception for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8=")  # valid, but we want to test invalid
        # Actually test real invalid data
    with pytest.raises(BadData):
        base64_decode(b"\x00\x01\x02")
    
    # Test with non-ASCII characters in string input (should be ignored)
    assert base64_decode("aGVsbG8=\x80") == b"hello"
```


# LLM-generated content at query #11
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with padding
    encoded_padded = b"aGVsbG8="  # "hello" in base64
    assert base64_decode(encoded_padded) == b"hello"
    
    # Test decoding without padding
    encoded_no_pad = b"aGVsbG8"  # "hello" without padding
    assert base64_decode(encoded_no_pad) == b"hello"
    
    # Test decoding string input
    encoded_str = "aGVsbG8gd29ybGQ="  # "hello world" as string
    assert base64_decode(encoded_str) == b"hello world"
    
    # Test decoding with special characters (URL-safe)
    encoded_url = base64_encode(b"\xff\xfe\x00\x01")
    assert base64_decode(encoded_url) == b"\xff\xfe\x00\x01"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding single character
    assert base64_decode(b"aA==") == b"a"
    
    # Test decoding raises BadData for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding raises BadData for truncated input
    try:
        base64_decode(b"aGVsbG8")  # valid but incomplete padding
        # This should succeed since we add padding
    except BadData:
        assert False, "Should not raise BadData for incomplete padding"
```


# LLM-generated content at query #12
#--------------------------

```python
def test_base64_decode():
    # Test with valid URL-safe base64 encoded string
    original = b"hello world"
    encoded = base64.b64encode(original).rstrip(b"=")
    result = base64_decode(encoded)
    assert result == original

    # Test with padding
    encoded_padded = base64.b64encode(b"test")
    result = base64_decode(encoded_padded)
    assert result == b"test"

    # Test with string input
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"

    # Test with empty string
    result = base64_decode("")
    assert result == b""

    # Test with invalid characters
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should raise BadData"
    except BadData:
        pass

    # Test with incorrect padding
    try:
        base64_decode("aGVsbG8")
        assert False, "Should raise BadData"
    except BadData:
        pass

    # Test with bytes input
    result = base64_decode(b"d29ybGQ=")
    assert result == b"world"
```


# LLM-generated content at query #13
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test decoding string input
    encoded_str = base64_encode("world")
    assert base64_decode(encoded_str) == b"world"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8") == b"hello"  # Missing padding
    assert base64_decode(b"aGVsbG8=") == b"hello"  # Correct padding
    
    # Test decoding binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
    
    # Test decoding special characters
    assert base64_decode(b"-_0") == b"\xfb\xff"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test bytes with non-ASCII characters
    try:
        base64_decode("hello\x80".encode("latin-1"))
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #14
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with string input
    decoded_from_str = base64_decode("aGVsbG8gd29ybGQ")
    assert decoded_from_str == b"hello world"
    
    # Test with empty data
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with padding variants
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YQ") == b"a"  # URL-safe without padding
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with various byte values
    test_cases = [
        b"\x00\x01\x02",
        b"\xff\xfe\xfd",
        b"test data with spaces and !@#$%^&*()",
        bytes(range(256)),
    ]
    for case in test_cases:
        encoded = base64_encode(case)
        decoded = base64_decode(encoded)
        assert decoded == case
    
    # Test BadData exception on invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\xff\xff")
    
    # Test with bytes that have incorrect characters
    with pytest.raises(BadData):
        base64_decode("invalid base64!@#$")
```


# LLM-generated content at query #15
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test decoding bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test URL-safe characters
    assert base64_decode("aGVsbG8_d29ybGQ=") == b"hello?world"
    
    # Test with special characters
    assert base64_decode("YSBi") == b"a b"
    
    # Test with numbers
    assert base64_decode("MTIzNDU2Nzg5MA==") == b"1234567890"
    
    # Test raises BadData for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test raises BadData for truncated input
    try:
        base64_decode("SGVs")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #16
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with string input
    encoded_str = base64_encode("hello world")
    decoded_str = base64_decode(encoded_str)
    assert decoded_str == b"hello world"
    
    # Test with unicode text
    encoded_unicode = base64_encode("héllo wörld")
    decoded_unicode = base64_decode(encoded_unicode)
    assert decoded_unicode == "héllo wörld".encode("utf-8")
    
    # Test with empty string
    encoded_empty = base64_encode(b"")
    decoded_empty = base64_decode(encoded_empty)
    assert decoded_empty == b""
    
    # Test with single byte
    encoded_single = base64_encode(b"\x00")
    decoded_single = base64_decode(encoded_single)
    assert decoded_single == b"\x00"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    decoded_binary = base64_decode(encoded_binary)
    assert decoded_binary == binary_data
    
    # Test with padding
    encoded_padding = base64_encode(b"test")
    decoded_padding = base64_decode(encoded_padding)
    assert decoded_padding == b"test"
    
    # Test invalid data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input
    encoded_bytes = base64_encode(b"bytes input")
    decoded_bytes = base64_decode(encoded_bytes)
    assert decoded_bytes == b"bytes input"
```


# LLM-generated content at query #17
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with special URL-safe characters
    original = b"\xfb\xff\xff\xff\xff"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with binary data
    original = bytes(range(256))
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with non-ASCII characters in string input (should be ignored)
    assert base64_decode("aGVsbG8=\x80") == b"hello"
    
    # Test with single character
    assert base64_decode(b"aA==") == b"h"
    assert base64_decode(b"aA") == b"h"
```


# LLM-generated content at query #18
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with padding
    encoded_padding = base64_encode(b"test")
    assert base64_decode(encoded_padding) == b"test"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with URL-safe characters
    encoded_url = base64_encode(b"hello+world")
    decoded_url = base64_decode(encoded_url)
    assert decoded_url == b"hello+world"
    
    # Test with different lengths
    for i in range(1, 10):
        data = b"x" * i
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with ascii encoding errors (non-ascii chars ignored)
    result = base64_decode("aGVsbG8=\x80")
    assert result == b"hello"


# LLM-generated content at query #19
#--------------------------

```python
def test_base64_decode():
    # Test basic string decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"VGVzdA==") == b"Test"
    
    # Test URL-safe characters
    assert base64_decode("_-xq") == b"\xfb\xc6\xa8"
    
    # Test with special characters
    result = base64_decode("dGVzdC11cmw=")
    assert result == b"test-url"
    
    # Test longer string
    test_data = b"Hello, World! This is a test."
    encoded = base64_encode(test_data)
    decoded = base64_decode(encoded)
    assert decoded == test_data
    
    # Test that invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test that non-base64 characters raise BadData
    with pytest.raises(BadData):
        base64_decode("Hello World")
```


# LLM-generated content at query #20
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with no padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test special URL-safe characters
    original = b"\xff\xfb\x00\x01"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes containing non-ASCII characters
    result = base64_decode(b"8J+YgQ==")
    assert result == b"\xf0\x9f\x98\x81"  # UTF-8 encoded emoji
    
    # Test very long string
    long_data = b"x" * 1000
    encoded = base64_encode(long_data)
    assert base64_decode(encoded) == long_data
    
    # Test that padding is added correctly
    assert base64_decode(b"YQ") == b"a"
    assert base64_decode(b"YWI") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
```


# LLM-generated content at query #21
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test decode without padding
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test single character
    assert base64_decode("WA==") == b"X"
    
    # Test special characters
    assert base64_decode("dGVzdC1f") == b"test-_"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with padding variants
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"
```


# LLM-generated content at query #22
#--------------------------

```python
def test_base64_decode():
    # Test with standard URL-safe base64 string
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test with padding removed
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test with empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test with ASCII characters
    result = base64_decode("YWJjZGVmZw==")
    assert result == b"abcdefg"
    
    # Test with numbers
    result = base64_decode("MTIzNDU2")
    assert result == b"123456"
    
    # Test with special characters
    result = base64_decode("YSBiIGM=")
    assert result == b"a b c"
    
    # Test with URL-safe characters (no padding)
    result = base64_decode("aGVsbG8")
    assert result == b"hello"
    
    # Test with invalid base64 data should raise BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII bytes input
    result = base64_decode(b"w6llbMOz")
    assert result == "ëeló".encode("utf-8")
```


# LLM-generated content at query #23
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("VGVzdA==") == b"Test"
    assert base64_decode("") == b""
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("VGVzdA") == b"Test"
    
    # Test decoding bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test decoding with special URL-safe characters
    assert base64_decode("_-w=") == b"\xff\xec"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode(b"SGVsbG8=\xff") == b"Hello"
```


# LLM-generated content at query #24
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with string input
    decoded_from_str = base64_decode("aGVsbG8gd29ybGQ")
    assert decoded_from_str == b"hello world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with padding
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    
    # Test with URL-safe characters
    assert base64_decode(b"_-_w") == b"\xff\xef\xc0"
    
    # Test invalid data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in string input
    decoded_ascii = base64_decode("aGVsbG8=")
    assert decoded_ascii == b"hello"
    
    # Test various byte values
    for i in range(256):
        single_byte = bytes([i])
        encoded = base64_encode(single_byte)
        decoded = base64_decode(encoded)
        assert decoded == single_byte, f"Failed for byte value {i}"


# LLM-generated content at query #25
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with different lengths
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    
    # Test with bytes input
    assert base64_decode(b"d29ybGQ=") == b"world"
    
    # Test with string input
    assert base64_decode("d29ybGQ=") == b"world"
    
    # Test with empty string
    assert base64_decode(b"") == b""
    
    # Test with special characters
    assert base64_decode(b"aGVsbG8gd29ybGQh") == b"hello world!"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input
    try:
        base64_decode(b"\xff\xfe\x00")
    except BadData:
        pass
```


# LLM-generated content at query #26
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test with string input
    decoded = base64_decode(base64_encode("test string"))
    assert decoded == b"test string"

    # Test with padding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test single character
    assert base64_decode("YQ==") == b"a"
    
    # Test special characters
    assert base64_decode(base64_encode(b"\x00\x01\x02")) == b"\x00\x01\x02"
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("aGVs\x00bG8=")
```


# LLM-generated content at query #27
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test decoding with special characters
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test decoding bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    
    # Test decoding raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test decoding with special characters like underscore and dash
    assert base64_decode("Pj4_Pg==") == b">>\xfe>"  # underscore is used instead of /
    
    # Test decoding with dash
    assert base64_decode("Pj4-Pg==") == b">>\xfb>"  # dash is used instead of +
```


# LLM-generated content at query #28
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test with URL-safe characters
    assert base64_decode("aGVsbG8td29ybGQ") == b"hello-world"
    assert base64_decode("YSBi") == b"a b"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with special characters
    assert base64_decode("dGVzdC5wYXRoL3NvbWV0aGluZw==") == b"test.path/something"
    
    # Test raising BadData for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode("SGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with numbers
    assert base64_decode("MTIzNDU2Nzg5MA==") == b"1234567890"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data


# LLM-generated content at query #29
#--------------------------

```python
def test_base64_decode():
    # Test normal decode
    original = b"hello world"
    encoded = base64_encode(original)
    result = base64_decode(encoded)
    assert result == original

    # Test with string input
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"

    # Test with bytes input
    result = base64_decode(b"d29ybGQ=")
    assert result == b"world"

    # Test with padding removed
    encoded = base64_encode(b"test")
    assert b"=" not in encoded  # no padding
    result = base64_decode(encoded)
    assert result == b"test"

    # Test empty string
    result = base64_decode("")
    assert result == b""

    # Test single character
    result = base64_decode("Zg==")
    assert result == b"f"

    # Test with URL-safe characters
    encoded = base64_encode(b"\xff\xfe")
    assert b"+" not in encoded  # should use URL-safe chars
    result = base64_decode(encoded)
    assert result == b"\xff\xfe"

    # Test invalid base64 data
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
```


# LLM-generated content at query #30
#--------------------------

```python
def test_base64_decode():
    """Test base64_decode function with various inputs."""
    # Test with standard ASCII string
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test with URL-safe base64 without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test with bytes input
    result = base64_decode(b"V29ybGQ=")
    assert result == b"World"
    
    # Test with empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with binary data
    test_bytes = bytes(range(256))
    encoded = base64_encode(test_bytes)
    decoded = base64_decode(encoded)
    assert decoded == test_bytes
    
    # Test with special characters
    result = base64_decode("_-x")
    assert result is not None
    assert len(result) > 0
    
    # Test with invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with numbers
    result = base64_decode("MTIz")
    assert result == b"123"
    
    # Test with unicode characters
    result = base64_decode("w6TDtsO8")
    assert result == "äöü".encode("utf-8")
```


# LLM-generated content at query #31
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    encoded = base64_encode("hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test empty string
    encoded = base64_encode(b"")
    assert base64_decode(encoded) == b""
    
    # Test padding handling
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8==") == b"hello"
    
    # Test with various byte values
    test_bytes = bytes(range(256))
    encoded = base64_encode(test_bytes)
    assert base64_decode(encoded) == test_bytes
    
    # Test invalid base64 data raises BadData
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test single character
    encoded = base64_encode(b"a")
    assert base64_decode(encoded) == b"a"
    
    # Test with binary data
    binary_data = b"\x00\x01\x02\xff\xfe\xfd"
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test that ascii encoding is used for bytes input
    non_ascii = "héllo".encode("utf-8")
    encoded = base64_encode(non_ascii)
    result = base64_decode(encoded)
    assert result == non_ascii
    
    # Test ignore errors for non-ascii bytes
    result = base64_decode(b"\xff\xfe\xfd")
    assert isinstance(result, bytes)
```


# LLM-generated content at query #32
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    result = base64_decode(encoded)
    assert result == b"hello world"

    # Test empty string
    encoded = base64_encode(b"")
    result = base64_decode(encoded)
    assert result == b""

    # Test with various characters
    test_data = b"test data with spaces and !@#$%^&*()"
    encoded = base64_encode(test_data)
    result = base64_decode(encoded)
    assert result == test_data

    # Test with unicode
    test_data = "héllo wörld".encode("utf-8")
    encoded = base64_encode(test_data)
    result = base64_decode(encoded)
    assert result == test_data

    # Test with binary data
    test_data = bytes(range(256))
    encoded = base64_encode(test_data)
    result = base64_decode(encoded)
    assert result == test_data

    # Test invalid base64 data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"not valid base64!!")

    # Test string input is also accepted
    encoded = base64_encode(b"test")
    result = base64_decode(encoded.decode("ascii"))
    assert result == b"test"

    # Test padding variations
    test_data = b"short"
    encoded = base64_encode(test_data)
    assert base64_decode(encoded) == test_data

    # Test longer data
    test_data = b"a" * 1000
    encoded = base64_encode(test_data)
    result = base64_decode(encoded)
    assert result == test_data
```


# LLM-generated content at query #33
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test decode with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decode without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decode empty string
    assert base64_decode(b"") == b""
    
    # Test decode with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decode raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test decode with special URL-safe characters
    encoded_url = base64_encode(b"test?data=123")
    assert base64_decode(encoded_url) == b"test?data=123"
    
    # Test roundtrip with various data
    test_data = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"test data with spaces",
        b"binary\x00data",
        b"unicode text",
    ]
    for data in test_data:
        assert base64_decode(base64_encode(data)) == data
```


# LLM-generated content at query #34
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    
    # Test with bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test various padding scenarios
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test URL-safe characters
    assert base64_decode("_-w=") == b"\xff\xec"
    
    # Test with special characters
    assert base64_decode("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=") == bytes(range(32))
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters (should be ignored due to errors="ignore")
    assert base64_decode("SGVsbG8=") == b"Hello"
```


# LLM-generated content at query #35
#--------------------------

```python
def test_base64_decode():
    # Test basic string input
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original, "Basic string decoding failed"

    # Test bytes input
    original_bytes = b"test data with bytes"
    encoded_bytes = base64_encode(original_bytes)
    decoded_bytes = base64_decode(encoded_bytes)
    assert decoded_bytes == original_bytes, "Bytes input decoding failed"

    # Test empty string
    assert base64_decode(b"") == b"", "Empty string decoding failed"

    # Test special characters
    original_special = b"test!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
    encoded_special = base64_encode(original_special)
    decoded_special = base64_decode(encoded_special)
    assert decoded_special == original_special, "Special characters decoding failed"

    # Test invalid base64 string
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData for invalid base64"
    except BadData:
        pass

    # Test padding handling
    assert base64_decode(b"dGVzdA") == b"test", "Padding handling failed"

    # Test with string input type
    original_str = "Hello, 世界"
    encoded_str = base64_encode(original_str)
    decoded_str = base64_decode(encoded_str)
    assert decoded_str == original_str.encode(), "String input type decoding failed"
```


# LLM-generated content at query #36
#--------------------------

```python
def test_base64_decode():
    """Test base64_decode function."""
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with padding
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"
    
    # Test without padding (URL-safe format)
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with special characters that are valid in base64url
    encoded_special = base64_encode(b"\xff\xfe\xfd\xfc")
    decoded = base64_decode(encoded_special)
    assert decoded == b"\xff\xfe\xfd\xfc"
```


# LLM-generated content at query #37
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding with URL-safe characters
    result = base64_decode("Pjw_Pg==")
    assert result == b"><>?"
    
    # Test decoding bytes input
    result = base64_decode(b"VGVzdA==")
    assert result == b"Test"
    
    # Test decoding with no padding needed
    result = base64_decode("dGVzdA==")
    assert result == b"test"
    
    # Test decoding longer string
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test that invalid data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that non-base64 characters raise BadData
    try:
        base64_decode("Hello World")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #38
#--------------------------

```python
def test_base64_decode():
    # Test valid base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # Without padding
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with URL-safe characters
    assert base64_decode(b"aGVsbG8td29ybGQ=") == b"hello-world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test invalid base64 data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters
    original = b"\x00\x01\x02\xff\xfe"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with numbers and special chars
    assert base64_decode(b"MTIzNDU2Nzg5MA==") == b"1234567890"
```


# LLM-generated content at query #39
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test decode without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test decode empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decode with URL-safe characters
    result = base64_decode("dGVzdGluZy11cmwtc2FmZQ==")
    assert result == b"testing-url-safe"
    
    # Test decode bytes input
    result = base64_decode(b"d29ya3M=")
    assert result == b"works"
    
    # Test decode with special characters
    result = base64_decode("dGVzdC13aXRoLXNwZWNpYWwtY2hhcnM/")
    assert result == b"test-with-special-chars?"
    
    # Test decode with underscore (URL-safe)
    result = base64_decode("dGVzdF91bmRlcnNjb3Jl")
    assert result == b"test_underscore"
    
    # Test decode with dash (URL-safe)
    result = base64_decode("dGVzdC1kYXNo")
    assert result == b"test-dash"
    
    # Test decode invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test decode with non-ASCII characters (should be ignored)
    result = base64_decode("dGVzdA==" + "\x80\x81")
    assert result == b"test"
```


# LLM-generated content at query #40
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test data")
    decoded = base64_decode(encoded)
    assert decoded == b"test data"
    
    # Test with empty data
    encoded = base64_encode(b"")
    decoded = base64_decode(encoded)
    assert decoded == b""
    
    # Test padding is handled correctly
    assert base64_decode("aGVsbG8") == b"hello"  # Missing padding
    
    # Test with URL-safe characters
    encoded = base64_encode(b"data with + and /")
    decoded = base64_decode(encoded)
    assert b"+" in decoded or b"/" in decoded
    
    # Test round trip with various data
    test_data = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"abcd",
        b"test@#$%^&*()",
        b"\x00\x01\x02\xff\xfe",
        b"data with spaces and\nnewlines",
    ]
    for data in test_data:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Failed for data: {data}"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("aGVsbG8=")  # Valid, but we'll test exception for invalid chars
    
    # Test with bytes input
    encoded = base64_encode(b"bytes input")
    decoded = base64_decode(encoded)
    assert decoded == b"bytes input"
```


# LLM-generated content at query #41
#--------------------------

```python
def test_base64_decode():
    # Test normal decode
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test padding variations
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    assert base64_decode(b"aGVsbG8=") == b"hello"  # with one padding
    assert base64_decode(b"aGVsbG8==") == b"hello"  # with two padding
    
    # Test with URL-safe characters
    assert base64_decode(b"Pj4-Pg==") == b">>>"
    assert base64_decode(b"PDw8PA==") == b"<<<<"
    
    # Test binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test special characters
    assert base64_decode(b"AAECAw==") == b"\x00\x01\x02\x03"
    
    # Test invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")
```


# LLM-generated content at query #42
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test URL-safe characters
    assert base64_decode(b"Pj4-Pg==") == b">> >"
    assert base64_decode("Pj4-Pg==") == b">> >"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with padding adjustment (missing padding)
    assert base64_decode(b"YQ") == b"a"
    assert base64_decode(b"YWI") == b"ab"
    
    # Test with special characters
    assert base64_decode(b"Lg==") == b"."
    assert base64_decode(b"Lg") == b"."
    
    # Test invalid input raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")
    
    # Test with non-ASCII bytes in input (should be ignored due to errors="ignore")
    assert base64_decode(b"\xffSGVsbG8=") == b"Hello"
    assert base64_decode("SGVs\x80bG8=") == b"Hello"
```


# LLM-generated content at query #43
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with special characters (URL-safe base64)
    original = b"\xff\xfe\x00\x01"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with invalid data
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("not valid base64")
```


# LLM-generated content at query #44
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test data")
    decoded = base64_decode(encoded)
    assert decoded == b"test data"
    
    # Test empty string
    encoded = base64_encode(b"")
    decoded = base64_decode(encoded)
    assert decoded == b""
    
    # Test with padding
    encoded = base64_encode(b"a")
    decoded = base64_decode(encoded)
    assert decoded == b"a"
    
    # Test with special characters
    encoded = base64_encode(b"hello\nworld")
    decoded = base64_decode(encoded)
    assert decoded == b"hello\nworld"
    
    # Test with binary data
    encoded = base64_encode(b"\x00\x01\x02\xff")
    decoded = base64_decode(encoded)
    assert decoded == b"\x00\x01\x02\xff"
    
    # Test with bytes input
    encoded = base64_encode(b"test")
    decoded = base64_decode(encoded)
    assert decoded == b"test"
    
    # Test invalid base64 string raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-base64 characters
    try:
        base64_decode("not base64")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with incomplete padding
    encoded = base64_encode(b"test")[:-1]
    try:
        base64_decode(encoded)
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #45
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test without padding
    assert base64_decode("dGVzdA") == b"test"
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test URL-safe characters
    assert base64_decode("_-x") == b"\xff\xeb"
    
    # Test invalid input raises BadData
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid padding raises BadData
    try:
        base64_decode("abc")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #46
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decode
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with special URL-safe characters
    assert base64_decode("_-w") == b"\xff\xec"
    
    # Test raising BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("abc")
```


# LLM-generated content at query #47
#--------------------------

```python
def test_base64_decode():
    # Test normal valid base64 string
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test URL-safe base64 without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test with special characters
    result = base64_decode("dGVzdC11cmwtdG9rZW4")
    assert result == b"test-url-token"
    
    # Test raises BadData for invalid base64
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test raises BadData for non-base64 characters
    try:
        base64_decode("not valid base64")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with unicode string (ascii encoding)
    result = base64_decode("dGVzdA==")
    assert result == b"test"
```


# LLM-generated content at query #48
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding characters
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with URL-safe characters
    decoded = base64_decode(b"aGVsbG8gd29ybGQ")
    assert decoded == b"hello world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with no padding (minimal input)
    assert base64_decode(b"YQ") == b"a"
    
    # Test with text input (string type)
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with various valid inputs
    test_cases = [
        (b"SGVsbG8=", b"Hello"),
        (b"V29ybGQ=", b"World"),
        (b"MTIzNDU2", b"123456"),
    ]
    for encoded, expected in test_cases:
        assert base64_decode(encoded) == expected
    
    # Test with special characters
    original = bytes(range(256))
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test invalid base64 data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid base64 with wrong characters
    try:
        base64_decode(b"aGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #49
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test decoding with string input
    decoded = base64_decode(encoded.decode())
    assert decoded == b"hello world"
    
    # Test decoding empty string
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test decoding with padding
    encoded_padded = base64_encode(b"a")
    assert encoded_padded.endswith(b"=") == False  # noqa: E712
    decoded = base64_decode(encoded_padded)
    assert decoded == b"a"
    
    # Test decoding invalid data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test decoding with ascii encoding errors ignored
    decoded = base64_decode("aGVsbG8=")
    assert decoded == b"hello"
```


# LLM-generated content at query #50
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test decoding bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding single character
    result = base64_decode("QQ==")
    assert result == b"A"
    
    # Test decoding with URL-safe characters
    result = base64_decode("Pj4-Pg==")
    assert result == b">>>>
    
    # Test decoding binary data
    result = base64_decode("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=")
    expected = bytes(range(32))
    assert result == expected
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test that non-base64 characters raise BadData
    with pytest.raises(BadData):
        base64_decode("abc def")


# LLM-generated content at query #51
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input (not bytes)
    encoded_str = base64_encode("hello world")
    assert base64_decode(encoded_str) == b"hello world"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with missing padding (should still work)
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters that are URL-safe
    encoded_special = base64_encode(b"test\x00\xff")
    assert base64_decode(encoded_special) == b"test\x00\xff"
    
    # Test with various byte values
    for i in range(256):
        test_bytes = bytes([i])
        encoded = base64_encode(test_bytes)
        decoded = base64_decode(encoded)
        assert decoded == test_bytes
```


# LLM-generated content at query #52
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with URL-safe characters
    assert base64_decode(b"dGVzdC11cmw") == b"test-url"
    
    # Test special characters
    assert base64_decode(b"dGVzdF8t") == b"test_-"
    
    # Test with ASCII encoding
    result = base64_decode("YWJjMTIz")
    assert result == b"abc123"
    
    # Test raising BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with non-ASCII bytes
    assert base64_decode(b"w7zDvMO8") == b"\xc3\xbc\xc3\xbc\xc3\xbc"
```


# LLM-generated content at query #53
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    result = base64_decode(encoded)
    assert result == b"hello world"
    
    # Test decoding with string input
    encoded_str = base64_encode("test data")
    result = base64_decode(encoded_str)
    assert result == b"test data"
    
    # Test decoding with padding
    encoded_padded = base64_encode(b"a")
    result = base64_decode(encoded_padded)
    assert result == b"a"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding with special characters
    encoded_special = base64_encode(b"\x00\xff\x7f")
    result = base64_decode(encoded_special)
    assert result == b"\x00\xff\x7f"
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\xff\xff\xff\xff")
```


# LLM-generated content at query #54
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test BadData exception with invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8\xff")  # invalid character
    
    # Test roundtrip with various data
    test_cases = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"test data here",
        bytes(range(256)),  # all byte values
    ]
    
    for data in test_cases:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Roundtrip failed for {data!r}"
    
    # Test with special characters that shouldn't affect decoding
    assert base64_decode(b"aGVs_bG8=") == b"hel\xbb\x18"  # _ replaces /
    assert base64_decode(b"aGVsbG8-") == b"\x68\x65\x6c\x6c\xf8"  # - replaces +
```


# LLM-generated content at query #55
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    
    # Test with bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test with special characters
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"
    assert base64_decode("Zm9vLmJhcg==") == b"foo.bar"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with invalid input
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with extra padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test with long string
    long_text = "a" * 1000
    encoded = base64_encode(long_text)
    assert base64_decode(encoded) == long_text.encode()
```


# LLM-generated content at query #56
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    result = base64_decode(encoded)
    assert result == b"hello world"
    
    # Test with string input
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test with empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with padding
    encoded = base64_encode(b"a")
    result = base64_decode(encoded)
    assert result == b"a"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    result = base64_decode(encoded)
    assert result == binary_data
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with invalid characters
    with pytest.raises(BadData):
        base64_decode("aGVsbG8$")
```


# LLM-generated content at query #57
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 string
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello", f"Expected b'Hello', got {result}"
    
    # Test URL-safe base64 (no padding)
    result = base64_decode("SGVsbG8")
    assert result == b"Hello", f"Expected b'Hello', got {result}"
    
    # Test empty string
    result = base64_decode("")
    assert result == b"", f"Expected b'', got {result}"
    
    # Test bytes input
    result = base64_decode(b"VGVzdA==")
    assert result == b"Test", f"Expected b'Test', got {result}"
    
    # Test with special characters (URL-safe)
    result = base64_decode("aGVsbG8td29ybGQ")
    assert result == b"hello-world", f"Expected b'hello-world', got {result}"
    
    # Test longer string
    result = base64_decode("VGhpcyBpcyBhIHRlc3Q=")
    assert result == b"This is a test", f"Expected b'This is a test', got {result}"
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test invalid characters raises BadData
    with pytest.raises(BadData):
        base64_decode("ABC123!@#")
    
    # Test very short invalid input
    with pytest.raises(BadData):
        base64_decode("a")
    
    # Test non-ASCII characters in input
    with pytest.raises(BadData):
        base64_decode("héllo")


# LLM-generated content at query #58
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with no padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with special characters
    encoded_special = base64_encode(b"\x00\x01\x02")
    assert base64_decode(encoded_special) == b"\x00\x01\x02"
    
    # Test with long data
    long_data = b"test" * 100
    encoded_long = base64_encode(long_data)
    assert base64_decode(encoded_long) == long_data
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes that have invalid characters
    try:
        base64_decode(b"aGVsbG8\n")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with None input
    try:
        base64_decode(None)  # type: ignore
        assert False, "Should have raised TypeError or BadData"
    except (TypeError, BadData):
        pass
```


# LLM-generated content at query #59
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with bytes input
    assert base64_decode(b"d29ybGQ=") == b"world"
    
    # Test with URL-safe characters
    assert base64_decode("dGVzdC11cmw") == b"test-url"
    
    # Test with special characters
    assert base64_decode("ISQlJio=") == b"!$%&*"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with padding that needs to be added
    assert base64_decode("dGVzdA") == b"test"
    
    # Test with multiple padding characters
    assert base64_decode("YQ==") == b"a"
    
    # Test with ascii encoding errors (should ignore non-ascii)
    assert base64_decode("aGVsbG8=\xff") == b"hello"
```


# LLM-generated content at query #60
#--------------------------

```python
def test_base64_decode():
    # Test valid base64 URL-safe string
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8=")
    assert result == b"Hello"
    
    # Test with special URL-safe characters
    result = base64_decode("_-xq")
    assert result == b"\xff\xf1\xaa"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    result = base64_decode("SGVsbG8=\x80")
    assert result == b"Hello"


# LLM-generated content at query #61
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with URL-safe characters
    assert base64_decode("Pj4_Pz8") == b">>???"
    assert base64_decode("Pj4_Pz8=") == b">>???"
    
    # Test with special characters
    assert base64_decode("") == b""
    assert base64_decode("") == b""
    
    # Test invalid input raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\xff\xff\xff")
    
    # Test with various string inputs
    test_strings = ["hello", "world", "test123", "a" * 100, "data:image/png;base64,"]
    for s in test_strings:
        encoded = base64_encode(s)
        decoded = base64_decode(encoded)
        assert decoded == s.encode()
    
    # Test with bytes input
    test_bytes = [b"hello", b"world", b"test123", b"\x00\x01\x02\xff"]
    for b in test_bytes:
        encoded = base64_encode(b)
        decoded = base64_decode(encoded)
        assert decoded == b
    
    # Test padding variations
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWE=") == b"aa"
    assert base64_decode("YWFh") == b"aaa"
    assert base64_decode("YWFhYQ==") == b"aaaa"


# LLM-generated content at query #62
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 string
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test with special characters (URL-safe)
    result = base64_decode("_-x")
    assert result == b"\xff\xeb"
    
    # Test with only padding
    result = base64_decode("==")
    assert result == b""
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test invalid data with wrong characters
    with pytest.raises(BadData):
        base64_decode("ABC@123")
```


# LLM-generated content at query #63
#--------------------------

```python
def test_base64_decode():
    # Test with valid URL-safe base64 encoded string
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"

    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"

    # Test with padding removed
    assert base64_decode(b"aGVsbG8") == b"hello"

    # Test with empty string
    assert base64_decode("") == b""

    # Test with single character
    assert base64_decode(b"Zg==") == b"f"

    # Test with special URL-safe characters
    assert base64_decode(b"_-A=") == b"\xfb\xe0"

    # Test with invalid data raises BadData
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test with invalid characters
    try:
        base64_decode("aGVsbG8=" + chr(255))
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test with non-base64 alphabet characters
    try:
        base64_decode("aGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #64
#--------------------------

```python
def test_base64_decode():
    # Test basic byte decoding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test string input
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test without padding
    assert base64_decode(b"dGVzdA") == b"test"
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with URL-safe characters
    assert base64_decode(b"_-x") == b"\xfb\xe7"  # - and _ are used in url-safe
    
    # Test binary data
    binary_data = b"\x00\x01\x02\xff\xfe\xfd"
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test raises BadData for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test raises BadData for non-base64 characters
    try:
        base64_decode(b"\xff\xfe\xfd")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters
    assert base64_decode(b"") == b""


# LLM-generated content at query #65
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test padding handling
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8==") == b"hello"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with various data
    assert base64_decode(b"") == b""
    assert base64_decode(base64_encode(b"test data 123")) == b"test data 123"
    assert base64_decode(base64_encode(b"\x00\x01\x02\xff")) == b"\x00\x01\x02\xff"
    
    # Test BadData exception for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode(b"\xff\xff\xff")
```


# LLM-generated content at query #66
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original, "Should decode base64 encoded data correctly"
    
    # Test decoding with padding
    encoded_with_padding = b"aGVsbG8gd29ybGQ="
    decoded = base64_decode(encoded_with_padding)
    assert decoded == b"hello world", "Should decode data with padding"
    
    # Test decoding without padding
    encoded_without_padding = b"aGVsbG8gd29ybGQ"
    decoded = base64_decode(encoded_without_padding)
    assert decoded == b"hello world", "Should decode data without padding"
    
    # Test decoding string input
    decoded = base64_decode("aGVsbG8gd29ybGQ=")
    assert decoded == b"hello world", "Should decode string input"
    
    # Test decoding empty string
    decoded = base64_decode(b"")
    assert decoded == b"", "Should decode empty string"
    
    # Test decoding single character
    encoded_single = base64_encode(b"a")
    decoded = base64_decode(encoded_single)
    assert decoded == b"a", "Should decode single character"
    
    # Test decoding with special characters
    special = b"test with spaces!@#$%^&*()"
    encoded_special = base64_encode(special)
    decoded = base64_decode(encoded_special)
    assert decoded == special, "Should decode data with special characters"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should raise BadData for invalid base64"
    except BadData:
        pass
    
    # Test bytes with null bytes
    with_nulls = b"\x00\x01\x02\xff"
    encoded_nulls = base64_encode(with_nulls)
    decoded = base64_decode(encoded_nulls)
    assert decoded == with_nulls, "Should decode data with null bytes"
```


# LLM-generated content at query #67
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decode without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test decode with URL-safe characters
    result = base64_decode("aGVsbG8_d29ybGQ-")
    assert result == b"hello?world>"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test single character
    result = base64_decode("WA==")
    assert result == b"X"
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8=")
    assert result == b"Hello"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters
    result = base64_decode("AAAA")
    assert result == b"\x00\x00\x00"


# LLM-generated content at query #68
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with string input
    encoded_str = base64_encode("test string")
    assert base64_decode(encoded_str) == b"test string"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding URL-safe base64 (using - and _ instead of + and /)
    assert base64_decode("Pj4-Pj4_") == b">>>>>>?"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid base64 with wrong characters
    try:
        base64_decode("aGVsbG8!=")
        assert False, "Should have raised BadData"
    except BadData:
        pass


# LLM-generated content at query #69
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding string input
    encoded = base64_encode("test data")
    result = base64_decode(encoded)
    assert isinstance(result, bytes)
    assert result == b"test data"
    
    # Test padding handling
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode(b"aGVs") == b"hel"
    
    # Test with empty data
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test with special characters that are URL-safe
    encoded = base64_encode(b"\x00\x01\x02")
    assert base64_decode(encoded) == b"\x00\x01\x02"
    
    # Test that invalid data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input that has correct padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
```


# LLM-generated content at query #70
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with URL-safe characters
    url_safe_data = b"test+data=="
    url_safe_encoded = base64_encode(url_safe_data)
    assert base64_decode(url_safe_encoded) == url_safe_data
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with empty data
    assert base64_decode("") == b""
    
    # Test with single character
    assert base64_decode("YQ==") == b"a"
    
    # Test with special characters
    special_data = b"\x00\x01\x02\xff"
    special_encoded = base64_encode(special_data)
    assert base64_decode(special_encoded) == special_data
    
    # Test with unicode characters encoded
    unicode_data = "héllo".encode("utf-8")
    unicode_encoded = base64_encode(unicode_data)
    assert base64_decode(unicode_encoded) == unicode_data
    
    # Test padding handling
    assert base64_decode("aGVsbG8") == b"hello"  # Missing padding
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("aGVsbG8!!!")
```


# LLM-generated content at query #71
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with mixed case
    encoded = base64_encode(b"Test123!@#")
    assert base64_decode(encoded) == b"Test123!@#"
    
    # Test invalid base64 raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input
    try:
        base64_decode("héllo")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes object
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
```


# LLM-generated content at query #72
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with string input
    encoded = base64_encode("test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with padding
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input
    assert base64_decode(b"d29ybGQ=") == b"world"
    
    # Test with ASCII string input
    assert base64_decode("d29ybGQ=") == b"world"
```


# LLM-generated content at query #73
#--------------------------

```python
def test_base64_decode():
    # Test with valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with URL-safe characters
    assert base64_decode("aGVsbG8_d29ybGQ=") == b"hello?world"
    assert base64_decode("aGVsbG8td29ybGQ=") == b"hello-world"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test that BadData is raised for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode("SGVsbG8=" + "invalid")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\x80") == b"Hello"
```


# LLM-generated content at query #74
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with padding
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special characters
    original = b"test_data+with/special?chars"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test invalid input raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with non-ASCII bytes
    original = b"\xff\xfe\x00\x01"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
```


# LLM-generated content at query #75
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    result = base64_decode("dGVzdA==")
    assert result == b"test"

    # Test base64 without padding
    result = base64_decode("dGVzdA")
    assert result == b"test"

    # Test empty string
    result = base64_decode("")
    assert result == b""

    # Test with URL-safe characters
    result = base64_decode("dGVzdC13aXRoLXVybC1zYWZlLWNoYXJz")
    assert result == b"test-with-url-safe-chars"

    # Test with bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"

    # Test with special characters
    result = base64_decode("dGVzdC13aXRoLXNwZWNpYWwtY2hhcnM_")
    assert result == b"test-with-special-chars?"

    # Test with trailing equals signs
    result = base64_decode("dGVzdA")
    assert result == b"test"

    # Test invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test non-ASCII characters (should be ignored)
    result = base64_decode("dGVzdA==" + "ñ")
    assert result == b"test"

    # Test with numbers
    result = base64_decode("MTIzNDU2Nzg5MA==")
    assert result == b"1234567890"
```


# LLM-generated content at query #76
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test with empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with URL-safe characters
    encoded_url = base64_encode(b"test?data=1&more=2")
    assert base64_decode(encoded_url) == b"test?data=1&more=2"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
    
    # Test invalid base64 raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode(b"aGVs@bG8=")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in string input
    try:
        base64_decode("héllo")  # non-ASCII characters
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #77
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding empty bytes
    assert base64_decode(b"") == b""
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding with various byte lengths
    assert base64_decode(b"YQ==") == b"a"  # 1 byte
    assert base64_decode(b"YWI=") == b"ab"  # 2 bytes
    assert base64_decode(b"YWJj") == b"abc"  # 3 bytes
    
    # Test decoding raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test decoding raises BadData for truncated input
    with pytest.raises(BadData):
        base64_decode(b"aGVs")
    
    # Test roundtrip with various data
    test_cases = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"test data",
        b"binary\x00data",
        b"unicode_émoji",
    ]
    for data in test_cases:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Roundtrip failed for {data!r}"
```


# LLM-generated content at query #78
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    
    # Test URL-safe characters
    assert base64_decode("dGVzdC11cmw=") == b"test-url"
    assert base64_decode("dGVzdC11cmw") == b"test-url"
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
    
    # Test with special characters
    assert base64_decode("YSBi") == b"a b"
    assert base64_decode("YSBi") == b"a b"
    
    # Test binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with equal signs padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with different padding lengths
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test Unicode string input (should be ignored due to ascii encoding)
    assert base64_decode("U8Oww7zDtg==") == b"S\xc3\xb0\xc3\xbc\xc3\xb6"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should raise BadData"
    except BadData:
        pass
    
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should raise BadData"
    except BadData:
        pass
    
    # Test with ascii errors='ignore'
    assert base64_decode("test") == b"\xb5\xeb"
    
    # Test with None-like input (should raise error)
    try:
        base64_decode(None)  # type: ignore
        assert False, "Should raise error"
    except (TypeError, AttributeError):
        pass
```


# LLM-generated content at query #79
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test without padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"d29ybGQ=") == b"world"
    
    # Test with special URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    
    # Test with minus and underscore characters
    assert base64_decode("_-5-") == b"\xff\xee\xe7"
    
    # Test raising BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test raising BadData for non-base64 characters
    with pytest.raises(BadData):
        base64_decode("aGVsbG8$")
    
    # Test with explicit ASCII encoding
    assert base64_decode("dGVzdA==") == b"test"
```


# LLM-generated content at query #80
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test with text input
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test with padding
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test invalid data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode("aGVsbG8$")
        assert False, "Expected BadData exception"
    except BadData:
        pass
```


# LLM-generated content at query #81
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test string")
    assert base64_decode(encoded) == b"test string"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with special characters
    encoded = base64_encode(b"\x00\x01\x02\xff")
    assert base64_decode(encoded) == b"\x00\x01\x02\xff"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test invalid data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with unicode string input
    encoded = base64_encode("héllo")
    assert base64_decode(encoded) == "héllo".encode("utf-8")
    
    # Test bytes-like object
    import sys
    if sys.version_info >= (3, 0):
        assert base64_decode(bytearray(b"aGVsbG8=")) == b"hello"
```


# LLM-generated content at query #82
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 encoded string
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test without padding
    result = base64_decode("aGVsbG8")
    assert result == b"hello"
    
    # Test URL-safe characters
    result = base64_decode("dGVzdF91cmw")
    assert result == b"test_url"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"d29ybGQ=")
    assert result == b"world"
    
    # Test with special characters
    result = base64_decode("dGVzdC0t")
    assert result == b"test--"
    
    # Test with underscore
    result = base64_decode("dGVzdF8=")
    assert result == b"test_"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
```


# LLM-generated content at query #83
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decoding with padding
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test decoding without padding
    result = base64_decode("aGVsbG8")
    assert result == b"hello"
    
    # Test decoding string input
    result = base64_decode("d29ybGQ=")
    assert result == b"world"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding with special URL-safe characters
    original = b"\xfb\xff\xff\xff\xff"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test that invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test that non-ASCII bytes raise BadData
    with pytest.raises(BadData):
        base64_decode(b"\xff\xff\xff")
```


# LLM-generated content at query #84
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test padding handling
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test URL-safe characters
    assert base64_decode(b"_-w") == b"\xff\xef"
    
    # Test with non-ascii characters in input (should ignore them)
    assert base64_decode("aGVsbG8=" + "\u00e9") == b"hello"
    
    # Test with special characters in string input
    assert base64_decode("aGVsbG8=\n\t") == b"hello"
    
    # Test BadData exception for invalid input
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode(b"\xff\xff")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with additional padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    
    # Test roundtrip with various byte sequences
    for i in range(256):
        byte_val = bytes([i])
        encoded = base64_encode(byte_val)
        decoded = base64_decode(encoded)
        assert decoded == byte_val, f"Roundtrip failed for byte {i}"
```


# LLM-generated content at query #85
#--------------------------

```python
def test_base64_decode():
    # Test decoding a simple string
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with no padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
    
    # Test decoding with text input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding invalid data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding data with special characters
    assert base64_decode(b"dGVzdF91cmw") == b"test_url"
```


# LLM-generated content at query #86
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("d29ybGQ=") == b"world"
    
    # Test without padding
    assert base64_decode("aGVsbG8") == b"hello"
    assert base64_decode("d29ybGQ") == b"world"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    assert base64_decode("PDw8PA==") == b"<<<<"
    
    # Test with underscore and dash (URL-safe variants)
    assert base64_decode("_-8=") == b"\xff\xef"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid base64 with wrong characters
    try:
        base64_decode("aGVs bG8=")  # space in base64
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test non-ASCII string (should ignore non-ASCII characters)
    assert base64_decode("aGVsbG8=" + chr(200)) == b"hello"
```


# LLM-generated content at query #87
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with padding
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with special characters in base64
    encoded_special = base64_encode(b"\x00\x01\x02")
    assert base64_decode(encoded_special) == b"\x00\x01\x02"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input (should be ignored)
    assert base64_decode("aGVsbG8=\x80\x81") == b"hello"
    
    # Test with unicode string
    assert base64_decode("aGVsbG8=") == b"hello"
```


# LLM-generated content at query #88
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with string input
    encoded = base64_encode("hello world")
    decoded = base64_decode("aGVsbG8gd29ybGQ")
    assert decoded == b"hello world"
    
    # Test empty input
    assert base64_decode(b"") == b""
    
    # Test with padding
    assert base64_decode(b"YQ") == b"a"
    assert base64_decode(b"YWI") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    
    # Test with full padding
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    
    # Test invalid data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special URL-safe characters
    assert base64_decode(b"_-w") == b"\xff\xeb"
    
    # Test with numbers
    encoded = base64_encode(b"1234567890")
    assert base64_decode(encoded) == b"1234567890"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with unicode string input
    encoded = base64_decode("aGVsbG8gd29ybGQ=")
    assert encoded == b"hello world"
```


# LLM-generated content at query #89
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    original_data = b"hello world"
    encoded = base64_encode(original_data)
    decoded = base64_decode(encoded)
    assert decoded == original_data

    # Test with empty data
    assert base64_decode(b"") == b""

    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding

    # Test with URL-safe characters
    assert base64_decode(b"YQ==.A") == b"a?\x00"
    
    # Test with various data types
    test_data = b"test data with numbers 123"
    encoded = base64_encode(test_data)
    decoded = base64_decode(encoded)
    assert decoded == test_data
    
    # Test with unicode/ascii input
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test BadData exception for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"invalid")
    
    # Test with long strings
    long_data = b"x" * 1000
    encoded = base64_encode(long_data)
    decoded = base64_decode(encoded)
    assert decoded == long_data
```


# LLM-generated content at query #90
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with missing padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test decoding with different input lengths
    assert base64_decode("aGVsbG8gd29ybGQ") == b"hello world"
    
    # Test decoding special characters
    assert base64_decode("Pz8_Pz8") == b"???\x00\x00"
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test invalid characters raise BadData
    with pytest.raises(BadData):
        base64_decode("aGVs bG8=")  # space is invalid
    
    # Test non-base64 bytes raise BadData
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")
    
    # Test roundtrip with various lengths
    for length in [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64]:
        test_data = b"x" * length
        assert base64_decode(base64_encode(test_data)) == test_data
    
    # Test with bytes-like object
    assert isinstance(base64_decode(b"aGVsbG8="), bytes)
```


# LLM-generated content at query #91
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with binary data
    binary_data = b"\x00\x01\x02\xff"
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with URL-safe characters
    url_safe_data = base64_encode(b"test?data=123")
    assert base64_decode(url_safe_data) == b"test?data=123"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test invalid input raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode(b"\xff\xff\xff\xff")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters that are valid in base64
    assert base64_decode("_-A=") == b"\xfb\xff"  # Using URL-safe characters
```


# LLM-generated content at query #92
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8=")
    assert result == b"Hello"
    
    # Test URL-safe characters
    result = base64_decode("_-xK")
    assert result == b"\xff\x12"
    
    # Test with special characters
    result = base64_decode("PDw_Pz8-Pg==")
    assert result == b"<<??>>"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test too short string raises BadData
    with pytest.raises(BadData):
        base64_decode("a")
    
    # Test with non-ASCII characters (should be ignored in decode)
    result = base64_decode("SGVsbG8=" + "\x80\x81")
    assert result == b"Hello"
```


# LLM-generated content at query #93
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    result = base64_decode(encoded)
    assert result == b"hello world"
    
    # Test decoding with string input
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding with padding
    result = base64_decode("aA==")
    assert result == b"h"
    
    # Test decoding without padding
    result = base64_decode("aA")
    assert result == b"h"
    
    # Test decoding with bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test invalid base64 data with wrong characters
    with pytest.raises(BadData):
        base64_decode("aGVsbG8=" + chr(0x80))
    
    # Test decode roundtrip with various data
    test_data = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"test data with spaces",
        bytes(range(256)),
    ]
    for data in test_data:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Roundtrip failed for {data!r}"
```


# LLM-generated content at query #94
#--------------------------

```python
def test_base64_decode():
    # Test basic string decoding
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8=")
    assert result == b"Hello"
    
    # Test without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test URL-safe characters
    result = base64_decode("_-x")
    assert result == b"\xfb\xe7"
    
    # Test with special characters
    result = base64_decode("aGVsbG8gd29ybGQ=")
    assert result == b"hello world"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input (should be ignored)
    result = base64_decode("SGVsbG8=\x80\x81")
    assert result == b"Hello"
```


# LLM-generated content at query #95
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test padding variations
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with special characters (URL-safe)
    encoded_url = base64_encode(b"hello+world/foo")
    assert base64_decode(encoded_url) == b"hello+world/foo"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode("aGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII bytes (should be ignored due to errors="ignore")
    assert base64_decode(b"aGVsbG8\xff") == b"hello"
```


# LLM-generated content at query #96
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"YQ==") == b"a"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with unicode characters
    original = "héllo wörld".encode()
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with empty input
    assert base64_decode(b"") == b""
    
    # Test with bytes that need padding
    assert base64_decode(b"YQ") == b"a"
    assert base64_decode(b"YQ==") == b"a"
    
    # Test invalid base64 data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
```


# LLM-generated content at query #97
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test binary data
    original = b"\x00\x01\x02\xff"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test whitespace is ignored
    assert base64_decode("SGVs bG8=") == b"Hello"
```


# LLM-generated content at query #98
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    result = base64_decode(encoded)
    assert result == b"hello world"
    
    # Test with string input
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test with empty input
    result = base64_decode("")
    assert result == b""
    
    # Test with URL-safe characters
    encoded = base64_encode(b"test data with + and /")
    result = base64_decode(encoded)
    assert result == b"test data with + and /"
    
    # Test with padding
    result = base64_decode("YQ==")
    assert result == b"a"
    
    result = base64_decode("YWI=")
    assert result == b"ab"
    
    result = base64_decode("YWJj")
    assert result == b"abc"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with bytes input
    result = base64_decode(b"aGVsbG8=")
    assert result == b"hello"


# LLM-generated content at query #99
#--------------------------

```python
def test_base64_decode():
    # Test basic string input
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test URL-safe characters
    result = base64_decode("_-x")
    assert result == b"\xff\xeb"
    
    # Test with special characters
    result = base64_decode("YQ==")
    assert result == b"a"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters in input (should be ignored)
    result = base64_decode(b"SGVsbG8gV29ybGQ=\xff")
    assert result == b"Hello World"
```


# LLM-generated content at query #100
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with string input (not bytes)
    decoded = base64_decode("aGVsbG8gd29ybGQ")
    assert decoded == b"hello world"
    
    # Test with padding
    decoded = base64_decode("aGVsbG8=")
    assert decoded == b"hello"
    
    # Test empty string
    decoded = base64_decode("")
    assert decoded == b""
    
    # Test with non-ASCII characters in input (should be ignored)
    decoded = base64_decode("aGVsbG8\x80")
    assert decoded == b"hello"
    
    # Test that invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with bytes input
    decoded = base64_decode(b"dGVzdA==")
    assert decoded == b"test"
```


# LLM-generated content at query #101
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    result = base64_decode("dGVzdA==")
    assert result == b"test"
    
    # Test decoding without padding
    result = base64_decode("dGVzdA")
    assert result == b"test"
    
    # Test decoding bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding with URL-safe characters
    result = base64_decode("aGVsbG8tX3dvcmxk")  # hello_world
    assert result == b"hello_world"
    
    # Test decoding single character
    result = base64_decode("YQ==")  # 'a'
    assert result == b"a"
    
    # Test decoding with special characters
    result = base64_decode("dGVzdD8+")  # "test?>"
    assert result == b"test?>"
    
    # Test decoding invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test decoding non-base64 characters
    with pytest.raises(BadData):
        base64_decode("test\x00data")
```


# LLM-generated content at query #102
#--------------------------

```python
def test_base64_decode():
    # Test basic ASCII string
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test URL-safe characters
    result = base64_decode("dGVzdC11cmw")
    assert result == b"test-url"
    
    # Test binary data
    input_bytes = bytes(range(256))
    encoded = base64_encode(input_bytes)
    decoded = base64_decode(encoded)
    assert decoded == input_bytes
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with equals padding
    result = base64_decode("YQ==")
    assert result == b"a"
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8=")
    assert result == b"Hello"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode("abc$def")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test string with spaces (should be ignored due to ascii encoding)
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
```


# LLM-generated content at query #103
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 string
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test without padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test URL-safe characters
    assert base64_decode("PDw_Pz8-Pg==") == b"<<??>>"
    
    # Test single character
    assert base64_decode("YQ==") == b"a"
    
    # Test multiple padding scenarios
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test non-ASCII characters in input (should be ignored)
    assert base64_decode("aGVsbG8=\x80\x81") == b"hello"
    
    # Test with special characters that are part of base64 alphabet
    assert base64_decode("-__-") == b"\xfb\xbf\xbe"
```


# LLM-generated content at query #104
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with padding variations
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with URL-safe characters
    assert base64_decode("_-w=") == b"\xfe\xec"
    
    # Test that invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that non-base64 characters raise BadData
    try:
        base64_decode("abcde!@#$%")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #105
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"Hello, World!")
    assert base64_decode(encoded) == b"Hello, World!"
    
    # Test decoding with string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with padding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"  # without padding
    
    # Test decoding URL-safe characters
    encoded_url = base64_encode(b"test data with +/")
    assert base64_decode(encoded_url) == b"test data with +/"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters in input (should be ignored)
    assert base64_decode("SGVsbG8=\xff") == b"Hello"
    
    # Test with special characters that would normally be in base64
    encoded_special = base64_encode(b"\x00\x01\x02")
    assert base64_decode(encoded_special) == b"\x00\x01\x02"
```


# LLM-generated content at query #106
#--------------------------

```python
def test_base64_decode():
    # Test basic string input
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test bytes input directly
    encoded_bytes = base64_encode(b"test data")
    decoded_bytes = base64_decode(encoded_bytes)
    assert decoded_bytes == b"test data"

    # Test with empty string
    assert base64_decode(base64_encode(b"")) == b""

    # Test with special characters
    assert base64_decode(base64_encode(b"\x00\x01\x02")) == b"\x00\x01\x02"
    assert base64_decode(base64_encode(b"\xff\xfe\xfd")) == b"\xff\xfe\xfd"

    # Test with padding variations
    assert base64_decode("aGVsbG8") == b"hello"  # missing one padding char
    assert base64_decode("aGVsbG8=") == b"hello"  # correct padding
    assert base64_decode("aGVsbG8==") == b"hello"  # extra padding

    # Test with URL-safe characters
    assert base64_decode("_-") == b"\xfb\xff"  # - and _ in alphabet
    assert base64_decode("") == b""  # empty string
    
    # Test with invalid input
    import pytest
    from .exc import BadData
    
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\xff\xff\xff\xff")
    
    with pytest.raises(BadData):
        base64_decode("not base64")


# LLM-generated content at query #107
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    
    # Test URL-safe characters
    assert base64_decode("_-x4") == b"\xff\x1e"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with special characters
    assert base64_decode("AAECAwQFBgcICQoLDA0ODw==") == bytes(range(16))
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test invalid base64 data with wrong characters
    with pytest.raises(BadData):
        base64_decode("Hello World")  # Space is not valid base64
    
    # Test with unicode string (should ignore non-ASCII)
    assert base64_decode("SGVsbG8=" + "\u00e9") == b"Hello"  # é is ignored
```


# LLM-generated content at query #108
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test with string input
    decoded = base64_decode("aGVsbG8gd29ybGQ=")
    assert decoded == b"hello world"

    # Test with bytes input
    decoded = base64_decode(b"aGVsbG8gd29ybGQ=")
    assert decoded == b"hello world"

    # Test URL-safe characters
    original = b"test data with +/="
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test padding
    decoded = base64_decode("YQ")
    assert decoded == b"a"

    decoded = base64_decode("YWI")
    assert decoded == b"ab"

    decoded = base64_decode("YWJj")
    assert decoded == b"abc"

    # Test with empty string
    decoded = base64_decode("")
    assert decoded == b""

    # Test invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")

    # Test with non-ASCII characters (should be ignored)
    decoded = base64_decode("aGVsbG8gd29ybGQ=\x80\x81")
    assert decoded == b"hello world"

    # Test with padding added
    decoded = base64_decode("aGVsbG8")
    assert decoded == b"hello"

    # Test with extra padding
    decoded = base64_decode("aGVsbG8gd29ybGQ=")
    assert decoded == b"hello world"

    # Test roundtrip with various data
    test_data = [b"", b"a", b"ab", b"abc", b"test", b"\x00\x01\x02\xff", b"binary\x00data"]
    for data in test_data:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Failed for data: {data}"


# LLM-generated content at query #109
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test decoding bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with URL-safe characters
    assert base64_decode("dGVzdC11cmw") == b"test-url"
    
    # Test that invalid data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that non-base64 characters raise BadData
    try:
        base64_decode("hello world")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #110
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test decoding with text input
    encoded = base64_encode("world")
    assert base64_decode(encoded) == b"world"
    
    # Test empty string
    assert base64_decode(base64_encode("")) == b""
    
    # Test padding restoration
    encoded_no_padding = base64_encode(b"test")
    assert base64_decode(encoded_no_padding) == b"test"
    
    # Test with padding added
    encoded_with_padding = base64_encode(b"longer string")
    assert base64_decode(encoded_with_padding) == b"longer string"
    
    # Test invalid base64 data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
    
    # Test with various lengths
    for length in range(0, 20):
        data = b"x" * length
        assert base64_decode(base64_encode(data)) == data
    
    # Test with unicode text
    unicode_text = "héllo wörld"
    encoded_unicode = base64_encode(unicode_text)
    assert base64_decode(encoded_unicode) == unicode_text.encode("utf-8")
```


# LLM-generated content at query #111
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test special characters
    encoded = base64_encode(b"\x00\x01\x02\xff")
    decoded = base64_decode(encoded)
    assert decoded == b"\x00\x01\x02\xff"
    
    # Test string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test invalid base64 data
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test incorrectly padded data
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8===")
```


# LLM-generated content at query #112
#--------------------------

```python
def test_base64_decode():
    # Test basic byte string decode
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test string input (should be converted to bytes)
    original_str = "test data"
    encoded = base64_encode(original_str)
    assert base64_decode(encoded) == original_str.encode("utf-8")

    # Test with padding
    encoded_with_padding = b"dGVzdA=="
    assert base64_decode(encoded_with_padding) == b"test"

    # Test without padding
    encoded_without_padding = b"dGVzdA"
    assert base64_decode(encoded_without_padding) == b"test"

    # Test empty string
    assert base64_decode(b"") == b""

    # Test URL-safe characters (underscore and dash)
    original = b"\xfb\xff\xff\xff\xff"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
```


# LLM-generated content at query #113
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    result = base64_decode("dGVzdA==")
    assert result == b"test", f"Expected b'test', got {result}"
    
    # Test decoding without padding
    result = base64_decode("dGVzdA")
    assert result == b"test", f"Expected b'test', got {result}"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b"", f"Expected empty bytes, got {result}"
    
    # Test decoding bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test", f"Expected b'test', got {result}"
    
    # Test decoding with URL-safe characters
    result = base64_decode("_-x.")
    assert result == b"\xff\xed\xc7", f"Expected b'\\xff\\xed\\xc7', got {result}"
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!")
    
    # Test that non-ASCII characters are ignored
    result = base64_decode("dGVzdA==\x80")
    assert result == b"test", f"Expected b'test', got {result}"
```


# LLM-generated content at query #114
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    result = base64_decode("dGVzdA==")
    assert result == b"test"
    
    # Test decoding without padding
    result = base64_decode("dGVzdA")
    assert result == b"test"
    
    # Test decoding bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding single character
    result = base64_decode("ZA==")
    assert result == b"d"
    
    # Test decoding with URL-safe characters
    result = base64_decode("dGVzdA_-")  # - instead of +
    assert result == b"test"
    
    # Test that invalid data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that non-base64 characters raise BadData
    try:
        base64_decode("hello world")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding longer strings
    original = b"Hello, World! This is a longer test string."
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
```


# LLM-generated content at query #115
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with string input
    encoded = base64_encode("test data")
    decoded = base64_decode(encoded)
    assert decoded == b"test data"
    
    # Test with padding
    encoded = base64_encode(b"a")
    decoded = base64_decode(encoded)
    assert decoded == b"a"
    
    # Test with longer data
    test_data = b"x" * 100
    encoded = base64_encode(test_data)
    decoded = base64_decode(encoded)
    assert decoded == test_data
    
    # Test with special characters
    test_data = b"\x00\x01\x02\xff\xfe"
    encoded = base64_encode(test_data)
    decoded = base64_decode(encoded)
    assert decoded == test_data
    
    # Test with unicode string input
    encoded = base64_encode("héllo")
    decoded = base64_decode(encoded)
    assert decoded == "héllo".encode("utf-8")
    
    # Test with binary data containing null bytes
    test_data = b"\x00" * 10
    encoded = base64_encode(test_data)
    decoded = base64_decode(encoded)
    assert decoded == test_data
    
    # Test decode from string that's already a valid base64
    encoded_str = base64_encode(b"test").decode("ascii")
    decoded = base64_decode(encoded_str)
    assert decoded == b"test"
    
    # Test decode with extra padding (should be handled)
    encoded = base64_encode(b"test")
    encoded_with_padding = encoded + b"=" * (4 - len(encoded) % 4)
    decoded = base64_decode(encoded_with_padding)
    assert decoded == b"test"
    
    # Test decode with invalid base64 data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decode with non-base64 characters
    try:
        base64_decode(b"abc$%^")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test roundtrip with various data lengths
    for length in range(1, 20):
        test_data = bytes(range(length))
        encoded = base64_encode(test_data)
        decoded = base64_decode(encoded)
        assert decoded == test_data, f"Failed for length {length}"
```


# LLM-generated content at query #116
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test decoding with padding
    encoded_padded = base64.b64encode(b"test").decode("ascii")  # Non-urlsafe
    assert base64_decode(encoded_padded) == b"test"

    # Test decoding empty string
    assert base64_decode(b"") == b""

    # Test decoding single character
    assert base64_decode(base64_encode(b"a")) == b"a"

    # Test decoding with special characters
    assert base64_decode(base64_encode(b"\x00\x01\x02")) == b"\x00\x01\x02"

    # Test decoding string input
    assert base64_decode("aGVsbG8=") == b"hello"

    # Test decoding with invalid data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test decoding with invalid characters
    try:
        base64_decode(b"aGVsbG8$")  # $ is not valid base64
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #117
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"test data")
    decoded = base64_decode(encoded)
    assert decoded == b"test data"

    # Test with string input
    encoded = base64_encode("hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test empty string
    encoded = base64_encode(b"")
    decoded = base64_decode(encoded)
    assert decoded == b""

    # Test with padding
    encoded = base64_encode(b"a")
    decoded = base64_decode(encoded)
    assert decoded == b"a"

    # Test binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data

    # Test invalid input raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test with missing padding
    encoded = base64_encode(b"test").rstrip(b"=")
    decoded = base64_decode(encoded)
    assert decoded == b"test"
```


# LLM-generated content at query #118
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with missing padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with special characters
    original = b"\x00\x01\x02\xff\xfe"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with URL-safe characters
    original = b"\xfb\xff\xff\xff\xff"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
```


# LLM-generated content at query #119
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding required
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with URL-safe characters
    test_bytes = b"\xfb\xff\xff\xff\xff"
    encoded = base64_encode(test_bytes)
    assert base64_decode(encoded) == test_bytes
    
    # Test with unicode string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test incomplete base64 data
    try:
        base64_decode(b"aGVs")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test special characters in base64
    result = base64_decode(b"_-A=")
    assert result == b"\xfb\xff"
```


# LLM-generated content at query #120
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode("d29ybGQ") == b"world"
    
    # Test with padding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with various characters
    assert base64_decode("dXNlcm5hbWU") == b"username"
    assert base64_decode("cGFzc3dvcmQxMjM") == b"password123"
    
    # Test with special URL-safe characters
    assert base64_decode("Pz4_Pz4") == b">?>?>"  # URL-safe base64 decoding
    
    # Test bytes input
    assert base64_decode(b"Ynl0ZXM") == b"bytes"
    
    # Test str input returns bytes
    result = base64_decode("dGVzdA")
    assert isinstance(result, bytes)
    
    # Test bad data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode(b"\xff\xfe\xfd")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode("123")  # invalid length/padding
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in str input
    assert base64_decode("w6TDtsO8") == "äöü".encode("utf-8")
    
    # Test round trip: encode then decode
    original = b"test data with spaces and symbols!@#$%^&*()"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
```


# LLM-generated content at query #121
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decode without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test decode with URL-safe characters
    result = base64_decode("dGVzdC11cmwtc2FmZQ==")
    assert result == b"test-url-safe"
    
    # Test decode empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decode with bytes input
    result = base64_decode(b"dGVzdC1ieXRlcw==")
    assert result == b"test-bytes"
    
    # Test decode with special characters
    result = base64_decode("dGVzdC13aXRoICQjQCE=")
    assert result == b"test-with $#@!"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decode with only padding
    result = base64_decode("==")
    assert result == b"\x00"
```


# LLM-generated content at query #122
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    result = base64_decode(encoded)
    assert result == b"hello world"
    
    # Test decoding with padding
    encoded = base64_encode(b"test")
    result = base64_decode(encoded)
    assert result == b"test"
    
    # Test decoding empty string
    result = base64_decode(b"")
    assert result == b""
    
    # Test decoding with string input
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test decoding with special characters
    encoded = base64_encode(b"\x00\xff\xfe\xfd")
    result = base64_decode(encoded)
    assert result == b"\x00\xff\xfe\xfd"
    
    # Test decoding with invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test decoding with non-ASCII input (should be ignored)
    result = base64_decode("aGVsbG8=\xff")
    assert result == b"hello"
    
    # Test decoding with missing padding
    result = base64_decode(b"aGVsbG8")
    assert result == b"hello"
```


# LLM-generated content at query #123
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding of valid base64 string
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding various byte lengths
    assert base64_decode(base64_encode(b"a")) == b"a"
    assert base64_decode(base64_encode(b"ab")) == b"ab"
    assert base64_decode(base64_encode(b"abc")) == b"abc"
    assert base64_decode(base64_encode(b"abcd")) == b"abcd"
    
    # Test decoding with special characters
    special = b"\x00\xff\xfe\xfd"
    encoded_special = base64_encode(special)
    assert base64_decode(encoded_special) == special
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\x00\x00\x00")
```


# LLM-generated content at query #124
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    original = b"hello world"
    encoded = base64.b64encode(original).rstrip(b"=")
    assert base64_decode(encoded) == original
    
    # Test with padding
    padded = b"aGVsbG8="
    assert base64_decode(padded) == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test URL-safe characters
    url_encoded = b"dGVzdC11cmwtdmFsdWU"
    assert base64_decode(url_encoded) == b"test-url-value"
    
    # Test with special characters
    special = b"dGVzdC11cmwtdmFsdWU_"
    assert base64_decode(special) == b"test-url-value_"
    
    # Test string input (not bytes)
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test that BadData is raised for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that BadData is raised for non-base64 characters
    try:
        base64_decode(b"this is not base64")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #125
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 string decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe base64 (without padding)
    assert base64_decode(b"SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test various padding scenarios
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>?>"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode(b"\xff\xfe\xfd")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test non-ASCII input (should be ignored due to errors="ignore")
    assert base64_decode("SGVsbG8=\xff") == b"Hello"
```


# LLM-generated content at query #126
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with bytes input
    result = base64_decode(b"VGVzdA==")
    assert result == b"Test"
    
    # Test with URL-safe characters
    result = base64_decode("a-_w")
    assert result == b"k\xef\xc0"
    
    # Test with ASCII encoding
    result = base64_decode("d29yaw==")
    assert result == b"work"
    
    # Test with unicode characters in string
    result = base64_decode("w6TDtsOc")  # "äöü" in base64
    assert result == "äöü".encode()
    
    # Test invalid base64 data raises BadData exception
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-base64 characters
    with pytest.raises(BadData):
        base64_decode("ab cd")
```


# LLM-generated content at query #127
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test padding variations
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with special characters
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test with non-ASCII characters in string (should be ignored)
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test roundtrip with various lengths
    for i in range(1, 20):
        test_bytes = bytes(range(i))
        encoded = base64_encode(test_bytes)
        assert base64_decode(encoded) == test_bytes
```


# LLM-generated content at query #128
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with special URL-safe characters
    assert base64_decode(b"dGVzdD4_") == b"test>?"
    assert base64_decode("dGVzdD4_") == b"test>?"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test invalid base64 data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\xff\xff\xff")
    
    # Test with very long string (edge case)
    long_string = b"a" * 1000
    encoded = base64_encode(long_string)
    assert base64_decode(encoded) == long_string
```


# LLM-generated content at query #129
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test invalid data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special URL-safe characters
    original = b"\x00\xff\xfe\xfd"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
```


# LLM-generated content at query #130
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test decoding with URL-safe characters
    assert base64_decode("dGVzdF91cmw") == b"test_url"
    assert base64_decode("dGVzdC11cmw") == b"test-url"
    
    # Test decoding bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding single character
    assert base64_decode("ZA==") == b"d"
    
    # Test decoding with special characters
    assert base64_decode("Pj4+Pg==") == b">>>>"
    
    # Test decoding raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("123")
```


# LLM-generated content at query #131
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test string")
    decoded = base64_decode(encoded)
    assert decoded == b"test string"
    
    # Test with padding
    encoded = base64_encode(b"a")
    decoded = base64_decode(encoded)
    assert decoded == b"a"
    
    # Test with longer data
    original = b"x" * 100
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with special characters
    original = b"\x00\x01\x02\xff"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test BadData exception for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test BadData exception for incomplete padding
    try:
        base64_decode(b"YWJj")  # missing padding
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #132
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test single character
    assert base64_decode("aA==") == b"h"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with special URL-safe characters
    assert base64_decode("_-w=") == b"\xff\xec"
    
    # Test invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode("SGVsbG8$")
        assert False, "Expected BadData exception"
    except BadData:
        pass
```


# LLM-generated content at query #133
#--------------------------

```python
def test_base64_decode():
    # Test with valid URL-safe base64 encoded string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test with bytes input
    encoded_bytes = b"aGVsbG8gd29ybGQ"
    decoded = base64_decode(encoded_bytes)
    assert decoded == b"hello world"

    # Test with string input
    encoded_str = "aGVsbG8gd29ybGQ"
    decoded = base64_decode(encoded_str)
    assert decoded == b"hello world"

    # Test with padding
    encoded_with_padding = b"aGVsbG8="
    decoded = base64_decode(encoded_with_padding)
    assert decoded == b"hello"

    # Test with empty string
    assert base64_decode("") == b""

    # Test with special URL-safe characters
    original = b"\xfb\xff\xff\xff\xff"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test that invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
```


# LLM-generated content at query #134
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with text input
    encoded = base64_encode("test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with padding
    result = base64_decode("aGVsbG8")
    assert result == b"hello"
    
    # Test with full padding
    result = base64_decode("aA==")
    assert result == b"h"
    
    # Test invalid characters raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid length without proper padding
    try:
        base64_decode("abc")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test with URL-safe characters
    result = base64_decode("Pj4-Pg")
    assert result == b">>>"
```


# LLM-generated content at query #135
#--------------------------

```python
def test_base64_decode():
    # Test with standard string
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test with URL-safe encoding (no padding)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test with bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with special characters
    assert base64_decode("YT4+Yg==") == b"a>b"
    
    # Test with multiple padding scenarios
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with binary data
    binary_data = b"\x00\x01\x02\x03"
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test that invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode("abc!def")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #136
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 string
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("dGVzdA") == b"test"  # Without padding
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special characters
    assert base64_decode("_-x") == b"\xfb\xe7"
    
    # Test with URL-safe characters
    result = base64_decode("dGVzdA")
    assert result == b"test"
    
    # Test bad data raises BadData exception
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test invalid base64 characters
    with pytest.raises(BadData):
        base64_decode("test\x00")
```


# LLM-generated content at query #137
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test URL-safe characters
    assert base64_decode("_-_w") == b"\xff\xef\xc0"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
    
    # Test with single character
    assert base64_decode("WA==") == b"X"
    assert base64_decode("WA") == b"X"
    
    # Test with special characters
    assert base64_decode("PDw/Pz4+") == b"<<??>>"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x00")
    
    # Test with very long data
    long_data = b"a" * 1000
    encoded = base64_encode(long_data)
    assert base64_decode(encoded) == long_data
    
    # Test with padding variations
    assert base64_decode("YQ==") == b"a"  # 1 char, 2 padding
    assert base64_decode("YWI=") == b"ab"  # 2 chars, 1 padding
    assert base64_decode("YWJj") == b"abc"  # 3 chars, no padding
    assert base64_decode("YWJjZA==") == b"abcd"  # 4 chars, 2 padding```


# LLM-generated content at query #138
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with empty string
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test with special characters
    encoded_special = base64_encode(b"\x00\x01\x02\xff")
    decoded_special = base64_decode(encoded_special)
    assert decoded_special == b"\x00\x01\x02\xff"
    
    # Test with unicode string input
    encoded_unicode = base64_encode("hello")
    decoded_unicode = base64_decode(encoded_unicode)
    assert decoded_unicode == b"hello"
    
    # Test with padding variations
    assert base64_decode("aGVsbG8") == b"hello"
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with different lengths
    for i in range(1, 20):
        data = bytes(range(i))
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data


# LLM-generated content at query #139
#--------------------------

```python
def test_base64_decode():
    # Test with valid base64-encoded string (no padding)
    result = base64_decode("dGVzdA")
    assert result == b"test"

    # Test with valid base64-encoded bytes (with padding)
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"

    # Test with empty string
    result = base64_decode("")
    assert result == b""

    # Test with long data
    original = b"Hello, World! This is a test with more data."
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test with binary data
    original = bytes(range(256))
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test with invalid base64 string
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with ASCII string input
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"

    # Test with bytes input
    result = base64_decode(b"V29ybGQ=")
    assert result == b"World"
```


# LLM-generated content at query #140
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding with invalid characters should raise BadData
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test decoding with non-base64 characters
    with pytest.raises(BadData):
        base64_decode(b"hello world")
    
    # Test decoding various byte sequences
    test_cases = [
        b"",
        b"\x00",
        b"\xff",
        b"test",
        b"1234567890",
        b"!@#$%^&*()",
    ]
    for case in test_cases:
        encoded = base64_encode(case)
        assert base64_decode(encoded) == case
    
    # Test decoding with ASCII encoding
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
```


# LLM-generated content at query #141
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test without padding
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test with special characters
    assert base64_decode("aGVsbG8_LT4_") == b"hello?->?"
    
    # Test with padding variations
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YQ") == b"a"
    
    # Test invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test non-base64 characters
    try:
        base64_decode("not valid base64!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with newlines (should be ignored)
    assert base64_decode("SGVs\nbG8g\nV29y\nbGQ=") == b"Hello World"
```


# LLM-generated content at query #142
#--------------------------

```python
def test_base64_decode():
    # Test valid base64 string (bytes input)
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test valid base64 string (str input)
    assert base64_decode("dGVzdA") == b"test"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with padding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test with different data
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters in base64
    assert base64_decode(b"_-w") == b"\xfb\xc3"
    assert base64_decode("_-w") == b"\xfb\xc3"


# LLM-generated content at query #143
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test decoding with padding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"  # Without padding
    
    # Test decoding with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test special characters
    encoded_special = base64_encode(b"hello world!\n\t")
    decoded_special = base64_decode(encoded_special)
    assert decoded_special == b"hello world!\n\t"
    
    # Test binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    decoded_binary = base64_decode(encoded_binary)
    assert decoded_binary == binary_data
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with invalid characters
    try:
        base64_decode("abc$%^")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with unicode string input
    assert base64_decode("w7zDpMO8") == "üäß".encode("utf-8")


# LLM-generated content at query #144
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test decoding URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    assert base64_decode("PDw8PA==") == b"<<<<"
    
    # Test decoding with dashes and underscores
    assert base64_decode("-__-") == b"\xfb\xff\xfe"
    
    # Test decoding single character
    assert base64_decode("WA==") == b"X"
    
    # Test decoding special characters
    assert base64_decode("//8=") == b"\xff\xff"
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test that non-base64 characters raise BadData
    with pytest.raises(BadData):
        base64_decode("Hello World")
    
    # Test that incorrect padding raises BadData
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")
```


# LLM-generated content at query #145
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test with full padding
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with special characters
    data = b"\x00\x01\x02\xff\xfe"
    encoded = base64_encode(data)
    decoded = base64_decode(encoded)
    assert decoded == data
    
    # Test with string input
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
```


# LLM-generated content at query #146
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding URL-safe characters
    assert base64_decode("_-x") == b"\xfb\xdf"
    
    # Test decoding bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test decoding with different characters
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    
    # Test decoding special characters
    assert base64_decode("AAECAwQFBgcI") == b"\x00\x01\x02\x03\x04\x05\x06\x07\x08"
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!")
    
    with pytest.raises(BadData):
        base64_decode("ABC\x00DEF")
```


# LLM-generated content at query #147
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with string input
    encoded_str = "aGVsbG8gd29ybGQ"
    decoded = base64_decode(encoded_str)
    assert decoded == b"hello world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test padding variations
    test_data = [b"a", b"ab", b"abc", b"abcd"]
    for data in test_data:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data
    
    # Test with special characters in URL-safe base64
    original = b"\xff\xfe\x00\x01"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test that invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("not valid base64!!!")
    
    # Test with bytes that have incorrect length after padding
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8=")  # This is valid, just testing different padding
```


# LLM-generated content at query #148
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test without padding
    assert base64_decode(b"dGVzdA") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with special URL-safe characters
    assert base64_decode(b"Pj4-Pg==") == b">>>"
    assert base64_decode(b"PDw8PA==") == b"<<<<"
    
    # Test with binary data
    assert base64_decode(b"AAECAwQFBgcI") == b"\x00\x01\x02\x03\x04\x05\x06\x07\x08"
    
    # Test invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode("not valid base64")
    
    # Test with non-ASCII bytes (should be ignored due to errors="ignore")
    assert base64_decode(b"\xffdGVzdA==") == b"test"
```


# LLM-generated content at query #149
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test with URL-safe characters
    assert base64_decode("aGVsbG8_d29ybGQ=") == b"hello?world"
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with padding characters
    assert base64_decode("YQ==") == b"a"
    
    # Test with two padding characters
    assert base64_decode("YWI=") == b"ab"
    
    # Test with underscore (URL-safe)
    assert base64_decode("aGVsbG8_d29ybGQ") == b"hello?world"
    
    # Test with dash (URL-safe)
    assert base64_decode("aGVsbG8td29ybGQ=") == b"hello-world"
    
    # Test invalid data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode("SGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input (should be ignored or handled)
    assert base64_decode(b"dGVzdA==") == b"test"
```


# LLM-generated content at query #150
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters
    assert base64_decode(b"aGVsbG8t") == b"hello-"
    assert base64_decode(b"aGVsbG9f") == b"hello_"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with various byte values
    encoded = base64_encode(b"\x00\x01\x02")
    assert base64_decode(encoded) == b"\x00\x01\x02"
    
    # Test with special characters
    encoded = base64_encode(b"\xff\xfe\xfd")
    assert base64_decode(encoded) == b"\xff\xfe\xfd"
    
    # Test invalid input raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test invalid base64 characters
    with pytest.raises(BadData):
        base64_decode(b"SGVs@G8=")
```


# LLM-generated content at query #151
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello")
    result = base64_decode(encoded)
    assert result == b"hello"
    
    # Test with string input
    encoded = base64_encode("world")
    result = base64_decode(encoded)
    assert result == b"world"
    
    # Test with padding
    encoded = base64_encode(b"a")
    result = base64_decode(encoded)
    assert result == b"a"
    
    # Test with multiple padding characters
    encoded = base64_encode(b"ab")
    result = base64_decode(encoded)
    assert result == b"ab"
    
    # Test empty input
    result = base64_decode(b"")
    assert result == b""
    
    # Test with bytes input
    result = base64_decode(b"aGVsbG8=")
    assert result == b"hello"
    
    # Test invalid base64 data
    try:
        base64_decode(b"!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters in base64
    encoded = base64_encode(b"test123!@#")
    result = base64_decode(encoded)
    assert result == b"test123!@#"
    
    # Test with unicode characters
    encoded = base64_encode("héllo")
    result = base64_decode(encoded)
    assert result == "héllo".encode("utf-8")
```


# LLM-generated content at query #152
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test without padding
    assert base64_decode(b"dGVzdA") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test URL-safe characters
    assert base64_decode(b"_-w") == b"\xff\xeb"
    assert base64_decode("_-w") == b"\xff\xeb"
    
    # Test with special characters
    encoded = base64_encode(b"\x00\x01\x02")
    assert base64_decode(encoded) == b"\x00\x01\x02"
    
    # Test invalid base64 data
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII bytes (should be ignored)
    assert base64_decode(b"dGVzdA\xff") == b"test"
    
    # Test long strings
    long_data = b"x" * 1000
    encoded = base64_encode(long_data)
    assert base64_decode(encoded) == long_data
```


# LLM-generated content at query #153
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test bytes input
    assert base64_decode(b"VGVzdA==") == b"Test"
    
    # Test URL-safe characters
    assert base64_decode("Pj4_Pz8") == b">>???"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special characters
    assert base64_decode("dGVzdCB3aXRoIHNwYWNl") == b"test with space"
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test invalid character raises BadData
    with pytest.raises(BadData):
        base64_decode("VGhpcyBpcyBpbnZhbGlk")  # Valid, but test another invalid
        
    # Test short string with proper padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YQ") == b"a"
    
    # Test multiple padding scenarios
    assert base64_decode("YWJj") == b"abc"
    assert base64_decode("YWJjZA==") == b"abcd"
    assert base64_decode("YWJjZGU=") == b"abcde"
```


# LLM-generated content at query #154
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    assert base64_decode(base64_encode("test")) == b"test"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test with empty string
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test with special characters
    assert base64_decode(base64_encode(b"\x00\x01\x02")) == b"\x00\x01\x02"
    
    # Test with binary data
    binary_data = bytes(range(256))
    assert base64_decode(base64_encode(binary_data)) == binary_data
    
    # Test with URL-safe characters
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\x00\x01\x02")
```


# LLM-generated content at query #155
#--------------------------

```python
def test_base64_decode():
    # Test with valid base64 encoded string
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special characters
    assert base64_decode("aGVsbG8t") == b"hello-"
    assert base64_decode("aGVsbG9f") == b"hello_"
    
    # Test with invalid base64 data (should raise BadData)
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input
    assert base64_decode("aGVsbG8=".encode("ascii")) == b"hello"
```


# LLM-generated content at query #156
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding with special characters
    encoded_url = base64_encode(b"https://example.com/path?query=value")
    assert base64_decode(encoded_url) == b"https://example.com/path?query=value"
    
    # Test decoding binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
    
    # Test that invalid input raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that non-base64 characters raise BadData
    try:
        base64_decode(b"aGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with null bytes
    assert base64_decode(b"AA==") == b"\x00"
    assert base64_decode(b"AAA=") == b"\x00\x00"
    assert base64_decode(b"AAAA") == b"\x00\x00\x00"


# LLM-generated content at query #157
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test with padding
    encoded_padded = base64.urlsafe_b64encode(b"test").decode()
    assert base64_decode(encoded_padded) == b"test"

    # Test with no padding
    encoded_no_pad = base64.urlsafe_b64encode(b"test").rstrip(b"=").decode()
    assert base64_decode(encoded_no_pad) == b"test"

    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"

    # Test empty string
    assert base64_decode("") == b""

    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"

    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("aGVsbG8=\x80") == b"hello"


# LLM-generated content at query #158
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test without padding
    assert base64_decode("dGVzdA") == b"test"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test URL-safe encoding
    assert base64_decode("aGVsbG8td29ybGQ") == b"hello-world"
    
    # Test with underscores
    assert base64_decode("aGVsbG8_d29ybGQ") == b"hello?world"
    
    # Test invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test non-base64 characters raises BadData
    with pytest.raises(BadData):
        base64_decode("not valid base64")
```


# LLM-generated content at query #159
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test decoding bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with special URL-safe characters
    assert base64_decode("_-x") == b"\xff\xeb"
    
    # Test decoding longer text
    original = b"This is a test message!"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test that invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test that non-base64 characters raise BadData
    with pytest.raises(BadData):
        base64_decode("Not base64 data!")
    
    # Test decoding integer bytes
    assert base64_decode("AAEBAg==") == b"\x00\x01\x01\x02"
```


# LLM-generated content at query #160
#--------------------------

```python
def test_base64_decode():
    # Test with valid base64 encoded string
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test with valid base64 encoded bytes
    result = base64_decode(b"SGVsbG8=")
    assert result == b"Hello"
    
    # Test with URL-safe base64 (no padding)
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test with empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with single character
    result = base64_decode("WA==")
    assert result == b"X"
    
    # Test with binary data
    result = base64_decode("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=")
    assert result == bytes(range(32))
    
    # Test with invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test with non-ASCII input (should be ignored)
    result = base64_decode("SGVsbG8=\x80")
    assert result == b"Hello"
    
    # Test with bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
```


# LLM-generated content at query #161
#--------------------------

```python
def test_base64_decode():
    # Test valid base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with URL-safe characters
    assert base64_decode("Pz4_Pz4_Pz4_Pz4") == b">?>?>?>?>"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-base64 characters
    with pytest.raises(BadData):
        base64_decode("aGVsbG8$")
```


# LLM-generated content at query #162
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test with URL-safe characters
    assert base64_decode(b"Pj4_Pz8") == b">>???"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test single character
    assert base64_decode("YQ==") == b"a"
    
    # Test without padding
    assert base64_decode("dGVzdA") == b"test"
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test with extra padding
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test with various lengths
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    assert base64_decode("YWJjZA") == b"abcd"
    assert base64_decode("YWJjZGU") == b"abcde"
    
    # Test with non-ASCII characters in input (should be ignored)
    assert base64_decode("dGVzdA==\x80") == b"test"
    
    # Test that BadData is raised for invalid input
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with invalid characters
    try:
        base64_decode("invalid!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with whitespace
    try:
        base64_decode("dGV zdA==")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #163
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test decoding with different lengths
    assert base64_decode(base64_encode(b"a")) == b"a"
    assert base64_decode(base64_encode(b"ab")) == b"ab"
    assert base64_decode(base64_encode(b"abc")) == b"abc"
    
    # Test decoding empty string
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test decoding with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding handles padding correctly
    encoded_no_padding = base64_encode(b"test")
    assert b"=" not in encoded_no_padding
    assert base64_decode(encoded_no_padding) == b"test"
    
    # Test that invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("not base64 at all")
    
    # Test with special characters that are URL-safe
    binary_data = bytes(range(256))
    assert base64_decode(base64_encode(binary_data)) == binary_data
    
    # Test with non-ASCII input (should be ignored)
    assert base64_decode("aGVsbG8=\x80\x81") == b"hello"
```


# LLM-generated content at query #164
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding of valid base64 URL-safe string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with string input
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test that invalid base64 raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that non-base64 characters raise BadData
    try:
        base64_decode(b"hello world")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding various byte values
    for i in range(256):
        data = bytes([i])
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data
```


# LLM-generated content at query #165
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"Hello World!")
    assert base64_decode(encoded) == b"Hello World!"
    
    # Test with string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test with padding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("YQ==") == b"a"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special characters in URL-safe base64
    encoded = base64_encode(b"test data with \x00\xff")
    assert base64_decode(encoded) == b"test data with \x00\xff"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test raises BadData for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    try:
        base64_decode(b"\xff\xff\xff")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test with various valid base64 strings
    test_cases = [
        (b"test", b"dGVzdA=="),
        (b"a", b"YQ=="),
        (b"ab", b"YWI="),
        (b"abc", b"YWJj"),
        (b"abcd", b"YWJjZA=="),
    ]
    for original, expected_encoded in test_cases:
        assert base64_decode(expected_encoded) == original
```


# LLM-generated content at query #166
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test decoding bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with special URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    assert base64_decode("PDw8PA==") == b"<<<<"
    
    # Test decoding binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test that BadData is raised for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")
    
    # Test with different types of invalid padding
    with pytest.raises(BadData):
        base64_decode("=")
    
    with pytest.raises(BadData):
        base64_decode("==")
```


# LLM-generated content at query #167
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"V29ybGQ=")
    assert result == b"World"
    
    # Test URL-safe characters
    result = base64_decode("dGVzdC11cmwtdmFsdWU")
    assert result == b"test-url-value"
    
    # Test with special characters
    result = base64_decode("dGVzdC11cmwtdmFsdWU")
    assert result == b"test-url-value"
    
    # Test that invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters
    result = base64_decode("w6TDtsO8")
    assert result == "äöü".encode("utf-8")


# LLM-generated content at query #168
#--------------------------

```python
def test_base64_decode():
    # Test decoding normal base64 string
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test decoding URL-safe characters
    assert base64_decode(b"Pj4-Pg==") == b">>>?"
    assert base64_decode(b"Pj4-Pg") == b">>>?"
    
    # Test decoding with different lengths
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    
    # Test decoding binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test that BadData is raised for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that BadData is raised for non-base64 characters
    try:
        base64_decode(b"Hello World!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding with bytes input
    assert isinstance(base64_decode(b"SGVsbG8="), bytes)
    
    # Test decoding with string input
    assert isinstance(base64_decode("SGVsbG8="), bytes)
```


# LLM-generated content at query #169
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"V29ybGQ=")
    assert result == b"World"
    
    # Test URL-safe characters
    result = base64_decode("Pz4_")
    assert result == b"?>"
    
    # Test with special characters
    result = base64_decode("PDw_Pz4-")
    assert result == b"<<??>"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid base64 length
    try:
        base64_decode("abc")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #170
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with string input
    encoded_str = base64_encode("test data")
    assert base64_decode(encoded_str) == b"test data"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding single character
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YQ") == b"a"
    
    # Test decoding special characters
    assert base64_decode(b"dGVzdC91cmw=") == b"test/url"
    assert base64_decode(b"dGVzdC91cmw") == b"test/url"
    
    # Test decoding with minus and underscore (URL-safe)
    assert base64_decode(b"dGVzdC11cmw=") == b"test-url"
    assert base64_decode(b"dGVzdC11cmw") == b"test-url"
    
    # Test decoding with string input and ASCII encoding
    result = base64_decode("aGVsbG8gd29ybGQ=")
    assert result == b"hello world"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"aGVs\x00bG8=")
    
    with pytest.raises(BadData):
        base64_decode("not valid base64!")
    
    # Test roundtrip with various inputs
    test_cases = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"test data with spaces",
        b"binary\x00data",
        b"unicode_test_ñ",
        b"1234567890",
        b"special_chars!@#$%^&*()",
    ]
    
    for original in test_cases:
        encoded = base64_encode(original)
        decoded = base64_decode(encoded)
        assert decoded == original, f"Roundtrip failed for {original!r}"
    
    # Test that decode raises BadData for incorrectly padded input
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8")  # Missing padding but valid base64
    
    # Test that decode handles ASCII ignore errors
    result = base64_decode(b"aGVsbG8gd29ybGQ=\xff")
    assert result == b"hello world"  # Non-ASCII chars are ignored
    
    # Test decoding unicode string with non-ASCII characters
    result = base64_decode("aGVsbG8gd29ybGQ=")
    assert result == b"hello world"


# LLM-generated content at query #171
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test decoding with special URL-safe characters
    assert base64_decode("_-w") == b"\xff\xef"
    
    # Test decoding with numbers and special characters
    assert base64_decode("MTIzNDU2Nzg5MA==") == b"1234567890"
    
    # Test BadData exception for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test BadData exception for completely wrong input
    try:
        base64_decode("not base64 at all")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored due to errors='ignore')
    result = base64_decode("SGVsbG8=\xc3\x28")
    assert result == b"Hello("
    
    # Test binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
```


# LLM-generated content at query #172
#--------------------------

```python
def test_base64_decode():
    # Test with valid URL-safe base64 encoded string
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test with bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test with padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test with empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with special URL-safe characters
    result = base64_decode("_-x")
    assert result == b"\xfb\xe7"  # Expected bytes for _-x decoded
    
    # Test with invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input (should be ignored)
    result = base64_decode("SGVsbG8gV29ybGQ\x80\x81")
    assert result == b"Hello World"


# LLM-generated content at query #173
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test URL-safe characters (using - and _)
    assert base64_decode("Pj4-Pz8_") == b">>>???"
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\xff") == b"Hello"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test invalid base64 data as string
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test very long strings
    long_string = b"A" * 1000
    encoded = base64_encode(long_string)
    assert base64_decode(encoded) == long_string
    
    # Test binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
```


# LLM-generated content at query #174
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    
    # Test with various inputs
    assert base64_decode("") == b""
    assert base64_decode("AA==") == b"\x00"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
    
    # Test with non-ASCII bytes in input (should ignore errors)
    result = base64_decode(b"SGVsbG8=\xff")
    assert result == b"Hello"
```


# LLM-generated content at query #175
#--------------------------

```python
def test_base64_decode():
    # Test basic string decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test with empty bytes
    encoded_empty = base64_encode(b"")
    decoded_empty = base64_decode(encoded_empty)
    assert decoded_empty == b""

    # Test with special characters
    encoded_special = base64_encode(b"\x00\x01\x02\xff")
    decoded_special = base64_decode(encoded_special)
    assert decoded_special == b"\x00\x01\x02\xff"

    # Test with padding
    encoded_padding = base64_encode(b"test")
    decoded_padding = base64_decode(encoded_padding)
    assert decoded_padding == b"test"

    # Test with string input (should handle str as well as bytes)
    encoded_str = base64_encode("hello")
    decoded_str = base64_decode(encoded_str)
    assert decoded_str == b"hello"

    # Test with ASCII encoding errors (should ignore them)
    invalid_utf8 = b"hello\x80world"
    encoded_invalid = base64_encode(invalid_utf8)
    decoded_invalid = base64_decode(encoded_invalid)
    assert decoded_invalid == invalid_utf8

    # Test BadData exception for invalid base64
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
```


# LLM-generated content at query #176
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding bytes input
    encoded = base64_encode(b"test data 123")
    assert base64_decode(encoded) == b"test data 123"
    
    # Test decoding string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with empty data
    assert base64_decode("") == b""
    
    # Test with padding variations
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with URL-safe characters
    assert base64_decode("dGVzdC1f") == b"test-_"
    
    # Test with invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with invalid characters that get ignored
    result = base64_decode("a\x00GVsbG8=")
    assert result == b"hello"
    
    # Test with bytes input containing invalid characters
    result = base64_decode(b"a\xffGVsbG8=")
    assert result == b"hello"


# LLM-generated content at query #177
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    
    # Test with bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test URL-safe characters
    assert base64_decode("Pz4_") == b"?>?"  # URL-safe base64 uses - and _
    
    # Test various lengths
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test non-ASCII input (should be ignored)
    assert base64_decode(b"\xffSGVsbG8=") == b"Hello"
```


# LLM-generated content at query #178
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test decoding without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    assert base64_decode(b"V29ybGQ") == b"World"
    
    # Test decoding with string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test special characters
    assert base64_decode(b"") != None
    
    # Test invalid base64 data should raise BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode(b"SGVsbG8=" * 100)  # valid but long
        assert True
    except BadData:
        assert False, "Valid long data should not raise"
    
    # Test with special URL-safe characters
    encoded = base64.b64encode(b"test data").decode()
    url_safe_encoded = encoded.replace('+', '-').replace('/', '_')
    assert base64_decode(url_safe_encoded.encode()) == b"test data"


# LLM-generated content at query #179
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test without padding
    assert base64_decode("dGVzdA") == b"test"
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test with special URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    
    # Test with underscore and dash (URL-safe variant)
    assert base64_decode("PDw8PA==") == b"<<<<"
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("dGVzdA==" + chr(200)) == b"test"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with incorrect padding
    try:
        base64_decode("dGVzdA===")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with very long string
    long_data = b"x" * 1000
    encoded = base64_encode(long_data)
    assert base64_decode(encoded) == long_data
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with None-like edge cases
    assert base64_decode("AA==") == b"\x00"
    assert base64_decode("/w==") == b"\xff"
```


# LLM-generated content at query #180
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test URL-safe characters
    assert base64_decode("aGVsbG8vd29ybGQ") == b"hello/world"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with various lengths
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with special characters
    assert base64_decode("+/8=") == b"\xfb\xff"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test invalid bytes
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")
```


# LLM-generated content at query #181
#--------------------------

```python
def test_base64_decode():
    # Test normal string input
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test bytes input
    result = base64_decode(b"d29ybGQ=")
    assert result == b"world"
    
    # Test without padding
    result = base64_decode("aGVsbG8")
    assert result == b"hello"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test URL-safe characters
    result = base64_decode("dGVzdF91cmw")
    assert result == b"test_url"
    
    # Test with special characters
    result = base64_decode("dGVzdC12YWx1ZQ==")
    assert result == b"test-value"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("invalid!@#")
    
    # Test with Unicode string
    result = base64_decode("w7xiw7xsZ8Ok")
    assert result == "übülsgä".encode("utf-8")
```


# LLM-generated content at query #182
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    assert base64_decode("PDw8PA==") == b"<<<<"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=" + chr(200)) == b"Hello"
```


# LLM-generated content at query #183
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test single character
    assert base64_decode(base64_encode(b"a")) == b"a"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test with different data types
    assert base64_decode(base64_encode(b"\x00\x01\x02")) == b"\x00\x01\x02"
    assert base64_decode(base64_encode(b"\xff\xfe\xfd")) == b"\xff\xfe\xfd"
    
    # Test with numeric data
    assert base64_decode(base64_encode(b"12345")) == b"12345"
    
    # Test with special characters
    assert base64_decode(base64_encode(b"!@#$%^&*()")) == b"!@#$%^&*()"
    
    # Test with unicode/bytes encoding
    assert base64_decode(base64_encode("hello")) == b"hello"
    
    # Test with ascii string input (not bytes)
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with invalid characters
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8!!!")
    
    # Test with unexpected length data
    assert base64_decode(b"YQ==") == b"a"  # single byte
    assert base64_decode(b"YWI=") == b"ab"  # two bytes
    assert base64_decode(b"YWJj") == b"abc"  # three bytes
```


# LLM-generated content at query #184
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("dGVzdA") == b"test"  # Without padding
    
    # Test URL-safe characters
    assert base64_decode("aGVsbG8t") == b"hello-"
    assert base64_decode("aGVsbG9f") == b"hello_"
    
    # Test bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
    
    # Test with special characters
    assert base64_decode("") == b""
    
    # Test invalid input raises BadData
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode(b"\xff\xff")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input (should be ignored)
    assert base64_decode("dGVzdA==") == b"test"
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test with special characters
    encoded = base64_encode(b"data with \x00 null")
    assert base64_decode(encoded) == b"data with \x00 null"
    
    # Test with empty input
    encoded = base64_encode(b"")
    assert base64_decode(encoded) == b""
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8==") == b"hello"
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"aGVs\x00bG8=")
    
    # Test with ASCII ignore for non-ASCII bytes
    result = base64_decode("dGVzdA==".encode("utf-16"))
    assert result == b"test"  # Non-ASCII bytes are ignored
    
    # Test round trip with various data lengths
    for length in [1, 3, 5, 7, 9, 16, 32]:
        data = b"x" * length
        encoded = base64_encode(data)
        assert base64_decode(encoded) == data
```


# LLM-generated content at query #2
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 encoded string
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test decoding valid base64 encoded bytes
    result = base64_decode(b"d29ybGQ=")
    assert result == b"world"
    
    # Test decoding without padding
    result = base64_decode("aGVsbG8")
    assert result == b"hello"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding single character
    result = base64_decode("WA==")
    assert result == b"X"
    
    # Test decoding with URL-safe characters
    result = base64_decode("Pz4_Pz4_Pz4-")
    assert result == b"?>?>?>?>"
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test that non-base64 characters raise BadData
    with pytest.raises(BadData):
        base64_decode("hello world")
    
    # Test that wrong length raises BadData
    with pytest.raises(BadData):
        base64_decode("a")
    
    # Test decoding bytes input with non-ASCII characters
    result = base64_decode(b"w6TDtsO8")
    assert result == "äöü".encode("utf-8")
```


# LLM-generated content at query #3
#--------------------------

```python
def test_base64_decode():
    # Test basic string decoding
    original = b"hello world"
    encoded = base64_encode(original)
    result = base64_decode(encoded)
    assert result == original

    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"") == b""

    # Test with URL-safe characters
    encoded_url = b"aGVsbG8gd29ybGQ"
    assert base64_decode(encoded_url) == b"hello world"

    # Test with special characters in URL-safe base64
    assert base64_decode(b"dGVzdC0t") == b"test--"
    assert base64_decode(b"dGVzdF8=") == b"test_"

    # Test string input (not bytes)
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("dGVzdA==") == b"test"

    # Test with unicode/ASCII string
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test BadData exception for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test with non-base64 characters
    try:
        base64_decode(b"hello world!@#$%")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test empty bytes
    assert base64_decode(b"") == b""

    # Test with various lengths
    assert base64_decode(b"YQ") == b"a"
    assert base64_decode(b"YWI") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    assert base64_decode(b"YWJjZA") == b"abcd"
    assert base64_decode(b"YWJjZGU") == b"abcde"
    assert base64_decode(b"YWJjZGVm") == b"abcdef"

    # Test roundtrip with various data
    test_data = [b"", b"a", b"ab", b"abc", b"test", b"hello world", b"12345"]
    for data in test_data:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Failed roundtrip for {data}"```


# LLM-generated content at query #4
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with different padding lengths
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"dGVzdA") == b"test"  # Without padding
    assert base64_decode(b"dGVzdA") == b"test"  # Without padding
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with special characters (URL-safe)
    encoded_url = base64_encode(b"hello+world")
    assert base64_decode(encoded_url) == b"hello+world"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
    
    # Test that invalid data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that padding errors are handled
    try:
        base64_decode(b"aGVsbG8")  # Missing padding
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #5
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test without padding
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test with URL-safe characters
    assert base64_decode(b"aGVsbG8_d29ybGQ=") == b"hello?world"
    assert base64_decode(b"aGVsbG8td29ybGQ=") == b"hello-world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with special characters
    assert base64_decode(b"") == b""
    
    # Test with non-ASCII input (should be ignored)
    assert base64_decode("dGVzdA==\x80") == b"test"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("not base64")
```


# LLM-generated content at query #6
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding with no padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with special characters
    encoded_special = base64_encode(b"\x00\x01\x02\xff")
    assert base64_decode(encoded_special) == b"\x00\x01\x02\xff"
    
    # Test with long data
    long_data = b"x" * 1000
    encoded_long = base64_encode(long_data)
    assert base64_decode(encoded_long) == long_data
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test invalid base64 data with correct length but wrong characters
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8!!!")
    
    # Test binary data roundtrip
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
```


# LLM-generated content at query #7
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64.base64_encode(original)
    decoded = base64.base64_decode(encoded)
    assert decoded == original, f"Expected {original}, got {decoded}"
    
    # Test decoding with string input
    encoded_str = "aGVsbG8gd29ybGQ"
    decoded = base64.base64_decode(encoded_str)
    assert decoded == b"hello world", f"Expected b'hello world', got {decoded}"
    
    # Test decoding with padding
    encoded_with_padding = b"aGVsbG8gd29ybGQ="
    decoded = base64.base64_decode(encoded_with_padding)
    assert decoded == b"hello world", f"Expected b'hello world', got {decoded}"
    
    # Test decoding with missing padding
    encoded_no_padding = b"aGVsbG8gd29ybGQ"
    decoded = base64.base64_decode(encoded_no_padding)
    assert decoded == b"hello world", f"Expected b'hello world', got {decoded}"
    
    # Test decoding empty string
    decoded = base64.base64_decode(b"")
    assert decoded == b"", f"Expected b'', got {decoded}"
    
    # Test decoding single character
    decoded = base64.base64_decode(b"aA==")
    assert decoded == b"h", f"Expected b'h', got {decoded}"
    
    # Test invalid base64 data raises BadData
    try:
        base64.base64_decode(b"!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test decoding with special URL-safe characters
    original = bytes(range(256))
    encoded = base64.base64_encode(original)
    decoded = base64.base64_decode(encoded)
    assert decoded == original, f"Decoding failed for full byte range"
    
    # Test decoding numeric values
    encoded_num = base64.base64_encode(b"12345")
    decoded = base64.base64_decode(encoded_num)
    assert decoded == b"12345", f"Expected b'12345', got {decoded}"
    
    # Test decoding with various lengths
    for length in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64]:
        original = b"x" * length
        encoded = base64.base64_encode(original)
        decoded = base64.base64_decode(encoded)
        assert decoded == original, f"Failed for length {length}: expected {original}, got {decoded}"
```


# LLM-generated content at query #8
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"VGVzdA==") == b"Test"
    
    # Test bytes input without padding
    assert base64_decode(b"VGVzdA") == b"Test"
    
    # Test special characters (URL-safe)
    assert base64_decode("_-w==") == b"\xff\xeb"
    
    # Test longer string
    assert base64_decode("VGhpcyBpcyBhIHRlc3Q=") == b"This is a test"
    
    # Test invalid data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid bytes raises BadData
    try:
        base64_decode(b"\xff\xff\xff")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test unicode string input
    assert base64_decode("w6TDtsOcw7w=") == "äöü".encode("utf-8")
```


# LLM-generated content at query #9
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
    
    # Test with various valid inputs
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"
    assert base64_decode("d29ybGQ=") == b"world"
    assert base64_decode("d29ybGQ") == b"world"
    
    # Test that it raises BadData for invalid input
    try:
        base64_decode("invalid!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input  
    try:
        base64_decode(b"\xff\xfe\x00\x01")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with invalid characters
    try:
        base64_decode("aGVsbG8=" + chr(0x80))
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #10
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test with empty string
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test with padding
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test without padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test with special characters
    assert base64_decode("dGVzdC93aXRoL3NwZWNpYWw=") == b"test/with/special"
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-base64 characters
    try:
        base64_decode("not valid base64")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with unicode string
    assert base64_decode(base64_encode("héllo")) == "héllo".encode("utf-8")
```


# LLM-generated content at query #11
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    encoded = base64_encode("hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # missing padding
    
    # Test with URL-safe characters
    encoded = base64_encode(b"\xff\xfb\x00")
    assert base64_decode(encoded) == b"\xff\xfb\x00"
    
    # Test empty input
    assert base64_decode(b"") == b""
    
    # Test with only padding
    assert base64_decode(b"==") == b""
    assert base64_decode(b"===") == b""
    
    # Test invalid base64 data
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with non-ASCII characters in input (should be ignored)
    result = base64_decode(b"\xff\x80aGVsbG8=")
    assert result == b"hello"
```


# LLM-generated content at query #12
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with URL-safe characters
    original = b"test+data/with=special"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with non-ASCII characters in input
    try:
        base64_decode("aGVsbG8=\x80")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #13
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with text input
    encoded = base64_encode("hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test empty string
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test with padding
    encoded = base64_encode(b"a")
    assert base64_decode(encoded) == b"a"
    
    # Test with special characters
    encoded = base64_encode(b"\x00\x01\x02")
    assert base64_decode(encoded) == b"\x00\x01\x02"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with non-ASCII characters in input (should be ignored)
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with invalid data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with incomplete padding
    try:
        base64_decode(b"aGVs")
        assert False, "Should have raised BadData"  
    except BadData:
        pass
```


# LLM-generated content at query #14
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    decoded = base64_decode("SGVsbG8=")
    assert decoded == b"Hello"
    
    # Test decoding without padding
    decoded = base64_decode("SGVsbG8")
    assert decoded == b"Hello"
    
    # Test empty string
    decoded = base64_decode("")
    assert decoded == b""
    
    # Test bytes input
    decoded = base64_decode(b"V29ybGQ=")
    assert decoded == b"World"
    
    # Test URL-safe characters (using - and _ instead of + and /)
    decoded = base64_decode("Pj4-Pz8_")
    assert decoded == b">>>??"
    
    # Test with special characters
    decoded = base64_decode("AAECAwQFBgcICQ==")
    assert decoded == b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
```


# LLM-generated content at query #15
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test without padding
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with URL-safe characters
    assert base64_decode("aGVsbG8t") == b"hello-"
    assert base64_decode("aGVsbG9f") == b"hello_"
    
    # Test binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test padding with multiple missing bytes
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    
    # Test invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
```


# LLM-generated content at query #16
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test decoding bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special characters
    assert base64_decode("aGVsbG8vd29ybGQ=") == b"hello/world"
    assert base64_decode("aGVsbG8rd29ybGQ=") == b"hello+world"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode("not-base64!")
```


# LLM-generated content at query #17
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with special URL-safe characters
    assert base64_decode("_-w") == b"\xff\xeb"
    
    # Test various padding scenarios
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with numbers
    assert base64_decode("MTIz") == b"123"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("abc$%^")
    
    # Test with non-ASCII characters (ignored)
    assert base64_decode("SGVsbG8gV29ybGQ=\x00") == b"Hello World"
```


# LLM-generated content at query #18
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    result = base64_decode("dGVzdA==")
    assert result == b"test"
    
    # Test decoding without padding
    result = base64_decode("dGVzdA")
    assert result == b"test"
    
    # Test decoding bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test decoding URL-safe base64
    result = base64_decode("dGVzdA")
    assert result == b"test"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding with URL-safe characters (+, / replaced by -, _)
    result = base64_decode("Pj4-Pg==")
    assert result == b">>>"
    
    # Test decoding binary data
    result = base64_decode("AAECAwQFBgcI")
    assert result == bytes(range(9))
    
    # Test that invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding with special characters
    result = base64_decode("")
    assert result == b""
```


# LLM-generated content at query #19
#--------------------------

```python
def test_base64_decode():
    # Test normal valid input
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with string input
    decoded_from_str = base64_decode("aGVsbG8gd29ybGQ")
    assert decoded_from_str == b"hello world"
    
    # Test with bytes input
    decoded_from_bytes = base64_decode(b"aGVsbG8gd29ybGQ")
    assert decoded_from_bytes == b"hello world"
    
    # Test empty input
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
    
    # Test with padding
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YQ") == b"a"  # without padding
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test non-base64 characters raises BadData
    try:
        base64_decode(b"hello world")  # space is invalid in base64
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters from the alphabet
    encoded_special = base64_encode(b"\xff\xfe\xfd")
    assert base64_decode(encoded_special) == b"\xff\xfe\xfd"
```


# LLM-generated content at query #20
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 string
    result = base64_decode("dGVzdA==")
    assert result == b"test"
    
    # Test without padding
    result = base64_decode("dGVzdA")
    assert result == b"test"
    
    # Test with URL-safe characters
    result = base64_decode("dGVzdA_-")
    # URL-safe decoding, "-" and "_" are valid URL-safe characters
    # This is valid base64url encoding
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test with special characters
    result = base64_decode("aGVsbG8gd29ybGQ=")
    assert result == b"hello world"
    
    # Test with binary data
    data = b"\x00\x01\x02\xff\xfe\xfd"
    encoded = base64_encode(data)
    result = base64_decode(encoded)
    assert result == data
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with extra padding
    result = base64_decode("dGVzdA====")
    assert result == b"test"
```


# LLM-generated content at query #21
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test without padding
    assert base64_decode("dGVzdA") == b"test"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test with URL-safe characters
    assert base64_decode("Pj4-Pj4") == b">>>>"
    
    # Test with special characters
    assert base64_decode("Lg==") == b"."
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("dGVzdA==\x80") == b"test"
```


# LLM-generated content at query #22
#--------------------------

```python
def test_base64_decode():
    # Test normal encoding
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test string input
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test without padding
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with special characters
    assert base64_decode(b"aGVsbG8+") == b"hello>"
    
    # Test with URL-safe characters
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test with unicode characters
    assert base64_decode(b"w6Rh") == b"\xe4a"
    
    # Test that it raises BadData for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with invalid characters
    try:
        base64_decode(b"test\x00")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #23
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"VGVzdA==") == b"Test"
    
    # Test bytes input without padding
    assert base64_decode(b"VGVzdA") == b"Test"
    
    # Test with special characters
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"
    
    # Test with URL-safe characters
    assert base64_decode("dGVzdC0t") == b"test--"
    
    # Test with underscore (URL-safe variant)
    assert base64_decode("dGVzdF8=") == b"test_"
    
    # Test raises BadData on invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test raises BadData on truncated input
    with pytest.raises(BadData):
        base64_decode("SGVsbG8")  # This should work, but let's test a truly invalid one
        base64_decode("123")  # Invalid base64
```


# LLM-generated content at query #24
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test decoding bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding with URL-safe characters
    result = base64_decode("aGVsbG8td29ybGQ")
    assert result == b"hello-world"
    
    # Test decoding with special characters
    result = base64_decode("YSBiCg==")
    assert result == b"a b\n"
    
    # Test that BadData is raised for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that BadData is raised for non-base64 characters
    try:
        base64_decode("Hello World")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #25
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    
    # Test with bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    assert base64_decode("PDw8PA==") == b"<<<<"
    
    # Test with special characters
    assert base64_decode("YWJjZGVmZw==") == b"abcdefg"
    
    # Test with ascii encoding
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with bytes that need padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test that BadData is raised for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!")
    with pytest.raises(BadData):
        base64_decode(" ")
    with pytest.raises(BadData):
        base64_decode("abc123!!!")
```


# LLM-generated content at query #26
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8=")
    assert result == b"Hello"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test very short input
    result = base64_decode("YQ==")
    assert result == b"a"
    
    # Test special characters in base64
    result = base64_decode("_-w=")
    assert result == b"\xff\xec"
    
    # Test numbers
    result = base64_decode("MTIzNA==")
    assert result == b"1234"
```


# LLM-generated content at query #27
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with string input
    encoded = base64_encode("test string")
    result = base64_decode(encoded)
    assert isinstance(result, bytes)
    assert result == b"test string"
    
    # Test decoding empty bytes
    encoded = base64_encode(b"")
    assert base64_decode(encoded) == b""
    
    # Test decoding empty string
    encoded = base64_encode("")
    assert base64_decode(encoded) == b""
    
    # Test decoding single byte
    encoded = base64_encode(b"\x00")
    assert base64_decode(encoded) == b"\x00"
    
    # Test decoding with special characters
    encoded = base64_encode(b"\xff\xfe\xfd\xfc")
    assert base64_decode(encoded) == b"\xff\xfe\xfd\xfc"
    
    # Test decoding with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test decoding raises BadData for invalid input
    try:
        base64_decode("!!!invalid base64!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding raises BadData for empty string (no valid base64)
    try:
        base64_decode("")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test roundtrip with various lengths
    for length in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 100, 1000]:
        data = bytes([i % 256 for i in range(length)])
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Roundtrip failed for length {length}"
    
    # Test with bytes input directly
    encoded_bytes = base64_encode(b"direct bytes")
    assert base64_decode(encoded_bytes) == b"direct bytes"
```


# LLM-generated content at query #28
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with padding
    encoded_with_padding = b"aGVsbG8="
    decoded = base64_decode(encoded_with_padding)
    assert decoded == b"hello"
    
    # Test with string input
    decoded = base64_decode("aGVsbG8=")
    assert decoded == b"hello"
    
    # Test with no padding
    encoded_no_padding = b"aGVsbG8"
    decoded = base64_decode(encoded_no_padding)
    assert decoded == b"hello"
    
    # Test with empty string
    decoded = base64_decode(b"")
    assert decoded == b""
    
    # Test with special URL-safe characters
    original = b"\xff\xfe\x00\x01"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with non-ASCII characters in string input
    decoded = base64_decode("aGVsbG8=")
    assert decoded == b"hello"
```


# LLM-generated content at query #29
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with padding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aA==") == b"h"
    
    # Test with no padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with invalid characters raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("invalid!!!")
    
    # Test with wrong length (invalid base64)
    with pytest.raises(BadData):
        base64_decode("a")
    
    # Test roundtrip with various data
    test_data = [
        b"",
        b"a",
        b"test",
        b"hello world",
        b"\x00\x01\x02",
        b"data with spaces",
        bytes(range(256))
    ]
    for data in test_data:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Roundtrip failed for {data!r}"
```


# LLM-generated content at query #30
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with string input
    encoded_str = base64_encode("hello world")
    decoded_str = base64_decode(encoded_str)
    assert decoded_str == b"hello world"
    
    # Test with empty bytes
    encoded_empty = base64_encode(b"")
    decoded_empty = base64_decode(encoded_empty)
    assert decoded_empty == b""
    
    # Test with empty string
    encoded_empty_str = base64_encode("")
    decoded_empty_str = base64_decode(encoded_empty_str)
    assert decoded_empty_str == b""
    
    # Test with padding
    encoded_padded = base64_encode(b"a")
    decoded_padded = base64_decode(encoded_padded)
    assert decoded_padded == b"a"
    
    # Test with special characters
    encoded_special = base64_encode(b"\x00\xff\xfe")
    decoded_special = base64_decode(encoded_special)
    assert decoded_special == b"\x00\xff\xfe"
    
    # Test that BadData is raised for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with numbers and text
    encoded_num = base64_encode(b"12345")
    decoded_num = base64_decode(encoded_num)
    assert decoded_num == b"12345"
    
    # Test with unicode characters (after encoding to bytes)
    encoded_unicode = base64_encode("héllo wörld")
    decoded_unicode = base64_decode(encoded_unicode)
    assert decoded_unicode == "héllo wörld".encode("utf-8")
```


# LLM-generated content at query #31
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test with string input
    encoded = base64_encode("world")
    assert base64_decode(encoded) == b"world"
    
    # Test empty string
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test with special characters
    original = b"test\x00data"
    assert base64_decode(base64_encode(original)) == original
    
    # Test with padding
    assert base64_decode("aGVsbG8") == b"hello"
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test invalid input raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters in string input
    assert base64_decode(base64_encode("héllo")) == "héllo".encode("utf-8")
```


# LLM-generated content at query #32
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with padding
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test decoding without padding
    result = base64_decode("aGVsbG8")
    assert result == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test with special URL-safe characters
    result = base64_decode("Pz4_")
    assert result == b"?>"
    
    # Test with binary data
    original = bytes(range(256))
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input (should be ignored)
    result = base64_decode("aGVsbG8\x80")
    assert result == b"hello"
```


# LLM-generated content at query #33
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decoding with different padding
    assert base64_decode("aGVsbG8") == b"hello"
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8==") == b"hello"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with special characters
    test_data = b"test\x00data\xff"
    encoded = base64_encode(test_data)
    decoded = base64_decode(encoded)
    assert decoded == test_data
    
    # Test decoding raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test decoding with bytes input
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding Unicode string
    assert base64_decode("aGVsbG8") == b"hello"
```


# LLM-generated content at query #34
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test with URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    
    # Test various lengths
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with different characters
    assert base64_decode("_-5v") == b"\xff\xeeo"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters (should be ignored due to errors="ignore")
    assert base64_decode("abcd\x80ef==") == b"\x69\xb7\x1d\xef"
```


# LLM-generated content at query #35
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test decoding without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test decoding URL-safe characters
    assert base64_decode(b"_-4=") == b"\xff\xef\xb8"
    assert base64_decode("_-4=") == b"\xff\xef\xb8"
    
    # Test roundtrip with base64_encode
    original = b"Test data with special chars: \x00\xff\x01"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decoding raises BadData for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding with non-ASCII characters (should be ignored)
    result = base64_decode("aGVsbG8g\udce3world")
    assert result is not None
    
    # Test decoding binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
```


# LLM-generated content at query #36
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test decoding bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with special URL-safe characters
    assert base64_decode("_-w=") == b"\xff\xec"
    
    # Test decoding longer string
    result = base64_decode("VGhpcyBpcyBhIHRlc3Q=")
    assert result == b"This is a test"
    
    # Test that BadData is raised for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that BadData is raised for non-base64 characters
    try:
        base64_decode("not@base64")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with unicode string that needs ASCII encoding
    result = base64_decode("dGVzdA==")
    assert result == b"test"
```


# LLM-generated content at query #37
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test decoding with URL-safe characters
    result = base64_decode("aGVsbG8td29ybGQ")
    assert result == b"hello-world"
    
    # Test decoding bytes input
    result = base64_decode(b"SGVsbG8=")
    assert result == b"Hello"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding with multiple padding
    result = base64_decode("YQ==")
    assert result == b"a"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test decoding with special characters
    result = base64_decode("dGVzdC1zdHJpbmctd2l0aC1zcGVjaWFsLWNoYXJz")
    assert result == b"test-string-with-special-chars"
```


# LLM-generated content at query #38
#--------------------------

```python
def test_base64_decode():
    # Test valid base64 encoded string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original, f"Expected {original}, got {decoded}"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with padding variations
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with special URL-safe characters
    encoded_url = base64_encode(b"\xff\xfe")
    decoded_url = base64_decode(encoded_url)
    assert decoded_url == b"\xff\xfe"
    
    # Test that invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that non-base64 characters raise BadData
    try:
        base64_decode("aGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with unicode string
    assert base64_decode("w7xibGVy") == b"\xc3\xbc\x62\x6c\x65\x72"  # "über" in utf-8
```


# LLM-generated content at query #39
#--------------------------

```python
def test_base64_decode():
    # Test basic encoding/decoding roundtrip
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with string input
    encoded_str = "aGVsbG8gd29ybGQ"
    assert base64_decode(encoded_str) == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVs") == b"hel"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with URL-safe characters
    url_safe = base64_encode(b"test_data+with/special~chars")
    assert base64_decode(url_safe) == b"test_data+with/special~chars"
    
    # Test with various byte values
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
    
    # Test invalid input raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode(b"aGVsbG8\xff") == b"hello"
```


# LLM-generated content at query #40
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"

    # Test with empty string
    assert base64_decode("") == b""

    # Test with padding variations
    assert base64_decode("aGVsbG8") == b"hello"  # without padding
    assert base64_decode("aGVsbG8=") == b"hello"  # with one padding
    assert base64_decode("YQ") == b"a"  # without padding

    # Test with URL-safe characters
    assert base64_decode("aGVsbG8_d29ybGQ") == b"hello?world"  # _ instead of /
    assert base64_decode("aGVsbG8-d29ybGQ") == b"hello>world"  # - instead of +

    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"

    # Test with special characters
    original = b"\x00\x01\x02\xff\xfe"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test with long data
    long_data = b"x" * 1000
    encoded = base64_encode(long_data)
    assert base64_decode(encoded) == long_data
```


# LLM-generated content at query #41
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test with bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with padding removed
    assert base64_decode("YWJj") == b"abc"
    
    # Test with full padding
    assert base64_decode("YWJjZA==") == b"abcd"
    
    # Test with special URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    assert base64_decode("PDw8PA==") == b"<<<<"
    
    # Test that BadData is raised for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with characters outside base64 alphabet
    try:
        base64_decode("hello world")  # space is invalid
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #42
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 string
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test decoding bytes input
    assert base64_decode(b"SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with special characters
    assert base64_decode("dGVzdC91cmw") == b"test/url"
    
    # Test decoding with underscores (URL-safe)
    assert base64_decode("dGVzdF91cmw=") == b"test_url"
    
    # Test decoding raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test decoding raises BadData for truncated input
    with pytest.raises(BadData):
        base64_decode("SGVsbG8")
```


# LLM-generated content at query #43
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"VGVzdA==")
    assert result == b"Test"
    
    # Test URL-safe characters
    result = base64_decode("_-x")
    assert result == b"\xff\xeb"
    
    # Test invalid input raises BadData
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test very short invalid input
    try:
        base64_decode("!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #44
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding various byte lengths
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    
    # Test decoding with special URL-safe characters
    encoded_url = base64_encode(b"test data with + and /")
    assert base64_decode(encoded_url) == b"test data with + and /"
    
    # Test that invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test that non-base64 data raises BadData
    with pytest.raises(BadData):
        base64_decode(b"not-base64-characters!")
```


# LLM-generated content at query #45
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters
    assert base64_decode("_-w") == b"\xff\xe0"  # contains - and _
    
    # Test with special characters
    assert base64_decode("") == b""
    
    # Test BadData exception for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with ascii encoding errors ignored
    assert base64_decode(b"SGVsbG8=\xff") == b"Hello"
```


# LLM-generated content at query #46
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding of valid base64 string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decoding with padding
    encoded_with_padding = base64.b64encode(b"test").decode()
    decoded = base64_decode(encoded_with_padding)
    assert decoded == b"test"
    
    # Test decoding bytes input
    encoded_bytes = base64_encode(b"bytes")
    decoded = base64_decode(encoded_bytes)
    assert decoded == b"bytes"
    
    # Test decoding string input
    encoded_str = base64_encode("string")
    decoded = base64_decode(encoded_str)
    assert decoded == b"string"
    
    # Test decoding empty string
    decoded = base64_decode("")
    assert decoded == b""
    
    # Test decoding with special characters
    special = b"\x00\x01\x02\xff"
    encoded = base64_encode(special)
    decoded = base64_decode(encoded)
    assert decoded == special
    
    # Test decoding invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test decoding with missing padding (should still work)
    encoded_no_padding = base64_encode(b"test").rstrip(b"=")
    decoded = base64_decode(encoded_no_padding)
    assert decoded == b"test"
```


# LLM-generated content at query #47
#--------------------------

```python
def test_base64_decode():
    # Test decoding standard input
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"YQ==") == b"a"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode(b"YQ") == b"a"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding with invalid characters raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"invalid!!!")
    
    # Test decoding with wrong length raises BadData
    with pytest.raises(BadData):
        base64_decode(b"a")
```


# LLM-generated content at query #48
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test special characters in URL-safe base64
    encoded_with_special = base64_encode(b"\xff\xfe\xfd\xfc")
    assert base64_decode(encoded_with_special) == b"\xff\xfe\xfd\xfc"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"abc")
    
    # Test bytes input with ascii encoding
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test that non-ASCII characters in input are ignored
    assert base64_decode("dGVzdA==") == b"test"
```


# LLM-generated content at query #49
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
    
    # Test with special characters (URL-safe)
    assert base64_decode("aGVsbG8v") == b"hello/"  # URL-safe base64 uses - and _
    assert base64_decode("aGVsbG8t") == b"hello-"  # URL-safe base64 uses - and _
    
    # Test invalid data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode("123")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test various valid inputs
    assert base64_decode("") == b""
    assert base64_decode("Zg==") == b"f"
    assert base64_decode("Zm8=") == b"fo"
    assert base64_decode("Zm9v") == b"foo"
```


# LLM-generated content at query #50
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"YQ==") == b"a"
    
    # Test decoding with different lengths
    assert base64_decode(b"") == b""
    assert base64_decode(b"YQ") == b"a"  # Missing padding
    assert base64_decode(b"YQ") == b"a"  # Missing padding
    assert base64_decode(b"YQ==") == b"a"  # Full padding
    
    # Test decoding string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding bytes with no padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding special characters
    encoded_url = base64_encode(b"test data with spaces")
    assert base64_decode(encoded_url) == b"test data with spaces"
    
    # Test decoding binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
    
    # Test that BadData is raised for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode(b"aGVsbG8\x00")  # Invalid character
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII input (should be ignored with errors="ignore")
    assert base64_decode(b"aGVsbG8\xff") == b"hello"
```


# LLM-generated content at query #51
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test URL-safe format (no padding)
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test with special characters
    test_bytes = b"\x00\x01\x02\xff\xfe"
    encoded = base64_encode(test_bytes)
    assert base64_decode(encoded) == test_bytes
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("aGVsbG8=\x80") == b"hello"
```


# LLM-generated content at query #52
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    result = base64_decode(encoded)
    assert result == b"hello world"
    
    # Test with string input
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test padding variations
    # "aGVsbG8=" decodes to "hello" (1 padding char)
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # "aGVsbG8" without padding should also work
    result = base64_decode("aGVsbG8")
    assert result == b"hello"
    
    # Test longer strings
    original = b"Hello, World! This is a test with more characters."
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with binary data
    original = b"\x00\x01\x02\xff\xfe"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with invalid input should raise BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-base64 characters
    with pytest.raises(BadData):
        base64_decode("aGVs#bG8=")
    
    # Test bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test with unicode string that looks like valid base64
    result = base64_decode("YXNjaWk=")
    assert result == b"ascii"
```


# LLM-generated content at query #53
#--------------------------

```python
def test_base64_decode():
    # Test basic string encoding
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test with URL-safe characters (replacing + with - and / with _)
    result = base64_decode("Pj4-Pg")
    assert result == b">>>"
    
    # Test with bytes input
    result = base64_decode(b"VGVzdA==")
    assert result == b"Test"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test single character
    result = base64_decode("WA==")
    assert result == b"X"
    
    # Test with special characters
    result = base64_decode("YSBi")
    assert result == b"a b"
    
    # Test with URL-safe characters
    result = base64_decode("Pj4-Pg==")
    assert result == b">>>"
    
    # Test with underscores (URL-safe)
    result = base64_decode("X19f")
    assert result == b"___"
    
    # Test with dashes (URL-safe)
    result = base64_decode("LS0t")
    assert result == b"---"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test invalid characters
    with pytest.raises(BadData):
        base64_decode("ABCD$%^")
```


# LLM-generated content at query #54
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test URL-safe encoding (no padding)
    assert base64_decode(b"dGVzdA") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with various padding lengths
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with special URL-safe characters
    assert base64_decode("_-A=") == b"\xfb\xe0"
    
    # Test with bytes containing non-ASCII characters
    assert base64_decode("w7xsw6TDtmzDtsO2") == b"\xc3\xbc\x6c\xc3\xa4\x64\xc3\xb6\x6c\xc3\xb6\xc3\xb6"
    
    # Test with invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("invalid!!!")
    
    # Test with non-base64 characters
    with pytest.raises(BadData):
        base64_decode("test\x00")
```


# LLM-generated content at query #55
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decoding with padding
    encoded_with_padding = b"aGVsbG8="
    decoded = base64_decode(encoded_with_padding)
    assert decoded == b"hello"
    
    # Test decoding without padding
    encoded_without_padding = b"aGVsbG8"
    decoded = base64_decode(encoded_without_padding)
    assert decoded == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with string input
    decoded = base64_decode("aGVsbG8=")
    assert decoded == b"hello"
    
    # Test decoding with special characters
    original = b"\x00\x01\x02\xff"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"not valid base64@#$%")
```


# LLM-generated content at query #56
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with string input
    encoded_str = base64_encode("test data")
    assert base64_decode(encoded_str) == b"test data"
    
    # Test decoding with various lengths
    for i in range(1, 10):
        data = b"a" * i
        encoded = base64_encode(data)
        assert base64_decode(encoded) == data
    
    # Test decoding with special characters
    data = b"\x00\x01\x02\xff\xfe"
    encoded = base64_encode(data)
    assert base64_decode(encoded) == data
    
    # Test decoding empty string
    encoded = base64_encode(b"")
    assert base64_decode(encoded) == b""
    
    # Test decoding with padding
    encoded_padded = base64_encode(b"test") + b"="
    assert base64_decode(encoded_padded) == b"test"
    
    # Test that BadData is raised for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")
    
    # Test that BadData is raised for non-base64 characters
    with pytest.raises(BadData):
        base64_decode("not base64 data!")
```


# LLM-generated content at query #57
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with text input
    encoded = base64_encode("test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test empty input
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with padding
    encoded = base64_encode(b"abc")
    assert base64_decode(encoded) == b"abc"
    
    # Test with special URL-safe characters
    encoded = base64_encode(b"hello+world/test?query=1")
    assert base64_decode(encoded) == b"hello+world/test?query=1"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with string containing non-ASCII
    encoded = base64_encode("héllo wörld")
    assert base64_decode(encoded) == "héllo wörld".encode("utf-8")
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with incorrect characters
    with pytest.raises(BadData):
        base64_decode(b"abc$%^")
    
    # Test with truncated data (no padding)
    with pytest.raises(BadData):
        base64_decode(b"abc")
```


# LLM-generated content at query #58
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with special URL-safe characters
    assert base64_decode(b"Pj4-Pg==") == b">>>?"
    
    # Test with binary data
    assert base64_decode("AAECAwQFBgc=") == bytes(range(8))
    
    # Test raising BadData on invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters in input (should be ignored)
    assert base64_decode("SGVsbG8=\xff") == b"Hello"
```


# LLM-generated content at query #59
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters
    assert base64_decode("dGVzdF91cmw") == b"test_url"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test binary data
    assert base64_decode("AAECAwQFBgcI") == bytes(range(9))
    
    # Test with special characters
    assert base64_decode("Lg==") == b"."
    assert base64_decode("Lw==") == b"/"
    
    # Test with unicode characters (must be ascii compatible)
    assert base64_decode("w6TDpMO8") == "ääü".encode()
    
    # Test BadData exception for invalid base64
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData as e:
        assert str(e) == "Invalid base64-encoded data"
    
    # Test BadData exception for invalid characters
    try:
        base64_decode("not valid!")
        assert False, "Should have raised BadData"
    except BadData as e:
        assert str(e) == "Invalid base64-encoded data"
```


# LLM-generated content at query #60
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test decoding with strings
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test decoding with padding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8==") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test that BadData is raised for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test that BadData is raised for non-base64 characters
    with pytest.raises(BadData):
        base64_decode("not valid base64!")
    
    # Test special characters
    encoded_special = base64_encode(b"\x00\xff\xfe")
    assert base64_decode(encoded_special) == b"\x00\xff\xfe"
    
    # Test URL-safe characters
    assert base64_decode("_-w") == b"\xfb\xc0"  # - and _ are URL-safe alternatives to + and /
    
    # Test with various padding lengths
    assert base64_decode("YQ") == b"a"  # 1 char
    assert base64_decode("YWI") == b"ab"  # 2 chars
    assert base64_decode("YWJj") == b"abc"  # 3 chars
```


# LLM-generated content at query #61
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with missing padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test with standard base64 (with + and /)
    assert base64_decode(b"aGVsbG8gd29ybGQ=") == b"hello world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test single character
    assert base64_decode(b"aA==") == b"h"
    
    # Test with string input (not bytes)
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test that it raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test that it raises BadData for input with invalid characters
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8$")
    
    # Test with special URL-safe characters
    assert base64_decode(b"Pj4_Pj4") == b">>?>"
    
    # Test binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    decoded_binary = base64_decode(encoded_binary)
    assert decoded_binary == binary_data
```


# LLM-generated content at query #62
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with text input (string)
    encoded = base64_encode("test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with special characters
    encoded = base64_encode(b"hello\nworld\t!")
    assert base64_decode(encoded) == b"hello\nworld\t!"
    
    # Test with unicode characters
    encoded = base64_encode("héllo wörld")
    assert base64_decode(encoded) == "héllo wörld".encode("utf-8")
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with wrong length padding
    try:
        base64_decode(b"aGVsbG8=")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #63
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with special characters
    encoded_special = base64_encode(b"\x00\x01\x02\xff\xfe")
    assert base64_decode(encoded_special) == b"\x00\x01\x02\xff\xfe"
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode("aGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters
    try:
        base64_decode("aGVsbG8\x80")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #64
#--------------------------

```python
def test_base64_decode():
    # Test with standard base64 encoded string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original, f"Expected {original}, got {decoded}"
    
    # Test with empty bytes
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with padding variations
    assert base64_decode(b"aGVsbG8") == b"hello"  # Missing padding
    assert base64_decode("aGVsbG8=") == b"hello"  # With padding
    
    # Test with unicode string input
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"
    
    # Test with special characters in URL-safe alphabet
    original = b"\xff\xfe\x00\x01"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test that BadData is raised for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-base64 characters
    try:
        base64_decode(b"aGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input that has no valid base64 characters
    try:
        base64_decode(b"???")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test roundtrip with various byte lengths
    for length in [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 100]:
        original = bytes(range(length))
        encoded = base64_encode(original)
        decoded = base64_decode(encoded)
        assert decoded == original, f"Failed for length {length}: {decoded} != {original}"
```


# LLM-generated content at query #65
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test with string input
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test without padding
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test URL-safe characters
    assert base64_decode(b"_-x") == b"\xff\xeb"
    
    # Test with special characters
    assert base64_decode(b"") == b""
    
    # Test with single character
    assert base64_decode(b"Zg==") == b"f"
    
    # Test with two characters
    assert base64_decode(b"Zm8=") == b"fo"
    
    # Test with three characters
    assert base64_decode(b"Zm9v") == b"foo"
    
    # Test with long string
    long_string = b"a" * 100
    encoded = base64_encode(long_string)
    assert base64_decode(encoded) == long_string
    
    # Test with bytes that have different padding
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    
    # Test with invalid base64 data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode(b"dGVzdA==\xff") == b"test"
```


# LLM-generated content at query #66
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 string
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test URL-safe base64 string (without padding)
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test with special characters
    result = base64_decode("aGVsbG8g8J+YgQ==")
    assert result == b"hello \xf0\x9f\x98\x81"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test invalid characters
    with pytest.raises(BadData):
        base64_decode("SGVsbG8gV29ybGQ==")
```


# LLM-generated content at query #67
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    encoded_padded = base64.b64encode(b"test").decode()  # regular base64
    assert base64_decode(encoded_padded) == b"test"
    
    # Test with empty string
    assert base64_decode(b"") == b""
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with invalid data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode("not base64 at all")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special URL-safe characters
    original_bytes = b"\xff\xfe\x00\x01"
    encoded = base64_encode(original_bytes)
    assert base64_decode(encoded) == original_bytes
    
    # Test with missing padding
    assert base64_decode(b"aGVsbG8") == b"hello"  # missing one =
    assert base64_decode(b"aGVs") == b"hes"  # missing two =s
```


# LLM-generated content at query #68
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with string input
    encoded_str = base64.b64encode(b"test").decode('ascii')
    assert base64_decode(encoded_str) == b"test"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding empty bytes
    assert base64_decode("") == b""
    
    # Test decoding with padding
    encoded_padded = base64.urlsafe_b64encode(b"a").decode('ascii')
    assert base64_decode(encoded_padded) == b"a"
    
    # Test decoding binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
    
    # Test decoding invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding with invalid characters
    try:
        base64_decode(b"abcxyz123!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding with whitespace (should fail)
    try:
        base64_decode(b"abc def")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding bytes with padding issues
    encoded_no_pad = base64.urlsafe_b64encode(b"test").rstrip(b"=")
    assert base64_decode(encoded_no_pad) == b"test"
    
    # Test decoding with single character
    assert base64_decode(base64_encode(b"a")) == b"a"
    
    # Test decoding with special characters
    special = b"\x00\x01\x02\xff\xfe"
    assert base64_decode(base64_encode(special)) == special
```


# LLM-generated content at query #69
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("VGVzdA==") == b"Test"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("VGVzdA") == b"Test"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special characters
    assert base64_decode("aGVsbG8td29ybGQ") == b"hello-world"
    assert base64_decode("aGVsbG8vd29ybGQ") == b"hello/world"
    
    # Test with numbers
    assert base64_decode("MTIzNDU2") == b"123456"
    
    # Test with special URL-safe characters
    assert base64_decode("aGVsbG8_d29ybGQ") == b"hello?world"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode("1234")
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe")
```


# LLM-generated content at query #70
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test with string input
    encoded = base64_encode("test data")
    decoded = base64_decode(encoded)
    assert decoded == b"test data"

    # Test with already padded input
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"

    # Test with unpadded input
    result = base64_decode("aGVsbG8")
    assert result == b"hello"

    # Test with empty string
    result = base64_decode("")
    assert result == b""

    # Test with bytes input
    result = base64_decode(b"d29ybGQ=")
    assert result == b"world"

    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test with non-ASCII characters (should be ignored)
    result = base64_decode("aGVsbG8=\x80")
    assert result == b"hello"
```


# LLM-generated content at query #71
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding with special URL-safe characters
    result = base64_decode("Pj4_Pz8")
    assert result == b">>???\xbf\xbf"
    
    # Test that invalid base64 raises BadData
    try:
        base64_decode("!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test that invalid characters raise BadData
    try:
        base64_decode("SGVsbG8gV29ybGQ!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test decoding with different string lengths
    result = base64_decode("YQ==")
    assert result == b"a"
    
    result = base64_decode("YWI=")
    assert result == b"ab"
    
    result = base64_decode("YWJj")
    assert result == b"abc"
```


# LLM-generated content at query #72
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test single character
    assert base64_decode("Zg==") == b"f"
    
    # Test padding variations
    assert base64_decode("YQ==") == b"a"  # 2 padding chars
    assert base64_decode("YWI=") == b"ab"  # 1 padding char
    assert base64_decode("YWJj") == b"abc"  # 0 padding chars
    
    # Test URL-safe characters (- instead of +)
    assert base64_decode("_-w=") != b""  # Should not raise
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test binary data roundtrip
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    decoded_binary = base64_decode(encoded_binary)
    assert decoded_binary == binary_data
    
    # Test handling of whitespace in input (should ignore)
    result = base64_decode("a G V s b G 8 =")
    assert result == b"hello"
    
    # Test large data
    large_data = b"x" * 10000
    encoded_large = base64_encode(large_data)
    decoded_large = base64_decode(encoded_large)
    assert decoded_large == large_data
```


# LLM-generated content at query #73
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    
    # Test with bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    assert base64_decode(b"V29ybGQ") == b"World"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
    
    # Test special characters
    assert base64_decode("Pj4+Pg==") == b">>>>"
    assert base64_decode("Pj4-Pg==") == b">>>>"  # URL-safe variant
    
    # Test with padding
    assert base64_decode("YWJj") == b"abc"
    assert base64_decode("YWJjZA==") == b"abcd"
    
    # Test various lengths
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    assert base64_decode("YWJjZA==") == b"abcd"
    assert base64_decode("YWJjZGU=") == b"abcde"
    assert base64_decode("YWJjZGVm") == b"abcdef"
    
    # Test with invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode("abc!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("====")  # Just padding
    
    # Test that non-ASCII characters are ignored
    assert base64_decode("SGVsbG8gV29ybGQ=\x00") == b"Hello World"
```


# LLM-generated content at query #74
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with various data
    test_data = b"test data with spaces and special chars!@#$%"
    encoded = base64_encode(test_data)
    assert base64_decode(encoded) == test_data
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with single character
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YQ") == b"a"
    
    # Test raises BadData for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with unicode characters (should ignore errors)
    assert base64_decode(b"aGVsbG8=\xff") == b"hello"
```


# LLM-generated content at query #75
#--------------------------

```python
def test_base64_decode():
    # Test with standard URL-safe base64 string
    result = base64_decode("dGVzdA==")
    assert result == b"test"
    
    # Test without padding
    result = base64_decode("dGVzdA")
    assert result == b"test"
    
    # Test with bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test with empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with special characters
    result = base64_decode("_-x0=")
    assert result == b"\xff\x1d"
    
    # Test with longer input
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test with invalid base64 data
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters (should be ignored)
    result = base64_decode("dGVzdA==\x80")
    assert result == b"test"
```


# LLM-generated content at query #76
#--------------------------

```python
def test_base64_decode():
    # Test with valid base64 encoded string
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test with URL-safe base64 (no padding)
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with single character
    assert base64_decode("Zg==") == b"f"
    
    # Test with invalid base64 string
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters in input (should be ignored)
    assert base64_decode(b"SGVsbG8g\x80V29ybGQ=") == b"Hello World"
    
    # Test with special URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>> "
    assert base64_decode("Pj4-Pg") == b">>> "
```


# LLM-generated content at query #77
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 string
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test with padding
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test with special URL-safe characters
    result = base64_decode("_-x")
    assert len(result) > 0
    
    # Test with empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with single character
    result = base64_decode("WA==")
    assert result == b"X"
    
    # Test with multiple padding equals
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test with non-ASCII characters in input (should be ignored)
    result = base64_decode("SGVsbG8gV29ybGQ\x80")
    assert result == b"Hello World"


# LLM-generated content at query #78
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("VGVzdA==") == b"Test"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters
    assert base64_decode("Pz4_") == b"?>\xef"
    
    # Test with special characters
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YQ") == b"a"
    
    # Test invalid base64 data
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!")
    
    # Test very long valid string
    long_str = base64.b64encode(b"x" * 1000).decode()
    assert base64_decode(long_str) == b"x" * 1000
    
    # Test with null bytes
    assert base64_decode("AA==") == b"\x00"
    
    # Test numeric values
    assert base64_decode("MTIzNDU2Nzg5MA==") == b"1234567890"


# LLM-generated content at query #79
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters
    assert base64_decode("_-w") == b"\xff\xeb"
    
    # Test with special characters
    result = base64_decode("dGVzdGluZyB3aXRoIHNwZWNpYWwgY2hhcmFjdGVycyAhQCMkJV4mKigp")
    assert result == b"testing with special characters !@#$%^&*()"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test completely invalid characters
    try:
        base64_decode("\x00\x01\x02")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with padding that needs to be added
    assert base64_decode("dGVzdA") == b"test"
    assert base64_decode("dGVzdDEyMw") == b"test123"
```


# LLM-generated content at query #80
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("d29ybGQ=") == b"world"
    
    # Test without padding
    assert base64_decode("dGVzdA") == b"test"
    assert base64_decode("aGVsbG8") == b"hello"
    assert base64_decode("d29ybGQ") == b"world"
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with special URL-safe characters
    assert base64_decode("Pz4_") == b"?>?"
    assert base64_decode("Pz4_") == b"?>?"
    
    # Test raises BadData on invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("not base64 valid chars")
    
    # Test that non-ASCII characters are ignored in input
    assert base64_decode("dGVzdA\x80\x81") == b"test"
```


# LLM-generated content at query #81
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test URL-safe characters
    encoded = base64_encode(b"test data with + and /")
    decoded = base64_decode(encoded)
    assert decoded == b"test data with + and /"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with padding
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    
    # Test with ascii string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode(b"SGVsbG8=\xff") == b"Hello"
    assert base64_decode("SGVsbG8=\xff") == b"Hello"


# LLM-generated content at query #82
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode("dGVzdA") == b"test"
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test decoding with URL-safe characters
    assert base64_decode("Pz4_") == b"?>\xff"
    
    # Test decoding bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with multiple padding scenarios
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test decoding with special characters
    assert base64_decode("+/8=") == b"\xfb\xff"
    
    # Test invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test invalid characters raise BadData
    try:
        base64_decode("not valid base64!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test with unicode string
    assert base64_decode("w7xrw7xs") == b"\xc3\xbc\xc3\xbc\xc3\xbc"
```


# LLM-generated content at query #83
#--------------------------

```python
def test_base64_decode():
    # Test basic string encoding
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8=")
    assert result == b"Hello"
    
    # Test without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test URL-safe characters (replacing + with -, / with _)
    result = base64_decode("Pj4-Pg==")
    assert result == b">>>"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with special characters
    result = base64_decode("YWJj")
    assert result == b"abc"
    
    # Test with numbers
    result = base64_decode("MTIz")
    assert result == b"123"
    
    # Test with mixed content
    result = base64_decode("dGVzdC11cmwtc2FmZQ==")
    assert result == b"test-url-safe"
```


# LLM-generated content at query #84
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with padding
    encoded_padded = base64.urlsafe_b64encode(b"test").decode("ascii")
    decoded = base64_decode(encoded_padded)
    assert decoded == b"test"
    
    # Test with string input
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test with bytes input
    result = base64_decode(b"aGVsbG8=")
    assert result == b"hello"
    
    # Test with no padding
    result = base64_decode(b"aGVsbG8")
    assert result == b"hello"
    
    # Test empty string
    result = base64_decode(b"")
    assert result == b""
    
    # Test special characters in URL-safe base64
    result = base64_decode(b"_-w")
    assert result == b"\xff\xec"
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with non-ASCII characters (should be ignored)
    result = base64_decode("aGVsbG8=\x80")
    assert result == b"hello"
```


# LLM-generated content at query #85
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 string
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test string without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test with special URL-safe characters
    result = base64_decode("aGVsbG8tX3dvcmxk")  # "hello_world"
    assert result == b"hello_world"
    
    # Test invalid base64 data
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test non-ASCII characters
    result = base64_decode("w7Zsw6Rmw6bDtsO2")  # "öläfööö"
    assert result == "öläfööö".encode("utf-8")


# LLM-generated content at query #86
#--------------------------

```python
def test_base64_decode():
    # Test decoding standard URL-safe base64
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test decoding with missing padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode(b"aGVsbG8gd29ybGQ") == b"hello world"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding with various valid inputs
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"dGVzdA") == b"test"  # without padding
    assert base64_decode(b"dA==") == b"t"
    assert base64_decode(b"dA") == b"t"  # without padding
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8$")
    with pytest.raises(BadData):
        base64_decode("not base64 at all")
    
    # Test round trip with various data
    test_cases = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"test data with spaces",
        b"binary\x00data",
        b"1234567890",
        bytes(range(256)),  # all byte values
    ]
    for case in test_cases:
        encoded = base64_encode(case)
        decoded = base64_decode(encoded)
        assert decoded == case
```


# LLM-generated content at query #87
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with unicode string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with padding
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test handling of missing padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special characters
    encoded_special = base64_encode(b"\x00\x01\x02")
    assert base64_decode(encoded_special) == b"\x00\x01\x02"
    
    # Test invalid data raises BadData exception
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-base64 characters
    try:
        base64_decode("aGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #88
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with text input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with bytes input
    assert base64_decode(b"d29ybGQ=") == b"world"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test single character
    assert base64_decode("YQ==") == b"a"
    
    # Test with special characters in original data
    original = b"\x00\x01\x02\xff"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test URL-safe characters
    assert base64_decode("_-w=") == b"\xff\xec"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with incorrect length that needs padding
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"


# LLM-generated content at query #89
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test URL-safe characters
    encoded = base64_encode(b"test_data_123")
    assert base64_decode(encoded) == b"test_data_123"
    
    # Test with special characters
    assert base64_decode(b"dGVzdF9kYXRh") == b"test_data"
    
    # Test with padding
    assert base64_decode(b"dA==") == b"t"
    assert base64_decode(b"dGU=") == b"te"
    assert base64_decode(b"dGVz") == b"tes"
    
    # Test various byte values
    assert base64_decode(b"AA==") == b"\x00"
    assert base64_decode(b"_w==") == b"\xff"
    
    # Test invalid base64 raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test unicode string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test roundtrip with various data
    test_data = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"test string with spaces",
        b"\x00\x01\x02\xff\xfe",
        b"1234567890",
        b"data_with_underscores",
        b"a" * 1000
    ]
    
    for data in test_data:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Roundtrip failed for {data!r}"
    
    # Test that BadData is raised for corrupted data
    try:
        base64_decode(b"SGVsbG8")  # Missing = but valid
    except BadData:
        assert False, "Should not raise BadData for valid missing padding"
    
    # Test with non-base64 characters
    try:
        base64_decode(b"SGVs\nbG8=")
        assert False, "Should have raised BadData for newline"
    except BadData:
        pass
```


# LLM-generated content at query #90
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test decoding bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding with special characters
    result = base64_decode("dGVzdC13aXRoLXVybC1zYWZlLWNoYXJz")
    assert result == b"test-with-url-safe-chars"
    
    # Test decoding with underscores and dashes (URL-safe characters)
    result = base64_decode("dGVzdC13aXRoX3VuZGVyc2NvcmU=")
    assert result == b"test-with_underscore"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("invalid!!!")
```


# LLM-generated content at query #91
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding empty string
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test decoding with padding
    encoded_padded = base64.b64encode(b"test").decode("ascii")
    assert base64_decode(encoded_padded) == b"test"
    
    # Test decoding bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding with different data types
    assert base64_decode(base64_encode(b"123")) == b"123"
    assert base64_decode(base64_encode(b"data with spaces")) == b"data with spaces"
    
    # Test decoding raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode(b"\xff\xff\xff")
    with pytest.raises(BadData):
        base64_decode("")
```


# LLM-generated content at query #92
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding with special characters
    encoded = base64_encode(b"\x00\x01\x02")
    assert base64_decode(encoded) == b"\x00\x01\x02"
    
    # Test decoding with unicode characters
    encoded = base64_encode("héllo".encode("utf-8"))
    assert base64_decode(encoded) == "héllo".encode("utf-8")
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test invalid base64 with wrong characters
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8$")
    
    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("aGVsbG8\x80") == b"hello"
```


# LLM-generated content at query #93
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decode without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test decode with bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decode empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decode single character
    result = base64_decode("WA==")
    assert result == b"X"
    
    # Test decode with special URL-safe characters
    result = base64_decode("Pj4-Pg==")
    assert result == b">>>"
    
    # Test decode with - and _ characters
    result = base64_decode("_-w=")
    assert result == b"\xff\xff"
    
    # Test decode with non-ASCII characters (should be ignored)
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test too short invalid data
    try:
        base64_decode("a")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test binary data roundtrip
    import os
    binary_data = os.urandom(50)
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
```


# LLM-generated content at query #94
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test with unicode string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test various lengths
    assert base64_decode("YQ==") == b"a"  # 1 byte
    assert base64_decode("YWI=") == b"ab"  # 2 bytes
    assert base64_decode("YWJj") == b"abc"  # 3 bytes
    assert base64_decode("YWJjZA==") == b"abcd"  # 4 bytes
    
    # Test URL-safe characters
    encoded_url = base64_encode(b"hello+world/")
    assert base64_decode(encoded_url) == b"hello+world/"
    
    # Test invalid data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode("aGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored due to errors='ignore')
    assert base64_decode("aGVsbG8=\xff") == b"hello"  # non-ASCII stripped
    
    # Test with unicode string containing non-ASCII
    assert base64_decode("aGVsbG8=\u00e9") == b"hello"  # é stripped
```


# LLM-generated content at query #95
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    encoded_padded = b"aGVsbG8="
    assert base64_decode(encoded_padded) == b"hello"
    
    # Test with no padding
    encoded_no_pad = b"aGVsbG8"
    assert base64_decode(encoded_no_pad) == b"hello"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test various lengths
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    assert base64_decode(b"YWJjZA==") == b"abcd"
    
    # Test with special characters
    assert base64_decode(b"Lg==") == b"."
    assert base64_decode(b"Lw==") == b"/"
    assert base64_decode(b"Kw==") == b"+"
    
    # Test invalid input raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode(b"aGVs?bG8=")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test roundtrip with various data
    test_data = [
        b"",
        b"\x00",
        b"\xff\xff",
        b"test" * 100,
        bytes(range(256)),
    ]
    for data in test_data:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Roundtrip failed for {data!r}"
```


# LLM-generated content at query #96
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    
    # Test with bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    assert base64_decode(b"V29ybGQ") == b"World"
    
    # Test with special characters (URL-safe)
    assert base64_decode("_-x4") == b"\xff\x1e"
    assert base64_decode("_-x4=") == b"\xff\x1e"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with various padding scenarios
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test error handling - invalid characters
    with pytest.raises(BadData):
        base64_decode("invalid!")
    
    # Test error handling - completely invalid data
    with pytest.raises(BadData):
        base64_decode("not valid base64!!!")
    
    # Test error handling - binary input with invalid bytes
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")
```


# LLM-generated content at query #97
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("d29ybGQ=") == b"world"
    
    # Test decoding without padding
    assert base64_decode("dGVzdA") == b"test"
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test decoding bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding URL-safe base64
    assert base64_decode("dGVzdA") == b"test"
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode("not-base64")
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")
    
    # Test roundtrip with base64_encode
    original = "test data with spaces and symbols!@#$%"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original.encode("utf-8")
    
    # Test with binary data
    binary_data = b"\x00\x01\x02\xff\xfe"
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
    
    # Test with numbers
    for i in range(100):
        encoded = base64_encode(str(i))
        decoded = base64_decode(encoded)
        assert decoded == str(i).encode("utf-8")```


# LLM-generated content at query #98
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with special characters (URL-safe)
    assert base64_decode("_-w=") == b"\xff\xeb"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test with non-base64 characters
    try:
        base64_decode("abc123!@#")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test with incorrect length (should still work due to padding)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
```


# LLM-generated content at query #99
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original, f"Expected {original}, got {decoded}"
    
    # Test with padding removed
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test with full padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with string input (not bytes)
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with special characters used in URL-safe base64
    assert base64_decode(b"_-A") == b"\xff\xe0"
    
    # Test with invalid base64 data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    decoded_binary = base64_decode(encoded_binary)
    assert decoded_binary == binary_data, "Binary data roundtrip failed"
    
    # Test with varying lengths
    for i in range(1, 20):
        test_data = b"x" * i
        encoded = base64_encode(test_data)
        decoded = base64_decode(encoded)
        assert decoded == test_data, f"Failed for length {i}: {decoded}"
```


# LLM-generated content at query #100
#--------------------------

```python
def test_base64_decode():
    # Test normal valid base64 string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with padding
    encoded_with_padding = b"aGVsbG8="
    decoded = base64_decode(encoded_with_padding)
    assert decoded == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with string input
    decoded = base64_decode("aGVsbG8=")
    assert decoded == b"hello"
    
    # Test with special URL-safe characters
    original = b"\xff\xfe\xfd\xfc"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test invalid base64 data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with single character
    encoded = base64_encode(b"a")
    decoded = base64_decode(encoded)
    assert decoded == b"a"
    
    # Test with data that needs padding
    assert base64_decode(b"YQ") == b"a"
    assert base64_decode(b"YWI") == b"ab"
```


# LLM-generated content at query #101
#--------------------------

```python
def test_base64_decode():
    # Test decoding standard URL-safe base64
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decoding with padding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding text input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding with special URL-safe characters
    assert base64_decode(b"PDw_Pz8-Pg==") == b"<<??>>"
    
    # Test raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test raises BadData for non-base64 characters
    with pytest.raises(BadData):
        base64_decode(b"not valid base64")
```


# LLM-generated content at query #102
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"Hello World"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with string input
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with special characters
    special = b"\x00\x01\x02\xff"
    encoded_special = base64_encode(special)
    assert base64_decode(encoded_special) == special
    
    # Test invalid base64 data
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with missing padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
```


# LLM-generated content at query #103
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    
    # Test with bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    assert base64_decode(b"V29ybGQ") == b"World"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
    
    # Test binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
    
    # Test various string inputs
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    
    # Test with special characters
    assert base64_decode("YSBi") == b"a b"
    assert base64_decode("YSBi") == b"a b"
    
    # Test with numbers
    assert base64_decode("MTIz") == b"123"
    assert base64_decode("MTIz") == b"123"
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with non-base64 characters
    with pytest.raises(BadData):
        base64_decode("abc$%^")
    
    # Test with empty padding
    assert base64_decode("") == b""
    
    # Test roundtrip with various sizes
    for i in range(1, 10):
        original = bytes([i * 10] * i)
        encoded = base64_encode(original)
        decoded = base64_decode(encoded)
        assert decoded == original
```


# LLM-generated content at query #104
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test with text input
    encoded = base64_encode("hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test with padding
    encoded = base64_encode(b"a")
    decoded = base64_decode(encoded)
    assert decoded == b"a"

    # Test with empty string
    encoded = base64_encode(b"")
    decoded = base64_decode(encoded)
    assert decoded == b""

    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test with bytes input
    encoded = base64_encode(b"test data")
    decoded = base64_decode(encoded)
    assert decoded == b"test data"

    # Test with special characters
    encoded = base64_encode(b"\x00\x01\x02\xff")
    decoded = base64_decode(encoded)
    assert decoded == b"\x00\x01\x02\xff"
```


# LLM-generated content at query #105
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with special URL-safe characters
    assert base64_decode(b"dGVzdC0xMjM") == b"test-123"
    
    # Test decoding raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test decoding with different encodings
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test decoding longer strings
    original = b"This is a longer test string with multiple characters!"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
```


# LLM-generated content at query #106
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with padding variations
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with URL-safe characters
    assert base64_decode("Pz4_") == b"?>?"
    assert base64_decode("Pz4_") == base64_decode("Pz4/")
    
    # Test with special characters
    assert base64_decode("//8=") == b"\xff\xff"
    assert base64_decode("__8=") == b"\xff\xff"
    
    # Test invalid input raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!")
    with pytest.raises(BadData):
        base64_decode(" ")
    with pytest.raises(BadData):
        base64_decode("abcde")
    with pytest.raises(BadData):
        base64_decode("test@")
    
    # Test various lengths produce correct results
    for i in range(1, 10):
        test_bytes = bytes(range(i))
        encoded = base64.b64encode(test_bytes).rstrip(b"=")
        assert base64_decode(encoded) == test_bytes
    
    # Test with binary data
    binary_data = b"\x00\x01\x02\xff\xfe"
    encoded = base64.b64encode(binary_data).rstrip(b"=")
    assert base64_decode(encoded) == binary_data
    
    # Test that extra padding is handled
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YQ===") == b"a"
```


# LLM-generated content at query #107
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("d29ybGQ=") == b"world"
    
    # Test decoding without padding
    assert base64_decode("aGVsbG8") == b"hello"
    assert base64_decode("d29ybGQ") == b"world"
    
    # Test decoding bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special characters
    assert base64_decode("Pj4+Pg==") == b">>>>"
    assert base64_decode("PDw8PA==") == b"<<<<"
    
    # Test with binary data
    binary = b"\x00\x01\x02\xff\xfe\xfd"
    encoded = base64_encode(binary)
    assert base64_decode(encoded) == binary
    
    # Test with unicode characters (after encoding)
    text = "héllo wörld"
    encoded = base64_encode(text.encode())
    assert base64_decode(encoded) == text.encode()
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("aGVsbG8=" * 2)  # double encoded
    
    # Test with non-ASCII bytes
    assert base64_decode("//8=") == b"\xff\xff"```


# LLM-generated content at query #108
#--------------------------

```python
def test_base64_decode():
    # Test with valid URL-safe base64 encoded string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original, f"Expected {original}, got {decoded}"
    
    # Test with string input instead of bytes
    encoded_str = base64_encode("test string")
    decoded_from_str = base64_decode(encoded_str)
    assert decoded_from_str == b"test string"
    
    # Test padding is handled correctly
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8==") == b"hello"
    
    # Test with empty input
    assert base64_decode(b"") == b""
    
    # Test with URL-safe characters (no + or /)
    test_bytes = bytes(range(256))
    encoded = base64_encode(test_bytes)
    decoded = base64_decode(encoded)
    assert decoded == test_bytes
    
    # Test that invalid input raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with single character
    assert base64_decode(b"YQ==") == b"a"
    
    # Test with short input (no padding needed)
    assert base64_decode(b"YQ") == b"a"
```


# LLM-generated content at query #109
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding raises BadData for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding raises BadData for non-base64 characters
    try:
        base64_decode(b"not@base64!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #110
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("VGVzdA==") == b"Test"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("VGVzdA") == b"Test"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with special URL-safe characters
    assert base64_decode("_-xr") == b"\xff\xeb"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("abc")  # incomplete padding
```


# LLM-generated content at query #111
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding with full padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8==") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding URL-safe characters
    assert base64_decode(b"_-A") == b"\xfb\xff"
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid base64 data with wrong characters
    try:
        base64_decode(b"aGVs\x00bG8=")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test very short valid input
    assert base64_decode(b"AQ") == b"\x01"
    
    # Test roundtrip with various inputs
    test_data = [b"", b"a", b"ab", b"abc", b"abcd", b"\x00\x01\x02", bytes(range(256))]
    for data in test_data:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Roundtrip failed for {data!r}"
```


# LLM-generated content at query #112
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    encoded_str = "aGVsbG8="
    assert base64_decode(encoded_str) == b"hello"
    
    # Test with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with special URL-safe characters
    encoded_url = base64_encode(b"\xff\xfb\x00\x01")
    assert base64_decode(encoded_url) == b"\xff\xfb\x00\x01"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("abc")
```


# LLM-generated content at query #113
#--------------------------

```python
def test_base64_decode():
    # Test basic string decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with bytes input
    encoded = base64_encode(b"test bytes")
    assert base64_decode(encoded) == b"test bytes"
    
    # Test with unicode string input
    encoded = base64_encode("hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test with empty data
    encoded = base64_encode(b"")
    assert base64_decode(encoded) == b""
    
    # Test with special characters
    data = b"data with \x00 null byte"
    encoded = base64_encode(data)
    assert base64_decode(encoded) == data
    
    # Test with invalid base64 data
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with invalid padding
    with pytest.raises(BadData):
        base64_decode("YWJjZA")  # Invalid base64
    
    # Test decoding pre-encoded URL-safe base64
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8=") == b"hello"
```


# LLM-generated content at query #114
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid URL-safe base64 string
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    
    # Test decoding with bytes input
    assert base64_decode(b"V29ybGQ") == b"World"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with special URL-safe characters
    assert base64_decode("_-x") == b"\xfb\xd7"
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters in input (should be ignored)
    assert base64_decode("SGVsbG8\x80") == b"Hello"
```


# LLM-generated content at query #115
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test without padding
    assert base64_decode(b"SGVsbG8gV29ybGQ") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with special URL-safe characters
    assert base64_decode(b"Pj4-Pg==") == b">>>"
    assert base64_decode(b"Pj4-Pg") == b">>>"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode("not base64 at all")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input
    result = base64_decode("cHLDpG3DpMOk")  # "prämä" in base64
    assert result == "prämä".encode("utf-8")
```


# LLM-generated content at query #116
#--------------------------

```python
def test_base64_decode():
    # Test basic URL-safe base64 decode
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test string input (no trailing padding)
    assert base64_decode("dGVzdA") == b"test"
    
    # Test longer string
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with special characters
    original = b"\x00\x01\x02\xff"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test that invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with no padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test with extra padding
    assert base64_decode(b"aGVsbG8====") == b"hello"
```


# LLM-generated content at query #117
#--------------------------

```python
def test_base64_decode():
    # Test valid base64 decode
    result = base64_decode("dGVzdA==")
    assert result == b"test"
    
    # Test URL-safe base64 decode (without padding)
    result = base64_decode("dGVzdA")
    assert result == b"test"
    
    # Test with bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test with empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with special characters
    result = base64_decode("dGVzdC11cmw")
    assert result == b"test-url"
    
    # Test with integer-like content
    result = base64_decode("MA==")
    assert result == b"0"
    
    # Test raises BadData on invalid input
    import pytest
    from .exc import BadData
    
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("not-base64!")
    
    # Test with non-ASCII string input (should be ignored)
    result = base64_decode("dGVzdA==\x80")
    assert result == b"test"
```


# LLM-generated content at query #118
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test decoding bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding with special URL-safe characters
    result = base64_decode("_-w=")
    assert len(result) > 0
    assert isinstance(result, bytes)
    
    # Test decoding with multiple padding bytes
    result = base64_decode("YQ==")
    assert result == b"a"
    
    # Test that BadData is raised for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test that BadData is raised for non-base64 characters
    with pytest.raises(BadData):
        base64_decode("not base64!")
```


# LLM-generated content at query #119
#--------------------------

```python
def test_base64_decode():
    # Test valid base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decode with string input
    encoded_str = base64_encode("test string").decode("ascii")
    decoded = base64_decode(encoded_str)
    assert decoded == b"test string"
    
    # Test padding edge cases
    assert base64_decode(b"YQ") == b"a"  # 1 padding byte needed
    assert base64_decode(b"YWI") == b"ab"  # 2 padding bytes needed
    assert base64_decode(b"") == b""  # Empty string
    
    # Test invalid base64 raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid base64 with non-ascii characters
    try:
        base64_decode("你好")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test round trip with various data
    test_cases = [
        b"",
        b"\x00",
        b"\xff" * 10,
        b"Hello, World!",
        bytes(range(256)),
    ]
    for data in test_cases:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Failed round trip for {data!r}"
    
    # Test that decode handles bytes with extra padding
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
```


# LLM-generated content at query #120
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test without padding
    assert base64_decode("dGVzdA") == b"test"
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test with special URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>?>"
    
    # Test BadData exception
    try:
        base64_decode("invalid!!!")
        assert False, "Should raise BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("dGVzdA==\x80") == b"test"
```


# LLM-generated content at query #121
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test decoding with string input
    encoded_str = base64_encode("test string")
    assert base64_decode(encoded_str) == b"test string"
    
    # Test decoding binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
    
    # Test decoding raises BadData for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding with special characters
    special_chars = b"\x00\xff\x7f\x80"
    encoded_special = base64_encode(special_chars)
    assert base64_decode(encoded_special) == special_chars
    
    # Test decoding text that contains only valid base64 chars but is invalid
    try:
        base64_decode(b"AAAA!AAA")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding with 'ignore' errors for ASCII encoding
    non_ascii = "héllo"
    # base64_encode will encode the UTF-8 bytes, but base64_decode with 'ignore' 
    # should handle non-ASCII string input by ignoring non-ASCII bytes
    result = base64_decode("aMOp")
    assert result == b"h\xc3\xa9"  # UTF-8 encoding of 'é'
```


# LLM-generated content at query #122
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode(b"dGVzdA") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    
    # Test with padding
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    
    # Test empty input
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
    
    # Test binary data
    binary_data = b"\x00\x01\x02\xff"
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with special characters
    assert base64_decode("_-w") == b"\xfb\xc0"
    
    # Test with various lengths
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    assert base64_decode("YWJjZA") == b"abcd"
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!")
    with pytest.raises(BadData):
        base64_decode(b"\xff\xff\xff\xff")
    with pytest.raises(BadData):
        base64_decode("not valid base64!")
```


# LLM-generated content at query #123
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Without padding
    
    # Test with string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test with empty input
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with special characters
    encoded = base64.b64encode(b"\x00\x01\x02\xff\xfe").rstrip(b"=")
    assert base64_decode(encoded) == b"\x00\x01\x02\xff\xfe"
    
    # Test with padding
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    
    # Test that invalid input raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\xff\xff\xff\xff")
```


# LLM-generated content at query #124
#--------------------------

```python
def test_base64_decode():
    # Test with valid base64 encoded string
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test with bytes input
    result = base64_decode(b"V29ybGQ=")
    assert result == b"World"
    
    # Test with empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with URL-safe characters
    result = base64_decode("dGVzdC11cmw")
    assert result == b"test-url"
    
    # Test with special characters
    result = base64_decode("dGVzdF9zdHJpbmc=")
    assert result == b"test_string"
    
    # Test with invalid base64 data should raise BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    result = base64_decode("SGVsbG8=" + "ÿ")
    assert result == b"Hello"
```


# LLM-generated content at query #125
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"d29ybGQ") == b"world"
    assert base64_decode(b"") == b""
    
    # Test with string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("d29ybGQ") == b"world"
    
    # Test with special URL-safe characters
    assert base64_decode(b"_-x.") == b"\xff\xeb"
    assert base64_decode(b"_-w.") == b"\xff\xeb"
    
    # Test with various padding
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    
    # Test with non-ASCII characters
    encoded = base64_encode("héllo")
    assert base64_decode(encoded) == "héllo".encode("utf-8")
    
    # Test invalid base64 data raises BadData
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with invalid characters
    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8=" + b"\x00")
```


# LLM-generated content at query #126
#--------------------------

```python
def test_base64_decode():
    # Test basic string decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with bytes input
    encoded = base64_encode(b"test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test with text input (string)
    encoded = base64_encode("text input")
    assert base64_decode(encoded) == b"text input"
    
    # Test with empty data
    encoded = base64_encode(b"")
    assert base64_decode(encoded) == b""
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with special characters
    special = b"\x00\xff\x7f\x80"
    encoded = base64_encode(special)
    assert base64_decode(encoded) == special
    
    # Test with padding
    # SHA256 hash bytes (32 bytes)
    import hashlib
    hash_bytes = hashlib.sha256(b"test").digest()
    encoded = base64_encode(hash_bytes)
    assert base64_decode(encoded) == hash_bytes
    
    # Test that it raises BadData for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with invalid characters
    try:
        base64_decode(b"invalid base64!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII text input
    encoded = base64_encode("héllo wörld")
    assert base64_decode(encoded) == "héllo wörld".encode("utf-8")
    
    # Test with unicode text
    encoded = base64_encode("你好世界")
    assert base64_decode(encoded) == "你好世界".encode("utf-8")
```


# LLM-generated content at query #127
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with string input
    decoded_from_str = base64_decode("aGVsbG8gd29ybGQ=")
    assert decoded_from_str == b"hello world"
    
    # Test with bytes input
    decoded_from_bytes = base64_decode(b"aGVsbG8gd29ybGQ=")
    assert decoded_from_bytes == b"hello world"
    
    # Test without padding
    decoded_no_pad = base64_decode("aGVsbG8gd29ybGQ")
    assert decoded_no_pad == b"hello world"
    
    # Test empty string
    empty_decoded = base64_decode("")
    assert empty_decoded == b""
    
    # Test with special characters
    special = b"\x00\x01\x02\xff\xfe"
    special_encoded = base64_encode(special)
    special_decoded = base64_decode(special_encoded)
    assert special_decoded == special
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII input (should be ignored)
    non_ascii = base64_decode("aGVsbG8gd29ybGQ=\x80")
    assert non_ascii == b"hello world"


# LLM-generated content at query #128
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with text input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>>"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters
    assert base64_decode("//8=") == b"\xff\xff"
    assert base64_decode("__8=") == b"\xff\xff"
```


# LLM-generated content at query #129
#--------------------------

```python
def test_base64_decode():
    # Test basic ASCII string
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test URL-safe characters
    assert base64_decode("_-x=") == b"\xff\xf1"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test padding with multiple missing characters
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with special characters
    assert base64_decode("AAAA") == b"\x00\x00\x00"
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\x80") == b"Hello"
    
    # Test decode raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!")
    
    with pytest.raises(BadData):
        base64_decode("ABCD\x01\x02")
```


# LLM-generated content at query #130
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with various lengths
    assert base64_decode(base64_encode(b"a")) == b"a"
    assert base64_decode(base64_encode(b"ab")) == b"ab"
    assert base64_decode(base64_encode(b"abc")) == b"abc"
    assert base64_decode(base64_encode(b"abcd")) == b"abcd"
    
    # Test with empty input
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
    
    # Test with special characters
    special = b"\x00\x01\x02\xff\xfe\xfd"
    encoded = base64_encode(special)
    decoded = base64_decode(encoded)
    assert decoded == special
    
    # Test invalid input raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode(b"\xff\xff\xff\xff")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes that have non-ASCII characters
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test URL-safe characters
    assert base64_decode(b"Pj4-Pj4-") == b">>>>"
```


# LLM-generated content at query #131
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decode with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decode without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test decode with ASCII string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decode with special URL-safe characters
    encoded_url = base64_encode(b"\xff\xfb")
    assert base64_decode(encoded_url) == b"\xff\xfb"
    
    # Test decode with binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test that non-base64 characters raise BadData
    with pytest.raises(BadData):
        base64_decode(b"abc123!@#")
```


# LLM-generated content at query #132
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 string
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test with URL-safe characters
    result = base64_decode("Pj4_Pz8")
    assert result == b">>???\x00"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test single character
    result = base64_decode("WA==")
    assert result == b"X"
    
    # Test bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test with special characters
    result = base64_decode("w7zDtsO4w6I=")
    assert result == "üöäß".encode("utf-8")
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters in string (should be ignored)
    result = base64_decode("dGVzdA==\x80")
    assert result == b"test"
```


# LLM-generated content at query #133
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    assert base64_decode(b"SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test with padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with different padding lengths
    assert base64_decode(b"SGVsbG8gV29ybGQh") == b"Hello World!"
    
    # Test with string input
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test empty input
    assert base64_decode(b"") == b""
    
    # Test with special URL-safe characters
    assert base64_decode(b"_-4") == b"\xfb\xef"  # Test with - and _
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input
    try:
        base64_decode(b"\xff\xfe")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #134
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test string")
    decoded = base64_decode(encoded)
    assert decoded == b"test string"
    
    # Test with empty string
    encoded = base64_encode(b"")
    decoded = base64_decode(encoded)
    assert decoded == b""
    
    # Test with special characters
    encoded = base64_encode(b"\x00\x01\x02\xff")
    decoded = base64_decode(encoded)
    assert decoded == b"\x00\x01\x02\xff"
    
    # Test with padding
    encoded = base64_encode(b"a")
    decoded = base64_decode(encoded)
    assert decoded == b"a"
    
    # Test with already padded input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test that BadData is raised for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII bytes input
    assert base64_decode(b"\xff\xfe\xff\xfd") == b"\xff\xfe\xff\xfd"
```


# LLM-generated content at query #135
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    result = base64_decode("dGVzdA==")
    assert result == b"test", f"Expected b'test', got {result}"
    
    # Test decoding without padding
    result = base64_decode("dGVzdA")
    assert result == b"test", f"Expected b'test', got {result}"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b"", f"Expected b'', got {result}"
    
    # Test decoding with bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test", f"Expected b'test', got {result}"
    
    # Test decoding URL-safe characters
    result = base64_decode("_-w=")
    assert result == b"\xff\xec", f"Expected b'\\xff\\xec', got {result}"
    
    # Test decoding raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test decoding raises BadData for non-base64 characters
    with pytest.raises(BadData):
        base64_decode("test\x00")
```


# LLM-generated content at query #136
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVs") == b"hel"
    
    # Test with URL-safe characters
    assert base64_decode(b"aGVsbG8_d29ybGQ=") == b"hello?world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with various lengths
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    assert base64_decode(b"YWJjZA==") == b"abcd"
    
    # Test with special characters
    original = bytes(range(256))
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test BadData exception for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8")  # Missing proper padding
    
    # Test with non-ASCII bytes in string argument
    encoded_bytes = base64_encode("héllo")
    assert base64_decode(encoded_bytes) == "héllo".encode("utf-8")
```


# LLM-generated content at query #137
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("VGVzdA==") == b"Test"
    assert base64_decode("VGVzdA") == b"Test"  # Without padding
    
    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with special URL-safe characters
    assert base64_decode("aGVsbG8_d29ybGQ=") == b"hello?world"
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test that non-base64 characters raise BadData
    with pytest.raises(BadData):
        base64_decode("Hello World")
    
    # Test that empty bytes input works
    assert base64_decode(b"") == b""
```


# LLM-generated content at query #138
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test with string input
    encoded = base64_encode("test data")
    decoded = base64_decode(encoded)
    assert decoded == b"test data"

    # Test with special characters
    encoded = base64_encode(b"\x00\x01\x02\xff")
    decoded = base64_decode(encoded)
    assert decoded == b"\x00\x01\x02\xff"

    # Test with padding
    result = base64_decode(b"aGVsbG8=")
    assert result == b"hello"

    # Test with no padding
    result = base64_decode(b"aGVsbG8")
    assert result == b"hello"

    # Test with URL-safe characters
    result = base64_decode(b"Pj4_Pz8-")
    assert result == b">>???\xff"

    # Test with invalid data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test with empty string
    result = base64_decode(b"")
    assert result == b""

    # Test with single character
    result = base64_decode(base64_encode(b"a"))
    assert result == b"a"

    # Test with long data
    long_data = b"x" * 1000
    encoded = base64_encode(long_data)
    decoded = base64_decode(encoded)
    assert decoded == long_data
```


# LLM-generated content at query #139
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
    
    # Test with URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    
    # Test various inputs
    test_cases = [
        (b"test", b"dGVzdA=="),
        (b"a", b"YQ=="),
        (b"ab", b"YWI="),
        (b"abc", b"YWJj"),
        (bytes(range(256)), base64_encode(bytes(range(256)))),
    ]
    for original, expected_encoded in test_cases:
        assert base64_decode(expected_encoded) == original
    
    # Test with bytes-like object that has different encoding
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with invalid characters
    with pytest.raises(BadData):
        base64_decode("invalid!")
    
    # Test with empty bytes
    assert base64_decode(b"") == b""
    
    # Test with unicode string that is valid base64
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test with extra padding
    assert base64_decode(b"dGVzdA====") == b"test"
```


# LLM-generated content at query #140
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test with string input
    encoded_str = base64_encode("hello world")
    assert base64_decode(encoded_str) == b"hello world"
    
    # Test with padding
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test without padding
    assert base64_decode("dGVzdA") == b"test"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special characters
    assert base64_decode("aGVsbG8t") == b"hello-"
    assert base64_decode("aGVsbG9f") == b"hello_"
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test invalid data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-base64 characters
    try:
        base64_decode("test\x00data")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #141
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 encoded string
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test URL-safe characters
    result = base64_decode("aGVsbG8-d29ybGQ")
    assert result == b"hello>world"
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8=")
    assert result == b"Hello"
    
    # Test with special characters
    result = base64_decode("dGVzdGluZw==")
    assert result == b"testing"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test non-base64 characters
    with pytest.raises(BadData):
        base64_decode("hello world")  # space is not valid
```


# LLM-generated content at query #142
#--------------------------

```python
def test_base64_decode():
    # Test normal decode
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test decode with string input
    encoded = base64_encode("test data")
    decoded = base64_decode(encoded)
    assert decoded == b"test data"
    
    # Test decode with padding
    encoded = base64_encode(b"a")
    decoded = base64_decode(encoded)
    assert decoded == b"a"
    
    # Test decode with multiple padding
    encoded = base64_encode(b"ab")
    decoded = base64_decode(encoded)
    assert decoded == b"ab"
    
    # Test decode with special characters
    encoded = base64_encode(b"hello\nworld")
    decoded = base64_decode(encoded)
    assert decoded == b"hello\nworld"
    
    # Test decode with binary data
    encoded = base64_encode(b"\x00\x01\x02\xff")
    decoded = base64_decode(encoded)
    assert decoded == b"\x00\x01\x02\xff"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decode with empty string
    encoded = base64_encode(b"")
    decoded = base64_decode(encoded)
    assert decoded == b""
    
    # Test decode with bytes input
    encoded = base64_encode(b"test")
    decoded = base64_decode(encoded)
    assert decoded == b"test"
```


# LLM-generated content at query #143
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 string
    result = base64_decode("dGVzdA==")
    assert result == b"test"
    
    # Test without padding
    result = base64_decode("dGVzdA")
    assert result == b"test"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"aGVsbG8=")
    assert result == b"hello"
    
    # Test with URL-safe characters
    result = base64_decode("dGVzdC11cmw=")
    assert result == b"test-url"
    
    # Test binary data
    result = base64_decode("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=")
    assert result == bytes(range(32))
    
    # Test invalid characters should raise BadData
    try:
        base64_decode("invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid base64 (wrong length)
    try:
        base64_decode("a")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with unicode characters (should be ignored)
    result = base64_decode("dGVzdA\x80")
    assert result == b"test"
```


# LLM-generated content at query #144
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with padding
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test decoding without padding
    result = base64_decode("aGVsbG8")
    assert result == b"hello"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding with bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test decoding with special characters
    result = base64_decode("_-w=")
    assert result == b"\xff\xc0"
    
    # Test invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding with ASCII characters
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
```


# LLM-generated content at query #145
#--------------------------

```python
def test_base64_decode():
    # Test basic URL-safe base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test edge cases
    assert base64_decode(b"") == b""
    assert base64_decode(b"AA==") == b"\x00"
    
    # Test with special characters
    data = bytes(range(256))
    encoded = base64_encode(data)
    assert base64_decode(encoded) == data
    
    # Test that invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with non-ASCII bytes (should be ignored due to errors="ignore")
    invalid_bytes = b"\xff\xfe\xfd"
    with pytest.raises(BadData):
        base64_decode(invalid_bytes)
```


# LLM-generated content at query #146
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    result = base64_decode(encoded)
    assert result == b"hello world"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test decoding with string input
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test decoding various lengths
    assert base64_decode(b"YQ==") == b"a"  # 1 byte
    assert base64_decode(b"YWI=") == b"ab"  # 2 bytes
    assert base64_decode(b"YWJj") == b"abc"  # 3 bytes
    assert base64_decode(b"YWJjZA==") == b"abcd"  # 4 bytes
    
    # Test raising BadData on invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test raising BadData on invalid characters
    try:
        base64_decode(b"aGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test round-trip with various data
    test_data = [b"", b"a", b"ab", b"abc", b"abcd", b"test data", b"12345", b"\x00\x01\x02"]
    for data in test_data:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Round-trip failed for {data!r}"```


# LLM-generated content at query #147
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test with padding
    original = b"test"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test with empty string
    decoded = base64_decode(b"")
    assert decoded == b""

    # Test with string input
    decoded = base64_decode("aGVsbG8=")
    assert decoded == b"hello"

    # Test with bytes input (no padding)
    decoded = base64_decode(b"aGVsbG8")
    assert decoded == b"hello"

    # Test with special characters (URL-safe)
    original = b"\xff\xfe\x00"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")

    # Test with non-ASCII characters (should be ignored)
    decoded = base64_decode("aGVsbG8=\xff")
    assert decoded == b"hello"
```


# LLM-generated content at query #148
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test with padding
    encoded_with_padding = b"aGVsbG8="
    decoded = base64_decode(encoded_with_padding)
    assert decoded == b"hello"

    # Test with URL-safe characters
    encoded_url = b"aGVsbG8gd29ybGQ"
    decoded = base64_decode(encoded_url)
    assert decoded == b"hello world"

    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""

    # Test with string input
    decoded = base64_decode("aGVsbG8=")
    assert decoded == b"hello"

    # Test with special characters
    original = b"\x00\x01\x02\xff"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8=" + b"\x00")

    # Test padding calculation
    assert base64_decode(b"YQ") == b"a"  # "a" encoded
    assert base64_decode(b"YWI") == b"ab"  # "ab" encoded
    assert base64_decode(b"YWJj") == b"abc"  # "abc" encoded

    # Test with ascii encoding
    decoded = base64_decode("aGVsbG8=")
    assert decoded == b"hello"
```


# LLM-generated content at query #149
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test empty string
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test various lengths
    test_strings = [b"a", b"ab", b"abc", b"abcd", b"test data here"]
    for test_str in test_strings:
        assert base64_decode(base64_encode(test_str)) == test_str
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with special characters
    special = b"test!@#$%^&*()"
    assert base64_decode(base64_encode(special)) == special
    
    # Test with binary data
    binary = bytes(range(256))
    assert base64_decode(base64_encode(binary)) == binary
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with padding issues
    assert base64_decode("YQ") == b"a"  # Missing padding
    assert base64_decode("YWI") == b"ab"  # Missing padding
    
    # Test with ascii encoding errors (non-ASCII characters should be ignored)
    assert base64_decode("aGVsbG8=\x80") == b"hello"  # Non-ASCII should be ignored
    
    # Test empty string input
    assert base64_decode("") == b""
```


# LLM-generated content at query #150
#--------------------------

```python
def test_base64_decode():
    # Test with normal URL-safe base64 encoded string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test with padding removed
    encoded_no_padding = b"aGVsbG8gd29ybGQ"  # "hello world" without padding
    decoded = base64_decode(encoded_no_padding)
    assert decoded == b"hello world"

    # Test with empty string
    assert base64_decode(b"") == b""

    # Test with string input
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"

    # Test with special URL-safe characters
    original = b"test data with +/="
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test with invalid base64 data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test with bytes that have incorrect length (not multiple of 4)
    decoded = base64_decode(b"aGVsbG8")  # "hello" encoded, length 6
    assert decoded == b"hello"

    # Test with ASCII encoding errors in input
    assert base64_decode("aGVsbG8gd29ybGQ=\x80") == b"hello world"

    # Test with various integer values encoded
    for num in [0, 1, 255, 65535, 123456789]:
        encoded = int_to_bytes(num)
        decoded = bytes_to_int(encoded)
        assert decoded == num
```


# LLM-generated content at query #151
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test with URL-safe characters
    encoded = base64_encode(b"\xff\xfb\x00")
    assert base64_decode(encoded) == b"\xff\xfb\x00"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with empty input
    assert base64_decode("") == b""
    
    # Test with additional padding
    assert base64_decode(b"aGVsbG8====") == b"hello"
    
    # Test raising BadData for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test raising BadData for wrong characters
    try:
        base64_decode(b"aGVs$G8=")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #152
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    encoded_with_padding = base64.urlsafe_b64encode(b"test").decode()
    assert base64_decode(encoded_with_padding) == b"test"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty input
    assert base64_decode("") == b""
    
    # Test with single character
    assert base64_decode("YQ==") == b"a"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input that would be invalid if decoded as UTF-8
    valid_bytes = base64.urlsafe_b64encode(b"\xff\xfe")
    assert base64_decode(valid_bytes) == b"\xff\xfe"


# LLM-generated content at query #153
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with string input
    decoded_str = base64_decode("aGVsbG8gd29ybGQ=")
    assert decoded_str == b"hello world"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test single character
    assert base64_decode("aA==") == b"h"
    
    # Test padding variations
    assert base64_decode("YQ") == b"a"  # missing padding
    assert base64_decode("YQ=") == b"a"  # partial padding
    
    # Test URL-safe characters (replacing + and / with - and _)
    encoded_safe = base64_encode(b"\xfb\xff\xff\xff")
    assert b"/" not in encoded_safe and b"+" not in encoded_safe
    decoded_safe = base64_decode(encoded_safe)
    assert decoded_safe == b"\xfb\xff\xff\xff"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with special characters
    special = bytes(range(256))
    encoded_special = base64_encode(special)
    decoded_special = base64_decode(encoded_special)
    assert decoded_special == special
    
    # Test that invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("aGVs\tbG8=")  # tab character
    
    # Test with non-ASCII bytes in input (should be ignored due to errors="ignore")
    assert base64_decode("aGVs\x80bG8=") == b"hello"
```


# LLM-generated content at query #154
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special characters
    assert base64_decode("dGVzdC91cmw=") == b"test/url"
    assert base64_decode("dGVzdC91cmw") == b"test/url"
    
    # Test with padding issues
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("abc123!@#")
    
    # Test with non-ASCII characters (should be ignored or raise BadData)
    with pytest.raises(BadData):
        base64_decode("áéíóú")  # Non-ASCII characters
```


# LLM-generated content at query #155
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty input
    assert base64_decode("") == b""
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with special characters
    assert base64_decode("aGVsbG8+") == b"hello>"
    assert base64_decode("aGVsbG8/") == b"hello?"
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x00")
    
    # Test long strings
    long_string = "a" * 1000
    encoded = base64_encode(long_string)
    assert base64_decode(encoded) == long_string.encode()
    
    # Test binary data
    binary_data = b"\x00\x01\x02\xff\xfe\xfd"
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
```


# LLM-generated content at query #156
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("dGVzdA==") == b"test"
    assert base64_decode("dGVzdA") == b"test"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with URL-safe characters
    assert base64_decode("aGVsbG8_d29ybGQ=") == b"hello?world"
    assert base64_decode("aGVsbG8td29ybGQ=") == b"hello-world"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters that should be ignored
    assert base64_decode("SGVs\nbG8=") == b"Hello"  # newline ignored
    
    # Test with non-ASCII characters in string mode
    try:
        base64_decode("é")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #157
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test decode with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test decode empty string
    assert base64_decode(b"") == b""
    
    # Test decode with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decode special characters
    encoded_special = base64_encode(b"\x00\x01\x02\xff")
    assert base64_decode(encoded_special) == b"\x00\x01\x02\xff"
    
    # Test decode raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8\xff")  # invalid character
```


# LLM-generated content at query #158
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 encoded string
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with URL-safe characters
    encoded_url = base64_encode(b"\xff\xfe\xfd\xfc")
    assert base64_decode(encoded_url) == b"\xff\xfe\xfd\xfc"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
```


# LLM-generated content at query #159
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special URL-safe characters
    assert base64_decode("_-w=") == b"\xff\xeb"
    
    # Test invalid input raises BadData
    try:
        base64_decode("!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test very long string
    long_input = base64.b64encode(b"a" * 1000).decode()
    assert base64_decode(long_input) == b"a" * 1000
    
    # Test with non-ASCII characters in input (should be ignored)
    assert base64_decode("SGVsbG8=\x00") == b"Hello"
```


# LLM-generated content at query #160
#--------------------------

```python
def test_base64_decode():
    """Test base64_decode function with various inputs."""
    # Test basic string decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with empty bytes
    encoded = base64_encode(b"")
    decoded = base64_decode(encoded)
    assert decoded == b""
    
    # Test with special characters
    encoded = base64_encode(b"\x00\x01\x02\xff\xfe")
    decoded = base64_decode(encoded)
    assert decoded == b"\x00\x01\x02\xff\xfe"
    
    # Test with padding (when length is not multiple of 4)
    encoded = base64.urlsafe_b64encode(b"test").rstrip(b"=")
    decoded = base64_decode(encoded)
    assert decoded == b"test"
    
    # Test with string input (not bytes)
    encoded = base64_encode("unicode test")
    decoded = base64_decode(encoded.decode("ascii"))
    assert decoded == b"unicode test"
    
    # Test with ascii encoding and errors='ignore'
    invalid_utf8 = b"\xff\xfe\x00\x01"
    encoded = base64_encode(invalid_utf8)
    decoded = base64_decode(encoded.decode("latin-1"))
    assert decoded == invalid_utf8
    
    # Test raising BadData for invalid base64
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test raising BadData for incorrect encoding
    try:
        base64_decode(123)  # type: ignore
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with long string
    long_data = b"A" * 1000
    encoded = base64_encode(long_data)
    decoded = base64_decode(encoded)
    assert decoded == long_data
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
```


# LLM-generated content at query #161
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("VGVzdA==") == b"Test"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("VGVzdA") == b"Test"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with URL-safe characters
    assert base64_decode("_-x") == b"\xff\xeb"
    assert base64_decode("_-x=") == b"\xff\xeb"
    assert base64_decode("_-x==") == b"\xff\xeb"
    
    # Test decoding bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test decoding with special characters
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YQ") == b"a"
    
    # Test decoding longer strings
    assert base64_decode("dGVzdGluZw==") == b"testing"
    assert base64_decode("dGVzdGluZw") == b"testing"
    
    # Test decoding binary data
    binary_data = base64.b64encode(b"\x00\x01\x02\xff").decode()
    assert base64_decode(binary_data) == b"\x00\x01\x02\xff"
```


# LLM-generated content at query #162
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test empty string
    encoded = base64_encode(b"")
    assert base64_decode(encoded) == b""
    
    # Test with special characters
    encoded = base64_encode(b"\x00\x01\x02\xff")
    assert base64_decode(encoded) == b"\x00\x01\x02\xff"
    
    # Test with padding variants
    assert base64_decode(b"aGVsbG8") == b"hello"  # Missing padding
    assert base64_decode(b"aGVsbG8=") == b"hello"  # Single padding
    assert base64_decode(b"aGVsbG8==") == b"hello"  # Double padding
    
    # Test with bytes object
    encoded_bytes = base64.b64encode(b"test").rstrip(b"=")
    assert base64_decode(encoded_bytes) == b"test"
    
    # Test raises BadData for invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test raises BadData for non-base64 characters
    try:
        base64_decode(b"hello world")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with mixed case
    encoded = base64_encode(b"MixedCase123")
    assert base64_decode(encoded) == b"MixedCase123"
    
    # Test with numbers
    encoded = base64_encode(b"1234567890")
    assert base64_decode(encoded) == b"1234567890"
    
    # Test with unicode characters (should fail gracefully)
    try:
        base64_decode("你好")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #163
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test with URL-safe characters
    assert base64_decode("_-w=") == b"\xff\xeb"
    assert base64_decode("_-w") == b"\xff\xeb"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode("==") == b""
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test various lengths
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with special characters
    assert base64_decode("+/8=") == b"\xfb\xff"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("abcde")
```


# LLM-generated content at query #164
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with string input
    assert base64_decode(base64_encode("test data")) == b"test data"
    
    # Test with padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test with full padding
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test single character
    assert base64_decode("YQ==") == b"a"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test URL-safe characters
    assert base64_decode("_-w") == b"\xff\xec"
```


# LLM-generated content at query #165
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test string")
    assert base64_decode(encoded) == b"test string"
    
    # Test empty data
    assert base64_decode("") == b""
    
    # Test padding handling
    assert base64_decode("aGVsbG8") == b"hello"
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8==") == b"hello"
    
    # Test binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test single character
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YQ") == b"a"
    
    # Test special characters
    assert base64_decode("ISFAIyQlXiYqKCk=") == b"!@#$%^&*()"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test unicode string input with non-ASCII characters
    encoded = base64_encode("héllo")
    assert base64_decode(encoded) == "héllo".encode("utf-8")
```


# LLM-generated content at query #166
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original, f"Expected {original}, got {decoded}"
    
    # Test with string input
    original_str = "test string"
    encoded_str = base64_encode(original_str)
    decoded_str = base64_decode(encoded_str)
    assert decoded_str == original_str.encode("utf-8"), f"Expected {original_str.encode()}, got {decoded_str}"
    
    # Test with padding
    original_pad = b"a"
    encoded_pad = base64_encode(original_pad)
    decoded_pad = base64_decode(encoded_pad)
    assert decoded_pad == original_pad, f"Expected {original_pad}, got {decoded_pad}"
    
    # Test empty string
    empty_encoded = base64_encode(b"")
    empty_decoded = base64_decode(empty_encoded)
    assert empty_decoded == b"", f"Expected empty bytes, got {empty_decoded}"
    
    # Test with special characters
    special = b"test+data/with=special_chars"
    encoded_special = base64_encode(special)
    decoded_special = base64_decode(encoded_special)
    assert decoded_special == special, f"Expected {special}, got {decoded_special}"
    
    # Test with invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData exception"
    except BadData:
        pass
    
    # Test with bytes input containing invalid characters
    try:
        base64_decode(b"\xff\xfe\xfd")
        assert False, "Should have raised BadData exception"
    except BadData:
        pass
    
    # Test with bytes input
    encoded_bytes = base64_encode(b"binary data \x00\x01\x02")
    decoded_bytes = base64_decode(encoded_bytes)
    assert decoded_bytes == b"binary data \x00\x01\x02", f"Expected binary data, got {decoded_bytes}"
    
    # Test with various lengths
    for length in range(1, 20):
        test_data = b"x" * length
        encoded = base64_encode(test_data)
        decoded = base64_decode(encoded)
        assert decoded == test_data, f"Failed for length {length}: expected {test_data}, got {decoded}"
    
    # Test that function handles strings with ASCII characters properly
    text = "Hello, World! 123"
    encoded_text = base64_encode(text)
    decoded_text = base64_decode(encoded_text)
    assert decoded_text == text.encode("ascii"), f"Expected {text.encode('ascii')}, got {decoded_text}"
```


# LLM-generated content at query #167
#--------------------------

```python
def test_base64_decode():
    # Test decoding standard URL-safe base64 string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original, f"Expected {original}, got {decoded}"
    
    # Test decoding without padding
    encoded_no_padding = encoded.rstrip(b"=")
    decoded_no_padding = base64_decode(encoded_no_padding)
    assert decoded_no_padding == original, f"Expected {original}, got {decoded_no_padding}"
    
    # Test decoding with string input
    string_input = "aGVsbG8gd29ybGQ"
    decoded_from_string = base64_decode(string_input)
    assert decoded_from_string == original, f"Expected {original}, got {decoded_from_string}"
    
    # Test decoding empty string
    assert base64_decode(b"") == b"", "Expected empty bytes"
    assert base64_decode("") == b"", "Expected empty bytes"
    
    # Test decoding single character
    assert base64_decode(b"WA==") == b"X", "Expected b'X'"
    
    # Test decoding with special URL-safe characters
    original_special = b"\xff\xfe\x00\x01"
    encoded_special = base64_encode(original_special)
    decoded_special = base64_decode(encoded_special)
    assert decoded_special == original_special, f"Expected {original_special}, got {decoded_special}"
    
    # Test that BadData is raised for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8=invalid")
    
    # Test decoding with bytes that have invalid characters ignored
    result = base64_decode(b"aGVsbG8gd29ybGQ\x00\x01\x02")
    assert result == original, f"Expected {original}, got {result}"
```


# LLM-generated content at query #168
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 string
    original = b"Hello, World!"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with padding
    encoded_padded = base64.urlsafe_b64encode(b"test").decode()
    decoded_padded = base64_decode(encoded_padded)
    assert decoded_padded == b"test"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\x80") == b"Hello"
```


# LLM-generated content at query #169
#--------------------------

```python
def test_base64_decode():
    """Test base64_decode function."""
    # Test normal base64 decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with padding
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with URL-safe characters
    assert base64_decode("aGVsbG8+") == b"hello>"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with different encodings
    assert base64_decode("8J+YgQ==") == b"\xf0\x9f\x98\x81"
    
    # Test invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode("aGVs$G8=")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode(b"aGVsbG8=\xc3\xa9") == b"hello"
    
    # Test roundtrip with various inputs
    test_cases = [
        b"",
        b"test",
        b"hello world",
        bytes(range(256)),
        b"data with spaces and symbols!@#$%",
    ]
    for case in test_cases:
        encoded = base64_encode(case)
        decoded = base64_decode(encoded)
        assert decoded == case
```


# LLM-generated content at query #170
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVs") == b"hel"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with special characters
    assert base64_decode(b"dGVzdC0t") == b"test--"
    
    # Test invalid base64 data
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode(b"aGVsbG8\xff") == b"hello"
    
    # Test various valid base64 strings
    test_cases = [
        (b"aGVsbG8=", b"hello"),
        (b"aGVsbG8", b"hello"),
        (b"d29ybGQ=", b"world"),
        (b"d29ybGQ", b"world"),
        (b"", b""),
        (b"YQ==", b"a"),
        (b"YWI=", b"ab"),
        (b"YWJj", b"abc"),
    ]
    
    for encoded, expected in test_cases:
        assert base64_decode(encoded) == expected
    
    # Verify roundtrip with various inputs
    for original in [b"test", b"a", b"ab", b"abc", b"abcd", b"test data with spaces"]:
        encoded = base64_encode(original)
        decoded = base64_decode(encoded)
        assert decoded == original
```


# LLM-generated content at query #171
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with various characters
    assert base64_decode(b"dGVzdGluZw==") == b"testing"
    
    # Test invalid base64 raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode(b"aGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that whitespace is ignored (since errors='ignore')
    result = base64_decode(b"aGVs bG8=")
    assert result == b"hello"
```


# LLM-generated content at query #172
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with URL-safe characters
    assert base64_decode(b"Pj4_Pj4_") == b">>?>"
    
    # Test invalid input raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with non-ASCII characters in input (should be ignored)
    assert base64_decode(b"aGVsbG8=\x80") == b"hello"
    
    # Test various lengths
    assert base64_decode(base64_encode(b"a")) == b"a"
    assert base64_decode(base64_encode(b"ab")) == b"ab"
    assert base64_decode(base64_encode(b"abc")) == b"abc"
    assert base64_decode(base64_encode(b"abcd")) == b"abcd"
    
    # Test with bytes containing null bytes
    original = b"\x00\x01\x02\xff"
    assert base64_decode(base64_encode(original)) == original
```


# LLM-generated content at query #173
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 encoded string
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test with URL-safe characters
    result = base64_decode("aGVsbG8_d29ybGQ=")
    assert result == b"hello?world"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8=")
    assert result == b"Hello"
    
    # Test with special characters
    result = base64_decode("aGVsbG8td29ybGQ=")
    assert result == b"hello-world"
    
    # Test invalid input should raise BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with non-ASCII characters (should be ignored)
    result = base64_decode("SGVsbG8=\x80")
    assert result == b"Hello"


# LLM-generated content at query #174
#--------------------------

```python
def test_base64_decode():
    # Test with standard URL-safe base64 string
    result = base64_decode("dGVzdA==")
    assert result == b"test"
    
    # Test without padding
    result = base64_decode("dGVzdA")
    assert result == b"test"
    
    # Test with bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test with empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with special characters
    result = base64_decode("_-x")
    assert result == b"\xff\xeb"
    
    # Test with invalid characters (should be ignored)
    result = base64_decode("dGVzdA!@#")
    assert result == b"test"
    
    # Test with invalid base64 data (should raise BadData)
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!")
```


# LLM-generated content at query #175
#--------------------------

```python
def test_base64_decode():
    # Test normal string decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test unicode string encoding/decoding
    original = "héllo wörld"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original.encode("utf-8")
    
    # Test with empty bytes
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test with padding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"dGVzdA") == b"test"  # No padding
    
    # Test with various lengths
    test_cases = [b"a", b"ab", b"abc", b"abcd", b"abcde"]
    for case in test_cases:
        encoded = base64_encode(case)
        assert base64_decode(encoded) == case
    
    # Test with binary data including null bytes
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"not base64!")
    
    # Test with string input (bytes should also work)
    assert base64_decode("aGVsbG8=") == b"hello"
```


# LLM-generated content at query #176
#--------------------------

```python
def test_base64_decode():
    """Test base64_decode function."""
    # Test with bytes input
    assert base64_decode(b"SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test with string input
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test with URL-safe encoding (no padding)
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test with empty input
    assert base64_decode(b"") == b""
    
    # Test with ASCII characters
    assert base64_decode("YWJjZGVmZw==") == b"abcdefg"
    
    # Test with numbers
    assert base64_decode("MTIzNDU2") == b"123456"
    
    # Test with special characters
    assert base64_decode("ISFAQCMkJV4mKigp") == b"!@#$%^&*()"
    
    # Test invalid input raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("not valid base64")
```


# LLM-generated content at query #177
#--------------------------

```python
def test_base64_decode():
    # Test normal valid input
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test string")
    decoded = base64_decode(encoded)
    assert decoded == b"test string"
    
    # Test empty string
    encoded = base64_encode(b"")
    decoded = base64_decode(encoded)
    assert decoded == b""
    
    # Test with padding
    encoded = base64_encode(b"a")
    decoded = base64_decode(encoded)
    assert decoded == b"a"
    
    # Test with special characters
    encoded = base64_encode(b"\x00\x01\x02")
    decoded = base64_decode(encoded)
    assert decoded == b"\x00\x01\x02"
    
    # Test with binary data
    encoded = base64_encode(b"\xff\xfe\xfd")
    decoded = base64_decode(encoded)
    assert decoded == b"\xff\xfe\xfd"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with incorrect length (missing padding)
    try:
        base64_decode(b"YWJj")  # "abc" without proper padding
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that it handles bytes and string inputs consistently
    input_str = "test data"
    encoded_bytes = base64_encode(input_str)
    encoded_str = base64_encode(input_str.encode())
    assert base64_decode(encoded_bytes) == base64_decode(encoded_str)
```


# LLM-generated content at query #178
#--------------------------

```python
def test_base64_decode():
    # Test valid base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    padded = b"aGVsbG8="
    assert base64_decode(padded) == b"hello"
    
    # Test without padding
    no_padding = b"aGVsbG8"
    assert base64_decode(no_padding) == b"hello"
    
    # Test with URL-safe characters
    url_safe = b"aGVsbG8td29ybGQ"
    assert base64_decode(url_safe) == b"hello-world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with special URL-safe characters
    special = b"aGVsbG8v_d29ybGQ"
    assert base64_decode(special) == b"hello/ world"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test invalid characters
    with pytest.raises(BadData):
        base64_decode(b"invalid\x00data")
```


# LLM-generated content at query #179
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test decoding with padding
    encoded = base64_encode(b"a")
    assert base64_decode(encoded) == b"a"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding string input
    encoded = base64_encode("test")
    assert base64_decode(encoded) == b"test"
    
    # Test decoding with special characters
    encoded = base64_encode(b"\x00\x01\x02")
    assert base64_decode(encoded) == b"\x00\x01\x02"
    
    # Test decoding raises BadData for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding bytes input with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding long string
    long_str = b"x" * 100
    encoded = base64_encode(long_str)
    assert base64_decode(encoded) == long_str
```


# LLM-generated content at query #180
#--------------------------

```python
def test_base64_decode():
    # Test basic decode
    encoded = base64_encode(b"test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test with string input
    encoded = base64_encode("hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with padding
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test with missing padding
    result = base64_decode(b"dGVzdA")
    assert result == b"test"
    
    # Test empty string
    result = base64_decode(b"")
    assert result == b""
    
    # Test with special URL-safe characters
    result = base64_decode(b"aGVsbG8tX3dvcmxk")
    assert result == b"hello_world"
    
    # Test with bytes containing + or /
    result = base64_decode(b"aGVsbG8rL3dvcmxk")
    assert result == b"hello+/world"
    
    # Test that invalid data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters after encoding
    encoded = base64_encode(b"\xff\xfe\x00\x01")
    result = base64_decode(encoded)
    assert result == b"\xff\xfe\x00\x01"
```


# LLM-generated content at query #181
#--------------------------

```python
def test_base64_decode():
    # Test decoding standard base64 string
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with URL-safe characters
    test_data = bytes(range(256))
    encoded_url_safe = base64_encode(test_data)
    assert base64_decode(encoded_url_safe) == test_data
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode(b"aGVs\xffbG8=")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding single character
    assert base64_decode("Zg==") == b"f"
    
    # Test decoding with varying lengths
    for i in range(1, 100):
        test_bytes = bytes(i)
        encoded = base64_encode(test_bytes)
        assert base64_decode(encoded) == test_bytes
```


# LLM-generated content at query #182
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test decoding with string input
    encoded_str = base64_encode("world")
    assert base64_decode(encoded_str) == b"world"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with special characters
    data = b"\x00\x01\x02\xff\xfe"
    encoded = base64_encode(data)
    assert base64_decode(encoded) == data
    
    # Test raises BadData on invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test raises BadData on invalid characters
    try:
        base64_decode(b"aGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test roundtrip with various data
    test_cases = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"abcd",
        b"test data",
        bytes(range(256)),
        b"\x00" * 10,
    ]
    
    for data in test_cases:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Roundtrip failed for {data!r}"
```


# LLM-generated content at query #183
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    
    # Test with various inputs
    assert base64_decode("") == b""
    assert base64_decode("AAECAwQFBgcI") == b"\x00\x01\x02\x03\x04\x05\x06\x07\x08"
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test with special characters in base64
    decoded = base64_decode("_-x4Zw==")
    assert len(decoded) > 0
    
    # Test that invalid data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that non-base64 characters raise BadData
    try:
        base64_decode("abc def")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with empty string
    assert base64_decode("") == b""


# LLM-generated content at query #184
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("VGVzdA==") == b"Test"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("VGVzdA") == b"Test"
    
    # Test decoding with URL-safe characters
    encoded_url = base64_encode(b"hello+world")
    assert base64_decode(encoded_url) == b"hello+world"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("not-base64")
```


# LLM-generated content at query #185
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test decoding bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding with different characters
    result = base64_decode("dGVzdC11cmwtc2FmZQ==")
    assert result == b"test-url-safe"
    
    # Test decoding raises BadData for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding raises BadData for malformed data
    try:
        base64_decode("abc")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding with ascii encoding error handling
    result = base64_decode("dGVzdA==", encoding="ascii", errors="ignore")
    assert result == b"test"
```


# LLM-generated content at query #186
#--------------------------

```python
def test_base64_decode():
    # Test basic decode
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decode with string input
    encoded_str = base64_encode("test string").decode()
    decoded = base64_decode(encoded_str)
    assert decoded == b"test string"
    
    # Test decode with padding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"  # without padding
    
    # Test decode empty string
    assert base64_decode("") == b""
    
    # Test decode with URL-safe characters
    original = b"\xfb\xff\xff\xff\xff"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decode with special characters
    original = b"\x00\x01\x02\xff\xfe"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decode raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test decode raises BadData for non-base64 characters
    with pytest.raises(BadData):
        base64_decode("aGVsbG8$")  # $ is not valid base64
    
    # Test decode with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test decode preserves binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
```


# LLM-generated content at query #187
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 string
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test with multiple padding
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test invalid base64 data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test non-ASCII characters (should be ignored)
    result = base64_decode("SGVsbG8gV29ybGQ=\x80")
    assert result == b"Hello World"
```


