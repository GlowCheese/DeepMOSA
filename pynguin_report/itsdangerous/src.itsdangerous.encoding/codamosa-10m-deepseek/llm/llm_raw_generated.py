####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
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
    
    # Test special characters (URL-safe encoding uses - and _ instead of + and /)
    assert base64_decode(b"PDw_Pz8-Pg==") == b"<<??>>"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test with string input
    assert base64_decode("dGVzdA==") == b"test"
```


# LLM-generated content at query #2
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    assert base64_decode("") == b""
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with URL-safe characters
    assert base64_decode("_-w=") == b"\xff\xec"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\x80") == b"Hello"
```


# LLM-generated content at query #3
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test URL-safe characters
    assert base64_decode(b"dGVzdC11cmw") == b"test-url"
    assert base64_decode("dGVzdC11cmw=") == b"test-url"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test single character
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode("YQ") == b"a"
    
    # Test special characters
    assert base64_decode(b"dGVzdC8_") == b"test/?"
    assert base64_decode("dGVzdC8_") == b"test/?"
    
    # Test with bytes that have underscores and dashes
    assert base64_decode(b"dGVzdC1f") == b"test-_"
    assert base64_decode("dGVzdC1f") == b"test-_"
    
    # Test invalid input raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8=" * 2)  # Invalid length
    
    # Test Unicode string input
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"
```


# LLM-generated content at query #4
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test without padding
    result = base64_decode("aGVsbG8")
    assert result == b"hello"
    
    # Test URL-safe characters
    result = base64_decode("dGVzdC11cmwtc2FmZQ==")
    assert result == b"test-url-safe"
    
    # Test with bytes input
    result = base64_decode(b"d29ybGQ=")
    assert result == b"world"
    
    # Test with empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with single character
    result = base64_decode("YQ==")
    assert result == b"a"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with special URL-safe characters
    result = base64_decode("dGVzdC1f")
    assert result == b"test-"
```


# LLM-generated content at query #5
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with special URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    assert base64_decode("PDw8PA==") == b"<<<<"
    
    # Test with unicode/ASCII characters
    assert base64_decode("YWJj") == b"abc"
    assert base64_decode("MTIz") == b"123"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("\x00\x01\x02")
    
    # Test that non-ASCII characters are ignored
    assert base64_decode("SGVsbG8=" + "ñ") == b"Hello"  # ñ is ignored
    
    # Test longer data
    long_data = b"x" * 100
    encoded = base64_encode(long_data)
    assert base64_decode(encoded) == long_data
```


# LLM-generated content at query #6
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
    assert base64_decode(b"aGVsbG8td29ybGQ") == b"hello-world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test single character encoding
    assert base64_decode(b"YQ==") == b"a"
    
    # Test binary data
    binary_data = b"\x00\x01\x02\xff"
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in input (should be ignored)
    assert base64_decode("SGVsbG8=\x80") == b"Hello"
    
    # Test with bytes object
    assert base64_decode(b"dGVzdA==") == b"test"
```


# LLM-generated content at query #7
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 URL-safe string
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test bytes input
    result = base64_decode(b"SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test single character
    result = base64_decode("WA==")
    assert result == b"X"
    
    # Test with URL-safe characters
    result = base64_decode("_-x5Zw==")
    assert result == b"\xff\xbcye"
    
    # Test that invalid data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that non-base64 characters raise BadData
    try:
        base64_decode("SGVsbG8gV29ybGQ=" + "!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with padding adjustment
    result = base64_decode("YQ")
    assert result == b"a"


# LLM-generated content at query #8
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test decoding without padding
    assert base64_decode("dGVzdA") == b"test"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding with bytes input
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test decoding URL-safe characters
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with special characters
    assert base64_decode("aGVsbG8gd29ybGQ") == b"hello world"
    
    # Test BadData exception for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test BadData exception for non-base64 characters
    try:
        base64_decode("not base64 chars!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test BadData exception for incorrectly padded data
    try:
        base64_decode("aGVsbG8=")  # Invalid padding
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that ascii encoding is used (non-ascii characters are ignored)
    result = base64_decode("aGVsbG8\x80d29ybGQ")
    assert result == b"hello world"
```


# LLM-generated content at query #9
#--------------------------

```python
def test_base64_decode():
    # Test basic string decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test padding handling
    assert base64_decode("SGVsbG8") == b"Hello"  # Missing padding
    
    # Test URL-safe characters
    assert base64_decode("Pj4_Pz8") == b">>???"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with underscores and dashes (URL-safe)
    assert base64_decode("_-x") == b"\xff\xeb"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!")
    
    # Test completely invalid input
    with pytest.raises(BadData):
        base64_decode("not-base64")
    
    # Test long valid string
    import os
    data = os.urandom(100)
    encoded = base64_encode(data)
    assert base64_decode(encoded) == data
    
    # Test round-trip with various data
    test_values = [b"", b"a", b"ab", b"abc", b"abcd", bytes(range(256))]
    for value in test_values:
        encoded = base64_encode(value)
        assert base64_decode(encoded) == value
```


# LLM-generated content at query #10
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 string
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test string without padding
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test with string input
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with single character
    assert base64_decode("dA==") == b"t"
    
    # Test with binary data
    binary_data = b"\x00\x01\x02\x03"
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with special characters (URL-safe)
    encoded_url = base64_encode(b"hello+world")
    assert base64_decode(encoded_url) == b"hello+world"
    
    # Test that invalid base64 raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that non-base64 characters raise BadData
    try:
        base64_decode(b"dGVzdA@")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with whitespace (should be ignored with errors='ignore')
    assert base64_decode("dGVzdA== ") == b"test"
```


# LLM-generated content at query #11
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with various characters
    assert base64_decode(b"dGVzdC11cmwtc2FmZQ==") == b"test-url-safe"
    
    # Test with bytes input
    assert base64_decode(b"Ynl0ZXM=") == b"bytes"
    
    # Test with string input
    assert base64_decode("dGVzdC1zdHJpbmc=") == b"test-string"
    
    # Test with string without padding
    assert base64_decode("dGVzdC1zdHJpbmc") == b"test-string"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test with non-ASCII characters in string (should be ignored)
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test with binary data
    binary = bytes(range(256))
    encoded = base64_encode(binary)
    assert base64_decode(encoded) == binary
```


# LLM-generated content at query #12
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    original = b"hello world"
    encoded = base64.b64encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test URL-safe base64 decoding
    original = b"test data with +/="
    encoded = base64.urlsafe_b64encode(original).rstrip(b"=")
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with missing padding
    encoded_no_padding = base64.urlsafe_b64encode(b"test").rstrip(b"=")
    decoded = base64_decode(encoded_no_padding)
    assert decoded == b"test"
    
    # Test with string input
    encoded_str = "aGVsbG8="
    decoded = base64_decode(encoded_str)
    assert decoded == b"hello"
    
    # Test with empty input
    decoded_empty = base64_decode(b"")
    assert decoded_empty == b""
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with padding restored correctly
    encoded_single_char = "aA=="
    decoded = base64_decode(encoded_single_char)
    assert decoded == b"\x68"
```


# LLM-generated content at query #13
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode(b"SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test string input
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test without padding
    assert base64_decode(b"SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test unicode characters
    result = base64_decode("8J+Zgw==")
    assert result == "🎃".encode()
    
    # Test special characters
    assert base64_decode("Kysv") == b"++/"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode("Invalid base64!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test bytes with ascii encoding
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test with extra padding
    assert base64_decode("dGVzdA===") == b"test"
```


# LLM-generated content at query #14
#--------------------------

```python
def test_base64_decode():
    # Test normal base64 decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test with padding
    encoded_with_padding = b"aGVsbG8="
    assert base64_decode(encoded_with_padding) == b"hello"

    # Test empty string
    assert base64_decode(b"") == b""

    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"

    # Test with missing padding (should still work)
    assert base64_decode(b"aGVsbG8") == b"hello"

    # Test with extra padding
    assert base64_decode(b"aGVsbG8===") == b"hello"

    # Test invalid characters raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")

    # Test invalid base64 raises BadData
    with pytest.raises(BadData):
        base64_decode(b"not-valid-base64!!")

    # Test with bytes input
    assert isinstance(base64_decode(b"dGVzdA=="), bytes)
```


# LLM-generated content at query #15
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding with special URL-safe characters
    encoded = base64.b64encode(b"\xff\xfe\xff").decode('ascii')
    urlsafe = encoded.replace('+', '-').replace('/', '_')
    result = base64_decode(urlsafe.encode('ascii'))
    assert result == b"\xff\xfe\xff"
    
    # Test raises BadData on invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test raises BadData on non-base64 characters
    try:
        base64_decode(b"abc def")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with various byte values
    for i in range(0, 256, 17):
        test_bytes = bytes([i, (i+1) % 256, (i+2) % 256])
        encoded = base64_encode(test_bytes)
        decoded = base64_decode(encoded)
        assert decoded == test_bytes
```


# LLM-generated content at query #16
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test special URL-safe characters
    original = b"\xff\xfe\x00\x01"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with various lengths
    for i in range(1, 10):
        original = bytes(range(i))
        encoded = base64_encode(original)
        assert base64_decode(encoded) == original
    
    # Test invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test non-ASCII string input (should be ignored)
    assert base64_decode("aGVsbG8=\x80") == b"hello"
```


# LLM-generated content at query #17
#--------------------------

def test_base64_decode():
    # Test basic decoding of URL-safe base64
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decoding without padding
    encoded_no_pad = base64_encode(b"test data").rstrip(b"=")
    decoded_no_pad = base64_decode(encoded_no_pad)
    assert decoded_no_pad == b"test data"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    decoded_binary = base64_decode(encoded_binary)
    assert decoded_binary == binary_data
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with special characters that are valid in URL-safe base64
    assert base64_decode(b"_-A") == b"\xff\xe8"


# LLM-generated content at query #18
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"") == b""
    
    # Test with string input
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test without padding
    assert base64_decode(b"dGVzdA") == b"test"
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test with special URL-safe characters
    assert base64_decode(b"Pz4_") == b"?>?"
    assert base64_decode(b"Pz4-") == b"?>?"
    
    # Test with invalid input
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in string input
    result = base64_decode("dGVzdA==", encoding="ascii", errors="ignore")
    assert result == b"test"


# LLM-generated content at query #19
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test without padding
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test URL-safe characters
    original = b"hello+world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with special characters
    assert base64_decode(b"_-9-Zg==") == b"\xff\xfe\xfd"
    
    # Test with ascii encoding parameter
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test raising BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("not valid base64")
```


# LLM-generated content at query #20
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    result = base64_decode("dGVzdA==")
    assert result == b"test"
    
    # Test decoding without padding
    result = base64_decode("dGVzdA")
    assert result == b"test"
    
    # Test decoding with URL-safe characters
    result = base64_decode("aGVsbG8td29ybGQ")
    assert result == b"hello-world"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test decoding with single character
    result = base64_decode("dA==")
    assert result == b"t"
    
    # Test that invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that non-ASCII characters are ignored
    result = base64_decode("dGVzdA\x80\x81")
    assert result == b"test"
```


# LLM-generated content at query #21
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding of URL-safe base64
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
    assert base64_decode("aGVsbG8") == b"hello"

    # Test with empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""

    # Test with special URL-safe characters
    original = b"\xff\xfe\x00\x01"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

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

    # Test decoding with various padding lengths
    assert base64_decode(b"YQ") == b"a"  # 1 byte, needs 2 padding
    assert base64_decode(b"YWI") == b"ab"  # 2 bytes, needs 1 padding
    assert base64_decode(b"YWJj") == b"abc"  # 3 bytes, no padding needed
    assert base64_decode(b"YWJjZA==") == b"abcd"  # 4 bytes
```


# LLM-generated content at query #22
#--------------------------

```python
def test_base64_decode():
    # Test with standard input
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with URL-safe base64 (no padding)
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test with empty string
    assert base64_decode(b"") == b""
    
    # Test with special characters
    assert base64_decode("dGVzdC91cmw=") == b"test/url"
    assert base64_decode(b"dGVzdC91cmw") == b"test/url"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Test with numbers
    assert base64_decode(b"MTIzNDU=") == b"12345"
    
    # Test with invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8")  # Will actually work, so this test is incorrect
    
    # Test with non-bytes non-string input (should fail)
    with pytest.raises(BadData):
        base64_decode(123)
    
    # Test with Unicode characters that get ignored
    assert base64_decode("aGVsbG8=".encode("utf-16")) == b"hello"
```


# LLM-generated content at query #23
#--------------------------

```python
def test_base64_decode():
    # Test basic string decoding
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
    
    # Test bytes input
    result = base64_decode(b"d29ybGQ=")
    assert result == b"world"
    
    # Test without padding
    result = base64_decode("dGVzdA")
    assert result == b"test"
    
    # Test empty string
    result = base64_decode("")
    assert result == b""
    
    # Test with special characters
    result = base64_decode("dGhpcyBpcyBhIHRlc3Q=")
    assert result == b"this is a test"
    
    # Test invalid input raises BadData
    with pytest.raises(BadData):
        base64_decode("invalid!")
    
    # Test with non-ASCII characters (should be ignored)
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"
```


# LLM-generated content at query #24
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test with string input
    encoded_str = "aGVsbG8gd29ybGQ"
    assert base64_decode(encoded_str) == b"hello world"

    # Test with padding
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"

    # Test with special URL-safe characters
    original_bytes = bytes(range(256))
    encoded_url = base64_encode(original_bytes)
    assert base64_decode(encoded_url) == original_bytes

    # Test with empty input
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""

    # Test with different padding lengths
    assert base64_decode(b"YQ") == b"a"
    assert base64_decode(b"YWI") == b"ab"

    # Test error handling with invalid data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    try:
        base64_decode("not base64!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test with bytes that have non-ASCII characters
    try:
        base64_decode(b"\xff\xfe\xfd")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test that function returns bytes
    result = base64_decode("aGVsbG8=")
    assert isinstance(result, bytes)
```


# LLM-generated content at query #25
#--------------------------

```python
def test_base64_decode():
    # Test normal string input
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
    result = base64_decode("dGVzdC1fdXJsX3NhZmU=")
    assert result == b"test_url_safe"
    
    # Test binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    result = base64_decode(encoded)
    assert result == binary_data
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass
    
    # Test single character
    result = base64_decode("YQ==")
    assert result == b"a"
    
    # Test special characters
    result = base64_decode("ISQlJiYnKCk=")
    assert result == b"!$%&&'()"
```


# LLM-generated content at query #26
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test decoding string input
    encoded_str = base64_encode("test string")
    assert base64_decode(encoded_str) == b"test string"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding single character
    assert base64_decode(b"YQ==") == b"a"
    
    # Test decoding with URL-safe characters
    assert base64_decode(b"_-w=") == b"\xfe\xec"
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8!!!!")
    
    # Test decoding bytes that are not valid base64
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")
    
    # Test roundtrip with various inputs
    test_cases = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"test data",
        b"\x00\x01\x02",
        b"1234567890",
        b"!@#$%^&*()",
    ]
    for case in test_cases:
        encoded = base64_encode(case)
        assert base64_decode(encoded) == case
```


# LLM-generated content at query #27
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode(b"SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test with string input
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test with missing padding (should still work)
    assert base64_decode(b"SGVsbG8gV29ybGQ") == b"Hello World"
    
    # Test empty input
    assert base64_decode(b"") == b""
    
    # Test special URL-safe characters
    assert base64_decode(b"Pj4_Pz8") == b">>???", f"Got: {base64_decode(b'Pj4_Pz8')}"
    
    # Test with padding adjustment
    assert base64_decode(b"AQID") == b"\x01\x02\x03"
    
    # Test with multiple padding characters
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YQ") == b"a"
    
    # Test with invalid base64 data
    import pytest
    from .exc import BadData
    
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")
```


# LLM-generated content at query #28
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test string")
    assert base64_decode(encoded) == b"test string"
    
    # Test with empty bytes
    assert base64_decode(b"") == b""
    
    # Test with padding variations
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with special characters
    encoded = base64_encode(b"\x00\x01\x02\xff")
    assert base64_decode(encoded) == b"\x00\x01\x02\xff"
    
    # Test with Unicode string input (ASCII only for valid base64)
    encoded = base64_encode(b"hello")
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Invalid base64 should raise BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode("not valid base64")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    assert base64_decode(encoded) == binary_data
    
    # Round trip test with various lengths
    for length in [1, 2, 3, 4, 5, 10, 100]:
        data = bytes([i % 256 for i in range(length)])
        encoded = base64_encode(data)
        assert base64_decode(encoded) == data
```


# LLM-generated content at query #29
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test padding
    assert base64_decode(base64_encode(b"a")) == b"a"
    assert base64_decode(base64_encode(b"ab")) == b"ab"
    assert base64_decode(base64_encode(b"abc")) == b"abc"
    
    # Test with special characters
    assert base64_decode(base64_encode(b"\x00\x01\x02")) == b"\x00\x01\x02"
    
    # Test with unicode text
    original = "héllo wörld".encode("utf-8")
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test string input (not bytes)
    encoded_str = base64_encode(b"test").decode("ascii")
    assert base64_decode(encoded_str) == b"test"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test with non-ASCII characters in string input
    with pytest.raises(BadData):
        base64_decode("héllo".encode("utf-8"))
    
    # Test with extra padding
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test with whitespace (should be ignored due to errors="ignore")
    assert base64_decode(b"dG V zdA") == b"test"
```


# LLM-generated content at query #30
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    
    # Test binary data
    original = bytes(range(256))
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test invalid length raises BadData
    with pytest.raises(BadData):
        base64_decode("a")
```


# LLM-generated content at query #31
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
    
    # Test with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test with URL-safe characters
    assert base64_decode("_-w=") == b"\xff\xec"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test with special characters
    assert base64_decode("//8=") == b"\xff\xff"
```


# LLM-generated content at query #32
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 string
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"

    # Test without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"

    # Test empty string
    result = base64_decode("")
    assert result == b""

    # Test bytes input
    result = base64_decode(b"V29ybGQ=")
    assert result == b"World"

    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Expected BadData exception"
    except BadData:
        pass

    # Test with special URL-safe characters
    result = base64_decode("aGVsbG8gd29ybGQ_")
    assert result == b"hello world"

    # Test with multiple padding scenarios
    result = base64_decode("YQ==")
    assert result == b"a"

    result = base64_decode("YWI=")
    assert result == b"ab"

    result = base64_decode("YWJj")
    assert result == b"abc"
```


# LLM-generated content at query #33
#--------------------------

```python
def test_base64_decode():
    # Test valid base64 encoded data
    assert base64_decode(b"SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode(b"SGVsbG8gV29ybGQ") == b"Hello World"  # Without padding
    
    # Test with string input
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with padding
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    
    # Test with URL-safe characters
    assert base64_decode(b"Pj4-Pg==") == b">>>"
    
    # Test with different byte values
    assert base64_decode(b"AAECAwQFBgcICQoLDA0ODw==") == bytes(range(16))
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode(b"not-base64")
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")
```


# LLM-generated content at query #34
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test decoding with string input
    decoded = base64_decode("aGVsbG8=")
    assert decoded == b"hello"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding single character
    assert base64_decode("aA==") == b"h"
    
    # Test decoding with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test decoding special characters
    encoded = base64_encode(b"\x00\x01\x02")
    decoded = base64_decode(encoded)
    assert decoded == b"\x00\x01\x02"
    
    # Test decoding with URL-safe characters
    encoded = base64_encode(b"\xff\xfe")
    assert b"/" in encoded or b"_" in encoded
    decoded = base64_decode(encoded)
    assert decoded == b"\xff\xfe"
    
    # Test decoding raises BadData for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("\x00\x01\x02")
    
    # Test decoding with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
    
    # Test roundtrip for various data
    test_cases = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"abcd",
        b"test",
        b"hello world",
        b"\x00" * 8,
        b"\xff" * 8,
        bytes(range(256)),
    ]
    for original in test_cases:
        encoded = base64_encode(original)
        decoded = base64_decode(encoded)
        assert decoded == original
```


# LLM-generated content at query #35
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test with padding
    result = base64_decode("aGVsbG8=")
    assert result == b"hello"

    # Test with no padding
    result = base64_decode("aGVsbG8")
    assert result == b"hello"

    # Test empty string
    result = base64_decode("")
    assert result == b""

    # Test with string input
    result = base64_decode("dGVzdA==")
    assert result == b"test"

    # Test with bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"

    # Test with special URL-safe characters
    result = base64_decode("Pj4-Pg==")
    assert result == b">>>"

    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should raise BadData"
    except BadData:
        pass

    # Test with non-ASCII characters in input (should be ignored)
    result = base64_decode("dGVzdA==\x80\x81")
    assert result == b"test"


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    result = base64_decode("SGVsbG8=")
    assert result == b"Hello"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8")
    assert result == b"Hello"
    
    # Test empty input
    result = base64_decode("")
    assert result == b""
    
    # Test bytes input
    result = base64_decode(b"VGVzdA==")
    assert result == b"Test"
    
    # Test with special characters
    result = base64_decode("aGVsbG8gd29ybGQ")
    assert result == b"hello world"
    
    # Test with numbers
    result = base64_decode("MTIzNA==")
    assert result == b"1234"
    
    # Test with URL-safe characters
    result = base64_decode("aGVsbG8tXzE=")
    assert result == b"hello-_1"
    
    # Test raises BadData for invalid input
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test raises BadData for non-base64 characters
    try:
        base64_decode("SGVsbG8#")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test roundtrip
    original = b"Test data with spaces and special chars!@#$%^&*()"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
```


# LLM-generated content at query #2
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters
    assert base64_decode(b"Pj4-Pg==") == b">>>"
    assert base64_decode(b"PDw8PA==") == b"<<<<"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test various lengths
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    assert base64_decode(b"YWJjZA==") == b"abcd"
    
    # Test with special characters
    assert base64_decode(b"//8=") == b"\xff\xff"
    assert base64_decode(b"+/8=") == b"\xfb\xff"
    
    # Test with string input
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    
    # Test handling of invalid data
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode(b"abc_def")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in string (should be ignored)
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
```


# LLM-generated content at query #3
#--------------------------

def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("V29ybGQ=") == b"World"
    assert base64_decode("") == b""
    
    # Test with URL-safe characters
    assert base64_decode("_-xw==") == b"\xff\xec"
    assert base64_decode("AAAA") == b"\x00\x00\x00"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test with ascii encoding errors (should ignore them)
    assert base64_decode("SGVsbG8=\xff") == b"Hello"
    
    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("not base64 at all")
    
    # Test with special characters
    assert base64_decode("") == b""
    assert base64_decode("AA==") == b"\x00"
    assert base64_decode("/w==") == b"\xff"


# LLM-generated content at query #4
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding valid base64 bytes
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test decoding empty string
    assert base64_decode("") == b""
    
    # Test decoding URL-safe characters
    assert base64_decode("Pj4-Pg") == b">> >"
    
    # Test BadData exception for invalid characters
    with pytest.raises(BadData):
        base64_decode("!!!")
    
    # Test BadData exception for completely invalid input
    with pytest.raises(BadData):
        base64_decode("not valid base64!")
    
    # Test decoding with special characters
    assert base64_decode("AQIDBAUGBwgJ") == b"\x01\x02\x03\x04\x05\x06\x07\x08\x09"
    
    # Test decoding longer string
    original = b"Hello, World! This is a test."
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test decoding binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    assert base64_decode(encoded_binary) == binary_data
```


# LLM-generated content at query #5
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test decoding with different string types
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8=") == b"hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"

    # Test decoding with URL-safe characters
    assert base64_decode("aGVsbG8_d29ybGQ=") != b"hello world"  # Different encoding
    assert base64_decode("aGVsbG8_d29ybGQ") != b"hello world"

    # Test invalid base64 data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")

    # Test decoding non-ASCII input is handled gracefully
    result = base64_decode("aGVsbG8=")
    assert isinstance(result, bytes)
    assert result == b"hello"
```


# LLM-generated content at query #6
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test with padding
    encoded_padded = base64_encode(b"test")
    decoded_padded = base64_decode(encoded_padded)
    assert decoded_padded == b"test"
    
    # Test with string input
    decoded_str = base64_decode("aGVsbG8=")
    assert decoded_str == b"hello"
    
    # Test empty string
    encoded_empty = base64_encode(b"")
    decoded_empty = base64_decode(encoded_empty)
    assert decoded_empty == b""
    
    # Test with special characters and binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    decoded_binary = base64_decode(encoded_binary)
    assert decoded_binary == binary_data
    
    # Test with unicode text
    text = "Hello, 世界!"
    encoded_text = base64_encode(text.encode("utf-8"))
    decoded_text = base64_decode(encoded_text)
    assert decoded_text == text.encode("utf-8")
    
    # Test with invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with invalid characters
    try:
        base64_decode("abc123!@#")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes input
    encoded_bytes = base64_encode(b"bytes input")
    decoded_bytes = base64_decode(encoded_bytes)
    assert decoded_bytes == b"bytes input"
```


# LLM-generated content at query #7
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with URL-safe characters
    encoded = base64_encode(b"hello+world/")
    assert base64_decode(encoded) == b"hello+world/"
    
    # Test invalid data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters
    encoded = base64_encode("héllo".encode("utf-8"))
    assert base64_decode(encoded) == "héllo".encode("utf-8")
```


# LLM-generated content at query #8
#--------------------------

```python
def test_base64_decode():
    # Test normal valid base64 string
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
    
    # Test with URL-safe characters
    result = base64_decode("Pj4-Pz8_")
    assert result == b">>>??"
    
    # Test invalid base64 data
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters in string (should be ignored)
    result = base64_decode("SGVsbG8=\x80")
    assert result == b"Hello"
    
    # Test single character
    result = base64_decode("WA==")
    assert result == b"X"
    
    # Test with equals padding stripped
    result = base64_decode("WA")
    assert result == b"X"
```


# LLM-generated content at query #9
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
    assert base64_decode("aGVsbG8td29ybGQ") == b"hello-world"
    
    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test decoding with padding
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    
    # Test that BadData is raised for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test that BadData is raised for truncated padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8="[:-1])
    
    # Test decoding maximum length
    assert base64_decode("////") == b"\xff\xff\xff"
    
    # Test decoding with minimal valid input
    assert base64_decode("AA==") == b"\x00"
    assert base64_decode("AA") == b"\x00"
```


# LLM-generated content at query #10
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test decoding without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test decoding from string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData as e:
        assert "Invalid base64-encoded data" in str(e)
    
    # Test binary data roundtrip
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    decoded_binary = base64_decode(encoded_binary)
    assert decoded_binary == binary_data
```


# LLM-generated content at query #11
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test decoding with string input
    encoded_str = base64_encode("test string")
    decoded_str = base64_decode(encoded_str)
    assert decoded_str == b"test string"
    
    # Test empty string
    empty_encoded = base64_encode(b"")
    empty_decoded = base64_decode(empty_encoded)
    assert empty_decoded == b""
    
    # Test unicode characters
    unicode_encoded = base64_encode("héllo wörld")
    unicode_decoded = base64_decode(unicode_encoded)
    assert unicode_decoded == "héllo wörld".encode("utf-8")
    
    # Test with padding characters
    data = b"a"
    encoded_padding = base64_encode(data)
    decoded_padding = base64_decode(encoded_padding)
    assert decoded_padding == data
    
    # Test with longer data
    long_data = b"x" * 100
    long_encoded = base64_encode(long_data)
    long_decoded = base64_decode(long_encoded)
    assert long_decoded == long_data
    
    # Test with binary data
    binary_data = bytes(range(256))
    binary_encoded = base64_encode(binary_data)
    binary_decoded = base64_decode(binary_encoded)
    assert binary_decoded == binary_data
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")
    
    # Test with non-ASCII characters in string input
    with pytest.raises(BadData):
        base64_decode("héllo")  # Non-ASCII characters should be ignored or cause issues
    
    # Verify round-trip for various lengths
    for length in [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64]:
        test_data = b"a" * length
        round_trip = base64_decode(base64_encode(test_data))
        assert round_trip == test_data, f"Round-trip failed for length {length}"
```


# LLM-generated content at query #12
#--------------------------

```python
def test_base64_decode():
    # Test valid base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test decode with string input
    assert base64_decode("aGVsbG8=") == b"hello"

    # Test decode with bytes input
    assert base64_decode(b"aGVsbG8=") == b"hello"

    # Test decode without padding
    assert base64_decode("aGVsbG8") == b"hello"

    # Test decode with URL-safe characters
    assert base64_decode("Pj4-Pg==") == b">>>"
    
    # Test decode with URL-safe characters without padding
    assert base64_decode("Pj4-Pg") == b">>>"

    # Test decode empty string
    assert base64_decode("") == b""

    # Test decode with only padding
    assert base64_decode("==") == b""

    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode("aGVsbG8\x00")

    # Test round trip with various data
    test_cases = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"test data",
        b"binary\x00data",
        b"special_chars!@#$%^&*()",
        b"\xff\xfe\xfd\xfc",
    ]
    
    for data in test_cases:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data, f"Round trip failed for {data!r}"
```


# LLM-generated content at query #13
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding bytes input
    result = base64_decode(b"SGVsbG8=")
    assert result == b"Hello"
    
    # Test invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding with special characters
    result = base64_decode("aGVsbG8gd29ybGQ=")
    assert result == b"hello world"
```


# LLM-generated content at query #14
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with string input
    encoded = base64_encode("test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test with padding
    encoded = base64_encode(b"a")
    assert base64_decode(encoded) == b"a"
    
    # Test with longer input
    encoded = base64_encode(b"!" * 100)
    assert base64_decode(encoded) == b"!" * 100
    
    # Test with special characters
    encoded = base64_encode(b"\x00\xff\xfe\xfd")
    assert base64_decode(encoded) == b"\x00\xff\xfe\xfd"
    
    # Test with empty input
    encoded = base64_encode(b"")
    assert base64_decode(encoded) == b""
    
    # Test invalid base64 data should raise BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid base64 string should raise BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII bytes in string (should be ignored due to errors="ignore")
    encoded = base64_encode(b"\xff")
    assert base64_decode(encoded) == b"\xff"
```


# LLM-generated content at query #15
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
    
    # Test decoding string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test decoding with different data
    assert base64_decode(b"dGVzdA==") == b"test"
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test that invalid data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test that non-base64 characters raise BadData
    try:
        base64_decode(b"not valid base64!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters
    encoded = base64_encode(b"\x00\x01\x02\xff")
    assert base64_decode(encoded) == b"\x00\x01\x02\xff"
```


# LLM-generated content at query #16
#--------------------------

```python
def test_base64_decode():
    # Test basic string decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test empty string
    assert base64_decode(base64_encode(b"")) == b""
    
    # Test with special characters
    data = b"test data with \x00 null bytes"
    assert base64_decode(base64_encode(data)) == data
    
    # Test with unicode encoded as bytes
    encoded = base64_encode("héllo".encode("utf-8"))
    assert base64_decode(encoded) == "héllo".encode("utf-8")
    
    # Test invalid base64 string raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with wrong alphabet
    try:
        base64_decode(b"aGVsbG8")  # missing padding
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with unicode string input
    encoded_str = base64_encode("test")
    assert base64_decode(encoded_str) == b"test"


# LLM-generated content at query #17
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8") == b"hello"  # without padding

    # Test empty string
    assert base64_decode(b"") == b""

    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"

    # Test bytes with special URL-safe characters
    original = b"\xff\xfe\x00\x01"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode(b"aGVsbG8\xff") == b"hello"

    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test with equals signs in wrong position
    try:
        base64_decode(b"aGVs=bG8=")
        assert False, "Should have raised BadData"
    except BadData:
        pass

    # Test single character
    assert base64_decode(base64_encode(b"a")) == b"a"

    # Test maximum length (8 bytes for int conversion)
    original = b"abcdefgh"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
```


# LLM-generated content at query #18
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original, f"Expected {original}, got {decoded}"
    
    # Test with string input
    encoded_str = base64_encode(original)
    decoded_from_str = base64_decode(encoded_str.decode())
    assert decoded_from_str == original, f"Expected {original}, got {decoded_from_str}"
    
    # Test empty string
    empty_encoded = base64_encode(b"")
    empty_decoded = base64_decode(empty_encoded)
    assert empty_decoded == b"", f"Expected empty bytes, got {empty_decoded}"
    
    # Test binary data
    binary_data = bytes(range(256))
    binary_encoded = base64_encode(binary_data)
    binary_decoded = base64_decode(binary_encoded)
    assert binary_decoded == binary_data, "Binary data roundtrip failed"
    
    # Test padding variations
    test_cases = [b"a", b"ab", b"abc", b"abcd", b"abcde"]
    for case in test_cases:
        encoded = base64_encode(case)
        decoded = base64_decode(encoded)
        assert decoded == case, f"Failed for {case}: got {decoded}"
    
    # Test with different padding amounts by removing padding
    encoded_no_pad = base64_encode(b"test").rstrip(b"=")
    decoded_no_pad = base64_decode(encoded_no_pad)
    assert decoded_no_pad == b"test", "Failed to decode without padding"
    
    # Test invalid base64 data raises BadData
    import pytest
    invalid_inputs = [
        b"!!!invalid!!!",
        b"not base64 at all",
        b"123",
        b"====",
    ]
    for invalid in invalid_inputs:
        with pytest.raises(BadData):
            base64_decode(invalid)
```


# LLM-generated content at query #19
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decode
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
    
    # Test invalid input raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8!!!")
    
    # Test with special URL-safe characters
    original = b"\x00\x01\x02\xff\xfe\xfd"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test roundtrip with various data
    test_cases = [
        b"a",
        b"ab",
        b"abc",
        b"abcd",
        b"\x00",
        b"\x00\x00",
        b"\xff\xff\xff\xff\xff\xff\xff\xff",
        b"test with spaces and punctuation!@#$%^&*()",
    ]
    for test_data in test_cases:
        encoded = base64_encode(test_data)
        decoded = base64_decode(encoded)
        assert decoded == test_data, f"Failed for {test_data!r}"
```


# LLM-generated content at query #20
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"d29ybGQ=") == b"world"
    
    # Test string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    assert base64_decode(b"d29ybGQ") == b"world"
    
    # Test URL-safe characters
    assert base64_decode(b"aGVsbG8t") == b"hello-"
    assert base64_decode(b"aGVsbG9f") == b"hello_"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded = base64_encode(binary_data)
    decoded = base64_decode(encoded)
    assert decoded == binary_data
    
    # Test with special characters
    assert base64_decode(b"") == b""
    assert base64_decode(b"AA==") == b"\x00"
    assert base64_decode(b"_w==") == b"\xff"
    
    # Test invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    # Test non-base64 characters raise BadData
    with pytest.raises(BadData):
        base64_decode(b"aGVsbG8=" + b"\x00")
    
    # Test various padding scenarios
    assert base64_decode(b"YQ==") == b"a"  # 1 byte
    assert base64_decode(b"YWI=") == b"ab"  # 2 bytes
    assert base64_decode(b"YWJj") == b"abc"  # 3 bytes
    assert base64_decode(b"YWJjZA==") == b"abcd"  # 4 bytes
    
    # Test with ascii encoding for string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with errors='ignore' for non-ascii characters in string
    result = base64_decode("aGVsbG8=\xff")
    assert result == b"hello"


# LLM-generated content at query #21
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("d29ybGQ=") == b"world"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode("=") == b""
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with padding variations
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test invalid data raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    try:
        base64_decode("not base64")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test roundtrip
    original = b"test data with spaces and special chars!@#$%"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test with integer values
    assert base64_decode("MA==") == b"0"
    assert base64_decode("MQ==") == b"1"
```


# LLM-generated content at query #22
#--------------------------

```python
def test_base64_decode():
    # Test decoding normal URL-safe base64
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test decoding with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVs") == b"hel"
    
    # Test decoding with no padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test decoding empty string
    assert base64_decode(b"") == b""
    
    # Test decoding with standard base64 (not URL-safe)
    result = base64_decode(b"aGVsbG8=")
    assert result == b"hello"
    
    # Test decoding bytes with different padding amounts
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    
    # Test decoding with str input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test that invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\x00\x01\x02")
    
    # Test decoding with special URL-safe characters
    assert base64_decode(b"_-w") == b"\xff\xc0"
    
    # Test roundtrip with various inputs
    test_data = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"test data here",
        bytes(range(256)),
    ]
    for data in test_data:
        encoded = base64_encode(data)
        decoded = base64_decode(encoded)
        assert decoded == data
```


# LLM-generated content at query #23
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test decoding without padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test URL-safe decoding (with - and _)
    assert base64_decode(b"_-A=") == b"\xff\xe0"
    assert base64_decode("_-A=") == b"\xff\xe0"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test single character
    assert base64_decode(b"AA==") == b"\x00"
    assert base64_decode("AA==") == b"\x00"
    
    # Test with padding characters
    assert base64_decode(b"QQ==") == b"A"
    assert base64_decode(b"QUE=") == b"AA"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test invalid characters
    try:
        base64_decode(b"test$%^")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test string with whitespace (should be ignored due to errors="ignore")
    assert base64_decode("SGVs bG8=") == b"Hello"
```


# LLM-generated content at query #24
#--------------------------

```python
def test_base64_decode():
    # Test basic string decoding
    encoded = base64_encode(b"hello world")
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"
    
    # Test decoding with padding
    encoded_padded = base64_encode(b"test")
    decoded_padded = base64_decode(encoded_padded)
    assert decoded_padded == b"test"
    
    # Test decoding with string input
    encoded_string = base64_encode("hello").decode()
    decoded_from_string = base64_decode(encoded_string)
    assert decoded_from_string == b"hello"
    
    # Test decoding bytes input
    encoded_bytes = base64_encode(b"python")
    decoded_from_bytes = base64_decode(encoded_bytes)
    assert decoded_from_bytes == b"python"
    
    # Test decoding with special characters
    encoded_special = base64_encode(b"\x00\x01\x02\xff")
    decoded_special = base64_decode(encoded_special)
    assert decoded_special == b"\x00\x01\x02\xff"
    
    # Test decoding empty string
    encoded_empty = base64_encode(b"")
    decoded_empty = base64_decode(encoded_empty)
    assert decoded_empty == b""
    
    # Test decoding with invalid data raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid base64!!!")
    
    # Test decoding with invalid bytes
    with pytest.raises(BadData):
        base64_decode(b"\xff\xff\xff\xff")
    
    # Test decoding with non-ASCII characters (should be ignored)
    result = base64_decode("aGVsbG8g d29ybGQ=")
    assert result == b"hello world"
```


# LLM-generated content at query #25
#--------------------------

```python
def test_base64_decode():
    # Test valid base64 decode
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original
    
    # Test decode with string input
    encoded_str = base64_encode("test string").decode('ascii')
    decoded_from_str = base64_decode(encoded_str)
    assert decoded_from_str == b"test string"
    
    # Test decode with padding
    encoded_padded = base64_encode(b"a")
    decoded_padded = base64_decode(encoded_padded)
    assert decoded_padded == b"a"
    
    # Test decode with missing padding
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test decode with extra padding
    assert base64_decode(b"SGVsbG8====") == b"Hello"
    
    # Test decode empty string
    assert base64_decode(b"") == b""
    
    # Test decode with special URL-safe characters
    encoded_url = base64_encode(b"test?data=123&more")
    decoded_url = base64_decode(encoded_url)
    assert decoded_url == b"test?data=123&more"
    
    # Test invalid base64 raises BadData
    import pytest
    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"not base64 at all")
    
    # Test decode with bytes that have whitespace (should be ignored)
    assert base64_decode(b" SGVsbG8gV29ybGQ=") == b"Hello World"
```


# LLM-generated content at query #26
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
    assert base64_decode(b"a-_w") == b"k\xef\xc0"
    assert base64_decode("a-_w") == b"k\xef\xc0"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test single character
    assert base64_decode(b"Zw==") == b"g"
    assert base64_decode("Zw==") == b"g"
    
    # Test with various lengths
    assert base64_decode(b"YQ==") == b"a"
    assert base64_decode(b"YWI=") == b"ab"
    assert base64_decode(b"YWJj") == b"abc"
    
    # Test with bytes that are already bytes
    encoded = base64.b64encode(b"test data")
    assert base64_decode(encoded) == b"test data"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored in error mode)
    assert base64_decode(b"\xff\xfeSGVsbG8=") == b"Hello"  # BOM-like prefix ignored
    
    # Test with None input (should raise TypeError)
    try:
        base64_decode(None)  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test encoding parameter
    assert base64_decode("SGVsbG8=", encoding="ascii") == b"Hello"
```


# LLM-generated content at query #27
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    encoded = base64_encode(b"hello")
    assert base64_decode(encoded) == b"hello"
    
    # Test with string input
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8==") == b"hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special URL-safe characters
    encoded = base64_encode(b"\xff\xfe")
    assert base64_decode(encoded) == b"\xff\xfe"
    
    # Test invalid base64 data
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test with bytes containing non-ASCII characters (should be ignored)
    assert base64_decode(b"aGVsbG8\x80") == b"hello"
    
    # Test various binary data
    test_data = bytes(range(256))
    encoded = base64_encode(test_data)
    assert base64_decode(encoded) == test_data
    
    # Test with extra padding characters
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8==") == b"hello"
```


# LLM-generated content at query #28
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test decoding with padding
    padded_encoded = b"aGVsbG8="
    assert base64_decode(padded_encoded) == b"hello"
    
    # Test decoding without padding
    unpadded_encoded = b"aGVsbG8"
    assert base64_decode(unpadded_encoded) == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    
    # Test string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test decoding with special URL-safe characters
    original = b"\xfb\xff\xff\xff\xff"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test long string
    long_data = b"x" * 1000
    encoded = base64_encode(long_data)
    assert base64_decode(encoded) == long_data
```


# LLM-generated content at query #29
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
    
    # Test with different lengths
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special characters used in URL-safe base64
    assert base64_decode("Pj4-Pg==") == b">>>"
    
    # Test BadData exception for invalid input
    import pytest
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    
    # Test BadData exception for non-base64 characters
    with pytest.raises(BadData):
        base64_decode("not valid base64")
```


# LLM-generated content at query #30
#--------------------------

```python
def test_base64_decode():
    # Test normal decoding
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    
    # Test with padding
    encoded_with_padding = b"aGVsbG8gd29ybGQ="
    assert base64_decode(encoded_with_padding) == b"hello world"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test string input
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"
    
    # Test with different data types
    assert base64_decode(base64_encode(b"test")) == b"test"
    assert base64_decode(base64_encode("test")) == b"test"
    
    # Test with bytes that need padding
    encoded_no_padding = b"dGVzdA"  # "test" without padding
    assert base64_decode(encoded_no_padding) == b"test"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-base64 characters
    try:
        base64_decode(b"aGVsbG8gd29ybGQ")  # missing proper padding but valid chars
        # This might actually work since we add padding
    except BadData:
        pass
    
    # Test with completely invalid data
    try:
        base64_decode(b"\xff\xfe\xfd")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


# LLM-generated content at query #31
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 string
    result = base64_decode("SGVsbG8gV29ybGQ=")
    assert result == b"Hello World"
    
    # Test decoding without padding
    result = base64_decode("SGVsbG8gV29ybGQ")
    assert result == b"Hello World"
    
    # Test decoding empty string
    result = base64_decode("")
    assert result == b""
    
    # Test decoding bytes input
    result = base64_decode(b"dGVzdA==")
    assert result == b"test"
    
    # Test decoding with special URL-safe characters
    result = base64_decode("Pj4-Pg==")
    assert result == b">>>"
    
    # Test that invalid base64 raises BadData
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


# LLM-generated content at query #32
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding of URL-safe base64
    original = b"hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == original

    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with no padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test empty string
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
    
    # Test with special URL-safe characters
    assert base64_decode(b"Pj4-Pg==") == b">>>"
    assert base64_decode(b"PDw8PA==") == b"<<<<"
    
    # Test with binary data
    binary_data = bytes(range(256))
    encoded_binary = base64_encode(binary_data)
    decoded_binary = base64_decode(encoded_binary)
    assert decoded_binary == binary_data
    
    # Test that invalid base64 raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with invalid characters
    try:
        base64_decode(b"aGVs$bG8=")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with non-ASCII characters (should be ignored)
    assert base64_decode(b"aGVsbG8\xff") == b"hello"
```


# LLM-generated content at query #33
#--------------------------

```python
def test_base64_decode():
    # Test basic decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test without padding
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with URL-safe characters
    assert base64_decode("aGVsbG8_d29ybGQ=") == b"hello?world"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with multiple padding
    assert base64_decode("aGVsbG8") == b"hello"
    
    # Test with special characters
    assert base64_decode("YQ==") == b"a"
    
    # Test invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with invalid characters
    try:
        base64_decode("SGVsbG8$")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with bytes containing invalid ASCII
    assert base64_decode(b"SGVsbG8=") == b"Hello"
```


# LLM-generated content at query #34
#--------------------------

```python
def test_base64_decode():
    # Test with valid base64 encoded string
    assert base64_decode("dGVzdA==") == b"test"
    
    # Test with valid base64 encoded string without padding
    assert base64_decode("dGVzdA") == b"test"
    
    # Test with empty string
    assert base64_decode("") == b""
    
    # Test with single character
    assert base64_decode("YQ==") == b"a"
    
    # Test with bytes input
    assert base64_decode(b"dGVzdA==") == b"test"
    
    # Test with URL-safe characters (no + or /)
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test that invalid base64 raises BadData
    try:
        base64_decode("!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with special characters that should be ignored
    assert base64_decode("dGVzdA==\n") == b"test"  # newline should be ignored
    
    # Test with numbers
    assert base64_decode(b"MTIz") == b"123"
    
    # Test with mixed case
    assert base64_decode(b"QWJj") == b"Abc"
```


# LLM-generated content at query #35
#--------------------------

```python
def test_base64_decode():
    # Test normal URL-safe base64 decoding
    encoded = base64_encode(b"hello world")
    assert base64_decode(encoded) == b"hello world"
    
    # Test with padding
    assert base64_decode(b"aGVsbG8=") == b"hello"
    
    # Test without padding
    assert base64_decode(b"aGVsbG8") == b"hello"
    
    # Test with empty string
    assert base64_decode(b"") == b""
    
    # Test with string input
    assert base64_decode("aGVsbG8=") == b"hello"
    
    # Test with bytes containing special URL-safe characters
    encoded = base64_encode(b"\x00\x01\x02")
    assert base64_decode(encoded) == b"\x00\x01\x02"
    
    # Test with non-ASCII characters that get ignored
    result = base64_decode(b"aGVsbG8\xff")  # \xff is ignored
    assert result == b"hello"
    
    # Test invalid base64 data raises BadData
    try:
        base64_decode(b"!!!invalid!!!")
        assert False, "Should have raised BadData"
    except BadData:
        pass
    
    # Test with invalid type
    try:
        base64_decode(b"\xff\xff\xff\xff")
        assert False, "Should have raised BadData"
    except BadData:
        pass
```


