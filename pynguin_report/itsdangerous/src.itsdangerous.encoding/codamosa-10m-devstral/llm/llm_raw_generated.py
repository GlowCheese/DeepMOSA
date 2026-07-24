####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"
    assert base64_decode("") == b""

    # Test URL-safe base64 decoding
    assert base64_decode("PGJpZ25hbWU+") == b"<big name>"
    assert base64_decode("PGJpZ25hbWU") == b"<big name>"
    assert base64_decode("YWJjZGVmZ2g=") == b"abcdefgh"

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!\n") == b"Hello"
    assert base64_decode("SGVsbG8?") == b"Hello"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!=")  # Invalid character with padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")  # Invalid character at end
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=SGVsbG8=")  # Invalid length


# LLM-generated content at query #2
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test decoding a valid base64 bytes
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding a URL-safe base64 string
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding a URL-safe base64 bytes
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding an empty string
    assert base64_decode("") == b""

    # Test decoding an empty bytes
    assert base64_decode(b"") == b""

    # Test decoding a string with padding
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test decoding a string without padding
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding a string with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test decoding a string with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()")

    # Test decoding a string with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()")

    # Test decoding a string with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()")

    # Test decoding a string with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()")

    # Test decoding a string with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()")


# LLM-generated content at query #3
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode("") == b""  # Empty string

    # Test URL-safe characters
    assert base64_decode("PGJhc2U2NF90ZXN0Pg==") == b"<base64_test>"
    assert base64_decode("PGJhc2U2NF90ZXN0Pg") == b"<base64_test>"

    # Test with special characters
    assert base64_decode("8J+YgA==") == b"\x10\xff\x00"  # Non-ASCII bytes

    # Test error cases
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")  # Invalid padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")  # Too much padding

    # Test with non-ASCII input that gets ignored
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #4
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG


# LLM-generated content at query #5
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"  # URL-safe characters without padding

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 100 + "!")  # Invalid character at end

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8\u00ff\u00ff") == b"Hello"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8-") == b"Hello-"
    assert base64_decode("SGVsbG8_") == b"Hello_"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"  # Extra padding should be handled


# LLM-generated content at query #6
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # bytes input without padding

    # Test URL-safe characters
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8-")  # - instead of +
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8_")  # _ instead of /

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsbG8=😊") == b"Hello"
    assert base64_decode("SGVsbG8=ñ") == b"Hello"

    # Test with different padding lengths
    assert base64_decode("SGVsbG8") == b"Hello"  # no padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # 1 padding
    assert base64_decode("SGVsbG8==") == b"Hello"  # 2 padding (invalid but handled)


# LLM-generated content at query #7
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # missing padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("") == b""

    # Test URL-safe characters
    assert base64_decode("PGJpZ2ZpbG0+") == b"<bigfilm>"
    assert base64_decode("PGJpZ2ZpbG0") == b"<bigfilm>"
    assert base64_decode("YWJjZGVmZw==") == b"abcdefg"
    assert base64_decode("YWJjZGVmZw") == b"abcdefg"

    # Test with special characters
    assert base64_decode("8J+YgA==") == b"\x10\xff\xee"
    assert base64_decode("8J+YgA") == b"\x10\xff\xee"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8=\u00ff\u00fe") == b"Hello"


# LLM-generated content at query #8
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"

    # Test empty string
    assert base64_decode("") == b""

    # Test URL-safe characters
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q") == b"<bigfoot>"  # Without padding

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=")  # Valid but with invalid padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")  # Too much padding

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #9
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Missing padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("") == b""

    # Test URL-safe base64
    assert base64_decode("PGJyb2FkY2FzdD4=") == b"<broadcast>"
    assert base64_decode("PGJyb2FkY2FzdD4") == b"<broadcast>"

    # Test with padding
    assert base64_decode("YWJj") == b"abc"
    assert base64_decode("YWJjZA==") == b"abcd"
    assert base64_decode("YWJjZGU=") == b"abcde"

    # Test invalid base64
    with pytest.raises(BadData):
        base64_decode("!!!!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"
    assert base64_decode("SGVsbG8=ñ") == b"Hello"

    # Test with whitespace (should be ignored)
    assert base64_decode(" SGVsbG8= ") == b"Hello"
    assert base64_decode("SG Vs bG8=") == b"Hello"


# LLM-generated content at query #10
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 encoded strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8")
    assert base64_decode("SGVsbG8") == base64_decode("SGVsbG8")

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #11
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #12
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Without padding

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("PGJpZCBpZD0iMTAwIj5UZXN0PC9iaWQ+") == b'<bid id="100">Test</bid>'
    assert base64_decode("PGJpZCBpZD0iMTAwIj5UZXN0PC9iaWQ-") == b'<bid id="100">Test</bid>'

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid@Base64!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"
    assert base64_decode(b"SGVsbG8=\xff") == b"Hello"


# LLM-generated content at query #13
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello?"

    # Test empty string
    assert base64_decode("") == b""

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")  # Invalid character


# LLM-generated content at query #14
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding URL-safe base64
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello?"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with incorrect padding (should be handled)
    assert base64_decode("SGVsbG") == b"Hell"


# LLM-generated content at query #15
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding URL-safe base64
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode(b"SGVsbG8!@#") == b"Hello"

    # Test decoding with BadData exception
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 100)  # Invalid length

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8é") == b"Hello"
    assert base64_decode(b"SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #16
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 encoded strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #17
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # bytes input without padding

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8-")  # URL-safe variant
    assert base64_decode("SGVsbG8") == base64_decode("SGVsbG8_")  # URL-safe variant

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #18
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("PGJpZ25hbWU+") == b"<big-name>"
    assert base64_decode("PGJpZ25hbWU") == b"<big-name>"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"
    assert base64_decode(b"SGVsbG8=\xff") == b"Hello"


# LLM-generated content at query #19
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe characters
    assert base64_decode("PGJhcj4=") == b"<bar>"
    assert base64_decode("PGJhcj4") == b"<bar>"
    assert base64_decode("PGJhcj4=") == b"<bar>"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")  # invalid character

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=é") == b"Hello"
    assert base64_decode("SGVsbG8=😊") == b"Hello"


# LLM-generated content at query #20
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # without padding

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8=") == base64.urlsafe_b64decode("SGVsbG8=")
    assert base64_decode("SGVsbG8") == base64.urlsafe_b64decode("SGVsbG8")

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #21
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with special characters
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello"
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode(b"SGVsbG8=\xff") == b"Hello"


# LLM-generated content at query #22
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Without padding

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8-")
    assert base64_decode("SGVsbG8") == base64_decode("SGVsbG8_")

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #23
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("PGJpZj4=") == b"<bif>"
    assert base64_decode("PGJpZj4") == b"<bif>"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid!@#")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"

    # Test decoding with incorrect padding (should still work)
    assert base64_decode("SGVs") == b"Hel"
    assert base64_decode("SGV") == b"He"


# LLM-generated content at query #24
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe characters
    assert base64_decode("PGJhc2U2NGVudGVzdD4=") == b"<base64test>"
    assert base64_decode("PGJhc2U2NGVudGVzdD4") == b"<base64test>"
    assert base64_decode("PGJhc2U2NGVudGVzdD4=") == b"<base64test>"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("invalid!base64")
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"


# LLM-generated content at query #25
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("PGJpZCBieT0iYSI+") == b'<bid by="a">'
    assert base64_decode("PGJpZCBieT0iYSI") == b'<bid by="a">'

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!\n") == b"Hello"
    assert base64_decode(b"SGVsbG8!\n") == b"Hello"

    # Test raising BadData for invalid base64
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")

    with pytest.raises(BadData):
        base64_decode(b"!!!invalid!!!")

    # Test with non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=" + "é") == b"Hello"
    assert base64_decode(b"SGVsbG8=" + "é".encode("utf-8")) == b"Hello"


# LLM-generated content at query #26
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with padding
    assert base64_decode("aGVsbG8=") == b"helloo"
    assert base64_decode("aGVsbG8") == b"helloo"

    # Test decoding with URL-safe characters
    assert base64_decode("PGJvZHk+") == b"<body>"
    assert base64_decode("PGJvZHk") == b"<body>"

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("invalid!base64")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"
    assert base64_decode(b"SGVsbG8=\xff") == b"Hello"


# LLM-generated content at query #27
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding a URL-safe base64 string
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding an empty string
    assert base64_decode("") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("InvalidBase64!")

    # Test decoding with incorrect padding (should be handled)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"


# LLM-generated content at query #28
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"

    # Test decoding a URL-safe base64 string
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test decoding with missing padding (should add it)
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding a bytes input
    assert base64_decode(b"SGVsbG8gd29ybGQ=") == b"Hello world"

    # Test decoding an empty string
    assert base64_decode("") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8gd29ybGQ!@#") == b"Hello world"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid!!")

    # Test decoding with incorrect padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=")


# LLM-generated content at query #29
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("") == b""

    # Test URL-safe characters
    assert base64_decode("PGJpZz5IZWxsbyBXb3JsZCE8L2JpZz4=") == b"<big>Hello World!</big>"
    assert base64_decode("PGJpZz5IZWxsbyBXb3JsZCE8L2JpZz4") == b"<big>Hello World!</big>"

    # Test with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWE=") == b"aa"
    assert base64_decode("YWFh") == b"aaa"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"

    # Test with different padding lengths
    assert base64_decode("QUJD") == b"ABC"
    assert base64_decode("QUJDRA==") == b"ABCD"
    assert base64_decode("QUJDREU=") == b"ABCDE"
    assert base64_decode("QUJDREVG") == b"ABCDEF"


# LLM-generated content at query #30
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Without padding

    # Test URL-safe characters
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8-")
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8_")

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsb\xffG8=") == b"Hello"


# LLM-generated content at query #31
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding

    # Test URL-safe base64
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello"
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #32
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # bytes input
    assert base64_decode("") == b""  # empty string

    # Test URL-safe characters
    assert base64_decode("PGJpZ2ZpbG0+") == b"<bigfilm>"
    assert base64_decode("PGJpZ2ZpbG0") == b"<bigfilm>"
    assert base64_decode("YWJjZGVmZw==") == b"abcdefg"
    assert base64_decode("YWJjZGVmZw") == b"abcdefg"

    # Test with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test with non-ASCII input (should be ignored)
    assert base64_decode("SGVsb\xffG8=") == b"Hell"  # non-ASCII chars ignored


# LLM-generated content at query #33
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode(b"aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"  # without padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"

    # Test URL-safe base64 decoding
    assert base64_decode("aGVsbG8") == b"hello"
    assert base64_decode("aGVsbG8-") == b"hello"
    assert base64_decode("aGVsbG8_") == b"hello"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("invalid!base64")
    with pytest.raises(BadData):
        base64_decode("aGVsbG8!")

    # Test non-ASCII input is handled
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8=\u00ff") == b"hello"  # non-ASCII ignored


# LLM-generated content at query #34
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # without padding

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8= ") == b"Hello"  # with whitespace

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")  # invalid character with padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8= =")  # invalid padding
    with pytest.raises(BadData):
        base64_decode(123)  # invalid type


# LLM-generated content at query #35
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # bytes input
    assert base64_decode("") == b""

    # Test URL-safe characters
    assert base64_decode("SGVsbG8-") == b"Hello"
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test with padding
    assert base64_decode("QUJDRA==") == b"ABCD"
    assert base64_decode("QUJD") == b"ABCD"  # without padding

    # Test invalid base64
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\xff") == b"Hello"

    # Test with extra padding
    assert base64_decode("SGVsbG8=====") == b"Hello"


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe characters
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8=")
    assert base64_decode("SGVsbG8-") == base64_decode("SGVsbG8+")  # URL-safe variant

    # Test empty string
    assert base64_decode("") == b""

    # Test with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=")  # Correct but with invalid padding
    with pytest.raises(BadData):
        base64_decode(123)  # Invalid type

    # Test with non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #2
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Missing padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"  # Missing padding

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    # Test decoding with incorrect padding (should still work)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"  # Extra padding

    # Test URL-safe base64 characters
    assert base64_decode("PGJpZ25hbWU+") == b"<big name>"
    assert base64_decode("PGJpZ25hbWU+") == b"<big name>"


# LLM-generated content at query #3
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Missing padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"  # Missing padding

    # Test URL-safe characters
    assert base64_decode("PGJpZ2ZpbG0+") == b"<bigfilm>"
    assert base64_decode("PGJpZ2ZpbG0") == b"<bigfilm>"  # Missing padding
    assert base64_decode("PGJpZ2ZpbG0-") == b"<bigfilm>"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")

    # Test non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"
    assert base64_decode("SGVsbG8=äöü") == b"Hello"

    # Test with different padding lengths
    assert base64_decode("a") == b"\xed"
    assert base64_decode("ab") == b"\xed\x95"
    assert base64_decode("abc") == b"\xed\x95\x9c"


# LLM-generated content at query #4
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q=") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q==") == b"<bigfoot>"

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode(b"SGVsbG8!@#") == b"Hello"

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8é") == b"Hello"
    assert base64_decode(b"SGVsbG8\xff") == b"Hello"

    # Test decoding with incorrect padding (should be fixed)
    assert base64_decode("SGVsbG") == b"Hell"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 100 + "!")

    # Test decoding with very long string
    long_string = "a" * 1000
    encoded = base64.urlsafe_b64encode(long_string.encode()).rstrip(b"=")
    assert base64_decode(encoded) == long_string.encode()


# LLM-generated content at query #5
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input

    # Test URL-safe characters
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q=") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q==") == b"<bigfoot>"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")  # Invalid character

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"

    # Test with different padding lengths
    assert base64_decode("a") == b"\xab"
    assert base64_decode("ab") == b"\xab\xc0"
    assert base64_decode("abc") == b"\xab\xcd\xef"


# LLM-generated content at query #6
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello"
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #7
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"

    # Test longer strings
    long_string = "VGVzdCBkYXRhIHRoYXQgaXMgYSB0ZXN0IGZvciBib29rZWVwaW5n"
    assert base64_decode(long_string) == b"Test data that is a test for bookkeeping"


# LLM-generated content at query #8
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode("") == b""  # Empty string
    assert base64_decode("Zg==") == b"f"  # Single character
    assert base64_decode("Zm8=") == b"fo"  # Two characters
    assert base64_decode("Zm9v") == b"foo"  # Three characters
    assert base64_decode("Zm9vYg==") == b"foob"  # Four characters
    assert base64_decode("Zm9vYmE=") == b"fooba"  # Five characters
    assert base64_decode("Zm9vYmFy") == b"foobar"  # Six characters

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8")  # Standard vs URL-safe
    assert base64_decode("PGJyPg==") == b"<br>"  # URL-safe characters
    assert base64_decode("PGJyPg") == b"<br>"  # URL-safe without padding

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")  # Invalid character with padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")  # Too much padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG")  # Invalid length
    with pytest.raises(BadData):
        base64_decode(123)  # Invalid type (not str or bytes)

    # Test encoding parameter
    assert base64_decode("4pyT") == b"\xe4\xbd\xa0\xe5\xa5\xbd"  # UTF-8 encoded Chinese
    assert base64_decode("\xc3\xa4\xc3\xb6\xc3\xbc") == b""  # Invalid ASCII (ignored)

    # Test errors parameter
    assert base64_decode("\xc3\xa4\xc3\xb6\xc3\xbc") == b""  # Invalid ASCII (ignored)


# LLM-generated content at query #9
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding URL-safe base64
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello?"

    # Test decoding with extra characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8==!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #10
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # bytes input

    # Test URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello?"
    assert base64_decode("SGVsbG8_") == b"Hello/"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #11
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""

    # Test padding handling
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    assert base64_decode("YWJj") == b"abc"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("YQ#=")

    with pytest.raises(BadData):
        base64_decode(12345)  # type: ignore

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"


# LLM-generated content at query #12
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with special characters
    assert base64_decode("SGVsbG8-") == b"Hello"
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"


# LLM-generated content at query #13
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # missing padding
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"
    assert base64_decode("SGVsbG8-V29ybGQh") == b"Hello+World!"
    assert base64_decode("SGVsbG8_V29ybGQh") == b"Hello/World!"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #14
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Missing padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("") == b""

    # Test URL-safe characters
    assert base64_decode("PGJvZHk+") == b"<body>"
    assert base64_decode("PGJvZHk-") == b"<body>"  # URL-safe variant

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test with non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"


# LLM-generated content at query #15
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("PGJpZ2ZpbG0+") == b"<bigfilm>"
    assert base64_decode("PGJpZ2ZpbG0") == b"<bigfilm>"

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#$") == b"Hello"
    assert base64_decode(b"SGVsbG8\xff\xfe") == b"Hello"

    # Test decoding with incorrect padding (should be handled)
    assert base64_decode("SGVsbG") == b"Hell"
    assert base64_decode("SGVsb") == b"Hel"

    # Test decoding invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # invalid character that can't be ignored
    with pytest.raises(BadData):
        base64_decode("SGVsbG8==")  # incorrect padding length
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")  # invalid character after padding


# LLM-generated content at query #16
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding URL-safe base64
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello?"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode(b"SGVsbG8!@#") == b"Hello"

    # Test raising BadData for invalid base64
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8é") == b"Hello"
    assert base64_decode(b"SGVsbG8\xc3\xa9") == b"Hello"


# LLM-generated content at query #17
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"


# LLM-generated content at query #18
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("") == b""

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe characters
    assert base64_decode("PGJvZHk+") == b"<body>"
    assert base64_decode("PGJvZHk") == b"<body>"
    assert base64_decode("YWJjX2RlZg==") == b"abc_def"
    assert base64_decode("YWJjX2RlZg") == b"abc_def"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=")  # valid but with invalid padding (shouldn't raise)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")  # too much padding

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8=\xff") == b"Hello"


# LLM-generated content at query #19
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # bytes input without padding

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with special characters
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8gd29ybGQh") == b"Hello world!"
    assert base64_decode("SGVsbG8gd29ybGQh") == b"Hello world!"

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"
    assert base64_decode(b"SGVsbG8\xff") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"  # extra padding should be handled


# LLM-generated content at query #20
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("") == b""

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe characters
    assert base64_decode("PGJpZ25hbWU+") == b"<bigname>"
    assert base64_decode("PGJpZ25hbWU") == b"<bigname>"
    assert base64_decode("YWJjZGVmZw==") == b"abcdefg"
    assert base64_decode("YWJjZGVmZw") == b"abcdefg"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")  # Invalid character

    # Test with non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"

    # Test with padding
    assert base64_decode("SGVs") == b"Hel"
    assert base64_decode("SGV") == b"He"
    assert base64_decode("SG") == b"H"


# LLM-generated content at query #21
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello?"  # URL-safe character

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode(b"Invalid!")

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #22
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe characters
    assert base64_decode("PGJhc2U2ND4=") == b"<base64>"
    assert base64_decode("PGJhc2U2ND4") == b"<base64>"
    assert base64_decode("PGJhc2U2ND4=") == b"<base64>"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"


# LLM-generated content at query #23
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe characters
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8-")
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8_")

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode(b"Invalid!")

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode(b"SGVsbG8=\xff") == b"Hello"


# LLM-generated content at query #24
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"
    assert base64_decode("") == b""

    # Test URL-safe characters
    assert base64_decode("PGJpZj4=") == b"<bif>"
    assert base64_decode("PGJpZj4") == b"<bif>"
    assert base64_decode("YWJjZGVmZw==") == b"abcdefg"

    # Test with padding added automatically
    assert base64_decode("YWJj") == b"ab"
    assert base64_decode("YWJjZGU=") == b"abcde"
    assert base64_decode("YWJjZGVm") == b"abcdef"

    # Test error handling
    with pytest.raises(BadData):
        base64_decode("invalid!base64")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=☺") == b"Hello"
    assert base64_decode("aGVsbG8gd29ybGQ=😊") == b"hello world"

    # Test with spaces (should be ignored)
    assert base64_decode(" SGVsbG8= ") == b"Hello"
    assert base64_decode("aGVs bG8g d29y bGQ=") == b"hello world"


# LLM-generated content at query #25
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding URL-safe base64
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello?"
    assert base64_decode("SGVsbG8_") == b"Hello/"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8é") == b"Hello"


# LLM-generated content at query #26
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"  # With special chars

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8gV29ybGQ-") == b"Hello World?"
    assert base64_decode("SGVsbG8gV29ybGQ_") == b"Hello World/"

    # Test empty string
    assert base64_decode("") == b""

    # Test strings with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test BadData exception for invalid base64
    with pytest.raises(BadData):
        base64_decode("Invalid@@@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=Invalid")

    # Test with different encodings (should handle ascii)
    assert base64_decode("SGVsbG8=", encoding="ascii") == b"Hello"


# LLM-generated content at query #27
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")  # Invalid character

    # Test strings with incorrect padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Correctly handles missing padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"


# LLM-generated content at query #28
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("PGJpZ2ZpbG0+") == b"<bigfilm>"
    assert base64_decode("PGJpZ2ZpbG0") == b"<bigfilm>"

    # Test decoding with special characters
    assert base64_decode("8J+YgA==") == b"\xe4\xbd\xa0\xe5\xa5\xbd"  # "你好" in UTF-8

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\xff=") == b"Hello"

    # Test decoding with incorrect padding (should still work)
    assert base64_decode("SGVsbG8====") == b"Hello"


# LLM-generated content at query #29
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe characters
    assert base64_decode("PGJhcj4=") == b"<bar>"
    assert base64_decode("PGJhcj4") == b"<bar>"
    assert base64_decode("PGJhcj4=") == b"<bar>"
    assert base64_decode("PGJhcj4") == b"<bar>"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode(b"Invalid!")

    # Test non-ASCII characters are ignored (as per encoding="ascii", errors="ignore")
    assert base64_decode("SGVsbG8=😊") == b"Hello"
    assert base64_decode(b"SGVsbG8=\xff") == b"Hello"


# LLM-generated content at query #30
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("") == b""

    # Test URL-safe characters
    assert base64_decode("PGJvZHk+") == b"<body>"
    assert base64_decode("PGJvZHk-") == b"<body>"
    assert base64_decode("PGJvZHk_") == b"<body>"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\xff\xfe") == b"Hello"


# LLM-generated content at query #31
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("PGJpZ25hbWU+") == b"<big name>"
    assert base64_decode("PGJpZ25hbWU") == b"<big name>"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsb\xffG8=") == b"Hell"


# LLM-generated content at query #32
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding URL-safe base64
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello?"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!\n") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with incorrect padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8====")


# LLM-generated content at query #33
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #34
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8---") == b"Hello"  # Extra padding

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #35
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Missing padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"  # URL-safe variant

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test with padding
    assert base64_decode("YWJj") == b"abc"
    assert base64_decode("YWJjZA==") == b"abcd"
    assert base64_decode("YWJjZGU=") == b"abcde"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!Base64@String")

    with pytest.raises(BadData):
        base64_decode("YWJj!ZA==")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("YWJj\xff\xfe") == b"abc"


