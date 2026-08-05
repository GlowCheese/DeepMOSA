####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
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
    assert base64_decode("SGVsbG8_") == b"Hello/"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with ignore errors
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test decoding invalid base64 raises BadData
    with pytest.raises(BadData):
        base64_decode("Invalid@@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character


# LLM-generated content at query #2
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello+"
    assert base64_decode("SGVsbG8_") == b"Hello/"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding with ignored characters
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test decoding with empty string
    assert base64_decode("") == b""

    # Test decoding with invalid base64
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8==!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!@#")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!@#$%^&*()")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!@#$%^&*()_+-=[]{};:'\",./<>?|`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!@#$%^&*()_+-=[]{};:'\",./<>?|`~1234567890")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!@#$%^&*()_+-=[]{};:'\",./<>?|`~1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!@#$%^&*()_+-=[]{};:'\",./<>?|`~1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=[]{};:'\",./<>?|`~1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!@#$%^&*()_+-=[]{};:'\",./<>?|`~1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=[]{};:'\",./<>?|`~1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=[]{};:'\",./<>?|`~1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


# LLM-generated content at query #3
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe characters
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8")
    assert base64_decode("SGVsbG8") == base64_decode("SGVsbG8")
    assert base64_decode("SGVsbG8") == base64_decode("SGVsbG8")

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8==")  # Incorrect padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")  # Incorrect padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")  # Invalid character with padding

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\xff") == b"Hello"
    assert base64_decode("SGVsbG8=\xff\xff") == b"Hello"


# LLM-generated content at query #4
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
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello"
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")

    # Test decoding with non-ASCII characters
    assert base64_decode("8J+YgA==") == b"\x1f\x8b\x08"
    assert base64_decode("8J+YgA") == b"\x1f\x8b\x08"

    # Test decoding with extra padding
    assert base64_decode("SGVsbG8=====") == b"Hello"


# LLM-generated content at query #5
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding a URL-safe base64 string
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with ignore errors
    assert base64_decode("SGVsbG8!") == b"Hello"
    assert base64_decode(b"SGVsbG8!") == b"Hello"

    # Test decoding invalid base64 string raises BadData
    with pytest.raises(BadData):
        base64_decode("Invalid@")

    with pytest.raises(BadData):
        base64_decode(b"Invalid@")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8é") == b"Hello"
    assert base64_decode(b"SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #6
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

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
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #7
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 encoded strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Missing padding

    # Test URL-safe characters
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q=") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q==") == b"<bigfoot>"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #8
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Missing padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"  # URL-safe variant

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode(b"SGVsbG8!@#") == b"Hello"

    # Test decoding with incorrect padding (should be handled)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode(b"SGVsbG8\xff") == b"Hello"

    # Test decoding invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character that breaks decoding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")  # Invalid padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8==!")  # Invalid padding


# LLM-generated content at query #9
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"  # Without padding
    assert base64_decode("") == b""

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe characters
    assert base64_decode("PGJpZ25hbWU+") == b"<big name>"
    assert base64_decode("PGJpZ25hbWU") == b"<big name>"
    assert base64_decode("-_") == b"\xfb\xff"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"


# LLM-generated content at query #10
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello"
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")  # invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # invalid character

    # Test decoding with non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"  # extra padding


# LLM-generated content at query #11
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Without padding

    # Test URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello"
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode(b"Invalid!")

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode(b"SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #12
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe characters
    assert base64_decode("PGJyPg==") == b"<br>"
    assert base64_decode("PGJyPg") == b"<br>"
    assert base64_decode("PGJyPg==") == b"<br>"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 1000)  # Invalid padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 1000 + "!")  # Invalid character with padding

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8=\u00ff\u00ff") == b"Hello"


# LLM-generated content at query #13
#--------------------------

```python
def test_base64_decode():
    # Test valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode("") == b""  # Empty string

    # Test URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8===") == b"Hello"  # Extra padding

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character

    # Test non-ASCII input
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test with errors parameter
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test with different encoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"


# LLM-generated content at query #14
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
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!\n") == b"Hello"
    assert base64_decode(b"SGVsbG8!\n") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid@@")

    with pytest.raises(BadData):
        base64_decode(b"Invalid@@")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8é") == b"Hello"
    assert base64_decode(b"SGVsbG8\xc3\xa9") == b"Hello"


# LLM-generated content at query #16
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode("") == b""

    # Test URL-safe characters
    assert base64_decode("PGJpZ25hbWU+") == b"<big name>"
    assert base64_decode("PGJpZ25hbWU") == b"<big name>"
    assert base64_decode("YWJjZGVmZ2g=") == b"abcdefgh"
    assert base64_decode("YWJjZGVmZ2g") == b"abcdefgh"

    # Test with padding needed
    assert base64_decode("YWJj") == b"ab"
    assert base64_decode("YWJjZGU=") == b"abcde"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Hello!!")

    with pytest.raises(BadData):
        base64_decode("12345*")

    with pytest.raises(BadData):
        base64_decode("!@#$%^")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=☺") == b"Hello"
    assert base64_decode("SGVsbG8=😊") == b"Hello"

    # Test with whitespace (should be ignored)
    assert base64_decode(" SGVsbG8= ") == b"Hello"
    assert base64_decode("SG Vs bG 8=") == b"Hello"

    # Test with newlines (should be ignored)
    assert base64_decode("SGVs\nbG8=") == b"Hello"
    assert base64_decode("SGVs\r\nbG8=") == b"Hello"


# LLM-generated content at query #17
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8-V29ybGQ=") == b"Hello-World"
    assert base64_decode("SGVsbG8_V29ybGQ") == b"Hello_World"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8===") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#$") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with incorrect padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"


# LLM-generated content at query #18
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

    # Test strings with special characters
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")  # Invalid character

    # Test strings with incorrect padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Missing padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # Correct padding
    assert base64_decode("SGVsbG8==") == b"Hello"  # Extra padding

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"


# LLM-generated content at query #19
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8


# LLM-generated content at query #20
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("") == b""
    assert base64_decode("Zg==") == b"f"
    assert base64_decode("Zg") == b"f"  # Without padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"

    # Test URL-safe characters
    assert base64_decode("PGJyPg==") == b"<br>"
    assert base64_decode("PGJyPg") == b"<br>"
    assert base64_decode("-_=") == b"\xff"
    assert base64_decode("-_") == b"\xff"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=SGVsbG8=")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\x00\x01\x02") == b"Hello"
    assert base64_decode("SGVsbG8=\xff\xfe\xfd") == b"Hello"

    # Test with different padding lengths
    assert base64_decode("SGVs") == b"H"
    assert base64_decode("SGV") == b"H"
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

    # Test URL-safe characters
    assert base64_decode("PGJhcmV2YWx1ZT4=") == b"<barevalue>"
    assert base64_decode("PGJhcmV2YWx1ZT4") == b"<barevalue>"
    assert base64_decode("PGJhcmV2YWx1ZT4=") == b"<barevalue>"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!=")  # Invalid character with padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")  # Invalid character at end
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")  # Incorrect padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8====")  # Too much padding

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8=\u00ff\u00fe") == b"Hello"


# LLM-generated content at query #22
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

    # Test URL-safe characters
    assert base64_decode("PGJhcj4=") == b"<bar>"  # Standard base64
    assert base64_decode("PGJhcj4") == b"<bar>"  # Without padding
    assert base64_decode("PGJhcj4=") == b"<bar>"  # With padding
    assert base64_decode("PGJhcj4") == b"<bar>"  # Without padding
    assert base64_decode("PGJhcj4=") == b"<bar>"  # With padding
    assert base64_decode("PGJhcj4") == b"<bar>"  # Without padding

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8?")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8%")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8^")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8&")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8*")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8(")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8)")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8_")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8+")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8 ")  # Invalid character (space)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8\t")  # Invalid character (tab)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8\n")  # Invalid character (newline)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8\r")  # Invalid character (carriage return)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x00")  # Invalid character (null byte)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x01")  # Invalid character (control character)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x1f")  # Invalid character (control character)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x7f")  # Invalid character (control character)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8\xff")  # Invalid character (control character)


# LLM-generated content at query #23
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
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode(b"SGVsbG8!@#") == b"Hello"

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8é") == b"Hello"
    assert base64_decode(b"SGVsbG8\xc3\xa9") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8!")

    # Test decoding with incorrect padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")

    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8===")


# LLM-generated content at query #24
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe characters
    assert base64_decode("PGJvZHk+") == b"<body>"
    assert base64_decode("PGJvZHk") == b"<body>"
    assert base64_decode("PGJvZHk=") == b"<body>"
    assert base64_decode("PGJvZHk+") == b"<body>"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"


# LLM-generated content at query #25
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\xff") == b"Hello"


# LLM-generated content at query #26
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
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #27
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
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q") == b"<bigfoot>"
    assert base64_decode("YWJjZGVmZ2g=") == b"abcdefgh"
    assert base64_decode("YWJjZGVmZ2g") == b"abcdefgh"

    # Test with special characters
    assert base64_decode("8J+YgA==") == b"\x10\xff\x00"
    assert base64_decode("8J+YgA") == b"\x10\xff\x00"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")  # Invalid padding

    # Test with non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"
    assert base64_decode("SGVsbG8=äöü") == b"Hello"


# LLM-generated content at query #28
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"

    # Test decoding URL-safe characters
    assert base64_decode("PGJhc2U2ND4=") == b"<base64>"
    assert base64_decode("PGJhc2U2ND4") == b"<base64>"
    assert base64_decode("PGJhc2U2ND4=") == b"<base64>"
    assert base64_decode("PGJhc2U2ND4") == b"<base64>"

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"
    assert base64_decode(b"SGVsbG8\xff") == b"Hello"

    # Test decoding invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8%")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8^")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8&")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8*")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8(")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8)")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8+")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8/")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8<")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8>")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8[")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8]")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8{")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8}")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8|")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\\")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\"")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8'")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8`")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8 ")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\t")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\n")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\r")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x0b")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x0c")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x0d")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x0e")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x0f")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x10")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x11")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x12")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x13")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x14")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x15")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x16")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x17")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x18")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x19")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x1a")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x1b")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x1c")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x1d")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x1e")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x1f")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x7f")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x80")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x81")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x82")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x83")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x84")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x85")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x86")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x87")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x88")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x89")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x8a")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x8b")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x8c")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x8d")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x8e")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x8f")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x90")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x91")

    with pytest.raises(BadData


# LLM-generated content at query #29
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Without padding

    # Test URL-safe characters
    assert base64_decode("PGJyb250Zz4=") == b"<bront>"
    assert base64_decode("PGJyb250Zz4") == b"<bront>"  # Without padding
    assert base64_decode("PGJyb250Zz4=") == b"<bront>"
    assert base64_decode("PGJyb250Zz4") == b"<bront>"  # Without padding

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=")  # Invalid padding (should not raise)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8")  # Invalid padding (should not raise)

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode(b"SGVsbG8=\xff") == b"Hello"


# LLM-generated content at query #30
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
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"
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
    assert base64_decode("SGVsbG8=äöü") == b"Hello"


# LLM-generated content at query #31
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8===") == b"Hello"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8-V29ybGQ=") == b"Hello-World"
    assert base64_decode("SGVsbG8_V29ybGQ=") == b"Hello_World"

    # Test decoding bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#$") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("InvalidBase64!")

    # Test decoding with incorrect padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=InvalidPadding=")


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
    assert base64_decode("SGVsbG8-") == b"Hello"
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with ignore errors
    assert base64_decode("SGVsbG8!") == b"Hello"
    assert base64_decode("SGVsbG8\xff") == b"Hello"

    # Test decoding with invalid base64
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!#")
    with pytest.raises(BadData):
        base64_decode("SGVsbG8\xff\xff")


# LLM-generated content at query #33
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"  # Without padding

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8-V29ybGQ") == b"Hello-World"

    # Test decoding bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!\n") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid@@")

    # Test decoding with incorrect padding (should handle)
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"


# LLM-generated content at query #34
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode("") == b""  # Empty string

    # Test URL-safe characters
    assert base64_decode("PGJyb2FkY2FzdD4=") == b"<broadcast>"
    assert base64_decode("PGJyb2FkY2FzdD4") == b"<broadcast>"
    assert base64_decode("PGJyb2FkY2FzdD4=") == b"<broadcast>"

    # Test with special characters
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"
    assert base64_decode("aGVsbG8gd29ybGQ") == b"hello world"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character

    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")  # Incorrect padding

    # Test with non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #35
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 encoded strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Without padding

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding with invalid base64 data
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=äöü") == b"Hello"
    assert base64_decode("SGVsbG8=😊") == b"Hello"


# LLM-generated content at query #36
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test raising BadData for invalid input
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"


# LLM-generated content at query #37
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode("") == b""

    # Test URL-safe characters
    assert base64_decode("PGJpZ25lcnM+") == b"<bigners>"
    assert base64_decode("PGJpZ25lcnM-") == b"<bigners>"

    # Test with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\x00\x01") == b"Hello"
    assert base64_decode("SGVsbG8=\x7f") == b"Hello"


# LLM-generated content at query #38
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe characters
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8")
    assert base64_decode("SGVsbG8") == base64_decode("SGVsbG8")
    assert base64_decode("SGVsbG8") == base64_decode("SGVsbG8")

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test padding handling
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWE=") == b"aa"
    assert base64_decode("YWE") == b"aa"
    assert base64_decode("YWFh") == b"aaa"
    assert base64_decode("YWFh") == b"aaa"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=")  # Invalid padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8")  # Invalid length
    with pytest.raises(BadData):
        base64_decode("SGVsbG8")  # Invalid length


# LLM-generated content at query #39
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode("") == b""
    assert base64_decode("Zg==") == b"f"
    assert base64_decode("Zm8=") == b"fo"
    assert base64_decode("Zm9v") == b"foo"
    assert base64_decode("Zm9vYg==") == b"foob"
    assert base64_decode("Zm9vYmE=") == b"fooba"
    assert base64_decode("Zm9vYmFy") == b"foobar"

    # Test URL-safe characters
    assert base64_decode("PGJhcj4=") == b"<bar>"
    assert base64_decode("PGJhcj4") == b"<bar>"
    assert base64_decode("PGJhcj4=") == b"<bar>"
    assert base64_decode("PGJhcj4") == b"<bar>"

    # Test with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")  # Invalid character

    # Test with non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"

    # Test with padding that's too long
    assert base64_decode("SGVsbG8=====") == b"Hello"


# LLM-generated content at query #40
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 encoded strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with URL-safe characters
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q") == b"<bigfoot>"

    # Test decoding with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWE=") == b"aa"
    assert base64_decode("YWFh") == b"aaa"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#$") == b"Hello"

    # Test raising BadData for invalid base64
    with pytest.raises(BadData):
        base64_decode("Invalid@@@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"


# LLM-generated content at query #41
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"  # Without padding

    # Test decoding URL-safe base64 strings
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 1000)  # Invalid length

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"


# LLM-generated content at query #42
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
    assert base64_decode("a") == b"\xab"
    assert base64_decode("ab") == b"\xab"
    assert base64_decode("abc") == b"\xab"

    # Test decoding with URL-safe characters
    assert base64_decode("PDw_Pz8-Pg") == b"<<???>"

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test raising BadData for invalid base64
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 100 + "!")

    # Test with non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=äöü") == b"Hello"


# LLM-generated content at query #43
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe characters
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8-")
    assert base64_decode("SGVsbG8") == base64_decode("SGVsbG8_")

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #44
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"  # Without padding

    # Test URL-safe characters
    assert base64_decode("SGVsbG8gd29ybGQh") == b"Hello world!"
    assert base64_decode("SGVsbG8gd29ybGQhIQ==") == b"Hello world!!"
    assert base64_decode("SGVsbG8gd29ybGQhIQ") == b"Hello world!!"  # Without padding

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 string
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 100 + "!")  # Invalid character at end

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8=\u00ff\u00fe") == b"Hello"


# LLM-generated content at query #45
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"  # without padding

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8-V29ybGQ") == b"Hello-World"
    assert base64_decode("SGVsbG8_V29ybGQ") == b"Hello_World"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=")  # valid but with incorrect padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")  # too much padding


# LLM-generated content at query #46
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
    assert base64_decode("SGVsbG8gV29ybGQ-") == b"Hello World+"
    assert base64_decode("SGVsbG8gV29ybGQ_") == b"Hello World/"

    # Test with non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")

    # Test with random bytes
    import os
    random_bytes = os.urandom(32)
    encoded = base64.urlsafe_b64encode(random_bytes).rstrip(b"=")
    assert base64_decode(encoded) == random_bytes


# LLM-generated content at query #47
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"  # without padding

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8_V29ybGQ=") == b"Hello?World"
    assert base64_decode("SGVsbG8-V29ybGQ") == b"Hello-World"

    # Test decoding bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!\n") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid@Base64")

    # Test decoding with incorrect padding
    assert base64_decode("SGVsbG8==") == b"Hello"  # extra padding should be handled


# LLM-generated content at query #48
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

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8-") == b"Hello+"
    assert base64_decode("SGVsbG8_") == b"Hello/"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode(b"SGVsbG8!@#") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 1000)  # Invalid length

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8é") == b"Hello"
    assert base64_decode("SGVsbG8é=") == b"Hello"


# LLM-generated content at query #49
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
    assert base64_decode("SGVsbG8====") == b"Hello"  # Extra padding

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"

    # Test longer strings
    assert base64_decode("VGhpcyBpcyBhIGxvbmdlciBzdHJpbmcgdGhhdCBzaG91bGQgYmUgdGVzdGVk") == b"This is a longer string that should be tested"


# LLM-generated content at query #50
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("") == b""

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello"
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #51
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe base64 decoding
    assert base64_decode("PGJpZ25hbWU+") == b"<big name>"
    assert base64_decode("PGJpZ25hbWU+") == b"<big name>"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 100 + "!")  # invalid character at end

    # Test strings with incorrect padding
    assert base64_decode("SGVsbG8") == b"Hello"  # missing padding
    assert base64_decode("SGVsbG8==") == b"Hello"  # extra padding

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8=\u00ff\u00ff") == b"Hello"


# LLM-generated content at query #52
#--------------------------

```python
def test_base64_decode():
    # Test with valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding

    # Test with valid base64 bytes
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # without padding

    # Test with URL-safe base64
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test with empty string
    assert base64_decode("") == b""

    # Test with invalid base64 string
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    # Test with invalid base64 bytes
    with pytest.raises(BadData):
        base64_decode(b"Invalid!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode(b"SGVsbG8!@#") == b"Hello"


# LLM-generated content at query #53
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
    assert base64_decode("PGJpZ25hbWU+") == b"<big name>"
    assert base64_decode("PGJpZ25hbWU") == b"<big name>"
    assert base64_decode("-_=") == b"\xfb\xff\xff"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")  # Invalid character

    # Test with non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"


# LLM-generated content at query #54
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding with padding added
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid base64 string
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #55
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
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!\n") == b"Hello"

    # Test raising BadData for invalid base64
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    # Test decoding longer string
    long_string = "This is a longer string to test base64 decoding."
    encoded = base64.urlsafe_b64encode(long_string.encode()).rstrip(b"=")
    assert base64_decode(encoded) == long_string.encode()


# LLM-generated content at query #56
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
    assert base64_decode("SGVsbG8-") == b"Hello\xff"
    assert base64_decode("SGVsbG8_") == b"Hello\xfb"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!\x00") == b"Hello"
    assert base64_decode(b"SGVsbG8!\x00") == b"Hello"

    # Test raising BadData for invalid base64
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\x00")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 100)  # Invalid length


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe characters
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


# LLM-generated content at query #2
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8-V29ybGQ=") == b"Hello-World"
    assert base64_decode("SGVsbG8_V29ybGQ") == b"Hello_World"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8===") == b"Hello"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"
    assert base64_decode("SGVsbG8😊") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""


# LLM-generated content at query #3
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding a URL-safe base64 string
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding a string with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding an empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with incorrect padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")

    # Test decoding a string with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #4
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding a URL-safe base64 string
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding a string with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding an empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with incorrect padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8====")

    # Test decoding a string with non-ASCII characters (should ignore them)
    assert base64_decode("SGVsbG8=😊") == b"Hello"


# LLM-generated content at query #5
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"

    # Test with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"

    # Test with URL-safe characters
    assert base64_decode("SGVsbG8gd29ybGQ-") == b"Hello world+"
    assert base64_decode("SGVsbG8gd29ybGQ_") == b"Hello world/"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 100 + "!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8gd29ybGQ=\u00ff") == b"Hello world"


# LLM-generated content at query #6
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"  # Without padding

    # Test decoding URL-safe base64 strings
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"
    assert base64_decode("SGVsbG8gV29ybGQh-") == b"Hello World!-"

    # Test decoding bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!\x00") == b"Hello"

    # Test raising BadData for invalid base64
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 100 + "!")  # Invalid character at end


# LLM-generated content at query #7
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8===") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8-V29ybGQ=") == b"Hello-World"
    assert base64_decode("SGVsbG8_V29ybGQ=") == b"Hello_World"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid@Base64")
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #8
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test with padding
    assert base64_decode("aQ==") == b"i"
    assert base64_decode("aWE=") == b"ia"
    assert base64_decode("aWFi") == b"iab"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"


# LLM-generated content at query #9
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

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test decoding with incorrect padding (should be handled)
    assert base64_decode("SGVsbG") == b"Hell"

    # Test decoding invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 1000)  # Invalid length


# LLM-generated content at query #10
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Missing padding
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"

    # Test URL-safe characters
    assert base64_decode("PGJyPg==") == b"<br>"
    assert base64_decode("PGJyPg") == b"<br>"
    assert base64_decode("PGJyPg==") == b"<br>"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"


# LLM-generated content at query #11
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"
    assert base64_decode("SGVsbG8gV29ybGQh-") == b"Hello World!"

    # Test empty string
    assert base64_decode("") == b""

    # Test padding handling
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"
    assert base64_decode("SGVsbG8gV29ybGQh==") == b"Hello World!"
    assert base64_decode("SGVsbG8gV29ybGQh=") == b"Hello World!"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8gV29ybGQ!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8gV29ybGQ!\x00") == b"Hello World!"


# LLM-generated content at query #12
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode(b"SGVsbG8gd29ybGQ=") == b"Hello world"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("PGJyIC8+") == b"<br />"
    assert base64_decode("PGJyL14=") == b"<br/~"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"


# LLM-generated content at query #13
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
    assert base64_decode("SGVsbG8___") == b"Hello"  # Multiple padding chars

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode(123)  # Invalid type

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00\x01") == b"Hello"


# LLM-generated content at query #14
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
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


# LLM-generated content at query #15
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"  # Without padding

    # Test decoding URL-safe base64 strings
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8")
    assert base64_decode("SGVsbG8-V29ybGQ=") == base64_decode("SGVsbG8-V29ybGQ")

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #16
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding

    # Test decoding a URL-safe base64 string
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello"
    assert base64_decode("SGVsbG8_") == b"Hello"

    # Test decoding an empty string
    assert base64_decode("") == b""

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with invalid padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")

    # Test decoding a string with invalid length (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG")

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")


# LLM-generated content at query #17
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding an empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding a string with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode(b"SGVsbG8!@#") == b"Hello"

    # Test decoding a string with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8!")

    # Test decoding a string with incorrect padding
    assert base64_decode("SGVsbG") == b"Hell"
    assert base64_decode(b"SGVsbG") == b"Hell"

    # Test decoding a string with URL-safe characters
    assert base64_decode("SGVsbG8=") == base64_decode("SGVsbG8=")
    assert base64_decode(b"SGVsbG8=") == base64_decode(b"SGVsbG8=")

    # Test decoding a string with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=ñ") == b"Hello"
    assert base64_decode(b"SGVsbG8=\xc3\xb1") == b"Hello"


# LLM-generated content at query #18
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("aGVsbG8=") == b"hello"  # Lowercase
    assert base64_decode("aGVsbG8") == b"hello"  # Lowercase without padding
    assert base64_decode("SGVsbG8h") == b"Hello!"  # With special character
    assert base64_decode("SGVsbG8h") == b"Hello!"  # With special character without padding
    assert base64_decode("") == b""  # Empty string
    assert base64_decode("Zg==") == b"f"  # Single character
    assert base64_decode("Zg") == b"f"  # Single character without padding

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello"  # URL-safe character
    assert base64_decode("SGVsbG8_") == b"Hello"  # URL-safe character

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8-") == b"Hello"
    assert base64_decode(b"SGVsbG8_") == b"Hello"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8$")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8%")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8&")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8*")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8+")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8/")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8:")  # Invalid character


# LLM-generated content at query #19
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8h") == b"Hello!"
    assert base64_decode(b"SGVsbG8h") == b"Hello!"
    assert base64_decode("aGVsbG8=") == b"hello"
    assert base64_decode("aGVsbG8") == b"hello"  # Missing padding

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8h") == base64_decode("SGVsbG8h")
    assert base64_decode("SGVsbG8h") == base64_decode("SGVsbG8h")
    assert base64_decode("-__") == b"\xff\xff"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8h==")  # Invalid padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8h=")  # Invalid padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8h===")  # Invalid padding

    # Test non-ASCII characters are ignored
    assert base64_decode("SGVsbG8h\xff") == b"Hello!"
    assert base64_decode("SGVsbG8h\x00") == b"Hello!"


# LLM-generated content at query #20
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
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q") == b"<bigfoot>"

    # Test decoding with special characters
    assert base64_decode("SGVsbG8h") == b"Hello!"
    assert base64_decode("SGVsbG8_") == b"Hello_"

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"

    # Test decoding with extra padding
    assert base64_decode("SGVsbG8====") == b"Hello"
    assert base64_decode("SGVsbG8=====") == b"Hello"


# LLM-generated content at query #21
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
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid@Base64")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"
    assert base64_decode("SGVsbG8😊") == b"Hello"


# LLM-generated content at query #22
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
    assert base64_decode("SGVsbG8-") == b"Hello-"
    assert base64_decode("SGVsbG8_") == b"Hello_"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode("SGVsbG8\x00\x01") == b"Hello"

    # Test decoding with BadData exception
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8\xff")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8é") == b"Hello"


# LLM-generated content at query #23
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
    assert base64_decode("SGVsbG8-") == b"Hello"  # URL-safe variant
    assert base64_decode("SGVsbG8_") == b"Hello"  # URL-safe variant

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #24
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"  # Without padding

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe characters
    assert base64_decode("PGJyPg==") == b"<br>"
    assert base64_decode("PGJyPg") == b"<br>"
    assert base64_decode("PGJyPg==") == b"<br>"
    assert base64_decode("PGJyPg") == b"<br>"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #25
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("") == b""
    assert base64_decode("Zg==") == b"f"
    assert base64_decode("Zm8=") == b"fo"
    assert base64_decode("Zm9v") == b"foo"
    assert base64_decode("Zm9vYg==") == b"foob"
    assert base64_decode("Zm9vYmE=") == b"fooba"
    assert base64_decode("Zm9vYmFy") == b"foobar"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with URL-safe characters
    assert base64_decode("PGJyPg==") == b"<br>"
    assert base64_decode("PGJyPg") == b"<br>"
    assert base64_decode("PGJyPg==") == b"<br>"

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=äöü") == b"Hello"
    assert base64_decode("SGVsbG8=😊") == b"Hello"


# LLM-generated content at query #26
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

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with padding
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"

    # Test decoding URL-safe base64
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#$") == b"Hello"
    assert base64_decode("SGVsbG8!@#$") == b"Hello"

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8é") == b"Hello"
    assert base64_decode("SGVsbG8é") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")


# LLM-generated content at query #27
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Without padding

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test strings with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode(b"SGVsbG8!@#") == b"Hello"

    # Test invalid base64 strings (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=")  # Invalid padding
    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8=")  # Invalid padding


# LLM-generated content at query #28
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

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8===") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!@#$%^&*()_+-=[]{};:'\",./<>?`~")

    with pytest.raises(BadData):
       


# LLM-generated content at query #29
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"  # Without padding

    # Test URL-safe base64 decoding
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"
    assert base64_decode("SGVsbG8-V29ybGQh") == b"Hello+World!"  # URL-safe '-'
    assert base64_decode("SGVsbG8_V29ybGQh") == b"Hello/World!"  # URL-safe '_'

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test padding handling
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWE") == b"aa"
    assert base64_decode("YWFh") == b"aaa"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=")  # Invalid padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")  # Too much padding

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #30
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding with padding added
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8_") == b"Hello?"

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    # Test decoding with invalid padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")


# LLM-generated content at query #31
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test decoding a valid base64 string without padding
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding a valid base64 string with URL-safe characters
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test decoding a valid base64 string with URL-safe characters and no padding
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding an empty string
    assert base64_decode("") == b""

    # Test decoding a string with invalid characters (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding a string with invalid padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")

    # Test decoding a string with non-ASCII characters (should ignore them)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"

    # Test decoding a string with whitespace (should ignore them)
    assert base64_decode(" SGVsbG8= ") == b"Hello"

    # Test decoding a string with newlines (should ignore them)
    assert base64_decode("SGVs\nbG8=") == b"Hello"


# LLM-generated content at query #32
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
    assert base64_decode("PGJpZ25hbWU+") == b"<big name>"
    assert base64_decode("PGJpZ25hbWU") == b"<big name>"
    assert base64_decode("YWJjZGVmZ2g=") == b"abcdefgh"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # Invalid character
    with pytest.raises(BadData):
        base64_decode(123)  # Invalid type

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8=\u00ff\u00fe") == b"Hello"


# LLM-generated content at query #33
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Without padding

    # Test URL-safe characters
    assert base64_decode("PGJhcj4=") == b"<bar>"
    assert base64_decode("PGJhcj4") == b"<bar>"  # Without padding
    assert base64_decode("PGJhcj4=") == b"<bar>"
    assert base64_decode("PGJhcj4") == b"<bar>"  # Without padding

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

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"  # Without padding


# LLM-generated content at query #34
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
    assert base64_decode("PGJpZ25hbWU+") == b"<big name>"
    assert base64_decode("PGJpZ25hbWU") == b"<big name>"  # Without padding
    assert base64_decode("YWJjZGVmZw==") == b"abcdefg"

    # Test with padding handling
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWE=") == b"aa"
    assert base64_decode("YWFh") == b"aaa"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")  # Invalid character with padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")  # Too much padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG")  # Invalid length

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8=\u00ff\u00fe") == b"Hello"


# LLM-generated content at query #35
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding a URL-safe base64 string
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode(b"SGVsbG8!@#") == b"Hello"

    # Test decoding with incorrect padding (should be handled)
    assert base64_decode("SGVsbG") == b"Hello"
    assert base64_decode(b"SGVsbG") == b"Hello"

    # Test decoding with BadData exception
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8é") == b"Hello"
    assert base64_decode(b"SGVsbG8\xc3\xa9") == b"Hello"


# LLM-generated content at query #36
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode(b"SGVsbG8gd29ybGQ=") == b"Hello world"

    # Test decoding with padding
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ==") == b"Hello world"

    # Test decoding URL-safe base64
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ==") == b"Hello world"

    # Test decoding with special characters
    assert base64_decode("SGVsbG8gd29ybGQ-") == b"Hello world"
    assert base64_decode("SGVsbG8gd29ybGQ_") == b"Hello world"

    # Test decoding an empty string
    assert base64_decode("") == b""

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8gd29ybGQ!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8gd29ybGQ=\u00ff") == b"Hello world"


# LLM-generated content at query #37
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

    # Test decoding with padding
    assert base64_decode("a") == b"\x80"
    assert base64_decode("aa") == b"\x80\x80"
    assert base64_decode("aaa") == b"\x80\x80\x80"

    # Test decoding URL-safe base64
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#$") == b"Hello"

    # Test raising BadData for invalid base64
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=" * 1000 + "!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8\u00ff") == b"Hello"


# LLM-generated content at query #38
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8-") == b"Hello\x00"
    assert base64_decode("SGVsbG8_") == b"Hello\x00"

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test raising BadData for invalid base64
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8==!")


# LLM-generated content at query #39
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8-V29ybGQ") == b"Hello-World"
    assert base64_decode("SGVsbG8_V29ybGQ") == b"Hello_World"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8===") == b"Hello"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid@Base64")
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #40
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
    assert base64_decode("PGJvZHk-") == b"<body>"
    assert base64_decode("PGJvZHk_") == b"<body>"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!@#$")
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8=\xff") == b"Hello"

    # Test with different padding lengths
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"


# LLM-generated content at query #41
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # bytes input
    assert base64_decode("") == b""  # Empty string

    # Test URL-safe characters
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q") == b"<bigfoot>"  # Without padding
    assert base64_decode("YWJjX2RlZg==") == b"abc_def"
    assert base64_decode("YWJjX2RlZg") == b"abc_def"  # Without padding

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=")  # Valid but with invalid padding (shouldn't raise)
    with pytest.raises(BadData):
        base64_decode(123)  # Invalid input type (should raise TypeError from want_bytes)


# LLM-generated content at query #42
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8-V29ybGQ=") == b"Hello-World"
    assert base64_decode("SGVsbG8_V29ybGQ=") == b"Hello_World"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid@Base64")
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"
    assert base64_decode("SGVsbG8=\u00ff\u00fe") == b"Hello"


# LLM-generated content at query #43
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe characters
    assert base64_decode("PGJvZHk+") == b"<body>"
    assert base64_decode("PGJvZHk") == b"<body>"
    assert base64_decode("PGJvZHk-Pg==") == b"<body>"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test padding handling
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWE=") == b"aa"
    assert base64_decode("YWFh") == b"aaa"
    assert base64_decode("YWFhYQ==") == b"aaaa"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!=")  # Invalid character with padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8====")  # Too much padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG")  # Invalid length


# LLM-generated content at query #44
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # bytes input without padding

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"
    assert base64_decode("SGVsbG8= ") == b"Hello"  # with whitespace
    assert base64_decode(" SGVsbG8=") == b"Hello"  # with leading whitespace

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")  # invalid character with padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8==!")  # invalid character with padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8= =")  # invalid padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8= = =")  # invalid padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8= = = =")  # invalid padding


# LLM-generated content at query #45
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"
    assert base64_decode("SGVsbG8-V29ybGQ") == b"Hello World"
    assert base64_decode("SGVsbG8_V29ybGQ") == b"Hello World"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8gV29ybGQ\xff") == b"Hello World"


# LLM-generated content at query #46
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding a URL-safe base64 string
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding a string with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding an empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding a string with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode(b"SGVsbG8!@#") == b"Hello"

    # Test decoding a string with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8!")

    # Test decoding a string with incorrect padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8===")

    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8===")


# LLM-generated content at query #47
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # bytes input without padding

    # Test URL-safe characters
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q=") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q==") == b"<bigfoot>"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8@")  # invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8#")  # invalid character

    # Test non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8\xff") == b"Hello"
    assert base64_decode("SGVsbG8\x00") == b"Hello"


# LLM-generated content at query #48
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"

    # Test decoding a URL-safe base64 string
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test decoding with extra padding (should be ignored)
    assert base64_decode("SGVsbG8=====") == b"Hello"

    # Test decoding bytes input
    assert base64_decode(b"SGVsbG8gd29ybGQ=") == b"Hello world"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8gd29ybGQ!@#") == b"Hello world"

    # Test raising BadData for invalid base64
    with pytest.raises(BadData):
        base64_decode("Invalid@@@")

    # Test raising BadData for incorrect padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8gd29ybGQ====")


# LLM-generated content at query #49
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"  # With padding
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8


# LLM-generated content at query #50
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
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q") == b"<bigfoot>"

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"

    # Test decoding with incorrect padding (should be fixed)
    assert base64_decode("SGVsbG") == b"Hell"

    # Test raising BadData for invalid base64
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode(b"Invalid!")


# LLM-generated content at query #51
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid characters (should ignore)
    assert base64_decode("SGVsbG8@") == b"Hello"
    assert base64_decode("SGVsbG8#") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test decoding with incorrect padding (should handle)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding with non-ASCII characters (should ignore)
    assert base64_decode("SGVsbG8\xff") == b"Hello"


# LLM-generated content at query #52
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe characters
    assert base64_decode("PGJhcmZvbz4=") == b"<barfoo>"
    assert base64_decode("PGJhcmZvbz4") == b"<barfoo>"
    assert base64_decode("PGJhcmZvb-") == b"<barfoo>"

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


# LLM-generated content at query #53
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test decoding URL-safe base64
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with ignore errors
    assert base64_decode("SGVsbG8!") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test decoding with invalid padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8====")


# LLM-generated content at query #54
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"

    # Test decoding with URL-safe characters
    assert base64_decode("SGVsbG8-V29ybGQ=") == b"Hello-World"
    assert base64_decode("SGVsbG8_V29ybGQ=") == b"Hello_World"

    # Test decoding with missing padding
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8gV29ybGQ=") == b"Hello World"

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test decoding with invalid characters (should ignore them)
    assert base64_decode("SGVsbG8!\n\r=") == b"Hello"

    # Test decoding with invalid base64 (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("Invalid@@Base64")

    # Test decoding with incorrect padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8====")


# LLM-generated content at query #55
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("aGVsbG8gd29ybGQ=") == b"hello world"
    assert base64_decode("aGVsbG8gd29ybGQ") == b"hello world"  # Without padding
    assert base64_decode("") == b""

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Without padding

    # Test decoding with URL-safe characters
    assert base64_decode("PGJyPg==") == b"<br>"
    assert base64_decode("PGJyPg") == b"<br>"  # Without padding
    assert base64_decode("PGJyPg==") == b"<br>"
    assert base64_decode("PGJyPg") == b"<br>"  # Without padding

    # Test decoding with invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid@Base64")
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=")  # Invalid padding

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"
    assert base64_decode("SGVsbG8😊") == b"Hello"  # Without padding


# LLM-generated content at query #56
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"  # Without padding

    # Test decoding with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test URL-safe characters
    assert base64_decode("PGJyb2tlbiBieWU9ImZvbyI+") == b"<broken bye=\"foo\">"
    assert base64_decode("PGJyb2tlbiBieWU9ImZvbyI+") == b"<broken bye=\"foo\">"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    # Test with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8=\u00ff") == b"Hello"


# LLM-generated content at query #57
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Without padding
    assert base64_decode(b"SGVsbG8=") == b"Hello"  # Bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Bytes input without padding

    # Test URL-safe characters
    assert base64_decode("PGJpZ2Zvb3Q+") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q=") == b"<bigfoot>"
    assert base64_decode("PGJpZ2Zvb3Q==") == b"<bigfoot>"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test with non-ASCII input (should be ignored)
    assert base64_decode("SGVsbG8=😊") == b"Hello"
    assert base64_decode("SGVsbG8=äöü") == b"Hello"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8==!")  # Invalid character
    with pytest.raises(BadData):
        base64_decode("SGVsbG8====")  # Too much padding
    with pytest.raises(BadData):
        base64_decode("SGVsbG8=1")  # Invalid character


# LLM-generated content at query #58
#--------------------------

```python
def test_base64_decode():
    # Test decoding a valid base64 string
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test decoding a URL-safe base64 string
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test decoding with padding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8==") == b"Hello"

    # Test decoding an empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8!@#") == b"Hello"
    assert base64_decode(b"SGVsbG8!@#") == b"Hello"

    # Test decoding with non-ASCII characters (should be ignored)
    assert base64_decode("SGVsbG8é") == b"Hello"
    assert base64_decode(b"SGVsbG8\xc3\xa9") == b"Hello"

    # Test decoding with invalid base64 data (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8!")

    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8!")

    # Test decoding with incorrect padding (should raise BadData)
    with pytest.raises(BadData):
        base64_decode("SGVsbG8====")

    with pytest.raises(BadData):
        base64_decode(b"SGVsbG8====")


# LLM-generated content at query #59
#--------------------------

```python
def test_base64_decode():
    # Test decoding valid base64 strings
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Missing padding
    assert base64_decode("SGVsbG8gV29ybGQ=") == b"Hello World"
    assert base64_decode("SGVsbG8gV29ybGQ") == b"Hello World"  # Missing padding

    # Test URL-safe base64 strings
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"
    assert base64_decode("SGVsbG8gV29ybGQh") == b"Hello World!"

    # Test empty string
    assert base64_decode("") == b""

    # Test strings with invalid characters (should be ignored)
    assert base64_decode("SGVsbG8gV29ybGQ!\n") == b"Hello World!"

    # Test invalid base64 strings
    with pytest.raises(BadData):
        base64_decode("Invalid!")

    with pytest.raises(BadData):
        base64_decode("SGVsbG8gV29ybGQ!")  # Invalid padding

    with pytest.raises(BadData):
        base64_decode("SGVsbG8gV29ybGQ===")  # Too much padding


