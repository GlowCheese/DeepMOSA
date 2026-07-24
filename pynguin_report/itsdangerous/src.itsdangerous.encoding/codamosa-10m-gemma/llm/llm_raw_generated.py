####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (using - and _ instead of + and /)
    # '?' in base64 urlsafe is often represented by different chars
    # Let's use a known sequence: b'\xff\xfe' -> base64 urlsafe is z_4=
    assert base64_decode("z_4") == b"\xff\xfe"

    # Test with input as bytes
    assert base64_decode(b"V29ybGQ=") == b"World"

    # Test stripping of padding (since encode rstrips it)
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test with invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""

    # Test handling of different encodings/strings via want_bytes logic
    assert base64_decode("YQ==") == b"a"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == original_text.encode("utf-8")

    # Test decoding with padding missing (function should handle it)
    # "abc" in urlsafe b64 is "YWJj" (no padding needed)
    # "ab" in urlsafe b64 is "YWI" (needs one '=')
    encoded_no_padding = b"YWI" 
    assert base64_decode(encoded_no_padding) == b"ab"

    # Test decoding bytes directly
    encoded_bytes = b"c29tZSBkYXRh"
    assert base64_decode(encoded_bytes) == b"some data"

    # Test decoding string directly
    assert base64_decode("Y29kZQ==") == b"code"

    # Test handling of URL-safe characters (- and _)
    # "sub-item_1" encoded is "c3ViLWl0ZW1fMQ"
    encoded_url_safe = b"c3ViLWl0ZW1fMQ"
    assert base64_decode(encoded_url_safe) == b"sub-item_1"

    # Test error handling for invalid base64 data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding cycle
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test URL-safe specific characters (using - and _)
    url_safe_input = b"\xff\xfe\xfd\x21\x22"
    encoded_url_safe = base64_encode(url_safe_input)
    assert base64_decode(encoded_url_safe) == url_safe_input

    # Test decoding with string input (utf-8)
    text_input = "python testing"
    encoded_text = base64_encode(text_input)
    assert base64_decode(encoded_text) == text_input.encode("utf-8")

    # Test decoding with missing padding (the function handles this internally)
    unpadded_encoded = b"SGVsbG8"  # "Hello" without padding
    assert base64_decode(unpadded_encoded) == b"Hello"

    # Test decoding with existing padding
    padded_encoded = b"SGVsbG8="
    assert base64_decode(padded_encoded) == b"Hello"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode(b"!!!not_base64!!!")

    # Test empty input
    assert base64_decode(b"") == b""
    assert base64_decode("") == b""
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replacing + and / with - and _)
    # "Subject?" in base64 is "U3ViamVjdD8="
    # In urlsafe, if it contained + or /, they would be - or _
    assert base64_decode("U3ViamVjdD8") == b"Subject?"

    # Test bytes input
    assert base64_decode(b"R29vZA") == b"Good"

    # Test stripping of padding (the function handles missing =)
    assert base64_decode("Y29kZQ") == b"code"
    assert base64_decode("Y29kZ") == b"code" # Partial/Shortened logic check

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid base64 data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!Invalid!!!")

    # Test various padding scenarios
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YQ==") == b"a"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == original_text.encode("utf-8")

    # Test with bytes input
    original_bytes = b"\x01\x02\x03\xff"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe characters (no padding needed, contains - and _)
    # "_" is part of the alphabet, we test a string that results in these chars
    url_safe_input = "testing_special-chars"
    encoded_url = base64_encode(url_safe_input)
    assert base64_decode(encoded_url) == url_safe_input.encode("utf-8")

    # Test padding reconstruction (input without padding characters)
    # "abc" in base64 is "YWJj", length 4. 
    # "ab" in base64 is "YWI=", length 4. Let's use a string that truncates the '='
    truncated_padding = base64_encode(b"a")  # "YQ==" becomes "YQ"
    assert base64_decode(truncated_padding) == b"a"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        # Using characters outside the URL-safe base64 alphabet (like symbols not in - or _)
        # Note: urlsafe_b64decode is quite permissive, but non-alphabet chars 
        # that break structural integrity will trigger it.
        base64_decode(b"!!!")

    # Test empty input
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test string input instead of bytes
    assert base64_decode("Ym9i") == b"bob"
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard URL-safe base64 encoding/decoding
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    assert base64_decode(b"SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding (the function handles missing padding)
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test different characters from the URL-safe alphabet (- and _)
    # "test-abc_" in urlsafe is encoded differently than standard
    encoded = base64_encode("test-abc_")
    assert base64_decode(encoded) == b"test-abc_"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test encoding/decoding round trip for various inputs
    inputs = [b"simple", "complex string with spaces", b"\x00\x01\x02"]
    for i in inputs:
        encoded = base64_encode(i)
        assert base64_decode(encoded) == want_bytes(i)

def test_want_bytes():
    assert want_bytes("hello") == b"hello"
    assert want_bytes(b"hello") == b"hello"
    assert want_bytes("abc", encoding="ascii") == b"abc"

def test_int_conversion():
    val = 123456789
    encoded = int_to_bytes(val)
    assert bytes_to_int(encoded) == val
    
    # Test boundary for 64-bit unsigned int
    max_val = (1 << 64) - 1
    encoded_max = int_to_bytes(max_val)
    assert bytes_to_int(encoded_max) == max_val
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replacing + and / with - and _)
    # "subject?" in standard B64 is "c3ViamVjdD8="
    # In urlsafe it might involve different chars if the input was different
    assert base64_decode("YV9iLWNfZA") == base64.urlsafe_b64decode(b"YV9iLWNfZA" + b"==")

    # Test padding handling (function adds missing padding)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("Y2Fi") == b"cab"

    # Test with different types (bytes input)
    assert base64_decode(b"dGVzdA") == b"test"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding cycle
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test with bytes input
    original_bytes = b"\x01\x02\x03\xff"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe specific characters (using - and _)
    # 'base64' with padding normally uses '+' and '/'
    # urlsafe replaces them with '-' and '_'
    url_safe_input = b"data\xff\xfe"
    # Manually creating a string that would use the URL-safe alphabet
    # In urlsafe, '+' becomes '-' and '/' becomes '_'
    encoded_url_safe = base64.urlsafe_b64encode(b"\xfb\xef") 
    # b'\xfb\xef' encoded is b'++8=' -> urlsafe is b'--8='
    assert base64_decode("--8") == b"\xfb\xef"

    # Test with missing padding (the function should handle it via the modulo logic)
    assert base64_decode("SGVsbG8") == b"Hello"  # "SGVsbG8" needs padding to be valid

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe specific characters (- and _)
    # "abc" in base64 is "YWJj"
    # Special chars test: payload that results in - or _
    # Using a known pattern for urlsafe
    assert base64_decode("YV9iLWM") == b"a_b-c" 
    
    # Test padding handling (the function manually adds missing =)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    
    # Test with different types (bytes input)
    assert base64_decode(b"dGVzd") == b"test"
    
    # Test error handling for invalid base64 data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Test padding recovery
    
    # Test bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test URL-safe specific characters (substituting + and / with - and _)
    # "a/b+" in standard B64 is "YS9i+"
    # In urlsafe, it should be encoded using '-' and '_'
    assert base64_decode("YS1iXw") == b"a/b+" 
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test handling of non-ascii characters in input (should be ignored due to errors="ignore")
    # "SGVsbG8=" with an injected emoji/non-ascii byte that gets stripped by ascii encoding
    assert base64_decode("SGVsbG8🚀") == b"Hello"
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from .exc import BadData

def test_base64_decode():
    # Test standard valid input (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Test padding recovery
    
    # Test URL-safe specific characters (- and _)
    # Base64 for "+/" is "Ky8="; URL-safe version is "Ky8" (with padding logic)
    assert base64_decode("Ky8") == b"+/"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with different types
    assert isinstance(base64_decode("YQ=="), bytes)
    assert isinstance(base64_decode(b"YQ"), bytes)

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test decoding non-ascii characters in input string (should be ignored per implementation)
    # 'abc' + 'non-ascii' -> if encoding is ascii and errors='ignore', it should handle it
    assert base64_decode("YWJj\x00") == b"abc"
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe specific characters (replace + with - and / with _)
    # "Subject?" in standard B64 is "U3ViamVjdD8=" -> no special chars here
    # Let's use a case that forces '-' or '_'
    # Standard: "foo/bar+" -> Base64: "Zm9vL2JhcCs=" 
    # URL-safe: "Zm9vL2JhcCs=" -> "Zm9vX2JhcCs" (actually urlsafe replaces / with _)
    assert base64_decode("Zm9vX2JhcHM") == b"foo_bahms" # Example of decoding replacement
    
    # Test padding recovery (the function handles missing '=')
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test with different types (str and bytes)
    assert base64_decode(b"Ym9i") == b"bob"
    assert base64_decode("Ym9i") == b"bob"

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test with non-ascii characters in input (should be ignored via errors='ignore')
    # 'SGVsbG8' + unicode emoji
    assert base64_decode("SGVsbG8\U0001F600") == b"Hello"
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe specific characters (replace + with - and / with _)
    # "subject?" in standard B64 is "c3ViamVjdD8="
    # In urlsafe it's "c3ViamVjdD8=" (no change here, but testing structure)
    assert base64_decode("YV9iLWNfZA") == b"a_b-c_d" # Manually constructed payload
    
    # Test padding recovery (the function adds missing '=')
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YmI") == b"bb"
    assert base64_decode("Y2Nj") == b"ccc"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test with different encoding/types
    assert base64_decode(b"SGVsbG8") == b"Hello"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"invalid_chars_!@#$")
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (using - and _ instead of + and /)
    # "base64" encoded with urlsafe often involves these if the input is specific
    assert base64_decode("YV9iLWNfZA") == b"a_b-c_d" # padded internally by function
    
    # Test input as bytes
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test with missing padding (the function handles padding automatically)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test error case: Invalid base64 data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test complex sequence
    original = b"\x00\xff\x00\xaa\xbb\xcc"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard valid input (string)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test valid input without padding (URL-safe style)
    # "Hello" in base64 is "SGVsbG8=" -> stripped to "SGVsbG8"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test input that requires padding restoration
    # "a" -> "YQ==" -> stripped "YQ" -> needs "=="
    assert base64_decode("YQ") == b"a"
    
    # Test complex string with URL-safe characters (- and _)
    # bytes: b'\xfe\xff' -> base64: 'v7//' -> urlsafe: 'v7__'
    assert base64_decode("v7__") == b"\xfe\xff"

    # Test input with different types (bytes)
    assert base64_decode(b"V29ybGQ") == b"World"

    # Test invalid base64 data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test empty input
    assert base64_decode("") == b""
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard valid input (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test URL-safe specific characters (- and _)
    # "a/b+" in standard B64 is "YScvYis=" 
    # In urlsafe it's "YScfYis" (with different mapping)
    # Let's use a known URL-safe string: 'subject?' -> 'c3ViamVjdD8'
    assert base64_decode("c3ViamVjdD8") == b"subject?"
    
    # Test empty input
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test with bytes input
    assert base64_decode(b"TWFu") == b"Man"

    # Test decoding with different padding needs
    # 'abc' -> 'YWJj' (len 4)
    # 'ab' -> 'YWI' (needs padding to YWI=)
    assert base64_decode("YWI") == b"ab"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!NotBase64!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"\xff\xfe\xfd")
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe alphabet (using - and _ instead of + and /)
    # "Subject?" in base64 is "U3ViamVjdD8=" 
    # In urlsafe, if there were + or /, they'd be - and _
    assert base64_decode("Y29uZmlybV90ZXN0") == b"confirm_test"
    assert base64_decode("Y29uZmlybS10ZXN0") == b"confirm-test"

    # Test bytes input
    assert base64_decode(b"YWJj") == b"abc"

    # Test missing padding (the function handles padding internally)
    assert base64_decode("YWJj") == b"abc"
    assert base64_decode("YQ") == b"a"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")

    # Test error handling for non-ascii characters via the encoding logic
    # Since it uses ascii with ignore, it should strip them rather than crash
    # but if the resulting base64 structure is corrupted, it raises BadData.
    with pytest.raises(BadData):
        base64_decode("????") 
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test basic functionality with string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replacing + and /)
    # "a+b/" in standard B64 is "YStiLw=="
    # In urlsafe it is "YStiLw"
    assert base64_decode("YStiLw") == b"a+b/"
    
    # Test with bytes input
    assert base64_decode(b"R29vZA") == b"Good"
    
    # Test padding reconstruction (input without '=')
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YmI") == b"bb"
    assert base64_decode("Y2Nj") == b"ccc"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test characters outside ASCII range (should be ignored per encoding="ascii", errors="ignore")
    # '€' is not valid ASCII, so it should be stripped/ignored by the function logic
    valid_part = base64_encode("test")
    assert base64_decode(f"test€{valid_part.decode()}".encode()) == b"test"
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe specific characters (replace + with - and / with _)
    # "subject?" in base64 is "c3ViamVjdD8=" -> urlsafe: "c3ViamVjdD8"
    assert base64_decode("c3ViamVjdD8") == b"subject?"
    
    # Test input as bytes
    assert base64_decode(b"Ym9vaQ") == b"booi"
    
    # Test with missing padding (the function handles padding internally)
    assert base64_decode("Ym9vaQ") == b"booi"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid base64 data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")

    # Test case with mixed types and encoding issues
    assert base64_decode("YW55IGNhcm5hbCBwbGVhc3VyZS4=") == b"any carnal pleasure."
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe specific characters (replace + and / with - and _)
    # Original: "subject?query=1" -> Base64: "c3ViamVjdD9xdWVyeT0x"
    assert base64_decode("c3ViamVjdD9xdWVyeT0x") == b"subject?query=1"
    
    # Test padding handling (the function adds missing padding automatically)
    assert base64_decode("YQ") == b"a"      # Needs ==
    assert base64_decode("YmI") == b"bb"    # Needs =
    assert base64_decode("Y2Jj") == b"ccc"  # No padding needed
    
    # Test bytes input
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test different encodings/types via want_bytes logic
    assert base64_decode("abc") == b"abc"
    
    # Test invalid data raises BadData
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!invalid!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe specific characters (replace + with - and / with _)
    # Standard B64 for "\xff\xfe" is "/v4="
    # URL-safe B64 for "\xff\xfe" is "_v4="
    assert base64_decode("_v4") == b"\xff\xfe"
    
    # Test bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test missing padding (the function handles padding internally)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("Ym") == b"b" # Note: 'Ym' is not valid for 'b', but testing logic flow
    assert base64_decode("YQ") == b"a"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test decoding with alphanumeric characters and URL-safe symbols
    input_str = "abc-123_XYZ"
    encoded = base64_encode(input_str)
    assert base64_decode(encoded) == want_bytes(input_str)
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test with bytes input
    original_bytes = b"\x00\xff\xfe\xfd"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe characters (handling '-' and '_')
    # 'abc?' in urlsafe is 'YWJjPw' -> without padding check
    url_safe_input = "YWJj" 
    assert base64_decode(url_safe_input) == b"abc"

    # Test decoding with missing padding (the function should handle it)
    # 'abcd' encoded is 'YWJjZA==' -> 'YWJjZA' is valid input for the function
    assert base64_decode("YWJjZA") == b"abcd"

    # Test with different string types (str vs bytes)
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test invalid base64 data raises BadData
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!NotBase64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test with bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test URL-safe specific characters (- and _)
    # "_-" in urlsafe corresponds to specific byte sequences
    assert base64_decode("-_") == b"\xf8" 
    
    # Test complex string
    input_str = "python_is_great-123"
    encoded = base64_encode(input_str)
    assert base64_decode(encoded) == input_str.encode()

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        # Using characters outside the valid alphabet/structure if possible
        # Though urlsafe handles many, certain malformations trigger errors
        base64_decode("!!!") 

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (no padding needed in input)
    # 'a' -> 'YQ', '?' -> 'Pw' -> URL safe is 'Pw' but often involves '-' or '_'
    # Let's use a known urlsafe string: "subject?" -> "c3ViamVjdD8"
    assert base64_decode("c3ViamVjdD8") == b"subject?"
    
    # Test with padding stripped (the function handles reconstruction)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YQ==") == b"a"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with special URL-safe characters ( - and _ )
    # "_" is index 63 in urlsafe, "-" is index 62
    # Testing a sequence that results in these chars
    # b'\xff\xef' -> base64 '____' in standard, but urlsafe uses '-' '_'
    assert base64_decode("____") == b"\xff\xef"
    assert base64_decode("----") == b"\xfb\xbf"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test non-ascii characters in input (should be ignored due to errors="ignore")
    # 'SGVsbG8' + emoji
    assert base64_decode("SGVsbG8" + "🚀") == b"Hello"
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test with bytes input
    original_bytes = b"\x00\xff\xfe\x01"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe characters (specifically '-' and '_')
    # Base64 for '?' in standard is 'Pw==' -> urlsafe is 'Pw'
    # Using characters that trigger the alphabet change
    special_input = b"\xff\xef" 
    encoded_special = base64_encode(special_input)
    assert b"-" in encoded_special or b"_" in encoded_special or len(encoded_special) > 0
    assert base64_decode(encoded_special) == special_input

    # Test padding recovery (the function adds '=' automatically)
    # 'a' encoded is 'YQ==' -> rstrip makes it 'YQ'
    assert base64_decode("YQ") == b"a"
    assert base64_decode(b"YQ") == b"a"

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        # Using characters outside the valid base64 alphabet/logic 
        # that trigger ValueError in urlsafe_b64decode
        base64_decode("!!!")

    # Test with empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Test padding auto-recovery
    
    # Test bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test special characters in URL-safe alphabet ( - and _ )
    # "base64_encode" for "subject?param=value" results in "c3ViamVjdD9wYXJhbT12YWx1ZQ"
    assert base64_decode("c3ViamVjdD9wYXJhbT12YWx1ZQ") == b"subject?param=value"
    
    # Test case sensitivity and characters
    assert base64_decode("YWJjZA") == b"abcd"

    # Test error handling with invalid base64 data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding provided
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    
    # Test URL-safe characters (using '-' and '_' instead of '+' and '/')
    # 'a/b' in standard B64 is 'YS9i', in URL-safe it is 'YS1i'
    assert base64_decode("YS1i") == b"a/b"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid base64 data triggers BadData exception
    # Using a character outside the URL-safe alphabet (like space or special symbol not in set)
    # Note: urlsafe_b64decode is quite permissive, but characters like '!' will fail
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test case with varying lengths requiring different padding calculations
    assert base64_decode("YQ") == b"a"      # Needs ==
    assert base64_decode("YmI") == b"bb"   # Needs =
    assert base64_decode("Y2Ji") == b"cbb" # Needs no padding
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test basic functionality with standard string
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe character usage (replace + with - and / with _)
    # "data/test" encoded normally is "ZGF0YS90ZXN0" 
    # but urlsafe uses underscores.
    assert base64_decode("ZGF0YS10ZXN0") == b"data-test"
    
    # Test decoding without padding (the function handles padding internally)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test with bytes input instead of string
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test error handling for invalid base64 data
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test case sensitivity and alphabet boundaries
    assert base64_decode("YQ") == b"a"  # 'a' is YQ== (padding added)
    assert base64_decode("Yg") == b"b" # 'b' is Yg== (padding added)
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe specific characters (no padding needed)
    # '?' in urlsafe is usually encoded/handled via '-' and '_'
    assert base64_decode("YV9iLWNfZA") == b"a_b-c_d"
    
    # Test handling of missing padding (the function adds it automatically)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test with different input types (bytes vs str)
    assert base64_decode(b"d29ybGQ=") == b"world"
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test string input (URL safe)
    original_str = "testing 123"
    encoded_str = base64_encode(original_str)
    assert base64_decode(encoded_str) == original_str.encode("utf-8")

    # Test padding recovery (the function handles missing '=')
    # 'abc' in urlsafe b64 is 'YWJj' (no padding needed)
    # 'ab' in urlsafe b64 is 'YWI=' -> 'YWI' (padding stripped by encode)
    encoded_no_pad = b"YWI" 
    assert base64_decode(encoded_no_pad) == b"ab"

    # Test URL-safe characters (hyphen and underscore)
    # Using bytes that result in '-' and '_' in urlsafe b64
    url_safe_bytes = b"\xff\xef" # results in characters involving - or _
    encoded_url = base64_encode(url_safe_bytes)
    assert b"-" in encoded_url or b"_" in encoded_url
    assert base64_decode(encoded_url) == url_safe_bytes

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        # Using characters outside the valid alphabet/structure that trigger error
        base64_decode(b"!!!")

    # Test empty input
    assert base64_decode(b"") == b""
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from .exc import BadData

def test_base64_decode():
    # Test standard URL-safe encoding/decoding roundtrip
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test decoding with bytes input
    encoded_bytes = b"python-is-fun"
    decoded_bytes = base64_decode(encoded_bytes)
    assert decoded_bytes == b"python\xbe\x9a\xc5" # Note: direct manual b64 encoding check

    # Test decoding with padding missing (function should handle it via modulo logic)
    # "abc" in urlsafe b64 is "YWJj" (no padding needed)
    # "ab" in urlsafe b64 is "YWI=" -> stripped to "YWI"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode(b"YWI") == b"ab"

    # Test decoding with special characters (URL safe: - and _)
    # '-' is index 62, '_' is index 63
    special_bytes = b"\xff\xef\xbe"
    # Manual calculation for urlsafe: base64.urlsafe_b64encode(b'\xff\xef\xbe') -> b'__7-'
    encoded_special = base64_encode(b"\xff\xef\xbe")
    assert base64_decode(encoded_special) == b"\xff\xef\xbe"

    # Test decoding with empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!not_base64!!!")

    # Test that it ignores non-ascii characters in input due to errors="ignore" 
    # but still fails if the resulting structure is not valid b64
    # (Though 'ignore' prevents UnicodeDecodeError, urlsafe_b64decode handles the logic)
    invalid_input = "abc\u1234" 
    # The function uses encoding="ascii", errors="ignore". 
    # "\u1234" will be dropped. "abc" becomes b"abc", which is valid b64 for 'i'
    assert base64_decode("abc\u1234") == base64_decode("abc")
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == original_text.encode("utf-8")

    # Test with bytes input
    original_bytes = b"\x01\x02\x03\xff"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe characters (using '-' and '_')
    # 'abc?' in urlsafe is 'YWJjPw' -> but let's use a known pattern
    # b'\xff\xfe' encoded is '____' in some variants, 
    # specifically checking that '-' and '_' are handled by base64_decode
    special_chars = "testing-with_underscores"
    encoded_special = base64_encode(special_chars)
    assert b"-" in encoded_special or b"_" in encoded_special
    assert base64_decode(encoded_special).decode("utf-8") == special_chars

    # Test padding reconstruction (the function handles missing '=')
    # 'any' -> 'YW55' (length 4, no padding needed)
    # 'an' -> 'YW4=' (length 3, needs 1 padding)
    encoded_no_padding = b"YW4" 
    assert base64_decode(encoded_no_padding) == b"an"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        # Use characters outside the valid URL-safe base64 alphabet
        # (Note: urlsafe uses A-Z, a-z, 0-9, -, _)
        # Using '!' which is not in the alphabet and would fail decoding
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding cycle
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test with bytes input
    original_bytes = b"\x00\xff\xfe\x01"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe characters (specifically '-' and '_')
    # 'abc?' in base64 urlsafe is 'YWJjPw' but with padding logic check
    # Let's use a known string that results in '-' or '_'
    # base64.urlsafe_b64encode(b'\xfb') -> b'-_==' -> stripped: b'-_'
    encoded_special = b"-_"
    assert base64_decode(encoded_special) == b"\xfb"

    # Test string input for decode function
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test padding recovery (the function adds '=' back)
    # 'YQ' is 'a' without padding. Length 2 -> needs 2 '=' to make 4.
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == original_text.encode("utf-8")

    # Test with bytes input
    original_bytes = b"\x01\x02\x03\xff"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe characters (using - and _)
    # In standard b64, + and / are used. urlsafe uses - and _
    url_safe_input = b"\xff\xef" # results in encoded string with specific chars
    encoded_url = base64_encode(url_safe_input)
    assert b"+" not in encoded_url
    assert b"/" not in encoded_url
    assert base64_decode(encoded_url) == url_safe_input

    # Test padding recovery (the function handles missing =)
    no_padding = b"YW55IGNhcm5hbCBwbGVhc3VyZQ" # "any carnal pleasure" without padding
    assert base64_decode(no_padding) == b"any carnal pleasure"

    # Test decoding string input instead of bytes
    encoded_str = "SGVsbG8="
    assert base64_decode(encoded_str) == b"Hello"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode(b"!!!not_base64!!!")

    # Test empty input
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test URL-safe characters (substituting + and / with - and _)
    # "subjects?" in standard B64 is "c3ViamVjdHM/"
    # In urlsafe it is "c3ViamVjdHM_"
    assert base64_decode("c3ViamVjdHM_") == b"subjects?"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test different encodings/types
    assert base64_decode(b"Ym9i") == b"bob"

    # Test invalid data (should raise BadData)
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test non-ascii characters in input string 
    # The function uses encoding='ascii' with errors='ignore' for decoding
    # So 'abc\xff' becomes 'abc'
    assert base64_decode("YWJj\xff") == b"abc"
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from .exc import BadData

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original = "hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded).decode("utf-8") == original

    # Test with bytes input
    original_bytes = b"pytest testing"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe characters (using - and _)
    # "_" and "-" are part of the urlsafe alphabet
    special_chars = "testing-chars_123"
    encoded_special = base64_encode(special_chars)
    assert base64_decode(encoded_special).decode("utf-8") == special_chars

    # Test decoding without padding (the function handles padding internally)
    unpadded = "SGVsbG8" # "Hello" in b64 is SGVsbG8=, but we strip it
    assert base64_decode(unpadded).decode("utf-8") == "Hello"

    # Test decoding with input that is already bytes
    assert base64_decode(b"YmFzZTY0") == b"base64"

    # Test invalid data raises BadData
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        # Using characters outside the valid base64 alphabet (like !)
        base64_decode("!!!invalid!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (urlsafe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe specific characters (- and _)
    # "base64-" in urlsafe is "YmFzZTY0-" (with padding logic)
    # Let's use a known string: 'subject?' -> 'c3ViamVjdD8'
    assert base64_decode("c3ViamVjdD8") == b"subject?"
    
    # Test with different types of input (str and bytes)
    assert base64_decode(b"YmFzZTY0") == b"base64"
    
    # Test automatic padding handling
    # 'abc' needs '==' to be 'YWJj' -> length 4. 'abc' as base64 is 'YWJj'
    # If input is 'YQ' (a), len is 2, needs == to be 'YQ=='
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"

    # Test error handling for invalid data
    with pytest.raises(BadData):
        # Using characters outside the valid URL-safe alphabet/logic 
        # that trigger ValueError in urlsafe_b64decode
        base64_decode("!!!")

    # Test empty input
    assert base64_decode("") == b""
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replacing + and / with - and _)
    # "Subject?" in base64 is "U3ViamVjdD8=" -> URL safe "U3ViamVjdD8="
    # Let's use a case that specifically uses the '-' or '_'
    # b'\xff\xfe' in urlsafe is '__4='
    assert base64_decode("__4=") == b"\xff\xfe"
    
    # Test bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test missing padding (the function should handle it)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("V29ybGQ") == b"World"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!Invalid!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test bytes input
    original_bytes = b"\x00\xff\xfe\x12"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe specific characters (padding removal/restoration)
    # 'abc' in urlsafe b64 is 'YWJj'
    # 'abcd' in urlsafe b64 is 'YWJjZA' (no padding)
    encoded_no_padding = b"YWJjZA"
    assert base64_decode(encoded_no_padding) == b"abcd"

    # Test string input for decoding
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test error handling with invalid characters (though urlsafe handles most, 
    # we test the BadData exception logic if possible via structurally broken data)
    with pytest.raises(BadData):
        # Using a character that is not in the URL-safe alphabet and causes issues
        # Note: urlsafe_b64decode is quite lenient with whitespace/non-alphabet, 
        # but we trigger it by passing something that fails the underlying decode logic.
        base64_decode(b"!!!")

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test basic encoding/decoding roundtrip
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == original_text.encode("utf-8")

    # Test decoding with bytes input
    original_bytes = b"\x00\xff\xfe"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe specific characters (e.g., + and / replaced by - and _)
    # 'data?+' in standard b64 is 'ZGF0YT8r' -> urlsafe is 'ZGF0YT8n' (if using different chars)
    # Let's use a known case: bytes that result in '-' or '_'
    special_bytes = b"\xfb\xff" 
    encoded_special = base64_encode(special_bytes)
    assert b"-" in encoded_special or b"_" in encoded_special
    assert base64_decode(encoded_special) == special_bytes

    # Test padding reconstruction (function should handle missing '=')
    # 'abcd' -> 'YWJjZA==' (len 8). 'YWJjZA' (len 6) needs two '='
    truncated = b"YWJjZA"
    assert base64_decode(truncated) == b"abcd"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!NotBase64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding needed (original was "Hello" -> "SGVsbG8=")
    # The function handles missing '=' automatically
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test bytes input
    assert base64_decode(b"YmFzZTY0") == b"base64"
    
    # Test special characters in URL-safe alphabet (- and _)
    # 'a' + '/' becomes 'YS8=' in standard, but 'YS0_' or similar in urlsafe
    # Let's use a known urlsafe string: 'testing_123-abc'
    test_str = "testing_123-abc"
    encoded = base64_encode(test_str)
    assert base64_decode(encoded) == test_str.encode("utf-8")

    # Test invalid base64 data triggers BadData
    with pytest.raises(BadData):
        base64_decode("!!!NotBase64!!!")

    # Test empty string
    assert base64_decode("") == b""

    # Test case sensitivity/encoding handling (ASCII ignore)
    # The function uses encoding="ascii", errors="ignore"
    # Non-ascii characters should be stripped during decoding process
    # but the logic applies to the input string before decoding.
    assert base64_decode("SGVsbG8\xff") == b"Hello"
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (using - and _ instead of + and /)
    # Base64 for '\xfb\xff\xbe' is '+/++' in standard, '-_++' in urlsafe
    assert base64_decode("-_++") == b"\xfb\xff\xbe"
    
    # Test input as bytes
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test missing padding (function should handle it)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test string with different encoding/unusual characters
    assert base64_decode("py5m") == b"\xa9\xce\x66"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!Invalid!!!")
    
    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding cycle
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == original_text.encode("utf-8")

    # Test decoding with padding missing (the function handles it)
    # 'a' in urlsafe b64 is 'YQ==' -> 'YQ' is valid input for the logic
    assert base64_decode("YQ") == b"a"

    # Test decoding bytes input
    encoded_bytes = b"SGVsbG8=" # "Hello" (though function strips =)
    assert base64_decode(encoded_bytes) == b"Hello"

    # Test URL-safe characters specifically (- and _)
    # Using a string that results in '-' or '_'
    # '?' -> 'Pw' -> urlsafe is same, but let's use known chars
    # '_is' in urlsafe base64
    special_bytes = b"\xff\xef" 
    encoded_special = base64_encode(special_bytes)
    assert base64_decode(encoded_special) == special_bytes

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        # Using characters outside the base64 alphabet (e.g., non-ascii/invalid symbols)
        # Note: urlsafe_b64decode is quite lenient, but certain malformed 
        # structures or characters that cannot be decoded will trigger error.
        # We use a character that breaks the structure if possible.
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (no padding required)
    # "subject" in urlsafe is "c3ViamVjdA"
    assert base64_decode("c3ViamVjdA") == b"subject"
    
    # Test with different padding/length scenarios
    # 'a' -> 'YQ==' (len 4)
    # 'ab' -> 'YWI=' (len 4)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with URL-safe characters '-' and '_'
    # Using a known string that results in these chars
    # b'\xff\xef' -> '____' or similar depending on padding
    # Let's use a specific value: b'\xfb\xff' -> b'-_8' 
    # In urlsafe, '-' is index 62 and '_' is index 63.
    # base64_decode should handle the decoding of these specifically.
    assert base64_decode("-_8") == b"\xfb\xff"

    # Test error handling for invalid base64 data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test with non-ascii characters (should be ignored per encoding="ascii", errors="ignore")
    # If we pass a string that becomes invalid after stripping/ignoring:
    # "SGVsbG8" + unicode char that gets stripped
    assert base64_decode("SGVsbG8\u1234") == b"Hello"
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from .exc import BadData

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test with padding recovery (input without =)
    # "py" in base64 is "cHk=" -> "cHk"
    assert base64_decode("cHk") == b"py"
    
    # Test complex string with URL-safe characters (- and _)
    # "test_data-123" encoded: dGVzdF9kYXRhLTEyMw
    assert base64_decode("dGVzdF9kYXRhLTEyMw") == b"test_data-123"
    
    # Test with different encodings/types (bytes input)
    assert base64_decode(b"Ym90") == b"bot"

    # Test invalid data raises BadData
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe specific characters (replace + with - and / with _)
    # Standard: "i\xbe\xf3\xde" -> Base64: "i76z3g==" 
    # In URL safe, certain chars change, but urlsafe_b64encode uses '-' and '_'
    assert base64_decode("Y29uZ29fcmF0") == b"congo_rat"
    assert base64_decode("Y29uZ28tdmF0") == b"congo-vat"

    # Test missing padding (function should handle it via the modulo logic)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG") == b"Hel"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test bytes input with non-ascii characters handled by errors="ignore"
    # The function uses encoding="ascii", errors="ignore" internally
    assert base64_decode(b"SGVsbG8\xff") == b"Hello"
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters ( - and _ )
    # Base64 for "aa/bb" is "YWF/YmI=" -> URL safe "YWF_YmI="
    assert base64_decode("YWF_YmI") == b"aa/bb"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test padding handling (function should handle missing =)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YmI") == b"bb"
    assert base64_decode("Y21i") == b"cmi"

    # Test different encodings (input is treated as ascii for decoding)
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"invalid_base64_chars_#$")
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding cycle
    original_text = "hello world"
    encoded = base64_encode(original_text)
    assert base64_decode(encoded) == b"hello world"

    # Test with bytes input
    original_bytes = b"\x00\xff\x00\xff"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test urlsafe specific characters (handling '-' and '_')
    # 'abc?' in urlsafe base64 might involve these characters
    special_data = b"\xff\xfe\xfd\xfc"
    encoded_special = base64_encode(special_data)
    assert base64_decode(encoded_special) == special_data

    # Test decoding string input (not bytes)
    encoded_str = "SGVsbG8=" # 'Hello' with padding
    # Note: our implementation strips '=', but decode adds it back via padding logic
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test padding recovery (input without padding)
    # 'any' -> 'YW55' (no padding needed)
    # 'an' -> 'YW4=' (needs 1 padding)
    assert base64_decode("YW4") == b"an"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test URL-safe characters ( - and _ )
    # "abc/def" in urlsafe is "YWJjX2RlZg"
    assert base64_decode("YWJjX2RlZg") == b"abc_def"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test different encodings/types
    assert base64_decode("VGVzdA==") == b"Test"
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test input with non-ascii characters (should be ignored by encoding="ascii", errors="ignore")
    # "SGVsbG8=" is Hello, adding a unicode character that disappears in ascii ignore mode
    assert base64_decode("SGVsbG8=🔥") == b"Hello"
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (urlsafe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test URL-safe specific characters (- and _)
    # "a/b" in standard B64 is "YS9i", in URL-safe it is "YS1i"
    assert base64_decode("YS1i") == b"a/b"
    
    # Test different types of input (str and bytes)
    assert base64_decode(b"VGVzdA") == b"Test"
    
    # Test empty input
    assert base64_decode("") == b""
    
    # Test with padding already present
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")  # Non-alphabet characters that fail decoding logic
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding loop
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    assert base64_decode(encoded.decode("ascii")) == original

    # Test with URL-safe characters (padding removal check)
    # '?' in standard b64 is '_' in urlsafe
    special = b"\xff\xfe\xfd" 
    encoded_special = base64_encode(special)
    assert b"=" not in encoded_special
    assert base64_decode(encoded_special) == special

    # Test decoding string input
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test with different padding requirements (handling missing '=')
    # 'abc' -> 'YWJj' (no padding needed)
    # 'abcd' -> 'YWJjZA==' (needs padding)
    assert base64_decode("YWJj") == b"abc"
    assert base64_decode("YWJjZA") == b"abcd"

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!invalid!!!")

    # Test empty input
    assert base64_decode("") == b""
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original_text = "hello world"
    encoded = base64_encode(original_text)
    assert base64_decode(encoded) == b"hello world"

    # Test bytes input
    original_bytes = b"\x00\xff\xde\xad\xbe\xef"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe characters (handling '-' and '_')
    # 'abc?' in urlsafe is 'YWJjPw' -> but let's use a known pattern with - or _
    # Base64 for b'\xff\xfe' is '____' in some contexts, 
    # let's use a specific string that results in '-' or '_'
    special_bytes = b"\xfb\xff" # Resulting in urlsafe chars
    encoded_special = base64_encode(special_bytes)
    assert any(char in encoded_special for char in [b'-', b'_'])
    assert base64_decode(encoded_special) == special_bytes

    # Test padding recovery (input without '=')
    no_padding = b"YWJj" # "abc" is YWJj, if we strip it:
    stripped = base64.urlsafe_b64encode(b"abc").rstrip(b"=")
    assert base64_decode(stripped) == b"abc"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from .exc import BadData

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == b"hello world"

    # Test decoding with bytes input
    encoded_bytes = b"python-is-fun"
    # Note: manual creation of urlsafe b64 without padding
    # 'python-is-fun' -> base64 is 'cHl0aG9uLWlzLWZ1bg==' 
    # URL safe stripped: 'cHl0aG9uLWlzLWZ1bg'
    encoded_urlsafe = base64_encode(b"python-is-fun")
    assert base64_decode(encoded_urlsafe) == b"python-is-fun"

    # Test handling of padding (the function adds it automatically)
    padding_needed = b"abc" # 'abc' -> 'YWJj' (no padding needed)
    # Let's use a string that specifically requires padding
    # 'a' -> 'YQ==' -> 'YQ'
    assert base64_decode(b"YQ") == b"a"

    # Test handling of special characters in URL-safe alphabet
    special_bytes = b"testing_123-abc"
    encoded_special = base64_encode(special_bytes)
    assert base64_decode(encoded_special) == special_bytes

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        # Using characters outside the URL-safe base64 alphabet
        # (though urlsafe_b64decode is somewhat forgiving, 
        # very broken sequences or certain non-ascii can trigger it)
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input (URL-safe)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test bytes input
    assert base64_decode(b"V29ybGQ") == b"World"
    
    # Test with padding already present
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test complex characters (URL-safe specific: - and _)
    # "test-case_" in base64 urlsafe is "dGVzdC1jYXNlXw"
    assert base64_decode("dGVzdC1jYXNlXw") == b"test-case_"
    
    # Test empty input
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")

    # Test decoding of a known long string
    long_str = "python_is_awesome_123"
    encoded = base64_encode(long_str)
    assert base64_decode(encoded) == want_bytes(long_str)
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (using - and _ instead of + and /)
    # 'subject?' in base64 is 'c3ViamVjdD8=' 
    # Using urlsafe version: 'c3ViamVjdD8'
    assert base64_decode("c3ViamVjdD8") == b"subject?"
    
    # Test with bytes input
    assert base64_decode(b"Ym90") == b"bot"
    
    # Test padding stripping (the function handles missing padding)
    assert base64_decode("Ym90") == b"bot"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not-base64!!!")

    # Test edge case with specific URL-safe characters
    # '>>' is encoded as 'Pj4=' in standard, 'Pj4' in urlsafe
    assert base64_decode("Pj4") == b">>"
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from .exc import BadData

def test_base64_decode():
    # Test standard valid encoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test URL-safe specific characters (replace + with - and / with _)
    # Original: "subject?param=value" -> Base64: "c3ViamVjdD9wYXJhbT12YWx1ZQ=="
    # URL safe: "c3ViamVjdD9wYXJhbT12YWx1ZQ" (no padding needed in logic)
    assert base64_decode("c3ViamVjdD9wYXJhbT12YWx1ZQ") == b"subject?param=value"
    
    # Test with different input types (bytes)
    assert base64_decode(b"Ym9i") == b"bob"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test error handling for invalid base64 characters
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test edge case: valid but extremely long input (verifying padding logic)
    long_input = "a" * 100
    try:
        base64_decode(long_input)
    except BadData:
        pytest.fail("base64_decode raised BadData unexpectedly on valid-length input")

    # Test error handling for invalid structure (though urlsafe_b64decode is lenient, 
    # certain malformed inputs should trigger the exception via ValueError/TypeError)
    with pytest.raises(BadData):
        base64_decode(None) # type: ignore
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original = "hello world"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == want_bytes(original)

    # Test with bytes input
    original_bytes = b"\x00\xff\xfe\x01"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe characters (using '-' and '_' instead of '+' and '/')
    # '+' is 43, '/' is 47. In urlsafe, these are replaced by '-' and '_'
    url_safe_input = b"abc/def+"
    encoded_url_safe = base64_encode(url_safe_input)
    # Verify the output doesn't contain '+' or '/'
    assert b"+" not in encoded_url_safe
    assert b"/" not in encoded_url_safe
    assert base64_decode(encoded_url_safe) == url_safe_input

    # Test decoding with missing padding (the function handles padding internally)
    unpadded = b"SGVsbG8"  # "Hello" without '='
    assert base64_decode(unpadded) == b"Hello"

    # Test decoding with extra padding
    overpadded = b"SGVsbG8===="
    assert base64_decode(overpadded) == b"Hello"

    # Test invalid base64 data triggers BadData exception
    with pytest.raises(BadData):
        base64_decode(b"!!!not_base64!!!")

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (no padding)
    # 'a' in base64 is 'YQ=='
    assert base64_decode("YQ") == b"a"
    
    # Test complex string with urlsafe chars (- and _)
    # Example: bytes that result in '-' or '_'
    # Using a known urlsafe payload
    payload = b"\xff\xef" 
    encoded = base64_encode(payload)
    assert base64_decode(encoded) == payload

    # Test input as bytes instead of str
    assert base64_decode(b"V29ybGQ=") == b"World"

    # Test empty string
    assert base64_decode("") == b""

    # Test error handling with invalid data
    with pytest.raises(BadData):
        # Invalid characters for base64 (e.g., non-ascii or illegal symbols)
        # Note: urlsafe_b64decode is quite permissive, but specific 
        # structural failures should trigger BadData.
        base64_decode("!!!") 
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test with URL-safe characters (+/ replaced by -_)
    # 'subject?' in standard B64 is 'c3ViamVjdD8='
    # In urlsafe it uses '-' and '_'
    assert base64_decode("YV9i") == b"a_b" 
    
    # Test padding recovery (input without =)
    assert base64_decode("YW55") == b"any"  # len 4, no pad needed
    assert base64_decode("YW55YQ") == b"anya" # len 6 -> needs ==
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test with different input types (bytes)
    assert base64_decode(b"dGVzdA") == b"test"
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (using '-' instead of '+')
    # 'plus' in urlsafe is '-'
    assert base64_decode("Y29tZS13b3JsZA") == b"come-world" 
    
    # Test padding handling (the function adds '=' internally)
    assert base64_decode("YW55") == b"any"  # 'any' is 'YW55', needs no padding, but check logic
    assert base64_decode("YQ") == b"a"      # 'a' -> 'YQ==', function should handle it
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test different encodings/types
    assert base64_decode(b"Ynlic3Rlc3Q=") == b"bytestest"
```


