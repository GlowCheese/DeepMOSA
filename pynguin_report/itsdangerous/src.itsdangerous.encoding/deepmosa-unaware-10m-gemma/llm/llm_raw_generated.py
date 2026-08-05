####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (replace + with - and / with _)
    # "subject?" in standard b64 is "c3ViamVjdD8=" 
    # In urlsafe it's "c3ViamVjdD8=" (no change here, but testing the logic)
    assert base64_decode("Y29uY29y") == b"concor"
    
    # Test padding stripping/handling (the function adds padding back)
    assert base64_decode("YW55") == b"any" # "any" is 3 bytes, needs padding to 4
    assert base64_decode("YQ") == b"a"     # "a" is 1 byte, needs padding to 4
    
    # Test with different input types (bytes)
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")
        
    # Test with empty string
    assert base64_decode("") == b""

    # Test case where characters are outside ascii range (should be ignored via errors="ignore")
    # 'é' is non-ascii. If we pass it, the function should handle it without crashing 
    # as long as the resulting bytes are valid base64.
    # However, the specific implementation uses encoding="ascii" and errors="ignore".
    assert base64_decode("SGVsbG8") == b"Hello"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe specific characters (replace + with - and / with _)
    # "a/b+" in standard B64 is "YS9i+"
    # In URL-safe it should be "YS9i" or similar depending on padding
    assert base64_decode("YS1i") == b"a/b" # Using urlsafe logic
    
    # Test with different input types (bytes)
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test padding recovery (input without =)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("YQ") == b"a"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test error handling with invalid base64 characters
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test encoding/decoding roundtrip
    original = "Python_is_Awesome_123"
    encoded = base64_encode(original)
    decoded = base64_decode(encoded)
    assert decoded == want_bytes(original)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe specific characters (- and _)
    # "test_data" in urlsafe b64 is "dGVzdF9kYXRh"
    assert base64_decode("dGVzdF9kYXRh") == b"test_data"
    
    # Test missing padding (the function should handle it)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!Invalid!!!")

    # Test with different input types (bytes)
    assert base64_decode(b"YmFzZTY0") == b"base64"
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test basic functionality (standard string)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test with bytes input
    assert base64_decode(b"V29ybGQ") == b"World"
    
    # Test URL-safe characters (using - and _)
    # "a/b" in standard B64 is "YS9i", in URL-safe it is "YS1i"
    assert base64_decode("YS1i") == b"a/b"
    
    # Test automatic padding handling
    # "abc" needs 1 '=' to be length 4
    assert base64_decode("YWJj") == b"abc"
    # "abcd" needs no padding
    assert base64_decode("YWJjZA") == b"abcd"

    # Test with different encodings (via want_bytes logic)
    assert base64_decode("4pyo") == b"\xe2\x9c\x94"  # Checkmark symbol

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test empty input
    assert base64_decode("") == b""
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (replace + with - and / with _)
    # "a+b/c" in standard B64 is "YStiL2M="
    # In urlsafe it is "YStiL2M=" (no change here, but check hyphen/underscore)
    # Let's use a known urlsafe specific case:
    assert base64_decode("YV9i") == b"a_b"
    assert base64_decode("YV8") == b"a_"
    
    # Test padding-less input (the function handles adding '=' automatically)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"

    # Test bytes input
    assert base64_decode(b"Ym9i") == b"bob"

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test edge case: empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (no padding needed)
    # 'base64' in urlsafe is 'YmFzZTY0'
    assert base64_decode("YmFzZTY0") == b"base64"
    
    # Test with missing padding (the function handles padding internally)
    assert base64_decode("YmFzZTY0") == b"base64"
    assert base64_decode("SGVsbG8") == b"Hello"

    # Test hyphen and underscore (urlsafe specific)
    # '-' and '_' represent 62 and 63 in urlsafe alphabet
    # '_-' is part of a valid sequence
    valid_urlsafe = base64_encode(b"\xff\xfe\xfd")
    assert base64_decode(valid_urlsafe) == b"\xff\xfe\xfd"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Test padding logic
    
    # Test bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test URL-safe characters (hyphen and underscore)
    # "a/b+" in standard B64 is "Yy9i+"
    # In URL-safe, it is "Yy1i_" (with padding stripped)
    assert base64_decode("Yy1i") == b"c/b" # Note: base64.urlsafe uses - and _ instead of + and /
    
    # Test complex string
    input_str = "some_string_with_-chars"
    encoded = base64_encode(input_str)
    assert base64_decode(encoded).decode() == input_str

    # Test error case with invalid characters (not in alphabet and not recoverable)
    # Note: urlsafe_b64decode is quite permissive, but non-alphabet chars 
    # that trigger ValueError/TypeError should raise BadData.
    with pytest.raises(BadData):
        # Using a character that makes the structure invalid for decoding logic
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard functionality
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test URL-safe characters (using - and _ instead of + and /)
    # "A/B" in standard B64 is "QS9C", in urlsafe it's "QS1C" (depending on specific chars)
    # Let's use a known pattern: bytes [253, 255] -> b'\xfd\xff'
    # Standard: +/ -> b'\xFB\xFF'
    # URL-safe: -_ -> b'\xFB\xFF'
    assert base64_decode("-_8") == b"\xfb\xff"

    # Test empty input
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test various lengths to trigger padding logic
    assert base64_decode("YQ") == b"a"      # len 2 -> needs ==
    assert base64_decode("YmI") == b"bb"   # len 3 -> needs =
    assert base64_decode("Y2Ji") == b"cbb" # len 4 -> needs nothing

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")  # Non-base64 characters that cannot be parsed

    # Test encoding/errors handling (ASCII requirement)
    # Since the function uses encoding="ascii", errors="ignore", 
    # non-ascii bytes should be stripped without crashing.
    assert base64_decode("SGVsbG8\xff") == b"Hello"
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding (the function handles missing padding)
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (hyphen and underscore)
    # "base64-_" encoded is "YmFzZTY0LS1f"
    assert base64_decode("YmFzZTY0LS1f") == b"base64-+_"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        # Using a character outside the URL-safe alphabet/valid range if possible, 
        # but since urlsafe_b64decode is quite permissive, we trigger via error types.
        # Specifically, providing something that cannot be processed as base64.
        base64_decode("!!!") 
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test URL-safe characters (hyphen and underscore)
    # "a/b?" in standard B64 is "YS9iPw==" -> URL safe is "YS1iPw"
    assert base64_decode("YS1iPw") == b"a/b?"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test complex encoding
    original = b"\x00\xff\xfe\x01\x02\x03"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!NotBase64!!!")
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (- and _)
    # "a/b+" in standard B64 is "YS9i+"
    # In URL-safe it should be "YS1i_" (using - and _)
    assert base64_decode("YS1i") == b"a/b" # Note: logic handles padding automatically
    
    # Test different types of input (str vs bytes)
    assert base64_decode("YW55IGNhcm5hbCBwbGVhc3VyZS4=") == b"any carnal pleasure."
    assert base64_decode(b"YW55IGNhcm5hbCBwbGVhc3VyZS4=") == b"any carnal pleasure."

    # Test padding recovery (the function adds missing '=')
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test empty input
    assert base64_decode("") == b""
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from .exc import BadData

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (dash and underscore)
    # 'a_b-' in base64 is part of a larger sequence, let's use a known one
    # '_' and '-' are used instead of '+' and '/'
    assert base64_decode("YV9iLWM") == b"a_b-c" # Note: padding handles automatically
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test with different types (bytes input)
    assert base64_decode(b"V29ybGQ=") == b"World"

    # Test padding auto-correction
    # 'abc' -> 3 chars, needs 1 '=' to be 4 chars
    assert base64_decode("YWJj") == b"abc" 
    
    # Test invalid data
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test non-ascii characters in input (should be ignored via errors="ignore")
    # 'SGVsbG8=' with a trailing emoji
    assert base64_decode("SGVsbG8=🚀") == b"Hello"
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe, no padding)
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding provided
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    
    # Test with bytes input
    assert base64_decode(b"YmFzZTY0") == b"base64"
    
    # Test special characters (URL-safe '-' and '_')
    # '_' -> 'fQ==' or similar. Let's use a known sequence:
    # 'test_data' encoded URL-safe is 'dGVzdF9kYXRh'
    assert base64_decode("dGVzdF9kYXRh") == b"test_data"
    
    # Test empty string
    assert base64_decode("") == b""

    # Test invalid data (not base64 characters)
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test error with non-ascii input that leads to decoding failure
    # Note: want_bytes uses ascii/ignore for decode, so it might strip them 
    # before reaching the try block. We test characters that break structure.
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        # Using a sequence that is technically invalid even after padding logic
        base64_decode("\x00\x01\x02") 
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe encoding (using '-' instead of '+')
    # 'plus' in base64 is '+' but urlsafe uses '-'
    assert base64_decode("YW55LWNvbmZpZ3VyYXRpb24=") == b"any-configuration"
    
    # Test input without padding (function should handle it)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test with non-ascii characters that should be ignored due to errors='ignore'
    # 'abc' + unicode char + 'def'
    # The function uses encoding='ascii', errors='ignore' for the input decoding phase
    assert base64_decode("YWJj\u1234ZGVm") == b"abcdef"
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe encoding)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe specific characters (no padding needed in logic, but testing content)
    # '?' becomes '?' in standard, but urlsafe uses '-' and '_'
    # Let's use a known string that results in _ and -
    # "subject?test" -> base64 -> "c3ViamVjdD90ZXN0" (standard)
    # urlsafe: "c3ViamVjdD90ZXN0" 
    # Let's use a simpler one: b'\xff\xfe' -> urlsafe is '__4='
    assert base64_decode("__4") == b"\xff\xfe"

    # Test with different input types (bytes)
    assert base64_decode(b"Ym9i") == b"bob"

    # Test case sensitivity/padding logic
    # The function handles missing padding via: string += b"=" * (-len(string) % 4)
    assert base64_decode("Ym9i") == b"bob"  # len 4, no pad needed
    assert base64_decode("Ym9") == b"bo"    # len 3, needs 1 pad

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")  # Non-base64 characters that cause ValueError in urlsafe_b64decode

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
    
    # Test URL-safe characters (no padding needed)
    assert base64_decode("YV9i") == b"a_b"
    
    # Test input as bytes
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test automatic padding handling (missing '=' characters)
    assert base64_decode("YQ") == b"a"  # 'YQ' needs '==' to be 'YQ=='
    assert base64_decode("YWI") == b"ab" # 'YWI' needs '=' to be 'YWI='
    
    # Test with different types of input (str vs bytes)
    assert base64_decode("YW55IGNhcm5hbCBwbGVhc3VyZS4=") == b"any carnal pleasure."
    
    # Test error case: Invalid base64 characters/data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        # Using an invalid character that breaks the decoding logic
        base64_decode("!!!")

    # Test edge case: empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with input as bytes
    assert base64_decode(b"YmFzZTY0") == b"base64"
    
    # Test case with padding required (input length 3)
    # "abc" -> "YWJj"
    assert base64_decode("YWJj") == b"abc"
    
    # Test URL-safe characters (dash and underscore)
    # '-' is 62, '_' is 63 in urlsafe
    # Using a known string that uses these chars
    assert base64_decode("-_") == b"\xff\xef"

    # Test with extra padding already present
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"

    # Test invalid data raises BadData
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (urlsafe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test URL-safe specific characters
    # '?' in urlsafe is often represented by different chars depending on implementation, 
    # but standard urlsafe uses '-' and '_' instead of '+' and '/'
    assert base64_decode("-_") == b"\xfb\xff" 
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test different encodings/types
    assert base64_decode(b"Ym9i") == b"bob"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")

    # Test with padding-less input (the function handles it)
    assert base64_decode("Ym9i") == b"bob" # 'Ym9i' is 4 chars, no padding needed
    assert base64_decode("Ym9") == b"bo"  # 'Ym9' needs one '=' to be 'Ym9='
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replace + with - and / with _)
    # "a+b/c" in urlsafe is "a-b_c"
    assert base64_decode("a-b_c") == b"a+b/c"
    
    # Test bytes input
    assert base64_decode(b"VGVzdA") == b"Test"
    
    # Test missing padding (the function handles padding automatically)
    assert base64_decode("Ym9v") == b"boo"  # "Ym9v" is 4 chars, no padding needed
    assert base64_decode("Ym9") == b"bo"   # "Ym9" needs 1 '='
    
    # Test with unexpected characters (should ignore based on encoding="ascii", errors="ignore")
    # Note: urlsafe_b64decode might still fail if non-base64 chars are present, 
    # but the function handles padding and ascii conversion.
    assert base64_decode("SGVsbG8\n") == b"Hello"

    # Test invalid base64 data raises BadData
    with pytest.raises(BadData):
        # Using a character that is not in the URL-safe alphabet and cannot be part of valid sequence
        # While 'ignore' handles encoding, urlsafe_b64decode will raise ValueError for bad sequences
        base64_decode("!!!") 

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #20
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
    original_bytes = b"\x00\xff\x00\xff"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe specific characters (no padding needed in source)
    # 'abc' -> 'YWJj'
    # 'abcd' -> 'YWJjZA' (stripped '=')
    assert base64_decode("YWJjZA") == b"abcd"
    
    # Test decoding with string input instead of bytes
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test handling of invalid base64 data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!invalid!!!")

    # Test decoding empty string
    assert base64_decode("") == b""

    # Test with different padding scenarios (the function handles padding automatically)
    # 'a' -> 'YQ==' -> stripped is 'YQ'
    assert base64_decode("YQ") == b"a"
    # 'ab' -> 'YWI=' -> stripped is 'YWI'
    assert base64_decode("YWI") == b"ab"
```


# LLM-generated content at query #21
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
    original_bytes = b"\x00\xff\xde\xad\xbe\xef"
    encoded_bytes = base64_encode(original_bytes)
    decoded_bytes = base64_decode(encoded_bytes)
    assert decoded_bytes == original_bytes

    # Test URL-safe characters (specifically '-' and '_')
    # 'abc/' in standard b64 is 'YWJjLw==' -> urlsafe is 'YWJjXw'
    url_safe_input = "YWJjXw" 
    assert base64_decode(url_safe_input) == b"abc/"

    # Test padding restoration (input without '=')
    no_padding = "YWJj" # Represents 'abc' but missing padding
    assert base64_decode(no_padding) == b"abc"

    # Test with different string types
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test error handling for invalid data
    with pytest.raises(BadData):
        base64_decode("!!!Invalid!!!")

    # Test empty input
    assert base64_decode("") == b""
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test urlsafe specific characters (replace + and / with - and _)
    # "subject?" in base64 is "c3ViamVjdD8="
    # Using URL safe version
    assert base64_decode("c3ViamVjdD8") == b"subject?"

    # Test input as bytes
    assert base64_decode(b"YW55IGNhcm5hbCBwbGVhc3VyZS4=") == b"any carnal pleasure."

    # Test padding handling (the function adds missing '=' automatically)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("Yg") == b"b"
    assert base64_decode("Yw") == b"c"

    # Test error handling for invalid data
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""

    # Test complex characters
    input_str = "testing-123_abc"
    encoded = base64_encode(input_str)
    assert base64_decode(encoded) == want_bytes(input_str)
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (replacing + with - and / with _)
    # "a+b/c" in urlsafe is "a-b_c"
    assert base64_decode("YS1iX2M=") == b"a-b_c"
    
    # Test padding handling (function adds missing =)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    
    # Test different types of input
    assert isinstance(base64_decode("SGVsbG8="), bytes)
    assert isinstance(base64_decode(b"SGVsbG8="), bytes)

    # Test error handling for invalid base64 data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test with empty input
    assert base64_decode("") == b""
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (hyphen and underscore)
    # "base64-_" encoded with urlsafe is "YmFzZTY0LS1f"
    assert base64_decode("YmFzZTY0LS1f") == b"base64-_"
    
    # Test input as bytes
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test missing padding (the function should handle it)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("YmFzZTY0LS1f") == b"base64-_"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test different encodings/input types via want_bytes logic internally
    assert base64_decode("123") == b"123"
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (no padding needed)
    # "subject?" in urlsafe is "c3ViamVjdD8"
    assert base64_decode("c3ViamVjdD8") == b"subject?"
    
    # Test with missing padding (the function should handle it)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid base64 data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test with non-ascii characters (should be ignored due to errors='ignore')
    # 'SGVsbG8=' is 'Hello', adding a unicode character that becomes invalid in ascii
    assert base64_decode("SGVsbG8=🚀") == b"Hello"
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from .exc import BadData

def test_base64_decode():
    # Test standard case (urlsafe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Test automatic padding logic
    
    # Test URL-safe specific characters (+ -> -, / -> _)
    # Standard B64 for "\xff\xfe" is "/v4="
    # URL-safe B64 for "\xff\xfe" is "_v4="
    assert base64_decode("_v4") == b"\xff\xfe"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test different encodings/types
    assert base64_decode("YmFzZTY0") == b"base64"

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        # Using characters outside the urlsafe alphabet/invalid sequences
        base64_decode("!!!")

    # Test complex string with padding requirements
    # "abcde" -> "YWJjZGU=" (Length 7, needs 1 '=')
    assert base64_decode("YWJjZGU") == b"abcde"
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replacing + and / with - and _)
    # "subject?" in urlsafe is "c3ViamVjdD8"
    assert base64_decode("c3ViamVjdD8") == b"subject?"
    
    # Test bytes input
    assert base64_decode(b"Ym95") == b"boy"
    
    # Test missing padding (the function handles padding automatically)
    assert base64_decode("Ym95") == b"boy"
    assert base64_decode("c3ViamVjdD") == b"subject" # incomplete but handled by padding logic
    
    # Test with different types of input (string vs bytes)
    assert base64_decode("YW55IGNhcm5hbCBwbGVhc3VyZS4=") == b"any carnal pleasure."
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (dash and underscore)
    # 'a' + '/' in standard is 'YQ=='
    # In urlsafe, '-' replaces '+' and '_' replaces '/'
    assert base64_decode("YQ--") == b"a\xff" # Example of padding/special chars
    
    # Test different types (str vs bytes)
    assert base64_decode("dGVzdA") == b"test"
    assert base64_decode(b"dGVzdA") == b"test"

    # Test with padding required by the function logic
    # 'abcd' -> 'YWJjZA==' (len 8)
    # 'abc' -> 'YWJj' (len 4)
    assert base64_decode("YWJj") == b"abc"
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe specific characters (no padding)
    # "a/b+" in standard is "YS9i+"
    # In urlsafe, "/" becomes "_" and "+" becomes "-"
    assert base64_decode("YS1i") == b"a/b" # 'YS1i' decodes to bytes representing 'a/b' logic
    
    # Test with padding missing (function should handle it)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test decoding of complex string
    original = b"testing_123_special_chars_!@#"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding needed
    assert base64_decode("YW55IGNhcm5hbCBwbGVhc3VyZS4") == b"any carnal pleasure."
    
    # Test bytes input
    assert base64_decode(b"YmFzZTY0") == b"base64"
    
    # Test URL-safe characters (dash and underscore)
    # 'a-b_' in urlsafe is different from standard '+' and '/'
    assert base64_decode("-_") == b"\xfb\xff"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test case sensitivity/encoding via want_bytes logic (ASCII)
    assert base64_decode("VGVzdA") == b"Test"
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (urlsafe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Test padding recovery
    
    # Test urlsafe characters (- and _)
    # "aa/bb" in standard B64 is "YWEvYmI=" 
    # In urlsafe, '/' becomes '_'
    assert base64_decode("YWFfYmI") == b"aab_b".replace(b"_", b"/") # Logic check for URL safe
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test various input types
    assert base64_decode("YW55IGNhcm5hbCBwbGVhc3VyZS4=") == b"any carnal pleasure."
    
    # Test with invalid characters (should ignore based on 'errors="ignore"' in code)
    # The function uses encoding='ascii' and errors='ignore' for the input string
    assert base64_decode("SGVsbG8$") == b"Hello"

    # Test error raising for truly invalid base64 structure
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!") # Non-base64 characters that fail decoding logic
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (using - and _ instead of + and /)
    # Base64 for '\xff\xfe' is '//4=' -> urlsafe is '__4='
    assert base64_decode("__4") == b"\xff\xfe"
    
    # Test without padding (the function handles adding '=' manually)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test bytes input instead of string
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test case with extra padding (should not break)
    assert base64_decode("SGVsbG8=====") == b"Hello"
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (urlsafe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test with urlsafe characters (- and _)
    # "abc" in base64 is "YWJj"
    # Special chars test: input bytes that result in '-' or '_'
    # b'\xff\xef' -> '____' in some variants, but let's use known urlsafe mapping
    # b'\xfa\xff' maps to urlsafe specific chars
    assert base64_decode("-_") == base64.urlsafe_b64decode(b"-_" + b"==")

    # Test padding recovery (the function adds missing '=')
    assert base64_decode("YQ") == b"a"  # YQ==
    assert base64_decode("YmI") == b"bb" # YmI=

    # Test with string input
    assert base64_decode("dGVzdA") == b"test"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test empty input
    assert base64_decode("") == b""
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replace + with - and / with _)
    # "subject?" in base64 is "c3ViamVjdD8="
    # URL safe version should work without padding
    assert base64_decode("c3ViamVjdD8") == b"subject?"
    
    # Test bytes input
    assert base64_decode(b"Ym9i") == b"bob"
    
    # Test with different padding requirements (automatic padding logic)
    assert base64_decode("YW55") == b"any"  # length 4, no padding needed
    assert base64_decode("YW55YQ") == b"anya" # length 6, needs '=='
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test urlsafe specific characters (replace + with - and / with _)
    # "a+b/" in urlsafe is "a-b_"
    assert base64_decode("a-b_") == b"a+b/"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test handling of missing padding (the function adds it automatically)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("YQ") == b"a"  # "YQ=="
    
    # Test with different encoding/ascii-safe characters
    assert base64_decode("abc") == b"abc" # Note: 'abc' padding makes it 'abc=' -> b'i\xb7' is wrong, 
    # Actually 'abc' (3 chars) needs 1 '=' to be 4. base64.urlsafe_b64decode(b'abc=') is b'i\xb7'
    # Let's use a known valid one:
    assert base64_decode("YW55") == b"any"
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #36
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding cycle
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == original_text.encode("utf-8")

    # Test with bytes input
    original_bytes = b"\x00\xff\x00\xff"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test handling of missing padding (the function adds it manually)
    # "any" in urlsafe b64 is "YW55" (length 4, no padding needed)
    # "an" in urlsafe b64 is "YW4=" -> stripped to "YW4"
    encoded_no_padding = b"YW4" 
    assert base64_decode(encoded_no_padding) == b"an"

    # Test with URL-safe characters (hyphen and underscore)
    # Base64 for '+' is '-' and '/' is '_' in urlsafe
    url_safe_input = b"test+data/with_special_chars"
    encoded_url = base64_encode(url_safe_input)
    assert b"+" not in encoded_url
    assert b"/" not in encoded_url
    assert base64_decode(encoded_url) == url_safe_input

    # Test invalid base64 data raises BadData
    # Using a character outside the valid alphabet (like a space or non-b64 char)
    # Note: urlsafe_b64decode is quite permissive with some characters, 
    # but certain structural violations trigger errors.
    with pytest.raises(BadData):
        # Passing something that cannot be decoded as base64
        # We use a character that causes ValueError in the underlying library
        base64_decode(b"!!!")

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
```


# LLM-generated content at query #37
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe alphabet (using - and _ instead of + and /)
    # "subject?foo=bar" in urlsafe base64 is "c3ViamVjdD9mb289YmFy"
    assert base64_decode("c3ViamVjdD9mb289YmFy") == b"subject?foo=bar"
    
    # Test with padding missing (the function handles adding =)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test input as bytes
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test edge case with specific URL-safe characters
    # Base64 for bytes [251, 191, 191] is "vz//", in urlsafe it is "vz__"
    assert base64_decode("vz__") == b"\xfb\xbf\xbf"
```


# LLM-generated content at query #38
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (replace + with - and / with _)
    # Standard: "i/8=" -> URL-safe: "i_8="
    assert base64_decode("i_8") == b"\x8b\xff"
    
    # Test different types of input (str vs bytes)
    assert base64_decode(b"YmFzZTY0") == b"base64"
    
    # Test padding recovery logic
    # "YQ" is length 2, needs two "=" to be "YQ=="
    assert base64_decode("YQ") == b"a"
    
    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""

    # Test alphanumeric and special symbols in alphabet
    # "abc123_-" 
    input_str = "YWJjMTIzXy1f"
    expected = b"abc123_-"
    assert base64_decode(input_str) == expected
```


# LLM-generated content at query #39
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (hyphen and underscore)
    # 'base64' encoded with urlsafe becomes 'YmFzZTY0'
    # Testing specifically the mapping of + to - and / to _
    assert base64_decode("YV9i") == b"a_b"  # '_' is part of alphabet
    assert base64_decode("YV-i") == b"a\xbe" # '-' used in urlsafe
    
    # Test padding handling (function adds missing =)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    
    # Test different encodings/types
    assert base64_decode(b"\x53\x6f\x6d\x65") == b"Some"
    
    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #40
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

    # Test URL-safe characters (replacing + with - and / with _)
    # Standard b64 for "\xfb\xff" is "+/++" -> urlsafe is "-_++"
    url_safe_input = b"-_+"
    # Manually constructed valid urlsafe string without padding
    # 'a' -> 'YQ' (no padding needed)
    assert base64_decode("YQ") == b"a"
    
    # Test decoding with implicit padding recovery
    # 'YQ' is length 2, needs two '=' to be length 4
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YW55") == b"any"

    # Test error handling for invalid data
    with pytest.raises(BadData):
        # Using characters outside the URL-safe alphabet or corrupted sequences
        base64_decode("!!!")

    # Test with different types (str vs bytes)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
```


# LLM-generated content at query #41
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replacing + with - and / with _)
    # "Subject?" in standard B64 is "U3ViamVjdD8="
    # In urlsafe it remains same if no + or / are present, 
    # but let's test a known conversion.
    # Example: \xff\xfe becomes something with - or _
    assert base64_decode("-_8=") == b"\xfb\xff"

    # Test missing padding (function should handle it)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test with various input types
    assert isinstance(base64_decode("YQ=="), bytes)
    assert isinstance(base64_decode(b"YQ=="), bytes)

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test with characters outside base64 alphabet (should ignore via ascii/ignore)
    # Note: the implementation uses errors="ignore" for the initial encoding step, 
    # but urlsafe_b64decode will still fail if it encounters non-alphabet chars.
    with pytest.dumps(BadData):
        base64_decode("invalid@charset")
```


# LLM-generated content at query #42
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding characters present (should handle both padded and unpadded)
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    
    # Test URL-safe specific characters (- and _)
    # "a/b" in standard b64 is "YS9i", in urlsafe it is "YS1i"
    assert base64_decode("YS1i") == b"a/b"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")

    # Test edge case: single character (not valid b64 on its own)
    # The function adds padding, but the underlying decoder will fail if content is corrupted
    with pytest.raises(BadData):
        base64_decode("?")
```


# LLM-generated content at query #43
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding already present
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test special characters in URL-safe alphabet (- and _)
    # "a-b_" in base64 is "YS1iXw=="
    assert base64_decode("YS1iXw") == b"a-b_"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test case sensitivity/non-ascii handling (should ignore non-ascii via errors="ignore")
    # 'abc' + emoji -> the emoji is ignored, leaving 'abc'
    assert base64_decode("YWJj\U0001F600") == b"abc"
```


# LLM-generated content at query #44
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
    original_bytes = b"\x00\xff\xde\xad\xbe\xef"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe specific characters (replacing + and / with - and _)
    # Standard b64 for \xfb\xff is "+/8=" -> URL safe is "-_8"
    url_safe_input = b"-_8" 
    assert base64_decode(url_safe_input) == b"\xfb\xff"

    # Test padding handling (the function adds missing padding internally)
    # "abc" encoded is "YWJj" (no padding needed)
    # "ab" encoded is "YWI=" -> stripped to "YWI" in base64_encode
    assert base64_decode("YWI") == b"ab"

    # Test with string input instead of bytes
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #45
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test with padding removed (the function handles manual padding)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (- and _)
    # "a/b+" in standard B64 is "YS9i+"
    # In URL-safe, it is "YS1i" (depending on implementation)
    # Let's use a known pair: 0xFB (251) -> '++8=' in std, '-_8' in urlsafe
    assert base64_decode("-_8") == b"\xff\xbf"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test non-ascii characters in input (should be ignored due to errors="ignore")
    # 'abc' + unicode char
    assert base64_decode("YWJj\u1234") == b"abc"
```


# LLM-generated content at query #46
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (hyphen and underscore instead of + and /)
    # "a\xbf" in standard B64 is "Yv8=" -> In URL safe it's "Yv8"
    assert base64_decode("Yv8") == b"a\xbf"
    
    # Test different padding requirements (automatic padding)
    assert base64_decode("YW55") == b"any"  # No padding needed
    assert base64_decode("YW55YQ") == b"anya" # Needs 2 '='
    assert base64_decode("YW55YWJj") == b"anyabc" # Needs 1 '='
    
    # Test input types (bytes vs str)
    assert base64_decode(b"Ym9i") == b"bob"
    
    # Test error handling for invalid characters/data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")
        
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode(b"\xff\xff\xff")
```


# LLM-generated content at query #47
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
    original_bytes = b"\x00\xff\xde\xad\xbe\xef"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe characters (using - and _)
    # '_' is index 63, '-' is index 62 in urlsafe
    special_chars = b"\xff\xef" # results in specific padding/chars
    encoded_special = base64_encode(special_chars)
    assert b"-" in encoded_special or b"_" in encoded_special
    assert base64_decode(encoded_special) == special_chars

    # Test decoding without padding (the function handles padding internally)
    unpadded_encoded = b"SGVsbG8" # "Hello" without '='
    assert base64_decode(unpadded_encoded) == b"Hello"

    # Test decoding with existing padding
    padded_encoded = b"SGVsbG8="
    assert base64_decode(padded_encoded) == b"Hello"

    # Test error handling for invalid base64 data
    invalid_data = b"!@#$%^&*"
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode(invalid_data)

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
```


# LLM-generated content at query #48
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test URL-safe characters (- and _)
    # 'base64' in urlsafe is 'YmFzZTY0'
    # Using a case that produces '-' or '_'
    # Example: bytes [251, 255] -> b'-_7=' -> rstrip '=' -> b'-_7'
    assert base64_decode("-_7") == b"\xfb\xff"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test with different encodings/types
    assert base64_decode("V29ybGQ=") == b"World"

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode(b"not_base64_@#$")
```


# LLM-generated content at query #49
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replace + with - and / with _)
    # "Subject?" in standard B64 is "U3ViamVjdD8="
    # In urlsafe it might involve different chars if the input was larger
    assert base64_decode("U3ViamVjdD8") == b"Subject?"
    
    # Test padding stripping (the function handles missing padding)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("Ym") == b"b"  # This would be invalid, but 'Ym' is 'b' + padding
    assert base64_decode("YmI=") == b"bb"
    
    # Test input with different types
    assert isinstance(base64_decode("YmI"), bytes)
    assert isinstance(base64_decode(b"YmI"), bytes)

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")
    
    with pytest.raises(BadData):
        base64_decode(b"invalid_chars_@#$")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #50
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding cycle
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == original_text.encode("utf-8")

    # Test decoding without padding (the function handles padding internally)
    padding_less = base64.urlsafe_b64encode(b"test").rstrip(b"=")
    assert base64_decode(padding_less) == b"test"

    # Test decoding bytes input
    encoded_bytes = b"dGVzdA"  # "test" without padding
    assert base64_decode(encoded_bytes) == b"test"

    # Test decoding with different characters (URL safe check)
    # '-' and '_' are used in urlsafe instead of '+' and '/'
    special_chars = b"\xff\xfe\xfd"
    encoded_special = base64_encode(special_chars)
    assert base64_decode(encoded_special) == special_chars

    # Test error handling for invalid base64 data
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test empty input
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
```


# LLM-generated content at query #51
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding required (urlsafe_b64encode strips it)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YmI") == b"bb"
    
    # Test with bytes input
    assert base64_decode(b"YW55IGNhcm5hbCBwbGVhc3VyZS4=") == b"any carnal pleasure."
    
    # Test with special URL-safe characters (- and _)
    # "test-with_special" encoded is "dGVzdC13aXRoX3NwZWNpYWw=" -> stripped: "dGVzdC13aXRoX3NwZWNpYWw"
    assert base64_decode("dGVzdC13aXRoX3NwZWNpYWw") == b"test-with_special"

    # Test invalid data raises BadData
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!not_base64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #52
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
    original_bytes = b"\x00\xff\xde\xad\xbe\xef"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe specific characters (substituting + and / with - and _)
    # "+" becomes "-" and "/" becomes "_" in urlsafe
    url_safe_data = b"\xfb\xff\xbf" 
    # Standard base64 for this is "+/+/". urlsafe is "-_-_"
    encoded_url = base64_encode(url_safe_data)
    assert b"+" not in encoded_url
    assert b"/" not in encoded_url
    assert base64_decode(encoded_url) == url_safe_data

    # Test decoding string input directly
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test padding recovery (the function handles missing =)
    no_padding = "SGVsbG8"  # Missing '='
    assert base64_decode(no_padding) == b"Hello"

    # Test invalid data raises BadData
    invalid_data = "!!!" # Not valid base64 characters in this context or malformed
    with pytest.raises(BadData):
        base64_decode(invalid_data)

    # Test empty string
    assert base64_decode("") == b""
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (replace + with - and / with _)
    # Standard: "a+b/c" -> "Yiticy==" 
    # URL-safe: "a-b_c" -> "YS1iX2M="
    assert base64_decode("YS1iX2M") == b"a-b_c"
    
    # Test different encoding/padding scenarios
    assert base64_decode("YQ==") == b"a"
    assert base64_decode("YWI=") == b"ab"
    assert base64_decode("YWJj") == b"abc"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test bytes input
    assert base64_decode(b"VGVzdA") == b"Test"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from .exc import BadData

def test_base64_decode():
    # Test basic functionality (standard case)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Test padding recovery
    
    # Test URL-safe specific characters
    # '-' is 62, '_' is 63 in urlsafe
    assert base64_decode("YV9i") == b"a\xbe" 
    assert base64_decode("YV8b") == b"a\xbf"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test different encodings/types
    assert base64_decode(u"YWJj") == b"abc"

    # Test error handling with invalid characters (non-alphabet)
    # Note: urlsafe_b64decode ignores non-alphabet chars if they aren't part of the logic, 
    # but certain malformed structures should trigger BadData.
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test error handling with invalid padding/structure that cannot be recovered
    # (Though the function tries to add padding, extreme corruption triggers it)
    with pytest.raises(BadData):
        # Using a character that causes an error in decoding logic if possible
        base64_decode("invalid-base64-content-\x00")
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from .exc import BadData

def test_base64_decode():
    # Test standard decoding
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test URL-safe characters (using - and _ instead of + and /)
    # Standard B64 for "subject?a=1" is "c3ViamVjdD9hPTE="
    # URL-safe B64 replaces + with - and / with _
    assert base64_decode("c3ViamVjdD9hPTE") == b"subject?a=1"
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test decoding with different types (bytes input)
    assert base64_decode(b"Ym9i") == b"bob"

    # Test invalid data raises BadData
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test decoding with non-ascii characters (should be ignored due to encoding='ascii', errors='ignore')
    # 'SGVsbG8' is 'Hello'. If we append non-ascii, it should just decode the valid part.
    assert base64_decode("SGVsbG8©") == b"Hello"
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replace + with - and / with _)
    # "Subject?" in base64 is "U3ViamVjdD8="
    # URL safe variant of a string containing + or /
    assert base64_decode("Y29uZmlybV90ZXN0") == b"confirm_test"
    
    # Test bytes input
    assert base64_decode(b"R29vZA") == b"Good"
    
    # Test padding recovery (input without trailing =)
    # "abcde" -> "YWJjZGU=" 
    assert base64_decode("YWJjZGU") == b"abcde"
    
    # Test empty string
    assert base64_decode("") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test handling of non-ascii characters in input via encoding/errors logic
    # The function uses ascii/ignore for decoding, so it should strip them
    assert base64_decode("YWJjZGU\xff") == b"abcde"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard URL-safe base64 encoding/decoding
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    assert base64_decode(b"SGVsbG8gd29ybGQ") == b"Hello world"

    # Test with padding requirement (the function handles missing '=')
    # "any string" in urlsafe b64 is "YW55IHN0cmluZw" (length 14)
    # 14 % 4 = 2. Needs 2 '=' to reach 16.
    assert base64_decode("YW55IHN0cmluZw") == b"any string"

    # Test with existing padding
    assert base64_decode("YW55IHN0cmluZw==") == b"any string"

    # Test special characters in URL-safe alphabet (- and _)
    # '-' is 45, '_' is 46
    # Using bytes that result in these chars: \xff\xef -> '_-' (approx)
    # Let's use a known value: b'\xfa\xff' -> base64 urlsafe is '-_8'
    assert base64_decode("-_8") == b"\xfa\xff"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test non-ascii characters in input (should be ignored due to errors="ignore" in encoding="ascii")
    # 'abc' + emoji -> 'abc'
    assert base64_decode("YWJj\U0001F600") == b"abc"
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding needed
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YmI") == b"bb"
    
    # Test input as bytes
    assert base64_decode(b"YmFzZTY0") == b"base64"
    
    # Test special characters in URL-safe alphabet (- and _)
    # "a_b-" in base64 is "YV9i" (with padding)
    assert base64_decode("YV9i") == b"a_b"
    
    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!not_base64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (using - and _ instead of + and /)
    # 'subject?' in urlsafe is 'c3ViamVjdD8='
    # Let's use a known pattern: '+' becomes '-' and '/' becomes '_'
    assert base64_decode("YV9iLWM") == b"a_b-c" # manually padded/handled by function logic
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test missing padding (the function should handle it)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("YQ") == b"a"
    
    # Test with different encoding/ascii handling
    assert base64_decode("SGVsbG8\n") == b"Hello"

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
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test with URL-safe characters (replace + and / with - and _)
    # "subject?" in standard B64 is "c3ViamVjdD8="
    # In urlsafe it uses '-' or '_' depending on the byte values
    assert base64_decode("YV9iLWM") == b"a_b-c" 
    
    # Test padding stripping (the function handles missing padding)
    assert base64_decode("YW55IGNhcm5hbCBwbGVhc3VyZS4") == b"any carnal pleasure."
    
    # Test with different types (bytes input)
    assert base64_decode(b"dGVzdA") == b"test"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding required
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test bytes input
    assert base64_decode(b"YmFzZTY0") == b"base64"
    
    # Test URL-safe characters (substituting + and / with - and _)
    # "subject?" in standard B64 is "c3ViamVjdD8=" 
    # In urlsafe it's "c3ViamVjdD8"
    assert base64_decode("c3ViamVjdD8") == b"subject?"
    
    # Test with different encoding/ascii-compatible string
    assert base64_decode("YW55IGNhcm5hbCBwbGVhc3VyZS4=") == b"any carnal pleasure."

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!NotBase64!!!")
    
    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test bytes input
    assert base64_decode(b"YmFzZTY0") == b"base64"
    
    # Test URL-safe characters (dash and underscore)
    # "a-b_" in urlsafe b64 is 'YS1iXw'
    assert base64_decode("YS1iXw") == b"a-b_"
    
    # Test with explicit padding provided (even though function handles it)
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid base64 data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test non-ascii characters in input (should be ignored via encoding='ascii', errors='ignore')
    # 'SGVsbG8' + emoji
    assert base64_decode("SGVsbG8🚀") == b"Hello"
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original
    assert base64_decode(encoded.decode("ascii")) == original

    # Test with padding (the function handles missing padding internally)
    unpadded = b"teststring"
    encoded_no_pad = base64_encode(unpadded)
    assert b"=" not in encoded_no_pad
    assert base64_decode(encoded_no_pad) == unpadded

    # Test with URL-safe characters (hyphen and underscore)
    # 'abc?' in urlsafe is 'YWJjPw' -> mapping to '-' or '_'
    # Let's use a known pattern: b'\xff\xfe' encodes to '____' in some contexts, 
    # but specifically checking '-' and '_'
    special_bytes = b"\xbe\xef" # produces certain characters
    encoded_special = base64_encode(special_bytes)
    assert base64_decode(encoded_special) == special_bytes

    # Test with empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test error handling for invalid characters (not in urlsafe alphabet)
    # Note: base64.urlsafe_b64decode might ignore some non-alphabet chars, 
    # but characters that break the structure should raise BadData.
    with pytest.raises(BadData):
        # Using a character that is not part of the URL-safe alphabet and disrupts decoding
        # Specifically, characters like '!' are not in the alphabet
        base64_decode("abc!!!")

    # Test error handling for malformed input
    with pytest.raises(BadData):
        # Providing something that cannot be decoded even with padding
        base64_decode(b"\x00\x01\x02") # This might actually decode, but 
        # let's use an invalid sequence if possible.
        # Actually, base64_decode is quite robust due to error='ignore'.
        # We trigger BadData via the TypeError/ValueError in base64.urlsafe_b64decode
        pass

    # Test data type flexibility (str vs bytes)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (hyphen and underscore)
    # 'a/b+' in standard becomes 'a_b-' in urlsafe
    assert base64_decode("YV9iLWM=") == b"a_b-c"
    
    # Test padding handling (stripping '=' should be handled by the function's logic)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test different encodings/types
    assert base64_decode(b"\x00\x01\x02") == b"\x00\x01\x02"
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == original_text.encode("utf-8")

    # Test decoding with padding missing (the function handles this internally)
    unpadded_encoded = b"SGVsbG8"  # "Hello" without padding
    assert base64_decode(unpadded_encoded) == b"Hello"

    # Test decoding bytes input
    encoded_bytes = b"Unittest"
    assert base64_decode(encoded_bytes) == b"Unittest"

    # Test URL-safe characters (hyphen and underscore)
    # 'a' + '/' in standard B64 is 'YQ/' -> in urlsafe it is 'YQ-'
    urlsafe_input = b"YQ-" 
    assert base64_decode(urlsafe_input) == b"a/"

    # Test decoding with invalid characters (should raise BadData via try-except)
    with pytest.raises(BadData):
        base64_decode(b"!!!not_base64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (urlsafe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test URL-safe specific characters (+ replaced by - and / replaced by _)
    # "subject?" in urlsafe is "c3ViamVjdD8"
    assert base64_decode("c3ViamVjdD8") == b"subject?"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with different input types (bytes)
    assert base64_decode(b"YmFzZTY0") == b"base64"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!Invalid!!!")

    # Test with extra padding (should still work due to logic)
    assert base64_decode("SGVsbG8=====") == b"Hello"
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe character substitution ( - and _ )
    # "abc?" in urlsafe is 'YWJjPw' -> padding needed: 'YWJjPw==' 
    # but function handles padding automatically.
    assert base64_decode("YWJj") == b"abc"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test without padding (function should handle it)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"

    # Test with URL-safe characters specifically
    # Using '-' instead of '+' and '_' instead of '/'
    # Original bytes: \xff\xfe\xfd\xfc (not valid ascii, let's use a known urlsafe string)
    # 'u_8' -> represents specific bytes in base64
    assert base64_decode("u_8") == b"\xb8\xbe"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input (URL-safe)
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test input with padding (should handle extra/missing padding)
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test URL-safe characters specifically (using '-' and '_')
    # "base64_encode" of certain bytes results in '-' and '_' instead of '+' and '/'
    # Example: data that would produce '+' and '/'
    original_bytes = b"\xff\xef" 
    encoded = base64_encode(original_bytes)
    assert base64_decode(encoded) == original_bytes

    # Test invalid base64 data triggers BadData exception
    with pytest.raises(BadData):
        base64_decode("!!!NotBase64!!!")

    # Test edge case: very long string
    long_str = "A" * 100
    assert base64_decode(long_str) == base64.urlsafe_b64decode(long_str + "=" * (-len(long_str) % 4))
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from .exc import BadData

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test urlsafe variation (using - instead of + and _ instead of /)
    # "a/b+" in standard B64 is "YS9i+"
    # In URL-safe it is "YS9i" with padding or "YS1i"
    assert base64_decode("YS1i") == b"a/b" # Testing '-' mapping to '_' logic implicitly via urlsafe
    
    # Test input as bytes
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test missing padding (the function handles adding '=' automatically)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"

    # Test with different string types (str and bytes)
    assert base64_decode("YW55IGNhcm5hbCBwbGVhc3VyZS4=") == b"any carnal pleasure."
    assert base64_decode(b"YW55IGNhcm5hbCBwbGVhc3VyZS4=") == b"any carnal pleasure."

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    with pytest.raises(BadData):
        base64_decode(b"invalid_chars_@#$%^&*")
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (replace + with - and / with _)
    # Standard: "subject?query=1" -> "c3ViamVjdD9xdWVyeT0x" (no special chars)
    # Let's use a case that produces - and _
    # b'\xff\xef' in urlsafe is '____' or similar
    assert base64_decode("abc/def+") == b"abc\xfe\xef" # This would fail if not urlsafe
    # Actually, let's test specific URL-safe replacements:
    # standard b64 for \xFB\xFF is b'+/8=' -> urlsafe is b'-_8='
    assert base64_decode("-_8") == b"\xfb\xff"

    # Test padding handling (the function adds missing =)
    assert base64_decode("YQ") == b"a"      # Needs ==
    assert base64_decode("YmI") == b"bb"    # Needs =
    assert base64_decode("Y2Nj") == b"ccc"  # No padding needed

    # Test with different input types (str and bytes)
    assert base64_decode(b"YWJj") == b"abc"
    assert base64_decode("YWJj") == b"abc"

    # Test error handling
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")  # Invalid characters for urlsafe b64 decoding logic

    # Test empty input
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (using - and _ instead of + and /)
    # Base64 for \xff\xfe is u_4= (standard would be u/4=)
    assert base64_decode("u_4") == b"\xff\xfe"
    
    # Test different encodings/types
    assert base64_decode(b"YmFzZTY0") == b"base64"
    
    # Test padding handling (function adds missing =)
    assert base64_decode("YW55") == b"any"  # 'any' is 3 bytes, needs 1 '='
    assert base64_decode("YW55YQ") == b"anya" # 'anya' is 4 bytes, no padding needed
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe specific characters (replace + and / with - and _)
    assert base64_decode("YV9iLWNfZA") == b"a_b-c_d"
    
    # Test bytes input
    assert base64_decode(b"R29vZA") == b"Good"
    
    # Test with missing padding (the function should handle it)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YmI") == b"bb"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test unicode/non-ascii characters in input (should be ignored via ascii errors="ignore")
    # "SGVsbG8=" with a trailing non-ascii char should still decode correctly
    assert base64_decode("SGVsbG8=🚀") == b"Hello"
```


# LLM-generated content at query #21
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

    # Test handling of URL-safe characters (no padding needed)
    # 'abc' -> 'YWJj'
    assert base64_decode("YWJj") == b"abc"
    
    # Test string input instead of bytes for decode
    assert base64_decode("SGVsbG8=") == b"Hello"

    # Test decoding with missing padding (the function handles this via modulo)
    assert base64_decode("YQ") == b"a"  # 'YQ' needs '==' to be 'YQ=='
    assert base64_decode("YWI") == b"ab" # 'YWI' needs '=' to be 'YWI='

    # Test error handling with invalid characters/data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        # Using a character that is not in the URL-safe alphabet (e.g., space or !)
        # Note: urlsafe_b64decode might ignore some chars, but certain sequences trigger errors
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""

    # Test decoding input that results in invalid bytes for the expected format
    # (Though urlsafe_b64decode is quite permissive with padding, 
    # we test the exception path via BadData)
    with pytest.raises(BadData):
        base64_decode(b"\x00\x01\x02") # This specific case might pass depending on padding logic,
                                        # but let's target a structural failure if possible.
                                        # An actual TypeError/ValueError in base64.urlsafe_b64decode:
                                        # Passing non-bytes/non-ascii to the inner logic.
        pass 
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replace + with - and / with _)
    assert base64_decode("YW55LWNhcm5hbCBwb3N0") == b"any-carnal post"
    
    # Test input as bytes
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test padding recovery (input without trailing '=')
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("YQ") == b"a"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid base64 data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!Invalid!!!")

    # Test with specific encoding/ASCII compatibility
    assert base64_decode("Ym9i") == b"bob"
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (using '-' and '_' instead of '+' and '/')
    # '?' in urlsafe is encoded as '_' and '/' is encoded as '-'
    # Let's use a known pattern: \xff\xfe -> _w==
    assert base64_decode("_w") == b"\xff\xfe"
    
    # Test padding handling (the function adds '=' automatically)
    assert base64_decode("YQ") == b"a"  # 'YQ=='
    assert base64_decode("YmI") == b"bb" # 'YmI='
    
    # Test different encodings/types
    assert base64_decode(b"VGVzdA") == b"Test"
    
    # Test error handling for invalid base64 data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test empty input
    assert base64_decode("") == b""
```


# LLM-generated content at query #24
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

    # Test URL-safe characters (using - and _ instead of + and /)
    # Base64 for b'\xfb\xff' is '+/8=' in standard, but '-_8=' in urlsafe
    url_safe_input = b"-_8" 
    # Note: base64_decode handles padding internally via the modulo logic
    assert base64_decode(url_safe_input) == b"\xfb\xff"

    # Test with missing padding (the function should handle it)
    encoded_no_padding = base64_encode("testing")[:-1]
    assert base64_decode(encoded_no_padding) == b"testing"

    # Test error handling with invalid base64 data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        # Using characters outside the URL-safe alphabet/valid structure
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from .exc import BadData

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (no padding needed)
    # 'abc' in urlsafe b64 is 'YWJj'
    assert base64_decode("YWJj") == b"abc"
    
    # Test input as bytes
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test missing padding (the function should handle it via the modulo logic)
    assert base64_decode("YWJj") == b"abc"
    assert base64_decode("YQ") == b"a"
    
    # Test with hyphen and underscore (URL-safe specific)
    # b'\xff\xef' in urlsafe is '__8='
    assert base64_decode("__8") == b"\xff\xef"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding (the function should handle missing padding)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("Ym") == b"b"
    assert base64_decode("Y2M") == b"cc"
    
    # Test URL-safe characters (hyphen and underscore)
    # In urlsafe, '-' is '+' and '_' is '/'
    # 'i\xff' in standard base64 is 'i//==' -> 'i__=' in urlsafe
    assert base64_decode("i__") == b"i\xff"
    assert base64_decode("aa--") == b"a\xee" # Example of hyphen usage
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test case sensitivity/encoding issues (ASCII ignore)
    # Using non-ascii characters that should be ignored by the encoding=ascii, errors=ignore logic
    assert base64_decode("SGVsbG8\xff") == b"Hello"
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe, no padding)
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding present
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"
    
    # Test with bytes input
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test special characters in URL-safe alphabet (- and _)
    # "aa" -> "YWE=" -> "YWE"
    # "-" and "_" are used for indices 62 and 63
    assert base64_decode("-_") == b"\xf8"
    
    # Test empty string
    assert base64_decode("") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding roundtrip
    original = b"hello world"
    encoded = base64_encode(original)
    assert base64_decode(encoded) == original

    # Test with string input
    original_str = "pytest unit testing"
    encoded_str = base64_encode(original_str)
    assert base64_decode(encoded_str) == original_str.encode("utf-8")

    # Test URL-safe specific characters (e.g., '+' and '/' replaced by '-' and '_')
    # The byte sequence that would normally result in non-URL safe chars
    complex_bytes = b"\xff\xfe\xff" 
    encoded_url_safe = base64_encode(complex_bytes)
    assert b"+" not in encoded_url_safe
    assert b"/" not in encoded_url_safe
    assert base64_decode(encoded_url_safe) == complex_bytes

    # Test padding recovery (the function adds '=' back manually)
    unpadded = b"YWJj" # "abc" in base64, length 4, no padding needed
    unpadded_short = b"YWI" # "ab" in base64, needs padding
    assert base64_decode(unpadded) == b"abc"
    assert base64_decode(unpadded_short) == b"ab"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        # Use characters outside the valid base64 alphabet/urlsafe range
        base64_decode(b"!@#$%^&*")

    # Test empty input
    assert base64_decode(b"") == b""
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replace + with - and / with _)
    # 'subject?' in standard B64 is 'c3ViamVjdD8=' 
    # In urlsafe, if it were problematic:
    assert base64_decode("YV9i") == b"a_b"
    
    # Test missing padding (the function should handle this)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("c3ViamVjdD") == b"subject"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test bytes input specifically
    assert base64_decode(b"YmFzZTY0") == b"base64"
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard encoding/decoding cycle
    original_text = "hello world"
    encoded = base64_encode(original_text)
    decoded = base64_decode(encoded)
    assert decoded == original_text.encode("utf-8")

    # Test decoding with padding missing (the function handles this)
    # "python" in urlsafe b64 is 'cHl0aG9u'
    # Adding a character to make it non-multiple of 4: 'cHl0aG9u_' -> needs padding
    incomplete_encoded = b"cHl0aG9u" 
    assert base64_decode(incomplete_encoded) == b"python"

    # Test decoding bytes instead of str
    encoded_bytes = b"YW55IGNhcm5hbCBwbGVhc3VyZS4="
    assert base64_decode(encoded_bytes) == b"any carnal pleasure."

    # Test URL-safe characters (hyphen and underscore)
    # '-' and '_' are used in urlsafe instead of '+' and '/'
    url_safe_input = b"_-ab" 
    # Decodes to bytes that would be invalid in standard b64 but valid in urlsafe
    try:
        base64_decode(url_safe_input)
    except BadData:
        pytest.fail("base64_decode failed to handle url-safe characters")

    # Test invalid data raises BadData
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        # Using characters outside the base64 alphabet/urlsafe range 
        # specifically triggering a ValueError in b64decode
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (using - and _ instead of + and /)
    # "base64".encode() is b'base64', urlsafe would be 'YmFzZTY0'
    # Let's use a known case: bytes that produce '+' or '/' in standard B64
    # Standard B64 for b'\xff\xfe' is '/v4='
    # URL-safe should be '_v4='
    assert base64_decode("_v4") == b"\xff\xfe"
    
    # Test bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test with missing padding (the function should handle it)
    assert base64_decode("YmFzZTY0") == b"base64"
    
    # Test empty string
    assert base64_decode("") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!NotBase64!!!")

    # Test with non-ascii characters in input (should be ignored per encoding="ascii", errors="ignore")
    assert base64_decode("SGVsbG8=🚀") == b"Hello"
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe specific characters (replace + and / with - and _)
    # Standard: "subject? data" -> "c3ViamVjdD8gZGF0YQ=="
    # URL-safe: "c3ViamVjdD8gZGF0YQ" (no padding needed in our function)
    assert base64_decode("c3ViamVjdD8gZGF0YQ") == b"subject? data"
    
    # Test with different input types (str and bytes)
    assert base64_decode(b"Ym9v") == b"boo"
    assert base64_decode("Ym9v") == b"boo"

    # Test padding restoration logic
    # "abc" -> "YWJj" (length 4, no padding)
    # "ab" -> "YWI=" (length 3, needs 1 padding)
    assert base64_decode("YWI") == b"ab"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (dash and underscore)
    # "_" is part of the alphabet, "-" is part of the alphabet
    # "YV9i" -> "a_b"
    assert base64_decode("YV9i") == b"a_b"
    
    # Test padding handling (striping equals during encoding)
    assert base64_decode("YW55IGNhcm5hbCBwbGVhc3VyZS4") == b"any carnal pleasure."
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test with different types (bytes input)
    assert base64_decode(b"dGVzdA") == b"test"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!invalid!!!")
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (no padding needed)
    # 'abc' in urlsafe base64 is 'YWJj'
    assert base64_decode("YWJj") == b"abc"
    
    # Test with hyphen and underscore (URL-safe alphabet)
    # '-' and '_' are used instead of '+' and '/'
    # Example string that uses these characters
    encoded = "-_8" # This is a valid sequence in urlsafe
    # We can also test decoding known urlsafe strings
    assert base64_decode("-_") == b"\xf1\xbd" 

    # Test padding recovery (the function handles missing '=')
    assert base64_decode("YWJj") == b"abc"  # 'YWJj' is 4 chars, no pad needed
    assert base64_decode("YWI") == b"ab"    # 'YWI' is 3 chars, needs 1 '='

    # Test different encodings via string input
    assert base64_decode("8J+Zjw==") == b"\xf0\x9f\x98\x98" # 😎 emoji

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode(b"not_base64_at_all")
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard valid input
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe specific characters (replace + and / with - and _)
    # "data/test?" -> "ZGF0YS90ZXN0Pw==" in standard, 
    # but urlsafe uses "ZGF0YS90ZXN0Pw" (no padding needed)
    assert base64_decode("ZGF0YS10ZXN0Xw") == b"data-test_"
    
    # Test automatic padding handling
    assert base64_decode("YQ") == b"a"      # Needs ==
    assert base64_decode("YmI") == b"bb"   # Needs =
    assert base64_decode("Y2Nj") == b"ccc" # Needs no padding
    
    # Test different encodings/types
    assert base64_decode(b"\x41\x42\x43") == b"ABC"
    
    # Test invalid data triggers BadData exception
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test empty input
    assert base64_decode("") == b""
```


# LLM-generated content at query #36
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test URL-safe characters (- and _)
    # "data/test" in urlsafe b64 is "ZGF0YS90ZXN0"
    # Let's use a known case: "subject?" -> "c3ViamVjdD8" 
    # In urlsafe, '+' becomes '-' and '/' becomes '_'
    assert base64_decode("-_") == base64.urlsafe_b64decode(b"-_==")
    
    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test different encodings/types
    assert base64_decode(b"Ym9v") == b"boo"

    # Test error handling for invalid data
    with pytest.raises(BadData):
        base64_decode("!!!")  # Invalid characters for base64 logic in some contexts or structure
        
    # Test with specific edge case: single character (invalid padding/length)
    # The function adds padding, but if the content itself is not valid b64
    with pytest.raises(BadData):
        base64_decode("A") # This might decode to something unexpected or error depending on implementation, 
                           # but we test for the custom exception raised by the try-except block.
```


# LLM-generated content at query #37
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding requirement (input length not multiple of 4)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    
    # Test input as bytes
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (hyphen and underscore)
    # "a-b_" in urlsafe is different from standard b64
    assert base64_decode("YV8tYg") == b"a_b"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #38
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (replace +/ with -_)
    # 'a+b/' in urlsafe is 'a-b_'
    assert base64_decode("a-b_") == b"a+b/"
    
    # Test input without padding (function should handle it)
    assert base64_decode("SGVsbG8") == b"Hello"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid base64 data raises BadData
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test ASCII encoding requirement (handling non-ascii via ignore)
    # 'SGVsbG8=' + unicode emoji that should be ignored by ascii/ignore logic
    assert base64_decode("SGVsbG8=🚀") == b"Hello"
```


# LLM-generated content at query #39
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test URL-safe specific characters (- and _)
    # "A_B-C" in urlsafe is different from standard base64
    input_str = "YV9iX2M-" 
    expected = b"a_b_c" # Depending on actual encoding, let's use a known pair
    assert base64_decode("YV9iX2M") == base64_decode("YV9iX2M=")

    # Test complex string
    complex_str = "Python_is_fun_123"
    encoded = base64_encode(complex_str)
    assert base64_decode(encoded) == complex_str.encode()

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        # Using characters outside the URL-safe alphabet if they cause decode error
        # Note: urlsafe_b64decode is lenient with some chars, but certain 
        # malformed structures or non-ascii bytes (if processed) trigger errors.
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #40
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test urlsafe variant (using '-' instead of '+')
    # '+' is replaced by '-' and '/' by '_' in urlsafe
    # 'a/b+' encoded via urlsafe: 'YS9iKw==' -> 'YS9iKw' (after rstrip)
    assert base64_decode("YS9iKw") == b"a/b+"

    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test padding recovery (the function handles missing '=')
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG") == b"Hel"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test non-ascii characters in input (should be ignored by encoding="ascii", errors="ignore")
    # 'SGVsbG8=' + '©' -> the © is dropped, resulting in b"Hello"
    assert base64_decode("SGVsbG8=©") == b"Hello"
```


# LLM-generated content at query #41
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
    original_bytes = b"\x00\xff\xfe\x12"
    encoded_bytes = base64_encode(original_bytes)
    assert base64_decode(encoded_bytes) == original_bytes

    # Test URL-safe characters (no padding needed, uses - and _)
    # Base64 for '>>?' is 'Pj4/' in standard, but 'Pj4-' or similar in urlsafe
    # Let's use a known urlsafe string: 'abc' -> 'YWJj'
    assert base64_decode("YWJj") == b"abc"
    
    # Test with missing padding (the function should handle it)
    # 'YQ' is 'a', needs '==' to be length 4
    assert base64_decode("YQ") == b"a"

    # Test with extra padding (the function handles it via modulo logic)
    assert base64_decode("YQ==") == b"a"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #42
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test basic functionality with standard string
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe character substitution ( - and _ )
    # "subject?" in urlsafe is "c3ViamVjdD8"
    assert base64_decode("c3ViamVjdD8") == b"subject?"
    
    # Test with bytes input
    assert base64_decode(b"Ym9uam91cg") == b"bonjour"
    
    # Test handling of missing padding (the function adds it automatically)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"

    # Test with different encodings/types via want_bytes logic
    assert base64_decode("VGVzdA==") == b"Test"

    # Test error handling for invalid data
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test edge case: empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #43
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe encoding (using '-' instead of '+')
    assert base64_decode("YV9i") == b"a_b"
    
    # Test bytes input
    assert base64_decode(b"dGVzdA") == b"test"
    
    # Test handling of missing padding (the function adds it automatically)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("YWI") == b"ab"
    assert base64_decode("YWJj") == b"abc"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test input with non-ascii characters (should be ignored due to errors="ignore")
    # 'SGVsbG8=' is 'Hello'. If we append invalid utf-8 bytes, it should still decode correctly.
    assert base64_decode(b"SGVsbG8=\xff") == b"Hello"
```


# LLM-generated content at query #44
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test urlsafe variations (replacing + and / with - and _)
    assert base64_decode("SGVsbG8") == b"Hello"  # Testing padding logic
    assert base64_decode("YV9iLWNfZA") == base64_encode("a_b-c_d")
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test padding recovery (the function adds missing =)
    # "abc" in base64 is "YWJj" (length 4, no padding needed)
    # "ab" in base64 is "YWI=" (length 3, needs 1 padding)
    assert base64_decode("YWI") == b"ab"
    
    # Test error handling with invalid characters/data
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test edge case: empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #45
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test URL-safe characters (using - and _ instead of + and /)
    # "Subject?" in standard B64 is "U3ViamVjdD8=" 
    # but let's use a known pattern: ">>\xff" -> "Pj4_"; ">>\xfe" -> "Pj4-"
    assert base64_decode("Pj4_") == b">>\xff"
    assert base64_decode("Pj4-") == b">>\xfe"

    # Test empty string
    assert base64_decode("") == b""
    assert base64_decode(b"") == b""

    # Test padding recovery (the function adds = manually)
    assert base64_decode("YQ") == b"a"  # YQ==
    assert base64_decode("YWI") == b"ab" # YWI=

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test non-ascii input (encoding="ascii", errors="ignore" should strip them)
    # "abc" + emoji -> "abc"
    assert base64_decode("YWJj\U0001F600") == b"abc"
```


# LLM-generated content at query #46
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (URL-safe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode("SGVsbG8") == b"Hello"  # Testing automatic padding
    
    # Test bytes input
    assert base64_decode(b"V29ybGQ=") == b"World"
    
    # Test URL-safe characters (using - and _ instead of + and /)
    # "a/b+" in standard B64 is "YS9i+"
    # In URL-safe it is "YS9i" with different mapping for chars 62 and 63
    # Let's use a known string: b'\xff\xfe' -> base64 urlsafe: '____'
    assert base64_decode("____") == b"\xff\xfe"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test invalid data (Non-base64 characters that cause failure)
    with pytest.raises(BadData):
        # Using a character outside the URL-safe alphabet logic if possible, 
        # though urlsafe_b64decode is quite lenient with whitespace/extra chars.
        # However, passing something that fundamentally breaks decoding:
        base64_decode("!!!") 

    # Test different encodings (ASCII)
    assert base64_decode("YQ==", encoding="ascii") == b"a"
```


# LLM-generated content at query #47
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard URL-safe base64 encoding/decoding
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    assert base64_decode(b"SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding-less input (function handles manual padding)
    assert base64_decode("YQ") == b"a"
    assert base64_decode("Ym") == b"b"
    
    # Test with URL-safe characters (- and _)
    # "base64-" encoded is "YmFzZTY0LQ"
    # "base64_" encoded is "YmFzZTY0Xw"
    assert base64_decode("YmFzZTY0LQ") == b"base64-"
    assert base64_decode("YmFzZTY0Xw") == b"base64_"

    # Test with different encodings (input as string)
    assert base64_decode("8J+Zjw==".replace("=", "")) == b"\xf0\x9f\x98\x8e"

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test with empty input
    assert base64_decode("") == b""
```


# LLM-generated content at query #48
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard string input
    assert base64_decode("SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test bytes input
    assert base64_decode(b"SGVsbG8gd29ybGQ") == b"Hello world"
    
    # Test with padding required (input length 7, needs 1 '=')
    # 'abcde' -> 'YWJjZGU=' (length 8)
    assert base64_decode("YWJjZGU") == b"abcde"
    
    # Test URL-safe characters (- and _)
    # '-' is 62, '_' is 63 in urlsafe
    assert base64_decode("-_==") == b"\xff\xef" # Using specific byte sequences
    # More reliable check for urlsafe:
    assert base64_decode("YV9i") == b"a_b"
    assert base64_decode("YV-i") == b"a-b"

    # Test with existing padding (should not break)
    assert base64_decode("SGVsbG8gd29ybGQ=") == b"Hello world"

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")
```


# LLM-generated content at query #49
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case (urlsafe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"
    
    # Test with URL-safe characters (+ replaced by - and / replaced by _)
    # "subject?" in standard B64 is "c3ViamVjdD8=" -> urlsafe "c3ViamVjdD8"
    assert base64_decode("c3ViamVjdD8") == b"subject?"
    
    # Test padding handling (the function adds missing padding)
    assert base64_decode("YQ") == b"a"      # Needs ==
    assert base64_decode("YmI") == b"bb"   # Needs =
    assert base64_decode("Y2Jj") == b"ccc" # No padding needed
    
    # Test with different types (bytes input)
    assert base64_decode(b"VGVzdA") == b"Test"

    # Test error handling for invalid data
    with pytest.raises(BadData, match="Invalid base64-encoded data"):
        base64_decode("!!!")

    # Test with empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #50
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

    # Test decoding bytes directly
    encoded_bytes = b"pythoo_n" # urlsafe version of something
    # "python" in urlsafe b64 without padding is 'cHl0aG9u'
    # Let's use a known value: 'test' -> 'dGVzdA'
    assert base64_decode("dGVzdA") == b"test"

    # Test with different input types (str vs bytes)
    assert base64_decode(b"dGVzdA") == b"test"

    # Test padding handling (the function adds '=' internally)
    # 'dGVzdA' has length 6. 6 % 4 = 2. Needs 2 padding chars.
    assert base64_decode("dGVzdA") == b"test"

    # Test urlsafe specific characters (- and _)
    # '_' is index 63 in urlsafe, '-' is index 62
    # Let's encode a string that produces these
    test_str = b"\xff\xfe\xfd\xfc" # High bytes to force special chars
    encoded_urlsafe = base64_encode(test_str)
    assert b"-" in encoded_urlsafe or b"_" in encoded_urlsafe
    assert base64_decode(encoded_urlsafe) == test_str

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!not_base64!!!")

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #51
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test URL-safe characters (using '-' instead of '+')
    # 'base64' with '+' would be 'YmFzZTY0Kys='
    # 'base64' with '-' is 'YmFzZTY0KC0='
    assert base64_decode("YmFzZTY0KC0=") == b"base64(+)" # wait, urlsafe swaps + and / for - and _
    # Correcting logic: urlsafe_b64encode replaces + with - and / with _
    assert base64_decode("YmFzZTY0LS1f") == b"base64-+_"

    # Test input as bytes
    assert base64_decode(b"SGVsbG8=") == b"Hello"

    # Test padding reconstruction (input without '=')
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("YmFzZTY0") == b"base64"

    # Test empty string
    assert base64_decode("") == b""

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test unicode/non-ascii input (should be ignored via errors="ignore" in encoding)
    # 'abc' is valid, adding a non-ascii char that gets stripped/ignored
    assert base64_decode("SGVsbG8©") == b"Hello"
```


# LLM-generated content at query #52
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard case
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8=") == b"Hello"
    
    # Test URL-safe specific characters (replace + with - and / with _)
    # Original: "subject?query=1/2" -> Base64: "c3ViamVjdD9xdWVyeT0xLzI="
    # URL Safe: "c3ViamVjdD9xdWVyeT0xXzI=" (if / was replaced)
    # Using a known urlsafe string:
    assert base64_decode("YV9iLWNfZA") == b"a_b-c_d" # testing alphabet coverage
    
    # Test missing padding (the function should handle this via the modulo logic)
    assert base64_decode("SGVsbG8") == b"Hello"
    assert base64_decode("SGVsbG8=") == b"Hello"
    
    # Test different encodings/types
    assert base64_decode(b"YmFzZTY0") == b"base64"
    
    # Test invalid data raises BadData
    with pytest.raises(BadData):
        # Using characters outside the base64 alphabet that trigger ValueError/TypeError
        # Note: urlsafe_b64decode is quite lenient with non-alphabet chars, 
        # but certain structures or invalid byte sequences will trigger it.
        base64_decode("!!!") 

    # Test empty string
    assert base64_decode("") == b""
```


# LLM-generated content at query #53
#--------------------------

```python
import pytest

def test_base64_decode():
    # Test standard valid base64 (urlsafe)
    assert base64_decode("SGVsbG8=") == b"Hello"
    assert base64_decode(b"SGVsbG8") == b"Hello"  # Testing padding logic
    
    # Test urlsafe specific characters (- and _)
    # "a_b-c" encoded via urlsafe is "YV9iLWM="
    assert base64_decode("YV9iLWM") == b"a_b-c"
    
    # Test empty string
    assert base64_decode("") == b""
    
    # Test different input types (bytes)
    assert base64_decode(b"V29ybGQ=") == b"World"

    # Test invalid data raises BadData
    with pytest.raises(BadData):
        base64_decode("!!!")

    # Test decoding with extra padding characters handled correctly
    assert base64_decode("SGVsbG8======") == b"Hello"
```


