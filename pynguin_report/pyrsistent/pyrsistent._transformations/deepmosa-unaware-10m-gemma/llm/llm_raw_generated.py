####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that fit the pattern
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("0") is True
    
    # Test regex failing on non-matching strings
    assert matcher_digit("abc") is False
    assert matcher_digit("12a") is False
    
    # Test regex failing on non-string types (per implementation requirement)
    assert matcher_digit(123) is False
    assert matcher_digit(None) is False
    assert matcher_digit(["123"]) is False

    # Test complex pattern
    matcher_email = rex(r"^[a-z]+@domain\.com$")
    assert matcher_email("user@domain.com") is True
    assert matcher_email("USER@domain.com") is False  # Case sensitive
    assert matcher_email("user@domain.org") is False

    # Test empty string matching
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("post_test") is False

    # Test regex matching non-string types (should return False)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["a", "b"]) is False

    # Test complex regex
    matcher_email = rex(r"[^@]+@[^@]+\.[^@]+")
    assert matcher_email("user@example.com") is True
    assert matcher_email("invalid-email") is False

    # Test exact match
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("exact_extra") is False
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with exact string match
    matcher_exact = rex(r"^apple$")
    assert matcher_exact("apple") is True
    assert matcher_exact("apples") is False
    assert matcher_exact("pineapple") is False

    # Test regex matcher with pattern matching
    matcher_pattern = rex(r"^[0-9]+$")
    assert matcher_pattern("123") is True
    assert matcher_pattern("abc") is False
    assert matcher_pattern("") is False

    # Test regex matcher with prefix matching
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("suffix") is False

    # Test regex matcher with non-string types (should return False via isinstance check)
    matcher_str_only = rex(r".*")
    assert matcher_str_only(123) is False
    assert matcher_str_only(None) is False
    assert matcher_str_only(["a"]) is False

    # Test regex matcher with empty string and pattern
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_data") is True
    assert matcher_prefix("data_pre") is False

    # Test regex matching non-strings (should return False based on isinstance check)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["a"]) is False

    # Test exact match
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("test_1") is False
    assert matcher_complex("TEST_01") is False
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with exact string match
    matcher_exact = rex(r"^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("123") is False

    # Test regex matcher with pattern matching (digits)
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    # Test regex matcher with prefix matching
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("aprefix") is False

    # Test regex matcher with non-string types (should return False, not crash)
    matcher_any = rex(r".*")
    assert matcher_any("anything") is True
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["list"]) is False

    # Test regex matcher with complex pattern
    matcher_complex = rex(r"^[a-z]+\d{2}$")
    assert matcher_complex("abc12") is True
    assert matcher_complex("abc1") is False
    assert matcher_complex("ABC12") is False
    assert matcher_complex("12abc") is False
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher for strings that match pattern
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("0") is True
    
    # Test regex matcher for strings that do not match pattern
    assert matcher_digits("abc") is False
    assert matcher_digits("12a") is False
    
    # Test regex matcher with prefix/suffix logic
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("apple") is False
    
    # Test non-string inputs (should return False per implementation)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False
    
    # Test empty string matching
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test case sensitivity
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with exact match
    matcher_exact = rex(r"^apple$")
    assert matcher_exact("apple") is True
    assert matcher_exact("apples") is False
    assert matcher_exact("pineapple") is False

    # Test regex matcher with partial match (prefix)
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("re") is False

    # Test regex matcher with pattern
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False

    # Test rex handling of non-string types (should return False, not crash)
    matcher_str = rex(r".*")
    assert matcher_str("anything") is True
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["a"]) is False

    # Test regex matcher with character classes
    matcher_class = rex(r"[a-z]+")
    assert matcher_class("abc") is True
    assert matcher_class("ABC") is False
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string matches
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex matcher with non-string types (should return False)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["a", "b"]) is False

    # Test regex matcher with complex patterns
    matcher_email = rex(r"^[a-z]+@example\.com$")
    assert matcher_email("user@example.com") is True
    assert matcher_email("user@other.com") is False
    assert matcher_email("USER@example.com") is False

    # Test regex matcher with empty string pattern
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching string keys
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit("") is False
    assert matcher_digit(123) is False  # Should handle non-string types gracefully

    # Test regex matching specific pattern
    matcher_prefix = rex(r"^test_")
    assert matcher_prefix("test_user") is True
    assert matcher_prefix("prod_user") is False

    # Test regex exact match
    matcher_exact = rex(r"exact_match")
    assert matcher_exact("exact_match") is True
    assert matcher_exact("exact_match_extra") is False

    # Test case sensitivity
    matcher_case = rex(r"[A-Z]+")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False

    # Test None/Empty behavior
    matcher_none = rex(r".*")
    assert matcher_none("") is True
    assert matcher_none(None) is False
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string keys that match
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("45") is True
    
    # Test regex matcher with string keys that do not match
    assert matcher_digits("abc") is False
    assert matcher_digits("12a") is False
    
    # Test regex matcher with non-string keys (should return False per implementation)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False

    # Test specific pattern matching
    matcher_prefix = rex(r"^test_")
    assert matcher_prefix("test_case") is True
    assert matcher_prefix("testing") is True
    assert matcher_prefix("not_test") is False

    # Test exact match via regex
    matcher_exact = rex("^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("exact_extra") is False
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_rex():
    # Test regular expression matcher with strings that match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("0") is True
    
    # Test regular expression matcher with strings that do not match
    assert matcher_digit("abc") is False
    assert matcher_digit("12a") is False
    
    # Test regular expression matcher with non-string types (should return False)
    assert matcher_digit(123) is False
    assert matcher_digit(None) is False
    assert matcher_digit(["123"]) is False

    # Test complex regex
    matcher_complex = rex(r"^user_\d{2}$")
    assert matcher_complex("user_01") is True
    assert matcher_complex("user_99") is True
    assert matcher_complex("user_1") is False
    assert matcher_complex("admin_01") is False

    # Test empty string match
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test case sensitivity (default behavior)
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex matching non-string types (should return False via isinstance check)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["a"]) is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_[0-9]{2}$")
    assert matcher_complex("abc_12") is True
    assert matcher_complex("abc_1") is False
    assert matcher_complex("ABC_12") is False
    assert matcher_complex("abc_ab") is False

    # Test ny (matches anything) as a baseline comparison
    assert ny("anything") is True
    assert ny(None) is True
    assert ny(123) is True
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types via isinstance check

    # Test with a prefix match
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_data") is True
    assert matcher_prefix("data_pre") is False
    assert matcher_prefix("") is False

    # Test with a character class
    matcher_chars = rex(r"[a-z]+")
    assert matcher_chars("hello") is True
    assert matcher_chars("Hello") is False
    assert matcher_chars("123") is False

    # Test with complex pattern (email-like)
    matcher_email = rex(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    assert matcher_email("test.user@example.com") is True
    assert matcher_email("invalid-email") is False

    # Test with None/Non-string types explicitly
    matcher_any = rex(r".*")
    assert matcher_any("anything") is True
    assert matcher_any(None) is False
    assert matcher_any(True) is False
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that match the pattern
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("0") is True
    
    # Test regex not matching strings that don't match the pattern
    assert matcher_digits("abc") is False
    assert matcher_digits("12a") is False
    
    # Test regex with specific prefix
    matcher_prefix = rex(r"^test_")
    assert matcher_prefix("test_case") is True
    assert matcher_prefix("t_case") is False
    
    # Test case sensitivity
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False
    
    # Test non-string inputs (should return False per implementation)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False

    # Test empty string matching
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_rex():
    # Test regex pattern matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False
    assert matcher_prefix("pre") is False

    # Test regex pattern failing on non-string types (should return False, not raise error)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test exact match pattern
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False

    # Test case sensitivity (default behavior of re.compile)
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string keys that match
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("0") is True
    
    # Test regex matcher with string keys that do not match
    assert matcher_digits("abc") is False
    assert matcher_digits("12a") is False
    
    # Test regex matcher with non-string keys (should return False per implementation)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits([]) is False

    # Test complex regex pattern
    matcher_email = rex(r"[^@]+@[^@]+\.[^@]+")
    assert matcher_email("test@example.com") is True
    assert matcher_email("invalid-email") is False

    # Test exact match pattern
    matcher_exact = rex("^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not exact") is False
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher for exact match
    matcher_exact = rex(r"^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("123") is False

    # Test regex matcher with pattern
    matcher_pattern = rex(r"^\d+$")
    assert matcher_pattern("123") is True
    assert matcher_pattern("abc") is False
    assert matcher_pattern("") is False

    # Test regex matcher for prefix
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("post") is False

    # Test non-string inputs (should return False per implementation)
    matcher_str_only = rex(r".*")
    assert matcher_str_only(123) is False
    assert matcher_str_only(None) is False
    assert matcher_str_only(["abc"]) is False

    # Test empty string match
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that match the pattern
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("0") is True
    
    # Test regex non-matching strings
    assert matcher_digits("abc") is False
    assert matcher_digits("12a") is False
    
    # Test regex with specific prefix
    matcher_prefix = rex(r"^test_")
    assert matcher_prefix("test_case") is True
    assert matcher_prefix("testing") is True
    assert matcher_prefix("pre_test") is False
    
    # Test non-string inputs (should return False per implementation)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_[0-9]{2}$")
    assert matcher_complex("abc_12") is True
    assert matcher_complex("abc_1") is False
    assert matcher_complex("ABC_12") is False
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_data") is True
    assert matcher_prefix("data_pre") is False

    # Test non-string inputs (should return False, not crash)
    assert matcher_digit(123) is False
    assert matcher_digit(None) is False
    assert matcher_digit(["123"]) is False

    # Test exact match
    matcher_exact = rex("^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("exac") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("abc_1") is False
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_start_a = rex("^a")
    assert matcher_start_a("apple") is True
    assert matcher_start_a("banana") is False
    assert matcher_start_a(123) is False  # Should handle non-string types safely
    assert matcher_start_a(None) is False

    # Test with regex pattern for digits
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("123a") is False
    assert matcher_digits("") is False

    # Test with complex pattern
    matcher_email_part = rex(r".+@.+\..+")
    assert matcher_email_part("test@example.com") is True
    assert matcher_email_part("invalid-email") is False

    # Test behavior with empty string
    matcher_empty = rex("^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test that it returns a callable (lambda) as specified in the implementation
    result = rex(".*")
    assert callable(result)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that satisfy the pattern
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("0") is True
    
    # Test regex failing on non-matching strings
    assert matcher_digits("123a") is False
    assert matcher_digits("") is False

    # Test regex with specific pattern (starts with 'test_')
    matcher_prefix = rex(r"^test_.*")
    assert matcher_prefix("test_function") is True
    assert matcher_prefix("test_123") is True
    assert matcher_prefix("function_test") is False

    # Test regex with case sensitivity
    matcher_case = rex(r"ABC")
    assert matcher_case("ABC") is True
    assert matcher_case("abc") is False

    # Test behavior with non-string types (should return False per implementation)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("abc_12") is True
    assert matcher_complex("abc_1") is False
    assert matcher_complex("ABC_12") is False
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with exact string match
    matcher_exact = rex(r"^apple$")
    assert matcher_exact("apple") is True
    assert matcher_exact("apples") is False
    assert matcher_exact("pineapple") is False

    # Test regex matcher with partial match (starts with)
    matcher_start = rex(r"^pre")
    assert matcher_start("prefix") is True
    assert matcher_start("reprefix") is False

    # Test regex matcher with digit pattern
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False

    # Test regex matcher with non-string input (should return False, not crash)
    matcher_str = rex(r".*")
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["a"]) is False

    # Test regex matcher with empty string pattern
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test complex regex (case insensitive or character classes)
    matcher_class = rex(r"[a-z]+[0-9]")
    assert matcher_class("abc1") is True
    assert matcher_class("ABC1") is False
    assert matcher_class("123") is False
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that match the pattern
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("0") is True
    
    # Test regex matching strings that do not match the pattern
    assert matcher_digits("abc") is False
    assert matcher_digits("12a") is False
    
    # Test regex with specific characters
    matcher_prefix = rex(r"^pre_.*")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("post_test") is False
    
    # Test case sensitivity
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False
    
    # Test non-string inputs (should return False, not raise error)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False
    
    # Test complex regex
    matcher_complex = rex(r"^[a-z]{3}-\d{2}$")
    assert matcher_complex("abc-12") is True
    assert matcher_complex("abcd-12") is False
    assert matcher_complex("abc-1") is False
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_rex():
    # Test regular expression matcher with string keys that match
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("0") is True
    
    # Test regular expression matcher with string keys that do not match
    assert matcher_digits("abc") is False
    assert matcher_digits("12a") is False
    
    # Test regular expression matcher with non-string keys (should return False)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits([]) is False

    # Test complex regex pattern
    matcher_email = rex(r"[a-z]+@[a-z]+\.com$")
    assert matcher_email("test@example.com") is True
    assert matcher_email("user@domain.net") is False
    assert matcher_email(None) is False

    # Test empty string match if pattern allows
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with exact match
    matcher_exact = rex(r"^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("abc ") is False

    # Test regex matcher with partial match (as per re.match behavior)
    matcher_start = rex(r"^[0-9]+")
    assert matcher_start("123") is True
    assert matcher_start("123abc") is True
    assert matcher_start("a123") is False

    # Test regex matcher with character classes
    matcher_chars = rex(r"^[a-z]+$")
    assert matcher_chars("hello") is True
    assert matcher_chars("Hello") is False
    assert matcher_chars("h1") is False

    # Test regex matcher with non-string types (should return False, not crash)
    matcher_str = rex(r".*")
    assert matcher_str("anything") is True
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["a"]) is False

    # Test regex matcher with empty string
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("post_test") is False

    # Test regex non-string types (should return False via isinstance check)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["string"]) is False

    # Test exact match
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("exact_extra") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("item_01") is True
    assert matcher_complex("ITEM_01") is False
    assert matcher_complex("item_1") is False
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_rex():
    # Test matcher for exact string match
    matcher_exact = rex("^hello$")
    assert matcher_exact("hello") is True
    assert matcher_exact("hello world") is False
    assert matcher_exact(123) is False
    assert matcher_exact(None) is False

    # Test matcher for prefix
    matcher_prefix = rex("^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("append") is False

    # Test matcher for pattern matching (digits)
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("123a") is False
    assert matcher_digits("") is False

    # Test matcher with non-string types should always be False
    matcher_any = rex(".*")
    assert matcher_any("anything") is True
    assert matcher_any(["list"]) is False
    assert matcher_any({"key": "val"}) is False
    assert matcher_any(True) is False

    # Test complex regex (case insensitive via flags if passed, 
    # though rex uses re.compile internally with the provided string)
    matcher_complex = rex(r"^[a-z]+_[0-9]{2}$")
    assert matcher_complex("abc_12") is True
    assert matcher_complex("ABC_12") is False
    assert matcher_complex("abc_1") is False
    assert matcher_complex("abc_123") is False
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string keys that match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("0") is True
    
    # Test regex matcher with string keys that do not match
    assert matcher_digit("abc") is False
    assert matcher_digit("12a") is False
    
    # Test regex matcher with non-string keys (should return False via isinstance check)
    assert matcher_digit(123) is False
    assert matcher_digit(None) is False
    assert matcher_digit(["123"]) is False

    # Test complex pattern
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("abc_99") is True
    assert matcher_complex("TEST_01") is False  # Case sensitive
    assert matcher_complex("test_1") is False   # Wrong digit count
    assert matcher_complex("test_ab") is False  # Non-digits in value part

    # Test empty pattern (matches everything that is a string)
    matcher_empty = rex(r".*")
    assert matcher_empty("") is True
    assert matcher_empty("anything") is True
    assert matcher_empty(123) is False
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that match the pattern
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("0") is True
    
    # Test regex mismatching strings
    assert matcher_digit("abc") is False
    assert matcher_digit("12a") is False

    # Test prefix matching
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("apre") is False

    # Test case sensitivity
    matcher_case = rex(r"Hello")
    assert matcher_case("Hello") is True
    assert matcher_case("hello") is False

    # Test non-string inputs (should return False per implementation)
    assert rex(r".*")(123) is False
    assert rex(r".*")(None) is False
    assert rex(r".*")([]) is False

    # Test empty string matching
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("abc_1") is False
    assert matcher_complex("abc_123") is False
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False
    
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("post_test") is False

    # Test non-string types (should return False per implementation)
    matcher_any_str = rex(r".*")
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["a"]) is False

    # Test exact match
    matcher_exact = rex("^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("hello_99") is True
    assert matcher_complex("hello_9") is False
    assert matcher_complex("HELLO_99") is False
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest

def test_rex():
    # Test numeric/non-string input (should return False)
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits([]) is False

    # Test successful matches
    matcher_letters = rex(r"[a-z]+")
    assert matcher_letters("abc") is True
    assert matcher_letters("python") is True
    
    # Test failed string matches
    assert matcher_letters("ABC") is False  # Case sensitive
    assert matcher_letters("123") is False
    
    # Test complex regex (start/end anchors)
    matcher_exact = rex(r"^fixed$" )
    assert matcher_exact("fixed") is True
    assert matcher_exact("not fixed") is False
    assert matcher_exact("fixedly") is False

    # Test empty string match
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest

def test_rex():
    # Test exact match
    matcher_exact = rex(r"^apple$")
    assert matcher_exact("apple") is True
    assert matcher_exact("apples") is False
    assert matcher_exact("pineapple") is False

    # Test pattern match
    matcher_pattern = rex(r"^[0-9]+$")
    assert matcher_pattern("123") is True
    assert matcher_pattern("abc") is False
    assert matcher_pattern("") is False

    # Test partial match (if not anchored)
    matcher_partial = rex(r"test")
    assert matcher_partial("testing") is True
    assert matcher_partial("great test day") is True
    assert matcher_partial("t") is False

    # Test non-string inputs (should return False via isinstance check)
    matcher_str = rex(r".*")
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["apple"]) is False
    assert matcher_str(True) is False

    # Test empty string with wildcard
    matcher_wildcard = rex(r".*")
    assert matcher_wildcard("") is True
    assert matcher_wildcard("anything") is True
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex with case sensitivity
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False

    # Test non-string inputs (should return False, not raise error)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("user_01") is True
    assert matcher_complex("user_1") is False
    assert matcher_complex("User_01") is False
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest

def test_rex():
    # Test regex for exact match
    matcher_exact = rex("^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("123") is False

    # Test regex for prefix match
    matcher_prefix = rex("^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("aprefix") is False

    # Test regex for digit matching
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("12a") is False
    assert matcher_digits("") is False

    # Test behavior with non-string types (should return False)
    matcher_str_only = rex(".*")
    assert matcher_str_only("anything") is True
    assert matcher_str_only(None) is False
    assert matcher_str_only(123) is False
    assert matcher_str_only(["a", "b"]) is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("test_1") is False
    assert matcher_complex("test_abc") is False
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest

def test_rex():
    # Test basic regex matching for strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False
    
    # Test non-string inputs (should return False per implementation)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False

    # Test prefix matching
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("re") is False
    assert matcher_prefix(None) is False

    # Test exact match
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("exactness") is False

    # Test complex pattern (word boundaries and characters)
    matcher_complex = rex(r"\b[A-Z][a-z]\b")
    assert matcher_complex("Ab") is True
    assert matcher_complex("Abc") is False
    assert matcher_complex("aB") is False

    # Test empty regex (matches everything that is a string)
    matcher_all = rex(r".*")
    assert matcher_all("anything") is True
    assert matcher_all("") is True
    assert matcher_all(123) is False
```


# LLM-generated content at query #36
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching digits
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False
    assert matcher_digits(123) is False  # Should handle non-string types

    # Test regex matching specific prefix
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("post_test") is False
    assert matcher_prefix(None) is False

    # Test regex case sensitivity
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("Hello") is False

    # Test complex pattern
    matcher_complex = rex(r"^\w+@\w+\.com$")
    assert matcher_complex("user@domain.com") is True
    assert matcher_complex("user@domain.org") is False
    assert matcher_complex("!@#$") is False

    # Test empty string pattern
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #37
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string matches
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    # Test regex matcher with prefix/suffix
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("post_test") is False

    # Test regex matcher with non-string types (should return False, not raise error)
    matcher_any = rex(r".*")
    assert matcher_any("anything") is True
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["list"]) is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("hello_99") is True
    assert matcher_complex("hello_9") is False
    assert matcher_complex("HELLO_99") is False
```


# LLM-generated content at query #38
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex non-string inputs (should return False via isinstance check)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["string"]) is False

    # Test case sensitivity (default behavior of re.compile)
    matcher_case = rex(r"abc")
    assert matcher_case("abc") is True
    assert matcher_case("ABC") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("hello_99") is True
    assert matcher_complex("hello_9") is False
    assert matcher_complex("HELLO_99") is False
    assert matcher_complex("abc_def") is False
```


# LLM-generated content at query #39
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that follow the pattern
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    # Test regex matching specific characters
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_data") is True
    assert matcher_prefix("post_data") is False

    # Test case sensitivity
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False

    # Test non-string inputs (should return False as per implementation)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]{3}-\d{2}$")
    assert matcher_complex("abc-12") is True
    assert matcher_complex("abcd-12") is False
    assert matcher_complex("abc-1") is False
```


# LLM-generated content at query #40
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that satisfy the pattern
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("0") is True
    
    # Test regex not matching strings
    assert matcher_digit("abc") is False
    assert matcher_digit("12a") is False
    
    # Test regex with specific characters
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("post_test") is False
    
    # Test non-string inputs (should return False via isinstance check)
    assert matcher_digit(123) is False
    assert matcher_digit(None) is False
    assert matcher_digit(["123"]) is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("hello_99") is True
    assert matcher_complex("hello_9") is False
    assert matcher_complex("HELLO_99") is False
```


# LLM-generated content at query #41
#--------------------------

```python
import pytest

def test_rex():
    # Test with exact string match
    matcher_exact = rex(r"^hello$")
    assert matcher_exact("hello") is True
    assert matcher_exact("hello world") is False
    assert matcher_exact("hi") is False

    # Test with pattern matching (starts with)
    matcher_starts = rex(r"^pre_")
    assert matcher_starts("pre_test") is True
    assert matcher_starts("test_pre") is False
    assert matcher_starts("pre") is True

    # Test with digits
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("12a") is False
    assert matcher_digits("") is False

    # Test non-string types (should return False, not crash)
    matcher_str = rex(r".*")
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["string"]) is False
    assert matcher_str(True) is False

    # Test case sensitivity
    matcher_case = rex(r"ABC")
    assert matcher_case("ABC") is True
    assert matcher_case("abc") is False

    # Test complex regex (character classes)
    matcher_class = rex(r"^[a-z]+[0-9]$")
    assert matcher_class("abc1") is True
    assert matcher_class("123") is False
    assert matcher_class("abc") is False
```


# LLM-generated content at query #42
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_data") is True
    assert matcher_prefix("data_pre") is False

    # Test regex matching non-string types (should return False via isinstance check)
    matcher_any_str = rex(r".*")
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["a"]) is False

    # Test complex regex
    matcher_email_simple = rex(r"[a-z]+@[a-z]+\.com")
    assert matcher_email_simple("test@example.com") is True
    assert matcher_email_simple("TEST@example.com") is False  # Case sensitive
    assert matcher_email_simple("test@example.net") is False

    # Test empty string and exact match
    matcher_exact = rex("^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("ex") is False
```


# LLM-generated content at query #43
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string matches
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex matcher with non-string types (should return False)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["a"]) is False
    assert matcher_any(True) is False

    # Test exact match regex
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("item_01") is True
    assert matcher_complex("ITEM_01") is False
    assert matcher_complex("item_1") is False
    assert matcher_complex("item_abc") is False
```


# LLM-generated content at query #44
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string keys that match
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    # Test regex matcher with prefix
    matcher_prefix = rex(r"^test_")
    assert matcher_prefix("test_case") is True
    assert matcher_prefix("example_test") is False

    # Test regex matcher with non-string keys (should return False per implementation)
    matcher_any = rex(r".*")
    assert matcher_any("anything") is True
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any([]) is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("data_01") is True
    assert matcher_complex("data_1") is False
    assert matcher_complex("DATA_01") is False
    assert matcher_complex("abc_def") is False

    # Test empty string matching
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #45
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string keys that match
    matcher_digits = rex(r'^\d+$')
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    # Test regex matcher with string keys that do not match
    matcher_prefix = rex(r'^test_')
    assert matcher_prefix("test_item") is True
    assert matcher_prefix("item_test") is False

    # Test regex matcher with non-string types (should return False via isinstance check)
    matcher_any = rex('.*')
    assert matcher_any("anything") is True
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(['a', 'b']) is False

    # Test complex regex pattern
    matcher_complex = rex(r'^[a-z]+_\d{2}$')
    assert matcher_complex("data_01") is True
    assert matcher_complex("DATA_01") is False
    assert matcher_complex("data_1") is False
    assert matcher_complex("data_abc") is False

    # Test empty string pattern match/mismatch
    matcher_empty = rex('^$')
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #46
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit("") is False

    matcher_prefix = rex(r"pre_.*")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("post_test") is False

    # Test regex with non-string types (should return False, not crash)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["a"]) is False

    # Test exact match regex
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("exact_extra") is False

    # Test case sensitivity (default behavior)
    matcher_case = rex(r"abc")
    assert matcher_case("abc") is True
    assert matcher_case("ABC") is False
```


# LLM-generated content at query #47
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that match the pattern
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("0") is True
    
    # Test regex not matching strings (pattern mismatch)
    matcher_letters = rex(r"^[a-z]+$")
    assert matcher_letters("abc") is True
    assert matcher_letters("abc1") is False
    
    # Test non-string types (should return False per implementation)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False
    
    # Test complex regex pattern
    matcher_email = rex(r"[^@]+@[^@]+\.[^@]+")
    assert matcher_email("test@example.com") is True
    assert matcher_email("invalid-email") is False
    
    # Test empty string with specific pattern
    matcher_non_empty = rex(r".+")
    assert matcher_non_empty("a") is True
    assert matcher_non_empty("") is False
```


# LLM-generated content at query #48
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string keys that match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("0") is True
    
    # Test regex matcher with string keys that do not match
    assert matcher_digit("abc") is False
    assert matcher_digit("12a") is False
    
    # Test regex matcher with non-string keys (should return False per implementation)
    assert matcher_digit(123) is False
    assert matcher_digit(None) is False
    assert matcher_digit([]) is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("abc_99") is True
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("test_1") is False
    assert matcher_complex("test_aa") is False

    # Test empty regex (matches everything that is a string)
    matcher_empty = rex(r".*")
    assert matcher_empty("") is True
    assert matcher_empty("anything") is True
    assert matcher_empty(123) is False
```


# LLM-generated content at query #49
#--------------------------

```python
import pytest

def test_rex():
    # Test regular expression matcher with string matches
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False
    assert matcher_prefix("pre") is False

    # Test regular expression matcher with non-string types (should return False)
    matcher_any = rex(r".*")
    assert matcher_any("anything") is True
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["list"]) is False

    # Test exact match via regex
    matcher_exact = rex(r"^exact_string$")
    assert matcher_exact("exact_string") is True
    assert matcher_exact("exact_string_extra") is False

    # Test complex pattern (case insensitive logic simulation)
    matcher_complex = rex(r"^[A-Z][a-z]+$")
    assert matcher_complex("Hello") is True
    assert matcher_complex("hello") is False
    assert matcher_complex("HELL") is False
```


# LLM-generated content at query #50
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that fit the pattern
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("0") is True
    
    # Test regex non-matching strings
    assert matcher_digits("abc") is False
    assert matcher_digits("12a") is False
    
    # Test case sensitivity/specific patterns
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False
    
    # Test non-string inputs (should return False per implementation)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False
    
    # Test empty string against pattern
    matcher_any = rex(r".*")
    assert matcher_any("") is True
    
    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("abc_12") is True
    assert matcher_complex("abc_1") is False
    assert matcher_complex("ABC_12") is False
```


# LLM-generated content at query #51
#--------------------------

```python
import pytest

def test_rex():
    # Test regex for exact match
    matcher_exact = rex("^hello$")
    assert matcher_exact("hello") is True
    assert matcher_exact("hello world") is False
    assert matcher_exact("hi") is False

    # Test regex for prefix match
    matcher_prefix = rex("^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("apre") is False

    # Test regex for suffix match
    matcher_suffix = rex("ing$")
    assert matcher_suffix("running") is True
    assert matcher_suffix("sing") is True
    assert matcher_suffix("singers") is False

    # Test regex for character class
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("12a") is False
    assert matcher_digit("") is False

    # Test non-string input (should return False, not raise error)
    matcher_str = rex(".*")
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["a"]) is False

    # Test empty string match
    matcher_empty = rex("^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #52
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with exact match
    matcher_exact = rex(r"^apple$")
    assert matcher_exact("apple") is True
    assert matcher_exact("apples") is False
    assert matcher_exact("pineapple") is False

    # Test regex matcher with pattern
    matcher_pattern = rex(r"^\d+$")
    assert matcher_pattern("123") is True
    assert matcher_pattern("abc") is False
    assert matcher_pattern("") is False

    # Test regex matcher with partial match (no anchors)
    matcher_partial = rex(r"cat")
    assert matcher_partial("cat") is True
    assert matcher_partial("category") is True
    assert matcher_partial("scat") is True
    assert matcher_partial("dog") is False

    # Test regex matcher with non-string inputs (should return False)
    matcher_str = rex(r".*")
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["apple"]) is False
```


# LLM-generated content at query #53
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher for exact match
    matcher_exact = rex(r"^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("123") is False

    # Test regex matcher for pattern match (starts with)
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("suffix") is False

    # Test regex matcher for digit patterns
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("12a") is False

    # Test type safety: non-string keys should return False (as per isinstance check)
    matcher_str = rex(r".*")
    assert matcher_str("anything") is True
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["a"]) is False

    # Test complex regex (case insensitive via flag if needed, but rex uses re.compile directly)
    # Since rex doesn't expose flags, we test standard regex behavior
    matcher_complex = rex(r"^[a-z]+\d$")
    assert matcher_complex("abc1") is True
    assert matcher_complex("ABC1") is False
    assert matcher_complex("123") is False

    # Test empty string match
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #54
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test non-string types (should return False via isinstance check)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["a"]) is False

    # Test exact match
    matcher_exact = rex("^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False

    # Test case sensitivity (default behavior of re.compile)
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]{2}-\d{3}$")
    assert matcher_complex("ab-123") is True
    assert matcher_complex("abc-123") is False
    assert matcher_complex("ab-12") is False
```


# LLM-generated content at query #55
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string keys that match
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    # Test regex matcher with specific pattern
    matcher_prefix = rex(r"^test_")
    assert matcher_prefix("test_case") is True
    assert matcher_prefix("sample_test") is False

    # Test regex matcher with non-string keys (should return False)
    matcher_any = rex(r".*")
    assert matcher_any("anything") is True
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["list"]) is False

    # Test exact match regex
    matcher_exact = rex("^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("exact_suffix") is False

    # Test case sensitivity (default behavior)
    matcher_case = rex(r"^[a-z]+$")
    assert matcher_case("abc") is True
    assert matcher_case("ABC") is False
```


# LLM-generated content at query #56
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False
    
    matcher_prefix = rex(r"^test_")
    assert matcher_prefix("test_function") is True
    assert matcher_prefix("function_test") is False
    
    # Test non-string types (should return False per implementation)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test complex regex
    matcher_email = rex(r"^[a-z]+@[a-z]+\.com$")
    assert matcher_email("user@gmail.com") is True
    assert matcher_email("user@gmail.net") is False
    assert matcher_email("User@gmail.com") is False
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex matching non-string types (should return False via isinstance check)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["a"]) is False

    # Test case sensitivity
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False

    # Test empty string match
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_rex():
    # Test regex pattern matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False
    assert matcher_digits(123) is False  # Should return False for non-string types

    # Test regex pattern matching specific prefix
    matcher_prefix = rex(r"^test_")
    assert matcher_prefix("test_case") is True
    assert matcher_prefix("testing") is True
    assert matcher_prefix("not_test") is False

    # Test regex pattern with word boundaries or complex structure
    matcher_complex = rex(r"\buser\d\b")
    assert matcher_complex("user1") is True
    assert matcher_complex("user2") is True
    assert matcher_complex("users1") is False
    assert matcher_complex("myuser1") is False

    # Test case sensitivity
    matcher_caps = rex(r"^[A-Z]+$")
    assert matcher_caps("HELLO") is True
    assert matcher_caps("Hello") is False

    # Test none/null behavior (rex handles isinstance check)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str("") is True
    assert matcher_any_str(None) is False
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex non-matching types (should return False for non-strings)
    matcher_any = rex(r".*")
    assert matcher_any("anything") is True
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["list"]) is False

    # Test exact match
    matcher_exact = rex("^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False

    # Test case sensitivity (default behavior)
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("post_test") is False

    # Test regex matching non-string types (should return False per implementation)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test exact match
    matcher_exact = rex("^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("ex") is False
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string matching
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    # Test regex matcher with prefix
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_data") is True
    assert matcher_prefix("data_pre") is False

    # Test regex matcher with non-string inputs (should return False via isinstance check)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["string"]) is False

    # Test regex matcher with specific characters
    matcher_chars = rex(r"[a-z]+")
    assert matcher_chars("hello") is True
    assert matcher_chars("Hello") is False
    assert matcher_chars("123") is False

    # Test exact match
    matcher_exact = rex("^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that satisfy the pattern
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("0") is True
    
    # Test regex non-matching strings
    assert matcher_digit("abc") is False
    assert matcher_digit("12a") is False
    
    # Test regex with specific characters
    matcher_start_a = rex(r"^a.*")
    assert matcher_start_a("apple") is True
    assert matcher_start_a("banana") is False
    
    # Test behavior with non-string types (should return False, not crash)
    assert matcher_digit(123) is False
    assert matcher_digit(None) is False
    assert matcher_digit(["123"]) is False

    # Test complex regex pattern
    matcher_email = rex(r"[^@]+@[^@]+\.[^@]+")
    assert matcher_email("test@example.com") is True
    assert matcher_email("invalid-email") is False
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that follow the pattern
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    # Test regex matching specific characters
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("post_test") is False

    # Test regex with case sensitivity
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False

    # Test non-string inputs (should return False via isinstance check)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test complex regex (word boundaries and specific structure)
    matcher_complex = rex(r"\buser_\d{2}\b")
    assert matcher_complex("user_01") is True
    assert matcher_complex("user_1") is False
    assert matcher_complex("my_user_01_data") is False  # regex.match checks from start of string
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with exact match
    matcher_exact = rex(r"^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("ab") is False

    # Test regex matcher with pattern matching
    matcher_pattern = rex(r"^[0-9]+$")
    assert matcher_pattern("123") is True
    assert matcher_pattern("abc") is False
    assert matcher_pattern("") is False

    # Test regex matcher with partial match (re.match behavior)
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("suffix") is False

    # Test regex matcher with non-string input (should return False per implementation)
    matcher_str_only = rex(r".*")
    assert matcher_str_only(123) is False
    assert matcher_str_only(None) is False
    assert matcher_str_only(["test"]) is False

    # Test regex matcher with empty string and empty pattern
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+\d{2}$")
    assert matcher_complex("test12") is True
    assert matcher_complex("TEST12") is False
    assert matcher_complex("abc1") is False
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_data") is True
    assert matcher_prefix("data_pre") is False

    # Test regex matching non-string types (should return False)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["a"]) is False

    # Test exact match
    matcher_exact = rex(r"^hello$")
    assert matcher_exact("hello") is True
    assert matcher_exact("hello world") is False

    # Test case sensitivity (regex default)
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("ABC") is True
    assert matcher_case("abc") is False
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher for exact match
    matcher_exact = rex(r"^apple$")
    assert matcher_exact("apple") is True
    assert matcher_exact("apples") is False
    assert matcher_exact("pineapple") is False

    # Test regex matcher for prefix/pattern
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("preheat") is True
    assert matcher_prefix("post") is False

    # Test regex matcher with digit patterns
    matcher_digits = rex(r"\d+")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    # Test type safety: should return False for non-string types (as per implementation)
    matcher_str = rex(r".*")
    assert matcher_str("anything") is True
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["a"]) is False

    # Test complex regex (case sensitivity/anchors)
    matcher_complex = rex(r"^[A-Z][a-z]+$")
    assert matcher_complex("Hello") is True
    assert matcher_complex("hello") is False
    assert matcher_complex("HELLO") is False
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that satisfy the pattern
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("0") is True
    
    # Test regex failing on non-matching strings
    assert matcher_digit("abc") is False
    assert matcher_digit("12a") is False
    
    # Test regex pattern for specific prefix
    matcher_prefix = rex(r"^pre_.*")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("post_test") is False
    
    # Test handling of non-string types (should return False via isinstance check)
    assert matcher_digit(123) is False
    assert matcher_digit(None) is False
    assert matcher_digit(["123"]) is False
    
    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("hello_99") is True
    assert matcher_complex("hello_9") is False
    assert matcher_complex("HELLO_99") is False
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex non-matching types (should return False for non-strings)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test exact match
    matcher_exact = rex("^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("abc_12") is True
    assert matcher_complex("abc_1") is False
    assert matcher_complex("ABC_12") is False
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string matches
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex matcher with non-string types (should return False, not crash)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False

    # Test exact match pattern
    matcher_exact = rex(r"exact")
    assert matcher_exact("exact") is True
    assert matcher_exact("exact_match") is False

    # Test case sensitivity (default behavior of re.match)
    matcher_case = rex(r"[a-z]+")
    assert matcher_case("abc") is True
    assert matcher_case("ABC") is False
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex not matching non-string types
    matcher_any = rex(r".*")
    assert matcher_any("anything") is True
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["a"]) is False

    # Test exact match
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False

    # Test complex regex (word boundaries and characters)
    matcher_complex = rex(r"\b[A-Z]{3}\b")
    assert matcher_complex("ABC") is True
    assert matcher_complex("abcd") is False
    assert matcher_complex("AB") is False
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    pattern = r"^user_\d+$"
    matcher = rex(pattern)
    
    assert matcher("user_123") is True
    assert matcher("user_0") is True
    assert matcher("user_abc") is False
    assert matcher("admin_123") is False
    assert matcher("user_123_extra") is False

    # Test non-string types (should return False as per implementation)
    assert matcher(123) is False
    assert matcher(None) is False
    assert matcher(["user_123"]) is False

    # Test exact match
    exact_matcher = rex("^fixed$")
    assert exact_matcher("fixed") is True
    assert exact_matcher("fixed_suffix") is False

    # Test empty string pattern (matches everything)
    empty_pattern_matcher = rex(".*")
    assert empty_pattern_matcher("") is True
    assert empty_pattern_matcher("anything") is True
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_start_a = rex("^a")
    assert matcher_start_a("apple") is True
    assert matcher_start_a("banana") is False
    assert matcher_start_a(123) is False  # Should handle non-string input gracefully
    assert matcher_start_a(None) is False

    # Test with a complex regex (digits only)
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("12345") is True
    assert matcher_digits("123a45") is False
    assert matcher_digits("") is False

    # Test with word boundaries
    matcher_word = rex(r"\bcat\b")
    assert matcher_word("the cat sat") is True
    assert matcher_word("category") is False
    assert matcher_word("concatenate") is False

    # Test with character classes
    matcher_vowel = rex(r"^[aeiou]$")
    assert matcher_vowel("a") is True
    assert matcher_vowel("e") is True
    assert matcher_vowel("b") is False

    # Test equality for identical regex patterns
    matcher1 = rex(r".*")
    matcher2 = rex(r".*")
    assert matcher1 == matcher2
    assert matcher1("anything") is True
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from pyrsistent import pmap, v

def test_transform():
    # Test Case 1: Simple scalar transformation (no path)
    # command is a callable applied to the root
    assert transform(10, [([], inc)]) == 11
    assert transform(10, [([], dec)]) == 9
    assert transform(10, [([], lambda x: x * 2)]) == 20

    # Test Case 2: Simple dictionary update via path
    data = pmap({'a': 1, 'b': 2})
    # Path ['a'] with command inc
    assert transform(data, [['a'], inc]) == pmap({'a': 2, 'b': 2})

    # Test Case 3: Nested dictionary update
    data = pmap({'users': pmap({'alice': {'age': 25}, 'bob': {'age': 30}})})
    path = ['users', 'alice', 'age']
    assert transform(data, [path, inc]) == pmap({
        'users': pmap({
            'alice': pmap({'age': 26}),
            'bob': pmap({'age': 30})
        })
    })

    # Test Case 4: Using regex matcher (rex) in path
    data = pmap({'user_1': 10, 'user_2': 20, 'admin': 50})
    matcher = rex(r'user_.*')
    # If key matches regex, increment the value
    assert transform(data, [[matcher], inc]) == pmap({'user_1': 11, 'user_2': 21, 'admin': 50})

    # Test Case 5: Using any matcher (ny) in path
    data = pmap({'a': 1, 'b': 2})
    assert transform(data, [[ny], inc]) == pmap({'a': 2, 'b': 3})

    # Test Case 6: Discarding elements
    data = pmap({'a': 1, 'b': 2, 'c': 3})
    # Discard keys that match a predicate (e.g., key is 'b')
    assert transform(data, [[lambda k, v: k == 'b', discard]]) == pmap({'a': 1, 'c': 3})
    # Discard using specific key
    assert transform(data, [['a', discard]]) == pmap({'b': 2, 'c': 3})

    # Test Case 7: Binary predicate in path (filtering by value)
    data = pmap({'a': 10, 'b': 20, 'c': 30})
    # If value > 15, increment it
    predicate = lambda k, v: v > 15
    assert transform(data, [[predicate], inc]) == pmap({'a': 10, 'b': 21, 'c': 31})

    # Test Case 8: Unary predicate in path (filtering by key)
    data = pmap({'item_1': 1, 'item_2': 2, 'other': 3})
    predicate_unary = lambda k: k.startswith('item')
    assert transform(data, [[predicate_unary], inc]) == pmap({'item_1': 2, 'item_2': 3, 'other': 3})

    # Test Case 9: Expanding structure (adding new keys via path)
    data = pmap({'a': 1})
    # Path doesn't exist, command returns a value for the new node
    assert transform(data, [['new_key', 'sub_key'], lambda x: 100]) == pmap({'a': 1, 'new_key': pmap({'sub_key': 100})})

    # Test Case 9: List/Vector transformation
    data = v(1, 2, 3)
    # Transform index 1 (the second element)
    assert transform(data, [[1], inc]) == v(1, 3, 3)

    # Test Case 10: Complex multi-step transformation
    data = pmap({'vals': pmap({'x': 1, 'y': 2})})
    transformations = [
        ['vals', 'x', inc],
        ['vals', 'y', dec],
        ['vals', 'z', lambda _: 5]
    ]
    expected = pmap({
        'vals': pmap({'x': 2, 'y': 1, 'z': 5})
    })
    assert transform(data, transformations) == expected

    # Test Case 11: Discarding non-existent key (should not raise error)
    data = pmap({'a': 1})
    assert transform(data, [['non_existent', discard]]) == pmap({'a': 1})
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_inc():
    assert inc(1) == 2
    assert inc(0) == 1
    assert inc(-1) == 0
    assert inc(10.5) == 11.5
    with pytest.raises(TypeError):
        inc("string")
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from pyrsistent import pmap, v

def test_transform():
    # Test Case 1: Basic value replacement in a flat map
    data1 = pmap({'a': 1, 'b': 2})
    trans1 = [(['a'], inc)]
    assert transform(data1, trans1) == pmap({'a': 2, 'b': 2})

    # Test Case 2: Nested path update
    data2 = pmap({'users': pmap({'alice': {'age': 25}, 'bob': {'age': 30}})})
    trans2 = [(['users', 'alice', 'age'], inc)]
    assert transform(data2, trans2) == pmap({'users': pmap({'alice': {'age': 26}, 'bob': {'age': 30}})})

    # Test Case 3: Using rex (regex matcher) to update multiple keys
    data3 = pmap({'user_1': 10, 'user_2': 20, 'admin': 50})
    trans3 = [([rex('user_.*')], inc)]
    # Note: rex returns a function that takes the key. _get_keys_and_values handles unary predicates.
    assert transform(data3, trans3) == pmap({'user_1': 11, 'user_2': 21, 'admin': 50})

    # Test Case 4: Using ny (matches any) to update all values in a structure
    data4 = pmap({'x': 1, 'y': 2})
    trans4 = [([ny], inc)]
    assert transform(data4, trans4) == pmap({'x': 2, 'y': 3})

    # Test Case 5: Using discard command to remove elements
    data5 = pmap({'keep': 1, 'remove_me': 2, 'also_remove': 3})
    trans5 = [(['remove_me'], discard), (['also_remove'], discard)]
    assert transform(data5, trans5) == pmap({'keep': 1})

    # Test Case 6: Using binary predicate in path (key and value)
    # We define a lambda that checks if the value is greater than 15
    data6 = pmap({'a': 10, 'b': 20, 'c': 30})
    trans6 = [([lambda k, v: v > 15], inc)]
    assert transform(dict(data6.items()), trans6) == pmap({'a': 10, 'b': 21, 'c': 31})

    # Test Case 7: Discarding a non-existent key (should not raise error)
    data7 = pmap({'a': 1})
    trans7 = [(['non_existent'], discard)]
    assert transform(data7, trans7) == pmap({'a': 1})

    # Test Case 8: Expanding structure with a new key
    data8 = pmap({})
    trans8 = [(['new_key'], 100)]
    assert transform(data8, trans8) == pmap({'new_key': 100})

    # Test Case 9: Deeply nested vector/sequence update (using index as key)
    # Note: _items for lists uses enumerate, so path can be integers
    data9 = v(v(1, 2), v(3, 4))
    trans9 = [([0, 0], inc)] # Update first element of first vector
    assert transform(data9, trans9) == v(v(2, 2), v(3, 4))

    # Test Case 10: Complex multi-step transformation
    data10 = pmap({'vals': v(1, 2, 3)})
    trans10 = [
        (['vals', 0], inc), # 1 -> 2
        (['vals', 1], dec), # 2 -> 1
        (['vals', 2], lambda x: x * 10) # 3 -> 30
    ]
    assert transform(data10, trans10) == pmap({'vals': v(2, 1, 30)})
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from pyrsistent import pmap, v

def test_transform():
    # Test 1: Basic value transformation (incrementing a number)
    data1 = pmap({'a': 1, 'b': 2})
    trans1 = [('a', inc)]
    assert transform(data1, trans1) == pmap({'a': 2, 'b': 2})

    # Test 2: Nested transformation (deep path)
    data2 = pmap({'users': pmap({'alice': {'score': 10}, 'bob': {'score': 20}})})
    trans2 = [('users', 'alice', 'score', inc)]
    assert transform(data2, trans2) == pmap({'users': pmap({'alice': {'score': 11}, 'bob': {'score': 20}})})

    # Test 3: Transformation using a Rex matcher (regex)
    data3 = pmap({'user_1': 10, 'user_2': 20, 'admin': 50})
    matcher = rex(r'user_.*')
    trans3 = [('user_.*', inc)] # Note: transform uses the matcher as a key spec
    # Because our _get_keys_and_values handles callables, we pass the lambda directly
    trans3 = [(matcher, inc)]
    assert transform(dataly := pmap({'user_1': 10, 'user_2': 20, 'admin': 50}), trans3) == \
           pmap({'user_1': 11, 'user_2': 21, 'admin': 50})

    # Test 4: Transformation using a Unary predicate (filtering/mapping)
    # Match only keys that are integers and increment them
    data4 = pmap({1: 10, 2: 20, 'a': 30})
    trans4 = [(lambda k: isinstance(k, int), inc)]
    assert transform(data4, trans4) == pmap({1: 11, 2: 21, 'a': 30})

    # Test 5: Transformation using a Binary predicate (key and value)
    # Increment values that are greater than 15
    data5 = pmap({'small': 5, 'large': 20})
    trans5 = [(lambda k, v: v > 15, inc)]
    assert transform(data5, trans5) == pmap({'small': 5, 'large': 21})

    # Test 6: Discarding an element
    data6 = pmap({'keep': 1, 'remove': 2})
    trans6 = [('remove', discard)]
    assert transform(data6, trans6) == pmap({'keep': 1})

    # Test 7: Discarding non-existent key (should not raise error)
    data7 = pmap({'a': 1})
    trans7 = [('non_existent', discard)]
    assert transform(data7, trans7) == pmap({'a': 1})

    # Test 8: Transformation on a vector/sequence (using index as key)
    data8 = v(10, 20, 30)
    trans8 = [(0, inc)]
    assert transform(data8, trans8) == v(11, 20, 30)

    # Test 9: Complex nested discard
    data9 = pmap({'a': pmap({'b': 1, 'c': 2})})
    trans9 = [('a', 'b', discard)]
    assert transform(data9, trans9) == pmap({'a': pmap({'c': 2})})

    # Test 10: Expansion of structure (adding new keys via path)
    data10 = pmap({})
    trans10 = [('new_key', inc)]
    # Since 'new_key' doesn't exist, _do_to_path uses pmap() as base for the value
    assert transform(data10, trans10) == pmap({'new_key': 1})

    # Test 11: Any matcher (ny)
    data11 = pmap({'x': 1, 'y': 2})
    trans11 = [(ny, inc)]
    assert transform(data11, trans11) == pmap({'x': 2, 'y': 3})

    # Test 12: Error handling for invalid arity in predicate
    with pytest.raises(ValueError, match="callable in transform path must take 1 or 2 arguments"):
        trans_err = [(lambda x, y, z: True, inc)]
        transform(data11, trans_err)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_inc():
    assert inc(1) == 2
    assert inc(0) == 1
    assert inc(-1) == 0
    assert inc(10.5) == 11.5
    with pytest.raises(TypeError):
        inc("string")
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_dec():
    assert dec(1) == 0
    assert dec(0) == -1
    assert dec(100) == 99
    assert dec(-5) == -6
    assert dec(1.5) == 0.5
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector

def test_discard():
    # Test discarding from a pmap (dictionary-like)
    data_map = pmap({'a': 1, 'b': 2, 'c': 3})
    evolver_map = data_map.evolver()
    discard(evolver_map, 'b')
    result_map = evolver_map.persistent()
    assert result_map == pmap({'a': 1, 'c': 3})
    assert 'b' not in result_map

    # Test discarding from a pvector (list-like)
    data_vec = pvector([10, 20, 30])
    evolver_vec = data_vec.evolver()
    discard(evolver_vec, 1)  # Remove index 1 (value 20)
    result_vec = evolver_vec.persistent()
    assert result_vec == pvector([10, 30])

    # Test discarding a non-existent key in pmap (should not raise KeyError)
    evolver_map_empty = data_map.evolver()
    discard(evolver_map_empty, 'non_existent')
    assert evolver_map_empty.persistent() == data_map

    # Test discarding a non-existent index in pvector (should not raise IndexError)
    evulnerable_vec = data_vec.evolver()
    discard(evulnerable_vec, 99)
    assert vulnerable_vec.persistent() == data_vec

    # Test discarding from an empty structure
    empty_map = pmap()
    evolver_empty = empty_map.evolver()
    discard(evolver_empty, 'any')
    assert evolver_empty.persistent() == pmap()
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_inc():
    assert inc(1) == 2
    assert inc(0) == 1
    assert inc(-1) == 0
    assert inc(5.5) == 6.5
    with pytest.raises(TypeError):
        inc("string")
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from pyrsistent import pmap, v

def test_transform():
    # Test 1: Simple value transformation (no path)
    data1 = pmap({'a': 1})
    trans1 = [([], inc)]
    assert transform(data1, trans1) == pmap({'a': 2})

    # Test 2: Nested dictionary update
    data2 = pmap({'user': pmap({'age': 25, 'name': 'Alice'})})
    trans2 = [(['user', 'age'], inc)]
    assert transform(data2, trans2) == pmap({'user': pmap({'age': 26, 'name': 'Alice'})})

    # Test 3: Using regex matcher (rex) to update multiple keys
    data3 = pmap({'id_1': 10, 'id_2': 20, 'other': 30})
    trans3 = [[rex('id_.*'), inc]]
    # Note: _get_keys_and_values with unary callable uses it as predicate on key
    assert transform(data3, trans3) == psm_map({'id_1': 11, 'id_2': 21, 'other': 30})
    # Helper to fix the expected result name in test context
    def psm_map(d): return pmap(d)

    # Test 4: Discarding an element
    data4 = pmap({'a': 1, 'b': 2})
    trans4 = [(['a'], discard)] # This is a simplification of the logic provided
    # The implementation of _update_structure handles discard specifically when path is empty
    # or via the command. Let's test the specific discard logic in the code.
    data4_alt = pmap({'a': 1, 'b': 2})
    trans4_alt = [([], discard)] # This would require key logic. 
    # Testing the provided discard implementation:
    e = data4_alt.evolver()
    discard(e, 'a')
    assert e.persistent() == pmap({'b': 2})

    # Test 5: Using binary predicate (arity 2) in path
    # Match keys where value is even
    data5 = pmap({'a': 1, 'b': 2, 'c': 4})
    def is_even_val(k, v): return v % 2 == 0
    trans5 = [[is_even_val], inc]
    # Since is_even_val is arity 2, it filters (k, v) pairs. 
    # 'b' and 'c' match, so their values become 3 and 5.
    assert transform(data5, trans5) == pmap({'a': 1, 'b': 3, 'c': 5})

    # Test 6: Using unary predicate (arity 1) in path
    # Match keys that are strings starting with 'x'
    data6 = pmap({'x1': 10, 'y1': 20})
    trans6 = [[lambda k: k.startswith('x'), inc]]
    assert transform(data6, trans6) == pmap({'x1': 11, 'y1': 20})

    # Test 7: Deleting a key from a list/vector via path
    data7 = v(pmap({'a': 1}), pmap({'b': 2}))
    trans7 = [[0, 'a'], discard] # Logic for index-based discard in _update_structure
    # Note: The provided code's 'discard' logic is specifically tuned for empty path or specific behavior.
    # Let's test the structure update with a simple replacement instead to ensure robustness.
    data7_simple = v(1, 2, 3)
    trans7_simple = [[0, 10]]
    assert transform(data7_simple, trans7_simple) == v(10, 2, 3)

    # Test 8: Deeply nested update with expansion (creating pmap where none existed)
    data8 = pmap({'a': 1})
    trans8 = [['new_path', 'sub_key'], lambda x: 99]
    assert transform(data8, trans8) == pmap({'a': 1, 'new_path': pmap({'sub_key': 99})})

    # Test 9: Using ny (any) matcher
    data9 = pmap({'a': 1, 'b': 2})
    trans9 = [[ny, inc]]
    assert transform(data9, trans9) == pmap({'a': 2, 'b': 3})

def psm_map(d): return pmap(d)
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest

def test_inc():
    assert inc(1) == 2
    assert inc(0) == 1
    assert inc(-1) == 0
    assert inc(10.5) == 11.5
    with pytest.raises(TypeError):
        inc("string")
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_ny():
    # ny should return True regardless of the input provided to it
    assert ny(None) is True
    assert ny(1) is True
    assert ny("any string") is True
    assert ny([]) is True
    assert ny({}) is True
    assert ny(True) is True
    assert ny(False) is True
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest

def test_rex():
    # Test exact match
    matcher_exact = rex(r"^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("abc ") is False

    # Test pattern match (starts with)
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("apple") is False

    # Test digit match
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit("") is False

    # Test non-string inputs (should return False, not crash)
    matcher_any = rex(r".*")
    assert matcher_any(None) is False
    assert matcher_any(123) is False
    assert matcher_any(["a"]) is False
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_dec():
    assert dec(1) == 0
    assert dec(0) == -1
    assert dec(100) == 99
    assert dec(-5) == -6
    assert dec(0.5) == -0.5
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest
from pyrsistent import pmap, v

def test_transform():
    # Test Case 1: Simple value update (no path)
    data1 = pmap({'a': 1, 'b': 2})
    # Path is empty, command is inc
    res1 = transform(data1, [])
    assert res1 == pmap({'a': 1, 'b': 2}) # No change if no path provided in chunks loop logic
    
    # Test Case 2: Single level transformation (direct key)
    data2 = pmap({'a': 1, 'b': 2})
    # Path ['a'], command inc
    res2 = transform(data2, [['a'], inc])
    assert res2 == pmap({'a': 2, 'b': 2})

    # Test Case 3: Nested transformation
    data3 = pmap({'users': pmap({'alice': pmap({'score': 10}), 'bob': pmap({'score': 5})})})
    # Path ['users', 'alice', 'score'], command inc
    res3 = transform(data3, [['users', 'alice', 's'], inc]) # Note: _do_to_path uses path[0] as key
    # Wait, the implementation of transform iterates through chunks of 2.
    # If transformations is [['a', 'b'], cmd], path is ['a', 'b'].
    # _do_to_path(structure, ['a', 'b'], cmd) calls _get_keys_and_values on 'a'
    
    data4 = pmap({'a': pmap({'b': 10})})
    res4 = transform(data4, [['a', 'b'], inc])
    assert res4 == pmap({'a': pmap({'b': 11})})

    # Test Case 4: Using Rex matcher in path
    data5 = pmap({'apple': 1, 'banana': 2, 'cherry': 3})
    # Use rex to find keys starting with 'a' and increment their values
    res5 = transform(data5, [[rex('a.*'), inc]])
    assert res5 == pmap({'apple': 2, 'banana': 2, 'cherry': 3})

    # Test Case 5: Using Discard command
    data6 = pmap({'a': 1, 'b': 2, 'c': 3})
    res6 = transform(data6, [['a', 'b'], discard])
    assert res6 == pmap({'c': 3})

    # Test Case 6: Using Binary Predicate (arity 2) in path
    data7 = pmap({'a': 10, 'b': 20, 'c': 5})
    # Find keys where value > 9 and increment them
    res7 = transform(data7, [[lambda k, v: v > 9, inc]])
    assert res7 == pmap({'a': 11, 'b': 21, 'c': 5})

    # Test Case 7: Using Unary Predicate (arity 1) in path
    data8 = v(10, 20, 30) # Vector/Sequence
    # Find elements where index is even and increment them
    res8 = transform(data8, [[lambda i: i % 2 == 0, inc]])
    assert res8 == v(11, 20, 31)

    # Test Case 8: Deeply nested transformation
    data9 = pmap({'outer': pmap({'inner': pmap({'target': 1})})})
    res9 = transform(data9, [['outer', 'inner', 'target'], inc])
    assert res9 == pmap({'outer': pmap({'inner': pmap({'target': 2})})})

    # Test Case 9: Expansion of non-existent path (using pmap creation)
    data10 = pmap({'a': 1})
    # Path ['b', 'c'], command inc. Since 'b' doesn't exist, it uses _EMPTY_SENTINEL logic
    res10 = transform(data10, [['b', 'c'], inc])
    assert res10 == pmap({'a': 1, 'b': pmap({'c': 2})})

    # Test Case 10: Discarding non-existent key (should not raise error)
    data11 = pmap({'a': 1})
    res11 = transform(data11, [['non_existent'], discard])
    assert res11 == pmap({'a': 1})

    # Test Case 11: Ny matcher (matches everything)
    data12 = pmap({'a': 1, 'b': 2})
    res12 = transform(data12, [[ny, inc]])
    assert res12 == pmap({'a': 2, 'b': 3})
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector

def test_discard():
    # Test discarding from a pmap (dictionary-like)
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    e1 = m.evolver()
    discard(e1, 'b')
    result_m = e1.persistent()
    assert 'b' not in result_m
    assert result_m['a'] == 1
    assert result_m['c'] == 3

    # Test discarding a non-existent key (should not raise error)
    e2 = m.evolver()
    discard(e2, 'non_existent')
    result_m2 = e2.persistent()
    assert len(result_m2) == 3
    assert result_m2['a'] == 1

    # Test discarding from a pvector (list-like)
    v = pvector([10, 20, 30])
    e3 = v.evolver()
    discard(e3, 1) # remove index 1 (value 20)
    result_v = e3.persistent()
    assert result_v[0] == 10
    assert result_v[1] == 30
    assert len(result_v) == 2

    # Test discarding from pvector with non-existent index (should not raise error)
    e4 = v.evolver()
    discard(e4, 99)
    result_v2 = e4.persistent()
    assert len(result_v2) == 3
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest

def test_ny():
    """
    Tests the 'ny' matcher which is designed to return True 
    regardless of the input provided.
    """
    assert ny(None) is True
    assert ny(1) is True
    assert ny("test") is True
    assert ny([]) is True
    assert ny({"key": "value"}) is True
    assert ny(True) is True
    assert ny(False) is True
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that match the pattern
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("0") is True
    
    # Test regex non-matching strings
    assert matcher_digits("abc") is False
    assert matcher_digits("12a") is False
    
    # Test regex matching specific prefix
    matcher_prefix = rex(r"^test_")
    assert matcher_prefix("test_function") is True
    assert matcher_prefix("testing") is True
    assert matcher_prefix("my_test_function") is False
    
    # Test case sensitivity (default behavior)
    matcher_case = rex(r"ABC")
    assert matcher_case("ABC") is True
    assert matcher_case("abc") is False
    
    # Test non-string inputs (should return False per implementation)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False
    
    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("user_01") is True
    assert matcher_complex("admin_99") is True
    assert matcher_complex("user_1") is False
    assert matcher_complex("USER_01") is False
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit("") is False

    # Test regex matching specific patterns
    matcher_prefix = rex(r"^user_")
    assert matcher_prefix("user_admin") is True
    assert matcher_prefix("guest_admin") is False

    # Test case sensitivity (default behavior)
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False

    # Test non-string inputs (should return False, not crash)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test empty string with regex
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test complex regex
    matcher_email_simple = rex(r"[a-z]+@[a-z]+\.com")
    assert matcher_email_simple("test@example.com") is True
    assert matcher_email_simple("test@example.net") is False
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest

def test_ny():
    """
    Test that the ny matcher returns True for any input, 
    regardless of its type or value.
    """
    assert ny(None) is True
    assert ny(0) is True
    assert ny(1) is True
    assert ny("") is True
    assert ny("anything") is True
    assert ny([]) is True
    assert ny({}) is True
    assert ny(False) is True
    assert ny([1, 2, 3]) is True
```


# LLM-generated content at query #36
#--------------------------

```python
import pytest

def test_dec():
    assert dec(1) == 0
    assert dec(0) == -1
    assert dec(100) == 99
    assert dec(-5) == -6
    with pytest.raises(TypeError):
        dec("string")
```


# LLM-generated content at query #37
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector

def test_discard():
    # Test discarding from a pmap (dictionary-like)
    data_map = pmap({"a": 1, "b": 2, "c": 3})
    evolver_map = data_map.evolver()
    discard(evolver_map, "a")
    new_map = evolver_map.persistent()
    assert "a" not in new_map
    assert new_map["b"] == 2
    assert new_map["c"] == 3

    # Test discarding a non-existent key (should not raise KeyError)
    evolver_map_empty = data_map.evolver()
    discard(evolver_map_empty, "non_existent")
    new_map_empty = evolver_map_empty.persistent()
    assert new_map_empty == data_map

    # Test discarding from a pvector (list-like) via index
    data_vec = pvector([10, 20, 30])
    evolver_vec = data_vec.evolver()
    discard(evolver_vec, 1) # remove index 1 (value 20)
    new_vec = evolver_vec.persistent()
    assert new_vec == pvector([10, 30])

    # Test discarding from a pvector with invalid index (should not raise error)
    evolver_vec_invalid = data_vec.evolver()
    discard(evolver_vec_invalid, 99)
    new_vec_invalid = evolver_vec_invalid.persistent()
    assert new_vec_invalid == data_vec

    # Test discarding from a standard dict (if used as evolver)
    data_dict = {"x": 100}
    discard(data_dict, "x")
    assert "x" not in data_dict
```


# LLM-generated content at query #38
#--------------------------

```python
import pytest

def test_inc():
    assert inc(1) == 2
    assert inc(0) == 1
    assert inc(-1) == 0
    assert inc(99) == 100
    with pytest.raises(TypeError):
        inc("string")
```


# LLM-generated content at query #39
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector

def test_discard():
    # Test discarding from a pmap (dictionary-like)
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    e = m.evolver()
    discard(e, 'a')
    result = e.persistent()
    assert 'a' not in result
    assert result['b'] == 2
    assert result['c'] == 3

    # Test discarding a non-existent key (should not raise error)
    e_empty = m.evolver()
    discard(e_empty, 'non_existent')
    result_empty = e_empty.persistent()
    assert result_empty == m

    # Test discarding from a pvector (list-like)
    v = pvector([10, 20, 30])
    ev = v.evolver()
    discard(ev, 1)  # remove index 1 (value 20)
    result_v = ev.persistent()
    assert result_v == pvector([10, 30])

    # Test discarding from a pvector with an out of bounds index
    ev_out = v.evolver()
    discard(ev_out, 5)
    assert ev_out.persistent() == v

    # Test discard functionality within the transform logic (integration check)
    # Path: ['a'], Command: discard
    struct = pmap({'a': 1, 'b': 2})
    transformed = transform(struct, [['a'], discard])
    assert 'a' not in transformed
    assert transformed['b'] == 2

    # Test nested discard via transform
    struct_nested = pmap({'outer': pmap({'inner': 1, 'stay': 2})})
    transformed_nested = transform(struct_nested, [['outer', 'inner'], discard])
    assert 'inner' not in transformed_nested['outer']
    assert transformed_nested['outer']['stay'] == 2
```


# LLM-generated content at query #40
#--------------------------

```python
import pytest

def test_dec():
    assert dec(1) == 0
    assert dec(0) == -1
    assert dec(100) == 99
    assert dec(-5) == -6
    with pytest.raises(TypeError):
        dec("string")
```


# LLM-generated content at query #41
#--------------------------

```python
import pytest

def test_rex():
    # Test exact match
    matcher_exact = rex(r"apple")
    assert matcher_exact("apple") is True
    assert matcher_exact("apples") is False
    assert matcher_exact(123) is False
    assert matcher_exact(None) is False

    # Test regex pattern match
    matcher_pattern = rex(r"^a.*e$")
    assert matcher_pattern("apple") is True
    assert matcher_pattern("ace") is True
    assert matcher_pattern("abcde") is True
    assert matcher_pattern("applepie") is False
    assert matcher_pattern("banana") is False

    # Test digit matching
    matcher_digits = rex(r"\d+")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    # Test non-string types handling (should return False, not crash)
    assert rex(r".*")(None) is False
    assert rex(r".*")([]) is False
    assert rex(r".*")({}) is False
```


# LLM-generated content at query #42
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex with case sensitivity
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False

    # Test non-string inputs (should return False via isinstance check)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test complex regex
    matcher_email_sim = rex(r"[a-z]+@[a-z]+\.com")
    assert matcher_email_sim("user@domain.com") is True
    assert matcher_email_sim("user@domain.net") is False
```


# LLM-generated content at query #43
#--------------------------

```python
import pytest

def test_ny():
    # Test that ny always returns True regardless of input
    assert ny(None) is True
    assert ny(1) is True
    assert ny("string") is True
    assert ny([]) is True
    assert ny({}) is True
    assert ny(False) is True
    assert ny(object()) is True
```


# LLM-generated content at query #44
#--------------------------

```python
import pytest

def test_inc():
    assert inc(1) == 2
    assert inc(0) == 1
    assert inc(-1) == 0
    assert inc(10.5) == 11.5
    with pytest.raises(TypeError):
        inc("string")
```


# LLM-generated content at query #45
#--------------------------

```python
import pytest

def test_ny():
    """
    Test the ny matcher, which should return True for any input value.
    """
    assert ny(None) is True
    assert ny(True) is True
    assert ny(False) is False # Wait, no. The function returns True regardless of input.
    # Correction: ny(_) returns True. It doesn't matter what the argument is.
    assert ny(123) is True
    assert ny("any string") is True
    assert ny([]) is True
    assert ny({}) is True
    assert ny(object()) is True
```


# LLM-generated content at query #46
#--------------------------

```python
import pytest

def test_ny():
    """
    Tests that the 'ny' matcher returns True regardless of the input provided.
    """
    assert ny(None) is True
    assert ny(0) is True
    assert ny("") is True
    assert ny([]) is True
    assert ny({}) is True
    assert ny(False) is True
    assert ny("any string") is True
    assert ny(123) is True
    assert ny([1, 2, 3]) is True
```


# LLM-generated content at query #47
#--------------------------

```python
import pytest
from pyrsistent import pmap, v

def test_transform():
    # Setup initial structure: a nested pmap
    # { 'a': 1, 'b': {'c': 2, 'd': 3}, 'e': [10, 20] }
    initial = pmap({
        'a': 1,
        'b': pmap({'c': 2, 'd': 3}),
        'e': v(10, 20)
    })

    # Test Case 1: Simple increment (unary function on path)
    # Path ['a'] -> inc
    t1 = transform(initial, [['a'], inc])
    assert t1['a'] == 2

    # Test Case 2: Nested transformation
    # Path ['b', 'c'] -> inc
    t2 = transform(initial, [['b', 'name_of_key_not_needed_but_path_is', 'c'], inc]) 
    # Wait, the logic for path is [path_elements..., command]
    # Based on _chunks(transformations, 2): path is a list of keys, then command.
    # In transform(structure, transformations), transformations is an iterable of (path, command)
    # where path is a list/sequence.
    
    t2 = transform(initial, [(['b', 'c'], inc)])
    assert t2['b']['c'] == 3
    assert t2['b']['d'] == 3 # untouched

    # Test Case 3: Using rex matcher (regex)
    # Match any key starting with 'a' or 'b' and increment its value
    t3 = transform(initial, [[rex('^[ab]'), inc]])
    assert t3['a'] == 2
    assert t3['b']['c'] == 3 # Note: rex applied to keys in the level. 
    # Actually, rex matches the key. If path is [rex('^[ab]')], 
    # it finds 'a' and 'b', then applies inc to their values.
    # Let's re-verify: _do_to_path(structure, path, command)
    # if path is ['a'], kvs = [( 'a', 1 )]. result = _do_to_path(1, [], inc) -> inc(1) = 2.
    
    t3 = transform(initial, [[rex('^[ab]'), inc]])
    assert t3['a'] == 2
    assert t3['b'] == pmap({'c': 3, 'd': 4}) # b is a pmap, so it recurses.

    # Test Case 4: Discarding an element
    t4 = transform(initial, [['a'], discard])
    assert 'a' not in t4
    assert t4['b']['c'] == 2

    # Test Case 5: Binary predicate (matching key and value)
    # Match keys where value > 1
    def val_gt_1(k, v):
        return v > 1
    
    t5 = transform(initial, [[val_gt_1, inc]])
    # 'a' is 1 (not matched), 'b' is pmap (not matched because it checks items of initial)
    # Wait, _get_keys_and_values uses _items. For pmap, items are (key, value).
    # If val_gt_1(k, v) returns True for 'b', then it calls _do_to_path(v, [], inc)
    assert t5['a'] == 1
    assert t5['b']['c'] == 3
    assert t5['b']['d'] == 4

    # Test Case 6: Using ny (matches everything)
    t6 = transform(initial, [[ny, dec]])
    assert t6['a'] == 0
    assert t6['b']['c'] == 1
    assert t6['e'] == v(9, 19)

    # Test Case 7: Complex nested path and multiple transformations
    transformations = [
        (['b', 'c'], inc),      # b.c becomes 3
        (['e', 0], inc),        # e[0] becomes 11
        (['a'], dec),           # a becomes 0
        (['non_existent'], lambda x: pmap({'new': True})) # expansion
    ]
    t7 = transform(initial, transformations)
    assert t7['b']['c'] == 3
    assert t7['e'][0] == 11
    assert t7['a'] == 0
    assert t7['non_existent'] == pmap({'new': True})

    # Test Case 8: Discarding non-existent key (should not crash)
    t8 = transform(initial, [['ghost', 'path'], discard])
    assert 'ghost' not in t8

    # Test Case 9: Empty path returns command result on structure
    t9 = transform(initial, [[[], lambda x: 'new_root'])
    assert t9 == 'new_root'

    # Test Case 10: Chunking logic check (the input transformations is a flat list of [path, cmd, path, cmd])
    # The code uses _chunks(transformations, 2). So input must be [[path1, cmd1], [path2, cmd2]]
    t10 = transform(initial, [(['a'], inc), (['b', 'c'], dec)])
    assert t10['a'] == 2
    assert t10['b']['c'] == 1
```


# LLM-generated content at query #48
#--------------------------

```python
import pytest

def test_rex():
    # Test regex for matching exact string
    matcher_exact = rex(r"^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact(123) is False
    assert matcher_exact(None) is False

    # Test regex for pattern matching (digits)
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False
    assert matcher_digits(123) is False

    # Test regex for prefix matching
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("aprefix") is False

    # Test regex with character classes
    matcher_class = rex(r"^[a-z]+$")
    assert matcher_class("hello") is True
    assert matcher_class("Hello") is False
    assert matcher_class("h1") is False

    # Test non-string types (should always return False)
    matcher_any = rex(r".*")
    assert matcher_any(["a"]) is False
    assert matcher_any({"key": "val"}) is False
    assert matcher_any(None) is False
```


# LLM-generated content at query #49
#--------------------------

```python
import pytest

def test_inc():
    assert inc(1) == 2
    assert inc(0) == 1
    assert inc(-1) == 0
    assert inc(10.5) == 11.5
    with pytest.raises(TypeError):
        inc("string")
```


# LLM-generated content at query #50
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False
    assert matcher_prefix("pre") is False # r.match matches from start, but doesn't require end unless $ used

    # Test non-string inputs (should return False per implementation)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["a"]) is False

    # Test exact match
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("exact_extra") is False

    # Test case sensitivity (default behavior of re.compile)
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False
```


# LLM-generated content at query #51
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings that match pattern
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("0") is True
    
    # Test regex not matching strings
    assert matcher_digits("abc") is False
    assert matcher_digits("12a") is False
    
    # Test regex not matching non-string types (should return False via isinstance check)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False

    # Test complex regex
    matcher_email = rex(r"^[a-z]+@example\.com$")
    assert matcher_email("test@example.com") is True
    assert matcher_email("test@gmail.com") is False
    assert matcher_email("123@example.com") is False # regex doesn't allow digits in this pattern

    # Test exact match
    matcher_exact = rex("^hello$")
    assert matcher_exact("hello") is True
    assert matcher_exact("hello world") is False
```


