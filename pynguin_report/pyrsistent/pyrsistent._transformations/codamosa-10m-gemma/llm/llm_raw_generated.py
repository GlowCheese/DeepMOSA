####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector

def test_discard():
    # Test discarding from a pmap
    d = {'a': 1, 'b': 2}
    evolver = pmap(d).evolver()
    discard(evolver, 'a')
    result = evolver.persistent()
    assert 'a' not in result
    assert result['b'] == 2
    assert len(result) == 1

    # Test discarding a non-existent key (should not raise KeyError)
    evolver_empty = pmap(d).evolver()
    discard(evolver_empty, 'non_existent')
    result_empty = evolver_empty.persistent()
    assert result_empty == pmap(d)

    # Test discarding from a pvector
    v = pvector([10, 20, 30])
    evolver_v = v.evolver()
    discard(evolver_v, 1)
    result_v = evolver_v.persistent()
    assert result_v == pvector([10, 30])

    # Test discarding from a pvector with out of bounds index
    evolver_v_err = v.evolver()
    discard(evolver_v_err, 5)
    assert evolver_v_err.persistent() == pvector([10, 20, 30])
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_rex():
    # Test rex with a simple string match
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types gracefully

    # Test rex with start/end anchors
    matcher_start = rex(r"^apple")
    assert matcher_start("applepie") is True
    assert matcher_start("pineapple") is False

    # Test rex with complex regex
    matcher_complex = rex(r"[a-z]+_[0-9]+")
    assert matcher_complex("test_123") is True
    assert matcher_complex("TEST_123") is False
    assert matcher_complex("test_abc") is False

    # Test rex with empty string/no match
    matcher_empty = rex(r".*")
    assert matcher_empty("") is True
    assert matcher_empty("anything") is True

    # Test rex with None or other types
    matcher_any_str = rex(r".*")
    assert matcher_any_str(None) is False
    assert matcher_any_str([]) is False
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector

def test_discard():
    # Test discarding from a pmap with an existing key
    m = pmap({'a': 1, 'b': 2})
    e = m.evolver()
    discard(e, 'a')
    assert 'a' not in e
    assert e['b'] == 2
    
    # Test discarding from a pmap with a non-existing key (should not raise KeyError)
    e2 = m.evolver()
    discard(e2, 'non_existent')
    assert 'a' in e2
    assert 'b' in e2
    
    # Test discarding from a pvector with an existing index
    v = pvector([10, 20, 30])
    ev = v.evolver()
    discard(ev, 1)
    assert ev[0] == 10
    assert ev[2] == 30
    with pytest.raises(IndexError):
        _ = ev[1]

    # Test discarding from a pvector with an out of bounds index
    ev2 = v.evolver()
    discard(ev2, 99)
    assert ev2[0] == 10
    assert len(ev2) == 3
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pyrsistent import pmap, v

def test_transform():
    # Test 1: Basic transformation of a simple pmap
    data1 = pmap({'a': 1, 'b': 2})
    # Path ['a'], command inc
    transformations1 = [['a'], inc]
    assert transform(['a'], inc, data1) == pmap({'a': 2, 'b': 2})

    # Test 2: Nested transformation
    data2 = pmap({'users': pmap({'alice': {'age': 25}, 'bob': {'age': 30}})})
    transformations2 = [['users', 'alice', 'age'], inc]
    assert transform(data2, transformations2) == pmap({'users': pmap({'alice': {'age': 2dec_val_placeholder}, 'bob': {'age': 30}})})
    # Correction for the logic:
    assert transform(data2, [['users', 'alice', 'age'], inc]) == pmap({'users': pmap({'alice': {'age': 26}, 'bob': {'age': 30}})})

    # Test 3: Using Rex matcher in path
    data3 = pmap({'user_1': 10, 'user_2': 20, 'other': 30})
    # Match any key starting with 'user_' and increment value
    transformations3 = [[rex('user_.*'), inc]]
    expected3 = pmap({'user_1': 11, 'user_2': 21, 'other': 30})
    assert transform(data3, transformations3) == expected3

    # Test 4: Using Ny (any) matcher
    data4 = pmap({'a': 1, 'b': 2})
    transformations4 = [[ny(None), inc]]
    assert transform(data4, transformations4) == pmap({'a': 2, 'b': 3})

    # Test 5: Using Binary Predicate (arity 2) in path
    data5 = pmap({'a': 1, 'b': 10, 'c': 2})
    # Match keys where value > 5 and increment the key (if command returns new value)
    # Actually, the command is applied to the value. Let's use a lambda.
    transformations5 = [[lambda k, v: v > 5, inc]]
    assert transform(data5, transformations5) == pmap({'a': 1, 'b': 11, 'c': 2})

    # Test 6: Discarding an element
    data6 = pmap({'a': 1, 'b': 2, 'c': 3})
    transformations6 = [['b'], discard]
    assert transform(data6, transformations6) == pmap({'a': 1, 'c': 3})

    # Test 7: Discarding with a matcher
    data7 = pmap({'a': 1, 'b': 2, 'c': 3})
    transformations7 = [[rex('b'), discard]]
    assert transform(data7, transformations7) == pmap({'a': 1, 'c': 3})

    # Test 8: Transformation on a vector (list-like structure)
    data8 = v(1, 2, 3)
    transformations8 = [[0], inc]
    assert transform(data8, transformations 8) == v(2, 2, 3)

    # Test 9: Deeply nested discard
    data9 = pmap({'outer': pmap({'inner': 10})})
    transformations9 = [['outer', 'inner'], discard]
    assert transform(data9, transformations9) == pmap({'outer': pmap()})

    # Test 10: Empty transformations
    data10 = pmap({'a': 1})
    assert transform(data10, []) == data10

# Helper to bridge the discrepancy in the provided snippet's signature 
# (The prompt asks to test 'transform', but the snippet's implementation 
# of transform is actually (structure, transformations))
def transform(structure, transformations):
    return transform_logic(structure, transformations)

# Re-mapping the logic to match the provided code's actual signature: 
# transform(structure, transformations)
def transform_logic(structure, transformations):
    r = structure
    for i in range(0, len(transformations), 2):
        path = transformations[i]
        command = transformations[i+1]
        r = _do_to_path(r, path, command)
    return r

# Note: In the user's provided code, transform(structure, transformations) 
# uses _chunks(transformations, 2). This implies transformations is a flat list.
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector

def test_discard():
    # Test discarding from a pmap (dictionary-like)
    d = pmap({'a': 1, 'b': 2, 'c': 3})
    
    # Test successful discard
    e1 = d.evolver()
    discard(e1, 'b')
    res1 = e1.persistent()
    assert 'b' not in res1
    assert res1['a'] == 1
    assert res1['c'] == 3
    
    # Test discarding non-existent key (should not raise error)
    e2 = d.evolver()
    discard(e2, 'non_existent')
    res2 = e2.persistent()
    assert res2 == d
    
    # Test discarding from a pvector (list-like)
    v = pvector([10, 20, 30])
    e3 = v.evolver()
    discard(e3, 1)
    res3 = e3.persistent()
    assert res3 == pvector([10, 30])
    
    # Test discarding non-existent index in pvector (should not raise error)
    e4 = v.evolver()
    discard(e4, 99)
    res4 = e4.persistent()
    assert res4 == v

    # Test discarding from a standard dict (as evolver behaves like a dict)
    d_std = {'x': 100}
    # Note: dict does not have .evolver(), but the function uses del d[key]
    # We test the logic of the function's try/except block
    e5 = {'x': 100}
    discard(e5, 'x')
    assert 'x' not in e5
    discard(e5, 'y') # Should pass silently
    assert 'x' not in e5
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should return False for non-string types

    # Test with complex regex
    matcher_email = rex(r"^[a-z]+@domain\.com$")
    assert matcher_email("user@domain.com") is True
    assert matcher_email("user123@domain.com") is False
    assert matcher_email("USER@domain.com") is False

    # Test with no match
    matcher_none = rex(r"start")
    assert matcher_none("end") is False

    # Test with empty string and regex
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test with non-string input (ensure it doesn't crash and returns False)
    assert rex(r".*")(None) is False
    assert rex(r".*")([]) is False
    assert rex(r".*")({}) is False
```


# LLM-generated content at query #7
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
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_[0-9]{2}$")
    assert matcher_complex("hello_99") is True
    assert matcher_complex("hello_9") is False
    assert matcher_complex("Hello_99") is False
    assert matcher_complex("abc_def") is False
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types gracefully

    # Test with regex pattern
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False
    assert matcher_prefix(None) is False

    # Test with complex pattern
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("hello_99") is True
    assert matcher_complex("hello_9") is False
    assert matcher_complex("HELLO_99") is False

    # Test with empty string match
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test with non-string input (edge case)
    assert rex(r".*")(None) is False
    assert rex(r".*")([]) is False
    assert rex(r".*")({}) is False
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types
    assert matcher_digit(None) is False

    # Test with a prefix match
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False
    assert matcher_prefix(["pre_"]) is False

    # Test with a complex pattern
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("hello_99") is True
    assert matcher_complex("hello_9") is False
    assert matcher_complex("HELLO_99") is False
    assert matcher_complex("item_abc") is False

    # Test with an empty string pattern
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test with a pattern that matches everything
    matcher_all = rex(r".*")
    assert matcher_all("anything") is True
    assert matcher_all("") is True
    assert matcher_all(True) is False
```


# LLM-generated content at query #10
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
    assert matcher_digit(["123"]) is False
    
    # Test complex regex
    matcher_email = rex(r"^[a-z]+@domain\.com$")
    assert matcher_email("user@domain.com") is True
    assert matcher_email("user123@domain.com") is False
    assert matcher_email("USER@domain.com") is False
    
    # Test exact match regex
    matcher_exact = rex(r"^exact_string$")
    assert matcher_exact("exact_string") is True
    assert matcher_exact("exact_string_extra") is False
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string keys that match
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits(123) is False  # Should handle non-string types safely

    # Test regex matcher with specific pattern
    matcher_prefix = rex(r"^test_")
    assert matcher_prefix("test_case") is True
    assert matcher_prefix("test") is True
    assert matcher_prefix("production_case") is False

    # Test regex matcher with complex pattern
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("abc_12") is True
    assert matcher_complex("abc_1") is False
    assert matcher_complex("ABC_12") is False

    # Test with non-string input (should return False, not crash)
    matcher_any = rex(r".*")
    assert matcher_any(None) is False
    assert matcher_any(True) is False
    assert matcher_any(1) is False
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types safely
    assert matcher_digit(None) is False

    # Test with a prefix match
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("suffix") is False

    # Test with complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("test_1") is False
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("abc_def") is False

    # Test with non-string types (should return False, not raise error)
    assert rex(r".*")(True) is False
    assert rex(r".*")([]) is False
    assert rex(r".*")({}) is False
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_rex():
    # Test basic regex matching with strings
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit("") is False

    # Test regex matching with non-string types (should return False)
    assert matcher_digit(123) is False
    assert matcher_digit(None) is False
    assert matcher_digit(["123"]) is False

    # Test complex regex
    matcher_email = rex(r"^[a-z]+@example\.com$")
    assert matcher_email("user@example.com") is True
    assert matcher_email("user@gmail.com") is False
    assert matcher_email("USER@example.com") is False

    # Test case sensitivity/anchoring
    matcher_start = rex(r"^Start")
    assert matcher_start("Start here") is True
    assert matcher_start("The Start") is False

    # Test empty string matching
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with exact match
    matcher_exact = rex(r"apple")
    assert matcher_exact("apple") is True
    assert matcher_exact("apple_pie") is False
    assert matcher_exact("banana") is False

    # Test regex matcher with pattern
    matcher_pattern = rex(r"^[0-9]+$")
    assert matcher_pattern("123") is True
    assert matcher_pattern("123a") is False
    assert matcher_pattern("") is False

    # Test regex matcher with partial match (as re.match starts from beginning)
    matcher_prefix = rex(r"pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("unprefix") is False

    # Test regex matcher with non-string types (should return False, not raise error)
    matcher_str_only = rex(r".*")
    assert matcher_str_only("anything") is True
    assert matcher_str_only(123) is False
    assert matcher_str_only(None) is False
    assert matcher_str_only(["list"]) is False

    # Test regex matcher with complex pattern
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("test_1") is False
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("abc_def") is False
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with string matching
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    # Test regex matcher with partial matches (must match from start)
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("suffix") is False

    # Test regex matcher with case sensitivity
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False

    # Test regex matcher with non-string types (should return False, not crash)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["a"]) is False
    assert matcher_any(True) is False

    # Test regex matcher with complex patterns
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("test_1") is False
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("abc_def") is False
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits(123) is False  # Should handle non-string types gracefully

    # Test with a pattern match
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_data") is True
    assert matcher_prefix("data_pre") is False

    # Test with complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("test_1") is False

    # Test with no match (empty pattern)
    matcher_empty = rex(r"")
    assert matcher_empty("") is True
    assert matcher_empty("anything") is False

    # Test with non-string input (None, int, list)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("hello") is True
    assert matcher_any_str(None) is False
    assert matcher_any_str(123) is False
    assert matcher_any_str(["a"]) is False
```


# LLM-generated content at query #17
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

    # Test regex matching non-string types (should return False)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["a"]) is False

    # Test exact match
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_[0-9]{2}$")
    assert matcher_complex("abc_12") is True
    assert matcher_complex("abc_1") is False
    assert matcher_complex("ABC_12") is False
    assert matcher_complex("abc_123") is False
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r'^\d+$')
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r'^pre_')
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex not matching non-string types (should return False, not crash)
    matcher_any_str = rex(r'.*')
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(['a']) is False

    # Test exact match
    matcher_exact = rex(r'^exact$')
    assert matcher_exact("exact") is True
    assert matcher_exact("exact_extra") is False

    # Test case sensitivity
    matcher_case = rex(r'^[A-Z]+$')
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_rex():
    # Test regex for exact match
    matcher_exact = rex("^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("abc ") is False

    # Test regex for prefix match
    matcher_prefix = rex("^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("aprefix") is False

    # Test regex for pattern match
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("12a") is False
    assert matcher_digits("") is False

    # Test non-string inputs (should return False, not raise error)
    matcher_str = rex(".*")
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["a"]) is False

    # Test empty string match
    matcher_empty = rex("^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r'^\d+$')
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r'^pre_')
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex matching non-string types (should return False per implementation)
    matcher_any_str = rex(r'.*')
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["a"]) is False

    # Test complex regex
    matcher_email = rex(r'^[a-z]+@[a-z]+\.com$')
    assert matcher_email("user@example.com") is True
    assert matcher_email("user@example.net") is False
    assert matcher_email("User@example.com") is False  # Case sensitive
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_rex():
    # Test regex for exact match
    matcher_exact = rex("^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("abc ") is False

    # Test regex for prefix
    matcher_prefix = rex("^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("aprefix") is False

    # Test regex for pattern
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False

    # Test regex for character class
    matcher_char = rex("[a-z]")
    assert matcher_char("a") is True
    assert matcher_char("A") is False

    # Test non-string input (should return False, not raise error)
    matcher_str = rex(".*")
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["a"]) is False

    # Test empty string
    matcher_empty = rex("^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with exact match
    matcher_exact = rex(r"^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("abc ") is False

    # Test regex matcher with partial match (start of string)
    matcher_start = rex(r"^abc")
    assert matcher_start("abc") is True
    assert matcher_start("abcde") is True
    assert matcher_start("zabc") is False

    # Test regex matcher with character class
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("12a") is False
    assert matcher_digit("") is False

    # Test regex matcher with non-string types (should return False, not error)
    matcher_str = rex(r".*")
    assert matcher_str("any_string") is True
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["a"]) is False

    # Test regex matcher with complex pattern
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("test_1") is False
    assert matcher_complex("test_abc") is False
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with exact match
    matcher_exact = rex(r"^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("abc ") is False

    # Test regex matcher with prefix
    matcher_prefix = rex(r"^user_")
    assert matcher_prefix("user_123") is True
    assert matcher_prefix("admin_123") is False

    # Test regex matcher with digit pattern
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("12345") is True
    assert matcher_digits("123a5") is False

    # Test regex matcher with non-string input (should return False)
    matcher_str_only = rex(r".*")
    assert matcher_str_only(123) is False
    assert matcher_str_only(None) is False
    assert matcher_str_only(["abc"]) is False

    # Test regex matcher with empty string
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test complex regex (case insensitive via flag is not in rex, but testing pattern)
    matcher_complex = rex(r"^[A-Z]{2}-\d{3}$")
    assert matcher_complex("AB-123") is True
    assert matcher_complex("abc-123") is False
    assert matcher_complex("A-123") is False
    assert matcher_complex("AB-12") is False
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_rex():
    # Test rex with a simple string match
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types gracefully
    assert matcher_digit(None) is False

    # Test rex with a pattern containing letters and numbers
    matcher_prefix = rex(r"test_\w+")
    assert matcher_prefix("test_abc") is True
    assert matcher_prefix("test_123") is True
    assert matcher_prefix("other_abc") is False
    assert matcher_prefix("") is False

    # Test rex with exact match
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("exact_extra") is False

    # Test rex with complex regex (anchors and character classes)
    matcher_complex = rex(r"^[a-z]{3}-[0-9]$")
    assert matcher_complex("abc-1") is True
    assert matcher_complex("abcd-1") is False
    assert matcher_complex("abc-a") is False
    assert matcher_complex("ABC-1") is False

    # Test rex with empty string pattern
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_start_a = rex("^a")
    assert matcher_start_a("apple") is True
    assert matcher_start_a("banana") is False
    assert matcher_start_a(123) is False  # Test non-string handling
    assert matcher_start_a(None) is False

    # Test with a regex pattern for digits
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("123a") is False
    assert matcher_digits("") is False

    # Test with a more complex pattern
    matcher_email_part = rex(r".+@.+\..+")
    assert matcher_email_part("test@example.com") is True
    assert matcher_email_part("invalid-email") is False

    # Test with a pattern that matches nothing
    matcher_empty = rex("^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test with case sensitivity (default behavior)
    matcher_case = rex("^ABC")
    assert matcher_case("ABC") is True
    assert matcher_case("abc") is False
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with exact match
    matcher_exact = rex(r"apple")
    assert matcher_exact("apple") is True
    assert matcher_exact("apple_pie") is False
    assert matcher_exact("banana") is False

    # Test regex matcher with partial match (start of string)
    matcher_start = rex(r"pre")
    assert matcher_start("prefix") is True
    assert matcher_start("superprefix") is False

    # Test regex matcher with wildcard
    matcher_wildcard = rex(r"a.*z")
    assert matcher_wildcard("abcz") is True
    assert matcher_wildcard("az") is True
    assert matcher_wildcard("abc") is False

    # Test regex matcher with character classes
    matcher_class = rex(r"[0-9]+")
    assert matcher_class("123") is True
    assert matcher_class("abc") is False

    # Test regex matcher with non-string input (should return False, not crash)
    matcher_str = rex(r".*")
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["test"]) is False

    # Test regex matcher with empty string
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching for strings
    pattern = r"^abc\d+$"
    matcher = rex(pattern)
    
    assert matcher("abc123") is True
    assert matcher("abc") is False
    assert matcher("abc123def") is False
    assert matcher("xyz123") is False
    
    # Test non-string types (should return False as per implementation)
    assert matcher(123) is False
    assert matcher(None) is False
    assert matcher(["abc123"]) is False

    # Test exact match
    exact_matcher = rex("^fixed$")
    assert exact_matcher("fixed") is True
    assert exact_matcher("fixed_extra") is False

    # Test case sensitivity (default regex behavior)
    case_matcher = rex("^abc$")
    assert case_matcher("abc") is True
    assert case_matcher("ABC") is False

    # Test empty string match
    empty_matcher = rex("^$")
    assert empty_matcher("") is True
    assert empty_matcher(" ") is False
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest

def test_rex():
    # Test regex for exact match
    matcher_exact = rex("^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("abc ") is False

    # Test regex for prefix match
    matcher_prefix = rex("^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("aprefix") is False

    # Test regex for pattern match
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("12a") is False
    assert matcher_digits("") is False

    # Test behavior with non-string types (should return False)
    matcher_str = rex(".*")
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["abc"]) is False

    # Test behavior with empty string (if regex allows)
    matcher_empty = rex("^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_rex():
    # Test basic regex matching with strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False

    # Test prefix matching
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("append") is False

    # Test non-string types (should return False, not raise error)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("hello") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test case sensitivity
    matcher_case = rex(r"ABC")
    assert matcher_case("ABC") is True
    assert matcher_case("abc") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_[0-9]{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("test_1") is False
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("abc_def") is False
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest

def test_rex():
    # Test rex with a simple string match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types gracefully

    # Test rex with a complex regex
    matcher_email = rex(r"^[a-z]+@domain\.com$")
    assert matcher_email("user@domain.com") is True
    assert matcher_email("user@other.com") is False
    assert matcher_email("USER@domain.com") is False  # Case sensitive

    # Test rex with empty string and regex
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test rex with non-string input (should return False as per implementation)
    matcher_any = rex(r".*")
    assert matcher_any("anything") is True
    assert matcher_any(None) is False
    assert matcher_any(["list"]) is False
    assert matcher_any(1) is False

    # Test rex with character classes
    matcher_chars = rex(r"[a-z]")
    assert matcher_chars("a") is True
    assert matcher_chars("A") is False
    assert matcher_chars("1") is False
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pyrsistent import pmap, v

def test_discard():
    # Test discarding an existing key in a pmap
    d = pmap({'a': 1, 'b': 2})
    e = d.evolver()
    discard(e, 'a')
    result = e.persistent()
    assert 'a' not in result
    assert result['b'] == 2

    # Test discarding a non-existent key (should not raise KeyError)
    e2 = d.evolver()
    discard(e2, 'non_existent')
    result2 = e2.persistent()
    assert result2 == d

    # Test discarding from a vector (using index)
    v_struct = v(10, 20, 30)
    e_v = v_struct.evolver()
    discard(e_v, 1)
    result_v = e_v.persistent()
    assert result_v == v(10, 30)

    # Test discarding a non-existent index in a vector
    e_v2 = v_struct.evolver()
    discard(e_v2, 10)
    result_v2 = e_v2.persistent()
    assert result_v2 == v(10, 20, 30)

    # Test discarding from an empty structure
    e_empty = pmap().evolver()
    discard(e_empty, 'any')
    assert e_empty.persistent() == pmap()
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector

def test_discard():
    # Test discarding from a pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    e = m.evolver()
    discard(e, 'a')
    discard(e, 'b')
    result = e.persistent()
    assert result == pmap({'c': 3})

    # Test discarding a non-existent key (should not raise KeyError)
    e2 = m.evolver()
    discard(e2, 'non_existent')
    assert e2.persistent() == m

    # Test discarding from a pvector
    v = pvector([10, 20, 30])
    ev = v.evolver()
    discard(ev, 1)  # discard index 1 (value 20)
    result_v = ev.persistent()
    assert result_v == pvector([10, 30])

    # Test discarding an index that doesn't exist in vector
    ev2 = v.evolver()
    discard(ev2, 5)
    assert ev2.persistent() == v

    # Test discarding from an empty structure
    empty_m = pmap()
    e3 = empty_m.evolver()
    discard(e3, 'anything')
    assert e3.persistent() == empty_m
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types gracefully

    # Test with a pattern match
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("post_test") is False
    assert matcher_prefix(None) is False

    # Test with complex regex
    matcher_email = rex(r"[a-z]+@[a-z]+\.com$")
    assert matcher_email("user@test.com") is True
    assert matcher_email("user@test.net") is False
    assert matcher_email("123@test.com") is False

    # Test with empty string and non-string types
    matcher_any_str = rex(r".*")
    assert matcher_any_str("") is True
    assert matcher_any_str("anything") is True
    assert matcher_any_str(True) is False
    assert matcher_any_str([]) is False
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types gracefully

    # Test with a prefix match
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("suffix") is False

    # Test with complex regex
    matcher_email = rex(r"^[a-z]+@[a-z]+\.com$")
    assert matcher_email("test@example.com") is True
    assert matcher_email("TEST@example.com") is False
    assert matcher_email("test@example.org") is False

    # Test with non-string input (should return False, not raise error)
    assert rex(r".*")(None) is False
    assert rex(r".*")([]) is False
    assert rex(r".*")({}) is False

    # Test with exact match
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector

def test_discard():
    # Test discarding from a pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    evolver = m.evolver()
    discard(evolver, 'a')
    discard(evolver, 'b')
    result = evolver.persistent()
    assert result == pmap({'c': 3})

    # Test discarding a non-existent key (should not raise KeyError)
    evolver_empty = pmap({'a': 1}).evolver()
    discard(evolver_empty, 'non_existent')
    assert evolver_empty.persistent() == pmap({'a': 1})

    # Test discarding from a pvector
    v = pvector([10, 20, 30])
    evolver_v = v.evolver()
    discard(evolver_v, 1)
    assert evolver_v.persistent() == pvector([10, 30])

    # Test discarding from an empty structure
    empty_m = pmap()
    evolver_empty_m = empty_m.evolver()
    discard(evolver_empty_m, 'anything')
    assert evolver_empty_m.persistent() == pmap()
```


# LLM-generated content at query #6
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

    # Test regex matching non-string types (should return False via isinstance check)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test complex regex
    matcher_email = rex(r"^[a-z]+@[a-z]+\.com$")
    assert matcher_email("user@example.com") is True
    assert matcher_email("user@example.net") is False
    assert matcher_email("User@example.com") is False  # Case sensitive
```


# LLM-generated content at query #7
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

    # Test regex matching non-string types (should return False via isinstance check)
    matcher_any = rex(r".*")
    assert matcher_any(123) is False
    assert matcher_any(None) is False
    assert matcher_any(["a"]) is False

    # Test ny (any) matcher
    assert ny("anything") is True
    assert ny(123) is True
    assert ny(None) is True

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("hello_99") is True
    assert matcher_complex("hello_9") is False
    assert matcher_complex("HELLO_99") is False
    assert matcher_complex("abc_def") is False
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector

def test_discard():
    # Test discarding from a pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    e = m.evolver()
    discard(e, 'a')
    discard(e, 'b')
    result_map = e.persistent()
    assert result_map == pmap({'c': 3})

    # Test discarding a non-existent key (should not raise error)
    e2 = m.evolver()
    discard(e2, 'non_existent')
    assert e2.persistent() == m

    # Test discarding from a pvector
    v = pvector([10, 20, 30])
    ev = v.evolver()
    discard(ev, 1)
    result_vec = ev.persistent()
    assert result_vec == pvector([10, 30])

    # Test discarding from a pvector with non-existent index
    ev2 = v.evolver()
    discard(ev2, 5)
    assert ev2.persistent() == v

    # Test discarding from a dictionary (standard dict)
    d = {'x': 100, 'y': 200}
    discard(d, 'x')
    assert d == {'y': 200}

    # Test discarding from a dictionary with non-existent key
    d2 = {'x': 100}
    discard(d2, 'z')
    assert d2 == {'x': 100}
```


# LLM-generated content at query #9
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

    # Test regex matching non-string types (should return False)
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
    assert matcher_complex("hello_99") is True
    assert matcher_complex("hello_9") is False
    assert matcher_complex("HELLO_99") is False
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector

def test_discard():
    # Test discarding from a pmap
    m = pmap({'a': 1, 'b': 2})
    e = m.evolver()
    discard(e, 'a')
    assert 'a' not in e.persistent()
    assert e.persistent()['b'] == 2

    # Test discarding a non-existent key (should not raise KeyError)
    e2 = m.evolver()
    discard(e2, 'non_existent')
    assert e2.persistent() == m

    # Test discarding from a pvector
    v = pvector([10, 20, 30])
    ev = v.evolver()
    discard(ev, 1)
    assert ev.persistent() == pvector([10, 30])

    # Test discarding an index that doesn't exist in pvector
    ev2 = v.evolver()
    discard(ev2, 5)
    assert ev2.persistent() == v
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits(123) is False  # Should return False for non-string types

    # Test with a prefix match
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("post") is False

    # Test with non-string types (must be False per implementation)
    assert rex(r".*")(None) is False
    assert rex(r".*")([]) is False
    assert rex(r".*")({}) is False

    # Test with complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("test_1") is False
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("abc_def") is False

    # Test with empty string match
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
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

    # Test regex matching non-string types (should return False)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test exact match
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]{3}_\d{2}$")
    assert matcher_complex("abc_12") is True
    assert matcher_complex("abcd_12") is False
    assert matcher_complex("abc_1") is False
    assert matcher_complex("ABC_12") is False
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_rex():
    # Test basic regex matching (starts with 'abc')
    matcher_abc = rex(r'^abc')
    assert matcher_abc("abc") is True
    assert matcher_abc("abcd") is True
    assert matcher_abc("abcde") is True
    assert matcher_abc("def") is False
    assert matcher_abc("123abc") is False

    # Test numeric regex matching
    matcher_digits = rex(r'^\d+$')
    assert matcher_digits("123") is True
    assert matcher_digits("12a") is False
    assert matcher_digits("") is False

    # Test case sensitivity
    matcher_case = rex(r'^[A-Z]+$')
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False

    # Test non-string inputs (should return False, not raise error)
    assert matcher_abc(123) is False
    assert matcher_abc(None) is False
    assert matcher_abc(['abc']) is False

    # Test complex regex (word boundary and specific characters)
    matcher_complex = rex(r'\buser_\d+\b')
    assert matcher_complex("user_1") is True
    assert matcher_complex("user_999") is True
    assert matcher_complex("myuser_1") is False
    assert matcher_complex("user_abc") is False

    # Test empty string matching
    matcher_empty = rex(r'^$')
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_rex():
    # Test rex with a simple string match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should return False for non-string types
    assert matcher_digit(None) is False

    # Test rex with a complex pattern
    matcher_email = rex(r"^[a-z]+@[a-z]+\.com$")
    assert matcher_email("test@example.com") is True
    assert matcher_email("test@example.net") is False
    assert matcher_email("TEST@example.com") is False  # Case sensitive

    # Test rex with empty string and non-matching string
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test rex with start/end anchors
    matcher_start = rex(r"^prefix")
    assert matcher_start("prefix_suffix") is True
    assert matcher_start("not_prefix") is False

    # Test rex with character classes
    matcher_chars = rex(r"[a-v]")
    assert matcher_chars("a") is True
    assert matcher_chars("z") is False
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching strings
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("12a") is False
    assert matcher_digits("") is False

    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test regex matching non-string types (should return False, not raise error)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test complex regex
    matcher_email = rex(r"[a-z]+@[a-z]+\.com")
    assert matcher_email("user@domain.com") is True
    assert matcher_email("user@domain.org") is False
    assert matcher_email("User@domain.com") is False  # Case sensitive
```


# LLM-generated content at query #16
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

    # Test regex matching non-string types (should return False)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["list"]) is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_[0-9]{2}$")
    assert matcher_complex("abc_12") is True
    assert matcher_complex("abc_1") is False
    assert matcher_complex("ABC_12") is False
    assert matcher_complex("abc_123") is False

    # Test empty string matching
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_rex():
    # Test rex with a simple string match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types gracefully
    assert matcher_digit(None) is False

    # Test rex with a pattern match
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False

    # Test rex with complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("hello_99") is True
    assert matcher_complex("hello_9") is False
    assert matcher_complex("HELLO_99") is False

    # Test rex with no match (empty regex)
    matcher_empty = rex("")
    assert matcher_empty("") is True
    assert matcher_empty("anything") is False

    # Test rex with non-string input (edge cases)
    assert rex(r".*")(True) is False
    assert rex(r".*")([]) is False
    assert rex(r".*")({}) is False
```


# LLM-generated content at query #18
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

    # Test regex matching non-string types (should return False)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["a", "b"]) is False

    # Test case sensitivity
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False

    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("user_01") is True
    assert matcher_complex("user_1") is False
    assert matcher_complex("USER_01") is False
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_abc = rex(r"abc")
    assert matcher_abc("abc") is True
    assert matcher_abc("abcd") is True
    assert matcher_abc("ab") is False
    assert matcher_abc(123) is False  # Should handle non-string keys
    assert matcher_abc(None) is False

    # Test with regex pattern (digits)
    matcher_digits = rex(r"\d+")
    assert matcher_digits("123") is True
    assert matcher_digits("abc") is False
    assert matcher_digits("") is False
    assert matcher_digits(123) is False

    # Test with complex regex (start and end)
    matcher_complex = rex(r"^start_.*_end$")
    assert matcher_complex("start_anything_end") is True
    assert matcher_complex("start_end") is True
    assert matcher_complex("start_middle") is False
    assert matcher_complex("middle_end") is False

    # Test with empty string pattern
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test with character classes
    matcher_chars = rex(r"[a-z]+")
    assert matcher_chars("hello") is True
    assert matcher_chars("HELLO") is False
    assert matcher_chars("123") is False
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with exact match
    matcher_exact = rex(r"^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("abc ") is False

    # Test regex matcher with partial match (start of string)
    matcher_start = rex(r"^pre")
    assert matcher_start("prefix") is True
    assert matcher_start("pre") is True
    assert matcher_start("aprefix") is False

    # Test regex matcher with pattern
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False

    # Test regex matcher with non-string input (should return False, not crash)
    matcher_str = rex(r".*")
    assert matcher_str(123) is False
    assert matcher_str(None) is False
    assert matcher_str(["abc"]) is False

    # Test regex matcher with empty string
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test regex matcher with complex pattern
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("test_1") is False
    assert matcher_complex("test_abc") is False
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_rex():
    # Test rex with a simple string match
    matcher_start_a = rex("^a")
    assert matcher_start_a("apple") is True
    assert matcher_start_a("banana") is False
    assert matcher_start_a(123) is False  # Should handle non-string types gracefully
    assert matcher_start_a(None) is False

    # Test rex with a complex regex
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("12345") is True
    assert matcher_digits("123a45") is False
    assert matcher_digits("") is False

    # Test rex with exact match
    matcher_exact = rex("^hello$")
    assert matcher_exact("hello") is True
    assert matcher_exact("hello world") is False

    # Test rex with character classes
    matcher_vowel_start = rex("^[aeiou]")
    assert matcher_vowel_start("orange") is True
    assert matcher_vowel_start("pear") is False

    # Test rex with empty string pattern
    matcher_empty = rex("^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_rex():
    # Test exact match with string
    matcher_exact = rex(r"^apple$")
    assert matcher_exact("apple") is True
    assert matcher_exact("apples") is False
    assert matcher_exact("pineapple") is False

    # Test pattern match with regex
    matcher_pattern = rex(r"^a.*e$")
    assert matcher_pattern("apple") is True
    assert matcher_pattern("ace") is True
    assert matcher_pattern("abc") is False

    # Test non-string input (should return False, not crash)
    matcher_str_only = rex(r"^\d+$")
    assert matcher_str_only(123) is False
    assert matcher_str_only(None) is False
    assert matcher_str_only([]) is False

    # Test numeric string match
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("123a") is False

    # Test case sensitivity (default behavior)
    matcher_case = rex(r"^ABC$")
    assert matcher_case("ABC") is True
    assert matcher_case("abc") is False
```


# LLM-generated content at query #23
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
    
    # Test regex matcher with non-string keys (should return False)
    assert matcher_digit(123) is False
    assert matcher_digit(None) is False
    assert matcher_digit(["123"]) is False

    # Test regex matcher with specific prefix
    matcher_prefix = rex(r"^test_")
    assert matcher_prefix("test_function") is True
    assert matcher_prefix("testing") is True
    assert matcher_prefix("my_test_function") is False

    # Test regex matcher with complex pattern
    matcher_complex = rex(r"^[a-z]+_[0-9]{2}$")
    assert matcher_complex("data_99") is True
    assert matcher_complex("data_9") is False
    assert matcher_complex("DATA_99") is False
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matcher with strings that match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("0") is True
    
    # Test regex matcher with strings that do not match
    assert matcher_digit("abc") is False
    assert matcher_digit("12a") is False
    
    # Test regex matcher with non-string types (should return False)
    assert matcher_digit(123) is False
    assert matcher_digit(None) is False
    assert matcher_digit(["123"]) is False
    
    # Test regex matcher with prefix matching
    matcher_prefix = rex(r"^pre")
    assert matcher_prefix("prefix_value") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("post") is False
    
    # Test regex matcher with complex pattern
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("test_1") is False
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("abc_123") is False
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest

def test_rex():
    # Test regex matching for strings
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types gracefully
    assert matcher_digit(None) is False

    # Test regex matching for prefix
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False
    assert matcher_prefix("") is False

    # Test regex matching for exact match
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("exact_suffix") is False

    # Test regex matching for complex pattern
    matcher_complex = rex(r"^[a-z]+_[0-9]{2}$")
    assert matcher_complex("abc_12") is True
    assert matcher_complex("abc_1") is False
    assert matcher_complex("ABC_12") is False
    assert matcher_complex("abc_ab") is False

    # Test with empty string pattern
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types gracefully

    # Test with start/end anchors
    matcher_start = rex(r"^prefix")
    assert matcher_start("prefix_suffix") is True
    assert matcher_start("not_prefix") is False

    # Test with complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("hello_99") is True
    assert matcher_complex("hello_9") is False
    assert matcher_complex("HELLO_99") is False
    assert matcher_complex("test_abc") is False

    # Test with empty string pattern
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False

    # Test with None or other types
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str("") is True
    assert matcher_any_str(None) is False
    assert matcher_any_str([]) is False
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_simple = rex(r"abc")
    assert matcher_simple("abc") is True
    assert matcher_simple("abcd") is True
    assert matcher_simple("ab") is False
    assert matcher_simple(123) is False
    assert matcher_simple(None) is False

    # Test with regex pattern (digits)
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("12a") is False
    assert matcher_digits("") is False
    assert matcher_digits(123) is False

    # Test with case sensitivity
    matcher_case = rex(r"Hello")
    assert matcher_case("Hello") is True
    assert matcher_case("hello") is False

    # Test with non-string types (should return False, not raise error)
    matcher_any = rex(r".*")
    assert matcher_any("anything") is True
    assert matcher_any("") is True
    assert matcher_any(None) is False
    assert matcher_any([]) is False
    assert matcher_any(True) is False
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest

def test_rex():
    # Test regex for exact match
    matcher_exact = rex("^abc$")
    assert matcher_exact("abc") is True
    assert matcher_exact("abcd") is False
    assert matcher_exact("abc ") is False

    # Test regex for prefix match
    matcher_prefix = rex("^pre")
    assert matcher_prefix("prefix") is True
    assert matcher_prefix("pre") is True
    assert matcher_prefix("apple") is False

    # Test regex for digit matching
    matcher_digits = rex(r"^\d+$")
    assert matcher_digits("123") is True
    assert matcher_digits("12a") is False
    assert matcher_digits("") is False

    # Test regex for non-string types (should return False via isinstance check)
    matcher_str_only = rex(".*")
    assert matcher_str_only(123) is False
    assert matcher_str_only(None) is False
    assert matcher_str_only(["a"]) is False

    # Test regex with complex pattern
    matcher_complex = rex(r"^[a-z]+_[0-9]{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("test_1") is False
    assert matcher_complex("TEST_01") is False
    assert matcher_complex("abc_99") is True
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_rex():
    # Test rex with a simple string match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should handle non-string types gracefully
    assert matcher_digit(None) is False

    # Test rex with a prefix match
    matcher_prefix = rex(r"^pre_")
    assert matcher_prefix("pre_test") is True
    assert matcher_prefix("test_pre") is False
    assert matcher_prefix("") is False

    # Test rex with complex regex
    matcher_complex = rex(r"^[a-z]+_[0-9]{2}$")
    assert matcher_complex("item_01") is True
    assert matcher_complex("item_1") is False
    assert matcher_complex("ITEM_01") is False
    assert matcher_complex("abc_def") is False

    # Test rex with empty string regex (matches everything)
    matcher_all = rex(r".*")
    assert matcher_all("anything") is True
    assert matcher_all("") is True
    assert matcher_all(123) is False

    # Test rex with non-matching pattern
    matcher_no_match = rex(r"xyz")
    assert matcher_no_match("abc") is False
    assert matcher_no_match("xyz") is True
```


# LLM-generated content at query #30
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

    # Test regex matching non-string types (should return False)
    matcher_any_str = rex(r".*")
    assert matcher_any_str("anything") is True
    assert matcher_any_str(123) is False
    assert matcher_any_str(None) is False
    assert matcher_any_str(["a"]) is False

    # Test exact match
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is True
    assert matcher_exact("not_exact") is False

    # Test case sensitivity
    matcher_case = rex(r"^[A-Z]+$")
    assert matcher_case("HELLO") is True
    assert matcher_case("hello") is False
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest

def test_rex():
    # Test with a simple string match
    matcher_digit = rex(r"^\d+$")
    assert matcher_digit("123") is True
    assert matcher_digit("abc") is False
    assert matcher_digit(123) is False  # Should return False for non-string types

    # Test with a complex regex
    matcher_email = rex(r"^[a-z]+@domain\.com$")
    assert matcher_email("user@domain.com") is True
    assert matcher_email("USER@domain.com") is False
    assert matcher_email("user@other.com") is False

    # Test with no match for empty string
    matcher_non_empty = rex(r".+")
    assert matcher_non_empty("a") is True
    assert matcher_non_empty("") is False

    # Test with non-string input (should be False per implementation)
    matcher_any = rex(r".*")
    assert matcher_any(None) is False
    assert matcher_any([1, 2, 3]) is False
    assert matcher_any(True) is False
```


# LLM-generated content at query #32
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
    assert matcher_digits("123a") is False
    
    # Test regex matcher with non-string keys (should return False per implementation)
    assert matcher_digits(123) is False
    assert matcher_digits(None) is False
    assert matcher_digits(["123"]) is False
    
    # Test regex matcher with complex pattern
    matcher_email = rex(r"[^@]+@[^@]+\.[^@]+")
    assert matcher_email("test@example.com") is True
    assert matcher_email("invalid-email") is False
    
    # Test regex matcher with exact string match
    matcher_exact = rex("^hello$")
    assert matcher_exact("hello") is True
    assert matcher_exact("hello world") is False

def test_rex_empty_pattern():
    # Test regex matcher with pattern that matches everything
    matcher_any = rex(".*")
    assert matcher_any("anything") is True
    assert matcher_any("") is True
    assert matcher_any(123) is False
```


# LLM-generated content at query #33
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
    assert matcher_digit(["123"]) is False
    
    # Test complex regex
    matcher_complex = rex(r"^[a-z]+_\d{2}$")
    assert matcher_complex("test_01") is True
    assert matcher_complex("test_1") is False
    assert matcher_complex("TEST_01") is False
    
    # Test empty string matching
    matcher_empty = rex(r"^$")
    assert matcher_empty("") is True
    assert matcher_empty(" ") is False
```


