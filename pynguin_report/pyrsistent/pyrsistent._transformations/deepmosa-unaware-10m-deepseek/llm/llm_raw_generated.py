####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher('test123extra') is None
    assert matcher(123) is False
    assert matcher(['test123']) is False

    matcher = rex(r'^[a-z]+$')
    assert matcher('abc') is not None
    assert matcher('ABC') is None
    assert matcher('abc123') is None
    assert matcher('') is not None

    matcher = rex(r'\d+')
    assert matcher('123') is not None
    assert matcher('abc123') is not None
    assert matcher('123abc') is not None
    assert matcher('abc') is None


# LLM-generated content at query #2
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching strings
    assert matcher('test_string') is not None
    assert matcher('testing') is not None
    assert matcher('no_match') is None
    
    # Test non-string inputs return None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'value'}) is None
    assert matcher(None) is None
    
    # Test with different patterns
    number_matcher = rex(r'^\d+$')
    assert number_matcher('123') is not None
    assert number_matcher('abc') is None
    
    # Test that pattern is compiled correctly
    email_matcher = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert email_matcher('test@example.com') is not None
    assert email_matcher('invalid_email') is None
    
    # Test empty string matching
    empty_matcher = rex(r'^$')
    assert empty_matcher('') is not None
    assert empty_matcher('a') is None


# LLM-generated content at query #3
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    number_matcher = rex(r'^\d+$')
    assert number_matcher('123') is not None
    assert number_matcher('abc') is None
    
    # Test pattern with special characters
    email_matcher = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert email_matcher('test@example.com') is not None
    assert email_matcher('invalid_email') is None
    
    # Test that returned function can be used as predicate
    keys = ['test1', 'test2', 'other', 'test3']
    filtered = [k for k in keys if rex(r'^test')(k)]
    assert filtered == ['test1', 'test2', 'test3']


# LLM-generated content at query #4
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    assert matcher({'key': 'test_123'}) is None
    assert matcher(None) is None
    assert matcher('test_abc') is None
    matcher2 = rex(r'^[A-Z]+$')
    assert matcher2('HELLO') is not None
    assert matcher2('Hello') is None
    assert matcher2('123') is None
    assert matcher2('') is None
    matcher3 = rex(r'.*')
    assert matcher3('any_string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None


# LLM-generated content at query #5
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher(123) is None
    assert matcher('other_123') is None
    
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('hello') is not None
    assert matcher2('Hello') is None
    assert matcher2('123') is None
    assert matcher2('') is None
    
    matcher3 = rex(r'.*')
    assert matcher3('any_string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None


# LLM-generated content at query #6
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None

    matcher = rex(r"a.b")
    assert matcher("axb") is not None
    assert matcher("a.b") is not None
    assert matcher("ab") is None

    matcher = rex(r"\d{3}-\d{2}")
    assert matcher("123-45") is not None
    assert matcher("12-345") is None
    assert matcher("abc-de") is None

    matcher = rex(r"")
    assert matcher("") is not None
    assert matcher("any") is not None


# LLM-generated content at query #7
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    
    # Test with non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    matcher2 = rex(r'[a-z]+')
    assert matcher2('hello') is not None
    assert matcher2('HELLO') is None
    assert matcher2('123') is None
    
    # Test with empty pattern
    matcher3 = rex(r'')
    assert matcher3('') is not None
    assert matcher3('anything') is not None
    
    # Test that returned function is callable
    assert callable(rex(r'.*'))


# LLM-generated content at query #8
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    
    matcher2 = rex(r'[a-z]+')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('') is None
    
    matcher3 = rex(r'.*')
    assert matcher3('any_string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None


# LLM-generated content at query #9
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('no_match') is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    digit_matcher = rex(r'^\d+$')
    assert digit_matcher('123') is not None
    assert digit_matcher('abc') is None
    
    # Test case sensitivity
    case_matcher = rex(r'^[A-Z]+$')
    assert case_matcher('UPPER') is not None
    assert case_matcher('lower') is None
    
    # Test empty string
    empty_matcher = rex(r'^$')
    assert empty_matcher('') is not None
    assert empty_matcher('a') is None
    
    # Test that pattern is compiled
    pattern = rex(r'^a+$')
    # Should match one or more 'a's
    assert pattern('a') is not None
    assert pattern('aaa') is not None
    assert pattern('aaab') is None


# LLM-generated content at query #10
#--------------------------

```python
def test_rex():
    # Test basic string matching
    matcher = rex(r'^a')
    assert matcher('apple') is not None
    assert matcher('banana') is None
    
    # Test with compiled pattern behavior
    matcher = rex(r'\d+')
    assert matcher('123') is not None
    assert matcher('abc') is None
    assert matcher('123abc') is not None
    
    # Test that non-strings return None
    matcher = rex(r'.*')
    assert matcher(123) is None
    assert matcher(['a', 'b']) is None
    assert matcher({'key': 'value'}) is None
    assert matcher(None) is None
    
    # Test full string matching
    matcher = rex(r'^test$')
    assert matcher('test') is not None
    assert matcher('testing') is None
    
    # Test case sensitivity
    matcher = rex(r'^Hello')
    assert matcher('Hello World') is not None
    assert matcher('hello world') is None
    
    # Test complex pattern
    matcher = rex(r'^[A-Z][a-z]+\d{2,3}$')
    assert matcher('Test123') is not None
    assert matcher('Test12') is not None
    assert matcher('test123') is None
    assert matcher('Test1234') is None
    assert matcher('T123') is None


# LLM-generated content at query #11
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    assert matcher({}) is None
    
    # Test with different patterns
    matcher_digit = rex(r'^\d+$')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    matcher_ends_with = rex(r'.*ing$')
    assert matcher_ends_with('testing') is not None
    assert matcher_ends_with('test') is None
    
    # Test that compiled regex works correctly
    matcher_email = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert matcher_email('test@example.com') is not None
    assert matcher_email('invalid_email') is None


# LLM-generated content at query #12
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    assert matcher({'key': 'test_123'}) is None

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None
    assert matcher2('') is None

    matcher3 = rex(r'.*')
    assert matcher3('any_string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None

    matcher4 = rex(r'\d{3}-\d{2}-\d{4}')
    assert matcher4('123-45-6789') is not None
    assert matcher4('12-345-6789') is None
    assert matcher4('123-45-678') is None


# LLM-generated content at query #13
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test0') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher(123) is None
    assert matcher(['test123']) is None
    
    # Test with special regex characters
    matcher = rex(r'^a.b$')
    assert matcher('a.b') is not None
    assert matcher('axb') is not None
    assert matcher('a.b.c') is None
    
    # Test with case insensitive matching
    matcher = rex(r'^hello$')
    assert matcher('hello') is not None
    assert matcher('Hello') is None
    
    # Test with complex pattern
    matcher = rex(r'^\w+@\w+\.\w+$')
    assert matcher('test@example.com') is not None
    assert matcher('invalid_email') is None
    assert matcher('@example.com') is None
    
    # Test that returned function is callable
    matcher = rex(r'.*')
    assert callable(matcher)
    
    # Test empty pattern
    matcher = rex(r'')
    assert matcher('') is not None
    assert matcher('any') is not None


# LLM-generated content at query #14
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher(123) is None
    assert matcher('test123extra') is None

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3(123) is None


# LLM-generated content at query #15
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None

    matcher = rex(r'^[a-z]+$')
    assert matcher('abc') is not None
    assert matcher('ABC') is None
    assert matcher('abc123') is None
    assert matcher('') is not None

    matcher = rex(r'\d{3}-\d{2}-\d{4}')
    assert matcher('123-45-6789') is not None
    assert matcher('12-45-6789') is None
    assert matcher('123-456-789') is None
    assert matcher('abc-def-ghij') is None


# LLM-generated content at query #16
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher(123) is None
    assert matcher('test123extra') is None

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None
    assert matcher2('') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3(123) is None

    matcher4 = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher4('123-45-6789') is not None
    assert matcher4('12-345-6789') is None
    assert matcher4('123-45-678') is None


# LLM-generated content at query #17
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher(123) is None
    assert matcher('test123extra') is None
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('abc123') is None
    assert matcher2('ABC') is None
    assert matcher2('') is None
    assert matcher2(123) is None


# LLM-generated content at query #18
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher(123) is None
    assert matcher('test123extra') is None

    matcher = rex(r'^a.*b$')
    assert matcher('ab') is not None
    assert matcher('axxxb') is not None
    assert matcher('ba') is None
    assert matcher('a') is None
    assert matcher('b') is None

    matcher = rex(r'^\d+$')
    assert matcher('123') is not None
    assert matcher('') is None
    assert matcher('123a') is None
    assert matcher('a123') is None
    assert matcher(123) is None


# LLM-generated content at query #19
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher('test123extra') is None
    assert matcher(123) is False
    assert matcher(['test123']) is False
    assert matcher({'key': 'test123'}) is False

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None
    assert matcher2('') is not None

    matcher3 = rex(r'\d+')
    assert matcher3('123') is not None
    assert matcher3('abc123def') is not None
    assert matcher3('abc') is None


# LLM-generated content at query #20
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None
    assert matcher2('') is None
    
    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3(123) is None


# LLM-generated content at query #21
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    
    matcher2 = rex(r'[a-z]+')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is not None
    
    matcher3 = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher3('123-45-6789') is not None
    assert matcher3('12-345-6789') is None
    assert matcher3('123-45-67890') is None


# LLM-generated content at query #22
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    digit_matcher = rex(r'^\d+$')
    assert digit_matcher('123') is not None
    assert digit_matcher('abc') is None
    
    # Test with complex pattern
    email_matcher = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert email_matcher('test@example.com') is not None
    assert email_matcher('invalid-email') is None
    
    # Test that returned function can be used as predicate
    keys = ['a1', 'b2', 'c', 'd3', 'e']
    pattern = rex(r'^\w\d$')
    filtered = [k for k in keys if pattern(k)]
    assert filtered == ['a1', 'b2', 'd3']


# LLM-generated content at query #23
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    
    # Test with non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    assert matcher({}) is None
    
    # Test with different patterns
    matcher2 = rex(r'[a-z]+')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('123') is None
    
    # Test with empty pattern
    matcher3 = rex(r'')
    assert matcher3('') is not None
    assert matcher3('anything') is not None
    
    # Test that it returns a callable
    assert callable(rex(r'.*'))
    
    # Test that matcher returns True/False-like values
    result = rex(r'^match$')('match')
    assert bool(result) is True
    result = rex(r'^match$')('nomatch')
    assert bool(result) is False


# LLM-generated content at query #24
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test0') is not None
    assert matcher('test') is None  # No digits
    assert matcher('123test') is None  # Wrong pattern
    assert matcher(123) is None  # Not a string
    
    # Test with non-matching string
    matcher2 = rex(r'^[A-Z]+$')
    assert matcher2('HELLO') is not None
    assert matcher2('Hello') is None  # Lowercase letter
    assert matcher2('123') is None
    assert matcher2('') is None
    
    # Test with complex pattern
    matcher3 = rex(r'^[a-z]+_[a-z]+$')
    assert matcher3('hello_world') is not None
    assert matcher3('test_case') is not None
    assert matcher3('Hello_World') is None  # Uppercase
    assert matcher3('hello-world') is None  # Wrong separator
    assert matcher3('hello') is None  # Missing underscore
    
    # Test that it returns a callable
    assert callable(rex(r'.*'))
    
    # Test empty pattern
    matcher4 = rex(r'')
    assert matcher4('') is not None
    assert matcher4('any') is not None


# LLM-generated content at query #25
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    assert matcher({"key": "test_123"}) is None

    matcher2 = rex(r"a.b")
    assert matcher2("acb") is not None
    assert matcher2("a b") is not None
    assert matcher2("ab") is None

    matcher3 = rex(r"\d{3}-\d{2}-\d{4}")
    assert matcher3("123-45-6789") is not None
    assert matcher3("12-345-6789") is None
    assert matcher3("abc-def-ghij") is None

    matcher4 = rex(r"")
    assert matcher4("") is not None
    assert matcher4("any string") is not None


# LLM-generated content at query #26
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    
    matcher2 = rex(r'[a-z]+')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('') is None
    
    matcher3 = rex(r'.*')
    assert matcher3('any_string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None


# LLM-generated content at query #27
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    matcher_digits = rex(r'^\d+$')
    assert matcher_digits('123') is not None
    assert matcher_digits('abc') is None
    
    # Test pattern with special characters
    matcher_email = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert matcher_email('test@example.com') is not None
    assert matcher_email('invalid_email') is None
    
    # Test that returned matcher works with re.match object methods
    matcher = rex(r'^a(b)c$')
    match = matcher('abc')
    assert match is not None
    assert match.group(0) == 'abc'
    assert match.group(1) == 'b'


# LLM-generated content at query #28
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    assert matcher({'key': 'test_123'}) is None
    assert matcher(None) is None

    matcher2 = rex(r'[a-z]+')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('123') is None
    assert matcher2('') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('anything') is not None
    assert matcher3('123!@#') is not None
    assert matcher3(123) is None

    matcher4 = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher4('123-45-6789') is not None
    assert matcher4('12-345-6789') is None
    assert matcher4('123-45-67890') is None
    assert matcher4('abc-de-fghi') is None


# LLM-generated content at query #29
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('no_match') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'value'}) is None
    
    # Test with different patterns
    matcher_digits = rex(r'^\d+$')
    assert matcher_digits('123') is not None
    assert matcher_digits('abc') is None
    
    matcher_email = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert matcher_email('test@example.com') is not None
    assert matcher_email('invalid_email') is None
    
    # Test that pattern is compiled correctly
    pattern = rex(r'^a.*b$')
    assert pattern('axxxb') is not None
    assert pattern('ab') is not None
    assert pattern('ba') is None


# LLM-generated content at query #30
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'test'}) is None
    
    # Test with different patterns
    matcher_digit = rex(r'^\d+$')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    matcher_word = rex(r'^\w+$')
    assert matcher_word('hello') is not None
    assert matcher_word('hello world') is None
    
    # Test that pattern is compiled correctly
    matcher_email = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert matcher_email('test@example.com') is not None
    assert matcher_email('invalid-email') is None


# LLM-generated content at query #31
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\w+$")
    assert matcher("test_function") == True
    assert matcher("test_method") == True
    assert matcher("test") == True
    assert matcher("_test") == False
    assert matcher("test_") == False
    assert matcher("not_test") == False
    assert matcher("test-with-dash") == False
    assert matcher(123) == False
    assert matcher(None) == False
    assert matcher(["test"]) == False

    matcher2 = rex(r"\d+")
    assert matcher2("123") == True
    assert matcher2("abc") == False
    assert matcher2("123abc") == True
    assert matcher2("") == False

    matcher3 = rex(r"^[A-Z][a-z]*$")
    assert matcher3("Hello") == True
    assert matcher3("hello") == False
    assert matcher3("H") == True
    assert matcher3("") == False

    matcher4 = rex(r".*")
    assert matcher4("") == True
    assert matcher4("anything") == True
    assert matcher4("123!@#") == True


# LLM-generated content at query #32
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    assert matcher({"key": "test_123"}) is None
    assert matcher(None) is None

    matcher = rex(r"[a-z]+")
    assert matcher("abc") is not None
    assert matcher("ABC") is None
    assert matcher("123") is None
    assert matcher("") is None

    matcher = rex(r".*")
    assert matcher("") is not None
    assert matcher("any string") is not None
    assert matcher("123") is not None
    assert matcher(123) is None


# LLM-generated content at query #33
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher(123) is False
    assert matcher('test123extra') is None
    
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('hello') is not None
    assert matcher2('Hello') is None
    assert matcher2('hello123') is None
    
    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3(123) is False


# LLM-generated content at query #34
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    matcher_digit = rex(r'^\d+$')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    matcher_email = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert matcher_email('test@example.com') is not None
    assert matcher_email('invalid-email') is None
    
    # Test that pattern is compiled only once
    pattern = r'^foo'
    matcher1 = rex(pattern)
    matcher2 = rex(pattern)
    # Both should work the same
    assert matcher1('foo_bar') is not None
    assert matcher2('foo_bar') is not None
    assert matcher1('bar_foo') is None
    assert matcher2('bar_foo') is None


# LLM-generated content at query #35
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher(123) is None
    assert matcher('other_123') is None

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('hello') is not None
    assert matcher2('Hello') is None
    assert matcher2('hello123') is None
    assert matcher2('') is None

    matcher3 = rex(r'.*')
    assert matcher3('any string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None


# LLM-generated content at query #36
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    assert matcher({'key': 'test_123'}) is None

    matcher = rex(r'^[a-z]+$')
    assert matcher('abc') is not None
    assert matcher('ABC') is None
    assert matcher('abc123') is None
    assert matcher('') is None
    assert matcher('a') is not None

    matcher = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher('123-45-6789') is not None
    assert matcher('12-345-6789') is None
    assert matcher('123-456-789') is None
    assert matcher('123-45-67890') is None
    assert matcher('abc-de-fghi') is None


# LLM-generated content at query #37
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    
    # Test with non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher(["test_123"]) is None
    
    # Test with different patterns
    matcher2 = rex(r"[a-z]+")
    assert matcher2("hello") is not None
    assert matcher2("HELLO") is None
    assert matcher2("123") is None
    
    # Test with empty pattern
    matcher3 = rex(r"")
    assert matcher3("") is not None
    assert matcher3("anything") is not None
    
    # Test that returned function is callable
    assert callable(rex(r".*"))


# LLM-generated content at query #38
#--------------------------

```python
def test_rex():
    # Test basic string matching
    matcher = rex(r'^test')
    assert matcher('test_string') is not None
    assert matcher('string_test') is None
    
    # Test with non-string input
    assert matcher(123) is False
    assert matcher(None) is False
    assert matcher([]) is False
    
    # Test case sensitivity
    matcher = rex(r'^test')
    assert matcher('test') is not None
    assert matcher('Test') is None
    
    # Test full match
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_abc') is None
    assert matcher('prefix_test_123') is None
    
    # Test with special regex characters
    matcher = rex(r'^a\.b$')
    assert matcher('a.b') is not None
    assert matcher('a_b') is None
    
    # Test empty string
    matcher = rex(r'^$')
    assert matcher('') is not None
    assert matcher('a') is None
    
    # Test compiled regex object is created correctly
    matcher = rex(r'pattern')
    result = matcher('pattern')
    assert result is not None
    assert hasattr(result, 'group')


# LLM-generated content at query #39
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching strings with pattern
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    assert matcher('testing') is not None
    
    # Test non-matching strings
    assert matcher('no_match') is None
    assert matcher('TEST') is None  # case sensitive
    
    # Test with non-string inputs (should return None)
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'test'}) is None
    
    # Test with different patterns
    matcher_digits = rex(r'^\d+$')
    assert matcher_digits('123') is not None
    assert matcher_digits('abc') is None
    
    matcher_email = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert matcher_email('test@example.com') is not None
    assert matcher_email('invalid_email') is None
    
    # Test that returned function works as a predicate
    keys = ['test1', 'test2', 'other', 'test3', 123]
    filtered = [k for k in keys if rex(r'^test')(k)]
    assert filtered == ['test1', 'test2', 'test3']


# LLM-generated content at query #40
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher('test123extra') is None
    assert matcher(123) is False
    assert matcher(['test123']) is False
    assert matcher(None) is False

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None
    assert matcher2('') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3(123) is False


# LLM-generated content at query #41
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None
    assert matcher2('') is None


# LLM-generated content at query #42
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('no_match') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    assert matcher({}) is None
    
    # Test with different patterns
    digit_matcher = rex(r'^\d+$')
    assert digit_matcher('123') is not None
    assert digit_matcher('abc') is None
    
    # Test with complex pattern
    email_matcher = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert email_matcher('test@example.com') is not None
    assert email_matcher('invalid_email') is None
    
    # Test that returned matcher can be used multiple times
    matcher = rex(r'^a')
    assert matcher('apple') is not None
    assert matcher('banana') is None
    assert matcher('apricot') is not None


# LLM-generated content at query #43
#--------------------------

```python
def test_rex():
    matcher = rex(r'^a\d+$')
    assert matcher('a123') is not None
    assert matcher('a') is None
    assert matcher('123') is None
    assert matcher('a123b') is None
    assert matcher(123) is False
    assert matcher(['a123']) is False

    matcher2 = rex(r'^test.*')
    assert matcher2('testing') is not None
    assert matcher2('test') is not None
    assert matcher2('atest') is None
    assert matcher2(True) is False

    matcher3 = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher3('123-45-6789') is not None
    assert matcher3('12-345-6789') is None
    assert matcher3('123-45-67890') is None
    assert matcher3(123456789) is False


# LLM-generated content at query #44
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    assert matcher({"key": "test_123"}) is None
    assert matcher(None) is None

    matcher2 = rex(r"[a-z]+")
    assert matcher2("abc") is not None
    assert matcher2("ABC") is None
    assert matcher2("abc123") is not None
    assert matcher2("123") is None

    matcher3 = rex(r"\d{3}-\d{2}-\d{4}")
    assert matcher3("123-45-6789") is not None
    assert matcher3("12-345-6789") is None
    assert matcher3("123-45-67890") is not None
    assert matcher3("123-45-678") is None

    matcher4 = rex(r"")
    assert matcher4("") is not None
    assert matcher4("any") is not None


# LLM-generated content at query #45
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None

    matcher = rex(r'^[a-z]+$')
    assert matcher('hello') is not None
    assert matcher('Hello') is None
    assert matcher('hello123') is None
    assert matcher('') is not None

    matcher = rex(r'\d{3}-\d{2}-\d{4}')
    assert matcher('123-45-6789') is not None
    assert matcher('12-345-6789') is None
    assert matcher('abc-def-ghij') is None

    matcher = rex(r'.*')
    assert matcher('any_string') is not None
    assert matcher('') is not None
    assert matcher('with spaces') is not None


# LLM-generated content at query #46
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher(['test']) is None
    
    # Test with different patterns
    matcher_digit = rex(r'^\d+$')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    # Test with complex pattern
    matcher_email = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert matcher_email('test@example.com') is not None
    assert matcher_email('invalid_email') is None
    
    # Test that pattern is compiled only once
    pattern = r'^foo'
    matcher1 = rex(pattern)
    matcher2 = rex(pattern)
    # Both should work the same
    assert matcher1('foo_bar') is not None
    assert matcher2('foo_bar') is not None
    assert matcher1('bar_foo') is None
    assert matcher2('bar_foo') is None


# LLM-generated content at query #47
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_") is None
    assert matcher("test_abc") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    assert rex(r"^a.*z$")("abcz") is not None
    assert rex(r"^a.*z$")("abczy") is None
    matcher2 = rex(r"\d+")
    assert matcher2("123") is not None
    assert matcher2("abc") is None


# LLM-generated content at query #48
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    assert matcher(None) is None

    matcher = rex(r"a.b")
    assert matcher("acb") is not None
    assert matcher("a b") is not None
    assert matcher("ab") is None

    matcher = rex(r"\d{3}-\d{2}")
    assert matcher("123-45") is not None
    assert matcher("12-345") is None
    assert matcher("abc-de") is None

    matcher = rex(r"")
    assert matcher("") is not None
    assert matcher("any") is not None

    matcher = rex(r"^[A-Z][a-z]*$")
    assert matcher("Hello") is not None
    assert matcher("hello") is None
    assert matcher("HELLO") is None


# LLM-generated content at query #49
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('no_match') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'test'}) is None
    
    # Test with different patterns
    matcher_digit = rex(r'\d+')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    matcher_exact = rex(r'^exact$')
    assert matcher_exact('exact') is not None
    assert matcher_exact('exact_extra') is None
    
    # Test that returned function uses compiled regex
    pattern = r'^a.*b$'
    matcher = rex(pattern)
    assert matcher('axxxb') is not None
    assert matcher('axxxc') is None


# LLM-generated content at query #50
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher(['test']) is None
    
    # Test with different patterns
    digit_matcher = rex(r'\d+')
    assert digit_matcher('123') is not None
    assert digit_matcher('abc') is None
    
    # Test that pattern is compiled correctly
    pattern = rex(r'^[a-z]+$')
    assert pattern('lowercase') is not None
    assert pattern('MixedCase') is None
    assert pattern('123') is None


# LLM-generated content at query #51
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    assert matcher({"key": "test_123"}) is None
    assert matcher(None) is None

    matcher2 = rex(r"a.b")
    assert matcher2("acb") is not None
    assert matcher2("a b") is not None
    assert matcher2("ab") is None

    matcher3 = rex(r"\d{3}-\d{2}-\d{4}")
    assert matcher3("123-45-6789") is not None
    assert matcher3("12-345-6789") is None
    assert matcher3("123-456-789") is None

    matcher4 = rex(r"")
    assert matcher4("") is not None
    assert matcher4("any string") is not None

    matcher5 = rex(r"^[A-Z][a-z]*$")
    assert matcher5("Hello") is not None
    assert matcher5("hello") is None
    assert matcher5("HELLO") is None
    assert matcher5("Hello123") is None


# LLM-generated content at query #52
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_") is None
    assert matcher("test_abc") is None
    assert matcher(123) is None
    assert matcher("other") is None
    
    matcher2 = rex(r"a.b")
    assert matcher2("axb") is not None
    assert matcher2("a b") is not None
    assert matcher2("ab") is None
    
    matcher3 = rex(r"\d+")
    assert matcher3("123") is not None
    assert matcher3("abc") is None
    assert matcher3("") is None


# LLM-generated content at query #53
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    assert matcher({'key': 'test_123'}) is None

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('lowercase') is not None
    assert matcher2('MixedCase') is None
    assert matcher2('UPPERCASE') is None
    assert matcher2('with123') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3('another') is not None

    matcher4 = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher4('123-45-6789') is not None
    assert matcher4('12-345-6789') is None
    assert matcher4('123-456-789') is None


# LLM-generated content at query #54
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    matcher_digit = rex(r'^\d+$')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    matcher_word = rex(r'^\w+$')
    assert matcher_word('hello') is not None
    assert matcher_word('hello world') is None
    
    # Test that pattern is compiled
    matcher = rex(r'^a.*z$')
    assert matcher('abcz') is not None
    assert matcher('axyz') is not None
    assert matcher('abc') is None


# LLM-generated content at query #55
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r"^test")
    assert callable(matcher)
    
    # Test matching string
    assert matcher("test_string") is not None
    assert matcher("test") is not None
    
    # Test non-matching string
    assert matcher("other_string") is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    matcher2 = rex(r"\d+")
    assert matcher2("123") is not None
    assert matcher2("abc") is None
    
    # Test that returned function works with re.match object
    result = matcher("test_value")
    assert hasattr(result, "group")  # Should be a match object
    assert result.group() == "test"
    
    # Test empty pattern
    matcher3 = rex(r"")
    assert matcher3("") is not None
    assert matcher3("anything") is not None


# LLM-generated content at query #56
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_") is None
    assert matcher("test_abc") is None
    assert matcher(123) is None
    assert matcher("other") is None

    matcher = rex(r"a.*b")
    assert matcher("ab") is not None
    assert matcher("axxxb") is not None
    assert matcher("ba") is None
    assert matcher("") is None

    matcher = rex(r"\d+")
    assert matcher("123") is not None
    assert matcher("abc") is None
    assert matcher("123abc") is not None


# LLM-generated content at query #57
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    assert matcher({'key': 'test_123'}) is None
    assert matcher(None) is None
    assert matcher('test_abc') is None
    matcher2 = rex(r'^[A-Z]+$')
    assert matcher2('HELLO') is not None
    assert matcher2('Hello') is None
    assert matcher2('123') is None


# LLM-generated content at query #58
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    assert matcher({"key": "test_123"}) is None
    assert matcher(None) is None
    assert matcher("") is None
    matcher2 = rex(r"^[a-z]+$")
    assert matcher2("abc") is not None
    assert matcher2("ABC") is None
    assert matcher2("abc123") is None


# LLM-generated content at query #59
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching strings
    assert matcher('test_string') is not None
    assert matcher('testing') is not None
    assert matcher('no_match') is None
    
    # Test non-string inputs return None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'test'}) is None
    assert matcher(None) is None
    
    # Test with different patterns
    digit_matcher = rex(r'^\d+$')
    assert digit_matcher('123') is not None
    assert digit_matcher('abc') is None
    
    # Test with complex patterns
    email_matcher = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert email_matcher('test@example.com') is not None
    assert email_matcher('invalid_email') is None
    
    # Test that rex compiles pattern only once
    pattern = r'^foo'
    matcher1 = rex(pattern)
    matcher2 = rex(pattern)
    assert matcher1('foo_bar') is not None
    assert matcher2('foo_bar') is not None


# LLM-generated content at query #60
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\w+$")
    assert matcher("test_function") is not None
    assert matcher("test_function").group() == "test_function"
    assert matcher("test_123") is not None
    assert matcher("invalid") is None
    assert matcher("test-with-dash") is None
    assert matcher(123) is None
    assert matcher(["test_list"]) is None
    assert matcher(None) is None
    assert matcher("") is None
    assert matcher("test_") is not None
    assert matcher("test_").group() == "test_"
    matcher2 = rex(r"\d+")
    assert matcher2("123") is not None
    assert matcher2("123").group() == "123"
    assert matcher2("abc") is None
    assert matcher2("123abc") is not None
    assert matcher2("123abc").group() == "123"


# LLM-generated content at query #61
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher(123) is None
    assert matcher('other_123') is None
    
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('hello') is not None
    assert matcher2('Hello') is None
    assert matcher2('hello123') is None
    assert matcher2('') is None
    
    matcher3 = rex(r'.*')
    assert matcher3('any string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None


# LLM-generated content at query #62
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None
    assert matcher2('') is None
    
    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3(123) is None


# LLM-generated content at query #63
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_") is None
    assert matcher("test_abc") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    
    matcher2 = rex(r"a.b")
    assert matcher2("axb") is not None
    assert matcher2("a.b") is not None
    assert matcher2("ab") is None
    
    matcher3 = rex(r"\d+")
    assert matcher3("123") is not None
    assert matcher3("abc") is None
    assert matcher3("") is None


# LLM-generated content at query #64
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher('test_abc') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    assert matcher(None) is None

    matcher2 = rex(r'[a-z]+')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('123') is None
    assert matcher2('') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('anything') is not None
    assert matcher3('123!@#') is not None
    assert matcher3(123) is None

    matcher4 = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher4('123-45-6789') is not None
    assert matcher4('12-345-6789') is None
    assert matcher4('123-456-789') is None
    assert matcher4('abc-de-fghi') is None


# LLM-generated content at query #65
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    
    matcher2 = rex(r'[a-z]+')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('') is None
    assert matcher2(123) is None
    
    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('anything') is not None
    assert matcher3(123) is None


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'test'}) is None
    
    # Test with different patterns
    matcher_digit = rex(r'\d+')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    # Test pattern with special characters
    matcher_email = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert matcher_email('test@example.com') is not None
    assert matcher_email('invalid_email') is None
    
    # Test that returned matcher can be used as predicate
    assert rex(r'^a')('apple') is not None
    assert rex(r'^a')('banana') is None


# LLM-generated content at query #2
#--------------------------

```python
def test_rex():
    # Test with matching pattern
    matcher = rex(r'^a\d+$')
    assert matcher('a123') is not None
    assert matcher('a1') is not None
    assert matcher('a') is None
    assert matcher('123') is None
    assert matcher('a123b') is None
    
    # Test with non-string input
    assert matcher(123) is False
    assert matcher(None) is False
    assert matcher([]) is False
    
    # Test with different patterns
    matcher2 = rex(r'^[A-Z]+$')
    assert matcher2('ABC') is not None
    assert matcher2('AbC') is None
    assert matcher2('abc') is None
    
    # Test with empty pattern
    matcher3 = rex(r'')
    assert matcher3('') is not None
    assert matcher3('anything') is not None
    
    # Test that returned function is callable
    assert callable(rex(r'.*'))


# LLM-generated content at query #3
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'value'}) is None
    
    # Test with different patterns
    matcher_digit = rex(r'\d+')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    # Test with complex pattern
    matcher_email = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert matcher_email('test@example.com') is not None
    assert matcher_email('invalid-email') is None


# LLM-generated content at query #4
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_") is None
    assert matcher("test_abc") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    assert rex(r"^a.*z$")("abcz") is not None
    assert rex(r"^a.*z$")("abcy") is None
    matcher2 = rex(r"\d+")
    assert matcher2("123") is not None
    assert matcher2("abc") is None


# LLM-generated content at query #5
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('no_match') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    digit_matcher = rex(r'^\d+$')
    assert digit_matcher('123') is not None
    assert digit_matcher('abc') is None
    
    # Test with complex pattern
    email_matcher = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert email_matcher('test@example.com') is not None
    assert email_matcher('invalid_email') is None
    
    # Test that pattern is compiled only once
    pattern = r'^foo'
    matcher1 = rex(pattern)
    matcher2 = rex(pattern)
    # Both should work the same way
    assert matcher1('foo_bar') is not None
    assert matcher2('foo_bar') is not None
    assert matcher1('bar_foo') is None
    assert matcher2('bar_foo') is None


# LLM-generated content at query #6
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher('test_abc') is None
    
    # Test with non-string input
    assert matcher(123) is False
    assert matcher(None) is False
    assert matcher([]) is False
    assert matcher({}) is False
    
    # Test with different patterns
    matcher2 = rex(r'[a-z]+')
    assert matcher2('hello') is not None
    assert matcher2('HELLO') is None
    assert matcher2('123') is None
    
    matcher3 = rex(r'\d{3}-\d{2}-\d{4}')
    assert matcher3('123-45-6789') is not None
    assert matcher3('12-345-6789') is None
    assert matcher3('abc-de-fghi') is None
    
    # Test that returned function is callable
    assert callable(rex(r'.*'))
    
    # Test with empty pattern
    matcher4 = rex(r'')
    assert matcher4('') is not None
    assert matcher4('anything') is not None
    
    # Test with special regex characters
    matcher5 = rex(r'\.\*\?\+')
    assert matcher5('.*?+') is not None
    assert matcher5('test') is None


# LLM-generated content at query #7
#--------------------------

```python
def test_rex():
    matcher = rex(r'^a\d+$')
    assert matcher('a123') is not None
    assert matcher('a') is None
    assert matcher('123') is None
    assert matcher('a12b') is None
    assert matcher(123) is False
    assert matcher(['a123']) is False

    matcher2 = rex(r'^test.*')
    assert matcher2('test_string') is not None
    assert matcher2('not_test') is None

    matcher3 = rex(r'.*')
    assert matcher3('any_string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is False


# LLM-generated content at query #8
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test0') is not None
    assert matcher('test') is None  # no digits
    assert matcher('123test') is None  # wrong pattern
    assert matcher(123) is None  # not a string
    
    # Test with non-matching string
    matcher = rex(r'^[A-Z]+$')
    assert matcher('HELLO') is not None
    assert matcher('Hello') is None  # lowercase letter
    assert matcher('123') is None  # digits
    assert matcher('') is None  # empty string
    
    # Test with complex pattern
    matcher = rex(r'^[a-z]+@[a-z]+\.[a-z]+$')
    assert matcher('test@example.com') is not None
    assert matcher('user@domain.org') is not None
    assert matcher('Test@example.com') is None  # uppercase
    assert matcher('test@example') is None  # no TLD
    assert matcher('@example.com') is None  # no username
    
    # Test that it returns a callable
    matcher = rex(r'.*')
    assert callable(matcher)
    assert matcher('any string') is not None
    assert matcher('') is not None  # empty string matches .*
    
    # Test with special characters
    matcher = rex(r'^\d+\.\d+$')
    assert matcher('3.14') is not None
    assert matcher('0.5') is not None
    assert matcher('3') is None  # no decimal point
    assert matcher('.5') is None  # no leading digit
    assert matcher('3.') is None  # no trailing digit


# LLM-generated content at query #9
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test0') is not None
    assert matcher('test') is None  # No digits
    assert matcher('123test') is None  # Wrong pattern
    assert matcher(123) is None  # Not a string
    
    # Test with special characters
    matcher = rex(r'^a\.b$')
    assert matcher('a.b') is not None
    assert matcher('ab') is None
    
    # Test with case sensitivity
    matcher = rex(r'^[A-Z]+$')
    assert matcher('ABC') is not None
    assert matcher('abc') is None
    
    # Test with complex pattern
    matcher = rex(r'^\w+@\w+\.\w+$')
    assert matcher('test@example.com') is not None
    assert matcher('invalid-email') is None
    
    # Test that returned function is callable
    matcher = rex(r'.*')
    assert callable(matcher)
    assert matcher('any string') is not None


# LLM-generated content at query #10
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    assert matcher({'key': 'test_123'}) is None
    
    matcher2 = rex(r'[a-z]+')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('123') is None
    assert matcher2('abc123') is not None
    
    matcher3 = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher3('123-45-6789') is not None
    assert matcher3('12-45-6789') is None
    assert matcher3('123-456-789') is None
    assert matcher3('abc-de-fghi') is None


# LLM-generated content at query #11
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    assert matcher({"key": "test_123"}) is None

    matcher2 = rex(r"a.b")
    assert matcher2("acb") is not None
    assert matcher2("a b") is not None
    assert matcher2("ab") is None

    matcher3 = rex(r"\d{3}-\d{2}")
    assert matcher3("123-45") is not None
    assert matcher3("12-345") is None
    assert matcher3("abc-de") is None


# LLM-generated content at query #12
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    assert matcher({"key": "test_123"}) is None

    matcher2 = rex(r"a.b")
    assert matcher2("acb") is not None
    assert matcher2("ab") is None
    assert matcher2("a\nb") is None

    matcher3 = rex(r"\d{3}-\d{2}-\d{4}")
    assert matcher3("123-45-6789") is not None
    assert matcher3("12-345-6789") is None
    assert matcher3("123-456-789") is None

    matcher4 = rex(r"")
    assert matcher4("") is not None
    assert matcher4("any string") is not None


# LLM-generated content at query #13
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test0') is not None
    assert matcher('test') is None  # no digits
    assert matcher('123test') is None  # wrong pattern
    assert matcher(123) is None  # not a string
    
    # Test with non-matching string
    matcher2 = rex(r'^[A-Z]+$')
    assert matcher2('HELLO') is not None
    assert matcher2('Hello') is None  # lowercase letter
    assert matcher2('123') is None  # digits
    assert matcher2('') is None  # empty string
    
    # Test with complex pattern
    matcher3 = rex(r'^[a-z]{3}-\d{2}$')
    assert matcher3('abc-12') is not None
    assert matcher3('xyz-99') is not None
    assert matcher3('ABC-12') is None  # uppercase
    assert matcher3('abcd-12') is None  # too many letters
    assert matcher3('abc-123') is None  # too many digits
    
    # Test that returns callable
    assert callable(rex(r'.*'))
    
    # Test with special regex characters
    matcher4 = rex(r'^a\.b$')
    assert matcher4('a.b') is not None
    assert matcher4('aXb') is None


# LLM-generated content at query #14
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    assert matcher(None) is None
    
    matcher2 = rex(r"^[a-z]+$")
    assert matcher2("hello") is not None
    assert matcher2("Hello") is None
    assert matcher2("hello123") is None
    
    matcher3 = rex(r".*")
    assert matcher3("") is not None
    assert matcher3("any string") is not None
    assert matcher3(123) is None


# LLM-generated content at query #15
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    
    # Test with non-string input
    assert matcher(123) is False
    assert matcher(None) is False
    assert matcher(["test_123"]) is False
    
    # Test with different patterns
    matcher2 = rex(r"[a-z]+")
    assert matcher2("hello") is not None
    assert matcher2("HELLO") is None
    assert matcher2("123") is None
    
    # Test with empty string
    matcher3 = rex(r"")
    assert matcher3("") is not None
    assert matcher3("any") is not None
    
    # Test that returned function is callable
    assert callable(rex(r".*"))
    
    # Test with special regex characters
    matcher4 = rex(r"\d+\.\d+")
    assert matcher4("3.14") is not None
    assert matcher4("abc") is None


# LLM-generated content at query #16
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher(123) is None
    assert matcher('other_123') is None
    
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('hello') is not None
    assert matcher2('Hello') is None
    assert matcher2('hello123') is None
    assert matcher2('') is None
    
    matcher3 = rex(r'.*')
    assert matcher3('any string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None


# LLM-generated content at query #17
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher('test123extra') is None
    assert matcher(123) is False
    assert matcher(['test123']) is False
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('abc123') is None
    assert matcher2('ABC') is None
    assert matcher2('') is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test0') is not None
    assert matcher('test') is None  # no digits
    assert matcher('123test') is None  # wrong pattern
    assert matcher(123) is None  # not a string
    
    # Test with non-matching string
    matcher = rex(r'^[A-Z]+$')
    assert matcher('HELLO') is not None
    assert matcher('Hello') is None  # lowercase letter
    assert matcher('123') is None  # digits
    assert matcher('HELLO123') is None  # mixed
    
    # Test with empty pattern
    matcher = rex(r'')
    assert matcher('') is not None
    assert matcher('any') is not None
    
    # Test with complex pattern
    matcher = rex(r'^[a-z]+@[a-z]+\.[a-z]{2,3}$')
    assert matcher('test@example.com') is not None
    assert matcher('user@domain.org') is not None
    assert matcher('Test@example.com') is None  # uppercase
    assert matcher('test@example') is None  # no TLD
    assert matcher('test@example.comm') is None  # TLD too long
    
    # Test that returned function is callable
    matcher = rex(r'.*')
    assert callable(matcher)
    
    # Test with special characters
    matcher = rex(r'^\d+\.\d+$')
    assert matcher('3.14') is not None
    assert matcher('0.5') is not None
    assert matcher('3') is None  # no decimal
    assert matcher('3.') is None  # incomplete


# LLM-generated content at query #19
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None

    matcher2 = rex(r'^[A-Z][a-z]+$')
    assert matcher2('Hello') is not None
    assert matcher2('hello') is None
    assert matcher2('HELLO') is None
    assert matcher2('Hello123') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3(123) is None

    matcher4 = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher4('123-45-6789') is not None
    assert matcher4('12-345-6789') is None
    assert matcher4('123-45-678') is None
    assert matcher4('abc-de-fghi') is None


# LLM-generated content at query #20
#--------------------------

```python
def test_rex():
    matcher = rex(r'^a\d+$')
    assert matcher('a123') is not None
    assert matcher('a') is None
    assert matcher('123') is None
    assert matcher('a123b') is None
    assert matcher(123) is False
    assert matcher(['a123']) is False

    matcher = rex(r'^test.*')
    assert matcher('test_string') is not None
    assert matcher('not_test') is None
    assert matcher('') is None

    matcher = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher('123-45-6789') is not None
    assert matcher('12-345-6789') is None
    assert matcher('abc-def-ghij') is None


# LLM-generated content at query #21
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    assert matcher({"key": "test_123"}) is None
    assert matcher(None) is None
    assert matcher("") is None
    matcher2 = rex(r"a.b")
    assert matcher2("acb") is not None
    assert matcher2("a b") is not None
    assert matcher2("ab") is None


# LLM-generated content at query #22
#--------------------------

```python
def test_rex():
    matcher = rex(r'^a\d+$')
    assert matcher('a123') is not None
    assert matcher('a') is None
    assert matcher('123') is None
    assert matcher('a12b') is None
    assert matcher(123) is False
    assert matcher(['a123']) is False

    matcher = rex(r'^test.*')
    assert matcher('test_string') is not None
    assert matcher('not_test') is None
    assert matcher('') is None

    matcher = rex(r'^\d{3}-\d{2}$')
    assert matcher('123-45') is not None
    assert matcher('12-345') is None
    assert matcher('abc-de') is None


# LLM-generated content at query #23
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'test'}) is None
    
    # Test with different patterns
    matcher_digits = rex(r'^\d+$')
    assert matcher_digits('123') is not None
    assert matcher_digits('abc') is None
    
    matcher_email = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert matcher_email('test@example.com') is not None
    assert matcher_email('invalid_email') is None
    
    # Test that returned function works with re.match object methods
    matcher_groups = rex(r'^(\d+)-(\d+)$')
    match = matcher_groups('123-456')
    assert match is not None
    assert match.groups() == ('123', '456')


# LLM-generated content at query #24
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test0') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher(123) is None
    
    # Test with non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    matcher2 = rex(r'[a-z]+')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('123') is None
    
    # Test with empty pattern
    matcher3 = rex(r'')
    assert matcher3('') is not None
    assert matcher3('anything') is not None
    
    # Test that returned function is callable
    assert callable(matcher)
    
    # Test that pattern matching works correctly
    matcher4 = rex(r'^[A-Z][a-z]*$')
    assert matcher4('Test') is not None
    assert matcher4('test') is None
    assert matcher4('TEST') is None
    assert matcher4('Test123') is None


# LLM-generated content at query #25
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher(['test']) is None
    
    # Test with different patterns
    matcher_digit = rex(r'^\d+$')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    matcher_word = rex(r'^[A-Za-z]+$')
    assert matcher_word('hello') is not None
    assert matcher_word('hello123') is None
    
    # Test that returned function works with transform
    # This tests integration with the transform system
    from pyrsistent import pmap
    structure = pmap({'test_key': 1, 'other_key': 2})
    
    # Use rex in a transform path
    result = transform(structure, [rex(r'^test'), inc])
    assert result['test_key'] == 2
    assert result['other_key'] == 2


# LLM-generated content at query #26
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\w+$")
    assert matcher("test_function") is not None
    assert matcher("test_function").group() == "test_function"
    assert matcher("test_123") is not None
    assert matcher("test_") is not None
    assert matcher("not_test") is None
    assert matcher("test-with-dash") is None
    assert matcher("") is None
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher(["test_list"]) is None
    matcher2 = rex(r"\d+")
    assert matcher2("123") is not None
    assert matcher2("abc") is None
    assert matcher2("123abc") is not None
    assert matcher2("abc123") is not None
    matcher3 = rex(r"^[A-Z][a-z]*$")
    assert matcher3("Hello") is not None
    assert matcher3("hello") is None
    assert matcher3("HELLO") is None
    assert matcher3("H") is not None
    assert matcher3("") is None


# LLM-generated content at query #27
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string patterns
    assert matcher('test_string') is not None
    assert matcher('testing') is not None
    assert matcher('no_match') is None
    
    # Test with different patterns
    digit_matcher = rex(r'^\d+$')
    assert digit_matcher('123') is not None
    assert digit_matcher('abc') is None
    assert digit_matcher('123abc') is None
    
    # Test that non-strings return None
    assert digit_matcher(123) is None
    assert digit_matcher([1, 2, 3]) is None
    assert digit_matcher({'key': 'value'}) is None
    
    # Test case sensitivity
    case_sensitive = rex(r'^Test')
    assert case_sensitive('Test') is not None
    assert case_sensitive('test') is None
    
    # Test with special regex characters
    special_matcher = rex(r'^a\.b$')
    assert special_matcher('a.b') is not None
    assert special_matcher('aXb') is None
    
    # Test empty string matching
    empty_matcher = rex(r'^$')
    assert empty_matcher('') is not None
    assert empty_matcher('a') is None


# LLM-generated content at query #28
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'value'}) is None
    
    # Test with different patterns
    number_matcher = rex(r'^\d+$')
    assert number_matcher('123') is not None
    assert number_matcher('abc') is None
    
    # Test with complex pattern
    email_matcher = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert email_matcher('test@example.com') is not None
    assert email_matcher('invalid-email') is None
    
    # Test that compiled regex is used
    pattern = r'^a+b*$'
    matcher = rex(pattern)
    assert matcher('aaaab') is not None
    assert matcher('b') is None


# LLM-generated content at query #29
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('no_match') is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher(['test']) is None
    
    # Test with different patterns
    matcher_digit = rex(r'^\d+$')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    # Test partial matches
    matcher_partial = rex(r'hello')
    assert matcher_partial('hello world') is not None
    assert matcher_partial('world hello') is not None
    
    # Test case sensitivity
    matcher_case = rex(r'^TEST$')
    assert matcher_case('TEST') is not None
    assert matcher_case('test') is None


# LLM-generated content at query #30
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test0') is not None
    assert matcher('test') is None  # No digits
    assert matcher('123test') is None  # Wrong pattern
    assert matcher(123) is None  # Not a string
    
    # Test with non-matching string
    matcher2 = rex(r'^[A-Z]+$')
    assert matcher2('HELLO') is not None
    assert matcher2('Hello') is None  # Lowercase letter
    assert matcher2('123') is None
    assert matcher2('') is None
    
    # Test with complex pattern
    matcher3 = rex(r'^[a-z]+_[a-z]+$')
    assert matcher3('hello_world') is not None
    assert matcher3('test_case') is not None
    assert matcher3('Hello_World') is None  # Uppercase
    assert matcher3('hello-world') is None  # Wrong separator
    assert matcher3('hello') is None  # No underscore
    
    # Test that returns compiled regex matcher function
    assert callable(rex(r'.*'))
    assert hasattr(rex(r'.*'), '__call__')


# LLM-generated content at query #31
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r"^test")
    assert callable(matcher)
    
    # Test matching string
    assert matcher("test_string") is not None
    assert matcher("test") is not None
    
    # Test non-matching string
    assert matcher("other_string") is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher(["test"]) is None
    
    # Test with different patterns
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is not None
    assert matcher_digit("abc") is None
    
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is not None
    assert matcher_exact("exact_extra") is None
    
    # Test that pattern is compiled
    matcher = rex(r"pattern")
    assert hasattr(matcher, '__call__')


# LLM-generated content at query #32
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher(123) is None
    assert matcher('other_123') is None
    
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('hello') is not None
    assert matcher2('Hello') is None
    assert matcher2('123') is None
    assert matcher2('') is None
    
    matcher3 = rex(r'.*')
    assert matcher3('any string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None


# LLM-generated content at query #33
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher(123) is None
    assert matcher('test123extra') is None

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('abc123') is None
    assert matcher2('ABC') is None
    assert matcher2('') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3(123) is None


# LLM-generated content at query #34
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test0') is not None
    
    # Test with non-matching string
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher('test_abc') is None
    
    # Test with non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher(['test']) is None
    
    # Test with different patterns
    matcher2 = rex(r'[a-z]+')
    assert matcher2('hello') is not None
    assert matcher2('HELLO') is None
    
    # Test with empty string pattern
    matcher3 = rex(r'')
    assert matcher3('') is not None
    assert matcher3('any') is not None
    
    # Test with complex pattern
    matcher4 = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher4('123-45-6789') is not None
    assert matcher4('12-345-6789') is None


# LLM-generated content at query #35
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher(123) is None
    assert matcher('test123extra') is None

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3(123) is None


# LLM-generated content at query #36
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher(123) is None
    assert matcher('other_123') is None
    
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('hello') is not None
    assert matcher2('Hello') is None
    assert matcher2('123') is None
    assert matcher2('') is None
    
    matcher3 = rex(r'.*')
    assert matcher3('any_string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None


# LLM-generated content at query #37
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher(123) is None
    assert matcher('test123extra') is None

    matcher2 = rex(r'^a.*b$')
    assert matcher2('ab') is not None
    assert matcher2('axxxb') is not None
    assert matcher2('ba') is None
    assert matcher2('a') is None

    matcher3 = rex(r'^\d+$')
    assert matcher3('123') is not None
    assert matcher3('12a3') is None
    assert matcher3('') is None


# LLM-generated content at query #38
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None
    assert matcher2('') is None
    
    matcher3 = rex(r'.*')
    assert matcher3('any_string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None


# LLM-generated content at query #39
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r"^test")
    assert callable(matcher)
    
    # Test matching string
    assert matcher("test_string") is not None
    assert matcher("test") is not None
    
    # Test non-matching string
    assert matcher("other_string") is None
    
    # Test with non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher(["test"]) is None
    
    # Test with different patterns
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is not None
    assert matcher_digit("abc") is None
    
    # Test with complex pattern
    matcher_email = rex(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    assert matcher_email("test@example.com") is not None
    assert matcher_email("invalid_email") is None
    
    # Test that pattern is compiled only once
    pattern = r"^test"
    matcher1 = rex(pattern)
    matcher2 = rex(pattern)
    # Both should work the same way
    assert matcher1("test") is not None
    assert matcher2("test") is not None


# LLM-generated content at query #40
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_") is None
    assert matcher("test_abc") is None
    assert matcher(123) is None
    assert matcher("other") is None

    matcher = rex(r"a.*b")
    assert matcher("ab") is not None
    assert matcher("axxxb") is not None
    assert matcher("ba") is None
    assert matcher("") is None

    matcher = rex(r"\d{3}-\d{2}")
    assert matcher("123-45") is not None
    assert matcher("12-345") is None
    assert matcher("abc-de") is None


# LLM-generated content at query #41
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    
    matcher = rex(r'[a-z]+')
    assert matcher('abc') is not None
    assert matcher('ABC') is None
    assert matcher('') is not None
    
    matcher = rex(r'\d{3}-\d{2}-\d{4}')
    assert matcher('123-45-6789') is not None
    assert matcher('12-345-6789') is None
    assert matcher('abc-def-ghij') is None


# LLM-generated content at query #42
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test with non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    matcher_digit = rex(r'^\d+$')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    # Test with complex pattern
    matcher_email = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert matcher_email('test@example.com') is not None
    assert matcher_email('invalid_email') is None
    
    # Test that returned matcher works as predicate
    assert rex(r'^a')('apple') is not None
    assert rex(r'^a')('banana') is None


# LLM-generated content at query #43
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'test'}) is None
    
    # Test with different patterns
    matcher_digit = rex(r'^\d+$')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    matcher_word = rex(r'^\w+$')
    assert matcher_word('hello') is not None
    assert matcher_word('hello world') is None
    
    # Test that pattern is compiled correctly
    pattern = re.compile(r'^a.*b$')
    matcher_custom = rex(r'^a.*b$')
    test_string = 'axyzb'
    assert bool(matcher_custom(test_string)) == bool(pattern.match(test_string))


# LLM-generated content at query #44
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_0") is not None
    assert matcher("test_") is None
    assert matcher("123_test") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    assert matcher({"key": "test_123"}) is None
    assert matcher(None) is None
    assert matcher("") is None
    matcher2 = rex(r"a.b")
    assert matcher2("acb") is not None
    assert matcher2("ab") is None
    matcher3 = rex(r"\d{3}-\d{2}")
    assert matcher3("123-45") is not None
    assert matcher3("12-345") is None


# LLM-generated content at query #45
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r"^test")
    assert callable(matcher)
    
    # Test matching string
    assert matcher("test_string") is not None
    assert matcher("test") is not None
    
    # Test non-matching string
    assert matcher("other_string") is None
    
    # Test with non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is not None
    assert matcher_digit("abc") is None
    
    matcher_exact = rex(r"^exact$")
    assert matcher_exact("exact") is not None
    assert matcher_exact("exact_extra") is None
    
    # Test that returned function can be used as predicate in transform
    from pyrsistent import pmap
    data = pmap({"test1": 1, "test2": 2, "other": 3})
    
    # This tests integration with transform system
    result = transform(data, [rex(r"^test"), inc])
    assert result["test1"] == 2
    assert result["test2"] == 3
    assert result["other"] == 3


# LLM-generated content at query #46
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    matcher_digits = rex(r'^\d+$')
    assert matcher_digits('123') is not None
    assert matcher_digits('abc') is None
    
    # Test pattern with special characters
    matcher_email = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert matcher_email('test@example.com') is not None
    assert matcher_email('invalid_email') is None
    
    # Test that returned matcher can be used as predicate
    assert rex(r'^a')('apple') is not None
    assert rex(r'^a')('banana') is None


# LLM-generated content at query #47
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    digit_matcher = rex(r'^\d+$')
    assert digit_matcher('123') is not None
    assert digit_matcher('abc') is None
    
    # Test with complex pattern
    email_matcher = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert email_matcher('test@example.com') is not None
    assert email_matcher('invalid_email') is None
    
    # Test that pattern is compiled only once
    pattern = r'^foo'
    matcher1 = rex(pattern)
    matcher2 = rex(pattern)
    # Both should work the same
    assert matcher1('foo_bar') is not None
    assert matcher2('foo_bar') is not None
    assert matcher1('bar_foo') is None
    assert matcher2('bar_foo') is None


# LLM-generated content at query #48
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('other_string') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'test'}) is None
    
    # Test with different patterns
    matcher_digit = rex(r'^\d+$')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    matcher_word = rex(r'^\w+$')
    assert matcher_word('hello') is not None
    assert matcher_word('hello world') is None
    
    # Test that returned function works with re.match object
    matcher = rex(r'^a(b)c$')
    match = matcher('abc')
    assert match is not None
    assert match.group(1) == 'b'


# LLM-generated content at query #49
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher('test123extra') is None
    assert matcher(123) is False
    assert matcher(['test123']) is False

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('abc123') is None
    assert matcher2('ABC') is None

    matcher3 = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher3('123-45-6789') is not None
    assert matcher3('12-345-6789') is None
    assert matcher3('123-45-67890') is None

    matcher4 = rex(r'.*')
    assert matcher4('') is not None
    assert matcher4('any string') is not None
    assert matcher4('another') is not None


# LLM-generated content at query #50
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_') is None
    assert matcher('test_abc') is None
    assert matcher(123) is None
    assert matcher('other_123') is None
    
    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('hello') is not None
    assert matcher2('Hello') is None
    assert matcher2('123') is None
    assert matcher2('') is None
    
    matcher3 = rex(r'.*')
    assert matcher3('any string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None


# LLM-generated content at query #51
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string patterns
    assert matcher('test_string') is not None
    assert matcher('testing') is not None
    assert matcher('no_match') is None
    
    # Test that non-strings return None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'test'}) is None
    
    # Test with different patterns
    digit_matcher = rex(r'^\d+$')
    assert digit_matcher('123') is not None
    assert digit_matcher('abc') is None
    
    # Test full string matching
    exact_matcher = rex(r'^exact$')
    assert exact_matcher('exact') is not None
    assert exact_matcher('exact_extra') is None
    
    # Test compiled regex properties
    matcher = rex(r'pattern')
    result = matcher('pattern')
    assert hasattr(result, 'group')
    assert result.string == 'pattern'


# LLM-generated content at query #52
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('no_match') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'test'}) is None
    
    # Test with different patterns
    digit_matcher = rex(r'\d+')
    assert digit_matcher('123') is not None
    assert digit_matcher('abc') is None
    
    # Test pattern with special characters
    email_matcher = rex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert email_matcher('test@example.com') is not None
    assert email_matcher('invalid_email') is None
    
    # Test that returned function can be used as predicate
    keys = ['test1', 'test2', 'other', 'test3']
    filtered = [k for k in keys if rex(r'^test')(k)]
    assert filtered == ['test1', 'test2', 'test3']


# LLM-generated content at query #53
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r"^test")
    assert callable(matcher)
    
    # Test matching string
    assert matcher("test_string") is not None
    assert matcher("test") is not None
    
    # Test non-matching string
    assert matcher("other_string") is None
    
    # Test non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher(["test"]) is None
    
    # Test with different patterns
    digit_matcher = rex(r"\d+")
    assert digit_matcher("123") is not None
    assert digit_matcher("abc") is None
    
    # Test full string matching
    exact_matcher = rex(r"^exact$")
    assert exact_matcher("exact") is not None
    assert exact_matcher("exact_extra") is None
    assert exact_matcher("prefix_exact") is None
    
    # Test that returned function can be used as predicate
    keys = ["a1", "b2", "c", "d3", "e4f"]
    pattern = rex(r"^[a-z]\d$")
    filtered = [k for k in keys if pattern(k)]
    assert filtered == ["a1", "b2", "d3"]


# LLM-generated content at query #54
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    assert matcher(None) is None

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None
    assert matcher2('') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3(123) is None


# LLM-generated content at query #55
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r"^test")
    assert callable(matcher)
    
    # Test matching string
    assert matcher("test_string") is not None
    assert matcher("test") is not None
    
    # Test non-matching string
    assert matcher("no_match") is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher(["list"]) is None
    
    # Test with different patterns
    matcher_digit = rex(r"\d+")
    assert matcher_digit("123") is not None
    assert matcher_digit("abc") is None
    
    # Test with complex pattern
    matcher_email = rex(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    assert matcher_email("test@example.com") is not None
    assert matcher_email("invalid_email") is None
    
    # Test that returned function can be used as predicate
    matcher_a = rex(r"^a")
    items = ["apple", "banana", "apricot", "cherry"]
    filtered = [item for item in items if matcher_a(item)]
    assert filtered == ["apple", "apricot"]


# LLM-generated content at query #56
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher(123) is None
    assert matcher('test123extra') is None

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None
    assert matcher2('') is None

    matcher3 = rex(r'.*')
    assert matcher3('any string') is not None
    assert matcher3('') is not None
    assert matcher3(123) is None

    matcher4 = rex(r'^\d{3}-\d{2}-\d{4}$')
    assert matcher4('123-45-6789') is not None
    assert matcher4('12-345-6789') is None
    assert matcher4('123-45-678') is None


# LLM-generated content at query #57
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    assert matcher(['test_123']) is None
    assert matcher({'key': 'test_123'}) is None

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None
    assert matcher2('') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3(123) is None


# LLM-generated content at query #58
#--------------------------

```python
def test_rex():
    # Test that rex returns a callable
    matcher = rex(r'^test')
    assert callable(matcher)
    
    # Test matching string
    assert matcher('test_string') is not None
    assert matcher('test') is not None
    
    # Test non-matching string
    assert matcher('no_match') is None
    
    # Test non-string input returns None
    assert matcher(123) is None
    assert matcher(['test']) is None
    assert matcher({'key': 'test'}) is None
    
    # Test with different patterns
    matcher_digit = rex(r'\d+')
    assert matcher_digit('123') is not None
    assert matcher_digit('abc') is None
    
    matcher_exact = rex(r'^exact$')
    assert matcher_exact('exact') is not None
    assert matcher_exact('exact_extra') is None
    assert matcher_exact('extra_exact') is None
    
    # Test that returned function can be used as predicate in transform
    from pyrsistent import pmap
    structure = pmap({'test1': 1, 'test2': 2, 'other': 3})
    
    # This tests integration with transform system
    matcher = rex(r'^test')
    result = [(k, v) for k, v in structure.items() if matcher(k)]
    assert len(result) == 2
    assert ('test1', 1) in result
    assert ('test2', 2) in result
    assert ('other', 3) not in result


# LLM-generated content at query #59
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher(123) is None
    assert matcher('test123extra') is None

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3(123) is None


# LLM-generated content at query #60
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    
    # Test with non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    matcher2 = rex(r'[a-z]+')
    assert matcher2('hello') is not None
    assert matcher2('HELLO') is None
    assert matcher2('123') is None
    
    # Test with empty pattern
    matcher3 = rex(r'')
    assert matcher3('') is not None
    assert matcher3('anything') is not None
    
    # Test that returns callable
    assert callable(rex(r'.*'))
    
    # Test with special regex characters
    matcher4 = rex(r'\.\*\+\?')
    assert matcher4('.*+?') is not None
    assert matcher4('test') is None


# LLM-generated content at query #61
#--------------------------

```python
def test_rex():
    matcher = rex(r'^test\d+$')
    assert matcher('test123') is not None
    assert matcher('test') is None
    assert matcher('123test') is None
    assert matcher('test123extra') is None
    assert matcher(123) is False
    assert matcher(['test123']) is False
    assert matcher({'key': 'test123'}) is False

    matcher2 = rex(r'^[a-z]+$')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('abc123') is None
    assert matcher2('') is None

    matcher3 = rex(r'.*')
    assert matcher3('') is not None
    assert matcher3('any string') is not None
    assert matcher3('another') is not None


# LLM-generated content at query #62
#--------------------------

```python
def test_rex():
    # Test with matching string
    matcher = rex(r'^test_\d+$')
    assert matcher('test_123') is not None
    assert matcher('test_0') is not None
    assert matcher('test_') is None
    assert matcher('123_test') is None
    assert matcher(123) is None
    
    # Test with non-string input
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None
    
    # Test with different patterns
    matcher2 = rex(r'[a-z]+')
    assert matcher2('abc') is not None
    assert matcher2('ABC') is None
    assert matcher2('123') is None
    
    # Test with empty pattern
    matcher3 = rex(r'')
    assert matcher3('') is not None
    assert matcher3('any') is not None
    
    # Test that returned function is callable
    assert callable(rex(r'.*'))


# LLM-generated content at query #63
#--------------------------

```python
def test_rex():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_") is None
    assert matcher("test_abc") is None
    assert matcher(123) is None
    assert matcher(["test_123"]) is None
    
    matcher2 = rex(r"a.*b")
    assert matcher2("ab") is not None
    assert matcher2("axxxb") is not None
    assert matcher2("ac") is None
    assert matcher2("") is None
    
    matcher3 = rex(r"\d+")
    assert matcher3("123") is not None
    assert matcher3("abc") is None
    assert matcher3("123abc") is not None


