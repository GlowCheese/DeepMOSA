####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    from jinja2 import Environment
    
    def my_filter(value):
        return value.upper()
    
    extension_class = simple_filter(my_filter)
    
    assert extension_class.__name__ == "my_filter"
    assert issubclass(extension_class, Extension)


def test_simple_filter_registers_filter_in_environment():
    from jinja2 import Environment
    
    def my_filter(value):
        return value.upper()
    
    env = Environment()
    extension_class = simple_filter(my_filter)
    extension_instance = extension_class(env)
    
    assert "my_filter" in env.filters
    assert env.filters["my_filter"] is my_filter


def test_simple_filter_with_different_function_names():
    from jinja2 import Environment
    
    def uppercase_text(value):
        return value.upper()
    
    def lowercase_text(value):
        return value.lower()
    
    ext_class_1 = simple_filter(uppercase_text)
    ext_class_2 = simple_filter(lowercase_text)
    
    assert ext_class_1.__name__ == "uppercase_text"
    assert ext_class_2.__name__ == "lowercase_text"


def test_simple_filter_function_is_callable():
    from jinja2 import Environment
    
    def double_value(value):
        return value * 2
    
    env = Environment()
    extension_class = simple_filter(double_value)
    extension_instance = extension_class(env)
    
    filter_func = env.filters["double_value"]
    result = filter_func(5)
    
    assert result == 10


def test_simple_filter_with_complex_filter_function():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    env = Environment()
    extension_class = simple_filter(reverse_string)
    extension_instance = extension_class(env)
    
    filter_func = env.filters["reverse_string"]
    result = filter_func("hello")
    
    assert result == "olleh"


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment, Extension
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(my_custom_filter)
    
    assert issubclass(FilterExtension, Extension)
    assert FilterExtension.__name__ == "my_custom_filter"
    
    env = Environment()
    extension_instance = FilterExtension(env)
    
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] is my_custom_filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment, Extension
    
    def reverse_string(text):
        return text[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == "reverse_string"
    
    env = Environment()
    FilterExtension(env)
    
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"]("abc") == "cba"


def test_simple_filter_multiple_extensions():
    from jinja2 import Environment
    
    def filter_one(x):
        return x + 1
    
    def filter_two(x):
        return x * 2
    
    Ext1 = simple_filter(filter_one)
    Ext2 = simple_filter(filter_two)
    
    env = Environment()
    Ext1(env)
    Ext2(env)
    
    assert env.filters["filter_one"](5) == 6
    assert env.filters["filter_two"](5) == 10


# LLM-generated content at query #3
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    from jinja2 import Environment, Extension
    
    def my_filter(value):
        return value.upper()
    
    result = simple_filter(my_filter)
    
    assert issubclass(result, Extension)
    assert result.__name__ == "my_filter"


def test_simple_filter_registers_filter_in_environment():
    from jinja2 import Environment
    
    def my_filter(value):
        return value.upper()
    
    env = Environment()
    extension_class = simple_filter(my_filter)
    extension_instance = extension_class(env)
    
    assert "my_filter" in env.filters
    assert env.filters["my_filter"] == my_filter


def test_simple_filter_filter_works_in_template():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    env = Environment()
    extension_class = simple_filter(reverse_string)
    extension_instance = extension_class(env)
    
    template = env.from_string("{{ text|reverse_string }}")
    result = template.render(text="hello")
    
    assert result == "olleh"


def test_simple_filter_with_multiple_filters():
    from jinja2 import Environment
    
    def add_prefix(value):
        return "prefix_" + value
    
    def add_suffix(value):
        return value + "_suffix"
    
    env = Environment()
    ext1 = simple_filter(add_prefix)(env)
    ext2 = simple_filter(add_suffix)(env)
    
    assert "add_prefix" in env.filters
    assert "add_suffix" in env.filters
    assert env.filters["add_prefix"]("test") == "prefix_test"
    assert env.filters["add_suffix"]("test") == "test_suffix"


def test_simple_filter_preserves_function_behavior():
    from jinja2 import Environment
    
    def multiply_by_two(value):
        return value * 2
    
    env = Environment()
    extension_class = simple_filter(multiply_by_two)
    extension_instance = extension_class(env)
    
    assert env.filters["multiply_by_two"](5) == 10
    assert env.filters["multiply_by_two"](3) == 6


# LLM-generated content at query #4
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    from jinja2.ext import Extension
    
    def my_custom_filter(value):
        return str(value).upper()
    
    FilterExtension = simple_filter(my_custom_filter)
    
    assert issubclass(FilterExtension, Extension)
    assert FilterExtension.__name__ == "my_custom_filter"
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] is my_custom_filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    from jinja2.ext import Extension
    
    def reverse_string(value):
        return str(value)[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == "reverse_string"
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"]("abc") == "cba"


def test_simple_filter_multiple_extensions():
    from jinja2 import Environment
    
    def filter_one(value):
        return value + "_one"
    
    def filter_two(value):
        return value + "_two"
    
    ExtOne = simple_filter(filter_one)
    ExtTwo = simple_filter(filter_two)
    
    env = Environment()
    ext_one = ExtOne(env)
    ext_two = ExtTwo(env)
    
    assert "filter_one" in env.filters
    assert "filter_two" in env.filters
    assert env.filters["filter_one"]("test") == "test_one"
    assert env.filters["filter_two"]("test") == "test_two"


# LLM-generated content at query #5
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment, Extension
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(my_custom_filter)
    
    assert issubclass(FilterExtension, Extension)
    assert FilterExtension.__name__ == "my_custom_filter"
    
    env = Environment(extensions=[FilterExtension])
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter
    
    result = env.filters["my_custom_filter"]("hello")
    assert result == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment, Extension
    
    def reverse_string(value):
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == "reverse_string"
    
    env = Environment(extensions=[FilterExtension])
    assert "reverse_string" in env.filters
    
    result = env.filters["reverse_string"]("hello")
    assert result == "olleh"


def test_simple_filter_extension_initialization():
    from jinja2 import Environment, Extension
    
    def test_filter(value):
        return f"filtered_{value}"
    
    FilterExtension = simple_filter(test_filter)
    env = Environment()
    
    extension_instance = FilterExtension(env)
    
    assert isinstance(extension_instance, Extension)
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("data") == "filtered_data"


# LLM-generated content at query #6
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    from jinja2 import Environment
    
    def my_filter(value):
        return value.upper()
    
    FilterClass = simple_filter(my_filter)
    
    assert FilterClass.__name__ == "my_filter"
    assert issubclass(FilterClass, Extension)


def test_simple_filter_registers_filter_in_environment():
    from jinja2 import Environment
    
    def my_filter(value):
        return value.upper()
    
    env = Environment()
    FilterClass = simple_filter(my_filter)
    instance = FilterClass(env)
    
    assert "my_filter" in env.filters
    assert env.filters["my_filter"] == my_filter


def test_simple_filter_with_different_function_names():
    from jinja2 import Environment
    
    def lowercase_filter(value):
        return value.lower()
    
    FilterClass = simple_filter(lowercase_filter)
    env = Environment()
    instance = FilterClass(env)
    
    assert FilterClass.__name__ == "lowercase_filter"
    assert "lowercase_filter" in env.filters


def test_simple_filter_function_is_callable():
    from jinja2 import Environment
    
    def reverse_filter(value):
        return value[::-1]
    
    FilterClass = simple_filter(reverse_filter)
    env = Environment()
    instance = FilterClass(env)
    
    result = env.filters["reverse_filter"]("hello")
    assert result == "olleh"


def test_simple_filter_with_multiple_arguments():
    from jinja2 import Environment
    
    def concat_filter(value, suffix):
        return value + suffix
    
    FilterClass = simple_filter(concat_filter)
    env = Environment()
    instance = FilterClass(env)
    
    result = env.filters["concat_filter"]("hello", " world")
    assert result == "hello world"


# LLM-generated content at query #7
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(my_custom_filter)
    
    assert FilterExtension.__name__ == "my_custom_filter"
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == "reverse_string"
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"]("abc") == "cba"


def test_simple_filter_multiple_extensions():
    from jinja2 import Environment
    
    def add_prefix(value):
        return "prefix_" + value
    
    def add_suffix(value):
        return value + "_suffix"
    
    FilterExtension1 = simple_filter(add_prefix)
    FilterExtension2 = simple_filter(add_suffix)
    
    env = Environment()
    ext1 = FilterExtension1(env)
    ext2 = FilterExtension2(env)
    
    assert "add_prefix" in env.filters
    assert "add_suffix" in env.filters
    assert env.filters["add_prefix"]("test") == "prefix_test"
    assert env.filters["add_suffix"]("test") == "test_suffix"


def test_simple_filter_returns_extension_class():
    from jinja2.ext import Extension
    
    def dummy_filter(value):
        return value
    
    FilterExtension = simple_filter(dummy_filter)
    
    assert isinstance(FilterExtension, type)
    assert issubclass(FilterExtension, Extension)


def test_simple_filter_with_numeric_filter():
    from jinja2 import Environment
    
    def double(value):
        return value * 2
    
    FilterExtension = simple_filter(double)
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert env.filters["double"](5) == 10
    assert env.filters["double"]([1, 2]) == [1, 2, 1, 2]


# LLM-generated content at query #8
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment, Extension
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterClass = simple_filter(my_custom_filter)
    
    assert issubclass(FilterClass, Extension)
    assert FilterClass.__name__ == "my_custom_filter"
    
    env = Environment()
    filter_instance = FilterClass(env)
    
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] is my_custom_filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment, Extension
    
    def reverse_string(value):
        return value[::-1]
    
    FilterClass = simple_filter(reverse_string)
    
    assert FilterClass.__name__ == "reverse_string"
    
    env = Environment()
    FilterClass(env)
    
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"]("abc") == "cba"


def test_simple_filter_multiple_instances():
    from jinja2 import Environment
    
    def add_prefix(value):
        return f"prefix_{value}"
    
    FilterClass = simple_filter(add_prefix)
    
    env1 = Environment()
    env2 = Environment()
    
    FilterClass(env1)
    FilterClass(env2)
    
    assert env1.filters["add_prefix"]("test") == "prefix_test"
    assert env2.filters["add_prefix"]("test") == "prefix_test"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    
    def my_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(my_filter)
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "my_filter" in env.filters
    assert env.filters["my_filter"] is my_filter
    assert FilterExtension.__name__ == "my_filter"


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    
    def reverse_text(value):
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_text)
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "reverse_text" in env.filters
    assert env.filters["reverse_text"]("hello") == "olleh"
    assert FilterExtension.__name__ == "reverse_text"


def test_simple_filter_integration_with_template():
    from jinja2 import Environment
    
    def double(value):
        return value * 2
    
    FilterExtension = simple_filter(double)
    
    env = Environment(extensions=[FilterExtension])
    
    template = env.from_string("{{ value|double }}")
    result = template.render(value=5)
    
    assert result == "10"


def test_simple_filter_multiple_filters():
    from jinja2 import Environment
    
    def add_prefix(value):
        return "prefix_" + value
    
    def add_suffix(value):
        return value + "_suffix"
    
    FilterExtension1 = simple_filter(add_prefix)
    FilterExtension2 = simple_filter(add_suffix)
    
    env = Environment()
    FilterExtension1(env)
    FilterExtension2(env)
    
    assert "add_prefix" in env.filters
    assert "add_suffix" in env.filters
    assert env.filters["add_prefix"]("test") == "prefix_test"
    assert env.filters["add_suffix"]("test") == "test_suffix"


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    from jinja2.ext import Extension
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(my_custom_filter)
    
    assert issubclass(FilterExtension, Extension)
    assert FilterExtension.__name__ == "my_custom_filter"
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] is my_custom_filter
    
    result = env.filters["my_custom_filter"]("hello")
    assert result == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    from jinja2.ext import Extension
    
    def reverse_string(value):
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == "reverse_string"
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "reverse_string" in env.filters
    result = env.filters["reverse_string"]("abcdef")
    assert result == "fedcba"


def test_simple_filter_multiple_extensions():
    from jinja2 import Environment
    
    def filter_one(value):
        return value + "_one"
    
    def filter_two(value):
        return value + "_two"
    
    Extension1 = simple_filter(filter_one)
    Extension2 = simple_filter(filter_two)
    
    env1 = Environment()
    env2 = Environment()
    
    ext1 = Extension1(env1)
    ext2 = Extension2(env2)
    
    assert "filter_one" in env1.filters
    assert "filter_two" in env2.filters
    assert "filter_two" not in env1.filters
    assert "filter_one" not in env2.filters
    
    assert env1.filters["filter_one"]("test") == "test_one"
    assert env2.filters["filter_two"]("test") == "test_two"


# LLM-generated content at query #3
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment, Extension
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterClass = simple_filter(my_custom_filter)
    
    assert issubclass(FilterClass, Extension)
    assert FilterClass.__name__ == "my_custom_filter"
    
    env = Environment()
    filter_instance = FilterClass(env)
    
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] is my_custom_filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    FilterClass = simple_filter(reverse_string)
    
    assert FilterClass.__name__ == "reverse_string"
    
    env = Environment()
    FilterClass(env)
    
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"]("abc") == "cba"


def test_simple_filter_multiple_instances():
    from jinja2 import Environment
    
    def add_prefix(value):
        return f"prefix_{value}"
    
    FilterClass = simple_filter(add_prefix)
    
    env1 = Environment()
    env2 = Environment()
    
    FilterClass(env1)
    FilterClass(env2)
    
    assert "add_prefix" in env1.filters
    assert "add_prefix" in env2.filters
    assert env1.filters["add_prefix"]("test") == "prefix_test"
    assert env2.filters["add_prefix"]("test") == "prefix_test"


# LLM-generated content at query #4
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment, Extension
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterClass = simple_filter(my_custom_filter)
    
    assert issubclass(FilterClass, Extension)
    assert FilterClass.__name__ == "my_custom_filter"
    
    env = Environment()
    filter_instance = FilterClass(env)
    
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment, Extension
    
    def reverse_text(value):
        return value[::-1]
    
    FilterClass = simple_filter(reverse_text)
    
    assert FilterClass.__name__ == "reverse_text"
    
    env = Environment()
    FilterClass(env)
    
    assert env.filters["reverse_text"]("abc") == "cba"


def test_simple_filter_multiple_instances():
    from jinja2 import Environment
    
    def add_prefix(value):
        return f"prefix_{value}"
    
    FilterClass = simple_filter(add_prefix)
    
    env1 = Environment()
    env2 = Environment()
    
    FilterClass(env1)
    FilterClass(env2)
    
    assert env1.filters["add_prefix"]("test") == "prefix_test"
    assert env2.filters["add_prefix"]("test") == "prefix_test"


# LLM-generated content at query #5
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(my_custom_filter)
    
    assert FilterExtension.__name__ == "my_custom_filter"
    
    env = Environment()
    extension_instance = FilterExtension(env)
    
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] is my_custom_filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == "reverse_string"
    
    env = Environment()
    FilterExtension(env)
    
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"]("abc") == "cba"


def test_simple_filter_multiple_instances():
    from jinja2 import Environment
    
    def add_prefix(value):
        return f"prefix_{value}"
    
    FilterExtension = simple_filter(add_prefix)
    
    env1 = Environment()
    env2 = Environment()
    
    FilterExtension(env1)
    FilterExtension(env2)
    
    assert "add_prefix" in env1.filters
    assert "add_prefix" in env2.filters
    assert env1.filters["add_prefix"]("test") == "prefix_test"
    assert env2.filters["add_prefix"]("test") == "prefix_test"


# LLM-generated content at query #6
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    from jinja2 import Environment
    
    def my_custom_filter(value):
        return value.upper()
    
    extension_class = simple_filter(my_custom_filter)
    assert extension_class.__name__ == "my_custom_filter"


def test_simple_filter_registers_filter_in_environment():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    extension_class = simple_filter(reverse_string)
    env = Environment(extensions=[extension_class])
    
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"] == reverse_string


def test_simple_filter_extension_is_callable():
    from jinja2 import Environment
    
    def double_value(value):
        return value * 2
    
    extension_class = simple_filter(double_value)
    env = Environment(extensions=[extension_class])
    
    result = env.filters["double_value"](5)
    assert result == 10


def test_simple_filter_with_string_processing():
    from jinja2 import Environment
    
    def add_prefix(value):
        return f"prefix_{value}"
    
    extension_class = simple_filter(add_prefix)
    env = Environment(extensions=[extension_class])
    
    result = env.filters["add_prefix"]("test")
    assert result == "prefix_test"


def test_simple_filter_preserves_function_behavior():
    from jinja2 import Environment
    
    def multiply_by_three(value):
        return value * 3
    
    extension_class = simple_filter(multiply_by_three)
    env = Environment(extensions=[extension_class])
    
    assert env.filters["multiply_by_three"](4) == 12
    assert env.filters["multiply_by_three"](0) == 0
    assert env.filters["multiply_by_three"](-2) == -6


# LLM-generated content at query #7
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    from jinja2.ext import Extension
    
    def my_custom_filter(value):
        return str(value).upper()
    
    FilterExtension = simple_filter(my_custom_filter)
    
    assert issubclass(FilterExtension, Extension)
    assert FilterExtension.__name__ == "my_custom_filter"
    
    env = Environment()
    extension_instance = FilterExtension(env)
    
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    from jinja2.ext import Extension
    
    def reverse_string(value):
        return str(value)[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == "reverse_string"
    
    env = Environment()
    extension_instance = FilterExtension(env)
    
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"]("hello") == "olleh"


def test_simple_filter_multiple_extensions():
    from jinja2 import Environment
    from jinja2.ext import Extension
    
    def add_prefix(value):
        return f"prefix_{value}"
    
    def add_suffix(value):
        return f"{value}_suffix"
    
    PrefixExtension = simple_filter(add_prefix)
    SuffixExtension = simple_filter(add_suffix)
    
    env = Environment()
    PrefixExtension(env)
    SuffixExtension(env)
    
    assert "add_prefix" in env.filters
    assert "add_suffix" in env.filters
    assert env.filters["add_prefix"]("test") == "prefix_test"
    assert env.filters["add_suffix"]("test") == "test_suffix"


# LLM-generated content at query #8
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment, Extension
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(my_custom_filter)
    
    assert issubclass(FilterExtension, Extension)
    assert FilterExtension.__name__ == "my_custom_filter"
    
    env = Environment()
    extension_instance = FilterExtension(env)
    
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] is my_custom_filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == "reverse_string"
    
    env = Environment()
    extension_instance = FilterExtension(env)
    
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"]("abc") == "cba"


def test_simple_filter_multiple_extensions():
    from jinja2 import Environment
    
    def filter_one(value):
        return value + "_one"
    
    def filter_two(value):
        return value + "_two"
    
    Ext1 = simple_filter(filter_one)
    Ext2 = simple_filter(filter_two)
    
    env = Environment()
    ext1_instance = Ext1(env)
    ext2_instance = Ext2(env)
    
    assert "filter_one" in env.filters
    assert "filter_two" in env.filters
    assert env.filters["filter_one"]("test") == "test_one"
    assert env.filters["filter_two"]("test") == "test_two"


