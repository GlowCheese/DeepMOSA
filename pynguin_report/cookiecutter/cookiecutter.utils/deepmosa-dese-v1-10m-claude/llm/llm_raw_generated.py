####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    
    def uppercase_filter(text: str) -> str:
        return text.upper()
    
    extension_class = simple_filter(uppercase_filter)
    
    env = Environment(extensions=[extension_class])
    
    assert "uppercase_filter" in env.filters
    assert env.filters["uppercase_filter"] is uppercase_filter
    assert extension_class.__name__ == "uppercase_filter"
    assert env.filters["uppercase_filter"]("hello") == "HELLO"


def test_simple_filter_multiple_filters():
    from jinja2 import Environment
    
    def reverse_filter(text: str) -> str:
        return text[::-1]
    
    def double_filter(value: int) -> int:
        return value * 2
    
    reverse_extension = simple_filter(reverse_filter)
    double_extension = simple_filter(double_filter)
    
    env = Environment(extensions=[reverse_extension, double_extension])
    
    assert "reverse_filter" in env.filters
    assert "double_filter" in env.filters
    assert env.filters["reverse_filter"]("hello") == "olleh"
    assert env.filters["double_filter"](5) == 10


def test_simple_filter_extension_inheritance():
    from jinja2 import Extension
    
    def custom_filter(value: str) -> str:
        return f"[{value}]"
    
    extension_class = simple_filter(custom_filter)
    
    assert issubclass(extension_class, Extension)
    assert extension_class.__name__ == "custom_filter"


def test_simple_filter_with_template():
    from jinja2 import Environment
    
    def add_prefix(text: str) -> str:
        return f"PREFIX_{text}"
    
    extension_class = simple_filter(add_prefix)
    env = Environment(extensions=[extension_class])
    
    template = env.from_string("{{ 'test' | add_prefix }}")
    result = template.render()
    
    assert result == "PREFIX_test"


# LLM-generated content at query #2
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
    
    def reverse_string(text):
        return text[::-1]
    
    FilterClass = simple_filter(reverse_string)
    
    assert FilterClass.__name__ == "reverse_string"
    
    env = Environment()
    FilterClass(env)
    
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"]("abc") == "cba"


def test_simple_filter_multiple_instances():
    from jinja2 import Environment, Extension
    
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


def test_simple_filter_with_numeric_operation():
    from jinja2 import Environment
    
    def double(num):
        return num * 2
    
    FilterClass = simple_filter(double)
    env = Environment()
    FilterClass(env)
    
    assert env.filters["double"](5) == 10
    assert env.filters["double"](0) == 0
    assert env.filters["double"](-3) == -6


# LLM-generated content at query #3
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(my_custom_filter)
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] is my_custom_filter
    assert FilterExtension.__name__ == "my_custom_filter"


def test_simple_filter_with_arguments():
    from jinja2 import Environment
    
    def multiply(value, factor=2):
        return value * factor
    
    FilterExtension = simple_filter(multiply)
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "multiply" in env.filters
    assert env.filters["multiply"](5) == 10
    assert env.filters["multiply"](5, 3) == 15


def test_simple_filter_multiple_filters():
    from jinja2 import Environment
    
    def filter_one(value):
        return value + "_one"
    
    def filter_two(value):
        return value + "_two"
    
    FilterExtensionOne = simple_filter(filter_one)
    FilterExtensionTwo = simple_filter(filter_two)
    
    env = Environment()
    ext1 = FilterExtensionOne(env)
    ext2 = FilterExtensionTwo(env)
    
    assert "filter_one" in env.filters
    assert "filter_two" in env.filters
    assert env.filters["filter_one"]("test") == "test_one"
    assert env.filters["filter_two"]("test") == "test_two"


def test_simple_filter_extension_inheritance():
    from jinja2 import Environment, Extension
    
    def my_filter(value):
        return str(value).lower()
    
    FilterExtension = simple_filter(my_filter)
    
    assert issubclass(FilterExtension, Extension)


def test_simple_filter_function_preserved():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert env.filters["reverse_string"]("hello") == "olleh"


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


def test_simple_filter_multiple_filters():
    from jinja2 import Environment, Extension
    
    def filter_one(value):
        return value + "_one"
    
    def filter_two(value):
        return value + "_two"
    
    FilterClass1 = simple_filter(filter_one)
    FilterClass2 = simple_filter(filter_two)
    
    assert FilterClass1.__name__ == "filter_one"
    assert FilterClass2.__name__ == "filter_two"
    
    env = Environment()
    FilterClass1(env)
    FilterClass2(env)
    
    assert env.filters["filter_one"]("test") == "test_one"
    assert env.filters["filter_two"]("test") == "test_two"


def test_simple_filter_with_complex_logic():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    FilterClass = simple_filter(reverse_string)
    env = Environment()
    FilterClass(env)
    
    assert env.filters["reverse_string"]("hello") == "olleh"
    assert env.filters["reverse_string"]("world") == "dlrow"


# LLM-generated content at query #5
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    from jinja2 import Environment
    
    def uppercase_filter(text):
        return text.upper()
    
    extension_class = simple_filter(uppercase_filter)
    
    assert extension_class.__name__ == "uppercase_filter"
    assert issubclass(extension_class, Extension)


def test_simple_filter_registers_filter_in_environment():
    from jinja2 import Environment
    
    def reverse_filter(text):
        return text[::-1]
    
    extension_class = simple_filter(reverse_filter)
    env = Environment(extensions=[extension_class])
    
    assert "reverse_filter" in env.filters
    assert env.filters["reverse_filter"] == reverse_filter


def test_simple_filter_filter_works_correctly():
    from jinja2 import Environment
    
    def add_exclamation(text):
        return text + "!"
    
    extension_class = simple_filter(add_exclamation)
    env = Environment(extensions=[extension_class])
    
    result = env.from_string("{{ 'hello' | add_exclamation }}").render()
    assert result == "hello!"


def test_simple_filter_with_multiple_parameters():
    from jinja2 import Environment
    
    def repeat_filter(text, times=2):
        return text * times
    
    extension_class = simple_filter(repeat_filter)
    env = Environment(extensions=[extension_class])
    
    result = env.from_string("{{ 'x' | repeat_filter(3) }}").render()
    assert result == "xxx"


def test_simple_filter_preserves_function_name():
    from jinja2 import Environment
    
    def my_custom_filter(value):
        return str(value).lower()
    
    extension_class = simple_filter(my_custom_filter)
    
    assert extension_class.__name__ == "my_custom_filter"


# LLM-generated content at query #6
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
    
    extension_class = simple_filter(my_filter)
    env = Environment(extensions=[extension_class])
    
    assert "my_filter" in env.filters
    assert env.filters["my_filter"] is my_filter


def test_simple_filter_with_different_filter_functions():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    extension_class = simple_filter(reverse_string)
    env = Environment(extensions=[extension_class])
    
    assert extension_class.__name__ == "reverse_string"
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"]("hello") == "olleh"


def test_simple_filter_multiple_extensions():
    from jinja2 import Environment
    
    def filter_one(value):
        return value + "1"
    
    def filter_two(value):
        return value + "2"
    
    ext_one = simple_filter(filter_one)
    ext_two = simple_filter(filter_two)
    
    env = Environment(extensions=[ext_one, ext_two])
    
    assert "filter_one" in env.filters
    assert "filter_two" in env.filters
    assert env.filters["filter_one"]("test") == "test1"
    assert env.filters["filter_two"]("test") == "test2"


def test_simple_filter_preserves_filter_function():
    from jinja2 import Environment
    
    def my_custom_filter(value, multiplier=2):
        return value * multiplier
    
    extension_class = simple_filter(my_custom_filter)
    env = Environment(extensions=[extension_class])
    
    assert env.filters["my_custom_filter"]("x", multiplier=3) == "xxx"


# LLM-generated content at query #7
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
    assert env.filters["my_custom_filter"] is my_custom_filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"


def test_simple_filter_multiple_filters():
    from jinja2 import Environment, Extension
    
    def filter_one(value):
        return value + "_one"
    
    def filter_two(value):
        return value + "_two"
    
    ExtOne = simple_filter(filter_one)
    ExtTwo = simple_filter(filter_two)
    
    env = Environment(extensions=[ExtOne, ExtTwo])
    
    assert "filter_one" in env.filters
    assert "filter_two" in env.filters
    assert env.filters["filter_one"]("test") == "test_one"
    assert env.filters["filter_two"]("test") == "test_two"


def test_simple_filter_with_numeric_operation():
    from jinja2 import Environment
    
    def double(value):
        return value * 2
    
    FilterExtension = simple_filter(double)
    env = Environment(extensions=[FilterExtension])
    
    assert env.filters["double"](5) == 10
    assert env.filters["double"]("ab") == "abab"


def test_simple_filter_extension_initialization():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    env = Environment()
    
    instance = FilterExtension(env)
    assert isinstance(instance, Extension)
    assert "reverse_string" in env.filters


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
    
    env = Environment(extensions=[FilterExtension])
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter
    
    result = env.filters["my_custom_filter"]("hello")
    assert result == "HELLO"


def test_simple_filter_with_complex_function():
    from jinja2 import Environment, Extension
    
    def reverse_string(text):
        return text[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == "reverse_string"
    
    env = Environment(extensions=[FilterExtension])
    assert env.filters["reverse_string"]("hello") == "olleh"


def test_simple_filter_multiple_extensions():
    from jinja2 import Environment, Extension
    
    def filter1(value):
        return value + "_filter1"
    
    def filter2(value):
        return value + "_filter2"
    
    Extension1 = simple_filter(filter1)
    Extension2 = simple_filter(filter2)
    
    env = Environment(extensions=[Extension1, Extension2])
    
    assert "filter1" in env.filters
    assert "filter2" in env.filters
    assert env.filters["filter1"]("test") == "test_filter1"
    assert env.filters["filter2"]("test") == "test_filter2"


def test_simple_filter_with_numeric_function():
    from jinja2 import Environment
    
    def double(value):
        return value * 2
    
    FilterExtension = simple_filter(double)
    env = Environment(extensions=[FilterExtension])
    
    assert env.filters["double"](5) == 10
    assert env.filters["double"]([1, 2]) == [1, 2, 1, 2]


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    
    def reverse_string(value):
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == "reverse_string"
    
    env = Environment()
    FilterExtension(env)
    
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"]("abc") == "cba"


def test_simple_filter_multiple_extensions():
    from jinja2 import Environment, Extension
    
    def filter_one(value):
        return value + "1"
    
    def filter_two(value):
        return value + "2"
    
    ExtOne = simple_filter(filter_one)
    ExtTwo = simple_filter(filter_two)
    
    env1 = Environment()
    env2 = Environment()
    
    ExtOne(env1)
    ExtTwo(env2)
    
    assert "filter_one" in env1.filters
    assert "filter_two" not in env1.filters
    assert "filter_two" in env2.filters
    assert "filter_one" not in env2.filters


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    from jinja2.ext import Extension
    
    def uppercase_filter(text):
        return text.upper()
    
    FilterClass = simple_filter(uppercase_filter)
    
    assert issubclass(FilterClass, Extension)
    assert FilterClass.__name__ == "uppercase_filter"
    
    env = Environment()
    filter_instance = FilterClass(env)
    
    assert "uppercase_filter" in env.filters
    assert env.filters["uppercase_filter"] is uppercase_filter
    assert env.filters["uppercase_filter"]("hello") == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    from jinja2.ext import Extension
    
    def reverse_filter(text):
        return text[::-1]
    
    FilterClass = simple_filter(reverse_filter)
    
    assert FilterClass.__name__ == "reverse_filter"
    
    env = Environment()
    FilterClass(env)
    
    assert "reverse_filter" in env.filters
    assert env.filters["reverse_filter"]("abc") == "cba"


def test_simple_filter_multiple_instances():
    from jinja2 import Environment
    
    def add_suffix(text):
        return text + "_suffix"
    
    FilterClass = simple_filter(add_suffix)
    
    env1 = Environment()
    env2 = Environment()
    
    FilterClass(env1)
    FilterClass(env2)
    
    assert "add_suffix" in env1.filters
    assert "add_suffix" in env2.filters
    assert env1.filters["add_suffix"]("test") == "test_suffix"
    assert env2.filters["add_suffix"]("test") == "test_suffix"


def test_simple_filter_preserves_function_behavior():
    from jinja2 import Environment
    
    def multiply_by_two(num):
        return num * 2
    
    FilterClass = simple_filter(multiply_by_two)
    env = Environment()
    FilterClass(env)
    
    assert env.filters["multiply_by_two"](5) == 10
    assert env.filters["multiply_by_two"](0) == 0
    assert env.filters["multiply_by_two"](-3) == -6


# LLM-generated content at query #3
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
    assert env.filters["my_filter"] == my_filter


def test_simple_filter_with_different_function_names():
    from jinja2 import Environment
    
    def custom_transform(value):
        return value + "!"
    
    extension_class = simple_filter(custom_transform)
    env = Environment()
    extension_instance = extension_class(env)
    
    assert extension_class.__name__ == "custom_transform"
    assert "custom_transform" in env.filters
    assert env.filters["custom_transform"]("hello") == "hello!"


def test_simple_filter_filter_functionality():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    env = Environment()
    extension_class = simple_filter(reverse_string)
    extension_instance = extension_class(env)
    
    result = env.filters["reverse_string"]("hello")
    assert result == "olleh"


def test_simple_filter_multiple_filters():
    from jinja2 import Environment
    
    def filter_one(value):
        return value + "1"
    
    def filter_two(value):
        return value + "2"
    
    env = Environment()
    ext_class_one = simple_filter(filter_one)
    ext_class_two = simple_filter(filter_two)
    
    ext_one = ext_class_one(env)
    ext_two = ext_class_two(env)
    
    assert "filter_one" in env.filters
    assert "filter_two" in env.filters
    assert env.filters["filter_one"]("test") == "test1"
    assert env.filters["filter_two"]("test") == "test2"


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
        return "prefix_" + value
    
    FilterClass = simple_filter(add_prefix)
    
    env1 = Environment()
    env2 = Environment()
    
    FilterClass(env1)
    FilterClass(env2)
    
    assert "add_prefix" in env1.filters
    assert "add_prefix" in env2.filters
    assert env1.filters["add_prefix"]("test") == "prefix_test"
    assert env2.filters["add_prefix"]("data") == "prefix_data"


# LLM-generated content at query #5
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    from jinja2 import Environment
    
    def my_filter(value):
        return value.upper()
    
    extension_class = simple_filter(my_filter)
    assert extension_class.__name__ == "my_filter"
    assert hasattr(extension_class, '__init__')


def test_simple_filter_registers_filter_in_environment():
    from jinja2 import Environment
    
    def my_filter(value):
        return value.upper()
    
    extension_class = simple_filter(my_filter)
    env = Environment(extensions=[extension_class])
    assert "my_filter" in env.filters
    assert env.filters["my_filter"] == my_filter


def test_simple_filter_filter_works_correctly():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    extension_class = simple_filter(reverse_string)
    env = Environment(extensions=[extension_class])
    template = env.from_string("{{ text | reverse_string }}")
    result = template.render(text="hello")
    assert result == "olleh"


def test_simple_filter_with_multiple_parameters():
    from jinja2 import Environment
    
    def add_numbers(a, b):
        return a + b
    
    extension_class = simple_filter(add_numbers)
    env = Environment(extensions=[extension_class])
    template = env.from_string("{{ 5 | add_numbers(3) }}")
    result = template.render()
    assert result == "8"


def test_simple_filter_preserves_function_name():
    from jinja2 import Environment
    
    def custom_filter_name(value):
        return str(value).lower()
    
    extension_class = simple_filter(custom_filter_name)
    assert extension_class.__name__ == "custom_filter_name"


# LLM-generated content at query #6
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(my_custom_filter)
    
    env = Environment()
    extension_instance = FilterExtension(env)
    
    assert FilterExtension.__name__ == "my_custom_filter"
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] == my_custom_filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"


def test_simple_filter_multiple_filters():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    def add_prefix(value):
        return "PREFIX_" + value
    
    FilterExtension1 = simple_filter(reverse_string)
    FilterExtension2 = simple_filter(add_prefix)
    
    env = Environment()
    FilterExtension1(env)
    FilterExtension2(env)
    
    assert FilterExtension1.__name__ == "reverse_string"
    assert FilterExtension2.__name__ == "add_prefix"
    assert env.filters["reverse_string"]("abc") == "cba"
    assert env.filters["add_prefix"]("test") == "PREFIX_test"


def test_simple_filter_with_numeric_filter():
    from jinja2 import Environment
    
    def double_number(value):
        return value * 2
    
    FilterExtension = simple_filter(double_number)
    
    env = Environment()
    FilterExtension(env)
    
    assert FilterExtension.__name__ == "double_number"
    assert env.filters["double_number"](5) == 10
    assert env.filters["double_number"](0) == 0


def test_simple_filter_returns_extension_class():
    from jinja2 import Extension
    
    def dummy_filter(value):
        return value
    
    FilterExtension = simple_filter(dummy_filter)
    
    assert isinstance(FilterExtension, type)
    assert issubclass(FilterExtension, Extension)


def test_simple_filter_extension_initialization():
    from jinja2 import Environment
    
    def test_filter(value):
        return f"[{value}]"
    
    FilterExtension = simple_filter(test_filter)
    env = Environment()
    
    extension = FilterExtension(env)
    
    assert extension.environment is env
    assert "test_filter" in env.filters


# LLM-generated content at query #7
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    from jinja2 import Environment, Extension
    
    def my_filter(value):
        return value.upper()
    
    FilterClass = simple_filter(my_filter)
    assert issubclass(FilterClass, Extension)


def test_simple_filter_registers_filter_in_environment():
    from jinja2 import Environment
    
    def my_filter(value):
        return value.upper()
    
    FilterClass = simple_filter(my_filter)
    env = Environment()
    FilterClass(env)
    
    assert "my_filter" in env.filters
    assert env.filters["my_filter"] is my_filter


def test_simple_filter_sets_class_name():
    def my_custom_filter(value):
        return value.lower()
    
    FilterClass = simple_filter(my_custom_filter)
    assert FilterClass.__name__ == "my_custom_filter"


def test_simple_filter_works_with_template_rendering():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    FilterClass = simple_filter(reverse_string)
    env = Environment()
    FilterClass(env)
    
    template = env.from_string("{{ text|reverse_string }}")
    result = template.render(text="hello")
    assert result == "olleh"


def test_simple_filter_with_multiple_arguments():
    from jinja2 import Environment
    
    def multiply(value, factor):
        return value * factor
    
    FilterClass = simple_filter(multiply)
    env = Environment()
    FilterClass(env)
    
    template = env.from_string("{{ num|multiply(3) }}")
    result = template.render(num=5)
    assert result == "15"


# LLM-generated content at query #8
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    
    # Define a simple filter function
    def my_custom_filter(value):
        return value.upper()
    
    # Apply the decorator
    FilterExtension = simple_filter(my_custom_filter)
    
    # Create a Jinja2 environment and instantiate the extension
    env = Environment()
    extension = FilterExtension(env)
    
    # Verify the filter was registered in the environment
    assert 'my_custom_filter' in env.filters
    assert env.filters['my_custom_filter'] == my_custom_filter
    
    # Verify the extension class name matches the filter function name
    assert FilterExtension.__name__ == 'my_custom_filter'
    
    # Test that the filter works correctly
    result = env.filters['my_custom_filter']('hello')
    assert result == 'HELLO'


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    
    # Define another filter function
    def reverse_string(value):
        return value[::-1]
    
    # Apply the decorator
    ReverseExtension = simple_filter(reverse_string)
    
    # Create environment and instantiate extension
    env = Environment()
    extension = ReverseExtension(env)
    
    # Verify the filter was registered
    assert 'reverse_string' in env.filters
    assert ReverseExtension.__name__ == 'reverse_string'
    
    # Test the filter functionality
    result = env.filters['reverse_string']('abc')
    assert result == 'cba'


def test_simple_filter_returns_extension_class():
    from jinja2 import Extension
    
    def dummy_filter(value):
        return value
    
    result = simple_filter(dummy_filter)
    
    # Verify it returns a class that is a subclass of Extension
    assert isinstance(result, type)
    assert issubclass(result, Extension)


