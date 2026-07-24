####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment, Extension
    
    def my_filter(value):
        return value.upper()
    
    FilterClass = simple_filter(my_filter)
    
    assert issubclass(FilterClass, Extension)
    assert FilterClass.__name__ == "my_filter"
    
    env = Environment()
    filter_instance = FilterClass(env)
    
    assert "my_filter" in env.filters
    assert env.filters["my_filter"] is my_filter
    assert env.filters["my_filter"]("hello") == "HELLO"


def test_simple_filter_multiple_filters():
    from jinja2 import Environment, Extension
    
    def first_filter(value):
        return value + "_first"
    
    def second_filter(value):
        return value + "_second"
    
    FirstFilterClass = simple_filter(first_filter)
    SecondFilterClass = simple_filter(second_filter)
    
    env = Environment()
    FirstFilterClass(env)
    SecondFilterClass(env)
    
    assert "first_filter" in env.filters
    assert "second_filter" in env.filters
    assert env.filters["first_filter"]("test") == "test_first"
    assert env.filters["second_filter"]("test") == "test_second"


def test_simple_filter_preserves_filter_function():
    from jinja2 import Environment
    
    def custom_filter(value, suffix=""):
        return str(value) + suffix
    
    FilterClass = simple_filter(custom_filter)
    env = Environment()
    FilterClass(env)
    
    assert env.filters["custom_filter"]("hello", suffix="!") == "hello!"
    assert env.filters["custom_filter"]("world") == "world"


def test_simple_filter_with_complex_function():
    from jinja2 import Environment
    
    def reverse_filter(value):
        return value[::-1]
    
    FilterClass = simple_filter(reverse_filter)
    env = Environment()
    FilterClass(env)
    
    assert env.filters["reverse_filter"]("hello") == "olleh"
    assert env.filters["reverse_filter"]("12345") == "54321"


# LLM-generated content at query #2
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
    
    def my_custom_filter(value):
        return value.upper()
    
    extension_class = simple_filter(my_custom_filter)
    env = Environment(extensions=[extension_class])
    
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] is my_custom_filter


def test_simple_filter_filter_works_in_jinja_template():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    extension_class = simple_filter(reverse_string)
    env = Environment(extensions=[extension_class])
    
    template = env.from_string("{{ text|reverse_string }}")
    result = template.render(text="hello")
    assert result == "olleh"


def test_simple_filter_with_multiple_filters():
    from jinja2 import Environment
    
    def add_prefix(value):
        return f"prefix_{value}"
    
    def add_suffix(value):
        return f"{value}_suffix"
    
    ext1 = simple_filter(add_prefix)
    ext2 = simple_filter(add_suffix)
    env = Environment(extensions=[ext1, ext2])
    
    assert "add_prefix" in env.filters
    assert "add_suffix" in env.filters
    assert env.filters["add_prefix"]("test") == "prefix_test"
    assert env.filters["add_suffix"]("test") == "test_suffix"


def test_simple_filter_preserves_function_name():
    from jinja2 import Environment
    
    def my_filter_function(value):
        return value
    
    extension_class = simple_filter(my_filter_function)
    assert extension_class.__name__ == "my_filter_function"


def test_simple_filter_with_lambda():
    from jinja2 import Environment
    
    filter_func = lambda x: x * 2
    filter_func.__name__ = "double"
    
    extension_class = simple_filter(filter_func)
    env = Environment(extensions=[extension_class])
    
    assert "double" in env.filters
    assert env.filters["double"](5) == 10


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_simple_filter_returns_extension_class():
    from jinja2 import Environment, Extension
    
    def test_filter(value):
        return value.upper()
    
    result = simple_filter(test_filter)
    assert issubclass(result, Extension)


def test_simple_filter_registers_filter_in_environment():
    from jinja2 import Environment
    
    def test_filter(value):
        return value.upper()
    
    env = Environment()
    extension_class = simple_filter(test_filter)
    extension_class(env)
    
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter


def test_simple_filter_extension_name_matches_function_name():
    def my_custom_filter(value):
        return value.lower()
    
    extension_class = simple_filter(my_custom_filter)
    assert extension_class.__name__ == 'my_custom_filter'


def test_simple_filter_filter_works_correctly():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    env = Environment()
    extension_class = simple_filter(reverse_string)
    extension_class(env)
    
    result = env.filters['reverse_string']('hello')
    assert result == 'olleh'


def test_simple_filter_with_multiple_filters():
    from jinja2 import Environment
    
    def filter1(value):
        return value + '_1'
    
    def filter2(value):
        return value + '_2'
    
    env = Environment()
    ext1 = simple_filter(filter1)(env)
    ext2 = simple_filter(filter2)(env)
    
    assert env.filters['filter1']('test') == 'test_1'
    assert env.filters['filter2']('test') == 'test_2'


def test_simple_filter_preserves_function_behavior():
    from jinja2 import Environment
    
    def add_prefix(value, prefix='prefix_'):
        return prefix + value
    
    env = Environment()
    extension_class = simple_filter(add_prefix)
    extension_class(env)
    
    result = env.filters['add_prefix']('test', prefix='custom_')
    assert result == 'custom_test'


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter_returns_extension_class():
    from jinja2 import Environment, Extension
    
    def test_filter(value):
        return value.upper()
    
    result = simple_filter(test_filter)
    assert issubclass(result, Extension)


def test_simple_filter_registers_filter_in_environment():
    from jinja2 import Environment
    
    def test_filter(value):
        return value.upper()
    
    env = Environment()
    extension_class = simple_filter(test_filter)
    extension_class(env)
    
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter


def test_simple_filter_extension_name_matches_function_name():
    def my_custom_filter(value):
        return value.lower()
    
    extension_class = simple_filter(my_custom_filter)
    assert extension_class.__name__ == 'my_custom_filter'


def test_simple_filter_filter_works_in_template():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    env = Environment()
    extension_class = simple_filter(reverse_string)
    extension_class(env)
    
    template = env.from_string('{{ text | reverse_string }}')
    result = template.render(text='hello')
    assert result == 'olleh'


def test_simple_filter_with_multiple_filters():
    from jinja2 import Environment
    
    def filter_one(value):
        return value + '_one'
    
    def filter_two(value):
        return value + '_two'
    
    env = Environment()
    ext_one = simple_filter(filter_one)
    ext_two = simple_filter(filter_two)
    
    ext_one(env)
    ext_two(env)
    
    assert 'filter_one' in env.filters
    assert 'filter_two' in env.filters
    assert env.filters['filter_one']('test') == 'test_one'
    assert env.filters['filter_two']('test') == 'test_two'


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
    
    extension_class = simple_filter(my_filter)
    env = Environment(extensions=[extension_class])
    assert "my_filter" in env.filters
    assert env.filters["my_filter"] is my_filter


def test_simple_filter_filter_function_works():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    extension_class = simple_filter(reverse_string)
    env = Environment(extensions=[extension_class])
    result = env.filters["reverse_string"]("hello")
    assert result == "olleh"


def test_simple_filter_with_multiple_arguments():
    from jinja2 import Environment
    
    def add_numbers(a, b):
        return a + b
    
    extension_class = simple_filter(add_numbers)
    env = Environment(extensions=[extension_class])
    result = env.filters["add_numbers"](5, 3)
    assert result == 8


def test_simple_filter_extension_initialization():
    from jinja2 import Environment
    
    def test_filter(value):
        return value
    
    extension_class = simple_filter(test_filter)
    env = Environment()
    extension_instance = extension_class(env)
    assert "test_filter" in env.filters


# LLM-generated content at query #4
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    from jinja2 import Environment
    
    def my_filter(value):
        return value.upper()
    
    extension_class = simple_filter(my_filter)
    assert extension_class.__name__ == "my_filter"


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
    
    def custom_transform(text):
        return text.lower()
    
    extension_class = simple_filter(custom_transform)
    env = Environment()
    extension_instance = extension_class(env)
    
    assert extension_class.__name__ == "custom_transform"
    assert "custom_transform" in env.filters


def test_simple_filter_filter_works_correctly():
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
    
    def filter1(value):
        return value + "1"
    
    def filter2(value):
        return value + "2"
    
    env = Environment()
    ext1 = simple_filter(filter1)(env)
    ext2 = simple_filter(filter2)(env)
    
    assert "filter1" in env.filters
    assert "filter2" in env.filters
    assert env.filters["filter1"]("test") == "test1"
    assert env.filters["filter2"]("test") == "test2"


# LLM-generated content at query #5
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


def test_simple_filter_filter_works_correctly():
    from jinja2 import Environment
    
    def reverse_filter(value):
        return value[::-1]
    
    extension_class = simple_filter(reverse_filter)
    env = Environment(extensions=[extension_class])
    
    template = env.from_string("{{ text|reverse_filter }}")
    result = template.render(text="hello")
    
    assert result == "olleh"


def test_simple_filter_with_multiple_arguments():
    from jinja2 import Environment
    
    def multiply_filter(value, factor):
        return value * factor
    
    extension_class = simple_filter(multiply_filter)
    env = Environment(extensions=[extension_class])
    
    assert env.filters["multiply_filter"] is multiply_filter
    assert env.filters["multiply_filter"](3, 4) == 12


def test_simple_filter_preserves_function_name():
    from jinja2 import Environment
    
    def custom_filter_name(value):
        return str(value)
    
    extension_class = simple_filter(custom_filter_name)
    
    assert extension_class.__name__ == "custom_filter_name"


def test_simple_filter_extension_initialization():
    from jinja2 import Environment
    
    def test_filter(value):
        return value
    
    extension_class = simple_filter(test_filter)
    env = Environment()
    
    extension_instance = extension_class(env)
    
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter


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
    
    extension_class = simple_filter(my_filter)
    env = Environment(extensions=[extension_class])
    
    assert "my_filter" in env.filters
    assert env.filters["my_filter"] is my_filter


def test_simple_filter_filter_works_correctly():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    extension_class = simple_filter(reverse_string)
    env = Environment(extensions=[extension_class])
    
    template = env.from_string("{{ text|reverse_string }}")
    result = template.render(text="hello")
    
    assert result == "olleh"


def test_simple_filter_with_multiple_arguments():
    from jinja2 import Environment
    
    def multiply(value, factor):
        return value * factor
    
    extension_class = simple_filter(multiply)
    env = Environment(extensions=[extension_class])
    
    template = env.from_string("{{ num|multiply(3) }}")
    result = template.render(num=5)
    
    assert result == "15"


def test_simple_filter_extension_initialization():
    from jinja2 import Environment
    
    def test_filter(value):
        return f"filtered: {value}"
    
    extension_class = simple_filter(test_filter)
    env = Environment()
    
    extension_instance = extension_class(env)
    
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment, Extension
    
    def my_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(my_filter)
    
    assert issubclass(FilterExtension, Extension)
    assert FilterExtension.__name__ == "my_filter"
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "my_filter" in env.filters
    assert env.filters["my_filter"] is my_filter
    assert env.filters["my_filter"]("hello") == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment, Extension
    
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
    
    def filter1(value):
        return value + "1"
    
    def filter2(value):
        return value + "2"
    
    Extension1 = simple_filter(filter1)
    Extension2 = simple_filter(filter2)
    
    env1 = Environment()
    env2 = Environment()
    
    Extension1(env1)
    Extension2(env2)
    
    assert "filter1" in env1.filters
    assert "filter2" in env2.filters
    assert env1.filters["filter1"]("test") == "test1"
    assert env2.filters["filter2"]("test") == "test2"


# LLM-generated content at query #3
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(my_custom_filter)
    
    env = Environment()
    env.add_extension(FilterExtension)
    
    assert 'my_custom_filter' in env.filters
    assert env.filters['my_custom_filter']('hello') == 'HELLO'
    assert FilterExtension.__name__ == 'my_custom_filter'


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    
    def reverse_string(text):
        return text[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    env = Environment()
    env.add_extension(FilterExtension)
    
    assert 'reverse_string' in env.filters
    assert env.filters['reverse_string']('abc') == 'cba'
    assert FilterExtension.__name__ == 'reverse_string'


def test_simple_filter_extension_initialization():
    from jinja2 import Environment
    
    def add_prefix(value):
        return f"prefix_{value}"
    
    FilterExtension = simple_filter(add_prefix)
    env = Environment()
    
    extension_instance = FilterExtension(env)
    
    assert 'add_prefix' in env.filters
    assert env.filters['add_prefix']('test') == 'prefix_test'


def test_simple_filter_multiple_filters():
    from jinja2 import Environment
    
    def filter1(value):
        return value + "1"
    
    def filter2(value):
        return value + "2"
    
    FilterExtension1 = simple_filter(filter1)
    FilterExtension2 = simple_filter(filter2)
    
    env = Environment()
    env.add_extension(FilterExtension1)
    env.add_extension(FilterExtension2)
    
    assert env.filters['filter1']('test') == 'test1'
    assert env.filters['filter2']('test') == 'test2'


# LLM-generated content at query #4
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    from jinja2 import Environment
    
    def my_filter(value):
        return value.upper()
    
    FilterClass = simple_filter(my_filter)
    assert FilterClass.__name__ == "my_filter"


def test_simple_filter_registers_filter_in_environment():
    from jinja2 import Environment
    
    def my_filter(value):
        return value.upper()
    
    FilterClass = simple_filter(my_filter)
    env = Environment()
    extension = FilterClass(env)
    
    assert "my_filter" in env.filters
    assert env.filters["my_filter"] is my_filter


def test_simple_filter_extension_is_callable():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    FilterClass = simple_filter(reverse_string)
    env = Environment()
    extension = FilterClass(env)
    
    result = env.filters["reverse_string"]("hello")
    assert result == "olleh"


def test_simple_filter_with_multiple_arguments():
    from jinja2 import Environment
    
    def add_numbers(a, b):
        return a + b
    
    FilterClass = simple_filter(add_numbers)
    env = Environment()
    extension = FilterClass(env)
    
    result = env.filters["add_numbers"](5, 3)
    assert result == 8


def test_simple_filter_preserves_function_behavior():
    from jinja2 import Environment
    
    def double(value):
        return value * 2
    
    FilterClass = simple_filter(double)
    env = Environment()
    extension = FilterClass(env)
    
    assert env.filters["double"](10) == 20
    assert env.filters["double"]("ab") == "abab"


# LLM-generated content at query #5
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    
    def test_uppercase(text):
        return text.upper()
    
    filter_class = simple_filter(test_uppercase)
    
    env = Environment()
    filter_instance = filter_class(env)
    
    assert filter_class.__name__ == "test_uppercase"
    assert "test_uppercase" in env.filters
    assert env.filters["test_uppercase"] == test_uppercase
    assert env.filters["test_uppercase"]("hello") == "HELLO"


def test_simple_filter_multiple_filters():
    from jinja2 import Environment
    
    def add_prefix(text):
        return "prefix_" + text
    
    def add_suffix(text):
        return text + "_suffix"
    
    filter_class1 = simple_filter(add_prefix)
    filter_class2 = simple_filter(add_suffix)
    
    env = Environment()
    filter_instance1 = filter_class1(env)
    filter_instance2 = filter_class2(env)
    
    assert filter_class1.__name__ == "add_prefix"
    assert filter_class2.__name__ == "add_suffix"
    assert "add_prefix" in env.filters
    assert "add_suffix" in env.filters
    assert env.filters["add_prefix"]("test") == "prefix_test"
    assert env.filters["add_suffix"]("test") == "test_suffix"


def test_simple_filter_with_numeric_function():
    from jinja2 import Environment
    
    def double(num):
        return num * 2
    
    filter_class = simple_filter(double)
    
    env = Environment()
    filter_instance = filter_class(env)
    
    assert filter_class.__name__ == "double"
    assert "double" in env.filters
    assert env.filters["double"](5) == 10
    assert env.filters["double"](0) == 0


# LLM-generated content at query #6
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
    
    result = env.filters["my_custom_filter"]("hello")
    assert result == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    
    def reverse_string(value):
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    assert FilterExtension.__name__ == "reverse_string"
    
    env = Environment()
    FilterExtension(env)
    
    assert "reverse_string" in env.filters
    result = env.filters["reverse_string"]("abc")
    assert result == "cba"


def test_simple_filter_multiple_extensions():
    from jinja2 import Environment
    
    def add_exclamation(value):
        return str(value) + "!"
    
    def add_question(value):
        return str(value) + "?"
    
    Extension1 = simple_filter(add_exclamation)
    Extension2 = simple_filter(add_question)
    
    env = Environment()
    Extension1(env)
    Extension2(env)
    
    assert "add_exclamation" in env.filters
    assert "add_question" in env.filters
    assert env.filters["add_exclamation"]("hello") == "hello!"
    assert env.filters["add_question"]("hello") == "hello?"


# LLM-generated content at query #7
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    
    def test_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(test_filter)
    
    assert FilterExtension.__name__ == "test_filter"
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter
    assert env.filters["test_filter"]("hello") == "HELLO"


def test_simple_filter_with_multiple_filters():
    from jinja2 import Environment
    
    def reverse_filter(value):
        return value[::-1]
    
    FilterExtension = simple_filter(reverse_filter)
    
    assert FilterExtension.__name__ == "reverse_filter"
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert "reverse_filter" in env.filters
    assert env.filters["reverse_filter"]("abc") == "cba"


def test_simple_filter_integration_with_template():
    from jinja2 import Environment
    
    def double_filter(value):
        return value * 2
    
    FilterExtension = simple_filter(double_filter)
    
    env = Environment(extensions=[FilterExtension])
    
    template = env.from_string("{{ 'x' | double_filter }}")
    result = template.render()
    
    assert result == "xx"


def test_simple_filter_preserves_filter_function():
    from jinja2 import Environment
    
    def custom_filter(value, suffix="!"):
        return str(value) + suffix
    
    FilterExtension = simple_filter(custom_filter)
    env = Environment()
    extension = FilterExtension(env)
    
    assert env.filters["custom_filter"] is custom_filter
    assert env.filters["custom_filter"]("test") == "test!"
    assert env.filters["custom_filter"]("test", suffix="?") == "test?"


# LLM-generated content at query #8
#--------------------------

```python
def test_simple_filter():
    from jinja2 import Environment
    
    def my_custom_filter(value):
        return value.upper()
    
    FilterExtension = simple_filter(my_custom_filter)
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert FilterExtension.__name__ == "my_custom_filter"
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] is my_custom_filter
    assert env.filters["my_custom_filter"]("hello") == "HELLO"


def test_simple_filter_with_different_function():
    from jinja2 import Environment
    
    def reverse_string(text):
        return text[::-1]
    
    FilterExtension = simple_filter(reverse_string)
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert FilterExtension.__name__ == "reverse_string"
    assert "reverse_string" in env.filters
    assert env.filters["reverse_string"]("abc") == "cba"


def test_simple_filter_multiple_instances():
    from jinja2 import Environment
    
    def add_prefix(value):
        return "prefix_" + value
    
    FilterExtension = simple_filter(add_prefix)
    
    env1 = Environment()
    env2 = Environment()
    
    extension1 = FilterExtension(env1)
    extension2 = FilterExtension(env2)
    
    assert "add_prefix" in env1.filters
    assert "add_prefix" in env2.filters
    assert env1.filters["add_prefix"]("test") == "prefix_test"
    assert env2.filters["add_prefix"]("test") == "prefix_test"


def test_simple_filter_numeric_operation():
    from jinja2 import Environment
    
    def double_number(num):
        return num * 2
    
    FilterExtension = simple_filter(double_number)
    
    env = Environment()
    extension = FilterExtension(env)
    
    assert env.filters["double_number"](5) == 10
    assert env.filters["double_number"](0) == 0
    assert env.filters["double_number"](-3) == -6


