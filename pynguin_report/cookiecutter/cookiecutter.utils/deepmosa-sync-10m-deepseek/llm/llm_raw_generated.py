####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_simple_filter_registers_filter():
    from jinja2 import Environment
    def test_filter(value):
        return value.upper()
    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

def test_simple_filter_extension_name():
    def my_custom_filter(value):
        return value
    extension_class = simple_filter(my_custom_filter)
    assert extension_class.__name__ == 'my_custom_filter'

def test_simple_filter_works_in_template():
    from jinja2 import Environment
    def double(value):
        return value * 2
    extension_class = simple_filter(double)
    env = Environment(extensions=[extension_class])
    template = env.from_string('{{ 5|double }}')
    result = template.render()
    assert result == '10'

def test_simple_filter_with_string_manipulation():
    from jinja2 import Environment
    def reverse_string(value):
        return value[::-1]
    extension_class = simple_filter(reverse_string)
    env = Environment(extensions=[extension_class])
    template = env.from_string('{{ "hello"|reverse_string }}')
    result = template.render()
    assert result == 'olleh'

def test_simple_filter_multiple_extensions():
    from jinja2 import Environment
    def add_one(value):
        return value + 1
    def subtract_one(value):
        return value - 1
    extension_class1 = simple_filter(add_one)
    extension_class2 = simple_filter(subtract_one)
    env = Environment(extensions=[extension_class1, extension_class2])
    assert 'add_one' in env.filters
    assert 'subtract_one' in env.filters
    template = env.from_string('{{ 10|add_one|subtract_one }}')
    result = template.render()
    assert result == '10'


# LLM-generated content at query #2
#--------------------------

def test_simple_filter_creates_extension_with_correct_name():
    def my_test_filter(x):
        return x.upper()
    extension_class = simple_filter(my_test_filter)
    result = extension_class.__name__
    expected = "my_test_filter"
    assert result == expected

def test_simple_filter_registers_filter_in_environment():
    def my_test_filter(x):
        return x.upper()
    extension_class = simple_filter(my_test_filter)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_class(env)
    result = "hello"
    filter_func = env.filters.get("my_test_filter")
    assert filter_func is not None
    result = filter_func("hello")
    expected = "HELLO"
    assert result == expected

def test_simple_filter_extension_inherits_from_extension():
    def dummy_filter(x):
        return x
    extension_class = simple_filter(dummy_filter)
    from jinja2.ext import Extension
    result = issubclass(extension_class, Extension)
    expected = True
    assert result == expected

def test_simple_filter_with_different_filter_function():
    def add_suffix(text):
        return text + "_suffix"
    extension_class = simple_filter(add_suffix)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_class(env)
    filter_func = env.filters.get("add_suffix")
    assert filter_func is not None
    result = filter_func("test")
    expected = "test_suffix"
    assert result == expected


# LLM-generated content at query #3
#--------------------------

def test_simple_filter_registers_filter():
    from jinja2 import Environment
    def test_filter(value):
        return value.upper()
    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter

def test_simple_filter_extension_name():
    def my_custom_filter(x):
        return x
    extension_class = simple_filter(my_custom_filter)
    assert extension_class.__name__ == 'my_custom_filter'

def test_simple_filter_works_in_template():
    from jinja2 import Environment
    def double(value):
        return value * 2
    extension_class = simple_filter(double)
    env = Environment(extensions=[extension_class])
    template = env.from_string('{{ 5|double }}')
    result = template.render()
    assert result == '10'

def test_simple_filter_with_multiple_filters():
    from jinja2 import Environment
    def add_one(value):
        return value + 1
    def add_two(value):
        return value + 2
    extension_class_one = simple_filter(add_one)
    extension_class_two = simple_filter(add_two)
    env = Environment(extensions=[extension_class_one, extension_class_two])
    assert env.filters['add_one'] is add_one
    assert env.filters['add_two'] is add_two
    template = env.from_string('{{ 5|add_one }} and {{ 5|add_two }}')
    result = template.render()
    assert result == '6 and 7'


# LLM-generated content at query #4
#--------------------------

def test_simple_filter_registers_filter():
    from jinja2 import Environment
    def my_filter(value):
        return value.upper()
    extension_class = simple_filter(my_filter)
    env = Environment(extensions=[extension_class])
    result = env.filters.get('my_filter')
    assert result is my_filter

def test_simple_filter_extension_name():
    def another_filter(value):
        return value
    extension_class = simple_filter(another_filter)
    assert extension_class.__name__ == 'another_filter'

def test_simple_filter_works_in_template():
    from jinja2 import Environment
    def double(value):
        return value * 2
    extension_class = simple_filter(double)
    env = Environment(extensions=[extension_class])
    template = env.from_string('{{ 5|double }}')
    result = template.render()
    assert result == '10'

def test_simple_filter_with_string_argument():
    from jinja2 import Environment
    def repeat(value, times):
        return value * times
    extension_class = simple_filter(repeat)
    env = Environment(extensions=[extension_class])
    template = env.from_string('{{ "a"|repeat(3) }}')
    result = template.render()
    assert result == 'aaa'


# LLM-generated content at query #5
#--------------------------

def test_simple_filter_decorator():
    from jinja2 import Environment
    def test_filter(value):
        return value.upper()
    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    result = env.from_string("{{ 'hello' | test_filter }}").render()
    assert result == 'HELLO'
def test_simple_filter_registers_correct_name():
    from jinja2 import Environment
    def custom_filter(x):
        return x * 2
    extension_class = simple_filter(custom_filter)
    env = Environment(extensions=[extension_class])
    template = env.from_string("{{ 5 | custom_filter }}")
    result = template.render()
    assert result == '10'
def test_simple_filter_extension_class_name():
    def example_filter(val):
        return val + 1
    extension_class = simple_filter(example_filter)
    assert extension_class.__name__ == 'example_filter'
def test_simple_filter_with_multiple_filters():
    from jinja2 import Environment
    def filter_a(x):
        return x + "a"
    def filter_b(x):
        return x + "b"
    extension_a = simple_filter(filter_a)
    extension_b = simple_filter(filter_b)
    env = Environment(extensions=[extension_a, extension_b])
    template = env.from_string("{{ 'test' | filter_a | filter_b }}")
    result = template.render()
    assert result == 'testab'


# LLM-generated content at query #6
#--------------------------

def test_simple_filter_creates_extension_with_correct_name():
    def my_test_filter(x):
        return x.upper()
    extension_class = simple_filter(my_test_filter)
    result = extension_class.__name__
    expected = "my_test_filter"
    assert result == expected

def test_simple_filter_registers_filter_in_environment():
    def my_test_filter(x):
        return x.upper()
    extension_class = simple_filter(my_test_filter)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_class(env)
    result = "hello"
    filter_func = env.filters.get("my_test_filter")
    assert filter_func is not None
    result = filter_func(result)
    expected = "HELLO"
    assert result == expected

def test_simple_filter_returns_extension_subclass():
    def my_test_filter(x):
        return x
    extension_class = simple_filter(my_test_filter)
    from jinja2.ext import Extension
    result = issubclass(extension_class, Extension)
    expected = True
    assert result == expected

def test_simple_filter_works_with_different_filter_names():
    def another_filter(x):
        return x * 2
    extension_class = simple_filter(another_filter)
    result = extension_class.__name__
    expected = "another_filter"
    assert result == expected

def test_simple_filter_registers_function_correctly():
    def add_suffix(x):
        return x + "_suffix"
    extension_class = simple_filter(add_suffix)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_class(env)
    filter_func = env.filters.get("add_suffix")
    assert filter_func is not None
    result = filter_func("test")
    expected = "test_suffix"
    assert result == expected


# LLM-generated content at query #7
#--------------------------

def test_simple_filter_creates_extension_with_correct_name():
    def my_filter(x):
        return x * 2
    extension_class = simple_filter(my_filter)
    result = extension_class.__name__
    expected = "my_filter"
    assert result == expected

def test_simple_filter_registers_filter_in_environment():
    def my_filter(x):
        return x * 2
    extension_class = simple_filter(my_filter)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_class(env)
    result = env.filters.get("my_filter")
    expected = my_filter
    assert result == expected

def test_simple_filter_extension_inherits_from_extension():
    def dummy_filter(x):
        return x
    extension_class = simple_filter(dummy_filter)
    from jinja2.ext import Extension
    result = issubclass(extension_class, Extension)
    expected = True
    assert result == expected

def test_simple_filter_works_with_different_filter_names():
    def another_test_filter(x):
        return x.upper()
    extension_class = simple_filter(another_test_filter)
    result = extension_class.__name__
    expected = "another_test_filter"
    assert result == expected

def test_simple_filter_registered_filter_is_callable():
    def add_suffix(x):
        return x + "_suffix"
    extension_class = simple_filter(add_suffix)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_class(env)
    filter_func = env.filters["add_suffix"]
    result = filter_func("test")
    expected = "test_suffix"
    assert result == expected


# LLM-generated content at query #8
#--------------------------

def test_simple_filter_decorator():
    from jinja2 import Environment
    def test_filter(value):
        return value.upper()
    decorated = simple_filter(test_filter)
    env = Environment(extensions=[decorated])
    result = env.from_string('{{ "hello" | test_filter }}').render()
    assert result == "HELLO"
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_simple_filter_decorator_creates_extension():
    def dummy_filter(value):
        return value.upper()
    extension_class = simple_filter(dummy_filter)
    assert extension_class.__name__ == "dummy_filter"
    assert issubclass(extension_class, Extension)

def test_simple_filter_registers_filter_in_environment():
    def dummy_filter(value):
        return value.upper()
    extension_class = simple_filter(dummy_filter)
    env = Environment()
    extension_instance = extension_class(env)
    assert "dummy_filter" in env.filters
    assert env.filters["dummy_filter"] is dummy_filter

def test_simple_filter_filter_function_works():
    def dummy_filter(value):
        return value.upper()
    extension_class = simple_filter(dummy_filter)
    env = Environment()
    extension_class(env)
    result = env.filters["dummy_filter"]("test")
    assert result == "TEST"

def test_simple_filter_with_different_function_name():
    def another_filter(value):
        return value.lower()
    extension_class = simple_filter(another_filter)
    assert extension_class.__name__ == "another_filter"
    env = Environment()
    extension_class(env)
    assert "another_filter" in env.filters
    assert env.filters["another_filter"] is another_filter
    result = env.filters["another_filter"]("TEST")
    assert result == "test"


# LLM-generated content at query #2
#--------------------------

def test_simple_filter_registers_filter():
    from jinja2 import Environment
    def my_filter(value):
        return value.upper()
    extension_class = simple_filter(my_filter)
    env = Environment(extensions=[extension_class])
    assert 'my_filter' in env.filters
    assert env.filters['my_filter'] is my_filter

def test_simple_filter_extension_name():
    def another_filter(value):
        return value
    extension_class = simple_filter(another_filter)
    assert extension_class.__name__ == 'another_filter'

def test_simple_filter_works_in_template():
    from jinja2 import Environment
    def double(value):
        return value * 2
    extension_class = simple_filter(double)
    env = Environment(extensions=[extension_class])
    template = env.from_string('{{ 5|double }}')
    result = template.render()
    assert result == '10'


# LLM-generated content at query #3
#--------------------------

def test_simple_filter_decorator():
    from jinja2 import Environment
    def test_filter(value):
        return value.upper()
    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    result = env.from_string("{{ 'hello' | test_filter }}").render()
    assert result == 'HELLO'


# LLM-generated content at query #4
#--------------------------

def test_simple_filter_decorator():
    from jinja2 import Environment
    def test_filter(value):
        return value.upper()
    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    result = env.from_string("{{ 'hello' | test_filter }}").render()
    assert result == "HELLO"
def test_simple_filter_registers_correct_name():
    from jinja2 import Environment
    def custom_filter(value):
        return value + " processed"
    extension_class = simple_filter(custom_filter)
    env = Environment(extensions=[extension_class])
    result = env.from_string("{{ 'test' | custom_filter }}").render()
    assert result == "test processed"
def test_simple_filter_returns_extension_subclass():
    from jinja2.ext import Extension
    def dummy_filter(value):
        return value
    extension_class = simple_filter(dummy_filter)
    assert issubclass(extension_class, Extension)
def test_simple_filter_extension_name():
    def my_filter(value):
        return value
    extension_class = simple_filter(my_filter)
    assert extension_class.__name__ == "my_filter"


# LLM-generated content at query #5
#--------------------------

def test_simple_filter_registers_filter():
    from jinja2 import Environment
    def my_filter(value):
        return value.upper()
    extension_class = simple_filter(my_filter)
    env = Environment(extensions=[extension_class])
    assert 'my_filter' in env.filters
    assert env.filters['my_filter'] is my_filter

def test_simple_filter_extension_name():
    def another_filter(value):
        return value
    extension_class = simple_filter(another_filter)
    assert extension_class.__name__ == 'another_filter'

def test_simple_filter_works_in_template():
    from jinja2 import Environment
    def double(value):
        return value * 2
    extension_class = simple_filter(double)
    env = Environment(extensions=[extension_class])
    template = env.from_string('{{ 5|double }}')
    result = template.render()
    assert result == '10'


# LLM-generated content at query #6
#--------------------------

def test_simple_filter_registers_filter():
    from jinja2 import Environment
    def my_filter(x):
        return x.upper()
    extension_class = simple_filter(my_filter)
    env = Environment(extensions=[extension_class])
    assert 'my_filter' in env.filters
    assert env.filters['my_filter'] is my_filter

def test_simple_filter_extension_name():
    def my_filter(x):
        return x.upper()
    extension_class = simple_filter(my_filter)
    assert extension_class.__name__ == 'my_filter'

def test_simple_filter_works_in_template():
    from jinja2 import Environment
    def my_filter(x):
        return x.upper()
    extension_class = simple_filter(my_filter)
    env = Environment(extensions=[extension_class])
    template = env.from_string("{{ 'hello' | my_filter }}")
    result = template.render()
    assert result == 'HELLO'

def test_simple_filter_with_different_function():
    from jinja2 import Environment
    def another_filter(x):
        return len(x)
    extension_class = simple_filter(another_filter)
    env = Environment(extensions=[extension_class])
    assert 'another_filter' in env.filters
    assert env.filters['another_filter'] is another_filter
    template = env.from_string("{{ 'hello' | another_filter }}")
    result = template.render()
    assert result == '5'


# LLM-generated content at query #7
#--------------------------

def test_simple_filter_decorator():
    from jinja2 import Environment
    def test_filter(value):
        return value.upper()
    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    result = env.filters['test_filter']('hello')
    assert result == 'HELLO'
def test_simple_filter_extension_name():
    def my_filter(value):
        return value
    extension_class = simple_filter(my_filter)
    assert extension_class.__name__ == 'my_filter'
def test_simple_filter_registration():
    from jinja2 import Environment
    call_count = 0
    def counting_filter(value):
        nonlocal call_count
        call_count += 1
        return value
    extension_class = simple_filter(counting_filter)
    env = Environment(extensions=[extension_class])
    env.filters['counting_filter']('test')
    assert call_count == 1
def test_simple_filter_multiple_instances():
    from jinja2 import Environment
    def add_suffix(value):
        return value + '_suffix'
    extension_class = simple_filter(add_suffix)
    env1 = Environment(extensions=[extension_class])
    env2 = Environment(extensions=[extension_class])
    result1 = env1.filters['add_suffix']('foo')
    result2 = env2.filters['add_suffix']('bar')
    assert result1 == 'foo_suffix'
    assert result2 == 'bar_suffix'


# LLM-generated content at query #8
#--------------------------

def test_simple_filter_registers_filter():
    from jinja2 import Environment
    def my_filter(value):
        return value.upper()
    extension_class = simple_filter(my_filter)
    env = Environment(extensions=[extension_class])
    assert 'my_filter' in env.filters
    assert env.filters['my_filter'] is my_filter

def test_simple_filter_extension_name():
    def another_filter(value):
        return value
    extension_class = simple_filter(another_filter)
    assert extension_class.__name__ == 'another_filter'

def test_simple_filter_works_in_template():
    from jinja2 import Environment
    def double(value):
        return value * 2
    extension_class = simple_filter(double)
    env = Environment(extensions=[extension_class])
    template = env.from_string('{{ 5|double }}')
    result = template.render()
    assert result == '10'


