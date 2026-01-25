####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_simple_filter_decorates_function():
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

def test_simple_filter_registers_in_environment():
    from jinja2 import Environment
    def custom_filter(x):
        return x * 2
    extension_class = simple_filter(custom_filter)
    env = Environment()
    extension_instance = extension_class(env)
    assert 'custom_filter' in env.filters
    assert env.filters['custom_filter'](5) == 10

def test_simple_filter_with_different_function():
    from jinja2 import Environment
    def add_suffix(val):
        return val + '_suffix'
    extension_class = simple_filter(add_suffix)
    env = Environment(extensions=[extension_class])
    result = env.filters['add_suffix']('test')
    assert result == 'test_suffix'


# LLM-generated content at query #2
#--------------------------

def test_simple_filter_creates_extension_with_correct_name():
    def dummy_filter(x):
        return x
    extension_type = simple_filter(dummy_filter)
    result = extension_type.__name__
    expected = "dummy_filter"
    assert result == expected

def test_simple_filter_registers_filter_in_environment():
    def test_filter(x):
        return x.upper()
    extension_type = simple_filter(test_filter)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_type(env)
    result = env.filters.get("test_filter")
    expected = test_filter
    assert result == expected

def test_simple_filter_extension_inherits_from_extension():
    def some_filter(x):
        return x
    extension_type = simple_filter(some_filter)
    from jinja2.ext import Extension
    result = issubclass(extension_type, Extension)
    expected = True
    assert result == expected

def test_simple_filter_preserves_filter_functionality():
    def add_suffix(value):
        return value + "_suffix"
    extension_type = simple_filter(add_suffix)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_type(env)
    filter_func = env.filters["add_suffix"]
    result = filter_func("test")
    expected = "test_suffix"
    assert result == expected


# LLM-generated content at query #3
#--------------------------

def test_simple_filter_creates_extension_with_correct_name():
    def dummy_filter(x):
        return x
    extension_type = simple_filter(dummy_filter)
    assert extension_type.__name__ == "dummy_filter"

def test_simple_filter_registers_filter_in_environment():
    def dummy_filter(x):
        return x
    extension_type = simple_filter(dummy_filter)
    from jinja2 import Environment
    env = Environment(extensions=[extension_type])
    assert "dummy_filter" in env.filters
    assert env.filters["dummy_filter"] is dummy_filter

def test_simple_filter_extension_is_subclass_of_extension():
    def dummy_filter(x):
        return x
    extension_type = simple_filter(dummy_filter)
    from jinja2 import Extension
    assert issubclass(extension_type, Extension)

def test_simple_filter_preserves_filter_functionality():
    def double(x):
        return x * 2
    extension_type = simple_filter(double)
    from jinja2 import Environment
    env = Environment(extensions=[extension_type])
    result = env.from_string("{{ 5|double }}").render()
    assert result == "10"

def test_simple_filter_with_different_function_names():
    def my_custom_filter(x):
        return x.upper()
    extension_type = simple_filter(my_custom_filter)
    assert extension_type.__name__ == "my_custom_filter"
    from jinja2 import Environment
    env = Environment(extensions=[extension_type])
    assert "my_custom_filter" in env.filters
    assert env.filters["my_custom_filter"] is my_custom_filter


# LLM-generated content at query #4
#--------------------------

def test_simple_filter_creates_extension_with_correct_name():
    def dummy_filter(x):
        return x
    extension_type = simple_filter(dummy_filter)
    assert extension_type.__name__ == "dummy_filter"

def test_simple_filter_registers_filter_in_environment():
    def test_filter(x):
        return x.upper()
    extension_type = simple_filter(test_filter)
    env = Environment()
    extension = extension_type(env)
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

def test_simple_filter_extension_is_subclass_of_extension():
    def some_filter(x):
        return x
    extension_type = simple_filter(some_filter)
    assert issubclass(extension_type, Extension)

def test_simple_filter_works_with_multiple_filters():
    def filter_a(x):
        return x + "a"
    def filter_b(x):
        return x + "b"
    extension_type_a = simple_filter(filter_a)
    extension_type_b = simple_filter(filter_b)
    env = Environment()
    extension_a = extension_type_a(env)
    extension_b = extension_type_b(env)
    assert env.filters["filter_a"] is filter_a
    assert env.filters["filter_b"] is filter_b

def test_simple_filter_preserves_filter_functionality():
    def double(x):
        return x * 2
    extension_type = simple_filter(double)
    env = Environment()
    extension = extension_type(env)
    result = env.filters["double"](5)
    assert result == 10


# LLM-generated content at query #5
#--------------------------

def test_simple_filter_decorator():
    from jinja2 import Environment
    def test_filter(value):
        return value.upper()
    extension_class = simple_filter(test_filter)
    env = Environment()
    ext_instance = extension_class(env)
    result = env.filters['test_filter']('hello')
    assert result == 'HELLO'
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter
    assert extension_class.__name__ == 'test_filter'


# LLM-generated content at query #6
#--------------------------

def test_simple_filter_creates_extension_with_correct_name():
    def my_test_filter(x):
        return x
    extension_class = simple_filter(my_test_filter)
    result = extension_class.__name__
    expected = 'my_test_filter'
    assert result == expected

def test_simple_filter_registers_filter_in_environment():
    def my_test_filter(x):
        return x
    extension_class = simple_filter(my_test_filter)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_class(env)
    result = 'my_test_filter' in env.filters
    assert result == True
    result = env.filters['my_test_filter'] is my_test_filter
    assert result == True

def test_simple_filter_registered_filter_is_callable():
    def my_test_filter(x):
        return x.upper()
    extension_class = simple_filter(my_test_filter)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_class(env)
    result = env.filters['my_test_filter']('hello')
    expected = 'HELLO'
    assert result == expected


# LLM-generated content at query #7
#--------------------------

def test_simple_filter_creates_extension_with_correct_name():
    def my_test_filter(x):
        return x
    extension_class = simple_filter(my_test_filter)
    result = extension_class.__name__
    expected = "my_test_filter"
    assert result == expected

def test_simple_filter_registers_filter_in_environment():
    def my_test_filter(x):
        return x
    extension_class = simple_filter(my_test_filter)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_class(env)
    result = "my_test_filter" in env.filters
    expected = True
    assert result == expected

def test_simple_filter_registered_function_works():
    def my_test_filter(x):
        return x.upper()
    extension_class = simple_filter(my_test_filter)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_class(env)
    filter_func = env.filters["my_test_filter"]
    result = filter_func("hello")
    expected = "HELLO"
    assert result == expected


# LLM-generated content at query #8
#--------------------------

def test_simple_filter_creates_extension_with_correct_name():
    def dummy_filter(x):
        return x
    extension_type = simple_filter(dummy_filter)
    assert extension_type.__name__ == 'dummy_filter'

def test_simple_filter_registers_filter_in_environment():
    def dummy_filter(x):
        return x
    extension_type = simple_filter(dummy_filter)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_type(env)
    assert 'dummy_filter' in env.filters
    assert env.filters['dummy_filter'] is dummy_filter

def test_simple_filter_extension_is_subclass_of_extension():
    def dummy_filter(x):
        return x
    extension_type = simple_filter(dummy_filter)
    from jinja2 import Extension
    assert issubclass(extension_type, Extension)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
    result = env.from_string("{{ 'input' | custom_filter }}").render()
    assert result == "input processed"
def test_simple_filter_returns_extension_subclass():
    from jinja2.ext import Extension
    def dummy_filter(value):
        return value
    extension_class = simple_filter(dummy_filter)
    assert issubclass(extension_class, Extension)
def test_simple_filter_extension_name_matches_function():
    def unique_filter_name(value):
        return value
    extension_class = simple_filter(unique_filter_name)
    assert extension_class.__name__ == "unique_filter_name"


# LLM-generated content at query #2
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
    assert result is my_filter

def test_simple_filter_extension_works_in_template():
    def my_filter(x):
        return x.upper()
    extension_class = simple_filter(my_filter)
    from jinja2 import Environment
    env = Environment(extensions=[extension_class])
    template = env.from_string("{{ 'hello' | my_filter }}")
    result = template.render()
    expected = "HELLO"
    assert result == expected


# LLM-generated content at query #3
#--------------------------

def test_simple_filter_registers_filter():
    from jinja2 import Environment
    def dummy_filter(value):
        return value.upper()
    extension_class = simple_filter(dummy_filter)
    env = Environment(extensions=[extension_class])
    assert 'dummy_filter' in env.filters
    assert env.filters['dummy_filter'] is dummy_filter

def test_simple_filter_extension_name():
    def my_filter(value):
        return value
    extension_class = simple_filter(my_filter)
    assert extension_class.__name__ == 'my_filter'

def test_simple_filter_works_in_template():
    from jinja2 import Environment
    def double(value):
        return value * 2
    extension_class = simple_filter(double)
    env = Environment(extensions=[extension_class])
    template = env.from_string('{{ 5|double }}')
    result = template.render()
    assert result == '10'


# LLM-generated content at query #4
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
    from jinja2 import Environment
    extension_class = simple_filter(my_filter)
    env = Environment(extensions=[extension_class])
    result = env.filters.get("my_filter")
    assert result is my_filter

def test_simple_filter_extension_applies_filter():
    def my_filter(x):
        return x.upper()
    from jinja2 import Environment
    extension_class = simple_filter(my_filter)
    env = Environment(extensions=[extension_class])
    template = env.from_string("{{ 'hello' | my_filter }}")
    result = template.render()
    expected = "HELLO"
    assert result == expected

def test_simple_filter_with_different_function_name():
    def another_filter(x):
        return x + " processed"
    from jinja2 import Environment
    extension_class = simple_filter(another_filter)
    result = extension_class.__name__
    expected = "another_filter"
    assert result == expected
    env = Environment(extensions=[extension_class])
    template = env.from_string("{{ 'test' | another_filter }}")
    rendered = template.render()
    expected_rendered = "test processed"
    assert rendered == expected_rendered


# LLM-generated content at query #5
#--------------------------

def test_simple_filter_decorator():
    from jinja2 import Environment
    def test_filter(value):
        return value.upper()
    ExtensionClass = simple_filter(test_filter)
    env = Environment()
    ext_instance = ExtensionClass(env)
    result = env.filters['test_filter']('hello')
    assert result == 'HELLO'
    assert 'test_filter' in env.filters
    assert env.filters['test_filter'] is test_filter


# LLM-generated content at query #6
#--------------------------

def test_simple_filter_creates_extension_with_correct_name():
    def dummy_filter(x):
        return x
    extension_class = simple_filter(dummy_filter)
    assert extension_class.__name__ == "dummy_filter"

def test_simple_filter_registers_filter_in_environment():
    def test_filter(x):
        return x * 2
    extension_class = simple_filter(test_filter)
    env = Environment()
    extension_instance = extension_class(env)
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter

def test_simple_filter_extension_works_with_jinja2():
    def uppercase_filter(s):
        return s.upper()
    extension_class = simple_filter(uppercase_filter)
    env = Environment(extensions=[extension_class])
    template = env.from_string("{{ 'hello' | uppercase_filter }}")
    result = template.render()
    assert result == "HELLO"

def test_simple_filter_preserves_original_function():
    def original_func(value):
        return value + 1
    extension_class = simple_filter(original_func)
    env = Environment()
    extension_instance = extension_class(env)
    assert env.filters["original_func"](5) == 6


# LLM-generated content at query #7
#--------------------------

def test_simple_filter_registers_filter():
    from jinja2 import Environment
    def test_filter(value):
        return value.upper()
    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    result = env.filters.get('test_filter')
    assert result is test_filter

def test_simple_filter_extension_name():
    def my_filter(value):
        return value
    extension_class = simple_filter(my_filter)
    assert extension_class.__name__ == 'my_filter'

def test_simple_filter_works_in_template():
    from jinja2 import Environment
    def double(value):
        return value * 2
    extension_class = simple_filter(double)
    env = Environment(extensions=[extension_class])
    template = env.from_string('{{ 5|double }}')
    result = template.render()
    assert result == '10'

def test_simple_filter_with_string():
    from jinja2 import Environment
    def repeat(value, times=2):
        return value * times
    extension_class = simple_filter(repeat)
    env = Environment(extensions=[extension_class])
    template = env.from_string('{{ "ab"|repeat(3) }}')
    result = template.render()
    assert result == 'ababab'


# LLM-generated content at query #8
#--------------------------

def test_simple_filter_creates_extension_with_correct_name():
    def my_test_filter(x):
        return x
    extension_class = simple_filter(my_test_filter)
    result = extension_class.__name__
    expected = "my_test_filter"
    assert result == expected

def test_simple_filter_registers_filter_in_environment():
    def my_test_filter(x):
        return x
    extension_class = simple_filter(my_test_filter)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_class(env)
    result = "my_test_filter" in env.filters
    expected = True
    assert result == expected

def test_simple_filter_registered_function_works():
    def my_test_filter(x):
        return x.upper()
    extension_class = simple_filter(my_test_filter)
    from jinja2 import Environment
    env = Environment()
    extension_instance = extension_class(env)
    filter_func = env.filters["my_test_filter"]
    result = filter_func("hello")
    expected = "HELLO"
    assert result == expected

def test_simple_filter_extension_is_subclass_of_extension():
    def my_test_filter(x):
        return x
    extension_class = simple_filter(my_test_filter)
    from jinja2.ext import Extension
    result = issubclass(extension_class, Extension)
    expected = True
    assert result == expected


