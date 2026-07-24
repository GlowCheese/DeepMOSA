####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_simple_filter_returns_extension_class():
    def dummy_filter():
        pass

    result = simple_filter(dummy_filter)
    assert isinstance(result, type)
    assert issubclass(result, Extension)

def test_simple_filter_extension_name_matches_filter_name():
    def dummy_filter():
        pass

    result = simple_filter(dummy_filter)
    assert result.__name__ == "dummy_filter"

def test_simple_filter_adds_filter_to_environment():
    def test_filter(value):
        return value.upper()

    extension_class = simple_filter(test_filter)
    environment = Environment()
    extension = extension_class(environment)

    assert "test_filter" in environment.filters
    assert environment.filters["test_filter"] is test_filter


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter_creates_extension_with_correct_name():
    def test_filter(x):
        return x

    extension = simple_filter(test_filter)
    assert extension.__name__ == "test_filter"

def test_simple_filter_adds_filter_to_environment():
    def test_filter(x):
        return x.upper()

    extension = simple_filter(test_filter)
    env = Environment(extensions=[extension])
    assert env.filters["test_filter"] is test_filter

def test_simple_filter_preserves_filter_functionality():
    def test_filter(x):
        return x * 2

    extension = simple_filter(test_filter)
    env = Environment(extensions=[extension])
    template = env.from_string("{{ 'hello' | test_filter }}")
    assert template.render() == "hellohello"


# LLM-generated content at query #3
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    def test_filter(x):
        return x * 2

    extension_class = simple_filter(test_filter)
    assert extension_class.__name__ == "test_filter"
    assert issubclass(extension_class, Extension)

def test_simple_filter_extension_registers_filter():
    def test_filter(x):
        return x * 2

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](5) == 10


# LLM-generated content at query #4
#--------------------------

```python
def test_simple_filter_creates_extension_with_correct_name():
    def test_filter(value):
        return value.upper()

    extension_class = simple_filter(test_filter)
    assert extension_class.__name__ == "test_filter"

def test_simple_filter_extension_adds_filter_to_environment():
    def test_filter(value):
        return value.upper()

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]({"value": "hello"}) == "HELLO"


# LLM-generated content at query #5
#--------------------------

```python
def test_simple_filter_returns_extension_class():
    def dummy_filter(x):
        return x

    result = simple_filter(dummy_filter)
    assert isinstance(result, type)
    assert issubclass(result, Extension)

def test_simple_filter_extension_name():
    def dummy_filter(x):
        return x

    result = simple_filter(dummy_filter)
    assert result.__name__ == "dummy_filter"

def test_simple_filter_adds_filter_to_environment():
    def dummy_filter(x):
        return x

    extension_class = simple_filter(dummy_filter)
    env = Environment(extensions=[extension_class])
    assert "dummy_filter" in env.filters
    assert env.filters["dummy_filter"] is dummy_filter


# LLM-generated content at query #6
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    def dummy_filter(x):
        return x

    extension_class = simple_filter(dummy_filter)
    assert extension_class.__name__ == "dummy_filter"
    assert issubclass(extension_class, Extension)

def test_simple_filter_extension_registers_filter():
    def test_filter(x):
        return x * 2

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](5) == 10


# LLM-generated content at query #7
#--------------------------

```python
def test_simple_filter_creates_extension_with_correct_name():
    def dummy_filter(x):
        return x

    extension_class = simple_filter(dummy_filter)
    assert extension_class.__name__ == "dummy_filter"

def test_simple_filter_extension_adds_filter_to_environment():
    def test_filter(x):
        return x * 2

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert env.filters["test_filter"](5) == 10

def test_simple_filter_extension_inherits_from_extension():
    def sample_filter(x):
        return x

    extension_class = simple_filter(sample_filter)
    assert issubclass(extension_class, Extension)


# LLM-generated content at query #8
#--------------------------

```python
def test_simple_filter_returns_extension_type():
    def dummy_filter(x):
        return x
    result = simple_filter(dummy_filter)
    assert isinstance(result, type)
    assert issubclass(result, Extension)

def test_simple_filter_extension_name_matches_filter_name():
    def my_custom_filter(x):
        return x
    extension_class = simple_filter(my_custom_filter)
    assert extension_class.__name__ == "my_custom_filter"

def test_simple_filter_adds_filter_to_environment():
    def test_filter(x):
        return x * 2
    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](5) == 10


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    def dummy_filter(x):
        return x

    extension_class = simple_filter(dummy_filter)

    assert isinstance(extension_class, type)
    assert issubclass(extension_class, Extension)
    assert extension_class.__name__ == "dummy_filter"

def test_simple_filter_extension_registers_filter():
    def test_filter(x):
        return x.upper()

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])

    assert "test_filter" in env.filters
    assert env.filters["test_filter"](123) == "123"


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter_creates_extension_with_correct_name():
    def dummy_filter(x):
        return x

    extension = simple_filter(dummy_filter)
    assert extension.__name__ == "dummy_filter"

def test_simple_filter_adds_filter_to_environment():
    def test_filter(x):
        return x * 2

    environment = Environment()
    extension = simple_filter(test_filter)
    extension_instance = extension(environment)
    assert environment.filters["test_filter"] == test_filter


# LLM-generated content at query #3
#--------------------------

```python
def test_simple_filter_creates_extension_with_correct_name():
    def example_filter(x):
        return x * 2

    extension = simple_filter(example_filter)
    assert extension.__name__ == "example_filter"

def test_simple_filter_adds_filter_to_environment():
    def test_filter(x):
        return x.upper()

    extension = simple_filter(test_filter)
    env = Environment(extensions=[extension])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](123) == 123

def test_simple_filter_preserves_original_function():
    def original_function(x):
        return x + 1

    extension = simple_filter(original_function)
    env = Environment(extensions=[extension])
    assert env.filters["original_function"](5) == 6


# LLM-generated content at query #4
#--------------------------

```python
def test_simple_filter_creates_extension_with_correct_name():
    def test_filter(x):
        return x
    extension = simple_filter(test_filter)
    assert extension.__name__ == "test_filter"

def test_simple_filter_adds_filter_to_environment():
    def test_filter(x):
        return x
    extension = simple_filter(test_filter)
    env = Environment(extensions=[extension])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](5) == 5

def test_simple_filter_preserves_original_function_behavior():
    def upper_filter(text):
        return text.upper()
    extension = simple_filter(upper_filter)
    env = Environment(extensions=[extension])
    assert env.filters["upper_filter"]("hello") == "HELLO"


# LLM-generated content at query #5
#--------------------------

```python
def test_simple_filter_returns_extension_type():
    def dummy_filter(x):
        return x

    result = simple_filter(dummy_filter)
    assert isinstance(result, type)
    assert issubclass(result, Extension)

def test_simple_filter_extension_name_matches_filter_name():
    def dummy_filter(x):
        return x

    result = simple_filter(dummy_filter)
    assert result.__name__ == "dummy_filter"

def test_simple_filter_adds_filter_to_environment():
    def dummy_filter(x):
        return x

    env = Environment()
    extension = simple_filter(dummy_filter)(env)
    assert dummy_filter.__name__ in env.filters
    assert env.filters[dummy_filter.__name__] is dummy_filter


# LLM-generated content at query #6
#--------------------------

```python
def test_simple_filter_returns_extension_class():
    def dummy_filter():
        pass

    result = simple_filter(dummy_filter)
    assert isinstance(result, type)
    assert issubclass(result, Extension)

def test_simple_filter_extension_name_matches_filter_name():
    def test_filter():
        pass

    extension_class = simple_filter(test_filter)
    assert extension_class.__name__ == "test_filter"

def test_simple_filter_adds_filter_to_environment():
    def uppercase_filter(text):
        return text.upper()

    extension_class = simple_filter(uppercase_filter)
    env = Environment(extensions=[extension_class])
    assert "uppercase_filter" in env.filters
    assert env.filters["uppercase_filter"]("hello") == "HELLO"


# LLM-generated content at query #7
#--------------------------

```python
def test_simple_filter_returns_extension_class():
    def dummy_filter(x):
        return x

    result = simple_filter(dummy_filter)
    assert isinstance(result, type)
    assert issubclass(result, Extension)

def test_simple_filter_extension_name_matches_filter_name():
    def my_custom_filter(x):
        return x

    extension_class = simple_filter(my_custom_filter)
    assert extension_class.__name__ == "my_custom_filter"

def test_simple_filter_adds_filter_to_environment():
    def test_filter(x):
        return x.upper()

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](123) == 123


# LLM-generated content at query #8
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    def test_filter(value):
        return value.upper()

    extension_class = simple_filter(test_filter)
    assert extension_class.__name__ == test_filter.__name__
    assert issubclass(extension_class, Extension)

def test_simple_filter_extension_adds_filter_to_environment():
    def test_filter(value):
        return value.upper()

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter


