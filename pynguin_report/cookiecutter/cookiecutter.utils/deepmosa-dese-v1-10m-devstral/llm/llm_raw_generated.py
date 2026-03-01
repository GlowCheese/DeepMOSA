####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    def test_filter(value):
        return value

    extension_class = simple_filter(test_filter)

    assert extension_class.__name__ == "test_filter"
    assert issubclass(extension_class, Extension)

def test_simple_filter_extension_registers_filter():
    def test_filter(value):
        return value

    extension_class = simple_filter(test_filter)
    environment = Environment(extensions=[extension_class])

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

def test_simple_filter_extension_adds_filter_to_environment():
    def test_filter(x):
        return x

    extension = simple_filter(test_filter)
    env = Environment()
    extension(env)
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter


# LLM-generated content at query #3
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


# LLM-generated content at query #4
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

def test_simple_filter_extension_inherits_from_base():
    def sample_filter(x):
        return x.upper()

    extension_class = simple_filter(sample_filter)
    assert issubclass(extension_class, Extension)

def test_simple_filter_extension_initializes_correctly():
    def example_filter(x):
        return str(x)

    extension_class = simple_filter(example_filter)
    env = Environment()
    extension_instance = extension_class(env)
    assert isinstance(extension_instance, Extension)


# LLM-generated content at query #5
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

    result = simple_filter(my_custom_filter)
    assert result.__name__ == "my_custom_filter"

def test_simple_filter_adds_filter_to_environment():
    def test_filter(x):
        return x.upper()

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](123) == 123


# LLM-generated content at query #6
#--------------------------

```python
def test_simple_filter_returns_extension_type():
    def dummy_filter(x):
        return x

    result = simple_filter(dummy_filter)
    assert isinstance(result, type)
    assert issubclass(result, Extension)


# LLM-generated content at query #7
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    def dummy_filter(x):
        return x

    extension_class = simple_filter(dummy_filter)
    assert isinstance(extension_class, type)
    assert issubclass(extension_class, Extension)

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


# LLM-generated content at query #8
#--------------------------

```python
def test_simple_filter_returns_extension_class():
    def dummy_filter(x):
        return x

    extension_class = simple_filter(dummy_filter)
    assert isinstance(extension_class, type)
    assert issubclass(extension_class, Extension)

def test_simple_filter_extension_name_matches_filter_name():
    def dummy_filter(x):
        return x

    extension_class = simple_filter(dummy_filter)
    assert extension_class.__name__ == "dummy_filter"

def test_simple_filter_adds_filter_to_environment():
    def dummy_filter(x):
        return x

    extension_class = simple_filter(dummy_filter)
    env = Environment(extensions=[extension_class])
    assert "dummy_filter" in env.filters
    assert env.filters["dummy_filter"] is dummy_filter


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_simple_filter_creates_extension_with_correct_name():
    def test_filter(x):
        return x

    extension = simple_filter(test_filter)
    assert extension.__name__ == "test_filter"

def test_simple_filter_extension_adds_filter_to_environment():
    def test_filter(x):
        return x

    extension = simple_filter(test_filter)
    env = Environment()
    extension(env)
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter


# LLM-generated content at query #2
#--------------------------

```python
def test_simple_filter_creates_extension_with_correct_name():
    def test_filter(x):
        return x * 2

    extension = simple_filter(test_filter)
    assert extension.__name__ == "test_filter"

def test_simple_filter_extension_adds_filter_to_environment():
    def test_filter(x):
        return x * 2

    extension = simple_filter(test_filter)
    env = Environment()
    extension(env)
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](5) == 10


# LLM-generated content at query #3
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    def test_filter(x):
        return x * 2

    extension_class = simple_filter(test_filter)
    assert isinstance(extension_class, type)
    assert issubclass(extension_class, Extension)
    assert extension_class.__name__ == "test_filter"

def test_simple_filter_extension_adds_filter_to_environment():
    def test_filter(x):
        return x * 2

    extension_class = simple_filter(test_filter)
    environment = Environment(extensions=[extension_class])
    assert "test_filter" in environment.filters
    assert environment.filters["test_filter"](5) == 10


# LLM-generated content at query #4
#--------------------------

```python
def test_simple_filter_creates_extension_with_correct_name():
    def test_filter(x): return x
    extension_class = simple_filter(test_filter)
    assert extension_class.__name__ == "test_filter"

def test_simple_filter_extension_adds_filter_to_environment():
    def test_filter(x): return x
    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](5) == 5


# LLM-generated content at query #5
#--------------------------

```python
def test_simple_filter_returns_extension_type():
    def dummy_filter(value):
        return value

    result = simple_filter(dummy_filter)
    assert isinstance(result, type)
    assert issubclass(result, Extension)

def test_simple_filter_extension_name_matches_filter_name():
    def my_custom_filter(value):
        return value

    extension_class = simple_filter(my_custom_filter)
    assert extension_class.__name__ == "my_custom_filter"

def test_simple_filter_adds_filter_to_environment():
    def test_filter(value):
        return f"filtered_{value}"

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"]("input") == "filtered_input"


# LLM-generated content at query #6
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
    assert env.filters["test_filter"](123) == 123


# LLM-generated content at query #7
#--------------------------

```python
def test_simple_filter_creates_extension_with_correct_name():
    def test_filter():
        pass
    extension = simple_filter(test_filter)
    assert extension.__name__ == "test_filter"

def test_simple_filter_extension_adds_filter_to_environment():
    def test_filter(x):
        return x
    extension = simple_filter(test_filter)
    env = Environment()
    ext = extension(env)
    assert "test_filter" in env.filters
    assert env.filters["test_filter"] is test_filter


# LLM-generated content at query #8
#--------------------------

```python
def test_simple_filter_creates_extension_with_correct_name():
    def dummy_filter(x):
        return x

    extension_class = simple_filter(dummy_filter)
    assert extension_class.__name__ == "dummy_filter"

def test_simple_filter_extension_registers_filter():
    def uppercase_filter(text):
        return text.upper()

    extension_class = simple_filter(uppercase_filter)
    env = Environment(extensions=[extension_class])
    assert "uppercase_filter" in env.filters
    assert env.filters["uppercase_filter"]("hello") == "HELLO"

def test_simple_filter_extension_inherits_from_extension():
    def dummy_filter(x):
        return x

    extension_class = simple_filter(dummy_filter)
    assert issubclass(extension_class, Extension)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    def example_filter(value):
        return value.upper()

    extension_class = simple_filter(example_filter)

    assert isinstance(extension_class, type)
    assert issubclass(extension_class, Extension)
    assert extension_class.__name__ == "example_filter"

def test_simple_filter_extension_adds_filter_to_environment():
    def example_filter(value):
        return value.upper()

    extension_class = simple_filter(example_filter)
    environment = Environment(extensions=[extension_class])

    assert "example_filter" in environment.filters
    assert environment.filters["example_filter"] is example_filter


# LLM-generated content at query #2
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


# LLM-generated content at query #3
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
        return x * 2

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](5) == 10


# LLM-generated content at query #4
#--------------------------

```python
def test_simple_filter_creates_extension_with_correct_name():
    def test_filter(x):
        return x * 2

    extension_class = simple_filter(test_filter)
    assert extension_class.__name__ == "test_filter"

def test_simple_filter_extension_adds_filter_to_environment():
    def test_filter(x):
        return x * 2

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](5) == 10


# LLM-generated content at query #5
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
        return x.upper()

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](123) == "123"


# LLM-generated content at query #6
#--------------------------

```python
def test_simple_filter_returns_extension_class():
    def dummy_filter(x):
        return x

    result = simple_filter(dummy_filter)
    assert isinstance(result, type)
    assert issubclass(result, Extension)
    assert result.__name__ == dummy_filter.__name__

def test_simple_filter_extension_adds_filter_to_environment():
    def test_filter(x):
        return x.upper()

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert test_filter.__name__ in env.filters
    assert env.filters[test_filter.__name__] is test_filter


# LLM-generated content at query #7
#--------------------------

```python
def test_simple_filter_creates_extension_class():
    def dummy_filter(x):
        return x

    extension_class = simple_filter(dummy_filter)
    assert extension_class.__name__ == "dummy_filter"
    assert issubclass(extension_class, Extension)

def test_simple_filter_extension_initialization():
    def dummy_filter(x):
        return x

    extension_class = simple_filter(dummy_filter)
    env = Environment()
    extension = extension_class(env)
    assert env.filters["dummy_filter"] is dummy_filter


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
    def my_filter(x):
        return x

    extension_class = simple_filter(my_filter)
    assert extension_class.__name__ == "my_filter"

def test_simple_filter_adds_filter_to_environment():
    def test_filter(x):
        return x * 2

    extension_class = simple_filter(test_filter)
    env = Environment(extensions=[extension_class])
    assert "test_filter" in env.filters
    assert env.filters["test_filter"](5) == 10


