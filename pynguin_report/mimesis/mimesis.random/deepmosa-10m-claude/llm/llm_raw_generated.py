####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_generate_string_by_mask_default_mask():
    from random import Random
    rng = Random(42)
    result = rng.generate_string_by_mask()
    assert isinstance(result, str)
    assert len(result) == 4
    assert result[0].isupper()
    assert result[1].isdigit()
    assert result[2].isdigit()
    assert result[3].isdigit()


def test_generate_string_by_mask_custom_mask():
    from random import Random
    rng = Random(42)
    result = rng.generate_string_by_mask(mask="@@##")
    assert isinstance(result, str)
    assert len(result) == 4
    assert result[0].isupper()
    assert result[1].isupper()
    assert result[2].isdigit()
    assert result[3].isdigit()


def test_generate_string_by_mask_with_static_chars():
    from random import Random
    rng = Random(42)
    result = rng.generate_string_by_mask(mask="ABC@#XYZ")
    assert isinstance(result, str)
    assert len(result) == 8
    assert result[0] == 'A'
    assert result[1] == 'B'
    assert result[2] == 'C'
    assert result[3].isupper()
    assert result[4].isdigit()
    assert result[5] == 'X'
    assert result[6] == 'Y'
    assert result[7] == 'Z'


def test_generate_string_by_mask_custom_placeholders():
    from random import Random
    rng = Random(42)
    result = rng.generate_string_by_mask(mask="xxyy", char="x", digit="y")
    assert isinstance(result, str)
    assert len(result) == 4
    assert result[0].isupper()
    assert result[1].isupper()
    assert result[2].isdigit()
    assert result[3].isdigit()


def test_generate_string_by_mask_same_placeholder_raises_error():
    from random import Random
    rng = Random(42)
    try:
        rng.generate_string_by_mask(char="@", digit="@")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "same placeholder" in str(e).lower()


def test_generate_string_by_mask_empty_mask():
    from random import Random
    rng = Random(42)
    result = rng.generate_string_by_mask(mask="")
    assert result == ""


def test_generate_string_by_mask_only_static_chars():
    from random import Random
    rng = Random(42)
    result = rng.generate_string_by_mask(mask="HELLO")
    assert result == "HELLO"


def test_generate_string_by_mask_long_mask():
    from random import Random
    rng = Random(42)
    result = rng.generate_string_by_mask(mask="@" * 10 + "#" * 10)
    assert isinstance(result, str)
    assert len(result) == 20
    for i in range(10):
        assert result[i].isupper()
    for i in range(10, 20):
        assert result[i].isdigit()


# LLM-generated content at query #2
#--------------------------

```python
def test_random_constructor():
    from random import Random
    random_instance = Random()
    assert isinstance(random_instance, Random)
    assert hasattr(random_instance, 'randints')
    assert hasattr(random_instance, '_generate_string')
    assert hasattr(random_instance, 'generate_string_by_mask')
    assert hasattr(random_instance, 'uniform')
    assert hasattr(random_instance, 'randbytes')
    assert hasattr(random_instance, 'weighted_choice')
    assert hasattr(random_instance, 'choice_enum_item')


# LLM-generated content at query #3
#--------------------------

```python
def test_random_constructor():
    from random import Random
    rand = Random()
    assert isinstance(rand, Random)
    assert hasattr(rand, 'randints')
    assert hasattr(rand, '_generate_string')
    assert hasattr(rand, 'generate_string_by_mask')
    assert hasattr(rand, 'uniform')
    assert hasattr(rand, 'randbytes')
    assert hasattr(rand, 'weighted_choice')
    assert hasattr(rand, 'choice_enum_item')
    assert hasattr(rand, 'random')
    assert hasattr(rand, 'choices')
    assert hasattr(rand, 'choice')
    assert hasattr(rand, 'getrandbits')


# LLM-generated content at query #4
#--------------------------

```python
def test_random_constructor():
    from random import Random
    random_instance = Random()
    assert isinstance(random_instance, Random)
    assert hasattr(random_instance, 'randints')
    assert hasattr(random_instance, '_generate_string')
    assert hasattr(random_instance, 'generate_string_by_mask')
    assert hasattr(random_instance, 'uniform')
    assert hasattr(random_instance, 'randbytes')
    assert hasattr(random_instance, 'weighted_choice')
    assert hasattr(random_instance, 'choice_enum_item')


# LLM-generated content at query #5
#--------------------------

```python
def test_random_constructor():
    from random import Random
    random_instance = Random()
    assert isinstance(random_instance, Random)
    assert hasattr(random_instance, 'randints')
    assert hasattr(random_instance, '_generate_string')
    assert hasattr(random_instance, 'generate_string_by_mask')
    assert hasattr(random_instance, 'uniform')
    assert hasattr(random_instance, 'randbytes')
    assert hasattr(random_instance, 'weighted_choice')
    assert hasattr(random_instance, 'choice_enum_item')


# LLM-generated content at query #6
#--------------------------

```python
def test_random_constructor():
    from random import Random
    random_instance = Random()
    assert isinstance(random_instance, Random)
    assert hasattr(random_instance, 'randints')
    assert hasattr(random_instance, '_generate_string')
    assert hasattr(random_instance, 'generate_string_by_mask')
    assert hasattr(random_instance, 'uniform')
    assert hasattr(random_instance, 'randbytes')
    assert hasattr(random_instance, 'weighted_choice')
    assert hasattr(random_instance, 'choice_enum_item')


# LLM-generated content at query #7
#--------------------------

```python
def test_random_constructor():
    from random import Random
    random_instance = Random()
    assert isinstance(random_instance, Random)
    assert hasattr(random_instance, 'randints')
    assert hasattr(random_instance, '_generate_string')
    assert hasattr(random_instance, 'generate_string_by_mask')
    assert hasattr(random_instance, 'uniform')
    assert hasattr(random_instance, 'randbytes')
    assert hasattr(random_instance, 'weighted_choice')
    assert hasattr(random_instance, 'choice_enum_item')


# LLM-generated content at query #8
#--------------------------

```python
def test_random_constructor():
    from random import Random
    rand = Random()
    assert isinstance(rand, Random)
    assert hasattr(rand, 'randints')
    assert hasattr(rand, '_generate_string')
    assert hasattr(rand, 'generate_string_by_mask')
    assert hasattr(rand, 'uniform')
    assert hasattr(rand, 'randbytes')
    assert hasattr(rand, 'weighted_choice')
    assert hasattr(rand, 'choice_enum_item')


# LLM-generated content at query #9
#--------------------------

```python
def test_random_constructor():
    from random import Random
    random_instance = Random()
    assert isinstance(random_instance, Random)
    assert hasattr(random_instance, 'randints')
    assert hasattr(random_instance, '_generate_string')
    assert hasattr(random_instance, 'generate_string_by_mask')
    assert hasattr(random_instance, 'uniform')
    assert hasattr(random_instance, 'randbytes')
    assert hasattr(random_instance, 'weighted_choice')
    assert hasattr(random_instance, 'choice_enum_item')


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_random_constructor():
    """Test that Random class can be instantiated."""
    from random import Random
    random_instance = Random()
    assert isinstance(random_instance, Random)
    assert hasattr(random_instance, 'randints')
    assert hasattr(random_instance, '_generate_string')
    assert hasattr(random_instance, 'generate_string_by_mask')
    assert hasattr(random_instance, 'uniform')
    assert hasattr(random_instance, 'randbytes')
    assert hasattr(random_instance, 'weighted_choice')
    assert hasattr(random_instance, 'choice_enum_item')


# LLM-generated content at query #2
#--------------------------

```python
def test_generate_string_by_mask_default_mask():
    from random import Random
    rand = Random()
    result = rand.generate_string_by_mask()
    assert len(result) == 4
    assert result[0].isupper()
    assert result[1].isdigit()
    assert result[2].isdigit()
    assert result[3].isdigit()


def test_generate_string_by_mask_custom_mask():
    from random import Random
    rand = Random()
    result = rand.generate_string_by_mask(mask="@@##")
    assert len(result) == 4
    assert result[0].isupper()
    assert result[1].isupper()
    assert result[2].isdigit()
    assert result[3].isdigit()


def test_generate_string_by_mask_all_chars():
    from random import Random
    rand = Random()
    result = rand.generate_string_by_mask(mask="@@@@")
    assert len(result) == 4
    assert all(c.isupper() for c in result)


def test_generate_string_by_mask_all_digits():
    from random import Random
    rand = Random()
    result = rand.generate_string_by_mask(mask="####")
    assert len(result) == 4
    assert all(c.isdigit() for c in result)


def test_generate_string_by_mask_with_static_chars():
    from random import Random
    rand = Random()
    result = rand.generate_string_by_mask(mask="@-#-@")
    assert len(result) == 5
    assert result[0].isupper()
    assert result[1] == "-"
    assert result[2].isdigit()
    assert result[3] == "-"
    assert result[4].isupper()


def test_generate_string_by_mask_custom_placeholders():
    from random import Random
    rand = Random()
    result = rand.generate_string_by_mask(mask="X*X*", char="X", digit="*")
    assert len(result) == 4
    assert result[0].isupper()
    assert result[1].isdigit()
    assert result[2].isupper()
    assert result[3].isdigit()


def test_generate_string_by_mask_empty_mask():
    from random import Random
    rand = Random()
    result = rand.generate_string_by_mask(mask="")
    assert result == ""


def test_generate_string_by_mask_only_static_chars():
    from random import Random
    rand = Random()
    result = rand.generate_string_by_mask(mask="ABC-XYZ")
    assert result == "ABC-XYZ"


def test_generate_string_by_mask_same_placeholder_raises_error():
    from random import Random
    rand = Random()
    try:
        rand.generate_string_by_mask(mask="@#@#", char="@", digit="@")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "same placeholder" in str(e).lower()


def test_generate_string_by_mask_long_mask():
    from random import Random
    rand = Random()
    result = rand.generate_string_by_mask(mask="@#@#@#@#@#")
    assert len(result) == 10
    for i in range(10):
        if i % 2 == 0:
            assert result[i].isupper()
        else:
            assert result[i].isdigit()


# LLM-generated content at query #3
#--------------------------

```python
def test_random_constructor():
    """Test that Random class can be instantiated."""
    from random import Random as StdRandom
    random_instance = Random()
    assert isinstance(random_instance, Random)
    assert isinstance(random_instance, StdRandom)


# LLM-generated content at query #4
#--------------------------

```python
def test_random_constructor():
    from random import Random
    random_instance = Random()
    assert isinstance(random_instance, Random)
    assert hasattr(random_instance, 'randints')
    assert hasattr(random_instance, '_generate_string')
    assert hasattr(random_instance, 'generate_string_by_mask')
    assert hasattr(random_instance, 'uniform')
    assert hasattr(random_instance, 'randbytes')
    assert hasattr(random_instance, 'weighted_choice')
    assert hasattr(random_instance, 'choice_enum_item')


# LLM-generated content at query #5
#--------------------------

```python
def test_random_constructor():
    from random import Random
    random_instance = Random()
    assert isinstance(random_instance, Random)
    assert hasattr(random_instance, 'randints')
    assert hasattr(random_instance, '_generate_string')
    assert hasattr(random_instance, 'generate_string_by_mask')
    assert hasattr(random_instance, 'uniform')
    assert hasattr(random_instance, 'randbytes')
    assert hasattr(random_instance, 'weighted_choice')
    assert hasattr(random_instance, 'choice_enum_item')


# LLM-generated content at query #6
#--------------------------

```python
def test_random_constructor():
    from random import Random
    rand_instance = Random()
    assert isinstance(rand_instance, Random)
    assert hasattr(rand_instance, 'randints')
    assert hasattr(rand_instance, '_generate_string')
    assert hasattr(rand_instance, 'generate_string_by_mask')
    assert hasattr(rand_instance, 'uniform')
    assert hasattr(rand_instance, 'randbytes')
    assert hasattr(rand_instance, 'weighted_choice')
    assert hasattr(rand_instance, 'choice_enum_item')


# LLM-generated content at query #7
#--------------------------

```python
def test_random_constructor():
    from random import Random
    random_instance = Random()
    assert isinstance(random_instance, Random)
    assert hasattr(random_instance, 'randints')
    assert hasattr(random_instance, '_generate_string')
    assert hasattr(random_instance, 'generate_string_by_mask')
    assert hasattr(random_instance, 'uniform')
    assert hasattr(random_instance, 'randbytes')
    assert hasattr(random_instance, 'weighted_choice')
    assert hasattr(random_instance, 'choice_enum_item')


# LLM-generated content at query #8
#--------------------------

```python
def test_random_constructor():
    from random import Random as StdRandom
    
    random_instance = Random()
    
    assert isinstance(random_instance, Random)
    assert isinstance(random_instance, StdRandom)
    assert hasattr(random_instance, 'random')
    assert hasattr(random_instance, 'randints')
    assert hasattr(random_instance, 'generate_string_by_mask')
    assert hasattr(random_instance, 'uniform')
    assert hasattr(random_instance, 'randbytes')
    assert hasattr(random_instance, 'weighted_choice')
    assert hasattr(random_instance, 'choice_enum_item')


# LLM-generated content at query #9
#--------------------------

```python
def test_random_constructor():
    from random import Random as StdRandom
    rand = Random()
    assert isinstance(rand, StdRandom)
    assert hasattr(rand, 'randints')
    assert hasattr(rand, '_generate_string')
    assert hasattr(rand, 'generate_string_by_mask')
    assert hasattr(rand, 'uniform')
    assert hasattr(rand, 'randbytes')
    assert hasattr(rand, 'weighted_choice')
    assert hasattr(rand, 'choice_enum_item')


