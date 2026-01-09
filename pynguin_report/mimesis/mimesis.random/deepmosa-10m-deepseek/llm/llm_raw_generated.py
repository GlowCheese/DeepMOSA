####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_generate_string_by_mask_default_mask():
    r = Random()
    result = r.generate_string_by_mask()
    assert len(result) == 4
    assert result[0].isalpha() and result[0].isupper()
    assert all(c.isdigit() for c in result[1:4])


def test_generate_string_by_mask_custom_mask():
    r = Random()
    mask = "@@###-@@"
    result = r.generate_string_by_mask(mask=mask)
    assert len(result) == 8
    assert result[0].isalpha() and result[0].isupper()
    assert result[1].isalpha() and result[1].isupper()
    assert all(c.isdigit() for c in result[2:5])
    assert result[5] == '-'
    assert result[6].isalpha() and result[6].isupper()
    assert result[7].isalpha() and result[7].isupper()


def test_generate_string_by_mask_custom_placeholders():
    r = Random()
    mask = "AA99"
    result = r.generate_string_by_mask(mask=mask, char='A', digit='9')
    assert len(result) == 4
    assert result[0].isalpha() and result[0].isupper()
    assert result[1].isalpha() and result[1].isupper()
    assert result[2].isdigit()
    assert result[3].isdigit()


def test_generate_string_by_mask_same_placeholder_error():
    r = Random()
    try:
        r.generate_string_by_mask(mask="@@##", char='@', digit='@')
        assert False
    except ValueError as e:
        assert "same placeholder" in str(e)


def test_generate_string_by_mask_literal_characters():
    r = Random()
    mask = "CODE-@##"
    result = r.generate_string_by_mask(mask=mask)
    assert result.startswith("CODE-")
    assert result[5].isalpha() and result[5].isupper()
    assert result[6].isdigit()
    assert result[7].isdigit()


def test_generate_string_by_mask_empty_mask():
    r = Random()
    result = r.generate_string_by_mask(mask="")
    assert result == ""


def test_generate_string_by_mask_only_literals():
    r = Random()
    mask = "FIXED"
    result = r.generate_string_by_mask(mask=mask)
    assert result == "FIXED"


def test_generate_string_by_mask_only_placeholders():
    r = Random()
    mask = "@@@@@"
    result = r.generate_string_by_mask(mask=mask)
    assert len(result) == 5
    assert all(c.isalpha() and c.isupper() for c in result)


def test_generate_string_by_mask_mixed_placeholders_and_literals():
    r = Random()
    mask = "X@Y#Z"
    result = r.generate_string_by_mask(mask=mask)
    assert len(result) == 5
    assert result[0] == 'X'
    assert result[1].isalpha() and result[1].isupper()
    assert result[2] == 'Y'
    assert result[3].isdigit()
    assert result[4] == 'Z'


# LLM-generated content at query #2
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_inherits_from_base_random():
    instance = Random()
    from random import Random as BaseRandom
    assert isinstance(instance, BaseRandom)

def test_constructor_has_randints_method():
    instance = Random()
    assert hasattr(instance, 'randints')

def test_constructor_has_generate_string_by_mask_method():
    instance = Random()
    assert hasattr(instance, 'generate_string_by_mask')

def test_constructor_has_uniform_method():
    instance = Random()
    assert hasattr(instance, 'uniform')

def test_constructor_has_randbytes_method():
    instance = Random()
    assert hasattr(instance, 'randbytes')

def test_constructor_has_weighted_choice_method():
    instance = Random()
    assert hasattr(instance, 'weighted_choice')

def test_constructor_has_choice_enum_item_method():
    instance = Random()
    assert hasattr(instance, 'choice_enum_item')

def test_constructor_has_random_method():
    instance = Random()
    assert hasattr(instance, 'random')

def test_constructor_has_choices_method():
    instance = Random()
    assert hasattr(instance, 'choices')

def test_constructor_has_choice_method():
    instance = Random()
    assert hasattr(instance, 'choice')


# LLM-generated content at query #3
#--------------------------

def test_random_constructor_default():
    r = Random()
    assert isinstance(r, Random)

def test_random_constructor_with_seed():
    r1 = Random(42)
    r2 = Random(42)
    assert r1.random() == r2.random()

def test_random_constructor_inherits_from_random():
    r = Random()
    assert isinstance(r, random_module.Random)


# LLM-generated content at query #4
#--------------------------

def test_constructor_creates_instance():
    r = Random()
    assert isinstance(r, Random)

def test_constructor_inherits_from_base_random():
    r = Random()
    assert isinstance(r, random_module.Random)

def test_constructor_seed_produces_reproducible_random():
    r1 = Random(12345)
    r2 = Random(12345)
    val1 = r1.random()
    val2 = r2.random()
    assert val1 == val2

def test_constructor_no_arguments():
    r = Random()
    val = r.random()
    assert isinstance(val, float)
    assert 0.0 <= val < 1.0

def test_constructor_with_int_seed():
    r = Random(42)
    val = r.random()
    assert isinstance(val, float)

def test_constructor_with_float_seed():
    r = Random(3.14)
    val = r.random()
    assert isinstance(val, float)

def test_constructor_with_str_seed():
    r = Random("seed")
    val = r.random()
    assert isinstance(val, float)

def test_constructor_with_bytes_seed():
    r = Random(b"bytes_seed")
    val = r.random()
    assert isinstance(val, float)

def test_constructor_with_bytearray_seed():
    r = Random(bytearray(b"bytearray_seed"))
    val = r.random()
    assert isinstance(val, float)

def test_constructor_with_none_seed():
    r = Random(None)
    val = r.random()
    assert isinstance(val, float)

def test_constructor_instance_has_randints_method():
    r = Random()
    result = r.randints()
    assert isinstance(result, list)
    assert all(isinstance(i, int) for i in result)

def test_constructor_instance_has_generate_string_by_mask_method():
    r = Random()
    result = r.generate_string_by_mask()
    assert isinstance(result, str)

def test_constructor_instance_has_uniform_method():
    r = Random()
    result = r.uniform(1.0, 10.0)
    assert isinstance(result, float)

def test_constructor_instance_has_randbytes_method():
    r = Random()
    result = r.randbytes()
    assert isinstance(result, bytes)

def test_constructor_instance_has_weighted_choice_method():
    r = Random()
    choices = {"a": 0.5, "b": 0.5}
    result = r.weighted_choice(choices)
    assert result in choices

def test_constructor_instance_has_choice_enum_item_method():
    r = Random()
    from enum import Enum
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    result = r.choice_enum_item(Color)
    assert result in [Color.RED, Color.GREEN, Color.BLUE]


# LLM-generated content at query #5
#--------------------------

def test_constructor_without_seed():
    r = Random()
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_with_seed():
    r = Random(seed=42)
    first_random = r.random()
    r2 = Random(seed=42)
    second_random = r2.random()
    assert first_random == second_random

def test_constructor_with_none_seed():
    r = Random(seed=None)
    assert isinstance(r, Random)

def test_constructor_inherits_random_methods():
    r = Random()
    result = r.randint(1, 10)
    assert 1 <= result <= 10

def test_constructor_state_initialization():
    r1 = Random(seed=123)
    state = r1.getstate()
    r2 = Random()
    r2.setstate(state)
    assert r1.random() == r2.random()


# LLM-generated content at query #6
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_accepts_seed():
    seed = 42
    instance = Random(seed)
    assert isinstance(instance, Random)

def test_constructor_without_seed():
    instance = Random()
    assert isinstance(instance, Random)


# LLM-generated content at query #7
#--------------------------

def test_constructor_without_seed():
    r = Random()
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_with_seed():
    r = Random(seed=42)
    first_random = r.random()
    r2 = Random(seed=42)
    second_random = r2.random()
    assert first_random == second_random

def test_constructor_with_none_seed():
    r = Random(seed=None)
    assert isinstance(r, Random)

def test_constructor_inherits_random_methods():
    r = Random()
    result = r.randint(1, 10)
    assert 1 <= result <= 10


# LLM-generated content at query #8
#--------------------------

def test_constructor_initializes_without_seed():
    r = Random()
    assert isinstance(r, Random)

def test_constructor_initializes_with_seed():
    r = Random(seed=42)
    assert isinstance(r, Random)

def test_constructor_initializes_with_none_seed():
    r = Random(seed=None)
    assert isinstance(r, Random)

def test_constructor_initializes_with_int_seed():
    r = Random(seed=12345)
    assert isinstance(r, Random)

def test_constructor_initializes_with_float_seed():
    r = Random(seed=3.14159)
    assert isinstance(r, Random)

def test_constructor_initializes_with_str_seed():
    r = Random(seed="test_seed")
    assert isinstance(r, Random)

def test_constructor_initializes_with_bytes_seed():
    r = Random(seed=b"bytes_seed")
    assert isinstance(r, Random)

def test_constructor_initializes_with_bytearray_seed():
    r = Random(seed=bytearray(b"bytearray_seed"))
    assert isinstance(r, Random)

def test_constructor_initializes_with_memoryview_seed():
    mv = memoryview(b"memoryview_seed")
    r = Random(seed=mv)
    assert isinstance(r, Random)

def test_constructor_initializes_with_version_parameter():
    r = Random(version=2)
    assert isinstance(r, Random)


# LLM-generated content at query #9
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_inherits_from_base_random():
    instance = Random()
    assert isinstance(instance, random_module.Random)

def test_constructor_supports_seed():
    instance1 = Random(12345)
    instance2 = Random(12345)
    val1 = instance1.random()
    val2 = instance2.random()
    assert val1 == val2

def test_constructor_without_seed_produces_different_instances():
    instance1 = Random()
    instance2 = Random()
    val1 = instance1.random()
    val2 = instance2.random()
    assert val1 != val2

def test_constructor_accepts_none_seed():
    instance = Random(None)
    val = instance.random()
    assert isinstance(val, float)

def test_constructor_accepts_int_seed():
    instance = Random(42)
    val = instance.random()
    assert isinstance(val, float)

def test_constructor_accepts_str_seed():
    instance = Random("seed")
    val = instance.random()
    assert isinstance(val, float)

def test_constructor_accepts_bytes_seed():
    instance = Random(b"seed")
    val = instance.random()
    assert isinstance(val, float)

def test_constructor_accepts_bytearray_seed():
    instance = Random(bytearray(b"seed"))
    val = instance.random()
    assert isinstance(val, float)

def test_constructor_has_expected_methods():
    instance = Random()
    assert hasattr(instance, "randints")
    assert hasattr(instance, "generate_string_by_mask")
    assert hasattr(instance, "uniform")
    assert hasattr(instance, "randbytes")
    assert hasattr(instance, "weighted_choice")
    assert hasattr(instance, "choice_enum_item")


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_inherits_from_base_random():
    instance = Random()
    assert isinstance(instance, random_module.Random)

def test_constructor_default_seed_reproducibility():
    instance1 = Random()
    instance2 = Random()
    first_values1 = [instance1.random() for _ in range(5)]
    first_values2 = [instance2.random() for _ in range(5)]
    assert first_values1 != first_values2

def test_constructor_with_same_seed_reproducibility():
    seed = 42
    instance1 = Random(seed)
    instance2 = Random(seed)
    values1 = [instance1.random() for _ in range(10)]
    values2 = [instance2.random() for _ in range(10)]
    assert values1 == values2

def test_constructor_with_different_seed_non_reproducibility():
    instance1 = Random(123)
    instance2 = Random(456)
    values1 = [instance1.random() for _ in range(10)]
    values2 = [instance2.random() for _ in range(10)]
    assert values1 != values2

def test_constructor_accepts_int_seed():
    instance = Random(100)
    value = instance.random()
    assert isinstance(value, float)

def test_constructor_accepts_none_seed():
    instance = Random(None)
    value = instance.random()
    assert isinstance(value, float)

def test_constructor_accepts_float_seed():
    instance = Random(3.14)
    value = instance.random()
    assert isinstance(value, float)

def test_constructor_accepts_str_seed():
    instance = Random("seed")
    value = instance.random()
    assert isinstance(value, float)

def test_constructor_accepts_bytes_seed():
    instance = Random(b"bytes_seed")
    value = instance.random()
    assert isinstance(value, float)

def test_constructor_accepts_bytearray_seed():
    instance = Random(bytearray(b"bytearray_seed"))
    value = instance.random()
    assert isinstance(value, float)

def test_constructor_instance_has_expected_methods():
    instance = Random()
    assert hasattr(instance, "randints")
    assert hasattr(instance, "generate_string_by_mask")
    assert hasattr(instance, "uniform")
    assert hasattr(instance, "randbytes")
    assert hasattr(instance, "weighted_choice")
    assert hasattr(instance, "choice_enum_item")

def test_constructor_instance_methods_are_callable():
    instance = Random()
    assert callable(instance.randints)
    assert callable(instance.generate_string_by_mask)
    assert callable(instance.uniform)
    assert callable(instance.randbytes)
    assert callable(instance.weighted_choice)
    assert callable(instance.choice_enum_item)


# LLM-generated content at query #2
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_accepts_seed():
    seed = 42
    instance = Random(seed)
    assert isinstance(instance, Random)

def test_constructor_creates_different_instances():
    instance1 = Random()
    instance2 = Random()
    assert instance1 is not instance2

def test_constructor_with_same_seed_produces_same_sequence():
    seed = 123
    instance1 = Random(seed)
    instance2 = Random(seed)
    val1 = instance1.random()
    val2 = instance2.random()
    assert val1 == val2

def test_constructor_with_different_seed_produces_different_sequence():
    instance1 = Random(1)
    instance2 = Random(2)
    val1 = instance1.random()
    val2 = instance2.random()
    assert val1 != val2


# LLM-generated content at query #3
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_inherits_from_random():
    instance = Random()
    assert isinstance(instance, random_module.Random)

def test_constructor_has_randints_method():
    instance = Random()
    assert hasattr(instance, 'randints')
    assert callable(instance.randints)

def test_constructor_has_generate_string_by_mask_method():
    instance = Random()
    assert hasattr(instance, 'generate_string_by_mask')
    assert callable(instance.generate_string_by_mask)

def test_constructor_has_uniform_method():
    instance = Random()
    assert hasattr(instance, 'uniform')
    assert callable(instance.uniform)

def test_constructor_has_randbytes_method():
    instance = Random()
    assert hasattr(instance, 'randbytes')
    assert callable(instance.randbytes)

def test_constructor_has_weighted_choice_method():
    instance = Random()
    assert hasattr(instance, 'weighted_choice')
    assert callable(instance.weighted_choice)

def test_constructor_has_choice_enum_item_method():
    instance = Random()
    assert hasattr(instance, 'choice_enum_item')
    assert callable(instance.choice_enum_item)

def test_constructor_has__generate_string_method():
    instance = Random()
    assert hasattr(instance, '_generate_string')
    assert callable(instance._generate_string)

def test_constructor_can_be_seeded():
    instance1 = Random(12345)
    instance2 = Random(12345)
    val1 = instance1.random()
    val2 = instance2.random()
    assert val1 == val2


# LLM-generated content at query #4
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_inherits_from_random():
    instance = Random()
    assert isinstance(instance, random_module.Random)

def test_constructor_has_randints_method():
    instance = Random()
    assert callable(instance.randints)

def test_constructor_has_generate_string_by_mask_method():
    instance = Random()
    assert callable(instance.generate_string_by_mask)

def test_constructor_has_uniform_method():
    instance = Random()
    assert callable(instance.uniform)

def test_constructor_has_randbytes_method():
    instance = Random()
    assert callable(instance.randbytes)

def test_constructor_has_weighted_choice_method():
    instance = Random()
    assert callable(instance.weighted_choice)

def test_constructor_has_choice_enum_item_method():
    instance = Random()
    assert callable(instance.choice_enum_item)


# LLM-generated content at query #5
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_accepts_seed():
    seed = 42
    instance = Random(seed)
    assert isinstance(instance, Random)

def test_constructor_creates_unique_instances():
    instance1 = Random()
    instance2 = Random()
    assert instance1 is not instance2

def test_constructor_with_same_seed_produces_same_sequence():
    seed = 123
    instance1 = Random(seed)
    instance2 = Random(seed)
    val1 = instance1.random()
    val2 = instance2.random()
    assert val1 == val2

def test_constructor_with_different_seed_produces_different_sequence():
    instance1 = Random(1)
    instance2 = Random(2)
    val1 = instance1.random()
    val2 = instance2.random()
    assert val1 != val2


# LLM-generated content at query #6
#--------------------------

def test_constructor_without_seed():
    r = Random()
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_with_seed():
    r1 = Random(42)
    r2 = Random(42)
    val1 = r1.randint(1, 100)
    val2 = r2.randint(1, 100)
    assert val1 == val2

def test_constructor_with_none_seed():
    r = Random(None)
    assert isinstance(r, Random)

def test_constructor_with_string_seed():
    r = Random("seed")
    assert isinstance(r, Random)

def test_constructor_with_float_seed():
    r = Random(3.14)
    assert isinstance(r, Random)

def test_constructor_inherits_methods():
    r = Random()
    result = r.randint(1, 10)
    assert 1 <= result <= 10


# LLM-generated content at query #7
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_accepts_seed():
    seed = 42
    instance = Random(seed)
    assert isinstance(instance, Random)

def test_constructor_creates_different_instances():
    instance1 = Random()
    instance2 = Random()
    assert instance1 is not instance2

def test_constructor_with_same_seed_produces_same_random_sequence():
    seed = 123
    instance1 = Random(seed)
    instance2 = Random(seed)
    val1 = instance1.random()
    val2 = instance2.random()
    assert val1 == val2

def test_constructor_with_different_seed_produces_different_random_sequence():
    instance1 = Random(1)
    instance2 = Random(2)
    val1 = instance1.random()
    val2 = instance2.random()
    assert val1 != val2

def test_constructor_inherits_from_random_module():
    instance = Random()
    assert isinstance(instance, random_module.Random)


# LLM-generated content at query #8
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_inherits_from_base_random():
    instance = Random()
    assert isinstance(instance, random_module.Random)

def test_constructor_accepts_seed():
    instance = Random(seed=42)
    first_random = instance.random()
    instance2 = Random(seed=42)
    second_random = instance2.random()
    assert first_random == second_random

def test_constructor_without_seed_produces_different_values():
    instance1 = Random()
    instance2 = Random()
    assert instance1.random() != instance2.random()

def test_constructor_sets_custom_methods():
    instance = Random()
    assert hasattr(instance, 'randints')
    assert hasattr(instance, 'generate_string_by_mask')
    assert hasattr(instance, 'uniform')
    assert hasattr(instance, 'randbytes')
    assert hasattr(instance, 'weighted_choice')
    assert hasattr(instance, 'choice_enum_item')
    assert hasattr(instance, '_generate_string')


