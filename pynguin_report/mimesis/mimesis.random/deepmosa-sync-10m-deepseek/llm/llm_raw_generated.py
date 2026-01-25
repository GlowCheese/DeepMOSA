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
    mask = "@@##-@@##"
    result = r.generate_string_by_mask(mask=mask)
    assert len(result) == 9
    parts = result.split('-')
    assert len(parts) == 2
    assert len(parts[0]) == 4 and len(parts[1]) == 4
    assert parts[0][0:2].isalpha() and parts[0][0:2].isupper()
    assert parts[0][2:4].isdigit()
    assert parts[1][0:2].isalpha() and parts[1][0:2].isupper()
    assert parts[1][2:4].isdigit()


def test_generate_string_by_mask_custom_placeholders():
    r = Random()
    mask = "AA99"
    result = r.generate_string_by_mask(mask=mask, char='A', digit='9')
    assert len(result) == 4
    assert result[0:2].isalpha() and result[0:2].isupper()
    assert result[2:4].isdigit()


def test_generate_string_by_mask_fixed_characters():
    r = Random()
    mask = "CODE-@##-END"
    result = r.generate_string_by_mask(mask=mask)
    assert result.startswith("CODE-")
    assert result.endswith("-END")
    middle = result[5:8]
    assert middle[0].isalpha() and middle[0].isupper()
    assert middle[1:3].isdigit()


def test_generate_string_by_mask_same_placeholder_error():
    r = Random()
    try:
        r.generate_string_by_mask(mask="@@##", char='@', digit='@')
        assert False
    except ValueError as e:
        assert "same placeholder" in str(e)


def test_generate_string_by_mask_empty_mask():
    r = Random()
    result = r.generate_string_by_mask(mask="")
    assert result == ""


def test_generate_string_by_mask_only_char_placeholder():
    r = Random()
    mask = "@@@"
    result = r.generate_string_by_mask(mask=mask)
    assert len(result) == 3
    assert all(c.isalpha() and c.isupper() for c in result)


def test_generate_string_by_mask_only_digit_placeholder():
    r = Random()
    mask = "###"
    result = r.generate_string_by_mask(mask=mask)
    assert len(result) == 3
    assert all(c.isdigit() for c in result)


def test_generate_string_by_mask_no_placeholders():
    r = Random()
    mask = "FIXED-STRING"
    result = r.generate_string_by_mask(mask=mask)
    assert result == "FIXED-STRING"


# LLM-generated content at query #2
#--------------------------

def test_constructor_initializes_without_seed():
    r = Random()
    assert r is not None

def test_constructor_initializes_with_int_seed():
    r = Random(seed=42)
    assert r is not None

def test_constructor_initializes_with_float_seed():
    r = Random(seed=3.14)
    assert r is not None

def test_constructor_initializes_with_str_seed():
    r = Random(seed="test")
    assert r is not None

def test_constructor_initializes_with_none_seed():
    r = Random(seed=None)
    assert r is not None

def test_constructor_initializes_with_bytes_seed():
    r = Random(seed=b"seed")
    assert r is not None

def test_constructor_initializes_with_bytearray_seed():
    r = Random(seed=bytearray(b"seed"))
    assert r is not None

def test_constructor_initializes_with_memoryview_seed():
    r = Random(seed=memoryview(b"seed"))
    assert r is not None

def test_constructor_initializes_with_empty_seed():
    r = Random(seed="")
    assert r is not None

def test_constructor_initializes_with_negative_int_seed():
    r = Random(seed=-123)
    assert r is not None


# LLM-generated content at query #3
#--------------------------

def test_constructor_initializes_without_seed():
    r = Random()
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_int_seed():
    r = Random(42)
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_none_seed():
    r = Random(None)
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_float_seed():
    r = Random(3.14)
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_str_seed():
    r = Random("seed")
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)


# LLM-generated content at query #4
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_inherits_from_random():
    instance = Random()
    assert isinstance(instance, random_module.Random)

def test_constructor_accepts_seed():
    seed = 42
    instance1 = Random(seed)
    instance2 = Random(seed)
    val1 = instance1.randint(1, 100)
    val2 = instance2.randint(1, 100)
    assert val1 == val2

def test_constructor_without_seed_produces_different_sequences():
    instance1 = Random()
    instance2 = Random()
    seq1 = [instance1.randint(1, 100) for _ in range(5)]
    seq2 = [instance2.randint(1, 100) for _ in range(5)]
    assert seq1 != seq2

def test_constructor_initializes_random_state():
    instance = Random(123)
    first_value = instance.random()
    instance2 = Random(123)
    first_value2 = instance2.random()
    assert first_value == first_value2


# LLM-generated content at query #5
#--------------------------

def test_constructor_initializes_without_seed():
    r = Random()
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_int_seed():
    r = Random(seed=42)
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_none_seed():
    r = Random(seed=None)
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_float_seed():
    r = Random(seed=3.14)
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_str_seed():
    r = Random(seed="test_seed")
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_bytes_seed():
    r = Random(seed=b"bytes_seed")
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_bytearray_seed():
    r = Random(seed=bytearray(b"bytearray_seed"))
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)


# LLM-generated content at query #6
#--------------------------

def test_constructor_initializes_without_seed():
    r = Random()
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_int_seed():
    r = Random(42)
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_none_seed():
    r = Random(None)
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_float_seed():
    r = Random(3.14)
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_str_seed():
    r = Random("seed")
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_bytes_seed():
    r = Random(b"seed")
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_bytearray_seed():
    r = Random(bytearray(b"seed"))
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_memoryview_seed():
    r = Random(memoryview(b"seed"))
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_version_arg():
    r = Random(version=2)
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)

def test_constructor_initializes_with_int_seed_and_version():
    r = Random(42, version=2)
    assert isinstance(r, Random)
    assert isinstance(r, random_module.Random)


# LLM-generated content at query #7
#--------------------------

def test_constructor_initializes_without_seed():
    rng = Random()
    result = rng.randint(1, 10)
    assert 1 <= result <= 10

def test_constructor_initializes_with_int_seed():
    rng = Random(42)
    result1 = rng.randint(1, 100)
    rng2 = Random(42)
    result2 = rng2.randint(1, 100)
    assert result1 == result2

def test_constructor_initializes_with_none_seed():
    rng = Random(None)
    result = rng.randint(1, 10)
    assert 1 <= result <= 10

def test_constructor_initializes_and_produces_different_defaults():
    rng1 = Random()
    rng2 = Random()
    results1 = [rng1.randint(1, 1000) for _ in range(5)]
    results2 = [rng2.randint(1, 1000) for _ in range(5)]
    assert results1 != results2

def test_constructor_initializes_with_same_seed_produces_same_sequence():
    seed = 12345
    rng1 = Random(seed)
    rng2 = Random(seed)
    seq1 = [rng1.random() for _ in range(10)]
    seq2 = [rng2.random() for _ in range(10)]
    assert seq1 == seq2


# LLM-generated content at query #8
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_accepts_seed():
    seed = 42
    instance1 = Random(seed)
    instance2 = Random(seed)
    val1 = instance1.randint(1, 100)
    val2 = instance2.randint(1, 100)
    assert val1 == val2

def test_constructor_without_seed_produces_different_sequences():
    instance1 = Random()
    instance2 = Random()
    seq1 = [instance1.randint(1, 100) for _ in range(5)]
    seq2 = [instance2.randint(1, 100) for _ in range(5)]
    assert seq1 != seq2

def test_constructor_inherits_from_random_random():
    instance = Random()
    assert isinstance(instance, random_module.Random)


# LLM-generated content at query #9
#--------------------------

def test_constructor_default():
    r = Random()
    assert isinstance(r, Random)

def test_constructor_with_seed():
    r1 = Random(42)
    r2 = Random(42)
    assert r1.random() == r2.random()

def test_constructor_inherits_from_random():
    r = Random()
    assert isinstance(r, random_module.Random)

def test_constructor_no_args():
    r = Random()
    result = r.randint(1, 10)
    assert 1 <= result <= 10

def test_constructor_with_none_seed():
    r = Random(None)
    result = r.randint(1, 10)
    assert 1 <= result <= 10


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_without_seed():
    r = Random()
    assert isinstance(r, random_module.Random)
    assert isinstance(r, Random)

def test_constructor_with_seed():
    r = Random(seed=42)
    first_random = r.random()
    r2 = Random(seed=42)
    second_random = r2.random()
    assert first_random == second_random

def test_constructor_inherits_random_methods():
    r = Random()
    result = r.randint(1, 10)
    assert 1 <= result <= 10

def test_constructor_initializes_custom_methods():
    r = Random()
    ints = r.randints()
    assert len(ints) == 3
    assert all(isinstance(i, int) for i in ints)


# LLM-generated content at query #2
#--------------------------

def test_generate_string_by_mask_default_mask():
    r = Random()
    result = r.generate_string_by_mask()
    assert len(result) == 4
    assert result[0].isalpha() and result[0].isupper()
    assert result[1].isdigit()
    assert result[2].isdigit()
    assert result[3].isdigit()

def test_generate_string_by_mask_custom_mask():
    r = Random()
    result = r.generate_string_by_mask(mask="@@##", char="@", digit="#")
    assert len(result) == 4
    assert result[0].isalpha() and result[0].isupper()
    assert result[1].isalpha() and result[1].isupper()
    assert result[2].isdigit()
    assert result[3].isdigit()

def test_generate_string_by_mask_different_placeholders():
    r = Random()
    result = r.generate_string_by_mask(mask="AA99", char="A", digit="9")
    assert len(result) == 4
    assert result[0].isalpha() and result[0].isupper()
    assert result[1].isalpha() and result[1].isupper()
    assert result[2].isdigit()
    assert result[3].isdigit()

def test_generate_string_by_mask_same_placeholder_raises_valueerror():
    r = Random()
    try:
        r.generate_string_by_mask(mask="@@##", char="@", digit="@")
        assert False
    except ValueError as e:
        assert "same placeholder" in str(e)

def test_generate_string_by_mask_empty_mask():
    r = Random()
    result = r.generate_string_by_mask(mask="", char="@", digit="#")
    assert result == ""

def test_generate_string_by_mask_no_placeholders():
    r = Random()
    result = r.generate_string_by_mask(mask="fixed", char="@", digit="#")
    assert result == "fixed"

def test_generate_string_by_mask_mixed_characters():
    r = Random()
    result = r.generate_string_by_mask(mask="A@1#", char="@", digit="#")
    assert len(result) == 4
    assert result[0] == "A"
    assert result[1].isalpha() and result[1].isupper()
    assert result[2] == "1"
    assert result[3].isdigit()

def test_generate_string_by_mask_long_mask():
    r = Random()
    mask = "@" * 100 + "#" * 100
    result = r.generate_string_by_mask(mask=mask, char="@", digit="#")
    assert len(result) == 200
    for i in range(100):
        assert result[i].isalpha() and result[i].isupper()
    for i in range(100, 200):
        assert result[i].isdigit()


# LLM-generated content at query #3
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_accepts_seed():
    instance = Random(12345)
    result1 = instance.random()
    instance2 = Random(12345)
    result2 = instance2.random()
    assert result1 == result2

def test_constructor_without_seed_produces_different_instances():
    instance1 = Random()
    instance2 = Random()
    result1 = instance1.random()
    result2 = instance2.random()
    assert result1 != result2

def test_constructor_creates_instance_of_correct_type():
    instance = Random()
    assert type(instance).__name__ == 'Random'

def test_constructor_initializes_random_state():
    instance = Random()
    result = instance.random()
    assert isinstance(result, float)
    assert 0.0 <= result < 1.0


# LLM-generated content at query #4
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_accepts_seed():
    seed = 12345
    instance1 = Random(seed)
    instance2 = Random(seed)
    val1 = instance1.randint(1, 100)
    val2 = instance2.randint(1, 100)
    assert val1 == val2

def test_constructor_default_seed_produces_different_sequences():
    instance1 = Random()
    instance2 = Random()
    vals1 = [instance1.randint(1, 100) for _ in range(5)]
    vals2 = [instance2.randint(1, 100) for _ in range(5)]
    assert vals1 != vals2

def test_constructor_with_none_seed():
    instance = Random(None)
    val = instance.randint(1, 10)
    assert 1 <= val <= 10


# LLM-generated content at query #5
#--------------------------

def test_constructor_creates_instance():
    instance = Random()
    assert isinstance(instance, Random)

def test_constructor_creates_instance_with_seed():
    instance = Random(12345)
    assert isinstance(instance, Random)

def test_constructor_creates_instance_without_arguments():
    instance = Random()
    assert instance is not None

def test_constructor_creates_instance_with_int_seed():
    instance = Random(42)
    assert isinstance(instance, Random)

def test_constructor_creates_instance_with_float_seed():
    instance = Random(3.14)
    assert isinstance(instance, Random)

def test_constructor_creates_instance_with_str_seed():
    instance = Random("seed")
    assert isinstance(instance, Random)

def test_constructor_creates_instance_with_none_seed():
    instance = Random(None)
    assert isinstance(instance, Random)

def test_constructor_creates_instance_with_bytes_seed():
    instance = Random(b"seed")
    assert isinstance(instance, Random)

def test_constructor_creates_instance_with_bytearray_seed():
    instance = Random(bytearray(b"seed"))
    assert isinstance(instance, Random)


# LLM-generated content at query #6
#--------------------------

def test_constructor_initializes_without_seed():
    r = Random()
    assert isinstance(r, Random)

def test_constructor_initializes_with_int_seed():
    r = Random(42)
    assert isinstance(r, Random)

def test_constructor_initializes_with_none_seed():
    r = Random(None)
    assert isinstance(r, Random)

def test_constructor_initializes_with_float_seed():
    r = Random(3.14)
    assert isinstance(r, Random)

def test_constructor_initializes_with_str_seed():
    r = Random("seed")
    assert isinstance(r, Random)

def test_constructor_initializes_with_bytes_seed():
    r = Random(b"seed")
    assert isinstance(r, Random)

def test_constructor_initializes_with_bytearray_seed():
    r = Random(bytearray(b"seed"))
    assert isinstance(r, Random)

def test_constructor_initializes_with_memoryview_seed():
    r = Random(memoryview(b"seed"))
    assert isinstance(r, Random)

def test_constructor_initializes_with_version_argument():
    r = Random(version=2)
    assert isinstance(r, Random)


# LLM-generated content at query #7
#--------------------------

def test_constructor_initializes_without_seed():
    r = Random()
    assert isinstance(r, Random)

def test_constructor_initializes_with_seed():
    r = Random(12345)
    assert isinstance(r, Random)

def test_constructor_initializes_with_none():
    r = Random(None)
    assert isinstance(r, Random)

def test_constructor_instance_of_parent():
    r = Random()
    assert isinstance(r, random_module.Random)


# LLM-generated content at query #8
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

def test_constructor_default_seed_produces_different_sequences():
    instance1 = Random()
    instance2 = Random()
    val1 = instance1.random()
    val2 = instance2.random()
    assert val1 != val2


# LLM-generated content at query #9
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

def test_constructor_default_instance_has_random_method():
    instance = Random()
    result = instance.random()
    assert isinstance(result, float)
    assert 0.0 <= result < 1.0

def test_constructor_instance_inherits_from_base_random():
    instance = Random()
    assert isinstance(instance, random_module.Random)


