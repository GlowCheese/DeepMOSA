####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_maybe_parse_user_type_with_preserved_type():
    assert maybe_parse_user_type(list) == [list]

def test_maybe_parse_user_type_with_string():
    assert maybe_parse_user_type("str") == ["str"]

def test_maybe_parse_user_type_with_non_iterable_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    assert maybe_parse_user_type([int, str]) == [int, str]

def test_maybe_parse_user_type_with_nested_iterable_of_types():
    assert maybe_parse_user_type([int, [str, float]]) == [int, str, float]

def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_empty_invariants():
    assert _invariant_errors("test", []) == []

def test_all_valid():
    invariants = [
        lambda x: (True, "error1"),
        lambda x: (True, "error2"),
    ]
    assert _invariant_errors("test", invariants) == []

def test_all_invalid():
    invariants = [
        lambda x: (False, "error1"),
        lambda x: (False, "error2"),
    ]
    assert _invariant_errors("test", invariants) == ["error1", "error2"]

def test_mixed_validity():
    invariants = [
        lambda x: (True, "error1"),
        lambda x: (False, "error2"),
        lambda x: (True, "error3"),
    ]
    assert _invariant_errors("test", invariants) == ["error2"]


# LLM-generated content at query #3
#--------------------------

```python
def test__invariant_errors_empty_invariants():
    result = _invariant_errors(5, [])
    assert result == []

def test__invariant_errors_all_valid():
    def always_valid(x):
        return (True, "valid")
    result = _invariant_errors(5, [always_valid, always_valid])
    assert result == []

def test__invariant_errors_some_invalid():
    def sometimes_invalid(x):
        return (x != 5, f"invalid for {x}")
    result = _invariant_errors(5, [sometimes_invalid])
    assert result == ["invalid for 5"]

def test__invariant_errors_multiple_invalid():
    def always_invalid(x):
        return (False, f"always invalid {x}")
    def sometimes_invalid(x):
        return (x != 5, f"invalid for {x}")
    result = _invariant_errors(5, [always_invalid, sometimes_invalid])
    assert result == ["always invalid 5", "invalid for 5"]


# LLM-generated content at query #4
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"1": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: "1.0"})
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5, initial={1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0}

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #5
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.5, 2: 2.25}

def test_checked_pmap_constructor_with_invalid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.5, 2: 2.25})
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) == 1
        assert e.error_codes[0] == 'Invalid mapping'

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5, initial={1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.5, 2: 2.25}

def test_checked_pmap_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #6
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.5, 2: 2.25}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({"a": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({1: "1.5"})
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.5})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap(size=5, initial={1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.5}


# LLM-generated content at query #7
#--------------------------

```python
def test_invariant_errors_with_valid_invariants():
    elem = "test"
    invariants = [lambda x: (True, "data1"), lambda x: (True, "data2")]
    result = _invariant_errors(elem, invariants)
    assert result == []


# LLM-generated content at query #8
#--------------------------

```python
def test_checked_type_instantiation():
    with pytest.raises(TypeError):
        CheckedType()


# LLM-generated content at query #9
#--------------------------

```python
def test__checked_type_create_with_instance_of_cls():
    class TestClass:
        pass
    instance = TestClass()
    assert _checked_type_create(TestClass, instance) is instance

def test__checked_type_create_with_non_instance_of_cls():
    class TestClass:
        _checked_types = []
    assert isinstance(_checked_type_create(TestClass, "data"), TestClass)

def test__checked_type_create_with_checked_type_in_source_data():
    class TestCheckedType(CheckedType):
        pass
    class TestClass:
        _checked_types = ["TestCheckedType"]
    data = [TestCheckedType(), "other_data"]
    result = _checked_type_create(TestClass, data)
    assert isinstance(result, TestClass)
    assert isinstance(result[0], TestCheckedType)
    assert result[1] == "other_data"

def test__checked_type_create_with_ignore_extra():
    class TestCheckedType(CheckedType):
        pass
    class TestClass:
        _checked_types = ["TestCheckedType"]
    data = [{"extra": "data"}]
    result = _checked_type_create(TestClass, data, ignore_extra=True)
    assert isinstance(result, TestClass)


# LLM-generated content at query #10
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({"a": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({1: "1.0"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap(size=5, initial={1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0}

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #11
#--------------------------

```python
def test_store_types_with_single_type():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = int
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int]

def test_store_types_with_string():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = 'str'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == ['str']

def test_store_types_with_preserved_iterable():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = list
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [list]

def test_store_types_with_iterable_of_types():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = [int, str, list]
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str, list]

def test_store_types_with_invalid_type():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = 123
    try:
        _store_types(dct, bases, destination_name, source_name)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_store_types_with_base_classes():
    dct = {}
    class Base1:
        type = int
    class Base2:
        type = str
    bases = [Base1, Base2]
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str]

def test_store_types_with_mixed_sources():
    dct = {'type': list}
    class Base:
        type = int
    bases = [Base]
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [list, int]


# LLM-generated content at query #12
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    result = CheckedPSet()
    assert isinstance(result, CheckedPSet)
    assert len(result) == 0

def test_checked_pset_constructor_with_list_initial():
    class TestSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestSet([1, 2, 3])
    assert isinstance(result, TestSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checked_pset_constructor_with_pmap_initial():
    class TestSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    pmap = pmap({1: True, 2: True, 3: True})
    result = TestSet(pmap)
    assert isinstance(result, TestSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checked_pset_constructor_with_invalid_type():
    class TestSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestSet([1, 2, "3"])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pset_constructor_with_invalid_invariant():
    class TestSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestSet([1, 2, -3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test_checkedpset_constructor_with_empty_initial():
    result = CheckedPSet()
    assert isinstance(result, CheckedPSet)
    assert len(result) == 0

def test_checkedpset_constructor_with_list_initial():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestCheckedPSet([1, 2, 3])
    assert isinstance(result, TestCheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checkedpset_constructor_with_pmap_initial():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    pmap = pmap({1: True, 2: True, 3: True})
    result = TestCheckedPSet(pmap)
    assert isinstance(result, TestCheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checkedpset_constructor_with_invalid_type():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestCheckedPSet(["invalid"])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checkedpset_constructor_with_invalid_invariant():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestCheckedPSet([-1, -2, -3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_initial_data():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    initial_data = {1: "one", 2: "two"}
    result = TestMap(initial_data)
    assert isinstance(result, TestMap)
    assert dict(result) == initial_data

def test_checked_pmap_constructor_with_size():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap(size=5)
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_invalid_key_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({"invalid_key": "value"})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({1: 123})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")

    result = TestMap({1: 2, 3: 4})
    assert isinstance(result, TestMap)
    assert dict(result) == {1: 2, 3: 4}

def test_checked_pmap_constructor_with_invariant_violation():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")

    try:
        TestMap({1: 2, 3: 2})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #15
#--------------------------

```python
def test_maybe_parse_user_type_with_non_iterable_type():
    assert maybe_parse_user_type(int) == [int]


# LLM-generated content at query #16
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
        __invariant__ = lambda k, v: (len(v) == k, "Invalid mapping")

    test_map = TestCheckedPMap({1: "a", 2: "bb", 3: "ccc"})
    assert isinstance(test_map, TestCheckedPMap)
    assert dict(test_map) == {1: "a", 2: "bb", 3: "ccc"}

def test_checked_pmap_constructor_with_invalid_key_type():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestCheckedPMap({"a": "value"})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestCheckedPMap({1: 123})
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
        __invariant__ = lambda k, v: (len(v) == k, "Invalid mapping")

    try:
        TestCheckedPMap({1: "abc"})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    test_map = TestCheckedPMap(size=10)
    assert isinstance(test_map, TestCheckedPMap)
    assert len(test_map) == 0


# LLM-generated content at query #17
#--------------------------

```python
def test_maybe_parse_user_type_with_non_iterable_type():
    assert maybe_parse_user_type(int) == [int]


# LLM-generated content at query #18
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"1": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: "1.0"})
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap(size=5, initial={1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0}


# LLM-generated content at query #19
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "data1"), (True, "data2"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_all_false():
    result = [(False, "data1"), (False, "data2"), (False, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("data1", "data2", "data3")

def test_merge_invariant_results_mixed():
    result = [(True, "data1"), (False, "data2"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("data2",)

def test_merge_invariant_results_empty():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


# LLM-generated content at query #20
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_type():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = int
    try:
        _check_types(it, expected_types, source_class)
    except CheckedValueTypeError as e:
        assert str(e) == "Type int can only be used with ('int',), not str"

def test_check_types_with_multiple_valid_types():
    it = [1, 2.0, 3]
    expected_types = [int, float]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_empty_iterable():
    it = []
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_custom_exception():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = int
    custom_exception = ValueError
    try:
        _check_types(it, expected_types, source_class, custom_exception)
    except ValueError as e:
        assert str(e) == "Type int can only be used with ('int',), not str"


# LLM-generated content at query #21
#--------------------------

```python
def test_constructor_with_valid_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_constructor_with_invalid_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 1.5, 2: 2.25})
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) == 2
        assert all('Invalid mapping' in error for error in e.error_codes)

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_empty_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({'a': 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 'a'})
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_initial_data():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap({1: "a", 2: "b"})
    assert isinstance(result, TestMap)
    assert dict(result) == {1: "a", 2: "b"}

def test_checked_pmap_constructor_with_size():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap(size=5)
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_invalid_key_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({"a": "b"})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({1: 2})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invariant_violation():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")

    try:
        TestMap({2: 1})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_valid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")

    result = TestMap({1: 2, 3: 4})
    assert isinstance(result, TestMap)
    assert dict(result) == {1: 2, 3: 4}


# LLM-generated content at query #23
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = type(it)
    _check_types(it, expected_types, source_class)


# LLM-generated content at query #24
#--------------------------

```python
def test_checked_pset_constructor_empty():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestCheckedPSet()
    assert isinstance(result, TestCheckedPSet)
    assert len(result) == 0

def test_checked_pset_constructor_with_valid_elements():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestCheckedPSet([1, 2, 3])
    assert isinstance(result, TestCheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checked_pset_constructor_with_invalid_type():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestCheckedPSet([1, 2, "3"])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pset_constructor_with_invalid_invariant():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestCheckedPSet([1, 2, -3])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) == 1
        assert e.error_codes[0][1] == "Non-positive"

def test_checked_pset_constructor_from_pmap():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    pmap_instance = pmap({1: True, 2: True, 3: True})
    result = TestCheckedPSet(pmap_instance)
    assert isinstance(result, TestCheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


# LLM-generated content at query #25
#--------------------------

```python
def test_checked_type_instantiation():
    with pytest.raises(TypeError):
        CheckedType()


# LLM-generated content at query #26
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap({1: "a", 2: "b"})
    assert isinstance(result, TestMap)
    assert dict(result) == {1: "a", 2: "b"}

def test_checked_pmap_constructor_with_invalid_key_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({"a": "b", "c": "d"})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({1: 2, 3: 4})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")

    try:
        TestMap({1: 0, 2: 3})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_valid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")

    result = TestMap({1: 2, 3: 4})
    assert isinstance(result, TestMap)
    assert dict(result) == {1: 2, 3: 4}

def test_checked_pmap_constructor_empty():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_size():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap(size=5)
    assert isinstance(result, TestMap)
    assert len(result) == 0


# LLM-generated content at query #27
#--------------------------

```python
def test_constructor_with_empty_initial():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector()
    assert isinstance(result, TestVector)
    assert len(result) == 0

def test_constructor_with_list_initial():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector([1, 2, 3])
    assert isinstance(result, TestVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_constructor_with_pvector_initial():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    pv = python_pvector([1, 2, 3])
    result = TestVector(pv)
    assert isinstance(result, TestVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_constructor_with_invalid_type():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector([1, "2", 3])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_constructor_with_invalid_invariant():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector([1, -2, 3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #28
#--------------------------

```python
def test_check_types_with_empty_iterable():
    _check_types([], [int, str], object)


# LLM-generated content at query #29
#--------------------------

```python
def test_constructor_with_empty_initial():
    result = CheckedPVector()
    assert isinstance(result, CheckedPVector)
    assert len(result) == 0

def test_constructor_with_list_initial():
    result = CheckedPVector([1, 2, 3])
    assert isinstance(result, CheckedPVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_constructor_with_python_pvector_initial():
    pv = python_pvector([1, 2, 3])
    result = CheckedPVector(pv)
    assert isinstance(result, CheckedPVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_constructor_with_invalid_type():
    try:
        CheckedPVector([1, 'a', 3])
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

def test_constructor_with_invalid_invariant():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([1, -2, 3])
    except InvariantException:
        pass
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #30
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.5, 2: 2.25}

def test_checked_pmap_constructor_with_invalid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 1.0, 2: 2.25})
    except InvariantException as e:
        assert e.error_codes == ['Invalid mapping']
    else:
        assert False, "Expected InvariantException"

def test_checked_pmap_constructor_with_empty_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=10)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


# LLM-generated content at query #31
#--------------------------

```python
def test_checked_type_create_predicate_false():
    class TestClass:
        pass

    class NotCheckedType:
        pass

    source_data = NotCheckedType()
    assert not isinstance(source_data, TestClass)


# LLM-generated content at query #32
#--------------------------

```python
def test_checked_type_create_with_checked_type_subclass():
    class TestCheckedType(CheckedType):
        pass

    class TestClass:
        _checked_types = [TestCheckedType]

    source_data = [TestCheckedType(), TestCheckedType()]
    result = TestClass._checked_type_create(source_data)
    assert result == TestClass(source_data)


# LLM-generated content at query #33
#--------------------------

```python
def test_checked_pvector_constructor_with_empty_initial():
    result = CheckedPVector()
    assert isinstance(result, CheckedPVector)
    assert len(result) == 0

def test_checked_pvector_constructor_with_list_initial():
    result = CheckedPVector([1, 2, 3])
    assert isinstance(result, CheckedPVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_checked_pvector_constructor_with_python_pvector_initial():
    pv = python_pvector([1, 2, 3])
    result = CheckedPVector(pv)
    assert isinstance(result, CheckedPVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_checked_pvector_constructor_with_tuple_initial():
    result = CheckedPVector((1, 2, 3))
    assert isinstance(result, CheckedPVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_checked_pvector_constructor_with_generator_initial():
    result = CheckedPVector(x for x in [1, 2, 3])
    assert isinstance(result, CheckedPVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #34
#--------------------------

```python
def test_constructor_empty_initial():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector()
    assert isinstance(result, TestVector)
    assert len(result) == 0

def test_constructor_list_initial():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector([1, 2, 3])
    assert isinstance(result, TestVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_constructor_pvector_initial():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    pv = python_pvector([1, 2, 3])
    result = TestVector(pv)
    assert isinstance(result, TestVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_constructor_type_error():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector([1, "not an int", 3])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_constructor_invariant_error():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector([1, -2, 3])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) == 1
        assert e.error_codes[0][1] == "Non-positive"


# LLM-generated content at query #35
#--------------------------

```python
def test__check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)

def test__check_types_with_invalid_types():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = int
    try:
        _check_types(it, expected_types, source_class)
    except CheckedValueTypeError:
        pass

def test__check_types_with_multiple_valid_types():
    it = [1, 2.0, 3]
    expected_types = [int, float]
    source_class = int
    _check_types(it, expected_types, source_class)

def test__check_types_with_multiple_invalid_types():
    it = [1, 'a', 3.0]
    expected_types = [int]
    source_class = int
    try:
        _check_types(it, expected_types, source_class)
    except CheckedValueTypeError:
        pass

def test__check_types_with_empty_iterable():
    it = []
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)

def test__check_types_with_empty_expected_types():
    it = [1, 2, 3]
    expected_types = []
    source_class = int
    _check_types(it, expected_types, source_class)

def test__check_types_with_custom_exception():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = int
    try:
        _check_types(it, expected_types, source_class, exception_type=ValueError)
    except ValueError:
        pass


# LLM-generated content at query #36
#--------------------------

```python
def test_checked_type_instantiation():
    """Test that CheckedType cannot be instantiated directly."""
    with pytest.raises(TypeError):
        CheckedType()


# LLM-generated content at query #37
#--------------------------

```python
def test_restore_pickle_creates_instance_with_factory_fields():
    class TestClass:
        @classmethod
        def create(cls, data, _factory_fields):
            return cls(data, _factory_fields)

        def __init__(self, data, _factory_fields):
            self.data = data
            self._factory_fields = _factory_fields

    data = {"key": "value"}
    result = _restore_pickle(TestClass, data)
    assert isinstance(result, TestClass)
    assert result.data == data
    assert result._factory_fields == set()


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_maybe_parse_user_type_with_preserved_type():
    assert maybe_parse_user_type(list) == [list]

def test_maybe_parse_user_type_with_string():
    assert maybe_parse_user_type("str") == ["str"]

def test_maybe_parse_user_type_with_non_iterable_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    assert maybe_parse_user_type([int, str]) == [int, str]

def test_maybe_parse_user_type_with_iterable_of_strings():
    assert maybe_parse_user_type(["int", "str"]) == ["int", "str"]

def test_maybe_parse_user_type_with_nested_iterable():
    assert maybe_parse_user_type([int, [str, "float"]]) == [int, str, "float"]

def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"1": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: "1.0"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_violated_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5, initial={1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0}


# LLM-generated content at query #3
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"1": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: "1.0"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_empty_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [
            lambda k, v: (int(v) == k, 'Invalid mapping'),
            lambda k, v: (v > 0, 'Value must be positive')
        ]

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_multiple_invariants_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [
            lambda k, v: (int(v) == k, 'Invalid mapping'),
            lambda k, v: (v > 0, 'Value must be positive')
        ]

    try:
        IntToFloatMap({1: -1.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_invariant_errors_with_no_errors():
    elem = 5
    invariants = [lambda x: (True, "Error1"), lambda x: (True, "Error2")]
    assert _invariant_errors(elem, invariants) == []

def test_invariant_errors_with_one_error():
    elem = 5
    invariants = [lambda x: (False, "Error1"), lambda x: (True, "Error2")]
    assert _invariant_errors(elem, invariants) == ["Error1"]

def test_invariant_errors_with_multiple_errors():
    elem = 5
    invariants = [lambda x: (False, "Error1"), lambda x: (False, "Error2")]
    assert _invariant_errors(elem, invariants) == ["Error1", "Error2"]

def test_invariant_errors_with_all_errors():
    elem = 5
    invariants = [lambda x: (False, "Error1"), lambda x: (False, "Error2"), lambda x: (False, "Error3")]
    assert _invariant_errors(elem, invariants) == ["Error1", "Error2", "Error3"]


# LLM-generated content at query #5
#--------------------------

```python
def test_maybe_parse_user_type_with_preserved_type():
    assert maybe_parse_user_type(list) == [list]

def test_maybe_parse_user_type_with_string():
    assert maybe_parse_user_type("str") == ["str"]

def test_maybe_parse_user_type_with_non_iterable_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    assert maybe_parse_user_type([int, str]) == [int, str]

def test_maybe_parse_user_type_with_nested_iterable_of_types():
    assert maybe_parse_user_type([int, [str, float]]) == [int, str, float]

def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_initial_data():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    initial_data = {1: "one", 2: "two"}
    result = TestMap(initial_data)
    assert isinstance(result, TestMap)
    assert dict(result) == initial_data

def test_checked_pmap_constructor_with_size():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap(size=5)
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_invalid_key_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({"a": "one"})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({1: 1})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (v == k * 2, "Value must be double the key")

    try:
        TestMap({1: 3.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_valid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (v == k * 2, "Value must be double the key")

    result = TestMap({1: 2.0, 2: 4.0})
    assert isinstance(result, TestMap)
    assert dict(result) == {1: 2.0, 2: 4.0}


# LLM-generated content at query #7
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 1.5, 2: 2.25})
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) == 2
        assert all('Invalid mapping' in error for error in e.error_codes)

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap(size=10)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_type_mismatch():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({'a': 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_initial_data():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    initial_data = {1: "one", 2: "two"}
    result = TestMap(initial_data)
    assert isinstance(result, TestMap)
    assert dict(result) == initial_data

def test_checked_pmap_constructor_with_size():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    initial_data = {1: "one", 2: "two"}
    result = TestMap(initial_data, size=4)
    assert isinstance(result, TestMap)
    assert dict(result) == initial_data

def test_checked_pmap_constructor_with_invalid_key_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({"invalid": "value"})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({1: 123})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
        __invariant__ = lambda k, v: (len(v) == k, "Length mismatch")

    try:
        TestMap({1: "one"})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test__check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = list
    _check_types(it, expected_types, source_class)

def test__check_types_with_invalid_types():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = list
    try:
        _check_types(it, expected_types, source_class)
    except CheckedValueTypeError as e:
        assert str(e) == "Type list can only be used with ('int',), not str"

def test__check_types_with_multiple_valid_types():
    it = [1, 2.0, 3]
    expected_types = [int, float]
    source_class = list
    _check_types(it, expected_types, source_class)

def test__check_types_with_multiple_invalid_types():
    it = [1, 'a', 3.0]
    expected_types = [int, float]
    source_class = list
    try:
        _check_types(it, expected_types, source_class)
    except CheckedValueTypeError as e:
        assert str(e) == "Type list can only be used with ('int', 'float'), not str"

def test__check_types_with_empty_iterable():
    it = []
    expected_types = [int]
    source_class = list
    _check_types(it, expected_types, source_class)

def test__check_types_with_empty_expected_types():
    it = [1, 'a', 3]
    expected_types = []
    source_class = list
    _check_types(it, expected_types, source_class)

def test__check_types_with_custom_exception():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = list
    custom_exception = ValueError
    try:
        _check_types(it, expected_types, source_class, custom_exception)
    except ValueError as e:
        assert str(e) == "Type list can only be used with ('int',), not str"


# LLM-generated content at query #10
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"1": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: "1.0"})
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5, initial={1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0}

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #11
#--------------------------

```python
def test_checked_type_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)


# LLM-generated content at query #12
#--------------------------

```python
def test_store_invariants_with_valid_invariants():
    class Base:
        def invariant1(self):
            return True, "data1"

    class Derived(Base):
        def invariant2(self):
            return True, "data2"

    dct = {}
    bases = (Derived,)
    store_invariants(dct, bases, "destination", "invariant")
    assert "destination" in dct
    assert len(dct["destination"]) == 2
    assert all(callable(inv) for inv in dct["destination"])

def test_store_invariants_with_invalid_invariant():
    class Base:
        invariant = "not_callable"

    dct = {}
    bases = (Base,)
    try:
        store_invariants(dct, bases, "destination", "invariant")
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"

def test_store_invariants_with_no_invariants():
    class Base:
        pass

    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, "destination", "invariant")
    assert "destination" in dct
    assert len(dct["destination"]) == 0

def test_store_invariants_with_inherited_invariants():
    class Base:
        def invariant(self):
            return True, "data"

    class Derived(Base):
        pass

    dct = {}
    bases = (Derived,)
    store_invariants(dct, bases, "destination", "invariant")
    assert "destination" in dct
    assert len(dct["destination"]) == 1
    assert callable(dct["destination"][0])

def test_store_invariants_with_multiple_inheritance():
    class Base1:
        def invariant1(self):
            return True, "data1"

    class Base2:
        def invariant2(self):
            return True, "data2"

    class Derived(Base1, Base2):
        pass

    dct = {}
    bases = (Derived,)
    store_invariants(dct, bases, "destination", "invariant")
    assert "destination" in dct
    assert len(dct["destination"]) == 2
    assert all(callable(inv) for inv in dct["destination"])


# LLM-generated content at query #13
#--------------------------

```python
def test_is_preserved_predicate():
    class PreservedType(_preserved_iterable_types):
        pass

    t = PreservedType
    is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
    assert is_preserved is True


# LLM-generated content at query #14
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({'a': 1.0, 2: 2.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: '1.0', 2: 2.0})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0, 2: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5, initial={1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_18():
    assert maybe_parse_user_type(int) == [int]


# LLM-generated content at query #16
#--------------------------

```python
def test_constructor_with_empty_initial():
    result = CheckedPVector()
    assert isinstance(result, CheckedPVector)
    assert len(result) == 0

def test_constructor_with_list_initial():
    result = CheckedPVector([1, 2, 3])
    assert isinstance(result, CheckedPVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_constructor_with_python_pvector_initial():
    pv = python_pvector([1, 2, 3])
    result = CheckedPVector(pv)
    assert isinstance(result, CheckedPVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #17
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    result = CheckedPSet()
    assert isinstance(result, CheckedPSet)
    assert len(result) == 0

def test_checked_pset_constructor_with_list_initial():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestCheckedPSet([1, 2, 3])
    assert isinstance(result, TestCheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checked_pset_constructor_with_pmap_initial():
    pmap_instance = pmap({1: True, 2: True, 3: True})
    result = CheckedPSet(pmap_instance)
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checked_pset_constructor_with_invalid_type():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestCheckedPSet(["invalid"])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pset_constructor_with_invalid_invariant():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestCheckedPSet([-1, -2])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_restore_pickle_returns_instance_of_cls():
    class TestClass:
        @classmethod
        def create(cls, data, _factory_fields):
            return cls()

    data = {"key": "value"}
    result = _restore_pickle(TestClass, data)
    assert isinstance(result, TestClass)


# LLM-generated content at query #19
#--------------------------

```python
def test_wrap_invariant_with_single_bool_result():
    def invariant():
        return (True, "data")
    wrapped = wrap_invariant(invariant)
    assert wrapped() == (True, "data")

def test_wrap_invariant_with_multiple_bool_results():
    def invariant():
        return [(True, "data1"), (False, "data2"), (True, "data3")]
    wrapped = wrap_invariant(invariant)
    assert wrapped() == (False, ("data2",))

def test_wrap_invariant_with_all_true_results():
    def invariant():
        return [(True, "data1"), (True, "data2")]
    wrapped = wrap_invariant(invariant)
    assert wrapped() == (True, ())

def test_wrap_invariant_with_all_false_results():
    def invariant():
        return [(False, "data1"), (False, "data2")]
    wrapped = wrap_invariant(invariant)
    assert wrapped() == (False, ("data1", "data2"))


# LLM-generated content at query #20
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({"a": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap(size=5, initial={1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0}

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 1.5, 2: 2.25})
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) == 2

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap(size=5)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


# LLM-generated content at query #2
#--------------------------

```python
def test_checked_type_instantiation():
    with pytest.raises(TypeError):
        CheckedType()


# LLM-generated content at query #3
#--------------------------

```python
def test_checked_pvector_constructor_empty():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, 'Non-positive')

    result = TestVector()
    assert result.tolist() == []
    assert isinstance(result, TestVector)

def test_checked_pvector_constructor_from_list():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, 'Non-positive')

    result = TestVector([1, 2, 3])
    assert result.tolist() == [1, 2, 3]
    assert isinstance(result, TestVector)

def test_checked_pvector_constructor_from_pvector():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, 'Non-positive')

    pv = python_pvector([1, 2, 3])
    result = TestVector(pv)
    assert result.tolist() == [1, 2, 3]
    assert isinstance(result, TestVector)


# LLM-generated content at query #4
#--------------------------

```python
def test_maybe_parse_user_type_with_preserved_iterable_type():
    result = maybe_parse_user_type(list)
    assert result == [list]

def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]

def test_maybe_parse_user_type_with_non_iterable_type():
    result = maybe_parse_user_type(int)
    assert result == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)

def test_maybe_parse_user_type_with_iterable_of_strings():
    result = maybe_parse_user_type(["int", "str"])
    assert result == ("int", "str")

def test_maybe_parse_user_type_with_nested_iterable():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)

def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_store_invariants_with_valid_invariants():
    class Base:
        def invariant(self):
            return True

    class Derived(Base):
        pass

    dct = {}
    bases = (Derived,)
    destination_name = 'invariants'
    source_name = 'invariant'

    store_invariants(dct, bases, destination_name, source_name)

    assert destination_name in dct
    assert len(dct[destination_name]) == 1
    assert callable(dct[destination_name][0])

def test_store_invariants_with_multiple_inherited_invariants():
    class Base1:
        def invariant1(self):
            return True

    class Base2:
        def invariant2(self):
            return True

    class Derived(Base1, Base2):
        pass

    dct = {}
    bases = (Derived,)
    destination_name = 'invariants'
    source_name = 'invariant'

    store_invariants(dct, bases, destination_name, source_name)

    assert destination_name in dct
    assert len(dct[destination_name]) == 2
    assert all(callable(inv) for inv in dct[destination_name])

def test_store_invariants_with_non_callable_invariant():
    class Base:
        invariant = True

    dct = {}
    bases = (Base,)
    destination_name = 'invariants'
    source_name = 'invariant'

    try:
        store_invariants(dct, bases, destination_name, source_name)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"

def test_store_invariants_with_no_invariants():
    class Base:
        pass

    dct = {}
    bases = (Base,)
    destination_name = 'invariants'
    source_name = 'invariant'

    store_invariants(dct, bases, destination_name, source_name)

    assert destination_name in dct
    assert len(dct[destination_name]) == 0

def test_store_invariants_with_direct_invariant():
    def invariant():
        return True

    dct = {'invariant': invariant}
    bases = ()
    destination_name = 'invariants'
    source_name = 'invariant'

    store_invariants(dct, bases, destination_name, source_name)

    assert destination_name in dct
    assert len(dct[destination_name]) == 1
    assert callable(dct[destination_name][0])


# LLM-generated content at query #6
#--------------------------

```python
def test_restore_pickle_creates_instance_with_factory_fields():
    class TestClass:
        @classmethod
        def create(cls, data, _factory_fields):
            assert _factory_fields == set()
            return cls()

    data = {"key": "value"}
    result = _restore_pickle(TestClass, data)
    assert isinstance(result, TestClass)


# LLM-generated content at query #7
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"a": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: "1.0"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5, initial={1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0}


# LLM-generated content at query #8
#--------------------------

```python
def test_empty_invariants():
    result = _invariant_errors("test", [])
    assert result == []

def test_all_invariants_pass():
    def always_valid(x):
        return True, "Valid"
    result = _invariant_errors("test", [always_valid, always_valid])
    assert result == []

def test_single_invariant_fails():
    def always_invalid(x):
        return False, "Invalid"
    result = _invariant_errors("test", [always_invalid])
    assert result == ["Invalid"]

def test_multiple_invariants_mixed():
    def valid(x):
        return True, "Valid"
    def invalid(x):
        return False, "Invalid"
    result = _invariant_errors("test", [valid, invalid, valid])
    assert result == ["Invalid"]

def test_invariant_with_different_data():
    def length_check(x):
        return len(x) > 3, f"Length is {len(x)}"
    result = _invariant_errors("hi", [length_check])
    assert result == ["Length is 2"]


# LLM-generated content at query #9
#--------------------------

```python
def test_invariant_errors_with_invalid_invariant():
    elem = "test"
    invariants = [lambda x: (False, "Error1"), lambda x: (True, None)]
    result = _invariant_errors(elem, invariants)
    assert result == ["Error1"]


# LLM-generated content at query #10
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"1": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: "1.0"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5, initial={1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0}


# LLM-generated content at query #11
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 1.5, 2: 2.25})
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) == 2
        assert all("Invalid mapping" in error for error in e.error_codes)

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap(size=5)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({"a": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({1: "a"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_checked_pvector_constructor_empty():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector()
    assert result.tolist() == []

def test_checked_pvector_constructor_with_list():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector([1, 2, 3])
    assert result.tolist() == [1, 2, 3]

def test_checked_pvector_constructor_with_pvector():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    pv = python_pvector([1, 2, 3])
    result = TestVector(pv)
    assert result.tolist() == [1, 2, 3]

def test_checked_pvector_constructor_type_error():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector([1, 2, "3"])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pvector_constructor_invariant_error():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector([1, 2, -3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = list
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_types():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = list
    try:
        _check_types(it, expected_types, source_class)
    except CheckedValueTypeError as e:
        assert str(e) == "Type list can only be used with ('int',), not str"

def test_check_types_with_multiple_valid_types():
    it = [1, 2.0, 3]
    expected_types = [int, float]
    source_class = list
    _check_types(it, expected_types, source_class)

def test_check_types_with_empty_iterable():
    it = []
    expected_types = [int]
    source_class = list
    _check_types(it, expected_types, source_class)

def test_check_types_with_empty_expected_types():
    it = [1, 2, 3]
    expected_types = []
    source_class = list
    _check_types(it, expected_types, source_class)

def test_check_types_with_custom_exception():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = list
    custom_exception = ValueError
    try:
        _check_types(it, expected_types, source_class, custom_exception)
    except ValueError as e:
        assert str(e) == "Type list can only be used with ('int',), not str"

def test_check_types_with_string_type_names():
    it = [1, 2, 3]
    expected_types = ['builtins.int']
    source_class = list
    _check_types(it, expected_types, source_class)


# LLM-generated content at query #14
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_initial_data():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    initial_data = {1: "one", 2: "two"}
    result = TestMap(initial_data)
    assert isinstance(result, TestMap)
    assert dict(result) == initial_data

def test_checked_pmap_constructor_with_size():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap(size=5)
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_invalid_key_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({"invalid": "value"})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({1: 123})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (v == k * 2, "Value must be double the key")

    try:
        TestMap({1: 3.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_valid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (v == k * 2, "Value must be double the key")

    result = TestMap({1: 2.0, 2: 4.0})
    assert isinstance(result, TestMap)
    assert dict(result) == {1: 2.0, 2: 4.0}


# LLM-generated content at query #15
#--------------------------

```python
def test_store_invariants_with_no_invariants():
    dct = {}
    bases = ()
    destination_name = 'invariants'
    source_name = 'invariant'
    store_invariants(dct, bases, destination_name, source_name)
    assert destination_name not in dct

def test_store_invariants_with_single_invariant():
    dct = {}
    bases = ()
    destination_name = 'invariants'
    source_name = 'invariant'
    dct[source_name] = lambda x: (True, x)
    store_invariants(dct, bases, destination_name, source_name)
    assert destination_name in dct
    assert len(dct[destination_name]) == 1
    assert callable(dct[destination_name][0])

def test_store_invariants_with_multiple_invariants():
    dct = {}
    bases = ()
    destination_name = 'invariants'
    source_name = 'invariant'
    dct[source_name] = lambda x: (True, x)
    dct[source_name + '2'] = lambda x: (False, x)
    store_invariants(dct, bases, destination_name, source_name)
    assert destination_name in dct
    assert len(dct[destination_name]) == 1

def test_store_invariants_with_inherited_invariants():
    class Base:
        invariant = lambda self: (True, self)

    class Derived(Base):
        pass

    dct = Derived.__dict__.copy()
    bases = Derived.__bases__
    destination_name = 'invariants'
    source_name = 'invariant'
    store_invariants(dct, bases, destination_name, source_name)
    assert destination_name in dct
    assert len(dct[destination_name]) == 1
    assert callable(dct[destination_name][0])

def test_store_invariants_with_non_callable_invariant():
    dct = {}
    bases = ()
    destination_name = 'invariants'
    source_name = 'invariant'
    dct[source_name] = "not callable"
    try:
        store_invariants(dct, bases, destination_name, source_name)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_store_invariants_with_multiple_inherited_invariants():
    class Base1:
        invariant = lambda self: (True, self)

    class Base2:
        invariant = lambda self: (False, self)

    class Derived(Base1, Base2):
        pass

    dct = Derived.__dict__.copy()
    bases = Derived.__bases__
    destination_name = 'invariants'
    source_name = 'invariant'
    store_invariants(dct, bases, destination_name, source_name)
    assert destination_name in dct
    assert len(dct[destination_name]) == 2
    assert all(callable(inv) for inv in dct[destination_name])


# LLM-generated content at query #16
#--------------------------

```python
def test_wrap_invariant_with_single_bool_result():
    def invariant():
        return (True, "data")
    wrapped = wrap_invariant(invariant)
    assert wrapped() == (True, "data")

def test_wrap_invariant_with_multiple_bool_results():
    def invariant():
        return [(True, "data1"), (False, "data2")]
    wrapped = wrap_invariant(invariant)
    assert wrapped() == (False, ("data2",))

def test_wrap_invariant_with_all_true_results():
    def invariant():
        return [(True, "data1"), (True, "data2")]
    wrapped = wrap_invariant(invariant)
    assert wrapped() == (True, tuple())

def test_wrap_invariant_with_empty_result_list():
    def invariant():
        return []
    wrapped = wrap_invariant(invariant)
    assert wrapped() == (True, tuple())

def test_wrap_invariant_with_args_and_kwargs():
    def invariant(arg1, arg2=None):
        return (arg2 is not None, arg1)
    wrapped = wrap_invariant(invariant)
    assert wrapped("test", arg2=True) == (True, "test")


# LLM-generated content at query #17
#--------------------------

```python
def test_store_invariants_with_valid_callable_invariants():
    dct = {'__invariants__': lambda: True}
    bases = ()
    destination_name = '__invariants__'
    source_name = '__invariants__'
    store_invariants(dct, bases, destination_name, source_name)
    assert dct[destination_name] == (lambda: True,)


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_18():
    assert maybe_parse_user_type(int) == [int]


# LLM-generated content at query #19
#--------------------------

```python
def test_is_preserved_predicate():
    class PreservedType(_preserved_iterable_types):
        pass

    t = PreservedType
    is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
    assert is_preserved


# LLM-generated content at query #20
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({"a": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({1: "not a float"})
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap(size=10)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


# LLM-generated content at query #21
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_initial_data():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap({1: "one", 2: "two"})
    assert isinstance(result, TestMap)
    assert dict(result) == {1: "one", 2: "two"}

def test_checked_pmap_constructor_with_size():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap(size=5)
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_invalid_key_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({"a": "one"})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({1: 1})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invariant_violation():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")

    try:
        TestMap({1: 0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_valid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")

    result = TestMap({1: 2, 3: 4})
    assert isinstance(result, TestMap)
    assert dict(result) == {1: 2, 3: 4}


# LLM-generated content at query #22
#--------------------------

```python
def test_empty_invariants():
    assert _invariant_errors("test", []) == []

def test_all_valid_invariants():
    def always_valid(x):
        return (True, "valid")
    assert _invariant_errors("test", [always_valid, always_valid]) == []

def test_single_invalid_invariant():
    def invalid(x):
        return (False, "invalid")
    assert _invariant_errors("test", [invalid]) == ["invalid"]

def test_mixed_invariants():
    def valid(x):
        return (True, "valid")
    def invalid(x):
        return (False, "invalid")
    assert _invariant_errors("test", [valid, invalid, valid]) == ["invalid"]


# LLM-generated content at query #23
#--------------------------

```python
def test_merge_invariant_results_all_passing():
    result = [(True, "data1"), (True, "data2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_some_failing():
    result = [(True, "data1"), (False, "data2"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("data2",)

def test_merge_invariant_results_all_failing():
    result = [(False, "data1"), (False, "data2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("data1", "data2")

def test_merge_invariant_results_empty_input():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


# LLM-generated content at query #24
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)


# LLM-generated content at query #25
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"a": 1.0, 2: 2.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: "1.0", 2: 2.0})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0, 2: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5, initial={1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}


# LLM-generated content at query #26
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({"a": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({1: "1.0"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap(size=5, initial={1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0}

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #27
#--------------------------

```python
def test__checked_type_create_with_instance_of_cls():
    class TestClass:
        _checked_types = []

    instance = TestClass()
    result = _checked_type_create(TestClass, instance)
    assert result is instance

def test__checked_type_create_with_non_instance_and_no_checked_types():
    class TestClass:
        _checked_types = []

    source_data = "test_data"
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == source_data

def test__checked_type_create_with_checked_type_in_list():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"checked_{data}"

    class TestClass:
        _checked_types = ["__main__.CheckedType"]

    source_data = ["data1", "data2"]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == ["checked_data1", "checked_data2"]

def test__checked_type_create_with_instance_in_checked_types():
    class CheckedType:
        pass

    class TestClass:
        _checked_types = ["__main__.CheckedType"]

    source_data = [CheckedType(), "other_data"]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == [CheckedType(), "other_data"]

def test__checked_type_create_with_ignore_extra():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"checked_{data}_{ignore_extra}"

    class TestClass:
        _checked_types = ["__main__.CheckedType"]

    source_data = ["data1", "data2"]
    result = _checked_type_create(TestClass, source_data, ignore_extra=True)
    assert isinstance(result, TestClass)
    assert result == ["checked_data1_True", "checked_data2_True"]


# LLM-generated content at query #28
#--------------------------

```python
def test_store_types_with_single_type():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = int
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int]

def test_store_types_with_multiple_types():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = [int, str]
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str]

def test_store_types_with_string_type():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = 'int'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == ['int']

def test_store_types_with_inherited_type():
    class Base:
        type = int
    dct = {}
    bases = [Base]
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int]

def test_store_types_with_mixed_types():
    class Base:
        type = int
    dct = {'type': str}
    bases = [Base]
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str]

def test_store_types_with_nested_iterable():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = [[int, str], float]
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str, float]

def test_store_types_with_invalid_type():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = 123
    try:
        _store_types(dct, bases, destination_name, source_name)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #29
#--------------------------

```python
def test_checked_type_create_with_checked_type_subclass():
    class TestCheckedType(CheckedType):
        pass

    class TestClass:
        _checked_types = (TestCheckedType,)

    source_data = [TestCheckedType(), TestCheckedType()]
    result = TestClass._checked_type_create(source_data)
    assert isinstance(result, TestClass)


# LLM-generated content at query #30
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0, 2: 3.0})
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) == 2
        assert all('Invalid mapping' in error for error in e.error_codes)

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=10)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_empty_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({'a': 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 'a'})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass


