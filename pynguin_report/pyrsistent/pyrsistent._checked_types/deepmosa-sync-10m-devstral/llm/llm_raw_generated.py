####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_empty_invariants():
    assert _invariant_errors("test", []) == []

def test_all_invariants_pass():
    def always_valid(x): return (True, "valid")
    assert _invariant_errors("test", [always_valid, always_valid]) == []

def test_single_invariant_fails():
    def always_invalid(x): return (False, "invalid")
    assert _invariant_errors("test", [always_invalid]) == ["invalid"]

def test_multiple_invariants_mixed_results():
    def valid_invariant(x): return (True, "valid")
    def invalid_invariant(x): return (False, "invalid")
    assert _invariant_errors("test", [valid_invariant, invalid_invariant, valid_invariant]) == ["invalid"]


# LLM-generated content at query #2
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
    store_invariants(dct, bases, "invariants", "invariant1")
    assert len(dct["invariants"]) == 1
    assert callable(dct["invariants"][0])

    store_invariants(dct, bases, "invariants", "invariant2")
    assert len(dct["invariants"]) == 1
    assert callable(dct["invariants"][0])

def test_store_invariants_with_non_callable_invariant():
    class Base:
        invariant = "not_callable"

    dct = {}
    bases = (Base,)
    try:
        store_invariants(dct, bases, "invariants", "invariant")
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"

def test_store_invariants_with_multiple_inherited_invariants():
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
    store_invariants(dct, bases, "invariants", "invariant1")
    store_invariants(dct, bases, "invariants", "invariant2")
    assert len(dct["invariants"]) == 2
    assert all(callable(inv) for inv in dct["invariants"])


# LLM-generated content at query #3
#--------------------------

```python
def test_checked_pvector_constructor_empty():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector()
    assert isinstance(result, TestVector)
    assert len(result) == 0

def test_checked_pvector_constructor_with_list():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector([1, 2, 3])
    assert isinstance(result, TestVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_checked_pvector_constructor_with_pvector():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    pv = python_pvector([1, 2, 3])
    result = TestVector(pv)
    assert isinstance(result, TestVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_checked_pvector_constructor_type_error():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector([1, 2, "3"])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pvector_constructor_invariant_error():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector([1, 2, -3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #4
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

    result = IntToFloatMap(size=10)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_empty_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


# LLM-generated content at query #5
#--------------------------

```python
def test_maybe_parse_user_type_with_preserved_type():
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
    assert result == [int, str]

def test_maybe_parse_user_type_with_nested_iterable():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == [int, str, float]

def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #6
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

def test_checked_pset_constructor_with_invalid_type():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestCheckedPSet(["a", "b", "c"])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pset_constructor_with_invalid_invariant():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestCheckedPSet([-1, -2, -3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_serialize_default_format():
    class TestSet(CheckedPSet):
        __type__ = (int,)
    test_instance = TestSet([1, 2, 3])
    result = test_instance.serialize()
    assert result == {1, 2, 3}

def test_serialize_with_custom_format():
    class TestSet(CheckedPSet):
        __type__ = (int,)
        __serializer__ = lambda fmt, x: str(x) if fmt == 'str' else x
    test_instance = TestSet([1, 2, 3])
    result = test_instance.serialize(format='str')
    assert result == {'1', '2', '3'}

def test_serialize_empty_set():
    class TestSet(CheckedPSet):
        __type__ = (int,)
    test_instance = TestSet()
    result = test_instance.serialize()
    assert result == set()


# LLM-generated content at query #8
#--------------------------

```python
def test__checked_type_create_with_already_correct_type():
    class TestType:
        _checked_types = []

    instance = TestType()
    result = _checked_type_create(TestType, instance)
    assert result is instance

def test__checked_type_create_without_checked_types():
    class TestType:
        _checked_types = []

    source_data = "test_data"
    result = _checked_type_create(TestType, source_data)
    assert isinstance(result, TestType)
    assert result == source_data

def test__checked_type_create_with_checked_type_in_list():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"checked_{data}"

    class TestType:
        _checked_types = ["__main__.CheckedType"]

    source_data = ["data1", "data2"]
    result = _checked_type_create(TestType, source_data)
    assert isinstance(result, TestType)
    assert result == ["checked_data1", "checked_data2"]

def test__checked_type_create_with_matching_type_in_list():
    class CheckedType:
        pass

    class TestType:
        _checked_types = ["__main__.CheckedType"]

    source_data = [CheckedType(), "data"]
    result = _checked_type_create(TestType, source_data)
    assert isinstance(result, TestType)
    assert result == source_data


# LLM-generated content at query #9
#--------------------------

```python
def test_maybe_parse_user_type_preserved():
    assert maybe_parse_user_type(_preserved_iterable_types) == [_preserved_iterable_types]


# LLM-generated content at query #10
#--------------------------

```python
def test_InvariantException___str__():
    exception = InvariantException(error_codes=("error1", "error2"), missing_fields=("field1", "field2"))
    assert str(exception) == ", invariant_errors=[error1, error2], missing_fields=[field1, field2]"


# LLM-generated content at query #11
#--------------------------

```python
def test_maybe_parse_user_type_with_preserved_iterable_type():
    assert maybe_parse_user_type(list) == [list]
    assert maybe_parse_user_type(tuple) == [tuple]
    assert maybe_parse_user_type(set) == [set]

def test_maybe_parse_user_type_with_string():
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("str") == ["str"]

def test_maybe_parse_user_type_with_non_iterable_type():
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(float) == [float]

def test_maybe_parse_user_type_with_iterable_of_types():
    assert maybe_parse_user_type([int, str]) == [int, str]
    assert maybe_parse_user_type((float, bool)) == [float, bool]

def test_maybe_parse_user_type_with_nested_iterable():
    assert maybe_parse_user_type([int, [str, float]]) == [int, str, float]
    assert maybe_parse_user_type((bool, (list, tuple))) == [bool, list, tuple]

def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    try:
        maybe_parse_user_type(None)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_store_invariants_non_callable_invariant():
    dct = {'source_name': 'not_callable'}
    bases = ()
    destination_name = 'destination'
    source_name = 'source_name'
    try:
        store_invariants(dct, bases, destination_name, source_name)
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        assert False, "TypeError not raised"


# LLM-generated content at query #13
#--------------------------

```python
def test_checkedpset_constructor_with_valid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    result = Positives([1, 2, 3])
    assert isinstance(result, Positives)
    assert result == pset([1, 2, 3])

def test_checkedpset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([1, '2', 3])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checkedpset_constructor_with_invalid_invariant():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([1, -2, 3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checkedpset_constructor_with_pmap_input():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    pmap_input = pmap({1: True, 2: True, 3: True})
    result = Positives(pmap_input)
    assert isinstance(result, Positives)
    assert result == pset([1, 2, 3])

def test_checkedpset_constructor_empty():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    result = Positives()
    assert isinstance(result, Positives)
    assert result == pset()

def test_checkedpset_constructor_with_duplicate_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    result = Positives([1, 2, 2, 3])
    assert isinstance(result, Positives)
    assert result == pset([1, 2, 3])


# LLM-generated content at query #14
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_types():
    it = [1, "2", 3]
    expected_types = [int]
    source_class = int
    try:
        _check_types(it, expected_types, source_class)
    except CheckedValueTypeError:
        pass

def test_check_types_with_empty_iterable():
    it = []
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_empty_expected_types():
    it = [1, 2, 3]
    expected_types = []
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_string_type_name():
    it = [1, 2, 3]
    expected_types = ["builtins.int"]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_multiple_valid_types():
    it = [1, 2.0, 3]
    expected_types = [int, float]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_custom_exception():
    it = [1, "2", 3]
    expected_types = [int]
    source_class = int
    try:
        _check_types(it, expected_types, source_class, exception_type=ValueError)
    except ValueError:
        pass


# LLM-generated content at query #15
#--------------------------

```python
def test_store_types_with_single_type_in_dict():
    dct = {'source': int}
    bases = []
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == [int]

def test_store_types_with_string_in_dict():
    dct = {'source': 'str'}
    bases = []
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == ['str']

def test_store_types_with_preserved_type_in_dict():
    dct = {'source': list}
    bases = []
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == [list]

def test_store_types_with_iterable_of_types_in_dict():
    dct = {'source': [int, str]}
    bases = []
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == [int, str]

def test_store_types_with_nested_iterable_of_types_in_dict():
    dct = {'source': [[int, str], list]}
    bases = []
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == [int, str, list]

def test_store_types_with_multiple_bases():
    dct = {}
    base1 = type('Base1', (), {'source': int})
    base2 = type('Base2', (), {'source': str})
    bases = [base1, base2]
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == [int, str]

def test_store_types_with_invalid_type_in_dict():
    dct = {'source': 123}
    bases = []
    try:
        _store_types(dct, bases, 'destination', 'source')
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_store_types_with_missing_source_in_dict_and_bases():
    dct = {}
    bases = []
    _store_types(dct, bases, 'destination', 'source')
    assert 'destination' not in dct


# LLM-generated content at query #16
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


# LLM-generated content at query #17
#--------------------------

```python
def test_store_invariants_empty():
    dct = {}
    bases = ()
    destination_name = "dest"
    source_name = "src"
    store_invariants(dct, bases, destination_name, source_name)
    assert dct == {}

def test_store_invariants_no_invariant_in_bases():
    dct = {}
    class Base:
        pass
    bases = (Base,)
    destination_name = "dest"
    source_name = "src"
    store_invariants(dct, bases, destination_name, source_name)
    assert dct == {}

def test_store_invariants_single_invariant_in_dct():
    dct = {"src": lambda x: (True, "data")}
    bases = ()
    destination_name = "dest"
    source_name = "src"
    store_invariants(dct, bases, destination_name, source_name)
    assert len(dct[destination_name]) == 1
    assert callable(dct[destination_name][0])

def test_store_invariants_multiple_invariants_in_bases():
    dct = {}
    class Base1:
        src = lambda x: (True, "data1")
    class Base2:
        src = lambda x: (False, "data2")
    bases = (Base1, Base2)
    destination_name = "dest"
    source_name = "src"
    store_invariants(dct, bases, destination_name, source_name)
    assert len(dct[destination_name]) == 2
    assert all(callable(inv) for inv in dct[destination_name])

def test_store_invariants_non_callable_invariant():
    dct = {"src": "not_callable"}
    bases = ()
    destination_name = "dest"
    source_name = "src"
    try:
        store_invariants(dct, bases, destination_name, source_name)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_store_invariants_inherited_invariants():
    dct = {}
    class Base:
        src = lambda x: (True, "base_data")
    class Derived(Base):
        pass
    bases = (Derived,)
    destination_name = "dest"
    source_name = "src"
    store_invariants(dct, bases, destination_name, source_name)
    assert len(dct[destination_name]) == 1
    assert callable(dct[destination_name][0])

def test_store_invariants_wrapped_invariant():
    dct = {"src": lambda x: [(True, "data1"), (False, "data2")]}
    bases = ()
    destination_name = "dest"
    source_name = "src"
    store_invariants(dct, bases, destination_name, source_name)
    wrapped = dct[destination_name][0]
    result = wrapped(None)
    assert result == (False, ("data2",))


# LLM-generated content at query #18
#--------------------------

```python
def test_checked_pset_constructor_with_valid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    result = Positives([1, 2, 3])
    assert isinstance(result, Positives)
    assert result == pset([1, 2, 3])

def test_checked_pset_constructor_with_pmap():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    pmap_instance = pmap({1: True, 2: True, 3: True})
    result = Positives(pmap_instance)
    assert isinstance(result, Positives)
    assert result == pset([1, 2, 3])

def test_checked_pset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([1, 'a', 3])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pset_constructor_with_invalid_invariant():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([1, -2, 3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pset_constructor_empty():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    result = Positives()
    assert isinstance(result, Positives)
    assert result == pset()

def test_checked_pset_constructor_with_duplicate_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    result = Positives([1, 2, 2, 3])
    assert isinstance(result, Positives)
    assert result == pset([1, 2, 3])


