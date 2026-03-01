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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_store_invariants_with_valid_callable_invariants():
    class Base:
        def invariant(self):
            return True, "data"

    class Derived(Base):
        pass

    dct = {}
    store_invariants(dct, (Derived,), "wrapped_invariants", "invariant")
    assert "wrapped_invariants" in dct
    assert len(dct["wrapped_invariants"]) == 1
    assert callable(dct["wrapped_invariants"][0])

def test_store_invariants_with_multiple_inherited_invariants():
    class Base1:
        def invariant(self):
            return True, "data1"

    class Base2:
        def invariant(self):
            return False, "data2"

    class Derived(Base1, Base2):
        pass

    dct = {}
    store_invariants(dct, (Derived,), "wrapped_invariants", "invariant")
    assert len(dct["wrapped_invariants"]) == 2

def test_store_invariants_with_non_callable_invariant():
    class Base:
        invariant = "not_callable"

    dct = {}
    try:
        store_invariants(dct, (Base,), "wrapped_invariants", "invariant")
    except TypeError as e:
        assert str(e) == "Invariants must be callable"
    else:
        assert False, "Expected TypeError"


# LLM-generated content at query #2
#--------------------------

```python
def test_maybe_parse_user_type_preserved_type():
    assert maybe_parse_user_type(list) == [list]

def test_maybe_parse_user_type_string():
    assert maybe_parse_user_type("int") == ["int"]

def test_maybe_parse_user_type_non_iterable_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_iterable_of_types():
    assert maybe_parse_user_type([int, str]) == [int, str]

def test_maybe_parse_user_type_nested_iterable_of_types():
    assert maybe_parse_user_type([int, [str, float]]) == [int, str, float]

def test_maybe_parse_user_type_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_checked_pset_new_with_empty_initial():
    result = CheckedPSet()
    assert isinstance(result, CheckedPSet)
    assert len(result) == 0

def test_checked_pset_new_with_iterable_initial():
    result = CheckedPSet([1, 2, 3])
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checked_pset_new_with_duplicate_elements():
    result = CheckedPSet([1, 2, 2, 3])
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checked_pset_new_with_pmap_initial():
    pmap_instance = pmap([(1, True), (2, True), (3, True)])
    result = CheckedPSet(pmap_instance)
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


# LLM-generated content at query #4
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
        pass

    class TestClass:
        _checked_types = ["module.CheckedType"]

    source_data = [{"key": "value"}]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert len(result) == 1
    assert isinstance(result[0], CheckedType)

def test__checked_type_create_with_ignore_extra():
    class CheckedType:
        pass

    class TestClass:
        _checked_types = ["module.CheckedType"]

    source_data = [{"key": "value", "extra": "data"}]
    result = _checked_type_create(TestClass, source_data, ignore_extra=True)
    assert isinstance(result, TestClass)
    assert len(result) == 1
    assert isinstance(result[0], CheckedType)


# LLM-generated content at query #5
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_items():
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
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
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


# LLM-generated content at query #6
#--------------------------

```python
def test_empty_invariants():
    assert _invariant_errors("test", []) == []

def test_all_invariants_pass():
    def always_valid(x):
        return (True, "valid")
    assert _invariant_errors("test", [always_valid, always_valid]) == []

def test_single_invariant_fails():
    def always_invalid(x):
        return (False, "invalid")
    assert _invariant_errors("test", [always_invalid]) == ["invalid"]

def test_multiple_invariants_mixed():
    def valid_invariant(x):
        return (True, "valid")
    def invalid_invariant(x):
        return (False, "invalid")
    assert _invariant_errors("test", [valid_invariant, invalid_invariant, valid_invariant]) == ["invalid"]

def test_multiple_invariants_all_fail():
    def invalid_invariant1(x):
        return (False, "error1")
    def invalid_invariant2(x):
        return (False, "error2")
    assert _invariant_errors("test", [invalid_invariant1, invalid_invariant2]) == ["error1", "error2"]


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


# LLM-generated content at query #8
#--------------------------

```python
def test_constructor_with_empty_initial():
    result = CheckedPVector()
    assert result.tolist() == []

def test_constructor_with_list_initial():
    result = CheckedPVector([1, 2, 3])
    assert result.tolist() == [1, 2, 3]

def test_constructor_with_python_pvector_initial():
    pv = python_pvector([1, 2, 3])
    result = CheckedPVector(pv)
    assert result.tolist() == [1, 2, 3]


# LLM-generated content at query #9
#--------------------------

```python
def test_maybe_parse_user_type_with_preserved_type():
    assert maybe_parse_user_type(list) == [list]

def test_maybe_parse_user_type_with_string():
    assert maybe_parse_user_type("int") == ["int"]

def test_maybe_parse_user_type_with_non_iterable_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    assert maybe_parse_user_type([int, str]) == [int, str]

def test_maybe_parse_user_type_with_nested_iterable():
    assert maybe_parse_user_type([int, [str, float]]) == [int, str, float]

def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_checked_pset_constructor_with_valid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    result = Positives([1, 2, 3])
    assert isinstance(result, Positives)
    assert result == pset([1, 2, 3])

def test_checked_pset_constructor_with_pmap_input():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    pmap_input = pmap({1: True, 2: True, 3: True})
    result = Positives(pmap_input)
    assert isinstance(result, Positives)
    assert result == pset([1, 2, 3])

def test_checked_pset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives(['a', 'b', 'c'])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pset_constructor_with_invalid_invariant():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([-1, -2, -3])
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

def test_checked_pset_constructor_with_mixed_valid_invalid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([1, -2, 3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


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


# LLM-generated content at query #12
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

def test_wrap_invariant_with_empty_result_list():
    def invariant():
        return []

    wrapped = wrap_invariant(invariant)
    assert wrapped() == (True, ())

def test_wrap_invariant_with_args_and_kwargs():
    def invariant(a, b, c=3):
        return [(a, b), (c,)]

    wrapped = wrap_invariant(invariant)
    assert wrapped(1, 2) == (True, ((1, 2), (3,)))
    assert wrapped(1, 2, c=4) == (True, ((1, 2), (4,)))


# LLM-generated content at query #13
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_types():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = int
    try:
        _check_types(it, expected_types, source_class)
    except CheckedValueTypeError as e:
        assert e.source_class == int
        assert e.expected_types == [int]
        assert e.actual_type == str
        assert e.value == 'a'
        assert "Type int can only be used with ('int',), not str" in str(e)

def test_check_types_with_empty_expected_types():
    it = [1, 'a', 3]
    expected_types = []
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_multiple_valid_types():
    it = [1, 2.0, 3]
    expected_types = [int, float]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_custom_exception():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = int
    custom_exception = Exception
    try:
        _check_types(it, expected_types, source_class, custom_exception)
    except Exception as e:
        assert isinstance(e, custom_exception)


# LLM-generated content at query #14
#--------------------------

```python
def test_maybe_parse_user_type_with_non_iterable_type():
    assert maybe_parse_user_type(int) == [int]


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

```python
def test_invariants_must_be_callable():
    class NonCallableInvariant:
        pass

    class TestClass:
        pass

    try:
        store_invariants(TestClass.__dict__, (TestClass,), 'destination', 'source')
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        assert False, "Expected TypeError was not raised"


# LLM-generated content at query #17
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)


# LLM-generated content at query #18
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


# LLM-generated content at query #19
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

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


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

    result = IntToFloatMap(size=5)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


# LLM-generated content at query #21
#--------------------------

```python
def test__invariant_errors_with_valid_data():
    elem = 5
    invariants = [lambda x: (x > 0, "Positive"), lambda x: (x < 10, "Less than 10")]
    result = _invariant_errors(elem, invariants)
    assert result == []


# LLM-generated content at query #22
#--------------------------

```python
def test_checked_type_create_with_checked_type_subclass():
    class TestCheckedType(CheckedType):
        _checked_types = {int}

    source_data = [1, 2, 3]
    result = TestCheckedType._checked_type_create(TestCheckedType, source_data)
    assert all(isinstance(item, int) for item in result)


# LLM-generated content at query #23
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

def test_store_types_with_string_type():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = 'str'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == ['str']

def test_store_types_with_iterable_types():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = [int, str]
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str]

def test_store_types_with_mixed_types():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = [int, 'str', float]
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, 'str', float]

def test_store_types_with_inherited_types():
    class Base:
        type = int
    dct = {}
    bases = [Base]
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int]

def test_store_types_with_multiple_inherited_types():
    class Base1:
        type = int
    class Base2:
        type = str
    dct = {}
    bases = [Base1, Base2]
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str]

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


# LLM-generated content at query #24
#--------------------------

```python
def test_checked_pvector_constructor_empty():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector()
    assert isinstance(result, TestVector)
    assert len(result) == 0

def test_checked_pvector_constructor_with_list():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector([1, 2, 3])
    assert isinstance(result, TestVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_checked_pvector_constructor_with_pvector():
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


# LLM-generated content at query #25
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
        Positives(['a', 'b', 'c'])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pset_constructor_with_invalid_invariant():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([-1, -2, -3])
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

def test_checked_pset_constructor_with_mixed_valid_invalid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([1, -2, 3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #26
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = type(it)
    assert _check_types(it, expected_types, source_class) is None


# LLM-generated content at query #27
#--------------------------

```python
def test__check_types_with_valid_types():
    class TestClass: pass
    it = [1, 2, 3]
    expected_types = [int]
    _check_types(it, expected_types, TestClass)

def test__check_types_with_invalid_type():
    class TestClass: pass
    it = [1, "a", 3]
    expected_types = [int]
    try:
        _check_types(it, expected_types, TestClass)
    except CheckedValueTypeError as e:
        assert e.source_class == TestClass
        assert e.expected_types == [int]
        assert e.actual_type == str
        assert e.value == "a"
        assert "Type TestClass can only be used with ('int',), not str" in str(e)

def test__check_types_with_multiple_valid_types():
    class TestClass: pass
    it = [1, 2.0, 3]
    expected_types = [int, float]
    _check_types(it, expected_types, TestClass)

def test__check_types_with_custom_exception():
    class TestClass: pass
    class CustomException(Exception): pass
    it = [1, "a", 3]
    expected_types = [int]
    try:
        _check_types(it, expected_types, TestClass, CustomException)
    except CustomException as e:
        assert e.source_class == TestClass
        assert e.expected_types == [int]
        assert e.actual_type == str
        assert e.value == "a"
        assert "Type TestClass can only be used with ('int',), not str" in str(e)

def test__check_types_with_empty_iterable():
    class TestClass: pass
    it = []
    expected_types = [int]
    _check_types(it, expected_types, TestClass)

def test__check_types_with_empty_expected_types():
    class TestClass: pass
    it = [1, "a", 3]
    expected_types = []
    _check_types(it, expected_types, TestClass)


# LLM-generated content at query #28
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

    pmap_instance = pmap({1: True, 2: True, 3: True})
    result = TestCheckedPSet(pmap_instance)
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


# LLM-generated content at query #29
#--------------------------

```python
def test_maybe_parse_user_type_preserved_type():
    class PreservedType(_preserved_iterable_types):
        pass
    result = maybe_parse_user_type(PreservedType)
    assert result == [PreservedType]


# LLM-generated content at query #30
#--------------------------

```python
def test_is_preserved_predicate():
    class PreservedType(_preserved_iterable_types):
        pass

    assert isinstance(PreservedType, type)
    assert issubclass(PreservedType, _preserved_iterable_types)


# LLM-generated content at query #31
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    assert _check_types(it, expected_types, source_class) is None


# LLM-generated content at query #32
#--------------------------

```python
def test_isinstance_check():
    result = (True,)
    assert isinstance(result[0], bool)


# LLM-generated content at query #33
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

def test_checked_pmap_constructor_with_empty_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}

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
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


# LLM-generated content at query #34
#--------------------------

```python
def test_store_invariants_with_valid_callable_invariants():
    class Base:
        def invariant1(self):
            return True, "data1"

    class Derived(Base):
        def invariant2(self):
            return True, "data2"

    dct = {}
    bases = (Derived,)
    store_invariants(dct, bases, "invariants", "invariant1")
    assert "invariants" in dct
    assert len(dct["invariants"]) == 1
    assert callable(dct["invariants"][0])

def test_store_invariants_with_inherited_invariants():
    class Base:
        def invariant1(self):
            return True, "data1"

    class Derived(Base):
        def invariant2(self):
            return True, "data2"

    dct = {}
    bases = (Derived,)
    store_invariants(dct, bases, "invariants", "invariant1")
    store_invariants(dct, bases, "invariants", "invariant2")
    assert len(dct["invariants"]) == 2
    assert all(callable(inv) for inv in dct["invariants"])

def test_store_invariants_with_non_callable_invariant():
    class Base:
        invariant1 = "not_callable"

    dct = {}
    bases = (Base,)
    try:
        store_invariants(dct, bases, "invariants", "invariant1")
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"

def test_store_invariants_with_no_invariants():
    class Base:
        pass

    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, "invariants", "nonexistent")
    assert "invariants" in dct
    assert len(dct["invariants"]) == 0

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
    store_invariants(dct, bases, "invariants", "invariant1")
    store_invariants(dct, bases, "invariants", "invariant2")
    assert len(dct["invariants"]) == 2
    assert all(callable(inv) for inv in dct["invariants"])


# LLM-generated content at query #35
#--------------------------

```python
def test_checked_type_create_returns_source_data_when_isinstance_of_cls():
    class TestClass:
        pass

    instance = TestClass()
    result = _checked_type_create(TestClass, instance)
    assert result is instance


# LLM-generated content at query #36
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


# LLM-generated content at query #38
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

def test_constructor_with_checked_pvector_initial():
    initial = CheckedPVector([1, 2, 3])
    result = CheckedPVector(initial)
    assert isinstance(result, CheckedPVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #39
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

def test_store_types_with_iterable():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = [int, str]
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str]

def test_store_types_with_nested_iterable():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = [[int, str], 'float']
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str, 'float']

def test_store_types_with_inherited_type():
    class Base:
        type = int
    dct = {}
    bases = [Base]
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int]

def test_store_types_with_inherited_and_own_type():
    class Base:
        type = int
    dct = {'type': str}
    bases = [Base]
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str]

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


# LLM-generated content at query #40
#--------------------------

```python
def test_invariant_errors_with_no_invariants():
    assert _invariant_errors("test", []) == []


# LLM-generated content at query #41
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = type(it)
    assert _check_types(it, expected_types, source_class) is None


# LLM-generated content at query #42
#--------------------------

```python
def test_invariant_errors_with_invalid_invariant():
    elem = "test"
    invariants = [lambda x: (False, "error")]
    assert _invariant_errors(elem, invariants) == ["error"]


# LLM-generated content at query #43
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


# LLM-generated content at query #44
#--------------------------

```python
def test_checked_type_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)


# LLM-generated content at query #45
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


# LLM-generated content at query #46
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

def test_checked_pmap_constructor_with_invalid_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"1": 1.0, 2: 2.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 1.5, 2: 2.25})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=10, initial={1: 1.0, 2: 2.0})
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


# LLM-generated content at query #47
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
        IntToFloatMap({"1": 1.0, 2: 2.0})
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

    result = IntToFloatMap(size=2, initial={1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}


# LLM-generated content at query #48
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


# LLM-generated content at query #49
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


# LLM-generated content at query #50
#--------------------------

```python
def test_checked_type_create_predicate_false():
    class TestClass:
        pass

    source_data = TestClass()
    assert not isinstance(source_data, TestClass)


# LLM-generated content at query #51
#--------------------------

```python
def test_maybe_parse_user_type_preserved():
    class PreservedType(_preserved_iterable_types):
        pass
    result = maybe_parse_user_type(PreservedType)
    assert result == [PreservedType]


# LLM-generated content at query #52
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
    assert repr(result) == "IntToFloatMap({1: 1.0, 2: 2.0})"

def test_checked_pmap_constructor_with_invalid_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"a": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
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

def test_checked_pmap_constructor_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #53
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

def test_store_types_with_string_type():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = 'int'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == ['int']

def test_store_types_with_iterable_types():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = [int, str]
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str]

def test_store_types_with_multiple_bases():
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

def test_store_types_with_mixed_types():
    dct = {}
    class Base:
        type = 'float'
    bases = [Base]
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = [int, str]
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str, 'float']

def test_store_types_with_no_source_name():
    dct = {}
    bases = []
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert destination_name not in dct


# LLM-generated content at query #54
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_items():
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


# LLM-generated content at query #55
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


# LLM-generated content at query #56
#--------------------------

```python
def test_store_invariants_with_valid_callable_invariants():
    class Base:
        def invariant_1(self):
            return True

    class Derived(Base):
        def invariant_2(self):
            return True

    dct = {}
    bases = (Derived,)
    store_invariants(dct, bases, 'invariants', 'invariant_1')
    assert 'invariants' in dct
    assert len(dct['invariants']) == 1
    assert all(callable(inv) for inv in dct['invariants'])

def test_store_invariants_with_non_callable_invariant():
    class Base:
        invariant_1 = True

    dct = {}
    bases = (Base,)
    try:
        store_invariants(dct, bases, 'invariants', 'invariant_1')
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test_store_invariants_with_multiple_inherited_invariants():
    class Base1:
        def invariant_1(self):
            return True

    class Base2:
        def invariant_2(self):
            return True

    class Derived(Base1, Base2):
        pass

    dct = {}
    bases = (Derived,)
    store_invariants(dct, bases, 'invariants', 'invariant_1')
    assert 'invariants' in dct
    assert len(dct['invariants']) == 1

def test_store_invariants_with_no_invariants():
    class Base:
        pass

    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, 'invariants', 'invariant_1')
    assert 'invariants' in dct
    assert len(dct['invariants']) == 0

def test_store_invariants_with_wrapped_invariants():
    class Base:
        def invariant_1(self):
            return [(True, 'data1'), (False, 'data2')]

    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, 'invariants', 'invariant_1')
    assert 'invariants' in dct
    assert len(dct['invariants']) == 1
    wrapped_invariant = dct['invariants'][0]
    result = wrapped_invariant()
    assert result == (False, ('data2',))


# LLM-generated content at query #57
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


# LLM-generated content at query #58
#--------------------------

```python
def test_invariant_result_not_merged_when_first_element_is_bool():
    def dummy_invariant():
        return (True, "some message")

    wrapped = wrap_invariant(dummy_invariant)
    result = wrapped()

    assert result == (True, "some message")


# LLM-generated content at query #59
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


# LLM-generated content at query #60
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    assert CheckedPSet() == pset()

def test_checked_pset_constructor_with_list_initial():
    assert CheckedPSet([1, 2, 3]) == pset([1, 2, 3])

def test_checked_pset_constructor_with_pset_initial():
    assert CheckedPSet(pset([1, 2, 3])) == pset([1, 2, 3])

def test_checked_pset_constructor_with_pmap_initial():
    assert CheckedPSet(pmap({1: True, 2: True, 3: True})) == pset([1, 2, 3])

def test_checked_pset_constructor_with_duplicate_elements():
    assert CheckedPSet([1, 2, 2, 3]) == pset([1, 2, 3])

def test_checked_pset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([1, 2, 'a'])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pset_constructor_with_invalid_invariant():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([1, 2, -3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pset_constructor_with_valid_types_and_invariants():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    assert Positives([1, 2, 3]) == pset([1, 2, 3])
    assert Positives([1.0, 2.0, 3.0]) == pset([1.0, 2.0, 3.0])


# LLM-generated content at query #61
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


# LLM-generated content at query #62
#--------------------------

```python
def test__checked_type_create_with_instance_of_cls():
    class TestClass:
        pass
    instance = TestClass()
    assert _checked_type_create(TestClass, instance) is instance

def test__checked_type_create_with_non_instance_and_no_checked_types():
    class TestClass:
        pass
    source_data = "test"
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == source_data

def test__checked_type_create_with_checked_type_in_source_data():
    class TestClass:
        _checked_types = ['some.module.CheckedType']
    source_data = ["test", "data"]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == source_data

def test__checked_type_create_with_checked_type_not_in_source_data():
    class TestClass:
        _checked_types = ['some.module.CheckedType']
    source_data = ["test", "data"]
    with patch('some.module.CheckedType.create') as mock_create:
        mock_create.return_value = "created"
        result = _checked_type_create(TestClass, source_data)
        assert isinstance(result, TestClass)
        assert result == ["created", "created"]
        assert mock_create.call_count == 2


# LLM-generated content at query #63
#--------------------------

```python
def test_checked_type_create_with_non_cls_instance():
    class TestClass:
        pass

    source_data = "not an instance of TestClass"
    result = _checked_type_create(TestClass, source_data)
    assert result == TestClass(source_data)


# LLM-generated content at query #64
#--------------------------

```python
def test_isinstance_predicate():
    class TestClass:
        pass

    instance = TestClass()
    assert isinstance(instance, TestClass)


# LLM-generated content at query #65
#--------------------------

```python
def test_check_types_with_valid_types():
    class TestClass: pass
    items = [1, 2, 3]
    expected_types = [int]
    _check_types(items, expected_types, TestClass)

def test_check_types_with_invalid_types():
    class TestClass: pass
    items = [1, 'a', 3]
    expected_types = [int]
    try:
        _check_types(items, expected_types, TestClass)
    except CheckedValueTypeError as e:
        assert str(e) == "Type TestClass can only be used with ('int',), not str"

def test_check_types_with_multiple_valid_types():
    class TestClass: pass
    items = [1, 2.0, 3]
    expected_types = [int, float]
    _check_types(items, expected_types, TestClass)

def test_check_types_with_empty_expected_types():
    class TestClass: pass
    items = [1, 'a', 3.0]
    expected_types = []
    _check_types(items, expected_types, TestClass)

def test_check_types_with_custom_exception():
    class TestClass: pass
    class CustomException(Exception): pass
    items = [1, 'a', 3]
    expected_types = [int]
    try:
        _check_types(items, expected_types, TestClass, CustomException)
    except CustomException as e:
        assert str(e) == "Type TestClass can only be used with ('int',), not str"


# LLM-generated content at query #66
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

def test_checked_pmap_constructor_empty_initial():
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
def test_merge_invariant_results_all_true():
    result = [(True, "data1"), (True, "data2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_all_false():
    result = [(False, "data1"), (False, "data2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("data1", "data2")

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


# LLM-generated content at query #2
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

    result = IntToFloatMap(size=2, initial={1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0}

def test_checked_pmap_constructor_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #3
#--------------------------

```python
def test_empty_invariants():
    assert _invariant_errors("test", []) == []

def test_all_invariants_pass():
    def always_valid(x):
        return (True, "valid")
    assert _invariant_errors("test", [always_valid, always_valid]) == []

def test_single_invariant_fails():
    def always_invalid(x):
        return (False, "invalid")
    assert _invariant_errors("test", [always_invalid]) == ["invalid"]

def test_multiple_invariants_some_fail():
    def valid(x):
        return (True, "valid")
    def invalid(x):
        return (False, "invalid")
    result = _invariant_errors("test", [valid, invalid, valid])
    assert result == ["invalid"]

def test_multiple_invariants_all_fail():
    def invalid1(x):
        return (False, "error1")
    def invalid2(x):
        return (False, "error2")
    result = _invariant_errors("test", [invalid1, invalid2])
    assert sorted(result) == ["error1", "error2"]


# LLM-generated content at query #4
#--------------------------

```python
def test_get_type_with_type_instance():
    assert get_type(int) == int
    assert get_type(str) == str
    assert get_type(list) == list

def test_get_type_with_string_type_name():
    assert get_type('builtins.int') == int
    assert get_type('builtins.str') == str
    assert get_type('builtins.list') == list

def test_get_type_with_custom_class_string():
    assert get_type('collections.OrderedDict') == __import__('collections', fromlist=['OrderedDict']).OrderedDict


# LLM-generated content at query #5
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
    store_invariants(dct, bases, "wrapped_invariants", "invariant1")
    assert len(dct["wrapped_invariants"]) == 1
    assert callable(dct["wrapped_invariants"][0])

    store_invariants(dct, bases, "wrapped_invariants", "invariant2")
    assert len(dct["wrapped_invariants"]) == 1
    assert callable(dct["wrapped_invariants"][0])

def test_store_invariants_with_inherited_invariants():
    class Base:
        def invariant(self):
            return True, "base_data"

    class Derived(Base):
        pass

    dct = {}
    bases = (Derived,)
    store_invariants(dct, bases, "wrapped_invariants", "invariant")
    assert len(dct["wrapped_invariants"]) == 1
    assert callable(dct["wrapped_invariants"][0])

def test_store_invariants_with_multiple_inherited_invariants():
    class Base1:
        def invariant(self):
            return True, "base1_data"

    class Base2:
        def invariant(self):
            return True, "base2_data"

    class Derived(Base1, Base2):
        pass

    dct = {}
    bases = (Derived,)
    store_invariants(dct, bases, "wrapped_invariants", "invariant")
    assert len(dct["wrapped_invariants"]) == 2
    assert all(callable(inv) for inv in dct["wrapped_invariants"])

def test_store_invariants_with_non_callable_invariant():
    class Base:
        invariant = "not_callable"

    dct = {}
    bases = (Base,)
    try:
        store_invariants(dct, bases, "wrapped_invariants", "invariant")
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"

def test_store_invariants_with_no_invariants():
    class Base:
        pass

    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, "wrapped_invariants", "invariant")
    assert "wrapped_invariants" not in dct

def test_store_invariants_with_wrapped_invariant_execution():
    class Base:
        def invariant(self):
            return [(True, "data1"), (False, "data2")]

    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, "wrapped_invariants", "invariant")
    wrapped = dct["wrapped_invariants"][0]
    result = wrapped()
    assert result == (False, ("data2",))


# LLM-generated content at query #6
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
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
    except InvariantException as e:
        assert len(e.error_codes) == 2


# LLM-generated content at query #7
#--------------------------

```python
def test_checked_type_instantiation():
    with pytest.raises(TypeError):
        CheckedType()


# LLM-generated content at query #8
#--------------------------

```python
def test_constructor_with_empty_initial():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector()
    assert isinstance(result, TestVector)
    assert len(result) == 0

def test_constructor_with_list_initial():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector([1, 2, 3])
    assert isinstance(result, TestVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_constructor_with_pvector_initial():
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

def test_constructor_with_invalid_type():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector(["invalid"])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_constructor_with_invalid_invariant():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector([-1])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #9
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

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #10
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

def test_checked_pmap_constructor_with_invariant_violation():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (v == k * 2.0, "Value must be twice the key")

    try:
        TestMap({1: 3.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_valid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (v == k * 2.0, "Value must be twice the key")

    result = TestMap({1: 2.0, 2: 4.0})
    assert isinstance(result, TestMap)
    assert dict(result) == {1: 2.0, 2: 4.0}


# LLM-generated content at query #11
#--------------------------

```python
def test_store_invariants_predicate():
    dct = {'source': lambda x: x}
    bases = []
    destination_name = 'destination'
    source_name = 'source'
    assert all(callable(invariant) for invariant in [dct[source_name]])


# LLM-generated content at query #12
#--------------------------

```python
def test__check_types_with_correct_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)

def test__check_types_with_incorrect_types():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = int
    try:
        _check_types(it, expected_types, source_class)
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.source_class == int
        assert e.expected_types == [int]
        assert e.actual_type == str
        assert e.value == 'a'

def test__check_types_with_multiple_expected_types():
    it = [1, 2.0, 3]
    expected_types = [int, float]
    source_class = int
    _check_types(it, expected_types, source_class)

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

def test__check_types_with_string_type_names():
    it = [1, 2, 3]
    expected_types = ['builtins.int']
    source_class = int
    _check_types(it, expected_types, source_class)

def test__check_types_with_custom_exception():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = int
    try:
        _check_types(it, expected_types, source_class, ValueError)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test_maybe_parse_user_type_with_preserved_iterable_type():
    assert maybe_parse_user_type(list) == [list]

def test_maybe_parse_user_type_with_string():
    assert maybe_parse_user_type("str") == ["str"]

def test_maybe_parse_user_type_with_non_iterable_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    assert maybe_parse_user_type([int, str]) == [int, str]

def test_maybe_parse_user_type_with_nested_iterable_of_types():
    assert maybe_parse_user_type([[int, str], float]) == [int, str, float]

def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_18():
    assert maybe_parse_user_type(int) == [int]


# LLM-generated content at query #15
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

def test_checked_pmap_constructor_empty_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


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
def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"a": 1.0, 2: 2.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 1.5, 2: 2.0})
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

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #19
#--------------------------

```python
def test_restore_pickle_returns_instance():
    class TestClass:
        @classmethod
        def create(cls, data, _factory_fields):
            return cls()

    data = {"key": "value"}
    result = _restore_pickle(TestClass, data)
    assert isinstance(result, TestClass)

def test_restore_pickle_calls_create_with_data():
    class TestClass:
        created_with = None

        @classmethod
        def create(cls, data, _factory_fields):
            cls.created_with = data
            return cls()

    data = {"key": "value"}
    _restore_pickle(TestClass, data)
    assert TestClass.created_with == data

def test_restore_pickle_passes_empty_factory_fields():
    class TestClass:
        factory_fields = None

        @classmethod
        def create(cls, data, _factory_fields):
            cls.factory_fields = _factory_fields
            return cls()

    data = {"key": "value"}
    _restore_pickle(TestClass, data)
    assert TestClass.factory_fields == set()


# LLM-generated content at query #20
#--------------------------

```python
def test__checked_type_create_with_instance_of_cls():
    class TestClass:
        pass
    instance = TestClass()
    assert _checked_type_create(TestClass, instance) is instance

def test__checked_type_create_with_non_instance_and_no_checked_types():
    class TestClass:
        def __init__(self, data):
            self.data = data
    result = _checked_type_create(TestClass, "test_data")
    assert isinstance(result, TestClass)
    assert result.data == "test_data"

def test__checked_type_create_with_checked_type_in_list():
    class CheckedType:
        _checked_types = ["module.SubCheckedType"]

        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)

    class SubCheckedType(CheckedType):
        pass

    class TestClass:
        _checked_types = [CheckedType]

    result = _checked_type_create(TestClass, ["data1", "data2"], ignore_extra=True)
    assert isinstance(result, TestClass)
    assert all(isinstance(item, CheckedType) for item in result)

def test__checked_type_create_with_matching_type_in_list():
    class CheckedType:
        pass

    class TestClass:
        _checked_types = [CheckedType]

    data = [CheckedType(), "other_data"]
    result = _checked_type_create(TestClass, data)
    assert isinstance(result, TestClass)
    assert result == data


# LLM-generated content at query #21
#--------------------------

```python
def test_store_invariants_with_non_callable_invariant():
    dct = {'invariant': 'not_callable'}
    bases = []
    destination_name = 'destination'
    source_name = 'invariant'

    with pytest.raises(TypeError) as excinfo:
        store_invariants(dct, bases, destination_name, source_name)

    assert str(excinfo.value) == 'Invariants must be callable'


# LLM-generated content at query #22
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    assert _check_types(it, expected_types, source_class) is None


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_18():
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    assert maybe_parse_user_type(float) == [float]


# LLM-generated content at query #24
#--------------------------

```python
def test_checked_pmap_new_with_valid_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.5, 2: 2.25}

def test_checked_pmap_new_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({"a": 1.5, 2: 2.25})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_new_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({1: "1.5", 2: 2.25})
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
        pass

def test_checked_pmap_new_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.5, 2: 2.25})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_new_with_empty_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}

def test_checked_pmap_new_with_predefined_size():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=16)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.5, 2: 2.25}


# LLM-generated content at query #25
#--------------------------

```python
def test_wrap_invariant_with_single_bool_result():
    def mock_invariant():
        return (True, "data")

    wrapped = wrap_invariant(mock_invariant)
    result = wrapped()
    assert result == (True, "data")

def test_wrap_invariant_with_multiple_bool_results():
    def mock_invariant():
        return [(True, "data1"), (False, "data2"), (True, "data3")]

    wrapped = wrap_invariant(mock_invariant)
    result = wrapped()
    assert result == (False, ("data2",))

def test_wrap_invariant_with_all_true_results():
    def mock_invariant():
        return [(True, "data1"), (True, "data2")]

    wrapped = wrap_invariant(mock_invariant)
    result = wrapped()
    assert result == (True, ())

def test_wrap_invariant_with_all_false_results():
    def mock_invariant():
        return [(False, "data1"), (False, "data2")]

    wrapped = wrap_invariant(mock_invariant)
    result = wrapped()
    assert result == (False, ("data1", "data2"))


# LLM-generated content at query #26
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = list
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_types():
    it = [1, "a", 3]
    expected_types = [int]
    source_class = list
    try:
        _check_types(it, expected_types, source_class)
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == [int]
        assert e.actual_type == str
        assert e.value == "a"
        assert "Type list can only be used with ('int',), not str" in str(e)

def test_check_types_with_empty_iterable():
    it = []
    expected_types = [int]
    source_class = list
    _check_types(it, expected_types, source_class)

def test_check_types_with_multiple_valid_types():
    it = [1, 2.0, 3]
    expected_types = [int, float]
    source_class = list
    _check_types(it, expected_types, source_class)

def test_check_types_with_multiple_invalid_types():
    it = [1, "a", 3.0, None]
    expected_types = [int, float]
    source_class = list
    try:
        _check_types(it, expected_types, source_class)
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == [int, float]
        assert e.actual_type == str
        assert e.value == "a"
        assert "Type list can only be used with ('int', 'float'), not str" in str(e)

def test_check_types_with_custom_exception():
    it = [1, "a", 3]
    expected_types = [int]
    source_class = list
    custom_exception = ValueError
    try:
        _check_types(it, expected_types, source_class, custom_exception)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Type list can only be used with ('int',), not str" in str(e)


# LLM-generated content at query #27
#--------------------------

```python
def test_checked_pmap_new_without_size_uses_evolver():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap({1: "a", 2: "b"})
    assert result == TestMap({1: "a", 2: "b"})


# LLM-generated content at query #28
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    result = CheckedPSet()
    assert isinstance(result, CheckedPSet)
    assert len(result) == 0

def test_checked_pset_constructor_with_list_initial():
    result = CheckedPSet([1, 2, 3])
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checked_pset_constructor_with_pset_initial():
    initial = pset([1, 2, 3])
    result = CheckedPSet(initial)
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checked_pset_constructor_with_pmap_initial():
    initial = pmap({1: True, 2: True, 3: True})
    result = CheckedPSet(initial)
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checked_pset_constructor_with_invalid_initial_type():
    try:
        CheckedPSet("invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pset_constructor_with_invalid_element_type():
    class StrictSet(CheckedPSet):
        __type__ = int

    try:
        StrictSet([1, 2, "invalid"])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pset_constructor_with_invalid_invariant():
    class PositiveSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x >= 0, "Negative")

    try:
        PositiveSet([1, 2, -3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #29
#--------------------------

```python
def test__invariant_errors_returns_empty_list_when_all_invariants_valid():
    elem = object()
    invariants = [lambda x: (True, None), lambda x: (True, None)]
    assert _invariant_errors(elem, invariants) == []


# LLM-generated content at query #30
#--------------------------

```python
def test_is_preserved_evaluates_to_true():
    class PreservedType(_preserved_iterable_types):
        pass

    t = PreservedType
    is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
    assert is_preserved is True


# LLM-generated content at query #31
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
        IntToFloatMap({'a': 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 'a'})
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


# LLM-generated content at query #32
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
        IntToFloatMap({"1": 1.0, 2: 2.0})
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

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #33
#--------------------------

```python
def test_restore_pickle_returns_instance_with_factory_fields():
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


# LLM-generated content at query #34
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
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_wrong_types():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"a": 1.0, 2: "b"})
        assert False, "Expected CheckedKeyTypeError or CheckedValueTypeError"
    except (CheckedKeyTypeError, CheckedValueTypeError):
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


# LLM-generated content at query #35
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_items():
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

def test_checked_pmap_constructor_with_invalid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")

    try:
        TestMap({2: 1})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #36
#--------------------------

```python
def test_checked_type_create_with_checked_type_subclass():
    class TestCheckedType(CheckedType):
        pass

    class TestClass:
        _checked_types = [TestCheckedType]

    source_data = [TestCheckedType(), TestCheckedType()]
    result = _checked_type_create(TestClass, source_data)
    assert result == TestClass(source_data)


# LLM-generated content at query #37
#--------------------------

```python
def test_checked_pmap_new_without_size_uses_evolver():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    initial = {1: "a", 2: "b"}
    result = TestMap(initial)
    assert isinstance(result, TestMap)
    assert dict(result) == initial


# LLM-generated content at query #38
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = type(it)
    assert _check_types(it, expected_types, source_class) is None


# LLM-generated content at query #39
#--------------------------

```python
def test_empty_invariants():
    assert _invariant_errors("test", []) == []

def test_no_errors():
    assert _invariant_errors(5, [lambda x: (True, "error1"), lambda x: (True, "error2")]) == []

def test_single_error():
    assert _invariant_errors(5, [lambda x: (True, "error1"), lambda x: (False, "error2")]) == ["error2"]

def test_multiple_errors():
    assert _invariant_errors(5, [lambda x: (False, "error1"), lambda x: (False, "error2")]) == ["error1", "error2"]

def test_mixed_errors():
    assert _invariant_errors(5, [lambda x: (False, "error1"), lambda x: (True, "error2"), lambda x: (False, "error3")]) == ["error1", "error3"]


# LLM-generated content at query #40
#--------------------------

```python
def test_merge_invariant_results_with_false_verdict():
    result = [(False, "data1"), (True, "data2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False


# LLM-generated content at query #41
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    assert _check_types(it, expected_types, source_class) is None


# LLM-generated content at query #42
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

def test_checked_pmap_constructor_with_invalid_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"a": 1.0, 2: 2.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 1.5, 2: 2.0})
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


# LLM-generated content at query #43
#--------------------------

```python
def test_is_preserved_predicate():
    class PreservedType(_preserved_iterable_types):
        pass

    assert isinstance(PreservedType, type) and issubclass(PreservedType, _preserved_iterable_types)


# LLM-generated content at query #44
#--------------------------

```python
def test_checked_type_create_predicate_false():
    class MockCheckedType(CheckedType):
        pass

    class MockClass:
        _checked_types = (MockCheckedType,)
        def __init__(self, data):
            self.data = data

    source_data = [1, 2, 3]
    assert not isinstance(source_data, MockClass)


# LLM-generated content at query #45
#--------------------------

```python
def test_checked_type_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)


# LLM-generated content at query #46
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

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #47
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
        IntToFloatMap({1: 1.5, 3: 2.25})
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

def test_checked_pmap_constructor_with_empty_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({'a': 1.5, 2: 2.25})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: '1.5', 2: 2.25})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass


# LLM-generated content at query #48
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

def test_store_types_with_base_class_type():
    dct = {}
    class Base:
        type = float
    bases = [Base]
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [float]

def test_store_types_with_mixed_types():
    dct = {}
    class Base:
        type = [bool, list]
    bases = [Base]
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = dict
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [bool, list, dict]

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


# LLM-generated content at query #49
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = type(it)
    _check_types(it, expected_types, source_class)


# LLM-generated content at query #50
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


# LLM-generated content at query #51
#--------------------------

```python
def test_predicate_evaluates_to_true():
    dct = {'source_name': 'value'}
    bases = []
    source_name = 'source_name'
    assert source_name in dct


# LLM-generated content at query #52
#--------------------------

```python
def test_invariant_errors_with_valid_invariants():
    elem = "test"
    invariants = [lambda x: (True, "data1"), lambda x: (True, "data2")]
    assert _invariant_errors(elem, invariants) == []


# LLM-generated content at query #53
#--------------------------

```python
def test_checked_type_create_returns_source_data_when_input_is_instance_of_cls():
    class TestClass:
        _checked_types = []

    source_data = TestClass()
    result = _checked_type_create(TestClass, source_data)
    assert result is source_data


# LLM-generated content at query #54
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_types():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = int
    try:
        _check_types(it, expected_types, source_class)
    except CheckedValueTypeError:
        pass
    else:
        assert False, "Expected CheckedValueTypeError"

def test_check_types_with_empty_iterable():
    it = []
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_multiple_valid_types():
    it = [1, 2.0, 3]
    expected_types = [int, float]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_string_type_name():
    it = [1, 2, 3]
    expected_types = ['builtins.int']
    source_class = int
    _check_types(it, expected_types, source_class)


# LLM-generated content at query #55
#--------------------------

```python
def test_checked_pmap_new_without_size():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap({1: "a", 2: "b"})
    assert result == {1: "a", 2: "b"}


# LLM-generated content at query #56
#--------------------------

```python
def test_invariant_errors_returns_empty_list_when_all_invariants_valid():
    elem = "test"
    invariants = [lambda x: (True, "data1"), lambda x: (True, "data2")]
    assert _invariant_errors(elem, invariants) == []


# LLM-generated content at query #57
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

    result = IntToFloatMap(size=5)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #58
#--------------------------

```python
def test__check_types_with_empty_iterable():
    _check_types([], [int], str)


# LLM-generated content at query #59
#--------------------------

```python
def test_checked_type_instantiation():
    with pytest.raises(TypeError):
        CheckedType()


# LLM-generated content at query #60
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

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #61
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


# LLM-generated content at query #62
#--------------------------

```python
def test_predicate_at_line_18():
    assert maybe_parse_user_type(int) == [int]


# LLM-generated content at query #63
#--------------------------

```python
def test_constructor_empty_initial():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestCheckedPMap()
    assert isinstance(result, TestCheckedPMap)
    assert dict(result) == {}

def test_constructor_with_valid_initial():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    initial = {1: "one", 2: "two"}
    result = TestCheckedPMap(initial)
    assert isinstance(result, TestCheckedPMap)
    assert dict(result) == initial

def test_constructor_with_invalid_key_type():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    initial = {"1": "one"}
    try:
        TestCheckedPMap(initial)
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_constructor_with_invalid_value_type():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    initial = {1: 1}
    try:
        TestCheckedPMap(initial)
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_constructor_with_invariant_violation():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")

    initial = {1: 0}
    try:
        TestCheckedPMap(initial)
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_constructor_with_size_parameter():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestCheckedPMap(size=10)
    assert isinstance(result, TestCheckedPMap)
    assert dict(result) == {}


# LLM-generated content at query #64
#--------------------------

```python
def test__checked_type_create_with_instance_of_cls():
    class TestClass:
        _checked_types = []

    instance = TestClass()
    assert _checked_type_create(TestClass, instance) is instance

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
            return f"created_{data}"

    class TestClass:
        _checked_types = ["__main__.CheckedType"]

    source_data = ["data1", "data2"]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == ["created_data1", "created_data2"]

def test__checked_type_create_with_checked_type_and_matching_instance():
    class CheckedType:
        pass

    class TestClass:
        _checked_types = ["__main__.CheckedType"]

    instance = CheckedType()
    source_data = [instance, "other_data"]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == [instance, "other_data"]

def test__checked_type_create_with_ignore_extra():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}_{ignore_extra}"

    class TestClass:
        _checked_types = ["__main__.CheckedType"]

    source_data = ["data1", "data2"]
    result = _checked_type_create(TestClass, source_data, ignore_extra=True)
    assert isinstance(result, TestClass)
    assert result == ["created_data1_True", "created_data2_True"]


# LLM-generated content at query #65
#--------------------------

```python
def test_maybe_types_is_truthy():
    dct = {'source_name': 'value'}
    bases = []
    destination_name = 'dest'
    source_name = 'source_name'

    _store_types(dct, bases, destination_name, source_name)

    assert dct[destination_name]


# LLM-generated content at query #66
#--------------------------

```python
def test_checked_type_create_predicate_false():
    class TestClass:
        pass

    class NotCheckedType:
        pass

    source_data = NotCheckedType()
    assert not isinstance(source_data, TestClass)


# LLM-generated content at query #67
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


# LLM-generated content at query #68
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


# LLM-generated content at query #69
#--------------------------

```python
def test_check_types_with_empty_expected_types():
    it = [1, 2, 3]
    expected_types = []
    source_class = str
    _check_types(it, expected_types, source_class)


# LLM-generated content at query #70
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


# LLM-generated content at query #71
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "data1"), (True, "data2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_all_false():
    result = [(False, "data1"), (False, "data2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("data1", "data2")

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


# LLM-generated content at query #72
#--------------------------

```python
def test_is_preserved_evaluates_to_true():
    class PreservedType(_preserved_iterable_types):
        pass

    t = PreservedType
    is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
    assert is_preserved is True


# LLM-generated content at query #73
#--------------------------

```python
def test__checked_type_create_with_instance_of_cls():
    class TestClass:
        _checked_types = []

    instance = TestClass()
    result = _checked_type_create(TestClass, instance)
    assert result is instance

def test__checked_type_create_with_non_checked_type():
    class TestClass:
        _checked_types = []

    source_data = [1, 2, 3]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == source_data

def test__checked_type_create_with_checked_type():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}"

    class TestClass:
        _checked_types = ["__main__.CheckedType"]

    source_data = ["a", "b", "c"]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == ["created_a", "created_b", "created_c"]

def test__checked_type_create_with_mixed_checked_and_non_checked_types():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}"

    class TestClass:
        _checked_types = ["__main__.CheckedType"]

    source_data = [CheckedType(), "a", "b"]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == [CheckedType(), "created_a", "created_b"]

def test__checked_type_create_with_ignore_extra():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}_ignore_{ignore_extra}"

    class TestClass:
        _checked_types = ["__main__.CheckedType"]

    source_data = ["a", "b"]
    result = _checked_type_create(TestClass, source_data, ignore_extra=True)
    assert isinstance(result, TestClass)
    assert result == ["created_a_ignore_True", "created_b_ignore_True"]


# LLM-generated content at query #74
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

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #75
#--------------------------

```python
def test_is_preserved_evaluates_to_true():
    class PreservedType(_preserved_iterable_types):
        pass

    t = PreservedType
    is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
    assert is_preserved is True


# LLM-generated content at query #76
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
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

def test_checked_pset_constructor_with_pmap_initial():
    class TestCheckedPSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    initial_pmap = pmap({1: True, 2: True, 3: True})
    result = TestCheckedPSet(initial_pmap)
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
        TestCheckedPSet(["invalid"])
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
    except InvariantException as e:
        assert len(e.error_codes) == 3
        assert all("Non-positive" in str(error) for error in e.error_codes)


# LLM-generated content at query #77
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

    result = TestMap({}, size=5)
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

def test_checked_pmap_constructor_with_invalid_invariant():
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


# LLM-generated content at query #78
#--------------------------

```python
def test_checked_type_instantiation():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)


