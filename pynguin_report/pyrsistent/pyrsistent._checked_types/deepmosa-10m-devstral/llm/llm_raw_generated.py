####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__new__with_valid_key_type():
    class TestClass(metaclass=_CheckedMapTypeMeta):
        __key_type__ = int
    assert TestClass._checked_key_types == [int]

def test__new__with_valid_value_type():
    class TestClass(metaclass=_CheckedMapTypeMeta):
        __value_type__ = str
    assert TestClass._checked_value_types == [str]

def test__new__with_invalid_invariant():
    try:
        class TestClass(metaclass=_CheckedMapTypeMeta):
            __invariant__ = "not_callable"
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test__new__with_valid_invariant():
    def test_inv():
        return True
    class TestClass(metaclass=_CheckedMapTypeMeta):
        __invariant__ = test_inv
    assert TestClass._checked_invariants == (wrap_invariant(test_inv),)

def test__new__with_default_serializer():
    class TestClass(metaclass=_CheckedMapTypeMeta):
        pass
    assert hasattr(TestClass, '__serializer__')
    assert callable(TestClass.__serializer__)

def test__new__with_custom_serializer():
    def custom_serializer(self, _, key, value):
        return key, value
    class TestClass(metaclass=_CheckedMapTypeMeta):
        __serializer__ = custom_serializer
    assert TestClass.__serializer__ == custom_serializer

def test__new__with_slots():
    class TestClass(metaclass=_CheckedMapTypeMeta):
        pass
    assert TestClass.__slots__ == ()


# LLM-generated content at query #2
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    result = CheckedPSet()
    assert isinstance(result, CheckedPSet)
    assert len(result) == 0

def test_checked_pset_constructor_with_list_initial():
    class TestCheckedPSet(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestCheckedPSet([1, 2, 3])
    assert isinstance(result, TestCheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checked_pset_constructor_with_pmap_initial():
    class TestCheckedPSet(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    pmap_instance = pmap({1: True, 2: True, 3: True})
    result = TestCheckedPSet(pmap_instance)
    assert isinstance(result, TestCheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_checked_pset_constructor_with_invalid_type():
    class TestCheckedPSet(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestCheckedPSet(["invalid"])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pset_constructor_with_invalid_invariant():
    class TestCheckedPSet(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestCheckedPSet([-1, -2])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_checked_pvector_constructor_empty():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, 'Non-positive')

    result = TestVector()
    assert result.tolist() == []

def test_checked_pvector_constructor_valid_elements():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, 'Non-positive')

    result = TestVector([1, 2, 3])
    assert result.tolist() == [1, 2, 3]

def test_checked_pvector_constructor_from_python_pvector():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, 'Non-positive')

    pv = python_pvector([1, 2, 3])
    result = TestVector(pv)
    assert result.tolist() == [1, 2, 3]

def test_checked_pvector_constructor_invalid_type():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, 'Non-positive')

    try:
        TestVector(['a', 'b'])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pvector_constructor_invalid_invariant():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, 'Non-positive')

    try:
        TestVector([-1, 2, 3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_maybe_parse_user_type_with_preserved_type():
    assert maybe_parse_user_type(list) == [list]

def test_maybe_parse_user_type_with_string():
    assert maybe_parse_user_type('int') == ['int']

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


# LLM-generated content at query #5
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

def test_check_types_with_multiple_valid_types():
    it = [1, 2.0, 3]
    expected_types = [int, float]
    source_class = int
    _check_types(it, expected_types, source_class)

def test_check_types_with_multiple_invalid_types():
    it = [1, 'a', 3.0]
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

def test_check_types_with_custom_exception():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = int
    try:
        _check_types(it, expected_types, source_class, exception_type=ValueError)
    except ValueError:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_serialize_default():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x >= 0, "Negative")

    vec = TestVector([1, 2, 3])
    assert vec.serialize() == [1, 2, 3]

def test_serialize_with_format():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x >= 0, "Negative")

    vec = TestVector([1, 2, 3])
    assert vec.serialize(str) == ['1', '2', '3']


# LLM-generated content at query #7
#--------------------------

```python
def test_serialize_default_format():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    vec = TestVector([1, 2, 3])
    assert vec.serialize() == [1, 2, 3]

def test_serialize_with_custom_serializer():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")
        __serializer__ = lambda fmt, v: str(v) if fmt == "str" else v

    vec = TestVector([1, 2, 3])
    assert vec.serialize("str") == ["1", "2", "3"]
    assert vec.serialize() == [1, 2, 3]

def test_serialize_empty_vector():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    vec = TestVector()
    assert vec.serialize() == []

def test_serialize_mixed_types():
    class TestVector(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    vec = TestVector([1, 2.5, 3])
    assert vec.serialize() == [1, 2.5, 3]


# LLM-generated content at query #8
#--------------------------

```python
def test_check_types_with_empty_expected_types():
    _check_types([], [], str)


# LLM-generated content at query #9
#--------------------------

```python
def test_empty_invariants():
    assert _invariant_errors("test", []) == []

def test_all_invariants_pass():
    def always_valid(x): return (True, "valid")
    assert _invariant_errors("test", [always_valid, always_valid]) == []

def test_single_invariant_fails():
    def fails(x): return (False, "error")
    assert _invariant_errors("test", [fails]) == ["error"]

def test_multiple_invariants_mixed():
    def passes(x): return (True, "ok")
    def fails(x): return (False, "error")
    assert _invariant_errors("test", [passes, fails, passes]) == ["error"]

def test_multiple_invariants_fail():
    def fails1(x): return (False, "error1")
    def fails2(x): return (False, "error2")
    assert _invariant_errors("test", [fails1, fails2]) == ["error1", "error2"]


# LLM-generated content at query #10
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = type(it)
    assert _check_types(it, expected_types, source_class) is None


# LLM-generated content at query #11
#--------------------------

```python
def test_check_types_with_valid_types():
    class TestClass: pass
    it = [1, 2, 3]
    expected_types = [int]
    _check_types(it, expected_types, TestClass)

def test_check_types_with_invalid_type():
    class TestClass: pass
    it = [1, 'a', 3]
    expected_types = [int]
    try:
        _check_types(it, expected_types, TestClass)
    except CheckedValueTypeError as e:
        assert e.source_class == TestClass
        assert e.expected_types == (int,)
        assert e.actual_type == str
        assert e.value == 'a'

def test_check_types_with_multiple_valid_types():
    class TestClass: pass
    it = [1, 2.0, 3]
    expected_types = [int, float]
    _check_types(it, expected_types, TestClass)

def test_check_types_with_empty_iterable():
    class TestClass: pass
    it = []
    expected_types = [int]
    _check_types(it, expected_types, TestClass)

def test_check_types_with_empty_expected_types():
    class TestClass: pass
    it = [1, 'a', 3.0]
    expected_types = []
    _check_types(it, expected_types, TestClass)

def test_check_types_with_custom_exception():
    class TestClass: pass
    class CustomException(Exception): pass
    it = [1, 'a', 3]
    expected_types = [int]
    try:
        _check_types(it, expected_types, TestClass, CustomException)
    except CustomException as e:
        assert e.source_class == TestClass
        assert e.expected_types == (int,)
        assert e.actual_type == str
        assert e.value == 'a'


# LLM-generated content at query #12
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
        assert e.error_codes[0] == ('Invalid mapping',)

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=2)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_empty_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({'a': 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: '1.5'})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test__check_types_with_empty_iterable():
    _check_types([], [int, str], str)


# LLM-generated content at query #14
#--------------------------

```python
def test_constructor_with_empty_args():
    exception = InvariantException()
    assert exception.invariant_errors == ()
    assert exception.missing_fields == ()
    assert str(exception) == ", invariant_errors=[], missing_fields=[]"

def test_constructor_with_error_codes():
    exception = InvariantException(error_codes=("error1", "error2"))
    assert exception.invariant_errors == ("error1", "error2")
    assert exception.missing_fields == ()
    assert str(exception) == ", invariant_errors=[error1, error2], missing_fields=[]"

def test_constructor_with_missing_fields():
    exception = InvariantException(missing_fields=("field1", "field2"))
    assert exception.invariant_errors == ()
    assert exception.missing_fields == ("field1", "field2")
    assert str(exception) == ", invariant_errors=[], missing_fields=[field1, field2]"

def test_constructor_with_both_args():
    exception = InvariantException(error_codes=("error1",), missing_fields=("field1",))
    assert exception.invariant_errors == ("error1",)
    assert exception.missing_fields == ("field1",)
    assert str(exception) == ", invariant_errors=[error1], missing_fields=[field1]"

def test_constructor_with_callable_error_codes():
    exception = InvariantException(error_codes=(lambda: "error1", lambda: "error2"))
    assert exception.invariant_errors == ("error1", "error2")
    assert exception.missing_fields == ()
    assert str(exception) == ", invariant_errors=[error1, error2], missing_fields=[]"

def test_constructor_with_mixed_error_codes():
    exception = InvariantException(error_codes=("error1", lambda: "error2"))
    assert exception.invariant_errors == ("error1", "error2")
    assert exception.missing_fields == ()
    assert str(exception) == ", invariant_errors=[error1, error2], missing_fields=[]"


# LLM-generated content at query #15
#--------------------------

```python
def test_store_invariants_with_valid_invariants():
    class Base:
        def invariant(self):
            return True, "data"

    class Derived(Base):
        pass

    dct = {}
    store_invariants(dct, (Derived,), 'wrapped_invariants', 'invariant')
    assert 'wrapped_invariants' in dct
    assert len(dct['wrapped_invariants']) == 1
    assert callable(dct['wrapped_invariants'][0])
    result = dct['wrapped_invariants'][0]()
    assert result == (True, "data")

def test_store_invariants_with_multiple_invariants():
    class Base1:
        def invariant(self):
            return True, "data1"

    class Base2:
        def invariant(self):
            return False, "data2"

    class Derived(Base1, Base2):
        pass

    dct = {}
    store_invariants(dct, (Derived,), 'wrapped_invariants', 'invariant')
    assert 'wrapped_invariants' in dct
    assert len(dct['wrapped_invariants']) == 2
    assert all(callable(inv) for inv in dct['wrapped_invariants'])
    result1 = dct['wrapped_invariants'][0]()
    result2 = dct['wrapped_invariants'][1]()
    assert result1 == (True, "data1")
    assert result2 == (False, "data2")

def test_store_invariants_with_non_callable_invariant():
    class Base:
        invariant = "not callable"

    dct = {}
    try:
        store_invariants(dct, (Base,), 'wrapped_invariants', 'invariant')
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"

def test_store_invariants_with_no_invariants():
    class Base:
        pass

    dct = {}
    store_invariants(dct, (Base,), 'wrapped_invariants', 'invariant')
    assert 'wrapped_invariants' in dct
    assert len(dct['wrapped_invariants']) == 0

def test_store_invariants_with_inherited_invariants():
    class Base:
        def invariant(self):
            return True, "base_data"

    class Middle(Base):
        pass

    class Derived(Middle):
        pass

    dct = {}
    store_invariants(dct, (Derived,), 'wrapped_invariants', 'invariant')
    assert 'wrapped_invariants' in dct
    assert len(dct['wrapped_invariants']) == 1
    result = dct['wrapped_invariants'][0]()
    assert result == (True, "base_data")


# LLM-generated content at query #16
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


# LLM-generated content at query #17
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

def test_checked_pvector_constructor_with_invalid_type():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    with pytest.raises(TypeError):
        Positives(['a', 'b', 'c'])

def test_checked_pvector_constructor_with_invalid_invariant():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    with pytest.raises(InvariantException):
        Positives([-1, -2, -3])


# LLM-generated content at query #18
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

def test_checked_pvector_constructor_with_invalid_type():
    try:
        CheckedPVector([1, 'a', 3])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pvector_constructor_with_invalid_invariant():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([1, -2, 3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #19
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
    result = TestMap(initial_data, size=5)
    assert isinstance(result, TestMap)
    assert dict(result) == initial_data

def test_checked_pmap_constructor_invalid_key_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({"invalid": "value"})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_invalid_value_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestMap({1: 123})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_invalid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
        __invariant__ = lambda k, v: (k == len(v), "Key must equal length of value")

    try:
        TestMap({1: "invalid"})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_invariants_are_callable():
    dct = {'inv1': lambda: True, 'inv2': lambda: False}
    bases = []
    destination_name = 'invariants'
    source_name = 'inv1'

    store_invariants(dct, bases, destination_name, source_name)

    assert all(callable(inv) for inv in dct[destination_name])


# LLM-generated content at query #21
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

    source_data = ["data1", "data2"]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == source_data

def test__checked_type_create_with_checked_type_create_method():
    class CheckedType(CheckedType):
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}"

    class TestClass:
        _checked_types = ["module.CheckedType"]

    source_data = ["data1", "data2"]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == ["created_data1", "created_data2"]

def test__checked_type_create_with_ignore_extra():
    class CheckedType(CheckedType):
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}"

    class TestClass:
        _checked_types = ["module.CheckedType"]

    source_data = ["data1", "data2"]
    result = _checked_type_create(TestClass, source_data, ignore_extra=True)
    assert isinstance(result, TestClass)
    assert result == ["created_data1", "created_data2"]


# LLM-generated content at query #22
#--------------------------

```python
def test_store_invariants_raises_type_error_for_non_callable():
    dct = {'invariant': 123}
    bases = ()
    destination_name = 'invariants'
    source_name = 'invariant'
    try:
        store_invariants(dct, bases, destination_name, source_name)
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        assert False, "Expected TypeError was not raised"


# LLM-generated content at query #23
#--------------------------

```python
def test_is_preserved_evaluates_to_true():
    class PreservedType(_preserved_iterable_types):
        pass

    t = PreservedType
    is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
    assert is_preserved


# LLM-generated content at query #24
#--------------------------

```python
def test_invariant_errors_with_invalid_invariant():
    def invalid_invariant(x):
        return (False, "Error message")

    result = _invariant_errors("test_elem", [invalid_invariant])
    assert result == ["Error message"]


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


# LLM-generated content at query #27
#--------------------------

```python
def test_check_types_with_empty_expected_types():
    _check_types([1, 2, 3], [], str)


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

def test_store_types_with_base_class_type():
    class Base:
        type = float
    dct = {}
    bases = [Base]
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [float]

def test_store_types_with_mixed_types():
    class Base:
        type = [bool, list]
    dct = { 'type': str }
    bases = [Base]
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [str, bool, list]

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
        __invariant__ = lambda k, v: (v == k * 2, "Value must be twice the key")

    try:
        TestMap({1: 3.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #30
#--------------------------

```python
def test_checked_pset_constructor_with_valid_initial_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    result = Positives([1, 2, 3])
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

def test_checked_pset_constructor_with_pmap_input():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    pmap_input = pmap({1: True, 2: True, 3: True})
    result = Positives(pmap_input)
    assert isinstance(result, Positives)
    assert result == pset([1, 2, 3])

def test_checked_pset_constructor_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    result = Positives()
    assert isinstance(result, Positives)
    assert result == pset()


# LLM-generated content at query #31
#--------------------------

```python
def test__invariant_errors_returns_empty_list_when_all_invariants_pass():
    elem = "test"
    invariants = [lambda x: (True, "data1"), lambda x: (True, "data2")]
    assert _invariant_errors(elem, invariants) == []


# LLM-generated content at query #32
#--------------------------

```python
def test_checked_type_create_with_checked_type_instance():
    class TestCheckedType(CheckedType):
        pass

    class TestClass:
        _checked_types = (TestCheckedType,)

    source_data = TestCheckedType()
    result = TestClass._checked_type_create(source_data)
    assert result is source_data


# LLM-generated content at query #33
#--------------------------

```python
def test_maybe_types_assignment():
    dct = {'source_name': 'value'}
    bases = []
    destination_name = 'dest'
    source_name = 'source_name'
    _store_types(dct, bases, destination_name, source_name)
    assert destination_name in dct


# LLM-generated content at query #34
#--------------------------

```python
def test_checked_type_create_predicate_false():
    assert not isinstance(source_data, cls)


# LLM-generated content at query #35
#--------------------------

```python
def test_predicate_evaluates_to_false():
    source_data = "not_a_class_instance"
    result = isinstance(source_data, str)
    assert not result


# LLM-generated content at query #36
#--------------------------

```python
def test_checked_type_create_predicate_false():
    class TestClass:
        _checked_types = ()

    assert not isinstance({}, TestClass)


# LLM-generated content at query #37
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_valid_items():
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
        __value_type__ = float
        __invariant__ = lambda k, v: (k == int(v), "Key must equal int(value)")

    try:
        TestMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #38
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

    result = IntToFloatMap(size=5, initial={1: 1.0, 2: 2.0})
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
        IntToFloatMap({1: "a"})
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

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_at_line_18():
    assert maybe_parse_user_type(int) == [int]


# LLM-generated content at query #40
#--------------------------

```python
def test_checked_type_instantiation():
    with pytest.raises(TypeError):
        CheckedType()


# LLM-generated content at query #41
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.5, 2: 2.25}

def test_checked_pmap_constructor_with_invalid_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 1.5, 2: 3.25})
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

def test_checked_pmap_constructor_empty_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


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


# LLM-generated content at query #43
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


# LLM-generated content at query #44
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


# LLM-generated content at query #45
#--------------------------

```python
def test_predicate_evaluates_to_false():
    source_data = "not_a_class_instance"
    cls = str
    assert not isinstance(source_data, cls)


# LLM-generated content at query #46
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

def test__checked_type_create_with_checked_type_in_source_data():
    class TestClass:
        _checked_types = ['module.CheckedType']

    class CheckedType:
        @staticmethod
        def create(data, ignore_extra=False):
            return f"created_{data}"

    source_data = ["data1", "data2"]
    with patch('_get_class', return_value=CheckedType):
        result = _checked_type_create(TestClass, source_data)
        assert isinstance(result, TestClass)
        assert result == ["created_data1", "created_data2"]

def test__checked_type_create_with_checked_type_not_in_source_data():
    class TestClass:
        _checked_types = ['module.CheckedType']

    class CheckedType:
        @staticmethod
        def create(data, ignore_extra=False):
            return f"created_{data}"

    source_data = [1, 2]
    with patch('_get_class', return_value=CheckedType):
        result = _checked_type_create(TestClass, source_data)
        assert isinstance(result, TestClass)
        assert result == [1, 2]

def test__checked_type_create_with_ignore_extra():
    class TestClass:
        _checked_types = ['module.CheckedType']

    class CheckedType:
        @staticmethod
        def create(data, ignore_extra=False):
            return f"created_{data}_{ignore_extra}"

    source_data = ["data1", "data2"]
    with patch('_get_class', return_value=CheckedType):
        result = _checked_type_create(TestClass, source_data, ignore_extra=True)
        assert isinstance(result, TestClass)
        assert result == ["created_data1_True", "created_data2_True"]


# LLM-generated content at query #47
#--------------------------

```python
def test_empty_invariants():
    assert _invariant_errors("test", []) == []

def test_no_errors():
    def always_valid(x):
        return True, "valid"
    assert _invariant_errors("test", [always_valid]) == []

def test_single_error():
    def always_invalid(x):
        return False, "invalid"
    assert _invariant_errors("test", [always_invalid]) == ["invalid"]

def test_multiple_errors():
    def invalid1(x):
        return False, "error1"
    def invalid2(x):
        return False, "error2"
    assert _invariant_errors("test", [invalid1, invalid2]) == ["error1", "error2"]

def test_mixed_valid_invalid():
    def valid(x):
        return True, "valid"
    def invalid(x):
        return False, "invalid"
    assert _invariant_errors("test", [valid, invalid, valid]) == ["invalid"]


# LLM-generated content at query #48
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_data():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestCheckedPMap({1: "a", 2: "b"})
    assert isinstance(result, TestCheckedPMap)
    assert dict(result) == {1: "a", 2: "b"}

def test_checked_pmap_constructor_with_invalid_key_type():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestCheckedPMap({"a": "b"})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    try:
        TestCheckedPMap({1: 2})
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
        pass

def test_checked_pmap_constructor_with_invariant_violation():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")

    try:
        TestCheckedPMap({2: 1})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestCheckedPMap(size=5)
    assert isinstance(result, TestCheckedPMap)
    assert len(result) == 0


# LLM-generated content at query #49
#--------------------------

```python
def test_is_preserved_evaluates_to_true():
    class PreservedType(_preserved_iterable_types):
        pass

    t = PreservedType
    is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
    assert is_preserved is True


# LLM-generated content at query #50
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
    class TestCheckedType(CheckedType):
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"checked_{data}"

    class TestClass:
        _checked_types = ["module.TestCheckedType"]

    source_data = ["data1", "data2"]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == ["checked_data1", "checked_data2"]

def test__checked_type_create_with_instance_in_checked_types():
    class TestCheckedType(CheckedType):
        pass

    class TestClass:
        _checked_types = ["module.TestCheckedType"]

    instance = TestCheckedType()
    source_data = [instance, "other_data"]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == [instance, "other_data"]

def test__checked_type_create_with_ignore_extra():
    class TestCheckedType(CheckedType):
        @classmethod
        def create(cls, data, ignore_extra=False):
            return data if ignore_extra else f"checked_{data}"

    class TestClass:
        _checked_types = ["module.TestCheckedType"]

    source_data = ["data1", "data2"]
    result = _checked_type_create(TestClass, source_data, ignore_extra=True)
    assert isinstance(result, TestClass)
    assert result == ["data1", "data2"]


# LLM-generated content at query #51
#--------------------------

```python
def test_wrap_invariant_with_single_bool_result():
    def mock_invariant():
        return (True, "data")

    wrapped = wrap_invariant(mock_invariant)
    result = wrapped()
    assert result == (True, "data")

def test_wrap_invariant_with_false_bool_result():
    def mock_invariant():
        return (False, "error")

    wrapped = wrap_invariant(mock_invariant)
    result = wrapped()
    assert result == (False, "error")

def test_wrap_invariant_with_multiple_results():
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


# LLM-generated content at query #52
#--------------------------

```python
def test_maybe_parse_user_type_with_preserved_type():
    class PreservedType(_preserved_iterable_types):
        pass
    assert maybe_parse_user_type(PreservedType) == [PreservedType]

def test_maybe_parse_user_type_with_string():
    assert maybe_parse_user_type("str") == ["str"]

def test_maybe_parse_user_type_with_non_iterable_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    assert maybe_parse_user_type([int, str]) == [int, str]

def test_maybe_parse_user_type_with_nested_iterable():
    assert maybe_parse_user_type([int, [str, float]]) == [int, str, float]

def test_maybe_parse_user_type_with_invalid_input():
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)


# LLM-generated content at query #53
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


# LLM-generated content at query #54
#--------------------------

```python
def test_wrap_invariant_with_non_bool_result():
    def mock_invariant(*args, **kwargs):
        return [1, 2, 3]

    wrapped = wrap_invariant(mock_invariant)
    result = wrapped()
    assert result == [1, 2, 3]


# LLM-generated content at query #55
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

    result = IntToFloatMap(size=5, initial={1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0}


# LLM-generated content at query #56
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = type(it)
    assert _check_types(it, expected_types, source_class) is None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_serialize_default_format():
    class TestSet(CheckedPSet):
        __type__ = int
    test_instance = TestSet([1, 2, 3])
    assert test_instance.serialize() == {1, 2, 3}

def test_serialize_with_custom_serializer():
    class TestSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda fmt, x: str(x) if fmt == 'str' else x
    test_instance = TestSet([1, 2, 3])
    assert test_instance.serialize('str') == {'1', '2', '3'}

def test_serialize_empty_set():
    class TestSet(CheckedPSet):
        __type__ = int
    test_instance = TestSet()
    assert test_instance.serialize() == set()

def test_serialize_with_different_types():
    class TestSet(CheckedPSet):
        __type__ = (int, str)
    test_instance = TestSet([1, 'a', 2, 'b'])
    assert test_instance.serialize() == {1, 'a', 2, 'b'}


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
    store_invariants(dct, bases, "wrapped_invariants", "invariant1")
    assert "wrapped_invariants" in dct
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
    assert "wrapped_invariants" in dct
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
    assert "wrapped_invariants" in dct
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
    assert "wrapped_invariants" in dct
    assert len(dct["wrapped_invariants"]) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_checked_pset_constructor_with_valid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    result = Positives([1, 2, 3])
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

def test_checked_pset_constructor_with_pmap_input():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    pmap_input = pmap({1: True, 2: True, 3: True})
    result = Positives(pmap_input)
    assert isinstance(result, Positives)
    assert result == pset([1, 2, 3])

def test_checked_pset_constructor_empty():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    result = Positives()
    assert isinstance(result, Positives)
    assert result == pset()

def test_checked_pset_constructor_with_mixed_valid_types():
    class Numbers(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    result = Numbers([1, 2.5, 3])
    assert isinstance(result, Numbers)
    assert result == pset([1, 2.5, 3])


# LLM-generated content at query #4
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

def test_wrap_invariant_with_empty_list():
    def invariant():
        return []
    wrapped = wrap_invariant(invariant)
    assert wrapped() == (True, ())

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


# LLM-generated content at query #5
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    class TestSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestSet()
    assert isinstance(result, TestSet)
    assert len(result) == 0

def test_checked_pset_constructor_with_valid_elements():
    class TestSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestSet([1, 2, 3])
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
        TestSet(["a", "b"])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pset_constructor_with_invalid_invariant():
    class TestSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestSet([-1, -2])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pset_constructor_with_pmap_initial():
    class TestSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    pmap = pmap({1: True, 2: True})
    result = TestSet(pmap)
    assert isinstance(result, TestSet)
    assert len(result) == 2
    assert 1 in result
    assert 2 in result


# LLM-generated content at query #6
#--------------------------

```python
def test_checked_type_instantiation():
    with pytest.raises(TypeError):
        CheckedType()


# LLM-generated content at query #7
#--------------------------

```python
def test_store_types_with_single_type():
    dct = {}
    bases = ()
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = int
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int]

def test_store_types_with_string_type():
    dct = {}
    bases = ()
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = 'int'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == ['int']

def test_store_types_with_iterable_types():
    dct = {}
    bases = ()
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = [int, str]
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str]

def test_store_types_with_inherited_types():
    dct = {}
    class Base:
        type = int
    bases = (Base,)
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int]

def test_store_types_with_multiple_inherited_types():
    dct = {}
    class Base1:
        type = int
    class Base2:
        type = str
    bases = (Base1, Base2)
    destination_name = 'types'
    source_name = 'type'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [int, str]

def test_store_types_with_preserved_iterable_type():
    dct = {}
    bases = ()
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = list
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == [list]

def test_store_types_with_invalid_type():
    dct = {}
    bases = ()
    destination_name = 'types'
    source_name = 'type'
    dct[source_name] = 123
    try:
        _store_types(dct, bases, destination_name, source_name)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "data1"), (True, "data2")]
    assert _merge_invariant_results(result) == (True, ())

def test_merge_invariant_results_all_false():
    result = [(False, "data1"), (False, "data2")]
    assert _merge_invariant_results(result) == (False, ("data1", "data2"))

def test_merge_invariant_results_mixed():
    result = [(True, "data1"), (False, "data2"), (True, "data3")]
    assert _merge_invariant_results(result) == (False, ("data2",))

def test_merge_invariant_results_empty():
    result = []
    assert _merge_invariant_results(result) == (True, ())


# LLM-generated content at query #9
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

    source_data = [1, 2, 3]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert result == source_data

def test__checked_type_create_with_checked_type_in_source_data():
    class TestCheckedType(CheckedType):
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)

    class TestClass:
        _checked_types = ['module.TestCheckedType']

    source_data = [TestCheckedType([1, 2]), 3, 4]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert len(result) == 3
    assert isinstance(result[0], TestCheckedType)
    assert result[1] == 3
    assert result[2] == 4

def test__checked_type_create_with_non_checked_type_in_source_data():
    class TestCheckedType(CheckedType):
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)

    class TestClass:
        _checked_types = ['module.TestCheckedType']

    source_data = [1, 2, 3]
    result = _checked_type_create(TestClass, source_data)
    assert isinstance(result, TestClass)
    assert len(result) == 3
    assert all(isinstance(item, TestCheckedType) for item in result)


# LLM-generated content at query #10
#--------------------------

```python
def test_checked_pvector_constructor_with_empty_initial():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector()
    assert result.tolist() == []

def test_checked_pvector_constructor_with_list_initial():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    result = TestVector([1, 2, 3])
    assert result.tolist() == [1, 2, 3]

def test_checked_pvector_constructor_with_pvector_initial():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    pv = python_pvector([1, 2, 3])
    result = TestVector(pv)
    assert result.tolist() == [1, 2, 3]

def test_checked_pvector_constructor_with_invalid_type():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector([1, "2", 3])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_checked_pvector_constructor_with_invalid_invariant():
    class TestVector(CheckedPVector):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector([1, -2, 3])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


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
def test_checked_pmap_constructor_with_valid_data():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = TestMap({1: 1.0, 2: 2.0})
    assert isinstance(result, TestMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_with_invalid_key_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        TestMap({"a": 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_value_type():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        TestMap({1: "not_a_float"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass

def test_checked_pmap_constructor_with_invalid_invariant():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        TestMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = TestMap(size=5, initial={1: 1.0})
    assert isinstance(result, TestMap)
    assert dict(result) == {1: 1.0}


# LLM-generated content at query #13
#--------------------------

```python
def test_maybe_parse_user_type_with_preserved_type():
    assert maybe_parse_user_type(list) == [list]
    assert maybe_parse_user_type(tuple) == [tuple]
    assert maybe_parse_user_type(dict) == [dict]

def test_maybe_parse_user_type_with_string():
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("str") == ["str"]

def test_maybe_parse_user_type_with_non_iterable_type():
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(float) == [float]

def test_maybe_parse_user_type_with_iterable_of_types():
    assert maybe_parse_user_type([int, str]) == [int, str]
    assert maybe_parse_user_type((float, list)) == [float, list]

def test_maybe_parse_user_type_with_nested_iterable():
    assert maybe_parse_user_type([int, [str, float]]) == [int, str, float]
    assert maybe_parse_user_type((list, (dict, tuple))) == [list, dict, tuple]

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


# LLM-generated content at query #14
#--------------------------

```python
def test_empty_invariants():
    assert _invariant_errors("test", []) == []

def test_single_invariant_pass():
    assert _invariant_errors("test", [lambda x: (True, "error")]) == []

def test_single_invariant_fail():
    assert _invariant_errors("test", [lambda x: (False, "error")]) == ["error"]

def test_multiple_invariants_all_pass():
    assert _invariant_errors("test", [lambda x: (True, "error1"), lambda x: (True, "error2")]) == []

def test_multiple_invariants_some_fail():
    assert _invariant_errors("test", [lambda x: (True, "error1"), lambda x: (False, "error2")]) == ["error2"]

def test_multiple_invariants_all_fail():
    assert _invariant_errors("test", [lambda x: (False, "error1"), lambda x: (False, "error2")]) == ["error1", "error2"]


# LLM-generated content at query #15
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


# LLM-generated content at query #16
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
    assert "wrapped_invariants" in dct
    assert len(dct["wrapped_invariants"]) == 1
    assert callable(dct["wrapped_invariants"][0])

def test_store_invariants_with_inherited_invariants():
    class Base:
        def invariant1(self):
            return True, "data1"

    class Derived(Base):
        def invariant2(self):
            return True, "data2"

    dct = {}
    bases = (Derived,)
    store_invariants(dct, bases, "wrapped_invariants", "invariant1")
    assert "wrapped_invariants" in dct
    assert len(dct["wrapped_invariants"]) == 1
    assert callable(dct["wrapped_invariants"][0])

def test_store_invariants_with_non_callable_invariant():
    class Base:
        invariant1 = "not callable"

    dct = {}
    bases = (Base,)
    try:
        store_invariants(dct, bases, "wrapped_invariants", "invariant1")
    except TypeError as e:
        assert str(e) == "Invariants must be callable"
    else:
        assert False, "Expected TypeError"


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

    result = IntToFloatMap(size=10)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_nested_checked_types():
    class NestedMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = CheckedPMap

    inner_map = CheckedPMap({1: 1.0, 2: 2.0})
    result = NestedMap({1: inner_map})
    assert isinstance(result, NestedMap)
    assert dict(result) == {1: inner_map}


# LLM-generated content at query #18
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


# LLM-generated content at query #19
#--------------------------

```python
def test_empty_invariants():
    assert _invariant_errors("test", []) == []

def test_all_valid_invariants():
    assert _invariant_errors(5, [lambda x: (x > 0, "positive"), lambda x: (x < 10, "less than 10")]) == []

def test_single_invalid_invariant():
    assert _invariant_errors(-1, [lambda x: (x >= 0, "non-negative")]) == ["non-negative"]

def test_multiple_invalid_invariants():
    assert _invariant_errors(15, [lambda x: (x < 10, "less than 10"), lambda x: (x % 2 == 0, "even")]) == ["less than 10", "even"]

def test_mixed_valid_and_invalid_invariants():
    assert _invariant_errors(7, [lambda x: (x > 0, "positive"), lambda x: (x < 5, "less than 5"), lambda x: (x % 2 == 1, "odd")]) == ["less than 5"]


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
        assert e.error_codes[0] == 'Invalid mapping'
        assert e.error_codes[1] == 'Invalid mapping'

def test_checked_pmap_constructor_with_size():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


# LLM-generated content at query #21
#--------------------------

```python
def test_checked_type_create_returns_source_data_when_input_is_instance_of_cls():
    class TestClass:
        _checked_types = {}

    source_data = TestClass()
    result = _checked_type_create(TestClass, source_data)
    assert result is source_data


# LLM-generated content at query #22
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

def test_checked_pmap_constructor_with_wrong_types():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({"1": 1.0, 2: "2.0"})
        assert False, "Expected CheckedKeyTypeError or CheckedValueTypeError"
    except (CheckedKeyTypeError, CheckedValueTypeError):
        pass

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap(size=5, initial={1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.0, 2: 2.0}

def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #23
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
        _checked_types = []

        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)

    class TestClass(CheckedType):
        _checked_types = ['module.CheckedType']

    result = _checked_type_create(TestClass, ["data1", "data2"])
    assert isinstance(result, TestClass)
    assert len(result) == 2
    assert all(isinstance(item, CheckedType) for item in result)

def test__checked_type_create_with_matching_type_in_list():
    class CheckedType:
        _checked_types = []

    class TestClass(CheckedType):
        _checked_types = ['module.CheckedType']

    data = [CheckedType(), CheckedType()]
    result = _checked_type_create(TestClass, data)
    assert isinstance(result, TestClass)
    assert result == data


# LLM-generated content at query #24
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
        TestVector(["invalid"])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_constructor_with_invalid_invariant():
    class TestVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    try:
        TestVector([-1, -2])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_invariants_not_all_callable():
    dct = {'source_name': 'not_callable'}
    bases = []
    destination_name = 'destination'
    source_name = 'source_name'

    try:
        store_invariants(dct, bases, destination_name, source_name)
        assert False, "Expected TypeError was not raised"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'


# LLM-generated content at query #26
#--------------------------

```python
def test_checked_pmap_constructor_with_valid_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.5, 2: 2.25}

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=10)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_invalid_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.5, 2: 2.25})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_checked_pmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({"1": 1.5, 2: 2.25})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_checked_pmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: "1.5", 2: 2.25})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass


# LLM-generated content at query #27
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

def test_checked_pmap_constructor_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_evaluates_to_true():
    dct = {'source_name': 'value'}
    bases = []
    source_name = 'source_name'
    assert source_name in dct


# LLM-generated content at query #29
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


# LLM-generated content at query #30
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
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap(size=5)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_empty_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {}

def test_checked_pmap_constructor_with_type_mismatch():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({'a': 1.0})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


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

    result = IntToFloatMap(size=4, initial={1: 1.0, 2: 2.0})
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


# LLM-generated content at query #32
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
    except CheckedValueTypeError:
        pass

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

def test_check_types_with_custom_exception():
    it = [1, 'a', 3]
    expected_types = [int]
    source_class = list
    custom_exception = ValueError
    try:
        _check_types(it, expected_types, source_class, custom_exception)
    except ValueError:
        pass


# LLM-generated content at query #33
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
    cpv = CheckedPVector([1, 2, 3])
    result = CheckedPVector(cpv)
    assert isinstance(result, CheckedPVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #34
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    assert _check_types(it, expected_types, source_class) is None


# LLM-generated content at query #35
#--------------------------

```python
def test_check_types_with_valid_types():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = int
    _check_types(it, expected_types, source_class)


