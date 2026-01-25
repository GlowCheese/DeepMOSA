####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_invariant_errors_no_errors():
    elem = 5
    invariants = [lambda x: (True, None), lambda x: (True, "ok")]
    result = _invariant_errors(elem, invariants)
    expected = []
    assert result == expected

def test_invariant_errors_some_errors():
    elem = "test"
    invariants = [lambda x: (True, None), lambda x: (False, "error1"), lambda x: (False, "error2")]
    result = _invariant_errors(elem, invariants)
    expected = ["error1", "error2"]
    assert result == expected

def test_invariant_errors_all_errors():
    elem = []
    invariants = [lambda x: (False, "err1"), lambda x: (False, "err2")]
    result = _invariant_errors(elem, invariants)
    expected = ["err1", "err2"]
    assert result == expected

def test_invariant_errors_empty_invariants():
    elem = object()
    invariants = []
    result = _invariant_errors(elem, invariants)
    expected = []
    assert result == expected

def test_invariant_errors_passes_elem_to_invariants():
    captured = []
    def capture_inv(e):
        captured.append(e)
        return (True, None)
    elem = 42
    invariants = [capture_inv]
    _invariant_errors(elem, invariants)
    assert captured == [elem]


# LLM-generated content at query #2
#--------------------------

def test_constructor_creates_empty_checkedpset():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet()
    assert result == set()

def test_constructor_accepts_iterable_initial():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet([1, 2, 3])
    assert result == {1, 2, 3}

def test_constructor_enforces_type_constraint():
    class TestSet(CheckedPSet):
        __type__ = int
    try:
        TestSet([1, 'a'])
    except TypeError:
        pass
    else:
        assert False

def test_constructor_enforces_invariant_constraint():
    class Positives(CheckedPSet):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -1])
    except InvariantException:
        pass
    else:
        assert False

def test_constructor_with_pmap_initial():
    class TestSet(CheckedPSet):
        __type__ = int
    pmap_initial = pmap({1: True, 2: True})
    result = TestSet(pmap_initial)
    assert result == {1, 2}

def test_constructor_duplicates_removed():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet([1, 1, 2, 2])
    assert result == {1, 2}

def test_constructor_repr_custom_class_name():
    class TestSet(CheckedPSet):
        __type__ = int
    instance = TestSet([1, 2])
    assert repr(instance).startswith('TestSet')

def test_constructor_str_matches_repr():
    class TestSet(CheckedPSet):
        __type__ = int
    instance = TestSet([1, 2])
    assert str(instance) == repr(instance)

def test_constructor_serialize_produces_set():
    class TestSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda format, v: str(v)
    instance = TestSet([1, 2])
    serialized = instance.serialize()
    assert serialized == {'1', '2'}

def test_constructor_pickle_support():
    class TestSet(CheckedPSet):
        __type__ = int
    import pickle
    original = TestSet([1, 2, 3])
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)
    assert original == restored

def test_constructor_evolver_returns_evolver_instance():
    class TestSet(CheckedPSet):
        __type__ = int
    instance = TestSet([1, 2])
    evolver = instance.evolver()
    evolver.add(3)
    new_instance = evolver.persistent()
    assert new_instance == {1, 2, 3}

def test_constructor_create_classmethod():
    class TestSet(CheckedPSet):
        __type__ = int
    instance = TestSet.create([1, 2, 3])
    assert instance == {1, 2, 3}

def test_constructor_inheritance_type_check():
    class BaseSet(CheckedPSet):
        __type__ = int
    class DerivedSet(BaseSet):
        pass
    result = DerivedSet([1, 2])
    assert result == {1, 2}

def test_constructor_empty_invariant():
    class TestSet(CheckedPSet):
        __type__ = int
        __invariant__ = None
    result = TestSet([1, -1])
    assert result == {1, -1}

def test_constructor_multiple_types():
    class TestSet(CheckedPSet):
        __type__ = (int, str)
    result = TestSet([1, 'a'])
    assert result == {1, 'a'}


# LLM-generated content at query #3
#--------------------------

def test_maybe_parse_user_type_with_preserved_type():
    result = maybe_parse_user_type(list)
    assert result == [list]

def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]

def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    result = maybe_parse_user_type((int, str))
    assert result == (int, str)

def test_maybe_parse_user_type_with_nested_iterable():
    result = maybe_parse_user_type(([int, str], float))
    assert result == (int, str, float)

def test_maybe_parse_user_type_with_empty_iterable():
    result = maybe_parse_user_type([])
    assert result == ()

def test_maybe_parse_user_type_raises_type_error_for_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False
    except TypeError:
        assert True

def test_maybe_parse_user_type_with_tuple_of_strings():
    result = maybe_parse_user_type(("str", "int"))
    assert result == ("str", "int")

def test_maybe_parse_user_type_with_list_of_types():
    result = maybe_parse_user_type([list, dict])
    assert result == (list, dict)

def test_maybe_parse_user_type_with_complex_nested_iterable():
    result = maybe_parse_user_type((int, (str, [float])))
    assert result == (int, str, float)


# LLM-generated content at query #4
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial

def test_constructor_with_checkedpmap_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(initial)
    assert result is initial

def test_constructor_type_check_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_type_check_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'a'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_invariant_check():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({}, size=0)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_size_and_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == {1: 1.5}

def test_constructor_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v > 0, 'Value non-positive')]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException as e:
        assert len(e.error_codes) == 1
    try:
        IntToFloatMap({1: 0.0})
        assert False
    except InvariantException as e:
        assert len(e.error_codes) == 1

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == "IntToFloatMap({1: 1.5})"


# LLM-generated content at query #5
#--------------------------

def test__restore_pickle():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields):
            return (cls, data, _factory_fields)
    data = {"key": "value"}
    result = MockClass._restore_pickle(data)
    expected = (MockClass, data, set())
    assert result == expected


# LLM-generated content at query #6
#--------------------------

def test_check_types_with_valid_types():
    expected_types = [int, str]
    source_class = list
    it = [1, "hello", 2, "world"]
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_type():
    expected_types = [int]
    source_class = list
    it = [1, "invalid"]
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == (int,)
        assert e.actual_type == str

def test_check_types_with_no_expected_types():
    expected_types = None
    source_class = list
    it = [1, "hello", 2.5]
    _check_types(it, expected_types, source_class)

def test_check_types_with_empty_iterable():
    expected_types = [int]
    source_class = list
    it = []
    _check_types(it, expected_types, source_class)

def test_check_types_with_custom_exception_type():
    class CustomError(Exception):
        pass
    expected_types = [int]
    source_class = list
    it = ["invalid"]
    try:
        _check_types(it, expected_types, source_class, CustomError)
        assert False
    except CustomError:
        assert True

def test_check_types_with_type_string():
    expected_types = ["builtins.int", "builtins.str"]
    source_class = list
    it = [1, "hello"]
    _check_types(it, expected_types, source_class)

def test_check_types_with_mixed_type_and_string():
    expected_types = [int, "builtins.str"]
    source_class = list
    it = [1, "hello"]
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_type_string():
    expected_types = ["builtins.int"]
    source_class = list
    it = ["invalid"]
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == (int,)
        assert e.actual_type == str


# LLM-generated content at query #7
#--------------------------

def test_merge_invariant_results_all_true():
    result = [(True, []), (True, []), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_one_false():
    result = [(True, []), (False, ["error1"]), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)

def test_merge_invariant_results_multiple_false():
    result = [(False, ["error1"]), (False, ["error2"]), (False, ["error3"])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2", "error3")

def test_merge_invariant_results_empty_input():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_mixed_data():
    result = [(True, []), (False, ["err1", "err2"]), (False, ["err3"])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == (["err1", "err2"], "err3")


# LLM-generated content at query #8
#--------------------------

def test_store_types_single_dict():
    dct = {'source': int}
    _store_types(dct, [], 'dest', 'source')
    assert dct['dest'] == (int,)

def test_store_types_multiple_dicts():
    dct = {'source': str}
    base1 = type('Base1', (), {'source': float})
    base2 = type('Base2', (), {})
    _store_types(dct, [base1, base2], 'dest', 'source')
    assert dct['dest'] == (str, float)

def test_store_types_string_type():
    dct = {'source': 'MyClass'}
    _store_types(dct, [], 'dest', 'source')
    assert dct['dest'] == ('MyClass',)

def test_store_types_iterable_source():
    dct = {'source': [int, str]}
    _store_types(dct, [], 'dest', 'source')
    assert dct['dest'] == (int, str)

def test_store_types_preserved_iterable():
    dct = {'source': list}
    _store_types(dct, [], 'dest', 'source')
    assert dct['dest'] == (list,)

def test_store_types_no_source_in_dict():
    dct = {}
    _store_types(dct, [], 'dest', 'source')
    assert dct['dest'] == ()

def test_store_types_source_in_base_only():
    dct = {}
    base = type('Base', (), {'source': bool})
    _store_types(dct, [base], 'dest', 'source')
    assert dct['dest'] == (bool,)

def test_store_types_multiple_sources():
    dct = {'source': complex}
    base1 = type('Base1', (), {'source': bytes})
    base2 = type('Base2', (), {'source': memoryview})
    _store_types(dct, [base1, base2], 'dest', 'source')
    assert dct['dest'] == (complex, bytes, memoryview)

def test_store_types_nested_iterable():
    dct = {'source': [[int, str], float]}
    _store_types(dct, [], 'dest', 'source')
    assert dct['dest'] == (int, str, float)

def test_store_types_invalid_type_raises():
    dct = {'source': 123}
    try:
        _store_types(dct, [], 'dest', 'source')
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #9
#--------------------------

def test_checkedtype_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()
    assert hasattr(CheckedType, 'create')
    assert hasattr(CheckedType, 'serialize')
    assert callable(CheckedType.create)
    assert callable(CheckedType.serialize)


# LLM-generated content at query #10
#--------------------------

def test_serialize_with_default_format():
    class TestSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda format, v: str(v)
    test_set = TestSet([1, 2, 3])
    result = test_set.serialize()
    expected = {"1", "2", "3"}
    assert result == expected

def test_serialize_with_custom_format():
    class TestSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda format, v: f"{v}:{format}"
    test_set = TestSet([1, 2])
    result = test_set.serialize("hex")
    expected = {"1:hex", "2:hex"}
    assert result == expected

def test_serialize_empty_set():
    class TestSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda format, v: str(v)
    test_set = TestSet([])
    result = test_set.serialize()
    expected = set()
    assert result == expected

def test_serialize_with_complex_serializer():
    class TestSet(CheckedPSet):
        __type__ = str
        __serializer__ = lambda format, v: v.upper() if format == "upper" else v
    test_set = TestSet(["a", "b"])
    result = test_set.serialize("upper")
    expected = {"A", "B"}
    assert result == expected


# LLM-generated content at query #11
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert result == initial
    assert isinstance(result, IntToFloatMap)

def test_constructor_type_check_key_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {'a': 1.5}
    try:
        IntToFloatMap(initial)
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_type_check_value_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 'string'}
    try:
        IntToFloatMap(initial)
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    initial = {1: 2.5}
    try:
        IntToFloatMap(initial)
        assert False
    except InvariantException:
        assert True

def test_constructor_invariant_satisfied():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    initial = {1: 1.0}
    result = IntToFloatMap(initial)
    assert result == initial
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [
            lambda k, v: (k >= 0, 'Key must be non-negative'),
            lambda k, v: (v > 0, 'Value must be positive')
        ]
    initial = {1: 1.5}
    result = IntToFloatMap(initial)
    assert result == initial
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_multiple_invariants_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [
            lambda k, v: (k >= 0, 'Key must be non-negative'),
            lambda k, v: (v > 0, 'Value must be positive')
        ]
    initial = {-1: 1.5}
    try:
        IntToFloatMap(initial)
        assert False
    except InvariantException:
        assert True

def test_constructor_with_checked_type_key():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key_instance = KeyType()
    value_instance = ValueType()
    initial = {key_instance: value_instance}
    result = CustomMap(initial)
    assert result == initial
    assert isinstance(result, CustomMap)

def test_constructor_with_checked_type_creation():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return cls()
    class ValueType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return cls()
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    initial = {'raw_key': 'raw_value'}
    result = CustomMap(initial)
    assert all(isinstance(k, KeyType) for k in result.keys())
    assert all(isinstance(v, ValueType) for v in result.values())

def test_constructor_with_checked_type_and_regular_type_mix():
    class KeyType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = (KeyType, int)
        __value_type__ = float
    key_instance = KeyType()
    initial = {key_instance: 1.5, 2: 2.5}
    result = CustomMap(initial)
    assert result == initial
    assert isinstance(result, CustomMap)

def test_constructor_with_checked_type_and_regular_type_mix_creation():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return cls()
    class CustomMap(CheckedPMap):
        __key_type__ = (KeyType, int)
        __value_type__ = float
    initial = {'raw_key': 1.5, 2: 2.5}
    result = CustomMap(initial)
    assert any(isinstance(k, KeyType) for k in result.keys())
    assert any(isinstance(k, int) for k in result.keys())
    assert all(isinstance(v, float) for v in result.values())

def test_constructor_with_checked_type_no_creation_when_type_matches():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return cls()
    class CustomMap(CheckedPMap):
        __key_type__ = (KeyType, int)
        __value_type__ = float
    key_instance = KeyType()
    initial = {key_instance: 1.5}
    result = CustomMap(initial)
    assert key_instance in result
    assert result[key_instance] == 1.5

def test_constructor_with_checked_type_value_creation():
    class ValueType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return cls()
    class CustomMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = (ValueType, float)
    initial = {1: 'raw_value', 2: 2.5}
    result = CustomMap(initial)
    assert any(isinstance(v, ValueType) for v in result.values())
    assert any(isinstance(v, float) for v in result.values())

def test_constructor_with_checked_type_value_no_creation_when_type_matches():
    class ValueType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return cls()
    class CustomMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = (ValueType, float)
    value_instance = ValueType()
    initial = {1: value_instance}
    result = CustomMap(initial)
    assert result[1] is value_instance

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap(size=0)
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_size_and_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5}
    result = IntToFloatMap(initial, size=1)
    assert result == initial
    assert isinstance(result, IntToFloatMap)

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    expected_repr = "IntToFloatMap({1: 1.5})"
    assert repr(result) == expected_repr
    assert str(result) == expected_repr

def test_constructor_with_initial_as_checkedpmap_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(initial)
    assert result is initial

def test_constructor_with_initial_as_checkedpmap_subclass_instance():
    class ParentMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    class ChildMap(ParentMap):
        pass
    initial = ParentMap({1: 1.5})
    result = ChildMap(initial)
    assert result == initial
    assert isinstance(result, ChildMap)


# LLM-generated content at query #12
#--------------------------

def test__checked_type_create_with_same_class():
    class MockClass:
        _checked_types = []
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance

def test__checked_type_create_without_checked_types():
    class MockClass:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert result.data == source_data

def test__checked_type_create_with_checked_type_and_matching_data():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
        def __init__(self, data):
            self.data = data
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    checked_instance = CheckedType([1, 2])
    source_data = [checked_instance, checked_instance]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert result.data == source_data

def test__checked_type_create_with_checked_type_and_non_matching_data():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
        def __init__(self, data):
            self.data = data
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    source_data = [[1, 2], [3, 4]]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert all(isinstance(item, CheckedType) for item in result.data)
    assert result.data[0].data == [1, 2]
    assert result.data[1].data == [3, 4]

def test__checked_type_create_with_ignore_extra_true():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data, ignore_extra)
        def __init__(self, data, ignore_extra):
            self.data = data
            self.ignore_extra = ignore_extra
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    source_data = [[1, 2]]
    result = _checked_type_create(MockClass, source_data, ignore_extra=True)
    assert isinstance(result, MockClass)
    assert isinstance(result.data[0], CheckedType)
    assert result.data[0].ignore_extra == True

def test__checked_type_create_with_multiple_checked_types():
    class CheckedTypeA:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
        def __init__(self, data):
            self.data = data
    class CheckedTypeB:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
        def __init__(self, data):
            self.data = data
    class MockClass:
        _checked_types = ['__main__.CheckedTypeA', '__main__.CheckedTypeB']
        def __init__(self, data):
            self.data = data
    source_data = [CheckedTypeA([1]), CheckedTypeB([2])]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert result.data == source_data

def test__checked_type_create_with_mixed_data_and_checked_type():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
        def __init__(self, data):
            self.data = data
    class MockClass:
        _checked_types = ['__main__.CheckedType', int]
        def __init__(self, data):
            self.data = data
    source_data = [CheckedType([1]), 5]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert result.data == source_data


# LLM-generated content at query #13
#--------------------------

def test_constructor_with_empty_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector()
    assert isinstance(result, TestVector)
    assert len(result) == 0


def test_constructor_with_list_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector([1, 2, 3])
    assert isinstance(result, TestVector)
    assert list(result) == [1, 2, 3]


def test_constructor_with_python_pvector_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    pv = python_pvector([4, 5, 6])
    result = TestVector(pv)
    assert isinstance(result, TestVector)
    assert list(result) == [4, 5, 6]


def test_constructor_type_check_failure():
    class TestVector(CheckedPVector):
        __type__ = int
    try:
        TestVector([1, 'a', 3])
        assert False
    except TypeError:
        assert True


def test_constructor_invariant_check_failure():
    class Positives(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -2, 3])
        assert False
    except InvariantException:
        assert True


def test_constructor_invariant_check_success():
    class Positives(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 2, 3])
    assert isinstance(result, Positives)
    assert list(result) == [1, 2, 3]


def test_constructor_with_multiple_types():
    class TestVector(CheckedPVector):
        __type__ = (int, float)
    result = TestVector([1, 2.5, 3])
    assert isinstance(result, TestVector)
    assert list(result) == [1, 2.5, 3]


def test_constructor_with_no_type_specified():
    class TestVector(CheckedPVector):
        pass
    result = TestVector([1, 'a', 3.5])
    assert isinstance(result, TestVector)
    assert list(result) == [1, 'a', 3.5]


def test_constructor_with_iterable():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector(range(3))
    assert isinstance(result, TestVector)
    assert list(result) == [0, 1, 2]


# LLM-generated content at query #14
#--------------------------

def test_check_types_with_expected_types_and_matching_element():
    class MockType:
        pass
    class MockSourceClass:
        pass
    class MockExceptionType(Exception):
        pass
    it = [MockType()]
    expected_types = [MockType]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        result = True
    except MockExceptionType:
        result = False
    assert result == True

def test_check_types_with_expected_types_and_non_matching_element():
    class MockType:
        pass
    class OtherType:
        pass
    class MockSourceClass:
        pass
    class MockExceptionType(Exception):
        pass
    it = [OtherType()]
    expected_types = [MockType]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        result = False
    except MockExceptionType:
        result = True
    assert result == True

def test_check_types_with_empty_expected_types():
    class MockSourceClass:
        pass
    class MockExceptionType(Exception):
        pass
    it = [object()]
    expected_types = []
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        result = True
    except MockExceptionType:
        result = False
    assert result == True

def test_check_types_with_expected_types_and_multiple_matching_elements():
    class MockType:
        pass
    class MockSourceClass:
        pass
    class MockExceptionType(Exception):
        pass
    it = [MockType(), MockType()]
    expected_types = [MockType]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        result = True
    except MockExceptionType:
        result = False
    assert result == True

def test_check_types_with_expected_types_and_mixed_elements_first_non_matching():
    class MockType:
        pass
    class OtherType:
        pass
    class MockSourceClass:
        pass
    class MockExceptionType(Exception):
        pass
    it = [OtherType(), MockType()]
    expected_types = [MockType]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        result = False
    except MockExceptionType:
        result = True
    assert result == True


# LLM-generated content at query #15
#--------------------------

def test_serialize_returns_set_of_serialized_values():
    class MockSerializer:
        def __call__(self, format, value):
            return f"serialized_{value}"

    class TestCheckedPSet(CheckedPSet):
        __serializer__ = MockSerializer()

    test_instance = TestCheckedPSet([1, 2, 3])
    result = test_instance.serialize()
    expected = {"serialized_1", "serialized_2", "serialized_3"}
    assert result == expected
    assert isinstance(result, set)


# LLM-generated content at query #16
#--------------------------

def test_constructor_with_empty_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector()
    assert len(result) == 0
    assert isinstance(result, TestVector)


def test_constructor_with_list_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector([1, 2, 3])
    assert list(result) == [1, 2, 3]
    assert isinstance(result, TestVector)


def test_constructor_with_python_pvector_initial():
    from pyrsistent import pvector
    class TestVector(CheckedPVector):
        __type__ = int
    pv = pvector([4, 5, 6])
    result = TestVector(pv)
    assert list(result) == [4, 5, 6]
    assert isinstance(result, TestVector)


def test_constructor_type_check_passes():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector([7, 8, 9])
    assert all(isinstance(x, int) for x in result)


def test_constructor_type_check_fails():
    class TestVector(CheckedPVector):
        __type__ = int
    try:
        TestVector([10, 'invalid', 12])
        assert False
    except TypeError:
        assert True


def test_constructor_invariant_check_passes():
    class Positives(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 2, 3])
    assert list(result) == [1, 2, 3]


def test_constructor_invariant_check_fails():
    class Positives(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -2, 3])
        assert False
    except InvariantException:
        assert True


def test_constructor_with_multiple_types():
    class TestVector(CheckedPVector):
        __type__ = (int, float)
    result = TestVector([1, 2.5, 3])
    assert list(result) == [1, 2.5, 3]


def test_constructor_with_no_type_or_invariant():
    class TestVector(CheckedPVector):
        pass
    result = TestVector([1, 'a', 3.5])
    assert list(result) == [1, 'a', 3.5]


def test_constructor_returns_same_type():
    class TestVector(CheckedPVector):
        __type__ = int
    instance = TestVector([1, 2, 3])
    result = TestVector(instance)
    assert result is instance


# LLM-generated content at query #17
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert result == initial
    assert isinstance(result, IntToFloatMap)

def test_constructor_type_check_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_type_check_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'string'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_invariant_check():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_invariant_pass():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result == {1: 1.0, 2: 2.0}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_checked_key_type():
    class MyKey(CheckedType):
        pass
    class MyValue(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = MyKey
        __value_type__ = MyValue
    key = MyKey()
    value = MyValue()
    result = CustomMap({key: value})
    assert result == {key: value}
    assert isinstance(result, CustomMap)

def test_constructor_with_checked_value_type():
    class MyKey(CheckedType):
        pass
    class MyValue(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = MyKey
        __value_type__ = MyValue
    key = MyKey()
    value = MyValue()
    result = CustomMap({key: value})
    assert result == {key: value}
    assert isinstance(result, CustomMap)

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [
            lambda k, v: (k >= 0, 'Key negative'),
            lambda k, v: (v > 0, 'Value non-positive')
        ]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException as e:
        assert any('Key negative' in str(err) for err in e.error_codes)
    try:
        IntToFloatMap({1: -1.5})
        assert False
    except InvariantException as e:
        assert any('Value non-positive' in str(err) for err in e.error_codes)

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert result == {1: 1.5}
    assert isinstance(result, IntToFloatMap)


# LLM-generated content at query #18
#--------------------------

def test_maybe_parse_user_type_with_preserved_type():
    result = maybe_parse_user_type(list)
    assert result == [list]

def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]

def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    result = maybe_parse_user_type((int, str))
    assert result == (int, str)

def test_maybe_parse_user_type_with_nested_iterable():
    result = maybe_parse_user_type([(int, float), str])
    assert result == (int, float, str)

def test_maybe_parse_user_type_with_empty_iterable():
    result = maybe_parse_user_type([])
    assert result == ()

def test_maybe_parse_user_type_raises_type_error_for_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False
    except TypeError:
        assert True

def test_maybe_parse_user_type_with_tuple_of_strings():
    result = maybe_parse_user_type(("list", "dict"))
    assert result == ("list", "dict")

def test_maybe_parse_user_type_with_mixed_iterable():
    result = maybe_parse_user_type([int, "str", (float,)])
    assert result == (int, "str", float)


# LLM-generated content at query #19
#--------------------------

def test_store_invariants_adds_destination_name():
    class Base1:
        def inv1(self):
            return True, ()
    class Base2:
        def inv2(self):
            return True, ()
    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, 'invariants', 'inv1')
    assert len(dct['invariants']) == 1
    store_invariants(dct, bases, 'invariants', 'inv2')
    assert len(dct['invariants']) == 1

def test_store_invariants_inherits_from_bases():
    class Base:
        def base_inv(self):
            return True, ()
    class Derived(Base):
        def derived_inv(self):
            return True, ()
    dct = Derived.__dict__.copy()
    bases = Derived.__bases__
    store_invariants(dct, bases, 'invariants', 'base_inv')
    assert len(dct['invariants']) == 1
    store_invariants(dct, bases, 'invariants', 'derived_inv')
    assert len(dct['invariants']) == 1

def test_store_invariants_wraps_functions():
    def invariant():
        return True, ()
    dct = {'invariant': invariant}
    bases = ()
    store_invariants(dct, bases, 'invariants', 'invariant')
    stored = dct['invariants'][0]
    assert stored() == (True, ())

def test_store_invariants_merges_results():
    def invariant():
        return [(True, ()), (False, ('error',))]
    dct = {'invariant': invariant}
    bases = ()
    store_invariants(dct, bases, 'invariants', 'invariant')
    stored = dct['invariants'][0]
    assert stored() == (False, ('error',))

def test_store_invariants_raises_type_error_for_non_callable():
    dct = {'invariant': 'not callable'}
    bases = ()
    try:
        store_invariants(dct, bases, 'invariants', 'invariant')
        assert False
    except TypeError:
        pass

def test_store_invariants_handles_multiple_inheritance():
    class BaseA:
        def inv_a(self):
            return True, ()
    class BaseB:
        def inv_b(self):
            return True, ()
    class Derived(BaseA, BaseB):
        pass
    dct = Derived.__dict__.copy()
    bases = Derived.__bases__
    store_invariants(dct, bases, 'invariants', 'inv_a')
    assert len(dct['invariants']) == 1
    store_invariants(dct, bases, 'invariants', 'inv_b')
    assert len(dct['invariants']) == 1

def test_store_invariants_skips_duplicate_classes():
    class Base:
        def inv(self):
            return True, ()
    class Derived1(Base):
        pass
    class Derived2(Base):
        pass
    class Multi(Derived1, Derived2):
        pass
    dct = Multi.__dict__.copy()
    bases = Multi.__bases__
    store_invariants(dct, bases, 'invariants', 'inv')
    assert len(dct['invariants']) == 1

def test_store_invariants_empty_when_no_source():
    dct = {}
    bases = ()
    store_invariants(dct, bases, 'invariants', 'nonexistent')
    assert len(dct['invariants']) == 0

def test_store_invariants_preserves_existing_destination():
    def inv1():
        return True, ()
    def inv2():
        return True, ()
    dct = {'inv1': inv1, 'inv2': inv2}
    bases = ()
    store_invariants(dct, bases, 'invariants', 'inv1')
    assert len(dct['invariants']) == 1
    store_invariants(dct, bases, 'invariants', 'inv2')
    assert len(dct['invariants']) == 1


# LLM-generated content at query #20
#--------------------------

def test_check_types_with_valid_types():
    expected_types = [int, float]
    source_class = list
    it = [1, 2.5, 3]
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_type():
    expected_types = [int]
    source_class = list
    it = [1, "string", 3]
    exception_raised = False
    try:
        _check_types(it, expected_types, source_class)
    except CheckedValueTypeError:
        exception_raised = True
    assert exception_raised

def test_check_types_with_no_expected_types():
    expected_types = []
    source_class = list
    it = [1, "string", 3.14]
    _check_types(it, expected_types, source_class)

def test_check_types_with_string_type_names():
    expected_types = ["builtins.int", "builtins.float"]
    source_class = list
    it = [1, 2.5]
    _check_types(it, expected_types, source_class)

def test_check_types_with_mixed_type_objects_and_strings():
    expected_types = [int, "builtins.float"]
    source_class = list
    it = [1, 2.5]
    _check_types(it, expected_types, source_class)

def test_check_types_with_custom_exception_type():
    expected_types = [int]
    source_class = list
    it = [1, "string"]
    exception_raised = False
    try:
        _check_types(it, expected_types, source_class, exception_type=ValueError)
    except ValueError:
        exception_raised = True
    assert exception_raised

def test_check_types_with_empty_iterable():
    expected_types = [int]
    source_class = list
    it = []
    _check_types(it, expected_types, source_class)

def test_check_types_with_non_iterable():
    expected_types = [int]
    source_class = list
    it = 42
    exception_raised = False
    try:
        _check_types(it, expected_types, source_class)
    except TypeError:
        exception_raised = True
    assert exception_raised


# LLM-generated content at query #21
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({})
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_valid_key_value_types():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_invalid_key_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_with_invalid_value_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'invalid'})
        assert False
    except CheckedValueTypeError:
        assert True

def test_constructor_with_invariant_violation_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result == {1: 1.0, 2: 2.0}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k > 0, 'Key must be positive'), lambda k, v: (v > 0, 'Value must be positive')]
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_multiple_invariants_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k > 0, 'Key must be positive'), lambda k, v: (v > 0, 'Value must be positive')]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_checked_key_type():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key_instance = KeyType()
    value_instance = ValueType()
    result = CustomMap({key_instance: value_instance})
    assert result == {key_instance: value_instance}
    assert isinstance(result, CustomMap)

def test_constructor_with_initial_as_same_class_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result == {1: 1.5}
    assert result is original

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_undefined_size_constant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=_UNDEFINED_CHECKED_PMAP_SIZE)
    assert result == {1: 1.5}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_non_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap([(1, 1.5), (2, 2.25)])
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_checked_type_create_method():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return source_data * 2
    class ValueType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return source_data + 1
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = CustomMap.create({1: 5})
    assert result == {2: 6}
    assert isinstance(result, CustomMap)

def test_constructor_with_checked_type_create_method_and_existing_type():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return source_data * 2
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = (int, KeyType)
        __value_type__ = (float, ValueType)
    result = CustomMap.create({1: 5.0})
    assert result == {1: 5.0}
    assert isinstance(result, CustomMap)

def test_constructor_with_ignore_extra_in_create():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap.create({1: 1.5}, ignore_extra=True)
    assert result == {1: 1.5}
    assert isinstance(result, IntToFloatMap)

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_str():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert str(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_serialize():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __serializer__ = lambda format, k, v: (str(k), str(v))
    result = IntToFloatMap({1: 1.5})
    serialized = result.serialize()
    assert serialized == {'1': '1.5'}

def test_constructor_pickle_support():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    import pickle
    original = IntToFloatMap({1: 1.5})
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)
    assert restored == original
    assert isinstance(restored, IntToFloatMap)


# LLM-generated content at query #22
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial

def test_constructor_respects_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial

def test_constructor_respects_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial

def test_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    initial = {1: 1.0}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial

def test_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    initial = {1: 2.0}
    try:
        IntToFloatMap(initial)
        assert False
    except InvariantException as e:
        assert 'Invalid mapping' in str(e)

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v > 0, 'Value non-positive')]
    initial = {1: 1.5}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial

def test_constructor_with_multiple_invariant_violations():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v > 0, 'Value non-positive')]
    initial = {-1: 0.0}
    try:
        IntToFloatMap(initial)
        assert False
    except InvariantException as e:
        errors = str(e)
        assert 'Key negative' in errors
        assert 'Value non-positive' in errors

def test_constructor_type_check_failure_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {'a': 1.5}
    try:
        IntToFloatMap(initial)
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_type_check_failure_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 'not_a_float'}
    try:
        IntToFloatMap(initial)
        assert False
    except InvariantException:
        assert True

def test_constructor_with_checked_key_type():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key = KeyType()
    value = ValueType()
    initial = {key: value}
    result = CustomMap(initial)
    assert isinstance(result, CustomMap)
    assert dict(result) == initial

def test_constructor_with_checked_value_type():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key = KeyType()
    value = ValueType()
    initial = {key: value}
    result = CustomMap(initial)
    assert isinstance(result, CustomMap)
    assert dict(result) == initial

def test_constructor_with_checked_type_create():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
    class ValueType(CheckedType):
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    initial = {'raw_key': 'raw_value'}
    result = CustomMap(initial)
    assert isinstance(result, CustomMap)
    assert all(isinstance(k, KeyType) for k in result.keys())
    assert all(isinstance(v, ValueType) for v in result.values())

def test_constructor_with_checked_type_create_not_needed():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key = KeyType()
    value = ValueType()
    initial = {key: value}
    result = CustomMap(initial)
    assert isinstance(result, CustomMap)
    assert dict(result) == initial

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5}
    result = IntToFloatMap(initial, size=1)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial

def test_constructor_with_undefined_size():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5}
    result = IntToFloatMap(initial, size=_UNDEFINED_CHECKED_PMAP_SIZE)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5}
    result = IntToFloatMap(initial)
    expected_repr = "IntToFloatMap({1: 1.5})"
    assert repr(result) == expected_repr

def test_constructor_str():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5}
    result = IntToFloatMap(initial)
    expected_str = "IntToFloatMap({1: 1.5})"
    assert str(result) == expected_str

def test_constructor_with_self_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    instance = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(instance)
    assert result is instance

def test_constructor_with_checked_pmap_subclass_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    class SubMap(IntToFloatMap):
        pass
    instance = SubMap({1: 1.5})
    result = IntToFloatMap(instance)
    assert dict(result) == dict(instance)


# LLM-generated content at query #23
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert result == IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)


def test_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert dict(result) == initial
    assert isinstance(result, IntToFloatMap)


def test_constructor_type_check_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True


def test_constructor_type_check_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'string'})
        assert False
    except CheckedTypeError:
        assert True


def test_constructor_invariant_check():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True


def test_constructor_invariant_pass():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert dict(result) == {1: 1.0, 2: 2.0}


def test_constructor_with_checked_key_type():
    class InnerChecked(CheckedType):
        pass
    class MyMap(CheckedPMap):
        __key_type__ = InnerChecked
        __value_type__ = int
    source = {InnerChecked(): 5}
    result = MyMap(source)
    assert isinstance(list(result.keys())[0], InnerChecked)
    assert list(result.values())[0] == 5


def test_constructor_with_checked_value_type():
    class InnerChecked(CheckedType):
        pass
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = InnerChecked
    source = {5: InnerChecked()}
    result = MyMap(source)
    assert list(result.keys())[0] == 5
    assert isinstance(list(result.values())[0], InnerChecked)


def test_constructor_preserves_subclass_type():
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    instance = MyMap({1: 'a'})
    result = MyMap(instance)
    assert result is instance


def test_constructor_repr():
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    instance = MyMap({1: 'a'})
    assert repr(instance) == "MyMap({1: 'a'})"


# LLM-generated content at query #24
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_checkedpmap_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 1
    assert result[1] == 1.5

def test_constructor_type_check_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({"a": 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_type_check_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: "string"})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_invariant_check():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_invariant_check_success():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({}, size=0)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_size_and_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 1
    assert result[1] == 1.5

def test_constructor_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k > 0, 'Key not positive'), lambda k, v: (v > 0, 'Value not positive')]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException as e:
        assert len(e.error_codes) == 1
    try:
        IntToFloatMap({1: -1.5})
        assert False
    except InvariantException as e:
        assert len(e.error_codes) == 1
    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5


# LLM-generated content at query #25
#--------------------------

def test_merge_invariant_results_all_true():
    result = [(True, []), (True, []), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_one_false():
    result = [(True, []), (False, "error1"), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)

def test_merge_invariant_results_multiple_false():
    result = [(False, "error1"), (False, "error2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2", "error3")

def test_merge_invariant_results_empty():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_mixed_data():
    result = [(True, []), (False, 42), (False, "test"), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == (42, "test")


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_with_empty_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector()
    assert isinstance(result, TestVector)
    assert len(result) == 0


def test_constructor_with_list_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector([1, 2, 3])
    assert isinstance(result, TestVector)
    assert list(result) == [1, 2, 3]


def test_constructor_with_tuple_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector((1, 2, 3))
    assert isinstance(result, TestVector)
    assert list(result) == [1, 2, 3]


def test_constructor_with_python_pvector_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    pv = python_pvector([1, 2, 3])
    result = TestVector(pv)
    assert isinstance(result, TestVector)
    assert list(result) == [1, 2, 3]


def test_constructor_type_check_failure():
    class TestVector(CheckedPVector):
        __type__ = int
    try:
        TestVector([1, "two", 3])
        assert False
    except TypeError:
        assert True


def test_constructor_invariant_check_failure():
    class Positives(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -2, 3])
        assert False
    except InvariantException:
        assert True


def test_constructor_with_invariant_success():
    class Positives(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 2, 3])
    assert isinstance(result, Positives)
    assert list(result) == [1, 2, 3]


def test_constructor_with_multiple_types():
    class TestVector(CheckedPVector):
        __type__ = (int, float)
    result = TestVector([1, 2.5, 3])
    assert isinstance(result, TestVector)
    assert list(result) == [1, 2.5, 3]


def test_constructor_with_no_type_or_invariant():
    class TestVector(CheckedPVector):
        pass
    result = TestVector([1, "two", 3.0])
    assert isinstance(result, TestVector)
    assert list(result) == [1, "two", 3.0]


def test_constructor_returns_same_type():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector([1, 2, 3])
    assert result.__class__.__name__ == "TestVector"


# LLM-generated content at query #2
#--------------------------

def test___new___creates_checked_pset_from_empty_initial():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet()
    assert isinstance(result, TestSet)
    assert len(result) == 0


def test___new___creates_checked_pset_from_iterable_initial():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet([1, 2, 3])
    assert isinstance(result, TestSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test___new___creates_checked_pset_from_pmap():
    class TestSet(CheckedPSet):
        __type__ = int
    pmap_instance = pmap({1: True, 2: True})
    result = TestSet(pmap_instance)
    assert isinstance(result, TestSet)
    assert len(result) == 2
    assert 1 in result
    assert 2 in result


def test___new___enforces_type_check_on_iterable_initial():
    class TestSet(CheckedPSet):
        __type__ = int
    try:
        TestSet([1, 'a'])
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test___new___enforces_invariant_on_iterable_initial():
    class PositiveSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, 'Must be positive')
    try:
        PositiveSet([1, -1])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


def test___new___handles_duplicates_in_iterable():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet([1, 2, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test___new___returns_empty_checked_pset_for_empty_iterable():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet([])
    assert isinstance(result, TestSet)
    assert len(result) == 0


# LLM-generated content at query #3
#--------------------------

def test__checked_type_create_with_same_class():
    class MockClass:
        pass
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance

def test__checked_type_create_with_checked_type_subclass():
    class CheckedType:
        _checked_types = ["__main__.MockSubType"]
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
    class MockSubType(CheckedType):
        def __init__(self, data):
            self.data = data
    __import__('__main__', fromlist=['MockSubType'])
    source_data = [1, 2, 3]
    result = _checked_type_create(CheckedType, source_data)
    assert isinstance(result, CheckedType)
    assert all(isinstance(item, MockSubType) for item in result.data)

def test__checked_type_create_without_checked_type():
    class MockClass:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert result.data == source_data

def test__checked_type_create_with_ignore_extra():
    class CheckedType:
        _checked_types = ["__main__.MockSubType"]
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
    class MockSubType(CheckedType):
        def __init__(self, data):
            self.data = data
    __import__('__main__', fromlist=['MockSubType'])
    source_data = [1, 2, 3]
    result = _checked_type_create(CheckedType, source_data, ignore_extra=True)
    assert isinstance(result, CheckedType)
    assert all(isinstance(item, MockSubType) for item in result.data)

def test__checked_type_create_with_matching_type():
    class CheckedType:
        _checked_types = ["__main__.MockSubType", "__main__.OtherType"]
        def __init__(self, data):
            self.data = data
    class MockSubType:
        pass
    class OtherType:
        pass
    __import__('__main__', fromlist=['MockSubType', 'OtherType'])
    source_data = [MockSubType(), OtherType()]
    result = _checked_type_create(CheckedType, source_data)
    assert isinstance(result, CheckedType)
    assert result.data == source_data


# LLM-generated content at query #4
#--------------------------

def test_CheckedType_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()
    assert hasattr(instance, 'create')
    assert hasattr(instance, 'serialize')


# LLM-generated content at query #5
#--------------------------

def test_serialize_with_custom_serializer():
    class CustomSerializer:
        def __call__(self, format, key, value):
            return (f"{key}_{format}", f"{value}_{format}")
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
        __serializer__ = CustomSerializer()
    instance = TestMap({1: "a", 2: "b"})
    result = instance.serialize("fmt")
    expected = {"1_fmt": "a_fmt", "2_fmt": "b_fmt"}
    assert result == expected

def test_serialize_without_format():
    class CustomSerializer:
        def __call__(self, format, key, value):
            return (key, f"{value}:{format}")
    class TestMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
        __serializer__ = CustomSerializer()
    instance = TestMap({"x": 10, "y": 20})
    result = instance.serialize()
    expected = {"x": "10:None", "y": "20:None"}
    assert result == expected

def test_serialize_empty_map():
    class CustomSerializer:
        def __call__(self, format, key, value):
            return (key, value)
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
        __serializer__ = CustomSerializer()
    instance = TestMap({})
    result = instance.serialize("any")
    expected = {}
    assert result == expected

def test_serialize_uses_defined_serializer():
    class FixedSerializer:
        def __call__(self, format, key, value):
            return ("fixed_key", "fixed_value")
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
        __serializer__ = FixedSerializer()
    instance = TestMap({1: "a", 2: "b"})
    result = instance.serialize()
    expected = {"fixed_key": "fixed_value", "fixed_key": "fixed_value"}
    assert list(result.keys()) == ["fixed_key", "fixed_key"]
    assert list(result.values()) == ["fixed_value", "fixed_value"]

def test_serialize_preserves_order_from_items():
    class OrderedSerializer:
        def __call__(self, format, key, value):
            return (key, value)
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
        __serializer__ = OrderedSerializer()
    instance = TestMap({3: "c", 1: "a", 2: "b"})
    result = instance.serialize()
    keys = list(result.keys())
    values = list(result.values())
    assert keys == [3, 1, 2]
    assert values == ["c", "a", "b"]


# LLM-generated content at query #6
#--------------------------

def test__restore_pickle():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields):
            return (cls, data, _factory_fields)
    data = "test_data"
    result = MockClass._restore_pickle(data)
    expected = (MockClass, data, set())
    assert result == expected


# LLM-generated content at query #7
#--------------------------

def test_invariant_errors_no_errors():
    elem = 5
    invariants = [lambda x: (True, "ok1"), lambda x: (True, "ok2")]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_single_error():
    elem = "test"
    invariants = [lambda x: (False, "error1"), lambda x: (True, "ok")]
    result = _invariant_errors(elem, invariants)
    assert result == ["error1"]

def test_invariant_errors_multiple_errors():
    elem = []
    invariants = [lambda x: (False, "err1"), lambda x: (False, "err2"), lambda x: (False, "err3")]
    result = _invariant_errors(elem, invariants)
    assert result == ["err1", "err2", "err3"]

def test_invariant_errors_mixed_results():
    elem = 3.14
    invariants = [lambda x: (True, "pass1"), lambda x: (False, "fail1"), lambda x: (True, "pass2"), lambda x: (False, "fail2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["fail1", "fail2"]

def test_invariant_errors_empty_invariants():
    elem = object()
    invariants = []
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_all_pass():
    elem = None
    invariants = [lambda x: (True, "all good"), lambda x: (True, "fine")]
    result = _invariant_errors(elem, invariants)
    assert result == []


# LLM-generated content at query #8
#--------------------------

def test_store_types_single_dict():
    dct = {'source': int}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (int,)

def test_store_types_multiple_dicts():
    dct = {'source': str}
    base = type('Base', (), {'source': float})
    _store_types(dct, [base], 'destination', 'source')
    assert dct['destination'] == (str, float)

def test_store_types_no_source():
    dct = {}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == ()

def test_store_types_iterable_source():
    dct = {'source': [int, str]}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (int, str)

def test_store_types_string_source():
    dct = {'source': 'MyType'}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == ('MyType',)

def test_store_types_preserved_iterable():
    dct = {'source': tuple}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (tuple,)

def test_store_types_mixed_sources():
    dct = {'source': list}
    base1 = type('Base1', (), {'source': 'Custom'})
    base2 = type('Base2', (), {})
    _store_types(dct, [base1, base2], 'destination', 'source')
    assert dct['destination'] == (list, 'Custom')

def test_store_types_nested_iterable():
    dct = {'source': [[int, str], float]}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (int, str, float)

def test_store_types_overwrites_destination():
    dct = {'source': bool, 'destination': 'old'}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (bool,)

def test_store_types_invalid_source_raises():
    dct = {'source': 123}
    try:
        _store_types(dct, [], 'destination', 'source')
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #9
#--------------------------

def test_initial_is_pmap():
    from pyrsistent import pmap, pset
    from pyrsistent._checked_types import CheckedPSet
    class TestSet(CheckedPSet):
        __type__ = int
    pmap_instance = pmap({1: True, 2: True})
    result = TestSet(pmap_instance)
    assert isinstance(result, TestSet)
    assert len(result) == 2
    assert 1 in result
    assert 2 in result


# LLM-generated content at query #10
#--------------------------

def test_store_invariants_single_class_no_inheritance():
    class A:
        def invariant_A(self):
            return True, ()
    dct = {}
    bases = (A,)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert len(dct['invariants']) == 1
    assert dct['invariants'][0]() == (True, ())

def test_store_invariants_multiple_inheritance():
    class A:
        def invariant_A(self):
            return True, ()
    class B:
        def invariant_B(self):
            return False, ('error',)
    class C(A, B):
        pass
    dct = {}
    bases = (C,)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert len(dct['invariants']) == 2
    results = [inv() for inv in dct['invariants']]
    assert (True, ()) in results
    assert (False, ('error',)) in results

def test_store_invariants_inheritance_chain():
    class GrandParent:
        def invariant_GP(self):
            return True, ()
    class Parent(GrandParent):
        def invariant_P(self):
            return False, ('parent error',)
    class Child(Parent):
        pass
    dct = {}
    bases = (Child,)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert len(dct['invariants']) == 2
    results = [inv() for inv in dct['invariants']]
    assert (True, ()) in results
    assert (False, ('parent error',)) in results

def test_store_invariants_diamond_inheritance():
    class A:
        def invariant_A(self):
            return True, ()
    class B(A):
        pass
    class C(A):
        def invariant_C(self):
            return False, ('C error',)
    class D(B, C):
        pass
    dct = {}
    bases = (D,)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert len(dct['invariants']) == 2
    results = [inv() for inv in dct['invariants']]
    assert (True, ()) in results
    assert (False, ('C error',)) in results

def test_store_invariants_with_local_definition():
    class A:
        def invariant_A(self):
            return True, ()
    dct = {'invariant': lambda self: (False, ('local error',))}
    bases = (A,)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert len(dct['invariants']) == 2
    results = [inv() for inv in dct['invariants']]
    assert (True, ()) in results
    assert (False, ('local error',)) in results

def test_store_invariants_no_invariants():
    class A:
        pass
    dct = {}
    bases = (A,)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert dct['invariants'] == ()

def test_store_invariants_non_callable_raises_typeerror():
    class A:
        invariant = 'not a function'
    dct = {}
    bases = (A,)
    try:
        store_invariants(dct, bases, 'invariants', 'invariant')
        assert False
    except TypeError:
        assert True

def test_store_invariants_wrap_invariant_merges_results():
    def invariant_merging(self):
        return [(True, ()), (False, ('err1',)), (False, ('err2',))]
    dct = {'invariant': invariant_merging}
    bases = ()
    store_invariants(dct, bases, 'invariants', 'invariant')
    wrapped_inv = dct['invariants'][0]
    result = wrapped_inv()
    assert result == (False, ('err1', 'err2'))

def test_store_invariants_wrap_invariant_single_bool_result():
    def invariant_single(self):
        return False, ('single error',)
    dct = {'invariant': invariant_single}
    bases = ()
    store_invariants(dct, bases, 'invariants', 'invariant')
    wrapped_inv = dct['invariants'][0]
    result = wrapped_inv()
    assert result == (False, ('single error',))

def test_store_invariants_custom_destination_and_source_names():
    class A:
        def custom_invariant(self):
            return True, ()
    dct = {'custom_invariant': lambda self: (False, ('local',))}
    bases = (A,)
    store_invariants(dct, bases, 'dest', 'custom_invariant')
    assert len(dct['dest']) == 2
    results = [inv() for inv in dct['dest']]
    assert (True, ()) in results
    assert (False, ('local',)) in results


# LLM-generated content at query #11
#--------------------------

def test_constructor_with_valid_key_value_types():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_invalid_key_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
    except CheckedKeyTypeError:
        pass
    else:
        assert False, "Expected CheckedKeyTypeError"

def test_constructor_with_invalid_value_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'invalid'})
    except CheckedTypeError:
        pass
    else:
        assert False, "Expected CheckedTypeError"

def test_constructor_with_invariant_violation_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
    except InvariantException:
        pass
    else:
        assert False, "Expected InvariantException"

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result == {1: 1.0, 2: 2.0}

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({})
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_no_arguments():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_checked_type_key():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key = KeyType()
    value = ValueType()
    result = CustomMap({key: value})
    assert result == {key: value}

def test_constructor_with_checked_type_creation():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, data):
            return cls()
    class ValueType(CheckedType):
        @classmethod
        def create(cls, data):
            return cls()
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = CustomMap({'raw_key': 'raw_value'})
    assert all(isinstance(k, KeyType) for k in result.keys())
    assert all(isinstance(v, ValueType) for v in result.values())

def test_constructor_preserves_original_if_already_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result is original


# LLM-generated content at query #12
#--------------------------

def test___new___creates_empty_checkedpmap():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert result == {}
    assert isinstance(result, IntToFloatMap)


def test___new___creates_checkedpmap_from_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert result == initial
    assert isinstance(result, IntToFloatMap)


def test___new___creates_checkedpmap_from_iterable_of_pairs():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = [(1, 1.5), (2, 2.25)]
    result = IntToFloatMap(initial)
    assert dict(result) == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)


def test___new___with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=10)
    assert result == {1: 1.5}
    assert isinstance(result, IntToFloatMap)


def test___new___enforces_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True


def test___new___enforces_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'string'})
        assert False
    except CheckedTypeError:
        assert True


def test___new___enforces_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True


def test___new___passes_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0})
    assert result == {1: 1.0}
    assert isinstance(result, IntToFloatMap)


# LLM-generated content at query #13
#--------------------------

def test_serialize_with_default_format():
    class IntSet(CheckedPSet):
        __type__ = int
    pset = IntSet([1, 2, 3])
    result = pset.serialize()
    expected = {1, 2, 3}
    assert result == expected

def test_serialize_with_custom_serializer():
    class IntSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda format, v: str(v)
    pset = IntSet([1, 2, 3])
    result = pset.serialize()
    expected = {'1', '2', '3'}
    assert result == expected

def test_serialize_with_format_argument():
    class IntSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda format, v: f"{format}{v}"
    pset = IntSet([1, 2, 3])
    result = pset.serialize('num_')
    expected = {'num_1', 'num_2', 'num_3'}
    assert result == expected

def test_serialize_empty_set():
    class IntSet(CheckedPSet):
        __type__ = int
    pset = IntSet([])
    result = pset.serialize()
    expected = set()
    assert result == expected

def test_serialize_with_complex_serializer():
    class IntSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda format, v: {'value': v}
    pset = IntSet([1, 2])
    result = pset.serialize()
    expected = [{'value': 1}, {'value': 2}]
    assert all(item in result for item in expected) and len(result) == len(expected)


# LLM-generated content at query #14
#--------------------------

def test_store_invariants_inherits_and_stores_callables():
    from unittest.mock import Mock
    mock_invariant1 = Mock()
    mock_invariant2 = Mock()
    mock_invariant1.return_value = True
    mock_invariant2.return_value = True
    base1 = {'invariant': mock_invariant1}
    base2 = {'invariant': mock_invariant2}
    dct = {}
    store_invariants(dct, (base1, base2), 'dest', 'invariant')
    assert 'dest' in dct
    assert isinstance(dct['dest'], tuple)
    assert len(dct['dest']) == 2
    assert all(callable(inv) for inv in dct['dest'])

def test_store_invariants_raises_type_error_for_non_callable():
    base1 = {'invariant': lambda: True}
    base2 = {'invariant': 'not_callable'}
    dct = {}
    try:
        store_invariants(dct, (base1, base2), 'dest', 'invariant')
        assert False
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test_store_invariants_handles_missing_source_name():
    base1 = {}
    base2 = {'invariant': lambda: True}
    dct = {}
    store_invariants(dct, (base1, base2), 'dest', 'invariant')
    assert 'dest' in dct
    assert len(dct['dest']) == 1
    assert callable(dct['dest'][0])

def test_store_invariants_uses_dct_source_name_first():
    dct_invariant = lambda: 'dct'
    base_invariant = lambda: 'base'
    dct = {'invariant': dct_invariant}
    base = {'invariant': base_invariant}
    store_invariants(dct, (base,), 'dest', 'invariant')
    assert dct['dest'][0]() == 'dct'
    assert len(dct['dest']) == 2

def test_store_invariants_wraps_invariants():
    invariant = lambda x: x > 0
    dct = {'invariant': invariant}
    store_invariants(dct, (), 'dest', 'invariant')
    stored = dct['dest'][0]
    assert stored(5) is True
    assert stored(-1) is False


# LLM-generated content at query #15
#--------------------------

def test__check_types_with_matching_types():
    expected_types = [int, str]
    source_class = list
    exception_type = CheckedValueTypeError
    it = [1, "hello", 2, "world"]
    _check_types(it, expected_types, source_class, exception_type)

def test__check_types_with_non_matching_type():
    expected_types = [int]
    source_class = list
    exception_type = CheckedValueTypeError
    it = [1, "hello"]
    try:
        _check_types(it, expected_types, source_class, exception_type)
        assert False
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == (int,)
        assert e.actual_type == str
        assert e.value == "hello"

def test__check_types_with_no_expected_types():
    expected_types = None
    source_class = list
    exception_type = CheckedValueTypeError
    it = [1, "hello", 2.5]
    _check_types(it, expected_types, source_class, exception_type)

def test__check_types_with_empty_iterable():
    expected_types = [int]
    source_class = list
    exception_type = CheckedValueTypeError
    it = []
    _check_types(it, expected_types, source_class, exception_type)

def test__check_types_with_string_type_names():
    expected_types = ["builtins.int", "builtins.str"]
    source_class = list
    exception_type = CheckedValueTypeError
    it = [1, "hello"]
    _check_types(it, expected_types, source_class, exception_type)

def test__check_types_with_mixed_type_and_string():
    expected_types = [int, "builtins.str"]
    source_class = list
    exception_type = CheckedValueTypeError
    it = [1, "hello"]
    _check_types(it, expected_types, source_class, exception_type)

def test__check_types_with_custom_exception_type():
    class CustomError(Exception):
        pass
    expected_types = [int]
    source_class = list
    exception_type = CustomError
    it = [1, "hello"]
    try:
        _check_types(it, expected_types, source_class, exception_type)
        assert False
    except CustomError:
        pass

def test__check_types_with_none_in_iterable():
    expected_types = [int, type(None)]
    source_class = list
    exception_type = CheckedValueTypeError
    it = [1, None, 2]
    _check_types(it, expected_types, source_class, exception_type)


# LLM-generated content at query #16
#--------------------------

def test_constructor_with_empty_initial():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet()
    assert len(result) == 0
    assert isinstance(result, TestSet)

def test_constructor_with_list_initial():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet([1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    assert isinstance(result, TestSet)

def test_constructor_with_tuple_initial():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet((1, 2, 3))
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_constructor_with_set_initial():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet({1, 2, 3})
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_constructor_with_pmap_initial():
    class TestSet(CheckedPSet):
        __type__ = int
    pmap_initial = pmap({1: True, 2: True})
    result = TestSet(pmap_initial)
    assert len(result) == 2
    assert 1 in result
    assert 2 in result

def test_constructor_type_check_pass():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet([1, 2, 3])
    assert isinstance(result, TestSet)

def test_constructor_type_check_fail():
    class TestSet(CheckedPSet):
        __type__ = int
    try:
        TestSet([1, 'a', 3])
        assert False
    except TypeError:
        assert True

def test_constructor_invariant_check_pass():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result

def test_constructor_invariant_check_fail():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -2, 3])
        assert False
    except InvariantException:
        assert True

def test_constructor_duplicate_elements():
    class TestSet(CheckedPSet):
        __type__ = int
    result = TestSet([1, 2, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_constructor_no_type_or_invariant():
    class TestSet(CheckedPSet):
        pass
    result = TestSet([1, 'a', 3.5])
    assert len(result) == 3
    assert 1 in result
    assert 'a' in result
    assert 3.5 in result


# LLM-generated content at query #17
#--------------------------

def test_checked_type_create_with_checked_type_and_mismatched_data():
    class MockCheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}"

    class MockTypes:
        _checked_types = [MockCheckedType]

    types = [MockCheckedType]
    source_data = ["data1", "data2"]
    result = _checked_type_create(MockTypes, source_data, ignore_extra=False)
    assert result == ["created_data1", "created_data2"]


# LLM-generated content at query #18
#--------------------------

def test_invariant_exception_constructor_with_no_arguments():
    exception = InvariantException()
    assert exception.invariant_errors == ()
    assert exception.missing_fields == ()
    assert str(exception) == ", invariant_errors=[], missing_fields=[]"

def test_invariant_exception_constructor_with_error_codes():
    error_data = ("error1", "error2")
    exception = InvariantException(error_codes=error_data)
    assert exception.invariant_errors == ("error1", "error2")
    assert exception.missing_fields == ()
    assert str(exception) == ", invariant_errors=[error1, error2], missing_fields=[]"

def test_invariant_exception_constructor_with_callable_error_codes():
    def error_func():
        return "dynamic_error"
    exception = InvariantException(error_codes=(error_func,))
    assert exception.invariant_errors == ("dynamic_error",)
    assert exception.missing_fields == ()
    assert str(exception) == ", invariant_errors=[dynamic_error], missing_fields=[]"

def test_invariant_exception_constructor_with_missing_fields():
    missing = ("field1", "field2")
    exception = InvariantException(missing_fields=missing)
    assert exception.invariant_errors == ()
    assert exception.missing_fields == ("field1", "field2")
    assert str(exception) == ", invariant_errors=[], missing_fields=[field1, field2]"

def test_invariant_exception_constructor_with_both_arguments():
    error_data = ("err1", "err2")
    missing = ("missing1",)
    exception = InvariantException(error_codes=error_data, missing_fields=missing)
    assert exception.invariant_errors == ("err1", "err2")
    assert exception.missing_fields == ("missing1",)
    assert str(exception) == ", invariant_errors=[err1, err2], missing_fields=[missing1]"

def test_invariant_exception_constructor_with_additional_args_and_kwargs():
    exception = InvariantException("extra_arg", another="extra_kwarg")
    assert exception.invariant_errors == ()
    assert exception.missing_fields == ()
    assert str(exception) == ", invariant_errors=[], missing_fields=[]"


# LLM-generated content at query #19
#--------------------------

def test_constructor_with_no_arguments():
    exception = InvariantException()
    assert exception.invariant_errors == ()
    assert exception.missing_fields == ()
    assert str(exception) == ", invariant_errors=[], missing_fields=[]"

def test_constructor_with_error_codes():
    error_data = ("error1", "error2")
    exception = InvariantException(error_codes=error_data)
    assert exception.invariant_errors == ("error1", "error2")
    assert exception.missing_fields == ()
    assert str(exception) == ", invariant_errors=[error1, error2], missing_fields=[]"

def test_constructor_with_callable_error_codes():
    def error_func():
        return "dynamic_error"
    exception = InvariantException(error_codes=(error_func,))
    assert exception.invariant_errors == ("dynamic_error",)
    assert exception.missing_fields == ()
    assert str(exception) == ", invariant_errors=[dynamic_error], missing_fields=[]"

def test_constructor_with_missing_fields():
    missing = ("field1", "field2")
    exception = InvariantException(missing_fields=missing)
    assert exception.invariant_errors == ()
    assert exception.missing_fields == ("field1", "field2")
    assert str(exception) == ", invariant_errors=[], missing_fields=[field1, field2]"

def test_constructor_with_both_arguments():
    error_data = ("err1", "err2")
    missing = ("missing1",)
    exception = InvariantException(error_codes=error_data, missing_fields=missing)
    assert exception.invariant_errors == ("err1", "err2")
    assert exception.missing_fields == ("missing1",)
    assert str(exception) == ", invariant_errors=[err1, err2], missing_fields=[missing1]"

def test_constructor_with_additional_args_and_kwargs():
    exception = InvariantException("extra_arg", another="extra_kwarg")
    assert exception.invariant_errors == ()
    assert exception.missing_fields == ()
    assert str(exception) == ", invariant_errors=[], missing_fields=[]"


# LLM-generated content at query #20
#--------------------------

def test_serialize_with_default_format():
    class TestSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda format, v: str(v)
    test_set = TestSet([1, 2, 3])
    result = test_set.serialize()
    expected = {'1', '2', '3'}
    assert result == expected

def test_serialize_with_custom_format():
    class TestSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda format, v: f"{v}:{format}"
    test_set = TestSet([1, 2])
    result = test_set.serialize('fmt')
    expected = {'1:fmt', '2:fmt'}
    assert result == expected

def test_serialize_empty_set():
    class TestSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda format, v: str(v)
    test_set = TestSet()
    result = test_set.serialize()
    expected = set()
    assert result == expected

def test_serialize_with_complex_serializer():
    class TestSet(CheckedPSet):
        __type__ = str
        __serializer__ = lambda format, v: v.upper() if format == 'upper' else v
    test_set = TestSet(['a', 'b'])
    result = test_set.serialize('upper')
    expected = {'A', 'B'}
    assert result == expected


# LLM-generated content at query #21
#--------------------------

def test_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives()
    assert isinstance(result, Positives)
    assert len(result) == 0


def test_constructor_with_list_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 2, 3])
    assert isinstance(result, Positives)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_constructor_with_duplicates_in_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 2, 2, 3])
    assert isinstance(result, Positives)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_constructor_with_pmap_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    pmap_initial = pmap({1: True, 2: True})
    result = Positives(pmap_initial)
    assert isinstance(result, Positives)
    assert len(result) == 2
    assert 1 in result
    assert 2 in result


def test_constructor_type_violation():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, 'a', 3])
        assert False
    except TypeError:
        pass


def test_constructor_invariant_violation():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -2, 3])
        assert False
    except InvariantException:
        pass


def test_constructor_with_no_type_or_invariant():
    class SimpleSet(CheckedPSet):
        pass
    result = SimpleSet([1, 2, 3])
    assert isinstance(result, SimpleSet)
    assert len(result) == 3


def test_constructor_with_iterable_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives((1, 2, 3))
    assert isinstance(result, Positives)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


# LLM-generated content at query #22
#--------------------------

def test_check_types_with_expected_types_and_matching_element():
    class MockType:
        pass
    class MockSourceClass:
        __name__ = "MockSourceClass"
    class MockExceptionType(Exception):
        pass
    it = [MockType()]
    expected_types = [MockType]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    result = _check_types(it, expected_types, source_class, exception_type)
    assert result is None

def test_check_types_with_expected_types_and_non_matching_element():
    class MockType:
        pass
    class MockOtherType:
        pass
    class MockSourceClass:
        __name__ = "MockSourceClass"
    class MockExceptionType(Exception):
        pass
    it = [MockOtherType()]
    expected_types = [MockType]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        assert False
    except MockExceptionType:
        assert True

def test_check_types_with_empty_expected_types():
    class MockSourceClass:
        __name__ = "MockSourceClass"
    class MockExceptionType(Exception):
        pass
    it = [object()]
    expected_types = []
    source_class = MockSourceClass
    exception_type = MockExceptionType
    result = _check_types(it, expected_types, source_class, exception_type)
    assert result is None

def test_check_types_with_multiple_expected_types_and_matching_element():
    class MockType1:
        pass
    class MockType2:
        pass
    class MockSourceClass:
        __name__ = "MockSourceClass"
    class MockExceptionType(Exception):
        pass
    it = [MockType1()]
    expected_types = [MockType1, MockType2]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    result = _check_types(it, expected_types, source_class, exception_type)
    assert result is None

def test_check_types_with_multiple_expected_types_and_non_matching_element():
    class MockType1:
        pass
    class MockType2:
        pass
    class MockOtherType:
        pass
    class MockSourceClass:
        __name__ = "MockSourceClass"
    class MockExceptionType(Exception):
        pass
    it = [MockOtherType()]
    expected_types = [MockType1, MockType2]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        assert False
    except MockExceptionType:
        assert True

def test_check_types_with_none_expected_types():
    class MockSourceClass:
        __name__ = "MockSourceClass"
    class MockExceptionType(Exception):
        pass
    it = [object()]
    expected_types = None
    source_class = MockSourceClass
    exception_type = MockExceptionType
    result = _check_types(it, expected_types, source_class, exception_type)
    assert result is None


# LLM-generated content at query #23
#--------------------------

def test_invariant_errors_returns_empty_list_when_all_invariants_valid():
    elem = "test_element"
    invariants = [lambda x: (True, "ok1"), lambda x: (True, "ok2")]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_returns_data_for_invalid_invariants():
    elem = 5
    invariants = [lambda x: (True, "valid"), lambda x: (False, "error1"), lambda x: (False, "error2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["error1", "error2"]

def test_invariant_errors_returns_empty_list_for_empty_invariants():
    elem = None
    invariants = []
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_passes_elem_to_each_invariant():
    captured = []
    def capturing_invariant(e):
        captured.append(e)
        return (True, None)
    elem = object()
    invariants = [capturing_invariant, capturing_invariant]
    _invariant_errors(elem, invariants)
    assert captured == [elem, elem]

def test_invariant_errors_ignores_data_from_valid_invariants():
    elem = 10
    invariants = [lambda x: (True, "ignore_me"), lambda x: (False, "include_me")]
    result = _invariant_errors(elem, invariants)
    assert result == ["include_me"]


# LLM-generated content at query #24
#--------------------------

```python
def test_size_is_not_undefined_checked_pmap_size():
    from pyrsistent._checked_types import CheckedPMap, _UNDEFINED_CHECKED_PMAP_SIZE
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    result = TestMap({1: 2, 3: 4}, size=10)
    assert result == {1: 2, 3: 4}


# LLM-generated content at query #25
#--------------------------

def test_merge_invariant_results_all_true():
    result = [(True, []), (True, []), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_one_false():
    result = [(True, []), (False, ["error1"]), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)

def test_merge_invariant_results_multiple_false():
    result = [(False, ["error1"]), (False, ["error2"]), (False, ["error3"])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2", "error3")

def test_merge_invariant_results_empty():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_mixed_data():
    result = [(True, []), (False, ["error_a"]), (False, ["error_b", "error_c"])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error_a", ["error_b", "error_c"])

def test_merge_invariant_results_only_false_single():
    result = [(False, ["only_error"])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("only_error",)


# LLM-generated content at query #26
#--------------------------

def test_serialize_returns_dict_from_serializer():
    class MockSerializer:
        def __call__(self, format, key, value):
            return (key, value, format)
    class TestMap(CheckedPMap):
        __serializer__ = MockSerializer()
        __key_type__ = int
        __value_type__ = str
    test_map = TestMap({1: 'a', 2: 'b'})
    result = test_map.serialize('test_format')
    expected = {1: (1, 'a', 'test_format'), 2: (2, 'b', 'test_format')}
    assert result == expected


# LLM-generated content at query #27
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {'a': 1.5}
    try:
        IntToFloatMap(initial)
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 'invalid'}
    try:
        IntToFloatMap(initial)
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    initial = {1: 2.5}
    try:
        IntToFloatMap(initial)
        assert False
    except InvariantException:
        assert True

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    initial = {1: 1.0, 2: 2.0}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_constructor_with_checked_key_type():
    class InnerChecked(CheckedType):
        pass
    class OuterMap(CheckedPMap):
        __key_type__ = InnerChecked
        __value_type__ = int
    inner = InnerChecked()
    initial = {inner: 5}
    result = OuterMap(initial)
    assert isinstance(result, OuterMap)
    assert result[inner] == 5

def test_constructor_with_checked_value_type():
    class InnerChecked(CheckedType):
        pass
    class OuterMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = InnerChecked
    inner = InnerChecked()
    initial = {1: inner}
    result = OuterMap(initial)
    assert isinstance(result, OuterMap)
    assert result[1] is inner

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v > 0, 'Value non-positive')]
    initial = {1: 1.5}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5

def test_constructor_with_multiple_invariants_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v > 0, 'Value non-positive')]
    initial = {-1: 1.5}
    try:
        IntToFloatMap(initial)
        assert False
    except InvariantException:
        assert True

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap(size=10)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_initial_and_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5}
    result = IntToFloatMap(initial, size=10)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5

def test_constructor_returns_same_instance_if_already_checked_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result is original

def test_constructor_with_none_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap(None)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


# LLM-generated content at query #28
#--------------------------

def test_serialize_returns_dict_from_serializer():
    class MockSerializer:
        def __call__(self, format, key, value):
            return (key, value, format)
    class TestMap(CheckedPMap):
        __serializer__ = MockSerializer()
    test_map = TestMap({1: 'a', 2: 'b'})
    result = test_map.serialize('test_format')
    expected = {1: (1, 'a', 'test_format'), 2: (2, 'b', 'test_format')}
    assert result == expected


# LLM-generated content at query #29
#--------------------------

def test_maybe_parse_user_type_with_preserved_type():
    result = maybe_parse_user_type(list)
    assert result == [list]

def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]

def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    result = maybe_parse_user_type((int, str))
    assert result == (int, str)

def test_maybe_parse_user_type_with_nested_iterable():
    result = maybe_parse_user_type([(int, str), float])
    assert result == (int, str, float)

def test_maybe_parse_user_type_with_empty_iterable():
    result = maybe_parse_user_type([])
    assert result == ()

def test_maybe_parse_user_type_with_mixed_iterable():
    result = maybe_parse_user_type([int, "str", (float,)])
    assert result == (int, "str", float)

def test_maybe_parse_user_type_raises_type_error_for_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False
    except TypeError:
        assert True

def test_maybe_parse_user_type_raises_type_error_for_invalid_iterable_element():
    try:
        maybe_parse_user_type([int, 123])
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #30
#--------------------------

def test_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives()
    assert result == set()


def test_constructor_with_list_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 2, 3])
    assert result == {1, 2, 3}


def test_constructor_with_set_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives({4, 5})
    assert result == {4, 5}


def test_constructor_with_pmap_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    pmap_initial = pset([6, 7])._map
    result = Positives(pmap_initial)
    assert result == {6, 7}


def test_constructor_type_check_failure():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives(['a'])
        assert False
    except TypeError:
        assert True


def test_constructor_invariant_check_failure():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([-1])
        assert False
    except InvariantException:
        assert True


def test_constructor_duplicate_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 1, 2, 2])
    assert result == {1, 2}


def test_constructor_with_no_type_or_invariant():
    class SimpleSet(CheckedPSet):
        pass
    result = SimpleSet([1, 'a', 3.14])
    assert result == {1, 'a', 3.14}


def test_constructor_repr_output():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    instance = Positives([1, 2])
    assert repr(instance) == "Positives([1, 2])"


def test_constructor_str_output():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    instance = Positives([3, 4])
    assert str(instance) == "Positives([3, 4])"


# LLM-generated content at query #31
#--------------------------

def test_wrap_invariant_single_bool_true():
    def invariant():
        return True, ()
    wrapped = wrap_invariant(invariant)
    result = wrapped()
    assert result == (True, ())

def test_wrap_invariant_single_bool_false():
    def invariant():
        return False, ("error",)
    wrapped = wrap_invariant(invariant)
    result = wrapped()
    assert result == (False, ("error",))

def test_wrap_invariant_multiple_results_all_true():
    def invariant():
        return [(True, ()), (True, ())]
    wrapped = wrap_invariant(invariant)
    result = wrapped()
    assert result == (True, ())

def test_wrap_invariant_multiple_results_one_false():
    def invariant():
        return [(True, ()), (False, ("error1",)), (True, ())]
    wrapped = wrap_invariant(invariant)
    result = wrapped()
    assert result == (False, ("error1",))

def test_wrap_invariant_multiple_results_multiple_false():
    def invariant():
        return [(False, ("error1",)), (True, ()), (False, ("error2",))]
    wrapped = wrap_invariant(invariant)
    result = wrapped()
    assert result == (False, ("error1", "error2",))

def test_wrap_invariant_with_args():
    def invariant(x, y):
        return x > y, ()
    wrapped = wrap_invariant(invariant)
    result = wrapped(5, 3)
    assert result == (True, ())

def test_wrap_invariant_with_kwargs():
    def invariant(a, b=0):
        return a == b, ()
    wrapped = wrap_invariant(invariant)
    result = wrapped(5, b=5)
    assert result == (True, ())

def test_wrap_invariant_empty_result_list():
    def invariant():
        return []
    wrapped = wrap_invariant(invariant)
    result = wrapped()
    assert result == (True, ())

def test_wrap_invariant_single_tuple_in_list():
    def invariant():
        return [(False, ("error",))]
    wrapped = wrap_invariant(invariant)
    result = wrapped()
    assert result == (False, ("error",))


# LLM-generated content at query #32
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_checkedpmap_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 1
    assert result[1] == 1.5

def test_constructor_type_check_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({"a": 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_type_check_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: "string"})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_invariant_check():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_invariant_check_passes():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({}, size=0)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_size_and_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 1
    assert result[1] == 1.5

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    expected_repr = "IntToFloatMap({1: 1.5})"
    assert repr(result) == expected_repr

def test_constructor_str():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    expected_str = "IntToFloatMap({1: 1.5})"
    assert str(result) == expected_str


# LLM-generated content at query #33
#--------------------------

def test_check_types_with_valid_types():
    expected_types = [int, str]
    source_class = list
    it = [1, "hello", 2, "world"]
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_type():
    expected_types = [int]
    source_class = list
    it = [1, "invalid", 3]
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError as e:
        assert e.args[0] == list
        assert e.args[1] == [int]
        assert e.args[2] == str
        assert e.args[3] == "invalid"

def test_check_types_with_no_expected_types():
    expected_types = []
    source_class = list
    it = [1, "any", 3.14, None]
    _check_types(it, expected_types, source_class)

def test_check_types_with_string_type_names():
    expected_types = ["builtins.int", "builtins.str"]
    source_class = list
    it = [1, "valid", 2]
    _check_types(it, expected_types, source_class)

def test_check_types_with_mixed_type_and_string():
    expected_types = [int, "builtins.str"]
    source_class = list
    it = [1, "mixed", 3]
    _check_types(it, expected_types, source_class)

def test_check_types_with_custom_exception_type():
    class CustomError(Exception):
        pass
    expected_types = [int]
    source_class = list
    it = [1, "error", 2]
    try:
        _check_types(it, expected_types, source_class, CustomError)
        assert False
    except CustomError as e:
        assert e.args[0] == list
        assert e.args[1] == [int]
        assert e.args[2] == str
        assert e.args[3] == "error"

def test_check_types_with_empty_iterable():
    expected_types = [int, str]
    source_class = list
    it = []
    _check_types(it, expected_types, source_class)

def test_check_types_with_single_invalid_element():
    expected_types = [float]
    source_class = tuple
    it = [3.14, "not_float"]
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError as e:
        assert e.args[0] == tuple
        assert e.args[1] == [float]
        assert e.args[2] == str
        assert e.args[3] == "not_float"

def test_check_types_with_all_invalid_elements():
    expected_types = [bool]
    source_class = set
    it = ["false", "true"]
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError as e:
        assert e.args[0] == set
        assert e.args[1] == [bool]
        assert e.args[2] == str
        assert e.args[3] == "false"


# LLM-generated content at query #34
#--------------------------

def test_predicate_at_line_2_evaluates_to_false():
    class MockClass:
        pass
    obj = MockClass()
    source_data = obj
    cls = MockClass
    result = isinstance(source_data, cls)
    assert result is False


# LLM-generated content at query #35
#--------------------------

def test_constructor_with_valid_key_value_types():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_invalid_key_type_raises_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_with_invalid_value_type_raises_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'invalid'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_with_invariant_violation_raises_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0})
    assert result == {1: 1.0}

def test_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({})
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k > 0, 'Key must be positive'), lambda k, v: (v > 0, 'Value must be positive')]
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}

def test_constructor_with_invariant_list_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k > 0, 'Key must be positive'), lambda k, v: (v > 0, 'Value must be positive')]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_checked_type_key():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key_instance = KeyType()
    value_instance = ValueType()
    result = CustomMap({key_instance: value_instance})
    assert result == {key_instance: value_instance}

def test_constructor_with_initial_as_same_class_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result == original
    assert result is not original

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert result == {1: 1.5}

def test_constructor_with_checked_type_value():
    class InnerType(CheckedType):
        pass
    class OuterMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = InnerType
    inner = InnerType()
    result = OuterMap({1: inner})
    assert result == {1: inner}

def test_constructor_with_mixed_valid_types():
    class MixedMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = (float, str)
    result = MixedMap({1: 1.5, 'a': 'test'})
    assert result == {1: 1.5, 'a': 'test'}

def test_constructor_with_invalid_mixed_type():
    class MixedMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = (float, str)
    try:
        MixedMap({1.5: 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_with_no_type_specification():
    class NoTypeMap(CheckedPMap):
        pass
    result = NoTypeMap({1: 'a', 2: 'b'})
    assert result == {1: 'a', 2: 'b'}

def test_constructor_with_inheritance():
    class BaseMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    class DerivedMap(BaseMap):
        pass
    result = DerivedMap({1: 1.5})
    assert result == {1: 1.5}
    assert isinstance(result, DerivedMap)

def test_constructor_with_create_method():
    class InnerChecked(CheckedType):
        @classmethod
        def create(cls, data):
            return cls()
    class OuterMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = InnerChecked
    result = OuterMap({1: {}})
    assert isinstance(list(result.values())[0], InnerChecked)

def test_constructor_with_checked_key_type_create():
    class KeyChecked(CheckedType):
        @classmethod
        def create(cls, data):
            return 42
    class ValueChecked(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyChecked
        __value_type__ = ValueChecked
    value_instance = ValueChecked()
    result = CustomMap({'any': value_instance})
    assert result == {42: value_instance}

def test_constructor_with_serialize_method():
    class SerializableMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __serializer__ = lambda format, k, v: (k, v)
    instance = SerializableMap({1: 1.5})
    serialized = instance.serialize()
    assert serialized == {1: 1.5}

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    instance = IntToFloatMap({1: 1.5})
    assert repr(instance) == "IntToFloatMap({1: 1.5})"

def test_constructor_str():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    instance = IntToFloatMap({1: 1.5})
    assert str(instance) == "IntToFloatMap({1: 1.5})"


