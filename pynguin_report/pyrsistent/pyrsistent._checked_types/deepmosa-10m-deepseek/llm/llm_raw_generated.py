####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    expected_repr = "IntToFloatMap({1: 1.5, 2: 2.25})"
    assert repr(result) == expected_repr
    assert str(result) == expected_repr


# LLM-generated content at query #2
#--------------------------

def test__store_types_with_single_dict():
    dct = {'source': int}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (int,)

def test__store_types_with_multiple_dicts():
    dct = {'source': str}
    base1 = type('Base1', (), {'source': float})
    base2 = type('Base2', (), {})
    _store_types(dct, [base1, base2], 'destination', 'source')
    assert dct['destination'] == (str, float)

def test__store_types_with_iterable_source():
    dct = {'source': [int, str]}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (int, str)

def test__store_types_with_string_source():
    dct = {'source': 'CustomType'}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == ('CustomType',)

def test__store_types_with_preserved_iterable_type():
    dct = {'source': list}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (list,)

def test__store_types_with_no_source_in_dict_or_bases():
    dct = {}
    base = type('Base', (), {})
    _store_types(dct, [base], 'destination', 'source')
    assert 'destination' not in dct

def test__store_types_with_source_in_base_only():
    dct = {}
    base = type('Base', (), {'source': bool})
    _store_types(dct, [base], 'destination', 'source')
    assert dct['destination'] == (bool,)

def test__store_types_with_nested_iterable_source():
    dct = {'source': [[int, str], float]}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (int, str, float)

def test__store_types_raises_on_invalid_type():
    dct = {'source': 123}
    try:
        _store_types(dct, [], 'destination', 'source')
        assert False
    except TypeError:
        assert True

def test__store_types_overwrites_destination_key():
    dct = {'source': int, 'destination': 'old_value'}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (int,)


# LLM-generated content at query #3
#--------------------------

def test_invariant_errors_no_errors():
    elem = 5
    invariants = [lambda x: (True, "ok1"), lambda x: (True, "ok2")]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_all_errors():
    elem = "test"
    invariants = [lambda x: (False, "err1"), lambda x: (False, "err2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["err1", "err2"]

def test_invariant_errors_mixed_results():
    elem = [1, 2]
    invariants = [lambda x: (True, "ok"), lambda x: (False, "err"), lambda x: (True, "fine")]
    result = _invariant_errors(elem, invariants)
    assert result == ["err"]

def test_invariant_errors_empty_invariants():
    elem = None
    invariants = []
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_elem_passed_correctly():
    captured = []
    def capture_inv(e):
        captured.append(e)
        return (True, "seen")
    invariants = [capture_inv]
    elem = {"key": "value"}
    _invariant_errors(elem, invariants)
    assert captured == [elem]


# LLM-generated content at query #4
#--------------------------

def test_store_invariants_adds_destination_with_wrapped_invariants():
    base_invariant = lambda self: (True, ())
    child_invariant = lambda self: (False, ("error",))
    bases = (type('Base', (), {'invariant': base_invariant}),)
    dct = {'invariant': child_invariant}
    store_invariants(dct, bases, 'invariants', 'invariant')
    result = dct['invariants']
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0](None) == (True, ())
    assert result[1](None) == (False, ('error',))

def test_store_invariants_raises_type_error_for_non_callable():
    bases = ()
    dct = {'invariant': 'not callable'}
    try:
        store_invariants(dct, bases, 'invariants', 'invariant')
        assert False
    except TypeError:
        pass

def test_store_invariants_inherits_from_multiple_bases():
    inv1 = lambda self: (True, ())
    inv2 = lambda self: (True, ())
    base1 = type('Base1', (), {'invariant': inv1})
    base2 = type('Base2', (), {'invariant': inv2})
    bases = (base1, base2)
    dct = {}
    store_invariants(dct, bases, 'invariants', 'invariant')
    result = dct['invariants']
    assert len(result) == 2
    assert result[0](None) == (True, ())
    assert result[1](None) == (True, ())

def test_store_invariants_handles_duplicate_inheritance():
    shared_invariant = lambda self: (True, ())
    base = type('Base', (), {'invariant': shared_invariant})
    bases = (base, base)
    dct = {}
    store_invariants(dct, bases, 'invariants', 'invariant')
    result = dct['invariants']
    assert len(result) == 1
    assert result[0](None) == (True, ())

def test_store_invariants_with_no_invariants_found():
    bases = ()
    dct = {}
    store_invariants(dct, bases, 'invariants', 'invariant')
    result = dct['invariants']
    assert result == ()

def test_store_invariants_wraps_invariant_returning_list():
    def list_invariant(self):
        return [(True, ()), (False, ('err1',)), (False, ('err2',))]
    bases = ()
    dct = {'invariant': list_invariant}
    store_invariants(dct, bases, 'invariants', 'invariant')
    wrapped = dct['invariants'][0]
    verdict, data = wrapped(None)
    assert verdict == False
    assert data == ('err1', 'err2')

def test_store_invariants_wraps_invariant_returning_bool_tuple():
    def bool_invariant(self):
        return False, ('err',)
    bases = ()
    dct = {'invariant': bool_invariant}
    store_invariants(dct, bases, 'invariants', 'invariant')
    wrapped = dct['invariants'][0]
    verdict, data = wrapped(None)
    assert verdict == False
    assert data == ('err',)


# LLM-generated content at query #5
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
    pvec = python_pvector([4, 5, 6])
    result = TestVector(pvec)
    assert isinstance(result, TestVector)
    assert list(result) == [4, 5, 6]


def test_constructor_type_check_success():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector([7, 8, 9])
    assert all(isinstance(x, int) for x in result)


def test_constructor_type_check_failure():
    class TestVector(CheckedPVector):
        __type__ = int
    try:
        TestVector([10, 'invalid', 12])
        assert False
    except TypeError:
        assert True


def test_constructor_invariant_check_success():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1.5, 2, 3.0])
    assert list(result) == [1.5, 2, 3.0]


def test_constructor_invariant_check_failure():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -2, 3])
        assert False
    except InvariantException:
        assert True


def test_constructor_with_multiple_types():
    class TestVector(CheckedPVector):
        __type__ = (int, str)
    result = TestVector([1, 'a', 2, 'b'])
    assert list(result) == [1, 'a', 2, 'b']


def test_constructor_with_no_type_or_invariant():
    class TestVector(CheckedPVector):
        pass
    result = TestVector([1, 'a', None])
    assert list(result) == [1, 'a', None]


def test_constructor_returns_same_type():
    class TestVector(CheckedPVector):
        __type__ = int
    instance = TestVector([1, 2, 3])
    result = TestVector(instance)
    assert result is instance


# LLM-generated content at query #6
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

def test_constructor_with_correct_types():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.0, 3: 3.14})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[3] == 3.14

def test_constructor_with_wrong_key_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.0})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_with_wrong_value_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'string'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_with_invariant_violation_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.0})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_constructor_with_checked_key_type():
    class MyKey(CheckedType):
        pass
    class MyMap(CheckedPMap):
        __key_type__ = MyKey
        __value_type__ = int
    key_instance = MyKey()
    result = MyMap({key_instance: 5})
    assert isinstance(result, MyMap)
    assert result[key_instance] == 5

def test_constructor_with_checked_value_type():
    class MyValue(CheckedType):
        pass
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = MyValue
    value_instance = MyValue()
    result = MyMap({1: value_instance})
    assert isinstance(result, MyMap)
    assert result[1] is value_instance

def test_constructor_with_checked_types_and_create():
    class MyKey(CheckedType):
        @classmethod
        def create(cls, data):
            return cls()
    class MyValue(CheckedType):
        @classmethod
        def create(cls, data):
            return cls()
    class MyMap(CheckedPMap):
        __key_type__ = MyKey
        __value_type__ = MyValue
    source = {'raw_key': 'raw_value'}
    result = MyMap.create(source)
    assert isinstance(result, MyMap)
    assert all(isinstance(k, MyKey) for k in result.keys())
    assert all(isinstance(v, MyValue) for v in result.values())

def test_constructor_with_initial_as_same_checkedpmap_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result is original

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    repr_str = repr(result)
    assert repr_str.startswith('IntToFloatMap')
    assert '1: 1.5' in repr_str

def test_constructor_str():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    str_str = str(result)
    assert str_str.startswith('IntToFloatMap')
    assert '1: 1.5' in str_str

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
    assert result[1] == 1.5

def test_constructor_serialize():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __serializer__ = lambda format, k, v: (k, v)
    result = IntToFloatMap({1: 1.5})
    serialized = result.serialize()
    assert serialized == {1: 1.5}

def test_constructor_pickle_support():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    import pickle
    original = IntToFloatMap({1: 1.5})
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)
    assert isinstance(restored, IntToFloatMap)
    assert restored[1] == 1.5


# LLM-generated content at query #7
#--------------------------

def test_invariant_errors_returns_empty_list_when_all_invariants_valid():
    elem = {"key": "value"}
    invariants = [lambda x: (True, None), lambda x: (True, "ok")]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_returns_data_list_for_invalid_invariants():
    elem = 5
    invariants = [lambda x: (True, "valid"), lambda x: (False, "error1"), lambda x: (False, "error2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["error1", "error2"]

def test_invariant_errors_handles_mixed_valid_and_invalid_invariants():
    elem = "test"
    invariants = [lambda x: (False, "err1"), lambda x: (True, "ignored"), lambda x: (False, "err2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["err1", "err2"]

def test_invariant_errors_returns_empty_list_for_empty_invariants():
    elem = object()
    invariants = []
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_passes_elem_to_each_invariant():
    captured = []
    def capturing_invariant(e):
        captured.append(e)
        return (True, None)
    elem = [1, 2, 3]
    invariants = [capturing_invariant, capturing_invariant]
    _invariant_errors(elem, invariants)
    assert captured == [elem, elem]


# LLM-generated content at query #8
#--------------------------

def test_maybe_parse_user_type_preserved_type():
    result = maybe_parse_user_type(list)
    assert result == [list]

def test_maybe_parse_user_type_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]

def test_maybe_parse_user_type_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]

def test_maybe_parse_user_type_iterable_of_types():
    result = maybe_parse_user_type((int, str))
    assert result == (int, str)

def test_maybe_parse_user_type_nested_iterable():
    result = maybe_parse_user_type([(int, str), float])
    assert result == (int, str, float)

def test_maybe_parse_user_type_empty_iterable():
    result = maybe_parse_user_type([])
    assert result == ()

def test_maybe_parse_user_type_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False
    except TypeError:
        assert True

def test_maybe_parse_user_type_mixed_valid_iterable():
    result = maybe_parse_user_type([int, "str", (float,)])
    assert result == (int, "str", float)


# LLM-generated content at query #9
#--------------------------

def test_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives()
    assert result == Positives([])


def test_constructor_with_list_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 2, 3])
    assert result == Positives({1, 2, 3})


def test_constructor_with_set_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives({1.5, 2.5})
    assert result == Positives({1.5, 2.5})


def test_constructor_with_pmap_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    pmap_initial = pmap({1: True, 2: True})
    result = Positives(pmap_initial)
    assert result == Positives({1, 2})


def test_constructor_type_violation():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, 'a'])
    except TypeError:
        pass
    else:
        assert False


def test_constructor_invariant_violation():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -1])
    except InvariantException:
        pass
    else:
        assert False


def test_constructor_duplicate_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 1, 2])
    assert len(result) == 2
    assert 1 in result
    assert 2 in result


# LLM-generated content at query #10
#--------------------------

def test_check_types_with_valid_types():
    expected_types = [int, str]
    source_class = list
    data = [1, "hello", 2, "world"]
    _check_types(data, expected_types, source_class)

def test_check_types_with_invalid_type():
    expected_types = [int]
    source_class = list
    data = [1, "invalid", 3]
    try:
        _check_types(data, expected_types, source_class)
        assert False
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == (int,)
        assert e.actual_type == str
        assert e.value == "invalid"

def test_check_types_with_no_expected_types():
    expected_types = None
    source_class = list
    data = [1, "hello", 3.14, [1, 2]]
    _check_types(data, expected_types, source_class)

def test_check_types_with_empty_iterable():
    expected_types = [int, str]
    source_class = list
    data = []
    _check_types(data, expected_types, source_class)

def test_check_types_with_string_type_name():
    expected_types = ["builtins.int", "builtins.str"]
    source_class = list
    data = [1, "hello"]
    _check_types(data, expected_types, source_class)

def test_check_types_with_custom_exception_type():
    class CustomError(Exception):
        pass
    expected_types = [int]
    source_class = list
    data = [1, "invalid"]
    try:
        _check_types(data, expected_types, source_class, CustomError)
        assert False
    except CustomError as e:
        assert e.source_class == list
        assert e.expected_types == (int,)
        assert e.actual_type == str
        assert e.value == "invalid"

def test_check_types_with_mixed_type_names_and_types():
    expected_types = [int, "builtins.str"]
    source_class = list
    data = [1, "hello", 2]
    _check_types(data, expected_types, source_class)

def test_check_types_with_invalid_type_name():
    expected_types = ["nonexistent.module.Class"]
    source_class = list
    data = [1]
    try:
        _check_types(data, expected_types, source_class)
        assert False
    except (ImportError, AttributeError):
        assert True


# LLM-generated content at query #11
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

def test_constructor_invariant_check_success():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k > 0, 'Key must be positive'), lambda k, v: (v > 0, 'Value must be positive')]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_multiple_invariants_success():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k > 0, 'Key must be positive'), lambda k, v: (v > 0, 'Value must be positive')]
    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5

def test_constructor_with_checked_key_type():
    class KeyMap(CheckedPMap):
        __key_type__ = CheckedType
        __value_type__ = int
    class MyKey(CheckedType):
        pass
    result = KeyMap({MyKey(): 1})
    assert isinstance(result, KeyMap)
    assert len(result) == 1

def test_constructor_with_checked_value_type():
    class ValueMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = CheckedType
    class MyValue(CheckedType):
        pass
    result = ValueMap({1: MyValue()})
    assert isinstance(result, ValueMap)
    assert len(result) == 1

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


# LLM-generated content at query #12
#--------------------------

def test_invariant_errors_returns_empty_list_when_all_invariants_valid():
    elem = "test_element"
    invariants = [lambda x: (True, "ok1"), lambda x: (True, "ok2")]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_returns_data_for_invalid_invariants():
    elem = 5
    invariants = [lambda x: (True, "ok"), lambda x: (False, "error1"), lambda x: (False, "error2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["error1", "error2"]

def test_invariant_errors_handles_empty_invariants_list():
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


# LLM-generated content at query #13
#--------------------------

def test_invariant_errors_returns_empty_list_when_all_invariants_valid():
    elem = {"name": "test"}
    invariants = [lambda x: (True, None), lambda x: (True, None)]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_returns_data_list_when_some_invariants_invalid():
    elem = {"name": "test"}
    invariants = [lambda x: (True, None), lambda x: (False, "error1"), lambda x: (False, "error2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["error1", "error2"]

def test_invariant_errors_returns_data_list_when_all_invariants_invalid():
    elem = {"name": "test"}
    invariants = [lambda x: (False, "error1"), lambda x: (False, "error2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["error1", "error2"]

def test_invariant_errors_passes_elem_to_each_invariant():
    captured = []
    def capturing_invariant(e):
        captured.append(e)
        return (True, None)
    elem = {"id": 123}
    invariants = [capturing_invariant, capturing_invariant]
    _invariant_errors(elem, invariants)
    assert captured == [elem, elem]

def test_invariant_errors_handles_empty_invariants_list():
    elem = {"name": "test"}
    invariants = []
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_includes_only_data_from_invalid_invariants():
    elem = {"value": 5}
    invariants = [lambda x: (True, "ignored"), lambda x: (False, "included"), lambda x: (True, "ignored2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["included"]


# LLM-generated content at query #14
#--------------------------

def test_constructor_with_valid_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_empty_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({})
    assert result == {}
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

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    result = IntToFloatMap({1: 2.0})
    assert result == {1: 2.0}

def test_constructor_with_multiple_invariants_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    try:
        IntToFloatMap({-1: 2.0})
        assert False
    except InvariantException as e:
        assert 'Key negative' in str(e.error_codes)

def test_constructor_with_checked_key_type():
    class KeyMap(CheckedPMap):
        __key_type__ = CheckedType
        __value_type__ = int
    class MyCheckedType(CheckedType):
        pass
    my_instance = MyCheckedType()
    result = KeyMap({my_instance: 1})
    assert result == {my_instance: 1}

def test_constructor_with_checked_value_type():
    class ValueMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = CheckedType
    class MyCheckedType(CheckedType):
        pass
    my_instance = MyCheckedType()
    result = ValueMap({1: my_instance})
    assert result == {1: my_instance}

def test_constructor_with_initial_as_same_class_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result is original

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert result == {1: 1.5}

def test_constructor_with_undefined_size_constant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=_UNDEFINED_CHECKED_PMAP_SIZE)
    assert result == {1: 1.5}

def test_constructor_repr_output():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == 'IntToFloatMap({1: 1.5})'

def test_constructor_str_output():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert str(result) == 'IntToFloatMap({1: 1.5})'


# LLM-generated content at query #15
#--------------------------

def test_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives()
    assert result == Positives([])
    assert len(result) == 0


def test_constructor_with_list_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 2, 3])
    assert result == Positives({1, 2, 3})
    assert len(result) == 3


def test_constructor_with_set_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives({4, 5})
    assert result == Positives({4, 5})
    assert len(result) == 2


def test_constructor_with_pmap_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    pmap_initial = pmap({6: True, 7: True})
    result = Positives(pmap_initial)
    assert result == Positives({6, 7})
    assert len(result) == 2


def test_constructor_type_check_failure():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, 'a'])
        assert False
    except TypeError:
        assert True


def test_constructor_invariant_check_failure():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -1])
        assert False
    except InvariantException:
        assert True


def test_constructor_duplicate_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 1, 2, 2])
    assert result == Positives({1, 2})
    assert len(result) == 2


def test_constructor_no_type_or_invariant():
    class SimpleSet(CheckedPSet):
        pass
    result = SimpleSet([1, 2, 3])
    assert result == SimpleSet({1, 2, 3})
    assert len(result) == 3


# LLM-generated content at query #16
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
    pv = python_pvector([1, 2, 3])
    result = TestVector(pv)
    assert isinstance(result, TestVector)
    assert list(result) == [1, 2, 3]


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


def test_constructor_with_tuple_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector((1, 2, 3))
    assert isinstance(result, TestVector)
    assert list(result) == [1, 2, 3]


def test_constructor_with_generator_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector(x for x in range(3))
    assert isinstance(result, TestVector)
    assert list(result) == [0, 1, 2]


def test_constructor_repr_output():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector([1, 2, 3])
    assert repr(result) == "TestVector([1, 2, 3])"


# LLM-generated content at query #17
#--------------------------

def test__restore_pickle():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields):
            return f"created with {data} and {_factory_fields}"
    result = MockClass._restore_pickle("test_data")
    assert result == "created with test_data and set()"


# LLM-generated content at query #18
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
    except CheckedTypeError:
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
    result = IntToFloatMap({1: 1.0})
    assert result == {1: 1.0}

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({})
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
    key_instance = KeyType()
    value_instance = ValueType()
    result = CustomMap({key_instance: value_instance})
    assert result == {key_instance: value_instance}

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
    result = CustomMap.create({'raw_key': 'raw_value'})
    assert all(isinstance(k, KeyType) for k in result.keys())
    assert all(isinstance(v, ValueType) for v in result.values())

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert result == {1: 1.5}

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


# LLM-generated content at query #19
#--------------------------

def test_store_invariants_adds_destination_tuple():
    class Base1:
        def inv1(self):
            return True, ()
    class Base2:
        def inv2(self):
            return True, ()
    dct = {}
    store_invariants(dct, (Base1, Base2), 'my_invariants', 'inv1')
    result = dct['my_invariants']
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert callable(result[0])
    dct2 = {}
    store_invariants(dct2, (Base1, Base2), 'my_invariants', 'inv2')
    result2 = dct2['my_invariants']
    assert isinstance(result2, tuple)
    assert len(result2) == 1
    assert callable(result2[0])

def test_store_invariants_inherits_from_all_bases():
    class BaseA:
        def invariant(self):
            return True, ()
    class BaseB(BaseA):
        pass
    class BaseC:
        def invariant(self):
            return True, ()
    dct = {}
    store_invariants(dct, (BaseB, BaseC), 'invariants', 'invariant')
    result = dct['invariants']
    assert len(result) == 2
    assert all(callable(inv) for inv in result)

def test_store_invariants_includes_local_dict():
    class Base:
        def invariant(self):
            return True, ()
    def local_invariant(self):
        return True, ()
    dct = {'invariant': local_invariant}
    store_invariants(dct, (Base,), 'invariants', 'invariant')
    result = dct['invariants']
    assert len(result) == 2
    assert all(callable(inv) for inv in result)

def test_store_invariants_raises_typeerror_for_noncallable():
    class Base:
        invariant = "not a function"
    dct = {}
    try:
        store_invariants(dct, (Base,), 'invariants', 'invariant')
        assert False
    except TypeError:
        pass

def test_store_invariants_raises_typeerror_for_noncallable_in_local_dict():
    dct = {'invariant': 42}
    try:
        store_invariants(dct, (), 'invariants', 'invariant')
        assert False
    except TypeError:
        pass

def test_store_invariants_handles_missing_source_name():
    class Base:
        pass
    dct = {}
    store_invariants(dct, (Base,), 'invariants', 'nonexistent')
    result = dct['invariants']
    assert isinstance(result, tuple)
    assert len(result) == 0

def test_store_invariants_wraps_invariants():
    def invariant_returns_list(self):
        return [(False, "error1"), (True, ()), (False, "error2")]
    dct = {'invariant': invariant_returns_list}
    store_invariants(dct, (), 'invariants', 'invariant')
    wrapped_inv = dct['invariants'][0]
    result = wrapped_inv(None)
    assert result == (False, ("error1", "error2"))

def test_store_invariants_does_not_duplicate_inherited():
    class Base:
        def invariant(self):
            return True, ()
    class Derived(Base):
        pass
    dct = {}
    store_invariants(dct, (Derived,), 'invariants', 'invariant')
    result = dct['invariants']
    assert len(result) == 1
    assert callable(result[0])

def test_store_invariants_with_multiple_inheritance_diamond():
    class Root:
        def invariant(self):
            return True, ()
    class A(Root):
        pass
    class B(Root):
        pass
    class C(A, B):
        pass
    dct = {}
    store_invariants(dct, (C,), 'invariants', 'invariant')
    result = dct['invariants']
    assert len(result) == 1
    assert callable(result[0])


# LLM-generated content at query #20
#--------------------------

def test_check_types_with_no_expected_types():
    it = [1, "a", 3.14]
    result = _check_types(it, [], list, ValueError)
    assert result is None

def test_check_types_with_matching_types():
    it = [1, 2, 3]
    result = _check_types(it, [int], list, ValueError)
    assert result is None

def test_check_types_with_matching_multiple_types():
    it = [1, "a", 2]
    result = _check_types(it, [int, str], list, ValueError)
    assert result is None

def test_check_types_raises_exception_on_mismatch():
    it = [1, "a", 3.14]
    try:
        _check_types(it, [int], list, ValueError)
        assert False
    except ValueError as e:
        assert "Type list can only be used with (int,), not str" in str(e)

def test_check_types_uses_custom_exception_type():
    it = [1, "a"]
    try:
        _check_types(it, [int], list, TypeError)
        assert False
    except TypeError:
        assert True

def test_check_types_with_none_in_iterable():
    it = [None, 1]
    result = _check_types(it, [type(None), int], list, ValueError)
    assert result is None


# LLM-generated content at query #21
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
    setattr(__import__('__main__'), 'MockSubType', MockSubType)
    source_data = [1, 2, 3]
    result = _checked_type_create(CheckedType, source_data)
    assert isinstance(result, CheckedType)
    assert all(isinstance(item, MockSubType) for item in result.data)

def test__checked_type_create_with_matching_type_in_list():
    class MockClass:
        _checked_types = [int, str]
    source_data = [1, "hello"]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert result == source_data

def test__checked_type_create_without_checked_types():
    class MockClass:
        pass
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert result == source_data

def test__checked_type_create_ignore_extra():
    class CheckedType:
        _checked_types = ["__main__.MockSubType"]
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
    class MockSubType(CheckedType):
        def __init__(self, data):
            self.data = data
    __import__('__main__', fromlist=['MockSubType'])
    setattr(__import__('__main__'), 'MockSubType', MockSubType)
    source_data = [1, 2, 3]
    result = _checked_type_create(CheckedType, source_data, ignore_extra=True)
    assert isinstance(result, CheckedType)
    assert all(isinstance(item, MockSubType) for item in result.data)


# LLM-generated content at query #22
#--------------------------

def test_check_types_with_valid_types():
    class TestClass:
        pass
    it = [1, 2, 3]
    expected_types = [int]
    result = _check_types(it, expected_types, TestClass)
    assert result is None

def test_check_types_with_mixed_valid_types():
    class TestClass:
        pass
    it = [1, "hello", 3.14]
    expected_types = [int, str, float]
    result = _check_types(it, expected_types, TestClass)
    assert result is None

def test_check_types_with_empty_expected_types():
    class TestClass:
        pass
    it = [1, "hello", object()]
    expected_types = []
    result = _check_types(it, expected_types, TestClass)
    assert result is None

def test_check_types_with_no_elements():
    class TestClass:
        pass
    it = []
    expected_types = [int]
    result = _check_types(it, expected_types, TestClass)
    assert result is None

def test_check_types_with_custom_exception_type():
    class CustomError(Exception):
        pass
    class TestClass:
        pass
    it = [1, "invalid", 3]
    expected_types = [int]
    try:
        _check_types(it, expected_types, TestClass, CustomError)
        assert False
    except CustomError:
        assert True


# LLM-generated content at query #23
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
    assert len(result) == 1
    assert result[1] == 1.5

def test_constructor_creates_new_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    map1 = IntToFloatMap({1: 1.5})
    map2 = IntToFloatMap(map1)
    assert map1 is not map2
    assert map1 == map2

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k > 0, 'Key must be positive'), lambda k, v: (v > 0, 'Value must be positive')]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException:
        assert True
    try:
        IntToFloatMap({1: -1.5})
        assert False
    except InvariantException:
        assert True
    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5


# LLM-generated content at query #24
#--------------------------

def test_wrap_invariant_single_bool_true():
    def invariant_true():
        return True, "data"
    wrapped = wrap_invariant(invariant_true)
    result = wrapped()
    assert result == (True, "data")

def test_wrap_invariant_single_bool_false():
    def invariant_false():
        return False, "error"
    wrapped = wrap_invariant(invariant_false)
    result = wrapped()
    assert result == (False, "error")

def test_wrap_invariant_multiple_results_all_true():
    def invariant_all_true():
        return [(True, "data1"), (True, "data2")]
    wrapped = wrap_invariant(invariant_all_true)
    result = wrapped()
    assert result == (True, ())

def test_wrap_invariant_multiple_results_one_false():
    def invariant_one_false():
        return [(True, "data1"), (False, "error1"), (True, "data2")]
    wrapped = wrap_invariant(invariant_one_false)
    result = wrapped()
    assert result == (False, ("error1",))

def test_wrap_invariant_multiple_results_all_false():
    def invariant_all_false():
        return [(False, "error1"), (False, "error2")]
    wrapped = wrap_invariant(invariant_all_false)
    result = wrapped()
    assert result == (False, ("error1", "error2"))

def test_wrap_invariant_with_args():
    def invariant_with_args(x, y):
        return x > y, f"{x} > {y}"
    wrapped = wrap_invariant(invariant_with_args)
    result = wrapped(5, 3)
    assert result == (True, "5 > 3")

def test_wrap_invariant_with_kwargs():
    def invariant_with_kwargs(a, b=0):
        return a == b, f"{a} == {b}"
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(0, b=0)
    assert result == (True, "0 == 0")

def test_wrap_invariant_multiple_results_with_args():
    def invariant_multi_with_args(x):
        return [(x > 0, "positive"), (x % 2 == 0, "even")]
    wrapped = wrap_invariant(invariant_multi_with_args)
    result = wrapped(4)
    assert result == (True, ())

def test_wrap_invariant_empty_result_list():
    def invariant_empty():
        return []
    wrapped = wrap_invariant(invariant_empty)
    result = wrapped()
    assert result == (True, ())


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_store_invariants_adds_destination_name():
    class Base:
        def base_invariant(self):
            return True, ()
    dct = {}
    store_invariants(dct, (Base,), 'invariants', 'invariant')
    assert 'invariants' in dct
    assert isinstance(dct['invariants'], tuple)

def test_store_invariants_includes_inherited_invariants():
    class Base:
        def base_invariant(self):
            return True, ()
    class Child(Base):
        def child_invariant(self):
            return True, ()
    dct = {}
    store_invariants(dct, (Child,), 'invariants', 'invariant')
    assert len(dct['invariants']) == 2

def test_store_invariants_wraps_invariants():
    class TestClass:
        def test_invariant(self):
            return True, ()
    dct = {}
    store_invariants(dct, (TestClass,), 'invariants', 'invariant')
    wrapped_invariant = dct['invariants'][0]
    result = wrapped_invariant()
    assert result == (True, ())

def test_store_invariants_merges_multiple_results():
    def merging_invariant():
        return [(True, ()), (False, ('error',))]
    dct = {'invariant': merging_invariant}
    store_invariants(dct, (), 'invariants', 'invariant')
    wrapped_invariant = dct['invariants'][0]
    result = wrapped_invariant()
    assert result == (False, ('error',))

def test_store_invariants_raises_type_error_for_non_callable():
    dct = {'invariant': 'not a function'}
    try:
        store_invariants(dct, (), 'invariants', 'invariant')
        assert False
    except TypeError:
        assert True

def test_store_invariants_handles_empty_invariants():
    dct = {}
    store_invariants(dct, (), 'invariants', 'invariant')
    assert dct['invariants'] == ()

def test_store_invariants_avoids_duplicate_inheritance():
    class Base:
        def base_invariant(self):
            return True, ()
    class Child1(Base):
        pass
    class Child2(Base):
        pass
    class GrandChild(Child1, Child2):
        pass
    dct = {}
    store_invariants(dct, (GrandChild,), 'invariants', 'invariant')
    assert len(dct['invariants']) == 1

def test_store_invariants_uses_correct_source_and_destination_names():
    class TestClass:
        def my_invariant(self):
            return True, ()
    dct = {}
    store_invariants(dct, (TestClass,), 'dest', 'my_invariant')
    assert 'dest' in dct
    assert len(dct['dest']) == 1

def test_store_invariants_handles_multiple_inheritance():
    class Base1:
        def invariant1(self):
            return True, ()
    class Base2:
        def invariant2(self):
            return True, ()
    class Child(Base1, Base2):
        pass
    dct = {}
    store_invariants(dct, (Child,), 'invariants', 'invariant')
    assert len(dct['invariants']) == 2

def test_store_invariants_works_with_instance_methods():
    class TestClass:
        def invariant(self):
            return True, ()
    instance = TestClass()
    dct = {}
    store_invariants(dct, (TestClass,), 'invariants', 'invariant')
    wrapped_invariant = dct['invariants'][0]
    result = wrapped_invariant(instance)
    assert result == (True, ())


# LLM-generated content at query #2
#--------------------------

def test_store_types_single_dict():
    dct = {'source': int}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (int,)

def test_store_types_multiple_dicts():
    dct = {'source': str}
    base1 = type('Base1', (), {'source': float})
    base2 = type('Base2', (), {})
    _store_types(dct, [base1, base2], 'destination', 'source')
    assert dct['destination'] == (str, float)

def test_store_types_no_source():
    dct = {}
    base = type('Base', (), {'other': 5})
    _store_types(dct, [base], 'destination', 'source')
    assert 'destination' not in dct

def test_store_types_string_type():
    dct = {'source': 'MyClass'}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == ('MyClass',)

def test_store_types_iterable_source():
    dct = {'source': [int, str]}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (int, str)

def test_store_types_preserved_iterable():
    dct = {'source': tuple}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (tuple,)

def test_store_types_mixed_sources():
    dct = {'source': [list, 'Custom']}
    base = type('Base', (), {'source': dict})
    _store_types(dct, [base], 'destination', 'source')
    assert dct['destination'] == (list, 'Custom', dict)

def test_store_types_overwrites_destination():
    dct = {'source': int, 'destination': 'old'}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (int,)

def test_store_types_empty_iterable():
    dct = {'source': []}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == ()

def test_store_types_nested_iterable():
    dct = {'source': [[int, str], float]}
    _store_types(dct, [], 'destination', 'source')
    assert dct['destination'] == (int, str, float)


# LLM-generated content at query #3
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
    except CheckedValueTypeError:
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

def test_constructor_with_checked_type_key_and_value():
    class KeyType(CheckedType):
        __type__ = int
    class ValueType(CheckedType):
        __type__ = str
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = CustomMap({1: 'a', 2: 'b'})
    assert result == {1: 'a', 2: 'b'}
    assert isinstance(result, CustomMap)

def test_constructor_with_checked_type_key_and_value_using_create():
    class KeyType(CheckedType):
        __type__ = int
    class ValueType(CheckedType):
        __type__ = str
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    source = {'1': 'a', '2': 'b'}
    result = CustomMap.create(source)
    assert result == {1: 'a', 2: 'b'}
    assert isinstance(result, CustomMap)

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [
            lambda k, v: (k > 0, 'Key must be positive'),
            lambda k, v: (v > 0, 'Value must be positive')
        ]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException:
        assert True
    try:
        IntToFloatMap({1: -1.5})
        assert False
    except InvariantException:
        assert True
    result = IntToFloatMap({1: 1.5})
    assert result == {1: 1.5}

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert result == {1: 1.5, 2: 2.25}
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
        __serializer__ = lambda format, k, v: (k, v)
    result = IntToFloatMap({1: 1.5})
    serialized = result.serialize()
    assert serialized == {1: 1.5}

def test_constructor_pickling():
    import pickle
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)
    assert original == restored
    assert isinstance(restored, IntToFloatMap)


# LLM-generated content at query #4
#--------------------------

def test_invariant_errors_no_errors():
    elem = 5
    invariants = [lambda x: (True, "ok1"), lambda x: (True, "ok2")]
    result = _invariant_errors(elem, invariants)
    expected = []
    assert result == expected

def test_invariant_errors_some_errors():
    elem = "test"
    invariants = [lambda x: (True, "ok"), lambda x: (False, "err1"), lambda x: (False, "err2")]
    result = _invariant_errors(elem, invariants)
    expected = ["err1", "err2"]
    assert result == expected

def test_invariant_errors_all_errors():
    elem = None
    invariants = [lambda x: (False, "err1"), lambda x: (False, "err2")]
    result = _invariant_errors(elem, invariants)
    expected = ["err1", "err2"]
    assert result == expected

def test_invariant_errors_empty_invariants():
    elem = 10
    invariants = []
    result = _invariant_errors(elem, invariants)
    expected = []
    assert result == expected

def test_invariant_errors_passes_elem_to_invariants():
    captured = []
    def capture_inv(x):
        captured.append(x)
        return (True, "ok")
    elem = 42
    invariants = [capture_inv]
    _invariant_errors(elem, invariants)
    assert captured == [elem]


# LLM-generated content at query #5
#--------------------------

def test_maybe_parse_user_type_preserved_type():
    result = maybe_parse_user_type(list)
    assert result == [list]

def test_maybe_parse_user_type_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]

def test_maybe_parse_user_type_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]

def test_maybe_parse_user_type_iterable_of_types():
    result = maybe_parse_user_type((int, str))
    assert result == (int, str)

def test_maybe_parse_user_type_nested_iterable():
    result = maybe_parse_user_type([(int, str), float])
    assert result == (int, str, float)

def test_maybe_parse_user_type_empty_iterable():
    result = maybe_parse_user_type([])
    assert result == ()

def test_maybe_parse_user_type_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False
    except TypeError:
        assert True

def test_maybe_parse_user_type_iterable_with_strings():
    result = maybe_parse_user_type(["int", "str"])
    assert result == ("int", "str")

def test_maybe_parse_user_type_mixed_iterable():
    result = maybe_parse_user_type([int, "str"])
    assert result == (int, "str")

def test_maybe_parse_user_type_deeply_nested():
    result = maybe_parse_user_type([[(int,)], [str]])
    assert result == (int, str)


# LLM-generated content at query #6
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
    from pyrsistent import pvector
    class TestVector(CheckedPVector):
        __type__ = int
    pv = pvector([1, 2, 3])
    result = TestVector(pv)
    assert isinstance(result, TestVector)
    assert list(result) == [1, 2, 3]


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


# LLM-generated content at query #7
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
    result = maybe_parse_user_type((int, (str, float)))
    assert result == (int, str, float)

def test_maybe_parse_user_type_with_empty_iterable():
    result = maybe_parse_user_type(())
    assert result == ()

def test_maybe_parse_user_type_raises_type_error_for_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False
    except TypeError:
        assert True

def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(["int", "str"])
    assert result == ("int", "str")

def test_maybe_parse_user_type_with_mixed_iterable():
    result = maybe_parse_user_type([int, "str"])
    assert result == (int, "str")


# LLM-generated content at query #8
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

def test_invariant_errors_elem_passed_to_invariants():
    captured = []
    def capture_inv(x):
        captured.append(x)
        return (True, "ok")
    elem = {"key": "value"}
    invariants = [capture_inv, capture_inv]
    _invariant_errors(elem, invariants)
    assert captured == [elem, elem]


# LLM-generated content at query #9
#--------------------------

def test_check_types_with_no_expected_types():
    result = _check_types([1, "a", 3.14], None, list)
    assert result is None

def test_check_types_with_single_matching_type():
    result = _check_types([1, 2, 3], [int], list)
    assert result is None

def test_check_types_with_multiple_matching_types():
    result = _check_types([1, "a", 3.14], [int, str, float], list)
    assert result is None

def test_check_types_with_one_non_matching_element():
    try:
        _check_types([1, "a", 3.14], [int, str], list)
        assert False
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == (int, str)
        assert e.actual_type == float
        assert e.value == 3.14

def test_check_types_with_custom_exception_type():
    class CustomError(Exception):
        pass
    try:
        _check_types([1, "a"], [int], list, CustomError)
        assert False
    except CustomError as e:
        assert e.source_class == list
        assert e.expected_types == (int,)
        assert e.actual_type == str
        assert e.value == "a"

def test_check_types_with_type_strings():
    result = _check_types([1, 2], ["builtins.int"], list)
    assert result is None

def test_check_types_with_mixed_type_objects_and_strings():
    result = _check_types([1, "a"], [int, "builtins.str"], list)
    assert result is None

def test_check_types_with_empty_iterable():
    result = _check_types([], [int, str], list)
    assert result is None

def test_check_types_with_non_iterable_source_class():
    class DummyClass:
        pass
    result = _check_types([1, 2, 3], [int], DummyClass)
    assert result is None


# LLM-generated content at query #10
#--------------------------

def test_serialize_with_default_format():
    class TestSet(CheckedPSet):
        __type__ = int
    test_set = TestSet([1, 2, 3])
    result = test_set.serialize()
    expected = {1, 2, 3}
    assert result == expected

def test_serialize_with_custom_serializer():
    class TestSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda format, v: v * 2
    test_set = TestSet([1, 2, 3])
    result = test_set.serialize()
    expected = {2, 4, 6}
    assert result == expected

def test_serialize_with_format_argument():
    class TestSet(CheckedPSet):
        __type__ = str
        __serializer__ = lambda format, v: f"{format}:{v}"
    test_set = TestSet(['a', 'b'])
    result = test_set.serialize('prefix')
    expected = {'prefix:a', 'prefix:b'}
    assert result == expected

def test_serialize_empty_set():
    class TestSet(CheckedPSet):
        __type__ = int
    test_set = TestSet([])
    result = test_set.serialize()
    expected = set()
    assert result == expected

def test_serialize_with_complex_serializer():
    class TestSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda format, v: {'value': v}
    test_set = TestSet([1, 2])
    result = test_set.serialize()
    expected = [{'value': 1}, {'value': 2}]
    assert all(item in result for item in expected) and len(result) == len(expected)


# LLM-generated content at query #11
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

def test_constructor_with_correct_types():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.0, 3: 3.14})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[3] == 3.14

def test_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.0})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'string'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_with_invariant_violation():
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
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    result = IntToFloatMap({1: 2.0, 3: 4.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 2.0
    assert result[3] == 4.0

def test_constructor_with_multiple_invariants_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    try:
        IntToFloatMap({-1: 2.0})
        assert False
    except InvariantException as e:
        assert 'Key negative' in str(e.error_codes)

def test_constructor_with_checked_type_key():
    class InnerChecked(CheckedType):
        pass
    class OuterMap(CheckedPMap):
        __key_type__ = InnerChecked
        __value_type__ = int
    inner = InnerChecked()
    result = OuterMap({inner: 5})
    assert isinstance(result, OuterMap)
    assert result[inner] == 5

def test_constructor_with_checked_type_value():
    class InnerChecked(CheckedType):
        pass
    class OuterMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = InnerChecked
    inner = InnerChecked()
    result = OuterMap({1: inner})
    assert isinstance(result, OuterMap)
    assert result[1] == inner

def test_constructor_with_union_types_key():
    class IntOrStrMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = float
    result1 = IntOrStrMap({1: 1.5})
    result2 = IntOrStrMap({'a': 2.5})
    assert isinstance(result1, IntOrStrMap)
    assert isinstance(result2, IntOrStrMap)
    assert result1[1] == 1.5
    assert result2['a'] == 2.5

def test_constructor_with_union_types_value():
    class MapWithUnionValue(CheckedPMap):
        __key_type__ = int
        __value_type__ = (float, str)
    result1 = MapWithUnionValue({1: 1.5})
    result2 = MapWithUnionValue({2: 'text'})
    assert isinstance(result1, MapWithUnionValue)
    assert isinstance(result2, MapWithUnionValue)
    assert result1[1] == 1.5
    assert result2[2] == 'text'

def test_constructor_with_checkedpmap_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.0})
    result = IntToFloatMap(original)
    assert result is original

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


# LLM-generated content at query #12
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_valid_initial_dict():
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
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v > 0, 'Value non-positive')]
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 2

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

def test_constructor_with_checked_type_key():
    class InnerKey(CheckedType):
        pass
    class MyMap(CheckedPMap):
        __key_type__ = InnerKey
        __value_type__ = int
    key = InnerKey()
    initial = {key: 5}
    result = MyMap(initial)
    assert isinstance(result, MyMap)
    assert result[key] == 5

def test_constructor_with_checked_type_value():
    class InnerValue(CheckedType):
        pass
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = InnerValue
    value = InnerValue()
    initial = {1: value}
    result = MyMap(initial)
    assert isinstance(result, MyMap)
    assert result[1] is value

def test_constructor_with_union_key_type():
    class MyMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = float
    initial = {1: 1.5, 'a': 2.5}
    result = MyMap(initial)
    assert isinstance(result, MyMap)
    assert result[1] == 1.5
    assert result['a'] == 2.5

def test_constructor_with_union_value_type():
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = (float, str)
    initial = {1: 1.5, 2: 'two'}
    result = MyMap(initial)
    assert isinstance(result, MyMap)
    assert result[1] == 1.5
    assert result[2] == 'two'

def test_constructor_with_self_instance():
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    instance = MyMap({1: 1.5})
    result = MyMap(instance)
    assert result is instance

def test_constructor_with_size_parameter():
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = MyMap({1: 1.5}, size=1)
    assert isinstance(result, MyMap)
    assert result[1] == 1.5

def test_constructor_with_empty_dict_and_size():
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = MyMap({}, size=0)
    assert isinstance(result, MyMap)
    assert len(result) == 0

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


# LLM-generated content at query #13
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

def test_constructor_invariant_pass():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_constructor_with_checkedpmap_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result is original

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5

def test_constructor_multiple_invariants():
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
        assert 'Key negative' in str(e.error_codes)
    try:
        IntToFloatMap({1: 0.0})
        assert False
    except InvariantException as e:
        assert 'Value non-positive' in str(e.error_codes)

def test_constructor_checked_type_create_key():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return int(source_data)
    class MyMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = int
    result = MyMap({"1": 10})
    assert isinstance(result, MyMap)
    assert result[1] == 10

def test_constructor_checked_type_create_value():
    class ValueType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return str(source_data)
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = ValueType
    result = MyMap({1: 100})
    assert isinstance(result, MyMap)
    assert result[1] == "100"

def test_constructor_checked_type_create_both():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return int(source_data)
    class ValueType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return str(source_data)
    class MyMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = MyMap({"1": 100})
    assert isinstance(result, MyMap)
    assert result[1] == "100"

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


# LLM-generated content at query #14
#--------------------------

def test_CheckedType_constructor_creates_instance():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)

def test_CheckedType_constructor_slots_are_empty():
    instance = CheckedType()
    assert not hasattr(instance, '__dict__')
    assert instance.__slots__ == ()

def test_CheckedType_constructor_has_required_methods():
    instance = CheckedType()
    assert hasattr(instance, 'create')
    assert hasattr(instance, 'serialize')

def test_CheckedType_constructor_create_is_classmethod():
    assert isinstance(CheckedType.create, classmethod)

def test_CheckedType_constructor_serialize_is_method():
    instance = CheckedType()
    assert callable(instance.serialize)


# LLM-generated content at query #15
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
        assert True
    except MockExceptionType:
        assert False

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
        assert False
    except MockExceptionType:
        assert True

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
        assert True
    except MockExceptionType:
        assert False

def test_check_types_with_none_expected_types():
    class MockSourceClass:
        pass
    class MockExceptionType(Exception):
        pass
    it = [object()]
    expected_types = None
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        assert True
    except MockExceptionType:
        assert False

def test_check_types_with_multiple_expected_types_and_matching_element():
    class MockType1:
        pass
    class MockType2:
        pass
    class MockSourceClass:
        pass
    class MockExceptionType(Exception):
        pass
    it = [MockType2()]
    expected_types = [MockType1, MockType2]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        assert True
    except MockExceptionType:
        assert False

def test_check_types_with_multiple_expected_types_and_non_matching_element():
    class MockType1:
        pass
    class MockType2:
        pass
    class OtherType:
        pass
    class MockSourceClass:
        pass
    class MockExceptionType(Exception):
        pass
    it = [OtherType()]
    expected_types = [MockType1, MockType2]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        assert False
    except MockExceptionType:
        assert True

def test_check_types_with_iterator_and_matching_element():
    class MockType:
        pass
    class MockSourceClass:
        pass
    class MockExceptionType(Exception):
        pass
    it = iter([MockType()])
    expected_types = [MockType]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        assert True
    except MockExceptionType:
        assert False

def test_check_types_with_multiple_elements_and_matching_elements():
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
        assert True
    except MockExceptionType:
        assert False

def test_check_types_with_multiple_elements_and_one_non_matching_element():
    class MockType:
        pass
    class OtherType:
        pass
    class MockSourceClass:
        pass
    class MockExceptionType(Exception):
        pass
    it = [MockType(), OtherType()]
    expected_types = [MockType]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        assert False
    except MockExceptionType:
        assert True

def test_check_types_with_default_exception_type():
    class MockType:
        pass
    class OtherType:
        pass
    class MockSourceClass:
        pass
    it = [OtherType()]
    expected_types = [MockType]
    source_class = MockSourceClass
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError:
        assert True


# LLM-generated content at query #16
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
    class MockType:
        pass
    class MockSourceClass:
        __name__ = "MockSourceClass"
    class MockExceptionType(Exception):
        pass
    it = [MockType()]
    expected_types = []
    source_class = MockSourceClass
    exception_type = MockExceptionType
    result = _check_types(it, expected_types, source_class, exception_type)
    assert result is None

def test_check_types_with_none_expected_types():
    class MockType:
        pass
    class MockSourceClass:
        __name__ = "MockSourceClass"
    class MockExceptionType(Exception):
        pass
    it = [MockType()]
    expected_types = None
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

def test_check_types_with_multiple_elements_all_matching():
    class MockType:
        pass
    class MockSourceClass:
        __name__ = "MockSourceClass"
    class MockExceptionType(Exception):
        pass
    it = [MockType(), MockType(), MockType()]
    expected_types = [MockType]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    result = _check_types(it, expected_types, source_class, exception_type)
    assert result is None

def test_check_types_with_multiple_elements_one_non_matching():
    class MockType:
        pass
    class MockOtherType:
        pass
    class MockSourceClass:
        __name__ = "MockSourceClass"
    class MockExceptionType(Exception):
        pass
    it = [MockType(), MockOtherType(), MockType()]
    expected_types = [MockType]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    try:
        _check_types(it, expected_types, source_class, exception_type)
        assert False
    except MockExceptionType:
        assert True

def test_check_types_with_default_exception_type():
    class MockType:
        pass
    class MockOtherType:
        pass
    class MockSourceClass:
        __name__ = "MockSourceClass"
    it = [MockOtherType()]
    expected_types = [MockType]
    source_class = MockSourceClass
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError:
        assert True

def test_check_types_with_get_type_function():
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


# LLM-generated content at query #17
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


# LLM-generated content at query #18
#--------------------------

def test__checked_type_create_with_same_class():
    class MockClass:
        pass
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance

def test__checked_type_create_with_checked_type_subclass():
    class CheckedType:
        _checked_types = ["__main__.MockType"]
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
    class MockType(CheckedType):
        def __init__(self, data):
            self.data = data
    source_data = [1, 2, 3]
    result = _checked_type_create(CheckedType, source_data)
    assert isinstance(result, CheckedType)
    assert all(isinstance(item, MockType) for item in result.data)

def test__checked_type_create_with_matching_type_in_list():
    class MockType:
        pass
    class MockClass:
        _checked_types = [MockType]
    instance = MockType()
    source_data = [instance]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert result.data[0] is instance

def test__checked_type_create_without_checked_type():
    class MockClass:
        def __init__(self, data):
            self.data = data
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert result.data == source_data

def test__checked_type_create_with_ignore_extra():
    class CheckedType:
        _checked_types = ["__main__.MockType"]
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
    class MockType(CheckedType):
        def __init__(self, data):
            self.data = data
    source_data = [1, 2, 3]
    result = _checked_type_create(CheckedType, source_data, ignore_extra=True)
    assert isinstance(result, CheckedType)
    assert all(isinstance(item, MockType) for item in result.data)


# LLM-generated content at query #19
#--------------------------

def test_preserved_type():
    class MockPreserved:
        pass
    MockPreserved.__bases__ = (_preserved_iterable_types,)
    result = maybe_parse_user_type(MockPreserved)
    assert result == [MockPreserved]

def test_string_type():
    result = maybe_parse_user_type("int")
    assert result == ["int"]

def test_simple_type():
    result = maybe_parse_user_type(int)
    assert result == [int]

def test_iterable_of_types():
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)

def test_nested_iterable():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)

def test_invalid_type_raises():
    try:
        maybe_parse_user_type(123)
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #20
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
    assert dict(result) == initial

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
    initial = {1: 1.0, 2: 2.0}
    result = IntToFloatMap(initial)
    assert dict(result) == initial

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({}, size=0)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_checked_key_type_create():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return int(source_data)
    class MyMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = int
    initial = {'1': 10, '2': 20}
    result = MyMap(initial)
    expected = {1: 10, 2: 20}
    assert dict(result) == expected

def test_constructor_with_checked_value_type_create():
    class ValueType(CheckedType):
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return str(source_data)
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = ValueType
    initial = {1: 100, 2: 200}
    result = MyMap(initial)
    expected = {1: '100', 2: '200'}
    assert dict(result) == expected

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


def test_constructor_with_set_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives({4, 5})
    assert isinstance(result, Positives)
    assert len(result) == 2
    assert 4 in result
    assert 5 in result


def test_constructor_with_tuple_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives((6, 7, 8))
    assert isinstance(result, Positives)
    assert len(result) == 3
    assert 6 in result
    assert 7 in result
    assert 8 in result


def test_constructor_with_pmap_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    pmap_initial = pmap({9: True, 10: True})
    result = Positives(pmap_initial)
    assert isinstance(result, Positives)
    assert len(result) == 2
    assert 9 in result
    assert 10 in result


def test_constructor_type_check_failure():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, 'a'])
        assert False
    except TypeError:
        assert True


def test_constructor_invariant_check_failure():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -1])
        assert False
    except InvariantException:
        assert True


def test_constructor_duplicate_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 1, 2, 2])
    assert isinstance(result, Positives)
    assert len(result) == 2
    assert 1 in result
    assert 2 in result


def test_constructor_no_type_or_invariant():
    class SimpleSet(CheckedPSet):
        pass
    result = SimpleSet([1, 2, 3])
    assert isinstance(result, SimpleSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


# LLM-generated content at query #22
#--------------------------

def test_predicate_at_line_18_evaluates_to_true_for_type_and_not_iterable():
    result = maybe_parse_user_type(int)
    assert result == [int]


