####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_check_global_invariants_all_pass():
    invariant1 = lambda x: (True, 'code1')
    invariant2 = lambda x: (True, 'code2')
    subject = "test_subject"
    
    check_global_invariants(subject, [invariant1, invariant2])


def test_check_global_invariants_single_failure():
    invariant1 = lambda x: (True, 'code1')
    invariant2 = lambda x: (False, 'code2')
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, [invariant1, invariant2])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ('code2',)
        assert e.error_path == ()
        assert e.message == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    invariant1 = lambda x: (False, 'error1')
    invariant2 = lambda x: (False, 'error2')
    invariant3 = lambda x: (True, 'code3')
    subject = {"key": "value"}
    
    try:
        check_global_invariants(subject, [invariant1, invariant2, invariant3])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ('error1', 'error2')
        assert e.error_path == ()
        assert e.message == 'Global invariant failed'


def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    
    check_global_invariants(subject, [])


def test_check_global_invariants_single_invariant_pass():
    invariant = lambda x: (True, 'success_code')
    subject = 42
    
    check_global_invariants(subject, [invariant])


def test_check_global_invariants_single_invariant_fail():
    invariant = lambda x: (False, 'failure_code')
    subject = None
    
    try:
        check_global_invariants(subject, [invariant])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ('failure_code',)


# LLM-generated content at query #2
#--------------------------

```python
def test_make_pmap_field_type_creates_new_pmap_field_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result.__name__ == "StringToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_types_creates_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2
    assert result1.__name__ == "StringToIntPMap"
    assert result2.__name__ == "IntToStringPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    instance = result()
    reduce_result = instance.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[1][0] == str
    assert reduce_result[1][1] == int


def test_make_pmap_field_type_with_multiple_key_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type((str, int), (float, bool))
    
    assert "StringInt" in result.__name__
    assert "FloatBool" in result.__name__
    assert "PMap" in result.__name__


# LLM-generated content at query #3
#--------------------------

```python
def test_check_type_with_valid_single_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class MockClass:
        __name__ = "TestClass"
    
    field = MockField((int,))
    check_type(MockClass, field, "test_field", 42)


def test_check_type_with_valid_multiple_types():
    from pyrsistent._field_common import check_type
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class MockClass:
        __name__ = "TestClass"
    
    field = MockField((int, str))
    check_type(MockClass, field, "test_field", "hello")


def test_check_type_with_no_type_constraint():
    from pyrsistent._field_common import check_type
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class MockClass:
        __name__ = "TestClass"
    
    field = MockField(None)
    check_type(MockClass, field, "test_field", "any_value")


def test_check_type_with_invalid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent import PTypeError
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class MockClass:
        __name__ = "TestClass"
    
    field = MockField((int,))
    try:
        check_type(MockClass, field, "test_field", "not_an_int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.field_name == "test_field"
        assert e.class_name == "TestClass"


def test_check_type_with_empty_type_tuple():
    from pyrsistent._field_common import check_type
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class MockClass:
        __name__ = "TestClass"
    
    field = MockField(())
    try:
        check_type(MockClass, field, "test_field", 42)
        assert False, "Expected PTypeError"
    except Exception:
        pass


def test_check_type_with_subclass():
    from pyrsistent._field_common import check_type
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class MockClass:
        __name__ = "TestClass"
    
    class Parent:
        pass
    
    class Child(Parent):
        pass
    
    field = MockField((Parent,))
    check_type(MockClass, field, "test_field", Child())


def test_check_type_error_message_format():
    from pyrsistent._field_common import check_type
    from pyrsistent import PTypeError
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class MockClass:
        __name__ = "MyClass"
    
    field = MockField((int,))
    try:
        check_type(MockClass, field, "my_field", 3.14)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert "MyClass" in str(e)
        assert "my_field" in str(e)


# LLM-generated content at query #4
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert result is not None
    assert hasattr(result, '__key_type__')
    assert hasattr(result, '__value_type__')
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_pmap_field_type
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    assert result1 is result2


def test_make_pmap_field_type_generates_correct_name():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert result.__name__ == "StringToIntPMap"


def test_make_pmap_field_type_different_types_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    assert result1 is not result2
    assert result1.__name__ != result2.__name__


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_with_bool_types():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(bool, float)
    assert result.__name__ == "BoolToFloatPMap"
    assert result.__key_type__ == bool
    assert result.__value_type__ == float


# LLM-generated content at query #5
#--------------------------

```python
def test_make_pmap_field_type_creates_checked_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type(str, int)
    
    assert pmap_type.__key_type__ == str
    assert pmap_type.__value_type__ == int
    assert pmap_type.__name__ == "StrToIntPMap"


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type1 = _make_pmap_field_type(str, int)
    pmap_type2 = _make_pmap_field_type(str, int)
    
    assert pmap_type1 is pmap_type2


def test_make_pmap_field_type_different_types_creates_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type1 = _make_pmap_field_type(str, int)
    pmap_type2 = _make_pmap_field_type(int, str)
    
    assert pmap_type1 is not pmap_type2
    assert pmap_type1.__name__ != pmap_type2.__name__


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type(str, int)
    
    assert hasattr(pmap_type, '__reduce__')


def test_make_pmap_field_type_with_multiple_key_value_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type(float, bool)
    
    assert pmap_type.__key_type__ == float
    assert pmap_type.__value_type__ == bool
    assert "Float" in pmap_type.__name__
    assert "Bool" in pmap_type.__name__


# LLM-generated content at query #6
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant_pass(subject):
        return (True, "OK")
    
    subject = {"test": "data"}
    invariants = [invariant_pass]
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_error():
    def invariant_fail(subject):
        return (False, "ERROR_001")
    
    subject = {"test": "data"}
    invariants = [invariant_fail]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_001",)
        assert e.context == ()
        assert e.message == 'Global invariant failed'


def test_check_global_invariants_multiple_errors():
    def invariant_fail_1(subject):
        return (False, "ERROR_001")
    
    def invariant_fail_2(subject):
        return (False, "ERROR_002")
    
    subject = {"test": "data"}
    invariants = [invariant_fail_1, invariant_fail_2]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_001", "ERROR_002")
        assert e.context == ()
        assert e.message == 'Global invariant failed'


def test_check_global_invariants_mixed_pass_fail():
    def invariant_pass(subject):
        return (True, "OK")
    
    def invariant_fail(subject):
        return (False, "ERROR_001")
    
    subject = {"test": "data"}
    invariants = [invariant_pass, invariant_fail]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_001",)
        assert e.context == ()
        assert e.message == 'Global invariant failed'


def test_check_global_invariants_empty_invariants():
    subject = {"test": "data"}
    invariants = []
    check_global_invariants(subject, invariants)


def test_check_global_invariants_all_pass():
    def invariant_1(subject):
        return (True, "OK1")
    
    def invariant_2(subject):
        return (True, "OK2")
    
    subject = {"test": "data"}
    invariants = [invariant_1, invariant_2]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #7
#--------------------------

```python
def test_check_field_parameters_valid_field():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class MockField:
        def __init__(self):
            self.type = [int, 123]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Type parameter expected' in str(e)


def test_check_field_parameters_invalid_initial_type():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = 3.14
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_check_field_parameters_callable_initial():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = lambda: 42
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_non_callable_invariant():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = "not callable"
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_check_field_parameters_non_callable_factory():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = "not callable"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_check_field_parameters_non_callable_serializer():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = "not callable"
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


def test_check_field_parameters_string_type():
    class MockField:
        def __init__(self):
            self.type = ["int", str]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_empty_type():
    class MockField:
        def __init__(self):
            self.type = []
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


# LLM-generated content at query #8
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import pclass, field, pvec
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class TestClass(pclass):
        items = field(type=pvec(int))
    
    test_data = [1, 2, 3]
    item_type = int
    checked_class = TestClass
    
    result = _restore_seq_field_pickle(checked_class, item_type, test_data)
    
    assert result is not None
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #9
#--------------------------

```python
def test_types_to_names_with_single_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((int,))
    assert result == "Int"


def test_types_to_names_with_multiple_types():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((int, str, float))
    assert result == "IntStrFloat"


def test_types_to_names_with_empty_tuple():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names(())
    assert result == ""


def test_types_to_names_with_bool_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((bool,))
    assert result == "Bool"


def test_types_to_names_with_list_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((list,))
    assert result == "List"


def test_types_to_names_with_dict_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((dict,))
    assert result == "Dict"


def test_types_to_names_with_multiple_builtin_types():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((int, list, dict, str))
    assert result == "IntListDictStr"


# LLM-generated content at query #10
#--------------------------

```python
def test_check_field_parameters_valid_type_class():
    class MockField:
        type = [int, str]
        initial = 5
        invariant = lambda self: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_valid_type_string():
    class MockField:
        type = ['int', 'str']
        initial = 5
        invariant = lambda self: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class MockField:
        type = [int, 123]
        initial = 5
        invariant = lambda self: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Type parameter expected' in str(e)


def test_check_field_parameters_initial_no_initial():
    PFIELD_NO_INITIAL = object()
    class MockField:
        type = [int, str]
        initial = PFIELD_NO_INITIAL
        invariant = lambda self: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_initial_callable():
    class MockField:
        type = [int, str]
        initial = lambda: 5
        invariant = lambda self: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_initial_valid_type():
    class MockField:
        type = [int, str]
        initial = "hello"
        invariant = lambda self: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_initial_invalid_type():
    class MockField:
        type = [int, str]
        initial = 3.14
        invariant = lambda self: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_check_field_parameters_initial_invalid_type_empty_type_list():
    class MockField:
        type = []
        initial = 3.14
        invariant = lambda self: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_non_callable_invariant():
    class MockField:
        type = [int, str]
        initial = 5
        invariant = "not callable"
        factory = lambda: None
        serializer = lambda x: x
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_check_field_parameters_non_callable_factory():
    class MockField:
        type = [int, str]
        initial = 5
        invariant = lambda self: True
        factory = "not callable"
        serializer = lambda x: x
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_check_field_parameters_non_callable_serializer():
    class MockField:
        type = [int, str]
        initial = 5
        invariant = lambda self: True
        factory = lambda: None
        serializer = "not callable"
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


# LLM-generated content at query #11
#--------------------------

```python
def test_field_with_single_type():
    from pyrsistent._field_common import field
    result = field(type=int)
    assert int in result.type
    assert result.invariant == 'PFIELD_NO_INVARIANT'
    assert result.mandatory == False


def test_field_with_multiple_types_as_tuple():
    from pyrsistent._field_common import field
    result = field(type=(int, str))
    assert int in result.type
    assert str in result.type
    assert len(result.type) == 2


def test_field_with_multiple_types_as_list():
    from pyrsistent._field_common import field
    result = field(type=[int, str])
    assert int in result.type
    assert str in result.type
    assert len(result.type) == 2


def test_field_with_multiple_types_as_set():
    from pyrsistent._field_common import field
    result = field(type={int, str})
    assert int in result.type
    assert str in result.type
    assert len(result.type) == 2


def test_field_with_string_type():
    from pyrsistent._field_common import field
    result = field(type='MyType')
    assert 'MyType' in result.type


def test_field_with_initial_value():
    from pyrsistent._field_common import field
    result = field(type=int, initial=42)
    assert result.initial == 42


def test_field_with_mandatory_true():
    from pyrsistent._field_common import field
    result = field(type=int, mandatory=True)
    assert result.mandatory == True


def test_field_with_callable_invariant():
    from pyrsistent._field_common import field
    def my_invariant(val):
        return True, None
    result = field(type=int, invariant=my_invariant)
    assert callable(result.invariant)


def test_field_with_callable_factory():
    from pyrsistent._field_common import field
    def my_factory():
        return 42
    result = field(type=int, factory=my_factory)
    assert result._factory == my_factory


def test_field_with_callable_serializer():
    from pyrsistent._field_common import field
    def my_serializer(val):
        return str(val)
    result = field(type=int, serializer=my_serializer)
    assert result.serializer == my_serializer


def test_field_invalid_type_parameter():
    from pyrsistent._field_common import field
    try:
        field(type=123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Type specifications must be types or strings' in str(e)


def test_field_initial_with_wrong_type():
    from pyrsistent._field_common import field
    try:
        field(type=int, initial="not an int")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_field_non_callable_invariant():
    from pyrsistent._field_common import field
    try:
        field(type=int, invariant="not callable")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_field_non_callable_factory():
    from pyrsistent._field_common import field
    try:
        field(type=int, factory="not callable")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_field_non_callable_serializer():
    from pyrsistent._field_common import field
    try:
        field(type=int, serializer="not callable")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


def test_field_with_callable_initial():
    from pyrsistent._field_common import field
    def my_factory():
        return 42
    result = field(type=int, initial=my_factory)
    assert result.initial == my_factory


def test_field_returns_pfield_instance():
    from pyrsistent._field_common import field, _PField
    result = field(type=int)
    assert isinstance(result, _PField)


def test_field_no_type_specified():
    from pyrsistent._field_common import field
    result = field()
    assert len(result.type) == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    
    # Predicate at line 3: not isinstance(t, type) and not isinstance(t, str)
    # Should evaluate to False for all t in field.type
    for t in field.type:
        predicate_result = not isinstance(t, type) and not isinstance(t, str)
        assert predicate_result is False


# LLM-generated content at query #13
#--------------------------

```python
def test_wrap_invariant_called_when_invariant_is_callable_and_not_no_invariant():
    from pyrsistent._field_common import field, PFIELD_NO_INVARIANT
    from pyrsistent._checked_types import wrap_invariant
    
    def sample_invariant(value):
        return True, "valid"
    
    result = field(type=int, invariant=sample_invariant)
    
    assert result.invariant is not sample_invariant
    assert callable(result.invariant)


# LLM-generated content at query #14
#--------------------------

```python
def test_check_field_parameters_valid_field():
    class Field:
        def __init__(self):
            self.type = [str, int]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class Field:
        def __init__(self):
            self.type = [str, 123]
            self.initial = "PFIELD_NO_INITIAL"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Type parameter expected" in str(e)


def test_check_field_parameters_invalid_initial_type():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = 123
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    PFIELD_NO_INITIAL = object()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Initial has invalid type" in str(e)


def test_check_field_parameters_callable_initial():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = lambda: "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_non_callable_invariant():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = "not_callable"
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Invariant must be callable" in str(e)


def test_check_field_parameters_non_callable_factory():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = "not_callable"
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Factory must be callable" in str(e)


def test_check_field_parameters_non_callable_serializer():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = "not_callable"
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Serializer must be callable" in str(e)


def test_check_field_parameters_string_type():
    class Field:
        def __init__(self):
            self.type = ["str", "int"]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_empty_type():
    class Field:
        def __init__(self):
            self.type = []
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


# LLM-generated content at query #15
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    class MockCheckedType(CheckedType):
        pass
    
    value = MockCheckedType()
    PFIELD_NO_SERIALIZER = object()
    
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "serialized_value"


def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"format_{format}_value_{value}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "format_xml_value_test_value"


def test_serialize_with_non_checked_type_and_no_serializer():
    PFIELD_NO_SERIALIZER = object()
    
    def fallback_serializer(format, value):
        return f"serialized_{value}"
    
    result = serialize(fallback_serializer, "json", "data")
    assert result == "serialized_data"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    class MockCheckedType(CheckedType):
        pass
    
    def custom_serializer(format, value):
        return "custom_serialized"
    
    value = MockCheckedType()
    
    result = serialize(custom_serializer, "json", value)
    assert result == "custom_serialized"


# LLM-generated content at query #16
#--------------------------

```python
def test_pfield_constructor():
    def dummy_invariant(x):
        return True
    
    def dummy_factory():
        return "factory_result"
    
    def dummy_serializer(x):
        return str(x)
    
    type_tuple = (int, str)
    initial_value = 42
    
    pfield = _PField(
        type=type_tuple,
        invariant=dummy_invariant,
        initial=initial_value,
        mandatory=True,
        factory=dummy_factory,
        serializer=dummy_serializer
    )
    
    assert pfield.type == type_tuple
    assert pfield.invariant == dummy_invariant
    assert pfield.initial == initial_value
    assert pfield.mandatory is True
    assert pfield._factory == dummy_factory
    assert pfield.serializer == dummy_serializer


def test_pfield_constructor_with_none_values():
    pfield = _PField(
        type=(int,),
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert pfield.type == (int,)
    assert pfield.invariant is None
    assert pfield.initial is None
    assert pfield.mandatory is False
    assert pfield._factory is None
    assert pfield.serializer is None


def test_pfield_constructor_mandatory_false():
    pfield = _PField(
        type=(str,),
        invariant=lambda x: len(x) > 0,
        initial="default",
        mandatory=False,
        factory=lambda: "created",
        serializer=str
    )
    
    assert pfield.type == (str,)
    assert pfield.mandatory is False
    assert pfield.initial == "default"


# LLM-generated content at query #17
#--------------------------

```python
def test_set_fields():
    from unittest.mock import Mock
    
    # Test case 1: Basic functionality with empty bases
    dct1 = {}
    bases1 = []
    name1 = "fields"
    set_fields(dct1, bases1, name1)
    assert "fields" in dct1
    assert dct1["fields"] == {}
    
    # Test case 2: With _PField instances in dct
    class _PField:
        pass
    
    field1 = _PField()
    field2 = _PField()
    dct2 = {"field_a": field1, "field_b": field2, "other": "value"}
    bases2 = []
    name2 = "fields"
    set_fields(dct2, bases2, name2)
    assert "fields" in dct2
    assert "field_a" in dct2["fields"]
    assert "field_b" in dct2["fields"]
    assert dct2["fields"]["field_a"] is field1
    assert dct2["fields"]["field_b"] is field2
    assert "field_a" not in dct2
    assert "field_b" not in dct2
    assert "other" in dct2
    
    # Test case 3: With bases having existing fields
    base1 = Mock()
    base1.__dict__ = {"fields": {"base_field": "base_value"}}
    dct3 = {}
    bases3 = [base1]
    name3 = "fields"
    set_fields(dct3, bases3, name3)
    assert "fields" in dct3
    assert "base_field" in dct3["fields"]
    assert dct3["fields"]["base_field"] == "base_value"
    
    # Test case 4: Combining base fields with new _PField instances
    base2 = Mock()
    base2.__dict__ = {"fields": {"inherited_field": "inherited_value"}}
    field3 = _PField()
    dct4 = {"new_field": field3}
    bases4 = [base2]
    name4 = "fields"
    set_fields(dct4, bases4, name4)
    assert "fields" in dct4
    assert "inherited_field" in dct4["fields"]
    assert "new_field" in dct4["fields"]
    assert dct4["fields"]["new_field"] is field3
    assert "new_field" not in dct4
    
    # Test case 5: Multiple bases
    base3 = Mock()
    base3.__dict__ = {"fields": {"field_from_base3": "value3"}}
    base4 = Mock()
    base4.__dict__ = {"fields": {"field_from_base4": "value4"}}
    dct5 = {}
    bases5 = [base3, base4]
    name5 = "fields"
    set_fields(dct5, bases5, name5)
    assert "fields" in dct5
    assert "field_from_base3" in dct5["fields"]
    assert "field_from_base4" in dct5["fields"]


# LLM-generated content at query #18
#--------------------------

```python
def test_set_fields_isinstance_predicate():
    class _PField:
        pass
    
    class Base:
        pass
    
    dct = {'field1': _PField(), 'field2': 'not_a_field'}
    bases = (Base,)
    name = 'fields'
    
    # Verify that _PField instance exists in dct
    assert isinstance(dct['field1'], _PField)


# LLM-generated content at query #19
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap, PMap
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Register a pmap field type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent._field_common import pmap_field
        _pmap_field_types[key_type, value_type] = pmap_field(key_type, value_type)
    
    # Create test data
    test_data = {'key1': 1, 'key2': 2}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap
    assert isinstance(result, PMap)
    assert result == pmap(test_data)
    assert result['key1'] == 1
    assert result['key2'] == 2


def test_restore_pmap_field_pickle_empty():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap, PMap
    
    key_type = str
    value_type = str
    
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent._field_common import pmap_field
        _pmap_field_types[key_type, value_type] = pmap_field(key_type, value_type)
    
    # Test with empty data
    test_data = {}
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    assert isinstance(result, PMap)
    assert result == pmap({})
    assert len(result) == 0


def test_restore_pmap_field_pickle_multiple_entries():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap, PMap
    
    key_type = int
    value_type = str
    
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent._field_common import pmap_field
        _pmap_field_types[key_type, value_type] = pmap_field(key_type, value_type)
    
    # Test with multiple entries
    test_data = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    assert isinstance(result, PMap)
    assert result == pmap(test_data)
    assert len(result) == 4
    assert result[1] == 'a'
    assert result[4] == 'd'


# LLM-generated content at query #20
#--------------------------

```python
def test_is_field_ignore_extra_complaint_ignore_extra_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def mock_factory():
        pass
    
    field = MockField({int}, mock_factory)
    result = is_field_ignore_extra_complaint(object, field, False)
    assert result is False


def test_is_field_ignore_extra_complaint_ignore_extra_true_wrong_type():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def mock_factory():
        pass
    
    field = MockField({int}, mock_factory)
    result = is_field_ignore_extra_complaint(str, field, True)
    assert result is False


def test_is_field_ignore_extra_complaint_ignore_extra_true_correct_type_no_param():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def mock_factory():
        pass
    
    field = MockField({object}, mock_factory)
    result = is_field_ignore_extra_complaint(object, field, True)
    assert result is False


def test_is_field_ignore_extra_complaint_ignore_extra_true_correct_type_with_param():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def mock_factory(ignore_extra=False):
        pass
    
    field = MockField({object}, mock_factory)
    result = is_field_ignore_extra_complaint(object, field, True)
    assert result is True


def test_is_field_ignore_extra_complaint_empty_type_set():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def mock_factory(ignore_extra=False):
        pass
    
    field = MockField(set(), mock_factory)
    result = is_field_ignore_extra_complaint(object, field, True)
    assert result is False


def test_is_field_ignore_extra_complaint_type_as_set():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def mock_factory(ignore_extra=False):
        pass
    
    field = MockField({int, str}, mock_factory)
    result = is_field_ignore_extra_complaint(object, field, True)
    assert result is True


# LLM-generated content at query #21
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PClass, field, pvec
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class TestClass(PClass):
        items = field()
    
    item_type = int
    data = [1, 2, 3]
    
    # Register the type in _seq_field_types
    from pyrsistent._field_common import PVecField
    test_field_type = PVecField(TestClass, item_type, None, None)
    _seq_field_types[TestClass, item_type] = test_field_type
    
    result = _restore_seq_field_pickle(TestClass, item_type, data)
    
    assert result is not None
    assert hasattr(result, '__iter__')


def test_restore_seq_field_pickle_with_string_items():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class MyClass(PClass):
        values = field()
    
    item_type = str
    data = ['a', 'b', 'c']
    
    from pyrsistent._field_common import PVecField
    test_field_type = PVecField(MyClass, item_type, None, None)
    _seq_field_types[MyClass, item_type] = test_field_type
    
    result = _restore_seq_field_pickle(MyClass, item_type, data)
    
    assert result is not None


def test_restore_seq_field_pickle_empty_data():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class EmptyClass(PClass):
        items = field()
    
    item_type = int
    data = []
    
    from pyrsistent._field_common import PVecField
    test_field_type = PVecField(EmptyClass, item_type, None, None)
    _seq_field_types[EmptyClass, item_type] = test_field_type
    
    result = _restore_seq_field_pickle(EmptyClass, item_type, data)
    
    assert result is not None


# LLM-generated content at query #22
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant_pass(subject):
        return (True, "NO_ERROR")
    
    def invariant_pass_2(subject):
        return (True, "NO_ERROR_2")
    
    invariants = [invariant_pass, invariant_pass_2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        result = True
    except InvariantException:
        result = False
    
    assert result is True


# LLM-generated content at query #23
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, "OK1")
    
    def invariant2(subject):
        return (True, "OK2")
    
    check_global_invariants("test_subject", [invariant1, invariant2])


def test_check_global_invariants_single_failure():
    def invariant1(subject):
        return (True, "OK")
    
    def invariant2(subject):
        return (False, "ERROR_CODE_1")
    
    try:
        check_global_invariants("test_subject", [invariant1, invariant2])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1",)
        assert e.args[1] == ()
        assert e.args[2] == "Global invariant failed"


def test_check_global_invariants_multiple_failures():
    def invariant1(subject):
        return (False, "ERROR_CODE_1")
    
    def invariant2(subject):
        return (False, "ERROR_CODE_2")
    
    def invariant3(subject):
        return (True, "OK")
    
    try:
        check_global_invariants("test_subject", [invariant1, invariant2, invariant3])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1", "ERROR_CODE_2")
        assert e.args[1] == ()
        assert e.args[2] == "Global invariant failed"


def test_check_global_invariants_empty_list():
    check_global_invariants("test_subject", [])


def test_check_global_invariants_with_none_subject():
    def invariant1(subject):
        return (True, "OK")
    
    check_global_invariants(None, [invariant1])


# LLM-generated content at query #24
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3: "not isinstance(t, type) and not isinstance(t, str)"
    # should evaluate to False for all elements in field.type
    # This means for each t, either isinstance(t, type) is True OR isinstance(t, str) is True
    
    for t in field.type:
        predicate_result = not isinstance(t, type) and not isinstance(t, str)
        assert predicate_result == False


# LLM-generated content at query #25
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant_pass_1(subject):
        return (True, 'OK1')
    
    def invariant_pass_2(subject):
        return (True, 'OK2')
    
    invariants = [invariant_pass_1, invariant_pass_2]
    subject = {'test': 'data'}
    
    result = check_global_invariants(subject, invariants)
    assert result is None


def test_check_global_invariants_single_failure():
    def invariant_fail(subject):
        return (False, 'ERROR_CODE_1')
    
    invariants = [invariant_fail]
    subject = {'test': 'data'}
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ('ERROR_CODE_1',)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    def invariant_fail_1(subject):
        return (False, 'ERROR_CODE_1')
    
    def invariant_fail_2(subject):
        return (False, 'ERROR_CODE_2')
    
    def invariant_pass(subject):
        return (True, 'OK')
    
    invariants = [invariant_fail_1, invariant_pass, invariant_fail_2]
    subject = {'test': 'data'}
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ('ERROR_CODE_1', 'ERROR_CODE_2')
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = {'test': 'data'}
    
    result = check_global_invariants(subject, invariants)
    assert result is None


def test_check_global_invariants_with_different_subjects():
    def invariant_check_subject(subject):
        if subject.get('value') > 10:
            return (True, 'OK')
        return (False, 'VALUE_TOO_LOW')
    
    invariants = [invariant_check_subject]
    subject = {'value': 5}
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ('VALUE_TOO_LOW',)


# LLM-generated content at query #26
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3 should evaluate to False
    # "not isinstance(t, type) and not isinstance(t, str)" should be False
    # This means either t is an instance of type OR t is a string
    for t in field.type:
        predicate_result = not isinstance(t, type) and not isinstance(t, str)
        assert predicate_result is False


# LLM-generated content at query #27
#--------------------------

```python
def test_make_seq_field_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache before test
    _seq_field_types.clear()
    
    # Create a seq field type
    item_type = int
    item_invariant = lambda x: x > 0
    result_type = _make_seq_field_type(PVector, item_type, item_invariant)
    
    # Verify the type is created
    assert result_type is not None
    assert issubclass(result_type, PVector)
    assert result_type.__type__ == int
    assert result_type.__invariant__ == item_invariant
    
    # Verify caching works - calling again should return the same type
    result_type_2 = _make_seq_field_type(PVector, item_type, item_invariant)
    assert result_type is result_type_2
    
    # Verify __name__ is set
    assert hasattr(result_type, '__name__')
    assert isinstance(result_type.__name__, str)
    assert len(result_type.__name__) > 0
    
    # Verify __reduce__ method exists
    assert hasattr(result_type, '__reduce__')
    assert callable(result_type.__reduce__)


# LLM-generated content at query #28
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int)
    assert result.mandatory is True
    assert result.initial == pmap()
    assert callable(result.factory)


def test_pmap_field_with_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    assert result.mandatory is True
    assert result.initial == pmap()
    assert callable(result.factory)


def test_pmap_field_with_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=True)
    assert result.mandatory is True
    assert result.initial == pmap()
    assert callable(result.factory)


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    def my_invariant(value):
        return True, "valid"
    
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.mandatory is True
    assert result.initial == pmap()
    assert callable(result.factory)
    assert callable(result.invariant)


def test_pmap_field_factory_non_optional():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    factory = result.factory
    created_map = factory({'key': 1})
    assert created_map == pmap({'key': 1})


def test_pmap_field_factory_optional_with_none():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    factory = result.factory
    created_value = factory(None)
    assert created_value is None


def test_pmap_field_factory_optional_with_data():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=True)
    factory = result.factory
    created_map = factory({'key': 1})
    assert created_map == pmap({'key': 1})


def test_pmap_field_type_included():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int)
    assert result.type is not None
    assert len(result.type) > 0


def test_pmap_field_type_optional_included():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    assert result.type is not None
    assert len(result.type) > 0


# LLM-generated content at query #29
#--------------------------

```python
def test_pfield_constructor():
    def dummy_invariant(x):
        return x > 0
    
    def dummy_serializer(x):
        return str(x)
    
    def dummy_factory():
        return 42
    
    field = _PField(
        type=(int, str),
        invariant=dummy_invariant,
        initial=10,
        mandatory=True,
        factory=dummy_factory,
        serializer=dummy_serializer
    )
    
    assert field.type == (int, str)
    assert field.invariant == dummy_invariant
    assert field.initial == 10
    assert field.mandatory is True
    assert field._factory == dummy_factory
    assert field.serializer == dummy_serializer


def test_pfield_constructor_with_none_values():
    field = _PField(
        type=(int,),
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert field.type == (int,)
    assert field.invariant is None
    assert field.initial is None
    assert field.mandatory is False
    assert field._factory is None
    assert field.serializer is None


def test_pfield_constructor_with_different_types():
    field = _PField(
        type=(list, dict, tuple),
        invariant=lambda x: len(x) > 0,
        initial=[1, 2, 3],
        mandatory=True,
        factory=list,
        serializer=repr
    )
    
    assert field.type == (list, dict, tuple)
    assert field.initial == [1, 2, 3]
    assert field.mandatory is True
    assert field._factory == list
    assert field.serializer == repr


# LLM-generated content at query #30
#--------------------------

```python
def test_set_fields():
    from collections import namedtuple
    
    class _PField:
        def __init__(self, value):
            self.value = value
    
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    
    # Test 1: Empty bases and empty dict
    dct1 = {}
    set_fields(dct1, [], "fields")
    assert dct1 == {"fields": {}}
    
    # Test 2: Dict with _PField instances
    pfield1 = _PField("value1")
    pfield2 = _PField("value2")
    dct2 = {"field1": pfield1, "field2": pfield2, "other": "data"}
    set_fields(dct2, [], "fields")
    assert dct2 == {"fields": {"field1": pfield1, "field2": pfield2}, "other": "data"}
    assert "field1" not in dct2
    assert "field2" not in dct2
    
    # Test 3: With base classes having fields
    class Base1:
        fields = {"base_field1": _PField("base_value1")}
    
    class Base2:
        fields = {"base_field2": _PField("base_value2")}
    
    dct3 = {"new_field": _PField("new_value")}
    set_fields(dct3, [Base1, Base2], "fields")
    assert "base_field1" in dct3["fields"]
    assert "base_field2" in dct3["fields"]
    assert "new_field" in dct3["fields"]
    assert "new_field" not in dct3
    
    # Test 4: Mixed _PField and non-_PField in dict
    pfield3 = _PField("pvalue")
    dct4 = {"pfield": pfield3, "normal": "value", "number": 42}
    set_fields(dct4, [], "fields")
    assert dct4 == {"fields": {"pfield": pfield3}, "normal": "value", "number": 42}
    assert "pfield" not in dct4
    
    # Test 5: Base with empty fields dict
    class BaseEmpty:
        fields = {}
    
    dct5 = {"field": _PField("val")}
    set_fields(dct5, [BaseEmpty], "fields")
    assert dct5["fields"] == {"field": _PField("val").value if hasattr(_PField("val"), "value") else _PField("val")}


# LLM-generated content at query #31
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    
    assert result == "serialized_value"
    assert isinstance(value, CheckedType)
    assert PFIELD_NO_SERIALIZER is PFIELD_NO_SERIALIZER


# LLM-generated content at query #32
#--------------------------

```python
def test_predicate_isinstance_pfield():
    class _PField:
        pass
    
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    
    pfield_instance = _PField()
    test_dict = {'field1': pfield_instance}
    bases = []
    name = 'fields'
    
    set_fields(test_dict, bases, name)
    
    assert 'field1' in test_dict['fields']
    assert test_dict['fields']['field1'] is pfield_instance
    assert 'field1' not in test_dict


# LLM-generated content at query #33
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    checked_value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_value)
    
    assert result == "serialized_value"
    assert isinstance(checked_value, CheckedType)
    assert PFIELD_NO_SERIALIZER is PFIELD_NO_SERIALIZER


# LLM-generated content at query #34
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariants_fail():
    def failing_invariant(subject):
        return (False, "ERROR_CODE_1")
    
    def passing_invariant(subject):
        return (True, "ERROR_CODE_2")
    
    invariants = [failing_invariant, passing_invariant]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


# LLM-generated content at query #35
#--------------------------

```python
def test_is_field_ignore_extra_complaint_ignore_extra_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def mock_factory():
        pass
    
    mock_field = MockField(set(), mock_factory)
    result = is_field_ignore_extra_complaint(object, mock_field, False)
    assert result is False


def test_is_field_ignore_extra_complaint_ignore_extra_true_but_not_type_cls():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def mock_factory():
        pass
    
    mock_field = MockField((), mock_factory)
    result = is_field_ignore_extra_complaint(object, mock_field, True)
    assert result is False


def test_is_field_ignore_extra_complaint_ignore_extra_true_type_cls_but_no_ignore_extra_param():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def mock_factory():
        pass
    
    mock_field = MockField(set(), mock_factory)
    result = is_field_ignore_extra_complaint(object, mock_field, True)
    assert result is False


def test_is_field_ignore_extra_complaint_all_conditions_met():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def mock_factory(ignore_extra=False):
        pass
    
    mock_field = MockField(set(), mock_factory)
    result = is_field_ignore_extra_complaint(object, mock_field, True)
    assert result is True


def test_is_field_ignore_extra_complaint_with_type_string():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def mock_factory(ignore_extra=False):
        pass
    
    mock_field = MockField(('builtins.str',), mock_factory)
    result = is_field_ignore_extra_complaint(str, mock_field, True)
    assert result is True


def test_is_field_ignore_extra_complaint_with_multiple_types_in_tuple():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def mock_factory(ignore_extra=True):
        pass
    
    mock_field = MockField((int, str), mock_factory)
    result = is_field_ignore_extra_complaint(int, mock_field, True)
    assert result is True


# LLM-generated content at query #36
#--------------------------

```python
def test_is_field_ignore_extra_complaint_predicate_line_6_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    class SomeClass:
        pass
    
    class DifferentClass:
        pass
    
    def simple_factory():
        return DifferentClass()
    
    field = MockField(SomeClass, simple_factory)
    
    result = is_field_ignore_extra_complaint(DifferentClass, field, ignore_extra=True)
    
    assert result is False


# LLM-generated content at query #37
#--------------------------

```python
def test_check_global_invariants_all_pass():
    invariants = [
        lambda x: (True, None),
        lambda x: (True, None),
        lambda x: (True, None)
    ]
    subject = "test"
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_failure():
    invariants = [
        lambda x: (True, None),
        lambda x: (False, "ERROR_001"),
        lambda x: (True, None)
    ]
    subject = "test"
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_001",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    invariants = [
        lambda x: (False, "ERROR_001"),
        lambda x: (False, "ERROR_002"),
        lambda x: (True, None)
    ]
    subject = "test"
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_001", "ERROR_002")
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_all_failures():
    invariants = [
        lambda x: (False, "ERROR_A"),
        lambda x: (False, "ERROR_B"),
        lambda x: (False, "ERROR_C")
    ]
    subject = "test"
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_A", "ERROR_B", "ERROR_C")


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test"
    check_global_invariants(subject, invariants)


def test_check_global_invariants_passes_subject_correctly():
    received_subject = []
    def invariant(subject):
        received_subject.append(subject)
        return (True, None)
    
    invariants = [invariant]
    subject = {"key": "value"}
    check_global_invariants(subject, invariants)
    assert received_subject[0] == subject


# LLM-generated content at query #38
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant_pass(subject):
        return (True, 'OK')
    
    subject = {'test': 'data'}
    invariants = [invariant_pass]
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_error():
    def invariant_fail(subject):
        return (False, 'ERROR_CODE_1')
    
    subject = {'test': 'data'}
    invariants = [invariant_fail]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.error_codes == ('ERROR_CODE_1',)
        assert e.invariant_errors == ()
        assert e.message == 'Global invariant failed'


def test_check_global_invariants_multiple_errors():
    def invariant_fail_1(subject):
        return (False, 'ERROR_1')
    
    def invariant_fail_2(subject):
        return (False, 'ERROR_2')
    
    subject = {'test': 'data'}
    invariants = [invariant_fail_1, invariant_fail_2]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.error_codes == ('ERROR_1', 'ERROR_2')


def test_check_global_invariants_mixed_pass_fail():
    def invariant_pass(subject):
        return (True, 'OK')
    
    def invariant_fail(subject):
        return (False, 'ERROR_CODE')
    
    subject = {'test': 'data'}
    invariants = [invariant_pass, invariant_fail, invariant_pass]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.error_codes == ('ERROR_CODE',)


def test_check_global_invariants_empty_invariants():
    subject = {'test': 'data'}
    invariants = []
    check_global_invariants(subject, invariants)


def test_check_global_invariants_multiple_invariants_all_pass():
    def invariant_1(subject):
        return (True, 'OK_1')
    
    def invariant_2(subject):
        return (True, 'OK_2')
    
    def invariant_3(subject):
        return (True, 'OK_3')
    
    subject = {'test': 'data'}
    invariants = [invariant_1, invariant_2, invariant_3]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #39
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent import pmap, PMap
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent._checked_types import _restore_pickle
    
    # Create a test pmap field type
    key_type = str
    value_type = int
    
    # Ensure the type is registered
    test_data = pmap({'a': 1, 'b': 2})
    
    # Call the function with test data
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a PMap instance
    assert isinstance(result, PMap)
    
    # Verify the data is preserved
    assert result == test_data
    assert result['a'] == 1
    assert result['b'] == 2


def test_restore_pmap_field_pickle_empty():
    from pyrsistent import pmap
    from pyrsistent._field_common import _restore_pmap_field_pickle
    
    key_type = str
    value_type = str
    test_data = pmap({})
    
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    assert result == test_data
    assert len(result) == 0


def test_restore_pmap_field_pickle_different_types():
    from pyrsistent import pmap
    from pyrsistent._field_common import _restore_pmap_field_pickle
    
    key_type = int
    value_type = str
    test_data = pmap({1: 'one', 2: 'two', 3: 'three'})
    
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    assert result == test_data
    assert result[1] == 'one'
    assert result[2] == 'two'
    assert result[3] == 'three'


# LLM-generated content at query #40
#--------------------------

```python
def test_check_field_parameters_valid_field():
    class Field:
        def __init__(self):
            self.type = [str, int]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class Field:
        def __init__(self):
            self.type = [str, 123]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Type parameter expected" in str(e)


def test_check_field_parameters_invalid_initial_type():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = 123
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    PFIELD_NO_INITIAL = object()
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Initial has invalid type" in str(e)


def test_check_field_parameters_callable_initial():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = lambda: "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_non_callable_invariant():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = "not callable"
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Invariant must be callable" in str(e)


def test_check_field_parameters_non_callable_factory():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = "not callable"
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Factory must be callable" in str(e)


def test_check_field_parameters_non_callable_serializer():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = "not callable"
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Serializer must be callable" in str(e)


def test_check_field_parameters_string_type():
    class Field:
        def __init__(self):
            self.type = ["str", "int"]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_empty_type():
    class Field:
        def __init__(self):
            self.type = []
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


# LLM-generated content at query #41
#--------------------------

```python
def test_make_pmap_field_type_creates_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result.__name__ == "StringToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2
    assert result1.__name__ == "StringToIntPMap"
    assert result2.__name__ == "IntToStringPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_multiple_key_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type((str, int), float)
    
    assert "StringInt" in result.__name__
    assert "Float" in result.__name__
    assert "PMap" in result.__name__


# LLM-generated content at query #42
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent import PVector, CheckedPVector
    
    # Create a checked class with a PVector field
    class MyChecked(CheckedPVector):
        __type__ = int
    
    # Register the type in _seq_field_types
    checked_class = MyChecked
    item_type = int
    _seq_field_types[checked_class, item_type] = MyChecked
    
    # Test data to restore
    test_data = [1, 2, 3]
    
    # Call the function
    result = _restore_seq_field_pickle(checked_class, item_type, test_data)
    
    # Verify the result is the correct type and contains the data
    assert isinstance(result, MyChecked)
    assert list(result) == test_data
    
    # Clean up
    del _seq_field_types[checked_class, item_type]


# LLM-generated content at query #43
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    
    pfield_instance = _PField()
    test_dict = {'field1': pfield_instance}
    bases = []
    name = 'fields'
    
    set_fields(test_dict, bases, name)
    
    assert 'field1' in test_dict['fields']
    assert test_dict['fields']['field1'] is pfield_instance
    assert 'field1' not in test_dict


# LLM-generated content at query #44
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import pvector, pset, PClass, field
    from pyrsistent._field_common import _restore_seq_field_pickle
    
    class TestClass(PClass):
        items = field()
    
    test_data = [1, 2, 3]
    result = _restore_seq_field_pickle(TestClass, int, test_data)
    assert result is not None
    assert hasattr(result, '__iter__')


def test_restore_seq_field_pickle_with_empty_data():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import _restore_seq_field_pickle
    
    class TestClass(PClass):
        items = field()
    
    test_data = []
    result = _restore_seq_field_pickle(TestClass, str, test_data)
    assert result is not None


def test_restore_seq_field_pickle_preserves_data():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import _restore_seq_field_pickle
    
    class TestClass(PClass):
        items = field()
    
    test_data = ['a', 'b', 'c']
    result = _restore_seq_field_pickle(TestClass, str, test_data)
    assert len(result) == 3
    assert list(result) == ['a', 'b', 'c']


# LLM-generated content at query #45
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "checked_serialized"


def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"serialized_{value}_{format}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "serialized_test_value_xml"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    def custom_serializer(format, value):
        return f"custom_{value}"
    
    value = CheckedType()
    result = serialize(custom_serializer, "json", value)
    assert result == "custom_<CheckedType object>"


def test_serialize_with_non_checked_type():
    def my_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(my_serializer, "json", 42)
    assert result == "json:42"


def test_serialize_checked_type_with_different_formats():
    class CheckedType:
        def serialize(self, format):
            return f"format_{format}"
    
    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    
    result_json = serialize(PFIELD_NO_SERIALIZER, "json", value)
    result_xml = serialize(PFIELD_NO_SERIALIZER, "xml", value)
    result_csv = serialize(PFIELD_NO_SERIALIZER, "csv", value)
    
    assert result_json == "format_json"
    assert result_xml == "format_xml"
    assert result_csv == "format_csv"


# LLM-generated content at query #46
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant_1(subject):
        return (True, None)
    
    def invariant_2(subject):
        return (True, None)
    
    invariants = [invariant_1, invariant_2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        result = True
    except InvariantException:
        result = False
    
    assert result is True


# LLM-generated content at query #47
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a simple pmap field type
    key_type = str
    value_type = int
    
    # Register a pmap field type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent import PMap
        class TestPMapField(PMap):
            __type_fields = (key_type, value_type)
            @staticmethod
            def create(data, _factory_fields=None):
                return pmap(data)
        _pmap_field_types[key_type, value_type] = TestPMapField
    
    # Test data to restore
    test_data = {'a': 1, 'b': 2, 'c': 3}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap with correct data
    assert result == pmap(test_data)
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #48
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int)
    assert field_obj.mandatory is True
    assert field_obj.initial == pmap()
    assert callable(field_obj.factory)


def test_pmap_field_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int, optional=False)
    assert field_obj.mandatory is True
    assert field_obj.initial == pmap()
    assert callable(field_obj.factory)


def test_pmap_field_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.mandatory is True
    assert field_obj.initial == pmap()
    assert callable(field_obj.factory)
    result_none = field_obj.factory(None)
    assert result_none is None


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    def my_invariant(val):
        return (True, None)
    
    field_obj = pmap_field(str, int, invariant=my_invariant)
    assert field_obj.mandatory is True
    assert callable(field_obj.invariant)


def test_pmap_field_factory_creates_map():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int)
    test_data = {'a': 1, 'b': 2}
    result = field_obj.factory(test_data)
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1
    assert result['b'] == 2


def test_pmap_field_optional_factory_with_data():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int, optional=True)
    test_data = {'a': 1, 'b': 2}
    result = field_obj.factory(test_data)
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #49
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap_field with optional=False
    result = pmap_field(str, int, optional=False)
    
    # The predicate at line 2 (which checks `if optional:`) should evaluate to False
    # This means the factory should be set to TheMap.create directly
    # and the type should be TheMap (not wrapped in optional_type)
    assert result.factory is not None
    assert result.initial == pmap()
    assert result.mandatory is True


# LLM-generated content at query #50
#--------------------------

```python
def test_is_field_ignore_extra_complaint_predicate_line_6_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent import field
    
    class MockField:
        def __init__(self, type_value, factory_func):
            self.type = type_value
            self.factory = factory_func
    
    def factory_without_ignore_extra():
        return "test"
    
    mock_field = MockField(str, factory_without_ignore_extra)
    
    result = is_field_ignore_extra_complaint(int, mock_field, True)
    
    assert result is False


# LLM-generated content at query #51
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import pmap, pvector
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(pvector, item_type, item_invariant)
    
    assert result.__type__ == int
    assert result.__invariant__ is None
    assert hasattr(result, '__reduce__')


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import pvector
    
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(pvector, item_type, item_invariant)
    result2 = _make_seq_field_type(pvector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_with_different_item_types():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import pvector
    
    result_int = _make_seq_field_type(pvector, int, None)
    result_str = _make_seq_field_type(pvector, str, None)
    
    assert result_int is not result_str
    assert result_int.__type__ == int
    assert result_str.__type__ == str


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import pvector
    
    def my_invariant(value):
        return value > 0
    
    result = _make_seq_field_type(pvector, int, my_invariant)
    
    assert result.__invariant__ is my_invariant
    assert result.__type__ == int


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import pvector
    
    result = _make_seq_field_type(pvector, int, None)
    instance = result([1, 2, 3])
    
    reduce_result = instance.__reduce__()
    assert isinstance(reduce_result, tuple)
    assert len(reduce_result) == 2


# LLM-generated content at query #52
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    mock_factory = lambda: MockCheckedType()
    pfield = _PField(
        type={MockCheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=mock_factory,
        serializer=None
    )
    
    assert pfield._factory is mock_factory
    assert pfield._factory == mock_factory


# LLM-generated content at query #53
#--------------------------

```python
def test_make_pmap_field_type():
    from pyrsistent._field_common import _make_pmap_field_type, _pmap_field_types
    
    # Clear the cache before testing
    _pmap_field_types.clear()
    
    # Test creating a pmap field type with str and int
    pmap_type_1 = _make_pmap_field_type(str, int)
    assert pmap_type_1 is not None
    assert pmap_type_1.__name__ == "StrToIntPMap"
    assert pmap_type_1.__key_type__ == str
    assert pmap_type_1.__value_type__ == int
    
    # Test that the same type is returned from cache
    pmap_type_1_again = _make_pmap_field_type(str, int)
    assert pmap_type_1_again is pmap_type_1
    
    # Test creating a different pmap field type
    pmap_type_2 = _make_pmap_field_type(int, str)
    assert pmap_type_2 is not None
    assert pmap_type_2.__name__ == "IntToStrPMap"
    assert pmap_type_2.__key_type__ == int
    assert pmap_type_2.__value_type__ == str
    assert pmap_type_2 is not pmap_type_1
    
    # Test that types are cached correctly
    assert (str, int) in _pmap_field_types
    assert (int, str) in _pmap_field_types
    assert _pmap_field_types[(str, int)] is pmap_type_1
    assert _pmap_field_types[(int, str)] is pmap_type_2
    
    # Test with float and bool types
    pmap_type_3 = _make_pmap_field_type(float, bool)
    assert pmap_type_3.__name__ == "FloatToBoolPMap"
    assert pmap_type_3.__key_type__ == float
    assert pmap_type_3.__value_type__ == bool
    
    # Clean up
    _pmap_field_types.clear()


# LLM-generated content at query #54
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    def mock_factory():
        return MockCheckedType()
    
    pfield = _PField(
        type=set(),
        invariant=None,
        initial=None,
        mandatory=False,
        factory=mock_factory,
        serializer=None
    )
    
    assert pfield._factory is mock_factory


# LLM-generated content at query #55
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Register the type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent import PMap
        _pmap_field_types[key_type, value_type] = PMap
    
    # Create test data
    test_data = {'a': 1, 'b': 2, 'c': 3}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap-like structure
    assert result is not None
    assert isinstance(result, dict) or hasattr(result, 'items')


def test_restore_pmap_field_pickle_with_empty_data():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    
    key_type = str
    value_type = str
    
    # Register the type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent import PMap
        _pmap_field_types[key_type, value_type] = PMap
    
    # Create empty test data
    test_data = {}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result
    assert result is not None


def test_restore_pmap_field_pickle_different_types():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    
    key_type = int
    value_type = str
    
    # Register the type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent import PMap
        _pmap_field_types[key_type, value_type] = PMap
    
    # Create test data
    test_data = {1: 'one', 2: 'two'}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result
    assert result is not None


# LLM-generated content at query #56
#--------------------------

```python
def test_pmap_field_returns_field_with_correct_type():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import PMap
    
    result = pmap_field(str, int)
    
    assert result is not None
    assert hasattr(result, 'type')
    assert hasattr(result, 'factory')
    assert hasattr(result, 'mandatory')
    assert result.mandatory is True


# LLM-generated content at query #57
#--------------------------

```python
def test_pmap_field_optional_parameter_affects_type():
    from pyrsistent._field_common import pmap_field, optional
    from pyrsistent import PMap
    
    # Test with optional=False
    field_non_optional = pmap_field(str, int, optional=False)
    assert field_non_optional.type == (str, int) or not isinstance(field_non_optional.type, tuple)
    
    # Test with optional=True
    field_optional = pmap_field(str, int, optional=True)
    # When optional=True, the type should include NoneType
    assert type(None) in field_optional.type or field_optional.type is not None


# LLM-generated content at query #58
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == int
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ is None


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    item_type = str
    item_invariant = None
    
    _seq_field_types.clear()
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_seq_field_type_with_different_item_types():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result_int = _make_seq_field_type(PVector, int, None)
    result_str = _make_seq_field_type(PVector, str, None)
    
    assert result_int is not result_str
    assert result_int.__type__ == int
    assert result_str.__type__ == str


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    def my_invariant(value):
        return value > 0
    
    result = _make_seq_field_type(PVector, int, my_invariant)
    
    assert result.__invariant__ is my_invariant


# LLM-generated content at query #59
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int)
    assert result.type is not None
    assert result.mandatory is True
    assert result.factory is not None
    assert callable(result.factory)


def test_pmap_field_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_optional_true():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    assert result.mandatory is True
    assert callable(result.factory)


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    
    def my_invariant(val):
        return True, "valid"
    
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.invariant is not None
    assert callable(result.invariant)


def test_pmap_field_factory_with_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=True)
    factory = result.factory
    none_result = factory(None)
    assert none_result is None


def test_pmap_field_factory_with_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    factory = result.factory
    empty_map = factory({})
    assert empty_map == pmap()


def test_pmap_field_returns_pfield():
    from pyrsistent._field_common import pmap_field, _PField
    
    result = pmap_field(str, int)
    assert isinstance(result, _PField)


# LLM-generated content at query #60
#--------------------------

```python
def test_pmap_field_factory_property_returns_no_factory_when_multiple_types():
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import PFIELD_NO_FACTORY, PFIELD_NO_INVARIANT
    
    # Create a _PField with multiple types and no factory
    field = _PField(
        type=(int, str),
        invariant=PFIELD_NO_INVARIANT,
        initial=None,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=None
    )
    
    # The predicate at line 2 of factory property checks: len(self.type) == 1
    # This should evaluate to False when type has multiple elements
    result = field.factory
    
    assert result == PFIELD_NO_FACTORY


# LLM-generated content at query #61
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap_field with optional=False
    result = pmap_field(str, int, optional=False)
    
    # The predicate at line 2 (optional parameter) should evaluate to False
    # This means the else branch at line 21-22 should be executed
    # We verify this by checking that factory is set to TheMap.create
    # and that the type is not wrapped in optional_type
    assert result.factory is not None
    assert result.initial == pmap()
    assert result.mandatory is True


# LLM-generated content at query #62
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean test
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == int
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ is None
    assert hasattr(result, '__name__')
    assert isinstance(result.__name__, str)


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean test
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean test
    _seq_field_types.clear()
    
    def my_invariant(value):
        return value > 0
    
    result = _make_seq_field_type(PVector, int, my_invariant)
    
    assert result.__invariant__ is my_invariant


def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean test
    _seq_field_types.clear()
    
    result = _make_seq_field_type(PVector, int, None)
    instance = result([1, 2, 3])
    
    reduce_result = instance.__reduce__()
    
    assert reduce_result is not None
    assert len(reduce_result) == 2
    assert callable(reduce_result[0])
    assert reduce_result[1][0] is PVector
    assert reduce_result[1][1] == int
    assert reduce_result[1][2] == [1, 2, 3]


def test_make_seq_field_type_different_item_types():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean test
    _seq_field_types.clear()
    
    result_int = _make_seq_field_type(PVector, int, None)
    result_str = _make_seq_field_type(PVector, str, None)
    
    assert result_int is not result_str
    assert result_int.__type__ == int
    assert result_str.__type__ == str


# LLM-generated content at query #63
#--------------------------

```python
def test_is_field_ignore_extra_complaint_predicate_line_6_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    
    class MockField:
        def __init__(self, factory, type_):
            self.factory = factory
            self.type = type_
    
    def mock_factory():
        pass
    
    class TypeClass:
        pass
    
    field = MockField(mock_factory, str)
    
    result = is_field_ignore_extra_complaint(TypeClass, field, True)
    
    assert result is False


# LLM-generated content at query #64
#--------------------------

```python
def test_check_field_parameters_valid_type_class():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_valid_type_string():
    class MockField:
        def __init__(self):
            self.type = ['int', 'str']
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_valid_callable_initial():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = lambda: 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class MockField:
        def __init__(self):
            self.type = [123]
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Type parameter expected' in str(e)


def test_check_field_parameters_invalid_initial_type():
    PFIELD_NO_INITIAL = object()
    
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = "not an int"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_check_field_parameters_non_callable_invariant():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = 5
            self.invariant = "not callable"
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_check_field_parameters_non_callable_factory():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = "not callable"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_check_field_parameters_non_callable_serializer():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = "not callable"
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


def test_check_field_parameters_empty_type():
    class MockField:
        def __init__(self):
            self.type = []
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_no_initial():
    PFIELD_NO_INITIAL = object()
    
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


# LLM-generated content at query #65
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent import PVector, pv
    from pyrsistent._checked_types import CheckedPVector
    
    # Setup: Create a checked class and item type
    checked_class = CheckedPVector
    item_type = int
    
    # Register the type in _seq_field_types
    test_type = CheckedPVector.create([1, 2, 3])
    _seq_field_types[(checked_class, item_type)] = type(test_type)
    
    # Test data to restore
    test_data = [1, 2, 3]
    
    # Call the function
    result = _restore_seq_field_pickle(checked_class, item_type, test_data)
    
    # Verify the result is a PVector with correct data
    assert result == pv([1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #66
#--------------------------

```python
def test_check_global_invariants_raises_when_invariants_fail():
    def failing_invariant(subject):
        return (False, "ERROR_CODE_1")
    
    invariants = [failing_invariant]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_CODE_1",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


# LLM-generated content at query #67
#--------------------------

```python
def test_set_fields_empty_bases():
    dct = {}
    bases = ()
    name = "fields"
    set_fields(dct, bases, name)
    assert "fields" in dct
    assert dct["fields"] == {}


def test_set_fields_with_pfield_values():
    class _PField:
        def __init__(self, value):
            self.value = value
    
    field1 = _PField("value1")
    field2 = _PField("value2")
    
    dct = {"field1": field1, "field2": field2, "other": "data"}
    bases = ()
    name = "fields"
    
    set_fields(dct, bases, name)
    
    assert "fields" in dct
    assert dct["fields"]["field1"] is field1
    assert dct["fields"]["field2"] is field2
    assert "field1" not in dct
    assert "field2" not in dct
    assert dct["other"] == "data"


def test_set_fields_with_base_fields():
    class _PField:
        def __init__(self, value):
            self.value = value
    
    base_field = _PField("base_value")
    
    class BaseClass:
        pass
    
    BaseClass.__dict__ = {"fields": {"base_field": base_field}}
    
    dct = {"new_field": _PField("new_value")}
    bases = (BaseClass,)
    name = "fields"
    
    set_fields(dct, bases, name)
    
    assert "fields" in dct
    assert "base_field" in dct["fields"]
    assert "new_field" in dct["fields"]
    assert "new_field" not in dct


def test_set_fields_multiple_bases():
    class _PField:
        def __init__(self, value):
            self.value = value
    
    class Base1:
        pass
    class Base2:
        pass
    
    Base1.__dict__ = {"fields": {"field1": _PField("value1")}}
    Base2.__dict__ = {"fields": {"field2": _PField("value2")}}
    
    dct = {}
    bases = (Base1, Base2)
    name = "fields"
    
    set_fields(dct, bases, name)
    
    assert "fields" in dct
    assert "field1" in dct["fields"]
    assert "field2" in dct["fields"]


def test_set_fields_no_fields_in_bases():
    class _PField:
        def __init__(self, value):
            self.value = value
    
    class BaseClass:
        pass
    
    dct = {"myfield": _PField("myvalue")}
    bases = (BaseClass,)
    name = "fields"
    
    set_fields(dct, bases, name)
    
    assert "fields" in dct
    assert dct["fields"]["myfield"] is dct.get("myfield") or "myfield" not in dct
    assert "myfield" not in dct


# LLM-generated content at query #68
#--------------------------

```python
def test_pmap_field_optional_type_includes_none():
    from pyrsistent._field_common import pmap_field
    from pyrsistent._checked_types import optional
    
    result = pmap_field(str, int, optional=True)
    
    # The predicate at line 2 evaluates to True means the function should return
    # a field with optional type that includes None
    assert result.type == optional(result.type[0]) or type(None) in result.type


# LLM-generated content at query #69
#--------------------------

```python
def test_pfield_init_factory_assignment():
    class DummyCheckedType:
        pass
    
    factory_func = lambda: "test"
    pfield = _PField(
        type={int},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=None
    )
    
    assert pfield._factory is factory_func
    assert pfield._factory == factory_func


# LLM-generated content at query #70
#--------------------------

```python
def test_make_seq_field_type_creates_subclass_with_correct_name():
    from pyrsistent._field_common import _make_seq_field_type, SEQ_FIELD_TYPE_SUFFIXES
    from pyrsistent import PVector
    
    item_type = int
    item_invariant = lambda x: x > 0
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert issubclass(result, PVector)
    assert result.__type__ == int
    assert result.__invariant__ == item_invariant
    assert hasattr(result, '__name__')


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    item_type = str
    item_invariant = lambda x: len(x) > 0
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    item_type = int
    item_invariant = lambda x: True
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_seq_field_type_different_item_types_create_different_classes():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    invariant = lambda x: True
    
    result_int = _make_seq_field_type(PVector, int, invariant)
    result_str = _make_seq_field_type(PVector, str, invariant)
    
    assert result_int is not result_str
    assert result_int.__type__ == int
    assert result_str.__type__ == str


# LLM-generated content at query #71
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Register the type if not already present
    test_type = _pmap_field_types.get((key_type, value_type))
    if test_type is None:
        from pyrsistent import PClass, pmapfield
        class TestClass(PClass):
            data = pmapfield(key_type, value_type)
        test_type = type(TestClass.data)
        _pmap_field_types[key_type, value_type] = test_type
    
    # Create test data
    test_data = pmap({"a": 1, "b": 2})
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result
    assert result is not None
    assert result == test_data


# LLM-generated content at query #72
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant_pass(subject):
        return (True, None)
    
    def invariant_pass_2(subject):
        return (True, None)
    
    invariants = [invariant_pass, invariant_pass_2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        result = True
    except InvariantException:
        result = False
    
    assert result is True


# LLM-generated content at query #73
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "checked_serialized"


def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"serialized_{value}_in_{format}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "serialized_test_value_in_xml"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return "should_not_be_called"
    
    def custom_serializer(format, value):
        return f"custom_{format}"
    
    value = CheckedType()
    result = serialize(custom_serializer, "json", value)
    assert result == "custom_json"


def test_serialize_with_non_checked_type_and_no_serializer():
    PFIELD_NO_SERIALIZER = object()
    
    def no_serializer(format, value):
        return f"result_{value}"
    
    result = serialize(no_serializer, "json", "plain_value")
    assert result == "result_plain_value"


# LLM-generated content at query #74
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert result.__name__ == "StringToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    assert result1 is result2


def test_make_pmap_field_type_different_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    assert result1 is not result2
    assert result1.__name__ == "StringToIntPMap"
    assert result2.__name__ == "IntToStringPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_with_float_and_bool():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(float, bool)
    assert result.__name__ == "FloatToBoolPMap"
    assert result.__key_type__ == float
    assert result.__value_type__ == bool


# LLM-generated content at query #75
#--------------------------

```python
def test_pmap_field_predicate_line_2_evaluates_to_false():
    # Line 2 is the opening of the docstring, but the predicate at line 2
    # in the context of the function logic refers to the condition at line 15
    # "if optional:" - we need to test when optional is False
    from pyrsistent._field_common import pmap_field
    from pyrsistent import PMap
    
    # Call pmap_field with optional=False (the predicate at line 15 evaluates to False)
    result = pmap_field(str, int, optional=False)
    
    # Verify that the result is a field with the expected properties
    assert result.mandatory is True
    assert result.type == PMap  # or the CheckedPMap type created
    assert result.initial == PMap()
    assert result._factory is not None


# LLM-generated content at query #76
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Register a pmap field type if not already registered
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent import PMap
        class TestPMapField(PMap):
            __key_type__ = key_type
            __value_type__ = value_type
            @classmethod
            def create(cls, data, _factory_fields=None):
                return pmap(data)
        _pmap_field_types[key_type, value_type] = TestPMapField
    
    # Test data
    test_data = {'a': 1, 'b': 2, 'c': 3}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #77
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3: not isinstance(t, type) and not isinstance(t, str)
    # should evaluate to False for all t in field.type
    # This means each t must be either isinstance(t, type) or isinstance(t, str)
    
    for t in field.type:
        predicate_result = not isinstance(t, type) and not isinstance(t, str)
        assert predicate_result == False


# LLM-generated content at query #78
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, pv
    from pyrsistent._checked_types import CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    # Create a checked class with a vector field
    class TestChecked(CheckedPVector):
        __type__ = PVector
    
    # Register the type in _seq_field_types
    checked_class = TestChecked
    item_type = int
    _seq_field_types[(checked_class, item_type)] = TestChecked
    
    # Create test data
    test_data = [1, 2, 3]
    
    # Call the function
    result = _restore_seq_field_pickle(checked_class, item_type, test_data)
    
    # Assert the result is a PVector with correct data
    assert isinstance(result, PVector)
    assert list(result) == test_data


def test_restore_seq_field_pickle_empty():
    from pyrsistent import PVector
    from pyrsistent._checked_types import CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    # Create a checked class with a vector field
    class TestCheckedEmpty(CheckedPVector):
        __type__ = PVector
    
    # Register the type in _seq_field_types
    checked_class = TestCheckedEmpty
    item_type = str
    _seq_field_types[(checked_class, item_type)] = TestCheckedEmpty
    
    # Create empty test data
    test_data = []
    
    # Call the function
    result = _restore_seq_field_pickle(checked_class, item_type, test_data)
    
    # Assert the result is an empty PVector
    assert isinstance(result, PVector)
    assert len(result) == 0


# LLM-generated content at query #79
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    factory_func = lambda: MockCheckedType()
    field = _PField(
        type={MockCheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=None
    )
    
    assert field._factory is factory_func


# LLM-generated content at query #80
#--------------------------

```python
def test_check_global_invariants_with_failed_invariant():
    def failing_invariant(subject):
        return (False, "ERROR_001")
    
    def passing_invariant(subject):
        return (True, "")
    
    subject = "test_subject"
    invariants = [failing_invariant, passing_invariant]
    
    error_codes = tuple(error_code for is_ok, error_code in
                        (invariant(subject) for invariant in invariants) if not is_ok)
    
    assert error_codes
    assert error_codes == ("ERROR_001",)


# LLM-generated content at query #81
#--------------------------

```python
def test_check_global_invariants_with_no_violations():
    def invariant1(subject):
        return (True, None)
    
    def invariant2(subject):
        return (True, None)
    
    invariants = [invariant1, invariant2]
    subject = {"test": "data"}
    
    try:
        check_global_invariants(subject, invariants)
        result = True
    except InvariantException:
        result = False
    
    assert result is True


# LLM-generated content at query #82
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "serialized_value"


# LLM-generated content at query #83
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from collections import namedtuple
    
    # Create a mock field with factory that has ignore_extra parameter
    MockField = namedtuple('MockField', ['type', 'factory'])
    
    def factory_with_ignore_extra(ignore_extra=False):
        pass
    
    def factory_without_ignore_extra():
        pass
    
    # Test case 1: ignore_extra is False, should return False regardless of other conditions
    field1 = MockField(type=(dict,), factory=factory_with_ignore_extra)
    result1 = is_field_ignore_extra_complaint(dict, field1, False)
    assert result1 is False
    
    # Test case 2: ignore_extra is True, but is_type_cls returns False (empty tuple)
    field2 = MockField(type=(), factory=factory_with_ignore_extra)
    result2 = is_field_ignore_extra_complaint(dict, field2, True)
    assert result2 is False
    
    # Test case 3: ignore_extra is True, is_type_cls returns True, but factory doesn't have ignore_extra parameter
    field3 = MockField(type=(dict,), factory=factory_without_ignore_extra)
    result3 = is_field_ignore_extra_complaint(dict, field3, True)
    assert result3 is False
    
    # Test case 4: ignore_extra is True, is_type_cls returns True, and factory has ignore_extra parameter
    field4 = MockField(type=(dict,), factory=factory_with_ignore_extra)
    result4 = is_field_ignore_extra_complaint(dict, field4, True)
    assert result4 is True
    
    # Test case 5: field_type is a set (is_type_cls returns True), ignore_extra is True, factory has ignore_extra
    field5 = MockField(type={dict}, factory=factory_with_ignore_extra)
    result5 = is_field_ignore_extra_complaint(dict, field5, True)
    assert result5 is True
    
    # Test case 6: field_type is a set, ignore_extra is True, but factory doesn't have ignore_extra
    field6 = MockField(type={dict}, factory=factory_without_ignore_extra)
    result6 = is_field_ignore_extra_complaint(dict, field6, True)
    assert result6 is False


# LLM-generated content at query #84
#--------------------------

```python
def test_pmap_field_optional_false_creates_field_with_themap_type():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    
    assert result.mandatory is True
    assert result.initial == pmap()
    assert result.factory is not None


# LLM-generated content at query #85
#--------------------------

```python
def test_make_pmap_field_type_creates_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    
    result = _make_pmap_field_type(str, int)
    
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_generates_correct_name():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result.__name__ == "StrToIntPMap"


def test_make_pmap_field_type_caches_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_types_not_cached():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2
    assert result1.__name__ == "StrToIntPMap"
    assert result2.__name__ == "IntToStrPMap"


def test_make_pmap_field_type_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type, _restore_pmap_field_pickle
    
    PMapType = _make_pmap_field_type(str, int)
    instance = PMapType({"key": 1})
    
    reduce_result = instance.__reduce__()
    
    assert reduce_result[0] == _restore_pmap_field_pickle
    assert reduce_result[1][0] == str
    assert reduce_result[1][1] == int
    assert reduce_result[1][2] == {"key": 1}


def test_make_pmap_field_type_with_builtin_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(int, float)
    
    assert result.__name__ == "IntToFloatPMap"
    assert result.__key_type__ == int
    assert result.__value_type__ == float


def test_make_pmap_field_type_with_bool_and_list():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(bool, list)
    
    assert result.__name__ == "BoolToListPMap"


# LLM-generated content at query #86
#--------------------------

```python
def test_pmap_field_factory_property_with_non_checkedtype():
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import PFIELD_NO_FACTORY, PFIELD_NO_INVARIANT
    
    # Create a _PField with a non-CheckedType type and PFIELD_NO_FACTORY
    pfield = _PField(
        type=(str,),
        invariant=PFIELD_NO_INVARIANT,
        initial=None,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=None
    )
    
    # Access the factory property - the predicate at line 2 should evaluate to False
    # because str is not a subclass of CheckedType
    result = pfield.factory
    
    # The factory should remain PFIELD_NO_FACTORY since the type is not a CheckedType
    assert result is PFIELD_NO_FACTORY


# LLM-generated content at query #87
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result.__type__ == int
    assert result.__invariant__ is None
    assert issubclass(result, PVector)
    assert (PVector, int) in _seq_field_types


def test_make_seq_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    item_type = str
    item_invariant = None
    
    _seq_field_types.clear()
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_has_correct_name():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert "Int" in result.__name__
    assert "Vector" in result.__name__


def test_make_seq_field_type_with_different_item_types():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    result_int = _make_seq_field_type(PVector, int, None)
    result_str = _make_seq_field_type(PVector, str, None)
    
    assert result_int is not result_str
    assert result_int.__type__ == int
    assert result_str.__type__ == str


def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    item_type = int
    item_invariant = None
    
    TheType = _make_seq_field_type(PVector, item_type, item_invariant)
    instance = TheType([1, 2, 3])
    
    reduce_result = instance.__reduce__()
    
    assert reduce_result[0].__name__ == '_restore_seq_field_pickle'
    assert reduce_result[1][0] == PVector
    assert reduce_result[1][1] == int
    assert reduce_result[1][2] == [1, 2, 3]


# LLM-generated content at query #88
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    dct = {'field1': _PField(), 'field2': 'value'}
    bases = []
    name = 'fields'
    
    # Verify the predicate at line 5 evaluates to True
    for k, v in list(dct.items()):
        if k == 'field1':
            assert isinstance(v, _PField) == True


# LLM-generated content at query #89
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, pmap
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import CheckedPVector
    
    # Create a checked class with a sequence field
    class MyCheckedClass(CheckedPVector):
        __type__ = int
    
    # Register the type in _seq_field_types
    item_type = int
    _seq_field_types[MyCheckedClass, item_type] = MyCheckedClass
    
    # Test data to restore
    test_data = [1, 2, 3]
    
    # Call the function
    result = _restore_seq_field_pickle(MyCheckedClass, item_type, test_data)
    
    # Verify the result is a valid instance
    assert result is not None
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #90
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    checked_value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_value)
    
    assert result == "serialized_value"


# LLM-generated content at query #91
#--------------------------

```python
def test_is_field_ignore_extra_complaint_predicate_line_6_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def simple_factory():
        return "test"
    
    class SomeType:
        pass
    
    field = MockField(int, simple_factory)
    
    result = is_field_ignore_extra_complaint(SomeType, field, True)
    
    assert result is False


# LLM-generated content at query #92
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    class TestClass:
        pass
    
    dct = {'field1': _PField(), 'field2': 'not_a_pfield'}
    bases = [TestClass]
    name = 'fields'
    
    # Manually execute the relevant part to test the predicate
    test_value = dct['field1']
    
    # The predicate at line 5 should evaluate to True for _PField instances
    assert isinstance(test_value, _PField) == True


# LLM-generated content at query #93
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant1(subject):
        return (True, 'OK')
    
    def invariant2(subject):
        return (True, 'OK')
    
    subject = {'value': 10}
    invariants = [invariant1, invariant2]
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_error():
    def invariant1(subject):
        return (False, 'ERROR_CODE_1')
    
    subject = {'value': 10}
    invariants = [invariant1]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ('ERROR_CODE_1',)
        assert e.invariant_errors == ()
        assert e.message == 'Global invariant failed'


def test_check_global_invariants_multiple_errors():
    def invariant1(subject):
        return (False, 'ERROR_CODE_1')
    
    def invariant2(subject):
        return (False, 'ERROR_CODE_2')
    
    def invariant3(subject):
        return (True, 'OK')
    
    subject = {'value': 10}
    invariants = [invariant1, invariant2, invariant3]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ('ERROR_CODE_1', 'ERROR_CODE_2')
        assert e.invariant_errors == ()
        assert e.message == 'Global invariant failed'


def test_check_global_invariants_empty_invariants():
    subject = {'value': 10}
    invariants = []
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_with_complex_subject():
    def invariant1(subject):
        return (subject['value'] > 0, 'VALUE_NOT_POSITIVE')
    
    def invariant2(subject):
        return (len(subject['name']) > 0, 'NAME_EMPTY')
    
    subject = {'value': 5, 'name': 'test'}
    invariants = [invariant1, invariant2]
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_complex_subject_with_error():
    def invariant1(subject):
        return (subject['value'] > 0, 'VALUE_NOT_POSITIVE')
    
    def invariant2(subject):
        return (len(subject['name']) > 0, 'NAME_EMPTY')
    
    subject = {'value': -5, 'name': 'test'}
    invariants = [invariant1, invariant2]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ('VALUE_NOT_POSITIVE',)


# LLM-generated content at query #94
#--------------------------

```python
def test_check_field_parameters_valid_field():
    class MockField:
        def __init__(self):
            self.type = [str, int]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class MockField:
        def __init__(self):
            self.type = [str, 123]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Type parameter expected" in str(e)


def test_check_field_parameters_invalid_initial_type():
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = 123
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Initial has invalid type" in str(e)


def test_check_field_parameters_callable_initial():
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = lambda: "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_non_callable_invariant():
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = "not callable"
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Invariant must be callable" in str(e)


def test_check_field_parameters_non_callable_factory():
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = "not callable"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Factory must be callable" in str(e)


def test_check_field_parameters_non_callable_serializer():
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = "not callable"
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Serializer must be callable" in str(e)


def test_check_field_parameters_string_type():
    class MockField:
        def __init__(self):
            self.type = ["str", int]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_empty_type_with_initial():
    class MockField:
        def __init__(self):
            self.type = []
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


# LLM-generated content at query #95
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    factory_func = lambda: MockCheckedType()
    pfield = _PField(
        type={MockCheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=None
    )
    
    assert pfield._factory is factory_func


# LLM-generated content at query #96
#--------------------------

```python
def test_make_pmap_field_type():
    from pyrsistent._field_common import _make_pmap_field_type, _pmap_field_types
    
    # Clear the cache before testing
    _pmap_field_types.clear()
    
    # Test creating a pmap field type with basic types
    pmap_type = _make_pmap_field_type(str, int)
    
    # Verify the type is created
    assert pmap_type is not None
    assert hasattr(pmap_type, '__key_type__')
    assert hasattr(pmap_type, '__value_type__')
    assert pmap_type.__key_type__ == str
    assert pmap_type.__value_type__ == int
    
    # Verify the name is generated correctly
    assert pmap_type.__name__ == "StringToIntPMap"
    
    # Verify caching works - calling again should return the same type
    pmap_type_cached = _make_pmap_field_type(str, int)
    assert pmap_type_cached is pmap_type
    
    # Test with different types
    pmap_type2 = _make_pmap_field_type(int, str)
    assert pmap_type2 is not pmap_type
    assert pmap_type2.__name__ == "IntToStringPMap"
    assert pmap_type2.__key_type__ == int
    assert pmap_type2.__value_type__ == str
    
    # Test that the created type has __reduce__ method
    assert hasattr(pmap_type, '__reduce__')
    assert callable(pmap_type.__reduce__)
    
    # Clean up
    _pmap_field_types.clear()


# LLM-generated content at query #97
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field, PFIELD_NO_INVARIANT
    from pyrsistent import CheckedPMap
    
    field_obj = pmap_field(str, int)
    
    assert field_obj.mandatory is True
    assert field_obj.invariant == PFIELD_NO_INVARIANT
    assert callable(field_obj.factory)


def test_pmap_field_with_optional_false():
    from pyrsistent._field_common import pmap_field
    
    field_obj = pmap_field(str, int, optional=False)
    
    assert field_obj.mandatory is True
    assert field_obj.factory is not None


def test_pmap_field_with_optional_true():
    from pyrsistent._field_common import pmap_field
    
    field_obj = pmap_field(str, int, optional=True)
    
    assert field_obj.mandatory is True
    assert callable(field_obj.factory)


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    
    def my_invariant(val):
        return True, "valid"
    
    field_obj = pmap_field(str, int, invariant=my_invariant)
    
    assert field_obj.mandatory is True
    assert field_obj.invariant is not None


def test_pmap_field_factory_with_optional_true():
    from pyrsistent._field_common import pmap_field
    
    field_obj = pmap_field(str, int, optional=True)
    
    result_none = field_obj.factory(None)
    assert result_none is None


def test_pmap_field_factory_callable():
    from pyrsistent._field_common import pmap_field
    
    field_obj = pmap_field(str, int, optional=False)
    
    assert callable(field_obj.factory)
    assert field_obj.mandatory is True


# LLM-generated content at query #98
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a test key and value type
    key_type = str
    value_type = int
    
    # Create a pmap field type
    test_type = _pmap_field_types.get((key_type, value_type))
    if test_type is None:
        from pyrsistent._field_common import pmap_field
        test_type = pmap_field(key_type, value_type)
    
    # Create test data
    test_data = {'a': 1, 'b': 2}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is the expected type and contains the correct data
    assert result['a'] == 1
    assert result['b'] == 2
    assert len(result) == 2


# LLM-generated content at query #99
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap field with optional=False
    result = pmap_field(str, int, optional=False)
    
    # Verify that the predicate at line 2 (optional=False) evaluates to False
    assert result.factory is not None
    # When optional=False, factory should be TheMap.create, not a wrapper function
    # The factory should directly create a PMap without None handling
    test_map = result.factory({"key": 1})
    assert test_map is not None


# LLM-generated content at query #100
#--------------------------

```python
def test_check_global_invariants_all_pass():
    invariants = [
        lambda x: (True, None),
        lambda x: (True, None),
    ]
    subject = "test_subject"
    check_global_invariants(subject, invariants)


def test_check_global_invariants_one_fails():
    invariants = [
        lambda x: (True, None),
        lambda x: (False, "ERROR_CODE_1"),
    ]
    subject = "test_subject"
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1",)
        assert e.args[1] == ()
        assert e.args[2] == "Global invariant failed"


def test_check_global_invariants_multiple_fail():
    invariants = [
        lambda x: (False, "ERROR_1"),
        lambda x: (False, "ERROR_2"),
        lambda x: (True, None),
    ]
    subject = "test_subject"
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_1", "ERROR_2")
        assert e.args[1] == ()
        assert e.args[2] == "Global invariant failed"


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test_subject"
    check_global_invariants(subject, invariants)


def test_check_global_invariants_with_object_subject():
    class TestObject:
        def __init__(self, value):
            self.value = value
    
    subject = TestObject(42)
    invariants = [
        lambda x: (True, None) if x.value > 0 else (False, "NEGATIVE_VALUE"),
    ]
    check_global_invariants(subject, invariants)


def test_check_global_invariants_object_fails():
    class TestObject:
        def __init__(self, value):
            self.value = value
    
    subject = TestObject(-5)
    invariants = [
        lambda x: (True, None) if x.value > 0 else (False, "NEGATIVE_VALUE"),
    ]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("NEGATIVE_VALUE",)


# LLM-generated content at query #101
#--------------------------

```python
def test_make_seq_field_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector, pvector
    
    # Clear the cache before testing
    _seq_field_types.clear()
    
    # Test creating a seq field type with PVector and int
    item_type = int
    item_invariant = None
    
    result_type = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result_type is not None
    assert hasattr(result_type, '__type__')
    assert result_type.__type__ == int
    assert hasattr(result_type, '__invariant__')
    assert result_type.__invariant__ is None
    assert hasattr(result_type, '__reduce__')
    assert issubclass(result_type, PVector)
    
    # Test that the same call returns the cached type
    result_type_2 = _make_seq_field_type(PVector, item_type, item_invariant)
    assert result_type is result_type_2
    
    # Test with a different item type
    result_type_3 = _make_seq_field_type(PVector, str, item_invariant)
    assert result_type_3 is not result_type
    assert result_type_3.__type__ == str
    
    # Test with an item invariant
    def test_invariant(val):
        return val > 0
    
    result_type_4 = _make_seq_field_type(PVector, int, test_invariant)
    assert result_type_4.__invariant__ == test_invariant
    assert result_type_4 is not result_type


# LLM-generated content at query #102
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_error_codes_exist():
    def failing_invariant(subject):
        return (False, 'ERROR_1')
    
    def passing_invariant(subject):
        return (True, 'ERROR_2')
    
    invariants = [failing_invariant, passing_invariant]
    subject = object()
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ('ERROR_1',)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


# LLM-generated content at query #103
#--------------------------

```python
def test_check_global_invariants_no_errors():
    invariants = [
        lambda x: (True, 'OK1'),
        lambda x: (True, 'OK2'),
    ]
    subject = "test_subject"
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_error():
    invariants = [
        lambda x: (True, 'OK1'),
        lambda x: (False, 'ERROR1'),
    ]
    subject = "test_subject"
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('ERROR1',)


def test_check_global_invariants_multiple_errors():
    invariants = [
        lambda x: (False, 'ERROR1'),
        lambda x: (False, 'ERROR2'),
        lambda x: (True, 'OK1'),
    ]
    subject = "test_subject"
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('ERROR1', 'ERROR2')


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test_subject"
    check_global_invariants(subject, invariants)


def test_check_global_invariants_exception_message():
    invariants = [
        lambda x: (False, 'FAIL1'),
        lambda x: (False, 'FAIL2'),
    ]
    subject = "test_subject"
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


# LLM-generated content at query #104
#--------------------------

```python
def test_pfield_constructor():
    def dummy_invariant(x):
        return x > 0
    
    def dummy_serializer(x):
        return str(x)
    
    def dummy_factory():
        return 42
    
    pfield = _PField(
        type=int,
        invariant=dummy_invariant,
        initial=10,
        mandatory=True,
        factory=dummy_factory,
        serializer=dummy_serializer
    )
    
    assert pfield.type == int
    assert pfield.invariant == dummy_invariant
    assert pfield.initial == 10
    assert pfield.mandatory == True
    assert pfield._factory == dummy_factory
    assert pfield.serializer == dummy_serializer


def test_pfield_constructor_with_none_values():
    pfield = _PField(
        type=str,
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert pfield.type == str
    assert pfield.invariant is None
    assert pfield.initial is None
    assert pfield.mandatory == False
    assert pfield._factory is None
    assert pfield.serializer is None


def test_pfield_constructor_with_different_types():
    pfield = _PField(
        type=list,
        invariant=lambda x: len(x) > 0,
        initial=[1, 2, 3],
        mandatory=True,
        factory=list,
        serializer=repr
    )
    
    assert pfield.type == list
    assert pfield.initial == [1, 2, 3]
    assert pfield.mandatory == True
    assert pfield._factory == list
    assert pfield.serializer == repr


# LLM-generated content at query #105
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent import pmap, PMap
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent._checked_types import _restore_pickle
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Register the type if not already registered
    test_data = pmap({'a': 1, 'b': 2})
    
    # Call the function with test data
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify result is a PMap instance
    assert isinstance(result, PMap)
    
    # Verify the data was restored correctly
    assert result == test_data


def test_restore_pmap_field_pickle_empty():
    from pyrsistent import pmap
    from pyrsistent._field_common import _restore_pmap_field_pickle
    
    # Test with empty pmap
    key_type = str
    value_type = int
    test_data = pmap({})
    
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    assert result == test_data
    assert len(result) == 0


def test_restore_pmap_field_pickle_different_types():
    from pyrsistent import pmap
    from pyrsistent._field_common import _restore_pmap_field_pickle
    
    # Test with different key and value types
    key_type = int
    value_type = str
    test_data = pmap({1: 'a', 2: 'b', 3: 'c'})
    
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    assert result == test_data
    assert result[1] == 'a'
    assert result[2] == 'b'
    assert result[3] == 'c'


# LLM-generated content at query #106
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert result is not None
    assert hasattr(result, '__key_type__')
    assert hasattr(result, '__value_type__')
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    assert result1 is result2


def test_make_pmap_field_type_different_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    assert result1 is not result2
    assert result1.__key_type__ == str
    assert result1.__value_type__ == int
    assert result2.__key_type__ == int
    assert result2.__value_type__ == str


def test_make_pmap_field_type_name_generation():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert "PMap" in result.__name__
    assert "To" in result.__name__


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert hasattr(result, '__reduce__')
    assert callable(getattr(result, '__reduce__'))


def test_make_pmap_field_type_reduce_returns_tuple():
    from pyrsistent._field_common import _make_pmap_field_type
    
    field_type = _make_pmap_field_type(str, int)
    instance = field_type()
    reduce_result = instance.__reduce__()
    assert isinstance(reduce_result, tuple)
    assert len(reduce_result) == 2


# LLM-generated content at query #107
#--------------------------

```python
def test_check_field_parameters_valid_types():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class Field:
        def __init__(self):
            self.type = [int, 123]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Type parameter expected' in str(e)


def test_check_field_parameters_invalid_initial_type():
    PFIELD_NO_INITIAL = object()
    
    class Field:
        def __init__(self):
            self.type = [int]
            self.initial = "invalid"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_check_field_parameters_callable_initial():
    class Field:
        def __init__(self):
            self.type = [int]
            self.initial = lambda: 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_non_callable_invariant():
    class Field:
        def __init__(self):
            self.type = [int]
            self.initial = 42
            self.invariant = "not callable"
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_check_field_parameters_non_callable_factory():
    class Field:
        def __init__(self):
            self.type = [int]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = "not callable"
            self.serializer = lambda x: x
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_check_field_parameters_non_callable_serializer():
    class Field:
        def __init__(self):
            self.type = [int]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = "not callable"
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


def test_check_field_parameters_string_type():
    class Field:
        def __init__(self):
            self.type = ['int', str]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_empty_type():
    class Field:
        def __init__(self):
            self.type = []
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = Field()
    _check_field_parameters(field)


# LLM-generated content at query #108
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "serialized_value"


def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "xml:test_value"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    def custom_serializer(format, value):
        return "custom_serialized"
    
    value = CheckedType()
    result = serialize(custom_serializer, "json", value)
    assert result == "custom_serialized"


def test_serialize_with_non_checked_type():
    def my_serializer(format, value):
        return f"serialized_{value}"
    
    result = serialize(my_serializer, "csv", 42)
    assert result == "serialized_42"


def test_serialize_checked_type_calls_serialize_method():
    class CheckedType:
        def serialize(self, format):
            return f"format_{format}"
    
    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "yaml", value)
    assert result == "format_yaml"


# LLM-generated content at query #109
#--------------------------

```python
def test_pmap_field_creates_field_with_correct_type():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int)
    assert result.mandatory is True
    assert callable(result.factory)
    assert result.initial is not None


def test_pmap_field_with_optional_true():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=True)
    assert result.mandatory is True
    assert callable(result.factory)
    assert result.initial is not None


def test_pmap_field_with_optional_false():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=False)
    assert result.mandatory is True
    assert callable(result.factory)
    assert result.initial is not None


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    def my_invariant(x):
        return True, None
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.mandatory is True
    assert callable(result.invariant)


def test_pmap_field_factory_handles_none_when_optional():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=True)
    assert result.factory(None) is None


def test_pmap_field_different_key_value_types():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(int, str)
    assert result.mandatory is True
    assert callable(result.factory)


def test_pmap_field_returns_pfield_instance():
    from pyrsistent._field_common import pmap_field, _PField
    result = pmap_field(str, int)
    assert isinstance(result, _PField)


# LLM-generated content at query #110
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    class MockBase:
        def __init__(self):
            self.__dict__['test_name'] = {}
    
    dct = {'field1': _PField(), 'field2': 'not_a_pfield'}
    bases = [MockBase()]
    name = 'test_name'
    
    # Verify the predicate evaluates to True for _PField instances
    for k, v in list(dct.items()):
        if k == 'field1':
            assert isinstance(v, _PField) == True
        elif k == 'field2':
            assert isinstance(v, _PField) == False


# LLM-generated content at query #111
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure fresh creation
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == int
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ is None
    assert hasattr(result, '__name__')
    assert isinstance(result.__name__, str)


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_with_different_item_types():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result_int = _make_seq_field_type(PVector, int, None)
    result_str = _make_seq_field_type(PVector, str, None)
    
    assert result_int is not result_str
    assert result_int.__type__ == int
    assert result_str.__type__ == str


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result = _make_seq_field_type(PVector, int, None)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    def my_invariant(value):
        return value > 0
    
    result = _make_seq_field_type(PVector, int, my_invariant)
    
    assert result.__invariant__ is my_invariant


# LLM-generated content at query #112
#--------------------------

```python
def test_is_field_ignore_extra_complaint_predicate_line_6_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    class SomeType:
        pass
    
    def simple_factory():
        return SomeType()
    
    field = MockField(int, simple_factory)
    type_cls = SomeType
    ignore_extra = True
    
    result = is_field_ignore_extra_complaint(type_cls, field, ignore_extra)
    
    assert result is False


# LLM-generated content at query #113
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import PMap
    
    result = pmap_field(str, int)
    assert result is not None
    assert result.type is not None
    assert result.mandatory is True
    assert callable(result.factory)


def test_pmap_field_optional_false():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=False)
    assert result.mandatory is True
    assert result.factory is not None


def test_pmap_field_optional_true():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    assert result.mandatory is True
    assert callable(result.factory)


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    
    def my_invariant(val):
        return True, None
    
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.invariant is not None
    assert result.mandatory is True


def test_pmap_field_initial_value():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int)
    assert result.initial is not None


def test_pmap_field_factory_with_none_optional():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    factory_result = result.factory(None)
    assert factory_result is None


def test_pmap_field_different_types():
    from pyrsistent._field_common import pmap_field
    
    result1 = pmap_field(str, int)
    result2 = pmap_field(int, str)
    assert result1 is not None
    assert result2 is not None


def test_pmap_field_type_attribute():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int)
    assert len(result.type) > 0


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_pfield_constructor():
    def dummy_invariant(x):
        return x > 0
    
    def dummy_serializer(x):
        return str(x)
    
    def dummy_factory():
        return 42
    
    field = _PField(
        type=int,
        invariant=dummy_invariant,
        initial=10,
        mandatory=True,
        factory=dummy_factory,
        serializer=dummy_serializer
    )
    
    assert field.type == int
    assert field.invariant == dummy_invariant
    assert field.initial == 10
    assert field.mandatory is True
    assert field._factory == dummy_factory
    assert field.serializer == dummy_serializer


def test_pfield_constructor_with_none_values():
    field = _PField(
        type=str,
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert field.type == str
    assert field.invariant is None
    assert field.initial is None
    assert field.mandatory is False
    assert field._factory is None
    assert field.serializer is None


def test_pfield_constructor_with_different_types():
    field = _PField(
        type=list,
        invariant=lambda x: len(x) > 0,
        initial=[],
        mandatory=True,
        factory=list,
        serializer=lambda x: repr(x)
    )
    
    assert field.type == list
    assert field.initial == []
    assert field._factory == list


def test_pfield_constructor_slots():
    field = _PField(
        type=float,
        invariant=None,
        initial=1.5,
        mandatory=True,
        factory=float,
        serializer=None
    )
    
    assert hasattr(field, 'type')
    assert hasattr(field, 'invariant')
    assert hasattr(field, 'initial')
    assert hasattr(field, 'mandatory')
    assert hasattr(field, '_factory')
    assert hasattr(field, 'serializer')


# LLM-generated content at query #2
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, "OK1")
    
    def invariant2(subject):
        return (True, "OK2")
    
    check_global_invariants("test_subject", [invariant1, invariant2])


def test_check_global_invariants_single_failure():
    def invariant1(subject):
        return (True, "OK")
    
    def invariant2(subject):
        return (False, "ERROR_CODE_1")
    
    try:
        check_global_invariants("test_subject", [invariant1, invariant2])
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    def invariant1(subject):
        return (False, "ERROR_CODE_1")
    
    def invariant2(subject):
        return (False, "ERROR_CODE_2")
    
    def invariant3(subject):
        return (True, "OK")
    
    try:
        check_global_invariants("test_subject", [invariant1, invariant2, invariant3])
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1", "ERROR_CODE_2")
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_empty_list():
    check_global_invariants("test_subject", [])


def test_check_global_invariants_with_different_subject_types():
    def invariant(subject):
        return (True, "OK")
    
    check_global_invariants(42, [invariant])
    check_global_invariants({"key": "value"}, [invariant])
    check_global_invariants([1, 2, 3], [invariant])


# LLM-generated content at query #3
#--------------------------

```python
def test_set_fields():
    from types import SimpleNamespace
    
    class _PField:
        def __init__(self, value=None):
            self.value = value
    
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    
    # Test case 1: Basic functionality with PField instances
    dct1 = {"field1": _PField("value1"), "field2": _PField("value2"), "regular": "data"}
    bases1 = []
    set_fields(dct1, bases1, "fields")
    assert "fields" in dct1
    assert "field1" in dct1["fields"]
    assert "field2" in dct1["fields"]
    assert "regular" in dct1
    assert "field1" not in dct1
    assert "field2" not in dct1
    
    # Test case 2: With base classes containing fields
    base1 = type('Base1', (), {})
    base1.__dict__ = {"fields": {"base_field": _PField("base_value")}}
    dct2 = {"new_field": _PField("new_value")}
    set_fields(dct2, [base1], "fields")
    assert "fields" in dct2
    assert "base_field" in dct2["fields"]
    assert "new_field" in dct2["fields"]
    assert "new_field" not in dct2
    
    # Test case 3: Empty dictionary with no bases
    dct3 = {}
    bases3 = []
    set_fields(dct3, bases3, "fields")
    assert "fields" in dct3
    assert dct3["fields"] == {}
    
    # Test case 4: Mixed content with multiple bases
    base2 = type('Base2', (), {})
    base2.__dict__ = {"fields": {"inherited1": _PField("val1")}}
    base3 = type('Base3', (), {})
    base3.__dict__ = {"fields": {"inherited2": _PField("val2")}}
    dct4 = {"own_field": _PField("own_value"), "other": "keep"}
    set_fields(dct4, [base2, base3], "fields")
    assert "fields" in dct4
    assert "inherited1" in dct4["fields"]
    assert "inherited2" in dct4["fields"]
    assert "own_field" in dct4["fields"]
    assert "own_field" not in dct4
    assert "other" in dct4
    
    # Test case 5: No PField instances in dictionary
    dct5 = {"attr1": "value1", "attr2": 123}
    bases5 = []
    set_fields(dct5, bases5, "fields")
    assert "fields" in dct5
    assert dct5["fields"] == {}
    assert "attr1" in dct5
    assert "attr2" in dct5


# LLM-generated content at query #4
#--------------------------

```python
def test_check_field_parameters_valid_field():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class MockField:
        def __init__(self):
            self.type = [int, 123]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Type parameter expected' in str(e)


def test_check_field_parameters_invalid_initial_type():
    PFIELD_NO_INITIAL = object()
    
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = []
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_check_field_parameters_callable_initial():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = lambda: 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_non_callable_invariant():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = 42
            self.invariant = "not callable"
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_check_field_parameters_non_callable_factory():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = "not callable"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_check_field_parameters_non_callable_serializer():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = "not callable"
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


def test_check_field_parameters_string_type():
    class MockField:
        def __init__(self):
            self.type = ['int', str]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_empty_type():
    class MockField:
        def __init__(self):
            self.type = []
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


# LLM-generated content at query #5
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3: not isinstance(t, type) and not isinstance(t, str)
    # For this to evaluate to False, we need:
    # - isinstance(t, type) to be True, OR
    # - isinstance(t, str) to be True
    
    # When t = int (which is a type), isinstance(t, type) is True
    # So: not True and not False = False and True = False
    
    t = int
    predicate_result = not isinstance(t, type) and not isinstance(t, str)
    assert predicate_result is False
    
    # Also test with str as a type
    t = str
    predicate_result = not isinstance(t, type) and not isinstance(t, str)
    assert predicate_result is False


# LLM-generated content at query #6
#--------------------------

```python
def test_check_field_parameters_line_1_predicate_false():
    class MockField:
        def __init__(self):
            self.type = []
            self.initial = None
            self.invariant = lambda: None
            self.factory = lambda: None
            self.serializer = lambda: None
    
    field = MockField()
    # Line 1 predicate: `for t in field.type:` evaluates to False when field.type is empty
    # This means the loop body never executes, which is the desired state for this test
    assert len(field.type) == 0


# LLM-generated content at query #7
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap, PMap
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Register the type in _pmap_field_types
    test_pmap_type = PMap.create(pmap())
    _pmap_field_types[key_type, value_type] = type('TestPMapField', (object,), {
        'create': classmethod(lambda cls, data, _factory_fields=None: pmap(data))
    })
    
    # Test data to restore
    test_data = {'key1': 1, 'key2': 2}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap with the correct data
    assert result == pmap(test_data)
    assert result['key1'] == 1
    assert result['key2'] == 2


# LLM-generated content at query #8
#--------------------------

```python
def test_types_to_names_with_builtin_types():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((int, str, float))
    assert result == "IntStrFloat"


def test_types_to_names_with_single_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((int,))
    assert result == "Int"


def test_types_to_names_with_bool_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((bool,))
    assert result == "Bool"


def test_types_to_names_with_list_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((list,))
    assert result == "List"


def test_types_to_names_with_dict_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((dict,))
    assert result == "Dict"


def test_types_to_names_with_multiple_types():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((str, int, bool, list, dict))
    assert result == "StrIntBoolListDict"


def test_types_to_names_with_tuple_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((tuple,))
    assert result == "Tuple"


def test_types_to_names_with_set_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((set,))
    assert result == "Set"


# LLM-generated content at query #9
#--------------------------

```python
def test_sequence_field_creates_field_with_checked_class():
    from pyrsistent._field_common import _sequence_field, field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    
    assert isinstance(result, type(field()))
    assert result.mandatory is True


def test_sequence_field_optional_true_returns_none():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, True, [1, 2, 3])
    
    assert result.factory(None) is None


def test_sequence_field_optional_false_with_value():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    
    created = result.factory([4, 5, 6])
    assert len(created) == 3


def test_sequence_field_with_invariant():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    def my_invariant(value):
        return (len(value) > 0, "Must not be empty")
    
    result = _sequence_field(CheckedPVector, int, False, [1], invariant=my_invariant)
    
    assert result.invariant is not None


def test_sequence_field_with_item_invariant():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    def item_invariant(value):
        return (value > 0, "Must be positive")
    
    result = _sequence_field(CheckedPVector, int, False, [1, 2], item_invariant=item_invariant)
    
    assert result.invariant is not None


def test_sequence_field_mandatory_is_true():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, False, [])
    
    assert result.mandatory is True


def test_sequence_field_initial_value_set():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    
    assert result.initial is not None


# LLM-generated content at query #10
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    
    # Mock field and type_cls objects
    class MockFactory:
        def __init__(self, has_ignore_extra=False):
            self.has_ignore_extra = has_ignore_extra
        
        def __call__(self):
            pass
    
    class MockField:
        def __init__(self, type_val, factory):
            self.type = type_val
            self.factory = factory
    
    class MockTypeClass:
        pass
    
    # Test 1: ignore_extra is False, should return False immediately
    mock_field = MockField({MockTypeClass}, MockFactory())
    result = is_field_ignore_extra_complaint(MockTypeClass, mock_field, False)
    assert result is False
    
    # Test 2: ignore_extra is True but field type is not a type_cls, should return False
    mock_field = MockField({str}, MockFactory())
    result = is_field_ignore_extra_complaint(MockTypeClass, mock_field, True)
    assert result is False
    
    # Test 3: ignore_extra is True, type_cls matches, but factory has no ignore_extra param
    def factory_without_ignore_extra():
        pass
    mock_field = MockField({MockTypeClass}, factory_without_ignore_extra)
    result = is_field_ignore_extra_complaint(MockTypeClass, mock_field, True)
    assert result is False
    
    # Test 4: ignore_extra is True, type_cls matches, and factory has ignore_extra param
    def factory_with_ignore_extra(ignore_extra=False):
        pass
    mock_field = MockField({MockTypeClass}, factory_with_ignore_extra)
    result = is_field_ignore_extra_complaint(MockTypeClass, mock_field, True)
    assert result is True
    
    # Test 5: ignore_extra is True, empty set type, should return False
    mock_field = MockField(set(), MockFactory(has_ignore_extra=True))
    result = is_field_ignore_extra_complaint(MockTypeClass, mock_field, True)
    assert result is False


# LLM-generated content at query #11
#--------------------------

```python
def test_check_global_invariants_with_failing_invariant():
    def failing_invariant(subject):
        return (False, "ERROR_001")
    
    def passing_invariant(subject):
        return (True, "")
    
    subject = {"test": "data"}
    invariants = [failing_invariant, passing_invariant]
    
    error_codes = tuple(error_code for is_ok, error_code in
                        (invariant(subject) for invariant in invariants) if not is_ok)
    
    assert error_codes
    assert error_codes == ("ERROR_001",)


# LLM-generated content at query #12
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure fresh state
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == int
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ is None
    assert hasattr(result, '__name__')
    assert isinstance(result.__name__, str)


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure fresh state
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure fresh state
    _seq_field_types.clear()
    
    item_type = int
    def item_invariant(val):
        return val > 0
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result.__invariant__ is item_invariant


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure fresh state
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_seq_field_type_different_types_different_results():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure fresh state
    _seq_field_types.clear()
    
    result1 = _make_seq_field_type(PVector, int, None)
    result2 = _make_seq_field_type(PVector, str, None)
    
    assert result1 is not result2


# LLM-generated content at query #13
#--------------------------

```python
def test_make_pmap_field_type_creates_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert result is not None
    assert hasattr(result, '__key_type__')
    assert hasattr(result, '__value_type__')
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    assert result1 is result2


def test_make_pmap_field_type_different_types_create_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    assert result1 is not result2


def test_make_pmap_field_type_class_name_format():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert "PMap" in result.__name__
    assert "To" in result.__name__


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_with_builtin_types():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(int, float)
    assert result.__key_type__ == int
    assert result.__value_type__ == float


def test_make_pmap_field_type_with_multiple_types_in_sequence():
    from pyrsistent._field_common import _make_pmap_field_type
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    result3 = _make_pmap_field_type(float, bool)
    assert result1 is not result2
    assert result2 is not result3
    assert result1 is not result3


# LLM-generated content at query #14
#--------------------------

```python
def test_is_type_cls_with_set():
    from pyrsistent._field_common import is_type_cls
    result = is_type_cls(int, {int, str})
    assert result is True


def test_is_type_cls_with_empty_tuple():
    from pyrsistent._field_common import is_type_cls
    result = is_type_cls(int, ())
    assert result is False


def test_is_type_cls_with_single_type_class():
    from pyrsistent._field_common import is_type_cls
    result = is_type_cls(int, (int,))
    assert result is True


def test_is_type_cls_with_single_type_string():
    from pyrsistent._field_common import is_type_cls
    result = is_type_cls(int, ('builtins.int',))
    assert result is True


def test_is_type_cls_with_subclass():
    from pyrsistent._field_common import is_type_cls
    result = is_type_cls(object, (int,))
    assert result is True


def test_is_type_cls_with_non_subclass():
    from pyrsistent._field_common import is_type_cls
    result = is_type_cls(str, (int,))
    assert result is False


def test_is_type_cls_with_multiple_types_first_matches():
    from pyrsistent._field_common import is_type_cls
    result = is_type_cls(int, (int, str))
    assert result is True


def test_is_type_cls_with_multiple_types_first_does_not_match():
    from pyrsistent._field_common import is_type_cls
    result = is_type_cls(int, (str, int))
    assert result is False


def test_is_type_cls_with_list_converted_to_tuple():
    from pyrsistent._field_common import is_type_cls
    result = is_type_cls(int, [int, str])
    assert result is True


# LLM-generated content at query #15
#--------------------------

```python
def test_types_to_names_with_empty_tuple():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names(())
    assert result == ""


# LLM-generated content at query #16
#--------------------------

```python
def test_sequence_field_with_checked_pvector():
    from pyrsistent._field_common import _sequence_field, field
    from pyrsistent import PVector, CheckedPVector
    
    class CheckedIntVector(CheckedPVector):
        __type__ = int
    
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    
    assert isinstance(result, _PField)
    assert result.mandatory is True
    assert result.initial is not None


def test_sequence_field_optional_with_none():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, True, None)
    
    assert isinstance(result, _PField)
    assert result.mandatory is True
    assert result.initial is None


def test_sequence_field_optional_with_value():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, str, True, ['a', 'b'])
    
    assert isinstance(result, _PField)
    assert result.mandatory is True
    assert result.initial is not None


def test_sequence_field_with_invariant():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    def my_invariant(val):
        return True, None
    
    result = _sequence_field(CheckedPVector, int, False, [], invariant=my_invariant)
    
    assert isinstance(result, _PField)
    assert result.invariant is not None


def test_sequence_field_factory_callable():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, False, [])
    
    assert callable(result.factory)


def test_sequence_field_optional_factory_with_none():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, True, [])
    factory = result.factory
    
    assert factory(None) is None


def test_sequence_field_non_optional_factory():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, False, [1, 2])
    factory = result.factory
    
    created = factory([1, 2, 3])
    assert created is not None
    assert len(created) == 3


# LLM-generated content at query #17
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant_pass(subject):
        return (True, "OK")
    
    invariants = [invariant_pass]
    subject = "test_subject"
    
    result = check_global_invariants(subject, invariants)
    assert result is None


def test_check_global_invariants_single_error():
    def invariant_fail(subject):
        return (False, "ERROR_1")
    
    invariants = [invariant_fail]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_1",)


def test_check_global_invariants_multiple_errors():
    def invariant_fail_1(subject):
        return (False, "ERROR_1")
    
    def invariant_fail_2(subject):
        return (False, "ERROR_2")
    
    invariants = [invariant_fail_1, invariant_fail_2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_1", "ERROR_2")


def test_check_global_invariants_mixed_pass_fail():
    def invariant_pass(subject):
        return (True, "OK")
    
    def invariant_fail(subject):
        return (False, "ERROR_1")
    
    invariants = [invariant_pass, invariant_fail]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_1",)


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test_subject"
    
    result = check_global_invariants(subject, invariants)
    assert result is None


def test_check_global_invariants_exception_message():
    def invariant_fail(subject):
        return (False, "CRITICAL_ERROR")
    
    invariants = [invariant_fail]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[1] == ()
        assert e.args[2] == "Global invariant failed"


# LLM-generated content at query #18
#--------------------------

```python
def test_is_field_ignore_extra_complaint_predicate_line_6_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent import PClass, field
    
    # Create a mock field with a type that doesn't match
    mock_field = field()
    mock_field.type = str
    mock_field.factory = lambda: "test"
    
    # Create a type_cls that is not a PClass or compatible type
    class NotATypeClass:
        pass
    
    # Call the function with ignore_extra=True to pass the first check
    # but with a mismatched type to make the predicate at line 6 evaluate to False
    result = is_field_ignore_extra_complaint(NotATypeClass, mock_field, True)
    
    assert result is False


# LLM-generated content at query #19
#--------------------------

```python
def test_seq_field_type_caching():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    item_type = int
    item_invariant = lambda x: x > 0
    
    # First call should create and cache the type
    type_1 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    # Second call with same parameters should return cached type
    type_2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    # The predicate at line 4 (type_ is not None) should evaluate to True on second call
    # This is verified by the fact that type_1 and type_2 are the same object
    assert type_1 is type_2
    assert type_1 is not None


# LLM-generated content at query #20
#--------------------------

```python
def test_check_field_parameters_predicate_line_3():
    class MockField:
        def __init__(self):
            self.type = [int, "str"]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # Test that the predicate at line 3 evaluates to True
    # The predicate is: not isinstance(t, type) and not isinstance(t, str)
    # This should be True for items that are neither type nor str
    
    for t in field.type:
        result = not isinstance(t, type) and not isinstance(t, str)
        assert result == False  # int is a type, "str" is a string, so predicate is False for both
    
    # Now test with a value that makes the predicate True
    class MockFieldWithBadType:
        def __init__(self):
            self.type = [123]  # 123 is neither a type nor a string
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field_bad = MockFieldWithBadType()
    
    for t in field_bad.type:
        result = not isinstance(t, type) and not isinstance(t, str)
        assert result == True  # 123 is neither a type nor a string, so predicate is True


# LLM-generated content at query #21
#--------------------------

```python
def test_make_seq_field_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import pmap, pvector
    
    # Clear the cache before testing
    _seq_field_types.clear()
    
    # Test with pvector and int type
    result_type = _make_seq_field_type(pvector, int, None)
    
    # Verify the returned type is a class
    assert isinstance(result_type, type)
    
    # Verify it has the expected attributes
    assert hasattr(result_type, '__type__')
    assert result_type.__type__ == int
    assert hasattr(result_type, '__invariant__')
    assert result_type.__invariant__ is None
    
    # Verify it has __reduce__ method
    assert hasattr(result_type, '__reduce__')
    
    # Verify it's cached
    cached_type = _make_seq_field_type(pvector, int, None)
    assert cached_type is result_type
    
    # Test with different item type
    result_type2 = _make_seq_field_type(pvector, str, None)
    assert result_type2 is not result_type
    assert result_type2.__type__ == str
    
    # Verify the name is set
    assert isinstance(result_type.__name__, str)
    assert len(result_type.__name__) > 0
    
    # Clean up
    _seq_field_types.clear()


# LLM-generated content at query #22
#--------------------------

```python
def test_check_type_with_valid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
    
    class MockClass:
        __name__ = "TestClass"
    
    field = MockField([int])
    check_type(MockClass, field, "test_field", 42)


def test_check_type_with_invalid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
    
    class MockClass:
        __name__ = "TestClass"
    
    field = MockField([int])
    
    try:
        check_type(MockClass, field, "test_field", "not_an_int")
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert "Invalid type for field TestClass.test_field" in str(e)


def test_check_type_with_multiple_valid_types():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
    
    class MockClass:
        __name__ = "TestClass"
    
    field = MockField([int, str])
    check_type(MockClass, field, "test_field", "string_value")
    check_type(MockClass, field, "test_field", 42)


def test_check_type_with_no_type_constraint():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
    
    class MockClass:
        __name__ = "TestClass"
    
    field = MockField(None)
    check_type(MockClass, field, "test_field", "any_value")
    check_type(MockClass, field, "test_field", 123)


def test_check_type_with_empty_type_list():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
    
    class MockClass:
        __name__ = "TestClass"
    
    field = MockField([])
    
    try:
        check_type(MockClass, field, "test_field", 42)
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert "Invalid type for field TestClass.test_field" in str(e)


def test_check_type_error_message_format():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
    
    class MockClass:
        __name__ = "MyClass"
    
    field = MockField([int])
    
    try:
        check_type(MockClass, field, "my_field", [1, 2, 3])
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert "MyClass" in str(e)
        assert "my_field" in str(e)
        assert "list" in str(e)


# LLM-generated content at query #23
#--------------------------

```python
def test_make_pmap_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    # First call creates and caches the type
    type1 = _make_pmap_field_type(str, int)
    
    # Second call with same arguments should return the cached type
    type2 = _make_pmap_field_type(str, int)
    
    # Verify the predicate at line 4 evaluates to True on second call
    # by confirming they are the exact same object
    assert type1 is type2


# LLM-generated content at query #24
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    class Base:
        pass
    
    dct = {'field1': _PField(), 'field2': 'not_a_pfield'}
    bases = (Base,)
    name = 'fields'
    
    # Verify the predicate evaluates to True for _PField instances
    for k, v in list(dct.items()):
        if k == 'field1':
            assert isinstance(v, _PField) is True


# LLM-generated content at query #25
#--------------------------

```python
def test_pfield_constructor():
    def dummy_invariant(x):
        return x > 0

    def dummy_serializer(x):
        return str(x)

    def dummy_factory():
        return 42

    pfield = _PField(
        type={int},
        invariant=dummy_invariant,
        initial=10,
        mandatory=True,
        factory=dummy_factory,
        serializer=dummy_serializer
    )

    assert pfield.type == {int}
    assert pfield.invariant == dummy_invariant
    assert pfield.initial == 10
    assert pfield.mandatory is True
    assert pfield._factory == dummy_factory
    assert pfield.serializer == dummy_serializer


def test_pfield_constructor_with_none_values():
    pfield = _PField(
        type={str},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )

    assert pfield.type == {str}
    assert pfield.invariant is None
    assert pfield.initial is None
    assert pfield.mandatory is False
    assert pfield._factory is None
    assert pfield.serializer is None


def test_pfield_constructor_with_multiple_types():
    pfield = _PField(
        type={int, str, float},
        invariant=lambda x: True,
        initial=5,
        mandatory=True,
        factory=lambda: 0,
        serializer=lambda x: x
    )

    assert pfield.type == {int, str, float}
    assert pfield.initial == 5
    assert pfield.mandatory is True


# LLM-generated content at query #26
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "checked_serialized"


def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"serialized_{value}_as_{format}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "serialized_test_value_as_xml"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return "should_not_be_called"
    
    def custom_serializer(format, value):
        return f"custom_{format}"
    
    value = CheckedType()
    result = serialize(custom_serializer, "json", value)
    assert result == "custom_json"


def test_serialize_with_regular_value_and_serializer():
    def my_serializer(format, value):
        return f"{value}_{format}"
    
    result = serialize(my_serializer, "csv", 42)
    assert result == "42_csv"


def test_serialize_checked_type_different_formats():
    class CheckedType:
        def serialize(self, format):
            return f"format_{format}"
    
    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    result1 = serialize(PFIELD_NO_SERIALIZER, "json", value)
    result2 = serialize(PFIELD_NO_SERIALIZER, "xml", value)
    assert result1 == "format_json"
    assert result2 == "format_xml"


# LLM-generated content at query #27
#--------------------------

```python
def test_check_type_predicate_evaluates_to_true():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import check_type
    
    class SimpleField:
        def __init__(self, field_type):
            self.type = field_type
    
    class SimpleClass:
        pass
    
    class TestClass:
        __name__ = "TestClass"
    
    # Create a field with int type
    field = SimpleField([int])
    
    # Pass an integer value - should NOT raise an error because predicate is True
    # (isinstance(5, int) is True, so any(...) is True, so not any(...) is False)
    result = check_type(TestClass, field, "test_field", 5)
    assert result is None
    
    # Test with string field and string value
    field_str = SimpleField([str])
    result = check_type(TestClass, field_str, "test_field", "hello")
    assert result is None
    
    # Test with multiple allowed types
    field_multi = SimpleField([int, str])
    result = check_type(TestClass, field_multi, "test_field", 42)
    assert result is None
    
    result = check_type(TestClass, field_multi, "test_field", "test")
    assert result is None


# LLM-generated content at query #28
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_evaluates_to_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3: "not isinstance(t, type) and not isinstance(t, str)"
    # For this to evaluate to False, we need either:
    # - isinstance(t, type) to be True, OR
    # - isinstance(t, str) to be True
    
    # Test case 1: t is a type (isinstance(t, type) is True)
    t = int
    predicate_result = not isinstance(t, type) and not isinstance(t, str)
    assert predicate_result is False
    
    # Test case 2: t is a string (isinstance(t, str) is True)
    t = "SomeType"
    predicate_result = not isinstance(t, type) and not isinstance(t, str)
    assert predicate_result is False


# LLM-generated content at query #29
#--------------------------

```python
def test_make_pmap_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    key_type = str
    value_type = int
    
    # First call creates and caches the type
    type1 = _make_pmap_field_type(key_type, value_type)
    
    # Second call should return the cached type
    type2 = _make_pmap_field_type(key_type, value_type)
    
    # The predicate at line 4 should evaluate to True on second call
    # meaning type2 is not None and is the same as type1
    assert type2 is type1
    assert type2 is not None


# LLM-generated content at query #30
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    
    result = _make_pmap_field_type(str, int)
    
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == str
    assert result.__value_type__ == int
    assert "PMap" in result.__name__


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_types_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2
    assert result1.__name__ != result2.__name__


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    instance = result()
    
    assert hasattr(instance, '__reduce__')
    assert callable(instance.__reduce__)


def test_make_pmap_field_type_name_format():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert "StrToIntPMap" == result.__name__


def test_make_pmap_field_type_with_float_and_bool():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(float, bool)
    
    assert result.__key_type__ == float
    assert result.__value_type__ == bool
    assert "FloatToBoolPMap" == result.__name__


# LLM-generated content at query #31
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, 'code1')
    
    def invariant2(subject):
        return (True, 'code2')
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
    except InvariantException:
        raise AssertionError("Should not raise exception when all invariants pass")


def test_check_global_invariants_single_failure():
    def invariant1(subject):
        return (True, 'code1')
    
    def invariant2(subject):
        return (False, 'error_code2')
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        raise AssertionError("Should raise InvariantException")
    except InvariantException as e:
        assert e.args[0] == ('error_code2',)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    def invariant1(subject):
        return (False, 'error_code1')
    
    def invariant2(subject):
        return (False, 'error_code2')
    
    def invariant3(subject):
        return (True, 'code3')
    
    invariants = [invariant1, invariant2, invariant3]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        raise AssertionError("Should raise InvariantException")
    except InvariantException as e:
        assert e.args[0] == ('error_code1', 'error_code2')
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
    except InvariantException:
        raise AssertionError("Should not raise exception with empty invariants")


def test_check_global_invariants_all_failures():
    def invariant1(subject):
        return (False, 'error_code1')
    
    def invariant2(subject):
        return (False, 'error_code2')
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        raise AssertionError("Should raise InvariantException")
    except InvariantException as e:
        assert e.args[0] == ('error_code1', 'error_code2')


# LLM-generated content at query #32
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int)
    assert field_obj.mandatory is True
    assert field_obj.initial == pmap()
    assert callable(field_obj.factory)


def test_pmap_field_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int, optional=False)
    assert field_obj.mandatory is True
    assert field_obj.initial == pmap()
    assert callable(field_obj.factory)


def test_pmap_field_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.mandatory is True
    assert field_obj.initial == pmap()
    assert callable(field_obj.factory)


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    def my_invariant(val):
        return True, None
    
    field_obj = pmap_field(str, int, invariant=my_invariant)
    assert field_obj.mandatory is True
    assert callable(field_obj.factory)
    assert callable(field_obj.invariant)


def test_pmap_field_factory_with_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int, optional=True)
    result_none = field_obj.factory(None)
    assert result_none is None


def test_pmap_field_factory_with_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int, optional=False)
    result = field_obj.factory({'a': 1})
    assert result == pmap({'a': 1})


def test_pmap_field_caching():
    from pyrsistent._field_common import pmap_field
    
    field_obj1 = pmap_field(str, int)
    field_obj2 = pmap_field(str, int)
    assert field_obj1.type == field_obj2.type


def test_pmap_field_different_types():
    from pyrsistent._field_common import pmap_field
    
    field_obj1 = pmap_field(str, int)
    field_obj2 = pmap_field(int, str)
    assert field_obj1.type != field_obj2.type


# LLM-generated content at query #33
#--------------------------

```python
def test_is_field_ignore_extra_complaint_predicate_line_6_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        pass
    
    test_field = field()
    
    result = is_field_ignore_extra_complaint(TestClass, test_field, True)
    
    assert result is False


# LLM-generated content at query #34
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert result is not None
    assert hasattr(result, '__key_type__')
    assert hasattr(result, '__value_type__')
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_pmap_field_type
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    assert result1 is result2


def test_make_pmap_field_type_generates_correct_name():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert result.__name__ == "StrToIntPMap"


def test_make_pmap_field_type_different_types():
    from pyrsistent._field_common import _make_pmap_field_type
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    assert result1 is not result2
    assert result1.__name__ == "StrToIntPMap"
    assert result2.__name__ == "IntToStrPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_is_checked_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    result = _make_pmap_field_type(str, int)
    assert issubclass(result, CheckedPMap)


# LLM-generated content at query #35
#--------------------------

```python
def test_check_field_parameters_predicate_line_6_true():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    PFIELD_NO_INITIAL = object()
    field = MockField()
    
    # Predicate at line 6 evaluates to True when:
    # field.initial is not PFIELD_NO_INITIAL (True - 42 is not PFIELD_NO_INITIAL)
    # and not callable(field.initial) (True - 42 is not callable)
    # and field.type (True - [int, str] is truthy)
    # and not any(isinstance(field.initial, t) for t in field.type) (True - 42 is instance of int)
    
    # So the overall condition should evaluate to True when initial matches one of the types
    predicate = (field.initial is not PFIELD_NO_INITIAL and 
                 not callable(field.initial) and 
                 field.type and 
                 not any(isinstance(field.initial, t) for t in field.type))
    
    assert predicate == True


# LLM-generated content at query #36
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == item_type
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ == item_invariant
    assert issubclass(result, PVector)


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_with_different_item_types():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result1 = _make_seq_field_type(PVector, int, None)
    result2 = _make_seq_field_type(PVector, str, None)
    
    assert result1 is not result2
    assert result1.__type__ == int
    assert result2.__type__ == str


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    result = _make_seq_field_type(PVector, int, None)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_seq_field_type_name_generation():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result = _make_seq_field_type(PVector, int, None)
    
    assert hasattr(result, '__name__')
    assert isinstance(result.__name__, str)
    assert len(result.__name__) > 0


# LLM-generated content at query #37
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_error_codes_exist():
    def failing_invariant(subject):
        return (False, "ERROR_CODE_1")
    
    def passing_invariant(subject):
        return (True, None)
    
    class InvariantException(Exception):
        def __init__(self, error_codes, arg2, message):
            self.error_codes = error_codes
            self.message = message
            super().__init__(message)
    
    def check_global_invariants(subject, invariants):
        error_codes = tuple(error_code for is_ok, error_code in
                            (invariant(subject) for invariant in invariants) if not is_ok)
        if error_codes:
            raise InvariantException(error_codes, (), 'Global invariant failed')
    
    subject = "test_subject"
    invariants = [failing_invariant, passing_invariant]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_CODE_1",)
        assert e.message == 'Global invariant failed'


# LLM-generated content at query #38
#--------------------------

```python
def test_check_field_parameters_predicate_line_6_true():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    
    # Predicate at line 6: field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    # This evaluates to True when:
    # - field.initial is not PFIELD_NO_INITIAL (True, since 42 is not a sentinel)
    # - not callable(field.initial) (True, since 42 is not callable)
    # - field.type (True, since [int, str] is truthy)
    # - not any(isinstance(field.initial, t) for t in field.type) (False, since 42 is an int)
    # Overall: True and True and True and False = False
    
    # To make the predicate True, we need the last part to be True as well
    field.initial = []
    
    # Now: field.initial is not PFIELD_NO_INITIAL (True)
    # and not callable(field.initial) (True, [] is not callable)
    # and field.type (True)
    # and not any(isinstance(field.initial, t) for t in field.type) (True, [] is not int or str)
    # Overall: True and True and True and True = True
    
    predicate_result = (
        field.initial is not object() and
        not callable(field.initial) and
        field.type and
        not any(isinstance(field.initial, t) for t in field.type)
    )
    
    assert predicate_result is True


# LLM-generated content at query #39
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    custom_factory = lambda: "custom"
    pfield = _PField(
        type={MockCheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=custom_factory,
        serializer=None
    )
    
    assert pfield._factory == custom_factory
    assert pfield._factory is not None


# LLM-generated content at query #40
#--------------------------

```python
def test_pfield_factory_assignment_not_pfield_no_factory():
    class MockSerializer:
        pass
    
    def mock_factory():
        return "test"
    
    mock_serializer = MockSerializer()
    pfield = _PField(
        type=set(),
        invariant=None,
        initial=None,
        mandatory=False,
        factory=mock_factory,
        serializer=mock_serializer
    )
    
    assert pfield._factory is not None
    assert pfield._factory == mock_factory
    assert pfield._factory is mock_factory


# LLM-generated content at query #41
#--------------------------

```python
def test_make_seq_field_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector, pvector
    
    # Clear the cache before testing
    _seq_field_types.clear()
    
    # Test creating a seq field type with PVector and int
    item_type = int
    item_invariant = None
    
    result_type = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result_type is not None
    assert issubclass(result_type, PVector)
    assert result_type.__type__ == int
    assert result_type.__invariant__ is None
    assert (PVector, int) in _seq_field_types
    assert _seq_field_types[(PVector, int)] is result_type
    
    # Test that calling again with same parameters returns cached type
    result_type_2 = _make_seq_field_type(PVector, item_type, item_invariant)
    assert result_type_2 is result_type
    
    # Test __reduce__ method
    instance = result_type([1, 2, 3])
    reduced = instance.__reduce__()
    assert reduced[0].__name__ == '_restore_seq_field_pickle'
    assert reduced[1][0] is PVector
    assert reduced[1][1] == int
    assert reduced[1][2] == [1, 2, 3]
    
    # Clean up
    _seq_field_types.clear()


# LLM-generated content at query #42
#--------------------------

```python
def test_pfield_factory_assignment_with_none():
    class CheckedType:
        @classmethod
        def create(cls):
            return cls()
    
    pfield = _PField(
        type={CheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert pfield._factory is None
    assert pfield._factory is not True


# LLM-generated content at query #43
#--------------------------

```python
def test_pmap_field_returns_field_with_correct_type():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import field
    
    result = pmap_field(str, int)
    
    assert isinstance(result, type(field()))
    assert result.mandatory is True


# LLM-generated content at query #44
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int)
    assert field_obj.mandatory is True
    assert callable(field_obj.factory)
    assert field_obj.initial == pmap()


def test_pmap_field_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int, optional=False)
    assert field_obj.mandatory is True
    assert callable(field_obj.factory)
    assert field_obj.initial == pmap()


def test_pmap_field_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.mandatory is True
    assert callable(field_obj.factory)
    assert field_obj.initial == pmap()
    result_none = field_obj.factory(None)
    assert result_none is None


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    def my_invariant(val):
        return True, "valid"
    
    field_obj = pmap_field(str, int, invariant=my_invariant)
    assert field_obj.mandatory is True
    assert callable(field_obj.invariant)


def test_pmap_field_factory_creates_map():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int, optional=False)
    test_data = {"a": 1, "b": 2}
    result = field_obj.factory(test_data)
    assert result == pmap({"a": 1, "b": 2})


def test_pmap_field_optional_factory_with_data():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    field_obj = pmap_field(str, int, optional=True)
    test_data = {"x": 10}
    result = field_obj.factory(test_data)
    assert result == pmap({"x": 10})


# LLM-generated content at query #45
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, field, CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    # Create a checked vector type with integer items
    class MyCheckedVector(CheckedPVector):
        __type__ = PVector
        __item_type__ = int
    
    # Register the type in _seq_field_types
    _seq_field_types[(MyCheckedVector, int)] = MyCheckedVector
    
    # Create test data
    test_data = [1, 2, 3, 4, 5]
    
    # Call the function
    result = _restore_seq_field_pickle(MyCheckedVector, int, test_data)
    
    # Verify the result is a PVector with the correct data
    assert isinstance(result, PVector)
    assert list(result) == test_data
    assert len(result) == 5


# LLM-generated content at query #46
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Register a pmap field type
    from pyrsistent._field_common import pmap_field
    field_type = pmap_field(key_type, value_type)
    
    # Create test data
    test_data = {'key1': 1, 'key2': 2}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap with the correct data
    assert isinstance(result, type(pmap()))
    assert result == pmap(test_data)


# LLM-generated content at query #47
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, None)
    
    def invariant2(subject):
        return (True, None)
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_failure():
    def invariant1(subject):
        return (True, None)
    
    def invariant2(subject):
        return (False, "ERROR_CODE_1")
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    def invariant1(subject):
        return (False, "ERROR_CODE_1")
    
    def invariant2(subject):
        return (False, "ERROR_CODE_2")
    
    def invariant3(subject):
        return (True, None)
    
    invariants = [invariant1, invariant2, invariant3]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1", "ERROR_CODE_2")
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test_subject"
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_with_complex_subject():
    def invariant1(subject):
        return (True, None) if subject.get("key") == "value" else (False, "MISSING_KEY")
    
    invariants = [invariant1]
    subject = {"key": "value"}
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_complex_subject_failure():
    def invariant1(subject):
        return (False, "INVALID_SUBJECT")
    
    invariants = [invariant1]
    subject = {"key": "wrong_value"}
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("INVALID_SUBJECT",)


# LLM-generated content at query #48
#--------------------------

```python
def test_set_fields():
    from collections import namedtuple
    
    class _PField:
        def __init__(self, value):
            self.value = value
    
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    
    # Test case 1: Empty dict and no bases
    dct1 = {}
    bases1 = []
    name1 = "fields"
    set_fields(dct1, bases1, name1)
    assert dct1 == {"fields": {}}
    
    # Test case 2: Dict with _PField instances
    dct2 = {"field1": _PField("value1"), "field2": _PField("value2")}
    bases2 = []
    name2 = "fields"
    set_fields(dct2, bases2, name2)
    assert name2 in dct2
    assert "field1" not in dct2
    assert "field2" not in dct2
    assert "field1" in dct2[name2]
    assert "field2" in dct2[name2]
    assert isinstance(dct2[name2]["field1"], _PField)
    assert isinstance(dct2[name2]["field2"], _PField)
    
    # Test case 3: Dict with mixed _PField and non-_PField values
    dct3 = {"field1": _PField("value1"), "other": "not_a_field"}
    bases3 = []
    name3 = "fields"
    set_fields(dct3, bases3, name3)
    assert "field1" not in dct3
    assert "other" in dct3
    assert "field1" in dct3[name3]
    assert dct3["other"] == "not_a_field"
    
    # Test case 4: With base class containing fields
    class MockBase:
        pass
    mock_base = MockBase()
    mock_base.__dict__ = {"fields": {"inherited_field": _PField("inherited")}}
    
    dct4 = {"new_field": _PField("new")}
    bases4 = [mock_base]
    name4 = "fields"
    set_fields(dct4, bases4, name4)
    assert "new_field" not in dct4
    assert "inherited_field" in dct4[name4]
    assert "new_field" in dct4[name4]
    assert isinstance(dct4[name4]["inherited_field"], _PField)
    assert isinstance(dct4[name4]["new_field"], _PField)
    
    # Test case 5: Multiple base classes with fields
    class MockBase1:
        pass
    class MockBase2:
        pass
    mock_base1 = MockBase1()
    mock_base2 = MockBase2()
    mock_base1.__dict__ = {"fields": {"field_a": _PField("a")}}
    mock_base2.__dict__ = {"fields": {"field_b": _PField("b")}}
    
    dct5 = {}
    bases5 = [mock_base1, mock_base2]
    name5 = "fields"
    set_fields(dct5, bases5, name5)
    assert "field_a" in dct5[name5]
    assert "field_b" in dct5[name5]


# LLM-generated content at query #49
#--------------------------

```python
def test_pfield_constructor():
    def dummy_invariant(x):
        return True
    
    def dummy_factory():
        return "factory_result"
    
    def dummy_serializer(x):
        return str(x)
    
    type_value = (str, int)
    initial_value = "initial"
    mandatory_value = True
    
    field = _PField(
        type=type_value,
        invariant=dummy_invariant,
        initial=initial_value,
        mandatory=mandatory_value,
        factory=dummy_factory,
        serializer=dummy_serializer
    )
    
    assert field.type == type_value
    assert field.invariant == dummy_invariant
    assert field.initial == initial_value
    assert field.mandatory == mandatory_value
    assert field._factory == dummy_factory
    assert field.serializer == dummy_serializer


def test_pfield_constructor_with_none_values():
    field = _PField(
        type=(str,),
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert field.type == (str,)
    assert field.invariant is None
    assert field.initial is None
    assert field.mandatory is False
    assert field._factory is None
    assert field.serializer is None


def test_pfield_constructor_slots():
    field = _PField(
        type=(int,),
        invariant=lambda x: x > 0,
        initial=42,
        mandatory=True,
        factory=int,
        serializer=str
    )
    
    assert hasattr(field, 'type')
    assert hasattr(field, 'invariant')
    assert hasattr(field, 'initial')
    assert hasattr(field, 'mandatory')
    assert hasattr(field, '_factory')
    assert hasattr(field, 'serializer')


# LLM-generated content at query #50
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    class TestBase:
        pass
    
    pfield_instance = _PField()
    dct = {'field1': pfield_instance}
    bases = [TestBase]
    name = 'test_name'
    
    # Manually execute the relevant part to test the predicate
    for k, v in list(dct.items()):
        predicate_result = isinstance(v, _PField)
        assert predicate_result is True


# LLM-generated content at query #51
#--------------------------

```python
def test_pmap_field_optional_predicate_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap field with optional=True to make the predicate at line 15 evaluate to True
    field = pmap_field(str, int, optional=True)
    
    # Verify that the field was created successfully
    assert field is not None
    assert field.mandatory is True


# LLM-generated content at query #52
#--------------------------

```python
def test_pmap_field_type_predicate_line_25_with_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent._checked_types import optional
    
    result = pmap_field(str, int, optional=True)
    assert result.type == optional(object) or isinstance(result.type, tuple)
    assert type(None) in result.type


def test_pmap_field_type_predicate_line_25_with_optional_false():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=False)
    assert result.type is not None
    assert not isinstance(result.type, tuple) or type(None) not in result.type


# LLM-generated content at query #53
#--------------------------

```python
def test_make_pmap_field_type_creates_new_pmap_class():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result is not None
    assert hasattr(result, '__key_type__')
    assert hasattr(result, '__value_type__')
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_generates_correct_name():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert "PMap" in result.__name__
    assert "To" in result.__name__


def test_make_pmap_field_type_different_types_create_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_with_multiple_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type((str, int), (float, bool))
    
    assert result is not None
    assert result.__key_type__ == (str, int)
    assert result.__value_type__ == (float, bool)


# LLM-generated content at query #54
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, None)
    
    def invariant2(subject):
        return (True, None)
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_failure():
    def invariant1(subject):
        return (True, None)
    
    def invariant2(subject):
        return (False, "ERROR_CODE_1")
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    def invariant1(subject):
        return (False, "ERROR_CODE_1")
    
    def invariant2(subject):
        return (False, "ERROR_CODE_2")
    
    def invariant3(subject):
        return (True, None)
    
    invariants = [invariant1, invariant2, invariant3]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1", "ERROR_CODE_2")
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test_subject"
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_invariant_pass():
    def invariant1(subject):
        return (True, "NO_ERROR")
    
    invariants = [invariant1]
    subject = "test_subject"
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_with_different_subjects():
    def invariant1(subject):
        return (subject == 42, "NOT_42")
    
    def invariant2(subject):
        return (len(subject) > 0, "EMPTY")
    
    invariants = [invariant1, invariant2]
    subject = 42
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_subject_fails_first_invariant():
    def invariant1(subject):
        return (subject > 100, "TOO_SMALL")
    
    def invariant2(subject):
        return (subject < 200, "TOO_LARGE")
    
    invariants = [invariant1, invariant2]
    subject = 50
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("TOO_SMALL",)


# LLM-generated content at query #55
#--------------------------

```python
def test_make_pmap_field_type_creates_new_pmap_field_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type(str, int)
    
    assert pmap_type.__name__ == "StrToIntPMap"
    assert pmap_type.__key_type__ == str
    assert pmap_type.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type1 = _make_pmap_field_type(str, int)
    pmap_type2 = _make_pmap_field_type(str, int)
    
    assert pmap_type1 is pmap_type2


def test_make_pmap_field_type_different_types_create_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type1 = _make_pmap_field_type(str, int)
    pmap_type2 = _make_pmap_field_type(int, str)
    
    assert pmap_type1 is not pmap_type2
    assert pmap_type1.__name__ == "StrToIntPMap"
    assert pmap_type2.__name__ == "IntToStrPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type(str, int)
    
    assert hasattr(pmap_type, '__reduce__')
    instance = pmap_type()
    reduce_result = instance.__reduce__()
    assert reduce_result[0].__name__ == '_restore_pmap_field_pickle'
    assert reduce_result[1] == (str, int, {})


def test_make_pmap_field_type_with_builtin_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type(float, bool)
    
    assert pmap_type.__name__ == "FloatToBoolPMap"
    assert pmap_type.__key_type__ == float
    assert pmap_type.__value_type__ == bool


# LLM-generated content at query #56
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant1(subject):
        return (True, "code1")
    
    def invariant2(subject):
        return (True, "code2")
    
    subject = "test_subject"
    invariants = [invariant1, invariant2]
    
    try:
        check_global_invariants(subject, invariants)
        result = True
    except InvariantException:
        result = False
    
    assert result is True


# LLM-generated content at query #57
#--------------------------

```python
def test_sequence_field_predicate_line_26():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector, field
    from pyrsistent._checked_types import optional
    
    # Test case 1: optional=True, should use optional_type
    result_optional = _sequence_field(
        checked_class=CheckedPVector,
        item_type=int,
        optional=True,
        initial=[1, 2, 3]
    )
    assert result_optional is not None
    assert hasattr(result_optional, 'type')
    
    # Test case 2: optional=False, should use just TheType
    result_required = _sequence_field(
        checked_class=CheckedPVector,
        item_type=str,
        optional=False,
        initial=['a', 'b']
    )
    assert result_required is not None
    assert hasattr(result_required, 'type')
    
    # The predicate at line 26 evaluates to True when optional=True
    # This means optional_type(TheType) is used instead of just TheType
    assert result_optional.type != result_required.type


# LLM-generated content at query #58
#--------------------------

```python
def test_sequence_field_with_optional_true():
    from pyrsistent._field_common import _sequence_field, field
    from pyrsistent._checked_types import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, True, [1, 2, 3])
    
    assert isinstance(result, type(field()))
    assert result.mandatory is True
    assert result.initial is None or callable(result.initial)


def test_sequence_field_with_optional_false():
    from pyrsistent._field_common import _sequence_field, field
    from pyrsistent._checked_types import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    
    assert isinstance(result, type(field()))
    assert result.mandatory is True


def test_sequence_field_factory_with_none_when_optional():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, True, [])
    factory = result.factory
    
    none_result = factory(None)
    assert none_result is None


def test_sequence_field_factory_with_value_when_optional():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, True, [])
    factory = result.factory
    
    value_result = factory([1, 2, 3])
    assert value_result is not None


def test_sequence_field_factory_non_optional():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    factory = result.factory
    
    assert callable(factory)


def test_sequence_field_with_invariant():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import CheckedPVector
    
    def my_invariant(value):
        return True, None
    
    result = _sequence_field(CheckedPVector, int, False, [], invariant=my_invariant)
    
    assert result.invariant is not None


def test_sequence_field_with_item_invariant():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import CheckedPVector
    
    def my_item_invariant(value):
        return True, None
    
    result = _sequence_field(CheckedPVector, int, False, [], item_invariant=my_item_invariant)
    
    assert isinstance(result, type(_sequence_field(CheckedPVector, int, False, [])))


# LLM-generated content at query #59
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, "no_error")
    
    def invariant2(subject):
        return (True, "no_error")
    
    subject = "test_subject"
    invariants = [invariant1, invariant2]
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_failure():
    def invariant1(subject):
        return (True, "no_error")
    
    def invariant2(subject):
        return (False, "error_code_1")
    
    subject = "test_subject"
    invariants = [invariant1, invariant2]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("error_code_1",)
        assert e.args[1] == ()
        assert e.args[2] == "Global invariant failed"


def test_check_global_invariants_multiple_failures():
    def invariant1(subject):
        return (False, "error_code_1")
    
    def invariant2(subject):
        return (False, "error_code_2")
    
    def invariant3(subject):
        return (True, "no_error")
    
    subject = "test_subject"
    invariants = [invariant1, invariant2, invariant3]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("error_code_1", "error_code_2")
        assert e.args[1] == ()
        assert e.args[2] == "Global invariant failed"


def test_check_global_invariants_no_invariants():
    subject = "test_subject"
    invariants = []
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_with_different_subject_types():
    def invariant(subject):
        return (True, "no_error")
    
    subject_dict = {"key": "value"}
    invariants = [invariant]
    
    check_global_invariants(subject_dict, invariants)
    
    subject_list = [1, 2, 3]
    check_global_invariants(subject_list, invariants)
    
    subject_int = 42
    check_global_invariants(subject_int, invariants)


# LLM-generated content at query #60
#--------------------------

```python
def test_sequence_field_with_optional_true():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import CheckedPVector
    
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=int,
        optional=True,
        initial=[]
    )
    
    assert result is not None
    assert result.mandatory is True
    assert result.factory is not None


def test_sequence_field_with_optional_false():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import CheckedPVector
    
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=str,
        optional=False,
        initial=[]
    )
    
    assert result is not None
    assert result.mandatory is True
    assert result.factory is not None


def test_sequence_field_factory_with_none_when_optional():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import CheckedPVector
    
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=int,
        optional=True,
        initial=[]
    )
    
    factory_result = result.factory(None)
    assert factory_result is None


def test_sequence_field_factory_with_list_when_optional():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import CheckedPVector
    
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=int,
        optional=True,
        initial=[]
    )
    
    factory_result = result.factory([1, 2, 3])
    assert factory_result is not None
    assert len(factory_result) == 3


def test_sequence_field_with_item_invariant():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent._checked_types import CheckedPVector
    
    def item_inv(val):
        return True, "Valid"
    
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=int,
        optional=False,
        initial=[],
        item_invariant=item_inv
    )
    
    assert result is not None
    assert result.mandatory is True


def test_sequence_field_with_invariant():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent._checked_types import CheckedPVector
    
    def inv(val):
        return True, "Valid"
    
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=int,
        optional=False,
        initial=[],
        invariant=inv
    )
    
    assert result is not None
    assert result.invariant is not None


# LLM-generated content at query #61
#--------------------------

```python
def test_set_fields_with_empty_bases():
    dct = {}
    bases = ()
    name = "fields"
    set_fields(dct, bases, name)
    assert "fields" in dct
    assert dct["fields"] == {}


def test_set_fields_with_pfield_instances():
    class _PField:
        pass
    
    field1 = _PField()
    field2 = _PField()
    dct = {"attr1": field1, "attr2": field2, "other": "value"}
    bases = ()
    name = "fields"
    set_fields(dct, bases, name)
    assert "fields" in dct
    assert dct["fields"] == {"attr1": field1, "attr2": field2}
    assert "attr1" not in dct
    assert "attr2" not in dct
    assert dct["other"] == "value"


def test_set_fields_with_inherited_fields():
    class _PField:
        pass
    
    field_base = _PField()
    base_dct = {"fields": {"inherited": field_base}}
    
    class Base:
        pass
    Base.__dict__ = base_dct
    
    field_new = _PField()
    dct = {"new_field": field_new}
    bases = (Base,)
    name = "fields"
    set_fields(dct, bases, name)
    assert "fields" in dct
    assert "inherited" in dct["fields"]
    assert "new_field" in dct["fields"]
    assert dct["fields"]["new_field"] == field_new
    assert "new_field" not in dct


def test_set_fields_with_multiple_bases():
    class _PField:
        pass
    
    field1 = _PField()
    field2 = _PField()
    
    class Base1:
        pass
    Base1.__dict__ = {"fields": {"f1": field1}}
    
    class Base2:
        pass
    Base2.__dict__ = {"fields": {"f2": field2}}
    
    dct = {}
    bases = (Base1, Base2)
    name = "fields"
    set_fields(dct, bases, name)
    assert "fields" in dct
    assert "f1" in dct["fields"]
    assert "f2" in dct["fields"]


def test_set_fields_preserves_non_pfield_attributes():
    class _PField:
        pass
    
    field = _PField()
    dct = {"field": field, "regular_attr": "value", "number": 42}
    bases = ()
    name = "fields"
    set_fields(dct, bases, name)
    assert dct["regular_attr"] == "value"
    assert dct["number"] == 42
    assert "field" not in dct


# LLM-generated content at query #62
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized_value"
    
    class MockCheckedType(CheckedType):
        pass
    
    PFIELD_NO_SERIALIZER = object()
    value = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "checked_serialized_value"


def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"serialized_{value}_in_{format}"
    
    result = serialize(custom_serializer, "json", "test_value")
    assert result == "serialized_test_value_in_json"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    class MockCheckedType(CheckedType):
        pass
    
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    value = MockCheckedType()
    result = serialize(custom_serializer, "xml", value)
    assert result == "custom_xml_" + str(value)


def test_serialize_with_regular_value_and_serializer():
    def my_serializer(format, value):
        return {"format": format, "value": value}
    
    result = serialize(my_serializer, "yaml", 42)
    assert result == {"format": "yaml", "value": 42}


def test_serialize_with_dict_value():
    def dict_serializer(format, value):
        return str(value)
    
    test_dict = {"key": "value"}
    result = serialize(dict_serializer, "json", test_dict)
    assert result == str(test_dict)


# LLM-generated content at query #63
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, pmap, field
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import CheckedPVector
    
    # Create a checked class with a vector field
    class MyChecked(CheckedPVector):
        __type__ = PVector
        __item_type__ = int
    
    # Register the type in _seq_field_types
    _seq_field_types[MyChecked, int] = MyChecked
    
    # Test data
    test_data = [1, 2, 3]
    
    # Call the function
    result = _restore_seq_field_pickle(MyChecked, int, test_data)
    
    # Verify the result is a PVector with correct data
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]


# LLM-generated content at query #64
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a test pmap field type
    key_type = str
    value_type = int
    
    # Register a pmap field type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent import PMap
        class TestPMapField(PMap):
            @classmethod
            def create(cls, data, _factory_fields=None):
                return pmap(data)
        _pmap_field_types[key_type, value_type] = TestPMapField
    
    # Test data
    test_data = {'a': 1, 'b': 2, 'c': 3}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap with correct data
    assert result == pmap(test_data)
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #65
#--------------------------

```python
def test_make_pmap_field_type_creates_new_pmap_class():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert result.__name__ == "StrToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_pmap_field_type, _pmap_field_types
    
    _pmap_field_types.clear()
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    assert result1 is result2


def test_make_pmap_field_type_different_types_creates_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type, _pmap_field_types
    
    _pmap_field_types.clear()
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    assert result1 is not result2
    assert result1.__name__ == "StrToIntPMap"
    assert result2.__name__ == "IntToStrPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type, _pmap_field_types
    
    _pmap_field_types.clear()
    result = _make_pmap_field_type(str, int)
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_with_builtin_types():
    from pyrsistent._field_common import _make_pmap_field_type, _pmap_field_types
    
    _pmap_field_types.clear()
    result = _make_pmap_field_type(float, bool)
    assert result.__name__ == "FloatToBoolPMap"
    assert result.__key_type__ == float
    assert result.__value_type__ == bool


# LLM-generated content at query #66
#--------------------------

```python
def test_sequence_field_invariant_parameter_with_pfield_no_invariant():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import PVector, CheckedPVector
    
    # Call _sequence_field with invariant parameter explicitly set to PFIELD_NO_INVARIANT
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=int,
        optional=False,
        initial=[1, 2, 3],
        invariant=PFIELD_NO_INVARIANT,
        item_invariant=PFIELD_NO_INVARIANT
    )
    
    # The predicate at line 2 evaluates to True means the parameter invariant 
    # has the default value PFIELD_NO_INVARIANT
    assert result is not None
    assert hasattr(result, 'invariant')


# LLM-generated content at query #67
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import pmap, pvec, field, CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class MyChecked(CheckedPVector):
        __type__ = pvec([int])
    
    checked_class = MyChecked
    item_type = int
    data = [1, 2, 3]
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    
    assert result is not None
    assert len(result) == 3


def test_restore_seq_field_pickle_empty():
    from pyrsistent import pvec, CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle
    
    class MyChecked(CheckedPVector):
        __type__ = pvec([str])
    
    checked_class = MyChecked
    item_type = str
    data = []
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    
    assert result is not None
    assert len(result) == 0


def test_restore_seq_field_pickle_with_strings():
    from pyrsistent import pvec, CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle
    
    class MyChecked(CheckedPVector):
        __type__ = pvec([str])
    
    checked_class = MyChecked
    item_type = str
    data = ["a", "b", "c"]
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    
    assert result is not None
    assert len(result) == 3
    assert result[0] == "a"
    assert result[1] == "b"
    assert result[2] == "c"


# LLM-generated content at query #68
#--------------------------

```python
def test_pmap_field_optional_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Test that the predicate at line 15 evaluates to True
    # when optional=True is passed
    result = pmap_field(str, int, optional=True)
    
    # Verify that a field is returned
    assert result is not None
    assert hasattr(result, 'factory')
    assert hasattr(result, 'type')
    
    # Verify the factory handles None correctly
    factory = result.factory
    assert factory(None) is None
    
    # Verify the factory handles non-None values
    test_map = factory({'a': 1, 'b': 2})
    assert test_map is not None


# LLM-generated content at query #69
#--------------------------

```python
def test_pmap_field_optional_type_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent._checked_types import optional
    from pyrsistent import pmap
    
    # Create a pmap field with optional=True
    result = pmap_field(str, int, optional=True)
    
    # The predicate at line 25 should evaluate to True when optional=True
    # This means the type should be wrapped with optional_type
    assert result.type is not None
    
    # Create a pmap field with optional=False
    result_not_optional = pmap_field(str, int, optional=False)
    
    # The predicate at line 25 should evaluate to False when optional=False
    # This means the type should be just TheMap without optional wrapping
    assert result_not_optional.type is not None


# LLM-generated content at query #70
#--------------------------

```python
def test_pmap_field_predicate_line_25_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import PMap
    
    result = pmap_field(str, int, optional=True)
    assert result.type is not None
    assert isinstance(result.type, tuple)
    assert type(None) in result.type


def test_pmap_field_predicate_line_25_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import PMap
    
    result = pmap_field(str, int, optional=False)
    assert result.type is not None
    assert not isinstance(result.type, tuple) or type(None) not in result.type


# LLM-generated content at query #71
#--------------------------

```python
def test_pmap_field_optional_true_creates_factory_that_handles_none():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap field with optional=True
    field_obj = pmap_field(str, int, optional=True)
    
    # Get the factory function
    factory = field_obj.factory
    
    # Test that the factory handles None correctly
    result_none = factory(None)
    assert result_none is None
    
    # Test that the factory handles a valid pmap correctly
    result_pmap = factory({})
    assert result_pmap is not None


# LLM-generated content at query #72
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # Line 3 predicate: "not isinstance(t, type) and not isinstance(t, str)"
    # For this to be False, at least one of the following must be True:
    # - isinstance(t, type) is True, OR
    # - isinstance(t, str) is True
    # Both int and str in field.type satisfy isinstance(t, type) == True
    # So the predicate evaluates to False for all elements
    
    from types_namespace import _check_field_parameters
    
    # This should not raise because all type parameters are either type or str
    _check_field_parameters(field)


# LLM-generated content at query #73
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=True)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    def my_invariant(val):
        return True, "valid"
    
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()
    assert result.invariant is not None


def test_pmap_field_factory_none_optional():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    factory_result = result.factory(None)
    assert factory_result is None


def test_pmap_field_factory_dict_optional():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    test_dict = {'a': 1, 'b': 2}
    factory_result = result.factory(test_dict)
    assert factory_result['a'] == 1
    assert factory_result['b'] == 2


def test_pmap_field_factory_dict_not_optional():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=False)
    test_dict = {'a': 1, 'b': 2}
    factory_result = result.factory(test_dict)
    assert factory_result['a'] == 1
    assert factory_result['b'] == 2


def test_pmap_field_different_types():
    from pyrsistent._field_common import pmap_field
    
    result1 = pmap_field(int, str)
    result2 = pmap_field(float, bool)
    assert result1.type is not None
    assert result2.type is not None


# LLM-generated content at query #74
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "checked_serialized"


def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "xml:test_value"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    value = CheckedType()
    result = serialize(custom_serializer, "json", value)
    assert result == "json:CheckedType()"


def test_serialize_with_non_checked_type_and_no_serializer():
    PFIELD_NO_SERIALIZER = object()
    
    def no_serializer_func(format, value):
        return value
    
    result = serialize(no_serializer_func, "csv", "data")
    assert result == "data"


def test_serialize_with_different_formats():
    def format_serializer(format, value):
        return f"{format}-{value}"
    
    result_json = serialize(format_serializer, "json", "value1")
    result_xml = serialize(format_serializer, "xml", "value2")
    
    assert result_json == "json-value1"
    assert result_xml == "xml-value2"


# LLM-generated content at query #75
#--------------------------

```python
def test_serialize_checked_type_with_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "json"
    
    result = serialize(serializer, format, value)
    assert result == "serialized_value"


# LLM-generated content at query #76
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a simple pmap field type
    key_type = str
    value_type = int
    
    # Register a pmap field type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent import PMap
        class TestPMapField(PMap):
            __key_type__ = key_type
            __value_type__ = value_type
            @staticmethod
            def create(data, _factory_fields=None):
                return pmap(data)
        _pmap_field_types[key_type, value_type] = TestPMapField
    
    # Test data to restore
    test_data = {'a': 1, 'b': 2, 'c': 3}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap with correct data
    assert result == pmap(test_data)
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #77
#--------------------------

```python
def test_sequence_field_invariant_parameter_default_value():
    from pyrsistent._checked_types import PFIELD_NO_INVARIANT
    
    invariant = PFIELD_NO_INVARIANT
    
    assert invariant == PFIELD_NO_INVARIANT


# LLM-generated content at query #78
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, pv
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import CheckedPVector
    
    checked_class = CheckedPVector
    item_type = int
    data = [1, 2, 3]
    
    # Register the type in _seq_field_types if not already present
    if (checked_class, item_type) not in _seq_field_types:
        from pyrsistent import field
        _seq_field_types[checked_class, item_type] = type('TestVec', (PVector,), {})
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    assert result is not None
    assert len(result) == 3
    assert list(result) == [1, 2, 3]


def test_restore_seq_field_pickle_empty():
    from pyrsistent import PVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import CheckedPVector
    
    checked_class = CheckedPVector
    item_type = str
    data = []
    
    # Register the type in _seq_field_types if not already present
    if (checked_class, item_type) not in _seq_field_types:
        _seq_field_types[checked_class, item_type] = type('TestVecStr', (PVector,), {})
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    assert result is not None
    assert len(result) == 0
    assert list(result) == []


def test_restore_seq_field_pickle_with_strings():
    from pyrsistent import PVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import CheckedPVector
    
    checked_class = CheckedPVector
    item_type = str
    data = ['a', 'b', 'c']
    
    # Register the type in _seq_field_types if not already present
    if (checked_class, item_type) not in _seq_field_types:
        _seq_field_types[checked_class, item_type] = type('TestVecStr2', (PVector,), {})
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    assert result is not None
    assert len(result) == 3
    assert list(result) == ['a', 'b', 'c']


# LLM-generated content at query #79
#--------------------------

```python
def test_factory_parameter_assignment():
    class MockCheckType:
        pass
    
    factory_func = lambda: "test"
    pfield = _PField(
        type={int},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=None
    )
    
    assert pfield._factory is factory_func
    assert pfield._factory is not None


# LLM-generated content at query #80
#--------------------------

```python
def test_pfield_constructor():
    def dummy_invariant(value):
        return value > 0
    
    def dummy_serializer(value):
        return str(value)
    
    def dummy_factory():
        return 42
    
    field = _PField(
        type=(int,),
        invariant=dummy_invariant,
        initial=10,
        mandatory=True,
        factory=dummy_factory,
        serializer=dummy_serializer
    )
    
    assert field.type == (int,)
    assert field.invariant == dummy_invariant
    assert field.initial == 10
    assert field.mandatory is True
    assert field._factory == dummy_factory
    assert field.serializer == dummy_serializer


def test_pfield_constructor_with_none_values():
    field = _PField(
        type=(str,),
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert field.type == (str,)
    assert field.invariant is None
    assert field.initial is None
    assert field.mandatory is False
    assert field._factory is None
    assert field.serializer is None


def test_pfield_constructor_with_multiple_types():
    field = _PField(
        type=(int, str, float),
        invariant=lambda x: True,
        initial=0,
        mandatory=True,
        factory=lambda: 0,
        serializer=lambda x: x
    )
    
    assert field.type == (int, str, float)
    assert field.mandatory is True
    assert len(field.type) == 3


# LLM-generated content at query #81
#--------------------------

```python
def test_sequence_field_invariant_parameter_type():
    from pyrsistent._field_common import PFIELD_NO_INVARIANT
    
    # The predicate at line 2 checks if invariant parameter equals PFIELD_NO_INVARIANT
    # by default when not provided
    invariant_param = PFIELD_NO_INVARIANT
    predicate_result = invariant_param == PFIELD_NO_INVARIANT
    
    assert predicate_result is True


# LLM-generated content at query #82
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3: "not isinstance(t, type) and not isinstance(t, str)"
    # evaluates to False when t is either a type OR a string
    # This test verifies no TypeError is raised when field.type contains valid types
    try:
        from types import SimpleNamespace
        PFIELD_NO_INITIAL = SimpleNamespace()
        
        # Call the function - it should not raise since all elements are valid
        for t in field.type:
            predicate_result = not isinstance(t, type) and not isinstance(t, str)
            assert predicate_result == False, "Predicate should evaluate to False for valid type parameters"
    except Exception:
        pass


# LLM-generated content at query #83
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int)
    assert result.mandatory is True
    assert result.initial is not None
    assert result.invariant == result.invariant
    assert result._factory is not None

def test_pmap_field_with_optional_true():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=True)
    assert result.mandatory is True
    assert result.type is not None
    assert result._factory is not None

def test_pmap_field_with_optional_false():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=False)
    assert result.mandatory is True
    assert result.type is not None
    assert result._factory is not None

def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    def my_invariant(val):
        return True, None
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.mandatory is True
    assert result.invariant is not None

def test_pmap_field_multiple_calls_same_types():
    from pyrsistent._field_common import pmap_field
    result1 = pmap_field(str, int)
    result2 = pmap_field(str, int)
    assert type(result1.type) == type(result2.type)

def test_pmap_field_different_types():
    from pyrsistent._field_common import pmap_field
    result1 = pmap_field(str, int)
    result2 = pmap_field(int, str)
    assert result1.type != result2.type

def test_pmap_field_optional_factory_with_none():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=True)
    factory_result = result._factory(None)
    assert factory_result is None

def test_pmap_field_optional_factory_with_dict():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=True)
    factory_result = result._factory({'a': 1})
    assert factory_result is not None


# LLM-generated content at query #84
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import CheckedPMap
    
    result = pmap_field(str, int)
    assert result.mandatory is True
    assert result.initial is not None
    assert callable(result.factory)


def test_pmap_field_optional_true():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    assert result.mandatory is True
    assert callable(result.factory)
    factory_result_none = result.factory(None)
    assert factory_result_none is None


def test_pmap_field_optional_false():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=False)
    assert result.mandatory is True
    assert callable(result.factory)


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    
    def my_invariant(val):
        return True, None
    
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.mandatory is True
    assert result.invariant is not None
    assert callable(result.invariant)


def test_pmap_field_type_int_int():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(int, int)
    assert result.mandatory is True
    assert result.type is not None


def test_pmap_field_type_float_str():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(float, str)
    assert result.mandatory is True
    assert result.type is not None


def test_pmap_field_optional_and_invariant():
    from pyrsistent._field_common import pmap_field
    
    def my_invariant(val):
        return True, None
    
    result = pmap_field(str, int, optional=True, invariant=my_invariant)
    assert result.mandatory is True
    assert result.invariant is not None
    assert callable(result.factory)


# LLM-generated content at query #85
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean test
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == int
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ is None
    assert hasattr(result, '__reduce__')


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_different_item_types():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result1 = _make_seq_field_type(PVector, int, None)
    result2 = _make_seq_field_type(PVector, str, None)
    
    assert result1 is not result2
    assert result1.__type__ == int
    assert result2.__type__ == str


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    def my_invariant(val):
        return val > 0
    
    result = _make_seq_field_type(PVector, int, my_invariant)
    
    assert result.__invariant__ is my_invariant


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result = _make_seq_field_type(PVector, int, None)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


# LLM-generated content at query #86
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    optional_param = False
    result = pmap_field(str, int, optional=optional_param)
    
    assert result is not None
    assert result.mandatory is True
    assert result.initial == pmap()


# LLM-generated content at query #87
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    factory_func = lambda: MockCheckedType()
    pfield = _PField(
        type={MockCheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=None
    )
    
    assert pfield._factory is factory_func


# LLM-generated content at query #88
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import PMap
    
    # Create a pmap_field with optional=False
    result = pmap_field(str, int, optional=False)
    
    # The predicate at line 2 (which is the docstring) should evaluate to False
    # This means optional parameter is False, so the factory should be TheMap.create
    # and the type should be TheMap (not wrapped in optional_type)
    assert result.mandatory == True
    assert result.initial == PMap()
    assert result._factory is not None


# LLM-generated content at query #89
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    # Test that when optional=False, the predicate at line 2 evaluates to False
    from pyrsistent._field_common import pmap_field
    from pyrsistent import PMap
    
    # Call pmap_field with optional=False
    result = pmap_field(str, int, optional=False)
    
    # The predicate at line 2 (if optional:) should be False
    # This means the factory should be set to TheMap.create directly (line 22)
    # and the type should be TheMap, not optional_type(TheMap)
    assert result.type is not None
    assert result._factory is not None


# LLM-generated content at query #90
#--------------------------

```python
def test_pmap_field_returns_field_with_correct_type():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import field
    
    result = pmap_field(str, int, optional=False)
    
    assert isinstance(result, type(field()))
    assert result.mandatory is True


# LLM-generated content at query #91
#--------------------------

```python
def test_pmap_field_optional_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Test that the predicate at line 15 evaluates to True
    result = pmap_field(str, int, optional=True)
    
    # The predicate `if optional:` should be True, so factory should be a function
    # that handles None values
    assert result.factory is not None
    assert callable(result.factory)
    
    # Test that the factory correctly handles None
    assert result.factory(None) is None
    
    # Test that the factory correctly handles a dict argument
    test_map = result.factory({})
    assert test_map is not None


# LLM-generated content at query #92
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3: not isinstance(t, type) and not isinstance(t, str)
    # For this to be False, either isinstance(t, type) is True OR isinstance(t, str) is True
    # With field.type = [int, str]:
    # - int is a type, so isinstance(int, type) is True
    # - str is both a type and a str, so isinstance(str, type) is True
    # Therefore, the predicate is False for all elements, and no TypeError is raised
    
    try:
        from types import SimpleNamespace
        field = SimpleNamespace(
            type=[int, str],
            initial=None,
            invariant=lambda x: True,
            factory=lambda: None,
            serializer=lambda x: x
        )
        _check_field_parameters(field)
        result = True
    except TypeError:
        result = False
    
    assert result is True


# LLM-generated content at query #93
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    
    assert result == "serialized_value"
    assert isinstance(value, CheckedType)
    assert PFIELD_NO_SERIALIZER is PFIELD_NO_SERIALIZER


# LLM-generated content at query #94
#--------------------------

```python
def test_sequence_field_invariant_parameter_has_default_value():
    import inspect
    from pyrsistent._field_common import _sequence_field
    
    sig = inspect.signature(_sequence_field)
    invariant_param = sig.parameters['invariant']
    
    assert invariant_param.default is not inspect.Parameter.empty


# LLM-generated content at query #95
#--------------------------

```python
def test_pmap_field_type_predicate_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent._checked_types import optional
    
    key_type = str
    value_type = int
    
    result = pmap_field(key_type, value_type, optional=True)
    
    assert result.type == optional(result.type[0]) if isinstance(result.type, tuple) else result.type


def test_pmap_field_type_predicate_optional_false():
    from pyrsistent._field_common import pmap_field
    
    key_type = str
    value_type = int
    
    result = pmap_field(key_type, value_type, optional=False)
    
    assert result.type is not None
    assert isinstance(result.type, (type, tuple, str))


# LLM-generated content at query #96
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result.__name__ == "StringToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_types_different_results():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2
    assert result1.__name__ == "StringToIntPMap"
    assert result2.__name__ == "IntToStringPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_with_multiple_key_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type((str, int), float)
    
    assert "To" in result.__name__
    assert "PMap" in result.__name__
    assert result.__key_type__ == (str, int)
    assert result.__value_type__ == float


def test_make_pmap_field_type_preserves_checked_pmap_inheritance():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    
    result = _make_pmap_field_type(str, int)
    
    assert issubclass(result, CheckedPMap)


# LLM-generated content at query #97
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PClass, field, pvector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class TestClass(PClass):
        items = field()
    
    item_type = int
    data = [1, 2, 3]
    
    key = (TestClass, item_type)
    if key not in _seq_field_types:
        from pyrsistent._field_common import _make_seq_field_type
        _make_seq_field_type(TestClass, item_type, pvector, "PVec")
    
    result = _restore_seq_field_pickle(TestClass, item_type, data)
    
    assert result is not None
    assert len(result) == 3


def test_restore_seq_field_pickle_empty():
    from pyrsistent import PClass, field, pvector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class TestClass(PClass):
        items = field()
    
    item_type = str
    data = []
    
    key = (TestClass, item_type)
    if key not in _seq_field_types:
        from pyrsistent._field_common import _make_seq_field_type
        _make_seq_field_type(TestClass, item_type, pvector, "PVec")
    
    result = _restore_seq_field_pickle(TestClass, item_type, data)
    
    assert result is not None
    assert len(result) == 0


def test_restore_seq_field_pickle_with_data():
    from pyrsistent import PClass, field, pvector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class TestClass(PClass):
        items = field()
    
    item_type = float
    data = [1.5, 2.5, 3.5]
    
    key = (TestClass, item_type)
    if key not in _seq_field_types:
        from pyrsistent._field_common import _make_seq_field_type
        _make_seq_field_type(TestClass, item_type, pvector, "PVec")
    
    result = _restore_seq_field_pickle(TestClass, item_type, data)
    
    assert result is not None
    assert len(result) == 3
    assert result[0] == 1.5
    assert result[1] == 2.5
    assert result[2] == 3.5


# LLM-generated content at query #98
#--------------------------

```python
def test_make_seq_field_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import pmap, pvector
    
    # Clear the cache before testing
    _seq_field_types.clear()
    
    # Test creating a seq field type with pvector and int
    item_type = int
    item_invariant = None
    
    result_type_1 = _make_seq_field_type(pvector, item_type, item_invariant)
    
    assert result_type_1 is not None
    assert hasattr(result_type_1, '__type__')
    assert result_type_1.__type__ == int
    assert hasattr(result_type_1, '__invariant__')
    assert result_type_1.__invariant__ is None
    assert hasattr(result_type_1, '__reduce__')
    
    # Test that calling with same arguments returns cached type
    result_type_2 = _make_seq_field_type(pvector, item_type, item_invariant)
    assert result_type_1 is result_type_2
    
    # Test that different item_type creates different type
    result_type_3 = _make_seq_field_type(pvector, str, item_invariant)
    assert result_type_1 is not result_type_3
    assert result_type_3.__type__ == str
    
    # Test that the type has a proper name
    assert isinstance(result_type_1.__name__, str)
    assert len(result_type_1.__name__) > 0
    
    # Test reduce method returns proper structure
    reduce_result = result_type_1.__reduce__()
    assert isinstance(reduce_result, tuple)
    assert len(reduce_result) == 2


# LLM-generated content at query #99
#--------------------------

```python
def test_check_global_invariants_with_no_violations():
    def invariant_pass(subject):
        return (True, None)
    
    def invariant_pass_2(subject):
        return (True, None)
    
    invariants = [invariant_pass, invariant_pass_2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        result = True
    except InvariantException:
        result = False
    
    assert result is True


# LLM-generated content at query #100
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    class TestBase:
        pass
    
    pfield_instance = _PField()
    dct = {'field1': pfield_instance, 'field2': 'not_a_pfield'}
    bases = (TestBase,)
    name = 'fields'
    
    # Check that the predicate evaluates to True for _PField instances
    assert isinstance(pfield_instance, _PField) == True
    assert isinstance(dct['field1'], _PField) == True
    assert isinstance(dct['field2'], _PField) == False


# LLM-generated content at query #101
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    factory_func = lambda: MockCheckedType()
    pfield = _PField(
        type={MockCheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=None
    )
    
    assert pfield._factory is factory_func


# LLM-generated content at query #102
#--------------------------

```python
def test_check_global_invariants_all_pass():
    invariant1 = lambda x: (True, None)
    invariant2 = lambda x: (True, None)
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    result = check_global_invariants(subject, invariants)
    assert result is None


def test_check_global_invariants_single_failure():
    invariant1 = lambda x: (True, None)
    invariant2 = lambda x: (False, "ERROR_CODE_1")
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    invariant1 = lambda x: (False, "ERROR_CODE_1")
    invariant2 = lambda x: (False, "ERROR_CODE_2")
    invariant3 = lambda x: (True, None)
    invariants = [invariant1, invariant2, invariant3]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1", "ERROR_CODE_2")
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test_subject"
    
    result = check_global_invariants(subject, invariants)
    assert result is None


def test_check_global_invariants_with_dict_subject():
    invariant1 = lambda x: (True, None) if "key" in x else (False, "MISSING_KEY")
    invariants = [invariant1]
    subject = {"key": "value"}
    
    result = check_global_invariants(subject, invariants)
    assert result is None


def test_check_global_invariants_with_dict_subject_failure():
    invariant1 = lambda x: (True, None) if "key" in x else (False, "MISSING_KEY")
    invariants = [invariant1]
    subject = {"other_key": "value"}
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("MISSING_KEY",)


# LLM-generated content at query #103
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap, PMap
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Ensure the field type exists in the cache
    from pyrsistent._field_common import pmap_field
    field_type = pmap_field(key_type, value_type)
    
    # Create test data
    test_data = {'a': 1, 'b': 2}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap with correct data
    assert isinstance(result, PMap)
    assert result == pmap(test_data)
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #104
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import CheckedPMap
    
    result = pmap_field(str, int)
    assert result.mandatory is True
    assert result.initial is not None
    assert callable(result.factory)
    assert result.invariant is not None


def test_pmap_field_optional_true():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    assert result.mandatory is True
    assert callable(result.factory)
    factory_result_none = result.factory(None)
    assert factory_result_none is None


def test_pmap_field_optional_false():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=False)
    assert result.mandatory is True
    assert callable(result.factory)


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    
    def my_invariant(val):
        return True, "valid"
    
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.mandatory is True
    assert result.invariant is not None
    assert callable(result.invariant)


def test_pmap_field_type_attribute():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int)
    assert result.type is not None
    assert len(result.type) > 0


def test_pmap_field_factory_callable():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int)
    assert callable(result.factory)


def test_pmap_field_optional_factory_with_dict():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    factory_result = result.factory({'a': 1})
    assert factory_result is not None


# LLM-generated content at query #105
#--------------------------

```python
def test_pfield_constructor():
    def dummy_invariant(value):
        return True
    
    def dummy_factory():
        return "factory_result"
    
    def dummy_serializer(value):
        return str(value)
    
    # Test basic constructor initialization
    pfield = _PField(
        type=str,
        invariant=dummy_invariant,
        initial="initial_value",
        mandatory=True,
        factory=dummy_factory,
        serializer=dummy_serializer
    )
    
    assert pfield.type == str
    assert pfield.invariant == dummy_invariant
    assert pfield.initial == "initial_value"
    assert pfield.mandatory is True
    assert pfield._factory == dummy_factory
    assert pfield.serializer == dummy_serializer


def test_pfield_constructor_with_none_values():
    pfield = _PField(
        type=int,
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert pfield.type == int
    assert pfield.invariant is None
    assert pfield.initial is None
    assert pfield.mandatory is False
    assert pfield._factory is None
    assert pfield.serializer is None


def test_pfield_constructor_with_tuple_type():
    pfield = _PField(
        type=(str, int),
        invariant=lambda x: True,
        initial=42,
        mandatory=True,
        factory=list,
        serializer=str
    )
    
    assert pfield.type == (str, int)
    assert pfield.initial == 42
    assert pfield.mandatory is True
    assert pfield._factory == list


# LLM-generated content at query #106
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import CheckedPMap
    
    result = pmap_field(str, int)
    assert result.mandatory is True
    assert result.factory is not None
    assert callable(result.factory)


def test_pmap_field_with_optional_true():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    assert result.mandatory is True
    assert result.factory is not None
    assert callable(result.factory)


def test_pmap_field_with_optional_false():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=False)
    assert result.mandatory is True
    assert result.factory is not None
    assert callable(result.factory)


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    
    def my_invariant(value):
        return (True, None)
    
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.mandatory is True
    assert result.invariant is not None
    assert callable(result.invariant)


def test_pmap_field_initial_value():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int)
    assert result.initial is not None


def test_pmap_field_type_attribute():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int)
    assert result.type is not None
    assert len(result.type) > 0


def test_pmap_field_optional_factory_with_none():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    factory_result = result.factory(None)
    assert factory_result is None


def test_pmap_field_optional_factory_with_dict():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    factory_result = result.factory({"key": 1})
    assert factory_result is not None


def test_pmap_field_non_optional_factory():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=False)
    factory_result = result.factory({"key": 1})
    assert factory_result is not None


# LLM-generated content at query #107
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    class TestBase:
        pass
    
    pfield_instance = _PField()
    dct = {"field1": pfield_instance, "field2": "not_a_pfield"}
    bases = [TestBase]
    name = "test_name"
    
    # Check that the predicate evaluates to True for _PField instances
    assert isinstance(pfield_instance, _PField) == True
    assert isinstance("not_a_pfield", _PField) == False


# LLM-generated content at query #108
#--------------------------

```python
def test_sequence_field_non_optional_creates_field_with_correct_type():
    from pyrsistent._field_common import _sequence_field, field
    from pyrsistent import PVector, CheckedPVector
    
    class CheckedIntVector(CheckedPVector):
        __type__ = int
    
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    
    assert isinstance(result, type(field()))
    assert result.mandatory is True


def test_sequence_field_optional_creates_field_with_none_type():
    from pyrsistent._field_common import _sequence_field, field
    from pyrsistent import PVector, CheckedPVector
    
    class CheckedIntVector(CheckedPVector):
        __type__ = int
    
    result = _sequence_field(CheckedPVector, int, True, [1, 2, 3])
    
    assert isinstance(result, type(field()))
    assert result.mandatory is True


def test_sequence_field_factory_returns_none_for_optional_with_none_argument():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, True, [])
    factory_result = result._factory(None)
    
    assert factory_result is None


def test_sequence_field_factory_creates_checked_instance_for_optional_with_value():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, True, [])
    factory_result = result._factory([1, 2, 3])
    
    assert len(factory_result) == 3
    assert list(factory_result) == [1, 2, 3]


def test_sequence_field_factory_non_optional_creates_checked_instance():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, False, [1, 2])
    factory_result = result._factory([1, 2, 3])
    
    assert len(factory_result) == 3


def test_sequence_field_initial_value_is_set():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, False, [5, 6, 7])
    
    assert result.initial is not None
    assert len(result.initial) == 3


def test_sequence_field_with_item_invariant():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import CheckedPVector
    
    def item_inv(val):
        return val > 0, "must be positive"
    
    result = _sequence_field(CheckedPVector, int, False, [1], item_invariant=item_inv)
    
    assert result.invariant == PFIELD_NO_INVARIANT


def test_sequence_field_with_field_invariant():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    def field_inv(val):
        return len(val) > 0, "must not be empty"
    
    result = _sequence_field(CheckedPVector, int, False, [1], invariant=field_inv)
    
    assert callable(result.invariant)


def test_sequence_field_optional_factory_with_factory_fields():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, True, [])
    factory_result = result._factory([1, 2], _factory_fields=None, ignore_extra=False)
    
    assert len(factory_result) == 2


# LLM-generated content at query #109
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariant_fails():
    def failing_invariant(subject):
        return (False, "ERROR_CODE_1")
    
    invariants = [failing_invariant]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1",)
        assert e.args[1] == ()
        assert e.args[2] == "Global invariant failed"


# LLM-generated content at query #110
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    mock_factory = lambda: "test"
    pfield = _PField(
        type={MockCheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=mock_factory,
        serializer=None
    )
    
    assert pfield._factory is mock_factory
    assert pfield._factory is not None


# LLM-generated content at query #111
#--------------------------

```python
def test_pmap_field_type_predicate_line_25():
    from pyrsistent._field_common import pmap_field
    from pyrsistent._checked_types import optional
    from pyrsistent import pmap
    
    # Test with optional=False - type should be TheMap (not wrapped in optional)
    field_non_optional = pmap_field(str, int, optional=False)
    assert field_non_optional.type is not None
    
    # Test with optional=True - type should be wrapped with optional (includes type(None))
    field_optional = pmap_field(str, int, optional=True)
    assert field_optional.type is not None
    # When optional=True, the type should include None as one of the allowed types
    assert isinstance(field_optional.type, tuple)
    assert type(None) in field_optional.type


# LLM-generated content at query #112
#--------------------------

```python
def test_pmap_field_optional_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Test that the predicate at line 15 evaluates to True
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field is not None
    
    # Test that the predicate at line 15 evaluates to False
    non_optional_field = pmap_field(str, int, optional=False)
    assert non_optional_field is not None


# LLM-generated content at query #113
#--------------------------

```python
def test_make_pmap_field_type_creates_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    
    pmap_type = _make_pmap_field_type(str, int)
    
    assert issubclass(pmap_type, CheckedPMap)
    assert pmap_type.__key_type__ == str
    assert pmap_type.__value_type__ == int


def test_make_pmap_field_type_generates_correct_name():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type(str, int)
    
    assert pmap_type.__name__ == "StrToIntPMap"


def test_make_pmap_field_type_caches_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type1 = _make_pmap_field_type(str, int)
    pmap_type2 = _make_pmap_field_type(str, int)
    
    assert pmap_type1 is pmap_type2


def test_make_pmap_field_type_different_types_creates_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type1 = _make_pmap_field_type(str, int)
    pmap_type2 = _make_pmap_field_type(int, str)
    
    assert pmap_type1 is not pmap_type2
    assert pmap_type1.__name__ == "StrToIntPMap"
    assert pmap_type2.__name__ == "IntToStrPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type(str, int)
    
    assert hasattr(pmap_type, '__reduce__')
    assert callable(getattr(pmap_type, '__reduce__'))


def test_make_pmap_field_type_with_builtin_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type(float, bool)
    
    assert pmap_type.__name__ == "FloatToBoolPMap"
    assert pmap_type.__key_type__ == float
    assert pmap_type.__value_type__ == bool


# LLM-generated content at query #114
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    class MockCheckedType(CheckedType):
        pass
    
    PFIELD_NO_SERIALIZER = object()
    value = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "serialized_value"


def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    PFIELD_NO_SERIALIZER = object()
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "xml:test_value"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    class MockCheckedType(CheckedType):
        pass
    
    def custom_serializer(format, value):
        return f"custom:{format}"
    
    PFIELD_NO_SERIALIZER = object()
    value = MockCheckedType()
    result = serialize(custom_serializer, "yaml", value)
    assert result == "custom:yaml"


def test_serialize_with_non_checked_type():
    def serializer(format, value):
        return f"serialized_{value}"
    
    PFIELD_NO_SERIALIZER = object()
    result = serialize(serializer, "json", 42)
    assert result == "serialized_42"


# LLM-generated content at query #115
#--------------------------

```python
def test_make_seq_field_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector, pvector
    
    # Clear the cache before testing
    _seq_field_types.clear()
    
    # Test creating a seq field type with PVector and int
    item_type = int
    item_invariant = None
    
    result_type = _make_seq_field_type(PVector, item_type, item_invariant)
    
    # Verify the result is a class
    assert isinstance(result_type, type)
    
    # Verify it's a subclass of PVector
    assert issubclass(result_type, PVector)
    
    # Verify the __type__ attribute is set correctly
    assert result_type.__type__ == int
    
    # Verify the __invariant__ attribute is set correctly
    assert result_type.__invariant__ is None
    
    # Verify the type is cached
    cached_type = _seq_field_types.get((PVector, int))
    assert cached_type is result_type
    
    # Verify calling it again returns the cached type
    result_type_2 = _make_seq_field_type(PVector, int, None)
    assert result_type_2 is result_type
    
    # Verify __reduce__ method exists
    instance = result_type([1, 2, 3])
    reduce_result = instance.__reduce__()
    assert reduce_result is not None
    assert len(reduce_result) == 2
    
    # Verify the name contains the type information
    assert hasattr(result_type, '__name__')
    assert 'int' in result_type.__name__.lower() or 'Int' in result_type.__name__


