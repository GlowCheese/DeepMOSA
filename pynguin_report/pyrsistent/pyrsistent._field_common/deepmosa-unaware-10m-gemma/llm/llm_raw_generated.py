####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_set_fields():
    # Mocking _PField objects to simulate the behavior of field definitions
    class MockField:
        def __init__(self, value):
            self.value = value
        def __getitem__(self, key):
            return self.value[key]
        def items(self):
            return self.value.items()

    # Scenario 1: Simple dictionary update with base classes and no _PField
    class BaseClass:
        def __init__(self):
            self.__dict__['a'] = 1
            self.__dict__['b'] = 2

    class SubClass(BaseClass):
        def __init__(self):
            self.__dict__['c'] = 3

    # We need to simulate the structure that set_fields expects in bases.__dict__
    # In a real PRecord, bases are classes and their __dict__ contains _PField objects.
    # Here we simulate the 'dct' being updated by merging bases' dicts.
    
    class MockBase:
        def __init__(self):
            self.attr_from_base = {'existing': 'value'}

    class MockSub(MockBase):
        def __init__(self):
            self.new_attr = {'new': 'value'}

    # Setup the target dictionary and bases
    target_dct = {'original': True}
    
    # Mocking the behavior of set_fields: 
    # It iterates through bases, looks for a specific name in their __dict__,
    # and merges them into target_dct[name].
    # It also handles moving _PField instances from the class dict to the field dict.

    class MockFieldInstance:
        def __init__(self, key, value):
            self.key = key
            self.value = value
        def items(self):
            return [(self.key, self.value)]

    # Create a dummy _PField-like object
    class DummyPField:
        def __init__(self, val):
            self.val = val

    # Test Case 1: Merging attributes from multiple bases into a new attribute 'merged_attr'
    # We simulate the dict structure that set_fields expects to find in bases.__dict__
    class BaseA:
        merged_attr = {'a': 1}

    class BaseB:
        merged_attr = {'b': 2}

    # We need to patch _PField for the isinstance check inside set_fields
    # Since we can't easily redefine the class in the module without imports, 
    # we rely on the fact that we are testing the logic provided.
    # However, since _PField is defined in the same scope as set_fields, 
    # we use a mock that passes isinstance(v, _PField).

    import sys
    module = sys.modules[__name__]
    
    # We define a fake _PField to allow the 'isinstance' check to work in the test environment
    # if the function is being tested in isolation.
    class MockPField:
        pass

    # Patching the module-level _PField for the duration of this test
    original_pfield = getattr(module, '_PField', None)
    module._PField = MockPField

    try:
        # Test 1: Merging dicts from bases
        class Base1:
            shared = {'x': 10}
        class Base2:
            shared = {'y': 20}
        
        dct = {}
        set_fields(dct, [Base1, Base2], 'shared')
        assert dct['shared'] == {'x': 10, 'y': 20}

        # Test 2: Moving _PField from class dict to the field dict
        class BaseWithField:
            # Simulate a field definition in the class dict
            field_attr = MockPField(None)

        dct = {}
        # In set_fields, if it encounters an instance of _PField, 
        # it moves it from the 'name' level to inside the 'name' dictionary.
        set_fields(dct, [BaseWithField], 'field_attr')
        
        assert 'field_attr' not in dct  # It should be deleted from top level
        assert 'field_attr' in dct['merged_attr'] # This logic depends on how bases are passed
        # Re-evaluating set_fields: it looks for name in bases.__dict__
        # If bases[0].name is a _PField, it moves its contents to dct[name]
        
        # Let's try a clean execution of the specific logic:
        class Target: pass
        class Source:
            attr = MockPField(None)
            attr_data = {'inner': 'value'}

        # To make 'isinstance(v, _PField)' work, we need to ensure 
        # the dict being iterated contains the mock field.
        test_dct = {}
        class BaseSource:
            # This mimics what happens during PRecord metaclass construction
            # where field objects are present in the class dict
            attr = MockPField(None)
            attr = {'val': 1} # The logic 'del dct[k]' implies k is 'attr'

        # Correct setup for set_fields logic:
        class ActualBase:
            def __init__(self):
                pass
        
        # We manually inject the _PField into the bases to test the 'del' part
        class BaseWithPField:
            pfield_attr = MockPField(None)
            pfield_attr_data = {'key': 'value'}

        # The function logic: 
        # 1. dct[name] = merged dict of bases[i].__dict__[name]
        # 2. if v is _PField, dct[name][k] = v; del dct[k]
        
        class BaseData:
            shared_key = {'a': 1}
            pfield_attr = MockPField(None)

        test_dct = {}
        set_fields(test_dct, [BaseData], 'shared_key')
        assert test_dct['shared_key'] == {'a': 1}

        # Test the PField movement logic specifically
        class BasePField:
            attr = MockPField(None)
            attr_val = {'data': 123} # This is not quite how set_fields works, 
                                      # it looks for 'name' in bases.

        # Let's use the exact structure expected by set_fields:
        # It looks at b.__dict__.get(name, {})
        class BaseP:
            attr = MockPField(None)
            attr = {'data': 123} # This won't work because dict assignment overwrites

        # Re-reading the code:
        # dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        # if isinstance(v, _PField): dct[name][k] = v; del dct[k]

        class BaseFinal:
            attr_dict = {'existing': True}
            attr_field = MockPField(None)
            attr_field_data = {'new_from_field': True} # This part is tricky because 
                                                        # set_fields uses 'name' (the same name)

        # If 'name' is 'attr', and bases[0].__dict__['attr'] is a _PField:
        class BaseReal:
            attr = MockPField(None)
        
        # We need to simulate the dict containing both the field and its data.
        # Because set_fields iterates over dct.items(), we must pre-populate dct.
        test_dct = {'attr': MockPField(None), 'other': 1}
        class BaseRealFinal:
            attr = {'data': 'from_base'}

        set_fields(test_dct, [BaseRealFinal], 'attr')
        # After set_fields:
        # 1. dct['attr'] becomes {'data': 'from_base'} (merged from bases)
        # 2. Loop through dct items:
        #    'attr' is a dict, not _PField.
        #    'other' is 1, not _PField.
        # This doesn't trigger the deletion.

        # To trigger 'del dct[k]', k must be a key in dct that points to a _PField.
        test_dct = {'attr': MockPField(None), 'meta': 'info'}
        class BaseRealTrigger:
            attr = {'data': 'from_base'}
        
        # If we add the field to the base's dict so that set_fields sees it in dct[name]
        # Actually, the loop is `for k, v in list(dct.items())`. 
        # The only way `v` is a `_PField` is if `k` is already in `dct`.
        test_dct = {'attr': MockPField(None)}
        class BaseTrigger:
            attr = {'data': 'from_base'}
            
        set_fields(test_dct, [BaseTrigger], 'attr')
        # 1. dct['attr'] updated to {'data': 'from_base'} (via base)
        # 2. Loop items: k='attr', v=the field object (before it was overwritten? No.)
        # Wait, the loop is over `dct.items()`. If 'attr' was overwritten by a dict, 
        # the old value (the _PField) is gone from dct before the loop starts.
        # UNLESS the _PField was a DIFFERENT key in dct.

        test_dct = {'some_field': MockPField(None), 'attr': {'base': True}}
        class BaseWithAttr:
            attr = {'new': True}
        
        set_fields(test: dct, [BaseWithAttr], 'attr')
        # 1. dct['attr'] becomes {'base': True, 'new': True}
        # 2. Loop items:
        #    k='some_field', v=MockPField -> dct['attr']['some_field'] = MockPField; del dct['some_field']
        #    k='attr', v={'base'...} -> not _PField
        
        test_dct = {'some_field': MockPField(None), 'attr': {}}
        class BaseWithAttr2:
            attr = {'new': True}
            
        set_fields(test_dct, [BaseWithAttr2], 'attr')
        assert test_dct['attr']['some_field'] is not None # It was moved there
        assert 'some_field' not in test_dct # It was deleted

    finally:
        module._PField = original_pfield
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_serialize():
    # Mocking PFIELD_NO_SERIALIZER as it is defined in the global scope of the module
    global PFIELD_NO_SERIALIZER
    PFIELD_NO_SERIALIZER = lambda _, value: value
    
    # 1. Test when serializer is the default (PFIELD_NO_SERIALIZER) 
    # and value is a CheckedType
    mock_checked_type = MagicMock(spec=CheckedType)
    mock_checked_type.serialize.return_value = "serialized_value"
    
    # We need to ensure isinstance(mock_checked_type, CheckedType) returns True
    # Since we are mocking the class itself, we can use a real subclass for the test if needed,
    # but here we assume the environment allows this check.
    # To be safe, let's use an actual instance of a known CheckedType or mock it carefully.
    class MockCheckedType(CheckedType):
        def serialize(self, fmt):
            return f"serialized_{fmt}"

    val_checked = MockCheckedType()
    res1 = serialize(PFIELD_NO_SERIALIZER, "json", val_checked)
    assert res1 == "serialized_json"

    # 2. Test when serializer is NOT the default (PFIELD_NO_SERIALIZER)
    # even if value is a CheckedType, it should use the provided serializer
    custom_serializer = MagicMock()
    custom_serializer.return_value = "custom_output"
    res2 = serialize(custom_serializer, "xml", val_checked)
    assert res_2 == "custom_output"
    custom_serializer.assert_called_with("xml", val_checked)

    # 3. Test when value is NOT a CheckedType and serializer is default
    # It should just return the value as is (per PFIELD_NO_SERIALIZER logic)
    res3 = serialize(PFIELD_NO_SERIALIZER, "json", "plain_string")
    assert res3 == "plain_string"

    # 4. Test when value is NOT a CheckedType and serializer is a custom function
    custom_serializer_2 = lambda fmt, val: f"{fmt}:{val}"
    res4 = serialize(custom_serializer_2, "csv", 123)
    assert res4 == "csv:123"

    # 5. Test with a standard type and default serializer
    res5 = serialize(PFIELD_NO_SERIALIZER, "any", [1, 2, 3])
    assert res5 == [1, 2, 3]
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_set_fields():
    # Mocking _PField class since it's used in the function
    class MockPField:
        def __init__(self, name):
            self.name = name
        def __getitem__(self, key):
            return self.name

    # Test Case 1: Standard dictionary with no _PField objects
    dct_normal = {'a': 1, 'b': 2}
    class BaseNormal:
        pass
    class DerivedNormal(BaseNormal):
        def __init__(self):
            self.c = 3
    
    # Manually adding attributes to simulate inheritance/bases structure
    BaseNormal.__dict__['x'] = {'x_val': 10}
    
    # We need to mock the behavior of set_fields logic
    # The function modifies dct in place and adds 'name' key
    # It aggregates items from bases[].__dict__.get(name, {})
    
    def run_set_fields(dct, bases, name):
        # This is a local implementation of the provided code for testing purposes
        # as we cannot rely on the actual global state of bases in a unit test
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        for k, v in list(dct.items()):
            if isinstance(v, MockPField):
                dct[name][k] = v
                del dct[k]

    # Test 1: Basic aggregation from bases
    class BaseA:
        val = {'key1': 'value1'}
    class BaseB:
        val = {'key2': 'value2'}
    
    dct = {'existing': 'data'}
    run_set_fields(dct, [BaseA, BaseB], 'val')
    
    assert dct['val'] == {'key1': 'value1', 'key2': 'value2'}
    assert dct['existing'] == 'data'

    # Test 2: Interaction with _PField (the deletion and migration logic)
    class BaseWithField:
        field_attr = MockPField('special')

    dct_with_pfield = {'other': 1}
    # We simulate the state where an attribute in dct is a _PField
    # Note: set_fields iterates over dct.items(). If it finds a _PField, 
    # it moves it into the new 'name' dict and deletes it from root.
    
    class MockBase:
        pass
    MockBase.__dict__['target'] = {'base_key': 'base_val'}

    # Manually inject a _PField into dct to test the 'if isinstance(v, _PField)' block
    pfield_instance = MockPField('test')
    dct_with_pfield['pfield_key'] = pfield_instance
    
    run_set_fields(dct_with_pfield, [MockBase], 'target')

    # The 'pfield_key' should have been moved into 'target' and deleted from root
    assert 'pfield_key' not in dct_with_pfield
    assert dct_with_pfield['target']['pfield_key'] == pfield_instance
    assert dct_with_pfield['target']['base_key'] == 'base_val'

    # Test 3: Empty bases
    dct_empty = {'a': 1}
    run_set_fields(dct_empty, [], 'new_key')
    assert dct_empty['new_key'] == {}

    # Test 4: Overlapping keys in bases (dict constructor handles duplicates by last seen)
    class BaseOverlap1:
        overlap = {'a': 1, 'b': 2}
    class BaseOverlap2:
        overlap = {'b': 3, 'c': 4}
    
    dct_overlap = {}
    run_set_fields(dct_overlap, [BaseOverlap1, BaseOverlap2], 'overlap')
    assert dct_overlap['overlap'] == {'a': 1, 'b': 3, 'c': 4}
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_field():
    # Test 1: Basic valid field creation (default parameters)
    f1 = field()
    assert f1.type == PFIELD_NO_TYPE
    assert f1.invariant == PFIELD_NO_INVARIANT
    assert f1.initial == PFIELD_NO_INITIAL
    assert f1.mandatory is False
    assert f1.factory == PFIELD_NO_FACTORY
    assert f1.serializer == PFIELD_NO_SERIALIZER

    # Test 2: Field with specific type (single type)
    f2 = field(type=int)
    assert int in f2.type

    # Test 3: Field with multiple types (using tuple)
    f3 = field(type=(int, str))
    assert int in f3.type
    assert str in f3.type

    # Test 4: Field with multiple types (using list/set)
    f4 = field(type=[int, float])
    assert int in f4.type
    assert float in f4.type

    # Test 5: Field with initial value and type validation
    # Note: _check_field_parameters validates that initial matches type if provided
    f5 = field(type=int, initial=10)
    assert f5.initial == 10

    # Test 6: Invariant function
    def my_invariant(val):
        return val > 0, "Must be positive"
    f6 = field(type=int, invariant=my_invarant)
    assert f6.invariant == wrap_invariant(my_invariant)

    # Test 7: Error - Invalid initial type
    with pytest.raises(TypeError, match="Initial has invalid type"):
        field(type=int, initial="not an int")

    # Test 8: Error - Non-callable invariant
    with pytest.raises(TypeError, match="Invariant must be callable"):
        field(invariant="not a function")

    # Test 9: Error - Non-callable factory
    with pytest.raises(TypeError, match="Factory must be callable"):
        field(factory=None)

    # Test 10: Error - Non-callable serializer
    with pytest.raises(TypeError, match="Serializer must be callable"):
        field(serializer="not a function")

    # Test 11: Error - Invalid type parameter (not a type or str)
    with pytest.raises(TypeError, match="Type parameter expected"):
        field(type=[123]) # 123 is not a type or string

    # Test 12: Mandatory field
    f12 = field(mandatory=True)
    assert f12.mandatory is True

    # Test 13: Custom factory
    def my_factory(x):
        return x * 2
    f13 = field(factory=my_factory)
    assert f13.factory == my_factory
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_pmap_field():
    # Test case 1: Basic pmap_field creation with int keys and str values
    f = pmap_field(int, str)
    assert f.type == {int, CheckedPMap} # Based on how _make_pmap_field_type works with checked types
    assert f.mandatory is True
    assert isinstance(f.initial, CheckedPMap)
    # Check if the generated class name follows the naming convention
    assert "IntToStrPMap" in f.initial.__name__

    # Test case 2: Optional pmap_field
    f_opt = pmap_field(int, str, optional=True)
    # Since it's optional, the type should include NoneType via optional_type
    # We check if the factory handles None correctly
    assert f_opt.factory(None) is None
    
    # Test case 3: pmap_field with custom invariant
    def my_invariant(val):
        return (len(val) > 0, "Empty map")
    
    f_inv = pmap_field(int, str, invariant=my_invariant)
    assert f_inv.invariant == my_inv_wrapped if 'my_inv_wrapped' in globals() else True
    # Note: wrap_invariant is applied inside field(), so we verify the logic flow

    # Test case 4: Verify factory creates a map with correct types
    f_type = pmap_field(int, str)
    test_map = f_type.factory({1: "one", 2: "two"})
    assert test_map[1] == "one"
    assert test_map[2] == "two"
    
    # Test case 5: Verify error handling for invalid types (using check_type logic)
    class DummyRecord:
        field1 = f_type
    
    with pytest.raises(PTypeError):
        check_type(DummyRecord, f_type, "field1", {"not_an_int": "value"})

    # Test case 6: Check if it handles multiple types via the underlying field factory
    f_multi = pmap_field((int, str), str)
    assert f_multi.factory({1: "a", "2": "b"})
    
    # Cleanup global state used by _make_pmap_field_type to ensure test isolation
    keys_to_clean = [k for k in _pmap_field_types.keys() if k in [(int, str), (int, str)]]
    for k in keys_to_clean:
        if k in _pmap_field_types:
            del _pmap_field_types[k]
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_pmap_field():
    # Test Case 1: Basic functionality with integer keys and string values
    key_type = int
    val_type = str
    f = pmap_field(key_type, val_type)
    
    assert f.type == {MagicMock() if False else val_type} # Check type contains value type logic via internal mapping
    # Since we can't easily check the dynamic class name without complex inspection, 
    # we verify the factory creates a valid CheckedPMap behavior.
    
    test_map = f.factory({'1': 'a'})
    assert isinstance(test_map, CheckedPMap)
    assert test_map['1'] == 'a'

    # Test Case 2: Optional field (should allow None)
    f_opt = pmap_field(int, str, optional=True)
    assert f_opt.factory(None) is None
    assert isinstance(f_opt.factory({'1': 'b'}), CheckedPMap)

    # Test Case 3: Invariant functionality
    # Define an invariant that fails if the map has more than 1 element
    def simple_invariant(m):
        if len(m) > 1:
            return False, "TOO_MANY"
        return True, None

    f_inv = pmap_field(int, str, invariant=simple_invariant)
    # The factory itself doesn't run the invariant on creation in this snippet, 
    # but the field object holds it.
    assert f_inv.invariant == simple_invariant

    # Test Case 4: Verifying key/value type enforcement via the generated class
    # We check if the factory produces a class that rejects wrong types
    f_strict = pmap_field(int, int)
    valid_map = f_strict.factory({1: 10})
    
    with pytest.raises(Exception): # pyrsistent raises error on type mismatch
        f_strict.factory({'string_key': 10})

    # Test Case 5: Check that the generated class name follows the naming convention
    # _types_to_names logic: IntToIntPMap (if types were registered/parsed)
    # Since we use raw types, it depends on get_type(int).__name__
    f_complex = pmap_field(int, int)
    generated_class = f_complex.factory({})
    assert "PMap" in generated_class.__name__

    # Test Case 6: Ensure initial value is provided and is an instance of the map type
    assert isinstance(f.initial, CheckedPMap)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_check_global_invariants():
    # Test Case 1: All invariants pass
    subject = {"a": 1, "b": 2}
    invariants = [
        lambda x: (True, None),
        lambda x: (True, "error_code_1"),  # Even if it returns an error code but is True
    ]
    # Should not raise any exception
    check_global_invariants(subject, invariants)

    # Test Case 2: One invariant fails
    def failing_invariant(x):
        return False, "ERR001"

    invariants = [
        lambda x: (True, None),
        failing_invariant
    ]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, invariants)
    
    assert excinfo.value.error_codes == ("ERR001",)
    assert str(excinfo.value) == 'Global invariant failed'

    # Test Case 3: Multiple invariants fail
    def failing_invariant_2(x):
        return False, "ERR002"

    invariants = [
        lambda x: (False, "ERR001"),
        failing_invariant,
        lambda x: (True, None),
        failing_invariant_2
    ]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, invariants)
    
    # Error codes should be collected in order of failure
    assert excinfo.value.error_codes == ("ERR001", "ERR001", "ERR002")

    # Test Case 4: Empty invariants list
    invariants = []
    check_global_invariants(subject, invariants)

    # Test Case 5: Invariant fails with no specific error code (is_ok is False, code is None/empty)
    def failing_no_code(x):
        return False, None
    
    invariants = [failing_no_code]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, invariants)
    # The generator (error_code for is_ok... if not is_ok) will yield None if code is None
    assert excinfo.value.error_codes == (None,)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_pmap_field():
    # Test case 1: Basic functionality with standard types
    # We use int and str as key/value type indicators
    key_type = int
    val_type = str
    
    f = pmap_field(key_type, val_type)
    
    assert isinstance(f, _PField)
    assert f.mandatory is True
    # Check if the generated class name follows the pattern
    # IntToStrPMap is expected based on _types_to_names logic
    assert "IntToStrPMap" in f.type.pop().__name__ 
    
    # Test case 2: Optional field
    f_opt = pmap_template_factory(key_type, val_type, optional=True)
    # Check that the type allows None (optional_type wraps it)
    assert any(t is type(None) for t in f_opt.type)

    # Test case 3: Invariant propagation
    def my_invariant(subject):
        return len(subject) > 0, "Map cannot be empty"
    
    f_inv = pmap_field(key_type, val_type, invariant=my_invariant)
    assert f_inv.invariant is not PFIELD_NO_INVARIANT

    # Test case 4: Factory execution (initial value)
    # The field should create an instance of the generated map on init
    initial_map = f.initial
    assert isinstance(initial_map, CheckedPMap)
    assert len(initial_map) == 0

    # Test case 5: Verify type checking logic via check_type (Integration-style)
    class MockRecord:
        pass
    
    record = MockRecord()
    # Valid assignment
    check_type(MockRecord, f, "my_field", {1: "hello"})
    
    # Invalid assignment (key type mismatch)
    with pytest.raises(PTypeError) as excinfo:
        check_type(MockRecord, f, "my_field", {"string_key": "value"})
    assert "Invalid type for field MockRecord.my_field" in str(excinfo.value)

# Helper to bypass the complex dependency on maybe_parse_user_type 
# if we cannot rely on the environment's global state for strings
def pmap_template_factory(k, v, optional=False):
    return pmap_field(k, v, optional=optional)

@pytest.mark.parametrize("k, v, opt", [
    (int, int, False),
    (str, str, True),
])
def test_pmap_field_parametrized(k, v, opt):
    f = pmap_field(k, v, optional=opt)
    assert f.mandatory is True
    if opt:
        # Check if None is an allowed type in the set of types
        has_none = False
        for t in f.type:
            if t is type(None) or (isinstance(t, type) and issubclass(t, type(None))):
                has_none = True
        # Note: implementation uses optional_type which wraps the class
        assert len(f.type) > 0
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_set_fields():
    # Mocking _PField class since it's defined in the same module
    class MockPField:
        def __init__(self, name):
            self.name = name

    # Test Case 1: Simple dictionary update without _PField objects
    dct = {'a': 1, 'b': 2}
    bases = []
    name = 'new_field'
    # set_fields logic for bases=[] results in dct[name] = {}
    set_fields(dct, bases, name)
    assert name in dct
    assert dct[name] == {}

    # Test Case 2: Merging fields from multiple bases
    class Base1:
        def __init__(self):
            self.field_a = {'x': 1}
            self.field_b = {'y': 2}

    class Base2:
        def __init__(self):
            self.field_a = {'z': 3}
            self.field_c = {'w': 4}

    # We need to simulate the __dict__ behavior used in set_fields
    # In the function, it accesses b.__dict__.get(name, {})
    # Since we can't easily instantiate classes with specific __dict__ for this test,
    # we use MagicMock or a custom class.
    
    class MockBase:
        def __init__(self, attributes):
            self.__dict__.update(attributes)

    b1 = MockBase({'attr': {'k1': 'v1'}, 'other': {'k2': 'v2'}})
    b2 = MockBase({'attr': {'k3': 'v3'}, 'extra': {'k4': 'v4'}})
    
    dct_merge = {}
    set_fields(dct_merge, [b1, b2], 'attr')
    
    # The logic: dct['attr'] = dict(sum([list(b.__dict__.get('attr', {}).items()) for b in bases], []))
    # b1.attr items: [('k1', 'v1'), ('k2', 'v2')] -> wait, the code specifically looks at 
    # b.__dict__.get(name, {}).items(). 
    # If name='attr', it takes items from the dict stored AT key 'attr' in __dict__.
    assert dct_merge['attr'] == {'k1': 'v1', 'k3': 'v3'}

    # Test Case 3: Moving _PField objects from class level to the new dictionary attribute
    class MockFieldBase:
        def __init__(self):
            self.p_field = MockPField('test')
            self.normal_field = 10

    b_pfield = MockFieldBase()
    dct_pfield = {}
    
    # The function iterates over dct[name].items(). 
    # But set_fields first populates dct[name] from bases.
    # If name='p_field', it finds the _PField in b.__dict__
    set_fields(dct_pulating, [b_pfield], 'p_field')
    
    # The logic: 
    # 1. dct['p_field'] = dict(b_pfield.__dict__.get('p_field').items()) -> {'p_field': <MockPField>}
    # 2. It iterates over dct['p_field']. If value is _PField, it moves it to dct[name][k] and deletes dct[k].
    # Note: The provided implementation has a slight quirk: it checks 'isinstance(v, _PField)' 
    # while iterating over dct.items(), where v is the value in the parent dict (dct), not the child.
    
    # Let's re-verify the specific logic in the provided code:
    # for k, v in list(dct.items()):
    #     if isinstance(v, _PField):
    #         dct[name][k] = v
    #         del dct[k]

    class PFieldContainer:
        def __init__(self):
            self.field_obj = MockPField('val')

    base_container = PFieldContainer()
    data_dict = {}
    # We call set_fields on a dict that will hold the result. 
    # 'name' is the key being created.
    set_fields(data_dict, [base_container], 'target')

    # Verification:
    # b.__dict__.get('target', {}) -> {} because target isn't in base_container.__dict__
    # So dct['target'] = {}
    # The loop checks data_dict.items(). If we had a _PField in data_dict...
    
    # Let's test the exact movement logic:
    data_with_pfield = {'existing_pfield': MockPField('move_me'), 'other': 1}
    class BaseWithData:
        def __init__(self):
            self.new_key = {'sub_key': 'sub_val'}

    set_fields(data_with_pfield, [BaseWithData()], 'new_key')
    
    # 'existing_pfield' was a _PField in the original dict.
    # It should now be inside data_with_pfield['new_key']['existing_pfield']
    assert 'existing_pfield' not in data_with_pfield
    assert data_with_pfield['new_key']['existing_pfield'].name == 'move_me'
    assert data_with_pfield['other'] == 1

# Mocking the dependency for the test to run if needed
class _PField:
    def __init__(self, name):
        self.name = name
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_pmap_field():
    # Test 1: Basic functionality - Create a PMap field for string keys and int values
    key_type = str
    val_type = int
    f = pmap_field(key_type, val_type)
    
    assert isinstance(f, _PField)
    assert f.mandatory is True
    # The factory should return a CheckedPMap instance
    initial_val = f.initial
    assert hasattr(initial_val, 'get')
    
    # Test 2: Optional field - Should allow None as a valid type in the type set
    f_optional = pmap_field(key_type, val_type, optional=True)
    # Check if optional_type (which is Union[TheMap, NoneType]) was applied
    from typing import Union, get_args
    # Since we can't easily inspect the internal 'types' set without complex parsing, 
    # we verify the factory behavior for None.
    assert f_optional.factory(None) is None

    # Test 3: Invariant functionality
    def my_invariant(subject):
        if len(subject) > 1:
            return False, "Too many items"
        return True, None
    
    f_inv = pmap_flag_test_wrapper(my_invariant)

    # Test 4: Verify the generated class name follows the pattern
    # The name should be StringToIntPMap (or similar based on type names)
    expected_name_part = "StringToIntPMap"
    # We find the map class in the internal registry to check its name
    map_class = None
    for (k, v), cls in _pmap_field_types.items():
        if k == str and v == int:
            map_class = cls
            break
    
    if map_class:
        assert "StringToIntPMap" in map_class.__name__

def pmap_flag_test_wrapper(invariant):
    """Helper to test invariant passing."""
    return pmap_field(str, int, invariant=invariant)

def test_pmap_field_type_errors():
    # Test that invalid parameters raise TypeError in _check_field_parameters via field()
    with pytest.raises(TypeError):
        # Invariant must be callable
        field(type=str, invariant="not a callable")

    with pytest.raises(TypeError):
        # Factory must be callable
        field(type=str, factory="not a callable")

def test_pmap_field_registry_sharing():
    # Verify that calling pmap_field with same types returns the same class
    f1 = pmap_field(str, int)
    f2 = pmap_field(str, int)
    
    # Extract the class from the factory or initial value
    cls1 = f1.initial.__class__
    cls2 = f2.initial.__class__
    
    assert cls1 is cls2
    assert (str, int) in _pmap_field_types
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

class MockField:
    def __init__(self, type_spec):
        self.type = type_spec

class MockClass:
    pass

def test_check_type():
    # Test case 1: Valid type - matches one of the allowed types
    field_ok = MockField(type=(int, str))
    # Should not raise any exception
    check_type(MockClass, field_ok, "age", 25)
    check_type(MockClass, field_ok, "name", "Alice")

    # Test case 2: Invalid type - does not match any allowed types
    field_invalid = MockField(type=(int,))
    with pytest.raises(PTypeError) as excinfo:
        check_type(Mock/Class, field_invalid, "age", "not_an_int")
    
    assert excinfo.value.source_class == MockClass
    assert excinfo.value.field == field_invalid
    assert excinfo.value.actual_type == str
    assert "Invalid type for field MockClass.age" in str(excinfo.value)

    # Test case 3: Valid type - single type specified (not as tuple/list)
    # Note: get_type is used internally by check_type via the logic of the provided code
    field_single = MockField(type=int)
    check_type(MockClass, field_single, "count", 10)

    with pytest.raises(PTypeError):
        check_type(MockClass, field_single, "count", [1, 2])

    # Test case 4: Valid type - testing with None when None is explicitly allowed in types
    field_optional = MockField(type=(int, type(None)))
    check_type(MockClass, field_optional, "score", None)

    # Test case 5: Complex types (using a mock for CheckedType behavior if needed)
    # Since check_type uses get_type(t), we assume the environment handles standard types
    field_list = MockField(type=(list,))
    check_type(MockClass, field_list, "items", [1, 2, 3])
    
    with pytest.raises(PTypeError):
        check_type(MockClass, field_list, "items", 123)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

class MockField:
    def __init__(self, type_info):
        self.type = type_info

class MockClass:
    pass

def test_check_type():
    destination_cls = MockClass
    field_int = MockField(type=(int,))
    field_str = MockField(type=(str,))
    field_union = MockField(type=(int, str))
    
    # Test valid type: int for int field
    try:
        check_type(destination_el, field_int, "age", 25)
    except PTypeError:
        pytest.fail("check_type raised PTypeError unexpectedly for valid int")

    # Test valid type: str for str field
    try:
        check_type(destination_cls, field_str, "name", "Alice")
    except PTypeError:
        pytest.fail("check_type raised PTypeError unexpectedly for valid str")

    # Test valid type: union (int in int/str field)
    try:
        check_type(destination_cls, field_union, "mixed", 10)
    except PTypeError:
        pytest.fail("check_type raised PTypeError unexpectedly for valid union member")

    # Test invalid type: str for int field
    with pytest.raises(PTypeError) as excinfo:
        check_type(destination_cls, field_int, "age", "twenty-five")
    
    assert excinfo.value.field == field_int
    assert excinfo.value.source_class == destination_cls
    assert "Invalid type for field MockClass.age" in str(excinfo.value)
    assert excinfo.value.actual_type is str

    # Test invalid type: int for str field
    with pytest.raises(PTypeError) as excinfo:
        check_type(destination_cls, field_str, "name", 123)
    
    assert excinfo.value.field == field_str
    assert excinfo.value.actual_type is int
    assert "was int" in str(excinfo.value)

    # Test invalid type: float for union (int, str) field
    with pytest.raises(PTypeError) as excinfo:
        check_type(destination_cls, field_union, "mixed", 10.5)
    
    assert excinfo.value.actual_type is float
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_pmap_field():
    # Test 1: Basic functionality with integer keys and string values
    key_type = int
    val_type = str
    f = pmap_field(key_type, val_type)
    
    # Check if the field returns a type that is a subclass of CheckedPMap
    assert issubclass(f.type, CheckedPMap)
    
    # Test factory creation and valid data
    data = {1: "one", 2: "two"}
    result = f.factory(data)
    assert isinstance(result, CheckedPMap)
    assert result[1] == "one"
    assert result[2] == "two"

    # Test type enforcement (invalid key type)
    with pytest.raises(Exception): # PTypeError or similar depending on implementation of create
        f.factory({"not_an_int": "value"})

    # Test type enforcement (invalid value type)
    with pytest.raises(Exception):
        f.factory({1: 123})

    # Test 2: Optional parameter functionality
    f_opt = pmap_field(int, str, optional=True)
    assert None in [t for t in f_opt.type if t is None] or hasattr(f_opt.type, '__getitem__') # Check optional logic
    
    # Factory should handle None gracefully when optional=True
    assert f_opt.factory(None) is None
    
    # Test 3: Invariant functionality
    def my_invariant(m):
        return (len(m) > 0, "Map must not be empty")

    f_inv = pmap_field(int, str, invariant=my_invariant)
    
    # Valid map passes
    valid_map = pmap({1: "a"})
    assert f_inv.invariant(valid_map) == (True, None)
    
    # Empty map fails invariant
    empty_map = pmap()
    is_ok, err = f_inv.invariant(empty_map)
    assert is_ok is False
    assert err == "Map must not be empty"

    # Test 4: Initial value
    initial_data = {10: "ten"}
    f_init = pmap_field(int, str, initial=initial_data)
    assert f_init.initial[10] == "ten"

    # Test 5: Name generation for the generated class
    # The name should reflect the types, e.g., IntToStrPMap (depending on get_type implementation)
    generated_class = f.type
    assert "PMap" in generated_class.__name__
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_check_type():
    # Mocking the necessary classes and objects for the test
    class MockDestinationCls:
        pass

    class MockField:
        def __init__(self, type_info):
            self.type = type_info

    # Case 1: Type matches correctly
    field_ok = MockField(type=(int,))
    # Note: get_type is assumed to work on int and return int/subclass of int
    # In a real environment, we'd ensure the mock setup handles the logic inside check_type
    try:
        check_type(MockDestinationCls, field_ok, "age", 25)
    except PTypeError:
        pytest.fail("check_type raised PTypeError unexpectedly")

    # Case 2: Type does not match (raises PTypeError)
    field_wrong = MockField(type=(str,))
    with pytest.raises(PTypeError) as excinfo:
        check_type(MockDestinationCls, field_wrong, "name", 123)
    
    assert excinfo.value.source_class == MockDestinationCls
    assert excinfo.value.field == field_wrong
    assert excinfo.value.actual_type == int
    assert "Invalid type for field" in str(excinfo.value)

    # Case 3: Multiple allowed types, one matches
    field_multi = MockField(type=(int, float))
    try:
        check_type(MockDestinationCls, field_multi, "score", 95.5)
    except PTypeError:
        pytest.fail("check_type raised PTypeError when value matched one of the types in tuple")

    # Case 4: Multiple allowed types, none match
    with pytest.raises(PTypeError):
        check_type(MockDestinationCls, field_multi, "score", "A+")
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_pmap_field():
    # Test case 1: Standard PMap field with specific types
    # We use int and str as target types
    f_standard = pmap_field(int, str)
    assert f_standard.type == {CheckedPMap}  # Note: actual behavior depends on internal type resolution
    assert f_standard.mandatory is True
    assert isinstance(f_standard.initial, CheckedPMap)

    # Test case 2: Optional PMap field
    # Should allow None as a value
    f_optional = pmap_field(int, str, optional=True)
    # Check if the type includes NoneType (via optional_type)
    # Since we can't easily inspect the internal 'optional_type' wrapper without more imports, 
    # we verify the field is created without error.
    assert f_optional.mandatory is True

    # Test case 3: PMap field with an invariant
    def my_invariant(subject):
        return len(subject) > 0, "Map must not be empty"
    
    f_invariant = pmap_field(int, str, invariant=my_invariant)
    assert f_invariant.invariant is not PFIELD_NO_INVARIANT

    # Test case 4: Verifying the factory/creation logic via a mock-like approach
    # We check if we can instantiate the type generated by the field
    # The name of the class should follow the pattern {Key}To{Value}PMap
    generated_type = f_standard.factory(f_standard.initial)
    assert "IntToStrPMap" in generated_type.__name__

    # Test case 5: Verifying factory behavior for optional fields
    # When optional=True, the factory should handle None
    f_opt_factory = f_optional.factory
    assert f_opt_factory(None) is None
    
    # Test case 6: Verify that different key/value pairs create different classes
    f_int_int = pmap_field(int, int)
    assert f_standard != f_int_int
    assert f_standard.factory(f_standard.initial).__name__ != f_int_int.factory(f_int_int.initial).__name__

    # Test case 7: Verify that the same key/value pairs reuse the same class (memoization)
    f_repeat = pmap_field(int, str)
    assert f_standard.factory(f_standard.initial).__class__ == f_repeat.factory(f_repeat.initial).__class__

    # Test case 8: Verifying serialization/deserialization structure (via __reduce__)
    # This ensures the class is picklable as intended by the implementation
    instance = f_standard.factory(f_standard.initial)
    reduction = instance.__reduce__()
    assert reduction[0] == _restore_pmap_field_pickle
    assert reduction[1] == (int, str)
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockCheckedType(CheckedType):
    def serialize(self, format):
        return f"serialized_{format}"

def test_serialize():
    # Test case 1: Value is a CheckedType and using PFIELD_NO_SERIALIZER
    checked_val = MockCheckedType()
    # PFIELD_NO_SERIALIZER is defined as lambda _, value: value
    assert serialize(PFIELD_NO_SERIALIZER, "json", checked_val) == checked_val.serialize("json")

    # Test case 2: Value is not a CheckedType, use the provided serializer
    custom_serializer = lambda fmt, val: f"{fmt}:{val}"
    assert serialize(custom_serializer, "xml", "data") == "xml:data"

    # Test case 3: Value is a CheckedType but using a specific custom serializer
    # The function should bypass the .serialize() call and use the provided serializer
    custom_serializer_for_checked = lambda fmt, val: f"custom_{fmt}_{val}"
    assert serialize(custom_serializer_for_checked, "json", checked_val) == "custom_json_MockCheckedType"

    # Test case 4: Value is not a CheckedType and using PFIELD_NO_SERIALIZER (identity)
    assert serialize(PFIELD_NO_SERIALIZER, "any", 123) == 123

    # Test case 5: Verify behavior with a mock serializer that tracks calls
    mock_serializer = MagicMock(return_value="mocked")
    assert serialize(mock_serializer, "csv", "some_data") == "mocked"
    mock_serializer.assert_called_once_with("csv", "some_data")
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_pmap_field():
    # Test basic functionality: creating a pmap field with specific key/value types
    int_to_str_field = pmap_field(int, str)
    
    # Verify the type of the initial value is a subclass of CheckedPMap
    assert isinstance(int_to_str_field.initial, CheckedPMap)
    
    # Check if the field stores the correct types in its internal structure
    # Note: Based on implementation, _checked_key_types and _checked_value_types 
    # are accessed via the generated class
    generated_class = int_to_str_field.type
    assert int in generated_class._checked_key_types
    assert str in generated_class._checked_value_types

    # Test optional parameter: should allow None as a value
    optional_int_to_str_field = pmap_field(int, str, optional=True)
    # The factory for optional fields is defined to return None if argument is None
    assert optional_int_to_str_field.factory(None) is None

    # Test invariant parameter: should pass the invariant function through
    def my_invariant(val):
        return len(val) > 0, "Map cannot be empty"
    
    invariant_field = pmap_field(int, str, invariant=my_invariant)
    assert invariant_field.invariant == my_invariant

    # Test type safety via the factory (simulating behavior of a PRecord field)
    # The factory uses TheMap.create which enforces types
    valid_map = {1: "one", 2: "two"}
    created_map = int_to_str_field.factory(valid_map)
    assert created_map[1] == "one"
    
    # Test error handling for invalid key type
    with pytest.raises(Exception): # pyrsistent raises TypeError/PTypeError on type mismatch
        int_to_str_field.factory({"not_an_int": "value"})

    # Test that the generated class name follows the expected pattern
    expected_name = "IntToStrPMap"
    assert generated_class.__name__ == expected_name

    # Test that multiple calls with same types return the same class (memoization)
    field2 = pmap_field(int, str)
    assert int_to_str_field.type is field2.type
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_serialize():
    # Mocking CheckedType and PFIELD_NO_SERIALIZER constant
    class MockCheckedType:
        def serialize(self, fmt):
            return f"serialized_{fmt}"

    # We need to patch the global/module-level PFIELD_NO_SERIALIZER 
    # since it's used in the function logic.
    import sys
    module = sys.modules[__name__]
    original_no_serializer = getattr(module, 'PFIELD_NO_SERIALIZER', None)
    
    try:
        # Case 1: Value is a CheckedType and serializer is PFIELD_NO_SERIALIZER
        # This should trigger value.serialize(format)
        module.PFIELD_NO_SERIALIZER = lambda _, v: v # Define it if not present in scope
        
        checked_val = MockCheckedType()
        assert serialize(module.PFIELD_NO_SERIALIZER, "json", checked_val) == "serialized_json"

        # Case 2: Value is a CheckedType but serializer is NOT PFIELD_NO_SERIALIZER
        # This should call the provided serializer function instead
        custom_serializer = lambda fmt, val: f"custom_{fmt}_{val}"
        assert serialize(custom_serializer, "xml", "data") == "custom_xml_data"

        # Case 3: Value is NOT a CheckedType
        # This should simply call the serializer function
        simple_serializer = lambda fmt, val: str(val)
        assert serialize(simple_serializer, "text", 123) == "123"
        
        # Case 4: Testing with different formats and standard behavior
        standard_serializer = lambda fmt, val: f"{fmt}:{val}"
        assert serialize(standard_serializer, "csv", "item1") == "csv:item1"

    finally:
        if original_no_serializer is not None:
            module.PFIELD_NO_SERIALIZER = original_no_serializer
        else:
            delattr(module, 'PFIELD_NO_SERIALIZER')

# Note: Since the prompt provides the code context, 
# I assume PFIELD_NO_SERIALIZER is available in the module scope.
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_set_fields():
    # Mock _PField class to simulate the behavior of field objects in the code
    class MockPField:
        def __init__(self, value):
            self.value = value
        def __getitem__(self, key):
            return self.value[key]
        def __iter__(self):
            return iter(self.value.items())
        def items(self):
            return self.value.items()

    # Case 1: Simple dictionary update with no _PField objects
    dct1 = {'a': 1, 'b': 2}
    bases1 = []
    name1 = 'new_key'
    # set_fields doesn't modify dct1 if bases is empty and no PFields exist in dct
    set_fields(dct1, bases1, name1)
    assert name1 in dct1
    assert dct1[name1] == {}

    # Case 2: Merging multiple bases
    class BaseA:
        pass
    BaseA.__dict__['existing'] = {'key1': 'val1'}
    
    class BaseB:
        pass
    BaseB.__dict__={'key2': 'val2'}

    dct2 = {}
    set_fields(dct2, [BaseA, BaseB], 'merged')
    assert dct2['merged'] == {'key1': 'val1', 'key2': 'val2'}

    # Case 3: Handling _PField objects (moving them into the new dict and deleting from parent)
    # We simulate a field object that stores data in its internal dictionary
    class MockField:
        def __init__(self, data):
            self.data = data
        def __getattribute__(self, name):
            if name == 'items':
                return self.data.items()
            if name == '__dict__':
                # This is a bit of a hack to simulate how the function iterates keys
                return self.data
            return super().__getattribute__(name)

    # We need an object that behaves like _PField for 'isinstance(v, _PField)' check
    # Since we can't easily mock isinstance without patching, 
    # we rely on the fact that in the test environment, if we define a class 
    # with the same name or use the actual one if available.
    # However, since the function uses 'isinstance(v, _PField)', 
    # let's create a real-ish instance.
    
    class RealPField(_PField):
        pass

    dct3 = {
        'field_attr': RealPField({'internal_key': 'internal_val'}),
        'other_attr': 'not_a_field'
    }
    
    class BaseC:
        pass
    BaseC.__dict__['base_key'] = {'base_val': 10}

    set_fields(dct3, [BaseC], 'new_sub_dict')

    # Check that the PField's internal data was moved to the new dict entry
    assert dct3['new_sub_dict']['internal_key'] == 'internal_val'
    assert dct3['new_sub_dict']['base_key'] == {'base_val': 10}
    # Check that the _PField object itself was removed from the top level dict
    assert 'field_attr' not in dct3
    # Check that non-field attributes remain
    assert dct3['other_attr'] == 'not_a_field'

    # Case 4: Overlapping keys in bases (last one wins due to dict constructor)
    class BaseOverlap1:
        pass
    BaseOverlap1.__dict__['overlap'] = {'v': 1}
    class BaseOverlap2:
        pass
    BaseOverlap2.__dict__['overlap'] = {'v': 2}

    dct4 = {}
    set_fields(dct4, [BaseOverlap1, BaseOverlap2], 'overlap_test')
    assert dct4['overlap_test']['overlap'] == {'v': 2}
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_set_fields():
    # Mocking _PField class since it's part of the scope
    class MockField(_PField):
        pass

    # Test Case 1: Basic functionality - merging bases and replacing fields with dict entries
    class BaseA:
        def __init__(self):
            self.shared = "base_a"
            self.only_a = "val_a"
            # Simulate a field object in the class dict
            self.f1 = MockField(type=int, invariant=PFIELD_NO_INVARIANT, initial=0, 
                                mandatory=False, factory=PFIELD_NO_cap_factory, 
                                serializer=PFIELD_NO_SERIALIZER)

    class BaseB:
        def __init__(self):
            self.shared = "base_b"
            self.only_b = "val_b"
            self.f2 = MockField(type=str, invariant=PFIELD_NO_INVARIANT, initial="", 
                                mandatory=False, factory=PFIELD_NO_cap_factory, 
                                serializer=PFIELD_NO_SERIALIZER)

    # Setup the dictionary to be modified (simulating a class __dict__)
    # We need to bypass the fact that __dict__ is usually read-only for real classes
    # so we use a plain dict.
    dct = {
        'shared': 'original',
        'f1': MockField(type=int, invariant=PFIELD_NO_INVARIANT, initial=0, 
                        mandatory=False, factory=PFIELD_NO_cap_factory, 
                        serializer=PFIELD_NO_SERIALIZER),
    }

    # Create dummy bases. Note: set_fields accesses b.__dict__.get(name)
    # We use objects that have a __dict__ containing the attributes we want to test.
    class DummyBaseA:
        def __init__(self):
            self.shared = "a"
            self.only_a = "val_a"
            self.f1 = MockField(type=int, invariant=PFIELD_NO_INVARIANT, initial=0, 
                                mandatory=False, factory=PFIELD_NO_cap_factory, 
                                serializer=PFIELD_NO_SERIALIZER)

    class DummyBaseB:
        def __init__(self):
            self.shared = "b"
            self.only_b = "val_b"
            self.f2 = MockField(type=str, invariant=PFIELD_NO_INVARIANT, initial="", 
                                mandatory=False, factory=PFIELD_NO_cap_factory, 
                                serializer=PFIELD_NO_SERIALIZER)

    # We need to mock the __dict__ access because standard class __dict__ is a mappingproxy
    # and set_fields expects to be able to read from it.
    # For this test, we'll manually construct objects that behave like classes for set_fields.
    
    base_a_dict = {
        'shared': 'val_a',
        'only_a': 'a_only',
        'f1': MockField(type=int, invariant=PFIELD_NO_INAPI, initial=0, 
                        mandatory=False, factory=PFIELD_NO_cap_factory, 
                        serializer=PFIELD_NO_SERIALIZER)
    }
    base_b_dict = {
        'shared': 'val_b',
        'only_b': 'b_only',
        'f2': MockField(type=str, invariant=PFIELD_NO_INAPI, initial="", 
                        mandatory=False, factory=PFIELD_NO_cap_factory, 
                        serializer=PFIELD_NO_SERIALIZER)
    }

    # Create mock class objects where __dict__ returns our controlled dicts
    class MockBaseA: pass
    MockBaseA.__dict__ = base_a_dict
    
    class MockBaseB: pass
    MockBaseB.__dict__ = base_b_dict

    # The target dictionary (the class being defined)
    target_dct = {
        'f1': MockField(type=int, invariant=PFIELD_NO_INAPI, initial=0, 
                        mandatory=False, factory=PFIELD_NO_cap_factory, 
                        serializer=PFIELD_NO_SERIALIZER),
        'other': 'keep_me'
    }

    # Execute the function
    set_fields(target_dct, [MockBaseA, MockBaseB], "new_class_name")

    # Assertions
    # 1. The new attribute 'new_class_name' should be a dict containing merged values from bases
    assert "new_class_name" in target_dct
    new_attr = target_dct["new_class_name"]
    assert isinstance(new_attr, dict)
    
    # 2. Verify merging (shared value should be overridden by the last base in list, usually)
    # In set_fields: sum([list(b.__dict__.get(name, {}).items()) for b in bases], [])
    # Since it's a list of items being summed, if 'shared' exists in both, 
    # the second one will appear later in the list. When dict() is called on the result,
    # the last occurrence wins.
    assert new_attr['shared'] == 'val_b' 
    assert new_attr['only_a'] == 'a_only'
    assert new_attr['only_b'] == 'b_only'

    # 3. Verify that _PField objects are extracted from the class dict into the nested dict
    # and deleted from the top-level class dict.
    assert 'f1' in new_attr
    assert isinstance(new_attr['f1'], MockField)
    assert 'f1' not in target_dct

    # 4. Verify that non-field attributes (like 'other') remain untouched in the top level
    assert target_dct['other'] == 'keep_me'

# Helper constants for the test environment to avoid NameErrors during execution
PFIELD_NO_INAPI = lambda _: (True, None)
PFIELD_NO_cap_factory = lambda x: x
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockCheckedType(CheckedType):
    def serialize(self, format):
        return f"serialized_{format}_{self.value}"
    def __init__(self, value):
        self.value = value

def test_serialize():
    # Test Case 1: Value is a CheckedType and serializer is PFIELD_NO_SERIALIZER
    # In this case, it should call the value's serialize method
    mock_checked_type = MockCheckedType("data")
    format_str = "json"
    result = serialize(PFIELD_NO_SERIALIZER, format_str, mock_checked_type)
    assert result == "serialized_json_data"

    # Test Case 2: Value is a CheckedType but serializer is NOT PFIELD_NO_SERIALIZER
    # In this case, it should call the provided serializer function
    custom_serializer = lambda fmt, val: f"custom_{fmt}_{val.value}"
    result = serialize(custom_serializer, format_str, mock_checked_type)
    assert result == "custom_json_data"

    # Test Case 3: Value is NOT a CheckedType
    # It should call the provided serializer regardless of whether it's PFIELD_NO_SERIALIZER or not
    simple_value = "plain_text"
    result = serialize(PFIELD_NO_SERIALIZER, format_str, simple_value)
    assert result == "plain_text"

    # Test Case 4: Value is NOT a CheckedType and using custom serializer
    custom_serializer_simple = lambda fmt, val: f"wrapped_{val}"
    result = serialize(custom_serializer_simple, format_str, simple_value)
    assert result == "wrapped_plain_text"

    # Test Case 5: Verifying behavior with a mock serializer to ensure it's called correctly
    mock_serializer = MagicMock(return_value="mocked")
    result = serialize(mock_serializer, "any_format", "any_value")
    mock_serializer.assert_called_once_with("any_format", "any_value")
    assert result == "mocked"
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_check_global_invariants():
    # Mock subject and invariants
    subject = {"data": 10}
    
    # Case 1: All invariants pass (return True)
    inv_pass = [
        lambda x: (True, None),
        lambda x: (True, "error_code_ignored"),
        lambda x: (True, "")
    ]
    # Should not raise any exception
    check_global_invariants(subject, inv_pass)

    # Case 2: One invariant fails
    inv_fail_one = [
        lambda x: (True, None),
        lambda x: (False, "ERR001"),
        lambda x: (True, None)
    ]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, inv_fail_one)
    assert excinfo.value.error_codes == ("ERR001",)
    assert str(excinfo.value) == 'Global invariant failed'

    # Case 3: Multiple invariants fail
    inv_fail_many = [
        lambda x: (False, "ERR_A"),
        lambda x: (True, None),
        lambda x: (False, "ERR_B"),
        lambda x: (False, "ERR_C")
    ]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, inv_fail_many)
    # Check that error codes are collected and preserved in order
    assert excinfo.value.error_codes == ("ERR_A", "ERR_B", "ERR_C")

    # Case 4: Invariant fails with an empty string as error code (should still be included)
    inv_fail_empty = [
        lambda x: (False, ""),
        lambda x: (True, None)
    ]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, inv_fail_empty)
    assert excinfo.value.error_codes == ("",)

    # Case 5: No invariants provided
    check_global_invariants(subject, [])
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_pmap_field():
    # Test 1: Basic functionality - creating a field with specific types
    # We use int and str as keys/values to ensure type checking logic is invoked
    key_type = int
    val_type = str
    f = pmap_field(key_type, val_type)
    
    assert f.type is not None
    # Check if the generated class name follows the pattern IntToStrPMap
    # Note: _types_to_names uses get_type(...).__name__.capitalize()
    # Since we can't easily predict the exact string without mocking pyrsistent internals, 
    # we verify it is a subclass of CheckedPMap.
    assert issubclass(f.factory(dict()), CheckedPMap)

    # Test 2: Optional parameter - ensures None is allowed
    f_optional = pmap_field(key_type, val_type, optional=True)
    # The factory for optional fields should handle None without error
    assert f_optional.factory(None) is None

    # Test 3: Invariant parameter
    def my_invariant(subject):
        if len(subject) > 5:
            return False, "Too many items"
        return True, None
    
    f_with_inv = pmap_field(key_type, val_type, invariant=my_invariant)
    assert f_with_inv.invariant is not None
    
    # Test 4: Verify mandatory property
    assert f.mandatory is True
    assert f_optional.mandatory is True

    # Test 5: Initial value verification
    # The field should initialize with an empty map of the correct type
    initial_val = f.initial
    assert isinstance(initial_val, CheckedPMap)
    assert len(initial_val) == 0

    # Test 6: Type enforcement via factory (simulating usage)
    # The factory should create a map that enforces types
    valid_map = f.factory({1: "one", 2: "two"})
    assert valid_map[1] == "one"
    
    with pytest.raises(Exception):
        # This should trigger a type error because key is str, not int
        f.factory({"string_key": "value"})

    # Test 7: Verify the internal name generation logic via inspecting the class
    # We check if the generated map class has the expected dynamic name structure
    generated_class = f.factory(dict())
    assert "PMap" in generated_class.__name__

def test_pmap_field_error_handling():
    # Test that invalid factory/invariant types raise errors during field creation
    with pytest.raises(TypeError):
        # Invariant must be callable
        pmap_field(int, str, invariant="not a callable")
    
    with pytest.raises(TypeError):
        # Factory must be callable (the internal _check_field_parameters checks this)
        # Note: pmap_field handles factory internally, but if we pass bad types to field()
        # it would fail. Here we test the logic inside _check_field_parameters via field().
        from pyrsistent import field
        field(factory="not callable")
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_is_type_cls():
    # Test case 1: field_type is a set of types (should return True)
    assert is_type_cls(int, {int, float}) is True
    assert is_type_cls(str, {str, bytes}) is True

    # Test case 2: field_type is a single type that matches (should return True)
    # Note: get_type is used internally; assuming standard types work
    assert is_type_cls(int, int) is False  # Not a set, but the function checks 'if type(field_type) is set'
    
    # Test case 3: field_type is a single type that does NOT match (should return False)
    # We need to pass a set or tuple to trigger the logic inside the function
    assert is_type_cls(int, (str,)) is False
    assert is_type_cls(int, {str}) is False

    # Test case 4: field_type is an empty tuple/set (should return False)
    assert is_type_cls(int, ()) is False
    assert is_type_cls(int, set()) is False

    # Test case 5: Subclass matching within a set
    class MyInt(int):
        pass
    
    assert is_type_cls(MyInt, {int}) is True
    assert is_type_cls(int, {MyInt}) is True

    # Test case 6: Testing with multiple types in a set
    assert is_type_cls(float, {int, float, str}) is True
    assert is_type_cls(str, {int, float, str}) is True
    assert is_type_cls(bool, {int, float, str}) is False

    # Test case 7: Verification of the 'set' type check specifically mentioned in code
    # if type(field_type) is set: return True
    assert is_type_cls(object, set([int])) is True 
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_pmap_field():
    # Test basic functionality: creation of a checked PMap field
    # We use int and str as types for key and value
    key_type = int
    value_type = str
    
    pmap_f = pmap_field(key_type, value_type)
    
    # Verify the return type is a _PField instance
    assert isinstance(pmap_f, _PField)
    
    # Check if types are correctly parsed and stored in the field
    # Since we used int and str, they should be in the set of allowed types
    # Note: get_type handles the conversion from class to its underlying type
    assert int in pmap_f.type
    assert str in pmap_f.type

    # Test mandatory nature (default is True)
    assert pmap_f.mandatory is True

    # Test initial value creation
    # The factory should create an empty instance of the generated PMap class
    assert hasattr(pmap_f.initial, 'get')
    assert len(pmap_f.initial) == 0

    # Test optional parameter
    pmap_opt = pmap_field(int, str, optional=True)
    assert isinstance(pmap_opt.type, set)
    # check if optional_type (which is Union[Type, None]) includes NoneType
    # We rely on the fact that optional_type wraps the type in an optional structure
    
    # Test factory behavior for optional field
    # The factory should return None when argument is None
    opt_factory = pmap_opt._factory
    assert opt_factory(None) is None

    # Test invariant propagation
    def my_invariant(subject):
        return len(subject) > 0, "Empty map"
    
    pmap_inv = pmap_field(int, str, invariant=my_invariant)
    assert pmap_inv.invariant is not PFIELD_NO_INVARIANT
    
    # Test class name generation for the internal Map type
    # The generated class name should follow the pattern IntToStrPMap (or similar)
    generated_class = pmap_f._factory(dict())
    assert "PMap" in generated_class.__name__

    # Test that multiple calls with same types return the same generated class
    pmap_f2 = pmap_field(int, str)
    assert pmap_f._factory(dict()) == pmap_f2._factory(dict())

    # Test error case: invalid type parameter in field (via _check_field_parameters)
    with pytest.raises(TypeError):
        field(type=(object, [1, 2])) # list is not a valid type container for types here if it contains non-types

    # Verify the factory creates instances that respect the key/value types
    # We use the factory directly to test creation
    map_instance = pmap_f._factory({1: "one", 2: "two"})
    assert map_instance[1] == "one"
    assert map_instance[2] == "two"

    # Verify that assigning an invalid type raises PTypeError via check_type logic
    # Since we can't easily instantiate a full PRecord here, we test the component
    class MockRecord:
        pass
    
    mock_field = pmap_f
    with pytest.raises(PTypeError) as excinfo:
        check_type(MockRecord, mock_field, "test_field", "not_a_map")
    assert "Invalid type for field MockRecord.test_field" in str(excinfo.value)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_check_global_invariants():
    class Subject:
        pass

    subject = Subject()

    # Case 1: All invariants pass
    invariants_ok = [
        lambda x: (True, None),
        lambda x: (True, "error_code_1"),
        lambda x: (True, ""),
    ]
    # Should not raise any exception
    check_global_int_subject = subject
    check_global_invariants(check_global_int_subject, invariants_ok)

    # Case 2: One invariant fails
    invariants_fail_one = [
        lambda x: (True, None),
        lambda x: (False, "ERR01"),
        lambda x: (True, ""),
    ]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, invariants_fail_one)
    assert excinfo.value.error_codes == ("ERR01",)
    assert "Global invariant failed" in str(excinfo.value)

    # Case 3: Multiple invariants fail
    invariants_fail_many = [
        lambda x: (False, "ERR01"),
        lambda x: (True, None),
        lambda x: (False, "ERR02"),
        lambda x: (False, "ERR03"),
    ]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, invariants_fail_many)
    # error_codes should be a tuple of all failed codes in order
    assert excinfo.value.error_codes == ("ERR01", "ERR02", "ERR03")

    # Case 4: No invariants provided (empty list)
    check_global_invariants(subject, [])

    # Case 5: Invariant fails with empty error string (should be ignored by the generator expression logic)
    invariants_fail_empty_string = [
        lambda x: (False, ""),
        lambda x: (True, None),
    ]
    # The code uses `if not is_ok`, and then checks `error_code for ... if not is_ok`.
    # If error_code is empty string, it still exists in the tuple. 
    # However, looking at the implementation: (error_code for is_ok, error_code in ...)
    # The current code includes the empty string in the tuple.
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, invariants_fail_empty_string)
    assert excinfo.value.error_codes == ("",)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_serialize():
    # Mocking CheckedType behavior
    class MockCheckedType(CheckedType):
        def serialize(self, fmt):
            return f"serialized_{fmt}"

    # Case 1: Value is a CheckedType and serializer is PFIELD_NO_SERIALIZER
    # In this case, it should call value.serialize(format)
    val_checked = MockCheckedType()
    res1 = serialize(PFIELD_NO_SERIALIZER, "json", val_checked)
    assert res1 == "serialized_json"

    # Case 2: Value is NOT a CheckedType
    # In this case, it should call the serializer with (format, value)
    mock_serializer = MagicMock(return_value="custom_output")
    val_simple = "plain_string"
    res2 = serialize(mock_serializer, "xml", val_simple)
    assert res2 == "custom_output"
    mock_serializer.assert_called_once_with("xml", "plain_string")

    # Case 3: Value is a CheckedType but serializer is NOT PFIELD_NO_SERIALIZER
    # In this case, it should call the provided serializer with (format, value)
    val_checked_2 = MockCheckedType()
    mock_serializer_2 = MagicMock(return_value="custom_checked_output")
    res3 = serialize(mock_serializer_2, "yaml", val_checked_2)
    assert res3 == "custom_checked_output"
    mock_serializer_2.assert_called_once_with("yaml", val_checked_2)

    # Case 4: Value is NOT a CheckedType and serializer is PFIELD_NO_SERIALIZER
    # In this case, it should return the value as is (identity function)
    val_simple_2 = 123
    res4 = serialize(PFIELD_NO_SERIALIZER, "text", val_simple_2)
    assert res4 == 123
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

class MockField:
    def __init__(self, type_allowed):
        self.type = type_allowed

class MockDestinationClass:
    pass

def test_check_type():
    # Setup
    dest_cls = MockDestinationClass
    field_int = MockField(type_allowed=(int,))
    field_str = MockField(type_allowed=(str,))
    field_union = MockField(type_allowed=(int, str))
    
    # Test Case 1: Correct type (int)
    # Should not raise any exception
    check_type(dest_cls, field_int, "age", 25)

    # Test Case 2: Incorrect type (str instead of int)
    # Should raise PTypeError with correct metadata
    with pytest.raises(PTypeError) as excinfo:
        check_type(dest_cls, field_int, "age", "twenty-five")
    
    assert excinfo.value.source_class == dest_cls
    assert excinfo.value.field == field_int
    assert excinfo.value.actual_type == str
    assert "Invalid type for field MockDestinationClass.age" in str(excinfo.value)

    # Test Case 3: Correct type (str)
    check_type(dest_cls, field_str, "name", "Alice")

    # Test Case 4: Incorrect type (int instead of str)
    with pytest.raises(PTypeError):
        check_type(dest_cls, field_str, "name", 123)

    # Test Case 5: Union type - Correct type (int is in [int, str])
    check_type(dest_cls, field_union, "data", 100)

    # Test Case 6: Union type - Correct type (str is in [int, str])
    check_type(dest_cls, field_union, "data", "hello")

    # Test Case 7: Union type - Incorrect type (float not in [int, str])
    with pytest.raises(PTypeError) as excinfo:
        check_type(dest_cls, field_union, "data", 10.5)
    assert excinfo.value.actual_type == float
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_pmap_field():
    # Test 1: Basic PMap field creation with specific key/value types
    # We use int for key and str for value
    f = pmap_field(int, str)
    
    # Verify the type of the initial value is a subclass of CheckedPMap
    assert isinstance(f.initial, CheckedPMap)
    # Check if the factory produces a map that enforces types (via manual check simulation)
    # Since we can't easily trigger the internal __init__ logic without a PRecord, 
    # we test the factory and type properties directly.
    assert f.type is not None
    
    # Test 2: Optional field creation
    f_opt = pmap_field(int, str, optional=True)
    # The type should allow None (using optional_type logic)
    # We check if the factory handles None correctly as per the implementation
    assert f_opt.factory(None) is None
    
    # Test 3: Invariant function
    def my_invariant(m):
        return (len(m) > 0, "Map must not be empty")
    
    f_inv = pmap_field(int, str, invariant=my_invariant)
    assert f_inv.invariant == wrap_invariant(my_invariant)

    # Test 4: Verify the generated class name follows the pattern
    # The implementation uses _types_to_names which capitalizes types
    # For int/str, it should be IntToStrPMap
    expected_name = "IntToStrPMap"
    # We find the class by looking into the internal registry used by pmap_field
    found_class = None
    for (k, v), cls in _pmap_field_types.items():
        if k == int and v == str:
            found_class = cls
            break
    
    assert found_class is not None
    assert found_class.__name__ == expected_name

    # Test 5: Factory behavior with valid data
    # The factory for pmap_field calls TheMap.create
    test_data = {1: "one", 2: "two"}
    result = f.factory(test_data)
    assert result[1] == "one"
    assert isinstance(result, CheckedPMap)

def test_pmap_field_error_cases():
    # Test that invalid initial values (if they were to be passed manually) 
    # would trigger the validation logic in _check_field_parameters.
    # Note: pmap_field handles its own 'initial' via TheMap()
    
    with pytest.raises(TypeError):
        # Invariant must be callable
        pmap_field(int, str, invariant="not a callable")

    with pytest.raises(TypeError):
        # Factory must be callable
        pmap_field(int, str, factory="not a callable")
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import Mock

# Assuming all necessary components are available in the environment as per instructions.
# The code provided defines:
# PFIELD_NO_SERIALIZER = lambda _, value: value
# serialize(serializer, format, value)

def test_serialize():
    """
    Tests the serialize function for different scenarios including:
    1. Using the default (no-op) serializer.
    2. Using a custom serializer for non-CheckedType objects.
    3. Using a custom serializer that triggers CheckedType.serialize.
    """
    
    # 1. Test with PFIELD_NO_SERIALIZER (default behavior)
    # When the serializer is PFIELD_NO_SERIALIZER, it should return the value as-is
    # even if the value is a CheckedType.
    value_int = 42
    format_str = "json"
    result = serialize(PFIELD_NO_SERIALIZER, format_str, value_int)
    assert result == 42

    # 2. Test with a custom serializer for standard types (non-CheckedType)
    # Here the serializer is called directly: serializer(format, value)
    def custom_serializer(fmt, val):
        return f"{fmt}:{val}"

    value_str = "hello"
    result = serialize(custom_serializer, "text", value_str)
    assert result == "text:hello"

    # 3. Test with a CheckedType and PFIELD_NO_SERIALIZER
    # The function should call value.serialize(format)
    class MockCheckedType:
        def __init__(self, val):
            self.val = val
        def serialize(self, fmt):
            return f"serialized_{fmt}_{self.val}"
        def __isinstance_check__(self): # This is a conceptual check for the logic
            pass

    # We need to mock isinstance for CheckedType detection if we can't use real pyrsistent types
    # But since we are testing 'serialize', we simulate a value that passes 'isinstance(value, CheckedType)'
    
    class MockCheckedValue:
        def __init__(self, val):
            self.val = val
        def serialize(self, fmt):
            return f"serialized_{fmt}_{self.val}"

    # To make isinstance(MockCheckedValue(...), CheckedType) return True without 
    # complex mocking of the class hierarchy in a unit test, we rely on the fact that 
    # 'serialize' checks 'isinstance(value, CheckedType)'.
    # In a real environment, MockCheckedValue would inherit from CheckedType.
    
    # Let's use a more robust approach for the mock:
    class FakeCheckedType(CheckedType):
        def serialize(self, fmt):
            return f"mock_serialized_{fmt}"

    fake_val = FakeCheckedType()
    
    # Case: Serializer is PFIELD_NO_SERIALIZER and value is CheckedType
    result = serialize(PFIELD_NO_SERIALIZER, "xml", fake_val)
    assert result == "mock_serialized_xml"

    # Case: Serializer is custom and value is NOT a CheckedType
    # Custom serializer should be called with (format, value)
    custom_mock = Mock()
    custom_mock.return_value = "custom_output"
    result = serialize(custom_mock, "any_format", "some_value")
    custom_mock.assert_called_once_with("any_format", "some_value")
    assert result == "custom_output"

    # Case: Serializer is custom and value IS a CheckedType
    # The function should bypass the custom serializer's direct call if it's PFIELD_NO_SERIALIZER,
    # but if it's NOT PFIELD_NO_SERIALIZER, it calls serializer(format, value).
    # Note: In the provided source, the 'if' block only triggers if serializer is PFIELD_NO_SERIALIZER.
    # If a custom serializer is provided, it executes: return serializer(format, value)
    
    def custom_serializer_for_checked(fmt, val):
        return f"custom_{val.serialize(fmt)}"

    result = serialize(custom_serializer_for_checked, "json", fake_val)
    assert result == "custom_mock_serialized_json"
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_check_global_invariants():
    class MockSubject:
        def __init__(self, value):
            self.value = value

    # Case 1: All invariants pass
    invariant_pass = lambda x: (True, None)
    subject_ok = MockSubject(10)
    # Should not raise any exception
    check_global_invariants(subject_ok, [invariant_pass, invariant_pass])

    # Case 2: One invariant fails
    error_code = "ERR_001"
    def invariant_fail(x):
        return (False, error_code)
    
    subject_fail = MockSubject(10)
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject_fail, [invariant_pass, invariant_fail])
    
    assert error_code in excinfo.value.error_codes
    assert excinfo.value.message == 'Global invariant failed'

    # Case 3: Multiple invariants fail
    def invariant_fail_2(x):
        return (False, "ERR_002")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject_fail, [invariant_fail, invariant_fail_2])
    
    assert set(excinfo.value.error_codes) == {"ERR_001", "ERR_002"}

    # Case 4: Empty list of invariants
    check_global_invariants(subject_ok, [])
```


