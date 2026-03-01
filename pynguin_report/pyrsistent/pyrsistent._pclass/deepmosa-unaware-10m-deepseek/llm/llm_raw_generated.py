####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PClassEvolver_remove():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    # Create initial instance
    instance = TestClass(x=1, y=2, z=3)
    
    # Get evolver and remove a field
    evolver = instance.evolver()
    evolver.remove('y')
    
    # Check that field is removed in persistent instance
    result = evolver.persistent()
    assert not hasattr(result, 'y')
    assert result.x == 1
    assert result.z == 3
    
    # Verify original instance is unchanged
    assert instance.y == 2
    
    # Test removing non-existent field raises AttributeError
    evolver2 = instance.evolver()
    try:
        evolver2.remove('non_existent')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert str(e) == "'non_existent'"
    
    # Test removing field and then setting it again
    evolver3 = instance.evolver()
    evolver3.remove('x')
    evolver3.set('x', 10)
    result3 = evolver3.persistent()
    assert result3.x == 10
    assert result3.y == 2
    assert result3.z == 3
    
    # Test removing all fields
    evolver4 = instance.evolver()
    evolver4.remove('x')
    evolver4.remove('y')
    evolver4.remove('z')
    result4 = evolver4.persistent()
    assert not hasattr(result4, 'x')
    assert not hasattr(result4, 'y')
    assert not hasattr(result4, 'z')
    
    # Test that remove returns self for chaining
    evolver5 = instance.evolver()
    evolver5.remove('x').remove('y')
    result5 = evolver5.persistent()
    assert not hasattr(result5, 'x')
    assert not hasattr(result5, 'y')
    assert result5.z == 3
    
    # Test that removing a field removes it from factory_fields
    evolver6 = instance.evolver()
    evolver6.set('x', 100)  # This adds 'x' to factory_fields
    evolver6.remove('x')    # This should remove 'x' from factory_fields
    result6 = evolver6.persistent()
    assert result6.x == 1  # Should revert to original value since it was removed


# LLM-generated content at query #2
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    # Test that equal objects have equal hash
    obj1 = SimpleClass(x=1, y=2)
    obj2 = SimpleClass(x=1, y=2)
    assert hash(obj1) == hash(obj2)
    
    # Test that different objects have different hash (with high probability)
    obj3 = SimpleClass(x=1, y=3)
    assert hash(obj1) != hash(obj3)
    
    # Test hash with None values
    class ClassWithNone(PClass):
        a = field()
        b = field()
    
    obj4 = ClassWithNone(a=None, b=5)
    obj5 = ClassWithNone(a=None, b=5)
    assert hash(obj4) == hash(obj5)
    
    # Test hash consistency across multiple calls
    obj6 = SimpleClass(x=10, y=20)
    hash1 = hash(obj6)
    hash2 = hash(obj6)
    assert hash1 == hash2
    
    # Test hash with string values
    class StringClass(PClass):
        name = field()
        value = field()
    
    obj7 = StringClass(name="test", value=100)
    obj8 = StringClass(name="test", value=100)
    assert hash(obj7) == hash(obj8)
    
    # Test hash with mixed types
    class MixedClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj9 = MixedClass(a=1, b="hello", c=None)
    obj10 = MixedClass(a=1, b="hello", c=None)
    assert hash(obj9) == hash(obj10)
    
    # Test that hash differs when field values differ
    obj11 = MixedClass(a=1, b="hello", c=None)
    obj12 = MixedClass(a=2, b="hello", c=None)
    assert hash(obj11) != hash(obj12)
    
    # Test hash with mandatory fields only
    class MandatoryClass(PClass):
        required = field(mandatory=True)
    
    obj13 = MandatoryClass(required="value1")
    obj14 = MandatoryClass(required="value1")
    obj15 = MandatoryClass(required="value2")
    assert hash(obj13) == hash(obj14)
    assert hash(obj13) != hash(obj15)


# LLM-generated content at query #3
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test basic serialization without custom serializer
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=10, y="test")
    result = obj.serialize()
    assert result == {"x": 10, "y": "test"}
    
    # Test serialization with custom serializer function
    def custom_serializer(value):
        return f"serialized_{value}"
    
    class WithSerializer(PClass):
        data = field(serializer=custom_serializer)
        other = field()
    
    obj = WithSerializer(data="value", other=42)
    result = obj.serialize()
    assert result == {"data": "serialized_value", "other": 42}
    
    # Test serialization with format parameter
    def format_serializer(format, value):
        if format == "json":
            return str(value)
        return value
    
    class FormatClass(PClass):
        num = field(serializer=format_serializer)
        text = field()
    
    obj = FormatClass(num=100, text="hello")
    result = obj.serialize(format="json")
    assert result == {"num": "100", "text": "hello"}
    
    # Test that format is passed correctly to serializer
    result = obj.serialize(format="other")
    assert result == {"num": 100, "text": "hello"}
    
    # Test serialization with missing optional fields
    class OptionalFields(PClass):
        required = field(mandatory=True)
        optional = field()
    
    obj = OptionalFields(required="req")
    result = obj.serialize()
    assert result == {"required": "req"}
    
    # Test serialization with nested PClass
    class Inner(PClass):
        value = field()
    
    class Outer(PClass):
        inner = field()
        name = field()
    
    inner_obj = Inner(value=5)
    outer_obj = Outer(inner=inner_obj, name="test")
    result = outer_obj.serialize()
    assert result["name"] == "test"
    assert isinstance(result["inner"], dict)
    assert result["inner"]["value"] == 5
    
    # Test that serialize returns a new dict, not the internal representation
    result1 = obj.serialize()
    result2 = obj.serialize()
    assert result1 == result2
    assert result1 is not result2
    
    # Test with field that has initial value
    class WithInitial(PClass):
        x = field(initial=100)
        y = field()
    
    obj = WithInitial(y=50)
    result = obj.serialize()
    assert result == {"x": 100, "y": 50}
    
    # Test with multiple custom serializers
    def double_serializer(value):
        return value * 2
    
    def upper_serializer(value):
        return value.upper()
    
    class MultiSerializer(PClass):
        number = field(serializer=double_serializer)
        text = field(serializer=upper_serializer)
    
    obj = MultiSerializer(number=21, text="hello")
    result = obj.serialize()
    assert result == {"number": 42, "text": "HELLO"}


# LLM-generated content at query #4
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation with fields
    class TestClass(PClass):
        x = field()
        y = field()
    
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    assert isinstance(TestClass._pclass_fields['x'], type(field()))
    assert isinstance(TestClass._pclass_fields['y'], type(field()))
    
    # Test that __slots__ are properly set
    assert '__slots__' in TestClass.__dict__
    slots = TestClass.__slots__
    assert '_pclass_frozen' in slots
    assert 'x' in slots
    assert 'y' in slots
    
    # Test that __weakref__ is only added to top-level PClass
    assert '__weakref__' in slots
    
    # Test inheritance - child class should not have __weakref__ in slots
    class ChildClass(TestClass):
        z = field()
    
    child_slots = ChildClass.__slots__
    assert 'z' in child_slots
    assert '_pclass_frozen' in child_slots
    assert '__weakref__' not in child_slots
    
    # Test invariants storage
    assert hasattr(TestClass, '_pclass_invariants')
    assert isinstance(TestClass._pclass_invariants, tuple)
    
    # Test with invariant
    class InvariantClass(PClass):
        x = field()
        
        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"
    
    assert len(InvariantClass._pclass_invariants) == 1
    assert InvariantClass._pclass_invariants[0].__name__ == 'check_x'
    
    # Test that metaclass properly handles multiple inheritance
    class Mixin:
        pass
    
    class MixedClass(PClass, Mixin):
        a = field()
    
    assert hasattr(MixedClass, '_pclass_fields')
    assert 'a' in MixedClass._pclass_fields
    assert '_pclass_frozen' in MixedClass.__slots__
    
    # Test that fields from parent classes are inherited
    class ParentClass(PClass):
        parent_field = field()
    
    class InheritedClass(ParentClass):
        child_field = field()
    
    assert 'parent_field' in InheritedClass._pclass_fields
    assert 'child_field' in InheritedClass._pclass_fields
    assert 'parent_field' in InheritedClass.__slots__
    assert 'child_field' in InheritedClass.__slots__


# LLM-generated content at query #5
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class SimpleClass(PClass):
        x = field()
        y = field()

    # Test basic repr with multiple fields
    instance = SimpleClass(x=10, y="hello")
    assert repr(instance) == "SimpleClass(x=10, y='hello')"

    # Test repr with single field
    class SingleFieldClass(PClass):
        name = field()

    instance = SingleFieldClass(name="test")
    assert repr(instance) == "SingleFieldClass(name='test')"

    # Test repr with no fields (edge case)
    class EmptyClass(PClass):
        pass

    instance = EmptyClass()
    assert repr(instance) == "EmptyClass()"

    # Test repr with special characters in string values
    instance = SimpleClass(x=1, y="line\nbreak")
    assert repr(instance) == "SimpleClass(x=1, y='line\\nbreak')"

    # Test repr with None values
    instance = SimpleClass(x=None, y=None)
    assert repr(instance) == "SimpleClass(x=None, y=None)"

    # Test repr with boolean values
    instance = SimpleClass(x=True, y=False)
    assert repr(instance) == "SimpleClass(x=True, y=False)"

    # Test repr with list values
    instance = SimpleClass(x=[1, 2, 3], y="test")
    assert repr(instance) == "SimpleClass(x=[1, 2, 3], y='test')"

    # Test repr with dict values
    instance = SimpleClass(x={'a': 1}, y=2)
    assert repr(instance) == "SimpleClass(x={'a': 1}, y=2)"

    # Test repr preserves field order as defined in class
    class OrderedClass(PClass):
        a = field()
        b = field()
        c = field()

    instance = OrderedClass(c=3, a=1, b=2)
    assert repr(instance) == "OrderedClass(a=1, b=2, c=3)"

    # Test repr with numeric field names
    class NumericFieldClass(PClass):
        field1 = field()
        field2 = field()

    instance = NumericFieldClass(field1=100, field2=200)
    assert repr(instance) == "NumericFieldClass(field1=100, field2=200)"


# LLM-generated content at query #6
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=10, y=20)
    result = obj.serialize()
    assert result == {'x': 10, 'y': 20}

    class TestClassWithSerializer(PClass):
        x = field(serializer=lambda v, f: v * 2)
        y = field()

    obj = TestClassWithSerializer(x=5, y=10)
    result = obj.serialize()
    assert result == {'x': 10, 'y': 10}

    obj = TestClassWithSerializer(x=3, y=7)
    result = obj.serialize(format='custom')
    assert result == {'x': 6, 'y': 7}

    class TestClassPartial(PClass):
        x = field(mandatory=True)
        y = field(initial=100)

    obj = TestClassPartial(x=50)
    result = obj.serialize()
    assert result == {'x': 50, 'y': 100}

    class TestClassEmpty(PClass):
        pass

    obj = TestClassEmpty()
    result = obj.serialize()
    assert result == {}

    class TestClassNested(PClass):
        x = field(serializer=lambda v, f: f"value_{v}")

    obj = TestClassNested(x=42)
    result = obj.serialize()
    assert result == {'x': 'value_42'}


# LLM-generated content at query #7
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field
    import pickle
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test basic pickling and unpickling
    obj = TestClass(x=10, y="test")
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    
    assert obj == unpickled
    assert obj.x == unpickled.x
    assert obj.y == unpickled.y
    
    # Test that __reduce__ returns correct tuple structure
    reduce_result = obj.__reduce__()
    assert len(reduce_result) == 2
    assert reduce_result[0] is _restore_pickle
    assert len(reduce_result[1]) == 3
    assert reduce_result[1][0] is TestClass
    
    # Test data in reduce tuple
    class_from_reduce, data_from_reduce, *_ = reduce_result[1]
    assert class_from_reduce is TestClass
    assert data_from_reduce == {'x': 10, 'y': 'test'}
    
    # Test with missing optional field
    class OptionalClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=False, initial=5)
    
    obj2 = OptionalClass(x=20)  # y will use default
    reduce_result2 = obj2.__reduce__()
    data_from_reduce2 = reduce_result2[1][1]
    assert data_from_reduce2 == {'x': 20, 'y': 5}
    
    # Test that pickled object maintains immutability
    unpickled2 = pickle.loads(pickle.dumps(obj2))
    with pytest.raises(AttributeError):
        unpickled2.x = 30
    
    # Test with nested PClass
    class Inner(PClass):
        a = field()
    
    class Outer(PClass):
        inner = field(type=Inner)
        value = field()
    
    inner_obj = Inner(a=100)
    outer_obj = Outer(inner=inner_obj, value="outer")
    unpickled_outer = pickle.loads(pickle.dumps(outer_obj))
    
    assert outer_obj == unpickled_outer
    assert outer_obj.inner.a == unpickled_outer.inner.a
    assert isinstance(unpickled_outer.inner, Inner)


# LLM-generated content at query #8
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class SimpleClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = SimpleClass(x=1, y=2)
    obj2 = SimpleClass(x=1, y=2)
    assert obj1 == obj2
    assert not (obj1 != obj2)

    # Test inequality with different values
    obj3 = SimpleClass(x=1, y=3)
    assert obj1 != obj3
    assert not (obj1 == obj3)

    # Test inequality with different field values
    obj4 = SimpleClass(x=2, y=2)
    assert obj1 != obj4

    # Test equality with same object
    assert obj1 == obj1

    # Test equality with different class instances
    class OtherClass(PClass):
        x = field()
        y = field()

    obj5 = OtherClass(x=1, y=2)
    assert obj1 != obj5
    assert not (obj1 == obj5)

    # Test equality with non-PClass object
    assert obj1 != "not a pclass"
    assert not (obj1 == "not a pclass")

    # Test equality with None
    assert obj1 != None
    assert not (obj1 == None)

    # Test equality with missing fields (optional fields)
    class ClassWithOptional(PClass):
        x = field(mandatory=True)
        y = field(mandatory=False)

    obj6 = ClassWithOptional(x=1)
    obj7 = ClassWithOptional(x=1)
    assert obj6 == obj7

    # Test equality when one has optional field set, other doesn't
    obj8 = ClassWithOptional(x=1, y=2)
    assert obj6 != obj8

    # Test equality with nested PClass objects
    class InnerClass(PClass):
        value = field()

    class OuterClass(PClass):
        inner = field()
        name = field()

    inner1 = InnerClass(value=10)
    inner2 = InnerClass(value=10)
    outer1 = OuterClass(inner=inner1, name="test")
    outer2 = OuterClass(inner=inner2, name="test")
    assert outer1 == outer2

    # Test inequality with different nested PClass values
    inner3 = InnerClass(value=20)
    outer3 = OuterClass(inner=inner3, name="test")
    assert outer1 != outer3

    # Test hash consistency with equality
    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)

    # Test equality with default values
    class ClassWithDefault(PClass):
        x = field()
        y = field(initial=10)

    obj9 = ClassWithDefault(x=1)
    obj10 = ClassWithDefault(x=1, y=10)
    assert obj9 == obj10

    # Test equality with different default values
    obj11 = ClassWithDefault(x=1, y=20)
    assert obj9 != obj11


# LLM-generated content at query #9
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=10, y="test")
    result = obj.serialize()
    
    assert result == {"x": 10, "y": "test"}
    
    class TestClassWithSerializer(PClass):
        x = field(type=int, serializer=lambda v, f: v * 2)
        y = field(type=str, serializer=lambda v, f: v.upper())
    
    obj = TestClassWithSerializer(x=5, y="hello")
    result = obj.serialize()
    
    assert result == {"x": 10, "y": "HELLO"}
    
    result_with_format = obj.serialize(format="custom")
    assert result_with_format == {"x": 10, "y": "HELLO"}
    
    class TestClassPartial(PClass):
        x = field(mandatory=True)
        y = field(mandatory=False)
    
    obj = TestClassPartial(x=1)
    result = obj.serialize()
    
    assert result == {"x": 1}
    
    class TestClassNested(PClass):
        inner = field(type=TestClass)
    
    inner_obj = TestClass(x=1, y=2)
    nested_obj = TestClassNested(inner=inner_obj)
    result = nested_obj.serialize()
    
    assert result == {"inner": inner_obj}
    
    class TestClassEmpty(PClass):
        pass
    
    obj = TestClassEmpty()
    result = obj.serialize()
    
    assert result == {}


# LLM-generated content at query #10
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class SimpleClass(PClass):
        x = field()
        y = field()

    obj = SimpleClass(x=10, y="test")
    assert repr(obj) == "SimpleClass(x=10, y='test')"

    class EmptyClass(PClass):
        pass

    obj = EmptyClass()
    assert repr(obj) == "EmptyClass()"

    class NestedClass(PClass):
        name = field()
        value = field()

    obj = NestedClass(name="nested", value=SimpleClass(x=1, y=2))
    expected = "NestedClass(name='nested', value=SimpleClass(x=1, y=2))"
    assert repr(obj) == expected

    class SpecialCharsClass(PClass):
        field_with_underscore = field()
        field_with_dash = field()

    obj = SpecialCharsClass(field_with_underscore="a", field_with_dash="b")
    assert repr(obj) == "SpecialCharsClass(field_with_underscore='a', field_with_dash='b')"

    class DefaultFieldClass(PClass):
        x = field(initial=5)
        y = field()

    obj = DefaultFieldClass(y=10)
    assert repr(obj) == "DefaultFieldClass(x=5, y=10)"

    class BooleanClass(PClass):
        flag = field()
        value = field()

    obj = BooleanClass(flag=True, value=False)
    assert repr(obj) == "BooleanClass(flag=True, value=False)"

    class NoneClass(PClass):
        x = field()
        y = field()

    obj = NoneClass(x=None, y="something")
    assert repr(obj) == "NoneClass(x=None, y='something')"

    class ListFieldClass(PClass):
        items = field()

    obj = ListFieldClass(items=[1, 2, 3])
    assert repr(obj) == "ListFieldClass(items=[1, 2, 3])"

    class DictFieldClass(PClass):
        data = field()

    obj = DictFieldClass(data={"a": 1, "b": 2})
    assert repr(obj) == "DictFieldClass(data={'a': 1, 'b': 2})"


# LLM-generated content at query #11
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field
    from pyrsistent._checked_types import InvariantException

    class SimpleClass(PClass):
        x = field()
        y = field(mandatory=True)
        z = field(initial=10)

    # Test basic instantiation with all required fields
    obj = SimpleClass(x=5, y=20)
    assert obj.x == 5
    assert obj.y == 20
    assert obj.z == 10  # Default initial value

    # Test with all fields provided
    obj = SimpleClass(x=1, y=2, z=3)
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 3

    # Test missing mandatory field
    try:
        SimpleClass(x=5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "SimpleClass.y" in e.missing_fields

    # Test extra fields not allowed
    try:
        SimpleClass(x=1, y=2, extra=3)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "extra" in str(e)

    # Test type checking
    class TypedClass(PClass):
        value = field(type=int)

    try:
        TypedClass(value="not an int")
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test successful type checking
    obj = TypedClass(value=42)
    assert obj.value == 42

    # Test field invariants
    class PositiveClass(PClass):
        number = field(invariant=lambda x: (x > 0, "Must be positive"))

    try:
        PositiveClass(number=-5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Must be positive" in str(e)

    # Test successful invariant
    obj = PositiveClass(number=5)
    assert obj.number == 5

    # Test callable initial
    class CallableInitialClass(PClass):
        counter = field(initial=lambda: 100)

    obj = CallableInitialClass()
    assert obj.counter == 100

    # Test multiple missing fields
    class MultiMandatoryClass(PClass):
        a = field(mandatory=True)
        b = field(mandatory=True)
        c = field(initial=0)

    try:
        MultiMandatoryClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 2
        assert "MultiMandatoryClass.a" in e.missing_fields
        assert "MultiMandatoryClass.b" in e.missing_fields

    # Test with factory_fields parameter
    obj = SimpleClass(x=1, y=2, z=3, _factory_fields={'x'})
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 3

    # Test ignore_extra parameter
    class IgnoreExtraClass(PClass):
        a = field()
        b = field()

    obj = IgnoreExtraClass(a=1, b=2, c=3, ignore_extra=True)
    assert obj.a == 1
    assert obj.b == 2
    assert not hasattr(obj, 'c')

    # Test global invariants
    class GlobalInvariantClass(PClass):
        x = field()
        y = field()

        @staticmethod
        def __invariant__(obj):
            return obj.x + obj.y > 0, "Sum must be positive"

    try:
        GlobalInvariantClass(x=-5, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Sum must be positive" in str(e)

    # Test successful global invariant
    obj = GlobalInvariantClass(x=3, y=2)
    assert obj.x == 3
    assert obj.y == 2

    # Test that object is frozen after creation
    obj = SimpleClass(x=1, y=2)
    try:
        obj.x = 10
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test equality
    obj1 = SimpleClass(x=1, y=2, z=3)
    obj2 = SimpleClass(x=1, y=2, z=3)
    obj3 = SimpleClass(x=2, y=2, z=3)
    assert obj1 == obj2
    assert obj1 != obj3
    assert obj1 != "not a PClass"

    # Test hash
    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)

    # Test repr
    repr_str = repr(obj1)
    assert "SimpleClass" in repr_str
    assert "x=1" in repr_str
    assert "y=2" in repr_str
    assert "z=3" in repr_str


# LLM-generated content at query #12
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test 1: Basic serialization without custom serializer
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=10, y="test")
    result = obj.serialize()
    assert result == {"x": 10, "y": "test"}
    
    # Test 2: Serialization with custom serializer function
    def custom_serializer(value, format=None):
        if isinstance(value, int):
            return f"int:{value}"
        return value
    
    class CustomSerializedClass(PClass):
        a = field(serializer=custom_serializer)
        b = field()
    
    obj = CustomSerializedClass(a=42, b="normal")
    result = obj.serialize()
    assert result == {"a": "int:42", "b": "normal"}
    
    # Test 3: Serialization with format parameter passed to serializer
    def format_aware_serializer(value, format=None):
        if format == "json":
            return {"value": value}
        return value
    
    class FormatAwareClass(PClass):
        data = field(serializer=format_aware_serializer)
    
    obj = FormatAwareClass(data="test_data")
    result = obj.serialize(format="json")
    assert result == {"data": {"value": "test_data"}}
    
    # Test 4: Serialization with missing optional fields
    class OptionalFieldClass(PClass):
        required = field(mandatory=True)
        optional = field()
    
    obj = OptionalFieldClass(required="must_have")
    result = obj.serialize()
    assert result == {"required": "must_have"}
    
    # Test 5: Serialization with nested PClass objects
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
        name = field()
    
    inner = InnerClass(value=100)
    outer = OuterClass(inner=inner, name="container")
    result = outer.serialize()
    assert result["name"] == "container"
    assert isinstance(result["inner"], dict)
    assert result["inner"]["value"] == 100
    
    # Test 6: Serialization with field that has initial value
    class InitialValueClass(PClass):
        x = field(initial=5)
        y = field()
    
    obj = InitialValueClass(y=10)
    result = obj.serialize()
    assert result == {"x": 5, "y": 10}
    
    # Test 7: Serialization with multiple custom serializers
    def int_serializer(value, format=None):
        return value * 2
    
    def str_serializer(value, format=None):
        return value.upper()
    
    class MultiSerializerClass(PClass):
        number = field(serializer=int_serializer)
        text = field(serializer=str_serializer)
    
    obj = MultiSerializerClass(number=21, text="hello")
    result = obj.serialize()
    assert result == {"number": 42, "text": "HELLO"}
    
    # Test 8: Serialization returns new dict, doesn't modify original
    obj = SimpleClass(x=1, y=2)
    result1 = obj.serialize()
    result2 = obj.serialize()
    assert result1 == result2
    assert result1 is not result2
    
    # Test 9: Serialization with None values
    class NullableClass(PClass):
        a = field()
        b = field()
    
    obj = NullableClass(a=None, b="not_null")
    result = obj.serialize()
    assert result == {"a": None, "b": "not_null"}


# LLM-generated content at query #13
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test 1: Basic serialization without custom serializer
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=10, y="test")
    result = obj.serialize()
    assert result == {"x": 10, "y": "test"}
    
    # Test 2: Serialization with custom serializer function
    def custom_serializer(value):
        return f"serialized_{value}"
    
    class WithSerializer(PClass):
        data = field(serializer=custom_serializer)
        other = field()
    
    obj = WithSerializer(data="value", other=42)
    result = obj.serialize()
    assert result == {"data": "serialized_value", "other": 42}
    
    # Test 3: Serialization with format parameter
    def format_serializer(format, value):
        if format == "json":
            return {"value": value}
        return value
    
    class FormatSerializer(PClass):
        item = field(serializer=format_serializer)
    
    obj = FormatSerializer(item="data")
    result = obj.serialize(format="json")
    assert result == {"item": {"value": "data"}}
    
    # Test 4: Serialization with missing optional field
    class OptionalFields(PClass):
        required = field(mandatory=True)
        optional = field(mandatory=False, initial=100)
    
    obj = OptionalFields(required=50)
    result = obj.serialize()
    assert result == {"required": 50, "optional": 100}
    
    # Test 5: Serialization with factory fields
    class FactoryClass(PClass):
        value = field()
        
        @classmethod
        def create_with_double(cls, **kwargs):
            if 'value' in kwargs:
                kwargs['value'] = kwargs['value'] * 2
            return cls(**kwargs)
    
    obj = FactoryClass(value=5)
    result = obj.serialize()
    assert result == {"value": 5}
    
    # Test 6: Empty serialization for class with no values set
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    result = obj.serialize()
    assert result == {}
    
    # Test 7: Nested serialization with custom serializer
    def nested_serializer(value):
        if isinstance(value, SimpleClass):
            return value.serialize()
        return value
    
    class Container(PClass):
        nested = field(serializer=nested_serializer)
        plain = field()
    
    nested = SimpleClass(x=1, y=2)
    obj = Container(nested=nested, plain="text")
    result = obj.serialize()
    assert result == {"nested": {"x": 1, "y": 2}, "plain": "text"}
    
    # Test 8: Serialization with None values
    class WithNone(PClass):
        data = field()
        none_field = field()
    
    obj = WithNone(data=None, none_field="exists")
    result = obj.serialize()
    assert result == {"data": None, "none_field": "exists"}
    
    # Test 9: Multiple fields with different serializers
    def upper_serializer(value):
        return value.upper() if isinstance(value, str) else value
    
    def double_serializer(value):
        return value * 2 if isinstance(value, (int, float)) else value
    
    class MixedSerializers(PClass):
        text = field(serializer=upper_serializer)
        number = field(serializer=double_serializer)
        plain = field()
    
    obj = MixedSerializers(text="hello", number=3, plain=[1, 2, 3])
    result = obj.serialize()
    assert result == {"text": "HELLO", "number": 6, "plain": [1, 2, 3]}


# LLM-generated content at query #14
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field
    import pickle

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic pickling and unpickling
    obj = TestClass(x=10, y="hello")
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    
    assert unpickled == obj
    assert unpickled.x == 10
    assert unpickled.y == "hello"
    assert isinstance(unpickled, TestClass)

    # Test that __reduce__ returns correct tuple structure
    reduce_result = obj.__reduce__()
    assert len(reduce_result) == 2
    assert reduce_result[0] == obj._restore_pickle
    assert isinstance(reduce_result[1], tuple)
    assert len(reduce_result[1]) == 3
    assert reduce_result[1][0] == TestClass
    assert isinstance(reduce_result[1][1], dict)
    assert reduce_result[1][2] is None

    # Test that data dict contains all fields
    data_dict = reduce_result[1][1]
    assert data_dict == {'x': 10, 'y': 'hello'}

    # Test with missing optional field
    class TestClass2(PClass):
        x = field(mandatory=True)
        y = field(initial=5)

    obj2 = TestClass2(x=20)
    reduce_result2 = obj2.__reduce__()
    data_dict2 = reduce_result2[1][1]
    assert data_dict2 == {'x': 20, 'y': 5}

    # Test that pickling preserves field values correctly
    obj3 = TestClass(x=[1, 2, 3], y={'a': 1})
    pickled3 = pickle.dumps(obj3)
    unpickled3 = pickle.loads(pickled3)
    
    assert unpickled3.x == [1, 2, 3]
    assert unpickled3.y == {'a': 1}

    # Test with nested PClass
    class Inner(PClass):
        value = field()

    class Outer(PClass):
        inner = field()

    inner_obj = Inner(value=42)
    outer_obj = Outer(inner=inner_obj)
    
    pickled_outer = pickle.dumps(outer_obj)
    unpickled_outer = pickle.loads(pickled_outer)
    
    assert unpickled_outer.inner.value == 42
    assert isinstance(unpickled_outer.inner, Inner)

    # Test that __reduce__ works with _restore_pickle function
    restore_func, args = outer_obj.__reduce__()
    restored = restore_func(*args)
    assert restored == outer_obj


# LLM-generated content at query #15
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    # Test basic set with keyword arguments
    obj = TestClass(x=1, y=2, z=3)
    obj2 = obj.set(x=10)
    assert obj2.x == 10
    assert obj2.y == 2
    assert obj2.z == 3
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    obj3 = obj.set('y', 20)
    assert obj3.y == 20
    assert obj3.x == 1
    assert obj3.z == 3

    # Test set with both positional and keyword (positional should be ignored)
    obj4 = obj.set('x', 100, y=200)
    assert obj4.x == 100  # Keyword takes precedence
    assert obj4.y == 200
    assert obj4.z == 3

    # Test set multiple fields
    obj5 = obj.set(x=30, z=50)
    assert obj5.x == 30
    assert obj5.y == 2
    assert obj5.z == 50

    # Test that set returns new instance of same class
    assert isinstance(obj2, TestClass)
    assert obj is not obj2

    # Test with mandatory field
    class MandatoryClass(PClass):
        required = field(mandatory=True)
        optional = field()

    mandatory_obj = MandatoryClass(required=1, optional=2)
    mandatory_obj2 = mandatory_obj.set(required=10)
    assert mandatory_obj2.required == 10
    assert mandatory_obj2.optional == 2

    # Test with initial value field
    class InitialClass(PClass):
        with_initial = field(initial=5)
        regular = field()

    initial_obj = InitialClass(regular=1)
    assert initial_obj.with_initial == 5
    initial_obj2 = initial_obj.set(with_initial=10)
    assert initial_obj2.with_initial == 10
    assert initial_obj2.regular == 1

    # Test that factory fields are properly handled
    class FactoryClass(PClass):
        x = field(type=int, factory=lambda v: v * 2)

    factory_obj = FactoryClass(x=5)
    assert factory_obj.x == 10
    factory_obj2 = factory_obj.set(x=3)
    assert factory_obj2.x == 6  # Factory function applied

    # Test equality after set
    obj = TestClass(x=1, y=2, z=3)
    obj2 = obj.set(x=10)
    obj3 = TestClass(x=10, y=2, z=3)
    assert obj2 == obj3
    assert obj != obj2

    # Test hash consistency
    obj = TestClass(x=1, y=2, z=3)
    obj2 = obj.set(x=10)
    obj3 = TestClass(x=10, y=2, z=3)
    assert hash(obj2) == hash(obj3)
    assert hash(obj) != hash(obj2)

    # Test with nested PClass
    class InnerClass(PClass):
        a = field()

    class OuterClass(PClass):
        inner = field(type=InnerClass)
        value = field()

    inner = InnerClass(a=1)
    outer = OuterClass(inner=inner, value=10)
    new_inner = InnerClass(a=2)
    outer2 = outer.set(inner=new_inner)
    assert outer2.inner == new_inner
    assert outer2.value == 10
    assert outer.inner == inner  # Original unchanged


# LLM-generated content at query #16
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    # Test basic set with keyword arguments
    obj = TestClass(x=1, y=2, z=3)
    obj2 = obj.set(x=10)
    assert obj2.x == 10
    assert obj2.y == 2
    assert obj2.z == 3
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    obj3 = obj.set('y', 20)
    assert obj3.y == 20
    assert obj3.x == 1
    assert obj3.z == 3

    # Test set with both positional and keyword (should use keyword)
    obj4 = obj.set('x', 100, y=200)
    assert obj4.x == 100
    assert obj4.y == 200
    assert obj4.z == 3

    # Test set with multiple fields
    obj5 = obj.set(x=100, z=300)
    assert obj5.x == 100
    assert obj5.y == 2
    assert obj5.z == 300

    # Test that set returns new instance of same class
    assert isinstance(obj2, TestClass)
    assert obj2 is not obj

    # Test with field that has factory
    class FactoryClass(PClass):
        x = field(type=int, factory=lambda v: v * 2)

    factory_obj = FactoryClass(x=5)
    assert factory_obj.x == 10
    factory_obj2 = factory_obj.set(x=3)
    assert factory_obj2.x == 6

    # Test with mandatory field
    class MandatoryClass(PClass):
        x = field(mandatory=True)
        y = field()

    mandatory_obj = MandatoryClass(x=1, y=2)
    mandatory_obj2 = mandatory_obj.set(y=3)
    assert mandatory_obj2.x == 1
    assert mandatory_obj2.y == 3

    # Test that original remains frozen
    obj = TestClass(x=1, y=2, z=3)
    try:
        obj.x = 10
        assert False, "Should not be able to set attribute on frozen instance"
    except AttributeError:
        pass

    # Test with initial values
    class InitialClass(PClass):
        x = field(initial=10)
        y = field()

    initial_obj = InitialClass(y=5)
    assert initial_obj.x == 10
    initial_obj2 = initial_obj.set(x=20)
    assert initial_obj2.x == 20
    assert initial_obj2.y == 5

    # Test equality after set
    obj1 = TestClass(x=1, y=2, z=3)
    obj2 = obj1.set(x=10)
    obj3 = TestClass(x=10, y=2, z=3)
    assert obj2 == obj3
    assert obj1 != obj2

    # Test hash consistency
    obj1 = TestClass(x=1, y=2, z=3)
    obj2 = obj1.set(x=10)
    obj3 = TestClass(x=10, y=2, z=3)
    assert hash(obj2) == hash(obj3)


# LLM-generated content at query #17
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field
    from pyrsistent._checked_types import InvariantException

    # Test basic PClass creation with fields
    class SimpleClass(PClass):
        x = field()
        y = field()

    obj = SimpleClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2

    # Test with default values
    class ClassWithDefaults(PClass):
        x = field()
        y = field(initial=10)

    obj = ClassWithDefaults(x=1)
    assert obj.x == 1
    assert obj.y == 10

    # Test with callable initial
    counter = 0
    def make_default():
        nonlocal counter
        counter += 1
        return counter

    class ClassWithCallableInitial(PClass):
        x = field(initial=make_default)

    obj1 = ClassWithCallableInitial()
    obj2 = ClassWithCallableInitial()
    assert obj1.x == 1
    assert obj2.x == 2

    # Test mandatory field missing
    class ClassWithMandatory(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        ClassWithMandatory(y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "ClassWithMandatory.x" in e.missing_fields

    # Test type checking
    class ClassWithType(PClass):
        x = field(type=int)

    try:
        ClassWithType(x="not an int")
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    obj = ClassWithType(x=42)
    assert obj.x == 42

    # Test field invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class ClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)

    try:
        ClassWithInvariant(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Value must be positive" in e.invariant_errors

    obj = ClassWithInvariant(x=5)
    assert obj.x == 5

    # Test extra fields not allowed
    class SimpleClass2(PClass):
        x = field()

    try:
        SimpleClass2(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'y' are not among the specified fields" in str(e)

    # Test global invariants
    def global_invariant(obj):
        return obj.x + obj.y == 10, "Sum must be 10"

    class ClassWithGlobalInvariant(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    try:
        ClassWithGlobalInvariant(x=1, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Sum must be 10" in e.invariant_errors

    obj = ClassWithGlobalInvariant(x=4, y=6)
    assert obj.x == 4
    assert obj.y == 6

    # Test with factory fields
    class ClassForFactoryTest(PClass):
        x = field()
        y = field()

    obj = ClassForFactoryTest(x=1, y=2)
    new_obj = obj.set(x=3)
    assert new_obj.x == 3
    assert new_obj.y == 2

    # Test that object becomes frozen after creation
    obj = SimpleClass(x=1, y=2)
    try:
        obj.x = 3
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test equality
    obj1 = SimpleClass(x=1, y=2)
    obj2 = SimpleClass(x=1, y=2)
    obj3 = SimpleClass(x=1, y=3)
    assert obj1 == obj2
    assert obj1 != obj3
    assert obj1 != "not a PClass"

    # Test hash
    obj1 = SimpleClass(x=1, y=2)
    obj2 = SimpleClass(x=1, y=2)
    assert hash(obj1) == hash(obj2)


# LLM-generated content at query #18
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class SimpleClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = SimpleClass(x=1, y=2)
    obj2 = SimpleClass(x=1, y=2)
    assert obj1 == obj2
    assert not (obj1 != obj2)

    # Test inequality with different values
    obj3 = SimpleClass(x=1, y=3)
    assert obj1 != obj3
    assert not (obj1 == obj3)

    # Test equality with missing values (using default initial)
    class ClassWithInitial(PClass):
        x = field(initial=10)
        y = field()

    obj4 = ClassWithInitial(y=2)  # x defaults to 10
    obj5 = ClassWithInitial(x=10, y=2)
    assert obj4 == obj5

    # Test inequality with different classes
    class OtherClass(PClass):
        x = field()
        y = field()

    obj6 = OtherClass(x=1, y=2)
    assert obj1 != obj6
    assert not (obj1 == obj6)

    # Test equality with same object
    assert obj1 == obj1

    # Test comparison with non-PClass object
    assert obj1 != "not a PClass"
    assert obj1 != 123
    assert obj1 != {"x": 1, "y": 2}

    # Test with None values
    class ClassWithNone(PClass):
        x = field()
        y = field()

    obj7 = ClassWithNone(x=None, y=2)
    obj8 = ClassWithNone(x=None, y=2)
    obj9 = ClassWithNone(x=1, y=2)
    assert obj7 == obj8
    assert obj7 != obj9

    # Test with all fields different
    obj10 = SimpleClass(x=10, y=20)
    obj11 = SimpleClass(x=30, y=40)
    assert obj10 != obj11

    # Test with one field missing (mandatory field)
    class ClassMandatory(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)

    obj12 = ClassMandatory(x=1, y=2)
    obj13 = ClassMandatory(x=1, y=2)
    assert obj12 == obj13

    # Test __ne__ method explicitly
    assert obj1.__ne__(obj2) is False
    assert obj1.__ne__(obj3) is True
    assert obj1.__ne__("string") is True


# LLM-generated content at query #19
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field
    import pickle

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic pickling and unpickling
    obj = TestClass(x=10, y="hello")
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    
    assert obj == unpickled
    assert obj.x == unpickled.x
    assert obj.y == unpickled.y
    
    # Test that __reduce__ returns correct tuple structure
    reduce_result = obj.__reduce__()
    assert len(reduce_result) == 2
    assert reduce_result[0] == _restore_pickle
    assert len(reduce_result[1]) == 3
    assert reduce_result[1][0] == TestClass
    
    # Test data in reduce tuple contains all fields
    data = reduce_result[1][1]
    assert data == {'x': 10, 'y': 'hello'}
    
    # Test with missing optional field
    class TestClass2(PClass):
        x = field(mandatory=True)
        y = field(mandatory=False, initial=5)
    
    obj2 = TestClass2(x=20)
    reduce_result2 = obj2.__reduce__()
    data2 = reduce_result2[1][1]
    assert data2 == {'x': 20, 'y': 5}
    
    # Test that unpickled object maintains immutability
    unpickled2 = pickle.loads(pickle.dumps(obj2))
    with pytest.raises(AttributeError):
        unpickled2.x = 30
    
    # Test with nested pickling
    class Container(PClass):
        item = field()
    
    nested = Container(item=TestClass(x=1, y=2))
    nested_unpickled = pickle.loads(pickle.dumps(nested))
    assert nested == nested_unpickled
    assert isinstance(nested_unpickled.item, TestClass)
    assert nested_unpickled.item.x == 1
    assert nested_unpickled.item.y == 2


# LLM-generated content at query #20
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field
    from pyrsistent._checked_types import InvariantException

    class SimpleClass(PClass):
        x = field()
        y = field(mandatory=True)
        z = field(initial=10)

    # Test basic instantiation with all required fields
    obj = SimpleClass(x=5, y=20)
    assert obj.x == 5
    assert obj.y == 20
    assert obj.z == 10  # Default initial value

    # Test with all fields provided
    obj = SimpleClass(x=1, y=2, z=3)
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 3

    # Test missing mandatory field
    try:
        SimpleClass(x=5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "SimpleClass.y" in str(e)

    # Test extra field not in specification
    try:
        SimpleClass(x=1, y=2, w=99)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'w' are not among the specified fields" in str(e)

    # Test with multiple extra fields
    try:
        SimpleClass(x=1, y=2, extra1=3, extra2=4)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'extra1', 'extra2'" in str(e) or "'extra2', 'extra1'" in str(e)

    # Test field type checking
    class TypedClass(PClass):
        value = field(type=int)

    # Valid type
    obj = TypedClass(value=42)
    assert obj.value == 42

    # Invalid type
    try:
        TypedClass(value="not an int")
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test field invariant
    class PositiveClass(PClass):
        number = field(invariant=lambda x: (x > 0, "Must be positive"))

    # Valid invariant
    obj = PositiveClass(number=5)
    assert obj.number == 5

    # Invalid invariant
    try:
        PositiveClass(number=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Must be positive" in str(e)

    # Test multiple invariants failing
    class MultiInvariantClass(PClass):
        a = field(mandatory=True, invariant=lambda x: (x > 0, "a positive"))
        b = field(mandatory=True, invariant=lambda x: (x < 10, "b less than 10"))

    try:
        MultiInvariantClass(a=-1, b=20)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "a positive" in str(e)
        assert "b less than 10" in str(e)

    # Test callable initial
    counter = 0

    def make_counter():
        nonlocal counter
        counter += 1
        return counter

    class CallableInitialClass(PClass):
        id = field(initial=make_counter)
        data = field(mandatory=True)

    obj1 = CallableInitialClass(data="first")
    obj2 = CallableInitialClass(data="second")
    assert obj1.id == 1
    assert obj2.id == 2

    # Test factory fields parameter
    class FactoryClass(PClass):
        x = field(type=int, factory=int)

    # This should work with factory processing
    obj = FactoryClass(x="123", _factory_fields={"x"})
    assert obj.x == 123

    # Test ignore_extra parameter
    obj = SimpleClass.create({"x": 1, "y": 2, "extra": 99}, ignore_extra=True)
    assert obj.x == 1
    assert obj.y == 2
    assert not hasattr(obj, 'extra')

    # Test that object becomes frozen after creation
    obj = SimpleClass(x=1, y=2)
    try:
        obj.x = 99
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)

    # Test global invariants
    class GlobalInvariantClass(PClass):
        x = field()
        y = field()

        @staticmethod
        def __invariant__(obj):
            return obj.x + obj.y > 0, "Sum must be positive"

    # Valid global invariant
    obj = GlobalInvariantClass(x=5, y=2)
    assert obj.x == 5
    assert obj.y == 2

    # Invalid global invariant
    try:
        GlobalInvariantClass(x=-5, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Sum must be positive" in str(e)


# LLM-generated content at query #21
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field
    from pyrsistent._checked_types import InvariantException

    # Test basic instantiation with field
    class SimpleClass(PClass):
        x = field()
        y = field()

    obj = SimpleClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2

    # Test with mandatory field missing
    class MandatoryClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        MandatoryClass(y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "MandatoryClass.x" in str(e)

    # Test with field initial value
    class InitialClass(PClass):
        x = field(initial=10)
        y = field()

    obj = InitialClass(y=20)
    assert obj.x == 10
    assert obj.y == 20

    # Test with callable initial
    class CallableInitialClass(PClass):
        x = field(initial=lambda: 100)
        y = field()

    obj = CallableInitialClass(y=200)
    assert obj.x == 100
    assert obj.y == 200

    # Test with extra fields (should raise AttributeError)
    class ExtraFieldClass(PClass):
        x = field()

    try:
        ExtraFieldClass(x=1, z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'z' are not among the specified fields" in str(e)

    # Test with multiple extra fields
    try:
        ExtraFieldClass(x=1, y=2, z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'y', 'z'" in str(e) or "'z', 'y'" in str(e)

    # Test field invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        x = field(invariant=positive_invariant)

    # Should work with positive value
    obj = InvariantClass(x=5)
    assert obj.x == 5

    # Should fail with negative value
    try:
        InvariantClass(x=-5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

    # Test with factory fields parameter
    class FactoryClass(PClass):
        x = field(type=int)
        y = field(type=int)

    # This should work normally
    obj = FactoryClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2

    # Test with ignore_extra parameter
    class IgnoreExtraClass(PClass):
        x = field()

    # Should ignore extra field when ignore_extra=True
    obj = IgnoreExtraClass(x=1, y=2, ignore_extra=True)
    assert obj.x == 1
    assert not hasattr(obj, 'y')

    # Test multiple invariants failing
    def even_invariant(value):
        return value % 2 == 0, "Value must be even"

    class MultiInvariantClass(PClass):
        x = field(invariant=positive_invariant)
        y = field(invariant=even_invariant)

    try:
        MultiInvariantClass(x=-1, y=3)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

    # Test that object is frozen after creation
    obj = SimpleClass(x=1, y=2)
    try:
        obj.x = 3
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)

    # Test with no fields
    class EmptyClass(PClass):
        pass

    obj = EmptyClass()
    assert isinstance(obj, EmptyClass)

    # Test that _pclass_frozen is set to True
    assert obj._pclass_frozen is True


# LLM-generated content at query #22
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test basic serialization without custom serializer
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=10, y="test")
    result = obj.serialize()
    assert result == {"x": 10, "y": "test"}
    
    # Test serialization with custom serializer function
    def custom_serializer(value):
        return f"serialized_{value}"
    
    class WithSerializer(PClass):
        data = field(type=str, serializer=custom_serializer)
        number = field(type=int)
    
    obj = WithSerializer(data="hello", number=42)
    result = obj.serialize()
    assert result == {"data": "serialized_hello", "number": 42}
    
    # Test serialization with format parameter
    def format_serializer(format_param, value):
        if format_param == "json":
            return {"value": value}
        return value
    
    class FormatSerializer(PClass):
        item = field(serializer=format_serializer)
    
    obj = FormatSerializer(item="test")
    result = obj.serialize(format="json")
    assert result == {"item": {"value": "test"}}
    
    # Test serialization with None serializer
    class NoSerializer(PClass):
        a = field()
        b = field(serializer=None)
    
    obj = NoSerializer(a=1, b=2)
    result = obj.serialize()
    assert result == {"a": 1, "b": 2}
    
    # Test serialization with missing optional field
    class OptionalField(PClass):
        required = field(mandatory=True)
        optional = field(mandatory=False, initial=100)
    
    obj = OptionalField(required=50)
    result = obj.serialize()
    assert result == {"required": 50, "optional": 100}
    
    # Test serialization with nested serialization
    def nested_serializer(value):
        return value.serialize() if hasattr(value, 'serialize') else value
    
    class NestedClass(PClass):
        name = field()
    
    class ContainerClass(PClass):
        nested = field(serializer=nested_serializer)
        value = field()
    
    nested = NestedClass(name="inner")
    container = ContainerClass(nested=nested, value=5)
    result = container.serialize()
    assert result == {"nested": {"name": "inner"}, "value": 5}
    
    # Test serialization with multiple custom serializers
    class MultiSerializer(PClass):
        str_field = field(serializer=lambda x: x.upper())
        int_field = field(serializer=lambda x: x * 2)
        list_field = field(serializer=lambda x: len(x))
    
    obj = MultiSerializer(str_field="hello", int_field=5, list_field=[1, 2, 3])
    result = obj.serialize()
    assert result == {"str_field": "HELLO", "int_field": 10, "list_field": 3}
    
    # Test that serialize returns a new dict, not a reference to internal data
    obj = SimpleClass(x=1, y=2)
    result = obj.serialize()
    result["x"] = 999
    assert obj.x == 1  # Original should not be modified


# LLM-generated content at query #23
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation with fields
    class TestClass(PClass):
        x = field()
        y = field()
    
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    assert TestClass.__slots__ == ('_pclass_frozen', 'x', 'y', '__weakref__')
    
    # Test that __slots__ includes all fields
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2
    
    # Test inheritance with PClass
    class ParentClass(PClass):
        a = field()
    
    class ChildClass(ParentClass):
        b = field()
    
    assert 'a' in ChildClass._pclass_fields
    assert 'b' in ChildClass._pclass_fields
    assert ChildClass.__slots__ == ('_pclass_frozen', 'a', 'b', '__weakref__')
    
    # Test that only top-level PClass gets __weakref__
    class RegularClass:
        pass
    
    class MixedClass(PClass, RegularClass):
        z = field()
    
    assert MixedClass.__slots__ == ('_pclass_frozen', 'z', '__weakref__')
    
    # Test invariants storage
    def custom_invariant(value):
        return value > 0, "Value must be positive"
    
    class WithInvariant(PClass):
        value = field(invariant=custom_invariant)
    
    assert hasattr(WithInvariant, '_pclass_invariants')
    assert len(WithInvariant._pclass_invariants) == 1
    
    # Test that non-PClass inheritance doesn't add __weakref__ multiple times
    class AnotherBase:
        pass
    
    class MultiInherit(AnotherBase, PClass):
        field1 = field()
    
    # Should still have __weakref__ since it inherits from PClass
    assert '__weakref__' in MultiInherit.__slots__
    
    # Test that fields dictionary is properly set
    assert isinstance(TestClass._pclass_fields, dict)
    assert all(isinstance(field, type(TestClass._pclass_fields['x'])) 
               for field in TestClass._pclass_fields.values())


# LLM-generated content at query #24
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test 1: Basic serialization without custom serializer
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=10, y="test")
    result = obj.serialize()
    assert result == {"x": 10, "y": "test"}
    
    # Test 2: Serialization with custom serializer function
    def custom_serializer(value):
        return f"serialized_{value}"
    
    class WithSerializer(PClass):
        data = field(serializer=custom_serializer)
        other = field()
    
    obj = WithSerializer(data="value", other=42)
    result = obj.serialize()
    assert result == {"data": "serialized_value", "other": 42}
    
    # Test 3: Serialization with format parameter
    def format_serializer(format, value):
        if format == "json":
            return str(value)
        return value
    
    class FormatSerializer(PClass):
        num = field(serializer=format_serializer)
        text = field()
    
    obj = FormatSerializer(num=100, text="hello")
    result = obj.serialize(format="json")
    assert result == {"num": "100", "text": "hello"}
    
    # Test 4: Serialization with missing optional field
    class OptionalFields(PClass):
        required = field(mandatory=True)
        optional = field()
    
    obj = OptionalFields(required="req")
    result = obj.serialize()
    assert result == {"required": "req"}
    
    # Test 5: Serialization with nested PClass
    class Inner(PClass):
        value = field()
    
    class Outer(PClass):
        inner = field()
        name = field()
    
    inner_obj = Inner(value=5)
    outer_obj = Outer(inner=inner_obj, name="test")
    result = outer_obj.serialize()
    assert result["name"] == "test"
    assert isinstance(result["inner"], dict)
    assert result["inner"]["value"] == 5
    
    # Test 6: Serialization with factory fields
    class FactoryClass(PClass):
        x = field()
        y = field()
    
    obj = FactoryClass(x=1, y=2)
    evolver = obj.evolver()
    evolver.x = 10
    updated = evolver.persistent()
    result = updated.serialize()
    assert result == {"x": 10, "y": 2}
    
    # Test 7: Serialization with None values
    class WithNone(PClass):
        a = field()
        b = field()
    
    obj = WithNone(a=None, b="not_none")
    result = obj.serialize()
    assert result == {"a": None, "b": "not_none"}
    
    # Test 8: Serialization with empty PClass
    class Empty(PClass):
        pass
    
    obj = Empty()
    result = obj.serialize()
    assert result == {}
    
    # Test 9: Serialization with multiple custom serializers
    def double_serializer(value):
        return value * 2
    
    def upper_serializer(value):
        return value.upper()
    
    class MultipleSerializers(PClass):
        number = field(serializer=double_serializer)
        text = field(serializer=upper_serializer)
        plain = field()
    
    obj = MultipleSerializers(number=5, text="hello", plain="world")
    result = obj.serialize()
    assert result == {"number": 10, "text": "HELLO", "plain": "world"}


# LLM-generated content at query #25
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test 1: Basic serialization without custom serializer
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=10, y="test")
    result = obj.serialize()
    assert result == {"x": 10, "y": "test"}
    
    # Test 2: Serialization with custom serializer function
    def custom_serializer(value):
        return f"serialized_{value}"
    
    class WithSerializer(PClass):
        data = field(serializer=custom_serializer)
        other = field()
    
    obj = WithSerializer(data="value", other=42)
    result = obj.serialize()
    assert result == {"data": "serialized_value", "other": 42}
    
    # Test 3: Serialization with format parameter
    def format_serializer(format, value):
        if format == "json":
            return {"value": value}
        return value
    
    class FormatClass(PClass):
        item = field(serializer=format_serializer)
    
    obj = FormatClass(item="data")
    result = obj.serialize(format="json")
    assert result == {"item": {"value": "data"}}
    
    # Test 4: Serialization with missing optional field
    class OptionalClass(PClass):
        required = field(mandatory=True)
        optional = field(mandatory=False, initial=100)
    
    obj = OptionalClass(required=50)
    result = obj.serialize()
    assert result == {"required": 50, "optional": 100}
    
    # Test 5: Serialization with nested PClass
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
        name = field()
    
    inner = InnerClass(value=5)
    outer = OuterClass(inner=inner, name="test")
    result = outer.serialize()
    assert result["name"] == "test"
    assert isinstance(result["inner"], dict)
    assert result["inner"] == {"value": 5}
    
    # Test 6: Serialization with multiple custom serializers
    def double_serializer(value):
        return value * 2
    
    def upper_serializer(value):
        return value.upper()
    
    class MultiSerializerClass(PClass):
        number = field(serializer=double_serializer)
        text = field(serializer=upper_serializer)
        plain = field()
    
    obj = MultiSerializerClass(number=21, text="hello", plain="world")
    result = obj.serialize()
    assert result == {"number": 42, "text": "HELLO", "plain": "world"}
    
    # Test 7: Serialization with None values
    class NullableClass(PClass):
        data = field()
        none_field = field()
    
    obj = NullableClass(data=None, none_field="exists")
    result = obj.serialize()
    assert result == {"data": None, "none_field": "exists"}
    
    # Test 8: Serialization with empty PClass
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    result = obj.serialize()
    assert result == {}
    
    # Test 9: Serialization with factory fields
    def create_list(value):
        return [value]
    
    class FactoryClass(PClass):
        items = field(factory=create_list)
    
    obj = FactoryClass(items=1)
    result = obj.serialize()
    assert result == {"items": [1]}
    
    # Test 10: Serialization with format parameter but no serializer
    class NoSerializerClass(PClass):
        field1 = field()
        field2 = field()
    
    obj = NoSerializerClass(field1=1, field2=2)
    result = obj.serialize(format="any_format")
    assert result == {"field1": 1, "field2": 2}


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PClassEvolver_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    evolver = original.evolver()

    evolver.set('x', 10)
    assert evolver._pclass_evolver_data['x'] == 10
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'x' in evolver._factory_fields

    evolver.set('y', 20)
    assert evolver._pclass_evolver_data['y'] == 20
    assert 'y' in evolver._factory_fields

    persistent = evolver.persistent()
    assert persistent.x == 10
    assert persistent.y == 20

    evolver2 = original.evolver()
    evolver2.set('x', 1)
    assert evolver2._pclass_evolver_data_is_dirty is False
    assert evolver2.persistent() is original

    evolver3 = original.evolver()
    evolver3['x'] = 30
    assert evolver3._pclass_evolver_data['x'] == 30
    assert evolver3._pclass_evolver_data_is_dirty is True


# LLM-generated content at query #2
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    # Test basic set with keyword arguments
    instance = TestClass(x=1, y=2, z=3)
    new_instance = instance.set(x=10)
    assert instance.x == 1
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert new_instance.z == 3

    # Test set with positional arguments
    instance2 = TestClass(x=5, y=6, z=7)
    new_instance2 = instance2.set('y', 20)
    assert instance2.y == 6
    assert new_instance2.y == 20
    assert new_instance2.x == 5
    assert new_instance2.z == 7

    # Test set multiple fields with kwargs
    instance3 = TestClass(x=1, y=2, z=3)
    new_instance3 = instance3.set(x=100, y=200)
    assert new_instance3.x == 100
    assert new_instance3.y == 200
    assert new_instance3.z == 3

    # Test set with both args and kwargs (args should take precedence)
    instance4 = TestClass(x=1, y=2, z=3)
    new_instance4 = instance4.set('x', 50, y=60)
    assert new_instance4.x == 50  # From positional arg
    assert new_instance4.y == 60  # From keyword arg
    assert new_instance4.z == 3

    # Test set on empty instance (with default fields)
    class TestClassWithDefaults(PClass):
        x = field(initial=0)
        y = field(initial=lambda: 10)
        z = field()

    instance5 = TestClassWithDefaults(z=30)
    new_instance5 = instance5.set(x=5)
    assert instance5.x == 0
    assert instance5.y == 10
    assert new_instance5.x == 5
    assert new_instance5.y == 10
    assert new_instance5.z == 30

    # Test that original instance remains unchanged after multiple sets
    instance6 = TestClass(x=1, y=2, z=3)
    intermediate = instance6.set(x=10)
    final = intermediate.set(y=20)
    assert instance6.x == 1 and instance6.y == 2 and instance6.z == 3
    assert intermediate.x == 10 and intermediate.y == 2 and intermediate.z == 3
    assert final.x == 10 and final.y == 20 and final.z == 3

    # Test set with field that has factory
    class TestClassWithFactory(PClass):
        x = field(type=int, factory=lambda v: v * 2)

    instance7 = TestClassWithFactory(x=5)
    new_instance7 = instance7.set(x=10)
    assert instance7.x == 10  # Factory applied: 5 * 2 = 10
    assert new_instance7.x == 20  # Factory applied: 10 * 2 = 20

    # Test that set returns new instance of same class
    instance8 = TestClass(x=1, y=2, z=3)
    new_instance8 = instance8.set(x=100)
    assert isinstance(new_instance8, TestClass)
    assert type(new_instance8) is type(instance8)

    # Test set with invariant checking
    class TestClassWithInvariant(PClass):
        x = field(type=int)
        y = field(type=int)

        @staticmethod
        def __invariant__(x, y):
            return x + y > 0, "Sum must be positive"

    instance9 = TestClassWithInvariant(x=5, y=3)
    # This should work
    new_instance9 = instance9.set(x=10)
    assert new_instance9.x == 10 and new_instance9.y == 3

    # Test set preserves other attributes when not specified
    class ComplexClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()

    instance10 = ComplexClass(a=1, b=2, c=3, d=4)
    new_instance10 = instance10.set(b=20, d=40)
    assert new_instance10.a == 1
    assert new_instance10.b == 20
    assert new_instance10.c == 3
    assert new_instance10.d == 40


# LLM-generated content at query #3
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class SimpleClass(PClass):
        x = field()
        y = field()

    class EmptyClass(PClass):
        pass

    class ClassWithTypes(PClass):
        name = field(type=str)
        value = field(type=int)

    # Test basic representation
    obj1 = SimpleClass(x=10, y="test")
    assert repr(obj1) == "SimpleClass(x=10, y='test')"

    # Test with different value types
    obj2 = SimpleClass(x=[1, 2, 3], y={"key": "value"})
    assert repr(obj2) == "SimpleClass(x=[1, 2, 3], y={'key': 'value'})"

    # Test empty class
    obj3 = EmptyClass()
    assert repr(obj3) == "EmptyClass()"

    # Test with typed fields
    obj4 = ClassWithTypes(name="Alice", value=42)
    assert repr(obj4) == "ClassWithTypes(name='Alice', value=42)"

    # Test with None values
    class ClassWithNone(PClass):
        a = field()
        b = field()

    obj5 = ClassWithNone(a=None, b="not_none")
    assert repr(obj5) == "ClassWithNone(a=None, b='not_none')"

    # Test with boolean values
    obj6 = SimpleClass(x=True, y=False)
    assert repr(obj6) == "SimpleClass(x=True, y=False)"

    # Test with numeric values
    obj7 = SimpleClass(x=3.14, y=-5)
    assert repr(obj7) == "SimpleClass(x=3.14, y=-5)"

    # Test that representation is consistent
    obj8 = SimpleClass(x="hello", y="world")
    repr1 = repr(obj8)
    repr2 = repr(obj8)
    assert repr1 == repr2
    assert repr1 == "SimpleClass(x='hello', y='world')"


# LLM-generated content at query #4
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field
    from pyrsistent._checked_types import InvariantException

    # Test basic instantiation with required fields
    class SimpleClass(PClass):
        x = field()
        y = field()

    obj = SimpleClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2

    # Test with optional fields with initial values
    class WithInitial(PClass):
        x = field()
        y = field(initial=10)
        z = field(mandatory=False)

    obj1 = WithInitial(x=5)
    assert obj1.x == 5
    assert obj1.y == 10
    assert not hasattr(obj1, 'z')

    obj2 = WithInitial(x=5, y=20, z=30)
    assert obj2.x == 5
    assert obj2.y == 20
    assert obj2.z == 30

    # Test with callable initial
    counter = 0
    def make_counter():
        nonlocal counter
        counter += 1
        return counter

    class CallableInitial(PClass):
        x = field(initial=make_counter)

    obj3 = CallableInitial()
    assert obj3.x == 1
    obj4 = CallableInitial()
    assert obj4.x == 2

    # Test missing mandatory fields
    class MandatoryFields(PClass):
        a = field(mandatory=True)
        b = field(mandatory=True)

    try:
        MandatoryFields(a=1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MandatoryFields.b' in e.missing_fields

    # Test extra fields not allowed
    class StrictClass(PClass):
        x = field()

    try:
        StrictClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'y' are not among the specified fields" in str(e)

    # Test field type checking
    class TypedClass(PClass):
        x = field(type=int)

    try:
        TypedClass(x="not an int")
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    obj5 = TypedClass(x=42)
    assert obj5.x == 42

    # Test field invariants
    class PositiveClass(PClass):
        x = field(invariant=lambda x: (x > 0, 'x must be positive'))

    try:
        PositiveClass(x=-5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'x must be positive' in e.invariant_errors

    obj6 = PositiveClass(x=5)
    assert obj6.x == 5

    # Test global invariants
    class GlobalInvariantClass(PClass):
        x = field()
        y = field()
        
        @staticmethod
        def __invariant__(obj):
            return obj.x + obj.y > 0, 'Sum must be positive'

    try:
        GlobalInvariantClass(x=-5, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'Sum must be positive' in e.invariant_errors

    obj7 = GlobalInvariantClass(x=2, y=3)
    assert obj7.x == 2
    assert obj7.y == 3

    # Test with factory fields parameter
    class FactoryClass(PClass):
        x = field()
        y = field()

    obj8 = FactoryClass(x=1, y=2, _factory_fields={'x'})
    assert obj8.x == 1
    assert obj8.y == 2

    # Test ignore_extra parameter
    class IgnoreExtraClass(PClass):
        x = field()

    obj9 = IgnoreExtraClass(x=1, y=2, ignore_extra=True)
    assert obj9.x == 1
    assert not hasattr(obj9, 'y')

    # Test object becomes frozen after creation
    class FrozenTest(PClass):
        x = field()

    obj10 = FrozenTest(x=1)
    try:
        obj10.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test equality and hash
    class EqClass(PClass):
        x = field()
        y = field()

    obj11 = EqClass(x=1, y=2)
    obj12 = EqClass(x=1, y=2)
    obj13 = EqClass(x=3, y=4)
    
    assert obj11 == obj12
    assert obj11 != obj13
    assert hash(obj11) == hash(obj12)
    assert hash(obj11) != hash(obj13)

    # Test with nested PClasses
    class Inner(PClass):
        a = field()

    class Outer(PClass):
        inner = field(type=Inner)
        value = field()

    inner_obj = Inner(a=10)
    outer_obj = Outer(inner=inner_obj, value=20)
    assert outer_obj.inner.a == 10
    assert outer_obj.value == 20


# LLM-generated content at query #5
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    # Test basic set with keyword arguments
    obj = TestClass(x=1, y=2, z=3)
    obj2 = obj.set(x=10)
    assert obj2.x == 10
    assert obj2.y == 2
    assert obj2.z == 3
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    obj3 = obj.set('y', 20)
    assert obj3.y == 20
    assert obj3.x == 1
    assert obj3.z == 3

    # Test set multiple fields
    obj4 = obj.set(x=100, z=300)
    assert obj4.x == 100
    assert obj4.y == 2
    assert obj4.z == 300

    # Test that set returns new instance of same class
    assert isinstance(obj2, TestClass)
    assert obj2 is not obj

    # Test set with non-existent field raises AttributeError
    obj = TestClass(x=1, y=2, z=3)
    try:
        obj.set('nonexistent', 5)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "nonexistent" in str(e)

    # Test set preserves field invariants
    class RestrictedClass(PClass):
        value = field(type=int, invariant=lambda x: (x > 0, 'value must be positive'))

    obj = RestrictedClass(value=5)
    obj2 = obj.set(value=10)
    assert obj2.value == 10

    # Test set with invalid value raises InvariantException
    obj = RestrictedClass(value=5)
    try:
        obj.set(value=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test set with factory fields
    class FactoryClass(PClass):
        items = field(type=list, factory=lambda x: list(x))

    obj = FactoryClass(items=[1, 2, 3])
    obj2 = obj.set(items=[4, 5, 6])
    assert obj2.items == [4, 5, 6]
    assert obj.items == [1, 2, 3]

    # Test set with missing optional field
    class OptionalClass(PClass):
        required = field(mandatory=True)
        optional = field(mandatory=False, initial=0)

    obj = OptionalClass(required=1)
    assert obj.optional == 0
    obj2 = obj.set(optional=5)
    assert obj2.optional == 5
    assert obj2.required == 1

    # Test set maintains hash equality
    obj = TestClass(x=1, y=2, z=3)
    obj2 = obj.set(x=1)
    assert hash(obj) == hash(obj2)
    assert obj == obj2

    obj3 = obj.set(x=4)
    assert hash(obj) != hash(obj3)
    assert obj != obj3

    # Test set with evolver integration
    evolver = obj.evolver()
    evolver['x'] = 99
    obj4 = evolver.persistent()
    obj5 = obj.set(x=99)
    assert obj4 == obj5
    assert obj4.x == 99


# LLM-generated content at query #6
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation with fields
    class TestClass(PClass):
        x = field()
        y = field()
    
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    assert TestClass.__slots__ == ('_pclass_frozen', 'x', 'y', '__weakref__')
    
    # Test that __slots__ includes all field names
    class AnotherClass(PClass):
        a = field()
        b = field()
        c = field()
    
    assert AnotherClass.__slots__ == ('_pclass_frozen', 'a', 'b', 'c', '__weakref__')
    
    # Test that invariants are stored
    class WithInvariant(PClass):
        value = field()
        
        def __invariant__(self):
            return self.value > 0, "Value must be positive"
    
    assert hasattr(WithInvariant, '_pclass_invariants')
    assert len(WithInvariant._pclass_invariants) == 1
    
    # Test inheritance - child class should have parent's fields
    class Parent(PClass):
        parent_field = field()
    
    class Child(Parent):
        child_field = field()
    
    assert 'parent_field' in Child._pclass_fields
    assert 'child_field' in Child._pclass_fields
    assert Child.__slots__ == ('_pclass_frozen', 'parent_field', 'child_field', '__weakref__')
    
    # Test that only top-level PClass gets __weakref__
    class Base(PClass):
        pass
    
    class Derived(Base):
        pass
    
    # Both should have __weakref__ since they inherit from PClass
    assert '__weakref__' in Base.__slots__
    assert '__weakref__' in Derived.__slots__
    
    # Test that non-PClass inheritance works correctly
    class RegularBase:
        pass
    
    class MixedClass(RegularBase, PClass):
        field1 = field()
    
    # Should still have PClass functionality
    assert hasattr(MixedClass, '_pclass_fields')
    assert 'field1' in MixedClass._pclass_fields
    # __weakref__ should be present since it's a PClass
    assert '__weakref__' in MixedClass.__slots__


# LLM-generated content at query #7
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    # Test basic set with keyword arguments
    obj = TestClass(x=1, y=2, z=3)
    obj2 = obj.set(x=10)
    assert obj2.x == 10
    assert obj2.y == 2
    assert obj2.z == 3
    assert obj.x == 1  # Original unchanged
    
    # Test set with positional arguments
    obj3 = obj.set('y', 20)
    assert obj3.y == 20
    assert obj3.x == 1
    assert obj3.z == 3
    
    # Test set with both positional and keyword (should use keyword)
    obj4 = obj.set('x', 100, y=200)
    assert obj4.x == 100
    assert obj4.y == 200
    assert obj4.z == 3
    
    # Test set multiple fields
    obj5 = obj.set(x=30, z=50)
    assert obj5.x == 30
    assert obj5.y == 2
    assert obj5.z == 50
    
    # Test that set returns new instance
    obj6 = TestClass(x=1, y=2, z=3)
    obj7 = obj6.set(x=99)
    assert obj6 is not obj7
    assert obj6.x == 1
    assert obj7.x == 99
    
    # Test with field that has factory
    class FactoryClass(PClass):
        x = field(type=int, factory=lambda v: v * 2)
        y = field()
    
    factory_obj = FactoryClass(x=5, y=10)
    factory_obj2 = factory_obj.set(x=3)
    assert factory_obj2.x == 6  # Factory should be applied
    assert factory_obj2.y == 10
    
    # Test with mandatory field
    class MandatoryClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    mandatory_obj = MandatoryClass(x=1, y=2)
    mandatory_obj2 = mandatory_obj.set(x=10)
    assert mandatory_obj2.x == 10
    assert mandatory_obj2.y == 2
    
    # Test that original instance remains frozen
    frozen_obj = TestClass(x=1, y=2, z=3)
    try:
        frozen_obj.x = 10
        assert False, "Should not be able to set attribute on frozen instance"
    except AttributeError:
        pass
    
    # Test set preserves other attributes
    class ComplexClass(PClass):
        a = field()
        b = field()
        c = field()
    
    complex_obj = ComplexClass(a=1, b=2, c=3)
    complex_obj2 = complex_obj.set(a=100)
    complex_obj3 = complex_obj2.set(b=200)
    assert complex_obj3.a == 100
    assert complex_obj3.b == 200
    assert complex_obj3.c == 3
    
    # Test set with non-existent field should raise AttributeError when creating new instance
    obj = TestClass(x=1, y=2, z=3)
    try:
        obj.set(nonexistent=10)
        assert False, "Should raise AttributeError for non-existent field"
    except AttributeError:
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field

    class SimpleClass(PClass):
        x = field()
        y = field()

    # Test that equal instances have same hash
    instance1 = SimpleClass(x=1, y=2)
    instance2 = SimpleClass(x=1, y=2)
    assert instance1 == instance2
    assert hash(instance1) == hash(instance2)

    # Test that different instances have different hashes
    instance3 = SimpleClass(x=2, y=2)
    assert instance1 != instance3
    assert hash(instance1) != hash(instance3)

    # Test hash with missing optional fields
    class OptionalClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=False, initial=0)

    instance4 = OptionalClass(x=1)
    instance5 = OptionalClass(x=1, y=0)
    assert instance4 == instance5
    assert hash(instance4) == hash(instance5)

    # Test hash consistency across multiple calls
    instance6 = SimpleClass(x=10, y=20)
    assert hash(instance6) == hash(instance6)

    # Test hash with different field orders (should be same if values are same)
    class MultiFieldClass(PClass):
        a = field()
        b = field()
        c = field()

    instance7 = MultiFieldClass(a=1, b=2, c=3)
    instance8 = MultiFieldClass(c=3, a=1, b=2)
    assert instance7 == instance8
    assert hash(instance7) == hash(instance8)

    # Test hash with None values
    class WithNoneClass(PClass):
        x = field()
        y = field()

    instance9 = WithNoneClass(x=None, y=5)
    instance10 = WithNoneClass(x=None, y=5)
    assert instance9 == instance10
    assert hash(instance9) == hash(instance10)

    # Test that hash works correctly with evolver modifications
    instance11 = SimpleClass(x=1, y=2)
    evolver = instance11.evolver()
    evolver.set('x', 3)
    instance12 = evolver.persistent()
    assert instance11 != instance12
    assert hash(instance11) != hash(instance12)

    # Test hash with different types in same field
    instance13 = SimpleClass(x=1, y=2)
    instance14 = SimpleClass(x=1.0, y=2.0)
    assert instance13 != instance14
    assert hash(instance13) != hash(instance14)

    # Test that hash is based on actual values, not object identity
    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    class ContainerClass(PClass):
        items = field()

    instance15 = ContainerClass(items=list1)
    instance16 = ContainerClass(items=list2)
    assert instance15 == instance16
    assert hash(instance15) == hash(instance16)


# LLM-generated content at query #9
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    # Test basic set with keyword arguments
    instance = TestClass(x=1, y=2, z=3)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert new_instance.z == 3
    assert instance.x == 1  # Original unchanged

    # Test set with positional arguments
    instance = TestClass(x=1, y=2, z=3)
    new_instance = instance.set('y', 20)
    assert new_instance.x == 1
    assert new_instance.y == 20
    assert new_instance.z == 3

    # Test set with both positional and keyword (positional should be ignored)
    instance = TestClass(x=1, y=2, z=3)
    new_instance = instance.set('x', 100, z=30)
    assert new_instance.x == 100  # From positional
    assert new_instance.y == 2
    assert new_instance.z == 30  # From keyword

    # Test set multiple fields
    instance = TestClass(x=1, y=2, z=3)
    new_instance = instance.set(x=10, y=20, z=30)
    assert new_instance.x == 10
    assert new_instance.y == 20
    assert new_instance.z == 30

    # Test that set returns new instance of same class
    instance = TestClass(x=1, y=2, z=3)
    new_instance = instance.set(x=10)
    assert isinstance(new_instance, TestClass)
    assert new_instance is not instance

    # Test with field that has factory
    class FactoryClass(PClass):
        x = field(type=int, factory=lambda v: v * 2)
        y = field()

    instance = FactoryClass(x=5, y=10)
    new_instance = instance.set(x=3)
    assert new_instance.x == 6  # Factory applied
    assert new_instance.y == 10

    # Test with mandatory field
    class MandatoryClass(PClass):
        x = field(mandatory=True)
        y = field()

    instance = MandatoryClass(x=1, y=2)
    new_instance = instance.set(y=20)
    assert new_instance.x == 1
    assert new_instance.y == 20

    # Test that original remains frozen after set
    instance = TestClass(x=1, y=2, z=3)
    new_instance = instance.set(x=10)
    
    # Original should still be frozen
    try:
        instance.x = 100
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # New instance should be frozen
    try:
        new_instance.x = 100
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test equality after set
    instance1 = TestClass(x=1, y=2, z=3)
    instance2 = instance1.set(x=10)
    instance3 = TestClass(x=10, y=2, z=3)
    assert instance2 == instance3
    assert instance1 != instance2

    # Test hash consistency after set
    instance1 = TestClass(x=1, y=2, z=3)
    instance2 = instance1.set(x=10)
    instance3 = TestClass(x=10, y=2, z=3)
    assert hash(instance2) == hash(instance3)
    assert hash(instance1) != hash(instance2)

    # Test set with non-existent field should raise AttributeError when creating new instance
    instance = TestClass(x=1, y=2, z=3)
    try:
        instance.set(nonexistent=100)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "nonexistent" in str(e)

    # Test that set preserves other attributes from original
    class ComplexClass(PClass):
        a = field()
        b = field(initial=100)
        c = field(mandatory=True)

    instance = ComplexClass(a=1, c=3)  # b uses initial value
    assert instance.b == 100
    new_instance = instance.set(a=10)
    assert new_instance.a == 10
    assert new_instance.b == 100  # Preserved from original
    assert new_instance.c == 3


# LLM-generated content at query #10
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    # Test basic set with keyword arguments
    obj = TestClass(x=1, y=2, z=3)
    obj2 = obj.set(x=10)
    assert obj2.x == 10
    assert obj2.y == 2
    assert obj2.z == 3
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    obj3 = obj.set('y', 20)
    assert obj3.y == 20
    assert obj3.x == 1
    assert obj3.z == 3

    # Test set with both positional and keyword (positional should be ignored)
    obj4 = obj.set('x', 100, y=200)
    assert obj4.x == 100  # From positional
    assert obj4.y == 200  # From keyword
    assert obj4.z == 3

    # Test set with multiple fields
    obj5 = obj.set(x=30, z=50)
    assert obj5.x == 30
    assert obj5.y == 2
    assert obj5.z == 50

    # Test that set returns new instance of same class
    assert isinstance(obj2, TestClass)
    assert obj is not obj2

    # Test with field that has factory
    class FactoryClass(PClass):
        x = field(type=int, factory=lambda v: v * 2)

    factory_obj = FactoryClass(x=5)
    assert factory_obj.x == 10
    factory_obj2 = factory_obj.set(x=3)
    assert factory_obj2.x == 6

    # Test with mandatory field
    class MandatoryClass(PClass):
        x = field(mandatory=True)
        y = field()

    mandatory_obj = MandatoryClass(x=1, y=2)
    mandatory_obj2 = mandatory_obj.set(y=3)
    assert mandatory_obj2.x == 1
    assert mandatory_obj2.y == 3

    # Test that trying to set non-existent field raises AttributeError
    obj = TestClass(x=1, y=2, z=3)
    try:
        obj.set(nonexistent=10)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "nonexistent" in str(e)

    # Test with invariant
    class InvariantClass(PClass):
        x = field(invariant=lambda x: (x > 0, 'x must be positive'))

    inv_obj = InvariantClass(x=1)
    inv_obj2 = inv_obj.set(x=5)
    assert inv_obj2.x == 5

    # Test that invariant violation raises InvariantException
    try:
        inv_obj.set(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with initial value
    class InitialClass(PClass):
        x = field(initial=10)
        y = field()

    initial_obj = InitialClass(y=5)
    assert initial_obj.x == 10
    initial_obj2 = initial_obj.set(x=20)
    assert initial_obj2.x == 20
    assert initial_obj2.y == 5

    # Test equality after set
    obj1 = TestClass(x=1, y=2, z=3)
    obj2 = obj1.set(x=10)
    obj3 = TestClass(x=10, y=2, z=3)
    assert obj2 == obj3
    assert obj1 != obj2

    # Test hash consistency
    obj1 = TestClass(x=1, y=2, z=3)
    obj2 = obj1.set(x=10)
    obj3 = TestClass(x=10, y=2, z=3)
    assert hash(obj2) == hash(obj3)
    assert hash(obj1) != hash(obj2)

    # Test with _factory_fields parameter (internal use)
    obj = TestClass(x=1, y=2, z=3)
    # This simulates what happens internally when using evolver
    obj2 = obj.set(x=100, y=200)
    assert obj2.x == 100
    assert obj2.y == 200
    assert obj2.z == 3


# LLM-generated content at query #11
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation with fields
    class TestClass(PClass):
        x = field()
        y = field(type=int, mandatory=True)
    
    # Check that _pclass_fields is properly set
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    
    # Check that __slots__ includes all fields plus _pclass_frozen
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'x' in TestClass.__slots__
    assert 'y' in TestClass.__slots__
    
    # Check that invariants storage is set up
    assert hasattr(TestClass, '_pclass_invariants')
    assert isinstance(TestClass._pclass_invariants, tuple)
    
    # Test that CheckedType base class gets __weakref__ in __slots__
    assert '__weakref__' in TestClass.__slots__
    
    # Test PClass inheritance hierarchy
    class BaseClass(PClass):
        a = field()
    
    class DerivedClass(BaseClass):
        b = field()
    
    # Check that derived class has both fields
    assert 'a' in DerivedClass._pclass_fields
    assert 'b' in DerivedClass._pclass_fields
    
    # Check that __slots__ includes all fields from hierarchy
    assert 'a' in DerivedClass.__slots__
    assert 'b' in DerivedClass.__slots__
    assert '_pclass_frozen' in DerivedClass.__slots__
    
    # Test that only direct PClass gets __weakref__
    assert '__weakref__' in BaseClass.__slots__
    assert '__weakref__' not in DerivedClass.__slots__
    
    # Test with custom invariant
    class InvariantClass(PClass):
        value = field(type=int)
        
        @invariant
        def value_positive(self):
            return self.value > 0, "Value must be positive"
    
    # Check that invariant is stored
    assert len(InvariantClass._pclass_invariants) == 1
    
    # Test that metaclass properly handles empty PClass
    class EmptyClass(PClass):
        pass
    
    assert hasattr(EmptyClass, '_pclass_fields')
    assert len(EmptyClass._pclass_fields) == 0
    assert '_pclass_frozen' in EmptyClass.__slots__
    assert '__weakref__' in EmptyClass.__slots__


# LLM-generated content at query #12
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class SimpleClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = SimpleClass(x=1, y=2)
    obj2 = SimpleClass(x=1, y=2)
    assert obj1 == obj2
    assert not (obj1 != obj2)

    # Test inequality with different values
    obj3 = SimpleClass(x=1, y=3)
    assert obj1 != obj3
    assert not (obj1 == obj3)

    # Test equality with same object
    assert obj1 == obj1

    # Test equality with different types
    assert obj1 != "not a PClass"
    assert obj1 != 123
    assert obj1 != None

    # Test equality with subclass (should be False)
    class SubClass(SimpleClass):
        pass

    subclass_obj = SubClass(x=1, y=2)
    assert obj1 != subclass_obj

    # Test equality with missing attributes
    class ClassWithOptional(PClass):
        x = field(mandatory=True)
        y = field(mandatory=False)

    obj4 = ClassWithOptional(x=1)
    obj5 = ClassWithOptional(x=1)
    assert obj4 == obj5

    # Test equality when one has default value
    class ClassWithDefault(PClass):
        x = field()
        y = field(initial=10)

    obj6 = ClassWithDefault(x=1)
    obj7 = ClassWithDefault(x=1, y=10)
    assert obj6 == obj7

    # Test inequality with different default values
    obj8 = ClassWithDefault(x=1, y=20)
    assert obj6 != obj8

    # Test equality with complex nested structures
    class NestedClass(PClass):
        name = field()
        value = field()

    class ContainerClass(PClass):
        nested = field()
        count = field()

    nested1 = NestedClass(name="test", value=42)
    nested2 = NestedClass(name="test", value=42)
    container1 = ContainerClass(nested=nested1, count=1)
    container2 = ContainerClass(nested=nested2, count=1)
    assert container1 == container2

    # Test inequality with different nested values
    nested3 = NestedClass(name="test", value=43)
    container3 = ContainerClass(nested=nested3, count=1)
    assert container1 != container3

    # Test __eq__ returns NotImplemented for non-PClass types
    result = obj1.__eq__("string")
    assert result is NotImplemented

    # Test hash equality follows value equality
    obj9 = SimpleClass(x=1, y=2)
    obj10 = SimpleClass(x=1, y=2)
    assert hash(obj9) == hash(obj10)

    # Test that different values have different hashes (usually)
    obj11 = SimpleClass(x=1, y=3)
    assert hash(obj9) != hash(obj11)


# LLM-generated content at query #13
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    # Test basic set with keyword arguments
    obj = TestClass(x=1, y=2, z=3)
    obj2 = obj.set(x=10)
    assert obj2.x == 10
    assert obj2.y == 2
    assert obj2.z == 3
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    obj3 = obj.set('y', 20)
    assert obj3.x == 1
    assert obj3.y == 20
    assert obj3.z == 3

    # Test set with both positional and keyword (positional should be ignored)
    obj4 = obj.set('x', 100, y=200)
    assert obj4.x == 100  # From positional
    assert obj4.y == 200  # From keyword
    assert obj4.z == 3

    # Test set multiple fields
    obj5 = obj.set(x=100, z=300)
    assert obj5.x == 100
    assert obj5.y == 2
    assert obj5.z == 300

    # Test that set returns new instance of same class
    assert isinstance(obj2, TestClass)
    assert obj is not obj2

    # Test with field that has initial value
    class TestClassWithInitial(PClass):
        x = field(initial=5)
        y = field()

    obj_init = TestClassWithInitial(y=10)
    assert obj_init.x == 5
    obj_init2 = obj_init.set(x=20)
    assert obj_init2.x == 20
    assert obj_init2.y == 10

    # Test with mandatory field
    class TestClassMandatory(PClass):
        x = field(mandatory=True)
        y = field()

    obj_mand = TestClassMandatory(x=1, y=2)
    obj_mand2 = obj_mand.set(x=10)
    assert obj_mand2.x == 10
    assert obj_mand2.y == 2

    # Test that original remains frozen
    try:
        obj.x = 999
        assert False, "Should not be able to set attribute on frozen PClass"
    except AttributeError:
        pass

    # Test with nested set operations
    obj6 = obj.set(x=50).set(y=60).set(z=70)
    assert obj6.x == 50
    assert obj6.y == 60
    assert obj6.z == 70

    # Test that __eq__ works correctly after set
    obj7 = TestClass(x=1, y=2, z=3)
    obj8 = obj7.set(x=10)
    assert obj7 != obj8
    obj9 = obj7.set(x=1)  # Same values
    assert obj7 == obj9

    # Test with field that has factory
    class TestClassFactory(PClass):
        x = field(factory=lambda v: v * 2)
        y = field()

    obj_fact = TestClassFactory(x=5, y=10)
    assert obj_fact.x == 10  # Factory applied
    obj_fact2 = obj_fact.set(x=3)
    assert obj_fact2.x == 6  # Factory should be applied on set
    assert obj_fact2.y == 10


# LLM-generated content at query #14
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field
    import pickle

    class SimpleClass(PClass):
        x = field()
        y = field()

    # Test basic pickling and unpickling
    obj = SimpleClass(x=10, y="test")
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    
    assert obj == unpickled
    assert obj.x == unpickled.x
    assert obj.y == unpickled.y
    assert isinstance(unpickled, SimpleClass)

    # Test pickling with missing optional fields
    class ClassWithOptional(PClass):
        required = field(mandatory=True)
        optional = field()

    obj2 = ClassWithOptional(required=1)
    pickled2 = pickle.dumps(obj2)
    unpickled2 = pickle.loads(pickled2)
    
    assert obj2 == unpickled2
    assert obj2.required == unpickled2.required
    assert not hasattr(unpickled2, 'optional')

    # Test pickling with all fields present
    obj3 = ClassWithOptional(required=1, optional="value")
    pickled3 = pickle.dumps(obj3)
    unpickled3 = pickle.loads(pickled3)
    
    assert obj3 == unpickled3
    assert obj3.optional == unpickled3.optional

    # Test that __reduce__ returns correct tuple structure
    reduce_result = obj.__reduce__()
    assert len(reduce_result) == 2
    assert reduce_result[0] is obj.__class__.__reduce__.__func__.__module__ + '._restore_pickle'
    assert isinstance(reduce_result[1], tuple)
    assert len(reduce_result[1]) == 3
    assert reduce_result[1][0] is SimpleClass
    assert isinstance(reduce_result[1][1], dict)
    assert reduce_result[1][2] == ()

    # Test that data dict contains correct field values
    data_dict = reduce_result[1][1]
    assert data_dict['x'] == 10
    assert data_dict['y'] == "test"
    assert len(data_dict) == 2

    # Test pickling with nested PClass objects
    class InnerClass(PClass):
        value = field()

    class OuterClass(PClass):
        inner = field()

    inner_obj = InnerClass(value=42)
    outer_obj = OuterClass(inner=inner_obj)
    
    pickled_outer = pickle.dumps(outer_obj)
    unpickled_outer = pickle.loads(pickled_outer)
    
    assert outer_obj == unpickled_outer
    assert outer_obj.inner == unpickled_outer.inner
    assert isinstance(unpickled_outer.inner, InnerClass)
    assert unpickled_outer.inner.value == 42

    # Test that pickling preserves field types and invariants
    class TypedClass(PClass):
        number = field(type=int)
        text = field(type=str)

    typed_obj = TypedClass(number=100, text="hello")
    pickled_typed = pickle.dumps(typed_obj)
    unpickled_typed = pickle.loads(pickled_typed)
    
    assert typed_obj == unpickled_typed
    assert isinstance(unpickled_typed.number, int)
    assert isinstance(unpickled_typed.text, str)


# LLM-generated content at query #15
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation with fields
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Verify that _pclass_fields is set correctly
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    
    # Verify that slots are set correctly
    assert TestClass.__slots__ == ('_pclass_frozen', 'x', 'y', '__weakref__')
    
    # Test that invariants storage is set up
    assert hasattr(TestClass, '_pclass_invariants')
    
    # Test inheritance - child class should have parent's fields
    class ParentClass(PClass):
        parent_field = field()
    
    class ChildClass(ParentClass):
        child_field = field()
    
    assert 'parent_field' in ChildClass._pclass_fields
    assert 'child_field' in ChildClass._pclass_fields
    
    # Verify slots include all fields
    assert 'parent_field' in ChildClass.__slots__
    assert 'child_field' in ChildClass.__slots__
    
    # Test that __weakref__ is only added to top-level PClass
    assert '__weakref__' in TestClass.__slots__
    assert '__weakref__' in ParentClass.__slots__
    assert '__weakref__' in ChildClass.__slots__
    
    # Test that non-PClass inheritance works correctly
    class Mixin:
        mixin_field = "mixin"
    
    class MixedClass(PClass, Mixin):
        z = field()
    
    # Should still have PClass functionality
    assert 'z' in MixedClass._pclass_fields
    assert 'z' in MixedClass.__slots__
    
    # Test with custom __slots__ in base class
    class BaseWithSlots:
        __slots__ = ('base_slot',)
    
    class SlottedClass(PClass, BaseWithSlots):
        a = field()
    
    # PClassMeta should handle multiple inheritance with slots
    assert 'a' in SlottedClass.__slots__
    assert '_pclass_frozen' in SlottedClass.__slots__
    
    # Test that field types are properly stored
    from pyrsistent import field
    
    class TypedClass(PClass):
        required = field(type=str, mandatory=True)
        optional = field(type=int, initial=0)
    
    assert TypedClass._pclass_fields['required'].type == str
    assert TypedClass._pclass_fields['required'].mandatory is True
    assert TypedClass._pclass_fields['optional'].type == int
    assert TypedClass._pclass_fields['optional'].initial == 0
    
    # Test that class can be instantiated after metaclass processing
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #16
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    # Test basic serialization
    obj = TestClass(x=1, y="test", z=[1, 2, 3])
    result = obj.serialize()
    assert result == {'x': 1, 'y': 'test', 'z': [1, 2, 3]}
    
    # Test serialization with custom serializer
    def custom_serializer(value, format=None):
        if isinstance(value, int):
            return f"int:{value}"
        return value
    
    class TestClassWithSerializer(PClass):
        x = field(serializer=custom_serializer)
        y = field()
    
    obj = TestClassWithSerializer(x=42, y="hello")
    result = obj.serialize()
    assert result == {'x': 'int:42', 'y': 'hello'}
    
    # Test serialization with format parameter
    def format_serializer(value, format=None):
        if format == 'json':
            return f"json:{value}"
        return value
    
    class TestClassWithFormat(PClass):
        x = field(serializer=format_serializer)
        y = field()
    
    obj = TestClassWithFormat(x=100, y="world")
    result = obj.serialize(format='json')
    assert result == {'x': 'json:100', 'y': 'world'}
    
    # Test serialization with missing optional field
    class TestClassOptional(PClass):
        x = field(mandatory=True)
        y = field(mandatory=False)
    
    obj = TestClassOptional(x=1)
    result = obj.serialize()
    assert result == {'x': 1}
    
    # Test serialization with initial value
    class TestClassInitial(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClassInitial(y=20)
    result = obj.serialize()
    assert result == {'x': 10, 'y': 20}
    
    # Test serialization with factory
    def factory_func(value):
        return value * 2
    
    class TestClassFactory(PClass):
        x = field(factory=factory_func)
    
    obj = TestClassFactory(x=5)
    result = obj.serialize()
    assert result == {'x': 10}
    
    # Test that serialize returns a new dict, not a reference
    obj = TestClass(x=1, y=2)
    result = obj.serialize()
    result['x'] = 999
    assert obj.x == 1  # Original should not be modified
    
    # Test serialization of nested PClass
    class InnerClass(PClass):
        a = field()
    
    class OuterClass(PClass):
        inner = field()
        value = field()
    
    inner = InnerClass(a=42)
    outer = OuterClass(inner=inner, value=100)
    result = outer.serialize()
    assert isinstance(result['inner'], dict)
    assert result['inner'] == {'a': 42}
    assert result['value'] == 100
    
    # Test serialization with multiple fields and various types
    class ComplexClass(PClass):
        int_field = field()
        str_field = field()
        list_field = field()
        dict_field = field()
        none_field = field()
    
    obj = ComplexClass(
        int_field=42,
        str_field="hello",
        list_field=[1, 2, 3],
        dict_field={'a': 1},
        none_field=None
    )
    result = obj.serialize()
    assert result['int_field'] == 42
    assert result['str_field'] == "hello"
    assert result['list_field'] == [1, 2, 3]
    assert result['dict_field'] == {'a': 1}
    assert result['none_field'] is None


# LLM-generated content at query #17
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2
    assert not (obj1 != obj2)

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3
    assert not (obj1 == obj3)

    # Test equality with same object
    assert obj1 == obj1

    # Test inequality with different class
    class OtherClass(PClass):
        x = field()
        y = field()

    obj4 = OtherClass(x=1, y=2)
    assert obj1 != obj4
    assert not (obj1 == obj4)

    # Test equality with missing values (using default initial)
    class ClassWithInitial(PClass):
        x = field()
        y = field(initial=10)

    obj5 = ClassWithInitial(x=1)
    obj6 = ClassWithInitial(x=1)
    assert obj5 == obj6

    # Test inequality when one attribute is missing
    class ClassWithMandatory(PClass):
        x = field(mandatory=True)
        y = field()

    obj7 = ClassWithMandatory(x=1, y=2)
    obj8 = ClassWithMandatory(x=1)
    assert obj7 != obj8

    # Test equality with None values
    obj9 = TestClass(x=None, y=None)
    obj10 = TestClass(x=None, y=None)
    assert obj9 == obj10

    # Test inequality with mixed None values
    obj11 = TestClass(x=1, y=None)
    obj12 = TestClass(x=None, y=2)
    assert obj11 != obj12

    # Test comparison with non-PClass object
    assert obj1 != "not a PClass"
    assert obj1 != 123
    assert obj1 != {"x": 1, "y": 2}

    # Test that NotImplemented is returned for non-matching types
    result = obj1.__eq__("not a PClass")
    assert result is NotImplemented

    # Test equality with evolved objects
    evolver = obj1.evolver()
    evolver.set('x', 10)
    obj13 = evolver.persistent()
    obj14 = TestClass(x=10, y=2)
    assert obj13 == obj14

    # Test inequality after removal and re-addition
    evolver2 = obj1.evolver()
    evolver2.remove('x')
    evolver2.set('x', 1)
    obj15 = evolver2.persistent()
    assert obj1 == obj15


# LLM-generated content at query #18
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation with fields
    class TestClass(PClass):
        x = field()
        y = field()
    
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    assert TestClass.__slots__ == ('_pclass_frozen', 'x', 'y', '__weakref__')
    
    # Test inheritance with PClass
    class ParentClass(PClass):
        a = field()
    
    class ChildClass(ParentClass):
        b = field()
    
    assert 'a' in ChildClass._pclass_fields
    assert 'b' in ChildClass._pclass_fields
    assert ChildClass.__slots__ == ('_pclass_frozen', 'a', 'b')
    
    # Test that __weakref__ is only added to top-level PClass
    assert '__weakref__' in ParentClass.__slots__
    assert '__weakref__' not in ChildClass.__slots__
    
    # Test with invariants
    def invariant_func(obj):
        return True, ""
    
    class ClassWithInvariant(PClass):
        __invariant__ = invariant_func
        value = field()
    
    assert invariant_func in ClassWithInvariant._pclass_invariants
    
    # Test multiple invariants
    def invariant_func2(obj):
        return True, ""
    
    class ClassWithMultipleInvariants(PClass):
        __invariant__ = [invariant_func, invariant_func2]
        value = field()
    
    assert len(ClassWithMultipleInvariants._pclass_invariants) == 2
    assert invariant_func in ClassWithMultipleInvariants._pclass_invariants
    assert invariant_func2 in ClassWithMultipleInvariants._pclass_invariants
    
    # Test that non-PClass inheritance doesn't get PClass treatment
    class RegularClass:
        pass
    
    class MixedClass(RegularClass, PClass):
        z = field()
    
    # Should still have PClass attributes
    assert hasattr(MixedClass, '_pclass_fields')
    assert 'z' in MixedClass._pclass_fields
    assert '_pclass_frozen' in MixedClass.__slots__
    
    # Test empty PClass
    class EmptyPClass(PClass):
        pass
    
    assert EmptyPClass.__slots__ == ('_pclass_frozen', '__weakref__')
    assert len(EmptyPClass._pclass_fields) == 0


# LLM-generated content at query #19
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    # Test basic set with keyword arguments
    obj = TestClass(x=1, y=2, z=3)
    obj2 = obj.set(x=10)
    assert obj2.x == 10
    assert obj2.y == 2
    assert obj2.z == 3
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    obj3 = obj.set('y', 20)
    assert obj3.y == 20
    assert obj3.x == 1
    assert obj3.z == 3

    # Test set multiple fields
    obj4 = obj.set(x=100, z=300)
    assert obj4.x == 100
    assert obj4.y == 2
    assert obj4.z == 300

    # Test set with non-existent field should raise AttributeError
    obj = TestClass(x=1, y=2, z=3)
    try:
        obj.set(nonexistent=5)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "nonexistent" in str(e)

    # Test set preserves other attributes when using positional args
    obj = TestClass(x=1, y=2, z=3)
    obj5 = obj.set('x', 99)
    assert obj5.x == 99
    assert obj5.y == 2
    assert obj5.z == 3

    # Test set returns new instance of same class
    obj = TestClass(x=1, y=2, z=3)
    obj6 = obj.set(x=7)
    assert isinstance(obj6, TestClass)
    assert obj6 is not obj

    # Test set with field that has factory
    class FactoryClass(PClass):
        x = field(type=int, factory=lambda v: v * 2)

    factory_obj = FactoryClass(x=5)
    assert factory_obj.x == 10
    factory_obj2 = factory_obj.set(x=3)
    assert factory_obj2.x == 6

    # Test set maintains equality and hash consistency
    obj = TestClass(x=1, y=2, z=3)
    obj7 = obj.set(x=1)  # Same value
    assert obj == obj7
    assert hash(obj) == hash(obj7)
    
    obj8 = obj.set(x=4)  # Different value
    assert obj != obj8
    assert hash(obj) != hash(obj8)

    # Test set with mandatory field
    class MandatoryClass(PClass):
        x = field(mandatory=True)
        y = field()

    mand_obj = MandatoryClass(x=1, y=2)
    mand_obj2 = mand_obj.set(y=3)
    assert mand_obj2.x == 1
    assert mand_obj2.y == 3

    # Test set with initial value field
    class InitialClass(PClass):
        x = field(initial=10)
        y = field()

    init_obj = InitialClass(y=5)
    assert init_obj.x == 10
    init_obj2 = init_obj.set(x=20)
    assert init_obj2.x == 20
    assert init_obj2.y == 5


# LLM-generated content at query #20
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation with fields
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Verify that _pclass_fields is set correctly
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    
    # Verify that slots are set correctly
    assert TestClass.__slots__ == ('_pclass_frozen', 'x', 'y', '__weakref__')
    
    # Test that invariants storage is set up
    assert hasattr(TestClass, '_pclass_invariants')
    
    # Test inheritance - child class should have parent's fields
    class ParentClass(PClass):
        parent_field = field()
    
    class ChildClass(ParentClass):
        child_field = field()
    
    assert 'parent_field' in ChildClass._pclass_fields
    assert 'child_field' in ChildClass._pclass_fields
    
    # Test that __slots__ includes all fields
    assert 'parent_field' in ChildClass.__slots__
    assert 'child_field' in ChildClass.__slots__
    
    # Test that only top-level PClass gets __weakref__
    assert '__weakref__' in ParentClass.__slots__
    assert '__weakref__' in ChildClass.__slots__
    
    # Test that non-PClass inheritance works correctly
    class Mixin:
        mixin_field = 42
    
    class MixedClass(PClass, Mixin):
        pclass_field = field()
    
    # Should still have PClass functionality
    assert 'pclass_field' in MixedClass._pclass_fields
    assert MixedClass.__slots__ == ('_pclass_frozen', 'pclass_field', '__weakref__')
    
    # Mixin fields should be accessible as class attributes
    assert MixedClass.mixin_field == 42


# LLM-generated content at query #21
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation with fields
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Verify that _pclass_fields is set correctly
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    
    # Verify that slots are set correctly
    assert TestClass.__slots__ == ('_pclass_frozen', 'x', 'y', '__weakref__')
    
    # Test that invariants storage is set up
    assert hasattr(TestClass, '_pclass_invariants')
    
    # Test inheritance - child class should have parent's fields
    class ParentClass(PClass):
        parent_field = field()
    
    class ChildClass(ParentClass):
        child_field = field()
    
    assert 'parent_field' in ChildClass._pclass_fields
    assert 'child_field' in ChildClass._pclass_fields
    
    # Verify slots include all fields
    assert 'parent_field' in ChildClass.__slots__
    assert 'child_field' in ChildClass.__slots__
    
    # Test that non-PClass inheritance works correctly
    class Mixin:
        mixin_field = field()
    
    class MixedClass(PClass, Mixin):
        own_field = field()
    
    # Only PClass fields should be in _pclass_fields
    assert 'own_field' in MixedClass._pclass_fields
    assert 'mixin_field' not in MixedClass._pclass_fields
    
    # Test __weakref__ is only added to top-level PClass
    assert '__weakref__' in TestClass.__slots__
    
    # Test that class can be instantiated
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2
    
    # Test that fields are properly initialized
    class WithInitial(PClass):
        x = field(initial=10)
        y = field()
    
    instance2 = WithInitial(y=20)
    assert instance2.x == 10
    assert instance2.y == 20


# LLM-generated content at query #22
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field
    import pickle

    class SimpleClass(PClass):
        x = field()
        y = field()

    # Test basic pickling and unpickling
    obj = SimpleClass(x=10, y="test")
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    
    assert obj == unpickled
    assert obj.x == unpickled.x
    assert obj.y == unpickled.y
    assert isinstance(unpickled, SimpleClass)

    # Test with missing optional field
    class ClassWithOptional(PClass):
        required = field(mandatory=True)
        optional = field()

    obj2 = ClassWithOptional(required=1)
    pickled2 = pickle.dumps(obj2)
    unpickled2 = pickle.loads(pickled2)
    
    assert obj2 == unpickled2
    assert obj2.required == unpickled2.required
    assert not hasattr(unpickled2, 'optional')

    # Test with all fields present
    obj3 = ClassWithOptional(required=1, optional=2)
    pickled3 = pickle.dumps(obj3)
    unpickled3 = pickle.loads(pickled3)
    
    assert obj3 == unpickled3
    assert obj3.optional == unpickled3.optional

    # Test that __reduce__ returns correct tuple structure
    reduce_result = obj.__reduce__()
    assert len(reduce_result) == 2
    assert reduce_result[0] == obj._restore_pickle
    assert isinstance(reduce_result[1], tuple)
    assert len(reduce_result[1]) == 3
    assert reduce_result[1][0] == SimpleClass
    assert isinstance(reduce_result[1][1], dict)
    assert reduce_result[1][2] == 0  # Default pickle protocol data

    # Test that data dict contains correct field values
    data_dict = reduce_result[1][1]
    assert data_dict['x'] == 10
    assert data_dict['y'] == "test"

    # Test with nested pickling
    class ContainerClass(PClass):
        item = field()

    container = ContainerClass(item=obj)
    pickled_container = pickle.dumps(container)
    unpickled_container = pickle.loads(pickled_container)
    
    assert container == unpickled_container
    assert container.item == unpickled_container.item
    assert isinstance(unpickled_container.item, SimpleClass)


# LLM-generated content at query #23
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field
    import pickle

    class SimpleClass(PClass):
        x = field()
        y = field()

    # Test basic pickling and unpickling
    obj = SimpleClass(x=10, y="test")
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    
    assert obj == unpickled
    assert obj.x == unpickled.x
    assert obj.y == unpickled.y
    
    # Test that __reduce__ returns correct tuple structure
    reduce_result = obj.__reduce__()
    assert len(reduce_result) == 2
    assert reduce_result[0] == obj._restore_pickle
    assert isinstance(reduce_result[1], tuple)
    assert len(reduce_result[1]) == 3
    
    # Test restore function with data
    restore_func, args = reduce_result
    cls, data, extra = args
    assert cls is SimpleClass
    assert data == {"x": 10, "y": "test"}
    
    # Test pickling with missing optional field
    class ClassWithOptional(PClass):
        required = field(mandatory=True)
        optional = field()
    
    obj2 = ClassWithOptional(required=1)
    pickled2 = pickle.dumps(obj2)
    unpickled2 = pickle.loads(pickled2)
    
    assert obj2 == unpickled2
    assert obj2.required == unpickled2.required
    assert not hasattr(obj2, 'optional')
    assert not hasattr(unpickled2, 'optional')
    
    # Test that __reduce__ only includes fields that exist on the instance
    reduce_result2 = obj2.__reduce__()
    _, args2 = reduce_result2
    _, data2, _ = args2
    assert data2 == {"required": 1}
    assert "optional" not in data2
    
    # Test pickling with nested PClass
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
        name = field()
    
    inner_obj = InnerClass(value=42)
    outer_obj = OuterClass(inner=inner_obj, name="outer")
    
    pickled3 = pickle.dumps(outer_obj)
    unpickled3 = pickle.loads(pickled3)
    
    assert outer_obj == unpickled3
    assert outer_obj.name == unpickled3.name
    assert outer_obj.inner == unpickled3.inner
    assert outer_obj.inner.value == unpickled3.inner.value
    
    # Test that hash remains consistent after pickling
    original_hash = hash(obj)
    unpickled_hash = hash(unpickled)
    assert original_hash == unpickled_hash
    
    # Test with field that has initial value
    class ClassWithInitial(PClass):
        x = field(initial=100)
        y = field()
    
    obj3 = ClassWithInitial(y=50)
    pickled4 = pickle.dumps(obj3)
    unpickled4 = pickle.loads(pickled4)
    
    assert obj3 == unpickled4
    assert obj3.x == unpickled4.x == 100
    assert obj3.y == unpickled4.y == 50


# LLM-generated content at query #24
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=10, y=20)
    result = obj.serialize()
    assert result == {'x': 10, 'y': 20}

    class TestClassWithSerializer(PClass):
        x = field(serializer=lambda v, f: v * 2)
        y = field()

    obj = TestClassWithSerializer(x=5, y=15)
    result = obj.serialize()
    assert result == {'x': 10, 'y': 15}

    obj = TestClassWithSerializer(x=3, y=7)
    result = obj.serialize(format='custom')
    assert result == {'x': 6, 'y': 7}

    class TestClassPartial(PClass):
        x = field(mandatory=True)
        y = field(initial=100)

    obj = TestClassPartial(x=50)
    result = obj.serialize()
    assert result == {'x': 50, 'y': 100}

    class TestClassEmpty(PClass):
        pass

    obj = TestClassEmpty()
    result = obj.serialize()
    assert result == {}

    class TestClassNested(PClass):
        x = field(serializer=lambda v, f: {'value': v})
        y = field()

    obj = TestClassNested(x=42, y='test')
    result = obj.serialize()
    assert result == {'x': {'value': 42}, 'y': 'test'}


# LLM-generated content at query #25
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class SimpleClass(PClass):
        x = field()
        y = field()

    class OtherClass(PClass):
        x = field()
        y = field()

    # Test equality with same instance
    obj1 = SimpleClass(x=1, y=2)
    assert obj1 == obj1

    # Test equality with different instance but same values
    obj2 = SimpleClass(x=1, y=2)
    assert obj1 == obj2
    assert obj2 == obj1

    # Test inequality with different values
    obj3 = SimpleClass(x=2, y=2)
    assert obj1 != obj3
    assert not (obj1 == obj3)

    # Test inequality with different class but same values
    obj4 = OtherClass(x=1, y=2)
    assert obj1 != obj4
    assert not (obj1 == obj4)

    # Test equality with missing fields (optional fields)
    class ClassWithOptional(PClass):
        x = field(mandatory=True)
        y = field(mandatory=False)

    obj5 = ClassWithOptional(x=1)
    obj6 = ClassWithOptional(x=1)
    assert obj5 == obj6

    # Test inequality when one has optional field set
    obj7 = ClassWithOptional(x=1, y=2)
    assert obj5 != obj7

    # Test comparison with non-PClass object
    assert obj1 != "not a PClass"
    assert obj1 != 123
    assert obj1 != None
    assert obj1 != {"x": 1, "y": 2}

    # Test hash consistency with equality
    obj8 = SimpleClass(x=1, y=2)
    obj9 = SimpleClass(x=1, y=2)
    assert hash(obj8) == hash(obj9)
    assert obj8 == obj9

    # Test with nested PClass objects
    class NestedClass(PClass):
        inner = field()

    inner1 = SimpleClass(x=1, y=2)
    inner2 = SimpleClass(x=1, y=2)
    inner3 = SimpleClass(x=3, y=4)
    
    nested1 = NestedClass(inner=inner1)
    nested2 = NestedClass(inner=inner2)
    nested3 = NestedClass(inner=inner3)
    
    assert nested1 == nested2
    assert nested1 != nested3

    # Test with None values
    class ClassWithNone(PClass):
        x = field()
        y = field()

    obj10 = ClassWithNone(x=None, y=2)
    obj11 = ClassWithNone(x=None, y=2)
    obj12 = ClassWithNone(x=1, y=2)
    
    assert obj10 == obj11
    assert obj10 != obj12

    # Test __ne__ method
    assert obj1.__ne__(obj2) is False
    assert obj1.__ne__(obj3) is True
    assert obj1.__ne__("string") is True


