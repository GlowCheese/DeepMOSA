####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_CheckedPSet_serialize():
    class TestSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    # Test with simple integers
    test_set = TestSet([1, 2, 3])
    serialized = test_set.serialize()
    assert serialized == {1, 2, 3}

    # Test with custom serializer
    class CustomSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda self, _, value: str(value)

    custom_set = CustomSet([1, 2, 3])
    serialized_custom = custom_set.serialize()
    assert serialized_custom == {'1', '2', '3'}

    # Test with nested CheckedType
    class NestedSet(CheckedPSet):
        __type__ = TestSet

    nested_set = NestedSet([TestSet([1, 2]), TestSet([3, 4])])
    serialized_nested = nested_set.serialize()
    assert serialized_nested == {frozenset({1, 2}), frozenset({3, 4})}

    # Test with empty set
    empty_set = TestSet()
    serialized_empty = empty_set.serialize()
    assert serialized_empty == set()


# LLM-generated content at query #2
#--------------------------

```python
def test_CheckedPSet___new__():
    # Test empty initialization
    class TestSet(CheckedPSet):
        __type__ = int
    s = TestSet()
    assert len(s) == 0

    # Test initialization with iterable
    s = TestSet([1, 2, 3])
    assert len(s) == 3
    assert 1 in s
    assert 2 in s
    assert 3 in s

    # Test initialization with PMap
    from pyrsistent import pset
    pm = pset([1, 2, 3])
    s = TestSet(pm)
    assert len(s) == 3
    assert 1 in s

    # Test type checking
    class StrSet(CheckedPSet):
        __type__ = str
    s = StrSet(["a", "b"])
    assert len(s) == 2

    with pytest.raises(CheckedValueTypeError):
        StrSet([1, 2])

    # Test with multiple types
    class MultiSet(CheckedPSet):
        __type__ = (int, str)
    s = MultiSet([1, "a"])
    assert len(s) == 2

    # Test invariant
    class PositiveSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x >= 0, "Negative value")

    s = PositiveSet([1, 2, 3])
    assert len(s) == 3

    with pytest.raises(InvariantException):
        PositiveSet([-1, 2])

    # Test with checked type
    class InnerType(CheckedType):
        def __init__(self, value):
            self.value = value

        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            if isinstance(source_data, cls):
                return source_data
            return cls(source_data)

        def serialize(self, format=None):
            return self.value

    class ContainerSet(CheckedPSet):
        __type__ = InnerType

    s = ContainerSet([InnerType(1), InnerType(2)])
    assert len(s) == 2

    # Test with raw data that should be converted
    s = ContainerSet.create([{"value": 1}, {"value": 2}])
    assert len(s) == 2
    assert all(isinstance(x, InnerType) for x in s)


# LLM-generated content at query #3
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: Invariant returns a single boolean result
    def single_bool_invariant(x):
        return x > 0, "Positive number required"

    wrapped = wrap_invariant(single_bool_invariant)
    assert wrapped(5) == (True, ())
    assert wrapped(-1) == (False, ("Positive number required",))

    # Test case 2: Invariant returns multiple boolean results
    def multi_bool_invariant(x):
        return (x > 0, "Positive required"), (x % 2 == 0, "Even required")

    wrapped = wrap_invariant(multi_bool_invariant)
    assert wrapped(4) == (True, ())
    assert wrapped(5) == (False, ("Even required",))
    assert wrapped(-1) == (False, ("Positive required", "Even required"))

    # Test case 3: Invariant returns non-boolean results (should be treated as truthy)
    def non_bool_invariant(x):
        return (x, "Value is truthy"), (not x, "Value is falsy")

    wrapped = wrap_invariant(non_bool_invariant)
    assert wrapped(1) == (True, ())
    assert wrapped(0) == (False, ("Value is falsy",))


# LLM-generated content at query #4
#--------------------------

```python
def test_CheckedPSet_serialize():
    class TestSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    # Test with valid elements
    s = TestSet([1, 2, 3])
    assert s.serialize() == {1, 2, 3}

    # Test with nested CheckedType
    class InnerSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda self, _, x: x * 2

    inner = InnerSet([4, 5])
    outer = TestSet([inner])
    assert outer.serialize() == {InnerSet([4, 5]).serialize()}

    # Test with custom serializer
    class CustomSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda self, _, x: str(x)

    cs = CustomSet([6, 7, 8])
    assert cs.serialize() == {"6", "7", "8"}

    # Test empty set
    empty = TestSet()
    assert empty.serialize() == set()


# LLM-generated content at query #5
#--------------------------

```python
def test_CheckedPSet___new__():
    # Test with empty initial
    class TestSet(CheckedPSet):
        __type__ = (int,)
    s = TestSet()
    assert isinstance(s, CheckedPSet)
    assert len(s) == 0

    # Test with iterable initial
    s = TestSet([1, 2, 3])
    assert len(s) == 3
    assert 1 in s
    assert 2 in s
    assert 3 in s

    # Test with PMap initial
    from pyrsistent import pset
    pm = pset([1, 2, 3])
    s = TestSet(pm)
    assert len(s) == 3
    assert 1 in s
    assert 2 in s
    assert 3 in s

    # Test with invalid type
    with pytest.raises(CheckedValueTypeError):
        TestSet([1, 2, "invalid"])

    # Test with invariant
    class PositiveSet(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda x: (x >= 0, "Negative value")

    s = PositiveSet([1, 2, 3])
    assert len(s) == 3

    with pytest.raises(InvariantException):
        PositiveSet([1, -2, 3])


# LLM-generated content at query #6
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean
    def simple_invariant(x):
        return x > 0, "Positive"
    wrapped = wrap_invariant(simple_invariant)
    assert wrapped(1) == (True, "Positive")
    assert wrapped(-1) == (False, "Positive")

    # Test with an invariant that returns multiple results
    def multi_invariant(x):
        return (x > 0, "Positive"), (x % 2 == 0, "Even")
    wrapped = wrap_invariant(multi_invariant)
    assert wrapped(2) == (True, ("Positive", "Even"))
    assert wrapped(1) == (False, ("Positive",))
    assert wrapped(-1) == (False, ())

    # Test with an invariant that returns a single False result
    def false_invariant(x):
        return False, "Always False"
    wrapped = wrap_invariant(false_invariant)
    assert wrapped(1) == (False, "Always False")


# LLM-generated content at query #7
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean
    def simple_invariant(x):
        return x > 0, "Value must be positive"

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(5) == (True, ())
    assert wrapped_simple(-1) == (False, ("Value must be positive",))

    # Test with an invariant that returns multiple test results
    def complex_invariant(x):
        return (x > 0, "Value must be positive"), (x < 10, "Value must be less than 10")

    wrapped_complex = wrap_invariant(complex_invariant)
    assert wrapped_complex(5) == (True, ())
    assert wrapped_complex(-1) == (False, ("Value must be positive",))
    assert wrapped_complex(15) == (False, ("Value must be less than 10",))
    assert wrapped_complex(15) == (False, ("Value must be less than 10",))

    # Test with an invariant that returns mixed results
    def mixed_invariant(x):
        return (x > 0, "Value must be positive"), (True, None)

    wrapped_mixed = wrap_invariant(mixed_invariant)
    assert wrapped_mixed(5) == (True, ())
    assert wrapped_mixed(-1) == (False, ("Value must be positive",))


# LLM-generated content at query #8
#--------------------------

```python
def test_CheckedPSet_serialize():
    class TestSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Non-positive")

    # Test with valid data
    test_set = TestSet([1, 2, 3])
    assert test_set.serialize() == {1, 2, 3}

    # Test with nested CheckedType
    class InnerSet(CheckedPSet):
        __type__ = int

    class OuterSet(CheckedPSet):
        __type__ = (int, InnerSet)

    inner = InnerSet([4, 5])
    outer = OuterSet([1, 2, inner])
    serialized = outer.serialize()
    assert serialized == {1, 2, {4, 5}}

    # Test with custom serializer
    class CustomSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda self, _, value: str(value)

    custom_set = CustomSet([1, 2, 3])
    assert custom_set.serialize() == {'1', '2', '3'}

    # Test with empty set
    empty_set = TestSet()
    assert empty_set.serialize() == set()


# LLM-generated content at query #9
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean result
    def simple_invariant(x):
        return x > 0, "Positive"

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(1) == (True, "Positive")
    assert wrapped_simple(-1) == (False, "Positive")

    # Test with an invariant that returns multiple results
    def complex_invariant(x):
        return (x > 0, "Positive"), (x % 2 == 0, "Even")

    wrapped_complex = wrap_invariant(complex_invariant)
    assert wrapped_complex(2) == (True, ())
    assert wrapped_complex(1) == (False, ("Even",))
    assert wrapped_complex(-1) == (False, ("Positive", "Even"))

    # Test with an invariant that returns a mix of single and multiple results
    def mixed_invariant(x):
        if x > 0:
            return True, "Positive"
        else:
            return (False, "Negative"), (False, "Non-positive")

    wrapped_mixed = wrap_invariant(mixed_invariant)
    assert wrapped_mixed(1) == (True, "Positive")
    assert wrapped_mixed(-1) == (False, ("Negative", "Non-positive"))


# LLM-generated content at query #10
#--------------------------

```python
def test_CheckedPSet_serialize():
    class TestSet(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    # Test with empty set
    empty_set = TestSet()
    assert empty_set.serialize() == set()

    # Test with valid elements
    test_set = TestSet([1, 2, 3])
    assert test_set.serialize() == {1, 2, 3}

    # Test with custom serializer
    class CustomSet(CheckedPSet):
        __type__ = (int,)
        __serializer__ = lambda self, _, value: value * 2

    custom_set = CustomSet([1, 2, 3])
    assert custom_set.serialize() == {2, 4, 6}

    # Test with nested CheckedType
    class NestedSet(CheckedPSet):
        __type__ = (TestSet,)

    nested_set = NestedSet([TestSet([1, 2]), TestSet([3, 4])])
    assert nested_set.serialize() == [set([1, 2]), set([3, 4])]


# LLM-generated content at query #11
#--------------------------

```python
def test_CheckedPSet_serialize():
    class TestCheckedPSet(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Value must be positive")

    # Test with valid data
    test_set = TestCheckedPSet([1, 2, 3])
    serialized = test_set.serialize()
    assert serialized == {1, 2, 3}

    # Test with custom serializer
    class CustomSerializer(CheckedPSet):
        __type__ = (int,)
        __serializer__ = lambda self, _, value: str(value)

    custom_set = CustomSerializer([1, 2, 3])
    serialized_custom = custom_set.serialize()
    assert serialized_custom == {"1", "2", "3"}

    # Test with nested CheckedType
    class NestedCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value

        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            if isinstance(source_data, cls):
                return source_data
            return cls(source_data)

        def serialize(self, format=None):
            return self.value

    class NestedCheckedPSet(CheckedPSet):
        __type__ = (NestedCheckedType,)

    nested_set = NestedCheckedPSet([NestedCheckedType(1), NestedCheckedType(2)])
    serialized_nested = nested_set.serialize()
    assert serialized_nested == {1, 2}


# LLM-generated content at query #12
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with a single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]

    # Test with a preserved iterable type (Enum)
    class TestEnum(Enum):
        A = 1
        B = 2
    assert maybe_parse_user_type(TestEnum) == [TestEnum]

    # Test with a string
    assert maybe_parse_user_type("int") == ["int"]

    # Test with an iterable of types
    assert maybe_parse_user_type([int, str]) == [int, str]
    assert maybe_parse_user_type((int, str)) == [int, str]

    # Test with nested iterables
    assert maybe_parse_user_type([(int, str), float]) == [int, str, float]

    # Test with invalid input
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)  # Not a type, string, or iterable


# LLM-generated content at query #13
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean
    def simple_invariant(x):
        return x > 0, "Positive"

    wrapped = wrap_invariant(simple_invariant)
    assert wrapped(5) == (True, "Positive")
    assert wrapped(-1) == (False, "Positive")

    # Test with an invariant that returns multiple test results
    def complex_invariant(x):
        return [(x > 0, "Positive"), (x % 2 == 0, "Even")]

    wrapped = wrap_invariant(complex_invariant)
    assert wrapped(4) == (True, ("Positive", "Even"))
    assert wrapped(5) == (False, ("Positive",))
    assert wrapped(-1) == (False, ())


# LLM-generated content at query #14
#--------------------------

```python
def test_store_invariants():
    # Test basic invariant storage
    class TestClass:
        pass

    dct = {}
    bases = (TestClass,)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert 'invariants' in dct
    assert dct['invariants'] == ()

    # Test single invariant
    def test_inv():
        return True, 'test'

    class TestClassWithInvariant:
        invariant = test_inv

    dct = {}
    bases = (TestClassWithInvariant,)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert len(dct['invariants']) == 1
    assert dct['invariants'][0]() == (True, 'test')

    # Test multiple invariants
    def test_inv2():
        return True, 'test2'

    class TestClassWithMultipleInvariants(TestClassWithInvariant):
        invariant = test_inv2

    dct = {}
    bases = (TestClassWithMultipleInvariants,)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert len(dct['invariants']) == 2
    assert dct['invariants'][0]() == (True, 'test')
    assert dct['invariants'][1]() == (True, 'test2')

    # Test invariant inheritance
    class TestClassInherited(TestClassWithInvariant):
        pass

    dct = {}
    bases = (TestClassInherited,)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert len(dct['invariants']) == 1
    assert dct['invariants'][0]() == (True, 'test')

    # Test non-callable invariant raises TypeError
    class TestClassWithNonCallableInvariant:
        invariant = 'not_callable'

    dct = {}
    bases = (TestClassWithNonCallableInvariant,)
    try:
        store_invariants(dct, bases, 'invariants', 'invariant')
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

    # Test invariant that returns multiple results
    def test_inv_multiple():
        return [(True, 'test1'), (False, 'test2')]

    class TestClassWithMultipleResultInvariant:
        invariant = test_inv_multiple

    dct = {}
    bases = (TestClassWithMultipleResultInvariant,)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert len(dct['invariants']) == 1
    result = dct['invariants'][0]()
    assert result == (False, ('test2',))


# LLM-generated content at query #15
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean result
    def simple_invariant(x):
        return x > 0, "Value must be positive"

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(5) == (True, ())
    assert wrapped_simple(-1) == (False, ("Value must be positive",))

    # Test with an invariant that returns multiple test results
    def complex_invariant(x):
        return [(x > 0, "Value must be positive"), (x % 2 == 0, "Value must be even")]

    wrapped_complex = wrap_invariant(complex_invariant)
    assert wrapped_complex(4) == (True, ())
    assert wrapped_complex(5) == (False, ("Value must be even",))
    assert wrapped_complex(-1) == (False, ("Value must be positive", "Value must be even"))

    # Test with an invariant that returns a mix of boolean and multiple results
    def mixed_invariant(x):
        if x < 0:
            return False, "Value must be positive"
        return [(x % 2 == 0, "Value must be even"), (x % 3 == 0, "Value must be divisible by 3")]

    wrapped_mixed = wrap_invariant(mixed_invariant)
    assert wrapped_mixed(6) == (True, ())
    assert wrapped_mixed(5) == (False, ("Value must be even", "Value must be divisible by 3"))
    assert wrapped_mixed(-1) == (False, ("Value must be positive",))


# LLM-generated content at query #16
#--------------------------

```python
def test_store_invariants():
    # Test basic invariant storage
    class A(CheckedType):
        __invariant__ = lambda self: (True, "ok")

    assert hasattr(A, '_invariant')
    assert len(A._invariant) == 1
    assert A._invariant[0](None) == (True, "ok")

    # Test multiple invariants
    class B(CheckedType):
        __invariant__ = lambda self: (True, "ok")

    class C(B):
        __invariant__ = lambda self: (False, "error")

    assert len(C._invariant) == 2
    assert C._invariant[0](None) == (True, "ok")
    assert C._invariant[1](None) == (False, "error")

    # Test non-callable invariant raises TypeError
    with pytest.raises(TypeError):
        class D(CheckedType):
            __invariant__ = "not callable"

    # Test invariant that returns multiple results
    class E(CheckedType):
        __invariant__ = lambda self: [(True, "ok1"), (False, "error1")]

    assert E._invariant[0](None) == (False, ("error1",))

    # Test inheritance with multiple invariants
    class F(CheckedType):
        __invariant__ = lambda self: (True, "ok")

    class G(F):
        __invariant__ = lambda self: (True, "ok2")

    class H(G):
        __invariant__ = lambda self: (False, "error2")

    assert len(H._invariant) == 3
    assert H._invariant[0](None) == (True, "ok")
    assert H._invariant[1](None) == (True, "ok2")
    assert H._invariant[2](None) == (False, "error2")


# LLM-generated content at query #17
#--------------------------

```python
def test_wrap_invariant():
    # Test single boolean return
    def single_true():
        return True, "OK"
    assert wrap_invariant(single_true)() == (True, "OK")

    def single_false():
        return False, "Error"
    assert wrap_invariant(single_false)() == (False, "Error")

    # Test multiple results that need merging
    def multiple_true():
        return [(True, "OK1"), (True, "OK2")]
    assert wrap_invariant(multiple_true)() == (True, ())

    def multiple_false():
        return [(True, "OK"), (False, "Error1"), (False, "Error2")]
    assert wrap_invariant(multiple_false)() == (False, ("Error1", "Error2"))

    def mixed_results():
        return [(True, "OK"), (False, "Error"), (True, "OK2")]
    assert wrap_invariant(mixed_results)() == (False, ("Error",))

    # Test empty results
    def empty_results():
        return []
    assert wrap_invariant(empty_results)() == (True, ())


# LLM-generated content at query #18
#--------------------------

```python
def test_CheckedPSet_serialize():
    class TestSet(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda x: (x > 0, "Non-positive")

    # Test with empty set
    empty_set = TestSet()
    assert empty_set.serialize() == set()

    # Test with single element
    single_set = TestSet([1])
    assert single_set.serialize() == {1}

    # Test with multiple elements
    multi_set = TestSet([1, 2, 3])
    assert multi_set.serialize() == {1, 2, 3}

    # Test with custom serializer
    class CustomSet(CheckedPSet):
        __type__ = (int,)
        __serializer__ = lambda self, _, value: value * 2

    custom_set = CustomSet([1, 2, 3])
    assert custom_set.serialize() == {2, 4, 6}

    # Test with nested CheckedType
    class NestedSet(CheckedPSet):
        __type__ = (TestSet,)

    nested_set = NestedSet([TestSet([1, 2]), TestSet([3, 4])])
    assert nested_set.serialize() == [set([1, 2]), set([3, 4])]


# LLM-generated content at query #19
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]

    # Test preserved iterable type (Enum)
    class TestEnum(Enum):
        A = 1
        B = 2
    assert maybe_parse_user_type(TestEnum) == [TestEnum]

    # Test string type
    assert maybe_parse_user_type('int') == ['int']
    assert maybe_parse_user_type('str') == ['str']

    # Test iterable of types
    assert maybe_parse_user_type([int, str]) == [int, str]
    assert maybe_parse_user_type((int, str)) == [int, str]

    # Test nested iterable of types
    assert maybe_parse_user_type([int, [str, float]]) == [int, str, float]
    assert maybe_parse_user_type((int, (str, float))) == [int, str, float]

    # Test mixed iterable with strings and types
    assert maybe_parse_user_type([int, 'str', float]) == [int, 'str', float]

    # Test invalid input
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)

    with pytest.raises(TypeError):
        maybe_parse_user_type(None)

    with pytest.raises(TypeError):
        maybe_parse_user_type(lambda x: x)


# LLM-generated content at query #20
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean result
    def simple_invariant():
        return True, "Success"

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple() == (True, "Success")

    # Test with an invariant that returns multiple results
    def multi_invariant():
        return [(True, "Test1"), (False, "Test2"), (True, "Test3")]

    wrapped_multi = wrap_invariant(multi_invariant)
    assert wrapped_multi() == (False, ("Test2",))

    # Test with an invariant that returns a single false result
    def false_invariant():
        return False, "Failure"

    wrapped_false = wrap_invariant(false_invariant)
    assert wrapped_false() == (False, "Failure")

    # Test with an invariant that returns multiple all-true results
    def all_true_invariant():
        return [(True, "Test1"), (True, "Test2")]

    wrapped_all_true = wrap_invariant(all_true_invariant)
    assert wrapped_all_true() == (True, ())

    # Test with an invariant that returns multiple all-false results
    def all_false_invariant():
        return [(False, "Test1"), (False, "Test2")]

    wrapped_all_false = wrap_invariant(all_false_invariant)
    assert wrapped_all_false() == (False, ("Test1", "Test2"))


# LLM-generated content at query #21
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with a single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]

    # Test with a preserved iterable type (Enum)
    class TestEnum(Enum):
        A = 1
        B = 2
    assert maybe_parse_user_type(TestEnum) == [TestEnum]

    # Test with a string
    assert maybe_parse_user_type("int") == ["int"]

    # Test with a tuple of types
    assert maybe_parse_user_type((int, str)) == [int, str]

    # Test with a nested iterable
    assert maybe_parse_user_type([(int, str), float]) == [int, str, float]

    # Test with an invalid type (should raise TypeError)
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)

    # Test with a non-iterable object (should raise TypeError)
    with pytest.raises(TypeError):
        maybe_parse_user_type(object())


# LLM-generated content at query #22
#--------------------------

```python
def test_wrap_invariant():
    # Test single boolean return
    def single_bool_invariant(x):
        return x > 0, "Positive"

    wrapped = wrap_invariant(single_bool_invariant)
    assert wrapped(1) == (True, "Positive")
    assert wrapped(-1) == (False, "Positive")

    # Test multiple boolean returns
    def multi_bool_invariant(x):
        return (x > 0, "Positive"), (x % 2 == 0, "Even")

    wrapped = wrap_invariant(multi_bool_invariant)
    assert wrapped(2) == (True, ("Positive", "Even"))
    assert wrapped(1) == (False, ("Positive",))
    assert wrapped(-1) == (False, ())


# LLM-generated content at query #23
#--------------------------

```python
def test_wrap_invariant():
    # Test with a single boolean return
    def single_bool_invariant():
        return True, "Success"
    wrapped = wrap_invariant(single_bool_invariant)
    assert wrapped() == (True, "Success")

    # Test with multiple boolean returns
    def multi_bool_invariant():
        return (True, "Success1"), (False, "Error1"), (True, "Success2")
    wrapped = wrap_invariant(multi_bool_invariant)
    assert wrapped() == (False, ("Error1",))

    # Test with no errors
    def no_error_invariant():
        return (True, "Success1"), (True, "Success2")
    wrapped = wrap_invariant(no_error_invariant)
    assert wrapped() == (True, ())

    # Test with all errors
    def all_error_invariant():
        return (False, "Error1"), (False, "Error2")
    wrapped = wrap_invariant(all_error_invariant)
    assert wrapped() == (False, ("Error1", "Error2"))


# LLM-generated content at query #24
#--------------------------

```python
def test_store_invariants():
    # Test basic invariant storage
    class TestClass1:
        pass

    dct = {}
    bases = (TestClass1,)
    store_invariants(dct, bases, 'invariants', 'invariant')

    assert 'invariants' in dct
    assert dct['invariants'] == ()

    # Test with a single invariant
    def test_invariant1(obj):
        return True, "Test passed"

    class TestClass2:
        invariant = test_invariant1

    dct = {}
    bases = (TestClass2,)
    store_invariants(dct, bases, 'invariants', 'invariant')

    assert 'invariants' in dct
    assert len(dct['invariants']) == 1
    assert callable(dct['invariants'][0])

    # Test with multiple invariants
    def test_invariant2(obj):
        return True, "Another test passed"

    class TestClass3:
        invariant = test_invariant2

    class TestClass4(TestClass2, TestClass3):
        pass

    dct = {}
    bases = (TestClass4,)
    store_invariants(dct, bases, 'invariants', 'invariant')

    assert 'invariants' in dct
    assert len(dct['invariants']) == 2
    assert all(callable(inv) for inv in dct['invariants'])

    # Test with non-callable invariant (should raise TypeError)
    class TestClass5:
        invariant = "not a function"

    dct = {}
    bases = (TestClass5,)

    try:
        store_invariants(dct, bases, 'invariants', 'invariant')
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

    # Test with invariant that returns multiple results
    def test_invariant3(obj):
        return [(True, "Test 1"), (False, "Test 2")]

    class TestClass6:
        invariant = test_invariant3

    dct = {}
    bases = (TestClass6,)
    store_invariants(dct, bases, 'invariants', 'invariant')

    assert 'invariants' in dct
    assert len(dct['invariants']) == 1
    result = dct['invariants'][0](None)
    assert result == (False, ("Test 2",))


# LLM-generated content at query #25
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean
    def simple_invariant(x):
        return x > 0, "Value must be positive"

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(5) == (True, ())
    assert wrapped_simple(-1) == (False, ("Value must be positive",))

    # Test with an invariant that returns multiple test results
    def complex_invariant(x):
        return (x > 0, "Value must be positive"), (x % 2 == 0, "Value must be even")

    wrapped_complex = wrap_invariant(complex_invariant)
    assert wrapped_complex(4) == (True, ())
    assert wrapped_complex(5) == (False, ("Value must be even",))
    assert wrapped_complex(-1) == (False, ("Value must be positive", "Value must be even"))

    # Test with an invariant that returns a single False with no message
    def no_message_invariant(x):
        return False, None

    wrapped_no_msg = wrap_invariant(no_message_invariant)
    assert wrapped_no_msg(10) == (False, (None,))

    # Test with an invariant that returns a single True with no message
    def true_no_message_invariant(x):
        return True, None

    wrapped_true_no_msg = wrap_invariant(true_no_message_invariant)
    assert wrapped_true_no_msg(10) == (True, ())


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__CheckedMapTypeMeta___new__():
    # Test basic creation
    class TestClass(metaclass=_CheckedMapTypeMeta):
        pass
    assert hasattr(TestClass, '_checked_key_types')
    assert hasattr(TestClass, '_checked_value_types')
    assert hasattr(TestClass, '_checked_invariants')
    assert hasattr(TestClass, '__serializer__')

    # Test with key types
    class TestKeyClass(metaclass=_CheckedMapTypeMeta):
        __key_type__ = int
    assert TestKeyClass._checked_key_types == [int]

    # Test with value types
    class TestValueClass(metaclass=_CheckedMapTypeMeta):
        __value_type__ = str
    assert TestValueClass._checked_value_types == [str]

    # Test with multiple key types
    class TestMultiKeyClass(metaclass=_CheckedMapTypeMeta):
        __key_type__ = [int, str]
    assert TestMultiKeyClass._checked_key_types == [int, str]

    # Test with multiple value types
    class TestMultiValueClass(metaclass=_CheckedMapTypeMeta):
        __value_type__ = [str, float]
    assert TestMultiValueClass._checked_value_types == [str, float]

    # Test with invariants
    def test_invariant(x):
        return x > 0, "Value must be positive"

    class TestInvariantClass(metaclass=_CheckedMapTypeMeta):
        __invariant__ = test_invariant
    assert len(TestInvariantClass._checked_invariants) == 1

    # Test with multiple invariants
    def test_invariant2(x):
        return x < 100, "Value must be less than 100"

    class TestMultiInvariantClass(metaclass=_CheckedMapTypeMeta):
        __invariant__ = [test_invariant, test_invariant2]
    assert len(TestMultiInvariantClass._checked_invariants) == 2

    # Test with custom serializer
    def custom_serializer(self, _, key, value):
        return str(key), str(value)

    class TestSerializerClass(metaclass=_CheckedMapTypeMeta):
        __serializer__ = custom_serializer
    assert TestSerializerClass.__serializer__ == custom_serializer

    # Test with default serializer
    class TestDefaultSerializerClass(metaclass=_CheckedMapTypeMeta):
        pass
    assert TestDefaultSerializerClass.__serializer__ is not None

    # Test with slots
    class TestSlotsClass(metaclass=_CheckedMapTypeMeta):
        pass
    assert TestSlotsClass.__slots__ == ()


# LLM-generated content at query #2
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean
    def simple_invariant(x):
        return x > 0, "Value must be positive"

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(5) == (True, ())
    assert wrapped_simple(-1) == (False, ("Value must be positive",))

    # Test with an invariant that returns multiple test results
    def multi_invariant(x):
        return (x > 0, "Positive"), (x % 2 == 0, "Even")

    wrapped_multi = wrap_invariant(multi_invariant)
    assert wrapped_multi(4) == (True, ())
    assert wrapped_multi(3) == (False, ("Even",))
    assert wrapped_multi(-2) == (False, ("Positive", "Even"))
    assert wrapped_multi(-1) == (False, ("Positive", "Even"))

    # Test with an invariant that returns a single False with no message
    def bool_invariant(x):
        return False, None

    wrapped_bool = wrap_invariant(bool_invariant)
    assert wrapped_bool(10) == (False, (None,))

    # Test with an invariant that returns a single True with no message
    def true_invariant(x):
        return True, None

    wrapped_true = wrap_invariant(true_invariant)
    assert wrapped_true(10) == (True, ())


# LLM-generated content at query #3
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with a single type
    assert maybe_parse_user_type(int) == [int]

    # Test with a single string
    assert maybe_parse_user_type("str") == ["str"]

    # Test with an Enum type
    class TestEnum(Enum):
        A = 1
        B = 2
    assert maybe_parse_user_type(TestEnum) == [TestEnum]

    # Test with a list of types
    assert maybe_parse_user_type([int, str]) == [int, str]

    # Test with a tuple of types
    assert maybe_parse_user_type((int, str)) == [int, str]

    # Test with a list of strings
    assert maybe_parse_user_type(["int", "str"]) == ["int", "str"]

    # Test with a nested iterable
    assert maybe_parse_user_type([int, [str, float]]) == [int, str, float]

    # Test with an invalid type (should raise TypeError)
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)

    # Test with a non-preserved iterable type
    assert maybe_parse_user_type(list) == [list]


# LLM-generated content at query #4
#--------------------------

```python
def test_get_type():
    # Test with a built-in type
    assert get_type(int) == int

    # Test with a string representing a built-in type
    assert get_type('builtins.int') == int

    # Test with a custom class
    class CustomClass:
        pass

    assert get_type(CustomClass) == CustomClass

    # Test with a string representing a custom class
    assert get_type('__main__.CustomClass') == CustomClass

    # Test with a string representing a class from a module
    assert get_type('enum.Enum') == Enum


# LLM-generated content at query #5
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with a single type
    assert maybe_parse_user_type(int) == [int]

    # Test with an Enum type (preserved iterable)
    class TestEnum(Enum):
        A = 1
        B = 2
    assert maybe_parse_user_type(TestEnum) == [TestEnum]

    # Test with a string
    assert maybe_parse_user_type("str_type") == ["str_type"]

    # Test with a non-iterable type
    assert maybe_parse_user_type(float) == [float]

    # Test with an iterable of types
    assert maybe_parse_user_type([int, str]) == [int, str]

    # Test with nested iterables
    assert maybe_parse_user_type([int, [str, float]]) == [int, str, float]

    # Test with a string in an iterable
    assert maybe_parse_user_type(["str_type", int]) == ["str_type", int]

    # Test with an invalid type (should raise TypeError)
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)

    # Test with a non-preserved iterable type (e.g., list)
    with pytest.raises(TypeError):
        maybe_parse_user_type([])


# LLM-generated content at query #6
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test with empty dict
    class TestMap1(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    m1 = TestMap1()
    assert isinstance(m1, CheckedPMap)
    assert len(m1) == 0

    # Test with initial dict
    class TestMap2(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    m2 = TestMap2({1: "a", 2: "b"})
    assert isinstance(m2, CheckedPMap)
    assert len(m2) == 2
    assert m2[1] == "a"
    assert m2[2] == "b"

    # Test with size parameter
    class TestMap3(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    m3 = TestMap3(size=0)
    assert isinstance(m3, CheckedPMap)
    assert len(m3) == 0

    # Test type checking for keys
    class TestMap4(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    with pytest.raises(CheckedKeyTypeError):
        TestMap4({"a": "b"})

    # Test type checking for values
    class TestMap5(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    with pytest.raises(CheckedValueTypeError):
        TestMap5({1: 2})

    # Test with invariant
    class TestMap6(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")
    m6 = TestMap6({1: 2, 3: 4})
    assert isinstance(m6, CheckedPMap)
    assert len(m6) == 2

    with pytest.raises(InvariantException):
        TestMap6({1: 0})

    # Test with nested CheckedType
    class InnerType(CheckedPVector):
        __type__ = int

    class TestMap7(CheckedPMap):
        __key_type__ = int
        __value_type__ = InnerType
    m7 = TestMap7({1: InnerType([1, 2, 3])})
    assert isinstance(m7, CheckedPMap)
    assert len(m7) == 1
    assert m7[1] == InnerType([1, 2, 3])

    # Test create with nested CheckedType
    m7_created = TestMap7.create({1: [1, 2, 3]})
    assert isinstance(m7_created, CheckedPMap)
    assert len(m7_created) == 1
    assert m7_created[1] == InnerType([1, 2, 3])


# LLM-generated content at query #7
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean
    def simple_invariant(x):
        return x > 0, "Positive"

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(5) == (True, "Positive")
    assert wrapped_simple(-1) == (False, "Positive")

    # Test with an invariant that returns multiple results
    def complex_invariant(x):
        return (x > 0, "Positive"), (x % 2 == 0, "Even")

    wrapped_complex = wrap_invariant(complex_invariant)
    assert wrapped_complex(4) == (True, ("Positive", "Even"))
    assert wrapped_complex(3) == (False, ("Positive",))
    assert wrapped_complex(-2) == (False, ("Even",))
    assert wrapped_complex(-1) == (False, ())

    # Test with an invariant that returns a single False result
    def false_invariant(x):
        return False, "Always False"

    wrapped_false = wrap_invariant(false_invariant)
    assert wrapped_false(10) == (False, "Always False")

    # Test with an invariant that returns a single True result
    def true_invariant(x):
        return True, "Always True"

    wrapped_true = wrap_invariant(true_invariant)
    assert wrapped_true(10) == (True, "Always True")


# LLM-generated content at query #8
#--------------------------

```python
def test_CheckedPMap_serialize():
    class TestCheckedPMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    # Test with simple types
    cmap = TestCheckedPMap({1: "a", 2: "b"})
    assert cmap.serialize() == {1: "a", 2: "b"}

    # Test with nested CheckedType
    class InnerCheckedPMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int

    inner_map = InnerCheckedPMap({"x": 1, "y": 2})
    outer_map = TestCheckedPMap({1: inner_map, 2: "b"})
    assert outer_map.serialize() == {1: {"x": 1, "y": 2}, 2: "b"}

    # Test with custom serializer
    def custom_serializer(format, key, value):
        return str(key), str(value).upper()

    class CustomSerializedMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
        __serializer__ = custom_serializer

    custom_map = CustomSerializedMap({1: "a", 2: "b"})
    assert custom_map.serialize() == {"1": "A", "2": "B"}

    # Test empty map
    empty_map = TestCheckedPMap()
    assert empty_map.serialize() == {}


# LLM-generated content at query #9
#--------------------------

```python
def test_get_type():
    # Test with a built-in type
    assert get_type(int) == int

    # Test with a custom type
    class CustomClass:
        pass
    assert get_type(CustomClass) == CustomClass

    # Test with a string representing a built-in type
    assert get_type('builtins.int') == int

    # Test with a string representing a custom type
    assert get_type('pytest.CustomClass') == CustomClass

    # Test with an invalid type string
    try:
        get_type('invalid.type')
        assert False, "Expected an exception"
    except (ImportError, AttributeError):
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_get_type():
    # Test with a direct type
    assert get_type(int) == int

    # Test with a string type
    assert get_type('builtins.int') == int

    # Test with a custom class
    class TestClass:
        pass
    assert get_type('__main__.TestClass') == TestClass

    # Test with a nested module class
    from collections import OrderedDict
    assert get_type('collections.OrderedDict') == OrderedDict

    # Test with invalid type string (should raise AttributeError)
    try:
        get_type('nonexistent.module.Class')
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test with invalid type string format (should raise ValueError)
    try:
        get_type('invalid.type.string')
        assert False, "Expected ValueError"
    except (ValueError, ImportError):
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: Invariant returns a single boolean result
    def simple_invariant(x):
        return x > 0, "Positive"

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(5) == (True, "Positive")
    assert wrapped_simple(-1) == (False, "Positive")

    # Test case 2: Invariant returns multiple results that need merging
    def complex_invariant(x):
        return [(x > 0, "Positive"), (x % 2 == 0, "Even")]

    wrapped_complex = wrap_invariant(complex_invariant)
    assert wrapped_complex(4) == (True, ())
    assert wrapped_complex(3) == (False, ("Even",))
    assert wrapped_complex(-2) == (False, ("Positive", "Even"))

    # Test case 3: Invariant returns mixed results (some boolean, some tuples)
    def mixed_invariant(x):
        return [(x > 0, "Positive"), (x < 10, "Single digit")]

    wrapped_mixed = wrap_invariant(mixed_invariant)
    assert wrapped_mixed(5) == (True, ())
    assert wrapped_mixed(15) == (False, ("Single digit",))
    assert wrapped_mixed(-3) == (False, ("Positive",))

    # Test case 4: Invariant returns empty result (edge case)
    def empty_invariant(x):
        return []

    wrapped_empty = wrap_invariant(empty_invariant)
    assert wrapped_empty(10) == (True, ())


# LLM-generated content at query #12
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a boolean
    def simple_invariant(x):
        return x > 0, "Positive"
    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(5) == (True, "Positive")
    assert wrapped_simple(-1) == (False, "Positive")

    # Test with an invariant that returns multiple results
    def multi_invariant(x):
        return (x > 0, "Positive"), (x % 2 == 0, "Even")
    wrapped_multi = wrap_invariant(multi_invariant)
    assert wrapped_multi(4) == (True, ("Positive", "Even"))
    assert wrapped_multi(3) == (False, ("Positive",))
    assert wrapped_multi(-2) == (False, ("Even",))

    # Test with an invariant that returns a single non-boolean result
    def single_result_invariant(x):
        return (True, "Always True"), (x == 0, "Zero")
    wrapped_single = wrap_invariant(single_result_invariant)
    assert wrapped_single(0) == (True, ("Always True", "Zero"))
    assert wrapped_single(1) == (False, ("Always True",))

    # Test with an invariant that returns an empty result
    def empty_invariant(x):
        return ()
    wrapped_empty = wrap_invariant(empty_invariant)
    assert wrapped_empty(5) == (True, ())


# LLM-generated content at query #13
#--------------------------

```python
def test__CheckedMapTypeMeta___new__():
    class TestMetaClass(metaclass=_CheckedMapTypeMeta):
        __key_type__ = int
        __value_type__ = str
        __invariant__ = lambda k, v: (len(v) > 0, "Empty value")

    assert hasattr(TestMetaClass, '_checked_key_types')
    assert TestMetaClass._checked_key_types == [int]

    assert hasattr(TestMetaClass, '_checked_value_types')
    assert TestMetaClass._checked_value_types == [str]

    assert hasattr(TestMetaClass, '_checked_invariants')
    assert len(TestMetaClass._checked_invariants) == 1

    assert hasattr(TestMetaClass, '__serializer__')
    assert callable(TestMetaClass.__serializer__)

    assert hasattr(TestMetaClass, '__slots__')
    assert TestMetaClass.__slots__ == ()


# LLM-generated content at query #14
#--------------------------

```python
def test_get_type():
    # Test with a built-in type
    assert get_type(int) == int

    # Test with a custom class
    class CustomClass:
        pass
    assert get_type(CustomClass) == CustomClass

    # Test with a string representing a built-in type
    assert get_type('builtins.int') == int

    # Test with a string representing a custom class
    assert get_type('collections.abc.Iterable') == Iterable

    # Test with a string representing a non-existent module
    try:
        get_type('nonexistent.module.Class')
        assert False, "Expected ImportError"
    except ImportError:
        pass

    # Test with a string representing a non-existent class
    try:
        get_type('builtins.NonexistentClass')
        assert False, "Expected AttributeError"
    except AttributeError:
        pass


# LLM-generated content at query #15
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean
    def simple_invariant(x):
        return x > 0, "Value must be positive"

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(5) == (True, ())
    assert wrapped_simple(-1) == (False, ("Value must be positive",))

    # Test with an invariant that returns multiple test results
    def complex_invariant(x):
        return (x > 0, "Value must be positive"), (x % 2 == 0, "Value must be even")

    wrapped_complex = wrap_invariant(complex_invariant)
    assert wrapped_complex(4) == (True, ())
    assert wrapped_complex(5) == (False, ("Value must be even",))
    assert wrapped_complex(-1) == (False, ("Value must be positive", "Value must be even"))

    # Test with an invariant that returns a single tuple (not multiple results)
    def single_tuple_invariant(x):
        return (x > 0, "Value must be positive")

    wrapped_single_tuple = wrap_invariant(single_tuple_invariant)
    assert wrapped_single_tuple(5) == (True, ())
    assert wrapped_single_tuple(-1) == (False, ("Value must be positive",))


# LLM-generated content at query #16
#--------------------------

```python
def test_store_invariants():
    # Test basic invariant storage
    class TestClass1:
        __invariants__ = lambda self: (True, "test")

    dct = {}
    bases = (TestClass1,)
    store_invariants(dct, bases, '__stored_invariants__', '__invariants__')
    assert '__stored_invariants__' in dct
    assert len(dct['__stored_invariants__']) == 1
    assert callable(dct['__stored_invariants__'][0])

    # Test multiple invariants
    class TestClass2:
        __invariants__ = lambda self: (True, "test2")

    dct = {}
    bases = (TestClass1, TestClass2)
    store_invariants(dct, bases, '__stored_invariants__', '__invariants__')
    assert len(dct['__stored_invariants__']) == 2

    # Test inherited invariants
    class BaseClass:
        __invariants__ = lambda self: (True, "base")

    class DerivedClass(BaseClass):
        pass

    dct = {}
    bases = (DerivedClass,)
    store_invariants(dct, bases, '__stored_invariants__', '__invariants__')
    assert len(dct['__stored_invariants__']) == 1

    # Test non-callable invariant raises TypeError
    class BadClass:
        __invariants__ = "not callable"

    dct = {}
    bases = (BadClass,)
    try:
        store_invariants(dct, bases, '__stored_invariants__', '__invariants__')
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test invariant wrapping
    def complex_invariant(self):
        return [(True, "test1"), (False, "test2")]

    class ComplexClass:
        __invariants__ = complex_invariant

    dct = {}
    bases = (ComplexClass,)
    store_invariants(dct, bases, '__stored_invariants__', '__invariants__')
    wrapped = dct['__stored_invariants__'][0]
    result = wrapped(None)
    assert result == (False, ("test2",))


# LLM-generated content at query #17
#--------------------------

```python
def test__CheckedMapTypeMeta___new__():
    # Test basic creation
    class TestClass(metaclass=_CheckedMapTypeMeta):
        pass
    assert hasattr(TestClass, '_checked_key_types')
    assert hasattr(TestClass, '_checked_value_types')
    assert hasattr(TestClass, '_checked_invariants')
    assert hasattr(TestClass, '__serializer__')
    assert hasattr(TestClass, '__slots__')

    # Test type storage
    class TestClassWithTypes(metaclass=_CheckedMapTypeMeta):
        __key_type__ = int
        __value_type__ = str
    assert TestClassWithTypes._checked_key_types == [int]
    assert TestClassWithTypes._checked_value_types == [str]

    # Test multiple types
    class TestClassWithMultipleTypes(metaclass=_CheckedMapTypeMeta):
        __key_type__ = (int, str)
        __value_type__ = (float, bool)
    assert TestClassWithMultipleTypes._checked_key_types == [int, str]
    assert TestClassWithMultipleTypes._checked_value_types == [float, bool]

    # Test invariant storage
    def test_invariant(x):
        return True, "Test"

    class TestClassWithInvariant(metaclass=_CheckedMapTypeMeta):
        __invariant__ = test_invariant
    assert len(TestClassWithInvariant._checked_invariants) == 1

    # Test inherited types and invariants
    class BaseClass(metaclass=_CheckedMapTypeMeta):
        __key_type__ = int
        __value_type__ = str
        __invariant__ = test_invariant

    class DerivedClass(BaseClass):
        pass

    assert DerivedClass._checked_key_types == [int]
    assert DerivedClass._checked_value_types == [str]
    assert len(DerivedClass._checked_invariants) == 1

    # Test default serializer
    class TestClassWithSerializer(metaclass=_CheckedMapTypeMeta):
        pass
    assert callable(TestClassWithSerializer.__serializer__)

    # Test slots
    class TestClassWithSlots(metaclass=_CheckedMapTypeMeta):
        pass
    assert TestClassWithSlots.__slots__ == ()


# LLM-generated content at query #18
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with a single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]

    # Test with a single string
    assert maybe_parse_user_type("test") == ["test"]

    # Test with an iterable of types
    assert maybe_parse_user_type([int, str]) == [int, str]
    assert maybe_parse_user_type((int, str)) == [int, str]

    # Test with an iterable of strings
    assert maybe_parse_user_type(["test1", "test2"]) == ["test1", "test2"]

    # Test with a preserved iterable type (Enum)
    class TestEnum(Enum):
        A = 1
        B = 2

    assert maybe_parse_user_type(TestEnum) == [TestEnum]

    # Test with nested iterables
    assert maybe_parse_user_type([int, [str, "test"]]) == [int, str, "test"]

    # Test with invalid input (should raise TypeError)
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)

    with pytest.raises(TypeError):
        maybe_parse_user_type(None)


# LLM-generated content at query #19
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test with empty initial dict
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    empty_map = TestMap()
    assert len(empty_map) == 0

    # Test with initial dict
    initial_data = {1: "a", 2: "b"}
    test_map = TestMap(initial_data)
    assert len(test_map) == 2
    assert test_map[1] == "a"
    assert test_map[2] == "b"

    # Test with size parameter
    size_map = TestMap({}, size=10)
    assert len(size_map) == 0  # Size is just for internal use

    # Test with invalid key type
    with pytest.raises(CheckedKeyTypeError):
        TestMap({"invalid": "value"})

    # Test with invalid value type
    with pytest.raises(CheckedValueTypeError):
        TestMap({1: 123})

    # Test with valid types
    valid_map = TestMap({1: "one", 2: "two"})
    assert valid_map[1] == "one"
    assert valid_map[2] == "two"

    # Test with invariant
    class InvariantMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k < v, "Key must be less than value")

    valid_invariant_map = InvariantMap({1: 2, 3: 4})
    assert valid_invariant_map[1] == 2

    with pytest.raises(InvariantException):
        InvariantMap({5: 3})


# LLM-generated content at query #20
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test with empty dict
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    empty_map = TestMap()
    assert isinstance(empty_map, CheckedPMap)
    assert len(empty_map) == 0

    # Test with initial dict
    initial_data = {1: "a", 2: "b"}
    map_with_data = TestMap(initial_data)
    assert isinstance(map_with_data, CheckedPMap)
    assert len(map_with_data) == 2
    assert map_with_data[1] == "a"
    assert map_with_data[2] == "b"

    # Test with size parameter
    map_with_size = TestMap(size=10)
    assert isinstance(map_with_size, CheckedPMap)
    assert len(map_with_size) == 0

    # Test with both initial and size (size should be ignored)
    map_with_both = TestMap(initial_data, size=10)
    assert isinstance(map_with_both, CheckedPMap)
    assert len(map_with_both) == 2
    assert map_with_both[1] == "a"
    assert map_with_both[2] == "b"

    # Test type checking for keys
    try:
        TestMap({"not_int": "value"})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

    # Test type checking for values
    try:
        TestMap({1: 123})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass

    # Test with correct types but wrong invariant
    class InvariantMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k == v, "Key must equal value")

    try:
        InvariantMap({1: 2})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with correct types and correct invariant
    correct_map = InvariantMap({1: 1, 2: 2})
    assert isinstance(correct_map, CheckedPMap)
    assert len(correct_map) == 2
    assert correct_map[1] == 1
    assert correct_map[2] == 2


# LLM-generated content at query #21
#--------------------------

```python
def test_store_invariants():
    # Test basic invariant storage
    class TestClass1:
        pass

    dct = {}
    bases = (TestClass1,)
    store_invariants(dct, bases, 'invariants', 'invariants')

    assert 'invariants' in dct
    assert dct['invariants'] == ()

    # Test with a single invariant
    def test_inv1():
        return True, "Test1"

    class TestClass2:
        invariants = test_inv1

    dct = {}
    bases = (TestClass2,)
    store_invariants(dct, bases, 'invariants', 'invariants')

    assert 'invariants' in dct
    assert len(dct['invariants']) == 1
    assert callable(dct['invariants'][0])

    # Test with multiple invariants
    def test_inv2():
        return True, "Test2"

    class TestClass3:
        invariants = test_inv2

    class TestClass4(TestClass2, TestClass3):
        pass

    dct = {}
    bases = (TestClass4,)
    store_invariants(dct, bases, 'invariants', 'invariants')

    assert 'invariants' in dct
    assert len(dct['invariants']) == 2
    assert all(callable(inv) for inv in dct['invariants'])

    # Test with non-callable invariant (should raise TypeError)
    class TestClass5:
        invariants = "not_callable"

    dct = {}
    bases = (TestClass5,)
    try:
        store_invariants(dct, bases, 'invariants', 'invariants')
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"

    # Test with inherited invariants
    class BaseClass:
        invariants = test_inv1

    class DerivedClass(BaseClass):
        pass

    dct = {}
    bases = (DerivedClass,)
    store_invariants(dct, bases, 'invariants', 'invariants')

    assert 'invariants' in dct
    assert len(dct['invariants']) == 1
    assert callable(dct['invariants'][0])

    # Test with multiple inheritance
    class BaseClass1:
        invariants = test_inv1

    class BaseClass2:
        invariants = test_inv2

    class MultiDerived(BaseClass1, BaseClass2):
        pass

    dct = {}
    bases = (MultiDerived,)
    store_invariants(dct, bases, 'invariants', 'invariants')

    assert 'invariants' in dct
    assert len(dct['invariants']) == 2
    assert all(callable(inv) for inv in dct['invariants'])

    # Test with diamond inheritance
    class TopBase:
        pass

    class LeftBase(TopBase):
        invariants = test_inv1

    class RightBase(TopBase):
        invariants = test_inv2

    class DiamondDerived(LeftBase, RightBase):
        pass

    dct = {}
    bases = (DiamondDerived,)
    store_invariants(dct, bases, 'invariants', 'invariants')

    assert 'invariants' in dct
    assert len(dct['invariants']) == 2
    assert all(callable(inv) for inv in dct['invariants'])


# LLM-generated content at query #22
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with a single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]

    # Test with a single string
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("str") == ["str"]

    # Test with an iterable of types
    assert maybe_parse_user_type([int, str]) == [int, str]
    assert maybe_parse_user_type((int, str)) == [int, str]

    # Test with an iterable of strings
    assert maybe_parse_user_type(["int", "str"]) == ["int", "str"]
    assert maybe_parse_user_type(("int", "str")) == ["int", "str"]

    # Test with a preserved iterable type (Enum)
    class TestEnum(Enum):
        A = 1
        B = 2

    assert maybe_parse_user_type(TestEnum) == [TestEnum]

    # Test with a nested iterable
    assert maybe_parse_user_type([int, [str, float]]) == [int, str, float]
    assert maybe_parse_user_type((int, (str, float))) == [int, str, float]

    # Test with invalid input
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)

    with pytest.raises(TypeError):
        maybe_parse_user_type(None)


# LLM-generated content at query #23
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean
    def simple_invariant(x):
        return x > 0, "Positive"

    wrapped = wrap_invariant(simple_invariant)
    assert wrapped(5) == (True, "Positive")
    assert wrapped(-1) == (False, "Positive")

    # Test with an invariant that returns multiple test results
    def complex_invariant(x):
        return (x > 0, "Positive"), (x % 2 == 0, "Even")

    wrapped = wrap_invariant(complex_invariant)
    assert wrapped(4) == (True, ("Positive", "Even"))
    assert wrapped(3) == (False, ("Positive",))
    assert wrapped(-1) == (False, tuple())

    # Test with an invariant that returns a single False
    def false_invariant(x):
        return False, "Always False"

    wrapped = wrap_invariant(false_invariant)
    assert wrapped(10) == (False, "Always False")

    # Test with an invariant that returns a single True
    def true_invariant(x):
        return True, "Always True"

    wrapped = wrap_invariant(true_invariant)
    assert wrapped(10) == (True, "Always True")


# LLM-generated content at query #24
#--------------------------

```python
def test__CheckedMapTypeMeta___new__():
    class TestClass(metaclass=_CheckedMapTypeMeta):
        __key_type__ = int
        __value_type__ = str
        __invariant__ = lambda x: (x > 0, "Non-positive")

    assert hasattr(TestClass, '_checked_key_types')
    assert hasattr(TestClass, '_checked_value_types')
    assert hasattr(TestClass, '_checked_invariants')
    assert TestClass._checked_key_types == [int]
    assert TestClass._checked_value_types == [str]
    assert len(TestClass._checked_invariants) == 1
    assert hasattr(TestClass, '__serializer__')
    assert TestClass.__slots__ == ()


# LLM-generated content at query #25
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with a single type
    assert maybe_parse_user_type(int) == [int]

    # Test with a preserved iterable type (Enum)
    class TestEnum(Enum):
        A = 1
        B = 2
    assert maybe_parse_user_type(TestEnum) == [TestEnum]

    # Test with a string
    assert maybe_parse_user_type("str_type") == ["str_type"]

    # Test with a tuple of types
    assert maybe_parse_user_type((int, str)) == [int, str]

    # Test with a nested iterable
    assert maybe_parse_user_type([(int, str), float]) == [int, str, float]

    # Test with an invalid type (should raise TypeError)
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)


# LLM-generated content at query #26
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean
    def simple_invariant(x):
        return x > 0, "Value must be positive"

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(5) == (True, ())
    assert wrapped_simple(-1) == (False, ("Value must be positive",))

    # Test with an invariant that returns multiple test results
    def complex_invariant(x):
        return (x > 0, "Value must be positive"), (x % 2 == 0, "Value must be even")

    wrapped_complex = wrap_invariant(complex_invariant)
    assert wrapped_complex(4) == (True, ())
    assert wrapped_complex(5) == (False, ("Value must be even",))
    assert wrapped_complex(-1) == (False, ("Value must be positive", "Value must be even"))

    # Test with an invariant that returns a single False with no message
    def no_message_invariant(x):
        return False, None

    wrapped_no_msg = wrap_invariant(no_message_invariant)
    assert wrapped_no_msg(10) == (False, (None,))


# LLM-generated content at query #27
#--------------------------

```python
def test_wrap_invariant():
    # Test with a single boolean return
    def single_bool_invariant():
        return True, "OK"
    wrapped = wrap_invariant(single_bool_invariant)
    assert wrapped() == (True, "OK")

    # Test with multiple results that need merging
    def multi_result_invariant():
        return [(True, "OK1"), (False, "ERROR1"), (True, "OK2")]
    wrapped = wrap_invariant(multi_result_invariant)
    assert wrapped() == (False, ("ERROR1",))

    # Test with all passing results
    def all_pass_invariant():
        return [(True, "OK1"), (True, "OK2")]
    wrapped = wrap_invariant(all_pass_invariant)
    assert wrapped() == (True, ())

    # Test with all failing results
    def all_fail_invariant():
        return [(False, "ERROR1"), (False, "ERROR2")]
    wrapped = wrap_invariant(all_fail_invariant)
    assert wrapped() == (False, ("ERROR1", "ERROR2"))

    # Test with empty results
    def empty_invariant():
        return []
    wrapped = wrap_invariant(empty_invariant)
    assert wrapped() == (True, ())


# LLM-generated content at query #28
#--------------------------

```python
def test_store_invariants():
    # Test basic invariant storage
    dct = {}
    bases = ()
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert dct['invariants'] == ()

    # Test single invariant
    def test_inv():
        return True, ()
    dct = {'invariant': test_inv}
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert len(dct['invariants']) == 1
    assert dct['invariants'][0] == wrap_invariant(test_inv)

    # Test multiple invariants
    def test_inv2():
        return True, ()
    dct = {'invariant': test_inv}
    bases = (type('Base', (), {'invariant': test_inv2}),)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert len(dct['invariants']) == 2
    assert dct['invariants'][0] == wrap_invariant(test_inv)
    assert dct['invariants'][1] == wrap_invariant(test_inv2)

    # Test inherited invariants
    class Base1:
        def invariant(self):
            return True, ()
    class Base2(Base1):
        pass
    dct = {}
    bases = (Base2,)
    store_invariants(dct, bases, 'invariants', 'invariant')
    assert len(dct['invariants']) == 1
    assert dct['invariants'][0] == wrap_invariant(Base1.invariant)

    # Test non-callable invariant raises TypeError
    dct = {'invariant': 'not_callable'}
    try:
        store_invariants(dct, bases, 'invariants', 'invariant')
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'


# LLM-generated content at query #29
#--------------------------

```python
def test_store_invariants():
    # Test basic invariant storage
    class A(CheckedType):
        __invariant__ = lambda self: (True, "OK")

    assert A.__invariant__ == ((True, "OK"),)

    # Test multiple invariants
    class B(CheckedType):
        __invariant__ = lambda self: (True, "OK")

    class C(B):
        __invariant__ = lambda self: (False, "Error")

    assert C.__invariant__ == ((True, "OK"), (False, "Error"))

    # Test non-callable invariant raises TypeError
    with pytest.raises(TypeError):
        class D(CheckedType):
            __invariant__ = "not callable"

    # Test invariant wrapping
    def multi_invariant(self):
        return [(True, "OK"), (False, "Error")]

    class E(CheckedType):
        __invariant__ = multi_invariant

    result = E.__invariant__[0](None)
    assert result == (False, ("Error",))

    # Test inheritance with no new invariants
    class F(B):
        pass

    assert F.__invariant__ == ((True, "OK"),)

    # Test multiple inheritance
    class G(CheckedType):
        __invariant__ = lambda self: (True, "G")

    class H(CheckedType):
        __invariant__ = lambda self: (True, "H")

    class I(G, H):
        pass

    assert I.__invariant__ == ((True, "G"), (True, "H"))


# LLM-generated content at query #30
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean
    def simple_invariant(x):
        return x > 0, "Positive"
    wrapped = wrap_invariant(simple_invariant)
    assert wrapped(5) == (True, "Positive")
    assert wrapped(-1) == (False, "Positive")

    # Test with an invariant that returns multiple results
    def multi_invariant(x):
        return (x > 0, "Positive"), (x % 2 == 0, "Even")
    wrapped = wrap_invariant(multi_invariant)
    assert wrapped(4) == (True, ("Positive", "Even"))
    assert wrapped(3) == (False, ("Positive",))
    assert wrapped(-2) == (False, ("Even",))
    assert wrapped(-1) == (False, ())

    # Test with an invariant that returns a single False result
    def false_invariant(x):
        return False, "Always False"
    wrapped = wrap_invariant(false_invariant)
    assert wrapped(100) == (False, "Always False")

    # Test with an invariant that returns a single True result
    def true_invariant(x):
        return True, "Always True"
    wrapped = wrap_invariant(true_invariant)
    assert wrapped(-100) == (True, "Always True")


