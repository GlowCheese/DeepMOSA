####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_immutable_basic_creation():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

def test_immutable_set_single_field():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3)
    assert p.x == 1
    assert p.y == 2
    assert p2.x == 3
    assert p2.y == 2

def test_immutable_set_multiple_fields():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3, y=4)
    assert p2.x == 3
    assert p2.y == 4

def test_immutable_set_no_changes():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set()
    assert p2 is p

def test_immutable_set_invalid_member():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=5)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "is not a member" in str(e)

def test_immutable_frozen_member():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    assert p.id_ == 17
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.id_ == 17

def test_immutable_frozen_member_cannot_be_set():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    try:
        p.set(id_=18)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Cannot set frozen members" in str(e)

def test_immutable_with_comma_separated_string():
    Point = immutable('x, y, z', name='Point')
    p = Point(1, 2, 3)
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3

def test_immutable_with_space_separated_string():
    Point = immutable('x y z', name='Point')
    p = Point(1, 2, 3)
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3

def test_immutable_with_list():
    Point = immutable(['x', 'y'], name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

def test_immutable_empty_members():
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'

def test_immutable_default_name():
    ImmClass = immutable('x, y')
    obj = ImmClass(1, 2)
    assert 'Immutable' in repr(obj)

def test_immutable_repr_uses_class_name():
    CustomPoint = immutable('x, y', name='CustomPoint')
    p = CustomPoint(5, 10)
    assert 'CustomPoint' in repr(p)
    assert 'ImmutableBase' not in repr(p)

def test_immutable_single_member():
    Value = immutable('val', name='Value')
    v = Value(42)
    assert v.val == 42
    v2 = v.set(val=100)
    assert v2.val == 100

def test_immutable_multiple_frozen_members():
    Data = immutable('x, id_, y, uuid_', name='Data')
    d = Data(1, 'id1', 2, 'uuid1')
    d2 = d.set(x=10)
    assert d2.x == 10
    try:
        d.set(id_='id2')
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Cannot set frozen members" in str(e)

def test_immutable_set_with_frozen_and_normal_member():
    Data = immutable('x, id_', name='Data')
    d = Data(1, 'id1')
    d2 = d.set(x=5)
    assert d2.x == 5
    assert d2.id_ == 'id1'

def test_immutable_verbose_mode():
    Point = immutable('x, y', name='Point', verbose=False)
    p = Point(1, 2)
    assert p.x == 1


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_immutable_basic_creation():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

def test_immutable_string_parsing_with_commas():
    Point = immutable('x, y, z', name='Point')
    p = Point(1, 2, 3)
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3

def test_immutable_string_parsing_without_commas():
    Point = immutable('x y z', name='Point')
    p = Point(1, 2, 3)
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3

def test_immutable_list_members():
    Point = immutable(['x', 'y'], name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

def test_immutable_set_method():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3)
    assert p.x == 1
    assert p.y == 2
    assert p2.x == 3
    assert p2.y == 2

def test_immutable_set_method_multiple_fields():
    Point = immutable('x, y, z', name='Point')
    p = Point(1, 2, 3)
    p2 = p.set(x=10, z=30)
    assert p2.x == 10
    assert p2.y == 2
    assert p2.z == 30

def test_immutable_set_method_no_changes():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set()
    assert p is p2

def test_immutable_set_nonexistent_member():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=3)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "is not a member" in str(e)

def test_immutable_frozen_member():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2.id_ == 17

def test_immutable_frozen_member_cannot_be_modified():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    try:
        p.set(id_=18)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Cannot set frozen members" in str(e)

def test_immutable_multiple_frozen_members():
    Data = immutable('a, b_, c, d_', name='Data')
    d = Data(1, 2, 3, 4)
    d2 = d.set(a=10, c=30)
    assert d2.a == 10
    assert d2.b_ == 2
    assert d2.c == 30
    assert d2.d_ == 4

def test_immutable_multiple_frozen_members_error():
    Data = immutable('a, b_, c, d_', name='Data')
    d = Data(1, 2, 3, 4)
    try:
        d.set(b_=20)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Cannot set frozen members" in str(e)

def test_immutable_repr():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert 'Point' in repr(p)
    assert 'x=1' in repr(p)
    assert 'y=2' in repr(p)

def test_immutable_empty_members():
    Empty = immutable('', name='Empty')
    e = Empty()
    assert e is not None

def test_immutable_single_member():
    Single = immutable('value', name='Single')
    s = Single(42)
    assert s.value == 42
    s2 = s.set(value=100)
    assert s2.value == 100

def test_immutable_as_base_class():
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p = PositivePoint(1, 2)
    assert p.x == 1
    assert p.y == 2

def test_immutable_as_base_class_validation():
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    try:
        p = PositivePoint(-1, 2)
        assert False, "Should raise Exception"
    except Exception as e:
        assert 'Coordinates must be positive!' in str(e)

def test_immutable_as_base_class_set_with_validation():
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p = PositivePoint(1, 2)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2

def test_immutable_as_base_class_set_invalid():
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p = PositivePoint(1, 2)
    try:
        p2 = p.set(y=-3)
        assert False, "Should raise Exception"
    except Exception as e:
        assert 'Coordinates must be positive!' in str(e)

def test_immutable_verbose_mode():
    Point = immutable('x, y', name='Point', verbose=False)
    p = Point(1, 2)
    assert p.x == 1

def test_immutable_namedtuple_behavior():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    x, y = p
    assert x == 1
    assert y == 2

def test_immutable_equality():
    Point = immutable('x, y', name='Point')
    p1 = Point(1, 2)
    p2 = Point(1, 2)
    p3 = Point(2, 3)
    assert p1 == p2
    assert p1 != p3

def test_immutable_hashable():
    Point = immutable('x, y', name='Point')
    p1 = Point(1, 2)
    p2 = Point(1, 2)
    assert hash(p1) == hash(p2)
    s = {p1, p2}
    assert len(s) == 1


