####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_immutable_basic_usage():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "Point(x=1, y=2)"

def test_immutable_set_method():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "Point(x=3, y=2)"

def test_immutable_inheritance():
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    p = PositivePoint(1, 2)
    assert p.x == 1
    assert p.y == 2
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    try:
        p.set(y=-3)
        assert False, "Expected Exception"
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!'

def test_immutable_frozen_members():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    assert p.id_ == 17
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2.id_ == 17
    try:
        p.set(id_=18)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

def test_immutable_empty_members():
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == "Empty()"

def test_immutable_no_name():
    Point = immutable('x, y')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "Immutable(x=1, y=2)"

def test_immutable_invalid_member():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'z' is not a member"

def test_immutable_no_kwargs():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set()
    assert p2 is p


