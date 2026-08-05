####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_immutable_basic_functionality():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3)
    assert p.x == 1
    assert p.y == 2
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p) == 'Point(x=1, y=2)'
    assert repr(p2) == 'Point(x=3, y=2)'

def test_immutable_no_members():
    Empty = immutable('', name='Empty')
    e = Empty()
    e2 = e.set()
    assert e == e2
    assert repr(e) == 'Empty()'

def test_immutable_frozen_members():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.id_ == 17
    try:
        p.set(id_=18)
    except AttributeError as e:
        assert "Cannot set frozen members 'id_'" in str(e)

def test_immutable_invalid_member_setting():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=5)
    except AttributeError as e:
        assert "'z' is not a member" in str(e)

def test_immutable_inheritance():
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
    assert repr(p2) == 'PositivePoint(x=3, y=2)'

def test_immutable_complex_members_string():
    Point = immutable('x,y,z', name='Point')
    p = Point(1, 2, 3)
    assert p.z == 3
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
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

def test_immutable_set_method():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p != p2

def test_immutable_set_multiple():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=10, y=20)
    assert p2.x == 10
    assert p2.y == 20

def test_immutable_invalid_member_set():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=3)
    except AttributeError as e:
        assert "'z' is not a member" in str(e)

def test_immutable_frozen_members():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.id_ == 17
    try:
        p.set(id_=18)
    except AttributeError as e:
        assert "Cannot set frozen members 'id_'" in str(e)

def test_immutable_empty_members():
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    e2 = e.set()
    assert e == e2

def test_immutable_inheritance():
    Base = immutable('x, y', name='Base')
    class Derived(Base):
        __slots__ = tuple()
        def __new__(cls, x, y):
            return super().__new__(cls, x, y)
    
    d = Derived(5, 10)
    d2 = d.set(x=1)
    assert d2.x == 1
    assert d2.y == 10
    assert repr(d2) == 'Derived(x=1, y=10)'

def test_immutable_no_kwargs_returns_self():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.set() is p
```


