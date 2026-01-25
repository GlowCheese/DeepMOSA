####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_immutable():
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "Point(x=1, y=2)"

    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "Point(x=3, y=2)"

    # Test inheritance
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

    # Test frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    assert p.id_ == 17
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2.id_ == 17

    # Test frozen member error
    try:
        p.set(id_=18)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    # Test non-existent member error
    try:
        p.set(z=5)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'z' is not a member"

    # Test empty members
    Empty = immutable(name='Empty')
    e = Empty()
    assert repr(e) == "Empty()"

    # Test verbose flag
    immutable('x, y', name='Point', verbose=True)


# LLM-generated content at query #2
#--------------------------

```python
def test_immutable():
    # Test basic creation and usage
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "Point(x=1, y=2)"

    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "Point(x=3, y=2)"

    # Test inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    p = PositivePoint(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "PositivePoint(x=1, y=2)"

    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "PositivePoint(x=3, y=2)"

    try:
        p.set(y=-3)
        assert False, "Expected Exception not raised"
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!'

    # Test frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    assert p.x == 1
    assert p.y == 2
    assert p.id_ == 17
    assert repr(p) == "Point(x=1, y=2, id_=17)"

    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2.id_ == 17
    assert repr(p2) == "Point(x=3, y=2, id_=17)"

    try:
        p.set(id_=18)
        assert False, "Expected AttributeError not raised"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    # Test invalid member
    try:
        p.set(z=5)
        assert False, "Expected AttributeError not raised"
    except AttributeError as e:
        assert str(e) == "'z' is not a member"

    # Test empty set
    p3 = p.set()
    assert p3 is p

    # Test verbose mode
    try:
        immutable('x, y', name='Point', verbose=True)
    except Exception:
        assert False, "Verbose mode should not raise an exception"


# LLM-generated content at query #3
#--------------------------

```python
def test_immutable():
    # Test basic creation and attribute access
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == 'Point(x=3, y=2)'

    # Test inheritance
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

    # Test frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    assert p.id_ == 17
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2.id_ == 17

    try:
        p.set(id_=18)
        assert False, "Expected AttributeError for frozen member"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    # Test invalid member access
    try:
        p.set(z=5)
        assert False, "Expected AttributeError for invalid member"
    except AttributeError as e:
        assert str(e) == "'z' is not a member"

    # Test empty members
    Empty = immutable(name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'

    # Test verbose mode (just ensure it doesn't crash)
    try:
        immutable('x, y', name='Point', verbose=True)
    except Exception as e:
        assert False, f"Verbose mode raised unexpected exception: {e}"


# LLM-generated content at query #4
#--------------------------

```python
def test_immutable():
    # Test basic creation and attribute access
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "Point(x=1, y=2)"

    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "Point(x=3, y=2)"

    # Test inheritance
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

    # Test frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2.id_ == 17

    try:
        p.set(id_=18)
        assert False, "Expected AttributeError for frozen member"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    # Test invalid member set
    try:
        p.set(z=5)
        assert False, "Expected AttributeError for invalid member"
    except AttributeError as e:
        assert "'z' is not a member" in str(e)


# LLM-generated content at query #5
#--------------------------

```python
def test_immutable():
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "Point(x=1, y=2)"

    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "Point(x=3, y=2)"

    # Test inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    p = PositivePoint(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "PositivePoint(x=1, y=2)"

    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "PositivePoint(x=3, y=2)"

    try:
        p.set(y=-3)
        assert False, "Expected Exception not raised"
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!'

    # Test frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    assert p.x == 1
    assert p.y == 2
    assert p.id_ == 17
    assert repr(p) == "Point(x=1, y=2, id_=17)"

    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2.id_ == 17
    assert repr(p2) == "Point(x=3, y=2, id_=17)"

    try:
        p.set(id_=18)
        assert False, "Expected AttributeError not raised"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    # Test invalid member
    try:
        p.set(z=3)
        assert False, "Expected AttributeError not raised"
    except AttributeError as e:
        assert str(e) == "'z' is not a member"


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_immutable():
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "Point(x=1, y=2)"

    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "Point(x=3, y=2)"

    # Test inheritance
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

    # Test frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    assert p.id_ == 17
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2.id_ == 17

    # Test setting frozen member raises error
    try:
        p.set(id_=18)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    # Test setting non-existent member raises error
    try:
        p.set(z=5)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'z' is not a member"

    # Test empty members
    Empty = immutable(name='Empty')
    e = Empty()
    assert repr(e) == "Empty()"

    # Test verbose flag
    try:
        immutable('x, y', name='Point', verbose=True)
    except Exception:
        assert False, "Verbose flag should not raise an exception"


# LLM-generated content at query #2
#--------------------------

```python
def test_immutable():
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "Point(x=1, y=2)"

    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "Point(x=3, y=2)"

    # Test frozen members
    PointWithID = immutable('x, y, id_', name='PointWithID')
    p3 = PointWithID(1, 2, id_=17)
    assert p3.id_ == 17
    p4 = p3.set(x=3)
    assert p4.id_ == 17
    try:
        p3.set(id_=18)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    # Test inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    p5 = PositivePoint(1, 2)
    assert p5.x == 1
    assert p5.y == 2
    p6 = p5.set(x=3)
    assert p6.x == 3
    assert p6.y == 2
    try:
        p5.set(y=-3)
        assert False, "Should have raised Exception"
    except Exception as e:
        assert str(e) == "Coordinates must be positive!"

    # Test invalid member
    try:
        p.set(z=5)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert str(e) == "'z' is not a member"

    # Test empty kwargs
    p7 = p.set()
    assert p7 is p


# LLM-generated content at query #3
#--------------------------

```python
def test_immutable():
    # Test basic immutable creation and usage
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "Point(x=1, y=2)"

    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "Point(x=3, y=2)"

    # Test inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    p = PositivePoint(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "PositivePoint(x=1, y=2)"

    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "PositivePoint(x=3, y=2)"

    try:
        p.set(y=-3)
        assert False, "Should have raised an exception"
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!'

    # Test frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    assert p.x == 1
    assert p.y == 2
    assert p.id_ == 17
    assert repr(p) == "Point(x=1, y=2, id_=17)"

    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2.id_ == 17
    assert repr(p2) == "Point(x=3, y=2, id_=17)"

    try:
        p.set(id_=18)
        assert False, "Should have raised an AttributeError"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    # Test invalid member
    try:
        p.set(z=5)
        assert False, "Should have raised an AttributeError"
    except AttributeError as e:
        assert str(e) == "'z' is not a member"

    # Test empty set call
    assert p.set() is p

    # Test verbose mode
    try:
        immutable('x, y', name='Point', verbose=True)
    except Exception:
        assert False, "verbose mode should not raise an exception"


# LLM-generated content at query #4
#--------------------------

```python
def test_immutable():
    # Test basic creation and attributes
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "Point(x=1, y=2)"

    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "Point(x=3, y=2)"

    # Test inheritance
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

    # Test frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    assert p.id_ == 17
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2.id_ == 17

    # Test frozen member error
    try:
        p.set(id_=18)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    # Test invalid member error
    try:
        p.set(z=5)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'z' is not a member"

    # Test empty members
    Empty = immutable(name='Empty')
    e = Empty()
    assert repr(e) == "Empty()"

    # Test verbose mode
    Point = immutable('x, y', name='Point', verbose=True)
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2


# LLM-generated content at query #5
#--------------------------

```python
def test_immutable():
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "Point(x=1, y=2)"

    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "Point(x=3, y=2)"

    # Test inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    p = PositivePoint(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == "PositivePoint(x=1, y=2)"

    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == "PositivePoint(x=3, y=2)"

    try:
        p.set(y=-3)
        assert False, "Expected Exception not raised"
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!'

    # Test frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    assert p.x == 1
    assert p.y == 2
    assert p.id_ == 17
    assert repr(p) == "Point(x=1, y=2, id_=17)"

    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2.id_ == 17
    assert repr(p2) == "Point(x=3, y=2, id_=17)"

    try:
        p.set(id_=18)
        assert False, "Expected AttributeError not raised"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    # Test invalid member
    try:
        p.set(z=5)
        assert False, "Expected AttributeError not raised"
    except AttributeError as e:
        assert str(e) == "'z' is not a member"

    # Test empty kwargs
    p3 = p.set()
    assert p3 is p


