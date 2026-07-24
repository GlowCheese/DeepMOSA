####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_immutable():
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

    # Test set method (returning new instance)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p != p2
    assert repr(p2) == 'Point(x=3, y=2)'

    # Test set method (no changes)
    p3 = p.set()
    assert p3 is p

    # Test invalid member update
    with pytest.raises(AttributeError) as excinfo:
        p.set(z=10)
    assert "'z' is not a member" in str(excinfo.value)

    # Test frozen members (trailing underscore)
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p_id = PointWithId(1, 2, id_=17)
    
    # Updating non-frozen member works
    p_id_updated = p_id.set(x=10)
    assert p_id_updated.x == 10
    assert p_id_updated.id_ == 17
    
    # Updating frozen member raises error
    with pytest.raises(AttributeError) as excinfo:
        p_id.set(id_=18)
    assert "Cannot set frozen members 'id_'" in str(excinfo.value)

    # Test inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    pp1 = PositivePoint(1, 2)
    assert pp1.x == 1
    assert repr(pp1) == 'PositivePoint(x=1, y=2)'
    
    pp2 = pp1.set(y=5)
    assert pp2.y == 5
    assert repr(pp2) == 'PositivePoint(x=1, y=5)'

    with pytest.raises(Exception, match='Coordinates must be positive!'):
        PositivePoint(-1, 2)

    # Test empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    assert e.set(x=1) == e # Should return self if no kwargs or invalid logic

    # Test comma-separated and space-separated string inputs
    PointAlt = immutable('x y z', name='PointAlt')
    pa = PointAlt(1, 2, 3)
    assert pa.z == 3
    assert repr(pa) == 'PointAlt(x=1, y=2, z=3)'
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_immutable():
    # Test basic functionality (Standalone use)
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'
    
    # Test set method (creating new instance)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2 is not p
    assert repr(p2) == 'Point(x=3, y=2)'
    
    # Test set with no changes
    p3 = p.set()
    assert p3 is p

    # Test error on non-existent member
    with pytest.raises(AttributeError) as excinfo:
        p.set(z=10)
    assert "'z' is not a member" in str(excinfo.value)

    # Test inheritance and custom logic
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    pp1 = PositivePoint(1, 2)
    assert pp1.x == 1
    assert repr(pp1) == 'PositivePoint(x=1, y=2)'
    
    pp2 = pp1.set(x=5)
    assert pp2.x == 5
    assert pp2.y == 2
    
    with pytest.raises(Exception, match='Coordinates must be positive!'):
        PositivePoint(-1, 2)

    # Test frozen members (trailing underscore)
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p_id = PointWithId(1, 2, id_=17)
    assert p_id.id_ == 17
    
    # Allowed to update non-frozen members
    p_id_updated = p_id.set(x=10)
    assert p_id_updated.x == 10
    assert p_id_updated.id_ == 17
    
    # Disallowed to update frozen members
    with pytest.raises(AttributeError) as excinfo:
        p_id.set(id_=18)
    assert 'Cannot set frozen members \'id_\'' in str(excinfo.value)

    # Test empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    assert e.set(foo=1) is e # set() returns self if no kwargs or if invalid is caught early

    # Test space separated string input
    SpacePoint = immutable('a b c', name='SpacePoint')
    sp = SpacePoint(1, 2, 3)
    assert sp.a == 1
    assert sp.b == 2
    assert sp.c == 3
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_immutable():
    # Test basic functionality (Standalone use)
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

    # Test .set() method for basic updates
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == 'Point(x=3, y=2)'
    assert p != p2

    # Test .set() with no arguments
    p3 = p.set()
    assert p3 == p

    # Test error on non-existent member
    with pytest.raises(AttributeError) as excinfo:
        p.set(z=10)
    assert "'z' is not a member" in str(excinfo.value)

    # Test inheritance and custom logic
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    pp = PositivePoint(1, 2)
    assert pp.x == 1
    assert pp.y == 2
    assert repr(pp) == 'PositivePoint(x=1, y=2)'
    
    pp2 = pp.set(x=5)
    assert pp2.x == 5
    assert pp2.y == 2
    assert isinstance(pp2, PositivePoint)

    with pytest.raises(Exception, match='Coordinates must be positive!'):
        PositivePoint(-1, 2)

    # Test frozen members (trailing underscore)
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p_id = PointWithId(1, 2, id_=17)
    assert p_id.id_ == 17
    
    # Updating non-frozen member
    p_id_updated = p_id.set(x=10)
    assert p_id_updated.x == 10
    assert p_id_updated.id_ == 17

    # Updating frozen member should raise AttributeError
    with pytest.raises(AttributeError) as excinfo:
        p_id.set(id_=18)
    assert 'Cannot set frozen members \'id_\'' in str(excinfo.value)

    # Test empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    assert e.set(any=1) == e # Should return self if no kwargs or invalid logic handles it

    # Test verbose flag (should not crash)
    PointVerbose = immutable('x', name='PointVerbose', verbose=True)
    pv = PointVerbose(1)
    assert pv.x == 1
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_immutable():
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

    # Test .set() for existing members
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == 'Point(x=3, y=2)'
    assert p != p2

    # Test .set() with no changes
    p3 = p.set()
    assert p3 == p

    # Test .set() with invalid member
    with pytest.raises(AttributeError) as excinfo:
        p.set(z=10)
    assert "'z'" in str(excinfo.value)

    # Test inheritance and custom logic
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    pp = PositivePoint(1, 2)
    assert pp.x == 1
    pp_updated = pp.set(x=5)
    assert pp_updated.x == 5
    assert isinstance(pp_updated, PositivePoint)

    with pytest.raises(Exception, match='Coordinates must be positive!'):
        PositivePoint(-1, 2)

    # Test frozen members (trailing underscore)
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p_id = PointWithId(1, 2, id_=17)
    assert p_id.id_ == 17
    
    # Updating non-frozen member works
    p_id_new = p_id.set(x=10)
    assert p_id_new.x == 10
    assert p_id_new.id_ == 17

    # Updating frozen member raises AttributeError
    with pytest.raises(AttributeError) as excinfo:
        p_id.set(id_=18)
    assert 'Cannot set frozen members \'id_\'' in str(excinfo.value)

    # Test empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    assert e.set(x=1) == e

    # Test comma-separated string parsing
    Complex = immutable('a, b, c', name='Complex')
    c = Complex(1, 2, 3)
    assert c.a == 1 and c.b == 2 and c.c == 3

    # Test verbose mode (should not crash)
    _ = immutable('x', name='VerboseTest', verbose=True)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_immutable():
    # Test basic functionality: standalone namedtuple-like class
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

    # Test .set() method for updates
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == 'Point(x=3, y=2)'
    assert p != p2  # Ensure it returns a new instance

    # Test .set() with no arguments returns self
    p3 = p.set()
    assert p3 is p

    # Test .set() with invalid field
    with pytest.raises(AttributeError) as excinfo:
        p.set(z=10)
    assert "'z' is not a member" in str(excinfo.value)

    # Test frozen members (ending with underscore)
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p_id = PointWithId(1, 2, id_=17)
    assert p_id.id_ == 17
    
    # Updating non-frozen field works
    p_id_updated = p_id.set(x=10)
    assert p_id_updated.x == 10
    assert p_id_updated.id_ == 17

    # Updating frozen field raises AttributeError
    with pytest.raises(AttributeError) as excinfo:
        p_id.set(id_=18)
    assert "Cannot set frozen members 'id_'" in str(excinfo.value)

    # Test inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    pp1 = PositivePoint(1, 2)
    assert pp1.x == 1
    assert repr(pp1) == 'PositivePoint(x=1, y=2)'
    
    pp2 = pp1.set(y=5)
    assert pp2.y == 5
    assert isinstance(pp2, PositivePoint)

    with pytest.raises(Exception, match='Coordinates must be positive!'):
        PositivePoint(-1, 2)

    # Test empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    assert e.set() is e

    # Test space-separated members string
    SpaceSeparated = immutable('a b c', name='SpaceSeparated')
    ss = SpaceSeparastated(1, 2, 3)
    assert ss.a == 1
    assert ss.b == 2
    assert ss.c == 3

    # Test SyntaxError handling for invalid template generation
    # (Simulating an edge case where name might break template)
    with pytest.raises(SyntaxError):
        immutable('x', name='class InvalidName:')

```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_immutable():
    # Test basic standalone functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

    # Test .set() functionality
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == 'Point(x=3, y=2)'
    assert p != p2

    # Test .set() with no changes
    p3 = p.set()
    assert p3 == p

    # Test .set() with invalid attribute
    with pytest.raises(AttributeError) as excinfo:
        p.set(z=10)
    assert "'z'" in str(excinfo.value)

    # Test inheritance and custom logic
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    pp1 = PositivePoint(1, 2)
    assert pp1.x == 1
    assert pp1.y == 2
    
    pp2 = pp1.set(x=5)
    assert pp2.x == 5
    assert pp2.y == 2
    assert isinstance(pp2, PositivePoint)

    with pytest.raises(Exception, match='Coordinates must be positive!'):
        PositivePoint(-1, 2)

    # Test frozen members (trailing underscore)
    PointWithId = immutable('x, y, id_', name='PointWithId')
    pid = PointWithId(1, 2, id_=17)
    assert pid.id_ == 17
    
    # Updating non-frozen member
    pid2 = pid.set(x=10)
    assert pid2.x == 10
    assert pid2.id_ == 17

    # Updating frozen member should raise AttributeError
    with pytest.raises(AttributeError) as excinfo:
        pid.set(id_=18)
    assert 'Cannot set frozen members \'id_\'' in str(excinfo.value)

    # Test empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    assert e.set(x=1) == e # Should return self if no kwargs or invalid check fails gracefully

    # Test comma-separated string vs list-like string
    PointAlt = immutable('a b c', name='Alt')
    pa = PointAlt(1, 2, 3)
    assert pa.a == 1
    assert pa.c == 3
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_immutable():
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

    # Test set method (creating new instance)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p != p2
    assert repr(p2) == 'Point(x=3, y=2)'

    # Test set method (no changes)
    p3 = p.set()
    assert p3 == p

    # Test invalid member update
    with pytest.raises(AttributeError, match="'z' is not a member"):
        p.set(z=10)

    # Test frozen members (trailing underscore)
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p_id = PointWithId(1, 2, id_=17)
    assert p_id.id_ == 17
    
    # Updating non-frozen member
    p_id_updated = p_id.set(x=10)
    assert p_id_updated.x == 10
    assert p_id_updated.id_ == 17
    
    # Updating frozen member
    with pytest.raises(AttributeError, match="Cannot set frozen members 'id_'"):
        p_id.set(id_=18)

    # Test inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    pp = PositivePoint(1, 2)
    assert pp.x == 1
    assert pp.y == 2
    assert repr(pp) == 'PositivePoint(x=1, y=2)'
    
    pp_updated = pp.set(y=5)
    assert pp_updated.y == 5
    assert repr(pp_updated) == 'PositivePoint(x=1, y=5)'

    with pytest.raises(Exception, match='Coordinates must be positive!'):
        PositivePoint(-1, 2)

    # Test empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    assert e.set(x=1) == e # Should return self if no kwargs or invalid logic

    # Test different string formats for members
    PointSpace = immutable('x y z', name='PointSpace')
    ps = PointSpace(1, 2, 3)
    assert ps.z == 3
    assert repr(ps) == 'PointSpace(x=1, y=2, z=3)'
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_immutable():
    # Test basic functionality (standalone class)
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

    # Test .set() method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == 'Point(x=3, y=2)'
    assert p != p2  # Ensure it's a new instance

    # Test .set() with no arguments
    p3 = p.set()
    assert p3 == p

    # Test .set() with invalid field
    with pytest.raises(AttributeError, match="'z' is not a member"):
        p.set(z=10)

    # Test inheritance and custom logic
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    pp1 = PositivePoint(1, 2)
    assert pp1.x == 1
    assert pp1.y == 2
    
    pp2 = pp1.set(x=5)
    assert pp2.x == 5
    assert pp2.y == 2
    assert isinstance(pp2, PositivePoint)

    with pytest.raises(Exception, match='Coordinates must be positive!'):
        PositivePoint(-1, 2)

    # Test frozen members (trailing underscore)
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p_id = PointWithId(1, 2, id_=17)
    assert p_id.id_ == 17
    
    # Updating non-frozen member
    p_id_updated = p_id.set(x=10)
    assert p_id_updated.x == 10
    assert p_id_updated.id_ == 17

    # Updating frozen member should raise AttributeError
    with pytest.raises(AttributeError, match="Cannot set frozen members 'id_'"):
        p_id.set(id_=18)

    # Test empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    assert e.set(x=1) == e # Should return self if no kwargs or invalid (though set logic handles empty)

    # Test verbose mode (should not crash)
    PointVerbose = immutable('x', name='VerbosePoint', verbose=True)
    pv = PointVerbose(1)
    assert pv.x == 1

    # Test space-separated string input
    SpacePoint = immutable('x y z', name='SpacePoint')
    sp = SpacePoint(1, 2, 3)
    assert sp.z == 3
    assert repr(sp) == 'SpacePoint(x=1, y=2, z=3)'
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_immutable():
    # Test basic functionality with string members
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

    # Test .set() functionality
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == 'Point(x=3, y=2)'
    assert p != p2  # Ensure it's a new instance

    # Test .set() with no arguments returns self
    p3 = p.set()
    assert p3 is p

    # Test .set() with invalid field
    with pytest.raises(AttributeError, match="'z' is not a member"):
        p.set(z=10)

    # Test inheritance and custom logic
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    pp = PositivePoint(1, 2)
    assert pp.x == 1
    pp_updated = pp.set(x=5)
    assert pp_updated.x == 5
    assert isinstance(pp_updated, PositivePoint)
    
    with pytest.raises(Exception, match='Coordinates must be positive!'):
        PositivePoint(-1, 2)

    # Test frozen members (trailing underscore)
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p_id = PointWithId(1, 2, id_=17)
    assert p_id.id_ == 17
    
    # Updating non-frozen member works
    p_id_updated = p_id.set(x=10)
    assert p_id_updated.x == 10
    assert p_id_updated.id_ == 17

    # Updating frozen member raises AttributeError
    with pytest.raises(AttributeError, match="Cannot set frozen members 'id_'"):
        p_id.set(id_=18)

    # Test empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    assert e.set(x=1) == e # Should not raise error even if field doesn't exist if kwargs is empty, 
                           # but here kwargs is not empty, so it checks membership.
    
    # Test list of members instead of string
    ListPoint = immutable(['a', 'b'], name='ListPoint')
    lp = ListPoint(1, 2)
    assert lp.a == 1
    assert lp.b == 2

    # Test SyntaxError on invalid template generation
    # (This is hard to trigger via arguments without breaking the logic, 
    # but we test the behavior of the decorator logic)
    with pytest.raises(SyntaxError):
        # Forcing a syntax error by passing a name that is not a valid identifier
        # Note: The template uses {class_name} in a class definition.
        invalid_name = "class 123(base):" 
        immutable('x', name=invalid_name)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_immutable():
    # Test basic standalone usage
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

    # Test .set() functionality
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == 'Point(x=3, y=2)'
    assert p != p2  # Ensure original is not mutated

    # Test .set() with no arguments
    p3 = p.set()
    assert p3 == p

    # Test error for non-existent member
    with pytest.raises(AttributeError) as excinfo:
        p.set(z=10)
    assert "'z' is not a member" in str(excinfo.value)

    # Test inheritance and custom logic
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    pp1 = PositivePoint(1, 2)
    assert pp1.x == 1
    assert pp1.y == 2
    
    pp2 = pp1.set(x=5)
    assert pp2.x == 5
    assert pp2.y == 2
    assert isinstance(pp2, PositivePoint)

    with pytest.raises(Exception, match='Coordinates must be positive!'):
        PositivePoint(-1, 2)

    # Test frozen members (trailing underscore)
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p_id = PointWithId(1, 2, id_=17)
    assert p_id.id_ == 17
    
    # Updating non-frozen member works
    p_id_updated = p_id.set(x=10)
    assert p_id_updated.x == 10
    assert p_id_updated.id_ == 17

    # Updating frozen member raises AttributeError
    with pytest.raises(AttributeError) as excinfo:
        p_id.set(id_=18)
    assert 'Cannot set frozen members \'id_\'' in str(excinfo.value)

    # Test empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    assert e.set(a=1) == e # Should return self if no kwargs or if invalid (though logic returns self on empty kwargs)
    
    # Test space-separated string members
    SpaceSep = immutable('a b c', name='SpaceSep')
    s = SpaceSep(1, 2, 3)
    assert s.a == 1
    assert s.c == 3
```


