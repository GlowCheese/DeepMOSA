####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_immutable():
    # Test basic creation and attribute access
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p.x == 1  # Original should be unchanged

    # Test repr
    assert repr(p) == "Point(x=1, y=2)"
    assert repr(p2) == "Point(x=3, y=2)"

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

    pp2 = pp.set(x=3)
    assert pp2.x == 3
    assert pp2.y == 2

    # Test frozen members
    PointWithID = immutable('x, y, id_', name='PointWithID')
    p_id = PointWithID(1, 2, id_=17)
    assert p_id.id_ == 17

    p_id2 = p_id.set(x=3)
    assert p_id2.x == 3
    assert p_id2.id_ == 17

    # Test setting frozen member raises error
    try:
        p_id.set(id_=18)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Cannot set frozen members id_" in str(e)

    # Test invalid member name
    try:
        p.set(z=5)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'z' is not a member" in str(e)

    # Test empty set call
    assert p.set() is p

    # Test verbose mode (just check it doesn't crash)
    try:
        immutable('x, y', name='Test', verbose=True)
    except Exception:
        assert False, "verbose mode should not raise exception"


