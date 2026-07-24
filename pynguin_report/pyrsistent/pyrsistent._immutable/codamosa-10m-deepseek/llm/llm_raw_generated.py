####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function immutable
def test_immutable():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3)
    assert p.x == 1 and p.y == 2
    assert p2.x == 3 and p2.y == 2

    PointWithFrozen = immutable('x, y, id_', name='PointWithFrozen')
    p3 = PointWithFrozen(1, 2, id_=17)
    p4 = p3.set(x=3)
    assert p3.x == 1 and p3.y == 2 and p3.id_ == 17
    assert p4.x == 3 and p4.y == 2 and p4.id_ == 17

    exception_raised = False
    try:
        p3.set(id_=18)
    except AttributeError:
        exception_raised = True
    assert exception_raised

    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    p5 = PositivePoint(1, 2)
    p6 = p5.set(x=3)
    assert p5.x == 1 and p5.y == 2
    assert p6.x == 3 and p6.y == 2

    exception_raised = False
    try:
        p5.set(y=-3)
    except Exception:
        exception_raised = True
    assert exception_raised

if __name__ == '__main__':
    test_immutable()


# LLM-generated content at query #2
#--------------------------

# Unit test for function immutable
def test_immutable():
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2

    # Test frozen members
    PointWithID = immutable('x, y, id_', name='PointWithID')
    p = PointWithID(1, 2, id_=17)
    assert p.id_ == 17
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.id_ == 17
    try:
        p.set(id_=18)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    # Test subclassing
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
        assert str(e) == "Coordinates must be positive!"

    print("All tests passed.")

if __name__ == "__main__":
    test_immutable()


# LLM-generated content at query #3
#--------------------------

# Unit test for function immutable
def test_immutable():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2

    PointWithFrozen = immutable('x, y, id_', name='PointWithFrozen')
    p_frozen = PointWithFrozen(1, 2, id_=17)
    assert p_frozen.id_ == 17

    try:
        p_frozen.set(id_=18)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    p_positive = PositivePoint(1, 2)
    assert p_positive.x == 1
    assert p_positive.y == 2

    try:
        p_positive.set(y=-3)
        assert False, "Should have raised Exception"
    except Exception as e:
        assert str(e) == "Coordinates must be positive!"

    print("All tests passed.")

# Run the unit test
if __name__ == "__main__":
    test_immutable()


# LLM-generated content at query #4
#--------------------------

# Unit test for function immutable
def test_immutable():
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert str(p) == 'Point(x=1, y=2)'
    
    # Test set functionality
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    
    # Test frozen members
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p = PointWithId(1, 2, id_=17)
    assert p.id_ == 17
    
    # Should be able to change non-frozen members
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.id_ == 17
    
    # Should raise when trying to change frozen member
    try:
        p.set(id_=18)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass
    
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
    
    # Should work with positive values
    p2 = p.set(x=3)
    assert p2.x == 3
    
    # Should raise with negative values
    try:
        p.set(y=-3)
        assert False, "Should have raised Exception"
    except Exception:
        pass
    
    print("All tests passed!")

if __name__ == '__main__':
    test_immutable()


# LLM-generated content at query #5
#--------------------------

# Unit test for function immutable
def test_immutable():
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2
    
    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2
    
    # Test repr
    assert repr(p) == 'Point(x=1, y=2)'
    
    # Test frozen members
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p = PointWithId(1, 2, 17)
    try:
        p.set(id_=18)
        assert False, "Should not be able to modify frozen member"
    except AttributeError:
        pass
    
    # Test inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p = PositivePoint(1, 2)
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2
    
    try:
        PositivePoint(-1, 2)
        assert False, "Should not allow negative coordinates"
    except Exception:
        pass
    
    print("All tests passed!")

if __name__ == '__main__':
    test_immutable()


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function immutable
def test_immutable():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Initial values not set correctly"
    
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2, "set() method failed to update x"
    assert p.x == 1 and p.y == 2, "Original instance modified by set()"
    
    try:
        p.set(z=4)
        assert False, "Setting non-existent attribute should raise AttributeError"
    except AttributeError:
        pass
    
    # Test frozen members
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p_id = PointWithId(1, 2, id_=17)
    try:
        p_id.set(id_=18)
        assert False, "Modifying frozen member should raise AttributeError"
    except AttributeError:
        pass
    
    # Test inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p_pos = PositivePoint(1, 2)
    try:
        PositivePoint(-1, 2)
        assert False, "Negative coordinates should raise Exception"
    except Exception:
        pass
    
    print("All tests passed!")

if __name__ == '__main__':
    test_immutable()


# LLM-generated content at query #2
#--------------------------

# Unit test for function immutable
def test_immutable():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2

    PointWithFrozen = immutable('x, y, id_', name='PointWithFrozen')
    p3 = PointWithFrozen(1, 2, id_=17)
    assert p3.id_ == 17

    try:
        p3.set(id_=18)
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_"

    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    p4 = PositivePoint(1, 2)
    assert p4.x == 1
    assert p4.y == 2

    try:
        p4.set(y=-3)
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!'

    # Test repr
    assert repr(p) == 'Point(x=1, y=2)'
    assert repr(p2) == 'Point(x=3, y=2)'
    assert repr(p3) == 'PointWithFrozen(x=1, y=2, id_=17)'


# LLM-generated content at query #3
#--------------------------

# Unit test for function immutable
def test_immutable():
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2
    
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2
    
    # Test repr
    assert repr(p) == 'Point(x=1, y=2)'
    
    # Test invalid attribute
    try:
        p.set(z=5)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass
    
    # Test frozen members
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p = PointWithId(1, 2, id_=17)
    assert p.id_ == 17
    
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.id_ == 17
    
    try:
        p.set(id_=18)
        assert False, "Should have raised AttributeError for frozen member"
    except AttributeError:
        pass
    
    # Test inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p = PositivePoint(1, 2)
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2
    
    try:
        PositivePoint(-1, 2)
        assert False, "Should have raised Exception for negative coordinate"
    except Exception:
        pass
    
    try:
        p.set(y=-3)
        assert False, "Should have raised Exception for negative coordinate in set"
    except Exception:
        pass
    
    print("All tests passed!")

if __name__ == '__main__':
    test_immutable()


# LLM-generated content at query #4
#--------------------------

# Unit test for function immutable
def test_immutable():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Initialization failed"

    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2, "Set method failed"

    try:
        p.set(z=3)
        assert False, "Setting non-existent attribute should raise AttributeError"
    except AttributeError:
        pass

    Point2 = immutable('x, y, id_', name='Point2')
    p3 = Point2(1, 2, id_=17)
    assert p3.id_ == 17, "Frozen member initialization failed"

    try:
        p3.set(id_=18)
        assert False, "Setting frozen attribute should raise AttributeError"
    except AttributeError:
        pass

    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    p4 = PositivePoint(1, 2)
    assert p4.x == 1 and p4.y == 2, "Inheritance initialization failed"

    try:
        PositivePoint(-1, 2)
        assert False, "Negative coordinates should raise Exception"
    except Exception:
        pass

    print("All tests passed.")

if __name__ == "__main__":
    test_immutable()


# LLM-generated content at query #5
#--------------------------

# Unit test for function immutable
def test_immutable():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2

    try:
        p.set(z=4)
        assert False, "Setting a non-existent member should raise an AttributeError"
    except AttributeError:
        pass

    PointWithFrozen = immutable('x, y, id_', name='PointWithFrozen')
    p = PointWithFrozen(1, 2, id_=17)
    try:
        p.set(id_=18)
        assert False, "Setting a frozen member should raise an AttributeError"
    except AttributeError:
        pass

    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')

    p = PositivePoint(1, 2)
    try:
        PositivePoint(-1, 2)
        assert False, "Creating a PositivePoint with negative coordinates should raise an Exception"
    except Exception:
        pass

    assert p.set(x=3) == PositivePoint(3, 2)

    print("All tests passed!")

if __name__ == "__main__":
    test_immutable()


