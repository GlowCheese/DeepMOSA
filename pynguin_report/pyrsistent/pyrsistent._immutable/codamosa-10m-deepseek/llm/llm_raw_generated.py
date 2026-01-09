####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function immutable
def test_immutable(): 
    # Test 1: Basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Basic instantiation failed"
    
    # Test 2: set method
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2, "set method failed"
    
    # Test 3: Frozen member
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p_id = PointWithId(1, 2, id_=17)
    try:
        p_id.set(id_=18)
        assert False, "Should not allow setting frozen member"
    except AttributeError:
        pass  # Expected
    
    # Test 4: Inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    pp = PositivePoint(1, 2)
    assert pp.x == 1 and pp.y == 2, "Inheritance failed"
    
    try:
        PositivePoint(-1, 2)
        assert False, "Should have raised exception for negative coordinate"
    except Exception:
        pass  # Expected
    
    print("All tests passed!")

# Run the unit test
if __name__ == "__main__":
    test_immutable()


# LLM-generated content at query #2
#--------------------------

# Unit test for function immutable
def test_immutable(): 
    # Test 1: Basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Basic instantiation failed"
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2, "set method failed"
    assert p.x == 1 and p.y == 2, "Original object mutated"
    print("Test 1 passed")

    # Test 2: Frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    try:
        p.set(id_=18)
        assert False, "Frozen member should not be settable"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_", "Unexpected error message"
    print("Test 2 passed")

    # Test 3: Inheritance and custom validation
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p = PositivePoint(1, 2)
    assert p.x == 1 and p.y == 2, "Inheritance instantiation failed"
    try:
        PositivePoint(-1, 2)
        assert False, "Validation should have failed"
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!', "Unexpected error message"
    print("Test 3 passed")

    # Test 4: Non-existent member
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=3)
        assert False, "Setting non-existent member should fail"
    except AttributeError as e:
        assert "'z' is not a member" in str(e), "Unexpected error message"
    print("Test 4 passed")

    # Test 5: Empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    e2 = e.set()
    assert e == e2, "Empty immutable should be equal to itself after set"
    print("Test 5 passed")

    # Test 6: String representation
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert repr(p) == 'Point(x=1, y=2)', f"Unexpected repr: {repr(p)}"
    print("Test 6 passed")

    # Test 7: Mixed members with and without trailing underscore
    Mixed = immutable('a, b_, c, d_', name='Mixed')
    m = Mixed(1, 2, 3, 4)
    assert m.a == 1 and m.b_ == 2 and m.c == 3 and m.d_ == 4, "Mixed member instantiation failed"
    m2 = m.set(a=10, c=30)
    assert m2.a == 10 and m2.c == 30 and m2.b_ == 2 and m2.d_ == 4, "Setting non-frozen members failed"
    try:
        m.set(b_=20)
        assert False, "Frozen member b_ should not be settable"
    except AttributeError:
        pass
    try:
        m.set(d_=40)
        assert False, "Frozen member d_ should not be settable"
    except AttributeError:
        pass
    print("Test 7 passed")

    # Test 8: Verbose mode (just ensure it doesn't crash)
    Point = immutable('x, y', name='Point', verbose=False)
    print("Test 8 passed")

    # Test 9: Using comma-separated string
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Comma-separated string failed"
    print("Test 9 passed")

    # Test 10: Using space-separated string
    Point = immutable('x y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Space-separated string failed"
    print("Test 10 passed")

    print("All tests passed!")

if __name__ == "__main__":
    test_immutable()


# LLM-generated content at query #3
#--------------------------

# Unit test for function immutable
def test_immutable(): 
    # Test 1: Basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Basic instantiation failed"
    
    # Test 2: Set method
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2, "Set method failed"
    
    # Test 3: Frozen members
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p_id = PointWithId(1, 2, id_=17)
    try:
        p_id.set(id_=18)
        assert False, "Should not allow setting frozen member"
    except AttributeError as e:
        assert "Cannot set frozen members id_" in str(e), "Frozen member test failed"
    
    # Test 4: Inheritance and custom validation
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    pp = PositivePoint(1, 2)
    assert pp.x == 1 and pp.y == 2, "Inheritance with custom validation failed"
    
    try:
        PositivePoint(-1, 2)
        assert False, "Should raise exception for negative coordinates"
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!', "Custom validation exception message mismatch"
    
    # Test 5: Non-existent member error
    try:
        p.set(z=10)
        assert False, "Should raise AttributeError for non-existent member"
    except AttributeError as e:
        assert "'z' is not a member" in str(e), "Non-existent member error message mismatch"
    
    print("All tests passed!")

# Run the unit test
if __name__ == "__main__":
    test_immutable()


# LLM-generated content at query #4
#--------------------------

# Unit test for function immutable
def test_immutable():    
    # Test 1: Basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Basic instantiation failed"
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2, "Setting attribute failed"
    assert p.x == 1 and p.y == 2, "Original object mutated"
    
    # Test 2: Frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    try:
        p.set(id_=18)
        assert False, "Should not be able to set frozen member"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_", "Unexpected error message"
    
    # Test 3: Inheritance and custom validation
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p = PositivePoint(1, 2)
    try:
        PositivePoint(-1, 2)
        assert False, "Should have raised exception for negative coordinate"
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!', "Unexpected error message"
    
    # Test 4: Non-existent attribute
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=3)
        assert False, "Should not be able to set non-existent attribute"
    except AttributeError as e:
        assert "'z' is not a member" in str(e), "Unexpected error message"
    
    # Test 5: Empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    e2 = e.set()
    assert e == e2, "Empty set should return same object"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_immutable()


# LLM-generated content at query #5
#--------------------------

# Unit test for function immutable
def test_immutable():    
    # Test 1: Basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Basic instantiation failed"
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2, "Setting attribute failed"
    assert p.x == 1 and p.y == 2, "Original object mutated"
    print("Test 1 passed: Basic functionality")

    # Test 2: Frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    try:
        p.set(id_=18)
        assert False, "Should not allow setting frozen member"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_", f"Unexpected error message: {e}"
    print("Test 2 passed: Frozen members")

    # Test 3: Inheritance and validation
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p = PositivePoint(1, 2)
    assert p.x == 1 and p.y == 2, "Inheritance instantiation failed"
    try:
        PositivePoint(-1, 2)
        assert False, "Should not allow negative coordinates"
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!', f"Unexpected error message: {e}"
    print("Test 3 passed: Inheritance and validation")

    # Test 4: String representation
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert repr(p) == 'Point(x=1, y=2)', f"Unexpected repr: {repr(p)}"
    print("Test 4 passed: String representation")

    # Test 5: No members
    Empty = immutable(name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()', f"Unexpected repr for empty: {repr(e)}"
    print("Test 5 passed: No members")

    # Test 6: Invalid attribute set
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=3)
        assert False, "Should not allow setting non-existent attribute"
    except AttributeError as e:
        assert "'z' is not a member" in str(e), f"Unexpected error message: {e}"
    print("Test 6 passed: Invalid attribute set")

    # Test 7: Multiple frozen members
    Person = immutable('name, age, id_, ssn_', name='Person')
    p = Person('Alice', 30, id_=123, ssn_=456)
    try:
        p.set(id_=124, ssn_=457)
        assert False, "Should not allow setting multiple frozen members"
    except AttributeError as e:
        assert "Cannot set frozen members id_, ssn_" in str(e), f"Unexpected error message: {e}"
    print("Test 7 passed: Multiple frozen members")

    # Test 8: Empty set call
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set()
    assert p is p2, "Empty set should return same object"
    print("Test 8 passed: Empty set call")

    # Test 9: Mixed valid and invalid attribute set
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(x=3, z=4)
        assert False, "Should not allow mixed valid and invalid attributes"
    except AttributeError as e:
        assert "'z' is not a member" in str(e), f"Unexpected error message: {e}"
    print("Test 9 passed: Mixed valid and invalid attribute set")

    # Test 10: Complex member names
    Complex = immutable('x1, y2, z_3', name='Complex')
    c = Complex(1, 2, 3)
    assert c.x1 == 1 and c.y2 == 2 and c.z_3 == 3, "Complex member names failed"
    c2 = c.set(x1=4)
    assert c2.x1 == 4 and c2.y2 == 2 and c2.z_3 == 3, "Setting complex member failed"
    print("Test 10 passed: Complex member names")

    print("All tests passed!")

if __name__ == "__main__":
    test_immutable()


####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function immutable
def test_immutable():    
    # Test 1: Basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Basic instantiation failed"
    
    # Test 2: set method
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2, "set method failed"
    
    # Test 3: Frozen member
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p = PointWithId(1, 2, id_=17)
    try:
        p.set(id_=18)
        assert False, "Should not be able to set frozen member"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_", "Frozen member test failed"
    
    # Test 4: Inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p = PositivePoint(1, 2)
    assert p.x == 1 and p.y == 2, "Inheritance instantiation failed"
    
    try:
        PositivePoint(-1, 2)
        assert False, "Should have raised exception for negative coordinate"
    except Exception as e:
        assert str(e) == "Coordinates must be positive!", "Inheritance validation failed"
    
    print("All tests passed!")

# Run the unit test
if __name__ == "__main__":
    test_immutable()


# LLM-generated content at query #2
#--------------------------

# Unit test for function immutable
def test_immutable(): 
    # Test 1: Basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Basic instantiation failed"
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2, "Setting attribute failed"
    assert p.x == 1 and p.y == 2, "Original object mutated"
    print("Test 1 passed: Basic functionality")

    # Test 2: Frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    try:
        p.set(id_=18)
        assert False, "Should have raised AttributeError for frozen member"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_", f"Unexpected error message: {e}"
    print("Test 2 passed: Frozen members")

    # Test 3: Inheritance and validation
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p = PositivePoint(1, 2)
    assert p.x == 1 and p.y == 2, "Inheritance instantiation failed"
    try:
        PositivePoint(-1, 2)
        assert False, "Should have raised Exception for negative coordinate"
    except Exception as e:
        assert str(e) == "Coordinates must be positive!", f"Unexpected error message: {e}"
    print("Test 3 passed: Inheritance and validation")

    # Test 4: Non-existent attribute
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=3)
        assert False, "Should have raised AttributeError for non-existent member"
    except AttributeError as e:
        assert "'z' is not a member" in str(e), f"Unexpected error message: {e}"
    print("Test 4 passed: Non-existent attribute")

    # Test 5: No mutation when no kwargs
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set()
    assert p is p2, "Should return same object when no kwargs provided"
    print("Test 5 passed: No mutation when no kwargs")

    # Test 6: String representation
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    repr_str = repr(p)
    assert repr_str == "Point(x=1, y=2)", f"Unexpected representation: {repr_str}"
    print("Test 6 passed: String representation")

    # Test 7: Multiple frozen members
    Point = immutable('x, y, id_, version_', name='Point')
    p = Point(1, 2, id_=17, version_=1)
    try:
        p.set(id_=18, version_=2)
        assert False, "Should have raised AttributeError for multiple frozen members"
    except AttributeError as e:
        assert "Cannot set frozen members id_, version_" in str(e), f"Unexpected error message: {e}"
    print("Test 7 passed: Multiple frozen members")

    # Test 8: Mixed updates with frozen members
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    p2 = p.set(x=3, y=4)
    assert p2.x == 3 and p2.y == 4 and p2.id_ == 17, "Mixed update with frozen member failed"
    print("Test 8 passed: Mixed updates with frozen members")

    # Test 9: Empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    repr_str = repr(e)
    assert repr_str == "Empty()", f"Unexpected representation for empty: {repr_str}"
    print("Test 9 passed: Empty members")

    # Test 10: Verbose mode (just ensure it doesn't crash)
    Point = immutable('x, y', name='Point', verbose=False)
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Verbose mode instantiation failed"
    print("Test 10 passed: Verbose mode")

    print("All tests passed!")

if __name__ == "__main__":
    test_immutable()


# LLM-generated content at query #3
#--------------------------

# Unit test for function immutable
def test_immutable():    
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3)
    assert p.x == 1 and p.y == 2
    assert p2.x == 3 and p2.y == 2
    print("Basic functionality test passed.")

    # Test frozen members
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p = PointWithId(1, 2, id_=17)
    p2 = p.set(x=3)
    assert p2.id_ == 17
    try:
        p.set(id_=18)
        print("ERROR: Should have raised AttributeError for frozen member")
    except AttributeError as e:
        if "Cannot set frozen members id_" in str(e):
            print("Frozen member test passed.")
        else:
            print("ERROR: Unexpected error message:", e)

    # Test error on invalid member
    try:
        p.set(z=5)
        print("ERROR: Should have raised AttributeError for invalid member")
    except AttributeError as e:
        if "'z' is not a member" in str(e):
            print("Invalid member test passed.")
        else:
            print("ERROR: Unexpected error message:", e)

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
        p.set(y=-3)
        print("ERROR: Should have raised Exception for negative coordinate")
    except Exception as e:
        if "Coordinates must be positive" in str(e):
            print("Inheritance test passed.")
        else:
            print("ERROR: Unexpected error message:", e)

    print("All tests passed.")

if __name__ == "__main__":
    test_immutable()


# LLM-generated content at query #4
#--------------------------

# Unit test for function immutable
def test_immutable():    
    # Test 1: Basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Basic instantiation failed"
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2, "set method failed"
    assert p.x == 1 and p.y == 2, "Original object mutated"
    
    # Test 2: Frozen members
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p = PointWithId(1, 2, id_=17)
    try:
        p.set(id_=18)
        assert False, "Should not allow setting frozen member"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_", "Unexpected error message"
    
    # Test 3: Inheritance
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p = PositivePoint(1, 2)
    assert p.x == 1 and p.y == 2, "Inheritance instantiation failed"
    try:
        PositivePoint(-1, 2)
        assert False, "Should not allow negative coordinates"
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!', "Unexpected error message"
    
    # Test 4: Non-existent member
    p = Point(1, 2)
    try:
        p.set(z=3)
        assert False, "Should not allow setting non-existent member"
    except AttributeError as e:
        assert "'z' is not a member" in str(e), "Unexpected error message"
    
    # Test 5: Empty set
    p = Point(1, 2)
    p2 = p.set()
    assert p == p2, "Empty set should return same object"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_immutable()


# LLM-generated content at query #5
#--------------------------

# Unit test for function immutable
def test_immutable():    
    # Test basic functionality
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2, "Basic instantiation failed"
    p2 = p.set(x=3)
    assert p2.x == 3 and p2.y == 2, "Set method failed"
    assert p.x == 1 and p.y == 2, "Original object mutated"
    
    # Test frozen members
    PointWithId = immutable('x, y, id_', name='PointWithId')
    p = PointWithId(1, 2, id_=17)
    try:
        p.set(id_=18)
        assert False, "Should not be able to set frozen member"
    except AttributeError as e:
        assert str(e) == "Cannot set frozen members id_", "Wrong error message for frozen member"
    
    # Test inheritance and custom validation
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    
    p = PositivePoint(1, 2)
    assert p.x == 1 and p.y == 2, "Inheritance failed"
    try:
        PositivePoint(-1, 2)
        assert False, "Should have raised exception for negative coordinate"
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!', "Custom validation failed"
    
    # Test error on invalid member
    p = Point(1, 2)
    try:
        p.set(z=3)
        assert False, "Should not be able to set non-existent member"
    except AttributeError as e:
        assert "'z' is not a member" in str(e), "Wrong error message for invalid member"
    
    print("All tests passed.")

if __name__ == "__main__":
    test_immutable()


