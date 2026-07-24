####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_immutable():
    # Test basic immutable creation with string members
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'
    
    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p.x == 1  # Original unchanged
    
    # Test set with no kwargs returns same instance
    p3 = p.set()
    assert p3 is p
    
    # Test set with multiple fields
    p4 = p.set(x=5, y=6)
    assert p4.x == 5
    assert p4.y == 6
    
    # Test invalid member raises AttributeError
    with pytest.raises(AttributeError, match="'z' is not a member"):
        p.set(z=10)
    
    # Test frozen members (ending with _)
    Point2 = immutable('x, y, id_', name='Point2')
    p5 = Point2(1, 2, id_=17)
    assert p5.id_ == 17
    
    # Can set non-frozen members
    p6 = p5.set(x=3)
    assert p6.x == 3
    assert p6.id_ == 17
    
    # Cannot set frozen members
    with pytest.raises(AttributeError, match="Cannot set frozen members id_"):
        p5.set(id_=18)
    
    # Test with list of members
    Point3 = immutable(['a', 'b'], name='Point3')
    p7 = Point3(10, 20)
    assert p7.a == 10
    assert p7.b == 20
    
    # Test with comma-separated string with spaces
    Point4 = immutable('p, q, r', name='Point4')
    p8 = Point4(1, 2, 3)
    assert p8.p == 1
    assert p8.q == 2
    assert p8.r == 3
    
    # Test with comma-separated string with extra commas and spaces
    Point5 = immutable('x,  y,  z', name='Point5')
    p9 = Point5(1, 2, 3)
    assert p9.x == 1
    assert p9.y == 2
    assert p9.z == 3
    
    # Test empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    
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
    assert isinstance(pp2, PositivePoint)
    assert pp2.x == 3
    
    with pytest.raises(Exception, match='Coordinates must be positive!'):
        pp.set(y=-3)
    
    # Test multiple frozen members
    Multi = immutable('a, b_, c, d_', name='Multi')
    m = Multi(1, 2, 3, 4)
    
    m2 = m.set(a=10, c=30)
    assert m2.a == 10
    assert m2.b_ == 2
    assert m2.c == 30
    assert m2.d_ == 4
    
    with pytest.raises(AttributeError, match="Cannot set frozen members b_, d_"):
        m.set(b_=20, d_=40)
    
    # Test invalid attribute error message format
    Point6 = immutable('x, y', name='Point6')
    p10 = Point6(1, 2)
    with pytest.raises(AttributeError, match="is not a member"):
        p10.set(invalid=999)
    
    # Test namedtuple functionality is preserved
    Point7 = immutable('x, y', name='Point7')
    p11 = Point7(1, 2)
    assert p11[0] == 1
    assert p11[1] == 2
    assert len(p11) == 2


# LLM-generated content at query #2
#--------------------------

```python
def test_immutable():
    # Test basic immutable creation with string members
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'
    
    # Test set method
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p.x == 1  # Original unchanged
    
    # Test set with no arguments
    p3 = p.set()
    assert p3 is p
    
    # Test invalid member
    with pytest.raises(AttributeError, match="is not a member"):
        p.set(z=5)
    
    # Test frozen members
    Point_frozen = immutable('x, y, id_', name='PointFrozen')
    pf = Point_frozen(1, 2, id_=17)
    assert pf.id_ == 17
    
    # Can modify non-frozen members
    pf2 = pf.set(x=3)
    assert pf2.x == 3
    assert pf2.id_ == 17
    
    # Cannot modify frozen members
    with pytest.raises(AttributeError, match="Cannot set frozen members"):
        pf.set(id_=18)
    
    # Test with list of members
    Point_list = immutable(['x', 'y'], name='PointList')
    pl = Point_list(5, 6)
    assert pl.x == 5
    assert pl.y == 6
    
    # Test with comma-separated string
    Point_comma = immutable('a, b, c', name='PointComma')
    pc = Point_comma(1, 2, 3)
    assert pc.a == 1
    assert pc.b == 2
    assert pc.c == 3
    
    # Test with empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    
    # Test with single member
    Single = immutable('x', name='Single')
    s = Single(42)
    assert s.x == 42
    
    # Test multiple frozen members
    Multi_frozen = immutable('x_, y_, z', name='MultiFrozen')
    mf = Multi_frozen(1, 2, 3)
    with pytest.raises(AttributeError, match="Cannot set frozen members"):
        mf.set(x_=10)
    with pytest.raises(AttributeError, match="Cannot set frozen members"):
        mf.set(y_=20)
    mf2 = mf.set(z=30)
    assert mf2.z == 30
    
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
    assert isinstance(pp2, PositivePoint)
    assert pp2.x == 3
    
    with pytest.raises(Exception, match='Coordinates must be positive'):
        pp.set(y=-3)
    
    # Test verbose mode doesn't break functionality
    Verbose = immutable('a, b', name='Verbose', verbose=False)
    v = Verbose(1, 2)
    assert v.a == 1
    
    # Test multiple sets
    p_chain = Point(1, 2).set(x=10).set(y=20)
    assert p_chain.x == 10
    assert p_chain.y == 20


# LLM-generated content at query #3
#--------------------------

def test_immutable():
    # Test basic creation and access
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

    # Test set method with single field
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p.x == 1  # Original unchanged
    assert repr(p2) == 'Point(x=3, y=2)'

    # Test set method with multiple fields
    p3 = p.set(x=5, y=6)
    assert p3.x == 5
    assert p3.y == 6

    # Test set with no arguments returns same instance
    p4 = p.set()
    assert p4 is p

    # Test set with invalid member raises AttributeError
    try:
        p.set(z=10)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "is not a member" in str(e)

    # Test frozen members
    FrozenPoint = immutable('x, y, id_', name='FrozenPoint')
    fp = FrozenPoint(1, 2, id_=17)
    assert fp.x == 1
    assert fp.y == 2
    assert fp.id_ == 17

    # Test setting non-frozen member works
    fp2 = fp.set(x=3)
    assert fp2.x == 3
    assert fp2.id_ == 17

    # Test setting frozen member raises AttributeError
    try:
        fp.set(id_=18)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Cannot set frozen members" in str(e)

    # Test multiple frozen members
    try:
        fp.set(x=5, id_=18)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Cannot set frozen members" in str(e)

    # Test with string members input
    Point2 = immutable('a, b, c', name='Point2')
    p = Point2(1, 2, 3)
    assert p.a == 1
    assert p.b == 2
    assert p.c == 3

    # Test with list members input
    Point3 = immutable(['x', 'y'], name='Point3')
    p = Point3(10, 20)
    assert p.x == 10
    assert p.y == 20

    # Test empty immutable
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'

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

    try:
        PositivePoint(-1, 2)
        assert False, "Should have raised Exception"
    except Exception as e:
        assert "Coordinates must be positive" in str(e)

    # Test set on inherited class
    pp2 = pp.set(x=3)
    assert pp2.x == 3
    assert pp2.y == 2
    assert isinstance(pp2, PositivePoint)

    try:
        pp.set(y=-3)
        assert False, "Should have raised Exception"
    except Exception as e:
        assert "Coordinates must be positive" in str(e)

    # Test verbose mode doesn't break functionality
    VerbosePoint = immutable('x, y', name='VerbosePoint', verbose=False)
    vp = VerbosePoint(1, 2)
    assert vp.x == 1


# LLM-generated content at query #4
#--------------------------

```python
def test_immutable():
    # Test basic immutable creation with string members
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

    # Test set method creates new instance
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p.x == 1  # Original unchanged
    assert repr(p2) == 'Point(x=3, y=2)'

    # Test set with multiple values
    p3 = p.set(x=5, y=6)
    assert p3.x == 5
    assert p3.y == 6

    # Test set with no kwargs returns same instance
    p4 = p.set()
    assert p4 is p

    # Test set with invalid member raises AttributeError
    with pytest.raises(AttributeError, match="is not a member"):
        p.set(z=10)

    # Test frozen members (ending with underscore)
    Point_frozen = immutable('x, y, id_', name='Point_frozen')
    pf = Point_frozen(1, 2, id_=17)
    assert pf.id_ == 17

    # Test setting non-frozen member works
    pf2 = pf.set(x=3)
    assert pf2.x == 3
    assert pf2.id_ == 17

    # Test setting frozen member raises AttributeError
    with pytest.raises(AttributeError, match="Cannot set frozen members"):
        pf.set(id_=18)

    # Test with list of members
    Point_list = immutable(['x', 'y'], name='Point_list')
    pl = Point_list(10, 20)
    assert pl.x == 10
    assert pl.y == 20

    # Test with comma-separated string with spaces
    Point_spaces = immutable('x, y, z', name='Point_spaces')
    ps = Point_spaces(1, 2, 3)
    assert ps.x == 1
    assert ps.y == 2
    assert ps.z == 3

    # Test empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'

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
    assert isinstance(pp2, PositivePoint)
    assert pp2.x == 3

    # Test that invalid coordinates raise exception
    with pytest.raises(Exception, match='Coordinates must be positive'):
        PositivePoint(-1, 2)

    with pytest.raises(Exception, match='Coordinates must be positive'):
        pp.set(y=-3)

    # Test multiple frozen members
    MultiFreeze = immutable('a, b_, c, d_', name='MultiFreeze')
    mf = MultiFreeze(1, 2, 3, 4)
    mf2 = mf.set(a=10, c=30)
    assert mf2.a == 10
    assert mf2.c == 30

    with pytest.raises(AttributeError, match="Cannot set frozen members"):
        mf.set(b_=20)

    with pytest.raises(AttributeError, match="Cannot set frozen members"):
        mf.set(d_=40)

    # Test verbose output doesn't crash
    Point_verbose = immutable('x, y', name='Point_verbose', verbose=True)
    pv = Point_verbose(1, 2)
    assert pv.x == 1


# LLM-generated content at query #5
#--------------------------

def test_immutable():
    # Test basic immutable creation with simple members
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    
    # Test set method returns new instance
    p2 = p.set(x=3)
    assert p.x == 1  # Original unchanged
    assert p2.x == 3
    assert p2.y == 2
    
    # Test set with no arguments returns same instance
    p3 = p.set()
    assert p3 is p
    
    # Test repr
    assert repr(p) == 'Point(x=1, y=2)'
    
    # Test with list of members
    Point2 = immutable(['a', 'b', 'c'], name='Point2')
    p4 = Point2(1, 2, 3)
    assert p4.a == 1
    assert p4.b == 2
    assert p4.c == 3
    
    # Test frozen members (ending with underscore)
    FrozenPoint = immutable('x, y, id_', name='FrozenPoint')
    fp = FrozenPoint(1, 2, id_=17)
    assert fp.x == 1
    assert fp.id_ == 17
    
    # Test can modify non-frozen members
    fp2 = fp.set(x=5)
    assert fp2.x == 5
    assert fp2.id_ == 17
    
    # Test cannot modify frozen members
    with pytest.raises(AttributeError, match='Cannot set frozen members'):
        fp.set(id_=18)
    
    # Test invalid member name raises error
    with pytest.raises(AttributeError, match='is not a member'):
        p.set(z=10)
    
    # Test multiple invalid members
    with pytest.raises(AttributeError, match='is not a member'):
        p.set(z=10, w=20)
    
    # Test with empty members
    Empty = immutable('', name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    
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
    assert isinstance(pp2, PositivePoint)
    assert pp2.x == 3
    
    with pytest.raises(Exception, match='Coordinates must be positive'):
        PositivePoint(-1, 2)
    
    with pytest.raises(Exception, match='Coordinates must be positive'):
        pp.set(y=-3)
    
    # Test comma-separated string parsing
    Comma = immutable('x, y, z', name='Comma')
    c = Comma(1, 2, 3)
    assert c.x == 1
    assert c.y == 2
    assert c.z == 3
    
    # Test verbose parameter doesn't break functionality
    VerbosePoint = immutable('x, y', name='VerbosePoint', verbose=False)
    vp = VerbosePoint(1, 2)
    assert vp.x == 1
    
    # Test multiple frozen members
    MultiFreeze = immutable('a, b_, c, d_', name='MultiFreeze')
    mf = MultiFreeze(1, 2, 3, 4)
    mf2 = mf.set(a=10, c=30)
    assert mf2.a == 10
    assert mf2.c == 30
    
    with pytest.raises(AttributeError, match='Cannot set frozen members'):
        mf.set(b_=20)
    
    with pytest.raises(AttributeError, match='Cannot set frozen members'):
        mf.set(d_=40)


