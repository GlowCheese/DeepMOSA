####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_immutable_basic_creation():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert str(p) == 'Point(x=1, y=2)'

def test_immutable_set_method():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p.x == 1
    assert p.y == 2

def test_immutable_set_multiple_fields():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3, y=4)
    assert p2.x == 3
    assert p2.y == 4

def test_immutable_set_no_change():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set()
    assert p2 == p
    assert p2 is p

def test_immutable_invalid_field_error():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=3)
        assert False
    except AttributeError as e:
        assert "'z' is not a member" in str(e)

def test_immutable_frozen_member():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    assert p.id_ == 17
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.id_ == 17
    try:
        p.set(id_=18)
        assert False
    except AttributeError as e:
        assert "Cannot set frozen members id_" in str(e)

def test_immutable_multiple_frozen_members():
    Thing = immutable('a, b_, c, d_', name='Thing')
    t = Thing(1, 2, 3, 4)
    assert t.b_ == 2
    assert t.d_ == 4
    t2 = t.set(a=10, c=30)
    assert t2.a == 10
    assert t2.c == 30
    assert t2.b_ == 2
    assert t2.d_ == 4
    try:
        t.set(b_=20, d_=40)
        assert False
    except AttributeError as e:
        assert "Cannot set frozen members" in str(e)
        assert "b_" in str(e)
        assert "d_" in str(e)

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
        assert False
    except Exception as e:
        assert 'Coordinates must be positive!' in str(e)

def test_immutable_no_members():
    Empty = immutable(name='Empty')
    e = Empty()
    assert str(e) == 'Empty()'
    e2 = e.set()
    assert e2 == e

def test_immutable_single_member():
    Single = immutable('x', name='Single')
    s = Single(5)
    assert s.x == 5
    s2 = s.set(x=10)
    assert s2.x == 10

def test_immutable_repr_custom_name():
    Thing = immutable('a, b', name='MyThing')
    t = Thing(1, 2)
    assert str(t) == 'MyThing(a=1, b=2)'

def test_immutable_members_as_list():
    Point = immutable(['x', 'y'], name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    p2 = p.set(x=3)
    assert p2.x == 3

def test_immutable_members_with_commas_and_spaces():
    Point = immutable('x,  y   ,z', name='Point')
    p = Point(1, 2, 3)
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3
    p2 = p.set(y=20)
    assert p2.y == 20
    assert p2.x == 1
    assert p2.z == 3


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_immutable_creates_point():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

def test_immutable_set_updates_field():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2

def test_immutable_set_returns_new_instance():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3)
    assert p is not p2

def test_immutable_set_no_changes_returns_self():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set()
    assert p is p2

def test_immutable_repr_uses_class_name():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert repr(p) == 'Point(x=1, y=2)'

def test_immutable_inheritance_with_custom_new():
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    p = PositivePoint(1, 2)
    assert p.x == 1
    assert p.y == 2

def test_immutable_inheritance_set_respects_custom_new():
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

def test_immutable_frozen_member_cannot_be_set():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    try:
        p.set(id_=18)
        assert False
    except AttributeError as e:
        assert str(e) == 'Cannot set frozen members id_'

def test_immutable_frozen_member_allows_other_updates():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert p2.id_ == 17

def test_immutable_set_invalid_member_raises_error():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=3)
        assert False
    except AttributeError as e:
        assert "'z' is not a member" in str(e)

def test_immutable_with_string_members():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

def test_immutable_with_list_members():
    Point = immutable(['x', 'y'], name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

def test_immutable_with_comma_separated_string():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

def test_immutable_no_members():
    Empty = immutable(name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'

def test_immutable_set_on_no_members_returns_self():
    Empty = immutable(name='Empty')
    e = Empty()
    e2 = e.set()
    assert e is e2

def test_immutable_multiple_frozen_members():
    Thing = immutable('a_, b, c_', name='Thing')
    t = Thing(a_=1, b=2, c_=3)
    try:
        t.set(a_=4, c_=5)
        assert False
    except AttributeError as e:
        assert 'Cannot set frozen members a_, c_' == str(e)

def test_immutable_set_multiple_fields():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3, y=4)
    assert p2.x == 3
    assert p2.y == 4


# LLM-generated content at query #2
#--------------------------

def test_immutable_raises_syntax_error_on_invalid_template():
    try:
        immutable(members='x, y', name='InvalidClass', verbose=False)
    except SyntaxError:
        pass
    else:
        assert False, "Expected SyntaxError not raised"


# LLM-generated content at query #3
#--------------------------

def test_predicate_at_line_91_evaluates_to_false():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3)
    assert p.x == 1
    assert p2.x == 3
    assert p2.y == 2


# LLM-generated content at query #4
#--------------------------

def test_immutable_with_empty_members_and_name_immutable():
    result = immutable(members='', name='Immutable', verbose=False)
    instance = result()
    assert instance._fields == ()
    assert repr(instance) == 'Immutable()'
    assert instance.set() is instance

def test_immutable_with_single_member_and_no_frozen():
    Point = immutable('x', name='Point')
    p = Point(5)
    p2 = p.set(x=10)
    assert p2.x == 10
    assert p.x == 5

def test_immutable_with_multiple_members_and_frozen():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.id_ == 17
    try:
        p.set(id_=18)
        assert False
    except AttributeError as e:
        assert str(e) == 'Cannot set frozen members id_'

def test_immutable_inheritance_and_custom_new():
    class PositivePoint(immutable('x, y')):
        __slots__ = tuple()
        def __new__(cls, x, y):
            if x > 0 and y > 0:
                return super(PositivePoint, cls).__new__(cls, x, y)
            raise Exception('Coordinates must be positive!')
    p = PositivePoint(1, 2)
    p2 = p.set(x=3)
    assert p2.x == 3
    try:
        p.set(y=-3)
        assert False
    except Exception as e:
        assert str(e) == 'Coordinates must be positive!'

def test_immutable_with_invalid_member_in_set():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=3)
        assert False
    except AttributeError as e:
        assert "'z' is not a member" in str(e)

def test_immutable_repr_correct_class_name():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert repr(p) == 'Point(x=1, y=2)'

def test_immutable_with_empty_kwargs_in_set():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.set() is p

def test_immutable_with_frozen_member_and_multiple_fields_to_modify():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    try:
        p.set(x=3, id_=18)
        assert False
    except AttributeError as e:
        assert str(e) == 'Cannot set frozen members id_'

def test_immutable_with_no_frozen_members():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3, y=4)
    assert p2.x == 3
    assert p2.y == 4

def test_immutable_verbose_false_no_output(capsys):
    Point = immutable('x, y', name='Point', verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''


# LLM-generated content at query #5
#--------------------------

def test_immutable_creates_point():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2
    assert repr(p) == 'Point(x=1, y=2)'

def test_immutable_set_updates_field():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3)
    assert p2.x == 3
    assert p2.y == 2
    assert repr(p2) == 'Point(x=3, y=2)'

def test_immutable_set_multiple_fields():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set(x=3, y=4)
    assert p2.x == 3
    assert p2.y == 4

def test_immutable_set_no_change():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    p2 = p.set()
    assert p2 is p

def test_immutable_set_invalid_field():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    try:
        p.set(z=3)
        assert False
    except AttributeError as e:
        assert "'z' is not a member" in str(e)

def test_immutable_frozen_member():
    Point = immutable('x, y, id_', name='Point')
    p = Point(1, 2, id_=17)
    assert p.id_ == 17
    p2 = p.set(x=3)
    assert p2.id_ == 17
    try:
        p.set(id_=18)
        assert False
    except AttributeError as e:
        assert 'Cannot set frozen members id_' in str(e)

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
    try:
        p.set(y=-3)
        assert False
    except Exception as e:
        assert 'Coordinates must be positive!' in str(e)

def test_immutable_string_members():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

def test_immutable_list_members():
    Point = immutable(['x', 'y'], name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

def test_immutable_members_with_commas():
    Point = immutable('x, y', name='Point')
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2

def test_immutable_no_members():
    Empty = immutable(name='Empty')
    e = Empty()
    assert repr(e) == 'Empty()'
    e2 = e.set()
    assert e2 is e

def test_immutable_repr_custom_name():
    Point = immutable('x, y', name='MyPoint')
    p = Point(1, 2)
    assert repr(p) == 'MyPoint(x=1, y=2)'

def test_immutable_set_frozen_multiple():
    Obj = immutable('a, b_, c, d_', name='Obj')
    o = Obj(1, 2, 3, 4)
    assert o.b_ == 2
    assert o.d_ == 4
    o2 = o.set(a=5, c=6)
    assert o2.a == 5
    assert o2.c == 6
    assert o2.b_ == 2
    assert o2.d_ == 4
    try:
        o.set(b_=7, d_=8)
        assert False
    except AttributeError as e:
        assert 'Cannot set frozen members' in str(e)
        assert 'b_' in str(e)
        assert 'd_' in str(e)


