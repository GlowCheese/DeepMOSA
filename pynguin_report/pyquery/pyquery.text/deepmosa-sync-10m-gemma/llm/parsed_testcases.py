####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello  '
    var_1 = 123
    var_2 = 'world\n\nnext'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'hello'
    var_5 = 'world next'
    var_6 = [var_4, var_1, var_5]
    var_7 = module_0._merge_original_parts(var_3)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'part1'
    var_1 = '  part2  '
    var_2 = '\npart3\t'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'part2'
    var_5 = 'part3'
    var_6 = [var_0, var_4, var_5]
    var_7 = module_0._merge_original_parts(var_3)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'content'
    var_1 = ''
    var_2 = '  '
    var_3 = 'more content'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = [var_0, var_3]
    var_6 = module_0._merge_original_parts(var_4)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = module_0._merge_original_parts(var_3)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '   only whitespace   '
    var_1 = [var_0]
    var_2 = []
    var_3 = module_0._merge_original_parts(var_1)
    var_4 = bool(var_3 == var_2)
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = None
    var_2 = '  middle  '
    var_3 = True
    var_4 = '  end  '
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = 'middle'
    var_7 = 'end'
    var_8 = [var_0, var_1, var_6, var_3, var_7]
    var_9 = module_0._merge_original_parts(var_5)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0._merge_original_parts(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = False
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [None, 1, False])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = None
    var_2 = 'end'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['start', None, 'end'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'start'
    var_2 = 'end'
    var_3 = [var_0, var_1, var_0, var_2, var_0]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['start', None, 'end'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'start'
    var_2 = 1
    var_3 = 'end'
    var_4 = 2
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._strip_artifical_nl(var_5)
    var_7 = bool(var_6 == ['start', 1, 'end'])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b', 'c'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = None
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['start', None, 1])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'end'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['start' if False else None, 1, 'end'])
    assert var_5 is True
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0._strip_artifical_nl(var_6)
    var_8 = bool(var_7 == ['end'])
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'start'
    var_3 = 'end'
    var_4 = 2
    var_5 = [var_0, var_1, var_2, var_0, var_3, var_4, var_0]
    var_6 = module_0._strip_artifical_nl(var_5)
    var_7 = bool(var_6 == ['start', None, 'end'])
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_extract_text_simple_string. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_nested_elements. Retrieved 4/17 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 6/22 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/15 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello World'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start '
    var_2 = 'span'
    var_3 = 'Middle'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Part 1'
    var_2 = 'br'
    var_3 = 'span'
    var_4 = 'Part 2'
    var_5 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_none_stripping. Retrieved 6/17 statements.
# Partially parsed test_extract_text_array_squash_logic. Retrieved 6/17 statements.
# Partially parsed test_extract_text_array_separator_logic. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = 'start'
    var_5 = 'end'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = 'parent'
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'content'
    var_2 = None



# Parsed testcases at query #5
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b', 'c'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', None, 'b'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_1, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', None, 'b'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = [var_0, var_0, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)
    var_4 = bool(var_3 == [None, 'a'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = [var_0, var_1, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)
    var_4 = bool(var_3 == ['a', None])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    var_3 = bool(var_2 == [None])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_0, var_1, var_2, var_1, var_3]
    var_5 = module_0._squash_artifical_nl(var_4)
    var_6 = bool(var_5 == ['a', None, 'b', None, 'c'])
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  Hello  '
    var_1 = 123
    var_2 = '  World \n next line  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['Hello', 123, 'World next line'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'Part 1'
    var_1 = '  Part 2  '
    var_2 = 'Part 3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['Part 1', 'Part 2', 'Part 3'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'Content'
    var_2 = '   '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['Content'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = None
    var_2 = 'B'
    var_3 = False
    var_4 = 'C'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._merge_original_parts(var_5)
    var_7 = bool(var_6 == ['A', None, 'B', False, 'C'])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'Line 1\n\nLine 2'
    var_1 = '  Trailing space  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['Line 1 Line 2', 'Trailing space'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #7
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello  '
    var_1 = 123
    var_2 = '  world \n  '
    var_3 = [var_0, var_1, var_2]
    var_4 = 'hello'
    var_5 = 'world'
    var_6 = [var_4, var_1, var_5]
    var_7 = module_0._merge_original_parts(var_3)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'part1'
    var_1 = '  part2  '
    var_2 = '\npart3\t'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'part2'
    var_5 = 'part3'
    var_6 = [var_0, var_4, var_5]
    var_7 = module_0._merge_original_parts(var_3)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = '   '
    var_2 = 'content'
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0._merge_original_parts(var_3)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = False
    var_4 = 'c'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = [var_0, var_1, var_2, var_3, var_4]
    var_7 = module_0._merge_original_parts(var_5)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  only one  '
    var_1 = [var_0]
    var_2 = 'only one'
    var_3 = [var_2]
    var_4 = module_0._merge_original_parts(var_1)
    var_5 = bool(var_4 == var_3)
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0._merge_original_parts(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.extract_text(var_0)
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['a b'])
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello  '
    var_1 = 'world  '
    var_2 = '\n\n'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['hello world'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'part1'
    var_1 = 123
    var_2 = '  part2  '
    var_3 = True
    var_4 = ' \n part3 '
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._merge_original_parts(var_5)
    var_7 = bool(var_6 == ['part1', 123, 'part2', True, 'part3'])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = '  '
    var_2 = 'content'
    var_3 = ' '
    var_4 = [var_0, var_1, var_2, var_0, var_3]
    var_5 = module_0._merge_original_parts(var_4)
    var_6 = bool(var_5 == ['content'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = False
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [1, None, False])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  line\nbreak  '
    var_1 = '   more   space  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['line break more space'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'only one'
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == ['only one'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_extract_text_predicate_true. Retrieved 4/5 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = '\n'
    var_3 = module_0.extract_text(var_0, var_2, var_2, var_1)



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    pass

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    assert var_1 is True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_extract_text_basic_structure. Retrieved 4/17 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 6/18 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 2/11 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/13 statements.
# Partially parsed test_extract_text_complex_nesting. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = 'span'
    var_3 = 'World'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Part 1'
    var_2 = 'b'
    var_3 = 'Part 2'
    var_4 = '|'
    var_5 = '@'
    var_6 = '@'

def test_case_0():
    var_0 = 'div'
    var_1 = ''

def test_case_0():
    var_0 = 'div'
    var_1 = '  Space  '
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start '
    var_2 = 'span'
    var_3 = 'Middle'
    var_4 = 'b'
    var_5 = 'End'
    var_6 = 'Start'
    var_7 = 'Middle'
    var_8 = 'End'



# Parsed testcases at query #14
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = '|'
    var_2 = '-'
    var_3 = False
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)
    assert var_4 == ''



# Parsed testcases at query #15
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_extract_text_basic_string. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_none_separator. Retrieved 2/10 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/9 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello World'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = 'Part 1'
    var_3 = 'b'
    var_4 = 'Part 2'
    var_5 = 'Part 1'
    var_6 = 'Part 2'
    var_7 = 'Part 3'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 5/17 statements.
# Partially parsed test_extract_text_array_with_text_and_children. Retrieved 6/21 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 3/15 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = 'module'
    var_3 = 'p'
    var_4 = 'span'

def test_case_0():
    var_0 = 'p'
    var_1 = 'span'
    var_2 = 'hello'
    var_3 = ' world'
    var_4 = 'div'
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'span'
    var_2 = 'content'

def test_case_0():
    var_0 = lambda x: x



# Parsed testcases at query #18
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'dummy'
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'a'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_extract_text_basic_structure. Retrieved 9/23 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 10/25 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 6/18 statements.
# Partially parsed test_extract_text_no_squash. Retrieved 10/25 statements.


import re as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = '\\s+'
    var_5 = module_0.compile(var_4)
    var_6 = 'div'
    var_7 = 'Hello'
    var_8 = ' World'

import re as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = '\\s+'
    var_5 = module_0.compile(var_4)
    var_6 = 'div'
    var_7 = 'Part1'
    var_8 = 'Part2'
    var_9 = '|'

import re as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '\\s+'
    var_3 = module_0.compile(var_2)
    var_4 = 'div'
    var_5 = None

import re as module_0

def test_case_0():
    var_0 = []
    var_1 = 'span'
    var_2 = [var_1]
    var_3 = '\\s+'
    var_4 = module_0.compile(var_3)
    var_5 = 'div'
    var_6 = 'Start'
    var_7 = 'Middle'
    var_8 = False
    var_9 = '\n'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_extract_text_basic_structure. Retrieved 10/36 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 9/26 statements.


def test_case_0():
    var_0 = 'mock_module'
    var_1 = 'p'
    var_2 = 'div'
    var_3 = 'span'
    var_4 = 'b'
    var_5 = '\\s+'
    var_6 = 'World'
    var_7 = 'Hello '
    var_8 = '\n'
    var_9 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'span'
    var_2 = '\\s+'
    var_3 = 'Inner'
    var_4 = 'div'
    var_5 = 'Outer '
    var_6 = '\n'
    var_7 = '|'
    var_8 = True
    var_9 = '|'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 10/18 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'part1'
    var_1 = 'part2'
    var_2 = [var_0, var_1]
    var_3 = lambda x: x
    var_4 = lambda x: x
    var_5 = lambda x: x
    var_6 = None
    var_7 = '\n'
    var_8 = False
    var_9 = module_0.extract_text(var_6, var_7, var_7, var_8)
    assert var_9 == 'part1part2'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_extract_text_basic_string. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_children. Retrieved 4/16 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 6/16 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/12 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start '
    var_2 = 'span'
    var_3 = 'Middle'

def test_case_0():
    var_0 = 'p'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = 'Content'
    var_5 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = ' Hello '
    var_2 = False

def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 4/10 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'extract_text_array'
    var_1 = []
    var_2 = False
    var_3 = module_0.extract_text(var_1, squash_space=var_2)
    assert var_3 == 'test'



# Parsed testcases at query #24
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_extract_text_predicate_true. Retrieved 12/25 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'mock_module'
    var_1 = None
    var_2 = True
    var_3 = 'text'
    var_4 = [var_1, var_2, var_3]
    var_5 = globals()
    var_6 = 'extract_text_array'
    var_7 = '_merge_original_parts'
    var_8 = '_squash_artifical_nl'
    var_9 = '_strip_artifical_nl'
    var_10 = '\n'
    var_11 = module_0.extract_text(var_1, var_10, var_10, var_2)
    assert var_11 == '\n\ntext'



# Parsed testcases at query #26
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #27
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_extract_text_predicate_evaluates_to_true. Retrieved 2/3 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.extract_text(var_0)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_extract_text_predicate_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_extract_text_array_predicate_true.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_extract_text_basic_string. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_none_as_newline. Retrieved 5/17 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 5/21 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/10 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'Part 1'
    var_3 = 'span'
    var_4 = 'Part 2'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'
    var_2 = 'p'
    var_3 = 'Content'
    var_4 = '|'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = False



# Parsed testcases at query #32
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_extract_text_array_predicate_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_extract_text_simple_string. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 5/17 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 3/12 statements.
# Partially parsed test_extract_text_with_none_elements. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Part1'
    var_2 = 'span'
    var_3 = 'Part2'
    var_4 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'span'
    var_3 = 'Middle'
    var_4 = '\n'



# Parsed testcases at query #35
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'part1\npart2'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/6 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 2/13 statements.
# Partially parsed test_extract_text_array_with_structure. Retrieved 2/15 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/11 statements.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = 'span'
    var_1 = True

def test_case_0():
    var_0 = 'b'
    var_1 = True

def test_case_0():
    var_0 = 'p'
    var_1 = True
    var_2 = 'content'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 7/13 statements.
# Partially parsed test_extract_text_array_with_text_and_children. Retrieved 10/24 statements.
# Partially parsed test_extract_text_array_with_separators. Retrieved 5/16 statements.
# Partially parsed test_extract_text_array_no_strip_no_squash. Retrieved 7/18 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = 'p'
    var_3 = [var_2]
    var_4 = 'span'
    var_5 = 'b'
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 'p'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = 'hello'
    var_5 = 'b'
    var_6 = 'world'
    var_7 = '!'
    var_8 = 'div'
    var_9 = 'start '

def test_case_0():
    var_0 = 'p'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = 'content'

def test_case_0():
    var_0 = 'p'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = 'div'
    var_5 = 'a'
    var_6 = False

def test_case_0():
    var_0 = lambda x: x



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello   '
    var_1 = 'world'
    var_2 = 123
    var_3 = '  \n  next  '
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._merge_original_parts(var_4)
    var_6 = bool(var_5 == ['hello world', 123, 'next'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'part1'
    var_1 = '   part2   '
    var_2 = 'part3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['part1 part2 part3'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = '  '
    var_2 = 'content'
    var_3 = ' '
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._merge_original_parts(var_4)
    var_6 = bool(var_5 == ['content'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = False
    var_3 = [var_0, var_1, var_2, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [None, True, False, 0])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  leading'
    var_1 = 'middle   '
    var_2 = 'trailing  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['leading middle trailing'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._strip_artifical_nl(var_2)
    var_4 = bool(var_3 == [None, 1, None])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = 1
    var_2 = 2
    var_3 = 'end'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['start', 1, 2, 'end'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'start'
    var_2 = 1
    var_3 = 2
    var_4 = 'end'
    var_5 = [var_0, var_1, var_2, var_3, var_4, var_0]
    var_6 = module_0._strip_artifical_nl(var_5)
    var_7 = bool(var_6 == ['start', 1, 2, 'end'])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'start'
    var_2 = 1
    var_3 = 'end'
    var_4 = [var_0, var_0, var_1, var_2, var_3, var_0, var_0]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['start', 1, 'end'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b', 'c'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [1, 2, 3])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', None, 'b'])
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello  '
    var_1 = 'world'
    var_2 = '  \n  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['hello world'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'part1'
    var_1 = 123
    var_2 = 'part2 '
    var_3 = 'sub-list'
    var_4 = [var_3]
    var_5 = '  part3  '
    var_6 = [var_0, var_1, var_2, var_4, var_5]
    var_7 = module_0._merge_original_parts(var_6)
    var_8 = bool(var_7 == ['part1', 123, 'part2', ['sub-list'], 'part3'])
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'content'
    var_2 = '   '
    var_3 = 'more content'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._merge_original_parts(var_4)
    var_6 = bool(var_5 == ['content', 'more content'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [1, 2, 3])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  '
    var_1 = '\n\t'
    var_2 = ' '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = 'key'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = 'c'
    var_7 = [var_0, var_1, var_2, var_5, var_6]
    var_8 = module_0._merge_original_parts(var_7)
    var_9 = bool(var_8 == ['a', None, 'b', {'key': 'val'}, 'c'])
    assert var_9 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_nested_elements. Retrieved 5/19 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 4/19 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 4/18 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello World'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = 'Part 1 '
    var_3 = 'b'
    var_4 = 'Part 2'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'
    var_2 = 'span'
    var_3 = 'Text'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = 'Hello'
    var_3 = False

def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #5
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b', 'c'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', None, 'b'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_1, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', None, 'b'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = [var_0, var_0, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)
    var_4 = bool(var_3 == [None, 'a'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = [var_0, var_1, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)
    var_4 = bool(var_3 == ['a', None])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    var_3 = bool(var_2 == [None])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_1, var_3, var_1, var_4]
    var_6 = module_0._squash_artifical_nl(var_5)
    var_7 = bool(var_6 == [1, None, 2, None, 3, None, 4])
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #7
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = '\n'
    var_2 = True
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = True
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_extract_text_basic. Retrieved 11/25 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 11/27 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 11/25 statements.
# Partially parsed test_extract_text_whitespace_squashing. Retrieved 8/20 statements.
# Partially parsed test_extract_text_no_squash. Retrieved 9/21 statements.


import re as module_0

def test_case_0():
    var_0 = 'br'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = 'b'
    var_4 = 'i'
    var_5 = [var_2, var_3, var_4]
    var_6 = '\\s+'
    var_7 = module_0.compile(var_6)
    var_8 = 'div'
    var_9 = 'Hello'
    var_10 = ' World'

import re as module_0

def test_case_0():
    var_0 = 'br'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = '\\s+'
    var_5 = module_0.compile(var_4)
    var_6 = 'div'
    var_7 = 'Part1'
    var_8 = 'Part2'
    var_9 = '\n'
    var_10 = '|'

import re as module_0

def test_case_0():
    var_0 = 'br'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = '\\s+'
    var_5 = module_0.compile(var_4)
    var_6 = 'div'
    var_7 = 'Start'
    var_8 = 'p'
    var_9 = 'Middle'
    var_10 = '\n'

import re as module_0

def test_case_0():
    var_0 = 'br'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = '\\s+'
    var_5 = module_0.compile(var_4)
    var_6 = 'div'
    var_7 = '  Too   Much   '

import re as module_0

def test_case_0():
    var_0 = 'br'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = '\\s+'
    var_5 = module_0.compile(var_4)
    var_6 = 'div'
    var_7 = '  Keep   Space  '
    var_8 = False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = False
    var_1 = lambda x: x
    var_2 = var_1(var_0)
    assert var_2 is False



# Parsed testcases at query #12
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_extract_text_basic_structure. Retrieved 11/28 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 9/27 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 7/23 statements.
# Partially parsed test_extract_text_empty_node. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'Hello'
    var_3 = 'span'
    var_4 = 'b'
    var_5 = 'br'
    var_6 = 're'
    var_7 = __import__(var_6)
    var_8 = '\\s+'
    var_9 = '|'
    var_10 = '@'

def test_case_0():
    var_0 = 'p'
    var_1 = 'div'
    var_2 = 'br'
    var_3 = 're'
    var_4 = __import__(var_3)
    var_5 = '\\s+'
    var_6 = 'Part1'
    var_7 = 'BLOCK'
    var_8 = 'SEP'
    var_9 = 'SEP'

def test_case_0():
    var_0 = 'p'
    var_1 = 'div'
    var_2 = 're'
    var_3 = __import__(var_2)
    var_4 = '\\s+'
    var_5 = '  Hello   \n  World  '
    var_6 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'div'
    var_2 = 're'
    var_3 = __import__(var_2)
    var_4 = '\\s+'



# Parsed testcases at query #14
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'test'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_children_and_separators. Retrieved 5/11 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 4/9 statements.
# Partially parsed test_extract_text_complex_nesting. Retrieved 7/15 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/3 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'p'
    var_1 = 'First'
    var_2 = 'b'
    var_3 = 'Bold'
    var_4 = 'div'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Part 1'
    var_2 = 'div'
    var_3 = False

def test_case_0():
    var_0 = 'i'
    var_1 = 'italic'
    var_2 = 'span'
    var_3 = 'middle '
    var_4 = 'p'
    var_5 = 'outer '
    var_6 = 'div'
    var_7 = 'outer'
    var_8 = 'middle'
    var_9 = 'italic'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'span'
    var_1 = '  too   many   spaces  '
    var_2 = 'div'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_extract_text_simple_node. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_nested_nodes. Retrieved 6/11 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 6/11 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 6/11 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 9/16 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello '
    var_2 = 'b'
    var_3 = 'World'
    var_4 = 'div'
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Part 1'
    var_2 = 'span'
    var_3 = 'Part 2'
    var_4 = 'div'
    var_5 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = 'World'
    var_3 = 'div'
    var_4 = None
    var_5 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = 'p'
    var_3 = 'World'
    var_4 = 'div'
    var_5 = None
    var_6 = 'A'
    var_7 = 'br'
    var_8 = '|'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Start'
    var_2 = ' End'
    var_3 = 'div'
    var_4 = None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_extract_text_basic_elements. Retrieved 10/27 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 9/26 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 5/19 statements.
# Partially parsed test_extract_text_no_squash. Retrieved 7/21 statements.


def test_case_0():
    var_0 = '\\s+'
    var_1 = 'br'
    var_2 = 'p'
    var_3 = 'span'
    var_4 = 'b'
    var_5 = 'i'
    var_6 = 'div'
    var_7 = 'Hello '
    var_8 = 'World'
    var_9 = '!'

def test_case_0():
    var_0 = '\\s+'
    var_1 = 'br'
    var_2 = 'div'
    var_3 = 'Part1'
    var_4 = 'span'
    var_5 = 'Part2'
    var_6 = 'Part3'
    var_7 = '|'
    var_8 = '-'

def test_case_0():
    var_0 = '\\s+'
    var_1 = 'div'
    var_2 = ''
    var_3 = []
    var_4 = None

def test_case_0():
    var_0 = '\\s+'
    var_1 = 'br'
    var_2 = 'div'
    var_3 = '  spaced  '
    var_4 = []
    var_5 = None
    var_6 = False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 8/15 statements.
# Partially parsed test_extract_text_array_with_text_and_children. Retrieved 7/22 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 5/16 statements.
# Partially parsed test_extract_text_array_squash_logic. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = 'p'
    var_3 = 'br'
    var_4 = [var_2, var_3]
    var_5 = 'span'
    var_6 = 'b'
    var_7 = [var_5, var_6]

def test_case_0():
    var_0 = 'p'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = 'hello'
    var_5 = 'div'
    var_6 = 'start'

def test_case_0():
    var_0 = 'p'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = 'content'

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'div'
    var_3 = 'a'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_extract_text_predicate_true. Retrieved 2/3 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.extract_text(var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_extract_text_array_simple_text. Retrieved 7/19 statements.
# Partially parsed test_extract_text_array_with_children_and_tails. Retrieved 6/23 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 1/8 statements.
# Partially parsed test_extract_text_array_squash_logic. Retrieved 6/22 statements.
# Partially parsed test_extract_text_array_none_handling. Retrieved 6/20 statements.
# Partially parsed test_extract_text_array_no_strip_no_squash. Retrieved 8/22 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'div'
    var_2 = [var_0, var_1]
    var_3 = 'span'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = 'Hello'

def test_case_0():
    var_0 = 'p'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = 'Inner'
    var_5 = 'Start '

def test_case_0():
    var_0 = lambda x: x

def test_case_0():
    var_0 = 'div'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = 'B'
    var_5 = 'A'

def test_case_0():
    var_0 = 'div'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = 'B'
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = 'B'
    var_5 = 'C'
    var_6 = 'A'
    var_7 = False



# Parsed testcases at query #22
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_extract_text_simple_string. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_children. Retrieved 4/16 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 6/19 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 4/16 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello World'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start '
    var_2 = 'span'
    var_3 = 'Middle'

def test_case_0():
    var_0 = 'p'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = 'First'
    var_5 = 'Second'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Line1\n'
    var_2 = 'Line2'
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None



# Parsed testcases at query #24
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'content'
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda dom, squash_artifical_nl: var_3
    var_5 = lambda x: x
    var_6 = lambda x: x
    var_7 = lambda x: x
    var_8 = None
    var_9 = True
    var_10 = '\n'
    var_11 = '\n'
    var_12 = module_0.extract_text(var_8, var_10, var_11, var_9)
    assert var_12 == 'content'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_extract_text_array_predicate_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_extract_text_predicate_evaluates_to_true. Retrieved 2/3 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.extract_text(var_0)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_extract_text_array_predicate_is_true. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #28
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_extract_text_predicate_true. Retrieved 6/24 statements.


def test_case_0():
    var_0 = 'extract_text_array'
    var_1 = '_merge_original_parts'
    var_2 = '_squash_artifical_nl'
    var_3 = '_strip_artifical_nl'
    var_4 = '\n'
    var_5 = True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_extract_text_simple_string. Retrieved 2/13 statements.
# Partially parsed test_extract_text_with_children. Retrieved 5/19 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 12/28 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello World'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = 'Hello '
    var_3 = 'b'
    var_4 = 'World'

import re as module_0

def test_case_0():
    var_0 = 'br'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = 'b'
    var_4 = 'i'
    var_5 = [var_2, var_3, var_4]
    var_6 = '\\s+'
    var_7 = module_0.compile(var_6)
    var_8 = 'div'
    var_9 = 'Part 1'
    var_10 = '\n'
    var_11 = '|'

import re as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '\\s+'
    var_3 = module_0.compile(var_2)
    var_4 = 'div'
    var_5 = 'p'
    var_6 = 'Line 1'
    var_7 = False



# Parsed testcases at query #31
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = True
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_extract_text_array_predicate_false.




# Parsed testcases at query #33
#--------------------------

# Partially parsed test_extract_text_array_predicate_false. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_children. Retrieved 4/17 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 3/12 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 3/12 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start '
    var_2 = 'span'
    var_3 = 'Middle'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Line1'
    var_2 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Spaced   Text  '
    var_2 = True

def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_extract_text_array_predicate_true. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'div'



