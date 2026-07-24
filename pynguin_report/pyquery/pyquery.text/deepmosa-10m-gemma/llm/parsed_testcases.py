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
    var_5 = 'hello world'
    var_6 = 'next'
    var_7 = [var_5, var_2, var_6]
    var_8 = module_0._merge_original_parts(var_4)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'part1'
    var_1 = '   '
    var_2 = 'part2\n\npart3'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'part2 part3'
    var_5 = [var_0, var_4]
    var_6 = module_0._merge_original_parts(var_3)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0._merge_original_parts(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = True
    var_2 = 'end'
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = module_0._merge_original_parts(var_3)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'text'
    var_2 = 0
    var_3 = 'more text'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = [var_0, var_1, var_2, var_3]
    var_6 = module_0._merge_original_parts(var_4)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '   '
    var_1 = '  \t  '
    var_2 = 'valid'
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0._merge_original_parts(var_3)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_children. Retrieved 4/16 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 5/21 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 7/22 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 6/22 statements.
# Partially parsed test_extract_text_empty_node. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = 'div'
    var_3 = 'Hello '

def test_case_0():
    var_0 = 'br'
    var_1 = 'div'
    var_2 = 'Line1'
    var_3 = 'span'
    var_4 = 'Line2'

def test_case_0():
    var_0 = 'br'
    var_1 = 'div'
    var_2 = 'A'
    var_3 = 'p'
    var_4 = 'B'
    var_5 = '|'
    var_6 = '-'

def test_case_0():
    var_0 = 'br'
    var_1 = 'div'
    var_2 = 'A'
    var_3 = 'p'
    var_4 = 'B'
    var_5 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 9/43 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'extract_text_array'
    var_1 = None
    var_2 = '_merge_original_parts'
    var_3 = '_squash_artifical_nl'
    var_4 = '_strip_artifical_nl'
    var_5 = 'dummy'
    var_6 = '\n'
    var_7 = False
    var_8 = module_0.extract_text(var_5, var_6, var_6, var_7)
    assert var_8 == 'part1\npart2'



# Parsed testcases at query #4
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
    var_0 = 'only'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == ['only'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = 1
    var_2 = 'end'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['start', 1, 'end'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'content'
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._strip_artifical_nl(var_2)
    var_4 = bool(var_3 == ['content'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'content'
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3, var_0]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['content'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'start'
    var_2 = 0
    var_3 = 'end'
    var_4 = [var_0, var_0, var_1, var_2, var_3, var_0, var_0]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['start', 0, 'end'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [1, 2, 3])
    assert var_5 is True



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
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_block_elements. Retrieved 8/22 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 9/21 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

import pyquery.text as module_0

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'start'
    var_5 = 'span'
    var_6 = 'p'
    var_7 = module_0.extract_text_array(var_6)
    var_8 = bool(var_7 == ['start', 'child', ' tail'])
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = 'start'
    var_5 = 'span'
    var_6 = 'p'
    var_7 = False
    var_8 = module_0.extract_text_array(var_6, var_7, var_7)
    var_9 = bool(var_8 == [None, 'start', 'child', None])
    assert var_9 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = 'content'
    var_2 = 'p'
    var_3 = module_0.extract_text_array(var_2)
    var_4 = bool(var_3 == [True, 'content'])
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_extract_text_squash_space_true. Retrieved 1/5 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_extract_text_simple_string. Retrieved 2/12 statements.
# Partially parsed test_extract_text_with_children. Retrieved 4/16 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 2/12 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 2/13 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = 'div'
    var_3 = 'Hello '

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Part1'

def test_case_0():
    var_0 = 'div'
    var_1 = '  spaced  '
    var_2 = False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_extract_text_squash_space_true. Retrieved 5/11 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = '  text  '
    var_1 = None
    var_2 = '\n'
    var_3 = True
    var_4 = module_0.extract_text(var_1, var_2, var_2, var_3)
    assert var_4 == 'text'



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = bool(not True)
    assert var_0 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extract_text_basic_string. Retrieved 2/12 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/20 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/10 statements.
# Partially parsed test_extract_text_separator_tag. Retrieved 3/14 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 3/12 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Part 1'
    var_2 = 'b'
    var_3 = 'Part 2'
    var_4 = 'div'
    var_5 = ' End'
    var_6 = 'Part 1'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Text'
    var_2 = []

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = '|'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Inside'
    var_2 = 'div'
    var_3 = ' Outside'
    var_4 = 'Outside'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_extract_text_predicate_evaluates_to_true. Retrieved 13/19 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'part1'
    var_1 = None
    var_2 = 'part2'
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x
    var_5 = lambda x: x
    var_6 = lambda x: x
    var_7 = 'dummy_dom'
    var_8 = [var_7]
    var_9 = '\n'
    var_10 = ' '
    var_11 = True
    var_12 = module_0.extract_text(var_8, var_9, var_10, var_11)
    assert var_12 == 'part1 part2'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_nested_elements. Retrieved 4/17 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 3/15 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 3/14 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/13 statements.
# Partially parsed test_extract_text_with_none_parts. Retrieved 3/15 statements.


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
    var_1 = 'Content'
    var_2 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Word  '
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  Word  '
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Text'
    var_2 = '\n'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_extract_text_array_predicate_is_true. Retrieved 1/16 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_extract_text_simple_node. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_none_separator. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 2/4 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 3/5 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/5 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 6/12 statements.
# Partially parsed test_extract_text_empty_node. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'div'
    var_1 = 'start'
    var_2 = 'span'
    var_3 = 'middle'

def test_case_0():
    var_0 = 'p'
    var_1 = 'para'

def test_case_0():
    var_0 = 'div'
    var_1 = '  hello   '
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  hello   '
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = 'b'
    var_3 = 'bold'
    var_4 = 'div'
    var_5 = 'outer '

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = 'span'
    var_3 = 'b'
    var_4 = '|'



# Parsed testcases at query #17
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_extract_text_evaluates_true_at_line_11. Retrieved 6/15 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = '  text  '
    var_2 = [var_1]
    var_3 = 'some_dom'
    var_4 = True
    var_5 = module_0.extract_text(var_3, squash_space=var_4)
    assert var_5 == 'text'



# Parsed testcases at query #19
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = False
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)
    assert var_3 == 'part1part2'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_extract_text_predicate_true. Retrieved 3/4 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_extract_text_array_predicate_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_extract_text_array_predicate_false.




# Parsed testcases at query #23
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_simple_string. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_children. Retrieved 5/22 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_none_handling. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello World'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start '
    var_2 = 'span'
    var_3 = 'Middle'
    var_4 = 'Middle'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Part 1'

def test_case_0():
    var_0 = 'div'
    var_1 = '  spaced  '
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 7/15 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = lambda x: x
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = None
    var_5 = False
    var_6 = module_0.extract_text(var_4, squash_space=var_5)
    assert var_6 == ''



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_extract_text_predicate_true. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'part1'
    var_1 = 'part2'
    var_2 = True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_none_parts_squashed. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 3/7 statements.
# Partially parsed test_extract_text_complex_nesting. Retrieved 6/12 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 5/9 statements.
# Partially parsed test_extract_text_no_squash. Retrieved 4/8 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/3 statements.
# Partially parsed test_extract_text_with_multiple_children_and_tails. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Text'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Part 1'
    var_2 = 'Part 2'
    var_3 = 'div'
    var_4 = '\n'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Inside'
    var_2 = 'div'

def test_case_0():
    var_0 = 'span'
    var_1 = 'bold'
    var_2 = 'p'
    var_3 = 'start '
    var_4 = ' end'
    var_5 = 'div'

def test_case_0():
    var_0 = 'p'
    var_1 = 'A'
    var_2 = 'div'
    var_3 = '|'
    var_4 = '-'

def test_case_0():
    var_0 = 'p'
    var_1 = 'A'
    var_2 = 'div'
    var_3 = False

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'span'
    var_1 = 'one'
    var_2 = '! '
    var_3 = 'two'
    var_4 = '?'
    var_5 = 'div'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/12 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/19 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 2/14 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = 'b'
    var_3 = '!'
    var_4 = 'div'
    var_5 = 'Hello '

def test_case_0():
    var_0 = 'p'
    var_1 = 'Part1'

def test_case_0():
    var_0 = 'div'
    var_1 = None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_extract_text_array_predicate_false. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_extract_text_predicate_evaluates_to_true. Retrieved 2/3 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.extract_text(var_0)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_children_and_tails. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello '

def test_case_0():
    var_0 = 'p'
    var_1 = 'Line1'
    var_2 = None
    var_3 = 'div'
    var_4 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Part1'
    var_2 = 'Part2'
    var_3 = 'div'
    var_4 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
    var_2 = 'span'
    var_3 = 'B'
    var_4 = None
    var_5 = 'div'
    var_6 = None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 9/19 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 9/19 statements.
# Partially parsed test_extract_text_array_with_children_and_tails. Retrieved 12/25 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 9/19 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 13/26 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = 'SEPARATORS'
    var_3 = 'p'
    var_4 = [var_3]
    var_5 = 'INLINE_TAGS'
    var_6 = 'span'
    var_7 = [var_6]
    var_8 = module_0.extract_text_array(var_5)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'div'
    var_1 = 'hello'
    var_2 = 'SEPARATORS'
    var_3 = 'p'
    var_4 = [var_3]
    var_5 = 'INLINE_TAGS'
    var_6 = 'span'
    var_7 = [var_6]
    var_8 = module_0.extract_text_array(var_5)
    var_9 = bool(var_8 == ['hello'])
    assert var_9 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'start'
    var_5 = 'SEPARATORS'
    var_6 = 'p'
    var_7 = [var_6]
    var_8 = 'INLINE_TAGS'
    var_9 = 'span'
    var_10 = [var_9]
    var_11 = module_0.extract_text_array(var_8)
    var_12 = bool(var_11 == ['start', 'inner', ' tail'])
    assert var_12 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = 'content'
    var_2 = 'SEPARATORS'
    var_3 = 'p'
    var_4 = [var_3]
    var_5 = 'INLINE_TAGS'
    var_6 = 'span'
    var_7 = [var_6]
    var_8 = module_0.extract_text_array(var_5)
    var_9 = bool(var_8 == [True, 'content'])
    assert var_9 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = None
    var_3 = 'div'
    var_4 = 'start'
    var_5 = 'SEPARATORS'
    var_6 = 'p'
    var_7 = [var_6]
    var_8 = 'INLINE_TAGS'
    var_9 = 'span'
    var_10 = [var_9]
    var_11 = False
    var_12 = module_0.extract_text_array(var_8, var_11, var_11)
    var_13 = bool(var_12 == [None, 'start', 'inner', None])
    assert var_13 is True

def test_case_0():
    var_0 = lambda x: x
    var_1 = 'hidden'



# Parsed testcases at query #34
#--------------------------

# Failed to parse test_extract_text_array_predicate_true.




# Parsed testcases at query #35
#--------------------------

# Failed to parse test_extract_text_array_predicate_true.




####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_extract_text_simple_string. Retrieved 2/12 statements.
# Partially parsed test_extract_text_with_children. Retrieved 4/16 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 2/13 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = 'div'
    var_3 = 'Hello '
    var_4 = 'Hello'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Part1'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = '|'



# Parsed testcases at query #2
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello   world  '
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == ['hello world'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'part1'
    var_1 = 123
    var_2 = 'part2'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['part1', 123, 'part2'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = ' '
    var_2 = 'world'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['hello world'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 'c'
    var_4 = 'd'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._merge_original_parts(var_5)
    var_7 = bool(var_6 == ['a b', None, 'c d'])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = ' '
    var_1 = '\n'
    var_2 = '\t'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  start  '
    var_1 = True
    var_2 = '  middle  '
    var_3 = 0.5
    var_4 = '  end  '
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._merge_original_parts(var_5)
    var_7 = bool(var_6 == ['start', True, 'middle', 0.5, 'end'])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'line1\n'
    var_1 = 'line2'
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['line1 line2'])
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = '\n'
    var_2 = False
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)



# Parsed testcases at query #4
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
    var_4 = [var_0, var_1, var_2, var_1, var_3]
    var_5 = module_0._squash_artifical_nl(var_4)
    var_6 = bool(var_5 == [1, None, 2, None, 3])
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_extract_text_simple_string. Retrieved 10/18 statements.
# Partially parsed test_extract_text_with_children_and_tails. Retrieved 8/23 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 10/24 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 8/15 statements.


import re as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello World'
    var_2 = []
    var_3 = '\\s+'
    var_4 = module_0.compile(var_3)
    var_5 = 'span'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = 'br'
    var_9 = [var_8]

import re as module_0

def test_case_0():
    var_0 = '\\s+'
    var_1 = module_0.compile(var_0)
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = []
    var_5 = 'inner'
    var_6 = 'div'
    var_7 = 'start '

import re as module_0

def test_case_0():
    var_0 = '\\s+'
    var_1 = module_0.compile(var_0)
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = 'br'
    var_5 = [var_4]
    var_6 = 'div'
    var_7 = 'A'
    var_8 = 'B'
    var_9 = '\n'

import re as module_0

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = '\\s+'
    var_4 = module_0.compile(var_3)
    var_5 = 'span'
    var_6 = [var_5]
    var_7 = []



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_extract_text_predicate_true. Retrieved 7/8 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'part1'
    var_1 = None
    var_2 = 'part2'
    var_3 = [var_0, var_1, var_2]
    var_4 = '\n'
    var_5 = True
    var_6 = module_0.extract_text(var_3, var_4, var_4, var_5)



# Parsed testcases at query #7
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
    var_0 = 'only'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == ['only'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = 1
    var_2 = 'end'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['start', 1, 'end'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'content'
    var_2 = 1
    var_3 = [var_0, var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['content', 1])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'content'
    var_2 = None
    var_3 = [var_0, var_1, var_2, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [1, 'content'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'middle'
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._strip_artifical_nl(var_2)
    var_4 = bool(var_3 == ['middle'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'target'
    var_3 = False
    var_4 = [var_0, var_1, var_2, var_3, var_0]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['target', False])
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



# Parsed testcases at query #8
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = '\n'
    var_2 = '\n'
    var_3 = True
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_extract_text_simple. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_children. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 5/10 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 4/9 statements.
# Partially parsed test_extract_text_no_squash. Retrieved 6/11 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 2/4 statements.
# Partially parsed test_extract_text_whitespace_squashing. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = 'div'
    var_3 = 'Hello '

def test_case_0():
    var_0 = 'br'
    var_1 = 'div'
    var_2 = 'Line1'
    var_3 = '\n'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Paragraph'
    var_2 = 'div'
    var_3 = 'Start'
    var_4 = True

def test_case_0():
    var_0 = 'br'
    var_1 = 'div'
    var_2 = 'A'
    var_3 = '|'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Content'
    var_2 = 'div'
    var_3 = 'Start'
    var_4 = False
    var_5 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = ''

def test_case_0():
    var_0 = 'div'
    var_1 = '  Too   Much   Space  '
    var_2 = True



# Parsed testcases at query #10
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 3/4 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extract_text_simple_node. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/12 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 4/8 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 4/8 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 4/9 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 6/12 statements.
# Partially parsed test_extract_text_none_handling. Retrieved 5/9 statements.
# Partially parsed test_extract_text_empty_node. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello '
    var_2 = 'b'
    var_3 = 'World'
    var_4 = 'div'
    var_5 = ''

def test_case_0():
    var_0 = 'br'
    var_1 = 'div'
    var_2 = 'Part 1'
    var_3 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Extra   '
    var_2 = ''
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  Extra   '
    var_2 = ''
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Start '
    var_2 = 'b'
    var_3 = 'Middle'
    var_4 = 'div'
    var_5 = ''

def test_case_0():
    var_0 = 'div'
    var_1 = 'Block 1'
    var_2 = 'p'
    var_3 = 'Block 2'
    var_4 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_nested_structure. Retrieved 5/18 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Child'
    var_2 = 'div'
    var_3 = 'Text'
    var_4 = True
    var_5 = 'Text'
    var_6 = 'Child'
    var_7 = 'Tail'

def test_case_0():
    var_0 = lambda x: x
    var_1 = 'foo'

def test_case_0():
    var_0 = 'br'
    var_1 = None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_extract_text_array_predicate_is_true. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #15
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = ' '
    var_2 = 'world'
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = module_0.extract_text(var_3, squash_space=var_4)
    assert var_5 == ' hello '



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #17
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_extract_text_predicate_true. Retrieved 4/5 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = True
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_single_text_node. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_none_squashing. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_stripping_logic. Retrieved 3/16 statements.
# Partially parsed test_extract_text_array_inline_tags. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_complex_structure. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'div'
    var_1 = 'start'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'content'
    var_2 = True

def test_case_0():
    var_0 = 'b'
    var_1 = 'inner'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'root'
    var_2 = None
    var_3 = 'span'
    var_4 = 'child_text'
    var_5 = ' child_tail'
    var_6 = True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 4/16 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 3/14 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 6/19 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello World'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Part 1'
    var_2 = 'Part 2'
    var_3 = 'div'

def test_case_0():
    var_0 = 'p'
    var_1 = '  Extra   Spaces  '
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'Content'
    var_2 = 'div'
    var_3 = '|'
    var_4 = False
    var_5 = '|Content|'

def test_case_0():
    var_0 = 'div'
    var_1 = None



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_extract_text_predicate_false.




# Parsed testcases at query #23
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_extract_text_squash_space_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True
    var_1 = bool(True)
    assert var_1 is True



# Parsed testcases at query #25
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 6/13 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 6/13 statements.
# Partially parsed test_extract_text_array_with_children_and_tails. Retrieved 9/19 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 7/14 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 5/12 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = set()
    var_3 = 'span'
    var_4 = [var_3]
    var_5 = set(var_4)

def test_case_0():
    var_0 = 'div'
    var_1 = 'hello'
    var_2 = set()
    var_3 = 'span'
    var_4 = [var_3]
    var_5 = set(var_4)

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'parent'
    var_5 = set()
    var_6 = 'span'
    var_7 = [var_6]
    var_8 = set(var_7)

def test_case_0():
    var_0 = 'p'
    var_1 = 'content'
    var_2 = 'p'
    var_3 = {var_2}
    var_4 = 'span'
    var_5 = [var_4]
    var_6 = set(var_5)

def test_case_0():
    var_0 = 'span'
    var_1 = 'inline'
    var_2 = set()
    var_3 = 'span'
    var_4 = {var_3}

def test_case_0():
    var_0 = lambda x: x
    var_1 = 'ignore'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_extract_text_simple_node. Retrieved 2/12 statements.
# Partially parsed test_extract_text_with_nesting. Retrieved 4/16 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 4/19 statements.
# Partially parsed test_extract_text_with_none_as_newline. Retrieved 3/15 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 5/17 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = 'div'
    var_3 = 'Hello '

def test_case_0():
    var_0 = 'p'
    var_1 = 'Separator'
    var_2 = 'div'
    var_3 = 'Start'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Inner'
    var_2 = 'Outer'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Inner'
    var_2 = 'Outer'
    var_3 = ' | '
    var_4 = ' - '

def test_case_0():
    var_0 = 'span'
    var_1 = 'Inside'
    var_2 = ' After'
    var_3 = 'div'
    var_4 = 'Before '



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_simple_string. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_nesting. Retrieved 5/17 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 4/13 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'
    var_2 = 'span'
    var_3 = 'hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = 'div'
    var_3 = None
    var_4 = ' end'
    var_5 = 'inner'
    var_6 = 'end'

def test_case_0():
    var_0 = 'p'
    var_1 = 'part1'
    var_2 = '|'
    var_3 = '|part1|'

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'text'
    var_2 = 'div'
    var_3 = None
    var_4 = ' tail'



# Parsed testcases at query #29
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'part1\npart2'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_simple_string. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_nested_elements. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_none_as_newline. Retrieved 6/14 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 4/9 statements.
# Partially parsed test_extract_text_strips_whitespace. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello '
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Part 1'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Inner'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Outer'
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
    var_2 = None
    var_3 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Trim Me  '
    var_2 = None



# Parsed testcases at query #31
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_extract_text_array_predicate_true. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_extract_text_array_predicate_true. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = lambda : var_1



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_extract_text_array_empty_node. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_children_and_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_logic. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_no_strip_no_squash. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'parent'

def test_case_0():
    var_0 = 'span'
    var_1 = 'a'
    var_2 = None
    var_3 = 'div'
    var_4 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'a'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = False



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_extract_text_array_predicate_false.




