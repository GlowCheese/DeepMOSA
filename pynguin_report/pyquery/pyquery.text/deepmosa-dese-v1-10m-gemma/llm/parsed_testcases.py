####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  Hello   '
    var_1 = 123
    var_2 = '  World\n\n  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'Part1'
    var_1 = '  Part2  '
    var_2 = 'Part3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = '  '
    var_2 = 'Content'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'Text'
    var_2 = 0.5
    var_3 = '  More Text  '
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._merge_original_parts(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)

def test_case_0():
    var_0 = '\tTab\n'
    var_1 = '\rNewline\r'
    var_2 = '  Space  '
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_extract_text_simple_string. Retrieved 2/12 statements.
# Partially parsed test_extract_text_with_children_and_whitespace. Retrieved 5/17 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 6/19 statements.
# Partially parsed test_extract_text_empty_node. Retrieved 2/12 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 9/23 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = 'div'
    var_3 = 'Hello '
    var_4 = None

def test_case_0():
    var_0 = 'br'
    var_1 = 'div'
    var_2 = 'Line1'
    var_3 = None
    var_4 = '\n'
    var_5 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'End'
    var_2 = 'span'
    var_3 = 'Middle'
    var_4 = '!'
    var_5 = 'div'
    var_6 = 'Start '
    var_7 = '\n'
    var_8 = ' '



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_extract_text_simple_string. Retrieved 2/10 statements.
# Partially parsed test_extract_text_with_children. Retrieved 4/14 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 3/13 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 6/16 statements.
# Partially parsed test_extract_text_empty_node. Retrieved 2/10 statements.


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

def test_case_0():
    var_0 = 'span'
    var_1 = 'Part'
    var_2 = 'div'
    var_3 = 'Start '
    var_4 = None
    var_5 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 4/11 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 5/12 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 5/12 statements.
# Partially parsed test_extract_text_array_with_children_and_tails. Retrieved 8/18 statements.
# Partially parsed test_extract_text_array_squash_logic. Retrieved 5/12 statements.
# Partially parsed test_extract_text_array_no_strip_and_no_squash. Retrieved 6/13 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'div'
    var_1 = 'hello'
    var_2 = []
    var_3 = 'span'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'p'
    var_1 = 'content'
    var_2 = 'p'
    var_3 = [var_2]
    var_4 = []

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'start'
    var_5 = []
    var_6 = 'span'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = 'span'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = 'span'
    var_4 = [var_3]
    var_5 = False

def test_case_0():
    var_0 = lambda x: x
    var_1 = 'ignored'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_extract_text_basic_structure. Retrieved 5/18 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 4/16 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 2/12 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 3/13 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'Hello'
    var_3 = '\n'
    var_4 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'
    var_2 = '\n'
    var_3 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = ''

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   '
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   '
    var_2 = False



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #7
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #8
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0._strip_artifical_nl(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = 'middle'
    var_2 = 'end'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'start'
    var_2 = 'middle'
    var_3 = 'end'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = 'middle'
    var_2 = 'end'
    var_3 = None
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'start'
    var_2 = 'middle'
    var_3 = 'end'
    var_4 = [var_0, var_1, var_2, var_3, var_0]
    var_5 = module_0._strip_artifical_nl(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'start'
    var_3 = 2
    var_4 = 'end'
    var_5 = True
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_0, var_5]
    var_7 = module_0._strip_artifical_nl(var_6)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_0, var_1, var_2, var_0, var_0]
    var_4 = module_0._strip_artifical_nl(var_3)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_extract_text_simple_node. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_separator_true. Retrieved 4/9 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 6/11 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/3 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 7/12 statements.
# Partially parsed test_extract_text_no_squash. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'part1'
    var_2 = 'part2'
    var_3 = 'div'

def test_case_0():
    var_0 = 'p'
    var_1 = 'content'
    var_2 = 'div'
    var_3 = '\n\ncontent\n\n'

def test_case_0():
    var_0 = 'span'
    var_1 = '  leading  '
    var_2 = 'b'
    var_3 = 'middle'
    var_4 = '  trailing  '
    var_5 = 'div'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'span'
    var_1 = 'A'
    var_2 = 'p'
    var_3 = 'B'
    var_4 = 'div'
    var_5 = '|'
    var_6 = '#'

def test_case_0():
    var_0 = 'span'
    var_1 = '  space  '
    var_2 = 'div'
    var_3 = False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_extract_text_simple_node. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_nested_nodes. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 2/4 statements.
# Partially parsed test_extract_text_no_squash. Retrieved 3/5 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 5/11 statements.
# Partially parsed test_extract_text_empty_node. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = 'i'
    var_3 = 'italic'
    var_4 = 'div'

def test_case_0():
    var_0 = 'span'
    var_1 = 'part1'
    var_2 = 'part2'
    var_3 = 'p'
    var_4 = '|'

def test_case_0():
    var_0 = 'span'
    var_1 = 'start'
    var_2 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = '  multiple   spaces  '

def test_case_0():
    var_0 = 'div'
    var_1 = '  keep  spaces  '
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = 'b'
    var_3 = ' World'
    var_4 = 'div'

def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_extract_text_array_predicate_true. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #13
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_1, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = [var_0, var_1, var_2, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = [var_0, var_1, var_2, var_1, var_1, var_3, var_1, var_4, var_1, var_1, var_1]
    var_6 = module_0._squash_artifical_nl(var_5)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/12 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 4/8 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 5/11 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 6/12 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 2/4 statements.
# Partially parsed test_extract_text_complex_nesting. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello World'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = 'b'
    var_3 = 'World'
    var_4 = 'p'
    var_5 = ''

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = 'p'
    var_3 = 'Start'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Block'
    var_2 = 'p'
    var_3 = ''

def test_case_0():
    var_0 = 'span'
    var_1 = 'Part 1'
    var_2 = 'Part 2'
    var_3 = 'p'
    var_4 = ''

def test_case_0():
    var_0 = 'span'
    var_1 = 'Part 1'
    var_2 = 'Part 2'
    var_3 = 'p'
    var_4 = ''
    var_5 = False

def test_case_0():
    var_0 = 'p'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Deep'
    var_2 = 'b'
    var_3 = 'Bold'
    var_4 = 'div'
    var_5 = 'Root '



# Parsed testcases at query #15
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_children_and_tails. Retrieved 6/12 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_elements_adding_newlines. Retrieved 3/8 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 6/13 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 5/10 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 5/10 statements.
# Partially parsed test_extract_text_empty_node. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_none_text. Retrieved 4/8 statements.


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
    var_0 = 'br'
    var_1 = 'div'
    var_2 = 'Part1'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Inner'
    var_2 = 'Outer'

def test_case_0():
    var_0 = 'span'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'div'
    var_4 = 'Start '
    var_5 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'A'
    var_2 = 'div'
    var_3 = 'Start '
    var_4 = False

def test_case_0():
    var_0 = 'br'
    var_1 = 'div'
    var_2 = 'A'
    var_3 = '|'
    var_4 = '-'

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = None
    var_2 = 'div'
    var_3 = 'Root'



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

# Partially parsed test_extract_text_squash_space_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #19
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = '\n'
    var_2 = True
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/4 statements.
# Partially parsed test_extract_text_nested_elements. Retrieved 4/8 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 6/11 statements.
# Partially parsed test_extract_text_with_tails. Retrieved 5/9 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 6/11 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 7/13 statements.
# Partially parsed test_extract_text_empty_node. Retrieved 3/5 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 7/12 statements.


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
    var_1 = 'Part 1'
    var_2 = 'Part 2'
    var_3 = 'div'
    var_4 = ''
    var_5 = '\n'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Start'
    var_2 = ' End'
    var_3 = 'div'
    var_4 = ''

def test_case_0():
    var_0 = 'p'
    var_1 = 'Line 1'
    var_2 = 'Line 2'
    var_3 = 'div'
    var_4 = ''
    var_5 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'Line 1'
    var_2 = 'Line 2'
    var_3 = 'div'
    var_4 = ''
    var_5 = False
    var_6 = 'Line 1\nLine 2'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []

def test_case_0():
    var_0 = 'p'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'div'
    var_4 = ''
    var_5 = ' | '
    var_6 = ' - '



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_extract_text_array_predicate_false. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #22
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_extract_text_predicate_true. Retrieved 2/3 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.extract_text(var_0)



# Parsed testcases at query #24
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = False
    var_3 = module_0.extract_text(var_1, squash_space=var_2)



# Parsed testcases at query #25
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'hello'



# Parsed testcases at query #26
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'part1part2'



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_extract_text_array_predicate_is_not_callable.




# Parsed testcases at query #28
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 3/17 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'dummy'
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'part1part2'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_extract_text_squash_space_true_predicate. Retrieved 3/6 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'some_dom'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_extract_text_array_predicate_false. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #32
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = lambda x: x
    var_2 = lambda x: x
    var_3 = None
    var_4 = True
    var_5 = 'text'
    var_6 = [var_3, var_4, var_5]
    var_7 = lambda dom, squash_artifical_nl: var_6
    var_8 = []
    var_9 = False
    var_10 = module_0.extract_text(var_8, squash_space=var_9)
    assert var_10 == '\n\ntext'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_extract_text_array_predicate_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None



# Parsed testcases at query #34
#--------------------------

# Failed to parse test_extract_text_array_predicate_true.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_extract_text_array_predicate_true. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #36
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_extract_text_simple_node. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_children. Retrieved 5/11 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 5/11 statements.
# Partially parsed test_extract_text_with_block_elements_newline. Retrieved 5/11 statements.
# Partially parsed test_extract_text_with_tails. Retrieved 4/9 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 5/11 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = 'Hello '
    var_3 = 'em'
    var_4 = 'World'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'
    var_2 = 'span'
    var_3 = 'Line 1'
    var_4 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'Block 1'
    var_3 = 'Block 2'
    var_4 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start '
    var_2 = 'span'
    var_3 = 'Middle'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'Part 1'
    var_3 = 'Part 2'
    var_4 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'Part 1'
    var_3 = 'Part 2'
    var_4 = False



# Parsed testcases at query #2
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello  '
    var_1 = 'world'
    var_2 = '\n\tnext  '
    var_3 = [var_0, var_1, var_2]
    var_4 = 'hello world next'
    var_5 = [var_4]
    var_6 = module_0._merge_original_parts(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start '
    var_1 = 123
    var_2 = ' middle '
    var_3 = True
    var_4 = ' end '
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = 'start'
    var_7 = 'middle'
    var_8 = 'end'
    var_9 = [var_6, var_1, var_7, var_3, var_8]
    var_10 = module_0._merge_original_parts(var_5)

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = '  '
    var_2 = 'content'
    var_3 = '   '
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = [var_2]
    var_6 = module_0._merge_original_parts(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = False
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = module_0._merge_original_parts(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = ' '
    var_1 = '\n'
    var_2 = '\t  \n'
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = module_0._merge_original_parts(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = '  part1  '
    var_1 = 'part2'
    var_2 = 42
    var_3 = '  part3  '
    var_4 = None
    var_5 = 'part4 '
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 'part1 part2'
    var_8 = 'part3'
    var_9 = 'part4'
    var_10 = [var_7, var_2, var_8, var_4, var_9]
    var_11 = module_0._merge_original_parts(var_6)



# Parsed testcases at query #3
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_extract_text_squash_space_true. Retrieved 5/7 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'part1'
    var_1 = 'part2'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.extract_text(var_2, squash_space=var_3)



# Parsed testcases at query #5
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._strip_artifical_nl(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = 1
    var_2 = 'end'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'start'
    var_2 = 1
    var_3 = 'end'
    var_4 = [var_0, var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = 1
    var_2 = 'end'
    var_3 = None
    var_4 = [var_0, var_1, var_2, var_3, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'start'
    var_2 = 1
    var_3 = 'end'
    var_4 = [var_0, var_1, var_2, var_3, var_0]
    var_5 = module_0._strip_artifical_nl(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'middle'
    var_3 = 1
    var_4 = [var_0, var_1, var_2, var_3, var_0]
    var_5 = module_0._strip_artifical_nl(var_4)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 6/13 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 6/13 statements.
# Partially parsed test_extract_text_array_with_children_and_tails. Retrieved 9/19 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 6/13 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 10/20 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = 'p'
    var_3 = [var_2]
    var_4 = 'span'
    var_5 = [var_4]

def test_case_0():
    var_0 = 'div'
    var_1 = 'hello'
    var_2 = 'p'
    var_3 = [var_2]
    var_4 = 'span'
    var_5 = [var_4]

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'hello '
    var_5 = 'p'
    var_6 = [var_5]
    var_7 = 'span'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'p'
    var_1 = 'content'
    var_2 = 'p'
    var_3 = [var_2]
    var_4 = 'span'
    var_5 = [var_4]

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = None
    var_3 = 'div'
    var_4 = 'outer'
    var_5 = 'p'
    var_6 = [var_5]
    var_7 = 'span'
    var_8 = [var_7]
    var_9 = False

def test_case_0():
    var_0 = lambda x: x
    var_1 = 'test'



# Parsed testcases at query #7
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'part1\npart2'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_extract_text_array_predicate_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #9
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = True
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)
    assert var_3 == 'part1part2'



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

# Partially parsed test_extract_text_basic_string. Retrieved 2/12 statements.
# Partially parsed test_extract_text_with_children. Retrieved 4/16 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 10/25 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 10/25 statements.
# Partially parsed test_extract_text_empty_node. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = 'p'
    var_3 = 'Hello '

import re as module_0

def test_case_0():
    var_0 = 'br'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = [var_2]
    var_4 = '\\s+'
    var_5 = module_0.compile(var_4)
    var_6 = 'p'
    var_7 = 'Line1'
    var_8 = None
    var_9 = '\n'

import re as module_0

def test_case_0():
    var_0 = []
    var_1 = 'span'
    var_2 = [var_1]
    var_3 = '\\s+'
    var_4 = module_0.compile(var_3)
    var_5 = 'Inner'
    var_6 = 'p'
    var_7 = 'Outer '
    var_8 = ' End'
    var_9 = False

def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extract_text_basic. Retrieved 4/8 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 5/10 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 6/11 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 7/13 statements.
# Partially parsed test_extract_text_empty. Retrieved 3/5 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 7/12 statements.
# Partially parsed test_extract_text_with_tail_and_none. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = 'span'
    var_3 = ' World'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Part1'
    var_2 = 'br'
    var_3 = 'p'
    var_4 = 'Part2'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'Middle'
    var_3 = 'span'
    var_4 = 'End'

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
    var_2 = 'span'
    var_3 = 'B'
    var_4 = ' C'
    var_5 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Outer '
    var_2 = 'p'
    var_3 = 'Inner'
    var_4 = 'b'
    var_5 = 'Bold'
    var_6 = ' Tail'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []

def test_case_0():
    var_0 = 'p'
    var_1 = 'A'
    var_2 = 'br'
    var_3 = 'span'
    var_4 = 'B'
    var_5 = '-'
    var_6 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = 'First'
    var_2 = 'span'
    var_3 = 'Second'
    var_4 = ' Third'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 7/13 statements.


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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_extract_text_array_predicate_true. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 13/26 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'mock_deps'
    var_1 = None
    var_2 = True
    var_3 = 'text'
    var_4 = [var_1, var_2, var_3]
    var_5 = globals()
    var_6 = 'extract_text_array'
    var_7 = '_merge_original_parts'
    var_8 = '_squash_artifical_nl'
    var_9 = '_strip_artifical_nl'
    var_10 = 'dummy'
    var_11 = False
    var_12 = module_0.extract_text(var_10, squash_space=var_11)
    assert var_12 == '\n\ntext'



# Parsed testcases at query #16
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_1, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = [var_0, var_0, var_1, var_0]
    var_3 = module_0._squash_artifical_nl(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = [var_0, var_1, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_extract_text_predicate_true. Retrieved 4/5 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = '\n'
    var_2 = True
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)



# Parsed testcases at query #18
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = '\n'
    var_3 = False
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 3/7 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 5/11 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 6/11 statements.
# Partially parsed test_extract_text_no_squash. Retrieved 5/10 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Part 1'
    var_2 = 'Part 2'
    var_3 = 'div'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Inline'
    var_2 = 'div'

def test_case_0():
    var_0 = 'b'
    var_1 = 'Bold'
    var_2 = 'p'
    var_3 = 'Paragraph'
    var_4 = 'div'

def test_case_0():
    var_0 = 'p'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'div'
    var_4 = ' | '
    var_5 = ' -> '

def test_case_0():
    var_0 = 'p'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'div'
    var_4 = False

def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_extract_text_basic_structure. Retrieved 4/8 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 6/11 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 5/9 statements.
# Partially parsed test_extract_text_no_squash. Retrieved 5/9 statements.
# Partially parsed test_extract_text_with_tails. Retrieved 5/9 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/3 statements.
# Partially parsed test_extract_text_complex_nesting. Retrieved 9/16 statements.
# Partially parsed test_extract_text_merge_parts. Retrieved 4/8 statements.
# Partially parsed test_extract_text_strip_artifical_nl_behavior. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = 'span'
    var_3 = ' World'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'First'
    var_3 = 'Second'
    var_4 = '\n'
    var_5 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Too   many    spaces  '
    var_2 = 'span'
    var_3 = '  more spaces '
    var_4 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Space '
    var_2 = 'span'
    var_3 = 'Between'
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'span'
    var_3 = 'Middle'
    var_4 = ' End'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'Line 1'
    var_3 = 'b'
    var_4 = 'Bold'
    var_5 = 'span'
    var_6 = ' Line 2'
    var_7 = '\n'
    var_8 = ' '

def test_case_0():
    var_0 = 'div'
    var_1 = 'Part1'
    var_2 = 'span'
    var_3 = 'Part2'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = 'Content'
    var_3 = None
    var_4 = 'End'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_extract_text_simple. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_children. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 6/11 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 5/10 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 6/11 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 7/14 statements.
# Partially parsed test_extract_text_whitespace_handling. Retrieved 2/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/3 statements.


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
    var_1 = 'Part 1'
    var_2 = 'Part 2'
    var_3 = 'div'
    var_4 = '\n'
    var_5 = ' '

def test_case_0():
    var_0 = 'div'
    var_1 = 'Line 1'
    var_2 = 'Line 2'
    var_3 = 'body'
    var_4 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Line 1'
    var_2 = 'Line 2'
    var_3 = 'body'
    var_4 = False
    var_5 = '\n'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Start '
    var_2 = 'b'
    var_3 = 'Bold'
    var_4 = 'p'
    var_5 = 'New Paragraph'
    var_6 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Too   Much   Space  '

def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #22
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_extract_text_simple_node. Retrieved 2/4 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 3/5 statements.
# Partially parsed test_extract_text_nested_structure. Retrieved 8/14 statements.
# Partially parsed test_extract_text_with_squash_space_true. Retrieved 3/5 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 3/5 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/3 statements.
# Partially parsed test_extract_text_complex_nesting_and_tails. Retrieved 6/13 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'p'
    var_1 = 'content'
    var_2 = '|'

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = 'b'
    var_3 = 'bold'
    var_4 = 'div'
    var_5 = 'start '
    var_6 = '\n'
    var_7 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = '  extra   spaces  '
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  extra   spaces  '
    var_2 = False

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'b'
    var_1 = 'Bold'
    var_2 = 'span'
    var_3 = 'Text'
    var_4 = 'div'
    var_5 = ''

def test_case_0():
    var_0 = 'p'
    var_1 = 'content'
    var_2 = '[B]'
    var_3 = '[S]'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_extract_text_predicate_true. Retrieved 3/4 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 10/20 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'mock_module'
    var_1 = 'part1'
    var_2 = None
    var_3 = 'part2'
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = 'some_dom'
    var_7 = '\n'
    var_8 = False
    var_9 = module_0.extract_text(var_6, var_7, var_7, var_8)
    assert var_9 == 'part1\npart2'



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_extract_text_array_predicate_false.




# Parsed testcases at query #27
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #28
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_extract_text_array_predicate_is_true. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'some_tag'



# Parsed testcases at query #30
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'dummy_dom'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.extract_text(var_1, squash_space=var_2)
    assert var_3 == 'part1part2'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_squash_logic. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'b'
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = 'start '
    var_5 = ' end'

def test_case_0():
    var_0 = lambda x: x
    var_1 = 'ignore'

def test_case_0():
    var_0 = 'b'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None
    var_6 = True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_extract_text_array_simple_text. Retrieved 9/23 statements.
# Partially parsed test_extract_text_array_with_children_and_none. Retrieved 12/29 statements.
# Partially parsed test_extract_text_array_with_separators. Retrieved 7/21 statements.
# Partially parsed test_extract_text_array_empty_dom. Retrieved 6/21 statements.
# Failed to parse test_extract_text_array_callable_tag.


def test_case_0():
    var_0 = 'SEPARATORS'
    var_1 = 'p'
    var_2 = 'div'
    var_3 = [var_1, var_2]
    var_4 = 'INLINE_TAGS'
    var_5 = 'span'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = 'Hello'

def test_case_0():
    var_0 = 'SEPARATORS'
    var_1 = 'p'
    var_2 = [var_1]
    var_3 = 'INLINE_TAGS'
    var_4 = 'span'
    var_5 = [var_4]
    var_6 = 'Inner'
    var_7 = 'b'
    var_8 = 'Bold'
    var_9 = 'div'
    var_10 = None
    var_11 = ' Tail'

def test_case_0():
    var_0 = 'SEPARATORS'
    var_1 = 'p'
    var_2 = [var_1]
    var_3 = 'INLINE_TAGS'
    var_4 = 'span'
    var_5 = [var_4]
    var_6 = 'Text'

def test_case_0():
    var_0 = 'SEPARATORS'
    var_1 = []
    var_2 = 'INLINE_TAGS'
    var_3 = []
    var_4 = 'div'
    var_5 = None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 8/21 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'mock_utils'
    var_1 = 'part1'
    var_2 = None
    var_3 = 'part2'
    var_4 = [var_1, var_2, var_3]
    var_5 = None
    var_6 = False
    var_7 = module_0.extract_text(var_5, squash_space=var_6)
    assert var_7 == 'part1\npart2'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_extract_text_basic_string. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 4/15 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/10 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 4/17 statements.
# Partially parsed test_extract_text_with_none_elements. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Part1'
    var_2 = 'br'
    var_3 = None

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start '
    var_2 = 'span'
    var_3 = 'Middle'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Text'



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_extract_text_array_predicate_true.




# Parsed testcases at query #36
#--------------------------

# Failed to parse test_extract_text_array_predicate_true.




