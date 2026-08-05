####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = ' world'
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello  '
    var_1 = '  world  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '   '
    var_1 = '  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == [42])
    assert var_3 is True

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
    var_0 = 'hello'
    var_1 = 42
    var_2 = 'world'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['hello', 42, 'world'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 'd'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._merge_original_parts(var_5)
    var_7 = bool(var_6 == ['ab', 1, 'cd'])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 'b'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._merge_original_parts(var_4)
    var_6 = bool(var_5 == ['a', 1, 2, 'b'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '   '
    var_1 = 1
    var_2 = '  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [1])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'hello'
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [1, 'hello world'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['hello world', 1])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [1, 'hello world'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['a', None, 'b'])
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/9 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 2/8 statements.
# Partially parsed test_extract_text_with_nested_structure. Retrieved 3/10 statements.
# Partially parsed test_extract_text_with_multiple_blocks. Retrieved 3/10 statements.
# Partially parsed test_extract_text_with_squash_space_disabled. Retrieved 3/10 statements.
# Partially parsed test_extract_text_with_custom_separator. Retrieved 4/10 statements.
# Partially parsed test_extract_text_with_leading_and_trailing_whitespace. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = 'p'

def test_case_0():
    var_0 = 'p'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'h1'
    var_2 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = 'p'
    var_3 = '---'

def test_case_0():
    var_0 = 'p'



# Parsed testcases at query #3
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_extract_text_returns_empty_string_for_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_returns_text_for_single_text_node. Retrieved 2/7 statements.
# Partially parsed test_extract_text_handles_separator_tag. Retrieved 6/14 statements.
# Partially parsed test_extract_text_handles_block_tags. Retrieved 6/14 statements.
# Partially parsed test_extract_text_strips_whitespace_with_squash_space. Retrieved 3/8 statements.
# Partially parsed test_extract_text_preserves_whitespace_without_squash. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'p'
    var_4 = 'Line1'
    var_5 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Block1'
    var_2 = None
    var_3 = 'body'
    var_4 = None
    var_5 = '\n'

def test_case_0():
    var_0 = 'p'
    var_1 = '  Hello   World  '
    var_2 = True

def test_case_0():
    var_0 = 'p'
    var_1 = '  Hello   World  '
    var_2 = False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_nested_elements. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator_and_block. Retrieved 4/9 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_squash_space_disabled. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_none_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_callable_tag. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'hr'
    var_1 = None
    var_2 = None
    var_3 = '---'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Line1'
    var_2 = None
    var_3 = '\n'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'p'
    var_4 = 'Hello '
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
    var_2 = None
    var_3 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'Bold'
    var_2 = ' '
    var_3 = 'i'
    var_4 = 'Italic'
    var_5 = None
    var_6 = 'p'
    var_7 = None
    var_8 = None

def test_case_0():
    var_0 = 'p'
    var_1 = '  Hello   World  '
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'Should not appear'
    var_1 = None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_true. Retrieved 6/8 statements.


def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = ''
    var_5 = '\n'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_br. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 1/4 statements.
# Partially parsed test_extract_text_block_nesting. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_only_whitespace. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<p>Line1<br/>Line2</p>'

def test_case_0():
    var_0 = '<hr/>'

def test_case_0():
    var_0 = '<p>Hello <b>World</b></p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<p>  Hello    World  </p>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>   </p>'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_block_separator. Retrieved 6/14 statements.
# Partially parsed test_extract_text_separator_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_empty. Retrieved 3/8 statements.
# Partially parsed test_extract_text_nested_with_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello '
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = '\n'
    var_3 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'Hello'
    var_5 = ' '
    var_6 = 'div'
    var_7 = None
    var_8 = None

def test_case_0():
    var_0 = 'p'
    var_1 = '  Hello   World  '
    var_2 = None
    var_3 = False



# Parsed testcases at query #9
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

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
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['a', 'b'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 'b'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['a', 1, 2, 'b'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)
    var_4 = bool(var_3 == ['', 'a'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == [''])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = [var_4]
    var_6 = [var_1, var_2, var_3, var_5]
    var_7 = module_0._strip_artifical_nl(var_6)
    var_8 = bool(var_7 == [[1], 'a', 'b', [2]])
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2.0
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_extract_text_squash_true_strips_result. Retrieved 9/10 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = ' '
    var_1 = 'text'
    var_2 = None
    var_3 = 'more '
    var_4 = '  '
    var_5 = [var_0, var_1, var_2, var_0, var_3, var_4]
    var_6 = '\n'
    var_7 = True
    var_8 = module_0.extract_text(var_5, var_6, var_6, var_7)



# Parsed testcases at query #11
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #12
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

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
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    var_3 = bool(var_2 == [None])
    assert var_3 is True

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
    var_3 = [var_0, var_1, var_1, var_2, var_1]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', None, 'b', None])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)
    var_4 = bool(var_3 == [None, 'a'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = [var_0, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)
    var_4 = bool(var_3 == ['a', None])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = None
    var_2 = 'y'
    var_3 = 'z'
    var_4 = [var_0, var_1, var_1, var_2, var_1, var_1, var_3]
    var_5 = module_0._squash_artifical_nl(var_4)
    var_6 = bool(var_5 == ['x', None, 'y', None, 'z'])
    assert var_6 is True



# Parsed testcases at query #13
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #14
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Node'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'p'
    var_7 = 'Hello'
    var_8 = None
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = module_1.extract_text(var_15)
    assert var_16 == 'Hello'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Node'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'span'
    var_7 = 'World'
    var_8 = None
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'p'
    var_18 = 'Hello '
    var_19 = [var_15]
    var_20 = lambda self: var_19
    var_21 = {var_2: var_17, var_3: var_18, var_4: var_8, var_5: var_20}
    var_22 = [var_0, var_16, var_21]
    var_23 = {}
    var_24 = module_0.type(*var_22, **var_23)
    var_25 = var_24()
    var_26 = module_1.extract_text(var_25)
    assert var_26 == 'Hello World'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Node'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'hr'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_7, var_5: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.extract_text(var_14)
    assert var_15 == '\n'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Node'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'div'
    var_7 = 'Inner'
    var_8 = None
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'body'
    var_18 = [var_15]
    var_19 = lambda self: var_18
    var_20 = {var_2: var_17, var_3: var_8, var_4: var_8, var_5: var_19}
    var_21 = [var_0, var_16, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()
    var_25 = module_1.extract_text(var_24)
    assert var_25 == 'Inner'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Node'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'p'
    var_7 = 'First'
    var_8 = None
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'Second'
    var_18 = []
    var_19 = lambda self: var_18
    var_20 = {var_2: var_6, var_3: var_17, var_4: var_8, var_5: var_19}
    var_21 = [var_0, var_16, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()
    var_25 = ()
    var_26 = 'body'
    var_27 = [var_15, var_24]
    var_28 = lambda self: var_27
    var_29 = {var_2: var_26, var_3: var_8, var_4: var_8, var_5: var_28}
    var_30 = [var_0, var_25, var_29]
    var_31 = {}
    var_32 = module_0.type(*var_30, **var_31)
    var_33 = var_32()
    var_34 = module_1.extract_text(var_33)
    assert var_34 == 'First\nSecond'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Node'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'b'
    var_7 = 'bold'
    var_8 = ' tail'
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'p'
    var_18 = 'Start '
    var_19 = None
    var_20 = [var_15]
    var_21 = lambda self: var_20
    var_22 = {var_2: var_17, var_3: var_18, var_4: var_19, var_5: var_21}
    var_23 = [var_0, var_16, var_22]
    var_24 = {}
    var_25 = module_0.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = module_1.extract_text(var_26)
    assert var_27 == 'Start bold tail'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Node'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'span'
    var_7 = '  spaced  '
    var_8 = None
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'p'
    var_18 = '  multiple   spaces  '
    var_19 = [var_15]
    var_20 = lambda self: var_19
    var_21 = {var_2: var_17, var_3: var_18, var_4: var_8, var_5: var_20}
    var_22 = [var_0, var_16, var_21]
    var_23 = {}
    var_24 = module_0.type(*var_22, **var_23)
    var_25 = var_24()
    var_26 = module_1.extract_text(var_25)
    assert var_26 == 'multiple spaces'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Node'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'div'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_7, var_5: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.extract_text(var_14)
    assert var_15 == ''



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_nl. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child_text'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'parent'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = False
    var_3 = True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_true. Retrieved 11/19 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = ' '
    var_3 = []
    var_4 = lambda : var_3
    var_5 = 'span'
    var_6 = 'world'
    var_7 = None
    var_8 = []
    var_9 = lambda : var_8
    var_10 = 'div'



# Parsed testcases at query #17
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'div'
    var_7 = 'hello'
    var_8 = None
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = True
    var_17 = False
    var_18 = module_1.extract_text_array(var_15, var_16, var_17)
    var_19 = bool(var_18 == ['hello'] or var_18 == ['hello', None])
    assert var_19 is True



# Parsed testcases at query #19
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'p'
    var_7 = 'Hello'
    var_8 = []
    var_9 = lambda : var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = True
    var_17 = False
    var_18 = module_1.extract_text_array(var_15, var_16, var_17)
    var_19 = bool(var_18 == ['Hello', None])
    assert var_19 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_extract_text_array_predicate_false. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 1/6 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/14 statements.
# Partially parsed test_extract_text_array_no_squash. Retrieved 4/14 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/9 statements.
# Partially parsed test_extract_text_array_no_strip. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'
    var_2 = 'span'
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'
    var_2 = 'span'
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = False



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_text_only. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_nested_inline. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_block_element. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_squash_nl. Retrieved 2/5 statements.
# Partially parsed test_extract_text_array_no_squash_nl. Retrieved 2/5 statements.
# Partially parsed test_extract_text_array_strip_leading_trailing_nl. Retrieved 2/5 statements.
# Partially parsed test_extract_text_array_no_strip_nl. Retrieved 2/5 statements.
# Partially parsed test_extract_text_array_keep_empty. Retrieved 2/5 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>Hello</p>'

def test_case_0():
    var_0 = '<br/>'

def test_case_0():
    var_0 = '<span>Hello <b>World</b></span>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<p>Hello<br/>World</p>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = True

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p>A</p></div>'
    var_1 = True

def test_case_0():
    var_0 = '<div><p>A</p></div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p></p></div>'
    var_1 = False

def test_case_0():
    var_0 = '<div></div>'
    var_1 = None



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_evaluates_true. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = 'br'
    var_3 = 'hr'
    var_4 = {var_2, var_3}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 16/21 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'INLINE_TAG'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = []
    var_15 = True
    var_16 = None
    var_17 = len(var_14)
    assert var_17 == 0



# Parsed testcases at query #25
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'valid_tag'
    var_7 = 'Some text'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = 'other_tag'
    var_17 = [var_16]
    var_18 = 'separator'
    var_19 = [var_18]
    var_20 = True
    var_21 = module_1.extract_text_array(var_15, var_20, var_20)
    var_22 = 'valid_tag'
    var_23 = bool('valid_tag' not in var_17)
    assert var_23 is True



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = 'text'
    var_2 = [var_0, var_1, var_0]
    var_3 = False
    assert var_3 is False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'br'
    var_1 = [var_0]
    var_2 = 'span'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = False



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 12/17 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'p'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_6, var_4: var_8}
    var_10 = True
    var_11 = False



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_strip_artifical_nl_true. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'p'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_6, var_4: var_8}
    var_10 = False
    var_11 = True



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_predicate_at_line_17_true.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_no_squash_or_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_complex. Retrieved 8/19 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'inline'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'p'
    var_4 = 'before '

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' after'
    var_3 = 'p'
    var_4 = 'before '

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'p'
    var_1 = 'first'
    var_2 = None
    var_3 = 'br'
    var_4 = None
    var_5 = None
    var_6 = 'body'
    var_7 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_separator_tag_returns_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'br'
    var_2 = {var_0, var_1}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_dom_tag_in_separators_returns_true. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'SEP'
    var_1 = {var_0}
    var_2 = 'p'
    var_3 = {var_2}
    var_4 = False



# Parsed testcases at query #34
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'p'
    var_7 = 'hello'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = False
    var_17 = True
    var_18 = module_1.extract_text_array(var_15, var_16, var_17)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_line_17_evaluates_to_false. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = False
    var_6 = None



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_extract_text_array_returns_empty_string_when_dom_tag_is_callable. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_separator_tag_adds_true. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_inline_tag_adds_none. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_text_appends_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_child_appends_child_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_tail_appends_tail. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl_removes_consecutive_nones. Retrieved 4/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_removes_leading_and_trailing_nones. Retrieved 3/7 statements.
# Partially parsed test_extract_text_array_returns_empty_list_for_empty_dom. Retrieved 1/4 statements.


def test_case_0():
    var_0 = lambda : None

def test_case_0():
    var_0 = 'br'

def test_case_0():
    var_0 = 'span'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = False
    var_2 = True

def test_case_0():
    var_0 = 'p'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '<root><child>text</child></root>'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_false. Retrieved 15/18 statements.


import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = []
    var_15 = []
    var_16 = module_1.extract_text_array(var_13)
    var_17 = var_13.tag
    var_18 = var_13.tag



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_strip_artifical_nl_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_squash_artifical_nl_is_false. Retrieved 15/16 statements.


import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'div'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_7, var_5: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = False
    var_16 = module_1.extract_text_array(var_14, var_15, var_15)
    var_17 = bool(var_16 == [None])
    assert var_17 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_false. Retrieved 3/11 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = False



# Parsed testcases at query #42
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'DIV'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.extract_text_array(var_14)
    var_16 = bool(var_15 == [None, None])
    assert var_16 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_single_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_no_squash_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_nested_separators. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = True

def test_case_0():
    var_0 = 'b'
    var_1 = 'child'
    var_2 = None
    var_3 = 'p'
    var_4 = 'parent'
    var_5 = True

def test_case_0():
    var_0 = 'b'
    var_1 = 'child'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'parent'
    var_5 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'a'
    var_5 = True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_extract_text_array_with_tag_not_in_inline_or_separators. Retrieved 5/9 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 7/12 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 4/8 statements.
# Partially parsed test_extract_text_array_with_child_and_tail. Retrieved 8/15 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 6/11 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 8/14 statements.
# Partially parsed test_extract_text_array_callable_tag_returns_empty_string. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/5 statements.
# Partially parsed test_extract_text_array_none_text_and_no_children. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = False
    var_2 = None
    var_3 = 'Hello'
    var_4 = [var_2, var_3, var_2]

def test_case_0():
    var_0 = 'br'
    var_1 = 'span'
    var_2 = False
    var_3 = True
    var_4 = 'World'
    var_5 = None
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 'b'
    var_1 = False
    var_2 = 'Bold'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'p'
    var_1 = 'a'
    var_2 = False
    var_3 = None
    var_4 = 'Start '
    var_5 = 'link'
    var_6 = ' end'
    var_7 = [var_3, var_4, var_5, var_6, var_3]

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = True
    var_3 = False
    var_4 = None
    var_5 = [var_4]

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = False
    var_3 = True
    var_4 = 'Content'
    var_5 = None
    var_6 = 'Child'
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = []

def test_case_0():
    var_0 = 'div'
    var_1 = False
    var_2 = None
    var_3 = [var_2, var_2]



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_squash_artifical_nl_false. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'div'
    var_5 = None
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_1: var_4, var_2: var_5, var_3: var_7}
    var_9 = False



# Parsed testcases at query #47
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'p'
    var_7 = 'Hello'
    var_8 = []
    var_9 = lambda : var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = False
    var_17 = True
    var_18 = module_1.extract_text_array(var_15, var_16, var_17)
    var_19 = bool(var_18 == ['Hello'] or var_18 == var_18)
    assert var_19 is True



# Parsed testcases at query #48
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'div'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = True
    var_16 = False
    var_17 = module_1.extract_text_array(var_14, var_15, var_16)
    var_18 = [var_7]
    var_19 = module_1._squash_artifical_nl(var_18)
    var_20 = var_17 == var_19
    var_21 = bool(var_20 or var_17 == [])
    assert var_21 is True



# Parsed testcases at query #49
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'p'
    var_7 = 'hello'
    var_8 = []
    var_9 = lambda : var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = False
    var_17 = module_1.extract_text_array(var_15, var_16, var_16)
    var_18 = bool(var_17 == ['hello', None])
    assert var_18 is True



# Parsed testcases at query #50
#--------------------------

# Failed to parse test_extract_text_array_returns_empty_string_for_callable_tag.
# Partially parsed test_extract_text_array_single_text_node. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_with_child_and_tail. Retrieved 5/15 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'inline'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'before '

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'b'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'c'
    var_2 = False



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_extract_text_array_with_callable_tag_returns_empty_string. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_separator_tag_adds_true. Retrieved 2/5 statements.
# Partially parsed test_extract_text_array_inline_tag_no_newline. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_block_tag_adds_none. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_with_child_and_tail. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/7 statements.
# Partially parsed test_extract_text_array_both_squash_and_strip. Retrieved 2/7 statements.


def test_case_0():
    var_0 = lambda : None

def test_case_0():
    var_0 = 'br'
    var_1 = False

def test_case_0():
    var_0 = 'span'
    var_1 = False

def test_case_0():
    var_0 = 'div'
    var_1 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = True
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = False
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_strip_artifical_nl_predicate_false. Retrieved 11/15 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'div'
    var_5 = None
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_1: var_4, var_2: var_5, var_3: var_7}
    var_9 = True
    var_10 = False



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_squash_artificial_nl_predicate_true. Retrieved 2/7 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_artifical_nl. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/9 statements.
# Partially parsed test_extract_text_array_with_nested_tags. Retrieved 3/9 statements.
# Partially parsed test_extract_text_array_mixed_content. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'span'

def test_case_0():
    var_0 = 'p'
    var_1 = 'span'



# Parsed testcases at query #55
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'p'
    var_6 = 'Hello'
    var_7 = []
    var_8 = lambda : var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = False
    var_15 = module_1.extract_text_array(var_13, var_14, var_14)
    var_16 = var_15[-1]
    assert var_16 is None



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = ' world'
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello  '
    var_1 = '  world  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == [1])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 1
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello', 1])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == [1, 'hello'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = ' world'
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['hello world', 1])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = ' world'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [1, 'hello world'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '   '
    var_1 = ' '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 'd'
    var_5 = 2
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0._merge_original_parts(var_6)
    var_8 = bool(var_7 == ['a b', 1, 'c d', 2])
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'hello'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [1, 2, 'hello'])
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_break. Retrieved 6/14 statements.
# Partially parsed test_extract_text_squash_whitespace. Retrieved 2/7 statements.
# Partially parsed test_extract_text_strip_artifical_nl. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello '

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = '\n'

def test_case_0():
    var_0 = 'p'
    var_1 = '  hello   world  '

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = '\n'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 7/15 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello '
    var_5 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None
    var_6 = '|'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'p'
    var_1 = '  Hello   World  '
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello '
    var_5 = None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_extract_text_with_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 1/6 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_inline_elements. Retrieved 1/6 statements.
# Partially parsed test_extract_text_with_nested_blocks. Retrieved 1/6 statements.
# Partially parsed test_extract_text_with_multiple_newlines_squashed. Retrieved 1/7 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_only_whitespace. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'



# Parsed testcases at query #6
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = var_5()
    var_7 = False
    var_8 = module_1.extract_text(var_6, squash_space=var_7)



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
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == [1])
    assert var_3 is True

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
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == [1, 2, 'a', 'b'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['a', 'b', 1, 2])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == [1, 'a', 'b', 2])
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

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [1, 'a', 'b'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b', 1])
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_extract_text_with_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 3/10 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_squash_space. Retrieved 2/6 statements.
# Partially parsed test_extract_text_without_squash_space. Retrieved 2/6 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 3/9 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'
    var_2 = '\n'

def test_case_0():
    var_0 = 'p'
    var_1 = True

def test_case_0():
    var_0 = 'p'
    var_1 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = '\n'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_squash_space_true. Retrieved 3/4 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #10
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag_no_text. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_text_only. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None

def test_case_0():
    var_0 = 'b'
    var_1 = None
    var_2 = None
    var_3 = 'b'
    var_4 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'before '

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'content'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_element. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_blocks. Retrieved 1/4 statements.
# Partially parsed test_extract_text_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_whitespace_squashing. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_block_symbol_custom. Retrieved 2/5 statements.
# Partially parsed test_extract_text_sep_symbol_custom. Retrieved 2/5 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_nested_with_tail. Retrieved 1/4 statements.
# Partially parsed test_extract_text_no_text_nodes. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello world</p>'

def test_case_0():
    var_0 = '<hr/>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<p>Hello <b>world</b></p>'

def test_case_0():
    var_0 = '<div><div>Nested</div></div>'

def test_case_0():
    var_0 = '<hr/><br/>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>  Hello   world  </p>'

def test_case_0():
    var_0 = '<p>Hello<b>bold</b>world</p>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<hr/>'
    var_1 = '|'

def test_case_0():
    var_0 = '<p>  Hello   world  </p>'
    var_1 = False

def test_case_0():
    var_0 = '<div>Start<b>bold</b>End</div>'

def test_case_0():
    var_0 = '<div><br/></div>'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_extract_text_with_separator_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_text_content. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_inline_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator_and_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_spaces. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_trailing_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_only_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div><br/></div>'

def test_case_0():
    var_0 = '<p>Hello</p>'

def test_case_0():
    var_0 = '<p>Hello <b>world</b></p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<div>Line1<br/>Line2</div>'

def test_case_0():
    var_0 = '<p>Hello    world</p>'

def test_case_0():
    var_0 = '<p>  Hello  </p>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<br/>'

def test_case_0():
    var_0 = '<div>Text <span>inline</span> more</div>'



# Parsed testcases at query #14
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #15
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    var_3 = bool(var_2 == ['a'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    var_3 = bool(var_2 == [None])
    assert var_3 is True

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
    var_4 = [var_0, var_1, var_1, var_2, var_1, var_1, var_3]
    var_5 = module_0._squash_artifical_nl(var_4)
    var_6 = bool(var_5 == ['a', None, 'b', None, 'c'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)
    var_4 = bool(var_3 == [None, 'a'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = [var_0, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)
    var_4 = bool(var_3 == ['a', None])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    var_3 = bool(var_2 == [None])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b', 'c'])
    assert var_5 is True



# Parsed testcases at query #16
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #17
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = var_5()
    var_7 = False
    var_8 = module_1.extract_text(var_6, squash_space=var_7)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_callable_dom_tag_returns_empty_string. Retrieved 1/4 statements.


def test_case_0():
    var_0 = lambda : None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_extract_text_squash_space_false. Retrieved 10/15 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = module_0.extract_text_array(var_0, var_1)
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = ''
    var_7 = None
    var_8 = '\n'
    var_9 = True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/10 statements.
# Partially parsed test_extract_text_with_inline_element. Retrieved 2/8 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 2/8 statements.
# Partially parsed test_extract_text_with_nested_blocks. Retrieved 2/9 statements.
# Partially parsed test_extract_text_multiple_separators. Retrieved 3/10 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_only_whitespace. Retrieved 1/5 statements.
# Partially parsed test_extract_text_mixed_blocks_and_inline. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = 'p'

def test_case_0():
    var_0 = 'p'
    var_1 = 'b'

def test_case_0():
    var_0 = 'p'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = 'p'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'b'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_break. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator_and_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_nested. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_squash_space_true. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty. Retrieved 1/4 statements.
# Partially parsed test_extract_text_only_whitespace. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello world</p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<hr>'

def test_case_0():
    var_0 = '<p>Before</p><hr><p>After</p>'

def test_case_0():
    var_0 = '<p>Hello <b>world</b></p>'

def test_case_0():
    var_0 = "<p>Hello <a href='#'>link</a> world</p>"

def test_case_0():
    var_0 = '<div><p>First <span>inline</span></p><p>Second</p></div>'

def test_case_0():
    var_0 = '<p>   Hello   world   </p>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>   </p>'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_extract_text_with_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_only_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_symbol_custom. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_sep_symbol_custom. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<hr/>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'

def test_case_0():
    var_0 = '<hr/><hr/>'

def test_case_0():
    var_0 = '<div>Start<p>Middle</p>End</div>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = 'Just text'

def test_case_0():
    var_0 = '<p>Line1</p><p>Line2</p>'
    var_1 = ' | '

def test_case_0():
    var_0 = '<hr/>'
    var_1 = ' - '



# Parsed testcases at query #23
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = None
    var_6 = lambda : var_5
    var_7 = []
    var_8 = lambda : var_7
    var_9 = {var_2: var_6, var_3: var_5, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.extract_text_array(var_13)
    assert var_14 == ''



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'not_callable'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_callable_dom_tag_returns_empty_string. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = None
    var_6 = lambda : var_5
    var_7 = []
    var_8 = lambda : var_7
    var_9 = {var_1: var_6, var_2: var_5, var_3: var_8, var_4: var_5}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 12/17 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = True
    var_11 = False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_strip_artifical_nl_false. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = False



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_extract_text_array_with_callable_tag_returns_empty_string. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_empty_element_returns_empty_list. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_single_text_node_returns_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_element_with_children_returns_text_array. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_separator_tag_adds_true. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_block_tag_adds_none. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl_true_combines_nones. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl_false_keeps_nones. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_true_removes_leading_trailing_none. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_false_keeps_leading_trailing_none. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_tail_text. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_nested_separators. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_empty_child_with_text. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_only_nones_with_squash_and_strip. Retrieved 3/7 statements.
# Partially parsed test_extract_text_array_multiple_children_with_artifical_nl. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_inline_tag_does_not_add_none. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_separator_and_inline_combination. Retrieved 3/9 statements.


def test_case_0():
    var_0 = lambda : None

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'br'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False
    var_3 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'
    var_2 = 'span'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_extract_text_array_simple_string. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_artificial_nl. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_squash_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_strip_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_empty. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'inline'
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'child'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'start'
    var_5 = None
    var_6 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = None
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'b'
    var_2 = None
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = lambda : None
    var_1 = 'x'
    var_2 = None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_strip_artifical_nl_false. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'
    var_2 = []
    var_3 = []
    var_4 = False



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_predicate_line_12_true.




# Parsed testcases at query #32
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'div'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.extract_text_array(var_14)
    var_16 = bool(var_15 == [])
    assert var_16 is True



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_predicate_at_line_17_evaluates_to_false.




# Parsed testcases at query #34
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'div'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = True
    var_16 = False
    var_17 = module_1.extract_text_array(var_14, var_15, var_16)
    var_18 = bool(var_17 is not None)
    assert var_18 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_nested_tags. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_mixed_elements. Retrieved 3/11 statements.
# Partially parsed test_extract_text_array_none_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_multiple_separators. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_with_separator. Retrieved 3/11 statements.
# Partially parsed test_extract_text_array_no_children. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_nested_inline. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_empty_string_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'br'

def test_case_0():
    var_0 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = True

def test_case_0():
    var_0 = 'div'
    var_1 = True

def test_case_0():
    var_0 = 'div'
    var_1 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'
    var_2 = 'span'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = 'br'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'span'
    var_1 = 'b'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = None
    var_3 = lambda : var_2
    var_4 = {var_1: var_3}



# Parsed testcases at query #36
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'SEPARATOR_TAG'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = False
    var_16 = module_1.extract_text_array(var_14, var_15, var_15)
    var_17 = bool(var_16 == [True])
    assert var_17 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_nested_elements. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_with_artifical_nl. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_squash_false. Retrieved 3/7 statements.
# Partially parsed test_extract_text_array_strip_false. Retrieved 3/7 statements.
# Partially parsed test_extract_text_array_with_text_and_children. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_separator_with_text. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'br'



# Parsed testcases at query #38
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'p'
    var_7 = None
    var_8 = []
    var_9 = lambda : var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = False
    var_16 = module_1.extract_text_array(var_14, var_15, var_15)
    var_17 = bool(var_16 == [None])
    assert var_17 is True



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_predicate_line7_false.




# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 13/17 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'inline_tag'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = [var_5]
    var_11 = 'separator_tag'
    var_12 = [var_11]



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_extract_text_array_predicate_line12_true.




# Parsed testcases at query #42
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'p'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_6, var_4: var_8}
    var_10 = True
    var_11 = False



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_extract_text_array_with_callable_tag_returns_empty_string. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag_adds_true. Retrieved 2/5 statements.
# Partially parsed test_extract_text_array_inline_tag_no_extra_nl. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_block_tag_adds_none. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl_combines_consecutive_none. Retrieved 7/14 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_removes_leading_and_trailing_none. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_default_parameters. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_nested_separators. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_separator_with_text. Retrieved 2/6 statements.


def test_case_0():
    var_0 = lambda : None
    var_1 = None

def test_case_0():
    var_0 = 'br'
    var_1 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'b'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = False

def test_case_0():
    var_0 = 'div'
    var_1 = True
    var_2 = False
    var_3 = None
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_3, var_4, var_3, var_5, var_3]

def test_case_0():
    var_0 = 'div'
    var_1 = False
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'span'

def test_case_0():
    var_0 = 'body'
    var_1 = 'br'
    var_2 = False

def test_case_0():
    var_0 = 'br'
    var_1 = False



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_strip_artifical_nl_true. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'p'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_6, var_4: var_8}
    var_10 = False
    var_11 = True



# Parsed testcases at query #45
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'p'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = False
    var_16 = True
    var_17 = module_1.extract_text_array(var_14, var_15, var_16)
    var_18 = bool(True)
    assert var_18 is True



# Parsed testcases at query #46
#--------------------------

# Failed to parse test_dom_tag_in_separators.




# Parsed testcases at query #47
#--------------------------

# Failed to parse test_extract_text_array_predicate_line12_false.




# Parsed testcases at query #48
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag_no_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_true. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_nested_structure. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = True

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'p'
    var_4 = 'before '
    var_5 = True

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' after'
    var_3 = 'p'
    var_4 = 'before '
    var_5 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'content'
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'nested'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = ' after div'
    var_6 = 'body'
    var_7 = 'start '
    var_8 = True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_strip_artifical_nl_executes_when_flag_true. Retrieved 13/19 statements.


def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'p'
    var_6 = 'hello'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_9}
    var_11 = False
    var_12 = True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_dom_tag_in_separators_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'br'
    var_1 = 'hr'
    var_2 = 'wbr'
    var_3 = {var_0, var_1, var_2}



# Parsed testcases at query #51
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.extract_text_array(var_13)
    var_15 = bool(var_14 == [])
    assert var_15 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'p'
    var_6 = 'Hello'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.extract_text_array(var_13)
    var_15 = bool(var_14 == ['Hello'])
    assert var_15 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'span'
    var_7 = 'World'
    var_8 = None
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'div'
    var_18 = [var_15]
    var_19 = lambda self: var_18
    var_20 = {var_2: var_17, var_3: var_8, var_5: var_19}
    var_21 = [var_0, var_16, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()
    var_25 = module_1.extract_text_array(var_24)
    var_26 = bool(var_25 == ['World'])
    assert var_26 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'br'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.extract_text_array(var_13)
    var_15 = bool(var_14 == [True])
    assert var_15 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.extract_text_array(var_13)
    var_15 = bool(var_14 == [])
    assert var_15 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = True
    var_15 = module_1.extract_text_array(var_13, var_14)
    var_16 = bool(var_15 == [])
    assert var_16 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = True
    var_15 = module_1.extract_text_array(var_13, strip_artifical_nl=var_14)
    var_16 = bool(var_15 == [])
    assert var_16 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'b'
    var_7 = 'Bold'
    var_8 = ' and '
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'p'
    var_18 = 'Start '
    var_19 = [var_15]
    var_20 = lambda self: var_19
    var_21 = {var_2: var_17, var_3: var_18, var_5: var_20}
    var_22 = [var_0, var_16, var_21]
    var_23 = {}
    var_24 = module_0.type(*var_22, **var_23)
    var_25 = var_24()
    var_26 = module_1.extract_text_array(var_25)
    var_27 = bool(var_26 == ['Start ', 'Bold', ' and '])
    assert var_27 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'br'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_7, var_5: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = ()
    var_16 = 'div'
    var_17 = 'A'
    var_18 = [var_14]
    var_19 = lambda self: var_18
    var_20 = {var_2: var_16, var_3: var_17, var_5: var_19}
    var_21 = [var_0, var_15, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()
    var_25 = module_1.extract_text_array(var_24)
    var_26 = bool(var_25 == ['A', True])
    assert var_26 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'div'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_7, var_5: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = ()
    var_16 = 'A'
    var_17 = [var_14]
    var_18 = lambda self: var_17
    var_19 = {var_2: var_6, var_3: var_16, var_5: var_18}
    var_20 = [var_0, var_15, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = False
    var_25 = module_1.extract_text_array(var_23, var_24)
    var_26 = bool(var_25 == ['A', None, None])
    assert var_26 is True



# Parsed testcases at query #52
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'INLINE_TAG'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = {var_6}
    var_16 = set()
    var_17 = module_1.extract_text_array(var_14)
    var_18 = None
    var_19 = bool(None not in var_17)
    assert var_19 is True



# Parsed testcases at query #53
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = True
    var_5 = var_4 == var_1



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_strip_artifical_nl_true. Retrieved 13/17 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'p'
    var_6 = 'Hello'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_9}
    var_11 = False
    var_12 = True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = []



# Parsed testcases at query #56
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = True
    assert var_1 is True



# Parsed testcases at query #57
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'div'
    var_7 = 'hello'
    var_8 = []
    var_9 = lambda : var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = module_1.extract_text_array(var_15)
    var_17 = 'hello'
    var_18 = bool('hello' in var_16)
    assert var_18 is True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_extract_text_array_predicate_line12_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div>text</div>'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/6 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 1/6 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_no_squash. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_no_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 1/6 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 2/10 statements.
# Partially parsed test_extract_text_array_separator_with_text. Retrieved 1/6 statements.
# Partially parsed test_extract_text_array_nested_separators. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_with_tail_after_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_strip_both_ends. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_consecutive_none. Retrieved 3/9 statements.
# Partially parsed test_extract_text_array_inline_tag_no_artifical. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_mixed_tags. Retrieved 6/18 statements.
# Partially parsed test_extract_text_array_empty_children. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'br'
    var_3 = 'Para1'
    var_4 = True
    var_5 = 'Para2'
    var_6 = 'Para1'
    var_7 = True
    var_8 = 'Para2'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/6 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 4/7 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 3/6 statements.
# Partially parsed test_extract_text_array_with_nested_elements. Retrieved 6/9 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl_false. Retrieved 7/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_false. Retrieved 6/9 statements.
# Partially parsed test_extract_text_array_with_text_and_tail. Retrieved 6/9 statements.
# Partially parsed test_extract_text_array_handles_callable_tag. Retrieved 4/8 statements.


def test_case_0():
    var_0 = '<div></div>'
    var_1 = True
    var_2 = []

def test_case_0():
    var_0 = '<p>Hello</p>'
    var_1 = True
    var_2 = 'Hello'
    var_3 = [var_2]

def test_case_0():
    var_0 = '<br/>'
    var_1 = True
    var_2 = [var_1]

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'
    var_1 = True
    var_2 = 'First'
    var_3 = None
    var_4 = 'Second'
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = False
    var_2 = True
    var_3 = 'A'
    var_4 = None
    var_5 = 'B'
    var_6 = [var_3, var_4, var_4, var_5, var_4]

def test_case_0():
    var_0 = '<div><p>Text</p></div>'
    var_1 = True
    var_2 = False
    var_3 = None
    var_4 = 'Text'
    var_5 = [var_3, var_4, var_3]

def test_case_0():
    var_0 = '<p>Hello <b>bold</b> world</p>'
    var_1 = True
    var_2 = 'Hello '
    var_3 = 'bold'
    var_4 = ' world'
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = '<div>test</div>'
    var_1 = None
    var_2 = True
    var_3 = ''



# Parsed testcases at query #61
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'div'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = True
    var_16 = False
    var_17 = module_1.extract_text_array(var_14, var_15, var_16)



# Parsed testcases at query #62
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = {}
    var_1 = '\n'
    var_2 = False
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)
    assert var_3 == ''



# Parsed testcases at query #63
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #64
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'p'
    var_7 = 'hello'
    var_8 = []
    var_9 = lambda : var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = False
    var_17 = module_1.extract_text_array(var_15, var_16, var_16)
    var_18 = bool(var_17 == ['hello', None])
    assert var_18 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_squash_artifical_nl_false. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'span'
    var_1 = [var_0]
    var_2 = False



# Parsed testcases at query #66
#--------------------------

# Failed to parse test_predicate_at_line_12_evaluates_to_false.




# Parsed testcases at query #67
#--------------------------

# Partially parsed test_extract_text_array_with_empty_dom. Retrieved 16/21 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 16/21 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 16/21 statements.
# Partially parsed test_extract_text_array_with_block_tag. Retrieved 17/22 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 20/32 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 18/23 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 19/24 statements.
# Partially parsed test_extract_text_array_with_nested_blocks. Retrieved 19/31 statements.


def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = 'span'
    var_11 = 'a'
    var_12 = {var_10, var_11}
    var_13 = 'br'
    var_14 = 'hr'
    var_15 = {var_13, var_14}

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'span'
    var_6 = 'Hello'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'a'
    var_12 = {var_5, var_11}
    var_13 = 'br'
    var_14 = 'hr'
    var_15 = {var_13, var_14}

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'br'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = 'span'
    var_11 = 'a'
    var_12 = {var_10, var_11}
    var_13 = 'hr'
    var_14 = {var_5, var_13}
    var_15 = False

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = 'span'
    var_11 = 'a'
    var_12 = {var_10, var_11}
    var_13 = 'br'
    var_14 = 'hr'
    var_15 = {var_13, var_14}
    var_16 = False

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'span'
    var_6 = 'World'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = '!'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'p'
    var_12 = 'Hello '
    var_13 = None
    var_14 = 'a'
    var_15 = {var_5, var_14}
    var_16 = 'br'
    var_17 = 'hr'
    var_18 = {var_16, var_17}
    var_19 = False

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = 'span'
    var_11 = 'a'
    var_12 = {var_10, var_11}
    var_13 = 'br'
    var_14 = 'hr'
    var_15 = {var_13, var_14}
    var_16 = True
    var_17 = False

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = 'Hello'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'span'
    var_12 = 'a'
    var_13 = {var_11, var_12}
    var_14 = 'br'
    var_15 = 'hr'
    var_16 = {var_14, var_15}
    var_17 = False
    var_18 = True

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = 'inner'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'div'
    var_12 = 'span'
    var_13 = 'a'
    var_14 = {var_12, var_13}
    var_15 = 'br'
    var_16 = 'hr'
    var_17 = {var_15, var_16}
    var_18 = True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_extract_text_array_with_none_dom_tag_callable. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_separator_tag_and_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_inline_tag_and_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_non_inline_non_separator_tag_and_children. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/8 statements.


def test_case_0():
    var_0 = lambda : None

def test_case_0():
    var_0 = 'br'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'start'

def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = False



# Parsed testcases at query #69
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 == '' or var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_predicate_line20_evaluates_to_false. Retrieved 16/17 statements.


import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'inline_tag'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = False
    var_16 = True
    var_17 = module_1.extract_text_array(var_14, var_15, var_16)
    var_18 = bool(var_17 == [])
    assert var_18 is True



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_children. Retrieved 2/9 statements.
# Partially parsed test_extract_text_block_elements. Retrieved 2/9 statements.
# Partially parsed test_extract_text_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_squash_whitespace. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_nested_blocks. Retrieved 2/8 statements.
# Partially parsed test_extract_text_multiple_separators. Retrieved 2/8 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 4/11 statements.
# Partially parsed test_extract_text_no_squash. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'body'
    var_1 = 'hr'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'body'
    var_1 = 'hr'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = ' | '
    var_3 = ' * '

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False



# Parsed testcases at query #72
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = '\n'
    var_3 = True
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #73
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.extract_text(var_1, squash_space=var_2)
    assert var_3 == 'text'



# Parsed testcases at query #74
#--------------------------

# Failed to parse test_extract_text_array_predicate_false.




# Parsed testcases at query #75
#--------------------------

# Partially parsed test_extract_text_array_with_callable_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_separator_tag_no_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag_no_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_block_tag_no_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_children_and_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 6/14 statements.


def test_case_0():
    var_0 = lambda : None

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'hello '

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = 'hello'
    var_5 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = 'hello'
    var_5 = True



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 1/7 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_predicate_line_12_true. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '<root><child1/><child2/></root>'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_extract_text_returns_stripped_result_when_squash_space_is_true. Retrieved 7/8 statements.


import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = var_5()
    var_7 = True
    var_8 = module_1.extract_text(var_6, squash_space=var_7)



# Parsed testcases at query #79
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = False
    var_3 = module_0.extract_text_array(var_0, var_1, var_2)
    assert var_3 == ''



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_false. Retrieved 1/7 statements.


def test_case_0():
    var_0 = False
    var_1 = None



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_predicate_false_when_tag_in_inline_tags. Retrieved 3/8 statements.


def test_case_0():
    var_0 = []
    var_1 = 'span'
    var_2 = [var_1]



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_strip_artifical_nl_true. Retrieved 5/6 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'text'
    var_2 = [var_0, var_1, var_0]
    var_3 = True
    var_4 = module_0._strip_artifical_nl(var_2)



# Parsed testcases at query #83
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_br. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_inline. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_no_squash. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello</div>'

def test_case_0():
    var_0 = '<div>Line1<br/>Line2</div>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<div><hr/>Content</div>'

def test_case_0():
    var_0 = '<div>Hello <b>World</b></div>'

def test_case_0():
    var_0 = '<div>  Hello   World  </div>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div><hr/>Content</div>'
    var_1 = '---'

def test_case_0():
    var_0 = '<div>  Hello   World  </div>'
    var_1 = False



# Parsed testcases at query #85
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_single_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_nested. Retrieved 5/13 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 3/8 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 8/19 statements.
# Partially parsed test_extract_text_block_symbol. Retrieved 6/14 statements.
# Partially parsed test_extract_text_sep_symbol. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello '

def test_case_0():
    var_0 = 'b'
    var_1 = 'Bold'
    var_2 = ' tail'

def test_case_0():
    var_0 = 'span'
    var_1 = 'first'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'second'
    var_5 = None
    var_6 = 'div'
    var_7 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'para'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = '|'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'a'
    var_5 = '|'



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_child. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<hr/>'

def test_case_0():
    var_0 = '<p>Hello <b>World</b> again</p>'

def test_case_0():
    var_0 = '<p></p>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = ' | '

def test_case_0():
    var_0 = '<hr/>'
    var_1 = '---'

def test_case_0():
    var_0 = '<p>  Hello   </p>'
    var_1 = False

def test_case_0():
    var_0 = '<span>Text <em>emphasized</em> end</span>'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 10/14 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 11/15 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 10/14 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 14/24 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 11/21 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 11/15 statements.
# Partially parsed test_extract_text_array_no_squash. Retrieved 11/21 statements.
# Partially parsed test_extract_text_array_no_strip. Retrieved 11/15 statements.


def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = 'Hello'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'br'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'b'
    var_6 = 'bold'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = ' text'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'p'
    var_12 = 'Some '
    var_13 = None

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = True

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = True

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = False

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = False



# Parsed testcases at query #89
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_single_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_artifical_nl. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_true. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_true. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_and_strip_false. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = 'parent'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'parent'

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = False



# Parsed testcases at query #91
#--------------------------

# Failed to parse test_predicate_at_line_17_evaluates_to_true.




# Parsed testcases at query #92
#--------------------------

# Failed to parse test_predicate_line_12_evaluates_to_true.




# Parsed testcases at query #93
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 4/7 statements.
# Partially parsed test_extract_text_single_string. Retrieved 5/8 statements.
# Partially parsed test_extract_text_with_child. Retrieved 8/14 statements.
# Partially parsed test_extract_text_separator. Retrieved 4/7 statements.
# Partially parsed test_extract_text_block_symbol. Retrieved 9/16 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 5/8 statements.
# Partially parsed test_extract_text_nested_blocks. Retrieved 6/12 statements.
# Partially parsed test_extract_text_artificial_newlines_stripped. Retrieved 6/12 statements.
# Partially parsed test_extract_text_separator_between_text. Retrieved 7/13 statements.
# Partially parsed test_extract_text_multiple_separators. Retrieved 9/16 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = lambda : var_2

def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = ' tail'
    var_5 = 'p'
    var_6 = 'before '
    var_7 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = []
    var_3 = lambda : var_2

def test_case_0():
    var_0 = 'p'
    var_1 = 'first'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = None
    var_5 = 'second'
    var_6 = []
    var_7 = lambda : var_6
    var_8 = 'div'

def test_case_0():
    var_0 = 'p'
    var_1 = ' hello '
    var_2 = []
    var_3 = lambda : var_2
    var_4 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'inner'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = None
    var_5 = 'div'

def test_case_0():
    var_0 = 'span'
    var_1 = 'text'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = None
    var_5 = 'div'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = []
    var_3 = lambda : var_2
    var_4 = 'p'
    var_5 = 'a'
    var_6 = 'b'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = []
    var_3 = lambda : var_2
    var_4 = []
    var_5 = lambda : var_4
    var_6 = 'p'
    var_7 = 'x'
    var_8 = 'y'



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_squash_artifical_nl_is_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '<div><p>text</p></div>'
    var_1 = False
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #95
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = None
    var_1 = 'hello'
    var_2 = True
    var_3 = 'world'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 'MockDOM'
    var_6 = ()
    var_7 = 'extract_text'
    var_8 = lambda self: var_4
    var_9 = {var_7: var_8}
    var_10 = [var_5, var_6, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.extract_text(var_13, squash_space=var_2)



# Parsed testcases at query #96
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Dom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'div'
    var_7 = 'hello'
    var_8 = []
    var_9 = lambda : var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = False
    var_17 = module_1.extract_text_array(var_15, var_16, var_16)
    var_18 = bool(var_17 == ['hello', None])
    assert var_18 is True



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_predicate_line12_false. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = ''
    var_2 = []
    var_3 = True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #98
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'p'
    var_7 = None
    var_8 = []
    var_9 = lambda : var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = False
    var_16 = module_1.extract_text_array(var_14, var_15, var_15)
    var_17 = bool(var_16 == [])
    assert var_17 is True



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_artifical_nl. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child_and_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_squash_and_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_no_squash. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_no_strip. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_separator_and_text. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'text'

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'before '

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'br'
    var_1 = 'text'



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 3/6 statements.


def test_case_0():
    var_0 = '<div><p>text</p></div>'
    var_1 = True
    var_2 = False



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_predicate_false. Retrieved 3/4 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #102
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = '\n'
    var_3 = True
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_child. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 2/8 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 1/5 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 3/10 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 3/8 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 3/9 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'

def test_case_0():
    var_0 = 'p'
    var_1 = 'strong'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = ' | '

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = ' --- '

def test_case_0():
    var_0 = 'p'
    var_1 = 'br'
    var_2 = False

def test_case_0():
    var_0 = 'p'
    var_1 = 'span'
    var_2 = 'strong'



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_extract_text_array_predicate_true. Retrieved 3/6 statements.


def test_case_0():
    var_0 = '<div><p>Hello</p></div>'
    var_1 = 0
    var_2 = dom.getchildren()[var_1]



# Parsed testcases at query #105
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda : var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.extract_text_array(var_13)



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_strip_artifical_nl_false. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = False



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_extract_text_simple. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty. Retrieved 1/4 statements.
# Partially parsed test_extract_text_no_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 3/6 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_multiple_separators. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<div><p>Line1</p><hr/><p>Line2</p></div>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<div><p><b>Bold</b> text</p></div>'

def test_case_0():
    var_0 = '<p>  Extra   spaces  </p>'

def test_case_0():
    var_0 = '<p></p>'

def test_case_0():
    var_0 = '<div><br/></div>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = '|'
    var_2 = '-'

def test_case_0():
    var_0 = '<p>  Hello  </p>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p>Text</p><hr/><hr/><p>More</p></div>'



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 10/15 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'div'
    var_5 = None
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_1: var_4, var_2: var_5, var_3: var_7}
    var_9 = True
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #109
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 == '' or var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 11/15 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 13/17 statements.
# Partially parsed test_extract_text_array_inline_tag_with_text. Retrieved 13/17 statements.
# Partially parsed test_extract_text_array_block_tag_artifical_nl. Retrieved 12/16 statements.
# Partially parsed test_extract_text_array_with_child_and_tail. Retrieved 16/26 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 13/17 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 14/18 statements.
# Partially parsed test_extract_text_array_both_squash_and_strip. Retrieved 13/17 statements.
# Partially parsed test_extract_text_array_separator_with_text. Retrieved 14/18 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 18/31 statements.
# Partially parsed test_extract_text_array_callable_tag_returns_empty. Retrieved 11/15 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = []

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'br'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = False
    var_11 = True
    var_12 = [var_11]

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'span'
    var_6 = 'hello'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = False
    var_12 = [var_6]

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = False
    var_11 = [var_6, var_6]

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'span'
    var_6 = 'inner'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = ' tail'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'div'
    var_12 = 'start'
    var_13 = None
    var_14 = False
    var_15 = [var_13, var_12, var_6, var_9, var_13]

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = True
    var_11 = False
    var_12 = [var_6]

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = 'a'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = False
    var_12 = True
    var_13 = [var_6]

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = 'a'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = True
    var_12 = [var_6]

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'br'
    var_6 = 'x'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = False
    var_12 = True
    var_13 = [var_12, var_6]

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'span'
    var_6 = 'a'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'b'
    var_12 = []
    var_13 = lambda self: var_12
    var_14 = {var_1: var_5, var_2: var_11, var_3: var_13, var_4: var_9}
    var_15 = 'div'
    var_16 = False
    var_17 = [var_9, var_6, var_9, var_11, var_9]

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = None
    var_6 = lambda : var_5
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_6, var_2: var_5, var_3: var_8, var_4: var_5}
    var_10 = ''



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_false. Retrieved 11/17 statements.


def test_case_0():
    var_0 = 'Element'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = False



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_predicate_at_line_1_is_false. Retrieved 3/5 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 10/14 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 11/15 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 14/24 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 10/14 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 11/15 statements.
# Partially parsed test_extract_text_array_artifical_nl_inserted. Retrieved 11/15 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 13/17 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 13/17 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 10/14 statements.
# Partially parsed test_extract_text_array_nested_structure. Retrieved 13/23 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 18/31 statements.


def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}

def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = 'Hello'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}

def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'b'
    var_6 = 'bold'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = ' tail'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'p'
    var_12 = 'before '
    var_13 = None

def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'br'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}

def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'span'
    var_6 = 'inline'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}

def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = None
    var_9 = {var_1: var_5, var_2: var_2, var_3: var_7, var_4: var_8}
    var_10 = False

def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = 'a'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = True
    var_12 = False

def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = 'b'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = False
    var_12 = True

def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = None
    var_6 = lambda : var_5
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_6, var_2: var_5, var_3: var_8, var_4: var_5}

def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = 'inner'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = ' after'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'outer '
    var_12 = None

def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'br'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = 'span'
    var_11 = 'sp'
    var_12 = []
    var_13 = lambda self: var_12
    var_14 = ' t'
    var_15 = {var_1: var_10, var_2: var_11, var_3: var_13, var_4: var_14}
    var_16 = 'p'
    var_17 = 'start'



# Parsed testcases at query #114
#--------------------------

# Partially parsed test_squash_artifical_nl_false. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'p'
    var_5 = None
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_1: var_4, var_2: var_5, var_3: var_7}
    var_9 = False



# Parsed testcases at query #115
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = True
    var_3 = module_0.extract_text(var_0, squash_space=var_2)
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #116
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #117
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = '\n'
    var_3 = False
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)



# Parsed testcases at query #118
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_false. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'body'
    var_1 = None



# Parsed testcases at query #119
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'p'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_7, var_5: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = False
    var_16 = True
    var_17 = module_1.extract_text_array(var_14, var_15, var_16)
    var_18 = bool(var_17 == [])
    assert var_18 is True



# Parsed testcases at query #120
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = False
    var_15 = module_1.extract_text_array(var_13, var_14, var_14)



# Parsed testcases at query #121
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_squash_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_false. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_nested_tags. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = None
    var_4 = lambda : var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = module_1.extract_text_array(var_9)
    assert var_10 == ''



# Parsed testcases at query #122
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'p'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = 'span'
    var_16 = 'a'
    var_17 = {var_15, var_16}
    var_18 = 'br'
    var_19 = 'hr'
    var_20 = {var_18, var_19}
    var_21 = False
    var_22 = module_1.extract_text_array(var_14, var_21, var_21)
    var_23 = var_22[-1]
    assert var_23 is None



# Parsed testcases at query #123
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'div'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_7, var_5: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = False
    var_16 = module_1.extract_text_array(var_14, var_15, var_15)
    var_17 = bool(var_16 == [])
    assert var_17 is True



# Parsed testcases at query #124
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    assert var_1 is True
    var_2 = module_0.extract_text_array(var_0, var_1)



# Parsed testcases at query #125
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 10/17 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 10/17 statements.
# Partially parsed test_extract_text_array_with_child_text. Retrieved 14/32 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 11/18 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 11/18 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 15/33 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 12/19 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 11/18 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 10/17 statements.
# Partially parsed test_extract_text_array_nested_artifical_nl. Retrieved 12/30 statements.


def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'div'
    var_5 = None
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_1: var_4, var_2: var_5, var_3: var_7}
    var_9 = []

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'div'
    var_5 = 'Hello'
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_1: var_4, var_2: var_5, var_3: var_7}
    var_9 = []

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'span'
    var_6 = 'World'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_9}
    var_11 = []
    var_12 = 'div'
    var_13 = 'Hello '

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'br'
    var_5 = None
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_1: var_4, var_2: var_5, var_3: var_7}
    var_9 = []
    var_10 = False

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'b'
    var_5 = 'bold'
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_1: var_4, var_2: var_5, var_3: var_7}
    var_9 = []
    var_10 = False

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'span'
    var_6 = 'inner'
    var_7 = ' tail'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_9}
    var_11 = []
    var_12 = 'div'
    var_13 = 'start'
    var_14 = False

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'div'
    var_5 = None
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_1: var_4, var_2: var_5, var_3: var_7}
    var_9 = []
    var_10 = True
    var_11 = False

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'div'
    var_5 = []
    var_6 = lambda self: var_5
    var_7 = {var_1: var_4, var_2: var_2, var_3: var_6}
    var_8 = []
    var_9 = False
    var_10 = True

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = None
    var_5 = lambda : var_4
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_1: var_5, var_2: var_4, var_3: var_7}
    var_9 = []

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_6, var_4: var_8}
    var_10 = []
    var_11 = 'a'



# Parsed testcases at query #126
#--------------------------

# Partially parsed test_predicate_line_20_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '<p>text</p>'
    var_1 = True



# Parsed testcases at query #127
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = '\n'
    var_2 = False
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #128
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #129
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/12 statements.
# Partially parsed test_extract_text_block_elements. Retrieved 2/9 statements.
# Partially parsed test_extract_text_whitespace_squash. Retrieved 1/5 statements.
# Partially parsed test_extract_text_empty. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 2/9 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 4/11 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/6 statements.
# Partially parsed test_extract_text_trailing_newline. Retrieved 2/7 statements.
# Partially parsed test_extract_text_leading_newline. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = '|'
    var_3 = '-'

def test_case_0():
    var_0 = 'p'
    var_1 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'



# Parsed testcases at query #130
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #131
#--------------------------

# Partially parsed test_extract_text_with_squash_space_returns_stripped_result. Retrieved 11/12 statements.


import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = var_5()
    var_7 = '  hello  '
    var_8 = None
    var_9 = '  world  '
    var_10 = [var_7, var_8, var_9]
    var_11 = True
    var_12 = module_1.extract_text(var_6, squash_space=var_11)
    assert var_12 == 'hello\nworld'



