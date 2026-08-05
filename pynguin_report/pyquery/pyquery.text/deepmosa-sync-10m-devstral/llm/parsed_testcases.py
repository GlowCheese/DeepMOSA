####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

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
    var_1 = 'world'
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 1
    var_2 = 'world'
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._merge_original_parts(var_4)
    var_6 = bool(var_5 == ['hello', 1, 'world', 2])
    assert var_6 is True

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
    var_0 = ''
    var_1 = 'hello'
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  '
    var_1 = '   '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

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
    var_0 = '  hello  '
    var_1 = 1
    var_2 = '  world  '
    var_3 = 2
    var_4 = '  '
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._merge_original_parts(var_5)
    var_7 = bool(var_6 == ['hello', 1, 'world', 2])
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello World'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Child'
    var_2 = 'Tail'
    var_3 = 'div'
    var_4 = 'Start'
    var_5 = 'End'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Text'
    var_2 = None
    var_3 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'First'
    var_2 = ' '
    var_3 = 'div'
    var_4 = 'Second'
    var_5 = ' '
    var_6 = 'div'
    var_7 = 'Start'
    var_8 = 'End'
    var_9 = True



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #4
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    var_3 = bool(var_2 == [None])
    assert var_3 is True

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
    var_4 = [var_0, var_1, var_2, var_1, var_1, var_3]
    var_5 = module_0._squash_artifical_nl(var_4)
    var_6 = bool(var_5 == ['a', None, 'b', None, 'c'])
    assert var_6 is True

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
    var_1 = 'b'
    var_2 = None
    var_3 = [var_0, var_1, var_2, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b', None])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [None, 'a', 'b'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_1, var_0, var_2, var_0]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [None, 'a', None, 'b', None])
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_non_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_complex_case. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = True



# Parsed testcases at query #6
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = module_0.extract_text(var_3, squash_space=var_4)
    assert var_5 == '\n\ntext'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dom_tag_not_callable. Retrieved 13/14 statements.


import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDOM'
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_strip_artifical_nl_predicate_false. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_strip_artifical_nl_predicate_false. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = None
    var_5 = False



# Parsed testcases at query #10
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'MockElement'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'div'
    var_7 = None
    var_8 = []
    var_9 = lambda : var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()



# Parsed testcases at query #11
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockElement'
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
    var_15 = bool(var_14 == [None, None])
    assert var_15 is True



# Parsed testcases at query #12
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'MockDom'
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_17. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = lambda : var_2



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_nested_children. Retrieved 7/18 statements.
# Partially parsed test_extract_text_array_with_artificial_newlines. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_without_squash_artificial_nl. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_without_strip_artificial_nl. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_callable_tag. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'span'
    var_3 = 'Middle'
    var_4 = 'End'
    var_5 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'div'
    var_3 = 'Middle'
    var_4 = 'span'
    var_5 = 'End'
    var_6 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'div'
    var_3 = 'Middle'
    var_4 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'div'
    var_3 = 'Middle'
    var_4 = False
    var_5 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = 'span'
    var_3 = 'Middle'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Hello'
    var_1 = True



# Parsed testcases at query #15
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockElement'
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
    var_13 = False
    var_14 = module_1.extract_text_array(var_12, var_13, var_13)
    var_15 = bool(var_14 == [None, None])
    assert var_15 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Failed to parse test_extract_text_array_callable_tag.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = False
    var_3 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_squash_artifical_nl_is_true. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'strong'
    var_1 = 'Bold'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = None



# Parsed testcases at query #19
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockElement'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'div'
    var_7 = None
    var_8 = []
    var_9 = lambda : var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_7, var_5: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = False
    var_16 = module_1.extract_text_array(var_14, var_15, var_15)
    var_17 = bool(var_16 == [None, None])
    assert var_17 is True



# Parsed testcases at query #20
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'test'
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = module_0.extract_text(var_3, squash_space=var_1)
    assert var_4 == '\ntest'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dom_tag_in_separators. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = 'p'
    var_3 = 'div'
    var_4 = 'br'
    var_5 = {var_2, var_3, var_4}
    var_6 = 'span'
    var_7 = 'a'
    var_8 = 'strong'
    var_9 = {var_6, var_7, var_8}
    var_10 = False



# Parsed testcases at query #22
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDOM'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'inline_tag'
    var_6 = None
    var_7 = []
    var_8 = lambda : var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = {var_5}
    var_15 = 'separator_tag'
    var_16 = {var_15}
    var_17 = module_1.extract_text_array(var_13)
    var_18 = bool(var_17 == [])
    assert var_18 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_non_inline_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_with_squash_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_with_strip_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_with_callable_tag. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None
    var_6 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = lambda : 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 6/11 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 6/11 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 9/17 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_non_inline_tag. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_with_multiple_children. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = None
    var_3 = 'p'
    var_4 = {var_3}
    var_5 = set()

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None
    var_3 = set()
    var_4 = 'span'
    var_5 = {var_4}

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'End'
    var_3 = 'span'
    var_4 = 'Child'
    var_5 = 'Tail'
    var_6 = set()
    var_7 = 'span'
    var_8 = {var_7}

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = set()
    var_4 = set()

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = set()
    var_4 = set()

def test_case_0():
    var_0 = lambda : 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = set()
    var_4 = set()

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'End'
    var_3 = 'span'
    var_4 = 'Child1'
    var_5 = 'Tail1'
    var_6 = 'span'
    var_7 = 'Child2'
    var_8 = 'Tail2'
    var_9 = set()
    var_10 = 'span'
    var_11 = {var_10}



# Parsed testcases at query #25
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #26
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockElement'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'div'
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
    var_17 = bool(var_16 == [None, None])
    assert var_17 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_extract_text_array_strip_artifical_nl_true. Retrieved 1/7 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_squash_artifical_nl_is_true. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = True
    var_5 = False



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = 'span'
    var_4 = 'World'
    var_5 = '!'
    var_6 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Inline'
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = lambda : 'div'
    var_1 = None
    var_2 = None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = False



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dom_text_is_not_none. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'some text'
    var_2 = lambda self: []



# Parsed testcases at query #32
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = module_0.extract_text(var_3, squash_space=var_1)
    assert var_4 == '\n\ntext\n'



# Parsed testcases at query #33
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'MockElement'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'p'
    var_6 = None
    var_7 = []
    var_8 = lambda : var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()



# Parsed testcases at query #34
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockElement'
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
    var_15 = len(var_14)
    var_16 = 1
    var_17 = var_15 == var_16
    var_18 = bool(var_17 and var_14[0] is None)
    assert var_18 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = False
    var_6 = True

def test_case_0():
    var_0 = False



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dom_tag_in_separators. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'separator_tag'
    var_1 = {var_0}
    var_2 = set()



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_separator_tag_in_dom. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = {var_0}
    var_2 = set()



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_strip_artifical_nl_predicate_false. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = False



# Parsed testcases at query #39
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
    var_8 = lambda : var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.extract_text_array(var_13)
    var_15 = len(var_14)
    var_16 = 1
    var_17 = var_15 == var_16
    var_18 = bool(var_17 and var_14[0] is None)
    assert var_18 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_no_squash_space. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_preformatted_content. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div>Hello<p>World</p></div>'

def test_case_0():
    var_0 = '<div>Hello<br/>World</div>'

def test_case_0():
    var_0 = '<div>Hello<div>World<span>!</span></div></div>'

def test_case_0():
    var_0 = '<div>Hello   <p>   World   </p>   </div>'
    var_1 = True

def test_case_0():
    var_0 = '<div>Hello<p>World</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>Hello<br/>World</div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>Hello   <p>   World   </p>   </div>'
    var_1 = False

def test_case_0():
    var_0 = '<div>Hello<pre>  World  </pre>Goodbye</div>'

def test_case_0():
    var_0 = '<div>Hello<br/><br/>World</div>'

def test_case_0():
    var_0 = '<div>Hello<b>Bold</b><i>Italic</i>World</div>'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_squash_artifical_nl_predicate. Retrieved 16/17 statements.


import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDOM'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'div'
    var_7 = None
    var_8 = []
    var_9 = lambda : var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = True
    var_16 = False
    var_17 = module_1.extract_text_array(var_14, var_15, var_16)



# Parsed testcases at query #42
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = module_0.extract_text(var_3, squash_space=var_4)
    assert var_5 == '\n\ntext'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 4/7 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 4/7 statements.
# Partially parsed test_extract_text_with_children. Retrieved 8/12 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 4/7 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 4/7 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 5/8 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 5/8 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 6/9 statements.
# Partially parsed test_extract_text_nested_elements. Retrieved 12/17 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = lambda self: []

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = lambda self: []

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = lambda self: []
    var_4 = 'div'
    var_5 = 'Hello'
    var_6 = None
    var_7 = lambda self: [MockChild()]

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = lambda self: []

def test_case_0():
    var_0 = 'p'
    var_1 = 'Paragraph'
    var_2 = None
    var_3 = lambda self: []

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello  '
    var_2 = None
    var_3 = lambda self: []
    var_4 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello  '
    var_2 = None
    var_3 = lambda self: []
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = lambda self: []
    var_4 = '|'
    var_5 = ';'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' text'
    var_3 = lambda self: []
    var_4 = 'span'
    var_5 = 'Some'
    var_6 = ' and'
    var_7 = lambda self: [MockGrandchild()]
    var_8 = 'div'
    var_9 = 'Start'
    var_10 = ' end'
    var_11 = lambda self: [MockChild()]



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_dom_tag_in_separators. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'separator_tag'
    var_1 = {var_0}
    var_2 = set()



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_non_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = lambda : 'div'
    var_1 = None
    var_2 = None



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_squash_artifical_nl_is_false. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = lambda : var_2
    var_4 = False



# Parsed testcases at query #47
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

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
    var_6 = bool(var_5 == [1, 2])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'end'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0._strip_artifical_nl(var_6)
    var_8 = bool(var_7 == [1, 2])
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 'b'
    var_4 = 3
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._strip_artifical_nl(var_5)
    var_7 = bool(var_6 == [1, 'a', 2, 'b', 3])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == [1])
    assert var_3 is True



# Parsed testcases at query #48
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

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
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'hello'
    var_3 = 'world'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['hello', 'world'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = 1
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['hello', 'world'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'hello'
    var_3 = 'world'
    var_4 = 3
    var_5 = 4
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0._strip_artifical_nl(var_6)
    var_8 = bool(var_7 == ['hello', 'world'])
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'hello'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['hello'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['hello'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 'b'
    var_4 = 3
    var_5 = 'c'
    var_6 = 4
    var_7 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0._strip_artifical_nl(var_7)
    var_9 = bool(var_8 == ['a', 'b', 'c'])
    assert var_9 is True

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
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == [1])
    assert var_3 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = lambda : 'div'
    var_1 = None



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 7/12 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 7/12 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 10/18 statements.
# Partially parsed test_extract_text_array_with_artificial_newlines. Retrieved 6/11 statements.
# Partially parsed test_extract_text_array_with_squash_artificial_nl. Retrieved 7/12 statements.
# Partially parsed test_extract_text_array_with_strip_artificial_nl. Retrieved 10/18 statements.
# Partially parsed test_extract_text_array_with_callable_tag. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = None
    var_3 = 'p'
    var_4 = {var_3}
    var_5 = set()
    var_6 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None
    var_3 = set()
    var_4 = 'span'
    var_5 = {var_4}
    var_6 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None
    var_6 = set()
    var_7 = 'span'
    var_8 = {var_7}
    var_9 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = set()
    var_4 = set()
    var_5 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = set()
    var_4 = set()
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None
    var_6 = set()
    var_7 = set()
    var_8 = False
    var_9 = True

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_dom_text_is_not_none. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'some text'



# Parsed testcases at query #52
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'Hello'
    var_2 = True
    var_3 = 'World'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.extract_text(var_4)
    assert var_5 == '\nHello\nWorld'



# Parsed testcases at query #53
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'Hello'
    var_2 = True
    var_3 = 'World'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.extract_text(var_4)
    assert var_5 == '\nHello\nWorld'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_non_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = lambda : 'div'
    var_1 = None
    var_2 = None



# Parsed testcases at query #55
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'Hello'
    var_3 = 'World'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.extract_text(var_4)
    assert var_5 == '\n\nHelloWorld'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_extract_text_array_with_children. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = 'span'
    var_3 = 'World'
    var_4 = []
    var_5 = '!'
    var_6 = None



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_non_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_callable_tag. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = lambda : 'div'
    var_1 = None
    var_2 = None



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_non_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = None
    var_1 = None



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 5/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 5/8 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 9/13 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 5/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 5/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 6/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 6/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = lambda self: []
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = lambda self: []
    var_4 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = lambda self: []
    var_4 = 'div'
    var_5 = 'Hello'
    var_6 = None
    var_7 = lambda self: [MockChild()]
    var_8 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = lambda self: []
    var_4 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None
    var_3 = lambda self: []
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = lambda self: []
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = lambda self: []
    var_4 = False
    var_5 = True

def test_case_0():
    var_0 = lambda : 'div'
    var_1 = None
    var_2 = None
    var_3 = lambda self: []



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_squash_artifical_nl_is_false. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = ' world'
    var_2 = '  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['hello world'])
    assert var_5 is True

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
    var_1 = 1
    var_2 = ' world'
    var_3 = 2
    var_4 = '  '
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._merge_original_parts(var_5)
    var_7 = bool(var_6 == ['hello world', 1, 2])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '   '
    var_1 = '  \n  '
    var_2 = '  \t  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello '
    var_1 = ' world  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello\n'
    var_1 = '\tworld'
    var_2 = '  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['hello world'])
    assert var_5 is True



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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [1, 2, 3])
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
    var_6 = bool(var_5 == ['a', 'b'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['a', 'b'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 3
    var_5 = 4
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0._strip_artifical_nl(var_6)
    var_8 = bool(var_7 == ['a', 'b'])
    assert var_8 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'start'
    var_3 = 'middle'
    var_4 = 'end'
    var_5 = 2
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5, var_0]
    var_7 = module_0._strip_artifical_nl(var_6)
    var_8 = bool(var_7 == ['start', 'middle', 'end'])
    assert var_8 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_single_text_node. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 5/10 statements.
# Partially parsed test_extract_text_nested_tags. Retrieved 6/14 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello  '
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello  '
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = '|'
    var_4 = '||'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = ' End'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'Python'
    var_5 = '!'
    var_6 = 'div'
    var_7 = 'Hello'
    var_8 = None

def test_case_0():
    var_0 = 'strong'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_single_text_node. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 6/14 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 4/9 statements.
# Partially parsed test_extract_text_custom_block_symbol. Retrieved 7/15 statements.
# Partially parsed test_extract_text_custom_sep_symbol. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Paragraph'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None

def test_case_0():
    var_0 = 'hr'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'Before'
    var_5 = 'After'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello  '
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'p'
    var_1 = 'Paragraph'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None
    var_6 = '\n\n'

def test_case_0():
    var_0 = 'hr'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'Before'
    var_5 = 'After'
    var_6 = '|'



# Parsed testcases at query #5
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [1, 2, 3])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [1, None, 2])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 2
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [1, None, 2])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    var_3 = bool(var_2 == [None])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [None, 1, 2])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [1, 2, None])
    assert var_5 is True

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



# Parsed testcases at query #6
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'text'
    var_2 = 'Hello'
    var_3 = {var_0: var_1, var_1: var_2}
    var_4 = 'World'
    var_5 = {var_0: var_1, var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = True
    var_8 = module_0.extract_text(var_6, squash_space=var_7)
    assert var_8 == 'HelloWorld'



# Parsed testcases at query #7
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = module_0.extract_text(var_3, squash_space=var_1)
    assert var_4 == '\n\ntext'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_text_only. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_nested_text. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Paragraph'
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello  '
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = '|'
    var_4 = '-'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_custom_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_preformatted_content. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div>Hello<p>World</p></div>'

def test_case_0():
    var_0 = '<div>Hello <span>World</span></div>'

def test_case_0():
    var_0 = '<div>Hello<br/>World</div>'

def test_case_0():
    var_0 = '<div>Hello<div>World<span>!</span></div></div>'

def test_case_0():
    var_0 = '<div>Hello   <p>   World   </p>   </div>'

def test_case_0():
    var_0 = '<div>Hello<p>World</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>Hello<br/>World</div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>  Hello   <p>   World   </p>   </div>'
    var_1 = False

def test_case_0():
    var_0 = '<pre>Hello   World</pre>'

def test_case_0():
    var_0 = '<div>Hello<span>World</span><p>!</p></div>'



# Parsed testcases at query #10
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '<div>Hello World</div>'
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_squash_space_false_when_result_not_stripped. Retrieved 8/10 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '  text  '
    var_2 = True
    var_3 = 'more text'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = False
    var_6 = module_0.extract_text(var_4, squash_space=var_5)
    assert var_6 == '\n  text  \nmore text'
    var_7 = '\n  text  \nmore text'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 3/8 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_multiple_blocks. Retrieved 9/20 statements.
# Partially parsed test_extract_text_custom_block_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_custom_sep_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_nested_elements. Retrieved 9/20 statements.
# Partially parsed test_extract_text_with_whitespace_handling. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello World'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Child Text'
    var_2 = 'Tail Text'
    var_3 = 'div'
    var_4 = 'Start'
    var_5 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Paragraph'
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'First'
    var_2 = None
    var_3 = 'p'
    var_4 = 'Second'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = '|'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = '|'

def test_case_0():
    var_0 = 'b'
    var_1 = 'Bold'
    var_2 = ' text'
    var_3 = 'p'
    var_4 = 'Start '
    var_5 = ' end'
    var_6 = 'div'
    var_7 = 'Outer '
    var_8 = None

def test_case_0():
    var_0 = 'div'
    var_1 = '  \n  Text  \n  '
    var_2 = None
    var_3 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = False



# Parsed testcases at query #14
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = module_0.extract_text(var_3, squash_space=var_4)
    assert var_5 == '\n\ntext'



# Parsed testcases at query #15
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '  text  '
    var_2 = True
    var_3 = '  more text  '
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = False
    var_6 = module_0.extract_text(var_4, squash_space=var_5)
    assert var_6 == '\n  text  \n  more text  '



# Parsed testcases at query #16
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.extract_text(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #17
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'hello'
    var_2 = True
    var_3 = 'world'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = False
    var_6 = module_0.extract_text(var_4, squash_space=var_5)
    assert var_6 == '\nhello\nworld'



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_callable_tag_returns_empty_string.




# Parsed testcases at query #19
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 8/16 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_complex_case. Retrieved 12/26 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None
    var_6 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None
    var_6 = False
    var_7 = True

def test_case_0():
    var_0 = None
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = ' '
    var_3 = 'br'
    var_4 = None
    var_5 = None
    var_6 = 'span'
    var_7 = 'World'
    var_8 = '!'
    var_9 = 'div'
    var_10 = None
    var_11 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello World'
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'End'
    var_3 = 'span'
    var_4 = 'Middle'
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Line1'
    var_2 = None
    var_3 = 'p'
    var_4 = 'Line2'
    var_5 = None

def test_case_0():
    var_0 = 'hr'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Part1'
    var_2 = None
    var_3 = 'p'
    var_4 = 'Part2'
    var_5 = None
    var_6 = '|'
    var_7 = '-'



# Parsed testcases at query #21
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = module_0.extract_text(var_3, squash_space=var_4)
    assert var_5 == '\n\ntext'



# Parsed testcases at query #22
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '<div>Hello World</div>'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'Hello World'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_extract_text_with_simple_text. Retrieved 3/6 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 7/11 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 4/7 statements.
# Partially parsed test_extract_text_with_block_tag. Retrieved 4/7 statements.
# Partially parsed test_extract_text_with_squash_space. Retrieved 4/7 statements.
# Partially parsed test_extract_text_without_squash_space. Retrieved 4/7 statements.
# Partially parsed test_extract_text_with_multiple_children. Retrieved 11/16 statements.
# Partially parsed test_extract_text_with_preformatted_content. Retrieved 3/6 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.extract_text(var_0)
    assert var_1 == ''

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = lambda self: []

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = lambda self: []
    var_4 = 'div'
    var_5 = 'Hello'
    var_6 = lambda self: [MockChild()]

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = lambda self: []
    var_3 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = lambda self: []
    var_3 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello  \n  World  '
    var_2 = lambda self: []
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello  \n  World  '
    var_2 = lambda self: []
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = ' '
    var_3 = lambda self: []
    var_4 = 'span'
    var_5 = 'World'
    var_6 = None
    var_7 = lambda self: []
    var_8 = 'div'
    var_9 = None
    var_10 = lambda self: [MockChild1(), MockChild2()]

def test_case_0():
    var_0 = 'pre'
    var_1 = '  Hello  \n  World  '
    var_2 = lambda self: []



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_callable_tag_returns_empty_string. Retrieved 3/6 statements.


def test_case_0():
    var_0 = lambda : None
    var_1 = None
    var_2 = lambda : []



# Parsed testcases at query #25
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.extract_text(var_3)
    assert var_4 == '\n\ntext'



# Parsed testcases at query #26
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '<div>Hello World</div>'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'Hello World'



# Parsed testcases at query #27
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.extract_text(var_3)
    assert var_4 == '\n\ntext'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_callable_tag_returns_empty_string. Retrieved 1/4 statements.


def test_case_0():
    var_0 = lambda : None



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #31
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = '\n'
    var_2 = '\n'
    var_3 = False
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #32
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.extract_text(var_3)
    assert var_4 == '\n\ntext'



# Parsed testcases at query #33
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockElement'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'getchildren'
    var_4 = None
    var_5 = lambda : var_4
    var_6 = []
    var_7 = lambda : var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = module_1.extract_text_array(var_12)
    assert var_13 == ''



# Parsed testcases at query #34
#--------------------------

# Failed to parse test_callable_tag_returns_empty_string.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None
    var_6 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None
    var_3 = False



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_non_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_complex_case. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'span'
    var_3 = 'Middle'
    var_4 = 'End'
    var_5 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'Text'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Text'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Text'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'p'
    var_3 = 'Para1'
    var_4 = 'Tail1'
    var_5 = 'p'
    var_6 = 'Para2'
    var_7 = 'Tail2'
    var_8 = True



# Parsed testcases at query #37
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'MockElement'
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



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 6/11 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 6/11 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 9/17 statements.
# Partially parsed test_extract_text_array_with_artificial_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 6/11 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 10/18 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None
    var_3 = 'span'
    var_4 = {var_3}
    var_5 = set()

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = set()
    var_4 = 'br'
    var_5 = {var_4}

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = 'End'
    var_3 = 'span'
    var_4 = 'Middle'
    var_5 = None
    var_6 = 'span'
    var_7 = {var_6}
    var_8 = set()

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = None
    var_3 = set()
    var_4 = set()

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = set()
    var_4 = set()
    var_5 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = 'span'
    var_4 = 'Middle'
    var_5 = None
    var_6 = 'span'
    var_7 = {var_6}
    var_8 = set()
    var_9 = True

def test_case_0():
    var_0 = 'Hello'
    var_1 = None



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_non_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_complex_case. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Child'
    var_2 = 'Tail'
    var_3 = 'div'
    var_4 = 'Parent'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'Child'
    var_2 = 'Tail'
    var_3 = 'div'
    var_4 = 'Parent'
    var_5 = True



# Parsed testcases at query #40
#--------------------------

# Failed to parse test_predicate_at_line_12_evaluates_to_false.




# Parsed testcases at query #41
#--------------------------

# Partially parsed test_extract_text_array_with_strip_artifical_nl_false. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = None
    var_5 = False



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_extract_text_array_with_strip_artifical_nl_false. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = 'p'
    var_3 = 'World'
    var_4 = '!'
    var_5 = False



# Parsed testcases at query #43
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
    var_8 = lambda : var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.extract_text_array(var_13)
    var_15 = bool(var_14 == [None, None])
    assert var_15 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_complex_case. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = 'span'
    var_4 = 'World'
    var_5 = None
    var_6 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = 'Tail'
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Inline'
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Start'
    var_2 = None
    var_3 = 'p'
    var_4 = 'Para'
    var_5 = 'Tail1'
    var_6 = 'span'
    var_7 = 'Span'
    var_8 = 'Tail2'
    var_9 = False



# Parsed testcases at query #45
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockElement'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'div'
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



# Parsed testcases at query #46
#--------------------------

# Failed to parse test_dom_text_is_not_none.




# Parsed testcases at query #47
#--------------------------

# Partially parsed test_squash_artifical_nl_is_false. Retrieved 1/7 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #48
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockElement'
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
    var_15 = bool(var_14 == [None, None])
    assert var_15 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_strip_artifical_nl_is_true. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = None
    var_5 = True



