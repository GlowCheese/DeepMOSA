####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello   world  '
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == ['hello world'])
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
    var_0 = '  hello '
    var_1 = 1
    var_2 = ' world  '
    var_3 = 2
    var_4 = '  foo  '
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._merge_original_parts(var_5)
    var_7 = bool(var_6 == ['hello', 1, 'world', 2, 'foo'])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  '
    var_1 = ''
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '   '
    var_1 = ' \n '
    var_2 = '  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = ' world'
    var_2 = 'foo bar'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['hello world foo bar'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello world  '
    var_1 = '  foo bar  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world foo bar'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = ' hello '
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [1, 'hello', 2])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = ' hello '
    var_1 = 1
    var_2 = ' world '
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._merge_original_parts(var_4)
    var_6 = bool(var_5 == ['hello', 1, 'world', 2])
    assert var_6 is True

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
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == [42])
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
    var_0 = '  hello '
    var_1 = ' world  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

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
    var_0 = 'hello'
    var_1 = 42
    var_2 = 'world'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['hello', 42, 'world'])
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
    var_1 = 42
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello', 42])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'world'
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == [42, 'world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 'd'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._merge_original_parts(var_5)
    var_7 = bool(var_6 == ['a b', 1, 'c d'])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = ''
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [1, 2])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0, var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_multiple_paragraphs. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 3/6 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '<p>Hello</p>'

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'

def test_case_0():
    var_0 = '<p>First</p><p>Second</p>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'

def test_case_0():
    var_0 = '<hr>'

def test_case_0():
    var_0 = '<p><b>Bold</b> text</p>'

def test_case_0():
    var_0 = '<p>Hello<br/>World</p>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>First</p><p>Second</p>'
    var_1 = ' | '
    var_2 = ' - '

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'
    var_1 = False



# Parsed testcases at query #4
#--------------------------




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
    var_3 = 'c'
    var_4 = [var_0, var_1, var_2, var_1, var_1, var_3]
    var_5 = module_0._squash_artifical_nl(var_4)
    var_6 = bool(var_5 == ['a', None, 'b', None, 'c'])
    assert var_6 is True

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
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [None, 'a', 'b'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b', None])
    assert var_5 is True

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
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_extract_text_basic_paragraph. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_leading_trailing_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_document. Retrieved 1/4 statements.
# Partially parsed test_extract_text_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '<p>Hello world</p>'

def test_case_0():
    var_0 = '<hr/><p>Text after separator</p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<p>Hello <b>bold</b> world</p>'

def test_case_0():
    var_0 = '<p>Hello    world</p>'

def test_case_0():
    var_0 = '<p>  Hello world  </p>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'
    var_1 = '<br>'

def test_case_0():
    var_0 = '<hr/><p>Text</p>'
    var_1 = '---'

def test_case_0():
    var_0 = '<p>Hello    world</p>'
    var_1 = False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_inline_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_separator_between_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 3/6 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<hr/>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>Hello <b>World</b></p>'

def test_case_0():
    var_0 = '<p>Before</p><hr/><p>After</p>'

def test_case_0():
    var_0 = '<div><p>Line1</p><p>Line2</p></div>'
    var_1 = '|'
    var_2 = '-'

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'
    var_1 = False

def test_case_0():
    var_0 = '<p>Hello</p>World'



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
    var_1 = 'world'
    var_2 = [var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)
    var_4 = bool(var_3 == ['hello', 'world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = 'hello'
    var_2 = [var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)
    var_4 = bool(var_3 == ['hello'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = '\n'
    var_2 = [var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)
    var_4 = bool(var_3 == ['hello'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = 'hello'
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._strip_artifical_nl(var_2)
    var_4 = bool(var_3 == ['hello'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = 'hello'
    var_2 = [var_0, var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)
    var_4 = bool(var_3 == ['hello'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = '\n'
    var_2 = [var_0, var_1, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)
    var_4 = bool(var_3 == ['hello'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = [var_0, var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = 1
    var_2 = 'hello'
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3, var_0]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == [1, 'hello', 2])
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
    var_0 = '\n'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True



# Parsed testcases at query #8
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
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == [42])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 1, 'b'])
    assert var_5 is True

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
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [1, 'a', 2])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 1, 2])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 3
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._strip_artifical_nl(var_5)
    var_7 = bool(var_6 == [1, 2, 'a', 'b', 3])
    assert var_7 is True

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
    var_1 = 2
    var_2 = 3
    var_3 = 'a'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == [1, 2, 3, 'a'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['a', 1, 2, 3])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == [1, 'x', 'y', 2])
    assert var_6 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_child. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 2/7 statements.
# Partially parsed test_extract_text_multiple_blocks. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_squash_space. Retrieved 2/6 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 2/6 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 3/10 statements.
# Partially parsed test_extract_text_nested_blocks. Retrieved 2/9 statements.


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
    var_2 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = True

def test_case_0():
    var_0 = 'div'
    var_1 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'



# Parsed testcases at query #10
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    assert var_1 is True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_block_tag. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 2/8 statements.
# Partially parsed test_extract_text_multiline_block. Retrieved 3/10 statements.
# Partially parsed test_extract_text_squash_whitespace. Retrieved 1/5 statements.
# Partially parsed test_extract_text_strip_artifical_newlines. Retrieved 2/7 statements.
# Partially parsed test_extract_text_custom_block_symbol. Retrieved 3/10 statements.
# Partially parsed test_extract_text_custom_sep_symbol. Retrieved 3/8 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 2/6 statements.
# Partially parsed test_extract_text_nested_tags. Retrieved 2/10 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_only_whitespace. Retrieved 1/5 statements.
# Partially parsed test_extract_text_block_and_separator. Retrieved 3/10 statements.


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
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'br'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = ' | '

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = ' - '

def test_case_0():
    var_0 = 'p'
    var_1 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'hr'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extract_text_array_with_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_artifical_nl_squash. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_strip_artifical_nl. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_without_squash. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_with_empty. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_callable_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_array_with_separator_in_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_nested_inline. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_artifical_nl_at_ends. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'inline'
    var_2 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'text'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None
    var_6 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'a'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None
    var_6 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = lambda : None
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' and '
    var_3 = 'i'
    var_4 = 'italic'
    var_5 = None
    var_6 = 'p'
    var_7 = 'start '
    var_8 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = 'after'
    var_3 = 'p'
    var_4 = 'before'
    var_5 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = None
    var_3 = 'span'
    var_4 = 'outer'
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'middle'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None
    var_6 = True



# Parsed testcases at query #13
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_extract_text_with_plain_text_node. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_pre_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_only_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 3/6 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_comment_node. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello world</p>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'

def test_case_0():
    var_0 = '<hr><p>Text</p>'

def test_case_0():
    var_0 = '<pre>  Hello   World  </pre>'

def test_case_0():
    var_0 = '<p>Hello <b>bold</b> world</p>'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '<p>   </p>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div><hr><p>C</p>'
    var_1 = '|'
    var_2 = ':'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><!-- comment --><p>Text</p></div>'



# Parsed testcases at query #16
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = False
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)



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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_squash_space_disabled. Retrieved 2/5 statements.
# Partially parsed test_extract_text_block_symbol_custom. Retrieved 2/5 statements.
# Partially parsed test_extract_text_sep_symbol_custom. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<p>Hello <b>World</b></p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<div><br/>Line</div>'

def test_case_0():
    var_0 = '<div>  Hello   World  </div>'

def test_case_0():
    var_0 = '<div><span>Hello</span> <span>World</span></div>'

def test_case_0():
    var_0 = '<div>Hello<b>bold</b>World</div>'

def test_case_0():
    var_0 = '<div>  Hello  </div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'
    var_1 = ' '

def test_case_0():
    var_0 = '<div><br/>Line</div>'
    var_1 = ' '



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_predicate_at_line10_evaluates_to_true.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_nested_tags. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_child_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_none_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = 'hello '

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'content'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'start'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'

def test_case_0():
    var_0 = 'p'
    var_1 = None

def test_case_0():
    var_0 = lambda : None
    var_1 = None



# Parsed testcases at query #22
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
    var_15 = True
    var_16 = module_1.extract_text_array(var_13, var_14, var_15)
    var_17 = bool(var_16 == [])
    assert var_17 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_array_no_artifical_nl_options. Retrieved 2/5 statements.
# Partially parsed test_extract_text_array_empty_text_nodes. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_only_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_deep_nesting. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_mixed_inline_and_block. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>Hello</p>'

def test_case_0():
    var_0 = '<br/>'

def test_case_0():
    var_0 = '<span>inline</span>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<p>Hello<b>bold</b>world</p>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p>A</p></div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p>A</p></div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p></p></div>'

def test_case_0():
    var_0 = '<div>   </div>'

def test_case_0():
    var_0 = '<div></div>'
    var_1 = None

def test_case_0():
    var_0 = '<div><p><span>deep</span></p></div>'

def test_case_0():
    var_0 = '<br/><br/>'

def test_case_0():
    var_0 = '<div><span>inline</span><p>block</p></div>'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_extract_text_array_with_text_and_no_children. Retrieved 11/15 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 11/15 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 11/15 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 15/25 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 12/16 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 12/16 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'Element'
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
    var_0 = 'Element'
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

def test_case_0():
    var_0 = 'Element'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'span'
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = None
    var_9 = {var_1: var_5, var_2: var_2, var_3: var_7, var_4: var_8}
    var_10 = False

def test_case_0():
    var_0 = 'Element'
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
    var_14 = False

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
    var_10 = True
    var_11 = False

def test_case_0():
    var_0 = 'Element'
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

def test_case_0():
    var_0 = 'Element'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = None
    var_6 = lambda : var_5
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_6, var_2: var_2, var_3: var_8, var_4: var_5}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 10/17 statements.


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
    var_9 = 0



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_extract_text_with_squash_space_true. Retrieved 3/4 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 17/18 statements.


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
    var_7 = 'hello'
    var_8 = None
    var_9 = []
    var_10 = lambda : var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = True
    var_17 = False
    var_18 = module_1.extract_text_array(var_15, var_16, var_17)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_extract_text_array_with_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_artificial_newlines. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_squash_and_strip. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_without_squash. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_without_strip. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_mixed_content. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_separator_and_text. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'
    var_2 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' tail'
    var_3 = 'span'
    var_4 = 'start'
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'para'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None
    var_6 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = None
    var_3 = 'p'
    var_4 = 'text'
    var_5 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = 'after'



# Parsed testcases at query #29
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
    var_6 = 'INLINE'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.extract_text_array(var_14)



# Parsed testcases at query #30
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
    var_9 = lambda : var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = 'span'
    var_16 = 'b'
    var_17 = 'i'
    var_18 = {var_15, var_16, var_17}
    var_19 = 'br'
    var_20 = 'hr'
    var_21 = {var_19, var_20}
    var_22 = module_1.extract_text_array(var_14)
    var_23 = var_22[-1]
    assert var_23 is None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_false. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'p'
    var_5 = 'hello'
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = []



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div>text</div>'
    var_1 = bool(True)
    assert var_1 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_single_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_newlines. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_strip_newlines. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_both_squash_and_strip. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'p'
    var_4 = 'before '
    var_5 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = lambda : None
    var_1 = None

def test_case_0():
    var_0 = 'a'
    var_1 = 'link'

def test_case_0():
    var_0 = 'i'
    var_1 = 'italic'
    var_2 = ' after'
    var_3 = 'p'
    var_4 = 'before '
    var_5 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'content'
    var_2 = None
    var_3 = 'div'
    var_4 = None



# Parsed testcases at query #34
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = '\n'
    var_3 = False
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 10/14 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 11/15 statements.
# Partially parsed test_extract_text_with_child. Retrieved 14/24 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 11/15 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 12/22 statements.
# Partially parsed test_extract_text_multiple_blocks. Retrieved 16/29 statements.
# Partially parsed test_extract_text_with_tail_on_separator. Retrieved 13/23 statements.


def test_case_0():
    var_0 = 'obj'
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
    var_0 = 'obj'
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
    var_0 = 'obj'
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
    var_0 = 'obj'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'br'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = '\n'

def test_case_0():
    var_0 = 'obj'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = 'Para'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'div'

def test_case_0():
    var_0 = 'obj'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = 'First'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'Second'
    var_12 = []
    var_13 = lambda self: var_12
    var_14 = {var_1: var_5, var_2: var_11, var_3: var_13, var_4: var_9}
    var_15 = 'div'

def test_case_0():
    var_0 = 'obj'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'br'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = ' after'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'div'
    var_12 = 'before '



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_predicate_line_17_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>text</p>'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 2/5 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 2/5 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello world</p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'
    var_1 = '\n'

def test_case_0():
    var_0 = '<div><br/>Separator</div>'
    var_1 = '\n'

def test_case_0():
    var_0 = '<p>  Hello   world  </p>'
    var_1 = True

def test_case_0():
    var_0 = '<p>  Hello   world  </p>'
    var_1 = False

def test_case_0():
    var_0 = '<p>Hello <b>bold</b> world</p>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Text before <span>inner</span> tail after</div>'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 11/16 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 12/17 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 11/16 statements.
# Partially parsed test_extract_text_array_with_artificial_newlines. Retrieved 11/16 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 12/17 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 12/17 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 15/27 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 12/17 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 13/25 statements.
# Partially parsed test_extract_text_array_with_separator_and_text. Retrieved 11/16 statements.
# Partially parsed test_extract_text_array_none_text. Retrieved 11/16 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 12/17 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 20/36 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 12/17 statements.
# Partially parsed test_extract_text_array_strip_leading_none. Retrieved 13/25 statements.
# Partially parsed test_extract_text_array_strip_trailing_none. Retrieved 13/25 statements.


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
    var_10 = True

def test_case_0():
    var_0 = 'MockDom'
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
    var_11 = True

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
    var_10 = True

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
    var_10 = False

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
    var_10 = True
    var_11 = False

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
    var_10 = False
    var_11 = True

def test_case_0():
    var_0 = 'MockDom'
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
    var_11 = 'div'
    var_12 = 'Hello '
    var_13 = None
    var_14 = True

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'b'
    var_6 = 'Bold'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = True

def test_case_0():
    var_0 = 'MockDom'
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
    var_11 = 'div'
    var_12 = False

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
    var_10 = True

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

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = 'Hello'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = ' World'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = True

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'b'
    var_6 = 'Bold'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = ' '
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'i'
    var_12 = 'Italic'
    var_13 = []
    var_14 = lambda self: var_13
    var_15 = None
    var_16 = {var_1: var_11, var_2: var_12, var_3: var_14, var_4: var_15}
    var_17 = 'p'
    var_18 = 'Text '
    var_19 = True

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = None
    var_6 = lambda : var_5
    var_7 = 'should not appear'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_1: var_6, var_2: var_7, var_3: var_9, var_4: var_5}
    var_11 = True

def test_case_0():
    var_0 = 'MockDom'
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
    var_11 = 'div'
    var_12 = True

def test_case_0():
    var_0 = 'MockDom'
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
    var_11 = 'div'
    var_12 = True



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
    var_5 = 'p'
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
    var_16 = bool(var_15 == [])
    assert var_16 is True



# Parsed testcases at query #40
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
    var_9 = lambda : var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = True
    var_16 = False
    var_17 = module_1.extract_text_array(var_14, var_15, var_16)
    var_18 = bool(True)
    assert var_18 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 7/14 statements.
# Partially parsed test_extract_text_array_with_artificial_nl. Retrieved 6/13 statements.
# Partially parsed test_extract_text_array_with_child_and_tail. Retrieved 10/20 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 7/14 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 7/14 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 7/14 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 13/26 statements.
# Partially parsed test_extract_text_array_separator_with_children. Retrieved 11/21 statements.
# Partially parsed test_extract_text_array_squash_consecutive_nl. Retrieved 9/19 statements.
# Partially parsed test_extract_text_array_strip_leading_trailing_nl. Retrieved 9/19 statements.
# Partially parsed test_extract_text_array_mixed_content. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = 'your_module.SEPARATORS'
    var_3 = 'br'
    var_4 = {var_3}
    var_5 = 'your_module.INLINE_TAGS'
    var_6 = set()

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
    var_2 = 'your_module.SEPARATORS'
    var_3 = set()
    var_4 = 'your_module.INLINE_TAGS'
    var_5 = set()

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'before '
    var_5 = 'your_module.SEPARATORS'
    var_6 = set()
    var_7 = 'your_module.INLINE_TAGS'
    var_8 = 'b'
    var_9 = {var_8}

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
    var_2 = 'your_module.SEPARATORS'
    var_3 = set()
    var_4 = 'your_module.INLINE_TAGS'
    var_5 = set()
    var_6 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
    var_2 = 'your_module.SEPARATORS'
    var_3 = set()
    var_4 = 'your_module.INLINE_TAGS'
    var_5 = set()
    var_6 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
    var_2 = 'your_module.SEPARATORS'
    var_3 = set()
    var_4 = 'your_module.INLINE_TAGS'
    var_5 = set()
    var_6 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'child1'
    var_2 = ' tail1'
    var_3 = 'span'
    var_4 = 'child2'
    var_5 = None
    var_6 = 'div'
    var_7 = 'start '
    var_8 = 'your_module.SEPARATORS'
    var_9 = set()
    var_10 = 'your_module.INLINE_TAGS'
    var_11 = 'span'
    var_12 = {var_11}

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = None
    var_3 = 'br'
    var_4 = None
    var_5 = 'your_module.SEPARATORS'
    var_6 = 'br'
    var_7 = {var_6}
    var_8 = 'your_module.INLINE_TAGS'
    var_9 = 'span'
    var_10 = {var_9}

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = 'div'
    var_3 = 'A'
    var_4 = 'your_module.SEPARATORS'
    var_5 = set()
    var_6 = 'your_module.INLINE_TAGS'
    var_7 = set()
    var_8 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'B'
    var_2 = 'div'
    var_3 = None
    var_4 = 'your_module.SEPARATORS'
    var_5 = set()
    var_6 = 'your_module.INLINE_TAGS'
    var_7 = set()
    var_8 = True

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' after bold '
    var_3 = 'p'
    var_4 = 'before '
    var_5 = 'your_module.SEPARATORS'
    var_6 = 'br'
    var_7 = {var_6}
    var_8 = 'your_module.INLINE_TAGS'
    var_9 = 'b'
    var_10 = {var_9}



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_strip_artifical_nl_true. Retrieved 2/7 statements.


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_dom_tag_in_separators. Retrieved 10/14 statements.


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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 13/18 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'p'
    var_5 = 'hello'
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_1: var_4, var_2: var_5, var_3: var_7}
    var_9 = []
    var_10 = []
    var_11 = True
    var_12 = False



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_nested_inline. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_squash_nl. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_strip_leading_nl. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_strip_trailing_nl. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_no_squash. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_no_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 4/14 statements.
# Partially parsed test_extract_text_array_with_separator_and_text. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'br'

def test_case_0():
    var_0 = 'span'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

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
    var_2 = 'b'
    var_3 = 'i'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_full_processing. Retrieved 5/13 statements.


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
    var_1 = 'text'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'p'
    var_4 = 'before'

def test_case_0():
    var_0 = 'a'
    var_1 = 'link'
    var_2 = ' after'
    var_3 = 'p'
    var_4 = 'click '

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_0, var_1, var_0, var_0, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [None, 'a', None, 'b'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b'])
    assert var_5 is True

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = 'hello '



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child_and_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_and_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 8/19 statements.
# Partially parsed test_extract_text_array_nested_separator. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_without_squash. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_without_strip. Retrieved 3/8 statements.


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
    var_1 = 'Inline'

def test_case_0():
    var_0 = 'b'
    var_1 = 'Bold'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'Text '

def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'
    var_2 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'
    var_2 = True

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'i'
    var_4 = 'italic'
    var_5 = ' after'
    var_6 = 'p'
    var_7 = 'start '

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'line1'

def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = False



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 12/16 statements.


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
    var_12 = bool(True)
    assert var_12 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_strip_artifical_nl_true. Retrieved 17/18 statements.


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



# Parsed testcases at query #50
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'text'
    var_2 = [var_0, var_1, var_0]
    var_3 = True
    assert var_3 is True
    var_4 = module_0._strip_artifical_nl(var_2)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_separator. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_nested_tags. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_children_with_tail. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello '
    var_5 = None
    var_6 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = None
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'a'
    var_1 = 'click'
    var_2 = ' here'
    var_3 = 'p'
    var_4 = 'Please '
    var_5 = None
    var_6 = True

def test_case_0():
    var_0 = lambda : None
    var_1 = 'test'
    var_2 = None
    var_3 = True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_block_tag. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_trailing_artificial_nl_stripped. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_squash_artificial_nl_false. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_false. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_nested_blocks. Retrieved 2/10 statements.
# Partially parsed test_extract_text_array_multiple_separators. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'br'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = False

def test_case_0():
    var_0 = 'p'
    var_1 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'
    var_2 = 'span'



# Parsed testcases at query #53
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #54
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_predicate_line12_true. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'br'
    var_1 = {var_0}
    var_2 = 'b'
    var_3 = 'i'
    var_4 = {var_2, var_3}



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_text_only. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_block_tag_with_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_nested_tags. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_separator_with_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_mixed_inline_block. Retrieved 2/10 statements.
# Partially parsed test_extract_text_array_artifical_nl_between_text. Retrieved 1/8 statements.
# Partially parsed test_extract_text_array_multiple_separators. Retrieved 2/7 statements.


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

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

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

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_single_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child_and_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_nested_structure. Retrieved 8/19 statements.
# Partially parsed test_extract_text_array_separator_with_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_multiple_artifical_nl_squashed. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_leading_and_trailing_nl_stripped. Retrieved 2/7 statements.


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
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'start '

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = ' after'
    var_6 = 'body'
    var_7 = 'before '

def test_case_0():
    var_0 = 'br'
    var_1 = 'text'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'br'
    var_4 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'mid'



# Parsed testcases at query #58
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

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = 'hello'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = module_1.extract_text_array(var_15)
    var_17 = bool(var_16 == ['hello'])
    assert var_17 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'br'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.extract_text_array(var_14)
    var_16 = bool(var_15 == [True])
    assert var_16 is True

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

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = 'world'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = '!'
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'div'
    var_18 = 'hello '
    var_19 = [var_15]
    var_20 = lambda self: var_19
    var_21 = None
    var_22 = {var_2: var_17, var_3: var_18, var_4: var_20, var_5: var_21}
    var_23 = [var_0, var_16, var_22]
    var_24 = {}
    var_25 = module_0.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = module_1.extract_text_array(var_26)
    var_28 = bool(var_27 == ['hello ', 'world', '!'])
    assert var_28 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = 'hello'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'div'
    var_18 = [var_15]
    var_19 = lambda self: var_18
    var_20 = {var_2: var_17, var_3: var_10, var_4: var_19, var_5: var_10}
    var_21 = [var_0, var_16, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()
    var_25 = module_1.extract_text_array(var_24)
    var_26 = bool(var_25 == ['hello'])
    assert var_26 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = 'hello'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'div'
    var_18 = [var_15]
    var_19 = lambda self: var_18
    var_20 = {var_2: var_17, var_3: var_10, var_4: var_19, var_5: var_10}
    var_21 = [var_0, var_16, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()
    var_25 = module_1.extract_text_array(var_24)
    var_26 = bool(var_25 == ['hello'])
    assert var_26 is True

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
    var_7 = 'a'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = [var_15]
    var_18 = lambda self: var_17
    var_19 = {var_2: var_6, var_3: var_10, var_4: var_18, var_5: var_10}
    var_20 = [var_0, var_16, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = False
    var_25 = module_1.extract_text_array(var_23, var_24)
    var_26 = bool(var_25 == [None, 'a', None, None])
    assert var_26 is True

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
    var_7 = 'a'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = [var_15]
    var_18 = lambda self: var_17
    var_19 = {var_2: var_6, var_3: var_10, var_4: var_18, var_5: var_10}
    var_20 = [var_0, var_16, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = False
    var_25 = module_1.extract_text_array(var_23, strip_artifical_nl=var_24)
    var_26 = bool(var_25 == [None, 'a', None])
    assert var_26 is True



# Parsed testcases at query #59
#--------------------------




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
    var_14 = True
    var_15 = module_1.extract_text_array(var_13, var_14, var_14)
    var_16 = bool(var_15 == [None, None] or var_15 == [])
    assert var_16 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_predicate_line12_false. Retrieved 1/10 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #61
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
    var_16 = bool(var_15 == [None, None])
    assert var_16 is True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_extract_text_array_empty_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_inline_tag_with_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_nested_inline. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_block_with_children. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/6 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<span>hello</span>'

def test_case_0():
    var_0 = '<br/>'

def test_case_0():
    var_0 = '<span>hello <b>world</b></span>'

def test_case_0():
    var_0 = '<div><p>first</p><p>second</p></div>'

def test_case_0():
    var_0 = '<div>a</div>'
    var_1 = True
    var_2 = False



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'not_INLINE_TAGS_not_SEPARATORS'
    var_1 = None
    var_2 = False
    var_3 = None



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_preformatted. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_blocks. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator_and_text. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<hr>'

def test_case_0():
    var_0 = '<span>Hello <b>World</b></span>'

def test_case_0():
    var_0 = '<pre>  Hello\n  World  </pre>'

def test_case_0():
    var_0 = '<div>  Hello   World  </div>'

def test_case_0():
    var_0 = '<div><section><p>A</p></section><p>B</p></div>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Start<b>bold</b>End</div>'

def test_case_0():
    var_0 = '<hr>Separator'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_returns_plain_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_none_tag_after_separator. Retrieved 2/12 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/13 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/13 statements.
# Partially parsed test_extract_text_array_squash_false_preserves_none. Retrieved 3/13 statements.
# Partially parsed test_extract_text_array_strip_false_keeps_leading_trailing. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False



# Parsed testcases at query #66
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #67
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #68
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = False
    var_3 = False
    var_4 = module_0._strip_artifical_nl(var_1)
    var_5 = var_4 if var_2 else var_1
    var_6 = bool(var_5 == [None])
    assert var_6 is True



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_predicate_line17_false. Retrieved 20/21 statements.


import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
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
    var_14 = 'span'
    var_15 = 'a'
    var_16 = 'b'
    var_17 = {var_14, var_15, var_16}
    var_18 = 'br'
    var_19 = 'hr'
    var_20 = {var_18, var_19}
    var_21 = module_1.extract_text_array(var_13)
    var_22 = bool(True)
    assert var_22 is True



# Parsed testcases at query #70
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #71
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 == var_2)
    assert var_3 is True



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_extract_text_returns_string_for_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_handles_nested_inline_tags. Retrieved 2/9 statements.
# Partially parsed test_extract_text_handles_block_tags. Retrieved 2/9 statements.
# Partially parsed test_extract_text_handles_separator_tags. Retrieved 2/7 statements.
# Partially parsed test_extract_text_handles_preformatted_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_handles_nested_block_and_inline. Retrieved 3/12 statements.
# Partially parsed test_extract_text_returns_empty_string_for_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_strips_trailing_newlines. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'p'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'

def test_case_0():
    var_0 = 'pre'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'b'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_squash_artifical_nl_true_when_squash_artifical_nl_is_true. Retrieved 13/17 statements.


def test_case_0():
    var_0 = 'MockDom'
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
    var_11 = True
    var_12 = False
    var_13 = bool(True)
    assert var_13 is True



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 7/11 statements.


def test_case_0():
    var_0 = None
    var_1 = True
    assert var_1 is True
    var_2 = []
    var_3 = ''
    var_4 = None
    var_5 = '\n'
    var_6 = True



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_squash_true. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_true. Retrieved 4/10 statements.
# Partially parsed test_extract_text_array_complex. Retrieved 3/13 statements.


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
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'span'
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'b'
    var_2 = 'i'



# Parsed testcases at query #76
#--------------------------

# Failed to parse test_predicate_at_line_17_evaluates_to_false.




# Parsed testcases at query #77
#--------------------------

# Partially parsed test_strip_artifical_nl_true. Retrieved 13/17 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = 'hello'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = False
    var_12 = True



# Parsed testcases at query #78
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDOM'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'div'
    var_6 = 'some text'
    var_7 = []
    var_8 = lambda : var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.extract_text_array(var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_14[0]
    assert var_16 is None
    var_17 = var_14[1]
    assert var_17 == 'some text'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_predicate_line20_evaluates_false. Retrieved 11/15 statements.


def test_case_0():
    var_0 = 'MockDOM'
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



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_predicate_line12_true. Retrieved 1/14 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 2/5 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_only_tail. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_block. Retrieved 1/4 statements.
# Partially parsed test_extract_text_multiple_blocks. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<p>Hello <b>World</b></p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'
    var_1 = ' | '

def test_case_0():
    var_0 = '<div><br/>Separator</div>'
    var_1 = ' | '

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'
    var_1 = True

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'
    var_1 = False

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Text</div>'

def test_case_0():
    var_0 = '<div><section><p>Paragraph</p></section></div>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p><p>Third</p></div>'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_strip_artifical_nl_false. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'
    var_2 = []
    var_3 = False



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 11/15 statements.
# Partially parsed test_extract_text_with_newline. Retrieved 11/15 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 10/14 statements.
# Partially parsed test_extract_text_nested. Retrieved 14/24 statements.
# Partially parsed test_extract_text_multiple_blocks. Retrieved 16/29 statements.
# Partially parsed test_extract_text_with_separator_inline. Retrieved 13/23 statements.
# Partially parsed test_extract_text_empty. Retrieved 10/14 statements.
# Partially parsed test_extract_text_whitespace_squash. Retrieved 11/15 statements.
# Partially parsed test_extract_text_nested_with_tail. Retrieved 14/24 statements.


def test_case_0():
    var_0 = 'obj'
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

def test_case_0():
    var_0 = 'obj'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = 'line1'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}

def test_case_0():
    var_0 = 'obj'
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
    var_0 = 'obj'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'span'
    var_6 = 'world'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = '!'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'div'
    var_12 = 'hello '
    var_13 = None

def test_case_0():
    var_0 = 'obj'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = 'first'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'second'
    var_12 = []
    var_13 = lambda self: var_12
    var_14 = {var_1: var_5, var_2: var_11, var_3: var_13, var_4: var_9}
    var_15 = 'div'

def test_case_0():
    var_0 = 'obj'
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
    var_12 = 'b'

def test_case_0():
    var_0 = 'obj'
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
    var_0 = 'obj'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'span'
    var_6 = '  hello   world  '
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}

def test_case_0():
    var_0 = 'obj'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'b'
    var_6 = 'bold'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = ' normal'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'p'
    var_12 = 'text '
    var_13 = None



# Parsed testcases at query #84
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Dom'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = var_5()
    var_7 = False
    var_8 = module_1.extract_text(var_6, squash_space=var_7)
    assert var_8 == ''



# Parsed testcases at query #85
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_child. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/9 statements.
# Partially parsed test_extract_text_with_nested_inline. Retrieved 2/8 statements.
# Partially parsed test_extract_text_strip_whitespace. Retrieved 1/5 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_none_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_block_symbol_custom. Retrieved 3/10 statements.
# Partially parsed test_extract_text_sep_symbol_custom. Retrieved 4/10 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/6 statements.


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

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = '<br>'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = 'p'
    var_3 = '---'

def test_case_0():
    var_0 = 'p'
    var_1 = False



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/8 statements.


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
    var_1 = 'inline'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'div'
    var_4 = 'before '

def test_case_0():
    var_0 = 'a'
    var_1 = 'link'
    var_2 = ' after'
    var_3 = 'p'
    var_4 = 'click '

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = False



# Parsed testcases at query #88
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
    var_7 = 'hello'
    var_8 = None
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = False
    var_17 = True
    var_18 = module_1.extract_text_array(var_15, var_16, var_17)
    var_19 = bool(var_18 == ['hello'])
    assert var_19 is True



# Parsed testcases at query #89
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
    var_17 = bool(var_16 == [None, None])
    assert var_17 is True



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_extract_text_simple_paragraph. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/11 statements.
# Partially parsed test_extract_text_multiple_paragraphs. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 2/9 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_inline. Retrieved 2/9 statements.
# Partially parsed test_extract_text_block_symbol_custom. Retrieved 3/10 statements.
# Partially parsed test_extract_text_sep_symbol_custom. Retrieved 4/10 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/10 statements.
# Partially parsed test_extract_text_with_whitespace_in_text. Retrieved 1/5 statements.


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
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'
    var_1 = 'a'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'br'
    var_3 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False

def test_case_0():
    var_0 = 'p'



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_nl_squash. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_with_nl_strip. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'hello'
    var_2 = False
    var_3 = True



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_extract_text_array_empty_dom_with_no_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_dom_with_string_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_dom_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_text_and_child_and_tail. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_and_strip. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_callable_tag_returns_empty_string. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_array_nested_children. Retrieved 9/20 statements.


def test_case_0():
    var_0 = None
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = 'tail'
    var_3 = 'div'
    var_4 = None
    var_5 = False

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
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = ' after '
    var_3 = 'p'
    var_4 = 'before '
    var_5 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'content'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = lambda : None
    var_1 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' '
    var_3 = 'i'
    var_4 = 'italic'
    var_5 = None
    var_6 = 'p'
    var_7 = None
    var_8 = False

def test_case_0():
    var_0 = 'strong'
    var_1 = 'nested'
    var_2 = None
    var_3 = 'span'
    var_4 = None
    var_5 = ' end'
    var_6 = 'div'
    var_7 = 'start '
    var_8 = False



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 13/17 statements.


def test_case_0():
    var_0 = 'MockDom'
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
    var_11 = True
    var_12 = False



# Parsed testcases at query #94
#--------------------------

# Failed to parse test_for_loop_iterates_over_children.




# Parsed testcases at query #95
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_artifical_nl. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator_and_text. Retrieved 4/12 statements.
# Partially parsed test_extract_text_array_nested. Retrieved 8/19 statements.


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
    var_0 = 'b'
    var_1 = 'bold'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = 'parent'

def test_case_0():
    var_0 = 'a'
    var_1 = 'link'
    var_2 = ' after'
    var_3 = 'p'
    var_4 = 'before '

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'b'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'c'
    var_2 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = 'p'
    var_3 = 'line1'

def test_case_0():
    var_0 = 'i'
    var_1 = 'italic'
    var_2 = None
    var_3 = 'b'
    var_4 = 'bold '
    var_5 = ' end'
    var_6 = 'div'
    var_7 = 'start '



# Parsed testcases at query #96
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 1/4 statements.
# Partially parsed test_extract_text_squash_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello</div>'

def test_case_0():
    var_0 = '<div><br/></div>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<div><b>Bold</b> text</div>'

def test_case_0():
    var_0 = '<div>  Hello   World  </div>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>A<br/>B</div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>  Hello </div>'
    var_1 = False



# Parsed testcases at query #98
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 1/4 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_block_symbol_custom. Retrieved 2/5 statements.
# Partially parsed test_extract_text_sep_symbol_custom. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_pre_tag. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello world</p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'
    var_1 = '\n'

def test_case_0():
    var_0 = '<div><br/>Break</div>'
    var_1 = '\n'

def test_case_0():
    var_0 = '<div><span>Inner</span> Outer</div>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>Hello <b>bold</b> world</p>'

def test_case_0():
    var_0 = '<div>  Multiple   spaces  </div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p>Line1</p><p>Line2</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>Text<br/>More</div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<pre>  Preserved  </pre>'



# Parsed testcases at query #100
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_squash_artifical_nl_false_when_squash_artifical_nl_is_false. Retrieved 10/14 statements.


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



# Parsed testcases at query #102
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
    var_14 = module_1.extract_text(var_13)
    assert var_14 == ''

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
    var_7 = 'Hello'
    var_8 = []
    var_9 = lambda : var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = module_1.extract_text(var_15)
    assert var_16 == 'Hello'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'b'
    var_7 = 'bold'
    var_8 = []
    var_9 = lambda : var_8
    var_10 = ' normal'
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'p'
    var_18 = 'Start '
    var_19 = [var_15]
    var_20 = lambda : var_19
    var_21 = None
    var_22 = {var_2: var_17, var_3: var_18, var_4: var_20, var_5: var_21}
    var_23 = [var_0, var_16, var_22]
    var_24 = {}
    var_25 = module_0.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = module_1.extract_text(var_26)
    assert var_27 == 'Start bold normal'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'br'
    var_7 = None
    var_8 = []
    var_9 = lambda : var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = ()
    var_16 = 'p'
    var_17 = 'Line1'
    var_18 = [var_14]
    var_19 = lambda : var_18
    var_20 = {var_2: var_16, var_3: var_17, var_4: var_19, var_5: var_7}
    var_21 = [var_0, var_15, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()
    var_25 = '\n'
    var_26 = module_1.extract_text(var_24, sep_symbol=var_25)
    assert var_26 == 'Line1\n'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'div'
    var_7 = 'Block'
    var_8 = []
    var_9 = lambda : var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'p'
    var_18 = [var_15]
    var_19 = lambda : var_18
    var_20 = {var_2: var_17, var_3: var_10, var_4: var_19, var_5: var_10}
    var_21 = [var_0, var_16, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()
    var_25 = '\n'
    var_26 = module_1.extract_text(var_24, var_25)
    assert var_26 == 'Block'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = '  spaced  '
    var_8 = []
    var_9 = lambda : var_8
    var_10 = '  text  '
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'p'
    var_18 = '  hello  '
    var_19 = [var_15]
    var_20 = lambda : var_19
    var_21 = None
    var_22 = {var_2: var_17, var_3: var_18, var_4: var_20, var_5: var_21}
    var_23 = [var_0, var_16, var_22]
    var_24 = {}
    var_25 = module_0.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = module_1.extract_text(var_26)
    assert var_27 == 'hello spaced text'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = '  spaced  '
    var_8 = []
    var_9 = lambda : var_8
    var_10 = '  text  '
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'p'
    var_18 = '  hello  '
    var_19 = [var_15]
    var_20 = lambda : var_19
    var_21 = None
    var_22 = {var_2: var_17, var_3: var_18, var_4: var_20, var_5: var_21}
    var_23 = [var_0, var_16, var_22]
    var_24 = {}
    var_25 = module_0.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = False
    var_28 = module_1.extract_text(var_26, squash_space=var_27)
    assert var_28 == '  hello    spaced    text  '

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
    var_9 = lambda : var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.extract_text(var_14)
    assert var_15 == ''

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
    var_7 = 'Alone'
    var_8 = []
    var_9 = lambda : var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = module_1.extract_text(var_15)
    assert var_16 == 'Alone'

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'b'
    var_7 = 'first'
    var_8 = []
    var_9 = lambda : var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'i'
    var_18 = 'second'
    var_19 = []
    var_20 = lambda : var_19
    var_21 = {var_2: var_17, var_3: var_18, var_4: var_20, var_5: var_10}
    var_22 = [var_0, var_16, var_21]
    var_23 = {}
    var_24 = module_0.type(*var_22, **var_23)
    var_25 = var_24()
    var_26 = ()
    var_27 = 'p'
    var_28 = [var_15, var_25]
    var_29 = lambda : var_28
    var_30 = {var_2: var_27, var_3: var_10, var_4: var_29, var_5: var_10}
    var_31 = [var_0, var_26, var_30]
    var_32 = {}
    var_33 = module_0.type(*var_31, **var_32)
    var_34 = var_33()
    var_35 = module_1.extract_text(var_34)
    assert var_35 == 'firstsecond'



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_extract_text_with_squash_space_true. Retrieved 3/4 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_squash_space_false_predicate. Retrieved 6/14 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0._squash_artifical_nl(var_0)
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = ''
    var_4 = None
    var_5 = True



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
    var_5 = 'tail'
    var_6 = 'p'
    var_7 = 'some text'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = module_1.extract_text_array(var_15)
    var_17 = 'some text'
    var_18 = bool('some text' in var_16)
    assert var_18 is True



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/17 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/17 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/10 statements.


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
    var_1 = 'text'

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
    var_0 = 'p'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'br'
    var_1 = 'hr'
    var_2 = [var_0, var_1]
    var_3 = 'p'
    var_4 = 'div'
    var_5 = 'span'
    var_6 = [var_3, var_4, var_5]



# Parsed testcases at query #108
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
    var_17 = True
    var_18 = module_1.extract_text_array(var_15, var_16, var_17)
    var_19 = bool(var_18 == ['hello'])
    assert var_19 is True



# Parsed testcases at query #109
#--------------------------




import builtins as module_0
import pyquery.text as module_1

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
    var_14 = {var_5}
    var_15 = set()
    var_16 = False
    var_17 = module_1.extract_text_array(var_13, var_16, var_16)



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_dom_with_text_only. Retrieved 2/7 statements.
# Partially parsed test_dom_with_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_dom_with_inline_tag_and_text. Retrieved 2/7 statements.
# Partially parsed test_dom_with_artificial_newlines_stripped. Retrieved 6/14 statements.
# Partially parsed test_dom_with_artificial_newlines_squashed. Retrieved 3/8 statements.
# Partially parsed test_dom_with_children_and_tail. Retrieved 5/13 statements.
# Partially parsed test_dom_with_nested_separators. Retrieved 5/13 statements.
# Partially parsed test_dom_skip_empty_strings. Retrieved 2/7 statements.
# Partially parsed test_dom_with_multiple_artificial_newlines_squashed. Retrieved 9/20 statements.
# Partially parsed test_dom_with_no_artificial_newlines_stripped. Retrieved 3/8 statements.


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
    var_1 = 'inline'

def test_case_0():
    var_0 = 'p'
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'before'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = 'after'
    var_3 = 'div'
    var_4 = 'start'

def test_case_0():
    var_0 = 'p'
    var_1 = ''

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'text'
    var_2 = True



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_strip_artifical_nl_false. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = False



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_single_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag_with_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_block_tag_artifical_newlines. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_with_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_block_tag_with_multiple_children. Retrieved 8/19 statements.
# Partially parsed test_extract_text_array_nested_block_tags. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_multiple_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_leading_trailing_artifical_nl. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'

def test_case_0():
    var_0 = 'p'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'before'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = lambda : None
    var_1 = None

def test_case_0():
    var_0 = 'hr'
    var_1 = 'separator'

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = None
    var_3 = 'b'
    var_4 = None

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
    var_1 = 'inner text'
    var_2 = None
    var_3 = 'div'
    var_4 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'content'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
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
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Content'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = ' World'
    var_3 = 'div'
    var_4 = None



# Parsed testcases at query #114
#--------------------------

# Partially parsed test_squash_artifical_nl_evaluates_true. Retrieved 14/18 statements.


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
    var_10 = []
    var_11 = []
    var_12 = True
    var_13 = False
    var_14 = bool(True)
    assert var_14 is True



# Parsed testcases at query #115
#--------------------------

# Partially parsed test_extract_text_array_with_none_dom_tag_callable. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_non_inline_non_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child_and_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_both_flags_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_complex_scenario. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_tag_in_separators. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_with_tag_in_inline_tags. Retrieved 6/11 statements.


def test_case_0():
    var_0 = lambda : None

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'start'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = ' after'
    var_3 = 'div'
    var_4 = 'before '

def test_case_0():
    var_0 = 'hr'
    var_1 = 'br'
    var_2 = {var_0, var_1}
    var_3 = 'hr'
    var_4 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'i'
    var_2 = 'span'
    var_3 = {var_0, var_1, var_2}
    var_4 = 'b'
    var_5 = 'bold'



# Parsed testcases at query #116
#--------------------------

# Partially parsed test_squash_artifical_nl_false. Retrieved 11/15 statements.


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
    var_10 = False



# Parsed testcases at query #117
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
    var_16 = module_1.extract_text_array(var_14, var_15, var_15)
    var_17 = bool(True)
    assert var_17 is True



# Parsed testcases at query #118
#--------------------------

# Partially parsed test_predicate_at_line17_evaluates_to_false. Retrieved 10/14 statements.


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



# Parsed testcases at query #119
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_artificial_newlines. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_separator_between_texts. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_nested_structure. Retrieved 7/18 statements.
# Partially parsed test_extract_text_array_multiple_none_values. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_with_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_multiple_none. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_with_trailing_none. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_only_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_empty_children. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_no_text_in_element. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 8/19 statements.
# Partially parsed test_extract_text_array_with_mixed_tags. Retrieved 8/19 statements.
# Partially parsed test_extract_text_array_strip_with_leading_none. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_keeps_one_none. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = 'parent'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = 'parent'

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'parent'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = 'parent'
    var_5 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'first'

def test_case_0():
    var_0 = 'span'
    var_1 = 'grandchild'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = 'div'
    var_6 = 'root'

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'br'
    var_1 = 'text'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'b'
    var_4 = 'bold'

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'

def test_case_0():
    var_0 = 'span'
    var_1 = None
    var_2 = 'tail'
    var_3 = 'div'
    var_4 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'first'
    var_2 = None
    var_3 = 'span'
    var_4 = 'second'
    var_5 = None
    var_6 = 'div'
    var_7 = 'parent'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'span'
    var_4 = 'text'
    var_5 = None
    var_6 = 'div'
    var_7 = 'start'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True
    var_6 = False



# Parsed testcases at query #120
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_child. Retrieved 2/7 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/9 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 2/8 statements.
# Partially parsed test_extract_text_squash_whitespace. Retrieved 1/5 statements.
# Partially parsed test_extract_text_block_symbol. Retrieved 3/10 statements.
# Partially parsed test_extract_text_sep_symbol. Retrieved 4/10 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = 'p'
    var_3 = '---'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/9 statements.
# Partially parsed test_extract_text_with_inline_element. Retrieved 2/8 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 2/9 statements.
# Partially parsed test_extract_text_multiple_blocks. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_whitespace. Retrieved 1/5 statements.
# Partially parsed test_extract_text_empty_element. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_blocks. Retrieved 2/8 statements.
# Partially parsed test_extract_text_with_comment. Retrieved 2/8 statements.


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
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'p'
    var_1 = 'comment'



# Parsed testcases at query #2
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
    var_4 = [var_0, var_1, var_1, var_2, var_1, var_3]
    var_5 = module_0._squash_artifical_nl(var_4)
    var_6 = bool(var_5 == ['a', None, 'b', None, 'c'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [None, 'a', 'b'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b', None])
    assert var_5 is True

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
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    var_3 = bool(var_2 == [None])
    assert var_3 is True



# Parsed testcases at query #3
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
    var_1 = 2
    var_2 = 'hello'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [1, 2, 'hello'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['hello', 1, 2])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [1, 'hello', 2])
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
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a', 'b', 'c'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = [var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)
    var_4 = bool(var_3 == [1, 'hello'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 1
    var_2 = [var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)
    var_4 = bool(var_3 == ['hello', 1])
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello   world  '
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == ['hello world'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [1, 'a b'])
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
    var_0 = '  foo  '
    var_1 = '  bar  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['foo bar'])
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
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [None, 'a b'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = 'y'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._merge_original_parts(var_4)
    var_6 = bool(var_5 == [1, 2, 'x y'])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == [42])
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = None
    var_2 = 'world'
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = module_0.extract_text(var_3, squash_space=var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



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

# Partially parsed test_extract_text_plain_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/9 statements.
# Partially parsed test_extract_text_with_squash_space. Retrieved 2/6 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 2/6 statements.
# Partially parsed test_extract_text_with_nested_inline. Retrieved 2/9 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail_only. Retrieved 2/8 statements.
# Partially parsed test_extract_text_custom_block_symbol. Retrieved 3/10 statements.
# Partially parsed test_extract_text_custom_sep_symbol. Retrieved 4/10 statements.


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
    var_1 = True

def test_case_0():
    var_0 = 'p'
    var_1 = False

def test_case_0():
    var_0 = 'p'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = ' | '

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = 'p'
    var_3 = ' --- '



# Parsed testcases at query #9
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #10
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #11
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_inline. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_collapse. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator_and_block. Retrieved 1/4 statements.
# Partially parsed test_extract_text_multiple_blocks_with_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_only_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_blocks. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div><hr/>Text after hr</div>'

def test_case_0():
    var_0 = '<div><p>First paragraph</p><p>Second paragraph</p></div>'

def test_case_0():
    var_0 = '<div>Hello <b>bold</b> world</div>'

def test_case_0():
    var_0 = '<div>  Too   many   spaces  </div>'

def test_case_0():
    var_0 = '<div><hr/><p>After hr</p></div>'

def test_case_0():
    var_0 = '<div>Start<p>Middle</p>End</div>'

def test_case_0():
    var_0 = '<hr/>'

def test_case_0():
    var_0 = '<div><div><p>Deep</p></div></div>'



# Parsed testcases at query #13
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = '\n'
    var_3 = False
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_extract_text_array_returns_empty_string_for_callable_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag_returns_true. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_inline_tag_returns_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_non_inline_non_separator_tag_adds_none. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_children_and_tail. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_with_squash_artifical_nl_false. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_with_strip_artifical_nl_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_combined_options. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_only_text_in_non_inline_tag. Retrieved 1/5 statements.


def test_case_0():
    var_0 = lambda : None
    var_1 = []

def test_case_0():
    var_0 = 'br'

def test_case_0():
    var_0 = 'span'

def test_case_0():
    var_0 = 'div'

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
    var_1 = 'p'
    var_2 = 'br'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #15
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.extract_text(var_0)
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #17
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #18
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_extract_text_array_returns_empty_string_for_callable_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_returns_string_list_with_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_returns_none_for_block_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_returns_true_for_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_separator_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 5/13 statements.


def test_case_0():
    var_0 = lambda : None
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'div'
    var_1 = None

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
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'a'
    var_1 = 'Click'
    var_2 = ' here'
    var_3 = 'p'
    var_4 = 'Please '

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'span'
    var_4 = 'normal'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_extract_text_returns_empty_string_for_callable_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_returns_text_for_simple_text_node. Retrieved 2/7 statements.
# Partially parsed test_extract_text_uses_block_symbol_for_non_inline_tags. Retrieved 2/7 statements.
# Partially parsed test_extract_text_uses_sep_symbol_for_separator_tags. Retrieved 2/7 statements.
# Partially parsed test_extract_text_joins_child_text. Retrieved 5/13 statements.
# Partially parsed test_extract_text_strips_whitespace_when_squash_space_true. Retrieved 2/7 statements.
# Partially parsed test_extract_text_preserves_whitespace_when_squash_space_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_handles_nested_separators. Retrieved 5/13 statements.
# Partially parsed test_extract_text_handles_multiple_children. Retrieved 8/19 statements.


def test_case_0():
    var_0 = lambda : None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Line1'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Child'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Parent'

def test_case_0():
    var_0 = 'p'
    var_1 = '  Hello World  '

def test_case_0():
    var_0 = 'p'
    var_1 = '  Hello World  '
    var_2 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'A'

def test_case_0():
    var_0 = 'span'
    var_1 = 'First'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'Second'
    var_5 = None
    var_6 = 'div'
    var_7 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'p'
    var_4 = 'Some '

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Line1'
    var_2 = '\n'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = {}
    var_2 = False



# Parsed testcases at query #23
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #24
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 != '')
    assert var_3 is True



# Parsed testcases at query #25
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #26
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'text'
    var_2 = True
    var_3 = 'more text'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda dom, squash_artifical_nl=True: var_4
    var_6 = lambda x: x
    var_7 = lambda x: x
    var_8 = lambda x: x
    var_9 = 'dom'
    var_10 = False
    var_11 = module_0.extract_text(var_9, squash_space=var_10)



# Parsed testcases at query #27
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = '  hello  '
    var_2 = (var_0, var_1)
    var_3 = 'br'
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = '  world  '
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]
    var_9 = True
    var_10 = module_0.extract_text(var_8, squash_space=var_9)
    assert var_10 == 'hello\nworld'



# Parsed testcases at query #28
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 8/19 statements.
# Partially parsed test_extract_text_array_nested_separator. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_multiple_none. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_leading_none. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_trailing_none. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_nested_structure. Retrieved 8/19 statements.


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
    var_1 = 'inline'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'p'
    var_4 = 'before'

def test_case_0():
    var_0 = 'a'
    var_1 = 'link'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'click '

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
    var_1 = 'text'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'one'
    var_2 = None
    var_3 = 'span'
    var_4 = 'two'
    var_5 = ' tail'
    var_6 = 'div'
    var_7 = 'start '

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'p'
    var_4 = 'line1'

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
    var_0 = 'div'
    var_1 = 'text'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = lambda : None
    var_1 = None

def test_case_0():
    var_0 = 'i'
    var_1 = 'italic'
    var_2 = ' after '
    var_3 = 'b'
    var_4 = 'bold'
    var_5 = None
    var_6 = 'p'
    var_7 = 'text '



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_10_evaluates_to_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '<div>text</div>'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_mixed_tags. Retrieved 5/13 statements.


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
    var_1 = 'text'

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
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'inline'
    var_2 = None
    var_3 = 'div'
    var_4 = 'start'



# Parsed testcases at query #32
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
    var_14 = 'br'
    var_15 = {var_14}
    var_16 = 'span'
    var_17 = 'b'
    var_18 = {var_16, var_17}
    var_19 = module_1.extract_text_array(var_13)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_squash_artifical_nl_false. Retrieved 13/23 statements.


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
    var_11 = []
    var_12 = False



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 10/14 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 11/15 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 10/14 statements.
# Partially parsed test_extract_text_array_with_artificial_nl. Retrieved 11/15 statements.
# Partially parsed test_extract_text_array_squash_and_strip. Retrieved 13/23 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl_false. Retrieved 13/23 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_false. Retrieved 13/23 statements.
# Partially parsed test_extract_text_array_nested_inline_tags. Retrieved 13/23 statements.
# Partially parsed test_extract_text_array_multiple_separators. Retrieved 14/27 statements.


def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda : var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = 'Hello'
    var_7 = []
    var_8 = lambda : var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'br'
    var_6 = None
    var_7 = []
    var_8 = lambda : var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = 'A'
    var_7 = []
    var_8 = lambda : var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'b'
    var_6 = None
    var_7 = []
    var_8 = lambda : var_7
    var_9 = ' world'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'p'
    var_12 = 'Hello'

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = 'A'
    var_7 = []
    var_8 = lambda : var_7
    var_9 = 'B'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = None
    var_12 = False

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'div'
    var_6 = 'A'
    var_7 = []
    var_8 = lambda : var_7
    var_9 = 'B'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = None
    var_12 = False

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'span'
    var_6 = 'inner'
    var_7 = []
    var_8 = lambda : var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'b'
    var_12 = 'start'

def test_case_0():
    var_0 = 'FakeElement'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'br'
    var_6 = None
    var_7 = []
    var_8 = lambda : var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}
    var_10 = []
    var_11 = lambda : var_10
    var_12 = {var_1: var_5, var_2: var_6, var_3: var_11, var_4: var_6}
    var_13 = 'div'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_single_text_node. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_block_tag_adds_artificial_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_child_and_tail. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_squash_and_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_separator_with_child. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = 'tail'
    var_3 = 'div'
    var_4 = 'before'
    var_5 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'content'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'content'
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'br'
    var_4 = None
    var_5 = False



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_strip_artifical_nl_true. Retrieved 13/17 statements.


def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = 'hello'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = False
    var_12 = True
    var_13 = bool(True)
    assert var_13 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line12_evaluates_to_false. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'some text'



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
    var_16 = module_1.extract_text_array(var_14, var_15)
    var_17 = bool(var_16 == [None])
    assert var_17 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_line12_true. Retrieved 1/13 statements.


def test_case_0():
    var_0 = []
    var_1 = bool(True)
    assert var_1 is True



# Parsed testcases at query #40
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
    var_6 = 'br'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.extract_text_array(var_14)
    var_16 = var_15[0]
    assert var_16 is True



# Parsed testcases at query #41
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'text'
    var_2 = True
    var_3 = [var_0, var_1, var_0, var_2, var_0]
    var_4 = False
    var_5 = module_0._strip_artifical_nl(var_3)
    var_6 = bool(var_5 == var_3)
    assert var_6 is True



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
    var_6 = 'div'
    var_7 = 'some text'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = module_1.extract_text_array(var_15)
    var_17 = var_15.text
    var_18 = bool(var_15.text is not None)
    assert var_18 is True



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
    var_16 = module_1.extract_text_array(var_14, var_15, var_15)
    var_17 = bool(var_16 == [])
    assert var_17 is True

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
    var_7 = 'Hello'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = True
    var_17 = module_1.extract_text_array(var_15, var_16, var_16)
    var_18 = bool(var_17 == ['Hello'])
    assert var_18 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'br'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = True
    var_16 = module_1.extract_text_array(var_14, var_15, var_15)
    var_17 = bool(var_16 == [True])
    assert var_17 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = 'inline'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = True
    var_17 = module_1.extract_text_array(var_15, var_16, var_16)
    var_18 = bool(var_17 == ['inline'])
    assert var_18 is True

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
    var_7 = 'child_text'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = ' tail'
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'parent_text'
    var_18 = [var_15]
    var_19 = lambda self: var_18
    var_20 = None
    var_21 = {var_2: var_6, var_3: var_17, var_4: var_19, var_5: var_20}
    var_22 = [var_0, var_16, var_21]
    var_23 = {}
    var_24 = module_0.type(*var_22, **var_23)
    var_25 = var_24()
    var_26 = True
    var_27 = module_1.extract_text_array(var_25, var_26, var_26)
    var_28 = bool(var_27 == ['parent_text', 'child_text', ' tail'])
    assert var_28 is True

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
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_2: var_6, var_3: var_3, var_4: var_8, var_5: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = ()
    var_16 = [var_14]
    var_17 = lambda self: var_16
    var_18 = {var_2: var_6, var_3: var_9, var_4: var_17, var_5: var_9}
    var_19 = [var_0, var_15, var_18]
    var_20 = {}
    var_21 = module_0.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = True
    var_24 = False
    var_25 = module_1.extract_text_array(var_22, var_23, var_24)
    var_26 = bool(var_25 == [None, 'text', None])
    assert var_26 is True

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
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_2: var_6, var_3: var_3, var_4: var_8, var_5: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = ()
    var_16 = [var_14]
    var_17 = lambda self: var_16
    var_18 = {var_2: var_6, var_3: var_9, var_4: var_17, var_5: var_9}
    var_19 = [var_0, var_15, var_18]
    var_20 = {}
    var_21 = module_0.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = False
    var_24 = True
    var_25 = module_1.extract_text_array(var_22, var_23, var_24)
    var_26 = bool(var_25 == ['text'])
    assert var_26 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'br'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = ' after_br'
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'div'
    var_18 = 'before_br'
    var_19 = [var_15]
    var_20 = lambda self: var_19
    var_21 = {var_2: var_17, var_3: var_18, var_4: var_20, var_5: var_7}
    var_22 = [var_0, var_16, var_21]
    var_23 = {}
    var_24 = module_0.type(*var_22, **var_23)
    var_25 = var_24()
    var_26 = True
    var_27 = module_1.extract_text_array(var_25, var_26, var_26)
    var_28 = bool(var_27 == ['before_br', True, ' after_br'])
    assert var_28 is True

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
    var_7 = 'child'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'parent'
    var_18 = [var_15]
    var_19 = lambda self: var_18
    var_20 = {var_2: var_6, var_3: var_17, var_4: var_19, var_5: var_10}
    var_21 = [var_0, var_16, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()
    var_25 = False
    var_26 = module_1.extract_text_array(var_24, var_25, var_25)
    var_27 = bool(var_26 == [None, 'parent', None, 'child', None, None])
    assert var_27 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_false. Retrieved 10/14 statements.


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

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_block_tag_no_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_nested_with_text. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_and_strip. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_no_squash. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_no_strip. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'
    var_2 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = 'hello '
    var_5 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'start'
    var_5 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'first'
    var_2 = None
    var_3 = 'span'
    var_4 = 'second'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = True



# Parsed testcases at query #46
#--------------------------

# Failed to parse test_predicate_line12_evaluates_to_false.




# Parsed testcases at query #47
#--------------------------

# Partially parsed test_extract_text_with_text_only. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_custom_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_only_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_script_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_style_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_list_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_deeply_nested_elements. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<br>Text after break'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<p>Hello <b>World</b> again</p>'

def test_case_0():
    var_0 = '<p>Line1<br><br>Line2</p>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = ' | '

def test_case_0():
    var_0 = '<p>A<br>B</p>'
    var_1 = ' -- '

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'
    var_1 = False

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>   </p>'

def test_case_0():
    var_0 = '<script>var x = 1;</script>'

def test_case_0():
    var_0 = '<style>body { color: red; }</style>'

def test_case_0():
    var_0 = '<ul><li>Item1</li><li>Item2</li></ul>'

def test_case_0():
    var_0 = '<div><span><b>Deep</b></span></div>'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_strip_artifical_nl_false. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'Dom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'div'
    var_5 = None
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_1: var_4, var_2: var_5, var_3: var_7}
    var_9 = False



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 13/17 statements.


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



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_6, var_4: var_8}
    var_10 = True
    var_11 = False



# Parsed testcases at query #51
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
    var_15 = module_1.extract_text_array(var_14)
    var_16 = var_15[-1]
    var_17 = bool(var_15[-1] is not None)
    assert var_17 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 2/11 statements.


def test_case_0():
    var_0 = set()
    var_1 = set()



# Parsed testcases at query #53
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
    var_6 = []
    var_7 = lambda : var_6
    var_8 = {var_2: var_5, var_3: var_3, var_4: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = module_1.extract_text_array(var_12)
    var_14 = 'text'
    var_15 = bool('text' in var_13)
    assert var_15 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_non_inline_tag_without_children. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_child_and_tail. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_array_nested_structure. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'div'
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
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'parent'
    var_5 = True

def test_case_0():
    var_0 = 'span'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'text'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = False
    var_6 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'text'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'first'
    var_2 = ' '
    var_3 = 'b'
    var_4 = 'second'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = True

def test_case_0():
    var_0 = 'i'
    var_1 = 'inner'
    var_2 = None
    var_3 = 'span'
    var_4 = None
    var_5 = ' after span'
    var_6 = 'div'
    var_7 = 'before '
    var_8 = True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_extract_text_array_with_none_dom_tag_callable. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_block_tag_and_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_nested_separators. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_no_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_no_strip_artifical_nl. Retrieved 3/8 statements.


def test_case_0():
    var_0 = lambda : None
    var_1 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = 'hello '

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'line1'

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = False



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_strip_artifical_nl_true. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'
    var_2 = False
    var_3 = True



# Parsed testcases at query #57
#--------------------------

# Failed to parse test_predicate_at_line_17_evaluates_to_true.




# Parsed testcases at query #58
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_single_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag_no_artifical. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_block_tag_with_artifical. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_nested_separator. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = 'before'
    var_5 = False

def test_case_0():
    var_0 = 'span'
    var_1 = None
    var_2 = ' after'
    var_3 = 'div'
    var_4 = 'before'
    var_5 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'hello'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = lambda : None
    var_1 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'line1'
    var_5 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'a'
    var_2 = ' '
    var_3 = 'b'
    var_4 = 'b'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = False



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_strip_artifical_nl_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_text_only. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_inline_tag_with_text. Retrieved 5/17 statements.
# Partially parsed test_extract_text_array_with_artificial_nl. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 8/25 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_with_tail_after_separator. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'span'
    var_4 = 'before '

def test_case_0():
    var_0 = 'div'
    var_1 = 'line1'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child1'
    var_2 = ' tail1 '
    var_3 = 'br'
    var_4 = None
    var_5 = ' tail2'
    var_6 = 'div'
    var_7 = 'start '

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = ' after'
    var_3 = 'div'
    var_4 = 'before '



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_extract_text_array_with_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_nested_tags. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_with_artifical_newlines. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_separator_with_text. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = 'hello '
    var_5 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = None
    var_3 = 'div'
    var_4 = None
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
    var_1 = 'content'
    var_2 = None
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = ' after'
    var_3 = 'div'
    var_4 = 'before '
    var_5 = None

def test_case_0():
    var_0 = lambda : None
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'hr'
    var_1 = 'separator'
    var_2 = None



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_predicate_line17_evaluates_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = set()
    var_1 = 'span'
    var_2 = 'b'
    var_3 = {var_1, var_2}



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_simple_text. Retrieved 2/7 statements.
# Partially parsed test_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_nested_children. Retrieved 6/14 statements.
# Partially parsed test_artificial_newlines. Retrieved 3/8 statements.
# Partially parsed test_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_strip_artifical_nl. Retrieved 7/15 statements.
# Partially parsed test_both_squash_and_strip. Retrieved 6/14 statements.
# Partially parsed test_callable_tag. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'inline'
    var_2 = False

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'start '
    var_5 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = False
    var_6 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'content'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = lambda : None
    var_1 = None



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_strip_artifical_nl_flag_true. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_6, var_4: var_8}
    var_10 = False
    var_11 = True
    var_12 = bool(True)
    assert var_12 is True



# Parsed testcases at query #65
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



# Parsed testcases at query #66
#--------------------------

# Failed to parse test_extract_text_array_empty_dom.
# Failed to parse test_extract_text_array_simple_text.
# Failed to parse test_extract_text_array_with_separator.
# Failed to parse test_extract_text_array_with_children.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 1/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 1/10 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 1/10 statements.
# Failed to parse test_extract_text_array_complex.
# Failed to parse test_extract_text_array_callable_tag.
# Failed to parse test_extract_text_array_inline_tag.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_single_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_nested_elements. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_with_artifical_nl_squash. Retrieved 4/11 statements.
# Partially parsed test_extract_text_array_with_artifical_nl_strip. Retrieved 4/11 statements.
# Partially parsed test_extract_text_array_both_flags. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_separator_multiple. Retrieved 4/11 statements.
# Partially parsed test_extract_text_array_mixed_content. Retrieved 4/12 statements.
# Partially parsed test_extract_text_array_strip_none_only. Retrieved 2/5 statements.
# Partially parsed test_extract_text_array_squash_consecutive_none. Retrieved 4/11 statements.


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
    var_2 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'
    var_2 = 'p'
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'
    var_2 = 'span'
    var_3 = True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Mock'
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

def test_case_0():
    var_0 = 'div'
    var_1 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True
    var_3 = False



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_extract_text_array_for_loop_child_not_none. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_extract_text_array_returns_empty_string_for_callable_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_separator_tag_returns_list_with_true. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_inline_tag_and_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_inline_tag_and_no_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_block_tag_adds_none_around_content. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_squash_artifical_nl_squashes_consecutive_nones. Retrieved 4/10 statements.
# Partially parsed test_extract_text_array_with_strip_artifical_nl_strips_leading_and_trailing_nones. Retrieved 4/10 statements.
# Partially parsed test_extract_text_array_handles_child_tail. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_with_separator_child. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_with_nested_blocks. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_preserves_artifical_nl_when_squash_false_and_strip_false. Retrieved 3/9 statements.
# Partially parsed test_extract_text_array_returns_empty_list_for_empty_block. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_only_text_in_block. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_squash_and_strip_on_complex_structure. Retrieved 2/10 statements.


def test_case_0():
    var_0 = lambda : None

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'span'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'start'

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'only'

def test_case_0():
    var_0 = 'div'
    var_1 = None



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_single_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 1/8 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_multiple_separators. Retrieved 2/7 statements.


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
    var_0 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'start'

def test_case_0():
    var_0 = 'br'
    var_1 = None



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_predicate_line17_evaluates_to_true. Retrieved 16/20 statements.


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
    var_10 = 'a'
    var_11 = 'span'
    var_12 = [var_10, var_11]
    var_13 = 'br'
    var_14 = [var_13]
    var_15 = False



# Parsed testcases at query #72
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



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_strip_artifical_nl_false_does_not_strip. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 'MockDom'
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



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_nested_inline. Retrieved 2/9 statements.
# Partially parsed test_extract_text_whitespace_squashing. Retrieved 1/5 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_only_block_elements. Retrieved 2/6 statements.
# Partially parsed test_extract_text_multiple_separators. Retrieved 2/8 statements.
# Partially parsed test_extract_text_custom_block_symbol. Retrieved 3/8 statements.
# Partially parsed test_extract_text_custom_sep_symbol. Retrieved 3/8 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/6 statements.


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
    var_1 = 'b'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'

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
    var_1 = False



# Parsed testcases at query #75
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'FakeDOM'
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
    var_18 = bool(var_17 == [])
    assert var_18 is True



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 2/3 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #77
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #78
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = 'some text'
    var_2 = [var_0, var_1, var_0]
    var_3 = False
    assert var_3 is False



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_squash_artifical_nl_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_block_tag. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_artificial_newlines. Retrieved 3/10 statements.
# Partially parsed test_extract_text_strips_leading_trailing_spaces. Retrieved 1/5 statements.
# Partially parsed test_extract_text_squashes_multiple_spaces. Retrieved 1/5 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_only_whitespace. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'

def test_case_0():
    var_0 = 'span'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'span'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_false. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'div'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_6, var_4: var_8}
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #82
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
    var_6 = 'custom_tag'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.extract_text_array(var_14)
    var_16 = var_15[-1]
    assert var_16 is None



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_squash_artifical_nl_false. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 'MockDom'
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
    var_11 = False



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_for_loop_predicate_false. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_artifical_nl_squash. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_artifical_nl_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_none_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 8/19 statements.
# Partially parsed test_extract_text_array_squash_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_false. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello '

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'p'
    var_1 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' '
    var_3 = 'i'
    var_4 = 'italic'
    var_5 = None
    var_6 = 'div'
    var_7 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'content'
    var_2 = False



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_nested_inline. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 2/9 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'br'

def test_case_0():
    var_0 = 'span'

def test_case_0():
    var_0 = 'p'
    var_1 = 'b'

def test_case_0():
    var_0 = 'p'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True



# Parsed testcases at query #87
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
    var_9 = lambda self: var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = False
    var_16 = module_1.extract_text_array(var_14, var_15)
    var_17 = bool(var_16 == [None])
    assert var_17 is True



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_nested_separator. Retrieved 5/13 statements.


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
    var_1 = 'text'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'p'
    var_4 = 'before'

def test_case_0():
    var_0 = 'a'
    var_1 = 'link'
    var_2 = ' after'
    var_3 = 'p'
    var_4 = 'start '

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'middle'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = ' next'
    var_3 = 'p'
    var_4 = 'first'



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_child. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 2/6 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_squash_space. Retrieved 1/5 statements.
# Partially parsed test_extract_text_nested_structure. Retrieved 2/10 statements.
# Partially parsed test_extract_text_multiple_separators. Retrieved 3/8 statements.
# Partially parsed test_extract_text_trailing_block. Retrieved 2/8 statements.
# Partially parsed test_extract_text_leading_block. Retrieved 2/8 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 1/5 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = 'br'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = '|'



# Parsed testcases at query #90
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
    var_15 = True
    var_16 = module_1.extract_text_array(var_14, var_15, var_15)
    var_17 = var_16[-1]
    assert var_17 is None



# Parsed testcases at query #91
#--------------------------

# Failed to parse test_predicate_at_line_17_evaluates_to_false.




# Parsed testcases at query #92
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
    var_7 = []
    var_8 = lambda : var_7
    var_9 = None
    var_10 = {var_2: var_6, var_3: var_3, var_4: var_8, var_5: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = False
    var_16 = module_1.extract_text_array(var_14, var_15, var_15)
    var_17 = bool(var_16 == ['text', None])
    assert var_17 is True



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_nested_elements. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.


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
    var_1 = 'child'
    var_2 = None
    var_3 = 'div'
    var_4 = 'parent'

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_nested_tags. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello '
    var_5 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'B'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' normal'
    var_3 = 'p'
    var_4 = 'Some '
    var_5 = False

def test_case_0():
    var_0 = lambda : None
    var_1 = None



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_text_no_children. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 6/14 statements.


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
    var_1 = 'world'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'p'
    var_4 = 'before '

def test_case_0():
    var_0 = 'a'
    var_1 = 'link'
    var_2 = ' after'
    var_3 = 'p'
    var_4 = 'click '

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'b'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_single_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_both_options. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_nested_structure. Retrieved 9/20 statements.


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
    var_1 = 'inline'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'start '

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = 'p'
    var_4 = None
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'content'
    var_2 = None
    var_3 = 'p'
    var_4 = None
    var_5 = False
    var_6 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'content'
    var_2 = None
    var_3 = 'p'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = lambda : None
    var_1 = 'ignored'

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = ' after'
    var_3 = 'div'
    var_4 = 'child '
    var_5 = None
    var_6 = 'body'
    var_7 = 'start '
    var_8 = True



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_extract_text_array_empty_dom_no_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_nested_separator. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl_true. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_true. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.


def test_case_0():
    var_0 = None
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'

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
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'before'

def test_case_0():
    var_0 = 'span'
    var_1 = 'a'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'a'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = lambda : None
    var_1 = None



# Parsed testcases at query #98
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_extract_text_array_with_callable_tag_returns_empty_string. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_single_text_node. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag_adds_true. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag_with_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_non_inline_tag_adds_none. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child_and_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl_false. Retrieved 6/11 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag_no_artifical_nl. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 8/19 statements.


def test_case_0():
    var_0 = lambda : None
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'hr'
    var_1 = 'separator'

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'start'

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = False
    var_3 = None
    var_4 = 'a'
    var_5 = [var_3, var_4, var_3]

def test_case_0():
    var_0 = 'div'
    var_1 = 'b'
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'inline'

def test_case_0():
    var_0 = 'b'
    var_1 = 'first'
    var_2 = None
    var_3 = 'i'
    var_4 = 'second'
    var_5 = None
    var_6 = 'div'
    var_7 = None



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_artifical_nl. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_squash_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_nl. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'p'
    var_4 = 'before '

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'inline'

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'a'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'b'
    var_2 = True



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_predicate_line_17_evaluates_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_nested_separator. Retrieved 5/13 statements.


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
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello '

def test_case_0():
    var_0 = 'a'
    var_1 = 'click'
    var_2 = ' here'
    var_3 = 'p'
    var_4 = 'Please '

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'B'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'C'
    var_2 = False

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'line1'



# Parsed testcases at query #103
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = 'text'
    var_2 = [var_0, var_1, var_0]
    var_3 = False
    var_4 = bool(not var_3 is True)
    assert var_4 is True



# Parsed testcases at query #104
#--------------------------




import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Elem'
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

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Elem'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
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
    var_17 = module_1.extract_text_array(var_15, var_16, var_16)
    var_18 = bool(var_17 == ['hello'])
    assert var_18 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Elem'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'br'
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

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Elem'
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
    var_16 = module_1.extract_text_array(var_14, var_15, var_15)
    var_17 = bool(var_16 == [None, None])
    assert var_17 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Elem'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = 'world'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = ' tail'
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'div'
    var_18 = 'hello '
    var_19 = [var_15]
    var_20 = lambda self: var_19
    var_21 = None
    var_22 = {var_2: var_17, var_3: var_18, var_4: var_20, var_5: var_21}
    var_23 = [var_0, var_16, var_22]
    var_24 = {}
    var_25 = module_0.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = False
    var_28 = module_1.extract_text_array(var_26, var_27, var_27)
    var_29 = bool(var_28 == [None, 'hello ', 'world', ' tail', None])
    assert var_29 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Elem'
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
    var_15 = True
    var_16 = False
    var_17 = module_1.extract_text_array(var_14, var_15, var_16)
    var_18 = bool(var_17 == [None])
    assert var_18 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Elem'
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
    var_18 = bool(var_17 == [])
    assert var_18 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Elem'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = None
    var_7 = lambda : var_6
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {var_2: var_7, var_3: var_6, var_4: var_9, var_5: var_6}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.extract_text_array(var_14)
    assert var_15 == ''

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Elem'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'strong'
    var_7 = 'bold'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = False
    var_17 = module_1.extract_text_array(var_15, var_16, var_16)
    var_18 = bool(var_17 == ['bold'])
    assert var_18 is True

import builtins as module_0
import pyquery.text as module_1

def test_case_0():
    var_0 = 'Elem'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = 'inner'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = ' after'
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = ()
    var_17 = 'div'
    var_18 = 'before '
    var_19 = [var_15]
    var_20 = lambda self: var_19
    var_21 = None
    var_22 = {var_2: var_17, var_3: var_18, var_4: var_20, var_5: var_21}
    var_23 = [var_0, var_16, var_22]
    var_24 = {}
    var_25 = module_0.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = True
    var_28 = module_1.extract_text_array(var_26, var_27, var_27)
    var_29 = bool(var_28 == ['before ', 'inner', ' after'])
    assert var_29 is True



