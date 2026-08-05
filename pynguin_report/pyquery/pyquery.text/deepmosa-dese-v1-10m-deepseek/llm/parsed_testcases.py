####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello   world  '
    var_1 = '  foo   bar  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello '
    var_1 = 1
    var_2 = '  world  '
    var_3 = 2
    var_4 = '  foo  '
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._merge_original_parts(var_5)

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = '  '
    var_2 = 'hello'
    var_3 = [var_0, var_1, var_2, var_1, var_0]
    var_4 = module_0._merge_original_parts(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = '  '
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._merge_original_parts(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = '  a  '
    var_1 = 1
    var_2 = '  b  '
    var_3 = 2
    var_4 = '  c  '
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._merge_original_parts(var_5)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = '  hello  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello  '
    var_1 = 1
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = '   a   '
    var_1 = '   b   '
    var_2 = 1
    var_3 = '   c   '
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._merge_original_parts(var_4)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 5/13 statements.
# Partially parsed test_extract_text_with_block_break. Retrieved 5/13 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 8/19 statements.
# Partially parsed test_extract_text_squash_whitespace. Retrieved 5/13 statements.
# Partially parsed test_extract_text_strip_artifical_nl. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'hr'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'Before'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Para'
    var_2 = 'Tail'
    var_3 = 'div'
    var_4 = 'Start'

def test_case_0():
    var_0 = 'b'
    var_1 = 'Bold'
    var_2 = ' '
    var_3 = 'i'
    var_4 = 'Italic'
    var_5 = None
    var_6 = 'div'
    var_7 = None

def test_case_0():
    var_0 = 'span'
    var_1 = '  multiple   spaces  '
    var_2 = None
    var_3 = 'div'
    var_4 = '  leading'

def test_case_0():
    var_0 = 'div'
    var_1 = '  text  '



# Parsed testcases at query #3
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
    var_0 = None
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = [var_0, var_0, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = [var_0, var_1]
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
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_extract_text_simple_string.
# Failed to parse test_extract_text_with_child.
# Failed to parse test_extract_text_with_separator.
# Failed to parse test_extract_text_with_artificial_newline.
# Failed to parse test_extract_text_empty_dom.
# Failed to parse test_extract_text_multiple_children.
# Failed to parse test_extract_text_with_squash_space.
# Failed to parse test_extract_text_with_strip.
# Failed to parse test_extract_text_nested_separators.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test_extract_text_returns_empty_string_for_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_returns_text_for_simple_text_element. Retrieved 2/7 statements.
# Partially parsed test_extract_text_handles_nested_elements. Retrieved 5/13 statements.
# Partially parsed test_extract_text_inserts_newline_for_block_elements. Retrieved 5/13 statements.
# Partially parsed test_extract_text_uses_sep_symbol_for_separator_tags. Retrieved 2/7 statements.
# Partially parsed test_extract_text_squashes_multiple_spaces. Retrieved 2/7 statements.
# Partially parsed test_extract_text_strips_leading_and_trailing_whitespace. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello '

def test_case_0():
    var_0 = 'div'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello   World'

def test_case_0():
    var_0 = 'span'
    var_1 = '  Hello  '



# Parsed testcases at query #6
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/9 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 2/9 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_newlines_squashed. Retrieved 2/11 statements.
# Partially parsed test_extract_text_strip_whitespace. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_nested_blocks. Retrieved 2/8 statements.
# Partially parsed test_extract_text_with_separator_and_block. Retrieved 3/9 statements.
# Partially parsed test_extract_text_with_multiple_separators. Retrieved 3/10 statements.


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
    var_1 = 'hr'
    var_2 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = 'p'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_extract_text_simple_string. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 7/15 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 7/15 statements.
# Partially parsed test_extract_text_strips_whitespace. Retrieved 3/8 statements.
# Partially parsed test_extract_text_handles_none_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 7/15 statements.
# Partially parsed test_extract_text_strips_artifical_newlines. Retrieved 3/8 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None
    var_6 = '\n'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None
    var_6 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' text'
    var_3 = 'div'
    var_4 = 'Some '
    var_5 = None

def test_case_0():
    var_0 = 'hr'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'Before'
    var_5 = None
    var_6 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello  '
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'first'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'second'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = None



# Parsed testcases at query #9
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'text'
    var_2 = 99
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'middle'
    var_3 = 3
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._strip_artifical_nl(var_5)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 42
    var_1 = ''
    var_2 = 99
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'hello'
    var_2 = [var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 42
    var_2 = [var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'text'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2.0
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'data'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = [var_0, var_1, var_4]
    var_6 = module_0._strip_artifical_nl(var_5)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'hello'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 42
    var_2 = 'first'
    var_3 = 'second'
    var_4 = 3.14
    var_5 = True
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0._strip_artifical_nl(var_6)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_extract_text_array_empty_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_artifical_nl. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_no_squash. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_nested_tags. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_callable_tag_returns_empty. Retrieved 2/6 statements.


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
    var_1 = 'p'
    var_2 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_child. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_none_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 4/13 statements.
# Partially parsed test_extract_text_squash_whitespace. Retrieved 1/5 statements.
# Partially parsed test_extract_text_strip_artifical_nl. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'
    var_2 = 'b'
    var_3 = 'i'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 1/4 statements.
# Partially parsed test_extract_text_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_pre_tag. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<br>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div><br>A<br>B</div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'
    var_1 = False

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '<p>Hello<b>bold</b>World</p>'

def test_case_0():
    var_0 = '<div><br><br></div>'

def test_case_0():
    var_0 = '<pre>  line1\n  line2  </pre>'



# Parsed testcases at query #13
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_extract_text_returns_empty_string_for_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_returns_text_for_simple_text_node. Retrieved 2/7 statements.
# Partially parsed test_extract_text_handles_block_element. Retrieved 5/13 statements.
# Partially parsed test_extract_text_handles_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_squashes_whitespace. Retrieved 2/7 statements.
# Partially parsed test_extract_text_strips_outer_whitespace. Retrieved 2/7 statements.
# Partially parsed test_extract_text_preserves_inline_text. Retrieved 5/13 statements.
# Partially parsed test_extract_text_handles_nested_blocks. Retrieved 7/18 statements.
# Partially parsed test_extract_text_collapses_multiple_newlines. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'p'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = None

def test_case_0():
    var_0 = 'hr'
    var_1 = None
    var_2 = '--'

def test_case_0():
    var_0 = 'span'
    var_1 = '  Hello   World  '

def test_case_0():
    var_0 = 'span'
    var_1 = '   Hello   '

def test_case_0():
    var_0 = 'strong'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'p'
    var_4 = 'This is '

def test_case_0():
    var_0 = 'div'
    var_1 = 'Inner'
    var_2 = 'div'
    var_3 = None
    var_4 = None
    var_5 = 'div'
    var_6 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'First'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Second'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = '\n'



# Parsed testcases at query #15
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #16
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = '\n'
    var_3 = True
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)



# Parsed testcases at query #17
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = '\n'
    var_3 = True
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 13/17 statements.
# Partially parsed test_extract_text_with_child. Retrieved 17/27 statements.
# Partially parsed test_extract_text_separator. Retrieved 13/17 statements.
# Partially parsed test_extract_text_block_symbol. Retrieved 16/26 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 18/28 statements.
# Partially parsed test_extract_text_no_squash. Retrieved 18/28 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'attrib'
    var_6 = 'p'
    var_7 = 'Hello'
    var_8 = None
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {}
    var_12 = {var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_11}

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'attrib'
    var_6 = 'b'
    var_7 = 'bold'
    var_8 = ' tail'
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {}
    var_12 = {var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_11}
    var_13 = 'p'
    var_14 = 'before '
    var_15 = None
    var_16 = {}

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'attrib'
    var_6 = 'br'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {}
    var_11 = {var_1: var_6, var_2: var_7, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = '|'

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'attrib'
    var_6 = 'div'
    var_7 = 'block'
    var_8 = None
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {}
    var_12 = {var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_11}
    var_13 = 'p'
    var_14 = {}
    var_15 = '\n'

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'attrib'
    var_6 = 'span'
    var_7 = '  spaced  '
    var_8 = '  tail  '
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {}
    var_12 = {var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_11}
    var_13 = 'p'
    var_14 = '  text  '
    var_15 = None
    var_16 = {}
    var_17 = True

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'attrib'
    var_6 = 'span'
    var_7 = '  spaced  '
    var_8 = '  tail  '
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {}
    var_12 = {var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_11}
    var_13 = 'p'
    var_14 = '  text  '
    var_15 = None
    var_16 = {}
    var_17 = False

def test_case_0():
    var_0 = 'Mock'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'tail'
    var_4 = 'getchildren'
    var_5 = 'attrib'
    var_6 = 'p'
    var_7 = None
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = {}
    var_11 = {var_1: var_6, var_2: var_7, var_3: var_7, var_4: var_9, var_5: var_10}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_false. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = {}
    var_3 = False



# Parsed testcases at query #20
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = '\n'
    var_3 = True
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_extract_text_empty_dom.
# Failed to parse test_extract_text_simple_text.
# Failed to parse test_extract_text_with_separator.
# Failed to parse test_extract_text_with_block_element.
# Partially parsed test_extract_text_squash_space. Retrieved 1/7 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 1/7 statements.
# Partially parsed test_extract_text_custom_block_symbol. Retrieved 1/12 statements.
# Partially parsed test_extract_text_custom_sep_symbol. Retrieved 1/7 statements.
# Failed to parse test_extract_text_nested_blocks.
# Failed to parse test_extract_text_with_tail.
# Failed to parse test_extract_text_inline_tag.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = '|'

def test_case_0():
    var_0 = '---'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_true. Retrieved 11/17 statements.


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
    var_10 = False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_2_evaluates_to_false. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'hello'
    var_2 = []
    var_3 = lambda : var_2



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_squash_space_true_strips_result. Retrieved 7/8 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '  hello  '
    var_2 = True
    var_3 = '  world  '
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = []
    var_6 = module_0.extract_text(var_5, squash_space=var_2)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_callable_dom_tag_returns_empty_string. Retrieved 11/14 statements.


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
    var_10 = True



# Parsed testcases at query #26
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_extract_text_array_empty_dom.
# Failed to parse test_extract_text_array_separator_tag.
# Failed to parse test_extract_text_array_inline_tag_with_text.
# Failed to parse test_extract_text_array_block_tag_with_text.
# Failed to parse test_extract_text_array_child_elements.
# Failed to parse test_extract_text_array_mixed_content.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 2/3 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 2/3 statements.
# Partially parsed test_extract_text_array_both_options. Retrieved 1/2 statements.
# Partially parsed test_extract_text_array_no_options. Retrieved 1/2 statements.


def test_case_0():
    var_0 = True
    var_1 = False

def test_case_0():
    var_0 = False
    var_1 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_br. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 1/4 statements.
# Partially parsed test_extract_text_block_inside_block. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_leading_trailing_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_multiple_blocks. Retrieved 1/4 statements.
# Partially parsed test_extract_text_mixed_inline_block. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello world</p>'

def test_case_0():
    var_0 = '<p>Line1<br>Line2</p>'

def test_case_0():
    var_0 = '<hr>'

def test_case_0():
    var_0 = '<span>Hello <b>world</b></span>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<p>  Hello   world  </p>'

def test_case_0():
    var_0 = '<div>  <p>Text</p>  </div>'

def test_case_0():
    var_0 = '<div><p>One</p><p>Two</p><p>Three</p></div>'

def test_case_0():
    var_0 = '<div>Start <p>Middle</p> End</div>'

def test_case_0():
    var_0 = '<div></div>'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_single_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_with_nl_squash. Retrieved 2/7 statements.


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
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = None
    var_3 = 'p'
    var_4 = 'before '

def test_case_0():
    var_0 = 'div'
    var_1 = None



# Parsed testcases at query #30
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #31
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = False
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)
    var_4 = len(var_3)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_squash_space_false_does_not_strip. Retrieved 4/8 statements.


def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = None
    var_3 = True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '\n'
    var_1 = False



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_false. Retrieved 10/13 statements.


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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_extract_text_array_empty_dom_no_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_plain_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag_no_newlines. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_nested_elements. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_false. Retrieved 3/8 statements.


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
    var_0 = 'b'
    var_1 = 'bold'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'parent'

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = False



# Parsed testcases at query #36
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_callable_dom_tag_returns_empty_string. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = None
    var_4 = lambda : var_3
    var_5 = {var_2: var_4}



# Parsed testcases at query #38
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    assert var_1 is True
    var_2 = True
    var_3 = module_0.extract_text_array(var_0, var_2)
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = module_0._squash_artifical_nl(var_4)
    var_6 = module_0._strip_artifical_nl(var_5)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_callable_tag_returns_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 5/9 statements.


def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = None
    var_3 = '\n'
    var_4 = True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_extract_text_no_content. Retrieved 2/7 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 5/13 statements.
# Partially parsed test_extract_text_block_tag. Retrieved 5/13 statements.
# Partially parsed test_extract_text_multiple_blocks. Retrieved 8/19 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_squash_whitespace. Retrieved 2/7 statements.
# Partially parsed test_extract_text_strip_artifical_newlines. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'a'

def test_case_0():
    var_0 = 'p'
    var_1 = 'inner'
    var_2 = None
    var_3 = 'div'
    var_4 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'first'
    var_2 = None
    var_3 = 'p'
    var_4 = 'second'
    var_5 = None
    var_6 = 'div'
    var_7 = None

def test_case_0():
    var_0 = 'a'
    var_1 = 'link'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'before '

def test_case_0():
    var_0 = 'div'
    var_1 = 'hello   world'

def test_case_0():
    var_0 = 'span'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'start'



# Parsed testcases at query #42
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #43
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 6/14 statements.
# Partially parsed test_extract_text_block_element. Retrieved 6/14 statements.
# Partially parsed test_extract_text_squash_whitespace. Retrieved 3/8 statements.
# Partially parsed test_extract_text_strip_leading_trailing. Retrieved 3/8 statements.
# Partially parsed test_extract_text_none_text. Retrieved 3/8 statements.


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
    var_1 = 'world'
    var_2 = '!'
    var_3 = 'p'
    var_4 = 'Hello '
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Second'
    var_2 = None
    var_3 = 'div'
    var_4 = 'First'
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = '  Hello   World  '
    var_2 = None

def test_case_0():
    var_0 = 'p'
    var_1 = '  Hello  '
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_true. Retrieved 8/12 statements.


def test_case_0():
    var_0 = None
    var_1 = True
    assert var_1 is True
    var_2 = None
    var_3 = True
    var_4 = 'hello'
    var_5 = [var_2, var_3, var_4, var_2]
    var_6 = ''
    var_7 = '\n'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_false. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'some_tag'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_squash_space_true_condition. Retrieved 10/15 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    assert var_1 is True
    var_2 = module_0.extract_text_array(var_0, var_1)
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = ''
    var_7 = None
    var_8 = '\n'
    var_9 = True



# Parsed testcases at query #48
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 2/8 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 1/5 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 2/6 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_only_block. Retrieved 2/6 statements.


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
    var_0 = 'p'
    var_1 = False

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'div'
    var_1 = 'br'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_callable_dom_tag_returns_empty_string. Retrieved 1/4 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #51
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_extract_text_returns_empty_string_for_callable_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_returns_text_for_simple_text_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_returns_text_with_separator. Retrieved 6/14 statements.
# Partially parsed test_extract_text_returns_text_with_block_symbol. Retrieved 6/14 statements.
# Partially parsed test_extract_text_squashes_space. Retrieved 3/8 statements.
# Partially parsed test_extract_text_handles_nested_inline_tags. Retrieved 6/14 statements.
# Partially parsed test_extract_text_returns_empty_for_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_strips_leading_and_trailing_whitespace. Retrieved 3/8 statements.


def test_case_0():
    var_0 = lambda : None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'Line1'
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Child'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Parent'
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = '  Hello   World  '
    var_2 = None

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' and '
    var_3 = 'p'
    var_4 = 'Text '
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'p'
    var_1 = '  content  '
    var_2 = None



# Parsed testcases at query #53
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #54
#--------------------------

# Failed to parse test_predicate_evaluates_to_false.




# Parsed testcases at query #55
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #56
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_callable_dom_tag_returns_empty_string. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = None
    var_4 = lambda : var_3
    var_5 = {var_2: var_4}



# Parsed testcases at query #58
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_extract_text_with_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_blocks. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_empty_element. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_only_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_leading_trailing_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_custom_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_pre_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_br_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_none_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_deeply_nested. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_blocks_and_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_only_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_squash_space_false_and_whitespace. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_artifical_nl_stripping. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_custom_block_and_sep. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_empty_children. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_text_and_tail. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_br. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_single_inline. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_only_block. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_only_blocks. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_text_after_block. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_text_before_block. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_comment. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_processing_instruction. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello world</p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<hr>'

def test_case_0():
    var_0 = '<span>Hello <b>world</b></span>'

def test_case_0():
    var_0 = '<div><div><p>Text</p></div></div>'

def test_case_0():
    var_0 = '<hr><hr>'

def test_case_0():
    var_0 = '<p>Hello <b>bold</b> and <i>italic</i></p>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>   </p>'

def test_case_0():
    var_0 = '<p>   Hello   </p>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<hr>'
    var_1 = '|'

def test_case_0():
    var_0 = '<p>Hello   world</p>'
    var_1 = False

def test_case_0():
    var_0 = '<pre>Hello\nworld</pre>'

def test_case_0():
    var_0 = '<p>Line1<br>Line2</p>'

def test_case_0():
    var_0 = '<div><p></p></div>'

def test_case_0():
    var_0 = '<div><span><p><b>Deep</b></p></span></div>'

def test_case_0():
    var_0 = '<div>Start<p>Middle</p>End</div>'

def test_case_0():
    var_0 = '<hr>'

def test_case_0():
    var_0 = '<p>  Hello  </p>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p>A</p></div>'

def test_case_0():
    var_0 = '<div><p>A</p><hr><p>B</p></div>'
    var_1 = '\n'

def test_case_0():
    var_0 = '<div><p></p><p></p></div>'

def test_case_0():
    var_0 = '<div>Hello<b>bold</b>world</div>'

def test_case_0():
    var_0 = '<p>A<br><br>B</p>'

def test_case_0():
    var_0 = '<div><hr><p>Text</p><hr></div>'

def test_case_0():
    var_0 = '<span>Text</span>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>   </p><p>   </p>'

def test_case_0():
    var_0 = '<div><p>Block</p>After</div>'

def test_case_0():
    var_0 = '<div>Before<p>Block</p></div>'

def test_case_0():
    var_0 = '<div><!-- comment --><p>Text</p></div>'

def test_case_0():
    var_0 = "<?xml version='1.0'?><div><p>Text</p></div>"

def test_case_0():
    pass



# Parsed testcases at query #60
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '  hello  '
    var_2 = [var_0, var_1, var_0]
    var_3 = False
    var_4 = module_0.extract_text(var_2, squash_space=var_3)
    assert var_4 == '\n  hello  \n'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_extract_text_with_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_nested_inline_tags. Retrieved 2/8 statements.
# Partially parsed test_extract_text_with_non_inline_and_separator. Retrieved 4/10 statements.
# Partially parsed test_extract_text_with_squash_space_disabled. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_only_separator. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_nested_block_elements. Retrieved 2/8 statements.
# Partially parsed test_extract_text_with_tail_after_block. Retrieved 2/8 statements.
# Partially parsed test_extract_text_with_multiple_none_parts. Retrieved 2/11 statements.
# Partially parsed test_extract_text_with_squash_space_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = '---'

def test_case_0():
    var_0 = 'p'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'hr'
    var_3 = '---'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'hr'
    var_1 = '---'

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
    var_2 = True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_block_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
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
    var_0 = 'div'
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_single_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_artifical_nl. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_nested. Retrieved 5/13 statements.
# Partially parsed test_extract_text_array_squash_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_strip_nl. Retrieved 5/10 statements.


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
    var_0 = 'p'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'world'
    var_2 = None
    var_3 = 'div'
    var_4 = 'hello '

def test_case_0():
    var_0 = 'p'
    var_1 = 'a'
    var_2 = None
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'p'
    var_1 = 'test'
    var_2 = None
    var_3 = False
    var_4 = True



# Parsed testcases at query #64
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #65
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_extract_text_with_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_paragraph_and_break. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_paragraphs. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_empty_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_only_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_squash_space_true. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_complex_structure. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_nested_blocks. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_mixed_inline_and_block. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_leading_trailing_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_artifical_newlines_squashed. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_document. Retrieved 1/4 statements.
# Partially parsed test_extract_text_single_text_node. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello world</p>'

def test_case_0():
    var_0 = '<div><p>First paragraph</p><br/><p>Second paragraph</p></div>'

def test_case_0():
    var_0 = '<hr/>'

def test_case_0():
    var_0 = '<div><hr/><p>Text</p><hr/></div>'

def test_case_0():
    var_0 = '<div><p>Hello <b>bold</b> world</p></div>'

def test_case_0():
    var_0 = '<div><p>First</p>Tail text<p>Second</p></div>'

def test_case_0():
    var_0 = '<div><p>Para1</p><p>Para2</p><p>Para3</p></div>'

def test_case_0():
    var_0 = '<div><p></p><p>Text</p><p></p></div>'

def test_case_0():
    var_0 = '<div><p>   </p><p>Text</p></div>'

def test_case_0():
    var_0 = '<div><p>Hello   world</p></div>'
    var_1 = True

def test_case_0():
    var_0 = '<div><p>Hello   world</p></div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div><hr/><p>Text</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '\n    <div>\n        <h1>Title</h1>\n        <p>First paragraph</p>\n        <hr/>\n        <p>Second paragraph</p>\n        <ul>\n            <li>Item 1</li>\n            <li>Item 2</li>\n        </ul>\n    </div>\n    '
    var_1 = 'Title\nFirst paragraph\nSecond paragraph\nItem 1\nItem 2'

def test_case_0():
    var_0 = '<div><div><p>Nested</p></div><p>Text</p></div>'

def test_case_0():
    var_0 = '<div><span>Inline</span><p>Block</p></div>'

def test_case_0():
    var_0 = '<div>  <p>Text</p>  </div>'

def test_case_0():
    var_0 = '<div><p>Text</p><p>More</p></div>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>Just text</p>'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'MockDOM'
    var_1 = ()
    var_2 = {}
    var_3 = 'text'
    var_4 = False
    var_5 = (var_3, var_4)
    var_6 = True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 2/6 statements.
# Partially parsed test_extract_text_with_newline. Retrieved 2/6 statements.
# Partially parsed test_extract_text_separator. Retrieved 2/6 statements.
# Partially parsed test_extract_text_whitespace_squashing. Retrieved 2/6 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 2/6 statements.
# Partially parsed test_extract_text_empty. Retrieved 2/6 statements.
# Partially parsed test_extract_text_br. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'
    var_1 = './/p'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'
    var_1 = './/div'

def test_case_0():
    var_0 = '<hr>'
    var_1 = './/hr'

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'
    var_1 = './/p'

def test_case_0():
    var_0 = '<p>Hello <b>World</b></p>'
    var_1 = './/p'

def test_case_0():
    var_0 = '<div></div>'
    var_1 = './/div'

def test_case_0():
    var_0 = '<p>Line1<br>Line2</p>'
    var_1 = './/p'



# Parsed testcases at query #2
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #3
#--------------------------




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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 2
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = [var_0, var_1, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = [var_0, var_0, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_0, var_2, var_0]
    var_4 = module_0._squash_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0._squash_artifical_nl(var_1)



# Parsed testcases at query #4
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello   world  '
    var_1 = '  foo  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello  '
    var_1 = 1
    var_2 = '  world  '
    var_3 = 2
    var_4 = '  foo  '
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._merge_original_parts(var_5)

import pyquery.text as module_0

def test_case_0():
    var_0 = '  '
    var_1 = 1
    var_2 = [var_0, var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = '   '
    var_1 = [var_0, var_0]
    var_2 = module_0._merge_original_parts(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello  '
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)



# Parsed testcases at query #5
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = True
    var_3 = module_0.extract_text(var_0, squash_space=var_2)



# Parsed testcases at query #6
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text_array(var_0, var_1)
    var_3 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #7
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #8
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)

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

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)

import pyquery.text as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = 2.5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._strip_artifical_nl(var_5)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_children. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_symbol_custom. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_sep_symbol_custom. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_pre_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_script_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_style_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_comment. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_mixed_blocks_and_inline. Retrieved 1/4 statements.
# Partially parsed test_extract_text_artifical_newline_stripping. Retrieved 1/4 statements.
# Partially parsed test_extract_text_leading_trailing_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_blocks. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<hr>'

def test_case_0():
    var_0 = '<div><span>Text</span></div>'

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '<div><b>Bold</b> and <i>Italic</i></div>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = '<br>'

def test_case_0():
    var_0 = '<hr>'
    var_1 = '---'

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'
    var_1 = False

def test_case_0():
    var_0 = '<pre>  Preserved  </pre>'

def test_case_0():
    var_0 = "<script>alert('test');</script>"

def test_case_0():
    var_0 = '<style>body { color: red; }</style>'

def test_case_0():
    var_0 = '<!-- comment --><p>Text</p>'

def test_case_0():
    var_0 = '<div><p>Para</p>Tail</div>'

def test_case_0():
    var_0 = '<hr><hr>'

def test_case_0():
    var_0 = '<div><p>One</p><span>Two</span><p>Three</p></div>'

def test_case_0():
    var_0 = '<div><p>Start</p></div>'

def test_case_0():
    var_0 = '  <p>Middle</p>  '

def test_case_0():
    var_0 = '<div><div><p>Deep</p></div></div>'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block. Retrieved 3/8 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 3/8 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 3/8 statements.
# Partially parsed test_extract_text_nested_children. Retrieved 8/19 statements.
# Partially parsed test_extract_text_with_block_and_separator. Retrieved 6/14 statements.
# Partially parsed test_extract_text_empty_child. Retrieved 5/13 statements.


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

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = '\n'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Line1'
    var_2 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = False

def test_case_0():
    var_0 = 'b'
    var_1 = 'Bold'
    var_2 = ' after bold'
    var_3 = 'span'
    var_4 = 'Some '
    var_5 = ' after span'
    var_6 = 'div'
    var_7 = 'Start '

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'p'
    var_4 = 'Line1'
    var_5 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = ''
    var_2 = None
    var_3 = 'div'
    var_4 = 'A'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_child. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/9 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 1/5 statements.
# Partially parsed test_extract_text_strip_newlines. Retrieved 2/9 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 3/10 statements.
# Partially parsed test_extract_text_block_symbol. Retrieved 3/10 statements.
# Partially parsed test_extract_text_sep_symbol. Retrieved 4/10 statements.


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

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'

def test_case_0():
    var_0 = 'div'

def test_case_0():
    var_0 = 'p'
    var_1 = 'b'
    var_2 = 'i'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = 'p'
    var_3 = '---'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_with_child. Retrieved 2/7 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 2/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/9 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 2/8 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 3/11 statements.
# Partially parsed test_extract_text_squash_whitespace. Retrieved 1/5 statements.
# Partially parsed test_extract_text_block_symbol. Retrieved 3/10 statements.
# Partially parsed test_extract_text_sep_symbol. Retrieved 4/10 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/10 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.


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
    var_1 = 'b'

def test_case_0():
    var_0 = 'p'
    var_1 = 'b'
    var_2 = 'i'

def test_case_0():
    var_0 = 'p'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = ' '

def test_case_0():
    var_0 = 'div'
    var_1 = 'hr'
    var_2 = 'p'
    var_3 = ' '

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False

def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'p'
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
    var_1 = 'inline'
    var_2 = False

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' tail'
    var_3 = 'p'
    var_4 = 'start'
    var_5 = False

def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'p'
    var_1 = 'text'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = lambda : None
    var_1 = None



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_predicate_at_line2_evaluates_to_false.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_inline. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_leading_trailing_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_multiple_spaces_squashed. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_only_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_comment_tag. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<div><br/>Text after br</div>'

def test_case_0():
    var_0 = '<p>Hello <b>bold</b> world</p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '  <p>  Text  </p>  '

def test_case_0():
    var_0 = '<p>Hello    world</p>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<br/>'

def test_case_0():
    var_0 = '<!-- comment --><p>text</p>'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_child. Retrieved 5/13 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 6/14 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 3/8 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 5/13 statements.
# Partially parsed test_extract_text_nested_blocks. Retrieved 8/19 statements.


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

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = '\n'

def test_case_0():
    var_0 = 'p'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Inner'
    var_2 = ' Tail'
    var_3 = 'div'
    var_4 = 'Start '

def test_case_0():
    var_0 = 'span'
    var_1 = 'deep'
    var_2 = None
    var_3 = 'div'
    var_4 = 'middle '
    var_5 = ' end'
    var_6 = 'div'
    var_7 = 'start '



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_line11_evaluates_to_true. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = {}
    var_3 = '  hello  '
    var_4 = True



# Parsed testcases at query #18
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = '\n'
    var_2 = False
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)
    assert var_3 == ''



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 11/15 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 10/14 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 12/22 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 19/32 statements.
# Partially parsed test_extract_text_whitespace_squashing. Retrieved 13/23 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 11/15 statements.
# Partially parsed test_extract_text_no_text_content. Retrieved 10/14 statements.


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

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'hr'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_6}

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
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'div'

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
    var_9 = ' and '
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'i'
    var_12 = 'Italic'
    var_13 = []
    var_14 = lambda self: var_13
    var_15 = None
    var_16 = {var_1: var_11, var_2: var_12, var_3: var_14, var_4: var_15}
    var_17 = 'p'
    var_18 = 'Start '

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'span'
    var_6 = '  spaced  '
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'div'
    var_12 = '  text  '

def test_case_0():
    var_0 = 'MockDom'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = 'tail'
    var_5 = 'p'
    var_6 = ''
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = None
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}

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



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_extract_text_array_predicate_true.




# Parsed testcases at query #21
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = '\n'
    var_2 = False
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 10/13 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 10/13 statements.
# Partially parsed test_extract_text_with_child. Retrieved 15/23 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 15/23 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 15/23 statements.
# Partially parsed test_extract_text_squash_space_enabled. Retrieved 16/24 statements.
# Partially parsed test_extract_text_squash_space_disabled. Retrieved 16/24 statements.


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

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'p'
    var_6 = 'Hello'
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}

def test_case_0():
    var_0 = 'MockDom'
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
    var_12 = ()
    var_13 = 'div'
    var_14 = 'Hello '

def test_case_0():
    var_0 = 'MockDom'
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
    var_11 = ()
    var_12 = 'div'
    var_13 = 'Line1'
    var_14 = ' '

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'p'
    var_7 = 'Block1'
    var_8 = None
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = ()
    var_13 = 'div'
    var_14 = '\n'

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'span'
    var_7 = '  spaced  '
    var_8 = '  text  '
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = ()
    var_13 = 'div'
    var_14 = '  hello  '
    var_15 = True

def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = 'span'
    var_7 = '  spaced  '
    var_8 = '  text  '
    var_9 = []
    var_10 = lambda self: var_9
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_10}
    var_12 = ()
    var_13 = 'div'
    var_14 = '  hello  '
    var_15 = False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_callable_dom_tag_returns_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #24
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    assert var_1 is True
    var_2 = module_0.extract_text_array(var_0, var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_single_text_node. Retrieved 5/13 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 9/20 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 9/20 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 6/14 statements.
# Partially parsed test_extract_text_strip_artifical_nl. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None
    var_3 = 'div'
    var_4 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'First'
    var_2 = None
    var_3 = 'p'
    var_4 = 'Second'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = '|'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'span'
    var_4 = 'text'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = ' '

def test_case_0():
    var_0 = 'span'
    var_1 = '  Hello  '
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
    var_2 = None
    var_3 = 'div'
    var_4 = None



# Parsed testcases at query #26
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = True
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 11/14 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 12/15 statements.
# Partially parsed test_extract_text_with_children. Retrieved 14/22 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 12/15 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 13/16 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 15/23 statements.
# Partially parsed test_extract_text_squash_space_true. Retrieved 15/23 statements.


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

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = 'Hello'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = 'World'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = ()
    var_13 = 'div'

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
    var_11 = '\n'

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'p'
    var_7 = 'Hello'
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = '\n'

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = '  '
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = ()
    var_13 = 'div'
    var_14 = False

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = 'span'
    var_7 = '  '
    var_8 = []
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_10}
    var_12 = ()
    var_13 = 'div'
    var_14 = True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_extract_text_squash_space_predicate_true. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'Dom'
    var_1 = ()
    var_2 = {}
    var_3 = True



# Parsed testcases at query #29
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = '\n'
    var_3 = False
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)



# Parsed testcases at query #30
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 1/5 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_array_artifical_nl. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_squash_nl. Retrieved 4/11 statements.
# Partially parsed test_extract_text_array_strip_nl. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_tail. Retrieved 2/8 statements.
# Partially parsed test_extract_text_array_nested_inline. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_no_squash_no_strip. Retrieved 3/10 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/6 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 3/11 statements.


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
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'b'

def test_case_0():
    var_0 = 'div'
    var_1 = 'span'

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'br'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_squash_space_true_strips_result. Retrieved 3/4 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_callable_dom_tag. Retrieved 1/4 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_callable_dom_tag_returns_empty_string. Retrieved 11/14 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'tail'
    var_5 = 'getchildren'
    var_6 = None
    var_7 = lambda : var_6
    var_8 = []
    var_9 = lambda : var_8
    var_10 = {var_2: var_7, var_3: var_6, var_4: var_6, var_5: var_9}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'not_callable'



# Parsed testcases at query #36
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 5/13 statements.
# Partially parsed test_extract_text_with_block. Retrieved 5/13 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 8/19 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'Line1'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Paragraph'
    var_2 = None
    var_3 = 'div'
    var_4 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'World'
    var_5 = None
    var_6 = 'div'
    var_7 = None

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello  '
    var_2 = False

def test_case_0():
    var_0 = 'hr'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'Before'
    var_5 = '---'

def test_case_0():
    var_0 = 'p'
    var_1 = 'Para'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = '\n\n'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_extract_text_predicate_false. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'MockDOM'
    var_1 = {}
    var_2 = 'MockBody'
    var_3 = {}
    var_4 = 'text'
    var_5 = False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_separator. Retrieved 4/9 statements.
# Partially parsed test_extract_text_block. Retrieved 4/9 statements.
# Partially parsed test_extract_text_whitespace_squash. Retrieved 4/9 statements.
# Partially parsed test_extract_text_nested_with_tail. Retrieved 6/14 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_no_text. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = ' world'
    var_2 = None
    var_3 = 'p'
    var_4 = 'Hello'
    var_5 = None

def test_case_0():
    var_0 = 'hr'
    var_1 = None
    var_2 = None
    var_3 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
    var_2 = None
    var_3 = '\n'

def test_case_0():
    var_0 = 'p'
    var_1 = '   Hello   '
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'b'
    var_1 = 'bold'
    var_2 = ' text'
    var_3 = 'p'
    var_4 = 'Some '
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = ''
    var_2 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = '\n'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_extract_text_array_no_children_no_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text_and_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/7 statements.
# Partially parsed test_extract_text_array_nested_tags. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'child_text'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'parent_text'
    var_5 = False

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
    var_1 = None
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = lambda : None
    var_1 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = ' mid_tail'
    var_6 = 'div'
    var_7 = None
    var_8 = False



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_extract_text_array_predicate_false. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None



# Parsed testcases at query #42
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_extract_text_with_squash_space_true. Retrieved 3/4 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #44
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = '\n'
    var_2 = True
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)
    assert var_3 == ''



# Parsed testcases at query #45
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_true. Retrieved 10/13 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    assert var_1 is True
    var_2 = True
    var_3 = module_0.extract_text_array(var_0, var_2)
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = module_0._squash_artifical_nl(var_4)
    var_6 = module_0._strip_artifical_nl(var_5)
    var_7 = ''
    var_8 = None
    var_9 = '\n'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_break. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator_and_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested_blocks. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_multiple_blocks. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_custom_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_whitespace_only_content. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello world</p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<hr>'

def test_case_0():
    var_0 = '<p>Before<hr>After</p>'

def test_case_0():
    var_0 = '<p>Hello <b>bold</b> world</p>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div><div><p>Deep</p></div></div>'

def test_case_0():
    var_0 = '<p>Hello <b>bold</b> tail</p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p><p>Third</p></div>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<p>Before<hr>After</p>'
    var_1 = '|'

def test_case_0():
    var_0 = '<p>  Hello   world  </p>'
    var_1 = False

def test_case_0():
    var_0 = '<p>   </p>'



# Parsed testcases at query #48
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_simple_text. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_nested_elements. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_multiple_children. Retrieved 10/21 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_squash_and_strip. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_separator_with_text. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_separator_in_nested. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_multiple_artifical_nl_squash. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_strip_removes_leading_trailing_none. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_empty_children. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl_with_only_none. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_squash_multiple_none. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'A'
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
    var_1 = 'Content'
    var_2 = None
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello '
    var_5 = None
    var_6 = False

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
    var_9 = False

def test_case_0():
    var_0 = 'a'
    var_1 = 'Link'
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'Text'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None
    var_6 = True

def test_case_0():
    var_0 = 'br'
    var_1 = 'break'
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = ' '
    var_3 = 'div'
    var_4 = 'Line1'
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
    var_1 = 'Middle'
    var_2 = None
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = lambda : None
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = True
    var_4 = False



# Parsed testcases at query #50
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == ''



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_at_line_2_evaluates_to_false. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'div'
    var_4 = {var_2: var_3}



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_squash_space_affects_condition. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_false. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'span'
    var_6 = None
    var_7 = []
    var_8 = lambda self: var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}



# Parsed testcases at query #54
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_span. Retrieved 1/4 statements.
# Partially parsed test_extract_text_block_symbol. Retrieved 1/4 statements.
# Partially parsed test_extract_text_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_nested. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty. Retrieved 1/4 statements.
# Partially parsed test_extract_text_only_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_mixed_inline_block. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<p>Hello <span>World</span></p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<div><hr/>Hello</div>'

def test_case_0():
    var_0 = '<div><p><b>Bold</b> text</p></div>'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>   </p>'

def test_case_0():
    var_0 = '<hr/><hr/>'

def test_case_0():
    var_0 = '<div>Text <span>inline</span><p>block</p></div>'

def test_case_0():
    var_0 = '<p>Hello</p>World'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_squash_space_true_strips_result. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'MockDom'
    var_1 = {}
    var_2 = True



# Parsed testcases at query #57
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_extract_text_with_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_inline_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_squash_space_disabled. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_only_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_pre_tag. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_mixed_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_none_text. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<hr><p>Text</p>'

def test_case_0():
    var_0 = '<p>Hello <b>World</b></p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<hr><p>Text</p>'
    var_1 = '---'

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<p>   </p>'

def test_case_0():
    var_0 = '<pre>  Preformatted  text  </pre>'

def test_case_0():
    var_0 = '<div><p>Line1</p><hr><p>Line2</p></div>'

def test_case_0():
    var_0 = '<p>Start<b>bold</b>End</p>'

def test_case_0():
    var_0 = '<hr><hr><p>Text</p>'

def test_case_0():
    var_0 = '<div><p></p></div>'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_callable_dom_tag_returns_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 2/7 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 2/7 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 5/13 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 5/13 statements.
# Partially parsed test_extract_text_with_inline_element. Retrieved 5/13 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 8/19 statements.
# Partially parsed test_extract_text_squash_whitespace. Retrieved 2/7 statements.
# Partially parsed test_extract_text_block_symbol_custom. Retrieved 6/14 statements.
# Partially parsed test_extract_text_sep_symbol_custom. Retrieved 6/14 statements.
# Partially parsed test_extract_text_nested_blocks. Retrieved 7/18 statements.


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
    var_3 = 'div'
    var_4 = 'Line1'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Block'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Before'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Inline'
    var_2 = None
    var_3 = 'p'
    var_4 = 'Start'

def test_case_0():
    var_0 = 'span'
    var_1 = 'A'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'B'
    var_5 = None
    var_6 = 'p'
    var_7 = None

def test_case_0():
    var_0 = 'p'
    var_1 = '  Hello   World  '

def test_case_0():
    var_0 = 'div'
    var_1 = 'Block'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Start'
    var_5 = ' | '

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'A'
    var_5 = ' | '

def test_case_0():
    var_0 = 'div'
    var_1 = 'Inner'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = 'div'
    var_6 = 'Outer'



# Parsed testcases at query #61
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_nested_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_comment. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_newlines. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<p>Hello World</p>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div><br/>Text</div>'
    var_1 = '---'

def test_case_0():
    var_0 = '<p>  Hello   World  </p>'
    var_1 = False

def test_case_0():
    var_0 = '<div><span>Hello</span> <span>World</span></div>'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '<p>Hello<!-- comment -->World</p>'

def test_case_0():
    var_0 = '<div>\n<p>First</p>\n<p>Second</p>\n</div>'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_callable_dom_tag_returns_empty_string. Retrieved 11/14 statements.


def test_case_0():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = 'tail'
    var_6 = None
    var_7 = lambda : var_6
    var_8 = []
    var_9 = lambda : var_8
    var_10 = {var_2: var_7, var_3: var_6, var_4: var_9, var_5: var_6}



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_paragraphs. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_line_break. Retrieved 1/4 statements.
# Partially parsed test_extract_text_strips_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_block_symbol_custom. Retrieved 2/5 statements.
# Partially parsed test_extract_text_sep_symbol_custom. Retrieved 2/5 statements.
# Partially parsed test_extract_text_squash_space_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello</div>'

def test_case_0():
    var_0 = '<div><p>First</p><p>Second</p></div>'

def test_case_0():
    var_0 = '<div><hr/><p>Text</p></div>'

def test_case_0():
    var_0 = '<div>Line1<br/>Line2</div>'

def test_case_0():
    var_0 = '<div>  Hello  </div>'

def test_case_0():
    var_0 = '<div><span>Hello</span> <b>World</b></div>'

def test_case_0():
    var_0 = '<div><p>A</p><p>B</p></div>'
    var_1 = ' | '

def test_case_0():
    var_0 = '<div>X<hr/>Y</div>'
    var_1 = ' | '

def test_case_0():
    var_0 = '<div>  Hello  </div>'
    var_1 = False



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = {}
    var_3 = []
    var_4 = True



# Parsed testcases at query #66
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_predicate_false. Retrieved 10/15 statements.


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



