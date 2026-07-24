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
    var_0 = '  hello  '
    var_1 = '  world  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
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
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._merge_original_parts(var_4)
    var_6 = bool(var_5 == ['hello world', 1, 2])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  '
    var_1 = [var_0, var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello\n  '
    var_1 = '  world  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello    world  '
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == ['hello world'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello\tworld  '
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == ['hello world'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello \n \t world  '
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == ['hello world'])
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_single_text_node. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_nested_elements. Retrieved 6/14 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = '\n'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = '\n'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None

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
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'World'
    var_5 = '!'
    var_6 = 'div'
    var_7 = None
    var_8 = None

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
    var_1 = 'Hello'
    var_2 = None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_extract_text_with_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_custom_block_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_custom_sep_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_complex_structure. Retrieved 9/20 statements.
# Partially parsed test_extract_text_with_preformatted_content. Retrieved 3/8 statements.


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
    var_1 = 'Nested'
    var_2 = ' Text'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = 'World'

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
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = 'World'
    var_3 = '|'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = '|'

def test_case_0():
    var_0 = 'span'
    var_1 = 'child1'
    var_2 = ' tail1'
    var_3 = 'div'
    var_4 = 'child2'
    var_5 = ' tail2'
    var_6 = 'div'
    var_7 = 'start'
    var_8 = 'end'

def test_case_0():
    var_0 = 'pre'
    var_1 = '  preformatted  \n  text  '
    var_2 = None



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
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 123
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == [123])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = '!'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['hello', 'world', '!'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 123
    var_1 = 'hello'
    var_2 = 456
    var_3 = 'world'
    var_4 = 789
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._strip_artifical_nl(var_5)
    var_7 = bool(var_6 == ['hello', 'world'])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = 123
    var_2 = 'middle'
    var_3 = 456
    var_4 = 'end'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._strip_artifical_nl(var_5)
    var_7 = bool(var_6 == ['start', 123, 'middle', 456, 'end'])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 123
    var_1 = 456
    var_2 = 789
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == [123, 456, 789])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 123
    var_1 = 'hello'
    var_2 = 456
    var_3 = 'world'
    var_4 = 789
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._strip_artifical_nl(var_5)
    var_7 = bool(var_6 == ['hello', 'world'])
    assert var_7 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 123
    var_1 = 'hello'
    var_2 = 456
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['hello'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'hello'
    var_2 = 123
    var_3 = 'world'
    var_4 = [var_0, var_1, var_2, var_3, var_0]
    var_5 = module_0._strip_artifical_nl(var_4)
    var_6 = bool(var_5 == ['hello', 123, 'world'])
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_custom_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_without_squash_space. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_preformatted_content. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'

def test_case_0():
    var_0 = '<div>Hello <strong>World</strong></div>'

def test_case_0():
    var_0 = '<div><h1>Title</h1><p>Content</p></div>'

def test_case_0():
    var_0 = '<div>  Hello   \n  World  </div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div><h1>Title</h1><p>Content</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>  Hello   \n  World  </div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><pre>  Hello   \n  World  </pre></div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><ul><li>Item 1</li><li>Item 2</li></ul><p>World</p></div>'



# Parsed testcases at query #6
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = '\n'
    var_2 = '\n'
    var_3 = True
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 5/10 statements.
# Partially parsed test_extract_text_nested_blocks. Retrieved 6/14 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_with_preformatted_content. Retrieved 4/9 statements.
# Partially parsed test_extract_text_strip_artificial_nl. Retrieved 3/8 statements.


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
    var_2 = ' Tail'
    var_3 = 'div'
    var_4 = 'Parent Text'
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
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = '|'
    var_4 = ';'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Inner'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Outer'
    var_5 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'First'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'Second'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = None

def test_case_0():
    var_0 = 'pre'
    var_1 = '  Preformatted  \n  Text  '
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_extract_text_with_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_multiple_whitespace. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_preformatted_content. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail_content. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div><h1>Title</h1><p>Content</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>  Hello   World  </div>'
    var_1 = False

def test_case_0():
    var_0 = '<div>  Hello   World  </div>'

def test_case_0():
    var_0 = '<div><span>Hello</span> <span>World</span></div>'

def test_case_0():
    var_0 = '<div><pre>  Hello   World  </pre></div>'

def test_case_0():
    var_0 = '<div><h1>Title</h1><p>Paragraph 1</p><p>Paragraph 2</p></div>'

def test_case_0():
    var_0 = '<div><p>Hello</p> World</div>'



# Parsed testcases at query #9
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
    var_3 = [var_0, var_1, var_1, var_2]
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
    var_2 = 'b'
    var_3 = [var_0, var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [None, 'a', 'b'])
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



# Parsed testcases at query #10
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = True
    var_4 = module_0.extract_text(var_1, var_2, var_2, var_3)
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.extract_text(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 6/14 statements.
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
    var_1 = 'World'
    var_2 = 'div'
    var_3 = None
    var_4 = False
    var_5 = True

def test_case_0():
    var_0 = False



# Parsed testcases at query #13
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '  text  '
    var_2 = True
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = module_0.extract_text(var_3, squash_space=var_4)
    assert var_5 == '\n  text  \n'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_squash_space_predicate_evaluates_to_true. Retrieved 3/4 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = '<div>Hello World</div>'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_custom_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_without_squashing_space. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_preformatted_content. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'

def test_case_0():
    var_0 = '<div>Hello <strong>World</strong></div>'

def test_case_0():
    var_0 = '<div><h1>Title</h1><p>Content</p></div>'

def test_case_0():
    var_0 = '<div>  Hello   \n  World  </div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div><h1>Title</h1><p>Content</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>  Hello   \n  World  </div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><pre>  Hello   \n  World  </pre></div>'

def test_case_0():
    var_0 = '<div><p>Hello <strong>World</strong></p><ul><li>Item</li></ul></div>'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 2/9 statements.


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
    var_0 = None
    var_1 = False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_squash_space_false. Retrieved 3/4 statements.


def test_case_0():
    var_0 = '\n'
    var_1 = '\n'
    var_2 = False
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_callable_dom_tag. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'MockDOM'
    var_1 = ()
    var_2 = 'tag'



# Parsed testcases at query #20
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = '  text  '
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = module_0.extract_text(var_3, squash_space=var_1)
    assert var_4 == '\n\ntext'



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
    var_6 = bool(not False)
    assert var_6 is True



# Parsed testcases at query #22
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



# Parsed testcases at query #23
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
    var_14 = var_13.tag
    var_15 = callable(var_14)
    var_16 = bool(not var_15)
    assert var_16 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_extract_text_with_block_and_sep_symbols. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_nested_elements. Retrieved 8/16 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_empty_dom. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_multiple_children. Retrieved 11/22 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = '\n'
    var_4 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None
    var_6 = '\n'
    var_7 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'First paragraph'
    var_2 = None
    var_3 = '\n'
    var_4 = True

def test_case_0():
    var_0 = 'strong'
    var_1 = 'Important'
    var_2 = None
    var_3 = '\n'
    var_4 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello  '
    var_2 = None
    var_3 = '\n'
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = '\n'
    var_4 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'First'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'Second'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = None
    var_9 = '\n'
    var_10 = True



# Parsed testcases at query #25
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = '\n'
    var_3 = False
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_single_text_node. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_nested_tags. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_with_none_tag. Retrieved 3/8 statements.


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
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = '|'

def test_case_0():
    var_0 = 'br'
    var_1 = None
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
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None

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
    var_0 = None
    var_1 = 'Hello'
    var_2 = None



# Parsed testcases at query #27
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '<div>Hello World</div>'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'Hello World'



# Parsed testcases at query #28
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '<div>Hello World</div>'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'Hello World'



# Parsed testcases at query #29
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '<div>Hello World</div>'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'Hello World'



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_callable_tag.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_callable_tag_returns_empty_string. Retrieved 1/4 statements.


def test_case_0():
    var_0 = lambda : None



# Parsed testcases at query #32
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
    var_14 = var_13.tag
    var_15 = callable(var_14)
    var_16 = bool(not var_15)
    assert var_16 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_child. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
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
    var_0 = None



# Parsed testcases at query #34
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.extract_text(var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True



# Parsed testcases at query #35
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = '\n'
    var_2 = '\n'
    var_3 = True
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_single_text_node. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 3/8 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 5/10 statements.
# Partially parsed test_extract_text_nested_elements. Retrieved 6/14 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_with_none_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 6/14 statements.


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
    var_1 = '  Hello   World  '
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = '|'
    var_4 = ';'

def test_case_0():
    var_0 = 'span'
    var_1 = 'nested'
    var_2 = ' text'
    var_3 = 'div'
    var_4 = 'Hello '
    var_5 = '!'

def test_case_0():
    var_0 = 'span'
    var_1 = 'First'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'Second'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'child'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'start'
    var_5 = ' end'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_squash_space_predicate. Retrieved 5/6 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = module_0.extract_text(var_3, squash_space=var_1)



# Parsed testcases at query #38
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'test'
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = module_0.extract_text(var_3, squash_space=var_4)
    assert var_5 == '\n\ntest'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_extract_text_array_with_callable_tag. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'obj'
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'getchildren'
    var_4 = None
    var_5 = []
    var_6 = lambda : var_5



# Parsed testcases at query #40
#--------------------------

# Failed to parse test_callable_tag_returns_empty_string.




# Parsed testcases at query #41
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = '\n'
    var_2 = '\n'
    var_3 = True
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_nested_tags. Retrieved 7/15 statements.
# Partially parsed test_extract_text_with_none_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_multiple_children. Retrieved 10/21 statements.
# Partially parsed test_extract_text_with_block_and_separator. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_whitespace_only. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 7/15 statements.


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
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = '\n\n'

def test_case_0():
    var_0 = 'hr'
    var_1 = None
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
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None
    var_6 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None

def test_case_0():
    var_0 = 'hr'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'World'
    var_5 = '!'
    var_6 = 'div'
    var_7 = None
    var_8 = None
    var_9 = True

def test_case_0():
    var_0 = 'hr'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = '|'
    var_4 = '-'

def test_case_0():
    var_0 = 'div'
    var_1 = '   \n  \t  '
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = '!'
    var_6 = True



# Parsed testcases at query #43
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'Hello'
    var_2 = True
    var_3 = 'World'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = False
    var_6 = module_0.extract_text(var_4, squash_space=var_5)
    assert var_6 == '\nHello\nWorld'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_text_only. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_nested_text. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_block_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_without_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_with_artificial_newlines. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_artificial_newlines_squashed. Retrieved 7/15 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 6/14 statements.


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
    var_3 = '\n'

def test_case_0():
    var_0 = 'p'
    var_1 = None
    var_2 = None
    var_3 = '\n'

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
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'World'
    var_5 = '!'
    var_6 = 'div'
    var_7 = None
    var_8 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None
    var_6 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = '!'



# Parsed testcases at query #45
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '\n'
    var_2 = '\n'
    var_3 = False
    var_4 = module_0.extract_text(var_0, var_1, var_2, var_3)



# Parsed testcases at query #46
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = '\n'
    var_2 = True
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_callable_tag_returns_empty_string. Retrieved 1/4 statements.


def test_case_0():
    var_0 = lambda : None



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_extract_text_with_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_complex_structure. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_callable_tag. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Hello World'
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'nested'
    var_2 = ' tail'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = 'World'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Block1'
    var_2 = None
    var_3 = '\n\n'

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = '|'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'span'
    var_1 = 'First'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'Second'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = None

def test_case_0():
    var_0 = 'strong'
    var_1 = 'bold'
    var_2 = ' text'
    var_3 = 'p'
    var_4 = 'Normal'
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'inner'
    var_2 = ' text'
    var_3 = 'div'
    var_4 = 'Start'
    var_5 = 'End'

def test_case_0():
    var_0 = 'Callable'
    var_1 = None



# Parsed testcases at query #49
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '  text  '
    var_2 = True
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = module_0.extract_text(var_3, squash_space=var_4)
    assert var_5 == '\n  text  \n'



# Parsed testcases at query #50
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



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_extract_text_array_with_callable_tag. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'MockElement'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = 'getchildren'
    var_5 = None
    var_6 = []
    var_7 = lambda : var_6



# Parsed testcases at query #52
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #53
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '<p>Hello</p><p>World</p>'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'Hello\nWorld'



# Parsed testcases at query #54
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_dom_tag_not_callable. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'div'



# Parsed testcases at query #56
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



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 3/6 statements.
# Partially parsed test_extract_text_without_squashing. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_preformatted_content. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'

def test_case_0():
    var_0 = '<div>Hello <strong>World</strong></div>'

def test_case_0():
    var_0 = '<div><h1>Title</h1><p>Content</p></div>'

def test_case_0():
    var_0 = '<div>Hello   \n   World</div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'
    var_1 = '|'
    var_2 = ';'

def test_case_0():
    var_0 = '<div>Hello   \n   World</div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><pre>Hello   \n   World</pre></div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><ul><li>Item 1</li><li>Item 2</li></ul></div>'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_callable_tag. Retrieved 1/8 statements.
# Partially parsed test_extract_text_array_complex_structure. Retrieved 8/19 statements.


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
    var_0 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = ' '
    var_3 = 'div'
    var_4 = 'Universe'
    var_5 = 'div'
    var_6 = 'Hello'
    var_7 = False



# Parsed testcases at query #60
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = '  text  '
    var_2 = True
    var_3 = '  more  '
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = False
    var_6 = module_0.extract_text(var_4, squash_space=var_5)
    assert var_6 == '\n  text  \n  more  '



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_extract_text_with_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_custom_symbols. Retrieved 3/6 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_preformatted_content. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_only_whitespace. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_mixed_content. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'

def test_case_0():
    var_0 = '<div>Hello <strong>World</strong></div>'

def test_case_0():
    var_0 = '<div><h1>Title</h1><p>Content</p></div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'
    var_1 = '|'
    var_2 = '||'

def test_case_0():
    var_0 = '<div>  Hello   \n  World  </div>'
    var_1 = True

def test_case_0():
    var_0 = '<div><pre>  Hello   \n  World  </pre></div>'
    var_1 = False

def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>   \n  \t  </div>'
    var_1 = True

def test_case_0():
    var_0 = '<div><p>Hello <em>World</em></p><ul><li>Item</li></ul></div>'



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
    var_0 = '  hello  '
    var_1 = '  world  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
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
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._merge_original_parts(var_4)
    var_6 = bool(var_5 == ['hello world', 1, 2])
    assert var_6 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  '
    var_1 = '   '
    var_2 = '\n'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [])
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
    var_0 = '  \n  '
    var_1 = '  text  '
    var_2 = '  \t  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['text'])
    assert var_5 is True



# Parsed testcases at query #3
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
    var_3 = [var_0, var_1, var_1, var_2]
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
    var_2 = 'b'
    var_3 = [var_0, var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [None, 'a', 'b'])
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 6/14 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 8/16 statements.
# Partially parsed test_extract_text_nested_elements. Retrieved 9/20 statements.
# Partially parsed test_extract_text_multiple_children. Retrieved 9/20 statements.
# Partially parsed test_extract_text_with_none_text. Retrieved 3/8 statements.


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
    var_2 = ' Tail'
    var_3 = 'div'
    var_4 = 'Start '
    var_5 = None

def test_case_0():
    var_0 = 'p'
    var_1 = 'Paragraph'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = 'div'
    var_4 = 'Before'
    var_5 = 'After'

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'p'
    var_1 = 'Paragraph'
    var_2 = None
    var_3 = 'div'
    var_4 = None
    var_5 = None
    var_6 = '|'
    var_7 = '-'

def test_case_0():
    var_0 = 'b'
    var_1 = 'Bold'
    var_2 = ' text'
    var_3 = 'p'
    var_4 = 'Start '
    var_5 = ' end'
    var_6 = 'div'
    var_7 = 'Outer '
    var_8 = ' outer'

def test_case_0():
    var_0 = 'span'
    var_1 = 'First'
    var_2 = ' '
    var_3 = 'span'
    var_4 = 'Second'
    var_5 = None
    var_6 = 'div'
    var_7 = None
    var_8 = None

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None



# Parsed testcases at query #7
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = '\n'
    var_2 = True
    var_3 = module_0.extract_text(var_0, var_1, var_1, var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.extract_text(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_extract_text_with_block_and_sep_symbols. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_nested_elements. Retrieved 8/16 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_squash_space. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_empty_dom. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_callable_tag. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = None
    var_3 = '\n'

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = 'World'
    var_3 = 'span'
    var_4 = 'Nested'
    var_5 = 'Text'
    var_6 = '\n'
    var_7 = False

def test_case_0():
    var_0 = 'hr'
    var_1 = None
    var_2 = None
    var_3 = '\n'

def test_case_0():
    var_0 = 'span'
    var_1 = 'Inline'
    var_2 = 'Text'
    var_3 = '\n'
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello  '
    var_2 = '  World  '
    var_3 = '\n'
    var_4 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = '\n'

def test_case_0():
    var_0 = 'Callable'
    var_1 = None
    var_2 = '\n'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_squash_space_predicate. Retrieved 3/4 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = '<div>Hello World</div>'
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 2/5 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 2/5 statements.
# Partially parsed test_extract_text_nested_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_tail. Retrieved 1/4 statements.
# Partially parsed test_extract_text_preformatted. Retrieved 2/5 statements.
# Partially parsed test_extract_text_multiple_blocks. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 3/6 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div>Hello<p>World</p></div>'
    var_1 = '\n'

def test_case_0():
    var_0 = '<div>Hello<br/>World</div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>  Hello   World  </div>'
    var_1 = True

def test_case_0():
    var_0 = '<div>  Hello   World  </div>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p>Hello<span>World</span></p></div>'

def test_case_0():
    var_0 = '<div><p>Hello</p>World</div>'

def test_case_0():
    var_0 = '<pre>  Hello   World  </pre>'
    var_1 = False

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'

def test_case_0():
    var_0 = '<div>Hello <strong>World</strong></div>'

def test_case_0():
    var_0 = '<div>Hello<br/>World</div>'

def test_case_0():
    var_0 = '<div>Hello<p>World</p></div>'
    var_1 = '|'
    var_2 = '-'



# Parsed testcases at query #4
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.extract_text(var_3)
    assert var_4 == '\n\ntext'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_squash_space_predicate. Retrieved 5/6 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = module_0.extract_text(var_3, squash_space=var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_sep_symbol. Retrieved 4/9 statements.
# Partially parsed test_extract_text_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_no_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_nested_blocks. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_inline_tags. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_preformatted. Retrieved 3/8 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 9/20 statements.


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
    var_4 = 'Parent'
    var_5 = None

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = 'Block'
    var_2 = None
    var_3 = '|'

def test_case_0():
    var_0 = 'br'
    var_1 = None
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
    var_0 = 'div'
    var_1 = 'Child'
    var_2 = 'Tail'
    var_3 = 'div'
    var_4 = 'Parent'
    var_5 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Inline'
    var_2 = 'Tail'
    var_3 = 'div'
    var_4 = 'Start'
    var_5 = 'End'

def test_case_0():
    var_0 = 'pre'
    var_1 = '  Preformatted  '
    var_2 = None

def test_case_0():
    var_0 = 'span'
    var_1 = 'Text1'
    var_2 = 'Tail1'
    var_3 = 'div'
    var_4 = 'Text2'
    var_5 = 'Tail2'
    var_6 = 'div'
    var_7 = 'Start'
    var_8 = 'End'



# Parsed testcases at query #7
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
    var_1 = 123
    var_2 = 'world'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['hello', 123, 'world'])
    assert var_5 is True

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
    var_0 = '  hello  '
    var_1 = 123
    var_2 = '  world  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == ['hello', 123, 'world'])
    assert var_5 is True

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
    var_0 = ''
    var_1 = [var_0, var_0]
    var_2 = module_0._merge_original_parts(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 123
    var_1 = 456
    var_2 = 'hello'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    var_5 = bool(var_4 == [123, 456, 'hello'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello  '
    var_1 = '  world'
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello'
    var_1 = 'world  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello\n'
    var_1 = '\tworld'
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = '  hello  \n'
    var_1 = '\t  world  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'hello!'
    var_1 = 'world?'
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['hello! world?'])
    assert var_4 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = 'héllo'
    var_1 = 'wörld'
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    var_4 = bool(var_3 == ['héllo wörld'])
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.extract_text(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_extract_text_with_block_and_sep_symbols. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 8/16 statements.
# Partially parsed test_extract_text_with_squash_space_false. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_empty_dom. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_callable_tag. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = 'World'
    var_3 = '\n'
    var_4 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = 'World'
    var_3 = '\n'
    var_4 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = '\n'
    var_4 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'nested'
    var_2 = 'text'
    var_3 = 'div'
    var_4 = 'Hello'
    var_5 = 'World'
    var_6 = '\n'
    var_7 = True

def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = 'World'
    var_3 = '\n'
    var_4 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None
    var_3 = '\n'
    var_4 = True

def test_case_0():
    var_0 = 'Hello'
    var_1 = 'World'
    var_2 = '\n'
    var_3 = True



# Parsed testcases at query #11
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
    var_0 = 123
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    var_3 = bool(var_2 == [123])
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
    var_2 = 2
    var_3 = 'b'
    var_4 = 3
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._strip_artifical_nl(var_5)
    var_7 = bool(var_6 == [1, 'a', 2, 'b', 3])
    assert var_7 is True

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
    var_1 = 'a'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    var_5 = bool(var_4 == ['a'])
    assert var_5 is True

import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'start'
    var_3 = 2
    var_4 = 'middle'
    var_5 = 3
    var_6 = 'end'
    var_7 = 4
    var_8 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_0]
    var_9 = module_0._strip_artifical_nl(var_8)
    var_10 = bool(var_9 == ['start', 2, 'middle', 3, 'end'])
    assert var_10 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_separator. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_block_element. Retrieved 3/8 statements.
# Partially parsed test_extract_text_nested_elements. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_without_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_custom_symbols. Retrieved 5/10 statements.
# Partially parsed test_extract_text_with_multiple_blocks. Retrieved 9/20 statements.


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
    var_2 = ' Tail'
    var_3 = 'div'
    var_4 = 'Start '
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
    var_0 = 'strong'
    var_1 = 'bold'
    var_2 = ' text'
    var_3 = 'div'
    var_4 = 'Start '
    var_5 = None

def test_case_0():
    var_0 = 'div'
    var_1 = '  Multiple   spaces  '
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  Multiple   spaces  '
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = 'Text'
    var_2 = None
    var_3 = '|'
    var_4 = ';'

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



# Parsed testcases at query #13
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
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_0, var_1, var_0, var_0, var_0, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    var_5 = bool(var_4 == [None, 1, None, 2])
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_squash_space_predicate. Retrieved 6/7 statements.


import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'text'
    var_3 = 'more text'
    var_4 = [var_0, var_1, var_2, var_0, var_3]
    var_5 = module_0.extract_text(var_4, squash_space=var_1)



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = True
    assert var_1 is True



# Parsed testcases at query #16
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = 'text'
    var_2 = True
    var_3 = 'more text'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = False
    var_6 = module_0.extract_text(var_4, squash_space=var_5)
    assert var_6 == '\ntext\nmore text'



# Parsed testcases at query #17
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '  test  '
    var_1 = True
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == 'test'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
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

def test_case_0():
    var_0 = lambda : 'div'
    var_1 = None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_block_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_inline_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_separator_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_whitespace_squashing. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_custom_block_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_custom_sep_symbol. Retrieved 2/5 statements.
# Partially parsed test_extract_text_without_squash_space. Retrieved 2/5 statements.
# Partially parsed test_extract_text_with_nested_elements. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_preformatted_content. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_leading_trailing_whitespace. Retrieved 1/4 statements.
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
    var_0 = '<div>Hello   World</div>'

def test_case_0():
    var_0 = '<div>Hello<p>World</p></div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>Hello<br/>World</div>'
    var_1 = '|'

def test_case_0():
    var_0 = '<div>Hello   World</div>'
    var_1 = False

def test_case_0():
    var_0 = '<div>Hello<div>World<span>!</span></div></div>'

def test_case_0():
    var_0 = '<div>Hello<pre>  World  </pre></div>'

def test_case_0():
    var_0 = '<div>Hello<br/><br/>World</div>'

def test_case_0():
    var_0 = '<div>  Hello World  </div>'

def test_case_0():
    var_0 = '<div>Hello<span>World</span><p>!</p></div>'



# Parsed testcases at query #20
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = '<p>Hello World</p>'
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    assert var_2 == '\nHello World\n'



# Parsed testcases at query #21
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = True
    var_4 = module_0.extract_text(var_1, var_2, var_2, var_3)
    assert var_4 == '\n'



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = 'some_dom'
    var_1 = True
    assert var_1 is True



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_callable_tag_returns_empty_string.




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




import pyquery.text as module_0

def test_case_0():
    var_0 = '<div>test</div>'
    var_1 = False
    var_2 = module_0.extract_text(var_0, squash_space=var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_callable_tag_returns_empty_string.




# Parsed testcases at query #28
#--------------------------




import pyquery.text as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.extract_text(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_extract_text_with_empty_dom. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_simple_text. Retrieved 1/4 statements.
# Partially parsed test_extract_text_with_nested_tags. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '<div></div>'

def test_case_0():
    var_0 = '<div>Hello World</div>'

def test_case_0():
    var_0 = '<div><p>Hello</p><p>World</p></div>'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_callable_tag_returns_empty_string. Retrieved 3/6 statements.


def test_case_0():
    var_0 = lambda : None
    var_1 = None
    var_2 = lambda : []



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_dom_tag_is_callable.




# Parsed testcases at query #32
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



# Parsed testcases at query #33
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
    var_14 = var_13.tag
    var_15 = callable(var_14)
    var_16 = bool(not var_15)
    assert var_16 is True



# Parsed testcases at query #34
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



# Parsed testcases at query #35
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
    var_13 = var_12()
    var_14 = var_13.tag
    var_15 = callable(var_14)
    var_16 = bool(not var_15)
    assert var_16 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_strip_artifical_nl_predicate. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = lambda : var_2
    var_4 = False
    var_5 = True



# Parsed testcases at query #37
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
    var_6 = 'test'
    var_7 = []
    var_8 = lambda : var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.extract_text_array(var_13)
    var_15 = bool(var_14 == ['test'])
    assert var_15 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_with_separator. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 5/10 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 5/10 statements.
# Failed to parse test_extract_text_array_callable_tag.


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



# Parsed testcases at query #39
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
    var_14 = False
    var_15 = module_1.extract_text_array(var_13, var_14, var_14)
    var_16 = bool(var_15 == [None, None])
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
    var_5 = 'br'
    var_6 = None
    var_7 = []
    var_8 = lambda : var_7
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.extract_text_array(var_13)
    var_15 = bool(var_14 == [True])
    assert var_15 is True



# Parsed testcases at query #41
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
    var_7 = []
    var_8 = lambda : var_7
    var_9 = None
    var_10 = {var_2: var_6, var_3: var_3, var_4: var_8, var_5: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = False
    var_16 = True
    var_17 = module_1.extract_text_array(var_14, var_15, var_16)
    var_18 = bool(var_17 == ['text'])
    assert var_18 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = lambda : var_2
    var_4 = True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_child_has_text. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'p'
    var_2 = 'some text'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_squash_and_strip_artificial_nl. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = None
    var_5 = True



# Parsed testcases at query #45
#--------------------------

# Failed to parse test_predicate_at_line_12_evaluates_to_false.




# Parsed testcases at query #46
#--------------------------

# Partially parsed test_strip_artifical_nl_predicate_false. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = lambda : var_2



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_strip_artifical_nl_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = lambda : var_2



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_non_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_complex_case. Retrieved 9/20 statements.


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
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = '!'
    var_5 = None
    var_6 = 'div'
    var_7 = 'Hello'
    var_8 = None



# Parsed testcases at query #49
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
    var_6 = 'span'
    var_7 = None
    var_8 = []
    var_9 = lambda : var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.extract_text_array(var_14)
    var_16 = bool(var_15 == [])
    assert var_16 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_non_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_complex_case. Retrieved 9/20 statements.
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
    var_1 = 'Hello'
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = None
    var_3 = 'div'
    var_4 = '!'
    var_5 = None
    var_6 = 'div'
    var_7 = 'Hello'
    var_8 = 'End'

def test_case_0():
    var_0 = None
    var_1 = None



# Parsed testcases at query #51
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
    var_17 = var_16[-1]
    assert var_17 is None



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_child_iteration. Retrieved 2/17 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'p'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = lambda : var_2
    var_4 = False



# Parsed testcases at query #54
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
    var_14 = 'span'
    var_15 = 'a'
    var_16 = 'strong'
    var_17 = {var_14, var_15, var_16}
    var_18 = 'br'
    var_19 = 'hr'
    var_20 = {var_18, var_19}
    var_21 = bool(var_13.tag not in var_17 and var_13.tag not in var_20)
    assert var_21 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_false. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'inline_tag'
    var_1 = {var_0}
    var_2 = 'separator_tag'
    var_3 = {var_2}



# Parsed testcases at query #56
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
    var_17 = len(var_16)
    var_18 = 2
    var_19 = var_17 == var_18
    var_20 = bool(var_19 and var_16[0] is None and (var_16[1] is None))
    assert var_20 is True



# Parsed testcases at query #57
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
    var_6 = 'span'
    var_7 = None
    var_8 = []
    var_9 = lambda : var_8
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_7}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.extract_text_array(var_14)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_strip_artifical_nl_is_false. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'text'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = False



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 12/24 statements.


def test_case_0():
    var_0 = 'inline1'
    var_1 = 'inline2'
    var_2 = {var_0, var_1}
    var_3 = 'sep1'
    var_4 = 'sep2'
    var_5 = {var_3, var_4}
    var_6 = 'block'
    var_7 = 'text'
    var_8 = 'tail'
    var_9 = 'child'
    var_10 = 'child_text'
    var_11 = 'child_tail'



# Parsed testcases at query #60
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
    var_16 = 'a'
    var_17 = 'em'
    var_18 = {var_15, var_16, var_17}
    var_19 = 'br'
    var_20 = 'hr'
    var_21 = {var_19, var_20}
    var_22 = False
    var_23 = module_1.extract_text_array(var_14, var_22, var_22)
    var_24 = bool(var_23 == [None, None])
    assert var_24 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_squash_artifical_nl_is_true. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'Hello'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = None
    var_5 = True
    var_6 = False



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_squash_and_strip_artificial_nl. Retrieved 14/28 statements.


def test_case_0():
    var_0 = 'span'
    var_1 = 'a'
    var_2 = 'em'
    var_3 = 'strong'
    var_4 = {var_0, var_1, var_2, var_3}
    var_5 = 'br'
    var_6 = 'p'
    var_7 = 'div'
    var_8 = {var_5, var_6, var_7}
    var_9 = 'Hello'
    var_10 = 'World'
    var_11 = 'Test'
    var_12 = 'Tail'
    var_13 = True



# Parsed testcases at query #63
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
    var_6 = 'inline'
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



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_dom_text_not_none. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = 'some text'
    var_2 = []
    var_3 = lambda : var_2
    var_4 = 'some text'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_squash_artifical_nl_false_when_tag_not_in_inline_or_separators. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = []
    var_3 = lambda : var_2
    var_4 = False



# Parsed testcases at query #66
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
    var_15 = 'span'
    var_16 = 'a'
    var_17 = 'strong'
    var_18 = {var_15, var_16, var_17}
    var_19 = 'br'
    var_20 = 'hr'
    var_21 = 'p'
    var_22 = {var_19, var_20, var_21}
    var_23 = bool(not (var_14.tag not in var_18 and var_14.tag not in var_22))
    assert var_23 is True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_text_only. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_inline_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_separator_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_non_inline_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 7/15 statements.
# Partially parsed test_extract_text_array_with_nested_children. Retrieved 10/21 statements.
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
    var_0 = 'span'
    var_1 = 'Hello'
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = None
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
    var_0 = 'b'
    var_1 = 'World'
    var_2 = None
    var_3 = 'span'
    var_4 = None
    var_5 = '!'
    var_6 = 'div'
    var_7 = 'Hello'
    var_8 = None
    var_9 = True

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
    var_3 = True



# Parsed testcases at query #68
#--------------------------

# Failed to parse test_predicate_at_line_12_evaluates_to_false.




# Parsed testcases at query #69
#--------------------------

# Partially parsed test_extract_text_array_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_text. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_array_squash_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_strip_artifical_nl. Retrieved 4/9 statements.
# Partially parsed test_extract_text_array_separator_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_array_inline_tag. Retrieved 3/8 statements.
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

def test_case_0():
    var_0 = lambda : None
    var_1 = None



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_extract_text_empty_dom. Retrieved 3/8 statements.
# Partially parsed test_extract_text_single_text_node. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_children. Retrieved 6/14 statements.
# Partially parsed test_extract_text_with_block_separator. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_inline_tag. Retrieved 3/8 statements.
# Partially parsed test_extract_text_with_squash_space. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_multiple_whitespaces. Retrieved 4/9 statements.
# Partially parsed test_extract_text_with_separator_tag. Retrieved 4/9 statements.
# Partially parsed test_extract_text_complex_structure. Retrieved 10/21 statements.
# Partially parsed test_extract_text_with_preformatted_content. Retrieved 4/9 statements.
# Partially parsed test_extract_text_strip_artificial_nl. Retrieved 3/8 statements.


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
    var_1 = 'First paragraph'
    var_2 = None
    var_3 = '\n'

def test_case_0():
    var_0 = 'strong'
    var_1 = 'Important'
    var_2 = None

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   World  '
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'div'
    var_1 = '  Hello   \n   World  '
    var_2 = None
    var_3 = True

def test_case_0():
    var_0 = 'br'
    var_1 = None
    var_2 = None
    var_3 = '\n'

def test_case_0():
    var_0 = 'span'
    var_1 = 'World'
    var_2 = '!'
    var_3 = 'div'
    var_4 = 'New line'
    var_5 = None
    var_6 = 'div'
    var_7 = 'Hello'
    var_8 = 'End'
    var_9 = '\n'

def test_case_0():
    var_0 = 'pre'
    var_1 = '  Code  \n  Block  '
    var_2 = None
    var_3 = False

def test_case_0():
    var_0 = 'div'
    var_1 = None
    var_2 = None



