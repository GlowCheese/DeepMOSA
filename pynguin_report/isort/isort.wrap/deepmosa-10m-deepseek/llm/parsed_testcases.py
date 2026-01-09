####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_line_no_wrapping_needed. Retrieved 7/10 statements.
# Partially parsed test_line_wrapping_with_import_split. Retrieved 7/10 statements.
# Partially parsed test_line_wrapping_with_as_split. Retrieved 8/11 statements.
# Partially parsed test_line_wrapping_with_dot_split. Retrieved 7/10 statements.
# Partially parsed test_line_wrapping_with_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrapping_with_noqa_comment. Retrieved 8/11 statements.
# Partially parsed test_line_wrapping_noqa_mode. Retrieved 7/10 statements.
# Partially parsed test_line_wrapping_noqa_mode_with_existing_noqa. Retrieved 7/10 statements.
# Partially parsed test_line_wrapping_without_parentheses. Retrieved 7/10 statements.
# Partially parsed test_line_wrapping_vertical_hanging_indent. Retrieved 7/10 statements.
# Partially parsed test_line_wrapping_vertical_grid_grouped. Retrieved 7/10 statements.
# Partially parsed test_line_wrapping_with_trailing_comma_and_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrapping_with_wrap_length. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 80
    var_1 = None
    var_2 = '    '
    var_3 = False
    var_4 = '# '
    var_5 = 'import os'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = '# '
    var_5 = 'from very_long_module_name import very_long_function_name'
    var_6 = '\n'
    var_7 = 'from very_long_module_name import('
    var_8 = 'very_long_function_name,'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = False
    var_5 = '# '
    var_6 = 'import very_long_module_name as vlm'
    var_7 = '\n'
    var_8 = 'import very_long_module_name as'
    var_9 = 'vlm'

def test_case_0():
    var_0 = 25
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = '# '
    var_5 = 'very_long_module_name.very_long_submodule'
    var_6 = '\n'
    var_7 = 'very_long_module_name.('
    var_8 = 'very_long_submodule,'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = '# '
    var_5 = 'from module import function  # some comment'
    var_6 = '\n'
    var_7 = 'from module import('
    var_8 = 'function,'
    var_9 = '# some comment'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = False
    var_5 = '# '
    var_6 = 'from module import function  # noqa'
    var_7 = '\n'
    var_8 = 'from module import(# noqa'
    var_9 = 'function)'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = False
    var_4 = '# '
    var_5 = 'from module import function'
    var_6 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = False
    var_4 = '# '
    var_5 = 'from module import function  # NOQA'
    var_6 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = False
    var_4 = '# '
    var_5 = 'from module import function'
    var_6 = '\n'
    var_7 = 'from module import\\'
    var_8 = 'function'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = '# '
    var_5 = 'from module import function'
    var_6 = '\n'
    var_7 = 'from module import('
    var_8 = 'function,'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = '# '
    var_5 = 'from module import function'
    var_6 = '\n'
    var_7 = 'from module import('
    var_8 = 'function,'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = '# '
    var_5 = 'from module import function  # comment'
    var_6 = '\n'

def test_case_0():
    var_0 = 80
    var_1 = 30
    var_2 = '    '
    var_3 = True
    var_4 = '# '
    var_5 = 'from very_long_module_name import very_long_function_name'
    var_6 = '\n'
    var_7 = 'from very_long_module_name import('
    var_8 = 'very_long_function_name,'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import_split. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_as_split. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_dot_split. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 6/9 statements.
# Partially parsed test_line_noqa_mode_with_long_line. Retrieved 4/7 statements.
# Partially parsed test_line_noqa_mode_with_existing_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_trailing_comma_and_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 5/8 statements.
# Partially parsed test_line_short_line_with_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_multiple_splitters. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 80
    var_1 = 'import os'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'from very_long_module_name import very_long_function_name'
    var_4 = '\n'
    var_5 = 'from very_long_module_name import('
    var_6 = 'very_long_function_name,'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'import very_long_module_name as very_long_alias'
    var_4 = '\n'
    var_5 = 'import very_long_module_name as'
    var_6 = 'very_long_alias'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'very_long_module_name.very_long_submodule.very_long_function'
    var_4 = '\n'
    var_5 = 'very_long_module_name.('
    var_6 = 'very_long_submodule.very_long_function'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # some comment'
    var_5 = '\n'
    var_6 = 'from module import('
    var_7 = 'something,  # some comment'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # noqa'
    var_5 = '\n'
    var_6 = 'from module import(  # noqa'
    var_7 = 'something,'

def test_case_0():
    var_0 = 20
    var_1 = '  # '
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = '  # '
    var_2 = 'from module import something  # NOQA'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = '    '
    var_3 = 'from module import something'
    var_4 = '\n'
    var_5 = 'from module import\\'
    var_6 = 'something'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # comment'
    var_5 = '\n'
    var_6 = 'something,  # comment'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'from module import something'
    var_4 = '\n'
    var_5 = 'from module import('
    var_6 = 'something,'

def test_case_0():
    var_0 = 80
    var_1 = '  # '
    var_2 = 'import os  # comment'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'cimport very_long_module_name'
    var_4 = '\n'
    var_5 = 'cimport very_long_module_name'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = 'from very_long_module_name import very_long_function_name as alias'
    var_4 = '\n'
    var_5 = 'from very_long_module_name import('
    var_6 = 'very_long_function_name as alias'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_import_statement_single_line. Retrieved 5/8 statements.
# Partially parsed test_import_statement_multi_line_grid. Retrieved 8/13 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 8/11 statements.
# Partially parsed test_import_statement_include_trailing_comma. Retrieved 8/12 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 11/20 statements.
# Partially parsed test_import_statement_custom_line_separator. Retrieved 7/10 statements.
# Partially parsed test_import_statement_remove_comments. Retrieved 9/12 statements.
# Partially parsed test_import_statement_wrap_length_overrides_line_length. Retrieved 9/14 statements.
# Partially parsed test_import_statement_explode_overrides_config. Retrieved 8/12 statements.


import isort.wrap as module_0


def test_case_0():
    var_0 = 'from module'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = [var_1, var_2]
    var_4 = True
    var_5 = module_0.import_statement(var_0, var_3, explode=var_4)
    var_6 = 'from module import (\n    item1,\n    item2,\n)'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 100
    var_1 = 'from module'
    var_2 = 'item1'
    var_3 = 'item2'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 20
    var_1 = 'from module'
    var_2 = 'item1'
    var_3 = 'item2'
    var_4 = 'item3'
    var_5 = 'item4'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = 'from module'
    var_2 = 'item1'
    var_3 = 'item2'
    var_4 = [var_2, var_3]
    var_5 = 'comment1'
    var_6 = 'comment2'
    var_7 = [var_5, var_6]
    var_8 = '# comment1'
    var_9 = '# comment2'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = 'item3'
    var_6 = [var_3, var_4, var_5]
    var_7 = ',\n)'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = 'item3'
    var_6 = 'item4'
    var_7 = 'item5'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = '\n'
    var_10 = -1

def test_case_0():
    var_0 = 20
    var_1 = 'from module'
    var_2 = 'item1'
    var_3 = 'item2'
    var_4 = 'item3'
    var_5 = [var_2, var_3, var_4]
    var_6 = '\r\n'
    var_7 = '\r\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = [var_3, var_4]
    var_6 = 'comment1'
    var_7 = 'comment2'
    var_8 = [var_6, var_7]
    var_9 = '# comment1'
    var_10 = '# comment2'

def test_case_0():
    var_0 = 100
    var_1 = 20
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = 'item3'
    var_6 = 'item4'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = '\n'

def test_case_0():
    var_0 = 100
    var_1 = False
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = ',\n)'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_71_true. Retrieved 4/6 statements.


import isort.settings as module_0
import isort.wrap as module_1


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'verylonglinewithoutnoqa'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    assert var_4 == 'verylonglinewithoutnoqa# NOQA'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_mode_noqa. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_mode_noqa_with_existing_noqa. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_vertical_hanging_indent. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_trailing_comma_and_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_without_trailing_comma_no_comment. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_content_empty_after_split. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 80
    var_1 = None
    var_2 = False
    var_3 = '  #'
    var_4 = '    '
    var_5 = 'import os'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = '  #'
    var_4 = '    '
    var_5 = 'from very_long_module_name import very_long_function_name'
    var_6 = '\n'
    var_7 = 'from very_long_module_name import('
    var_8 = 'very_long_function_name,'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = True
    var_3 = False
    var_4 = '  #'
    var_5 = '    '
    var_6 = 'very_long_module_name.very_long_function_name'
    var_7 = '\n'
    var_8 = 'very_long_module_name.('
    var_9 = 'very_long_function_name)'

def test_case_0():
    var_0 = 25
    var_1 = None
    var_2 = True
    var_3 = False
    var_4 = '  #'
    var_5 = '    '
    var_6 = 'import very_long_module_name as vlm'
    var_7 = '\n'
    var_8 = 'import very_long_module_name as vlm'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = True
    var_3 = '  #'
    var_4 = '    '
    var_5 = 'from module import function  # some comment'
    var_6 = '\n'
    var_7 = 'from module import('
    var_8 = 'function,  # some comment)'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = True
    var_3 = False
    var_4 = '  #'
    var_5 = '    '
    var_6 = 'from module import function  # noqa'
    var_7 = '\n'
    var_8 = 'from module import(  # noqa'
    var_9 = 'function)'

def test_case_0():
    var_0 = 10
    var_1 = None
    var_2 = False
    var_3 = '  #'
    var_4 = '    '
    var_5 = 'import os'
    var_6 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = None
    var_2 = False
    var_3 = '  #'
    var_4 = '    '
    var_5 = 'import os  # NOQA'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = False
    var_3 = '  #'
    var_4 = '    '
    var_5 = 'from module import function'
    var_6 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = True
    var_3 = '  #'
    var_4 = '    '
    var_5 = 'from module import function1, function2, function3'
    var_6 = '\n'
    var_7 = 'from module import('
    var_8 = 'function1,'
    var_9 = 'function2,'
    var_10 = 'function3,'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = True
    var_3 = '  #'
    var_4 = '    '
    var_5 = 'from module import function1, function2, function3'
    var_6 = '\n'
    var_7 = 'from module import('
    var_8 = 'function1,'
    var_9 = 'function2,'
    var_10 = 'function3,'

def test_case_0():
    var_0 = 35
    var_1 = None
    var_2 = True
    var_3 = '  #'
    var_4 = '    '
    var_5 = 'from module import function  # comment'
    var_6 = '\n'
    var_7 = 'from module import('
    var_8 = 'function,  # comment)'

def test_case_0():
    var_0 = 35
    var_1 = None
    var_2 = True
    var_3 = False
    var_4 = '  #'
    var_5 = '    '
    var_6 = 'from module import function'
    var_7 = '\n'
    var_8 = 'from module import('
    var_9 = 'function)'

def test_case_0():
    var_0 = 10
    var_1 = None
    var_2 = True
    var_3 = False
    var_4 = '  #'
    var_5 = '    '
    var_6 = 'import verylongmodulename'
    var_7 = '\n'
    var_8 = 'import('
    var_9 = 'verylongmodulename)'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_29_false. Retrieved 9/11 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'short_line'
    var_3 = len(var_2)
    var_4 = 2
    var_5 = var_3 + var_4
    var_6 = var_1.wrap_length
    var_7 = var_1.line_length
    var_8 = var_6 or var_7
    var_9 = var_5 > var_8
    assert var_9 is False



# Parsed testcases at query #7
#--------------------------






# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_30_true. Retrieved 18/20 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 95
    var_4 = var_2 * var_3
    var_5 = 'part1'
    var_6 = 'part2'
    var_7 = 'part3'
    var_8 = 'part4'
    var_9 = 'part5'
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = len(var_4)
    var_12 = 2
    var_13 = var_11 + var_12
    var_14 = var_1.wrap_length
    var_15 = var_1.line_length
    var_16 = var_14 or var_15
    var_17 = var_13 > var_16
    var_18 = var_17 and var_10
    assert var_18 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_noqa_mode. Retrieved 5/9 statements.
# Partially parsed test_line_wrap_with_existing_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_splitter_import. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_splitter_as. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_splitter_dot. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_comment_and_noqa. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_trailing_comma_and_comment. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 80
    var_1 = 'import os'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = '  #'
    var_2 = True
    var_3 = 'from very_long_module_name import very_long_function_name  # some comment'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = '  #'
    var_2 = 'from very_long_module_name import very_long_function_name'
    var_3 = '\n'
    var_4 = '  # NOQA'

def test_case_0():
    var_0 = 20
    var_1 = '  #'
    var_2 = 'from very_long_module_name import very_long_function_name  # NOQA'
    var_3 = '\n'

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = '    '
    var_3 = 'from very_long_module_name import very_long_function_name, another_function'
    var_4 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = False
    var_4 = 'import very_long_module_name as very_long_alias_name'
    var_5 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = 'very_long_module_name.very_long_submodule.very_long_function'
    var_4 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from very_long_module_name import very_long_function_name  # noqa'
    var_5 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = False
    var_2 = '    '
    var_3 = 'from very_long_module_name import very_long_function_name'
    var_4 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from very_long_module_name import very_long_function_name  # comment'
    var_5 = '\n'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_parentheses_and_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_parentheses_and_noqa_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_noqa_mode. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_existing_noqa. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_comment_and_trailing_comma. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 100
    var_1 = None
    var_2 = False
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'import os'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = False
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'from very_long_module_name import very_long_function_name'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = False
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'module.submodule.verylongsubmodule'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = False
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'import very_long_module_name as vlm'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'from module import something  # some comment'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'from module import something  # noqa'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = False
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'from module import something'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = False
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'from module import something  # NOQA'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'from module import something'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'import something  # comment'
    var_6 = '\n'



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_true. Retrieved 4/11 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from very_long_module_name import very_long_submodule_name'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = bool(var_1.use_parentheses)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_mode_noqa. Retrieved 8/12 statements.
# Partially parsed test_line_wrap_mode_noqa_with_existing_noqa. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 8/12 statements.
# Partially parsed test_line_wrap_without_trailing_comma. Retrieved 9/13 statements.
# Partially parsed test_line_wrap_with_comment_and_trailing_comma. Retrieved 8/12 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_cimport_splitter. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 80
    var_1 = None
    var_2 = False
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'import os'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'from very_long_module_name import very_long_function_name'
    var_6 = '\n'
    var_7 = 'import'
    var_8 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'module.submodule.very_long_attribute_name'
    var_6 = '\n'
    var_7 = '.'
    var_8 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'import very_long_module_name as vlm'
    var_6 = '\n'
    var_7 = 'as'
    var_8 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'import very_long_module_name # some comment'
    var_6 = '\n'
    var_7 = '#'
    var_8 = 'some comment'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'import very_long_module_name # noqa'
    var_6 = '\n'
    var_7 = '#'
    var_8 = 'noqa'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = False
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'import very_long_module_name'
    var_6 = '\n'
    var_7 = 'NOQA'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = False
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'import very_long_module_name # NOQA'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = False
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'from very_long_module_name import very_long_function_name'
    var_6 = '\n'
    var_7 = '\\'
    var_8 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'import very_long_module_name'
    var_6 = '\n'
    var_7 = ','

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = False
    var_4 = ' # '
    var_5 = '    '
    var_6 = 'import very_long_module_name'
    var_7 = '\n'
    var_8 = ','

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'import very_long_module_name # comment'
    var_6 = '\n'
    var_7 = '#'
    var_8 = 'comment'
    var_9 = ','

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'import very_long_module_name'
    var_6 = '\n'
    var_7 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = True
    var_3 = ' # '
    var_4 = '    '
    var_5 = 'cimport very_long_module_name'
    var_6 = '\n'
    var_7 = 'cimport'
    var_8 = '\n'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_15_true. Retrieved 6/9 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'some comment'
    var_3 = var_1.use_parentheses
    var_4 = 'noqa'
    var_5 = var_4 in var_2
    var_6 = var_3 and var_5



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_15_true. Retrieved 6/9 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# some comment'
    var_3 = var_1.use_parentheses
    var_4 = 'noqa'
    var_5 = var_4 in var_2
    var_6 = var_3 and var_5



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import_split. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_as_split. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_dot_split. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_noqa_mode_with_existing_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_comment_and_trailing_comma. Retrieved 8/12 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 6/9 statements.
# Partially parsed test_line_short_content_no_wrap. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_cimport_split. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_custom_line_separator. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 80

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = 'from very_long_module_name import very_long_function_name'
    var_6 = '\n'
    var_7 = 'from very_long_module_name import ('
    var_8 = 'very_long_function_name'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = 'import very_long_module_name as vlm'
    var_6 = '\n'

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = 'very_long_module_name.very_long_submodule'
    var_6 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  # '
    var_6 = 'from module import something  # some comment'
    var_7 = '\n'
    var_8 = 'from module import ('
    var_9 = 'something'
    var_10 = '# some comment'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  # '
    var_6 = 'from module import something  # noqa'
    var_7 = '\n'
    var_8 = 'from module import ('
    var_9 = '# noqa'

def test_case_0():
    var_0 = 30
    var_1 = '  # '
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = '  # '
    var_2 = 'from module import something  # NOQA'
    var_3 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = '  # '
    var_5 = 'from module import something'
    var_6 = '\n'
    var_7 = 'from module import ('
    var_8 = 'something,'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = '  # '
    var_5 = 'from module import something  # comment'
    var_6 = '\n'
    var_7 = 'from module import ('
    var_8 = 'something  # comment'
    var_9 = ')'

def test_case_0():
    var_0 = 30
    var_1 = False
    var_2 = '    '
    var_3 = None
    var_4 = 'from module import something'
    var_5 = '\n'
    var_6 = 'from module import \\'

def test_case_0():
    var_0 = 80
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = 'import os'
    var_6 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = 'cimport very_long_module_name'
    var_6 = '\n'
    var_7 = 'cimport very_long_module_name'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = 'from module import something'
    var_6 = '\n'
    var_7 = 'from module import ('

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = 'from module import something'
    var_6 = '\r\n'
    var_7 = '\r\n'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_mode_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_mode_noqa_with_existing_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_no_wrap_with_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_trailing_comma_and_comment. Retrieved 7/11 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_cimport_splitter. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 100

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from very_long_module_name import very_long_function_name'
    var_5 = '\n'
    var_6 = 'from very_long_module_name import('
    var_7 = '    very_long_function_name,'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = False
    var_5 = 'module.submodule.verylongclassname.verylongmethodname'
    var_6 = '\n'
    var_7 = 'module.submodule.verylongclassname.('
    var_8 = '    verylongmethodname'

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'import very_long_module_name as vlm'
    var_5 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # some comment'
    var_5 = '\n'
    var_6 = 'from module import(  # some comment'
    var_7 = '    something,'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # noqa'
    var_5 = '\n'
    var_6 = 'from module import(  # noqa'
    var_7 = '    something,'

def test_case_0():
    var_0 = 10
    var_1 = '  # '
    var_2 = 'import verylongmodule'
    var_3 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = '  # '
    var_2 = 'import module  # NOQA'
    var_3 = '\n'

def test_case_0():
    var_0 = 100
    var_1 = '  # '
    var_2 = 'import os  # comment'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import very_long_function_name'
    var_5 = '\n'
    var_6 = 'from module import\\'
    var_7 = '    very_long_function_name'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # comment'
    var_5 = '\n'
    var_6 = 'something,  # comment)'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = False
    var_5 = 'from module import something, another_thing'
    var_6 = '\n'
    var_7 = 'from module import('
    var_8 = '    something,'
    var_9 = '    another_thing'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'cimport very_long_module_name'
    var_5 = '\n'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import_split. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_as_split. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_dot_split. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_noqa_mode_with_existing_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_trailing_comma_and_comment. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 80

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'from very_long_module_name import very_long_function_name'
    var_4 = '\n'
    var_5 = 'from very_long_module_name import ('
    var_6 = 'very_long_function_name'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'import very_long_module_name as vlm'
    var_4 = '\n'
    var_5 = 'import very_long_module_name as ('
    var_6 = 'vlm'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'very_long_module_name.very_long_submodule'
    var_4 = '\n'
    var_5 = 'very_long_module_name.('
    var_6 = 'very_long_submodule'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # some comment'
    var_5 = '\n'
    var_6 = 'from module import ('
    var_7 = 'something'
    var_8 = '# some comment'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # noqa'
    var_5 = '\n'
    var_6 = 'from module import ('
    var_7 = 'something'
    var_8 = '# noqa'

def test_case_0():
    var_0 = 20
    var_1 = '  # '
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = '  # '
    var_2 = 'from module import something  # NOQA'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = '    '
    var_3 = 'from module import something'
    var_4 = '\n'
    var_5 = 'from module import \\'
    var_6 = 'something'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'from module import something'
    var_4 = '\n'
    var_5 = 'from module import ('
    var_6 = 'something'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # comment'
    var_5 = '\n'
    var_6 = 'from module import ('
    var_7 = 'something,'
    var_8 = '# comment'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_17_true. Retrieved 6/12 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'some_content'
    var_3 = 'some_comment'
    var_4 = var_1.include_trailing_comma
    var_5 = var_1.use_parentheses
    var_6 = ','



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_include_trailing_comma_with_parentheses_and_no_trailing_comma. Retrieved 5/11 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import something'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = ','



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_71_true. Retrieved 7/10 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 81
    var_4 = var_2 * var_3
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_1)
    var_7 = f'{var_1.comment_prefix} NOQA'



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_true. Retrieved 4/11 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from very_long_module_name import very_long_submodule_name'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = var_1.use_parentheses
    assert var_5 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import_split. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_as_split. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_dot_split. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 6/9 statements.
# Partially parsed test_line_noqa_mode_with_long_line. Retrieved 4/7 statements.
# Partially parsed test_line_noqa_mode_with_existing_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 7/11 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 100

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from very_long_module_name import very_long_function_name'
    var_5 = '\n'
    var_6 = 'from very_long_module_name import ('
    var_7 = 'very_long_function_name'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'import very_long_module_name as vlm'
    var_5 = '\n'

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'very_long_module_name.very_long_submodule'
    var_5 = '\n'
    var_6 = 'very_long_module_name.('
    var_7 = 'very_long_submodule'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from module import something  # some comment'
    var_5 = '\n'
    var_6 = 'from module import ('
    var_7 = 'something'
    var_8 = '# some comment'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from module import something  # noqa'
    var_5 = '\n'
    var_6 = 'from module import ('
    var_7 = 'something'
    var_8 = '# noqa'

def test_case_0():
    var_0 = 30
    var_1 = '  #'
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = '  #'
    var_2 = 'from module import something  # NOQA'
    var_3 = '\n'

def test_case_0():
    var_0 = 25
    var_1 = False
    var_2 = '    '
    var_3 = '  #'
    var_4 = True
    var_5 = 'very_long_module_name.very_long_submodule'
    var_6 = '\n'
    var_7 = 'very_long_module_name.\\'
    var_8 = 'very_long_submodule'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from module import something'
    var_5 = '\n'
    var_6 = ','

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from module import something'
    var_5 = '\n'
    var_6 = 'from module import ('
    var_7 = 'something'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_at_line_15_true. Retrieved 6/10 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'some comment with noqa'
    var_3 = var_1.use_parentheses
    var_4 = 'noqa'
    var_5 = var_4 in var_2
    var_6 = var_3 and var_5



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import_split. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_as_split. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_dot_split. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_mode_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_mode_noqa_with_existing_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 8/12 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 7/10 statements.
# Partially parsed test_line_short_content_no_wrap. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_comment_and_trailing_comma. Retrieved 8/12 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_splitter_at_start. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 100

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  #'
    var_6 = 'from very_long_module_name import very_long_function_name'
    var_7 = '\n'
    var_8 = 'from very_long_module_name import ('
    var_9 = 'very_long_function_name'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  #'
    var_6 = 'import very_long_module_name as vlm'
    var_7 = '\n'

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  #'
    var_6 = 'very_long_module_name.very_long_submodule'
    var_7 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  #'
    var_6 = 'from module import something  # some comment'
    var_7 = '\n'
    var_8 = '# some comment'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  #'
    var_6 = 'from module import something  # noqa'
    var_7 = '\n'
    var_8 = '# noqa'

def test_case_0():
    var_0 = 10
    var_1 = '  #'
    var_2 = 'import very_long_module'
    var_3 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = '  #'
    var_2 = 'import module  # NOQA'
    var_3 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = '  #'
    var_5 = 'from module import something'
    var_6 = '\n'
    var_7 = ','

def test_case_0():
    var_0 = 25
    var_1 = False
    var_2 = '    '
    var_3 = None
    var_4 = '  #'
    var_5 = 'very_long_module_name.very_long_submodule'
    var_6 = '\n'
    var_7 = '\\'

def test_case_0():
    var_0 = 100
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  #'
    var_6 = 'import os'
    var_7 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = '  #'
    var_5 = 'from module import something  # comment'
    var_6 = '\n'
    var_7 = '# comment'
    var_8 = ','

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  #'
    var_6 = 'from module import something'
    var_7 = '\n'
    var_8 = 'from module import ('

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  #'
    var_6 = 'cimport very_long_module_name'
    var_7 = '\n'
    var_8 = 'cimport very_long_module_name'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  #'
    var_6 = 'import very_long_module_name'
    var_7 = '\n'
    var_8 = 'import very_long_module_name'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_mode_noqa. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_mode_noqa_with_existing_noqa. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_trailing_comma_and_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_comment_prefix_in_last_line. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 80
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = False
    var_6 = 'import os'
    var_7 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'from very_long_module_name import very_long_function_name'
    var_6 = '\n'
    var_7 = 'from very_long_module_name import ('
    var_8 = 'very_long_function_name'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = False
    var_6 = 'module.submodule.verylongclass.verylongmethod'
    var_7 = '\n'
    var_8 = 'module.submodule.verylongclass.('
    var_9 = 'verylongmethod'

def test_case_0():
    var_0 = 25
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'import very_long_module_name as vlm'
    var_6 = '\n'
    var_7 = 'import very_long_module_name as ('
    var_8 = 'vlm'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'from module import something  # some comment'
    var_6 = '\n'
    var_7 = 'from module import ('
    var_8 = '# some comment'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'from module import something  # noqa'
    var_6 = '\n'
    var_7 = 'from module import ('
    var_8 = '# noqa'

def test_case_0():
    var_0 = 10
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = False
    var_5 = 'import verylongmodule'
    var_6 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = False
    var_5 = 'import module  # NOQA'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = False
    var_5 = 'from module import very_long_function_name'
    var_6 = '\n'
    var_7 = 'from module import \\'
    var_8 = 'very_long_function_name'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'from module import something  # comment'
    var_6 = '\n'
    var_7 = ','
    var_8 = '# comment'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'from module import something  # noqa'
    var_6 = '\n'
    var_7 = -1
    var_8 = '# noqa)'



# Parsed testcases at query #29
#--------------------------






# Parsed testcases at query #30
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import_split. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_as_split. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_dot_split. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_noqa_mode_with_existing_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_trailing_comma_and_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_without_trailing_comma. Retrieved 8/12 statements.
# Partially parsed test_line_short_content_no_wrap. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_multiple_splits. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_comment_inside_parentheses. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 100

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'from very_long_module_name import very_long_function_name'
    var_4 = '\n'
    var_5 = 'from very_long_module_name import ('
    var_6 = 'very_long_function_name,'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'import very_long_module_name as very_long_alias'
    var_4 = '\n'
    var_5 = 'import very_long_module_name as ('
    var_6 = 'very_long_alias'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'very_long_module_name.very_long_submodule.very_long_function'
    var_4 = '\n'
    var_5 = 'very_long_module_name.('
    var_6 = 'very_long_submodule.very_long_function'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # some comment'
    var_5 = '\n'
    var_6 = 'from module import ('
    var_7 = 'something,  # some comment'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # noqa'
    var_5 = '\n'
    var_6 = 'from module import (  # noqa'
    var_7 = 'something'

def test_case_0():
    var_0 = 20
    var_1 = '  # '
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = '  # '
    var_2 = 'from module import something  # NOQA'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = '    '
    var_3 = 'from module import something'
    var_4 = '\n'
    var_5 = 'from module import \\'
    var_6 = 'something'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'from module import something'
    var_4 = '\n'
    var_5 = 'from module import ('
    var_6 = 'something,'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # comment'
    var_5 = '\n'
    var_6 = 'something,  # comment'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = False
    var_4 = '  # '
    var_5 = 'from module import something  # comment'
    var_6 = '\n'
    var_7 = 'something  # comment'
    var_8 = ','

def test_case_0():
    var_0 = 100
    var_1 = 'import os'
    var_2 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = 'import very_long_module_name.submodule as alias'
    var_4 = '\n'
    var_5 = 'import very_long_module_name.('
    var_6 = 'submodule as alias'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  # '
    var_4 = 'from module import something  # noqa'
    var_5 = '\n'
    var_6 = 0
    var_7 = '(  # noqa'
    var_8 = 'something'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 80
    var_1 = True
    var_2 = '  # '
    var_3 = '    '
    var_4 = 'from module import very_long_name_that_exceeds_line_length_by_a_lot'
    var_5 = '\n'
    var_6 = 'import'
    var_7 = 'very_long_name_that_exceeds_line_length_by_a_lot'



# Parsed testcases at query #32
#--------------------------






####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_line_wrap_with_noqa_comment_and_parentheses. Retrieved 11/12 statements.



def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 80
    var_3 = 3
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'import os'


def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = True
    var_3 = '    '
    var_4 = None
    var_5 = False
    var_6 = '  # '
    var_7 = 'line_length'
    var_8 = 'multi_line_output'
    var_9 = 'use_parentheses'
    var_10 = 'indent'
    var_11 = 'wrap_length'
    var_12 = 'include_trailing_comma'
    var_13 = 'comment_prefix'
    var_14 = {var_7: var_0, var_8: var_1, var_9: var_2, var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_6}
    var_15 = module_0.Config(**var_14)
    var_16 = 'from very_long_module_name import very_long_function_name'
    var_17 = '\n'
    var_18 = module_1.line(var_16, var_17, var_15)
    var_19 = 'from very_long_module_name import ('
    var_20 = bool('from very_long_module_name import (' in var_18)
    assert var_20 is True
    var_21 = 'very_long_function_name'
    var_22 = bool('very_long_function_name' in var_18)
    assert var_22 is True


def test_case_0():
    var_0 = 30
    var_1 = 3
    var_2 = True
    var_3 = '    '
    var_4 = None
    var_5 = False
    var_6 = '  # '
    var_7 = 'line_length'
    var_8 = 'multi_line_output'
    var_9 = 'use_parentheses'
    var_10 = 'indent'
    var_11 = 'wrap_length'
    var_12 = 'include_trailing_comma'
    var_13 = 'comment_prefix'
    var_14 = {var_7: var_0, var_8: var_1, var_9: var_2, var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_6}
    var_15 = module_0.Config(**var_14)
    var_16 = 'import very_long_module_name as very_long_alias'
    var_17 = '\n'
    var_18 = module_1.line(var_16, var_17, var_15)
    var_19 = 'import very_long_module_name as ('
    var_20 = bool('import very_long_module_name as (' in var_18)
    assert var_20 is True
    var_21 = 'very_long_alias'
    var_22 = bool('very_long_alias' in var_18)
    assert var_22 is True


def test_case_0():
    var_0 = 30
    var_1 = 3
    var_2 = True
    var_3 = '    '
    var_4 = None
    var_5 = False
    var_6 = '  # '
    var_7 = 'line_length'
    var_8 = 'multi_line_output'
    var_9 = 'use_parentheses'
    var_10 = 'indent'
    var_11 = 'wrap_length'
    var_12 = 'include_trailing_comma'
    var_13 = 'comment_prefix'
    var_14 = {var_7: var_0, var_8: var_1, var_9: var_2, var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_6}
    var_15 = module_0.Config(**var_14)
    var_16 = 'very_long_module_name.very_long_submodule.very_long_function'
    var_17 = '\n'
    var_18 = module_1.line(var_16, var_17, var_15)
    var_19 = 'very_long_module_name.('
    var_20 = bool('very_long_module_name.(' in var_18)
    assert var_20 is True
    var_21 = 'very_long_submodule.very_long_function'
    var_22 = bool('very_long_submodule.very_long_function' in var_18)
    assert var_22 is True


def test_case_0():
    var_0 = 30
    var_1 = 3
    var_2 = True
    var_3 = '    '
    var_4 = None
    var_5 = False
    var_6 = '  # '
    var_7 = 'line_length'
    var_8 = 'multi_line_output'
    var_9 = 'use_parentheses'
    var_10 = 'indent'
    var_11 = 'wrap_length'
    var_12 = 'include_trailing_comma'
    var_13 = 'comment_prefix'
    var_14 = {var_7: var_0, var_8: var_1, var_9: var_2, var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_6}
    var_15 = module_0.Config(**var_14)
    var_16 = 'from module import something  # some comment'
    var_17 = '\n'
    var_18 = module_1.line(var_16, var_17, var_15)
    var_19 = 'from module import ('
    var_20 = bool('from module import (' in var_18)
    assert var_20 is True
    var_21 = 'something  # some comment'
    var_22 = bool('something  # some comment' in var_18)
    assert var_22 is True


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = False
    var_3 = '    '
    var_4 = None
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'use_parentheses'
    var_9 = 'indent'
    var_10 = 'wrap_length'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_2, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'import verylongmodule'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    assert var_17 == 'import verylongmodule  # NOQA'


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = False
    var_3 = '    '
    var_4 = None
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'use_parentheses'
    var_9 = 'indent'
    var_10 = 'wrap_length'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_2, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'import verylongmodule  # NOQA'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    assert var_17 == 'import verylongmodule  # NOQA'


def test_case_0():
    var_0 = 30
    var_1 = 3
    var_2 = True
    var_3 = '    '
    var_4 = None
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'use_parentheses'
    var_9 = 'indent'
    var_10 = 'wrap_length'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_2, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from module import very_long_function_name'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    var_18 = 'from module import ('
    var_19 = bool('from module import (' in var_17)
    assert var_19 is True
    var_20 = 'very_long_function_name,'
    var_21 = bool('very_long_function_name,' in var_17)
    assert var_21 is True


def test_case_0():
    var_0 = 30
    var_1 = 3
    var_2 = True
    var_3 = '    '
    var_4 = None
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'use_parentheses'
    var_9 = 'indent'
    var_10 = 'wrap_length'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_2, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from module import something  # noqa'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    var_18 = 'from module import ('
    var_19 = bool('from module import (' in var_17)
    assert var_19 is True
    var_20 = '# noqa'
    var_21 = bool('# noqa' in var_17)
    assert var_21 is True
    var_22 = ')'


def test_case_0():
    var_0 = 30
    var_1 = 4
    var_2 = True
    var_3 = '    '
    var_4 = None
    var_5 = False
    var_6 = '  # '
    var_7 = 'line_length'
    var_8 = 'multi_line_output'
    var_9 = 'use_parentheses'
    var_10 = 'indent'
    var_11 = 'wrap_length'
    var_12 = 'include_trailing_comma'
    var_13 = 'comment_prefix'
    var_14 = {var_7: var_0, var_8: var_1, var_9: var_2, var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_6}
    var_15 = module_0.Config(**var_14)
    var_16 = 'from module import very_long_function_name'
    var_17 = '\n'
    var_18 = module_1.line(var_16, var_17, var_15)
    var_19 = 'from module import ('
    var_20 = bool('from module import (' in var_18)
    assert var_20 is True
    var_21 = '\n'
    var_22 = bool('\n' in var_18)
    assert var_22 is True


def test_case_0():
    var_0 = 30
    var_1 = 3
    var_2 = False
    var_3 = '    '
    var_4 = None
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'use_parentheses'
    var_9 = 'indent'
    var_10 = 'wrap_length'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_2, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from module import very_long_function_name'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    var_18 = '\\'
    var_19 = bool('\\' in var_17)
    assert var_19 is True
    var_20 = '\n'
    var_21 = bool('\n' in var_17)
    assert var_21 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_29_true. Retrieved 16/18 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = 'part1'
    var_6 = 'part2'
    var_7 = 'part3'
    var_8 = [var_5, var_6, var_7]
    var_9 = len(var_4)
    var_10 = 2
    var_11 = var_9 + var_10
    var_12 = var_1.wrap_length
    var_13 = var_1.line_length
    var_14 = var_12 or var_13
    var_15 = var_11 > var_14
    var_16 = var_15 and var_8
    assert var_16 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 6/8 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 101
    var_4 = var_2 * var_3
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_1)
    assert var_6 == 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa# NOQA'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_comment_prefix_in_last_line_and_ends_with_parenthesis. Retrieved 11/21 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '\n'
    var_3 = 'from module import ('
    var_4 = 'import '
    var_5 = '# noqa'
    var_6 = '# noqa'
    var_7 = '    submodule'
    var_8 = '\n'
    var_9 = f'{var_3}{var_4}({var_6}{var_2}{var_7}{var_8})'
    var_10 = var_1.comment_prefix
    var_11 = -1
    var_12 = ')'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_import_split. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_dot_split. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_as_split. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_mode_noqa. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_mode_noqa_with_existing_noqa. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 8/12 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 80
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'import os'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'from very_long_module_name import very_long_function_name'
    var_6 = '\n'
    var_7 = 'import'
    var_8 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'module.submodule.verylongclass.verylongmethod'
    var_6 = '\n'
    var_7 = '.'
    var_8 = '\n'

def test_case_0():
    var_0 = 25
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'import very_long_module_name as vlm'
    var_6 = '\n'
    var_7 = 'as'
    var_8 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'import very_long_module_name  # some comment'
    var_6 = '\n'
    var_7 = '# some comment'
    var_8 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'import very_long_module_name  # noqa'
    var_6 = '\n'
    var_7 = '# noqa'
    var_8 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'import very_long_module_name_that_exceeds_length'
    var_6 = '\n'
    var_7 = '# NOQA'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'import module  # NOQA'
    var_6 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'from module import very_long_function_name'
    var_6 = '\n'
    var_7 = ','

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = False
    var_5 = True
    var_6 = 'import very_long_module_name'
    var_7 = '\n'
    var_8 = '\\'
    var_9 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'cimport very_long_module_name'
    var_6 = '\n'
    var_7 = 'cimport'
    var_8 = '\n'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_import_statement_single_line. Retrieved 7/9 statements.
# Partially parsed test_import_statement_multi_line_grid. Retrieved 9/11 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 10/12 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 12/23 statements.
# Partially parsed test_import_statement_include_trailing_comma. Retrieved 8/12 statements.
# Partially parsed test_import_statement_custom_indent. Retrieved 7/10 statements.
# Partially parsed test_import_statement_line_separator. Retrieved 7/9 statements.
# Partially parsed test_import_statement_wrap_line_single. Retrieved 7/10 statements.
# Partially parsed test_import_statement_no_comments_when_ignored. Retrieved 9/12 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_1.import_statement(var_2, var_5, explode=var_6)
    var_8 = 'from module import (\n    item1,\n    item2,\n)'
    var_9 = bool(var_7 == var_8)
    assert var_9 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = [var_3, var_4]
    var_6 = module_1.import_statement(var_2, var_5, config=var_1)
    var_7 = 'from module import item1, item2'
    var_8 = bool(var_6 == var_7)
    assert var_8 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = 'item3'
    var_6 = 'item4'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_1.import_statement(var_2, var_7, config=var_1)
    var_9 = 'from module import (item1, item2,\n                  item3, item4)'
    var_10 = bool(var_8 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'comment1'
    var_3 = 'comment2'
    var_4 = [var_2, var_3]
    var_5 = 'from module'
    var_6 = 'item1'
    var_7 = 'item2'
    var_8 = [var_6, var_7]
    var_9 = module_1.import_statement(var_5, var_8, var_4, config=var_1)
    var_10 = 'from module import (  # comment1\n    item1,  # comment2\n    item2,\n)'
    var_11 = bool(var_9 == var_10)
    assert var_11 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = 'item3'
    var_6 = 'item4'
    var_7 = 'item5'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_2, var_8, config=var_1)
    var_10 = '\n'
    var_11 = 0
    var_12 = -1


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = 'item3'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_1.import_statement(var_2, var_6, config=var_1)
    var_8 = ',\n)'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = 'item3'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_1.import_statement(var_2, var_6, config=var_1)
    var_8 = '    item1,'
    var_9 = bool('    item1,' in var_7)
    assert var_9 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = [var_3, var_4]
    var_6 = '\r\n'
    var_7 = module_1.import_statement(var_2, var_5, line_separator=var_6, config=var_1)
    var_8 = '\r\n'
    var_9 = bool('\r\n' in var_7)
    assert var_9 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = [var_3, var_4]
    var_6 = module_1.import_statement(var_2, var_5, config=var_1)
    var_7 = '\n'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'comment1'
    var_3 = 'comment2'
    var_4 = [var_2, var_3]
    var_5 = 'from module'
    var_6 = 'item1'
    var_7 = 'item2'
    var_8 = [var_6, var_7]
    var_9 = module_1.import_statement(var_5, var_8, var_4, config=var_1)
    var_10 = '# comment1'
    var_11 = bool('# comment1' not in var_9)
    assert var_11 is True
    var_12 = '# comment2'
    var_13 = bool('# comment2' not in var_9)
    assert var_13 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import_split. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_dot_split. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_as_split. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_noqa_mode. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_noqa_mode_with_existing_noqa. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 100

def test_case_0():
    var_0 = 'from very_long_module_name import very_long_function_name'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = '    '
    var_5 = 'from very_long_module_name import (\n    very_long_function_name,\n)'

def test_case_0():
    var_0 = 'very_long_module_name.very_long_submodule.very_long_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = '    '
    var_5 = False
    var_6 = 'very_long_module_name.very_long_submodule.(\n    very_long_function\n)'

def test_case_0():
    var_0 = 'import very_long_module_name as very_long_alias'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = '    '
    var_5 = 'import very_long_module_name as very_long_alias'

def test_case_0():
    var_0 = 'from module import something  # some comment'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = '    '
    var_5 = '  # '
    var_6 = 'from module import (  # some comment\n    something,\n)'

def test_case_0():
    var_0 = 'from module import something  # noqa'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = '    '
    var_5 = '  # '
    var_6 = 'from module import (  # noqa\n    something,\n)'

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = 30
    var_3 = '  # '
    var_4 = 'from module import something  # NOQA'

def test_case_0():
    var_0 = 'from module import something  # NOQA'
    var_1 = '\n'
    var_2 = 30
    var_3 = '  # '
    var_4 = 'from module import something  # NOQA'

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = 30
    var_3 = False
    var_4 = '    '
    var_5 = 'from module import \\\n    something'

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = '    '
    var_5 = 'from module import (\n    something,\n)'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_71_true. Retrieved 6/8 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 90
    var_4 = var_2 * var_3
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_1)
    var_7 = bool(var_6 == var_4 + var_1.comment_prefix + ' NOQA')
    assert var_7 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_65_false. Retrieved 15/28 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '\n'
    var_3 = 'from module import something'
    var_4 = 'import '
    var_5 = var_1.multi_line_output
    var_6 = '# noqa'
    var_7 = f'{var_1.comment_prefix}{var_6}'
    var_8 = '    submodule'
    var_9 = ''
    var_10 = f'{var_3}{var_4}({var_7}{var_2}{var_8}{var_9}{var_2})'
    var_11 = var_1.comment_prefix
    var_12 = -1
    var_13 = -1
    var_14 = ')'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_43_true. Retrieved 6/13 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'very_long_module_name'
    var_3 = 'as '
    var_4 = 'short_name'
    var_5 = '\n'
    var_6 = f'{var_2}{var_3}{var_4.lstrip()}'
    assert var_6 == 'very_long_module_name as short_name'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_29_false. Retrieved 9/11 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'short_line'
    var_3 = len(var_2)
    var_4 = 2
    var_5 = var_3 + var_4
    var_6 = var_1.wrap_length
    var_7 = var_1.line_length
    var_8 = var_6 or var_7
    var_9 = var_5 > var_8
    assert var_9 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_11_true. Retrieved 12/23 statements.


import re as module_0


def test_case_0():
    var_0 = 80
    var_1 = True
    var_2 = '  #'
    var_3 = 'from module import something as alias'
    var_4 = '\n'
    var_5 = 'as '
    var_6 = bool('as ' in var_3)
    assert var_6 is True
    var_7 = 'as '
    var_8 = '\\b'
    var_9 = module_0.escape(var_7)
    var_10 = var_8 + var_9
    var_11 = var_10 + var_8
    var_12 = module_0.search(var_11, var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import_split. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_as_split. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_dot_split. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_noqa_mode_with_existing_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_trailing_comma_and_comment. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 80
    var_1 = 'import os'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from very_long_module_name import very_long_function_name'
    var_5 = '\n'
    var_6 = 'from very_long_module_name import('
    var_7 = 'very_long_function_name'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'import very_long_module_name as very_long_alias'
    var_5 = '\n'
    var_6 = 'import very_long_module_name as'
    var_7 = 'very_long_alias'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'very_long_module_name.very_long_submodule.very_long_function'
    var_5 = '\n'
    var_6 = 'very_long_module_name.('
    var_7 = 'very_long_submodule.very_long_function'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from module import something  # some comment'
    var_5 = '\n'
    var_6 = 'from module import('
    var_7 = 'something,'
    var_8 = '  # some comment'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from module import something  # noqa'
    var_5 = '\n'
    var_6 = 'from module import(  # noqa'
    var_7 = 'something'

def test_case_0():
    var_0 = 20
    var_1 = '  #'
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = '  #'
    var_2 = 'from module import something  # NOQA'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from module import something'
    var_5 = '\n'
    var_6 = 'from module import\\'
    var_7 = 'something'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from module import something  # comment'
    var_5 = '\n'
    var_6 = 'something,'
    var_7 = '  # comment'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = '  #'
    var_4 = 'from module import something'
    var_5 = '\n'
    var_6 = 'from module import('
    var_7 = 'something'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_mode_noqa. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_mode_noqa_with_existing_noqa. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_without_trailing_comma. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_comment_and_trailing_comma. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 80
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = False
    var_5 = 'import os'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'from very_long_module_name import very_long_function_name'
    var_6 = '\n'
    var_7 = 'from very_long_module_name import('
    var_8 = 'very_long_function_name'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'module.submodule.very_long_attribute_name'
    var_6 = '\n'
    var_7 = 'module.submodule.('
    var_8 = 'very_long_attribute_name'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'import very_long_module_name as vlm'
    var_6 = '\n'
    var_7 = 'import very_long_module_name as'
    var_8 = 'vlm'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'import very_long_module_name  # some comment'
    var_6 = '\n'
    var_7 = '# some comment'
    var_8 = 'import very_long_module_name'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'import very_long_module_name  # NOQA'
    var_6 = '\n'
    var_7 = '# NOQA'
    var_8 = 'import very_long_module_name'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = False
    var_5 = 'import very_long_module_name'
    var_6 = '\n'
    var_7 = 'import very_long_module_name'
    var_8 = '\\'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = False
    var_5 = 'import very_long_module_name'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = False
    var_5 = 'import very_long_module_name  # NOQA'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'import very_long_module_name'
    var_6 = '\n'
    var_7 = ','

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = False
    var_6 = 'import very_long_module_name'
    var_7 = '\n'
    var_8 = ','

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = '# '
    var_4 = True
    var_5 = 'import very_long_module_name  # comment'
    var_6 = '\n'
    var_7 = '# comment'
    var_8 = ','



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_11_true. Retrieved 12/23 statements.



def test_case_0():
    var_0 = 50
    var_1 = True
    var_2 = '  # '
    var_3 = 'from module import very_long_name_that_exceeds_line_length'
    var_4 = '\n'
    var_5 = 'import '
    var_6 = bool('import ' in var_3)
    assert var_6 is True
    var_7 = 'import '
    var_8 = '\\b'
    var_9 = module_0.escape(var_7)
    var_10 = var_8 + var_9
    var_11 = var_10 + var_8
    var_12 = module_0.search(var_11, var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import_split. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_as_split. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_dot_split. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 8/11 statements.
# Partially parsed test_line_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_noqa_mode_with_existing_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 6/9 statements.
# Partially parsed test_line_wrap_with_comment_and_trailing_comma. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 100

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = 'from very_long_module_name import very_long_function_name'
    var_6 = '\n'
    var_7 = 'from very_long_module_name import ('
    var_8 = 'very_long_function_name'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = 'import very_long_module_name as very_long_alias'
    var_5 = '\n'
    var_6 = 'import very_long_module_name as ('
    var_7 = 'very_long_alias'

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = 'very.long.package.path.to.module'
    var_6 = '\n'
    var_7 = 'very.long.package.path.to.('
    var_8 = 'module'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  # '
    var_6 = 'from module import something  # some comment'
    var_7 = '\n'
    var_8 = 'from module import ('
    var_9 = 'something'
    var_10 = '# some comment'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  # '
    var_6 = 'from module import something  # noqa'
    var_7 = '\n'
    var_8 = 'from module import (  # noqa'
    var_9 = 'something'

def test_case_0():
    var_0 = 10
    var_1 = '  # '
    var_2 = 'import verylongmodule'
    var_3 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = '  # '
    var_2 = 'import module  # NOQA'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = '    '
    var_3 = None
    var_4 = 'from module import something'
    var_5 = '\n'
    var_6 = 'from module import \\'
    var_7 = 'something'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = 'from module import something'
    var_5 = '\n'
    var_6 = 'from module import ('
    var_7 = 'something,'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = '  # '
    var_5 = 'from module import something  # comment'
    var_6 = '\n'
    var_7 = 'from module import ('
    var_8 = 'something,  # comment'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 7/12 statements.



def test_case_0():
    var_0 = 'from module import something'
    var_1 = 'import '
    var_2 = '\\b'
    var_3 = module_0.escape(var_1)
    var_4 = var_2 + var_3
    var_5 = var_4 + var_2
    var_6 = module_0.search(var_5, var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_65_false. Retrieved 14/27 statements.


import isort.settings as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '\n'
    var_3 = 'from module import something'
    var_4 = 'import '
    var_5 = 'noqa'
    var_6 = '# noqa'
    var_7 = '    another_thing'
    var_8 = ','
    var_9 = '\n'
    var_10 = f'{var_3}{var_4}({var_6}{var_2}{var_7}{var_8}{var_9})'
    var_11 = var_1.comment_prefix
    var_12 = -1
    var_13 = -1
    var_14 = ')'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_71_true. Retrieved 6/8 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 81
    var_4 = var_2 * var_3
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_1)
    var_7 = bool(var_6 == var_4 + var_1.comment_prefix + ' NOQA')
    assert var_7 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_import_statement_default_formatter. Retrieved 7/11 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 8/11 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 8/13 statements.
# Partially parsed test_import_statement_single_line_no_wrap. Retrieved 5/9 statements.
# Partially parsed test_import_statement_include_trailing_comma. Retrieved 8/13 statements.
# Partially parsed test_import_statement_custom_line_separator. Retrieved 6/9 statements.
# Partially parsed test_import_statement_remove_comments. Retrieved 9/12 statements.
# Partially parsed test_import_statement_formatter_from_string. Retrieved 8/12 statements.
# Partially parsed test_import_statement_wrap_line_single_line. Retrieved 6/10 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_1.import_statement(var_2, var_5, explode=var_6)
    var_8 = 'from module import (\n    item1,\n    item2,\n)'
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

def test_case_0():
    var_0 = 50
    var_1 = 'from module'
    var_2 = 'item1'
    var_3 = 'item2'
    var_4 = 'item3'
    var_5 = [var_2, var_3, var_4]
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'comment1'
    var_2 = 'comment2'
    var_3 = [var_1, var_2]
    var_4 = 'from module'
    var_5 = 'item1'
    var_6 = 'item2'
    var_7 = [var_5, var_6]

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 10
    var_3 = range(var_2)
    var_4 = 'item'
    var_5 = [var_4 + str(i) for i in var_3]
    var_6 = 'from module'
    var_7 = '\n'

def test_case_0():
    var_0 = 100
    var_1 = 'from module'
    var_2 = 'item1'
    var_3 = [var_2]
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module'
    var_3 = 'item1'
    var_4 = 'item2'
    var_5 = 'item3'
    var_6 = [var_3, var_4, var_5]
    var_7 = ','

def test_case_0():
    var_0 = 20
    var_1 = 'from module'
    var_2 = 'item1'
    var_3 = 'item2'
    var_4 = [var_2, var_3]
    var_5 = '\r\n'
    var_6 = '\r\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'comment1'
    var_3 = 'comment2'
    var_4 = [var_2, var_3]
    var_5 = 'from module'
    var_6 = 'item1'
    var_7 = 'item2'
    var_8 = [var_6, var_7]

def test_case_0():
    var_0 = 50
    var_1 = 'from module'
    var_2 = 'item1'
    var_3 = 'item2'
    var_4 = 'item3'
    var_5 = 'item4'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'from module'
    var_2 = 'item1'
    var_3 = 'item2'
    var_4 = [var_2, var_3]
    var_5 = '\n'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_include_trailing_comma_true_use_parentheses_true_no_trailing_comma. Retrieved 13/21 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import something'
    var_3 = '\n'
    var_4 = 'comment'
    var_5 = 'import '
    var_6 = 'from module '
    var_7 = 'something'
    var_8 = [var_6, var_7]
    var_9 = var_1.include_trailing_comma
    var_10 = var_1.use_parentheses
    var_11 = ','
    var_12 = ''



# Parsed testcases at query #22
#--------------------------





def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 100
    var_3 = 3
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'import os'


def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = 18
    var_3 = '    '
    var_4 = True
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'wrap_length'
    var_9 = 'indent'
    var_10 = 'use_parentheses'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_4, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from very_long_module_name import very_long_function_name'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    var_18 = 'from very_long_module_name import('
    var_19 = bool('from very_long_module_name import(' in var_17)
    assert var_19 is True
    var_20 = '    very_long_function_name,'
    var_21 = bool('    very_long_function_name,' in var_17)
    assert var_21 is True


def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = 18
    var_3 = '    '
    var_4 = True
    var_5 = False
    var_6 = '  # '
    var_7 = 'line_length'
    var_8 = 'multi_line_output'
    var_9 = 'wrap_length'
    var_10 = 'indent'
    var_11 = 'use_parentheses'
    var_12 = 'include_trailing_comma'
    var_13 = 'comment_prefix'
    var_14 = {var_7: var_0, var_8: var_1, var_9: var_2, var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_6}
    var_15 = module_0.Config(**var_14)
    var_16 = 'import very_long_module_name as vlm'
    var_17 = '\n'
    var_18 = module_1.line(var_16, var_17, var_15)
    var_19 = 'import very_long_module_name as'
    var_20 = bool('import very_long_module_name as' in var_18)
    assert var_20 is True
    var_21 = '    vlm'
    var_22 = bool('    vlm' in var_18)
    assert var_22 is True


def test_case_0():
    var_0 = 30
    var_1 = 3
    var_2 = 28
    var_3 = '    '
    var_4 = True
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'wrap_length'
    var_9 = 'indent'
    var_10 = 'use_parentheses'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_4, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'very_long_module_name.very_long_submodule.very_long_function'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    var_18 = 'very_long_module_name.('
    var_19 = bool('very_long_module_name.(' in var_17)
    assert var_19 is True
    var_20 = '    very_long_submodule.very_long_function,'
    var_21 = bool('    very_long_submodule.very_long_function,' in var_17)
    assert var_21 is True


def test_case_0():
    var_0 = 30
    var_1 = 3
    var_2 = 28
    var_3 = '    '
    var_4 = True
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'wrap_length'
    var_9 = 'indent'
    var_10 = 'use_parentheses'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_4, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from module import something  # some comment'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    var_18 = 'from module import('
    var_19 = bool('from module import(' in var_17)
    assert var_19 is True
    var_20 = '    something,  # some comment'
    var_21 = bool('    something,  # some comment' in var_17)
    assert var_21 is True


def test_case_0():
    var_0 = 30
    var_1 = 3
    var_2 = 28
    var_3 = '    '
    var_4 = True
    var_5 = False
    var_6 = '  # '
    var_7 = 'line_length'
    var_8 = 'multi_line_output'
    var_9 = 'wrap_length'
    var_10 = 'indent'
    var_11 = 'use_parentheses'
    var_12 = 'include_trailing_comma'
    var_13 = 'comment_prefix'
    var_14 = {var_7: var_0, var_8: var_1, var_9: var_2, var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_6}
    var_15 = module_0.Config(**var_14)
    var_16 = 'from module import something  # noqa'
    var_17 = '\n'
    var_18 = module_1.line(var_16, var_17, var_15)
    var_19 = 'from module import(  # noqa'
    var_20 = bool('from module import(  # noqa' in var_18)
    assert var_20 is True
    var_21 = '    something'
    var_22 = bool('    something' in var_18)
    assert var_22 is True


def test_case_0():
    var_0 = 30
    var_1 = 5
    var_2 = 28
    var_3 = '    '
    var_4 = False
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'wrap_length'
    var_9 = 'indent'
    var_10 = 'use_parentheses'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_4, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from module import something'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    assert var_17 == 'from module import something  # NOQA'


def test_case_0():
    var_0 = 30
    var_1 = 5
    var_2 = 28
    var_3 = '    '
    var_4 = False
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'wrap_length'
    var_9 = 'indent'
    var_10 = 'use_parentheses'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_4, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from module import something  # NOQA'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    assert var_17 == 'from module import something  # NOQA'


def test_case_0():
    var_0 = 20
    var_1 = 4
    var_2 = 18
    var_3 = '    '
    var_4 = True
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'wrap_length'
    var_9 = 'indent'
    var_10 = 'use_parentheses'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_4, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from module import something'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    var_18 = 'from module import('
    var_19 = bool('from module import(' in var_17)
    assert var_19 is True
    var_20 = '    something,'
    var_21 = bool('    something,' in var_17)
    assert var_21 is True


def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = 18
    var_3 = '    '
    var_4 = False
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'wrap_length'
    var_9 = 'indent'
    var_10 = 'use_parentheses'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_4, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from module import something'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    var_18 = 'from module import\\'
    var_19 = bool('from module import\\' in var_17)
    assert var_19 is True
    var_20 = '    something'
    var_21 = bool('    something' in var_17)
    assert var_21 is True


def test_case_0():
    var_0 = 30
    var_1 = 3
    var_2 = 28
    var_3 = '    '
    var_4 = True
    var_5 = '  # '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'wrap_length'
    var_9 = 'indent'
    var_10 = 'use_parentheses'
    var_11 = 'include_trailing_comma'
    var_12 = 'comment_prefix'
    var_13 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_4, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from module import something  # comment'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    var_18 = 'from module import('
    var_19 = bool('from module import(' in var_17)
    assert var_19 is True
    var_20 = '    something,  # comment'
    var_21 = bool('    something,  # comment' in var_17)
    assert var_21 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_true. Retrieved 6/8 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 90
    var_4 = var_2 * var_3
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_1)
    var_7 = bool(var_6 == var_4 + var_1.comment_prefix + ' NOQA')
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_29_false. Retrieved 9/11 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'short_line'
    var_3 = len(var_2)
    var_4 = 2
    var_5 = var_3 + var_4
    var_6 = var_1.wrap_length
    var_7 = var_1.line_length
    var_8 = var_6 or var_7
    var_9 = var_5 > var_8
    assert var_9 is False



# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 7/11 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 7/11 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 8/12 statements.
# Partially parsed test_line_wrap_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_noqa_mode_with_existing_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_trailing_comma_and_comment. Retrieved 8/13 statements.
# Partially parsed test_line_wrap_with_noqa_comment_and_parentheses. Retrieved 9/14 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 80

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = 'from very_long_module_name import very_long_function_name'
    var_6 = '\n'
    var_7 = 'from very_long_module_name import'
    var_8 = 'very_long_function_name'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = 'module.submodule.verylongsubmodule.verylongfunction'
    var_5 = '\n'
    var_6 = 'module.submodule.'
    var_7 = 'verylongsubmodule.verylongfunction'
    var_8 = ','

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = 'import very_long_module_name as vlm'
    var_6 = '\n'
    var_7 = 'import very_long_module_name as'
    var_8 = 'vlm'
    var_9 = '\\'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = False
    var_5 = '  # '
    var_6 = 'import very_long_module  # some comment'
    var_7 = '\n'
    var_8 = 'import very_long_module'
    var_9 = '# some comment'

def test_case_0():
    var_0 = 10
    var_1 = '  # '
    var_2 = 'import very_long_module'
    var_3 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = '  # '
    var_2 = 'import module  # NOQA'
    var_3 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = '  # '
    var_5 = 'import very_long_module  # comment'
    var_6 = '\n'
    var_7 = 'import very_long_module'
    var_8 = '# comment'
    var_9 = ','

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = '    '
    var_3 = None
    var_4 = '  # '
    var_5 = 'import very_long_module  # noqa'
    var_6 = '\n'
    var_7 = 'import very_long_module'
    var_8 = '# noqa'
    var_9 = '('
    var_10 = ')'

def test_case_0():
    var_0 = 25
    var_1 = False
    var_2 = '    '
    var_3 = None
    var_4 = 'import very_long_module_name'
    var_5 = '\n'
    var_6 = 'import very_long_module_name'
    var_7 = '\\'



# Parsed testcases at query #28
#--------------------------





def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 80
    var_3 = 3
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'import os'


def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = True
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'from very_long_module_name import very_long_function_name'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = 'from very_long_module_name import('
    var_16 = bool('from very_long_module_name import(' in var_14)
    assert var_16 is True
    var_17 = 'very_long_function_name'
    var_18 = bool('very_long_function_name' in var_14)
    assert var_18 is True


def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = True
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'import very_long_module_name as very_long_alias'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = bool('import very_long_module_name as very_long_alias' in var_14 or 'import very_long_module_name as(' in var_14)
    assert var_15 is True


def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = True
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'very_long_module_name.very_long_submodule.very_long_function'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = 'very_long_module_name.very_long_submodule.very_long_function'
    var_16 = bool('very_long_module_name.very_long_submodule.very_long_function' == var_14)
    assert var_16 is True


def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = True
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'from module import something  # some comment'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = '  # some comment'
    var_16 = bool('  # some comment' in var_14)
    assert var_16 is True


def test_case_0():
    var_0 = 20
    var_1 = 5
    var_2 = False
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'import very_long_module_name_that_exceeds_length'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    assert var_14 == 'import very_long_module_name_that_exceeds_length  # NOQA'


def test_case_0():
    var_0 = 20
    var_1 = 5
    var_2 = False
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'import module  # NOQA'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    assert var_14 == 'import module  # NOQA'


def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = True
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'from module import something  # noqa'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = '  # noqa'
    var_16 = bool('  # noqa' in var_14)
    assert var_16 is True


def test_case_0():
    var_0 = 20
    var_1 = 4
    var_2 = True
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'from very_long_module import very_long_function'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = 'from very_long_module import('
    var_16 = bool('from very_long_module import(' in var_14)
    assert var_16 is True


def test_case_0():
    var_0 = 20
    var_1 = 5
    var_2 = True
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'from very_long_module import very_long_function'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = 'from very_long_module import('
    var_16 = bool('from very_long_module import(' in var_14)
    assert var_16 is True


def test_case_0():
    var_0 = 10
    var_1 = 3
    var_2 = True
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'import a'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    assert var_14 == 'import a'


def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = False
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'from very_long_module_name import very_long_function_name'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = '\\'
    var_16 = bool('\\' in var_14)
    assert var_16 is True


def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = True
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'from module import something  # comment'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = bool(',' in var_14 or '  # comment' in var_14)
    assert var_15 is True


def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = True
    var_3 = '  # '
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'wrap_length'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_0, var_7: var_2, var_8: var_2, var_9: var_3}
    var_11 = module_0.Config(**var_10)
    var_12 = 'cimport very_long_module_name'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = bool('cimport very_long_module_name' == var_14 or 'cimport(' in var_14)
    assert var_15 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_29_false. Retrieved 9/11 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'short_line'
    var_3 = len(var_2)
    var_4 = 2
    var_5 = var_3 + var_4
    var_6 = var_1.wrap_length
    var_7 = var_1.line_length
    var_8 = var_6 or var_7
    var_9 = var_5 > var_8
    assert var_9 is False



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 100
    var_1 = 'a'
    var_2 = 101
    var_3 = var_1 * var_2
    var_4 = '\n'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_29_false. Retrieved 9/11 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'short_line'
    var_3 = len(var_2)
    var_4 = 2
    var_5 = var_3 + var_4
    var_6 = var_1.wrap_length
    var_7 = var_1.line_length
    var_8 = var_6 or var_7
    var_9 = var_5 > var_8
    assert var_9 is False



# Parsed testcases at query #32
#--------------------------






# Parsed testcases at query #33
#--------------------------






# Parsed testcases at query #34
#--------------------------

# Partially parsed test_import_statement_explode_mode. Retrieved 12/15 statements.
# Partially parsed test_import_statement_grid_mode. Retrieved 13/16 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 16/19 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 19/30 statements.
# Partially parsed test_import_statement_single_line. Retrieved 11/14 statements.
# Partially parsed test_import_statement_custom_line_separator. Retrieved 13/16 statements.
# Partially parsed test_import_statement_ignore_comments. Retrieved 16/19 statements.
# Partially parsed test_import_statement_custom_wrap_length. Retrieved 13/19 statements.


def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = None
    var_6 = 80
    var_7 = False
    var_8 = '    '
    var_9 = '  # '
    var_10 = True
    var_11 = 'from module import (\n    a,\n    b,\n    c,\n)'

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = 'e'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = None
    var_8 = 40
    var_9 = False
    var_10 = '    '
    var_11 = '  # '
    var_12 = 'from module import (a, b, c, d,\n                   e)'

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'comment1'
    var_6 = 'comment2'
    var_7 = 'comment3'
    var_8 = [var_5, var_6, var_7]
    var_9 = None
    var_10 = 80
    var_11 = True
    var_12 = '    '
    var_13 = '  # '
    var_14 = False
    var_15 = 'from module import (\n    a,  # comment1\n    b,  # comment2\n    c,  # comment3\n)'

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = 'e'
    var_6 = 'f'
    var_7 = 'g'
    var_8 = 'h'
    var_9 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8]
    var_10 = None
    var_11 = 50
    var_12 = False
    var_13 = '    '
    var_14 = '  # '
    var_15 = True
    var_16 = '\n'
    var_17 = -1
    var_18 = -1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = None
    var_6 = 100
    var_7 = False
    var_8 = '    '
    var_9 = '  # '
    var_10 = 'from module import a, b, c'

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = None
    var_6 = 40
    var_7 = True
    var_8 = '    '
    var_9 = '  # '
    var_10 = False
    var_11 = '\r\n'
    var_12 = 'from module import (\r\n    a,\r\n    b,\r\n    c,\r\n)'

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'comment1'
    var_6 = 'comment2'
    var_7 = 'comment3'
    var_8 = [var_5, var_6, var_7]
    var_9 = None
    var_10 = 80
    var_11 = True
    var_12 = '    '
    var_13 = '  # '
    var_14 = False
    var_15 = 'from module import (\n    a,\n    b,\n    c,\n)'

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = 'e'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 30
    var_8 = 80
    var_9 = False
    var_10 = '    '
    var_11 = '  # '
    var_12 = '\n'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_import_statement_balanced_wrapping_predicate_false. Retrieved 25/46 statements.



def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = 'e'
    var_6 = 'f'
    var_7 = 'g'
    var_8 = 'h'
    var_9 = 'i'
    var_10 = 'j'
    var_11 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = []
    var_13 = '\n'
    var_14 = {}
    var_15 = module_0.Config(**var_14)
    var_16 = None
    var_17 = False
    var_18 = module_1.import_statement(var_0, var_11, var_12, var_13, var_15, var_16, var_17)
    var_19 = 1
    var_20 = -1
    var_21 = 0
    var_22 = -1
    var_23 = var_15.line_length
    var_24 = 10
    var_25 = var_23 > var_24



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_noqa_mode. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_noqa_mode_with_existing_noqa. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_with_noqa_comment_and_parentheses. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_vertical_hanging_indent. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 8/11 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 80
    var_1 = None
    var_2 = '    '
    var_3 = False
    var_4 = '# '
    var_5 = 'import os'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = False
    var_5 = '# '
    var_6 = 'from very_long_module_name import very_long_function_name'
    var_7 = '\n'
    var_8 = 'from very_long_module_name import('
    var_9 = 'very_long_function_name'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = False
    var_5 = '# '
    var_6 = 'module.submodule.verylongclassname.verylongmethodname'
    var_7 = '\n'
    var_8 = 'module.submodule.verylongclassname.('
    var_9 = 'verylongmethodname'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = False
    var_5 = '# '
    var_6 = 'import very_long_module_name as very_long_alias'
    var_7 = '\n'
    var_8 = 'import very_long_module_name as'
    var_9 = 'very_long_alias'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = False
    var_5 = '# '
    var_6 = 'import very_long_module_name  # some comment'
    var_7 = '\n'
    var_8 = 'import very_long_module_name  # some comment'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = False
    var_4 = '# '
    var_5 = 'import very_long_module_name'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = False
    var_4 = '# '
    var_5 = 'import very_long_module_name  # NOQA'
    var_6 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = '# '
    var_5 = 'from module import very_long_function_name'
    var_6 = '\n'
    var_7 = 'from module import('
    var_8 = 'very_long_function_name,'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = False
    var_5 = '# '
    var_6 = 'import very_long_module_name  # noqa'
    var_7 = '\n'
    var_8 = 'import very_long_module_name  # noqa'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = False
    var_5 = '# '
    var_6 = 'from module import very_long_function_name'
    var_7 = '\n'
    var_8 = 'from module import('
    var_9 = 'very_long_function_name'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = True
    var_4 = False
    var_5 = '# '
    var_6 = 'from module import very_long_function_name'
    var_7 = '\n'
    var_8 = 'from module import('
    var_9 = 'very_long_function_name'

def test_case_0():
    var_0 = 20
    var_1 = None
    var_2 = '    '
    var_3 = False
    var_4 = '# '
    var_5 = 'import very_long_module_name'
    var_6 = '\n'
    var_7 = 'import very_long_module_name'
    var_8 = '\\'



# Parsed testcases at query #37
#--------------------------






# Parsed testcases at query #38
#--------------------------






# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 12/28 statements.


def test_case_0():
    var_0 = 100
    var_1 = 'a'
    var_2 = 101
    var_3 = var_1 * var_2
    var_4 = '\n'
    var_5 = 99
    var_6 = var_1 * var_5
    var_7 = var_1 * var_5
    var_8 = var_1 * var_2
    var_9 = 'import '
    var_10 = var_1 * var_2
    var_11 = var_9 + var_10
    var_12 = 'NOQA'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_29_false. Retrieved 9/11 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'short_line'
    var_3 = len(var_2)
    var_4 = 2
    var_5 = var_3 + var_4
    var_6 = var_1.wrap_length
    var_7 = var_1.line_length
    var_8 = var_6 or var_7
    var_9 = var_5 > var_8
    assert var_9 is False



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_include_trailing_comma_with_parentheses_and_no_comma. Retrieved 4/11 statements.
# Partially parsed test_include_trailing_comma_with_parentheses_and_existing_comma. Retrieved 5/13 statements.
# Partially parsed test_include_trailing_comma_without_parentheses. Retrieved 6/15 statements.
# Partially parsed test_no_include_trailing_comma_with_parentheses. Retrieved 6/15 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import very_long_name_that_exceeds_line_length'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = ','
    var_6 = bool(',' in var_4)
    assert var_6 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import very_long_name_that_exceeds_line_length,'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = ','


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import very_long_name_that_exceeds_line_length'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = ','
    var_6 = 1


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import very_long_name_that_exceeds_line_length'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = ','
    var_6 = 0



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_65_false. Retrieved 14/27 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '\n'
    var_3 = 'from module import something'
    var_4 = 'import '
    var_5 = 'noqa'
    var_6 = '# noqa'
    var_7 = '    another_thing'
    var_8 = ','
    var_9 = '\n'
    var_10 = f'{var_3}{var_4}({var_6}{var_2}{var_7}{var_8}{var_9})'
    var_11 = var_1.comment_prefix
    var_12 = -1
    var_13 = -1
    var_14 = ')'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_at_line_65_false. Retrieved 14/27 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '\n'
    var_3 = 'from module import something'
    var_4 = 'import '
    var_5 = 'noqa'
    var_6 = '# noqa'
    var_7 = '    another_thing'
    var_8 = ','
    var_9 = '\n'
    var_10 = f'{var_3}{var_4}({var_6}{var_2}{var_7}{var_8}{var_9})'
    var_11 = var_1.comment_prefix
    var_12 = -1
    var_13 = -1
    var_14 = ')'



# Parsed testcases at query #44
#--------------------------






# Parsed testcases at query #45
#--------------------------






