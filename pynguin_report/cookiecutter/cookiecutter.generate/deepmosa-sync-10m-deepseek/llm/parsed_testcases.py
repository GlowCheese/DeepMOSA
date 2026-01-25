####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_3: var_1}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'existing': 'value'})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'existing'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = {var_5: var_2}
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_4, var_7, in_dictionary_variable=var_8)
    var_10 = bool(var_4 == {'nested': {'existing': 'value', 'new': 'value'}})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'choices': ['a', 'b']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'x'
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = bool(var_5 == {'choice': ['b', 'a', 'c']})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'x'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 3
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'nested': {'a': 1, 'b': 3}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'value': 'new'})
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/path/to/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_and_create_dir_with_existing_dir_no_overwrite. Retrieved 7/11 statements.
# Partially parsed test_render_and_create_dir_with_existing_dir_with_overwrite. Retrieved 6/9 statements.
# Partially parsed test_render_and_create_dir_with_new_dir. Retrieved 6/8 statements.
# Partially parsed test_render_and_create_dir_with_template. Retrieved 8/11 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'output_dir'
    var_3 = module_0.Environment()
    var_4 = False
    var_5 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = [var_0]
    var_2 = True
    var_3 = {}
    var_4 = '.'
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = module_1.render_and_create_dir(var_0, var_3, var_4, var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = [var_0]
    var_2 = True
    var_3 = {}
    var_4 = '.'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_0, var_3, var_4, var_5, var_2)
    var_7 = var_6[0]
    var_8 = var_6[1]
    assert var_8 is False

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'new_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = '.'
    var_4 = module_0.Environment()
    var_5 = False
    var_6 = module_1.render_and_create_dir(var_0, var_2, var_3, var_4, var_5)
    var_7 = var_6[0]
    var_8 = var_6[1]
    assert var_8 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'project'
    var_3 = {var_1: var_2}
    var_4 = '{{ name }}'
    var_5 = '.'
    var_6 = False
    var_7 = module_1.render_and_create_dir(var_4, var_3, var_5, var_0, var_6)
    var_8 = [var_2]
    var_9 = var_7[0]
    var_10 = var_7[1]
    assert var_10 is True
    var_11 = [var_2]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_render_and_create_dir_overwrites_existing_dir_when_overwrite_flag_is_true. Retrieved 4/13 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = True



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'readme.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'docs/index.md'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'main.py'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is False
    var_9 = 'src/utils.py'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'any_path.txt'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False



# Parsed testcases at query #6
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'new_value'
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.apply_overwrites_to_context(var_4, var_6, in_dictionary_variable=var_7)
    var_9 = var_4['key']
    assert var_9 == 'new_value'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 8/19 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 4/10 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 10/21 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 10/21 statements.
# Partially parsed test_generate_context_with_default_and_extra_context. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'w'
    var_1 = False
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter'
    var_6 = (var_2, var_3)
    var_7 = [var_6]
    var_8 = [var_7]

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'w'
    var_1 = False
    var_2 = 'invalid json'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'w'
    var_1 = False
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'new_value'
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'
    var_8 = (var_2, var_5)
    var_9 = [var_8]
    var_10 = [var_9]

def test_case_0():
    var_0 = 'w'
    var_1 = False
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'new_value'
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'
    var_8 = (var_2, var_5)
    var_9 = [var_8]
    var_10 = [var_9]

def test_case_0():
    var_0 = 'w'
    var_1 = False
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'default_value'
    var_6 = {var_2: var_5}
    var_7 = 'extra_value'
    var_8 = {var_2: var_7}
    var_9 = 'cookiecutter'
    var_10 = (var_2, var_7)
    var_11 = [var_10]
    var_12 = [var_11]



# Parsed testcases at query #8
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'src/templates/index.html'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'src/templates/*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'src/templates/index.html'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'src/static/*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'src/templates/index.html'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.is_copy_only_path(var_0, var_3)
    assert var_4 is False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 10/16 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 10/16 statements.
# Partially parsed test_generate_file_renders_text_file. Retrieved 12/18 statements.
# Partially parsed test_generate_file_returns_if_file_name_is_empty. Retrieved 9/13 statements.
# Partially parsed test_generate_file_applies_file_permissions. Retrieved 12/20 statements.
# Partially parsed test_generate_file_handles_new_lines. Retrieved 12/18 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'file.txt'
    var_8 = 'existing content'
    assert var_8 == 'existing content'
    var_9 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/binary.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = b'\x00\x01\x02\x03'
    assert var_7 == b'\x00\x01\x02\x03'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_9 = 'binary.bin'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = '{{ cookiecutter.var }}'
    assert var_9 == 'value'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_11 = 'file.txt'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_8 = ''

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'content'
    var_8 = 420
    var_9 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_10 = 'file.txt'
    var_11 = 511

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = 'line1\nline2'
    assert var_9 == 'line1\r\nline2'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_11 = 'file.txt'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_already_exists. Retrieved 12/18 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'example.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = True
    var_10 = 'existing content'
    var_11 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)



# Parsed testcases at query #11
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generate_file_creates_file_with_correct_content. Retrieved 11/16 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 12/19 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 11/16 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 10/12 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = 'Project: {{ cookiecutter.name }}'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = 'Project: {{ cookiecutter.name }}'
    var_10 = 'Existing content'
    var_11 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary_file.bin'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = b'\x00\x01\x02\x03'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7)



# Parsed testcases at query #13
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/fixtures/invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 12/20 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 10/16 statements.
# Partially parsed test_generate_file_renders_text_file. Retrieved 12/18 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 8/11 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = True
    var_8 = 'test content'
    var_9 = 'file.txt'
    var_10 = 'existing content'
    assert var_10 == 'existing content'
    var_11 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = b'\x00\x01\x02\x03'
    assert var_7 == b'\x00\x01\x02\x03'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_9 = 'file.bin'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = '{{ cookiecutter.variable }}'
    assert var_9 == 'value'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_11 = 'file.txt'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5)



# Parsed testcases at query #15
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_is_binary_predicate_evaluates_to_true. Retrieved 11/21 statements.


def test_case_0():
    var_0 = b'\x00\x01\x02\x03'
    var_1 = 'MockEnv'
    var_2 = ()
    var_3 = 'from_string'
    var_4 = 'MockTemplate'
    var_5 = ()
    var_6 = 'render'
    var_7 = lambda self, x: type(var_4, var_5, {var_6: lambda self, **kwargs: x})()
    var_8 = {var_3: var_7}
    var_9 = [var_1, var_2, var_8]
    var_10 = {}
    var_11 = False



# Parsed testcases at query #17
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.generate_context(default_context=var_2)
    var_4 = var_3['cookiecutter']['key']
    assert var_4 == 'value'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_template_syntax_error_has_translated_disabled. Retrieved 1/7 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generate_file_skips_when_file_exists. Retrieved 9/15 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 8/13 statements.
# Partially parsed test_generate_file_renders_text_file. Retrieved 12/17 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'existing content'
    assert var_7 == 'existing content'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = b'\x00\x01\x02\x03'
    assert var_6 == b'\x00\x01\x02\x03'
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'Hello {{ cookiecutter.name }}'
    assert var_10 == 'Hello Test'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_context_handles_invalid_json. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'invalid json'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_generate_context_handles_invalid_json. Retrieved 1/12 statements.


def test_case_0():
    var_0 = '{"invalid": json}'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 10/16 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 10/16 statements.
# Partially parsed test_generate_file_renders_text_file. Retrieved 12/18 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 8/11 statements.
# Partially parsed test_generate_file_applies_new_line_setting. Retrieved 12/18 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'file.txt'
    var_8 = 'existing content'
    assert var_8 == 'existing content'
    var_9 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = b'\x00\x01\x02'
    assert var_7 == b'\x00\x01\x02'
    var_8 = 'file.bin'
    var_9 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = '{{ cookiecutter.var }}'
    assert var_9 == 'value'
    var_10 = 'file.txt'
    var_11 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = 'content'
    assert var_9 == 'content\r\n'
    var_10 = 'file.txt'
    var_11 = module_1.generate_file(var_0, var_1, var_6, var_7)



# Parsed testcases at query #23
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1
import binaryornot.check as module_2

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = '/path/to/binary/file'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_5 = module_2.is_binary(var_1)
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #24
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'output_dir'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ' '
    var_1 = {}
    var_2 = 'output_dir'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #25
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_choice'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #26
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_choice'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 7/12 statements.
# Partially parsed test_generate_file_text. Retrieved 9/19 statements.
# Partially parsed test_generate_file_skip_exists. Retrieved 8/16 statements.
# Partially parsed test_generate_file_empty_name. Retrieved 6/9 statements.
# Partially parsed test_generate_file_new_lines. Retrieved 11/20 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.bin'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = b'\x00\x01\x02\x03'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 'Hello {{ name }}'
    var_6 = {var_1: var_5}
    var_7 = True
    var_8 = 'Hello {{ name }}'
    assert var_8 == 'Hello test'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'content'
    var_6 = 'existing'
    assert var_6 == 'existing'
    var_7 = module_1.generate_file(var_0, var_1, var_2, var_3, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = module_1.generate_file(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'line1\nline2'
    var_8 = {var_1: var_7}
    var_9 = True
    var_10 = 'line1\nline2'
    assert var_10 == b'line1\r\nline2'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_render_and_create_dir_raises_empty_dir_name_exception_for_empty_dirname. Retrieved 6/9 statements.
# Partially parsed test_render_and_create_dir_raises_empty_dir_name_exception_for_whitespace_dirname. Retrieved 6/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/output'
    var_5 = [var_4]
    var_6 = module_0.Environment()
    var_7 = bool(False)
    assert var_7 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = '   '
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/output'
    var_5 = [var_4]
    var_6 = module_0.Environment()
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_new_lines_in_context. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_already_exists. Retrieved 9/14 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'w'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_true. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_generate_context_with_valid_json_file. Retrieved 4/12 statements.
# Partially parsed test_generate_context_with_invalid_json_file. Retrieved 3/10 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/14 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/14 statements.
# Partially parsed test_generate_context_with_default_and_extra_context. Retrieved 8/16 statements.
# Partially parsed test_generate_context_with_nested_default_context. Retrieved 9/17 statements.
# Partially parsed test_generate_context_with_nested_extra_context. Retrieved 9/17 statements.
# Partially parsed test_generate_context_with_multichoice_default_context. Retrieved 8/16 statements.
# Partially parsed test_generate_context_with_multichoice_extra_context. Retrieved 8/16 statements.
# Partially parsed test_generate_context_with_boolean_default_context. Retrieved 6/14 statements.
# Partially parsed test_generate_context_with_boolean_extra_context. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = module_0.generate_context(var_0)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 'new_value'
    var_5 = {var_0: var_4}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 'new_value'
    var_5 = {var_0: var_4}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 'default_value'
    var_5 = {var_0: var_4}
    var_6 = 'extra_value'
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 'key'
    var_1 = 'nested_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 'new_value'
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}

def test_case_0():
    var_0 = 'key'
    var_1 = 'nested_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 'new_value'
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = [var_2]
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = [var_2]
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 'key'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 'yes'
    var_5 = {var_0: var_4}

def test_case_0():
    var_0 = 'key'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 'yes'
    var_5 = {var_0: var_4}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_47_evaluates_to_true. Retrieved 10/18 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/binary_file.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = None
    var_9 = module_1.generate_file(var_0, var_1, var_4, var_5)



# Parsed testcases at query #34
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = '/tmp'
    var_3 = ''
    var_4 = module_1.render_and_create_dir(var_3, var_1, var_2, var_0)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = '/tmp'
    var_3 = '   '
    var_4 = module_1.render_and_create_dir(var_3, var_1, var_2, var_0)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_generate_files_with_empty_context. Retrieved 4/9 statements.
# Partially parsed test_generate_files_with_overwrite_existing. Retrieved 9/16 statements.
# Partially parsed test_generate_files_with_skip_existing. Retrieved 9/16 statements.
# Partially parsed test_generate_files_with_hooks_disabled. Retrieved 9/14 statements.
# Partially parsed test_generate_files_with_keep_on_failure. Retrieved 9/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'test_output'
    var_2 = [var_1]
    var_3 = True
    var_4 = 'test_repo'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_output'
    var_6 = [var_5]
    var_7 = True
    var_8 = 'test_repo'
    var_9 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_output'
    var_6 = [var_5]
    var_7 = True
    var_8 = 'test_repo'
    var_9 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_output'
    var_6 = [var_5]
    var_7 = True
    var_8 = 'test_repo'
    var_9 = False

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'invalid_var'
    var_2 = '{{ invalid }}'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_output'
    var_6 = [var_5]
    var_7 = True
    var_8 = 'test_repo'
    var_9 = True
    var_10 = bool((output_dir / 'test_output').exists())
    assert var_10 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_delete_project_on_failure_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_generate_file_skips_when_file_exists. Retrieved 7/11 statements.
# Partially parsed test_generate_file_copies_binary_files. Retrieved 6/10 statements.
# Partially parsed test_generate_file_renders_text_files. Retrieved 8/15 statements.
# Partially parsed test_generate_file_preserves_file_permissions. Retrieved 9/17 statements.
# Partially parsed test_generate_file_handles_newlines. Retrieved 10/17 statements.
# Partially parsed test_generate_file_handles_template_syntax_error. Retrieved 6/12 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'w'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3, var_4)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = module_1.generate_file(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.bin'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = b'\x00\x01\x02'
    var_5 = module_1.generate_file(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 'Hello {{ name }}'
    var_6 = {var_1: var_5}
    var_7 = 'Hello {{ name }}'
    assert var_7 == 'Hello test'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'test'
    var_5 = 420
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_7 = 511
    var_8 = oct(var_5)

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'line1\nline2'
    var_8 = {var_1: var_7}
    var_9 = 'line1\nline2'
    assert var_9 == b'line1\r\nline2'

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
    var_2 = {}
    var_3 = 'Hello {{'
    var_4 = {var_1: var_3}
    var_5 = 'Hello {{'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #38
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_generate_file_creates_file_with_rendered_content. Retrieved 13/20 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 14/22 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 13/20 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = '{{ cookiecutter.variable }}'
    assert var_11 == 'value'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = '{{ cookiecutter.variable }}'
    var_12 = 'existing content'
    assert var_12 == 'existing content'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary.bin'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = b'\x00\x01\x02\x03'
    assert var_11 == b'\x00\x01\x02\x03'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_generate_file_skips_existing_file_when_skip_flag_is_true. Retrieved 10/16 statements.
# Partially parsed test_generate_file_copies_binary_file_without_rendering. Retrieved 9/15 statements.
# Partially parsed test_generate_file_renders_text_file_with_context. Retrieved 11/20 statements.
# Partially parsed test_generate_file_uses_configured_newline_character. Retrieved 11/20 statements.
# Partially parsed test_generate_file_detects_original_newline_character. Retrieved 9/18 statements.
# Partially parsed test_generate_file_skips_empty_filename. Retrieved 9/15 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'output.txt'
    var_8 = 'existing content'
    assert var_8 == 'existing content'
    var_9 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary.dat'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = b'\x00\x01\x02\x03'
    assert var_7 == b'\x00\x01\x02\x03'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'Project: {{ cookiecutter.name }}'
    var_8 = {var_1: var_7}
    var_9 = True
    var_10 = 'Project: {{ cookiecutter.name }}'
    assert var_10 == 'Project: Test Project'

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'Line1\nLine2'
    var_8 = {var_1: var_7}
    var_9 = True
    var_10 = 'Line1\nLine2'
    assert var_10 == b'Line1\r\nLine2'

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'Line1\nLine2'
    var_6 = {var_1: var_5}
    var_7 = True
    var_8 = 'Line1\nLine2'
    assert var_8 == b'Line1\rLine2'

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'content'
    var_6 = {var_1: var_5}
    var_7 = True
    var_8 = ''



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 6/12 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 2/8 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 10/16 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 10/16 statements.
# Partially parsed test_generate_context_with_dict_overwrite. Retrieved 11/17 statements.
# Partially parsed test_generate_context_with_invalid_default_context. Retrieved 6/13 statements.
# Partially parsed test_generate_context_with_boolean_overwrite. Retrieved 6/12 statements.
# Partially parsed test_generate_context_with_invalid_boolean_overwrite. Retrieved 6/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.generate_context(var_2)
    var_6 = bool(var_5 == {'cookiecutter': var_4})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = 'value3'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'new_value1'
    var_8 = {var_0: var_7, var_1: var_3}
    var_9 = module_0.generate_context(var_2, var_8)
    var_10 = var_9['cookiecutter']['key1']
    assert var_10 == 'new_value1'
    var_11 = var_9['cookiecutter']['key2']
    var_12 = bool(var_9['cookiecutter']['key2'] == ['value2', 'value3'])
    assert var_12 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = 'value3'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'new_value1'
    var_8 = {var_0: var_7, var_1: var_3}
    var_9 = module_0.generate_context(var_2, extra_context=var_8)
    var_10 = var_9['cookiecutter']['key1']
    assert var_10 == 'new_value1'
    var_11 = var_9['cookiecutter']['key2']
    var_12 = bool(var_9['cookiecutter']['key2'] == ['value2', 'value3'])
    assert var_12 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'subkey1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = 'value2'
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'new_value1'
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.generate_context(var_2, extra_context=var_9)
    var_11 = var_10['cookiecutter']['key1']['subkey1']
    assert var_11 == 'new_value1'
    var_12 = var_10['cookiecutter']['key2']
    assert var_12 == 'value2'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'invalid_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.generate_context(var_3, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.generate_context(var_3, extra_context=var_4)
    var_6 = var_5['cookiecutter']['key1']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.generate_context(var_0, extra_context=var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false_when_output_directory_not_created. Retrieved 2/4 statements.
# Partially parsed test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = True

def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #44
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 3/10 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 2/8 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 5/12 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 5/12 statements.
# Partially parsed test_generate_context_with_default_and_extra_context. Retrieved 7/14 statements.
# Partially parsed test_generate_context_with_invalid_default_context. Retrieved 9/16 statements.
# Partially parsed test_generate_context_with_invalid_extra_context. Retrieved 9/16 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'key'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'default_value'
    var_4 = {var_0: var_3}
    var_5 = 'extra_value'
    var_6 = {var_0: var_5}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'choice1'
    var_2 = 'choice2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'key'
    var_6 = 'invalid_choice'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'choice1'
    var_2 = 'choice2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'key'
    var_6 = 'invalid_choice'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_true. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #48
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'invalid_choice'
    var_2 = var_0.process_response(var_1)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 5/7 statements.
# Partially parsed test_generate_context_with_default. Retrieved 7/9 statements.
# Partially parsed test_generate_context_with_extra. Retrieved 7/9 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 9/11 statements.
# Partially parsed test_generate_context_with_list_overwrite. Retrieved 8/10 statements.
# Partially parsed test_generate_context_with_boolean_conversion. Retrieved 7/9 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 5/8 statements.
# Partially parsed test_generate_context_with_invalid_list_overwrite. Retrieved 8/11 statements.
# Partially parsed test_generate_context_with_invalid_boolean. Retrieved 7/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = None
    var_2 = None
    var_3 = '{"key": "value"}'
    var_4 = module_0.generate_context(var_0, var_1, var_2)
    var_5 = bool(var_4 == {'test': {'key': 'value'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'new_value'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = '{"key": "value"}'
    var_6 = module_0.generate_context(var_0, var_3, var_4)
    var_7 = bool(var_6 == {'test': {'key': 'new_value'}})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = None
    var_2 = 'key'
    var_3 = 'extra_value'
    var_4 = {var_2: var_3}
    var_5 = '{"key": "value"}'
    var_6 = module_0.generate_context(var_0, var_1, var_4)
    var_7 = bool(var_6 == {'test': {'key': 'extra_value'}})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = None
    var_2 = 'nested'
    var_3 = 'key'
    var_4 = 'nested_value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{"nested": {"key": "value"}}'
    var_8 = module_0.generate_context(var_0, var_1, var_6)
    var_9 = bool(var_8 == {'test': {'nested': {'key': 'nested_value'}}})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = None
    var_2 = 'choices'
    var_3 = 'new_choice'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = '{"choices": ["choice1", "choice2"]}'
    var_7 = module_0.generate_context(var_0, var_1, var_5)
    var_8 = bool(var_7 == {'test': {'choices': ['new_choice']}})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = None
    var_2 = 'flag'
    var_3 = 'yes'
    var_4 = {var_2: var_3}
    var_5 = '{"flag": false}'
    var_6 = module_0.generate_context(var_0, var_1, var_4)
    var_7 = bool(var_6 == {'test': {'flag': True}})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = None
    var_2 = None
    var_3 = 'invalid json'
    var_4 = module_0.generate_context(var_0, var_1, var_2)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = None
    var_2 = 'choices'
    var_3 = 'invalid_choice'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = '{"choices": ["choice1", "choice2"]}'
    var_7 = module_0.generate_context(var_0, var_1, var_5)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = None
    var_2 = 'flag'
    var_3 = 'invalid'
    var_4 = {var_2: var_3}
    var_5 = '{"flag": false}'
    var_6 = module_0.generate_context(var_0, var_1, var_4)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generate_context_with_none_default_context. Retrieved 2/3 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.generate_context(default_context=var_0)



# Parsed testcases at query #3
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_3: var_1}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'existing': 'value'})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.apply_overwrites_to_context(var_2, var_5, in_dictionary_variable=var_6)
    var_8 = bool(var_2 == {'existing': {}, 'new': 'value'})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'choices': ['a', 'b']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'x'
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = bool(var_5 == {'choice': ['b', 'a', 'c']})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'x'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 3
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'nested': {'a': 1, 'b': 3}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'key': 'new'})
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_render_and_create_dir_creates_directory_when_not_exist. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_raises_exception_when_dirname_empty. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_raises_exception_when_dir_exists. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_dir. Retrieved 4/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = True



# Parsed testcases at query #6
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = '1'
    var_2 = var_0.process_response(var_1)
    assert var_2 is True
    var_3 = 'true'
    var_4 = var_0.process_response(var_3)
    assert var_4 is True
    var_5 = 't'
    var_6 = var_0.process_response(var_5)
    assert var_6 is True
    var_7 = 'yes'
    var_8 = var_0.process_response(var_7)
    assert var_8 is True
    var_9 = 'y'
    var_10 = var_0.process_response(var_9)
    assert var_10 is True
    var_11 = 'on'
    var_12 = var_0.process_response(var_11)
    assert var_12 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_generate_context_json_decoding_error. Retrieved 3/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 'tmp_invalid.json'
    var_2 = module_0.generate_context(var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_generate_context_with_valid_json_file. Retrieved 4/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-context.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'test_stem'
    var_3 = bool('test_stem' in var_1)
    assert var_3 is True
    var_4 = 'test_stem'
    var_5 = var_1[var_4]

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid-context.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-context.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = var_4['test_stem']['key1']
    assert var_5 == 'value1'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-context.json'
    var_1 = 'key2'
    var_2 = 'value2'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = var_4['test_stem']['key2']
    assert var_5 == 'value2'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-context.json'
    var_1 = 'invalid_key'
    var_2 = 'invalid_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(False)
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-context.json'
    var_1 = 'invalid_key'
    var_2 = 'invalid_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_3: var_1}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'existing': 'value'})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = {var_5: var_2}
    var_7 = True
    var_8 = module_0.apply_overwrites_to_context(var_4, var_6, in_dictionary_variable=var_7)
    var_9 = bool(var_4 == {'existing': {'nested': 'value'}, 'new': 'value'})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'list_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = var_5['list_var']
    var_10 = bool(var_5['list_var'] == ['a', 'b'])
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'list_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['choice_var']
    var_9 = bool(var_5['choice_var'] == ['b', 'a', 'c'])
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 3
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = var_6['dict_var']
    var_12 = bool(var_6['dict_var'] == {'a': 1, 'b': 3})
    assert var_12 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['bool_var']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['var']
    assert var_6 == 'new'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_generate_context_with_invalid_json. Retrieved 1/12 statements.


def test_case_0():
    var_0 = '{"invalid": json}'



# Parsed testcases at query #11
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'invalid'
    var_2 = False
    var_3 = var_0.process_response(var_1)
    var_4 = True
    var_5 = bool(var_4)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 9/22 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 2/11 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 11/24 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 11/24 statements.
# Partially parsed test_generate_context_with_invalid_default_context. Retrieved 14/27 statements.
# Partially parsed test_generate_context_with_boolean_conversion. Retrieved 7/20 statements.
# Partially parsed test_generate_context_with_multichoice_overwrite. Retrieved 10/23 statements.
# Partially parsed test_generate_context_with_dict_overwrite. Retrieved 12/25 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'choice1'
    var_4 = 'choice2'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = module_0.generate_context(var_2)
    var_8 = 0

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'choice1'
    var_4 = 'choice2'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'new_value'
    var_8 = {var_0: var_7}
    var_9 = module_0.generate_context(var_2, var_8)
    var_10 = 0

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'choice1'
    var_4 = 'choice2'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'extra_value'
    var_8 = {var_0: var_7}
    var_9 = module_0.generate_context(var_2, extra_context=var_8)
    var_10 = 0

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    assert var_2 == 1
    var_3 = 'choice1'
    var_4 = 'choice2'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'invalid_key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = module_0.generate_context(var_0, var_9)
    var_11 = 0
    var_12 = var_4.message
    var_13 = str(var_12)
    var_14 = 'Invalid default received'
    var_15 = bool('Invalid default received' in var_13)
    assert var_15 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_key'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.generate_context(var_3, extra_context=var_4)
    var_6 = 0

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.generate_context(var_2, extra_context=var_7)
    var_9 = 0

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_value'
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.generate_context(var_2, extra_context=var_9)
    var_11 = 0



# Parsed testcases at query #13
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'variable'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = 'e'
    var_8 = [var_6, var_7]
    var_9 = {var_0: var_8}
    var_10 = True
    var_11 = module_0.apply_overwrites_to_context(var_5, var_9, in_dictionary_variable=var_10)
    var_12 = var_5['variable']
    var_13 = bool(var_5['variable'] == ['d', 'e'])
    assert var_13 is True



# Parsed testcases at query #14
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_apply_overwrites_to_context_line_46. Retrieved 9/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'subkey'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_2, var_6)
    var_8 = var_2[var_0]
    var_9 = var_2['key1']
    var_10 = bool(var_2['key1'] == {'subkey': 'value'})
    assert var_10 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_render_and_create_dir_raises_empty_dir_name_exception. Retrieved 4/7 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = [var_1]
    var_3 = module_0.Environment()
    var_4 = ''



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 4/6 statements.
# Partially parsed test_render_and_create_dir_raises_exception_for_existing_directory. Retrieved 6/9 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 5/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = [var_2, var_0]
    var_5 = True
    var_6 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = [var_2, var_0]
    var_5 = True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_apply_overwrites_to_context_dict_overwrite_false_case. Retrieved 9/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'not_a_dict'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = var_4[var_0]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generate_files_creates_project_directory. Retrieved 8/9 statements.
# Partially parsed test_generate_files_handles_existing_output_dir_with_overwrite. Retrieved 9/12 statements.
# Partially parsed test_generate_files_skips_existing_files_when_configured. Retrieved 11/18 statements.
# Partially parsed test_generate_files_executes_pre_and_post_hooks. Retrieved 14/23 statements.
# Partially parsed test_generate_files_keeps_project_on_failure_when_configured. Retrieved 9/10 statements.
# Partially parsed test_generate_files_copies_non_rendered_files. Retrieved 14/21 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)
    var_8 = 'test_project'
    var_9 = bool('test_project' in var_7)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output'
    var_7 = 'existing.txt'
    var_8 = True
    var_9 = 'existing content'
    assert var_9 == 'existing content'
    var_10 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output'
    var_7 = 'hooks'
    var_8 = 'pre_gen_project.py'
    var_9 = 'post_gen_project.py'
    var_10 = 'print("pre hook executed")'
    var_11 = 'print("post hook executed")'
    var_12 = True
    var_13 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_12)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'invalid_var'
    var_3 = '{{ invalid_var }}'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, keep_project_on_failure=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_copy_without_render'
    var_4 = 'test_project'
    var_5 = '*.bin'
    var_6 = [var_5]
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = {var_1: var_7}
    var_9 = '/tmp/output'
    var_10 = 'cookiecutter-{{ project_name }}'
    var_11 = 'test.bin'
    var_12 = b'\x00\x01\x02\x03'
    var_13 = module_0.generate_files(var_0, var_8, var_9)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_files_creates_project_directory. Retrieved 8/10 statements.
# Partially parsed test_generate_files_overwrites_existing_directory. Retrieved 9/13 statements.
# Partially parsed test_generate_files_skips_existing_files. Retrieved 11/17 statements.
# Partially parsed test_generate_files_fails_if_output_dir_exists. Retrieved 8/11 statements.
# Partially parsed test_generate_files_executes_hooks. Retrieved 10/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/output/dir'
    var_7 = module_0.generate_files(var_0, var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/output/dir'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/output/dir'
    var_7 = 'content'
    var_8 = True
    var_9 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_8)
    var_10 = 'existing_file.txt'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/output/dir'
    var_7 = module_0.generate_files(var_0, var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/output/dir'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_7)
    var_9 = 'hook_created_file.txt'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/output/dir'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, keep_project_on_failure=var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(os.path.exists(os.path.join(var_6, 'my_project')))
    assert var_10 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_render_and_create_dir_overwrites_existing_dir_when_overwrite_if_exists_is_true. Retrieved 5/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = '/tmp/test_output'
    var_1 = [var_0]
    var_2 = 'test_dir'
    var_3 = {}
    var_4 = module_0.Environment()
    var_5 = True



# Parsed testcases at query #22
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_choice'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_output_directory_created_and_not_keep_project_on_failure. Retrieved 2/4 statements.
# Partially parsed test_output_directory_not_created_and_not_keep_project_on_failure. Retrieved 2/4 statements.
# Partially parsed test_output_directory_created_and_keep_project_on_failure. Retrieved 2/4 statements.
# Partially parsed test_output_directory_not_created_and_keep_project_on_failure. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False

def test_case_0():
    var_0 = False
    var_1 = False

def test_case_0():
    var_0 = True
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    var_0 = True
    assert var_0 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_render_and_create_dir_raises_on_empty_dirname. Retrieved 4/7 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 7/11 statements.
# Partially parsed test_render_and_create_dir_raises_on_existing_dir. Retrieved 8/13 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_dir. Retrieved 8/14 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = [var_1]
    var_3 = module_0.Environment()
    var_4 = ''
    var_5 = bool(False)
    assert var_5 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = [var_3]
    var_5 = module_0.Environment()
    var_6 = '{{ name }}'
    var_7 = '/tmp/test'
    var_8 = [var_7]

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = [var_3]
    var_5 = module_0.Environment()
    var_6 = '/tmp/test'
    var_7 = [var_6]
    var_8 = True
    var_9 = '{{ name }}'
    var_10 = bool(False)
    assert var_10 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = [var_3]
    var_5 = module_0.Environment()
    var_6 = '/tmp/test'
    var_7 = [var_6]
    var_8 = True
    var_9 = '{{ name }}'
    var_10 = [var_6]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 5/8 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'test_output'
    var_3 = [var_2]
    var_4 = module_0.Environment()
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_render_and_create_dir_overwrite_existing. Retrieved 6/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/tmp/test_output'
    var_2 = [var_1]
    var_3 = module_0.Environment()
    var_4 = 'test_dir'
    var_5 = False
    var_6 = True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_overwrite_existing_directory. Retrieved 7/15 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = '/tmp/test_output'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'existing_dir'
    var_4 = 'existing_dir'
    var_5 = {}
    var_6 = module_0.Environment()
    var_7 = True



# Parsed testcases at query #29
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'some/file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'some/file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.csv'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'some/file.txt'
    var_1 = {}
    var_2 = module_0.is_copy_only_path(var_0, var_1)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'some/file.txt'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.is_copy_only_path(var_0, var_3)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'some/file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.csv'
    var_4 = '*.txt'
    var_5 = '*.json'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = module_0.is_copy_only_path(var_0, var_8)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'some/directory'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'some/*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True



# Parsed testcases at query #30
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'readme.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.md'
    var_3 = 'images/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'readme.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'readme.txt'
    var_2 = module_0.is_copy_only_path(var_1, var_0)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'readme.txt'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'docs/*'
    var_3 = 'static/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'docs/index.html'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'docs/*'
    var_3 = 'static/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'src/main.py'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is False



# Parsed testcases at query #31
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 4/6 statements.
# Partially parsed test_render_and_create_dir_raises_exception_for_existing_directory. Retrieved 8/10 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 5/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = False
    var_5 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_6 = True
    var_7 = bool(var_6)
    assert var_7 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = [var_2, var_0]
    var_5 = True
    var_6 = False
    var_7 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_8 = True
    var_9 = bool(var_8)
    assert var_9 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = [var_2, var_0]
    var_5 = True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_delete_project_on_failure_evaluates_to_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_delete_project_on_failure_evaluates_to_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 9/18 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 2/9 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 13/21 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 13/21 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = var_2[var_0]
    var_5 = '.'
    var_6 = var_4.split(var_5)[var_3]
    var_7 = (var_6, var_2)
    var_8 = [var_7]
    var_9 = [var_8]

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.generate_context(var_3, var_4)
    var_6 = 0
    var_7 = var_2[var_0]
    var_8 = '.'
    var_9 = var_7.split(var_8)[var_6]
    var_10 = {var_0: var_3}
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = [var_12]

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'extra_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.generate_context(var_3, extra_context=var_4)
    var_6 = 0
    var_7 = var_2[var_0]
    var_8 = '.'
    var_9 = var_7.split(var_8)[var_6]
    var_10 = {var_0: var_3}
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = [var_12]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 13/16 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 8/11 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 8/11 statements.
# Partially parsed test_generate_context_with_both_default_and_extra_context. Retrieved 13/16 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key1'
    var_8 = 'key2'
    var_9 = 'default1'
    var_10 = 'default2'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = module_0.generate_context(var_0, var_3, var_6)
    var_13 = var_12['test']
    var_14 = bool(var_12['test'] == {'key1': 'value1', 'key2': 'value2'})
    assert var_14 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_invalid.json'
    var_1 = 'invalid json'
    var_2 = module_0.generate_context(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key1'
    var_5 = 'default1'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_0, var_3)
    var_8 = var_7['test']
    var_9 = bool(var_7['test'] == {'key1': 'value1'})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key2'
    var_2 = 'value2'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'default2'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_0, extra_context=var_3)
    var_8 = var_7['test']
    var_9 = bool(var_7['test'] == {'key2': 'value2'})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key1'
    var_8 = 'key2'
    var_9 = 'default1'
    var_10 = 'default2'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = module_0.generate_context(var_0, var_3, var_6)
    var_13 = var_12['test']
    var_14 = bool(var_12['test'] == {'key1': 'value1', 'key2': 'value2'})
    assert var_14 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_generate_files_with_accept_hooks_false. Retrieved 4/7 statements.


def test_case_0():
    var_0 = {}
    var_1 = '.'
    var_2 = 'test_repo'
    var_3 = [var_2]
    var_4 = False



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_generate_files_creates_project_directory. Retrieved 8/10 statements.
# Partially parsed test_generate_files_with_existing_output_dir_and_overwrite. Retrieved 9/12 statements.
# Partially parsed test_generate_files_with_existing_output_dir_without_overwrite. Retrieved 9/11 statements.
# Partially parsed test_generate_files_with_copy_only_path. Retrieved 11/17 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 12/19 statements.
# Partially parsed test_generate_files_with_hooks_and_failure. Retrieved 12/18 statements.
# Partially parsed test_generate_files_with_hooks_and_failure_keep_project. Retrieved 12/18 statements.
# Partially parsed test_generate_files_with_skip_if_file_exists. Retrieved 11/17 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = False
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)
    var_9 = bool(True)
    assert var_9 is True
    var_10 = bool(False)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_copy_without_render'
    var_4 = 'test_project'
    var_5 = 'copy_me'
    var_6 = [var_5]
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = {var_1: var_7}
    var_9 = 'test_output'
    var_10 = module_0.generate_files(var_0, var_8, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = 'hooks'
    var_8 = 'pre_gen_project.py'
    var_9 = 'print("Hello from hook")'
    var_10 = True
    var_11 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_10)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = 'hooks'
    var_8 = 'pre_gen_project.py'
    var_9 = 'import sys; sys.exit(1)'
    var_10 = True
    var_11 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_10)
    var_12 = bool(not os.path.exists(var_6))
    assert var_12 is True
    var_13 = bool(False)
    assert var_13 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = 'hooks'
    var_8 = 'pre_gen_project.py'
    var_9 = 'import sys; sys.exit(1)'
    var_10 = True
    var_11 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_10, keep_project_on_failure=var_10)
    var_12 = bool(os.path.exists(var_6))
    assert var_12 is True
    var_13 = bool(False)
    assert var_13 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = 'existing content'
    var_8 = True
    var_9 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_8)
    var_10 = 'existing_file.txt'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_generate_file_creates_file_with_correct_content. Retrieved 13/18 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 13/18 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 14/22 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 13/16 statements.
# Partially parsed test_generate_file_applies_file_permissions. Retrieved 16/24 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = '{{ cookiecutter.variable }}'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/binary.bin'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = b'\x00\x01\x02\x03'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = '{{ cookiecutter.variable }}'
    var_12 = 'existing content'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_12 = ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = '{{ cookiecutter.variable }}'
    var_12 = 420
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_14 = 'file.txt'
    var_15 = 511



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_is_binary_predicate_at_line_47. Retrieved 6/12 statements.


import binaryornot.check as module_0

def test_case_0():
    var_0 = 'test_binary_file.bin'
    var_1 = 'test_text_file.txt'
    var_2 = b'\x00\x01\x02\x03'
    var_3 = 'Hello World'
    var_4 = module_0.is_binary(var_0)
    assert var_4 is True
    var_5 = module_0.is_binary(var_1)
    assert var_5 is False



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_template_syntax_error_is_raised. Retrieved 6/12 statements.


def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = False



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_new_lines_configuration_evaluates_to_true. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_is_binary_returns_true_for_binary_file. Retrieved 3/6 statements.


import binaryornot.check as module_0

def test_case_0():
    var_0 = 'test_binary_file.bin'
    var_1 = b'\x00\x01\x02\x03'
    var_2 = module_0.is_binary(var_0)
    assert var_2 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 13/16 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 3/7 statements.
# Partially parsed test_generate_context_without_default_and_extra_context. Retrieved 5/8 statements.
# Partially parsed test_generate_context_with_invalid_default_context. Retrieved 9/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'valid.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key1'
    var_8 = 'key2'
    var_9 = 'old_value1'
    var_10 = 'old_value2'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = module_0.generate_context(var_0, var_3, var_6)
    var_13 = var_12['valid']['key1']
    assert var_13 == 'value1'
    var_14 = var_12['valid']['key2']
    assert var_14 == 'value2'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = 'invalid json'
    var_2 = module_0.generate_context(var_0)
    var_3 = bool(True)
    assert var_3 is True
    var_4 = bool(False)
    assert var_4 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'valid.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = var_4['valid']['key1']
    assert var_5 == 'value1'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'valid.json'
    var_1 = 'key1'
    var_2 = 'invalid_value'
    var_3 = {var_1: var_2}
    var_4 = 'key1'
    var_5 = 'valid_value'
    var_6 = [var_5]
    var_7 = {var_4: var_6}
    var_8 = module_0.generate_context(var_0, var_3)



# Parsed testcases at query #46
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_template_syntax_error_has_translated_disabled. Retrieved 2/7 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_generate_file_creates_file. Retrieved 13/21 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 14/23 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 13/21 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 11/13 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1
import cookiecutter.utils as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = 'Hello {{ cookiecutter.variable }}'
    assert var_9 == 'Hello value'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_11 = 'template.txt'
    var_12 = module_2.rmtree(var_0)

import jinja2.environment as module_0
import cookiecutter.generate as module_1
import cookiecutter.utils as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = 'Hello {{ cookiecutter.variable }}'
    var_10 = 'template.txt'
    var_11 = 'Existing content'
    assert var_11 == 'Existing content'
    var_12 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)
    var_13 = module_2.rmtree(var_0)

import jinja2.environment as module_0
import cookiecutter.generate as module_1
import cookiecutter.utils as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary.dat'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = b'\x00\x01\x02\x03'
    assert var_9 == b'\x00\x01\x02\x03'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_11 = 'binary.dat'
    var_12 = module_2.rmtree(var_0)

import jinja2.environment as module_0
import cookiecutter.generate as module_1
import cookiecutter.utils as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_10 = module_2.rmtree(var_0)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_exists. Retrieved 10/17 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = '/path/to/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'file.txt'
    var_7 = True
    var_8 = 'existing content'
    var_9 = module_1.generate_file(var_0, var_1, var_4, var_5, var_7)



# Parsed testcases at query #50
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary_file.bin'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = 'test'
    var_6 = '\r\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.Environment()
    var_10 = module_1.generate_file(var_0, var_1, var_8, var_9)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'invalid_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)



# Parsed testcases at query #51
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_already_exists. Retrieved 10/16 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'template.txt'
    var_7 = True
    var_8 = 'existing content'
    assert var_8 == 'existing content'
    var_9 = module_1.generate_file(var_0, var_1, var_4, var_5, var_7)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_true. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_template_syntax_error_handling. Retrieved 9/11 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = '{% if %}'
    var_7 = {var_1: var_6}
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_exists. Retrieved 7/13 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'example.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'existing content'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3, var_5)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 8/13 statements.
# Partially parsed test_generate_file_text_file. Retrieved 11/16 statements.
# Partially parsed test_generate_file_skip_if_file_exists. Retrieved 9/17 statements.
# Partially parsed test_generate_file_empty_file_name. Retrieved 7/12 statements.
# Partially parsed test_generate_file_with_new_lines. Retrieved 11/16 statements.
# Partially parsed test_generate_file_template_syntax_error. Retrieved 7/11 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/binary_file.bin'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = b'binary content'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_7 = 'binary_file.bin'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = 'Text content with {{ cookiecutter.variable }}'
    assert var_9 == 'Text content with value'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/skip_file.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'Content'
    var_6 = 'skip_file.txt'
    var_7 = 'Existing content'
    assert var_7 == 'Existing content'
    var_8 = module_1.generate_file(var_0, var_1, var_2, var_3, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/empty_file'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'empty_file'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/new_line_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = 'Line 1\nLine 2'
    assert var_9 == 'Line 1\r\nLine 2'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/syntax_error_file.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = '{{ invalid syntax }}'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True



