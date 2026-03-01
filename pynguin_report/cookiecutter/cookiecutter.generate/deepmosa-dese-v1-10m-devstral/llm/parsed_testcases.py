####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__run_hook_from_repo_dir. Retrieved 12/19 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/path/to/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    assert var_4 == 1
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_10 = 0
    var_11 = var_6.category



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_nested'
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = 'e'
    var_8 = [var_6, var_7]
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_5, var_9)

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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'val1'
    var_4 = 'val2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_val2'
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_and_create_dir_existing_dir_no_overwrite. Retrieved 6/9 statements.
# Partially parsed test_render_and_create_dir_existing_dir_overwrite. Retrieved 6/8 statements.
# Partially parsed test_render_and_create_dir_new_dir. Retrieved 7/8 statements.


import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Path()
    var_3 = module_1.Environment()
    var_4 = module_2.render_and_create_dir(var_0, var_1, var_2, var_3)

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = module_0.Path()
    var_4 = module_1.Environment()
    var_5 = module_2.render_and_create_dir(var_1, var_2, var_3, var_4)

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = module_0.Path()
    var_3 = module_1.Environment()
    var_4 = True
    var_5 = module_2.render_and_create_dir(var_0, var_1, var_2, var_3, var_4)

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'new_dir'
    var_1 = {}
    var_2 = module_0.Path()
    var_3 = module_1.Environment()
    var_4 = module_2.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = 0
    var_6 = var_4[var_5]

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '{{ name }}'
    var_4 = module_0.Path()
    var_5 = module_1.Environment()
    var_6 = module_2.render_and_create_dir(var_3, var_2, var_4, var_5)



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'version'
    var_2 = '2.0.0'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter-bool.json'
    var_1 = 'is_active'
    var_2 = 'yes'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter-bool.json'
    var_1 = 'is_active'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_render_and_create_dir_creates_directory. Retrieved 7/8 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 8/9 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = 0
    var_6 = var_4[var_5]

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3, var_4)
    var_6 = 0
    var_7 = var_5[var_6]

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #7
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
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
    var_0 = 'file.py'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.is_copy_only_path(var_0, var_3)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = {}
    var_2 = module_0.is_copy_only_path(var_0, var_1)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'docs'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'docs'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'src/file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'src/*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'temp*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'temp_file.py'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True



# Parsed testcases at query #9
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_bool'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 6/15 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 6/14 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 8/20 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 7/16 statements.
# Partially parsed test_generate_files_without_hooks. Retrieved 7/16 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 6/17 statements.
# Partially parsed test_generate_files_keep_project_on_failure. Retrieved 6/14 statements.
# Partially parsed test_generate_files_delete_project_on_failure. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'
    var_5 = 'README.md'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'
    var_5 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'
    var_5 = 'existing content'
    assert var_5 == 'existing content'
    var_6 = True
    var_7 = 'new_file.txt'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'template_with_hooks'
    var_5 = True
    var_6 = 'hook_output.txt'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'template_with_hooks'
    var_5 = False
    var_6 = 'hook_output.txt'

def test_case_0():
    var_0 = 'project_name'
    assert var_0 == 'This file should not be rendered.'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'template_copy_without_render'
    var_5 = 'static_file.txt'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'template_with_error'
    var_5 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'template_with_error'
    var_5 = False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_json_decoding_error_raises_custom_exception. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = str(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 12/16 statements.
# Partially parsed test_generate_file_text_render. Retrieved 12/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 14/21 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 12/14 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/16 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = True
    var_12 = 'existing content'
    assert var_12 == 'existing content'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #14
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '.'
    var_3 = module_0.Environment()
    var_4 = False
    var_5 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #16
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'key'
    var_2 = 'default_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'key'
    var_2 = 'extra_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'key'
    var_2 = 'default_value'
    var_3 = {var_1: var_2}
    var_4 = 'extra_value'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'invalid_key'
    var_2 = 'invalid_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_cookiecutter_new_lines_predicate. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #18
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #19
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.bin'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_file_binary_skipped_if_exists. Retrieved 12/13 statements.
# Partially parsed test_generate_file_text_rendered. Retrieved 11/15 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 12/18 statements.
# Partially parsed test_generate_file_permissions_copied. Retrieved 11/16 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.bin'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = True
    var_10 = lambda x: var_9
    var_11 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = False
    var_9 = lambda x: var_8
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = False
    var_9 = lambda x: var_8
    var_10 = 'line1\nline2'
    var_11 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = False
    var_9 = lambda x: var_8
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'repo'
    var_1 = 'hook'
    var_2 = 'project'
    var_3 = {}
    var_4 = False
    var_5 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_1]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 3
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_and_create_dir_existing_dir_no_overwrite. Retrieved 7/11 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 7/10 statements.
# Partially parsed test_render_and_create_dir_new_dir. Retrieved 6/8 statements.
# Partially parsed test_render_and_create_dir_rendered_name. Retrieved 8/10 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = True
    var_2 = 'existing_dir'
    var_3 = {}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_2, var_3, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = True
    var_2 = 'existing_dir'
    var_3 = {}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_2, var_3, var_4, var_5, var_1)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/new_dir'
    var_1 = 'new_dir'
    var_2 = {}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = module_1.render_and_create_dir(var_1, var_2, var_3, var_4)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ project_name }}_dir'
    var_5 = '/tmp'
    var_6 = module_1.render_and_create_dir(var_4, var_2, var_5, var_3)
    var_7 = '/tmp/test_project_dir'



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_bool'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = 'Expected ValueError for invalid boolean overwrite'
    var_7 = AssertionError(var_6)



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_render_and_create_dir_existing_dir_no_overwrite. Retrieved 5/9 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 5/8 statements.
# Partially parsed test_render_and_create_dir_new_dir. Retrieved 4/7 statements.
# Partially parsed test_render_and_create_dir_rendered_name. Retrieved 6/8 statements.


import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Path()
    var_3 = module_1.Environment()
    var_4 = module_2.render_and_create_dir(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = True
    var_2 = 'test'
    var_3 = {}
    var_4 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = True
    var_2 = 'test'
    var_3 = {}
    var_4 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'new_dir'
    var_1 = 'test'
    var_2 = {}
    var_3 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'rendered_dir'
    var_4 = '{{ name }}'
    var_5 = module_0.Environment()



# Parsed testcases at query #7
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'key'
    var_2 = 'default_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'key'
    var_2 = 'extra_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'key'
    var_2 = 'default_value'
    var_3 = {var_1: var_2}
    var_4 = 'extra_value'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-nested-cookiecutter.json'
    var_1 = 'nested'
    var_2 = 'key'
    var_3 = 'nested_value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-list-cookiecutter.json'
    var_1 = 'choice'
    var_2 = 'option2'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-list-cookiecutter.json'
    var_1 = 'choice'
    var_2 = 'invalid_option'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-boolean-cookiecutter.json'
    var_1 = 'bool_var'
    var_2 = 'yes'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-boolean-cookiecutter.json'
    var_1 = 'bool_var'
    var_2 = 'invalid_bool'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)



# Parsed testcases at query #8
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'extra'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)



# Parsed testcases at query #9
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_bool'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #10
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'value'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'value'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_0, var_3, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = None
    var_2 = module_0.generate_context(var_0, var_1, var_1)



# Parsed testcases at query #11
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_bool'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #12
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_bool'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_string'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_generate_context_opens_file. Retrieved 5/9 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)



# Parsed testcases at query #14
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
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
    var_0 = 'file.py'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.is_copy_only_path(var_0, var_3)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = {}
    var_2 = module_0.is_copy_only_path(var_0, var_1)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.py'
    var_4 = '*.txt'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.is_copy_only_path(var_0, var_7)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dir/subdir/file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'dir/**'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True



# Parsed testcases at query #15
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_generate_context_with_valid_json. Retrieved 2/3 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 3/5 statements.
# Partially parsed test_generate_context_with_invalid_default_context. Retrieved 8/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = str(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'invalid_var'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = 0
    var_6 = var_1.message
    var_7 = str(var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = None
    var_2 = module_0.generate_context(var_0, var_1, var_1)



# Parsed testcases at query #17
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Path()
    var_3 = module_1.Environment()
    var_4 = module_2.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #18
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Path()
    var_3 = module_1.Environment()
    var_4 = module_2.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #19
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'extra'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter-nested.json'
    var_1 = 'nested'
    var_2 = 'key1'
    var_3 = 'overwritten'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter-list.json'
    var_1 = 'choices'
    var_2 = 'choice2'
    var_3 = 'choice1'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter-list.json'
    var_1 = 'choices'
    var_2 = 'invalid'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = module_0.generate_context(var_0, extra_context=var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter-bool.json'
    var_1 = 'flag'
    var_2 = 'yes'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter-bool.json'
    var_1 = 'flag'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'cookiecutter.json'
    var_4 = None



# Parsed testcases at query #21
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'cookiecutter.json'
    var_5 = module_0.generate_context(var_4, var_2, var_3)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_generate_context_opens_file_with_utf8_encoding. Retrieved 3/6 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = '{"key": "value"}'
    var_2 = module_0.generate_context(var_0)



# Parsed testcases at query #23
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'cookiecutter.json'
    var_5 = module_0.generate_context(var_4, var_2, var_3)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 6/15 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 6/14 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 8/20 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 7/16 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 6/17 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 5/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_templates'
    var_4 = 'basic'
    var_5 = 'README.md'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_templates'
    var_4 = 'basic'
    var_5 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_templates'
    var_4 = 'basic'
    var_5 = 'existing content'
    assert var_5 == 'existing content'
    var_6 = True
    var_7 = 'new_file.txt'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_templates'
    var_4 = 'with_hooks'
    var_5 = True
    var_6 = 'hook_marker.txt'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_templates'
    var_4 = 'copy_without_render'
    var_5 = 'static_file.txt'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_templates'
    var_4 = 'undefined_var'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_templates'
    var_4 = 'undefined_var'
    var_5 = True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 8/10 statements.
# Partially parsed test_generate_file_text. Retrieved 8/10 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 10/16 statements.
# Partially parsed test_generate_file_empty_name. Retrieved 8/10 statements.
# Partially parsed test_generate_file_newline_config. Retrieved 10/12 statements.
# Partially parsed test_generate_file_newline_detect. Retrieved 8/10 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = True
    var_8 = 'w'
    var_9 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = False
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'detect_newline.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 10/12 statements.
# Partially parsed test_generate_file_text. Retrieved 10/12 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 12/18 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 10/12 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 10/12 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = True
    var_10 = 'w'
    var_11 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'custom_newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_cookiecutter_new_lines_true. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_template_syntax_error_exception_handling. Retrieved 11/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = '{% if %}'
    var_10 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 9/14 statements.
# Partially parsed test_generate_file_text_file_with_newline_config. Retrieved 14/22 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 10/12 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/fake/template'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = module_2.generate_file(var_0, var_1, var_4, var_7)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = 'rendered content'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/fake/template'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 12/17 statements.
# Partially parsed test_generate_file_text_file. Retrieved 15/23 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 12/14 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.bin'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'w'
    var_13 = 'utf-8'
    var_14 = None

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'empty_dir/'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #31
#--------------------------




import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'fake_template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/fake/templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = module_2.generate_file(var_0, var_1, var_4, var_7)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_delete_project_on_failure_false_when_keep_project_on_failure_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 3/5 statements.


def test_case_0():
    var_0 = True
    var_1 = '/path/to/existing/file'
    var_2 = True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_67_is_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_at_line_39_evaluates_to_true. Retrieved 11/17 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'fake_infile.txt'
    var_2 = 'fake_key'
    var_3 = 'fake_value'
    var_4 = {var_2: var_3}
    var_5 = '/fake/template/dir'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = True
    var_10 = 'a'



# Parsed testcases at query #36
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 13/15 statements.
# Partially parsed test_generate_file_text_file. Retrieved 13/15 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 15/21 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 12/13 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'binary_file.png'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'text_file.txt'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = True
    var_12 = 'existing_file.txt'
    var_13 = 'w'
    var_14 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 14/25 statements.
# Partially parsed test_generate_file_text_file. Retrieved 19/40 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 13/20 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 11/17 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.jpg'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'Processing file %s'
    var_13 = 'Copying binary %s to %s without rendering'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = '/'
    var_13 = 'w'
    var_14 = 'utf-8'
    var_15 = '\n'
    var_16 = 'rendered content'
    var_17 = 'Processing file %s'
    var_18 = 'Writing contents to file %s'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'The resulting file already exists: %s'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{empty}}'
    var_2 = 'empty'
    var_3 = ''
    var_4 = {var_2: var_3}
    var_5 = '/fake/template'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_10 = 'The resulting file name is empty: %s'



# Parsed testcases at query #40
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'cookiecutter.json'
    var_5 = module_0.generate_context(var_4, var_2, var_3)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_os_walk_returns_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = '.'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 10/12 statements.
# Partially parsed test_generate_file_text. Retrieved 10/12 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 12/18 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 10/12 statements.
# Partially parsed test_generate_file_newline_config. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = True
    var_10 = 'w'
    var_11 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #44
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #45
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 6/20 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 7/22 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 8/26 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/25 statements.
# Partially parsed test_generate_files_hooks. Retrieved 10/32 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'
    var_6 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'
    var_6 = True
    var_7 = 'Existing content'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'cookiecutter'
    var_2 = 'test_project'
    var_3 = '_copy_without_render'
    var_4 = '*.bin'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = '{{cookiecutter.project_name}}'
    var_9 = 'test.bin'
    var_10 = b'\x00\x01\x02'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'
    var_6 = 'hooks'
    var_7 = 'pre_gen_project.py'
    var_8 = 'print("Pre-hook executed")'
    var_9 = True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 5/14 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 6/15 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 6/15 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 6/15 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'
    var_5 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'
    var_5 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'
    var_5 = False

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'
    var_5 = True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_cookiecutter_new_lines_is_true. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #50
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'valid.json'
    var_1 = None
    var_2 = module_0.generate_context(var_0, var_1, var_1)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #52
#--------------------------

# Failed to parse test_work_in_context_manager_returns_true.




# Parsed testcases at query #53
#--------------------------

# Partially parsed test_cookiecutter_new_lines_predicate. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #54
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_generate_context_opens_file. Retrieved 3/6 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = '{"key": "value"}'
    var_2 = module_0.generate_context(var_0)



# Parsed testcases at query #56
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'valid_repo'
    var_1 = 'invalid_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'output'
    var_5 = False
    var_6 = module_0.generate_files(var_0, var_3, var_4, var_5, var_5, var_5, var_5)



# Parsed testcases at query #57
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #58
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #59
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'Default Project'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test-data/cookiecutter.json'
    var_4 = module_0.generate_context(var_3, var_2)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_slug'
    var_1 = 'extra_slug'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test-data/cookiecutter.json'
    var_4 = module_0.generate_context(var_3, extra_context=var_2)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/empty.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'use_pytest'
    var_1 = 'yes'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test-data/cookiecutter.json'
    var_4 = module_0.generate_context(var_3, extra_context=var_2)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'use_pytest'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test-data/cookiecutter.json'
    var_4 = module_0.generate_context(var_3, extra_context=var_2)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'license'
    var_1 = 'MIT'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'tests/test-data/cookiecutter.json'
    var_5 = module_0.generate_context(var_4, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'license'
    var_1 = 'Invalid'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'tests/test-data/cookiecutter.json'
    var_5 = module_0.generate_context(var_4, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-data/cookiecutter.json'
    var_6 = module_0.generate_context(var_5, extra_context=var_4)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_os_walk_predicate. Retrieved 9/21 statements.


import cookiecutter.utils as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 62 evaluates to True.'
    var_1 = {}
    var_2 = module_0.create_env_with_context(var_1)
    var_3 = {}
    var_4 = 'test_dir'
    var_5 = 'test content'
    var_6 = '.'
    var_7 = '../templates'
    var_8 = [var_6, var_7]



# Parsed testcases at query #63
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #64
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = None
    var_2 = module_0.generate_context(var_0, var_1, var_1)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_work_in_context_manager_changes_and_restores_directory. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = True



# Parsed testcases at query #66
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #67
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'version'
    var_2 = '2.0.0'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'version'
    var_5 = '2.0.0'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_0, var_3, var_6)



# Parsed testcases at query #68
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #69
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'test_dir'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.is_copy_only_path(var_2, var_5)



# Parsed testcases at query #70
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #71
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'valid_template'
    var_1 = 'invalid_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'output'
    var_5 = False
    var_6 = module_0.generate_files(var_0, var_3, var_4, var_5, var_5, var_5, var_5)



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #73
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #74
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_predicate_at_line_62. Retrieved 23/34 statements.


import cookiecutter.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 62 evaluates to True.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter'
    var_3 = '_jinja2_env_vars'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_output'
    var_8 = 'test_project'
    var_9 = 'test_template'
    var_10 = '{{cookiecutter.project_name}}'
    var_11 = '.'
    var_12 = 'dir1'
    var_13 = 'dir2'
    var_14 = [var_12, var_13]
    var_15 = 'file1.txt'
    var_16 = 'file2.txt'
    var_17 = [var_15, var_16]
    var_18 = (var_11, var_14, var_17)
    var_19 = True
    var_20 = module_0.StrictEnvironment()
    var_21 = False
    var_22 = True



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_accept_hooks_predicate. Retrieved 16/21 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'fake'
    var_2 = 'context'
    var_3 = {var_1: var_2}
    var_4 = '/fake/output'
    var_5 = False
    var_6 = False
    var_7 = True
    var_8 = False
    var_9 = '/fake/project'
    var_10 = True
    var_11 = '/'
    var_12 = []
    var_13 = []
    var_14 = (var_11, var_12, var_13)
    var_15 = module_0.generate_files(var_0, var_3, var_4, var_5, var_6, var_7, var_8)



# Parsed testcases at query #77
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'extra'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 6/14 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 6/12 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 6/12 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 6/12 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 6/12 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/17 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 5/10 statements.
# Partially parsed test_generate_files_empty_dir_name. Retrieved 5/10 statements.
# Partially parsed test_generate_files_output_dir_exists. Retrieved 5/12 statements.
# Partially parsed test_generate_files_new_lines. Retrieved 9/15 statements.
# Partially parsed test_generate_files_hooks_failure. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'
    var_5 = 'README.md'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'
    var_5 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'
    var_5 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'
    var_5 = False

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'
    var_5 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'cookiecutter'
    var_2 = 'test_project'
    var_3 = '_copy_without_render'
    var_4 = '*.bin'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = 'test_data'
    var_9 = 'copy_template'
    var_10 = 'data.bin'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'undefined_template'

def test_case_0():
    var_0 = 'project_name'
    var_1 = ''
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'basic_template'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'cookiecutter'
    var_2 = 'test_project'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'test_data'
    var_8 = 'basic_template'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_data'
    var_4 = 'hook_failure_template'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 3/10 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 4/12 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 4/12 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 4/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = False

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = True



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_predicate_at_line_36_evaluates_to_true. Retrieved 8/9 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'valid_template_dir'
    var_1 = 'valid_key'
    var_2 = 'valid_value'
    var_3 = {var_1: var_2}
    var_4 = 'output_directory'
    var_5 = True
    var_6 = False
    var_7 = module_0.generate_files(var_0, var_3, var_4, var_5, var_6, var_6, var_6)



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_predicate_at_line_62_evaluates_to_false. Retrieved 16/25 statements.


import cookiecutter.utils as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'test_dir'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = 'test_output'
    var_8 = module_0.create_env_with_context(var_6)
    var_9 = 'test_project'
    var_10 = True
    var_11 = '.'
    var_12 = '../templates'
    var_13 = [var_11, var_12]
    var_14 = 'test_dir'
    var_15 = True



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_os_walk_predicate_false. Retrieved 11/25 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '.'
    var_1 = []
    var_2 = []
    var_3 = (var_0, var_1, var_2)
    var_4 = '.'
    var_5 = '.'
    var_6 = False
    var_7 = '.'
    var_8 = {}
    var_9 = False
    var_10 = module_0.generate_files(var_7, var_8, var_7, var_9, var_9, var_9, var_9)



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_predicate_at_line_62_evaluates_to_false. Retrieved 18/35 statements.


def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'fake_dir'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = '/fake/output'
    var_8 = '/fake/repo/{{cookiecutter.project_name}}'
    var_9 = '/fake/output/{{cookiecutter.project_name}}'
    var_10 = True
    var_11 = '.'
    var_12 = 'fake_dir'
    var_13 = [var_12]
    var_14 = 'fake_file'
    var_15 = [var_14]
    var_16 = (var_11, var_13, var_15)
    var_17 = False



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 6/21 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 7/18 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 8/25 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 12/30 statements.
# Partially parsed test_generate_files_with_copy_without_render. Retrieved 11/22 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 4/11 statements.
# Partially parsed test_generate_files_keep_project_on_failure. Retrieved 7/18 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 7/17 statements.
# Partially parsed test_generate_files_empty_context. Retrieved 4/16 statements.
# Partially parsed test_generate_files_nested_directories. Retrieved 7/19 statements.
# Partially parsed test_generate_files_binary_file. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'content'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'content'
    var_6 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'content'
    var_6 = 'new content'
    var_7 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'content'
    var_6 = 'hooks'
    var_7 = 'pre_gen_project.py'
    var_8 = 'print("pre hook")'
    var_9 = 'post_gen_project.py'
    var_10 = 'print("post hook")'
    var_11 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'cookiecutter'
    var_2 = 'test_project'
    var_3 = '_copy_without_render'
    var_4 = '*.bin'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = '{{cookiecutter.project_name}}'
    var_9 = 'file.bin'
    var_10 = 'binary content'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.undefined_var}}'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = '{{cookiecutter.undefined_var}}'
    var_6 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'content'
    var_6 = False

def test_case_0():
    var_0 = {}
    var_1 = 'project'
    var_2 = 'file.txt'
    var_3 = 'content'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'nested'
    var_5 = 'file.txt'
    var_6 = 'content'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'image.png'
    var_5 = b'\x89PNG\r\n\x1a\n'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #89
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'test'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'test_output'
    var_5 = True
    var_6 = False
    var_7 = module_0.generate_files(var_0, var_3, var_4, var_5, var_6, var_5, var_6)



# Parsed testcases at query #90
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_predicate_at_line_62_evaluates_to_false. Retrieved 18/28 statements.


import cookiecutter.utils as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = '_jinja2_env_vars'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = False
    var_8 = False
    var_9 = False
    var_10 = False
    var_11 = module_0.create_env_with_context(var_5)
    var_12 = '.'
    var_13 = '../templates'
    var_14 = [var_12, var_13]
    var_15 = 'test_dir'
    var_16 = True
    var_17 = 'another_dir'



# Parsed testcases at query #92
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_delete_project_on_failure_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_template_syntax_error_handling. Retrieved 10/13 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = '{% invalid syntax %}'
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7)



# Parsed testcases at query #96
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/context.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/context.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/context.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/empty.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/nested_context.json'
    var_1 = 'nested'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/list_context.json'
    var_1 = 'choice'
    var_2 = 'option2'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/list_context.json'
    var_1 = 'choice'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/bool_context.json'
    var_1 = 'flag'
    var_2 = 'yes'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/bool_context.json'
    var_1 = 'flag'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)



# Parsed testcases at query #97
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_generate_context_raises_context_decoding_exception_on_invalid_json. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = str(var_0)



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 3/8 statements.


def test_case_0():
    var_0 = True
    var_1 = 'existing_file.txt'
    var_2 = True



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname. Retrieved 4/7 statements.
# Partially parsed test_render_and_create_dir_existing_dir_no_overwrite. Retrieved 6/11 statements.
# Partially parsed test_render_and_create_dir_existing_dir_overwrite. Retrieved 6/10 statements.
# Partially parsed test_render_and_create_dir_new_dir. Retrieved 6/12 statements.
# Partially parsed test_render_and_create_dir_rendered_name. Retrieved 8/14 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = True
    var_2 = 'existing_dir'
    var_3 = {}
    var_4 = '/tmp'
    var_5 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = True
    var_2 = 'existing_dir'
    var_3 = {}
    var_4 = '/tmp'
    var_5 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'new_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = '/tmp/new_dir'
    var_5 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '{{ name }}_dir'
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = '/tmp/test_dir'
    var_7 = True



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_generate_file_binary_skips_rendering. Retrieved 12/15 statements.
# Partially parsed test_generate_file_text_renders_correctly. Retrieved 19/27 statements.
# Partially parsed test_generate_file_skips_existing_file. Retrieved 12/14 statements.
# Partially parsed test_generate_file_empty_filename_skips. Retrieved 11/13 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_11 = '/fake/project/binary.png'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = '\n'
    var_6 = 'test'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'templates'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = 'rendered'
    var_13 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_14 = '/fake/project/template.txt'
    var_15 = 'w'
    var_16 = 'utf-8'
    var_17 = '\n'
    var_18 = 'rendered'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #102
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_bool'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #103
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'extra'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_generate_context_with_valid_json_file. Retrieved 4/6 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'valid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'valid'
    var_3 = []



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 6/11 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 7/12 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 7/12 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 7/11 statements.
# Partially parsed test_generate_files_without_hooks. Retrieved 7/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'tests/mocks/pre_and_post_gen_hooks'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'tests/mocks/pre_and_post_gen_hooks'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = True

def test_case_0():
    var_0 = 'tests/mocks/pre_and_post_gen_hooks'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = True

def test_case_0():
    var_0 = 'tests/mocks/pre_and_post_gen_hooks'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = True

def test_case_0():
    var_0 = 'tests/mocks/pre_and_post_gen_hooks'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = False

def test_case_0():
    var_0 = 'tests/mocks/pre_and_post_gen_hooks'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = True



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 6/12 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 7/11 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 7/11 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 7/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 7/11 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 6/9 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = False

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template-undefined'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.txt'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'tests/test-template-copy'
    var_9 = 'copy.txt'



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 7/16 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 7/15 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 7/15 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 8/17 statements.
# Partially parsed test_generate_files_keep_project_on_failure. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test content'
    var_6 = 'test content'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test content'
    var_6 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test content'
    var_6 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test content'
    var_6 = 'print("Pre hook executed")'
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test content'
    var_6 = 'raise Exception("Hook failed")'
    var_7 = True



# Parsed testcases at query #109
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/11 statements.
# Partially parsed test_generate_files_with_overwrite. Retrieved 9/12 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 9/12 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 9/12 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 9/12 statements.
# Partially parsed test_generate_files_with_copy_without_render. Retrieved 12/17 statements.
# Partially parsed test_generate_files_output_dir_exists. Retrieved 9/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-templates/basic/'
    var_6 = module_0.generate_files(var_5, var_4)
    var_7 = 'README.md'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-templates/basic/'
    var_6 = True
    var_7 = module_0.generate_files(var_5, var_4, overwrite_if_exists=var_6)
    var_8 = 'README.md'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-templates/basic/'
    var_6 = True
    var_7 = module_0.generate_files(var_5, var_4, skip_if_file_exists=var_6)
    var_8 = 'README.md'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-templates/basic/'
    var_6 = False
    var_7 = module_0.generate_files(var_5, var_4, accept_hooks=var_6)
    var_8 = 'README.md'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-templates/basic/'
    var_6 = True
    var_7 = module_0.generate_files(var_5, var_4, keep_project_on_failure=var_6)
    var_8 = 'README.md'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.txt'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'tests/test-templates/basic/'
    var_9 = module_0.generate_files(var_8, var_7)
    var_10 = 'README.md'
    var_11 = 'copy_only.txt'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-templates/undefined_var/'
    var_6 = module_0.generate_files(var_5, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = ''
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-templates/basic/'
    var_6 = module_0.generate_files(var_5, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-output/'
    var_6 = True
    var_7 = 'tests/test-templates/basic/'
    var_8 = module_0.generate_files(var_7, var_4, var_5)



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 9/22 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 10/22 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 12/28 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 13/30 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/22 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    assert var_7 == 'Hello, test_project!'
    var_8 = 'test.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = True
    var_9 = 'test.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = 'Existing file'
    var_9 = True
    var_10 = 'test.txt'
    var_11 = 'existing.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = 'hooks'
    var_8 = 'print("Pre-hook executed")'
    var_9 = 'print("Post-hook executed")'
    var_10 = 'Hello, {{cookiecutter.project_name}}!'
    var_11 = True
    var_12 = 'test.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.md'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'template'
    var_9 = '{{cookiecutter.project_name}}'
    var_10 = '# {{cookiecutter.project_name}}'
    assert var_10 == '# {{cookiecutter.project_name}}'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = '{{cookiecutter.undefined_var}}'



# Parsed testcases at query #112
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true. Retrieved 3/5 statements.


def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = True



# Parsed testcases at query #114
#--------------------------

# Partially parsed test_accept_hooks_false. Retrieved 13/23 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/path/to/repo'
    var_6 = '/path/to/output'
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = 'file.txt'
    var_9 = 'content'
    var_10 = False
    var_11 = module_0.generate_files(var_5, var_4, var_6, accept_hooks=var_10)
    var_12 = ''



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = {var_5: var_2}
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_1]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
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



# Parsed testcases at query #3
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_string'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_render_and_create_dir_existing_dir. Retrieved 6/12 statements.
# Partially parsed test_render_and_create_dir_overwrite_existing. Retrieved 7/12 statements.
# Partially parsed test_render_and_create_dir_new_directory. Retrieved 6/10 statements.


import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'Test that EmptyDirNameException is raised when dirname is empty.'
    var_1 = ''
    var_2 = {}
    var_3 = module_0.Path()
    var_4 = module_1.Environment()
    var_5 = module_2.render_and_create_dir(var_1, var_2, var_3, var_4)

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that OutputDirExistsException is raised when directory exists.'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = '{{ name }}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that existing directory is overwritten when flag is set.'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = '{{ name }}'
    var_6 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that new directory is created successfully.'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = '{{ name }}'



# Parsed testcases at query #6
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = {}
    var_4 = False
    var_5 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #7
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 13/15 statements.
# Partially parsed test_generate_file_text_file. Retrieved 13/15 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 15/21 statements.
# Partially parsed test_generate_file_empty_outfile. Retrieved 13/15 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'binary.png'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'template.txt'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = True
    var_12 = 'existing.txt'
    var_13 = 'w'
    var_14 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{""}}'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 11/13 statements.
# Partially parsed test_generate_file_text. Retrieved 11/13 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 11/13 statements.
# Partially parsed test_generate_file_empty_outfile. Retrieved 11/13 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 11/13 statements.
# Partially parsed test_generate_file_newline_config. Retrieved 13/15 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/tmp/templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_10 = 'binary_file.png'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/tmp/templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_10 = 'text_file.txt'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/tmp/templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_10 = 'existing_file.txt'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'empty_dir'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/tmp/templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_10 = 'empty_dir'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/tmp/templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_10 = 'newline_file.txt'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'newline_file.txt'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_output_dir_exists_predicate. Retrieved 6/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = False
    var_5 = True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_render_and_create_dir_success. Retrieved 6/10 statements.
# Partially parsed test_render_and_create_dir_empty_dirname. Retrieved 4/7 statements.
# Partially parsed test_render_and_create_dir_exists_no_overwrite. Retrieved 7/12 statements.
# Partially parsed test_render_and_create_dir_exists_with_overwrite. Retrieved 7/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = '{{ project_name }}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = True
    var_6 = '{{ project_name }}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = True
    var_6 = '{{ project_name }}'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_template_syntax_error_raised_when_invalid_template. Retrieved 10/13 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'invalid_template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = '{% invalid syntax %}'
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 10/12 statements.
# Partially parsed test_generate_file_text. Retrieved 10/12 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 12/18 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 10/12 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 10/12 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = True
    var_10 = 'w'
    var_11 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_generate_context_raises_context_decoding_exception_on_invalid_json. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = str(var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_template_syntax_error_handling. Retrieved 10/13 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'fake_template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = '{% if %}'
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 9/13 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 11/13 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 11/13 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 10/12 statements.
# Partially parsed test_generate_files_without_hooks. Retrieved 10/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test basic file generation with minimal context.'
    var_1 = 'tests/test-template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'tests/output'
    var_8 = module_0.generate_files(var_1, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test overwriting existing output directory.'
    var_1 = 'tests/test-template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'tests/output'
    var_8 = module_0.generate_files(var_1, var_6, var_7)
    var_9 = True
    var_10 = module_0.generate_files(var_1, var_6, var_7, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test skipping existing files.'
    var_1 = 'tests/test-template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'tests/output'
    var_8 = module_0.generate_files(var_1, var_6, var_7)
    var_9 = True
    var_10 = module_0.generate_files(var_1, var_6, var_7, skip_if_file_exists=var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with hooks enabled.'
    var_1 = 'tests/test-template-with-hooks'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'tests/output'
    var_8 = True
    var_9 = module_0.generate_files(var_1, var_6, var_7, accept_hooks=var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with hooks disabled.'
    var_1 = 'tests/test-template-with-hooks'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'tests/output'
    var_8 = False
    var_9 = module_0.generate_files(var_1, var_6, var_7, accept_hooks=var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test keeping project directory on failure.'
    var_1 = 'tests/test-template-with-error'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'tests/output'
    var_8 = True
    var_9 = module_0.generate_files(var_1, var_6, var_7, keep_project_on_failure=var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test deleting project directory on failure.'
    var_1 = 'tests/test-template-with-error'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'tests/output'
    var_8 = False
    var_9 = module_0.generate_files(var_1, var_6, var_7, keep_project_on_failure=var_8)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_output_dir_exists_predicate. Retrieved 6/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = False
    var_5 = True



# Parsed testcases at query #18
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_render_and_create_dir_when_dir_exists. Retrieved 5/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 12/17 statements.
# Partially parsed test_generate_file_text. Retrieved 12/15 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_generate_files_success. Retrieved 9/11 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 9/11 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 9/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 9/11 statements.
# Partially parsed test_generate_files_no_context. Retrieved 4/6 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = './test_output'
    var_6 = './test_repo'
    var_7 = True
    var_8 = module_0.generate_files(var_6, var_4, var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = './test_output'
    var_6 = './test_repo_with_hooks'
    var_7 = True
    var_8 = module_0.generate_files(var_6, var_4, var_5, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = './test_output'
    var_6 = './test_repo'
    var_7 = True
    var_8 = module_0.generate_files(var_6, var_4, var_5, skip_if_file_exists=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = './test_output'
    var_6 = './test_repo_failure'
    var_7 = True
    var_8 = module_0.generate_files(var_6, var_4, var_5, keep_project_on_failure=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = './test_output'
    var_1 = './test_repo'
    var_2 = None
    var_3 = module_0.generate_files(var_1, var_2, var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_generate_context_raises_exception_on_invalid_json. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = str(var_0)



# Parsed testcases at query #24
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #25
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_value2'
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_open_file_success. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = '{"key": "value"}'
    var_2 = module_0.generate_context(var_0)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate_evaluates_to_true. Retrieved 9/15 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'fake_template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = True
    var_8 = 'fake content'



# Parsed testcases at query #28
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 13/20 statements.
# Partially parsed test_generate_file_text. Retrieved 15/22 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 13/19 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 14/17 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 13/18 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.jpg'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = b'\x00\x01\x02'
    assert var_11 == b'\x00\x01\x02'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = None
    var_6 = 'test'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello {{ cookiecutter.name }}!'
    assert var_13 == 'Hello test!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'existing content'
    assert var_11 == 'existing content'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{ cookiecutter.empty }}'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'empty'
    var_5 = None
    var_6 = ''
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = module_2.generate_file(var_0, var_1, var_8, var_11)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template_nl.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Line1\r\nLine2\r\n'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 1/3 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #31
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_file_name_is_empty_predicate. Retrieved 5/8 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'some_directory/'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = var_3.from_string(var_1)



# Parsed testcases at query #33
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 13/18 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = True
    var_12 = 'existing content'



# Parsed testcases at query #35
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Path()
    var_3 = module_1.Environment()
    var_4 = module_2.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_true. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #37
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'variable'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #38
#--------------------------




import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'empty_dir/'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_file_name_is_empty_when_outfile_is_directory. Retrieved 8/13 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5)



# Parsed testcases at query #40
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'output_dir'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #41
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
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
    var_0 = 'file.py'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.is_copy_only_path(var_0, var_3)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = {}
    var_2 = module_0.is_copy_only_path(var_0, var_1)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'src/static'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*/static'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'src/templates'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*/static'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 12/16 statements.
# Partially parsed test_generate_file_text_render. Retrieved 12/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 14/20 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = True
    var_12 = 'w'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = '{{ "" }}'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 6/13 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 7/14 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 7/14 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 7/14 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template-with-hooks'
    var_6 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_output_dir_exists_predicate. Retrieved 6/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = True



# Parsed testcases at query #45
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #46
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #47
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'cookiecutter.json'
    var_4 = None
    var_5 = module_0.generate_context(var_3, var_2, var_4)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 12/17 statements.
# Partially parsed test_generate_file_text_file. Retrieved 16/30 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 15/21 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.jpg'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = '/'
    var_13 = 'w'
    var_14 = 'utf-8'
    var_15 = '\n'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'empty_dir/'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'w'
    var_13 = 'utf-8'
    var_14 = '\r\n'



# Parsed testcases at query #49
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_render_and_create_dir_overwrite_if_exists_true. Retrieved 6/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = True



# Parsed testcases at query #51
#--------------------------




def test_case_0():
    var_0 = True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 1/3 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #53
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #54
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #55
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'use_pytest'
    var_2 = 'yes'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'use_pytest'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'framework'
    var_2 = 'flask'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'framework'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'config'
    var_2 = 'debug'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)



# Parsed testcases at query #56
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_new_lines_predicate_evaluates_to_true. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_render_and_create_dir_success. Retrieved 7/12 statements.
# Partially parsed test_render_and_create_dir_overwrite. Retrieved 8/15 statements.
# Partially parsed test_render_and_create_dir_exists_exception. Retrieved 8/13 statements.
# Partially parsed test_render_and_create_dir_empty_name_exception. Retrieved 6/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = '/tmp/test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = '/tmp/test_dir'
    var_7 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = '/tmp/test_dir'
    var_7 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 10/14 statements.
# Partially parsed test_generate_file_text_file. Retrieved 10/14 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 10/12 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 10/12 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 10/12 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'custom_newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/10 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 9/11 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 9/11 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 9/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 9/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = module_0.generate_files(var_5, var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, skip_if_file_exists=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = False
    var_8 = module_0.generate_files(var_5, var_4, var_6, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, keep_project_on_failure=var_7)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_generate_context_opens_file. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'utf-8'



# Parsed testcases at query #62
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'extra'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'invalid_key'
    var_3 = 'invalid'
    var_4 = 'value'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.generate_context(var_0, var_5)



# Parsed testcases at query #63
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #64
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 12/16 statements.
# Partially parsed test_generate_file_text_render. Retrieved 14/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 14/20 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 13/15 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.jpg'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = 'test'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'templates'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = False
    var_13 = module_2.generate_file(var_0, var_1, var_8, var_11, var_12)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = True
    var_12 = 'w'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{""}}'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_generate_context_with_default_context_none. Retrieved 3/4 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.generate_context(default_context=var_0)
    var_2 = []



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate_false. Retrieved 11/13 statements.


def test_case_0():
    var_0 = 'Test that delete_project_on_failure is False when output_directory_created is False.'
    var_1 = 'cookiecutter'
    var_2 = '_jinja2_env_vars'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_repo'
    var_7 = 'test_output'
    var_8 = False
    var_9 = True
    var_10 = False



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_generate_context_with_no_default_context. Retrieved 7/8 statements.


import cookiecutter.generate as module_0
import collections as module_1

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = None
    var_2 = module_0.generate_context(var_0, var_1, var_1)
    var_3 = 'cookiecutter'
    var_4 = module_1.OrderedDict()
    var_5 = (var_3, var_4)
    var_6 = [var_5]



# Parsed testcases at query #69
#--------------------------




def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_os_walk_predicate_true. Retrieved 19/31 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '{{cookiecutter.test}}'
    var_1 = '.'
    var_2 = '/'
    var_3 = '/project/dir'
    var_4 = True
    var_5 = 'dir1'
    var_6 = [var_5]
    var_7 = 'file1.txt'
    var_8 = [var_7]
    var_9 = (var_1, var_6, var_8)
    var_10 = './test-repo'
    var_11 = 'cookiecutter'
    var_12 = 'test'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = './output'
    var_17 = False
    var_18 = module_0.generate_files(var_10, var_15, var_16, var_17, var_17, var_17, var_17)
    assert var_18 == '/project/dir'



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_os_walk_returns_non_empty_iterator. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = '.'
    var_4 = list(var_0)
    var_5 = len(var_4)
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = len(var_7)
    assert var_8 == 3



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_delete_project_on_failure_is_true_when_output_directory_created_and_keep_project_on_failure_is_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 9/11 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 10/12 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 10/12 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 10/12 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 10/12 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 12/14 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test basic file generation.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'tests/fake-repo-pre'
    var_7 = 'tests/output'
    var_8 = module_0.generate_files(var_6, var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with overwrite.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'tests/fake-repo-pre'
    var_7 = 'tests/output'
    var_8 = True
    var_9 = module_0.generate_files(var_6, var_5, var_7, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with skip existing files.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'tests/fake-repo-pre'
    var_7 = 'tests/output'
    var_8 = True
    var_9 = module_0.generate_files(var_6, var_5, var_7, skip_if_file_exists=var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation without hooks.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'tests/fake-repo-pre'
    var_7 = 'tests/output'
    var_8 = False
    var_9 = module_0.generate_files(var_6, var_5, var_7, accept_hooks=var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with keep on failure.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'tests/fake-repo-pre'
    var_7 = 'tests/output'
    var_8 = True
    var_9 = module_0.generate_files(var_6, var_5, var_7, keep_project_on_failure=var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with undefined variable.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'tests/fake-repo-pre'
    var_7 = 'tests/output'
    var_8 = module_0.generate_files(var_6, var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with copy without render.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_copy_without_render'
    var_4 = 'test_project'
    var_5 = '*.md'
    var_6 = [var_5]
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = {var_1: var_7}
    var_9 = 'tests/fake-repo-pre'
    var_10 = 'tests/output'
    var_11 = module_0.generate_files(var_9, var_8, var_10)



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_undefined_error_in_render_and_create_dir. Retrieved 8/14 statements.


import cookiecutter.utils as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'test_output'
    var_5 = module_0.create_env_with_context(var_3)
    var_6 = 1
    var_7 = False



# Parsed testcases at query #75
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_generate_context_opens_file. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'utf-8'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_predicate_at_line_62_evaluates_to_false. Retrieved 16/36 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Ensure the predicate at line 62 evaluates to False.'
    var_1 = '_copy_without_render'
    var_2 = 'some_dir'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = 'some_dir'
    var_6 = '.'
    var_7 = [var_5]
    var_8 = []
    var_9 = (var_6, var_7, var_8)
    var_10 = 'template_dir'
    var_11 = 'project_dir'
    var_12 = True
    var_13 = 'repo_dir'
    var_14 = module_0.generate_files(var_13, var_4)
    var_15 = 'some_dir'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_generate_context_with_default_context_none. Retrieved 4/6 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.generate_context(default_context=var_0)
    var_2 = 'cookiecutter'
    var_3 = []



# Parsed testcases at query #79
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-valid-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #80
#--------------------------




def test_case_0():
    var_0 = True
    assert var_0 is True



# Parsed testcases at query #81
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid_cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_cookiecutter.json'
    var_1 = 'key'
    var_2 = 'default_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_cookiecutter.json'
    var_1 = 'key'
    var_2 = 'extra_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_cookiecutter.json'
    var_1 = 'invalid_key'
    var_2 = 'invalid_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/17 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 9/20 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 9/22 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 9/18 statements.
# Partially parsed test_generate_files_with_copy_without_render. Retrieved 11/20 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'basic_template'
    var_7 = 'README.md'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'basic_template'
    var_7 = True
    var_8 = 'README.md'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'basic_template'
    var_7 = 'existing content'
    assert var_7 == 'existing content'
    var_8 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'template_with_hooks'
    var_7 = True
    var_8 = 'hook_marker.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.bin'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'test_data'
    var_9 = 'template_with_binary'
    var_10 = 'data.bin'



# Parsed testcases at query #83
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'extra'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/nonexistent.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_predicate_at_line_62_evaluates_to_false. Retrieved 10/13 statements.


import cookiecutter.utils as module_0
import cookiecutter.find as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.create_env_with_context(var_4)
    var_6 = '.'
    var_7 = module_1.find_template(var_6, var_5)
    var_8 = False
    var_9 = True



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_accept_hooks_false_predicate. Retrieved 11/22 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = 'test.txt'
    var_9 = 'test'
    var_10 = False



# Parsed testcases at query #86
#--------------------------




def test_case_0():
    var_0 = False



# Parsed testcases at query #87
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.generate_context(default_context=var_0)



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_undefined_error_in_render_and_create_dir. Retrieved 7/14 statements.


import cookiecutter.utils as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.create_env_with_context(var_2)
    var_4 = 'test_repo'
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = True



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_work_in_context_manager_returns_true. Retrieved 2/3 statements.


import cookiecutter.utils as module_0

def test_case_0():
    var_0 = '/path/to/template'
    var_1 = module_0.work_in(var_0)



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/21 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 9/22 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 9/22 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 14/36 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'file.txt'
    var_7 = 'content'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'file.txt'
    var_7 = 'content'
    var_8 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'file.txt'
    var_7 = 'content'
    var_8 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'file.txt'
    var_7 = 'content'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'print("pre hook")'
    var_11 = 'post_gen_project.py'
    var_12 = 'print("post hook")'
    var_13 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'file.txt'
    var_7 = 'content'
    var_8 = True



# Parsed testcases at query #93
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 7/12 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 8/13 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 8/13 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 8/13 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 8/13 statements.
# Partially parsed test_generate_files_copy_only_paths. Retrieved 11/19 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template-with-hooks'
    var_6 = 'tests/output'
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.bin'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'tests/test-template-with-binaries'
    var_9 = 'tests/output'
    var_10 = 'test.bin'



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/24 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 14/38 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 9/23 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 10/25 statements.
# Partially parsed test_generate_files_binary_file. Retrieved 8/24 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/27 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'file.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'file.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'print("Pre hook")'
    var_11 = 'post_gen_project.py'
    var_12 = 'print("Post hook")'
    var_13 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'file.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'file.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = 'Modified content'
    var_9 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = b'\x00\x01\x02\x03'
    var_7 = 'binary.bin'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.md'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = '{{cookiecutter.project_name}}'
    var_9 = 'readme.md'
    var_10 = 'This should not be rendered: {{cookiecutter.project_name}}'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'file.txt'
    var_7 = 'Hello, {{cookiecutter.undefined_var}}!'



# Parsed testcases at query #97
#--------------------------




def test_case_0():
    var_0 = 'test_repo'
    var_1 = {}
    var_2 = 'test_output'
    var_3 = False
    var_4 = False
    var_5 = False
    var_6 = False



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/10 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 9/11 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 9/11 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 9/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 9/11 statements.
# Partially parsed test_generate_files_empty_context. Retrieved 4/6 statements.
# Partially parsed test_generate_files_default_output_dir. Retrieved 7/9 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/example-template'
    var_6 = 'tests/output'
    var_7 = module_0.generate_files(var_5, var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/example-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/example-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, skip_if_file_exists=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/example-template'
    var_6 = 'tests/output'
    var_7 = False
    var_8 = module_0.generate_files(var_5, var_4, var_6, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/example-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, keep_project_on_failure=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'tests/example-template'
    var_2 = 'tests/output'
    var_3 = module_0.generate_files(var_1, var_0, var_2)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/example-template'
    var_6 = module_0.generate_files(var_5, var_4)



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false_when_output_directory_not_created_and_keep_project_on_failure_is_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = False



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/10 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 9/11 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 9/11 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 9/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 9/11 statements.
# Partially parsed test_generate_files_custom_output_dir. Retrieved 8/10 statements.
# Partially parsed test_generate_files_empty_context. Retrieved 4/6 statements.
# Partially parsed test_generate_files_none_context. Retrieved 4/6 statements.
# Partially parsed test_generate_files_default_output_dir. Retrieved 7/9 statements.
# Partially parsed test_generate_files_all_options. Retrieved 9/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = module_0.generate_files(var_5, var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, skip_if_file_exists=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = False
    var_8 = module_0.generate_files(var_5, var_4, var_6, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, keep_project_on_failure=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/custom-output'
    var_7 = module_0.generate_files(var_5, var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'tests/test-template'
    var_2 = 'tests/output'
    var_3 = module_0.generate_files(var_1, var_0, var_2)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-template'
    var_1 = 'tests/output'
    var_2 = None
    var_3 = module_0.generate_files(var_0, var_2, var_1)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = module_0.generate_files(var_5, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, var_7, var_7, var_7, var_7)



# Parsed testcases at query #101
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 9/18 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 9/17 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 11/21 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 15/24 statements.
# Partially parsed test_generate_files_hooks. Retrieved 10/19 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 9/16 statements.
# Partially parsed test_generate_files_no_context. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'Test basic file generation with minimal context.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_templates'
    var_7 = 'basic_template'
    var_8 = 'README.md'

def test_case_0():
    var_0 = 'Test file generation with overwrite enabled.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_templates'
    var_7 = 'basic_template'
    var_8 = True

def test_case_0():
    var_0 = 'Test file generation with skip_if_file_exists enabled.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_templates'
    var_7 = 'basic_template'
    var_8 = 'README.md'
    var_9 = 'Modified content'
    assert var_9 == 'Modified content'
    var_10 = True

def test_case_0():
    var_0 = 'Test file generation with _copy_without_render setting.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_copy_without_render'
    var_4 = 'test_project'
    var_5 = '*.bin'
    var_6 = 'static/*'
    var_7 = [var_5, var_6]
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = {var_1: var_8}
    var_10 = 'test_templates'
    var_11 = 'copy_template'
    var_12 = 'data.bin'
    var_13 = 'static'
    var_14 = 'file.txt'

def test_case_0():
    var_0 = 'Test file generation with hooks enabled.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_templates'
    var_7 = 'hook_template'
    var_8 = True
    var_9 = 'hook_marker.txt'

def test_case_0():
    var_0 = 'Test that project is kept when generation fails and keep_project_on_failure is True.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_templates'
    var_7 = 'failing_template'
    var_8 = True

def test_case_0():
    var_0 = 'Test file generation with no context provided.'
    var_1 = 'test_templates'
    var_2 = 'no_context_template'
    var_3 = 'project'



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 6/15 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 6/14 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 8/18 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 6/13 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test-data'
    var_4 = 'basic-template'
    var_5 = 'README.md'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test-data'
    var_4 = 'basic-template'
    var_5 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test-data'
    var_4 = 'basic-template'
    var_5 = 'existing content'
    var_6 = True
    var_7 = 'new_file.txt'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test-data'
    var_4 = 'template-with-hooks'
    var_5 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test-data'
    var_4 = 'failing-template'
    var_5 = True



# Parsed testcases at query #104
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_os_walk_returns_non_empty_iterator. Retrieved 4/7 statements.


def test_case_0():
    var_0 = '.'
    var_1 = []
    var_2 = []
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_work_in_context_manager_changes_directory. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '/test/directory'



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_cookiecutter_new_lines_predicate. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_json_decoding_error_raises_context_decoding_exception. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = str(var_0)



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_apply_overwrites_to_context_boolean_conversion_failure. Retrieved 6/9 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_template_syntax_error_raises_exception. Retrieved 12/15 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template/dir'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = '{% if %}'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_undefined_error_raised_in_render_and_create_dir. Retrieved 8/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'valid_repo'
    var_1 = 'invalid'
    var_2 = '{{ undefined_var }}'
    var_3 = {var_1: var_2}
    var_4 = 'output'
    var_5 = True
    var_6 = module_0.generate_files(var_0, var_3, var_4, var_5)
    var_7 = str(var_0)



# Parsed testcases at query #114
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'some_binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #115
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Path()
    var_3 = module_1.Environment()
    var_4 = module_2.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #116
#--------------------------

# Partially parsed test_output_dir_exists_predicate. Retrieved 5/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = True



# Parsed testcases at query #117
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'version'
    var_2 = '2.0.0'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'version'
    var_5 = '2.0.0'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_0, var_3, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = None
    var_2 = module_0.generate_context(var_0, var_1, var_1)



# Parsed testcases at query #118
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 14/19 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = True
    var_12 = 'w'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #119
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #120
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #121
#--------------------------

# Partially parsed test_generate_context_with_default_context. Retrieved 11/15 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'cookiecutter.json'
    var_5 = module_0.generate_context(var_4, var_2, var_3)
    var_6 = 'cookiecutter'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = (var_7, var_8)
    var_10 = [var_9]



# Parsed testcases at query #122
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #123
#--------------------------

# Partially parsed test_generate_context_with_default_context_none. Retrieved 4/6 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.generate_context(default_context=var_0)
    var_2 = 'cookiecutter'
    var_3 = []



# Parsed testcases at query #124
#--------------------------

# Partially parsed test_work_in_context_manager_changes_and_restores_directory. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = True



# Parsed testcases at query #125
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #126
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/21 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 9/19 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 10/23 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 10/19 statements.
# Partially parsed test_generate_files_binary_file. Retrieved 8/17 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 4/10 statements.
# Partially parsed test_generate_files_hooks. Retrieved 14/31 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'cookiecutter'
    var_2 = 'test_project'
    var_3 = {}
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'cookiecutter'
    var_2 = 'test_project'
    var_3 = {}
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'cookiecutter'
    var_2 = 'test_project'
    var_3 = {}
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = 'Modified content'
    var_9 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'cookiecutter'
    var_2 = 'test_project'
    var_3 = '_copy_without_render'
    var_4 = '*.md'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = '{{cookiecutter.project_name}}'
    var_9 = 'readme.md'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'cookiecutter'
    var_2 = 'test_project'
    var_3 = {}
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = b'\x00\x01\x02\x03'
    var_7 = 'binary.bin'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '{{cookiecutter.project_name}}'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'cookiecutter'
    var_2 = 'test_project'
    var_3 = {}
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'print("Pre hook executed")'
    var_11 = 'post_gen_project.py'
    var_12 = 'print("Post hook executed")'
    var_13 = True



# Parsed testcases at query #127
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #128
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 9/22 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 10/22 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 10/24 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/22 statements.
# Partially parsed test_generate_files_hooks. Retrieved 13/30 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = '{{cookiecutter.project_name}}'
    assert var_7 == 'test_project'
    var_8 = 'test.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = True
    var_9 = 'test.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = 'existing'
    assert var_8 == 'existing'
    var_9 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.md'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'template'
    var_9 = '{{cookiecutter.project_name}}'
    var_10 = '{{cookiecutter.project_name}}'
    assert var_10 == '{{cookiecutter.project_name}}'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = 'hooks'
    var_8 = 'print("pre hook")'
    var_9 = 'print("post hook")'
    var_10 = '{{cookiecutter.project_name}}'
    var_11 = True
    var_12 = 'test.txt'



# Parsed testcases at query #129
#--------------------------




def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #130
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/10 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 9/11 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 9/11 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 9/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 9/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/tmp/test_output'
    var_6 = '/tmp/test_repo'
    var_7 = module_0.generate_files(var_6, var_4, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/tmp/test_output'
    var_6 = '/tmp/test_repo'
    var_7 = True
    var_8 = module_0.generate_files(var_6, var_4, var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/tmp/test_output'
    var_6 = '/tmp/test_repo'
    var_7 = True
    var_8 = module_0.generate_files(var_6, var_4, var_5, skip_if_file_exists=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/tmp/test_output'
    var_6 = '/tmp/test_repo'
    var_7 = False
    var_8 = module_0.generate_files(var_6, var_4, var_5, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/tmp/test_output'
    var_6 = '/tmp/test_repo'
    var_7 = True
    var_8 = module_0.generate_files(var_6, var_4, var_5, keep_project_on_failure=var_7)



# Parsed testcases at query #131
#--------------------------

# Partially parsed test_os_walk_predicate_false. Retrieved 4/7 statements.


def test_case_0():
    var_0 = '.'
    var_1 = []
    var_2 = []
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #132
#--------------------------

# Partially parsed test_render_and_create_dir_existing_dir_no_overwrite. Retrieved 7/10 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 7/9 statements.
# Partially parsed test_render_and_create_dir_new_dir. Retrieved 7/11 statements.
# Partially parsed test_render_and_create_dir_rendered_name. Retrieved 9/13 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = True
    var_2 = 'existing_dir'
    var_3 = {}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_2, var_3, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = True
    var_2 = 'existing_dir'
    var_3 = {}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_2, var_3, var_4, var_5, var_1)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'new_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = '/tmp/new_dir'
    var_6 = True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '{{ name }}_dir'
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_3, var_2, var_4, var_5)
    var_7 = '/tmp/test_dir'
    var_8 = True



# Parsed testcases at query #133
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'version'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)



# Parsed testcases at query #134
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'valid_context.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #135
#--------------------------

# Partially parsed test_accept_hooks_predicate_true. Retrieved 9/19 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = True



# Parsed testcases at query #136
#--------------------------

# Partially parsed test_template_syntax_error_exception_handling. Retrieved 11/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = '{% invalid syntax %}'
    var_10 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)



# Parsed testcases at query #137
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #138
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = 'new_nested'
    var_7 = 'new_value'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = True
    var_11 = module_0.apply_overwrites_to_context(var_4, var_9, in_dictionary_variable=var_10)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'val1'
    var_4 = 'val2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_val2'
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #139
#--------------------------

# Partially parsed test_skip_if_file_exists. Retrieved 14/22 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = True
    var_12 = 'existing content'
    assert var_12 == 'existing content'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #140
#--------------------------

# Partially parsed test_generate_context_with_default_context_none. Retrieved 3/4 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.generate_context(default_context=var_0)
    var_2 = []



