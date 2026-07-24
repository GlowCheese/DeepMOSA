####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 5/8 statements.
# Partially parsed test_render_and_create_dir_with_existing_dir_no_overwrite. Retrieved 7/12 statements.
# Partially parsed test_render_and_create_dir_with_existing_dir_and_overwrite. Retrieved 6/10 statements.
# Partially parsed test_render_and_create_dir_with_non_existing_dir. Retrieved 6/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = True
    var_2 = 'existing_dir'
    var_3 = {}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = False

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
    var_0 = '/tmp/non_existing_dir'
    var_1 = 'non_existing_dir'
    var_2 = {}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = False



# Parsed testcases at query #3
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'new_var'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.apply_overwrites_to_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing_var'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.apply_overwrites_to_context(var_2, var_5, in_dictionary_variable=var_6)

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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
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
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #4
#--------------------------




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



# Parsed testcases at query #5
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
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary.dat'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)

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

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
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
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_already_exists. Retrieved 8/14 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'file.txt'
    var_5 = True
    var_6 = 'existing content'
    var_7 = module_1.generate_file(var_0, var_1, var_2, var_3, var_5)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_file_name_is_empty_evaluates_to_true. Retrieved 8/10 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template/file.txt'
    var_2 = 'variable'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = ''
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5)



# Parsed testcases at query #8
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
    var_6 = 'x'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

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
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_generate_files_creates_project_directory. Retrieved 8/9 statements.
# Partially parsed test_generate_files_overwrites_existing_directory. Retrieved 9/11 statements.
# Partially parsed test_generate_files_skips_existing_files. Retrieved 9/10 statements.
# Partially parsed test_generate_files_runs_pre_and_post_hooks. Retrieved 9/10 statements.
# Partially parsed test_generate_files_handles_copy_only_paths. Retrieved 11/12 statements.
# Partially parsed test_generate_files_raises_error_on_existing_directory_without_overwrite. Retrieved 10/12 statements.
# Partially parsed test_generate_files_handles_template_syntax_errors. Retrieved 9/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output_dir'
    var_7 = module_0.generate_files(var_0, var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output_dir'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output_dir'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output_dir'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output_dir'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, keep_project_on_failure=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_copy_without_render'
    var_4 = 'test_project'
    var_5 = '*.txt'
    var_6 = [var_5]
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = {var_1: var_7}
    var_9 = '/tmp/output_dir'
    var_10 = module_0.generate_files(var_0, var_8, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = ''
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output_dir'
    var_7 = module_0.generate_files(var_0, var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output_dir'
    var_7 = True
    var_8 = False
    var_9 = module_0.generate_files(var_0, var_5, var_6, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '/tmp/output_dir'
    var_5 = module_0.generate_files(var_0, var_3, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo_dir'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/output_dir'
    var_7 = '{{ invalid_template }}'
    var_8 = module_0.generate_files(var_0, var_5, var_6)



# Parsed testcases at query #10
#--------------------------




import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'invalid_template'
    var_2 = var_0.get_template(var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 6/7 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 7/10 statements.
# Partially parsed test_render_and_create_dir_raises_exception_on_existing_directory. Retrieved 7/10 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = '/tmp/test_dir'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = '/tmp/test_dir'
    var_5 = True
    var_6 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = '/tmp/test_dir'
    var_5 = True
    var_6 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #12
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'output_dir'
    var_3 = None
    var_4 = module_0.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #15
#--------------------------




import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'invalid_template.jinja'
    var_2 = var_0.get_template(var_1)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname. Retrieved 4/7 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'output'
    var_2 = module_0.Environment()
    var_3 = ''



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 9/15 statements.
# Partially parsed test_generate_file_handles_binary_file. Retrieved 9/15 statements.
# Partially parsed test_generate_file_renders_text_file. Retrieved 11/20 statements.
# Partially parsed test_generate_file_uses_custom_newline. Retrieved 11/20 statements.
# Partially parsed test_generate_file_skips_empty_filename. Retrieved 9/12 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
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
    var_1 = 'test.bin'
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
    var_1 = 'test.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'Hello {{ cookiecutter.name }}'
    var_8 = {var_1: var_7}
    var_9 = True
    var_10 = 'Hello {{ cookiecutter.name }}'
    assert var_10 == 'Hello Test'

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'Hello\nWorld'
    var_8 = {var_1: var_7}
    var_9 = True
    var_10 = 'Hello\nWorld'
    assert var_10 == b'Hello\r\nWorld'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_8 = ''



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_exists. Retrieved 9/14 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/test/project'
    var_1 = 'test.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'test'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)



# Parsed testcases at query #19
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
    var_6 = 'x'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_value'
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

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
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_file_skips_existing_file. Retrieved 10/15 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 8/12 statements.
# Partially parsed test_generate_file_renders_text_file. Retrieved 10/14 statements.
# Partially parsed test_generate_file_uses_custom_newline. Retrieved 10/14 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 9/12 statements.


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
    var_6 = b'\x00\x01\x02\x03'
    assert var_6 == b'\x00\x01\x02\x03'
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = 'Hello {{ cookiecutter.name }}'
    assert var_8 == 'Hello Test'
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7)

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
    var_8 = 'Line 1\nLine 2'
    assert var_8 == 'Line 1\r\nLine 2'
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7)

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



# Parsed testcases at query #21
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'output'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '   '
    var_1 = {}
    var_2 = 'output'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_generate_file_creates_file_with_rendered_content. Retrieved 11/13 statements.
# Partially parsed test_generate_file_skips_existing_file. Retrieved 13/19 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 11/13 statements.
# Partially parsed test_generate_file_applies_file_permissions. Retrieved 13/18 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'w'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary.bin'
    var_2 = 'cookiecutter'
    var_3 = 'var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = b'\x00\x01\x02'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = 'var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 420
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_12 = 511



# Parsed testcases at query #23
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
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'binary_file.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_exists. Retrieved 8/11 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'template.txt'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_empty_dirname_raises_exception. Retrieved 3/10 statements.
# Partially parsed test_whitespace_dirname_raises_exception. Retrieved 3/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = '   '
    var_1 = {}
    var_2 = module_0.Environment()



# Parsed testcases at query #26
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'path/to/binary/file'
    var_1 = module_0.is_binary(var_0)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generate_file_creates_file_with_rendered_content. Retrieved 13/18 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 13/18 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 13/18 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 12/15 statements.
# Partially parsed test_generate_file_uses_custom_newline. Retrieved 13/18 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = '{{ cookiecutter.key }}'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'existing content'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary_file.bin'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp'
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
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = '{{ cookiecutter._new_lines }}'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #28
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_0, var_3, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_0, var_3, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = module_0.generate_context(var_0, var_3, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = None
    var_2 = 'key2'
    var_3 = 'value2'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_1, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = None
    var_2 = None
    var_3 = module_0.generate_context(var_0, var_1, var_2)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_exists. Retrieved 7/14 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'existing content'
    assert var_5 == 'existing content'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3, var_5)



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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
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
    var_0 = 'multichoice'
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
    var_0 = 'multichoice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_2, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

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
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/tmp/repo'
    var_1 = 'pre_gen_project'
    var_2 = '/tmp/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'Test Project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_value'
    var_4 = {var_0: var_3}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname_raises_exception. Retrieved 4/7 statements.
# Partially parsed test_render_and_create_dir_with_existing_dir_and_no_overwrite_raises_exception. Retrieved 6/12 statements.
# Partially parsed test_render_and_create_dir_with_existing_dir_and_overwrite_returns_dir_and_false. Retrieved 5/10 statements.
# Partially parsed test_render_and_create_dir_with_non_existing_dir_creates_dir_and_returns_dir_and_true. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_with_rendered_dirname_creates_correct_dir. Retrieved 7/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = module_0.Environment()
    var_3 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = module_0.Environment()
    var_3 = 'existing_dir'
    var_4 = True
    var_5 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = module_0.Environment()
    var_3 = 'existing_dir'
    var_4 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = module_0.Environment()
    var_3 = 'new_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = '{{ name }}_dir'
    var_6 = 'project_dir'



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-generate-context-valid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-generate-context-invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-generate-context-valid.json'
    var_1 = 'key1'
    var_2 = 'new_value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-generate-context-valid.json'
    var_1 = 'key1'
    var_2 = 'new_value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-generate-context-valid.json'
    var_1 = 'key1'
    var_2 = 'default_value1'
    var_3 = {var_1: var_2}
    var_4 = 'new_value1'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-generate-context-valid.json'
    var_1 = 'invalid_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-generate-context-valid.json'
    var_1 = 'invalid_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)



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

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = '0'
    var_2 = var_0.process_response(var_1)
    assert var_2 is False
    var_3 = 'false'
    var_4 = var_0.process_response(var_3)
    assert var_4 is False
    var_5 = 'f'
    var_6 = var_0.process_response(var_5)
    assert var_6 is False
    var_7 = 'no'
    var_8 = var_0.process_response(var_7)
    assert var_8 is False
    var_9 = 'n'
    var_10 = var_0.process_response(var_9)
    assert var_10 is False
    var_11 = 'off'
    var_12 = var_0.process_response(var_11)
    assert var_12 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'invalid'
    var_2 = var_0.process_response(var_1)



# Parsed testcases at query #7
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/fixtures/invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 5/7 statements.
# Partially parsed test_render_and_create_dir_raises_exception_for_existing_directory. Retrieved 7/10 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 6/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = '/tmp/test_dir'

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
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = '/tmp/existing_dir'
    var_5 = True
    var_6 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = '/tmp/existing_dir'
    var_5 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_generate_files_creates_project_dir. Retrieved 8/9 statements.
# Partially parsed test_generate_files_overwrites_existing_dir. Retrieved 9/12 statements.
# Partially parsed test_generate_files_skips_existing_files. Retrieved 11/18 statements.
# Partially parsed test_generate_files_runs_hooks. Retrieved 9/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/fake/output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/fake/output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/fake/output'
    var_7 = 'existing_file.txt'
    var_8 = 'w'
    var_9 = True
    var_10 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/fake/output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'cookiecutter'
    var_2 = 'invalid_var'
    var_3 = '{{ undefined_var }}'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/fake/output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, keep_project_on_failure=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/fake/repo'
    var_1 = 'cookiecutter'
    var_2 = 'invalid_var'
    var_3 = '{{ undefined_var }}'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/fake/output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)



# Parsed testcases at query #10
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_choice'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #11
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
    var_0 = 'nested'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.apply_overwrites_to_context(var_2, var_6, in_dictionary_variable=var_7)

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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
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



# Parsed testcases at query #12
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'invalid_input'
    var_2 = var_0.process_response(var_1)



# Parsed testcases at query #13
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'test_output'
    var_3 = module_0.Environment()
    var_4 = False
    var_5 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 6/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = True



# Parsed testcases at query #15
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'invalid'
    var_2 = var_0.process_response(var_1)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_generate_file_handles_binary_file. Retrieved 7/12 statements.
# Partially parsed test_generate_file_renders_text_file. Retrieved 11/16 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 9/17 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 7/12 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/binary_file.bin'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = b'binary content'
    assert var_5 == b'binary content'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3)

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
    var_9 = '{{ cookiecutter.variable }}'
    assert var_9 == 'value'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/skip_file.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'content'
    var_6 = 'skip_file.txt'
    var_7 = 'existing content'
    assert var_7 == 'existing content'
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_exists. Retrieved 6/16 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'existing file'
    var_2 = module_0.Environment()
    var_3 = {}
    var_4 = 'test.txt'
    var_5 = True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_exists. Retrieved 7/12 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'test.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'existing content'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3, var_5)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 5/7 statements.
# Partially parsed test_render_and_create_dir_raises_exception_for_existing_directory. Retrieved 6/9 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 6/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = '/tmp/test_dir'

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
    var_4 = True
    var_5 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = '/tmp/test_dir'



# Parsed testcases at query #21
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/test/project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_render_and_create_dir_overwrites_existing_dir_when_overwrite_flag_is_true. Retrieved 5/15 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 10/17 statements.
# Partially parsed test_generate_file_text. Retrieved 10/17 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 11/19 statements.
# Partially parsed test_generate_file_empty_name. Retrieved 9/13 statements.
# Partially parsed test_generate_file_newline_configuration. Retrieved 14/21 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.bin'
    var_2 = 'key'
    var_3 = 'value'
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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'Hello {{ key }}'
    assert var_7 == 'Hello value'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_9 = 'file.txt'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'Hello {{ key }}'
    var_8 = 'file.txt'
    var_9 = 'Existing content'
    assert var_9 == 'Existing content'
    var_10 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/'
    var_2 = 'key'
    var_3 = 'value'
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
    var_3 = 'key'
    var_4 = '_new_lines'
    var_5 = '\r\n'
    var_6 = {var_4: var_5}
    var_7 = 'value'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = module_0.Environment()
    var_10 = True
    var_11 = 'Hello {{ key }}'
    assert var_11 == 'Hello value\r\n'
    var_12 = module_1.generate_file(var_0, var_1, var_8, var_9)
    var_13 = 'file.txt'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_generate_file_creates_file_with_rendered_content. Retrieved 13/18 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 14/21 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 13/18 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 12/14 statements.
# Partially parsed test_generate_file_uses_configured_newline. Retrieved 15/20 statements.
# Partially parsed test_generate_file_applies_file_permissions. Retrieved 16/24 statements.


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
    var_11 = 'Hello {{ cookiecutter.variable }}'
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
    var_11 = 'Hello {{ cookiecutter.variable }}'
    var_12 = 'Existing content'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary_file.bin'
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

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = '_new_lines'
    var_5 = 'value'
    var_6 = '\r\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello {{ cookiecutter.variable }}'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)

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
    var_11 = 'Hello {{ cookiecutter.variable }}'
    var_12 = 420
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_14 = 'template.txt'
    var_15 = 511



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false_when_output_directory_not_created. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = False



# Parsed testcases at query #26
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
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary.dat'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)

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

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
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
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 5/7 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 6/10 statements.
# Partially parsed test_render_and_create_dir_raises_exception_for_existing_directory. Retrieved 7/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = '/tmp/test_dir'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = '/tmp/test_dir'
    var_5 = True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = '/tmp/test_dir'
    var_5 = True
    var_6 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #29
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_generate_context_with_valid_json_file. Retrieved 12/24 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = -1
    var_5 = '/'
    var_6 = tmp_path.split(var_5)[var_4]
    var_7 = '.'
    var_8 = var_4.split(var_7)[var_3]
    var_9 = -1
    var_10 = tmp_path.split(var_5)[var_9]
    var_11 = var_8.split(var_7)[var_3]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_template_syntax_error_raised. Retrieved 9/11 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = "raise TemplateSyntaxError('error', 1, 'test', 'test')"
    var_7 = exec(var_6)
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)



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
    var_6 = 'x'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

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
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_render_and_create_dir_successful_creation. Retrieved 5/18 statements.
# Partially parsed test_render_and_create_dir_already_exists_no_overwrite. Retrieved 4/19 statements.
# Partially parsed test_render_and_create_dir_already_exists_with_overwrite. Retrieved 5/20 statements.
# Partially parsed test_render_and_create_dir_with_template_variables. Retrieved 8/21 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ name }}'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = ''
    var_3 = '/tmp'
    var_4 = module_1.render_and_create_dir(var_2, var_1, var_3, var_0)

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'existing_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'existing_dir'
    var_4 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project'
    var_2 = 'version'
    var_3 = 'awesome'
    var_4 = '1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = '{{ project }}-v{{ version }}'
    var_7 = 'awesome-v1.0'



# Parsed testcases at query #3
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/example.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'templates/*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/example.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'other/*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/example.txt'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.is_copy_only_path(var_0, var_3)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/example.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = []
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.is_copy_only_path(var_0, var_5)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/example.txt'
    var_1 = None
    var_2 = module_0.is_copy_only_path(var_0, var_1)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'templates/*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 9/12 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context_only. Retrieved 6/9 statements.
# Partially parsed test_generate_context_with_extra_context_only. Retrieved 6/9 statements.
# Partially parsed test_generate_context_without_default_or_extra_context. Retrieved 3/6 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = '{"key1": "default_value1", "key2": "default_value2"}'
    var_8 = module_0.generate_context(var_0, var_3, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'invalid json'
    var_2 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = '{"key1": "default_value1"}'
    var_5 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'key2'
    var_2 = 'value2'
    var_3 = {var_1: var_2}
    var_4 = '{"key2": "default_value2"}'
    var_5 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"key1": "default_value1"}'
    var_2 = module_0.generate_context(var_0)



# Parsed testcases at query #5
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'invalid'
    var_2 = var_0.process_response(var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecation_warning. Retrieved 10/19 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_run_hook_from_repo_dir. Retrieved 8/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'hook_name'
    var_2 = '/path/to/project'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 'always'
    var_8 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)
    var_9 = 0

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'hook_name'
    var_2 = '/path/to/project'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_generate_file_creates_file_with_rendered_content. Retrieved 13/18 statements.
# Partially parsed test_generate_file_skips_existing_file_with_skip_if_file_exists. Retrieved 14/21 statements.
# Partially parsed test_generate_file_copies_binary_file_without_rendering. Retrieved 13/18 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 13/16 statements.
# Partially parsed test_generate_file_uses_new_line_from_context. Retrieved 15/20 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Project Name: {{ cookiecutter.name }}'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Project Name: {{ cookiecutter.name }}'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary.bin'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
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
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
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
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = 'Test Project'
    var_6 = '\r\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Project Name: {{ cookiecutter.name }}'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_file_name_is_empty_evaluates_to_true. Retrieved 4/9 statements.


def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'template/file.txt'
    var_2 = {}
    var_3 = None



# Parsed testcases at query #9
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'some_dir'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #10
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_dir. Retrieved 4/15 statements.
# Partially parsed test_render_and_create_dir_raises_on_existing_dir. Retrieved 5/19 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_dir. Retrieved 4/17 statements.
# Partially parsed test_render_and_create_dir_renders_template. Retrieved 6/17 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'output'
    var_3 = module_0.Environment()
    var_4 = False
    var_5 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3, var_4)

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'rendered'
    var_2 = {var_0: var_1}
    var_3 = '{{ name }}'
    var_4 = module_0.Environment()
    var_5 = False



# Parsed testcases at query #12
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/binary_file.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/empty_file'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = module_1.generate_file(var_0, var_1, var_6, var_7)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_exists. Retrieved 9/15 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'template.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'template.txt'
    var_6 = True
    var_7 = 'existing content'
    var_8 = module_1.generate_file(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_render_and_create_dir_existing_output_dir. Retrieved 6/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp/test'
    var_3 = module_0.Environment()
    var_4 = False
    var_5 = True



# Parsed testcases at query #15
#--------------------------




import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'invalid_template.jinja2'
    var_2 = var_0.get_template(var_1)



# Parsed testcases at query #16
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_choice'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #17
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 10/13 statements.
# Partially parsed test_generate_file_text_file. Retrieved 8/12 statements.
# Partially parsed test_generate_file_skip_if_file_exists. Retrieved 10/19 statements.
# Partially parsed test_generate_file_empty_file_name. Retrieved 7/10 statements.
# Partially parsed test_generate_file_new_lines_config. Retrieved 10/13 statements.


import jinja2.environment as module_0
import codecs as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/binary_file'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'wb'
    var_7 = module_1.open(var_1, var_6)
    var_8 = b'\x00\x01\x02'
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'Hello, World!'
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'Hello, World!'
    var_7 = True
    var_8 = 'Existing Content'
    assert var_8 == 'Existing Content'
    var_9 = module_1.generate_file(var_0, var_1, var_4, var_5, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/empty_file'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = 'Hello, World!'
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7)



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = '/path/to/binary/file'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = module_1.generate_file(var_0, var_1, var_2, var_3)



# Parsed testcases at query #21
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'invalid'
    var_2 = var_0.process_response(var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_generate_context_with_valid_context_file. Retrieved 2/3 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-context.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid-context.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test-context.json'
    var_4 = module_0.generate_context(var_3, var_2)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key2'
    var_1 = 'value2'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test-context.json'
    var_4 = module_0.generate_context(var_3, extra_context=var_2)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid_key'
    var_1 = 'invalid_value'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test-context.json'
    var_4 = module_0.generate_context(var_3, var_2)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid_key'
    var_1 = 'invalid_value'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test-context.json'
    var_4 = module_0.generate_context(var_3, extra_context=var_2)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_raises_on_existing_dir. Retrieved 7/16 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_dir. Retrieved 5/12 statements.
# Partially parsed test_render_and_create_dir_renders_template. Retrieved 6/11 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = False
    var_5 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3, var_4)

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'output'
    var_1 = 'test'
    var_2 = {}
    var_3 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'output'
    var_1 = 'test'
    var_2 = True
    var_3 = 'test'
    var_4 = {}
    var_5 = module_0.Environment()
    var_6 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'output'
    var_1 = 'test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'output'
    var_1 = 'name'
    var_2 = 'rendered'
    var_3 = {var_1: var_2}
    var_4 = '{{ name }}'
    var_5 = module_0.Environment()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_render_and_create_dir_existing_output_dir. Retrieved 5/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = True



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_cookiecutter_new_lines_set_to_true. Retrieved 7/8 statements.
# Partially parsed test_cookiecutter_new_lines_set_to_false. Retrieved 6/7 statements.
# Partially parsed test_cookiecutter_new_lines_not_set. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #27
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'output_dir'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_generate_file_creates_binary_file. Retrieved 9/14 statements.
# Partially parsed test_generate_file_creates_text_file. Retrieved 9/14 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 11/19 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 9/13 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.bin'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = b'binary content'
    assert var_7 == b'binary content'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'Hello, {{ key }}!'
    assert var_7 == 'Hello, value!'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/file.txt'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'Hello, {{ key }}!'
    var_8 = 'file.txt'
    var_9 = 'Existing content'
    assert var_9 == 'Existing content'
    var_10 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_8 = ''



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 8/13 statements.
# Partially parsed test_generate_file_text_file. Retrieved 10/18 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 9/17 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 7/10 statements.
# Partially parsed test_generate_file_newlines. Retrieved 12/20 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/binary_file.bin'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = b'\x00\x01\x02\x03'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_7 = 'binary_file.bin'

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/text_file.txt'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 'text_file.txt'
    var_6 = 'Hello {{ name }}'
    var_7 = {var_5: var_6}
    var_8 = True
    var_9 = 'Hello {{ name }}'
    assert var_9 == 'Hello test'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/text_file.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'content'
    var_6 = 'text_file.txt'
    var_7 = 'existing content'
    assert var_7 == 'existing content'
    var_8 = module_1.generate_file(var_0, var_1, var_2, var_3, var_7)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = ''
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_6 = ''

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'text_file.txt'
    var_8 = 'line1\nline2'
    var_9 = {var_7: var_8}
    var_10 = True
    var_11 = 'line1\nline2'
    assert var_11 == b'line1\r\nline2'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_generate_files_creates_project_directory. Retrieved 8/9 statements.
# Partially parsed test_generate_files_handles_copy_only_paths. Retrieved 13/17 statements.
# Partially parsed test_generate_files_raises_on_existing_output_without_overwrite. Retrieved 9/12 statements.
# Partially parsed test_generate_files_overwrites_existing_output. Retrieved 9/12 statements.
# Partially parsed test_generate_files_runs_pre_and_post_hooks. Retrieved 12/19 statements.
# Partially parsed test_generate_files_skips_hooks_when_disabled. Retrieved 11/16 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_copy_without_render'
    var_4 = 'test_project'
    var_5 = '*.txt'
    var_6 = [var_5]
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = {var_1: var_7}
    var_9 = 'output'
    var_10 = 'test'
    var_11 = module_0.generate_files(var_0, var_8, var_9)
    var_12 = 'test.txt'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'output'
    var_7 = False
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'output'
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
    var_6 = 'output'
    var_7 = 'hooks'
    var_8 = "print('pre hook')"
    var_9 = "print('post hook')"
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
    var_6 = 'output'
    var_7 = 'hooks'
    var_8 = "raise Exception('This should not run')"
    var_9 = False
    var_10 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'invalid_var'
    var_3 = '{{ invalid }}'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, keep_project_on_failure=var_7)



# Parsed testcases at query #31
#--------------------------




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
    var_0 = '   '
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_skip_if_file_exists_and_file_exists. Retrieved 12/18 statements.


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
    var_8 = 'template.txt'
    var_9 = True
    var_10 = 'existing content'
    var_11 = module_1.generate_file(var_0, var_1, var_6, var_7, var_9)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_render_and_create_dir_successful_creation. Retrieved 5/18 statements.
# Partially parsed test_render_and_create_dir_already_exists_no_overwrite. Retrieved 4/19 statements.
# Partially parsed test_render_and_create_dir_already_exists_with_overwrite. Retrieved 5/19 statements.
# Partially parsed test_render_and_create_dir_with_nested_path. Retrieved 6/20 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ name }}'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = ''
    var_3 = 'some/path'
    var_4 = module_1.render_and_create_dir(var_2, var_1, var_3, var_0)

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'existing'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'existing'
    var_4 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'parent/{{ name }}'
    var_5 = 'parent'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 7/12 statements.
# Partially parsed test_generate_file_text. Retrieved 9/19 statements.
# Partially parsed test_generate_file_skip_exists. Retrieved 8/16 statements.
# Partially parsed test_generate_file_newlines. Retrieved 11/20 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_binary.bin'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = b'\x00\x01\x02\x03'
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'name'
    var_3 = 'Test'
    var_4 = {var_2: var_3}
    var_5 = 'Hello {{ name }}'
    var_6 = {var_1: var_5}
    var_7 = True
    var_8 = 'Hello {{ name }}'
    assert var_8 == 'Hello Test'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_skip.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = 'Test'
    var_6 = 'Existing'
    assert var_6 == 'Existing'
    var_7 = module_1.generate_file(var_0, var_1, var_2, var_3, var_6)

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_newlines.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'Line1\nLine2'
    var_8 = {var_1: var_7}
    var_9 = True
    var_10 = 'Line1\nLine2'
    assert var_10 == 'Line1\r\nLine2'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_render_and_create_dir_creates_directory_when_not_exists. Retrieved 5/8 statements.
# Partially parsed test_render_and_create_dir_raises_exception_when_dirname_empty. Retrieved 4/7 statements.
# Partially parsed test_render_and_create_dir_raises_exception_when_dir_exists_and_no_overwrite. Retrieved 5/10 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 6/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = '/tmp/test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = True
    var_5 = '/tmp/existing_dir'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 7/12 statements.
# Partially parsed test_render_and_create_dir_raises_error_for_existing_directory. Retrieved 8/15 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 8/15 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = '{{ name }}_dir'
    var_6 = '/tmp/test_dir'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = ''
    var_6 = module_1.render_and_create_dir(var_5, var_2, var_3, var_4)

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = '{{ name }}_dir'
    var_6 = '/tmp/test_dir'
    var_7 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = '{{ name }}_dir'
    var_6 = '/tmp/test_dir'
    var_7 = True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_generate_file_with_binary_file. Retrieved 10/15 statements.
# Partially parsed test_generate_file_with_text_file. Retrieved 10/15 statements.
# Partially parsed test_generate_file_with_skip_if_file_exists. Retrieved 9/14 statements.
# Partially parsed test_generate_file_with_new_lines. Retrieved 13/20 statements.
# Partially parsed test_generate_file_with_empty_file_name. Retrieved 9/14 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/binary_file'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = b'\x00\x01\x02\x03'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_9 = 'binary_file'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'Hello, World!'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)
    var_9 = 'text_file.txt'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'Existing content'
    assert var_7 == 'Existing content'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/new_lines_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = 'Line 1\nLine 2'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)
    var_11 = -1
    var_12 = '\r\n'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/empty_file_name'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'empty_file_name'
    var_8 = module_1.generate_file(var_0, var_1, var_4, var_5)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_generate_file_handles_binary_file. Retrieved 6/10 statements.
# Partially parsed test_generate_file_renders_text_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_skips_existing_file. Retrieved 7/12 statements.
# Partially parsed test_generate_file_handles_empty_file_name. Retrieved 6/8 statements.
# Partially parsed test_generate_file_preserves_file_permissions. Retrieved 9/17 statements.
# Partially parsed test_generate_file_handles_newline_configuration. Retrieved 12/16 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/binary_file'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = b'\x00\x01\x02\x03'
    var_5 = module_1.generate_file(var_0, var_1, var_2, var_3)
    assert var_5 == b'\x00\x01\x02\x03'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = '{{ cookiecutter.variable }}'
    assert var_10 == 'value'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/skip_file.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'existing content'
    assert var_4 == 'existing content'
    var_5 = True
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_5 = ''

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/permissions_file'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'content'
    var_5 = 420
    var_6 = module_1.generate_file(var_0, var_1, var_2, var_3)
    var_7 = 'permissions_file'
    var_8 = 511

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/template/newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'line1\nline2'
    assert var_10 == 'line1\r\nline2'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 6/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = '/test/project'
    var_1 = 'test.txt'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = var_3.from_string(var_1)
    var_5 = 'test content'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_is_binary_returns_true_for_binary_file. Retrieved 8/10 statements.


import jinja2.environment as module_0
import binaryornot.check as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '/tmp/binary_file'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = b'\x00\x01\x02\x03'
    var_7 = module_1.is_binary(var_1)
    assert var_7 is True



