####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecated_warning. Retrieved 15/37 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_actual_function. Retrieved 14/25 statements.
# Partially parsed test_run_hook_from_repo_dir_with_false_delete_flag. Retrieved 9/20 statements.
# Partially parsed test_run_hook_from_repo_dir_with_true_delete_flag. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir issues a deprecation warning.'
    var_1 = 'repo'
    var_2 = 'project'
    assert var_2 == 1
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = None
    var_9 = []
    var_10 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_11 = 'always'
    var_12 = 0
    var_13 = var_3.category
    var_14 = len(var_9)
    assert var_14 == 1

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir delegates to run_hook_from_repo_dir.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_gen_project'
    var_9 = True
    var_10 = []
    var_11 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_12 = 'always'
    var_13 = len(var_10)
    assert var_13 == 1

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with delete_project_on_failure=False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = {}
    var_4 = 'post_gen_project'
    var_5 = []
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = 'always'
    var_8 = False

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with delete_project_on_failure=True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = {}
    var_4 = 'pre_gen_project'
    var_5 = []
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = 'always'
    var_8 = True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecated_function. Retrieved 14/24 statements.
# Partially parsed test_run_hook_from_repo_dir_passes_all_arguments. Retrieved 14/18 statements.
# Partially parsed test_run_hook_from_repo_dir_with_delete_true. Retrieved 9/13 statements.
# Partially parsed test_run_hook_from_repo_dir_with_delete_false. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir calls run_hook_from_repo_dir and issues deprecation warning.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'cookiecutter.generate.warnings.warn'
    var_3 = 'repo'
    var_4 = 'project'
    var_5 = 'post_gen_project'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = True
    var_12 = "The '_run_hook_from_repo_dir' function is deprecated, use 'cookiecutter.hooks.run_hook_from_repo_dir' instead"
    var_13 = 2

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir passes all arguments correctly.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'cookiecutter.generate.warnings.warn'
    var_3 = 'repo_path'
    var_4 = 'project_path'
    var_5 = 'pre_prompt'
    var_6 = 'cookiecutter'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = False
    var_12 = module_0._run_hook_from_repo_dir(var_3, var_5, var_4, var_10, var_11)
    var_13 = False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with delete_project_on_failure=True.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'cookiecutter.generate.warnings.warn'
    var_3 = 'repo'
    var_4 = 'hook'
    var_5 = 'project'
    var_6 = {}
    var_7 = True
    var_8 = module_0._run_hook_from_repo_dir(var_3, var_4, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with delete_project_on_failure=False.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'cookiecutter.generate.warnings.warn'
    var_3 = 'repo'
    var_4 = 'hook'
    var_5 = 'project'
    var_6 = {}
    var_7 = False
    var_8 = module_0._run_hook_from_repo_dir(var_3, var_4, var_5, var_6, var_7)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 11/24 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 8/21 statements.
# Partially parsed test_generate_file_skips_existing_file. Retrieved 10/23 statements.
# Partially parsed test_generate_file_returns_when_file_name_empty. Retrieved 9/19 statements.
# Partially parsed test_generate_file_uses_configured_newline. Retrieved 10/23 statements.
# Partially parsed test_generate_file_renders_filename_with_context. Retrieved 11/24 statements.
# Partially parsed test_generate_file_detects_newline_from_source. Retrieved 8/20 statements.
# Partially parsed test_generate_file_preserves_file_permissions. Retrieved 9/26 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test_{{cookiecutter.name}}.txt'
    var_3 = 'Hello {{cookiecutter.name}}!'
    var_4 = module_0.Environment()
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'World'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'test_World.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'binary.bin'
    var_3 = b'\x89PNG\r\n\x1a\n'
    var_4 = module_0.Environment()
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'Template content'
    var_4 = 'Existing content'
    var_5 = module_0.Environment()
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'content'
    var_4 = module_0.Environment()
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'line1\nline2\n'
    var_4 = module_0.Environment()
    var_5 = 'cookiecutter'
    var_6 = '_new_lines'
    var_7 = '\r\n'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = '{{cookiecutter.filename}}.txt'
    var_3 = 'content'
    var_4 = module_0.Environment()
    var_5 = 'cookiecutter'
    var_6 = 'filename'
    var_7 = 'myfile'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'myfile.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = b'line1\nline2\n'
    var_4 = module_0.Environment()
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'content'
    var_4 = 493
    var_5 = module_0.Environment()
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate_evaluates_to_true. Retrieved 9/30 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 39 evaluates to True when conditions are met.'
    var_1 = 'test_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'existing content'
    var_7 = 'test content'
    var_8 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 4/10 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 7/13 statements.
# Partially parsed test_render_and_create_dir_existing_dir_raises_exception. Retrieved 5/13 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 5/13 statements.
# Partially parsed test_render_and_create_dir_creates_nested_directories. Retrieved 4/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that EmptyDirNameException is raised when dirname is empty.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = "Test that a new directory is created when it doesn't exist."
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that directory name is rendered using Jinja2 template.'
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = '{{ project_name }}_dir'
    var_6 = 'my_project_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that OutputDirExistsException is raised when directory exists and overwrite_if_exists is False.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'existing_dir'
    var_4 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that existing directory is allowed when overwrite_if_exists is True.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'existing_dir'
    var_4 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that nested directory paths are created correctly.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'parent/child/grandchild'



# Parsed testcases at query #6
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_key'
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'option1'
    var_2 = 'option2'
    var_3 = 'option3'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'option1'
    var_2 = 'option2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'invalid'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)

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
    var_6 = 'invalid'
    var_7 = [var_2, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'true'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'false'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'db'
    var_2 = 'port'
    var_3 = 'sqlite'
    var_4 = 3306
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 5432
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'items'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_3, var_2]
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_7, var_10)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'count'
    var_2 = 'enabled'
    var_3 = 'old'
    var_4 = 5
    var_5 = False
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'new'
    var_8 = 10
    var_9 = 'yes'
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_6, var_10)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'value'
    var_2 = 'original'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_0.apply_overwrites_to_context(var_4, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'count'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = 10
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = '1'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = '0'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_generate_context_with_valid_json_file. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_invalid_json_raises_exception. Retrieved 4/8 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_both_default_and_extra_context. Retrieved 9/13 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_list_choice. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_boolean_variable_true. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_boolean_variable_false. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_invalid_boolean_conversion. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_preserves_order. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'Test generate_context with a valid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON raises ContextDecodingException.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"invalid json}'
    var_3 = module_0.generate_context(var_0)

def test_case_0():
    var_0 = 'Test generate_context applies default_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "default_project", "author": "default_author"}'
    var_3 = 'project_name'
    var_4 = 'overridden_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context applies extra_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "original", "version": "1.0"}'
    var_3 = 'version'
    var_4 = '2.0'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with both default and extra context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "original", "version": "1.0", "author": "original_author"}'
    var_3 = 'project_name'
    var_4 = 'default_override'
    var_5 = {var_3: var_4}
    var_6 = 'version'
    var_7 = '3.0'
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary structures.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project": {"name": "test", "version": "1.0"}}'
    var_3 = 'project'
    var_4 = 'version'
    var_5 = '2.0'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test generate_context with list choice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context converts string to boolean True.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": false}'
    var_3 = 'use_docker'
    var_4 = 'yes'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context converts string to boolean False.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'no'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with invalid boolean conversion.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'maybe'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with multichoice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["auth", "api", "admin"]}'
    var_3 = 'features'
    var_4 = 'api'
    var_5 = 'admin'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with non-existent file raises exception.'
    var_1 = '/non/existent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

def test_case_0():
    var_0 = 'Test generate_context preserves order of variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"first": "1", "second": "2", "third": "3"}'
    var_3 = 'cookiecutter'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_template_syntax_error_exception_translated_false. Retrieved 8/26 statements.


import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'test_template.txt'
    var_1 = 'test content'
    var_2 = 'test syntax error'
    var_3 = 1
    var_4 = module_0.TemplateSyntaxError(var_2, var_3)
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_generate_context_json_decoding_error. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'Test that ValueError is caught at line 20 and ContextDecodingException is raised.'
    var_1 = '{invalid json content'
    var_2 = False
    var_3 = None
    var_4 = True
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_template_syntax_error_predicate. Retrieved 3/6 statements.


import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'test error'
    var_1 = 1
    var_2 = module_0.TemplateSyntaxError(var_0, var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_is_binary_predicate_evaluates_to_true. Retrieved 12/28 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 47 evaluates to True for binary files.'
    var_1 = 'binary_file.bin'
    var_2 = b'\x89PNG\r\n\x1a\n'
    var_3 = 'output'
    var_4 = '__main__.is_binary'
    var_5 = True
    var_6 = '__main__.shutil.copyfile'
    var_7 = '__main__.shutil.copymode'
    var_8 = module_0.Environment()
    var_9 = 'cookiecutter'
    var_10 = {}
    var_11 = {var_9: var_10}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_true. Retrieved 4/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_file_name_is_empty_predicate_evaluates_to_true. Retrieved 11/28 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 35 evaluates to True when outfile is a directory.'
    var_1 = 'project'
    var_2 = 'some_dir'
    var_3 = 'templates'
    var_4 = 'test.txt'
    var_5 = 'test content'
    var_6 = module_0.Environment()
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_generate_context_catches_json_decode_error. Retrieved 4/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that generate_context catches ValueError (JSON decoding error) at line 20.'
    var_1 = 'cookiecutter.json'
    var_2 = '{invalid json content'
    var_3 = module_0.generate_context(var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 11/20 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello {{ name }}'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = var_6[var_2]
    var_9 = '_new_lines'
    var_10 = False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_render_and_create_dir_with_valid_dirname. Retrieved 7/12 statements.
# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 4/8 statements.
# Partially parsed test_render_and_create_dir_none_dirname_raises_exception. Retrieved 4/8 statements.
# Partially parsed test_render_and_create_dir_existing_dir_overwrite_false_raises_exception. Retrieved 8/15 statements.
# Partially parsed test_render_and_create_dir_existing_dir_overwrite_true_succeeds. Retrieved 7/14 statements.
# Partially parsed test_render_and_create_dir_with_context_variables. Retrieved 9/14 statements.
# Partially parsed test_render_and_create_dir_returns_tuple_with_path_and_flag. Retrieved 8/19 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir creates directory with rendered name.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = '{{ project_name }}'
    var_6 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises EmptyDirNameException for empty dirname.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises EmptyDirNameException for None dirname.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = None

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises OutputDirExistsException when dir exists and overwrite_if_exists is False.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'existing_project'
    var_4 = {var_2: var_3}
    var_5 = '{{ project_name }}'
    var_6 = True
    var_7 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir succeeds when dir exists and overwrite_if_exists is True.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'existing_project'
    var_4 = {var_2: var_3}
    var_5 = '{{ project_name }}'
    var_6 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir renders dirname with multiple context variables.'
    var_1 = module_0.Environment()
    var_2 = 'org'
    var_3 = 'repo'
    var_4 = 'myorg'
    var_5 = 'myrepo'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '{{ org }}/{{ repo }}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir returns tuple with Path and boolean flag.'
    var_1 = module_0.Environment()
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = '{{ name }}'
    var_6 = False
    var_7 = 1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generate_context_catches_value_error_on_invalid_json. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'Test that ValueError is caught and ContextDecodingException is raised at line 20.'
    var_1 = '{"invalid json": '
    var_2 = False
    var_3 = True
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_render_and_create_dir_raises_empty_dir_name_exception. Retrieved 3/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = ''



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 13/27 statements.
# Partially parsed test_generate_files_with_none_context. Retrieved 9/21 statements.
# Partially parsed test_generate_files_overwrite_if_exists. Retrieved 14/26 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 14/26 statements.
# Partially parsed test_generate_files_without_hooks. Retrieved 14/26 statements.
# Partially parsed test_generate_files_keep_project_on_failure. Retrieved 14/26 statements.
# Partially parsed test_generate_files_with_binary_file. Retrieved 15/28 statements.
# Partially parsed test_generate_files_with_copy_without_render. Retrieved 17/32 statements.
# Partially parsed test_generate_files_default_output_dir. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'Test generate_files with basic template structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'test.txt'
    var_5 = 'Hello {{cookiecutter.project_name}}'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'

def test_case_0():
    var_0 = 'Test generate_files with None context.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'test.txt'
    var_5 = 'test'
    var_6 = 'cookiecutter.generate.find_template'
    var_7 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_8 = None

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'test.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = True

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'test.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = True

def test_case_0():
    var_0 = 'Test generate_files with accept_hooks=False.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'test.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = False

def test_case_0():
    var_0 = 'Test generate_files with keep_project_on_failure=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'test.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = True

def test_case_0():
    var_0 = 'Test generate_files handles binary files correctly.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'binary.bin'
    var_5 = b'\x89PNG\r\n\x1a\n'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = 'cookiecutter.generate.is_binary'
    var_14 = True

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render context.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'static'
    var_5 = 'file.txt'
    var_6 = '{{no_render}}'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = '_copy_without_render'
    var_10 = 'project'
    var_11 = 'static/*'
    var_12 = [var_11]
    var_13 = {var_8: var_10, var_9: var_12}
    var_14 = {var_7: var_13}
    var_15 = 'cookiecutter.generate.find_template'
    var_16 = 'cookiecutter.generate.run_hook_from_repo_dir'

def test_case_0():
    var_0 = 'Test generate_files with default output directory.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_render_and_create_dir_with_valid_dirname. Retrieved 7/12 statements.
# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_with_none_dirname. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_directory_exists_without_overwrite. Retrieved 7/14 statements.
# Partially parsed test_render_and_create_dir_directory_exists_with_overwrite. Retrieved 7/14 statements.
# Partially parsed test_render_and_create_dir_with_nested_path. Retrieved 9/14 statements.
# Partially parsed test_render_and_create_dir_returns_correct_tuple. Retrieved 8/19 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir creates directory with valid dirname.'
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = '{{ project_name }}'
    var_6 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises EmptyDirNameException for empty dirname.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises EmptyDirNameException for None dirname.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = None

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises OutputDirExistsException when dir exists.'
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = '{{ project_name }}'
    var_6 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir succeeds when dir exists and overwrite is True.'
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = '{{ project_name }}'
    var_6 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir creates nested directory structure.'
    var_1 = 'org'
    var_2 = 'project'
    var_3 = 'myorg'
    var_4 = 'myproject'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{ org }}/{{ project }}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir returns tuple with Path and boolean.'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = '{{ name }}'
    var_6 = False
    var_7 = 1



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_new_lines_predicate_true. Retrieved 10/19 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello World'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = var_6[var_2]
    var_9 = False



# Parsed testcases at query #22
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that boolean conversion succeeds without raising InvalidResponse.'
    var_1 = 'debug'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'false'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean conversion succeeds with 'yes' string."
    var_1 = 'enabled'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean conversion succeeds with 'true' string."
    var_1 = 'flag'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'true'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean conversion succeeds with 'no' string."
    var_1 = 'active'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_render_and_create_dir_overwrite_if_exists_true. Retrieved 6/17 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that line 25 predicate evaluates to True when overwrite_if_exists is True.'
    var_1 = 'existing_dir'
    var_2 = var_0 / var_1
    var_3 = True
    var_4 = module_0.Environment()
    var_5 = {}



# Parsed testcases at query #24
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that InvalidResponse exception is caught and converted to ValueError at line 57.'
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'invalid_boolean_value'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate_evaluates_to_true. Retrieved 16/29 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 39 evaluates to True when conditions are met.'
    var_1 = 'test_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'test_file.txt'
    var_7 = 'existing content'
    var_8 = 'builtins.open'
    var_9 = 'test'
    var_10 = 'os.path.isdir'
    var_11 = False
    var_12 = 'shutil.copymode'
    var_13 = 'generate_file.logger'
    var_14 = True
    var_15 = 'The resulting file already exists: %s'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_25_true. Retrieved 5/14 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 (overwrite_if_exists) evaluates to True.'
    var_1 = 'test_dir'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_with_none_dirname. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/10 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 6/13 statements.
# Partially parsed test_render_and_create_dir_existing_dir_without_overwrite. Retrieved 4/13 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 4/13 statements.
# Partially parsed test_render_and_create_dir_nested_directory. Retrieved 3/10 statements.


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
    var_3 = None

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ project_name }}_dir'
    var_5 = 'my_project_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'existing_dir'
    var_3 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'existing_dir'
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'parent/child/nested'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_line_67_evaluates_to_true. Retrieved 8/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.Environment()
    var_6 = var_4[var_0]
    var_7 = False



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate_true. Retrieved 14/35 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'test.txt'
    var_3 = 'test.txt'
    var_4 = 'existing content'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = module_0.Environment()
    var_9 = False
    var_10 = "sys.modules['cookiecutter.generate']"
    var_11 = 'templates'
    var_12 = 'template content'
    var_13 = True
    assert var_13 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_file_name_is_empty_predicate_true. Retrieved 14/32 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 35 evaluates to True when outfile is a directory.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'output_dir'
    var_4 = 'test_file.txt'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = module_0.Environment()
    var_9 = 'test_file.txt'
    var_10 = 'test content'
    var_11 = False
    var_12 = 'output_dir'
    var_13 = False



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 16/34 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 15/31 statements.
# Partially parsed test_generate_file_skips_existing_file. Retrieved 14/31 statements.
# Partially parsed test_generate_file_handles_empty_filename. Retrieved 12/24 statements.
# Partially parsed test_generate_file_uses_configured_newline. Retrieved 14/30 statements.
# Partially parsed test_generate_file_detects_newline. Retrieved 13/29 statements.
# Partially parsed test_generate_file_template_syntax_error. Retrieved 13/27 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'templates'
    var_2 = 'test_{{cookiecutter.name}}.txt'
    var_3 = 'test_{{cookiecutter.name}}.txt'
    var_4 = 'Hello {{cookiecutter.name}}'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'world'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = module_0.Environment()
    var_11 = 'os.getcwd'
    var_12 = 'shutil.copymode'
    var_13 = '__main__.is_binary'
    var_14 = False
    var_15 = 'test_world.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'templates'
    var_2 = 'binary_file.bin'
    var_3 = b'\x89PNG\r\n\x1a\n'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.Environment()
    var_8 = 'shutil.copyfile'
    var_9 = 'shutil.copymode'
    var_10 = '__main__.is_binary'
    var_11 = True
    var_12 = 'shutil'
    var_13 = __import__(var_12)
    var_14 = 'copyfile'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'templates'
    var_2 = 'existing.txt'
    var_3 = 'content'
    var_4 = 'existing.txt'
    var_5 = 'existing content'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = module_0.Environment()
    var_10 = 'shutil.copymode'
    var_11 = '__main__.is_binary'
    var_12 = False
    var_13 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'templates'
    var_2 = '{{cookiecutter.skip}}'
    var_3 = ''
    var_4 = 'cookiecutter'
    var_5 = 'skip'
    var_6 = {var_5: var_3}
    var_7 = {var_4: var_6}
    var_8 = module_0.Environment()
    var_9 = 'shutil.copymode'
    var_10 = '__main__.is_binary'
    var_11 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'templates'
    var_2 = 'test.txt'
    var_3 = 'line1\nline2\n'
    var_4 = 'cookiecutter'
    var_5 = '_new_lines'
    var_6 = '\r\n'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.Environment()
    var_10 = 'shutil.copymode'
    var_11 = '__main__.is_binary'
    var_12 = False
    var_13 = 'test.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'templates'
    var_2 = 'test.txt'
    var_3 = 'line1\r\nline2\r\n'
    var_4 = ''
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = module_0.Environment()
    var_9 = 'shutil.copymode'
    var_10 = '__main__.is_binary'
    var_11 = False
    var_12 = 'test.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'templates'
    var_2 = 'bad.txt'
    var_3 = '{{cookiecutter.name'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.Environment()
    var_10 = 'shutil.copymode'
    var_11 = '__main__.is_binary'
    var_12 = False



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 9/20 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 10/24 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 7/20 statements.
# Partially parsed test_generate_file_renders_path. Retrieved 10/22 statements.
# Partially parsed test_generate_file_returns_on_empty_filename. Retrieved 7/17 statements.
# Partially parsed test_generate_file_handles_template_syntax_error. Retrieved 7/19 statements.
# Partially parsed test_generate_file_uses_configured_newline. Retrieved 9/21 statements.
# Partially parsed test_generate_file_detects_newline. Retrieved 7/19 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'test.txt'
    var_2 = 'Hello {{name}}'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'World'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'test.txt'
    var_2 = 'content'
    var_3 = 'test.txt'
    var_4 = 'existing'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = module_0.Environment()
    var_9 = True
    assert var_9 == 'existing'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'test.bin'
    var_2 = b'\x89PNG\r\n\x1a\n'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = '{{cookiecutter.filename}}.txt'
    var_2 = 'output.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter'
    var_5 = 'filename'
    var_6 = 'output'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'test.txt'
    var_2 = 'content'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'test.txt'
    var_2 = '{{invalid'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'test.txt'
    var_2 = 'content'
    var_3 = 'cookiecutter'
    var_4 = '_new_lines'
    var_5 = '\r\n'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'test.txt'
    var_2 = 'line1\nline2\n'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.Environment()



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_generate_context_predicate_line_38_evaluates_to_false. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 38 (if default_context:) evaluates to False.'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_generate_file_with_binary_file. Retrieved 11/28 statements.
# Partially parsed test_generate_file_with_text_file. Retrieved 14/30 statements.
# Partially parsed test_generate_file_with_templated_filename. Retrieved 15/30 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 14/31 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 12/29 statements.
# Partially parsed test_generate_file_with_custom_newline. Retrieved 14/29 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'template'
    var_2 = 'binary.bin'
    var_3 = b'\x89PNG\r\n\x1a\n'
    var_4 = 'your_module.is_binary'
    var_5 = True
    var_6 = lambda x: var_5
    var_7 = module_0.Environment()
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'template'
    var_2 = 'test.txt'
    var_3 = 'Hello {{ name }}!'
    var_4 = 'utf-8'
    var_5 = 'your_module.is_binary'
    var_6 = False
    var_7 = lambda x: var_6
    var_8 = module_0.Environment()
    var_9 = 'cookiecutter'
    var_10 = 'name'
    var_11 = 'World'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'template'
    var_2 = '{{ filename }}.txt'
    var_3 = 'Content'
    var_4 = 'utf-8'
    var_5 = 'your_module.is_binary'
    var_6 = False
    var_7 = lambda x: var_6
    var_8 = module_0.Environment()
    var_9 = 'cookiecutter'
    var_10 = 'filename'
    var_11 = 'output'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = 'output.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'template'
    var_2 = 'test.txt'
    var_3 = 'Original'
    var_4 = 'utf-8'
    var_5 = 'Existing'
    var_6 = 'your_module.is_binary'
    var_7 = False
    var_8 = lambda x: var_7
    var_9 = module_0.Environment()
    var_10 = 'cookiecutter'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'template'
    var_2 = 'test.txt'
    var_3 = 'Content'
    var_4 = 'utf-8'
    var_5 = 'your_module.is_binary'
    var_6 = False
    var_7 = lambda x: var_6
    var_8 = module_0.Environment()
    var_9 = 'cookiecutter'
    var_10 = {}
    var_11 = {var_9: var_10}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'template'
    var_2 = 'test.txt'
    var_3 = 'Line1\nLine2\n'
    var_4 = 'utf-8'
    var_5 = 'your_module.is_binary'
    var_6 = False
    var_7 = lambda x: var_6
    var_8 = module_0.Environment()
    var_9 = 'cookiecutter'
    var_10 = '_new_lines'
    var_11 = '\r\n'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_at_line_62_evaluates_to_false. Retrieved 4/18 statements.


def test_case_0():
    var_0 = "Test that the predicate at line 62 (for root, dirs, files in os.walk('.')) evaluates to False."
    var_1 = 'empty'
    var_2 = '.'
    var_3 = 0



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_apply_overwrites_to_context_boolean_various_yes_choices. Retrieved 4/8 statements.
# Partially parsed test_apply_overwrites_to_context_boolean_various_no_choices. Retrieved 4/8 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that line 57 evaluates to False when YesNoPrompt.process_response succeeds.'
    var_1 = 'debug'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test boolean conversion with a 'no' value."
    var_1 = 'enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

def test_case_0():
    var_0 = 'Test boolean conversion with various yes choices.'
    var_1 = 'flag'
    var_2 = False
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'Test boolean conversion with various no choices.'
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}



# Parsed testcases at query #37
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing_var'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'existing'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_var'
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

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
    var_7 = [var_2, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'default'
    var_2 = 'option1'
    var_3 = 'option2'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'default'
    var_2 = 'option1'
    var_3 = 'option2'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'invalid'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

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
    var_0 = 'config'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_value1'
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'items'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'x'
    var_9 = 'y'
    var_10 = [var_8, var_9]
    var_11 = {var_1: var_10}
    var_12 = {var_0: var_11}
    var_13 = True
    var_14 = module_0.apply_overwrites_to_context(var_7, var_12, in_dictionary_variable=var_13)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'var3'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'new_value1'
    var_8 = 'true'
    var_9 = {var_0: var_7, var_2: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag1'
    var_1 = 'flag2'
    var_2 = 'flag3'
    var_3 = False
    var_4 = {var_0: var_3, var_1: var_3, var_2: var_3}
    var_5 = '1'
    var_6 = 'true'
    var_7 = 'on'
    var_8 = {var_0: var_5, var_1: var_6, var_2: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.apply_overwrites_to_context(var_2, var_3)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_template_syntax_error_translated_false. Retrieved 8/22 statements.


import jinja2.exceptions as module_0

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'test error'
    var_6 = 1
    var_7 = module_0.TemplateSyntaxError(var_5, var_6)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_generate_context_json_decoding_error. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'Test that ValueError is caught at line 20 and ContextDecodingException is raised.'
    var_1 = '{ invalid json content }'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_render_and_create_dir_raises_empty_dirname_exception. Retrieved 4/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = module_0.Environment()
    var_3 = ''



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_is_binary_predicate_evaluates_to_true. Retrieved 14/29 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'binary_file.bin'
    var_1 = b'\x89PNG\r\n\x1a\n'
    var_2 = 'project'
    var_3 = 'shutil.copyfile'
    var_4 = None
    var_5 = lambda src, dst: var_4
    var_6 = 'shutil.copymode'
    var_7 = lambda src, dst: var_4
    var_8 = 'cookiecutter.generate.is_binary'
    var_9 = module_0.Environment()
    var_10 = 'cookiecutter'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = False



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate_true. Retrieved 11/27 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 39 evaluates to True when conditions are met.'
    var_1 = 'project'
    var_2 = 'test_file.txt'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.Environment()
    var_7 = True
    var_8 = 'existing content'
    assert var_8 == 'existing content'
    var_9 = False
    var_10 = '__main__.is_binary'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_both_defaults_and_extra. Retrieved 9/13 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 8/12 statements.
# Partially parsed test_generate_context_invalid_choice_raises_error. Retrieved 7/11 statements.
# Partially parsed test_generate_context_invalid_multichoice_raises_error. Retrieved 9/13 statements.
# Partially parsed test_generate_context_invalid_boolean_conversion. Retrieved 7/11 statements.
# Partially parsed test_generate_context_uses_custom_filename. Retrieved 3/7 statements.
# Partially parsed test_generate_context_complex_nested_structure. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'Test generate_context with a basic JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'

def test_case_0():
    var_0 = 'Test generate_context with default_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'project_name'
    var_4 = 'default_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with extra_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'version'
    var_4 = '2.0'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with both default and extra contexts.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0", "author": "Jane"}'
    var_3 = 'author'
    var_4 = 'DefaultAuthor'
    var_5 = {var_3: var_4}
    var_6 = 'version'
    var_7 = '3.0'
    var_8 = {var_6: var_7}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON raises ContextDecodingException.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project"'
    var_3 = module_0.generate_context(var_0)

def test_case_0():
    var_0 = 'Test generate_context with choice variable override.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with multichoice variable override.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["auth", "api", "db", "cache"]}'
    var_3 = 'features'
    var_4 = 'api'
    var_5 = 'db'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test generate_context with boolean variable override as string.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true, "use_ci": false}'
    var_3 = 'use_docker'
    var_4 = 'use_ci'
    var_5 = 'no'
    var_6 = 'yes'
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary override.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"host": "localhost", "port": 8000}}'
    var_3 = 'config'
    var_4 = 'port'
    var_5 = 9000
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises ValueError for invalid choice.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'BSD'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises ValueError for invalid multichoice.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["auth", "api"]}'
    var_3 = 'features'
    var_4 = 'auth'
    var_5 = 'unknown'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises ValueError for invalid boolean string.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'invalid_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)

def test_case_0():
    var_0 = 'Test generate_context uses the provided context_file name.'
    var_1 = 'custom_context.json'
    var_2 = '{"name": "test"}'

def test_case_0():
    var_0 = 'Test generate_context with complex nested structures.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"app": {"db": {"engine": "postgres", "version": "12"}}}'
    var_3 = 'app'
    var_4 = 'db'
    var_5 = 'version'
    var_6 = '13'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}



# Parsed testcases at query #44
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'static/image.png'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.png'
    var_4 = 'static/*'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.is_copy_only_path(var_0, var_7)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/index.html'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.png'
    var_4 = 'static/*'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.is_copy_only_path(var_0, var_7)
    assert var_8 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'static/image.png'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.is_copy_only_path(var_0, var_3)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'static/image.png'
    var_1 = {}
    var_2 = module_0.is_copy_only_path(var_0, var_1)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'static/image.png'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = []
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.is_copy_only_path(var_0, var_5)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'build/output.min.js'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'build/*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'README.md'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'README.md'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'docs/guide.pdf'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.png'
    var_4 = '*.jpg'
    var_5 = 'docs/*.pdf'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = module_0.is_copy_only_path(var_0, var_8)
    assert var_9 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 9/26 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_template.txt'
    var_1 = 'Hello {{ name }}'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'test_template.txt'
    var_8 = False



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 14/29 statements.
# Partially parsed test_generate_files_with_binary_file. Retrieved 13/27 statements.
# Partially parsed test_generate_files_empty_context. Retrieved 8/20 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 14/28 statements.
# Partially parsed test_generate_files_overwrite_if_exists. Retrieved 14/28 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 16/30 statements.
# Partially parsed test_generate_files_nested_directories. Retrieved 18/35 statements.
# Partially parsed test_generate_files_with_new_lines_config. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'Test basic generate_files functionality.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Hello {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = 'cookiecutter.generate.accept_hooks'
    var_13 = False

def test_case_0():
    var_0 = 'Test generate_files with binary files.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'binary.bin'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = False

def test_case_0():
    var_0 = 'Test generate_files with empty context.'
    var_1 = 'repo'
    var_2 = 'my_template'
    var_3 = 'test.txt'
    var_4 = 'Hello World'
    var_5 = 'output'
    var_6 = None
    var_7 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Hello {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = True
    var_13 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Hello {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = True
    var_13 = False

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render setting.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Hello {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = '_copy_without_render'
    var_9 = 'my_project'
    var_10 = '*.bin'
    var_11 = [var_10]
    var_12 = {var_7: var_9, var_8: var_11}
    var_13 = (var_6, var_12)
    var_14 = [var_13]
    var_15 = False

def test_case_0():
    var_0 = 'Test generate_files with nested directories.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'src'
    var_4 = '{{cookiecutter.module_name}}'
    var_5 = True
    var_6 = 'test.py'
    var_7 = '# {{cookiecutter.module_name}}'
    var_8 = 'output'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = 'module_name'
    var_12 = 'my_project'
    var_13 = 'my_module'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = (var_9, var_14)
    var_16 = [var_15]
    var_17 = False

def test_case_0():
    var_0 = 'Test generate_files with _new_lines configuration.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Line1\nLine2\n{{cookiecutter.project_name}}'
    var_5 = 'output'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_is_binary_predicate_evaluates_to_true. Retrieved 10/36 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'binary_file.bin'
    var_1 = b'\x89PNG\r\n\x1a\n'
    var_2 = 'project'
    var_3 = 'builtins.__import__'
    var_4 = lambda name, *args, **kwargs: __import__(name, *args, **kwargs)
    var_5 = module_0.Environment()
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'test.bin'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_generate_files_with_default_parameters. Retrieved 20/33 statements.
# Partially parsed test_generate_files_with_empty_context. Retrieved 15/27 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 20/32 statements.
# Partially parsed test_generate_files_with_skip_if_file_exists. Retrieved 20/32 statements.
# Partially parsed test_generate_files_without_hooks. Retrieved 18/29 statements.
# Partially parsed test_generate_files_keep_project_on_failure. Retrieved 20/32 statements.


def test_case_0():
    var_0 = 'Test generate_files with default parameters.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'output'
    var_9 = 'cookiecutter.generate.find_template'
    var_10 = 'cookiecutter.generate.create_env_with_context'
    var_11 = True
    var_12 = lambda ctx: StrictEnvironment(context=ctx, keep_trailing_newline=var_11)
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = None
    var_15 = lambda *args, **kwargs: var_14
    var_16 = 'cookiecutter.generate.os.walk'
    var_17 = []
    var_18 = []
    var_19 = lambda path: [(path, var_17, var_18)]

def test_case_0():
    var_0 = 'Test generate_files with None context.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.generate.find_template'
    var_5 = 'cookiecutter.generate.create_env_with_context'
    var_6 = True
    var_7 = lambda ctx: StrictEnvironment(context=ctx, keep_trailing_newline=var_6)
    var_8 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_9 = None
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'cookiecutter.generate.os.walk'
    var_12 = []
    var_13 = []
    var_14 = lambda path: [(path, var_12, var_13)]

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'output'
    var_9 = 'cookiecutter.generate.find_template'
    var_10 = 'cookiecutter.generate.create_env_with_context'
    var_11 = True
    var_12 = lambda ctx: StrictEnvironment(context=ctx, keep_trailing_newline=var_11)
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = None
    var_15 = lambda *args, **kwargs: var_14
    var_16 = 'cookiecutter.generate.os.walk'
    var_17 = []
    var_18 = []
    var_19 = lambda path: [(path, var_17, var_18)]

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'output'
    var_9 = 'cookiecutter.generate.find_template'
    var_10 = 'cookiecutter.generate.create_env_with_context'
    var_11 = True
    var_12 = lambda ctx: StrictEnvironment(context=ctx, keep_trailing_newline=var_11)
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = None
    var_15 = lambda *args, **kwargs: var_14
    var_16 = 'cookiecutter.generate.os.walk'
    var_17 = []
    var_18 = []
    var_19 = lambda path: [(path, var_17, var_18)]

def test_case_0():
    var_0 = 'Test generate_files with accept_hooks=False.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'output'
    var_9 = 'cookiecutter.generate.find_template'
    var_10 = 'cookiecutter.generate.create_env_with_context'
    var_11 = True
    var_12 = lambda ctx: StrictEnvironment(context=ctx, keep_trailing_newline=var_11)
    var_13 = 'cookiecutter.generate.os.walk'
    var_14 = []
    var_15 = []
    var_16 = lambda path: [(path, var_14, var_15)]
    var_17 = False

def test_case_0():
    var_0 = 'Test generate_files with keep_project_on_failure=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'output'
    var_9 = 'cookiecutter.generate.find_template'
    var_10 = 'cookiecutter.generate.create_env_with_context'
    var_11 = True
    var_12 = lambda ctx: StrictEnvironment(context=ctx, keep_trailing_newline=var_11)
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = None
    var_15 = lambda *args, **kwargs: var_14
    var_16 = 'cookiecutter.generate.os.walk'
    var_17 = []
    var_18 = []
    var_19 = lambda path: [(path, var_17, var_18)]



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_both_default_and_extra_context. Retrieved 8/12 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_nested_dict. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_list_choices. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_boolean_yes. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_multichoice. Retrieved 9/15 statements.
# Partially parsed test_generate_context_preserves_other_variables. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_custom_filename. Retrieved 3/7 statements.
# Partially parsed test_generate_context_empty_json. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_invalid_default_context. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'Test generate_context with a basic cookiecutter.json file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John Doe"}'

def test_case_0():
    var_0 = 'Test generate_context with default_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_3 = 'project_name'
    var_4 = 'default_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with extra_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_3 = 'project_name'
    var_4 = 'extra_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with both default_context and extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_3 = 'project_name'
    var_4 = 'default_project'
    var_5 = {var_3: var_4}
    var_6 = 'extra_project'
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project"'
    var_3 = module_0.generate_context(var_0)

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project": {"name": "my_project", "version": "1.0.0"}}'

def test_case_0():
    var_0 = 'Test generate_context with list/choice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with boolean variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'false'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with boolean variable set to yes.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": false}'
    var_3 = 'use_docker'
    var_4 = 'yes'
    var_5 = {var_3: var_4}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with non-existent file.'
    var_1 = '/nonexistent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

def test_case_0():
    var_0 = 'Test generate_context with multichoice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["auth", "api", "admin"]}'
    var_3 = 'features'
    var_4 = 'api'
    var_5 = 'admin'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test that generate_context preserves variables not in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John", "version": "1.0.0"}'
    var_3 = 'project_name'
    var_4 = 'new_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with custom context filename.'
    var_1 = 'custom.json'
    var_2 = '{"project_name": "my_project"}'

def test_case_0():
    var_0 = 'Test generate_context with empty JSON object.'
    var_1 = 'cookiecutter.json'
    var_2 = '{}'

def test_case_0():
    var_0 = 'Test generate_context with invalid default context value.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'InvalidLicense'
    var_5 = {var_3: var_4}



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_delete_project_on_failure_false_when_keep_project_on_failure_true. Retrieved 3/5 statements.
# Partially parsed test_delete_project_on_failure_false_when_output_directory_not_created. Retrieved 3/5 statements.
# Partially parsed test_delete_project_on_failure_false_both_conditions. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'Test that delete_project_on_failure evaluates to False when keep_project_on_failure is True.'
    var_1 = True
    var_2 = True

def test_case_0():
    var_0 = 'Test that delete_project_on_failure evaluates to False when output_directory_created is False.'
    var_1 = False
    var_2 = False

def test_case_0():
    var_0 = 'Test that delete_project_on_failure evaluates to False when both conditions result in False.'
    var_1 = False
    var_2 = True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_context_cookiecutter_new_lines_predicate_evaluates_to_true. Retrieved 12/21 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello {{ cookiecutter.name }}'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = 'World'
    var_6 = '\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.Environment()
    var_10 = var_8[var_2]
    var_11 = False



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_generate_context_file_open_predicate_line_18. Retrieved 2/10 statements.


def test_case_0():
    var_0 = "Test that the predicate at line 18 (with open(...)) evaluates to False when file doesn't exist."
    var_1 = 'non_existent.json'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 11/20 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_template.txt'
    var_1 = 'Hello {{ name }}'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = var_6[var_2]
    var_9 = '_new_lines'
    var_10 = False



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_predicate_line_67_evaluates_to_false. Retrieved 7/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = var_2[var_0]
    var_5 = '_new_lines'
    var_6 = False



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_generate_context_opens_file_with_utf8_encoding. Retrieved 9/17 statements.


import json as module_0

def test_case_0():
    var_0 = 'Test that generate_context opens the context file with utf-8 encoding.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'Test Author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'utf-8'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_generate_context_loads_json_file. Retrieved 2/6 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_both_default_and_extra. Retrieved 8/12 statements.
# Partially parsed test_generate_context_invalid_json_raises_exception. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 7/11 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 7/11 statements.
# Partially parsed test_generate_context_preserves_ordered_dict. Retrieved 3/10 statements.
# Partially parsed test_generate_context_invalid_choice_raises_error. Retrieved 6/10 statements.
# Partially parsed test_generate_context_invalid_multichoice_raises_error. Retrieved 8/12 statements.
# Partially parsed test_generate_context_invalid_boolean_conversion_raises_error. Retrieved 6/10 statements.
# Partially parsed test_generate_context_default_context_invalid_shows_warning. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": "my_project"}'

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": "my_project", "port": 8000}'
    var_2 = 'port'
    var_3 = 9000
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": "my_project"}'
    var_2 = 'project_name'
    var_3 = 'another_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": "my_project", "port": 8000}'
    var_2 = 'port'
    var_3 = 9000
    var_4 = {var_2: var_3}
    var_5 = 'project_name'
    var_6 = 'extra_project'
    var_7 = {var_5: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"invalid json}'
    var_2 = module_0.generate_context(var_0)

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_2 = 'license'
    var_3 = 'Apache'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_2 = 'features'
    var_3 = 'feature2'
    var_4 = 'feature3'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"use_docker": true}'
    var_2 = 'use_docker'
    var_3 = 'false'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"database": {"host": "localhost", "port": 5432}}'
    var_2 = 'database'
    var_3 = 'port'
    var_4 = 3306
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"z_field": "z", "a_field": "a"}'
    var_2 = 'cookiecutter'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"license": ["MIT", "Apache"]}'
    var_2 = 'license'
    var_3 = 'GPL'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, extra_context=var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"features": ["feature1", "feature2"]}'
    var_2 = 'features'
    var_3 = 'feature1'
    var_4 = 'feature3'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = module_0.generate_context(var_0, extra_context=var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"use_docker": true}'
    var_2 = 'use_docker'
    var_3 = 'invalid_bool'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, extra_context=var_4)

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"license": ["MIT", "Apache"]}'
    var_2 = 'license'
    assert var_2 == 1
    var_3 = 'GPL'
    var_4 = {var_2: var_3}
    var_5 = 'always'
    var_6 = 0



# Parsed testcases at query #57
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 54 (if accept_hooks:) evaluates to True.'
    var_1 = True
    assert var_1 is True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_generate_context_applies_overwrites_when_default_context_provided. Retrieved 9/19 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 38 evaluates to True and applies overwrites.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'initial_name'
    var_5 = 'initial_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'overridden_name'
    var_8 = {var_2: var_7}



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_generate_context_opens_file_with_utf8_encoding. Retrieved 9/16 statements.


import json as module_0

def test_case_0():
    var_0 = 'Test that generate_context opens the context file with utf-8 encoding.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'Test Author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'utf-8'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_generate_context_file_not_found. Retrieved 3/9 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 18 evaluates to False when file does not exist.'
    var_1 = '/tmp/non_existent_cookiecutter_file_12345.json'
    var_2 = module_0.generate_context(var_1)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_generate_files_with_minimal_context. Retrieved 14/33 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists_false. Retrieved 11/26 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists_true. Retrieved 16/36 statements.
# Partially parsed test_generate_files_with_copy_without_render. Retrieved 17/34 statements.
# Partially parsed test_generate_files_renders_text_files. Retrieved 15/31 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 18/37 statements.


def test_case_0():
    var_0 = 'Test generate_files with minimal context creates project directory.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Hello {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = 'COOKIECUTTER_REPO_DIR'
    var_13 = False

def test_case_0():
    var_0 = 'Test generate_files raises exception when output dir exists without overwrite.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'my_project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = {var_6: var_4}
    var_8 = (var_5, var_7)
    var_9 = [var_8]
    var_10 = False

def test_case_0():
    var_0 = 'Test generate_files overwrites existing output directory when flag is True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'New content'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'old.txt'
    var_8 = 'Old content'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = {var_10: var_6}
    var_12 = (var_9, var_11)
    var_13 = [var_12]
    var_14 = True
    var_15 = False

def test_case_0():
    var_0 = 'Test generate_files respects _copy_without_render setting.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'binary_files'
    var_4 = 'data.bin'
    var_5 = b'\x00\x01\x02{{cookiecutter.project_name}}'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = '_copy_without_render'
    var_10 = 'my_project'
    var_11 = 'binary_files/*'
    var_12 = [var_11]
    var_13 = {var_8: var_10, var_9: var_12}
    var_14 = (var_7, var_13)
    var_15 = [var_14]
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files renders text file contents.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'author'
    var_9 = 'my_project'
    var_10 = 'John Doe'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = (var_6, var_11)
    var_13 = [var_12]
    var_14 = False

def test_case_0():
    var_0 = 'Test generate_files skips existing files when skip_if_file_exists is True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'config.txt'
    var_4 = 'New config'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = False
    var_13 = 'Old config'
    var_14 = {var_7: var_8}
    var_15 = (var_6, var_14)
    var_16 = [var_15]
    var_17 = True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_generate_files_with_minimal_context. Retrieved 14/37 statements.
# Partially parsed test_generate_files_returns_project_dir_path. Retrieved 13/29 statements.
# Partially parsed test_generate_files_with_empty_context. Retrieved 8/21 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 16/39 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 14/31 statements.
# Partially parsed test_generate_files_renders_template_variables. Retrieved 15/32 statements.
# Partially parsed test_generate_files_creates_nested_directories. Retrieved 18/40 statements.


def test_case_0():
    var_0 = 'Test generate_files with minimal context.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = 'cookiecutter.generate.accept_hooks'
    var_13 = False

def test_case_0():
    var_0 = 'Test that generate_files returns the project directory path.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = False

def test_case_0():
    var_0 = 'Test generate_files with empty context defaults to OrderedDict.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'test'
    var_5 = 'output'
    var_6 = None
    var_7 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'new content'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'old_file.txt'
    var_8 = 'old content'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = {var_10: var_6}
    var_12 = (var_9, var_11)
    var_13 = [var_12]
    var_14 = True
    var_15 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'template content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = True
    var_13 = False

def test_case_0():
    var_0 = 'Test that generate_files properly renders template variables.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_slug}}'
    var_3 = 'config.txt'
    var_4 = 'Project: {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'project_slug'
    var_9 = 'My Project'
    var_10 = 'my_project'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = (var_6, var_11)
    var_13 = [var_12]
    var_14 = False

def test_case_0():
    var_0 = 'Test that generate_files creates nested directory structures.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'src'
    var_4 = '{{cookiecutter.package_name}}'
    var_5 = True
    var_6 = '__init__.py'
    var_7 = ''
    var_8 = 'output'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = 'package_name'
    var_12 = 'my_project'
    var_13 = 'my_package'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = (var_9, var_14)
    var_16 = [var_15]
    var_17 = False



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_true. Retrieved 11/20 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello {{ name }}'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = 'World'
    var_6 = '\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = var_8[var_2]
    var_10 = False



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_generate_files_with_minimal_context. Retrieved 16/29 statements.
# Partially parsed test_generate_files_returns_project_dir_path. Retrieved 16/30 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 19/34 statements.
# Partially parsed test_generate_files_with_skip_if_file_exists. Retrieved 17/30 statements.
# Partially parsed test_generate_files_without_hooks. Retrieved 16/31 statements.
# Partially parsed test_generate_files_default_context. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'Test generate_files with minimal context and simple template.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = False

def test_case_0():
    var_0 = 'Test that generate_files returns the project directory path.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = '{{cookiecutter.content}}'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'content'
    var_10 = 'Hello'
    var_11 = {var_8: var_6, var_9: var_10}
    var_12 = (var_7, var_11)
    var_13 = [var_12]
    var_14 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_15 = None
    var_16 = lambda *args, **kwargs: var_15
    var_17 = True
    var_18 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'existing.txt'
    var_4 = 'template content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = True
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files with accept_hooks=False.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project}}'
    var_3 = 'test.txt'
    var_4 = 'data'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = False
    var_15 = len(var_12)
    assert var_15 == 0

def test_case_0():
    var_0 = 'Test generate_files with None context defaults to OrderedDict.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = None
    var_8 = lambda *args, **kwargs: var_7
    var_9 = False



####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecated_warning. Retrieved 13/35 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_actual_function. Retrieved 14/26 statements.
# Partially parsed test_run_hook_from_repo_dir_with_false_delete_flag. Retrieved 11/23 statements.
# Partially parsed test_run_hook_from_repo_dir_with_true_delete_flag. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir issues a deprecation warning.'
    var_1 = 'repo'
    var_2 = 'project'
    assert var_2 == 1
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = []
    var_9 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_10 = 'always'
    var_11 = 0
    var_12 = var_3.category

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir delegates to run_hook_from_repo_dir.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_prompt'
    var_9 = True
    var_10 = []
    var_11 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_12 = 'always'
    var_13 = len(var_10)
    assert var_13 == 1

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with delete_project_on_failure=False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = []
    var_8 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_9 = 'always'
    var_10 = False

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with delete_project_on_failure=True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = []
    var_8 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_9 = 'always'
    var_10 = True



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'Jane'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'key2'
    var_6 = 'value2'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flavor'
    var_1 = 'vanilla'
    var_2 = 'chocolate'
    var_3 = 'strawberry'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flavor'
    var_1 = 'vanilla'
    var_2 = 'chocolate'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'invalid'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'toppings'
    var_1 = 'pepperoni'
    var_2 = 'mushroom'
    var_3 = 'onion'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'toppings'
    var_1 = 'pepperoni'
    var_2 = 'mushroom'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'invalid'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'debug'
    var_2 = 'timeout'
    var_3 = True
    var_4 = 30
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = False
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'true'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'false'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'options'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = {var_1: var_3}
    var_9 = {var_0: var_8}
    var_10 = True
    var_11 = module_0.apply_overwrites_to_context(var_7, var_9, in_dictionary_variable=var_10)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.apply_overwrites_to_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.apply_overwrites_to_context(var_2, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = '  yes  '
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'count'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = '10'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 8/12 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_invalid_choice_value. Retrieved 7/11 statements.
# Partially parsed test_generate_context_invalid_multichoice_value. Retrieved 9/13 statements.
# Partially parsed test_generate_context_boolean_invalid_string. Retrieved 7/11 statements.
# Partially parsed test_generate_context_default_context_warning. Retrieved 11/19 statements.
# Partially parsed test_generate_context_with_string_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_custom_filename. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'Test generate_context loads a basic JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'

def test_case_0():
    var_0 = 'Test generate_context applies default_context overwrites.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'project_name'
    var_4 = 'default_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context applies extra_context overwrites.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'version'
    var_4 = '2.0'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with choice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with multichoice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_3 = 'features'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test generate_context with boolean variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_ci": true}'
    var_3 = 'use_ci'
    var_4 = 'false'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"db": "postgres", "port": 5432}}'
    var_3 = 'config'
    var_4 = 'db'
    var_5 = 'mysql'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises ContextDecodingException for invalid JSON.'
    var_1 = 'cookiecutter.json'
    var_2 = '{invalid json}'
    var_3 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context handles missing file.'
    var_1 = '/nonexistent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid choice value.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'GPL'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid multichoice value.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["feature1", "feature2"]}'
    var_3 = 'features'
    var_4 = 'feature1'
    var_5 = 'feature3'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid boolean string.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_ci": true}'
    var_3 = 'use_ci'
    var_4 = 'invalid'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context issues warning for invalid default_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'GPL'
    var_5 = {var_3: var_4}
    var_6 = 'always'
    var_7 = module_0.generate_context(var_2, var_5)
    var_8 = 0
    var_9 = var_4.message
    var_10 = str(var_9)

def test_case_0():
    var_0 = 'Test generate_context with simple string variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"author": "John Doe", "email": "john@example.com"}'
    var_3 = 'author'
    var_4 = 'Jane Doe'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with custom context filename.'
    var_1 = 'template.json'
    var_2 = '{"name": "test"}'

def test_case_0():
    var_0 = 'Test generate_context converts yes strings to boolean True.'



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 24 evaluates to False when overwrite is not a subset of context_value.'
    var_1 = 'choices'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = 'd'
    var_8 = [var_2, var_7]
    var_9 = {var_1: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_with_none_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/9 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 6/12 statements.
# Partially parsed test_render_and_create_dir_existing_dir_no_overwrite. Retrieved 5/13 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_nested_directory_creation. Retrieved 3/9 statements.
# Partially parsed test_render_and_create_dir_returns_tuple. Retrieved 5/15 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = None

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'new_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{project_name}}_dir'
    var_5 = 'my_project_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'existing_dir'
    var_4 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'parent/child/grandchild'

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'test_dir'
    var_3 = 0
    var_4 = 1



# Parsed testcases at query #6
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that boolean conversion succeeds and predicate at line 57 evaluates to False.'
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean conversion succeeds with 'no' and predicate at line 57 evaluates to False."
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean conversion succeeds with 'true' and predicate at line 57 evaluates to False."
    var_1 = 'flag'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'true'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean conversion succeeds with '0' and predicate at line 57 evaluates to False."
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = '0'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean conversion succeeds with '1' and predicate at line 57 evaluates to False."
    var_1 = 'flag'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = '1'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = ''



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_with_list_choices. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_dict_choices. Retrieved 5/11 statements.
# Partially parsed test_generate_context_with_boolean_string. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_boolean_string_false. Retrieved 6/10 statements.
# Partially parsed test_generate_context_invalid_boolean_conversion. Retrieved 7/11 statements.
# Partially parsed test_generate_context_invalid_choice. Retrieved 7/11 statements.
# Partially parsed test_generate_context_custom_file_stem. Retrieved 3/7 statements.
# Partially parsed test_generate_context_multichoice_valid. Retrieved 9/15 statements.
# Partially parsed test_generate_context_multichoice_invalid. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'Test generate_context with a basic JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John Doe"}'

def test_case_0():
    var_0 = 'Test generate_context with default_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_3 = 'project_name'
    var_4 = 'default_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with extra_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0.0"}'
    var_3 = 'project_name'
    var_4 = 'extra_project'
    var_5 = {var_3: var_4}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON raises ContextDecodingException.'
    var_1 = 'cookiecutter.json'
    var_2 = '{invalid json}'
    var_3 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with non-existent file.'
    var_1 = '/nonexistent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

def test_case_0():
    var_0 = 'Test generate_context with list choices in context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"language": ["python", "javascript", "go"]}'
    var_3 = 'language'
    var_4 = 'javascript'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary in context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"debug": true, "timeout": 30}}'
    var_3 = 'config'
    var_4 = 'debug'

def test_case_0():
    var_0 = 'Test generate_context converting string to boolean.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'yes'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test generate_context converting string 'no' to boolean False."
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'no'
    var_5 = {var_3: var_4}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid boolean string raises ValueError.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'maybe'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid choice raises ValueError.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"language": ["python", "javascript"]}'
    var_3 = 'language'
    var_4 = 'ruby'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)

def test_case_0():
    var_0 = 'Test generate_context with custom file stem.'
    var_1 = 'template.json'
    var_2 = '{"project_name": "test"}'

def test_case_0():
    var_0 = 'Test generate_context with valid multichoice.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["auth", "api", "admin", "dashboard"]}'
    var_3 = 'features'
    var_4 = 'api'
    var_5 = 'admin'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid multichoice raises ValueError.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["auth", "api", "admin"]}'
    var_3 = 'features'
    var_4 = 'api'
    var_5 = 'invalid'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_with_none_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/7 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 5/9 statements.
# Partially parsed test_render_and_create_dir_existing_dir_raises_exception. Retrieved 3/10 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 4/10 statements.
# Partially parsed test_render_and_create_dir_creates_nested_directories. Retrieved 3/7 statements.
# Partially parsed test_render_and_create_dir_with_complex_template. Retrieved 7/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = None

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ project_name }}_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'existing_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'parent/child/grandchild'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'version'
    var_3 = 'test'
    var_4 = '1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = '{{ name }}-{{ version }}'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecated. Retrieved 13/30 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_actual_hook. Retrieved 11/16 statements.
# Partially parsed test_run_hook_from_repo_dir_with_delete_true. Retrieved 8/13 statements.
# Partially parsed test_run_hook_from_repo_dir_with_delete_false. Retrieved 10/15 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir issues a deprecation warning.'
    var_1 = '/path/to/repo'
    var_2 = 'post_gen_project'
    var_3 = '/path/to/project'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    assert var_7 == 1
    var_8 = {var_4: var_7}
    var_9 = True
    var_10 = 'always'
    var_11 = module_0._run_hook_from_repo_dir(var_1, var_2, var_3, var_8, var_9)
    var_12 = 0

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir delegates to run_hook_from_repo_dir.'
    var_1 = '/template'
    var_2 = 'pre_gen_project'
    var_3 = '/output'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'myproject'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = False
    var_10 = module_0._run_hook_from_repo_dir(var_1, var_2, var_3, var_8, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with delete_project_on_failure=True.'
    var_1 = '/repo'
    var_2 = 'post_gen_project'
    var_3 = '/project'
    var_4 = {}
    var_5 = True
    var_6 = module_0._run_hook_from_repo_dir(var_1, var_2, var_3, var_4, var_5)
    var_7 = True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with delete_project_on_failure=False.'
    var_1 = '/repo'
    var_2 = 'pre_gen_project'
    var_3 = '/project'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = False
    var_8 = module_0._run_hook_from_repo_dir(var_1, var_2, var_3, var_6, var_7)
    var_9 = False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 19/37 statements.
# Partially parsed test_generate_file_skips_existing_file. Retrieved 14/27 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 16/31 statements.
# Partially parsed test_generate_file_returns_when_filename_empty. Retrieved 11/21 statements.
# Partially parsed test_generate_file_uses_configured_newline. Retrieved 17/31 statements.
# Partially parsed test_generate_file_renders_output_filename. Retrieved 17/32 statements.
# Partially parsed test_generate_file_detects_newline_from_file. Retrieved 15/30 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'template.txt'
    var_3 = 'Hello {{ name }}!'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'World'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.Environment()
    var_10 = 'os.path.isdir'
    var_11 = False
    var_12 = 'os.path.exists'
    var_13 = 'generate_file.is_binary'
    var_14 = 'shutil.copymode'
    var_15 = 'os.getcwd'
    var_16 = 'os.chdir'
    var_17 = 'os.path.join'
    var_18 = -1

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'template.txt'
    var_2 = 'content'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.Environment()
    var_7 = 'os.path.isdir'
    var_8 = False
    var_9 = 'os.path.exists'
    var_10 = True
    var_11 = 'os.path.join'
    var_12 = -1
    var_13 = 'builtins.open'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'image.bin'
    var_2 = b'\x89PNG\r\n\x1a\n'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.Environment()
    var_7 = 'os.path.isdir'
    var_8 = False
    var_9 = 'os.path.exists'
    var_10 = 'generate_file.is_binary'
    var_11 = True
    var_12 = 'shutil.copyfile'
    var_13 = 'shutil.copymode'
    var_14 = 'os.path.join'
    var_15 = -1

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = 'os.path.isdir'
    var_6 = True
    var_7 = 'os.path.join'
    var_8 = -1
    var_9 = 'builtins.open'
    var_10 = 'template.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'template.txt'
    var_2 = 'Hello {{ name }}'
    var_3 = 'cookiecutter'
    var_4 = '_new_lines'
    var_5 = '\r\n'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.Environment()
    var_9 = 'os.path.isdir'
    var_10 = False
    var_11 = 'os.path.exists'
    var_12 = 'generate_file.is_binary'
    var_13 = 'shutil.copymode'
    var_14 = 'os.path.join'
    var_15 = -1
    var_16 = 'builtins.open'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = '{{ cookiecutter.filename }}.txt'
    var_2 = 'content'
    var_3 = 'cookiecutter'
    var_4 = 'filename'
    var_5 = 'output'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.Environment()
    var_9 = 'os.path.isdir'
    var_10 = False
    var_11 = 'os.path.exists'
    var_12 = 'generate_file.is_binary'
    var_13 = 'shutil.copymode'
    var_14 = 'os.path.join'
    var_15 = -1
    var_16 = 'builtins.open'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'template.txt'
    var_2 = 'line1\nline2\n'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.Environment()
    var_7 = 'os.path.isdir'
    var_8 = False
    var_9 = 'os.path.exists'
    var_10 = 'generate_file.is_binary'
    var_11 = 'shutil.copymode'
    var_12 = 'os.path.join'
    var_13 = -1
    var_14 = 'builtins.open'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generate_context_raises_context_decoding_exception_on_invalid_json. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'Test that ValueError during JSON loading is caught and raises ContextDecodingException.'
    var_1 = '{invalid json content}'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_file_name_is_empty_predicate_true. Retrieved 11/24 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'output_dir'
    var_3 = 'output_dir'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.Environment()
    var_8 = '__main__.logger'
    var_9 = '__main__.is_binary'
    var_10 = False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_true. Retrieved 12/21 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello {{ cookiecutter.name }}'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = 'World'
    var_6 = '\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.Environment()
    var_10 = var_8[var_2]
    var_11 = False



# Parsed testcases at query #15
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that boolean conversion succeeds and predicate at line 57 evaluates to False.'
    var_1 = 'my_bool'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean conversion with 'no' succeeds and predicate at line 57 evaluates to False."
    var_1 = 'my_bool'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean conversion with 'false' succeeds and predicate at line 57 evaluates to False."
    var_1 = 'enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'false'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_is_binary_predicate_evaluates_to_true. Retrieved 14/42 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'binary_file.bin'
    var_1 = b'\x89PNG\r\n\x1a\n'
    var_2 = 'project'
    var_3 = 'generate_file'
    var_4 = module_0.Environment()
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'binary_file.bin'
    var_9 = module_0.Environment()
    var_10 = 'cookiecutter'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = 'binary_file.bin'



# Parsed testcases at query #17
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing_var'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'existing'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_var'
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
    var_7 = [var_2, var_6]
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
    var_0 = 'use_feature'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'use_feature'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'use_feature'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'true'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'use_feature'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'false'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'use_feature'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 'Jane'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'items'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'x'
    var_9 = 'y'
    var_10 = 'z'
    var_11 = [var_8, var_9, var_10]
    var_12 = {var_1: var_11}
    var_13 = {var_0: var_12}
    var_14 = True
    var_15 = module_0.apply_overwrites_to_context(var_7, var_13, in_dictionary_variable=var_14)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'var3'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = 'value3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'new_value1'
    var_8 = 'new_value3'
    var_9 = {var_0: var_7, var_2: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_template_syntax_error_translated_attribute_set_to_false. Retrieved 9/30 statements.


import jinja2.environment as module_0
import jinja2.exceptions as module_1

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'test error'
    var_7 = 1
    var_8 = module_1.TemplateSyntaxError(var_6, var_7)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generate_context_json_decoding_error. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'Test that ValueError is caught at line 20 and ContextDecodingException is raised.'
    var_1 = '{invalid json content'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_files_with_context. Retrieved 15/33 statements.
# Partially parsed test_generate_files_empty_context. Retrieved 7/21 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 15/33 statements.
# Partially parsed test_generate_files_overwrite_if_exists. Retrieved 15/31 statements.
# Partially parsed test_generate_files_with_subdirectories. Retrieved 15/34 statements.


def test_case_0():
    var_0 = 'repo'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'test.txt'
    var_3 = 'Hello {{cookiecutter.name}}'
    var_4 = 'output'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'name'
    var_8 = 'my_project'
    var_9 = 'World'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = (var_5, var_10)
    var_12 = [var_11]
    var_13 = 'cookiecutter.generate.accept_hooks'
    var_14 = False

def test_case_0():
    var_0 = 'repo'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'test.txt'
    var_3 = 'content'
    var_4 = 'output'
    var_5 = None
    var_6 = False

def test_case_0():
    var_0 = 'repo'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'test.txt'
    var_3 = 'Hello {{cookiecutter.name}}'
    var_4 = 'output'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'name'
    var_8 = 'my_project'
    var_9 = 'World'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = (var_5, var_10)
    var_12 = [var_11]
    var_13 = True
    var_14 = False

def test_case_0():
    var_0 = 'repo'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'test.txt'
    var_3 = 'Hello {{cookiecutter.name}}'
    var_4 = 'output'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'name'
    var_8 = 'my_project'
    var_9 = 'World'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = (var_5, var_10)
    var_12 = [var_11]
    var_13 = True
    var_14 = False

def test_case_0():
    var_0 = 'repo'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = '{{cookiecutter.subdir}}'
    var_3 = 'test.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'subdir'
    var_9 = 'my_project'
    var_10 = 'src'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = (var_6, var_11)
    var_13 = [var_12]
    var_14 = False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_is_binary_predicate_evaluates_to_true. Retrieved 14/31 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 47 evaluates to True for binary files.'
    var_1 = 'test_binary.bin'
    var_2 = b'\x89PNG\r\n\x1a\n'
    var_3 = 'output'
    var_4 = 'shutil.copyfile'
    var_5 = 'shutil.copymode'
    var_6 = 'is_binary'
    var_7 = True
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = module_0.Environment()
    var_12 = 'test_binary.bin'
    var_13 = module_1.generate_file(var_0, var_12, var_10, var_11)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_render_and_create_dir_raises_on_empty_dirname. Retrieved 4/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that EmptyDirNameException is raised when dirname is empty.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = ''



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_with_none_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/9 statements.
# Partially parsed test_render_and_create_dir_with_template_dirname. Retrieved 6/12 statements.
# Partially parsed test_render_and_create_dir_existing_dir_raises_exception. Retrieved 5/13 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_with_nested_dirname. Retrieved 3/9 statements.
# Partially parsed test_render_and_create_dir_with_complex_template. Retrieved 10/16 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = None

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ project_name }}_dir'
    var_5 = 'my_project_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = True
    var_4 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'parent/child/grandchild'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'org'
    var_2 = 'project'
    var_3 = 'version'
    var_4 = 'acme'
    var_5 = 'widget'
    var_6 = '2.0'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = '{{ org }}/{{ project }}-v{{ version }}'
    var_9 = 'acme/widget-v2.0'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 15/36 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 14/33 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 11/25 statements.
# Partially parsed test_generate_file_returns_if_filename_empty. Retrieved 9/22 statements.
# Partially parsed test_generate_file_uses_configured_newline. Retrieved 16/38 statements.
# Partially parsed test_generate_file_detects_newline. Retrieved 14/38 statements.


def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'test.txt'
    var_3 = 'Hello {{ cookiecutter.name }}'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'World'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = lambda **kwargs: var_2
    var_10 = 'os.path.isdir'
    var_11 = False
    var_12 = 'os.path.exists'
    var_13 = 'shutil.copymode'
    var_14 = '__main__.is_binary'

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'image.png'
    var_3 = b'fake image data'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = lambda **kwargs: var_2
    var_8 = 'os.path.isdir'
    var_9 = False
    var_10 = 'os.path.exists'
    var_11 = 'shutil.copyfile'
    var_12 = 'shutil.copymode'
    var_13 = '__main__.is_binary'

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'test.txt'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = lambda **kwargs: var_2
    var_7 = 'os.path.isdir'
    var_8 = False
    var_9 = 'os.path.exists'
    var_10 = 'get_template'

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'test.txt'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = lambda **kwargs: var_2
    var_7 = 'os.path.isdir'
    var_8 = 'get_template'

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'test.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter'
    var_5 = '_new_lines'
    var_6 = '\r\n'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = lambda **kwargs: var_2
    var_10 = 'os.path.isdir'
    var_11 = False
    var_12 = 'os.path.exists'
    var_13 = 'shutil.copymode'
    var_14 = '__main__.is_binary'
    var_15 = 'builtins.open'

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'test.txt'
    var_3 = 'line1\nline2\n'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = lambda **kwargs: var_2
    var_8 = 'os.path.isdir'
    var_9 = False
    var_10 = 'os.path.exists'
    var_11 = 'shutil.copymode'
    var_12 = '__main__.is_binary'
    var_13 = 'builtins.open'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generate_file_with_binary_file. Retrieved 13/29 statements.
# Partially parsed test_generate_file_with_text_file. Retrieved 17/37 statements.
# Partially parsed test_generate_file_skip_if_file_exists. Retrieved 10/22 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 7/17 statements.
# Partially parsed test_generate_file_with_custom_newline. Retrieved 15/35 statements.


def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'binary_file.bin'
    var_3 = b'\x89PNG\r\n\x1a\n'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'os.path.isdir'
    var_8 = False
    var_9 = 'os.path.exists'
    var_10 = 'shutil.copyfile'
    var_11 = 'shutil.copymode'
    var_12 = 'generate_file.is_binary'

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'template.txt'
    var_3 = 'Hello {{ cookiecutter.name }}\n'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = '_new_lines'
    var_7 = 'World'
    var_8 = False
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'os.path.isdir'
    var_12 = 'os.path.exists'
    var_13 = 'shutil.copymode'
    var_14 = 'generate_file.is_binary'
    var_15 = 'builtins.open'
    var_16 = 'Hello World\n'

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'existing_file.txt'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'os.path.isdir'
    var_7 = False
    var_8 = 'os.path.exists'
    var_9 = 'shutil.copymode'

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'template'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'os.path.isdir'

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'template.txt'
    var_3 = 'Line 1\nLine 2\n'
    var_4 = 'cookiecutter'
    var_5 = '_new_lines'
    var_6 = '\r\n'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'os.path.isdir'
    var_10 = False
    var_11 = 'os.path.exists'
    var_12 = 'shutil.copymode'
    var_13 = 'generate_file.is_binary'
    var_14 = 'builtins.open'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_render_and_create_dir_raises_empty_dir_name_exception. Retrieved 3/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = ''



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate_evaluates_to_true. Retrieved 12/20 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 39 evaluates to True when conditions are met.'
    var_1 = 'test_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'test_file.txt'
    var_7 = 'existing content'
    var_8 = 'os.path.isdir'
    var_9 = False
    var_10 = 'is_binary'
    var_11 = True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate. Retrieved 8/16 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = True
    var_3 = True
    var_4 = False
    var_5 = False
    var_6 = False
    var_7 = True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate_true. Retrieved 13/23 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = 'test_file.txt'
    var_6 = 'existing content'
    var_7 = 'os.path.isdir'
    var_8 = False
    var_9 = 'os.path.exists'
    var_10 = True
    var_11 = '__main__.is_binary'
    var_12 = '__main__.logger'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_template_syntax_error_has_translated_set_to_false. Retrieved 9/25 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = '{% if true %}missing endif'
    var_2 = 'output'
    var_3 = module_0.Environment()
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = 'test.txt'



# Parsed testcases at query #31
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'README.md'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'README.md'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'template.html'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.md'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'docs/README.md'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.md'
    var_4 = 'docs/*'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.is_copy_only_path(var_0, var_7)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'image.png'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.png'
    var_4 = '*.jpg'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.is_copy_only_path(var_0, var_7)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'README.md'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.is_copy_only_path(var_0, var_3)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'README.md'
    var_1 = {}
    var_2 = module_0.is_copy_only_path(var_0, var_1)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'README.md'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = []
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.is_copy_only_path(var_0, var_5)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'static/style.css'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.md'
    var_4 = 'static/*'
    var_5 = '*.txt'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = module_0.is_copy_only_path(var_0, var_8)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'file?.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'file2.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False



# Parsed testcases at query #32
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.pyc'
    var_3 = '*.pyo'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test.pyc'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.pyc'
    var_3 = '*.pyo'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test.py'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '__pycache__/*'
    var_3 = 'node_modules/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = '__pycache__/module.pyc'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'test.pyc'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test.pyc'
    var_2 = module_0.is_copy_only_path(var_1, var_0)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test.pyc'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.bin'
    var_3 = '*.exe'
    var_4 = '*.dll'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'program.exe'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.bin'
    var_3 = '*.exe'
    var_4 = '*.dll'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'library.dll'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'README.md'
    var_3 = 'LICENSE'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.is_copy_only_path(var_2, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'docs/*.pdf'
    var_3 = 'images/*.png'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'docs/guide.pdf'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 14/32 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 12/29 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 14/33 statements.
# Partially parsed test_generate_file_renders_filename_with_context. Retrieved 13/29 statements.
# Partially parsed test_generate_file_returns_when_file_name_is_empty. Retrieved 12/27 statements.
# Partially parsed test_generate_file_uses_custom_newline_from_context. Retrieved 14/30 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'template.txt'
    var_3 = 'Hello {{name}}'
    var_4 = 'utf-8'
    var_5 = module_0.Environment()
    var_6 = 'name'
    var_7 = 'cookiecutter'
    var_8 = 'World'
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'template.txt'
    var_12 = 'r'
    var_13 = 'utf-8'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'binary.bin'
    var_3 = b'\x89PNG\r\n\x1a\n'
    var_4 = '__main__.is_binary'
    var_5 = 'shutil.copyfile'
    var_6 = 'shutil.copymode'
    var_7 = module_0.Environment()
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'binary.bin'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'template.txt'
    var_3 = 'content'
    var_4 = 'utf-8'
    var_5 = 'w'
    var_6 = 'existing'
    var_7 = module_0.Environment()
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = 'template.txt'
    var_12 = True
    var_13 = 'r'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = '{{filename}}.txt'
    var_3 = 'content'
    var_4 = 'utf-8'
    var_5 = module_0.Environment()
    var_6 = 'filename'
    var_7 = 'cookiecutter'
    var_8 = 'output'
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = '{{filename}}.txt'
    var_12 = 'output.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'template.txt'
    var_3 = 'content'
    var_4 = 'utf-8'
    var_5 = module_0.Environment()
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'os.path.isdir'
    var_10 = True
    var_11 = 'template.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'template.txt'
    var_3 = 'Hello {{name}}'
    var_4 = 'utf-8'
    var_5 = module_0.Environment()
    var_6 = 'name'
    var_7 = 'cookiecutter'
    var_8 = 'World'
    var_9 = '_new_lines'
    var_10 = '\r\n'
    var_11 = {var_9: var_10}
    var_12 = {var_6: var_8, var_7: var_11}
    var_13 = 'template.txt'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_with_none_dirname. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/10 statements.
# Partially parsed test_render_and_create_dir_with_template_variable. Retrieved 6/13 statements.
# Partially parsed test_render_and_create_dir_existing_directory_overwrite_false. Retrieved 5/14 statements.
# Partially parsed test_render_and_create_dir_existing_directory_overwrite_true. Retrieved 4/12 statements.
# Partially parsed test_render_and_create_dir_nested_directory. Retrieved 3/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = '/tmp/test_output'
    var_2 = {}
    var_3 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = '/tmp/test_output'
    var_2 = {}
    var_3 = None

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ project_name }}_dir'
    var_5 = 'my_project_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = True
    var_4 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'parent/child/nested'



