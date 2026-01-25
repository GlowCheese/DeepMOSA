####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_path_object. Retrieved 3/7 statements.
# Partially parsed test_get_file_name_with_string_path. Retrieved 4/5 statements.
# Partially parsed test_get_file_name_with_json_extension. Retrieved 3/4 statements.
# Partially parsed test_get_file_name_with_json_extension_and_path_object. Retrieved 2/6 statements.
# Partially parsed test_get_file_name_empty_directory. Retrieved 4/5 statements.
# Partially parsed test_get_file_name_complex_template_name. Retrieved 4/5 statements.
# Partially parsed test_get_file_name_template_with_multiple_dots. Retrieved 3/4 statements.
# Partially parsed test_get_file_name_template_with_multiple_dots_no_json. Retrieved 4/5 statements.


def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = [var_0]
    var_2 = 'template'
    var_3 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template.json'
    var_2 = module_0.get_file_name(var_0, var_1)

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = [var_0]
    var_2 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'mytemplate'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'mytemplate.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/home/user/replays'
    var_1 = 'my_template_v1'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'my_template_v1.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.backup.json'
    var_2 = module_0.get_file_name(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.backup'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'template.backup.json'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_writes_json_file. Retrieved 9/17 statements.
# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 10/18 statements.
# Partially parsed test_dump_adds_json_suffix_if_missing. Retrieved 9/15 statements.
# Partially parsed test_dump_does_not_add_duplicate_json_suffix. Retrieved 10/18 statements.
# Partially parsed test_dump_raises_error_if_cookiecutter_key_missing. Retrieved 6/10 statements.
# Partially parsed test_dump_preserves_context_structure. Retrieved 17/24 statements.


def test_case_0():
    var_0 = 'Test that dump creates a replay file with correct JSON content.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = "Test that dump creates the replay directory if it doesn't exist."
    var_1 = 'nonexistent'
    var_2 = 'replay'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test that dump adds .json suffix to template name if not present.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = "Test that dump doesn't add .json suffix if already present."
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'
    var_9 = 'test_template.json.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'cookiecutter key'

def test_case_0():
    var_0 = 'Test that dump preserves the full context structure in JSON.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'extra_key'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = 'nested'
    var_8 = 'my_project'
    var_9 = 'John Doe'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = {var_5: var_8, var_6: var_9, var_7: var_12}
    var_14 = 'extra_value'
    var_15 = {var_3: var_13, var_4: var_14}
    var_16 = 'test_template.json'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 9/17 statements.
# Partially parsed test_load_valid_json_with_json_extension. Retrieved 6/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'template'
    var_9 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'template'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'version'
    var_3 = '1.0'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 11/30 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'other_value'
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = 'template'
    var_11 = 'cookiecutter'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/14 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 8/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 7/12 statements.
# Partially parsed test_load_with_string_path. Retrieved 9/14 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.


import json as module_0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = 'utf-8'
    var_9 = 'template'
    var_10 = 'cookiecutter'

import json as module_0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = 'utf-8'

import json as module_0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'utf-8'
    var_7 = 'template'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'Context is required to contain a cookiecutter key'

import json as module_0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = 'utf-8'
    var_9 = 'template'

def test_case_0():
    var_0 = 'nonexistent_template'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 11/16 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key is present in context."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'test_author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = f'{var_2}.json'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'test.json'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/14 statements.
# Partially parsed test_load_with_template_name_already_having_json_extension. Retrieved 6/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'cookiecutter'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'nonexistent_template'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 10/15 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key is present in context."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'project_slug'
    var_6 = 'test_project'
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = f'{var_2}.json'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 10/13 statements.
# Partially parsed test_load_without_cookiecutter_key_raises_error. Retrieved 5/9 statements.
# Partially parsed test_load_file_not_found_raises_error. Retrieved 2/4 statements.
# Partially parsed test_load_with_path_object. Retrieved 10/13 statements.
# Partially parsed test_load_with_string_path. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'Test load function with a valid JSON file containing cookiecutter key.'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template.json'
    var_8 = '{"cookiecutter": {"project_name": "test_project"}}'
    var_9 = 'utf-8'
    var_10 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test load function when template_name already has .json extension.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template.json'
    var_8 = '{"cookiecutter": {"key": "value"}}'
    var_9 = 'utf-8'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'template'
    var_2 = 'template.json'
    var_3 = '{"other_key": "value"}'
    var_4 = 'utf-8'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'Test load function raises error when file does not exist.'
    var_1 = 'nonexistent'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'Test load function works with Path object as replay_dir.'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'data'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template.json'
    var_8 = '{"cookiecutter": {"data": "test"}}'
    var_9 = 'utf-8'

def test_case_0():
    var_0 = 'Test load function works with string path as replay_dir.'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'info'
    var_4 = 'content'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template.json'
    var_8 = '{"cookiecutter": {"info": "content"}}'
    var_9 = 'utf-8'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_template'
    var_7 = 'cookiecutter'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 11/30 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'other_value'
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = 'template'
    var_11 = 'cookiecutter'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_load_valid_json_with_cookiecutter_key_full_filename. Retrieved 6/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.
# Partially parsed test_load_invalid_json. Retrieved 3/10 statements.
# Partially parsed test_load_with_pathlib_path. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'author'
    var_3 = 'John Doe'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'template'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'nonexistent_template'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{ invalid json content'
    var_2 = 'template'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'other_value'
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = 'test.json'
    var_11 = 'cookiecutter'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/13 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 7/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'template.json'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_load_valid_json_with_json_extension. Retrieved 6/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'template'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/29 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = 'cookiecutter'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 9/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_with_path_object. Retrieved 9/12 statements.
# Partially parsed test_load_with_string_path. Retrieved 9/13 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.
# Partially parsed test_load_invalid_json_format. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'
    var_7 = '{"cookiecutter": {"project_name": "test_project"}}'
    var_8 = 'utf-8'
    var_9 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'
    var_7 = '{"cookiecutter": {"key": "value"}}'
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'template'
    var_1 = 'template.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = 'utf-8'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'
    var_7 = '{"cookiecutter": {"name": "test"}}'
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter'
    var_2 = 'value'
    var_3 = 123
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'
    var_7 = '{"cookiecutter": {"value": 123}}'
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'template'
    var_1 = 'template.json'
    var_2 = 'invalid json content'
    var_3 = 'utf-8'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dump_creates_replay_directory_and_writes_json. Retrieved 9/15 statements.
# Partially parsed test_dump_adds_json_suffix_if_missing. Retrieved 9/13 statements.
# Partially parsed test_dump_does_not_duplicate_json_suffix. Retrieved 10/16 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_with_nested_context. Retrieved 18/23 statements.
# Partially parsed test_dump_overwrites_existing_file. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'Test that dump creates directory and writes JSON file with context.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump adds .json suffix to template name if not present.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = "Test that dump doesn't add .json suffix if already present."
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'
    var_9 = 'my_template.json.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'cookiecutter key'

def test_case_0():
    var_0 = 'Test that dump correctly serializes nested context structure.'
    var_1 = 'replay'
    var_2 = 'complex_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'nested'
    var_6 = 'list'
    var_7 = 'test'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = {var_4: var_7, var_5: var_10, var_6: var_14}
    var_16 = {var_3: var_15}
    var_17 = 'complex_template.json'

def test_case_0():
    var_0 = 'Test that dump overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'old_value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new_value'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'template.json'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'other_value'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 'template'
    var_9 = 'cookiecutter'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_load_json_file_with_json_extension. Retrieved 6/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'template'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/27 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = 'cookiecutter'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 8/16 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 7/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/13 statements.
# Partially parsed test_load_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_load_with_path_object. Retrieved 8/15 statements.
# Partially parsed test_load_complex_cookiecutter_structure. Retrieved 18/24 statements.


def test_case_0():
    var_0 = 'Test load function with a valid JSON file containing cookiecutter key.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template'
    var_8 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test load function when template_name already has .json extension.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'template.json'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'template'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'Test load function works with Path object as replay_dir.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template'

def test_case_0():
    var_0 = 'Test load function with complex nested cookiecutter structure.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'other_data'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'options'
    var_7 = 'myproject'
    var_8 = 'John Doe'
    var_9 = 'feature1'
    var_10 = 'feature2'
    var_11 = True
    var_12 = False
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = {var_4: var_7, var_5: var_8, var_6: var_13}
    var_15 = 'value'
    var_16 = {var_2: var_14, var_3: var_15}
    var_17 = 'template'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dump_creates_replay_directory_and_writes_json. Retrieved 9/15 statements.
# Partially parsed test_dump_appends_json_suffix_when_not_present. Retrieved 9/13 statements.
# Partially parsed test_dump_does_not_append_json_suffix_when_already_present. Retrieved 9/13 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/9 statements.
# Partially parsed test_dump_writes_properly_formatted_json. Retrieved 11/16 statements.
# Partially parsed test_dump_with_nested_context. Retrieved 13/18 statements.


def test_case_0():
    var_0 = 'Test that dump creates the replay directory and writes context to JSON file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump appends .json suffix to template name if not present.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump does not append .json suffix if template name already ends with .json.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when context does not contain cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'Test that dump writes JSON with proper indentation.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'key1'
    var_5 = 'key2'
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my_template.json'
    var_11 = '  '

def test_case_0():
    var_0 = 'Test that dump correctly handles nested context dictionaries.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'nested'
    var_6 = 'test'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = {var_3: var_10}
    var_12 = 'my_template.json'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 6/15 statements.
# Partially parsed test_load_valid_json_without_json_extension. Retrieved 7/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/15 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/8 statements.
# Partially parsed test_load_with_path_object. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'author'
    var_3 = 'test_author'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'template.json'
    var_5 = module_0.load(var_0, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'config.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_with_valid_context. Retrieved 9/12 statements.
# Partially parsed test_load_with_template_name_already_having_json_extension. Retrieved 8/11 statements.
# Partially parsed test_load_without_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_with_path_object. Retrieved 9/12 statements.
# Partially parsed test_load_with_string_path. Retrieved 9/13 statements.
# Partially parsed test_load_with_complex_cookiecutter_data. Retrieved 15/18 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = '{"cookiecutter": {"project_name": "test_project"}}'
    var_8 = 'utf-8'
    var_9 = 'cookiecutter'

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '{"cookiecutter": {"key": "value"}}'
    var_7 = 'utf-8'

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = '{"other_key": "value"}'
    var_3 = 'utf-8'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'example'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = '{"cookiecutter": {"name": "example"}}'
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'value'
    var_3 = 123
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = '{"cookiecutter": {"value": 123}}'
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'complex_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'nested'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_5, var_3: var_6, var_4: var_9}
    var_11 = {var_1: var_10}
    var_12 = f'{var_0}.json'
    var_13 = '{"cookiecutter": {"project_name": "my_project", "author": "John Doe", "nested": {"key": "value"}}}'
    var_14 = 'utf-8'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'test'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 9/14 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key is present in context."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = f'{var_2}.json'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'other_value'
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = 'test_template'
    var_11 = 'cookiecutter'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 6/15 statements.
# Partially parsed test_load_json_without_suffix. Retrieved 7/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/15 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/8 statements.
# Partially parsed test_load_with_pathlib_path. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'template.json'
    var_5 = module_0.load(var_0, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 9/19 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'cookiecutter'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json. Retrieved 9/15 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 9/13 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_preserves_context_structure. Retrieved 17/22 statements.


def test_case_0():
    var_0 = 'Test that dump creates the replay directory and writes context to JSON file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = "Test that dump doesn't add .json suffix if template_name already has it."
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'cookiecutter key'

def test_case_0():
    var_0 = 'Test that dump preserves the full context structure in JSON file.'
    var_1 = 'replay'
    var_2 = 'complex_template'
    var_3 = 'cookiecutter'
    var_4 = 'extra_data'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = 'nested'
    var_8 = 'my_project'
    var_9 = 'John Doe'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = {var_5: var_8, var_6: var_9, var_7: var_12}
    var_14 = 'some_value'
    var_15 = {var_3: var_13, var_4: var_14}
    var_16 = 'complex_template.json'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 9/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/9 statements.
# Partially parsed test_load_with_path_object. Retrieved 10/13 statements.
# Partially parsed test_load_nonexistent_file. Retrieved 2/4 statements.
# Partially parsed test_load_with_complex_json_structure. Retrieved 21/24 statements.


def test_case_0():
    var_0 = 'Test load function with a valid JSON file containing cookiecutter key.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{"cookiecutter": {"project_name": "test_project"}}'
    var_8 = 'utf-8'
    var_9 = 'template'
    var_10 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test load function when template_name already has .json extension.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{"cookiecutter": {"key": "value"}}'
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'template.json'
    var_2 = '{"other_key": "value"}'
    var_3 = 'utf-8'
    var_4 = 'template'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'Test load function with Path object instead of string.'
    var_1 = 'config.json'
    var_2 = 'cookiecutter'
    var_3 = 'setting'
    var_4 = 'enabled'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{"cookiecutter": {"setting": "enabled"}}'
    var_8 = 'utf-8'
    var_9 = 'config'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError for nonexistent file.'
    var_1 = 'nonexistent'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'Test load function with complex JSON structure.'
    var_1 = 'complex.json'
    var_2 = 'cookiecutter'
    var_3 = 'extra_field'
    var_4 = 'project_name'
    var_5 = 'options'
    var_6 = 'nested'
    var_7 = 'myproject'
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_8, var_9, var_10]
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 'data'
    var_17 = {var_2: var_15, var_3: var_16}
    var_18 = '{"cookiecutter": {"project_name": "myproject", "options": ["a", "b", "c"], "nested": {"key": "value"}}, "extra_field": "data"}'
    var_19 = 'utf-8'
    var_20 = 'complex'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 4/7 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 3/6 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_with_empty_cookiecutter. Retrieved 4/7 statements.
# Partially parsed test_load_with_nested_cookiecutter_data. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {"project_name": "test"}}'
    var_2 = 'utf-8'
    var_3 = 'template'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {"key": "value"}}'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"other_key": "value"}'
    var_2 = 'utf-8'
    var_3 = 'template'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {}}'
    var_2 = 'utf-8'
    var_3 = 'template'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {"nested": {"key": "value"}}}'
    var_2 = 'utf-8'
    var_3 = 'template'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_load_raises_valueerror_when_cookiecutter_key_missing. Retrieved 6/15 statements.


import json as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'test_template'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test'
    var_7 = 'cookiecutter'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_string_path_and_template_without_json. Retrieved 4/7 statements.
# Partially parsed test_get_file_name_with_string_path_and_template_with_json. Retrieved 3/6 statements.
# Partially parsed test_get_file_name_with_path_object_and_template_without_json. Retrieved 3/8 statements.
# Partially parsed test_get_file_name_with_path_object_and_template_with_json. Retrieved 2/7 statements.
# Partially parsed test_get_file_name_with_nested_path_and_template_without_json. Retrieved 4/7 statements.
# Partially parsed test_get_file_name_with_nested_path_and_template_with_json. Retrieved 3/6 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'template'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'template.json'
    var_2 = module_0.get_file_name(var_0, var_1)

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = [var_0]
    var_2 = 'template'
    var_3 = 'template.json'

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = [var_0]
    var_2 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay_dir'
    var_1 = 'my_template'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'my_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay_dir'
    var_1 = 'my_template.json'
    var_2 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/15 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 6/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'template'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_creates_replay_directory. Retrieved 11/19 statements.
# Partially parsed test_dump_writes_json_file. Retrieved 11/20 statements.
# Partially parsed test_dump_missing_cookiecutter_key_raises_error. Retrieved 7/12 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 13/23 statements.
# Partially parsed test_dump_file_opened_with_utf8_encoding. Retrieved 11/19 statements.


def test_case_0():
    var_0 = "Test that dump creates the replay directory if it doesn't exist."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'builtins.open'
    var_10 = 'cookiecutter.replay.json.dump'

def test_case_0():
    var_0 = 'Test that dump writes context to json file.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'builtins.open'
    var_9 = 'cookiecutter.replay.json.dump'
    var_10 = 'cookiecutter.replay.make_sure_path_exists'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'cookiecutter'
    var_9 = bool('cookiecutter' in str(e).lower())
    assert var_9 is True

def test_case_0():
    var_0 = 'Test that dump handles template names with .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'builtins.open'
    var_9 = 'cookiecutter.replay.json.dump'
    var_10 = 'cookiecutter.replay.make_sure_path_exists'
    var_11 = 'test_template.json'
    var_12 = 0
    var_13 = '.json.json'

def test_case_0():
    var_0 = 'Test that dump opens file with utf-8 encoding.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'builtins.open'
    var_9 = 'cookiecutter.replay.json.dump'
    var_10 = 'cookiecutter.replay.make_sure_path_exists'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'test'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dump_creates_replay_file. Retrieved 11/19 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 9/17 statements.
# Partially parsed test_dump_raises_error_without_cookiecutter_key. Retrieved 6/10 statements.
# Partially parsed test_dump_creates_nested_directories. Retrieved 11/19 statements.
# Partially parsed test_dump_formats_json_with_indent. Retrieved 11/19 statements.


def test_case_0():
    var_0 = 'Test that dump creates a replay file with correct content.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my-template.json'

def test_case_0():
    var_0 = 'Test that dump handles template names that already end with .json.'
    var_1 = 'replay'
    var_2 = 'my-template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my-template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = "Test that dump creates nested directory structure if it doesn't exist."
    var_1 = 'nested'
    var_2 = 'replay'
    var_3 = 'dir'
    var_4 = 'my-template'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'my-template.json'

def test_case_0():
    var_0 = 'Test that dump formats JSON output with proper indentation.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my-template.json'
    var_11 = '  '



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dump_with_cookiecutter_in_context. Retrieved 9/14 statements.


def test_case_0():
    var_0 = "Test that dump function works when 'cookiecutter' key is in context."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = f'{var_2}.json'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_raises_when_cookiecutter_not_in_context. Retrieved 5/16 statements.
# Partially parsed test_load_succeeds_when_cookiecutter_in_context. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dump_creates_replay_directory. Retrieved 8/11 statements.
# Partially parsed test_dump_writes_json_file_with_correct_name. Retrieved 9/13 statements.
# Partially parsed test_dump_writes_json_file_with_json_suffix. Retrieved 9/13 statements.
# Partially parsed test_dump_does_not_double_suffix_json. Retrieved 9/13 statements.
# Partially parsed test_dump_writes_correct_json_content. Retrieved 11/16 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/9 statements.
# Partially parsed test_dump_with_nested_replay_directory. Retrieved 11/17 statements.
# Partially parsed test_dump_overwrites_existing_file. Retrieved 12/18 statements.
# Partially parsed test_dump_preserves_context_structure. Retrieved 18/23 statements.


def test_case_0():
    var_0 = "Test that dump creates the replay directory if it doesn't exist."
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test that dump writes a JSON file with the correct name.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump adds .json suffix to template name if not present.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = "Test that dump doesn't add .json suffix if template name already ends with .json."
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump writes the correct JSON content to file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'John Doe'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when cookiecutter key is not in context.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = "Test that dump creates nested directories if they don't exist."
    var_1 = 'nested'
    var_2 = 'replay'
    var_3 = 'path'
    var_4 = 'my_template'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump overwrites an existing replay file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'old'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump preserves complex context structures.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'nested'
    var_6 = 'test'
    var_7 = 'key'
    var_8 = 'list'
    var_9 = 'value'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = {var_7: var_9, var_8: var_13}
    var_15 = {var_4: var_6, var_5: var_14}
    var_16 = {var_3: var_15}
    var_17 = 'my_template.json'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/26 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'test'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_with_cookiecutter_key_present. Retrieved 9/31 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'cookiecutter'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dump_creates_replay_directory. Retrieved 8/11 statements.
# Partially parsed test_dump_writes_json_file. Retrieved 9/15 statements.
# Partially parsed test_dump_adds_json_suffix_when_missing. Retrieved 9/13 statements.
# Partially parsed test_dump_does_not_double_json_suffix. Retrieved 9/13 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_preserves_json_formatting. Retrieved 11/16 statements.
# Partially parsed test_dump_with_string_path. Retrieved 8/13 statements.


def test_case_0():
    var_0 = "Test that dump creates the replay directory if it doesn't exist."
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test that dump writes a valid JSON file with correct content.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump adds .json suffix to template name if not present.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = "Test that dump doesn't add .json suffix if template name already ends with .json."
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'cookiecutter'
    var_8 = bool('cookiecutter' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = 'Test that dump writes JSON with proper indentation.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test'
    var_7 = 'John'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my_template.json'
    var_11 = '  '

def test_case_0():
    var_0 = 'Test that dump works with string path instead of Path object.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 9/20 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = {}
    var_8 = module_0.dumps(var_5, **var_7)
    var_9 = 'utf-8'
    var_10 = {}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/15 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 6/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_with_path_object. Retrieved 9/16 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'cookiecutter'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'nested'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = f'{var_0}.json'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 11/16 statements.


def test_case_0():
    var_0 = "Test that dump function succeeds when 'cookiecutter' key exists in context."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'test_author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = f'{var_2}.json'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 3/6 statements.
# Partially parsed test_load_with_template_name_without_extension. Retrieved 4/7 statements.
# Partially parsed test_load_raises_error_when_cookiecutter_key_missing. Retrieved 4/8 statements.
# Partially parsed test_load_with_complex_cookiecutter_context. Retrieved 4/7 statements.
# Partially parsed test_load_with_path_object. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {"project_name": "test"}}'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {"key": "value"}}'
    var_2 = 'utf-8'
    var_3 = 'template'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"other_key": "value"}'
    var_2 = 'utf-8'
    var_3 = 'template.json'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'config.json'
    var_1 = '{"cookiecutter": {"name": "project", "version": "1.0", "nested": {"key": "value"}}}'
    var_2 = 'utf-8'
    var_3 = 'config'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {"test": "data"}}'
    var_2 = 'utf-8'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = 'cookiecutter'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dump_writes_json_file_with_valid_context. Retrieved 7/15 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 4/8 statements.
# Partially parsed test_dump_creates_replay_directory_if_not_exists. Retrieved 8/14 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 7/12 statements.
# Partially parsed test_dump_writes_properly_formatted_json. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_template.json'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'new_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_template.json'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test'
    var_5 = 'John'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = '"cookiecutter"'
    var_9 = '"project_name"'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'value'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 'test_template'
    var_9 = 'cookiecutter'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/28 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = 'cookiecutter'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 8/15 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 8/15 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/13 statements.
# Partially parsed test_load_with_pathlib_path. Retrieved 8/15 statements.
# Partially parsed test_load_with_string_path. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'Test load function with a valid JSON file containing cookiecutter key.'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template.json'
    var_8 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test load function when template_name already has .json extension.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template.json'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'template.json'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'Test load function works with pathlib.Path as replay_dir.'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template.json'

def test_case_0():
    var_0 = 'Test load function works with string as replay_dir.'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'data'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template.json'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_template'
    var_7 = 'cookiecutter'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'cookiecutter'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'test_template'
    var_9 = 'cookiecutter'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 9/18 statements.


def test_case_0():
    var_0 = "Test that dump function accepts context with 'cookiecutter' key."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = f'{var_2}.json'
    var_9 = 'cookiecutter'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'
    var_7 = 'cookiecutter'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/15 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 6/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'data'
    var_2 = 'no cookiecutter key'
    var_3 = {var_1: var_2}
    var_4 = 'template'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test'
    var_7 = 'cookiecutter'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json. Retrieved 9/15 statements.
# Partially parsed test_dump_with_json_suffix. Retrieved 9/13 statements.
# Partially parsed test_dump_without_cookiecutter_key_raises_error. Retrieved 6/9 statements.
# Partially parsed test_dump_with_existing_directory. Retrieved 10/15 statements.
# Partially parsed test_dump_overwrites_existing_file. Retrieved 13/20 statements.


def test_case_0():
    var_0 = 'Test that dump creates directory and writes context to JSON file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump handles template names that already end with .json.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'Test that dump works when directory already exists.'
    var_1 = 'replay'
    var_2 = True
    var_3 = 'my_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = True
    var_3 = 'my_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'old_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'new_project'
    var_10 = {var_5: var_9}
    var_11 = {var_4: var_10}
    var_12 = 'my_template.json'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 11/22 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'other_value'
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = 'test_template'
    var_11 = 'cookiecutter'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 3/6 statements.
# Partially parsed test_load_with_template_name_without_extension. Retrieved 4/7 statements.
# Partially parsed test_load_raises_error_when_cookiecutter_key_missing. Retrieved 4/8 statements.
# Partially parsed test_load_with_complex_context. Retrieved 4/7 statements.
# Partially parsed test_load_with_string_path. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {"project_name": "test_project"}}'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {"author": "test_author"}}'
    var_2 = 'utf-8'
    var_3 = 'template'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"project_name": "test_project"}'
    var_2 = 'utf-8'
    var_3 = 'template.json'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'config.json'
    var_1 = '{"cookiecutter": {"name": "app", "version": "1.0", "features": ["auth", "db"]}}'
    var_2 = 'utf-8'
    var_3 = 'config'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {"key": "value"}}'
    var_2 = 'utf-8'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'test.json'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/28 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dump_with_cookiecutter_in_context. Retrieved 8/13 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key is present in context."
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_load_with_valid_cookiecutter_context. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = f'{var_0}.json'
    var_9 = 'cookiecutter'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_load_without_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'test.json'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_dump_creates_replay_directory_and_writes_json. Retrieved 9/15 statements.
# Partially parsed test_dump_adds_json_suffix_if_not_present. Retrieved 9/13 statements.
# Partially parsed test_dump_does_not_double_add_json_suffix. Retrieved 9/13 statements.
# Partially parsed test_dump_raises_value_error_if_cookiecutter_key_missing. Retrieved 6/9 statements.
# Partially parsed test_dump_uses_path_object. Retrieved 9/13 statements.
# Partially parsed test_dump_uses_string_path. Retrieved 9/15 statements.
# Partially parsed test_dump_writes_json_with_correct_formatting. Retrieved 11/16 statements.
# Partially parsed test_dump_overwrites_existing_file. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'Test that dump creates the replay directory and writes JSON file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump adds .json suffix to template name if not present.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = "Test that dump doesn't add .json suffix if already present."
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError if cookiecutter key is missing from context.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'Test that dump works with Path object as replay_dir.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump works with string path as replay_dir.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump writes JSON with proper indentation.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'John'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my_template.json'
    var_11 = '  '

def test_case_0():
    var_0 = 'Test that dump overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'old_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new_project'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'my_template.json'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'test_template'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 7/16 statements.


import json as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'utf-8'
    var_7 = 'template'
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = 'cookiecutter'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_load_valid_context_with_cookiecutter_key. Retrieved 6/14 statements.
# Partially parsed test_load_valid_context_without_json_extension. Retrieved 7/15 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/14 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/5 statements.
# Partially parsed test_load_with_path_object. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'template.json'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



