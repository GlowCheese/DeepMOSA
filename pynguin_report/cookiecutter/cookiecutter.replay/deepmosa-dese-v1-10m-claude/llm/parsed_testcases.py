####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_path_object_without_json_extension. Retrieved 3/8 statements.
# Partially parsed test_get_file_name_with_string_path_without_json_extension. Retrieved 4/5 statements.
# Partially parsed test_get_file_name_with_json_extension. Retrieved 3/4 statements.
# Partially parsed test_get_file_name_with_path_object_and_json_extension. Retrieved 2/7 statements.
# Partially parsed test_get_file_name_with_empty_directory. Retrieved 4/5 statements.
# Partially parsed test_get_file_name_with_nested_path. Retrieved 4/5 statements.


def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = 'template.json'

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
    var_1 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'myfile'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'myfile.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/home/user/replays/data'
    var_1 = 'replay_template'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'replay_template.json'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json. Retrieved 23/35 statements.
# Partially parsed test_dump_raises_valueerror_without_cookiecutter_key. Retrieved 9/14 statements.
# Partially parsed test_dump_adds_json_suffix_to_template_name. Retrieved 15/24 statements.
# Partially parsed test_dump_does_not_add_json_suffix_if_already_present. Retrieved 15/24 statements.


def test_case_0():
    var_0 = 'Test that dump creates the replay directory and writes context to JSON file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = None
    var_10 = lambda x: var_9
    var_11 = 'builtins.open'
    var_12 = 'MockFile'
    var_13 = ()
    var_14 = '__enter__'
    var_15 = '__exit__'
    var_16 = 'write'
    var_17 = lambda self: self
    var_18 = lambda self, *args: var_9
    var_19 = lambda self, x: var_9
    var_20 = {var_14: var_17, var_15: var_18, var_16: var_19}
    var_21 = []
    var_22 = 'json.dump'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.replay.make_sure_path_exists'
    var_7 = None
    var_8 = lambda x: var_7

def test_case_0():
    var_0 = 'Test that dump adds .json suffix to template name if not present.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.replay.make_sure_path_exists'
    var_10 = None
    var_11 = lambda x: var_10
    var_12 = 'builtins.open'
    var_13 = 'json.dump'
    var_14 = lambda *args, **kwargs: var_10

def test_case_0():
    var_0 = 'Test that dump does not add .json suffix if template name already has it.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'cookiecutter.replay.make_sure_path_exists'
    var_10 = None
    var_11 = lambda x: var_10
    var_12 = 'builtins.open'
    var_13 = 'json.dump'
    var_14 = lambda *args, **kwargs: var_10



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json. Retrieved 9/15 statements.
# Partially parsed test_dump_adds_json_suffix_if_missing. Retrieved 9/13 statements.
# Partially parsed test_dump_does_not_duplicate_json_suffix. Retrieved 9/13 statements.
# Partially parsed test_dump_raises_value_error_if_cookiecutter_key_missing. Retrieved 6/9 statements.
# Partially parsed test_dump_writes_valid_json. Retrieved 15/20 statements.


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
    var_0 = "Test that dump adds .json suffix if template_name doesn't have it."
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = "Test that dump doesn't add .json suffix if template_name already has it."
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError if context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test that dump writes valid JSON that can be read back.'
    var_1 = 'replay'
    var_2 = 'template'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'version'
    var_6 = 'nested'
    var_7 = 'test'
    var_8 = '1.0'
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = {var_4: var_7, var_5: var_8, var_6: var_11}
    var_13 = {var_3: var_12}
    var_14 = 'template.json'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 13/18 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key is present in context."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'other_key'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = 'test_project'
    var_8 = 'test_author'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'other_value'
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = f'{var_2}.json'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dump_with_cookiecutter_in_context. Retrieved 11/16 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key is present in context."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = f'{var_2}.json'



# Parsed testcases at query #6
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

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'test'



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dump_with_cookiecutter_in_context. Retrieved 10/15 statements.


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = f'{var_1}.json'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 8/11 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_with_pathlib_path. Retrieved 9/12 statements.
# Partially parsed test_load_with_string_path. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = '{"cookiecutter": {"project_name": "my_project"}}'
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'version'
    var_3 = '1.0'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '{"cookiecutter": {"version": "1.0"}}'
    var_7 = 'utf-8'

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = '{"data": "value"}'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = '{"cookiecutter": {"key": "value"}}'
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = '{"cookiecutter": {"name": "test"}}'
    var_8 = 'utf-8'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_raises_valueerror_when_cookiecutter_key_missing. Retrieved 5/27 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 9/18 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'utf-8'



# Parsed testcases at query #13
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
    var_10 = 'test_template'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/17 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 6/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.


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

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 7/14 statements.
# Partially parsed test_load_valid_json_with_cookiecutter_key_explicit_extension. Retrieved 6/11 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
# Partially parsed test_load_empty_cookiecutter_key. Retrieved 5/10 statements.
# Partially parsed test_load_nested_cookiecutter_data. Retrieved 14/19 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = 'template'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'

def test_case_0():
    var_0 = 'other_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'template.json'
    var_4 = 'template'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'template.json'
    var_4 = 'template'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'nested'
    var_2 = 'list'
    var_3 = 'deep'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = {var_1: var_5, var_2: var_9}
    var_11 = {var_0: var_10}
    var_12 = 'template.json'
    var_13 = 'template'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 9/21 statements.


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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_raises_valueerror_when_cookiecutter_key_missing. Retrieved 7/16 statements.


import json as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = 'utf-8'
    var_6 = 'test_template'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/14 statements.
# Partially parsed test_load_without_json_extension. Retrieved 8/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 2/4 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'Test load function with valid JSON file containing cookiecutter key.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'

def test_case_0():
    var_0 = 'Test load function adds .json extension when not provided.'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'
    var_7 = 'template'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'template.json'
    var_5 = 'template.json'

def test_case_0():
    var_0 = "Test load function raises FileNotFoundError when file doesn't exist."
    var_1 = 'nonexistent.json'

def test_case_0():
    var_0 = 'Test load function works with Path object as replay_dir.'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 7/16 statements.


import json as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = 'utf-8'
    var_6 = 'test_template'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/23 statements.
# Partially parsed test_load_without_json_extension. Retrieved 7/14 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/13 statements.
# Partially parsed test_load_with_pathlib_path. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = 0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'template.json'
    var_5 = module_0.load(var_0, var_4)

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'version'
    var_3 = '1.0'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'test'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/16 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 6/14 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/5 statements.
# Partially parsed test_load_with_string_path. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'

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

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 11/16 statements.


def test_case_0():
    var_0 = "Test that dump function works when 'cookiecutter' key is in context."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'my_project'
    var_7 = 'John Doe'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = f'{var_2}.json'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 8/15 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 7/14 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/14 statements.
# Partially parsed test_load_with_path_object. Retrieved 8/15 statements.
# Partially parsed test_load_nonexistent_file. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'Test load function with valid JSON file containing cookiecutter key.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template'

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

def test_case_0():
    var_0 = 'Test load function works with Path object as replay_dir.'
    var_1 = 'config.json'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'config'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError for nonexistent file.'
    var_1 = 'nonexistent'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_with_valid_context. Retrieved 7/13 statements.
# Partially parsed test_load_with_template_name_already_has_json_extension. Retrieved 6/12 statements.
# Partially parsed test_load_with_path_object. Retrieved 9/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = 'template'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'nested'
    var_2 = 'data'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'config.json'
    var_8 = 'config'

def test_case_0():
    var_0 = 'other_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'template.json'
    var_4 = 'template'

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 8/16 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 7/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/13 statements.
# Partially parsed test_load_with_path_object. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'Test load function with valid JSON file containing cookiecutter key.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template'

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

def test_case_0():
    var_0 = 'Test load function with Path object as replay_dir.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/14 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 6/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.
# Partially parsed test_load_with_string_path. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 8/13 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key is present in context."
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 11/25 statements.


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
    var_10 = 'utf-8'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 6/13 statements.
# Partially parsed test_load_with_template_name_without_json_extension. Retrieved 7/14 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_load_with_string_replay_dir. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'template.json'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_dump_creates_replay_directory_and_writes_json_file. Retrieved 9/15 statements.
# Partially parsed test_dump_appends_json_extension_if_missing. Retrieved 9/13 statements.
# Partially parsed test_dump_does_not_duplicate_json_extension. Retrieved 9/13 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/9 statements.
# Partially parsed test_dump_writes_json_with_proper_formatting. Retrieved 11/16 statements.
# Partially parsed test_dump_overwrites_existing_replay_file. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'Test that dump creates directory and writes context to json file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump appends .json extension to template name if not present.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump does not add .json if template name already ends with .json.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when context missing cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test that dump writes json with proper indentation.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'nested'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump overwrites existing replay file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'old_value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'new_value'
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'my_template.json'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_load_validates_cookiecutter_key_exists. Retrieved 7/31 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 9/19 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = module_0.dumps(var_5)
    var_8 = 'utf-8'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'test.json'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/24 statements.
# Partially parsed test_load_without_json_extension. Retrieved 7/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/15 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/7 statements.
# Partially parsed test_load_with_path_object. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = 0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'template.json'
    var_5 = module_0.load(var_0, var_4)

def test_case_0():
    var_0 = 'nonexistent.json'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'version'
    var_3 = '1.0'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_template_without_json_extension. Retrieved 2/5 statements.
# Partially parsed test_get_file_name_with_template_with_json_extension. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/tmp/replay/template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/tmp/replay/template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = ''
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/tmp/replay/.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = '.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/tmp/replay/.json'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_creates_replay_directory. Retrieved 11/18 statements.
# Partially parsed test_dump_raises_error_without_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_writes_json_file. Retrieved 11/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 12/20 statements.
# Partially parsed test_dump_file_opened_with_correct_encoding. Retrieved 11/18 statements.


def test_case_0():
    var_0 = "Test that dump creates the replay directory if it doesn't exist."
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'cookiecutter.replay.json.dump'
    var_10 = 'builtins.open'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test that dump writes context to json file.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'builtins.open'
    var_10 = 'cookiecutter.replay.json.dump'

def test_case_0():
    var_0 = 'Test that dump handles template names with .json extension.'
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'builtins.open'
    var_10 = 'cookiecutter.replay.json.dump'
    var_11 = 0

def test_case_0():
    var_0 = 'Test that dump opens file with utf-8 encoding.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.replay.make_sure_path_exists'
    var_9 = 'builtins.open'
    var_10 = 'cookiecutter.replay.json.dump'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 6/14 statements.
# Partially parsed test_load_with_template_name_without_extension. Retrieved 7/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_with_path_object. Retrieved 6/13 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
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

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'nonexistent.json'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 8/16 statements.
# Partially parsed test_load_without_json_extension. Retrieved 7/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_with_path_object. Retrieved 6/13 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'template.json'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'nonexistent.json'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 11/20 statements.


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
    var_10 = 'test'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_with_valid_context. Retrieved 9/18 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'utf-8'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 11/16 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key exists in context."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'my_project'
    var_7 = 'John Doe'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = f'{var_2}.json'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dump_creates_replay_directory_and_writes_json_file. Retrieved 9/17 statements.
# Partially parsed test_dump_with_json_suffix_in_template_name. Retrieved 9/14 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/10 statements.
# Partially parsed test_dump_creates_nested_replay_directory. Retrieved 11/19 statements.
# Partially parsed test_dump_writes_json_with_proper_formatting. Retrieved 11/17 statements.


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
    var_0 = "Test that dump doesn't add .json suffix if template name already has it."
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = "Test that dump raises ValueError when context doesn't contain cookiecutter key."
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test that dump creates nested replay directories if they don't exist."
    var_1 = 'nested'
    var_2 = 'replay'
    var_3 = 'dir'
    var_4 = 'my_template'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump writes JSON with proper indentation.'
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dump_creates_replay_directory_and_writes_json. Retrieved 9/14 statements.
# Partially parsed test_dump_with_json_suffix_in_template_name. Retrieved 9/13 statements.
# Partially parsed test_dump_raises_valueerror_when_cookiecutter_key_missing. Retrieved 6/9 statements.
# Partially parsed test_dump_with_nested_replay_directory. Retrieved 11/18 statements.
# Partially parsed test_dump_overwrites_existing_replay_file. Retrieved 12/18 statements.
# Partially parsed test_dump_with_string_replay_dir. Retrieved 9/15 statements.


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
    var_0 = "Test that dump doesn't add .json suffix if template name already has it."
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test that dump creates nested directory structure if needed.'
    var_1 = 'nested'
    var_2 = 'replay'
    var_3 = 'dir'
    var_4 = 'template'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'template.json'

def test_case_0():
    var_0 = 'Test that dump overwrites an existing replay file.'
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

def test_case_0():
    var_0 = 'Test that dump works with string path instead of Path object.'
    var_1 = 'replay'
    var_2 = 'template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'template.json'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_raises_valueerror_when_cookiecutter_key_missing. Retrieved 7/18 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = 'utf-8'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 6/13 statements.
# Partially parsed test_load_template_name_without_json_extension. Retrieved 7/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.
# Partially parsed test_load_with_path_object. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'version'
    var_2 = '1.0'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'config.json'
    var_6 = 'config'

def test_case_0():
    var_0 = 'other_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'template.json'
    var_4 = 'template.json'

def test_case_0():
    var_0 = 'nonexistent.json'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 8/13 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key is present in context."
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 6/14 statements.
# Partially parsed test_load_with_template_name_without_json_extension. Retrieved 7/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.
# Partially parsed test_load_with_path_object. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'template.json'

def test_case_0():
    var_0 = 'nonexistent.json'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 9/21 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = module_0.dumps(var_5)
    var_8 = 'utf-8'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 9/19 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'utf-8'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = f'{var_1}.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_load_valid_json_with_json_extension. Retrieved 6/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/13 statements.
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

def test_case_0():
    var_0 = 'nonexistent'

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

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 9/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.
# Partially parsed test_load_invalid_json. Retrieved 4/8 statements.


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
    var_2 = '{"other_key": "value"}'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'template'
    var_1 = 'template.json'
    var_2 = 'invalid json content'
    var_3 = 'utf-8'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 7/14 statements.
# Partially parsed test_load_json_file_with_json_extension. Retrieved 6/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.
# Partially parsed test_load_with_string_path. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'

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

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'value'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key_raises_valueerror. Retrieved 7/16 statements.


import json as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = 'utf-8'
    var_6 = 'template'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json. Retrieved 8/14 statements.
# Partially parsed test_dump_with_json_suffix_in_template_name. Retrieved 8/13 statements.
# Partially parsed test_dump_raises_error_without_cookiecutter_key. Retrieved 5/9 statements.
# Partially parsed test_dump_overwrites_existing_file. Retrieved 11/17 statements.
# Partially parsed test_dump_with_string_path. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'version'
    var_4 = '1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '2'
    var_8 = {var_3: var_7}
    var_9 = {var_2: var_8}
    var_10 = 'my_template.json'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 9/12 statements.
# Partially parsed test_load_without_cookiecutter_key. Retrieved 8/12 statements.
# Partially parsed test_load_with_complex_context. Retrieved 18/21 statements.
# Partially parsed test_load_with_string_replay_dir. Retrieved 10/14 statements.


import json as module_0

def test_case_0():
    var_0 = 'Test load function with valid JSON file containing cookiecutter key.'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'
    var_8 = module_0.dumps(var_6)
    var_9 = 'utf-8'

import json as module_0

def test_case_0():
    var_0 = 'Test load function when template_name already has .json extension.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'utf-8'

import json as module_0

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = f'{var_1}.json'
    var_6 = module_0.dumps(var_4)
    var_7 = 'utf-8'

import json as module_0

def test_case_0():
    var_0 = 'Test load function with complex nested JSON structure.'
    var_1 = 'complex_template'
    var_2 = 'cookiecutter'
    var_3 = 'other_data'
    var_4 = 'project_name'
    var_5 = 'options'
    var_6 = 'my_project'
    var_7 = 'nested'
    var_8 = 'value'
    var_9 = 123
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = {var_4: var_6, var_5: var_11}
    var_13 = 'some_value'
    var_14 = {var_2: var_12, var_3: var_13}
    var_15 = f'{var_1}.json'
    var_16 = module_0.dumps(var_14)
    var_17 = 'utf-8'

import json as module_0

def test_case_0():
    var_0 = 'Test load function works with string path instead of Path object.'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'
    var_8 = module_0.dumps(var_6)
    var_9 = 'utf-8'



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/14 statements.
# Partially parsed test_load_with_template_name_already_has_json_extension. Retrieved 6/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_with_string_replay_dir. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 9/20 statements.


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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 11/20 statements.


def test_case_0():
    var_0 = "Test that dump function works when 'cookiecutter' key is present in context."
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/21 statements.


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

# Partially parsed test_load_requires_cookiecutter_key. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'test.json'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 13/22 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key is present in context."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'extra_key'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = 'test_project'
    var_8 = 'test_author'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'extra_value'
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = f'{var_2}.json'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 3/6 statements.
# Partially parsed test_load_without_json_extension. Retrieved 4/7 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_with_complex_json. Retrieved 3/6 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {"project_name": "test_project"}}'
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

def test_case_0():
    var_0 = 'config.json'
    var_1 = '{"cookiecutter": {"name": "test", "nested": {"value": 123}}}'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'nonexistent.json'



# Parsed testcases at query #35
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



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_load_raises_valueerror_when_cookiecutter_key_missing. Retrieved 7/16 statements.


import json as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = 'utf-8'
    var_6 = 'test_template'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_load_with_valid_context. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 10/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/9 statements.
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

def test_case_0():
    var_0 = 'Test load function works with Path object as replay_dir.'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template.json'
    var_8 = '{"cookiecutter": {"name": "test"}}'
    var_9 = 'utf-8'

def test_case_0():
    var_0 = 'Test load function works with string as replay_dir.'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'data'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template.json'
    var_8 = '{"cookiecutter": {"data": "value"}}'
    var_9 = 'utf-8'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 11/16 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key is in context."
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



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/15 statements.
# Partially parsed test_load_with_template_name_already_has_json_extension. Retrieved 6/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_load_with_pathlib_path. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #42
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



