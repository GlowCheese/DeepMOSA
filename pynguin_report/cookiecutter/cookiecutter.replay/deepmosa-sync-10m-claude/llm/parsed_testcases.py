####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_valid_json_file. Retrieved 7/14 statements.
# Partially parsed test_load_json_file_with_json_extension. Retrieved 6/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.


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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test.json'
    var_6 = 'cookiecutter'



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_valid_context. Retrieved 8/16 statements.
# Partially parsed test_dump_adds_json_suffix_when_missing. Retrieved 8/14 statements.
# Partially parsed test_dump_does_not_duplicate_json_suffix. Retrieved 8/14 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 5/9 statements.
# Partially parsed test_dump_creates_nested_directories. Retrieved 11/19 statements.
# Partially parsed test_dump_writes_valid_json_format. Retrieved 10/18 statements.


def test_case_0():
    var_0 = 'Test that dump creates a replay file with valid context.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test that dump adds .json suffix to template name if not present.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test that dump does not add .json suffix if template name already has it.'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = "Test that dump creates nested directories if they don't exist."
    var_1 = 'nested'
    var_2 = 'replay'
    var_3 = 'dir'
    var_4 = 'test_template'
    var_5 = 'cookiecutter'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'test_template.json'

def test_case_0():
    var_0 = 'Test that dump writes context in valid JSON format with proper indentation.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'value'
    var_5 = 'test'
    var_6 = 123
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_template.json'
    var_10 = '  '



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_with_valid_context. Retrieved 6/12 statements.
# Partially parsed test_load_with_template_name_without_json_extension. Retrieved 7/13 statements.
# Partially parsed test_load_with_path_object. Retrieved 6/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_with_complex_context. Retrieved 15/21 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = 'cookiecutter'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = 'template'
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'config.json'

def test_case_0():
    var_0 = 'other_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'template.json'
    var_4 = 'template.json'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'extra_field'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'nested'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_5, var_3: var_6, var_4: var_9}
    var_11 = 'extra_value'
    var_12 = {var_0: var_10, var_1: var_11}
    var_13 = 'template.json'
    var_14 = 'template'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 9/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 7/11 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.
# Partially parsed test_load_with_path_object. Retrieved 11/14 statements.


import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'
    var_7 = {}
    var_8 = module_0.dumps(var_5, **var_7)
    var_9 = 'utf-8'
    var_10 = 'cookiecutter'

import json as module_0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'
    var_7 = {}
    var_8 = module_0.dumps(var_5, **var_7)
    var_9 = 'utf-8'

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'template.json'
    var_5 = {}
    var_6 = module_0.dumps(var_3, **var_5)
    var_7 = 'utf-8'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = bool(False)
    assert var_1 is True

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter'
    var_2 = 'nested'
    var_3 = 'data'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'template.json'
    var_9 = {}
    var_10 = module_0.dumps(var_7, **var_9)
    var_11 = 'utf-8'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 6/14 statements.
# Partially parsed test_load_json_without_extension. Retrieved 7/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
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



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json. Retrieved 9/15 statements.
# Partially parsed test_dump_with_json_extension_in_template_name. Retrieved 9/15 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/9 statements.
# Partially parsed test_dump_creates_nested_directories. Retrieved 11/19 statements.
# Partially parsed test_dump_with_complex_context. Retrieved 21/27 statements.


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
    var_0 = 'Test that dump does not add extra .json when template_name already has it.'
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
    var_3 = 'some_key'
    var_4 = 'some_value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = "Test that dump creates nested directory structure if it doesn't exist."
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
    var_0 = 'Test that dump correctly serializes complex context structure.'
    var_1 = 'replay'
    var_2 = 'complex_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'features'
    var_7 = 'config'
    var_8 = 'test_project'
    var_9 = 'John Doe'
    var_10 = 'feature1'
    var_11 = 'feature2'
    var_12 = [var_10, var_11]
    var_13 = 'nested'
    var_14 = 'value'
    var_15 = True
    var_16 = 42
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = {var_4: var_8, var_5: var_9, var_6: var_12, var_7: var_17}
    var_19 = {var_3: var_18}
    var_20 = 'complex_template.json'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json. Retrieved 9/15 statements.
# Partially parsed test_dump_adds_json_suffix_if_not_present. Retrieved 9/13 statements.
# Partially parsed test_dump_does_not_duplicate_json_suffix. Retrieved 10/16 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_preserves_context_structure. Retrieved 22/27 statements.


def test_case_0():
    var_0 = 'Test that dump creates directory and writes context to JSON file.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my-template.json'

def test_case_0():
    var_0 = 'Test that dump adds .json suffix to template name if not present.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my-template.json'

def test_case_0():
    var_0 = 'Test that dump does not add .json suffix if already present.'
    var_1 = 'replay'
    var_2 = 'my-template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my-template.json'
    var_9 = 'my-template.json.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'my-template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'cookiecutter key'

def test_case_0():
    var_0 = 'Test that dump preserves the exact structure of the context.'
    var_1 = 'replay'
    var_2 = 'complex-template'
    var_3 = 'cookiecutter'
    var_4 = 'extra_key'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = 'nested'
    var_8 = 'my_project'
    var_9 = 'John Doe'
    var_10 = 'key'
    var_11 = 'list'
    var_12 = 'value'
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = {var_10: var_12, var_11: var_16}
    var_18 = {var_5: var_8, var_6: var_9, var_7: var_17}
    var_19 = 'extra_value'
    var_20 = {var_3: var_18, var_4: var_19}
    var_21 = 'complex-template.json'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_raises_valueerror_when_cookiecutter_not_in_context. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'cookiecutter'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dump_creates_replay_directory_and_writes_json. Retrieved 9/18 statements.
# Partially parsed test_dump_with_json_suffix_in_template_name. Retrieved 9/18 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/10 statements.
# Partially parsed test_dump_with_string_path. Retrieved 9/21 statements.
# Partially parsed test_dump_overwrites_existing_replay_file. Retrieved 12/20 statements.


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
    var_0 = "Test that dump doesn't add extra .json suffix if template name already has it."
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when cookiecutter key is missing from context.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'Test that dump works with string path instead of Path object.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dump_creates_replay_directory_and_writes_json. Retrieved 9/15 statements.
# Partially parsed test_dump_with_json_extension_in_template_name. Retrieved 9/15 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/9 statements.
# Partially parsed test_dump_with_existing_replay_directory. Retrieved 10/15 statements.
# Partially parsed test_dump_writes_json_with_proper_formatting. Retrieved 11/16 statements.
# Partially parsed test_dump_with_nested_context. Retrieved 15/20 statements.


def test_case_0():
    var_0 = 'Test that dump creates replay directory and writes context to JSON file.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test that dump handles template names that already have .json extension.'
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when context lacks cookiecutter key.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'Test that dump works when replay directory already exists.'
    var_1 = 'replay'
    var_2 = True
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'Test that dump writes JSON with proper indentation.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'John Doe'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'test_template.json'
    var_11 = '  '

def test_case_0():
    var_0 = 'Test that dump handles nested context dictionaries.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'config'
    var_6 = 'test_project'
    var_7 = 'debug'
    var_8 = 'version'
    var_9 = True
    var_10 = '1.0'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {var_4: var_6, var_5: var_11}
    var_13 = {var_3: var_12}
    var_14 = 'test_template.json'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 8/11 statements.
# Partially parsed test_load_with_template_name_without_json_extension. Retrieved 9/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_with_path_object. Retrieved 8/11 statements.
# Partially parsed test_load_with_string_path. Retrieved 8/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '{"cookiecutter": {"project_name": "test_project"}}'
    var_7 = 'utf-8'
    var_8 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '{"cookiecutter": {"project_name": "test_project"}}'
    var_7 = 'utf-8'
    var_8 = 'template'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"data": "value"}'
    var_2 = 'utf-8'
    var_3 = 'template.json'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '{"cookiecutter": {"key": "value"}}'
    var_7 = 'utf-8'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '{"cookiecutter": {"key": "value"}}'
    var_7 = 'utf-8'

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'replay.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'replay.json'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dump_writes_json_file_with_valid_context. Retrieved 7/15 statements.
# Partially parsed test_dump_adds_json_extension_if_missing. Retrieved 7/12 statements.
# Partially parsed test_dump_does_not_add_duplicate_json_extension. Retrieved 7/12 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 4/8 statements.
# Partially parsed test_dump_creates_replay_directory_if_not_exists. Retrieved 8/14 statements.
# Partially parsed test_dump_preserves_context_structure. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'my_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_template.json'

def test_case_0():
    var_0 = 'my_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_template.json'

def test_case_0():
    var_0 = 'my_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'my_template.json'

def test_case_0():
    var_0 = 'my_template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'cookiecutter key'

def test_case_0():
    var_0 = 'new_dir'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'

def test_case_0():
    var_0 = 'my_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'nested'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_5, var_3: var_6, var_4: var_9}
    var_11 = {var_1: var_10}



# Parsed testcases at query #13
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
    var_7 = {}
    var_8 = 'cookiecutter'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 3/6 statements.
# Partially parsed test_load_with_template_name_without_json_extension. Retrieved 4/7 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_with_path_object. Retrieved 3/6 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.
# Partially parsed test_load_with_complex_cookiecutter_structure. Retrieved 3/6 statements.


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
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'config.json'
    var_1 = '{"cookiecutter": {"name": "example"}}'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {"nested": {"key": "value"}, "list": [1, 2, 3]}, "other": "data"}'
    var_2 = 'utf-8'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'other_value'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 'test_template'
    var_9 = 'cookiecutter'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'test.json'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 11/20 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'other_value'
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = f'{var_0}.json'
    var_11 = 'cookiecutter'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json. Retrieved 9/15 statements.
# Partially parsed test_dump_with_json_suffix_in_template_name. Retrieved 9/15 statements.
# Partially parsed test_dump_raises_error_without_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_with_nested_context. Retrieved 18/23 statements.
# Partially parsed test_dump_with_string_replay_dir. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'Test that dump creates directory and writes context to JSON file.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = "Test that dump doesn't add extra .json suffix if template name already has it."
    var_1 = 'replay'
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = "Test that dump raises ValueError if context doesn't contain cookiecutter key."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'Test that dump correctly writes nested context data.'
    var_1 = 'replay'
    var_2 = 'nested_template'
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
    var_17 = 'nested_template.json'

def test_case_0():
    var_0 = 'Test that dump works with string path for replay_dir.'
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 4/7 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 3/6 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_with_path_object. Retrieved 4/7 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {"name": "test"}}'
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

def test_case_0():
    var_0 = 'config.json'
    var_1 = '{"cookiecutter": {"setting": "enabled"}}'
    var_2 = 'utf-8'
    var_3 = 'config'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #20
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



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_not_in_context. Retrieved 5/9 statements.


def test_case_0():
    var_0 = "Test that dump raises ValueError when 'cookiecutter' key is not in context."
    var_1 = 'test_template'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 3/6 statements.
# Partially parsed test_load_with_template_name_without_json_extension. Retrieved 4/7 statements.
# Partially parsed test_load_raises_error_when_cookiecutter_key_missing. Retrieved 4/8 statements.
# Partially parsed test_load_raises_error_when_file_not_found. Retrieved 1/3 statements.
# Partially parsed test_load_with_complex_cookiecutter_context. Retrieved 17/21 statements.


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

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = bool(False)
    assert var_1 is True

import json as module_0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'extra_key'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'options'
    var_6 = 'my_project'
    var_7 = 'John Doe'
    var_8 = 'opt1'
    var_9 = 'opt2'
    var_10 = [var_8, var_9]
    var_11 = {var_3: var_6, var_4: var_7, var_5: var_10}
    var_12 = 'extra_value'
    var_13 = {var_1: var_11, var_2: var_12}
    var_14 = {}
    var_15 = module_0.dumps(var_13, **var_14)
    var_16 = 'utf-8'
    var_17 = 'template'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 9/18 statements.


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
    var_10 = 'cookiecutter'



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/14 statements.
# Partially parsed test_load_without_json_extension. Retrieved 8/15 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/13 statements.
# Partially parsed test_load_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'Test load function with valid JSON file containing cookiecutter key.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test load function automatically adds .json extension.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'author'
    var_4 = 'test_author'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'template.json'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'template.json'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'Test load function raises FileNotFoundError when file does not exist.'
    var_1 = 'nonexistent.json'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'Test load function works with Path object as replay_dir.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'version'
    var_4 = '1.0'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_not_in_context. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 10/13 statements.
# Partially parsed test_load_without_json_extension. Retrieved 9/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.
# Partially parsed test_load_with_path_object. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'template.json'
    var_8 = '{"cookiecutter": {"project_name": "test_project", "author": "test_author"}}'
    var_9 = 'utf-8'
    var_10 = 'cookiecutter'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = '{"cookiecutter": {"project_name": "test_project"}}'
    var_7 = 'utf-8'
    var_8 = 'template'
    var_9 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"project_name": "test_project"}'
    var_2 = 'utf-8'
    var_3 = 'template.json'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = '{"cookiecutter": {"key": "value"}}'
    var_7 = 'utf-8'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 11/16 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key is present in context."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'my_project'
    var_7 = 'test_author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = f'{var_2}.json'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 8/11 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 7/11 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.
# Partially parsed test_load_with_complex_context. Retrieved 19/22 statements.
# Partially parsed test_load_with_path_object. Retrieved 9/12 statements.


import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = {}
    var_8 = module_0.dumps(var_5, **var_7)
    var_9 = 'utf-8'
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
    var_0 = 'template'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'
    var_5 = {}
    var_6 = module_0.dumps(var_3, **var_5)
    var_7 = 'utf-8'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = bool(False)
    assert var_1 is True

import json as module_0

def test_case_0():
    var_0 = 'complex'
    var_1 = 'cookiecutter'
    var_2 = 'extra_data'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'version'
    var_6 = 'nested'
    var_7 = 'my_project'
    var_8 = 'John Doe'
    var_9 = '1.0.0'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = {var_3: var_7, var_4: var_8, var_5: var_9, var_6: var_12}
    var_14 = 'additional'
    var_15 = {var_1: var_13, var_2: var_14}
    var_16 = f'{var_0}.json'
    var_17 = {}
    var_18 = module_0.dumps(var_15, **var_17)
    var_19 = 'utf-8'

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter'
    var_2 = 'test'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = {}
    var_8 = module_0.dumps(var_5, **var_7)
    var_9 = 'utf-8'



# Parsed testcases at query #33
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
    var_7 = bool(var_1)
    assert var_7 is True
    var_8 = 'cookiecutter'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_validates_cookiecutter_key_exists. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_template'
    var_7 = 'cookiecutter'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'replay.json'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'replay.json'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dump_writes_json_file_with_valid_context. Retrieved 8/16 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 5/9 statements.
# Partially parsed test_dump_creates_replay_directory_if_not_exists. Retrieved 7/12 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 8/13 statements.
# Partially parsed test_dump_preserves_context_structure. Retrieved 19/26 statements.


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
    var_1 = 'my_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'new_replay_dir'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'complex_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'nested'
    var_6 = 'my_project'
    var_7 = 'John Doe'
    var_8 = 'key'
    var_9 = 'list'
    var_10 = 'value'
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = {var_8: var_10, var_9: var_14}
    var_16 = {var_3: var_6, var_4: var_7, var_5: var_15}
    var_17 = {var_2: var_16}
    var_18 = 'complex_template.json'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'test.json'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'cookiecutter'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_template_name_without_json_extension. Retrieved 10/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/9 statements.
# Partially parsed test_load_with_path_object. Retrieved 9/12 statements.
# Partially parsed test_load_with_string_path. Retrieved 9/13 statements.
# Partially parsed test_load_with_nested_cookiecutter_data. Retrieved 13/16 statements.


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
    var_9 = 'cookiecutter'

def test_case_0():
    var_0 = "Test load function when template_name doesn't have .json extension."
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{"cookiecutter": {"project_name": "test_project"}}'
    var_8 = 'utf-8'
    var_9 = 'template'

def test_case_0():
    var_0 = 'Test load function raises ValueError when cookiecutter key is missing.'
    var_1 = 'template.json'
    var_2 = '{"other_key": "value"}'
    var_3 = 'utf-8'
    var_4 = 'template.json'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'Test load function with Path object as replay_dir.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{"cookiecutter": {"project_name": "test_project"}}'
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'Test load function with string path as replay_dir.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{"cookiecutter": {"project_name": "test_project"}}'
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'Test load function with nested data in cookiecutter.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'nested'
    var_5 = 'test'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = '{"cookiecutter": {"project_name": "test", "nested": {"key": "value"}}}'
    var_12 = 'utf-8'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/14 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 6/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.


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
    var_2 = 'version'
    var_3 = '1.0'
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
    var_0 = 'nonexistent_template'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 10/15 statements.


def test_case_0():
    var_0 = "Test that dump function accepts context with 'cookiecutter' key."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'project_slug'
    var_6 = 'test_project'
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = f'{var_2}.json'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 9/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_with_path_object. Retrieved 9/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.


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
    var_2 = '{"other_key": "value"}'
    var_3 = 'utf-8'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter'
    var_2 = 'data'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template.json'
    var_7 = '{"cookiecutter": {"data": "test"}}'
    var_8 = 'utf-8'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/15 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 6/12 statements.
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



# Parsed testcases at query #44
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



# Parsed testcases at query #45
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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_path_object. Retrieved 3/7 statements.
# Partially parsed test_get_file_name_with_string_path. Retrieved 4/5 statements.
# Partially parsed test_get_file_name_with_json_extension. Retrieved 3/4 statements.
# Partially parsed test_get_file_name_with_different_extension. Retrieved 4/5 statements.
# Partially parsed test_get_file_name_with_empty_directory. Retrieved 4/5 statements.
# Partially parsed test_get_file_name_with_complex_path. Retrieved 3/7 statements.


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

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template.txt'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'template.txt.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'template'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'template.json'

def test_case_0():
    var_0 = '/home/user/replays/2024'
    var_1 = [var_0]
    var_2 = 'game_replay'
    var_3 = 'game_replay.json'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_writes_json_file_with_context. Retrieved 8/16 statements.
# Partially parsed test_dump_creates_replay_directory_if_not_exists. Retrieved 9/15 statements.
# Partially parsed test_dump_adds_json_extension_if_not_present. Retrieved 8/13 statements.
# Partially parsed test_dump_does_not_duplicate_json_extension. Retrieved 9/16 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 5/9 statements.
# Partially parsed test_dump_writes_properly_formatted_json. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'Test that dump writes context to a JSON file.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = "Test that dump creates the replay directory if it doesn't exist."
    var_1 = 'new_replay_dir'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_template.json'

def test_case_0():
    var_0 = "Test that dump adds .json extension if template_name doesn't have it."
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = "Test that dump doesn't add .json if template_name already has it."
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = 'test_template.json.json'

def test_case_0():
    var_0 = "Test that dump raises ValueError if context doesn't have cookiecutter key."
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'Test that dump writes JSON with proper indentation.'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test'
    var_6 = {var_3: var_5, var_4: var_4}
    var_7 = {var_2: var_6}
    var_8 = '  '



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 8/16 statements.
# Partially parsed test_load_without_json_extension. Retrieved 7/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/12 statements.
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
    var_8 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'
    var_7 = 'cookiecutter'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dump_creates_replay_file. Retrieved 8/16 statements.
# Partially parsed test_dump_adds_json_extension_if_missing. Retrieved 8/14 statements.
# Partially parsed test_dump_does_not_add_extension_if_already_present. Retrieved 8/14 statements.
# Partially parsed test_dump_raises_error_when_cookiecutter_key_missing. Retrieved 5/9 statements.
# Partially parsed test_dump_creates_nested_directories. Retrieved 10/18 statements.
# Partially parsed test_dump_writes_valid_json. Retrieved 10/17 statements.


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
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
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
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'nested'
    var_1 = 'replay'
    var_2 = 'dir'
    var_3 = 'my_template'
    var_4 = 'cookiecutter'
    var_5 = 'project'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'version'
    var_5 = 'project'
    var_6 = '1.0'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'my_template.json'



# Parsed testcases at query #5
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
    var_7 = 'cookiecutter'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json_file. Retrieved 10/16 statements.
# Partially parsed test_dump_adds_json_extension_if_missing. Retrieved 9/13 statements.
# Partially parsed test_dump_preserves_json_extension. Retrieved 9/13 statements.
# Partially parsed test_dump_writes_correct_json_content. Retrieved 12/18 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_with_nested_replay_directory. Retrieved 11/17 statements.


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
    var_9 = 'utf-8'

def test_case_0():
    var_0 = "Test that dump adds .json extension if template_name doesn't have it."
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'my_template.json'

def test_case_0():
    var_0 = "Test that dump doesn't add double .json extension."
    var_1 = 'replay'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
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
    var_7 = 'John'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'my_template.json'
    var_11 = 'utf-8'

def test_case_0():
    var_0 = 'Test that dump raises ValueError when cookiecutter key is missing.'
    var_1 = 'replay'
    var_2 = 'my_template'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'cookiecutter'
    var_8 = bool('cookiecutter' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = 'Test that dump creates nested directories.'
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 11/20 statements.


def test_case_0():
    var_0 = "Test that dump function accepts context with 'cookiecutter' key."
    var_1 = 'replay'
    var_2 = 'test-template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = f'{var_2}.json'
    var_11 = 'cookiecutter'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 10/19 statements.


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
    var_10 = 'cookiecutter'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_requires_cookiecutter_key. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_template'
    var_7 = 'cookiecutter'



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 9/30 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = 'utf-8'
    var_10 = bool(var_1)
    assert var_10 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dump_creates_replay_directory. Retrieved 7/10 statements.
# Partially parsed test_dump_writes_json_file. Retrieved 8/12 statements.
# Partially parsed test_dump_writes_correct_json_content. Retrieved 10/15 statements.
# Partially parsed test_dump_with_json_extension_in_template_name. Retrieved 8/12 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 5/8 statements.
# Partially parsed test_dump_with_string_path. Retrieved 8/14 statements.
# Partially parsed test_dump_preserves_context_structure. Retrieved 16/21 statements.


def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

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
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'version'
    var_5 = 'test_project'
    var_6 = '1.0'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'my_template.json'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'extra_data'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'nested'
    var_7 = 'my_project'
    var_8 = 'John Doe'
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = {var_4: var_7, var_5: var_8, var_6: var_11}
    var_13 = 'something'
    var_14 = {var_2: var_12, var_3: var_13}
    var_15 = 'template.json'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'replay.json'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/17 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 6/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/15 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'my_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = f'{var_0}.json'
    var_9 = 'cookiecutter'

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'

def test_case_0():
    var_0 = 'nonexistent_template'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 8/14 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 7/13 statements.
# Partially parsed test_load_without_cookiecutter_key_raises_error. Retrieved 6/12 statements.
# Partially parsed test_load_with_path_object. Retrieved 8/14 statements.
# Partially parsed test_load_with_string_path. Retrieved 8/14 statements.


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
    var_7 = 'Context is required to contain a cookiecutter key'

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
    var_0 = 'Test load function works with string path as replay_dir.'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'template'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 3/6 statements.
# Partially parsed test_load_with_template_name_without_json_extension. Retrieved 4/7 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_with_path_object. Retrieved 3/6 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/3 statements.
# Partially parsed test_load_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_load_empty_cookiecutter. Retrieved 3/6 statements.


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

def test_case_0():
    var_0 = 'config.json'
    var_1 = '{"cookiecutter": {"name": "project"}}'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = 'invalid json content'
    var_2 = 'utf-8'
    var_3 = 'invalid.json'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"cookiecutter": {}}'
    var_2 = 'utf-8'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_load_json_file_with_json_extension. Retrieved 6/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_load_with_pathlib_path. Retrieved 7/14 statements.


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
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_raises_valueerror_when_cookiecutter_key_missing. Retrieved 5/27 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_with_valid_cookiecutter_context. Retrieved 9/19 statements.


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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 6/11 statements.
# Partially parsed test_load_with_template_name_without_extension. Retrieved 7/12 statements.
# Partially parsed test_load_with_path_object. Retrieved 6/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = 'cookiecutter'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = 'template'
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'template.json'
    var_4 = 'template.json'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #22
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
    var_0 = 'nonexistent_template'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test'
    var_7 = 'cookiecutter'



# Parsed testcases at query #24
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



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 9/18 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = 'utf-8'
    var_10 = 'cookiecutter'



# Parsed testcases at query #27
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
    var_7 = 'test_template'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'Context is required to contain a cookiecutter key'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_load_raises_error_when_cookiecutter_key_missing. Retrieved 7/16 statements.
# Partially parsed test_load_succeeds_when_cookiecutter_key_present. Retrieved 11/18 statements.


import json as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'utf-8'
    var_7 = 'template'
    var_8 = bool(False)
    assert var_8 is True

import json as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 'value'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = 'utf-8'
    var_11 = 'template'
    var_12 = 'cookiecutter'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 6/13 statements.
# Partially parsed test_load_with_template_name_without_extension. Retrieved 7/14 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/13 statements.
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
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'template.json'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'author'
    var_3 = 'test_author'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 11/17 statements.


def test_case_0():
    var_0 = "Test that dump succeeds when 'cookiecutter' key is present in context."
    var_1 = 'test-template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = 'Test Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.cookiecutters'
    var_10 = 'test-template.json'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 7/15 statements.
# Partially parsed test_load_with_json_extension_in_template_name. Retrieved 6/12 statements.
# Partially parsed test_load_without_cookiecutter_key. Retrieved 5/12 statements.
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
    var_6 = 'cookiecutter'

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
    var_2 = 'version'
    var_3 = '1.0'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template'



# Parsed testcases at query #32
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
    var_9 = 'cookiecutter'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_raises_error_when_cookiecutter_key_missing. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dump_writes_json_file. Retrieved 8/16 statements.
# Partially parsed test_dump_with_json_suffix_in_template_name. Retrieved 8/16 statements.
# Partially parsed test_dump_creates_nested_directories. Retrieved 10/18 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 5/9 statements.
# Partially parsed test_dump_with_string_path. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'

def test_case_0():
    var_0 = 'nested'
    var_1 = 'replay'
    var_2 = 'dir'
    var_3 = 'test_template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test'
    var_7 = 'cookiecutter'



# Parsed testcases at query #37
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



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_dump_with_cookiecutter_key_in_context. Retrieved 9/14 statements.


def test_case_0():
    var_0 = "Test that dump function succeeds when 'cookiecutter' key is in context."
    var_1 = 'replay'
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = f'{var_2}.json'



# Parsed testcases at query #40
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



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_load_context_without_cookiecutter_key. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'test.json'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_load_with_valid_context. Retrieved 9/15 statements.
# Partially parsed test_load_with_json_extension. Retrieved 6/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_with_path_object. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'template.json'
    var_8 = 'template'
    var_9 = 'cookiecutter'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'template.json'
    var_4 = 'template'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'value'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = 'template'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 8/11 statements.
# Partially parsed test_load_with_template_name_without_extension. Retrieved 9/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/8 statements.
# Partially parsed test_load_with_path_object. Retrieved 8/11 statements.
# Partially parsed test_load_with_complex_cookiecutter_structure. Retrieved 14/17 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = '{"cookiecutter": {"project_name": "test_project"}}'
    var_7 = 'utf-8'
    var_8 = 'cookiecutter'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = '{"cookiecutter": {"key": "value"}}'
    var_7 = 'utf-8'
    var_8 = 'template'

def test_case_0():
    var_0 = 'template.json'
    var_1 = '{"other_key": "value"}'
    var_2 = 'utf-8'
    var_3 = 'template.json'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'config.json'
    var_6 = '{"cookiecutter": {"name": "test"}}'
    var_7 = 'utf-8'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project'
    var_2 = 'author'
    var_3 = 'name'
    var_4 = 'version'
    var_5 = 'test'
    var_6 = '1.0'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'John'
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = {var_0: var_9}
    var_11 = 'template.json'
    var_12 = '{"cookiecutter": {"project": {"name": "test", "version": "1.0"}, "author": "John"}}'
    var_13 = 'utf-8'



