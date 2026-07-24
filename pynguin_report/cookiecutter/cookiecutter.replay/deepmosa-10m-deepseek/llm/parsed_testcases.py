####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_path_and_no_json_suffix. Retrieved 3/6 statements.
# Partially parsed test_get_file_name_with_path_and_json_suffix. Retrieved 3/6 statements.
# Partially parsed test_get_file_name_with_str_and_no_json_suffix. Retrieved 4/5 statements.
# Partially parsed test_get_file_name_with_str_and_json_suffix. Retrieved 4/5 statements.
# Partially parsed test_get_file_name_with_empty_template_name. Retrieved 3/6 statements.
# Partially parsed test_get_file_name_with_template_name_already_ending_with_json. Retrieved 3/6 statements.
# Partially parsed test_get_file_name_with_template_name_ending_with_dot_json_but_extra. Retrieved 3/6 statements.


def test_case_0():
    var_0 = '/some/dir'
    var_1 = [var_0]
    var_2 = 'template'
    var_3 = 'template.json'

def test_case_0():
    var_0 = '/some/dir'
    var_1 = [var_0]
    var_2 = 'template.json'
    var_3 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/some/dir'
    var_1 = 'template'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/some/dir'
    var_1 = 'template.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'template.json'

def test_case_0():
    var_0 = '/some/dir'
    var_1 = [var_0]
    var_2 = ''
    var_3 = '.json'

def test_case_0():
    var_0 = '/some/dir'
    var_1 = [var_0]
    var_2 = 'file.json'
    var_3 = 'file.json'

def test_case_0():
    var_0 = '/some/dir'
    var_1 = [var_0]
    var_2 = 'file.json.txt'
    var_3 = 'file.json.txt.json'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_success. Retrieved 11/14 statements.
# Partially parsed test_load_with_json_extension. Retrieved 11/14 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/11 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template.json'
    var_2 = '/tmp/test/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_creates_directory_and_file. Retrieved 9/16 statements.
# Partially parsed test_dump_handles_existing_json_suffix. Retrieved 9/15 statements.
# Partially parsed test_dump_creates_nested_directory. Retrieved 12/19 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nested/test/replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'template.json'
    var_9 = 'nested/test/replay'
    var_10 = 'nested/test'
    var_11 = 'nested'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/12 statements.


import json as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.dumps(var_0, **var_1)
    var_3 = 'fake_file.json'
    var_4 = [var_3]
    var_5 = 'template'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/21 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'not_cookiecutter'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'not_cookiecutter'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'fake_dir'
    var_8 = 'fake_template'
    var_9 = module_1.load(var_7, var_8)
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_9)
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = lambda x: var_4
    var_6 = 'fake_dir'
    var_7 = 'fake_template'
    var_8 = module_0.load(var_6, var_7)
    var_9 = 'cookiecutter'
    var_10 = bool('cookiecutter' in var_8)
    assert var_10 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_successful. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension. Retrieved 10/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 8/12 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_context_missing_cookiecutter_key. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #12
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #13
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'dummy_dir'
    var_8 = 'dummy_template'
    var_9 = module_1.load(var_7, var_8)
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_9)
    assert var_11 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #16
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'dummy_dir'
    var_8 = 'dummy_template'
    var_9 = module_1.load(var_7, var_8)
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_9)
    assert var_11 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_contains_cookiecutter_key. Retrieved 9/19 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'dummy_dir'
    var_8 = 'dummy_template'
    var_9 = module_1.load(var_7, var_8)
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_9)
    assert var_11 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'not_cookiecutter'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = lambda x: var_4
    var_6 = 'dummy_dir'
    var_7 = 'dummy_template'
    var_8 = module_0.load(var_6, var_7)
    var_9 = 'cookiecutter'
    var_10 = bool('cookiecutter' in var_8)
    assert var_10 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_success. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension. Retrieved 10/13 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/12 statements.
# Partially parsed test_load_json_decode_error. Retrieved 5/9 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template.json'
    var_2 = '/tmp/test/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_success. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension. Retrieved 10/13 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/12 statements.
# Partially parsed test_load_invalid_json. Retrieved 5/9 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template.json'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #26
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '{"cookiecutter": {"project_name": "test"}}'
    var_1 = 'dummy_dir'
    var_2 = 'dummy_template'
    var_3 = module_0.load(var_1, var_2)
    var_4 = 'cookiecutter'
    var_5 = bool('cookiecutter' in var_3)
    assert var_5 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 4/13 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = '{"some_key": "some_value"}'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_success. Retrieved 11/14 statements.
# Partially parsed test_load_with_json_extension. Retrieved 11/14 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/11 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_file. Retrieved 9/16 statements.
# Partially parsed test_dump_handles_existing_json_suffix. Retrieved 9/15 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'my_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'my_template.json'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_context_contains_cookiecutter_key. Retrieved 9/16 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = 'test_template'
    var_10 = module_1.load(var_1, var_9)
    var_11 = 'cookiecutter'
    var_12 = bool('cookiecutter' in var_10)
    assert var_12 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_success. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension. Retrieved 10/13 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/12 statements.
# Partially parsed test_load_json_decode_error. Retrieved 5/9 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template.json'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/11 statements.


import json as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.dumps(var_0, **var_1)
    var_3 = 'fake_dir'
    var_4 = [var_3]
    var_5 = 'fake_template'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 7/14 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = 'fake_dir'
    var_6 = [var_5]
    var_7 = 'fake_template'
    var_8 = module_1.load(var_1, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_cookiecutter_key_missing_raises_value_error. Retrieved 7/13 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = 'fake_dir'
    var_6 = [var_5]
    var_7 = 'fake_template'
    var_8 = module_1.load(var_1, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/21 statements.
# Partially parsed test_dump_uses_existing_directory. Retrieved 7/19 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 4/11 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 6/17 statements.
# Partially parsed test_dump_writes_indented_json. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 'new_subdir'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'

def test_case_0():
    var_0 = 'my_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'

def test_case_0():
    var_0 = 'my_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'my_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'my_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = '{'
    var_8 = '\n  '



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_context_missing_cookiecutter_key. Retrieved 7/16 statements.


import json as module_0

def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_load_contains_cookiecutter_key. Retrieved 9/19 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'fake_dir'
    var_8 = 'fake_template'
    var_9 = module_1.load(var_7, var_8)
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_9)
    assert var_11 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_file. Retrieved 7/19 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 4/11 statements.
# Partially parsed test_dump_handles_existing_json_extension. Retrieved 6/17 statements.
# Partially parsed test_dump_creates_nested_directories. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'my_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'Test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'

def test_case_0():
    var_0 = 'my_template'
    var_1 = 'project_name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'my_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'Test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'nested'
    var_1 = 'deep'
    var_2 = 'template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = f'{var_2}.json'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_load_returns_context_with_cookiecutter_key. Retrieved 9/17 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = 'test_template'
    var_10 = module_1.load(var_1, var_9)
    var_11 = 'cookiecutter'
    var_12 = bool('cookiecutter' in var_10)
    assert var_12 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_load_success. Retrieved 11/14 statements.
# Partially parsed test_load_with_json_extension. Retrieved 11/14 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/11 statements.
# Partially parsed test_load_json_decode_error. Retrieved 5/8 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_path_and_no_json_suffix. Retrieved 2/4 statements.
# Partially parsed test_get_file_name_with_json_suffix. Retrieved 2/4 statements.
# Partially parsed test_get_file_name_empty_template. Retrieved 2/4 statements.


def test_case_0():
    var_0 = '/some/dir'
    var_1 = [var_0]
    var_2 = 'template'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/some/dir'
    var_1 = 'template'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/some/dir/template.json'

def test_case_0():
    var_0 = '/another/dir'
    var_1 = [var_0]
    var_2 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path'
    var_1 = 'my.template.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/path/my.template.json'

def test_case_0():
    var_0 = '/empty'
    var_1 = [var_0]
    var_2 = ''



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/12 statements.
# Partially parsed test_dump_writes_correct_json_file. Retrieved 9/17 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 9/15 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = [var_0]
    var_9 = [var_0]

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = [var_0]
    var_9 = 'test_template.json'
    var_10 = [var_0]

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = [var_0]
    var_9 = 'test_template.json'
    var_10 = [var_0]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'not_cookiecutter'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_success. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension. Retrieved 10/13 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/12 statements.
# Partially parsed test_load_json_decode_error. Retrieved 5/9 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template.json'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_success. Retrieved 11/14 statements.
# Partially parsed test_load_with_json_extension. Retrieved 11/14 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/11 statements.
# Partially parsed test_load_path_object. Retrieved 10/15 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import json as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'template'
    var_3 = '/tmp/template.json'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {}
    var_10 = module_0.dumps(var_8, **var_9)
    var_11 = 'utf-8'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'some_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_success. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension. Retrieved 10/13 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/12 statements.
# Partially parsed test_load_json_decode_error. Retrieved 5/9 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/14 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_returns_context_with_cookiecutter_key. Retrieved 9/15 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = 'test_template'
    var_10 = module_1.load(var_1, var_9)
    var_11 = 'cookiecutter'
    var_12 = bool('cookiecutter' in var_10)
    assert var_12 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_success. Retrieved 6/10 statements.
# Partially parsed test_load_with_json_suffix. Retrieved 6/10 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 5/9 statements.
# Partially parsed test_load_json_decode_error. Retrieved 5/10 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = '/tmp/test_template.json'
    var_3 = '{"cookiecutter": {"key": "value"}}'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(var_4 == {'cookiecutter': {'key': 'value'}})
    assert var_5 is True
    var_6 = 'utf-8'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template.json'
    var_2 = '/tmp/test_template.json'
    var_3 = '{"cookiecutter": {"key": "value"}}'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(var_4 == {'cookiecutter': {'key': 'value'}})
    assert var_5 is True
    var_6 = 'utf-8'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = '/tmp/test_template.json'
    var_3 = '{"other_key": "value"}'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = '/tmp/test_template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = '/tmp/test_template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_success. Retrieved 11/14 statements.
# Partially parsed test_load_with_json_extension. Retrieved 11/14 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/11 statements.
# Partially parsed test_load_invalid_json. Retrieved 5/8 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template.json'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = '/tmp/replay/template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'not_cookiecutter'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_success. Retrieved 9/11 statements.
# Partially parsed test_load_with_json_extension. Retrieved 9/11 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 7/10 statements.
# Partially parsed test_load_json_decode_error. Retrieved 4/7 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = 'invalid json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/13 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'some_value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/18 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/21 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 4/11 statements.
# Partially parsed test_dump_writes_correct_json_content. Retrieved 9/19 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'

def test_case_0():
    var_0 = 'my_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter'
    var_2 = 'project'
    var_3 = 'version'
    var_4 = 'test'
    var_5 = '1.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = f'{var_0}.json'

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = 'data'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 7/13 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = 'fake_dir'
    var_6 = [var_5]
    var_7 = 'fake_template'
    var_8 = module_1.load(var_1, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_success. Retrieved 9/11 statements.
# Partially parsed test_load_with_json_extension. Retrieved 9/11 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 7/10 statements.
# Partially parsed test_load_json_decode_error. Retrieved 4/7 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'invalid json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/12 statements.
# Partially parsed test_dump_writes_correct_json_file. Retrieved 9/17 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 9/15 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = [var_0]
    var_9 = [var_0]

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = [var_0]
    var_9 = 'template.json'
    var_10 = [var_0]

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = [var_0]
    var_9 = 'template.json'
    var_10 = [var_0]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_success. Retrieved 11/14 statements.
# Partially parsed test_load_with_json_extension. Retrieved 11/14 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/11 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template.json'
    var_2 = '/tmp/test/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #33
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'fake_dir'
    var_8 = 'fake_template'
    var_9 = module_1.load(var_7, var_8)
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_9)
    assert var_11 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #37
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = lambda x: var_4
    var_6 = 'dummy_dir'
    var_7 = 'dummy_template'
    var_8 = module_0.load(var_6, var_7)
    var_9 = 'cookiecutter'
    var_10 = bool('cookiecutter' in var_8)
    assert var_10 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/13 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_load_success. Retrieved 11/14 statements.
# Partially parsed test_load_with_json_extension. Retrieved 11/14 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 7/10 statements.
# Partially parsed test_load_json_decode_error. Retrieved 4/7 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = '/tmp/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'Test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template.json'
    var_2 = '/tmp/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'invalid_template'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'nonexistent'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'corrupted'
    var_2 = 'invalid json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_dump_creates_directory. Retrieved 8/10 statements.
# Partially parsed test_dump_writes_correct_file. Retrieved 9/12 statements.
# Partially parsed test_dump_handles_existing_json_extension. Retrieved 8/11 statements.
# Partially parsed test_dump_writes_correct_json_content. Retrieved 9/13 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = [var_0]

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay2'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = [var_0]
    var_9 = f'{var_1}.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay3'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = [var_0]

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay4'
    var_1 = 'my_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'cookiecutter'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay5'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = [var_0]
    var_9 = f'{var_1}.json'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_dump_creates_file_with_cookiecutter_key. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #42
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_file. Retrieved 9/13 statements.
# Partially parsed test_dump_handles_existing_json_extension. Retrieved 9/12 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = '/tmp/test_replay/test_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = '/tmp/test_replay/test_template.json'



# Parsed testcases at query #44
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_dump_creates_file_with_cookiecutter_key. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/10 statements.
# Partially parsed test_dump_writes_correct_json_file. Retrieved 9/14 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 9/13 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'test_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'test_template.json'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_file. Retrieved 9/16 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 6/8 statements.
# Partially parsed test_dump_handles_existing_json_extension. Retrieved 9/15 statements.
# Partially parsed test_dump_uses_existing_directory. Retrieved 10/17 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'my_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'my_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = True
    var_2 = 'my_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.dump(var_0, var_2, var_7)
    var_9 = 'my_template.json'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_dump_creates_file_with_cookiecutter_key. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_dump_creates_file_with_cookiecutter_key. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_dump_creates_file_with_utf8_encoding. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #51
#--------------------------






