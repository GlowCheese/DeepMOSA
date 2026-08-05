####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test the full flow of generate_files.'
    var_1 = 'repo_dir'
    var_2 = 'output_dir'
    var_3 = 'context'
    var_4 = 'template_dir'
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'hello.txt'
    var_8 = 'subdir'
    var_9 = 'info.txt'
    var_10 = 'Info for test_template_name'
    var_11 = 'test_template_name'
    var_12 = 'test_project'

def test_case_0():
    var_0 = 'Test that the project directory is deleted on failure if configured.'
    var_1 = 'output_dir'
    var_2 = True
    var_3 = 'repo_dir'
    var_4 = 'context'
    var_5 = False



# Parsed testcases at query #2
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new'
    var_6 = 2
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = 'choices'
    var_10 = 'a'
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_10, var_11, var_12]
    var_14 = {var_9: var_13}
    var_15 = [var_10, var_12]
    var_16 = {var_9: var_15}
    var_17 = module_0.apply_overwrites_to_context(var_14, var_16)
    var_18 = [var_10, var_11, var_12]
    var_19 = {var_9: var_18}
    var_20 = 'z'
    var_21 = [var_10, var_20]
    var_22 = {var_9: var_21}
    var_23 = module_0.apply_overwrites_to_context(var_19, var_22)
    var_24 = 'choice'
    var_25 = [var_10, var_11, var_12]
    var_26 = {var_24: var_25}
    var_27 = {var_24: var_11}
    var_28 = module_0.apply_overwrites_to_context(var_26, var_27)
    var_29 = [var_10, var_11]
    var_30 = {var_24: var_29}
    var_31 = {var_24: var_20}
    var_32 = module_0.apply_overwrites_to_context(var_30, var_31)
    var_33 = 'settings'
    var_34 = 'theme'
    var_35 = 'font'
    var_36 = 'light'
    var_37 = 'serif'
    var_38 = {var_34: var_36, var_35: var_37}
    var_39 = {var_33: var_38}
    var_40 = 'dark'
    var_41 = {var_34: var_40}
    var_42 = {var_33: var_41}
    var_43 = False
    var_44 = module_0.apply_overwrites_to_context(var_39, var_42, in_dictionary_variable=var_43)
    var_45 = 'enabled'
    var_46 = {var_45: var_43}
    var_47 = 'yes'
    var_48 = {var_45: var_47}
    var_49 = module_0.apply_overwrites_to_context(var_46, var_48)
    var_50 = {var_45: var_43}
    var_51 = 'not-a-boolean-string'
    var_52 = {var_45: var_51}
    var_53 = module_0.apply_overwrites_to_context(var_50, var_52)
    var_54 = 'meta'
    var_55 = 'id'
    var_56 = {var_55: var_3}
    var_57 = {var_54: var_56}
    var_58 = 'new_key'
    var_59 = 'val'
    var_60 = {var_58: var_59}
    var_61 = {var_54: var_60}
    var_62 = True
    var_63 = module_0.apply_overwrites_to_context(var_57, var_61, in_dictionary_variable=var_62)
    var_64 = 'existing'
    var_65 = True
    var_66 = {var_64: var_65}
    var_67 = 'new_var'
    var_68 = 'ignored'
    var_69 = {var_67: var_68}
    var_70 = module_0.apply_overwrites_to_context(var_66, var_69, in_dictionary_variable=var_43)
    var_71 = 'list_var'
    var_72 = [var_10, var_11]
    var_73 = {var_71: var_72}
    var_74 = 'x'
    var_75 = 'y'
    var_76 = [var_74, var_75, var_20]
    var_77 = {var_71: var_76}
    var_78 = module_0.apply_overwrites_to_context(var_73, var_77)



# Parsed testcases at query #3
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'active'
    var_3 = 'old'
    var_4 = 1
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'new'
    var_8 = 2
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = 'existing'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = 'new_var'
    var_15 = 'ignored'
    var_16 = {var_14: var_15}
    var_17 = module_0.apply_overwrites_to_context(var_13, var_16)
    var_18 = {var_11: var_12}
    var_19 = 'added'
    var_20 = {var_14: var_19}
    var_21 = True
    var_22 = module_0.apply_overwrites_to_context(var_18, var_20, in_dictionary_variable=var_21)
    var_23 = 'choices'
    var_24 = 'a'
    var_25 = 'b'
    var_26 = 'c'
    var_27 = [var_24, var_25, var_26]
    var_28 = {var_23: var_27}
    var_29 = [var_24, var_26]
    var_30 = {var_23: var_29}
    var_31 = module_0.apply_overwrites_to_context(var_28, var_30)
    var_32 = [var_24, var_25, var_26]
    var_33 = {var_23: var_32}
    var_34 = 'd'
    var_35 = [var_34]
    var_36 = {var_23: var_35}
    var_37 = module_0.apply_overwrites_to_context(var_33, var_36)
    var_38 = 'choice'
    var_39 = [var_24, var_25, var_26]
    var_40 = {var_38: var_39}
    var_41 = {var_38: var_26}
    var_42 = [var_24, var_25]
    var_43 = {var_38: var_42}
    var_44 = 'z'
    var_45 = {var_38: var_44}
    var_46 = module_0.apply_overwrites_to_context(var_43, var_45)
    var_47 = 'config'
    var_48 = 'user'
    var_49 = 'port'
    var_50 = 'admin'
    var_51 = 80
    var_52 = {var_48: var_50, var_49: var_51}
    var_53 = {var_47: var_52}
    var_54 = 'debug'
    var_55 = 443
    var_56 = True
    var_57 = {var_49: var_55, var_54: var_56}
    var_58 = {var_47: var_57}
    var_59 = module_0.apply_overwrites_to_context(var_53, var_58)
    var_60 = 'flag'
    var_61 = False
    var_62 = {var_60: var_61}
    var_63 = 'yes'
    var_64 = {var_60: var_63}
    var_65 = module_0.apply_overwrites_to_context(var_62, var_64)
    var_66 = {var_60: var_61}
    var_67 = 'not-a-boolean'
    var_68 = {var_60: var_67}
    var_69 = module_0.apply_overwrites_to_context(var_66, var_68)
    var_70 = 'nested'
    var_71 = 'items'
    var_72 = [var_24, var_25]
    var_73 = {var_71: var_72}
    var_74 = {var_70: var_73}
    var_75 = [var_7]
    var_76 = {var_71: var_75}
    var_77 = {var_70: var_76}
    var_78 = True
    var_79 = module_0.apply_overwrites_to_context(var_74, var_77, in_dictionary_variable=var_78)



# Parsed testcases at query #4
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'output'
    var_6 = '{{ project_name }}_dir'
    var_7 = False
    var_8 = ''
    var_9 = '{{ project_name }}_dir'
    var_10 = False
    var_11 = 'overwrite_test'
    var_12 = 'old.txt'
    var_13 = 'old'
    var_14 = 'overwrite_test_dir'
    var_15 = '{{ project_name }}_overwrite'
    var_16 = 'my_project_overwrite'
    var_17 = True



# Parsed testcases at query #5
#--------------------------


import json as module_0
import jinja2.loaders as module_1
import jinja2.environment as module_2

def test_case_0():
    var_0 = '\n    Test the generate_files function by setting up a dummy template \n    structure and verifying the output.\n    '
    var_1 = 'template'
    var_2 = '{{ project_name }}'
    var_3 = 'config.txt'
    var_4 = 'Hello {{ user_name }}!'
    var_5 = 'utf-8'
    var_6 = 'static_assets'
    var_7 = 'data.bin'
    var_8 = b'\x00\x01\x02\x03'
    assert var_8 == 'Hello tester!'
    var_9 = 'cookiecutter.json'
    var_10 = 'project_name'
    var_11 = 'user_name'
    var_12 = 'cookiecutter'
    assert var_12 == b'\x00\x01\x02\x03'
    var_13 = 'my_project'
    var_14 = 'tester'
    var_15 = '_copy_without_render'
    var_16 = '_new_lines'
    var_17 = 'static_assets/*'
    var_18 = [var_17]
    var_19 = '\n'
    var_20 = {var_15: var_18, var_16: var_19}
    var_21 = {var_10: var_13, var_11: var_14, var_12: var_20}
    var_22 = module_0.dumps(var_21)
    var_23 = 'cookiecatcher'
    var_24 = [var_17]
    var_25 = {var_15: var_24, var_16: var_19}
    var_26 = {var_10: var_13, var_11: var_14, var_23: var_25}
    var_27 = module_1.FileSystemLoader(var_0)
    var_28 = module_2.Environment(loader=var_27)
    var_29 = 'output'
    var_30 = False
    var_31 = 'config.txt'
    var_32 = 'utf-8'
    var_33 = 'static_assets'
    var_34 = 'data.bin'
    var_35 = var_9 / var_34



# Parsed testcases at query #6
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*.md'
    var_4 = 'binary_file'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'test.txt'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True
    var_10 = 'docs/readme.md'
    var_11 = module_0.is_copy_only_path(var_10, var_7)
    assert var_11 is True
    var_12 = module_0.is_copy_only_path(var_4, var_7)
    assert var_12 is True
    var_13 = 'script.py'
    var_14 = module_0.is_copy_only_path(var_13, var_7)
    assert var_14 is False
    var_15 = 'docs/config.json'
    var_16 = module_0.is_copy_only_path(var_15, var_7)
    assert var_16 is False
    var_17 = {}
    var_18 = module_0.is_copy_only_path(var_8, var_17)
    assert var_18 is False
    var_19 = {}
    var_20 = {var_0: var_19}
    var_21 = module_0.is_copy_only_path(var_8, var_20)
    assert var_21 is False
    var_22 = []
    var_23 = {var_1: var_22}
    var_24 = {var_0: var_23}
    var_25 = module_0.is_copy_only_path(var_8, var_24)
    assert var_25 is False



# Parsed testcases at query #7
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test the full generation flow of generate_files.'
    var_1 = 'template_dir'
    var_2 = 'output_dir'
    var_3 = 'context'
    var_4 = 'root_name'
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = 'README.md'
    var_8 = 'utf-8'
    var_9 = 'data.bin'
    var_10 = 'src'
    var_11 = 'main.py'
    var_12 = 'mock'

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test that UndefinedVariableInTemplate is raised when a template variable is missing.'
    var_1 = 'template_dir'
    var_2 = 'output_dir'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'root_name'
    var_7 = module_0.Environment()
    var_8 = module_1.generate_files(var_0, var_5, var_1)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Tests the generate_file function for rendering, copying, and skipping.'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = 'World'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {}
    var_9 = 'template_dir'
    var_10 = 'project_dir'
    var_11 = 'text_file'
    var_12 = 'hello_World.txt'
    var_13 = 'utf-8'
    var_14 = 'binary_file'
    var_15 = 'data.bin'
    var_16 = 'collision.txt'
    var_17 = 'I am an old file'
    var_18 = 'collision_{{ suffix }}.txt'
    var_19 = 'New content'
    var_20 = 'suffix'
    var_21 = ''
    var_22 = True
    var_23 = 'newline_file'
    var_24 = 'newline.txt'

def test_case_0():
    var_0 = 'Test that an empty template result for a filename returns early.'
    var_1 = 'template_dir'
    var_2 = 'name'
    var_3 = ''
    var_4 = {var_2: var_3}
    var_5 = 'project_dir'
    var_6 = 'empty_test'
    var_7 = 'text_file'
    var_8 = {var_2: var_6}



# Parsed testcases at query #9
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Tests the render_and_create_dir function for various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_awesome_project'
    var_4 = {var_2: var_3}
    var_5 = 'output'
    var_6 = '{{ project_name }}_dir'
    var_7 = False
    var_8 = 'my_awesome_project_dir'
    var_9 = False
    var_10 = 'another_dir'
    var_11 = True
    var_12 = ''
    var_13 = '{{ non_existent_var }}'



# Parsed testcases at query #10
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'old_name'
    var_3 = '1.0.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_name'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = 'config'
    var_9 = 'key1'
    var_10 = 'key2'
    var_11 = 'val1'
    var_12 = 'val2'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = {var_8: var_13}
    var_15 = 'key3'
    var_16 = 'new_val2'
    var_17 = 'val3'
    var_18 = {var_10: var_16, var_15: var_17}
    var_19 = {var_8: var_18}
    var_20 = True
    var_21 = module_0.apply_overwrites_to_context(var_14, var_19, in_dictionary_variable=var_20)
    var_22 = 'features'
    var_23 = 'auth'
    var_24 = 'logging'
    var_25 = 'database'
    var_26 = [var_23, var_24, var_25]
    var_27 = {var_22: var_26}
    var_28 = [var_23, var_25]
    var_29 = {var_22: var_28}
    var_30 = module_0.apply_overwrites_to_context(var_27, var_29)
    var_31 = [var_23, var_24]
    var_32 = {var_22: var_31}
    var_33 = 'invalid_feature'
    var_34 = [var_23, var_33]
    var_35 = {var_22: var_34}
    var_36 = module_0.apply_overwrites_to_context(var_32, var_35)
    var_37 = 'theme'
    var_38 = 'light'
    var_39 = 'dark'
    var_40 = 'high-contrast'
    var_41 = [var_38, var_39, var_40]
    var_42 = {var_37: var_41}
    var_43 = {var_37: var_39}
    var_44 = module_0.apply_overwrites_to_context(var_42, var_43)
    var_45 = 'debug'
    var_46 = False
    var_47 = {var_45: var_46}
    var_48 = 'yes'
    var_49 = {var_45: var_48}
    var_50 = module_0.apply_overwrites_to_context(var_47, var_49)
    var_51 = {var_45: var_46}
    var_52 = 'not-a-boolean'
    var_53 = {var_45: var_52}
    var_54 = module_0.apply_overwrites_to_context(var_51, var_53)
    var_55 = 'existing'
    var_56 = 'val'
    var_57 = {var_55: var_56}
    var_58 = 'new_var'
    var_59 = 'ignored'
    var_60 = {var_58: var_59}
    var_61 = module_0.apply_overwrites_to_context(var_57, var_60, in_dictionary_variable=var_46)
    var_62 = 'parent'
    var_63 = {}
    var_64 = {var_62: var_63}
    var_65 = 'child'
    var_66 = 'value'
    var_67 = {var_65: var_66}
    var_68 = {var_62: var_67}
    var_69 = module_0.apply_overwrites_to_context(var_64, var_68, in_dictionary_variable=var_20)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the full generation flow: \n    1. Template finding\n    2. Directory creation via rendering\n    3. File rendering\n    4. Copying without rendering\n    5. Hook execution\n    '
    var_1 = 'template_root'
    var_2 = 'repo_dir'
    var_3 = 'context'
    var_4 = 'output_dir'
    var_5 = True
    var_6 = 'src'
    var_7 = 'test_user_module.py'
    var_8 = 'static'
    var_9 = 'README.txt'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Tests that UndefinedVariableInTemplate is raised when a template variable is missing.'
    var_1 = 'template_root'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'repo_dir'
    var_6 = 'output_dir'
    var_7 = module_0.generate_files(var_1, var_4, var_2)



# Parsed testcases at query #12
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_awesome_project'
    var_4 = {var_2: var_3}
    var_5 = 'output'
    var_6 = '{{ project_name }}_dir'
    var_7 = False
    var_8 = ''
    var_9 = 'my_awesome_project_dir'
    var_10 = False
    var_11 = '{{ project_name }}_new'
    var_12 = 'my_awesome_project_dir'
    var_13 = 'sub'
    var_14 = '{{ project_name }}_dir/sub'
    var_15 = True
    var_16 = 'Undefined Variable'
    var_17 = 'variable not found'
    var_18 = '{{ missing_var }}'



# Parsed testcases at query #13
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = '\n    Test the generate_files function by creating a dummy template structure,\n    mocking external dependencies (find_template, run_hook_from_repo_dir),\n    and verifying the file generation output.\n    '
    var_1 = 'my_template'
    var_2 = '{{project_name}}'
    var_3 = 'config.txt'
    var_4 = 'Hello {{user}}!'
    var_5 = 'static_assets'
    var_6 = 'data.bin'
    var_7 = 'binary content'
    assert var_7 == 'Hello tester!'
    var_8 = 'cookiecutter.json'
    var_9 = 'cookiecutter'
    var_10 = '_copy_without_render'
    var_11 = '_new_lines'
    assert var_11 == 'binary content'
    var_12 = 'static_assets/*'
    var_13 = [var_12]
    var_14 = '\n'
    var_15 = {var_10: var_13, var_11: var_14}
    var_16 = 'output'
    var_17 = 'project_name'
    var_18 = 'user'
    var_19 = 'my_project'
    var_20 = 'tester'
    var_21 = [var_12]
    var_22 = {var_10: var_21, var_11: var_14}
    var_23 = {var_17: var_19, var_18: var_20, var_9: var_22}
    var_24 = [var_0]
    var_25 = module_0.FileSystemLoader(var_24)
    var_26 = module_1.Environment(loader=var_25)
    var_27 = True
    var_28 = 'config.txt'
    var_29 = 'static_assets'
    var_30 = 'data.bin'
    var_31 = var_8 / var_30

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'Test that generate_files raises UndefinedVariableInTemplate when a variable is missing.'
    var_1 = 'error_template'
    var_2 = 'error_{{missing_var}}.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = [var_0]
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'out'
    var_11 = str(var_8)
    var_12 = module_2.generate_files(var_0, var_6, var_11)



# Parsed testcases at query #14
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_awesome_project'
    var_4 = {var_2: var_3}
    var_5 = 'output'
    var_6 = '{{ project_name }}_dir'
    var_7 = False
    var_8 = ''
    var_9 = 'already_exists'
    var_10 = 'already_exists'
    var_11 = False
    var_12 = 'already_exists'
    var_13 = False
    var_14 = 'overwrite_test'
    var_15 = True
    var_16 = 'version'
    var_17 = 'complex'
    var_18 = '1.0'
    var_19 = {var_13: var_17, var_16: var_18}
    var_20 = '{{ project_name }}_v{{ version }}'
    var_21 = {}
    var_22 = '{{ missing_var }}'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = '\n    Test generate_file with various scenarios: \n    1. Successful rendering of a text file.\n    2. Skipping creation if file exists.\n    3. Handling binary files (via mocking/simulating).\n    4. Handling directory as target (should return early).\n    '
    var_1 = 'template'
    var_2 = 'project'
    var_3 = 'hello_{{ name }}.txt'
    var_4 = 'Hello, {{ name }}!'
    var_5 = 'utf-8'
    var_6 = 'already_here.txt'
    var_7 = 'I should not be overwritten'
    var_8 = 'utf-arg'
    var_9 = 'name'
    var_10 = 'cookiecutter'
    var_11 = 'World'
    var_12 = '_new_lines'
    var_13 = '\n'
    var_14 = {var_12: var_13}
    var_15 = {var_9: var_11, var_10: var_14}
    var_16 = False
    var_17 = 'hello_World.txt'
    var_18 = 'Original Content'
    var_19 = True
    var_20 = 'empty_dir'
    var_21 = 'data.bin'
    var_22 = b'\x00\x01\x02\x03'
    var_23 = False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'old_name'
    var_3 = '1.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_name'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = {var_0: var_2}
    var_9 = 'new_var'
    var_10 = 'surprise'
    var_11 = {var_9: var_10}
    var_12 = False
    var_13 = module_0.apply_overwrites_to_context(var_8, var_11, in_dictionary_variable=var_12)
    var_14 = 'settings'
    var_15 = 'theme'
    var_16 = 'dark'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = 'font'
    var_20 = 'roboto'
    var_21 = {var_19: var_20}
    var_22 = {var_14: var_21}
    var_23 = True
    var_24 = 'choices'
    var_25 = 'a'
    var_26 = 'b'
    var_27 = 'c'
    var_28 = [var_25, var_26, var_27]
    var_29 = {var_24: var_28}
    var_30 = [var_25, var_27]
    var_31 = {var_24: var_30}
    var_32 = module_0.apply_overwrites_to_context(var_29, var_31)
    var_33 = [var_25, var_26, var_27]
    var_34 = {var_24: var_33}
    var_35 = 'z'
    var_36 = [var_25, var_35]
    var_37 = {var_24: var_36}
    var_38 = module_0.apply_overwrites_to_context(var_34, var_37)
    var_39 = 'choice'
    var_40 = [var_25, var_26, var_27]
    var_41 = {var_39: var_40}
    var_42 = {var_39: var_26}
    var_43 = module_0.apply_overwrites_to_context(var_41, var_42)
    var_44 = [var_25, var_26, var_27]
    var_45 = {var_39: var_44}
    var_46 = {var_39: var_35}
    var_47 = module_0.apply_overwrites_to_context(var_45, var_46)
    var_48 = 'enabled'
    var_49 = {var_48: var_12}
    var_50 = 'yes'
    var_51 = {var_48: var_50}
    var_52 = module_0.apply_overwrites_to_context(var_49, var_51)
    var_53 = {var_48: var_12}
    var_54 = 'not-a-boolean'
    var_55 = {var_48: var_54}
    var_56 = module_0.apply_overwrites_to_context(var_53, var_55)
    var_57 = 'meta'
    var_58 = 'author'
    var_59 = 'tags'
    var_60 = 'name'
    var_61 = 'email'
    var_62 = 'old'
    var_63 = 'old@test.com'
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = 'dev'
    var_66 = [var_65]
    var_67 = {var_58: var_64, var_59: var_66}
    var_68 = {var_57: var_67}
    var_69 = 'new'
    var_70 = {var_60: var_69}
    var_71 = 'prod'
    var_72 = [var_71]
    var_73 = {var_58: var_70, var_59: var_72}
    var_74 = {var_57: var_73}
    var_75 = module_0.apply_overwrites_to_context(var_68, var_74, in_dictionary_variable=var_23)



# Parsed testcases at query #2
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '\n    Test the generate_files function by creating a dummy template \n    and verifying that the project is generated correctly.\n    '
    var_1 = 'my_template'
    var_2 = '{{ project_name }}'
    var_3 = 'config.txt'
    var_4 = 'Project: {{ project_name }}\nAuthor: {{ author }}'
    var_5 = 'utf-8'
    var_6 = 'static_assets'
    var_7 = 'data.bin'
    var_8 = 'constant content'
    var_9 = 'cookiecutter.json'
    var_10 = 'project_name'
    var_11 = 'author'
    var_12 = '_copy_without_render'
    var_13 = 'test_project'
    var_14 = 'tester'
    var_15 = 'static_assets/'
    var_16 = [var_15]
    var_17 = {var_10: var_13, var_11: var_14, var_12: var_16}
    var_18 = module_0.dumps(var_17)
    var_19 = 'output'
    var_20 = 'cookiecutter'
    var_21 = 'my_awesome_app'
    var_22 = 'unit_test_runner'
    var_23 = [var_15]
    var_24 = {var_12: var_23}
    var_25 = {var_10: var_21, var_11: var_22, var_20: var_24}
    var_26 = True
    var_27 = False



# Parsed testcases at query #3
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new'
    var_6 = 2
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = 'choices'
    var_10 = 'a'
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_10, var_11, var_12]
    var_14 = {var_9: var_13}
    var_15 = [var_10, var_11]
    var_16 = {var_9: var_15}
    var_17 = module_0.apply_overwrites_to_context(var_14, var_16)
    var_18 = [var_10, var_11, var_12]
    var_19 = {var_9: var_18}
    var_20 = 'z'
    var_21 = [var_10, var_20]
    var_22 = {var_9: var_21}
    var_23 = module_0.apply_overwrites_to_context(var_19, var_22)
    var_24 = [var_10, var_11, var_12]
    var_25 = {var_9: var_24}
    var_26 = {var_9: var_11}
    var_27 = module_0.apply_overwrites_to_context(var_25, var_26)
    var_28 = 'sub'
    var_29 = 'key1'
    var_30 = 'key2'
    var_31 = 'val1'
    var_32 = 'val2'
    var_33 = {var_29: var_31, var_30: var_32}
    var_34 = {var_28: var_33}
    var_35 = 'key3'
    var_36 = 'new_val'
    var_37 = 'added'
    var_38 = {var_30: var_36, var_35: var_37}
    var_39 = {var_28: var_38}
    var_40 = module_0.apply_overwrites_to_context(var_34, var_39)
    var_41 = 'enabled'
    var_42 = False
    var_43 = {var_41: var_42}
    var_44 = 'yes'
    var_45 = {var_41: var_44}
    var_46 = module_0.apply_overwrites_to_context(var_43, var_45)
    var_47 = {var_41: var_42}
    var_48 = 'not-a-boolean'
    var_49 = {var_41: var_48}
    var_50 = module_0.apply_overwrites_to_context(var_47, var_49)
    var_51 = 'nested'
    var_52 = 'existing'
    var_53 = {var_52: var_3}
    var_54 = {var_51: var_53}
    var_55 = 'new_key'
    var_56 = {var_55: var_6}
    var_57 = {var_51: var_56}
    var_58 = True
    var_59 = module_0.apply_overwrites_to_context(var_54, var_57, in_dictionary_variable=var_58)
    var_60 = {var_52: var_58}
    var_61 = 'new_top_level'
    var_62 = 10
    var_63 = {var_61: var_62}
    var_64 = module_0.apply_overwrites_to_context(var_60, var_63, in_dictionary_variable=var_42)
    var_65 = 'options'
    var_66 = 'one'
    var_67 = 'two'
    var_68 = [var_66, var_67]
    var_69 = {var_65: var_68}
    var_70 = {var_65: var_67}
    var_71 = module_0.apply_overwrites_to_context(var_69, var_70)



# Parsed testcases at query #4
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'output'
    var_6 = '{{ project_name }}_dir'
    var_7 = False
    var_8 = 'my_project_dir'
    var_9 = False
    var_10 = 'my_project_dir'
    var_11 = True
    var_12 = ''
    var_13 = {}
    var_14 = '{{ missing_var }}'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test the generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'version'
    var_5 = 'my_project'
    var_6 = 'test_user'
    var_7 = '0.1.0'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'new_key'
    var_10 = 'new_author'
    var_11 = 'new_val'
    var_12 = {var_3: var_10, var_9: var_11}
    var_13 = '1.0.0'
    var_14 = {var_4: var_13}
    var_15 = 'invalid.json'
    var_16 = "{ 'broken': json }"
    var_17 = module_0.generate_context(var_16)
    var_18 = 'settings'
    var_19 = 'debug'
    var_20 = 'port'
    var_21 = True
    var_22 = 8080
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = {var_18: var_23}
    var_25 = 'complex.json'
    var_26 = 9000
    var_27 = {var_20: var_26}
    var_28 = {var_18: var_27}
    var_29 = 'non_existent_file.json'
    var_30 = module_0.generate_context(var_29)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'template_folder'
    var_1 = '.'
    var_2 = 'repo_dir'
    var_3 = 'context'
    var_4 = 'output_dir'
    var_5 = True
    var_6 = 'README.md'
    var_7 = 'static.txt'
    var_8 = 0

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'broken_{{undefined_var}}'
    var_2 = module_0.Environment()
    var_3 = 'repo_dir'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'output_dir'



# Parsed testcases at query #7
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'output'
    var_6 = 'test_dir'
    var_7 = False
    var_8 = '{{ project_name }}_dir'
    var_9 = ''
    var_10 = 'already_here'
    var_11 = 'already_here'
    var_12 = False
    var_13 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test error when template variable is missing in context.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'output'
    var_4 = '{{ project_name }}_dir'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '\n    Test the generate_file function covering:\n    1. Standard text file rendering.\n    2. Binary file copying (simulated).\n    3. Skip if file exists.\n    4. Template syntax error handling.\n    '
    var_1 = 'project'
    var_2 = 'template'
    var_3 = 'hello_{{ name }}.txt'
    var_4 = 'Hello, {{ name }}!'
    var_5 = 'data.bin'
    var_6 = b'\x00\x01\x02\x03'
    var_7 = 'name'
    var_8 = 'cookiecutter'
    var_9 = 'world'
    var_10 = '_new_lines'
    var_11 = '\n'
    var_12 = {var_10: var_11}
    var_13 = {var_7: var_9, var_8: var_12}
    var_14 = {var_3: var_4, var_5: var_6}
    var_15 = 'hello_world.txt'
    var_16 = 'existing.txt'
    var_17 = "Don't overwrite me"
    var_18 = 'I am new content'
    var_19 = True
    var_20 = 'error.txt'
    var_21 = '{{ unclosed_bracket'
    var_22 = {var_20: var_21}



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the full flow of generate_files:\n    1. Template is found.\n    2. Project directory is created via rendering.\n    3. Files are rendered or copied based on context.\n    4. Hooks are executed.\n    '
    var_1 = 'template_dir'
    var_2 = 'output_dir'
    var_3 = 'context'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = 'project_slug'
    var_7 = 'my_project_dir'
    var_8 = 'static.txt'
    var_9 = True
    var_10 = 'config.py.jinja2'



# Parsed testcases at query #10
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.bin'
    var_3 = 'docs/manual.pdf'
    var_4 = 'assets/*'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'test.bin'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True
    var_10 = module_0.is_copy_only_path(var_3, var_7)
    assert var_10 is True
    var_11 = 'assets/image.png'
    var_12 = module_0.is_copy_only_path(var_11, var_7)
    assert var_12 is True
    var_13 = 'src/main.py'
    var_14 = module_0.is_copy_only_path(var_13, var_7)
    assert var_14 is False
    var_15 = '*.txt'
    var_16 = [var_15]
    var_17 = {var_1: var_16}
    var_18 = {var_0: var_17}
    var_19 = 'script.py'
    var_20 = module_0.is_copy_only_path(var_19, var_18)
    assert var_20 is False
    var_21 = {}
    var_22 = {var_0: var_21}
    var_23 = module_0.is_copy_only_path(var_8, var_22)
    assert var_23 is False
    var_24 = {}
    var_25 = module_0.is_copy_only_path(var_8, var_24)
    assert var_25 is False
    var_26 = 'exact_file.txt'
    var_27 = [var_26]
    var_28 = {var_1: var_27}
    var_29 = {var_0: var_28}
    var_30 = module_0.is_copy_only_path(var_26, var_29)
    assert var_30 is True



# Parsed testcases at query #11
#--------------------------


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '\n    Tests the generate_file function covering:\n    1. Rendering a text file with context variables.\n    2. Handling binary files (copy without rendering).\n    3. Handling skip_if_file_exists logic.\n    4. Respecting _new_lines configuration in context.\n    '
    var_1 = 'template'
    var_2 = 'project'
    var_3 = 'hello_{{ name }}.txt'
    var_4 = 'Hello, {{ name }}!'
    assert var_4 == 'Hello, World!'
    var_5 = 'utf-8'
    var_6 = 'data.bin'
    var_7 = b'\x00\x01\x02\x03\xff'
    var_8 = 'already_here.txt'
    var_9 = "Don't overwrite me"
    var_10 = module_0.Environment()
    var_11 = 'cookiecutter'
    var_12 = 'name'
    var_13 = '_new_lines'
    var_14 = '\n'
    var_15 = {var_13: var_14}
    assert var_15 == "Don't overwrite me"
    var_16 = 'World'
    var_17 = {var_11: var_15, var_12: var_16}
    var_18 = 'hello_{{ name }}.txt'
    var_19 = module_1.generate_file(var_0, var_18, var_17, var_10)
    var_20 = 'hello_World.txt'
    var_21 = 'utf-8'
    var_22 = 'data.bin'
    var_23 = module_1.generate_file(var_5, var_22, var_17, var_10)
    var_24 = 'skip_{{ suffix }}.txt'
    var_25 = 'I should be skipped'
    var_26 = True
    var_27 = module_1.generate_file(var_12, var_24, var_17, var_10, var_26)



# Parsed testcases at query #12
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test the generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'test_project'
    var_5 = '0.1.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'cookiecutter'
    var_8 = 'overridden_name'
    var_9 = {var_2: var_8}
    var_10 = '1.0.0'
    var_11 = {var_3: var_10}
    var_12 = 'bad.json'
    var_13 = "{ 'broken': json }"
    var_14 = module_0.generate_context(var_13)
    var_15 = 'config.json'
    var_16 = 'key'
    var_17 = 'val'
    var_18 = {var_16: var_17}
    var_19 = 'non_existent_file.json'
    var_20 = module_0.generate_context(var_19)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context using mocks to avoid filesystem dependency.'
    var_1 = 'test.json'
    var_2 = module_0.generate_context(var_1)



# Parsed testcases at query #13
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '\n    Test the generate_files function by creating a minimal valid template \n    and verifying that the output directory is created with rendered content.\n    '
    var_1 = 'my_template'
    var_2 = '{{ project_name }}'
    var_3 = 'cookiecutter.json'
    var_4 = 'project_name'
    var_5 = 'cookiecutter'
    var_6 = 'my_awesome_project'
    var_7 = '_copy_without_render'
    var_8 = []
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = module_0.dumps(var_10)
    var_12 = 'hello.txt'
    var_13 = 'Hello, {{ project_name }}!'
    var_14 = 'static_dir'
    var_15 = 'info.txt'
    var_16 = 'Do not render me'
    var_17 = 'output'
    var_18 = 'static_dir/info.txt'
    var_19 = [var_18]
    var_20 = {var_7: var_19}
    var_21 = {var_4: var_6, var_5: var_20}
    var_22 = False



# Parsed testcases at query #14
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test the render_and_create_dir function with various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'output'
    var_6 = '{{ project_name }}_dir'
    var_7 = False
    var_8 = ''



# Parsed testcases at query #15
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test the generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'cookiecutter'
    var_4 = 'my_project'
    var_5 = '_copy_without_render'
    var_6 = '_new_lines'
    var_7 = '*.txt'
    var_8 = [var_7]
    var_9 = '\n'
    var_10 = {var_5: var_8, var_6: var_9}
    var_11 = {var_2: var_4, var_3: var_10}
    var_12 = module_0.dumps(var_11)
    var_13 = 'overridden_name'
    var_14 = {var_2: var_13}
    var_15 = 'default_name'
    var_16 = {var_2: var_15}
    var_17 = 'bad.json'
    var_18 = "{ 'broken': json }"
    var_19 = module_1.generate_context(var_0)
    var_20 = 'template.json'
    var_21 = 'version'
    var_22 = '1.0'
    var_23 = {var_21: var_22}
    var_24 = module_0.dumps(var_23)

def test_case_0():
    var_0 = 'Specific tests for the underlying logic of apply_overwrites_to_context \n    via generate_context behavior.'
    var_1 = 'test_gen_ctx'
    var_2 = True
    var_3 = 'config.json'
    var_4 = 'cookiecutter'
    var_5 = 'choice_var'
    var_6 = 'bool_var'
    var_7 = 'dict_var'
    var_8 = 'a'
    var_9 = 'b'
    var_10 = [var_8, var_9]
    var_11 = False
    var_12 = 'key'
    var_13 = 'old'
    var_14 = {var_12: var_13}
    var_15 = {var_5: var_10, var_6: var_11, var_7: var_14}
    var_16 = {var_4: var_15}
    var_17 = [var_9]
    var_18 = {var_5: var_17}
    var_19 = {var_4: var_18}
    var_20 = 'yes'
    var_21 = {var_6: var_20}
    var_22 = {var_4: var_21}
    var_23 = 'added'
    var_24 = 'new'
    var_25 = True
    var_26 = {var_12: var_24, var_23: var_25}
    var_27 = {var_7: var_26}
    var_28 = {var_4: var_27}



# Parsed testcases at query #16
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'output'
    var_6 = '{{ project_name }}_dir'
    var_7 = False
    var_8 = '{{ project_name }}_dir'
    var_9 = True
    var_10 = '{{ project_name }}_dir'
    var_11 = False
    var_12 = ''
    var_13 = 'Undefined Variable'
    var_14 = '{{ undefined_var }}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the path rendering works correctly for nested paths.'
    var_1 = module_0.Environment()
    var_2 = 'sub'
    var_3 = 'name'
    var_4 = 'folder'
    var_5 = 'app'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'out'
    var_8 = '{{ sub }}/{{ name }}'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = '\n    Unit test for generate_files.\n    Tests a successful project generation flow including:\n    - Template directory setup\n    - Context application\n    - Directory and file creation\n    - Hook execution (mocked)\n    '
    var_1 = 'template_repo'
    var_2 = '{{ project_name }}'
    var_3 = 'README.md'
    var_4 = 'Hello {{ project_name }}!'
    var_5 = 'utf-8'
    var_6 = 'static_assets'
    var_7 = 'logo.png'
    assert var_7 == 'Hello my_awesome_project!'
    var_8 = b'\x89PNG\r\n\x1a\n'
    var_9 = 'project_name'
    var_10 = 'cookiecutter'
    var_11 = 'my_awesome_project'
    assert var_11 == b'\x89PNG\r\n\x1a\n'
    var_12 = '_copy_without_render'
    var_13 = '_new_lines'
    var_14 = 'static_assets/'
    var_15 = [var_14]
    var_16 = '\n'
    var_17 = {var_12: var_15, var_13: var_16}
    var_18 = {var_9: var_11, var_10: var_17}
    var_19 = '{{ project_name }}'
    var_20 = 'project_name'
    var_21 = 'output'
    var_22 = True
    var_23 = 'README.md'
    var_24 = 'utf-8'
    var_25 = 'static_assets'
    var_26 = 'logo.png'
    var_27 = 0



