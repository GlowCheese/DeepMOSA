####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test apply_overwrites_to_context function with various scenarios.'
    var_1 = 'name'
    var_2 = 'old_value'
    var_3 = {var_1: var_2}
    var_4 = 'new_value'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = 'existing'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'new_var'
    var_11 = {var_10: var_4}
    var_12 = module_0.apply_overwrites_to_context(var_9, var_11)
    var_13 = 'nested'
    var_14 = {var_7: var_8}
    var_15 = {var_13: var_14}
    var_16 = {var_10: var_4}
    var_17 = {var_13: var_16}
    var_18 = True
    var_19 = module_0.apply_overwrites_to_context(var_15, var_17, in_dictionary_variable=var_18)
    var_20 = 'choice'
    var_21 = 'option1'
    var_22 = 'option2'
    var_23 = 'option3'
    var_24 = [var_21, var_22, var_23]
    var_25 = {var_20: var_24}
    var_26 = {var_20: var_22}
    var_27 = module_0.apply_overwrites_to_context(var_25, var_26)
    var_28 = [var_21, var_22]
    var_29 = {var_20: var_28}
    var_30 = 'invalid_option'
    var_31 = {var_20: var_30}
    var_32 = module_0.apply_overwrites_to_context(var_29, var_31)
    var_33 = 'multichoice'
    var_34 = 'opt1'
    var_35 = 'opt2'
    var_36 = 'opt3'
    var_37 = [var_34, var_35, var_36]
    var_38 = {var_33: var_37}
    var_39 = [var_34, var_36]
    var_40 = {var_33: var_39}
    var_41 = module_0.apply_overwrites_to_context(var_38, var_40)
    var_42 = [var_34, var_35]
    var_43 = {var_33: var_42}
    var_44 = 'invalid'
    var_45 = [var_34, var_44]
    var_46 = {var_33: var_45}
    var_47 = module_0.apply_overwrites_to_context(var_43, var_46)
    var_48 = 'config'
    var_49 = 'key1'
    var_50 = 'key2'
    var_51 = 'value1'
    var_52 = 'value2'
    var_53 = {var_49: var_51, var_50: var_52}
    var_54 = {var_48: var_53}
    var_55 = 'new_value1'
    var_56 = {var_49: var_55}
    var_57 = {var_48: var_56}
    var_58 = module_0.apply_overwrites_to_context(var_54, var_57)
    var_59 = 'flag'
    var_60 = {var_59: var_18}
    var_61 = 'y'
    var_62 = {var_59: var_61}
    var_63 = module_0.apply_overwrites_to_context(var_60, var_62)
    var_64 = {var_59: var_18}
    var_65 = 'n'
    var_66 = {var_59: var_65}
    var_67 = module_0.apply_overwrites_to_context(var_64, var_66)
    var_68 = {var_59: var_18}
    var_69 = 'invalid_bool'
    var_70 = {var_59: var_69}
    var_71 = module_0.apply_overwrites_to_context(var_68, var_70)
    var_72 = 'items'
    var_73 = 'a'
    var_74 = 'b'
    var_75 = 'c'
    var_76 = [var_73, var_74, var_75]
    var_77 = {var_72: var_76}
    var_78 = {var_13: var_77}
    var_79 = [var_74, var_75]
    var_80 = {var_72: var_79}
    var_81 = {var_13: var_80}
    var_82 = module_0.apply_overwrites_to_context(var_78, var_81, in_dictionary_variable=var_18)
    var_83 = 'key'
    var_84 = {var_83: var_8}
    var_85 = {}
    var_86 = module_0.apply_overwrites_to_context(var_84, var_85)
    var_87 = {var_83: var_8}
    var_88 = None
    var_89 = {var_83: var_88}
    var_90 = module_0.apply_overwrites_to_context(var_87, var_89)
    var_91 = 'project'
    var_92 = 'version'
    var_93 = 'options'
    var_94 = 'myproject'
    var_95 = '1.0'
    var_96 = [var_34, var_35]
    var_97 = {var_1: var_94, var_92: var_95, var_93: var_96}
    var_98 = {var_91: var_97}
    var_99 = 'newproject'
    var_100 = [var_35]
    var_101 = {var_1: var_99, var_93: var_100}
    var_102 = {var_91: var_101}
    var_103 = module_0.apply_overwrites_to_context(var_98, var_102, in_dictionary_variable=var_18)



# Parsed testcases at query #2
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir function.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False
    var_9 = ''
    var_10 = '{{cookiecutter.project_name}}'
    var_11 = False
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = True
    var_14 = 'org'
    var_15 = 'project'
    var_16 = 'myorg'
    var_17 = 'myproj'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = {var_11: var_18}
    var_20 = '{{cookiecutter.org}}/{{cookiecutter.project}}'
    var_21 = 'static_dir'
    var_22 = ''



# Parsed testcases at query #3
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir function.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test render_and_create_dir with empty directory name.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = '.'
    var_6 = ''
    var_7 = module_1.render_and_create_dir(var_6, var_3, var_5, var_4)

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir when directory exists and overwrite is False.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir when directory exists and overwrite is True.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with nested directory structure.'
    var_1 = 'cookiecutter'
    var_2 = 'org'
    var_3 = 'project'
    var_4 = 'myorg'
    var_5 = 'myproj'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.Environment()
    var_9 = '{{cookiecutter.org}}/{{cookiecutter.project}}'
    var_10 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with no template variables.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = 'static_dir_name'
    var_6 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir returns Path object.'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.name}}'
    var_8 = False



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test generate_file function with various file types and scenarios.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'templates'
    var_4 = 'test_{{cookiecutter.name}}.txt'
    var_5 = 'Hello {{cookiecutter.name}}!'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = '_new_lines'
    var_9 = 'world'
    var_10 = None
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {var_6: var_11}
    var_13 = 'test_{{cookiecutter.name}}.txt'
    var_14 = False
    var_15 = 'test_world.txt'
    var_16 = 'test_{{cookiecutter.name}}.txt'
    var_17 = True
    var_18 = 'binary.bin'
    var_19 = b'\x89PNG\r\n\x1a\n'
    var_20 = 'binaryornot.check.is_binary'
    var_21 = 'binary.bin'
    var_22 = False
    var_23 = 'empty_dir'
    var_24 = 'empty_dir'
    var_25 = False
    var_26 = 'newlines.txt'
    var_27 = 'line1\r\nline2\r\n'
    var_28 = ''
    var_29 = 'test'
    var_30 = '\r\n'
    var_31 = {var_7: var_29, var_8: var_30}
    var_32 = {var_6: var_31}
    var_33 = 'newlines.txt'
    var_34 = False
    var_35 = 'syntax_error.txt'
    var_36 = '{{cookiecutter.name'
    var_37 = 'syntax_error.txt'
    var_38 = False
    var_39 = 'executable.sh'
    var_40 = '#!/bin/bash\necho test'
    var_41 = 493
    var_42 = 'executable.sh'
    var_43 = False



# Parsed testcases at query #5
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test apply_overwrites_to_context function with various scenarios.'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'original'
    var_4 = 30
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'new_name'
    var_7 = {var_1: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = {var_1: var_3}
    var_10 = 'new_var'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = module_0.apply_overwrites_to_context(var_9, var_12)
    var_14 = 'color'
    var_15 = 'red'
    var_16 = 'green'
    var_17 = 'blue'
    var_18 = [var_15, var_16, var_17]
    var_19 = {var_14: var_18}
    var_20 = {var_14: var_16}
    var_21 = module_0.apply_overwrites_to_context(var_19, var_20)
    var_22 = [var_15, var_16, var_17]
    var_23 = {var_14: var_22}
    var_24 = 'yellow'
    var_25 = {var_14: var_24}
    var_26 = module_0.apply_overwrites_to_context(var_23, var_25)
    var_27 = 'languages'
    var_28 = 'python'
    var_29 = 'javascript'
    var_30 = 'java'
    var_31 = [var_28, var_29, var_30]
    var_32 = {var_27: var_31}
    var_33 = [var_28, var_30]
    var_34 = {var_27: var_33}
    var_35 = module_0.apply_overwrites_to_context(var_32, var_34)
    var_36 = [var_28, var_29, var_30]
    var_37 = {var_27: var_36}
    var_38 = 'rust'
    var_39 = [var_28, var_38]
    var_40 = {var_27: var_39}
    var_41 = module_0.apply_overwrites_to_context(var_37, var_40)
    var_42 = 'is_active'
    var_43 = True
    var_44 = {var_42: var_43}
    var_45 = 'y'
    var_46 = {var_42: var_45}
    var_47 = module_0.apply_overwrites_to_context(var_44, var_46)
    var_48 = {var_42: var_43}
    var_49 = 'maybe'
    var_50 = {var_42: var_49}
    var_51 = module_0.apply_overwrites_to_context(var_48, var_50)
    var_52 = 'config'
    var_53 = 'debug'
    var_54 = 'port'
    var_55 = 8000
    var_56 = {var_53: var_43, var_54: var_55}
    var_57 = {var_52: var_56}
    var_58 = False
    var_59 = {var_53: var_58}
    var_60 = {var_52: var_59}
    var_61 = module_0.apply_overwrites_to_context(var_57, var_60)
    var_62 = {var_53: var_43}
    var_63 = {var_52: var_62}
    var_64 = 'new_key'
    var_65 = 'new_value'
    var_66 = {var_64: var_65}
    var_67 = {var_52: var_66}
    var_68 = module_0.apply_overwrites_to_context(var_63, var_67, in_dictionary_variable=var_43)
    var_69 = 'items'
    var_70 = 'a'
    var_71 = 'b'
    var_72 = [var_70, var_71]
    var_73 = {var_69: var_72}
    var_74 = {var_52: var_73}
    var_75 = 'x'
    var_76 = [var_75, var_45]
    var_77 = {var_69: var_76}
    var_78 = {var_52: var_77}
    var_79 = module_0.apply_overwrites_to_context(var_74, var_78, in_dictionary_variable=var_43)
    var_80 = {var_1: var_3, var_2: var_4}
    var_81 = {}
    var_82 = module_0.apply_overwrites_to_context(var_80, var_81)
    var_83 = 'project'
    var_84 = 'features'
    var_85 = 'settings'
    var_86 = 'myproject'
    var_87 = 'auth'
    var_88 = 'api'
    var_89 = [var_87, var_88]
    var_90 = {var_53: var_43}
    var_91 = {var_1: var_86, var_84: var_89, var_85: var_90}
    var_92 = {var_83: var_91}
    var_93 = 'newproject'
    var_94 = {var_53: var_58}
    var_95 = {var_1: var_93, var_85: var_94}
    var_96 = {var_83: var_95}
    var_97 = module_0.apply_overwrites_to_context(var_92, var_96, in_dictionary_variable=var_43)



# Parsed testcases at query #6
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context function.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John Doe"}'
    var_3 = 'project_name'
    var_4 = 'override_project'
    var_5 = {var_3: var_4}
    var_6 = 'author'
    var_7 = 'Jane Doe'
    var_8 = {var_6: var_7}
    var_9 = 'default_project'
    var_10 = {var_3: var_9}
    var_11 = 'extra_project'
    var_12 = {var_3: var_11}
    var_13 = 'choice_context.json'
    var_14 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_15 = 'license'
    var_16 = 'Apache'
    var_17 = {var_15: var_16}
    var_18 = 'multi_choice.json'
    var_19 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_20 = 'features'
    var_21 = 'feature2'
    var_22 = 'feature3'
    var_23 = [var_21, var_22]
    var_24 = {var_20: var_23}
    var_25 = 'bool_context.json'
    var_26 = '{"use_docker": true}'
    var_27 = 'use_docker'
    var_28 = 'n'
    var_29 = {var_27: var_28}
    var_30 = 'nested.json'
    var_31 = '{"options": {"debug": false, "verbose": true}}'
    var_32 = 'options'
    var_33 = 'debug'
    var_34 = True
    var_35 = {var_33: var_34}
    var_36 = {var_32: var_35}
    var_37 = 'invalid.json'
    var_38 = '{invalid json}'
    var_39 = module_0.generate_context(var_0)
    var_40 = 'license'
    var_41 = 'InvalidLicense'
    var_42 = {var_40: var_41}
    var_43 = module_0.generate_context(var_0, extra_context=var_42)
    var_44 = 'features'
    var_45 = 'invalid'
    var_46 = [var_45]
    var_47 = {var_44: var_46}
    var_48 = module_0.generate_context(var_0, extra_context=var_47)
    var_49 = 'use_docker'
    var_50 = 'maybe'
    var_51 = {var_49: var_50}
    var_52 = module_0.generate_context(var_0, extra_context=var_51)
    var_53 = 'ordered.json'
    var_54 = '{"first": 1, "second": 2, "third": 3}'
    var_55 = 'cookiecutter'
    var_56 = 'custom_context.json'
    var_57 = '{"key": "value"}'
    var_58 = 'config.json'
    var_59 = '{"setting": "value"}'



# Parsed testcases at query #7
#--------------------------


import jinja2.environment as module_0
import jinja2.loaders as module_1
import jinja2.exceptions as module_2

def test_case_0():
    var_0 = 'Test generate_file function with various scenarios.'
    var_1 = 'binary_file.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'template_{{cookiecutter.project_name}}.txt'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'myproject'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = '.'
    var_13 = module_1.FileSystemLoader(var_12)
    var_14 = module_0.Environment(loader=var_13)
    var_15 = 'somefile'
    var_16 = 'cookiecutter'
    var_17 = {}
    var_18 = {var_16: var_17}
    var_19 = module_0.Environment()
    var_20 = 'existing_file.txt'
    var_21 = 'cookiecutter'
    var_22 = {}
    var_23 = {var_21: var_22}
    var_24 = module_0.Environment()
    var_25 = True
    var_26 = 'bad_template.txt'
    var_27 = 'cookiecutter'
    var_28 = {}
    var_29 = {var_27: var_28}
    var_30 = '.'
    var_31 = module_1.FileSystemLoader(var_30)
    var_32 = module_0.Environment(loader=var_31)
    var_33 = 'Bad syntax'
    var_34 = 1
    var_35 = module_2.TemplateSyntaxError(var_33, var_34)
    var_36 = 'newline_file.txt'
    var_37 = 'cookiecutter'
    var_38 = '_new_lines'
    var_39 = '\r\n'
    var_40 = {var_38: var_39}
    var_41 = {var_37: var_40}
    var_42 = '.'
    var_43 = module_1.FileSystemLoader(var_42)
    var_44 = module_0.Environment(loader=var_43)
    var_45 = 'detect_newline.txt'
    var_46 = 'cookiecutter'
    var_47 = {}
    var_48 = {var_46: var_47}
    var_49 = '.'
    var_50 = module_1.FileSystemLoader(var_49)
    var_51 = module_0.Environment(loader=var_50)
    var_52 = 'line1\nline2\n'
    var_53 = 'subdir'
    var_54 = 'template.txt'
    var_55 = 'cookiecutter'
    var_56 = {}
    var_57 = {var_55: var_56}
    var_58 = '.'
    var_59 = module_1.FileSystemLoader(var_58)
    var_60 = module_0.Environment(loader=var_59)
    var_61 = 0
    var_62 = '/'
    var_63 = var_56 > var_61



# Parsed testcases at query #8
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test is_copy_only_path function with various patterns and paths.'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.txt'
    var_4 = 'docs/*'
    var_5 = 'binary/*'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = 'file.txt'
    var_10 = module_0.is_copy_only_path(var_9, var_8)
    assert var_10 is True
    var_11 = 'docs/readme.md'
    var_12 = module_0.is_copy_only_path(var_11, var_8)
    assert var_12 is True
    var_13 = 'binary/image.png'
    var_14 = module_0.is_copy_only_path(var_13, var_8)
    assert var_14 is True
    var_15 = 'file.py'
    var_16 = module_0.is_copy_only_path(var_15, var_8)
    assert var_16 is False
    var_17 = 'script.sh'
    var_18 = module_0.is_copy_only_path(var_17, var_8)
    assert var_18 is False
    var_19 = 'source/code.txt'
    var_20 = module_0.is_copy_only_path(var_19, var_8)
    assert var_20 is False
    var_21 = []
    var_22 = {var_2: var_21}
    var_23 = {var_1: var_22}
    var_24 = module_0.is_copy_only_path(var_9, var_23)
    assert var_24 is False
    var_25 = module_0.is_copy_only_path(var_11, var_23)
    assert var_25 is False
    var_26 = {}
    var_27 = {var_1: var_26}
    var_28 = module_0.is_copy_only_path(var_9, var_27)
    assert var_28 is False
    var_29 = module_0.is_copy_only_path(var_11, var_27)
    assert var_29 is False
    var_30 = {}
    var_31 = module_0.is_copy_only_path(var_9, var_30)
    assert var_31 is False
    var_32 = '*.min.js'
    var_33 = 'node_modules/**'
    var_34 = '.*'
    var_35 = [var_32, var_33, var_34]
    var_36 = {var_2: var_35}
    var_37 = {var_1: var_36}
    var_38 = 'script.min.js'
    var_39 = module_0.is_copy_only_path(var_38, var_37)
    assert var_39 is True
    var_40 = 'node_modules/package'
    var_41 = module_0.is_copy_only_path(var_40, var_37)
    assert var_41 is True
    var_42 = '.gitignore'
    var_43 = module_0.is_copy_only_path(var_42, var_37)
    assert var_43 is True
    var_44 = 'script.js'
    var_45 = module_0.is_copy_only_path(var_44, var_37)
    assert var_45 is False
    var_46 = 'file?.txt'
    var_47 = [var_46]
    var_48 = {var_2: var_47}
    var_49 = {var_1: var_48}
    var_50 = 'file1.txt'
    var_51 = module_0.is_copy_only_path(var_50, var_49)
    assert var_51 is True
    var_52 = 'fileA.txt'
    var_53 = module_0.is_copy_only_path(var_52, var_49)
    assert var_53 is True
    var_54 = module_0.is_copy_only_path(var_9, var_49)
    assert var_54 is False
    var_55 = 'file12.txt'
    var_56 = module_0.is_copy_only_path(var_55, var_49)
    assert var_56 is False
    var_57 = 'specific/path/file.txt'
    var_58 = [var_57]
    var_59 = {var_2: var_58}
    var_60 = {var_1: var_59}
    var_61 = module_0.is_copy_only_path(var_57, var_60)
    assert var_61 is True
    var_62 = 'specific/path/other.txt'
    var_63 = module_0.is_copy_only_path(var_62, var_60)
    assert var_63 is False
    var_64 = 'different/path/file.txt'
    var_65 = module_0.is_copy_only_path(var_64, var_60)
    assert var_65 is False



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test generate_context with valid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'author_name'
    var_5 = 'My Project'
    var_6 = '{{ cookiecutter.project_name.lower().replace(" ", "_") }}'
    var_7 = 'John Doe'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}

def test_case_0():
    var_0 = 'Test generate_context with default_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author_name'
    var_4 = 'My Project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'Jane Doe'
    var_8 = {var_3: var_7}

def test_case_0():
    var_0 = 'Test generate_context with extra_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author_name'
    var_4 = 'My Project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'Another Project'
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{ invalid json }'

def test_case_0():
    var_0 = 'Test generate_context with choice variable and default context.'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = 'GPL'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_2: var_4}

def test_case_0():
    var_0 = 'Test generate_context with invalid choice in default context.'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = 'GPL'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = 'InvalidLicense'
    var_9 = {var_2: var_8}

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary variable.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'options'
    var_4 = 'My Project'
    var_5 = 'use_docker'
    var_6 = 'use_ci'
    var_7 = True
    var_8 = False
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_2: var_4, var_3: var_9}
    var_11 = {var_6: var_7}
    var_12 = {var_3: var_11}

def test_case_0():
    var_0 = 'Test generate_context with boolean variable and string override.'
    var_1 = 'cookiecutter.json'
    var_2 = 'use_docker'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test generate_context with multichoice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = 'features'
    var_3 = 'feature1'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = [var_4, var_5]
    var_9 = {var_2: var_8}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with non-existent file.'
    var_1 = '/nonexistent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)



# Parsed testcases at query #10
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'version'
    var_5 = 'my_project'
    var_6 = 'John Doe'
    var_7 = '0.1.0'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'cookiecutter2.json'
    var_11 = 'use_docker'
    var_12 = 'default_project'
    var_13 = 'Default Author'
    var_14 = True
    var_15 = {var_2: var_12, var_3: var_13, var_11: var_14}
    var_16 = module_0.dumps(var_15)
    var_17 = 'overridden_project'
    var_18 = {var_2: var_17, var_3: var_13}
    var_19 = 'cookiecutter3.json'
    var_20 = 'original_project'
    var_21 = '1.0.0'
    var_22 = {var_2: var_20, var_4: var_21}
    var_23 = module_0.dumps(var_22)
    var_24 = 'extra_project'
    var_25 = '2.0.0'
    var_26 = {var_2: var_24, var_4: var_25}
    var_27 = 'cookiecutter4.json'
    var_28 = 'python_version'
    var_29 = '3.8'
    var_30 = '3.9'
    var_31 = '3.10'
    var_32 = [var_29, var_30, var_31]
    var_33 = {var_28: var_32}
    var_34 = module_0.dumps(var_33)
    var_35 = {var_28: var_30}
    var_36 = 'cookiecutter5.json'
    var_37 = 'features'
    var_38 = 'docker'
    var_39 = 'ci'
    var_40 = 'docs'
    var_41 = 'tests'
    var_42 = [var_38, var_39, var_40, var_41]
    var_43 = {var_37: var_42}
    var_44 = module_0.dumps(var_43)
    var_45 = [var_38, var_41]
    var_46 = {var_37: var_45}
    var_47 = 'cookiecutter6.json'
    var_48 = 'project'
    var_49 = 'name'
    var_50 = 'slug'
    var_51 = {var_49: var_5, var_50: var_5}
    var_52 = {var_48: var_51}
    var_53 = module_0.dumps(var_52)
    var_54 = 'updated_project'
    var_55 = {var_49: var_54}
    var_56 = {var_48: var_55}
    var_57 = 'invalid.json'
    var_58 = '{invalid json content'
    var_59 = module_1.generate_context(var_0)
    var_60 = 'cookiecutter8.json'
    var_61 = 'use_ci'
    var_62 = False
    var_63 = {var_11: var_14, var_61: var_62}
    var_64 = module_0.dumps(var_63)
    var_65 = 'n'
    var_66 = 'y'
    var_67 = {var_11: var_65, var_61: var_66}
    var_68 = 'cookiecutter9.json'
    var_69 = [var_29, var_30, var_31]
    var_70 = {var_28: var_69}
    var_71 = module_0.dumps(var_70)
    var_72 = '3.7'
    var_73 = {var_28: var_72}
    var_74 = module_1.generate_context(var_0, extra_context=var_73)
    var_75 = 'cookiecutter10.json'
    var_76 = 'first'
    var_77 = '1'
    var_78 = (var_76, var_77)
    var_79 = 'second'
    var_80 = '2'
    var_81 = (var_79, var_80)
    var_82 = 'third'
    var_83 = '3'
    var_84 = (var_82, var_83)
    var_85 = [var_78, var_81, var_84]



# Parsed testcases at query #11
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir creates directory with rendered name.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises error when directory exists.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir overwrites existing directory.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises error for empty directory name.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises error for empty string dirname.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = '   '

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with nested context variables.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'author'
    var_4 = 'project'
    var_5 = 'john'
    var_6 = 'awesome'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '{{cookiecutter.author}}_{{cookiecutter.project}}'
    var_10 = 'john_awesome'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir works with string output_dir.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir returns tuple with Path and bool.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = 0
    var_9 = 1



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test generate_file function.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'template'
    var_4 = 'test_{{cookiecutter.name}}.txt'
    var_5 = 'Hello {{cookiecutter.greeting}}'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'greeting'
    var_9 = 'World'
    var_10 = {var_7: var_1, var_8: var_9}
    var_11 = {var_6: var_10}
    var_12 = 'test_project.txt'

def test_case_0():
    var_0 = 'Test generate_file with binary file.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'template'
    var_4 = 'test.bin'
    var_5 = b'\x89PNG\r\n\x1a\n'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = 'Test generate_file skips existing files.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'template'
    var_4 = 'test.txt'
    var_5 = 'New content'
    var_6 = 'Old content'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}

def test_case_0():
    var_0 = 'Test generate_file with empty directory name.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'template'
    var_4 = '.'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 'Test generate_file respects _new_lines configuration.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'template'
    var_4 = 'test.txt'
    var_5 = 'line1\nline2\n'
    var_6 = 'cookiecutter'
    var_7 = '_new_lines'
    var_8 = '\r\n'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}

def test_case_0():
    var_0 = 'Test generate_file raises on template syntax error.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'template'
    var_4 = 'test.txt'
    var_5 = '{{cookiecutter.name'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}

def test_case_0():
    var_0 = 'Test generate_file renders context variables correctly.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'template'
    var_4 = '{{cookiecutter.module}}.py'
    var_5 = 'def {{cookiecutter.function}}():\n    pass'
    var_6 = 'cookiecutter'
    var_7 = 'module'
    var_8 = 'function'
    var_9 = 'mymodule'
    var_10 = 'myfunc'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {var_6: var_11}
    var_13 = 'mymodule.py'



# Parsed testcases at query #13
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'author'
    var_5 = 'my_project'
    var_6 = "{{ cookiecutter.project_name.lower().replace(' ', '_') }}"
    var_7 = 'John Doe'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'cookiecutter2.json'
    var_11 = 'version'
    var_12 = 'default_project'
    var_13 = '1.0.0'
    var_14 = {var_2: var_12, var_11: var_13}
    var_15 = module_0.dumps(var_14)
    var_16 = 'overridden_project'
    var_17 = '2.0.0'
    var_18 = {var_2: var_16, var_11: var_17}
    var_19 = 'cookiecutter3.json'
    var_20 = 'license'
    var_21 = 'original_project'
    var_22 = 'MIT'
    var_23 = {var_2: var_21, var_20: var_22}
    var_24 = module_0.dumps(var_23)
    var_25 = 'extra_project'
    var_26 = 'Apache'
    var_27 = {var_2: var_25, var_20: var_26}
    var_28 = 'cookiecutter4.json'
    var_29 = 'python_version'
    var_30 = '3.8'
    var_31 = '3.9'
    var_32 = '3.10'
    var_33 = [var_30, var_31, var_32]
    var_34 = {var_29: var_33}
    var_35 = module_0.dumps(var_34)
    var_36 = {var_29: var_31}
    var_37 = 'cookiecutter5.json'
    var_38 = 'features'
    var_39 = 'feature1'
    var_40 = 'feature2'
    var_41 = 'feature3'
    var_42 = [var_39, var_40, var_41]
    var_43 = {var_38: var_42}
    var_44 = module_0.dumps(var_43)
    var_45 = [var_39, var_41]
    var_46 = {var_38: var_45}
    var_47 = 'cookiecutter'
    var_48 = 'cookiecutter6.json'
    var_49 = 'config'
    var_50 = 'debug'
    var_51 = 'port'
    var_52 = True
    var_53 = 8000
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = {var_49: var_54}
    var_56 = module_0.dumps(var_55)
    var_57 = 9000
    var_58 = {var_51: var_57}
    var_59 = {var_49: var_58}
    var_60 = 'cookiecutter7.json'
    var_61 = 'use_docker'
    var_62 = {var_61: var_52}
    var_63 = module_0.dumps(var_62)
    var_64 = 'n'
    var_65 = {var_61: var_64}
    var_66 = 'cookiecutter8.json'
    var_67 = '{invalid json}'
    var_68 = module_1.generate_context(var_0)
    var_69 = 'cookiecutter9.json'
    var_70 = 'framework'
    var_71 = 'django'
    var_72 = 'flask'
    var_73 = [var_71, var_72]
    var_74 = {var_70: var_73}
    var_75 = module_0.dumps(var_74)
    var_76 = 'fastapi'
    var_77 = {var_70: var_76}
    var_78 = module_1.generate_context(var_0, extra_context=var_77)
    var_79 = 'cookiecutter10.json'
    var_80 = 'packages'
    var_81 = 'numpy'
    var_82 = 'pandas'
    var_83 = [var_81, var_82]
    var_84 = {var_80: var_83}
    var_85 = module_0.dumps(var_84)
    var_86 = 'scipy'
    var_87 = [var_81, var_86]
    var_88 = {var_80: var_87}
    var_89 = module_1.generate_context(var_0, extra_context=var_88)
    var_90 = 'cookiecutter11.json'
    var_91 = 'enable_feature'
    var_92 = False
    var_93 = {var_91: var_92}
    var_94 = module_0.dumps(var_93)
    var_95 = 'maybe'
    var_96 = {var_91: var_95}
    var_97 = module_1.generate_context(var_0, extra_context=var_96)
    var_98 = 'cookiecutter12.json'
    var_99 = 'name'
    var_100 = 'original'
    var_101 = '1.0'
    var_102 = {var_99: var_100, var_11: var_101}
    var_103 = module_0.dumps(var_102)
    var_104 = 'default'
    var_105 = '2.0'
    var_106 = {var_99: var_104, var_11: var_105}
    var_107 = 'extra'
    var_108 = {var_99: var_107}



# Parsed testcases at query #14
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'author_name'
    var_5 = 'my_project'
    var_6 = "{{ cookiecutter.project_name.lower().replace(' ', '_') }}"
    var_7 = 'John Doe'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.dumps(var_8)

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with default_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author_name'
    var_4 = 'my_project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'Jane Smith'
    var_9 = {var_3: var_8}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with extra_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author_name'
    var_4 = 'my_project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'new_project'
    var_9 = {var_2: var_8}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with choice variable and overwrite.'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = 'GPL'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = {var_2: var_4}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with multi-choice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = 'features'
    var_3 = 'feature1'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = [var_3, var_5]
    var_10 = {var_2: var_9}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with dictionary variable.'
    var_1 = 'cookiecutter.json'
    var_2 = 'config'
    var_3 = 'debug'
    var_4 = 'port'
    var_5 = True
    var_6 = 8000
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 9000
    var_11 = {var_4: var_10}
    var_12 = {var_2: var_11}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with boolean variable string conversion.'
    var_1 = 'cookiecutter.json'
    var_2 = 'include_tests'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = 'n'
    var_7 = {var_2: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{invalid json}'
    var_3 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with nonexistent file.'
    var_1 = '/nonexistent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context with invalid choice value.'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = 'GPL'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = 'BSD'
    var_10 = {var_2: var_9}
    var_11 = module_1.generate_context(var_0, extra_context=var_10)

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context with invalid boolean string conversion.'
    var_1 = 'cookiecutter.json'
    var_2 = 'include_tests'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = 'maybe'
    var_7 = {var_2: var_6}
    var_8 = module_1.generate_context(var_0, extra_context=var_7)

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context preserves ordering with OrderedDict.'
    var_1 = 'cookiecutter.json'
    var_2 = 'z_field'
    var_3 = 'a_field'
    var_4 = 'm_field'
    var_5 = 'last'
    var_6 = 'first'
    var_7 = 'middle'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'cookiecutter'



# Parsed testcases at query #15
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir function.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False
    var_9 = False
    var_10 = True
    var_11 = ''
    var_12 = False
    var_13 = 'static_dir'
    var_14 = '{{cookiecutter.project_name}}/src'
    var_15 = 'src'



# Parsed testcases at query #16
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'author_name'
    var_5 = 'My Project'
    var_6 = "{{ cookiecutter.project_name.lower().replace(' ', '_') }}"
    var_7 = 'John Doe'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'Default Project'
    var_10 = {var_2: var_9}
    var_11 = 'Jane Smith'
    var_12 = {var_4: var_11}
    var_13 = 'license'
    var_14 = 'MIT'
    var_15 = 'Apache'
    var_16 = 'GPL'
    var_17 = [var_14, var_15, var_16]
    var_18 = 'test'
    var_19 = {var_13: var_17, var_2: var_18}
    var_20 = 'cookiecutter_choice.json'
    var_21 = {var_13: var_15}
    var_22 = 'features'
    var_23 = 'feature1'
    var_24 = 'feature2'
    var_25 = 'feature3'
    var_26 = [var_23, var_24, var_25]
    var_27 = {var_22: var_26, var_2: var_18}
    var_28 = 'cookiecutter_multichoice.json'
    var_29 = [var_24, var_25]
    var_30 = {var_22: var_29}
    var_31 = 'options'
    var_32 = 'debug'
    var_33 = 'verbose'
    var_34 = True
    var_35 = False
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = {var_2: var_18, var_31: var_36}
    var_38 = 'cookiecutter_dict.json'
    var_39 = {var_32: var_35}
    var_40 = {var_31: var_39}
    var_41 = 'use_docker'
    var_42 = {var_2: var_18, var_41: var_34}
    var_43 = 'cookiecutter_bool.json'
    var_44 = 'n'
    var_45 = {var_41: var_44}
    var_46 = 'invalid.json'
    var_47 = '{invalid json content'
    var_48 = module_0.generate_context(var_47)
    var_49 = 'license'
    var_50 = 'InvalidLicense'
    var_51 = {var_49: var_50}
    var_52 = module_0.generate_context(var_2, extra_context=var_51)
    var_53 = 'use_docker'
    var_54 = 'maybe'
    var_55 = {var_53: var_54}
    var_56 = module_0.generate_context(var_2, extra_context=var_55)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test generate_file function with various scenarios.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'templates'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_new_lines'
    var_7 = 'my_project'
    var_8 = None
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'test_{{ cookiecutter.project_name }}.txt'
    var_12 = 'Project: {{ cookiecutter.project_name }}'
    var_13 = 'test_my_project.txt'
    var_14 = 'binary_file.bin'
    var_15 = b'\x89PNG\r\n\x1a\n'
    var_16 = 'test_{{ cookiecutter.project_name }}/'
    var_17 = 'newline_test.txt'
    var_18 = 'Line 1\nLine 2\n'
    var_19 = '\n'
    var_20 = 'perm_test.txt'
    var_21 = 'test content'
    var_22 = 493



# Parsed testcases at query #18
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test generate_file function with various scenarios.'
    var_1 = 'binary_file.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'test_template.txt'
    var_7 = 'cookiecutter'
    var_8 = 'name'
    var_9 = 'test_project'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = module_0.Environment()
    var_13 = 'test.txt'
    var_14 = 'cookiecutter'
    var_15 = {}
    var_16 = {var_14: var_15}
    var_17 = module_0.Environment()
    var_18 = 'test.txt'
    var_19 = 'cookiecutter'
    var_20 = {}
    var_21 = {var_19: var_20}
    var_22 = module_0.Environment()
    var_23 = True
    var_24 = 'test.txt'
    var_25 = 'cookiecutter'
    var_26 = {}
    var_27 = {var_25: var_26}
    var_28 = module_0.Environment()
    var_29 = 'test.txt'
    var_30 = 'cookiecutter'
    var_31 = '_new_lines'
    var_32 = '\r\n'
    var_33 = {var_31: var_32}
    var_34 = {var_30: var_33}
    var_35 = module_0.Environment()
    var_36 = 'subdir\\template.txt'
    var_37 = 'cookiecutter'
    var_38 = {}
    var_39 = {var_37: var_38}
    var_40 = module_0.Environment()



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'test.txt'
    var_5 = 'Hello {{cookiecutter.name}}'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = (var_7, var_8)
    var_10 = 'name'
    var_11 = 'World'
    var_12 = (var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = False
    var_15 = True

def test_case_0():
    var_0 = 'Test generate_files with subdirectories.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'src'
    var_4 = 'main.py'
    var_5 = '# {{cookiecutter.name}}'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'my_app'
    var_10 = (var_8, var_9)
    var_11 = 'name'
    var_12 = 'MyApp'
    var_13 = (var_11, var_12)
    var_14 = [var_10, var_13]
    var_15 = True

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'old_file.txt'
    var_8 = 'old content'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = (var_10, var_6)
    var_12 = [var_11]
    var_13 = True

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'existing.txt'
    var_4 = 'new {{cookiecutter.name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = (var_7, var_8)
    var_10 = 'name'
    var_11 = 'Test'
    var_12 = (var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = True

def test_case_0():
    var_0 = 'Test generate_files with accept_hooks=False.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = False

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render context.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'



# Parsed testcases at query #20
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John Doe"}'
    var_3 = 'cookiecutter2.json'
    var_4 = '{"project_name": "default_name", "version": "1.0"}'
    var_5 = 'project_name'
    var_6 = 'overridden_name'
    var_7 = {var_5: var_6}
    var_8 = 'cookiecutter3.json'
    var_9 = '{"project_name": "original", "license": "MIT"}'
    var_10 = 'extra_override'
    var_11 = {var_5: var_10}
    var_12 = 'cookiecutter4.json'
    var_13 = '{"python_version": ["3.8", "3.9", "3.10"]}'
    var_14 = 'python_version'
    var_15 = '3.9'
    var_16 = {var_14: var_15}
    var_17 = 'cookiecutter5.json'
    var_18 = '{"features": ["auth", "api", "admin"]}'
    var_19 = 'features'
    var_20 = 'api'
    var_21 = 'admin'
    var_22 = [var_20, var_21]
    var_23 = {var_19: var_22}
    var_24 = 'cookiecutter6.json'
    var_25 = '{"database": {"engine": "postgresql", "port": 5432}}'
    var_26 = 'database'
    var_27 = 'port'
    var_28 = 3306
    var_29 = {var_27: var_28}
    var_30 = {var_26: var_29}
    var_31 = 'cookiecutter7.json'
    var_32 = '{"use_ci": true}'
    var_33 = 'use_ci'
    var_34 = 'n'
    var_35 = {var_33: var_34}
    var_36 = 'cookiecutter_invalid.json'
    var_37 = '{"invalid": json}'
    var_38 = module_0.generate_context(var_0)
    var_39 = 'cookiecutter8.json'
    var_40 = '{"env": ["dev", "staging", "prod"]}'
    var_41 = 'env'
    var_42 = 'invalid_env'
    var_43 = {var_41: var_42}
    var_44 = module_0.generate_context(var_0, extra_context=var_43)
    var_45 = 'cookiecutter9.json'
    var_46 = '{"options": ["a", "b", "c"]}'
    var_47 = 'options'
    var_48 = 'a'
    var_49 = 'invalid'
    var_50 = [var_48, var_49]
    var_51 = {var_47: var_50}
    var_52 = module_0.generate_context(var_0, extra_context=var_51)
    var_53 = 'cookiecutter10.json'
    var_54 = '{"first": 1, "second": 2, "third": 3}'
    var_55 = 'cookiecutter'
    var_56 = 'cookiecutter11.json'
    var_57 = '{}'



# Parsed testcases at query #21
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test apply_overwrites_to_context function with various scenarios.'
    var_1 = 'key1'
    var_2 = 'original_value'
    var_3 = {var_1: var_2}
    var_4 = 'new_value'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = 'value1'
    var_8 = {var_1: var_7}
    var_9 = 'key2'
    var_10 = 'value2'
    var_11 = {var_9: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_8, var_11)
    var_13 = {var_1: var_7}
    var_14 = {var_9: var_10}
    var_15 = True
    var_16 = module_0.apply_overwrites_to_context(var_13, var_14, in_dictionary_variable=var_15)
    var_17 = 'choices'
    var_18 = 'option1'
    var_19 = 'option2'
    var_20 = 'option3'
    var_21 = [var_18, var_19, var_20]
    var_22 = {var_17: var_21}
    var_23 = [var_19, var_20]
    var_24 = {var_17: var_23}
    var_25 = module_0.apply_overwrites_to_context(var_22, var_24)
    var_26 = [var_18, var_19]
    var_27 = {var_17: var_26}
    var_28 = 'invalid'
    var_29 = [var_28]
    var_30 = {var_17: var_29}
    var_31 = module_0.apply_overwrites_to_context(var_27, var_30)
    var_32 = 'choice'
    var_33 = 'default'
    var_34 = [var_33, var_18, var_19]
    var_35 = {var_32: var_34}
    var_36 = {var_32: var_18}
    var_37 = module_0.apply_overwrites_to_context(var_35, var_36)
    var_38 = [var_18, var_19]
    var_39 = {var_32: var_38}
    var_40 = {var_32: var_28}
    var_41 = module_0.apply_overwrites_to_context(var_39, var_40)
    var_42 = 'nested'
    var_43 = {var_1: var_7, var_9: var_10}
    var_44 = {var_42: var_43}
    var_45 = 'new_value1'
    var_46 = {var_1: var_45}
    var_47 = {var_42: var_46}
    var_48 = module_0.apply_overwrites_to_context(var_44, var_47)
    var_49 = 'flag'
    var_50 = False
    var_51 = {var_49: var_50}
    var_52 = 'y'
    var_53 = {var_49: var_52}
    var_54 = module_0.apply_overwrites_to_context(var_51, var_53)
    var_55 = {var_49: var_15}
    var_56 = 'n'
    var_57 = {var_49: var_56}
    var_58 = module_0.apply_overwrites_to_context(var_55, var_57)
    var_59 = {var_49: var_15}
    var_60 = {var_49: var_28}
    var_61 = module_0.apply_overwrites_to_context(var_59, var_60)
    var_62 = 'items'
    var_63 = 'item1'
    var_64 = 'item2'
    var_65 = [var_63, var_64]
    var_66 = {var_62: var_65}
    var_67 = 'new_item'
    var_68 = [var_67]
    var_69 = {var_62: var_68}
    var_70 = module_0.apply_overwrites_to_context(var_66, var_69, in_dictionary_variable=var_15)
    var_71 = 'config'
    var_72 = 'existing'
    var_73 = 'value'
    var_74 = {var_72: var_73}
    var_75 = {var_71: var_74}
    var_76 = 'new_key'
    var_77 = {var_76: var_4}
    var_78 = {var_71: var_77}
    var_79 = module_0.apply_overwrites_to_context(var_75, var_78)
    var_80 = {var_1: var_7, var_9: var_10}
    var_81 = {}
    var_82 = module_0.apply_overwrites_to_context(var_80, var_81)
    var_83 = 'level1'
    var_84 = 'level2'
    var_85 = 'key'
    var_86 = {var_85: var_73}
    var_87 = {var_84: var_86}
    var_88 = {var_83: var_87}
    var_89 = {var_85: var_4}
    var_90 = {var_84: var_89}
    var_91 = {var_83: var_90}
    var_92 = module_0.apply_overwrites_to_context(var_88, var_91)



# Parsed testcases at query #22
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test apply_overwrites_to_context function with various scenarios.'
    var_1 = 'var1'
    var_2 = 'old_value'
    var_3 = {var_1: var_2}
    var_4 = 'new_value'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = 'value1'
    var_8 = {var_1: var_7}
    var_9 = 'var2'
    var_10 = 'value2'
    var_11 = {var_9: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_8, var_11)
    var_13 = 'nested'
    var_14 = {var_1: var_7}
    var_15 = {var_13: var_14}
    var_16 = {var_9: var_10}
    var_17 = {var_13: var_16}
    var_18 = True
    var_19 = module_0.apply_overwrites_to_context(var_15, var_17, in_dictionary_variable=var_18)
    var_20 = 'choice_var'
    var_21 = 'option1'
    var_22 = 'option2'
    var_23 = 'option3'
    var_24 = [var_21, var_22, var_23]
    var_25 = {var_20: var_24}
    var_26 = {var_20: var_22}
    var_27 = module_0.apply_overwrites_to_context(var_25, var_26)
    var_28 = [var_21, var_22]
    var_29 = {var_20: var_28}
    var_30 = {var_20: var_23}
    var_31 = module_0.apply_overwrites_to_context(var_29, var_30)
    var_32 = 'multi_choice'
    var_33 = 'a'
    var_34 = 'b'
    var_35 = 'c'
    var_36 = [var_33, var_34, var_35]
    var_37 = {var_32: var_36}
    var_38 = [var_34, var_35]
    var_39 = {var_32: var_38}
    var_40 = module_0.apply_overwrites_to_context(var_37, var_39)
    var_41 = [var_33, var_34]
    var_42 = {var_32: var_41}
    var_43 = [var_33, var_35]
    var_44 = {var_32: var_43}
    var_45 = module_0.apply_overwrites_to_context(var_42, var_44)
    var_46 = 'bool_var'
    var_47 = False
    var_48 = {var_46: var_47}
    var_49 = 'y'
    var_50 = {var_46: var_49}
    var_51 = module_0.apply_overwrites_to_context(var_48, var_50)
    var_52 = {var_46: var_18}
    var_53 = 'n'
    var_54 = {var_46: var_53}
    var_55 = module_0.apply_overwrites_to_context(var_52, var_54)
    var_56 = {var_46: var_18}
    var_57 = 'invalid'
    var_58 = {var_46: var_57}
    var_59 = module_0.apply_overwrites_to_context(var_56, var_58)
    var_60 = 'key1'
    var_61 = 'key2'
    var_62 = {var_60: var_7, var_61: var_10}
    var_63 = {var_13: var_62}
    var_64 = 'new_value1'
    var_65 = {var_60: var_64}
    var_66 = {var_13: var_65}
    var_67 = module_0.apply_overwrites_to_context(var_63, var_66)
    var_68 = 'list_var'
    var_69 = [var_33, var_34, var_35]
    var_70 = {var_68: var_69}
    var_71 = {var_13: var_70}
    var_72 = [var_34]
    var_73 = {var_68: var_72}
    var_74 = {var_13: var_73}
    var_75 = module_0.apply_overwrites_to_context(var_71, var_74, in_dictionary_variable=var_18)
    var_76 = {var_1: var_7}
    var_77 = {}
    var_78 = module_0.apply_overwrites_to_context(var_76, var_77)
    var_79 = 'var3'
    var_80 = 'value3'
    var_81 = {var_1: var_7, var_9: var_10, var_79: var_80}
    var_82 = 'new1'
    var_83 = 'new3'
    var_84 = {var_1: var_82, var_79: var_83}
    var_85 = module_0.apply_overwrites_to_context(var_81, var_84)
    var_86 = 'var'
    var_87 = 'string_value'
    var_88 = {var_86: var_87}
    var_89 = 'value'
    var_90 = {var_13: var_89}
    var_91 = {var_86: var_90}
    var_92 = module_0.apply_overwrites_to_context(var_88, var_91)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'README.md'
    var_3 = '# {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}'
    var_4 = 'src'
    var_5 = 'main.py'
    var_6 = '# Project: {{cookiecutter.project_name}}'
    var_7 = 'repo'
    var_8 = 'cookiecutter.json'
    var_9 = '{"project_name": "my_project", "author": "John Doe"}'
    var_10 = 'output'
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = 'cookiecutter'
    var_16 = 'project_name'
    var_17 = 'my_project'
    var_18 = (var_16, var_17)
    var_19 = 'author'
    var_20 = 'John Doe'
    var_21 = (var_19, var_20)
    var_22 = [var_18, var_21]
    var_23 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'file.txt'
    var_3 = 'Content: {{cookiecutter.name}}'
    var_4 = 'repo'
    var_5 = 'cookiecutter.json'
    var_6 = '{"project_name": "proj", "name": "test"}'
    var_7 = 'output'
    var_8 = 'proj'
    var_9 = 'old_file.txt'
    var_10 = 'old content'
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = 'cookiecutter'
    var_16 = 'project_name'
    var_17 = (var_16, var_8)
    var_18 = 'name'
    var_19 = 'test'
    var_20 = (var_18, var_19)
    var_21 = [var_17, var_20]
    var_22 = True
    var_23 = False

def test_case_0():
    var_0 = 'Test generate_files with empty context.'
    var_1 = 'simple_project'
    var_2 = 'file.txt'
    var_3 = 'Static content'
    var_4 = 'repo'
    var_5 = 'cookiecutter.json'
    var_6 = '{"project_name": "default"}'
    var_7 = 'cookiecutter.generate.find_template'
    var_8 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_9 = None
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'output'
    var_12 = False



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function creates project from template.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'README.md'
    var_3 = '# {{cookiecutter.project_name}}\n'
    var_4 = 'file.txt'
    var_5 = 'Project: {{cookiecutter.project_name}}'
    var_6 = 'src'
    var_7 = 'main.py'
    var_8 = '# {{cookiecutter.project_name}}'
    var_9 = 'cookiecutter.json'
    var_10 = '{"project_name": "test_project"}'
    var_11 = 'output'
    var_12 = 'cookiecutter.generate.find_template'
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = 'cookiecutter'
    var_15 = 'project_name'
    var_16 = 'test_project'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = '{{cookiecutter.name}}'
    var_2 = 'file.txt'
    var_3 = 'output'
    var_4 = 'cookiecutter.generate.find_template'
    var_5 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'myproject'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = False
    var_12 = True

def test_case_0():
    var_0 = 'Test generate_files respects _copy_without_render setting.'
    var_1 = '{{cookiecutter.project}}'
    var_2 = 'binary.bin'
    var_3 = b'\x00\x01\x02{{cookiecutter.project}}'
    var_4 = 'rendered.txt'
    var_5 = 'output'
    var_6 = 'cookiecutter.generate.find_template'
    var_7 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_8 = 'cookiecutter'
    var_9 = 'project'
    var_10 = '_copy_without_render'
    var_11 = 'myproject'
    var_12 = '*.bin'
    var_13 = [var_12]
    var_14 = {var_9: var_11, var_10: var_13}
    var_15 = {var_8: var_14}
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = '{{cookiecutter.name}}'
    var_2 = 'file.txt'
    var_3 = 'new content'
    var_4 = 'output'
    var_5 = 'myproject'
    var_6 = 'old content'
    var_7 = 'cookiecutter.generate.find_template'
    var_8 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_9 = 'cookiecutter'
    var_10 = 'name'
    var_11 = {var_10: var_5}
    var_12 = {var_9: var_11}



# Parsed testcases at query #25
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'My Project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with default_context override.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'My Project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'Jane Smith'
    var_9 = {var_3: var_8}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with extra_context override.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'My Project'
    var_5 = '1.0.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = '2.0.0'
    var_9 = {var_3: var_8}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with choice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = 'python_version'
    var_3 = '3.8'
    var_4 = '3.9'
    var_5 = '3.10'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = {var_2: var_4}
    var_10 = 'cookiecutter'

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with multichoice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = 'features'
    var_3 = 'feature1'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = [var_4, var_5]
    var_10 = {var_2: var_9}

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context with invalid choice raises ValueError.'
    var_1 = 'cookiecutter.json'
    var_2 = 'python_version'
    var_3 = '3.8'
    var_4 = '3.9'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = '3.11'
    var_9 = {var_2: var_8}
    var_10 = module_1.generate_context(var_0, extra_context=var_9)

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context with invalid multichoice raises ValueError.'
    var_1 = 'cookiecutter.json'
    var_2 = 'features'
    var_3 = 'feature1'
    var_4 = 'feature2'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'feature3'
    var_9 = [var_3, var_8]
    var_10 = {var_2: var_9}
    var_11 = module_1.generate_context(var_0, extra_context=var_10)

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary variable.'
    var_1 = 'cookiecutter.json'
    var_2 = 'config'
    var_3 = 'debug'
    var_4 = 'port'
    var_5 = True
    var_6 = 8000
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 9000
    var_11 = {var_4: var_10}
    var_12 = {var_2: var_11}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with boolean variable.'
    var_1 = 'cookiecutter.json'
    var_2 = 'use_docker'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = 'n'
    var_7 = {var_2: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON raises ContextDecodingException.'
    var_1 = 'cookiecutter.json'
    var_2 = '{invalid json}'
    var_3 = module_0.generate_context(var_0)

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context with invalid boolean conversion raises ValueError.'
    var_1 = 'cookiecutter.json'
    var_2 = 'use_docker'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = 'maybe'
    var_7 = {var_2: var_6}
    var_8 = module_1.generate_context(var_0, extra_context=var_7)

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context with invalid default_context issues warning.'
    var_1 = 'cookiecutter.json'
    var_2 = 'version'
    var_3 = '1.0'
    var_4 = '2.0'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = '3.0'
    var_9 = {var_2: var_8}
    var_10 = module_1.generate_context(var_0, var_9)

def test_case_0():
    var_0 = 'Test that generate_context preserves key order.'
    var_1 = 'cookiecutter.json'
    var_2 = 'first'
    var_3 = 'value1'
    var_4 = (var_2, var_3)
    var_5 = 'second'
    var_6 = 'value2'
    var_7 = (var_5, var_6)
    var_8 = 'third'
    var_9 = 'value3'
    var_10 = (var_8, var_9)
    var_11 = [var_4, var_7, var_10]
    var_12 = 'cookiecutter'



# Parsed testcases at query #26
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_files function with various scenarios.'
    var_1 = '/project/dir'
    var_2 = True
    var_3 = '.'
    var_4 = 'subdir'
    var_5 = [var_4]
    var_6 = 'file.txt'
    var_7 = [var_6]
    var_8 = (var_3, var_5, var_7)
    var_9 = None
    var_10 = '/repo'
    var_11 = 'cookiecutter'
    var_12 = 'project_name'
    var_13 = 'test'
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = '/output'
    var_17 = module_0.generate_files(var_10, var_15, var_16)
    assert var_17 == '/project/dir'
    var_18 = '/project/dir'
    var_19 = True
    var_20 = '.'
    var_21 = []
    var_22 = []
    var_23 = (var_20, var_21, var_22)
    var_24 = None
    var_25 = []
    var_26 = '/repo'
    var_27 = '/output'
    var_28 = 'undefined variable'
    var_29 = '/repo'
    var_30 = 'cookiecutter'
    var_31 = {}
    var_32 = {var_30: var_31}
    var_33 = '/output'
    var_34 = module_0.generate_files(var_29, var_32, var_33)
    var_35 = '/project/dir'
    var_36 = True
    var_37 = '.'
    var_38 = []
    var_39 = []
    var_40 = (var_37, var_38, var_39)
    var_41 = None
    var_42 = '/repo'
    var_43 = 'cookiecutter'
    var_44 = {}
    var_45 = {var_43: var_44}
    var_46 = '/output'
    var_47 = False
    var_48 = module_0.generate_files(var_42, var_45, var_46, accept_hooks=var_47)
    assert var_48 == '/project/dir'
    var_49 = '/project/dir'
    var_50 = False
    var_51 = '.'
    var_52 = []
    var_53 = []
    var_54 = (var_51, var_52, var_53)
    var_55 = None
    var_56 = '/repo'
    var_57 = 'cookiecutter'
    var_58 = {}
    var_59 = {var_57: var_58}
    var_60 = '/output'
    var_61 = True
    var_62 = module_0.generate_files(var_56, var_59, var_60, var_61)
    assert var_62 == '/project/dir'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'README.md'
    var_3 = '# {{cookiecutter.project_name}}\n'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = 'cookiecutter.generate.find_template'
    var_10 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_11 = 'output'
    var_12 = False
    var_13 = True

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = 'cookiecutter.generate.find_template'
    var_10 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_11 = 'output'
    var_12 = False
    var_13 = True

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'file.txt'
    var_3 = '{{cookiecutter.content}}'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = (var_5, var_6)
    var_8 = 'content'
    var_9 = 'new content'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = 'cookiecutter.generate.find_template'
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = 'output'
    var_15 = True

def test_case_0():
    var_0 = 'Test generate_files with binary files.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'image.bin'
    var_3 = b'\x89PNG\r\n\x1a\n'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = 'cookiecutter.generate.find_template'
    var_10 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_11 = 'output'

def test_case_0():
    var_0 = 'Test generate_files with empty context.'
    var_1 = 'project'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter.generate.find_template'
    var_5 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_6 = 'output'



# Parsed testcases at query #28
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'my_project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with default_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'my_project'
    var_5 = '1.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'default_project'
    var_9 = {var_2: var_8}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with extra_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'my_project'
    var_5 = '1.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = '2.0'
    var_9 = {var_3: var_8}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{ invalid json }'
    var_3 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with non-existent file.'
    var_1 = 'nonexistent.json'
    var_2 = module_0.generate_context(var_1)

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with choice variable in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = 'GPL'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = {var_2: var_4}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with multichoice variable in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = 'features'
    var_3 = 'feature1'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = [var_3, var_5]
    var_10 = {var_2: var_9}
    var_11 = 'cookiecutter'

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary context.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'my_project'
    var_5 = 'name'
    var_6 = 'email'
    var_7 = 'John'
    var_8 = 'john@example.com'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_2: var_4, var_3: var_9}
    var_11 = module_0.dumps(var_10)
    var_12 = 'newemail@example.com'
    var_13 = {var_6: var_12}
    var_14 = {var_3: var_13}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with boolean variable.'
    var_1 = 'cookiecutter.json'
    var_2 = 'include_tests'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = 'y'
    var_7 = {var_2: var_6}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with custom context filename.'
    var_1 = 'custom_context.json'
    var_2 = 'project'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)

def test_case_0():
    var_0 = 'Test that generate_context preserves order.'
    var_1 = 'cookiecutter.json'
    var_2 = 'first'
    var_3 = 'value1'
    var_4 = (var_2, var_3)
    var_5 = 'second'
    var_6 = 'value2'
    var_7 = (var_5, var_6)
    var_8 = 'third'
    var_9 = 'value3'
    var_10 = (var_8, var_9)
    var_11 = [var_4, var_7, var_10]
    var_12 = 'cookiecutter'

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context with invalid choice in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'GPL'
    var_9 = {var_2: var_8}
    var_10 = module_1.generate_context(var_0, extra_context=var_9)

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid boolean conversion.'
    var_1 = 'cookiecutter.json'
    var_2 = 'include_tests'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = 'invalid_bool'
    var_7 = {var_2: var_6}



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Test generate_file function with various scenarios.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'template'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'author'
    var_7 = 'test_project'
    var_8 = 'Test Author'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'test_{{cookiecutter.project_name}}.txt'
    var_12 = 'Author: {{cookiecutter.author}}\nProject: {{cookiecutter.project_name}}'
    var_13 = 'test_test_project.txt'
    var_14 = 'binary_file.bin'
    var_15 = b'\x89PNG\r\n\x1a\n'
    var_16 = 'newline_test.txt'
    var_17 = 'Line 1\nLine 2\nLine 3'
    var_18 = 'empty_dir_file.txt'
    var_19 = 'content'
    var_20 = 'empty_result'
    var_21 = 'undefined_{{cookiecutter.missing_var}}.txt'
    var_22 = 'syntax_error.txt'
    var_23 = '{% if unclosed %}'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = 'template_repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project"}'
    var_6 = 'README.md'
    var_7 = '# {{cookiecutter.project_name}}\n'
    var_8 = 'src'
    var_9 = 'main.py'
    var_10 = '# Main file for {{cookiecutter.project_name}}'
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = 'cookiecutter'
    var_16 = 'project_name'
    var_17 = 'my_project'
    var_18 = {var_16: var_17}
    var_19 = {var_15: var_18}
    var_20 = False

def test_case_0():
    var_0 = 'Test generate_files with binary files.'
    var_1 = 'template_repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project"}'
    var_6 = 'image.png'
    var_7 = b'\x89PNG\r\n\x1a\n'
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_10 = None
    var_11 = lambda *args, **kwargs: var_10
    var_12 = 'cookiecutter'
    var_13 = 'project_name'
    var_14 = 'my_project'
    var_15 = {var_13: var_14}
    var_16 = {var_12: var_15}
    var_17 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists.'
    var_1 = 'template_repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project"}'
    var_6 = 'config.txt'
    var_7 = 'config={{cookiecutter.project_name}}'
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_10 = None
    var_11 = lambda *args, **kwargs: var_10
    var_12 = 'cookiecutter'
    var_13 = 'project_name'
    var_14 = 'my_project'
    var_15 = {var_13: var_14}
    var_16 = {var_12: var_15}
    var_17 = False
    var_18 = 'modified content'
    var_19 = True

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render.'
    var_1 = 'template_repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project", "_copy_without_render": ["*.txt"]}'
    var_6 = 'data.txt'
    var_7 = 'cookiecutter.generate.find_template'
    var_8 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_9 = None
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'cookiecutter'
    var_12 = 'project_name'
    var_13 = '_copy_without_render'
    var_14 = 'my_project'
    var_15 = '*.txt'
    var_16 = [var_15]
    var_17 = {var_12: var_14, var_13: var_16}
    var_18 = {var_11: var_17}
    var_19 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists.'
    var_1 = 'template_repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'



# Parsed testcases at query #31
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'my_project'
    var_5 = {var_2: var_4, var_3: var_4}
    var_6 = module_0.dumps(var_5)

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with custom JSON filename.'
    var_1 = 'custom_context.json'
    var_2 = 'author'
    var_3 = 'John Doe'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with default_context override.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'default_name'
    var_5 = '1.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'overridden_name'
    var_9 = {var_2: var_8}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with extra_context override.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'license'
    var_4 = 'original'
    var_5 = 'MIT'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'extra_override'
    var_9 = {var_2: var_8}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{ invalid json }'
    var_3 = module_0.generate_context(var_0)

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with choice variable override.'
    var_1 = 'cookiecutter.json'
    var_2 = 'python_version'
    var_3 = '3.8'
    var_4 = '3.9'
    var_5 = '3.10'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = {var_2: var_4}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with multi-choice variable override.'
    var_1 = 'cookiecutter.json'
    var_2 = 'optional_features'
    var_3 = 'feature_a'
    var_4 = 'feature_b'
    var_5 = 'feature_c'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = [var_3, var_5]
    var_10 = {var_2: var_9}
    var_11 = 'cookiecutter'

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary override.'
    var_1 = 'cookiecutter.json'
    var_2 = 'metadata'
    var_3 = 'author'
    var_4 = 'email'
    var_5 = 'Original Author'
    var_6 = 'original@example.com'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'New Author'
    var_11 = {var_3: var_10}
    var_12 = {var_2: var_11}

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with boolean variable string override.'
    var_1 = 'cookiecutter.json'
    var_2 = 'use_docker'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = 'n'
    var_7 = {var_2: var_6}

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context with invalid choice override.'
    var_1 = 'cookiecutter.json'
    var_2 = 'python_version'
    var_3 = '3.8'
    var_4 = '3.9'
    var_5 = '3.10'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = '3.11'
    var_10 = {var_2: var_9}
    var_11 = module_1.generate_context(var_0, extra_context=var_10)

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context with invalid boolean override.'
    var_1 = 'cookiecutter.json'
    var_2 = 'use_docker'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = 'maybe'
    var_7 = {var_2: var_6}
    var_8 = module_1.generate_context(var_0, extra_context=var_7)

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context with invalid default context shows warning.'
    var_1 = 'cookiecutter.json'
    var_2 = 'python_version'
    var_3 = '3.8'
    var_4 = '3.9'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'invalid_version'
    var_9 = {var_2: var_8}
    var_10 = module_1.generate_context(var_0, var_9)

import json as module_0

def test_case_0():
    var_0 = 'Test generate_context with None for optional context parameters.'
    var_1 = 'cookiecutter.json'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = None



# Parsed testcases at query #32
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir function.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = 'existing'
    var_9 = 'cookiecutter'
    var_10 = {}
    var_11 = {var_9: var_10}
    var_12 = module_0.Environment()
    var_13 = 'existing'
    var_14 = False
    var_15 = 'existing'
    var_16 = 'cookiecutter'
    var_17 = {}
    var_18 = {var_16: var_17}
    var_19 = module_0.Environment()
    var_20 = True
    var_21 = 'cookiecutter'
    var_22 = {}
    var_23 = {var_21: var_22}
    var_24 = module_0.Environment()
    var_25 = ''
    var_26 = 'cookiecutter'
    var_27 = {}
    var_28 = {var_26: var_27}
    var_29 = module_0.Environment()
    var_30 = None
    var_31 = 'cookiecutter'
    var_32 = 'company'
    var_33 = 'project'
    var_34 = 'acme'
    var_35 = 'widget'
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = {var_31: var_36}
    var_38 = module_0.Environment()
    var_39 = '{{cookiecutter.company}}_{{cookiecutter.project}}'
    var_40 = 'cookiecutter'
    var_41 = 'name'
    var_42 = 'my-awesome-project'
    var_43 = {var_41: var_42}
    var_44 = {var_40: var_43}
    var_45 = module_0.Environment()
    var_46 = '{{cookiecutter.name}}'
    var_47 = 'cookiecutter'
    var_48 = 'project'
    var_49 = 'test'
    var_50 = {var_48: var_49}
    var_51 = {var_47: var_50}
    var_52 = module_0.Environment()
    var_53 = '{{cookiecutter.project}}'
    var_54 = var_35 / var_49



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'My Project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}

def test_case_0():
    var_0 = 'Test generate_context raises ContextDecodingException for invalid JSON.'
    var_1 = 'cookiecutter.json'
    var_2 = '{invalid json content}'

def test_case_0():
    var_0 = 'Test generate_context applies default_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'use_docker'
    var_5 = 'Default Project'
    var_6 = '1.0.0'
    var_7 = True
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'Override Project'
    var_10 = False
    var_11 = {var_2: var_9, var_4: var_10}

def test_case_0():
    var_0 = 'Test generate_context applies extra_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'Original'
    var_5 = 'Original Author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'Extra Override'
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test generate_context with choice variable (list).'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = 'GPL'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_2: var_4}

def test_case_0():
    var_0 = 'Test generate_context raises ValueError for invalid choice.'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = 'InvalidLicense'
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test generate_context with multichoice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = 'features'
    var_3 = 'feature1'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_6]
    var_8 = {var_2: var_7}
    var_9 = [var_3, var_5]
    var_10 = {var_2: var_9}

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary variable.'
    var_1 = 'cookiecutter.json'
    var_2 = 'config'
    var_3 = 'database'
    var_4 = 'port'
    var_5 = 'postgresql'
    var_6 = 5432
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 3306
    var_10 = {var_4: var_9}
    var_11 = {var_2: var_10}

def test_case_0():
    var_0 = 'Test generate_context preserves OrderedDict order.'
    var_1 = 'cookiecutter.json'
    var_2 = 'first'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = 'second'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = 'third'
    var_9 = 3
    var_10 = (var_8, var_9)
    var_11 = [var_4, var_7, var_10]
    var_12 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test generate_context converts string to boolean for boolean variables.'
    var_1 = 'cookiecutter.json'
    var_2 = 'use_feature'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'n'
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test generate_context with custom context filename.'
    var_1 = 'custom_context.json'
    var_2 = 'custom_var'
    var_3 = 'custom_value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test generate_file function with various scenarios.'
    var_1 = 'project'
    var_2 = 'templates'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'my_project'
    var_7 = 'John Doe'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'test_{{cookiecutter.project_name}}.txt'
    var_11 = 'test_{{cookiecutter.project_name}}.txt'
    var_12 = 'Author: {{cookiecutter.author}}\nProject: {{cookiecutter.project_name}}'
    var_13 = 'test_my_project.txt'
    var_14 = True
    var_15 = 'Modified content'
    var_16 = 'subdir/{{cookiecutter.empty}}.txt'
    var_17 = 'subdir'
    var_18 = '{{cookiecutter.empty}}.txt'
    var_19 = 'content'
    var_20 = 'binary_file.bin'
    var_21 = b'\x89PNG\r\n\x1a\n'
    var_22 = 'newline_test.txt'
    var_23 = 'line1\nline2'
    var_24 = 'perm_test.sh'
    var_25 = "#!/bin/bash\necho 'test'"
    var_26 = 493

def test_case_0():
    var_0 = 'Test generate_file with undefined Jinja2 variable raises error.'
    var_1 = 'project'
    var_2 = 'templates'
    var_3 = 'cookiecutter'
    var_4 = 'defined_var'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test.txt'
    var_9 = '{{undefined_variable}}'

def test_case_0():
    var_0 = 'Test generate_file with invalid Jinja2 syntax raises error.'
    var_1 = 'project'
    var_2 = 'templates'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'test.txt'
    var_7 = '{% if true %}'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test generate_file function for rendering and creating files.'
    var_1 = 'template'
    var_2 = 'output'
    var_3 = '{{cookiecutter.project_name}}.txt'
    var_4 = 'Project: {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'author'
    var_8 = 'my_project'
    var_9 = 'John Doe'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = False
    var_13 = 'my_project.txt'
    var_14 = True
    var_15 = 'image.bin'
    var_16 = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    var_17 = 'test.txt'
    var_18 = 'test'
    var_19 = ''
    var_20 = 'newlines.txt'
    var_21 = 'line1\nline2\nline3\n'
    var_22 = '_new_lines'
    var_23 = '\r\n'
    var_24 = {var_6: var_8, var_7: var_9, var_22: var_23}
    var_25 = {var_5: var_24}
    var_26 = 'script.sh'
    var_27 = "#!/bin/bash\necho 'test'"
    var_28 = 493
    var_29 = 'error.txt'
    var_30 = '{% if unclosed %}'
    var_31 = 'error.txt'
    var_32 = False



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'README.md'
    var_3 = '# {{cookiecutter.project_name}}\n'
    var_4 = 'repo'
    var_5 = 'cookiecutter.json'
    var_6 = '{"project_name": "my_project"}'
    var_7 = 'output'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test_project'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = 'cookiecutter.generate.find_template'
    var_14 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_15 = None
    var_16 = lambda *args, **kwargs: var_15
    var_17 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'test_project'
    var_7 = 'old_file.txt'
    var_8 = 'old content'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = (var_10, var_6)
    var_12 = [var_11]
    var_13 = 'cookiecutter.generate.find_template'
    var_14 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_15 = None
    var_16 = lambda *args, **kwargs: var_15
    var_17 = True
    var_18 = False

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render setting.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'binary.bin'
    var_4 = b'\x00\x01\x02'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = (var_6, var_7)
    var_9 = '_copy_without_render'
    var_10 = [var_3]
    var_11 = (var_9, var_10)
    var_12 = [var_8, var_11]
    var_13 = 'output'
    var_14 = 'cookiecutter.generate.find_template'
    var_15 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_16 = None
    var_17 = lambda *args, **kwargs: var_16
    var_18 = False

def test_case_0():
    var_0 = 'Test generate_files raises UndefinedVariableInTemplate on undefined variable.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.undefined_var}}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = 'output'
    var_9 = 'cookiecutter.generate.find_template'
    var_10 = 'cookiecutter.generate.render_and_create_dir'
    var_11 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_12 = None
    var_13 = lambda *args, **kwargs: var_12
    var_14 = False

def test_case_0():
    var_0 = 'Test generate_files calls hooks when accept_hooks=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = 'output'



# Parsed testcases at query #37
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'my_project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'utf-8'
    var_9 = 'cookiecutter2.json'
    var_10 = 'version'
    var_11 = 'default_project'
    var_12 = '1.0.0'
    var_13 = {var_2: var_11, var_10: var_12}
    var_14 = module_0.dumps(var_13)
    var_15 = 'overridden_project'
    var_16 = {var_2: var_15}
    var_17 = 'cookiecutter3.json'
    var_18 = 'license'
    var_19 = 'original'
    var_20 = 'MIT'
    var_21 = {var_2: var_19, var_18: var_20}
    var_22 = module_0.dumps(var_21)
    var_23 = 'Apache'
    var_24 = {var_18: var_23}
    var_25 = 'invalid.json'
    var_26 = '{invalid json}'
    var_27 = module_1.generate_context(var_0)
    var_28 = 'cookiecutter4.json'
    var_29 = 'python_version'
    var_30 = '3.8'
    var_31 = '3.9'
    var_32 = '3.10'
    var_33 = [var_30, var_31, var_32]
    var_34 = {var_29: var_33}
    var_35 = module_0.dumps(var_34)
    var_36 = {var_29: var_31}
    var_37 = 'cookiecutter5.json'
    var_38 = 'features'
    var_39 = 'auth'
    var_40 = 'api'
    var_41 = 'admin'
    var_42 = 'logging'
    var_43 = [var_39, var_40, var_41, var_42]
    var_44 = {var_38: var_43}
    var_45 = module_0.dumps(var_44)
    var_46 = [var_40, var_42]
    var_47 = {var_38: var_46}
    var_48 = 'cookiecutter6.json'
    var_49 = 'use_docker'
    var_50 = True
    var_51 = {var_49: var_50}
    var_52 = module_0.dumps(var_51)
    var_53 = 'n'
    var_54 = {var_49: var_53}
    var_55 = 'cookiecutter7.json'
    var_56 = 'database'
    var_57 = 'engine'
    var_58 = 'postgresql'
    var_59 = '12'
    var_60 = {var_57: var_58, var_10: var_59}
    var_61 = {var_56: var_60}
    var_62 = module_0.dumps(var_61)
    var_63 = '13'
    var_64 = {var_10: var_63}
    var_65 = {var_56: var_64}
    var_66 = 'cookiecutter8.json'
    var_67 = 'GPL'
    var_68 = [var_20, var_23, var_67]
    var_69 = {var_18: var_68}
    var_70 = module_0.dumps(var_69)
    var_71 = 'BSD'
    var_72 = {var_18: var_71}
    var_73 = module_1.generate_context(var_0, extra_context=var_72)
    var_74 = 'cookiecutter9.json'
    var_75 = 'feature1'
    var_76 = 'feature2'
    var_77 = 'feature3'
    var_78 = [var_75, var_76, var_77]
    var_79 = {var_38: var_78}
    var_80 = module_0.dumps(var_79)
    var_81 = 'feature4'
    var_82 = [var_75, var_81]
    var_83 = {var_38: var_82}
    var_84 = module_1.generate_context(var_0, extra_context=var_83)



# Parsed testcases at query #38
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'author_name'
    var_5 = 'My Project'
    var_6 = "{{ cookiecutter.project_name.lower().replace(' ', '_') }}"
    var_7 = 'John Doe'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'Default Project'
    var_10 = 'Jane Doe'
    var_11 = {var_2: var_9, var_4: var_10}
    var_12 = 'Extra Project'
    var_13 = {var_2: var_12}
    var_14 = 'invalid.json'
    var_15 = '{invalid json content'
    var_16 = module_0.generate_context(var_15)
    var_17 = 'choice_cookiecutter.json'
    var_18 = 'license'
    var_19 = 'MIT'
    var_20 = 'Apache'
    var_21 = 'GPL'
    var_22 = [var_19, var_20, var_21]
    var_23 = {var_18: var_22}
    var_24 = {var_18: var_20}
    var_25 = 'multichoice_cookiecutter.json'
    var_26 = 'features'
    var_27 = 'feature1'
    var_28 = 'feature2'
    var_29 = 'feature3'
    var_30 = [var_27, var_28, var_29]
    var_31 = {var_26: var_30}
    var_32 = [var_28, var_29]
    var_33 = {var_26: var_32}
    var_34 = 'cookiecutter'
    var_35 = 'bool_cookiecutter.json'
    var_36 = 'use_docker'
    var_37 = True
    var_38 = {var_36: var_37}
    var_39 = 'n'
    var_40 = {var_36: var_39}
    var_41 = 'nested_cookiecutter.json'
    var_42 = 'author'
    var_43 = 'name'
    var_44 = 'email'
    var_45 = 'john@example.com'
    var_46 = {var_43: var_7, var_44: var_45}
    var_47 = {var_42: var_46}
    var_48 = 'jane@example.com'
    var_49 = {var_44: var_48}
    var_50 = {var_42: var_49}



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'Test generate_file function with various scenarios.'
    var_1 = 'test_{{ cookiecutter.project_name }}.txt'
    var_2 = 'Hello {{ cookiecutter.project_name }}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = '_new_lines'
    var_6 = 'myproject'
    var_7 = None
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}

def test_case_0():
    var_0 = 'Test generate_file with binary files.'
    var_1 = 'binary_{{ cookiecutter.name }}.bin'
    var_2 = b'\x89PNG\r\n\x1a\n'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_new_lines'
    var_6 = 'test'
    var_7 = None
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}

def test_case_0():
    var_0 = 'Test generate_file skips existing files when flag is set.'
    var_1 = 'existing_file.txt'
    var_2 = 'existing content'
    var_3 = 'template.txt'
    var_4 = 'new content'
    var_5 = 'cookiecutter'
    var_6 = '_new_lines'
    var_7 = None
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = '{{ cookiecutter.name }}'
    var_11 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test generate_file handles empty directory names.'
    var_1 = 'test.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()

def test_case_0():
    var_0 = 'Test generate_file uses custom newline character from context.'
    var_1 = 'test.txt'
    var_2 = 'line1\nline2\n'
    var_3 = 'cookiecutter'
    var_4 = '_new_lines'
    var_5 = '\r\n'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test generate_file raises TemplateSyntaxError on invalid template.'
    var_1 = 'invalid.txt'
    var_2 = '{% if invalid'
    var_3 = 'cookiecutter'
    var_4 = '_new_lines'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test generate_file handles mixed line endings.'
    var_1 = 'mixed.txt'
    var_2 = 'line1\nline2\r\nline3'
    var_3 = 'cookiecutter'
    var_4 = '_new_lines'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #40
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'author_name'
    var_5 = 'My Project'
    var_6 = "{{ cookiecutter.project_name.lower().replace(' ', '_') }}"
    var_7 = 'John Doe'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'cookiecutter2.json'
    var_10 = 'author'
    var_11 = 'Original'
    var_12 = 'Original Author'
    var_13 = {var_2: var_11, var_10: var_12}
    var_14 = 'Overridden'
    var_15 = {var_2: var_14}
    var_16 = 'cookiecutter3.json'
    var_17 = 'version'
    var_18 = '1.0.0'
    var_19 = {var_2: var_11, var_17: var_18}
    var_20 = '2.0.0'
    var_21 = {var_17: var_20}
    var_22 = 'cookiecutter4.json'
    var_23 = 'license'
    var_24 = 'MIT'
    var_25 = 'Apache'
    var_26 = 'BSD'
    var_27 = [var_24, var_25, var_26]
    var_28 = {var_23: var_27}
    var_29 = {var_23: var_25}
    var_30 = 'cookiecutter5.json'
    var_31 = 'features'
    var_32 = 'feature1'
    var_33 = 'feature2'
    var_34 = 'feature3'
    var_35 = [var_32, var_33, var_34]
    var_36 = {var_31: var_35}
    var_37 = [var_33, var_34]
    var_38 = {var_31: var_37}
    var_39 = 'cookiecutter6.json'
    var_40 = 'config'
    var_41 = 'debug'
    var_42 = 'timeout'
    var_43 = True
    var_44 = 30
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = {var_40: var_45}
    var_47 = False
    var_48 = {var_41: var_47}
    var_49 = {var_40: var_48}
    var_50 = 'cookiecutter7.json'
    var_51 = 'use_docker'
    var_52 = {var_51: var_43}
    var_53 = 'n'
    var_54 = {var_51: var_53}
    var_55 = 'invalid.json'
    var_56 = '{invalid json content'
    var_57 = module_0.generate_context(var_56)
    var_58 = 'cookiecutter9.json'
    var_59 = [var_24, var_25]
    var_60 = {var_23: var_59}
    var_61 = 'GPL'
    var_62 = {var_23: var_61}
    var_63 = module_0.generate_context(var_56, extra_context=var_62)
    var_64 = 'cookiecutter10.json'
    var_65 = [var_32, var_33]
    var_66 = {var_31: var_65}
    var_67 = [var_32, var_34]
    var_68 = {var_31: var_67}
    var_69 = module_0.generate_context(var_56, extra_context=var_68)
    var_70 = 'cookiecutter11.json'
    var_71 = 'use_feature'
    var_72 = {var_71: var_43}
    var_73 = 'maybe'
    var_74 = {var_71: var_73}
    var_75 = module_0.generate_context(var_56, extra_context=var_74)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir function with various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False
    var_9 = 'my_project'
    var_10 = var_2 / var_9
    var_11 = ''
    var_12 = ''
    var_13 = 'existing'
    var_14 = 'existing'
    var_15 = False
    var_16 = 'existing'
    var_17 = True
    var_18 = 'cookiecutter'
    var_19 = 'org'
    var_20 = 'project'
    var_21 = 'acme'
    var_22 = 'widget'
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = {var_18: var_23}
    var_25 = '{{cookiecutter.org}}_{{cookiecutter.project}}'
    var_26 = False
    var_27 = 'acme_widget'
    var_28 = 'parent/{{cookiecutter.project_name}}'
    var_29 = False
    var_30 = 'parent'
    var_31 = var_19 / var_30
    var_32 = 'my_project'
    var_33 = var_31 / var_32
    var_34 = 'test_project'
    var_35 = False



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test generate_file function with various input scenarios.'
    var_1 = 'templates'
    var_2 = 'project'
    var_3 = 'test_{{cookiecutter.project_name}}.txt'
    var_4 = 'Project: {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'author'
    var_8 = 'my_project'
    var_9 = 'John Doe'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = 'test_my_project.txt'

def test_case_0():
    var_0 = 'Test generate_file function with binary file.'
    var_1 = 'templates'
    var_2 = 'project'
    var_3 = 'image.bin'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = b'fake image data'
    var_6 = var_4 + var_5
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}

def test_case_0():
    var_0 = 'Test generate_file with skip_if_file_exists flag.'
    var_1 = 'templates'
    var_2 = 'project'
    var_3 = 'existing.txt'
    var_4 = 'Template content: {{cookiecutter.value}}'
    var_5 = 'Existing content'
    var_6 = 'cookiecutter'
    var_7 = 'value'
    var_8 = 'new_value'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = True

def test_case_0():
    var_0 = 'Test generate_file when rendered filename is empty/directory.'
    var_1 = 'templates'
    var_2 = 'project'
    var_3 = 'test.txt'
    var_4 = 'content'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 'Test generate_file respects custom newline configuration.'
    var_1 = 'templates'
    var_2 = 'project'
    var_3 = 'test.txt'
    var_4 = 'Line 1\nLine 2\nLine 3'
    var_5 = 'cookiecutter'
    var_6 = '_new_lines'
    var_7 = '\r\n'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}

def test_case_0():
    var_0 = 'Test generate_file raises TemplateSyntaxError for invalid templates.'
    var_1 = 'templates'
    var_2 = 'project'
    var_3 = 'bad.txt'
    var_4 = '{{cookiecutter.value | undefined_filter}}'
    var_5 = 'cookiecutter'
    var_6 = 'value'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'bad.txt'

def test_case_0():
    var_0 = 'Test generate_file preserves file permissions from template.'
    var_1 = 'templates'
    var_2 = 'project'
    var_3 = 'script.sh'
    var_4 = '#!/bin/bash\necho {{cookiecutter.message}}'
    var_5 = 493
    var_6 = 'cookiecutter'
    var_7 = 'message'
    var_8 = 'hello'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}



# Parsed testcases at query #3
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test apply_overwrites_to_context function with various scenarios.'
    var_1 = 'var1'
    var_2 = 'var2'
    var_3 = 'original'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'overwritten'
    var_7 = {var_1: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = 'value1'
    var_10 = {var_1: var_9}
    var_11 = 'new_var'
    var_12 = 'new_value'
    var_13 = {var_11: var_12}
    var_14 = module_0.apply_overwrites_to_context(var_10, var_13)
    var_15 = 'nested'
    var_16 = 'existing'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = {var_15: var_18}
    var_20 = 'new_key'
    var_21 = {var_20: var_12}
    var_22 = {var_15: var_21}
    var_23 = module_0.apply_overwrites_to_context(var_19, var_22)
    var_24 = 'choice'
    var_25 = 'option1'
    var_26 = 'option2'
    var_27 = 'option3'
    var_28 = [var_25, var_26, var_27]
    var_29 = {var_24: var_28}
    var_30 = {var_24: var_26}
    var_31 = module_0.apply_overwrites_to_context(var_29, var_30)
    var_32 = [var_25, var_26]
    var_33 = {var_24: var_32}
    var_34 = 'invalid_option'
    var_35 = {var_24: var_34}
    var_36 = module_0.apply_overwrites_to_context(var_33, var_35)
    var_37 = 'multichoice'
    var_38 = 'opt1'
    var_39 = 'opt2'
    var_40 = 'opt3'
    var_41 = [var_38, var_39, var_40]
    var_42 = {var_37: var_41}
    var_43 = [var_38, var_40]
    var_44 = {var_37: var_43}
    var_45 = module_0.apply_overwrites_to_context(var_42, var_44)
    var_46 = [var_38, var_39]
    var_47 = {var_37: var_46}
    var_48 = 'invalid'
    var_49 = [var_38, var_48]
    var_50 = {var_37: var_49}
    var_51 = module_0.apply_overwrites_to_context(var_47, var_50)
    var_52 = 'bool_var'
    var_53 = True
    var_54 = {var_52: var_53}
    var_55 = 'y'
    var_56 = {var_52: var_55}
    var_57 = module_0.apply_overwrites_to_context(var_54, var_56)
    var_58 = {var_52: var_53}
    var_59 = 'n'
    var_60 = {var_52: var_59}
    var_61 = module_0.apply_overwrites_to_context(var_58, var_60)
    var_62 = {var_52: var_53}
    var_63 = {var_52: var_48}
    var_64 = module_0.apply_overwrites_to_context(var_62, var_63)
    var_65 = 'key1'
    var_66 = 'key2'
    var_67 = {var_65: var_9, var_66: var_4}
    var_68 = {var_15: var_67}
    var_69 = 'new_value1'
    var_70 = {var_65: var_69}
    var_71 = {var_15: var_70}
    var_72 = module_0.apply_overwrites_to_context(var_68, var_71)
    var_73 = 'dict_var'
    var_74 = 'list_var'
    var_75 = 'a'
    var_76 = 'b'
    var_77 = 'c'
    var_78 = [var_75, var_76, var_77]
    var_79 = {var_74: var_78}
    var_80 = {var_73: var_79}
    var_81 = 'x'
    var_82 = [var_81, var_55]
    var_83 = {var_74: var_82}
    var_84 = {var_73: var_83}
    var_85 = module_0.apply_overwrites_to_context(var_80, var_84, in_dictionary_variable=var_53)
    var_86 = {var_1: var_9}
    var_87 = {}
    var_88 = module_0.apply_overwrites_to_context(var_86, var_87)
    var_89 = 'level1'
    var_90 = 'level2'
    var_91 = 'var'
    var_92 = {var_91: var_3}
    var_93 = {var_90: var_92}
    var_94 = {var_89: var_93}
    var_95 = 'modified'
    var_96 = {var_91: var_95}
    var_97 = {var_90: var_96}
    var_98 = {var_89: var_97}
    var_99 = module_0.apply_overwrites_to_context(var_94, var_98)
    var_100 = 'first'
    var_101 = 'second'
    var_102 = 'third'
    var_103 = [var_100, var_101, var_102]
    var_104 = {var_24: var_103}
    var_105 = {var_24: var_102}
    var_106 = module_0.apply_overwrites_to_context(var_104, var_105)
    var_107 = var_104[var_24]
    var_108 = len(var_107)
    assert var_108 == 3



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'README.md'
    var_3 = '# {{cookiecutter.project_name}}\n'
    var_4 = 'repo'
    var_5 = 'cookiecutter.json'
    var_6 = '{"project_name": "my_project"}'
    var_7 = 'output'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'my_project'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = 'cookiecutter.generate.find_template'
    var_14 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_15 = None
    var_16 = lambda *args, **kwargs: var_15
    var_17 = False
    var_18 = True

def test_case_0():
    var_0 = 'Test generate_files with nested directory structure.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'src'
    var_3 = 'main.py'
    var_4 = '# {{cookiecutter.project_name}}\n'
    var_5 = 'repo'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_project'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = 'cookiecutter.generate.find_template'
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = None
    var_15 = lambda *args, **kwargs: var_14
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'repo'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'old content'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = (var_9, var_6)
    var_11 = [var_10]
    var_12 = 'cookiecutter.generate.find_template'
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = None
    var_15 = lambda *args, **kwargs: var_14
    var_16 = True
    var_17 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'file.txt'
    var_3 = 'new content'
    var_4 = 'repo'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = True
    var_16 = False



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project", "author": "John Doe"}'
    var_6 = 'utf-8'
    var_7 = 'README.md'
    var_8 = '# {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}'
    var_9 = 'src'
    var_10 = 'main.py'
    var_11 = '# {{cookiecutter.project_name}} main module'
    var_12 = 'project_name'
    var_13 = 'author'
    var_14 = 'test_project'
    var_15 = 'Jane Smith'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project"}'
    var_6 = 'utf-8'
    var_7 = 'file.txt'
    var_8 = 'content'
    var_9 = False
    var_10 = True

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project"}'
    var_6 = 'utf-8'
    var_7 = 'file.txt'
    var_8 = 'original content'
    var_9 = False
    var_10 = 'modified content'
    var_11 = True

def test_case_0():
    var_0 = 'Test generate_files with binary files.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project"}'
    var_6 = 'utf-8'
    var_7 = 'image.png'
    var_8 = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    var_9 = False

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render setting.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'



# Parsed testcases at query #6
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    assert var_3 == 1
    var_4 = 'author_name'
    var_5 = 'My Project'
    var_6 = "{{ cookiecutter.project_name.lower().replace(' ', '_') }}"
    var_7 = 'John Doe'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'cookiecutter2.json'
    var_11 = 'version'
    var_12 = 'Original'
    var_13 = '1.0.0'
    var_14 = {var_2: var_12, var_11: var_13}
    var_15 = module_0.dumps(var_14)
    var_16 = 'Overwritten'
    var_17 = {var_2: var_16}
    var_18 = 'cookiecutter3.json'
    var_19 = 'author'
    var_20 = 'Original Author'
    var_21 = {var_2: var_12, var_19: var_20}
    var_22 = module_0.dumps(var_21)
    var_23 = 'Extra Author'
    var_24 = {var_19: var_23}
    var_25 = 'cookiecutter4.json'
    var_26 = 'license'
    var_27 = 'MIT'
    var_28 = 'Apache'
    var_29 = 'GPL'
    var_30 = [var_27, var_28, var_29]
    var_31 = {var_26: var_30}
    var_32 = module_0.dumps(var_31)
    var_33 = {var_26: var_28}
    var_34 = 'cookiecutter5.json'
    var_35 = 'features'
    var_36 = 'feature1'
    var_37 = 'feature2'
    var_38 = 'feature3'
    var_39 = [var_36, var_37, var_38]
    var_40 = {var_35: var_39}
    var_41 = module_0.dumps(var_40)
    var_42 = [var_36, var_38]
    var_43 = {var_35: var_42}
    var_44 = 'cookiecutter6.json'
    var_45 = 'config'
    var_46 = 'debug'
    var_47 = 'timeout'
    var_48 = False
    var_49 = 30
    var_50 = {var_46: var_48, var_47: var_49}
    var_51 = {var_45: var_50}
    var_52 = module_0.dumps(var_51)
    var_53 = True
    var_54 = {var_46: var_53}
    var_55 = {var_45: var_54}
    var_56 = 'cookiecutter7.json'
    var_57 = '{ invalid json }'
    var_58 = module_1.generate_context(var_0)
    var_59 = 'cookiecutter8.json'
    var_60 = 'choice_var'
    var_61 = 'option1'
    var_62 = 'option2'
    var_63 = [var_61, var_62]
    var_64 = {var_60: var_63}
    var_65 = module_0.dumps(var_64)
    var_66 = 'invalid_option'
    var_67 = {var_60: var_66}
    var_68 = 'always'
    var_69 = module_1.generate_context(var_2, var_67)
    var_70 = 0
    var_71 = var_5.message
    var_72 = str(var_71)
    var_73 = 'cookiecutter9.json'
    var_74 = 'use_feature'
    var_75 = {var_74: var_53}
    var_76 = module_0.dumps(var_75)
    var_77 = 'n'
    var_78 = {var_74: var_77}
    var_79 = 'cookiecutter10.json'
    var_80 = 'z_field'
    var_81 = 'a_field'
    var_82 = 'm_field'
    var_83 = 'last'
    var_84 = 'first'
    var_85 = 'middle'
    var_86 = {var_80: var_83, var_81: var_84, var_82: var_85}
    var_87 = module_0.dumps(var_86)
    var_88 = 'cookiecutter'
    var_89 = var_69[var_88]



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'version'
    var_5 = 'My Project'
    var_6 = 'John Doe'
    var_7 = '1.0.0'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}

def test_case_0():
    var_0 = 'Test generate_context with default_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'use_pytest'
    var_4 = 'My Project'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'Default Project'
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test generate_context with extra_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'My Project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'version'
    var_8 = 'Jane Doe'
    var_9 = '2.0.0'
    var_10 = {var_3: var_8, var_7: var_9}

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{ invalid json }'

def test_case_0():
    var_0 = 'Test that generate_context preserves OrderedDict.'
    var_1 = 'cookiecutter.json'
    var_2 = 'first'
    var_3 = 'value1'
    var_4 = (var_2, var_3)
    var_5 = 'second'
    var_6 = 'value2'
    var_7 = (var_5, var_6)
    var_8 = 'third'
    var_9 = 'value3'
    var_10 = (var_8, var_9)
    var_11 = [var_4, var_7, var_10]
    var_12 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test generate_context with choice variable in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = 'python_version'
    var_3 = '3.8'
    var_4 = '3.9'
    var_5 = '3.10'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test generate_context with boolean variable in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = 'use_docker'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'n'
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary variables.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project'
    var_3 = 'name'
    var_4 = 'version'
    var_5 = 'My Project'
    var_6 = '1.0.0'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '2.0.0'
    var_10 = {var_4: var_9}
    var_11 = {var_2: var_10}

def test_case_0():
    var_0 = 'Test generate_context with custom context filename.'
    var_1 = 'custom_context.json'
    var_2 = 'project_name'
    var_3 = 'Custom Project'
    var_4 = {var_2: var_3}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'author_name'
    var_5 = 'My Project'
    var_6 = 'my_project'
    var_7 = 'John Doe'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}

def test_case_0():
    var_0 = 'Test generate_context with default_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'Default Name'
    var_5 = '1.0.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'Overridden Name'
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test generate_context with extra_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'My Project'
    var_5 = '1.0.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'author'
    var_8 = '2.0.0'
    var_9 = 'Jane Doe'
    var_10 = {var_3: var_8, var_7: var_9}

def test_case_0():
    var_0 = 'Test generate_context with choice variable overwrite.'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = 'GPL'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_2: var_4}

def test_case_0():
    var_0 = 'Test generate_context with multichoice variable overwrite.'
    var_1 = 'cookiecutter.json'
    var_2 = 'features'
    var_3 = 'feature1'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = [var_4, var_5]
    var_9 = {var_2: var_8}

def test_case_0():
    var_0 = 'Test generate_context with boolean variable overwrite.'
    var_1 = 'cookiecutter.json'
    var_2 = 'use_docker'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'n'
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test generate_context with dictionary variable overwrite.'
    var_1 = 'cookiecutter.json'
    var_2 = 'author'
    var_3 = 'name'
    var_4 = 'email'
    var_5 = 'John Doe'
    var_6 = 'john@example.com'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'Jane Doe'
    var_10 = {var_3: var_9}
    var_11 = {var_2: var_10}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{ invalid json }'
    var_3 = module_0.generate_context(var_2)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid choice overwrite.'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = 'BSD'
    var_8 = {var_2: var_7}
    var_9 = module_0.generate_context(var_0, extra_context=var_8)

def test_case_0():
    var_0 = 'Test generate_context with custom context filename.'
    var_1 = 'custom.json'
    var_2 = 'project'
    var_3 = 'Custom Project'
    var_4 = {var_2: var_3}



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = 'template'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "my_project"}'
    var_5 = 'README.md'
    var_6 = '# {{cookiecutter.project_name}}'
    var_7 = 'src'
    var_8 = 'main.py'
    var_9 = 'output'
    var_10 = 'cookiecutter'
    var_11 = 'project_name'
    var_12 = 'my_project'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_16 = 'cookiecutter.generate.find_template'
    var_17 = False
    var_18 = True

def test_case_0():
    var_0 = 'Test generate_files with binary file.'
    var_1 = 'template'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "my_project"}'
    var_5 = 'image.bin'
    var_6 = b'\x89PNG\r\n\x1a\n'
    var_7 = 'output'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'my_project'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = 'cookiecutter.generate.find_template'
    var_15 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'template'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "my_project"}'
    var_5 = 'file.txt'
    var_6 = 'content'
    var_7 = 'output'
    var_8 = 'my_project'
    var_9 = 'old_file.txt'
    var_10 = 'old'
    var_11 = 'cookiecutter'
    var_12 = 'project_name'
    var_13 = {var_12: var_8}
    var_14 = {var_11: var_13}
    var_15 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_16 = 'cookiecutter.generate.find_template'
    var_17 = True
    var_18 = False

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render setting.'
    var_1 = 'template'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "my_project", "_copy_without_render": ["*.bin"]}'
    var_5 = 'template.txt'
    var_6 = '# {{cookiecutter.project_name}}'
    var_7 = 'data.bin'
    var_8 = b'\x00\x01\x02'
    var_9 = 'output'
    var_10 = 'cookiecutter'
    var_11 = 'project_name'
    var_12 = '_copy_without_render'
    var_13 = 'my_project'
    var_14 = '*.bin'
    var_15 = [var_14]
    var_16 = {var_11: var_13, var_12: var_15}
    var_17 = {var_10: var_16}
    var_18 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_19 = 'cookiecutter.generate.find_template'
    var_20 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'template'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "my_project"}'
    var_5 = 'file.txt'
    var_6 = 'new content'
    var_7 = 'output'
    var_8 = 'my_project'
    var_9 = 'old content'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = 'template'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}\n'
    var_5 = 'cookiecutter.json'
    var_6 = '{"project_name": "my_project"}'
    var_7 = 'output'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'my_project'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'template'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Test content'
    var_5 = 'cookiecutter.json'
    var_6 = '{"project_name": "test_project"}'
    var_7 = 'output'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'test_project'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = False
    var_14 = True

def test_case_0():
    var_0 = 'Test generate_files with binary files.'
    var_1 = 'template'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'image.png'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = b'\x00'
    var_6 = 100
    var_7 = var_5 * var_6
    var_8 = var_4 + var_7
    var_9 = 'cookiecutter.json'
    var_10 = '{"project_name": "binary_test"}'
    var_11 = 'output'
    var_12 = 'cookiecutter'
    var_13 = 'project_name'
    var_14 = 'binary_test'
    var_15 = (var_13, var_14)
    var_16 = [var_15]
    var_17 = False
    var_18 = b'\x89PNG'

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'template'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'existing.txt'
    var_4 = 'Template content'
    var_5 = 'cookiecutter.json'
    var_6 = '{"project_name": "skip_test"}'
    var_7 = 'output'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'skip_test'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = False
    var_14 = 'Modified content'
    var_15 = True

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render setting.'
    var_1 = 'template'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = '{{not_rendered}}.txt'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "copy_test", "_copy_without_render": ["*{{*.txt}"]}'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'copy_test'
    var_10 = (var_8, var_9)
    var_11 = '_copy_without_render'
    var_12 = '*{{*.txt}'
    var_13 = [var_12]
    var_14 = (var_11, var_13)
    var_15 = [var_10, var_14]



# Parsed testcases at query #11
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir function.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises error for empty directory name.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises error for empty string directory name.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises error when dir exists and overwrite is False.'
    var_1 = 'existing_project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = {var_3: var_1}
    var_5 = {var_2: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir overwrites when dir exists and overwrite is True.'
    var_1 = 'existing_project'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = {var_3: var_1}
    var_5 = {var_2: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with nested directory path.'
    var_1 = 'cookiecutter'
    var_2 = 'org'
    var_3 = 'project'
    var_4 = 'myorg'
    var_5 = 'myproj'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.Environment()
    var_9 = '{{cookiecutter.org}}/{{cookiecutter.project}}'
    var_10 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with plain directory name (no template variables).'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = 'plain_project'
    var_6 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with special characters in directory name.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my-project_v1.0'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir accepts string output_dir.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = '/path/to/repo'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = (var_2, var_5)
    var_7 = [var_6]
    var_8 = '/output'
    var_9 = '/path/to/repo'
    var_10 = '{{cookiecutter.project_name}}'
    var_11 = '/output/test_project'
    var_12 = True
    var_13 = '.'
    var_14 = []
    var_15 = []
    var_16 = (var_13, var_14, var_15)
    var_17 = None
    var_18 = False

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test generate_files handles UndefinedError in directory creation.'
    var_1 = '/path/to/repo'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = '/output'
    var_7 = '/path/to/repo'
    var_8 = '{{cookiecutter.project_name}}'
    var_9 = 'Undefined variable'
    var_10 = module_0.UndefinedError(var_9)
    var_11 = False

def test_case_0():
    var_0 = 'Test generate_files with accept_hooks=False.'
    var_1 = '/path/to/repo'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = (var_2, var_5)
    var_7 = [var_6]
    var_8 = '/output'
    var_9 = '/path/to/repo'
    var_10 = 'test'
    var_11 = '/output/test'
    var_12 = True
    var_13 = '.'
    var_14 = []
    var_15 = []
    var_16 = (var_13, var_14, var_15)
    var_17 = None
    var_18 = False

def test_case_0():
    var_0 = 'Test generate_files with keep_project_on_failure=True.'
    var_1 = '/path/to/repo'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = (var_2, var_5)
    var_7 = [var_6]
    var_8 = '/output'



# Parsed testcases at query #13
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir function.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'my_project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.Environment()
    var_9 = '{{cookiecutter.project_name}}'
    var_10 = False
    var_11 = True
    var_12 = ''
    var_13 = 'version'
    var_14 = 'test_app'
    var_15 = '1.0.0'
    var_16 = {var_2: var_14, var_13: var_15}
    var_17 = {var_1: var_16}
    var_18 = '{{cookiecutter.project_name}}_v{{cookiecutter.version}}'
    var_19 = 'complex'
    var_20 = 'simple_dir'
    var_21 = 'simple'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function with basic template rendering.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'README.md'
    var_3 = '# {{cookiecutter.project_name}}\n'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project"}'
    var_6 = 'output'
    var_7 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'test.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "test_proj"}'
    var_6 = 'output'
    var_7 = False
    var_8 = True

def test_case_0():
    var_0 = 'Test generate_files raises when directory exists and overwrite is False.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'cookiecutter.json'
    var_3 = '{"project_name": "test_proj"}'
    var_4 = 'output'
    var_5 = False
    var_6 = False

def test_case_0():
    var_0 = 'Test generate_files handles binary files correctly.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'image.bin'
    var_3 = b'\x89PNG\r\n\x1a\n'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_proj"}'
    var_6 = 'output'
    var_7 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'config.txt'
    var_3 = '{{cookiecutter.config}}'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_proj", "config": "new_config"}'
    var_6 = 'output'
    var_7 = False
    var_8 = 'existing_config'
    var_9 = True

def test_case_0():
    var_0 = 'Test generate_files respects _copy_without_render setting.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'static.txt'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "my_proj", "_copy_without_render": ["static.txt"]}'
    var_5 = 'output'



# Parsed testcases at query #15
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir function.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with empty directory name.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with existing directory and no overwrite.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with existing directory and overwrite enabled.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with nested path creation.'
    var_1 = 'cookiecutter'
    var_2 = 'org'
    var_3 = 'project'
    var_4 = 'myorg'
    var_5 = 'myproject'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.Environment()
    var_9 = '{{cookiecutter.org}}/{{cookiecutter.project}}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with special characters in directory name.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my-project_v1'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'



# Parsed testcases at query #16
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir creates directory with rendered name.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test render_and_create_dir raises error for empty directory name.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = ''
    var_6 = '.'
    var_7 = module_1.render_and_create_dir(var_5, var_4, var_6, var_1)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test render_and_create_dir raises error for empty string directory name.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = ''
    var_6 = '.'
    var_7 = module_1.render_and_create_dir(var_5, var_4, var_6, var_1)

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises error when dir exists and overwrite is False.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir overwrites when overwrite_if_exists is True.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with nested template variables.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'org'
    var_4 = 'project'
    var_5 = 'myorg'
    var_6 = 'myapp'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '{{cookiecutter.org}}_{{cookiecutter.project}}'
    var_10 = False
    var_11 = 'myorg_myapp'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir returns False for created when dir already exists.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'existing'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir works with Path objects.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with special characters in template.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'my-project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.name}}'
    var_8 = False



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'README.md'
    var_3 = '# {{cookiecutter.project_name}}\n'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'output'
    var_10 = 'cookiecutter.generate.find_template'
    var_11 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_12 = False
    var_13 = True

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render setting.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'binary.bin'
    var_3 = b'\x00\x01\x02\x03'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_copy_without_render'
    var_7 = 'my_project'
    var_8 = '*.bin'
    var_9 = [var_8]
    var_10 = {var_5: var_7, var_6: var_9}
    var_11 = {var_4: var_10}
    var_12 = 'output'
    var_13 = 'cookiecutter.generate.find_template'
    var_14 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_15 = True

def test_case_0():
    var_0 = 'Test generate_files with nested directory structure.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'src'
    var_3 = '{{cookiecutter.module_name}}'
    var_4 = True
    var_5 = 'main.py'
    var_6 = '# {{cookiecutter.module_name}}\n'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'module_name'
    var_10 = 'my_project'
    var_11 = 'my_module'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {var_7: var_12}
    var_14 = 'output'
    var_15 = 'cookiecutter.generate.find_template'
    var_16 = 'cookiecutter.generate.run_hook_from_repo_dir'

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'test.txt'
    var_3 = 'test content'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'output'
    var_10 = 'old content'
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = True

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'test.txt'
    var_3 = 'new content'



# Parsed testcases at query #18
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test generate_files function with basic project generation.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}\n'
    var_5 = 'cookiecutter.json'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'output'
    var_11 = 'cookiecutter'
    var_12 = {var_6: var_7}
    var_13 = {var_11: var_12}
    var_14 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'myproject'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = False
    var_12 = True

def test_case_0():
    var_0 = 'Test generate_files handles binary files correctly.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'image.bin'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'binproject'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'template content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'skipproject'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = True
    var_12 = False
    var_13 = 'modified content'

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render option.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = '{{no_render}}.txt'
    var_4 = 'output'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = '_copy_without_render'
    var_8 = 'myproject'
    var_9 = [var_3]
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_files raises error for undefined template variables.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = '{{undefined_var}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = False
    var_12 = module_0.generate_files(var_0, var_10, var_1, accept_hooks=var_11)

def test_case_0():
    var_0 = 'Test generate_files with empty context.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'



# Parsed testcases at query #19
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test is_copy_only_path function with various patterns and contexts.'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.pyc'
    var_4 = '__pycache__'
    var_5 = '*.egg-info'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = 'test.pyc'
    var_10 = module_0.is_copy_only_path(var_9, var_8)
    assert var_10 is True
    var_11 = module_0.is_copy_only_path(var_4, var_8)
    assert var_11 is True
    var_12 = 'my_package.egg-info'
    var_13 = module_0.is_copy_only_path(var_12, var_8)
    assert var_13 is True
    var_14 = 'test.py'
    var_15 = module_0.is_copy_only_path(var_14, var_8)
    assert var_15 is False
    var_16 = 'README.md'
    var_17 = module_0.is_copy_only_path(var_16, var_8)
    assert var_17 is False
    var_18 = 'src'
    var_19 = module_0.is_copy_only_path(var_18, var_8)
    assert var_19 is False
    var_20 = '*.bin'
    var_21 = 'static/*'
    var_22 = 'node_modules/**'
    var_23 = [var_20, var_21, var_22]
    var_24 = {var_2: var_23}
    var_25 = {var_1: var_24}
    var_26 = 'file.bin'
    var_27 = module_0.is_copy_only_path(var_26, var_25)
    assert var_27 is True
    var_28 = 'static/css'
    var_29 = module_0.is_copy_only_path(var_28, var_25)
    assert var_29 is True
    var_30 = 'file.txt'
    var_31 = module_0.is_copy_only_path(var_30, var_25)
    assert var_31 is False
    var_32 = []
    var_33 = {var_2: var_32}
    var_34 = {var_1: var_33}
    var_35 = module_0.is_copy_only_path(var_14, var_34)
    assert var_35 is False
    var_36 = {}
    var_37 = {var_1: var_36}
    var_38 = module_0.is_copy_only_path(var_14, var_37)
    assert var_38 is False
    var_39 = {}
    var_40 = module_0.is_copy_only_path(var_14, var_39)
    assert var_40 is False
    var_41 = 'dist/*'
    var_42 = 'build/**/*.o'
    var_43 = [var_41, var_42]
    var_44 = {var_2: var_43}
    var_45 = {var_1: var_44}
    var_46 = 'dist/package.tar.gz'
    var_47 = module_0.is_copy_only_path(var_46, var_45)
    assert var_47 is True
    var_48 = 'build/obj/file.o'
    var_49 = module_0.is_copy_only_path(var_48, var_45)
    assert var_49 is True
    var_50 = 'src/main.py'
    var_51 = module_0.is_copy_only_path(var_50, var_45)
    assert var_51 is False
    var_52 = '{{cookiecutter.project_name}}/*.bin'
    var_53 = '.git/*'
    var_54 = [var_52, var_53]
    var_55 = {var_2: var_54}
    var_56 = {var_1: var_55}
    var_57 = '{{cookiecutter.project_name}}/data.bin'
    var_58 = module_0.is_copy_only_path(var_57, var_56)
    assert var_58 is True
    var_59 = '.git/config'
    var_60 = module_0.is_copy_only_path(var_59, var_56)
    assert var_60 is True



# Parsed testcases at query #20
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'version'
    var_5 = 'My Project'
    var_6 = 'my_project'
    var_7 = '0.1.0'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'utf-8'
    var_11 = 'Default Project'
    var_12 = {var_2: var_11}
    var_13 = '1.0.0'
    var_14 = {var_4: var_13}
    var_15 = {var_2: var_11}
    var_16 = '2.0.0'
    var_17 = {var_4: var_16}
    var_18 = 'bad.json'
    var_19 = '{invalid json}'
    var_20 = module_1.generate_context(var_0)
    var_21 = 'choice_cookiecutter.json'
    var_22 = 'license'
    var_23 = 'MIT'
    var_24 = 'Apache'
    var_25 = 'GPL'
    var_26 = [var_23, var_24, var_25]
    var_27 = {var_22: var_26}
    var_28 = module_0.dumps(var_27)
    var_29 = {var_22: var_24}
    var_30 = 'multi_choice.json'
    var_31 = 'features'
    var_32 = 'feature1'
    var_33 = 'feature2'
    var_34 = 'feature3'
    var_35 = [var_32, var_33, var_34]
    var_36 = {var_31: var_35}
    var_37 = module_0.dumps(var_36)
    var_38 = [var_32, var_34]
    var_39 = {var_31: var_38}
    var_40 = 'bool_cookiecutter.json'
    var_41 = 'use_docker'
    var_42 = True
    var_43 = {var_41: var_42}
    var_44 = module_0.dumps(var_43)
    var_45 = 'n'
    var_46 = {var_41: var_45}
    var_47 = 'nested_cookiecutter.json'
    var_48 = 'author'
    var_49 = 'name'
    var_50 = 'email'
    var_51 = 'John Doe'
    var_52 = 'john@example.com'
    var_53 = {var_49: var_51, var_50: var_52}
    var_54 = {var_48: var_53}
    var_55 = module_0.dumps(var_54)
    var_56 = 'Jane Doe'
    var_57 = {var_49: var_56}
    var_58 = {var_48: var_57}
    var_59 = 'test'
    var_60 = 'value'
    var_61 = {var_59: var_60}
    var_62 = module_0.dumps(var_61)



# Parsed testcases at query #21
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'version'
    var_5 = 'My Project'
    var_6 = "{{ cookiecutter.project_name.lower().replace(' ', '_') }}"
    var_7 = '0.1.0'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'utf-8'
    var_11 = 'invalid.json'
    var_12 = '{invalid json}'
    var_13 = module_1.generate_context(var_0)
    var_14 = 'cookiecutter2.json'
    var_15 = 'name'
    var_16 = 'choice'
    var_17 = 'enabled'
    var_18 = 'default_name'
    var_19 = 'option1'
    var_20 = 'option2'
    var_21 = [var_19, var_20]
    var_22 = True
    var_23 = {var_15: var_18, var_16: var_21, var_17: var_22}
    var_24 = module_0.dumps(var_23)
    var_25 = 'overridden_name'
    var_26 = False
    var_27 = {var_15: var_25, var_16: var_20, var_17: var_26}
    var_28 = 'cookiecutter3.json'
    var_29 = 'author'
    var_30 = 'John Doe'
    var_31 = '1.0.0'
    var_32 = {var_29: var_30, var_4: var_31}
    var_33 = module_0.dumps(var_32)
    var_34 = 'Jane Doe'
    var_35 = '2.0.0'
    var_36 = {var_29: var_34, var_4: var_35}
    var_37 = 'cookiecutter4.json'
    var_38 = 'setting'
    var_39 = 'original'
    var_40 = {var_38: var_39}
    var_41 = module_0.dumps(var_40)
    var_42 = 'default'
    var_43 = {var_38: var_42}
    var_44 = 'extra'
    var_45 = {var_38: var_44}
    var_46 = 'cookiecutter5.json'
    var_47 = 'first'
    var_48 = '1'
    var_49 = (var_47, var_48)
    var_50 = 'second'
    var_51 = '2'
    var_52 = (var_50, var_51)
    var_53 = 'third'
    var_54 = '3'
    var_55 = (var_53, var_54)
    var_56 = [var_49, var_52, var_55]
    var_57 = 'custom.json'
    var_58 = 'key'
    var_59 = 'value'
    var_60 = {var_58: var_59}
    var_61 = module_0.dumps(var_60)
    var_62 = 'cookiecutter6.json'
    var_63 = 'a'
    var_64 = 'b'
    var_65 = [var_63, var_64]
    var_66 = {var_16: var_65}
    var_67 = module_0.dumps(var_66)
    var_68 = 'always'
    var_69 = 'choice'
    var_70 = 'invalid_choice'
    var_71 = {var_69: var_70}
    var_72 = module_1.generate_context(var_2, var_71)
    var_73 = 0
    var_74 = var_9.message
    var_75 = str(var_74)
    var_76 = 'cookiecutter7.json'
    var_77 = 'project'
    var_78 = 'nested'
    var_79 = 'test'
    var_80 = 'deep'
    var_81 = {var_59: var_80}
    var_82 = {var_15: var_79, var_78: var_81}
    var_83 = {var_77: var_82}
    var_84 = module_0.dumps(var_83)
    var_85 = 'cookiecutter8.json'
    var_86 = 'items'
    var_87 = 'item1'
    var_88 = 'item2'
    var_89 = 'item3'
    var_90 = [var_87, var_88, var_89]
    var_91 = {var_86: var_90}
    var_92 = module_0.dumps(var_91)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'My Project'
    var_5 = '{{ cookiecutter.project_name.lower().replace(" ", "_") }}'
    var_6 = {var_2: var_4, var_3: var_5}

def test_case_0():
    var_0 = 'Test generate_context with default_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'use_feature'
    var_5 = 'My Project'
    var_6 = 'Default Author'
    var_7 = True
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'Custom Author'
    var_10 = 'yes'
    var_11 = {var_3: var_9, var_4: var_10}

def test_case_0():
    var_0 = 'Test generate_context with extra_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'My Project'
    var_5 = '0.1.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '1.0.0'
    var_8 = {var_3: var_7}

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{ invalid json }'

def test_case_0():
    var_0 = 'Test that generate_context preserves OrderedDict order.'
    var_1 = 'cookiecutter.json'
    var_2 = 'first_key'
    var_3 = 'value1'
    var_4 = (var_2, var_3)
    var_5 = 'second_key'
    var_6 = 'value2'
    var_7 = (var_5, var_6)
    var_8 = 'third_key'
    var_9 = 'value3'
    var_10 = (var_8, var_9)
    var_11 = [var_4, var_7, var_10]
    var_12 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test generate_context with choice variable overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = 'GPL'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_2: var_4}
    var_9 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project'
    var_3 = 'name'
    var_4 = 'version'
    var_5 = 'My Project'
    var_6 = '0.1.0'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '1.0.0'
    var_10 = {var_4: var_9}
    var_11 = {var_2: var_10}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with non-existent context file.'
    var_1 = '/nonexistent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)

def test_case_0():
    var_0 = 'Test generate_context with empty context.'
    var_1 = 'cookiecutter.json'
    var_2 = {}



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = 'template_repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "my_project"}'
    var_5 = 'README.md'
    var_6 = '# {{cookiecutter.project_name}}\nThis is a test project.'
    var_7 = 'src'
    var_8 = 'main.py'
    var_9 = "print('Hello {{cookiecutter.project_name}}')"
    var_10 = 'output'
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = 'cookiecutter'
    var_16 = 'project_name'
    var_17 = 'test_project'
    var_18 = {var_16: var_17}
    var_19 = (var_15, var_18)
    var_20 = [var_19]
    var_21 = False

def test_case_0():
    var_0 = 'Test generate_files function with hooks enabled.'
    var_1 = 'template_repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "my_project"}'
    var_5 = 'test.txt'
    var_6 = 'test content'
    var_7 = 'output'
    var_8 = []
    var_9 = 'cookiecutter.generate.find_template'
    var_10 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_11 = 'cookiecutter'
    var_12 = 'project_name'
    var_13 = 'test_project'
    var_14 = {var_12: var_13}
    var_15 = (var_11, var_14)
    var_16 = [var_15]
    var_17 = True
    var_18 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'template_repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "my_project"}'
    var_5 = 'test.txt'
    var_6 = 'new content'
    var_7 = 'output'
    var_8 = 'test_project'
    var_9 = 'old_file.txt'
    var_10 = 'old content'
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = 'cookiecutter'
    var_16 = 'project_name'
    var_17 = {var_16: var_8}
    var_18 = (var_15, var_17)
    var_19 = [var_18]
    var_20 = True
    var_21 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'template_repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "my_project"}'
    var_5 = 'config.txt'
    var_6 = 'new config'
    var_7 = 'output'
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_10 = None
    var_11 = lambda *args, **kwargs: var_10
    var_12 = 'cookiecutter'
    var_13 = 'project_name'
    var_14 = 'test_project'
    var_15 = {var_13: var_14}
    var_16 = (var_12, var_15)
    var_17 = [var_16]
    var_18 = True
    var_19 = False



# Parsed testcases at query #24
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test apply_overwrites_to_context function with various scenarios.'
    var_1 = 'var1'
    var_2 = 'original'
    var_3 = {var_1: var_2}
    var_4 = 'new_value'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = 'value1'
    var_8 = {var_1: var_7}
    var_9 = 'var2'
    var_10 = 'value2'
    var_11 = {var_9: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_8, var_11)
    var_13 = 'nested'
    var_14 = 'key1'
    var_15 = 'val1'
    var_16 = {var_14: var_15}
    var_17 = {var_13: var_16}
    var_18 = 'key2'
    var_19 = 'val2'
    var_20 = {var_18: var_19}
    var_21 = {var_13: var_20}
    var_22 = True
    var_23 = module_0.apply_overwrites_to_context(var_17, var_21, in_dictionary_variable=var_22)
    var_24 = 'choice'
    var_25 = 'option1'
    var_26 = 'option2'
    var_27 = 'option3'
    var_28 = [var_25, var_26, var_27]
    var_29 = {var_24: var_28}
    var_30 = {var_24: var_26}
    var_31 = module_0.apply_overwrites_to_context(var_29, var_30)
    var_32 = [var_25, var_26]
    var_33 = {var_24: var_32}
    var_34 = 'invalid_option'
    var_35 = {var_24: var_34}
    var_36 = module_0.apply_overwrites_to_context(var_33, var_35)
    var_37 = 'multichoice'
    var_38 = 'opt1'
    var_39 = 'opt2'
    var_40 = 'opt3'
    var_41 = [var_38, var_39, var_40]
    var_42 = {var_37: var_41}
    var_43 = [var_39, var_40]
    var_44 = {var_37: var_43}
    var_45 = module_0.apply_overwrites_to_context(var_42, var_44)
    var_46 = [var_38, var_39]
    var_47 = {var_37: var_46}
    var_48 = 'invalid'
    var_49 = [var_38, var_48]
    var_50 = {var_37: var_49}
    var_51 = module_0.apply_overwrites_to_context(var_47, var_50)
    var_52 = 'settings'
    var_53 = {var_14: var_15, var_18: var_19}
    var_54 = {var_52: var_53}
    var_55 = 'new_val1'
    var_56 = {var_14: var_55}
    var_57 = {var_52: var_56}
    var_58 = module_0.apply_overwrites_to_context(var_54, var_57)
    var_59 = 'flag'
    var_60 = {var_59: var_22}
    var_61 = 'y'
    var_62 = {var_59: var_61}
    var_63 = module_0.apply_overwrites_to_context(var_60, var_62)
    var_64 = {var_59: var_22}
    var_65 = 'n'
    var_66 = {var_59: var_65}
    var_67 = module_0.apply_overwrites_to_context(var_64, var_66)
    var_68 = {var_59: var_22}
    var_69 = {var_59: var_48}
    var_70 = module_0.apply_overwrites_to_context(var_68, var_69)
    var_71 = 'items'
    var_72 = 'a'
    var_73 = 'b'
    var_74 = 'c'
    var_75 = [var_72, var_73, var_74]
    var_76 = {var_71: var_75}
    var_77 = {var_13: var_76}
    var_78 = 'x'
    var_79 = [var_78, var_61]
    var_80 = {var_71: var_79}
    var_81 = {var_13: var_80}
    var_82 = module_0.apply_overwrites_to_context(var_77, var_81, in_dictionary_variable=var_22)
    var_83 = 'var3'
    var_84 = 'val3'
    var_85 = {var_1: var_15, var_9: var_19, var_83: var_84}
    var_86 = 'new1'
    var_87 = 'new3'
    var_88 = {var_1: var_86, var_83: var_87}
    var_89 = module_0.apply_overwrites_to_context(var_85, var_88)
    var_90 = {var_1: var_15}
    var_91 = {}
    var_92 = module_0.apply_overwrites_to_context(var_90, var_91)
    var_93 = 'level1'
    var_94 = 'level2'
    var_95 = 'level3'
    var_96 = 'value'
    var_97 = {var_95: var_96}
    var_98 = {var_94: var_97}
    var_99 = {var_93: var_98}
    var_100 = {var_95: var_4}
    var_101 = {var_94: var_100}
    var_102 = {var_93: var_101}
    var_103 = module_0.apply_overwrites_to_context(var_99, var_102)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test generate_file function with various scenarios.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_new_lines'
    var_4 = 'test_project'
    var_5 = '\n'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'test_{{ cookiecutter.project_name }}.txt'
    var_9 = 'Hello {{ cookiecutter.project_name }}!'
    var_10 = 'test_test_project.txt'

def test_case_0():
    var_0 = 'Test generate_file with binary files.'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'binary_file.bin'
    var_7 = b'\x89PNG\r\n\x1a\n'

def test_case_0():
    var_0 = 'Test generate_file with skip_if_file_exists flag.'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'existing_file.txt'
    var_7 = 'Original content'
    var_8 = 'Existing content'
    assert var_8 == 'Existing content'
    var_9 = True

def test_case_0():
    var_0 = 'Test generate_file when rendered filename is empty directory.'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test.txt'
    var_7 = 'test'
    var_8 = 'test.txt'
    var_9 = True

def test_case_0():
    var_0 = 'Test generate_file detects and uses file newlines.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'multiline.txt'
    var_5 = 'Line 1\r\nLine 2\r\n'

def test_case_0():
    var_0 = 'Test generate_file with template syntax error.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'bad_syntax.txt'
    var_5 = '{{ unclosed variable'

def test_case_0():
    var_0 = 'Test generate_file uses _new_lines from context.'
    var_1 = 'cookiecutter'
    var_2 = '_new_lines'
    var_3 = '\r\n'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test.txt'
    var_7 = 'Line 1\nLine 2\n'

def test_case_0():
    var_0 = 'Test generate_file preserves file permissions.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'script.sh'



# Parsed testcases at query #26
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John Doe"}'
    var_3 = 'cookiecutter2.json'
    var_4 = '{"project_name": "default_project", "version": "1.0"}'
    var_5 = 'project_name'
    var_6 = 'overridden_project'
    var_7 = {var_5: var_6}
    var_8 = 'cookiecutter3.json'
    var_9 = '{"project_name": "test", "license": "MIT"}'
    var_10 = 'license'
    var_11 = 'Apache'
    var_12 = {var_10: var_11}
    var_13 = 'invalid.json'
    var_14 = '{invalid json}'
    var_15 = module_0.generate_context(var_0)
    var_16 = 'complex.json'
    var_17 = '{"project_name": "complex", "options": {"debug": true, "level": 5}}'
    var_18 = 'choices.json'
    var_19 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_20 = {var_10: var_11}
    var_21 = 'bool.json'
    var_22 = '{"use_docker": true}'
    var_23 = 'use_docker'
    var_24 = 'n'
    var_25 = {var_23: var_24}
    var_26 = 'ordered.json'
    var_27 = '{"a": 1, "b": 2, "c": 3}'
    var_28 = 'cookiecutter'
    var_29 = 'nonexistent.json'
    var_30 = str(var_15)
    var_31 = module_0.generate_context(var_30)
    var_32 = 'empty.json'
    var_33 = '{}'



# Parsed testcases at query #27
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'author_name'
    var_5 = 'My Project'
    var_6 = '{{ cookiecutter.project_name.lower().replace(" ", "_") }}'
    var_7 = 'John Doe'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'Default Project'
    var_11 = 'Jane Doe'
    var_12 = {var_2: var_10, var_4: var_11}
    var_13 = 'Extra Project'
    var_14 = {var_2: var_13}
    var_15 = 'Default'
    var_16 = {var_2: var_15}
    var_17 = 'Extra'
    var_18 = {var_2: var_17}
    var_19 = 'choice_cookiecutter.json'
    var_20 = 'python_version'
    var_21 = '3.9'
    var_22 = '3.10'
    var_23 = '3.11'
    var_24 = [var_21, var_22, var_23]
    var_25 = {var_20: var_24}
    var_26 = module_0.dumps(var_25)
    var_27 = {var_20: var_22}
    var_28 = 'multi_cookiecutter.json'
    var_29 = 'features'
    var_30 = 'feature1'
    var_31 = 'feature2'
    var_32 = 'feature3'
    var_33 = [var_30, var_31, var_32]
    var_34 = {var_29: var_33}
    var_35 = module_0.dumps(var_34)
    var_36 = [var_31, var_32]
    var_37 = {var_29: var_36}
    var_38 = 'cookiecutter'
    var_39 = 'dict_cookiecutter.json'
    var_40 = 'database'
    var_41 = 'engine'
    var_42 = 'port'
    var_43 = 'postgresql'
    var_44 = 5432
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = {var_40: var_45}
    var_47 = module_0.dumps(var_46)
    var_48 = 3306
    var_49 = {var_42: var_48}
    var_50 = {var_40: var_49}
    var_51 = 'bool_cookiecutter.json'
    var_52 = 'use_docker'
    var_53 = False
    var_54 = {var_52: var_53}
    var_55 = module_0.dumps(var_54)
    var_56 = 'y'
    var_57 = {var_52: var_56}
    var_58 = 'invalid.json'
    var_59 = '{invalid json}'
    var_60 = module_1.generate_context(var_0)
    var_61 = 'python_version'
    var_62 = '3.12'
    var_63 = {var_61: var_62}
    var_64 = module_1.generate_context(var_0, extra_context=var_63)
    var_65 = 'features'
    var_66 = 'feature1'
    var_67 = 'invalid_feature'
    var_68 = [var_66, var_67]
    var_69 = {var_65: var_68}
    var_70 = module_1.generate_context(var_0, extra_context=var_69)
    var_71 = 'use_docker'
    var_72 = 'maybe'
    var_73 = {var_71: var_72}
    var_74 = module_1.generate_context(var_0, extra_context=var_73)
    var_75 = 'copy_cookiecutter.json'
    var_76 = '_copy_without_render'
    var_77 = 'Test'
    var_78 = '*.binary'
    var_79 = 'static/*'
    var_80 = [var_78, var_79]
    var_81 = {var_73: var_77, var_76: var_80}
    var_82 = module_0.dumps(var_81)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = 'repo'
    var_2 = 'output'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'README.md'
    var_5 = '# {{cookiecutter.project_name}}\n'
    var_6 = 'cookiecutter.json'
    var_7 = '{"project_name": "test_project"}'
    var_8 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_9 = 'cookiecutter.generate.find_template'
    var_10 = 'cookiecutter'
    var_11 = 'project_name'
    var_12 = 'test_project'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = True

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists.'
    var_1 = 'repo'
    var_2 = 'output'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = 'cookiecutter.generate.find_template'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'my_project'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = False
    var_14 = True

def test_case_0():
    var_0 = 'Test generate_files with empty context.'
    var_1 = 'repo'
    var_2 = 'output'
    var_3 = 'my_project'
    var_4 = 'file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = 'cookiecutter.generate.find_template'
    var_8 = None
    var_9 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists.'
    var_1 = 'repo'
    var_2 = 'output'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'new content'
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = 'cookiecutter.generate.find_template'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'my_project'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = True
    var_14 = False

def test_case_0():
    var_0 = 'Test generate_files with accept_hooks=False.'
    var_1 = 'repo'
    var_2 = 'output'
    var_3 = 'project'
    var_4 = 'file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = 'cookiecutter.generate.find_template'
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = False

def test_case_0():
    var_0 = 'Test generate_files with accept_hooks=True.'
    var_1 = 'repo'
    var_2 = 'output'
    var_3 = 'project'
    var_4 = 'file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = 'cookiecutter.generate.find_template'
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = True

def test_case_0():
    var_0 = 'Test generate_files with undefined variable in directory name.'
    var_1 = 'repo'
    var_2 = 'output'
    var_3 = '{{cookiecutter.undefined_var}}'
    var_4 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_5 = 'cookiecutter.generate.find_template'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}



# Parsed testcases at query #29
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir function.'
    var_1 = 'output'
    var_2 = module_0.Environment()
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'test_dir'
    var_9 = 'output2'
    var_10 = {var_4: var_5}
    var_11 = {var_3: var_10}
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'output3'
    var_14 = 'existing'
    var_15 = 'existing'
    var_16 = False
    var_17 = 'output4'
    var_18 = True
    var_19 = 'output5'
    var_20 = ''
    var_21 = 'output6'
    var_22 = None
    var_23 = 'output7'
    var_24 = 'author'
    var_25 = 'project'
    var_26 = 'john'
    var_27 = 'app'
    var_28 = {var_24: var_26, var_25: var_27}
    var_29 = {var_3: var_28}
    var_30 = '{{cookiecutter.author}}_{{cookiecutter.project}}'
    var_31 = 'output8'
    var_32 = 'parent/child'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function with various scenarios.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = var_2 / var_3
    var_5 = 'output'
    var_6 = 'cookiecutter.json'
    var_7 = var_2 / var_6
    var_8 = 'project_name'
    var_9 = 'author'
    var_10 = 'test_project'
    var_11 = 'Test Author'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'README.md'
    var_14 = var_4 / var_13
    var_15 = '# {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}'
    var_16 = 'cookiecutter'
    var_17 = (var_8, var_10)
    var_18 = (var_9, var_11)
    var_19 = [var_17, var_18]
    var_20 = str(var_2)
    var_21 = False

def test_case_0():
    var_0 = 'Test generate_files with empty context.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'simple_project'
    var_4 = var_2 / var_3
    var_5 = 'output'
    var_6 = 'cookiecutter.json'
    var_7 = var_2 / var_6
    var_8 = 'project_name'
    var_9 = 'default'
    var_10 = {var_8: var_9}
    var_11 = 'file.txt'
    var_12 = var_4 / var_11
    var_13 = 'content'
    var_14 = str(var_2)
    var_15 = None
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = '{{cookiecutter.name}}'
    var_4 = var_2 / var_3
    var_5 = 'output'
    var_6 = 'cookiecutter.json'
    var_7 = var_2 / var_6
    var_8 = 'name'
    var_9 = 'project'
    var_10 = {var_8: var_9}
    var_11 = 'file.txt'
    var_12 = var_4 / var_11
    var_13 = 'content'
    var_14 = 'cookiecutter'
    var_15 = 'name'
    var_16 = 'project'
    var_17 = (var_15, var_16)
    var_18 = [var_17]
    var_19 = str(var_2)
    var_20 = False
    var_21 = str(var_2)
    var_22 = True

def test_case_0():
    var_0 = 'Test generate_files with binary files.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'project'
    var_4 = var_2 / var_3
    var_5 = 'output'
    var_6 = 'cookiecutter.json'
    var_7 = var_2 / var_6
    var_8 = 'name'
    var_9 = 'project'
    var_10 = {var_8: var_9}
    var_11 = 'image.bin'
    var_12 = var_4 / var_11
    var_13 = b'\x89PNG\r\n\x1a\n'
    var_14 = 'cookiecutter'
    var_15 = 'name'
    var_16 = (var_15, var_10)
    var_17 = [var_16]
    var_18 = str(var_2)
    var_19 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'project'
    var_4 = var_2 / var_3
    var_5 = 'output'
    var_6 = 'cookiecutter.json'
    var_7 = var_2 / var_6
    var_8 = 'name'
    var_9 = 'project'
    var_10 = {var_8: var_9}
    var_11 = 'file.txt'
    var_12 = var_4 / var_11
    var_13 = 'original content'
    var_14 = 'cookiecutter'
    var_15 = 'name'
    var_16 = (var_15, var_10)
    var_17 = [var_16]
    var_18 = str(var_2)
    var_19 = True
    var_20 = False

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render context.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'project'
    var_4 = var_2 / var_3
    var_5 = 'output'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'README.md'
    var_3 = '# {{cookiecutter.project_name}}\n'
    var_4 = 'main.py'
    var_5 = "print('{{cookiecutter.greeting}}')\n"
    var_6 = 'src'
    var_7 = '{{cookiecutter.module_name}}.py'
    var_8 = "def hello():\n    return '{{cookiecutter.greeting}}'\n"
    var_9 = 'cookiecutter.json'
    var_10 = '{"project_name": "my_project", "greeting": "Hello", "module_name": "mymodule"}'
    var_11 = 'find_template'
    var_12 = 'run_hook_from_repo_dir'
    var_13 = 'cookiecutter'
    var_14 = 'project_name'
    var_15 = 'test_project'
    var_16 = (var_14, var_15)
    var_17 = 'greeting'
    var_18 = 'Hi'
    var_19 = (var_17, var_18)
    var_20 = 'module_name'
    var_21 = 'testmod'
    var_22 = (var_20, var_21)
    var_23 = [var_16, var_19, var_22]
    var_24 = 'output'
    var_25 = False
    var_26 = 'testmod.py'

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = '{{cookiecutter.name}}'
    var_2 = 'file.txt'
    var_3 = 'content: {{cookiecutter.value}}\n'
    var_4 = 'find_template'
    var_5 = 'run_hook_from_repo_dir'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'myapp'
    var_9 = (var_7, var_8)
    var_10 = 'value'
    var_11 = 'test1'
    var_12 = (var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = 'output'
    var_15 = False
    var_16 = True

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = '{{cookiecutter.name}}'
    var_2 = 'existing.txt'
    var_3 = 'new content\n'
    var_4 = 'find_template'
    var_5 = 'run_hook_from_repo_dir'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'myapp'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = 'output'
    var_12 = True
    var_13 = False
    var_14 = 'old content\n'

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render setting.'
    var_1 = '{{cookiecutter.name}}'



# Parsed testcases at query #32
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_files function with basic template rendering.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}'
    var_5 = 'cookiecutter.json'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'output'
    var_11 = 'cookiecutter'
    var_12 = {var_6: var_7}
    var_13 = {var_11: var_12}
    var_14 = '{'
    var_15 = False
    var_16 = module_1.generate_files(var_14, var_13, var_1, accept_hooks=var_15)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'old_file.txt'
    var_8 = 'old content'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = {var_10: var_6}
    var_12 = {var_9: var_11}
    var_13 = '{'
    var_14 = True
    var_15 = False
    var_16 = module_0.generate_files(var_13, var_12, var_1, var_14, accept_hooks=var_15)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_files with empty context.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'output'
    var_4 = False
    var_5 = module_0.generate_files(var_0, output_dir=var_1, accept_hooks=var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_files executes hooks when accept_hooks is True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = True
    var_10 = module_0.generate_files(var_0, var_8, var_1, accept_hooks=var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'file.txt'
    var_4 = 'new content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = True
    var_12 = False
    var_13 = module_0.generate_files(var_0, var_10, var_1, skip_if_file_exists=var_11, accept_hooks=var_12)

def test_case_0():
    var_0 = 'Test generate_files keeps project directory on failure when flag is set.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #33
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    assert var_3 == 1
    var_4 = 'my_project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'default_project'
    var_9 = {var_2: var_8}
    var_10 = 'Jane Smith'
    var_11 = {var_3: var_10}
    var_12 = {var_2: var_8}
    var_13 = 'extra_project'
    var_14 = {var_2: var_13}
    var_15 = 'invalid.json'
    var_16 = '{invalid json content'
    var_17 = module_1.generate_context(var_0)
    var_18 = 'license'
    var_19 = 'MIT'
    var_20 = 'Apache'
    var_21 = 'GPL'
    var_22 = [var_19, var_20, var_21]
    var_23 = {var_18: var_22}
    var_24 = module_0.dumps(var_23)
    var_25 = {var_18: var_20}
    var_26 = 'features'
    var_27 = 'feature1'
    var_28 = 'feature2'
    var_29 = 'feature3'
    var_30 = [var_27, var_28, var_29]
    var_31 = [var_30]
    var_32 = {var_26: var_31}
    var_33 = module_0.dumps(var_32)
    var_34 = [var_27, var_29]
    var_35 = {var_26: var_34}
    var_36 = 'use_docker'
    var_37 = False
    var_38 = {var_36: var_37}
    var_39 = module_0.dumps(var_38)
    var_40 = 'y'
    var_41 = {var_36: var_40}
    var_42 = 'project'
    var_43 = 'name'
    var_44 = 'version'
    var_45 = 'test'
    var_46 = '1.0'
    var_47 = {var_43: var_45, var_44: var_46}
    var_48 = {var_42: var_47}
    var_49 = module_0.dumps(var_48)
    var_50 = '2.0'
    var_51 = {var_44: var_50}
    var_52 = {var_42: var_51}
    var_53 = [var_19, var_20]
    var_54 = {var_18: var_53}
    var_55 = module_0.dumps(var_54)
    var_56 = 'InvalidChoice'
    var_57 = {var_18: var_56}
    var_58 = 'always'
    var_59 = module_1.generate_context(var_2, var_57)
    var_60 = 0
    var_61 = var_5.message
    var_62 = str(var_61)



# Parsed testcases at query #34
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context function with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'my_project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'cookiecutter2.json'
    var_8 = 'version'
    var_9 = 'default_project'
    var_10 = 'Jane Doe'
    var_11 = '1.0.0'
    var_12 = {var_2: var_9, var_3: var_10, var_8: var_11}
    var_13 = 'Admin'
    var_14 = '2.0.0'
    var_15 = {var_3: var_13, var_8: var_14}
    var_16 = 'cookiecutter3.json'
    var_17 = 'license'
    var_18 = 'base_project'
    var_19 = 'MIT'
    var_20 = {var_2: var_18, var_17: var_19}
    var_21 = 'override_project'
    var_22 = {var_2: var_21}
    var_23 = 'cookiecutter4.json'
    var_24 = 'project_type'
    var_25 = 'web'
    var_26 = 'api'
    var_27 = 'cli'
    var_28 = [var_25, var_26, var_27]
    var_29 = {var_24: var_28}
    var_30 = {var_24: var_26}
    var_31 = 'cookiecutter5.json'
    var_32 = 'features'
    var_33 = 'auth'
    var_34 = 'db'
    var_35 = 'cache'
    var_36 = 'logging'
    var_37 = [var_33, var_34, var_35, var_36]
    var_38 = {var_32: var_37}
    var_39 = [var_33, var_34]
    var_40 = {var_32: var_39}
    var_41 = 'cookiecutter'
    var_42 = 'cookiecutter6.json'
    var_43 = 'config'
    var_44 = 'test'
    var_45 = 'database'
    var_46 = 'port'
    var_47 = 'postgres'
    var_48 = 5432
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = {var_2: var_44, var_43: var_49}
    var_51 = 3306
    var_52 = {var_46: var_51}
    var_53 = {var_43: var_52}
    var_54 = 'invalid.json'
    var_55 = '{invalid json'
    var_56 = module_0.generate_context(var_55)
    var_57 = 'cookiecutter7.json'
    var_58 = 'use_docker'
    var_59 = True
    var_60 = {var_58: var_59}
    var_61 = 'n'
    var_62 = {var_58: var_61}
    var_63 = 'cookiecutter8.json'
    var_64 = 'z_last'
    var_65 = 'a_first'
    var_66 = 'm_middle'
    var_67 = 2
    var_68 = 3
    var_69 = {var_64: var_59, var_65: var_67, var_66: var_68}



# Parsed testcases at query #35
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir creates directory with rendered name.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises error for empty dirname.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises error for empty string dirname.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir raises error when directory exists.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir overwrites directory when flag is True.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test.txt'
    var_8 = 'content'
    var_9 = '{{cookiecutter.project_name}}'
    var_10 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with plain dirname without template variables.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'plain_project'
    var_6 = False
    var_7 = 'plain_project'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with nested path.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'org'
    var_4 = 'project'
    var_5 = 'myorg'
    var_6 = 'myproj'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '{{cookiecutter.org}}/{{cookiecutter.project}}'
    var_10 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir works with pathlib Path as output_dir.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir works with string as output_dir.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir handles special characters in project name.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my-project_v1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False



# Parsed testcases at query #36
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'Test generate_files function with various scenarios.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'output'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = var_2 / var_4
    var_6 = 'test.txt'
    var_7 = var_5 / var_6
    var_8 = 'Hello {{cookiecutter.author}}'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = 'author'
    var_12 = 'my_project'
    var_13 = 'Test Author'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = {var_9: var_14}
    var_16 = str(var_5)
    var_17 = module_0.FileSystemLoader(var_16)
    var_18 = module_1.Environment(loader=var_17)
    var_19 = str(var_2)
    var_20 = False
    var_21 = module_2.generate_files(var_19, var_15, var_17, accept_hooks=var_20)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'output'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = var_2 / var_4
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = str(var_5)
    var_12 = module_0.FileSystemLoader(var_11)
    var_13 = module_1.Environment(loader=var_12)
    var_14 = str(var_2)
    var_15 = False
    var_16 = module_2.generate_files(var_14, var_10, var_12, var_15, accept_hooks=var_15)
    var_17 = str(var_2)
    var_18 = True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'Test generate_files with hooks enabled.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'output'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = var_2 / var_4
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = str(var_5)
    var_12 = module_0.FileSystemLoader(var_11)
    var_13 = module_1.Environment(loader=var_12)
    var_14 = str(var_2)
    var_15 = True
    var_16 = module_2.generate_files(var_14, var_10, var_3, accept_hooks=var_15)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'Test generate_files handles UndefinedError in project directory name.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'output'
    var_4 = '{{cookiecutter.undefined_var}}'
    var_5 = var_2 / var_4
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = str(var_5)
    var_12 = module_0.FileSystemLoader(var_11)
    var_13 = module_1.Environment(loader=var_12)
    var_14 = str(var_2)
    var_15 = False
    var_16 = module_2.generate_files(var_14, var_10, var_12, accept_hooks=var_15)

import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'Test generate_files with empty context.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'output'
    var_4 = 'project'
    var_5 = var_2 / var_4
    var_6 = str(var_5)
    var_7 = module_0.FileSystemLoader(var_6)
    var_8 = module_1.Environment(loader=var_7)
    var_9 = str(var_2)
    var_10 = None
    var_11 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = 'output'



# Parsed testcases at query #37
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir function.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{cookiecutter.project_name}}'
    var_8 = False
    var_9 = False
    var_10 = True
    var_11 = ''
    var_12 = 'author'
    var_13 = 'version'
    var_14 = 'john'
    var_15 = '1.0'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = {var_1: var_16}
    var_18 = '{{cookiecutter.author}}_{{cookiecutter.version}}'
    var_19 = 'nested'
    var_20 = 'path_test'
    var_21 = 'test_dir'
    var_22 = 'str_test'
    var_23 = 'test_dir_str'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Test generate_files function.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project", "author": "John Doe"}'
    var_6 = 'utf-8'
    var_7 = 'README.md'
    var_8 = '# {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}'
    var_9 = 'src'
    var_10 = 'main.py'
    var_11 = '# {{cookiecutter.project_name}} main file'
    var_12 = False
    var_13 = 'my_project'

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project"}'
    var_6 = 'utf-8'
    var_7 = 'test.txt'
    var_8 = 'content'
    var_9 = 'my_project'
    var_10 = 'old_file.txt'
    var_11 = 'old content'
    var_12 = True
    var_13 = False

def test_case_0():
    var_0 = 'Test generate_files raises when directory exists and overwrite_if_exists=False.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project"}'
    var_6 = 'utf-8'
    var_7 = 'test.txt'
    var_8 = 'content'
    var_9 = 'my_project'
    var_10 = False

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render setting.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project", "_copy_without_render": ["*.bin"]}'
    var_6 = 'utf-8'
    var_7 = 'data.bin'
    var_8 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.json'
    var_5 = '{"project_name": "my_project"}'
    var_6 = 'utf-8'
    var_7 = 'config.txt'
    var_8 = 'new_content'
    var_9 = False
    var_10 = 'existing_content'
    var_11 = True

def test_case_0():
    var_0 = 'Test generate_files with context=None.'
    var_1 = 'repo'



# Parsed testcases at query #39
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test apply_overwrites_to_context function with various scenarios.'
    var_1 = 'name'
    var_2 = 'value'
    var_3 = 'original'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'overwritten'
    var_7 = {var_1: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = 'existing'
    var_10 = {var_9: var_2}
    var_11 = 'new_var'
    var_12 = 'new_value'
    var_13 = {var_11: var_12}
    var_14 = module_0.apply_overwrites_to_context(var_10, var_13)
    var_15 = 'choice'
    var_16 = 'option1'
    var_17 = 'option2'
    var_18 = 'option3'
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_15: var_19}
    var_21 = {var_15: var_17}
    var_22 = module_0.apply_overwrites_to_context(var_20, var_21)
    var_23 = [var_16, var_17]
    var_24 = {var_15: var_23}
    var_25 = 'invalid_option'
    var_26 = {var_15: var_25}
    var_27 = module_0.apply_overwrites_to_context(var_24, var_26)
    var_28 = 'multi'
    var_29 = 'a'
    var_30 = 'b'
    var_31 = 'c'
    var_32 = [var_29, var_30, var_31]
    var_33 = {var_28: var_32}
    var_34 = [var_30, var_31]
    var_35 = {var_28: var_34}
    var_36 = module_0.apply_overwrites_to_context(var_33, var_35)
    var_37 = [var_29, var_30]
    var_38 = {var_28: var_37}
    var_39 = 'x'
    var_40 = [var_29, var_39]
    var_41 = {var_28: var_40}
    var_42 = module_0.apply_overwrites_to_context(var_38, var_41)
    var_43 = 'config'
    var_44 = 'key1'
    var_45 = 'key2'
    var_46 = 'value1'
    var_47 = 'value2'
    var_48 = {var_44: var_46, var_45: var_47}
    var_49 = {var_43: var_48}
    var_50 = 'new_value1'
    var_51 = {var_44: var_50}
    var_52 = {var_43: var_51}
    var_53 = module_0.apply_overwrites_to_context(var_49, var_52)
    var_54 = 'enabled'
    var_55 = True
    var_56 = {var_54: var_55}
    var_57 = 'yes'
    var_58 = {var_54: var_57}
    var_59 = module_0.apply_overwrites_to_context(var_56, var_58)
    var_60 = False
    var_61 = {var_54: var_60}
    var_62 = 'no'
    var_63 = {var_54: var_62}
    var_64 = module_0.apply_overwrites_to_context(var_61, var_63)
    var_65 = {var_54: var_55}
    var_66 = 'invalid'
    var_67 = {var_54: var_66}
    var_68 = module_0.apply_overwrites_to_context(var_65, var_67)
    var_69 = {var_9: var_2}
    var_70 = {var_43: var_69}
    var_71 = 'new_key'
    var_72 = {var_71: var_12}
    var_73 = module_0.apply_overwrites_to_context(var_70, var_72, in_dictionary_variable=var_55)
    var_74 = 'outer'
    var_75 = 'inner'
    var_76 = 'deep'
    var_77 = {var_76: var_2}
    var_78 = {var_75: var_77}
    var_79 = {var_74: var_78}
    var_80 = {var_76: var_12}
    var_81 = {var_75: var_80}
    var_82 = {var_74: var_81}
    var_83 = module_0.apply_overwrites_to_context(var_79, var_82)
    var_84 = 'key'
    var_85 = {var_84: var_2}
    var_86 = {}
    var_87 = module_0.apply_overwrites_to_context(var_85, var_86)
    var_88 = 'items'
    var_89 = [var_29, var_30]
    var_90 = {var_88: var_89}
    var_91 = {var_43: var_90}
    var_92 = 'single'
    var_93 = {var_88: var_92}
    var_94 = {var_43: var_93}
    var_95 = module_0.apply_overwrites_to_context(var_91, var_94, in_dictionary_variable=var_55)



