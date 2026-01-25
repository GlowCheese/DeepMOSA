####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = '/home/user/config.yaml'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_3.insert(var_4, var_7)
    var_9 = '/home/user/project/file.py'
    var_10 = var_3.search(var_9)
    var_11 = module_0.Trie()
    var_12 = '/home/config.yaml'
    var_13 = 'level'
    var_14 = 1
    var_15 = {var_13: var_14}
    var_16 = var_11.insert(var_12, var_15)
    var_17 = 2
    var_18 = {var_13: var_17}
    var_19 = var_11.insert(var_4, var_18)
    var_20 = var_11.search(var_9)
    var_21 = module_0.Trie()
    var_22 = {var_5: var_6}
    var_23 = var_21.insert(var_4, var_22)
    var_24 = '/opt/other/file.py'
    var_25 = var_21.search(var_24)
    var_26 = module_0.Trie()
    var_27 = 'exact'
    var_28 = True
    var_29 = {var_27: var_28}
    var_30 = var_26.insert(var_4, var_29)
    var_31 = var_26.search(var_4)
    var_32 = module_0.Trie()
    var_33 = 'root'
    var_34 = True
    var_35 = {var_33: var_34}
    var_36 = var_32.insert(var_12, var_35)
    var_37 = 'user'
    var_38 = True
    var_39 = {var_37: var_38}
    var_40 = var_32.insert(var_4, var_39)
    var_41 = '/home/user/project/config.yaml'
    var_42 = 'project'
    var_43 = True
    var_44 = {var_42: var_43}
    var_45 = var_32.insert(var_41, var_44)
    var_46 = '/home/user/project/src/file.py'
    var_47 = var_32.search(var_46)
    var_48 = module_0.Trie()
    var_49 = 'nested'
    var_50 = 'list'
    var_51 = 'bool'
    var_52 = {var_5: var_6}
    var_53 = 3
    var_54 = [var_43, var_17, var_53]
    var_55 = False
    var_56 = {var_49: var_52, var_50: var_54, var_51: var_55}
    var_57 = '/app/config.yaml'
    var_58 = var_48.insert(var_57, var_56)
    var_59 = '/app/src/module/file.py'
    var_60 = var_48.search(var_59)
    var_61 = module_0.Trie()
    var_62 = True
    var_63 = {var_37: var_62}
    var_64 = var_61.insert(var_4, var_63)
    var_65 = '/home/other/different/file.py'
    var_66 = var_61.search(var_65)
    var_67 = 'root_config.yaml'
    var_68 = True
    var_69 = {var_33: var_68}
    var_70 = module_0.Trie(var_67, var_69)
    var_71 = '/any/path/file.py'
    var_72 = var_70.search(var_71)
    var_73 = module_0.Trie()
    var_74 = '/home/user/config1.yaml'
    var_75 = 'version'
    var_76 = {var_75: var_68}
    var_77 = var_73.insert(var_74, var_76)
    var_78 = '/home/user/config2.yaml'
    var_79 = {var_75: var_17}
    var_80 = var_73.insert(var_78, var_79)
    var_81 = var_73.search(var_9)



# Parsed testcases at query #2
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'setting'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = '/home/user/config.json'
    var_6 = 'value2'
    var_7 = {var_2: var_6}
    var_8 = var_0.insert(var_1, var_4)
    var_9 = var_0.insert(var_5, var_7)
    var_10 = var_0.root
    var_11 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'version'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = 2
    var_7 = {var_2: var_6}
    var_8 = var_0.insert(var_1, var_7)
    var_9 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'nested'
    var_3 = 'list'
    var_4 = 'bool'
    var_5 = 'none'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = True
    var_14 = None
    var_15 = {var_2: var_8, var_3: var_12, var_4: var_13, var_5: var_14}
    var_16 = var_0.insert(var_1, var_15)
    var_17 = var_0.root



# Parsed testcases at query #3
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.search(var_1)
    var_7 = module_0.Trie()
    var_8 = '/home/user/project/config.yaml'
    var_9 = 'setting1'
    var_10 = 'value1'
    var_11 = {var_9: var_10}
    var_12 = '/home/user/project/subdir/config.yaml'
    var_13 = 'setting2'
    var_14 = 'value2'
    var_15 = {var_13: var_14}
    var_16 = var_7.insert(var_8, var_11)
    var_17 = var_7.insert(var_12, var_15)
    var_18 = '/home/user/project/subdir/file.py'
    var_19 = var_7.search(var_18)
    var_20 = '/home/user/project/file.py'
    var_21 = var_7.search(var_20)
    var_22 = module_0.Trie()
    var_23 = '/path/to/config.yaml'
    var_24 = {}
    var_25 = var_22.insert(var_23, var_24)
    var_26 = var_22.search(var_23)
    var_27 = module_0.Trie()
    var_28 = '/home/user/config.yaml'
    var_29 = 'version'
    var_30 = 1
    var_31 = {var_29: var_30}
    var_32 = 2
    var_33 = {var_29: var_32}
    var_34 = var_27.insert(var_28, var_31)
    var_35 = var_27.insert(var_28, var_33)
    var_36 = var_27.search(var_28)
    var_37 = module_0.Trie()
    var_38 = '/a/b/c/d/e/config.yaml'
    var_39 = 'nested'
    var_40 = True
    var_41 = {var_39: var_40}
    var_42 = var_37.insert(var_38, var_41)
    var_43 = '/a/b/c/d/e/file.py'
    var_44 = var_37.search(var_43)



# Parsed testcases at query #4
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'config.yaml'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = None
    var_13 = module_0.TrieNode(var_1, var_12)
    var_14 = module_0.TrieNode()
    var_15 = module_0.TrieNode()
    var_16 = 'level1'
    var_17 = 'list'
    var_18 = 'string'
    var_19 = 'level2'
    var_20 = 'level3'
    var_21 = 1
    var_22 = 2
    var_23 = 3
    var_24 = [var_21, var_22, var_23]
    var_25 = {var_20: var_24}
    var_26 = {var_19: var_25}
    var_27 = [var_21, var_22, var_23]
    var_28 = {var_16: var_26, var_17: var_27, var_18: var_5}
    var_29 = 'complex.yaml'
    var_30 = module_0.TrieNode(var_29, var_28)



# Parsed testcases at query #5
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'setting1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = '/home/user/project/src/config.json'
    var_6 = 'setting2'
    var_7 = 'value2'
    var_8 = {var_6: var_7}
    var_9 = var_0.insert(var_1, var_4)
    var_10 = var_0.insert(var_5, var_8)
    var_11 = var_0.root
    var_12 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'version'
    var_3 = '1'
    var_4 = {var_2: var_3}
    var_5 = '2'
    var_6 = {var_2: var_5}
    var_7 = var_0.insert(var_1, var_4)
    var_8 = var_0.insert(var_1, var_6)
    var_9 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'nested'
    var_3 = 'list'
    var_4 = 'number'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = 42
    var_13 = {var_2: var_7, var_3: var_11, var_4: var_12}
    var_14 = var_0.insert(var_1, var_13)
    var_15 = var_0.root



# Parsed testcases at query #6
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/project/subdir/file.py'
    var_7 = var_0.search(var_6)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config1.json'
    var_2 = 'level'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = '/home/user/project/config2.json'
    var_6 = 2
    var_7 = {var_2: var_6}
    var_8 = '/home/user/project/subdir/config3.json'
    var_9 = 3
    var_10 = {var_2: var_9}
    var_11 = var_0.insert(var_1, var_4)
    var_12 = var_0.insert(var_5, var_7)
    var_13 = var_0.insert(var_8, var_10)
    var_14 = '/home/user/project/subdir/file.py'
    var_15 = var_0.search(var_14)
    var_16 = '/home/user/project/file.py'
    var_17 = var_0.search(var_16)
    var_18 = '/home/user/file.py'
    var_19 = var_0.search(var_18)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'version'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_5}
    var_7 = var_0.insert(var_1, var_4)
    var_8 = var_0.insert(var_1, var_6)
    var_9 = '/home/user/project/file.py'
    var_10 = var_0.search(var_9)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = '/home/user/file.py'
    var_5 = var_0.search(var_4)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/project/config.json'
    var_2 = 'nested'
    var_3 = 'boolean'
    var_4 = 'number'
    var_5 = 'key'
    var_6 = 'list'
    var_7 = 'value'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = {var_5: var_7, var_6: var_11}
    var_13 = True
    var_14 = 42
    var_15 = {var_2: var_12, var_3: var_13, var_4: var_14}
    var_16 = var_0.insert(var_1, var_15)
    var_17 = '/project/file.py'
    var_18 = var_0.search(var_17)



# Parsed testcases at query #7
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'path/to/config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)
    var_13 = module_0.TrieNode()
    var_14 = module_0.TrieNode()
    var_15 = 'config.yaml'
    var_16 = {}
    var_17 = module_0.TrieNode(var_15, var_16)
    var_18 = 'settings'
    var_19 = 'version'
    var_20 = 'debug'
    var_21 = 'timeout'
    var_22 = 'paths'
    var_23 = True
    var_24 = 30
    var_25 = '/path1'
    var_26 = '/path2'
    var_27 = [var_25, var_26]
    var_28 = {var_20: var_23, var_21: var_24, var_22: var_27}
    var_29 = '1.0.0'
    var_30 = {var_18: var_28, var_19: var_29}
    var_31 = 'complex_config.json'
    var_32 = module_0.TrieNode(var_31, var_30)



# Parsed testcases at query #8
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'setting1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = '/home/user/project/subdir/config.json'
    var_6 = 'setting2'
    var_7 = 'value2'
    var_8 = {var_6: var_7}
    var_9 = var_0.insert(var_1, var_4)
    var_10 = var_0.insert(var_5, var_8)
    var_11 = var_0.root
    var_12 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'version'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_5}
    var_7 = var_0.insert(var_1, var_4)
    var_8 = var_0.insert(var_1, var_6)
    var_9 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/a/b/c/d/e/f/config.json'
    var_2 = 'nested'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #9
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'config.yaml'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = 'setting'
    var_13 = 'enabled'
    var_14 = {var_12: var_13}
    var_15 = module_0.TrieNode(config_data=var_14)
    var_16 = module_0.TrieNode()
    var_17 = module_0.TrieNode()
    var_18 = None
    var_19 = module_0.TrieNode(var_1, var_18)
    var_20 = {}
    var_21 = module_0.TrieNode(var_1, var_20)
    var_22 = 'version'
    var_23 = 'settings'
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = 'deep'
    var_29 = 'structure'
    var_30 = True
    var_31 = {var_29: var_30}
    var_32 = {var_28: var_31}
    var_33 = {var_22: var_24, var_23: var_27, var_4: var_32}
    var_34 = 'complex.yaml'
    var_35 = module_0.TrieNode(var_34, var_33)



# Parsed testcases at query #10
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'root_value'
    var_5 = {var_3: var_4}
    var_6 = 'config.toml'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = var_7.search(var_1)
    var_9 = 'level'
    var_10 = 1
    var_11 = {var_9: var_10}
    var_12 = '/home/user/config.toml'
    var_13 = var_7.insert(var_12, var_11)
    var_14 = '/home/user/file.py'
    var_15 = var_7.search(var_14)
    var_16 = module_0.Trie()
    var_17 = 'root'
    var_18 = {var_9: var_17}
    var_19 = 'root_config.toml'
    var_20 = 'parent'
    var_21 = {var_9: var_20}
    var_22 = '/home/config.toml'
    var_23 = var_16.insert(var_22, var_21)
    var_24 = 'child'
    var_25 = {var_9: var_24}
    var_26 = var_16.insert(var_12, var_25)
    var_27 = '/home/user/project/file.py'
    var_28 = var_16.search(var_27)
    var_29 = '/home/other/file.py'
    var_30 = var_16.search(var_29)
    var_31 = '/file.py'
    var_32 = var_16.search(var_31)
    var_33 = module_0.Trie()
    var_34 = 'name'
    var_35 = 'config_a'
    var_36 = {var_34: var_35}
    var_37 = 'config_b'
    var_38 = {var_34: var_37}
    var_39 = 'config_c'
    var_40 = {var_34: var_39}
    var_41 = '/a/config.toml'
    var_42 = var_33.insert(var_41, var_36)
    var_43 = '/a/b/config.toml'
    var_44 = var_33.insert(var_43, var_38)
    var_45 = '/a/b/c/config.toml'
    var_46 = var_33.insert(var_45, var_40)
    var_47 = '/a/b/c/d/file.py'
    var_48 = var_33.search(var_47)
    var_49 = '/a/b/file.py'
    var_50 = var_33.search(var_49)
    var_51 = '/a/file.py'
    var_52 = var_33.search(var_51)
    var_53 = module_0.Trie()
    var_54 = 'test'
    var_55 = 'value'
    var_56 = {var_54: var_55}
    var_57 = '/home/project/config.toml'
    var_58 = var_53.insert(var_57, var_56)
    var_59 = '/other/path/file.py'
    var_60 = var_53.search(var_59)
    var_61 = module_0.Trie()
    var_62 = 'resolved'
    var_63 = True
    var_64 = {var_62: var_63}
    var_65 = '/absolute/path/config.toml'
    var_66 = var_61.insert(var_65, var_64)
    var_67 = '/absolute/path/subdir/file.py'
    var_68 = var_61.search(var_67)



# Parsed testcases at query #11
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = '/config.json'
    var_8 = var_3.insert(var_7, var_6)
    var_9 = '/file.py'
    var_10 = var_3.search(var_9)
    var_11 = module_0.Trie()
    var_12 = 'level'
    var_13 = 1
    var_14 = {var_12: var_13}
    var_15 = 2
    var_16 = {var_12: var_15}
    var_17 = '/home/user/project/config.json'
    var_18 = var_11.insert(var_17, var_14)
    var_19 = '/home/user/project/src/config.json'
    var_20 = var_11.insert(var_19, var_16)
    var_21 = '/home/user/project/src/main.py'
    var_22 = var_11.search(var_21)
    var_23 = module_0.Trie()
    var_24 = 'parent'
    var_25 = {var_12: var_24}
    var_26 = var_23.insert(var_17, var_25)
    var_27 = '/home/user/project/src/nested/file.py'
    var_28 = var_23.search(var_27)
    var_29 = module_0.Trie()
    var_30 = {var_12: var_13}
    var_31 = var_29.insert(var_17, var_30)
    var_32 = '/other/path/file.py'
    var_33 = var_29.search(var_32)
    var_34 = module_0.Trie()
    var_35 = 'name'
    var_36 = 'root'
    var_37 = {var_35: var_36}
    var_38 = 'home'
    var_39 = {var_35: var_38}
    var_40 = 'project'
    var_41 = {var_35: var_40}
    var_42 = var_34.insert(var_7, var_37)
    var_43 = '/home/config.json'
    var_44 = var_34.insert(var_43, var_39)
    var_45 = var_34.insert(var_17, var_41)
    var_46 = '/home/user/project/src/file.py'
    var_47 = var_34.search(var_46)
    var_48 = module_0.Trie()
    var_49 = 'stop'
    var_50 = 'here'
    var_51 = {var_49: var_50}
    var_52 = '/a/b/c/config.json'
    var_53 = var_48.insert(var_52, var_51)
    var_54 = '/a/x/y/z/file.py'
    var_55 = var_48.search(var_54)
    var_56 = module_0.Trie()
    var_57 = True
    var_58 = {var_36: var_57}
    var_59 = var_56.insert(var_7, var_58)
    var_60 = var_56.search(var_9)
    var_61 = module_0.Trie()
    var_62 = 'nested'
    var_63 = 'list'
    var_64 = 'bool'
    var_65 = {var_4: var_5}
    var_66 = 3
    var_67 = [var_57, var_15, var_66]
    var_68 = True
    var_69 = {var_62: var_65, var_63: var_67, var_64: var_68}
    var_70 = '/project/config.json'
    var_71 = var_61.insert(var_70, var_69)
    var_72 = '/project/src/main.py'
    var_73 = var_61.search(var_72)
    var_74 = module_0.Trie()
    var_75 = 'depth'
    var_76 = {var_75: var_68}
    var_77 = {var_75: var_15}
    var_78 = {var_75: var_66}
    var_79 = '/a/config.json'
    var_80 = var_74.insert(var_79, var_76)
    var_81 = '/a/b/config.json'
    var_82 = var_74.insert(var_81, var_77)
    var_83 = var_74.insert(var_52, var_78)
    var_84 = '/a/b/c/d/file.py'
    var_85 = var_74.search(var_84)



# Parsed testcases at query #12
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.config'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = '/path/to/config.json'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = None
    var_13 = module_0.TrieNode(var_1, var_12)
    var_14 = module_0.TrieNode()
    var_15 = 'child.config'
    var_16 = 'child'
    var_17 = {var_16: var_7}
    var_18 = module_0.TrieNode(var_15, var_17)
    var_19 = {}
    var_20 = module_0.TrieNode(var_1, var_19)
    var_21 = 'version'
    var_22 = 'settings'
    var_23 = 'paths'
    var_24 = '1.0'
    var_25 = 'debug'
    var_26 = 'timeout'
    var_27 = True
    var_28 = 30
    var_29 = {var_25: var_27, var_26: var_28}
    var_30 = '/path1'
    var_31 = '/path2'
    var_32 = [var_30, var_31]
    var_33 = {var_21: var_24, var_22: var_29, var_23: var_32}
    var_34 = 'complex.config'
    var_35 = module_0.TrieNode(var_34, var_33)



# Parsed testcases at query #13
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = '/config.yaml'
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/home/user/config.yaml'
    var_11 = 'level'
    var_12 = 1
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/home/user/file.py'
    var_16 = var_9.search(var_15)
    var_17 = module_0.Trie()
    var_18 = {var_11: var_12}
    var_19 = var_17.insert(var_10, var_18)
    var_20 = '/home/user/project/file.py'
    var_21 = var_17.search(var_20)
    var_22 = module_0.Trie()
    var_23 = '/home/config.yaml'
    var_24 = 'home'
    var_25 = {var_11: var_24}
    var_26 = var_22.insert(var_23, var_25)
    var_27 = 'user'
    var_28 = {var_11: var_27}
    var_29 = var_22.insert(var_10, var_28)
    var_30 = var_22.search(var_20)
    var_31 = module_0.Trie()
    var_32 = {var_11: var_12}
    var_33 = var_31.insert(var_3, var_32)
    var_34 = 2
    var_35 = {var_11: var_34}
    var_36 = var_31.insert(var_23, var_35)
    var_37 = 3
    var_38 = {var_11: var_37}
    var_39 = var_31.insert(var_10, var_38)
    var_40 = '/home/user/project/subdir/file.py'
    var_41 = var_31.search(var_40)
    var_42 = module_0.Trie()
    var_43 = {var_11: var_12}
    var_44 = var_42.insert(var_10, var_43)
    var_45 = '/var/log/file.py'
    var_46 = var_42.search(var_45)
    var_47 = module_0.Trie()
    var_48 = {}
    var_49 = var_47.insert(var_23, var_48)
    var_50 = '/home/file.py'
    var_51 = var_47.search(var_50)
    var_52 = 'name'
    var_53 = 'nested'
    var_54 = 'list'
    var_55 = 'test'
    var_56 = 'value'
    var_57 = {var_4: var_56}
    var_58 = [var_12, var_34, var_37]
    var_59 = {var_52: var_55, var_53: var_57, var_54: var_58}
    var_60 = module_0.Trie()
    var_61 = var_60.insert(var_23, var_59)
    var_62 = '/home/subdir/file.py'
    var_63 = var_60.search(var_62)
    var_64 = '/root_config.yaml'
    var_65 = 'root'
    var_66 = True
    var_67 = {var_65: var_66}
    var_68 = module_0.Trie(var_64, var_67)
    var_69 = True
    var_70 = {var_27: var_69}
    var_71 = var_68.insert(var_10, var_70)
    var_72 = var_68.search(var_15)
    var_73 = True
    var_74 = {var_65: var_73}
    var_75 = module_0.Trie(var_64, var_74)
    var_76 = True
    var_77 = {var_27: var_76}
    var_78 = var_75.insert(var_10, var_77)
    var_79 = '/var/file.py'
    var_80 = var_75.search(var_79)
    var_81 = module_0.Trie()
    var_82 = 'single'
    var_83 = True
    var_84 = {var_82: var_83}
    var_85 = var_81.insert(var_3, var_84)
    var_86 = '/file.py'
    var_87 = var_81.search(var_86)



# Parsed testcases at query #14
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = module_0.Trie()
    var_8 = '/var/configs/app.yaml'
    var_9 = {}
    var_10 = var_7.insert(var_8, var_9)
    var_11 = var_7.root
    var_12 = module_0.Trie()
    var_13 = '/home/user/config1.json'
    var_14 = '/home/user/project/config2.json'
    var_15 = 'type'
    var_16 = 'json'
    var_17 = {var_15: var_16}
    var_18 = 'yaml'
    var_19 = {var_15: var_18}
    var_20 = var_12.insert(var_13, var_17)
    var_21 = var_12.insert(var_14, var_19)
    var_22 = var_12.root
    var_23 = var_12.root
    var_24 = module_0.Trie()
    var_25 = '/etc/app/settings/config.json'
    var_26 = 'database'
    var_27 = 'cache'
    var_28 = 'host'
    var_29 = 'port'
    var_30 = 'localhost'
    var_31 = 5432
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = 'enabled'
    var_34 = 'ttl'
    var_35 = True
    var_36 = 3600
    var_37 = {var_33: var_35, var_34: var_36}
    var_38 = {var_26: var_32, var_27: var_37}
    var_39 = var_24.insert(var_25, var_38)
    var_40 = var_24.root
    var_41 = module_0.Trie()
    var_42 = '/home/user/config.json'
    var_43 = 'version'
    var_44 = {var_43: var_35}
    var_45 = 2
    var_46 = {var_43: var_45}
    var_47 = var_41.insert(var_42, var_44)
    var_48 = var_41.root
    var_49 = var_41.insert(var_42, var_46)
    var_50 = var_41.root



# Parsed testcases at query #15
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.yaml'
    var_3 = module_0.Trie(var_2)
    var_4 = 'key'
    var_5 = 'nested'
    var_6 = 'value'
    var_7 = 'inner'
    var_8 = 'data'
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = module_0.Trie(config_data=var_10)
    var_12 = '/path/to/config.yaml'
    var_13 = 'setting'
    var_14 = 'debug'
    var_15 = True
    var_16 = {var_13: var_6, var_14: var_15}
    var_17 = module_0.Trie(var_12, var_16)
    var_18 = 'test.yaml'
    var_19 = 'a'
    var_20 = {var_19: var_15}
    var_21 = module_0.Trie(var_18, var_20)
    var_22 = var_21.root.nodes
    var_23 = var_21.root.nodes
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = 'config.yaml'
    var_26 = {}
    var_27 = module_0.Trie(var_25, var_26)
    var_28 = 'database'
    var_29 = 'logging'
    var_30 = 'features'
    var_31 = 'host'
    var_32 = 'port'
    var_33 = 'localhost'
    var_34 = 5432
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = 'level'
    var_37 = 'INFO'
    var_38 = {var_36: var_37}
    var_39 = 2
    var_40 = 3
    var_41 = [var_15, var_39, var_40]
    var_42 = {var_28: var_35, var_29: var_38, var_30: var_41}
    var_43 = 'app.yaml'
    var_44 = module_0.Trie(var_43, var_42)



# Parsed testcases at query #16
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = var_10.root
    var_12 = module_0.Trie(var_2)
    var_13 = None
    var_14 = module_0.Trie(var_2, var_13)
    var_15 = 'file1.json'
    var_16 = 'a'
    var_17 = 1
    var_18 = {var_16: var_17}
    var_19 = module_0.Trie(var_15, var_18)
    var_20 = 'file2.json'
    var_21 = 'b'
    var_22 = 2
    var_23 = {var_21: var_22}
    var_24 = module_0.Trie(var_20, var_23)



# Parsed testcases at query #17
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/path/file.py'
    var_2 = var_0.search(var_1)
    var_3 = 'root'
    var_4 = True
    var_5 = {var_3: var_4}
    var_6 = '/config.yaml'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = var_7.search(var_6)
    var_9 = module_0.Trie()
    var_10 = 'level'
    var_11 = {var_10: var_4}
    var_12 = 2
    var_13 = {var_10: var_12}
    var_14 = 3
    var_15 = {var_10: var_14}
    var_16 = '/home/user/.config'
    var_17 = var_9.insert(var_16, var_11)
    var_18 = '/home/user/project/.config'
    var_19 = var_9.insert(var_18, var_13)
    var_20 = '/home/user/project/src/.config'
    var_21 = var_9.insert(var_20, var_15)
    var_22 = '/home/user/project/src/module.py'
    var_23 = var_9.search(var_22)
    var_24 = '/home/user/project/main.py'
    var_25 = var_9.search(var_24)
    var_26 = '/home/user/file.py'
    var_27 = var_9.search(var_26)
    var_28 = '/home/user/other/subdir/file.py'
    var_29 = var_9.search(var_28)
    var_30 = module_0.Trie()
    var_31 = '/var/config'
    var_32 = 'var'
    var_33 = {var_32: var_4}
    var_34 = var_30.insert(var_31, var_33)
    var_35 = var_30.search(var_26)
    var_36 = module_0.Trie()
    var_37 = 'name'
    var_38 = 'a'
    var_39 = {var_37: var_38}
    var_40 = 'b'
    var_41 = {var_37: var_40}
    var_42 = var_36.insert(var_16, var_39)
    var_43 = var_36.insert(var_18, var_41)
    var_44 = '/home/user/project/nonexistent/deep/file.py'
    var_45 = var_36.search(var_44)
    var_46 = module_0.Trie()
    var_47 = 'absolute'
    var_48 = {var_47: var_4}
    var_49 = '/absolute/path/.config'
    var_50 = var_46.insert(var_49, var_48)
    var_51 = '/absolute/path/file.py'
    var_52 = var_46.search(var_51)



# Parsed testcases at query #18
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = var_10.root
    var_12 = module_0.Trie(var_2)
    var_13 = ''
    var_14 = {}
    var_15 = module_0.Trie(var_13, var_14)
    var_16 = module_0.Trie()
    var_17 = module_0.Trie()



# Parsed testcases at query #19
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'config.yaml'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = 'setting'
    var_13 = 123
    var_14 = {var_12: var_13}
    var_15 = module_0.TrieNode(config_data=var_14)
    var_16 = module_0.TrieNode()
    var_17 = module_0.TrieNode()
    var_18 = {var_3: var_5}
    var_19 = module_0.TrieNode(var_1, var_18)



# Parsed testcases at query #20
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.config'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'path/to/config.json'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = None
    var_13 = module_0.TrieNode(var_1, var_12)
    var_14 = 'test'
    var_15 = var_0.nodes[var_14]
    var_16 = 'file1'
    var_17 = 'a'
    var_18 = 1
    var_19 = {var_17: var_18}
    var_20 = module_0.TrieNode(var_16, var_19)
    var_21 = 'file2'
    var_22 = 'b'
    var_23 = 2
    var_24 = {var_22: var_23}
    var_25 = module_0.TrieNode(var_21, var_24)



# Parsed testcases at query #21
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = 'config.yaml'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/home/user/config.yaml'
    var_11 = 'level'
    var_12 = 1
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/home/user/project/config.yaml'
    var_16 = 2
    var_17 = {var_11: var_16}
    var_18 = var_9.insert(var_15, var_17)
    var_19 = '/home/user/project/subdir/file.py'
    var_20 = var_9.search(var_19)
    var_21 = module_0.Trie()
    var_22 = {var_11: var_12}
    var_23 = var_21.insert(var_10, var_22)
    var_24 = {var_11: var_16}
    var_25 = var_21.insert(var_15, var_24)
    var_26 = '/home/user/other/file.py'
    var_27 = var_21.search(var_26)
    var_28 = module_0.Trie()
    var_29 = '/a/config.yaml'
    var_30 = 'depth'
    var_31 = {var_30: var_12}
    var_32 = var_28.insert(var_29, var_31)
    var_33 = '/a/b/config.yaml'
    var_34 = {var_30: var_16}
    var_35 = var_28.insert(var_33, var_34)
    var_36 = '/a/b/c/config.yaml'
    var_37 = 3
    var_38 = {var_30: var_37}
    var_39 = var_28.insert(var_36, var_38)
    var_40 = '/a/b/c/d/e/file.py'
    var_41 = var_28.search(var_40)
    var_42 = 'root_config.yaml'
    var_43 = 'root'
    var_44 = True
    var_45 = {var_43: var_44}
    var_46 = module_0.Trie(var_42, var_45)
    var_47 = {var_11: var_44}
    var_48 = var_46.insert(var_10, var_47)
    var_49 = '/other/path/file.py'
    var_50 = var_46.search(var_49)
    var_51 = module_0.Trie()
    var_52 = '/home/config.yaml'
    var_53 = {}
    var_54 = var_51.insert(var_52, var_53)
    var_55 = '/home/subdir/file.py'
    var_56 = var_51.search(var_55)
    var_57 = module_0.Trie()
    var_58 = 'settings'
    var_59 = 'paths'
    var_60 = 'debug'
    var_61 = 'version'
    var_62 = True
    var_63 = '1.0'
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = '/a'
    var_66 = '/b'
    var_67 = [var_65, var_66]
    var_68 = {var_58: var_64, var_59: var_67}
    var_69 = '/project/config.yaml'
    var_70 = var_57.insert(var_69, var_68)
    var_71 = '/project/src/module/file.py'
    var_72 = var_57.search(var_71)
    var_73 = module_0.Trie()
    var_74 = {var_11: var_62}
    var_75 = var_73.insert(var_10, var_74)
    var_76 = {var_11: var_16}
    var_77 = var_73.insert(var_15, var_76)
    var_78 = '/home/user/nonexistent/other/file.py'
    var_79 = var_73.search(var_78)
    var_80 = module_0.Trie()
    var_81 = 'type'
    var_82 = 'home'
    var_83 = {var_81: var_82}
    var_84 = var_80.insert(var_52, var_83)
    var_85 = '/home/project/config.yaml'
    var_86 = 'project'
    var_87 = {var_81: var_86}
    var_88 = var_80.insert(var_85, var_87)
    var_89 = '/home/project/file.py'
    var_90 = var_80.search(var_89)



# Parsed testcases at query #22
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.cfg'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'path/to/config.cfg'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = None
    var_13 = module_0.TrieNode(var_1, var_12)
    var_14 = {}
    var_15 = module_0.TrieNode(var_1, var_14)
    var_16 = module_0.TrieNode()
    var_17 = module_0.TrieNode()
    var_18 = 'string'
    var_19 = 'number'
    var_20 = 'list'
    var_21 = 'dict'
    var_22 = 42
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_23, var_28: var_24}
    var_30 = {var_18: var_5, var_19: var_22, var_20: var_26, var_21: var_29}
    var_31 = 'complex.cfg'
    var_32 = module_0.TrieNode(var_31, var_30)



# Parsed testcases at query #23
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/project/some_file.py'
    var_7 = var_0.search(var_6)
    var_8 = module_0.Trie()
    var_9 = '/var/configs/settings.yaml'
    var_10 = {}
    var_11 = var_8.insert(var_9, var_10)
    var_12 = '/var/configs/nested/file.py'
    var_13 = var_8.search(var_12)
    var_14 = module_0.Trie()
    var_15 = '/root/config.json'
    var_16 = 'level'
    var_17 = 1
    var_18 = {var_16: var_17}
    var_19 = (var_15, var_18)
    var_20 = '/root/sub/config.json'
    var_21 = 2
    var_22 = {var_16: var_21}
    var_23 = (var_20, var_22)
    var_24 = '/root/sub/deep/config.json'
    var_25 = 3
    var_26 = {var_16: var_25}
    var_27 = (var_24, var_26)
    var_28 = 0
    var_29 = var_19[var_28]
    var_30 = var_19[var_17]
    var_31 = var_14.insert(var_29, var_30)
    var_32 = var_23[var_28]
    var_33 = var_23[var_17]
    var_34 = var_14.insert(var_32, var_33)
    var_35 = var_27[var_28]
    var_36 = var_27[var_17]
    var_37 = var_14.insert(var_35, var_36)
    var_38 = module_0.Trie()
    var_39 = '/path/to/config.json'
    var_40 = 'version'
    var_41 = {var_40: var_17}
    var_42 = {var_40: var_21}
    var_43 = var_38.insert(var_39, var_41)
    var_44 = var_38.insert(var_39, var_42)
    var_45 = '/path/to/file.py'
    var_46 = var_38.search(var_45)
    var_47 = module_0.Trie()
    var_48 = '/a/b/c/d/e/f/config.json'
    var_49 = 'nested'
    var_50 = 'deep'
    var_51 = 'structure'
    var_52 = True
    var_53 = {var_51: var_52}
    var_54 = {var_50: var_53}
    var_55 = {var_49: var_54}
    var_56 = var_47.insert(var_48, var_55)
    var_57 = '/a/b/c/d/e/f/g/h/file.py'
    var_58 = var_47.search(var_57)



# Parsed testcases at query #24
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.conf'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = '/path/to/config.conf'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = 'config.conf'
    var_13 = {}
    var_14 = module_0.TrieNode(var_12, var_13)
    var_15 = None
    var_16 = module_0.TrieNode(var_12, var_15)
    var_17 = module_0.TrieNode()
    var_18 = module_0.TrieNode()
    var_19 = 'string'
    var_20 = 'number'
    var_21 = 'list'
    var_22 = 'dict'
    var_23 = 42
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = 'a'
    var_29 = 'b'
    var_30 = {var_28: var_24, var_29: var_25}
    var_31 = {var_19: var_5, var_20: var_23, var_21: var_27, var_22: var_30}
    var_32 = 'complex.conf'
    var_33 = module_0.TrieNode(var_32, var_31)



# Parsed testcases at query #25
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'path/to/config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 123
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'config.yaml'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = module_0.TrieNode(config_data=var_9)
    var_13 = 'test'
    var_14 = var_0.nodes[var_13]
    var_15 = 'test.json'
    var_16 = None
    var_17 = module_0.TrieNode(var_15, var_16)
    var_18 = 'a'
    var_19 = 1
    var_20 = {var_18: var_19}
    var_21 = 'b'
    var_22 = 2
    var_23 = {var_21: var_22}
    var_24 = 'file1'
    var_25 = module_0.TrieNode(var_24, var_20)
    var_26 = 'file2'
    var_27 = module_0.TrieNode(var_26, var_23)



# Parsed testcases at query #26
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.yaml'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = var_10.root
    var_12 = {}
    var_13 = module_0.Trie(var_2, var_12)
    var_14 = None
    var_15 = module_0.Trie(var_2, var_14)
    var_16 = module_0.Trie(var_2)



# Parsed testcases at query #27
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = var_10.root
    var_12 = module_0.Trie(var_2)
    var_13 = None
    var_14 = module_0.Trie(var_2, var_13)



# Parsed testcases at query #28
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = var_10.root
    var_12 = module_0.Trie(var_2)
    var_13 = None
    var_14 = module_0.Trie(var_2, var_13)
    var_15 = {}
    var_16 = module_0.Trie(var_2, var_15)
    var_17 = 'version'
    var_18 = 'settings'
    var_19 = 'paths'
    var_20 = '1.0'
    var_21 = 'debug'
    var_22 = 'timeout'
    var_23 = True
    var_24 = 30
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = '/path1'
    var_27 = '/path2'
    var_28 = [var_26, var_27]
    var_29 = {var_17: var_20, var_18: var_25, var_19: var_28}
    var_30 = module_0.Trie(var_2, var_29)



# Parsed testcases at query #29
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'config.yaml'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = None
    var_13 = module_0.TrieNode(var_1, var_12)
    var_14 = module_0.TrieNode()
    var_15 = module_0.TrieNode()
    var_16 = 'list'
    var_17 = 'dict'
    var_18 = 'string'
    var_19 = 'number'
    var_20 = 'boolean'
    var_21 = 'none'
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = 'a'
    var_27 = 'b'
    var_28 = {var_26: var_22, var_27: var_23}
    var_29 = 42
    var_30 = True
    var_31 = {var_16: var_25, var_17: var_28, var_18: var_5, var_19: var_29, var_20: var_30, var_21: var_12}
    var_32 = 'complex.yaml'
    var_33 = module_0.TrieNode(var_32, var_31)



# Parsed testcases at query #30
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'root_value'
    var_5 = {var_3: var_4}
    var_6 = 'config.json'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = 'level'
    var_11 = 1
    var_12 = {var_10: var_11}
    var_13 = 2
    var_14 = {var_10: var_13}
    var_15 = '/home/user/project/config.json'
    var_16 = var_9.insert(var_15, var_12)
    var_17 = '/home/user/project/src/config.json'
    var_18 = var_9.insert(var_17, var_14)
    var_19 = '/home/user/project/src/subdir/file.py'
    var_20 = var_9.search(var_19)
    var_21 = module_0.Trie()
    var_22 = 'parent'
    var_23 = True
    var_24 = {var_22: var_23}
    var_25 = '/home/user/config.json'
    var_26 = var_21.insert(var_25, var_24)
    var_27 = '/home/user/project/file.py'
    var_28 = var_21.search(var_27)
    var_29 = module_0.Trie()
    var_30 = 'root'
    var_31 = True
    var_32 = {var_30: var_31}
    var_33 = '/config.json'
    var_34 = var_29.insert(var_33, var_32)
    var_35 = '/file.py'
    var_36 = var_29.search(var_35)
    var_37 = module_0.Trie()
    var_38 = {var_10: var_31}
    var_39 = {var_10: var_13}
    var_40 = 3
    var_41 = {var_10: var_40}
    var_42 = '/a/config.json'
    var_43 = var_37.insert(var_42, var_38)
    var_44 = '/a/b/config.json'
    var_45 = var_37.insert(var_44, var_39)
    var_46 = '/a/b/c/config.json'
    var_47 = var_37.insert(var_46, var_41)
    var_48 = '/a/b/c/d/e/file.py'
    var_49 = var_37.search(var_48)
    var_50 = module_0.Trie()
    var_51 = 'test'
    var_52 = 'value'
    var_53 = {var_51: var_52}
    var_54 = var_50.insert(var_25, var_53)
    var_55 = '/home/user/different/path/file.py'
    var_56 = var_50.search(var_55)
    var_57 = module_0.Trie()
    var_58 = '/path/config.json'
    var_59 = {}
    var_60 = var_57.insert(var_58, var_59)
    var_61 = '/path/subdir/file.py'
    var_62 = var_57.search(var_61)



# Parsed testcases at query #31
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = module_0.Trie(var_2)
    var_12 = None
    var_13 = module_0.Trie(var_2, var_12)
    var_14 = module_0.Trie()
    var_15 = module_0.Trie()



# Parsed testcases at query #32
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = var_10.root
    var_12 = module_0.Trie(var_2)
    var_13 = {}
    var_14 = module_0.Trie(var_2, var_13)
    var_15 = module_0.Trie()
    var_16 = module_0.Trie()



# Parsed testcases at query #33
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = '/config.yaml'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_3)
    var_9 = module_0.Trie()
    var_10 = '/home/user/.config'
    var_11 = 'level'
    var_12 = 1
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/home/user/.config/file.py'
    var_16 = var_9.search(var_15)
    var_17 = module_0.Trie()
    var_18 = '/home/.config'
    var_19 = 'home'
    var_20 = {var_11: var_19}
    var_21 = var_17.insert(var_18, var_20)
    var_22 = 'user'
    var_23 = {var_11: var_22}
    var_24 = var_17.insert(var_10, var_23)
    var_25 = '/home/user/project/file.py'
    var_26 = var_17.search(var_25)
    var_27 = '/root.yaml'
    var_28 = 'root'
    var_29 = True
    var_30 = {var_28: var_29}
    var_31 = module_0.Trie(var_27, var_30)
    var_32 = True
    var_33 = {var_22: var_32}
    var_34 = var_31.insert(var_10, var_33)
    var_35 = '/other/path/file.py'
    var_36 = var_31.search(var_35)
    var_37 = module_0.Trie()
    var_38 = {var_11: var_32}
    var_39 = var_37.insert(var_18, var_38)
    var_40 = 2
    var_41 = {var_11: var_40}
    var_42 = var_37.insert(var_10, var_41)
    var_43 = '/home/user/project/.config'
    var_44 = 3
    var_45 = {var_11: var_44}
    var_46 = var_37.insert(var_43, var_45)
    var_47 = '/home/user/project/src/file.py'
    var_48 = var_37.search(var_47)
    var_49 = module_0.Trie()
    var_50 = 'data'
    var_51 = 'test'
    var_52 = {var_50: var_51}
    var_53 = var_49.insert(var_10, var_52)
    var_54 = '/home/other/file.py'
    var_55 = var_49.search(var_54)
    var_56 = module_0.Trie()
    var_57 = 'rules'
    var_58 = 'nested'
    var_59 = [var_32, var_40, var_44]
    var_60 = {var_4: var_5}
    var_61 = {var_57: var_59, var_58: var_60}
    var_62 = '/project/.config'
    var_63 = var_56.insert(var_62, var_61)
    var_64 = '/project/src/main.py'
    var_65 = var_56.search(var_64)
    var_66 = module_0.Trie()
    var_67 = {}
    var_68 = var_66.insert(var_18, var_67)
    var_69 = '/home/file.py'
    var_70 = var_66.search(var_69)
    var_71 = 'config'
    var_72 = {var_28: var_71}
    var_73 = module_0.Trie(var_27, var_72)
    var_74 = '/any/path/file.py'
    var_75 = var_73.search(var_74)



# Parsed testcases at query #34
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/project/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/home/user/project/some_file.py'
    var_7 = var_0.search(var_6)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'level'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = {var_1: var_4}
    var_6 = '/home/user/config.json'
    var_7 = '/home/user/project/config.json'
    var_8 = var_0.insert(var_6, var_3)
    var_9 = var_0.insert(var_7, var_5)
    var_10 = '/home/user/project/subdir/file.py'
    var_11 = var_0.search(var_10)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'version'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = {var_1: var_4}
    var_6 = '/home/user/project/config.json'
    var_7 = var_0.insert(var_6, var_3)
    var_8 = var_0.insert(var_6, var_5)
    var_9 = '/home/user/project/file.py'
    var_10 = var_0.search(var_9)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = '/home/user/file.py'
    var_5 = var_0.search(var_4)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'nested'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = '/home/user/a/b/c/d/e/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/home/user/a/b/c/d/e/f/g/file.py'
    var_7 = var_0.search(var_6)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/project/config.json'
    var_5 = var_0.insert(var_4, var_3)



# Parsed testcases at query #35
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/project/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = var_0.search(var_4)
    var_7 = module_0.Trie()
    var_8 = 'setting1'
    var_9 = 'value1'
    var_10 = {var_8: var_9}
    var_11 = '/home/user/project/config1.json'
    var_12 = 'setting2'
    var_13 = 'value2'
    var_14 = {var_12: var_13}
    var_15 = '/home/user/project/subfolder/config2.json'
    var_16 = var_7.insert(var_11, var_10)
    var_17 = var_7.insert(var_15, var_14)
    var_18 = var_7.search(var_11)
    var_19 = var_7.search(var_15)
    var_20 = module_0.Trie()
    var_21 = '/home/user/config.json'
    var_22 = 'old'
    var_23 = 'data'
    var_24 = {var_22: var_23}
    var_25 = 'new'
    var_26 = {var_25: var_23}
    var_27 = var_20.insert(var_21, var_24)
    var_28 = var_20.insert(var_21, var_26)
    var_29 = var_20.search(var_21)
    var_30 = module_0.Trie()
    var_31 = '/home/user/project/empty.json'
    var_32 = {}
    var_33 = var_30.insert(var_31, var_32)
    var_34 = var_30.search(var_31)
    var_35 = module_0.Trie()
    var_36 = '/a/b/c/d/config.json'
    var_37 = 'nested'
    var_38 = 'path'
    var_39 = {var_37: var_38}
    var_40 = var_35.insert(var_36, var_39)
    var_41 = var_35.root



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'config.yaml'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = None
    var_13 = module_0.TrieNode(var_1, var_12)
    var_14 = module_0.TrieNode()
    var_15 = module_0.TrieNode()
    var_16 = ''
    var_17 = 'test'
    var_18 = {var_7: var_17}
    var_19 = module_0.TrieNode(var_16, var_18)
    var_20 = module_0.TrieNode(var_1, var_12)
    var_21 = 'test2.yaml'
    var_22 = module_0.TrieNode(var_21, var_12)



# Parsed testcases at query #2
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = '/home/user/project/subdir/config.json'
    var_6 = 'key2'
    var_7 = 'value2'
    var_8 = {var_6: var_7}
    var_9 = var_0.insert(var_1, var_4)
    var_10 = var_0.insert(var_5, var_8)
    var_11 = var_0.root
    var_12 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = 'value2'
    var_6 = {var_2: var_5}
    var_7 = var_0.insert(var_1, var_4)
    var_8 = var_0.insert(var_1, var_6)
    var_9 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/a/b/c/d/e/config.json'
    var_2 = 'nested'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #3
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'config.yaml'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = None
    var_13 = module_0.TrieNode(var_1, var_12)
    var_14 = module_0.TrieNode()
    var_15 = module_0.TrieNode()
    var_16 = 'empty.yaml'
    var_17 = {}
    var_18 = module_0.TrieNode(var_16, var_17)
    var_19 = 'database'
    var_20 = 'logging'
    var_21 = 'list_data'
    var_22 = 'host'
    var_23 = 'port'
    var_24 = 'localhost'
    var_25 = 5432
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = 'level'
    var_28 = 'INFO'
    var_29 = {var_27: var_28}
    var_30 = 1
    var_31 = 2
    var_32 = 3
    var_33 = [var_30, var_31, var_32]
    var_34 = {var_19: var_26, var_20: var_29, var_21: var_33}
    var_35 = 'complex.yaml'
    var_36 = module_0.TrieNode(var_35, var_34)



# Parsed testcases at query #4
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/file.py'
    var_2 = var_0.search(var_1)
    var_3 = '/config.yaml'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/home/user/.config'
    var_11 = 'root'
    var_12 = 'config'
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = var_9.search(var_1)
    var_16 = module_0.Trie()
    var_17 = '/home/.config'
    var_18 = 'level'
    var_19 = 1
    var_20 = {var_18: var_19}
    var_21 = var_16.insert(var_17, var_20)
    var_22 = 2
    var_23 = {var_18: var_22}
    var_24 = var_16.insert(var_10, var_23)
    var_25 = var_16.search(var_1)
    var_26 = module_0.Trie()
    var_27 = 'home'
    var_28 = True
    var_29 = {var_27: var_28}
    var_30 = var_26.insert(var_17, var_29)
    var_31 = 'user'
    var_32 = True
    var_33 = {var_31: var_32}
    var_34 = var_26.insert(var_10, var_33)
    var_35 = '/home/user/project/.config'
    var_36 = 'project'
    var_37 = True
    var_38 = {var_36: var_37}
    var_39 = var_26.insert(var_35, var_38)
    var_40 = '/home/user/project/src/file.py'
    var_41 = var_26.search(var_40)
    var_42 = module_0.Trie()
    var_43 = True
    var_44 = {var_31: var_43}
    var_45 = var_42.insert(var_10, var_44)
    var_46 = '/home/other/file.py'
    var_47 = var_42.search(var_46)
    var_48 = module_0.Trie()
    var_49 = 'exact'
    var_50 = True
    var_51 = {var_49: var_50}
    var_52 = var_48.insert(var_35, var_51)
    var_53 = var_48.search(var_1)
    var_54 = module_0.Trie()
    var_55 = 'nested'
    var_56 = 'list'
    var_57 = {var_4: var_5}
    var_58 = 3
    var_59 = [var_50, var_22, var_58]
    var_60 = {var_55: var_57, var_56: var_59}
    var_61 = var_54.insert(var_10, var_60)
    var_62 = var_54.search(var_1)
    var_63 = module_0.Trie()
    var_64 = 'version'
    var_65 = {var_64: var_50}
    var_66 = var_63.insert(var_10, var_65)
    var_67 = {var_64: var_22}
    var_68 = var_63.insert(var_10, var_67)
    var_69 = '/home/user/file.py'
    var_70 = var_63.search(var_69)
    var_71 = module_0.Trie()
    var_72 = '/config'
    var_73 = 'single'
    var_74 = True
    var_75 = {var_73: var_74}
    var_76 = var_71.insert(var_72, var_75)
    var_77 = '/file.py'
    var_78 = var_71.search(var_77)



# Parsed testcases at query #5
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = var_10.root
    var_12 = module_0.Trie(var_2)
    var_13 = None
    var_14 = module_0.Trie(var_2, var_13)



# Parsed testcases at query #6
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = '/config.yaml'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = '/file.py'
    var_9 = var_7.search(var_8)
    var_10 = module_0.Trie()
    var_11 = '/home/user/config.yaml'
    var_12 = 'setting'
    var_13 = 'test'
    var_14 = {var_12: var_13}
    var_15 = var_10.insert(var_11, var_14)
    var_16 = '/home/user/script.py'
    var_17 = var_10.search(var_16)
    var_18 = module_0.Trie()
    var_19 = '/home/config.yaml'
    var_20 = 'level'
    var_21 = 'parent'
    var_22 = {var_20: var_21}
    var_23 = var_18.insert(var_19, var_22)
    var_24 = 'child'
    var_25 = {var_20: var_24}
    var_26 = var_18.insert(var_11, var_25)
    var_27 = '/home/user/subdir/script.py'
    var_28 = var_18.search(var_27)
    var_29 = module_0.Trie()
    var_30 = 'root'
    var_31 = True
    var_32 = {var_30: var_31}
    var_33 = var_29.insert(var_3, var_32)
    var_34 = 'home'
    var_35 = {var_34: var_31}
    var_36 = var_29.insert(var_19, var_35)
    var_37 = '/home/user/project/file.py'
    var_38 = var_29.search(var_37)
    var_39 = '/root_config.yaml'
    var_40 = 'root_level'
    var_41 = 'config'
    var_42 = {var_40: var_41}
    var_43 = module_0.Trie(var_39, var_42)
    var_44 = 'user_level'
    var_45 = {var_44: var_41}
    var_46 = var_43.insert(var_11, var_45)
    var_47 = '/other/path/file.py'
    var_48 = var_43.search(var_47)
    var_49 = module_0.Trie()
    var_50 = '/a/b/c/config.yaml'
    var_51 = 'depth'
    var_52 = 3
    var_53 = {var_51: var_52}
    var_54 = var_49.insert(var_50, var_53)
    var_55 = '/a/b/config.yaml'
    var_56 = 2
    var_57 = {var_51: var_56}
    var_58 = var_49.insert(var_55, var_57)
    var_59 = '/a/b/c/d/e/file.py'
    var_60 = var_49.search(var_59)
    var_61 = module_0.Trie()
    var_62 = '/existing/config.yaml'
    var_63 = 'exists'
    var_64 = {var_63: var_31}
    var_65 = var_61.insert(var_62, var_64)
    var_66 = '/existing/nonexistent/file.py'
    var_67 = var_61.search(var_66)
    var_68 = module_0.Trie()
    var_69 = '/path/config.yaml'
    var_70 = {}
    var_71 = var_68.insert(var_69, var_70)
    var_72 = '/path/file.py'
    var_73 = var_68.search(var_72)
    var_74 = module_0.Trie()
    var_75 = 'nested'
    var_76 = 'list'
    var_77 = 'bool'
    var_78 = {var_4: var_5}
    var_79 = [var_31, var_56, var_52]
    var_80 = {var_75: var_78, var_76: var_79, var_77: var_31}
    var_81 = var_74.insert(var_19, var_80)
    var_82 = '/home/subdir/file.py'
    var_83 = var_74.search(var_82)



# Parsed testcases at query #7
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/project/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/home/user/project/file.py'
    var_7 = var_0.search(var_6)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'level'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = '/home/user/config.json'
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = '/home/user/project/config.json'
    var_8 = var_0.insert(var_4, var_3)
    var_9 = var_0.insert(var_7, var_6)
    var_10 = '/home/user/project/file.py'
    var_11 = var_0.search(var_10)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'version'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_5}
    var_7 = var_0.insert(var_1, var_4)
    var_8 = var_0.insert(var_1, var_6)
    var_9 = '/home/user/project/file.py'
    var_10 = var_0.search(var_9)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = '/home/user/file.py'
    var_5 = var_0.search(var_4)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/src/module/config.json'
    var_2 = 'nested'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/project/src/module/submodule/file.py'
    var_7 = var_0.search(var_6)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/a/b/c/config.json'
    var_2 = 'test'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = 'c'
    var_7 = 'b'
    var_8 = 'a'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes[var_7]
    var_11 = var_10.nodes[var_6]



# Parsed testcases at query #8
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.cfg'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'app.cfg'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = None
    var_13 = module_0.TrieNode(var_1, var_12)
    var_14 = module_0.TrieNode()
    var_15 = module_0.TrieNode()
    var_16 = 'string'
    var_17 = 'number'
    var_18 = 'list'
    var_19 = 42
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = 'val'
    var_25 = {var_3: var_24}
    var_26 = {var_16: var_5, var_17: var_19, var_18: var_23, var_4: var_25}
    var_27 = 'complex.cfg'
    var_28 = module_0.TrieNode(var_27, var_26)



# Parsed testcases at query #9
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = '/root/config.json'
    var_4 = 'root'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/home/user/config.json'
    var_11 = 'level'
    var_12 = {var_11: var_5}
    var_13 = var_9.insert(var_10, var_12)
    var_14 = '/home/user/file.py'
    var_15 = var_9.search(var_14)
    var_16 = module_0.Trie()
    var_17 = '/home/config.json'
    var_18 = {var_11: var_5}
    var_19 = var_16.insert(var_17, var_18)
    var_20 = 2
    var_21 = {var_11: var_20}
    var_22 = var_16.insert(var_10, var_21)
    var_23 = '/home/user/project/file.py'
    var_24 = var_16.search(var_23)
    var_25 = module_0.Trie()
    var_26 = {var_11: var_5}
    var_27 = var_25.insert(var_17, var_26)
    var_28 = var_25.search(var_23)
    var_29 = module_0.Trie()
    var_30 = '/config.json'
    var_31 = {var_4: var_5}
    var_32 = var_29.insert(var_30, var_31)
    var_33 = 'home'
    var_34 = {var_33: var_5}
    var_35 = var_29.insert(var_17, var_34)
    var_36 = 'user'
    var_37 = {var_36: var_5}
    var_38 = var_29.insert(var_10, var_37)
    var_39 = '/home/user/project/config.json'
    var_40 = 'project'
    var_41 = {var_40: var_5}
    var_42 = var_29.insert(var_39, var_41)
    var_43 = '/home/user/project/src/file.py'
    var_44 = var_29.search(var_43)
    var_45 = module_0.Trie()
    var_46 = 'data'
    var_47 = 'test'
    var_48 = {var_46: var_47}
    var_49 = var_45.insert(var_10, var_48)
    var_50 = '/other/path/file.py'
    var_51 = var_45.search(var_50)
    var_52 = module_0.Trie()
    var_53 = 'rules'
    var_54 = 'options'
    var_55 = 3
    var_56 = [var_5, var_20, var_55]
    var_57 = 'strict'
    var_58 = {var_57: var_5}
    var_59 = {var_53: var_56, var_54: var_58}
    var_60 = '/project/config.json'
    var_61 = var_52.insert(var_60, var_59)
    var_62 = '/project/src/module/file.py'
    var_63 = var_52.search(var_62)
    var_64 = module_0.Trie()
    var_65 = 'exact'
    var_66 = {var_65: var_5}
    var_67 = var_64.insert(var_10, var_66)
    var_68 = var_64.search(var_10)
    var_69 = module_0.Trie()
    var_70 = {var_36: var_5}
    var_71 = var_69.insert(var_10, var_70)
    var_72 = '/home/admin/config.json'
    var_73 = 'admin'
    var_74 = {var_73: var_5}
    var_75 = var_69.insert(var_72, var_74)
    var_76 = var_69.search(var_14)



# Parsed testcases at query #10
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = var_10.root
    var_12 = module_0.Trie(var_2)
    var_13 = None
    var_14 = module_0.Trie(var_2, var_13)
    var_15 = 'config1.json'
    var_16 = 1
    var_17 = {var_7: var_16}
    var_18 = module_0.Trie(var_15, var_17)
    var_19 = 'config2.json'
    var_20 = 2
    var_21 = {var_7: var_20}
    var_22 = module_0.Trie(var_19, var_21)



# Parsed testcases at query #11
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = '/config.yaml'
    var_8 = var_3.insert(var_7, var_6)
    var_9 = '/file.py'
    var_10 = var_3.search(var_9)
    var_11 = module_0.Trie()
    var_12 = 'level'
    var_13 = 1
    var_14 = {var_12: var_13}
    var_15 = '/home/user/project/config.yaml'
    var_16 = var_11.insert(var_15, var_14)
    var_17 = '/home/user/project/file.py'
    var_18 = var_11.search(var_17)
    var_19 = module_0.Trie()
    var_20 = 'parent'
    var_21 = {var_12: var_20}
    var_22 = 'child'
    var_23 = {var_12: var_22}
    var_24 = '/home/user/config.yaml'
    var_25 = var_19.insert(var_24, var_21)
    var_26 = var_19.insert(var_15, var_23)
    var_27 = '/home/user/project/subdir/file.py'
    var_28 = var_19.search(var_27)
    var_29 = module_0.Trie()
    var_30 = {var_12: var_20}
    var_31 = var_29.insert(var_24, var_30)
    var_32 = '/home/user/other/file.py'
    var_33 = var_29.search(var_32)
    var_34 = module_0.Trie()
    var_35 = 'root'
    var_36 = True
    var_37 = {var_35: var_36}
    var_38 = {var_12: var_36}
    var_39 = 2
    var_40 = {var_12: var_39}
    var_41 = var_34.insert(var_7, var_37)
    var_42 = '/home/config.yaml'
    var_43 = var_34.insert(var_42, var_38)
    var_44 = var_34.insert(var_24, var_40)
    var_45 = var_34.search(var_17)
    var_46 = module_0.Trie()
    var_47 = True
    var_48 = {var_35: var_47}
    var_49 = var_46.insert(var_7, var_48)
    var_50 = var_46.search(var_17)
    var_51 = module_0.Trie()
    var_52 = 'deep'
    var_53 = True
    var_54 = {var_52: var_53}
    var_55 = '/a/b/c/d/config.yaml'
    var_56 = var_51.insert(var_55, var_54)
    var_57 = '/a/b/c/d/e/f/file.py'
    var_58 = var_51.search(var_57)
    var_59 = module_0.Trie()
    var_60 = {}
    var_61 = var_59.insert(var_7, var_60)
    var_62 = var_59.search(var_9)
    var_63 = module_0.Trie()
    var_64 = 'version'
    var_65 = {var_64: var_53}
    var_66 = {var_64: var_39}
    var_67 = var_63.insert(var_42, var_65)
    var_68 = var_63.insert(var_15, var_66)
    var_69 = '/home/user/other_dir/file.py'
    var_70 = var_63.search(var_69)



# Parsed testcases at query #12
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.cfg'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'path/to/config.cfg'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = None
    var_13 = module_0.TrieNode(var_1, var_12)
    var_14 = module_0.TrieNode()
    var_15 = module_0.TrieNode()
    var_16 = 'config.cfg'
    var_17 = {}
    var_18 = module_0.TrieNode(var_16, var_17)
    var_19 = 'mutable'
    var_20 = {var_19: var_7}
    var_21 = module_0.TrieNode(var_1, var_20)



# Parsed testcases at query #13
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'path/to/config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = 'config.json'
    var_12 = None
    var_13 = module_0.TrieNode(var_11, var_12)
    var_14 = module_0.TrieNode()
    var_15 = module_0.TrieNode()
    var_16 = ''
    var_17 = module_0.TrieNode(var_16)
    var_18 = 'list'
    var_19 = 'dict'
    var_20 = 'string'
    var_21 = 'number'
    var_22 = 'boolean'
    var_23 = 'null'
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = 'a'
    var_29 = 'b'
    var_30 = {var_28: var_24, var_29: var_25}
    var_31 = 'test'
    var_32 = 42
    var_33 = True
    var_34 = {var_18: var_27, var_19: var_30, var_20: var_31, var_21: var_32, var_22: var_33, var_23: var_12}
    var_35 = module_0.TrieNode(var_11, var_34)



# Parsed testcases at query #14
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/project/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = var_0.search(var_4)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'level'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = '/home/user/config.json'
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = '/home/user/project/config.json'
    var_8 = var_0.insert(var_4, var_3)
    var_9 = var_0.insert(var_7, var_6)
    var_10 = var_0.search(var_4)
    var_11 = var_0.search(var_7)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'version'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_5}
    var_7 = var_0.insert(var_1, var_4)
    var_8 = var_0.insert(var_1, var_6)
    var_9 = var_0.search(var_1)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'depth'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = '/a/b/c/config.json'
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = '/a/b/c/d/e/f/config.json'
    var_8 = var_0.insert(var_4, var_3)
    var_9 = var_0.insert(var_7, var_6)
    var_10 = var_0.search(var_4)
    var_11 = var_0.search(var_7)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = var_0.search(var_1)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = 'nested'
    var_3 = 'list'
    var_4 = 'string'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = 'test'
    var_13 = {var_2: var_7, var_3: var_11, var_4: var_12}
    var_14 = var_0.insert(var_1, var_13)
    var_15 = var_0.search(var_1)



# Parsed testcases at query #15
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/project/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = 'key2'
    var_7 = 'value2'
    var_8 = {var_6: var_7}
    var_9 = '/home/user/config.json'
    var_10 = var_0.insert(var_9, var_8)
    var_11 = 'key3'
    var_12 = 'value3'
    var_13 = {var_11: var_12}
    var_14 = '/home/user/project/src/config.json'
    var_15 = var_0.insert(var_14, var_13)
    var_16 = '/var/log/config.json'
    var_17 = {}
    var_18 = var_0.insert(var_16, var_17)
    var_19 = 'updated'
    var_20 = True
    var_21 = {var_19: var_20}
    var_22 = var_0.insert(var_16, var_21)



# Parsed testcases at query #16
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = '/config.json'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_3.insert(var_4, var_7)
    var_9 = '/file.py'
    var_10 = var_3.search(var_9)
    var_11 = module_0.Trie()
    var_12 = '/home/config.json'
    var_13 = 'level'
    var_14 = 1
    var_15 = {var_13: var_14}
    var_16 = var_11.insert(var_12, var_15)
    var_17 = '/home/project/config.json'
    var_18 = 2
    var_19 = {var_13: var_18}
    var_20 = var_11.insert(var_17, var_19)
    var_21 = '/home/project/src/file.py'
    var_22 = var_11.search(var_21)
    var_23 = module_0.Trie()
    var_24 = {var_13: var_14}
    var_25 = var_23.insert(var_12, var_24)
    var_26 = var_23.search(var_21)
    var_27 = module_0.Trie()
    var_28 = '/a/config.json'
    var_29 = {var_13: var_14}
    var_30 = var_27.insert(var_28, var_29)
    var_31 = '/a/b/config.json'
    var_32 = {var_13: var_18}
    var_33 = var_27.insert(var_31, var_32)
    var_34 = '/a/b/c/config.json'
    var_35 = 3
    var_36 = {var_13: var_35}
    var_37 = var_27.insert(var_34, var_36)
    var_38 = '/a/b/c/d/e/file.py'
    var_39 = var_27.search(var_38)
    var_40 = module_0.Trie()
    var_41 = 'home'
    var_42 = True
    var_43 = {var_41: var_42}
    var_44 = var_40.insert(var_12, var_43)
    var_45 = '/other/path/file.py'
    var_46 = var_40.search(var_45)
    var_47 = 'root.json'
    var_48 = 'root'
    var_49 = True
    var_50 = {var_48: var_49}
    var_51 = module_0.TrieNode(var_47, var_50)
    var_52 = True
    var_53 = {var_48: var_52}
    var_54 = module_0.Trie(var_47, var_53)
    var_55 = '/any/path/file.py'
    var_56 = var_54.search(var_55)
    var_57 = module_0.Trie()
    var_58 = {}
    var_59 = var_57.insert(var_4, var_58)
    var_60 = var_57.search(var_9)
    var_61 = module_0.Trie()
    var_62 = 'nested'
    var_63 = 'list'
    var_64 = {var_5: var_6}
    var_65 = [var_52, var_18, var_35]
    var_66 = {var_62: var_64, var_63: var_65}
    var_67 = '/path/config.json'
    var_68 = var_61.insert(var_67, var_66)
    var_69 = '/path/subdir/file.py'
    var_70 = var_61.search(var_69)
    var_71 = module_0.Trie()
    var_72 = {var_13: var_18}
    var_73 = var_71.insert(var_31, var_72)
    var_74 = '/a/x/y/z/file.py'
    var_75 = var_71.search(var_74)



# Parsed testcases at query #17
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'path/to/config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = 'setting1'
    var_12 = 'setting2'
    var_13 = True
    var_14 = 42
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = module_0.TrieNode(config_data=var_15)
    var_17 = module_0.TrieNode()
    var_18 = 'child.json'
    var_19 = module_0.TrieNode(var_18)
    var_20 = 'config.json'
    var_21 = {}
    var_22 = module_0.TrieNode(var_20, var_21)
    var_23 = 'test.json'
    var_24 = 'a'
    var_25 = {var_24: var_13}
    var_26 = module_0.TrieNode(var_23, var_25)
    var_27 = var_26.config_info
    var_28 = var_26.config_info
    var_29 = len(var_28)
    assert var_29 == 2



# Parsed testcases at query #18
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.config'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'app.config'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = 'settings.config'
    var_13 = None
    var_14 = module_0.TrieNode(var_12, var_13)
    var_15 = module_0.TrieNode()
    var_16 = module_0.TrieNode()
    var_17 = 'version'
    var_18 = 'settings'
    var_19 = 'items'
    var_20 = '1.0'
    var_21 = 'debug'
    var_22 = 'timeout'
    var_23 = True
    var_24 = 30
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 2
    var_27 = 3
    var_28 = [var_23, var_26, var_27]
    var_29 = {var_17: var_20, var_18: var_25, var_19: var_28}
    var_30 = '/path/to/config.yml'
    var_31 = module_0.TrieNode(var_30, var_29)



# Parsed testcases at query #19
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file/path.py'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = '/home/user/config.json'
    var_8 = var_3.insert(var_7, var_6)
    var_9 = '/home/user/project/file.py'
    var_10 = var_3.search(var_9)
    var_11 = module_0.Trie()
    var_12 = 'level'
    var_13 = 1
    var_14 = {var_12: var_13}
    var_15 = 2
    var_16 = {var_12: var_15}
    var_17 = '/home/config.json'
    var_18 = var_11.insert(var_17, var_14)
    var_19 = var_11.insert(var_7, var_16)
    var_20 = var_11.search(var_9)
    var_21 = module_0.Trie()
    var_22 = 'root'
    var_23 = True
    var_24 = {var_22: var_23}
    var_25 = var_21.insert(var_17, var_24)
    var_26 = var_21.search(var_9)
    var_27 = module_0.Trie()
    var_28 = {var_12: var_23}
    var_29 = {var_12: var_15}
    var_30 = 3
    var_31 = {var_12: var_30}
    var_32 = var_27.insert(var_17, var_28)
    var_33 = var_27.insert(var_7, var_29)
    var_34 = '/home/user/project/config.json'
    var_35 = var_27.insert(var_34, var_31)
    var_36 = '/home/user/project/src/file.py'
    var_37 = var_27.search(var_36)
    var_38 = module_0.Trie()
    var_39 = {var_4: var_5}
    var_40 = var_38.insert(var_7, var_39)
    var_41 = '/other/path/file.py'
    var_42 = var_38.search(var_41)
    var_43 = module_0.Trie()
    var_44 = 'exact'
    var_45 = True
    var_46 = {var_44: var_45}
    var_47 = var_43.insert(var_34, var_46)
    var_48 = var_43.search(var_9)
    var_49 = module_0.Trie()
    var_50 = {}
    var_51 = var_49.insert(var_17, var_50)
    var_52 = '/home/user/file.py'
    var_53 = var_49.search(var_52)
    var_54 = module_0.Trie()
    var_55 = 'data'
    var_56 = 'test'
    var_57 = {var_55: var_56}
    var_58 = var_54.insert(var_7, var_57)
    var_59 = '/home/file.py'
    var_60 = var_54.search(var_59)
    var_61 = 'root_config.json'
    var_62 = 'config'
    var_63 = {var_22: var_62}
    var_64 = module_0.Trie(var_61, var_63)
    var_65 = '/any/path/file.py'
    var_66 = var_64.search(var_65)



# Parsed testcases at query #20
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = '/config.yaml'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_3.insert(var_4, var_7)
    var_9 = '/file.py'
    var_10 = var_3.search(var_9)
    var_11 = module_0.Trie()
    var_12 = '/home/user/project/config.yaml'
    var_13 = 'level'
    var_14 = 1
    var_15 = {var_13: var_14}
    var_16 = var_11.insert(var_12, var_15)
    var_17 = '/home/user/project/src/module.py'
    var_18 = var_11.search(var_17)
    var_19 = module_0.Trie()
    var_20 = '/home/config.yaml'
    var_21 = {var_13: var_14}
    var_22 = var_19.insert(var_20, var_21)
    var_23 = '/home/user/config.yaml'
    var_24 = 2
    var_25 = {var_13: var_24}
    var_26 = var_19.insert(var_23, var_25)
    var_27 = 3
    var_28 = {var_13: var_27}
    var_29 = var_19.insert(var_12, var_28)
    var_30 = var_19.search(var_17)
    var_31 = module_0.Trie()
    var_32 = 'data'
    var_33 = 'test'
    var_34 = {var_32: var_33}
    var_35 = var_31.insert(var_23, var_34)
    var_36 = '/other/path/file.py'
    var_37 = var_31.search(var_36)
    var_38 = module_0.Trie()
    var_39 = 'project'
    var_40 = True
    var_41 = {var_39: var_40}
    var_42 = var_38.insert(var_12, var_41)
    var_43 = var_38.search(var_12)
    var_44 = module_0.Trie()
    var_45 = '/a/b/c/config.yaml'
    var_46 = 'nested'
    var_47 = True
    var_48 = {var_46: var_47}
    var_49 = var_44.insert(var_45, var_48)
    var_50 = '/a/b/c/d/e/f/file.py'
    var_51 = var_44.search(var_50)
    var_52 = module_0.Trie()
    var_53 = 'found'
    var_54 = True
    var_55 = {var_53: var_54}
    var_56 = var_52.insert(var_23, var_55)
    var_57 = '/home/other/path/file.py'
    var_58 = var_52.search(var_57)
    var_59 = module_0.Trie()
    var_60 = 'rules'
    var_61 = 'settings'
    var_62 = [var_54, var_24, var_27]
    var_63 = 'debug'
    var_64 = 'timeout'
    var_65 = True
    var_66 = 30
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = {var_60: var_62, var_61: var_67}
    var_69 = '/project/config.yaml'
    var_70 = var_59.insert(var_69, var_68)
    var_71 = '/project/src/utils/helper.py'
    var_72 = var_59.search(var_71)
    var_73 = module_0.Trie()
    var_74 = '/root/config.yaml'
    var_75 = 'root'
    var_76 = {var_13: var_75}
    var_77 = var_73.insert(var_74, var_76)
    var_78 = '/root/sub/config.yaml'
    var_79 = 'sub'
    var_80 = {var_13: var_79}
    var_81 = var_73.insert(var_78, var_80)
    var_82 = '/root/sub/file.py'
    var_83 = var_73.search(var_82)



# Parsed testcases at query #21
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = '/home/user/config.yaml'
    var_6 = 'key2'
    var_7 = 'value2'
    var_8 = {var_6: var_7}
    var_9 = var_0.insert(var_1, var_4)
    var_10 = var_0.insert(var_5, var_8)
    var_11 = var_0.root
    var_12 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = 'key'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = 'value2'
    var_6 = {var_2: var_5}
    var_7 = var_0.insert(var_1, var_4)
    var_8 = var_0.insert(var_1, var_6)
    var_9 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = 'nested'
    var_3 = 'list'
    var_4 = 'string'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = 'test'
    var_13 = {var_2: var_7, var_3: var_11, var_4: var_12}
    var_14 = var_0.insert(var_1, var_13)
    var_15 = var_0.root



# Parsed testcases at query #22
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = var_10.root
    var_12 = '/path/to/config.json'
    var_13 = module_0.Trie(var_12)
    var_14 = var_13.root
    var_15 = '/path/to/config.json'
    var_16 = {}
    var_17 = module_0.Trie(var_15, var_16)
    var_18 = var_17.root
    var_19 = '/path/to/config.json'
    var_20 = 'string'
    var_21 = 'number'
    var_22 = 'list'
    var_23 = 'dict'
    var_24 = 42
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_25, var_30: var_26}
    var_32 = 'deep'
    var_33 = 'structure'
    var_34 = 'here'
    var_35 = {var_33: var_34}
    var_36 = {var_32: var_35}
    var_37 = {var_20: var_5, var_21: var_24, var_22: var_28, var_23: var_31, var_4: var_36}
    var_38 = module_0.Trie(var_19, var_37)
    var_39 = var_38.root



# Parsed testcases at query #23
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test.config'
    var_3 = module_0.Trie(var_2)
    var_4 = 'key'
    var_5 = 'nested'
    var_6 = 'value'
    var_7 = 'inner'
    var_8 = 'data'
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = 'path/to/config.json'
    var_12 = module_0.Trie(var_11, var_10)
    var_13 = 'option'
    var_14 = 'setting'
    var_15 = {var_13: var_14}
    var_16 = module_0.Trie(config_data=var_15)
    var_17 = 'config.yml'
    var_18 = None
    var_19 = module_0.Trie(var_17, var_18)
    var_20 = 'config1.json'
    var_21 = 'id'
    var_22 = 1
    var_23 = {var_21: var_22}
    var_24 = module_0.Trie(var_20, var_23)
    var_25 = 'config2.json'
    var_26 = 2
    var_27 = {var_21: var_26}
    var_28 = module_0.Trie(var_25, var_27)



# Parsed testcases at query #24
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/project/file.py'
    var_7 = var_0.search(var_6)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = 'level'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = '/home/user/project/config.json'
    var_6 = 2
    var_7 = {var_2: var_6}
    var_8 = var_0.insert(var_1, var_4)
    var_9 = var_0.insert(var_5, var_7)
    var_10 = '/home/user/project/subdir/file.py'
    var_11 = var_0.search(var_10)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'version'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_5}
    var_7 = var_0.insert(var_1, var_4)
    var_8 = var_0.insert(var_1, var_6)
    var_9 = '/home/user/project/file.py'
    var_10 = var_0.search(var_9)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = '/home/user/file.py'
    var_5 = var_0.search(var_4)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/a/b/c/d/e/config.json'
    var_2 = 'nested'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/a/b/c/d/e/f/g/file.py'
    var_7 = var_0.search(var_6)



# Parsed testcases at query #25
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = var_10.root
    var_12 = module_0.Trie(var_2)
    var_13 = None
    var_14 = module_0.Trie(var_2, var_13)



# Parsed testcases at query #26
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file/path.py'
    var_2 = var_0.search(var_1)
    var_3 = '/config.yaml'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/home/user/project/config.yaml'
    var_11 = 'project'
    var_12 = 'settings'
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/home/user/project/file.py'
    var_16 = var_9.search(var_15)
    var_17 = module_0.Trie()
    var_18 = '/home/config.yaml'
    var_19 = 'level'
    var_20 = 'home'
    var_21 = {var_19: var_20}
    var_22 = var_17.insert(var_18, var_21)
    var_23 = '/home/user/config.yaml'
    var_24 = 'user'
    var_25 = {var_19: var_24}
    var_26 = var_17.insert(var_23, var_25)
    var_27 = {var_19: var_11}
    var_28 = var_17.insert(var_10, var_27)
    var_29 = '/home/user/project/src/file.py'
    var_30 = var_17.search(var_29)
    var_31 = module_0.Trie()
    var_32 = {var_19: var_24}
    var_33 = var_31.insert(var_23, var_32)
    var_34 = var_31.search(var_29)
    var_35 = module_0.Trie()
    var_36 = {var_19: var_20}
    var_37 = var_35.insert(var_18, var_36)
    var_38 = {var_19: var_24}
    var_39 = var_35.insert(var_23, var_38)
    var_40 = '/home/file.py'
    var_41 = var_35.search(var_40)
    var_42 = module_0.Trie()
    var_43 = 'paths'
    var_44 = 'debug'
    var_45 = 'timeout'
    var_46 = True
    var_47 = 30
    var_48 = {var_44: var_46, var_45: var_47}
    var_49 = '/a'
    var_50 = '/b'
    var_51 = [var_49, var_50]
    var_52 = {var_12: var_48, var_43: var_51}
    var_53 = '/app/config.yaml'
    var_54 = var_42.insert(var_53, var_52)
    var_55 = '/app/src/main.py'
    var_56 = var_42.search(var_55)
    var_57 = module_0.Trie()
    var_58 = 'root'
    var_59 = {var_58: var_46}
    var_60 = var_57.insert(var_3, var_59)
    var_61 = '/file.py'
    var_62 = var_57.search(var_61)
    var_63 = module_0.Trie()
    var_64 = '/completely/different/path/file.py'
    var_65 = var_63.search(var_64)
    var_66 = module_0.Trie()
    var_67 = '/a/b/config.yaml'
    var_68 = 'ab'
    var_69 = {var_19: var_68}
    var_70 = var_66.insert(var_67, var_69)
    var_71 = '/a/b/c/d/config.yaml'
    var_72 = 'abcd'
    var_73 = {var_19: var_72}
    var_74 = var_66.insert(var_71, var_73)
    var_75 = '/a/b/c/d/e/file.py'
    var_76 = var_66.search(var_75)



# Parsed testcases at query #27
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/file.py'
    var_2 = var_0.search(var_1)
    var_3 = '/root/config.json'
    var_4 = 'root'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/home/config.json'
    var_11 = 'level'
    var_12 = 'home'
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/home/user/config.json'
    var_16 = 'user'
    var_17 = {var_11: var_16}
    var_18 = var_9.insert(var_15, var_17)
    var_19 = '/home/user/project/config.json'
    var_20 = 'project'
    var_21 = {var_11: var_20}
    var_22 = var_9.insert(var_19, var_21)
    var_23 = var_9.search(var_1)
    var_24 = module_0.Trie()
    var_25 = {var_11: var_16}
    var_26 = var_24.insert(var_15, var_25)
    var_27 = '/home/user/project/subdir/file.py'
    var_28 = var_24.search(var_27)
    var_29 = module_0.Trie()
    var_30 = {var_11: var_12}
    var_31 = var_29.insert(var_10, var_30)
    var_32 = {var_11: var_16}
    var_33 = var_29.insert(var_15, var_32)
    var_34 = '/opt/other/file.py'
    var_35 = var_29.search(var_34)
    var_36 = module_0.Trie()
    var_37 = '/config.json'
    var_38 = {var_11: var_4}
    var_39 = var_36.insert(var_37, var_38)
    var_40 = {var_11: var_12}
    var_41 = var_36.insert(var_10, var_40)
    var_42 = {var_11: var_16}
    var_43 = var_36.insert(var_15, var_42)
    var_44 = var_36.search(var_1)
    var_45 = module_0.Trie()
    var_46 = {var_11: var_12}
    var_47 = var_45.insert(var_10, var_46)
    var_48 = '/home/file.py'
    var_49 = var_45.search(var_48)
    var_50 = module_0.Trie()
    var_51 = 'settings'
    var_52 = 'version'
    var_53 = 'debug'
    var_54 = 'timeout'
    var_55 = 30
    var_56 = {var_53: var_5, var_54: var_55}
    var_57 = '1.0'
    var_58 = {var_51: var_56, var_52: var_57}
    var_59 = var_50.insert(var_10, var_58)
    var_60 = '/home/user/file.py'
    var_61 = var_50.search(var_60)
    var_62 = module_0.Trie()
    var_63 = {var_11: var_16}
    var_64 = var_62.insert(var_15, var_63)
    var_65 = '/home/different/path/file.py'
    var_66 = var_62.search(var_65)
    var_67 = module_0.Trie()
    var_68 = {var_52: var_5}
    var_69 = var_67.insert(var_10, var_68)
    var_70 = 2
    var_71 = {var_52: var_70}
    var_72 = var_67.insert(var_10, var_71)
    var_73 = var_67.search(var_48)



# Parsed testcases at query #28
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'root_value'
    var_5 = {var_3: var_4}
    var_6 = 'config.yaml'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = var_7.search(var_1)
    var_9 = 'level'
    var_10 = 1
    var_11 = {var_9: var_10}
    var_12 = 2
    var_13 = {var_9: var_12}
    var_14 = module_0.Trie()
    var_15 = '/home/user/config.yaml'
    var_16 = var_14.insert(var_15, var_11)
    var_17 = '/home/user/project/config.yaml'
    var_18 = var_14.insert(var_17, var_13)
    var_19 = '/home/user/project/src/main.py'
    var_20 = var_14.search(var_19)
    var_21 = {var_9: var_10}
    var_22 = module_0.Trie()
    var_23 = var_22.insert(var_15, var_21)
    var_24 = var_22.search(var_19)
    var_25 = {var_9: var_10}
    var_26 = {var_9: var_12}
    var_27 = 3
    var_28 = {var_9: var_27}
    var_29 = module_0.Trie()
    var_30 = '/home/config.yaml'
    var_31 = var_29.insert(var_30, var_25)
    var_32 = var_29.insert(var_15, var_26)
    var_33 = var_29.insert(var_17, var_28)
    var_34 = var_29.search(var_19)
    var_35 = 'root'
    var_36 = True
    var_37 = {var_35: var_36}
    var_38 = module_0.Trie()
    var_39 = '/config.yaml'
    var_40 = var_38.insert(var_39, var_37)
    var_41 = '/main.py'
    var_42 = var_38.search(var_41)
    var_43 = {var_9: var_36}
    var_44 = module_0.Trie()
    var_45 = var_44.insert(var_15, var_43)
    var_46 = '/home/other/project/main.py'
    var_47 = var_44.search(var_46)
    var_48 = 'name'
    var_49 = 'config1'
    var_50 = {var_48: var_49}
    var_51 = 'config2'
    var_52 = {var_48: var_51}
    var_53 = module_0.Trie()
    var_54 = var_53.insert(var_15, var_50)
    var_55 = var_53.insert(var_15, var_52)
    var_56 = '/home/user/main.py'
    var_57 = var_53.search(var_56)
    var_58 = module_0.Trie()
    var_59 = {}
    var_60 = var_58.insert(var_30, var_59)
    var_61 = '/home/main.py'
    var_62 = var_58.search(var_61)
    var_63 = 'nested'
    var_64 = 'list'
    var_65 = 'value'
    var_66 = {var_3: var_65}
    var_67 = [var_36, var_12, var_27]
    var_68 = {var_63: var_66, var_64: var_67}
    var_69 = module_0.Trie()
    var_70 = var_69.insert(var_30, var_68)
    var_71 = '/home/project/main.py'
    var_72 = var_69.search(var_71)



# Parsed testcases at query #29
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/project/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'setting1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = 'setting2'
    var_11 = 'value2'
    var_12 = {var_10: var_11}
    var_13 = '/home/user/project/config.json'
    var_14 = '/home/user/project/subdir/config.json'
    var_15 = var_6.insert(var_13, var_9)
    var_16 = var_6.insert(var_14, var_12)
    var_17 = '/home/user/project/file.py'
    var_18 = var_6.search(var_17)
    var_19 = '/home/user/project/subdir/file.py'
    var_20 = var_6.search(var_19)
    var_21 = module_0.Trie()
    var_22 = '/path/to/config.json'
    var_23 = {}
    var_24 = var_21.insert(var_22, var_23)
    var_25 = '/path/to/file.py'
    var_26 = var_21.search(var_25)
    var_27 = module_0.Trie()
    var_28 = '/home/config.json'
    var_29 = 'version'
    var_30 = '1'
    var_31 = {var_29: var_30}
    var_32 = '2'
    var_33 = {var_29: var_32}
    var_34 = var_27.insert(var_28, var_31)
    var_35 = var_27.insert(var_28, var_33)
    var_36 = '/home/file.py'
    var_37 = var_27.search(var_36)
    var_38 = module_0.Trie()
    var_39 = '/a/b/c/d/e/config.json'
    var_40 = 'nested'
    var_41 = True
    var_42 = {var_40: var_41}
    var_43 = var_38.insert(var_39, var_42)
    var_44 = '/a/b/c/d/e/file.py'
    var_45 = var_38.search(var_44)



# Parsed testcases at query #30
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = '/config.yaml'
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/home/user/.config'
    var_11 = 'level'
    var_12 = 1
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/home/user/.config/file.py'
    var_16 = var_9.search(var_15)
    var_17 = module_0.Trie()
    var_18 = '/home/.config'
    var_19 = 'home'
    var_20 = {var_11: var_19}
    var_21 = var_17.insert(var_18, var_20)
    var_22 = 'user'
    var_23 = {var_11: var_22}
    var_24 = var_17.insert(var_10, var_23)
    var_25 = '/home/user/project/file.py'
    var_26 = var_17.search(var_25)
    var_27 = module_0.Trie()
    var_28 = {var_11: var_19}
    var_29 = var_27.insert(var_18, var_28)
    var_30 = var_27.search(var_25)
    var_31 = module_0.Trie()
    var_32 = '/other/path/.config'
    var_33 = 'other'
    var_34 = {var_11: var_33}
    var_35 = var_31.insert(var_32, var_34)
    var_36 = '/home/user/file.py'
    var_37 = var_31.search(var_36)
    var_38 = module_0.Trie()
    var_39 = {var_11: var_12}
    var_40 = var_38.insert(var_18, var_39)
    var_41 = 2
    var_42 = {var_11: var_41}
    var_43 = var_38.insert(var_10, var_42)
    var_44 = '/home/user/project/.config'
    var_45 = 3
    var_46 = {var_11: var_45}
    var_47 = var_38.insert(var_44, var_46)
    var_48 = '/home/user/project/src/file.py'
    var_49 = var_38.search(var_48)
    var_50 = module_0.Trie()
    var_51 = {var_11: var_22}
    var_52 = var_50.insert(var_10, var_51)
    var_53 = '/home/other/file.py'
    var_54 = var_50.search(var_53)
    var_55 = '/.config'
    var_56 = 'root'
    var_57 = True
    var_58 = {var_56: var_57}
    var_59 = module_0.Trie(var_55, var_58)
    var_60 = True
    var_61 = {var_22: var_60}
    var_62 = var_59.insert(var_10, var_61)
    var_63 = '/home/file.py'
    var_64 = var_59.search(var_63)
    var_65 = module_0.Trie()
    var_66 = {}
    var_67 = var_65.insert(var_18, var_66)
    var_68 = var_65.search(var_36)



# Parsed testcases at query #31
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'path/to/config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 123
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)
    var_13 = module_0.TrieNode()
    var_14 = 'child.json'
    var_15 = 'child'
    var_16 = 'data'
    var_17 = {var_15: var_16}
    var_18 = module_0.TrieNode(var_14, var_17)
    var_19 = 'config.json'
    var_20 = {}
    var_21 = module_0.TrieNode(var_19, var_20)
    var_22 = 'version'
    var_23 = 'settings'
    var_24 = 'paths'
    var_25 = '1.0'
    var_26 = 'debug'
    var_27 = 'timeout'
    var_28 = True
    var_29 = 30
    var_30 = {var_26: var_28, var_27: var_29}
    var_31 = '/path/1'
    var_32 = '/path/2'
    var_33 = [var_31, var_32]
    var_34 = {var_22: var_25, var_23: var_30, var_24: var_33}
    var_35 = 'complex.json'
    var_36 = module_0.TrieNode(var_35, var_34)



# Parsed testcases at query #32
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = module_0.Trie(var_2)
    var_4 = 'key'
    var_5 = 'nested'
    var_6 = 'value'
    var_7 = 'inner'
    var_8 = 'data'
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = module_0.Trie(config_data=var_10)
    var_12 = '/path/to/config.json'
    var_13 = 'setting1'
    var_14 = 'setting2'
    var_15 = 123
    var_16 = 'test'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = module_0.Trie(var_12, var_17)
    var_19 = module_0.Trie()
    var_20 = var_19.root
    var_21 = 'nodes'
    var_22 = hasattr(var_20, var_21)
    var_23 = var_19.root
    var_24 = 'config_info'
    var_25 = hasattr(var_23, var_24)
    var_26 = var_19.root.nodes
    var_27 = var_19.root.config_info
    var_28 = var_19.root.config_info
    var_29 = len(var_28)
    assert var_29 == 2



# Parsed testcases at query #33
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'data'
    var_7 = 123
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = var_10.root
    var_12 = module_0.Trie(var_2)
    var_13 = var_12.root
    var_14 = {}
    var_15 = module_0.Trie(var_2, var_14)
    var_16 = var_15.root
    var_17 = 'config1.json'
    var_18 = 'a'
    var_19 = 1
    var_20 = {var_18: var_19}
    var_21 = module_0.Trie(var_17, var_20)
    var_22 = 'config2.json'
    var_23 = 'b'
    var_24 = 2
    var_25 = {var_23: var_24}
    var_26 = module_0.Trie(var_22, var_25)



# Parsed testcases at query #34
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/project/file.py'
    var_7 = var_0.search(var_6)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = 'level'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = '/home/user/project/config.json'
    var_6 = 2
    var_7 = {var_2: var_6}
    var_8 = var_0.insert(var_1, var_4)
    var_9 = var_0.insert(var_5, var_7)
    var_10 = '/home/user/project/subdir/file.py'
    var_11 = var_0.search(var_10)
    var_12 = '/home/user/other/file.py'
    var_13 = var_0.search(var_12)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = '/home/user/file.py'
    var_5 = var_0.search(var_4)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = 'nested'
    var_3 = 'list'
    var_4 = 'number'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = 42
    var_13 = {var_2: var_7, var_3: var_11, var_4: var_12}
    var_14 = var_0.insert(var_1, var_13)
    var_15 = '/home/user/file.py'
    var_16 = var_0.search(var_15)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = 'version'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_5}
    var_7 = var_0.insert(var_1, var_4)
    var_8 = var_0.insert(var_1, var_6)
    var_9 = '/home/user/file.py'
    var_10 = var_0.search(var_9)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/a/b/c/d/config.json'
    var_2 = 'deep'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/a/b/c/d/file.py'
    var_7 = var_0.search(var_6)



# Parsed testcases at query #35
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = 'Test the insert method of Trie class'
    var_1 = module_0.Trie()
    var_2 = '/home/user/project/config.yaml'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_1.insert(var_2, var_7)
    var_9 = '/home/user/project/file.py'
    var_10 = var_1.search(var_9)
    var_11 = '/home/user/config.yaml'
    var_12 = 'key3'
    var_13 = 'value3'
    var_14 = {var_12: var_13}
    var_15 = var_1.insert(var_11, var_14)
    var_16 = var_1.search(var_9)
    var_17 = '/home/user/other/file.py'
    var_18 = var_1.search(var_17)
    var_19 = '/home/user/project/subdir/config.yaml'
    var_20 = {}
    var_21 = var_1.insert(var_19, var_20)
    var_22 = '/home/user/project/subdir/file.py'
    var_23 = var_1.search(var_22)
    var_24 = '/home/user/project/config.yaml'
    var_25 = 'updated'
    var_26 = True
    var_27 = {var_25: var_26}
    var_28 = var_1.insert(var_24, var_27)
    var_29 = var_1.search(var_9)
    var_30 = '/var/config.yaml'
    var_31 = 'nested'
    var_32 = 'list'
    var_33 = 'number'
    var_34 = 'key'
    var_35 = 'value'
    var_36 = {var_34: var_35}
    var_37 = 2
    var_38 = 3
    var_39 = [var_26, var_37, var_38]
    var_40 = 42
    var_41 = {var_31: var_36, var_32: var_39, var_33: var_40}
    var_42 = var_1.insert(var_30, var_41)
    var_43 = '/var/app/file.py'
    var_44 = var_1.search(var_43)



