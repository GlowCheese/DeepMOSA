####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/test_app'
    var_2 = 'config'
    var_3 = 'submodule'
    var_4 = 'env'
    var_5 = 'base'
    var_6 = {var_4: var_5}
    var_7 = 'debug'
    var_8 = 'sub'
    var_9 = True
    var_10 = {var_4: var_8, var_7: var_9}
    var_11 = 'leaf'
    var_12 = {var_4: var_11}
    var_13 = 'settings.json'
    var_14 = 'config.json'
    var_15 = 'extra.json'
    var_16 = 'module.py'
    var_17 = 'another_file.py'
    var_18 = 'other_module.py'
    var_19 = 'other_branch'
    var_20 = 'file.py'



# Parsed testcases at query #2
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'debug'
    var_3 = 'port'
    var_4 = True
    var_5 = 8080
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_0.insert(var_1, var_6)
    var_8 = var_0.root
    var_9 = False
    var_10 = {var_2: var_9}
    var_11 = var_0.insert(var_1, var_10)
    var_12 = '/tmp/other/settings.yaml'
    var_13 = 'env'
    var_14 = 'prod'
    var_15 = {var_13: var_14}
    var_16 = var_0.insert(var_12, var_15)
    var_17 = var_0.root
    var_18 = '/home/user/project/subdir/subfile.py'



# Parsed testcases at query #3
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/a/config.py'
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = '/tmp/b/config.py'
    var_6 = 'b'
    var_7 = 2
    var_8 = {var_6: var_7}
    var_9 = var_0.insert(var_1, var_4)
    var_10 = var_0.insert(var_5, var_8)
    var_11 = var_0.root
    var_12 = var_0.root



# Parsed testcases at query #4
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'num'
    var_5 = 'value'
    var_6 = 42
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = 'a'
    var_10 = 1
    var_11 = {var_9: var_10}
    var_12 = 'file.txt'
    var_13 = module_0.TrieNode(var_12, var_11)



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = module_0.Trie(var_2)
    var_4 = 'key'
    var_5 = 'version'
    var_6 = 'value'
    var_7 = 1
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = module_0.Trie(config_data=var_8)
    var_10 = module_0.Trie(var_2, var_8)



# Parsed testcases at query #7
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'a'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.TrieNode(config_data=var_9)
    var_11 = module_0.TrieNode(var_1, var_9)



# Parsed testcases at query #8
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'config'
    var_2 = 'sub'
    var_3 = 'module'
    var_4 = 'script.py'
    var_5 = 'other.py'
    var_6 = 'other'
    var_7 = 'random.py'
    var_8 = 'env'
    var_9 = 'dev'
    var_10 = {var_8: var_9}
    var_11 = 'debug'
    var_12 = 'prod'
    var_13 = True
    var_14 = {var_8: var_12, var_11: var_13}
    var_15 = module_0.Trie()
    var_16 = 'root_conf'
    var_17 = {var_16: var_13}



# Parsed testcases at query #9
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test_config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'id'
    var_5 = 'value'
    var_6 = 123
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'path/to/config.yaml'
    var_9 = module_0.TrieNode(var_8, var_7)



# Parsed testcases at query #10
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'a'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'path/to/config.json'
    var_11 = module_0.TrieNode(var_10, var_9)



# Parsed testcases at query #11
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = 'debug'
    var_3 = 'version'
    var_4 = True
    var_5 = {var_2: var_4, var_3: var_4}
    var_6 = var_0.insert(var_1, var_5)
    var_7 = var_0.root
    var_8 = '/home/user/project/subfolder/settings.json'
    var_9 = 'theme'
    var_10 = 'dark'
    var_11 = {var_9: var_10}
    var_12 = var_0.insert(var_8, var_11)
    var_13 = var_0.root
    var_14 = var_0.root
    var_15 = False
    var_16 = {var_2: var_15}
    var_17 = var_0.insert(var_1, var_16)
    var_18 = var_0.root



# Parsed testcases at query #12
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/home/user/other/settings.yaml'
    var_8 = 'debug'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = var_0.search(var_1)
    var_14 = var_0.search(var_7)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/test/config.ini'
    var_2 = 'version'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_5}
    var_7 = var_0.insert(var_1, var_4)
    var_8 = var_0.insert(var_1, var_6)
    var_9 = '/tmp/test/other_file.txt'
    var_10 = '/tmp/dir/file.txt'



# Parsed testcases at query #13
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'a'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = module_0.TrieNode()
    var_12 = module_0.TrieNode()



# Parsed testcases at query #14
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = 'value2'
    var_8 = {var_2: var_7}
    var_9 = var_0.insert(var_1, var_8)
    var_10 = '/home/user/other/settings.yaml'
    var_11 = 'theme'
    var_12 = 'dark'
    var_13 = {var_11: var_12}
    var_14 = var_0.insert(var_10, var_13)
    var_15 = var_0.root
    var_16 = '/home/user/project/subdir/new_file.txt'



# Parsed testcases at query #15
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'env'
    var_2 = 'prod'
    var_3 = {var_1: var_2}
    var_4 = 'debug'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = 'feature'
    var_8 = 'enabled'
    var_9 = {var_7: var_8}
    var_10 = '/tmp/project'
    var_11 = 'src'
    var_12 = 'utils'
    var_13 = 'core'
    var_14 = 'config.json'
    var_15 = 'settings.json'
    var_16 = 'app.json'
    var_17 = 'random_file.txt'
    var_18 = 'module.py'
    var_19 = 'logic.py'
    var_20 = 'other.py'



# Parsed testcases at query #16
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = 'env'
    var_3 = 'debug'
    var_4 = 'dev'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_0.insert(var_1, var_6)
    var_8 = var_0.root
    var_9 = 'prod'
    var_10 = False
    var_11 = {var_2: var_9, var_3: var_10}
    var_12 = var_0.insert(var_1, var_11)
    var_13 = var_0.root
    var_14 = '/home/user/other/settings.json'
    var_15 = 'theme'
    var_16 = 'dark'
    var_17 = {var_15: var_16}
    var_18 = var_0.insert(var_14, var_17)
    var_19 = var_0.root
    var_20 = 'other'
    var_21 = var_19.nodes[var_20]



# Parsed testcases at query #17
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = '/tmp/test_trie'
    var_1 = 'env'
    var_2 = 'base'
    var_3 = {var_1: var_2}
    var_4 = 'debug'
    var_5 = 'sub'
    var_6 = True
    var_7 = {var_1: var_5, var_4: var_6}
    var_8 = 'feature'
    var_9 = 'deep'
    var_10 = 'enabled'
    var_11 = {var_1: var_9, var_8: var_10}
    var_12 = 'config.json'
    var_13 = 'file.py'
    var_14 = 'other'
    var_15 = module_0.Trie()
    var_16 = '/tmp/unknown/path/file.py'



# Parsed testcases at query #18
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test_config.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'a'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.TrieNode(config_data=var_9)
    var_11 = module_0.TrieNode(var_1, var_9)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test_config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'a'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'path/to/config.yaml'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = 'child'
    var_13 = var_0.nodes[var_12]



# Parsed testcases at query #2
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/home/user/project/subdir/settings.yaml'
    var_8 = 'timeout'
    var_9 = 30
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = var_0.root
    var_14 = '/home/user/project/config.json'
    var_15 = 'new_value'
    var_16 = {var_2: var_15}
    var_17 = var_0.insert(var_14, var_16)
    var_18 = var_0.root



# Parsed testcases at query #3
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'num'
    var_5 = 'value'
    var_6 = 1
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(config_data=var_7)
    var_9 = 'path/to/config.yaml'
    var_10 = 'a'
    var_11 = {var_10: var_6}
    var_12 = module_0.TrieNode(var_9, var_11)



# Parsed testcases at query #4
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/test_project'
    var_2 = 'subdir'
    var_3 = 'deep'
    var_4 = 'config.json'
    var_5 = 'level'
    var_6 = 'name'
    var_7 = 1
    var_8 = 'first'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 2
    var_11 = 'second'
    var_12 = {var_5: var_10, var_6: var_11}
    var_13 = 'module.py'
    var_14 = 'other'
    var_15 = 'file.txt'
    var_16 = 'another_file.py'
    var_17 = 'completely_different'
    var_18 = 'file.py'
    var_19 = '/non/existent/path/file.py'
    var_20 = var_0.search(var_19)



# Parsed testcases at query #5
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
    var_7 = 'a'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = module_0.Trie(config_data=var_10)
    var_12 = module_0.Trie(var_2, var_10)



# Parsed testcases at query #6
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/project'
    var_2 = 'config.json'
    var_3 = 'src'
    var_4 = 'utils'
    var_5 = 'version'
    var_6 = '1.0'
    var_7 = {var_5: var_6}
    var_8 = 'feature'
    var_9 = '1.1'
    var_10 = 'enabled'
    var_11 = {var_5: var_9, var_8: var_10}
    var_12 = 'debug'
    var_13 = '1.2'
    var_14 = True
    var_15 = {var_5: var_13, var_12: var_14}
    var_16 = 'module.py'
    var_17 = 'main.py'
    var_18 = 'helper.py'
    var_19 = 'other'
    var_20 = 'file.py'
    var_21 = 'data'
    var_22 = {var_19: var_21}
    var_23 = {var_19: var_21}



# Parsed testcases at query #7
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'config.yaml'
    var_3 = module_0.Trie(var_2)
    var_4 = 'key'
    var_5 = 'nested'
    var_6 = 'value'
    var_7 = 'a'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = 'settings.json'
    var_12 = module_0.Trie(var_11, var_10)



# Parsed testcases at query #8
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test_config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'num'
    var_5 = 'value'
    var_6 = 123
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(config_data=var_7)
    var_9 = 'path/to/config.yaml'
    var_10 = 'enabled'
    var_11 = True
    var_12 = {var_10: var_11}
    var_13 = module_0.TrieNode(var_9, var_12)



# Parsed testcases at query #9
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'config.yaml'
    var_3 = module_0.Trie(var_2)
    var_4 = 'key'
    var_5 = 'num'
    var_6 = 'value'
    var_7 = 42
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = module_0.Trie(config_data=var_8)
    var_10 = module_0.Trie(var_2, var_8)



# Parsed testcases at query #10
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'base_config.yaml'
    var_3 = module_0.Trie(var_2)
    var_4 = 'key'
    var_5 = 'nested'
    var_6 = 'value'
    var_7 = 'a'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = module_0.Trie(config_data=var_10)



# Parsed testcases at query #11
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'debug'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/home/user/project/sub/settings.yaml'
    var_8 = 'port'
    var_9 = 8080
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = '/home/user/project/config.json'
    var_14 = False
    var_15 = {var_2: var_14}
    var_16 = var_0.insert(var_13, var_15)
    var_17 = var_0.root



# Parsed testcases at query #12
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'env'
    var_3 = 'debug'
    var_4 = 'dev'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_0.insert(var_1, var_6)
    var_8 = var_0.root
    var_9 = '/home/user/project/sub/module/settings.yaml'
    var_10 = 'timeout'
    var_11 = 30
    var_12 = {var_10: var_11}
    var_13 = var_0.insert(var_9, var_12)
    var_14 = var_0.root
    var_15 = '/home/user/project/config.json'
    var_16 = 'prod'
    var_17 = {var_2: var_16}
    var_18 = var_0.insert(var_15, var_17)
    var_19 = var_0.root
    var_20 = var_0.root



# Parsed testcases at query #13
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.yaml'
    var_2 = 'key'
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.TrieNode(var_1, var_8)
    var_10 = 'test.json'
    var_11 = None
    var_12 = module_0.TrieNode(var_10, var_11)



# Parsed testcases at query #14
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'config.yaml'
    var_3 = module_0.Trie(var_2)
    var_4 = 'key'
    var_5 = 'nested'
    var_6 = 'value'
    var_7 = 'a'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = module_0.Trie(config_data=var_10)
    var_12 = module_0.Trie(var_2, var_10)



# Parsed testcases at query #15
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'debug'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/home/user/project/subdir/subconfig.json'
    var_8 = 'api_key'
    var_9 = 'secret'
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = '/tmp/other/config.yaml'
    var_14 = 'version'
    var_15 = {var_14: var_3}
    var_16 = var_0.insert(var_13, var_15)
    var_17 = var_0.root



# Parsed testcases at query #16
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'id'
    var_5 = 'value'
    var_6 = 123
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'settings.json'
    var_9 = module_0.TrieNode(var_8, var_7)
    var_10 = var_9.nodes
    var_11 = var_9.nodes
    var_12 = len(var_11)
    assert var_12 == 0



# Parsed testcases at query #17
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/usr/local/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = 'new'
    var_8 = 'data'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_1, var_9)
    var_11 = '/tmp/test.yaml'
    var_12 = 'version'
    var_13 = 1
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = var_0.root
    var_17 = '/a/b/c/d/e/f.cfg'
    var_18 = 'deep'
    var_19 = True
    var_20 = {var_18: var_19}
    var_21 = var_0.insert(var_17, var_20)
    var_22 = var_0.root



# Parsed testcases at query #18
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
    var_7 = 'a'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = module_0.Trie(config_data=var_10)
    var_12 = module_0.Trie(var_2, var_10)



# Parsed testcases at query #19
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.yaml'
    var_2 = 'key'
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.TrieNode(var_1, var_8)
    var_10 = 'test.json'
    var_11 = None
    var_12 = module_0.TrieNode(var_10, var_11)



# Parsed testcases at query #20
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.yaml'
    var_3 = module_0.Trie(var_2)
    var_4 = 'key'
    var_5 = 'num'
    var_6 = 'value'
    var_7 = 123
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = module_0.Trie(config_data=var_8)
    var_10 = module_0.Trie(var_2, var_8)



# Parsed testcases at query #21
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
    var_7 = 'new_value'
    var_8 = {var_2: var_7}
    var_9 = var_0.insert(var_1, var_8)
    var_10 = '/tmp/other/settings.yaml'
    var_11 = 'debug'
    var_12 = True
    var_13 = {var_11: var_12}
    var_14 = var_0.insert(var_10, var_13)
    var_15 = var_0.root



# Parsed testcases at query #22
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'base.json'
    var_3 = module_0.Trie(var_2)
    var_4 = 'key'
    var_5 = 'nested'
    var_6 = 'value'
    var_7 = 'a'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = module_0.Trie(config_data=var_10)



# Parsed testcases at query #23
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'env'
    var_3 = 'debug'
    var_4 = 'dev'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_0.insert(var_1, var_6)
    var_8 = var_0.root
    var_9 = 'prod'
    var_10 = {var_2: var_9}
    var_11 = var_0.insert(var_1, var_10)
    var_12 = '/tmp/other/settings.yaml'
    var_13 = 'version'
    var_14 = {var_13: var_5}
    var_15 = var_0.insert(var_12, var_14)
    var_16 = var_0.root



# Parsed testcases at query #24
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.root
    var_6 = '/tmp/subdir/nested.json'
    var_7 = 'nested'
    var_8 = 'data'
    var_9 = {var_7: var_8}
    var_10 = var_0.root
    var_11 = var_0.root



