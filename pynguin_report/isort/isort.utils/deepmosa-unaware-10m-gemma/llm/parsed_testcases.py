####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test/path/config.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'a'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'config.json'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = var_11.nodes



# Parsed testcases at query #3
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/config.json'
    var_2 = 'key'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/tmp/deep/dir/settings.yaml'
    var_8 = 'timeout'
    var_9 = 30
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.search(var_1)
    var_13 = var_0.search(var_7)
    var_14 = '/tmp/config.json'
    var_15 = 'new_value'
    var_16 = {var_2: var_15}
    var_17 = var_0.insert(var_14, var_16)
    var_18 = var_0.search(var_1)
    var_19 = '/non/existent/path'
    var_20 = var_0.search(var_19)



# Parsed testcases at query #4
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/usr/local/config.yaml'
    var_2 = 'key'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = -1
    var_8 = '/etc/settings.json'
    var_9 = 'timeout'
    var_10 = 30
    var_11 = {var_9: var_10}
    var_12 = var_0.insert(var_8, var_11)
    var_13 = var_0.search(var_1)
    var_14 = var_0.search(var_8)
    var_15 = 'new_value'
    var_16 = {var_2: var_15}
    var_17 = var_0.insert(var_1, var_16)
    var_18 = var_0.search(var_1)
    var_19 = '/etc/settings.json'



# Parsed testcases at query #5
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/configs/app.yaml'
    var_2 = 'debug'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = 0
    var_7 = '/tmp/configs/app.yaml'
    var_8 = 2
    var_9 = var_0.root
    var_10 = '/tmp/configs/sub/db.yaml'
    var_11 = 'host'
    var_12 = 'localhost'
    var_13 = {var_11: var_12}
    var_14 = var_0.insert(var_10, var_13)
    var_15 = var_0.root
    var_16 = '/tmp/configs/app.yaml'
    var_17 = False
    var_18 = {var_2: var_17}
    var_19 = var_0.insert(var_16, var_18)
    var_20 = var_0.root
    var_21 = '/tmp/configs/app.yaml'



# Parsed testcases at query #6
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project'
    var_2 = 'config.json'
    var_3 = 'theme'
    var_4 = 'dark'
    var_5 = {var_3: var_4}
    var_6 = 'subdir'
    var_7 = 'settings.yaml'
    var_8 = 'font'
    var_9 = 'serif'
    var_10 = {var_8: var_9}
    var_11 = 'deep'
    var_12 = 'module.py'
    var_13 = 'other.py'
    var_14 = 'other_dir'
    var_15 = 'file.py'
    var_16 = module_0.Trie()
    var_17 = '/tmp/random_file.py'



# Parsed testcases at query #7
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = 'key'
    var_3 = 'enabled'
    var_4 = 'value'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_0.insert(var_1, var_6)
    var_8 = var_0.root
    var_9 = 'new_value'
    var_10 = {var_2: var_9}
    var_11 = var_0.insert(var_1, var_10)
    var_12 = '/home/user/project/other.json'
    var_13 = 'a'
    var_14 = {var_13: var_5}
    var_15 = var_0.insert(var_12, var_14)
    var_16 = var_0.root



# Parsed testcases at query #8
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'base_config.json'
    var_3 = 'key'
    var_4 = 'enabled'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.Trie(var_2, var_7)
    var_9 = var_8.root.nodes
    var_10 = 'empty.json'
    var_11 = {}
    var_12 = module_0.Trie(var_10, var_11)



# Parsed testcases at query #9
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = '/tmp/test_trie'
    var_1 = 'config'
    var_2 = 'subdir'
    var_3 = 'project'
    var_4 = 'src'
    var_5 = 'main.py'
    var_6 = 'utils'
    var_7 = 'helper.py'
    var_8 = module_0.Trie()
    var_9 = 'version'
    var_10 = '1.0'
    var_11 = {var_9: var_10}
    var_12 = 'debug'
    var_13 = True
    var_14 = {var_12: var_13}
    var_15 = 'feature'
    var_16 = 'enabled'
    var_17 = {var_15: var_16}
    var_18 = 'settings.json'
    var_19 = 'config.yaml'
    var_20 = 'extra_config.json'
    var_21 = 'app.py'
    var_22 = 'other_file.txt'
    var_23 = 'random_file.py'



# Parsed testcases at query #10
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'debug'
    var_4 = 'value'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_0.insert(var_1, var_6)
    var_8 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/settings.yaml'
    var_2 = 'version'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_2: var_5}
    var_7 = var_0.insert(var_1, var_4)
    var_8 = var_0.insert(var_1, var_6)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/app/base.json'
    var_2 = '/app/subdir/child.json'
    var_3 = 'scope'
    var_4 = 'global'
    var_5 = {var_3: var_4}
    var_6 = var_0.insert(var_1, var_5)
    var_7 = 'local'
    var_8 = {var_3: var_7}
    var_9 = var_0.insert(var_2, var_8)
    var_10 = '/app/other.py'



# Parsed testcases at query #11
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/config.yaml'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/tmp/subdir/deep/config.json'
    var_8 = 'nested'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = var_0.root
    var_14 = 'updated'
    var_15 = {var_2: var_14}
    var_16 = var_0.insert(var_1, var_15)
    var_17 = var_0.root



# Parsed testcases at query #12
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = '/path/to/config.yaml'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root.nodes
    var_8 = 'test.txt'
    var_9 = None
    var_10 = module_0.Trie(var_8, var_9)



# Parsed testcases at query #13
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/project'
    var_2 = 'configs'
    var_3 = 'subfolder'
    var_4 = 'app'
    var_5 = 'env'
    var_6 = 'prod'
    var_7 = {var_5: var_6}
    var_8 = 'debug'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = 'feature'
    var_12 = 'enabled'
    var_13 = {var_11: var_12}
    var_14 = 'module.py'
    var_15 = 'file.txt'
    var_16 = 'nested'
    var_17 = 'other'
    var_18 = 'unrelated'
    var_19 = 'sibling.py'



# Parsed testcases at query #14
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/project'
    var_2 = 'config.json'
    var_3 = 'subdir'
    var_4 = 'module.py'
    var_5 = 'nested'
    var_6 = 'version'
    var_7 = '1.0'
    var_8 = {var_6: var_7}
    var_9 = 'feature'
    var_10 = '2.0'
    var_11 = 'enabled'
    var_12 = {var_6: var_10, var_9: var_11}
    var_13 = 'other'
    var_14 = 'file.txt'
    var_15 = module_0.Trie()



# Parsed testcases at query #15
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test/path/config.yaml'
    var_2 = 'key'
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.TrieNode(var_1, var_8)
    var_10 = 'only_path.json'
    var_11 = module_0.TrieNode(var_10)
    var_12 = 'foo'
    var_13 = 'bar'
    var_14 = {var_12: var_13}
    var_15 = module_0.TrieNode(config_data=var_14)



# Parsed testcases at query #16
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/project'
    var_2 = 'config'
    var_3 = 'sub'
    var_4 = 'src'
    var_5 = 'app'
    var_6 = 'env'
    var_7 = 'version'
    var_8 = 'dev'
    var_9 = 1
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'prod'
    var_12 = 2
    var_13 = {var_6: var_11, var_7: var_12}
    var_14 = 'feature_x'
    var_15 = True
    var_16 = {var_14: var_15}
    var_17 = 'settings.yaml'
    var_18 = 'module'
    var_19 = 'utils.py'
    var_20 = 'other'
    var_21 = 'random.txt'
    var_22 = 'main.py'

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = 'key'
    var_3 = 'val'
    var_4 = {var_2: var_3}
    var_5 = module_0.TrieNode(var_1, var_4)



# Parsed testcases at query #17
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test/path/config.yaml'
    var_2 = 'key'
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.TrieNode(var_1, var_8)
    var_10 = 'empty.json'
    var_11 = {}
    var_12 = module_0.TrieNode(var_10, var_11)
    var_13 = 'none.txt'
    var_14 = None
    var_15 = module_0.TrieNode(var_13, var_14)



# Parsed testcases at query #18
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = 'env'
    var_3 = 'version'
    var_4 = 'prod'
    var_5 = 1
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_0.insert(var_1, var_6)
    var_8 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/settings.json'
    var_2 = 'key'
    var_3 = 'val1'
    var_4 = {var_2: var_3}
    var_5 = 'val2'
    var_6 = {var_2: var_5}
    var_7 = var_0.insert(var_1, var_4)
    var_8 = var_0.insert(var_1, var_6)

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/a/b/c/config.ini'
    var_2 = 'depth'
    var_3 = 'deep'
    var_4 = {var_2: var_3}
    var_5 = '/a/b/config.ini'
    var_6 = 'shallow'
    var_7 = {var_2: var_6}
    var_8 = var_0.insert(var_1, var_4)
    var_9 = var_0.insert(var_5, var_7)
    var_10 = '/a/b/c/other.txt'
    var_11 = var_0.search(var_10)
    var_12 = '/a/b/other.txt'
    var_13 = var_0.search(var_12)



# Parsed testcases at query #19
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/app'
    var_2 = 'config'
    var_3 = 'src'
    var_4 = 'modules'
    var_5 = 'utils.py'
    var_6 = 'version'
    var_7 = '1.0'
    var_8 = {var_6: var_7}
    var_9 = 'debug'
    var_10 = True
    var_11 = {var_9: var_10}
    var_12 = 'feature'
    var_13 = 'enabled'
    var_14 = {var_12: var_13}
    var_15 = 'other.py'
    var_16 = module_0.Trie()
    var_17 = module_0.Trie()
    var_18 = 'env'
    var_19 = 'prod'
    var_20 = {var_18: var_19}
    var_21 = 'dev'
    var_22 = {var_18: var_21}
    var_23 = 'module'
    var_24 = 'file.py'
    var_25 = module_0.Trie()
    var_26 = 'level'
    var_27 = 0
    var_28 = {var_26: var_27}
    var_29 = 'unrelated'
    var_30 = {var_26: var_10}
    var_31 = 'related'



# Parsed testcases at query #20
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/tmp/config.yaml'
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



# Parsed testcases at query #21
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



# Parsed testcases at query #22
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
    var_7 = '/home/user/project/settings.json'
    var_8 = 'theme'
    var_9 = 'dark'
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = '/home/user'
    var_13 = 'env'
    var_14 = 'prod'
    var_15 = {var_13: var_14}
    var_16 = var_0.insert(var_12, var_15)
    var_17 = 'root_config.py'
    var_18 = 'root'
    var_19 = {var_18: var_4}
    var_20 = var_0.insert(var_17, var_19)
    var_21 = module_0.Trie()
    var_22 = '/non/existent/path.txt'
    var_23 = var_21.search(var_22)



# Parsed testcases at query #23
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/usr/local/project/config.yaml'
    var_2 = 'version'
    var_3 = 'debug'
    var_4 = 1
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_0.insert(var_1, var_6)
    var_8 = var_0.root
    var_9 = -1
    var_10 = 2
    var_11 = {var_2: var_10}
    var_12 = var_0.insert(var_1, var_11)
    var_13 = '/usr/local/project/sub/settings.json'
    var_14 = 'theme'
    var_15 = 'dark'
    var_16 = {var_14: var_15}
    var_17 = var_0.insert(var_13, var_16)
    var_18 = var_0.root
    var_19 = var_0.root.config_info



# Parsed testcases at query #24
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test/path/config.yaml'
    var_2 = 'key'
    var_3 = 'num'
    var_4 = 'value'
    var_5 = 42
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.TrieNode(var_1, var_6)
    var_8 = 'only_path.json'
    var_9 = None
    var_10 = module_0.TrieNode(var_8, var_9)



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test/path/config.yaml'
    var_2 = 'key'
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.TrieNode(var_1, var_8)
    var_10 = 'only_file.json'
    var_11 = None
    var_12 = module_0.TrieNode(var_10, var_11)



# Parsed testcases at query #27
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.yaml'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = 'test.json'
    var_8 = {}
    var_9 = module_0.Trie(var_7, var_8)



# Parsed testcases at query #28
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/usr/local/project/config.json'
    var_2 = 'debug'
    var_3 = 'port'
    var_4 = True
    var_5 = 8080
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_0.insert(var_1, var_6)
    var_8 = var_0.root
    var_9 = '/usr/local/other/settings.yaml'
    var_10 = 'timeout'
    var_11 = 30
    var_12 = {var_10: var_11}
    var_13 = var_0.insert(var_9, var_12)
    var_14 = '/usr/local/project/config.json'
    var_15 = var_0.search(var_14)
    var_16 = var_0.root
    var_17 = 'Path parts for second config were not inserted correctly'



# Parsed testcases at query #29
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/project'
    var_2 = 'config'
    var_3 = 'sub'
    var_4 = 'env'
    var_5 = 'prod'
    var_6 = {var_4: var_5}
    var_7 = 'debug'
    var_8 = True
    var_9 = {var_7: var_8}
    var_10 = 'app.py'
    var_11 = 'settings.py'
    var_12 = 'other'
    var_13 = 'file.txt'
    var_14 = 'deep'
    var_15 = 'module.py'



# Parsed testcases at query #30
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = 'debug'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/home/user/project/subdir/settings.json'
    var_8 = 'port'
    var_9 = 8080
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = '/home/user/project/config.yaml'
    var_14 = False
    var_15 = {var_2: var_14}
    var_16 = var_0.insert(var_13, var_15)
    var_17 = var_0.root
    var_18 = '/home/user/project/subdir/other_file.txt'
    var_19 = '/home/user/project/random.txt'



# Parsed testcases at query #31
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.yaml'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = 'config.json'
    var_8 = {}
    var_9 = module_0.Trie(var_7, var_8)



# Parsed testcases at query #32
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'config.yaml'
    var_3 = 'key'
    var_4 = 'version'
    var_5 = 'value'
    var_6 = 1
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.Trie(var_2, var_7)
    var_9 = var_0.root.nodes
    var_10 = len(var_9)
    assert var_10 == 0



# Parsed testcases at query #33
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
    var_8 = False
    var_9 = {var_2: var_8}
    var_10 = var_0.insert(var_1, var_9)
    var_11 = '/tmp/settings.json'
    var_12 = 'env'
    var_13 = 'prod'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = '/home/user/project/sub/dir/deep.cfg'
    var_17 = 'key'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = var_0.insert(var_16, var_19)
    var_21 = var_0.root
    var_22 = '/home/user/project/src/main.py'



# Parsed testcases at query #34
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = ''
    var_2 = None
    var_3 = module_0.TrieNode(var_1, var_2)
    var_4 = 1
    var_5 = var_3.config_info[var_4]
    var_6 = 'key'
    var_7 = 'number'
    var_8 = 'value'
    var_9 = 123
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = '/path/to/config.yaml'
    var_12 = module_0.TrieNode(var_11, var_10)
    var_13 = 'test.json'
    var_14 = {}
    var_15 = module_0.TrieNode(var_13, var_14)



# Parsed testcases at query #35
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/project/settings.json'
    var_2 = 'env'
    var_3 = 'prod'
    var_4 = {var_2: var_3}
    var_5 = '/tmp/project/subfolder/app.json'
    var_6 = 'debug'
    var_7 = False
    var_8 = {var_6: var_7}
    var_9 = '/tmp/project/subfolder/deep/module.py'
    var_10 = '/tmp/other/file.txt'
    var_11 = '/tmp/project/other.py'
    var_12 = '/tmp/random.py'



# Parsed testcases at query #36
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'base_config.json'
    var_3 = module_0.Trie(var_2)
    var_4 = 'key'
    var_5 = 'nested'
    var_6 = 'value'
    var_7 = 'a'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = module_0.Trie(config_data=var_10)
    var_12 = 'app.yaml'
    var_13 = 'debug'
    var_14 = True
    var_15 = {var_13: var_14}
    var_16 = module_0.Trie(var_12, var_15)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = var_0.root
    var_8 = [var_7]
    var_9 = (var_1, var_4)
    var_10 = 'new_value'
    var_11 = {var_2: var_10}
    var_12 = var_0.insert(var_1, var_11)
    var_13 = var_0.root
    var_14 = '/tmp/a/b/c/d/e.txt'
    var_15 = 'depth'
    var_16 = 'deep'
    var_17 = {var_15: var_16}
    var_18 = var_0.insert(var_14, var_17)
    var_19 = '/tmp/other/config.json'
    var_20 = 'branch'
    var_21 = 'other'
    var_22 = {var_20: var_21}
    var_23 = var_0.insert(var_19, var_22)



# Parsed testcases at query #2
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.yaml'
    var_3 = 'key'
    var_4 = 'timeout'
    var_5 = 'value'
    var_6 = 30
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.Trie(var_2, var_7)
    var_9 = 'empty'
    var_10 = {}
    var_11 = module_0.Trie(var_9, var_10)



# Parsed testcases at query #3
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = 'debug'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = 'version'
    var_8 = False
    var_9 = {var_2: var_8, var_7: var_3}
    var_10 = var_0.insert(var_1, var_9)
    var_11 = '/home/user/project/subfolder/settings.json'
    var_12 = 'theme'
    var_13 = 'dark'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = var_0.root
    var_17 = var_0.root



# Parsed testcases at query #4
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'a'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = 'empty.json'
    var_12 = {}
    var_13 = module_0.Trie(var_11, var_12)



# Parsed testcases at query #5
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = {}
    var_3 = module_0.TrieNode(var_1, var_2)
    var_4 = 'key'
    var_5 = 'nested'
    var_6 = 'value'
    var_7 = 'a'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = '/path/to/config.yaml'
    var_12 = module_0.TrieNode(var_11, var_10)
    var_13 = 'id'
    var_14 = {var_13: var_8}
    var_15 = module_0.TrieNode(var_7, var_14)
    var_16 = 'b'
    var_17 = module_0.TrieNode(var_16, var_14)
    var_18 = 'none_test'
    var_19 = None
    var_20 = module_0.TrieNode(var_18, var_19)



# Parsed testcases at query #6
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'config.yaml'
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'a'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.Trie(var_2, var_9)
    var_11 = var_10.root.nodes



# Parsed testcases at query #7
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
    var_7 = '/home/user/project'
    var_8 = 'env'
    var_9 = 'prod'
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = '/home/user/project/subdir/app.py'
    var_13 = module_0.Trie()
    var_14 = '/tmp/test_dir'
    var_15 = 'config.json'
    var_16 = 'version'
    var_17 = 1
    var_18 = {var_16: var_17}
    var_19 = 'subdir'
    var_20 = 'file.py'
    var_21 = 2
    var_22 = {var_16: var_21}
    var_23 = '/tmp/other_dir'
    var_24 = 'branch'
    var_25 = 'other'
    var_26 = {var_24: var_25}
    var_27 = {var_16: var_21}
    var_28 = {var_24: var_25}



# Parsed testcases at query #8
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'api_key'
    var_3 = 'debug'
    var_4 = '12345'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '/home/user/project/subfolder/settings.yaml'
    var_8 = 'timeout'
    var_9 = 30
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_1, var_6)
    var_12 = var_0.insert(var_7, var_10)
    var_13 = '/home/user/project/other_file.txt'
    var_14 = var_0.search(var_13)
    var_15 = '/home/user/project/subfolder/deep_file.txt'
    var_16 = var_0.search(var_15)
    var_17 = '/home/user/project/subfolder/deep/extra.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/home/user/other.txt'
    var_20 = var_0.search(var_19)
    var_21 = '/home/user/new_config.json'
    var_22 = 'version'
    var_23 = '1.0'
    var_24 = {var_22: var_23}
    var_25 = var_0.insert(var_21, var_24)
    var_26 = '/home/user/project/subfolder/deep/extra.txt'
    var_27 = var_0.search(var_26)



# Parsed testcases at query #9
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.TrieNode(var_1, var_4)
    var_6 = 'only_file.json'
    var_7 = module_0.TrieNode(var_6)
    var_8 = 'none_data.json'
    var_9 = None
    var_10 = module_0.TrieNode(var_8, var_9)



# Parsed testcases at query #10
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test_config.yaml'
    var_2 = 'key'
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.TrieNode(var_1, var_8)
    var_10 = 'only_file.json'
    var_11 = module_0.TrieNode(var_10)
    var_12 = 'none_data.txt'
    var_13 = None
    var_14 = module_0.TrieNode(var_12, var_13)



# Parsed testcases at query #11
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config/settings.yaml'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'a'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'test.json'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = module_0.TrieNode()
    var_13 = module_0.TrieNode()



# Parsed testcases at query #12
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/project'
    var_2 = 'config.json'
    var_3 = 'subdir'
    var_4 = 'subconfig.json'
    var_5 = 'env'
    var_6 = 'production'
    var_7 = {var_5: var_6}
    var_8 = 'debug'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = 'module.py'
    var_12 = 'other_module.py'
    var_13 = 'deep'
    var_14 = '/other/path/file.py'
    var_15 = '/unrelated/file.py'



# Parsed testcases at query #13
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/mock/project'
    var_2 = 'config'
    var_3 = 'services'
    var_4 = 'auth'
    var_5 = 'project'
    var_6 = 'main.py'
    var_7 = 'env'
    var_8 = 'prod'
    var_9 = {var_7: var_8}
    var_10 = 'timeout'
    var_11 = 30
    var_12 = {var_10: var_11}
    var_13 = 'retries'
    var_14 = 3
    var_15 = {var_13: var_14}
    var_16 = 'other'
    var_17 = 'scope'
    var_18 = {var_17: var_16}
    var_19 = 'unrelated'
    var_20 = 'file.txt'
    var_21 = 'other_service.py'
    var_22 = module_0.Trie()
    var_23 = 'a'
    var_24 = 'b'
    var_25 = 'c'
    var_26 = 'd'
    var_27 = 'level'
    var_28 = 1
    var_29 = {var_27: var_28}
    var_30 = 2
    var_31 = {var_27: var_30}



# Parsed testcases at query #14
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
    var_8 = True
    var_9 = {var_7: var_8}
    var_10 = '/tmp/project'
    var_11 = 'configs'
    var_12 = 'auth'
    var_13 = 'src'
    var_14 = 'module.py'
    var_15 = 'settings.yaml'
    var_16 = 'subdir'
    var_17 = 'file.txt'
    var_18 = 'other_module.py'
    var_19 = module_0.Trie()
    var_20 = 'random_file.py'



# Parsed testcases at query #15
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/usr/local/configs/app.yaml'
    var_2 = 'env'
    var_3 = 'port'
    var_4 = 'production'
    var_5 = 8080
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_0.insert(var_1, var_6)
    var_8 = var_0.root
    var_9 = 'development'
    var_10 = {var_2: var_9}
    var_11 = var_0.insert(var_1, var_10)
    var_12 = '/etc/settings.json'
    var_13 = 'debug'
    var_14 = True
    var_15 = {var_13: var_14}
    var_16 = var_0.insert(var_12, var_15)
    var_17 = var_0.root



# Parsed testcases at query #16
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/base'
    var_2 = '/base/subdir'
    var_3 = '/base/subdir/deep'
    var_4 = 'level'
    var_5 = 'root'
    var_6 = {var_4: var_5}
    var_7 = 'sub'
    var_8 = {var_4: var_7}
    var_9 = 'deep'
    var_10 = {var_4: var_9}
    var_11 = 'test_trie_root'
    var_12 = 'subdir'
    var_13 = 'file.txt'
    var_14 = 'other_file.txt'
    var_15 = 'root_only.txt'
    var_16 = '/tmp/completely/different/path'
    var_17 = 'nothing.txt'
    var_18 = 'unconfigured'
    var_19 = module_0.Trie()
    var_20 = 'any.txt'



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test/path/config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'num'
    var_5 = 'value'
    var_6 = 42
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'config.json'
    var_9 = module_0.TrieNode(var_8, var_7)



# Parsed testcases at query #19
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/tmp/subdir/nested.json'
    var_8 = 'sub'
    var_9 = 'data'
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = 'Path parts not found in Trie'
    var_14 = var_12.config_info
    var_15 = [var_14]
    var_16 = 'new'
    var_17 = {var_16: var_9}
    var_18 = var_0.insert(var_1, var_17)
    var_19 = var_0.root

import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/empty.json'
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = var_0.root



# Parsed testcases at query #20
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/'
    var_2 = 'home'
    var_3 = 'user'
    var_4 = 'project'
    var_5 = 'env'
    var_6 = 'debug'
    var_7 = 'dev'
    var_8 = True
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'prod'
    var_11 = False
    var_12 = {var_5: var_10, var_6: var_11}
    var_13 = 'main.py'
    var_14 = 'utils/helper.py'
    var_15 = 'tmp'
    var_16 = 'other.py'



# Parsed testcases at query #21
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test/path/config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'num'
    var_5 = 'value'
    var_6 = 42
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'path.py'
    var_9 = module_0.TrieNode(var_8, var_7)
    var_10 = var_0.nodes



# Parsed testcases at query #22
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
    var_12 = '/home/user/other/settings.yaml'
    var_13 = 'theme'
    var_14 = 'dark'
    var_15 = {var_13: var_14}
    var_16 = var_0.insert(var_12, var_15)
    var_17 = var_0.root
    var_18 = '/config.ini'
    var_19 = 'version'
    var_20 = {var_19: var_4}
    var_21 = var_0.insert(var_18, var_20)
    var_22 = var_0.root
    var_23 = var_0.search(var_1)
    var_24 = var_0.search(var_12)



# Parsed testcases at query #23
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = 'new_value'
    var_8 = {var_2: var_7}
    var_9 = var_0.insert(var_1, var_8)
    var_10 = var_0.root
    var_11 = '/etc/settings.yaml'
    var_12 = 'debug'
    var_13 = True
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = var_0.root



# Parsed testcases at query #24
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test/path/config.yaml'
    var_2 = 'key'
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.TrieNode(var_1, var_8)
    var_10 = 'simple.json'
    var_11 = None
    var_12 = module_0.TrieNode(var_10, var_11)



# Parsed testcases at query #25
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test_config.yaml'
    var_2 = 'key'
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.TrieNode(var_1, var_8)
    var_10 = 'only_path.json'
    var_11 = module_0.TrieNode(var_10)
    var_12 = {var_5: var_6}
    var_13 = 'f1'
    var_14 = module_0.TrieNode(var_13, var_12)
    var_15 = 'f2'
    var_16 = module_0.TrieNode(var_15, var_12)



# Parsed testcases at query #26
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/project'
    var_2 = 'config.json'
    var_3 = 'env'
    var_4 = 'prod'
    var_5 = {var_3: var_4}
    var_6 = 'src'
    var_7 = 'module'
    var_8 = 'debug'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = 'utils'
    var_12 = 'helper.py'
    var_13 = 'unknown'
    var_14 = 'file.py'
    var_15 = '/etc/other/config.conf'



# Parsed testcases at query #27
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = 'test.yaml'
    var_1 = module_0.TrieNode(var_0)
    var_2 = var_1.nodes
    var_3 = var_1.nodes
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = 'key'
    var_6 = 'nested'
    var_7 = 'value'
    var_8 = 'a'
    var_9 = 1
    var_10 = {var_8: var_9}
    var_11 = {var_5: var_7, var_6: var_10}
    var_12 = 'path/to/config.json'
    var_13 = module_0.TrieNode(var_12, var_11)
    var_14 = 'empty.yaml'
    var_15 = {}
    var_16 = module_0.TrieNode(var_14, var_15)



# Parsed testcases at query #28
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/app'
    var_2 = 'configs'
    var_3 = 'sub'
    var_4 = 'env'
    var_5 = 'debug'
    var_6 = 'dev'
    var_7 = True
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'prod'
    var_10 = False
    var_11 = {var_4: var_9, var_5: var_10}
    var_12 = 'settings.yaml'
    var_13 = 'extra.yaml'
    var_14 = 'module.py'
    var_15 = 'other'
    var_16 = 'random'
    var_17 = 'file.txt'
    var_18 = 'root_cfg'
    var_19 = 'root'
    var_20 = {var_19: var_7}
    var_21 = module_0.Trie(var_18, var_20)
    var_22 = 'nonexistent.py'



# Parsed testcases at query #29
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'configs'
    var_2 = 'sub'
    var_3 = 'app'
    var_4 = 'main.py'
    var_5 = module_0.Trie()
    var_6 = 'env'
    var_7 = 'prod'
    var_8 = {var_6: var_7}
    var_9 = 'debug'
    var_10 = 'db'
    var_11 = False
    var_12 = 'sqlite'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = 'feature_x'
    var_15 = True
    var_16 = {var_14: var_15}
    var_17 = 'dev'
    var_18 = {var_6: var_17}
    var_19 = 'module.py'
    var_20 = 'other.py'
    var_21 = 'utils.py'
    var_22 = '/tmp/completely_different/file.py'
    var_23 = 'file.py'



# Parsed testcases at query #30
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = ''
    var_2 = {}
    var_3 = module_0.TrieNode(var_1, var_2)
    var_4 = '/etc/config.yaml'
    var_5 = 'key'
    var_6 = 'timeout'
    var_7 = 'value'
    var_8 = 30
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_0.TrieNode(var_4, var_9)
    var_11 = var_10.nodes



# Parsed testcases at query #31
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.yaml'
    var_2 = {}
    var_3 = module_0.TrieNode(var_1, var_2)
    var_4 = 'key'
    var_5 = 'nested'
    var_6 = 'value'
    var_7 = 'a'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}
    var_11 = '/path/to/config.json'
    var_12 = module_0.TrieNode(var_11, var_10)
    var_13 = 'default.cfg'
    var_14 = None
    var_15 = module_0.TrieNode(var_13, var_14)



# Parsed testcases at query #32
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.yaml'
    var_2 = 'version'
    var_3 = 'debug'
    var_4 = 1
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_0.insert(var_1, var_6)
    var_8 = var_0.search(var_1)
    var_9 = '/home/user/project/subdir/settings.json'
    var_10 = 'theme'
    var_11 = 'dark'
    var_12 = {var_10: var_11}
    var_13 = var_0.insert(var_9, var_12)
    var_14 = 0
    var_15 = '/home/user/project/config.yaml'
    var_16 = trie.search(var_15)[var_14]
    var_17 = trie.search(var_9)[var_14]
    var_18 = trie.search(var_9)[var_5]
    var_19 = '/home/user/project/subdir/subsubdir/extra.py'
    var_20 = var_0.search(var_19)
    var_21 = '/tmp/other/file.txt'
    var_22 = var_0.search(var_21)
    var_23 = '/home/user/project/config.yaml'
    var_24 = 2
    var_25 = {var_2: var_24}
    var_26 = var_0.insert(var_23, var_25)
    var_27 = var_0.search(var_1)



# Parsed testcases at query #33
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test/path/config.yaml'
    var_2 = 'key'
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.TrieNode(var_1, var_8)
    var_10 = 'only_path.json'
    var_11 = None
    var_12 = module_0.TrieNode(var_10, var_11)



# Parsed testcases at query #34
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/tmp/app'
    var_2 = 'config.json'
    var_3 = 'env'
    var_4 = 'version'
    var_5 = 'production'
    var_6 = 1
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'services'
    var_9 = 'auth'
    var_10 = '.env'
    var_11 = 'db'
    var_12 = 'postgres'
    var_13 = {var_11: var_12}
    var_14 = 'utils'
    var_15 = 'settings.py'
    var_16 = 'debug'
    var_17 = True
    var_18 = {var_16: var_17}
    var_19 = 'dummy.txt'
    var_20 = 'some_file.py'
    var_21 = 'module.py'
    var_22 = '/tmp/other/file.txt'
    var_23 = 'file.txt'



# Parsed testcases at query #35
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
    var_10 = 'config.yaml'
    var_11 = module_0.TrieNode(var_10, var_9)
    var_12 = var_0.nodes



