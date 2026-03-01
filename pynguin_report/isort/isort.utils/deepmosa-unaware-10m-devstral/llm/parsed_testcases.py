####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #2
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #3
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #4
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'to'
    var_13 = var_9.nodes[var_12]
    var_14 = var_13.nodes
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'config.json'
    var_17 = var_13.nodes[var_16]
    var_18 = '/path/to/new_config.json'
    var_19 = 'new_key'
    var_20 = 'new_value'
    var_21 = {var_19: var_20}
    var_22 = var_0.insert(var_18, var_21)
    var_23 = var_0.root.nodes[var_8]
    var_24 = var_23.nodes[var_12]
    var_25 = var_24.nodes
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = 'new_config.json'
    var_28 = var_0.root.nodes[var_8]
    var_29 = var_28.nodes[var_12]
    var_30 = var_29.nodes[var_27]
    var_31 = '/another/path/config.json'
    var_32 = 'another_key'
    var_33 = 'another_value'
    var_34 = {var_32: var_33}
    var_35 = var_0.insert(var_31, var_34)
    var_36 = var_0.root.nodes
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = 'another'
    var_39 = var_0.root.nodes[var_38]
    var_40 = var_39.nodes
    var_41 = len(var_40)
    assert var_41 == 1
    var_42 = var_39.nodes[var_8]
    var_43 = var_42.nodes
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = var_42.nodes[var_16]



# Parsed testcases at query #5
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_0.root



# Parsed testcases at query #6
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config1.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/root/subdir/config2.json'
    var_7 = 'subdir_value'
    var_8 = {var_1: var_7}
    var_9 = var_0.insert(var_6, var_8)
    var_10 = '/root/subdir/subsubdir/config3.json'
    var_11 = 'subsubdir_value'
    var_12 = {var_1: var_11}
    var_13 = var_0.insert(var_10, var_12)
    var_14 = var_0.search(var_4)
    var_15 = '/root/subdir/file.txt'
    var_16 = var_0.search(var_15)
    var_17 = '/root/subdir/subsubdir/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/root/default.json'
    var_20 = 'default'
    var_21 = 'config'
    var_22 = {var_20: var_21}
    var_23 = '/nonexistent/path/file.txt'
    var_24 = var_0.search(var_23)
    var_25 = module_0.Trie()
    var_26 = '/any/path'
    var_27 = var_25.search(var_26)



# Parsed testcases at query #7
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/another/path/config2.json'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/path/to/subdir/config3.json'
    var_12 = 'key3'
    var_13 = 'value3'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)



# Parsed testcases at query #8
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)



# Parsed testcases at query #9
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/root/config1.json'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/root/subdir/config2.json'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/root/subdir/subsubdir/config3.json'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = '/root/file.txt'
    var_17 = var_0.search(var_16)
    var_18 = '/root/subdir/file.txt'
    var_19 = var_0.search(var_18)
    var_20 = '/root/subdir/subsubdir/file.txt'
    var_21 = var_0.search(var_20)
    var_22 = '/nonexistent/file.txt'
    var_23 = var_0.search(var_22)
    var_24 = '/root/subdir/another_file.txt'
    var_25 = var_0.search(var_24)
    var_26 = module_0.Trie()
    var_27 = '/any/path/file.txt'
    var_28 = var_26.search(var_27)



# Parsed testcases at query #10
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = {}
    var_9 = module_0.Trie(var_2, var_8)
    var_10 = var_9.root



# Parsed testcases at query #11
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #12
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/path'
    var_2 = var_0.search(var_1)
    var_3 = '/root_config.json'
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = '/file.txt'
    var_9 = var_7.search(var_8)
    var_10 = module_0.Trie()
    var_11 = '/a/b/config1.json'
    var_12 = 'value1'
    var_13 = {var_4: var_12}
    var_14 = var_10.insert(var_11, var_13)
    var_15 = '/a/config2.json'
    var_16 = 'value2'
    var_17 = {var_4: var_16}
    var_18 = var_10.insert(var_15, var_17)
    var_19 = '/config3.json'
    var_20 = 'value3'
    var_21 = {var_4: var_20}
    var_22 = var_10.insert(var_19, var_21)
    var_23 = '/a/b/c/file.txt'
    var_24 = var_10.search(var_23)
    var_25 = '/a/file.txt'
    var_26 = var_10.search(var_25)
    var_27 = var_10.search(var_8)
    var_28 = '/root.json'
    var_29 = 'root'
    var_30 = 'data'
    var_31 = {var_29: var_30}
    var_32 = module_0.Trie(var_28, var_31)
    var_33 = '/a/b/config.json'
    var_34 = 'nested'
    var_35 = {var_34: var_30}
    var_36 = var_32.insert(var_33, var_35)
    var_37 = '/x/y/z/file.txt'
    var_38 = var_32.search(var_37)
    var_39 = module_0.Trie()
    var_40 = 'exact'
    var_41 = 'match'
    var_42 = {var_40: var_41}
    var_43 = var_39.insert(var_33, var_42)
    var_44 = var_39.search(var_33)



# Parsed testcases at query #13
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.cfg'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #14
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/root/config1.json'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/root/subdir/config2.json'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/root/subdir/subsubdir/config3.json'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = '/root/file.txt'
    var_17 = var_0.search(var_16)
    var_18 = '/root/subdir/file.txt'
    var_19 = var_0.search(var_18)
    var_20 = '/root/subdir/subsubdir/file.txt'
    var_21 = var_0.search(var_20)
    var_22 = '/root/nonexistent/file.txt'
    var_23 = var_0.search(var_22)
    var_24 = '/root/subdir/nonexistent/file.txt'
    var_25 = var_0.search(var_24)
    var_26 = '/nonexistent/file.txt'
    var_27 = var_0.search(var_26)
    var_28 = module_0.Trie()
    var_29 = '/any/path/file.txt'
    var_30 = var_28.search(var_29)



# Parsed testcases at query #15
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #16
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #17
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #18
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #19
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/path/to/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = var_0.search(var_4)
    var_7 = '/path/to/subdir/file.txt'
    var_8 = var_0.search(var_7)
    var_9 = module_0.Trie()
    var_10 = '/nonexistent/file.txt'
    var_11 = var_9.search(var_10)
    var_12 = '/path/to/subdir/deeper_config.json'
    var_13 = 'deeper'
    var_14 = 'config'
    var_15 = {var_13: var_14}
    var_16 = var_0.insert(var_12, var_15)
    var_17 = '/path/to/subdir/deeper/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/path/TO/UPPER_CONFIG.json'
    var_20 = 'upper'
    var_21 = 'case'
    var_22 = {var_20: var_21}
    var_23 = var_0.insert(var_19, var_22)
    var_24 = '/path/to/upper_config.json'
    var_25 = var_0.search(var_24)



# Parsed testcases at query #20
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/root/config1.json'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/root/subdir/config2.json'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/root/subdir/subsubdir/config3.json'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = '/root/file.txt'
    var_17 = var_0.search(var_16)
    var_18 = '/root/subdir/file.txt'
    var_19 = var_0.search(var_18)
    var_20 = '/root/subdir/subsubdir/file.txt'
    var_21 = var_0.search(var_20)
    var_22 = '/root/subdir/other/file.txt'
    var_23 = var_0.search(var_22)
    var_24 = '/root/other/file.txt'
    var_25 = var_0.search(var_24)
    var_26 = '/other/file.txt'
    var_27 = var_0.search(var_26)
    var_28 = module_0.Trie()
    var_29 = '/any/path/file.txt'
    var_30 = var_28.search(var_29)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root



# Parsed testcases at query #2
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #3
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = ''
    var_2 = {}
    var_3 = var_0.insert(var_1, var_2)
    var_4 = module_0.Trie()
    var_5 = '/a/b/config.json'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = var_4.insert(var_5, var_8)
    var_10 = module_0.Trie()
    var_11 = '/a/b/config1.json'
    var_12 = 'key1'
    var_13 = 'value1'
    var_14 = {var_12: var_13}
    var_15 = var_10.insert(var_11, var_14)
    var_16 = '/a/b/c/config2.json'
    var_17 = 'key2'
    var_18 = 'value2'
    var_19 = {var_17: var_18}
    var_20 = var_10.insert(var_16, var_19)
    var_21 = module_0.Trie()
    var_22 = {var_6: var_7}
    var_23 = var_21.insert(var_5, var_22)
    var_24 = '/a/b'
    var_25 = 'new_value'
    var_26 = {var_6: var_25}
    var_27 = var_21.insert(var_24, var_26)
    var_28 = module_0.Trie()
    var_29 = 'C:\\Users\\config.json'
    var_30 = {var_6: var_7}
    var_31 = var_28.insert(var_29, var_30)



# Parsed testcases at query #4
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = module_0.Trie()
    var_8 = '/path/to/nested/config.json'
    var_9 = 'nested'
    var_10 = 'data'
    var_11 = {var_9: var_10}
    var_12 = var_7.insert(var_8, var_11)
    var_13 = var_7.root
    var_14 = module_0.Trie()
    var_15 = '/path/to/config1.json'
    var_16 = 'key1'
    var_17 = 'value1'
    var_18 = {var_16: var_17}
    var_19 = var_14.insert(var_15, var_18)
    var_20 = '/path/to/config2.json'
    var_21 = 'key2'
    var_22 = 'value2'
    var_23 = {var_21: var_22}
    var_24 = var_14.insert(var_20, var_23)
    var_25 = var_14.root
    var_26 = var_14.root



# Parsed testcases at query #5
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = module_0.Trie()
    var_8 = '/path/to/config1.json'
    var_9 = 'key1'
    var_10 = 'value1'
    var_11 = {var_9: var_10}
    var_12 = var_7.insert(var_8, var_11)
    var_13 = '/path/to/subdir/config2.json'
    var_14 = 'key2'
    var_15 = 'value2'
    var_16 = {var_14: var_15}
    var_17 = var_7.insert(var_13, var_16)
    var_18 = var_7.root
    var_19 = var_7.root
    var_20 = module_0.Trie()
    var_21 = '/path/to/config.json'
    var_22 = {var_9: var_10}
    var_23 = var_20.insert(var_21, var_22)
    var_24 = '/path/to/config.json'
    var_25 = {var_14: var_15}
    var_26 = var_20.insert(var_24, var_25)
    var_27 = var_20.root



# Parsed testcases at query #6
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #7
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/path/to/another_config.json'
    var_7 = 'another_key'
    var_8 = 'another_value'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/different/path/config.json'
    var_12 = 'different_key'
    var_13 = 'different_value'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = '/empty/config.json'
    var_17 = {}
    var_18 = var_0.insert(var_16, var_17)
    var_19 = '/a/b/c/d/e/config.json'
    var_20 = 'nested'
    var_21 = {var_20: var_3}
    var_22 = var_0.insert(var_19, var_21)



# Parsed testcases at query #8
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'config.json'
    var_9 = 'to'
    var_10 = 'path'
    var_11 = var_0.root.nodes[var_10]
    var_12 = var_11.nodes[var_9]
    var_13 = var_12.nodes[var_8]
    var_14 = '/another/path/config.yaml'
    var_15 = 'another'
    var_16 = 'data'
    var_17 = {var_15: var_16}
    var_18 = var_0.insert(var_14, var_17)
    var_19 = var_0.root.nodes
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = '/path/to/nested/config.toml'
    var_22 = 'nested'
    var_23 = 'config'
    var_24 = {var_22: var_23}
    var_25 = var_0.insert(var_21, var_24)
    var_26 = 'config.toml'
    var_27 = var_0.root.nodes[var_10]
    var_28 = var_27.nodes[var_9]
    var_29 = var_28.nodes[var_22]
    var_30 = var_29.nodes[var_26]
    var_31 = ''
    var_32 = 'empty'
    var_33 = {var_32: var_10}
    var_34 = var_0.insert(var_31, var_33)



# Parsed testcases at query #9
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #10
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config1.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/root/subdir/config2.json'
    var_7 = 'value2'
    var_8 = {var_1: var_7}
    var_9 = var_0.insert(var_6, var_8)
    var_10 = '/root/subdir/subsubdir/config3.json'
    var_11 = 'value3'
    var_12 = {var_1: var_11}
    var_13 = var_0.insert(var_10, var_12)
    var_14 = '/root/file.txt'
    var_15 = var_0.search(var_14)
    var_16 = '/root/subdir/file.txt'
    var_17 = var_0.search(var_16)
    var_18 = '/root/subdir/subsubdir/file.txt'
    var_19 = var_0.search(var_18)
    var_20 = '/nonexistent/file.txt'
    var_21 = var_0.search(var_20)
    var_22 = '/root/subdir/other/file.txt'
    var_23 = var_0.search(var_22)
    var_24 = module_0.Trie()
    var_25 = '/any/path/file.txt'
    var_26 = var_24.search(var_25)



# Parsed testcases at query #11
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/root/config1.json'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/root/subdir/config2.json'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/root/subdir/subsubdir/config3.json'
    var_12 = 'key3'
    var_13 = 'value3'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = '/root/subdir/file.txt'
    var_17 = '/root/subdir/subsubdir/file.txt'
    var_18 = '/nonexistent/file.txt'
    var_19 = module_0.Trie()
    var_20 = '/any/path'
    var_21 = '/root_config.json'
    var_22 = 'root'
    var_23 = 'config'
    var_24 = {var_22: var_23}
    var_25 = module_0.Trie(var_21, var_24)



# Parsed testcases at query #12
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'config.json'
    var_9 = 'to'
    var_10 = 'path'
    var_11 = var_0.root.nodes[var_10]
    var_12 = var_11.nodes[var_9]
    var_13 = var_12.nodes[var_8]
    var_14 = '/another/path/config.yaml'
    var_15 = 'another'
    var_16 = 'data'
    var_17 = {var_15: var_16}
    var_18 = var_0.insert(var_14, var_17)
    var_19 = var_0.root.nodes
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = 'new'
    var_22 = {var_21: var_16}
    var_23 = var_0.insert(var_1, var_22)
    var_24 = var_0.root.nodes[var_10]
    var_25 = var_24.nodes[var_9]
    var_26 = var_25.nodes[var_8]
    var_27 = '/empty/config'
    var_28 = {}
    var_29 = var_0.insert(var_27, var_28)
    var_30 = 'config'
    var_31 = 'empty'
    var_32 = var_0.root.nodes[var_31]
    var_33 = var_32.nodes[var_30]



# Parsed testcases at query #13
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #14
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/etc/config.yaml'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/home/user/project/.config.yaml'
    var_7 = 'project_key'
    var_8 = 'project_value'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/home/user/.config.yaml'
    var_12 = 'user_key'
    var_13 = 'user_value'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = '/home/user/project/src/main.py'
    var_17 = '/var/log/app.log'
    var_18 = module_0.Trie()
    var_19 = '/any/path'
    var_20 = '/home/.config.yaml'
    var_21 = 'home_key'
    var_22 = 'home_value'
    var_23 = {var_21: var_22}
    var_24 = var_0.insert(var_20, var_23)
    var_25 = '/home/user/file.txt'



# Parsed testcases at query #15
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #16
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #17
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/root/config1.json'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/root/subdir/config2.json'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/root/subdir/subsubdir/config3.json'
    var_12 = 'key3'
    var_13 = 'value3'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = '/root/subdir/subsubdir/other_file.txt'
    var_17 = '/root/other_file.txt'
    var_18 = module_0.Trie()
    var_19 = '/some/non/existent/path/file.txt'
    var_20 = '/root/subdir/another_file.txt'



# Parsed testcases at query #18
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = {}
    var_9 = module_0.Trie(var_5, var_8)
    var_10 = var_9.root



# Parsed testcases at query #19
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root



# Parsed testcases at query #20
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #21
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #22
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'any/path'
    var_5 = var_0.search(var_4)
    var_6 = '/home/user/config.yaml'
    var_7 = var_0.insert(var_6, var_3)
    var_8 = var_0.search(var_6)
    var_9 = '/home/config.yaml'
    var_10 = 'parent'
    var_11 = {var_10: var_2}
    var_12 = var_0.insert(var_9, var_11)
    var_13 = '/home/user/file.txt'
    var_14 = var_0.search(var_13)
    var_15 = '/home/user/project/config.yaml'
    var_16 = 'project'
    var_17 = {var_16: var_2}
    var_18 = var_0.insert(var_15, var_17)
    var_19 = '/home/user/project/src/file.py'
    var_20 = var_0.search(var_19)
    var_21 = 'root'
    var_22 = {var_21: var_2}
    var_23 = ''
    var_24 = module_0.Trie(var_23, var_22)
    var_25 = '/non/existent/path'
    var_26 = var_24.search(var_25)
    var_27 = '/etc/app/config.yaml'
    var_28 = 'app'
    var_29 = {var_28: var_2}
    var_30 = var_0.insert(var_27, var_29)
    var_31 = '/etc/app/data/file.txt'
    var_32 = var_0.search(var_31)
    var_33 = '/etc/other/file.txt'
    var_34 = var_0.search(var_33)



# Parsed testcases at query #23
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #24
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)



# Parsed testcases at query #25
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = module_0.Trie()
    var_8 = '/another/path/file.yaml'
    var_9 = {}
    var_10 = var_7.insert(var_8, var_9)
    var_11 = var_7.root



# Parsed testcases at query #26
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #27
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #28
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/.config'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = '/home/user/.config'
    var_6 = 'key2'
    var_7 = 'value2'
    var_8 = {var_6: var_7}
    var_9 = var_0.insert(var_1, var_4)
    var_10 = var_0.insert(var_5, var_8)
    var_11 = '/home/user/project/file.txt'
    var_12 = var_0.search(var_11)
    var_13 = '/home/user/documents/file.txt'
    var_14 = var_0.search(var_13)
    var_15 = '/root/file.txt'
    var_16 = var_0.search(var_15)
    var_17 = '/home/user/project/src/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = module_0.Trie()
    var_20 = '/any/path'
    var_21 = var_19.search(var_20)



# Parsed testcases at query #29
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #30
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/another/path/config2.json'
    var_8 = 'key2'
    var_9 = 'value2'
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = var_0.root



# Parsed testcases at query #31
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.txt'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #32
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/root/config1.json'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/root/subdir/config2.json'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/root/subdir/subsubdir/config3.json'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = var_0.search(var_14)
    var_17 = '/root/subdir/subsubdir/otherfile.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/root/subdir/file.txt'
    var_20 = var_0.search(var_19)
    var_21 = '/root/otherfile.txt'
    var_22 = var_0.search(var_21)
    var_23 = '/nonexistent/path/file.txt'
    var_24 = var_0.search(var_23)
    var_25 = module_0.Trie()
    var_26 = '/any/path/file.txt'
    var_27 = var_25.search(var_26)
    var_28 = '/root/CaseSensitive/config4.json'
    var_29 = 'key4'
    var_30 = 'value4'
    var_31 = {var_29: var_30}
    var_32 = var_0.insert(var_28, var_31)
    var_33 = '/root/casesensitive/file.txt'
    var_34 = var_0.search(var_33)



# Parsed testcases at query #33
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = {}
    var_9 = module_0.Trie(var_5, var_8)
    var_10 = var_9.root



# Parsed testcases at query #34
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #35
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'to'
    var_13 = var_9.nodes[var_12]
    var_14 = var_13.nodes
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'config.json'
    var_17 = var_13.nodes[var_16]
    var_18 = '/another/path/config.yaml'
    var_19 = 'another_key'
    var_20 = 'another_value'
    var_21 = {var_19: var_20}
    var_22 = var_0.insert(var_18, var_21)
    var_23 = var_0.root.nodes
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'another'
    var_26 = var_0.root.nodes[var_25]
    var_27 = var_26.nodes
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = var_26.nodes[var_8]
    var_30 = var_29.nodes
    var_31 = len(var_30)
    assert var_31 == 1
    var_32 = 'config.yaml'
    var_33 = var_29.nodes[var_32]



# Parsed testcases at query #36
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/root/config1.json'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/root/subdir/config2.json'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/root/subdir/subsubdir/config3.json'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = '/root/file.txt'
    var_17 = var_0.search(var_16)
    var_18 = '/root/subdir/file.txt'
    var_19 = var_0.search(var_18)
    var_20 = '/root/subdir/subsubdir/file.txt'
    var_21 = var_0.search(var_20)
    var_22 = '/root/subdir/nonexistent/file.txt'
    var_23 = var_0.search(var_22)
    var_24 = '/nonexistent/file.txt'
    var_25 = var_0.search(var_24)
    var_26 = module_0.Trie()
    var_27 = '/any/path/file.txt'
    var_28 = var_26.search(var_27)



# Parsed testcases at query #37
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'any/file.txt'
    var_5 = var_0.search(var_4)
    var_6 = '/root/config.json'
    var_7 = var_0.insert(var_6, var_3)
    var_8 = '/root/file.txt'
    var_9 = var_0.search(var_8)
    var_10 = '/root/subdir/file.txt'
    var_11 = var_0.search(var_10)
    var_12 = 'nested_key'
    var_13 = 'nested_value'
    var_14 = {var_12: var_13}
    var_15 = '/root/subdir/config.json'
    var_16 = var_0.insert(var_15, var_14)
    var_17 = var_0.search(var_10)
    var_18 = '/root/subdir/deep/file.txt'
    var_19 = var_0.search(var_18)
    var_20 = '/root/other/file.txt'
    var_21 = var_0.search(var_20)
    var_22 = '/other/path/file.txt'
    var_23 = var_0.search(var_22)
    var_24 = '/home/user/.config'
    var_25 = 'user'
    var_26 = 'test'
    var_27 = {var_25: var_26}
    var_28 = var_0.insert(var_24, var_27)
    var_29 = '/home/user/docs/file.txt'
    var_30 = var_0.search(var_29)
    var_31 = '/home/other/file.txt'
    var_32 = var_0.search(var_31)



# Parsed testcases at query #38
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'to'
    var_13 = var_9.nodes[var_12]
    var_14 = var_13.nodes
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = '/path/to/another_config.json'
    var_17 = 'another_key'
    var_18 = 'another_value'
    var_19 = {var_17: var_18}
    var_20 = var_0.insert(var_16, var_19)
    var_21 = var_13.nodes
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = '/path/to/common/child_config.json'
    var_24 = 'child_key'
    var_25 = 'child_value'
    var_26 = {var_24: var_25}
    var_27 = var_0.insert(var_23, var_26)
    var_28 = var_13.nodes
    var_29 = len(var_28)
    assert var_29 == 3
    var_30 = 'common'
    var_31 = var_13.nodes[var_30]
    var_32 = var_31.nodes
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = '/another/path/config.json'
    var_35 = 'root_key'
    var_36 = 'root_value'
    var_37 = {var_35: var_36}
    var_38 = var_0.insert(var_34, var_37)
    var_39 = var_0.root.nodes
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = 'another'
    var_42 = var_0.root.nodes[var_41]
    var_43 = var_42.nodes
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = var_42.nodes[var_8]
    var_46 = var_45.nodes
    var_47 = len(var_46)
    assert var_47 == 1



# Parsed testcases at query #39
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #40
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/path'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = '/home/user/.config'
    var_8 = var_3.insert(var_7, var_6)
    var_9 = '/home/user/.config/file.txt'
    var_10 = var_3.search(var_9)
    var_11 = module_0.Trie()
    var_12 = 'parent'
    var_13 = 'config'
    var_14 = {var_12: var_13}
    var_15 = 'child'
    var_16 = {var_15: var_13}
    var_17 = '/home/user'
    var_18 = var_11.insert(var_17, var_14)
    var_19 = '/home/user/project'
    var_20 = var_11.insert(var_19, var_16)
    var_21 = '/home/user/project/src/file.py'
    var_22 = var_11.search(var_21)
    var_23 = '/home/user/other/file.py'
    var_24 = var_11.search(var_23)
    var_25 = module_0.Trie()
    var_26 = '/root/config'
    var_27 = 'root'
    var_28 = {var_27: var_13}
    var_29 = var_25.insert(var_26, var_28)
    var_30 = '/home/user/file.txt'
    var_31 = var_25.search(var_30)
    var_32 = module_0.Trie()
    var_33 = 'level'
    var_34 = 1
    var_35 = {var_33: var_34}
    var_36 = 2
    var_37 = {var_33: var_36}
    var_38 = 3
    var_39 = {var_33: var_38}
    var_40 = '/a'
    var_41 = var_32.insert(var_40, var_35)
    var_42 = '/a/b'
    var_43 = var_32.insert(var_42, var_37)
    var_44 = '/a/b/c'
    var_45 = var_32.insert(var_44, var_39)
    var_46 = '/a/b/c/d/file.txt'
    var_47 = var_32.search(var_46)
    var_48 = '/a/b/file.txt'
    var_49 = var_32.search(var_48)
    var_50 = '/a/file.txt'
    var_51 = var_32.search(var_50)
    var_52 = module_0.Trie()
    var_53 = {var_27: var_13}
    var_54 = '/root'
    var_55 = var_52.insert(var_54, var_53)
    var_56 = '/root/file.txt'
    var_57 = var_52.search(var_56)
    var_58 = '/other/file.txt'
    var_59 = var_52.search(var_58)
    var_60 = module_0.Trie()
    var_61 = 'os'
    var_62 = 'windows'
    var_63 = {var_61: var_62}
    var_64 = 'C:\\Users\\config'
    var_65 = var_60.insert(var_64, var_63)
    var_66 = 'C:\\Users\\config\\file.txt'
    var_67 = var_60.search(var_66)



# Parsed testcases at query #41
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #42
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/.config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/project/file.txt'
    var_7 = var_0.search(var_6)
    var_8 = module_0.Trie()
    var_9 = '/home/user/.config.json'
    var_10 = {var_2: var_3}
    var_11 = var_8.insert(var_9, var_10)
    var_12 = '/home/user/project/subdir/file.txt'
    var_13 = var_8.search(var_12)
    var_14 = module_0.Trie()
    var_15 = var_14.search(var_6)
    var_16 = module_0.Trie()
    var_17 = '/home/.config.json'
    var_18 = 'key1'
    var_19 = 'value1'
    var_20 = {var_18: var_19}
    var_21 = '/home/user/.config.json'
    var_22 = 'key2'
    var_23 = 'value2'
    var_24 = {var_22: var_23}
    var_25 = var_16.insert(var_17, var_20)
    var_26 = var_16.insert(var_21, var_24)
    var_27 = var_16.search(var_6)
    var_28 = module_0.Trie()
    var_29 = '/home/user/.config.json'
    var_30 = {var_2: var_3}
    var_31 = var_28.insert(var_29, var_30)
    var_32 = '/home/user/project/subdir/nested/file.txt'
    var_33 = var_28.search(var_32)



# Parsed testcases at query #43
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = 'root_config.json'
    var_1 = 'root'
    var_2 = 'config'
    var_3 = {var_1: var_2}
    var_4 = module_0.Trie(var_0, var_3)
    var_5 = '/some/nonexistent/path/file.txt'
    var_6 = var_4.search(var_5)
    var_7 = module_0.Trie()
    var_8 = '/home/user/.config.json'
    var_9 = 'user'
    var_10 = {var_9: var_2}
    var_11 = var_7.insert(var_8, var_10)
    var_12 = '/home/.global_config.json'
    var_13 = 'global'
    var_14 = {var_13: var_2}
    var_15 = var_7.insert(var_12, var_14)
    var_16 = '/home/user/documents/file.txt'
    var_17 = var_7.search(var_16)
    var_18 = module_0.Trie()
    var_19 = '/etc/app/config.json'
    var_20 = 'app'
    var_21 = {var_20: var_2}
    var_22 = var_18.insert(var_19, var_21)
    var_23 = '/etc/app/data/file.txt'
    var_24 = var_18.search(var_23)
    var_25 = module_0.Trie()
    var_26 = '/any/path/file.txt'
    var_27 = var_25.search(var_26)
    var_28 = module_0.Trie()
    var_29 = '/root_config.json'
    var_30 = {var_1: var_2}
    var_31 = var_28.insert(var_29, var_30)
    var_32 = '/file.txt'
    var_33 = var_28.search(var_32)
    var_34 = module_0.Trie()
    var_35 = '/a/b/config.json'
    var_36 = 'b'
    var_37 = {var_36: var_2}
    var_38 = var_34.insert(var_35, var_37)
    var_39 = '/a/config.json'
    var_40 = 'a'
    var_41 = {var_40: var_2}
    var_42 = var_34.insert(var_39, var_41)
    var_43 = '/a/b/c/file.txt'
    var_44 = var_34.search(var_43)



# Parsed testcases at query #44
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'root_config.json'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/root/config.json'
    var_11 = 'root'
    var_12 = True
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/root/subdir/config.json'
    var_16 = 'subdir'
    var_17 = {var_16: var_12}
    var_18 = var_9.insert(var_15, var_17)
    var_19 = '/root/subdir/subsubdir/config.json'
    var_20 = 'subsubdir'
    var_21 = {var_20: var_12}
    var_22 = var_9.insert(var_19, var_21)
    var_23 = '/root/subdir/subsubdir/file.txt'
    var_24 = var_9.search(var_23)
    var_25 = '/root/subdir/file.txt'
    var_26 = var_9.search(var_25)
    var_27 = '/root/file.txt'
    var_28 = var_9.search(var_27)
    var_29 = module_0.Trie()
    var_30 = {var_11: var_12}
    var_31 = var_29.insert(var_10, var_30)
    var_32 = '/different/path/file.txt'
    var_33 = var_29.search(var_32)
    var_34 = module_0.Trie()
    var_35 = {var_11: var_12}
    var_36 = var_34.insert(var_10, var_35)
    var_37 = {var_16: var_12}
    var_38 = var_34.insert(var_15, var_37)
    var_39 = '/root/subdir/nonexistent/file.txt'
    var_40 = var_34.search(var_39)
    var_41 = module_0.Trie()
    var_42 = '/Root/config.json'
    var_43 = {var_11: var_12}
    var_44 = var_41.insert(var_42, var_43)
    var_45 = var_41.search(var_27)



# Parsed testcases at query #45
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/.config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/project/file.py'
    var_7 = var_0.search(var_6)
    var_8 = module_0.Trie()
    var_9 = '/home/user/.config.json'
    var_10 = {var_2: var_3}
    var_11 = var_8.insert(var_9, var_10)
    var_12 = var_8.search(var_6)
    var_13 = module_0.Trie()
    var_14 = '/home/.config.json'
    var_15 = 'value1'
    var_16 = {var_2: var_15}
    var_17 = '/home/user/.config.json'
    var_18 = 'value2'
    var_19 = {var_2: var_18}
    var_20 = var_13.insert(var_14, var_16)
    var_21 = var_13.insert(var_17, var_19)
    var_22 = var_13.search(var_6)
    var_23 = module_0.Trie()
    var_24 = var_23.search(var_6)
    var_25 = module_0.Trie()
    var_26 = '/home/.config.json'
    var_27 = {var_2: var_3}
    var_28 = var_25.insert(var_26, var_27)
    var_29 = var_25.search(var_6)
    var_30 = module_0.Trie()
    var_31 = '/home/user/project/.config.json'
    var_32 = {var_2: var_3}
    var_33 = var_30.insert(var_31, var_32)
    var_34 = '/home/user/project/subdir/file.py'
    var_35 = var_30.search(var_34)
    var_36 = module_0.Trie()
    var_37 = '/home/user/.config.json'
    var_38 = {var_2: var_3}
    var_39 = var_36.insert(var_37, var_38)
    var_40 = '/home/otheruser/project/file.py'
    var_41 = var_36.search(var_40)
    var_42 = module_0.Trie()
    var_43 = '/.config.json'
    var_44 = {var_2: var_3}
    var_45 = var_42.insert(var_43, var_44)
    var_46 = var_42.search(var_6)



# Parsed testcases at query #46
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.yaml'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = ''
    var_11 = {}
    var_12 = module_0.Trie(var_10, var_11)
    var_13 = var_12.root



# Parsed testcases at query #47
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #48
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root



# Parsed testcases at query #49
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/path/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = '/a/b/config.txt'
    var_8 = var_3.insert(var_7, var_6)
    var_9 = '/a/b/file.txt'
    var_10 = var_3.search(var_9)
    var_11 = module_0.Trie()
    var_12 = {var_4: var_5}
    var_13 = '/a/config.txt'
    var_14 = var_11.insert(var_13, var_12)
    var_15 = '/a/b/c/file.txt'
    var_16 = var_11.search(var_15)
    var_17 = module_0.Trie()
    var_18 = {var_4: var_5}
    var_19 = var_17.insert(var_7, var_18)
    var_20 = '/x/y/z/file.txt'
    var_21 = var_17.search(var_20)
    var_22 = module_0.Trie()
    var_23 = 'value1'
    var_24 = {var_4: var_23}
    var_25 = 'value2'
    var_26 = {var_4: var_25}
    var_27 = var_22.insert(var_13, var_24)
    var_28 = var_22.insert(var_7, var_26)
    var_29 = var_22.search(var_15)
    var_30 = module_0.Trie()
    var_31 = {var_4: var_5}
    var_32 = '/config.txt'
    var_33 = var_30.insert(var_32, var_31)
    var_34 = var_30.search(var_15)



# Parsed testcases at query #50
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #51
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #52
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #53
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #54
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #55
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #56
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #57
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #58
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'to'
    var_13 = var_9.nodes[var_12]
    var_14 = var_13.nodes
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'config.json'
    var_17 = var_13.nodes[var_16]
    var_18 = '/path/to/another_config.json'
    var_19 = 'another_key'
    var_20 = 'another_value'
    var_21 = {var_19: var_20}
    var_22 = var_0.insert(var_18, var_21)
    var_23 = var_9.nodes
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = '/different/path/config.json'
    var_26 = 'different_key'
    var_27 = 'different_value'
    var_28 = {var_26: var_27}
    var_29 = var_0.insert(var_25, var_28)
    var_30 = var_0.root.nodes
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = '/empty/config.json'
    var_33 = {}
    var_34 = var_0.insert(var_32, var_33)
    var_35 = 'empty'
    var_36 = var_0.root.nodes[var_35]
    var_37 = var_36.nodes[var_16]



# Parsed testcases at query #59
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'to'
    var_13 = var_9.nodes[var_12]
    var_14 = var_13.nodes
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'config.json'
    var_17 = var_13.nodes[var_16]
    var_18 = '/another/path/config.yaml'
    var_19 = 'another_key'
    var_20 = 'another_value'
    var_21 = {var_19: var_20}
    var_22 = var_0.insert(var_18, var_21)
    var_23 = var_0.root.nodes
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'another'
    var_26 = var_0.root.nodes[var_25]
    var_27 = var_26.nodes
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = var_26.nodes[var_8]
    var_30 = var_29.nodes
    var_31 = len(var_30)
    assert var_31 == 1
    var_32 = 'config.yaml'
    var_33 = var_29.nodes[var_32]



# Parsed testcases at query #60
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/.config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/project/file.txt'
    var_7 = var_0.search(var_6)
    var_8 = module_0.Trie()
    var_9 = '/home/user/.config.json'
    var_10 = {var_2: var_3}
    var_11 = var_8.insert(var_9, var_10)
    var_12 = '/home/user/project/subdir/file.txt'
    var_13 = var_8.search(var_12)
    var_14 = module_0.Trie()
    var_15 = '/home/user/project/file.txt'
    var_16 = var_14.search(var_15)
    var_17 = module_0.Trie()
    var_18 = '/home/user/.config.json'
    var_19 = 'value1'
    var_20 = {var_2: var_19}
    var_21 = var_17.insert(var_18, var_20)
    var_22 = '/home/user/project/.config.json'
    var_23 = 'value2'
    var_24 = {var_2: var_23}
    var_25 = var_17.insert(var_22, var_24)
    var_26 = '/home/user/project/file.txt'
    var_27 = var_17.search(var_26)
    var_28 = module_0.Trie()
    var_29 = '/home/user/.config.json'
    var_30 = {var_2: var_19}
    var_31 = var_28.insert(var_29, var_30)
    var_32 = '/home/user/project/subdir/.config.json'
    var_33 = {var_2: var_23}
    var_34 = var_28.insert(var_32, var_33)
    var_35 = '/home/user/project/file.txt'
    var_36 = var_28.search(var_35)



# Parsed testcases at query #61
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #62
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root



# Parsed testcases at query #63
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #64
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #65
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'config.py'
    var_5 = module_0.TrieNode(var_4, var_3)
    var_6 = module_0.TrieNode(var_4)
    var_7 = module_0.TrieNode(config_data=var_3)



# Parsed testcases at query #66
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/path/to/another/config.json'
    var_8 = 'value2'
    var_9 = {var_2: var_8}
    var_10 = var_0.insert(var_7, var_9)
    var_11 = var_0.root



# Parsed testcases at query #67
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #68
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #69
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #70
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #71
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #72
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'to'
    var_13 = var_9.nodes[var_12]
    var_14 = var_13.nodes
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'config.json'
    var_17 = var_13.nodes[var_16]



# Parsed testcases at query #73
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #74
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #75
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #76
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/another/path/config.json'
    var_8 = {}
    var_9 = var_0.insert(var_7, var_8)
    var_10 = var_0.root



# Parsed testcases at query #77
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/path'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = '/root_config.py'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = '/file.py'
    var_9 = var_7.search(var_8)
    var_10 = module_0.Trie()
    var_11 = '/config.py'
    var_12 = 'root'
    var_13 = True
    var_14 = {var_12: var_13}
    var_15 = var_10.insert(var_11, var_14)
    var_16 = '/a/config.py'
    var_17 = 'a'
    var_18 = {var_17: var_13}
    var_19 = var_10.insert(var_16, var_18)
    var_20 = '/a/b/config.py'
    var_21 = 'b'
    var_22 = {var_21: var_13}
    var_23 = var_10.insert(var_20, var_22)
    var_24 = '/a/b/c/file.py'
    var_25 = var_10.search(var_24)
    var_26 = '/a/file.py'
    var_27 = var_10.search(var_26)
    var_28 = var_10.search(var_8)
    var_29 = module_0.Trie()
    var_30 = {var_21: var_13}
    var_31 = var_29.insert(var_20, var_30)
    var_32 = '/x/y/file.py'
    var_33 = var_29.search(var_32)
    var_34 = module_0.Trie()
    var_35 = {var_21: var_13}
    var_36 = var_34.insert(var_20, var_35)
    var_37 = var_34.search(var_20)
    var_38 = module_0.Trie()
    var_39 = '/A/config.py'
    var_40 = 'A'
    var_41 = {var_40: var_13}
    var_42 = var_38.insert(var_39, var_41)
    var_43 = var_38.search(var_26)



# Parsed testcases at query #78
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config1.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/root/subdir/config2.json'
    var_7 = 'value2'
    var_8 = {var_1: var_7}
    var_9 = var_0.insert(var_6, var_8)
    var_10 = '/root/subdir/subsubdir/config3.json'
    var_11 = 'value3'
    var_12 = {var_1: var_11}
    var_13 = var_0.insert(var_10, var_12)
    var_14 = '/root/file.txt'
    var_15 = var_0.search(var_14)
    var_16 = '/root/subdir/file.txt'
    var_17 = var_0.search(var_16)
    var_18 = '/root/subdir/subsubdir/file.txt'
    var_19 = var_0.search(var_18)
    var_20 = '/root/subdir/nonexistent/file.txt'
    var_21 = var_0.search(var_20)
    var_22 = '/nonexistent/path/file.txt'
    var_23 = var_0.search(var_22)
    var_24 = '/root/anotherdir/config4.json'
    var_25 = 'value4'
    var_26 = {var_1: var_25}
    var_27 = var_0.insert(var_24, var_26)
    var_28 = '/root/anotherdir/subdir/file.txt'
    var_29 = var_0.search(var_28)



# Parsed testcases at query #79
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/root/config1.json'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/root/subdir/config2.json'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/root/subdir/subsubdir/config3.json'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = var_0.search(var_14)
    var_17 = '/root/subdir/subsubdir/other_file.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/root/subdir/file.txt'
    var_20 = var_0.search(var_19)
    var_21 = '/root/another_file.txt'
    var_22 = var_0.search(var_21)
    var_23 = '/nonexistent/path/file.txt'
    var_24 = var_0.search(var_23)
    var_25 = '/root_config.json'
    var_26 = 'root_key'
    var_27 = 'root_value'
    var_28 = {var_26: var_27}
    var_29 = module_0.Trie(var_25, var_28)
    var_30 = '/any/path/file.txt'
    var_31 = var_29.search(var_30)



# Parsed testcases at query #80
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #81
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)



# Parsed testcases at query #82
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/another/path/config.yaml'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/path/to/nested/config.toml'
    var_12 = 'key3'
    var_13 = 'value3'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)



# Parsed testcases at query #83
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/another/path/config2.json'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)



# Parsed testcases at query #84
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/home/user/project/.config1'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/home/user/.config2'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/home/.config3'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = var_0.search(var_10)
    var_17 = '/home/user/project/subdir/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/home/user/other/file.txt'
    var_20 = var_0.search(var_19)
    var_21 = '/home/another/file.txt'
    var_22 = var_0.search(var_21)
    var_23 = module_0.Trie()
    var_24 = '/some/random/file.txt'
    var_25 = var_23.search(var_24)
    var_26 = '/home/user/project/src/.config4'
    var_27 = 'key4'
    var_28 = 'value4'
    var_29 = {var_27: var_28}
    var_30 = var_0.insert(var_26, var_29)
    var_31 = '/home/user/project/src/subdir/file.txt'
    var_32 = var_0.search(var_31)
    var_33 = '/home/file.txt'
    var_34 = var_0.search(var_33)



# Parsed testcases at query #85
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #86
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = {}
    var_9 = module_0.Trie(var_2, var_8)
    var_10 = var_9.root



# Parsed testcases at query #87
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/root/config1.json'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = '/root/subdir/config2.json'
    var_6 = 'key2'
    var_7 = 'value2'
    var_8 = {var_6: var_7}
    var_9 = var_0.insert(var_1, var_4)
    var_10 = var_0.insert(var_5, var_8)
    var_11 = '/root/file.txt'
    var_12 = var_0.search(var_11)
    var_13 = '/root/subdir/file.txt'
    var_14 = var_0.search(var_13)
    var_15 = '/root/nonexistent/file.txt'
    var_16 = var_0.search(var_15)
    var_17 = '/root/subdir/deeper/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = module_0.Trie()
    var_20 = '/any/path/file.txt'
    var_21 = var_19.search(var_20)



# Parsed testcases at query #88
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #89
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/another/path/config2.json'
    var_8 = 'key2'
    var_9 = 'value2'
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = var_0.root



# Parsed testcases at query #90
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #91
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/another/path/config.yaml'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)



# Parsed testcases at query #92
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #93
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #94
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/root/config1.yaml'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/root/subdir/config2.yaml'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/root/subdir/subsubdir/config3.yaml'
    var_12 = 'key3'
    var_13 = 'value3'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = '/root/file.txt'
    var_17 = var_0.search(var_16)
    var_18 = '/root/subdir/file.txt'
    var_19 = var_0.search(var_18)
    var_20 = '/root/subdir/subsubdir/file.txt'
    var_21 = var_0.search(var_20)
    var_22 = '/root/nonexistent/file.txt'
    var_23 = var_0.search(var_22)
    var_24 = '/root/subdir/anotherdir/file.txt'
    var_25 = var_0.search(var_24)
    var_26 = module_0.Trie()
    var_27 = '/any/path/file.txt'
    var_28 = var_26.search(var_27)
    var_29 = module_0.Trie()
    var_30 = 'C:\\root\\config1.yaml'
    var_31 = 'key1'
    var_32 = 'value1'
    var_33 = {var_31: var_32}
    var_34 = var_29.insert(var_30, var_33)
    var_35 = 'C:\\root\\file.txt'
    var_36 = var_29.search(var_35)



# Parsed testcases at query #95
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #96
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/root/config1.json'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/root/subdir/config2.json'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/root/subdir/subsubdir/config3.json'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = var_0.search(var_10)
    var_17 = '/root/subdir/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/root/subdir/subsubdir/file.txt'
    var_20 = var_0.search(var_19)
    var_21 = '/root/default.json'
    var_22 = 'default'
    var_23 = 'config'
    var_24 = {var_22: var_23}
    var_25 = '/nonexistent/path/file.txt'
    var_26 = var_0.search(var_25)
    var_27 = module_0.Trie()
    var_28 = '/any/path/file.txt'
    var_29 = var_27.search(var_28)



# Parsed testcases at query #97
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #98
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/path/to/config.py'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = var_0.search(var_4)
    var_7 = '/path/to/subdir/file.py'
    var_8 = var_0.search(var_7)
    var_9 = '/different/path/file.py'
    var_10 = var_0.search(var_9)
    var_11 = module_0.Trie()
    var_12 = '/any/path/file.py'
    var_13 = var_11.search(var_12)
    var_14 = '/path/config.py'
    var_15 = 'key2'
    var_16 = 'value2'
    var_17 = {var_15: var_16}
    var_18 = var_0.insert(var_14, var_17)
    var_19 = var_0.search(var_7)
    var_20 = '/path/other/file.py'
    var_21 = var_0.search(var_20)
    var_22 = module_0.Trie()
    var_23 = 'C:\\path\\to\\config.py'
    var_24 = var_22.insert(var_23, var_3)
    var_25 = 'C:\\path\\to\\subdir\\file.py'
    var_26 = var_22.search(var_25)



# Parsed testcases at query #99
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #100
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = {}
    var_9 = module_0.Trie(var_5, var_8)
    var_10 = var_9.root
    var_11 = None
    var_12 = module_0.Trie(var_5, var_11)
    var_13 = var_12.root



# Parsed testcases at query #101
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #102
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config1.yaml'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/root/subdir/config2.yaml'
    var_7 = var_0.insert(var_6, var_3)
    var_8 = '/root/subdir/subsubdir/config3.yaml'
    var_9 = var_0.insert(var_8, var_3)
    var_10 = var_0.search(var_4)
    var_11 = '/root/subdir/file.txt'
    var_12 = var_0.search(var_11)
    var_13 = '/root/subdir/subsubdir/file.txt'
    var_14 = var_0.search(var_13)
    var_15 = '/root/otherdir/file.txt'
    var_16 = var_0.search(var_15)
    var_17 = module_0.Trie()
    var_18 = '/any/path/file.txt'
    var_19 = var_17.search(var_18)
    var_20 = '/a/b/c/config.yaml'
    var_21 = var_0.insert(var_20, var_3)
    var_22 = '/a/b/c/d/file.txt'
    var_23 = var_0.search(var_22)
    var_24 = '/CaseSensitive/Config.yaml'
    var_25 = var_0.insert(var_24, var_3)
    var_26 = '/CaseSensitive/file.txt'
    var_27 = var_0.search(var_26)



# Parsed testcases at query #103
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #104
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/.config'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/home/user/.config/file.txt'
    var_7 = var_0.search(var_6)
    var_8 = '/home/user/.config/subdir/file.txt'
    var_9 = var_0.search(var_8)
    var_10 = '/different/path/file.txt'
    var_11 = var_0.search(var_10)
    var_12 = module_0.Trie()
    var_13 = '/any/path/file.txt'
    var_14 = var_12.search(var_13)
    var_15 = '/home/user/.config/subdir'
    var_16 = 'new_value'
    var_17 = {var_1: var_16}
    var_18 = var_0.insert(var_15, var_17)
    var_19 = var_0.search(var_8)
    var_20 = 'root'
    var_21 = 'config'
    var_22 = {var_20: var_21}
    var_23 = ''
    var_24 = module_0.Trie(var_23, var_22)
    var_25 = var_24.search(var_13)



# Parsed testcases at query #105
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #106
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/root/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/root/subdir/config.json'
    var_7 = 'subvalue'
    var_8 = {var_2: var_7}
    var_9 = var_0.insert(var_6, var_8)
    var_10 = '/root/subdir/file.txt'
    var_11 = '/root/otherdir/file.txt'
    var_12 = module_0.Trie()
    var_13 = '/any/path'
    var_14 = '/a/b/c/config.json'
    var_15 = 'deep'
    var_16 = {var_2: var_15}
    var_17 = var_0.insert(var_14, var_16)
    var_18 = '/a/b/c/d/file.txt'



# Parsed testcases at query #107
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'to'
    var_13 = var_9.nodes[var_12]
    var_14 = var_13.nodes
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'config.json'
    var_17 = var_13.nodes[var_16]



# Parsed testcases at query #108
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/config.json'
    var_7 = var_0.search(var_6)
    var_8 = module_0.Trie()
    var_9 = '/home/user/config.json'
    var_10 = {var_2: var_3}
    var_11 = var_8.insert(var_9, var_10)
    var_12 = '/home/user/subdir/file.txt'
    var_13 = var_8.search(var_12)
    var_14 = module_0.Trie()
    var_15 = '/root_config.json'
    var_16 = 'root'
    var_17 = 'config'
    var_18 = {var_16: var_17}
    var_19 = var_14.insert(var_15, var_18)
    var_20 = '/nonexistent/path/file.txt'
    var_21 = var_14.search(var_20)
    var_22 = module_0.Trie()
    var_23 = '/home/user/config.json'
    var_24 = 'key1'
    var_25 = 'value1'
    var_26 = {var_24: var_25}
    var_27 = var_22.insert(var_23, var_26)
    var_28 = '/home/user/subdir/config.json'
    var_29 = 'key2'
    var_30 = 'value2'
    var_31 = {var_29: var_30}
    var_32 = var_22.insert(var_28, var_31)
    var_33 = var_22.search(var_12)
    var_34 = module_0.Trie()
    var_35 = '/any/path/file.txt'
    var_36 = var_34.search(var_35)



# Parsed testcases at query #109
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.yaml'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = ''
    var_9 = {}
    var_10 = module_0.Trie(var_8, var_9)
    var_11 = var_10.root



# Parsed testcases at query #110
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'config.json'
    var_9 = 'to'
    var_10 = 'path'
    var_11 = var_0.root.nodes[var_10]
    var_12 = var_11.nodes[var_9]
    var_13 = var_12.nodes[var_8]
    var_14 = '/path/to/another/config.json'
    var_15 = 'key2'
    var_16 = 'value2'
    var_17 = {var_15: var_16}
    var_18 = var_0.insert(var_14, var_17)
    var_19 = var_0.root.nodes[var_10]
    var_20 = var_19.nodes[var_9]
    var_21 = var_20.nodes
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = 'another'
    var_24 = var_0.root.nodes[var_10]
    var_25 = var_24.nodes[var_9]
    var_26 = var_25.nodes[var_23]
    var_27 = var_26.nodes[var_8]



# Parsed testcases at query #111
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = 'root_config.json'
    var_1 = 'key'
    var_2 = 'root_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Trie(var_0, var_3)
    var_5 = '/some/random/path/file.txt'
    var_6 = var_4.search(var_5)
    var_7 = module_0.Trie()
    var_8 = '/project/.config.json'
    var_9 = 'project_value'
    var_10 = {var_1: var_9}
    var_11 = var_7.insert(var_8, var_10)
    var_12 = '/project/src/.config.json'
    var_13 = 'src_value'
    var_14 = {var_1: var_13}
    var_15 = var_7.insert(var_12, var_14)
    var_16 = '/project/src/module/file.py'
    var_17 = var_7.search(var_16)
    var_18 = module_0.Trie()
    var_19 = {var_1: var_9}
    var_20 = var_18.insert(var_8, var_19)
    var_21 = {var_1: var_13}
    var_22 = var_18.insert(var_12, var_21)
    var_23 = '/project/src/subdir/file.py'
    var_24 = var_18.search(var_23)
    var_25 = module_0.Trie()
    var_26 = '/any/path/file.txt'
    var_27 = var_25.search(var_26)
    var_28 = module_0.Trie()
    var_29 = '/.config.json'
    var_30 = 'root_config'
    var_31 = {var_1: var_30}
    var_32 = var_28.insert(var_29, var_31)
    var_33 = '/file.txt'
    var_34 = var_28.search(var_33)



# Parsed testcases at query #112
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = {}
    var_9 = module_0.Trie(var_5, var_8)
    var_10 = var_9.root
    var_11 = None
    var_12 = module_0.Trie(var_5, var_11)
    var_13 = var_12.root



# Parsed testcases at query #113
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #114
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #115
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #116
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/another/path/to/config2.json'
    var_8 = 'key2'
    var_9 = 'value2'
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = var_0.root



# Parsed testcases at query #117
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #118
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #119
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #120
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = var_0.root



# Parsed testcases at query #121
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/path/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = '/root_config.json'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/config1.json'
    var_11 = 'a'
    var_12 = 1
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/parent/config2.json'
    var_16 = 'b'
    var_17 = 2
    var_18 = {var_16: var_17}
    var_19 = var_9.insert(var_15, var_18)
    var_20 = '/parent/child/config3.json'
    var_21 = 'c'
    var_22 = 3
    var_23 = {var_21: var_22}
    var_24 = var_9.insert(var_20, var_23)
    var_25 = '/parent/child/file.txt'
    var_26 = var_9.search(var_25)
    var_27 = '/parent/other/file.txt'
    var_28 = var_9.search(var_27)
    var_29 = '/other/file.txt'
    var_30 = var_9.search(var_29)
    var_31 = module_0.Trie()
    var_32 = '/existing/config.json'
    var_33 = 'x'
    var_34 = {var_33: var_12}
    var_35 = var_31.insert(var_32, var_34)
    var_36 = '/non/existing/path/file.txt'
    var_37 = var_31.search(var_36)
    var_38 = module_0.Trie()
    var_39 = '/exact/config.json'
    var_40 = 'exact'
    var_41 = True
    var_42 = {var_40: var_41}
    var_43 = var_38.insert(var_39, var_42)
    var_44 = var_38.search(var_39)
    var_45 = module_0.Trie()
    var_46 = '/Case/Sensitive/Config.json'
    var_47 = 'case'
    var_48 = 'sensitive'
    var_49 = {var_47: var_48}
    var_50 = var_45.insert(var_46, var_49)
    var_51 = '/case/sensitive/file.txt'
    var_52 = var_45.search(var_51)



# Parsed testcases at query #122
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #123
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'some/file/path'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = '/path/to/config.json'
    var_8 = var_3.insert(var_7, var_6)
    var_9 = var_3.search(var_7)
    var_10 = module_0.Trie()
    var_11 = 'parent_key'
    var_12 = 'parent_value'
    var_13 = {var_11: var_12}
    var_14 = 'child_key'
    var_15 = 'child_value'
    var_16 = {var_14: var_15}
    var_17 = '/path/to'
    var_18 = var_10.insert(var_17, var_13)
    var_19 = '/path/to/subdir'
    var_20 = var_10.insert(var_19, var_16)
    var_21 = '/path/to/subdir/file.txt'
    var_22 = var_10.search(var_21)
    var_23 = module_0.Trie()
    var_24 = 'root_key'
    var_25 = 'root_value'
    var_26 = {var_24: var_25}
    var_27 = '/'
    var_28 = var_23.insert(var_27, var_26)
    var_29 = '/some/other/path/file.txt'
    var_30 = var_23.search(var_29)
    var_31 = module_0.Trie()
    var_32 = 'level1'
    var_33 = 'value1'
    var_34 = {var_32: var_33}
    var_35 = 'level2'
    var_36 = 'value2'
    var_37 = {var_35: var_36}
    var_38 = 'level3'
    var_39 = 'value3'
    var_40 = {var_38: var_39}
    var_41 = '/level1'
    var_42 = var_31.insert(var_41, var_34)
    var_43 = '/level1/level2'
    var_44 = var_31.insert(var_43, var_37)
    var_45 = '/level1/level2/level3'
    var_46 = var_31.insert(var_45, var_40)
    var_47 = '/level1/level2/level3/file.txt'
    var_48 = var_31.search(var_47)
    var_49 = module_0.Trie()
    var_50 = 'partial'
    var_51 = 'data'
    var_52 = {var_50: var_51}
    var_53 = '/partial/path'
    var_54 = var_49.insert(var_53, var_52)
    var_55 = '/partial/path/extra/file.txt'
    var_56 = var_49.search(var_55)
    var_57 = module_0.Trie()
    var_58 = '/some/other/path'
    var_59 = {var_4: var_5}
    var_60 = var_57.insert(var_58, var_59)
    var_61 = '/completely/different/path/file.txt'
    var_62 = var_57.search(var_61)
    var_63 = module_0.Trie()
    var_64 = 'root'
    var_65 = 'config'
    var_66 = {var_64: var_65}
    var_67 = ''
    var_68 = var_63.insert(var_67, var_66)
    var_69 = 'any/path/file.txt'
    var_70 = var_63.search(var_69)



# Parsed testcases at query #124
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #125
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/root/config1.json'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/root/subdir/config2.json'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/root/subdir/subsubdir/config3.json'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = '/root/file.txt'
    var_17 = var_0.search(var_16)
    var_18 = '/root/subdir/file.txt'
    var_19 = var_0.search(var_18)
    var_20 = '/root/subdir/subsubdir/file.txt'
    var_21 = var_0.search(var_20)
    var_22 = '/root/subdir/nonexistent/file.txt'
    var_23 = var_0.search(var_22)
    var_24 = '/nonexistent/file.txt'
    var_25 = var_0.search(var_24)
    var_26 = module_0.Trie()
    var_27 = '/any/path/file.txt'
    var_28 = var_26.search(var_27)



# Parsed testcases at query #126
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #127
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #128
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #129
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #130
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #131
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/root/config1.json'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/root/subdir/config2.json'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/root/subdir/subsubdir/config3.json'
    var_12 = 'key3'
    var_13 = 'value3'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = '/root/file.txt'
    var_17 = '/root/subdir/file.txt'
    var_18 = '/root/subdir/subsubdir/file.txt'
    var_19 = '/root/subdir/nonexistent/file.txt'
    var_20 = '/nonexistent/file.txt'
    var_21 = module_0.Trie()
    var_22 = '/any/path/file.txt'



# Parsed testcases at query #132
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/some/path'
    var_5 = var_0.search(var_4)
    var_6 = '/root/config.py'
    var_7 = var_0.insert(var_6, var_3)
    var_8 = '/root/file.py'
    var_9 = var_0.search(var_8)
    var_10 = '/root/subdir/config.py'
    var_11 = 'subvalue'
    var_12 = {var_1: var_11}
    var_13 = var_0.insert(var_10, var_12)
    var_14 = '/root/subdir/file.py'
    var_15 = var_0.search(var_14)
    var_16 = '/nonexistent/path'
    var_17 = var_0.search(var_16)
    var_18 = '/root/subdir/nested/file.py'
    var_19 = var_0.search(var_18)
    var_20 = '/root/subdir/nested/config.py'
    var_21 = 'nestedvalue'
    var_22 = {var_1: var_21}
    var_23 = var_0.insert(var_20, var_22)
    var_24 = var_0.search(var_18)



# Parsed testcases at query #133
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/path'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = '/some/config.py'
    var_8 = var_3.insert(var_7, var_6)
    var_9 = var_3.search(var_7)
    var_10 = module_0.Trie()
    var_11 = 'parent_key'
    var_12 = 'parent_value'
    var_13 = {var_11: var_12}
    var_14 = '/some/'
    var_15 = var_10.insert(var_14, var_13)
    var_16 = 'child_key'
    var_17 = 'child_value'
    var_18 = {var_16: var_17}
    var_19 = '/some/child/'
    var_20 = var_10.insert(var_19, var_18)
    var_21 = '/some/child/grandchild/file.py'
    var_22 = var_10.search(var_21)
    var_23 = module_0.Trie()
    var_24 = 'root_key'
    var_25 = 'root_value'
    var_26 = {var_24: var_25}
    var_27 = '/'
    var_28 = var_23.insert(var_27, var_26)
    var_29 = '/nonexistent/path/file.py'
    var_30 = var_23.search(var_29)
    var_31 = module_0.Trie()
    var_32 = 'level1'
    var_33 = 'data1'
    var_34 = {var_32: var_33}
    var_35 = '/level1/'
    var_36 = var_31.insert(var_35, var_34)
    var_37 = 'level2'
    var_38 = 'data2'
    var_39 = {var_37: var_38}
    var_40 = '/level1/level2/'
    var_41 = var_31.insert(var_40, var_39)
    var_42 = '/level1/level2/level3/file.py'
    var_43 = var_31.search(var_42)
    var_44 = '/level1/other/file.py'
    var_45 = var_31.search(var_44)



# Parsed testcases at query #134
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #135
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/etc/config.yaml'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = var_0.search(var_4)
    var_7 = '/etc/subdir/file.txt'
    var_8 = var_0.search(var_7)
    var_9 = module_0.Trie()
    var_10 = '/nonexistent/path/file.txt'
    var_11 = var_9.search(var_10)
    var_12 = '/etc/subdir/config.yaml'
    var_13 = 'value2'
    var_14 = {var_1: var_13}
    var_15 = var_0.insert(var_12, var_14)
    var_16 = '/etc/subdir/deep/file.txt'
    var_17 = var_0.search(var_16)
    var_18 = 'root'
    var_19 = 'config'
    var_20 = {var_18: var_19}
    var_21 = ''
    var_22 = module_0.Trie(var_21, var_20)
    var_23 = '/any/path/file.txt'
    var_24 = var_22.search(var_23)



# Parsed testcases at query #136
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #137
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #138
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #139
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #140
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #141
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/root/config1.json'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/root/subdir/config2.json'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/root/subdir/subsubdir/config3.json'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = var_0.search(var_10)
    var_17 = var_0.search(var_12)
    var_18 = var_0.search(var_14)
    var_19 = '/root/subdir/file.txt'
    var_20 = var_0.search(var_19)
    var_21 = '/root/subdir/subsubdir/file.txt'
    var_22 = var_0.search(var_21)
    var_23 = '/root/otherdir/file.txt'
    var_24 = var_0.search(var_23)
    var_25 = '/root/nonexistent/file.txt'
    var_26 = var_0.search(var_25)
    var_27 = module_0.Trie()
    var_28 = '/any/path'
    var_29 = var_27.search(var_28)
    var_30 = '/root'
    var_31 = var_0.search(var_30)
    var_32 = '/root/subdir'
    var_33 = var_0.search(var_32)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #2
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #3
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'config.json'
    var_9 = 'to'
    var_10 = 'path'
    var_11 = var_0.root.nodes[var_10]
    var_12 = var_11.nodes[var_9]
    var_13 = var_12.nodes[var_8]
    var_14 = module_0.Trie()
    var_15 = '/path/to/config1.json'
    var_16 = 'key1'
    var_17 = 'value1'
    var_18 = {var_16: var_17}
    var_19 = '/path/to/config2.json'
    var_20 = 'key2'
    var_21 = 'value2'
    var_22 = {var_20: var_21}
    var_23 = var_14.insert(var_15, var_18)
    var_24 = var_14.insert(var_19, var_22)
    var_25 = 'config1.json'
    var_26 = var_14.root.nodes[var_10]
    var_27 = var_26.nodes[var_9]
    var_28 = var_27.nodes[var_25]
    var_29 = 'config2.json'
    var_30 = var_14.root.nodes[var_10]
    var_31 = var_30.nodes[var_9]
    var_32 = var_31.nodes[var_29]
    var_33 = module_0.Trie()
    var_34 = '/path/to/config.json'
    var_35 = {var_16: var_17}
    var_36 = '/path/to/subdir/config.json'
    var_37 = {var_20: var_21}
    var_38 = var_33.insert(var_34, var_35)
    var_39 = var_33.insert(var_36, var_37)
    var_40 = var_33.root.nodes[var_10]
    var_41 = var_40.nodes[var_9]
    var_42 = var_41.nodes[var_8]
    var_43 = 'subdir'
    var_44 = var_33.root.nodes[var_10]
    var_45 = var_44.nodes[var_9]
    var_46 = var_45.nodes[var_43]
    var_47 = var_46.nodes[var_8]
    var_48 = module_0.Trie()
    var_49 = '/path/to/config.json'
    var_50 = {var_16: var_17}
    var_51 = {var_20: var_21}
    var_52 = var_48.insert(var_49, var_50)
    var_53 = var_48.insert(var_49, var_51)
    var_54 = var_48.root.nodes[var_10]
    var_55 = var_54.nodes[var_9]
    var_56 = var_55.nodes[var_8]



# Parsed testcases at query #4
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #5
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #6
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #7
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #8
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = module_0.Trie()
    var_8 = '/path/to/config1.json'
    var_9 = 'key1'
    var_10 = 'value1'
    var_11 = {var_9: var_10}
    var_12 = var_7.insert(var_8, var_11)
    var_13 = '/path/to/subdir/config2.json'
    var_14 = 'key2'
    var_15 = 'value2'
    var_16 = {var_14: var_15}
    var_17 = var_7.insert(var_13, var_16)
    var_18 = var_7.root
    var_19 = var_7.root
    var_20 = module_0.Trie()
    var_21 = '/path/to/config3.json'
    var_22 = 'key3'
    var_23 = 'value3'
    var_24 = {var_22: var_23}
    var_25 = var_20.insert(var_21, var_24)
    var_26 = '/path/to/config4.json'
    var_27 = 'key4'
    var_28 = 'value4'
    var_29 = {var_27: var_28}
    var_30 = var_20.insert(var_26, var_29)
    var_31 = var_20.root
    var_32 = var_20.root



# Parsed testcases at query #9
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/project/.config'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/home/user/.config'
    var_7 = 'root_value'
    var_8 = {var_1: var_7}
    var_9 = var_0.insert(var_6, var_8)
    var_10 = '/home/user/project/subdir/.config'
    var_11 = 'subdir_value'
    var_12 = {var_1: var_11}
    var_13 = var_0.insert(var_10, var_12)
    var_14 = var_0.search(var_4)
    var_15 = '/home/user/project/subdir/file.txt'
    var_16 = var_0.search(var_15)
    var_17 = '/home/user/other_project/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = module_0.Trie()
    var_20 = '/some/random/path'
    var_21 = var_19.search(var_20)
    var_22 = '/home/user/project'
    var_23 = var_0.search(var_22)
    var_24 = '/home/user/Project/.config'
    var_25 = 'case_value'
    var_26 = {var_1: var_25}
    var_27 = var_0.insert(var_24, var_26)
    var_28 = '/home/user/Project/file.txt'
    var_29 = var_0.search(var_28)



# Parsed testcases at query #10
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #11
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config1.json'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = '/home/user/config2.json'
    var_6 = 'key2'
    var_7 = 'value2'
    var_8 = {var_6: var_7}
    var_9 = '/home/config3.json'
    var_10 = 'key3'
    var_11 = 'value3'
    var_12 = {var_10: var_11}
    var_13 = var_0.insert(var_1, var_4)
    var_14 = var_0.insert(var_5, var_8)
    var_15 = var_0.insert(var_9, var_12)
    var_16 = '/home/user/project/test.py'
    var_17 = var_0.search(var_16)
    var_18 = '/home/user/subdir/test.py'
    var_19 = var_0.search(var_18)
    var_20 = '/home/other/test.py'
    var_21 = var_0.search(var_20)
    var_22 = '/nonexistent/path/test.py'
    var_23 = var_0.search(var_22)
    var_24 = module_0.Trie()
    var_25 = '/any/path/test.py'
    var_26 = var_24.search(var_25)



# Parsed testcases at query #12
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #13
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = module_0.Trie()
    var_7 = '/another/path/config.json'
    var_8 = {}
    var_9 = var_6.insert(var_7, var_8)
    var_10 = module_0.Trie()
    var_11 = '/deep/nested/path/to/config.json'
    var_12 = 'nested'
    var_13 = True
    var_14 = {var_12: var_13}
    var_15 = var_10.insert(var_11, var_14)
    var_16 = module_0.Trie()
    var_17 = '/test/config.json'
    var_18 = 'initial'
    var_19 = {var_18: var_13}
    var_20 = var_16.insert(var_17, var_19)
    var_21 = 'updated'
    var_22 = {var_21: var_13}
    var_23 = var_16.insert(var_17, var_22)



# Parsed testcases at query #14
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #15
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #16
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/path/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = '/some/path/config.json'
    var_8 = var_3.insert(var_7, var_6)
    var_9 = var_3.search(var_7)
    var_10 = module_0.Trie()
    var_11 = 'parent_key'
    var_12 = 'parent_value'
    var_13 = {var_11: var_12}
    var_14 = 'child_key'
    var_15 = 'child_value'
    var_16 = {var_14: var_15}
    var_17 = '/some/config.json'
    var_18 = var_10.insert(var_17, var_13)
    var_19 = var_10.insert(var_7, var_16)
    var_20 = '/some/path/subpath/file.txt'
    var_21 = var_10.search(var_20)
    var_22 = '/some/other/file.txt'
    var_23 = var_10.search(var_22)
    var_24 = module_0.Trie()
    var_25 = 'root_key'
    var_26 = 'root_value'
    var_27 = {var_25: var_26}
    var_28 = '/config.json'
    var_29 = var_24.insert(var_28, var_27)
    var_30 = '/any/path/file.txt'
    var_31 = var_24.search(var_30)
    var_32 = ''
    var_33 = {var_25: var_26}
    var_34 = module_0.Trie(var_32, var_33)
    var_35 = '/nonexistent/path/file.txt'
    var_36 = var_34.search(var_35)
    var_37 = module_0.Trie()
    var_38 = '/Some/Path/config.json'
    var_39 = 'case_key'
    var_40 = 'case_value'
    var_41 = {var_39: var_40}
    var_42 = var_37.insert(var_38, var_41)
    var_43 = var_37.search(var_1)
    var_44 = '/Some/Path/file.txt'
    var_45 = var_37.search(var_44)



# Parsed testcases at query #17
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #18
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #19
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #20
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #21
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #22
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    var_8 = var_0.root
    var_9 = module_0.Trie()
    var_10 = '/another/path/config.json'
    var_11 = {}
    var_12 = var_9.insert(var_10, var_11)
    var_13 = var_9.root
    var_14 = module_0.Trie()
    var_15 = '/a/b/c/config1.json'
    var_16 = 'a'
    var_17 = 1
    var_18 = {var_16: var_17}
    var_19 = var_14.insert(var_15, var_18)
    var_20 = '/a/b/d/config2.json'
    var_21 = 'b'
    var_22 = 2
    var_23 = {var_21: var_22}
    var_24 = var_14.insert(var_20, var_23)
    var_25 = '/a/e/config3.json'
    var_26 = 'c'
    var_27 = 3
    var_28 = {var_26: var_27}
    var_29 = var_14.insert(var_25, var_28)
    var_30 = var_14.root.nodes[var_16]
    var_31 = var_30.nodes[var_21]
    var_32 = var_31.nodes[var_26]
    var_33 = 'd'
    var_34 = var_31.nodes[var_33]
    var_35 = 'e'
    var_36 = var_30.nodes[var_35]



# Parsed testcases at query #23
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #24
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #25
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #26
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = 'project'
    var_7 = 'user'
    var_8 = 'home'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes[var_7]
    var_11 = var_10.nodes[var_6]
    var_12 = '/home/another/config.json'
    var_13 = 'another_key'
    var_14 = 'another_value'
    var_15 = {var_13: var_14}
    var_16 = var_0.insert(var_12, var_15)
    var_17 = 'another'
    var_18 = var_0.root.nodes[var_8]
    var_19 = var_18.nodes[var_17]
    var_20 = '/home/user/project/src/config.json'
    var_21 = 'src_key'
    var_22 = 'src_value'
    var_23 = {var_21: var_22}
    var_24 = var_0.insert(var_20, var_23)
    var_25 = 'src'
    var_26 = var_0.root.nodes[var_8]
    var_27 = var_26.nodes[var_7]
    var_28 = var_27.nodes[var_6]
    var_29 = var_28.nodes[var_25]
    var_30 = '/home/user/project/config.json'
    var_31 = 'new_key'
    var_32 = 'new_value'
    var_33 = {var_31: var_32}
    var_34 = var_0.insert(var_30, var_33)
    var_35 = var_0.root.nodes[var_8]
    var_36 = var_35.nodes[var_7]
    var_37 = var_36.nodes[var_6]



# Parsed testcases at query #27
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #28
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)



# Parsed testcases at query #29
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'config.json'
    var_9 = 'to'
    var_10 = 'path'
    var_11 = var_0.root.nodes[var_10]
    var_12 = var_11.nodes[var_9]
    var_13 = var_12.nodes[var_8]
    var_14 = module_0.Trie()
    var_15 = '/another/path/config.json'
    var_16 = {}
    var_17 = var_14.insert(var_15, var_16)
    var_18 = 'another'
    var_19 = var_14.root.nodes[var_18]
    var_20 = var_19.nodes[var_10]
    var_21 = var_20.nodes[var_8]
    var_22 = module_0.Trie()
    var_23 = '/common/path/config1.json'
    var_24 = 'key1'
    var_25 = 'value1'
    var_26 = {var_24: var_25}
    var_27 = var_22.insert(var_23, var_26)
    var_28 = '/common/path/subdir/config2.json'
    var_29 = 'key2'
    var_30 = 'value2'
    var_31 = {var_29: var_30}
    var_32 = var_22.insert(var_28, var_31)
    var_33 = 'config1.json'
    var_34 = 'common'
    var_35 = var_22.root.nodes[var_34]
    var_36 = var_35.nodes[var_10]
    var_37 = var_36.nodes[var_33]
    var_38 = 'config2.json'
    var_39 = 'subdir'
    var_40 = var_22.root.nodes[var_34]
    var_41 = var_40.nodes[var_10]
    var_42 = var_41.nodes[var_39]
    var_43 = var_42.nodes[var_38]



# Parsed testcases at query #30
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'to'
    var_13 = var_9.nodes[var_12]
    var_14 = var_13.nodes
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'config.json'
    var_17 = var_13.nodes[var_16]



# Parsed testcases at query #31
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #32
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #33
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #34
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/root/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/root/subdir/config.json'
    var_7 = 'subvalue'
    var_8 = {var_2: var_7}
    var_9 = var_0.insert(var_6, var_8)
    var_10 = var_0.search(var_1)
    var_11 = '/root/subdir/file.txt'
    var_12 = var_0.search(var_11)
    var_13 = '/root/other/subdir/file.txt'
    var_14 = var_0.search(var_13)
    var_15 = module_0.Trie()
    var_16 = '/any/path/file.txt'
    var_17 = var_15.search(var_16)
    var_18 = '/a/b/c/config.json'
    var_19 = 'deep'
    var_20 = 'config'
    var_21 = {var_19: var_20}
    var_22 = var_0.insert(var_18, var_21)
    var_23 = '/a/b/c/d/file.txt'
    var_24 = var_0.search(var_23)
    var_25 = '/Case/config.json'
    var_26 = 'case'
    var_27 = 'sensitive'
    var_28 = {var_26: var_27}
    var_29 = var_0.insert(var_25, var_28)
    var_30 = '/case/file.txt'
    var_31 = var_0.search(var_30)



# Parsed testcases at query #35
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test.json'
    var_3 = module_0.Trie(var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(config_data=var_6)
    var_8 = module_0.Trie(var_2, var_6)



# Parsed testcases at query #36
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/path'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'root_config.json'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = '/some/file.txt'
    var_9 = var_7.search(var_8)
    var_10 = module_0.Trie()
    var_11 = '/root/config.json'
    var_12 = 'root'
    var_13 = True
    var_14 = {var_12: var_13}
    var_15 = var_10.insert(var_11, var_14)
    var_16 = '/root/subdir/config.json'
    var_17 = 'subdir'
    var_18 = {var_17: var_13}
    var_19 = var_10.insert(var_16, var_18)
    var_20 = '/root/subdir/nested/config.json'
    var_21 = 'nested'
    var_22 = {var_21: var_13}
    var_23 = var_10.insert(var_20, var_22)
    var_24 = '/root/subdir/nested/file.txt'
    var_25 = var_10.search(var_24)
    var_26 = '/root/subdir/file.txt'
    var_27 = var_10.search(var_26)
    var_28 = '/root/file.txt'
    var_29 = var_10.search(var_28)
    var_30 = '/completely/different/path/file.txt'
    var_31 = var_10.search(var_30)
    var_32 = '/exact/config.json'
    var_33 = 'exact'
    var_34 = {var_33: var_13}
    var_35 = var_10.insert(var_32, var_34)
    var_36 = var_10.search(var_32)
    var_37 = '/Case/Sensitive/Config.json'
    var_38 = 'case'
    var_39 = 'sensitive'
    var_40 = {var_38: var_39}
    var_41 = var_10.insert(var_37, var_40)
    var_42 = '/case/sensitive/file.txt'
    var_43 = var_10.search(var_42)
    var_44 = '/Case/Sensitive/file.txt'
    var_45 = var_10.search(var_44)



# Parsed testcases at query #37
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/another/path/config.yaml'
    var_8 = 'another_key'
    var_9 = 'another_value'
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = var_0.root



# Parsed testcases at query #38
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #39
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #40
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #41
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #42
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/path'
    var_2 = var_0.search(var_1)
    var_3 = 'root_config.json'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/root/config.json'
    var_11 = 'root'
    var_12 = True
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/root/subdir/config.json'
    var_16 = 'subdir'
    var_17 = {var_16: var_12}
    var_18 = var_9.insert(var_15, var_17)
    var_19 = '/root/subdir/nested/config.json'
    var_20 = 'nested'
    var_21 = {var_20: var_12}
    var_22 = var_9.insert(var_19, var_21)
    var_23 = '/root/file.txt'
    var_24 = var_9.search(var_23)
    var_25 = '/root/subdir/file.txt'
    var_26 = var_9.search(var_25)
    var_27 = '/root/subdir/nested/file.txt'
    var_28 = var_9.search(var_27)
    var_29 = '/root/subdir/other/file.txt'
    var_30 = var_9.search(var_29)
    var_31 = '/other/path/file.txt'
    var_32 = var_9.search(var_31)
    var_33 = module_0.Trie()
    var_34 = '/Root/Config.json'
    var_35 = 'case'
    var_36 = 'sensitive'
    var_37 = {var_35: var_36}
    var_38 = var_33.insert(var_34, var_37)
    var_39 = '/root/config/file.txt'
    var_40 = var_33.search(var_39)



# Parsed testcases at query #43
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #44
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = var_0.root
    var_8 = '/another/path/config.json'
    var_9 = 'another_key'
    var_10 = 'another_value'
    var_11 = {var_9: var_10}
    var_12 = var_0.insert(var_8, var_11)
    var_13 = var_0.root



# Parsed testcases at query #45
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #46
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #47
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #48
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #49
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #50
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #51
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root



# Parsed testcases at query #52
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/path'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = '/home/user/project/.config'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_3.insert(var_4, var_7)
    var_9 = var_3.search(var_4)
    var_10 = module_0.Trie()
    var_11 = '/home/user/.config'
    var_12 = 'parent_key'
    var_13 = 'parent_value'
    var_14 = {var_12: var_13}
    var_15 = var_10.insert(var_11, var_14)
    var_16 = '/home/user/project/file.txt'
    var_17 = var_10.search(var_16)
    var_18 = module_0.Trie()
    var_19 = '/.config'
    var_20 = 'root_key'
    var_21 = 'root_value'
    var_22 = {var_20: var_21}
    var_23 = var_18.insert(var_19, var_22)
    var_24 = '/home/user/.config'
    var_25 = 'mid_key'
    var_26 = 'mid_value'
    var_27 = {var_25: var_26}
    var_28 = var_18.insert(var_24, var_27)
    var_29 = var_18.search(var_16)
    var_30 = module_0.Trie()
    var_31 = '/.config'
    var_32 = {var_20: var_21}
    var_33 = var_30.insert(var_31, var_32)
    var_34 = '/nonexistent/path/file.txt'
    var_35 = var_30.search(var_34)
    var_36 = module_0.Trie()
    var_37 = '/home/User/.config'
    var_38 = {var_5: var_6}
    var_39 = var_36.insert(var_37, var_38)
    var_40 = '/home/user/file.txt'
    var_41 = var_36.search(var_40)



# Parsed testcases at query #53
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test_config.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #54
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #55
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #56
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #57
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'any/path'
    var_5 = var_0.search(var_4)
    var_6 = '/home/user/config.json'
    var_7 = var_0.insert(var_6, var_3)
    var_8 = var_0.search(var_6)
    var_9 = '/home/user/subdir/file.txt'
    var_10 = var_0.search(var_9)
    var_11 = '/home/config.json'
    var_12 = 'root'
    var_13 = {var_1: var_12}
    var_14 = var_0.insert(var_11, var_13)
    var_15 = '/home/user/project/config.json'
    var_16 = 'project'
    var_17 = {var_1: var_16}
    var_18 = var_0.insert(var_15, var_17)
    var_19 = '/home/user/project/src/file.py'
    var_20 = var_0.search(var_19)
    var_21 = '/file.txt'
    var_22 = var_0.search(var_21)
    var_23 = '/etc/config.json'
    var_24 = 'system'
    var_25 = {var_1: var_24}
    var_26 = var_0.insert(var_23, var_25)
    var_27 = '/etc/nonexistent/file.txt'
    var_28 = var_0.search(var_27)
    var_29 = 'C:\\Users\\config.json'
    var_30 = 'windows'
    var_31 = {var_1: var_30}
    var_32 = var_0.insert(var_29, var_31)
    var_33 = 'C:\\Users\\Documents\\file.txt'
    var_34 = var_0.search(var_33)



# Parsed testcases at query #58
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #59
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #60
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'some/file.txt'
    var_2 = '/root_config.json'
    var_3 = 'root'
    var_4 = 'config'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = '/some/file.txt'
    var_8 = module_0.Trie()
    var_9 = '/parent/config.json'
    var_10 = 'parent'
    var_11 = {var_10: var_4}
    var_12 = var_8.insert(var_9, var_11)
    var_13 = '/parent/child/file.txt'
    var_14 = module_0.Trie()
    var_15 = '/parent/child/config.json'
    var_16 = 'child'
    var_17 = {var_16: var_4}
    var_18 = var_14.insert(var_15, var_17)
    var_19 = module_0.Trie()
    var_20 = {var_3: var_4}
    var_21 = var_19.insert(var_2, var_20)
    var_22 = {var_10: var_4}
    var_23 = var_19.insert(var_9, var_22)
    var_24 = {var_16: var_4}
    var_25 = var_19.insert(var_15, var_24)
    var_26 = '/parent/child/grandchild/file.txt'
    var_27 = module_0.Trie()
    var_28 = {var_3: var_4}
    var_29 = var_27.insert(var_2, var_28)
    var_30 = {var_10: var_4}
    var_31 = var_27.insert(var_9, var_30)
    var_32 = '/unrelated/path/file.txt'
    var_33 = module_0.Trie()
    var_34 = '/Parent/Config.json'
    var_35 = {var_10: var_4}
    var_36 = var_33.insert(var_34, var_35)
    var_37 = '/parent/config/file.txt'



# Parsed testcases at query #61
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = module_0.Trie()
    var_7 = '/another/path/config.json'
    var_8 = {}
    var_9 = var_6.insert(var_7, var_8)
    var_10 = module_0.Trie()
    var_11 = '/existing/config.json'
    var_12 = 'existing'
    var_13 = True
    var_14 = {var_12: var_13}
    var_15 = var_10.insert(var_11, var_14)
    var_16 = '/existing/another/config.json'
    var_17 = 'another'
    var_18 = {var_17: var_13}
    var_19 = var_10.insert(var_16, var_18)
    var_20 = module_0.Trie()
    var_21 = 'relative/config.json'
    var_22 = 'relative'
    var_23 = {var_22: var_13}
    var_24 = var_20.insert(var_21, var_23)
    var_25 = var_20.root
    var_26 = {var_22: var_13}



# Parsed testcases at query #62
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/project/.config'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = var_0.search(var_4)
    var_7 = '/home/user/project/src/main.py'
    var_8 = var_0.search(var_7)
    var_9 = '/home/user/project'
    var_10 = var_0.search(var_9)
    var_11 = '/root/config'
    var_12 = 'root'
    var_13 = 'config'
    var_14 = {var_12: var_13}
    var_15 = '/nonexistent/path'
    var_16 = var_0.search(var_15)
    var_17 = '/home/user/.config'
    var_18 = 'user'
    var_19 = {var_18: var_13}
    var_20 = var_0.insert(var_17, var_19)
    var_21 = var_0.search(var_7)
    var_22 = module_0.Trie()
    var_23 = '/any/path'
    var_24 = var_22.search(var_23)
    var_25 = '/home/user/Project/.config'
    var_26 = 'case'
    var_27 = 'sensitive'
    var_28 = {var_26: var_27}
    var_29 = var_0.insert(var_25, var_28)
    var_30 = '/home/user/project/src/main.py'
    var_31 = var_0.search(var_30)



# Parsed testcases at query #63
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/another/path/config.json'
    var_8 = 'key2'
    var_9 = 'value2'
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = var_0.root



# Parsed testcases at query #64
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/path'
    var_2 = var_0.search(var_1)
    var_3 = '/root_config.json'
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = '/some/file.txt'
    var_9 = var_7.search(var_8)
    var_10 = module_0.Trie()
    var_11 = '/config.json'
    var_12 = {var_4: var_5}
    var_13 = var_10.insert(var_11, var_12)
    var_14 = '/project/config.json'
    var_15 = 'project_value'
    var_16 = {var_4: var_15}
    var_17 = var_10.insert(var_14, var_16)
    var_18 = '/project/src/config.json'
    var_19 = 'src_value'
    var_20 = {var_4: var_19}
    var_21 = var_10.insert(var_18, var_20)
    var_22 = '/project/src/main.py'
    var_23 = var_10.search(var_22)
    var_24 = '/project/main.py'
    var_25 = var_10.search(var_24)
    var_26 = '/other/file.txt'
    var_27 = var_10.search(var_26)
    var_28 = module_0.Trie()
    var_29 = '/a/b/config.json'
    var_30 = 'value'
    var_31 = {var_4: var_30}
    var_32 = var_28.insert(var_29, var_31)
    var_33 = '/a/b/c/file.txt'
    var_34 = var_28.search(var_33)
    var_35 = '/a/file.txt'
    var_36 = var_28.search(var_35)
    var_37 = module_0.Trie()
    var_38 = '/Case/config.json'
    var_39 = 'case_value'
    var_40 = {var_4: var_39}
    var_41 = var_37.insert(var_38, var_40)
    var_42 = '/case/file.txt'
    var_43 = var_37.search(var_42)
    var_44 = '/Case/file.txt'
    var_45 = var_37.search(var_44)
    var_46 = module_0.Trie()
    var_47 = 'C:\\project\\config.json'
    var_48 = 'key'
    var_49 = 'win_value'
    var_50 = {var_48: var_49}
    var_51 = var_46.insert(var_47, var_50)
    var_52 = 'C:\\project\\src\\main.py'
    var_53 = var_46.search(var_52)



# Parsed testcases at query #65
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)



# Parsed testcases at query #66
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config1.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/root/subdir/config2.json'
    var_7 = 'value2'
    var_8 = {var_1: var_7}
    var_9 = var_0.insert(var_6, var_8)
    var_10 = '/root/subdir/subsubdir/config3.json'
    var_11 = 'value3'
    var_12 = {var_1: var_11}
    var_13 = var_0.insert(var_10, var_12)
    var_14 = var_0.search(var_10)
    var_15 = '/root/subdir/subsubdir/other_file.txt'
    var_16 = var_0.search(var_15)
    var_17 = '/root/other_dir/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = module_0.Trie()
    var_20 = '/some/path/file.txt'
    var_21 = var_19.search(var_20)
    var_22 = module_0.Trie()
    var_23 = 'C:\\root\\config.json'
    var_24 = 'os'
    var_25 = 'windows'
    var_26 = {var_24: var_25}
    var_27 = var_22.insert(var_23, var_26)
    var_28 = 'C:\\root\\subdir\\file.txt'
    var_29 = var_22.search(var_28)



# Parsed testcases at query #67
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #68
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/path'
    var_2 = var_0.search(var_1)
    var_3 = '/root_config.json'
    var_4 = 'root'
    var_5 = 'config'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = '/some/file.txt'
    var_9 = var_7.search(var_8)
    var_10 = module_0.Trie()
    var_11 = '/src/config.json'
    var_12 = 'src'
    var_13 = {var_12: var_5}
    var_14 = var_10.insert(var_11, var_13)
    var_15 = '/src/main.py'
    var_16 = var_10.search(var_15)
    var_17 = '/src/sub/file.py'
    var_18 = var_10.search(var_17)
    var_19 = module_0.Trie()
    var_20 = '/config.json'
    var_21 = {var_4: var_5}
    var_22 = var_19.insert(var_20, var_21)
    var_23 = {var_12: var_5}
    var_24 = var_19.insert(var_11, var_23)
    var_25 = '/src/sub/config.json'
    var_26 = 'sub'
    var_27 = {var_26: var_5}
    var_28 = var_19.insert(var_25, var_27)
    var_29 = var_19.search(var_17)
    var_30 = var_19.search(var_15)
    var_31 = '/other/file.py'
    var_32 = var_19.search(var_31)
    var_33 = module_0.Trie()
    var_34 = {var_4: var_5}
    var_35 = var_33.insert(var_20, var_34)
    var_36 = '/a/b/c/d/file.py'
    var_37 = var_33.search(var_36)
    var_38 = module_0.Trie()
    var_39 = '/file.py.config.json'
    var_40 = 'file'
    var_41 = {var_40: var_5}
    var_42 = var_38.insert(var_39, var_41)
    var_43 = '/file.py'
    var_44 = var_38.search(var_43)
    var_45 = var_38.search(var_39)
    var_46 = module_0.Trie()
    var_47 = 'C:\\src\\config.json'
    var_48 = {var_12: var_5}
    var_49 = var_46.insert(var_47, var_48)
    var_50 = 'C:\\src\\main.py'
    var_51 = var_46.search(var_50)



# Parsed testcases at query #69
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = 'config.json'
    var_8 = module_0.TrieNode(var_7, var_5)



# Parsed testcases at query #70
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'to'
    var_13 = var_9.nodes[var_12]
    var_14 = var_13.nodes
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'config.json'
    var_17 = var_13.nodes[var_16]



# Parsed testcases at query #71
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test.py'
    var_3 = module_0.Trie(var_2)
    var_4 = var_3.root
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = module_0.Trie(config_data=var_7)
    var_9 = var_8.root
    var_10 = {var_5: var_6}
    var_11 = module_0.Trie(var_2, var_10)
    var_12 = var_11.root



# Parsed testcases at query #72
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/path'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'root_config.py'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/root/config.py'
    var_11 = 'root'
    var_12 = 'config'
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/root/subdir/config.py'
    var_16 = 'subdir'
    var_17 = {var_16: var_12}
    var_18 = var_9.insert(var_15, var_17)
    var_19 = '/root/subdir/subsubdir/config.py'
    var_20 = 'subsubdir'
    var_21 = {var_20: var_12}
    var_22 = var_9.insert(var_19, var_21)
    var_23 = '/root/file.py'
    var_24 = var_9.search(var_23)
    var_25 = '/root/subdir/file.py'
    var_26 = var_9.search(var_25)
    var_27 = '/root/subdir/subsubdir/file.py'
    var_28 = var_9.search(var_27)
    var_29 = module_0.Trie()
    var_30 = {var_11: var_12}
    var_31 = var_29.insert(var_10, var_30)
    var_32 = '/other/path/file.py'
    var_33 = var_29.search(var_32)
    var_34 = module_0.Trie()
    var_35 = {var_11: var_12}
    var_36 = var_34.insert(var_10, var_35)
    var_37 = {var_16: var_12}
    var_38 = var_34.insert(var_15, var_37)
    var_39 = '/root/subdir/other/file.py'
    var_40 = var_34.search(var_39)



# Parsed testcases at query #73
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = 0
    var_7 = 'new_key'
    var_8 = 'new_value'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_1, var_9)
    var_11 = '/another/path/config.json'
    var_12 = 'another_key'
    var_13 = 'another_value'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = var_0.root



# Parsed testcases at query #74
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/.config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/project/file.txt'
    var_7 = var_0.search(var_6)
    var_8 = module_0.Trie()
    var_9 = '/home/user/.config.json'
    var_10 = 'parent_key'
    var_11 = 'parent_value'
    var_12 = {var_10: var_11}
    var_13 = var_8.insert(var_9, var_12)
    var_14 = var_8.search(var_6)
    var_15 = '/root/.config.json'
    var_16 = 'root_key'
    var_17 = 'root_value'
    var_18 = {var_16: var_17}
    var_19 = module_0.Trie(var_15, var_18)
    var_20 = var_19.search(var_6)
    var_21 = module_0.Trie()
    var_22 = 'root'
    var_23 = 'config'
    var_24 = {var_22: var_23}
    var_25 = 'project'
    var_26 = {var_25: var_23}
    var_27 = 'subdir'
    var_28 = {var_27: var_23}
    var_29 = '/.root_config.json'
    var_30 = var_21.insert(var_29, var_24)
    var_31 = '/home/user/project/.project_config.json'
    var_32 = var_21.insert(var_31, var_26)
    var_33 = '/home/user/project/subdir/.subdir_config.json'
    var_34 = var_21.insert(var_33, var_28)
    var_35 = '/home/user/project/subdir/file.txt'
    var_36 = var_21.search(var_35)
    var_37 = '/home/user/project/other_file.txt'
    var_38 = var_21.search(var_37)
    var_39 = module_0.Trie()
    var_40 = '/some/random/path/file.txt'
    var_41 = var_39.search(var_40)
    var_42 = module_0.Trie()
    var_43 = '/home/user/.config.json'
    var_44 = {var_2: var_3}
    var_45 = var_42.insert(var_43, var_44)
    var_46 = '/home/otheruser/file.txt'
    var_47 = var_42.search(var_46)
    var_48 = module_0.Trie()
    var_49 = '/Home/User/Project/.config.json'
    var_50 = 'case'
    var_51 = 'sensitive'
    var_52 = {var_50: var_51}
    var_53 = var_48.insert(var_49, var_52)
    var_54 = var_48.search(var_6)
    var_55 = '/Home/User/Project/file.txt'
    var_56 = var_48.search(var_55)



# Parsed testcases at query #75
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/file.py'
    var_2 = var_0.search(var_1)
    var_3 = '/root/config.json'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_0.insert(var_3, var_6)
    var_8 = '/root/file.py'
    var_9 = var_0.search(var_8)
    var_10 = '/root/subdir/file.py'
    var_11 = var_0.search(var_10)
    var_12 = '/root/subdir/config.json'
    var_13 = 'subdir_value'
    var_14 = {var_4: var_13}
    var_15 = var_0.insert(var_12, var_14)
    var_16 = var_0.search(var_10)
    var_17 = '/root/subdir/nested/file.py'
    var_18 = var_0.search(var_17)
    var_19 = '/root/subdir/nested/config.json'
    var_20 = 'nested_value'
    var_21 = {var_4: var_20}
    var_22 = var_0.insert(var_19, var_21)
    var_23 = var_0.search(var_17)
    var_24 = '/root/subdir/nested/deeper/file.py'
    var_25 = var_0.search(var_24)
    var_26 = '/nonexistent/path/file.py'
    var_27 = var_0.search(var_26)
    var_28 = '/root/CaseSensitive/config.json'
    var_29 = 'key'
    var_30 = 'case_value'
    var_31 = {var_29: var_30}
    var_32 = var_0.insert(var_28, var_31)
    var_33 = '/root/casesensitive/file.py'
    var_34 = var_0.search(var_33)



# Parsed testcases at query #76
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/.config1.json'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/.config2.json'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/home/.config3.json'
    var_12 = 'key3'
    var_13 = 'value3'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = var_0.search(var_1)
    var_17 = '/home/user/project/subdir/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/home/user/other/file.txt'
    var_20 = var_0.search(var_19)
    var_21 = '/home/other/file.txt'
    var_22 = var_0.search(var_21)
    var_23 = module_0.Trie()
    var_24 = '/some/random/path/file.txt'
    var_25 = var_23.search(var_24)
    var_26 = '/etc/app/config.json'
    var_27 = 'key4'
    var_28 = 'value4'
    var_29 = {var_27: var_28}
    var_30 = var_0.insert(var_26, var_29)
    var_31 = '/etc/app/data/file.txt'
    var_32 = var_0.search(var_31)



# Parsed testcases at query #77
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config1.py'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/root/subdir/config2.py'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/root/subdir/subsubdir/config3.py'
    var_12 = 'key3'
    var_13 = 'value3'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = var_0.search(var_4)
    var_17 = '/root/subdir/file.py'
    var_18 = var_0.search(var_17)
    var_19 = '/root/subdir/subsubdir/file.py'
    var_20 = var_0.search(var_19)
    var_21 = '/root/otherdir/file.py'
    var_22 = var_0.search(var_21)
    var_23 = module_0.Trie()
    var_24 = '/any/path/file.py'
    var_25 = var_23.search(var_24)
    var_26 = '/root/subdir'
    var_27 = var_0.search(var_26)
    var_28 = '/completely/different/path/file.py'
    var_29 = var_0.search(var_28)



# Parsed testcases at query #78
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #79
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #80
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #81
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #82
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = var_0.search(var_4)
    var_7 = '/root/child/file.txt'
    var_8 = var_0.search(var_7)
    var_9 = '/nonexistent/path/file.txt'
    var_10 = var_0.search(var_9)
    var_11 = '/root/subdir/config.json'
    var_12 = 'subvalue'
    var_13 = {var_1: var_12}
    var_14 = var_0.insert(var_11, var_13)
    var_15 = '/root/subdir/file.txt'
    var_16 = var_0.search(var_15)
    var_17 = '/'
    var_18 = 'root'
    var_19 = 'config'
    var_20 = {var_18: var_19}
    var_21 = var_0.insert(var_17, var_20)
    var_22 = '/any/path/file.txt'
    var_23 = var_0.search(var_22)



# Parsed testcases at query #83
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = {}
    var_9 = module_0.Trie(var_5, var_8)
    var_10 = var_9.root



# Parsed testcases at query #84
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'to'
    var_13 = var_9.nodes[var_12]
    var_14 = var_13.nodes
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'config.json'
    var_17 = var_13.nodes[var_16]



# Parsed testcases at query #85
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #86
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/root/subdir/config.json'
    var_7 = 'subdir_value'
    var_8 = {var_1: var_7}
    var_9 = var_0.insert(var_6, var_8)
    var_10 = '/root/subdir/deep/config.json'
    var_11 = 'deep_value'
    var_12 = {var_1: var_11}
    var_13 = var_0.insert(var_10, var_12)
    var_14 = var_0.search(var_10)
    var_15 = '/root/subdir/deep/file.txt'
    var_16 = var_0.search(var_15)
    var_17 = '/root/other/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/nonexistent/file.txt'
    var_20 = var_0.search(var_19)
    var_21 = module_0.Trie()
    var_22 = '/any/path'
    var_23 = var_21.search(var_22)



# Parsed testcases at query #87
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #88
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #89
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #90
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #91
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/home/user/.config1'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/home/user/project/.config2'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/home/user/project/src/.config3'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = var_0.search(var_14)
    var_17 = '/home/user/project/src/main.py'
    var_18 = var_0.search(var_17)
    var_19 = '/home/user/project/test.py'
    var_20 = var_0.search(var_19)
    var_21 = '/home/user/other/file.py'
    var_22 = var_0.search(var_21)
    var_23 = '/other/path/file.py'
    var_24 = var_0.search(var_23)
    var_25 = module_0.Trie()
    var_26 = '/any/path'
    var_27 = var_25.search(var_26)



# Parsed testcases at query #92
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #93
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #94
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/root/config1.json'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/root/subdir/config2.json'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/root/subdir/subsubdir/config3.json'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = var_0.search(var_14)
    var_17 = '/root/subdir/subsubdir/other_file.json'
    var_18 = var_0.search(var_17)
    var_19 = '/root/subdir/another_file.json'
    var_20 = var_0.search(var_19)
    var_21 = '/root/some_file.json'
    var_22 = var_0.search(var_21)
    var_23 = '/other_root/file.json'
    var_24 = var_0.search(var_23)
    var_25 = module_0.Trie()
    var_26 = '/any/path/file.json'
    var_27 = var_25.search(var_26)
    var_28 = '/home/user/config.json'
    var_29 = 'home'
    var_30 = 'user_config'
    var_31 = {var_29: var_30}
    var_32 = var_0.insert(var_28, var_31)
    var_33 = '/home/user/docs/file.txt'
    var_34 = var_0.search(var_33)



# Parsed testcases at query #95
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #96
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #97
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #98
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'to'
    var_13 = var_9.nodes[var_12]
    var_14 = var_13.nodes
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'config.json'
    var_17 = var_13.nodes[var_16]



# Parsed testcases at query #99
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.py'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #100
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.cfg'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #101
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #102
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)
    var_8 = {}
    var_9 = module_0.TrieNode(var_1, var_8)



# Parsed testcases at query #103
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #104
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #105
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #106
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root



# Parsed testcases at query #107
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/path'
    var_2 = var_0.search(var_1)
    var_3 = '/root_config.json'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/a/b/config.json'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/a/config.json'
    var_16 = 'root'
    var_17 = {var_11: var_16}
    var_18 = var_9.insert(var_15, var_17)
    var_19 = '/a/b/c/file.txt'
    var_20 = var_9.search(var_19)
    var_21 = '/a/file.txt'
    var_22 = var_9.search(var_21)
    var_23 = module_0.Trie()
    var_24 = {var_11: var_12}
    var_25 = var_23.insert(var_10, var_24)
    var_26 = '/x/y/z/file.txt'
    var_27 = var_23.search(var_26)
    var_28 = module_0.Trie()
    var_29 = {var_11: var_12}
    var_30 = var_28.insert(var_10, var_29)
    var_31 = var_28.search(var_10)
    var_32 = module_0.Trie()
    var_33 = '/A/B/config.json'
    var_34 = {var_11: var_12}
    var_35 = var_32.insert(var_33, var_34)
    var_36 = '/a/b/file.txt'
    var_37 = var_32.search(var_36)



# Parsed testcases at query #108
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/root/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/root/subdir/config.json'
    var_7 = 'subvalue'
    var_8 = {var_2: var_7}
    var_9 = var_0.insert(var_6, var_8)
    var_10 = var_0.search(var_1)
    var_11 = '/root/subdir/file.txt'
    var_12 = var_0.search(var_11)
    var_13 = '/root/other/file.txt'
    var_14 = var_0.search(var_13)
    var_15 = '/'
    var_16 = var_0.search(var_15)
    var_17 = '/nonexistent/path/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = module_0.Trie()
    var_20 = '/any/path'
    var_21 = var_19.search(var_20)



# Parsed testcases at query #109
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'config.json'
    var_9 = 'to'
    var_10 = 'path'
    var_11 = var_0.root.nodes[var_10]
    var_12 = var_11.nodes[var_9]
    var_13 = var_12.nodes[var_8]
    var_14 = '/another/path/config.yaml'
    var_15 = 'another'
    var_16 = 'config'
    var_17 = {var_15: var_16}
    var_18 = var_0.insert(var_14, var_17)
    var_19 = var_0.root.nodes
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = 'config.yaml'
    var_22 = var_0.root.nodes[var_15]
    var_23 = var_22.nodes[var_10]
    var_24 = var_23.nodes[var_21]
    var_25 = '/path/to/another/config.toml'
    var_26 = 'overlap'
    var_27 = 'test'
    var_28 = {var_26: var_27}
    var_29 = var_0.insert(var_25, var_28)
    var_30 = var_0.root.nodes[var_10]
    var_31 = var_30.nodes[var_9]
    var_32 = var_31.nodes
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = 'config.toml'
    var_35 = var_0.root.nodes[var_10]
    var_36 = var_35.nodes[var_9]
    var_37 = var_36.nodes[var_15]
    var_38 = var_37.nodes[var_34]



# Parsed testcases at query #110
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #111
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #112
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #113
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'config.json'
    var_9 = 'to'
    var_10 = 'path'
    var_11 = var_0.root.nodes[var_10]
    var_12 = var_11.nodes[var_9]
    var_13 = var_12.nodes[var_8]



# Parsed testcases at query #114
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #115
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #116
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #117
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #118
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #119
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = {var_7: var_8}
    var_10 = '/root/config1.json'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/root/subdir/config2.json'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/root/subdir/subsubdir/config3.json'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = var_0.search(var_14)
    var_17 = '/root/subdir/subsubdir/other_file.json'
    var_18 = var_0.search(var_17)
    var_19 = '/root/subdir/another_file.json'
    var_20 = var_0.search(var_19)
    var_21 = '/root/some_file.json'
    var_22 = var_0.search(var_21)
    var_23 = '/nonexistent/path/file.json'
    var_24 = var_0.search(var_23)
    var_25 = module_0.Trie()
    var_26 = '/any/path/file.json'
    var_27 = var_25.search(var_26)
    var_28 = '/root_config.json'
    var_29 = 'root'
    var_30 = 'config'
    var_31 = {var_29: var_30}
    var_32 = module_0.Trie(var_28, var_31)
    var_33 = var_32.search(var_26)



# Parsed testcases at query #120
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #121
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #122
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #123
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



# Parsed testcases at query #124
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #125
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = var_9.nodes
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'to'
    var_13 = var_9.nodes[var_12]
    var_14 = var_13.nodes
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'config.json'
    var_17 = var_13.nodes[var_16]
    var_18 = module_0.Trie()
    var_19 = '/path/to/config1.json'
    var_20 = 'key1'
    var_21 = 'value1'
    var_22 = {var_20: var_21}
    var_23 = '/path/to/config2.json'
    var_24 = 'key2'
    var_25 = 'value2'
    var_26 = {var_24: var_25}
    var_27 = var_18.insert(var_19, var_22)
    var_28 = var_18.insert(var_23, var_26)
    var_29 = var_18.root.nodes[var_8]
    var_30 = var_29.nodes[var_12]
    var_31 = var_30.nodes
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = 'config1.json'
    var_34 = var_30.nodes[var_33]
    var_35 = 'config2.json'
    var_36 = var_30.nodes[var_35]
    var_37 = module_0.Trie()
    var_38 = 'config.json'
    var_39 = {var_2: var_3}
    var_40 = var_37.insert(var_38, var_39)



# Parsed testcases at query #126
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #127
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test.py'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = {var_3: var_4}
    var_8 = module_0.TrieNode(var_1, var_7)



# Parsed testcases at query #128
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'config.json'
    var_9 = 'to'
    var_10 = 'path'
    var_11 = var_0.root.nodes[var_10]
    var_12 = var_11.nodes[var_9]
    var_13 = var_12.nodes[var_8]
    var_14 = '/another/path/config.yaml'
    var_15 = 'another_key'
    var_16 = 'another_value'
    var_17 = {var_15: var_16}
    var_18 = var_0.insert(var_14, var_17)
    var_19 = var_0.root.nodes
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = 'config.yaml'
    var_22 = 'another'
    var_23 = var_0.root.nodes[var_22]
    var_24 = var_23.nodes[var_10]
    var_25 = var_24.nodes[var_21]



# Parsed testcases at query #129
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)



# Parsed testcases at query #130
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = {}
    var_9 = module_0.Trie(var_5, var_8)
    var_10 = var_9.root



# Parsed testcases at query #131
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root



# Parsed testcases at query #132
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #133
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root



# Parsed testcases at query #134
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.py'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_5)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_4)
    var_11 = var_10.root



# Parsed testcases at query #135
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #136
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root_config.py'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/subdir/config.py'
    var_7 = 'subkey'
    var_8 = 'subvalue'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/subdir/file.txt'
    var_12 = var_0.search(var_11)
    var_13 = '/subdir/nested/file.txt'
    var_14 = var_0.search(var_13)
    var_15 = '/otherdir/file.txt'
    var_16 = var_0.search(var_15)
    var_17 = '/nonexistent/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = module_0.Trie()
    var_20 = '/any/path/file.txt'
    var_21 = var_19.search(var_20)
    var_22 = '/exact/location/config.py'
    var_23 = 'exact'
    var_24 = 'match'
    var_25 = {var_23: var_24}
    var_26 = var_0.insert(var_22, var_25)
    var_27 = var_0.search(var_22)



# Parsed testcases at query #137
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'test_config.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #138
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = var_6.root
    var_8 = None
    var_9 = module_0.Trie(var_5, var_8)
    var_10 = var_9.root



# Parsed testcases at query #139
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = var_0.root
    var_2 = '/path/to/config'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = var_6.root
    var_8 = module_0.Trie(var_2)
    var_9 = var_8.root
    var_10 = module_0.Trie(config_data=var_5)
    var_11 = var_10.root



# Parsed testcases at query #140
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/.config1.json'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = '/home/user/.config2.json'
    var_6 = 'key2'
    var_7 = 'value2'
    var_8 = {var_6: var_7}
    var_9 = '/home/.config3.json'
    var_10 = 'key3'
    var_11 = 'value3'
    var_12 = {var_10: var_11}
    var_13 = var_0.insert(var_1, var_4)
    var_14 = var_0.insert(var_5, var_8)
    var_15 = var_0.insert(var_9, var_12)
    var_16 = '/home/user/project/file.txt'
    var_17 = var_0.search(var_16)
    var_18 = '/home/user/other/file.txt'
    var_19 = var_0.search(var_18)
    var_20 = '/home/other/file.txt'
    var_21 = var_0.search(var_20)
    var_22 = '/other/file.txt'
    var_23 = var_0.search(var_22)
    var_24 = module_0.Trie()
    var_25 = '/any/path/file.txt'
    var_26 = var_24.search(var_25)



# Parsed testcases at query #141
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root
    var_7 = '/another/path/config2.json'
    var_8 = 'key2'
    var_9 = 'value2'
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = var_0.root



# Parsed testcases at query #142
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root_config.py'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/subdir/sub_config.py'
    var_7 = var_0.insert(var_6, var_3)
    var_8 = '/subdir/deep/deep_config.py'
    var_9 = var_0.insert(var_8, var_3)
    var_10 = '/file_in_root.txt'
    var_11 = var_0.search(var_10)
    var_12 = '/subdir/file.txt'
    var_13 = var_0.search(var_12)
    var_14 = '/subdir/deep/file.txt'
    var_15 = var_0.search(var_14)
    var_16 = '/nonexistent/path/file.txt'
    var_17 = var_0.search(var_16)
    var_18 = '/subdir/other/file.txt'
    var_19 = var_0.search(var_18)
    var_20 = module_0.Trie()
    var_21 = '/any/path/file.txt'
    var_22 = var_20.search(var_21)



# Parsed testcases at query #143
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/root/config1.json'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/root/subdir/config2.json'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/root/subdir/subsubdir/config3.json'
    var_12 = 'key3'
    var_13 = 'value3'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = var_0.search(var_11)
    var_17 = '/root/subdir/subsubdir/other_file.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/root/other_dir/file.txt'
    var_20 = var_0.search(var_19)
    var_21 = '/nonexistent/path/file.txt'
    var_22 = var_0.search(var_21)
    var_23 = module_0.Trie()
    var_24 = '/any/path/file.txt'
    var_25 = var_23.search(var_24)



# Parsed testcases at query #144
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/.config'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/project/file.txt'
    var_7 = var_0.search(var_6)
    var_8 = '/home/user/project/subdir/.config'
    var_9 = 'subvalue'
    var_10 = {var_2: var_9}
    var_11 = var_0.insert(var_8, var_10)
    var_12 = '/home/user/project/subdir/file.txt'
    var_13 = var_0.search(var_12)
    var_14 = '/home/user/other/file.txt'
    var_15 = var_0.search(var_14)
    var_16 = '/home/user/.config'
    var_17 = 'parentvalue'
    var_18 = {var_2: var_17}
    var_19 = var_0.insert(var_16, var_18)
    var_20 = var_0.search(var_12)
    var_21 = module_0.Trie()
    var_22 = {var_2: var_17}
    var_23 = var_21.insert(var_16, var_22)
    var_24 = var_21.search(var_6)
    var_25 = module_0.Trie()
    var_26 = '/.config'
    var_27 = 'rootvalue'
    var_28 = {var_2: var_27}
    var_29 = var_25.insert(var_26, var_28)
    var_30 = var_25.search(var_6)
    var_31 = module_0.Trie()
    var_32 = {var_2: var_27}
    var_33 = var_31.insert(var_26, var_32)
    var_34 = '/home/.config'
    var_35 = 'homevalue'
    var_36 = {var_2: var_35}
    var_37 = var_31.insert(var_34, var_36)
    var_38 = 'uservalue'
    var_39 = {var_2: var_38}
    var_40 = var_31.insert(var_16, var_39)
    var_41 = var_31.search(var_6)
    var_42 = module_0.Trie()
    var_43 = var_42.search(var_6)



# Parsed testcases at query #145
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/root/config1.json'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/root/subdir/config2.json'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = '/root/subdir/subsubdir/config3.json'
    var_12 = 'key3'
    var_13 = 'value3'
    var_14 = {var_12: var_13}
    var_15 = var_0.insert(var_11, var_14)
    var_16 = '/root/test.py'
    var_17 = '/root/subdir/test.py'
    var_18 = '/root/subdir/subsubdir/test.py'
    var_19 = '/nonexistent/test.py'
    var_20 = '/root/subdir/subsubdir/deep/test.py'



# Parsed testcases at query #146
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'test_config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(config_data=var_5)
    var_7 = module_0.TrieNode(var_1, var_5)



