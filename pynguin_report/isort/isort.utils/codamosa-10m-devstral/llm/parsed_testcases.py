####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'any/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = 'root_config.json'
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/parent/config.json'
    var_11 = 'parent_value'
    var_12 = {var_4: var_11}
    var_13 = var_9.insert(var_10, var_12)
    var_14 = '/parent/child/file.txt'
    var_15 = var_9.search(var_14)
    var_16 = module_0.Trie()
    var_17 = '/parent/child/config.json'
    var_18 = 'child_value'
    var_19 = {var_4: var_18}
    var_20 = var_16.insert(var_17, var_19)
    var_21 = '/parent/child/grandchild/file.txt'
    var_22 = var_16.search(var_21)
    var_23 = module_0.Trie()
    var_24 = '/root/config.json'
    var_25 = {var_4: var_5}
    var_26 = var_23.insert(var_24, var_25)
    var_27 = {var_4: var_11}
    var_28 = var_23.insert(var_10, var_27)
    var_29 = {var_4: var_18}
    var_30 = var_23.insert(var_17, var_29)
    var_31 = var_23.search(var_21)
    var_32 = {var_4: var_5}
    var_33 = module_0.Trie(var_3, var_32)
    var_34 = {var_4: var_11}
    var_35 = var_33.insert(var_10, var_34)
    var_36 = '/unrelated/path/file.txt'
    var_37 = var_33.search(var_36)
    var_38 = module_0.Trie()
    var_39 = {var_4: var_18}
    var_40 = var_38.insert(var_17, var_39)
    var_41 = var_38.search(var_17)



# Parsed testcases at query #2
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
    var_7 = '/another/path/config2.json'
    var_8 = 'key2'
    var_9 = 'value2'
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = var_0.root



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.yaml'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = var_0.root



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
    var_7 = module_0.TrieNode(var_1, var_5)



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
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/.config'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/home/user/project/.config'
    var_7 = 'project_value'
    var_8 = {var_1: var_7}
    var_9 = var_0.insert(var_6, var_8)
    var_10 = '/home/user/project/src/.config'
    var_11 = 'src_value'
    var_12 = {var_1: var_11}
    var_13 = var_0.insert(var_10, var_12)
    var_14 = '/home/user/project/src/file.py'
    var_15 = var_0.search(var_14)
    var_16 = '/home/user/project/src/subdir/file.py'
    var_17 = var_0.search(var_16)
    var_18 = '/home/user/other/file.py'
    var_19 = var_0.search(var_18)
    var_20 = '/other/path/file.py'
    var_21 = var_0.search(var_20)
    var_22 = module_0.Trie()
    var_23 = '/any/path/file.py'
    var_24 = var_22.search(var_23)



# Parsed testcases at query #10
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



# Parsed testcases at query #11
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
    var_7 = '/home/user/project/config1.json'
    var_8 = '/home/user/config2.json'
    var_9 = var_0.insert(var_7, var_3)
    var_10 = var_0.insert(var_8, var_6)
    var_11 = '/home/user/project/config1.json'
    var_12 = var_0.search(var_11)
    var_13 = '/home/user/project/subdir/file.txt'
    var_14 = var_0.search(var_13)
    var_15 = '/different/path/file.txt'
    var_16 = var_0.search(var_15)
    var_17 = 'root_key'
    var_18 = 'root_value'
    var_19 = {var_17: var_18}
    var_20 = ''
    var_21 = module_0.Trie(var_20, var_19)
    var_22 = '/any/path/file.txt'
    var_23 = var_21.search(var_22)
    var_24 = module_0.Trie()
    var_25 = var_24.search(var_22)



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



# Parsed testcases at query #13
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



# Parsed testcases at query #14
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



# Parsed testcases at query #15
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
    var_7 = module_0.TrieNode(var_1, var_5)



# Parsed testcases at query #16
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



# Parsed testcases at query #17
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
    var_26 = module_0.Trie()
    var_27 = '/any/path/file.txt'
    var_28 = var_26.search(var_27)
    var_29 = var_0.search(var_12)



# Parsed testcases at query #18
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



# Parsed testcases at query #19
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
    var_8 = 'key2'
    var_9 = 'value2'
    var_10 = {var_8: var_9}
    var_11 = var_0.insert(var_7, var_10)
    var_12 = var_0.root
    var_13 = var_0.root



# Parsed testcases at query #20
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
    var_10 = {}
    var_11 = module_0.Trie(var_5, var_10)
    var_12 = var_11.root



# Parsed testcases at query #21
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
    var_6 = '/another/deep/path/config.json'
    var_7 = 'another'
    var_8 = 'config'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_6, var_9)
    var_11 = 'new'
    var_12 = 'data'
    var_13 = {var_11: var_12}
    var_14 = var_0.insert(var_1, var_13)
    var_15 = '/empty/config.json'
    var_16 = {}
    var_17 = var_0.insert(var_15, var_16)
    var_18 = 'relative/config.json'
    var_19 = 'relative'
    var_20 = True
    var_21 = {var_19: var_20}
    var_22 = var_0.insert(var_18, var_21)
    var_23 = var_0.root



# Parsed testcases at query #23
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
    var_19 = '/root/another_file.txt'
    var_20 = var_0.search(var_19)
    var_21 = '/nonexistent/path/file.txt'
    var_22 = var_0.search(var_21)
    var_23 = module_0.Trie()
    var_24 = '/any/path/file.txt'
    var_25 = var_23.search(var_24)



# Parsed testcases at query #24
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
    var_10 = '/root/config1.yaml'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/root/subdir/config2.yaml'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/root/subdir/subsubdir/config3.yaml'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = var_0.search(var_14)
    var_17 = '/root/subdir/subsubdir/other_file.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/root/other_file.txt'
    var_20 = var_0.search(var_19)
    var_21 = '/nonexistent/path/file.txt'
    var_22 = var_0.search(var_21)
    var_23 = module_0.Trie()
    var_24 = '/any/path/file.txt'
    var_25 = var_23.search(var_24)
    var_26 = '/root/partial/config.yaml'
    var_27 = 'partial'
    var_28 = 'data'
    var_29 = {var_27: var_28}
    var_30 = var_0.insert(var_26, var_29)
    var_31 = '/root/partial/deep/file.txt'
    var_32 = var_0.search(var_31)



# Parsed testcases at query #25
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



# Parsed testcases at query #28
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



# Parsed testcases at query #29
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



# Parsed testcases at query #30
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



# Parsed testcases at query #31
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



# Parsed testcases at query #32
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
    var_7 = 'new_key'
    var_8 = 'new_value'
    var_9 = {var_7: var_8}
    var_10 = var_0.insert(var_1, var_9)
    var_11 = var_0.root
    var_12 = '/another/path/config.json'
    var_13 = 'another_key'
    var_14 = 'another_value'
    var_15 = {var_13: var_14}
    var_16 = var_0.insert(var_12, var_15)
    var_17 = var_0.root



# Parsed testcases at query #34
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
    var_14 = '/another/path/config.json'
    var_15 = 'another_key'
    var_16 = 'another_value'
    var_17 = {var_15: var_16}
    var_18 = var_0.insert(var_14, var_17)
    var_19 = var_0.root.nodes
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = 'another'
    var_22 = var_0.root.nodes[var_21]
    var_23 = var_22.nodes[var_10]
    var_24 = var_23.nodes[var_8]



# Parsed testcases at query #35
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



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
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
    var_19 = 'another'
    var_20 = var_0.root.nodes[var_10]
    var_21 = var_20.nodes[var_9]
    var_22 = var_21.nodes[var_19]
    var_23 = var_22.nodes[var_8]



# Parsed testcases at query #3
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'config.json'
    var_5 = module_0.TrieNode(var_4, var_3)
    var_6 = module_0.TrieNode(var_4)
    var_7 = ''
    var_8 = {}
    var_9 = module_0.TrieNode(var_7, var_8)



# Parsed testcases at query #4
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = 'file.txt'
    var_7 = var_0.search(var_6)
    var_8 = module_0.Trie()
    var_9 = 'dir/config.json'
    var_10 = {var_2: var_3}
    var_11 = var_8.insert(var_9, var_10)
    var_12 = 'dir/file.txt'
    var_13 = var_8.search(var_12)
    var_14 = module_0.Trie()
    var_15 = 'dir/subdir/config.json'
    var_16 = {var_2: var_3}
    var_17 = var_14.insert(var_15, var_16)
    var_18 = 'dir/subdir/file.txt'
    var_19 = var_14.search(var_18)
    var_20 = module_0.Trie()
    var_21 = 'nonexistent/file.txt'
    var_22 = var_20.search(var_21)
    var_23 = module_0.Trie()
    var_24 = 'value1'
    var_25 = {var_2: var_24}
    var_26 = var_23.insert(var_1, var_25)
    var_27 = 'value2'
    var_28 = {var_2: var_27}
    var_29 = var_23.insert(var_9, var_28)
    var_30 = var_23.search(var_12)
    var_31 = var_23.search(var_6)
    var_32 = module_0.Trie()
    var_33 = {var_2: var_3}
    var_34 = var_32.insert(var_9, var_33)
    var_35 = var_32.search(var_18)



# Parsed testcases at query #5
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



# Parsed testcases at query #6
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



# Parsed testcases at query #7
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
    var_18 = '/path/to/another/config.json'
    var_19 = 'key2'
    var_20 = 'value2'
    var_21 = {var_19: var_20}
    var_22 = var_0.insert(var_18, var_21)
    var_23 = '/different/path/config.json'
    var_24 = 'key3'
    var_25 = 'value3'
    var_26 = {var_24: var_25}
    var_27 = var_0.insert(var_23, var_26)
    var_28 = var_0.root.nodes
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = var_0.root.nodes[var_8]
    var_31 = var_30.nodes
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = var_30.nodes[var_12]
    var_34 = var_33.nodes
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = 'another'
    var_37 = var_33.nodes[var_36]
    var_38 = var_37.nodes
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = var_37.nodes[var_16]
    var_41 = 'different'
    var_42 = var_0.root.nodes[var_41]
    var_43 = var_42.nodes
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = var_42.nodes[var_8]
    var_46 = var_45.nodes
    var_47 = len(var_46)
    assert var_47 == 1
    var_48 = var_45.nodes[var_16]
    var_49 = 'key4'
    var_50 = 'value4'
    var_51 = {var_49: var_50}
    var_52 = var_0.insert(var_16, var_51)
    var_53 = var_0.root.nodes
    var_54 = len(var_53)
    assert var_54 == 3
    var_55 = var_0.root.nodes[var_16]



# Parsed testcases at query #9
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/.config'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/file.txt'
    var_7 = var_0.search(var_6)
    var_8 = module_0.Trie()
    var_9 = {var_2: var_3}
    var_10 = var_8.insert(var_1, var_9)
    var_11 = '/home/user/subdir/file.txt'
    var_12 = var_8.search(var_11)
    var_13 = module_0.Trie()
    var_14 = var_13.search(var_6)
    var_15 = module_0.Trie()
    var_16 = '/home/.config'
    var_17 = 'value1'
    var_18 = {var_2: var_17}
    var_19 = var_15.insert(var_16, var_18)
    var_20 = 'value2'
    var_21 = {var_2: var_20}
    var_22 = var_15.insert(var_1, var_21)
    var_23 = var_15.search(var_6)
    var_24 = module_0.Trie()
    var_25 = {var_2: var_17}
    var_26 = var_24.insert(var_16, var_25)
    var_27 = {var_2: var_20}
    var_28 = var_24.insert(var_1, var_27)
    var_29 = var_24.search(var_11)
    var_30 = module_0.Trie()
    var_31 = '/.config'
    var_32 = {var_2: var_3}
    var_33 = var_30.insert(var_31, var_32)
    var_34 = var_30.search(var_6)
    var_35 = module_0.Trie()
    var_36 = {var_2: var_3}
    var_37 = var_35.insert(var_16, var_36)
    var_38 = var_35.search(var_6)



# Parsed testcases at query #10
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
    var_19 = '/root/another_file.txt'
    var_20 = var_0.search(var_19)
    var_21 = '/nonexistent/path/file.txt'
    var_22 = var_0.search(var_21)
    var_23 = '/root/subdir'
    var_24 = var_0.search(var_23)
    var_25 = module_0.Trie()
    var_26 = '/any/path'
    var_27 = var_25.search(var_26)



# Parsed testcases at query #11
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
    var_30 = '/nonexistent/path/file.txt'
    var_31 = var_10.search(var_30)
    var_32 = module_0.Trie()
    var_33 = '/a/b/c/config.json'
    var_34 = 'deep'
    var_35 = {var_34: var_13}
    var_36 = var_32.insert(var_33, var_35)
    var_37 = '/a/b/file.txt'
    var_38 = var_32.search(var_37)
    var_39 = '/a/b/c/d/file.txt'
    var_40 = var_32.search(var_39)
    var_41 = module_0.Trie()
    var_42 = '/CaseSensitive/Config.json'
    var_43 = 'case'
    var_44 = 'sensitive'
    var_45 = {var_43: var_44}
    var_46 = var_41.insert(var_42, var_45)
    var_47 = '/casesensitive/config.json'
    var_48 = var_41.search(var_47)
    var_49 = var_41.search(var_42)



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



# Parsed testcases at query #14
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/.config'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/home/user/file.txt'
    var_7 = var_0.search(var_6)
    var_8 = module_0.Trie()
    var_9 = {var_2: var_3}
    var_10 = var_8.insert(var_1, var_9)
    var_11 = '/home/user/subdir/file.txt'
    var_12 = var_8.search(var_11)
    var_13 = module_0.Trie()
    var_14 = {var_2: var_3}
    var_15 = var_13.insert(var_1, var_14)
    var_16 = '/home/other/file.txt'
    var_17 = var_13.search(var_16)
    var_18 = module_0.Trie()
    var_19 = 'value1'
    var_20 = {var_2: var_19}
    var_21 = var_18.insert(var_1, var_20)
    var_22 = '/home/user/subdir/.config'
    var_23 = 'value2'
    var_24 = {var_2: var_23}
    var_25 = var_18.insert(var_22, var_24)
    var_26 = var_18.search(var_11)
    var_27 = module_0.Trie()
    var_28 = {var_2: var_19}
    var_29 = var_27.insert(var_1, var_28)
    var_30 = {var_2: var_23}
    var_31 = var_27.insert(var_22, var_30)
    var_32 = '/home/user/subdir/subsubdir/file.txt'
    var_33 = var_27.search(var_32)
    var_34 = module_0.Trie()
    var_35 = {var_2: var_3}
    var_36 = var_34.insert(var_1, var_35)
    var_37 = '/home/other/subdir/file.txt'
    var_38 = var_34.search(var_37)
    var_39 = module_0.Trie()
    var_40 = '/.config'
    var_41 = {var_2: var_3}
    var_42 = var_39.insert(var_40, var_41)
    var_43 = var_39.search(var_6)
    var_44 = module_0.Trie()
    var_45 = {var_2: var_3}
    var_46 = var_44.insert(var_1, var_45)
    var_47 = var_44.search(var_1)
    var_48 = module_0.Trie()
    var_49 = '/home/.config'
    var_50 = {var_2: var_19}
    var_51 = var_48.insert(var_49, var_50)
    var_52 = {var_2: var_23}
    var_53 = var_48.insert(var_1, var_52)
    var_54 = var_48.search(var_11)
    var_55 = module_0.Trie()
    var_56 = {var_2: var_19}
    var_57 = var_55.insert(var_1, var_56)
    var_58 = {var_2: var_23}
    var_59 = var_55.insert(var_22, var_58)
    var_60 = var_55.search(var_6)



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
    var_7 = module_0.TrieNode(var_1, var_5)



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



# Parsed testcases at query #18
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



# Parsed testcases at query #19
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



# Parsed testcases at query #20
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



# Parsed testcases at query #21
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
    var_10 = module_0.Trie(var_2, var_7)
    var_11 = var_10.root



# Parsed testcases at query #22
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



# Parsed testcases at query #23
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



# Parsed testcases at query #24
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
    var_15 = '/another/path'
    var_16 = {}
    var_17 = var_14.insert(var_15, var_16)
    var_18 = 'another'
    var_19 = var_14.root.nodes[var_18]
    var_20 = var_19.nodes[var_10]
    var_21 = module_0.Trie()
    var_22 = '/common/path/config1.json'
    var_23 = 'key1'
    var_24 = 'value1'
    var_25 = {var_23: var_24}
    var_26 = var_21.insert(var_22, var_25)
    var_27 = '/common/path/config2.json'
    var_28 = 'key2'
    var_29 = 'value2'
    var_30 = {var_28: var_29}
    var_31 = var_21.insert(var_27, var_30)
    var_32 = 'config1.json'
    var_33 = 'common'
    var_34 = var_21.root.nodes[var_33]
    var_35 = var_34.nodes[var_10]
    var_36 = var_35.nodes[var_32]
    var_37 = 'config2.json'
    var_38 = var_21.root.nodes[var_33]
    var_39 = var_38.nodes[var_10]
    var_40 = var_39.nodes[var_37]



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
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



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root_config.py'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/any/file.py'
    var_7 = var_0.search(var_6)
    var_8 = '/subdir/config.py'
    var_9 = 'subkey'
    var_10 = 'subvalue'
    var_11 = {var_9: var_10}
    var_12 = var_0.insert(var_8, var_11)
    var_13 = '/subdir/file.py'
    var_14 = var_0.search(var_13)
    var_15 = '/subdir/nested/config.py'
    var_16 = 'nested_key'
    var_17 = 'nested_value'
    var_18 = {var_16: var_17}
    var_19 = var_0.insert(var_15, var_18)
    var_20 = '/subdir/nested/file.py'
    var_21 = var_0.search(var_20)
    var_22 = '/subdir/nested/deep/file.py'
    var_23 = var_0.search(var_22)
    var_24 = '/another/config.py'
    var_25 = 'another_key'
    var_26 = 'another_value'
    var_27 = {var_25: var_26}
    var_28 = var_0.insert(var_24, var_27)
    var_29 = '/nonexistent/path/file.py'
    var_30 = var_0.search(var_29)
    var_31 = module_0.Trie()
    var_32 = var_31.search(var_6)



# Parsed testcases at query #5
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



# Parsed testcases at query #6
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



# Parsed testcases at query #7
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
    var_7 = var_0.insert(var_6, var_3)
    var_8 = '/subdir/subsubdir/config.py'
    var_9 = var_0.insert(var_8, var_3)
    var_10 = '/file_in_root.txt'
    var_11 = var_0.search(var_10)
    var_12 = '/subdir/file.txt'
    var_13 = var_0.search(var_12)
    var_14 = '/subdir/subsubdir/file.txt'
    var_15 = var_0.search(var_14)
    var_16 = '/nonexistent/path/file.txt'
    var_17 = var_0.search(var_16)
    var_18 = '/subdir/another_subdir/file.txt'
    var_19 = var_0.search(var_18)
    var_20 = module_0.Trie()
    var_21 = '/any/path/file.txt'
    var_22 = var_20.search(var_21)



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



# Parsed testcases at query #9
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



# Parsed testcases at query #10
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



# Parsed testcases at query #13
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
    var_6 = var_0.root.nodes
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'path'
    var_9 = var_0.root.nodes[var_8]
    var_10 = 'to'
    var_11 = var_9.nodes[var_10]
    var_12 = 'config.json'
    var_13 = var_11.nodes[var_12]
    var_14 = '/another/path/config.yaml'
    var_15 = 'another_key'
    var_16 = 'another_value'
    var_17 = {var_15: var_16}
    var_18 = var_0.insert(var_14, var_17)
    var_19 = var_0.root.nodes
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = 'another'
    var_22 = var_0.root.nodes[var_21]
    var_23 = var_22.nodes[var_8]
    var_24 = 'config.yaml'
    var_25 = var_23.nodes[var_24]



# Parsed testcases at query #15
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



# Parsed testcases at query #16
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



# Parsed testcases at query #17
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



# Parsed testcases at query #18
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
    var_10 = 'to'
    var_11 = var_9.nodes[var_10]
    var_12 = 'config.json'
    var_13 = var_11.nodes[var_12]



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)



# Parsed testcases at query #21
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'config.json'
    var_5 = module_0.TrieNode(var_4, var_3)
    var_6 = module_0.TrieNode(var_4)



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/project/.config'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/home/user/project/file.py'
    var_7 = var_0.search(var_6)
    var_8 = '/home/user/project/src/file.py'
    var_9 = var_0.search(var_8)
    var_10 = '/home/user/other/file.py'
    var_11 = var_0.search(var_10)
    var_12 = module_0.Trie()
    var_13 = '/some/random/file.py'
    var_14 = var_12.search(var_13)
    var_15 = '/home/user/.config'
    var_16 = 'value2'
    var_17 = {var_1: var_16}
    var_18 = var_0.insert(var_15, var_17)
    var_19 = var_0.search(var_6)
    var_20 = '/home/user/file.py'
    var_21 = var_0.search(var_20)



# Parsed testcases at query #24
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
    var_15 = module_0.Trie()
    var_16 = '/root_config.json'
    var_17 = 'root_key'
    var_18 = 'root_value'
    var_19 = {var_17: var_18}
    var_20 = var_15.insert(var_16, var_19)
    var_21 = '/nonexistent/path/file.txt'
    var_22 = var_15.search(var_21)
    var_23 = module_0.Trie()
    var_24 = '/any/path/file.txt'
    var_25 = var_23.search(var_24)
    var_26 = module_0.Trie()
    var_27 = '/home/.config.json'
    var_28 = 'level'
    var_29 = 1
    var_30 = {var_28: var_29}
    var_31 = var_26.insert(var_27, var_30)
    var_32 = '/home/user/.config.json'
    var_33 = 2
    var_34 = {var_28: var_33}
    var_35 = var_26.insert(var_32, var_34)
    var_36 = var_26.search(var_6)



# Parsed testcases at query #25
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'test_config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = module_0.Trie(config_data=var_4)



# Parsed testcases at query #26
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/home/user/project/config1.py'
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = '/home/user/config2.py'
    var_6 = 'key2'
    var_7 = 'value2'
    var_8 = {var_6: var_7}
    var_9 = '/home/project/config3.py'
    var_10 = 'key3'
    var_11 = 'value3'
    var_12 = {var_10: var_11}
    var_13 = var_0.insert(var_1, var_4)
    var_14 = var_0.insert(var_5, var_8)
    var_15 = var_0.insert(var_9, var_12)
    var_16 = '/home/user/project/config1.py'
    var_17 = var_0.search(var_16)
    var_18 = '/home/user/project/subdir/file.py'
    var_19 = var_0.search(var_18)
    var_20 = '/home/other/file.py'
    var_21 = var_0.search(var_20)
    var_22 = '/nonexistent/path/file.py'
    var_23 = var_0.search(var_22)
    var_24 = '/home/project/subdir/nested/file.py'
    var_25 = var_0.search(var_24)



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
    var_1 = '/root/config.json'
    var_2 = 'key'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_0.insert(var_1, var_4)
    var_6 = '/root/subdir/config.json'
    var_7 = 'value2'
    var_8 = {var_2: var_7}
    var_9 = var_0.insert(var_6, var_8)
    var_10 = '/root/subdir/subsubdir/config.json'
    var_11 = 'value3'
    var_12 = {var_2: var_11}
    var_13 = var_0.insert(var_10, var_12)
    var_14 = '/root/subdir/file.txt'
    var_15 = '/root/subdir/subsubdir/file.txt'
    var_16 = '/root/otherdir/file.txt'
    var_17 = module_0.Trie()
    var_18 = '/some/nonexistent/path/file.txt'



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/home/user/.config'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/home/user/project/.config'
    var_7 = 'project_value'
    var_8 = {var_1: var_7}
    var_9 = var_0.insert(var_6, var_8)
    var_10 = '/home/user/project/file.py'
    var_11 = var_0.search(var_10)
    var_12 = '/home/user/other_file.py'
    var_13 = var_0.search(var_12)
    var_14 = '/nonexistent/path/file.py'
    var_15 = var_0.search(var_14)
    var_16 = '/home/user/project/subdir/file.py'
    var_17 = var_0.search(var_16)
    var_18 = module_0.Trie()
    var_19 = '/any/path/file.py'
    var_20 = var_18.search(var_19)



# Parsed testcases at query #31
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
    var_14 = var_0.search(var_4)
    var_15 = '/root/subdir/file.txt'
    var_16 = var_0.search(var_15)
    var_17 = '/root/subdir/subsubdir/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/nonexistent/path/file.txt'
    var_20 = var_0.search(var_19)
    var_21 = module_0.Trie()
    var_22 = '/any/path/file.txt'
    var_23 = var_21.search(var_22)



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
    var_10 = '/path/to/config1.json'
    var_11 = var_0.insert(var_10, var_3)
    var_12 = '/path/to/subdir/config2.json'
    var_13 = var_0.insert(var_12, var_6)
    var_14 = '/another/path/config3.json'
    var_15 = var_0.insert(var_14, var_9)
    var_16 = var_0.search(var_10)
    var_17 = '/path/to/subdir/file.txt'
    var_18 = var_0.search(var_17)
    var_19 = '/path/to/nonexistent/file.txt'
    var_20 = var_0.search(var_19)
    var_21 = '/another/path/file.txt'
    var_22 = var_0.search(var_21)
    var_23 = module_0.Trie()
    var_24 = '/some/random/path/file.txt'
    var_25 = var_23.search(var_24)
    var_26 = 'root'
    var_27 = 'config'
    var_28 = {var_26: var_27}
    var_29 = ''
    var_30 = module_0.Trie(var_29, var_28)
    var_31 = '/any/path/file.txt'
    var_32 = var_30.search(var_31)



# Parsed testcases at query #33
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



# Parsed testcases at query #34
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root_config.py'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = '/any/file.py'
    var_7 = var_0.search(var_6)
    var_8 = '/subdir/config.py'
    var_9 = 'subdir_key'
    var_10 = 'subdir_value'
    var_11 = {var_9: var_10}
    var_12 = var_0.insert(var_8, var_11)
    var_13 = '/subdir/file.py'
    var_14 = var_0.search(var_13)
    var_15 = '/subdir/nested/file.py'
    var_16 = var_0.search(var_15)
    var_17 = module_0.Trie()
    var_18 = '/root.py'
    var_19 = 'root'
    var_20 = 'data'
    var_21 = {var_19: var_20}
    var_22 = var_17.insert(var_18, var_21)
    var_23 = '/nonexistent/path/file.py'
    var_24 = var_17.search(var_23)
    var_25 = module_0.Trie()
    var_26 = var_25.search(var_6)
    var_27 = var_0.search(var_8)



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
    var_6 = var_0.root



