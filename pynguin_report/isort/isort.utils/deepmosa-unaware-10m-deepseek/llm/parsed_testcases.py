####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
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
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = {}
    var_8 = module_0.Trie(var_1, var_7)



# Parsed testcases at query #2
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'list'
    var_5 = 'value'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = {var_3: var_5, var_4: var_9}
    var_11 = module_0.TrieNode(var_1, var_10)
    var_12 = {}
    var_13 = module_0.TrieNode(var_1, var_12)
    var_14 = None
    var_15 = module_0.TrieNode(var_1, var_14)



# Parsed testcases at query #3
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'key1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = 'key2'
    var_11 = 'value2'
    var_12 = {var_10: var_11}
    var_13 = '/a/b/config1.json'
    var_14 = var_6.insert(var_13, var_9)
    var_15 = '/a/b/c/config2.json'
    var_16 = var_6.insert(var_15, var_12)
    var_17 = var_6.root
    var_18 = var_6.root
    var_19 = module_0.Trie()
    var_20 = 'old'
    var_21 = 'data'
    var_22 = {var_20: var_21}
    var_23 = 'new'
    var_24 = {var_23: var_21}
    var_25 = '/same/path/config.json'
    var_26 = var_19.insert(var_25, var_22)
    var_27 = var_19.insert(var_25, var_24)
    var_28 = var_19.root
    var_29 = 'initial.json'
    var_30 = 'initial'
    var_31 = 'config'
    var_32 = {var_30: var_31}
    var_33 = module_0.Trie(var_29, var_32)
    var_34 = {var_23: var_31}
    var_35 = '/new/path/config.json'
    var_36 = var_33.insert(var_35, var_34)
    var_37 = var_33.root
    var_38 = module_0.Trie()
    var_39 = 'config1'
    var_40 = 'data1'
    var_41 = {var_39: var_40}
    var_42 = 'config2'
    var_43 = 'data2'
    var_44 = {var_42: var_43}
    var_45 = '/parent/dir/config1.json'
    var_46 = var_38.insert(var_45, var_41)
    var_47 = '/parent/dir/config2.json'
    var_48 = var_38.insert(var_47, var_44)
    var_49 = var_38.root



# Parsed testcases at query #4
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'settings'
    var_8 = 'debug'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = '/home/user/project/config.yaml'
    var_13 = var_6.insert(var_12, var_11)
    var_14 = 'home'
    var_15 = var_6.root.nodes[var_14]
    var_16 = 'user'
    var_17 = var_15.nodes[var_16]
    var_18 = 'project'
    var_19 = var_17.nodes[var_18]
    var_20 = module_0.Trie()
    var_21 = 'name'
    var_22 = 'config1'
    var_23 = {var_21: var_22}
    var_24 = 'config2'
    var_25 = {var_21: var_24}
    var_26 = '/a/b/c/config1.json'
    var_27 = var_20.insert(var_26, var_23)
    var_28 = '/a/b/config2.json'
    var_29 = var_20.insert(var_28, var_25)
    var_30 = 'a'
    var_31 = var_20.root.nodes[var_30]
    var_32 = 'b'
    var_33 = var_31.nodes[var_32]
    var_34 = 'c'
    var_35 = var_33.nodes[var_34]
    var_36 = module_0.Trie()
    var_37 = 'version'
    var_38 = {var_37: var_9}
    var_39 = 2
    var_40 = {var_37: var_39}
    var_41 = '/path/config.json'
    var_42 = var_36.insert(var_41, var_38)
    var_43 = var_36.insert(var_41, var_40)
    var_44 = module_0.Trie()
    var_45 = {}
    var_46 = '/empty/config.json'
    var_47 = var_44.insert(var_46, var_45)
    var_48 = var_44.root
    var_49 = module_0.Trie()
    var_50 = '/x/y/z/config1.json'
    var_51 = 'id'
    var_52 = {var_51: var_9}
    var_53 = var_49.insert(var_50, var_52)
    var_54 = '/x/y/z/w/config2.json'
    var_55 = {var_51: var_39}
    var_56 = var_49.insert(var_54, var_55)
    var_57 = 'x'
    var_58 = var_49.root.nodes[var_57]
    var_59 = 'y'
    var_60 = var_58.nodes[var_59]
    var_61 = 'z'
    var_62 = var_60.nodes[var_61]
    var_63 = 'w'
    var_64 = var_62.nodes[var_63]



# Parsed testcases at query #5
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'settings'
    var_8 = 'debug'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = '/home/user/project/src/config.yaml'
    var_13 = var_6.insert(var_12, var_11)
    var_14 = var_6.root
    var_15 = module_0.Trie()
    var_16 = 'name'
    var_17 = 'config1'
    var_18 = {var_16: var_17}
    var_19 = 'config2'
    var_20 = {var_16: var_19}
    var_21 = '/a/b/c/config1.json'
    var_22 = var_15.insert(var_21, var_18)
    var_23 = '/a/b/config2.json'
    var_24 = var_15.insert(var_23, var_20)
    var_25 = var_15.root
    var_26 = var_15.root
    var_27 = module_0.Trie()
    var_28 = 'version'
    var_29 = {var_28: var_9}
    var_30 = 2
    var_31 = {var_28: var_30}
    var_32 = '/path/to/config.json'
    var_33 = var_27.insert(var_32, var_29)
    var_34 = var_27.insert(var_32, var_31)
    var_35 = var_27.root
    var_36 = module_0.Trie()
    var_37 = '/empty/config.json'
    var_38 = {}
    var_39 = var_36.insert(var_37, var_38)
    var_40 = var_36.root
    var_41 = module_0.Trie()
    var_42 = 'test'
    var_43 = 'relative'
    var_44 = {var_42: var_43}
    var_45 = './relative/path/config.json'
    var_46 = var_41.insert(var_45, var_44)
    var_47 = var_41.root



# Parsed testcases at query #6
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = '/root/config.json'
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = '/root/subdir/file.txt'
    var_9 = var_7.search(var_8)
    var_10 = module_0.Trie()
    var_11 = '/project/config.json'
    var_12 = 'env'
    var_13 = 'project'
    var_14 = {var_12: var_13}
    var_15 = var_10.insert(var_11, var_14)
    var_16 = '/project/src/config.json'
    var_17 = 'src'
    var_18 = {var_12: var_17}
    var_19 = var_10.insert(var_16, var_18)
    var_20 = '/project/src/utils/file.py'
    var_21 = var_10.search(var_20)
    var_22 = module_0.Trie()
    var_23 = '/home/user/config.json'
    var_24 = 'user'
    var_25 = 'test'
    var_26 = {var_24: var_25}
    var_27 = var_22.insert(var_23, var_26)
    var_28 = '/home/user/docs/notes.txt'
    var_29 = var_22.search(var_28)
    var_30 = module_0.Trie()
    var_31 = '/a/config.json'
    var_32 = 'level'
    var_33 = 'a'
    var_34 = {var_32: var_33}
    var_35 = var_30.insert(var_31, var_34)
    var_36 = '/a/b/config.json'
    var_37 = 'b'
    var_38 = {var_32: var_37}
    var_39 = var_30.insert(var_36, var_38)
    var_40 = '/a/b/c/config.json'
    var_41 = 'c'
    var_42 = {var_32: var_41}
    var_43 = var_30.insert(var_40, var_42)
    var_44 = '/a/b/c/d/e/file.txt'
    var_45 = var_30.search(var_44)
    var_46 = module_0.Trie()
    var_47 = '/etc/app/config.json'
    var_48 = 'global'
    var_49 = True
    var_50 = {var_48: var_49}
    var_51 = var_46.insert(var_47, var_50)
    var_52 = '/var/log/app.log'
    var_53 = var_46.search(var_52)
    var_54 = module_0.Trie()
    var_55 = '/dir/config.json'
    var_56 = 'dir'
    var_57 = 'config'
    var_58 = {var_56: var_57}
    var_59 = var_54.insert(var_55, var_58)
    var_60 = '/dir/file.txt'
    var_61 = var_54.search(var_60)
    var_62 = '/config.json'
    var_63 = 'root'
    var_64 = {var_63: var_49}
    var_65 = module_0.Trie(var_62, var_64)
    var_66 = '/file.txt'
    var_67 = var_65.search(var_66)
    var_68 = module_0.Trie()
    var_69 = '/path/config.json'
    var_70 = 'version'
    var_71 = {var_70: var_49}
    var_72 = var_68.insert(var_69, var_71)
    var_73 = 2
    var_74 = {var_70: var_73}
    var_75 = var_68.insert(var_69, var_74)
    var_76 = '/path/sub/file.txt'
    var_77 = var_68.search(var_76)
    var_78 = module_0.Trie()
    var_79 = './local/config.json'
    var_80 = 'local'
    var_81 = {var_80: var_49}
    var_82 = var_78.insert(var_79, var_81)
    var_83 = './local/file.txt'
    var_84 = var_78.search(var_83)



# Parsed testcases at query #7
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'key1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = 'key2'
    var_11 = 'value2'
    var_12 = {var_10: var_11}
    var_13 = var_6.insert(var_4, var_9)
    var_14 = '/root/subdir/config.json'
    var_15 = var_6.insert(var_14, var_12)
    var_16 = var_6.root
    var_17 = module_0.Trie()
    var_18 = 'a'
    var_19 = 1
    var_20 = {var_18: var_19}
    var_21 = 'b'
    var_22 = 2
    var_23 = {var_21: var_22}
    var_24 = 'c'
    var_25 = 3
    var_26 = {var_24: var_25}
    var_27 = '/project/a/config.json'
    var_28 = var_17.insert(var_27, var_20)
    var_29 = '/project/b/config.json'
    var_30 = var_17.insert(var_29, var_23)
    var_31 = '/other/config.json'
    var_32 = var_17.insert(var_31, var_26)
    var_33 = var_17.root
    var_34 = var_17.root
    var_35 = var_17.root
    var_36 = module_0.Trie()
    var_37 = 'old'
    var_38 = 'data'
    var_39 = {var_37: var_38}
    var_40 = 'new'
    var_41 = {var_40: var_38}
    var_42 = '/path/config.json'
    var_43 = var_36.insert(var_42, var_39)
    var_44 = var_36.insert(var_42, var_41)
    var_45 = var_36.root
    var_46 = module_0.Trie()
    var_47 = 'test'
    var_48 = {var_47: var_38}
    var_49 = '/some/path/config.json'
    var_50 = var_46.insert(var_49, var_48)
    var_51 = var_46.root
    var_52 = '/some/path'



# Parsed testcases at query #8
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #9
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'key1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = 'key2'
    var_11 = 'value2'
    var_12 = {var_10: var_11}
    var_13 = '/a/b/config1.json'
    var_14 = var_6.insert(var_13, var_9)
    var_15 = '/a/b/c/config2.json'
    var_16 = var_6.insert(var_15, var_12)
    var_17 = 'a'
    var_18 = 'b'
    var_19 = 'c'
    var_20 = module_0.Trie()
    var_21 = 'key3'
    var_22 = 'value3'
    var_23 = {var_21: var_22}
    var_24 = 'key4'
    var_25 = 'value4'
    var_26 = {var_24: var_25}
    var_27 = '/x/y/config3.json'
    var_28 = var_20.insert(var_27, var_23)
    var_29 = '/x/y/config4.json'
    var_30 = var_20.insert(var_29, var_26)
    var_31 = 'x'
    var_32 = 'y'
    var_33 = module_0.Trie()
    var_34 = '/empty/config.json'
    var_35 = {}
    var_36 = var_33.insert(var_34, var_35)
    var_37 = 'empty'
    var_38 = module_0.Trie()
    var_39 = 'nested'
    var_40 = {var_1: var_2}
    var_41 = {var_39: var_40}
    var_42 = '/home/user/projects/src/main/config.yaml'
    var_43 = var_38.insert(var_42, var_41)
    var_44 = var_38.root
    var_45 = module_0.Trie()
    var_46 = '/common/prefix/config1.json'
    var_47 = 'id'
    var_48 = 1
    var_49 = {var_47: var_48}
    var_50 = var_45.insert(var_46, var_49)
    var_51 = '/common/prefix/extended/config2.json'
    var_52 = 2
    var_53 = {var_47: var_52}
    var_54 = var_45.insert(var_51, var_53)
    var_55 = '/common/other/config3.json'
    var_56 = 3
    var_57 = {var_47: var_56}
    var_58 = var_45.insert(var_55, var_57)
    var_59 = 'common'
    var_60 = 'prefix'
    var_61 = 'extended'
    var_62 = 'other'



# Parsed testcases at query #10
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = {}
    var_8 = module_0.Trie(var_1, var_7)
    var_9 = ''
    var_10 = module_0.Trie(var_9, var_4)



# Parsed testcases at query #11
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = ''
    var_8 = module_0.Trie(var_7, var_4)
    var_9 = var_0.root
    var_10 = var_5.root



# Parsed testcases at query #12
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'settings'
    var_8 = 'debug'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = '/home/user/project/config.json'
    var_13 = var_6.insert(var_12, var_11)
    var_14 = var_6.root
    var_15 = module_0.Trie()
    var_16 = 'name'
    var_17 = 'config1'
    var_18 = {var_16: var_17}
    var_19 = 'config2'
    var_20 = {var_16: var_19}
    var_21 = '/a/b/c/config1.json'
    var_22 = var_15.insert(var_21, var_18)
    var_23 = '/a/b/config2.json'
    var_24 = var_15.insert(var_23, var_20)
    var_25 = var_15.root
    var_26 = var_15.root
    var_27 = module_0.Trie()
    var_28 = 'version'
    var_29 = {var_28: var_9}
    var_30 = 2
    var_31 = {var_28: var_30}
    var_32 = '/path/config.json'
    var_33 = var_27.insert(var_32, var_29)
    var_34 = var_27.insert(var_32, var_31)
    var_35 = var_27.root
    var_36 = module_0.Trie()
    var_37 = '/empty/config.json'
    var_38 = {}
    var_39 = var_36.insert(var_37, var_38)
    var_40 = var_36.root
    var_41 = module_0.Trie()
    var_42 = 'test'
    var_43 = 'relative'
    var_44 = {var_42: var_43}
    var_45 = 'subdir'
    var_46 = True
    var_47 = 'subdir/config.json'
    var_48 = var_41.insert(var_47, var_44)
    var_49 = var_41.root



# Parsed testcases at query #13
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = '/root/config.json'
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = '/some/deep/nested/file.py'
    var_9 = var_7.search(var_8)
    var_10 = module_0.Trie()
    var_11 = '/home/user/project/config.json'
    var_12 = 'env'
    var_13 = 'project'
    var_14 = {var_12: var_13}
    var_15 = var_10.insert(var_11, var_14)
    var_16 = '/home/user/project/src/config.json'
    var_17 = 'src'
    var_18 = {var_12: var_17}
    var_19 = var_10.insert(var_16, var_18)
    var_20 = '/home/user/project/src/utils/config.json'
    var_21 = 'utils'
    var_22 = {var_12: var_21}
    var_23 = var_10.insert(var_20, var_22)
    var_24 = '/home/user/project/src/utils/helper.py'
    var_25 = var_10.search(var_24)
    var_26 = '/home/user/project/src/main.py'
    var_27 = var_10.search(var_26)
    var_28 = '/home/user/project/README.md'
    var_29 = var_10.search(var_28)
    var_30 = '/home/user/other/file.py'
    var_31 = var_10.search(var_30)
    var_32 = module_0.Trie()
    var_33 = '/a/b/config.json'
    var_34 = 'config'
    var_35 = 'ab'
    var_36 = {var_34: var_35}
    var_37 = var_32.insert(var_33, var_36)
    var_38 = '/a/b/c/config.json'
    var_39 = 'abc'
    var_40 = {var_34: var_39}
    var_41 = var_32.insert(var_38, var_40)
    var_42 = '/a/b/c/d/file.py'
    var_43 = var_32.search(var_42)
    var_44 = '/a/b/e/file.py'
    var_45 = var_32.search(var_44)
    var_46 = module_0.Trie()
    var_47 = '/absolute/path/config.json'
    var_48 = 'abs'
    var_49 = 'true'
    var_50 = {var_48: var_49}
    var_51 = var_46.insert(var_47, var_50)
    var_52 = './file.py'
    var_53 = var_46.search(var_52)
    var_54 = module_0.Trie()
    var_55 = '/dir/config.json'
    var_56 = 'exact'
    var_57 = 'match'
    var_58 = {var_56: var_57}
    var_59 = var_54.insert(var_55, var_58)
    var_60 = var_54.search(var_55)
    var_61 = module_0.Trie()
    var_62 = '/config.json'
    var_63 = 'level'
    var_64 = 'root'
    var_65 = {var_63: var_64}
    var_66 = var_61.insert(var_62, var_65)
    var_67 = '/usr/config.json'
    var_68 = 'usr'
    var_69 = {var_63: var_68}
    var_70 = var_61.insert(var_67, var_69)
    var_71 = '/usr/local/config.json'
    var_72 = 'local'
    var_73 = {var_63: var_72}
    var_74 = var_61.insert(var_71, var_73)
    var_75 = '/usr/local/bin/config.json'
    var_76 = 'bin'
    var_77 = {var_63: var_76}
    var_78 = var_61.insert(var_75, var_77)
    var_79 = '/usr/local/bin/script.py'
    var_80 = var_61.search(var_79)
    var_81 = '/usr/local/lib/file.py'
    var_82 = var_61.search(var_81)
    var_83 = '/usr/share/file.py'
    var_84 = var_61.search(var_83)
    var_85 = '/etc/file.py'
    var_86 = var_61.search(var_85)
    var_87 = module_0.Trie()
    var_88 = '//double//slash//config.json'
    var_89 = 'test'
    var_90 = 'empty'
    var_91 = {var_89: var_90}
    var_92 = var_87.insert(var_88, var_91)
    var_93 = module_0.Trie()
    var_94 = './local.config.json'
    var_95 = True
    var_96 = {var_72: var_95}
    var_97 = var_93.insert(var_94, var_96)



# Parsed testcases at query #14
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)
    var_13 = 'original'
    var_14 = {var_3: var_13}
    var_15 = module_0.TrieNode(var_1, var_14)



# Parsed testcases at query #15
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = '/root/config1.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = '/root/dir1/dir2/config2.json'
    var_11 = var_6.insert(var_10, var_9)
    var_12 = var_6.root
    var_13 = 'root'
    var_14 = var_12.nodes[var_13]
    var_15 = 'dir1'
    var_16 = var_14.nodes[var_15]
    var_17 = 'dir2'
    var_18 = var_16.nodes[var_17]
    var_19 = module_0.Trie()
    var_20 = 'key3'
    var_21 = 'value3'
    var_22 = {var_20: var_21}
    var_23 = 'key4'
    var_24 = 'value4'
    var_25 = {var_23: var_24}
    var_26 = '/home/user/project/config3.json'
    var_27 = var_19.insert(var_26, var_22)
    var_28 = '/home/user/project/config4.json'
    var_29 = var_19.insert(var_28, var_25)
    var_30 = var_19.root
    var_31 = 'home'
    var_32 = var_30.nodes[var_31]
    var_33 = 'user'
    var_34 = var_32.nodes[var_33]
    var_35 = 'project'
    var_36 = var_34.nodes[var_35]
    var_37 = module_0.Trie()
    var_38 = 'key5'
    var_39 = 'value5'
    var_40 = {var_38: var_39}
    var_41 = './relative/path/config5.json'
    var_42 = var_37.insert(var_41, var_40)
    var_43 = var_37.root
    var_44 = module_0.Trie()
    var_45 = 'key6'
    var_46 = 'value6'
    var_47 = {var_45: var_46}
    var_48 = 'key7'
    var_49 = 'value7'
    var_50 = {var_48: var_49}
    var_51 = '/same/path/config6.json'
    var_52 = var_44.insert(var_51, var_47)
    var_53 = var_44.insert(var_51, var_50)
    var_54 = var_44.root
    var_55 = 'same'
    var_56 = var_54.nodes[var_55]
    var_57 = 'path'
    var_58 = var_56.nodes[var_57]
    var_59 = 'initial.json'
    var_60 = 'initial'
    var_61 = 'config'
    var_62 = {var_60: var_61}
    var_63 = module_0.Trie(var_59, var_62)
    var_64 = 'key8'
    var_65 = 'value8'
    var_66 = {var_64: var_65}
    var_67 = '/new/config8.json'
    var_68 = var_63.insert(var_67, var_66)
    var_69 = var_63.root
    var_70 = 'new'
    var_71 = var_69.nodes[var_70]
    var_72 = module_0.Trie()
    var_73 = 'key9'
    var_74 = 'value9'
    var_75 = {var_73: var_74}
    var_76 = '/a/b/c/d/e/f/g/h/i/j/deep_config.json'
    var_77 = var_72.insert(var_76, var_75)
    var_78 = var_72.root



# Parsed testcases at query #16
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
    var_7 = module_0.Trie(var_2)
    var_8 = {}
    var_9 = module_0.Trie(var_2, var_8)
    var_10 = var_0.root
    var_11 = var_6.root
    var_12 = var_7.root
    var_13 = var_9.root



# Parsed testcases at query #17
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = '/root/config.json'
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/project/config.json'
    var_11 = 'env'
    var_12 = 'project'
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/project/src/config.json'
    var_16 = 'src'
    var_17 = {var_11: var_16}
    var_18 = var_9.insert(var_15, var_17)
    var_19 = '/project/src/utils/config.json'
    var_20 = 'utils'
    var_21 = {var_11: var_20}
    var_22 = var_9.insert(var_19, var_21)
    var_23 = '/project/main.py'
    var_24 = var_9.search(var_23)
    var_25 = '/project/src/module.py'
    var_26 = var_9.search(var_25)
    var_27 = '/project/src/utils/helper.py'
    var_28 = var_9.search(var_27)
    var_29 = '/project/src/utils/subdir/file.py'
    var_30 = var_9.search(var_29)
    var_31 = module_0.Trie()
    var_32 = '/a/b/config.json'
    var_33 = 'config'
    var_34 = 'b'
    var_35 = {var_33: var_34}
    var_36 = var_31.insert(var_32, var_35)
    var_37 = '/a/b/c/config.json'
    var_38 = 'c'
    var_39 = {var_33: var_38}
    var_40 = var_31.insert(var_37, var_39)
    var_41 = '/a/b/d/file.py'
    var_42 = var_31.search(var_41)
    var_43 = module_0.Trie()
    var_44 = '/home/user/project/config.json'
    var_45 = 'user'
    var_46 = 'test'
    var_47 = {var_45: var_46}
    var_48 = var_43.insert(var_44, var_47)
    var_49 = '/home/user/other/file.py'
    var_50 = var_43.search(var_49)
    var_51 = module_0.Trie()
    var_52 = './config.json'
    var_53 = 'relative'
    var_54 = True
    var_55 = {var_53: var_54}
    var_56 = var_51.insert(var_52, var_55)
    var_57 = './file.py'
    var_58 = var_51.search(var_57)
    var_59 = '/global/config.json'
    var_60 = 'scope'
    var_61 = 'global'
    var_62 = {var_60: var_61}
    var_63 = module_0.Trie(var_59, var_62)
    var_64 = '/home/user/config.json'
    var_65 = {var_60: var_45}
    var_66 = var_63.insert(var_64, var_65)
    var_67 = {var_60: var_12}
    var_68 = var_63.insert(var_44, var_67)
    var_69 = '/home/user/project/src/file.py'
    var_70 = var_63.search(var_69)
    var_71 = '/home/user/docs/readme.md'
    var_72 = var_63.search(var_71)
    var_73 = '/etc/system/file.conf'
    var_74 = var_63.search(var_73)
    var_75 = module_0.Trie()
    var_76 = '//config.json'
    var_77 = 'empty'
    var_78 = 'path'
    var_79 = {var_77: var_78}
    var_80 = var_75.insert(var_76, var_79)
    var_81 = '//file.txt'
    var_82 = var_75.search(var_81)
    var_83 = len(var_82)
    var_84 = 2
    var_85 = var_83 == var_84
    var_86 = module_0.Trie()
    var_87 = '/config.json'
    var_88 = 'root'
    var_89 = {var_88: var_54}
    var_90 = var_86.insert(var_87, var_89)
    var_91 = '/file.txt'
    var_92 = var_86.search(var_91)
    var_93 = module_0.Trie()
    var_94 = '/path/config.json'
    var_95 = 'version'
    var_96 = {var_95: var_54}
    var_97 = var_93.insert(var_94, var_96)
    var_98 = {var_95: var_84}
    var_99 = var_93.insert(var_94, var_98)
    var_100 = '/path/file.txt'
    var_101 = var_93.search(var_100)



# Parsed testcases at query #18
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = {}
    var_12 = module_0.TrieNode(var_1, var_11)
    var_13 = None
    var_14 = module_0.TrieNode(var_1, var_13)
    var_15 = module_0.TrieNode()
    var_16 = module_0.TrieNode()



# Parsed testcases at query #19
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = None
    var_7 = module_0.Trie(var_1, var_6)
    var_8 = ''
    var_9 = module_0.Trie(var_8, var_4)
    var_10 = var_0.root
    var_11 = var_5.root



# Parsed testcases at query #20
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = '/root/config.json'
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/home/user/project/config.json'
    var_11 = 'env'
    var_12 = 'project'
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/home/user/config.json'
    var_16 = 'user'
    var_17 = {var_11: var_16}
    var_18 = var_9.insert(var_15, var_17)
    var_19 = '/home/config.json'
    var_20 = 'home'
    var_21 = {var_11: var_20}
    var_22 = var_9.insert(var_19, var_21)
    var_23 = '/config.json'
    var_24 = 'root'
    var_25 = {var_11: var_24}
    var_26 = var_9.insert(var_23, var_25)
    var_27 = '/home/user/project/src/file.py'
    var_28 = var_9.search(var_27)
    var_29 = '/home/user/docs/readme.md'
    var_30 = var_9.search(var_29)
    var_31 = '/home/downloads/file.txt'
    var_32 = var_9.search(var_31)
    var_33 = '/etc/config/file.conf'
    var_34 = var_9.search(var_33)
    var_35 = module_0.Trie()
    var_36 = '/a/b/config.json'
    var_37 = 'name'
    var_38 = 'b_config'
    var_39 = {var_37: var_38}
    var_40 = var_35.insert(var_36, var_39)
    var_41 = '/a/b/c/d/file.txt'
    var_42 = var_35.search(var_41)
    var_43 = module_0.Trie()
    var_44 = '/x/y/config.json'
    var_45 = 'test'
    var_46 = 'value'
    var_47 = {var_45: var_46}
    var_48 = var_43.insert(var_44, var_47)
    var_49 = '/a/b/c/file.txt'
    var_50 = var_43.search(var_49)
    var_51 = module_0.Trie()
    var_52 = '/project/config.json'
    var_53 = 'exact'
    var_54 = 'match'
    var_55 = {var_53: var_54}
    var_56 = var_51.insert(var_52, var_55)
    var_57 = var_51.search(var_52)
    var_58 = module_0.Trie()
    var_59 = '/a/config.json'
    var_60 = 'level'
    var_61 = 'a'
    var_62 = {var_60: var_61}
    var_63 = var_58.insert(var_59, var_62)
    var_64 = 'b'
    var_65 = {var_60: var_64}
    var_66 = var_58.insert(var_36, var_65)
    var_67 = '/a/b/c/config.json'
    var_68 = 'c'
    var_69 = {var_60: var_68}
    var_70 = var_58.insert(var_67, var_69)
    var_71 = '/a/b/c/d/e/file.txt'
    var_72 = var_58.search(var_71)
    var_73 = module_0.Trie()
    var_74 = './config.json'
    var_75 = 'relative'
    var_76 = 'config'
    var_77 = {var_75: var_76}
    var_78 = var_73.insert(var_74, var_77)
    var_79 = './file.txt'
    var_80 = var_73.search(var_79)
    var_81 = 0
    var_82 = var_80[var_81]
    var_83 = '/default/config.json'
    var_84 = 'default'
    var_85 = True
    var_86 = {var_84: var_85}
    var_87 = module_0.Trie(var_83, var_86)
    var_88 = ''
    var_89 = var_87.search(var_88)
    var_90 = module_0.Trie()
    var_91 = 'C:\\Users\\Project\\config.json'
    var_92 = 'os'
    var_93 = 'windows'
    var_94 = {var_92: var_93}
    var_95 = var_90.insert(var_91, var_94)
    var_96 = 'C:\\Users\\config.json'
    var_97 = 'users'
    var_98 = {var_92: var_97}
    var_99 = var_90.insert(var_96, var_98)
    var_100 = 'C:\\Users\\Project\\src\\file.py'
    var_101 = var_90.search(var_100)



# Parsed testcases at query #21
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = {}
    var_8 = module_0.Trie(var_1, var_7)
    var_9 = var_0.root
    var_10 = var_5.root



# Parsed testcases at query #22
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
    var_7 = module_0.Trie(var_2)
    var_8 = {}
    var_9 = module_0.Trie(var_2, var_8)
    var_10 = ''
    var_11 = module_0.Trie(var_10, var_5)



# Parsed testcases at query #23
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = '/root_config.json'
    var_8 = var_3.insert(var_7, var_6)
    var_9 = '/file.txt'
    var_10 = var_3.search(var_9)
    var_11 = '/subdir/file.txt'
    var_12 = var_3.search(var_11)
    var_13 = module_0.Trie()
    var_14 = 'user'
    var_15 = 'config'
    var_16 = {var_14: var_15}
    var_17 = '/home/user/.config.json'
    var_18 = var_13.insert(var_17, var_16)
    var_19 = 'home'
    var_20 = {var_19: var_15}
    var_21 = '/home/.config.json'
    var_22 = var_13.insert(var_21, var_20)
    var_23 = 'root'
    var_24 = {var_23: var_15}
    var_25 = '/.config.json'
    var_26 = var_13.insert(var_25, var_24)
    var_27 = '/home/user/project/file.txt'
    var_28 = var_13.search(var_27)
    var_29 = '/home/other/file.txt'
    var_30 = var_13.search(var_29)
    var_31 = '/tmp/file.txt'
    var_32 = var_13.search(var_31)
    var_33 = module_0.Trie()
    var_34 = 'test'
    var_35 = 'data'
    var_36 = {var_34: var_35}
    var_37 = '/absolute/path/to/config.json'
    var_38 = var_33.insert(var_37, var_36)
    var_39 = '/absolute/path/to/file.txt'
    var_40 = var_33.search(var_39)
    var_41 = '/absolute/path/to/subdir/file.txt'
    var_42 = var_33.search(var_41)
    var_43 = module_0.Trie()
    var_44 = 'deep'
    var_45 = {var_44: var_15}
    var_46 = '/a/b/c/d/config.json'
    var_47 = var_43.insert(var_46, var_45)
    var_48 = '/x/y/z/file.txt'
    var_49 = var_43.search(var_48)
    var_50 = module_0.Trie()
    var_51 = 'level'
    var_52 = 'intermediate'
    var_53 = {var_51: var_52}
    var_54 = '/project/src/.config.json'
    var_55 = var_50.insert(var_54, var_53)
    var_56 = 'leaf'
    var_57 = {var_51: var_56}
    var_58 = '/project/src/components/.config.json'
    var_59 = var_50.insert(var_58, var_57)
    var_60 = '/project/src/components/Button.jsx'
    var_61 = var_50.search(var_60)
    var_62 = '/project/src/utils/helper.js'
    var_63 = var_50.search(var_62)
    var_64 = module_0.Trie()
    var_65 = 'version'
    var_66 = 1
    var_67 = {var_65: var_66}
    var_68 = '/config.json'
    var_69 = var_64.insert(var_68, var_67)
    var_70 = 2
    var_71 = {var_65: var_70}
    var_72 = var_64.insert(var_68, var_71)
    var_73 = var_64.search(var_9)
    var_74 = 'initial'
    var_75 = {var_74: var_15}
    var_76 = '/root.json'
    var_77 = module_0.Trie(var_76, var_75)
    var_78 = '/any/path/file.txt'
    var_79 = var_77.search(var_78)
    var_80 = module_0.Trie()
    var_81 = 'mixed'
    var_82 = {var_34: var_81}
    var_83 = '/home/user/project/.config.json'
    var_84 = var_80.insert(var_83, var_82)
    var_85 = '/home/user/project/../project/./src/file.py'
    var_86 = var_80.search(var_85)
    var_87 = module_0.Trie()
    var_88 = True
    var_89 = {var_23: var_88}
    var_90 = var_87.insert(var_25, var_89)
    var_91 = var_87.search(var_9)



# Parsed testcases at query #24
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'root_value'
    var_5 = {var_3: var_4}
    var_6 = '/root/config.json'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = 'root'
    var_11 = {var_3: var_10}
    var_12 = var_9.insert(var_6, var_11)
    var_13 = '/root/project/config.json'
    var_14 = 'project'
    var_15 = {var_3: var_14}
    var_16 = var_9.insert(var_13, var_15)
    var_17 = '/root/project/src/config.json'
    var_18 = 'src'
    var_19 = {var_3: var_18}
    var_20 = var_9.insert(var_17, var_19)
    var_21 = '/root/project/src/file.py'
    var_22 = var_9.search(var_21)
    var_23 = '/root/project/file.py'
    var_24 = var_9.search(var_23)
    var_25 = '/root/file.py'
    var_26 = var_9.search(var_25)
    var_27 = '/other/file.py'
    var_28 = var_9.search(var_27)
    var_29 = module_0.Trie()
    var_30 = '/a/b/config.json'
    var_31 = 'b'
    var_32 = {var_3: var_31}
    var_33 = var_29.insert(var_30, var_32)
    var_34 = '/a/b/c/config.json'
    var_35 = 'c'
    var_36 = {var_3: var_35}
    var_37 = var_29.insert(var_34, var_36)
    var_38 = '/a/b/c/file.py'
    var_39 = var_29.search(var_38)
    var_40 = '/a/b/file.py'
    var_41 = var_29.search(var_40)
    var_42 = module_0.Trie()
    var_43 = '/home/user/docs/config.json'
    var_44 = 'docs'
    var_45 = {var_3: var_44}
    var_46 = var_42.insert(var_43, var_45)
    var_47 = '/home/user/docs_backup/config.json'
    var_48 = 'docs_backup'
    var_49 = {var_3: var_48}
    var_50 = var_42.insert(var_47, var_49)
    var_51 = '/home/user/docs/file.txt'
    var_52 = var_42.search(var_51)
    var_53 = '/home/user/docs_backup/file.txt'
    var_54 = var_42.search(var_53)
    var_55 = module_0.Trie()
    var_56 = './config.json'
    var_57 = 'relative'
    var_58 = {var_3: var_57}
    var_59 = var_55.insert(var_56, var_58)
    var_60 = './file.txt'
    var_61 = var_55.search(var_60)
    var_62 = module_0.Trie()
    var_63 = '/exact/path/config.json'
    var_64 = 'exact'
    var_65 = {var_3: var_64}
    var_66 = var_62.insert(var_63, var_65)
    var_67 = var_62.search(var_63)
    var_68 = module_0.Trie()
    var_69 = '/empty/config.json'
    var_70 = {}
    var_71 = var_68.insert(var_69, var_70)
    var_72 = '/empty/file.txt'
    var_73 = var_68.search(var_72)
    var_74 = module_0.Trie()
    var_75 = '/a/config.json'
    var_76 = 'level'
    var_77 = 'a'
    var_78 = {var_76: var_77}
    var_79 = var_74.insert(var_75, var_78)
    var_80 = {var_76: var_31}
    var_81 = var_74.insert(var_30, var_80)
    var_82 = {var_76: var_35}
    var_83 = var_74.insert(var_34, var_82)
    var_84 = '/a/b/c/d/config.json'
    var_85 = 'd'
    var_86 = {var_76: var_85}
    var_87 = var_74.insert(var_84, var_86)
    var_88 = '/a/b/c/d/e/file.txt'
    var_89 = var_74.search(var_88)
    var_90 = '/a/b/c/file.txt'
    var_91 = var_74.search(var_90)
    var_92 = '/a/b/file.txt'
    var_93 = var_74.search(var_92)
    var_94 = '/a/file.txt'
    var_95 = var_74.search(var_94)



# Parsed testcases at query #25
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = var_0.search(var_1)
    var_3 = '/root/config.json'
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = '/root/subdir/file.py'
    var_9 = var_7.search(var_8)
    var_10 = module_0.Trie()
    var_11 = '/project/config.json'
    var_12 = 'project'
    var_13 = 'config'
    var_14 = {var_12: var_13}
    var_15 = var_10.insert(var_11, var_14)
    var_16 = '/project/src/config.json'
    var_17 = 'src'
    var_18 = {var_17: var_13}
    var_19 = var_10.insert(var_16, var_18)
    var_20 = '/project/src/utils/file.py'
    var_21 = var_10.search(var_20)
    var_22 = module_0.Trie()
    var_23 = '/home/user/config.json'
    var_24 = 'user'
    var_25 = {var_24: var_13}
    var_26 = var_22.insert(var_23, var_25)
    var_27 = '/home/user/project/config.json'
    var_28 = {var_12: var_13}
    var_29 = var_22.insert(var_27, var_28)
    var_30 = '/home/user/project/src/deep/nested/file.py'
    var_31 = var_22.search(var_30)
    var_32 = module_0.Trie()
    var_33 = '/a/b/config.json'
    var_34 = 'b'
    var_35 = {var_34: var_13}
    var_36 = var_32.insert(var_33, var_35)
    var_37 = '/a/c/file.py'
    var_38 = var_32.search(var_37)
    var_39 = module_0.Trie()
    var_40 = '/config.json'
    var_41 = 'root'
    var_42 = {var_41: var_13}
    var_43 = var_39.insert(var_40, var_42)
    var_44 = '/usr/config.json'
    var_45 = 'usr'
    var_46 = {var_45: var_13}
    var_47 = var_39.insert(var_44, var_46)
    var_48 = '/usr/local/config.json'
    var_49 = 'local'
    var_50 = {var_49: var_13}
    var_51 = var_39.insert(var_48, var_50)
    var_52 = '/usr/local/bin/file.py'
    var_53 = var_39.search(var_52)
    var_54 = module_0.Trie()
    var_55 = '/dir/config.json'
    var_56 = 'dir'
    var_57 = {var_56: var_13}
    var_58 = var_54.insert(var_55, var_57)
    var_59 = '/dir/file.py'
    var_60 = var_54.search(var_59)
    var_61 = module_0.Trie()
    var_62 = '/real/path/config.json'
    var_63 = 'real'
    var_64 = {var_63: var_13}
    var_65 = var_61.insert(var_62, var_64)
    var_66 = '/real/path/../path/file.py'
    var_67 = var_61.search(var_66)
    var_68 = module_0.Trie()
    var_69 = 'C:/project/config.json'
    var_70 = 'windows'
    var_71 = {var_70: var_13}
    var_72 = var_68.insert(var_69, var_71)
    var_73 = 'C:/project/src/config.json'
    var_74 = {var_17: var_13}
    var_75 = var_68.insert(var_73, var_74)
    var_76 = 'C:/project/src/file.py'
    var_77 = var_68.search(var_76)
    var_78 = module_0.Trie()
    var_79 = './config.json'
    var_80 = 'relative'
    var_81 = {var_80: var_13}
    var_82 = var_78.insert(var_79, var_81)
    var_83 = './file.py'
    var_84 = var_78.search(var_83)



# Parsed testcases at query #26
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #27
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(var_1, var_5)
    var_7 = {}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = None
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = {var_3: var_4}
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #28
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = {}
    var_12 = module_0.TrieNode(var_1, var_11)
    var_13 = None
    var_14 = module_0.TrieNode(var_1, var_13)
    var_15 = {var_3: var_5}
    var_16 = module_0.TrieNode(var_1, var_15)
    var_17 = ''
    var_18 = {var_3: var_5}
    var_19 = module_0.TrieNode(var_17, var_18)



# Parsed testcases at query #29
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #30
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = '/home/user/project/config.json'
    var_8 = 'setting'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = var_6.insert(var_7, var_10)
    var_12 = 'home'
    var_13 = var_6.root.nodes[var_12]
    var_14 = 'user'
    var_15 = var_13.nodes[var_14]
    var_16 = 'project'
    var_17 = var_15.nodes[var_16]
    var_18 = module_0.Trie()
    var_19 = 'config'
    var_20 = 'first'
    var_21 = {var_19: var_20}
    var_22 = 'second'
    var_23 = {var_19: var_22}
    var_24 = '/a/b/c/config1.json'
    var_25 = var_18.insert(var_24, var_21)
    var_26 = '/a/b/c/config2.json'
    var_27 = var_18.insert(var_26, var_23)
    var_28 = 'a'
    var_29 = var_18.root.nodes[var_28]
    var_30 = 'b'
    var_31 = var_29.nodes[var_30]
    var_32 = 'c'
    var_33 = var_31.nodes[var_32]
    var_34 = module_0.Trie()
    var_35 = '/x/y/config.json'
    var_36 = 'data'
    var_37 = 'y'
    var_38 = {var_36: var_37}
    var_39 = var_34.insert(var_35, var_38)
    var_40 = '/x/config.json'
    var_41 = 'x'
    var_42 = {var_36: var_41}
    var_43 = var_34.insert(var_40, var_42)
    var_44 = var_34.root.nodes[var_41]
    var_45 = var_44.nodes[var_37]
    var_46 = 'initial.json'
    var_47 = 'initial'
    var_48 = True
    var_49 = {var_47: var_48}
    var_50 = module_0.Trie(var_46, var_49)
    var_51 = '/new/config.json'
    var_52 = 'new'
    var_53 = {var_52: var_48}
    var_54 = var_50.insert(var_51, var_53)
    var_55 = module_0.Trie()
    var_56 = '/home/../project/config.json'
    var_57 = 'resolved'
    var_58 = {var_57: var_48}
    var_59 = var_55.insert(var_56, var_58)
    var_60 = var_55.root.nodes[var_16]



# Parsed testcases at query #31
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = {}
    var_8 = module_0.Trie(var_1, var_7)
    var_9 = var_0.root
    var_10 = var_5.root



# Parsed testcases at query #32
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)
    var_13 = {var_3: var_5}
    var_14 = module_0.TrieNode(var_1, var_13)



# Parsed testcases at query #33
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'config.yaml'
    var_9 = module_0.TrieNode(var_8, var_7)
    var_10 = {}
    var_11 = module_0.TrieNode(var_1, var_10)
    var_12 = None
    var_13 = module_0.TrieNode(var_1, var_12)



# Parsed testcases at query #34
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = ''
    var_8 = module_0.Trie(var_7, var_4)
    var_9 = var_0.root
    var_10 = var_5.root



# Parsed testcases at query #35
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #36
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'root_value'
    var_5 = {var_3: var_4}
    var_6 = '/root/config.json'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = 'nested_value'
    var_11 = {var_3: var_10}
    var_12 = '/a/b/c/config.json'
    var_13 = var_9.insert(var_12, var_11)
    var_14 = '/a/b/c/file.txt'
    var_15 = var_9.search(var_14)
    var_16 = module_0.Trie()
    var_17 = 'parent_value'
    var_18 = {var_3: var_17}
    var_19 = 'child_value'
    var_20 = {var_3: var_19}
    var_21 = '/a/b/config.json'
    var_22 = var_16.insert(var_21, var_18)
    var_23 = var_16.insert(var_12, var_20)
    var_24 = '/a/b/c/d/file.txt'
    var_25 = var_16.search(var_24)
    var_26 = module_0.Trie()
    var_27 = {var_3: var_17}
    var_28 = var_26.insert(var_21, var_27)
    var_29 = var_26.search(var_24)
    var_30 = {var_3: var_4}
    var_31 = module_0.Trie(var_6, var_30)
    var_32 = '/different/path/file.txt'
    var_33 = var_31.search(var_32)
    var_34 = module_0.Trie()
    var_35 = 'level1'
    var_36 = {var_3: var_35}
    var_37 = 'level2'
    var_38 = {var_3: var_37}
    var_39 = 'level3'
    var_40 = {var_3: var_39}
    var_41 = '/a/config.json'
    var_42 = var_34.insert(var_41, var_36)
    var_43 = var_34.insert(var_21, var_38)
    var_44 = var_34.insert(var_12, var_40)
    var_45 = '/a/b/c/d/e/file.txt'
    var_46 = var_34.search(var_45)
    var_47 = '/a/b/x/file.txt'
    var_48 = var_34.search(var_47)
    var_49 = '/a/x/file.txt'
    var_50 = var_34.search(var_49)
    var_51 = module_0.Trie()
    var_52 = 'resolved_value'
    var_53 = {var_3: var_52}
    var_54 = './config.json'
    var_55 = var_51.insert(var_54, var_53)
    var_56 = './file.txt'
    var_57 = var_51.search(var_56)
    var_58 = module_0.Trie()
    var_59 = 'empty_path_value'
    var_60 = {var_3: var_59}
    var_61 = ''
    var_62 = var_58.insert(var_61, var_60)
    var_63 = var_58.search(var_61)
    var_64 = module_0.Trie()
    var_65 = 'parent'
    var_66 = {var_3: var_65}
    var_67 = var_64.insert(var_21, var_66)
    var_68 = '/a/b/c'
    var_69 = var_64.search(var_68)



# Parsed testcases at query #37
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(var_1, var_5)
    var_7 = {}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = None
    var_10 = module_0.TrieNode(var_1, var_9)



# Parsed testcases at query #38
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'config1'
    var_8 = 'data1'
    var_9 = {var_7: var_8}
    var_10 = 'config2'
    var_11 = 'data2'
    var_12 = {var_10: var_11}
    var_13 = '/a/b/config1.json'
    var_14 = var_6.insert(var_13, var_9)
    var_15 = '/a/b/c/config2.json'
    var_16 = var_6.insert(var_15, var_12)
    var_17 = var_6.root
    var_18 = var_6.root
    var_19 = module_0.Trie()
    var_20 = 'old'
    var_21 = 'data'
    var_22 = {var_20: var_21}
    var_23 = 'new'
    var_24 = {var_23: var_21}
    var_25 = '/path/config.json'
    var_26 = var_19.insert(var_25, var_22)
    var_27 = var_19.insert(var_25, var_24)
    var_28 = var_19.root
    var_29 = module_0.Trie()
    var_30 = '/empty/config.json'
    var_31 = {}
    var_32 = var_29.insert(var_30, var_31)
    var_33 = var_29.root
    var_34 = module_0.Trie()
    var_35 = 'type'
    var_36 = 'user'
    var_37 = {var_35: var_36}
    var_38 = 'system'
    var_39 = {var_35: var_38}
    var_40 = '/common/base/user.json'
    var_41 = var_34.insert(var_40, var_37)
    var_42 = '/common/base/system.json'
    var_43 = var_34.insert(var_42, var_39)
    var_44 = var_34.root
    var_45 = var_34.root
    var_46 = module_0.Trie()
    var_47 = 'relative'
    var_48 = 'path'
    var_49 = {var_47: var_48}
    var_50 = './config.json'
    var_51 = var_46.insert(var_50, var_49)
    var_52 = var_46.root



# Parsed testcases at query #39
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #40
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = '/root/config1.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = '/root/dir1/dir2/config2.json'
    var_11 = var_6.insert(var_10, var_9)
    var_12 = 'root'
    var_13 = var_6.root.nodes[var_12]
    var_14 = 'dir1'
    var_15 = var_13.nodes[var_14]
    var_16 = 'dir2'
    var_17 = var_15.nodes[var_16]
    var_18 = module_0.Trie()
    var_19 = 'key3'
    var_20 = 'value3'
    var_21 = {var_19: var_20}
    var_22 = 'key4'
    var_23 = 'value4'
    var_24 = {var_22: var_23}
    var_25 = '/a/b/config3.json'
    var_26 = var_18.insert(var_25, var_21)
    var_27 = '/a/b/c/config4.json'
    var_28 = var_18.insert(var_27, var_24)
    var_29 = 'a'
    var_30 = var_18.root.nodes[var_29]
    var_31 = 'b'
    var_32 = var_30.nodes[var_31]
    var_33 = 'c'
    var_34 = var_32.nodes[var_33]
    var_35 = module_0.Trie()
    var_36 = 'key5'
    var_37 = 'value5'
    var_38 = {var_36: var_37}
    var_39 = 'key6'
    var_40 = 'value6'
    var_41 = {var_39: var_40}
    var_42 = '/x/y/config5.json'
    var_43 = var_35.insert(var_42, var_38)
    var_44 = var_35.insert(var_42, var_41)
    var_45 = module_0.Trie()
    var_46 = 'key7'
    var_47 = 'value7'
    var_48 = {var_46: var_47}
    var_49 = '/test/path/to/config'
    var_50 = '/../config7.json'
    var_51 = var_49 + var_50
    var_52 = var_45.insert(var_51, var_48)
    var_53 = 'test'
    var_54 = var_45.root.nodes[var_53]
    var_55 = 'path'
    var_56 = var_54.nodes[var_55]
    var_57 = module_0.Trie()
    var_58 = '/empty/config.json'
    var_59 = {}
    var_60 = var_57.insert(var_58, var_59)
    var_61 = module_0.Trie()
    var_62 = 'key8'
    var_63 = 'value8'
    var_64 = {var_62: var_63}
    var_65 = '/deep/nested/config8.json'
    var_66 = var_61.insert(var_65, var_64)



# Parsed testcases at query #41
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)
    var_13 = {var_3: var_5}
    var_14 = module_0.TrieNode(var_1, var_13)



# Parsed testcases at query #42
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'root_value'
    var_5 = {var_3: var_4}
    var_6 = '/root/config.json'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/project/config.json'
    var_11 = 'env'
    var_12 = 'project'
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/project/src/config.json'
    var_16 = 'src'
    var_17 = {var_11: var_16}
    var_18 = var_9.insert(var_15, var_17)
    var_19 = '/project/src/utils/config.json'
    var_20 = 'utils'
    var_21 = {var_11: var_20}
    var_22 = var_9.insert(var_19, var_21)
    var_23 = '/project/main.py'
    var_24 = var_9.search(var_23)
    var_25 = '/project/src/module.py'
    var_26 = var_9.search(var_25)
    var_27 = '/project/src/utils/helper.py'
    var_28 = var_9.search(var_27)
    var_29 = '/project/src/utils/subdir/file.py'
    var_30 = var_9.search(var_29)
    var_31 = '/project/tests/config.json'
    var_32 = 'tests'
    var_33 = {var_11: var_32}
    var_34 = var_9.insert(var_31, var_33)
    var_35 = '/project/tests/test_file.py'
    var_36 = var_9.search(var_35)
    var_37 = './project/src/module.py'
    var_38 = var_9.search(var_37)
    var_39 = '/other/file.py'
    var_40 = var_9.search(var_39)
    var_41 = module_0.Trie()
    var_42 = '/a/b/config.json'
    var_43 = 'level'
    var_44 = 'b'
    var_45 = {var_43: var_44}
    var_46 = var_41.insert(var_42, var_45)
    var_47 = '/a/b/c/d/config.json'
    var_48 = 'd'
    var_49 = {var_43: var_48}
    var_50 = var_41.insert(var_47, var_49)
    var_51 = '/a/b/c/file.py'
    var_52 = var_41.search(var_51)
    var_53 = module_0.Trie()
    var_54 = '/x/y/config1.json'
    var_55 = 'name'
    var_56 = 'config1'
    var_57 = {var_55: var_56}
    var_58 = var_53.insert(var_54, var_57)
    var_59 = '/x/y/config2.json'
    var_60 = 'config2'
    var_61 = {var_55: var_60}
    var_62 = var_53.insert(var_59, var_61)
    var_63 = '/x/y/z/file.py'
    var_64 = var_53.search(var_63)
    var_65 = ''
    var_66 = var_53.search(var_65)



# Parsed testcases at query #43
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = {}
    var_8 = module_0.Trie(var_1, var_7)
    var_9 = ''
    var_10 = module_0.Trie(var_9, var_4)



# Parsed testcases at query #44
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'settings'
    var_8 = 'debug'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = '/home/user/project/config.yaml'
    var_13 = var_6.insert(var_12, var_11)
    var_14 = var_6.root
    var_15 = module_0.Trie()
    var_16 = 'name'
    var_17 = 'config1'
    var_18 = {var_16: var_17}
    var_19 = 'config2'
    var_20 = {var_16: var_19}
    var_21 = '/a/b/c/config1.json'
    var_22 = var_15.insert(var_21, var_18)
    var_23 = '/a/b/config2.json'
    var_24 = var_15.insert(var_23, var_20)
    var_25 = var_15.root
    var_26 = var_15.root
    var_27 = module_0.Trie()
    var_28 = 'version'
    var_29 = {var_28: var_9}
    var_30 = 2
    var_31 = {var_28: var_30}
    var_32 = '/path/config.json'
    var_33 = var_27.insert(var_32, var_29)
    var_34 = var_27.insert(var_32, var_31)
    var_35 = var_27.root
    var_36 = 'initial.json'
    var_37 = 'initial'
    var_38 = {var_37: var_9}
    var_39 = module_0.Trie(var_36, var_38)
    var_40 = 'new'
    var_41 = 'config'
    var_42 = {var_40: var_41}
    var_43 = '/new/path/config.json'
    var_44 = var_39.insert(var_43, var_42)
    var_45 = var_39.root
    var_46 = module_0.Trie()
    var_47 = 'test'
    var_48 = 'relative'
    var_49 = {var_47: var_48}
    var_50 = './relative/path/config.json'
    var_51 = var_46.insert(var_50, var_49)
    var_52 = './relative/path'
    var_53 = var_46.root



# Parsed testcases at query #45
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'settings'
    var_8 = 'debug'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = '/home/user/project/config.yaml'
    var_13 = var_6.insert(var_12, var_11)
    var_14 = 'home'
    var_15 = var_6.root.nodes[var_14]
    var_16 = 'user'
    var_17 = var_15.nodes[var_16]
    var_18 = 'project'
    var_19 = var_17.nodes[var_18]
    var_20 = module_0.Trie()
    var_21 = 'a'
    var_22 = {var_21: var_9}
    var_23 = 'b'
    var_24 = 2
    var_25 = {var_23: var_24}
    var_26 = '/a/b/c/config1.json'
    var_27 = var_20.insert(var_26, var_22)
    var_28 = '/a/b/config2.json'
    var_29 = var_20.insert(var_28, var_25)
    var_30 = var_20.root.nodes[var_21]
    var_31 = var_30.nodes[var_23]
    var_32 = 'c'
    var_33 = var_31.nodes[var_32]
    var_34 = module_0.Trie()
    var_35 = 'version'
    var_36 = {var_35: var_9}
    var_37 = {var_35: var_24}
    var_38 = '/path/config.json'
    var_39 = var_34.insert(var_38, var_36)
    var_40 = var_34.insert(var_38, var_37)
    var_41 = module_0.Trie()
    var_42 = '/empty/config.json'
    var_43 = {}
    var_44 = var_41.insert(var_42, var_43)
    var_45 = module_0.Trie()
    var_46 = 'root'
    var_47 = {var_46: var_9}
    var_48 = 'config.json'
    var_49 = var_45.insert(var_48, var_47)



# Parsed testcases at query #46
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = {}
    var_8 = module_0.Trie(var_1, var_7)
    var_9 = var_0.root
    var_10 = var_5.root



# Parsed testcases at query #47
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = '/root/config.json'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/home/user/config.json'
    var_11 = 'theme'
    var_12 = 'dark'
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/home/user/file.txt'
    var_16 = var_9.search(var_15)
    var_17 = module_0.Trie()
    var_18 = '/home/user/project/config.json'
    var_19 = 'lint'
    var_20 = True
    var_21 = {var_19: var_20}
    var_22 = var_17.insert(var_18, var_21)
    var_23 = '/home/user/project/src/main.py'
    var_24 = var_17.search(var_23)
    var_25 = module_0.Trie()
    var_26 = {var_19: var_20}
    var_27 = var_25.insert(var_18, var_26)
    var_28 = '/home/user/project/src/utils/helper.py'
    var_29 = var_25.search(var_28)
    var_30 = module_0.Trie()
    var_31 = 'global'
    var_32 = {var_31: var_20}
    var_33 = var_30.insert(var_10, var_32)
    var_34 = 'project'
    var_35 = {var_34: var_20}
    var_36 = var_30.insert(var_18, var_35)
    var_37 = '/home/user/project/src/file.py'
    var_38 = var_30.search(var_37)
    var_39 = module_0.Trie()
    var_40 = {var_31: var_20}
    var_41 = var_39.insert(var_10, var_40)
    var_42 = {var_34: var_20}
    var_43 = var_39.insert(var_18, var_42)
    var_44 = '/home/user/other/file.txt'
    var_45 = var_39.search(var_44)
    var_46 = module_0.Trie()
    var_47 = {var_34: var_20}
    var_48 = var_46.insert(var_18, var_47)
    var_49 = '/var/log/file.txt'
    var_50 = var_46.search(var_49)
    var_51 = 'root'
    var_52 = {var_51: var_20}
    var_53 = module_0.Trie(var_3, var_52)
    var_54 = '/root/home/user/config.json'
    var_55 = 'user'
    var_56 = {var_55: var_20}
    var_57 = var_53.insert(var_54, var_56)
    var_58 = '/root/file.txt'
    var_59 = var_53.search(var_58)
    var_60 = module_0.Trie()
    var_61 = {var_34: var_20}
    var_62 = var_60.insert(var_18, var_61)
    var_63 = '/home/user/project/subdir/another/file.py'
    var_64 = var_60.search(var_63)
    var_65 = module_0.Trie()
    var_66 = 'C:\\Users\\Project\\config.json'
    var_67 = 'windows'
    var_68 = {var_67: var_20}
    var_69 = var_65.insert(var_66, var_68)
    var_70 = 'C:\\Users\\Project\\src\\file.py'
    var_71 = var_65.search(var_70)
    var_72 = module_0.Trie()
    var_73 = './config.json'
    var_74 = 'relative'
    var_75 = {var_74: var_20}
    var_76 = var_72.insert(var_73, var_75)
    var_77 = './src/file.py'
    var_78 = var_72.search(var_77)
    var_79 = module_0.Trie()
    var_80 = '/config.json'
    var_81 = {}
    var_82 = var_79.insert(var_80, var_81)
    var_83 = '/file.txt'
    var_84 = var_79.search(var_83)
    var_85 = module_0.Trie()
    var_86 = '/a/b/config.json'
    var_87 = 'level'
    var_88 = 'b'
    var_89 = {var_87: var_88}
    var_90 = var_85.insert(var_86, var_89)
    var_91 = '/a/b/c/config.json'
    var_92 = 'c'
    var_93 = {var_87: var_92}
    var_94 = var_85.insert(var_91, var_93)
    var_95 = '/a/b/c/d/config.json'
    var_96 = 'd'
    var_97 = {var_87: var_96}
    var_98 = var_85.insert(var_95, var_97)
    var_99 = '/a/b/file.txt'
    var_100 = var_85.search(var_99)
    var_101 = '/a/b/c/file.txt'
    var_102 = var_85.search(var_101)
    var_103 = '/a/b/c/d/file.txt'
    var_104 = var_85.search(var_103)
    var_105 = '/a/b/c/d/e/file.txt'
    var_106 = var_85.search(var_105)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = '/root/subdir/config.json'
    var_11 = var_6.insert(var_10, var_9)
    var_12 = 'root'
    var_13 = var_6.root.nodes[var_12]
    var_14 = 'subdir'
    var_15 = var_13.nodes[var_14]
    var_16 = module_0.Trie()
    var_17 = 'key3'
    var_18 = 'value3'
    var_19 = {var_17: var_18}
    var_20 = 'key4'
    var_21 = 'value4'
    var_22 = {var_20: var_21}
    var_23 = '/a/b/config1.json'
    var_24 = var_16.insert(var_23, var_19)
    var_25 = '/a/b/c/config2.json'
    var_26 = var_16.insert(var_25, var_22)
    var_27 = 'a'
    var_28 = var_16.root.nodes[var_27]
    var_29 = 'b'
    var_30 = var_28.nodes[var_29]
    var_31 = 'c'
    var_32 = var_30.nodes[var_31]
    var_33 = module_0.Trie()
    var_34 = 'key5'
    var_35 = 'value5'
    var_36 = {var_34: var_35}
    var_37 = 'key6'
    var_38 = 'value6'
    var_39 = {var_37: var_38}
    var_40 = '/x/y/configA.json'
    var_41 = var_33.insert(var_40, var_36)
    var_42 = '/x/y/configB.json'
    var_43 = var_33.insert(var_42, var_39)
    var_44 = 'x'
    var_45 = var_33.root.nodes[var_44]
    var_46 = 'y'
    var_47 = var_45.nodes[var_46]
    var_48 = module_0.Trie()
    var_49 = 'key7'
    var_50 = 'value7'
    var_51 = {var_49: var_50}
    var_52 = 'sub'
    var_53 = 'config.json'
    var_54 = True
    var_55 = var_11.parts
    var_56 = -1
    var_57 = var_55[var_56]
    var_58 = ''
    var_59 = var_57 if var_55 else var_58
    var_60 = var_48.root
    var_61 = module_0.Trie()
    var_62 = '/empty/config.json'
    var_63 = {}
    var_64 = var_61.insert(var_62, var_63)
    var_65 = 'empty'
    var_66 = var_61.root.nodes[var_65]
    var_67 = module_0.Trie()
    var_68 = 'key8'
    var_69 = 'value8'
    var_70 = {var_68: var_69}
    var_71 = 'key9'
    var_72 = 'value9'
    var_73 = {var_71: var_72}
    var_74 = '/same/path/config.json'
    var_75 = var_67.insert(var_74, var_70)
    var_76 = var_67.insert(var_74, var_73)
    var_77 = 'same'
    var_78 = var_67.root.nodes[var_77]
    var_79 = 'path'
    var_80 = var_78.nodes[var_79]



# Parsed testcases at query #2
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = ''
    var_8 = module_0.Trie(var_7, var_4)
    var_9 = var_0.root
    var_10 = var_5.root



# Parsed testcases at query #3
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'settings'
    var_8 = 'debug'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = '/home/user/project/config.yaml'
    var_13 = var_6.insert(var_12, var_11)
    var_14 = var_6.root
    var_15 = module_0.Trie()
    var_16 = 'name'
    var_17 = 'config1'
    var_18 = {var_16: var_17}
    var_19 = 'config2'
    var_20 = {var_16: var_19}
    var_21 = '/a/b/c/config1.json'
    var_22 = var_15.insert(var_21, var_18)
    var_23 = '/a/b/config2.json'
    var_24 = var_15.insert(var_23, var_20)
    var_25 = var_15.root
    var_26 = var_15.root
    var_27 = module_0.Trie()
    var_28 = 'version'
    var_29 = {var_28: var_9}
    var_30 = 2
    var_31 = {var_28: var_30}
    var_32 = '/path/config.json'
    var_33 = var_27.insert(var_32, var_29)
    var_34 = var_27.insert(var_32, var_31)
    var_35 = var_27.root
    var_36 = module_0.Trie()
    var_37 = 'test'
    var_38 = 'relative'
    var_39 = {var_37: var_38}
    var_40 = './config.json'
    var_41 = var_36.insert(var_40, var_39)
    var_42 = var_36.root
    var_43 = module_0.Trie()
    var_44 = {}
    var_45 = '/empty/config.json'
    var_46 = var_43.insert(var_45, var_44)
    var_47 = var_43.root



# Parsed testcases at query #4
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
    var_7 = module_0.Trie(var_2)
    var_8 = {}
    var_9 = module_0.Trie(var_2, var_8)
    var_10 = ''
    var_11 = module_0.Trie(var_10, var_5)



# Parsed testcases at query #5
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = {}
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #6
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'key1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = 'key2'
    var_11 = 'value2'
    var_12 = {var_10: var_11}
    var_13 = '/a/b/config1.json'
    var_14 = var_6.insert(var_13, var_9)
    var_15 = '/a/b/c/config2.json'
    var_16 = var_6.insert(var_15, var_12)
    var_17 = var_6.root
    var_18 = var_6.root
    var_19 = module_0.Trie()
    var_20 = 'old'
    var_21 = 'data'
    var_22 = {var_20: var_21}
    var_23 = 'new'
    var_24 = {var_23: var_21}
    var_25 = '/path/config.json'
    var_26 = var_19.insert(var_25, var_22)
    var_27 = var_19.insert(var_25, var_24)
    var_28 = var_19.root
    var_29 = module_0.Trie()
    var_30 = '/empty/config.json'
    var_31 = {}
    var_32 = var_29.insert(var_30, var_31)
    var_33 = var_29.root
    var_34 = module_0.Trie()
    var_35 = '/common/a/config1.json'
    var_36 = 'id'
    var_37 = 1
    var_38 = {var_36: var_37}
    var_39 = (var_35, var_38)
    var_40 = '/common/b/config2.json'
    var_41 = 2
    var_42 = {var_36: var_41}
    var_43 = (var_40, var_42)
    var_44 = '/common/a/sub/config3.json'
    var_45 = 3
    var_46 = {var_36: var_45}
    var_47 = (var_44, var_46)
    var_48 = [var_39, var_43, var_47]
    var_49 = var_34.root
    var_50 = 'root_config.json'
    var_51 = 'root'
    var_52 = True
    var_53 = {var_51: var_52}
    var_54 = module_0.Trie(var_50, var_53)
    var_55 = '/sub/config.json'
    var_56 = 'sub'
    var_57 = True
    var_58 = {var_56: var_57}
    var_59 = var_54.insert(var_55, var_58)
    var_60 = var_54.root



# Parsed testcases at query #7
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'config.yaml'
    var_9 = module_0.TrieNode(var_8, var_7)
    var_10 = 'config.txt'
    var_11 = {}
    var_12 = module_0.TrieNode(var_10, var_11)
    var_13 = module_0.TrieNode()
    var_14 = module_0.TrieNode()



# Parsed testcases at query #8
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #9
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(var_1, var_5)
    var_7 = {}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = None
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = module_0.TrieNode()
    var_12 = module_0.TrieNode()



# Parsed testcases at query #10
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'key1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = 'key2'
    var_11 = 'value2'
    var_12 = {var_10: var_11}
    var_13 = '/a/b/config1.json'
    var_14 = var_6.insert(var_13, var_9)
    var_15 = '/a/b/c/config2.json'
    var_16 = var_6.insert(var_15, var_12)
    var_17 = var_6.root
    var_18 = var_6.root
    var_19 = module_0.Trie()
    var_20 = 'old'
    var_21 = 'data'
    var_22 = {var_20: var_21}
    var_23 = 'new'
    var_24 = {var_23: var_21}
    var_25 = '/same/path/config.json'
    var_26 = var_19.insert(var_25, var_22)
    var_27 = var_19.insert(var_25, var_24)
    var_28 = var_19.root
    var_29 = module_0.Trie()
    var_30 = '/empty/config.json'
    var_31 = {}
    var_32 = var_29.insert(var_30, var_31)
    var_33 = var_29.root
    var_34 = module_0.Trie()
    var_35 = 'config'
    var_36 = '1'
    var_37 = {var_35: var_36}
    var_38 = '2'
    var_39 = {var_35: var_38}
    var_40 = '3'
    var_41 = {var_35: var_40}
    var_42 = '/common/a/config1.json'
    var_43 = var_34.insert(var_42, var_37)
    var_44 = '/common/b/config2.json'
    var_45 = var_34.insert(var_44, var_39)
    var_46 = '/common/a/sub/config3.json'
    var_47 = var_34.insert(var_46, var_41)
    var_48 = var_34.root
    var_49 = var_34.root
    var_50 = var_34.root
    var_51 = module_0.Trie()
    var_52 = 'rel'
    var_53 = 'path'
    var_54 = {var_52: var_53}
    var_55 = './relative/config.json'
    var_56 = var_51.insert(var_55, var_54)
    var_57 = var_51.root
    var_58 = module_0.Trie()
    var_59 = 'root'
    var_60 = {var_59: var_35}
    var_61 = '/root.json'
    var_62 = 'nested'
    var_63 = {var_62: var_35}
    var_64 = '/a/b/c/config.json'
    var_65 = var_58.insert(var_64, var_63)
    var_66 = var_58.root
    var_67 = ''
    var_68 = {}
    var_69 = module_0.Trie(var_67, var_68)
    var_70 = 'test'
    var_71 = {var_70: var_21}
    var_72 = '/test/config.json'
    var_73 = var_69.insert(var_72, var_71)
    var_74 = var_69.root



# Parsed testcases at query #11
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = '/root/config.json'
    var_3 = 'key'
    var_4 = 'root_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.Trie(var_2, var_5)
    var_7 = module_0.Trie()
    var_8 = '/home/user/project/config.json'
    var_9 = 'project_value'
    var_10 = {var_3: var_9}
    var_11 = var_7.insert(var_8, var_10)
    var_12 = '/home/user/project/file.py'
    var_13 = module_0.Trie()
    var_14 = {var_3: var_9}
    var_15 = var_13.insert(var_8, var_14)
    var_16 = '/home/user/project/subdir/file.py'
    var_17 = module_0.Trie()
    var_18 = '/home/user/config.json'
    var_19 = 'user_value'
    var_20 = {var_3: var_19}
    var_21 = var_17.insert(var_18, var_20)
    var_22 = {var_3: var_9}
    var_23 = var_17.insert(var_8, var_22)
    var_24 = '/home/user/project/src/config.json'
    var_25 = 'src_value'
    var_26 = {var_3: var_25}
    var_27 = var_17.insert(var_24, var_26)
    var_28 = '/home/user/project/src/file.py'
    var_29 = '/home/user/file.py'
    var_30 = '/config.json'
    var_31 = {var_3: var_4}
    var_32 = module_0.Trie(var_30, var_31)
    var_33 = '/a/b/c/d/e/f/file.py'
    var_34 = module_0.Trie()
    var_35 = {var_3: var_19}
    var_36 = var_34.insert(var_18, var_35)
    var_37 = '/different/path/file.py'
    var_38 = 'value'
    var_39 = {var_3: var_38}
    var_40 = module_0.Trie(var_30, var_39)
    var_41 = ''
    var_42 = module_0.Trie()
    var_43 = {var_3: var_9}
    var_44 = var_42.insert(var_8, var_43)
    var_45 = '/home/user/project'
    var_46 = module_0.Trie()
    var_47 = 'old_value'
    var_48 = {var_3: var_47}
    var_49 = var_46.insert(var_18, var_48)
    var_50 = 'extra'
    var_51 = 'new_value'
    var_52 = 'data'
    var_53 = {var_3: var_51, var_50: var_52}
    var_54 = var_46.insert(var_18, var_53)



# Parsed testcases at query #12
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)
    var_13 = {var_3: var_5}
    var_14 = 'test.json'
    var_15 = module_0.TrieNode(var_14, var_13)



# Parsed testcases at query #13
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = None
    var_7 = module_0.Trie(var_1, var_6)
    var_8 = ''
    var_9 = module_0.Trie(var_8, var_4)



# Parsed testcases at query #14
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = {}
    var_12 = module_0.TrieNode(var_1, var_11)
    var_13 = module_0.TrieNode()
    var_14 = module_0.TrieNode()



# Parsed testcases at query #15
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #16
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = ''
    var_8 = module_0.Trie(var_7, var_4)
    var_9 = var_0.root
    var_10 = var_5.root



# Parsed testcases at query #17
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = '/root/config.json'
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/home/user/config.json'
    var_11 = 'theme'
    var_12 = 'dark'
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/home/user/file.txt'
    var_16 = var_9.search(var_15)
    var_17 = module_0.Trie()
    var_18 = '/home/user/project/config.json'
    var_19 = 'version'
    var_20 = '1.0'
    var_21 = {var_19: var_20}
    var_22 = var_17.insert(var_18, var_21)
    var_23 = '/home/user/project/src/main.py'
    var_24 = var_17.search(var_23)
    var_25 = module_0.Trie()
    var_26 = 'user'
    var_27 = 'test'
    var_28 = {var_26: var_27}
    var_29 = var_25.insert(var_10, var_28)
    var_30 = 'project'
    var_31 = {var_30: var_27}
    var_32 = var_25.insert(var_18, var_31)
    var_33 = '/home/user/project/subdir/file.txt'
    var_34 = var_25.search(var_33)
    var_35 = module_0.Trie()
    var_36 = '/home/user/docs/config.json'
    var_37 = 'docs'
    var_38 = 'true'
    var_39 = {var_37: var_38}
    var_40 = var_35.insert(var_36, var_39)
    var_41 = '/home/user/src/main.py'
    var_42 = var_35.search(var_41)
    var_43 = '/global/config.json'
    var_44 = 'global'
    var_45 = {var_44: var_38}
    var_46 = module_0.Trie(var_43, var_45)
    var_47 = 'john'
    var_48 = {var_26: var_47}
    var_49 = var_46.insert(var_10, var_48)
    var_50 = 'myproject'
    var_51 = {var_30: var_50}
    var_52 = var_46.insert(var_18, var_51)
    var_53 = '/home/user/project/src/file.py'
    var_54 = var_46.search(var_53)
    var_55 = '/home/user/other/file.py'
    var_56 = var_46.search(var_55)
    var_57 = '/other/location/file.py'
    var_58 = var_46.search(var_57)
    var_59 = module_0.Trie()
    var_60 = 'data'
    var_61 = {var_27: var_60}
    var_62 = var_59.insert(var_18, var_61)
    var_63 = '/home/user/project/../project/file.txt'
    var_64 = var_59.search(var_63)
    var_65 = '/default/config.json'
    var_66 = 'default'
    var_67 = 'config'
    var_68 = {var_66: var_67}
    var_69 = module_0.Trie(var_65, var_68)
    var_70 = ''
    var_71 = var_69.search(var_70)
    var_72 = module_0.Trie()
    var_73 = '/config.json'
    var_74 = 'root'
    var_75 = {var_74: var_38}
    var_76 = var_72.insert(var_73, var_75)
    var_77 = var_72.search(var_73)



# Parsed testcases at query #18
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = {}
    var_8 = module_0.Trie(var_1, var_7)
    var_9 = var_0.root
    var_10 = var_5.root



# Parsed testcases at query #19
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #20
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = '/root/config.json'
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.Trie(var_3, var_6)
    var_8 = '/root/subdir/file.txt'
    var_9 = var_7.search(var_8)
    var_10 = module_0.Trie()
    var_11 = '/home/user/project/config.json'
    var_12 = 'project'
    var_13 = 'config'
    var_14 = {var_12: var_13}
    var_15 = var_10.insert(var_11, var_14)
    var_16 = '/home/user/config.json'
    var_17 = 'user'
    var_18 = {var_17: var_13}
    var_19 = var_10.insert(var_16, var_18)
    var_20 = '/home/user/project/src/file.py'
    var_21 = var_10.search(var_20)
    var_22 = module_0.Trie()
    var_23 = {var_12: var_13}
    var_24 = var_22.insert(var_11, var_23)
    var_25 = '/home/user/project/file.py'
    var_26 = var_22.search(var_25)
    var_27 = module_0.Trie()
    var_28 = {var_17: var_13}
    var_29 = var_27.insert(var_16, var_28)
    var_30 = '/home/user/project/src/config.json'
    var_31 = 'src'
    var_32 = {var_31: var_13}
    var_33 = var_27.insert(var_30, var_32)
    var_34 = '/home/user/project/src/module/file.py'
    var_35 = var_27.search(var_34)
    var_36 = module_0.Trie()
    var_37 = '/absolute/path/config.json'
    var_38 = 'abs'
    var_39 = {var_38: var_13}
    var_40 = var_36.insert(var_37, var_39)
    var_41 = './relative/file.txt'
    var_42 = var_36.search(var_41)
    var_43 = module_0.Trie()
    var_44 = '/a/config.json'
    var_45 = 'level'
    var_46 = 'a'
    var_47 = {var_45: var_46}
    var_48 = var_43.insert(var_44, var_47)
    var_49 = '/a/b/config.json'
    var_50 = 'b'
    var_51 = {var_45: var_50}
    var_52 = var_43.insert(var_49, var_51)
    var_53 = '/a/b/c/config.json'
    var_54 = 'c'
    var_55 = {var_45: var_54}
    var_56 = var_43.insert(var_53, var_55)
    var_57 = '/a/b/c/d/e/file.txt'
    var_58 = var_43.search(var_57)
    var_59 = module_0.Trie()
    var_60 = {var_45: var_46}
    var_61 = var_59.insert(var_44, var_60)
    var_62 = {var_45: var_54}
    var_63 = var_59.insert(var_53, var_62)
    var_64 = '/a/b/d/file.txt'
    var_65 = var_59.search(var_64)
    var_66 = module_0.Trie()
    var_67 = '/config.json'
    var_68 = 'root'
    var_69 = {var_68: var_13}
    var_70 = var_66.insert(var_67, var_69)
    var_71 = '/file.txt'
    var_72 = var_66.search(var_71)
    var_73 = module_0.Trie()
    var_74 = '/path/config.json'
    var_75 = 'version'
    var_76 = '1'
    var_77 = {var_75: var_76}
    var_78 = var_73.insert(var_74, var_77)
    var_79 = '2'
    var_80 = {var_75: var_79}
    var_81 = var_73.insert(var_74, var_80)
    var_82 = '/path/sub/file.txt'
    var_83 = var_73.search(var_82)



# Parsed testcases at query #21
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = ''
    var_8 = {}
    var_9 = module_0.Trie(var_7, var_8)
    var_10 = None
    var_11 = module_0.Trie(var_1, var_10)



# Parsed testcases at query #22
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = '/home/user/project/config.json'
    var_8 = 'setting'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = var_6.insert(var_7, var_10)
    var_12 = 'home'
    var_13 = var_6.root.nodes[var_12]
    var_14 = 'user'
    var_15 = var_13.nodes[var_14]
    var_16 = 'project'
    var_17 = var_15.nodes[var_16]
    var_18 = module_0.Trie()
    var_19 = '/a/b/config1.json'
    var_20 = 'config'
    var_21 = '1'
    var_22 = {var_20: var_21}
    var_23 = var_18.insert(var_19, var_22)
    var_24 = '/a/b/c/config2.json'
    var_25 = '2'
    var_26 = {var_20: var_25}
    var_27 = var_18.insert(var_24, var_26)
    var_28 = 'a'
    var_29 = var_18.root.nodes[var_28]
    var_30 = 'b'
    var_31 = var_29.nodes[var_30]
    var_32 = 'c'
    var_33 = var_31.nodes[var_32]
    var_34 = 'initial.json'
    var_35 = 'initial'
    var_36 = True
    var_37 = {var_35: var_36}
    var_38 = module_0.Trie(var_34, var_37)
    var_39 = '/new/config.json'
    var_40 = 'new'
    var_41 = {var_40: var_36}
    var_42 = var_38.insert(var_39, var_41)
    var_43 = module_0.Trie()
    var_44 = '/same/path/config.json'
    var_45 = 'version'
    var_46 = {var_45: var_21}
    var_47 = var_43.insert(var_44, var_46)
    var_48 = {var_45: var_25}
    var_49 = var_43.insert(var_44, var_48)
    var_50 = module_0.Trie()
    var_51 = './relative/config.json'
    var_52 = 'rel'
    var_53 = 'data'
    var_54 = {var_52: var_53}
    var_55 = var_50.insert(var_51, var_54)
    var_56 = module_0.Trie()
    var_57 = '/deep/nested/config.json'
    var_58 = 'deep'
    var_59 = {var_58: var_36}
    var_60 = var_56.insert(var_57, var_59)
    var_61 = var_56.root.nodes[var_58]
    var_62 = 'nested'
    var_63 = var_61.nodes[var_62]



# Parsed testcases at query #23
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = {}
    var_12 = module_0.TrieNode(var_1, var_11)
    var_13 = {var_3: var_5}
    var_14 = module_0.TrieNode(var_1, var_13)



# Parsed testcases at query #24
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = {}
    var_8 = module_0.Trie(var_1, var_7)
    var_9 = var_0.root
    var_10 = var_5.root



# Parsed testcases at query #25
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = module_0.Trie()
    var_4 = 'key'
    var_5 = 'root_value'
    var_6 = {var_4: var_5}
    var_7 = '/root_config.json'
    var_8 = var_3.insert(var_7, var_6)
    var_9 = var_3.search(var_7)
    var_10 = '/subdir/file.txt'
    var_11 = var_3.search(var_10)
    var_12 = module_0.Trie()
    var_13 = 'level'
    var_14 = 'root'
    var_15 = {var_13: var_14}
    var_16 = 'dir'
    var_17 = {var_13: var_16}
    var_18 = 'subdir'
    var_19 = {var_13: var_18}
    var_20 = '/config.json'
    var_21 = var_12.insert(var_20, var_15)
    var_22 = '/dir/config.json'
    var_23 = var_12.insert(var_22, var_17)
    var_24 = '/dir/subdir/config.json'
    var_25 = var_12.insert(var_24, var_19)
    var_26 = '/other/file.txt'
    var_27 = var_12.search(var_26)
    var_28 = '/dir/file.txt'
    var_29 = var_12.search(var_28)
    var_30 = '/dir/subdir/file.txt'
    var_31 = var_12.search(var_30)
    var_32 = '/dir/subdir/deeper/file.txt'
    var_33 = var_12.search(var_32)
    var_34 = module_0.Trie()
    var_35 = 'name'
    var_36 = 'test'
    var_37 = {var_35: var_36}
    var_38 = '/home/user/project/config.json'
    var_39 = var_34.insert(var_38, var_37)
    var_40 = '/home/user/project/src/../file.txt'
    var_41 = var_34.search(var_40)
    var_42 = module_0.Trie()
    var_43 = 'version'
    var_44 = '1.0'
    var_45 = {var_43: var_44}
    var_46 = '2.0'
    var_47 = {var_43: var_46}
    var_48 = '/path/config.json'
    var_49 = var_42.insert(var_48, var_45)
    var_50 = var_42.insert(var_48, var_47)
    var_51 = '/path/file.txt'
    var_52 = var_42.search(var_51)
    var_53 = module_0.Trie()
    var_54 = 'value'
    var_55 = {var_4: var_54}
    var_56 = var_53.insert(var_20, var_55)
    var_57 = ''
    var_58 = var_53.search(var_57)
    var_59 = module_0.Trie()
    var_60 = 'type'
    var_61 = 'global'
    var_62 = {var_60: var_61}
    var_63 = 'parent'
    var_64 = {var_60: var_63}
    var_65 = '/global/config.json'
    var_66 = module_0.Trie(var_65, var_62)
    var_67 = '/parent/config.json'
    var_68 = var_66.insert(var_67, var_64)
    var_69 = '/parent/child/file.txt'
    var_70 = var_66.search(var_69)
    var_71 = module_0.Trie()
    var_72 = 'config'
    var_73 = {var_72: var_14}
    var_74 = 'deep'
    var_75 = {var_72: var_74}
    var_76 = var_71.insert(var_20, var_73)
    var_77 = '/a/b/c/config.json'
    var_78 = var_71.insert(var_77, var_75)
    var_79 = '/a/b/d/file.txt'
    var_80 = var_71.search(var_79)
    var_81 = module_0.Trie()
    var_82 = 'os'
    var_83 = 'windows'
    var_84 = {var_82: var_83}
    var_85 = 'C:\\project\\config.json'
    var_86 = var_81.insert(var_85, var_84)
    var_87 = 'C:\\project\\src\\file.txt'
    var_88 = var_81.search(var_87)



# Parsed testcases at query #26
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = 'config.yaml'
    var_10 = {}
    var_11 = module_0.TrieNode(var_9, var_10)
    var_12 = 'config.toml'
    var_13 = None
    var_14 = module_0.TrieNode(var_12, var_13)



# Parsed testcases at query #27
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)
    var_13 = module_0.TrieNode()
    var_14 = module_0.TrieNode()



# Parsed testcases at query #28
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = None
    var_7 = module_0.Trie(var_1, var_6)
    var_8 = ''
    var_9 = {}
    var_10 = module_0.Trie(var_8, var_9)



# Parsed testcases at query #29
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
    var_7 = 'config.yaml'
    var_8 = 'setting'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = module_0.TrieNode(var_7, var_10)
    var_12 = 'a'
    var_13 = {var_12: var_9}
    var_14 = 'test.json'
    var_15 = module_0.TrieNode(var_14, var_13)



# Parsed testcases at query #30
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'root_value'
    var_5 = {var_3: var_4}
    var_6 = '/root/config.json'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/home/user/project/config.json'
    var_11 = 'env'
    var_12 = 'dev'
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/home/user/project/src/config.json'
    var_16 = 'src'
    var_17 = {var_11: var_16}
    var_18 = var_9.insert(var_15, var_17)
    var_19 = '/home/user/project/src/utils/config.json'
    var_20 = 'utils'
    var_21 = {var_11: var_20}
    var_22 = var_9.insert(var_19, var_21)
    var_23 = '/home/user/project/file.txt'
    var_24 = var_9.search(var_23)
    var_25 = '/home/user/project/src/main.py'
    var_26 = var_9.search(var_25)
    var_27 = '/home/user/project/src/utils/helper.py'
    var_28 = var_9.search(var_27)
    var_29 = module_0.Trie()
    var_30 = '/a/b/config.json'
    var_31 = 'level'
    var_32 = 'b'
    var_33 = {var_31: var_32}
    var_34 = var_29.insert(var_30, var_33)
    var_35 = '/a/b/c/d/file.txt'
    var_36 = var_29.search(var_35)
    var_37 = '/a/file.txt'
    var_38 = var_29.search(var_37)
    var_39 = module_0.Trie()
    var_40 = '/config.json'
    var_41 = 'global'
    var_42 = True
    var_43 = {var_41: var_42}
    var_44 = var_39.insert(var_40, var_43)
    var_45 = '/usr/config.json'
    var_46 = 'scope'
    var_47 = 'usr'
    var_48 = {var_46: var_47}
    var_49 = var_39.insert(var_45, var_48)
    var_50 = '/usr/local/config.json'
    var_51 = 'local'
    var_52 = {var_46: var_51}
    var_53 = var_39.insert(var_50, var_52)
    var_54 = '/usr/local/bin/file.txt'
    var_55 = var_39.search(var_54)
    var_56 = '/usr/bin/file.txt'
    var_57 = var_39.search(var_56)
    var_58 = '/etc/file.txt'
    var_59 = var_39.search(var_58)
    var_60 = module_0.Trie()
    var_61 = '/absolute/path/config.json'
    var_62 = 'test'
    var_63 = 'absolute'
    var_64 = {var_62: var_63}
    var_65 = var_60.insert(var_61, var_64)
    var_66 = '.'
    var_67 = 'test.py'
    var_68 = module_0.Trie()
    var_69 = 'exact'
    var_70 = {var_69: var_42}
    var_71 = '/exact/path/config.json'
    var_72 = var_68.insert(var_71, var_70)
    var_73 = var_68.search(var_71)
    var_74 = module_0.Trie()
    var_75 = '//double//slash//config.json'
    var_76 = 'slashes'
    var_77 = {var_62: var_76}
    var_78 = var_74.insert(var_75, var_77)
    var_79 = 'some//file.txt'
    var_80 = var_74.search(var_79)
    var_81 = len(var_80)
    var_82 = 2
    var_83 = var_81 == var_82
    var_84 = module_0.Trie()
    var_85 = '/parent/config.json'
    var_86 = 'parent'
    var_87 = {var_31: var_86}
    var_88 = var_84.insert(var_85, var_87)
    var_89 = '/parent/child/config.json'
    var_90 = 'child'
    var_91 = {var_31: var_90}
    var_92 = var_84.insert(var_89, var_91)
    var_93 = '/parent/child/grandchild/config.json'
    var_94 = 'grandchild'
    var_95 = {var_31: var_94}
    var_96 = var_84.insert(var_93, var_95)
    var_97 = '/parent/child/grandchild/file.txt'
    var_98 = var_84.search(var_97)
    var_99 = '/parent/child/middle/file.txt'
    var_100 = var_84.search(var_99)



# Parsed testcases at query #31
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'config1'
    var_8 = 'data1'
    var_9 = {var_7: var_8}
    var_10 = 'config2'
    var_11 = 'data2'
    var_12 = {var_10: var_11}
    var_13 = '/a/b/config1.json'
    var_14 = var_6.insert(var_13, var_9)
    var_15 = '/a/b/c/config2.json'
    var_16 = var_6.insert(var_15, var_12)
    var_17 = var_6.root
    var_18 = var_6.root
    var_19 = module_0.Trie()
    var_20 = 'old'
    var_21 = 'data'
    var_22 = {var_20: var_21}
    var_23 = 'new'
    var_24 = {var_23: var_21}
    var_25 = '/same/path/config.json'
    var_26 = var_19.insert(var_25, var_22)
    var_27 = var_19.insert(var_25, var_24)
    var_28 = var_19.root
    var_29 = module_0.Trie()
    var_30 = 'relative'
    var_31 = 'config'
    var_32 = {var_30: var_31}
    var_33 = './relative/path/config.json'
    var_34 = var_29.insert(var_33, var_32)
    var_35 = var_29.root
    var_36 = module_0.Trie()
    var_37 = 'file1'
    var_38 = {var_37: var_21}
    var_39 = 'file2'
    var_40 = {var_39: var_21}
    var_41 = 'file3'
    var_42 = {var_41: var_21}
    var_43 = '/projects/proj1/config.json'
    var_44 = var_36.insert(var_43, var_38)
    var_45 = '/projects/proj1/src/config.json'
    var_46 = var_36.insert(var_45, var_40)
    var_47 = '/projects/proj2/config.json'
    var_48 = var_36.insert(var_47, var_42)
    var_49 = var_36.root
    var_50 = var_36.root
    var_51 = var_36.root
    var_52 = module_0.Trie()
    var_53 = '/empty/config.json'
    var_54 = {}
    var_55 = var_52.insert(var_53, var_54)
    var_56 = var_52.root
    var_57 = 'root_config.json'
    var_58 = 'root'
    var_59 = {var_58: var_21}
    var_60 = module_0.Trie(var_57, var_59)
    var_61 = 'nested'
    var_62 = {var_61: var_21}
    var_63 = '/nested/config.json'
    var_64 = var_60.insert(var_63, var_62)
    var_65 = var_60.root



# Parsed testcases at query #32
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #33
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = 'key'
    var_3 = 'root_value'
    var_4 = {var_2: var_3}
    var_5 = '/root/config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = '/some/deep/nested/file.py'
    var_8 = module_0.Trie()
    var_9 = '/home/user/project/config.json'
    var_10 = 'env'
    var_11 = 'project'
    var_12 = {var_10: var_11}
    var_13 = var_8.insert(var_9, var_12)
    var_14 = '/home/user/project/src/config.json'
    var_15 = 'src'
    var_16 = {var_10: var_15}
    var_17 = var_8.insert(var_14, var_16)
    var_18 = '/home/user/project/src/utils/config.json'
    var_19 = 'utils'
    var_20 = {var_10: var_19}
    var_21 = var_8.insert(var_18, var_20)
    var_22 = '/home/user/project/src/utils/helper.py'
    var_23 = '/home/user/project/src/main.py'
    var_24 = '/home/user/project/README.md'
    var_25 = '/home/user/other/file.py'
    var_26 = module_0.Trie()
    var_27 = '/a/b/config.json'
    var_28 = 'config'
    var_29 = 'ab'
    var_30 = {var_28: var_29}
    var_31 = var_26.insert(var_27, var_30)
    var_32 = '/a/b/c/config.json'
    var_33 = 'abc'
    var_34 = {var_28: var_33}
    var_35 = var_26.insert(var_32, var_34)
    var_36 = '/a/b/c/d/file.py'
    var_37 = '/a/b/d/file.py'
    var_38 = module_0.Trie()
    var_39 = '/x/y/config.json'
    var_40 = 'test'
    var_41 = 'xy'
    var_42 = {var_40: var_41}
    var_43 = var_38.insert(var_39, var_42)
    var_44 = '/x/z/file.py'
    var_45 = module_0.Trie()
    var_46 = '/absolute/path/config.json'
    var_47 = 'abs'
    var_48 = 'true'
    var_49 = {var_47: var_48}
    var_50 = var_45.insert(var_46, var_49)
    var_51 = './some/relative/../path/file.py'
    var_52 = module_0.Trie()
    var_53 = '/config.json'
    var_54 = 'level'
    var_55 = 'root'
    var_56 = {var_54: var_55}
    var_57 = var_52.insert(var_53, var_56)
    var_58 = '/usr/config.json'
    var_59 = 'usr'
    var_60 = {var_54: var_59}
    var_61 = var_52.insert(var_58, var_60)
    var_62 = '/usr/local/config.json'
    var_63 = 'local'
    var_64 = {var_54: var_63}
    var_65 = var_52.insert(var_62, var_64)
    var_66 = '/usr/local/bin/config.json'
    var_67 = 'bin'
    var_68 = {var_54: var_67}
    var_69 = var_52.insert(var_66, var_68)
    var_70 = '/usr/local/bin/script.py'
    var_71 = (var_70, var_67)
    var_72 = '/usr/local/lib/file.py'
    var_73 = (var_72, var_63)
    var_74 = '/usr/share/file.py'
    var_75 = (var_74, var_59)
    var_76 = '/etc/file.py'
    var_77 = (var_76, var_55)
    var_78 = '/file.py'
    var_79 = (var_78, var_55)
    var_80 = [var_71, var_73, var_75, var_77, var_79]
    var_81 = 'level'
    var_82 = module_0.Trie()
    var_83 = '/empty/config.json'
    var_84 = {}
    var_85 = var_82.insert(var_83, var_84)
    var_86 = '/empty/file.py'



# Parsed testcases at query #34
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = None
    var_7 = module_0.Trie(var_1, var_6)
    var_8 = ''
    var_9 = module_0.Trie(var_8, var_4)



# Parsed testcases at query #35
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)
    var_13 = {var_3: var_5}
    var_14 = module_0.TrieNode(var_1, var_13)



# Parsed testcases at query #36
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'key1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = 'key2'
    var_11 = 'value2'
    var_12 = {var_10: var_11}
    var_13 = '/a/b/config1.json'
    var_14 = var_6.insert(var_13, var_9)
    var_15 = '/a/b/c/config2.json'
    var_16 = var_6.insert(var_15, var_12)
    var_17 = var_6.root
    var_18 = var_6.root
    var_19 = module_0.Trie()
    var_20 = 'old'
    var_21 = 'data'
    var_22 = {var_20: var_21}
    var_23 = 'new'
    var_24 = {var_23: var_21}
    var_25 = '/a/b/config.json'
    var_26 = var_19.insert(var_25, var_22)
    var_27 = var_19.insert(var_25, var_24)
    var_28 = var_19.root
    var_29 = module_0.Trie()
    var_30 = 'test'
    var_31 = 'relative'
    var_32 = {var_30: var_31}
    var_33 = './config.json'
    var_34 = var_29.insert(var_33, var_32)
    var_35 = var_29.root
    var_36 = module_0.Trie()
    var_37 = 'config'
    var_38 = '1'
    var_39 = {var_37: var_38}
    var_40 = '2'
    var_41 = {var_37: var_40}
    var_42 = '3'
    var_43 = {var_37: var_42}
    var_44 = '/common/prefix/a/config.json'
    var_45 = var_36.insert(var_44, var_39)
    var_46 = '/common/prefix/b/config.json'
    var_47 = var_36.insert(var_46, var_41)
    var_48 = '/common/different/config.json'
    var_49 = var_36.insert(var_48, var_43)
    var_50 = var_36.root
    var_51 = var_36.root
    var_52 = var_36.root
    var_53 = module_0.Trie()
    var_54 = '/empty/config.json'
    var_55 = {}
    var_56 = var_53.insert(var_54, var_55)
    var_57 = var_53.root



# Parsed testcases at query #37
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'key1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = 'key2'
    var_11 = 'value2'
    var_12 = {var_10: var_11}
    var_13 = '/a/b/config1.json'
    var_14 = var_6.insert(var_13, var_9)
    var_15 = '/a/b/c/config2.json'
    var_16 = var_6.insert(var_15, var_12)
    var_17 = var_6.root
    var_18 = var_6.root
    var_19 = module_0.Trie()
    var_20 = 'old'
    var_21 = 'data'
    var_22 = {var_20: var_21}
    var_23 = 'new'
    var_24 = {var_23: var_21}
    var_25 = '/a/b/config.json'
    var_26 = var_19.insert(var_25, var_22)
    var_27 = var_19.insert(var_25, var_24)
    var_28 = var_19.root
    var_29 = module_0.Trie()
    var_30 = 'test'
    var_31 = 'relative'
    var_32 = {var_30: var_31}
    var_33 = './config.json'
    var_34 = var_29.insert(var_33, var_32)
    var_35 = var_29.root
    var_36 = module_0.Trie()
    var_37 = '/common/a/config1.json'
    var_38 = 'id'
    var_39 = 1
    var_40 = {var_38: var_39}
    var_41 = (var_37, var_40)
    var_42 = '/common/b/config2.json'
    var_43 = 2
    var_44 = {var_38: var_43}
    var_45 = (var_42, var_44)
    var_46 = '/common/a/b/config3.json'
    var_47 = 3
    var_48 = {var_38: var_47}
    var_49 = (var_46, var_48)
    var_50 = [var_41, var_45, var_49]
    var_51 = var_36.root
    var_52 = module_0.Trie()
    var_53 = '/empty/config.json'
    var_54 = {}
    var_55 = var_52.insert(var_53, var_54)
    var_56 = var_52.root
    var_57 = 'root_config.json'
    var_58 = 'root'
    var_59 = True
    var_60 = {var_58: var_59}
    var_61 = module_0.Trie(var_57, var_60)
    var_62 = 'nested'
    var_63 = True
    var_64 = {var_62: var_63}
    var_65 = var_61.insert(var_25, var_64)
    var_66 = var_61.root



# Parsed testcases at query #38
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = 'key'
    var_3 = 'root_value'
    var_4 = {var_2: var_3}
    var_5 = '/root/config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = '/some/deep/nested/file.py'
    var_8 = module_0.Trie()
    var_9 = '/home/user/project/config.json'
    var_10 = 'env'
    var_11 = 'project'
    var_12 = {var_10: var_11}
    var_13 = var_8.insert(var_9, var_12)
    var_14 = '/home/user/project/src/config.json'
    var_15 = 'src'
    var_16 = {var_10: var_15}
    var_17 = var_8.insert(var_14, var_16)
    var_18 = '/home/user/project/src/utils/config.json'
    var_19 = 'utils'
    var_20 = {var_10: var_19}
    var_21 = var_8.insert(var_18, var_20)
    var_22 = '/home/user/project/src/utils/helper.py'
    var_23 = '/home/user/project/src/main.py'
    var_24 = '/home/user/project/README.md'
    var_25 = '/home/user/other/file.py'
    var_26 = module_0.Trie()
    var_27 = '/absolute/path/config.json'
    var_28 = 'test'
    var_29 = 'absolute'
    var_30 = {var_28: var_29}
    var_31 = var_26.insert(var_27, var_30)
    var_32 = './some/relative/../file.py'
    var_33 = module_0.Trie()
    var_34 = '/exact/path/config.json'
    var_35 = 'exact'
    var_36 = True
    var_37 = {var_35: var_36}
    var_38 = var_33.insert(var_34, var_37)
    var_39 = '/exact/path/file.py'
    var_40 = module_0.Trie()
    var_41 = '/parent/config.json'
    var_42 = 'parent'
    var_43 = {var_42: var_36}
    var_44 = var_40.insert(var_41, var_43)
    var_45 = '/parent/child/grandchild/file.py'
    var_46 = module_0.Trie()
    var_47 = '/a/config.json'
    var_48 = 'level'
    var_49 = 'a'
    var_50 = {var_48: var_49}
    var_51 = var_46.insert(var_47, var_50)
    var_52 = '/a/b/config.json'
    var_53 = 'b'
    var_54 = {var_48: var_53}
    var_55 = var_46.insert(var_52, var_54)
    var_56 = '/a/b/c/config.json'
    var_57 = 'c'
    var_58 = {var_48: var_57}
    var_59 = var_46.insert(var_56, var_58)
    var_60 = '/a/b/c/d/e/f.py'
    var_61 = '/a/b/d/e/f.py'
    var_62 = '/a/x/y/z.py'
    var_63 = module_0.Trie()
    var_64 = '/config.json'
    var_65 = 'root'
    var_66 = {var_65: var_36}
    var_67 = var_63.insert(var_64, var_66)
    var_68 = '/'
    var_69 = '/root_config.json'
    var_70 = {var_65: var_36}
    var_71 = module_0.Trie(var_69, var_70)
    var_72 = '/any/sub/directory/file.py'



# Parsed testcases at query #39
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.txt'
    var_2 = var_0.search(var_1)
    var_3 = 'key'
    var_4 = 'root_value'
    var_5 = {var_3: var_4}
    var_6 = '/root/config.json'
    var_7 = module_0.Trie(var_6, var_5)
    var_8 = var_7.search(var_1)
    var_9 = module_0.Trie()
    var_10 = '/home/user/config.json'
    var_11 = 'user'
    var_12 = 'config'
    var_13 = {var_11: var_12}
    var_14 = var_9.insert(var_10, var_13)
    var_15 = '/home/user/project/config.json'
    var_16 = 'project'
    var_17 = {var_16: var_12}
    var_18 = var_9.insert(var_15, var_17)
    var_19 = '/home/user/project/src/config.json'
    var_20 = 'src'
    var_21 = {var_20: var_12}
    var_22 = var_9.insert(var_19, var_21)
    var_23 = '/home/user/project/src/main.py'
    var_24 = var_9.search(var_23)
    var_25 = '/home/user/project/utils.py'
    var_26 = var_9.search(var_25)
    var_27 = '/home/user/other.py'
    var_28 = var_9.search(var_27)
    var_29 = '/other/file.py'
    var_30 = var_9.search(var_29)
    var_31 = module_0.Trie()
    var_32 = '/a/b/config.json'
    var_33 = 'level'
    var_34 = 'b'
    var_35 = {var_33: var_34}
    var_36 = var_31.insert(var_32, var_35)
    var_37 = '/a/b/c/config.json'
    var_38 = 'c'
    var_39 = {var_33: var_38}
    var_40 = var_31.insert(var_37, var_39)
    var_41 = '/a/b/c/d/file.txt'
    var_42 = var_31.search(var_41)
    var_43 = '/a/b/other/file.txt'
    var_44 = var_31.search(var_43)
    var_45 = module_0.Trie()
    var_46 = '/real/path/config.json'
    var_47 = 'real'
    var_48 = {var_47: var_12}
    var_49 = var_45.insert(var_46, var_48)
    var_50 = '/real/path/../path/file.txt'
    var_51 = var_45.search(var_50)
    var_52 = module_0.Trie()
    var_53 = '/a/config.json'
    var_54 = 'a'
    var_55 = {var_33: var_54}
    var_56 = var_52.insert(var_53, var_55)
    var_57 = {var_33: var_38}
    var_58 = var_52.insert(var_37, var_57)
    var_59 = '/a/b/file.txt'
    var_60 = var_52.search(var_59)
    var_61 = var_52.search(var_41)
    var_62 = 'root'
    var_63 = {var_62: var_12}
    var_64 = module_0.Trie(var_6, var_63)
    var_65 = ''
    var_66 = var_64.search(var_65)
    var_67 = module_0.Trie()
    var_68 = '/dir/config.json'
    var_69 = 'dir'
    var_70 = {var_69: var_12}
    var_71 = var_67.insert(var_68, var_70)
    var_72 = '/dir'
    var_73 = var_67.search(var_72)
    var_74 = module_0.Trie()
    var_75 = '/a/config1.json'
    var_76 = '1'
    var_77 = {var_12: var_76}
    var_78 = var_74.insert(var_75, var_77)
    var_79 = '/a/config2.json'
    var_80 = '2'
    var_81 = {var_12: var_80}
    var_82 = var_74.insert(var_79, var_81)
    var_83 = '/a/file.txt'
    var_84 = var_74.search(var_83)



# Parsed testcases at query #40
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.TrieNode(var_1, var_5)
    var_7 = {}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = None
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = {var_3: var_4}
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #41
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = None
    var_7 = module_0.Trie(var_1, var_6)
    var_8 = ''
    var_9 = module_0.Trie(var_8, var_4)



# Parsed testcases at query #42
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.TrieNode()
    var_1 = 'config.json'
    var_2 = module_0.TrieNode(var_1)
    var_3 = 'key'
    var_4 = 'setting'
    var_5 = 'value'
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.TrieNode(var_1, var_7)
    var_9 = {}
    var_10 = module_0.TrieNode(var_1, var_9)
    var_11 = None
    var_12 = module_0.TrieNode(var_1, var_11)



# Parsed testcases at query #43
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = None
    var_7 = module_0.Trie(var_1, var_6)
    var_8 = ''
    var_9 = module_0.Trie(var_8, var_4)



# Parsed testcases at query #44
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/some/file.py'
    var_2 = 'key'
    var_3 = 'root_value'
    var_4 = {var_2: var_3}
    var_5 = '/root/config.json'
    var_6 = module_0.Trie(var_5, var_4)
    var_7 = '/some/deep/nested/file.py'
    var_8 = module_0.Trie()
    var_9 = '/home/user/config.json'
    var_10 = 'env'
    var_11 = 'user'
    var_12 = {var_10: var_11}
    var_13 = var_8.insert(var_9, var_12)
    var_14 = '/home/user/project/config.json'
    var_15 = 'project'
    var_16 = {var_10: var_15}
    var_17 = var_8.insert(var_14, var_16)
    var_18 = '/home/user/project/src/config.json'
    var_19 = 'src'
    var_20 = {var_10: var_19}
    var_21 = var_8.insert(var_18, var_20)
    var_22 = '/home/user/project/utils.py'
    var_23 = '/home/user/project/src/main.py'
    var_24 = '/home/user/docs/readme.md'
    var_25 = module_0.Trie()
    var_26 = '/etc/app/config.json'
    var_27 = 'global'
    var_28 = True
    var_29 = {var_27: var_28}
    var_30 = var_25.insert(var_26, var_29)
    var_31 = module_0.Trie()
    var_32 = '/a/b/config.json'
    var_33 = 'level'
    var_34 = 'b'
    var_35 = {var_33: var_34}
    var_36 = var_31.insert(var_32, var_35)
    var_37 = '/a/b/c/config.json'
    var_38 = 'c'
    var_39 = {var_33: var_38}
    var_40 = var_31.insert(var_37, var_39)
    var_41 = '/a/b/d/file.py'
    var_42 = module_0.Trie()
    var_43 = './local/config.json'
    var_44 = 'local'
    var_45 = {var_44: var_28}
    var_46 = var_42.insert(var_43, var_45)
    var_47 = module_0.Trie()
    var_48 = 'root'
    var_49 = {var_48: var_28}
    var_50 = var_47.insert(var_5, var_49)
    var_51 = '/root/project/config.json'
    var_52 = {var_15: var_28}
    var_53 = var_47.insert(var_51, var_52)
    var_54 = '/root/project/src/file.py'
    var_55 = module_0.Trie()
    var_56 = '/empty/config.json'
    var_57 = {}
    var_58 = var_55.insert(var_56, var_57)
    var_59 = '/empty/file.py'
    var_60 = module_0.Trie()
    var_61 = '/level1/config.json'
    var_62 = {var_33: var_28}
    var_63 = var_60.insert(var_61, var_62)
    var_64 = '/level1/level2/config.json'
    var_65 = 2
    var_66 = {var_33: var_65}
    var_67 = var_60.insert(var_64, var_66)
    var_68 = '/level1/level2/level3/config.json'
    var_69 = 3
    var_70 = {var_33: var_69}
    var_71 = var_60.insert(var_68, var_70)
    var_72 = '/level1/level2/level3/level4/file.py'
    var_73 = module_0.Trie()
    var_74 = '/path with spaces/config.json'
    var_75 = 'has_spaces'
    var_76 = {var_75: var_28}
    var_77 = var_73.insert(var_74, var_76)
    var_78 = '/path-with-dashes/config.json'
    var_79 = 'has_dashes'
    var_80 = {var_79: var_28}
    var_81 = var_73.insert(var_78, var_80)
    var_82 = '/path with spaces/subdir/file.py'



# Parsed testcases at query #45
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = module_0.Trie(var_1)
    var_7 = {}
    var_8 = module_0.Trie(var_1, var_7)
    var_9 = var_0.root
    var_10 = var_5.root



# Parsed testcases at query #46
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'settings'
    var_8 = 'debug'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = '/home/user/project/src/config.yaml'
    var_13 = var_6.insert(var_12, var_11)
    var_14 = 'home'
    var_15 = var_6.root.nodes[var_14]
    var_16 = 'user'
    var_17 = var_15.nodes[var_16]
    var_18 = 'project'
    var_19 = var_17.nodes[var_18]
    var_20 = 'src'
    var_21 = var_19.nodes[var_20]
    var_22 = module_0.Trie()
    var_23 = 'name'
    var_24 = 'config_a'
    var_25 = {var_23: var_24}
    var_26 = 'config_b'
    var_27 = {var_23: var_26}
    var_28 = '/a/b/c/config.json'
    var_29 = var_22.insert(var_28, var_25)
    var_30 = '/a/b/d/config.json'
    var_31 = var_22.insert(var_30, var_27)
    var_32 = 'a'
    var_33 = var_22.root.nodes[var_32]
    var_34 = 'b'
    var_35 = var_33.nodes[var_34]
    var_36 = 'c'
    var_37 = var_35.nodes[var_36]
    var_38 = 'd'
    var_39 = var_35.nodes[var_38]
    var_40 = module_0.Trie()
    var_41 = 'version'
    var_42 = {var_41: var_9}
    var_43 = 2
    var_44 = {var_41: var_43}
    var_45 = '/path/config.json'
    var_46 = var_40.insert(var_45, var_42)
    var_47 = var_40.insert(var_45, var_44)
    var_48 = 'path'
    var_49 = var_40.root.nodes[var_48]
    var_50 = module_0.Trie()
    var_51 = 'test'
    var_52 = 'data'
    var_53 = {var_51: var_52}
    var_54 = '/etc/app/config.json'
    var_55 = var_50.insert(var_54, var_53)
    var_56 = module_0.Trie()
    var_57 = 'relative'
    var_58 = {var_57: var_51}
    var_59 = './config.json'
    var_60 = var_56.insert(var_59, var_58)
    var_61 = '/root.json'
    var_62 = {}
    var_63 = module_0.Trie(var_61, var_62)
    var_64 = 'root'
    var_65 = 'config'
    var_66 = {var_64: var_65}
    var_67 = var_63.insert(var_61, var_66)



# Parsed testcases at query #47
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = '/path/to/config.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Trie(var_1, var_4)
    var_6 = None
    var_7 = module_0.Trie(var_1, var_6)
    var_8 = module_0.Trie(var_1)



# Parsed testcases at query #48
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = module_0.Trie()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/root/config.json'
    var_5 = var_0.insert(var_4, var_3)
    var_6 = module_0.Trie()
    var_7 = 'settings'
    var_8 = 'debug'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = '/home/user/project/src/config.yaml'
    var_13 = var_6.insert(var_12, var_11)
    var_14 = var_6.root
    var_15 = module_0.Trie()
    var_16 = 'name'
    var_17 = 'config1'
    var_18 = {var_16: var_17}
    var_19 = 'config2'
    var_20 = {var_16: var_19}
    var_21 = '/a/b/c/config1.json'
    var_22 = var_15.insert(var_21, var_18)
    var_23 = '/a/b/c/config2.json'
    var_24 = var_15.insert(var_23, var_20)
    var_25 = var_15.root
    var_26 = 'initial.json'
    var_27 = 'initial'
    var_28 = {var_27: var_9}
    var_29 = module_0.Trie(var_26, var_28)
    var_30 = 'new'
    var_31 = 'config'
    var_32 = {var_30: var_31}
    var_33 = '/new/path/config.json'
    var_34 = var_29.insert(var_33, var_32)
    var_35 = var_29.root
    var_36 = module_0.Trie()
    var_37 = 'relative'
    var_38 = 'path'
    var_39 = {var_37: var_38}
    var_40 = './relative/config.json'
    var_41 = var_36.insert(var_40, var_39)
    var_42 = var_36.root
    var_43 = module_0.Trie()
    var_44 = 'version'
    var_45 = {var_44: var_9}
    var_46 = 2
    var_47 = {var_44: var_46}
    var_48 = '/same/path/config.json'
    var_49 = var_43.insert(var_48, var_45)
    var_50 = var_43.insert(var_48, var_47)
    var_51 = var_43.root



