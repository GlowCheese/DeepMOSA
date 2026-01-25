# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.utils as module_0

def test_case_0():
    var_0 = '/39b$0hf3e'
    var_1 = module_0.Trie(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = module_0.TrieNode()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.utils.TrieNode'
    assert var_2.nodes == {}
    assert var_2.config_info == ('', {})
    var_3 = module_0.TrieNode()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.utils.TrieNode'
    assert var_3.nodes == {}
    assert var_3.config_info == ('', {})
    var_4 = '?aBz!".g/'
    var_5 = {var_4: var_4}
    var_6 = module_0.Trie(config_data=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_6.root).__module__}.{type(var_6.root).__qualname__}' == 'isort.utils.TrieNode'

def test_case_1():
    var_0 = module_0.TrieNode()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.TrieNode'
    assert var_0.nodes == {}
    assert var_0.config_info == ('', {})

def test_case_2():
    var_0 = ''
    var_1 = None
    var_2 = module_0.Trie()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_2.root).__module__}.{type(var_2.root).__qualname__}' == 'isort.utils.TrieNode'
    var_3 = var_2.insert(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'W8'
    var_1 = None
    var_2 = ' <jPLOlpNES4kJ\rJ2'
    var_3 = 'X7"Uwx|);'
    var_4 = {var_1}
    var_5 = module_0.Trie(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_5.root).__module__}.{type(var_5.root).__qualname__}' == 'isort.utils.TrieNode'
    var_6 = var_5.insert(var_3, var_4)
    var_7 = module_0.Trie()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_7.root).__module__}.{type(var_7.root).__qualname__}' == 'isort.utils.TrieNode'
    var_8 = var_7.search(var_2)
    var_9 = ':#\x0b9u{J,p'
    var_10 = var_5.insert(var_9, var_1)
    var_5.insert(var_6, var_1)

def test_case_4():
    var_0 = 'Rl3Z`e.s;x^9q'
    var_1 = module_0.Trie(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = var_1.search(var_0)

def test_case_5():
    var_0 = ''
    var_1 = False
    var_2 = None
    var_3 = module_0.Trie(var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_3.root).__module__}.{type(var_3.root).__qualname__}' == 'isort.utils.TrieNode'
    var_4 = var_3.search(var_0)

def test_case_6():
    var_0 = 'eMdCk'
    var_1 = module_0.Trie()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = module_0.Trie()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_2.root).__module__}.{type(var_2.root).__qualname__}' == 'isort.utils.TrieNode'
    var_3 = var_2.insert(var_0, var_0)
    var_4 = var_2.search(var_0)
    var_5 = 783
    var_6 = ''
    var_7 = var_1.insert(var_6, var_3)
    var_8 = module_0.TrieNode(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'isort.utils.TrieNode'
    assert var_8.nodes == {}
    assert var_8.config_info == (783, {})

def test_case_7():
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '\tk/\r?%'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.Trie()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_2.root).__module__}.{type(var_2.root).__qualname__}' == 'isort.utils.TrieNode'
    var_3 = var_2.insert(var_0, var_1)
    var_4 = module_0.Trie()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_4.root).__module__}.{type(var_4.root).__qualname__}' == 'isort.utils.TrieNode'
    var_5 = var_2.search(var_0)
    var_6 = None
    var_7 = module_0.Trie()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_7.root).__module__}.{type(var_7.root).__qualname__}' == 'isort.utils.TrieNode'
    var_8 = var_7.insert(var_0, var_6)
    var_9 = module_0.Trie(var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_9.root).__module__}.{type(var_9.root).__qualname__}' == 'isort.utils.TrieNode'
    var_10 = 'O@'
    var_11 = var_2.search(var_10)
    var_12 = module_0.Trie()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_12.root).__module__}.{type(var_12.root).__qualname__}' == 'isort.utils.TrieNode'
    var_13 = {}
    var_14 = ''
    var_15 = var_2.search(var_14)
    var_16 = module_0.TrieNode(config_data=var_13)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'isort.utils.TrieNode'
    assert var_16.nodes == {}
    assert var_16.config_info == ('', {})
    var_17 = var_9.insert(var_14, var_6)
    var_18 = var_7.insert(var_14, var_6)
    var_2.insert(var_17, var_18)