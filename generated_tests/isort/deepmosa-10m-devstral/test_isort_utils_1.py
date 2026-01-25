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
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'

def test_case_2():
    var_0 = ''
    var_1 = None
    var_2 = module_0.Trie()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_2.root).__module__}.{type(var_2.root).__qualname__}' == 'isort.utils.TrieNode'
    var_3 = var_2.insert(var_0, var_1)

def test_case_3():
    var_0 = module_0.TrieNode()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.TrieNode'
    assert var_0.nodes == {}
    assert var_0.config_info == ('', {})
    var_1 = None
    var_2 = "dfN#7?IT_zx'VK|-"
    var_3 = module_0.TrieNode(config_data=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.utils.TrieNode'
    assert var_3.nodes == {}
    assert var_3.config_info == ('', {})
    var_4 = module_0.TrieNode(config_data=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.utils.TrieNode'
    assert var_4.nodes == {}
    assert var_4.config_info == ('', {})
    var_5 = module_0.Trie()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_5.root).__module__}.{type(var_5.root).__qualname__}' == 'isort.utils.TrieNode'
    var_6 = var_5.search(var_2)

def test_case_4():
    var_0 = '\\!y3'
    var_1 = "CF>e3SOJU\\^g;2o Y'"
    var_2 = '0z}D@hi%}q{Fa2?'
    var_3 = {var_2: var_1}
    var_4 = module_0.Trie()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_4.root).__module__}.{type(var_4.root).__qualname__}' == 'isort.utils.TrieNode'
    var_5 = var_4.insert(var_1, var_3)
    var_6 = var_4.insert(var_0, var_5)
    var_7 = None
    var_8 = module_0.Trie()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_8.root).__module__}.{type(var_8.root).__qualname__}' == 'isort.utils.TrieNode'
    var_9 = var_8.search(var_0)
    var_10 = var_8.insert(var_0, var_7)
    var_11 = var_8.search(var_0)
    var_12 = None
    var_13 = module_0.TrieNode(config_data=var_10)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.utils.TrieNode'
    assert var_13.nodes == {}
    assert var_13.config_info == ('', {})
    var_14 = module_0.TrieNode(var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'isort.utils.TrieNode'
    assert var_14.nodes == {}
    assert var_14.config_info == (None, {})
    var_15 = module_0.Trie(config_data=var_7)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_15.root).__module__}.{type(var_15.root).__qualname__}' == 'isort.utils.TrieNode'

def test_case_5():
    var_0 = '\\!y3'
    var_1 = module_0.Trie()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = None
    var_3 = module_0.Trie()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_3.root).__module__}.{type(var_3.root).__qualname__}' == 'isort.utils.TrieNode'
    var_4 = var_3.search(var_0)
    var_5 = var_3.insert(var_0, var_2)
    var_6 = var_3.search(var_0)
    var_7 = None
    var_8 = module_0.TrieNode(config_data=var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'isort.utils.TrieNode'
    assert var_8.nodes == {}
    assert var_8.config_info == ('', {})
    var_9 = module_0.TrieNode(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.utils.TrieNode'
    assert var_9.nodes == {}
    assert var_9.config_info == (None, {})
    var_10 = module_0.Trie(config_data=var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_10.root).__module__}.{type(var_10.root).__qualname__}' == 'isort.utils.TrieNode'
    var_11 = module_0.Trie()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_11.root).__module__}.{type(var_11.root).__qualname__}' == 'isort.utils.TrieNode'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '\\!y3'
    var_1 = "CF>e3SOJU\\^g;2o Y'"
    var_2 = '0z}D@hi%}q{Fa2?'
    var_3 = {var_2: var_1}
    var_4 = module_0.Trie()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_4.root).__module__}.{type(var_4.root).__qualname__}' == 'isort.utils.TrieNode'
    var_5 = var_4.insert(var_1, var_3)
    var_6 = var_4.insert(var_0, var_5)
    var_7 = None
    var_8 = module_0.Trie()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_8.root).__module__}.{type(var_8.root).__qualname__}' == 'isort.utils.TrieNode'
    var_9 = var_8.search(var_0)
    var_10 = var_8.insert(var_0, var_7)
    var_11 = module_0.Trie()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_11.root).__module__}.{type(var_11.root).__qualname__}' == 'isort.utils.TrieNode'
    var_12 = ''
    var_13 = var_4.search(var_12)
    var_14 = var_8.search(var_1)
    var_15 = module_0.Trie(var_0)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_15.root).__module__}.{type(var_15.root).__qualname__}' == 'isort.utils.TrieNode'
    var_15.insert(var_5, var_3)