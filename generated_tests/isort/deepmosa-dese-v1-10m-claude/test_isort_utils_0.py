# Check out: https://github.com/GlowCheese/deepmosa
import isort.utils as module_0
import pytest


def test_case_0():
    var_0 = 'f5X6l,T}O"'
    var_1 = {var_0: var_0}
    var_2 = module_0.Trie(config_data=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_2.root).__module__}.{type(var_2.root).__qualname__}' == 'isort.utils.TrieNode'
    var_3 = "kcv[uy'"
    var_4 = var_2.insert(var_3, var_3)

def test_case_1():
    var_0 = module_0.TrieNode()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.TrieNode'
    assert var_0.nodes == {}
    assert var_0.config_info == ('', {})

def test_case_2():
    var_0 = '[v+=m*\t*R\td\x0b>,J]'
    var_1 = 'r%'
    var_2 = module_0.Trie(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_2.root).__module__}.{type(var_2.root).__qualname__}' == 'isort.utils.TrieNode'
    var_3 = var_2.search(var_0)
    var_4 = 'X>'
    var_5 = module_0.Trie()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_5.root).__module__}.{type(var_5.root).__qualname__}' == 'isort.utils.TrieNode'
    var_6 = var_5.search(var_4)

def test_case_3():
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'

def test_case_4():
    var_0 = '[v+=m*\t*R\td\x0b>,J]'
    var_1 = module_0.Trie()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = 'r%'
    var_3 = None
    var_4 = var_1.insert(var_2, var_3)
    var_5 = var_1.search(var_0)
    var_6 = 'X>'
    var_7 = module_0.Trie()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_7.root).__module__}.{type(var_7.root).__qualname__}' == 'isort.utils.TrieNode'
    var_8 = var_7.search(var_6)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.TrieNode(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.TrieNode'
    assert var_1.nodes == {}
    assert var_1.config_info == (None, {})
    var_2 = 'j;AFQ?Fec%[85gm4p'
    var_3 = module_0.Trie(var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_3.root).__module__}.{type(var_3.root).__qualname__}' == 'isort.utils.TrieNode'
    var_4 = var_3.insert(var_2, var_0)
    var_5 = "b^<H\r=77J?j`'dg"
    var_6 = ''
    var_7 = var_3.search(var_6)
    var_8 = module_0.Trie(config_data=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_8.root).__module__}.{type(var_8.root).__qualname__}' == 'isort.utils.TrieNode'
    var_9 = var_8.search(var_5)
    var_10 = None
    var_11 = module_0.Trie()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_11.root).__module__}.{type(var_11.root).__qualname__}' == 'isort.utils.TrieNode'
    var_11.insert(var_10, var_11)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_0.TrieNode()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.TrieNode'
    assert var_0.nodes == {}
    assert var_0.config_info == ('', {})
    var_1 = module_0.Trie()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = module_0.Trie()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_2.root).__module__}.{type(var_2.root).__qualname__}' == 'isort.utils.TrieNode'
    var_3 = 'tc'
    var_4 = None
    var_5 = var_2.insert(var_3, var_4)
    var_6 = None
    var_7 = module_0.TrieNode(var_4, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.utils.TrieNode'
    assert var_7.nodes == {}
    assert var_7.config_info == (None, {})
    var_8 = {}
    var_9 = var_2.insert(var_3, var_8)
    var_10 = var_2.search(var_3)
    var_11 = 'I}M8  @0r|\\'
    var_12 = var_2.insert(var_11, var_6)
    var_13 = module_0.Trie(config_data=var_4)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_13.root).__module__}.{type(var_13.root).__qualname__}' == 'isort.utils.TrieNode'
    var_14 = var_2.search(var_3)
    var_15 = '\n'
    var_16 = '/<7U}'
    var_17 = {var_15: var_3, var_16: var_2, var_3: var_4}
    var_1.insert(var_9, var_17)