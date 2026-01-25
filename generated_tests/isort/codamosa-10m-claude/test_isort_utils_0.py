# Check out: https://github.com/GlowCheese/deepmosa
import isort.utils as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = 'f5X6l,T}O"'
    var_2 = {var_1: var_1}
    var_3 = module_0.Trie(config_data=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_3.root).__module__}.{type(var_3.root).__qualname__}' == 'isort.utils.TrieNode'
    var_4 = "kcv[uy'"
    var_5 = var_3.insert(var_4, var_0)
    var_3.search(var_0)

def test_case_1():
    var_0 = module_0.TrieNode()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.TrieNode'
    assert var_0.nodes == {}
    assert var_0.config_info == ('', {})

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = None
    var_2 = module_0.Trie(config_data=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_2.root).__module__}.{type(var_2.root).__qualname__}' == 'isort.utils.TrieNode'
    var_3 = 'Ao4/$%`t;l]4Ku@0'
    var_4 = {var_3: var_2}
    var_5 = var_2.insert(var_3, var_4)
    var_2.insert(var_0, var_0)

def test_case_3():
    var_0 = '[v+=m*\t*R\td\x0b>,J]'
    var_1 = 'r%'
    var_2 = module_0.Trie(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_2.root).__module__}.{type(var_2.root).__qualname__}' == 'isort.utils.TrieNode'
    var_3 = var_2.search(var_0)
    var_4 = 'X>'
    var_5 = var_2.search(var_4)

def test_case_4():
    var_0 = 'r%'
    var_1 = module_0.Trie()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = var_1.search(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.TrieNode(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.TrieNode'
    assert var_1.nodes == {}
    assert var_1.config_info == (None, {})
    var_2 = 'Z+]Un;M\x0cJ;3^~gZ!eQ'
    var_3 = module_0.Trie(config_data=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_3.root).__module__}.{type(var_3.root).__qualname__}' == 'isort.utils.TrieNode'
    var_4 = var_3.search(var_2)
    var_5 = module_0.Trie()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_5.root).__module__}.{type(var_5.root).__qualname__}' == 'isort.utils.TrieNode'
    var_6 = None
    var_7 = '%%_`<cT[~v'
    var_8 = '\n!'
    var_9 = {var_7: var_0, var_2: var_7, var_8: var_5}
    var_10 = var_3.insert(var_7, var_9)
    var_11 = var_3.search(var_8)
    var_12 = module_0.TrieNode()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'isort.utils.TrieNode'
    assert var_12.nodes == {}
    assert var_12.config_info == ('', {})
    var_5.search(var_6)

def test_case_6():
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'
    var_1 = '@,p*; &v9y\x0c'
    var_2 = "!/'bnp#W*|"
    var_3 = 'O\tk2I^JJ=Cu#z7='
    var_4 = ''
    var_5 = '%];=^tRu,jSD'
    var_6 = True
    var_7 = {var_3: var_0}
    var_8 = var_0.insert(var_3, var_7)
    var_9 = '_{&hT\x0bi@+'
    var_10 = 974
    var_11 = {var_5: var_6, var_2: var_5, var_9: var_10}
    var_12 = (var_3, var_11)
    var_13 = {var_2: var_0, var_3: var_3, var_4: var_12, var_2: var_5}
    var_14 = var_0.insert(var_1, var_13)
    var_15 = var_0.search(var_1)
    var_16 = None
    var_0.insert(var_16, var_16)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'
    var_1 = "!/'bnp#W*|"
    var_2 = 'O\tkcI^Jr=Cu#z7='
    var_3 = ''
    var_4 = '%];=^tRu,jSD'
    var_5 = True
    var_6 = '_{&hT\x0bi@+'
    var_7 = 974
    var_8 = {var_4: var_5, var_1: var_4, var_6: var_7}
    var_9 = (var_2, var_8)
    var_10 = {var_1: var_0, var_2: var_2, var_3: var_9, var_1: var_4}
    var_11 = var_0.insert(var_3, var_10)
    var_12 = var_0.search(var_3)
    var_13 = None
    var_0.insert(var_13, var_13)