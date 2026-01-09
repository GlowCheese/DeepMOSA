# Check out: https://github.com/GlowCheese/deepmosa
import isort.utils as module_0


def test_case_0():
    var_0 = module_0.TrieNode()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.TrieNode'
    assert var_0.nodes == {}
    assert var_0.config_info == ('', {})
    var_1 = module_0.Trie(config_data=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'

def test_case_1():
    var_0 = module_0.TrieNode()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.TrieNode'
    assert var_0.nodes == {}
    assert var_0.config_info == ('', {})

def test_case_2():
    var_0 = '\t`2[4\x0b'
    var_1 = module_0.Trie()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = var_1.insert(var_0, var_0)

def test_case_3():
    var_0 = ']npO#hrtZ."J]M\tY1'
    var_1 = module_0.Trie()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = var_1.search(var_0)

def test_case_4():
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'

def test_case_5():
    var_0 = '0z}D@hi%}q{Fa2?'
    var_1 = module_0.Trie()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = var_1.insert(var_0, var_0)
    var_3 = var_1.insert(var_0, var_2)

def test_case_6():
    var_0 = 'Rl3Z`e.s;x^9q'
    var_1 = module_0.Trie(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = var_1.search(var_0)

def test_case_7():
    var_0 = ''
    var_1 = module_0.Trie()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = var_1.insert(var_0, var_1)
    var_3 = var_1.search(var_0)