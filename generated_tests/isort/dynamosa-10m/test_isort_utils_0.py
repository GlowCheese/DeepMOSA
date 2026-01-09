# Check out: https://github.com/GlowCheese/deepmosa
import isort.utils as module_0


def test_case_0():
    var_0 = 'UEG~g\n%*i)Ooap{"\x0c@'
    var_1 = module_0.Trie(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'

def test_case_1():
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'

def test_case_2():
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'
    var_1 = ''
    var_2 = var_0.insert(var_1, var_1)

def test_case_3():
    var_0 = '[v+=m*\t*R\td\x0b,J]'
    var_1 = module_0.Trie(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    var_2 = var_1.search(var_0)

def test_case_4():
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'
    var_1 = ' }^{qc\x0c'
    var_2 = var_0.search(var_1)

def test_case_5():
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'
    var_1 = 'r%'
    var_2 = var_0.insert(var_1, var_1)
    var_3 = var_0.search(var_1)

def test_case_6():
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'
    var_1 = ''
    var_2 = var_0.insert(var_1, var_1)
    var_3 = var_0.search(var_1)

def test_case_7():
    var_0 = module_0.Trie()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_0.root).__module__}.{type(var_0.root).__qualname__}' == 'isort.utils.TrieNode'
    var_1 = '\x0cX}11'
    var_2 = var_0.insert(var_1, var_1)
    var_3 = var_0.insert(var_1, var_2)