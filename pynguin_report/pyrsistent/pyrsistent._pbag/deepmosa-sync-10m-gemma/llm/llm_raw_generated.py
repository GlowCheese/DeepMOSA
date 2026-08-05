####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_remove_existing_element_single_count():
    counts = pmap({1: 1})
    bag = PBag(counts)
    new_bag = bag.remove(1)
    assert 1 not in new_bag
    assert len(new_bag) == 0

def test_remove_existing_element_multiple_count():
    counts = pmap({1: 2})
    bag = PBag(counts)
    new_bag = bag.remove(1)
    assert new_bag.count(1) == 1
    assert len(new_bag) == 1

def test_remove_non_existent_element_raises_keyerror():
    counts = pmap({1: 1})
    bag = PBag(counts)
    try:
        bag.remove(2)
    except KeyError as e:
        assert e.args[0] == 2
    else:
        raise AssertionError("KeyError not raised")

def test_remove_persistence():
    counts = pmap({1: 1})
    bag = PBag(counts)
    new_bag = bag.remove(1)
    assert 1 in bag._counts
    assert 1 not in new_bag._counts


