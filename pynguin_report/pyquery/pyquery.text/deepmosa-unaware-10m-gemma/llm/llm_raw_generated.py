####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Case 1: Simple inline tag (no newlines)
    node1 = MockNode('span', text='Hello')
    assert extract_text_array(node1) == ['Hello']

    # Case 2: Block level tag (should trigger None for artificial newline)
    node2 = MockNode('div', text='Start')
    child2 = MockNode('b', text='Bold')
    node2.children = [child2]
    # Expected: [None, 'Start', 'Bold', None] -> stripped -> ['Start', 'Bold']
    # Actually, the logic adds None at start and end if not in INLINE_TAGS
    assert extract_text_array(node2) == ['Start', 'Bold']

    # Case 3: Separator tag (br)
    node3 = MockNode('br')
    node4 = MockNode('p', text='Text')
    node5 = MockNode('br')
    root3 = MockNode('div', children=[node3, node4, node5])
    # br adds True. p adds None. 
    # Result array before stripping: [None, True, 'Text', True, None]
    assert extract_text_array(root3) == ['Text']

    # Case 4: Nested structure with tails and text
    # <div><p>A<span>B</span>C</p></div>
    span = MockNode('span', text='B')
    p = MockNode('p', text='A', children=[span])
    p.tail = 'C' # Note: tail is usually on the child, but for testing structure:
    # Let's build it correctly: 
    # p (text='A') -> child span (text='B', tail='C')
    span_node = MockNode('span', text='B', tail='C')
    p_node = MockNode('p', text='A', children=[span_node])
    root_node = MockNode('div', children=[p_node])
    
    # extract_text_array(p_node) -> [None, 'A', 'B', 'C', None] -> ['A', 'B', 'C']
    assert extract_text_array(root_node) == ['A', 'B', 'C']

    # Case 5: Testing squash_artifical_nl=False
    # Ensure None values are preserved when requested
    node6 = MockNode('div', text='Block')
    res = extract_text_array(node6, squash_artifical_nl=False)
    assert None in res

    # Case 6: Empty node
    node7 = MockNode('div')
    assert extract_text_array(node7) == []

    # Case 7: Node with callable tag (simulating complex objects)
    class CallableTag:
        def __call__(self): pass
    node8 = MagicMock()
    node8.tag = CallableTag()
    assert extract_text_array(node8) == ''

    # Case 8: Complex mix of inline and block
    # <div><p>Text <b>Bold</b></p></div>
    b_node = MockNode('b', text='Bold')
    p_node = MockNode('p', text='Text ', children=[b_node])
    root_mix = MockNode('div', children=[p_node])
    # p is block -> [None, 'Text ', 'Bold', None]
    # div is block -> [None, ... , None]
    # result should be ['Text ', 'Bold']
    assert extract_text_array(root_mix) == ['Text ', 'Bold']
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag=None, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Test Case 1: Simple inline element (span)
    node1 = MockNode(tag='span', text='hello')
    assert extract_text_array(node1) == ['hello']

    # Test Case 2: Block element (div) with artificial newline logic
    # div is not in INLINE_TAGS, so it adds None at start and end
    node2 = MockNode(tag='div', text='content')
    # result should be [None, 'content', None], then stripped by _strip_artifical_nl
    assert extract_text_array(node2) == ['content']

    # Test Case 3: Separator element (br)
    node3 = MockNode(tag='br')
    # br is in SEPARATORS, so it adds True
    assert extract_text_array(node3) == [True]

    # Test Case 4: Nested structure with text and tails
    # <div><p>Text<span>Inner</span>Tail</p></div>
    child_span = MockNode(tag='span', text='Inner')
    child_p = MockNode(tag='p', text='Text', children=[child_span], tail='Tail')
    root_div = MockNode(tag='div', children=[child_p])
    
    # trace: 
    # child_p (p is not inline) -> [None, 'Text', ... (children), 'Tail', None]
    # child_span (span is inline) -> ['Inner']
    # result before strip: [None, 'Text', 'Inner', 'Tail', None, None]
    # after squash/strip: ['Text', 'Inner', 'tail'] -> simplified logic depends on depth
    result = extract_text_array(root_div)
    assert 'Text' in result
    assert 'Inner' in result
    assert 'Tail' in result

    # Test Case 5: Squash artificial newlines disabled
    node4 = MockNode(tag='div', text='A')
    # With squash=False, we expect the None markers to remain if not stripped
    # But extract_text_array calls _strip_artifical_nl by default.
    # Let's test with strip_artifical_nl=False manually via a mock-like setup
    node5 = MockNode(tag='div', text='A')
    res = extract_text_array(node5, strip_artifical_nl=False)
    assert res == [None, 'A', None]

    # Test Case 6: Empty node
    node6 = MockNode(tag='div')
    assert extract_text_array(node6) == []

    # Test Case 7: Node with callable tag (e.g., a function object)
    node7 = MagicMock()
    node7.tag = lambda: True
    assert extract_text_array(node7) == ''

    # Test Case 8: Multiple elements in sequence
    node8 = MockNode(tag='div', children=[
        MockNode(tag='b', text='Bold'),
        MockNode(tag='br'),
        MockNode(tag='i', text='Italic')
    ])
    # b is inline, br is separator (True), i is inline. 
    # div adds None at start/end.
    # Process: [None, 'Bold', True, 'Italic', None] -> stripped -> ['Bold', True, 'Italic']
    assert extract_text_array(node8) == ['Bold', True, 'Italic']

    # Test Case 9: Text and Tails combined
    node9 = MockNode(tag='div', text='Start')
    child9 = MockNode(tag='span', text='Middle')
    child9.tail = 'End'
    node9.children = [child9]
    # div (None) -> Start -> span (Middle) -> End (tail of span) -> div (None)
    assert extract_text_array(node9) == ['Start', 'Middle', 'End']
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag=None, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self._children = children or []

    def getchildren(self):
        return self._children

def test_extract_text_array():
    # Test Case 1: Simple Inline Tag (no separators/newlines)
    node1 = MockNode(tag='span', text='hello')
    assert extract_text_array(node1) == ['hello']

    # Test Case 2: Block Tag (should introduce None as artificial newline)
    node2 = MockNode(tag='div', text='content')
    # div is not in INLINE_TAGS, so it adds None at start and end
    assert extract_text_array(node2) == [None, 'content', None]

    # Test Case 3: Separator Tag (br should introduce True)
    node3 = MockNode(tag='br')
    assert extract_text_array(node3) == [True]

    # Test Case 4: Nested Structure with Tail and Children
    # <div><span>A</span>B</div>
    child_span = MockNode(tag='span', text='A')
    parent_div = MockNode(tag='div', children=[child_span])
    child_span.tail = 'B'
    # Expected: [None (div start), None (span start), 'A', 'B', None (div end)]
    # After _squash_artifical_nl and _strip_artifical_nl: ['A', 'B']
    assert extract_text_array(parent_div) == ['A', 'B']

    # Test Case 5: Complex structure with multiple blocks
    # <div><p>P1</p><p>P2</p></div>
    p1 = MockNode(tag='p', text='P1')
    p2 = MockNode(tag='p', text='P2')
    root = MockNode(tag='div', children=[p1, p2])
    # Logic: 
    # root starts with None
    # p1 adds None, 'P1', then None (at end of p1)
    # p2 adds None, 'P2', then None (at end of p2)
    # root ends with None
    # Resulting array before stripping: [None, None, 'P1', None, None, 'P2', None, None]
    # After _squash_artifical_nl: [None, 'P1', None, 'P2', None]
    # After _strip_artifical_nl: ['P1', None, 'P2'] -> wait, strip removes leading/trailing Nones
    # Let's verify the exact behavior of the provided _strip_artifical_nl logic.
    # It finds first string index and last string index. 
    # For [None, 'P1', None, 'P2', None], start_idx=1, end_idx=1 (index of 'P2' from end)
    # Result: ['P1', None, 'P2']
    assert extract_text_array(root) == ['P1', None, 'P2']

    # Test Case 6: Function with callable tag (should return empty string)
    node6 = MockNode(tag=lambda x: x)
    assert extract_text_array(node6) == ''

    # Test Case 7: Empty node
    node7 = MockNode(tag='div')
    assert extract_text_array(node7) == []

    # Test Case 8: Mixed inline and block with whitespace
    # <div><span> </span></div>
    node8 = MockNode(tag='div', children=[MockNode(tag='span', text='  ')])
    # [None, None, '  ', None, None] -> squash -> [None, '  ', None] -> strip -> ['  ']
    assert extract_text_array(node8) == ['  ']

    # Test Case 9: Testing squash_artifical_nl=False
    # Ensure that the raw array is returned without collapsing multiple Nones
    node9 = MockNode(tag='div', children=[MockNode(tag='p', text='A')])
    # Raw: [None, None, 'A', None, None]
    assert extract_text_array(node9, squash_artifical_nl=False) == [None, None, 'A', None, None]

    # Test Case 10: Testing strip_artifical_nl=False
    # Ensure that leading/trailing Nones are preserved
    node10 = MockNode(tag='div', text='A')
    assert extract_text_array(node10, strip_artifical_nl=False) == [None, 'A', None]
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

class MockNode:
    def __init__(self, tag, text=None, children=None, tail=None):
        self.tag = tag
        self.text = text
        self.children = children or []
        self.tail = tail

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Test 1: Simple text node (no tags)
    root1 = MockNode('div', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Test 2: Inline tags should not add None/True separators
    root2 = MockNode('p', text='Start ', children=[
        MockNode('span', text='Middle', tail=' End')
    ])
    assert extract_text_array(root2) == ['Start ', 'Middle', ' End']

    # Test 3: Block tags should add None (artificial newline)
    root3 = MockNode('div', text='Outer', children=[
        MockNode('p', text='Inner')
    ])
    # extract_text_array adds None at start/end for non-inline, and middle if not inline
    # The logic: [None (start), 'Outer', None (middle), None (start child), 'Inner', None (end child), None (end)]
    # After _squash_artifical_nl and _strip_artifical_nl:
    assert extract_text_array(root3) == ['Outer', 'Inner']

    # Test 4: Separator tag (br) should add True
    root4 = MockNode('div', text='A', children=[
        MockNode('br'),
        MockNode('b', text='B')
    ])
    assert extract_text_array(root4) == ['A', True, 'B']

    # Test 5: Complex nested structure with tails and whitespace
    root5 = MockNode('div', text='Top', children=[
        MockNode('p', text='Para1', children=[
            MockNode('a', text='Link')
        ], tail=' Tail'),
        MockNode('span', text='SpanText')
    ])
    # Expected behavior: 
    # 'Top' is in a block tag -> adds None.
    # child 'p' is block -> adds None. 
    # 'Para1' is text.
    # child 'a' is inline -> no extra separator.
    # tail of 'a' is None (not provided).
    # tail of 'p' is ' Tail'.
    # 'SpanText' is in a span (inline) -> no extra separator at start/end of node.
    # Resulting parts before stripping: [None, 'Top', None, 'Para1', ' Tail', 'SpanText']
    # After squash and strip: ['Top', 'Para1', ' Tail', 'SpanText']
    result = extract_text_array(root5)
    assert 'Top' in result
    assert 'Para1' in result
    assert ' Tail' in result

    # Test 6: Empty node
    root6 = MockNode('div')
    assert extract_text_array(root6) == []

    # Test 7: Callable tag (like a function/renderer error case)
    root7 = MockNode(lambda: None)
    assert extract_text_array(root7) == ''

    # Test 8: Verifying _squash_artifical_nl logic via extraction
    # If we have multiple Nones, they should become one None.
    root8 = MockNode('div', text='A', children=[
        MockNode('p', text='B'), # Block inside block
        MockNode('p', text='C')  # Block inside block
    ])
    # Part list: [None, 'A', None, None, 'B', None, None, 'C', None]
    # Squashed: [None, 'A', None, 'B', None, 'C', None]
    # Stripped: ['A', 'B', 'C']
    assert extract_text_array(root8) == ['A', 'B', 'C']

    # Test 9: Testing with squash_artifical_nl=False
    root9 = MockNode('div', text='A', children=[MockNode('p', text='B')])
    # Without squash/strip, the raw None values from the recursion remain
    raw_result = extract_text_array(root9, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in raw_result
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

class MockElement:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Test case 1: Simple inline element
    el1 = MockElement('span', text='Hello')
    assert extract_text_array(el1) == ['Hello']

    # Test case 2: Block element (should introduce None for artificial newline)
    el2 = MockElement('div', text='Start', children=[
        MockElement('b', text='Bold')
    ])
    # div is not in INLINE_TAGS, so it adds None at start and end
    # Result should be [None, 'Start', 'Bold', None] -> stripped to ['Start', 'Bold']
    assert extract_text_array(el2) == ['Start', 'Bold']

    # Test case 3: Separator element (br)
    el3 = MockElement('div', text='A', children=[
        MockElement('br'),
        MockElement('span', text='B')
    ])
    # br is in SEPARATORS, so it adds True
    assert extract_text_array(el3) == ['A', True, 'B']

    # Test case 4: Nested structure with tails
    el4 = MockElement('div', text='Outer', children=[
        MockElement('span', text='Inner', tail=' Tail')
    ])
    assert extract_text_array(el4) == ['Outer', 'Inner', ' Tail']

    # Test case 5: Complex structure with multiple blocks and inline tags
    # Structure: <div>Text <span>Inline</span><br>Next</div>
    el5 = MockElement('div', text='Text ', children=[
        MockElement('span', text='Inline'),
        MockElement('br'),
        Mockelse := MockElement('p', text='Paragraph')
    ])
    # Inside p: [None, 'Paragraph', None]
    # br adds True
    # div adds None at start/end
    # Final array after squash/strip logic: ['Text ', 'Inline', True, 'Paragraph']
    result = extract_text_array(el5)
    assert True in result
    assert 'Text ' in result
    assert 'Inline' in result
    assert 'Paragraph' in result

    # Test case 6: Function with callable tag (like a function object used as a node)
    class CallableTag:
        def __call__(self): pass
    el6 = MockElement(CallableTag())
    assert extract_text_array(el6) == ''

    # Test case 7: Empty element
    el7 = MockElement('div')
    assert extract_text_array(el7) == []

    # Test case 8: Whitespace handling in text
    el8 = MockElement('span', text='Line\nBreak\tTab')
    # Note: extract_text_array itself doesn't call squash_html_whitespace, 
    # but it preserves the strings. Testing preservation of raw text.
    assert extract_text_array(el8) == ['Line\nBreak\tTab']

def Mockelse := MockElement # Helper for complex nesting if needed
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockElement:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Test Case 1: Simple inline tag (no newlines/separators)
    span = MockElement('span', text='Hello')
    assert extract_text_array(span) == ['Hello']

    # Test Case 2: Block element (should add None for artificial newline)
    div = MockElement('div', text='Top')
    div.children = [MockElement('p', text='Middle')]
    # Expected: [None, 'Top', None, 'Middle', None] 
    # After squash_artifical_nl and strip_artifical_nl logic: ['Top', 'Middle']
    assert extract_text_array(div) == ['Top', 'Middle']

    # Test Case 3: Separator tag (br)
    br_node = MockElement('br')
    assert extract_text_array(br_node) == [True]

    # Test Case 4: Nested structure with tail text and inline tags
    # <div><p>Text <b>bold</b> tail</p></div>
    bold = MockElement('b', text='bold')
    p = MockElement('p', text='Text ', children=[bold])
    p.tail = ' tail'
    div = MockElement('div', text='', children=[p])
    
    # Trace:
    # p starts with None (block)
    # p.text is 'Text '
    # child b: ['b', 'bold']
    # p.tail is ' tail'
    # p ends with None
    # div starts with None, ends with None
    # Result after processing should strip outer Nones and merge
    result = extract_text_array(div)
    assert 'Text ' in result
    assert 'bold' in result
    assert ' tail' in result

    # Test Case 5: Complex whitespace handling
    complex_node = MockElement('div', text='  spaced  \n  text  ')
    result = extract_text_array(complex_node)
    # WHITESPACE_RE replaces \n and multiple spaces with single space ' '
    assert result == ['  spaced   text  ']

    # Test Case 6: Function object tag (should return empty string)
    func_node = MagicMock()
    func_node.tag = lambda x: x
    assert extract_text_array(func_node) == ''

    # Test Case 7: Empty element
    empty = MockElement('div')
    # Result should be [] because strip_artifical_nl handles empty sequences
    assert extract_text_array(empty) == []

    # Test Case 8: Testing squash_artifical_nl=False parameter
    # This prevents the None values from being squashed/stripped in the logic
    node = MockElement('div', text='A')
    node.children = [MockElement('span', text='B')]
    # With False, we expect the raw list including Nones
    res_no_squash = extract_text_array(node, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in res_no_squash
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test Case 1: Simple text node
    node1 = MockNode('p', text='Hello')
    assert extract_text(node1) == 'Hello'

    # Test Case 2: Nested block elements (div > p)
    # Should result in a newline between blocks because div is not in INLINE_TAGS
    node2 = MockNode('div', children=[
        MockNode('p', text='Line 1'),
        MockNode('p', text='Line 2')
    ])
    assert extract_text(node2) == 'Line 1\nLine 2'

    # Test Case 3: Inline elements (span inside p)
    # Should not insert newlines between inline elements
    node3 = MockNode('p', children=[
        MockNode('span', text='Part 1'),
        MockNode('b', text=' Part 2')
    ])
    assert extract_text(node3) == 'Part 1 Part 2'

    # Test Case 4: Separator element (br)
    # br is in SEPARATORS, should trigger sep_symbol (default \n)
    node4 = MockNode('div', children=[
        MockNode('span', text='First'),
        MockNode('br'),
        MockNode('span', text='Second')
    ])
    assert extract_text(node4) == 'First\nSecond'

    # Test Case 5: Whitespace squashing
    # Multiple spaces/newlines in text should be collapsed to single space
    node5 = MockNode('p', text='Too    many \n\n spaces')
    assert extract_text(node5) == 'Too many spaces'

    # Test Case 6: Complex structure with tails and mixed content
    # div (block) -> p (block) [text="A", child=span(text="B"), tail=" C"]
    node6 = MockNode('div', children=[
        MockNode('p', text='A', children=[
            MockNode('span', text='B')
        ], tail=' C')
    ])
    # Expected: 'A' + newline (from p end) + ' B' (from span/tail) -> flattened to 'A\n B' 
    # but since it's one block and the tail is part of the flow, let's check logic.
    # The function should strip artificial newlines at edges.
    assert extract_text(node6) == 'A\n B'

    # Test Case 7: Custom symbols
    node7 = MockNode('div', children=[
        MockNode('p', text='Start'),
        MockNode('p', text='End')
    ])
    assert extract_text(node7, block_symbol=' | ', sep_symbol=' -> ') == 'Start | End'

    # Test Case 8: Empty/None content
    node8 = MockNode('div', children=[
        MockNode('p', text=None),
        MockNode('p', text='')
    ])
    assert extract_text(node8) == ''

    # Test Case 9: Function as tag (callable)
    def dummy_tag(): pass
    node9 = MockNode(dummy_tag)
    assert extract_text(node9) == ''

    # Test Case 10: Testing _merge_original_parts via extract_text
    # Ensuring that strings are merged and whitespace is squashed
    node10 = MockNode('p', text='  leading')
    node10.children.append(MockNode('span', text='trailing  '))
    assert extract_text(node10) == 'leading trailing'
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockElement:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test case 1: Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Test case 2: Nested block elements with newlines
    # <div><p>Part 1</p><p>Part 2</p></div> -> "Part 1\nPart 2"
    child1 = MockElement('p', text='Part 1')
    child2 = MockElement('p', text='Part 2')
    root2 = MockElement('div', children=[child1, child2])
    assert extract_text(root2) == 'Part 1\nPart 2'

    # Test case 3: Inline elements (no added newlines/separators)
    # <div><span>Hello</span> <b>World</b></div> -> "Hello World"
    span = MockElement('span', text='Hello')
    b = MockElement('b', text='World')
    span.tail = ' '
    root3 = MockElement('div', children=[span, b])
    assert extract_text(root3) == 'Hello World'

    # Test case 4: Separator element (br tag)
    # <div>Line 1<br>Line 2</div> -> "Line 1\nLine 2"
    br = MockElement('br')
    child_br = MockElement('span', text='Line 2')
    br.tail = None # br is a separator, its presence triggers True in array
    root4 = MockElement('div', children=[MockElement('span', text='Line 1'), br, child_br])
    # Note: extract_text_array adds True for 'br' tag
    assert extract_text(root4) == 'Line 1\nLine 2'

    # Test case 5: Whitespace squashing
    # <div>  Too   much   space  </div> -> "Too much space"
    root5 = MockElement('div', text='  Too   much   space  ')
    assert extract_text(root5) == 'Too much space'

    # Test case 6: Complex structure with tails and mixed tags
    # <div>
    #   <div>Block</div>
    #   <span>Inline</span> tail
    # </div>
    inner_block = MockElement('div', text='Block')
    inline_span = MockElement('span', text='Inline')
    inline_span.tail = ' tail'
    root6 = MockElement('div', children=[inner_block, inline_span])
    # Expected: Block\nInline tail
    assert extract_text(root6) == 'Block\nInline tail'

    # Test case 7: Custom separators and symbols
    root7 = MockElement('div', children=[MockElement('p', text='A'), MockElement('p', text='B')])
    assert extract_text(root7, block_symbol=' | ', sep_symbol=' - ') == 'A - B'

    # Test case 8: Empty elements
    root8 = MockElement('div', children=[MockElement('p'), MockElement('span')])
    assert extract_text(root8) == ''

    # Test case 9: Elements with only tails
    child_tail = MockElement('span', text='Start')
    child_tail.tail = ' End'
    root9 = MockElement('div', children=[child_tail])
    assert extract_text(root9) == 'Start End'

    # Test case 10: Functionality of squash_space=False
    root10 = MockElement('div', text='  spaced  ')
    assert extract_text(root10, squash_space=False) == '  spaced  '
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockElement:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test Case 1: Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Test Case 2: Nested block elements with newlines (None in array)
    # <div><p>Line 1</p><p>Line 2</p></div>
    p1 = MockElement('p', text='Line 1')
    p2 = MockElement('p', text='Line 2')
    root2 = MockElement('div', children=[p1, p2])
    # extract_text inserts block_symbol (\n) for non-inline tags
    assert extract_text(root2) == 'Line 1\nLine 2'

    # Test Case 3: Inline elements (no added newlines)
    # <div><span>Part 1</span><span>Part 2</span></div>
    span1 = MockElement('span', text='Part 1')
    span2 = MockElement('span', text='Part 2')
    root3 = MockElement('div', children=[span1, span2])
    assert extract_text(root3) == 'Part 1Part 2'

    # Test Case 4: Separators (br tag)
    # <div>Line 1<br>Line 2</div>
    br = MockElement('br')
    p1_br = MockElement('p', text='Line 1')
    p1_br.tail = None # tail is handled via children loop logic
    # To simulate: <div> <p>Line 1</p> <br/> <p>Line 2</p> </div>
    # The 'br' tag returns True in extract_text_array, which becomes sep_symbol (\n)
    root4 = MockElement('div', children=[
        MockElement('p', text='Line 1'),
        br,
        MockElement('p', text='Line 2')
    ])
    # br returns True -> uses sep_symbol (\n). p tags return None -> uses block_symbol (\n)
    assert extract_text(root4) == 'Line 1\n\nLine 2'.replace('\n\n', '\n').strip()

    # Test Case 5: Whitespace squashing
    # <div>  Word   \t  with \n spaces  </div>
    root5 = MockElement('div', text='  Word   \t  with \n spaces  ')
    assert extract_text(root5) == 'Word with spaces'

    # Test Case 6: Complex structure with tails and nested blocks
    # <div><p>Start<br>Middle</p>End</div>
    # Note: br is a separator (True), p is block (None)
    br_tag = MockElement('br')
    p_tag = MockElement('p', text='Start')
    p_tag.tail = None # The content after </p> is the tail of p_tag
    # Reconstructing: <div><p>Start</p><br/><p>Middle</p>End</div>
    # Actually, in lxml/DOM, 'End' would be the .tail of the <p> tag.
    p_inner = MockElement('p', text='Start')
    br_tag = MockElement('br')
    p_second = MockElement('p', text='Middle')
    root6 = MockElement('div', children=[p_inner, br_tag, p_second])
    p_second.tail = 'End'
    # Expected: 'Start' (from p1) + '\n' (from br/sep) + 'Middle' (from p2) + 'End' (tail of p2)
    # Since p is block, it adds None (\n) at start and end. 
    # Final result depends on squash_space=True stripping edges.
    result = extract_text(root6)
    assert 'Start' in result
    assert 'Middle' in result
    assert 'End' in result

    # Test Case 7: Custom symbols
    root7 = MockElement('div', children=[MockElement('p', text='A'), MockElement('p', text='B')])
    assert extract_text(root7, block_symbol='|', sep_symbol='-') == 'A|B'

    # Test Case 8: Empty element
    root8 = MockElement('div')
    assert extract_text(root8) == ''

    # Test Case 9: Element with callable tag (e.g. custom function in some frameworks)
    class CallableTagElement:
        def __init__(self):
            self.tag = lambda x: x
        def getchildren(self): return []
    root9 = CallableTagElement()
    assert extract_text(root9) == ''
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Test Case 1: Simple inline tag (no separators/newlines)
    node1 = MockNode('span', text='Hello')
    assert extract_text_array(node1) == ['Hello']

    # Test Case 2: Block element (should introduce None for newlines)
    node2 = MockNode('div', text='Top')
    child2 = MockNode('b', text='Middle')
    tail2 = ' Bottom'
    node2.children.append(child2)
    # Logic: div is not in INLINE_TAGS -> starts with None, ends with None
    # b is inline -> no None added by tag itself
    # Result expected: [None, 'Top', 'Middle', ' Bottom', None]
    # After _squash_artifical_nl: [None, 'Top', 'Middle', ' Bottom', None] 
    # (Note: squash doesn't remove leading/trailing None, it just prevents duplicates)
    # After _strip_artifical_nl: ['Top', 'Middle', ' Bottom']
    assert extract_text_array(node2) == ['Top', 'append_child_logic_result_depends_on_structure'] 
    # Let's re-verify the logic specifically for block elements:
    # node2 (div): tag not in INLINE_TAGS -> r.append(None)
    # node2.text ('Top'): r.append('Top')
    # child2 (b): returns ['Middle'] (inline, no None added)
    # child2.tail (' Bottom'): r.append(' Bottom')
    # node2 end: tag not in INLINE_TAGS -> r.append(None)
    # Array before strip: [None, 'Top', 'Middle', ' Bottom', None]
    # After _strip_artifical_nl: ['Top', 'Middle', ' Bottom']
    assert extract_text_array(node2) == ['Top', 'Middle', ' Bottom']

    # Test Case 3: Separator tag (br)
    node3 = MockNode('br')
    # br is in SEPARATORS -> r.append(True)
    # Since it's in SEPARATORS, it doesn't trigger the "not in INLINE_TAGS" None logic
    assert extract_text_array(node3) == [True]

    # Test Case 4: Nested structure with complex whitespace
    # <div><p>Part 1</p><br>Part 2</div>
    root = MockNode('div', text='Start ')
    p = MockNode('p', text='Content')
    br = MockNode('br')
    tail_p = ' End'
    root.children.append(p)
    root.children.append(br)
    p.tail = tail_p # p is block, so it adds None at start/end of its scope

    # Trace:
    # root (div): [None, 'Start ']
    #   child p (p): [None, 'Content', ' End', None]
    #   child br (br): [True]
    #   root tail: (none)
    # root end: [None]
    # Combined internal: [None, 'Start ', None, 'Content', ' End', None, True, None]
    # _squash_artifical_nl: [None, 'Start ', None, 'Content', ' End', True, None]
    # _strip_artifical_nl: ['Start ', 'Content', ' End', True]
    
    res = extract_text_array(root)
    assert res == ['Start ', 'Content', ' End', True]

    # Test Case 5: Function as tag (should return empty string)
    node4 = MockNode(tag=lambda x: True)
    assert extract_text_array(node4) == ''

    # Test Case 6: Text with heavy whitespace (squash check)
    # Note: extract_text_array uses squash_artifical_nl=True by default for the logic
    node5 = MockNode('div', text='Line1\n\nLine2')
    # WHITESPACE_RE will turn \n\n into ' '
    assert extract_text_array(node5) == ['Line1 Line2']

    # Test Case 7: Empty node
    node6 = MockNode('div')
    # [None, None] -> squash -> [None] -> strip -> []
    assert extract_text_array(node6) == []
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockElement:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test case 1: Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_template_text_wrapper(root1) == 'Hello'

    # Test case 2: Nested block elements with whitespace handling
    # <div><p>Line 1</p><p>Line 2</p></div>
    p1 = MockElement('p', text='Line 1')
    p2 = MockElement('p', text='Line 2')
    root2 = MockElement('div', children=[p1, p2])
    # extract_text adds None (newline) for block tags
    assert extract_template_text_wrapper(root2) == 'Line 1\nLine 2'

    # Test case 3: Inline elements (should not trigger newlines)
    # <div><span>Part </span><b>Bold</b></div>
    span = MockElement('span', text='Part ')
    b = MockElement('b', text='Bold')
    root3 = MockElement('div', children=[span, b])
    assert extract_template_text_wrapper(root3) == 'Part Bold'

    # Test case 4: Separator tags (br)
    # <div>Line 1<br>Line 2</div>
    br = MockElement('br')
    p1_alt = MockElement('p', text='Line 1')
    p1_alt.tail = None # tail is empty
    # In the logic, br returns True (sep_symbol)
    root4 = MockElement('div', children=[p1_alt, br, MockElement('p', text='Line 2')])
    # Note: extract_text defaults sep_symbol to '\n'
    assert 'Line 1\nLine 2' in extract_template_text_wrapper(root4)

    # Test case 5: Complex structure with tails and mixed types
    # <div>Text<p>Inner</p>Tail</div>
    inner_p = MockElement('p', text='Inner')
    inner_p.tail = 'Tail'
    root5 = MockElement('div', text='Text', children=[inner_p])
    assert extract_templately_text_wrapper(root5) == 'Text\nInner\nTail'

    # Test case 6: Squashing whitespace
    root6 = MockElement('div', text='  Multiple   Spaces  ')
    assert extract_template_text_wrapper(root6) == 'Multiple Spaces'

    # Test case 7: Empty elements
    root7 = MockElement('div', children=[MockElement('span')])
    assert extract_template_text_wrapper(root7) == ''

def extract_template_text_wrapper(dom):
    """Helper to call the function with default parameters."""
    return extract_text(dom)

# Since I cannot include imports, I assume all functions are available in scope.
# To run this: pytest <filename>.py
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test Case 1: Simple text node
    node1 = MockNode('p', text='Hello')
    assert extract_text(node1) == 'Hello'

    # Test Case 2: Nested block elements (div > p > span)
    # Represents: <div><p><span>Inner</span></p></div>
    span = MockNode('span', text='Inner')
    p = MockNode('p', children=[span])
    div = MockNode('div', children=[p])
    # Expected behavior: block elements insert None (newlines)
    # div(None) -> p(None) -> span -> p(None) -> div(None)
    # Resulting string after joining with \n and stripping is 'Inner'
    assert extract_text(div) == 'Inner'

    # Test Case 3: Block elements with text and siblings
    # Represents: <div>Text<p>Para</p>More</div>
    p_node = MockNode('p', text='Para')
    div2 = MockNode('div', children=[p_node])
    div2.text = 'Text'
    p_node.tail = 'More'
    # Extraction: div starts with None, text 'Text', child p (None), 
    # p text 'Para', p tail 'More', div ends with None.
    assert extract_text(div2) == 'Text\nPara\nMore'

    # Test Case 4: Inline elements (no extra newlines)
    # Represents: <div><span>A</span><b>B</b></div>
    span_a = MockNode('span', text='A')
    b_b = MockNode('b', text='B')
    div3 = MockNode('div', children=[span_a, b_b])
    assert extract_text(div3) == 'AB'

    # Test Case 5: Separator element (br)
    # Represents: <div>Line1<br>Line2</div>
    br_node = MockNode('br')
    div4 = MockNode('div', children=[MockNode('p', text='Line1'), br_node, MockNode('p', text='Line2')])
    # Note: In the provided implementation, br triggers True (sep_symbol)
    assert extract_text(div4) == 'Line1\nLine2'

    # Test Case 6: Whitespace squashing
    # Represents: <div>  Too   much   space  </div>
    node5 = MockNode('div', text='  Too   much   space  ')
    assert extract_text(node5) == 'Too much space'

    # Test Case 7: Complex structure with mixed inline/block and tails
    # Represents: <div>Start<p>Middle<span>End</span></p>Tail</div>
    end_span = MockNode('span', text='End')
    mid_p = MockNode('p', text='Middle', children=[end_span])
    mid_p.tail = ' Middle' # testing tail processing
    start_div = MockNode('div', text='Start', children=[mid_p])
    start_div.tail = 'Tail'
    # The logic should merge these into a clean string
    result = extract_text(start_div)
    assert 'Start' in result
    assert 'Middle' in result
    assert 'End' in result
    assert 'Tail' in result

    # Test Case 8: Custom separators
    node6 = MockNode('div', children=[MockNode('p', text='A'), MockNode('br')])
    assert extract_text(node6, block_symbol='|', sep_symbol='|') == 'A|'
    # Note: strip_artifical_nl/squash_space might clean the trailing pipe depending on implementation 
    # of strip() in extract_text.

    # Test Case 9: Empty node
    node7 = MockNode('div')
    assert extract_text(node7) == ''

    # Test Case 10: Functionally callable tag (edge case in code)
    class CallableTag:
        def __call__(self): return True
    node8 = MockNode(tag=CallableTag())
    assert extract_text(node8) == ''
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag=None, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test Case 1: Simple Text Node
    node1 = MockNode(tag='p', text='Hello')
    assert extract_text(node1) == 'Hello'

    # Test Case 2: Nested Block Elements (div > p)
    # Should include artificial newlines (None) which become block_symbol (\n)
    node2 = MockNode(tag='div', children=[
        MockNode(tag='p', text='Line 1'),
        MockNode(tag='p', text='Line 2')
    ])
    # div is not in INLINE_TAGS -> adds None at start and end
    # p is not in INLINE_TAGS -> adds None at start and end
    # Resulting structure: [None, 'Line 1', None, None, 'Line 2', None]
    # Squashed/Stripped: ['Line 1', '\n', 'Line 2']
    assert extract_text(node2) == 'Line 1\nLine 2'

    # Test Case 3: Inline Elements (span inside div)
    # span is in INLINE_TAGS, so it should not trigger None/block symbols
    node3 = MockNode(tag='div', children=[
        MockNode(tag='p', children=[
            MockNode(tag='span', text='Inline')
        ], text='Start ')
    ])
    # Expected: 'Start Inline' (no extra newlines from span)
    assert extract_text(node3) == 'Start Inline'

    # Test Case 4: Separators (br tag)
    # br is in SEPARATORS, which returns True (sep_symbol)
    node4 = MockNode(tag='div', children=[
        MockNode(tag='p', text='Part A'),
        MockNode(tag='br'),
        MockNode(tag='p', text='Part B')
    ])
    # br returns True -> becomes sep_symbol (\n)
    assert extract_text(node4) == 'Part A\nPart B'

    # Test Case 5: Whitespace Squashing
    # Testing WHITESPACE_RE via squash_space=True (default)
    node5 = MockNode(tag='p', text='Too    many \t spaces')
    assert extract_text(node5) == 'Too many spaces'

    # Test Case 6: Tail Text handling
    # Tail of a child should be included in the stream
    node6 = MockNode(tag='div', children=[
        MockNode(tag='b', text='Bold', tail=' and BoldTail')
    ])
    assert extract_text(node6) == 'Bold and BoldTail'

    # Test Case 7: Custom Symbols
    # Testing block_symbol and sep_symbol arguments
    node7 = MockNode(tag='div', children=[
        MockNode(tag='p', text='A'),
        MockNode(tag='br'),
        MockNode(tag='p', text='B')
    ])
    assert extract_text(node7, block_symbol='|', sep_symbol='~') == 'A~B'

    # Test Case 8: Complex structure with mixed inline/block
    node8 = MockNode(tag='div', children=[
        MockNode(tag='h1', text='Title'),
        MockNode(tag='p', children=[
            MockNode(tag='strong', text='Important'),
            MockNode(tag='a', text=' Link', tail=' end.')
        ])
    ])
    # Title is block, p is block, strong/a are inline. 
    # Should result in 'Title\nImportant Link end.'
    assert extract_text(node8) == 'Title\nImportant Link end.'

    # Test Case 9: Empty node
    node9 = MockNode(tag='div', text=None, children=[])
    assert extract_text(node9) == ''

    # Test Case 10: Function with callable tag (e.g., some specialized DOM object)
    class CallableTagNode:
        def __init__(self):
            self.tag = lambda x: x
        def getchildren(self): return []
    
    node10 = CallableTagNode()
    assert extract_text(node10) == ''
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag=None, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test 1: Simple text node
    root1 = MockNode(tag='p', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Test 2: Nested block elements (div -> p)
    root2 = MockNode(tag='div', children=[
        MockNode(tag='p', text='Line 1'),
        MockNode(tag='p', text='Line 2')
    ])
    # Should insert None (block_symbol default '\n') between blocks
    assert extract_text(root2) == 'Line 1\nLine 2'

    # Test 3: Inline elements (span inside p)
    root3 = MockNode(tag='p', children=[
        MockNode(tag='span', text='Inside'),
        MockNode(tag='b', text='Bold')
    ])
    # Inline tags should not trigger newlines/None
    assert extract_text(root3) == 'InsideBold'

    # Test 4: Separator elements (br)
    root4 = MockNode(tag='p', children=[
        MockNode(tag='b', text='Part A'),
        MockNode(tag='br'),
        MockNode(tag='b', text='Part B')
    ])
    # br is in SEPARATORS, should use sep_symbol (default '\n')
    assert extract_text(root4) == 'Part A\nPart B'

    # Test 5: Tail text handling
    root5 = MockNode(tag='div', children=[
        MockNode(tag='span', text='Start', tail=' End')
    ])
    assert extract_text(root5) == 'Start End'

    # Test 6: Complex structure with whitespace squashing
    # Testing the squash_space=True (default) behavior on mixed content
    root6 = MockNode(tag='div', children=[
        MockNode(tag='p', text='  Space  '),
        MockNode(tag='p', text='More\nSpace')
    ])
    # WHITESPACE_RE should turn \n into ' '
    assert extract_text(root6) == 'Space\nMore Space'

    # Test 7: Custom symbols
    root7 = MockNode(tag='div', children=[
        MockNode(tag='p', text='A'),
        MockNode(tag='p', text='B')
    ])
    assert extract_text(root7, block_symbol=' | ', sep_symbol=' -> ') == 'A | B'

    # Test 8: Empty nodes
    root8 = MockNode(tag='div', children=[])
    assert extract_text(root8) == ''

    # Test 9: Handling of None text/tail
    root9 = MockNode(tag='p', text=None, children=[
        MockNode(tag='b', text='Only')
    ])
    assert extract_text(root9) == 'Only'

    # Test 10: Deeply nested structure
    root10 = MockNode(tag='div', children=[
        MockNode(tag='section', children=[
            MockNode(tag='p', children=[
                MockNode(tag='span', text='Deep')
            ])
        ])
    ])
    assert extract_text(root10) == 'Deep'

    # Test 11: Testing squash_space=False preserves original newlines in text
    root11 = MockNode(tag='p', text='Line\nBreak')
    # Note: WHITESPACE_RE is used inside extract_text_array via squash_artifical_nl logic
    # If we want to test the preservation of a raw \n, we look at how 
    # squash_html_whitespace behaves.
    assert extract_text(root11, squash_space=False) == 'Line\nBreak'

    # Test 12: Function as tag (edge case from code)
    root12 = MockNode(tag=lambda x: True)
    assert extract_text(root12) == ''
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockElement:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Test Case 1: Simple inline element (no None/True markers)
    el1 = MockElement('span', text='hello')
    assert extract_text_array(el1) == ['hello']

    # Test Case 2: Block element (should introduce None markers)
    el2 = MockElement('div', text='top', children=[
        MockElement('p', text='middle')
    ], tail='bottom')
    # div is block -> [None, 'top', None, 'middle', 'bottom', None]
    # strip_artifical_nl removes leading/trailing Nones
    # squash_artifical_nl merges consecutive Nones
    # Result should be ['top', 'middle', 'bottom'] or similar depending on internal logic
    # Let's trace: 
    # div start -> [None]
    # div text -> [None, 'top']
    # p child (recursive call with squash=False) -> [None, 'middle', None]
    # div tail -> [None, 'top', None, 'middle', None, 'bottom']
    # div end -> [None, 'top', None, 'middle', None, 'bottom', None]
    # After stripping/squashing: ['top', 'middle', 'bottom']
    assert extract_text_array(el2) == ['top', 'middle', 'bottom']

    # Test Case 3: Separator element (br tag -> True marker)
    el3 = MockElement('br')
    assert extract_text_array(el3) == [True]

    # Test Case 4: Nested structure with mixed inline and block
    # <div><a>link</a><p>text</p></div>
    child_a = MockElement('a', text='link')
    child_p = MockElement('p', text='text')
    root = MockElement('div', text='start', children=[child_a, child_p], tail='end')
    
    # Trace:
    # root is block -> [None]
    # root.text -> [None, 'start']
    # child_a (inline) -> ['link']
    # child_a.tail? None
    # child_p (block) -> [None, 'text', None]
    # child_p.tail? None
    # root.tail -> [None, 'start', 'link', None, 'text', None, 'end']
    # root end -> [None, 'start', 'link', None, 'text', None, 'end', None]
    # After processing: ['start', 'link', 'text', 'end']
    assert extract_text_array(root) == ['start', 'link', 'text', 'end']

    # Test Case 5: Callable tag (e.g., Comment or ProcessingInstruction)
    el4 = MockElement('comment')
    el4.tag = lambda: True # Simulate callable tag
    assert extract_text_array(el4) == ''

    # Test Case 6: Whitespace handling within text
    el5 = MockElement('span', text='hello\n\nworld')
    # Note: extract_text_array doesn't call squash_html_whitespace itself, 
    # but it depends on the input. We test that text is preserved as is in array mode
    assert extract_text_array(el5) == ['hello\n\nworld']

    # Test Case 7: Empty element
    el6 = MockElement('div')
    # [None, None] -> squash -> [None] -> strip -> []
    assert extract_text_array(el6) == []

    # Test Case 8: Testing specific flags (squash_artifical_nl=False)
    # This prevents the internal merging of Nones
    el7 = MockElement('div', text='a', children=[MockElement('b', text='b')], tail='c')
    # Without squash, we expect the raw markers to persist in the list
    res = extract_text_array(el7, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in res
    assert 'a' in res
    assert 'b' in res
    assert 'c' in res
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag=None, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test Case 1: Simple text node
    node1 = MockNode(tag='p', text='Hello')
    assert extract_text(node1) == 'Hello'

    # Test Case 2: Nested block elements (div with p)
    # Expected behavior: None is inserted at start and end of block tags
    # Resulting in \nHello\n -> stripped to Hello
    node2 = MockNode(tag='div', children=[
        MockNode(tag='p', text='Hello')
    ])
    assert extract_text(node2) == 'Hello'

    # Test Case 3: Inline elements (span inside p)
    # Inline tags do not trigger None/newline insertion
    node3 = MockNode(tag='p', children=[
        MockNode(tag='span', text='Inner'),
        MockNode(tag='b', text='Bold')
    ])
    assert extract_text(node3) == 'InnerBold'

    # Test Case 4: Separator elements (br)
    # br tag triggers True which maps to sep_symbol (\n)
    node4 = MockNode(tag='p', children=[
        MockNode(tag='span', text='Part1'),
        MockNode(tag='br'),
        MockNode(tag='span', text='Part2')
    ])
    assert extract_text(node4) == 'Part1\nPart2'

    # Test Case 5: Handling tails (text after a tag)
    node5 = MockNode(tag='div', children=[
        MockNode(tag='span', text='Start', tail=' End')
    ])
    assert extract_text(node5) == 'Start End'

    # Test Case 6: Whitespace squashing
    node6 = MockNode(tag='p', text='Line1\n\n\nLine2')
    assert extract_text(node6) == 'Line1 Line2'

    # Test Case 7: Complex structure with block and inline mix
    # div (block) -> p (block) -> span (inline) + tail
    node7 = MockNode(tag='div', children=[
        MockNode(tag='p', children=[
            MockNode(tag='span', text='Hello', tail=' World')
        ], tail='!')
    ])
    # Logic: [None, 'Hello', ' World', '!', None] -> squashed/stripped -> 'Hello World!'
    assert extract_text(node7) == 'Hello World!'

    # Test Case 8: Custom symbols
    node8 = MockNode(tag='div', children=[
        MockNode(tag='p', text='A'),
        MockNode(tag='br'),
        MockNode(tag='p', text='B')
    ])
    assert extract_text(node8, block_symbol='|', sep_symbol='~') == 'A~B'

    # Test Case 9: Empty node
    node9 = MockNode(tag='div', children=[])
    assert extract_text(node9) == ''

    # Test Case 10: Function/Callable tag (should return empty string)
    node10 = MockNode(tag=lambda x: x)
    assert extract_text(node10) == ''

    # Test Case 11: Text with multiple spaces and newlines
    node11 = MockNode(tag='p', text='  Too   many \t spaces  ')
    assert extract_text(node11) == 'Too many spaces'
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockElement:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Test Case 1: Simple inline element (no newlines/separators)
    el1 = MockElement('span', text='hello')
    assert extract_text_array(el1) == ['hello']

    # Test Case 2: Block element (should introduce None for artificial newline)
    el2 = MockElement('div', text='content')
    # div is not in INLINE_TAGS and not in SEPARATORS, so it adds None at start and end
    assert extract_text_array(el2) == ['content']

    # Test Case 3: Nested structure with tail and children
    # <div><p>Part 1<span>Part 2</span>Tail</p></div>
    child_span = MockElement('span', text='Part 2')
    child_p = MockElement('p', text='Part 1', children=[child_span], tail='Tail')
    root_div = MockElement('div', children=[child_p])
    
    # Manual trace of extract_text_array(root_div):
    # root_div is div (block) -> [None]
    # child_p is p (block) -> [None, 'Part 1']
    # child_span is span (inline) -> ['Part 2']
    # tail of child_span is None
    # tail of child_p is 'Tail'
    # end of root_div (block) -> [None]
    # Result before squash/strip: [None, None, 'Part 1', 'Part 2', 'Tail', None]
    # After _squash_artifical_nl: [None, 'Part 1', 'Part 2', 'Tail', None]
    # After _strip_artifical_nl: ['Part 1', 'Part 2', 'Tail']
    assert extract_text_array(root_div) == ['Part 1', 'Part 2', 'Tail']

    # Test Case 4: Separator element (br)
    el3 = MockElement('br')
    # br is in SEPARATORS, so it appends True
    assert extract_text_array(el3) == [True]

    # Test Case 5: Empty element
    el4 = MockElement('div')
    assert extract_text_array(el4) == []

    # Test Case 6: Element with callable tag (should return '')
    class CallableTag:
        def __call__(self): pass
    el5 = MockElement(CallableTag())
    assert extract_text_array(el5) == ''

    # Test Case 7: Complex whitespace handling via squash_artifical_nl=False/True
    # Testing the logic of _squash_artifical_nl within the function call
    el6 = MockElement('div', text='A')
    # If we bypass the stripping/squashing by setting params to False
    # The raw array would be [None, 'A', None]
    res = extract_text_array(el6, squash_artifical_nl=False, strip_artifical_nl=False)
    assert res == [None, 'A', None]

    # Test Case 8: Verifying whitespace regex in function dependency
    el7 = MockElement('span', text='Line\nBreak\tTab')
    # squash_html_whitespace is called inside extract_text but 
    # the array extraction uses the raw text. 
    # Let's check if a span with weird whitespace preserves it in the array
    assert extract_text_array(el7) == ['Line\nBreak\tTab']

    # Test Case 9: Multiple None values (artificial newlines) being squashed
    el8 = MockElement('div', text='Start')
    child1 = MockElement('p', text='Middle')
    child2 = MockElement('p', text='End')
    el8.children = [child1, child2]
    # Array: [None (div), None (p1), 'Middle', None (p1 tail/p2 start), None (p2), None (div)]
    # Should squash to [None, 'Middle', 'End', None] and then strip to ['Middle', 'End']
    assert extract_text_array(el8) == ['Middle', 'End']
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockElement:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Test Case 1: Simple inline element
    span = MockElement('span', text='Hello')
    assert extract_text_array(span) == ['Hello']

    # Test Case 2: Block element with text and child
    child_b = MockElement('b', text='World')
    div = MockElement('div', text='Start ', children=[child_b])
    child_b.tail = ' End'
    # div is block -> adds None at start and end. 
    # b is inline -> no None added by tag logic itself, but child processing happens.
    # Expected: [None, 'Start ', 'World', ' End', None] -> squashed/stripped
    assert extract_text_array(div) == ['Start ', 'World', ' End']

    # Test Case 3: Separator element (br)
    br = MockElement('br')
    assert extract_text_array(br) == [True]

    # Test Case 4: Nested structure with artificial newlines
    # <p>Part 1<br/>Part 2</p>
    br_elem = MockElement('br')
    p_elem = MockElement('p', text='Part 1', children=[br_elem])
    br_elem.tail = 'Part 2'
    # p is block (None), br is separator (True)
    # Array before squash: [None, 'Part 1', True, 'Part 2', None]
    assert extract_text_array(p_elem) == ['Part 1', True, 'Part 2']

    # Test Case 5: Whitespace squashing in text
    space_elem = MockElement('span', text='Line\n\nNext  Line')
    assert extract_text_array(space_elem) == ['Line Next Line']

    # Test Case 6: Empty element
    empty = MockElement('div')
    assert extract_text_array(empty) == []

    # Test Case 7: Complex hierarchy
    # <div>Text <span>Inner</span> Tail</div>
    inner_span = MockElement('span', text='Inner')
    root_div = MockElement('div', text='Text ', children=[inner_span])
    inner_span.tail = ' Tail'
    # Root is block -> [None, 'Text ', 'Inner', ' Tail', None]
    # After squash/strip: ['Text ', 'Inner', ' Tail']
    assert extract_text_array(root_div) == ['Text ', 'Inner', ' Tail']

    # Test Case 8: Callable tag (e.g. custom function in some DOM implementations)
    callable_node = MagicMock()
    callable_node.tag = lambda x: x
    assert extract_text_array(callable_node) == ''

    # Test Case 9: Testing strip_artifical_nl=False logic
    # Ensure that when flags are false, the None values (NoneL) persist
    p_elem_raw = MockElement('p', text='A', children=[MockElement('b', text='B')])
    # p is block -> [None, 'A', 'B', None]
    result = extract_text_array(p_elem_raw, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'A', 'B', None]
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockElement:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test 1: Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_ext_text_wrapper(root1) == 'Hello'

    # Test 2: Nested block elements (should introduce newlines/None separators)
    # <div><p>Part 1</p><p>Part 2</p></div>
    p1 = MockElement('p', text='Part 1')
    p2 = MockElement('p', text='Part 2')
    root2 = MockElement('div', children=[p1, p2])
    # extract_text with default block_symbol='\n'
    assert extract_ext_text_wrapper(root2) == 'Part 1\nPart 2'

    # Test 3: Inline elements (should not introduce newlines)
    # <div><span>Hello</span> <b>World</b></div>
    span = MockElement('span', text='Hello')
    b = MockElement('b', text='World')
    root3 = MockElement('div', children=[span, MagicMock(tag='text_node_marker', text=' ', tail=None), b])
    # Note: The logic relies on the tail of span being ' '
    span.tail = ' '
    b.tail = None
    assert extract_ext_text_wrapper(root3) == 'Hello World'

    # Test 4: Separator element <br> (should introduce sep_symbol)
    # <div>Line 1<br>Line 2</div>
    br = MockElement('br')
    p_line1 = MockElement('p', text='Line 1')
    p_line1.tail = None # The br is a child of div, but its presence as a tag in SEPARATORS matters
    # To simulate <p>Line 1<br>Line 2</p>:
    br_node = MockElement('br')
    text_node = MockElement('p', text='Line 1')
    text_node.tail = 'Line 2' # This is tricky in lxml simulation, let's use children
    root4 = MockElement('div', children=[p_line1, br_node, MagicMock(tag='span', text='Line 2')])
    # Re-constructing specifically for the logic:
    # div -> [None, 'Line 1', True, 'Line 2', None]
    # result: Line 1\nLine 2 (where \n comes from sep_symbol)
    root4 = MockElement('div', children=[
        MockElement('p', text='Line 1'),
        br_node,
        MockElement('span', text='Line 2')
    ])
    # Note: In the code, if tag is in SEPARATORS, it appends True (sep_symbol)
    assert extract_ext_text_wrapper(root4, sep_symbol='\n') == 'Line 1\nLine 2'

    # Test 5: Whitespace squashing
    root5 = MockElement('div', text='  Too   much \n whitespace  ')
    assert extract_ext_text_wrapper(root5) == 'Too much whitespace'

    # Test 6: Complex structure
    # <div><p><b>Bold</b><i>Italic</i></p><span>End</span></div>
    b_node = MockElement('b', text='Bold')
    i_node = MockElement('i', text='Italic')
    b_node.tail = ''
    p_node = MockElement('p', children=[b_node, i_node])
    span_node = MockElement('span', text='End')
    root6 = MockElement('div', children=[p_node, span_node])
    # Expected: 'BoldItalic\nEnd' (p is block, span is inline)
    assert extract_ext_text_wrapper(root6) == 'BoldItalic\nEnd'

def extract_ext_text_wrapper(dom, block_symbol='\n', sep_symbol='\n', squash_space=True):
    """Helper to call the target function with same signature."""
    return extract_text(dom, block_symbol=block_symbol, sep_symbol=sep_symbol, squash_space=squash_space)

def test_extract_text_empty():
    root = MockElement('div')
    assert extract_text(root) == ''

def test_extract_text_with_custom_symbols():
    # <div><p>A</p><p>B</p></div> -> 'A|B'
    p1 = MockElement('p', text='A')
    p2 = MockElement('p', text='B')
    root = MockElement('div', children=[p1, p2])
    assert extract_text(root, block_symbol='|') == 'A|B'

def test_extract_text_no_squash():
    # Testing squash_space=False
    root = MockElement('div', text='  Space  ')
    # If squash_space is False, it shouldn't strip the outer whitespace or squash internal
    # However, extract_text calls _merge_original_parts which internally uses squash_html_whitespace.
    # So even with squash_space=False, WHITESPACE_RE still runs on parts.
    # But let's test if it preserves single newlines if they were in the text.
    root = MockElement('div', text='Line\nBreak')
    assert extract_text(root, squash_space=False) == 'Line\nBreak'
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag=None, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self._children = children or []

    def getchildren(self):
        return self._children

def test_extract_text():
    # Helper to create nodes quickly
    def node(tag, text=None, tail=None, children=None):
        return MockNode(tag=tag, text=text, tail=tail, children=children)

    # Case 1: Basic single text node
    root1 = node('div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Case 2: Nested block elements (should introduce newlines/None logic)
    # <div><p>Line 1</p><p>Line 2</p></div>
    root2 = node('div', children=[
        node('p', text='Line 1'),
        node('p', text='Line 2')
    ])
    # extract_text uses block_symbol='\n' for None entries
    assert extract_text(root2) == 'Line 1\nLine 2'

    # Case 3: Inline elements (should not introduce newlines)
    # <div><span>Part 1</span><span>Part 2</span></div>
    root3 = node('div', children=[
        node('span', text='Part 1'),
        node('span', text='Part 2')
    ])
    assert extract_text(root3) == 'Part 1Part 2'

    # Case 4: Separator element (br tag)
    # <div>Line 1<br>Line 2</div>
    root4 = node('div', children=[
        node('br'),
        node('span', text='Line 2') # br is in SEPARATORS, uses sep_symbol='\n'
    ])
    # Since br is a separator, it uses sep_symbol. 
    # Note: extract_text_array logic with SEPARATORS adds True
    assert extract_text(root4) == 'Line 1\nLine 2'

    # Case 5: Handling tails
    # <div>Part A<span>Inline</span>Part B</div>
    root5 = node('div', children=[
        node('span', text='Inline', tail='Part B')
    ])
    assert extract_text(root5) == 'InlinePart B'

    # Case 6: Whitespace squashing
    # <div>  Too   much    space  </div>
    root6 = node('div', text='  Too   much    space  ')
    assert extract_text(root6) == 'Too much space'

    # Case 7: Complex structure with mixed block, inline, and tails
    # <div>
    #   <p>Hello <b>World</b></p>
    #   <br>
    #   Tail content
    # </div>
    root7 = node('div', children=[
        node('p', text='Hello ', children=[
            node('b', text='World')
        ]),
        node('br'),
        node('span', text='Tail content', tail='') # tail is None logic handled by extract_text_array
    ])
    # Expected: 'Hello World\nTail content'
    # Breakdown: 
    # p is block -> None
    # b is inline -> no None
    # br is separator -> True
    # span is inline -> no None
    res = extract_text(root7)
    assert 'Hello World' in res
    assert 'Tail content' in res

    # Case 8: Custom symbols
    root8 = node('div', children=[node('p', text='A'), node('p', text='B')])
    assert extract_text(root8, block_symbol='|', sep_symbol='-') == 'A|B'

    # Case 9: No squash space (preserve whitespace)
    root9 = node('div', text='  Space  ')
    assert extract_text(root9, squash_space=False) == '  Space  '

    # Case 10: Empty/None elements
    root10 = node('div', children=[node('p', text=None)])
    assert extract_text(root10) == ''

    # Case 11: Function-like tag (callable)
    class CallableTag:
        def __call__(self): pass
    root11 = MockNode(tag=lambda: None, text="Hidden")
    assert extract_text(root11) == ''
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag=None, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test Case 1: Simple text node
    root1 = MockNode(tag='div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Test Case 2: Nested block elements (div inside div)
    # Should introduce None (newline) between blocks
    root2 = MockNode(tag='div', children=[
        MockNode(tag='p', text='Para 1'),
        MockNode(tag='p', text='Para 2')
    ])
    assert extract_text(root2) == 'Para 1\nPara 2'

    # Test Case 3: Inline elements (span inside div)
    # Should not introduce newlines for inline tags
    root3 = MockNode(tag='div', children=[
        MockNode(tag='span', text='Inline'),
        MockNode(tag='b', text='Bold')
    ])
    assert extract_text(root3) == 'InlineBold'

    # Test Case 4: Separator tag (br)
    # Should introduce True (sep_symbol)
    root4 = MockNode(tag='div', children=[
        MockNode(tag='p', text='Line 1'),
        MockNode(tag='br'),
        MockNode(tag='p', text='Line 2')
    ])
    # br adds True -> sep_symbol (\n)
    assert extract_text(root4) == 'Line 1\n\nLine 2'

    # Test Case 5: Handling of tails
    root5 = MockNode(tag='div', children=[
        MockNode(tag='span', text='Start', tail=' End')
    ])
    assert extract_text(root5) == 'Start End'

    # Test Case 6: Whitespace squashing
    root6 = MockNode(tag='div', text='  Too   \n  many spaces  ')
    assert extract_text(root6) == 'Too many spaces'

    # Test Case 7: Complex structure with mixed types
    # div (block) -> p (block) -> span (inline) + tail
    root7 = MockNode(tag='div', children=[
        MockNode(tag='p', children=[
            MockNode(tag='span', text='Part A', tail=' and Part B')
        ], tail='!')
    ])
    # Structure: [None, 'p', None, 'span', 'Part A', ' and Part B', '!', None, None]
    # After processing: "Part A and Part B!"
    assert extract_text(root7) == 'Part A and Part B!'

    # Test Case 8: Custom symbols
    root8 = MockNode(tag='div', children=[
        MockNode(tag='p', text='A'),
        MockNode(tag='p', text='B')
    ])
    assert extract_text(root8, block_symbol=' | ', sep_symbol=' <br> ') == 'A | B'

    # Test Case 9: Empty node
    root9 = MockNode(tag='div', children=[])
    assert extract_text(root9) == ''

    # Test Case 10: Function-like tag (should return empty string per code logic)
    class FuncTag:
        def __call__(self): pass
    root10 = MockNode(tag=FuncTag())
    assert extract_text(root10) == ''
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag=None, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []
    
    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Test Case 1: Simple text node (No tags)
    root1 = MockNode(tag='p', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Test Case 2: Inline tag within block tag
    # <span> is inline, <p> is block. 
    # Expected: [None (for p), 'Hello', ' ', 'World', None (for p)] -> stripped -> ['Hello', ' ', 'World']
    root2 = MockNode(tag='p', text=None)
    child1 = MockNode(tag='span', text='Hello')
    child1.tail = ' World'
    root2.children = [child1]
    assert extract_text_array(root2) == ['Hello', ' World']

    # Test Case 3: Separator tag (br)
    # br is in SEPARATORS, should return True
    root3 = MockNode(tag='br')
    assert extract_text_array(root3) == [True]

    # Test Case 4: Nested Block tags (Nested newlines)
    # <div> contains <p>
    # <div> -> None, <p> -> None, text, <p> -> None, </div> -> None
    # Squash/Strip logic applies.
    root4 = MockNode(tag='div', text=None)
    child_p = MockNode(tag='p', text='Inner')
    root4.children = [child_p]
    # extract_text_array with default args will squash None,None to one None and strip edges
    assert extract_text_array(root4) == ['Inner']

    # Test Case 5: Complex structure with tails and multiple children
    # <p>Text <span>Bold</span> tail</p>
    root5 = MockNode(tag='p', text='Text ')
    span = MockNode(tag='span', text='Bold')
    span.tail = ' tail'
    root5.children = [span]
    # Expected: ['Text ', 'Bold', ' tail']
    assert extract_text_array(root5) == ['Text ', 'Bold', ' tail']

    # Test Case 6: Non-string callable tag (e.g., a function or object)
    root6 = MockNode(tag=lambda x: x)
    assert extract_text_array(root6) == ''

    # Test Case 7: Testing squash/strip disabled
    # If we disable stripping, the None markers should remain
    root7 = MockNode(tag='div', text=None)
    child7 = MockNode(tag='span', text='Content')
    root7.children = [child7]
    # Without stripping, we expect the padding Nones from the block tags
    result = extract_text_array(root7, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in result
    assert 'Content' in result

    # Test Case 8: Empty node
    root8 = MockNode(tag='div', text=None)
    assert extract_text_array(root8) == []
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockElement:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Test Case 1: Simple block element with text
    root1 = MockElement('div', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Test Case 2: Inline elements inside block elements (no extra newlines)
    # <div><span>A</span><span>B</span></div> -> ['A', 'B']
    root2 = MockElement('div', children=[
        MockElement('span', text='A'),
        MockElement('span', text='B')
    ])
    assert extract_text_array(root2) == ['A', 'B']

    # Test Case 3: Block elements triggering artificial newlines (None)
    # <div><p>A</p><p>B</p></div> -> ['A', 'B'] (after stripping/squashing)
    root3 = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='None', tail='B') # wait, tail belongs to the child
    ])
    # Correct structure for test 3:
    # <div><p>A</p><p>B</p></div>
    # P1 starts -> None appended. P1 text 'A'. P1 ends -> None appended.
    # P2 starts -> None appended. P2 text 'B'. P2 ends -> None appended.
    root3 = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='B')
    ])
    # With squash/strip=True (default):
    # [None, 'A', None, None, 'B', None] -> [_squash_artifical_nl_] -> [None, 'A', None, 'B', None] 
    # -> [_strip_artifical_nl_] -> ['A', 'B']
    assert extract_text_array(root3) == ['A', 'B']

    # Test Case 4: Separators (br) triggering True
    root4 = MockElement('div', children=[
        MockElement('b', text='Part1'),
        MockElement('br'),
        MockElement('b', text='Part2')
    ])
    # [None, 'Part1', True, 'Part2', None] -> ['Part1', True, 'Part2'] (after strip)
    # Note: br is in SEPARATORS.
    assert extract_text_array(root4) == ['Part1', True, 'Part2']

    # Test Case 5: Handling tails
    root5 = MockElement('div', children=[
        MockElement('span', text='Start', tail=' End')
    ])
    assert extract_text_array(root5) == ['Start', ' End']

    # Test Case 6: Function/Callable tag (should return empty string)
    root6 = MagicMock()
    root6.tag = lambda x: x
    assert extract_text_array(root6) == ''

    # Test Case 7: Deeply nested structure
    # <div><ul><li>Item</li></ul></div>
    root7 = MockElement('div', children=[
        MockElement('ul', children=[
            MockElement('li', text='Item')
        ])
    ])
    assert extract_text_array(root7) == ['Item']

    # Test Case 8: Testing squash/strip disabled
    # When squash/strip are False, we expect the raw None and True values preserved
    root8 = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='B')
    ])
    # Raw extraction: [None, 'A', None, None, 'B', None]
    result = extract_text_array(root8, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'A', None, None, 'B', None]

    # Test Case 9: Empty element
    root9 = MockElement('div')
    assert extract_text_array(root9) == []
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Case 1: Simple text node
    root1 = MockNode('div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Case 2: Nested block elements with spacing
    # <div><p>Line 1</p><p>Line 2</p></div> -> "Line 1\nLine 2"
    root2 = MockNode('div', children=[
        MockNode('p', text='Line 1'),
        MockNode('p', text='Line 2')
    ])
    assert extract_text(root2) == 'Line 1\nLine 2'

    # Case 3: Inline elements (should not trigger newlines/None markers)
    # <div><span>Part 1</span><span>Part 2</span></div> -> "Part 1Part 2"
    root3 = MockNode('div', children=[
        MockNode('span', text='Part 1'),
        MockNode('span', text='Part 2')
    ])
    assert extract_text(root3) == 'Part 1Part 2'

    # Case 4: Separator elements (br tag)
    # <div>A<br>B</div> -> "A\nB" (using default sep_symbol='\n')
    root4 = MockNode('div', children=[
        MockNode('a', text='A'),
        MockNode('br'),
        MockNode('a', text='B')
    ])
    # Note: br is in SEPARATORS, triggers True (sep_symbol)
    assert extract_text(root4) == 'A\nB'

    # Case 5: Handling tails
    # <div>Text<p>Inner</p>Tail</div> -> "Text\nInner\nTail"
    root5 = MockNode('div', text='Text', children=[
        MockNode('p', text='Inner', tail='Tail')
    ])
    assert extract_text(root5) == 'Text\nInner\nTail'

    # Case 6: Whitespace squashing
    # <div>  Multiple   Spaces  </div> -> "Multiple Spaces"
    root6 = MockNode('div', text='  Multiple   Spaces  ')
    assert extract_text(root6) == 'Multiple Spaces'

    # Case 7: Custom block and separator symbols
    # Using | as block and - as separator
    root7 = MockNode('div', children=[
        MockNode('p', text='Start'),
        MockNode('br'),
        MockNode('p', text='End')
    ])
    assert extract_text(root7, block_symbol='|', sep_symbol='-') == 'Start-End'

    # Case 8: Complex structure with mixed inline/block and tails
    # <div>Outer<span>Inner</span>Tail<br>New</div>
    root8 = MockNode('div', text='Outer', children=[
        MockNode('span', text='Inner', tail='Tail'),
        MockNode('br'),
        MockNode('p', text='New')
    ])
    # Breakdown: 
    # Outer (block) -> None
    # Inner (inline) -> 'Inner'
    # Tail (tail of span) -> 'Tail'
    # br (separator) -> True
    # New (block/p) -> None
    # Resulting sequence processed by _squash_artifical_nl and join
    assert extract_text(root8) == 'OuterInnerTail\nNew'

    # Case 9: Empty node
    root9 = MockNode('div', text=None, children=[])
    assert extract_text(root9) == ''

    # Case 10: Function/Callable tag (should return empty string)
    root10 = MockNode(tag=lambda x: True)
    assert extract_text(root10) == ''
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test Case 1: Simple Text Node
    node1 = MockNode('p', text='Hello')
    assert extract_post_processing_helper(node1) == 'Hello'

    # Test Case 2: Nested Block Elements (div > p > span)
    # Should insert \n for block elements and join them
    node2 = MockNode('div', children=[
        MockNode('p', text='Line 1'),
        MockNode('p', text='Line 2')
    ])
    assert extract_post_processing_helper(node2) == 'Line 1\nLine 2'

    # Test Case 3: Inline Elements (no newlines added between them)
    node3 = MockNode('p', children=[
        MockNode('span', text='Part 1'),
        MockNode('b', text=' Part 2')
    ])
    assert extract_post_processing_helper(node3) == 'Part 1 Part 2'

    # Test Case 4: Elements with tails (text following a tag)
    node4 = MockNode('div', children=[
        MockNode('span', text='Start', tail=' End')
    ])
    assert extract_post_processing_helper(node4) == 'Start End'

    # Test Case 5: Separator element (br)
    node5 = MockNode('p', children=[
        MockNode('b', text='Top'),
        MockNode('br'),
        MockNode('b', text='Bottom')
    ])
    # br uses sep_symbol which defaults to \n
    assert extract_post_processing_helper(node5) == 'Top\nBottom'

    # Test Case 6: Whitespace Squashing
    node6 = MockNode('p', text='  Too   much    space  ')
    assert extract_post_processing_helper(node6) == 'Too much space'

    # Test Case 7: Complex structure with mixed blocks and inlines
    # <div><p><b>Bold</b> Text</p><br><span>New</span></div>
    complex_node = MockNode('div', children=[
        MockNode('p', children=[
            MockNode('b', text='Bold'),
            MockNode('text_node', text=' Text', tail=None) # Note: logic uses child.tail
        ], tail=None),
        # Manually adding a sibling via the list to simulate tails correctly
        MockNode('br'),
        MockNode('span', text='New')
    ])
    # We need to fix the structure for a valid tree traversal in this test mock
    node_complex = MockNode('div', children=[
        MockNode('p', children=[
            MockNode('b', text='Bold', tail=' Text')
        ]),
        MockNode('br'),
        MockNode('span', text='New')
    ])
    # Result: 'Bold Text' (from p block) + '\n' (from br) + 'New' (from span)
    assert extract_post_processing_helper(node_complex) == 'Bold Text\nNew'

    # Test Case 8: Custom symbols
    node8 = MockNode('div', children=[MockNode('p', text='A'), MockNode('p', text='B')])
    assert extract_text(node8, block_symbol=' | ', sep_symbol=' > ') == 'A > B'

def extract_post_processing_helper(dom):
    """Helper to call the main function with default params."""
    return extract_text(dom)

def test_extract_text_edge_cases():
    # Empty node
    node_empty = MockNode('div', text=None, children=[])
    assert extract_text(node_empty) == ''

    # Node with only whitespace
    node_ws = MockNode('div', text='   \n\t  ')
    assert extract_text(node_ws) == ''

    # Functionality of squash_space=False (preserving structure/whitespace)
    node_preservation = MockNode('p', text='Line 1\nLine 2')
    # Note: The current implementation of extract_text calls _squash_artifical_nl 
    # inside extract_text_array if squash_artifical_nl is True.
    # Testing the behavior where we want to see raw-ish content.
    res = extract_text(node_preservation, squash_space=False)
    assert 'Line 1' in res
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Test Case 1: Simple text node (inline)
    node1 = MockNode('span', text='Hello')
    assert extract_text_array(node1) == ['Hello']

    # Test Case 2: Block element with text and child
    child1 = MockNode('b', text='Bold')
    parent1 = MockNode('div', children=[child1])
    # div is not in INLINE_TAGS, so it adds None at start/end of its scope
    # Note: extract_text_array logic for block tags adds [None, ..., None]
    # _strip_artifical_nl removes leading/trailing Nones
    assert extract_text_array(parent1) == ['Bold']

    # Test Case 3: Separator element (br)
    node2 = MockNode('br')
    # br is in SEPARATORS, so it adds True
    assert extract_text_array(node2) == [True]

    # Test Case 4: Complex structure with tails and nested blocks
    # <div><span>A</span>B</div>
    child_span = MockNode('span', text='A')
    parent_div = MockNode('div', children=[child_span])
    child_span.tail = 'B'
    # div (block) -> [None, span(inline)->['A'], tail->'B', None]
    # strip_artifical_nl removes the outer Nones
    assert extract_text_array(parent_div) == ['A', 'B']

    # Test Case 5: Nested block elements causing artificial newlines
    # <div><p>Text</p></div>
    child_p = MockNode('p', text='Inner')
    parent_div2 = MockNode('div', children=[child_p])
    # div(block) -> [None, p(block)->[None, 'Inner', None], None]
    # squash_artifical_nl removes consecutive Nones
    # strip_artifical_nl removes leading/trailing Nones
    assert extract_text_array(parent_div2) == ['Inner']

    # Test Case 6: Function as tag (should return empty string via logic in code)
    def dummy_tag(): pass
    node3 = MockNode(dummy_tag)
    assert extract_text_array(node3) == ''

    # Test Case 7: Whitespace squashing within text
    node4 = MockNode('span', text='Line\n\nBreak   Space')
    assert extract_text_array(node4) == ['Line Break Space']

    # Test Case 8: Testing squash_artifical_nl=False flag
    # Ensures Nones are preserved if requested (though strip usually cleans them)
    child_a = MockNode('span', text='A')
    parent_b = MockNode('div', children=[child_a])
    # Without squashing, the sequence is [None, 'A', None]
    # strip_artifical_nl will still trim the outer Nones if it's a block tag
    result = extract_text_array(parent_b, squash_artifical_nl=False)
    assert result == ['A']

    # Test Case 9: Empty node
    node5 = MockNode('div')
    assert extract_text_array(node5) == []
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockElement:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test Case 1: Simple text node
    dom1 = MockElement('div', text='Hello')
    assert extract_text(dom1) == 'Hello'

    # Test Case 2: Nested block elements (div with p)
    # Should insert None (new line symbol) at start and end of block tags
    dom2 = MockElement('div', children=[
        MockElement('p', text='Paragraph content')
    ])
    # extract_text_array produces [None, 'Paragraph content', None]
    # result becomes '\nParagraph content\n' -> stripped to 'Paragraph content'
    assert extract_text(dom2) == 'Paragraph content'

    # Test Case 3: Inline elements (span/b) - No extra newlines added
    dom3 = MockElement('div', children=[
        MockElement('span', text='Inline'),
        MockElement('b', text='Bold')
    ])
    # Array roughly: [None, 'Inline', 'Bold', None] -> 'InlineBold'
    # But because they are inline, no extra None is added between them.
    # However, the outer div adds None at start/end.
    assert extract_text(dom3) == 'InlineBold'

    # Test Case 4: Separator elements (br)
    dom4 = MockElement('div', children=[
        MockElement('p', text='Line 1'),
        MockElement('br'),
        MockElement('p', text='Line 2')
    ])
    # br adds True, which maps to sep_symbol (\n)
    assert extract_text(dom4) == 'Line 1\nLine 2'

    # Test Case 5: Whitespace squashing
    dom5 = MockElement('div', text='  Too   much\twhitespace  ')
    assert extract_text(dom5) == 'Too much whitespace'

    # Test Case 6: Complex structure with tails and interleaved text
    # <div><p>Part 1 <span>Inner</span> Part 2</p></div>
    child_span = MockElement('span', text='Inner')
    child_p = MockElement('p', text='Part 1 ', children=[child_span], tail=' Part 2')
    dom6 = MockElement('div', children=[child_p])
    
    # Expected: 'Part 1 Inner Part 2' (with whitespace squashed)
    assert extract_text(dom6) == 'Part 1 Inner Part 2'

    # Test Case 7: Custom block and separator symbols
    dom7 = MockElement('div', children=[
        MockElement('p', text='Block'),
        MockElement('br'),
        MockElement('p', text='End')
    ])
    assert extract_text(dom7, block_symbol='|', sep_symbol='-') == '|Block-End|'.strip('|') 
    # Note: strip() in extract_text removes leading/trailing block symbols if they are newlines.
    # Testing specific symbol behavior:
    assert extract_text(dom7, block_symbol='[B]', sep_symbol='[S]') == '[B]Block[S]End[B]'.strip('[B]')

    # Test Case 8: Empty element
    dom8 = MockElement('div', children=[])
    assert extract_text(dom8) == ''

    # Test Case 9: Elements with only tails
    child_tail = MockElement('span', text='', tail=' Only Tail')
    dom9 = MockElement('div', children=[child_tail])
    assert extract_text(dom9) == 'Only Tail'
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test Case 1: Simple text node
    node1 = MockNode('div', text='Hello')
    assert extract_util_text(node1) == 'Hello'

    # Test Case 2: Nested block elements with whitespace
    # <div><p>Line 1</p><p>Line 2</p></div>
    p1 = MockNode('p', text='Line 1')
    p2 = MockNode('p', text='Line 2')
    root = MockNode('div', children=[p1, p2])
    # Expected: 'Line 1\nLine 2' (None represents block separation)
    assert extract_util_text(root) == 'Line 1\nLine 2'

    # Test Case 3: Inline elements within blocks
    # <div><span>Bold <b>Text</b></span></div>
    b = MockNode('b', text='Text')
    span = MockNode('span', text='Bold ', children=[b])
    root2 = MockNode('div', children=[span])
    assert extract_util_text(root2) == 'Bold Text'

    # Test Case 4: Separator elements (br)
    # <div>Part 1<br>Part 2</div>
    br = MockNode('br')
    p1 = MockNode('span', text='Part 1')
    br.tail = 'Part 2' # tail of br is after the element
    root3 = MockNode('div', children=[p1, br])
    # True represents separator (sep_symbol defaults to \n)
    assert extract_util_text(root3) == 'Part 1\nPart 2'

    # Test Case 5: Complex structure with mixed whitespace and tails
    # <div>
    #   <div>Text</div>
    #   <span>More</span>
    # </div>
    inner_div = MockNode('div', text='Inner')
    span_node = MockNode('span', text='Span')
    root4 = MockNode('div', children=[inner_div, span_node])
    # Result should be squashed: 'Inner\nSpan'
    assert extract_util_text(root4) == 'Inner\nSpan'

    # Test Case 6: Testing squash_space=False
    # We want to see the raw None/True symbols if possible, 
    # but extract_text returns a string. We test that it doesn't strip.
    node5 = MockNode('div', text='  Trim Me  ')
    assert extract_util_text(node5, squash_space=False) == '  Trim Me  '

    # Test Case 7: Empty node
    node6 = MockNode('div')
    assert extract_util_text(node6) == ''

    # Test Case 8: Elements with only tails (no text)
    # <div><p>Text</p>Trailing</div>
    p3 = MockNode('p', text='Text')
    p3.tail = 'Trailing'
    root5 = MockNode('div', children=[p3])
    assert extract_util_text(root5) == 'Text\nTrailing'

def extract_util_text(dom, block_symbol='\n', sep_symbol='\n', squash_space=True):
    """Helper to call the function with default parameters for testing."""
    return extract_text(dom, block_symbol, sep_symbol, squash_space)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Case 1: Simple inline element (no newlines, no separators)
    node1 = MockNode('span', text='hello')
    assert extract_text_array(node1) == ['hello']

    # Case 2: Block element (should introduce None/newlines)
    node2 = MockNode('div', text='start')
    child2 = MockNode('p', text='middle')
    node2.children = [child2]
    # div is not in INLINE_TAGS, so it adds None at start and end
    # Note: _strip_artifical_nl removes leading/trailing Nones
    assert extract_text_array(node2) == ['start', 'middle']

    # Case 3: Element with tail
    node3 = MockNode('div')
    child3 = MockNode('b', text='bold')
    child3.tail = ' italic'
    node3.children = [child3]
    assert extract_text_array(node3) == ['bold', ' italic']

    # Case 4: Separator element (br)
    node4 = MockNode('br')
    assert extract_text_array(node4) == [True]

    # Case 5: Nested structure with complex whitespace/newlines
    # <div><p>A</p>B</div> -> ['A', 'B']
    node5 = MockNode('div')
    child5 = MockNode('p', text='A')
    child5.tail = 'B'
    node5.children = [child5]
    # The logic: div adds None at start/end, p adds None at start/end. 
    # strip_artifical_nl removes them.
    assert extract_text_array(node5) == ['A', 'B']

    # Case 6: Testing squash_artifical_nl=False (preserving the Nones)
    node6 = MockNode('div', text='A')
    child6 = MockNode('p', text='B')
    node6.children = [child6]
    # Without squashing, we see the structure of None markers
    result = extract_text_array(node6, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in result
    assert 'A' in result
    assert 'B' in result

    # Case 7: Empty node
    node7 = MockNode('div')
    assert extract_text_array(node7) == []

    # Case 8: Node with callable tag (returns empty string per implementation)
    class CallableTag:
        def __call__(self): pass
    node8 = MockNode(CallableTag())
    assert extract_text_array(node8) == ''

    # Case 9: Verify INLINE_TAGS behavior (no None added)
    node9 = MockNode('span', text='inline')
    assert extract_text_array(node9) == ['inline']

    # Case 10: Verify non-INLINE_TAGS behavior (None added)
    node10 = MockNode('div', text='block')
    # Since it's a block, it wraps in None. strip removes them. 
    # If we disable stripping/squashing, Nones should be present.
    result10 = extract_text_array(node10, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result10 == [None, 'block', None]
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockNode:
    def __init__(self, tag, text=None, tail=None, children=None):
        self.tag = tag
        self.text = text
        self.tail = tail
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text_array():
    # Test Case 1: Empty node
    root1 = MockNode('div')
    assert extract_text_array(root1) == []

    # Test Case 2: Simple text node (Inline)
    root2 = MockNode('span', text='hello')
    assert extract_text_array(root2) == ['hello']

    # Test Case 3: Block level node with text and trailing None (artificial NL)
    root3 = MockNode('div', text='start')
    assert extract_text_array(root3) == ['start']

    # Test Case 4: Nested structure with block elements
    # <div><p>Hello</p><span>World</span></div>
    child_p = MockNode('p', text='Hello')
    child_span = MockNode('span', text='World')
    root4 = MockNode('div', children=[child_p, child_span])
    # Expected: div (None) -> p (None) -> 'Hello' -> p (None) -> span ('World') -> span (None) -> div (None)
    # After squash/strip logic in extract_text_array:
    result4 = extract_text_array(root4)
    assert 'Hello' in result4
    assert 'World' in result4

    # Test Case 5: Separator tag <br>
    root5 = MockNode('br')
    assert extract_text_array(root5) == [True]

    # Test Case 6: Complex tree with tails and nested blocks
    # <div>Text<p><b>Bold</b></p>Tail</div>
    b = MockNode('b', text='Bold')
    p = MockNode('p', text='Inner', children=[b])
    root6 = MockNode('div', text='Outer', children=[p])
    p.tail = ' Tail' # This tail belongs to the <p> tag inside <div>
    
    result6 = extract_text_array(root6)
    # The logic should process:
    # div starts with None (block)
    # div text 'Outer'
    # p starts with None (block)
    # p text 'Inner'
    # b starts with 'b' is inline, so no None added before it.
    # b text 'Bold'
    # b ends (inline), tail of b is None.
    # p ends with None (block).
    # p tail is ' Tail'.
    # div ends with None (block).
    
    assert any('Outer' in s for s in result6 if s)
    assert any('Bold' in s for s in result6 if s)
    assert any('Tail' in s for s in result6 if s)

    # Test Case 7: Verifying squash_artifical_nl=False preserves None/True markers
    root7 = MockNode('div', text='A')
    child7 = MockNode('p', text='B')
    root7.children = [child7]
    
    raw_result = extract_text_array(root7, squash_artifical_nl=False, strip_artally_nl=False)
    # Check if markers exist in the un-squashed version
    assert None in raw_result or True in raw_result

    # Test Case 8: Function with callable tag (should return empty string per code logic)
    class CallableTag:
        def __call__(self): pass
    root8 = MockNode(CallableTag())
    assert extract_text_array(root8) == ''

    # Test Case 9: Inline tags do not add None/True markers before/after themselves
    root9 = MockNode('div', children=[MockNode('span', text='inline')])
    result9 = extract_text_array(root9)
    # Since span is in INLINE_TAGS, it shouldn't trigger the 'None' padding logic 
    # for the start of its own element if it were treated as a block.
    assert 'inline' in result9
```


