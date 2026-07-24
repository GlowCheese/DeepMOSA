####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
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
    assert extract_text(root1) == 'Hello'

    # Test 2: Nested block elements (div inside div) should add newlines/None markers
    root2 = MockElement('div', children=[
        MockElement('p', text='Paragraph 1'),
        MockElement('p', text='Paragraph 2')
    ])
    # Expected: 'Paragraph 1\nParagraph 2' (due to None markers in block elements)
    assert extract_text(root2) == 'Paragraph 1\nParagraph 2'

    # Test 3: Inline elements (span, b) should not add extra newlines
    root3 = MockElement('div', children=[
        MockElement('p', text='Text with ', children=[
            MockElement('span', text='inline'),
            MockElement('b', text='bold')
        ], tail=' end.')
    ])
    assert extract_text(root3) == 'Text with inlinebold end.'

    # Test 4: Separator elements (br) should add the sep_symbol
    root4 = MockElement('div', children=[
        MockElement('p', text='Line 1', children=[
            MockElement('br')
        ], tail='Line 2')
    ])
    assert extract_text(root4, sep_symbol=' ') == 'Line 1 Line 2'

    # Test 5: Whitespace squashing
    root5 = MockElement('div', text='  Too   much\nwhitespace  ')
    assert extract_text(root5) == 'Too much whitespace'

    # Test 6: Complex structure with tails and mixed tags
    # Structure: <div><p>Part 1 <span>inner</span> part 2</p><br>Next</div>
    root6 = MockElement('div', children=[
        MockElement('p', text='Part 1 ', children=[
            MockElement('span', text='inner')
        ], tail=' part 2'),
        MockElement('br'),
        MockElement('p', text='Next')
    ])
    # br is a separator, p is a block (None marker)
    # 'Part 1 inner part 2' + '\n' (from br) + '\n' (from block logic) + 'Next'
    # Resulting in stripped/squashed: 'Part 1 inner part 2\nNext'
    assert extract_text(root6, sep_symbol='|') == 'Part 1 inner part 2|Next'

    # Test 7: Empty element
    root7 = MockElement('div', children=[MockElement('p')])
    assert extract_text(root7) == ''

    # Test 8: Testing custom block and sep symbols
    root8 = MockElement('div', children=[
        MockElement('div', text='A'),
        MockElement('br'),
        MockElement('div', text='B')
    ])
    assert extract_text(root8, block_symbol='[B]', sep_symbol='[S]') == '[B]A[S][B]B[B]' 
    # Note: the logic of extract_text_array adds None at start/end of blocks.
    # Let's re-verify exact behavior for root8:
    # div (block) -> [None, 'A', None]
    # br (sep)    -> [True]
    # div (block) -> [None, 'B', None]
    # Combined parts: [None, 'A', None, True, None, 'B', None]
    # _squash_artifical_nl removes consecutive Nones.
    # Final join with block='[B]' and sep='[S]' -> '[B]A[S][B]B[B]' (if all None are processed)
    # Actually, looking at the code: 
    # root8 parts after extract_text_array: [None, 'A', None, True, None, 'n/a', None, 'B', None]
    # After squash_artifical_nl (removes duplicates): [None, 'A', True, None, 'B', None]
    # Result: '[B]A[S][B]B[B]' is theoretically correct based on implementation.
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

def test_extract_text():
    # Test Case 1: Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Test Case 2: Nested block elements with newlines
    # <div><p>Line 1</p><p>Line 2</p></div> -> "Line 1\nLine 2"
    child1 = MockElement('p', text='Line 1')
    child2 = MockElement('p', text='Line 2')
    root2 = MockElement('div', children=[child1, child2])
    assert extract_text(root2) == 'Line 1\nLine 2'

    # Test Case 3: Inline elements (no extra newlines/separators)
    # <div><span>Part </span><b>Bold</b></div> -> "Part Bold"
    span = MockElement('span', text='Part ')
    bold = MockElement('b', text='Bold')
    root3 = MockElement('div', children=[span, bold])
    assert extract_text(root3) == 'Part Bold'

    # Test Case 4: Separator tag (br)
    # <div>Line 1<br>Line 2</div> -> "Line 1\nLine 2"
    br = MockElement('br')
    child_text = MockElement('span', text='Line 1')
    child_text.tail = None # tail is handled by parent logic
    # To simulate <p>Line 1<br>Line 2</p>, 'Line 2' must be the tail of <br>
    br_with_tail = MockElement('br', tail='Line 2')
    root4 = MockElement('p', text='Line 1', children=[br_with_tail])
    # Note: In extract_text_array, br adds True (sep_symbol)
    assert extract_text(root4) == 'Line 1\nLine 2'

    # Test Case 5: Whitespace squashing
    # <div>  Too   much   space  </div> -> "Too much space"
    root5 = MockElement('div', text='  Too   much   space  ')
    assert extract_text(root5) == 'Too much space'

    # Test Case 6: Complex structure with tails and mixed tags
    # <div><p>Start <span>middle</span> end</p><footer>Bottom</footer></div>
    span_inner = MockElement('span', text='middle')
    p_tag = MockElement('p', text='Start ', children=[span_inner])
    span_inner.tail = ' end'
    footer = MockElement('footer', text='Bottom')
    root6 = MockElement('div', children=[p_tag, footer])
    # Expected: "Start middle end\nBottom"
    assert extract_text(root6) == 'Start middle end\nBottom'

    # Test Case 7: Custom symbols
    root7 = MockElement('div', children=[MockElement('p', text='A'), MockElement('p', text='B')])
    assert extract_text(root7, block_symbol=' | ', sep_symbol=' - ') == 'A - B'

    # Test Case 8: Empty element
    root8 = MockElement('div', children=[MockElement('p')])
    assert extract_text(root8) == ''

    # Test Case 9: Element with no text and no children
    root9 = MockElement('div')
    assert extract_text(root9) == ''

    # Test Case 10: Functionally callable tag (e.g., some custom objects)
    def dummy_tag(): pass
    dummy_tag.tag = 'custom'
    assert extract_text(dummy_tag) == ''
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
        self.children = children or []

    def getchildren(self):
        return self.children

def test_extract_text():
    # Test Case 1: Simple text node
    node1 = MockNode(tag='p', text='Hello')
    assert extract_text(node1) == 'Hello'

    # Test Case 2: Nested structure with block elements (adding newlines)
    # <p><div>Part 1</div>Part 2</p> -> "Part 1\nPart 2"
    child_div = MockNode(tag='div', text='Part 1')
    root_p = MockNode(tag='p', text=None, children=[child_div])
    child_div.tail = 'Part 2'
    assert extract_text(root_p) == 'Part 1\nPart 2'

    # Test Case 3: Inline elements (no extra newlines)
    # <p><span>Hello</span> <b>World</b></p> -> "Hello World"
    span = MockNode(tag='span', text='Hello')
    bold = MockNode(tag='b', text='World')
    root_inline = MockNode(tag='p', text=None, children=[span, bold])
    span.tail = ' '
    assert extract_text(root_inline) == 'Hello World'

    # Test Case 4: Separator elements (br tag adds sep_symbol)
    # <p>Line 1<br>Line 2</p> -> "Line 1\nLine 2"
    br = MockNode(tag='br')
    root_br = MockNode(tag='p', text='Line 1', children=[br])
    br.tail = 'Line 2'
    assert extract_text(root_br) == 'Line 1\nLine 2'

    # Test Case 5: Custom symbols
    # Testing custom block and sep symbols
    node_custom = MockNode(tag='div', text='Start')
    child_sep = MockNode(tag='br')
    node_custom.children = [child_sep]
    child_sep.tail = 'End'
    assert extract_text(node_custom, block_symbol='|', sep_symbol='*') == 'Start*End'

    # Test Case 6: Whitespace squashing
    # <p>  Too   many    spaces  </p> -> "Too many spaces"
    node_space = MockNode(tag='p', text='  Too   many    spaces  ')
    assert extract_text(node_space) == 'Too many spaces'

    # Test Case 7: Deeply nested structure with mixed types
    # <div><body><p>Text<span>Inner</span></p></div>
    inner_span = MockNode(tag='span', text='Inner')
    inner_p = MockNode(tag='p', text='Text', children=[inner_span])
    inner_body = MockNode(tag='body', text=None, children=[inner_p])
    root_div = MockNode(tag='div', text=None, children=[inner_body])
    assert extract_text(root_div) == 'TextInner'

    # Test Case 8: Empty node
    node_empty = MockNode(tag='div')
    assert extract_text(node_empty) == ''

    # Test Case 9: Function/Callable tag (should return empty string per code logic)
    node_callable = MockNode(tag=lambda x: x)
    assert extract_text(node_callable) == ''

    # Test Case 10: Handling of None text and tails
    node_none = MockNode(tag='div', text=None)
    assert extract_text(node_none) == ''
```


# LLM-generated content at query #4
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
    # Test 1: Simple Text node (not a tag)
    # Note: The function expects an object with .tag and .getchildren()
    # If callable(dom.tag), it returns ''
    root_simple = MockElement('div', text='Hello')
    assert extract_text_array(root_simple) == ['Hello']

    # Test 2: Inline tag (should not add None/newline markers)
    root_inline = MockElement('span', text='Inline')
    assert extract_text_array(root_inline) == ['Inline']

    # Test 3: Block tag (should add None markers at start and end)
    # Note: _strip_artifical_nl removes leading/trailing None from the result
    root_block = MockElement('div', text='Block')
    assert extract_text_array(root_block) == ['Block']

    # Test 4: Separator tag (br) -> should add True
    root_br = MockElement('br')
    assert extract_text_array(root_br) == [True]

    # Test 5: Nested structure with text and tails
    # <div>Text<span>Child</span>Tail</div>
    child = MockElement('span', text='Child')
    child.tail = 'Tail'
    root_nested = MockElement('div', text='Text', children=[child])
    # extract_text_array (recursive) uses squash/strip=False for children
    # The top level call uses default True/True
    assert extract_text_array(root_nested) == ['Text', 'Child', 'Tail']

    # Test 6: Complex hierarchy with whitespace and block markers
    # <div>A<p>B</p>C</div>
    # 1. div starts -> adds None (if squash/strip=True, this is stripped later)
    # 2. text 'A'
    # 3. child p:
    #    - p starts -> adds None
    #    - text 'B'
    #    - p ends -> adds None
    # 4. tail 'C'
    # 5. div ends -> adds None
    p_node = MockElement('p', text='B')
    root_complex = MockElement('div', text='A', children=[p_node])
    p_node.tail = 'C'
    
    # Because of strip_artifical_nl, the leading/trailing Nones are gone
    # but internal ones are squashed by _squash_artifical_nl
    result = extract_text_array(root_complex)
    assert result == ['A', 'B', 'C']

    # Test 7: Test callable tag edge case
    class CallableTagNode:
        def __init__(self):
            self.tag = lambda x: x
    assert extract_text_array(CallableTagNode()) == ''

    # Test 8: Empty element
    root_empty = MockElement('div')
    assert extract_text_array(root_empty) == []

    # Test 9: Verify whitespace squashing in the final output via regex
    root_ws = MockElement('div', text='Word\n\r  Next')
    # WHITESPACE_RE replaces \n, \r, etc with ' '
    assert extract_text_array(root_ws) == ['Word Next']
```


# LLM-generated content at query #5
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
    # Test Case 1: Simple inline element (no separators, no artificial NL)
    node1 = MockNode('span', text='hello')
    assert extract_text_array(node1) == ['hello']

    # Test Case 2: Block element (introduces None for artificial NL)
    node2 = MockNode('div', text='start')
    child2 = MockNode('b', text='bold')
    node2.children = [child2]
    # Expected: [None, 'start', 'bold', None] -> stripped -> ['start', 'bold']
    assert extract_text_array(node2) == ['start', 'bold']

    # Test Case 3: Separator element (br tag)
    node3 = MockNode('br')
    assert extract_text_array(node3) == [True]

    # Test Case 4: Nested structure with tails and text
    # <div>Text <span>Inner</span> Tail</div>
    root = MockNode('div', text='Text ')
    span = MockNode('span', text='Inner')
    span.tail = ' Tail'
    root.children = [span]
    # Process:
    # root tag div is not inline -> adds None
    # root.text is 'Text ' -> adds 'Text '
    # child span is inline -> no None added at start of span
    # span.text is 'Inner' -> adds 'Inner'
    # span.tail is ' Tail' -> adds ' Tail'
    # root tag div end -> adds None
    # Result before stripping: [None, 'Text ', 'Inner', ' Tail', None]
    # After _squash_artifical_nl and _strip_artifical_nl: ['Text ', 'Inner', ' Tail']
    # Note: extract_text_array calls itself with squash=False for children
    assert extract_text_array(root) == ['Text ', 'Inner', ' Tail']

    # Test Case 5: Complex nesting with multiple levels
    # <div>A<p>B</p>C</div>
    # div (block) -> [None]
    # div.text -> 'A'
    # p (block) -> [None, 'B', None]
    # p.tail -> 'C'
    # div end -> [None]
    # Total: [None, 'A', None, 'B', None, 'C', None]
    # Stripped: ['A', 'B', 'C']
    root2 = MockNode('div', text='A')
    p_node = MockNode('p', text='B')
    p_node.tail = 'C'
    root2.children = [p_node]
    assert extract_text_array(root2) == ['A', 'B', 'C']

    # Test Case 6: Functionality with squash_artifical_nl=False
    # This should preserve the None markers
    root3 = MockNode('div', text='A')
    root3.children = [MockNode('span', text='B')]
    result = extract_text_array(root3, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in result
    assert 'A' in result
    assert 'B' in result

    # Test Case 7: Callable tag (like a function/class) should return empty string
    def dummy_tag(): pass
    node_callable = MockNode(dummy_tag)
    assert extract_text_array(node_callable) == ''

    # Test Case 8: Empty node
    node_empty = MockNode('div')
    assert extract_text_array(node_empty) == []

    # Test Case 9: Whitespace handling in text
    node_ws = MockNode('span', text='  word  \n  ')
    # Note: extract_text_array itself doesn't call squash_html_whitespace, 
    # but it preserves the string as is.
    assert extract_text_array(node_ws) == ['  word  \n  ']
```


# LLM-generated content at query #6
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
    # Test 1: Simple text node (no tags)
    root1 = MockElement('div', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Test 2: Inline tags (should not add None or True)
    root2 = MockElement('p', text='Hello ', children=[
        MockElement('b', text='World')
    ])
    # 'b' is in INLINE_TAGS, so no None/True added around it. 
    # Tail of 'b' is None.
    assert extract_text_array(root2) == ['Hello ', 'World']

    # Test 3: Block tags (should add None as artificial newline)
    root3 = MockElement('div', text='Start', children=[
        MockElement('p', text='Middle')
    ])
    # div is not in INLINE_TAGS -> adds None at start and end.
    # p is not in INLINE_TAGs -> adds None at start and end.
    # strip_artifical_nl should remove the leading/trailing Nones.
    # Resulting array after squash/strip: ['Start', 'Middle']
    assert extract_text_array(root3) == ['Start', 'Middle']

    # Test 4: Separator tags (br)
    root4 = MockElement('div', text='Line1', children=[
        MockElement('br'),
        MockElement('span', text='Line2')
    ])
    # br is in SEPARATORS -> adds True.
    assert extract_text_array(root4) == ['Line1', True, 'Line2']

    # Test 5: Complex structure with tails and whitespace
    # <div>
    #   Text1
    #   <span>Span</span> Tail1
    #   <p>PContent</p>
    # </div>
    root5 = MockElement('div', text='Text1', children=[
        MockElement('span', text='Span', tail=' Tail1'),
        MockElement('p', text='PContent')
    ])
    # Expected: 'Text1' (from div.text), then 'Span' (child text), 
    # then ' Tail1' (child tail), then 'PContent' (child text).
    # Because they are block elements, Nones are added but stripped.
    assert extract_text_array(root5) == ['Text1', 'Span', ' Tail1', 'PContent']

    # Test 6: Function with callable tag (as per code logic)
    class CallableTag:
        def __call__(self): pass
    root6 = MockElement(CallableTag())
    assert extract_text_array(root6) == ''

    # Test 7: Testing squash/strip flags
    # If squash_artifical_nl=False, Nones should remain if they aren't at edges
    root7 = MockElement('div', text='A', children=[
        MockElement('p', text='B')
    ])
    # Internal None is not stripped by _strip_artifical_nl if it's between strings.
    res = extract_text_array(root7, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in res

    # Test 8: Empty element
    root8 = MockElement('div')
    assert extract_text_array(root8) == []
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
        self._children = children or []

    def getchildren(self):
        return self._children

def test_extract_text():
    # Test Case 1: Basic single text node
    root1 = MockNode(tag='p', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Test Case 2: Nested block elements with newlines
    # <div><p>Part 1</p><p>Part 2</p></div> -> "Part 1\nPart 2"
    child1 = MockNode(tag='p', text='Part 1')
    child2 = MockNode(tag='p', text='Part 2')
    root2 = MockNode(tag='div', children=[child1, child2])
    assert extract_text(root2) == 'Part 1\nPart 2'

    # Test Case 3: Inline elements (should not trigger newlines/None)
    # <div><span>Inner</span>Text</div> -> "InnerText"
    child3 = MockNode(tag='span', text='Inner')
    root3 = MockNode(tag='div', children=[child3])
    child3.tail = 'Text'
    assert extract_text(root3) == 'InnerText'

    # Test Case 4: Separator tags (br) should trigger sep_symbol (\n)
    child4 = MockNode(tag='br')
    root4 = MockNode(tag='p', children=[MockNode(tag='b', text='A'), child4, MockNode(tag='b', text='B')])
    assert extract_text(root4) == 'A\nB'

    # Test Case 5: Whitespace squashing
    # Verify that multiple spaces/newlines are reduced to single space
    root5 = MockNode(tag='p', text='Hello\n\n\nWorld')
    assert extract_text(root5) == 'Hello World'

    # Test Case 6: Complex structure with tails and mixed tags
    # <div><p><b>Bold</b><i>Italic</i></p>Tail</div>
    inner_b = MockNode(tag='b', text='Bold')
    inner_i = MockNode(tag='i', text='Italic')
    p_node = MockNode(tag='p', children=[inner_b, inner_i])
    root6 = MockNode(tag='div', children=[p_node])
    p_node.tail = 'Tail'
    # Result should be: Bold + Italic (inline) -> "BoldItalic" 
    # then P is block level so it adds \n
    # Then Tail follows the block content.
    assert extract_text(root6) == 'BoldItalic\nTail'

    # Test Case 7: Custom symbols
    root7 = MockNode(tag='div', children=[MockNode(tag='p', text='A'), MockNode(tag='p', text='B')])
    assert extract_text(root7, block_symbol=' | ', sep_symbol=' > ') == 'A | B'

    # Test Case 8: Empty nodes/No text
    root8 = MockNode(tag='div', children=[MockNode(tag='p'), MockNode(tag='p')])
    assert extract_text(root8) == ''

    # Test Case 9: Verify squash_space=False preserves original structure roughly
    # (Note: extract_text_array logic is quite aggressive, but we check if it bypasses the regex sub)
    root9 = MockNode(tag='p', text='Line\n\nBreak')
    # With squash_space=True (default), it becomes 'Line Break'
    assert extract_text(root9, squash_space=True) == 'Line Break'
    # Note: The function uses WHITESPACE_RE.sub(' ', text) in squash_html_whitespace 
    # which is called inside extract_text when squash_space is True.
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
    # Case 1: Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Case 2: Nested block elements (div inside div)
    # Should introduce artificial newlines (None in array) which become \n
    root2 = MockElement('div', children=[
        MockElement('p', text='Paragraph 1'),
        MockSHELL := MockElement('p', text='Paragraph 2')
    ])
    # Expected: div start -> None, p start -> None, 'Paragraph 1', p end -> None, 'Paragraph 2', p end -> None, div end -> None
    # After squash/strip: 'Paragraph 1\nParagraph 2'
    assert extract_text(root2) == 'Paragraph 1\nParagraph 2'

    # Case 3: Inline elements (span inside div)
    # Inline tags should not trigger artificial newlines
    root3 = MockElement('div', children=[
        MockElement('span', text='Inline'),
        MockElement('b', text='Bold')
    ])
    assert extract_text(root3) == 'InlineBold'

    # Case 4: Separator tag (br)
    # br should trigger the True symbol which becomes sep_symbol (\n by default)
    root4 = MockElement('div', children=[
        MockElement('span', text='Line1'),
        MockElement('br'),
        MockElement('span', text='Line2')
    ])
    assert extract_text(root4) == 'Line1\nLine2'

    # Case 5: Handling tails (text following an element)
    root5 = MockElement('div', children=[
        MockElement('span', text='Start', tail=' End')
    ])
    assert extract_text(root5) == 'Start End'

    # Case 6: Complex structure with whitespace and mixed tags
    # Testing squash_space=True (default)
    root6 = MockElement('div', children=[
        MockElement('h1', text='  Title  '),
        MockElement('p', children=[
            MockElement('a', text='Link'),
            MockElement('b', text='Bold')
        ], tail=' Tail'),
        MockElement('br'),
        MockElement('div', text='Bottom')
    ])
    # Title (block) -> \n
    # Link (inline) + Bold (inline) -> LinkBold
    # p end -> \n
    # Tail (tail of p) -> Tail
    # br -> \n
    # Bottom (block) -> \n
    # Result should be cleaned up by strip/squash logic
    assert extract_text(root6) == 'Title\nLinkBold\nTail\n\nBottom'

    # Case 7: Testing custom symbols
    root7 = MockElement('div', children=[
        MockElement('p', text='Part A'),
        MockElement('p', text='Part B')
    ])
    assert extract_text(root7, block_symbol=' | ', sep_symbol=' -> ') == 'Part A -> Part B'

    # Case 8: Testing squash_space=False
    # Should preserve the extra newlines/None entries
    root8 = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='B')
    ])
    # Without squash, we expect the raw separators to remain visible if not stripped
    result = extract_text(root8, squash_space=False)
    assert 'A' in result and 'B' in result

    # Case 9: Empty element
    root9 = MockElement('div', children=[])
    assert extract_text(root9) == ''

    # Case 10: Element with callable tag (like a function/decorator artifact)
    class CallableTag:
        def __call__(self): return True
    root10 = MockElement(CallableTag())
    assert extract_text(root10) == ''
```


# LLM-generated content at query #9
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
    # Test Case 1: Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Test Case 2: Inline tags (should not add None/True separators)
    root2 = MockElement('p', text='Hello ', children=[
        MockElement('b', text='World')
    ])
    # 'b' is in INLINE_TAGS, so no None added before or after it
    assert extract_text_array(root2) == ['Hello ', 'World']

    # Test Case 3: Block tags (should add None separators for artificial newlines)
    root3 = MockElement('div', text='Start', children=[
        MockElement('p', text='Middle', children=[
            MockElement('span', text='End')
        ])
    ])
    # div is block -> adds None at start and end. 
    # p is block -> adds None at start and end.
    # span is inline -> no separators.
    # Result after squash/strip logic in extract_text_array:
    # [None, 'Start', None, 'Middle', None, 'End', None] -> ['Start', 'Middle', 'End']
    assert extract_text_array(root3) == ['Start', 'Middle', 'End']

    # Test Case 4: Separator tag (br)
    root4 = MockElement('div', text='Line1', children=[
        MockElement('br'),
        MockElement('span', text='Line2')
    ])
    # br is in SEPARATORS -> adds True
    assert extract_text_array(root4) == ['Line1', True, 'Line2']

    # Test Case 5: Handling tails
    root5 = MockElement('div', text='Start', children=[
        MockElement('span', text='Inner', tail=' Tail')
    ])
    assert extract_text_array(root5) == ['Start', 'Inner', ' Tail']

    # Test Case 6: Nested block structures with complex spacing
    root6 = MockElement('div', text='A', children=[
        MockElement('p', text='B', children=[
            MockElement('i', text='C')
        ], tail=' C_tail')
    ])
    # Expected flow: 
    # div (block) -> [None, 'A']
    #   p (block) -> [None, 'B']
    #     i (inline) -> ['C']
    #   p tail -> ' C_tail'
    # div end -> [None]
    # Combined: [None, 'A', None, 'B', 'C', 'tail', None]
    # After _squash_artifical_nl and _strip_artifical_nl: ['A', 'B', 'C', 'tail']
    # Note: tail ' C_tail' is part of the stream.
    result = extract_text_array(root6)
    assert 'A' in result
    assert 'B' in result
    assert 'C' in result

    # Test Case 7: Callable tag (e.g., some custom logic or error state)
    class CallableTag:
        def __call__(self): pass
    root7 = MockElement(CallableTag())
    assert extract_text_array(root7) == ''

    # Test Case 8: Empty element
    root8 = MockElement('div')
    # div is block -> [None, None] -> stripped to []
    assert extract_text_array(root8) == []

    # Test Case 9: Whitespace squashing in text content
    root9 = MockElement('div', text='Hello\n\n  World')
    # WHITESPACE_RE converts \n\n to ' '
    assert extract_text_array(root9) == ['Hello World']

    # Test Case 10: Deeply nested with tail and inline
    root10 = MockElement('div', text='Outer', children=[
        MockElement('span', text='Inner', tail=' Tail')
    ])
    # div is block -> [None, 'Outer', None] (if no children tail)
    # span is inline -> ['Inner']
    # tail is ' Tail'
    # Result: [None, 'Outer', None, 'Inner', ' Tail', None]
    # Stripped: ['Outer', 'Inner', ' Tail']
    assert extract_text_array(root10) == ['Outer', 'Inner', ' Tail']

```


# LLM-generated content at query #10
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

    # Test Case 2: Nested block elements with newlines
    # <div><p>Part 1</p><p>Part 2</p></div> -> "Part 1\nPart 2"
    child1 = MockNode(tag='p', text='Part 1')
    child2 = MockNode(tag='p', text='Part 2')
    root2 = MockNode(tag='div', children=[child1, child2])
    assert extract_text(root2) == 'Part 1\nPart 2'

    # Test Case 3: Inline elements (should not trigger newlines/None)
    # <div><span>Inner</span><b>Bold</b></div> -> "InnerBold"
    child3 = MockNode(tag='span', text='Inner')
    child4 = MockNode(tag='b', text='Bold')
    root3 = MockNode(tag='div', children=[child3, child4])
    assert extract_text(root3) == 'InnerBold'

    # Test Case 4: Separator element <br> (should trigger True/sep_symbol)
    # <div>Line<br>Next</div> -> "Line\nNext"
    br_node = MockNode(tag='br')
    child5 = MockNode(tag='span', text='Next')
    root4 = MockNode(tag='div', children=[MockNode(tag='span', text='Line'), br_node, child5])
    # Note: tail of br_node is handled by the loop logic in extract_text_array
    # Let's construct it more precisely to match how lxml/DOM works
    br_node.tail = None 
    root4 = MockNode(tag='div', children=[
        MockNode(tag='span', text='Line'),
        MockNode(tag='br'),
        MockNode(tag='span', text='Next')
    ])
    assert extract_text(root4) == 'Line\nNext'

    # Test Case 5: Whitespace squashing
    # <div>  Too   much   space  </div> -> "Too much space"
    root5 = MockNode(tag='div', text='  Too   much   space  ')
    assert extract_text(root5) == 'Too much space'

    # Test Case 6: Complex structure with tails and mixed tags
    # <div>Start<p>Middle</p>End</div>
    # <p> is block (None), <span> is inline.
    p_node = MockNode(tag='p', text='Middle')
    p_node.tail = 'End'
    root6 = MockNode(tag='div', children=[MockNode(tag='span', text='Start'), p_node])
    # div (block) -> None, span (inline) -> Start, p (block) -> None, Middle, End, div (block) -> None
    # Result should be "Start\nMiddleEnd"
    assert extract_text(root6) == 'Start\nMiddleEnd'

    # Test Case 7: Custom symbols
    root7 = MockNode(tag='div', children=[MockNode(tag='p', text='A'), MockNode(tag='p', text='B')])
    assert extract_text(root7, block_symbol=' | ', sep_symbol=' > ') == 'A | B'

    # Test Case 8: No squashing enabled (raw extraction)
    root8 = MockNode(tag='div', text='  Space  ')
    assert extract_text(root8, squash_space=False) == '  Space  '

    # Test Case 9: Empty nodes
    root9 = MockNode(tag='div', children=[])
    assert extract_text(root9) == ''

    # Test Case 10: Function/Callable tag (should return empty string per code logic)
    root10 = MockNode(tag=lambda x: x)
    assert extract_text(root10) == ''
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

def test_extract_text_array():
    # Test Case 1: Simple text node (no tags)
    root1 = MockElement('div', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Test Case 2: Inline tags (should not add None/True separators)
    root2 = MockElement('p', text='Hello ', children=[
        MockElement('b', text='World', tail='!')
    ])
    # 'b' is in INLINE_TAGS, so no None added before or after it. 
    # Only the text/tail are processed.
    assert extract_text_array(root2) == ['Hello ', 'World', '!']

    # Test Case 3: Block tags (should add None separators)
    # div is not in INLINE_TAGS, so it adds [None, ..., None]
    root3 = MockElement('div', text='Start', children=[
        MockElement('p', text='Middle', tail='End')
    ])
    # extract_text_array(p) returns ['Middle'] (no separators because p is block but internal recursion has flags False)
    # However, the top level call sees div as block.
    # Process: [None, 'Start', None, 'Middle', 'End', None]
    # Squash/Strip logic applied in function:
    result3 = extract_text_array(root3)
    assert result3 == ['Start', 'Middle', 'End']

    # Test Case 4: Separator tag (br)
    root4 = MockElement('div', text='Line1', children=[
        MockElement('br'),
        MockElement('span', text='Line2')
    ])
    # br is in SEPARATORS, adds True
    result4 = extract_text_array(root4)
    assert True in result4

    # Test Case 5: Empty element
    root5 = MockElement('div')
    assert extract_text_array(root5) == []

    # Test Case 6: Nested complex structure
    # <div>Text<br/><span>Bold</span>Tail</div>
    root6 = MockElement('div', text='Text', children=[
        MockElement('br'),
        MockElement('span', text='Bold', tail='Tail')
    ])
    # Expected parts before squash: [None, 'Text', True, 'Bold', 'Tail', None]
    # After squash_artifical_nl=True (default): ['Text', True, 'Bold', 'Tail']
    result6 = extract_text_array(root6)
    assert result6 == ['Text', True, 'Bold', 'Tail']

    # Test Case 7: Callable tag (e.g. some custom objects/functions)
    class CallableTag:
        def __call__(self): pass
    root7 = MockElement(CallableTag())
    assert extract_text_array(root7) == ''

    # Test Case 8: Testing the recursion flags (squash_artifical_nl=False)
    # This tests that child elements don't squash their own None/True values during traversal
    child = MockElement('p', text='Inner')
    parent = MockElement('div', text='Outer', children=[child])
    # If we disable squash in the call, we see the raw separators
    result8 = extract_text_array(parent, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in result8
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
    # Case 1: Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Case 2: Nested block elements with newlines (None represents block separation)
    # <div><p>Part 1</p><p>Part 2</p></div> -> "Part 1\nPart 2"
    root2 = MockElement('div', children=[
        MockElement('p', text='Part 1'),
        MockElement('p', text='Part 2')
    ])
    assert extract_text(root2) == 'Part 1\nPart 2'

    # Case 3: Inline elements (no extra newlines added)
    # <div><span>Hello</span> <b>World</b></div> -> "Hello World"
    root3 = MockElement('div', children=[
        MockElement('span', text='Hello'),
        MockElement('b', text='World', tail=' ')
    ])
    assert extract_text(root3) == 'Hello World'

    # Case 4: Separator tags (br tag adds sep_symbol)
    # <div>Line 1<br>Line 2</div> -> "Line 1\nLine 2" (if sep_symbol is \n)
    root4 = MockElement('div', children=[
        MockElement('br'),
        MockElement('span', text='Line 2')
    ])
    # Note: extract_text_array adds True for 'br'. 
    # In extract_text, True becomes sep_symbol.
    assert extract_text(root4, sep_symbol='\n') == '\nLine 2'.strip()

    # Case 5: Complex structure with whitespace squashing
    # <div>  Text   \n  with \t tabs  </div>
    root5 = MockElement('div', text='  Text   \n  with \t tabs  ')
    assert extract_text(root5) == 'Text with tabs'

    # Case 6: Custom block and separator symbols
    # <div><p>A</p><p>B</p></div> -> "A|B"
    root6 = Mockelse = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='B')
    ])
    assert extract_text(root6, block_symbol='|', sep_symbol='|') == 'A|B'

    # Case 7: Deeply nested structure
    # <div><ul><li>Item 1</li><li>Item 2</li></ul></div>
    root7 = MockElement('div', children=[
        MockElement('ul', children=[
            MockElement('li', text='Item 1'),
            MockElement('li', text='Item 2')
        ])
    ])
    assert extract_text(root7) == 'Item 1\nItem 2'

    # Case 8: Elements with tails and no text
    # <div><p>Start</p>Tail</div>
    root8 = MockElement('div', children=[
        MockElement('p', text='Start', tail='Tail')
    ])
    assert extract_text(root8) == 'StartTail'

    # Case 9: Function returning empty string (callable tag)
    root9 = MockElement(lambda x: x, text='Hidden')
    assert extract_text(root9) == ''

    # Case 10: Handling of None/Empty parts via _merge_original_parts logic
    # Ensure that multiple consecutive block elements don't create double newlines
    root10 = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='B'),
        MockElement('p', text='C')
    ])
    assert extract_text(root10) == 'A\nB\nC'

def Mockelse: # Helper for clean code above
    pass
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
        self._children = children or []

    def getchildren(self):
        return self._children

def test_extract_text():
    # Case 1: Simple text node
    root1 = MockNode(tag='div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Case 2: Nested block elements (div inside div) should introduce newlines
    # Structure: <div><div>Text</div></div> -> 'Text' with artificial NLs
    root2 = MockNode(tag='div', text=None, children=[
        MockNode(tag='div', text='Inner', tail=None)
    ])
    assert extract_text(root2) == 'Inner'

    # Case 3: Inline elements (span) should NOT introduce newlines
    # Structure: <div><span>Part1</span><span>Part2</span></div> -> 'Part1Part2'
    root3 = MockNode(tag='div', text=None, children=[
        MockNode(tag='span', text='Part1', tail=None),
        MockNode(tag='span', text='Part2', tail=None)
    ])
    assert extract_text(root3) == 'Part1Part2'

    # Case 4: Separator element (br) should introduce sep_symbol
    root4 = MockNode(tag='div', text='Line1', children=[
        MockNode(tag='br', text=None, tail=None),
        MockNode(tag='span', text='Line2')
    ])
    # By default sep_symbol is '\n'
    assert extract_text(root4) == 'Line1\nLine2'

    # Case 5: Whitespace squashing
    root5 = MockNode(tag='div', text='  Too    many   spaces  ', children=[])
    assert extract_text(root5) == 'Too many spaces'

    # Case 6: Complex structure with tails and blocks
    # <div>
    #    <p>Hello</p>
    #    <span>World</span>
    # </div>
    root6 = MockNode(tag='div', text=None, children=[
        MockNode(tag='p', text='Hello', tail=None),
        MockNode(tag='span', text='World', tail=None)
    ])
    # p is block -> adds NL. span is inline -> no extra NL. 
    # But the div itself wraps content in NLs which are stripped.
    assert extract_text(root6) == 'Hello\nWorld'

    # Case 7: Custom symbols
    root7 = MockNode(tag='div', text=None, children=[
        MockNode(tag='p', text='A', tail=None),
        MockNode(tag='br', text=None, tail=None),
        MockNode(tag='p', text='B', tail=None)
    ])
    # block_symbol='\n', sep_symbol='|'
    assert extract_text(root7, block_symbol='\n', sep_symbol='|') == 'A\n|\nB'

    # Case 8: Empty nodes
    root8 = MockNode(tag='div', text=None, children=[])
    assert extract_text(root8) == ''

    # Case 9: Elements with tails
    root9 = MockNode(tag='div', text=None, children=[
        MockNode(tag='span', text='Start', tail=' End')
    ])
    assert extract_text(root9) == 'Start End'

    # Case 10: Non-string callable tag (e.g., custom object logic in some parsers)
    class CallableTag:
        def __call__(self): return True
    root10 = MockNode(tag=CallableTag())
    assert extract_text(root10) == ''
```


# LLM-generated content at query #14
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
    # Case 1: Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Case 2: Nested block elements (div inside div) should introduce newlines
    root2 = MockElement('div', children=[
        MockElement('p', text='Line 1'),
        Mockelse := MockElement('p', text='Line 2')
    ])
    assert extract_text(root2) == 'Line 1\nLine 2'

    # Case 3: Inline elements (span, b) should NOT introduce newlines
    root3 = MockElement('div', children=[
        MockElement('p', children=[
            MockElement('span', text='Hello '),
            MockElement('b', text='World')
        ])
    ])
    assert extract_text(root3) == 'Hello World'

    # Case 4: Separator element (br) should introduce the sep_symbol
    root4 = MockElement('div', children=[
        MockElement('p', children=[
            MockElement('b', text='Part A'),
            MockElement('br'),
            MockElement('b', text='Part B')
        ])
    ])
    # br triggers True in extract_text_array, which becomes sep_symbol (\n)
    assert extract_text(root4) == 'Part A\nPart B'

    # Case 5: Handling of tails (text following a tag)
    root5 = MockElement('div', children=[
        MockElement('span', text='Start', tail=' End')
    ])
    assert extract_text(root5) == 'Start End'

    # Case 6: Whitespace squashing
    root6 = MockElement('div', text='  Too   much\n\nwhitespace  ')
    assert extract_text(root6) == 'Too much whitespace'

    # Case 7: Complex structure with mixed block/inline and tails
    root7 = MockElement('div', children=[
        MockElement('h1', text='Title'),
        MockElement('p', children=[
            MockElement('strong', text='Bold'),
            MockElement('span', text=' Plain', tail='!'),
        ])
    ])
    # h1 is block -> \n. p is block -> \n at start and end. 
    # Internal elements are inline, no extra newlines from them.
    assert extract='Title\nBold Plain!'

    # Case 8: Custom symbols
    root8 = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='B')
    ])
    assert extract_text(root8, block_symbol=' | ', sep_symbol=' -> ') == 'A | B'

    # Case 9: Empty element
    root9 = MockElement('div', children=[MockElement('span')])
    assert extract_text(root9) == ''

    # Case 10: Function-like tag (edge case from code)
    class FuncTag:
        def __call__(self): pass
    root10 = MockElement(FuncTag())
    assert extract_text(root10) == ''
```


# LLM-generated content at query #15
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
    # Test Case 1: Simple text node (no tags)
    root1 = MockElement('div', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Test Case 2: Inline elements (should not add None/True separators)
    root2 = MockElement('p', text='Start ', children=[
        MockElement('b', text='Bold'),
        MockElement('i', text='Italic')
    ], tail=' End')
    # 'b' and 'i' are in INLINE_TAGS, so no None/True added around them
    assert extract_text_array(root2) == ['Start ', 'Bold', ' Italic', ' End']

    # Test Case 3: Block elements (should add None separators)
    root3 = MockElement('div', text='Top', children=[
        MockElement('p', text='Middle', children=[
            MockElement('span', text='Inner')
        ])
    ], tail=' Bottom')
    # div is block -> adds None at start and end. p is block -> adds None.
    # Resulting array before squash: [None, 'Top', None, 'Middle', None, 'Inner', None, ' End', None]
    # After _squash_artifical_nl and _strip_artally_nl:
    res3 = extract_text_array(root3)
    assert res3 == ['Top', 'Middle', 'Inner', ' End']

    # Test Case 4: Separator tag (br) should add True
    root4 = MockElement('div', text='Line1', children=[
        MockElement('br')
    ], tail='Line2')
    assert extract_text_array(root4) == [True, 'Line2']

    # Test Case 5: Empty element
    root5 = MockElement('div')
    assert extract_text_array(root5) == []

    # Test Case 6: Testing squash_artifical_nl=False logic
    # When False, the recursion uses different parameters to preserve structure
    root6 = MockElement('div', text='A', children=[MockElement('p', text='B')])
    res6 = extract_text_array(root6, squash_artifical_nl=False)
    # The internal call for 'p' will have Nones around it, 
    # but the top level call might treat them differently if called manually.
    # Since we test the public API, we check that it doesn't strip the None from the child
    assert None in res6

    # Test Case 7: Complex nested structure with whitespace and tails
    root7 = MockElement('div', text='  \n  ', children=[
        MockElement('span', text='Content'),
        MockElement('a', text='Link')
    ], tail='  tail  ')
    # '  \n  ' is squashed to ' ' by squash_html_whitespace inside the logic if processed via merge,
    # but extract_text_array uses WHITESPACE_RE on content indirectly via other functions.
    # Here we check the raw structure of the array returned.
    res7 = extract_text_array(root7)
    assert 'Content' in res7
    assert 'Link' in res7
    assert '  tail  ' in res7

    # Test Case 8: Function with callable tag (should return '')
    class CallableTag:
        def __call__(self): pass
    root8 = MockElement(CallableTag())
    assert extract_text_array(root8) == ''

    # Test Case 9: Verify strip_artifical_nl removes leading/trailing Nones
    root9 = MockElement('div', text=None, children=[
        MockElement('p', text='Content')
    ], tail=None)
    # div is block -> adds None at start and end. p is block -> adds None.
    # The array should be stripped of the outer Nones.
    assert extract_text_array(root9) == ['Content']
```


# LLM-generated content at query #16
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
    # Test Case 1: Simple text node (no tags)
    root1 = MockElement('div', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Test Case 2: Inline tags (should not add None/separator)
    root2 = MockElement('p', text='Hello ', children=[
        MockElement('b', text='World')
    ])
    # 'b' is in INLINE_TAGS, so no None/True added around it
    assert extract_text_array(root2) == ['Hello ', 'World']

    # Test Case 3: Block tags (should add None for artificial newlines)
    root3 = MockElement('div', text='Start', children=[
        MockElement('p', text='Middle')
    ])
    # div is block -> adds None at start and end. p is block -> adds None at start and end.
    # extract_text_array uses squash_artifical_nl=False in recursion, but top level defaults to True.
    # With squash=True: [None, 'Start', None, 'Middle', None] -> [None, 'Start', 'Middle', None]
    # strip_artifical_nl strips leading/trailing None.
    assert extract_text_array(root3) == ['Start', 'Middle']

    # Test Case 4: Separator tag (br)
    root4 = MockElement('div', text='Line1', children=[
        MockElement('br'),
        MockElement('span', text='Line2')
    ])
    # br is in SEPARATORS -> adds True.
    assert extract_text_array(root4) == ['Line1', True, 'Line2']

    # Test Case 5: Tail text handling
    root5 = MockElement('div', text='Start', children=[
        MockElement('span', text='Inner', tail=' End')
    ])
    assert extract_text_array(root5) == ['Start', 'Inner', ' End']

    # Test Case 6: Deeply nested block elements with whitespace squashing
    root6 = MockElement('div', text='  A  ', children=[
        MockElement('p', text='B', children=[
            MockElement('div', text='C')
        ])
    ])
    # Whitespace in WHITESPACE_RE is squashed by squash_html_whitespace (used via extract_text logic)
    # However, extract_text_array itself doesn't call squash_html_whitespace on the strings 
    # unless we are looking at how it interacts with text. 
    # Let's test the structure specifically.
    res = extract_text_array(root6)
    assert 'A' in res or '  A  ' in res # Depends on if WHITESPACE_RE is applied to raw text

    # Test Case 7: Empty element
    root7 = MockElement('div')
    assert extract_text_array(root7) == []

    # Test Case 8: Callable tag (e.g., a function object as tag)
    root8 = MockElement(lambda x: x)
    assert extract_text_array(root8) == ''

    # Test Case 9: Complex structure with mixed inline and block
    # <div>Start <p>Middle</p> End</div>
    root9 = MockElement('div', text='Start ', children=[
        MockElement('p', text='Middle', tail=' End')
    ])
    # div (block) -> [None, 'Start ', ...]
    # p (block) -> [..., None, 'Middle', ' End', None]
    # Squash/Strip logic applied at top level:
    assert extract_text_array(root9) == ['Start ', 'Middle', ' End']

    # Test Case 10: Verify strip_artifical_nl behavior with None at edges
    root10 = MockElement('div', text='A', children=[MockElement('b', text='B')])
    # If we set strip_artifical_nl=False, we should see the Nones
    res_no_strip = extract_text_array(root10, strip_artifical_nl=False)
    assert res_no_strip == [None, 'A', 'B']

```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
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
    # Test Case 1: Simple text node (no tags)
    # Note: In a real DOM, the root is usually a block element like <div> or <html>
    root1 = MockElement('div', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Test Case 2: Nested inline tags (span inside div)
    child_span = MockElement('span', text='World')
    root2 = MockElement('div', text='Hello ', children=[child_span])
    child_span.tail = '!'
    assert extract_text_array(root2) == ['Hello ', 'World', '!']

    # Test Case 3: Block elements (should introduce None/newlines)
    child_p = MockElement('p', text='Paragraph')
    root3 = MockElement('div', text='Start', children=[child_p])
    child_p.tail = 'End'
    # div is block -> starts with None, ends with None. p is block -> starts/ends with None.
    # extract_text_array (squash=True) should collapse these Nones.
    result3 = extract_text_array(root3)
    assert 'Start' in result3
    assert 'Paragraph' in result3
    assert 'End' in result3

    # Test Case 4: Separator tag (br)
    child_br = MockElement('br')
    root4 = MockElement('div', text='Line1', children=[child_br])
    child_br.tail = 'Line2'
    # br is in SEPARATORS, so it adds True
    assert extract_text_array(root4) == ['Line1', True, 'Line2']

    # Test Case 5: Inline tags (should not introduce None/newlines)
    child_b = MockElement('b', text='Bold')
    root5 = MockElement('div', text='Text ', children=[child_b])
    child_b.tail = ' more'
    assert extract_text_array(root5) == ['Text ', 'Bold', ' more']

    # Test Case 6: Empty element
    root6 = MockElement('div')
    assert extract_text_array(root6) == []

    # Test Case 7: testing squash_artifical_nl=False
    child_p_inner = MockElement('p', text='Inner')
    root7 = MockElement('div', text='Outer', children=[child_p_inner])
    # Without squashing, we expect the raw None markers from block elements
    result7 = extract_text_array(root7, squash_artifical_nl=False)
    assert None in result7
    assert 'Outer' in result7
    assert 'Inner' in result7

    # Test Case 8: Testing strip_artifical_nl=False
    child_p_strip = MockElement('p', text='NoStrip')
    root8 = MockElement('div', text='Start', children=[child_p_strip])
    result8 = extract_text_array(root8, strip_artifical_nl=False)
    # Should contain leading/trailing None markers if present
    assert any(x is None for x in result8)

    # Test Case 9: Callable tag (edge case from code)
    class CallableTag:
        def __call__(self): pass
    root9 = MockElement(CallableTag())
    assert extract_text_array(root9) == ''

    # Test Case 10: Complex structure with whitespace squashing
    # div (block) -> [None, 'A']
    #   p (block) -> [None, 'B', None]
    #   tail of p -> 'C'
    # result should be ['A', 'B', 'C'] after stripping/squashing Nones
    child_p_complex = MockElement('p', text='B')
    child_p_complex.tail = 'C'
    root10 = MockElement('div', text='A', children=[child_p_complex])
    assert extract_text_array(root10) == ['A', 'B', 'C']
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
    # Case 1: Simple block element with text
    root1 = MockElement('div', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Case 2: Inline element (should not add None/True separators)
    root2 = MockElement('span', text='World')
    assert extract_text_array(root2) == ['World']

    # Case 3: Nested block elements (should introduce None for newlines)
    child1 = MockElement('p', text='Inner')
    root3 = MockElement('div', text='Outer ', children=[child1])
    child1.tail = ' End'
    # div starts -> None, text='Outer ', child p starts -> None, text='Inner', 
    # tail=' End', div ends -> None. Squash and strip applied by extract_text_array default.
    # Trace: [None, 'Outer ', None, 'Inner', ' End', None]
    # _squash_artifical_nl converts consecutive Nones to one None.
    # _strip_artifical_nl removes leading/trailing Nones.
    assert extract_text_array(root3) == ['Outer ', 'Inner', ' End']

    # Case 4: Separator tag (br) should introduce True
    root4 = MockElement('br')
    assert extract_text_array(root4) == [True]

    # Case 5: Complex structure with mixed inline and block
    # <div><span>A</span><b>B</b></div>
    child_span = MockElement('span', text='A')
    child_b = MockElement('b', text='B')
    root5 = MockElement('div', text='Start ', children=[child_span, child_b])
    child_span.tail = ' Middle'
    child_b.tail = ' End'
    # Expected: [None (div start), 'Start ', 'A' (span), ' Middle' (tail), 'B' (b), ' End' (tail), None (div end)]
    # After stripping Nones: ['Start ', 'A', ' Middle', 'B', ' End']
    assert extract_text_array(root5) == ['Start ', 'A', ' Middle', 'B', ' End']

    # Case 6: Empty element
    root6 = MockElement('div')
    assert extract_text_array(root6) == []

    # Case 7: Function-like tag (callable)
    root7 = MagicMock()
    root7.tag = lambda x: x
    assert extract_text_array(root7) == ''

    # Case 8: Testing squash_artifical_nl=False and strip_artifical_nl=False
    # This should preserve the raw None/True markers
    root8 = MockElement('div', text='A', children=[MockElement('br')])
    # [None (div start), 'A', True (br), None (div end)]
    assert extract_text_array(root8, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'A', True, None]

    # Case 9: Testing with tails and specific separators
    child_a = MockElement('a', text='Link')
    child_a.tail = ' follow'
    root9 = MockElement('div', text='Go ', children=[child_a])
    assert extract_text_array(root9) == ['Go ', 'Link', ' follow']
```


# LLM-generated content at query #3
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
    # Test Case 1: Simple inline tag (span)
    el1 = MockElement('span', text='hello')
    assert extract_text_array(el1) == ['hello']

    # Test Case 2: Block tag (div) with artificial newline marker (None)
    el2 = MockElement('div', text='start ')
    child2 = MockElement('p', text='middle', tail=' end')
    el2.children.append(child2)
    # div is not in INLINE_TAGS, so it adds None at start and end
    assert extract_text_array(el2) == [None, 'start ', 'middle', ' end', None]

    # Test Case 3: Separator tag (br)
    el3 = MockElement('br')
    assert extract_text_array(el3) == [True]

    # Test Case 4: Nested structure with mixed tags
    # <div><p><b>bold</b> text</p></div>
    root = MockElement('div', text='outer ')
    p = MockElement('p', text='inner ')
    b = MockElement('b', text='bold')
    b.tail = ' text'
    p.children.append(b)
    root.children.append(p)
    # Expected: [None (div start), 'outer ', None (p start), 'inner ', 'bold', ' text', None (p end), None (div end)]
    # After _squash_artifical_nl and _strip_artifical_nl
    result = extract_text_array(root)
    assert None not in result
    assert 'outer ' in result
    assert 'bold' in result
    assert ' text' in result

    # Test Case 5: Empty element
    el4 = MockElement('div')
    assert extract_text_array(el4) == []

    # Test Case 6: Elements with only tail
    el5 = MockElement('div', text='start')
    child5 = MockElement('span', text='inner')
    child5.tail = ' end'
    el5.children.append(child5)
    # div (block) adds None at start/end. span is inline, no None added by tag itself.
    # structure: [None, 'start', 'inner', ' end', None]
    # after stripping: ['start', 'inner', ' end']
    assert extract_text_array(el5) == ['start', 'inner', ' end']

    # Test Case 7: Callable tag (like a function/special node in some DOM libs)
    class CallableTag:
        def __init__(self):
            self.tag = lambda x: x
    el6 = CallableTag()
    assert extract_text_array(el6) == ''

    # Test Case 8: Checking whitespace squashing within the array logic
    el7 = MockElement('div', text='  spaced  ')
    result7 = extract_text_array(el7, squash_artifical_nl=True)
    assert result7 == ['  spaced  '] # Note: extract_text_array doesn't call squash_html_whitespace on the string itself, only handles None

    # Test Case 9: Complex nesting with stripping enabled
    root_complex = MockElement('div', text='A')
    child_complex = MockElement('p', text='B')
    child_complex.tail = 'C'
    root_complex.children.append(child_complex)
    # [None, 'A', None, 'B', 'C', None] -> strip -> ['A', 'B', 'c']
    assert extract_text_array(root_complex) == ['A', 'B', 'C']

    # Test Case 10: Testing squash_artifical_nl=False
    el8 = MockElement('div', text='start')
    el8.children.append(MockElement('p', text='end'))
    # If False, we expect the None markers to remain
    assert None in extract_text_array(el8, squash_artifical_nl=False)
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
    # Case 1: Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Case 2: Nested block elements (should introduce newlines/None separators)
    # <div><p>Part 1</p><p>Part 2</p></div> -> "Part 1\nPart 2"
    root2 = MockElement('div', children=[
        MockElement('p', text='Part 1'),
        MockElement('p', text='Part 2')
    ])
    assert extract_text(root2) == 'Part 1\nPart 2'

    # Case 3: Inline elements (should not introduce newlines)
    # <div><span>Inside</span><span>Text</span></div> -> "InsideText"
    root3 = MockElement('div', children=[
        MockElement('span', text='Inside'),
        MockElement('span', text='Text')
    ])
    assert extract_text(root3) == 'InsideText'

    # Case 4: Separator elements (br tag)
    # <div>Line 1<br>Line 2</div> -> "Line 1\nLine 2" (using sep_symbol default)
    root4 = MockElement('div', children=[
        MockElement('p', text='Line 1'),
        MockElement('br'),
        MockElement('p', text='Line 2')
    ])
    # Note: extract_text_array returns True for br. 
    # extract_text replaces True with sep_symbol (\n)
    assert extract_text(root4) == 'Line 1\nLine 2'

    # Case 5: Whitespace squashing
    # <div>  Too   Much   Space  </div> -> "Too Much Space"
    root5 = MockElement('div', text='  Too   Much   Space  ')
    assert extract_text(root5) == 'Too Much Space'

    # Case 6: Tail text handling
    # <div><p>Text</p>Tail</div> -> "Text\nTail"
    root6 = MockElement('div', children=[
        MockElement('p', text='Text', tail='Tail')
    ])
    assert extract_text(root6) == 'Text\nTail'

    # Case 7: Complex structure with mixed inline/block and tails
    # <div><p><b>Bold</b><em>Italic</em></p>End</div>
    root7 = MockElement('div', children=[
        MockElement('p', children=[
            MockElement('b', text='Bold'),
            MockElement('em', text='Italic')
        ], tail='End')
    ])
    # p is block, b/em are inline. 
    # Structure: [None, 'p', None, 'b', 'Bold', 'em', 'Italic', 'End', None]
    assert extract_text(root7) == 'BoldItalic\nEnd'

    # Case 8: Custom block and separator symbols
    root8 = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='B')
    ])
    assert extract_text(root8, block_symbol=' | ', sep_symbol=' -> ') == 'A | B'

    # Case 9: Function with callable tag (should return empty string)
    class CallableTag:
        def __call__(self): pass
    root9 = MockElement(CallableTag())
    assert extract_text(root9) == ''

    # Case 10: Deeply nested structure
    root10 = MockElement('div', children=[
        MockElement('section', children=[
            MockElement('div', children=[
                MockElement('span', text='Deep')
            ])
        ])
    ])
    assert extract_text(root10) == 'Deep'
```


# LLM-generated content at query #5
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
    assert extract_template_text_wrapper(root1) == 'Hello'

    # Test Case 2: Nested block elements with artificial newlines
    # <div><p>Line 1</p><p>Line 2</p></div> -> "Line 1\nLine 2"
    child1 = MockElement('p', text='Line 1')
    child2 = MockElement('p', text='Line 2')
    root2 = MockElement('div', children=[child1, child2])
    assert extract_template_text_wrapper(root2) == 'Line 1\nLine 2'

    # Test Case 3: Inline elements (should not trigger newlines)
    # <div><span>Part 1</span><span>Part 2</span></div> -> "Part 1Part 2"
    child3 = MockElement('span', text='Part 1')
    child4 = MockElement('span', text='Part 2')
    root3 = MockElement('div', children=[child3, child4])
    assert extract_template_text_wrapper(root3) == 'Part 1Part 2'

    # Test Case 4: Separator element <br>
    # <div>A<br>B</div> -> "A\nB" (assuming sep_symbol is \n)
    child5 = MockElement('br')
    child6 = MockElement('span', text='B')
    child6.tail = None # ensure no tail interference
    root4 = MockElement('div', children=[child5, child6])
    # Note: extract_text_array returns True for <br> in SEPARATORS
    assert extract_template_text_wrapper(root4) == 'A\nB' if 'A' in str(root4) else True 
    # Re-verifying logic: extract_text uses sep_symbol for True (from br)
    root4 = MockElement('div', children=[MockElement('a', text='A'), MockElement('br'), MockElement('b', text='B')])
    assert extract_template_text_wrapper(root4) == 'A\nB'

    # Test Case 5: Whitespace squashing
    root5 = MockElement('div', text='  Too   much \t whitespace  ')
    assert extract_template_text_wrapper(root5) == 'Too much whitespace'

    # Test Case 6: Complex structure with tails and nesting
    # <div>Outer<p>Inner</p>Tail</div>
    inner = MockElement('p', text='Inner')
    root6 = MockElement('div', text='Outer', children=[inner])
    inner.tail = 'Tail'
    assert extract_template_text_wrapper(root6) == 'Outer\nInner\nTail'

    # Test Case 7: Empty elements
    root7 = MockElement('div', children=[MockElement('p')])
    assert extract_template_text_wrapper(root7) == ''

def extract_template_text_wrapper(dom):
    """Helper to call the actual function with default params."""
    return extract_text(dom)

def test_extract_text_custom_symbols():
    # Testing custom block and sep symbols
    child = MockElement('p', text='Content')
    root = MockElement('div', children=[child])
    # Default is \n for None (blocks) and \n for True (br)
    assert extract_text(root, block_symbol='|', sep_symbol='~') == 'Content'
    
    # Testing <br> with custom separator
    child_br = MockElement('br')
    child_txt = MockElement('span', text='Next')
    root_br = MockElement('div', children=[child_br, child_txt])
    assert extract_text(root_br, sep_symbol='~') == 'Next' # Since br is True, it joins with ~

def test_extract_text_no_squash():
    # When squash_space is False, whitespace should remain
    root = MockElement('div', text='  Space  ')
    assert extract_text(root, squash_space=False) == '  Space  '
```


# LLM-generated content at query #6
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
    # Test Case 1: Simple text node (no tags)
    root1 = MockElement('div', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Test Case 2: Inline tags (should not trigger None/True markers)
    root2 = MockElement('span', text='Inline', tail=' content')
    assert extract_text_array(root2) == ['Inline', ' content']

    # Test Case 3: Block-level tags (should trigger None markers for newlines)
    # div is not in INLINE_TAGS, so it adds None at start and end
    root3 = MockElement('div', text='Block')
    # Expected behavior: [None, 'Block', None] -> stripped to ['Block']
    assert extract_text_array(root3) == ['Block']

    # Test Case 4: Separator tag (br)
    root4 = MockElement('br')
    # br is in SEPARATORS, so it adds True
    assert extract_text_array(root4) == [True]

    # Test Case 5: Nested structure with mixed tags
    # <div><p>Text <span>inner</span></p>Tail</div>
    child_span = MockElement('span', text='inner')
    child_p = MockElement('p', text='Text ', children=[child_span], tail='Tail')
    root5 = MockElement('div', children=[child_p])
    
    # Trace:
    # p is block -> [None, 'Text ']
    # span is inline -> ['inner']
    # child_p.tail is 'Tail' -> ['Tail']
    # p end -> [None]
    # div start/end -> [None, ..., None]
    # Result after squash/strip: ['Text ', 'inner', 'Tail']
    assert extract_text_array(root5) == ['Text ', 'inner', 'Tail']

    # Test Case 6: Whitespace squashing in text
    root6 = MockElement('div', text='Line\n\t  \u200BBreak')
    assert extract_text_array(root6) == ['Line Break']

    # Test Case 7: Function/Callable tag (should return empty string)
    class CallableTag:
        def __call__(self): pass
    root7 = MockElement(CallableTag())
    assert extract_text_array(root7) == ''

    # Test Case 8: Complex nesting with artificial newlines
    # <div><p>A</p><br/><b>B</b></div>
    child_b = MockElement('b', text='B')
    child_br = MockElement('br')
    child_p = MockElement('p', text='A', children=[child_br, child_b])
    root8 = MockElement('div', children=[child_p])

    # Inside p: [None (start), 'A', True (br), 'B', None (end)]
    # Inside div: [None (start), ..., None (end)]
    # After _squash_artifical_nl and _strip_artifical_nl: ['A', True, 'B']
    assert extract_text_array(root8) == ['A', True, 'B']

    # Test Case 9: Testing strip_artifical_nl=False preservation
    child_p_raw = MockElement('p', text='A')
    root9 = MockElement('div', children=[child_p_raw])
    # Without stripping, the None markers from block tags should remain
    res = extract_text_array(root9, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in res
    assert 'A' in res

    # Test Case 10: Empty element
    root10 = MockElement('div')
    assert extract_text_array(root10) == []
```


# LLM-generated content at query #7
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
    # Mocking different DOM structures
    
    # 1. Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # 2. Nested block elements with artificial newlines
    # <div><p>Part 1</p><p>Part 2</p></div> -> "Part 1\nPart 2"
    child1 = MockElement('p', text='Part 1')
    child2 = MockElement('p', text='Part 2')
    root2 = MockElement('div', children=[child1, child2])
    assert extract_text(root2) == 'Part 1\nPart 2'

    # 3. Inline elements (should not trigger newlines/None)
    # <div><span>Hello</span> <b>World</b></div> -> "Hello World"
    span = MockElement('span', text='Hello')
    b_tag = MockElement('b', text='World')
    span.tail = ' '
    root3 = MockElement('div', children=[span, b_tag])
    assert extract_text(root3) == 'Hello World'

    # 4. Separator element (br)
    # <div>Line 1<br>Line 2</div> -> "Line 1\nLine 2"
    br = MockElement('br')
    line1 = MockElement('span', text='Line 1')
    br.tail = ''
    line1.tail = None # or handle via child structure
    # Re-constructing to ensure br is a child with tail
    root4 = MockElement('div', children=[
        MockElement('span', text='Line 1'),
        MockElement('br', tail=''),
        MockElement('span', text='Line 2')
    ])
    # Note: extract_text_array logic adds True for 'br' tag
    assert extract_text(root4) == 'Line 1\nLine 2'

    # 5. Whitespace squashing
    # <div>  Too   much   space  </div> -> "Too much space"
    root5 = MockElement('div', text='  Too   much   space  ')
    assert extract_text(root5) == 'Too much space'

    # 6. Complex nested structure
    # <div><div><p>A</p><span>B</span></p><p>C</p></div> -> "A B\nC"
    inner_span = MockElement('span', text='B')
    inner_p = MockElement('p', text='A', children=[inner_span])
    inner_span.tail = '' 
    inner_p.tail = None
    
    child_p2 = MockElement('p', text='C')
    root6 = MockElement('div', children=[inner_p, child_p2])
    # The logic for block elements adds None (newline) at start and end of the loop
    assert extract_text(root6) == 'A B\nC'

    # 7. Custom symbols
    # Test custom block/sep symbols
    root7 = MockElement('div', children=[
        MockElement('p', text='Part1'),
        MockElement('p', text='Part2')
    ])
    assert extract_text(root7, block_symbol='|', sep_symbol='-') == 'Part1-Part2'

    # 8. Testing squash_space=False
    # Should preserve the extra spaces and potentially raw newlines if configured
    root8 = MockElement('div', text='A   B')
    assert extract_text(root8, squash_space=False) == 'A   B'

    # 9. Empty elements
    root9 = MockElement('div', children=[MockElement('p'), MockElement('p')])
    assert extract_text(root9) == ''

    # 10. Function that returns empty string (callable tag)
    root10 = MagicMock()
    root10.tag = lambda: True
    assert extract_text(root10) == ''
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
    # Test 1: Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_template_text_wrapper(root1) == 'Hello'

    # Test 2: Nested block elements (should introduce newlines/None separators)
    root2 = MockElement('div', children=[
        MockElement('p', text='Paragraph 1'),
        MockElement('p', text='Paragraph 2')
    ])
    # div -> None, p -> None, text 'Paragraph 1', p -> None, text 'Paragraph 2', div -> None
    # result should be 'Paragraph 1\nParagraph 2'
    assert extract_template_text_wrapper(root2) == 'Paragraph 1\nParagraph 2'

    # Test 3: Inline elements (should not introduce newlines)
    root3 = MockElement('div', children=[
        MockElement('p', children=[
            MockElement('span', text='Inline'),
            MockElement('b', text=' Bold')
        ])
    ])
    assert extract_template_text_wrapper(root3) == 'Inline Bold'

    # Test 4: Separator tags (br tag should introduce True/sep_symbol)
    root4 = MockElement('div', children=[
        MockElement('p', children=[
            MockElement('span', text='Line 1'),
            MockElement('br'),
            MockElement('span', text='Line 2')
        ])
    ])
    # br is in SEPARATORS, so it injects True, which becomes sep_symbol (\n)
    assert extract_template_text_wrapper(root4) == 'Line 1\nLine 2'

    # Test 5: Whitespace squashing
    root5 = MockElement('div', text='  Too   much \n whitespace  ')
    assert extract_template_text_wrapper(root5) == 'Too much whitespace'

    # Test 6: Tail text handling
    root6 = MockElement('div', children=[
        MockElement('span', text='Start', tail=' End')
    ])
    assert extract_template_text_wrapper(root6) == 'Start End'

    # Test 7: Complex structure with mixed block, inline and tails
    root7 = MockElement('div', children=[
        MockElement('h1', text='Title'),
        MockElement('p', children=[
            MockElement('a', text='Link'),
            MockElement('b', text='Bold')
        ], tail='!'),
        MockElement('br')
    ])
    # Expected: Title (block) \n Link (inline) Bold (inline) ! (tail) \n br (sep)
    assert extract_template_text_wrapper(root7) == 'Title\nLink Bold!\n'

def extract_template_text_wrapper(dom):
    """Helper to call the function with default params as requested in signature."""
    return extract_text(dom)

def test_extract_text_custom_symbols():
    # Test custom block and separator symbols
    root = MockElement('div', children=[
        MockElement('p', text='Part 1'),
        MockElement('p', text='Part 2')
    ])
    assert extract_text(root, block_symbol=' | ', sep_symbol=' -> ') == 'Part 1 | Part 2'

def test_extract_text_no_squash():
    # Test with squash_space=False to see raw behavior (less stripping)
    root = MockElement('div', text='  Keep Spaces  ')
    # When squash_space is False, the strip() at the end of extract_text isn't called 
    # but the internal logic still processes. However, according to the code:
    # if squash_space: result = result.strip()
    # If we pass False, it should preserve leading/trailing spaces in the string.
    assert extract_text(root, squash_space=False) == '  Keep Spaces  '
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
    assert extract_template_text(root1) == 'Hello'

    # Test Case 2: Nested block elements with whitespace/newlines
    # <div><p>Line 1</p><p>Line 2</p></div> -> "Line 1\nLine 2"
    p1 = MockElement('p', text='Line 1')
    p2 = MockElement('p', text='Line 2')
    root2 = MockElement('div', children=[p1, p2])
    assert extract_text(root2) == 'Line 1\nLine 2'

    # Test Case 3: Inline elements (should not trigger newlines)
    # <div><span>Part 1</span><span>Part 2</span></div> -> "Part 1Part 2"
    span1 = MockElement('span', text='Part 1')
    span2 = MockElement('span', text='Part 2')
    root3 = MockElement('div', children=[span1, span2])
    assert extract_text(root3) == 'Part 1Part 2'

    # Test Case 4: Separator element <br> (should trigger sep_symbol)
    # <div>A<br>B</div> -> "A\nB"
    br = MockElement('br')
    p_a = MockElement('span', text='A')
    p_b = MockElement('span', text='B')
    br.tail = 'B' # tail of br is part of the parent's content flow
    # Reconstructing structure: <div><span>A</span><br>B</div>
    root4 = MockElement('div', children=[p_a, br])
    # Note: In lxml-style logic, if br is a child, its tail belongs to the parent.
    assert extract_text(root4) == 'A\nB'

    # Test Case 5: Complex structure with mixed types
    # <div><p><b>Bold</b> Text</p></div> -> "Bold Text"
    b = MockElement('b', text='Bold')
    p = MockElement('p', children=[b], text=' ') # space as text
    b.tail = ' Text'
    root5 = MockElement('div', children=[p])
    assert extract_text(root5) == 'Bold Text'

    # Test Case 6: Custom symbols
    # Testing block_symbol and sep_symbol parameters
    root6 = MockElement('div', children=[MockElement('p', text='P1'), MockElement('p', text='P2')])
    assert extract_text(root6, block_symbol='|', sep_symbol='-') == 'P1-P2'

    # Test Case 7: Whitespace squashing
    # <div>  Multiple   Spaces  </div> -> "Multiple Spaces"
    root7 = MockElement('div', text='  Multiple   Spaces  ')
    assert extract_text(root7) == 'Multiple Spaces'

    # Test Case 8: Function/Callable tag (should return empty string)
    class CallableTag:
        def __call__(self): pass
    root8 = MockElement(CallableTag())
    assert extract_text(root8) == ''

    # Test Case 9: Elements with tails
    # <div><span>Start</span>Tail</div> -> "StartTail"
    span = MockElement('span', text='Start')
    span.tail = 'Tail'
    root9 = MockElement('div', children=[span])
    assert extract_text(root9) == 'StartTail'

# Helper to avoid name collision in test runner if needed, 
# but the prompt asks for a specific signature.
def extract_template_text(dom):
    return extract_text(dom)
```


# LLM-generated content at query #10
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

    # Test Case 2: Nested block elements (div inside div)
    # Should introduce None/new-line markers and strip them
    root2 = MockElement('div', children=[
        MockElement('p', text='Line 1'),
        MockElement('p', text='Line 2')
    ])
    # extract_text with default squash_space=True will join parts with \n
    assert extract_text(root2) == 'Line 1\nLine 2'

    # Test Case 3: Inline elements (span, b) - should not introduce newlines
    root3 = MockElement('div', children=[
        MockElement('span', text='Start '),
        MockElement('b', text='Bold'),
        MockElement('span', text=' End')
    ])
    assert extract_text(root3) == 'Start Bold End'

    # Test Case 4: Separator element (br) - should introduce sep_symbol
    root4 = MockElement('div', children=[
        MockElement('p', text='Part A'),
        MockElement('br'),
        MockElement('p', text='Part B')
    ])
    # br triggers True in extract_text_array, which maps to sep_symbol (\n)
    assert extract_text(root4) == 'Part A\nPart B'

    # Test Case 5: Handling of tails (text following a tag)
    root5 = MockElement('div', children=[
        MockElement('span', text='Inline', tail=' Tail')
    ])
    assert extract_text(root5) == 'Inline Tail'

    # Test Case 6: Complex structure with whitespace and mixed tags
    root6 = MockElement('div', children=[
        MockElement('h1', text=' Title '),
        MockElement('p', children=[
            MockElement('b', text='Bold'),
            MockElement('a', text='Link')
        ], tail=' End')
    ])
    # Whitespace inside tags should be squashed by squash_space=True
    assert extract_text(root6) == 'Title\nBoldLink End'

    # Test Case 7: Custom block and separator symbols
    root7 = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('br'),
        MockElement('p', text='B')
    ])
    assert extract_text(root7, block_symbol=' | ', sep_symbol=' - ') == 'A - B'

    # Test Case 8: Empty elements
    root8 = MockElement('div', children=[
        MockElement('p'),
        MockElement('span')
    ])
    assert extract_text(root8) == ''

    # Test Case 9: Deeply nested structure
    root9 = MockElement('div', children=[
        MockElement('section', children=[
            MockElement('div', children=[
                MockElement('p', text='Deep')
            ])
        ])
    ])
    assert extract_text(root9) == 'Deep'

    # Test Case 10: Testing squash_space=False behavior
    # When False, it should preserve the artificial newlines (None markers) as strings/raw
    root10 = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='B')
    ])
    # Without squashing, the None in extract_text_array is joined by block_symbol (\n)
    # But it won't strip the extra whitespace/newlines from the logic.
    # In the provided code, if squash_space=False, the function returns 
    # result = ''.join(block_symbol if x is None else ...)
    # Since p is not in INLINE_TAGS, it adds a None.
    assert extract_text(root10, squash_space=False) == 'A\nB'

    # Test Case 11: Text with heavy whitespace
    root11 = MockElement('div', text='  Too    much \n space  ')
    assert extract_text(root11) == 'Too much space'
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

def test_extract_text_array():
    # Test case 1: Empty element
    root_empty = MockElement('div')
    assert extract_text_array(root_empty) == []

    # Test case 2: Simple text node with no tags
    root_text = MockElement('p', text='Hello')
    # Note: extract_text_array adds None for block elements at start and end
    # But strip_artifical_nl removes leading/trailing Nones.
    # For a single block element, r starts with [None, 'Hello', None] -> ['Hello']
    assert extract_text_array(root_text) == ['Hello']

    # Test case 3: Inline tags (no artificial newlines inserted)
    root_inline = MockElement('span', text='Inside')
    assert extract_text_array(root_inline) == ['Inside']

    # Test case 4: Nested block elements (testing None insertion/removal)
    # <div><p>Text</p></div> -> [None, 'Text', None] stripped to ['Text']
    child_p = MockElement('p', text='Child')
    root_div = MockElement('div', children=[child_p])
    assert extract_text_array(root_div) == ['Child']

    # Test case 5: Separator tag (br)
    root_br = MockElement('br')
    # br is in SEPARATORS, so it appends True
    assert extract_text_array(root_br) == [True]

    # Test case 6: Complex structure with tails and children
    # <div><p>A</p>Tail<span>B</span></div>
    span_b = MockElement('span', text='B')
    p_a = MockElement('p', text='A', children=[span_b])
    p_a.tail = 'Tail'
    root_complex = MockElement('div', children=[p_a])
    
    # Expected flow:
    # p_a is block -> [None, 'A']
    # span_b is inline -> ['B']
    # tail of p_a -> 'Tail'
    # result for p_a subtree: ['A', 'B', 'Tail']
    # root_complex is block -> [None, 'A', 'B', 'Tail', None]
    # After stripping artificial NL: ['A', 'B', 'Tail']
    assert extract_text_array(root_complex) == ['A', 'B', 'Tail']

    # Test case 7: Testing squash_artifical_nl=False
    child_p2 = MockElement('p', text='Text')
    root_div2 = MockElement('div', children=[child_p2])
    # Should preserve the None markers
    result_unsquashed = extract_text_array(root_div2, squash_artifical_nl=False)
    assert None in result_unsquashed

    # Test case 8: Testing strip_artifical_nl=False
    child_p3 = MocklyElement('p', text='Text') # Using helper logic
    root_div3 = MockElement('div', children=[MockElement('p', text='A')])
    result_unstripped = extract_text_array(root_div3, strip_artifical_nl=False)
    assert result_unstripped[0] is None
    assert result_unstripped[-1] is None

    # Test case 9: Callable tag (should return empty string)
    class CallableTag:
        def __call__(self): pass
    root_callable = MockElement(CallableTag())
    assert extract_text_array(root_callable) == ''

    # Test case 10: Whitespace handling in text
    root_ws = MockElement('p', text='  Word  \n  ')
    # WHITESPACE_RE replaces multiple whitespaces with single space
    assert extract_text_array(root_ws) == [' Word ']

def MocklyElement(tag, text=None):
    return MockElement(tag, text=text)
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
    # Case 1: Simple text node
    root1 = MockElement('div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Case 2: Nested block elements (div inside div)
    # Should introduce None (newline) markers which become block_symbol
    root2 = MockElement('div', children=[
        MockElement('p', text='Paragraph 1'),
        Mockelse := MockElement('p', text='Paragraph 2')
    ])
    # Structure: [None, 'Paragraph 1', None, None, 'Paragraph 2', None] -> squash/strip -> ['Paragraph 1', 'Paragraph 2']
    # Joined by \n
    assert extract_text(root2) == 'Paragraph 1\nParagraph 2'

    # Case 3: Inline elements (span inside div)
    # span is in INLINE_TAGS, so no extra None/True markers added around it
    root3 = MockElement('div', children=[
        MockElement('span', text='Inline'),
        MockElement('b', text=' Bold')
    ])
    assert extract_text(root3) == 'Inline Bold'

    # Case 4: Separator element (br)
    # br is in SEPARATORS, so it adds True (sep_symbol)
    root4 = MockElement('div', children=[
        MockElement('p', text='Line 1'),
        MockElement('br'),
        MockElement('p', text='Line 2')
    ])
    # br adds True -> becomes sep_symbol (\n)
    assert extract_text(root4) == 'Line 1\n\nLine 2'.replace('\n\n', '\n') # depending on how squash handles None

    # Case 5: Whitespace squashing
    root5 = MockElement('div', text='  Too   many \n spaces  ')
    assert extract_text(root5) == 'Too many spaces'

    # Case 6: Complex structure with tails and mixed tags
    # <div><p>Text<span>Inner</span>Tail</p></div>
    child_span = MockElement('span', text='Inner')
    child_p = MockElement('p', text='Text', children=[child_span], tail='Tail')
    root6 = MockElement('div', children=[child_p])
    
    # extract_text_array logic:
    # p is block -> adds None
    # p.text is 'Text'
    # child_span (inline) -> no extra markers
    # child_span.text is 'Inner'
    # child_span.tail is None
    # child_p.tail is 'Tail'
    # p ends -> adds None
    # div ends -> adds None
    assert extract_text(root6) == 'TextInnerTail'

    # Case 7: Custom symbols
    root7 = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='B')
    ])
    assert extract_text(root7, block_symbol=' | ', sep_symbol=' >>> ') == 'A | B'

    # Case 8: Empty element
    root8 = MockElement('div', children=[])
    assert extract_text(root8) == ''

    # Case 9: Element with only tail (no text)
    root9 = MockElement('div', children=[
        MockElement('span', text='', tail=' Only Tail')
    ])
    assert extract_text(root9) == 'Only Tail'
```


# LLM-generated content at query #13
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
    # Test 1: Simple text node (no tags)
    root1 = MockElement('div', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Test 2: Inline tag inside block tag
    # <div><span>Text</span></div> -> ['\n', 'Text', '\n'] stripped -> ['Text']
    root2 = MockElement('div', children=[
        MockElement('span', text='Inner')
    ])
    assert extract_text_array(root2) == ['Inner']

    # Test 3: Block tag inside block tag (inserts None/newlines)
    # <div><p>A</p><p>B</p></div> -> ['\n', 'A', '\n', 'B', '\n'] stripped -> ['A', 'B']
    root3 = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='B')
    ])
    # Note: extract_text_array logic with squash_artifical_nl=True (default) 
    # will merge the Nones.
    assert extract_text_array(root3) == ['A', 'B']

    # Test 4: Separator tag (br)
    # <div><br></div> -> [True]
    root4 = MockElement('div', children=[
        MockElement('br')
    ])
    assert extract_text_array(root4) == [True]

    # Test 5: Text with tails
    # <div>Text<span>Middle</span>Tail</div>
    root5 = MockElement('div', text='Start', children=[
        MockElement('span', text='Middle', tail='End')
    ])
    # extract_text_array process: ['\n', 'Start', '\n', 'Middle', 'End', '\n']
    # with squash/strip logic applied in the function:
    assert extract_text_array(root5) == ['Start', 'Middle', 'End']

    # Test 6: Empty element
    root6 = MockElement('div')
    assert extract_text_array(root6) == []

    # Test 7: Callable tag (should return empty string per code logic)
    class CallableTag:
        def __call__(self): pass
    root7 = MockElement(CallableTag())
    assert extract_text_array(root7) == ''

    # Test 8: Testing squash_artifical_nl=False (preserves the None/True markers)
    root8 = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='B')
    ])
    # Without squashing, we expect the raw structure including None for block tags
    result = extract_text_array(root8, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in result
    assert 'A' in result
    assert 'B' in result

    # Test 9: Testing whitespace squashing via WHITESPACE_RE
    root9 = MockElement('div', text='Hello\n\t\rWorld')
    assert extract_text_array(root9) == ['Hello World']
```


# LLM-generated content at query #14
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
    assert extract_template_text_helper(root1) == 'Hello'

    # Test Case 2: Nested block elements with whitespace
    # <div><p>Line 1</p><p>Line 2</p></div> -> "Line 1\nLine 2"
    child1 = MockElement('p', text='Line 1')
    child2 = MockElement('p', text='Line 2')
    root2 = MockElement('div', children=[child1, child2])
    assert extract_template_text_helper(root2) == 'Line 1\nLine 2'

    # Test Case 3: Inline elements (no newlines added)
    # <div><span>Part 1</span><span>Part 2</span></div> -> "Part 1Part 2"
    child3 = MockElement('span', text='Part 1')
    child4 = MockElement('span', text='Part 2')
    root3 = MockElement('div', children=[child3, child4])
    assert extract_template_text_helper(root3) == 'Part 1Part 2'

    # Test Case 4: Separator tag <br> (should trigger True/sep_symbol)
    # <div>A<br>B</div> -> "A\nB"
    child5 = MockElement('br')
    root4 = MockElement('div', text='A', children=[child5])
    child5.tail = 'B'
    assert extract_template_text_helper(root4) == 'A\nB'

    # Test Case 5: Complex structure with tails and mixed tags
    # <div>Text 1<span>Inner</span>Tail<p>Block</p></div>
    child6 = MockElement('span', text='Inner')
    child7 = MockElement('p', text='Block')
    root5 = MockElement('div', text='Text 1', children=[child6, child7])
    child6.tail = 'Tail'
    assert extract_template_text_helper(root5) == 'Text 1\nTail\nBlock'

    # Test Case 6: Checking whitespace squashing (squash_space=True)
    # <div>  Too   many    spaces  </div> -> "Too many spaces"
    root6 = MockElement('div', text='  Too   many    spaces  ')
    assert extract_template_text_helper(root6) == 'Too many spaces'

    # Test Case 7: Custom symbols
    # Using '|' as block and '-' as separator
    root7 = MockElement('div', children=[MockElement('p', text='A'), MockElement('br'), MockElement('p', text='B')])
    assert extract_text(root7, block_symbol='|', sep_symbol='-') == 'A-B'

    # Test Case 8: Empty element
    root8 = MockElement('div')
    assert extract_template_text_helper(root8) == ''

def extract_template_text_helper(dom):
    """Helper to call extract_text with default params for cleaner tests."""
    return extract_text(dom)
```


# LLM-generated content at query #15
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
    # Test Case 1: Simple text node (no tags)
    root1 = MockElement('div', text='Hello')
    assert extract_text_array(root1) == ['Hello']

    # Test Case 2: Inline tags (should not trigger None/True markers)
    root2 = MockElement('span', text='Inline', tail=' content')
    assert extract_text_array(root2) == ['Inline', ' content']

    # Test Case 3: Block tags (should trigger None markers for newlines)
    # structure: <div><p>Text</p></div>
    child_p = MockElement('p', text='Paragraph')
    root3 = MockElement('div', children=[child_p])
    # Expected: [None (start div), None (start p), 'Paragraph', None (end p), None (end div)]
    # After _squash_artifical_nl and _strip_artifical_nl, leading/trailing Nones are gone.
    assert extract_text_array(root3) == ['Paragraph']

    # Test Case 4: Separator tags (br)
    root4 = MockElement('div', children=[MockElement('br')])
    assert extract_text_array(root4) == [True]

    # Test Case 5: Complex nesting with mixed inline and block elements
    # structure: <div><a>Link</a><p>Text</p></div>
    link = MockElement('a', text='Click')
    para = MockElement('p', text='Body')
    root5 = MockElement('div', children=[link, para])
    # Process:
    # [None(div), 'Click'(a-text), None(p-start), 'Body'(p-text), None(p-end), None(div-end)]
    # Squashing Nones and stripping edges should result in ['Click', 'Body']
    assert extract_text_array(root5) == ['Click', 'Body']

    # Test Case 6: Handling tails
    child_tail = MockElement('b', text='Bold', tail=' follows')
    root6 = MockElement('div', children=[child_tail])
    assert extract_text_array(root6) == ['Bold', ' follows']

    # Test Case 7: Empty element
    root7 = MockElement('div', children=[])
    # Starts with None, ends with None -> stripped to empty list
    assert extract_text_array(root7) == []

    # Test Case 8: Functionality with squash_artifical_nl=False
    # Should preserve the raw None markers
    child_p_raw = MockElement('p', text='Raw')
    root8 = MockElement('div', children=[child_p_raw])
    result = extract_text_array(root8, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in result
    assert 'Raw' in result

    # Test Case 9: Callable tag (should return empty string per implementation)
    class CallableTag:
        def __call__(self): pass
    root9 = MockElement(CallableTag())
    assert extract_text_array(root9) == ''

    # Test Case 10: Whitespace squashing in text
    root10 = MockElement('div', text='Line\n\nBreak')
    assert extract_text_array(root10) == ['Line Break']
```


# LLM-generated content at query #16
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
    # Test case 1: Simple text node
    root1 = MockNode(tag='div', text='Hello')
    assert extract_text(root1) == 'Hello'

    # Test case 2: Nested block elements with newlines
    # <div><p>Line 1</p><p>Line 2</p></div> -> "Line 1\nLine 2"
    child1 = MockNode(tag='p', text='Line 1')
    child2 = MockNode(tag='p', text='Line 2')
    root2 = MockNode(tag='div', children=[child1, child2])
    assert extract_text(root2) == 'Line 1\nLine 2'

    # Test case 3: Inline elements (should not trigger newlines)
    # <div><span>Part 1</span><span>Part 2</span></div> -> "Part 1Part 2"
    child3 = MockNode(tag='span', text='Part 1')
    child4 = MockNode(tag='span', text='Part 2')
    root3 = MockNode(tag='div', children=[child3, child4])
    assert extract_text(root3) == 'Part 1Part 2'

    # Test case 4: Separator element <br>
    # <div>Line 1<br>Line 2</div> -> "Line 1\nLine 2" (where \n is sep_symbol)
    child5 = MockNode(tag='br')
    child6 = MockNode(tag='span', text='Line 2')
    root4 = MockNode(tag='div', children=[child5, child6])
    assert extract_text(root4) == 'Line 1\nLine 2'

    # Test case 5: Complex structure with tail and whitespace
    # <div>  <p>  Text  </p>  Tail  </div>
    child7 = MockNode(tag='p', text='  Text  ', tail='  Tail  ')
    root5 = MockNode(tag='div', children=[child7])
    # squash_space=True should clean up whitespace inside parts and around blocks
    assert extract_text(root5) == 'Text\nTail'

    # Test case 6: Custom symbols
    # Using different block and separator symbols
    root6 = MockNode(tag='div', children=[
        MockNode(tag='p', text='A'),
        MockNode(tag='br'),
        MockNode(tag='span', text='B')
    ])
    assert extract_text(root6, block_symbol='|', sep_symbol='-') == 'A-B'

    # Test case 7: Empty/None values
    root7 = MockNode(tag='div', text=None)
    assert extract_text(root7) == ''

    # Test case 8: Elements with no text and no children
    root8 = MockNode(tag='div', children=[MockNode(tag='span')])
    assert extract_text(root8) == ''

    # Test case 9: Testing squash_space=False
    # Should preserve the raw whitespace/newlines from the nodes
    child9 = MockNode(tag='p', text='  Space  ')
    root9 = MockNode(tag='div', children=[child9])
    assert extract_text(root9, squash_space=False) == '  Space  '

    # Test case 10: Functionality of _merge_original_parts via extract_text
    # Multiple text nodes in a row should be merged/squashed
    child10_a = MockNode(tag='span', text='Part ')
    child10_b = MockNode(tag='span', text='Another')
    root10 = MockNode(tag='div', children=[child10_a, child10_b])
    assert extract_text(root10) == 'Part Another'
```


