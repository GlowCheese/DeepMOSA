####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test inline tags
    dom = fromstring('<div><span>Hello</span> <strong>World</strong></div>')
    assert extract_text_array(dom) == ['Hello', ' ', 'World']

    # Test block tags with artificial newlines
    dom = fromstring('<div><p>Paragraph 1</p><p>Paragraph 2</p></div>')
    assert extract_text_array(dom) == [None, 'Paragraph 1', None, None, 'Paragraph 2', None]

    # Test separator tags (br)
    dom = fromstring('<div>Line 1<br>Line 2</div>')
    assert extract_text_array(dom) == ['Line 1', True, 'Line 2', None]

    # Test nested tags
    dom = fromstring('<div><p>Outer <span>Inner</span> text</p></div>')
    assert extract_text_array(dom) == [None, 'Outer ', 'Inner', ' text', None]

    # Test text with tail
    dom = fromstring('<div><p>First</p>Tail text</div>')
    assert extract_text_array(dom) == [None, 'First', None, 'Tail text', None]

    # Test squash_artifical_nl=False
    dom = fromstring('<div><p>P1</p><p>P2</p><p>P3</p></div>')
    assert extract_text_array(dom, squash_artifical_nl=False) == [
        None, 'P1', None, None, 'P2', None, None, 'P3', None
    ]

    # Test strip_artifical_nl=False
    dom = fromstring('<div><p>Content</p></div>')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Content', None]

    # Test empty element
    dom = fromstring('<div></div>')
    assert extract_text_array(dom) == [None, None]

    # Test self-closing tag
    dom = fromstring('<div><img src="test.jpg"/></div>')
    assert extract_text_array(dom) == [None, None]

    # Test mixed content
    dom = fromstring('''<div>
        <h1>Title</h1>
        <p>First paragraph<br>with break</p>
        <p>Second paragraph <em>with emphasis</em></p>
    </div>''')
    assert extract_text_array(dom) == [
        None, 'Title', None,
        None, 'First paragraph', True, 'with break', None,
        None, 'Second paragraph ', 'with emphasis', None
    ]


# LLM-generated content at query #2
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test inline tag with children
    child = MockDom('b', text='World')
    dom = MockDom('span', children=[child])
    assert extract_text_array(dom) == ['World']

    # Test block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test separator tag
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test nested tags
    child1 = MockDom('b', text='bold')
    child2 = MockDom('i', text='italic')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text_array(dom) == [None, 'bold', 'italic', None]

    # Test with tail text
    child = MockDom('b', text='bold', tail=' tail')
    dom = MockDom('div', children=[child])
    assert extract_text_array(dom) == [None, 'bold', ' tail', None]

    # Test squash_artifical_nl=False
    dom = MockDom('div', children=[MockDom('div'), MockDom('div')])
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, None, None, None]

    # Test strip_artifical_nl=False
    dom = MockDom('div', children=[MockDom('div'), MockDom('div')])
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, None, None, None]

    # Test callable tag
    dom = MockDom(lambda: 'callable')
    assert extract_text_array(dom) == ''

    # Test complex nested structure
    grandchild = MockDom('i', text='italic')
    child = MockDom('b', text='bold ', children=[grandchild], tail=' tail')
    dom = MockDom('div', text=' start ', children=[child], tail=' end ')
    assert extract_text_array(dom) == [None, ' start ', 'bold ', 'italic', ' tail', ' end ', None]


# LLM-generated content at query #3
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockInlineTag:
        tag = 'span'
        text = 'Hello'
        tail = 'World'
        def getchildren(self):
            return []

    inline_dom = MockInlineTag()
    assert extract_text_array(inline_dom) == ['Hello', 'World']

    # Test with separator tag
    class MockSeparatorTag:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []

    separator_dom = MockSeparatorTag()
    assert extract_text_array(separator_dom) == [True]

    # Test with block tag
    class MockBlockTag:
        tag = 'div'
        text = 'Start'
        tail = 'End'
        def getchildren(self):
            return []

    block_dom = MockBlockTag()
    assert extract_text_array(block_dom) == ['Start', None, 'End']

    # Test with nested tags
    class MockParentTag:
        tag = 'div'
        text = 'Parent'
        tail = None
        def getchildren(self):
            return [MockChildTag()]

    class MockChildTag:
        tag = 'span'
        text = 'Child'
        tail = 'Tail'
        def getchildren(self):
            return []

    parent_dom = MockParentTag()
    assert extract_text_array(parent_dom) == ['Parent', 'Child', 'Tail', None]

    # Test with squash_artifical_nl=False
    class MockTagWithNewlines:
        tag = 'div'
        text = 'Text'
        tail = None
        def getchildren(self):
            return []

    tag_dom = MockTagWithNewlines()
    assert extract_text_array(tag_dom, squash_artifical_nl=False) == ['Text', None, None]

    # Test with strip_artifical_nl=False
    assert extract_text_array(tag_dom, strip_artifical_nl=False) == ['Text']

    # Test with callable tag
    class MockCallableTag:
        tag = lambda: 'div'
        text = 'Text'
        tail = None
        def getchildren(self):
            return []

    callable_dom = MockCallableTag()
    assert extract_text_array(callable_dom) == ''


# LLM-generated content at query #4
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    inline_dom = MockDom('span', text='Hello')
    assert extract_text_array(inline_dom) == ['Hello']

    # Test with separator tag
    separator_dom = MockDom('br')
    assert extract_text_array(separator_dom) == [True]

    # Test with block tag
    block_dom = MockDom('div', text='Hello')
    assert extract_text_array(block_dom) == [None, 'Hello', None]

    # Test with nested tags
    nested_dom = MockDom('div', children=[
        MockDom('p', text='Paragraph'),
        MockDom('span', text='Inline')
    ])
    assert extract_text_array(nested_dom) == [None, 'Paragraph', 'Inline', None]

    # Test with tail text
    tail_dom = MockDom('div', children=[
        MockDom('span', text='Hello', tail='World')
    ])
    assert extract_text_array(tail_dom) == [None, 'Hello', 'World', None]

    # Test with squash_artifical_nl=False
    multi_nl_dom = MockDom('div', children=[
        MockDom('div', text='First'),
        MockDom('div', text='Second')
    ])
    assert extract_text_array(multi_nl_dom, squash_artifical_nl=False) == [
        None, 'First', None, None, 'Second', None, None
    ]

    # Test with strip_artifical_nl=False
    strip_test_dom = MockDom('div', children=[
        MockDom('div', text='Content')
    ])
    assert extract_text_array(strip_test_dom, strip_artifical_nl=False) == [
        None, 'Content', None
    ]

    # Test with callable tag
    callable_dom = MockDom(lambda: 'callable')
    assert extract_text_array(callable_dom) == ''

    # Test with empty dom
    empty_dom = MockDom('div')
    assert extract_text_array(empty_dom) == [None, None]


# LLM-generated content at query #5
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline element with text
    inline_elem = MockElement('span', text='Hello')
    assert extract_text_array(inline_elem) == ['Hello']

    # Test inline element with children
    child_elem = MockElement('strong', text='World')
    inline_elem_with_child = MockElement('span', children=[child_elem])
    assert extract_text_array(inline_elem_with_child) == ['World']

    # Test block element with text
    block_elem = MockElement('div', text='Block text')
    assert extract_text_array(block_elem) == [None, 'Block text', None]

    # Test separator element
    separator_elem = MockElement('br')
    assert extract_text_array(separator_elem) == [True]

    # Test nested elements
    nested_child = MockElement('b', text='nested')
    nested_parent = MockElement('p', children=[nested_child], tail=' tail')
    assert extract_text_array(nested_parent) == [None, 'nested', ' tail', None]

    # Test with squash_artifical_nl=False
    block_with_spaces = MockElement('div', text='  text  ')
    assert extract_text_array(block_with_spaces, squash_artifical_nl=False) == [None, '  text  ', None]

    # Test with strip_artifical_nl=False
    block_with_spaces = MockElement('div', text='text')
    assert extract_text_array(block_with_spaces, strip_artifical_nl=False) == [None, 'text', None]

    # Test complex structure
    complex_child1 = MockElement('span', text='First')
    complex_child2 = MockElement('div', text='Second')
    complex_parent = MockElement('body', children=[complex_child1, complex_child2])
    result = extract_text_array(complex_parent)
    assert result == [None, 'First', None, 'Second', None]

    # Test with callable tag
    callable_tag_elem = MockElement(lambda: 'div', text='Callable')
    assert extract_text_array(callable_tag_elem) == ['']


# LLM-generated content at query #6
#--------------------------

```python
def test_extract_text_array():
    # Test for inline tag
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test inline tag with children
    child = MockDom('strong', text='World')
    dom = MockDom('span', children=[child])
    assert extract_text_array(dom) == ['World']

    # Test block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test separator tag
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test nested tags
    child = MockDom('span', text='nested')
    dom = MockDom('div', children=[child])
    assert extract_text_array(dom) == [None, 'nested', None]

    # Test with tail text
    child = MockDom('span', text='Hello', tail='World')
    dom = MockDom('div', children=[child])
    assert extract_text_array(dom) == [None, 'Hello', 'World', None]

    # Test squash_artifical_nl=False
    dom = MockDom('div', children=[MockDom('div')])
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, None, None]

    # Test strip_artifical_nl=False
    dom = MockDom('div', text='Hello', children=[MockDom('div')])
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None, None]

    # Test callable tag
    dom = MockDom(lambda: 'callable')
    assert extract_text_array(dom) == ''

    # Test empty dom
    dom = MockDom('div')
    assert extract_text_array(dom) == [None, None]


# LLM-generated content at query #7
#--------------------------

```python
def test_extract_text():
    # Test basic inline tag
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test 1: Simple inline tag
    dom = MockElement('span', text='Hello World')
    assert extract_text(dom) == 'Hello World'

    # Test 2: Block tag with squash_space=True
    dom = MockElement('div', text='Hello', children=[
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test 3: Block tag with squash_space=False
    dom = MockElement('div', text='Hello', children=[
        MockElement('span', text='World')
    ])
    assert extract_text(dom, squash_space=False) == '\nHello\nWorld\n'

    # Test 4: Separator tag (br)
    dom = MockElement('div', children=[
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == '\nWorld'

    # Test 5: Multiple block tags
    dom = MockElement('div', children=[
        MockElement('p', text='First paragraph'),
        MockElement('p', text='Second paragraph')
    ])
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'

    # Test 6: Nested tags
    dom = MockElement('div', children=[
        MockElement('p', text='Outer', children=[
            MockElement('span', text='Inner')
        ])
    ])
    assert extract_text(dom) == 'Outer Inner'

    # Test 7: Tail text
    dom = MockElement('div', children=[
        MockElement('span', text='Hello', tail='World')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test 8: Custom block and separator symbols
    dom = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('p', text='Second')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'First|Second'

    # Test 9: Empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''

    # Test 10: Whitespace handling
    dom = MockElement('div', text='  Hello  ', children=[
        MockElement('span', text='  World  ')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test 11: Callable tag (should return empty string)
    dom = MockElement(lambda: 'div')
    assert extract_text(dom) == ''


# LLM-generated content at query #8
#--------------------------

```python
def test_extract_text():
    # Test simple inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    # Test 1: Simple inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 2: Block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 3: Nested tags
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test 4: Separator tag (br)
    dom = MockDom('br')
    assert extract_text(dom, sep_symbol='\n') == '\n'

    # Test 5: Multiple children with text and tails
    child1 = MockDom('span', text='Hello', tail=' ')
    child2 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom) == 'Hello World'

    # Test 6: Block symbol replacement
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child], tail=' ')
    assert extract_text(dom, block_symbol='\n') == 'Hello World\n'

    # Test 7: Squash space
    child = MockDom('span', text='  Hello  ')
    dom = MockDom('div', children=[child])
    assert extract_text(dom, squash_space=True) == 'Hello'

    # Test 8: Multiple separators
    child1 = MockDom('br')
    child2 = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child1, child2])
    assert extract_text(dom, sep_symbol='\n') == 'Hello\nWorld'

    # Test 9: Empty text
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test 10: Complex nested structure
    grandchild = MockDom('strong', text='nested')
    child = MockDom('span', text='deeply ', children=[grandchild], tail=' text')
    dom = MockDom('div', text='Some', children=[child], tail=' end')
    assert extract_text(dom) == 'Some deeply nested text end'


# LLM-generated content at query #9
#--------------------------

```python
def test_extract_text():
    from lxml import html

    # Test with simple inline elements
    dom = html.fromstring('<div><span>Hello</span> <span>World</span></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block elements
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with separators
    dom = html.fromstring('<div>Hello<br/>World</div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with nested elements
    dom = html.fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with whitespace squashing
    dom = html.fromstring('<div>Hello   World</div>')
    assert extract_text(dom) == 'Hello World'

    # Test with custom block and separator symbols
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == 'Hello|World'

    # Test with no squashing
    dom = html.fromstring('<div>Hello   World</div>')
    assert extract_text(dom, squash_space=False) == 'Hello   World'

    # Test with empty elements
    dom = html.fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with mixed content
    dom = html.fromstring('<div><p>Hello</p> <span>World</span> <p>!</p></div>')
    assert extract_text(dom) == 'Hello\nWorld\n!'

    # Test with script and style tags (should be ignored)
    dom = html.fromstring('<div><script>alert("Hello")</script>World</div>')
    assert extract_text(dom) == 'World'

    # Test with preformatted text
    dom = html.fromstring('<div><pre>Hello   World</pre></div>')
    assert extract_text(dom, squash_space=False) == 'Hello   World'

    # Test with multiple br tags
    dom = html.fromstring('<div>Hello<br/><br/>World</div>')
    assert extract_text(dom) == 'Hello\n\nWorld'

    # Test with leading and trailing whitespace
    dom = html.fromstring('<div>  Hello  </div>')
    assert extract_text(dom) == 'Hello'

    # Test with complex nested structure
    dom = html.fromstring('''
        <div>
            <h1>Title</h1>
            <p>Paragraph <strong>bold</strong> text</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    ''')
    assert extract_text(dom) == 'Title\nParagraph bold text\nItem 1\nItem 2'


# LLM-generated content at query #10
#--------------------------

```python
def test_extract_text_array():
    from lxml import html

    # Test with simple inline tag
    dom = html.fromstring('<div><span>Hello</span> <span>World</span></div>')
    result = extract_text_array(dom)
    assert result == ['Hello', ' ', 'World']

    # Test with block tag
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello', None, None, 'World', None]

    # Test with separator tag (br)
    dom = html.fromstring('<div>Hello<br>World</div>')
    result = extract_text_array(dom)
    assert result == ['Hello', True, 'World']

    # Test with nested tags
    dom = html.fromstring('<div><p>Hello <span>World</span></p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello ', 'World', None]

    # Test with text and tail
    dom = html.fromstring('<div><p>Hello</p> World</div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello', None, ' World', None]

    # Test with squash_artifical_nl=False
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = html.fromstring('<div><p>Hello</p></div>')
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'Hello', None]

    # Test with empty dom
    dom = html.fromstring('<div></div>')
    result = extract_text_array(dom)
    assert result == [None, None]

    # Test with text only
    dom = html.fromstring('<div>Hello World</div>')
    result = extract_text_array(dom)
    assert result == ['Hello World']

    # Test with multiple separators
    dom = html.fromstring('<div>Hello<br><br>World</div>')
    result = extract_text_array(dom)
    assert result == ['Hello', True, True, 'World']


# LLM-generated content at query #11
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline elements
    dom = fromstring("<div><span>Hello</span> <strong>World</strong></div>")
    assert extract_text(dom) == "Hello World"

    # Test with block elements and whitespace squashing
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with separators (br tags)
    dom = fromstring("<div>Hello<br/>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with nested elements
    dom = fromstring("<div><p>Hello <span>World</span></p></div>")
    assert extract_text(dom) == "Hello World"

    # Test with multiple whitespace
    dom = fromstring("<div>Hello   \n   World</div>")
    assert extract_text(dom) == "Hello World"

    # Test with custom block and separator symbols
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == "Hello|World"

    # Test with squash_space=False
    dom = fromstring("<div>  Hello  \n  World  </div>")
    assert extract_text(dom, squash_space=False) == "  Hello  \n  World  "

    # Test with empty elements
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with mixed content
    dom = fromstring("<div><p>Hello</p><span>World</span><p>!</p></div>")
    assert extract_text(dom) == "Hello\nWorld\n!"

    # Test with special characters and entities
    dom = fromstring("<div>Hello &amp; World</div>")
    assert extract_text(dom) == "Hello & World"

    # Test with script and style tags (should be ignored)
    dom = fromstring("<div>Hello<script>alert('xss')</script>World</div>")
    assert extract_text(dom) == "HelloWorld"


# LLM-generated content at query #12
#--------------------------

```python
def test_extract_text_array():
    # Test case 1: Simple inline tag
    from lxml.html import fromstring
    dom = fromstring('<div><span>Hello</span> <span>World</span></div>')
    result = extract_text_array(dom)
    assert result == ['Hello', ' ', 'World']

    # Test case 2: Block tag with artificial newlines
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello', None, None, 'World', None]

    # Test case 3: Separator tag (br)
    dom = fromstring('<div>Hello<br/>World</div>')
    result = extract_text_array(dom)
    assert result == ['Hello', True, 'World']

    # Test case 4: Nested tags
    dom = fromstring('<div><ul><li>Item 1</li><li>Item 2</li></ul></div>')
    result = extract_text_array(dom)
    assert result == [None, None, 'Item 1', None, None, 'Item 2', None, None, None]

    # Test case 5: Text with tail
    dom = fromstring('<div><span>Hello</span>World</div>')
    result = extract_text_array(dom)
    assert result == ['Hello', 'World']

    # Test case 6: Empty tag
    dom = fromstring('<div></div>')
    result = extract_text_array(dom)
    assert result == [None, None]

    # Test case 7: Mixed inline and block tags
    dom = fromstring('<div>Hello <strong>World</strong>!</div>')
    result = extract_text_array(dom)
    assert result == ['Hello ', 'World', '!']

    # Test case 8: squash_artifical_nl=False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'Hello', None, None, 'World', None]

    # Test case 9: strip_artifical_nl=False
    dom = fromstring('<div><p>Hello</p></div>')
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'Hello', None]

    # Test case 10: Both flags False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'Hello', None, None, 'World', None]


# LLM-generated content at query #13
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tags
    dom = fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"

    # Test with block tags
    dom = fromstring("<div>Hello<div>World</div></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with separator tags
    dom = fromstring("<p>Hello<br>World</p>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with squash_space=True
    dom = fromstring("<p>  Hello   <b>  World  </b>  </p>")
    assert extract_text(dom, squash_space=True) == "Hello World"

    # Test with squash_space=False
    dom = fromstring("<p>  Hello   <b>  World  </b>  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "

    # Test with custom block_symbol and sep_symbol
    dom = fromstring("<div>Hello<div>World</div></div>")
    assert extract_text(dom, block_symbol="|", sep_symbol="|") == "Hello|World"

    # Test with mixed content
    dom = fromstring("<div>Hello<p>World<br>!</p>Goodbye</div>")
    assert extract_text(dom) == "Hello\nWorld\n!\nGoodbye"

    # Test with empty dom
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with only whitespace
    dom = fromstring("<p>   \n  \t  </p>")
    assert extract_text(dom) == ""

    # Test with nested inline tags
    dom = fromstring("<p><b><i>Hello</i></b> World</p>")
    assert extract_text(dom) == "Hello World"

    # Test with script and style tags (should be ignored)
    dom = fromstring("<div>Hello<script>alert('xss')</script>World</div>")
    assert extract_text(dom) == "HelloWorld"

    # Test with preformatted content
    dom = fromstring("<pre>  Hello   World  </pre>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "


# LLM-generated content at query #14
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test block tag
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test nested tags
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test with separators
    child = MockDom('br')
    dom = MockDom('div', text='Hello', children=[child], tail='World')
    assert extract_text(dom, sep_symbol='\n') == 'Hello\nWorld'

    # Test squash_space
    dom = MockDom('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test block_symbol
    child = MockDom('p', text='Paragraph')
    dom = MockDom('div', children=[child])
    assert extract_text(dom, block_symbol='\n\n') == 'Paragraph'

    # Test with multiple children
    child1 = MockDom('span', text='First')
    child2 = MockDom('span', text='Second')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom) == 'FirstSecond'

    # Test with tail text
    child = MockDom('span', text='Child')
    dom = MockDom('div', children=[child], tail='Tail')
    assert extract_text(dom) == 'ChildTail'

    # Test with None text
    dom = MockDom('div', text=None)
    assert extract_text(dom) == ''

    # Test with empty children
    dom = MockDom('div', children=[])
    assert extract_text(dom) == ''

    # Test with callable tag
    dom = MockDom(tag=lambda: 'div', text='Hello')
    assert extract_text(dom) == ''


# LLM-generated content at query #15
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test 1: Simple inline element
    dom = MockElement('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 2: Block element
    dom = MockElement('div', text='Hello')
    assert extract_text(dom) == '\nHello\n'

    # Test 3: Nested elements
    child = MockElement('span', text='World')
    dom = MockElement('div', children=[child])
    assert extract_text(dom) == '\nWorld\n'

    # Test 4: Multiple children
    child1 = MockElement('span', text='Hello')
    child2 = MockElement('span', text='World')
    dom = MockElement('div', children=[child1, child2])
    assert extract_text(dom) == '\nHelloWorld\n'

    # Test 5: With tail text
    child = MockElement('span', text='Hello', tail='World')
    dom = MockElement('div', children=[child])
    assert extract_text(dom) == '\nHelloWorld\n'

    # Test 6: BR separator
    dom = MockElement('br')
    assert extract_text(dom) == '\n'

    # Test 7: Squash space
    dom = MockElement('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=True) == '\nHello World\n'

    # Test 8: Custom symbols
    dom = MockElement('div', text='Hello')
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == '|Hello|'

    # Test 9: Complex nested structure
    grandchild = MockElement('strong', text='nested')
    child = MockElement('span', text='Hello ', children=[grandchild], tail=' text')
    dom = MockElement('div', text='Start ', children=[child], tail=' End')
    assert extract_text(dom) == '\nStart Hello nested text End\n'

    # Test 10: Empty element
    dom = MockElement('div')
    assert extract_text(dom) == '\n\n'

    # Test 11: Callable tag
    dom = MockElement(tag=lambda: 'div')
    assert extract_text(dom) == ''

    # Test 12: Whitespace handling
    dom = MockElement('div', text='  \n  \t  \r  \f  \u200B  ')
    assert extract_text(dom, squash_space=True) == '\n \n'

    # Test 13: Multiple separators
    child1 = MockElement('br')
    child2 = MockElement('br')
    dom = MockElement('div', children=[child1, child2])
    assert extract_text(dom) == '\n\n\n'

    # Test 14: Mixed content
    child1 = MockElement('span', text='Hello')
    child2 = MockElement('br')
    child3 = MockElement('span', text='World')
    dom = MockElement('div', children=[child1, child2, child3])
    assert extract_text(dom) == '\nHello\nWorld\n'


# LLM-generated content at query #16
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test 1: Simple inline element
    dom = MockElement('span', text='Hello World')
    assert extract_text(dom) == 'Hello World'

    # Test 2: Block element with squash_space=True
    dom = MockElement('div', text='Hello', children=[
        MockElement('span', text='World')
    ], tail='Tail')
    assert extract_text(dom, squash_space=True) == 'Hello World Tail'

    # Test 3: Block element with squash_space=False
    dom = MockElement('div', text='Hello', children=[
        MockElement('span', text='World')
    ], tail='Tail')
    assert extract_text(dom, squash_space=False) == '\nHello World Tail\n'

    # Test 4: Separator element (br)
    dom = MockElement('div', children=[
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == '\nWorld'

    # Test 5: Nested block elements
    dom = MockElement('div', children=[
        MockElement('p', text='First paragraph'),
        MockElement('p', text='Second paragraph')
    ])
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'

    # Test 6: Mixed inline and block elements
    dom = MockElement('div', children=[
        MockElement('span', text='Inline '),
        MockElement('div', text='Block'),
        MockElement('span', text=' Inline')
    ])
    assert extract_text(dom) == 'Inline \nBlock\n Inline'

    # Test 7: Custom block and separator symbols
    dom = MockElement('div', children=[
        MockElement('br'),
        MockElement('span', text='Text')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == ';Text'

    # Test 8: Empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''

    # Test 9: Element with only whitespace
    dom = MockElement('div', text='   \n   ')
    assert extract_text(dom) == ''

    # Test 10: Complex nested structure
    dom = MockElement('div', children=[
        MockElement('h1', text='Title'),
        MockElement('p', children=[
            MockElement('span', text='First '),
            MockElement('strong', text='bold'),
            MockElement('span', text=' text')
        ]),
        MockElement('ul', children=[
            MockElement('li', text='Item 1'),
            MockElement('li', text='Item 2')
        ])
    ])
    assert extract_text(dom) == 'Title\nFirst bold text\nItem 1\nItem 2'


# LLM-generated content at query #17
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline element with text
    inline_elem = MockElement('span', text='Hello')
    assert extract_text_array(inline_elem) == ['Hello']

    # Test block element with text
    block_elem = MockElement('div', text='Hello')
    assert extract_text_array(block_elem) == [None, 'Hello', None]

    # Test separator element
    separator_elem = MockElement('br')
    assert extract_text_array(separator_elem) == [True]

    # Test nested elements
    parent = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('p', text='Second')
    ])
    assert extract_text_array(parent) == [None, 'First', None, 'Second', None]

    # Test with tail text
    parent_with_tail = MockElement('div', children=[
        MockElement('span', text='Hello', tail='World')
    ])
    assert extract_text_array(parent_with_tail) == [None, 'Hello', 'World', None]

    # Test squash_artifical_nl=False
    block_elem_no_squash = MockElement('div', text='Hello')
    assert extract_text_array(block_elem_no_squash, squash_artifical_nl=False) == [None, 'Hello', None]

    # Test strip_artifical_nl=False
    block_elem_no_strip = MockElement('div', text='Hello')
    assert extract_text_array(block_elem_no_strip, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test callable tag
    callable_elem = MockElement(lambda: 'div', text='Hello')
    assert extract_text_array(callable_elem) == ['']

    # Test complex nested structure
    complex_elem = MockElement('div', children=[
        MockElement('p', text='First', children=[
            MockElement('span', text='nested')
        ]),
        MockElement('br'),
        MockElement('p', text='Second')
    ])
    assert extract_text_array(complex_elem) == [None, 'First', 'nested', None, True, 'Second', None]


# LLM-generated content at query #18
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test basic inline tags
    dom = fromstring("<div><p>Hello <strong>World</strong></p></div>")
    assert extract_text(dom) == "Hello World"

    # Test block tags with squash_space
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, squash_space=True) == "Hello\nWorld"

    # Test block tags without squash_space
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, squash_space=False) == "\nHello\n\nWorld\n"

    # Test separator tags (br)
    dom = fromstring("<div>Line1<br/>Line2</div>")
    assert extract_text(dom) == "Line1\nLine2"

    # Test nested tags
    dom = fromstring("<div><ul><li>Item1</li><li>Item2</li></ul></div>")
    assert extract_text(dom) == "Item1\nItem2"

    # Test whitespace handling
    dom = fromstring("<div>  Hello   \n  World  </div>")
    assert extract_text(dom) == "Hello World"

    # Test with custom symbols
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == "Hello;World"

    # Test empty content
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test mixed content
    dom = fromstring("<div><p>Hello <br/> <strong>World</strong></p></div>")
    assert extract_text(dom) == "Hello \n World"

    # Test with script and style tags (should be ignored)
    dom = fromstring("<div><p>Hello</p><script>alert('xss')</script><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"


# LLM-generated content at query #19
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tag
    dom = fromstring('<p>Hello <b>World</b></p>')
    assert extract_text(dom) == 'Hello World'

    # Test with block tag
    dom = fromstring('<div>Hello</div><div>World</div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with separator tag
    dom = fromstring('<p>Hello<br>World</p>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with nested tags
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with whitespace squashing
    dom = fromstring('<p>Hello   \n  World</p>')
    assert extract_text(dom) == 'Hello World'

    # Test with custom symbols
    dom = fromstring('<div>Hello</div><div>World</div>')
    assert extract_text(dom, block_symbol='|', sep_symbol='||') == 'Hello|World'

    # Test with squash_space=False
    dom = fromstring('<p>Hello   \n  World</p>')
    assert extract_text(dom, squash_space=False) == 'Hello   \n  World'

    # Test with empty dom
    dom = fromstring('<p></p>')
    assert extract_text(dom) == ''

    # Test with text only
    dom = fromstring('Hello World')
    assert extract_text(dom) == 'Hello World'

    # Test with mixed content
    dom = fromstring('<div><p>Hello</p><span>World</span></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with multiple separators
    dom = fromstring('<p>Hello<br><br>World</p>')
    assert extract_text(dom) == 'Hello\n\nWorld'

    # Test with whitespace and separators
    dom = fromstring('<p>Hello   \n  <br>  \n  World</p>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with inline tag with whitespace
    dom = fromstring('<p>Hello <b>  \n  World  \n  </b></p>')
    assert extract_text(dom) == 'Hello World'

    # Test with script tag (inline but should be ignored)
    dom = fromstring('<p>Hello <script>alert("test");</script> World</p>')
    assert extract_text(dom) == 'Hello World'

    # Test with preformatted content (whitespace should be preserved if squash_space=False)
    dom = fromstring('<pre>Hello   \n  World</pre>')
    assert extract_text(dom, squash_space=False) == 'Hello   \n  World'


# LLM-generated content at query #20
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tags
    dom = fromstring('<div><span>Hello</span> <strong>World</strong></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block tags
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with separators
    dom = fromstring('<div>Hello<br/>World</div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with whitespace squashing
    dom = fromstring('<div>Hello   World</div>')
    assert extract_text(dom) == 'Hello World'

    # Test with leading/trailing whitespace
    dom = fromstring('<div>  Hello World  </div>')
    assert extract_text(dom) == 'Hello World'

    # Test with custom block and separator symbols
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol='-') == 'Hello|World'

    # Test with no squash_space
    dom = fromstring('<div>  Hello   World  </div>')
    assert extract_text(dom, squash_space=False) == '  Hello   World  '

    # Test with mixed content
    dom = fromstring('<div><p>Hello <br/> World</p></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with empty content
    dom = fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with only whitespace
    dom = fromstring('<div>   </div>')
    assert extract_text(dom) == ''

    # Test with script and style tags (should be ignored)
    dom = fromstring('<div>Hello <script>alert("test")</script> World</div>')
    assert extract_text(dom) == 'Hello World'


# LLM-generated content at query #21
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline elements
    dom = fromstring("<div><span>Hello</span> <span>World</span></div>")
    assert extract_text(dom) == "Hello World"

    # Test with block elements
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with separator elements
    dom = fromstring("<div>Hello<br/>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with nested elements
    dom = fromstring("<div><p>Hello <span>World</span></p></div>")
    assert extract_text(dom) == "Hello World"

    # Test with whitespace squashing
    dom = fromstring("<div>Hello   World</div>")
    assert extract_text(dom) == "Hello World"

    # Test with custom block and separator symbols
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == "Hello|World"

    # Test with no squashing
    dom = fromstring("<div>Hello   World</div>")
    assert extract_text(dom, squash_space=False) == "Hello   World"

    # Test with empty element
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with text only
    dom = fromstring("<div>Hello World</div>")
    assert extract_text(dom) == "Hello World"

    # Test with mixed content
    dom = fromstring("<div><p>Hello<br/>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with multiple separators
    dom = fromstring("<div>Hello<br/><br/>World</div>")
    assert extract_text(dom) == "Hello\nWorld"


# LLM-generated content at query #22
#--------------------------

```python
def test_extract_text_array():
    from lxml import html

    # Test with inline tags
    dom = html.fromstring('<div><span>Hello</span> <strong>World</strong></div>')
    result = extract_text_array(dom)
    assert result == ['Hello', ' ', 'World']

    # Test with block tags
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello', None, None, 'World', None]

    # Test with separators
    dom = html.fromstring('<div>Hello<br>World</div>')
    result = extract_text_array(dom)
    assert result == ['Hello', True, 'World']

    # Test with nested tags
    dom = html.fromstring('<div><p>Hello <span>World</span></p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello ', 'World', None]

    # Test with text and tail
    dom = html.fromstring('<div><p>Hello</p>World</div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello', None, 'World', None]

    # Test with squash_artifical_nl=False
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = html.fromstring('<div><p>Hello</p></div>')
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'Hello', None]

    # Test with empty dom
    dom = html.fromstring('<div></div>')
    result = extract_text_array(dom)
    assert result == [None, None]

    # Test with callable tag
    class CallableTag:
        def __init__(self):
            self.tag = lambda: None
    dom = CallableTag()
    result = extract_text_array(dom)
    assert result == ''


# LLM-generated content at query #23
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag
    inline_dom = MockElement('span', text='Hello')
    assert extract_text(inline_dom) == 'Hello'

    # Test block tag
    block_dom = MockElement('div', text='Hello')
    assert extract_text(block_dom) == 'Hello'

    # Test nested tags
    nested_dom = MockElement('div', children=[
        MockElement('p', text='Hello'),
        MockElement('p', text='World')
    ])
    assert extract_text(nested_dom) == 'Hello\nWorld'

    # Test with separators
    separator_dom = MockElement('div', children=[
        MockElement('p', text='Hello'),
        MockElement('br'),
        MockElement('p', text='World')
    ])
    assert extract_text(separator_dom) == 'Hello\nWorld'

    # Test with squash_space
    squash_dom = MockElement('div', text='  Hello  ', children=[
        MockElement('p', text='  World  ')
    ])
    assert extract_text(squash_dom, squash_space=True) == 'Hello\nWorld'

    # Test with custom symbols
    custom_dom = MockElement('div', children=[
        MockElement('p', text='Hello'),
        MockElement('p', text='World')
    ])
    assert extract_text(custom_dom, block_symbol='|', sep_symbol='-') == 'Hello|World'

    # Test with None text
    none_dom = MockElement('div', text=None, children=[
        MockElement('p', text='Hello')
    ])
    assert extract_text(none_dom) == 'Hello'

    # Test with empty text
    empty_dom = MockElement('div', text='', children=[
        MockElement('p', text='Hello')
    ])
    assert extract_text(empty_dom) == 'Hello'

    # Test with multiple children
    multi_dom = MockElement('div', children=[
        MockElement('p', text='Hello'),
        MockElement('p', text='World'),
        MockElement('p', text='!')
    ])
    assert extract_text(multi_dom) == 'Hello\nWorld\n!'

    # Test with tail text
    tail_dom = MockElement('div', children=[
        MockElement('p', text='Hello', tail='World')
    ])
    assert extract_text(tail_dom) == 'HelloWorld'


# LLM-generated content at query #24
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    inline_dom = MockDom('span', text='Hello', children=[], tail='World')
    assert extract_text_array(inline_dom) == ['Hello', 'World']

    # Test with block tag
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom) == [None, 'Hello', None, 'World']

    # Test with separator tag
    separator_dom = MockDom('br', text='Hello', children=[], tail='World')
    assert extract_text_array(separator_dom) == [True, 'Hello', 'World']

    # Test with nested tags
    nested_dom = MockDom('div', text='Hello',
                         children=[MockDom('span', text='Nested')],
                         tail='World')
    assert extract_text_array(nested_dom) == [None, 'Hello', 'Nested', None, 'World']

    # Test with squash_artifical_nl=False
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom, squash_artifical_nl=False) == [None, 'Hello', None, 'World']

    # Test with strip_artifical_nl=False
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom, strip_artifical_nl=False) == [None, 'Hello', None, 'World']

    # Test with callable tag
    callable_dom = MockDom(lambda: 'div', text='Hello', children=[], tail='World')
    assert extract_text_array(callable_dom) == ''

    # Test with None text and tail
    none_dom = MockDom('div', text=None, children=[], tail=None)
    assert extract_text_array(none_dom) == [None, None]

    # Test with empty children
    empty_children_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(empty_children_dom) == [None, 'Hello', None, 'World']


# LLM-generated content at query #25
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test 1: Simple inline element
    dom = MockElement('span', text='Hello World')
    assert extract_text(dom) == 'Hello World'

    # Test 2: Block element with squash_space=True
    dom = MockElement('div', children=[
        MockElement('p', text='First paragraph'),
        MockElement('p', text='Second paragraph')
    ])
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'

    # Test 3: Block element with squash_space=False
    dom = MockElement('div', children=[
        MockElement('p', text='First paragraph'),
        MockElement('p', text='Second paragraph')
    ])
    assert extract_text(dom, squash_space=False) == '\nFirst paragraph\n\nSecond paragraph\n'

    # Test 4: Separator element (br)
    dom = MockElement('div', children=[
        MockElement('span', text='First line'),
        MockElement('br'),
        MockElement('span', text='Second line')
    ])
    assert extract_text(dom) == 'First line\nSecond line'

    # Test 5: Nested elements
    dom = MockElement('div', children=[
        MockElement('p', text='Outer text',
                   children=[MockElement('span', text=' inner text ')]),
        MockElement('p', text='Another paragraph')
    ])
    assert extract_text(dom) == 'Outer text inner text\nAnother paragraph'

    # Test 6: Whitespace handling
    dom = MockElement('div', text='  \n  \t  ', children=[
        MockElement('span', text='  text  with  spaces  '),
        MockElement('p', text='\n\nMore text\n\n')
    ])
    assert extract_text(dom) == 'text with spaces\nMore text'

    # Test 7: Callable tag (should return empty string)
    dom = MockElement(lambda: None)
    assert extract_text(dom) == ''

    # Test 8: Custom symbols
    dom = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('p', text='Second')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'First|Second'

    # Test 9: Empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''

    # Test 10: Mixed content with tail text
    dom = MockElement('div', children=[
        MockElement('span', text='First', tail=' tail text '),
        MockElement('p', text='Second paragraph')
    ])
    assert extract_text(dom) == 'First tail text\nSecond paragraph'


# LLM-generated content at query #26
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline element
    inline_elem = MockElement('span', text='Hello')
    assert extract_text(inline_elem) == 'Hello'

    # Test block element
    block_elem = MockElement('div', text='Hello', children=[
        MockElement('p', text='World')
    ])
    assert extract_text(block_elem) == 'Hello\nWorld'

    # Test separator element
    sep_elem = MockElement('br')
    assert extract_text(sep_elem) == '\n'

    # Test nested elements
    nested_elem = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('p', text='Second')
    ])
    assert extract_text(nested_elem) == 'First\nSecond'

    # Test with tail text
    tail_elem = MockElement('div', children=[
        MockElement('span', text='Hello', tail='World')
    ])
    assert extract_text(tail_elem) == 'HelloWorld'

    # Test squash_space parameter
    space_elem = MockElement('div', text='  Hello  ', children=[
        MockElement('p', text='  World  ')
    ])
    assert extract_text(space_elem, squash_space=True) == 'Hello\nWorld'
    assert extract_text(space_elem, squash_space=False) == '  Hello  \n  World  '

    # Test custom symbols
    custom_elem = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('p', text='Second')
    ])
    assert extract_text(custom_elem, block_symbol='|', sep_symbol='-') == 'First|Second'

    # Test empty element
    empty_elem = MockElement('div')
    assert extract_text(empty_elem) == ''

    # Test mixed content
    mixed_elem = MockElement('div', text='Start', children=[
        MockElement('span', text='Middle'),
        MockElement('br'),
        MockElement('p', text='End')
    ])
    assert extract_text(mixed_elem) == 'StartMiddle\nEnd'

    # Test callable tag
    callable_elem = MockElement(lambda: 'div')
    assert extract_text(callable_elem) == ''


# LLM-generated content at query #27
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    inline_dom = MockDom('span', text='Hello')
    assert extract_text_array(inline_dom) == ['Hello']

    # Test with separator tag
    separator_dom = MockDom('br')
    assert extract_text_array(separator_dom) == [True]

    # Test with block tag
    block_dom = MockDom('div', text='Hello')
    assert extract_text_array(block_dom) == [None, 'Hello', None]

    # Test with nested tags
    nested_dom = MockDom('div', children=[
        MockDom('p', text='Para1'),
        MockDom('p', text='Para2')
    ])
    assert extract_text_array(nested_dom) == [None, 'Para1', None, 'Para2', None, None]

    # Test with text and tail
    tail_dom = MockDom('div', text='Start', children=[
        MockDom('span', text='Middle', tail='End')
    ])
    assert extract_text_array(tail_dom) == [None, 'Start', 'Middle', 'End', None]

    # Test with squash_artifical_nl=False
    block_dom = MockDom('div', text='Hello')
    assert extract_text_array(block_dom, squash_artifical_nl=False) == [None, 'Hello', None]

    # Test with strip_artifical_nl=False
    block_dom = MockDom('div', text='Hello')
    assert extract_text_array(block_dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with callable tag
    callable_dom = MockDom(lambda: 'div')
    assert extract_text_array(callable_dom) == ''

    # Test with None text
    none_text_dom = MockDom('div', text=None)
    assert extract_text_array(none_text_dom) == [None, None]

    # Test with empty children
    empty_children_dom = MockDom('div', text='Hello', children=[])
    assert extract_text_array(empty_children_dom) == [None, 'Hello', None]


# LLM-generated content at query #28
#--------------------------

```python
def test_extract_text():
    # Test basic inline tag
    dom = Mock(tag='span', text='Hello', getchildren=lambda: [])
    assert extract_text(dom) == 'Hello'

    # Test block tag with squash_space=True
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text(dom) == 'Hello'

    # Test separator tag
    dom = Mock(tag='br', text=None, getchildren=lambda: [])
    assert extract_text(dom, sep_symbol='\n') == '\n'

    # Test nested tags
    child = Mock(tag='span', text='World', getchildren=lambda: [])
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [child], tail='!')
    assert extract_text(dom) == 'Hello World!'

    # Test with squash_space=False
    dom = Mock(tag='div', text='  Hello  ', getchildren=lambda: [])
    assert extract_text(dom, squash_space=False) == '  Hello  '

    # Test with custom block and separator symbols
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text(dom, block_symbol='|', sep_symbol='-') == 'Hello'

    # Test with multiple block elements
    child1 = Mock(tag='p', text='First', getchildren=lambda: [])
    child2 = Mock(tag='p', text='Second', getchildren=lambda: [])
    dom = Mock(tag='div', text=None, getchildren=lambda: [child1, child2])
    assert extract_text(dom) == 'First\nSecond'

    # Test with separator elements
    child = Mock(tag='br', text=None, getchildren=lambda: [])
    dom = Mock(tag='div', text='Line1', getchildren=lambda: [child], tail='Line2')
    assert extract_text(dom) == 'Line1\nLine2'

    # Test with whitespace squashing
    dom = Mock(tag='div', text='  Hello   World  ', getchildren=lambda: [])
    assert extract_text(dom) == 'Hello World'

    # Test with empty text
    dom = Mock(tag='div', text='', getchildren=lambda: [])
    assert extract_text(dom) == ''

    # Test with None text
    dom = Mock(tag='div', text=None, getchildren=lambda: [])
    assert extract_text(dom) == ''

    # Test with callable tag
    dom = Mock(tag=lambda: 'div', text='Hello', getchildren=lambda: [])
    assert extract_text(dom) == ''


# LLM-generated content at query #29
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockInlineTag:
        tag = 'span'
        text = 'Hello'
        tail = 'World'
        def getchildren(self):
            return []

    inline_dom = MockInlineTag()
    assert extract_text_array(inline_dom) == ['Hello', 'World']

    # Test with block tag
    class MockBlockTag:
        tag = 'div'
        text = 'Hello'
        tail = 'World'
        def getchildren(self):
            return []

    block_dom = MockBlockTag()
    assert extract_text_array(block_dom) == [None, 'Hello', 'World', None]

    # Test with separator tag
    class MockSeparatorTag:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []

    separator_dom = MockSeparatorTag()
    assert extract_text_array(separator_dom) == [True]

    # Test with nested tags
    class MockParentTag:
        tag = 'div'
        text = 'Start'
        tail = 'End'
        def getchildren(self):
            return [MockChildTag()]

    class MockChildTag:
        tag = 'span'
        text = 'Middle'
        tail = 'Tail'
        def getchildren(self):
            return []

    parent_dom = MockParentTag()
    assert extract_text_array(parent_dom) == [None, 'Start', 'Middle', 'Tail', 'End', None]

    # Test with squash_artifical_nl=False
    class MockTagWithMultipleNone:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return []

    tag_dom = MockTagWithMultipleNone()
    assert extract_text_array(tag_dom, squash_artifical_nl=False) == [None, None]

    # Test with strip_artifical_nl=False
    class MockTagWithText:
        tag = 'div'
        text = 'Text'
        tail = None
        def getchildren(self):
            return []

    text_dom = MockTagWithText()
    assert extract_text_array(text_dom, strip_artifical_nl=False) == [None, 'Text', None]

    # Test with callable tag
    class MockCallableTag:
        def tag(self):
            return 'div'

    callable_dom = MockCallableTag()
    assert extract_text_array(callable_dom) == ''


# LLM-generated content at query #30
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    # Test inline tag
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test block tag
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test nested tags
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test separator tag
    dom = MockDom('br')
    assert extract_text(dom, sep_symbol='\n') == '\n'

    # Test squash_space
    dom = MockDom('div', text='Hello   World')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test block_symbol
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text(dom, block_symbol='\n') == 'Hello\nWorld'

    # Test strip_artifical_nl
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child], tail='!')
    assert extract_text(dom) == 'Hello World!'

    # Test complex nested structure
    grandchild = MockDom('strong', text='nested')
    child = MockDom('span', text='World', children=[grandchild])
    dom = MockDom('div', text='Hello', children=[child], tail='!')
    assert extract_text(dom) == 'Hello World nested!'

    # Test with multiple children
    child1 = MockDom('span', text='First')
    child2 = MockDom('span', text='Second')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom) == 'First Second'

    # Test with None text
    dom = MockDom('div', text=None)
    assert extract_text(dom) == ''

    # Test with empty string text
    dom = MockDom('div', text='')
    assert extract_text(dom) == ''

    # Test with whitespace only text
    dom = MockDom('div', text='   ')
    assert extract_text(dom, squash_space=True) == ''

    # Test with callable tag
    class CallableTagDom:
        def __init__(self):
            self.tag = lambda: 'div'
            self.text = 'Hello'

    dom = CallableTagDom()
    assert extract_text(dom) == ''


# LLM-generated content at query #31
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with inline tags
    dom = fromstring('<div><span>Hello</span> <strong>World</strong></div>')
    assert extract_text_array(dom) == ['Hello', ' ', 'World']

    # Test with block tags
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom) == [None, 'Hello', None, 'World', None]

    # Test with separators
    dom = fromstring('<div>Hello<br>World</div>')
    assert extract_text_array(dom) == ['Hello', True, 'World', None]

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text_array(dom) == [None, 'Hello ', 'World', None, None]

    # Test with squash_artifical_nl=False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None, None]

    # Test with strip_artifical_nl=False
    dom = fromstring('<div><p>Hello</p></div>')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with both flags False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Hello', None, None, 'World', None, None]

    # Test with empty dom
    dom = fromstring('<div></div>')
    assert extract_text_array(dom) == [None, None]

    # Test with text only
    dom = fromstring('<div>Hello World</div>')
    assert extract_text_array(dom) == ['Hello World', None]

    # Test with mixed content
    dom = fromstring('<div>Hello <br> <span>World</span> <p>!</p></div>')
    assert extract_text_array(dom) == ['Hello', True, ' ', 'World', ' ', None, '!', None, None]


# LLM-generated content at query #32
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    # Test inline tag
    dom = MockElement('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test block tag
    dom = MockElement('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test nested tags
    child = MockElement('span', text='World')
    dom = MockElement('div', text='Hello ', children=[child], tail='!')
    assert extract_text(dom) == 'Hello World!'

    # Test separator tag (br)
    dom = MockElement('br')
    assert extract_text(dom) == '\n'

    # Test multiple block tags
    child1 = MockElement('div', text='Line1')
    child2 = MockElement('div', text='Line2')
    dom = MockElement('div', children=[child1, child2])
    assert extract_text(dom) == 'Line1\nLine2'

    # Test with custom symbols
    dom = MockElement('div', text='Hello')
    assert extract_text(dom, block_symbol='|', sep_symbol='||') == 'Hello'

    # Test squash_space
    dom = MockElement('div', text='  Hello   World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test with None text
    dom = MockElement('div', text=None)
    assert extract_text(dom) == ''

    # Test with empty children
    dom = MockElement('div', text='Hello', children=[])
    assert extract_text(dom) == 'Hello'

    # Test with multiple separators
    child1 = MockElement('br')
    child2 = MockElement('br')
    dom = MockElement('div', children=[child1, child2])
    assert extract_text(dom) == '\n\n'

    # Test with mixed content
    child1 = MockElement('span', text='Hello')
    child2 = MockElement('br')
    child3 = MockElement('span', text='World')
    dom = MockElement('div', children=[child1, child2, child3])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with tail text
    child = MockElement('span', text='Hello')
    dom = MockElement('div', children=[child], tail='World')
    assert extract_text(dom) == 'HelloWorld'

    # Test with nested block tags
    inner_child = MockElement('span', text='Inner')
    child = MockElement('div', text='Middle ', children=[inner_child])
    dom = MockElement('div', text='Outer ', children=[child], tail=' End')
    assert extract_text(dom) == 'Outer Middle Inner End'

    # Test with whitespace handling
    dom = MockElement('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test with callable tag
    def callable_tag():
        return 'div'

    dom = MockElement(callable_tag, text='Hello')
    assert extract_text(dom) == ''


# LLM-generated content at query #33
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test block tag
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test nested tags
    dom = MockDom('div', children=[
        MockDom('span', text='Hello'),
        MockDom('span', text='World')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test with block_symbol
    dom = MockDom('div', children=[
        MockDom('span', text='Hello'),
        MockDom('span', text='World')
    ])
    assert extract_text(dom, block_symbol='\n') == 'Hello\nWorld'

    # Test with sep_symbol
    dom = MockDom('div', children=[
        MockDom('br'),
        MockDom('span', text='Hello'),
        MockDom('br'),
        MockDom('span', text='World')
    ])
    assert extract_text(dom, sep_symbol='\n') == 'Hello\nWorld'

    # Test squash_space
    dom = MockDom('div', text='  Hello  ', children=[
        MockDom('span', text='  World  ')
    ])
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test with tail text
    dom = MockDom('div', children=[
        MockDom('span', text='Hello', tail='World')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test with None text
    dom = MockDom('div', text=None, children=[
        MockDom('span', text='Hello')
    ])
    assert extract_text(dom) == 'Hello'

    # Test with callable tag
    dom = MockDom(tag=lambda: 'div', text='Hello')
    assert extract_text(dom) == ''

    # Test with separator tag
    dom = MockDom('div', children=[
        MockDom('br'),
        MockDom('span', text='Hello')
    ])
    assert extract_text(dom) == 'Hello'

    # Test with multiple separators
    dom = MockDom('div', children=[
        MockDom('br'),
        MockDom('br'),
        MockDom('span', text='Hello')
    ])
    assert extract_text(dom) == 'Hello'


# LLM-generated content at query #34
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    inline_dom = MockDom('span', text='Hello', children=[], tail='World')
    assert extract_text_array(inline_dom) == ['Hello', 'World']

    # Test with block tag
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom) == [None, 'Hello', None, 'World']

    # Test with separator tag
    separator_dom = MockDom('br', text=None, children=[], tail=None)
    assert extract_text_array(separator_dom) == [True]

    # Test with nested tags
    nested_dom = MockDom('div', text='Hello',
                         children=[MockDom('span', text='Nested')],
                         tail='World')
    assert extract_text_array(nested_dom) == [None, 'Hello', 'Nested', None, 'World']

    # Test with squash_artifical_nl=False
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom, squash_artifical_nl=False) == [None, 'Hello', None, 'World']

    # Test with strip_artifical_nl=False
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom, strip_artifical_nl=False) == [None, 'Hello', None, 'World']

    # Test with callable tag
    callable_dom = MockDom(lambda: 'div', text='Hello', children=[], tail='World')
    assert extract_text_array(callable_dom) == ''

    # Test with None text and tail
    none_dom = MockDom('div', text=None, children=[], tail=None)
    assert extract_text_array(none_dom) == [None, None]

    # Test with multiple children
    multi_child_dom = MockDom('div', text='Start',
                              children=[MockDom('span', text='Child1'),
                                       MockDom('span', text='Child2')],
                              tail='End')
    assert extract_text_array(multi_child_dom) == [None, 'Start', 'Child1', 'Child2', None, 'End']

    # Test with squash_artifical_nl and strip_artifical_nl
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom, squash_artifical_nl=True, strip_artifical_nl=True) == ['Hello', 'World']


# LLM-generated content at query #35
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    # Test with simple inline element
    dom = MockElement('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test with block element
    dom = MockElement('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test with nested elements
    child = MockElement('span', text='World')
    dom = MockElement('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test with separator element (br)
    dom = MockElement('br')
    assert extract_text(dom, sep_symbol='\n') == '\n'

    # Test with squash_space=True
    dom = MockElement('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test with squash_space=False
    dom = MockElement('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=False) == '  Hello  \n  World  '

    # Test with block_symbol and sep_symbol
    child1 = MockElement('span', text='Hello')
    child2 = MockElement('span', text='World')
    dom = MockElement('div', children=[child1, child2])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello;World'

    # Test with multiple levels of nesting
    grandchild = MockElement('strong', text='!')
    child = MockElement('span', text='World', children=[grandchild])
    dom = MockElement('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World!'

    # Test with tail text
    child = MockElement('span', text='Hello', tail='World')
    dom = MockElement('div', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test with None text and tail
    child = MockElement('span', text=None, tail=None)
    dom = MockElement('div', text=None, children=[child])
    assert extract_text(dom) == ''

    # Test with callable tag
    dom = MockElement(tag=lambda: 'div')
    assert extract_text(dom) == ''


# LLM-generated content at query #36
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    # Test with simple inline element
    dom = MockElement('span', text='Hello World')
    assert extract_text(dom) == 'Hello World'

    # Test with block element
    dom = MockElement('div', text='Hello', children=[
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test with separator element
    dom = MockElement('div', children=[
        MockElement('span', text='Hello'),
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom, sep_symbol='\n') == 'Hello\nWorld'

    # Test with nested block elements
    dom = MockElement('div', children=[
        MockElement('p', text='First paragraph'),
        MockElement('p', text='Second paragraph')
    ])
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'

    # Test with squash_space=True
    dom = MockElement('div', text='  Hello  ', children=[
        MockElement('span', text='  World  ')
    ])
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test with squash_space=False
    dom = MockElement('div', text='  Hello  ', children=[
        MockElement('span', text='  World  ')
    ])
    assert extract_text(dom, squash_space=False) == '  Hello   World  '

    # Test with custom block_symbol
    dom = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('p', text='Second')
    ])
    assert extract_text(dom, block_symbol='|') == 'First|Second'

    # Test with custom sep_symbol
    dom = MockElement('div', children=[
        MockElement('span', text='Hello'),
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom, sep_symbol='|') == 'Hello|World'

    # Test with empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''

    # Test with None text
    dom = MockElement('div', text=None)
    assert extract_text(dom) == ''

    # Test with mixed content
    dom = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('span', text='inline'),
        MockElement('br'),
        MockElement('p', text='Second')
    ])
    assert extract_text(dom) == 'First inline\nSecond'


# LLM-generated content at query #37
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test with simple inline tag
    dom = MockElement('p', children=[MockElement('b', text='Hello')])
    assert extract_text(dom) == 'Hello'

    # Test with block tag and squash_space=True
    dom = MockElement('div', children=[
        MockElement('p', text='First paragraph'),
        MockElement('p', text='Second paragraph')
    ])
    assert extract_text(dom, squash_space=True) == 'First paragraph\nSecond paragraph'

    # Test with separator tag (br)
    dom = MockElement('div', children=[
        MockElement('span', text='Line one'),
        MockElement('br'),
        MockElement('span', text='Line two')
    ])
    assert extract_text(dom, sep_symbol='\n') == 'Line one\nLine two'

    # Test with custom block_symbol
    dom = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('p', text='Second')
    ])
    assert extract_text(dom, block_symbol=' | ') == 'First | Second'

    # Test with whitespace squashing
    dom = MockElement('div', text='  \n  Hello  \n  ', children=[
        MockElement('span', text='  World  ', tail='  !  ')
    ])
    assert extract_text(dom, squash_space=True) == 'Hello World !'

    # Test with no squash_space
    dom = MockElement('div', text='  \n  Hello  \n  ', children=[
        MockElement('span', text='  World  ', tail='  !  ')
    ])
    assert extract_text(dom, squash_space=False) == '  \n  Hello  \n   World   !  '

    # Test with nested tags
    dom = MockElement('div', children=[
        MockElement('p', children=[
            MockElement('span', text='Nested '),
            MockElement('strong', text='text')
        ])
    ])
    assert extract_text(dom) == 'Nested text'

    # Test with empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''

    # Test with callable tag (should return empty string)
    dom = MockElement(lambda: None)
    assert extract_text(dom) == ''


# LLM-generated content at query #38
#--------------------------

```python
def test_extract_text():
    # Test basic inline tag handling
    dom = Mock(tag='p', text='Hello ')
    child = Mock(tag='strong', text='World', tail='!')
    dom.getchildren.return_value = [child]
    assert extract_text(dom) == 'Hello World!'

    # Test block tag handling
    dom = Mock(tag='div', text='Line1 ')
    child = Mock(tag='p', text='Line2', tail=' Line3')
    dom.getchildren.return_value = [child]
    assert extract_text(dom) == 'Line1\nLine2\nLine3'

    # Test separator handling
    dom = Mock(tag='div', text='First ')
    br = Mock(tag='br')
    child = Mock(tag='span', text='Second', tail=' Third')
    dom.getchildren.return_value = [br, child]
    assert extract_text(dom) == 'First\nSecond Third'

    # Test whitespace squashing
    dom = Mock(tag='div', text='  Multiple   spaces  ')
    assert extract_text(dom, squash_space=True) == 'Multiple spaces'

    # Test custom symbols
    dom = Mock(tag='div', text='Part1 ')
    child = Mock(tag='p', text='Part2', tail=' Part3')
    dom.getchildren.return_value = [child]
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Part1|Part2;Part3'

    # Test empty text
    dom = Mock(tag='div', text=None)
    assert extract_text(dom) == ''

    # Test nested inline tags
    dom = Mock(tag='div', text='Start ')
    b = Mock(tag='b', text='Bold ')
    i = Mock(tag='i', text='Italic', tail=' End')
    b.getchildren.return_value = [i]
    dom.getchildren.return_value = [b]
    assert extract_text(dom) == 'Start Bold Italic End'

    # Test with squash_space=False
    dom = Mock(tag='div', text='  No  squash  ')
    assert extract_text(dom, squash_space=False) == '  No  squash  '

    # Test multiple separators
    dom = Mock(tag='div', text='A ')
    br1 = Mock(tag='br')
    br2 = Mock(tag='br')
    child = Mock(tag='span', text='B')
    dom.getchildren.return_value = [br1, br2, child]
    assert extract_text(dom) == 'A\n\nB'

    # Test strip_artifical_nl behavior
    dom = Mock(tag='div', text='\n  Content  \n')
    assert extract_text(dom, squash_space=True) == 'Content'


# LLM-generated content at query #39
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test simple text extraction
    dom = fromstring("<div>Hello World</div>")
    assert extract_text(dom) == "Hello World"

    # Test with block elements
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with inline elements
    dom = fromstring("<div><span>Hello</span> <span>World</span></div>")
    assert extract_text(dom) == "Hello World"

    # Test with separators (br)
    dom = fromstring("<div>Hello<br/>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with nested elements
    dom = fromstring("<div><p>Hello <span>World</span></p></div>")
    assert extract_text(dom) == "Hello World"

    # Test with custom block and separator symbols
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == "Hello|World"

    # Test with squash_space=False
    dom = fromstring("<div>  Hello  World  </div>")
    assert extract_text(dom, squash_space=False) == "  Hello  World  "

    # Test with empty elements
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with mixed content
    dom = fromstring("<div><p>Hello</p><br/><p>World</p></div>")
    assert extract_text(dom) == "Hello\n\nWorld"

    # Test with text in tail
    dom = fromstring("<div><p>Hello</p> World</div>")
    assert extract_text(dom) == "Hello World"


# LLM-generated content at query #40
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tags
    dom = fromstring('<div><p>Hello <strong>World</strong></p></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block tags
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with separators
    dom = fromstring('<div><p>Hello<br>World</p></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World <em>!</em></span></p></div>')
    assert extract_text(dom) == 'Hello World !'

    # Test with whitespace squashing
    dom = fromstring('<div><p>Hello   World</p></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with custom symbols
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == 'Hello|World'

    # Test with squash_space=False
    dom = fromstring('<div><p>Hello   World</p></div>')
    assert extract_text(dom, squash_space=False) == 'Hello   World'

    # Test with empty content
    dom = fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with mixed content
    dom = fromstring('<div><p>Hello</p><br><p>World</p></div>')
    assert extract_text(dom) == 'Hello\n\nWorld'

    # Test with script and style tags (should be ignored)
    dom = fromstring('<div><p>Hello</p><script>alert("test")</script><p>World</p></div>')
    assert extract_text(dom) == 'Hello\nWorld'


# LLM-generated content at query #41
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline element
    inline_elem = MockElement('span', text='Hello')
    assert extract_text(inline_elem) == 'Hello'

    # Test block element
    block_elem = MockElement('div', text='Hello', children=[
        MockElement('p', text='World')
    ])
    assert extract_text(block_elem) == 'Hello\nWorld'

    # Test separator element (br)
    br_elem = MockElement('br')
    assert extract_text(br_elem) == '\n'

    # Test nested elements
    nested_elem = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('p', text='Second')
    ])
    assert extract_text(nested_elem) == 'First\nSecond'

    # Test with tail text
    elem_with_tail = MockElement('div', children=[
        MockElement('span', text='Hello', tail='World')
    ])
    assert extract_text(elem_with_tail) == 'Hello World'

    # Test squash_space=True (default)
    elem_with_spaces = MockElement('div', text='  Hello  ', children=[
        MockElement('p', text='  World  ')
    ])
    assert extract_text(elem_with_spaces) == 'Hello\nWorld'

    # Test squash_space=False
    assert extract_text(elem_with_spaces, squash_space=False) == '  Hello  \n  World  '

    # Test custom block and separator symbols
    assert extract_text(block_elem, block_symbol='|', sep_symbol=';') == 'Hello|World'

    # Test with None text
    elem_none_text = MockElement('div', text=None, children=[
        MockElement('p', text='Content')
    ])
    assert extract_text(elem_none_text) == 'Content'

    # Test empty element
    empty_elem = MockElement('div')
    assert extract_text(empty_elem) == ''

    # Test mixed content
    mixed_elem = MockElement('div', text='Start', children=[
        MockElement('span', text='Middle'),
        MockElement('br'),
        MockElement('p', text='End')
    ])
    assert extract_text(mixed_elem) == 'Start Middle\nEnd'


# LLM-generated content at query #42
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self.children = children or []

        def getchildren(self):
            return self.children

    # Test inline tag
    inline_elem = MockElement('span', text='Hello')
    assert extract_text(inline_elem) == 'Hello'

    # Test block tag
    block_elem = MockElement('div', text='Hello', children=[
        MockElement('p', text='World')
    ])
    assert extract_text(block_elem) == 'Hello\nWorld'

    # Test separator tag
    sep_elem = MockElement('br')
    assert extract_text(sep_elem) == '\n'

    # Test nested tags
    nested_elem = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('p', text='Second')
    ])
    assert extract_text(nested_elem) == 'First\nSecond'

    # Test with squash_space
    space_elem = MockElement('div', text='  Hello  ', children=[
        MockElement('p', text='  World  ')
    ])
    assert extract_text(space_elem, squash_space=True) == 'Hello\nWorld'

    # Test with custom symbols
    custom_elem = MockElement('div', children=[
        MockElement('p', text='A'),
        MockElement('p', text='B')
    ])
    assert extract_text(custom_elem, block_symbol='|', sep_symbol=';') == 'A|B'

    # Test empty element
    empty_elem = MockElement('div')
    assert extract_text(empty_elem) == ''

    # Test mixed content
    mixed_elem = MockElement('div', text='Start', children=[
        MockElement('span', text='Middle'),
        MockElement('br'),
        MockElement('p', text='End')
    ])
    assert extract_text(mixed_elem) == 'StartMiddle\nEnd'


# LLM-generated content at query #43
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    # Test 1: Simple inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test 2: Block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test 3: Separator tag (br)
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test 4: Nested tags
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child], tail='!')
    assert extract_text_array(dom) == [None, 'Hello', 'World', '!', None]

    # Test 5: Multiple children
    child1 = MockDom('span', text='Child1')
    child2 = MockDom('span', text='Child2')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text_array(dom) == [None, 'Child1', 'Child2', None]

    # Test 6: With squash_artifical_nl=False and strip_artifical_nl=False
    dom = MockDom('div', text='  Hello  ', children=[], tail='  World  ')
    assert extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, '  Hello  ', '  World  ', None]

    # Test 7: With callable tag (should return empty string)
    class CallableTagDom:
        def __init__(self):
            self.tag = lambda: None
    dom = CallableTagDom()
    assert extract_text_array(dom) == ''

    # Test 8: Complex nested structure
    grandchild = MockDom('strong', text='nested')
    child = MockDom('p', text='Some ', children=[grandchild], tail=' text')
    dom = MockDom('div', text='Start ', children=[child], tail=' End')
    result = extract_text_array(dom)
    assert result == [None, 'Start ', None, 'Some ', 'nested', ' text', None, ' End', None]


# LLM-generated content at query #44
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    inline_dom = MockDom('span', text='Hello', children=[], tail='World')
    assert extract_text_array(inline_dom) == ['Hello', 'World']

    # Test with separator tag
    separator_dom = MockDom('br', text='Hello', children=[], tail='World')
    assert extract_text_array(separator_dom) == ['Hello', True, 'World']

    # Test with block tag
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom) == [None, 'Hello', None, 'World']

    # Test with nested tags
    nested_dom = MockDom('div', text='Hello', children=[
        MockDom('span', text='Nested', children=[], tail='Text')
    ], tail='World')
    assert extract_text_array(nested_dom) == [None, 'Hello', 'Nested', 'Text', None, 'World']

    # Test with squash_artifical_nl=False
    block_dom_no_squash = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom_no_squash, squash_artifical_nl=False) == [None, 'Hello', None, 'World']

    # Test with strip_artifical_nl=False
    block_dom_no_strip = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom_no_strip, strip_artifical_nl=False) == [None, 'Hello', None, 'World']

    # Test with callable tag
    callable_dom = MockDom(lambda: 'div', text='Hello', children=[], tail='World')
    assert extract_text_array(callable_dom) == ''

    # Test with None text and tail
    none_dom = MockDom('div', text=None, children=[], tail=None)
    assert extract_text_array(none_dom) == [None, None]


# LLM-generated content at query #45
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test block tag with text and children
    dom = MockDom('div', text='Start',
                  children=[MockDom('span', text='Middle')],
                  tail='End')
    assert extract_text_array(dom) == [None, 'Start', 'Middle', None, 'End', None]

    # Test separator tag (br)
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test nested structure
    dom = MockDom('div',
                  children=[
                      MockDom('p', text='Para1',
                              children=[MockDom('br')],
                              tail='Tail1'),
                      MockDom('p', text='Para2')
                  ])
    assert extract_text_array(dom) == [None, 'Para1', True, 'Tail1', None, 'Para2', None]

    # Test with squash_artifical_nl=False
    dom = MockDom('div', text='\n\nText\n\n',
                  children=[MockDom('span', text='\nInner\n')])
    assert extract_text_array(dom, squash_artifical_nl=False) == [
        None, '\n\nText\n\n', '\nInner\n', None
    ]

    # Test with strip_artifical_nl=False
    dom = MockDom('div',
                  children=[MockDom('span', text='Content')])
    assert extract_text_array(dom, strip_artifical_nl=False) == [
        None, 'Content', None
    ]

    # Test empty dom
    dom = MockDom('div')
    assert extract_text_array(dom) == [None, None]

    # Test callable tag (should return empty string)
    dom = MockDom(lambda: None)
    assert extract_text_array(dom) == ''


# LLM-generated content at query #46
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    # Test inline element with text
    inline_elem = MockElement('span', text='Hello')
    assert extract_text_array(inline_elem) == ['Hello']

    # Test block element with text
    block_elem = MockElement('div', text='Hello')
    assert extract_text_array(block_elem) == [None, 'Hello', None]

    # Test separator element
    separator_elem = MockElement('br')
    assert extract_text_array(separator_elem) == [True]

    # Test nested elements
    parent = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('p', text='Second')
    ])
    assert extract_text_array(parent) == [None, 'First', None, 'Second', None]

    # Test with tail text
    parent_with_tail = MockElement('div', children=[
        MockElement('span', text='First', tail='Tail')
    ])
    assert extract_text_array(parent_with_tail) == [None, 'First', 'Tail', None]

    # Test squash_artifical_nl=False
    block_elem_no_squash = MockElement('div', text='Hello')
    assert extract_text_array(block_elem_no_squash, squash_artifical_nl=False) == [None, 'Hello', None]

    # Test strip_artifical_nl=False
    block_elem_no_strip = MockElement('div', text='Hello')
    assert extract_text_array(block_elem_no_strip, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test callable tag
    callable_tag_elem = MockElement(lambda: 'div', text='Hello')
    assert extract_text_array(callable_tag_elem) == ''

    # Test empty element
    empty_elem = MockElement('div')
    assert extract_text_array(empty_elem) == [None, None]


# LLM-generated content at query #47
#--------------------------

```python
def test_extract_text():
    # Test basic inline tag
    dom = Mock(tag='span', text='Hello', getchildren=lambda: [])
    assert extract_text(dom) == 'Hello'

    # Test block tag with squash_space
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [], tail=None)
    assert extract_text(dom, squash_space=True) == 'Hello'

    # Test block tag without squash_space
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [], tail=None)
    assert extract_text(dom, squash_space=False) == '\nHello\n'

    # Test nested tags
    child = Mock(tag='span', text='World', getchildren=lambda: [], tail='!')
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [child], tail=None)
    assert extract_text(dom) == 'HelloWorld!'

    # Test separator tag (br)
    dom = Mock(tag='br', text=None, getchildren=lambda: [], tail=None)
    assert extract_text(dom) == '\n'

    # Test multiple children with whitespace
    child1 = Mock(tag='span', text='Hello', getchildren=lambda: [], tail=' ')
    child2 = Mock(tag='span', text='World', getchildren=lambda: [], tail=None)
    dom = Mock(tag='div', text=None, getchildren=lambda: [child1, child2], tail=None)
    assert extract_text(dom) == 'Hello World'

    # Test with custom block and separator symbols
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [], tail=None)
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == 'Hello'

    # Test with whitespace squashing
    dom = Mock(tag='div', text='  Hello  ', getchildren=lambda: [], tail='  World  ')
    assert extract_text(dom) == 'Hello World'

    # Test empty dom
    dom = Mock(tag='div', text=None, getchildren=lambda: [], tail=None)
    assert extract_text(dom) == ''

    # Test with None text and tail
    dom = Mock(tag='div', text=None, getchildren=lambda: [], tail=None)
    assert extract_text(dom) == ''

    # Test with multiple block tags
    child = Mock(tag='p', text='Paragraph', getchildren=lambda: [], tail=None)
    dom = Mock(tag='div', text=None, getchildren=lambda: [child], tail=None)
    assert extract_text(dom) == 'Paragraph'

    # Test with preformatted content (no squash_space)
    dom = Mock(tag='pre', text='  Hello  \n  World  ', getchildren=lambda: [], tail=None)
    assert extract_text(dom, squash_space=False) == '\n  Hello  \n  World  \n'


# LLM-generated content at query #48
#--------------------------

```python
def test_extract_text():
    # Test simple inline tag
    dom = Mock(tag='span', text='Hello', getchildren=lambda: [])
    assert extract_text(dom) == 'Hello'

    # Test block tag with squash_space
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [], tail=None)
    assert extract_text(dom, squash_space=True) == 'Hello'

    # Test block tag without squash_space
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [], tail=None)
    assert extract_text(dom, squash_space=False) == '\nHello\n'

    # Test separator tag
    dom = Mock(tag='br', text=None, getchildren=lambda: [], tail=None)
    assert extract_text(dom) == '\n'

    # Test nested tags
    child = Mock(tag='span', text='World', getchildren=lambda: [], tail='!')
    dom = Mock(tag='div', text='Hello ', getchildren=lambda: [child], tail=None)
    assert extract_text(dom, squash_space=True) == 'Hello World!'

    # Test multiple children
    child1 = Mock(tag='span', text='Hello', getchildren=lambda: [], tail=' ')
    child2 = Mock(tag='span', text='World', getchildren=lambda: [], tail='!')
    dom = Mock(tag='div', text=None, getchildren=lambda: [child1, child2], tail=None)
    assert extract_text(dom, squash_space=True) == 'Hello World!'

    # Test with custom symbols
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [], tail=None)
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == 'Hello'

    # Test with whitespace squashing
    dom = Mock(tag='div', text='  Hello  ', getchildren=lambda: [], tail='  World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test with no squashing
    dom = Mock(tag='div', text='  Hello  ', getchildren=lambda: [], tail='  World  ')
    assert extract_text(dom, squash_space=False) == '\n  Hello   World  \n'

    # Test empty dom
    dom = Mock(tag='div', text=None, getchildren=lambda: [], tail=None)
    assert extract_text(dom) == ''

    # Test with only whitespace
    dom = Mock(tag='div', text='   ', getchildren=lambda: [], tail='   ')
    assert extract_text(dom, squash_space=True) == ''

    # Test with mixed content
    child = Mock(tag='br', text=None, getchildren=lambda: [], tail=None)
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [child], tail='World')
    assert extract_text(dom, squash_space=True) == 'Hello\nWorld'


# LLM-generated content at query #49
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with simple inline tag
    dom = fromstring('<p>Hello <b>World</b></p>')
    assert extract_text_array(dom) == ['Hello ', 'World']

    # Test with block tag
    dom = fromstring('<div>Hello <p>World</p></div>')
    assert extract_text_array(dom) == [None, 'Hello ', None, 'World', None]

    # Test with separator tag
    dom = fromstring('<p>Hello<br>World</p>')
    assert extract_text_array(dom) == ['Hello', True, 'World']

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text_array(dom) == [None, 'Hello ', 'World', None]

    # Test with text and tail
    dom = fromstring('<p>Hello <b>World</b>!</p>')
    assert extract_text_array(dom) == ['Hello ', 'World', '!']

    # Test with squash_artifical_nl=False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = fromstring('<div><p>Hello</p></div>')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with empty dom
    dom = fromstring('<div></div>')
    assert extract_text_array(dom) == [None, None]

    # Test with only text
    dom = fromstring('<p>Hello World</p>')
    assert extract_text_array(dom) == ['Hello World']

    # Test with callable tag
    class CallableTag:
        def __init__(self):
            self.tag = lambda: None
    dom = CallableTag()
    assert extract_text_array(dom) == ''


# LLM-generated content at query #50
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockInlineTag:
        tag = 'span'
        text = 'Hello'
        def getchildren(self):
            return []

    dom = MockInlineTag()
    assert extract_text_array(dom) == ['Hello']

    # Test with block tag
    class MockBlockTag:
        tag = 'div'
        text = 'Hello'
        def getchildren(self):
            return []

    dom = MockBlockTag()
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test with separator tag
    class MockSeparatorTag:
        tag = 'br'
        text = None
        def getchildren(self):
            return []

    dom = MockSeparatorTag()
    assert extract_text_array(dom) == [True]

    # Test with nested tags
    class MockChildTag:
        tag = 'span'
        text = 'World'
        tail = '!'
        def getchildren(self):
            return []

    class MockParentTag:
        tag = 'div'
        text = 'Hello'
        def getchildren(self):
            return [MockChildTag()]

    dom = MockParentTag()
    assert extract_text_array(dom) == [None, 'Hello', 'World', '!', None]

    # Test with squash_artifical_nl=False
    class MockTag:
        tag = 'div'
        text = 'Hello'
        def getchildren(self):
            return []

    dom = MockTag()
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None]

    # Test with strip_artifical_nl=False
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with callable tag
    class MockCallableTag:
        def tag(self):
            return 'div'

    dom = MockCallableTag()
    assert extract_text_array(dom) == ''

    # Test with None text and tail
    class MockNoneTextTag:
        tag = 'div'
        text = None
        def getchildren(self):
            return []

    dom = MockNoneTextTag()
    assert extract_text_array(dom) == [None, None]

    # Test with multiple children
    class MockChild1:
        tag = 'span'
        text = 'Hello'
        tail = ' '
        def getchildren(self):
            return []

    class MockChild2:
        tag = 'span'
        text = 'World'
        tail = '!'
        def getchildren(self):
            return []

    class MockParent:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockChild1(), MockChild2()]

    dom = MockParent()
    assert extract_text_array(dom) == [None, 'Hello', ' ', 'World', '!', None]


# LLM-generated content at query #51
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    dom = Mock(tag='span', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom) == ['Hello']

    # Test with block tag
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test with separator tag
    dom = Mock(tag='br', text=None, getchildren=lambda: [])
    assert extract_text_array(dom) == [True]

    # Test with nested tags
    child = Mock(tag='span', text='World', tail='!', getchildren=lambda: [])
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [child])
    assert extract_text_array(dom) == [None, 'Hello', 'World', '!', None]

    # Test with squash_artifical_nl=False
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None]

    # Test with strip_artifical_nl=False
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with multiple children
    child1 = Mock(tag='span', text='Hello', tail=' ', getchildren=lambda: [])
    child2 = Mock(tag='span', text='World', tail='!', getchildren=lambda: [])
    dom = Mock(tag='div', text=None, getchildren=lambda: [child1, child2])
    assert extract_text_array(dom) == [None, 'Hello', ' ', 'World', '!', None]

    # Test with empty dom
    dom = Mock(tag='div', text=None, getchildren=lambda: [])
    assert extract_text_array(dom) == [None, None]

    # Test with callable tag
    dom = Mock(tag=lambda: 'div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom) == ''


# LLM-generated content at query #52
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline elements
    dom = fromstring("<div><span>Hello</span> <span>World</span></div>")
    assert extract_text(dom) == "Hello World"

    # Test with block elements
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with separators (br)
    dom = fromstring("<div>Hello<br>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with nested elements
    dom = fromstring("<div><p>Hello <strong>World</strong></p></div>")
    assert extract_text(dom) == "Hello World"

    # Test with whitespace squashing
    dom = fromstring("<div>Hello   World</div>")
    assert extract_text(dom) == "Hello World"

    # Test with leading/trailing whitespace
    dom = fromstring("<div>  Hello World  </div>")
    assert extract_text(dom) == "Hello World"

    # Test with custom symbols
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == "Hello|World"

    # Test with squash_space=False
    dom = fromstring("<div>  Hello   World  </div>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "

    # Test with mixed content
    dom = fromstring("<div><p>Hello<br>World</p><p>Foo</p></div>")
    assert extract_text(dom) == "Hello\nWorld\nFoo"

    # Test with empty content
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with only whitespace
    dom = fromstring("<div>   </div>")
    assert extract_text(dom) == ""

    # Test with script and style tags (should be ignored)
    dom = fromstring("<div><script>alert('test')</script>Hello</div>")
    assert extract_text(dom) == "Hello"

    # Test with complex nested structure
    dom = fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph 1<br>with break</p>
            <p>Paragraph 2</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    assert extract_text(dom) == "Title\nParagraph 1\nwith break\nParagraph 2\nItem 1\nItem 2"


# LLM-generated content at query #53
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test with simple inline element
    dom = MockElement('p', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test with block element
    dom = MockElement('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test with nested elements
    child = MockElement('span', text='World')
    dom = MockElement('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test with separator element (br)
    dom = MockElement('div', children=[
        MockElement('span', text='Hello'),
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with squash_space=True
    dom = MockElement('div', text='  Hello  ', children=[
        MockElement('span', text='  World  ')
    ])
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test with squash_space=False
    dom = MockElement('div', text='  Hello  ', children=[
        MockElement('span', text='  World  ')
    ])
    assert extract_text(dom, squash_space=False) == '  Hello  World  '

    # Test with custom block_symbol and sep_symbol
    dom = MockElement('div', children=[
        MockElement('p', text='Hello'),
        MockElement('p', text='World')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == 'Hello|World'

    # Test with callable tag
    dom = MockElement(lambda: 'div', text='Hello')
    assert extract_text(dom) == ''

    # Test with None text and tail
    dom = MockElement('div', text=None, tail=None)
    assert extract_text(dom) == ''

    # Test with complex nested structure
    dom = MockElement('div', children=[
        MockElement('p', text='First paragraph'),
        MockElement('ul', children=[
            MockElement('li', text='Item 1'),
            MockElement('li', text='Item 2')
        ]),
        MockElement('p', text='Second paragraph')
    ])
    assert extract_text(dom) == 'First paragraph\nItem 1\nItem 2\nSecond paragraph'


# LLM-generated content at query #54
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline tag
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag with text
    inline_elem = MockElement('span', text='Hello')
    assert extract_text_array(inline_elem) == ['Hello']

    # Test inline tag with children
    child_elem = MockElement('strong', text='World')
    inline_elem = MockElement('span', children=[child_elem])
    assert extract_text_array(inline_elem) == ['World']

    # Test block tag with text
    block_elem = MockElement('div', text='Hello')
    assert extract_text_array(block_elem) == [None, 'Hello', None]

    # Test block tag with children
    child_elem = MockElement('p', text='World')
    block_elem = MockElement('div', children=[child_elem])
    assert extract_text_array(block_elem) == [None, 'World', None, None]

    # Test separator tag
    sep_elem = MockElement('br')
    assert extract_text_array(sep_elem) == [True]

    # Test nested structure
    grandchild = MockElement('em', text='!')
    child = MockElement('strong', text='World', children=[grandchild])
    parent = MockElement('div', text='Hello', children=[child], tail='Tail')
    assert extract_text_array(parent) == [None, 'Hello', 'World', '!', 'Tail', None]

    # Test squash_artifical_nl=False
    block_elem = MockElement('div', text='Hello')
    assert extract_text_array(block_elem, squash_artifical_nl=False) == [None, 'Hello', None]

    # Test strip_artifical_nl=False
    block_elem = MockElement('div', text='Hello')
    assert extract_text_array(block_elem, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with callable tag
    callable_elem = MockElement(lambda: 'div')
    assert extract_text_array(callable_elem) == ''

    # Test empty element
    empty_elem = MockElement('span')
    assert extract_text_array(empty_elem) == []

    # Test with multiple children and tails
    child1 = MockElement('span', text='First')
    child2 = MockElement('span', text='Second', tail='Tail2')
    parent = MockElement('div', children=[child1, child2], tail='ParentTail')
    result = extract_text_array(parent)
    assert result == [None, 'First', 'Second', 'Tail2', 'ParentTail', None]


# LLM-generated content at query #55
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tags
    dom = fromstring("<div><p>Hello <strong>World</strong></p></div>")
    assert extract_text(dom) == "Hello World"

    # Test with block tags
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with separator tags
    dom = fromstring("<div>Hello<br>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with nested tags
    dom = fromstring("<div><p>Hello <span>World <em>!</em></span></p></div>")
    assert extract_text(dom) == "Hello World !"

    # Test with whitespace squashing
    dom = fromstring("<div><p>Hello   World</p></div>")
    assert extract_text(dom) == "Hello World"

    # Test with custom block and separator symbols
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == "Hello|World"

    # Test with no squashing
    dom = fromstring("<div><p>Hello   World</p></div>")
    assert extract_text(dom, squash_space=False) == "Hello   World"

    # Test with empty dom
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with only whitespace
    dom = fromstring("<div>   </div>")
    assert extract_text(dom) == ""

    # Test with mixed content
    dom = fromstring("<div><p>Hello</p>World<span>!</span></div>")
    assert extract_text(dom) == "Hello\nWorld!"


# LLM-generated content at query #56
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    # Test inline tag with text
    dom = MockElement('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test block tag with text
    dom = MockElement('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test nested tags
    child = MockElement('span', text='World')
    dom = MockElement('div', text='Hello ', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test with separator tag (br)
    dom = MockElement('br')
    assert extract_text(dom, sep_symbol='\n') == '\n'

    # Test with block symbol
    child = MockElement('span', text='World')
    dom = MockElement('div', text='Hello ', children=[child], tail='!')
    assert extract_text(dom, block_symbol='\n') == 'Hello World!\n'

    # Test with squash_space=True (default)
    dom = MockElement('div', text='  Hello  ', children=[], tail='  World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test with squash_space=False
    dom = MockElement('div', text='  Hello  ', children=[], tail='  World  ')
    assert extract_text(dom, squash_space=False) == '  Hello  \n  World  '

    # Test with multiple separators
    child1 = MockElement('br')
    child2 = MockElement('span', text='World')
    dom = MockElement('div', text='Hello', children=[child1, child2])
    assert extract_text(dom, sep_symbol='\n') == 'Hello\nWorld'

    # Test with empty text
    dom = MockElement('div', text='')
    assert extract_text(dom) == ''

    # Test with None text
    dom = MockElement('div', text=None)
    assert extract_text(dom) == ''

    # Test with callable tag (should return empty string)
    dom = MockElement(tag=lambda: 'div', text='Hello')
    assert extract_text(dom) == ''


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag
    inline_elem = MockElement('span', text='Hello')
    assert extract_text(inline_elem) == 'Hello'

    # Test block tag
    block_elem = MockElement('div', text='Hello')
    assert extract_text(block_elem) == 'Hello'

    # Test nested elements
    nested_elem = MockElement('div', children=[
        MockElement('p', text='First paragraph'),
        MockElement('p', text='Second paragraph')
    ])
    assert extract_text(nested_elem) == 'First paragraph\nSecond paragraph'

    # Test with separators
    sep_elem = MockElement('div', children=[
        MockElement('p', text='Line 1'),
        MockElement('br'),
        MockElement('p', text='Line 2')
    ])
    assert extract_text(sep_elem) == 'Line 1\nLine 2'

    # Test squash_space parameter
    space_elem = MockElement('div', text='  Extra  spaces  ')
    assert extract_text(space_elem, squash_space=True) == 'Extra spaces'
    assert extract_text(space_elem, squash_space=False) == '  Extra  spaces  '

    # Test block_symbol and sep_symbol parameters
    custom_elem = MockElement('div', children=[
        MockElement('p', text='Part 1'),
        MockElement('p', text='Part 2')
    ])
    assert extract_text(custom_elem, block_symbol='|', sep_symbol=';') == 'Part 1|Part 2'

    # Test with None values (artificial newlines)
    none_elem = MockElement('div', children=[
        MockElement('p', text='Start'),
        MockElement('div'),
        MockElement('p', text='End')
    ])
    assert extract_text(none_elem) == 'Start\nEnd'

    # Test empty element
    empty_elem = MockElement('div')
    assert extract_text(empty_elem) == ''

    # Test with tail text
    tail_elem = MockElement('div', children=[
        MockElement('span', text='Hello', tail='World')
    ])
    assert extract_text(tail_elem) == 'HelloWorld'

    # Test callable tag
    callable_elem = MockElement(lambda: 'div', text='Should be empty')
    assert extract_text(callable_elem) == ''


# LLM-generated content at query #2
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with inline tag
    dom = fromstring("<div><span>Hello</span> <span>World</span></div>")
    assert extract_text_array(dom) == ['Hello', ' ', 'World']

    # Test with block tag
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text_array(dom) == [None, 'Hello', None, 'World', None]

    # Test with separator tag
    dom = fromstring("<div>Hello<br>World</div>")
    assert extract_text_array(dom) == ['Hello', True, 'World', None]

    # Test with nested tags
    dom = fromstring("<div><p>Hello <span>World</span></p></div>")
    assert extract_text_array(dom) == [None, 'Hello ', 'World', None, None]

    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None, None]

    # Test with strip_artifical_nl=False
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None, 'World', None]

    # Test with both flags False
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Hello', None, None, 'World', None, None]

    # Test with empty dom
    dom = fromstring("<div></div>")
    assert extract_text_array(dom) == [None, None]

    # Test with text only
    dom = fromstring("<div>Hello World</div>")
    assert extract_text_array(dom) == ['Hello World', None]

    # Test with multiple separators
    dom = fromstring("<div>Hello<br><br>World</div>")
    assert extract_text_array(dom) == ['Hello', True, True, 'World', None]

    # Test with mixed content
    dom = fromstring("<div><p>Hello<br>World</p><span>!</span></div>")
    assert extract_text_array(dom) == [None, 'Hello', True, 'World', '!', None]


# LLM-generated content at query #3
#--------------------------

```python
def test_extract_text_array():
    # Test inline tag handling
    inline_dom = type('MockElement', (), {
        'tag': 'span',
        'text': 'Hello',
        'getchildren': lambda: [],
        'tail': None
    })()
    assert extract_text_array(inline_dom) == ['Hello']

    # Test separator tag handling
    separator_dom = type('MockElement', (), {
        'tag': 'br',
        'text': None,
        'getchildren': lambda: [],
        'tail': None
    })()
    assert extract_text_array(separator_dom) == [True]

    # Test block tag handling
    block_dom = type('MockElement', (), {
        'tag': 'div',
        'text': 'Block',
        'getchildren': lambda: [],
        'tail': None
    })()
    assert extract_text_array(block_dom) == ['Block', None, None]

    # Test nested elements
    parent = type('MockElement', (), {
        'tag': 'div',
        'text': 'Parent',
        'getchildren': lambda: [child],
        'tail': 'Tail'
    })()
    child = type('MockElement', (), {
        'tag': 'span',
        'text': 'Child',
        'getchildren': lambda: [],
        'tail': 'ChildTail'
    })()
    assert extract_text_array(parent) == ['Parent', 'Child', 'ChildTail', None]

    # Test squash_artifical_nl
    dom_with_nl = type('MockElement', (), {
        'tag': 'div',
        'text': None,
        'getchildren': lambda: [],
        'tail': None
    })()
    assert extract_text_array(dom_with_nl, squash_artifical_nl=True) == []

    # Test strip_artifical_nl
    dom_with_nl = type('MockElement', (), {
        'tag': 'div',
        'text': None,
        'getchildren': lambda: [],
        'tail': None
    })()
    assert extract_text_array(dom_with_nl, strip_artifical_nl=True) == []

    # Test callable tag
    callable_dom = type('MockElement', (), {
        'tag': lambda: 'div',
        'text': 'Callable',
        'getchildren': lambda: [],
        'tail': None
    })()
    assert extract_text_array(callable_dom) == ''

    # Test complex structure
    complex_parent = type('MockElement', (), {
        'tag': 'div',
        'text': 'Start',
        'getchildren': lambda: [complex_child1, complex_child2],
        'tail': 'End'
    })()
    complex_child1 = type('MockElement', (), {
        'tag': 'span',
        'text': 'Child1',
        'getchildren': lambda: [],
        'tail': 'Tail1'
    })()
    complex_child2 = type('MockElement', (), {
        'tag': 'br',
        'text': None,
        'getchildren': lambda: [],
        'tail': 'Tail2'
    })()
    result = extract_text_array(complex_parent)
    assert result == ['Start', 'Child1', 'Tail1', True, 'Tail2', None]


# LLM-generated content at query #4
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test block tag
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test nested tags
    dom = MockDom('div', children=[
        MockDom('p', text='Hello'),
        MockDom('p', text='World')
    ])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test separator tag
    dom = MockDom('div', children=[
        MockDom('span', text='Hello'),
        MockDom('br'),
        MockDom('span', text='World')
    ])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test squash_space
    dom = MockDom('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test block_symbol and sep_symbol
    dom = MockDom('div', children=[
        MockDom('p', text='Hello'),
        MockDom('p', text='World')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello|World'

    # Test with tail text
    dom = MockDom('div', children=[
        MockDom('span', text='Hello', tail='World')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test empty dom
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test callable tag
    dom = MockDom(lambda: None)
    assert extract_text(dom) == ''


# LLM-generated content at query #5
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    # Test inline tag
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test block tag
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test with children
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test with tail
    child = MockDom('span', text='World', tail='!')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World!'

    # Test with separator (br)
    dom = MockDom('br')
    assert extract_text(dom) == '\n'

    # Test with nested block tags
    inner = MockDom('span', text='Inner')
    middle = MockDom('div', children=[inner])
    outer = MockDom('div', children=[middle])
    assert extract_text(outer) == 'Inner'

    # Test with squash_space=True (default)
    dom = MockDom('div', text='  Hello  \n  World  ')
    assert extract_text(dom) == 'Hello World'

    # Test with squash_space=False
    dom = MockDom('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=False) == '  Hello  \n  World  '

    # Test with custom block_symbol and sep_symbol
    dom = MockDom('div', children=[MockDom('br')])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == ';'

    # Test with multiple separators
    dom = MockDom('div', children=[MockDom('br'), MockDom('br')])
    assert extract_text(dom) == '\n\n'

    # Test with mixed content
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('br')
    child3 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2, child3])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with callable tag
    class CallableTagDom:
        def __init__(self):
            self.tag = lambda: None
    dom = CallableTagDom()
    assert extract_text(dom) == ''


# LLM-generated content at query #6
#--------------------------

```python
def test_extract_text():
    from lxml import html

    # Test with simple inline elements
    dom = html.fromstring('<div><span>Hello</span> <strong>World</strong></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block elements
    dom = html.fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom) == 'First\nSecond'

    # Test with separators (br)
    dom = html.fromstring('<div>Line1<br/>Line2</div>')
    assert extract_text(dom) == 'Line1\nLine2'

    # Test with nested elements
    dom = html.fromstring('<div><p>Outer <span>Inner</span> text</p></div>')
    assert extract_text(dom) == 'Outer Inner text'

    # Test with whitespace squashing
    dom = html.fromstring('<div>  Multiple   spaces   here  </div>')
    assert extract_text(dom) == 'Multiple spaces here'

    # Test with custom symbols
    dom = html.fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol='-') == 'First|Second'

    # Test with no squashing
    dom = html.fromstring('<div>  Multiple   spaces   here  </div>')
    assert extract_text(dom, squash_space=False) == '  Multiple   spaces   here  '

    # Test with mixed content
    dom = html.fromstring('<div><p>Text with <br/> break</p><p>New paragraph</p></div>')
    assert extract_text(dom) == 'Text with \n break\nNew paragraph'

    # Test with empty elements
    dom = html.fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with only whitespace
    dom = html.fromstring('<div>   \n  \t  </div>')
    assert extract_text(dom) == ''

    # Test with script and style tags (should be ignored)
    dom = html.fromstring('<div><script>alert("test")</script>Visible text</div>')
    assert extract_text(dom) == 'Visible text'


# LLM-generated content at query #7
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test case 1: Simple inline tag
    dom = MockElement('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test case 2: Block tag with text
    dom = MockElement('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test case 3: Nested tags
    child = MockElement('span', text='World')
    dom = MockElement('div', text='Hello ', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test case 4: Separator tag (br)
    dom = MockElement('div', children=[MockElement('br')])
    assert extract_text(dom) == '\n'

    # Test case 5: Multiple block tags
    child1 = MockElement('p', text='First')
    child2 = MockElement('p', text='Second')
    dom = MockElement('div', children=[child1, child2])
    assert extract_text(dom) == 'First\nSecond'

    # Test case 6: With squash_space=False
    dom = MockElement('div', text='  Hello  ')
    assert extract_text(dom, squash_space=False) == '  Hello  '

    # Test case 7: With custom block_symbol and sep_symbol
    dom = MockElement('div', children=[MockElement('p', text='A'), MockElement('br'), MockElement('p', text='B')])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'A;B'

    # Test case 8: Empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''

    # Test case 9: Callable tag (should return empty string)
    dom = MockElement(lambda: 'div')
    assert extract_text(dom) == ''

    # Test case 10: Whitespace handling
    dom = MockElement('div', text='  Hello  \n  World  ')
    assert extract_text(dom) == 'Hello World'

    # Test case 11: Tail text
    child = MockElement('span', text='Hello', tail=' World')
    dom = MockElement('div', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test case 12: Multiple levels of nesting
    grandchild = MockElement('strong', text='nested')
    child = MockElement('span', text='deeply ', children=[grandchild])
    dom = MockElement('div', text='Very ', children=[child])
    assert extract_text(dom) == 'Very deeply nested'


# LLM-generated content at query #8
#--------------------------

```python
def test_extract_text():
    # Test basic inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    dom = MockDom('p', text='Hello ')
    child = MockDom('b', text='World')
    dom.children.append(child)
    assert extract_text(dom) == 'Hello World'

    # Test block tag with squash_space
    dom = MockDom('div', text='Hello ')
    child = MockDom('p', text='World')
    dom.children.append(child)
    assert extract_text(dom) == 'Hello\nWorld'

    # Test separator tag
    dom = MockDom('p', text='Hello')
    child = MockDom('br')
    dom.children.append(child)
    child2 = MockDom('span', text='World')
    dom.children.append(child2)
    assert extract_text(dom) == 'Hello\nWorld'

    # Test nested tags
    dom = MockDom('div')
    p1 = MockDom('p', text='First ')
    dom.children.append(p1)
    span = MockDom('span', text='line')
    p1.children.append(span)
    p2 = MockDom('p', text='Second')
    dom.children.append(p2)
    assert extract_text(dom) == 'First line\nSecond'

    # Test with custom symbols
    dom = MockDom('div', text='A')
    child = MockDom('p', text='B')
    dom.children.append(child)
    assert extract_text(dom, block_symbol='|', sep_symbol='!') == 'A|B'

    # Test whitespace squashing
    dom = MockDom('div', text='  Hello   ')
    child = MockDom('p', text='  World  ')
    dom.children.append(child)
    assert extract_text(dom) == 'Hello\nWorld'

    # Test tail text
    dom = MockDom('div')
    child = MockDom('p', text='Hello', tail=' World')
    dom.children.append(child)
    assert extract_text(dom) == 'Hello World'

    # Test empty dom
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test with squash_space=False
    dom = MockDom('div', text='Hello ')
    child = MockDom('p', text=' World ')
    dom.children.append(child)
    assert extract_text(dom, squash_space=False) == 'Hello \n World '

    # Test with None tag (callable)
    class CallableTagDom:
        def __init__(self):
            self.tag = lambda: None
            self.text = 'Hello'
            self.children = []

        def getchildren(self):
            return self.children

    dom = CallableTagDom()
    assert extract_text(dom) == ''


# LLM-generated content at query #9
#--------------------------

```python
def test_extract_text_array():
    # Test inline tag handling
    dom = Mock(tag='span', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom) == ['Hello']

    # Test block tag handling
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test separator tag handling
    dom = Mock(tag='br', text=None, getchildren=lambda: [])
    assert extract_text_array(dom) == [True]

    # Test nested tags
    child = Mock(tag='span', text='World', tail='!', getchildren=lambda: [])
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [child])
    assert extract_text_array(dom) == [None, 'Hello', 'World', '!', None]

    # Test squash_artifical_nl
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom, squash_artifical_nl=True) == ['Hello']

    # Test strip_artifical_nl
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom, strip_artifical_nl=True) == ['Hello']

    # Test with multiple children
    child1 = Mock(tag='span', text='World', tail=' ', getchildren=lambda: [])
    child2 = Mock(tag='span', text='!', tail=None, getchildren=lambda: [])
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [child1, child2])
    assert extract_text_array(dom) == [None, 'Hello', 'World', ' ', '!', None]

    # Test with callable tag
    dom = Mock(tag=lambda: 'div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom) == ''

    # Test empty dom
    dom = Mock(tag='div', text=None, getchildren=lambda: [])
    assert extract_text_array(dom) == [None, None]


# LLM-generated content at query #10
#--------------------------

```python
def test_extract_text():
    # Test basic inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    # Test 1: Simple inline tag
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 2: Block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 3: Nested tags
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test 4: Separator tag (br)
    dom = MockDom('br')
    assert extract_text(dom) == '\n'

    # Test 5: Block tag with separator symbol
    dom = MockDom('div', text='Hello')
    assert extract_text(dom, block_symbol='|') == 'Hello'

    # Test 6: Multiple children with text and tails
    child1 = MockDom('span', text='Hello', tail=' ')
    child2 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom) == 'Hello World'

    # Test 7: Squash space enabled
    dom = MockDom('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test 8: Squash space disabled
    dom = MockDom('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=False) == '  Hello  \n  World  '

    # Test 9: Mixed inline and block tags
    span = MockDom('span', text='inline')
    div = MockDom('div', text='block')
    dom = MockDom('body', children=[span, div])
    assert extract_text(dom) == 'inline\nblock'

    # Test 10: Custom symbols
    dom = MockDom('div', text='Hello')
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello'

    # Test 11: Empty text
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test 12: Text with only whitespace (squash_space=True)
    dom = MockDom('div', text='   \n  \t  ')
    assert extract_text(dom, squash_space=True) == ''

    # Test 13: Text with only whitespace (squash_space=False)
    dom = MockDom('div', text='   \n  \t  ')
    assert extract_text(dom, squash_space=False) == '   \n  \t  '

    # Test 14: Nested block tags
    inner = MockDom('div', text='inner')
    outer = MockDom('div', children=[inner])
    assert extract_text(outer) == 'inner'

    # Test 15: Multiple separators
    br1 = MockDom('br')
    br2 = MockDom('br')
    dom = MockDom('div', children=[br1, br2])
    assert extract_text(dom) == '\n\n'


# LLM-generated content at query #11
#--------------------------

```python
def test_extract_text():
    # Test basic inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self._tail = tail

        def getchildren(self):
            return self.children

        @property
        def tail(self):
            return self._tail

    # Test 1: Simple inline tag
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 2: Block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 3: Nested tags
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test 4: Separator tag (br)
    dom = MockDom('br')
    assert extract_text(dom) == '\n'

    # Test 5: Multiple children with separators
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('br')
    child3 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2, child3])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test 6: Tail text
    child = MockDom('span', text='Hello', tail='World')
    dom = MockDom('div', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test 7: Squash whitespace
    dom = MockDom('div', text='Hello  \n  World')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test 8: Custom symbols
    dom = MockDom('div', text='Hello')
    child = MockDom('div', text='World')
    dom.children = [child]
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello|World'

    # Test 9: Empty text
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test 10: Complex nested structure
    grandchild = MockDom('strong', text='nested')
    child = MockDom('p', text='Some ', children=[grandchild], tail=' text')
    dom = MockDom('div', text='Start', children=[child], tail=' End')
    assert extract_text(dom) == 'Start\nSome nested text\nEnd'

    # Test 11: Whitespace handling
    dom = MockDom('div', text='  \n  \t  Hello  \n  ')
    assert extract_text(dom, squash_space=True) == 'Hello'

    # Test 12: Multiple block elements
    child1 = MockDom('p', text='First')
    child2 = MockDom('p', text='Second')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom) == 'First\nSecond'

    # Test 13: Inline elements don't add newlines
    child1 = MockDom('span', text='First')
    child2 = MockDom('span', text='Second')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom) == 'First Second'

    # Test 14: Mixed inline and block elements
    child1 = MockDom('span', text='Inline')
    child2 = MockDom('div', text='Block')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom) == 'Inline\nBlock'

    # Test 15: Callable tag (should return empty string)
    class CallableTag:
        def tag(self):
            return 'div'
    dom = CallableTag()
    assert extract_text(dom) == ''


# LLM-generated content at query #12
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with simple inline tags
    dom = fromstring('<div><span>Hello</span> <strong>World</strong></div>')
    result = extract_text_array(dom)
    assert result == ['Hello', ' ', 'World']

    # Test with block tags and separators
    dom = fromstring('<div><p>First</p><br/><p>Second</p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'First', None, True, None, 'Second', None]

    # Test with nested tags
    dom = fromstring('<div><p>Outer <span>Inner</span> text</p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Outer ', 'Inner', ' text', None]

    # Test with text and tail
    dom = fromstring('<div><p>Text</p>Tail</div>')
    result = extract_text_array(dom)
    assert result == [None, 'Text', None, 'Tail', None]

    # Test with squash_artifical_nl=False
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'First', None, None, 'Second', None]

    # Test with strip_artifical_nl=False
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'First', None, 'Second', None]

    # Test with empty element
    dom = fromstring('<div></div>')
    result = extract_text_array(dom)
    assert result == [None, None]

    # Test with self-closing tag
    dom = fromstring('<div><br/></div>')
    result = extract_text_array(dom)
    assert result == [None, True, None]

    # Test with mixed content
    dom = fromstring('<div>Before<span>Inside</span>After<br/>End</div>')
    result = extract_text_array(dom)
    assert result == [None, 'Before', 'Inside', 'After', True, 'End', None]


# LLM-generated content at query #13
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with inline tag
    dom = fromstring('<div><span>Hello</span> <span>World</span></div>')
    assert extract_text_array(dom) == ['Hello', ' ', 'World']

    # Test with block tag
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom) == [None, 'Hello', None, None, 'World', None]

    # Test with separator tag
    dom = fromstring('<div>Hello<br/>World</div>')
    assert extract_text_array(dom) == ['Hello', True, 'World', None]

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text_array(dom) == [None, 'Hello ', 'World', None, None]

    # Test with text and tail
    dom = fromstring('<div><p>Hello</p> World</div>')
    assert extract_text_array(dom) == [None, 'Hello', None, ' World', None]

    # Test with squash_artifical_nl=False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None, None]

    # Test with strip_artifical_nl=False
    dom = fromstring('<div><p>Hello</p></div>')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with empty dom
    dom = fromstring('<div></div>')
    assert extract_text_array(dom) == [None, None]

    # Test with only text
    dom = fromstring('<div>Hello World</div>')
    assert extract_text_array(dom) == ['Hello World', None]

    # Test with multiple separators
    dom = fromstring('<div>Hello<br/><br/>World</div>')
    assert extract_text_array(dom) == ['Hello', True, True, 'World', None]


# LLM-generated content at query #14
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    inline_dom = MockDom('span', text='Hello', children=[], tail='World')
    assert extract_text_array(inline_dom) == ['Hello', 'World']

    # Test with block tag
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom) == [None, 'Hello', 'World', None]

    # Test with separator tag
    separator_dom = MockDom('br', text='Hello', children=[], tail='World')
    assert extract_text_array(separator_dom) == [True, 'Hello', 'World']

    # Test with nested tags
    nested_dom = MockDom('div', text='Hello', children=[
        MockDom('span', text='Nested', children=[], tail='Text')
    ], tail='World')
    assert extract_text_array(nested_dom) == [None, 'Hello', 'Nested', 'Text', None, 'World', None]

    # Test with squash_artifical_nl=False
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom, squash_artifical_nl=False) == [None, 'Hello', 'World', None]

    # Test with strip_artifical_nl=False
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom, strip_artifical_nl=False) == [None, 'Hello', 'World', None]

    # Test with both squash_artifical_nl=False and strip_artifical_nl=False
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Hello', 'World', None]

    # Test with callable tag
    callable_dom = MockDom(lambda: 'div', text='Hello', children=[], tail='World')
    assert extract_text_array(callable_dom) == ''

    # Test with None text and tail
    none_dom = MockDom('div', text=None, children=[], tail=None)
    assert extract_text_array(none_dom) == [None, None]

    # Test with empty children
    empty_children_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(empty_children_dom) == [None, 'Hello', 'World', None]


# LLM-generated content at query #15
#--------------------------

```python
def test_extract_text_array():
    # Test basic inline tag handling
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    # Test inline tag with text
    inline_elem = MockElement('span', text='Hello')
    assert extract_text_array(inline_elem) == ['Hello']

    # Test block tag with text
    block_elem = MockElement('div', text='Hello')
    assert extract_text_array(block_elem) == [None, 'Hello', None]

    # Test separator tag
    separator_elem = MockElement('br')
    assert extract_text_array(separator_elem) == [True]

    # Test nested elements
    parent = MockElement('div', children=[
        MockElement('p', text='Paragraph'),
        MockElement('span', text='Inline')
    ])
    assert extract_text_array(parent) == [None, 'Paragraph', None, 'Inline', None]

    # Test with tail text
    elem_with_tail = MockElement('div', children=[
        MockElement('span', text='Text', tail='Tail')
    ])
    assert extract_text_array(elem_with_tail) == [None, 'Text', 'Tail', None]

    # Test squash_artifical_nl=False
    elem = MockElement('div', children=[
        MockElement('div', text='First'),
        MockElement('div', text='Second')
    ])
    assert extract_text_array(elem, squash_artifical_nl=False) == [None, 'First', None, None, 'Second', None]

    # Test strip_artifical_nl=False
    elem = MockElement('div', children=[
        MockElement('span', text='Content')
    ])
    assert extract_text_array(elem, strip_artifical_nl=False) == [None, 'Content', None]

    # Test callable tag
    class CallableTag:
        def __call__(self):
            return 'callable'

    callable_elem = MockElement(CallableTag())
    assert extract_text_array(callable_elem) == ''

    # Test empty element
    empty_elem = MockElement('div')
    assert extract_text_array(empty_elem) == [None, None]


# LLM-generated content at query #16
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test 1: Simple inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 2: Block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 3: Nested tags
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test 4: Block tag with separator
    dom = MockDom('br')
    assert extract_text(dom, sep_symbol='\n') == '\n'

    # Test 5: Multiple blocks with squash_space
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test 6: With whitespace squashing
    dom = MockDom('div', text='  Hello   World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test 7: Custom block and separator symbols
    dom = MockDom('div', text='Hello')
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello'

    # Test 8: Complex nested structure
    grandchild = MockDom('strong', text='nested')
    child = MockDom('p', text='Some ', children=[grandchild], tail=' text')
    dom = MockDom('div', text='Start ', children=[child], tail=' End')
    assert extract_text(dom, squash_space=True) == 'Start Some nested text End'

    # Test 9: Empty dom
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test 10: Callable tag
    dom = MockDom(tag=lambda: None)
    assert extract_text(dom) == ''


# LLM-generated content at query #17
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    inline_dom = MockDom('span', text='Hello', children=[], tail='World')
    assert extract_text_array(inline_dom) == ['Hello', 'World']

    # Test with separator tag
    br_dom = MockDom('br', text='Line1', children=[], tail='Line2')
    assert extract_text_array(br_dom) == [True, 'Line1', True, 'Line2']

    # Test with block tag
    div_dom = MockDom('div', text='Block1', children=[], tail='Block2')
    assert extract_text_array(div_dom) == [None, 'Block1', None, 'Block2']

    # Test with nested tags
    nested_dom = MockDom('div', text='Outer',
                         children=[MockDom('span', text='Inner', children=[])],
                         tail='After')
    assert extract_text_array(nested_dom) == [None, 'Outer', 'Inner', None, 'After']

    # Test with squash_artifical_nl=False
    multi_nl = MockDom('div', text='A', children=[], tail='B')
    assert extract_text_array(multi_nl, squash_artifical_nl=False) == [None, 'A', None, 'B']

    # Test with strip_artifical_nl=False
    strip_test = MockDom('div', text='Start', children=[], tail='End')
    assert extract_text_array(strip_test, strip_artifical_nl=False) == [None, 'Start', None, 'End']

    # Test with both flags False
    both_false = MockDom('div', text='X', children=[], tail='Y')
    assert extract_text_array(both_false, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'X', None, 'Y']

    # Test with empty text
    empty_dom = MockDom('div', text=None, children=[], tail=None)
    assert extract_text_array(empty_dom) == [None, None]

    # Test with callable tag
    callable_dom = MockDom(lambda: 'callable', text='Text', children=[], tail='Tail')
    assert extract_text_array(callable_dom) == ''

    # Test with complex nested structure
    complex_dom = MockDom('div',
                          text='Header',
                          children=[
                              MockDom('p', text='Para1',
                                      children=[MockDom('strong', text='Bold')],
                                      tail='AfterPara'),
                              MockDom('br'),
                              MockDom('span', text='SpanText')
                          ],
                          tail='Footer')
    assert extract_text_array(complex_dom) == [
        None, 'Header', None, 'Para1', 'Bold', 'AfterPara', True, 'SpanText', None, 'Footer'
    ]


# LLM-generated content at query #18
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with inline tags
    dom = fromstring('<div><span>Hello</span> <strong>World</strong></div>')
    assert extract_text_array(dom) == ['Hello', ' ', 'World']

    # Test with block tags
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom) == [None, 'Hello', None, None, 'World', None]

    # Test with separator tags
    dom = fromstring('<div>Hello<br/>World</div>')
    assert extract_text_array(dom) == ['Hello', True, 'World', None]

    # Test with mixed content
    dom = fromstring('<div><p>Hello <span>there</span></p><p>World</p></div>')
    assert extract_text_array(dom) == [None, 'Hello ', 'there', None, None, 'World', None]

    # Test with nested block tags
    dom = fromstring('<div><div><p>Hello</p></div></div>')
    assert extract_text_array(dom) == [None, None, 'Hello', None, None, None]

    # Test with text and tail
    dom = fromstring('<div>Hello<p>World</p>!</div>')
    assert extract_text_array(dom) == ['Hello', None, 'World', None, '!', None]

    # Test with empty dom
    dom = fromstring('<div></div>')
    assert extract_text_array(dom) == [None, None]

    # Test with only text
    dom = fromstring('<div>Hello World</div>')
    assert extract_text_array(dom) == ['Hello World', None]

    # Test with squash_artifical_nl=False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with both flags False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with callable tag
    class CallableTag:
        def tag(self):
            return 'div'
    dom = CallableTag()
    assert extract_text_array(dom) == ''


# LLM-generated content at query #19
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    inline_dom = MockDom('span', text='Hello', children=[], tail='World')
    assert extract_text_array(inline_dom) == ['Hello', 'World']

    # Test with separator tag
    separator_dom = MockDom('br', text=None, children=[], tail=None)
    assert extract_text_array(separator_dom) == [True]

    # Test with block tag
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom) == [None, 'Hello', None, 'World']

    # Test with nested tags
    nested_dom = MockDom('div', text='Hello',
                         children=[MockDom('span', text='Nested', children=[], tail='Text')],
                         tail='World')
    assert extract_text_array(nested_dom) == [None, 'Hello', 'Nested', 'Text', None, 'World']

    # Test with squash_artifical_nl=False
    block_dom_no_squash = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom_no_squash, squash_artifical_nl=False) == [None, 'Hello', None, 'World']

    # Test with strip_artifical_nl=False
    block_dom_no_strip = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(block_dom_no_strip, strip_artifical_nl=False) == [None, 'Hello', None, 'World']

    # Test with callable tag
    callable_dom = MockDom(lambda: 'callable', text='Hello', children=[], tail='World')
    assert extract_text_array(callable_dom) == ''

    # Test with None text and tail
    none_dom = MockDom('div', text=None, children=[], tail=None)
    assert extract_text_array(none_dom) == [None, None]

    # Test with empty children
    empty_children_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(empty_children_dom) == [None, 'Hello', None, 'World']


# LLM-generated content at query #20
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tags
    dom = fromstring("<p>Hello <strong>World</strong></p>")
    assert extract_text(dom) == "Hello World"

    # Test with block tags
    dom = fromstring("<div>Hello</div><div>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with separator tags
    dom = fromstring("<p>Hello<br>World</p>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with nested tags
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with whitespace squashing
    dom = fromstring("<p>Hello   \n  World</p>")
    assert extract_text(dom) == "Hello World"

    # Test with custom symbols
    dom = fromstring("<div>Hello</div><div>World</div>")
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == "Hello|World"

    # Test with no squashing
    dom = fromstring("<p>Hello   \n  World</p>")
    assert extract_text(dom, squash_space=False) == "Hello   \n  World"

    # Test with empty dom
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with only whitespace
    dom = fromstring("<p>   \n  </p>")
    assert extract_text(dom) == ""

    # Test with mixed content
    dom = fromstring("<div><p>Hello</p><br><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with script and style tags (should be ignored)
    dom = fromstring("<div>Hello<script>alert('xss')</script>World</div>")
    assert extract_text(dom) == "HelloWorld"


# LLM-generated content at query #21
#--------------------------

```python
def test_extract_text_array():
    # Test case 1: Empty DOM element
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []

    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [None, None]

    # Test case 2: DOM element with text and no children
    class MockDomWithText:
        tag = 'p'
        text = 'Hello World'
        def getchildren(self):
            return []

    dom = MockDomWithText()
    result = extract_text_array(dom)
    assert result == [None, 'Hello World', None]

    # Test case 3: DOM element with inline tag
    class MockDomInline:
        tag = 'span'
        text = 'Inline Text'
        def getchildren(self):
            return []

    dom = MockDomInline()
    result = extract_text_array(dom)
    assert result == ['Inline Text']

    # Test case 4: DOM element with separator tag (br)
    class MockDomSeparator:
        tag = 'br'
        text = None
        def getchildren(self):
            return []

    dom = MockDomSeparator()
    result = extract_text_array(dom)
    assert result == [True]

    # Test case 5: DOM element with children
    class MockChild:
        tag = 'strong'
        text = 'Child Text'
        tail = ' Tail Text'
        def getchildren(self):
            return []

    class MockDomWithChildren:
        tag = 'div'
        text = 'Parent Text'
        def getchildren(self):
            return [MockChild()]

    dom = MockDomWithChildren()
    result = extract_text_array(dom)
    assert result == [None, 'Parent Text', 'Child Text', ' Tail Text', None]

    # Test case 6: DOM element with nested children
    class MockNestedChild:
        tag = 'em'
        text = 'Nested Child'
        tail = ' Nested Tail'
        def getchildren(self):
            return []

    class MockChildWithNested:
        tag = 'strong'
        text = 'Child Text'
        tail = ' Tail Text'
        def getchildren(self):
            return [MockNestedChild()]

    class MockDomWithNestedChildren:
        tag = 'div'
        text = 'Parent Text'
        def getchildren(self):
            return [MockChildWithNested()]

    dom = MockDomWithNestedChildren()
    result = extract_text_array(dom)
    assert result == [None, 'Parent Text', 'Child Text', 'Nested Child', ' Nested Tail', ' Tail Text', None]

    # Test case 7: DOM element with squash_artifical_nl=False
    class MockDomWithMultipleNone:
        tag = 'div'
        text = 'Text'
        def getchildren(self):
            return []

    dom = MockDomWithMultipleNone()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'Text', None]

    # Test case 8: DOM element with strip_artifical_nl=False
    class MockDomWithLeadingTrailingNone:
        tag = 'div'
        text = 'Text'
        def getchildren(self):
            return []

    dom = MockDomWithLeadingTrailingNone()
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'Text', None]

    # Test case 9: DOM element with callable tag
    class MockDomCallableTag:
        def tag(self):
            return 'div'
        text = None
        def getchildren(self):
            return []

    dom = MockDomCallableTag()
    result = extract_text_array(dom)
    assert result == ''

    # Test case 10: DOM element with mixed content
    class MockChildInline:
        tag = 'span'
        text = 'Inline'
        tail = ' Tail'
        def getchildren(self):
            return []

    class MockChildBlock:
        tag = 'div'
        text = 'Block'
        tail = ' Tail'
        def getchildren(self):
            return []

    class MockDomWithMixedChildren:
        tag = 'body'
        text = 'Start'
        def getchildren(self):
            return [MockChildInline(), MockChildBlock()]

    dom = MockDomWithMixedChildren()
    result = extract_text_array(dom)
    assert result == [None, 'Start', 'Inline', ' Tail', None, 'Block', ' Tail', None, None]


# LLM-generated content at query #22
#--------------------------

```python
def test_extract_text():
    # Test case 1: Simple inline element
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    dom = MockElement('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test case 2: Block element with squash_space=True
    dom = MockElement('div', text='Hello', children=[
        MockElement('span', text='World')
    ])
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test case 3: Block element with squash_space=False
    dom = MockElement('div', text='Hello', children=[
        MockElement('span', text='World')
    ])
    assert extract_text(dom, squash_space=False) == 'Hello\nWorld'

    # Test case 4: Separator element (br)
    dom = MockElement('div', text='Hello', children=[
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test case 5: Nested block elements
    dom = MockElement('div', text='Hello', children=[
        MockElement('div', text='World', children=[
            MockElement('span', text='!')
        ])
    ])
    assert extract_text(dom, squash_space=True) == 'Hello World !'

    # Test case 6: Empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''

    # Test case 7: Element with tail text
    dom = MockElement('div', text='Hello', children=[
        MockElement('span', text='World', tail='!')
    ])
    assert extract_text(dom, squash_space=True) == 'Hello World !'

    # Test case 8: Custom block and separator symbols
    dom = MockElement('div', text='Hello', children=[
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello;World'

    # Test case 9: Whitespace squashing
    dom = MockElement('div', text='Hello   World', children=[
        MockElement('span', text='  !  ')
    ])
    assert extract_text(dom, squash_space=True) == 'Hello World !'

    # Test case 10: Multiple separators
    dom = MockElement('div', text='Hello', children=[
        MockElement('br'),
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello\nWorld'


# LLM-generated content at query #23
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with inline tag
    dom = fromstring('<span>Hello <b>World</b></span>')
    assert extract_text_array(dom) == ['Hello ', 'World']

    # Test with block tag
    dom = fromstring('<div>Hello <p>World</p></div>')
    assert extract_text_array(dom) == [None, 'Hello ', None, 'World', None]

    # Test with separator tag
    dom = fromstring('<div>Hello<br>World</div>')
    assert extract_text_array(dom) == [None, 'Hello', True, 'World', None]

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text_array(dom) == [None, None, 'Hello ', 'World', None, None]

    # Test with text and tail
    dom = fromstring('<div>Hello <p>World</p> Tail</div>')
    assert extract_text_array(dom) == [None, 'Hello ', None, 'World', None, ' Tail', None]

    # Test with squash_artifical_nl=False
    dom = fromstring('<div>Hello</div><div>World</div>')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = fromstring('<div>Hello</div>')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with empty dom
    dom = fromstring('<div></div>')
    assert extract_text_array(dom) == [None, None]

    # Test with only text
    dom = fromstring('<div>Hello World</div>')
    assert extract_text_array(dom) == [None, 'Hello World', None]

    # Test with multiple separators
    dom = fromstring('<div>Hello<br><br>World</div>')
    assert extract_text_array(dom) == [None, 'Hello', True, True, 'World', None]


# LLM-generated content at query #24
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    # Test case 1: Simple inline element
    dom = MockElement('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test case 2: Block element with inline children
    dom = MockElement('div', children=[
        MockElement('span', text='Hello'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test case 3: Separator element (br)
    dom = MockElement('div', children=[
        MockElement('span', text='Hello'),
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test case 4: Nested block elements
    dom = MockElement('div', children=[
        MockElement('div', children=[
            MockElement('span', text='Hello')
        ]),
        MockElement('div', children=[
            MockElement('span', text='World')
        ])
    ])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test case 5: Text with whitespace
    dom = MockElement('div', children=[
        MockElement('span', text='  Hello  '),
        MockElement('span', text='  World  ')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test case 6: Text with custom separators
    dom = MockElement('div', children=[
        MockElement('span', text='Hello'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == 'Hello|World'

    # Test case 7: Text with squash_space=False
    dom = MockElement('div', children=[
        MockElement('span', text='  Hello  '),
        MockElement('span', text='  World  ')
    ])
    assert extract_text(dom, squash_space=False) == '  Hello  \n  World  '

    # Test case 8: Empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''

    # Test case 9: Mixed content with tail text
    dom = MockElement('div', children=[
        MockElement('span', text='Hello', tail=' '),
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test case 10: Callable tag (should return empty string)
    dom = MockElement(lambda: 'div')
    assert extract_text(dom) == ''


# LLM-generated content at query #25
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test separator tag
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test nested tags
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text_array(dom) == [None, 'Hello', 'World', None]

    # Test with tail text
    child = MockDom('span', text='World', tail='!')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text_array(dom) == [None, 'Hello', 'World', '!', None]

    # Test squash_artifical_nl=False
    dom = MockDom('div', text='Hello', children=[MockDom('div', text='World')])
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, 'World', None, None]

    # Test strip_artifical_nl=False
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with callable tag
    dom = MockDom(lambda: 'div', text='Hello')
    assert extract_text_array(dom) == ''

    # Test empty dom
    dom = MockDom('div')
    assert extract_text_array(dom) == [None, None]


# LLM-generated content at query #26
#--------------------------

```python
def test_extract_text():
    # Test basic inline tag handling
    dom = type('obj', (object,), {'tag': 'span', 'text': 'Hello', 'getchildren': lambda: [], 'tail': None})()
    assert extract_text(dom) == 'Hello'

    # Test block tag handling with newlines
    dom = type('obj', (object,), {'tag': 'div', 'text': 'Hello', 'getchildren': lambda: [], 'tail': None})()
    assert extract_text(dom) == '\nHello\n'

    # Test separator tag handling
    dom = type('obj', (object,), {'tag': 'br', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    assert extract_text(dom) == '\n'

    # Test nested tags
    child = type('obj', (object,), {'tag': 'span', 'text': 'World', 'getchildren': lambda: [], 'tail': '!'})()
    parent = type('obj', (object,), {'tag': 'div', 'text': 'Hello', 'getchildren': lambda: [child], 'tail': None})()
    assert extract_text(parent) == '\nHelloWorld!\n'

    # Test whitespace squashing
    dom = type('obj', (object,), {'tag': 'div', 'text': '  Hello  ', 'getchildren': lambda: [], 'tail': None})()
    assert extract_text(dom, squash_space=True) == '\nHello\n'

    # Test custom symbols
    dom = type('obj', (object,), {'tag': 'div', 'text': 'Hello', 'getchildren': lambda: [], 'tail': None})()
    assert extract_text(dom, block_symbol='|', sep_symbol='-') == '|Hello|'

    # Test callable tag (should return empty string)
    dom = type('obj', (object,), {'tag': lambda: 'div', 'text': 'Hello', 'getchildren': lambda: [], 'tail': None})()
    assert extract_text(dom) == ''


# LLM-generated content at query #27
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test simple text extraction
    dom = fromstring('<div>Hello World</div>')
    assert extract_text(dom) == 'Hello World'

    # Test with inline tags
    dom = fromstring('<div>Hello <strong>World</strong></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block-level tags
    dom = fromstring('<div>Hello<div>World</div></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with separators (br)
    dom = fromstring('<div>Hello<br/>World</div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with custom block and separator symbols
    dom = fromstring('<div>Hello<div>World</div></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello|World'

    # Test with whitespace squashing
    dom = fromstring('<div>Hello   World</div>')
    assert extract_text(dom) == 'Hello World'

    # Test with leading/trailing whitespace
    dom = fromstring('<div>  Hello World  </div>')
    assert extract_text(dom) == 'Hello World'

    # Test with nested tags
    dom = fromstring('<div>Hello <span>World <b>!</b></span></div>')
    assert extract_text(dom) == 'Hello World !'

    # Test with multiple block elements
    dom = fromstring('<div>Hello</div><div>World</div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with mixed content
    dom = fromstring('<div>Hello<br/> <span>World</span> <div>!</div></div>')
    assert extract_text(dom) == 'Hello\nWorld\n!'

    # Test with squash_space=False
    dom = fromstring('<div>  Hello   World  </div>')
    assert extract_text(dom, squash_space=False) == '  Hello   World  '

    # Test with empty content
    dom = fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with only whitespace
    dom = fromstring('<div>   </div>')
    assert extract_text(dom) == ''

    # Test with self-closing tags
    dom = fromstring('<div><img src="test.jpg"/>Text</div>')
    assert extract_text(dom) == 'Text'

    # Test with complex structure
    dom = fromstring('''
        <div>
            <h1>Title</h1>
            <p>Paragraph 1<br/>Line 2</p>
            <p>Paragraph 2</p>
        </div>
    ''')
    assert extract_text(dom) == 'Title\nParagraph 1\nLine 2\nParagraph 2'


# LLM-generated content at query #28
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    # Test inline tag
    inline_dom = MockDom('span', text='Hello')
    assert extract_text(inline_dom) == 'Hello'

    # Test block tag
    block_dom = MockDom('div', text='Hello')
    assert extract_text(block_dom) == 'Hello'

    # Test nested tags
    nested_dom = MockDom('div', children=[
        MockDom('p', text='First paragraph'),
        MockDom('p', text='Second paragraph')
    ])
    assert extract_text(nested_dom) == 'First paragraph\nSecond paragraph'

    # Test separator tag
    sep_dom = MockDom('div', children=[
        MockDom('span', text='Line 1'),
        MockDom('br'),
        MockDom('span', text='Line 2')
    ])
    assert extract_text(sep_dom) == 'Line 1\nLine 2'

    # Test whitespace squashing
    ws_dom = MockDom('div', text='  Hello   world  ')
    assert extract_text(ws_dom) == 'Hello world'

    # Test with custom symbols
    custom_dom = MockDom('div', children=[
        MockDom('p', text='Part 1'),
        MockDom('p', text='Part 2')
    ])
    assert extract_text(custom_dom, block_symbol='|', sep_symbol=';') == 'Part 1|Part 2'

    # Test with squash_space=False
    no_squash_dom = MockDom('div', text='  Hello   world  ')
    assert extract_text(no_squash_dom, squash_space=False) == '  Hello   world  '

    # Test empty dom
    empty_dom = MockDom('div')
    assert extract_text(empty_dom) == ''

    # Test mixed content
    mixed_dom = MockDom('div', children=[
        MockDom('span', text='Hello'),
        MockDom('br'),
        MockDom('p', text='World')
    ])
    assert extract_text(mixed_dom) == 'Hello\nWorld'


# LLM-generated content at query #29
#--------------------------

```python
def test_extract_text_array():
    # Test inline tag handling
    dom = Mock(tag='span', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom) == ['Hello']

    # Test separator handling
    dom = Mock(tag='br', text=None, getchildren=lambda: [])
    assert extract_text_array(dom) == [True]

    # Test block tag handling
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test nested tags
    child = Mock(tag='span', text='World', getchildren=lambda: [], tail='!')
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [child], tail=None)
    assert extract_text_array(dom) == [None, 'Hello', 'World', '!', None]

    # Test squash_artifical_nl
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom, squash_artifical_nl=True) == ['Hello']

    # Test strip_artifical_nl
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom, strip_artifical_nl=True) == ['Hello']

    # Test with callable tag
    dom = Mock(tag=lambda: 'div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom) == ''

    # Test with None text and tail
    dom = Mock(tag='div', text=None, getchildren=lambda: [], tail=None)
    assert extract_text_array(dom) == [None, None]

    # Test with multiple children
    child1 = Mock(tag='span', text='Hello', getchildren=lambda: [], tail=' ')
    child2 = Mock(tag='span', text='World', getchildren=lambda: [], tail='!')
    dom = Mock(tag='div', text=None, getchildren=lambda: [child1, child2], tail=None)
    assert extract_text_array(dom) == [None, 'Hello', ' ', 'World', '!', None]


# LLM-generated content at query #30
#--------------------------

```python
def test_extract_text():
    # Test basic inline tag
    dom = Mock(tag='span', text='Hello', getchildren=lambda: [])
    assert extract_text(dom) == 'Hello'

    # Test block tag with squash_space
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text(dom) == 'Hello'

    # Test nested tags
    child = Mock(tag='span', text='World', getchildren=lambda: [], tail='!')
    dom = Mock(tag='div', text='Hello ', getchildren=lambda: [child], tail=None)
    assert extract_text(dom) == 'Hello World!'

    # Test separator tag
    child = Mock(tag='br', text=None, getchildren=lambda: [], tail='After')
    dom = Mock(tag='div', text='Before', getchildren=lambda: [child], tail=None)
    assert extract_text(dom) == 'Before\nAfter'

    # Test squash_space with multiple whitespace
    dom = Mock(tag='div', text='Hello   World', getchildren=lambda: [])
    assert extract_text(dom) == 'Hello World'

    # Test strip_artifical_nl
    child = Mock(tag='div', text='Inner', getchildren=lambda: [], tail='After')
    dom = Mock(tag='div', text='Before', getchildren=lambda: [child], tail=None)
    assert extract_text(dom) == 'Before\nInner\nAfter'

    # Test custom block_symbol and sep_symbol
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello'

    # Test with callable tag (should return empty string)
    dom = Mock(tag=lambda: 'div', text='Hello', getchildren=lambda: [])
    assert extract_text(dom) == ''


# LLM-generated content at query #31
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    # Test inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test inline tag with children
    child = MockDom('strong', text='World')
    dom = MockDom('span', children=[child])
    assert extract_text_array(dom) == ['World']

    # Test block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test block tag with children
    child = MockDom('p', text='World')
    dom = MockDom('div', children=[child])
    assert extract_text_array(dom) == [None, 'World', None, None]

    # Test separator tag
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test nested tags
    grandchild = MockDom('em', text='!')
    child = MockDom('strong', children=[grandchild], tail=' ')
    dom = MockDom('div', text='Hello', children=[child], tail='World')
    assert extract_text_array(dom) == [None, 'Hello', '!', ' ', 'World', None]

    # Test squash_artifical_nl=False
    dom = MockDom('div', children=[MockDom('div'), MockDom('div')])
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, None, None, None]

    # Test strip_artifical_nl=False
    dom = MockDom('div', children=[MockDom('div')])
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, None, None]

    # Test with callable tag
    dom = MockDom(lambda: 'div')
    assert extract_text_array(dom) == ''


# LLM-generated content at query #32
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tags
    dom = fromstring("<div><p>Hello <strong>world</strong></p></div>")
    assert extract_text(dom) == "Hello world"

    # Test with block tags
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with separators (br)
    dom = fromstring("<div>Hello<br>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with nested tags
    dom = fromstring("<div><p>Hello <span>world <em>!</em></span></p></div>")
    assert extract_text(dom) == "Hello world !"

    # Test with whitespace squashing
    dom = fromstring("<div><p>Hello   world</p></div>")
    assert extract_text(dom) == "Hello world"

    # Test with custom block and separator symbols
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == "Hello|World"

    # Test with squash_space=False
    dom = fromstring("<div><p>Hello   world</p></div>")
    assert extract_text(dom, squash_space=False) == "Hello   world"

    # Test with leading/trailing whitespace
    dom = fromstring("<div>  <p>Hello</p>  </div>")
    assert extract_text(dom) == "Hello"

    # Test with mixed content
    dom = fromstring("<div><p>Hello</p><br><p>World</p></div>")
    assert extract_text(dom) == "Hello\n\nWorld"

    # Test with empty content
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with only whitespace
    dom = fromstring("<div>   \n  \t  </div>")
    assert extract_text(dom) == ""

    # Test with script and style tags (should be ignored)
    dom = fromstring("<div><p>Hello</p><script>alert('xss')</script><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"


# LLM-generated content at query #33
#--------------------------

```python
def test_extract_text():
    # Test case 1: Simple inline element
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    dom = MockElement('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test case 2: Block element with squash_space=True
    dom = MockElement('div', text='Hello', children=[
        MockElement('p', text='World')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test case 3: Block element with squash_space=False
    dom = MockElement('div', text='Hello', children=[
        MockElement('p', text='World')
    ])
    assert extract_text(dom, squash_space=False) == '\nHello\nWorld\n'

    # Test case 4: Separator element (br)
    dom = MockElement('div', children=[
        MockElement('br'),
        MockElement('span', text='Hello')
    ])
    assert extract_text(dom) == '\nHello'

    # Test case 5: Nested elements with whitespace
    dom = MockElement('div', text='  Hello  ', children=[
        MockElement('span', text='  World  ')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test case 6: Custom block and separator symbols
    dom = MockElement('div', children=[
        MockElement('p', text='Hello'),
        MockElement('p', text='World')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello|World'

    # Test case 7: Empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''

    # Test case 8: Element with tail text
    dom = MockElement('div', children=[
        MockElement('span', text='Hello', tail='World')
    ])
    assert extract_text(dom) == 'HelloWorld'

    # Test case 9: Multiple separators
    dom = MockElement('div', children=[
        MockElement('br'),
        MockElement('br'),
        MockElement('span', text='Hello')
    ])
    assert extract_text(dom) == '\n\nHello'

    # Test case 10: Preformatted text (squash_space=False)
    dom = MockElement('pre', text='Hello  World')
    assert extract_text(dom, squash_space=False) == 'Hello  World'


# LLM-generated content at query #34
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test 1: Simple inline tag
    dom = MockElement('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 2: Block tag with squash_space=True
    dom = MockElement('div', text='Hello', children=[
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test 3: Block tag with squash_space=False
    dom = MockElement('div', text='Hello', children=[
        MockElement('span', text='World')
    ])
    assert extract_text(dom, squash_space=False) == '\nHello\nWorld\n'

    # Test 4: Separator tag (br)
    dom = MockElement('div', children=[
        MockElement('span', text='Hello'),
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test 5: Nested block tags
    dom = MockElement('div', children=[
        MockElement('p', text='First paragraph'),
        MockElement('p', text='Second paragraph')
    ])
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'

    # Test 6: Whitespace handling
    dom = MockElement('div', text='  Hello  ', children=[
        MockElement('span', text='  World  ')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test 7: Custom block and separator symbols
    dom = MockElement('div', children=[
        MockElement('span', text='Hello'),
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello;World'

    # Test 8: Empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''

    # Test 9: Element with only whitespace
    dom = MockElement('div', text='   \n  \t  ')
    assert extract_text(dom) == ''

    # Test 10: Mixed content with tail text
    dom = MockElement('div', children=[
        MockElement('span', text='Hello', tail='World')
    ])
    assert extract_text(dom) == 'Hello World'


# LLM-generated content at query #35
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with simple inline tag
    dom = fromstring("<p>Hello <b>World</b></p>")
    assert extract_text_array(dom) == ['Hello ', 'World']

    # Test with block tag
    dom = fromstring("<div>Hello <p>World</p></div>")
    assert extract_text_array(dom) == [None, 'Hello ', None, 'World', None]

    # Test with separator tag
    dom = fromstring("<p>Hello<br/>World</p>")
    assert extract_text_array(dom) == ['Hello ', True, 'World']

    # Test with nested tags
    dom = fromstring("<div><p>Hello <span>World</span></p></div>")
    assert extract_text_array(dom) == [None, 'Hello ', 'World', None]

    # Test with text and tail
    dom = fromstring("<p>Hello <b>World</b>!</p>")
    assert extract_text_array(dom) == ['Hello ', 'World', '!']

    # Test with squash_artifical_nl=False
    dom = fromstring("<div>Hello</div><div>World</div>")
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = fromstring("<div>Hello</div>")
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with empty dom
    dom = fromstring("<div></div>")
    assert extract_text_array(dom) == [None, None]

    # Test with only text
    dom = fromstring("Hello World")
    assert extract_text_array(dom) == ['Hello World']

    # Test with multiple separators
    dom = fromstring("<p>Hello<br/><br/>World</p>")
    assert extract_text_array(dom) == ['Hello ', True, 'World']


# LLM-generated content at query #36
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    # Test inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test separator tag
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test nested tags
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child], tail='!')
    assert extract_text_array(dom) == [None, 'Hello', 'World', '!', None]

    # Test with squash_artifical_nl=False
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None]

    # Test with strip_artifical_nl=False
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with both flags False
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with callable tag
    dom = MockDom(lambda: 'div', text='Hello')
    assert extract_text_array(dom) == ''

    # Test empty dom
    dom = MockDom('div')
    assert extract_text_array(dom) == [None, None]

    # Test with multiple children
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text_array(dom) == [None, 'Hello', 'World', None]


# LLM-generated content at query #37
#--------------------------

```python
def test_extract_text():
    from lxml import html

    # Test with simple inline tag
    dom = html.fromstring('<div><span>Hello</span> <span>World</span></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block elements
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with separator elements
    dom = html.fromstring('<div>Hello<br/>World</div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with nested tags
    dom = html.fromstring('<div><p>Hello <strong>World</strong></p></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with whitespace squashing
    dom = html.fromstring('<div>Hello   World</div>')
    assert extract_text(dom) == 'Hello World'

    # Test with custom symbols
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == 'Hello|World'

    # Test with squash_space=False
    dom = html.fromstring('<div>  Hello  World  </div>')
    assert extract_text(dom, squash_space=False) == '  Hello  World  '

    # Test with mixed content
    dom = html.fromstring('<div><p>Hello<br/>World</p><p>Foo</p></div>')
    assert extract_text(dom) == 'Hello\nWorld\nFoo'

    # Test with empty content
    dom = html.fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with only whitespace
    dom = html.fromstring('<div>   \n  \t  </div>')
    assert extract_text(dom) == ''

    # Test with preformatted content (should not squash whitespace)
    dom = html.fromstring('<pre>Hello   World</pre>')
    assert extract_text(dom, squash_space=False) == 'Hello   World'


# LLM-generated content at query #38
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tags
    html = fromstring('<div><span>Hello</span> <strong>World</strong></div>')
    assert extract_text(html) == 'Hello World'

    # Test with block tags
    html = fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(html) == 'First\nSecond'

    # Test with separator tags
    html = fromstring('<div>Line1<br/>Line2</div>')
    assert extract_text(html) == 'Line1\nLine2'

    # Test with nested tags
    html = fromstring('<div><p>Outer <span>Inner</span> text</p></div>')
    assert extract_text(html) == 'Outer Inner text'

    # Test with whitespace squashing
    html = fromstring('<div>  Multiple   spaces  </div>')
    assert extract_text(html) == 'Multiple spaces'

    # Test with custom symbols
    html = fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(html, block_symbol='|', sep_symbol='-') == 'First|Second'

    # Test with mixed content
    html = fromstring('<div><p>Text <br/> with <strong>formatting</strong></p></div>')
    assert extract_text(html) == 'Text \n with formatting'

    # Test with empty content
    html = fromstring('<div></div>')
    assert extract_text(html) == ''

    # Test with only whitespace
    html = fromstring('<div>   \n  \t  </div>')
    assert extract_text(html) == ''

    # Test with squash_space=False
    html = fromstring('<div>  Multiple   spaces  </div>')
    assert extract_text(html, squash_space=False) == '  Multiple   spaces  '


# LLM-generated content at query #39
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    inline_dom = MockDom('span', text='Hello', children=[], tail='World')
    assert extract_text_array(inline_dom) == ['Hello', 'World']

    # Test with separator tag
    separator_dom = MockDom('br', text='Line1', children=[], tail='Line2')
    assert extract_text_array(separator_dom) == [True, 'Line1', 'Line2', True]

    # Test with block tag
    block_dom = MockDom('div', text='Block1', children=[], tail='Block2')
    assert extract_text_array(block_dom) == [None, 'Block1', 'Block2', None]

    # Test with nested tags
    nested_dom = MockDom('div', text='Outer', children=[
        MockDom('span', text='Inner', children=[], tail='Tail')
    ], tail='End')
    assert extract_text_array(nested_dom) == [None, 'Outer', 'Inner', 'Tail', None, 'End', None]

    # Test with squash_artifical_nl=False
    block_dom_no_squash = MockDom('div', text='Block1', children=[], tail='Block2')
    assert extract_text_array(block_dom_no_squash, squash_artifical_nl=False) == [None, 'Block1', 'Block2', None]

    # Test with strip_artifical_nl=False
    block_dom_no_strip = MockDom('div', text='Block1', children=[], tail='Block2')
    assert extract_text_array(block_dom_no_strip, strip_artifical_nl=False) == [None, 'Block1', 'Block2', None]

    # Test with both flags False
    block_dom_no_flags = MockDom('div', text='Block1', children=[], tail='Block2')
    assert extract_text_array(block_dom_no_flags, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Block1', 'Block2', None]

    # Test with callable tag
    callable_dom = MockDom(lambda: 'callable', text='Text', children=[], tail='Tail')
    assert extract_text_array(callable_dom) == ''

    # Test with None text and tail
    none_dom = MockDom('span', text=None, children=[], tail=None)
    assert extract_text_array(none_dom) == []

    # Test with empty children
    empty_children_dom = MockDom('div', text='Text', children=[], tail='Tail')
    assert extract_text_array(empty_children_dom) == [None, 'Text', 'Tail', None]


# LLM-generated content at query #40
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test block tag
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test nested tags
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello ', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test with block_symbol
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello ', children=[child])
    assert extract_text(dom, block_symbol='|') == 'Hello World|'

    # Test with sep_symbol
    child = MockDom('br')
    dom = MockDom('div', text='Hello', children=[child], tail='World')
    assert extract_text(dom, sep_symbol='|') == 'Hello|World'

    # Test squash_space
    dom = MockDom('div', text='  Hello   World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test with multiple children
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom) == 'HelloWorld'

    # Test with tail text
    child = MockDom('span', text='Hello')
    dom = MockDom('div', children=[child], tail='World')
    assert extract_text(dom) == 'HelloWorld'

    # Test with None text
    dom = MockDom('div', text=None)
    assert extract_text(dom) == ''

    # Test with callable tag
    dom = MockDom(tag=lambda: 'div', text='Hello')
    assert extract_text(dom) == ''

    # Test with separator tag
    child = MockDom('br')
    dom = MockDom('div', text='Hello', children=[child], tail='World')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with multiple separators
    child1 = MockDom('br')
    child2 = MockDom('br')
    dom = MockDom('div', text='Hello', children=[child1, child2], tail='World')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with whitespace squashing
    dom = MockDom('div', text='Hello  \n  World')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test with strip_artifical_nl
    child = MockDom('div', text='Hello')
    dom = MockDom('div', children=[child])
    assert extract_text(dom, squash_space=True) == 'Hello'

    # Test with complex nesting
    grandchild = MockDom('span', text='!')
    child = MockDom('div', text='World', children=[grandchild])
    dom = MockDom('div', text='Hello ', children=[child])
    assert extract_text(dom) == 'Hello World!'

    # Test with mixed inline and block tags
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('div', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with empty dom
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test with only whitespace
    dom = MockDom('div', text='   \n   \t  ')
    assert extract_text(dom, squash_space=True) == ''

    # Test with multiple block tags
    child1 = MockDom('div', text='Hello')
    child2 = MockDom('div', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with mixed content
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('br')
    child3 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2, child3])
    assert extract_text(dom) == 'Hello\nWorld'


# LLM-generated content at query #41
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test nested tags
    dom = MockDom('div', children=[
        MockDom('p', text='First paragraph'),
        MockDom('p', text='Second paragraph')
    ])
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'

    # Test with separator tag (br)
    dom = MockDom('div', children=[
        MockDom('span', text='First'),
        MockDom('br'),
        MockDom('span', text='Second')
    ])
    assert extract_text(dom) == 'First\nSecond'

    # Test with whitespace squashing
    dom = MockDom('div', text='  Hello   world  ')
    assert extract_text(dom, squash_space=True) == 'Hello world'

    # Test with custom symbols
    dom = MockDom('div', children=[
        MockDom('p', text='First'),
        MockDom('p', text='Second')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'First|Second'

    # Test with None text
    dom = MockDom('div', text=None)
    assert extract_text(dom) == ''

    # Test with mixed content
    dom = MockDom('div', children=[
        MockDom('span', text='Hello'),
        MockDom('br'),
        MockDom('div', text='World')
    ])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with tail text
    dom = MockDom('div', children=[
        MockDom('span', text='Hello', tail=' world')
    ])
    assert extract_text(dom) == 'Hello world'

    # Test with multiple whitespace
    dom = MockDom('div', text='Hello\n\tworld')
    assert extract_text(dom, squash_space=True) == 'Hello world'

    # Test with empty dom
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test with callable tag
    dom = MockDom(tag=lambda: None)
    assert extract_text(dom) == ''


# LLM-generated content at query #42
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with inline tag
    dom = fromstring('<div><span>Hello</span> <span>World</span></div>')
    result = extract_text_array(dom)
    assert result == ['Hello', ' ', 'World']

    # Test with block tag
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello', None, None, 'World', None]

    # Test with separator tag
    dom = fromstring('<div>Hello<br/>World</div>')
    result = extract_text_array(dom)
    assert result == ['Hello', True, 'World']

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World</span></p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello ', 'World', None]

    # Test with squash_artifical_nl=True
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == [None, 'Hello', 'World', None]

    # Test with strip_artifical_nl=True
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['Hello', None, 'World']

    # Test with both squash and strip
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['Hello', 'World']

    # Test with empty dom
    dom = fromstring('<div></div>')
    result = extract_text_array(dom)
    assert result == [None, None]

    # Test with text only
    dom = fromstring('<div>Hello World</div>')
    result = extract_text_array(dom)
    assert result == ['Hello World']

    # Test with callable tag
    class CallableTag:
        def __init__(self):
            self.tag = lambda: None
    dom = CallableTag()
    result = extract_text_array(dom)
    assert result == ''


# LLM-generated content at query #43
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test inline tag with children
    child = MockDom('strong', text='World')
    dom = MockDom('span', children=[child])
    assert extract_text_array(dom) == ['World']

    # Test block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test separator tag
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test nested tags
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('strong', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text_array(dom) == [None, 'Hello', 'World', None]

    # Test with tail text
    child = MockDom('span', text='Hello', tail='World')
    dom = MockDom('div', children=[child])
    assert extract_text_array(dom) == [None, 'Hello', 'World', None]

    # Test squash_artifical_nl=False
    dom = MockDom('div', text='Hello', children=[MockDom('div', text='World')])
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, 'World', None]

    # Test strip_artifical_nl=False
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test callable tag
    dom = MockDom(lambda: 'div')
    assert extract_text_array(dom) == ''

    # Test None text and tail
    dom = MockDom('div', text=None, tail=None)
    assert extract_text_array(dom) == [None, None]


# LLM-generated content at query #44
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self._tail = tail

        def getchildren(self):
            return self.children

        @property
        def tail(self):
            return self._tail

    # Test 1: Simple inline tag
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 2: Block tag with text
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 3: Nested tags
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test 4: Block tags with separators
    dom = MockDom('div', children=[
        MockDom('p', text='First paragraph'),
        MockDom('p', text='Second paragraph')
    ])
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'

    # Test 5: BR tag separator
    dom = MockDom('div', children=[
        MockDom('span', text='First'),
        MockDom('br'),
        MockDom('span', text='Second')
    ])
    assert extract_text(dom) == 'First\nSecond'

    # Test 6: Whitespace squashing
    dom = MockDom('div', text='  Hello   \n  World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test 7: Custom block and separator symbols
    dom = MockDom('div', children=[
        MockDom('p', text='First'),
        MockDom('p', text='Second')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'First|Second'

    # Test 8: Tail text
    child = MockDom('span', text='World', tail='!')
    dom = MockDom('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World!'

    # Test 9: Empty dom
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test 10: Multiple whitespace and newlines
    dom = MockDom('div', text='  \n  Hello  \n  World  \n  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test 11: Inline tag with block child (should not happen in valid HTML)
    dom = MockDom('span', children=[
        MockDom('div', text='Block inside inline')
    ])
    assert extract_text(dom) == 'Block inside inline'

    # Test 12: Mixed content with separators
    dom = MockDom('div', children=[
        MockDom('span', text='First'),
        MockDom('br'),
        MockDom('span', text='Second'),
        MockDom('p', text='Third')
    ])
    assert extract_text(dom) == 'First\nSecond\nThird'

    # Test 13: No squash space
    dom = MockDom('div', text='  Hello   \n  World  ')
    assert extract_text(dom, squash_space=False) == '  Hello   \n  World  '

    # Test 14: Callable tag (should return empty string)
    dom = MockDom(lambda: None)
    assert extract_text(dom) == ''


# LLM-generated content at query #45
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    # Test inline tag
    inline_elem = MockElement('span', text='Hello')
    assert extract_text(inline_elem) == 'Hello'

    # Test block tag
    block_elem = MockElement('div', text='Hello')
    assert extract_text(block_elem) == 'Hello'

    # Test nested elements
    nested_elem = MockElement('div', children=[
        MockElement('p', text='Hello'),
        MockElement('p', text='World')
    ])
    assert extract_text(nested_elem) == 'Hello\nWorld'

    # Test separator tag (br)
    br_elem = MockElement('div', children=[
        MockElement('span', text='Hello'),
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(br_elem) == 'Hello\nWorld'

    # Test squash_space
    space_elem = MockElement('div', text='  Hello  ', children=[
        MockElement('span', text='  World  ')
    ])
    assert extract_text(space_elem, squash_space=True) == 'Hello World'

    # Test custom block and separator symbols
    custom_elem = MockElement('div', children=[
        MockElement('p', text='Hello'),
        MockElement('p', text='World')
    ])
    assert extract_text(custom_elem, block_symbol='|', sep_symbol=';') == 'Hello|World'

    # Test empty element
    empty_elem = MockElement('div')
    assert extract_text(empty_elem) == ''

    # Test tail text
    tail_elem = MockElement('div', children=[
        MockElement('span', text='Hello', tail='World')
    ])
    assert extract_text(tail_elem) == 'HelloWorld'

    # Test callable tag (should return empty string)
    callable_elem = MockElement(lambda: 'div', text='Hello')
    assert extract_text(callable_elem) == ''


# LLM-generated content at query #46
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self.children = children or []

        def getchildren(self):
            return self.children

    dom = MockDom('p', children=[MockDom('b', text='Hello')])
    assert extract_text(dom) == 'Hello'

    # Test with block tag and squash_space=True
    dom = MockDom('div', children=[
        MockDom('p', text='First paragraph'),
        MockDom('p', text='Second paragraph')
    ])
    assert extract_text(dom, squash_space=True) == 'First paragraph\nSecond paragraph'

    # Test with block tag and squash_space=False
    dom = MockDom('div', children=[
        MockDom('p', text='First paragraph'),
        MockDom('p', text='Second paragraph')
    ])
    assert extract_text(dom, squash_space=False) == '\nFirst paragraph\n\nSecond paragraph\n'

    # Test with separator tag (br)
    dom = MockDom('div', children=[
        MockDom('span', text='Line 1'),
        MockDom('br'),
        MockDom('span', text='Line 2')
    ])
    assert extract_text(dom) == 'Line 1\nLine 2'

    # Test with custom block and separator symbols
    dom = MockDom('div', children=[
        MockDom('p', text='First'),
        MockDom('p', text='Second')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'First|Second'

    # Test with nested tags
    dom = MockDom('div', children=[
        MockDom('p', children=[
            MockDom('b', text='Bold '),
            MockDom('i', text='Italic')
        ])
    ])
    assert extract_text(dom) == 'Bold Italic'

    # Test with whitespace handling
    dom = MockDom('div', children=[
        MockDom('p', text='  Extra   spaces  '),
        MockDom('p', text='\tTabs\nand\nnewlines')
    ])
    assert extract_text(dom, squash_space=True) == 'Extra spaces\nTabs and newlines'

    # Test with tail text
    dom = MockDom('div', children=[
        MockDom('span', text='First', tail=' tail text'),
        MockDom('span', text='Second')
    ])
    assert extract_text(dom) == 'First tail text Second'

    # Test with empty dom
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test with callable tag (should return empty string)
    class CallableTagDom:
        def __init__(self):
            self.tag = lambda: None
            self.text = None
            self.children = []

        def getchildren(self):
            return self.children

    dom = CallableTagDom()
    assert extract_text(dom) == ''


# LLM-generated content at query #47
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    inline_dom = MockDom('span', text='Hello', tail='World')
    assert extract_text_array(inline_dom) == ['Hello', 'World']

    # Test with separator tag
    separator_dom = MockDom('br')
    assert extract_text_array(separator_dom) == [True]

    # Test with block tag
    block_dom = MockDom('div', text='Hello', tail='World')
    assert extract_text_array(block_dom) == [None, 'Hello', 'World', None]

    # Test with nested tags
    nested_dom = MockDom('div', children=[
        MockDom('p', text='Paragraph', children=[
            MockDom('span', text='Span text')
        ]),
        MockDom('br'),
        MockDom('a', text='Link')
    ])
    assert extract_text_array(nested_dom) == [None, 'Paragraph', 'Span text', None, True, 'Link', None]

    # Test with squash_artifical_nl=False
    dom_with_nl = MockDom('div', children=[
        MockDom('p', text='Line1'),
        MockDom('p', text='Line2')
    ])
    assert extract_text_array(dom_with_nl, squash_artifical_nl=False) == [None, 'Line1', None, None, 'Line2', None]

    # Test with strip_artifical_nl=False
    dom_with_strip = MockDom('div', children=[
        MockDom('p', text='Content')
    ])
    assert extract_text_array(dom_with_strip, strip_artifical_nl=False) == [None, 'Content', None]

    # Test with callable tag
    class CallableTagDom:
        def __init__(self):
            self.tag = lambda: 'div'

    callable_dom = CallableTagDom()
    assert extract_text_array(callable_dom) == ''

    # Test with empty dom
    empty_dom = MockDom('div')
    assert extract_text_array(empty_dom) == [None, None]


# LLM-generated content at query #48
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test inline tag with children
    child = MockDom('strong', text='World')
    dom = MockDom('span', children=[child])
    assert extract_text_array(dom) == ['World']

    # Test block tag
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test separator tag
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test nested tags
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('strong', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text_array(dom) == [None, 'Hello', 'World', None]

    # Test with tail text
    child = MockDom('span', text='Hello', tail='World')
    dom = MockDom('div', children=[child])
    assert extract_text_array(dom) == [None, 'Hello', 'World', None]

    # Test squash_artifical_nl=False
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None]

    # Test strip_artifical_nl=False
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with callable tag
    dom = MockDom(lambda: 'div', text='Hello')
    assert extract_text_array(dom) == ''

    # Test empty dom
    dom = MockDom('div')
    assert extract_text_array(dom) == [None, None]


# LLM-generated content at query #49
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline element
    inline_elem = MockElement('span', text='Hello')
    assert extract_text(inline_elem) == 'Hello'

    # Test block element
    block_elem = MockElement('div', text='Hello', children=[
        MockElement('p', text='World')
    ])
    assert extract_text(block_elem) == 'Hello\nWorld'

    # Test with separator
    sep_elem = MockElement('div', children=[
        MockElement('br'),
        MockElement('span', text='Test')
    ])
    assert extract_text(sep_elem) == '\nTest'

    # Test with nested elements
    nested_elem = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('p', text='Second', children=[
            MockElement('span', text='Nested')
        ])
    ])
    assert extract_text(nested_elem) == 'First\nSecondNested'

    # Test with squash_space=False
    assert extract_text(nested_elem, squash_space=False) == 'First\nSecondNested'

    # Test with custom symbols
    assert extract_text(block_elem, block_symbol='|', sep_symbol=';') == 'Hello|World'

    # Test with whitespace handling
    whitespace_elem = MockElement('div', text='  Hello  ', children=[
        MockElement('p', text='  World  ')
    ])
    assert extract_text(whitespace_elem) == 'Hello\nWorld'

    # Test empty element
    empty_elem = MockElement('div')
    assert extract_text(empty_elem) == ''

    # Test with tail text
    tail_elem = MockElement('div', children=[
        MockElement('span', text='Hello', tail='World')
    ])
    assert extract_text(tail_elem) == 'HelloWorld'

    # Test with multiple separators
    multi_sep = MockElement('div', children=[
        MockElement('br'),
        MockElement('br'),
        MockElement('span', text='Test')
    ])
    assert extract_text(multi_sep) == '\nTest'


# LLM-generated content at query #50
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail

        def getchildren(self):
            return self.children

    # Test inline tag
    inline_dom = MockDom('span', text='Hello')
    assert extract_text(inline_dom) == 'Hello'

    # Test block tag
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text(block_dom) == 'Hello\nWorld'

    # Test separator tag
    separator_dom = MockDom('br')
    assert extract_text(separator_dom) == '\n'

    # Test nested structure
    nested_dom = MockDom('div', children=[
        MockDom('p', text='First paragraph'),
        MockDom('p', text='Second paragraph')
    ])
    assert extract_text(nested_dom) == 'First paragraph\nSecond paragraph'

    # Test with squash_space=False
    assert extract_text(block_dom, squash_space=False) == 'Hello\nWorld'

    # Test with custom symbols
    assert extract_text(block_dom, block_symbol='|', sep_symbol='|') == 'Hello|World'

    # Test whitespace squashing
    whitespace_dom = MockDom('div', text='  Hello  \n  World  ')
    assert extract_text(whitespace_dom) == 'Hello World'

    # Test with no text
    empty_dom = MockDom('div')
    assert extract_text(empty_dom) == ''

    # Test callable tag
    callable_dom = MockDom(lambda: 'div')
    assert extract_text(callable_dom) == ''

    # Test complex structure
    complex_dom = MockDom('div', children=[
        MockDom('p', text='First '),
        MockDom('span', text='inline '),
        MockDom('span', text='text'),
        MockDom('br'),
        MockDom('p', text='Second paragraph')
    ])
    assert extract_text(complex_dom) == 'First inline text\nSecond paragraph'


# LLM-generated content at query #51
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline elements
    dom = fromstring('<div><span>Hello</span> <span>World</span></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block elements
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom) == 'First\nSecond'

    # Test with separators (br)
    dom = fromstring('<div>Line1<br/>Line2</div>')
    assert extract_text(dom) == 'Line1\nLine2'

    # Test with nested elements
    dom = fromstring('<div><p>Outer <span>Inner</span> text</p></div>')
    assert extract_text(dom) == 'Outer Inner text'

    # Test with whitespace squashing
    dom = fromstring('<div>  Hello   World  </div>')
    assert extract_text(dom) == 'Hello World'

    # Test with custom symbols
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol='-') == 'First|Second'

    # Test with no squashing
    dom = fromstring('<div>  Hello   World  </div>')
    assert extract_text(dom, squash_space=False) == '  Hello   World  '

    # Test with mixed content
    dom = fromstring('<div><p>Text<br/>with<br/>breaks</p></div>')
    assert extract_text(dom) == 'Text\nwith\nbreaks'

    # Test empty element
    dom = fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with only whitespace
    dom = fromstring('<div>   \n  \t  </div>')
    assert extract_text(dom) == ''

    # Test with script and style tags (should be treated as block elements)
    dom = fromstring('<div><script>var x = 1;</script>Text</div>')
    assert extract_text(dom) == 'Text'


# LLM-generated content at query #52
#--------------------------

```python
def test_extract_text_array():
    from lxml import html

    # Test with inline tags
    dom = html.fromstring('<p>Hello <strong>World</strong></p>')
    result = extract_text_array(dom)
    assert result == ['Hello ', 'World']

    # Test with block tags
    dom = html.fromstring('<div>Hello <p>World</p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello ', None, 'World', None]

    # Test with separator tags
    dom = html.fromstring('<p>Hello<br>World</p>')
    result = extract_text_array(dom)
    assert result == ['Hello', True, 'World']

    # Test with nested tags
    dom = html.fromstring('<div><p>Hello <span>World</span></p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello ', 'World', None]

    # Test with text and tail
    dom = html.fromstring('<p>Hello <strong>World</strong>!</p>')
    result = extract_text_array(dom)
    assert result == ['Hello ', 'World', '!']

    # Test with squash_artifical_nl=False
    dom = html.fromstring('<div>Hello</div><div>World</div>')
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = html.fromstring('<div>Hello</div>')
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'Hello', None]

    # Test with empty dom
    dom = html.fromstring('<div></div>')
    result = extract_text_array(dom)
    assert result == []

    # Test with only text
    dom = html.fromstring('<p>Hello World</p>')
    result = extract_text_array(dom)
    assert result == ['Hello World']

    # Test with mixed content
    dom = html.fromstring('<div>Hello <br> <p>World</p> <span>!</span></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello ', True, ' ', None, 'World', None, '!', None]


# LLM-generated content at query #53
#--------------------------

```python
def test_extract_text():
    from lxml import html

    # Test with simple inline tags
    dom = html.fromstring('<div><span>Hello</span> <span>World</span></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block tags
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with separator tags
    dom = html.fromstring('<div>Hello<br>World</div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with nested tags
    dom = html.fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with whitespace squashing
    dom = html.fromstring('<div>Hello   World</div>')
    assert extract_text(dom) == 'Hello World'

    # Test with leading/trailing whitespace
    dom = html.fromstring('<div>  Hello World  </div>')
    assert extract_text(dom) == 'Hello World'

    # Test with custom block and separator symbols
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello|World'

    # Test with mixed content
    dom = html.fromstring('<div><p>Hello<br>World</p><p>Foo</p></div>')
    assert extract_text(dom) == 'Hello\nWorld\nFoo'

    # Test with empty content
    dom = html.fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with only whitespace
    dom = html.fromstring('<div>   \n  \t  </div>')
    assert extract_text(dom) == ''

    # Test with squash_space=False
    dom = html.fromstring('<div>Hello   World</div>')
    assert extract_text(dom, squash_space=False) == 'Hello   World'


# LLM-generated content at query #54
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tags
    dom = fromstring('<div><span>Hello</span> <strong>World</strong></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block tags
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom) == 'First\nSecond'

    # Test with separator tags
    dom = fromstring('<div>Line1<br/>Line2</div>')
    assert extract_text(dom) == 'Line1\nLine2'

    # Test with nested tags
    dom = fromstring('<div><p>Outer <span>Inner</span> text</p></div>')
    assert extract_text(dom) == 'Outer Inner text'

    # Test with whitespace handling
    dom = fromstring('<div>  \n  <p>  Text  </p>  \n  </div>')
    assert extract_text(dom) == 'Text'

    # Test with custom symbols
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'First|Second'

    # Test with squash_space=False
    dom = fromstring('<div>  \n  <p>  Text  </p>  \n  </div>')
    assert extract_text(dom, squash_space=False) == '  \n  Text  \n  '

    # Test with empty content
    dom = fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with mixed content
    dom = fromstring('<div><p>First<br/>line</p><p>Second</p></div>')
    assert extract_text(dom) == 'First\nline\nSecond'

    # Test with special characters
    dom = fromstring('<div><p>Hello\tWorld\nNewline</p></div>')
    assert extract_text(dom) == 'Hello World Newline'


# LLM-generated content at query #55
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test 1: Simple inline tag
    dom = MockElement('span', text='Hello World')
    assert extract_text(dom) == 'Hello World'

    # Test 2: Block tag with text
    dom = MockElement('div', text='Hello', children=[
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test 3: Multiple block tags
    dom = MockElement('div', text='Hello', children=[
        MockElement('p', text='World'),
        MockElement('p', text='Foo')
    ])
    assert extract_text(dom) == 'Hello\nWorld\nFoo'

    # Test 4: Separator tag (br)
    dom = MockElement('div', text='Hello', children=[
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test 5: Nested inline tags
    dom = MockElement('div', children=[
        MockElement('span', text='Hello', children=[
            MockElement('strong', text='World')
        ])
    ])
    assert extract_text(dom) == 'HelloWorld'

    # Test 6: With squash_space=True (default)
    dom = MockElement('div', text='  Hello  ', children=[
        MockElement('span', text='  World  ')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test 7: With squash_space=False
    dom = MockElement('div', text='  Hello  ', children=[
        MockElement('span', text='  World  ')
    ])
    assert extract_text(dom, squash_space=False) == '  Hello  World  '

    # Test 8: Custom block and separator symbols
    dom = MockElement('div', text='Hello', children=[
        MockElement('p', text='World'),
        MockElement('br'),
        MockElement('span', text='Foo')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello|World;Foo'

    # Test 9: Empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''

    # Test 10: Element with only whitespace
    dom = MockElement('div', text='   \n  \t  ')
    assert extract_text(dom) == ''

    # Test 11: Complex nested structure
    dom = MockElement('div', text='Start', children=[
        MockElement('p', text='Paragraph 1', children=[
            MockElement('span', text='Nested'),
            MockElement('br'),
            MockElement('strong', text='Bold')
        ]),
        MockElement('div', text='Div content', children=[
            MockElement('span', text='More text')
        ]),
        MockElement('p', text='End')
    ])
    assert extract_text(dom) == 'Start\nParagraph 1Nested\nBold\nDiv contentMore text\nEnd'

    # Test 12: Callable tag (should return empty string)
    class CallableTag:
        def __init__(self):
            self.tag = lambda: None
    dom = CallableTag()
    assert extract_text(dom) == ''


# LLM-generated content at query #56
#--------------------------

```python
def test_extract_text():
    # Test basic inline tag handling
    from lxml.html import fromstring
    dom = fromstring('<div><p>Hello <b>world</b></p></div>')
    assert extract_text(dom) == 'Hello world'

    # Test block symbol
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom, block_symbol='\n') == 'Hello\nWorld'

    # Test separator symbol
    dom = fromstring('<div><p>Hello<br>World</p></div>')
    assert extract_text(dom, sep_symbol='\n') == 'Hello\nWorld'

    # Test squash_space
    dom = fromstring('<div><p>  Hello   world  </p></div>')
    assert extract_text(dom, squash_space=True) == 'Hello world'

    # Test mixed content
    dom = fromstring('<div><p>Hello <span>world</span>!</p></div>')
    assert extract_text(dom) == 'Hello world!'

    # Test empty content
    dom = fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test nested tags
    dom = fromstring('<div><div><p>Hello <span>world</span></p></div></div>')
    assert extract_text(dom) == 'Hello world'

    # Test with custom symbols
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol='-') == 'Hello|World'

    # Test with no squash_space
    dom = fromstring('<div><p>  Hello   world  </p></div>')
    assert extract_text(dom, squash_space=False) == '  Hello   world  '

    # Test with separators
    dom = fromstring('<div><p>Hello<br><br>World</p></div>')
    assert extract_text(dom, sep_symbol='\n') == 'Hello\n\nWorld'

    # Test with inline tags and separators
    dom = fromstring('<div><p>Hello <b>world</b><br>!</p></div>')
    assert extract_text(dom, sep_symbol='\n') == 'Hello world\n!'


# LLM-generated content at query #57
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    class MockDom:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self._children = children or []
            self.tail = tail

        def getchildren(self):
            return self._children

    # Test 1: Simple inline element
    dom = MockDom('span', text='Hello World')
    assert extract_text(dom) == 'Hello World'

    # Test 2: Block element with squash_space=True
    dom = MockDom('div', text='Hello', children=[
        MockDom('span', text='World')
    ], tail='!')
    assert extract_text(dom) == 'Hello World !'

    # Test 3: Block element with squash_space=False
    dom = MockDom('div', text='Hello', children=[
        MockDom('span', text='World')
    ], tail='!')
    assert extract_text(dom, squash_space=False) == '\nHello\nWorld\n!\n'

    # Test 4: Separator element (br)
    dom = MockDom('div', children=[
        MockDom('span', text='Hello'),
        MockDom('br'),
        MockDom('span', text='World')
    ])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test 5: Nested block elements
    dom = MockDom('div', children=[
        MockDom('p', text='First paragraph'),
        MockDom('p', text='Second paragraph')
    ])
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'

    # Test 6: Mixed inline and block elements
    dom = MockDom('div', children=[
        MockDom('span', text='Inline '),
        MockDom('strong', text='bold'),
        MockDom('p', text='New block')
    ])
    assert extract_text(dom) == 'Inline bold\nNew block'

    # Test 7: Custom block and separator symbols
    dom = MockDom('div', children=[
        MockDom('span', text='Hello'),
        MockDom('br'),
        MockDom('span', text='World')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello;World'

    # Test 8: Empty text
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test 9: Text with multiple spaces (squash_space=True)
    dom = MockDom('div', text='Hello   World')
    assert extract_text(dom) == 'Hello World'

    # Test 10: Text with multiple spaces (squash_space=False)
    dom = MockDom('div', text='Hello   World')
    assert extract_text(dom, squash_space=False) == '\nHello   World\n'

    # Test 11: Callable tag (should return empty string)
    dom = MockDom(lambda: None)
    assert extract_text(dom) == ''


# LLM-generated content at query #58
#--------------------------

```python
def test_extract_text():
    from lxml import html

    # Test with simple inline elements
    dom = html.fromstring('<div><span>Hello</span> <span>World</span></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block elements
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with separators (br)
    dom = html.fromstring('<div>Hello<br/>World</div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with nested elements
    dom = html.fromstring('<div><p>Hello <strong>World</strong></p></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with whitespace squashing
    dom = html.fromstring('<div>Hello   World</div>')
    assert extract_text(dom) == 'Hello World'

    # Test with custom block and separator symbols
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello|World'

    # Test with no squashing
    dom = html.fromstring('<div>Hello   World</div>')
    assert extract_text(dom, squash_space=False) == 'Hello   World'

    # Test with mixed content
    dom = html.fromstring('<div><p>Hello<br/>World</p><p>Test</p></div>')
    assert extract_text(dom) == 'Hello\nWorld\nTest'

    # Test with empty elements
    dom = html.fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with only whitespace
    dom = html.fromstring('<div>   \n  \t  </div>')
    assert extract_text(dom) == ''

    # Test with script and style tags (should be ignored)
    dom = html.fromstring('<div>Hello<script>alert("xss")</script>World</div>')
    assert extract_text(dom) == 'HelloWorld'

    # Test with multiple separators
    dom = html.fromstring('<div>Hello<br/><br/>World</div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with text in tail
    dom = html.fromstring('<div><span>Hello</span>World</div>')
    assert extract_text(dom) == 'HelloWorld'

    # Test with complex nesting
    dom = html.fromstring('''<div>
        <p>Hello <span>World</span></p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </div>''')
    assert extract_text(dom) == 'Hello World\nItem 1\nItem 2'


# LLM-generated content at query #59
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline elements
    dom = fromstring("<div><span>Hello</span> <strong>World</strong></div>")
    assert extract_text(dom) == "Hello World"

    # Test with block elements and whitespace squashing
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with separators (br tags)
    dom = fromstring("<div>Hello<br>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with nested elements
    dom = fromstring("<div><p>Hello <span>World</span></p></div>")
    assert extract_text(dom) == "Hello World"

    # Test with custom block and separator symbols
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == "Hello|World"

    # Test with whitespace preservation
    dom = fromstring("<div>  Hello  World  </div>")
    assert extract_text(dom, squash_space=False) == "  Hello  World  "

    # Test with empty elements
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with mixed content
    dom = fromstring("<div><p>Hello</p><br><span>World</span></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with preformatted content (whitespace should be preserved if squash_space is False)
    dom = fromstring("<pre>  Hello  World  </pre>")
    assert extract_text(dom, squash_space=False) == "  Hello  World  "

    # Test with multiple separators
    dom = fromstring("<div>Hello<br><br>World</div>")
    assert extract_text(dom) == "Hello\nWorld"


# LLM-generated content at query #60
#--------------------------

```python
def test_extract_text():
    # Test basic inline tag handling
    dom = Mock(tag='p')
    dom.text = 'Hello'
    child1 = Mock(tag='strong')
    child1.text = 'World'
    child1.tail = '!'
    dom.getchildren = lambda: [child1]
    assert extract_text(dom) == 'Hello World!'

    # Test block tag handling with newlines
    dom = Mock(tag='div')
    dom.text = 'Line1'
    child1 = Mock(tag='p')
    child1.text = 'Line2'
    child1.tail = 'Line3'
    dom.getchildren = lambda: [child1]
    assert extract_text(dom) == 'Line1\nLine2\nLine3'

    # Test separator tag handling
    dom = Mock(tag='div')
    dom.text = 'First'
    child1 = Mock(tag='br')
    child1.tail = 'Second'
    dom.getchildren = lambda: [child1]
    assert extract_text(dom) == 'First\nSecond'

    # Test whitespace squashing
    dom = Mock(tag='p')
    dom.text = 'Hello   World'
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test whitespace preservation
    dom = Mock(tag='p')
    dom.text = 'Hello   World'
    assert extract_text(dom, squash_space=False) == 'Hello   World'

    # Test nested tags
    dom = Mock(tag='div')
    dom.text = 'Outer'
    child1 = Mock(tag='p')
    child1.text = 'Inner'
    child2 = Mock(tag='span')
    child2.text = 'Nested'
    child1.getchildren = lambda: [child2]
    dom.getchildren = lambda: [child1]
    assert extract_text(dom) == 'Outer\nInnerNested'

    # Test empty text
    dom = Mock(tag='p')
    dom.text = None
    assert extract_text(dom) == ''

    # Test multiple children
    dom = Mock(tag='div')
    child1 = Mock(tag='p')
    child1.text = 'First'
    child2 = Mock(tag='p')
    child2.text = 'Second'
    dom.getchildren = lambda: [child1, child2]
    assert extract_text(dom) == 'First\nSecond'

    # Test custom block and separator symbols
    dom = Mock(tag='div')
    dom.text = 'A'
    child1 = Mock(tag='p')
    child1.text = 'B'
    child1.tail = 'C'
    dom.getchildren = lambda: [child1]
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'A|B;C'

    # Test with no children
    dom = Mock(tag='p')
    dom.text = 'Simple text'
    assert extract_text(dom) == 'Simple text'

    # Test with callable tag (should return empty string)
    dom = Mock()
    dom.tag = lambda: 'div'
    assert extract_text(dom) == ''


# LLM-generated content at query #61
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with simple inline tag
    dom = fromstring('<p>Hello <b>World</b></p>')
    assert extract_text_array(dom) == ['Hello ', 'World']

    # Test with block tag
    dom = fromstring('<div>Hello <p>World</p></div>')
    assert extract_text_array(dom) == [None, 'Hello ', None, 'World', None]

    # Test with separator tag
    dom = fromstring('<p>Hello<br/>World</p>')
    assert extract_text_array(dom) == ['Hello ', True, 'World']

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text_array(dom) == [None, 'Hello ', 'World', None]

    # Test with text and tail
    dom = fromstring('<p>Hello <b>World</b>!</p>')
    assert extract_text_array(dom) == ['Hello ', 'World', '!']

    # Test with squash_artifical_nl=False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = fromstring('<div><p>Hello</p></div>')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with empty dom
    dom = fromstring('<div></div>')
    assert extract_text_array(dom) == [None, None]

    # Test with only text
    dom = fromstring('<p>Hello World</p>')
    assert extract_text_array(dom) == ['Hello World']

    # Test with multiple separators
    dom = fromstring('<p>Hello<br/><br/>World</p>')
    assert extract_text_array(dom) == ['Hello ', True, 'World']


# LLM-generated content at query #62
#--------------------------

```python
def test_extract_text():
    from lxml import html

    # Test with simple inline elements
    dom = html.fromstring('<div><span>Hello</span> <span>World</span></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block elements
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with separators (br)
    dom = html.fromstring('<div>Hello<br/>World</div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with nested elements
    dom = html.fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with whitespace squashing
    dom = html.fromstring('<div>Hello   World</div>')
    assert extract_text(dom) == 'Hello World'

    # Test with custom block and separator symbols
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == 'Hello|World'

    # Test with no squashing
    dom = html.fromstring('<div>Hello   World</div>')
    assert extract_text(dom, squash_space=False) == 'Hello   World'

    # Test with mixed content
    dom = html.fromstring('<div><p>Hello<br/>World</p><p>Foo</p></div>')
    assert extract_text(dom) == 'Hello\nWorld\nFoo'

    # Test with empty elements
    dom = html.fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with only whitespace
    dom = html.fromstring('<div>   </div>')
    assert extract_text(dom) == ''

    # Test with script and style tags (should be treated as block elements)
    dom = html.fromstring('<div><script>alert("Hello")</script>World</div>')
    assert extract_text(dom) == 'World'

    # Test with preformatted text (should not squash whitespace)
    dom = html.fromstring('<pre>Hello   World</pre>')
    assert extract_text(dom, squash_space=False) == 'Hello   World'


