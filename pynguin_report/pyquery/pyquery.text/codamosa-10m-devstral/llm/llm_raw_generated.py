####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test basic text extraction
    dom = fromstring("<div>Hello <b>World</b></div>")
    assert extract_text(dom) == "Hello World"

    # Test with block elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"

    # Test with inline elements
    dom = fromstring("<div>Text with <span>inline</span> elements</div>")
    assert extract_text(dom) == "Text with inline elements"

    # Test with separators (br)
    dom = fromstring("<div>Line1<br>Line2</div>")
    assert extract_text(dom) == "Line1\nLine2"

    # Test with nested elements
    dom = fromstring("<div><p>Outer <span>Inner</span> text</p></div>")
    assert extract_text(dom) == "Outer Inner text"

    # Test with whitespace squashing
    dom = fromstring("<div>  Multiple   spaces   here  </div>")
    assert extract_text(dom) == "Multiple spaces here"

    # Test with custom symbols
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol='||') == "First|Second"

    # Test with no squashing
    dom = fromstring("<div>  Text  </div>")
    assert extract_text(dom, squash_space=False) == "  Text  "

    # Test with empty content
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with only whitespace
    dom = fromstring("<div>   </div>")
    assert extract_text(dom) == ""

    # Test with mixed content
    dom = fromstring("<div><p>First</p>Text<br><p>Second</p></div>")
    assert extract_text(dom) == "First\nText\nSecond"

    # Test with script and style tags (should be ignored)
    dom = fromstring("<div><script>alert('xss')</script>Text</div>")
    assert extract_text(dom) == "Text"

    # Test with self-closing tags
    dom = fromstring("<div><img src='test.jpg'/>Image text</div>")
    assert extract_text(dom) == "Image text"

    # Test with complex nesting
    dom = fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph <strong>bold</strong> text</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    assert extract_text(dom) == "Title\nParagraph bold text\nItem 1\nItem 2"


# LLM-generated content at query #2
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tags
    dom = fromstring("<div><span>Hello</span> <strong>World</strong></div>")
    assert extract_text(dom) == "Hello World"

    # Test with block tags
    dom = fromstring("<div><p>First paragraph</p><p>Second paragraph</p></div>")
    assert extract_text(dom) == "First paragraph\nSecond paragraph"

    # Test with separators (br)
    dom = fromstring("<div>Line 1<br/>Line 2</div>")
    assert extract_text(dom) == "Line 1\nLine 2"

    # Test with nested tags
    dom = fromstring("<div><ul><li>Item 1</li><li>Item 2</li></ul></div>")
    assert extract_text(dom) == "Item 1\nItem 2"

    # Test with whitespace squashing
    dom = fromstring("<div>  Hello   \n  World  </div>")
    assert extract_text(dom) == "Hello World"

    # Test with custom symbols
    dom = fromstring("<div><p>Para1</p><p>Para2</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == "Para1|Para2"

    # Test with squash_space=False
    dom = fromstring("<div>  Hello   \n  World  </div>")
    assert extract_text(dom, squash_space=False) == "  Hello   \n  World  "

    # Test with empty content
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with mixed content
    dom = fromstring("<div><p>Text <span>with</span> inline</p><br/><p>Another</p></div>")
    assert extract_text(dom) == "Text with inline\nAnother"

    # Test with script and style tags (should be ignored)
    dom = fromstring("<div><script>alert('xss')</script><p>Content</p></div>")
    assert extract_text(dom) == "Content"


# LLM-generated content at query #3
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tags
    dom = fromstring('<div><span>Hello</span> <span>World</span></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block tags
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with separators
    dom = fromstring('<div>Hello<br>World</div>')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with whitespace squashing
    dom = fromstring('<div>Hello   World</div>')
    assert extract_text(dom) == 'Hello World'

    # Test with custom block and separator symbols
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello|World'

    # Test with squash_space=False
    dom = fromstring('<div>  Hello  World  </div>')
    assert extract_text(dom, squash_space=False) == '  Hello  World  '

    # Test with mixed content
    dom = fromstring('<div><p>Hello<br>World</p><p>Foo</p></div>')
    assert extract_text(dom) == 'Hello\nWorld\nFoo'

    # Test with empty content
    dom = fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with only whitespace
    dom = fromstring('<div>   \n  \t  </div>')
    assert extract_text(dom) == ''

    # Test with script and style tags (should be ignored)
    dom = fromstring('<div>Hello<script>alert("xss")</script>World</div>')
    assert extract_text(dom) == 'HelloWorld'

    # Test with preformatted content (whitespace should be preserved if squash_space=False)
    dom = fromstring('<pre>Hello   World</pre>')
    assert extract_text(dom, squash_space=False) == 'Hello   World'


# LLM-generated content at query #4
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
    dom = MockDom('div', text='Hello', children=[MockDom('div', text='World')])
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, 'World', None, None]

    # Test with strip_artifical_nl=False
    dom = MockDom('div', text='Hello', children=[MockDom('div', text='World')])
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None, 'World', None]

    # Test with both flags False
    dom = MockDom('div', text='Hello', children=[MockDom('div', text='World')])
    assert extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Hello', None, 'World', None, None]

    # Test callable tag
    dom = MockDom(tag=lambda: 'div', text='Hello')
    assert extract_text_array(dom) == ['']


# LLM-generated content at query #5
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

    dom = MockDom('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test with separator tag
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test with block tag
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test with nested tags
    child = MockDom('span', text='World')
    dom = MockDom('div', text='Hello', children=[child], tail='!')
    assert extract_text_array(dom) == [None, 'Hello', 'World', '!', None]

    # Test with squash_artifical_nl=False
    dom = MockDom('div', text='Hello', children=[MockDom('div', text='World')])
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = MockDom('div', text='Hello', children=[MockDom('div', text='World')])
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None, 'World', None]

    # Test with both flags False
    dom = MockDom('div', text='Hello', children=[MockDom('div', text='World')])
    assert extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Hello', None, 'World', None]

    # Test with callable tag
    class CallableTagDom:
        def __init__(self):
            self.tag = lambda: 'div'
            self.text = 'Hello'
            self.children = []

        def getchildren(self):
            return self.children

    dom = CallableTagDom()
    assert extract_text_array(dom) == ''

    # Test with None text and tail
    dom = MockDom('div', text=None, children=[MockDom('span', text=None, tail=None)])
    assert extract_text_array(dom) == [None, None]

    # Test with empty children
    dom = MockDom('div', text='Hello', children=[])
    assert extract_text_array(dom) == [None, 'Hello', None]


# LLM-generated content at query #6
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
        MockElement('p', text='First'),
        MockElement('p', text='Second')
    ])
    assert extract_text(nested_elem) == 'First\nSecond'

    # Test with separators
    sep_elem = MockElement('div', children=[
        MockElement('br'),
        MockElement('span', text='After break')
    ])
    assert extract_text(sep_elem) == '\nAfter break'

    # Test squash_space
    space_elem = MockElement('div', text='  Hello  ', children=[
        MockElement('span', text='  World  ')
    ])
    assert extract_text(space_elem) == 'Hello World'

    # Test custom symbols
    assert extract_text(block_elem, block_symbol='|', sep_symbol='||') == 'Hello'

    # Test with None text
    none_elem = MockElement('div', text=None)
    assert extract_text(none_elem) == ''

    # Test mixed content
    mixed_elem = MockElement('div', text='Start', children=[
        MockElement('p', text='Paragraph'),
        MockElement('br'),
        MockElement('span', text='End')
    ])
    assert extract_text(mixed_elem) == 'Start\nParagraph\nEnd'

    # Test with tail text
    tail_elem = MockElement('div', children=[
        MockElement('p', text='First', tail='Tail')
    ])
    assert extract_text(tail_elem) == 'FirstTail'

    # Test callable tag
    callable_elem = MockElement(lambda: 'callable')
    assert extract_text(callable_elem) == ''


# LLM-generated content at query #7
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
    space_elem = MockElement('div', text='  Hello   world  ')
    assert extract_text(space_elem, squash_space=True) == 'Hello world'
    assert extract_text(space_elem, squash_space=False) == '  Hello   world  '

    # Test custom symbols
    custom_elem = MockElement('div', children=[
        MockElement('p', text='Part 1'),
        MockElement('p', text='Part 2')
    ])
    assert extract_text(custom_elem, block_symbol='|', sep_symbol=';') == 'Part 1|Part 2'

    # Test with tail text
    tail_elem = MockElement('div', children=[
        MockElement('p', text='Main', tail=' tail')
    ])
    assert extract_text(tail_elem) == 'Main tail'

    # Test empty element
    empty_elem = MockElement('div')
    assert extract_text(empty_elem) == ''

    # Test mixed content
    mixed_elem = MockElement('div', children=[
        MockElement('span', text='Inline '),
        MockElement('div', text='Block'),
        MockElement('span', text=' content')
    ])
    assert extract_text(mixed_elem) == 'Inline \nBlock content'


# LLM-generated content at query #8
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
        MockDom('p', text='Hello'),
        MockDom('p', text='World')
    ])
    assert extract_text(nested_dom) == 'Hello\nWorld'

    # Test with separators
    sep_dom = MockDom('div', children=[
        MockDom('p', text='Hello'),
        MockDom('br'),
        MockDom('p', text='World')
    ])
    assert extract_text(sep_dom) == 'Hello\nWorld'

    # Test squash_space
    space_dom = MockDom('div', text='  Hello  ', children=[
        MockDom('p', text='  World  ')
    ])
    assert extract_text(space_dom, squash_space=True) == 'Hello World'
    assert extract_text(space_dom, squash_space=False) == '  Hello  \n  World  '

    # Test custom symbols
    custom_dom = MockDom('div', children=[
        MockDom('p', text='Hello'),
        MockDom('p', text='World')
    ])
    assert extract_text(custom_dom, block_symbol='|', sep_symbol='-') == 'Hello|World'

    # Test empty dom
    empty_dom = MockDom('div')
    assert extract_text(empty_dom) == ''

    # Test with tail
    tail_dom = MockDom('div', children=[
        MockDom('p', text='Hello', tail='World')
    ])
    assert extract_text(tail_dom) == 'Hello World'


# LLM-generated content at query #9
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with simple inline tag
    dom = fromstring('<span>Hello <b>World</b></span>')
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
    dom = fromstring('<ul><li>Item 1</li><li>Item 2</li></ul>')
    assert extract_text_array(dom) == [None, 'Item 1', None, 'Item 2', None]

    # Test with squash_artifical_nl=False
    dom = fromstring('<div>Hello</div><div>World</div>')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = fromstring('<div><div>Hello</div></div>')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, None, 'Hello', None, None]

    # Test with empty dom
    dom = fromstring('<div></div>')
    assert extract_text_array(dom) == [None, None]

    # Test with only text
    dom = fromstring('Just text')
    assert extract_text_array(dom) == ['Just text']

    # Test with mixed content
    dom = fromstring('<div>Text <span>more text</span> and <br> even more</div>')
    assert extract_text_array(dom) == [None, 'Text ', 'more text', ' and ', True, ' even more', None]


# LLM-generated content at query #10
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
    dom = MockDom('div', children=[MockDom('div')])
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, None, None]

    # Test strip_artifical_nl=False
    dom = MockDom('div', children=[MockDom('span', text='Hello')])
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with callable tag
    class CallableTagDom:
        def __init__(self):
            self.tag = lambda: 'div'
    dom = CallableTagDom()
    assert extract_text_array(dom) == ''


# LLM-generated content at query #11
#--------------------------

```python
def test_extract_text_array():
    # Test basic inline tag handling
    from lxml import html
    dom = html.fromstring('<div><span>Hello</span> <span>World</span></div>')
    assert extract_text_array(dom) == ['Hello', ' ', 'World']

    # Test block tag handling with artificial newlines
    dom = html.fromstring('<div><p>Paragraph 1</p><p>Paragraph 2</p></div>')
    assert extract_text_array(dom) == [None, 'Paragraph 1', None, None, 'Paragraph 2', None]

    # Test separator handling (br tag)
    dom = html.fromstring('<div>Line 1<br>Line 2</div>')
    assert extract_text_array(dom) == ['Line 1', True, 'Line 2']

    # Test nested tags
    dom = html.fromstring('<div><ul><li>Item 1</li><li>Item 2</li></ul></div>')
    assert extract_text_array(dom) == [None, None, 'Item 1', None, None, 'Item 2', None, None]

    # Test with squash_artifical_nl=False
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = html.fromstring('<div><p>Hello</p></div>')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test empty element
    dom = html.fromstring('<div></div>')
    assert extract_text_array(dom) == [None]

    # Test text with whitespace
    dom = html.fromstring('<div>  Hello   World  </div>')
    assert extract_text_array(dom) == ['  Hello   World  ']

    # Test mixed content
    dom = html.fromstring('<div><p>Hello<br>World</p><span>!</span></div>')
    assert extract_text_array(dom) == [None, 'Hello', True, 'World', None, '!']

    # Test with both flags False
    dom = html.fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]


# LLM-generated content at query #12
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
    inline_dom = MockDom('span', text='Hello')
    assert extract_text(inline_dom) == 'Hello'

    # Test block tag
    block_dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text(block_dom) == 'Hello\nWorld'

    # Test separator tag
    sep_dom = MockDom('br')
    assert extract_text(sep_dom) == '\n'

    # Test nested structure
    nested_dom = MockDom('div', children=[
        MockDom('p', text='First paragraph'),
        MockDom('p', text='Second paragraph')
    ])
    assert extract_text(nested_dom) == 'First paragraph\nSecond paragraph'

    # Test with squash_space=False
    dom = MockDom('div', text='  Hello  ', children=[], tail='  World  ')
    assert extract_text(dom, squash_space=False) == '  Hello  \n  World  '

    # Test with custom symbols
    dom = MockDom('div', children=[
        MockDom('p', text='Line1'),
        MockDom('p', text='Line2')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Line1|Line2'

    # Test with None tag (callable)
    class CallableTagDom:
        def __init__(self):
            self.tag = lambda: None
            self.text = None
            self._children = []

        def getchildren(self):
            return self._children

    callable_dom = CallableTagDom()
    assert extract_text(callable_dom) == ''

    # Test whitespace squashing
    dom = MockDom('div', text='  Hello    World  ', children=[], tail='  Tail  ')
    assert extract_text(dom) == 'Hello World\nTail'

    # Test complex structure
    complex_dom = MockDom('div', children=[
        MockDom('h1', text='Title'),
        MockDom('p', text='First '),
        MockDom('span', text='inline '),
        MockDom('span', text='text'),
        MockDom('p', text='Second paragraph'),
        MockDom('br'),
        MockDom('p', text='After break')
    ])
    assert extract_text(complex_dom) == 'Title\nFirst inline text\nSecond paragraph\n\nAfter break'


# LLM-generated content at query #13
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
    dom = MockDom('span', 'Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test block tag with text
    dom = MockDom('div', 'Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test separator tag
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test nested tags
    child = MockDom('span', 'World')
    dom = MockDom('div', 'Hello', [child], '!')
    assert extract_text_array(dom) == [None, 'Hello', 'World', '!', None]

    # Test with squash_artifical_nl=False
    dom = MockDom('div', 'Hello', [], 'World')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', 'World', None]

    # Test with strip_artifical_nl=False
    dom = MockDom('div', 'Hello', [], 'World')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', 'World', None]

    # Test with both flags False
    dom = MockDom('div', 'Hello', [], 'World')
    assert extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Hello', 'World', None]

    # Test callable tag
    dom = MockDom(lambda: 'div', 'Hello')
    assert extract_text_array(dom) == ''

    # Test empty dom
    dom = MockDom('div')
    assert extract_text_array(dom) == [None, None]


# LLM-generated content at query #14
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    class MockElement:
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

    # Test inline tag with text
    dom = MockElement('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test block tag with text
    dom = MockElement('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test separator tag
    dom = MockElement('br')
    assert extract_text_array(dom) == [True]

    # Test nested tags
    child = MockElement('span', text='World')
    dom = MockElement('div', text='Hello', children=[child], tail='!')
    assert extract_text_array(dom) == [None, 'Hello', 'World', '!', None]

    # Test with squash_artifical_nl=False
    dom = MockElement('div', text='Hello')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None]

    # Test with strip_artifical_nl=False
    dom = MockElement('div', text='Hello')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test callable tag
    dom = MockElement(lambda: None)
    assert extract_text_array(dom) == ''

    # Test complex nested structure
    grandchild = MockElement('b', text='bold')
    child1 = MockElement('span', text='Hello', children=[grandchild], tail=' ')
    child2 = MockElement('div', text='World')
    dom = MockElement('body', children=[child1, child2], tail='!')
    result = extract_text_array(dom)
    assert result == [None, 'Hello', 'bold', ' ', None, 'World', None, '!', None]

    # Test with multiple separators
    child1 = MockElement('br')
    child2 = MockElement('br')
    dom = MockElement('div', children=[child1, child2])
    assert extract_text_array(dom) == [None, True, True, None]


# LLM-generated content at query #15
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

    # Test with separators
    dom = fromstring("<p>Hello<br>World</p>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with whitespace squashing
    dom = fromstring("<p>Hello   \n  World</p>")
    assert extract_text(dom) == "Hello World"

    # Test with mixed content
    dom = fromstring("<div>Hello <p>World</p> <span>!</span></div>")
    assert extract_text(dom) == "Hello\nWorld !"

    # Test with custom symbols
    dom = fromstring("<div>Hello<div>World</div></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == "Hello|World"

    # Test with squash_space=False
    dom = fromstring("<p>Hello   \n  World</p>")
    assert extract_text(dom, squash_space=False) == "Hello   \n  World"

    # Test with empty content
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with nested tags
    dom = fromstring("<div><p>Hello <span>World</span></p></div>")
    assert extract_text(dom) == "Hello World"

    # Test with multiple separators
    dom = fromstring("<p>Hello<br><br>World</p>")
    assert extract_text(dom) == "Hello\nWorld"


# LLM-generated content at query #16
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml.html import fromstring

    # Test 1: Simple inline elements
    dom = fromstring("<div><span>Hello</span> <strong>World</strong></div>")
    assert extract_text(dom) == "Hello World"

    # Test 2: Block elements with squash_space=True
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test 3: Block elements with squash_space=False
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, squash_space=False) == "\nHello\n\nWorld\n"

    # Test 4: Separator elements (br)
    dom = fromstring("<div>Hello<br>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test 5: Mixed inline and block elements
    dom = fromstring("<div><p>Hello <span>there</span></p><p>World</p></div>")
    assert extract_text(dom) == "Hello there\nWorld"

    # Test 6: Whitespace handling
    dom = fromstring("<div>  Hello   World  </div>")
    assert extract_text(dom) == "Hello World"

    # Test 7: Custom block and separator symbols
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == "Hello|World"

    # Test 8: Empty elements
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test 9: Nested block elements
    dom = fromstring("<div><div><p>Hello</p></div><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test 10: Text with newlines and tabs
    dom = fromstring("<div>Hello\n\tWorld</div>")
    assert extract_text(dom) == "Hello World"

    # Test 11: Preformatted text (no squash_space)
    dom = fromstring("<pre>Hello   World</pre>")
    assert extract_text(dom, squash_space=False) == "Hello   World"

    # Test 12: Complex nested structure
    dom = fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph <em>with</em> emphasis</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    assert extract_text(dom) == "Title\nParagraph with emphasis\nItem 1\nItem 2"

    # Test 13: Inline elements with block children (should not happen in valid HTML)
    dom = fromstring("<span><div>Hello</div></span>")
    assert extract_text(dom) == "Hello"

    # Test 14: Multiple consecutive separators
    dom = fromstring("<div>Hello<br><br>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test 15: Text with special characters
    dom = fromstring("<div>Hello &amp; World</div>")
    assert extract_text(dom) == "Hello & World"


# LLM-generated content at query #17
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with simple inline tag
    dom = fromstring('<span>Hello</span>')
    assert extract_text_array(dom) == ['Hello']

    # Test with block tag
    dom = fromstring('<div>Hello</div>')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test with separator tag
    dom = fromstring('<br/>')
    assert extract_text_array(dom) == [True]

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>world</span></p></div>')
    assert extract_text_array(dom) == [None, None, 'Hello ', None, 'world', None, None, None]

    # Test with text and tail
    dom = fromstring('<div>Hello<span> world</span>!</div>')
    assert extract_text_array(dom) == [None, 'Hello', None, ' world', '!', None]

    # Test with squash_artifical_nl=False
    dom = fromstring('<div>Hello</div><div>World</div>')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = fromstring('<div>Hello</div>')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with both flags False
    dom = fromstring('<div>Hello</div><div>World</div>')
    assert extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with empty dom
    dom = fromstring('<div></div>')
    assert extract_text_array(dom) == [None, None]

    # Test with multiple separators
    dom = fromstring('<br/><br/>')
    assert extract_text_array(dom) == [True, True]

    # Test with mixed content
    dom = fromstring('''<div>
        <p>Hello<br/>world</p>
        <span>!</span>
    </div>''')
    assert extract_text_array(dom) == [None, None, 'Hello', True, 'world', None, '!', None]


# LLM-generated content at query #18
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tags
    dom = fromstring('<div><span>Hello</span> <strong>World</strong></div>')
    assert extract_text(dom) == 'Hello World'

    # Test with block tags
    dom = fromstring('<div><p>First paragraph</p><p>Second paragraph</p></div>')
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'

    # Test with separators (br tags)
    dom = fromstring('<div>Line 1<br/>Line 2</div>')
    assert extract_text(dom) == 'Line 1\nLine 2'

    # Test with nested tags
    dom = fromstring('<div><p>Outer <span>inner</span> text</p></div>')
    assert extract_text(dom) == 'Outer inner text'

    # Test with whitespace squashing
    dom = fromstring('<div>  Multiple   spaces   and\ntabs\t\n</div>')
    assert extract_text(dom) == 'Multiple spaces and tabs'

    # Test with custom symbols
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'First|Second'

    # Test with no squashing
    dom = fromstring('<div>  Text  </div>')
    assert extract_text(dom, squash_space=False) == '  Text  '

    # Test with empty content
    dom = fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with mixed content
    dom = fromstring('''<div>
        <p>Paragraph 1<br/>with break</p>
        <p>Paragraph 2 <span>with <strong>nested</strong> tags</span></p>
    </div>''')
    assert extract_text(dom) == 'Paragraph 1\nwith break\nParagraph 2 with nested tags'

    # Test with script and style tags (should be ignored)
    dom = fromstring('''<div>
        <p>Visible text</p>
        <script>var x = 1;</script>
        <style>body { color: red; }</style>
        <p>More text</p>
    </div>''')
    assert extract_text(dom) == 'Visible text\nMore text'


# LLM-generated content at query #19
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self.children = children or []

        def getchildren(self):
            return self.children

    # Test inline tag
    inline_dom = MockDom('span', text='Hello')
    assert extract_text(inline_dom) == 'Hello'

    # Test block tag
    block_dom = MockDom('div', text='Hello', children=[
        MockDom('p', text='World')
    ])
    assert extract_text(block_dom) == 'Hello\nWorld'

    # Test separator tag
    sep_dom = MockDom('div', children=[
        MockDom('br'),
        MockDom('span', text='Test')
    ])
    assert extract_text(sep_dom) == '\nTest'

    # Test squash_space
    space_dom = MockDom('div', text='  Hello  ', children=[
        MockDom('p', text='  World  ')
    ])
    assert extract_text(space_dom, squash_space=True) == 'Hello\nWorld'
    assert extract_text(space_dom, squash_space=False) == '  Hello  \n  World  '

    # Test custom symbols
    custom_dom = MockDom('div', children=[
        MockDom('p', text='Line1'),
        MockDom('p', text='Line2')
    ])
    assert extract_text(custom_dom, block_symbol='|', sep_symbol=';') == 'Line1|Line2'

    # Test empty dom
    empty_dom = MockDom('div')
    assert extract_text(empty_dom) == ''

    # Test nested tags
    nested_dom = MockDom('div', children=[
        MockDom('p', text='Outer', children=[
            MockDom('span', text='Inner')
        ])
    ])
    assert extract_text(nested_dom) == 'OuterInner'

    # Test tail text
    tail_dom = MockDom('div', children=[
        MockDom('p', text='Hello', tail='World')
    ])
    assert extract_text(tail_dom) == 'HelloWorld'


# LLM-generated content at query #20
#--------------------------

```python
def test_extract_text():
    # Test basic inline tags
    dom = Mock(tag='p', text='Hello ', children=[
        Mock(tag='strong', text='World', children=[], tail='!'),
    ])
    assert extract_text(dom) == 'Hello World!'

    # Test block tags with squash_space=True
    dom = Mock(tag='div', text='Line1', children=[
        Mock(tag='p', text='Line2', children=[], tail='Line3'),
    ])
    assert extract_text(dom) == 'Line1\nLine2\nLine3'

    # Test block tags with squash_space=False
    dom = Mock(tag='div', text='Line1', children=[
        Mock(tag='p', text='Line2', children=[], tail='Line3'),
    ])
    assert extract_text(dom, squash_space=False) == 'Line1\nLine2\nLine3'

    # Test separator tags (br)
    dom = Mock(tag='p', text='First', children=[
        Mock(tag='br', text=None, children=[], tail='Second'),
    ])
    assert extract_text(dom) == 'First\nSecond'

    # Test multiple whitespace squashing
    dom = Mock(tag='p', text='Hello   World', children=[], tail=None)
    assert extract_text(dom) == 'Hello World'

    # Test custom block and separator symbols
    dom = Mock(tag='div', text='A', children=[
        Mock(tag='p', text='B', children=[], tail='C'),
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'A|B;C'

    # Test empty dom
    dom = Mock(tag='div', text=None, children=[], tail=None)
    assert extract_text(dom) == ''

    # Test nested inline tags
    dom = Mock(tag='div', text='Start ', children=[
        Mock(tag='span', text='Middle ', children=[
            Mock(tag='strong', text='End', children=[], tail=None),
        ], tail='!'),
    ])
    assert extract_text(dom) == 'Start Middle End!'

    # Test whitespace handling with squash_space
    dom = Mock(tag='div', text='  Text  ', children=[], tail='  More  ')
    assert extract_text(dom) == 'Text More'

    # Test artificial newlines stripping
    dom = Mock(tag='div', text=None, children=[
        Mock(tag='p', text='Content', children=[], tail=None),
    ])
    assert extract_text(dom) == 'Content'

    # Test multiple block elements
    dom = Mock(tag='div', text=None, children=[
        Mock(tag='p', text='First', children=[], tail=None),
        Mock(tag='p', text='Second', children=[], tail=None),
    ])
    assert extract_text(dom) == 'First\nSecond'

class Mock:
    def __init__(self, tag, text, children, tail=None):
        self.tag = tag
        self.text = text
        self._children = children
        self.tail = tail

    def getchildren(self):
        return self._children


# LLM-generated content at query #21
#--------------------------

```python
def test_extract_text():
    # Test basic inline tags
    dom = Mock(tag='p', text='Hello ')
    child = Mock(tag='strong', text='World', tail='!')
    dom.getchildren.return_value = [child]
    assert extract_text(dom) == 'Hello World!'

    # Test block tags with squash_space
    dom = Mock(tag='div', text='Line1 ')
    child = Mock(tag='p', text='Line2', tail=' Line3')
    dom.getchildren.return_value = [child]
    assert extract_text(dom, squash_space=True) == 'Line1\nLine2 Line3'

    # Test separators (br tags)
    dom = Mock(tag='p', text='First ')
    br = Mock(tag='br')
    child = Mock(tag='span', text='Second')
    dom.getchildren.return_value = [br, child]
    assert extract_text(dom, sep_symbol='|') == 'First |Second'

    # Test whitespace squashing
    dom = Mock(tag='div', text='  Multiple   spaces  ')
    assert extract_text(dom, squash_space=True) == 'Multiple spaces'

    # Test nested tags
    dom = Mock(tag='div', text='Outer ')
    inner = Mock(tag='span', text='Inner ')
    child = Mock(tag='strong', text='Text')
    inner.getchildren.return_value = [child]
    dom.getchildren.return_value = [inner]
    assert extract_text(dom) == 'Outer Inner Text'

    # Test empty content
    dom = Mock(tag='div')
    assert extract_text(dom) == ''

    # Test callable tag
    dom = Mock(tag=lambda: 'div', text='Content')
    assert extract_text(dom) == ''

    # Test custom symbols
    dom = Mock(tag='div', text='A ')
    child = Mock(tag='p', text='B')
    dom.getchildren.return_value = [child]
    assert extract_text(dom, block_symbol='|', sep_symbol='~') == 'A |B'

    # Test with no squashing
    dom = Mock(tag='div', text='  Text  ')
    assert extract_text(dom, squash_space=False) == '  Text  '

    # Test multiple separators
    dom = Mock(tag='p', text='A')
    br1 = Mock(tag='br')
    br2 = Mock(tag='br')
    child = Mock(tag='span', text='B')
    dom.getchildren.return_value = [br1, br2, child]
    assert extract_text(dom, sep_symbol='|') == 'A ||B'

    # Test strip artificial newlines
    dom = Mock(tag='div')
    child = Mock(tag='p', text='Content')
    dom.getchildren.return_value = [child]
    assert extract_text(dom, squash_space=True) == 'Content'


# LLM-generated content at query #22
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

    # Test separator element (br)
    br_elem = MockElement('br')
    assert extract_text_array(br_elem) == [True]

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
    assert extract_text_array(elem, squash_artifical_nl=False) == [
        None, 'First', None, None, 'Second', None, None
    ]

    # Test strip_artifical_nl=False
    elem = MockElement('div', children=[
        MockElement('span', text='Content')
    ])
    assert extract_text_array(elem, strip_artifical_nl=False) == [None, 'Content', None]

    # Test with None text and tail
    elem = MockElement('div', text=None, tail=None)
    assert extract_text_array(elem) == [None, None]

    # Test callable tag
    class CallableTag:
        def __call__(self):
            return 'callable'

    callable_elem = MockElement(CallableTag(), text='Text')
    assert extract_text_array(callable_elem) == ['']


# LLM-generated content at query #23
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline element
    inline_dom = MockDom('span', text='Hello World')
    assert extract_text(inline_dom) == 'Hello World'

    # Test block element with squash_space=True
    block_dom = MockDom('div', children=[
        MockDom('p', text='First paragraph'),
        MockDom('p', text='Second paragraph')
    ])
    assert extract_text(block_dom) == 'First paragraph\nSecond paragraph'

    # Test with separators (br tag)
    br_dom = MockDom('div', children=[
        MockDom('span', text='Line 1'),
        MockDom('br'),
        MockDom('span', text='Line 2')
    ])
    assert extract_text(br_dom) == 'Line 1\nLine 2'

    # Test with nested elements
    nested_dom = MockDom('div', children=[
        MockDom('p', text='Outer '),
        MockDom('span', text='inner '),
        MockDom('strong', text='text')
    ])
    assert extract_text(nested_dom) == 'Outer inner text'

    # Test with whitespace handling
    whitespace_dom = MockDom('div', children=[
        MockDom('p', text='  Multiple   spaces  '),
        MockDom('p', text='\tTabs\tand\nnewlines')
    ])
    assert extract_text(whitespace_dom) == 'Multiple spaces\nTabs and newlines'

    # Test with custom symbols
    assert extract_text(block_dom, block_symbol='|', sep_symbol=';') == 'First paragraph;Second paragraph'

    # Test with squash_space=False
    assert extract_text(block_dom, squash_space=False) == '\nFirst paragraph\n\nSecond paragraph\n'

    # Test empty element
    empty_dom = MockDom('div')
    assert extract_text(empty_dom) == ''

    # Test with tail text
    tail_dom = MockDom('div', children=[
        MockDom('span', text='Hello', tail=' World')
    ])
    assert extract_text(tail_dom) == 'Hello World'


# LLM-generated content at query #24
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
    dom = fromstring("<div><p>Hello <span>World</span></p></div>")
    assert extract_text(dom) == "Hello World"

    # Test with whitespace squashing
    dom = fromstring("<p>Hello   \n   World</p>")
    assert extract_text(dom) == "Hello World"

    # Test with custom symbols
    dom = fromstring("<div>Hello</div><div>World</div>")
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == "Hello|World"

    # Test with no squashing
    dom = fromstring("<p>Hello   \n   World</p>")
    assert extract_text(dom, squash_space=False) == "Hello   \n   World"

    # Test with empty dom
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with mixed content
    dom = fromstring("<div>Hello<p>World<br>!</p>Goodbye</div>")
    assert extract_text(dom) == "Hello\nWorld\n!\nGoodbye"

    # Test with script and style tags (should be ignored)
    dom = fromstring("<div>Hello<script>alert('xss')</script><style>body{}</style>World</div>")
    assert extract_text(dom) == "HelloWorld"


# LLM-generated content at query #25
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
    inline_dom = MockElement('span', text='Hello World')
    assert extract_text(inline_dom) == 'Hello World'

    # Test block tag
    block_dom = MockElement('div', text='Hello', children=[
        MockElement('span', text='World')
    ])
    assert extract_text(block_dom) == 'Hello World'

    # Test separator tag
    sep_dom = MockElement('div', children=[
        MockElement('span', text='Hello'),
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(sep_dom) == 'Hello\nWorld'

    # Test with squash_space=True
    squash_dom = MockElement('div', text='  Hello  ', children=[
        MockElement('span', text='  World  ')
    ])
    assert extract_text(squash_dom) == 'Hello World'

    # Test with squash_space=False
    assert extract_text(squash_dom, squash_space=False) == '  Hello  World  '

    # Test custom block and separator symbols
    custom_dom = MockElement('div', children=[
        MockElement('span', text='Hello'),
        MockElement('br'),
        MockElement('span', text='World')
    ])
    assert extract_text(custom_dom, block_symbol='|', sep_symbol=';') == 'Hello;World'

    # Test nested block elements
    nested_dom = MockElement('div', children=[
        MockElement('p', text='First paragraph'),
        MockElement('p', text='Second paragraph')
    ])
    assert extract_text(nested_dom) == 'First paragraph\nSecond paragraph'

    # Test with tail text
    tail_dom = MockElement('div', children=[
        MockElement('span', text='Hello', tail='World')
    ])
    assert extract_text(tail_dom) == 'HelloWorld'

    # Test empty element
    empty_dom = MockElement('div')
    assert extract_text(empty_dom) == ''

    # Test callable tag
    callable_dom = MockElement(lambda: 'div')
    assert extract_text(callable_dom) == ''


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
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

    # Test inline tag with text
    dom = MockDom('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test inline tag with children
    child = MockDom('strong', text='World')
    dom = MockDom('span', children=[child])
    assert extract_text_array(dom) == ['World']

    # Test inline tag with text and children
    child = MockDom('strong', text='World')
    dom = MockDom('span', text='Hello ', children=[child])
    assert extract_text_array(dom) == ['Hello ', 'World']

    # Test inline tag with tail
    child = MockDom('strong', text='World', tail='!')
    dom = MockDom('span', children=[child])
    assert extract_text_array(dom) == ['World', '!']

    # Test block tag
    dom = MockDom('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test separator tag
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test nested block tags
    inner = MockDom('p', text='Inner')
    outer = MockDom('div', children=[inner])
    assert extract_text_array(outer) == [None, None, 'Inner', None, None]

    # Test mixed inline and block tags
    inline = MockDom('span', text='Inline')
    block = MockDom('div', text='Block')
    dom = MockDom('body', children=[inline, block])
    assert extract_text_array(dom) == ['Inline', None, 'Block', None, None]

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

    # Test with None text and tail
    dom = MockDom('span', text=None, tail=None)
    assert extract_text_array(dom) == []

    # Test with multiple children and tails
    child1 = MockDom('span', text='First', tail=' ')
    child2 = MockDom('strong', text='Second', tail='!')
    dom = MockDom('p', children=[child1, child2])
    assert extract_text_array(dom) == ['First', ' ', 'Second', '!']


# LLM-generated content at query #2
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
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, 'World', None]

    # Test strip_artifical_nl=False
    dom = MockDom('div', text='Hello', children=[MockDom('div', text='World')])
    assert extract_text_array(dom, strip_artifical_nl=False) == ['Hello', 'World']

    # Test callable tag
    dom = MockDom(lambda: None)
    assert extract_text_array(dom) == ''

    # Test empty dom
    dom = MockDom('div')
    assert extract_text_array(dom) == [None, None]


# LLM-generated content at query #3
#--------------------------

```python
def test_extract_text_array():
    from lxml.html import fromstring

    # Test with inline tags
    dom = fromstring('<div><span>Hello</span> <strong>World</strong></div>')
    result = extract_text_array(dom)
    assert result == ['Hello', ' ', 'World']

    # Test with block tags
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello', None, None, 'World', None]

    # Test with separator tags
    dom = fromstring('<div>Hello<br/>World</div>')
    result = extract_text_array(dom)
    assert result == ['Hello', True, 'World', None]

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World</span></p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello ', 'World', None, None]

    # Test with squash_artifical_nl=False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'Hello', None, None, 'World', None]

    # Test with empty dom
    dom = fromstring('<div></div>')
    result = extract_text_array(dom)
    assert result == [None, None]

    # Test with text only
    dom = fromstring('<div>Hello World</div>')
    result = extract_text_array(dom)
    assert result == ['Hello World', None]

    # Test with multiple separators
    dom = fromstring('<div>Hello<br/><br/>World</div>')
    result = extract_text_array(dom)
    assert result == ['Hello', True, 'World', None]

    # Test with mixed content
    dom = fromstring('<div><p>Hello<br/>World</p><span>!</span></div>')
    result = extract_text_array(dom)
    assert result == [None, 'Hello', True, 'World', None, '!', None]


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
            self.children = children or []

        def getchildren(self):
            return self.children

    # Test inline tag
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test block tag
    dom = MockDom('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test with children
    child = MockDom('span', text='World')
    dom = MockDom('div', children=[child])
    assert extract_text(dom) == 'World'

    # Test with tail
    child = MockDom('span', text='Hello', tail='World')
    dom = MockDom('div', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test with separators
    child = MockDom('br')
    dom = MockDom('div', children=[child])
    assert extract_text(dom) == '\n'

    # Test squash_space
    dom = MockDom('div', text='  Hello  ')
    assert extract_text(dom, squash_space=True) == 'Hello'

    # Test block_symbol and sep_symbol
    child = MockDom('br')
    dom = MockDom('div', text='Hello', children=[child], tail='World')
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello;|World'

    # Test complex structure
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('br')
    child3 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2, child3])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with None tag (callable)
    dom = MockDom(callable, text='Hello')
    assert extract_text(dom) == ''


# LLM-generated content at query #5
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
    dom = MockDom('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test block tag with squash_space
    dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test separator tag
    dom = MockDom('br')
    assert extract_text(dom, sep_symbol='\n') == '\n'

    # Test nested tags
    child = MockDom('span', text='nested')
    dom = MockDom('div', text='Hello', children=[child], tail='World')
    assert extract_text(dom) == 'Hello nested World'

    # Test with multiple children
    child1 = MockDom('span', text='child1')
    child2 = MockDom('span', text='child2')
    dom = MockDom('div', text='Hello', children=[child1, child2], tail='World')
    assert extract_text(dom) == 'Hello child1 child2 World'

    # Test with squash_space=False
    dom = MockDom('div', text='  Hello  ', children=[], tail='  World  ')
    assert extract_text(dom, squash_space=False) == '  Hello  \n  World  '

    # Test with custom symbols
    dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello|World'

    # Test with None text and tail
    dom = MockDom('div', text=None, children=[], tail=None)
    assert extract_text(dom) == ''

    # Test with empty children
    dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with callable tag
    dom = MockDom(tag=lambda: None)
    assert extract_text(dom) == ''


# LLM-generated content at query #6
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
    inline_elem_with_children = MockElement('span', children=[
        MockElement('strong', text='Hello'),
        MockElement('em', text='World')
    ])
    assert extract_text_array(inline_elem_with_children) == ['Hello', 'World']

    # Test block element with text
    block_elem = MockElement('div', text='Hello')
    assert extract_text_array(block_elem) == [None, 'Hello', None]

    # Test block element with children
    block_elem_with_children = MockElement('div', children=[
        MockElement('p', text='Paragraph 1'),
        MockElement('p', text='Paragraph 2')
    ])
    assert extract_text_array(block_elem_with_children) == [None, 'Paragraph 1', None, None, 'Paragraph 2', None, None]

    # Test separator element (br)
    br_elem = MockElement('br')
    assert extract_text_array(br_elem) == [True]

    # Test mixed content
    mixed_elem = MockElement('div', children=[
        MockElement('span', text='Inline '),
        MockElement('br'),
        MockElement('p', text='Block')
    ])
    assert extract_text_array(mixed_elem) == [None, 'Inline ', True, None, 'Block', None, None]

    # Test with squash_artifical_nl=False and strip_artifical_nl=False
    block_elem_no_squash = MockElement('div', children=[
        MockElement('p', text='Paragraph 1'),
        MockElement('p', text='Paragraph 2')
    ])
    assert extract_text_array(block_elem_no_squash, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Paragraph 1', None, None, 'Paragraph 2', None, None]

    # Test with tail text
    elem_with_tail = MockElement('div', children=[
        MockElement('span', text='Hello', tail='World')
    ])
    assert extract_text_array(elem_with_tail) == [None, 'Hello', 'World', None]

    # Test callable tag
    callable_tag_elem = MockElement(lambda: 'div', text='Hello')
    assert extract_text_array(callable_tag_elem) == ['']


# LLM-generated content at query #7
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tags
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

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
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test 5: With br tag
    child1 = MockDom('span', text='Hello')
    br = MockDom('br')
    child2 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, br, child2])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test 6: Custom block and separator symbols
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello;World'

    # Test 7: Squash space
    dom = MockDom('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test 8: No squash space
    dom = MockDom('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=False) == '  Hello  \n  World  '

    # Test 9: Complex nested structure
    grandchild = MockDom('strong', text='nested')
    child = MockDom('p', text='Some ', children=[grandchild], tail=' text')
    dom = MockDom('div', text='Start ', children=[child], tail=' End')
    assert extract_text(dom) == 'Start Some nested text End'

    # Test 10: Empty dom
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test 11: Callable tag (should return empty string)
    dom = MockDom(lambda: None)
    assert extract_text(dom) == ''


# LLM-generated content at query #8
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline tag
    dom = Mock(tag='span', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom) == ['Hello']

    # Test with a block tag
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test with a separator tag
    dom = Mock(tag='br', text=None, getchildren=lambda: [])
    assert extract_text_array(dom) == [True]

    # Test with nested tags
    child = Mock(tag='span', text='World', getchildren=lambda: [], tail='!')
    parent = Mock(tag='div', text='Hello', getchildren=lambda: [child], tail=None)
    assert extract_text_array(parent) == [None, 'Hello', 'World', '!', None]

    # Test with squash_artifical_nl=False
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None]

    # Test with strip_artifical_nl=False
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with both squash_artifical_nl and strip_artifical_nl False
    dom = Mock(tag='div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with callable tag
    dom = Mock(tag=lambda: 'div', text='Hello', getchildren=lambda: [])
    assert extract_text_array(dom) == ''

    # Test with None text and tail
    dom = Mock(tag='div', text=None, getchildren=lambda: [], tail=None)
    assert extract_text_array(dom) == [None, None]

    # Test with multiple children
    child1 = Mock(tag='span', text='Hello', getchildren=lambda: [], tail=' ')
    child2 = Mock(tag='span', text='World', getchildren=lambda: [], tail='!')
    parent = Mock(tag='div', text=None, getchildren=lambda: [child1, child2], tail=None)
    assert extract_text_array(parent) == [None, 'Hello', ' ', 'World', '!', None]

    # Test with nested block tags
    child = Mock(tag='p', text='World', getchildren=lambda: [], tail=None)
    parent = Mock(tag='div', text='Hello', getchildren=lambda: [child], tail=None)
    assert extract_text_array(parent) == [None, 'Hello', None, 'World', None, None]


# LLM-generated content at query #9
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
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

    # Test 4: Block tags with separators
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test 5: With br tag
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('br')
    child3 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2, child3])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test 6: Squash space
    dom = MockDom('div', text='  Hello   World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test 7: Custom block and sep symbols
    dom = MockDom('div', text='Hello')
    assert extract_text(dom, block_symbol='|', sep_symbol='-') == 'Hello'

    # Test 8: Complex nested structure
    grandchild = MockDom('strong', text='nested')
    child = MockDom('p', text='Some ', children=[grandchild], tail=' text')
    dom = MockDom('div', text='Start', children=[child], tail=' End')
    assert extract_text(dom) == 'Start\nSome nested text\nEnd'

    # Test 9: Empty dom
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test 10: Multiple br tags
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('br')
    child3 = MockDom('br')
    child4 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2, child3, child4])
    assert extract_text(dom) == 'Hello\nWorld'


# LLM-generated content at query #10
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
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
    block_elem = MockElement('div', text='Block text')
    assert extract_text_array(block_elem) == [None, 'Block text', None]

    # Test separator tag
    sep_elem = MockElement('br')
    assert extract_text_array(sep_elem) == [True]

    # Test nested structure
    child1 = MockElement('span', text='Child1')
    child2 = MockElement('div', text='Child2')
    parent = MockElement('body', children=[child1, child2])
    assert extract_text_array(parent) == ['Child1', None, 'Child2', None]

    # Test with tail text
    child = MockElement('span', text='Child', tail=' tail')
    parent = MockElement('div', children=[child])
    assert extract_text_array(parent) == [None, 'Child', ' tail', None]

    # Test squash_artifical_nl=False
    elem = MockElement('div', text='Text')
    assert extract_text_array(elem, squash_artifical_nl=False) == [None, 'Text', None]

    # Test strip_artifical_nl=False
    elem = MockElement('div', text='Text')
    assert extract_text_array(elem, strip_artifical_nl=False) == [None, 'Text', None]

    # Test callable tag
    callable_elem = MockElement(lambda: 'callable')
    assert extract_text_array(callable_elem) == ''

    # Test complex nested structure
    grandchild = MockElement('b', text='Grandchild')
    child = MockElement('p', text='Child ', children=[grandchild], tail=' tail')
    parent = MockElement('div', text='Parent ', children=[child], tail=' end')
    result = extract_text_array(parent)
    assert result == [None, 'Parent ', 'Child ', 'Grandchild', ' tail', ' end', None]


# LLM-generated content at query #11
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test basic inline tag
    dom = fromstring("<div><span>Hello</span> <span>World</span></div>")
    assert extract_text(dom) == "Hello World"

    # Test block tag with squash_space
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test separator tag (br)
    dom = fromstring("<div>Hello<br/>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test nested tags
    dom = fromstring("<div><div>Hello <span>World</span></div></div>")
    assert extract_text(dom) == "Hello World"

    # Test with custom symbols
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == "Hello|World"

    # Test with squash_space=False
    dom = fromstring("<div>  Hello  <span>  World  </span>  </div>")
    assert extract_text(dom, squash_space=False) == "  Hello    World  "

    # Test with whitespace handling
    dom = fromstring("<div><p>Hello</p>  \n  <p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test empty content
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with only whitespace
    dom = fromstring("<div>   \n  \t  </div>")
    assert extract_text(dom) == ""

    # Test mixed content
    dom = fromstring("<div>Hello<br/>  <p>World</p>  </div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with inline and block tags
    dom = fromstring("<div><span>Hello</span><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with multiple separators
    dom = fromstring("<div>Hello<br/><br/>World</div>")
    assert extract_text(dom) == "Hello\n\nWorld"

    # Test with no squash_space and custom symbols
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, squash_space=False, block_symbol='|', sep_symbol='|') == "Hello|World"


# LLM-generated content at query #12
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

    # Test with separators
    dom = fromstring('<div>Hello<br/>World</div>')
    assert extract_text_array(dom) == ['Hello', True, 'World', None]

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text_array(dom) == [None, 'Hello ', 'World', None, None]

    # Test with text and tail
    dom = fromstring('<div>Hello<p>World</p>!</div>')
    assert extract_text_array(dom) == [None, 'Hello', None, 'World', None, '!', None]

    # Test with squash_artifical_nl=False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = fromstring('<div><p>Hello</p><p>World</p></div>')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with empty dom
    dom = fromstring('<div></div>')
    assert extract_text_array(dom) == [None, None]

    # Test with only text
    dom = fromstring('<div>Hello World</div>')
    assert extract_text_array(dom) == ['Hello World', None]

    # Test with callable tag
    class CallableTag:
        def tag(self):
            return 'div'
    dom = CallableTag()
    assert extract_text_array(dom) == ''


# LLM-generated content at query #13
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
    dom = MockDom('div', text='Start', children=[
        MockDom('span', text='Middle'),
        MockDom('br'),
        MockDom('span', text='End')
    ], tail='After')
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'Start', 'Middle', True, 'End', 'After', None]

    # Test separator tag
    dom = MockDom('br')
    assert extract_text_array(dom) == [True]

    # Test nested block tags
    dom = MockDom('div', children=[
        MockDom('p', text='Paragraph', children=[
            MockDom('strong', text='Important')
        ])
    ])
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'Paragraph', 'Important', None, None]

    # Test squash_artifical_nl
    dom = MockDom('div', children=[
        MockDom('div', text='First'),
        MockDom('div', text='Second')
    ])
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, 'First', None, 'Second', None]

    # Test strip_artifical_nl
    dom = MockDom('div', children=[
        MockDom('div', text='Content')
    ])
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ['Content']

    # Test callable tag
    dom = MockDom(tag=lambda: None)
    assert extract_text_array(dom) == ''

    # Test empty dom
    dom = MockDom('div')
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]


# LLM-generated content at query #14
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

    # Test inline tag with text
    dom = MockElement('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test block tag with text
    dom = MockElement('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test nested tags
    child = MockElement('span', text='World')
    dom = MockElement('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test with separator tag
    child1 = MockElement('span', text='Hello')
    br = MockElement('br')
    child2 = MockElement('span', text='World')
    dom = MockElement('div', children=[child1, br, child2])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test with squash_space=True
    dom = MockElement('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'

    # Test with squash_space=False
    dom = MockElement('div', text='  Hello  \n  World  ')
    assert extract_text(dom, squash_space=False) == '  Hello  \n  World  '

    # Test with custom block_symbol and sep_symbol
    child1 = MockElement('span', text='Hello')
    br = MockElement('br')
    child2 = MockElement('span', text='World')
    dom = MockElement('div', children=[child1, br, child2])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'Hello;World'

    # Test with callable tag
    class CallableTagElement:
        def __init__(self):
            self.tag = lambda: 'div'
            self.text = 'Hello'
            self._children = []

        def getchildren(self):
            return self._children

    dom = CallableTagElement()
    assert extract_text(dom) == ''

    # Test with None text and tail
    dom = MockElement('div', text=None, tail=None)
    assert extract_text(dom) == ''

    # Test with multiple levels of nesting
    grandchild = MockElement('strong', text='!')
    child = MockElement('span', text='World', children=[grandchild])
    dom = MockElement('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World!'


# LLM-generated content at query #15
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline tag
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
    dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None, 'World', None]

    # Test with both flags False
    dom = MockDom('div', text='Hello', children=[], tail='World')
    assert extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'Hello', None, 'World', None]

    # Test with callable tag
    dom = MockDom(lambda: 'div', text='Hello')
    assert extract_text_array(dom) == ''

    # Test with None text and tail
    dom = MockDom('div', text=None, children=[], tail=None)
    assert extract_text_array(dom) == [None, None]

    # Test with multiple children
    child1 = MockDom('span', text='Hello')
    child2 = MockDom('span', text='World')
    dom = MockDom('div', children=[child1, child2])
    assert extract_text_array(dom) == [None, 'Hello', 'World', None]


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
    dom = MockElement('span', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 2: Block element with text
    dom = MockElement('div', text='Hello')
    assert extract_text(dom) == 'Hello'

    # Test 3: Nested elements
    child = MockElement('span', text='World')
    dom = MockElement('div', text='Hello', children=[child])
    assert extract_text(dom) == 'Hello World'

    # Test 4: Block element with separators
    br = MockElement('br')
    dom = MockElement('div', children=[br])
    assert extract_text(dom, sep_symbol='\n') == '\n'

    # Test 5: Multiple block elements
    child1 = MockElement('div', text='First')
    child2 = MockElement('div', text='Second')
    dom = MockElement('body', children=[child1, child2])
    assert extract_text(dom) == 'First\nSecond'

    # Test 6: Whitespace squashing
    dom = MockElement('div', text='Hello   World')
    assert extract_text(dom) == 'Hello World'

    # Test 7: Strip artificial newlines
    child = MockElement('div', text='Content')
    dom = MockElement('body', children=[child])
    assert extract_text(dom).strip() == 'Content'

    # Test 8: Custom block and separator symbols
    dom = MockElement('div', text='Test')
    assert extract_text(dom, block_symbol='|', sep_symbol='-') == 'Test'

    # Test 9: Mixed inline and block elements
    span = MockElement('span', text='inline')
    div = MockElement('div', text='block')
    dom = MockElement('body', children=[span, div])
    assert extract_text(dom) == 'inline\nblock'

    # Test 10: Empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''

    # Test 11: Element with tail text
    child = MockElement('span', text='child', tail='tail')
    dom = MockElement('div', children=[child])
    assert extract_text(dom) == 'child tail'

    # Test 12: Preformatted content (no squashing)
    dom = MockElement('pre', text='Hello   World')
    assert extract_text(dom, squash_space=False) == 'Hello   World'

    # Test 13: Multiple separators
    br1 = MockElement('br')
    br2 = MockElement('br')
    dom = MockElement('div', children=[br1, br2])
    assert extract_text(dom, sep_symbol='\n') == '\n\n'

    # Test 14: Complex nested structure
    span1 = MockElement('span', text='A')
    span2 = MockElement('span', text='B')
    div = MockElement('div', children=[span1, span2], tail='C')
    dom = MockElement('body', children=[div])
    assert extract_text(dom) == 'A B\nC'

    # Test 15: Callable tag (should return empty string)
    class CallableTag:
        def __call__(self):
            return 'div'
    dom = MockElement(CallableTag(), text='Hello')
    assert extract_text(dom) == ''


# LLM-generated content at query #17
#--------------------------

```python
def test_extract_text():
    from lxml.html import fromstring

    # Test with simple inline tags
    dom = fromstring("<div><p>Hello <b>World</b></p></div>")
    assert extract_text(dom) == "Hello World"

    # Test with block tags and whitespace squashing
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with separator tags (br)
    dom = fromstring("<div>Hello<br/>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test with nested tags
    dom = fromstring("<div><p>Hello <span>World</span></p></div>")
    assert extract_text(dom) == "Hello World"

    # Test with multiple whitespace
    dom = fromstring("<div><p>Hello   \n  World</p></div>")
    assert extract_text(dom) == "Hello World"

    # Test with custom block and separator symbols
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == "Hello|World"

    # Test with squash_space=False
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, squash_space=False) == "\nHello\n\nWorld\n"

    # Test with empty dom
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test with only whitespace
    dom = fromstring("<div>   \n  </div>")
    assert extract_text(dom) == ""

    # Test with mixed content
    dom = fromstring("<div><p>Hello</p><span>World</span><p>!</p></div>")
    assert extract_text(dom) == "Hello\nWorld\n!"

    # Test with self-closing tags
    dom = fromstring("<div><img src='test.jpg'/><p>Hello</p></div>")
    assert extract_text(dom) == "\nHello"

    # Test with script and style tags (should be treated as block tags)
    dom = fromstring("<div><script>alert('test')</script><p>Hello</p></div>")
    assert extract_text(dom) == "\nHello"

    # Test with preformatted content (should not squash whitespace)
    dom = fromstring("<div><pre>Hello   World</pre></div>")
    assert extract_text(dom, squash_space=False) == "\nHello   World\n"


# LLM-generated content at query #18
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag with text
    dom = MockElement('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test inline tag with children
    child = MockElement('strong', text='World')
    dom = MockElement('span', children=[child])
    assert extract_text_array(dom) == ['World']

    # Test inline tag with text and tail
    dom = MockElement('span', text='Hello', tail='World')
    assert extract_text_array(dom) == ['Hello', 'World']

    # Test block tag (non-inline)
    dom = MockElement('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test separator tag (br)
    dom = MockElement('br')
    assert extract_text_array(dom) == [True]

    # Test nested structure
    inner = MockElement('strong', text='nested')
    middle = MockElement('span', text='middle ', children=[inner], tail=' tail')
    outer = MockElement('div', text='outer ', children=[middle], tail=' end')
    result = extract_text_array(outer)
    assert result == [None, 'outer ', 'middle ', 'nested', ' tail', None, ' end', None]

    # Test squash_artifical_nl=False
    dom = MockElement('div', children=[
        MockElement('div', text='first'),
        MockElement('div', text='second')
    ])
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'first', None, None, 'second', None, None]

    # Test strip_artifical_nl=False
    dom = MockElement('div', children=[
        MockElement('div', text='content')
    ])
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, None, 'content', None, None]

    # Test callable tag
    dom = MockElement(tag=lambda: None)
    assert extract_text_array(dom) == ''

    # Test empty element
    dom = MockElement('div')
    assert extract_text_array(dom) == [None, None]


# LLM-generated content at query #19
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
    assert extract_text_array(dom) == ['Hello ', True, 'World']

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text_array(dom) == [None, 'Hello ', 'World', None]

    # Test with text and tail
    dom = fromstring('<p>Hello <b>World</b>!</p>')
    assert extract_text_array(dom) == ['Hello ', 'World', '!']

    # Test with squash_artifical_nl=False
    dom = fromstring('<div>Hello</div><div>World</div>')
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = fromstring('<div>Hello</div>')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with empty dom
    dom = fromstring('<div></div>')
    assert extract_text_array(dom) == [None, None]

    # Test with callable tag
    class CallableTag:
        tag = lambda: None
    dom = CallableTag()
    assert extract_text_array(dom) == ''

    # Test with None text and tail
    dom = fromstring('<p><b></b></p>')
    assert extract_text_array(dom) == [None, None]


# LLM-generated content at query #20
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

    # Test with separator tag (br)
    dom = fromstring('<p>Hello<br>World</p>')
    assert extract_text_array(dom) == ['Hello', True, 'World']

    # Test with nested tags
    dom = fromstring('<div><p>Hello <span>World</span></p></div>')
    assert extract_text_array(dom) == [None, 'Hello ', 'World', None]

    # Test with text and tail
    dom = fromstring('<p>Hello <b>World</b>!</p>')
    assert extract_text_array(dom) == ['Hello ', 'World', '!']

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
    dom = fromstring('Hello World')
    assert extract_text_array(dom) == ['Hello World']

    # Test with multiple separators
    dom = fromstring('<p>Hello<br><br>World</p>')
    assert extract_text_array(dom) == ['Hello', True, 'World']


# LLM-generated content at query #21
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    class MockDom:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self.children = children or []

        def getchildren(self):
            return self.children

    # Test inline tag
    inline_dom = MockDom('span', text='Hello')
    assert extract_text(inline_dom) == 'Hello'

    # Test block tag
    block_dom = MockDom('div', text='Hello', children=[
        MockDom('span', text='World')
    ])
    assert extract_text(block_dom) == 'Hello World'

    # Test separator tag
    sep_dom = MockDom('div', children=[
        MockDom('span', text='Hello'),
        MockDom('br'),
        MockDom('span', text='World')
    ])
    assert extract_text(sep_dom, sep_symbol='\n') == 'Hello\nWorld'

    # Test squash_space
    space_dom = MockDom('div', text='  Hello  ', children=[
        MockDom('span', text='  World  ')
    ])
    assert extract_text(space_dom, squash_space=True) == 'Hello World'

    # Test nested tags
    nested_dom = MockDom('div', children=[
        MockDom('p', text='Hello', children=[
            MockDom('span', text='World')
        ])
    ])
    assert extract_text(nested_dom) == 'Hello World'

    # Test with tails
    tail_dom = MockDom('div', children=[
        MockDom('span', text='Hello', tail='World')
    ])
    assert extract_text(tail_dom) == 'Hello World'

    # Test empty dom
    empty_dom = MockDom('div')
    assert extract_text(empty_dom) == ''

    # Test with callable tag
    callable_dom = MockDom(tag=lambda: None)
    assert extract_text(callable_dom) == ''


# LLM-generated content at query #22
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

    dom = MockDom('p', children=[
        MockDom('strong', text='Hello'),
        MockDom('br'),
        MockDom('em', text='World')
    ])
    assert extract_text(dom) == 'Hello\nWorld'

    # Test block tags with whitespace
    dom = MockDom('div', children=[
        MockDom('p', text='  First  '),
        MockDom('p', text='  Second  ')
    ])
    assert extract_text(dom) == 'First\nSecond'

    # Test nested tags
    dom = MockDom('div', children=[
        MockDom('ul', children=[
            MockDom('li', text='Item 1'),
            MockDom('li', text='Item 2')
        ])
    ])
    assert extract_text(dom) == 'Item 1\nItem 2'

    # Test with custom symbols
    dom = MockDom('div', children=[
        MockDom('p', text='A'),
        MockDom('p', text='B')
    ])
    assert extract_text(dom, block_symbol='|', sep_symbol=';') == 'A|B'

    # Test with squash_space=False
    dom = MockDom('div', children=[
        MockDom('p', text='  Hello  '),
        MockDom('p', text='  World  ')
    ])
    assert extract_text(dom, squash_space=False) == '  Hello  \n  World  '

    # Test empty dom
    dom = MockDom('div')
    assert extract_text(dom) == ''

    # Test with tail text
    dom = MockDom('div', children=[
        MockDom('span', text='Hello', tail=' World')
    ])
    assert extract_text(dom) == 'Hello World'

    # Test with separator tags
    dom = MockDom('div', children=[
        MockDom('p', text='Line 1'),
        MockDom('br'),
        MockDom('p', text='Line 2')
    ])
    assert extract_text(dom) == 'Line 1\nLine 2'

    # Test with mixed content
    dom = MockDom('div', children=[
        MockDom('p', text='Start '),
        MockDom('strong', text='bold'),
        MockDom(' text '),
        MockDom('em', text='italic'),
        MockDom(' end.')
    ])
    assert extract_text(dom) == 'Start bold text italic end.'

    # Test with callable tag
    def callable_tag():
        return 'callable'
    dom = MockDom(callable_tag)
    assert extract_text(dom) == ''


# LLM-generated content at query #23
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []

        def getchildren(self):
            return self._children

    # Test inline tag with text
    dom = MockElement('span', text='Hello')
    assert extract_text_array(dom) == ['Hello']

    # Test block tag with text
    dom = MockElement('div', text='Hello')
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test separator tag
    dom = MockElement('br')
    assert extract_text_array(dom) == [True]

    # Test nested tags
    child = MockElement('span', text='World')
    dom = MockElement('div', text='Hello', children=[child], tail='!')
    assert extract_text_array(dom) == [None, 'Hello', 'World', '!', None]

    # Test with squash_artifical_nl=False
    dom = MockElement('div', text='Hello', children=[MockElement('div', text='World')])
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None, 'World', None]

    # Test with strip_artifical_nl=False
    dom = MockElement('div', text='Hello')
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test empty element
    dom = MockElement('div')
    assert extract_text_array(dom) == [None, None]

    # Test callable tag
    dom = MockElement(tag=lambda: None)
    assert extract_text_array(dom) == ''

    # Test with multiple children and tails
    child1 = MockElement('span', text='Child1', tail='Tail1')
    child2 = MockElement('span', text='Child2', tail='Tail2')
    dom = MockElement('div', text='Start', children=[child1, child2], tail='End')
    assert extract_text_array(dom) == [None, 'Start', 'Child1', 'Tail1', 'Child2', 'Tail2', 'End', None]


# LLM-generated content at query #24
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

    inline_dom = MockDom('span', text='Hello', children=[
        MockDom('b', text='World', tail='!')
    ])
    assert extract_text_array(inline_dom) == ['Hello', 'World', '!']

    # Test with block tag
    block_dom = MockDom('div', text='Line1', children=[
        MockDom('p', text='Line2', tail='Line3')
    ])
    assert extract_text_array(block_dom) == [None, 'Line1', None, 'Line2', 'Line3', None]

    # Test with separator tag
    sep_dom = MockDom('br')
    assert extract_text_array(sep_dom) == [True]

    # Test with nested tags
    nested_dom = MockDom('div', children=[
        MockDom('p', text='Paragraph', children=[
            MockDom('span', text='Span text')
        ])
    ])
    assert extract_text_array(nested_dom) == [None, 'Paragraph', 'Span text', None]

    # Test with callable tag
    callable_dom = MockDom(lambda: 'div')
    assert extract_text_array(callable_dom) == ''

    # Test squash_artifical_nl=False
    dom = MockDom('div', children=[
        MockDom('div', text='A'),
        MockDom('div', text='B')
    ])
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'A', None, None, 'B', None]

    # Test strip_artifical_nl=False
    dom = MockDom('div', text='\n  ', children=[
        MockDom('span', text='Content')
    ])
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, '\n  ', 'Content', None]

    # Test with None text and tail
    dom = MockDom('div', text=None, children=[
        MockDom('span', text='Text', tail=None)
    ])
    assert extract_text_array(dom) == [None, 'Text', None]


# LLM-generated content at query #25
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag
    dom = type('obj', (object,), {'tag': 'span', 'text': 'Hello', 'getchildren': lambda: [], 'tail': None})()
    assert extract_text_array(dom) == ['Hello']

    # Test with separator tag
    dom = type('obj', (object,), {'tag': 'br', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    assert extract_text_array(dom) == [True]

    # Test with block tag
    dom = type('obj', (object,), {'tag': 'div', 'text': 'Hello', 'getchildren': lambda: [], 'tail': None})()
    assert extract_text_array(dom) == [None, 'Hello', None]

    # Test with nested tags
    child = type('obj', (object,), {'tag': 'span', 'text': 'World', 'getchildren': lambda: [], 'tail': '!'})()
    dom = type('obj', (object,), {'tag': 'div', 'text': 'Hello', 'getchildren': lambda: [child], 'tail': None})()
    assert extract_text_array(dom) == [None, 'Hello', 'World', '!', None]

    # Test with squash_artifical_nl=False
    dom = type('obj', (object,), {'tag': 'div', 'text': 'Hello', 'getchildren': lambda: [], 'tail': None})()
    assert extract_text_array(dom, squash_artifical_nl=False) == [None, 'Hello', None]

    # Test with strip_artifical_nl=False
    dom = type('obj', (object,), {'tag': 'div', 'text': 'Hello', 'getchildren': lambda: [], 'tail': None})()
    assert extract_text_array(dom, strip_artifical_nl=False) == [None, 'Hello', None]

    # Test with callable tag
    dom = type('obj', (object,), {'tag': lambda: 'div', 'text': 'Hello', 'getchildren': lambda: [], 'tail': None})()
    assert extract_text_array(dom) == ''

    # Test with None text and tail
    dom = type('obj', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    assert extract_text_array(dom) == [None, None]

    # Test with multiple children
    child1 = type('obj', (object,), {'tag': 'span', 'text': 'Hello', 'getchildren': lambda: [], 'tail': ' '})()
    child2 = type('obj', (object,), {'tag': 'span', 'text': 'World', 'getchildren': lambda: [], 'tail': '!'})()
    dom = type('obj', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda: [child1, child2], 'tail': None})()
    assert extract_text_array(dom) == [None, 'Hello', ' ', 'World', '!', None]


