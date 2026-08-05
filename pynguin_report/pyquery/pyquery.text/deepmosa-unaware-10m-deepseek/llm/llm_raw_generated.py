####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with block tag
    dom = html.fromstring("<div>Hello</div>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with separator tag
    dom = html.fromstring("<br>")
    assert extract_text_array(dom) == [True]
    
    # Test with nested inline tags
    dom = html.fromstring("<span><b>Hello</b> World</span>")
    assert extract_text_array(dom) == ["Hello", " World"]
    
    # Test with nested block tags
    dom = html.fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text_array(dom)
    assert None in result  # artificial newlines should be present
    
    # Test with text and children
    dom = html.fromstring("<div>Text <span>child</span> tail</div>")
    result = extract_text_array(dom)
    assert "Text " in result
    assert "child" in result
    assert " tail" in result
    
    # Test with empty dom
    dom = html.fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test with callable tag (should return empty string)
    dom = html.fromstring("<div><script>alert('test')</script></div>")
    assert extract_text_array(dom) == []
    
    # Test squash_artifical_nl=False
    dom = html.fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result
    
    # Test strip_artifical_nl=False
    dom = html.fromstring("<div><p>Hello</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # leading artificial newline
    
    # Test with multiple separators
    dom = html.fromstring("<br><br>")
    assert extract_text_array(dom) == [True, True]


# LLM-generated content at query #2
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    dom = html.fragment_fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with block tag
    dom = html.fragment_fromstring("<div>Hello World</div>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with separator tag
    dom = html.fragment_fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested tags
    dom = html.fragment_fromstring("<p>Hello <b>World</b></p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]
    
    # Test with multiple children
    dom = html.fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"]
    
    # Test with text and tail
    dom = html.fragment_fromstring("<p>Hello <b>World</b> again</p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World", " again"]
    
    # Test with empty tag
    dom = html.fragment_fromstring("<span></span>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test with None text
    dom = html.fragment_fromstring("<div><p>Test</p></div>")
    dom.text = None
    result = extract_text_array(dom)
    assert result == ["Test"]
    
    # Test with callable tag (should return empty string)
    class CallableTag:
        def __call__(self):
            pass
    dom = html.fragment_fromstring("<div>Test</div>")
    dom.tag = CallableTag()
    result = extract_text_array(dom)
    assert result == ""


# LLM-generated content at query #3
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml.html import fromstring
    dom = fromstring('<p>Hello World</p>')
    result = extract_text_array(dom)
    assert result == ['Hello World']
    
    # Test with inline tags (should not add artificial newlines)
    dom = fromstring('<span>inline</span>')
    result = extract_text_array(dom)
    assert result == ['inline']
    
    # Test with separator tag (br)
    dom = fromstring('<br>')
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested structure
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    result = extract_text_array(dom)
    assert result == [None, 'First', None, None, 'Second', None]
    
    # Test with child elements and tail text
    dom = fromstring('<div>Start <span>middle</span> End</div>')
    result = extract_text_array(dom)
    assert result == [None, 'Start ', 'middle', ' End', None]
    
    # Test with multiple nested levels
    dom = fromstring('<div><p>Text <b>bold</b> and <i>italic</i></p></div>')
    result = extract_text_array(dom)
    assert result == [None, None, 'Text ', 'bold', ' and ', 'italic', None, None]
    
    # Test with empty element
    dom = fromstring('<div></div>')
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with callable tag (should return empty string)
    class FakeElement:
        tag = lambda: None
    fake_dom = FakeElement()
    result = extract_text_array(fake_dom)
    assert result == ''
    
    # Test with separator and text
    dom = fromstring('<div>Line1<br>Line2</div>')
    result = extract_text_array(dom)
    assert result == [None, 'Line1', True, 'Line2', None]
    
    # Test squash_artifical_nl=False
    dom = fromstring('<div><p>A</p><p>B</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, 'A', None, None, None, 'B', None, None]
    
    # Test strip_artifical_nl=False
    dom = fromstring('<div><p>A</p></div>')
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, None, 'A', None, None]


# LLM-generated content at query #4
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with block element
    dom = fromstring("<div>Hello World</div>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with separator element
    dom = fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested elements
    dom = fromstring("<div>Hello <span>World</span></div>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]
    
    # Test with multiple levels
    dom = fromstring("<div>Line 1<br>Line 2</div>")
    result = extract_text_array(dom)
    assert result == ["Line 1", True, "Line 2"]
    
    # Test with block-level children
    dom = fromstring("<div><p>Para 1</p><p>Para 2</p></div>")
    result = extract_text_array(dom)
    assert result == ["Para 1", "Para 2"]
    
    # Test with whitespace text
    dom = fromstring("<div>   Hello   World   </div>")
    result = extract_text_array(dom)
    assert result == ["   Hello   World   "]
    
    # Test with mixed content
    dom = fromstring("<div>Text <b>bold</b> and <i>italic</i></div>")
    result = extract_text_array(dom)
    assert result == ["Text ", "bold", " and ", "italic"]
    
    # Test empty element
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test with callable tag (should return empty string)
    class MockDom:
        tag = lambda: None
    result = extract_text_array(MockDom())
    assert result == ""
    
    # Test squash_artifical_nl=False
    dom = fromstring("<div>Text</div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result  # Should have artificial newlines
    
    # Test strip_artifical_nl=False
    dom = fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None  # Should have leading/trailing None
    
    # Test complex nested structure
    html = """<div>
        <h1>Title</h1>
        <p>Paragraph with <a href="#">link</a></p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </div>"""
    dom = fromstring(html)
    result = extract_text_array(dom)
    assert "Title" in result
    assert "Paragraph with " in result
    assert "link" in result
    assert "Item 1" in result
    assert "Item 2" in result
    
    # Test with input element (inline)
    dom = fromstring("<div><input type='text'> </div>")
    result = extract_text_array(dom)
    assert " " in result  # whitespace after input
    
    # Test with multiple separators
    dom = fromstring("<div><br><br></div>")
    result = extract_text_array(dom)
    assert result == [True, True]


# LLM-generated content at query #5
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text_array(dom) == ["Hello World"]

    # Test with block-level tag
    dom = html.fromstring("<div>Hello World</div>")
    result = extract_text_array(dom)
    assert result[0] is None  # leading artificial newline
    assert result[1] == "Hello World"
    assert result[2] is None  # trailing artificial newline

    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    assert extract_text_array(dom) == [True]

    # Test with nested tags
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert len([x for x in result if x is None]) >= 4  # artificial newlines

    # Test with text and tail
    dom = html.fromstring("<div>Start<b>bold</b>End</div>")
    result = extract_text_array(dom)
    assert "Start" in result
    assert "bold" in result
    assert "End" in result

    # Test with mixed inline and block elements
    dom = html.fromstring("<div><span>inline</span><p>block</p></div>")
    result = extract_text_array(dom)
    assert result.count(None) >= 2  # at least two artificial newlines

    # Test with callable tag (special case)
    mock_dom = type('MockDom', (), {'tag': lambda: None})()
    assert extract_text_array(mock_dom) == ''

    # Test with multiple br tags
    dom = html.fromstring("<div>Line1<br/>Line2<br/>Line3</div>")
    result = extract_text_array(dom)
    assert result.count(True) == 2  # two separators

    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result.count(None) > 2  # more artificial newlines without squashing

    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # leading artificial newline preserved
    assert result[-1] is None  # trailing artificial newline preserved

    # Test with empty text
    dom = html.fromstring("<div></div>")
    assert extract_text_array(dom) == []

    # Test with only whitespace
    dom = html.fromstring("<div>   </div>")
    assert extract_text_array(dom) == ["   "]


# LLM-generated content at query #6
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    from lxml import html
    doc = html.fromstring("<p>Hello World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test nested inline elements
    doc = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(doc) == "Hello bold world"
    
    # Test block elements create newlines
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test separator elements (br)
    doc = html.fromstring("<p>Line 1<br>Line 2</p>")
    assert extract_text(doc) == "Line 1\nLine 2"
    
    # Test nested structure
    doc = html.fromstring("<div><p>Para 1</p><p>Para <b>2</b></p></div>")
    assert extract_text(doc) == "Para 1\nPara 2"
    
    # Test whitespace handling
    doc = html.fromstring("<p>   Extra   spaces   </p>")
    assert extract_text(doc) == "Extra spaces"
    
    # Test empty element
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test element with only tail text
    doc = html.fromstring("<div>Text<b>bold</b>tail</div>")
    assert extract_text(doc) == "Textboldtail"
    
    # Test custom block symbol
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc, block_symbol=" | ") == "First | Second"
    
    # Test custom separator symbol
    doc = html.fromstring("<p>Line 1<br>Line 2</p>")
    assert extract_text(doc, sep_symbol=" | ") == "Line 1 | Line 2"
    
    # Test squash_space=False
    doc = html.fromstring("<p>   Extra   spaces   </p>")
    assert extract_text(doc, squash_space=False) == "   Extra   spaces   "
    
    # Test inline elements don't create newlines
    doc = html.fromstring("<span>inline</span><span>together</span>")
    assert extract_text(doc) == "inlinetogether"
    
    # Test complex nested structure
    doc = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>First <b>paragraph</b></p>
            <p>Second paragraph<br>with break</p>
        </div>
    """)
    expected = "Title\nFirst paragraph\nSecond paragraph\nwith break"
    assert extract_text(doc) == expected


# LLM-generated content at query #7
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tags
    from lxml import html
    
    # Test basic text extraction
    doc = html.fromstring("<p>Hello World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with nested inline tags
    doc = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with separator tags (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with block-level tags
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with multiple block levels
    doc = html.fromstring("<div><p>Text</p></div>")
    assert extract_text(doc) == "Text"
    
    # Test with whitespace handling
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with empty content
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with mixed content
    doc = html.fromstring("<p>Hello <span>beautiful</span> World</p>")
    assert extract_text(doc) == "Hello beautiful World"
    
    # Test with nested block tags
    doc = html.fromstring("<div><p>First</p><div><p>Second</p></div></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with custom block_symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello   World  </p>")
    result = extract_text(doc, squash_space=False)
    assert '  ' in result  # Should preserve some whitespace
    
    # Test with anchor tags (inline)
    doc = html.fromstring("<p>Visit <a href='test'>link</a> here</p>")
    assert extract_text(doc) == "Visit link here"
    
    # Test with list items
    doc = html.fromstring("<ul><li>Item1</li><li>Item2</li></ul>")
    assert extract_text(doc) == "Item1\nItem2"
    
    # Test with heading tags
    doc = html.fromstring("<h1>Title</h1><p>Content</p>")
    assert extract_text(doc) == "Title\nContent"
    
    # Test with script tags (should be rendered as inline)
    doc = html.fromstring("<p>Text <script>alert('test')</script> more</p>")
    result = extract_text(doc)
    assert "Text" in result
    assert "more" in result
    
    # Test with multiple br tags
    doc = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with deeply nested structure
    doc = html.fromstring("<div><p><span><b>Deep</b></span></p></div>")
    assert extract_text(doc) == "Deep"


# LLM-generated content at query #8
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline tag
    from lxml import html
    dom = html.fromstring('<span>Hello World</span>')
    assert extract_text(dom) == 'Hello World'
    
    # Test with block tag (e.g., div)
    dom = html.fromstring('<div>Hello World</div>')
    assert extract_text(dom) == 'Hello World'
    
    # Test with nested tags
    dom = html.fromstring('<div><p>First paragraph</p><p>Second paragraph</p></div>')
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'
    
    # Test with separator tags (br)
    dom = html.fromstring('<span>Line1<br/>Line2</span>')
    assert extract_text(dom) == 'Line1\nLine2'
    
    # Test with multiple br tags
    dom = html.fromstring('<span>Line1<br/><br/>Line2</span>')
    assert extract_text(dom) == 'Line1\n\nLine2'
    
    # Test with inline tags within block
    dom = html.fromstring('<p>This is <b>bold</b> text</p>')
    assert extract_text(dom) == 'This is bold text'
    
    # Test with nested block tags
    dom = html.fromstring('<div><h1>Title</h1><p>Content</p></div>')
    assert extract_text(dom) == 'Title\nContent'
    
    # Test with whitespace squashing
    dom = html.fromstring('<p>  Too   much   space  </p>')
    assert extract_text(dom) == 'Too much space'
    
    # Test with custom block_symbol
    dom = html.fromstring('<div><p>Para1</p><p>Para2</p></div>')
    assert extract_text(dom, block_symbol=' | ') == 'Para1 | Para2'
    
    # Test with custom sep_symbol
    dom = html.fromstring('<span>Line1<br/>Line2</span>')
    assert extract_text(dom, sep_symbol=' - ') == 'Line1 - Line2'
    
    # Test with squash_space=False
    dom = html.fromstring('<div><p>Para1</p><p>Para2</p></div>')
    result = extract_text(dom, squash_space=False)
    assert 'Para1' in result and 'Para2' in result
    
    # Test empty content
    dom = html.fromstring('<div></div>')
    assert extract_text(dom) == ''
    
    # Test with text and tail text
    dom = html.fromstring('<p>Hello <b>world</b> again</p>')
    assert extract_text(dom) == 'Hello world again'
    
    # Test with multiple levels of nesting
    dom = html.fromstring('<div><ul><li>Item 1</li><li>Item 2</li></ul></div>')
    assert extract_text(dom) == 'Item 1\nItem 2'


# LLM-generated content at query #9
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]

    # Test with a separator tag (br)
    dom = html.fromstring("<br>")
    assert extract_text_array(dom) == [True]

    # Test with a block-level tag
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result[0] is None  # opening None
    assert result[1] == "Text"
    assert result[2] is None  # closing None

    # Test with nested inline tags
    dom = html.fromstring("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]

    # Test with nested block-level tags
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    # Should have None at start and end of each block
    assert result[0] is None  # div start
    assert result[1] is None  # p start
    assert result[2] == "First"
    assert result[3] is None  # p end
    assert result[4] is None  # p start
    assert result[5] == "Second"
    assert result[6] is None  # p end
    assert result[7] is None  # div end

    # Test with tail text
    dom = html.fromstring("<div>Start <span>middle</span> end</div>")
    result = extract_text_array(dom)
    assert result[0] is None  # div start
    assert result[1] == "Start "
    assert result[2] == "middle"
    assert result[3] == " end"
    assert result[4] is None  # div end

    # Test with separator tag inside inline
    dom = html.fromstring("<span>Line1<br>Line2</span>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]

    # Test with callable tag (should return empty string)
    dom = html.fromstring("<div>Test</div>")
    dom.tag = lambda: None
    assert extract_text_array(dom) == ""


# LLM-generated content at query #10
#--------------------------

```python
def test_extract_text():
    # Test simple inline tag
    from xml.etree import ElementTree as ET
    dom = ET.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"

    # Test block tag with newline
    dom = ET.fromstring("<div>Hello</div><div>World</div>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test separator tag (br)
    dom = ET.fromstring("<span>Hello<br/>World</span>")
    assert extract_text(dom) == "Hello\nWorld"

    # Test nested tags
    dom = ET.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"

    # Test whitespace squashing
    dom = ET.fromstring("<span>Hello    World</span>")
    assert extract_text(dom) == "Hello World"

    # Test with custom block and separator symbols
    dom = ET.fromstring("<div>Line1</div><div>Line2<br/>Line3</div>")
    assert extract_text(dom, block_symbol=" | ", sep_symbol=" - ") == "Line1 | Line2 - Line3"

    # Test with squash_space=False
    dom = ET.fromstring("<span>  Hello  </span>")
    assert extract_text(dom, squash_space=False) == "  Hello  "

    # Test empty content
    dom = ET.fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test complex nested structure
    dom = ET.fromstring("""
    <div>
        <h1>Title</h1>
        <p>Paragraph with <strong>bold</strong> text</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2<br/>with line break</li>
        </ul>
    </div>
    """)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with bold text" in result
    assert "Item 1" in result
    assert "Item 2" in result
    assert "with line break" in result

    # Test with non-inline tag inside inline tag
    dom = ET.fromstring("<span><div>Block inside inline</div></span>")
    assert extract_text(dom) == "Block inside inline"

    # Test multiple br tags
    dom = ET.fromstring("<span>Line1<br/><br/>Line2</span>")
    assert extract_text(dom) == "Line1\n\nLine2"

    # Test text at multiple levels
    dom = ET.fromstring("<div>Before <span>Inside</span> After</div>")
    assert extract_text(dom) == "Before Inside After"

    # Test with img tag (inline element)
    dom = ET.fromstring("<p>Text with <img src='test.jpg'/> image</p>")
    assert extract_text(dom) == "Text with  image"

    # Test with script tag (should be removed from output)
    dom = ET.fromstring("<div>Content <script>var x=1;</script> more</div>")
    assert extract_text(dom) == "Content more"

    # Test with script tag having inline display
    dom = ET.fromstring("<div>Content <script>var x=1;</script> more</div>")
    assert extract_text(dom) == "Content more"


# LLM-generated content at query #11
#--------------------------

```python
def test_extract_text():
    # Test simple text extraction
    from lxml import html
    dom = html.fromstring('<p>Hello World</p>')
    assert extract_text(dom) == 'Hello World'
    
    # Test with inline elements
    dom = html.fromstring('<p>Hello <strong>World</strong></p>')
    assert extract_text(dom) == 'Hello World'
    
    # Test with block elements
    dom = html.fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom) == 'First\nSecond'
    
    # Test with separator elements (br)
    dom = html.fromstring('<p>Line1<br/>Line2</p>')
    assert extract_text(dom) == 'Line1\nLine2'
    
    # Test with nested elements
    dom = html.fromstring('<div><p>Text with <span>inline</span> content</p></div>')
    assert extract_text(dom) == 'Text with inline content'
    
    # Test with empty elements
    dom = html.fromstring('<div></div>')
    assert extract_text(dom) == ''
    
    # Test with whitespace
    dom = html.fromstring('<p>  Hello   World  </p>')
    assert extract_text(dom) == 'Hello World'
    
    # Test custom block_symbol
    dom = html.fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom, block_symbol=' ') == 'First Second'
    
    # Test custom sep_symbol
    dom = html.fromstring('<p>Line1<br/>Line2</p>')
    assert extract_text(dom, sep_symbol=' ') == 'Line1 Line2'
    
    # Test with squash_space=False
    dom = html.fromstring('<p>  Hello   World  </p>')
    assert extract_text(dom, squash_space=False) == '  Hello   World  '
    
    # Test with multiple block elements
    dom = html.fromstring('<div><section><p>First</p></section><section><p>Second</p></section></div>')
    assert extract_text(dom) == 'First\nSecond'
    
    # Test with mixed inline and block
    dom = html.fromstring('<p>Start <span>middle</span> end</p>')
    assert extract_text(dom) == 'Start middle end'
    
    # Test with attributes (shouldn't affect extraction)
    dom = html.fromstring('<p class="test">Hello <strong id="strong">World</strong></p>')
    assert extract_text(dom) == 'Hello World'


# LLM-generated content at query #12
#--------------------------

```python
def test_extract_text():
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []
        
        def getchildren(self):
            return self._children

    # Test basic text extraction
    elem = MockElement('p', 'Hello World')
    assert extract_text(elem) == 'Hello World'

    # Test with child elements
    child = MockElement('strong', 'bold')
    elem = MockElement('p', 'This is ', [child], ' text')
    assert extract_text(elem) == 'This is bold text'

    # Test with separators (br)
    br = MockElement('br')
    elem = MockElement('div', 'Line1', [br], 'Line2')
    assert extract_text(elem) == 'Line1\nLine2'

    # Test with block elements
    block = MockElement('div', 'Block content')
    elem = MockElement('div', 'Before', [block], 'After')
    assert extract_text(elem) == 'Before\nBlock content\nAfter'

    # Test with nested elements
    inner = MockElement('span', 'inner')
    outer = MockElement('div', 'Outer ', [inner], ' end')
    assert extract_text(outer) == 'Outer inner end'

    # Test whitespace squashing
    elem = MockElement('p', 'Hello    World')
    assert extract_text(elem) == 'Hello World'

    # Test with empty text
    elem = MockElement('p', None)
    assert extract_text(elem) == ''

    # Test custom separator symbols
    elem = MockElement('div', 'Part1', [MockElement('br')], 'Part2')
    assert extract_text(elem, sep_symbol='|') == 'Part1|Part2'

    elem = MockElement('div', 'Part1', [MockElement('div', 'Inner')], 'Part2')
    assert extract_text(elem, block_symbol='|') == 'Part1|Inner|Part2'

    # Test with multiple nested block elements
    inner_div = MockElement('div', 'Middle')
    outer_div = MockElement('div', 'Start ', [inner_div], ' End')
    assert extract_text(outer_div) == 'Start\nMiddle\nEnd'

    # Test with consecutive separators
    br1 = MockElement('br')
    br2 = MockElement('br')
    elem = MockElement('div', 'Start', [br1, br2], 'End')
    assert extract_text(elem) == 'Start\n\nEnd'

    # Test with non-inline elements
    elem = MockElement('article', 'Content')
    assert extract_text(elem) == 'Content'

    # Test complex nested structure
    strong = MockElement('strong', 'important')
    em = MockElement('em', 'emphasis')
    child = MockElement('span', ' text ', [em], ' tail')
    elem = MockElement('p', 'Some ', [strong, child], ' end')
    assert extract_text(elem) == 'Some important text emphasis tail end'


# LLM-generated content at query #13
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml import etree
    dom = etree.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with block element
    dom = etree.fromstring("<div>Text</div>")
    assert extract_text_array(dom) == ["Text"]
    
    # Test with separator element
    dom = etree.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with nested elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert None in result  # artificial newlines should be present
    assert "First" in result
    assert "Second" in result
    
    # Test with inline element containing tail text
    dom = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert "Hello " in result
    assert "bold" in result
    assert " world" in result
    
    # Test with empty element
    dom = etree.fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test with element containing only whitespace
    dom = etree.fromstring("<div>   </div>")
    assert extract_text_array(dom) == ["   "]


# LLM-generated content at query #14
#--------------------------

```python
def test_extract_text():
    # Test basic inline tag extraction
    from lxml import html
    dom = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with separators (br tag)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with block elements
    dom = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test nested structure
    dom = html.fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"
    
    # Test whitespace handling
    dom = html.fromstring("<p>  Hello    World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with multiple whitespace characters
    dom = html.fromstring("<p>Hello\x20\x09\x0CWorld</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test empty content
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with custom block_symbol
    dom = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test with custom sep_symbol
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol='|') == "Line1|Line2"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>  Hello  World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello  World  "
    
    # Test with inline elements that should not create newlines
    dom = html.fromstring("<span>Hello</span> <span>World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with anchor tag
    dom = html.fromstring("<a href='#'>Click here</a>")
    assert extract_text(dom) == "Click here"
    
    # Test with image tag (self-closing)
    dom = html.fromstring("<p>Text <img src='test.jpg'> more text</p>")
    assert extract_text(dom) == "Text  more text"
    
    # Test with list elements
    dom = html.fromstring("<ul><li>Item 1</li><li>Item 2</li></ul>")
    assert extract_text(dom) == "Item 1\nItem 2"
    
    # Test complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> text</p>
            <ul>
                <li>List item 1</li>
                <li>List item 2</li>
            </ul>
        </div>
    """)
    expected = "Title\nParagraph with bold text\nList item 1\nList item 2"
    assert extract_text(dom) == expected
    
    # Test with pre tag (should preserve whitespace)
    dom = html.fromstring("<pre>  Preserved  whitespace  </pre>")
    assert extract_text(dom) == "  Preserved  whitespace  "
    
    # Test with script tag (should be ignored)
    dom = html.fromstring("<p>Text</p><script>alert('test');</script><p>More</p>")
    assert extract_text(dom) == "Text\nMore"
    
    # Test with multiple separators
    dom = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(dom) == "Line1\n\nLine2"
    
    # Test with deeply nested inline elements
    dom = html.fromstring("<p><b><i>Nested</i></b></p>")
    assert extract_text(dom) == "Nested"
    
    # Test with mixed inline and block elements
    dom = html.fromstring("<div><span>Inline</span><p>Block</p></div>")
    assert extract_text(dom) == "Inline\nBlock"
    
    # Test with text before and after children
    dom = html.fromstring("<p>Before <b>middle</b> after</p>")
    assert extract_text(dom) == "Before middle after"


# LLM-generated content at query #15
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with separators (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with block elements
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test nested elements
    doc = html.fromstring("<div><p>Paragraph <b>bold</b></p></div>")
    assert extract_text(doc) == "Paragraph bold"
    
    # Test with whitespace and newlines in HTML
    doc = html.fromstring("<p>  Hello    \n   world  </p>")
    assert extract_text(doc) == "Hello world"
    
    # Test empty content
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with custom symbols
    doc = html.fromstring("<p>Hello</p><p>World</p>")
    assert extract_text(doc, block_symbol=" | ") == "Hello | World"
    assert extract_text(doc, sep_symbol=" | ") == "Hello\nWorld"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello  "
    
    # Test with anchor tag (inline)
    doc = html.fromstring('<a href="#">Link</a>')
    assert extract_text(doc) == "Link"
    
    # Test complex structure
    doc = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> and <i>italic</i></p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(doc)
    assert "Title" in result
    assert "Paragraph with bold and italic" in result
    assert "Item 1" in result
    assert "Item 2" in result
```


# LLM-generated content at query #16
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]

    # Test with block element
    dom = html.fromstring("<div>Hello</div>")
    assert extract_text_array(dom) == ["Hello"]

    # Test with separator element
    dom = html.fromstring("<br>")
    assert extract_text_array(dom) == [True]

    # Test with nested elements
    dom = html.fromstring("<div><span>Hello</span> <span>World</span></div>")
    result = extract_text_array(dom)
    assert None in result  # artificial newlines from div
    assert "Hello" in result
    assert "World" in result

    # Test with text before and after children
    dom = html.fromstring("<p>Start <b>bold</b> End</p>")
    result = extract_text_array(dom)
    assert "Start " in result
    assert "bold" in result
    assert " End" in result

    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result.count(None) >= 2  # multiple artificial newlines

    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None  # leading/trailing artificial newlines

    # Test with both squash and strip disabled
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert len(result) > 1  # multiple parts including artificial newlines

    # Test with callable tag (should return empty)
    class MockElement:
        tag = lambda: None
    assert extract_text_array(MockElement()) == ""

    # Test empty element
    dom = html.fromstring("<div></div>")
    assert extract_text_array(dom) == []

    # Test element with only whitespace text
    dom = html.fromstring("<div>   </div>")
    assert extract_text_array(dom) == ["   "]


# LLM-generated content at query #17
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction from a simple paragraph
    from lxml import html
    doc = html.fromstring("<p>Hello World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with nested inline elements
    doc = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(doc) == "Hello bold world"
    
    # Test with separator (br tag)
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with block elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with custom block_symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello   World  "
    
    # Test empty document
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with only text
    doc = html.fromstring("Just text")
    assert extract_text(doc) == "Just text"
    
    # Test with nested block elements
    doc = html.fromstring("<div><section><p>Deep</p></section></div>")
    assert extract_text(doc) == "Deep"
    
    # Test multiple separators
    doc = html.fromstring("<p>A<br/>B<br/>C</p>")
    assert extract_text(doc) == "A\nB\nC"
    
    # Test with leading/trailing whitespace
    doc = html.fromstring("  <p>Content</p>  ")
    assert extract_text(doc) == "Content"


# LLM-generated content at query #18
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml import html
    dom = html.fromstring("<span>hello</span>")
    assert extract_text_array(dom) == ["hello"]
    
    # Test with block element
    dom = html.fromstring("<div><p>text</p></div>")
    result = extract_text_array(dom)
    assert result == ["text"]
    
    # Test with separator element (br)
    dom = html.fromstring("<div>line1<br/>line2</div>")
    result = extract_text_array(dom)
    assert True in result  # Contains separator
    
    # Test with nested elements
    dom = html.fromstring("<div><p>hello <b>world</b></p></div>")
    result = extract_text_array(dom)
    assert "hello" in result
    assert "world" in result
    
    # Test with text and tail
    dom = html.fromstring("<p>start <b>bold</b> end</p>")
    result = extract_text_array(dom)
    assert "start " in str(result)
    assert "bold" in str(result)
    assert " end" in str(result)
    
    # Test with None text
    dom = html.fromstring("<div><br/></div>")
    result = extract_text_array(dom)
    assert None in result or True in result
    
    # Test with callable tag
    class FakeElement:
        tag = lambda: None
    assert extract_text_array(FakeElement()) == ""
    
    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>a</p><p>b</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result  # Contains artificial newlines
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div><p>text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None  # May have leading/trailing None
    
    # Test with inline tag containing text
    dom = html.fromstring("<em>italic</em>")
    result = extract_text_array(dom)
    assert result == ["italic"]
    
    # Test with multiple children
    dom = html.fromstring("<ul><li>item1</li><li>item2</li></ul>")
    result = extract_text_array(dom)
    assert "item1" in str(result)
    assert "item2" in str(result)


# LLM-generated content at query #19
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import etree
    dom = etree.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with separator tag (br)
    dom = etree.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with block-level tag
    dom = etree.fromstring("<div>Text</div>")
    assert extract_text_array(dom) == [None, "Text", None]
    
    # Test with nested inline tags
    dom = etree.fromstring("<span><b>Bold</b> text</span>")
    result = extract_text_array(dom)
    assert result == ["Bold", " text"]
    
    # Test with nested block tags
    dom = etree.fromstring("<div><p>Paragraph</p></div>")
    assert extract_text_array(dom) == [None, None, "Paragraph", None, None]
    
    # Test with mixed content
    dom = etree.fromstring("<div>Start <span>middle</span> end</div>")
    result = extract_text_array(dom)
    assert result == [None, "Start ", "middle", " end", None]
    
    # Test with br separator
    dom = etree.fromstring("<div>Line1<br/>Line2</div>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None]
    
    # Test with squash_artifical_nl=False and strip_artifical_nl=False
    dom = etree.fromstring("<div>Text</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Text", None]
    
    # Test with empty element
    dom = etree.fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test with only text
    dom = etree.fromstring("<span>Only text</span>")
    assert extract_text_array(dom) == ["Only text"]
    
    # Test with consecutive block tags
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, "Second", None, None]
    
    # Test with callable tag (should return empty string)
    dom = etree.fromstring("<div>Test</div>")
    dom.tag = lambda: None
    assert extract_text_array(dom) == ''


# LLM-generated content at query #20
#--------------------------

```python
def test_extract_text():
    # Test simple inline elements
    from lxml import html
    doc = html.fromstring("<span>Hello</span>")
    assert extract_text(doc) == "Hello"
    
    # Test with block elements inserting newlines
    doc = html.fromstring("<div>Hello<p>World</p></div>")
    assert extract_text(doc) == "Hello\nWorld"
    
    # Test with separator elements
    doc = html.fromstring("<span>Hello<br/>World</span>")
    assert extract_text(doc) == "Hello\nWorld"
    
    # Test with nested structure
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with text and tail text
    doc = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(doc) == "Hello bold world"
    
    # Test whitespace squashing
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with multiple whitespace characters
    doc = html.fromstring("<p>Hello\t\nWorld</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test empty content
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with only whitespace
    doc = html.fromstring("<div>   </div>")
    assert extract_text(doc) == ""
    
    # Test complex nested structure
    doc = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> text</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(doc)
    expected = "Title\nParagraph with bold text\nItem 1\nItem 2"
    assert result == expected
    
    # Test with custom block symbol
    doc = html.fromstring("<div>Hello<p>World</p></div>")
    assert extract_text(doc, block_symbol=' ') == "Hello World"
    
    # Test with custom separator symbol
    doc = html.fromstring("<span>Hello<br/>World</span>")
    assert extract_text(doc, sep_symbol=' ') == "Hello World"
    
    # Test squash_space=False
    doc = html.fromstring("<p>  Hello   World  </p>")
    result = extract_text(doc, squash_space=False)
    assert "  " in result  # Should preserve some whitespace
    
    # Test with inline elements that shouldn't add newlines
    doc = html.fromstring("<span>Hello <em>emphasized</em> world</span>")
    assert extract_text(doc) == "Hello emphasized world"
    
    # Test with multiple separators
    doc = html.fromstring("<p>Line1<br/><br/>Line2</p>")
    assert extract_text(doc) == "Line1\n\nLine2"
    
    # Test with nested inline elements
    doc = html.fromstring("<p><b><i>Bold and italic</i></b></p>")
    assert extract_text(doc) == "Bold and italic"
```


# LLM-generated content at query #21
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    doc = html.fromstring("<span>Hello</span>")
    assert extract_text_array(doc) == ["Hello"]
    
    # Test with a block element
    doc = html.fromstring("<div>Hello</div>")
    assert extract_text_array(doc) == ["Hello"]
    
    # Test with nested elements
    doc = html.fromstring("<div><span>Hello</span> <span>World</span></div>")
    result = extract_text_array(doc)
    assert result == ["Hello", " World"]
    
    # Test with separator element (br)
    doc = html.fromstring("<span>Hello<br/>World</span>")
    result = extract_text_array(doc)
    assert result == ["Hello", True, "World"]
    
    # Test with block element containing inline elements
    doc = html.fromstring("<div><b>Bold</b> and <i>italic</i></div>")
    result = extract_text_array(doc)
    assert result == ["Bold", " and ", "italic"]
    
    # Test with nested block elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(doc)
    assert result == ["First", "Second"]
    
    # Test with text before and after child elements
    doc = html.fromstring("<div>Start <b>bold</b> End</div>")
    result = extract_text_array(doc)
    assert result == ["Start ", "bold", " End"]
    
    # Test with empty element
    doc = html.fromstring("<div></div>")
    assert extract_text_array(doc) == []
    
    # Test with only whitespace
    doc = html.fromstring("<div>   </div>")
    assert extract_text_array(doc) == ["   "]
    
    # Test squash_artifical_nl parameter
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(doc, squash_artifical_nl=True)
    assert result == ["First", "Second"]
    
    # Test strip_artifical_nl parameter
    doc = html.fromstring("<div><p>First</p></div>")
    result = extract_text_array(doc, strip_artifical_nl=True)
    assert result == ["First"]
    
    # Test with multiple separator elements
    doc = html.fromstring("<span>Line1<br/>Line2<br/>Line3</span>")
    result = extract_text_array(doc)
    assert result == ["Line1", True, "Line2", True, "Line3"]
    
    # Test with deeply nested elements
    doc = html.fromstring("<div><ul><li>Item</li></ul></div>")
    result = extract_text_array(doc)
    assert result == ["Item"]
    
    # Test with callable tag (should return empty string)
    doc = html.fromstring("<div>Test</div>")
    doc.tag = lambda: None  # Make tag callable
    assert extract_text_array(doc) == ""


# LLM-generated content at query #22
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline tag
    from lxml import etree
    dom = etree.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]

    # Test with a block tag
    dom = etree.fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == ["Hello"]
    assert None in result  # artificial newline markers

    # Test with separator tag (br)
    dom = etree.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]

    # Test with nested inline and block tags
    dom = etree.fromstring("<div><span>Hello</span> world</div>")
    result = extract_text_array(dom)
    assert "Hello" in result
    assert " world" in result
    assert None in result

    # Test with text and tail
    dom = etree.fromstring("<p>Hello <b>bold</b> text</p>")
    result = extract_text_array(dom)
    assert "Hello " in result
    assert "bold" in result
    assert " text" in result

    # Test with multiple br tags
    dom = etree.fromstring("<p>Line1<br/>Line2<br/>Line3</p>")
    result = extract_text_array(dom)
    assert result.count(True) == 2
    assert "Line1" in result
    assert "Line2" in result
    assert "Line3" in result

    # Test empty element
    dom = etree.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert len(result) == 0 or result == [None, None]

    # Test with no text content
    dom = etree.fromstring("<div><span></span></div>")
    result = extract_text_array(dom)
    assert None in result


# LLM-generated content at query #23
#--------------------------

```python
def test_extract_text():
    # Create a simple HTML structure for testing
    from lxml import etree
    
    # Test basic inline text
    dom = etree.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test nested inline elements
    dom = etree.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test block elements create newlines
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test separator elements
    dom = etree.fromstring("<p>Text<br/>More</p>")
    assert extract_text(dom) == "Text\nMore"
    
    # Test with custom block_symbol
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "First | Second"
    
    # Test with custom sep_symbol
    dom = etree.fromstring("<p>Text<br/>More</p>")
    assert extract_text(dom, sep_symbol=' ') == "Text More"
    
    # Test squash_space=False
    dom = etree.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test with multiple whitespace
    dom = etree.fromstring("<p>Hello    World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with links
    dom = etree.fromstring("<p>Click <a href='#'>here</a> now</p>")
    assert extract_text(dom) == "Click here now"
    
    # Test with lists
    dom = etree.fromstring("<ul><li>Item 1</li><li>Item 2</li></ul>")
    assert extract_text(dom) == "Item 1\nItem 2"
    
    # Test deeply nested structure
    dom = etree.fromstring("<div><p>Hello <b>bold <i>and italic</i></b> world</p></div>")
    assert extract_text(dom) == "Hello bold and italic world"
    
    # Test with script tag (inline but should be handled)
    dom = etree.fromstring("<p>Text <script>alert('test')</script> more</p>")
    assert extract_text(dom) == "Text alert('test') more"
    
    # Test empty document
    dom = etree.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test single text node
    dom = etree.fromstring("Just text")
    assert extract_text(dom) == "Just text"
    
    # Test with trailing/leading whitespace
    dom = etree.fromstring("  <p>  Hello  </p>  ")
    assert extract_text(dom) == "Hello"
    
    # Test mixed inline and block elements
    dom = etree.fromstring("<div><span>Inline</span><p>Block</p></div>")
    assert extract_text(dom) == "Inline\nBlock"


# LLM-generated content at query #24
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    from lxml.html import fromstring
    dom = fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element
    dom = fromstring("<div>Hello</div><div>World</div>")
    assert extract_text(dom) == "Hello\nWorld"
    
    # Test with nested elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator (br)
    dom = fromstring("<span>Line1<br>Line2</span>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with inline elements inside block
    dom = fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with whitespace squashing
    dom = fromstring("<p>  Hello    World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with empty element
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with text and tail
    dom = fromstring("<div>Hello<b>bold</b>World</div>")
    assert extract_text(dom) == "HelloboldWorld"
    
    # Test with custom block symbol
    dom = fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test with custom separator symbol
    dom = fromstring("<span>Line1<br>Line2</span>")
    assert extract_text(dom, sep_symbol='|') == "Line1|Line2"
    
    # Test with squash_space=False
    dom = fromstring("<p>  Hello    World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello    World  "
    
    # Test with multiple levels of nesting
    dom = fromstring("<div><span><b>Deep</b></span></div>")
    assert extract_text(dom) == "Deep"


# LLM-generated content at query #25
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    div = type('obj', (object,), {'tag': 'div', 'text': 'Hello', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(div) == 'Hello'
    
    # Test with separator (br)
    br = type('obj', (object,), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    div_with_br = type('obj', (object,), {'tag': 'div', 'text': 'Line1', 'tail': None, 'getchildren': lambda: [br]})()
    assert extract_text(div_with_br) == 'Line1\n'
    
    # Test with block element (div inside div)
    inner_div = type('obj', (object,), {'tag': 'div', 'text': 'Inner', 'tail': None, 'getchildren': lambda: []})()
    outer_div = type('obj', (object,), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda: [inner_div]})()
    assert extract_text(outer_div) == 'Inner'
    
    # Test with multiple children
    span = type('obj', (object,), {'tag': 'span', 'text': ' World', 'tail': None, 'getchildren': lambda: []})()
    div = type('obj', (object,), {'tag': 'div', 'text': 'Hello', 'tail': None, 'getchildren': lambda: [span]})()
    assert extract_text(div) == 'Hello World'
    
    # Test with tail text
    br = type('obj', (object,), {'tag': 'br', 'text': None, 'tail': ' tail', 'getchildren': lambda: []})()
    div = type('obj', (object,), {'tag': 'div', 'text': 'Hello', 'tail': None, 'getchildren': lambda: [br]})()
    assert extract_text(div) == 'Hello\ntail'
    
    # Test with custom symbols
    div = type('obj', (object,), {'tag': 'div', 'text': 'Hello', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(div, block_symbol='|', sep_symbol='-') == 'Hello'
    
    # Test squash_space=False
    div = type('obj', (object,), {'tag': 'div', 'text': '  Hello  ', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(div, squash_space=False) == '  Hello  '
    
    # Test with nested blocks
    inner = type('obj', (object,), {'tag': 'div', 'text': 'Inner', 'tail': None, 'getchildren': lambda: []})()
    outer = type('obj', (object,), {'tag': 'div', 'text': 'Start', 'tail': None, 'getchildren': lambda: [inner]})()
    assert extract_text(outer) == 'Start\nInner'
    
    # Test with callable tag (returns empty string)
    div = type('obj', (object,), {'tag': lambda: None, 'text': 'test', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(div) == ''


# LLM-generated content at query #26
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import html
    doc = html.fromstring("<span>Hello World</span>")
    assert extract_text(doc) == "Hello World"
    
    # Test with block element (div)
    doc = html.fromstring("<div>Hello World</div>")
    assert extract_text(doc) == "Hello World"
    
    # Test with nested elements
    doc = html.fromstring("<div><span>Hello</span> <span>World</span></div>")
    assert extract_text(doc) == "Hello World"
    
    # Test with separator (br)
    doc = html.fromstring("Line 1<br>Line 2")
    assert extract_text(doc) == "Line 1\nLine 2"
    
    # Test with multiple separators
    doc = html.fromstring("Line 1<br><br>Line 2")
    assert extract_text(doc) == "Line 1\n\nLine 2"
    
    # Test with block elements creating artificial newlines
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with inline elements within block
    doc = html.fromstring("<p>This is a <strong>bold</strong> text</p>")
    assert extract_text(doc) == "This is a bold text"
    
    # Test with whitespace normalization
    doc = html.fromstring("<p>  Hello    World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with empty element
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with text only
    doc = html.fromstring("Just text")
    assert extract_text(doc) == "Just text"
    
    # Test with nested block elements
    doc = html.fromstring("<div><div>Nested</div></div>")
    assert extract_text(doc) == "Nested"
    
    # Test with multiple children
    doc = html.fromstring("<ul><li>Item 1</li><li>Item 2</li></ul>")
    assert extract_text(doc) == "Item 1\nItem 2"
    
    # Test with custom separators
    doc = html.fromstring("Line 1<br>Line 2")
    assert extract_text(doc, sep_symbol='|') == "Line 1|Line 2"
    
    # Test with custom block symbol
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc, block_symbol='|') == "First|Second"
    
    # Test without squash_space
    doc = html.fromstring("<p>  Hello    World  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello    World  "
    
    # Test with mixed inline and block elements
    doc = html.fromstring("<div><b>Bold</b> text <i>italic</i></div>")
    assert extract_text(doc) == "Bold text italic"
    
    # Test with script tag (should be excluded)
    doc = html.fromstring("<div>Content</div><script>alert('test')</script>")
    assert extract_text(doc) == "Content"
```


# LLM-generated content at query #27
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tags
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with block elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with separator (br)
    doc = html.fromstring("Line1<br>Line2")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with nested inline and block
    doc = html.fromstring("<div><p>Hello <span>world</span></p></div>")
    assert extract_text(doc) == "Hello world"
    
    # Test with whitespace handling
    doc = html.fromstring("<p>  Hello    world  </p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with multiple whitespace and newlines
    doc = html.fromstring("<p>\n  Hello\n  world\n</p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with empty content
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with only whitespace
    doc = html.fromstring("<p>   </p>")
    assert extract_text(doc) == ""
    
    # Test with nested block elements
    doc = html.fromstring("<div><p>First</p><div><p>Second</p></div></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with multiple separators
    doc = html.fromstring("Line1<br><br>Line2")
    assert extract_text(doc) == "Line1\n\nLine2"
    
    # Test with custom symbols
    doc = html.fromstring("<p>Hello</p><p>World</p>")
    assert extract_text(doc, block_symbol=' | ') == "Hello | World"
    
    # Test with custom separator symbol
    doc = html.fromstring("Line1<br>Line2")
    assert extract_text(doc, sep_symbol=' - ') == "Line1 - Line2"
    
    # Test without squashing space
    doc = html.fromstring("<p>  Hello  world  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello  world  "
    
    # Test with mixed inline and block with tail text
    doc = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(doc) == "Hello bold world"
    
    # Test with multiple levels of nesting
    doc = html.fromstring("<div><ul><li>Item 1</li><li>Item 2</li></ul></div>")
    assert extract_text(doc) == "Item 1\nItem 2"


# LLM-generated content at query #28
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]

    # Test with separator element (br)
    dom = html.fromstring("<br/>")
    assert extract_text_array(dom) == [True]

    # Test with block element (div)
    dom = html.fromstring("<div>Text</div>")
    assert extract_text_array(dom) == ["Text"]

    # Test with nested elements
    dom = html.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    result = extract_text_array(dom)
    assert None in result  # Contains artificial newlines
    assert "Para1" in result
    assert "Para2" in result

    # Test with inline element containing text and child
    dom = html.fromstring("<a>Click <b>here</b></a>")
    result = extract_text_array(dom)
    assert result == ["Click ", "here"]

    # Test with separator inside inline
    dom = html.fromstring("<span>Line1<br/>Line2</span>")
    result = extract_text_array(dom)
    assert "Line1" in result
    assert True in result
    assert "Line2" in result

    # Test with text after element (tail)
    dom = html.fromstring("<div><p>Para</p>Tail text</div>")
    result = extract_text_array(dom)
    assert "Para" in result
    assert "Tail text" in result

    # Test with callable tag (returns empty string)
    class FakeElement:
        tag = lambda: None
    dom = FakeElement()
    assert extract_text_array(dom) == ""

    # Test squash_artifical_nl=False
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result.count(None) > 1

    # Test strip_artifical_nl=False
    dom = html.fromstring("<div>Content</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None

    # Test with multiple inline elements
    dom = html.fromstring("<span>First</span><span>Second</span>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"]


# LLM-generated content at query #29
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml import html
    dom = html.fromstring("<span>hello</span>")
    result = extract_text_array(dom)
    assert result == ["hello"]
    
    # Test with block element
    dom = html.fromstring("<div>hello</div>")
    result = extract_text_array(dom)
    assert result == [None, "hello", None]
    
    # Test with separator
    dom = html.fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested elements
    dom = html.fromstring("<div><span>hello</span> world</div>")
    result = extract_text_array(dom)
    assert result == [None, "hello", " world", None]
    
    # Test with multiple children
    dom = html.fromstring("<div><span>hello</span><br><span>world</span></div>")
    result = extract_text_array(dom)
    assert result == [None, "hello", True, "world", None]
    
    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><span>hello</span></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "hello", None, None]
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div>hello</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "hello", None]
    
    # Test with both False
    dom = html.fromstring("<div>hello</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None, "hello", None, None]
    
    # Test with text in parent and children
    dom = html.fromstring("<div>start <span>middle</span> end</div>")
    result = extract_text_array(dom)
    assert result == [None, "start ", "middle", " end", None]
    
    # Test empty element
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with callable tag (should return empty string)
    class MockElement:
        tag = lambda: None
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ""


# LLM-generated content at query #30
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"], f"Expected ['Hello World'], got {result}"

    # Test with block element
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == ["Hello"], f"Expected ['Hello'], got {result}"

    # Test with separator element
    dom = html.fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [], f"Expected [], got {result}"

    # Test with nested elements
    dom = html.fromstring("<div><span>Hello</span><span>World</span></div>")
    result = extract_text_array(dom)
    assert result == ["Hello", "World"], f"Expected ['Hello', 'World'], got {result}"

    # Test with block element containing inline elements
    dom = html.fromstring("<p>This is <b>bold</b> text</p>")
    result = extract_text_array(dom)
    assert result == ["This is ", "bold", " text"], f"Expected ['This is ', 'bold', ' text'], got {result}"

    # Test with separator inside block
    dom = html.fromstring("<div>Line1<br>Line2</div>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"], f"Expected ['Line1', True, 'Line2'], got {result}"

    # Test with None text
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [], f"Expected [], got {result}"

    # Test with multiple levels of nesting
    dom = html.fromstring("<div><p><span>Nested</span></p></div>")
    result = extract_text_array(dom)
    assert result == ["Nested"], f"Expected ['Nested'], got {result}"

    # Test with tail text
    dom = html.fromstring("<div>Start<b>bold</b>End</div>")
    result = extract_text_array(dom)
    assert result == ["Start", "bold", "End"], f"Expected ['Start', 'bold', 'End'], got {result}"

    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, "Hello", None], f"Expected [None, 'Hello', None], got {result}"

    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Hello", None], f"Expected [None, 'Hello', None], got {result}"

    # Test with both artificial nl options disabled
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Hello", None], f"Expected [None, 'Hello', None], got {result}"

    # Test with callable tag (should return empty string)
    class CallableTag:
        def __call__(self):
            pass
    dom = html.fromstring("<div>Test</div>")
    dom.tag = CallableTag()
    result = extract_text_array(dom)
    assert result == "", f"Expected '', got {result}"


# LLM-generated content at query #31
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    from lxml import html
    dom = html.fragment_fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element
    dom = html.fragment_fromstring("<div>Hello<br/>World</div>")
    assert extract_text(dom) == "Hello\nWorld"
    
    # Test with nested elements
    dom = html.fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator element
    dom = html.fragment_fromstring("<span>Line1<br/>Line2</span>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test whitespace squashing
    dom = html.fragment_fromstring("<div>  Hello   World  </div>")
    assert extract_text(dom) == "Hello World"
    
    # Test with multiple whitespace characters
    dom = html.fragment_fromstring("<div>Hello\t\tWorld</div>")
    assert extract_text(dom) == "Hello World"
    
    # Test stripping leading/trailing whitespace
    dom = html.fragment_fromstring("<div>  \n  Hello  \n  </div>")
    assert extract_text(dom) == "Hello"
    
    # Test with empty content
    dom = html.fragment_fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = html.fragment_fromstring("<div>   </div>")
    assert extract_text(dom) == ""
    
    # Test custom block_symbol
    dom = html.fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test custom sep_symbol
    dom = html.fragment_fromstring("<span>Line1<br/>Line2</span>")
    assert extract_text(dom, sep_symbol='|') == "Line1|Line2"
    
    # Test with squash_space=False
    dom = html.fragment_fromstring("<div>  Hello   World  </div>")
    result = extract_text(dom, squash_space=False)
    assert "  " in result  # Should preserve some whitespace
    
    # Test with mixed inline and block elements
    dom = html.fragment_fromstring(
        "<div><span>Inline</span><p>Block</p><b>Bold</b></div>"
    )
    assert extract_text(dom) == "Inline\nBlock\nBold"
    
    # Test with deeply nested structure
    dom = html.fragment_fromstring(
        "<div><div><p>Deep <b>nested</b> text</p></div></div>"
    )
    assert extract_text(dom) == "Deep nested text"


# LLM-generated content at query #32
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    from lxml import html
    doc = html.fromstring("<span>Hello World</span>")
    assert extract_text(doc) == "Hello World"
    
    # Test with block tag
    doc = html.fromstring("<div>Hello</div><div>World</div>")
    assert extract_text(doc) == "Hello\nWorld"
    
    # Test with separator tag
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with nested tags
    doc = html.fromstring("<div><p>Hello <b>World</b></p></div>")
    assert extract_text(doc) == "Hello World"
    
    # Test with whitespace
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with multiple whitespace and newlines
    doc = html.fromstring("<p>Hello\n\n\nWorld</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with inline tags in block tags
    doc = html.fromstring("<div><span>Hello</span> <span>World</span></div>")
    assert extract_text(doc) == "Hello World"
    
    # Test with empty content
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with nested separators
    doc = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with text before and after child
    doc = html.fromstring("<p>Start <b>bold</b> End</p>")
    assert extract_text(doc) == "Start bold End"


# LLM-generated content at query #33
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline tag
    class MockElement:
        tag = 'span'
        text = 'Hello'
        tail = None
        def getchildren(self):
            return []
    result = extract_text_array(MockElement())
    assert result == ['Hello']

    # Test with a separator tag (br)
    class MockBrElement:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []
    result = extract_text_array(MockBrElement())
    assert result == [True]

    # Test with a block-level tag
    class MockDivElement:
        tag = 'div'
        text = 'Text'
        tail = None
        def getchildren(self):
            return []
    result = extract_text_array(MockDivElement())
    assert result == [None, 'Text', None]

    # Test with nested elements
    class MockChildElement:
        tag = 'span'
        text = 'child'
        tail = ' tail'
        def getchildren(self):
            return []
    
    class MockParentElement:
        tag = 'div'
        text = 'parent '
        tail = None
        def getchildren(self):
            return [MockChildElement()]
    
    result = extract_text_array(MockParentElement())
    assert result == [None, 'parent ', 'child', ' tail', None]

    # Test with squash_artifical_nl=False and strip_artifical_nl=False
    class MockDivNoSquash:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text_array(MockDivNoSquash(), squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]

    # Test with multiple consecutive block-level elements
    class MockInnerDiv:
        tag = 'div'
        text = 'inner'
        tail = None
        def getchildren(self):
            return []
    
    class MockOuterDiv:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return [MockInnerDiv(), MockInnerDiv()]
    
    result = extract_text_array(MockOuterDiv())
    # After squash: [None, 'inner', 'inner', None] -> after strip: ['inner', 'inner']
    # But with default squash=True, strip=True
    assert result == ['inner', 'inner']

    # Test with empty element
    class MockEmptyElement:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return []
    result = extract_text_array(MockEmptyElement())
    # After squash: [] -> after strip: []
    assert result == []


# LLM-generated content at query #34
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with separator element (br)
    dom = fragment_fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with block element (div)
    dom = fragment_fromstring("<div>Content</div>")
    result = extract_text_array(dom)
    assert result == [None, "Content", None]
    
    # Test with nested elements
    dom = fragment_fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "Text", None, None]
    
    # Test with inline element inside block
    dom = fragment_fromstring("<div><span>Inline</span> Text</div>")
    result = extract_text_array(dom)
    assert result == [None, "Inline", " Text", None]
    
    # Test with separator inside block
    dom = fragment_fromstring("<div>Line1<br/>Line2</div>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None]
    
    # Test with text and tail
    dom = fragment_fromstring("<div>Start<b>Bold</b>End</div>")
    result = extract_text_array(dom)
    assert result == [None, "Start", "Bold", "End", None]
    
    # Test squash_artifical_nl=False
    dom = fragment_fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "A", None, None, None, "B", None, None]
    
    # Test strip_artifical_nl=False
    dom = fragment_fromstring("<div>Content</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Content", None]
    
    # Test both flags False
    dom = fragment_fromstring("<div>Content</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Content", None]
    
    # Test with empty element
    dom = fragment_fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test with only whitespace
    dom = fragment_fromstring("<div>   </div>")
    result = extract_text_array(dom)
    assert result == [None, "   ", None]
    
    # Test with callable tag (should return empty string)
    class FakeDom:
        tag = lambda: None
    assert extract_text_array(FakeDom()) == ''
    
    # Test with multiple block elements
    dom = fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, "Second", None, None]
    
    # Test with inline elements that are not in INLINE_TAGS
    dom = fragment_fromstring("<custom>Text</custom>")
    result = extract_text_array(dom)
    assert result == [None, "Text", None]


# LLM-generated content at query #35
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline tag
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<span>Hello World</span>")
    assert extract_text_array(dom) == ["Hello World"]
    
    # Test with separator tag (br)
    dom = fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with block tag (div)
    dom = fromstring("<div>Content</div>")
    assert extract_text_array(dom) == ["Content"]
    
    # Test with nested inline tags
    dom = fromstring("<span>Hello <b>World</b></span>")
    assert extract_text_array(dom) == ["Hello ", "World"]
    
    # Test with nested block tags
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text_array(dom) == ["First", None, "Second"]
    
    # Test with separators
    dom = fromstring("<div>Line1<br/>Line2</div>")
    assert extract_text_array(dom) == ["Line1", True, "Line2"]
    
    # Test with text and tail
    dom = fromstring("<div>Start <span>middle</span> end</div>")
    assert extract_text_array(dom) == ["Start ", "middle", " end"]
    
    # Test with squash_artifical_nl=True
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == ["A", None, "B"]
    
    # Test with strip_artifical_nl=True
    dom = fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["A"]
    
    # Test with both squash and strip
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["A", None, "B"]
    
    # Test with empty element
    dom = fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test with callable tag (should return empty string)
    class FakeDom:
        tag = lambda: None
    assert extract_text_array(FakeDom()) == ''
    
    # Test with multiple nested levels
    dom = fromstring("<div><p>Text <span>with <b>bold</b></span> more</p></div>")
    result = extract_text_array(dom)
    assert result == ["Text ", "with ", "bold", " more"]


# LLM-generated content at query #36
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from xml.etree import ElementTree as ET
    html = "<p>Hello <b>world</b></p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Hello world"
    
    # Test with separator (br)
    html = "<p>Line1<br/>Line2</p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with block elements
    html = "<div><p>First</p><p>Second</p></div>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "First\nSecond"
    
    # Test with nested elements
    html = "<div><p>Text with <b>bold</b> and <i>italic</i></p></div>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Text with bold and italic"
    
    # Test with whitespace squashing
    html = "<p>Hello    world</p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Hello world"
    
    # Test with multiple whitespace and newlines
    html = "<p>Hello\n\nworld</p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Hello world"
    
    # Test with empty content
    html = "<p></p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == ""
    
    # Test with only nested elements
    html = "<div><p><b>text</b></p></div>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "text"
    
    # Test with multiple br tags
    html = "<p>Line1<br/><br/>Line2</p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Line1\n\nLine2"
    
    # Test with custom separators
    html = "<p>Hello<br/>world</p>"
    dom = ET.fromstring(html)
    assert extract_text(dom, sep_symbol=" | ") == "Hello | world"
    
    # Test with custom block symbol
    html = "<div><p>First</p><p>Second</p></div>"
    dom = ET.fromstring(html)
    assert extract_text(dom, block_symbol=" | ") == "First | Second"
    
    # Test with squash_space disabled
    html = "<p>  Hello  </p>"
    dom = ET.fromstring(html)
    assert extract_text(dom, squash_space=False) == "  Hello  "
    
    # Test with text in attributes (should not be included)
    html = '<a href="test">Link text</a>'
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Link text"
    
    # Test with script and style elements
    html = "<div><script>var x = 1;</script>Content</div>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Content"
    
    # Test with mixed content including separators and blocks
    html = "<div><p>First line</p><br/><p>Second line</p></div>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "First line\nSecond line"
    
    # Test with leading/trailing whitespace
    html = "  <p>Content</p>  "
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Content"
    
    # Test with inline elements that have their own children
    html = "<p><span>Text <b>bold</b> more</span></p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Text bold more"


# LLM-generated content at query #37
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text node
    class MockElement:
        tag = 'p'
        text = 'Hello World'
        def getchildren(self):
            return []
        def __getitem__(self, key):
            return None
    
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['Hello World']
    
    # Test with separator tag (br)
    class MockBrElement:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
        def __getitem__(self, key):
            return None
    
    dom = MockBrElement()
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with inline tag
    class MockInlineElement:
        tag = 'span'
        text = 'inline text'
        def getchildren(self):
            return []
        def __getitem__(self, key):
            return None
    
    dom = MockInlineElement()
    result = extract_text_array(dom)
    assert result == ['inline text']
    
    # Test with block tag (not inline, not separator)
    class MockBlockElement:
        tag = 'div'
        text = 'block text'
        def getchildren(self):
            return []
        def __getitem__(self, key):
            return None
    
    dom = MockBlockElement()
    result = extract_text_array(dom)
    assert result == ['block text']
    
    # Test with None text
    class MockNoneTextElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
        def __getitem__(self, key):
            return None
    
    dom = MockNoneTextElement()
    result = extract_text_array(dom)
    assert result == []
    
    # Test with child elements
    class MockChildElement:
        tag = 'div'
        text = 'parent '
        class ChildElement:
            tag = 'span'
            text = 'child'
            tail = ' tail'
            def getchildren(self):
                return []
            def __getitem__(self, key):
                return None
        def getchildren(self):
            return [self.ChildElement()]
        def __getitem__(self, key):
            return None
    
    dom = MockChildElement()
    result = extract_text_array(dom)
    assert len(result) > 0


# LLM-generated content at query #38
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with block tag (artificial newlines)
    dom = html.fromstring("<div>Hello</div>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with separator tag
    dom = html.fromstring("<br>")
    assert extract_text_array(dom) == [True]
    
    # Test with nested tags
    dom = html.fromstring("<div><span>Hello</span> World</div>")
    result = extract_text_array(dom)
    assert None in result
    assert "Hello" in result
    assert " World" in result
    
    # Test with multiple children
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result.count(None) >= 2  # artificial newlines around each p tag
    
    # Test with text and tail
    dom = html.fromstring("<div>Start<b>bold</b>End</div>")
    result = extract_text_array(dom)
    assert "Start" in result
    assert "bold" in result
    assert "End" in result
    
    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    consecutive_none = sum(1 for i in range(len(result)-1) 
                          if result[i] is None and result[i+1] is None)
    assert consecutive_none > 0
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div><p>Content</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None  # should have leading/trailing None
    
    # Test with callable tag (should return empty string)
    class FakeDom:
        tag = lambda: None
    assert extract_text_array(FakeDom()) == ''
    
    # Test empty dom
    dom = html.fromstring("<div></div>")
    assert extract_text_array(dom) == []


# LLM-generated content at query #39
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tags (should not add artificial newlines)
    from lxml import etree
    dom = etree.fromstring("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert result == ['Hello ', 'World']
    
    # Test with block-level tags (should add artificial newlines)
    dom = etree.fromstring("<div>First<p>Second</p>Third</div>")
    result = extract_text_array(dom)
    assert None in result  # Should contain artificial newlines
    
    # Test with separator tags
    dom = etree.fromstring("<span>Line1<br/>Line2</span>")
    result = extract_text_array(dom)
    assert True in result  # Should contain separator markers
    
    # Test with nested block and inline tags
    dom = etree.fromstring("<div><span>Text</span><p>Paragraph</p></div>")
    result = extract_text_array(dom)
    # Should have artificial newlines around the paragraph
    assert len([x for x in result if x is None]) >= 2
    
    # Test with text and tail text
    dom = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert 'Hello ' in result
    assert 'bold' in result
    assert ' world' in result
    
    # Test with squash_artifical_nl=False
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    none_count = len([x for x in result if x is None])
    assert none_count > 2  # Multiple artificial newlines without squashing
    
    # Test with strip_artifical_nl=False
    dom = etree.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # Should start with artificial newline
    
    # Test empty element
    dom = etree.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test element with only text
    dom = etree.fromstring("<p>Just text</p>")
    result = extract_text_array(dom)
    assert 'Just text' in result
    
    # Test with custom tag that is not inline
    dom = etree.fromstring("<custom>Text</custom>")
    result = extract_text_array(dom)
    assert None in result
    
    # Test with multiple children and tails
    dom = etree.fromstring("<div>Start <b>bold</b> middle <i>italic</i> end</div>")
    result = extract_text_array(dom)
    assert 'Start ' in result
    assert 'bold' in result
    assert ' middle ' in result
    assert 'italic' in result
    assert ' end' in result


# LLM-generated content at query #40
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml.html import fromstring
    dom = fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"], f"Expected ['Hello'] but got {result}"

    # Test with separator tag (br)
    dom = fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True] but got {result}"

    # Test with block tag
    dom = fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result == [None, "Text", None], f"Expected [None, 'Text', None] but got {result}"

    # Test with nested inline and block tags
    dom = fromstring("<div><span>Hello</span> World</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", " World", None], f"Expected [None, 'Hello', ' World', None] but got {result}"

    # Test with multiple children
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, None, "Second", None, None], \
        f"Expected [None, None, 'First', None, None, None, 'Second', None, None] but got {result}"

    # Test with br separator between text
    dom = fromstring("Line1<br/>Line2")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"], f"Expected ['Line1', True, 'Line2'] but got {result}"

    # Test with empty tag
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None], f"Expected [None, None] but got {result}"

    # Test with text only (no tags)
    dom = fromstring("Just text")
    result = extract_text_array(dom)
    assert result == ["Just text"], f"Expected ['Just text'] but got {result}"

    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "Text", None, None], \
        f"Expected [None, None, 'Text', None, None] but got {result}"

    # Test with strip_artifical_nl=False
    dom = fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Text", None], \
        f"Expected [None, 'Text', None] but got {result}"

    # Test with callable tag (should return empty string)
    def custom_tag():
        pass
    dom = fromstring("<div>Text</div>")
    dom.tag = custom_tag
    result = extract_text_array(dom)
    assert result == '', f"Expected empty string but got {result}"


# LLM-generated content at query #41
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring('<span>Hello World</span>')
    assert extract_text(dom) == 'Hello World'
    
    # Test with block elements - should add newlines
    dom = html.fromstring('<div><p>First paragraph</p><p>Second paragraph</p></div>')
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'
    
    # Test with separator (br tag)
    dom = html.fromstring('<p>Line 1<br>Line 2</p>')
    assert extract_text(dom) == 'Line 1\nLine 2'
    
    # Test with nested inline elements
    dom = html.fromstring('<p>This is <strong>bold</strong> text</p>')
    assert extract_text(dom) == 'This is bold text'
    
    # Test with mixed content
    dom = html.fromstring('<div>Text <span>inline</span> more text</div>')
    assert extract_text(dom) == 'Text inline more text'
    
    # Test whitespace squashing
    dom = html.fromstring('<p>Hello    World</p>')
    assert extract_text(dom) == 'Hello World'
    
    # Test with custom block symbol
    dom = html.fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom, block_symbol='|') == 'First|Second'
    
    # Test with custom separator symbol
    dom = html.fromstring('<p>Line 1<br>Line 2</p>')
    assert extract_text(dom, sep_symbol=' | ') == 'Line 1 | Line 2'
    
    # Test with squash_space=False
    dom = html.fromstring('<p>Hello    World</p>')
    assert extract_text(dom, squash_space=False) == 'Hello    World'
    
    # Test empty element
    dom = html.fromstring('<div></div>')
    assert extract_text(dom) == ''
    
    # Test with only text
    dom = html.fromstring('<p>Just text</p>')
    assert extract_text(dom) == 'Just text'
    
    # Test with multiple nested block elements
    dom = html.fromstring('<div><section><h1>Title</h1><p>Content</p></section></div>')
    assert extract_text(dom) == 'Title\nContent'
    
    # Test with leading/trailing whitespace
    dom = html.fromstring('<p>  Hello World  </p>')
    assert extract_text(dom) == 'Hello World'
    
    # Test with inline elements that are not in INLINE_TAGS
    dom = html.fromstring('<div><custom>text</custom></div>')
    assert extract_text(dom) == 'text'
    
    # Test with multiple br tags
    dom = html.fromstring('<p>Line 1<br><br>Line 2</p>')
    assert extract_text(dom) == 'Line 1\n\nLine 2'
```


# LLM-generated content at query #42
#--------------------------

```python
def test_extract_text_array():
    from lxml import etree
    
    # Test with simple inline tag
    dom = etree.fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"], f"Expected ['Hello World'], got {result}"
    
    # Test with separator tag (br)
    dom = etree.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"
    
    # Test with block-level tag
    dom = etree.fromstring("<div>Content</div>")
    result = extract_text_array(dom)
    assert result == ["Content"], f"Expected ['Content'], got {result}"
    
    # Test with nested inline tags
    dom = etree.fromstring("<span><b>bold</b> text</span>")
    result = extract_text_array(dom)
    assert result == ["bold", " text"], f"Expected ['bold', ' text'], got {result}"
    
    # Test with nested block-level tag
    dom = etree.fromstring("<div><p>Paragraph</p></div>")
    result = extract_text_array(dom)
    assert None in result, f"Expected None in result, got {result}"
    
    # Test with text and tail
    dom = etree.fromstring("<div>Text1<span>inner</span>Text2</div>")
    result = extract_text_array(dom)
    assert "Text1" in result, f"Expected 'Text1' in result, got {result}"
    assert "inner" in result, f"Expected 'inner' in result, got {result}"
    assert "Text2" in result, f"Expected 'Text2' in result, got {result}"
    
    # Test with separator between text
    dom = etree.fromstring("<div>Line1<br/>Line2</div>")
    result = extract_text_array(dom)
    assert True in result, f"Expected True in result, got {result}"
    
    # Test with empty tag
    dom = etree.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [], f"Expected empty list, got {result}"
    
    # Test with only whitespace text
    dom = etree.fromstring("<div>   </div>")
    result = extract_text_array(dom)
    assert result == [], f"Expected empty list after whitespace squash, got {result}"
    
    # Test with multiple block-level tags
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert "First" in result, f"Expected 'First' in result, got {result}"
    assert "Second" in result, f"Expected 'Second' in result, got {result}"


# LLM-generated content at query #43
#--------------------------

```python
def test_extract_text():
    # Test basic paragraph
    from lxml import html
    doc = html.fromstring("<p>Hello world</p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with inline tags
    doc = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(doc) == "Hello bold world"
    
    # Test with separator (br tag)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test nested block elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with whitespace
    doc = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(doc) == "Hello world"
    
    # Test empty content
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test complex nested structure
    doc = html.fromstring("<div><h1>Title</h1><p>Some <b>text</b> here</p></div>")
    assert extract_text(doc) == "Title\nSome text here"
    
    # Test with multiple separators
    doc = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(doc) == "Line1\n\nLine2"
    
    # Test with custom block_symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test squash_space=False
    doc = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello   world  "
    
    # Test with span (inline tag)
    doc = html.fromstring("<p>Hello <span>world</span></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with div containing inline elements
    doc = html.fromstring("<div>Hello <b>bold</b> world</div>")
    assert extract_text(doc) == "Hello bold world"
    
    # Test with nested tags and whitespace
    doc = html.fromstring("<div>  <p>  First  </p>  <p>  Second  </p>  </div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with script tag (should be treated as inline)
    doc = html.fromstring("<div>Text <script>alert('test')</script> more</div>")
    assert extract_text(doc) == "Text more"


# LLM-generated content at query #44
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from xml.etree import ElementTree as ET
    
    # Test basic text extraction
    dom = ET.fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello World", None]
    
    # Test with inline tag (no artificial newlines)
    dom = ET.fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"]
    
    # Test with separator tag (br)
    dom = ET.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested tags
    dom = ET.fromstring("<p>Hello <b>World</b>!</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "World", "!", None]
    
    # Test with multiple levels
    dom = ET.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, None, "Second", None, None]
    
    # Test with br inside text
    dom = ET.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None]
    
    # Test with tail text
    dom = ET.fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "bold", " world", None]
    
    # Test empty element
    dom = ET.fromstring("<p></p>")
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with only inline tags
    dom = ET.fromstring("<span><b>text</b></span>")
    result = extract_text_array(dom)
    assert result == ["text"]
    
    # Test with callable tag (should return empty string)
    mock_dom = type('MockDom', (), {'tag': lambda: None})()
    result = extract_text_array(mock_dom)
    assert result == ''
    
    # Test squash_artifical_nl parameter
    dom = ET.fromstring("<p>Test</p>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, "Test", None]
    
    # Test strip_artifical_nl parameter
    dom = ET.fromstring("<p>Test</p>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Test", None]


# LLM-generated content at query #45
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text
    from lxml import etree
    dom = etree.HTML("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == ['Hello World']
    
    # Test with inline tags
    dom = etree.HTML("<p>Hello <b>bold</b> World</p>")
    result = extract_text_array(dom)
    assert result == ['Hello ', 'bold', ' World']
    
    # Test with separator tag (br)
    dom = etree.HTML("<p>Line1<br>Line2</p>")
    result = extract_text_array(dom)
    assert result == ['Line1', True, 'Line2']
    
    # Test with block-level tags
    dom = etree.HTML("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ['First', 'Second']
    
    # Test with nested inline tags
    dom = etree.HTML("<p><span>Hello <em>World</em></span></p>")
    result = extract_text_array(dom)
    assert result == ['Hello ', 'World']
    
    # Test with empty element
    dom = etree.HTML("<p></p>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test with text in tail
    dom = etree.HTML("<p>Before<b>Bold</b>After</p>")
    result = extract_text_array(dom)
    assert result == ['Before', 'Bold', 'After']


# LLM-generated content at query #46
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello World</p>")
    assert extract_text_array(dom) == [None, "Hello World", None]
    
    # Test with inline element
    dom = etree.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with separator element
    dom = etree.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with nested elements
    dom = etree.fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text_array(dom)
    assert result == ["Hello", None, "World"]
    
    # Test with inline element inside block element
    dom = etree.fromstring("<p>Hello <b>World</b>!</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "World", "!", None]
    
    # Test with text in tail
    dom = etree.fromstring("<div><p>Hello</p>Text after</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", None, "Text after", None]
    
    # Test with empty element
    dom = etree.fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test with multiple separators
    dom = etree.fromstring("<div><br/><br/></div>")
    result = extract_text_array(dom)
    assert result == [True, True]
    
    # Test with artificial newlines squashed
    dom = etree.fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ["Hello", None, "World"]
    
    # Test with artificial newlines stripped
    dom = etree.fromstring("<div><p>Hello</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["Hello"]
    
    # Test with both squashed and stripped
    dom = etree.fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["Hello", None, "World"]


# LLM-generated content at query #47
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline tag
    from lxml import html
    dom = html.fromstring("<span>hello world</span>")
    assert extract_text_array(dom) == ["hello world"]
    
    # Test with separator tag (br)
    dom = html.fromstring("<br>")
    assert extract_text_array(dom) == [True]
    
    # Test with block tag
    dom = html.fromstring("<div>text</div>")
    assert extract_text_array(dom) == [None, "text", None] or extract_text_array(dom) == ["text"]
    
    # Test with nested tags
    dom = html.fromstring("<div><p>first</p><p>second</p></div>")
    result = extract_text_array(dom)
    assert None in result
    assert "first" in result
    assert "second" in result
    
    # Test with text and tail
    dom = html.fromstring("<div>hello <b>bold</b> world</div>")
    result = extract_text_array(dom)
    assert "hello " in result
    assert "bold" in result
    assert " world" in result
    
    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>a</p><p>b</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert len([x for x in result if x is None]) >= 2
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div>text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None


# LLM-generated content at query #48
#--------------------------

```python
def test_extract_text():
    # Test with simple text node
    from lxml import etree
    dom = etree.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with inline tags (should not add newlines)
    dom = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with block-level tags
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator (br tag)
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested inline tags
    dom = etree.fromstring("<p><span>Text</span></p>")
    assert extract_text(dom) == "Text"
    
    # Test with whitespace handling
    dom = etree.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with multiple block levels
    dom = etree.fromstring("<div><p>Para1</p></div>")
    assert extract_text(dom) == "Para1"
    
    # Test with empty content
    dom = etree.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with custom block_symbol
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    assert extract_text(dom, block_symbol=' ') == "A B"
    
    # Test with custom sep_symbol
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    dom = etree.fromstring("<p>  Hello  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello  "
    
    # Test complex nested structure
    dom = etree.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> text</p>
            <p>Another paragraph<br/>with break</p>
        </div>
    """)
    expected = "Title\nParagraph with bold text\nAnother paragraph\nwith break"
    assert extract_text(dom) == expected
    
    # Test with inline tags that are not in INLINE_TAGS (should add newlines)
    dom = etree.fromstring("<custom>Text</custom>")
    assert extract_text(dom) == "Text"
    
    # Test with script tag
    dom = etree.fromstring("<script>var x = 1;</script>")
    assert extract_text(dom) == "var x = 1;"
```


# LLM-generated content at query #49
#--------------------------

```python
def test_extract_text_array():
    # Test with None text and no children
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(MockDom()) == [None, None]

    # Test with inline tag (span)
    class MockSpan:
        tag = 'span'
        text = 'hello'
        def getchildren(self):
            return []
        tail = None
    assert extract_text_array(MockSpan()) == ['hello']

    # Test with separator tag (br)
    class MockBr:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
        tail = None
    assert extract_text_array(MockBr()) == [True]

    # Test with nested structure
    class MockChild:
        tag = 'span'
        text = 'world'
        def getchildren(self):
            return []
        tail = '!'
    
    class MockParent:
        tag = 'div'
        text = 'Hello '
        def getchildren(self):
            return [MockChild()]
        tail = None
    assert extract_text_array(MockParent()) == [None, 'Hello ', 'world', '!', None]

    # Test squash_artifical_nl
    class MockDiv:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(MockDiv(), squash_artifical_nl=True) == [None]
    assert extract_text_array(MockDiv(), squash_artifical_nl=False) == [None, None]

    # Test strip_artifical_nl
    class MockInline:
        tag = 'span'
        text = 'test'
        def getchildren(self):
            return []
        tail = None
    class MockParent2:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockInline()]
        tail = None
    assert extract_text_array(MockParent2(), strip_artifical_nl=True) == ['test']
    assert extract_text_array(MockParent2(), strip_artifical_nl=False) == [None, 'test', None]

    # Test with callable tag
    class MockCallable:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(MockCallable()) == ''


# LLM-generated content at query #50
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with a separator tag (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with a block-level tag
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result == [None, "Text", None]
    
    # Test with nested inline tags
    dom = html.fromstring("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]
    
    # Test with nested block-level tags
    dom = html.fromstring("<div><p>Paragraph 1</p><p>Paragraph 2</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "Paragraph 1", None, None, None, "Paragraph 2", None, None]
    
    # Test with separator inside block
    dom = html.fromstring("<div>Line1<br/>Line2</div>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None]
    
    # Test with squash_artifical_nl=True (default)
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == [None, "Text", None]
    
    # Test with strip_artifical_nl=True (default)
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["Text"]
    
    # Test with both squash and strip
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["Text"]
    
    # Test with callable tag (edge case)
    class MockElement:
        tag = lambda: None  # callable tag
    mock_dom = MockElement()
    result = extract_text_array(mock_dom)
    assert result == ''


# LLM-generated content at query #51
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with block elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with line break
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with nested elements
    doc = html.fromstring("<div><p>Text with <span>span</span> inside</p></div>")
    assert extract_text(doc) == "Text with span inside"
    
    # Test with whitespace handling
    doc = html.fromstring("<p>  Hello    world  </p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with multiple line breaks
    doc = html.fromstring("<p>First<br/><br/>Third</p>")
    assert extract_text(doc) == "First\n\nThird"
    
    # Test with empty content
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with custom block_symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=' | ') == "First | Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with squash_space disabled
    doc = html.fromstring("<p>  Hello    world  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello    world  "
    
    # Test with multiple block elements
    doc = html.fromstring("<div><h1>Title</h1><p>Paragraph</p></div>")
    assert extract_text(doc) == "Title\nParagraph"
    
    # Test with nested block elements
    doc = html.fromstring("<div><section><p>Nested</p></section></div>")
    assert extract_text(doc) == "Nested"
    
    # Test with inline elements that shouldn't create breaks
    doc = html.fromstring("<p><strong>Bold</strong> and <em>italic</em></p>")
    assert extract_text(doc) == "Bold and italic"


# LLM-generated content at query #52
#--------------------------

```python
def test_extract_text_array():
    # Test with None text and no children
    from lxml import etree
    dom = etree.XML("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test with text content in inline tag
    dom = etree.XML("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with separator tag (br)
    dom = etree.XML("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with block tag (non-inline, non-separator)
    dom = etree.XML("<p>Text</p>")
    result = extract_text_array(dom)
    assert None in result
    assert "Text" in result
    
    # Test with nested inline tags
    dom = etree.XML("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert "Hello " in result
    assert "World" in result
    
    # Test with nested block tags
    dom = etree.XML("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert None in result
    assert "First" in result
    assert "Second" in result
    
    # Test with tail text
    dom = etree.XML("<div>Start<b>bold</b>End</div>")
    result = extract_text_array(dom)
    assert "Start" in result
    assert "bold" in result
    assert "End" in result
    
    # Test squash_artifical_nl=False
    dom = etree.XML("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result.count(None) > 1
    
    # Test strip_artifical_nl=False
    dom = etree.XML("<div><p>Text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None
    
    # Test with callable tag (should return empty)
    class FakeTag:
        def __call__(self):
            pass
    
    dom.tag = FakeTag()
    assert extract_text_array(dom) == ""


# LLM-generated content at query #53
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction from a simple element
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test nested inline elements
    dom = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test block elements create newlines
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test separator elements (br)
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test multiple whitespace squashing
    dom = html.fromstring("<p>Hello    World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test whitespace with newlines
    dom = html.fromstring("<p>Hello\n\nWorld</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test empty element
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test element with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test deeply nested structure
    dom = html.fromstring("<div><p><span>Deep</span> <b>nesting</b></p></div>")
    assert extract_text(dom) == "Deep nesting"
    
    # Test custom block_symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test custom sep_symbol
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol='|') == "Line1|Line2"
    
    # Test squash_space=False
    dom = html.fromstring("<p>Hello   World</p>")
    assert extract_text(dom, squash_space=False) == "Hello   World"
    
    # Test mixed inline and block elements
    dom = html.fromstring("<div><h1>Title</h1><p>Paragraph with <a>link</a></p></div>")
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with link" in result
    assert "\n" in result
    
    # Test script and style elements (should be excluded)
    dom = html.fromstring("<div><script>var x = 1;</script><p>Content</p></div>")
    assert extract_text(dom) == "Content"
    
    # Test multiple consecutive block elements
    dom = html.fromstring("<div><p>One</p><p>Two</p><p>Three</p></div>")
    assert extract_text(dom) == "One\nTwo\nThree"
    
    # Test with leading/trailing whitespace in elements
    dom = html.fromstring("<p>  Hello World  </p>")
    assert extract_text(dom) == "Hello World"```


# LLM-generated content at query #54
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with block elements creating newlines
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with separators (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with nested inline elements
    doc = html.fromstring("<p><span>Some <em>emphasized</em> text</span></p>")
    assert extract_text(doc) == "Some emphasized text"
    
    # Test with multiple whitespace
    doc = html.fromstring("<p>Hello    world</p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with leading/trailing whitespace
    doc = html.fromstring("  <p>Hello</p>  ")
    assert extract_text(doc) == "Hello"
    
    # Test with empty content
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with nested block elements
    doc = html.fromstring("<div><div><p>Deep</p></div><p>Text</p></div>")
    assert extract_text(doc) == "Deep\nText"
    
    # Test with custom symbols
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=' | ', sep_symbol=' - ') == "First | Second"
    
    # Test without squashing space
    doc = html.fromstring("<p>Hello    world</p>")
    assert extract_text(doc, squash_space=False) == "Hello    world"
    
    # Test with mixed inline and block elements
    doc = html.fromstring("<div><h1>Title</h1><p>Content <b>bold</b></p></div>")
    assert extract_text(doc) == "Title\nContent bold"
    
    # Test with script tag (should be empty)
    doc = html.fromstring("<div><script>var x = 1;</script><p>Text</p></div>")
    assert extract_text(doc) == "Text"
```


# LLM-generated content at query #55
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element
    dom = html.fromstring("<div><p>First paragraph</p><p>Second paragraph</p></div>")
    assert extract_text(dom) == "First paragraph\nSecond paragraph"
    
    # Test with separator element (br)
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested inline elements
    dom = html.fromstring("<p>This is <strong>bold</strong> and <em>italic</em></p>")
    assert extract_text(dom) == "This is bold and italic"
    
    # Test with whitespace squashing
    dom = html.fromstring("<p>  Multiple   spaces   here  </p>")
    assert extract_text(dom) == "Multiple spaces here"
    
    # Test with custom separators
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "First | Second"
    
    # Test with empty content
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with nested block elements
    dom = html.fromstring("<div><section><p>Deeply nested</p></section></div>")
    assert extract_text(dom) == "Deeply nested"
    
    # Test with mixed inline and block
    dom = html.fromstring("<div><h1>Title</h1><p>Content with <a href='#'>link</a></p></div>")
    result = extract_text(dom)
    assert "Title" in result
    assert "Content with link" in result
    assert "\n" in result
    
    # Test with lists
    dom = html.fromstring("<ul><li>Item 1</li><li>Item 2</li><li>Item 3</li></ul>")
    assert extract_text(dom) == "Item 1\nItem 2\nItem 3"
```


# LLM-generated content at query #56
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text_array(dom) == ["Hello World"]
    
    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with block tag (div)
    dom = html.fromstring("<div>Text</div>")
    assert extract_text_array(dom) == ["Text"]
    
    # Test with nested tags
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"] or result == [None, "First", None, "Second", None]
    
    # Test with mixed inline and block tags
    dom = html.fromstring("<div><span>Inline</span><p>Block</p></div>")
    result = extract_text_array(dom)
    assert "Inline" in result
    assert "Block" in result
    
    # Test with text after child tags (tail)
    dom = html.fromstring("<p>Before <b>bold</b> After</p>")
    result = extract_text_array(dom)
    assert "Before " in result
    assert "bold" in result
    assert " After" in result
    
    # Test with empty tags
    dom = html.fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test with only text
    dom = html.fromstring("Just text")
    assert extract_text_array(dom) == ["Just text"]
    
    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result  # Should have artificial newlines
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert len(result) >= 3  # Should have None at start and end
    
    # Test with callable tag (edge case)
    class MockDom:
        tag = lambda: None
    assert extract_text_array(MockDom()) == ''


# LLM-generated content at query #57
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    dom = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator elements (br)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested inline elements
    dom = html.fromstring("<p><span>Text <em>with</em> <strong>formatting</strong></span></p>")
    assert extract_text(dom) == "Text with formatting"
    
    # Test with multiple whitespace
    dom = html.fromstring("<p>   Hello    world   </p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with empty element
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with nested block elements
    dom = html.fromstring("<div><h1>Title</h1><p>Paragraph</p></div>")
    assert extract_text(dom) == "Title\nParagraph"
    
    # Test with mixed inline and block
    dom = html.fromstring("<div><p>Hello <b>bold</b></p><p>World</p></div>")
    assert extract_text(dom) == "Hello bold\nWorld"
    
    # Test with custom block_symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>   Hello    world   </p>")
    assert extract_text(dom, squash_space=False) == "   Hello    world   "
    
    # Test with complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>This is a <b>bold</b> and <i>italic</i> text</p>
            <br>
            <p>Second paragraph</p>
        </div>
    """)
    assert extract_text(dom) == "Title\nThis is a bold and italic text\nSecond paragraph"
    
    # Test with script tag (should be ignored)
    dom = html.fromstring("<p>Text<script>alert('test')</script>more text</p>")
    assert extract_text(dom) == "Textmore text"
    
    # Test with nested inline elements removing whitespace
    dom = html.fromstring("<p><b>Bold</b> <i>Italic</i></p>")
    assert extract_text(dom) == "Bold Italic"
    
    # Test with multiple br tags
    dom = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(dom) == "Line1\n\nLine2"


# LLM-generated content at query #58
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"]

    # Test with a block element
    dom = fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", None]
    assert result.count(None) == 2

    # Test with separator element (br)
    dom = fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]

    # Test with nested inline elements
    dom = fromstring("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert "Hello" in result
    assert "World" in result

    # Test with block containing inline
    dom = fromstring("<div><span>Hello</span> World</div>")
    result = extract_text_array(dom)
    assert None in result
    assert "Hello" in result
    assert " World" in result

    # Test with nested block elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    none_count = result.count(None)
    assert none_count >= 4  # Each div and p adds None before and after

    # Test with text and tail
    dom = fromstring("<div>Text1<b>Bold</b>Text2</div>")
    result = extract_text_array(dom)
    assert "Text1" in result
    assert "Bold" in result
    assert "Text2" in result

    # Test with squash_artifical_nl=False
    dom = fromstring("<div>Hello</div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result
    assert "Hello" in result

    # Test with strip_artifical_nl=False
    dom = fromstring("<div>Hello</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None

    # Test with empty element
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]

    # Test with multiple br separators
    dom = fromstring("<div>Line1<br/>Line2<br/>Line3</div>")
    result = extract_text_array(dom)
    assert "Line1" in result
    assert "Line2" in result
    assert "Line3" in result
    assert result.count(True) == 2

    # Test with callable tag (should return empty string)
    class FakeDom:
        tag = lambda: None
    fake_dom = FakeDom()
    result = extract_text_array(fake_dom)
    assert result == ''


# LLM-generated content at query #59
#--------------------------

```python
def test_extract_text():
    # Test simple inline element
    from lxml.html import fromstring
    dom = fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(dom) == "Hello world", "Simple inline elements failed"

    # Test block element separation
    dom = fromstring("<div><p>First paragraph</p><p>Second paragraph</p></div>")
    assert extract_text(dom) == "First paragraph\nSecond paragraph", "Block separation failed"

    # Test separator tag (br)
    dom = fromstring("<p>Line 1<br>Line 2</p>")
    assert extract_text(dom) == "Line 1\nLine 2", "BR separator failed"

    # Test whitespace squashing
    dom = fromstring("<p>Hello    world</p>")
    assert extract_text(dom) == "Hello world", "Whitespace squashing failed"

    # Test nested structure
    dom = fromstring("<div><p>Text with <b>bold</b> and <i>italic</i></p></div>")
    assert extract_text(dom) == "Text with bold and italic", "Nested structure failed"

    # Test empty element
    dom = fromstring("<p></p>")
    assert extract_text(dom) == "", "Empty element failed"

    # Test element with only whitespace
    dom = fromstring("<p>   </p>")
    assert extract_text(dom) == "", "Whitespace only element failed"

    # Test multiple block elements
    dom = fromstring("<div><h1>Title</h1><p>Content</p></div>")
    assert extract_text(dom) == "Title\nContent", "Multiple block elements failed"

    # Test with custom block_symbol
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "First | Second", "Custom block_symbol failed"

    # Test with custom sep_symbol
    dom = fromstring("<p>Line 1<br>Line 2</p>")
    assert extract_text(dom, sep_symbol=' | ') == "Line 1 | Line 2", "Custom sep_symbol failed"

    # Test squash_space=False
    dom = fromstring("<p>Hello    world</p>")
    assert extract_text(dom, squash_space=False) == "Hello    world", "squash_space=False failed"

    # Test inline tag without wrapping block
    dom = fromstring("<span>inline text</span>")
    assert extract_text(dom) == "inline text", "Inline tag without block failed"

    # Test complex nested with multiple text nodes
    dom = fromstring("<div>Start <p>Middle <b>bold</b> text</p> End</div>")
    assert extract_text(dom) == "Start\nMiddle bold text\nEnd", "Complex nested failed"


# LLM-generated content at query #60
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    from lxml import html
    doc = html.fromstring("<p>Hello world</p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with nested inline tags
    doc = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(doc) == "Hello bold world"
    
    # Test with separator tag (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with block elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with nested block elements
    doc = html.fromstring("<div><div><p>Deep</p></div></div>")
    assert extract_text(doc) == "Deep"
    
    # Test with whitespace normalization
    doc = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with multiple whitespace and newlines
    doc = html.fromstring("<p>\n  Hello\n  world\n</p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with mixed inline and block elements
    doc = html.fromstring("<div><span>Text</span><p>Paragraph</p></div>")
    assert extract_text(doc) == "Text\nParagraph"
    
    # Test with empty content
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with only whitespace
    doc = html.fromstring("<p>   </p>")
    assert extract_text(doc) == ""
    
    # Test with nested inline in block
    doc = html.fromstring("<div><span>Span</span><span>Another</span></div>")
    assert extract_text(doc) == "SpanAnother"
    
    # Test with custom block_symbol
    doc = html.fromstring("<p>First</p><p>Second</p>")
    assert extract_text(doc, block_symbol=" ") == "First Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc, sep_symbol=" ") == "Line1 Line2"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello   world  </p>")
    result = extract_text(doc, squash_space=False)
    assert "  " in result  # whitespace preserved
    
    # Test with link element
    doc = html.fromstring("<p>Click <a href='test'>here</a> please</p>")
    assert extract_text(doc) == "Click here please"
    
    # Test with list elements
    doc = html.fromstring("<ul><li>Item 1</li><li>Item 2</li></ul>")
    assert extract_text(doc) == "Item 1\nItem 2"
    
    # Test with multiple br tags
    doc = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(doc) == "Line1\n\nLine2"
    
    # Test with br at start
    doc = html.fromstring("<p><br>Start</p>")
    assert extract_text(doc) == "\nStart"
    
    # Test with br at end
    doc = html.fromstring("<p>End<br></p>")
    assert extract_text(doc) == "End\n"
    
    # Test with nested separators
    doc = html.fromstring("<div>Text<br><span>More</span></div>")
    assert extract_text(doc) == "Text\nMore"


# LLM-generated content at query #61
#--------------------------

```python
def test_extract_text_array():
    # Simple text node
    from lxml.html import fromstring
    dom = fromstring("<p>Hello</p>")
    assert extract_text_array(dom) == ["Hello"]

    # Inline tags should not add artificial newlines
    dom = fromstring("<p>Hello <b>world</b></p>")
    result = extract_text_array(dom)
    assert None not in result  # No artificial newlines for inline tags
    assert "Hello" in result
    assert "world" in result

    # Block tags should add artificial newlines
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result[0] is None  # Before first paragraph
    assert result[-1] is None  # After last paragraph

    # Separator tags (br)
    dom = fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert True in result  # br should produce True (separator)

    # Nested structure
    dom = fromstring("<div><p>Text <span>inline</span> content</p></div>")
    result = extract_text_array(dom)
    assert None in result  # Block tags should still produce None
    assert "Text " in result
    assert "inline" in result
    assert " content" in result

    # Empty elements
    dom = fromstring("<p></p>")
    assert extract_text_array(dom) == []

    # Mixed content with text and children
    dom = fromstring("<p>Start <b>bold</b> middle <i>italic</i> end</p>")
    result = extract_text_array(dom)
    assert "Start " in result
    assert "bold" in result
    assert " middle " in result
    assert "italic" in result
    assert " end" in result

    # Strip leading/trailing artificial newlines
    dom = fromstring("<div><p>Content</p></div>")
    result = extract_text_array(dom)
    assert result[0] is not None  # Leading None should be stripped
    assert result[-1] is not None  # Trailing None should be stripped

    # Squash artificial newlines
    dom = fromstring("<div><p>Content</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    # Should not have consecutive None values
    none_count = sum(1 for x in result if x is None)
    assert none_count <= 1

    # Multiple artificial newlines should be squashed
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    none_count = sum(1 for x in result if x is None)
    assert none_count <= 1

    # Test with callable tag (should return empty string)
    class MockElement:
        def __init__(self):
            self.tag = lambda: None
    mock_dom = MockElement()
    assert extract_text_array(mock_dom) == ''


# LLM-generated content at query #62
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b>!</p>")
    assert extract_text(doc) == "Hello world!"
    
    # Test with line break separator
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with block elements creating newlines
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with nested block elements
    doc = html.fromstring("<div><div><p>Deep</p></div></div>")
    assert extract_text(doc) == "Deep"
    
    # Test with multiple whitespace
    doc = html.fromstring("<p>Hello   world</p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with leading/trailing whitespace
    doc = html.fromstring("<p>  Hello  </p>")
    assert extract_text(doc) == "Hello"
    
    # Test with empty document
    doc = html.fromstring("<html></html>")
    assert extract_text(doc) == ""
    
    # Test with only inline elements
    doc = html.fromstring("<span>Hello</span><span>World</span>")
    assert extract_text(doc) == "HelloWorld"
    
    # Test with custom block_symbol
    doc = html.fromstring("<p>First</p><p>Second</p>")
    assert extract_text(doc, block_symbol='|') == "First|Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc, sep_symbol='|') == "Line1|Line2"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello   world  </p>")
    result = extract_text(doc, squash_space=False)
    assert "  " in result or result == "  Hello   world  "
    
    # Test with pre tag (should preserve whitespace)
    doc = html.fromstring("<pre>Hello\n  World</pre>")
    assert extract_text(doc) == "Hello\n  World"
    
    # Test with script tag (should be empty)
    doc = html.fromstring("<div>Text<script>alert('test')</script>More</div>")
    result = extract_text(doc)
    assert "alert" not in result
    assert "Text" in result
    assert "More" in result
    
    # Test with img tag (self-closing, no text)
    doc = html.fromstring("<p>Text<img src='test.jpg'>More</p>")
    assert extract_text(doc) == "TextMore"
    
    # Test with nested inline and block elements
    doc = html.fromstring("<div><p><b>Bold</b> text</p><p>Next</p></div>")
    assert extract_text(doc) == "Bold text\nNext"
    
    # Test with whitespace only content
    doc = html.fromstring("<p>   </p>")
    assert extract_text(doc) == ""
    
    # Test with multiple nested block elements and text
    doc = html.fromstring("<div>Start<p>Middle</p>End</div>")
    assert extract_text(doc) == "Start\nMiddle\nEnd"
    
    # Test with list elements
    doc = html.fromstring("<ul><li>Item1</li><li>Item2</li></ul>")
    result = extract_text(doc)
    assert "Item1" in result
    assert "Item2" in result
    assert result.count('\n') >= 1
    
    # Test with anchor elements (inline)
    doc = html.fromstring("<p><a href='#'>Link</a> text</p>")
    assert extract_text(doc) == "Link text"
    
    # Test with comment nodes (should be ignored)
    doc = html.fromstring("<div>Text<!-- comment -->More</div>")
    assert extract_text(doc) == "TextMore"
```


# LLM-generated content at query #63
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element
    dom = html.fromstring("<div>Hello World</div>")
    assert extract_text(dom) == "Hello World"
    
    # Test with nested elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator (br)
    dom = html.fromstring("<span>Line1<br/>Line2</span>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with whitespace squashing
    dom = html.fromstring("<div>  Hello   World  </div>")
    assert extract_text(dom) == "Hello World"
    
    # Test with custom block_symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test with custom sep_symbol
    dom = html.fromstring("<span>Line1<br/>Line2</span>")
    assert extract_text(dom, sep_symbol='|') == "Line1|Line2"
    
    # Test with squash_space=False
    dom = html.fromstring("<div>  Hello   World  </div>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test with empty element
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with text only
    dom = html.fromstring("Just text")
    assert extract_text(dom) == "Just text"
    
    # Test with mixed inline and block
    dom = html.fromstring("<div><span>Inline</span><p>Block</p></div>")
    assert extract_text(dom) == "Inline\nBlock"
    
    # Test with nested separators
    dom = html.fromstring("<div>Text<br/><br/>More text</div>")
    assert extract_text(dom) == "Text\n\nMore text"


# LLM-generated content at query #64
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from xml.etree.ElementTree import Element, SubElement
    
    # Test 1: Simple text element
    div = Element('div')
    div.text = "Hello World"
    result = extract_text_array(div)
    assert result == ["Hello World"]
    
    # Test 2: Element with child
    div = Element('div')
    span = SubElement(div, 'span')
    span.text = "Hello"
    span.tail = " World"
    result = extract_text_array(div)
    assert result == [None, "Hello", " World", None]
    
    # Test 3: Inline tag (should not add None separators)
    span = Element('span')
    span.text = "Hello"
    result = extract_text_array(span)
    assert result == ["Hello"]
    
    # Test 4: Separator tag (br)
    br = Element('br')
    result = extract_text_array(br)
    assert result == [True]
    
    # Test 5: Nested elements
    div = Element('div')
    p = SubElement(div, 'p')
    p.text = "First"
    br = SubElement(p, 'br')
    br.tail = "Second"
    result = extract_text_array(div)
    assert result == [None, "First", True, "Second", None]
    
    # Test 6: Empty element
    div = Element('div')
    result = extract_text_array(div)
    assert result == [None, None]
    
    # Test 7: Element with only whitespace text
    div = Element('div')
    div.text = "   "
    result = extract_text_array(div)
    assert result == [None, "   ", None]
    
    # Test 8: Nested inline elements
    div = Element('div')
    span = SubElement(div, 'span')
    span.text = "Hello"
    strong = SubElement(span, 'strong')
    strong.text = "World"
    strong.tail = "!"
    result = extract_text_array(div)
    assert result == [None, "Hello", "World", "!", None]
    
    # Test 9: Multiple children
    div = Element('div')
    p1 = SubElement(div, 'p')
    p1.text = "First"
    p2 = SubElement(div, 'p')
    p2.text = "Second"
    result = extract_text_array(div)
    assert result == [None, "First", None, None, "Second", None]
    
    # Test 10: Squash artificial newlines
    div = Element('div')
    p = SubElement(div, 'p')
    p.text = "Text"
    result = extract_text_array(div, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == ["Text"]
    
    # Test 11: Strip artificial newlines
    div = Element('div')
    div.text = "Start"
    span = SubElement(div, 'span')
    span.text = "Middle"
    span.tail = "End"
    result = extract_text_array(div, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["Start", "Middle", "End"]
    
    # Test 12: Both squash and strip
    div = Element('div')
    p = SubElement(div, 'p')
    p.text = "Content"
    result = extract_text_array(div, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["Content"]


# LLM-generated content at query #65
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element
    dom = html.fromstring("<div>Hello</div>")
    assert extract_text(dom) == "Hello"
    
    # Test with nested elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator (br tag)
    dom = html.fromstring("<span>Line1<br/>Line2</span>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with inline elements inside block
    dom = html.fromstring("<p>This is <strong>bold</strong> text</p>")
    assert extract_text(dom) == "This is bold text"
    
    # Test with multiple block elements
    dom = html.fromstring("<div><h1>Title</h1><p>Paragraph</p></div>")
    assert extract_text(dom) == "Title\nParagraph"
    
    # Test with whitespace consolidation
    dom = html.fromstring("<p>Hello    World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with custom block_symbol
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "A | B"
    
    # Test with custom sep_symbol
    dom = html.fromstring("<span>A<br/>B</span>")
    assert extract_text(dom, sep_symbol=' | ') == "A | B"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>  Hello  World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello  World  "
    
    # Test empty element
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with attributes (shouldn't affect text extraction)
    dom = html.fromstring('<a href="http://example.com">Click here</a>')
    assert extract_text(dom) == "Click here"
    
    # Test with list items
    dom = html.fromstring("<ul><li>Item 1</li><li>Item 2</li></ul>")
    assert extract_text(dom) == "Item 1\nItem 2"
    
    # Test with complex nested structure
    dom = html.fromstring("""
        <div>
            <h2>Section</h2>
            <p>Paragraph with <strong>bold</strong> and <em>italic</em></p>
            <ul>
                <li>First item</li>
                <li>Second item</li>
            </ul>
        </div>
    """)
    expected = "Section\nParagraph with bold and italic\nFirst item\nSecond item"
    assert extract_text(dom) == expected
    
    # Test with multiple consecutive blocks
    dom = html.fromstring("<div><p>A</p><p>B</p><p>C</p></div>")
    assert extract_text(dom) == "A\nB\nC"
    
    # Test with inline element containing block element (should still work)
    dom = html.fromstring("<span><p>Nested</p></span>")
    assert extract_text(dom) == "Nested"
    
    # Test with text before and after child elements
    dom = html.fromstring("<p>Start <strong>middle</strong> end</p>")
    assert extract_text(dom) == "Start middle end"


# LLM-generated content at query #66
#--------------------------

```python
def test_extract_text():
    # Test simple inline elements
    dom = type('Mock', (), {'tag': 'span', 'text': 'Hello', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(dom) == 'Hello'
    
    # Test separator elements (br)
    br_dom = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(br_dom) == '\n'
    
    # Test nested inline elements
    child = type('Mock', (), {'tag': 'b', 'text': 'World', 'tail': None, 'getchildren': lambda: []})()
    parent = type('Mock', (), {'tag': 'span', 'text': 'Hello ', 'tail': None, 'getchildren': lambda: [child]})()
    assert extract_text(parent) == 'Hello World'
    
    # Test block elements with artificial newlines
    child = type('Mock', (), {'tag': 'span', 'text': 'World', 'tail': None, 'getchildren': lambda: []})()
    parent = type('Mock', (), {'tag': 'div', 'text': 'Hello ', 'tail': None, 'getchildren': lambda: [child]})()
    assert extract_text(parent) == 'Hello World'
    
    # Test multiple children with separators
    child1 = type('Mock', (), {'tag': 'span', 'text': 'First', 'tail': None, 'getchildren': lambda: []})()
    child2 = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    child3 = type('Mock', (), {'tag': 'span', 'text': 'Second', 'tail': None, 'getchildren': lambda: []})()
    parent = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda: [child1, child2, child3]})()
    assert extract_text(parent) == 'First\nSecond'
    
    # Test whitespace squashing
    dom = type('Mock', (), {'tag': 'span', 'text': 'Hello    World', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(dom) == 'Hello World'
    
    # Test empty elements
    dom = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(dom) == ''
    
    # Test custom symbols
    dom = type('Mock', (), {'tag': 'div', 'text': 'Hello', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(dom, block_symbol='|', sep_symbol='-') == 'Hello'


# LLM-generated content at query #67
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    from lxml import html
    doc = html.fromstring("<p>Hello world</p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with nested elements
    doc = html.fromstring("<div><p>First paragraph</p><p>Second paragraph</p></div>")
    assert extract_text(doc) == "First paragraph\nSecond paragraph"
    
    # Test with separator tags (br)
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with inline tags
    doc = html.fromstring("<p>This is <strong>bold</strong> and <em>italic</em></p>")
    assert extract_text(doc) == "This is bold and italic"
    
    # Test with whitespace handling
    doc = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with multiple newlines
    doc = html.fromstring("<div><p>First</p><p>Second</p><p>Third</p></div>")
    assert extract_text(doc) == "First\nSecond\nThird"
    
    # Test with empty content
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with nested inline elements
    doc = html.fromstring("<p><span>Hello <strong>World</strong></span></p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with custom symbols
    doc = html.fromstring("<div><p>A</p><p>B</p></div>")
    assert extract_text(doc, block_symbol='|') == "A|B"
    
    # Test with separator custom symbol
    doc = html.fromstring("<p>A<br/>B</p>")
    assert extract_text(doc, sep_symbol='|') == "A|B"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello   world  "
    
    # Test with mixed inline and block elements
    doc = html.fromstring("<div><p>Text with <a href='#'>link</a> inside</p></div>")
    assert extract_text(doc) == "Text with link inside"
    
    # Test with pre tag (should preserve whitespace)
    doc = html.fromstring("<pre>  Preserved   whitespace  </pre>")
    assert extract_text(doc) == "  Preserved   whitespace  "
    
    # Test with nested block elements
    doc = html.fromstring("<div><div><p>Deeply</p></div><div><p>Nested</p></div></div>")
    assert extract_text(doc) == "Deeply\nNested"


# LLM-generated content at query #68
#--------------------------

```python
def test_extract_text_array():
    # Test with None input
    from lxml import html
    doc = html.fromstring("<p></p>")
    result = extract_text_array(doc)
    assert result == []

    # Test with simple text
    doc = html.fromstring("<p>Hello</p>")
    result = extract_text_array(doc)
    assert result == ['Hello']

    # Test with separator tag (br)
    doc = html.fromstring("<br>")
    result = extract_text_array(doc)
    assert result == [True]

    # Test with non-inline tag (div)
    doc = html.fromstring("<div></div>")
    result = extract_text_array(doc)
    assert result == [None, None]

    # Test with inline tag (span)
    doc = html.fromstring("<span></span>")
    result = extract_text_array(doc)
    assert result == []

    # Test with children
    doc = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(doc)
    assert result == [None, 'Text', None]

    # Test with multiple children
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(doc)
    assert result == [None, 'First', None, None, 'Second', None]

    # Test with tail text
    doc = html.fromstring("<div><p>Text</p>Tail</div>")
    result = extract_text_array(doc)
    assert result == [None, 'Text', 'Tail', None]

    # Test with nested inline elements
    doc = html.fromstring("<p><span>Hello</span> World</p>")
    result = extract_text_array(doc)
    assert result == ['Hello', ' World']

    # Test with separator inside
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    result = extract_text_array(doc)
    assert result == ['Line1', True, 'Line2']


# LLM-generated content at query #69
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    dom = type('Mock', (), {'tag': 'span', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == 'Hello'
    
    # Test with a block element
    dom = type('Mock', (), {'tag': 'div', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == 'Hello'
    
    # Test with separator
    dom = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == '\n'
    
    # Test with nested inline elements
    child = type('Mock', (), {'tag': 'b', 'text': 'World', 'tail': None, 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'span', 'text': 'Hello ', 'tail': None, 'getchildren': lambda self: [child]})()
    assert extract_text(parent) == 'Hello World'
    
    # Test with nested block elements
    child = type('Mock', (), {'tag': 'p', 'text': 'Paragraph', 'tail': None, 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: [child]})()
    assert extract_text(parent) == 'Paragraph'
    
    # Test with multiple children and tail text
    child1 = type('Mock', (), {'tag': 'b', 'text': 'bold', 'tail': ' and ', 'getchildren': lambda self: []})()
    child2 = type('Mock', (), {'tag': 'i', 'text': 'italic', 'tail': None, 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'span', 'text': 'Text: ', 'tail': None, 'getchildren': lambda self: [child1, child2]})()
    assert extract_text(parent) == 'Text: bold and italic'
    
    # Test with separators
    br = type('Mock', (), {'tag': 'br', 'text': None, 'tail': '\n', 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'div', 'text': 'Line1', 'tail': None, 'getchildren': lambda self: [br]})()
    result = extract_text(parent)
    assert '\n' in result
    assert 'Line1' in result
    
    # Test with custom symbols
    dom = type('Mock', (), {'tag': 'div', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom, block_symbol=' ') == 'Hello'
    
    # Test with squash_space=False
    dom = type('Mock', (), {'tag': 'span', 'text': '  Hello  ', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom, squash_space=False) == '  Hello  '
    
    # Test with whitespace squashing
    dom = type('Mock', (), {'tag': 'span', 'text': 'Hello   World', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == 'Hello World'
    
    # Test empty element
    dom = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == ''
    
    # Test with callable tag
    dom = type('Mock', (), {'tag': lambda: None, 'text': 'test', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == ''
    
    # Test with mixed inline and block elements
    child = type('Mock', (), {'tag': 'div', 'text': 'Block', 'tail': ' after', 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'span', 'text': 'Before ', 'tail': None, 'getchildren': lambda self: [child]})()
    result = extract_text(parent)
    assert 'Before' in result
    assert 'Block' in result
    assert 'after' in result
    
    # Test with multiple separators
    br1 = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    br2 = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'div', 'text': 'Start', 'tail': None, 'getchildren': lambda self: [br1, br2]})()
    result = extract_text(parent)
    assert result.count('\n') == 2
    assert 'Start' in result
```


# LLM-generated content at query #70
#--------------------------

```python
def test_extract_text():
    from xml.etree import ElementTree as ET
    
    # Test simple inline text
    dom = ET.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with nested inline elements
    dom = ET.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block elements
    dom = ET.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator (br)
    dom = ET.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with whitespace
    dom = ET.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with empty element
    dom = ET.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = ET.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test with custom block_symbol
    dom = ET.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    dom = ET.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    dom = ET.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test with complex nested structure
    dom = ET.fromstring("<div><p>Hello <b>World</b></p><p>Goodbye</p></div>")
    assert extract_text(dom) == "Hello World\nGoodbye"
    
    # Test with multiple separators
    dom = ET.fromstring("<p>A<br/>B<br/>C</p>")
    assert extract_text(dom) == "A\nB\nC"
    
    # Test with inline tags that are not displayed
    dom = ET.fromstring("<p>Hello <script>alert('test')</script>World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with attributes
    dom = ET.fromstring("<p class='test'>Hello</p>")
    assert extract_text(dom) == "Hello"
    
    # Test with special whitespace characters
    dom = ET.fromstring("<p>Hello\u200BWorld</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with non-breaking space (not in WHITESPACE_RE)
    dom = ET.fromstring("<p>Hello\u00A0World</p>")
    assert extract_text(dom) == "Hello\u00A0World"


# LLM-generated content at query #71
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from xml.etree import ElementTree as ET
    dom = ET.fromstring("<span>hello world</span>")
    result = extract_text_array(dom)
    assert result == ["hello world"], f"Expected ['hello world'], got {result}"
    
    # Test with separator element (br)
    dom = ET.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"
    
    # Test with block element
    dom = ET.fromstring("<div>text</div>")
    result = extract_text_array(dom)
    assert result == [None, "text", None] or result == ["text"], f"Unexpected result: {result}"
    
    # Test with mixed inline and block elements
    dom = ET.fromstring("<div><span>hello</span> <span>world</span></div>")
    result = extract_text_array(dom)
    assert None in result, "Expected artificial newlines in result"
    assert "hello" in result, "Expected 'hello' in result"
    assert "world" in result, "Expected 'world' in result"
    
    # Test with separator inside block
    dom = ET.fromstring("<div>line1<br/>line2</div>")
    result = extract_text_array(dom)
    assert True in result, "Expected separator (True) in result"
    assert "line1" in result, "Expected 'line1' in result"
    assert "line2" in result, "Expected 'line2' in result"
    
    # Test with nested inline elements
    dom = ET.fromstring("<p><b>bold</b> and <i>italic</i></p>")
    result = extract_text_array(dom)
    assert "bold" in result, "Expected 'bold' in result"
    assert "italic" in result, "Expected 'italic' in result"
    assert " and " in result, "Expected ' and ' in result"
    
    # Test with squash_artifical_nl=False
    dom = ET.fromstring("<div>a</div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, "a", None], f"Expected [None, 'a', None], got {result}"
    
    # Test with strip_artifical_nl=False
    dom = ET.fromstring("<div>text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert None in result, "Expected artificial newlines when strip_artifical_nl=False"
    
    # Test with empty element
    dom = ET.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [], f"Expected empty list, got {result}"
    
    # Test with callable tag (should return empty string)
    dom = ET.fromstring("<div>text</div>")
    dom.tag = lambda: None
    result = extract_text_array(dom)
    assert result == "", f"Expected empty string for callable tag, got {result}"


# LLM-generated content at query #72
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml.html import fromstring
    dom = fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element (div)
    dom = fromstring("<div>Block text</div>")
    assert extract_text(dom) == "Block text"
    
    # Test with nested inline elements
    dom = fromstring("<p>This is <b>bold</b> text</p>")
    assert extract_text(dom) == "This is bold text"
    
    # Test with separator (br)
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with block elements creating newlines
    dom = fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with whitespace squashing
    dom = fromstring("<p>  Extra   spaces  </p>")
    assert extract_text(dom) == "Extra spaces"
    
    # Test with block_symbol customization
    dom = fromstring("<div>A</div><div>B</div>")
    assert extract_text(dom, block_symbol='|') == "A|B"
    
    # Test with sep_symbol customization
    dom = fromstring("<p>A<br>B</p>")
    assert extract_text(dom, sep_symbol='|') == "A|B"
    
    # Test with squash_space=False
    dom = fromstring("<p>  Extra   spaces  </p>")
    assert extract_text(dom, squash_space=False) == "  Extra   spaces  "
    
    # Test with complex nested structure
    dom = fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> and <i>italic</i></p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    expected = "Title\nParagraph with bold and italic\nItem 1\nItem 2"
    assert extract_text(dom) == expected
    
    # Test empty div
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = fromstring("<div>   </div>")
    assert extract_text(dom) == ""
    
    # Test with separator at start and end
    dom = fromstring("<p><br>Text<br></p>")
    assert extract_text(dom) == "Text"
    
    # Test with only separators
    dom = fromstring("<p><br><br></p>")
    assert extract_text(dom) == ""
    
    # Test with custom symbols
    dom = fromstring("<div>A</div><div>B</div><p>C<br>D</p>")
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == "A|B|C|D"
```


# LLM-generated content at query #73
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import html
    doc = html.fromstring("<span>Hello World</span>")
    assert extract_text(doc) == "Hello World"
    
    # Test with block element
    doc = html.fromstring("<div>Hello</div>")
    assert extract_text(doc) == "Hello"
    
    # Test with separator element (br)
    doc = html.fromstring("Hello<br>World")
    assert extract_text(doc) == "Hello\nWorld"
    
    # Test nested elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with inline elements inside block
    doc = html.fromstring("<div>Hello <b>bold</b> world</div>")
    assert extract_text(doc) == "Hello bold world"
    
    # Test with multiple separators
    doc = html.fromstring("Line1<br><br>Line2")
    assert extract_text(doc) == "Line1\n\nLine2"
    
    # Test whitespace squashing
    doc = html.fromstring("<div>Hello    World</div>")
    assert extract_text(doc) == "Hello World"
    
    # Test newlines in text
    doc = html.fromstring("<div>Hello\nWorld</div>")
    assert extract_text(doc) == "Hello World"
    
    # Test with empty element
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with only whitespace
    doc = html.fromstring("<div>   </div>")
    assert extract_text(doc) == ""
    
    # Test complex nesting
    doc = html.fromstring("<div><p>Para1</p><p>Para2 with <b>bold</b> text</p></div>")
    assert extract_text(doc) == "Para1\nPara2 with bold text"
    
    # Test custom block symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=' | ') == "First | Second"
    
    # Test custom separator symbol
    doc = html.fromstring("Hello<br>World")
    assert extract_text(doc, sep_symbol=' | ') == "Hello | World"
    
    # Test with squash_space=False
    doc = html.fromstring("<div>  Hello  World  </div>")
    result = extract_text(doc, squash_space=False)
    assert "  " in result  # Should preserve some whitespace
    
    # Test with nested inline elements
    doc = html.fromstring("<div><span>Hello <em>emphasized</em></span></div>")
    assert extract_text(doc) == "Hello emphasized"
    
    # Test with multiple block elements
    doc = html.fromstring("<div><h1>Title</h1><p>Paragraph</p></div>")
    assert extract_text(doc) == "Title\nParagraph"


# LLM-generated content at query #74
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]

    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    assert extract_text_array(dom) == [True]

    # Test with block tag (div)
    dom = html.fromstring("<div>Text</div>")
    assert extract_text_array(dom) == ["Text"]

    # Test with nested inline tags
    dom = html.fromstring("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]

    # Test with block tag containing inline
    dom = html.fromstring("<div><span>Hello</span></div>")
    result = extract_text_array(dom)
    # Block tag adds None at start and end, but they get squashed
    assert result == ["Hello"]

    # Test with separator between text
    dom = html.fromstring("Line1<br/>Line2")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]

    # Test with nested block tags
    dom = html.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    result = extract_text_array(dom)
    # Each p tag adds None at start and end, but they get squashed
    assert result == ["Para1", None, "Para2"]

    # Test with text and tail text
    dom = html.fromstring("<div>Start<b>Bold</b>End</div>")
    result = extract_text_array(dom)
    assert result == ["Start", "Bold", "End"]

    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "Text", None, None]

    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == ["Text", None]

    # Test empty element
    dom = html.fromstring("<div></div>")
    assert extract_text_array(dom) == []

    # Test with callable tag (should return empty)
    class FakeDom:
        tag = lambda: None
    assert extract_text_array(FakeDom()) == ""


# LLM-generated content at query #75
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"], f"Expected ['Hello'], got {result}"

    # Test with separator tag
    dom = html.fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"

    # Test with block tag
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result == ["Text"], f"Expected ['Text'], got {result}"

    # Test with nested inline tags
    dom = html.fromstring("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"], f"Expected ['Hello ', 'World'], got {result}"

    # Test with nested block and inline tags
    dom = html.fromstring("<div><p>Paragraph</p></div>")
    result = extract_text_array(dom)
    assert result == ["Paragraph"], f"Expected ['Paragraph'], got {result}"

    # Test with separator inside block
    dom = html.fromstring("<div>Line1<br>Line2</div>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"], f"Expected ['Line1', True, 'Line2'], got {result}"

    # Test with multiple artificial newlines (None values)
    dom = html.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    result = extract_text_array(dom)
    assert result == ["Para1", "Para2"], f"Expected ['Para1', 'Para2'], got {result}"

    # Test with empty element
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [], f"Expected [], got {result}"

    # Test with only whitespace text
    dom = html.fromstring("<div>   </div>")
    result = extract_text_array(dom)
    assert result == ["   "], f"Expected ['   '], got {result}"

    # Test with squash_artifical_nl=False and strip_artifical_nl=False
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in result, f"Expected None in result, got {result}"
    assert "Text" in result, f"Expected 'Text' in result, got {result}"

    # Test with multiple levels of nesting
    dom = html.fromstring("<div><p><span>Deep <b>text</b></span></p></div>")
    result = extract_text_array(dom)
    assert result == ["Deep ", "text"], f"Expected ['Deep ', 'text'], got {result}"

    # Test with tail text
    dom = html.fromstring("<div>Before <b>bold</b> After</div>")
    result = extract_text_array(dom)
    assert result == ["Before ", "bold", " After"], f"Expected ['Before ', 'bold', ' After'], got {result}"

    # Test with callable tag (edge case)
    class MockElement:
        tag = lambda: None
    mock_dom = MockElement()
    result = extract_text_array(mock_dom)
    assert result == [], f"Expected [], got {result}"


# LLM-generated content at query #76
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element (should add newlines)
    dom = html.fromstring("<div>Hello</div><div>World</div>")
    assert extract_text(dom) == "Hello\nWorld"
    
    # Test with separator (br tag)
    dom = html.fromstring("Hello<br/>World")
    assert extract_text(dom) == "Hello\nWorld"
    
    # Test with nested elements
    dom = html.fromstring("<div><span>Hello</span> <span>World</span></div>")
    assert extract_text(dom) == "Hello World"
    
    # Test with multiple block elements
    dom = html.fromstring("<p>First</p><p>Second</p>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with nested block elements
    dom = html.fromstring("<div><p>Paragraph</p></div>")
    assert extract_text(dom) == "Paragraph"
    
    # Test with text before and after inline elements
    dom = html.fromstring("Start <b>bold</b> End")
    assert extract_text(dom) == "Start bold End"
    
    # Test with whitespace squashing
    dom = html.fromstring("<p>  Hello    World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with multiple whitespace characters
    dom = html.fromstring("<p>Hello\t\tWorld</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with custom block_symbol
    dom = html.fromstring("<div>A</div><div>B</div>")
    assert extract_text(dom, block_symbol=' | ') == "A | B"
    
    # Test with custom sep_symbol
    dom = html.fromstring("A<br/>B")
    assert extract_text(dom, sep_symbol=' | ') == "A | B"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>  Hello  World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello  World  "
    
    # Test empty content
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = html.fromstring("<div>   </div>")
    assert extract_text(dom) == ""
    
    # Test with complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> text</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with bold text" in result
    assert "Item 1" in result
    assert "Item 2" in result
    assert result.count("\n") == 4  # Title, paragraph, two list items
    
    # Test with mixed inline and block elements
    dom = html.fromstring("""
        <div>
            <span>Inline</span>
            <div>Block</div>
            <span>More inline</span>
        </div>
    """)
    result = extract_text(dom)
    assert "Inline" in result
    assert "Block" in result
    assert "More inline" in result
    
    # Test with leading/trailing whitespace in nested elements
    dom = html.fromstring("<div>  <span>  Hello  </span>  </div>")
    assert extract_text(dom) == "Hello"
    
    # Test with multiple separators
    dom = html.fromstring("Line1<br/><br/>Line2")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with deep nesting
    dom = html.fromstring("<div><div><div><span>Deep</span></div></div></div>")
    assert extract_text(dom) == "Deep"


# LLM-generated content at query #77
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from types import SimpleNamespace
    dom = SimpleNamespace(tag='span', text='Hello', tail=None, getchildren=lambda: [])
    result = extract_text_array(dom)
    assert result == ['Hello']
    
    # Test with separator tag
    dom = SimpleNamespace(tag='br', text=None, tail=None, getchildren=lambda: [])
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with block tag
    dom = SimpleNamespace(tag='div', text='Text', tail=None, getchildren=lambda: [])
    result = extract_text_array(dom)
    assert result == [None, 'Text', None]
    
    # Test with nested structure
    child = SimpleNamespace(tag='span', text='child', tail=' tail', getchildren=lambda: [])
    dom = SimpleNamespace(tag='div', text='parent ', tail=None, getchildren=lambda: [child])
    result = extract_text_array(dom)
    assert result == [None, 'parent ', 'child', ' tail', None]
    
    # Test with multiple children
    child1 = SimpleNamespace(tag='span', text='first', tail=' ', getchildren=lambda: [])
    child2 = SimpleNamespace(tag='br', text=None, tail=' ', getchildren=lambda: [])
    child3 = SimpleNamespace(tag='span', text='third', tail=None, getchildren=lambda: [])
    dom = SimpleNamespace(tag='div', text='', tail=None, getchildren=lambda: [child1, child2, child3])
    result = extract_text_array(dom)
    assert result == [None, 'first', ' ', True, ' ', 'third', None]
    
    # Test with callable tag (should return empty string)
    dom = SimpleNamespace(tag=lambda: None, text='text', tail=None, getchildren=lambda: [])
    result = extract_text_array(dom)
    assert result == []
    
    # Test with no text and no children
    dom = SimpleNamespace(tag='div', text=None, tail=None, getchildren=lambda: [])
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with squash_artifical_nl=False
    dom = SimpleNamespace(tag='div', text='text', tail=None, getchildren=lambda: [])
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'text', None]
    
    # Test with strip_artifical_nl=False
    dom = SimpleNamespace(tag='div', text='text', tail=None, getchildren=lambda: [])
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'text', None]


# LLM-generated content at query #78
#--------------------------

```python
def test_extract_text():
    # Test basic paragraph extraction
    from lxml.html import fromstring
    html = "<p>Hello world</p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello world"
    
    # Test nested inline elements
    html = "<p>Hello <b>bold</b> world</p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello bold world"
    
    # Test block elements adding newlines
    html = "<div>First</div><div>Second</div>"
    dom = fromstring(html)
    assert extract_text(dom) == "First\nSecond"
    
    # Test separator elements (br)
    html = "<p>Line1<br>Line2</p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test nested block elements
    html = "<div><p>Para1</p><p>Para2</p></div>"
    dom = fromstring(html)
    assert extract_text(dom) == "Para1\nPara2"
    
    # Test with whitespace
    html = "<p>  Hello   world  </p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello world"
    
    # Test empty content
    html = "<div></div>"
    dom = fromstring(html)
    assert extract_text(dom) == ""
    
    # Test mixed inline and block elements
    html = "<div><span>Inline</span><p>Block</p></div>"
    dom = fromstring(html)
    assert extract_text(dom) == "Inline\nBlock"
    
    # Test with custom separator symbol
    html = "<p>Line1<br>Line2</p>"
    dom = fromstring(html)
    assert extract_text(dom, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test without squashing whitespace
    html = "<p>  Hello   world  </p>"
    dom = fromstring(html)
    assert extract_text(dom, squash_space=False) == "Hello world"  # squash_space affects artificial newlines, not whitespace
    
    # Test complex nested structure
    html = """
    <div>
        <h1>Title</h1>
        <p>Paragraph with <b>bold</b> text</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </div>
    """
    dom = fromstring(html)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with bold text" in result
    assert "Item 1" in result
    assert "Item 2" in result
```


# LLM-generated content at query #79
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    from lxml import html
    dom = html.fromstring("<p>Hello world</p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with nested elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator tags (br)
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with inline tags (should not add newlines)
    dom = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with multiple whitespace
    dom = html.fromstring("<p>Hello   world</p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with leading/trailing whitespace
    dom = html.fromstring("<p>  Hello world  </p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with empty content
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> text</p>
            <br/>
            <p>After break</p>
        </div>
    """)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with bold text" in result
    assert "After break" in result
    
    # Test with custom block_symbol
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    assert extract_text(dom, block_symbol=' ') == "A B"
    
    # Test with custom sep_symbol
    dom = html.fromstring("<p>A<br/>B</p>")
    assert extract_text(dom, sep_symbol=' ') == "A B"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>Hello   world</p>")
    assert extract_text(dom, squash_space=False) == "Hello   world"
    
    # Test with inline elements that contain block elements
    dom = html.fromstring("<span><p>Test</p></span>")
    assert extract_text(dom) == "Test"


# LLM-generated content at query #80
#--------------------------

```python
def test_extract_text_array():
    from lxml import html
    
    # Test simple inline element
    doc = html.fromstring("<span>text</span>")
    assert extract_text_array(doc) == ["text"]
    
    # Test block element
    doc = html.fromstring("<div>text</div>")
    assert extract_text_array(doc) == [None, "text", None]
    
    # Test separator element
    doc = html.fromstring("<br>")
    assert extract_text_array(doc) == [True]
    
    # Test nested elements
    doc = html.fromstring("<div><span>hello</span><br><span>world</span></div>")
    result = extract_text_array(doc)
    assert None in result
    assert "hello" in result
    assert True in result
    assert "world" in result
    
    # Test with text and tail
    doc = html.fromstring("<div>before<span>inside</span>after</div>")
    result = extract_text_array(doc)
    assert "before" in result
    assert "inside" in result
    assert "after" in result
    
    # Test squash_artifical_nl=False
    doc = html.fromstring("<div><p>text</p></div>")
    result = extract_text_array(doc, squash_artifical_nl=False)
    assert result.count(None) > 1
    
    # Test strip_artifical_nl=False
    doc = html.fromstring("<div>text</div>")
    result = extract_text_array(doc, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None
    
    # Test empty element
    doc = html.fromstring("<div></div>")
    assert extract_text_array(doc) == [None, None]
    
    # Test callable tag returns empty string
    class CallableTag:
        def tag(self):
            pass
    doc = html.fromstring("<div>text</div>")
    doc.tag = CallableTag()
    assert extract_text_array(doc) == ''


# LLM-generated content at query #81
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml.html import fromstring
    dom = fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with block elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator elements (br)
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested elements
    dom = fromstring("<div><p>Text with <span>inline</span> content</p></div>")
    assert extract_text(dom) == "Text with inline content"
    
    # Test with multiple levels of nesting
    dom = fromstring("<div><p><b>Bold</b> and <i>italic</i></p></div>")
    assert extract_text(dom) == "Bold and italic"
    
    # Test with empty elements
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with whitespace
    dom = fromstring("<p>  Hello   world  </p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with custom block symbol
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "First | Second"
    
    # Test with custom separator symbol
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with squash_space=False
    dom = fromstring("<p>  Hello   world  </p>")
    result = extract_text(dom, squash_space=False)
    assert "  " in result  # whitespace is preserved
    
    # Test with list elements
    dom = fromstring("<ul><li>Item 1</li><li>Item 2</li></ul>")
    assert extract_text(dom) == "Item 1\nItem 2"
    
    # Test with mixed content
    dom = fromstring("<div>Start<p>Middle</p>End</div>")
    assert extract_text(dom) == "Start\nMiddle\nEnd"


# LLM-generated content at query #82
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple text node
    from xml.etree.ElementTree import Element
    div = Element('div')
    div.text = 'Hello World'
    assert extract_text(div) == 'Hello World'
    
    # Test 2: Inline tags should not add newlines
    span = Element('span')
    span.text = 'Inline'
    div = Element('div')
    div.append(span)
    assert extract_text(div) == 'Inline'
    
    # Test 3: Block tags should add newlines
    div1 = Element('div')
    div1.text = 'First'
    div2 = Element('div')
    div2.text = 'Second'
    parent = Element('div')
    parent.append(div1)
    parent.append(div2)
    assert extract_text(parent) == 'First\nSecond'
    
    # Test 4: Separator tags (br)
    br = Element('br')
    div = Element('div')
    div.text = 'Before'
    div.append(br)
    br.tail = 'After'
    assert extract_text(div) == 'Before\nAfter'
    
    # Test 5: Whitespace squashing
    div = Element('div')
    div.text = 'Hello    World'
    assert extract_text(div) == 'Hello World'
    
    # Test 6: Nested inline tags
    b = Element('b')
    b.text = 'Bold'
    span = Element('span')
    span.text = 'Normal'
    div = Element('div')
    div.append(b)
    div.append(span)
    assert extract_text(div) == 'BoldNormal'
    
    # Test 7: Empty element
    div = Element('div')
    assert extract_text(div) == ''
    
    # Test 8: Text with leading/trailing whitespace
    div = Element('div')
    div.text = '  Hello  '
    assert extract_text(div) == 'Hello'
    
    # Test 9: Multiple block elements with text
    p1 = Element('p')
    p1.text = 'First paragraph'
    p2 = Element('p')
    p2.text = 'Second paragraph'
    body = Element('body')
    body.append(p1)
    body.append(p2)
    assert extract_text(body) == 'First paragraph\nSecond paragraph'
    
    # Test 10: Mixed inline and block elements
    b = Element('b')
    b.text = 'Bold text'
    p = Element('p')
    p.text = 'Paragraph text'
    div = Element('div')
    div.append(b)
    div.append(p)
    assert extract_text(div) == 'Bold text\nParagraph text'
    
    # Test 11: Multiple br tags
    br1 = Element('br')
    br2 = Element('br')
    div = Element('div')
    div.text = 'Start'
    div.append(br1)
    br1.tail = 'Middle'
    div.append(br2)
    br2.tail = 'End'
    assert extract_text(div) == 'Start\nMiddle\nEnd'
    
    # Test 12: Custom block and sep symbols
    div = Element('div')
    div.text = 'Hello'
    br = Element('br')
    div.append(br)
    br.tail = 'World'
    assert extract_text(div, block_symbol='|', sep_symbol='|') == 'Hello|World'


# LLM-generated content at query #83
#--------------------------

```python
def test_extract_text():
    # Test simple inline element
    dom = type('Mock', (), {'tag': 'span', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == 'Hello'
    
    # Test block element
    dom = type('Mock', (), {'tag': 'div', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == 'Hello'
    
    # Test nested elements
    child = type('Mock', (), {'tag': 'span', 'text': 'World', 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': 'Hello ', 'tail': None, 'getchildren': lambda self: [child]})()
    assert extract_text(dom) == 'Hello World'
    
    # Test separator element (br)
    dom = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == '\n'
    
    # Test multiple separators
    child1 = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    child2 = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: [child1, child2]})()
    assert extract_text(dom) == '\n\n'
    
    # Test with tail text
    child = type('Mock', (), {'tag': 'span', 'text': 'World', 'tail': '!', 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': 'Hello ', 'tail': None, 'getchildren': lambda self: [child]})()
    assert extract_text(dom) == 'Hello World!'
    
    # Test whitespace squashing
    dom = type('Mock', (), {'tag': 'span', 'text': 'Hello   World', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == 'Hello World'
    
    # Test block_symbol parameter
    child = type('Mock', (), {'tag': 'div', 'text': 'World', 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: [child]})()
    assert extract_text(dom, block_symbol=' ') == 'Hello World'
    
    # Test sep_symbol parameter
    child = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: [child]})()
    assert extract_text(dom, sep_symbol=' ') == 'Hello '
    
    # Test with squash_space=False
    dom = type('Mock', (), {'tag': 'span', 'text': '  Hello  ', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom, squash_space=False) == '  Hello  '
    
    # Test empty element
    dom = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == ''
    
    # Test callable tag (should return empty string)
    dom = type('Mock', (), {'tag': lambda: None, 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == ''
    
    # Test complex nested structure
    child1 = type('Mock', (), {'tag': 'span', 'text': 'World', 'tail': None, 'getchildren': lambda self: []})()
    child2 = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    child3 = type('Mock', (), {'tag': 'span', 'text': 'Foo', 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': 'Hello ', 'tail': None, 'getchildren': lambda self: [child1, child2, child3]})()
    assert extract_text(dom) == 'Hello World\nFoo'


# LLM-generated content at query #84
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"]

    # Test with block element
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == ["Hello"]

    # Test with separator element (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]

    # Test with nested elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"]

    # Test with text and child elements
    dom = html.fromstring("<div>Start <span>Middle</span> End</div>")
    result = extract_text_array(dom)
    assert result == ["Start ", "Middle", " End"]

    # Test with inline elements inside block
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]

    # Test with multiple br separators
    dom = html.fromstring("<p>Line1<br/>Line2<br/>Line3</p>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2", True, "Line3"]

    # Test empty element
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []

    # Test with callable tag (should return empty string)
    class FakeTag:
        def __call__(self):
            pass
    fake_dom = type('FakeDom', (), {'tag': FakeTag()})()
    result = extract_text_array(fake_dom)
    assert result == ""

    # Test with text only
    dom = html.fromstring("Just text")
    result = extract_text_array(dom)
    assert result == ["Just text"]

    # Test with artificial newlines squashing disabled
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in result  # Should have artificial newlines

    # Test with artificial newlines stripping disabled
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result[0] == "First"  # Should start with "First"
    assert result[-1] == "Second"  # Should end with "Second"


# LLM-generated content at query #85
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"]
    
    # Test with separator element (br)
    dom = fragment_fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with block element
    dom = fragment_fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result == [None, "Text", None]
    
    # Test with nested elements
    dom = fragment_fromstring("<div><span>Hello</span> <span>World</span></div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", " ", "World", None]
    
    # Test with separator inside block
    dom = fragment_fromstring("<div>Line1<br>Line2</div>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None]
    
    # Test with nested non-inline elements
    dom = fragment_fromstring("<div><p>Paragraph</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "Paragraph", None, None]
    
    # Test with element that has no text
    dom = fragment_fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with text and tail
    dom = fragment_fromstring("<div>Hello<span>World</span>Again</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", "World", "Again", None]
    
    # Test with inline elements that should not get artificial newlines
    dom = fragment_fromstring("<b>Bold</b>")
    result = extract_text_array(dom)
    assert result == ["Bold"]
    
    # Test with multiple br elements
    dom = fragment_fromstring("<div><br><br></div>")
    result = extract_text_array(dom)
    assert result == [None, True, True, None]
    
    # Test squash_artifical_nl=False
    dom = fragment_fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "A", None, None, None, "B", None, None]
    
    # Test strip_artifical_nl=False
    dom = fragment_fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Text", None]
    
    # Test both squash and strip disabled
    dom = fragment_fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None, "A", None, None]


# LLM-generated content at query #86
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    from lxml.html import fromstring
    dom = fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with nested inline elements
    dom = fromstring("<p>Hello <b>bold</b> World</p>")
    assert extract_text(dom) == "Hello bold World"
    
    # Test with block elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator elements (br)
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with multiple br
    dom = fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with custom block_symbol
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    dom = fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test with empty content
    dom = fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with nested block elements
    dom = fromstring("<div><div><p>Deep</p></div></div>")
    assert extract_text(dom) == "Deep"
    
    # Test with mixed inline and block elements
    dom = fromstring("<div><p>Text with <span>inline</span></p><p>Another</p></div>")
    assert extract_text(dom) == "Text with inline\nAnother"
    
    # Test with whitespace handling
    dom = fromstring("<p>  Multiple   spaces  </p>")
    assert extract_text(dom) == "Multiple spaces"


# LLM-generated content at query #87
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"], f"Expected ['Hello'], got {result}"
    
    # Test with a separator element (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"
    
    # Test with a block element
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result == ["Text"], f"Expected ['Text'], got {result}"
    
    # Test with nested elements
    dom = html.fromstring("<div><span>Hello</span><span>World</span></div>")
    result = extract_text_array(dom)
    assert result == ["Hello", "World"], f"Expected ['Hello', 'World'], got {result}"
    
    # Test with br separator
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"], f"Expected ['Line1', True, 'Line2'], got {result}"
    
    # Test with text and tail
    dom = html.fromstring("<p>Before <b>bold</b> After</p>")
    result = extract_text_array(dom)
    assert result == ["Before ", "bold", " After"], f"Expected ['Before ', 'bold', ' After'], got {result}"
    
    # Test with artificial newlines (None values)
    dom = html.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    result = extract_text_array(dom)
    assert result == ["Para1", "Para2"], f"Expected ['Para1', 'Para2'], got {result}"
    
    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result, f"Expected None values in result, got {result}"
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None, f"Expected leading/trailing None, got {result}"
    
    # Test with empty element
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [], f"Expected empty list, got {result}"
    
    # Test with nested block elements
    dom = html.fromstring("<div><div>Nested</div></div>")
    result = extract_text_array(dom)
    assert result == ["Nested"], f"Expected ['Nested'], got {result}"
    
    # Test with callable tag (should return empty string)
    class MockElement:
        tag = lambda: None
    result = extract_text_array(MockElement())
    assert result == '', f"Expected empty string, got {result}"


# LLM-generated content at query #88
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml import html
    dom = html.fromstring("<span>test</span>")
    assert extract_text_array(dom) == ["test"]

    # Test with separator (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]  # True represents separator

    # Test with block element (div)
    dom = html.fromstring("<div>hello</div>")
    result = extract_text_array(dom)
    assert result == [None, "hello", None]  # None represents artificial newlines

    # Test with nested elements
    dom = html.fromstring("<div><span>text</span></div>")
    result = extract_text_array(dom)
    assert result == [None, "text", None]

    # Test with tail text
    dom = html.fromstring("<div><span>a</span>tail</div>")
    result = extract_text_array(dom)
    assert result == [None, "a", "tail", None]

    # Test squash_artifical_nl option
    dom = html.fromstring("<div><div>a</div></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == [None, "a", None]  # Squashed consecutive None

    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "a", None, None]  # Multiple None

    # Test strip_artifical_nl option
    dom = html.fromstring("<div>text</div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["text"]  # Stripped leading/trailing None

    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "text", None]  # Kept leading/trailing None

    # Test with callable tag (returns empty string)
    class MockElement:
        tag = lambda: None
    mock_dom = MockElement()
    assert extract_text_array(mock_dom) == ""

    # Test element with text and children
    dom = html.fromstring("<p>Hello <b>world</b>!</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "world", "!", None]

    # Test with multiple separators
    dom = html.fromstring("<br/><br/>")
    result = extract_text_array(dom)
    assert result == [True, True]


# LLM-generated content at query #89
#--------------------------

```python
def test_extract_text():
    # Test with simple text
    from lxml.html import fromstring
    dom = fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with inline tags
    dom = fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with separators
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with block elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with nested elements
    dom = fromstring("<div><p>Hello <span>world</span></p></div>")
    assert extract_text(dom) == "Hello world"
    
    # Test with whitespace
    dom = fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with empty text
    dom = fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with multiple separators
    dom = fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with custom block_symbol
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=" | ") == "First | Second"
    
    # Test with custom sep_symbol
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=" | ") == "Line1 | Line2"
    
    # Test with squash_space=False
    dom = fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "Hello   World"
    
    # Test with complex nested structure
    dom = fromstring("<div><h1>Title</h1><p>Paragraph with <b>bold</b> text</p><ul><li>Item 1</li><li>Item 2</li></ul></div>")
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with bold text" in result
    assert "Item 1" in result
    assert "Item 2" in result
    assert result.count("\n") == 3  # Title, paragraph, and two list items (separated by newlines)


# LLM-generated content at query #90
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<p>Hello World</p>")
    assert extract_text_array(dom) == ["Hello World"]

    # Test with inline elements
    dom = fragment_fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text_array(dom) == ["Hello ", "bold", " world"]

    # Test with separator element (br)
    dom = fragment_fromstring("<p>Line1<br>Line2</p>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]

    # Test with block element inside another
    dom = fragment_fromstring("<div><p>Para1</p><p>Para2</p></div>")
    result = extract_text_array(dom)
    assert result == [None, "Para1", None, None, "Para2", None]

    # Test with empty element
    dom = fragment_fromstring("<p></p>")
    assert extract_text_array(dom) == []

    # Test with nested inline elements
    dom = fragment_fromstring("<p>Text <span>span <em>em</em> text</span> end</p>")
    assert extract_text_array(dom) == ["Text ", "span ", "em", " text", " end"]

    # Test squash_artifical_nl=False
    dom = fragment_fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result

    # Test strip_artifical_nl=False
    dom = fragment_fromstring("<div><p>Hello</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None

    # Test with tail text
    dom = fragment_fromstring("<p>Hello<b>bold</b>tail</p>")
    assert extract_text_array(dom) == ["Hello", "bold", "tail"]


# LLM-generated content at query #91
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tags
    from lxml.html import fromstring
    dom = fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with separators (br tags)
    dom = fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with block elements
    dom = fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with nested elements
    dom = fromstring("<div><p>Paragraph <b>bold</b></p><p>Second</p></div>")
    result = extract_text(dom)
    assert "Paragraph bold" in result
    assert "Second" in result
    
    # Test with whitespace squashing
    dom = fromstring("<p>  Hello    world  </p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with custom symbols
    dom = fromstring("<p>First</p><p>Second</p>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test with empty content
    dom = fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with only text
    dom = fromstring("<p>Just text</p>")
    assert extract_text(dom) == "Just text"
    
    # Test with multiple br tags
    dom = fromstring("<p>Line1<br/><br/>Line2</p>")
    assert extract_text(dom) == "Line1\n\nLine2"
    
    # Test with script tag (inline)
    dom = fromstring("<div><script>var x=1;</script>Content</div>")
    assert extract_text(dom) == "Content"
    
    # Test with squash_space=False
    dom = fromstring("<p>  Hello  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello  "
```


# LLM-generated content at query #92
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml import html
    doc = html.fromstring("<span>hello</span>")
    assert extract_text_array(doc) == ["hello"]
    
    # Test with separator element (br)
    doc = html.fromstring("<br/>")
    assert extract_text_array(doc) == [True]
    
    # Test with block element
    doc = html.fromstring("<div>text</div>")
    assert extract_text_array(doc) == ["text"]
    
    # Test with nested inline elements
    doc = html.fromstring("<p><b>bold</b> and <i>italic</i></p>")
    result = extract_text_array(doc)
    assert None in result  # has artificial newlines from <p>
    assert "bold" in result
    assert "italic" in result
    
    # Test with text before and after child
    doc = html.fromstring("<p>start <b>bold</b> end</p>")
    result = extract_text_array(doc)
    assert "start " in result
    assert "bold" in result
    assert " end" in result
    
    # Test with separator inside block
    doc = html.fromstring("<p>line1<br/>line2</p>")
    result = extract_text_array(doc)
    assert "line1" in result
    assert True in result
    assert "line2" in result
    
    # Test with empty element
    doc = html.fromstring("<div></div>")
    assert extract_text_array(doc) == []
    
    # Test with only text
    doc = html.fromstring("just text")
    assert extract_text_array(doc) == ["just text"]
    
    # Test with multiple nested levels
    doc = html.fromstring("<div><p><span>deep</span></p></div>")
    result = extract_text_array(doc)
    assert "deep" in result
    
    # Test squash_artifical_nl=False
    doc = html.fromstring("<div><p>text</p></div>")
    result = extract_text_array(doc, squash_artifical_nl=False)
    none_count = sum(1 for x in result if x is None)
    assert none_count >= 2  # block start and end
    
    # Test strip_artifical_nl=False
    doc = html.fromstring("<div>text</div>")
    result = extract_text_array(doc, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None  # artificial newlines at start or end
    
    # Test with callable tag (should return empty string)
    doc = html.fromstring("<div>text</div>")
    doc.tag = lambda: None
    assert extract_text_array(doc) == ''


# LLM-generated content at query #93
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text node
    from lxml import html
    dom = html.fromstring("<p>Hello world</p>")
    assert extract_text_array(dom) == [None, "Hello world", None]
    
    # Test with inline tag
    dom = html.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with nested tags
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert None in result
    assert "First" in result
    assert "Second" in result
    
    # Test with mixed inline and block elements
    dom = html.fromstring("<div><span>inline</span><p>block</p></div>")
    result = extract_text_array(dom)
    assert "inline" in result
    assert "block" in result
    assert None in result  # block element newline
    
    # Test with text and tail text
    dom = html.fromstring("<p>Start <b>bold</b> end</p>")
    result = extract_text_array(dom)
    assert "Start " in result
    assert "bold" in result
    assert " end" in result
    
    # Test squash_artifical_nl parameter
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    result_with_squash = extract_text_array(dom, squash_artifical_nl=True)
    result_without_squash = extract_text_array(dom, squash_artifical_nl=False)
    assert len(result_with_squash) < len(result_without_squash)
    
    # Test strip_artifical_nl parameter
    dom = html.fromstring("<div><p>Content</p></div>")
    result_with_strip = extract_text_array(dom, strip_artifical_nl=True)
    result_without_strip = extract_text_array(dom, strip_artifical_nl=False)
    # With strip, shouldn't start or end with None
    if result_with_strip:
        assert result_with_strip[0] is not None
        assert result_with_strip[-1] is not None
    
    # Test with callable tag (should return empty string)
    class FakeElement:
        tag = lambda: None
    dom = FakeElement()
    assert extract_text_array(dom) == ''


# LLM-generated content at query #94
#--------------------------

```python
def test_extract_text():
    # Test simple text extraction from a div
    from lxml.html import fromstring
    html = "<div>Hello World</div>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello World"
    
    # Test inline tags
    html = "<p>Hello <b>World</b></p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello World"
    
    # Test block-level tags
    html = "<div><p>First paragraph</p><p>Second paragraph</p></div>"
    dom = fromstring(html)
    assert extract_text(dom) == "First paragraph\nSecond paragraph"
    
    # Test separator tags (br)
    html = "<div>Line1<br>Line2</div>"
    dom = fromstring(html)
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test nested inline elements
    html = "<div><span>Hello <em>World</em></span></div>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello World"
    
    # Test whitespace squashing
    html = "<div>Hello     World</div>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello World"
    
    # Test leading/trailing whitespace removal
    html = "<div>   Hello World   </div>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello World"
    
    # Test with custom block_symbol
    html = "<div><p>First</p><p>Second</p></div>"
    dom = fromstring(html)
    assert extract_text(dom, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    html = "<div>Line1<br>Line2</div>"
    dom = fromstring(html)
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    html = "<div>Hello   World</div>"
    dom = fromstring(html)
    assert extract_text(dom, squash_space=False) == "Hello   World"
    
    # Test empty content
    html = "<div></div>"
    dom = fromstring(html)
    assert extract_text(dom) == ""
    
    # Test nested block elements
    html = "<div><div><p>Nested</p></div></div>"
    dom = fromstring(html)
    assert extract_text(dom) == "Nested"
    
    # Test mixed content with text and tags
    html = "<div>Hello <p>World</p> Again</div>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello\nWorld\nAgain"
    
    # Test consecutive br tags
    html = "<div>Line1<br><br>Line2</div>"
    dom = fromstring(html)
    assert extract_text(dom) == "Line1\n\nLine2"


# LLM-generated content at query #95
#--------------------------

```python
def test_extract_text():
    # Test with simple text element
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with nested inline elements
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator element (br)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with custom block_symbol
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    assert extract_text(dom, block_symbol=' ') == "A B"
    
    # Test with custom sep_symbol
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test with mixed content
    dom = html.fromstring("<div>Start<p>Middle</p>End</div>")
    assert extract_text(dom) == "Start\nMiddle\nEnd"
    
    # Test with empty element
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test with nested block elements
    dom = html.fromstring("<div><section><p>A</p></section><p>B</p></div>")
    assert extract_text(dom) == "A\nB"
    
    # Test with inline elements only
    dom = html.fromstring("<span>Hello</span> <span>World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with script tag (should be treated as inline)
    dom = html.fromstring("<p>Text <script>alert('test')</script> more text</p>")
    assert extract_text(dom) == "Text alert('test') more text"
    
    # Test with trailing and leading whitespace
    dom = html.fromstring("  <p>Hello</p>  ")
    assert extract_text(dom) == "Hello"
    
    # Test complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> and <i>italic</i></p>
            <ul>
                <li>Item 1<br>with break</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with bold and italic" in result
    assert "Item 1\nwith break" in result
    assert "Item 2" in result


# LLM-generated content at query #96
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml.html import fromstring
    dom = fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with separator element (br)
    dom = fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with block element (div)
    dom = fromstring("<div>Content</div>")
    result = extract_text_array(dom)
    assert result == [None, "Content", None]
    
    # Test with nested elements
    dom = fromstring("<div>Hello <span>World</span></div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "World", None]
    
    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "Text", None, None]
    
    # Test with strip_artifical_nl=False
    dom = fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Text", None]
    
    # Test with both parameters False
    dom = fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None, "Text", None, None]
    
    # Test with empty element
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with callable tag (should return empty string)
    class FakeElement:
        tag = lambda: None
    result = extract_text_array(FakeElement())
    assert result == ""
    
    # Test with text and tail
    dom = fromstring("<div>Hello <b>bold</b> world</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "bold", " world", None]


# LLM-generated content at query #97
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple text extraction
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 2: Nested inline elements
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 3: Block elements with newlines
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test 4: Separator elements (br)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test 5: Multiple whitespace handling
    dom = html.fromstring("<p>Hello   World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 6: Empty elements
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test 7: Custom block and separator symbols
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' ') == "First Second"
    
    # Test 8: Leading and trailing whitespace
    dom = html.fromstring("<p>  Hello  </p>")
    assert extract_text(dom) == "Hello"
    
    # Test 9: Complex nested structure
    dom = html.fromstring("<div><h1>Title</h1><p>Text with <b>bold</b> and <i>italic</i></p></div>")
    result = extract_text(dom)
    assert "Title" in result
    assert "Text with bold and italic" in result
    assert "\n" in result
    
    # Test 10: Multiple br tags
    dom = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(dom) == "Line1\n\nLine2"
    
    # Test 11: Nested block elements
    dom = html.fromstring("<div><div><p>Nested</p></div></div>")
    assert extract_text(dom) == "Nested"
    
    # Test 12: Squash space disabled
    dom = html.fromstring("<p>Hello   World</p>")
    assert extract_text(dom, squash_space=False) == "Hello   World"
    
    # Test 13: Inline elements without text
    dom = html.fromstring("<p><b></b>Text</p>")
    assert extract_text(dom) == "Text"
    
    # Test 14: Mixed content with tail text
    dom = html.fromstring("<p>Start<b>bold</b>middle<i>italic</i>end</p>")
    assert extract_text(dom) == "Startboldmiddleitalicend"


# LLM-generated content at query #98
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    from lxml import etree
    import pytest
    
    # Test simple paragraph
    html = etree.fromstring("<p>Hello World</p>")
    assert extract_text(html) == "Hello World"
    
    # Test with inline tags
    html = etree.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(html) == "Hello World"
    
    # Test with separators (br tags)
    html = etree.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(html) == "Line1\nLine2"
    
    # Test with block elements (div)
    html = etree.fromstring("<div>Block1</div><div>Block2</div>")
    assert extract_text(html) == "Block1\nBlock2"
    
    # Test nested elements
    html = etree.fromstring("<div><p>Nested <b>text</b></p></div>")
    assert extract_text(html) == "Nested text"
    
    # Test whitespace handling
    html = etree.fromstring("<p>  Hello   World  </p>")
    assert extract_text(html) == "Hello World"
    
    # Test multiple whitespace characters
    html = etree.fromstring("<p>Hello\n\nWorld</p>")
    assert extract_text(html) == "Hello World"
    
    # Test empty element
    html = etree.fromstring("<p></p>")
    assert extract_text(html) == ""
    
    # Test element with only whitespace
    html = etree.fromstring("<p>   </p>")
    assert extract_text(html) == ""
    
    # Test mixed inline and block elements
    html = etree.fromstring("<div><span>Span</span><p>Paragraph</p></div>")
    assert extract_text(html) == "Span\nParagraph"
    
    # Test with custom symbols
    html = etree.fromstring("<p>Hello</p><p>World</p>")
    assert extract_text(html, block_symbol=" | ", sep_symbol=" - ") == "Hello | World"
    
    # Test with squash_space=False
    html = etree.fromstring("<p>  Hello   World  </p>")
    result = extract_text(html, squash_space=False)
    assert "  " in result  # Should preserve some whitespace
    
    # Test with script tag (should be treated as inline)
    html = etree.fromstring("<p>Text <script>var x=1;</script> more</p>")
    assert extract_text(html) == "Text var x=1; more"
    
    # Test complex nested structure
    html = etree.fromstring("""
        <div>
            <h1>Title</h1>
            <p>First <b>paragraph</b></p>
            <p>Second paragraph<br/>with break</p>
        </div>
    """)
    result = extract_text(html)
    assert "Title" in result
    assert "First paragraph" in result
    assert "Second paragraph" in result
    assert "with break" in result
    assert result.count("\n") >= 2  # Should have multiple newlines
    
    # Test with multiple br tags
    html = etree.fromstring("<p>Line1<br/><br/>Line2</p>")
    assert extract_text(html) == "Line1\n\nLine2"
    
    # Test leading/trailing whitespace removal
    html = etree.fromstring("<div>  <p>Content</p>  </div>")
    assert extract_text(html) == "Content"```


# LLM-generated content at query #99
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block-level tag (should add newline)
    dom = fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator tag (br)
    dom = fromstring("<span>Line1<br/>Line2</span>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested inline within block
    dom = fromstring("<p>Text with <strong>bold</strong> word</p>")
    assert extract_text(dom) == "Text with bold word"
    
    # Test with multiple block levels
    dom = fromstring("<div><p>Para1</p><p>Para2</p></div>")
    assert extract_text(dom) == "Para1\nPara2"
    
    # Test with whitespace squashing
    dom = fromstring("<p>  Lots   of   spaces  </p>")
    assert extract_text(dom) == "Lots of spaces"
    
    # Test with custom block_symbol
    dom = fromstring("<div>A</div><div>B</div>")
    assert extract_text(dom, block_symbol=' | ') == "A | B"
    
    # Test with custom sep_symbol
    dom = fromstring("<span>A<br/>B</span>")
    assert extract_text(dom, sep_symbol=' --- ') == "A --- B"
    
    # Test with squash_space=False
    dom = fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test with tail text
    dom = fromstring("<p>Start <b>bold</b> middle <i>italic</i> end</p>")
    assert extract_text(dom) == "Start bold middle italic end"
    
    # Test with empty content
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test with nested block tags
    dom = fromstring("<section><h1>Title</h1><p>Content</p></section>")
    assert extract_text(dom) == "Title\nContent"
    
    # Test with mixed inline and block
    dom = fromstring("<div><span>Inline</span><p>Block</p></div>")
    assert extract_text(dom) == "Inline\nBlock"
    
    # Test with multiple br tags
    dom = fromstring("<span>Line1<br/><br/>Line2</span>")
    assert extract_text(dom) == "Line1\n\nLine2"  # Multiple br = multiple separators
    
    # Test with text before and after nested elements
    dom = fromstring("<p>Before <b>bold</b> after</p>")
    assert extract_text(dom) == "Before bold after"
    
    # Test with custom symbols
    dom = fromstring("<div>A</div><div>B<br/>C</div>")
    assert extract_text(dom, block_symbol=' | ', sep_symbol=' - ') == "A | B - C"
    
    # Test with no squash_space and custom symbols
    dom = fromstring("<div>  A  </div><div>  B  </div>")
    assert extract_text(dom, block_symbol='|', squash_space=False) == "  A  |  B  "


# LLM-generated content at query #100
#--------------------------

```python
def test_extract_text_array():
    # Simple test with a text node
    class MockTextNode:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []
        
        def getchildren(self):
            return self._children
    
    # Test with inline tag (span) containing text
    span = MockTextNode('span', 'Hello')
    result = extract_text_array(span)
    assert result == ['Hello']
    
    # Test with block tag (div) containing text
    div = MockTextNode('div', 'Hello')
    result = extract_text_array(div)
    assert result == [None, 'Hello', None]
    
    # Test with separator tag (br)
    br = MockTextNode('br')
    result = extract_text_array(br)
    assert result == [True]
    
    # Test nested structure with inline inside block
    span = MockTextNode('span', 'World')
    div = MockTextNode('div', 'Hello ', [span], '!')
    result = extract_text_array(div)
    assert result == [None, 'Hello ', 'World', '!', None]
    
    # Test with squash_artifical_nl=False
    div = MockTextNode('div', 'Hello')
    result = extract_text_array(div, squash_artifical_nl=False)
    assert result == [None, 'Hello', None]
    
    # Test with strip_artifical_nl=False
    div = MockTextNode('div', 'Hello')
    result = extract_text_array(div, strip_artifical_nl=False)
    assert result == [None, 'Hello', None]
    
    # Test nested block elements
    inner_div = MockTextNode('div', 'Inner')
    outer_div = MockTextNode('div', 'Outer ', [inner_div], ' End')
    result = extract_text_array(outer_div)
    assert result == ['Outer ', 'Inner', ' End']
    
    # Test with callable tag (should return empty string)
    mock_dom = MockTextNode(lambda: None, 'text')
    result = extract_text_array(mock_dom)
    assert result == ''


# LLM-generated content at query #101
#--------------------------

```python
def test_extract_text():
    # Test simple paragraph
    from lxml.html import fromstring
    html = "<p>Hello world</p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello world"

    # Test with inline tags
    html = "<p>Hello <b>bold</b> world</p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello bold world"

    # Test with separators (br)
    html = "<p>Line 1<br>Line 2</p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Line 1\nLine 2"

    # Test with multiple block elements
    html = "<div><p>First</p><p>Second</p></div>"
    dom = fromstring(html)
    assert extract_text(dom) == "First\nSecond"

    # Test with nested elements
    html = "<div><p>Text with <span>span</span> inside</p></div>"
    dom = fromstring(html)
    assert extract_text(dom) == "Text with span inside"

    # Test with whitespace
    html = "<p>  Hello   world  </p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello world"

    # Test empty element
    html = "<div></div>"
    dom = fromstring(html)
    assert extract_text(dom) == ""

    # Test with only text
    html = "Just text"
    dom = fromstring(html)
    assert extract_text(dom) == "Just text"

    # Test with custom block symbol
    html = "<p>First</p><p>Second</p>"
    dom = fromstring(html)
    assert extract_text(dom, block_symbol='|') == "First|Second"

    # Test with custom separator symbol
    html = "<p>Line 1<br>Line 2</p>"
    dom = fromstring(html)
    assert extract_text(dom, sep_symbol='|') == "Line 1|Line 2"

    # Test with squash_space=False
    html = "<p>  Hello   world  </p>"
    dom = fromstring(html)
    assert extract_text(dom, squash_space=False) == "Hello   world"
```


# LLM-generated content at query #102
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple inline text
    from lxml import html
    dom = html.fromstring("<p>Hello world</p>")
    assert extract_text(dom) == "Hello world"

    # Test 2: Nested inline elements
    dom = html.fromstring("<p>Hello <strong>beautiful</strong> world</p>")
    assert extract_text(dom) == "Hello beautiful world"

    # Test 3: Block elements with artificial newlines
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"

    # Test 4: Separator elements (br)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"

    # Test 5: Nested block elements
    dom = html.fromstring("<div><div><p>Deep text</p></div></div>")
    assert extract_text(dom) == "Deep text"

    # Test 6: Mixed inline and block elements
    dom = html.fromstring("<div><p>Para <span>span</span></p><p>Next</p></div>")
    assert extract_text(dom) == "Para span\nNext"

    # Test 7: Empty elements
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""

    # Test 8: Text with whitespace
    dom = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(dom) == "Hello world"

    # Test 9: Custom block_symbol and sep_symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p><br></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == "First|Second|"

    # Test 10: With squash_space=False
    dom = html.fromstring("<p>  Hello   world  </p>")
    result = extract_text(dom, squash_space=False)
    assert "  " in result  # Whitespace preserved

    # Test 11: Multiple separators
    dom = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(dom) == "Line1\n\nLine2"

    # Test 12: Complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <a href="#">link</a> and <strong>bold</strong></p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    expected = "Title\nParagraph with link and bold\nItem 1\nItem 2"
    assert extract_text(dom) == expected

    # Test 13: Script and style content (should be empty)
    dom = html.fromstring("<div>Text <script>var x=1;</script> more</div>")
    assert extract_text(dom) == "Text  more"

    # Test 14: Empty text at boundaries
    dom = html.fromstring("<div><p></p>Middle<p></p></div>")
    assert extract_text(dom) == "Middle"

    # Test 15: Tail text after inline elements
    dom = html.fromstring("<p>Start <strong>bold</strong> end</p>")
    assert extract_text(dom) == "Start bold end"

    # Test 16: Nested separators
    dom = html.fromstring("<div><p>Text<br><span>More</span></p></div>")
    assert extract_text(dom) == "Text\nMore"

    # Test 17: Multiple block elements with text
    dom = html.fromstring("<div><p>A</p><p>B</p><p>C</p></div>")
    assert extract_text(dom) == "A\nB\nC"

    # Test 18: Inline elements in block elements
    dom = html.fromstring("<div><p><span><b>Deep</b></span> nested</p></div>")
    assert extract_text(dom) == "Deep nested"

    # Test 19: Separator at start/end
    dom = html.fromstring("<p><br>Text<br></p>")
    assert extract_text(dom) == "Text"

    # Test 20: Multiple whitespace characters
    dom = html.fromstring("<p>Hello\t\n\r world</p>")
    assert extract_text(dom) == "Hello world"


# LLM-generated content at query #103
#--------------------------

```python
def test_extract_text():
    # Test with simple text
    class MockElement:
        tag = 'p'
        text = 'Hello World'
        def getchildren(self):
            return []
    
    dom = MockElement()
    result = extract_text(dom)
    assert result == 'Hello World', f"Expected 'Hello World', got '{result}'"
    
    # Test with nested inline elements
    class MockChild:
        tag = 'strong'
        text = 'bold'
        tail = None
        def getchildren(self):
            return []
    
    class MockParent:
        tag = 'p'
        text = 'This is '
        def getchildren(self):
            return [MockChild()]
    
    dom = MockParent()
    result = extract_text(dom)
    assert result == 'This is bold', f"Expected 'This is bold', got '{result}'"
    
    # Test with separator tag (br)
    class MockBr:
        tag = 'br'
        text = None
        tail = '\nline2'
        def getchildren(self):
            return []
    
    class MockWithBr:
        tag = 'p'
        text = 'line1'
        def getchildren(self):
            return [MockBr()]
    
    dom = MockWithBr()
    result = extract_text(dom)
    assert result == 'line1\nline2', f"Expected 'line1\\nline2', got '{result}'"
    
    # Test with block element (div)
    class MockDiv:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return []
    
    class MockWithDiv:
        tag = 'body'
        text = None
        def getchildren(self):
            child = MockDiv()
            child.text = 'Block content'
            return [child]
    
    dom = MockWithDiv()
    result = extract_text(dom)
    assert result == 'Block content', f"Expected 'Block content', got '{result}'"
    
    # Test with multiple children and whitespace
    class MockSpan1:
        tag = 'span'
        text = 'first'
        tail = None
        def getchildren(self):
            return []
    
    class MockSpan2:
        tag = 'span'
        text = 'second'
        tail = None
        def getchildren(self):
            return []
    
    class MockParent2:
        tag = 'p'
        text = None
        def getchildren(self):
            return [MockSpan1(), MockSpan2()]
    
    dom = MockParent2()
    result = extract_text(dom)
    assert result == 'first second', f"Expected 'first second', got '{result}'"
    
    # Test with custom symbols
    dom = MockElement()
    result = extract_text(dom, block_symbol=' ', sep_symbol='|')
    assert result == 'Hello World', f"Expected 'Hello World', got '{result}'"
    
    # Test empty element
    class MockEmpty:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    
    dom = MockEmpty()
    result = extract_text(dom)
    assert result == '', f"Expected '', got '{result}'"


# LLM-generated content at query #104
#--------------------------

```python
def test_extract_text_array():
    # Test basic inline tag
    from lxml.html import fromstring
    dom = fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"]
    
    # Test block tag (div)
    dom = fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == ["Hello"]
    
    # Test separator tag (br)
    dom = fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test nested tags
    dom = fromstring("<div><span>Hello</span><span>World</span></div>")
    result = extract_text_array(dom)
    assert result == ["Hello", "World"]
    
    # Test with text and tail
    dom = fromstring("<div>Text1<span>Inner</span>Text2</div>")
    result = extract_text_array(dom)
    assert result == ["Text1", "Inner", "Text2"]
    
    # Test squash_artifical_nl parameter
    dom = fromstring("<div><div><span>Hello</span></div></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert None not in result
    
    # Test strip_artifical_nl parameter
    dom = fromstring("<div><span>Hello</span></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result[0] == "Hello"
    assert result[-1] == "Hello"
    
    # Test callable tag returns empty string
    class FakeDom:
        tag = lambda: None
    assert extract_text_array(FakeDom()) == ""
    
    # Test separator in nested context
    dom = fromstring("<div>Line1<br/>Line2</div>")
    result = extract_text_array(dom)
    assert True in result
    
    # Test with None text
    dom = fromstring("<div><span></span></div>")
    result = extract_text_array(dom)
    assert None not in result or len(result) == 0
    
    # Test multiple children with different types
    dom = fromstring("<div><p>Para1</p><br/><p>Para2</p></div>")
    result = extract_text_array(dom)
    assert len(result) >= 3
    assert "Para1" in result
    assert True in result
    assert "Para2" in result


# LLM-generated content at query #105
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    from lxml.html import fromstring
    dom = fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element
    dom = fromstring("<div>Hello World</div>")
    assert extract_text(dom) == "Hello World"
    
    # Test with nested elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator element (br)
    dom = fromstring("<span>Line1<br>Line2</span>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with inline elements inside block
    dom = fromstring("<p>This is <strong>bold</strong> text</p>")
    assert extract_text(dom) == "This is bold text"
    
    # Test with multiple block elements
    dom = fromstring("<div><h1>Title</h1><p>Content</p></div>")
    assert extract_text(dom) == "Title\nContent"
    
    # Test with whitespace normalization
    dom = fromstring("<p>Too   many    spaces</p>")
    assert extract_text(dom) == "Too many spaces"
    
    # Test with empty element
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with text only
    dom = fromstring("Plain text")
    assert extract_text(dom) == "Plain text"
    
    # Test with custom block symbol
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test with custom separator symbol
    dom = fromstring("<span>Line1<br>Line2</span>")
    assert extract_text(dom, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with squash_space disabled
    dom = fromstring("<p>Hello   World</p>")
    assert extract_text(dom, squash_space=False) == "Hello   World"
    
    # Test with complex nested structure
    dom = fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <a href="#">link</a> inside</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with link inside" in result
    assert "Item 1" in result
    assert "Item 2" in result
    
    # Test with leading/trailing whitespace
    dom = fromstring("  <p>Content</p>  ")
    assert extract_text(dom) == "Content"
    
    # Test with multiple br tags
    dom = fromstring("<span>Line1<br><br>Line2</span>")
    assert extract_text(dom) == "Line1\n\nLine2"
    
    # Test with inline tag that has no text
    dom = fromstring("<p>Text <img src='test.jpg'> more text</p>")
    assert extract_text(dom) == "Text  more text"
```


# LLM-generated content at query #106
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<span>hello</span>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["hello"], f"Expected ['hello'], got {result}"

    # Test with separator element (br)
    dom = fragment_fromstring("<br/>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == [True], f"Expected [True], got {result}"

    # Test with block element (div)
    dom = fragment_fromstring("<div>text</div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["text"], f"Expected ['text'], got {result}"

    # Test with nested inline elements
    dom = fragment_fromstring("<span>hello <b>world</b></span>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["hello ", "world"], f"Expected ['hello ', 'world'], got {result}"

    # Test with block containing inline
    dom = fragment_fromstring("<div><span>hello</span></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["hello"], f"Expected ['hello'], got {result}"

    # Test with multiple block elements
    dom = fragment_fromstring("<div>first</div><div>second</div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["first", "second"], f"Expected ['first', 'second'], got {result}"

    # Test with br separator between text
    dom = fragment_fromstring("text1<br/>text2")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["text1", True, "text2"], f"Expected ['text1', True, 'text2'], got {result}"

    # Test with nested block elements
    dom = fragment_fromstring("<div>outer<div>inner</div></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["outer", "inner"], f"Expected ['outer', 'inner'], got {result}"

    # Test with squash_artifical_nl=False
    dom = fragment_fromstring("<div>text</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == [None, "text", None], f"Expected [None, 'text', None], got {result}"

    # Test with empty element
    dom = fragment_fromstring("<div></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == [], f"Expected [], got {result}"

    # Test with callable tag (edge case)
    class FakeDom:
        tag = lambda: None
    result = extract_text_array(FakeDom(), squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == "", f"Expected '', got {result}"


# LLM-generated content at query #107
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator element (br)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested inline elements
    dom = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with whitespace normalization
    dom = html.fromstring("<p>  Hello    World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with empty element
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with multiple block elements
    dom = html.fromstring("<div><h1>Title</h1><p>Paragraph</p></div>")
    assert extract_text(dom) == "Title\nParagraph"
    
    # Test with custom block symbol
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    assert extract_text(dom, block_symbol='|') == "A|B"
    
    # Test with custom separator symbol
    dom = html.fromstring("<p>A<br>B</p>")
    assert extract_text(dom, sep_symbol='|') == "A|B"
    
    # Test without squashing spaces
    dom = html.fromstring("<p>  Hello  World  </p>")
    result = extract_text(dom, squash_space=False)
    assert result == "  Hello  World  "
    
    # Test with nested block elements
    dom = html.fromstring("<div><div><p>Deep</p></div></div>")
    assert extract_text(dom) == "Deep"
    
    # Test with mixed inline and block
    dom = html.fromstring("<div><span>Inline</span><p>Block</p></div>")
    assert extract_text(dom) == "Inline\nBlock"
    
    # Test with text before and after children
    dom = html.fromstring("<p>Start <b>middle</b> end</p>")
    assert extract_text(dom) == "Start middle end"
    
    # Test with multiple whitespace types
    dom = html.fromstring("<p>Hello\tWorld\nTest</p>")
    assert extract_text(dom) == "Hello World Test"


# LLM-generated content at query #108
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element
    dom = html.fromstring("<div>Hello<br>World</div>")
    assert extract_text(dom) == "Hello\nWorld"
    
    # Test with nested elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with inline elements inside block
    dom = html.fromstring("<p>This is <strong>bold</strong> text</p>")
    assert extract_text(dom) == "This is bold text"
    
    # Test with separator tags
    dom = html.fromstring("<div>Line1<br>Line2<br>Line3</div>")
    assert extract_text(dom) == "Line1\nLine2\nLine3"
    
    # Test with multiple whitespace
    dom = html.fromstring("<p>Hello     World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with leading/trailing whitespace
    dom = html.fromstring("<p>   Hello World   </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with empty content
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with nested block elements
    dom = html.fromstring("<div><div><p>Deep</p></div></div>")
    assert extract_text(dom) == "Deep"
    
    # Test with multiple block elements
    dom = html.fromstring("<div><h1>Title</h1><p>Paragraph</p></div>")
    assert extract_text(dom) == "Title\nParagraph"
    
    # Test with custom block_symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=" ") == "First Second"
    
    # Test with custom sep_symbol
    dom = html.fromstring("<div>Line1<br>Line2</div>")
    assert extract_text(dom, sep_symbol=" ") == "Line1 Line2"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>  Hello  World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello  World  "
    
    # Test with comment nodes (should be ignored)
    dom = html.fromstring("<div>Hello<!-- comment -->World</div>")
    assert extract_text(dom) == "HelloWorld"
    
    # Test with script tags (inline but should be included)
    dom = html.fromstring("<div><script>var x=1;</script>Text</div>")
    assert extract_text(dom) == "var x=1; Text"
    
    # Test with nested inline elements
    dom = html.fromstring("<p>Hello <em>emphasized</em> <strong>bold</strong></p>")
    assert extract_text(dom) == "Hello emphasized bold"


# LLM-generated content at query #109
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import etree
    dom = etree.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator
    dom = etree.fromstring("<div>Text<br/>More text</div>")
    assert extract_text(dom) == "Text\nMore text"
    
    # Test with nested inline elements
    dom = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with whitespace
    dom = etree.fromstring("<div>  Hello   World  </div>")
    assert extract_text(dom) == "Hello World"
    
    # Test with newlines in HTML
    dom = etree.fromstring("<div>\n  Line1\n  Line2\n</div>")
    assert extract_text(dom) == "Line1 Line2"
    
    # Test empty element
    dom = etree.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with only text
    dom = etree.fromstring("<p>Just text</p>")
    assert extract_text(dom) == "Just text"
    
    # Test with multiple levels of nesting
    dom = etree.fromstring("<div><ul><li>Item1</li><li>Item2</li></ul></div>")
    assert extract_text(dom) == "Item1\nItem2"
    
    # Test with custom block_symbol
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "First | Second"
    
    # Test with custom sep_symbol
    dom = etree.fromstring("<div>Text<br/>More text</div>")
    assert extract_text(dom, sep_symbol=' --- ') == "Text --- More text"


# LLM-generated content at query #110
#--------------------------

```python
def test_extract_text_array():
    # Test simple inline tag
    from lxml import html
    doc = html.fromstring("<span>Hello</span>")
    result = extract_text_array(doc)
    assert result == ["Hello"], f"Expected ['Hello'], got {result}"
    
    # Test block tag with children
    doc = html.fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text_array(doc)
    # div is block, adds None at start and end; p is block, adds None before and after text
    # After squash: None, "Hello", None, "World", None -> squash removes consecutive None
    expected = [None, "Hello", None, "World", None]
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test separator tag (br)
    doc = html.fromstring("<div>Hello<br/>World</div>")
    result = extract_text_array(doc)
    # div adds None at start and end; br adds True
    expected = [None, "Hello", True, "World", None]
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test nested inline tags
    doc = html.fromstring("<p><b>Hello</b> <i>World</i></p>")
    result = extract_text_array(doc)
    # p adds None; b and i are inline, no None
    expected = [None, "Hello", " ", "World", None]
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test with squash_artifical_nl=False
    doc = html.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(doc, squash_artifical_nl=False)
    # div: None, p: None, "A", None, p: None, "B", None, div: None
    expected = [None, None, "A", None, None, "B", None, None]
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test with strip_artifical_nl=False
    doc = html.fromstring("<div><p>Hello</p></div>")
    result = extract_text_array(doc, strip_artifical_nl=False)
    # div: None, p: None, "Hello", None, div: None -> squash -> None, "Hello", None
    expected = [None, "Hello", None]
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test tag with both text and tail
    doc = html.fromstring("<div>Start<p>Middle</p>End</div>")
    result = extract_text_array(doc)
    # div: None, "Start", p: None, "Middle", None, "End", None
    expected = [None, "Start", None, "Middle", None, "End", None]
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test empty document
    doc = html.fromstring("<div></div>")
    result = extract_text_array(doc)
    expected = [None, None]  # div adds None at start and end, both squashed? Actually: [None, None]
    # After squash: [None] (two consecutive None become one)
    assert result == [None], f"Expected [None], got {result}"
    
    # Test with callable tag (special case)
    class FakeElement:
        tag = lambda: None
    result = extract_text_array(FakeElement())
    assert result == "", f"Expected '', got {result}"
    
    # Test multiple separators
    doc = html.fromstring("<div><br/><br/></div>")
    result = extract_text_array(doc)
    # div: None, br: True, br: True, None
    expected = [None, True, True, None]
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test inline tag with no text
    doc = html.fromstring("<span></span>")
    result = extract_text_array(doc)
    assert result == [], f"Expected [], got {result}"
    
    # Test complex nested structure
    doc = html.fromstring("<div><p>Hello <b>World</b></p><br/><span>End</span></div>")
    result = extract_text_array(doc)
    # div: None, p: None, "Hello ", b: inline, "World", p: None, br: True, span: inline, "End", div: None
    expected = [None, None, "Hello ", "World", None, True, "End", None]
    # After squash: [None, "Hello ", "World", None, True, "End", None]
    assert result == expected, f"Expected {expected}, got {result}"


# LLM-generated content at query #111
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline tag
    class MockElement:
        tag = 'span'
        text = 'Hello'
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text(MockElement())
    assert result == 'Hello'
    
    # Test with block tag (div)
    class MockDiv:
        tag = 'div'
        text = 'Hello'
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text(MockDiv())
    assert result == 'Hello'
    
    # Test with separator tag (br)
    class MockBr:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text(MockBr())
    assert result == '\n'
    
    # Test nested elements
    class MockChild:
        tag = 'span'
        text = 'World'
        tail = None
        def getchildren(self):
            return []
    
    class MockParent:
        tag = 'div'
        text = 'Hello '
        tail = None
        def getchildren(self):
            return [MockChild()]
    
    result = extract_text(MockParent())
    assert result == 'Hello World'
    
    # Test with tail text
    class MockWithTail:
        tag = 'span'
        text = 'Hello'
        tail = ' World'
        def getchildren(self):
            return []
    
    result = extract_text(MockWithTail())
    assert result == 'Hello World'
    
    # Test with multiple children
    class MockChild1:
        tag = 'b'
        text = 'Bold'
        tail = ' '
        def getchildren(self):
            return []
    
    class MockChild2:
        tag = 'i'
        text = 'Italic'
        tail = None
        def getchildren(self):
            return []
    
    class MockParentWithChildren:
        tag = 'p'
        text = None
        tail = None
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    
    result = extract_text(MockParentWithChildren())
    assert result == 'Bold Italic'
    
    # Test with br separator
    class MockBrSeparator:
        tag = 'br'
        text = None
        tail = '\n'
        def getchildren(self):
            return []
    
    class MockTextBeforeBr:
        tag = 'span'
        text = 'Line1'
        tail = None
        def getchildren(self):
            return []
    
    class MockAfterBr:
        tag = 'span'
        text = 'Line2'
        tail = None
        def getchildren(self):
            return []
    
    class MockContentWithBr:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return [MockTextBeforeBr(), MockBrSeparator(), MockAfterBr()]
    
    result = extract_text(MockContentWithBr())
    assert result == 'Line1\nLine2'
    
    # Test with squash_space=False
    result = extract_text(MockParent(), squash_space=False)
    assert result == 'Hello World'
    
    # Test with custom symbols
    class MockCustomSymbols:
        tag = 'div'
        text = 'A'
        tail = None
        def getchildren(self):
            return [MockBr()]
    
    result = extract_text(MockCustomSymbols(), block_symbol='|', sep_symbol='-')
    assert result == 'A-'
    
    # Test empty element
    class MockEmpty:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text(MockEmpty())
    assert result == ''
    
    # Test with whitespace squashing
    class MockWhitespace:
        tag = 'div'
        text = 'Hello   World'
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text(MockWhitespace())
    assert result == 'Hello World'
```


# LLM-generated content at query #112
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple text extraction
    from lxml import html
    doc = html.fromstring("<p>Hello World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test 2: Nested inline elements
    doc = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(doc) == "Hello World"
    
    # Test 3: Block elements creating newlines
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test 4: Separator tags (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test 5: Multiple whitespace squashing
    doc = html.fromstring("<p>Hello    World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test 6: Whitespace with newlines
    doc = html.fromstring("<p>Hello\nWorld</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test 7: Empty element
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test 8: Element with only whitespace
    doc = html.fromstring("<p>   </p>")
    assert extract_text(doc) == ""
    
    # Test 9: Deeply nested structure
    doc = html.fromstring("<div><p><b>Text</b></p></div>")
    assert extract_text(doc) == "Text"
    
    # Test 10: Mixed inline and block elements
    doc = html.fromstring("<div><span>Item1</span><div>Item2</div></div>")
    assert extract_text(doc) == "Item1\nItem2"
    
    # Test 11: List structure
    doc = html.fromstring("<ul><li>First</li><li>Second</li></ul>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test 12: Multiple br tags
    doc = html.fromstring("<p>Text<br><br>More text</p>")
    assert extract_text(doc) == "Text\n\nMore text"
    
    # Test 13: Text with tail content
    doc = html.fromstring("<p>Start<b>bold</b>end</p>")
    assert extract_text(doc) == "Startboldend"
    
    # Test 14: Complex nested with separators
    doc = html.fromstring("<div><h1>Title</h1><p>Paragraph with <br> break</p></div>")
    assert extract_text(doc) == "Title\nParagraph with \n break"
    
    # Test 15: Custom block_symbol
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc, block_symbol=" | ") == "First | Second"
    
    # Test 16: Custom sep_symbol
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc, sep_symbol=" -- ") == "Line1 -- Line2"
    
    # Test 17: Disable squashing
    doc = html.fromstring("<p>  Hello  World  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello  World  "
    
    # Test 18: Empty document
    doc = html.fromstring("")
    assert extract_text(doc) == ""
    
    # Test 19: Multiple nested inline elements
    doc = html.fromstring("<p><i>Italic</i> and <b>bold</b></p>")
    assert extract_text(doc) == "Italic and bold"
    
    # Test 20: Pre tag (should preserve whitespace)
    doc = html.fromstring("<pre>  Preserved  \n  whitespace  </pre>")
    assert extract_text(doc) == "  Preserved  \n  whitespace  "


# LLM-generated content at query #113
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    dom = html.fromstring("<p>Hello <b>world</b>!</p>")
    assert extract_text(dom) == "Hello world!"
    
    # Test with block-level elements
    dom = html.fromstring("<div><p>First paragraph</p><p>Second paragraph</p></div>")
    assert extract_text(dom) == "First paragraph\nSecond paragraph"
    
    # Test with separator elements (br)
    dom = html.fromstring("<p>Line 1<br>Line 2</p>")
    assert extract_text(dom) == "Line 1\nLine 2"
    
    # Test with multiple separators
    dom = html.fromstring("<p>Line 1<br><br>Line 2</p>")
    assert extract_text(dom) == "Line 1\n\nLine 2"
    
    # Test with nested elements
    dom = html.fromstring("<div><span>Nested <b>bold</b> text</span></div>")
    assert extract_text(dom) == "Nested bold text"
    
    # Test with whitespace handling
    dom = html.fromstring("<p>   Multiple   spaces   </p>")
    assert extract_text(dom) == "Multiple spaces"
    
    # Test with empty content
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test with custom block symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test with custom separator symbol
    dom = html.fromstring("<p>Line 1<br>Line 2</p>")
    assert extract_text(dom, sep_symbol='|') == "Line 1|Line 2"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>   Multiple   spaces   </p>")
    assert extract_text(dom, squash_space=False) == "   Multiple   spaces   "
    
    # Test with complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> and <i>italic</i></p>
            <p>Another paragraph</p>
        </div>
    """)
    expected = "Title\nParagraph with bold and italic\nAnother paragraph"
    assert extract_text(dom) == expected
    
    # Test with anchor tags (inline)
    dom = html.fromstring("<p>Click <a href='#'>here</a> now</p>")
    assert extract_text(dom) == "Click here now"
```


# LLM-generated content at query #114
#--------------------------

```python
def test_extract_text():
    # Test with simple text element
    from lxml import html
    doc = html.fromstring("<p>Hello World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with nested inline elements
    doc = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with block elements (should add newlines)
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with separators (br)
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with multiple whitespace
    doc = html.fromstring("<p>Hello    World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with whitespace at start and end
    doc = html.fromstring("<p>  Hello World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with nested block elements
    doc = html.fromstring("<div><p>Text</p><div><p>Nested</p></div></div>")
    assert extract_text(doc) == "Text\nNested"
    
    # Test with empty element
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with text in tail
    doc = html.fromstring("<p>Hello<b>bold</b>world</p>")
    assert extract_text(doc) == "Helloboldworld"
    
    # Test with custom block symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=' ') == "First Second"
    
    # Test with custom separator symbol
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc, sep_symbol=' ') == "Line1 Line2"
    
    # Test without squashing space
    doc = html.fromstring("<p>Hello    World</p>")
    assert extract_text(doc, squash_space=False) == "Hello    World"
    
    # Test with multiple nested levels
    doc = html.fromstring("<div><p>Level1</p><div><p>Level2</p><p>Level2b</p></div></div>")
    assert extract_text(doc) == "Level1\nLevel2\nLevel2b"


# LLM-generated content at query #115
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tags
    from lxml.html import fromstring
    dom = fromstring("<span>hello</span>")
    assert extract_text_array(dom) == ["hello"]

    # Test with separator tag (br)
    dom = fromstring("<br>")
    assert extract_text_array(dom) == [True]

    # Test with block tag (div)
    dom = fromstring("<div>text</div>")
    result = extract_text_array(dom)
    assert result[0] is None  # artificial newline before
    assert "text" in result
    assert result[-1] is None  # artificial newline after

    # Test with nested tags
    dom = fromstring("<div><span>hello</span><br><span>world</span></div>")
    result = extract_text_array(dom)
    assert result[0] is None  # div opening
    assert "hello" in result
    assert True in result  # br separator
    assert "world" in result
    assert result[-1] is None  # div closing

    # Test with tail text
    dom = fromstring("<div>before<span>inside</span>after</div>")
    result = extract_text_array(dom)
    assert "before" in result
    assert "inside" in result
    assert "after" in result

    # Test with empty tags
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]

    # Test with text only (no tags)
    dom = fromstring("just text")
    assert extract_text_array(dom) == ["just text"]

    # Test with multiple levels of nesting
    dom = fromstring("<div><p><span>deep</span></p></div>")
    result = extract_text_array(dom)
    assert result[0] is None  # div
    assert result[1] is None  # p
    assert "deep" in result
    assert result[-1] is None  # p closing
    assert result[-2] is None  # div closing

    # Test with inline tags nested in block tags
    dom = fromstring("<div><a>link</a></div>")
    result = extract_text_array(dom)
    assert result[0] is None  # div opening
    assert "link" in result
    assert result[-1] is None  # div closing

    # Test with multiple br tags
    dom = fromstring("<div>text<br><br>more</div>")
    result = extract_text_array(dom)
    assert "text" in result
    assert True in result  # first br
    assert result.count(True) == 2  # two br separators
    assert "more" in result

    # Test with squash_artifical_nl=True (default)
    dom = fromstring("<div><p>text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    # Should have squashed consecutive None values
    none_count = sum(1 for x in result if x is None)
    assert none_count <= 2  # at most one at start and one at end

    # Test with strip_artifical_nl=True (default)
    dom = fromstring("<div><p>text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    # Should not start or end with None
    if result:
        assert result[0] is not None
        assert result[-1] is not None

    # Test with callable tag (edge case)
    class MockElement:
        tag = lambda: None
    result = extract_text_array(MockElement())
    assert result == ""

    # Test with multiple text nodes and tails
    dom = fromstring("<div>a<span>b</span>c<span>d</span>e</div>")
    result = extract_text_array(dom)
    assert "a" in result
    assert "b" in result
    assert "c" in result
    assert "d" in result
    assert "e" in result


# LLM-generated content at query #116
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml.html import fromstring
    dom = fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]

    # Test with block element
    dom = fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == ["Hello"]  # None (artificial newlines) are stripped

    # Test with separator element (br)
    dom = fromstring("<br/>")
    assert extract_text_array(dom) == [True]

    # Test with mixed inline and block elements
    dom = fromstring("<div>Hello <span>world</span></div>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "world"]

    # Test with nested elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert " ".join(x for x in result if isinstance(x, str)) == "First Second"

    # Test with text and tail
    dom = fromstring("<div>Hello <b>bold</b> text</div>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "bold", " text"]

    # Test with separator (br) in text
    dom = fromstring("<div>Line1<br/>Line2</div>")
    result = extract_text_array(dom)
    assert True in result  # Should contain separator

    # Test with empty element
    dom = fromstring("<div></div>")
    assert extract_text_array(dom) == []

    # Test with only whitespace
    dom = fromstring("<div>   </div>")
    result = extract_text_array(dom)
    assert all(isinstance(x, str) for x in result) or result == []

    # Test with callable tag (should return empty string)
    class FakeElement:
        tag = lambda: None
    fake_dom = FakeElement()
    assert extract_text_array(fake_dom) == ""

    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in result  # Should contain artificial newlines

    # Test with strip_artifical_nl=False
    dom = fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None  # Should have leading/trailing None


# LLM-generated content at query #117
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tags only
    from lxml import html
    doc = html.fromstring('<span>Hello <b>World</b></span>')
    result = extract_text_array(doc)
    assert result == ['Hello ', 'World']
    
    # Test with separator tag
    doc = html.fromstring('<br/>')
    result = extract_text_array(doc)
    assert result == [True]
    
    # Test with block-level tag
    doc = html.fromstring('<div>Text</div>')
    result = extract_text_array(doc)
    assert result == [None, 'Text', None]
    
    # Test with nested block and inline tags
    doc = html.fromstring('<div><p>Paragraph <b>bold</b></p></div>')
    result = extract_text_array(doc)
    assert result == [None, None, 'Paragraph ', 'bold', None, None]
    
    # Test with text and tail
    doc = html.fromstring('<p>Start <b>bold</b> end</p>')
    result = extract_text_array(doc)
    assert result == [None, 'Start ', 'bold', ' end', None]
    
    # Test with multiple separators
    doc = html.fromstring('<div>Line1<br/>Line2</div>')
    result = extract_text_array(doc)
    assert result == [None, 'Line1', True, 'Line2', None]
    
    # Test squash_artifical_nl=False
    doc = html.fromstring('<div>Text</div>')
    result = extract_text_array(doc, squash_artifical_nl=False)
    assert result == [None, 'Text', None]
    
    # Test strip_artifical_nl=False
    doc = html.fromstring('<div>Text</div>')
    result = extract_text_array(doc, strip_artifical_nl=False)
    assert result == [None, 'Text', None]
    
    # Test with empty element
    doc = html.fromstring('<div></div>')
    result = extract_text_array(doc)
    assert result == []
    
    # Test with only text
    doc = html.fromstring('Just text')
    result = extract_text_array(doc)
    assert result == ['Just text']


# LLM-generated content at query #118
#--------------------------

```python
def test_extract_text():
    # Test simple text extraction
    from lxml import html
    doc = html.fromstring("<p>Hello World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with inline tags
    doc = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with separator tags
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with block elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test nested structures
    doc = html.fromstring("<div><p>Hello <b>World</b></p><p>Goodbye</p></div>")
    assert extract_text(doc) == "Hello World\nGoodbye"
    
    # Test with whitespace
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with multiple whitespace characters
    doc = html.fromstring("<p>Hello\t\tWorld</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test empty content
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with custom block_symbol
    doc = html.fromstring("<p>First</p><p>Second</p>")
    assert extract_text(doc, block_symbol=' | ') == "First | Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with no whitespace squashing
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello   World  "
    
    # Test complex document
    doc = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> and <i>italic</i></p>
            <br>
            <p>After break</p>
        </div>
    """)
    expected = "Title\nParagraph with bold and italic\n\nAfter break"
    assert extract_text(doc) == expected
    
    # Test with nested inline tags
    doc = html.fromstring("<p><span>Hello <b>World</b></span></p>")
    assert extract_text(doc) == "Hello World"


# LLM-generated content at query #119
#--------------------------

```python
def test_extract_text():
    # Test with inline tags - should produce no newlines
    from lxml.html import fromstring
    dom = fromstring("<span>Hello <b>World</b></span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block tags - should produce newlines
    dom = fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator tags (br)
    dom = fromstring("Line1<br>Line2")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test nested structure
    dom = fromstring("<div><p>Paragraph 1</p><p>Paragraph 2</p></div>")
    assert extract_text(dom) == "Paragraph 1\nParagraph 2"
    
    # Test with custom block_symbol
    dom = fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom, block_symbol='<br>') == "First<br>Second"
    
    # Test with custom sep_symbol
    dom = fromstring("Line1<br>Line2")
    assert extract_text(dom, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with squash_space=False
    dom = fromstring("<span>Hello   World</span>")
    assert extract_text(dom, squash_space=False) == "Hello   World"
    
    # Test with squash_space=True (default)
    dom = fromstring("<span>Hello   World</span>")
    assert extract_text(dom, squash_space=True) == "Hello World"
    
    # Test empty content
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = fromstring("<div>   </div>")
    assert extract_text(dom) == ""
    
    # Test mixed inline and block elements
    dom = fromstring("<p>This is <b>bold</b> text</p><p>Another paragraph</p>")
    assert extract_text(dom) == "This is bold text\nAnother paragraph"
    
    # Test with list elements
    dom = fromstring("<ul><li>Item 1</li><li>Item 2</li></ul>")
    result = extract_text(dom)
    assert "Item 1" in result
    assert "Item 2" in result
    
    # Test with pre tag (should preserve whitespace)
    dom = fromstring("<pre>  Indented text  </pre>")
    assert extract_text(dom) == "  Indented text  "
    
    # Test complex nested structure
    html = """
    <div>
        <header>
            <h1>Title</h1>
        </header>
        <main>
            <p>First paragraph with <a href="#">link</a></p>
            <p>Second paragraph</p>
        </main>
    </div>
    """
    dom = fromstring(html)
    result = extract_text(dom)
    assert "Title" in result
    assert "First paragraph with link" in result
    assert "Second paragraph" in result
    
    # Test with None text content
    dom = fromstring("<div><br></div>")
    assert extract_text(dom) == ""


# LLM-generated content at query #120
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple text element
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 2: Nested inline elements
    dom = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test 3: Block elements with newlines
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test 4: Separator elements (br)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test 5: Nested block elements
    dom = html.fromstring("<div><p>Text</p></div>")
    assert extract_text(dom) == "Text"
    
    # Test 6: Multiple whitespace handling
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 7: Empty element
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test 8: Element with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test 9: Mixed inline and block elements
    dom = html.fromstring("<div><span>Hello</span><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"
    
    # Test 10: Custom symbols
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' | ', sep_symbol=' - ') == "First | Second"
    
    # Test 11: Multiple br tags
    dom = html.fromstring("<p>A<br><br>B</p>")
    assert extract_text(dom) == "A\n\nB"
    
    # Test 12: Deep nesting
    dom = html.fromstring("<div><div><p>Deep</p></div></div>")
    assert extract_text(dom) == "Deep"
    
    # Test 13: Inline elements inside block elements
    dom = html.fromstring("<p>Text with <span>inline</span> content</p>")
    assert extract_text(dom) == "Text with inline content"
    
    # Test 14: All inline tags should not create newlines
    for tag in ['a', 'b', 'i', 'em', 'strong', 'code', 'span']:
        dom = html.fromstring(f"<p>Before<{tag}>Middle</{tag}>After</p>")
        assert extract_text(dom) == "BeforeMiddleAfter", f"Failed for tag {tag}"
    
    # Test 15: Text with tail
    dom = html.fromstring("<p><b>Bold</b> and <i>italic</i> text</p>")
    assert extract_text(dom) == "Bold and italic text"
    
    # Test 16: Multiple block levels
    dom = html.fromstring("<div><section><p>Content</p></section></div>")
    assert extract_text(dom) == "Content"


# LLM-generated content at query #121
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple text extraction
    from lxml.html import fromstring
    dom = fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 2: Nested inline elements
    dom = fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 3: Block elements with newlines
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test 4: Separator elements (br)
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test 5: Multiple whitespace squashing
    dom = fromstring("<p>Hello    World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 6: Leading/trailing whitespace stripping
    dom = fromstring("<p>  Hello World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 7: Empty content
    dom = fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test 8: Nested block elements
    dom = fromstring("<div><div><p>A</p></div><div><p>B</p></div></div>")
    assert extract_text(dom) == "A\nB"
    
    # Test 9: Mixed inline and block
    dom = fromstring("<div><p>Hello <b>World</b></p><p>Second</p></div>")
    assert extract_text(dom) == "Hello World\nSecond"
    
    # Test 10: Script tag (should be ignored)
    dom = fromstring("<div><script>var x = 1;</script><p>Content</p></div>")
    assert extract_text(dom) == "Content"
    
    # Test 11: Custom block_symbol and sep_symbol
    dom = fromstring("<div><p>A</p><p>B<br>C</p></div>")
    assert extract_text(dom, block_symbol="|", sep_symbol="~") == "A|B~C"
    
    # Test 12: squash_space=False
    dom = fromstring("<p>  Hello  World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello  World  "


# LLM-generated content at query #122
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with nested inline elements
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator elements (br)
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with custom block_symbol
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    assert extract_text(dom, block_symbol='|') == "A|B"
    
    # Test with custom sep_symbol
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test with multiple whitespace in text
    dom = html.fromstring("<p>Hello   World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with empty element
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with deeply nested structure
    dom = html.fromstring("<div><span><b>Deep</b></span><p>Content</p></div>")
    assert extract_text(dom) == "Deep\nContent"
    
    # Test with mix of inline and block elements
    dom = html.fromstring("<div>Text <span>inline</span><p>block</p></div>")
    assert extract_text(dom) == "Text inline\nblock"


# LLM-generated content at query #123
#--------------------------

```python
def test_extract_text():
    # Test basic inline elements
    from lxml import html as lxml_html
    doc = lxml_html.fromstring("<p>Hello <b>world</b></p>")
    result = extract_text(doc)
    assert result == "Hello world", f"Expected 'Hello world', got '{result}'"
    
    # Test with separators (br)
    doc = lxml_html.fromstring("<p>Line1<br>Line2</p>")
    result = extract_text(doc)
    assert result == "Line1\nLine2", f"Expected 'Line1\nLine2', got '{result}'"
    
    # Test with block elements
    doc = lxml_html.fromstring("<div>First</div><div>Second</div>")
    result = extract_text(doc)
    assert result == "First\nSecond", f"Expected 'First\nSecond', got '{result}'"
    
    # Test with nested block elements
    doc = lxml_html.fromstring("<div><p>Text</p></div>")
    result = extract_text(doc)
    assert result == "Text", f"Expected 'Text', got '{result}'"
    
    # Test with whitespace squashing
    doc = lxml_html.fromstring("<p>Hello    world</p>")
    result = extract_text(doc)
    assert result == "Hello world", f"Expected 'Hello world', got '{result}'"
    
    # Test with leading/trailing whitespace
    doc = lxml_html.fromstring("  <p>Hello</p>  ")
    result = extract_text(doc)
    assert result == "Hello", f"Expected 'Hello', got '{result}'"
    
    # Test empty element
    doc = lxml_html.fromstring("<p></p>")
    result = extract_text(doc)
    assert result == "", f"Expected empty string, got '{result}'"
    
    # Test with multiple block elements and whitespace
    doc = lxml_html.fromstring("<div>  <p>First</p>  <p>Second</p>  </div>")
    result = extract_text(doc)
    assert result == "First\nSecond", f"Expected 'First\nSecond', got '{result}'"
    
    # Test with custom symbols
    doc = lxml_html.fromstring("<p>Hello</p><p>World</p>")
    result = extract_text(doc, block_symbol=" | ")
    assert result == "Hello | World", f"Expected 'Hello | World', got '{result}'"
    
    doc = lxml_html.fromstring("<p>Line1<br>Line2</p>")
    result = extract_text(doc, sep_symbol=" --- ")
    assert result == "Line1 --- Line2", f"Expected 'Line1 --- Line2', got '{result}'"
    
    # Test with squash_space=False
    doc = lxml_html.fromstring("<p>Hello <b>world</b></p>")
    result = extract_text(doc, squash_space=False)
    assert result == "Hello world", f"Expected 'Hello world', got '{result}'"
    
    # Test with anchor tag (inline)
    doc = lxml_html.fromstring('<a href="#">Click</a>')
    result = extract_text(doc)
    assert result == "Click", f"Expected 'Click', got '{result}'"
    
    # Test with script tag
    doc = lxml_html.fromstring("<script>var x = 1;</script>")
    result = extract_text(doc)
    assert result == "", f"Expected empty string, got '{result}'"
    
    # Test complex nested structure
    html = """
    <div>
        <h1>Title</h1>
        <p>This is a <b>bold</b> and <i>italic</i> text.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </div>
    """
    doc = lxml_html.fromstring(html)
    result = extract_text(doc)
    expected = "Title\nThis is a bold and italic text.\nItem 1\nItem 2"
    assert result == expected, f"Expected '{expected}', got '{result}'"


# LLM-generated content at query #124
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with a separator element (br)
    dom = html.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with a block element (div)
    dom = html.fromstring("<div>Text</div>")
    assert extract_text_array(dom) == [None, "Text", None]
    
    # Test with nested elements
    dom = html.fromstring("<div><span>Hello</span> World</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", " World", None]
    
    # Test with multiple children
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, None, "Second", None, None]
    
    # Test with separator inside text
    dom = html.fromstring("<div>Line1<br/>Line2</div>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None]
    
    # Test with callable tag (should return empty string)
    dom = html.fromstring("<div></div>")
    dom.tag = lambda: None
    assert extract_text_array(dom) == ""
    
    # Test with text and tail
    dom = html.fromstring("<div>Start<span>Middle</span>End</div>")
    result = extract_text_array(dom)
    assert result == [None, "Start", "Middle", "End", None]
    
    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "Text", None, None]
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Text", None]


# LLM-generated content at query #125
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from xml.etree.ElementTree import Element, SubElement
    
    # Test 1: Simple text
    dom = Element('p')
    dom.text = 'Hello World'
    assert extract_text(dom) == 'Hello World'
    
    # Test 2: Inline elements should not add newlines
    dom = Element('p')
    dom.text = 'Hello '
    span = SubElement(dom, 'span')
    span.text = 'World'
    assert extract_text(dom) == 'Hello World'
    
    # Test 3: Block elements should add newlines
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = 'First paragraph'
    p2 = SubElement(dom, 'p')
    p2.text = 'Second paragraph'
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'
    
    # Test 4: Separator elements (br)
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = 'Line1'
    br = SubElement(p, 'br')
    br.tail = 'Line2'
    assert extract_text(dom) == 'Line1\nLine2'
    
    # Test 5: Multiple block elements with nested inline
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = 'Text with '
    strong = SubElement(p1, 'strong')
    strong.text = 'bold'
    strong.tail = ' text'
    p2 = SubElement(dom, 'p')
    p2.text = 'Another paragraph'
    assert extract_text(dom) == 'Text with bold text\nAnother paragraph'
    
    # Test 6: Custom block_symbol and sep_symbol
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = 'First'
    br = SubElement(p1, 'br')
    br.tail = 'Second'
    p2 = SubElement(dom, 'p')
    p2.text = 'Third'
    assert extract_text(dom, block_symbol='|', sep_symbol='-') == 'First-Second|Third'
    
    # Test 7: Empty element
    dom = Element('div')
    assert extract_text(dom) == ''
    
    # Test 8: Nested block elements
    dom = Element('div')
    outer = SubElement(dom, 'div')
    outer.text = 'Outer '
    inner = SubElement(outer, 'p')
    inner.text = 'Inner'
    assert extract_text(dom) == 'Outer Inner'
    
    # Test 9: Whitespace handling
    dom = Element('p')
    dom.text = '  Hello   World  '
    assert extract_text(dom) == 'Hello World'
    
    # Test 10: Multiple whitespace characters
    dom = Element('p')
    dom.text = 'Hello\t\n\rWorld'
    assert extract_text(dom) == 'Hello World'
    
    # Test 11: squash_space=False
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = 'First'
    p2 = SubElement(dom, 'p')
    p2.text = 'Second'
    assert extract_text(dom, squash_space=False) == '\nFirst\nSecond\n'


# LLM-generated content at query #126
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml.html import fromstring
    dom = fromstring("<p>Hello World</p>")
    assert extract_text_array(dom) == ["Hello World"]

    # Test with inline tags
    dom = fromstring("<p>Hello <b>World</b></p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]

    # Test with separator tag
    dom = fromstring("<p>Line1<br>Line2</p>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]

    # Test with block-level tag
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, "First", None, None, "Second", None]

    # Test with nested tags
    dom = fromstring("<div><span>Text</span></div>")
    result = extract_text_array(dom)
    assert result == ["Text"]

    # Test with empty element
    dom = fromstring("<p></p>")
    assert extract_text_array(dom) == []

    # Test with text and tail
    dom = fromstring("<p>Hello<b>bold</b>world</p>")
    result = extract_text_array(dom)
    assert result == ["Hello", "bold", "world"]

    # Test with multiple inline tags
    dom = fromstring("<p><i>italic</i> and <b>bold</b></p>")
    result = extract_text_array(dom)
    assert result == ["italic", " and ", "bold"]

    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>Test</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result

    # Test with strip_artifical_nl=False
    dom = fromstring("<div><p>Test</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None and result[-1] is None

    # Test with callable tag
    class MockElement:
        def __init__(self):
            self.tag = lambda: None
    assert extract_text_array(MockElement()) == ""


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    from lxml import html
    doc = html.fromstring("<p>Hello world</p>")
    result = extract_text(doc)
    assert result == "Hello world"

    # Test with multiple inline elements
    doc = html.fromstring("<p>Hello <b>world</b> foo</p>")
    result = extract_text(doc)
    assert result == "Hello world foo"

    # Test with block elements
    doc = html.fromstring("<div><p>First paragraph</p><p>Second paragraph</p></div>")
    result = extract_text(doc)
    assert result == "First paragraph\nSecond paragraph"

    # Test with separators (br)
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text(doc)
    assert result == "Line1\nLine2"

    # Test with nested elements
    doc = html.fromstring("<div><span><b>Nested</b> text</span></div>")
    result = extract_text(doc)
    assert result == "Nested text"

    # Test with whitespace normalization
    doc = html.fromstring("<p>Hello    world</p>")
    result = extract_text(doc)
    assert result == "Hello world"

    # Test with empty content
    doc = html.fromstring("<p></p>")
    result = extract_text(doc)
    assert result == ""

    # Test with multiple separators
    doc = html.fromstring("<p>Line1<br/><br/>Line2</p>")
    result = extract_text(doc)
    assert result == "Line1\nLine2"

    # Test with block elements and nested inline
    doc = html.fromstring("<div><h1>Title</h1><p>Content with <b>bold</b> text</p></div>")
    result = extract_text(doc)
    assert result == "Title\nContent with bold text"

    # Test with custom block symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(doc, block_symbol=' | ')
    assert result == "First | Second"

    # Test with custom separator symbol
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text(doc, sep_symbol=' - ')
    assert result == "Line1 - Line2"

    # Test with squash_space=False
    doc = html.fromstring("<p>Hello    world</p>")
    result = extract_text(doc, squash_space=False)
    assert result == "Hello    world"

    # Test with complex nested structure
    doc = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>First paragraph</p>
            <p>Second paragraph<br/>with break</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(doc)
    assert "Title" in result
    assert "First paragraph" in result
    assert "Second paragraph" in result
    assert "with break" in result
    assert "Item 1" in result
    assert "Item 2" in result
    assert result.count('\n') > 0

    # Test with non-inline tag that is not a block element
    doc = html.fromstring("<div>Text</div>")
    result = extract_text(doc)
    assert result == "Text"

    # Test with mixed inline and block elements
    doc = html.fromstring("<p>Some <b>bold</b> and <i>italic</i> text</p>")
    result = extract_text(doc)
    assert result == "Some bold and italic text"


# LLM-generated content at query #2
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml.html import fromstring
    dom = fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == ['Hello World']

    # Test with inline tags (should not add artificial newlines)
    dom = fromstring("<p>Hello <b>World</b></p>")
    result = extract_text_array(dom)
    assert result == ['Hello ', 'World']

    # Test with separator tag (br)
    dom = fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert True in result  # Should contain separator marker

    # Test with block-level tags (should add artificial newlines)
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert None in result  # Should contain artificial newline markers

    # Test with nested elements
    dom = fromstring("<div>Text <span>inside</span> span</div>")
    result = extract_text_array(dom)
    assert result == ['Text ', 'inside', ' span']

    # Test with empty element
    dom = fromstring("<p></p>")
    result = extract_text_array(dom)
    assert result == []

    # Test with element having only text
    dom = fromstring("<span>Only text</span>")
    result = extract_text_array(dom)
    assert result == ['Only text']

    # Test with multiple separators
    dom = fromstring("<p>First<br/>Second<br/>Third</p>")
    result = extract_text_array(dom)
    assert result.count(True) == 2  # Two br tags

    # Test squash_artifical_nl parameter
    dom = fromstring("<div><p>Text</p></div>")
    result_no_squash = extract_text_array(dom, squash_artifical_nl=False)
    result_squash = extract_text_array(dom, squash_artifical_nl=True)
    assert len(result_no_squash) > len(result_squash)

    # Test strip_artifical_nl parameter
    dom = fromstring("<div><p>Text</p></div>")
    result_no_strip = extract_text_array(dom, strip_artifical_nl=False)
    result_strip = extract_text_array(dom, strip_artifical_nl=True)
    assert len(result_no_strip) >= len(result_strip)


# LLM-generated content at query #3
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from html.parser import HTMLParser
    from lxml.html import fromstring
    
    # Test basic text extraction
    dom = fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello World", None]
    
    # Test with nested inline tags
    dom = fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert None in result
    assert "Hello " in result
    assert "bold" in result
    assert " world" in result
    
    # Test with separator tag (br)
    dom = fromstring("<p>Line1<br>Line2</p>")
    result = extract_text_array(dom)
    assert True in result  # separator marker
    
    # Test with non-inline tags
    dom = fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert None in result  # artificial newline markers
    
    # Test empty element
    dom = fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test element with no text
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test squash_artifical_nl=True (default)
    dom = fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    # Should have consecutive None markers squashed
    none_count = sum(1 for x in result if x is None)
    assert none_count <= 2  # At most one at start and one at end
    
    # Test squash_artifical_nl=False
    dom = fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    # Should have multiple consecutive None markers
    none_count = sum(1 for x in result if x is None)
    assert none_count > 2
    
    # Test strip_artifical_nl=True (default)
    dom = fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    if result:
        assert result[0] is None  # Should start with None (block marker)
        assert result[-1] is None  # Should end with None (block marker)
    
    # Test strip_artifical_nl=False
    dom = fromstring("<p>Text</p>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    # Should have leading/trailing None markers
    if result:
        assert result[0] is None
        assert result[-1] is None
    
    # Test complex nested structure
    html = """
    <div>
        <p>First <b>paragraph</b></p>
        <br>
        <p>Second <i>paragraph</i></p>
    </div>
    """
    dom = fromstring(html)
    result = extract_text_array(dom)
    assert result is not None
    assert len(result) > 0
    
    # Test with callable tag (should return empty string)
    class FakeDom:
        def __init__(self):
            self.tag = lambda: None
    fake_dom = FakeDom()
    result = extract_text_array(fake_dom)
    assert result == ''


# LLM-generated content at query #4
#--------------------------

```python
def test_extract_text():
    # Test with a simple div containing text
    from lxml import html
    doc = html.fromstring("<div>Hello World</div>")
    assert extract_text(doc) == "Hello World"
    
    # Test with inline tags (should not add newlines)
    doc = html.fromstring("<p>Hello <strong>World</strong></p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with block elements (should add newlines)
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with separator tags (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with nested structure
    doc = html.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    assert extract_text(doc) == "Para1\nPara2"
    
    # Test with empty elements
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with only whitespace
    doc = html.fromstring("<div>   </div>")
    assert extract_text(doc) == ""
    
    # Test with custom block_symbol
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with squash_space=False
    doc = html.fromstring("<div>  Hello   World  </div>")
    assert extract_text(doc, squash_space=False) == "  Hello   World  "
    
    # Test with mixed inline and block elements
    doc = html.fromstring("<div><span>Inline</span><p>Block</p></div>")
    assert extract_text(doc) == "Inline\nBlock"
    
    # Test with complex nesting
    doc = html.fromstring("<div><p>Para1<br>Break</p><p>Para2</p></div>")
    assert extract_text(doc) == "Para1\nBreak\nPara2"
    
    # Test with multiple whitespace handling
    doc = html.fromstring("<p>Hello   World</p>")
    assert extract_text(doc) == "Hello World"
```


# LLM-generated content at query #5
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element - should add newline
    dom = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator element (br)
    dom = html.fromstring("Line1<br>Line2")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested elements
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with whitespace normalization
    dom = html.fromstring("<p>Hello    World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with multiple block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with empty element
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with text and tail
    dom = html.fromstring("<p>Hello<b>bold</b>world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with custom separators
    dom = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom, block_symbol=' ', sep_symbol=' ') == "First Second"
    
    # Test with no squashing
    dom = html.fromstring("<p>Hello    World</p>")
    assert extract_text(dom, squash_space=False) == "Hello    World"
    
    # Test with nested blocks
    dom = html.fromstring("<div><div><p>Deep</p></div></div>")
    assert extract_text(dom) == "Deep"
    
    # Test with img tag (inline but special)
    dom = html.fromstring("<p>Text<img src='test.png' alt='image'>more text</p>")
    assert extract_text(dom) == "Text more text"


# LLM-generated content at query #6
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with block elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with separators (br)
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with nested elements
    doc = html.fromstring("<div><p>Text with <b>bold</b> and <i>italic</i></p></div>")
    assert extract_text(doc) == "Text with bold and italic"
    
    # Test with whitespace handling
    doc = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with multiple block elements
    doc = html.fromstring("<div><h1>Title</h1><p>Content</p></div>")
    assert extract_text(doc) == "Title\nContent"
    
    # Test with empty element
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with text only
    doc = html.fromstring("<p>Just text</p>")
    assert extract_text(doc) == "Just text"
    
    # Test with custom block_symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello   world  "
    
    # Test with list elements
    doc = html.fromstring("<ul><li>Item1</li><li>Item2</li></ul>")
    assert extract_text(doc) == "Item1\nItem2"
    
    # Test with anchor element
    doc = html.fromstring("<p>Visit <a href='#'>here</a></p>")
    assert extract_text(doc) == "Visit here"
```


# LLM-generated content at query #7
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]

    # Test with block tag
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == ["Hello"]  # None values are stripped by default

    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]  # True represents separator

    # Test with nested tags
    dom = html.fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text_array(dom)
    assert "Hello" in result
    assert "World" in result

    # Test with text and tail
    dom = html.fromstring("<div>Hello <b>bold</b> text</div>")
    result = extract_text_array(dom)
    assert "Hello " in result
    assert "bold" in result
    assert " text" in result

    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result  # None values should be preserved

    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div><p>Hello</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # Leading None should be preserved

    # Test with callable tag
    class FakeElement:
        tag = lambda: None
    assert extract_text_array(FakeElement()) == ""

    # Test with empty element
    dom = html.fromstring("<div></div>")
    assert extract_text_array(dom) == []

    # Test with inline tag containing text
    dom = html.fromstring("<strong>Important</strong>")
    assert extract_text_array(dom) == ["Important"]

    # Test with multiple separators
    dom = html.fromstring("<div>Line1<br/>Line2<br/>Line3</div>")
    result = extract_text_array(dom)
    assert result.count(True) == 2  # Two br tags
    assert "Line1" in result
    assert "Line2" in result
    assert "Line3" in result


# LLM-generated content at query #8
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline text
    from lxml import html
    dom = html.fragment_fromstring("<span>Hello World</span>")
    assert extract_text_array(dom) == ["Hello World"]
    
    # Test with separator tag (br)
    dom = html.fragment_fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with block tag
    dom = html.fragment_fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result == ["Text"]
    assert None in result  # Should have artificial newlines
    
    # Test with nested elements
    dom = html.fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert "First" in result
    assert "Second" in result
    assert None in result
    
    # Test with inline elements inside block
    dom = html.fragment_fromstring("<p>Hello <strong>World</strong></p>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with br separator
    dom = html.fragment_fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert True in result  # Should contain separator
    
    # Test without squash_artifical_nl
    dom = html.fragment_fromstring("<div>Text</div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result
    assert len([x for x in result if x is None]) == 2  # Both before and after
    
    # Test without strip_artifical_nl
    dom = html.fragment_fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # Starts with artificial newline
    
    # Test with non-callable tag
    class MockElement:
        tag = "div"
        text = "Test"
        tail = None
        def getchildren(self):
            return []
    mock = MockElement()
    result = extract_text_array(mock)
    assert "Test" in result
    
    # Test with callable tag (should return empty string)
    class MockCallableTag:
        def tag(self):
            pass
        text = "Test"
        def getchildren(self):
            return []
    mock = MockCallableTag()
    assert extract_text_array(mock) == ''


# LLM-generated content at query #9
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    doc = html.fromstring('<p>Hello <b>world</b></p>')
    assert extract_text(doc) == 'Hello world'
    
    # Test with block elements
    doc = html.fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(doc) == 'First\nSecond'
    
    # Test with separator elements (br)
    doc = html.fromstring('<p>Line1<br/>Line2</p>')
    assert extract_text(doc) == 'Line1\nLine2'
    
    # Test with nested elements
    doc = html.fromstring('<div><span>Text</span><b>Bold</b></div>')
    assert extract_text(doc) == 'TextBold'
    
    # Test with whitespace normalization
    doc = html.fromstring('<p>  Hello   World  </p>')
    assert extract_text(doc) == 'Hello World'
    
    # Test with empty content
    doc = html.fromstring('<div></div>')
    assert extract_text(doc) == ''
    
    # Test with multiple levels of nesting
    doc = html.fromstring('<div><ul><li>Item1</li><li>Item2</li></ul></div>')
    assert extract_text(doc) == 'Item1\nItem2'
    
    # Test with custom block_symbol
    doc = html.fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(doc, block_symbol=' ') == 'First Second'
    
    # Test with custom sep_symbol
    doc = html.fromstring('<p>Line1<br/>Line2</p>')
    assert extract_text(doc, sep_symbol=' ') == 'Line1 Line2'
    
    # Test with squash_space=False
    doc = html.fromstring('<p>  Hello   World  </p>')
    assert extract_text(doc, squash_space=False) == '  Hello   World  '
    
    # Test with mixed content
    doc = html.fromstring('<div><p>Hello <b>world</b></p><p>Second <i>paragraph</i></p></div>')
    assert extract_text(doc) == 'Hello world\nSecond paragraph'
```


# LLM-generated content at query #10
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml.html import fromstring
    dom = fromstring("<span>hello</span>")
    result = extract_text_array(dom)
    assert result == ["hello"]

    # Test with separator tag (br)
    dom = fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [True]

    # Test with block tag (div)
    dom = fromstring("<div>text</div>")
    result = extract_text_array(dom)
    assert result == [None, "text", None]

    # Test with nested tags
    dom = fromstring("<div><span>hello</span> world</div>")
    result = extract_text_array(dom)
    assert result == [None, "hello", " world", None]

    # Test with multiple children
    dom = fromstring("<div><p>first</p><p>second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "first", None, None, None, "second", None, None]

    # Test with br separator
    dom = fromstring("<div>line1<br>line2</div>")
    result = extract_text_array(dom)
    assert result == [None, "line1", True, "line2", None]

    # Test with squash_artifical_nl=True (default)
    dom = fromstring("<div><p>text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == [None, "text", None]  # consecutive Nones squashed

    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "text", None, None]  # all Nones preserved

    # Test with strip_artifical_nl=True (default)
    dom = fromstring("<div><span>text</span></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["text"]  # leading/trailing Nones stripped

    # Test with strip_artifical_nl=False
    dom = fromstring("<div><span>text</span></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "text", None]  # leading/trailing Nones preserved

    # Test empty element
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []

    # Test element with only text
    dom = fromstring("<p>hello</p>")
    result = extract_text_array(dom)
    assert result == [None, "hello", None]

    # Test callable tag (should return empty string)
    dom = fromstring("<div>text</div>")
    dom.tag = lambda: None  # make it callable
    result = extract_text_array(dom)
    assert result == ''


# LLM-generated content at query #11
#--------------------------

```python
def test_extract_text():
    # Test with simple text
    class MockElement:
        tag = 'p'
        text = 'Hello World'
        def getchildren(self):
            return []
    dom = MockElement()
    assert extract_text(dom) == 'Hello World'
    
    # Test with inline tags
    class MockInlineElement:
        tag = 'span'
        text = 'inline'
        def getchildren(self):
            return []
    class MockParent:
        tag = 'p'
        text = 'Start '
        def getchildren(self):
            return [MockInlineElement()]
        tail = None
    dom = MockParent()
    assert extract_text(dom) == 'Start inline'
    
    # Test with block elements
    class MockBlockChild:
        tag = 'div'
        text = 'Block1'
        def getchildren(self):
            return []
        tail = ' '
    class MockBlockParent:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockBlockChild()]
        tail = None
    dom = MockBlockParent()
    assert extract_text(dom) == 'Block1'
    
    # Test with separator tags (br)
    class MockBr:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
        tail = None
    class MockWithBr:
        tag = 'p'
        text = 'Line1'
        def getchildren(self):
            return [MockBr()]
        tail = 'Line2'
    dom = MockWithBr()
    assert extract_text(dom) == 'Line1\nLine2'
    
    # Test with nested elements
    class MockNestedInner:
        tag = 'strong'
        text = 'bold'
        def getchildren(self):
            return []
        tail = ' text'
    class MockNestedOuter:
        tag = 'p'
        text = 'Some '
        def getchildren(self):
            return [MockNestedInner()]
        tail = None
    dom = MockNestedOuter()
    assert extract_text(dom) == 'Some bold text'
    
    # Test with empty element
    class MockEmpty:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockEmpty()
    assert extract_text(dom) == ''
    
    # Test with multiple block elements
    class MockBlock1:
        tag = 'div'
        text = 'First'
        def getchildren(self):
            return []
        tail = None
    class MockBlock2:
        tag = 'div'
        text = 'Second'
        def getchildren(self):
            return []
        tail = None
    class MockContainer:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockBlock1(), MockBlock2()]
        tail = None
    dom = MockContainer()
    assert extract_text(dom) == 'First\nSecond'


# LLM-generated content at query #12
#--------------------------

```python
def test_extract_text():
    # Test simple inline text
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test nested tags
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test inline tags without line breaks
    dom = html.fromstring("<p><strong>Bold</strong> and <em>italic</em></p>")
    assert extract_text(dom) == "Bold and italic"
    
    # Test separator tags (br)
    dom = html.fromstring("<p>Line 1<br/>Line 2</p>")
    assert extract_text(dom) == "Line 1\nLine 2"
    
    # Test whitespace squashing
    dom = html.fromstring("<p>   Extra    spaces   </p>")
    assert extract_text(dom) == "Extra spaces"
    
    # Test empty content
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test deeply nested structure
    dom = html.fromstring("<div><ul><li>Item 1</li><li>Item 2</li></ul></div>")
    result = extract_text(dom)
    assert result == "Item 1\nItem 2"
    
    # Test custom block_symbol
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "A | B"
    
    # Test custom sep_symbol
    dom = html.fromstring("<p>A<br/>B</p>")
    assert extract_text(dom, sep_symbol=' | ') == "A | B"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>   Extra    spaces   </p>")
    assert extract_text(dom, squash_space=False) == "Extra    spaces"
    
    # Test with all parameters customized
    dom = html.fromstring("<div><p>Hello<br/>World</p><p>Second</p></div>")
    result = extract_text(dom, block_symbol=' | ', sep_symbol=' - ', squash_space=False)
    assert result == "Hello - World | Second"
    
    # Test with script tag (should be ignored)
    dom = html.fromstring("<p>Text<script>var x = 1;</script>More</p>")
    assert extract_text(dom) == "TextMore"
    
    # Test with nested inline tags
    dom = html.fromstring("<p><span><strong>Deep</strong></span></p>")
    assert extract_text(dom) == "Deep"
    
    # Test multiple br tags
    dom = html.fromstring("<p>A<br/><br/>B</p>")
    assert extract_text(dom) == "A\n\nB"
```


# LLM-generated content at query #13
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import etree
    dom = etree.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element
    dom = etree.fromstring("<div>Hello</div>")
    assert extract_text(dom) == "Hello"
    
    # Test with nested elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator element (br)
    dom = etree.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with inline elements inside block
    dom = etree.fromstring("<p>Hello <strong>World</strong></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with multiple whitespace
    dom = etree.fromstring("<p>Hello    World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with whitespace around tags
    dom = etree.fromstring("<p>  Hello World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with empty element
    dom = etree.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with custom block_symbol
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "First | Second"
    
    # Test with custom sep_symbol
    dom = etree.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with squash_space=False
    dom = etree.fromstring("<p>Hello  World</p>")
    assert extract_text(dom, squash_space=False) == "Hello  World"
    
    # Test with nested blocks
    dom = etree.fromstring("<div><div><p>Deep</p></div></div>")
    assert extract_text(dom) == "Deep"


# LLM-generated content at query #14
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tags
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"

    # Test with separators (br tags)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"

    # Test with block elements
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc) == "First\nSecond"

    # Test with nested inline elements
    doc = html.fromstring("<p>Text with <span>nested <em>emphasis</em></span> here</p>")
    assert extract_text(doc) == "Text with nested emphasis here"

    # Test with whitespace handling
    doc = html.fromstring("<p>  Hello    world  </p>")
    assert extract_text(doc) == "Hello world"

    # Test empty content
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""

    # Test with only whitespace
    doc = html.fromstring("<p>   </p>")
    assert extract_text(doc) == ""

    # Test with custom block_symbol
    doc = html.fromstring("<div>A</div><div>B</div>")
    assert extract_text(doc, block_symbol=' | ') == "A | B"

    # Test with custom sep_symbol
    doc = html.fromstring("<p>A<br>B</p>")
    assert extract_text(doc, sep_symbol=' | ') == "A | B"

    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello    world  </p>")
    result = extract_text(doc, squash_space=False)
    assert "  " in result  # Should preserve some whitespace

    # Test with nested block elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"

    # Test with mixed inline and block elements
    doc = html.fromstring("<div>Some <span>inline</span> text</div><div>More text</div>")
    assert extract_text(doc) == "Some inline text\nMore text"

    # Test with script tags (inline but should be handled)
    doc = html.fromstring("<div>Text <script>var x = 1;</script> more text</div>")
    assert extract_text(doc) == "Text  more text"

    # Test with multiple br tags
    doc = html.fromstring("<p>A<br><br>B</p>")
    assert extract_text(doc) == "A\n\nB"

    # Test with leading/trailing whitespace in nested elements
    doc = html.fromstring("<div>  <p>  Hello  </p>  </div>")
    assert extract_text(doc) == "Hello"

    # Test with no block elements (all inline)
    doc = html.fromstring("<span>Inline</span> <span>content</span>")
    assert extract_text(doc) == "Inline content"

    # Test with empty tags
    doc = html.fromstring("<div><br></div>")
    assert extract_text(doc) == "\n"

    # Test complex nested structure
    doc = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <strong>bold</strong> text</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(doc)
    assert "Title" in result
    assert "Paragraph with bold text" in result
    assert "Item 1" in result
    assert "Item 2" in result
```


# LLM-generated content at query #15
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<span>hello</span>")
    result = extract_text_array(dom)
    assert result == ["hello"]

    # Test with separator tag (br)
    dom = fragment_fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [True]

    # Test with block-level tag
    dom = fragment_fromstring("<div>text</div>")
    result = extract_text_array(dom)
    assert result == ["text"]

    # Test with nested inline tags
    dom = fragment_fromstring("<span>hello <b>world</b></span>")
    result = extract_text_array(dom)
    assert result == ["hello ", "world"]

    # Test with nested block-level tag
    dom = fragment_fromstring("<div>hello <p>world</p></div>")
    result = extract_text_array(dom)
    assert result == ["hello ", "world"]

    # Test with tail text
    dom = fragment_fromstring("<div><span>hello</span> world</div>")
    result = extract_text_array(dom)
    assert result == ["hello", " world"]

    # Test with multiple separators
    dom = fragment_fromstring("<br><br>")
    result = extract_text_array(dom)
    assert result == [True, True]

    # Test with None text
    dom = fragment_fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []

    # Test with squash_artifical_nl=False
    dom = fragment_fromstring("<div>hello</div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, "hello", None]

    # Test with strip_artifical_nl=False
    dom = fragment_fromstring("<div>hello</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "hello", None]

    # Test with callable tag (should return empty)
    class MockTag:
        def __call__(self):
            pass
    dom = fragment_fromstring("<div>text</div>")
    dom.tag = MockTag()
    result = extract_text_array(dom)
    assert result == ""

    # Test with empty dom
    dom = fragment_fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []


# LLM-generated content at query #16
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b>!</p>")
    assert extract_text(doc) == "Hello world!"
    
    # Test with block elements creating newlines
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with separator elements (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with nested elements
    doc = html.fromstring("<div><span>Hello</span> <span>World</span></div>")
    assert extract_text(doc) == "Hello World"
    
    # Test with multiple levels of nesting
    doc = html.fromstring("<div><p><b>Bold</b> text</p><p>More text</p></div>")
    assert extract_text(doc) == "Bold text\nMore text"
    
    # Test with whitespace handling
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with empty elements
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with only whitespace
    doc = html.fromstring("<p>   </p>")
    assert extract_text(doc) == ""
    
    # Test custom block and sep symbols
    doc = html.fromstring("<div><p>First</p><br><p>Second</p></div>")
    assert extract_text(doc, block_symbol=" | ", sep_symbol=" - ") == "First - Second"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>Hello  World</p>")
    assert extract_text(doc, squash_space=False) == "Hello  World"
    
    # Test complex document
    doc = html.fromstring("""
        <html>
            <body>
                <h1>Title</h1>
                <p>Paragraph with <a href="#">link</a> inside</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
        </html>
    """)
    result = extract_text(doc)
    assert "Title" in result
    assert "Paragraph with link inside" in result
    assert "Item 1" in result
    assert "Item 2" in result
    
    # Test with inline elements that should not create newlines
    doc = html.fromstring("<p><strong>Bold</strong> and <em>italic</em></p>")
    assert extract_text(doc) == "Bold and italic"


# LLM-generated content at query #17
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    class MockElement:
        tag = 'span'
        text = 'Hello'
        def getchildren(self):
            return []
    assert extract_text_array(MockElement()) == ['Hello']

    # Test with block element (should add None markers)
    class MockDiv:
        tag = 'div'
        text = 'Hello'
        def getchildren(self):
            return []
    result = extract_text_array(MockDiv())
    assert result[0] is None
    assert 'Hello' in result
    assert result[-1] is None

    # Test with separator element (br)
    class MockBr:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    result = extract_text_array(MockBr())
    assert True in result

    # Test with nested elements
    class MockChild:
        tag = 'span'
        text = 'World'
        tail = '!'
        def getchildren(self):
            return []
    
    class MockParent:
        tag = 'div'
        text = 'Hello '
        def getchildren(self):
            return [MockChild()]
    
    result = extract_text_array(MockParent())
    assert result[0] is None
    assert 'Hello ' in result
    assert 'World' in result
    assert '!' in result
    assert result[-1] is None

    # Test with multiple children and separators
    class MockBr2:
        tag = 'br'
        text = None
        tail = '\n'
        def getchildren(self):
            return []
    
    class MockContainer:
        tag = 'div'
        text = 'Line1'
        def getchildren(self):
            return [MockBr2()]
    
    result = extract_text_array(MockContainer())
    assert True in result
    assert '\n' in result

    # Test with callable tag (edge case)
    class MockCallable:
        tag = lambda: None
        text = 'Should be empty'
        def getchildren(self):
            return []
    assert extract_text_array(MockCallable()) == ''

    # Test squash_artifical_nl parameter
    class MockBlock:
        tag = 'div'
        text = 'A'
        def getchildren(self):
            return []
    
    # Without squashing
    result_no_squash = extract_text_array(MockBlock(), squash_artifical_nl=False)
    none_count_no_squash = sum(1 for x in result_no_squash if x is None)
    assert none_count_no_squash == 2

    # With squashing (default)
    result_squash = extract_text_array(MockBlock(), squash_artifical_nl=True)
    none_count_squash = sum(1 for x in result_squash if x is None)
    assert none_count_squash == 1

    # Test strip_artifical_nl parameter
    class MockBlockWithText:
        tag = 'div'
        text = 'Content'
        def getchildren(self):
            return []
    
    # Without stripping
    result_no_strip = extract_text_array(MockBlockWithText(), strip_artifical_nl=False)
    assert result_no_strip[0] is None
    assert result_no_strip[-1] is None

    # With stripping (default)
    result_strip = extract_text_array(MockBlockWithText(), strip_artifical_nl=True)
    assert result_strip[0] == 'Content'
    assert len(result_strip) == 1


# LLM-generated content at query #18
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml.html import fromstring
    dom = fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with separator tag (br)
    dom = fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with block tag
    dom = fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result == [None, "Text", None]
    
    # Test with nested tags
    dom = fromstring("<div><span>Hello</span> World</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", " World", None]
    
    # Test with multiple children
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, None, "Second", None, None]
    
    # Test with separator inside block
    dom = fromstring("<div>Line1<br>Line2</div>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None]
    
    # Test with inline tag inside block
    dom = fromstring("<div><b>Bold</b> text</div>")
    result = extract_text_array(dom)
    assert result == [None, "Bold", " text", None]
    
    # Test with callable tag (should return empty string)
    from lxml.etree import Comment
    dom = fromstring("<div><!-- comment --></div>")
    # Comment tag is callable, so extract_text_array returns ''
    result = extract_text_array(dom.find('.//comment()'))
    assert result == ''
    
    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>Para</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "Para", None, None]
    
    # Test with strip_artifical_nl=False
    dom = fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Text", None]
    
    # Test with both flags False
    dom = fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None, "Text", None, None]
    
    # Test with empty tag
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test with text only (no tags)
    dom = fromstring("Just text")
    result = extract_text_array(dom)
    assert result == ["Just text"]


# LLM-generated content at query #19
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    
    # Test 1: Simple text
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 2: Text with inline tags
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 3: Text with line break
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol='\n') == "Line1\nLine2"
    
    # Test 4: Text with multiple block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First Second"
    
    # Test 5: Text with block elements and custom symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' ') == "First Second"
    
    # Test 6: Complex nested structure
    dom = html.fromstring("<div><p>Hello <b>World</b></p><p>Second <i>line</i></p></div>")
    assert extract_text(dom) == "Hello World Second line"
    
    # Test 7: Text with separator tags
    dom = html.fromstring("<div>Before<br>After</div>")
    assert extract_text(dom, sep_symbol='\n') == "Before\nAfter"
    
    # Test 8: Empty element
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test 9: Mixed inline and block elements
    dom = html.fromstring("<div><span>Text</span><p>Paragraph</p></div>")
    assert extract_text(dom) == "Text Paragraph"
    
    # Test 10: Test with squash_space=False
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test 11: Test with block_symbol='\n'
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol='\n') == "First\nSecond"
    
    # Test 12: Deeply nested structure
    dom = html.fromstring("<div><div><p>Nested <b>text</b></p></div></div>")
    assert extract_text(dom) == "Nested text"


# LLM-generated content at query #20
#--------------------------

```python
def test_extract_text_array():
    # Test with text only
    from lxml import html
    dom = html.fragment_fromstring("Hello World")
    assert extract_text_array(dom) == ["Hello World"]
    
    # Test with inline tag
    dom = html.fragment_fromstring("<b>Bold</b> text")
    result = extract_text_array(dom)
    assert result == ["Bold text"]
    # Note: The space after </b> might be handled differently, adjust based on behavior
    
    # Test with block tag
    dom = html.fragment_fromstring("<div>Block</div>")
    result = extract_text_array(dom)
    # Block tags add None at start and end
    assert result[0] is None and result[-1] is None
    assert result[1:-1] == ["Block"]
    
    # Test with separator tag (br)
    dom = html.fragment_fromstring("Line1<br>Line2")
    result = extract_text_array(dom)
    # br adds True (separator) between text
    assert result[0] == "Line1" and result[1] is True and result[2] == "Line2"
    
    # Test with nested tags
    dom = html.fragment_fromstring("<div><span>Inner</span></div>")
    result = extract_text_array(dom)
    # Block tag -> None, inline span -> "Inner", block tag -> None
    assert len(result) == 3
    assert result[0] is None and result[1] == "Inner" and result[2] is None
    
    # Test with tail text
    dom = html.fragment_fromstring("<div>Before<b>Bold</b>After</div>")
    result = extract_text_array(dom)
    assert None in result
    assert "Before" in result and "Bold" in result and "After" in result
    
    # Test squash_artifical_nl
    dom = html.fragment_fromstring("<div>Text</div>")
    result_no_squash = extract_text_array(dom, squash_artifical_nl=False)
    assert result_no_squash[0] is None and result_no_squash[-1] is None and result_no_squash[1] == "Text"
    result_squash = extract_text_array(dom, squash_artifical_nl=True)
    assert result_squash[0] is None and result_squash[1] == "Text"
    # Only one None because adjacent Nones are squashed
    
    # Test strip_artifical_nl
    dom = html.fragment_fromstring("<div></div>")
    result_no_strip = extract_text_array(dom, strip_artifical_nl=False)
    assert None in result_no_strip
    result_strip = extract_text_array(dom, strip_artifical_nl=True)
    assert result_strip == []  # Only None values are stripped
    
    # Test with callable tag (should return empty string)
    dom = html.HtmlElement()
    dom.tag = lambda: None
    assert extract_text_array(dom) == ""
    
    # Test empty dom
    from lxml import html as lh
    dom = lh.fromstring("<div></div>")
    result = extract_text_array(dom)
    # None at start and end, but squashed to one, then stripped
    assert result == [] or result == [None]  # Depending on squashing/stripping


# LLM-generated content at query #21
#--------------------------

```python
def test_extract_text_array():
    # Test with None dom
    assert extract_text_array(None) == []
    
    # Test with empty dom
    assert extract_text_array(type('obj', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: []})()) == []
    
    # Test with inline tag
    mock_span = type('obj', (object,), {
        'tag': 'span',
        'text': 'hello',
        'getchildren': lambda self: [],
        'tail': None
    })()
    result = extract_text_array(mock_span)
    assert 'hello' in result
    
    # Test with separator tag (br)
    mock_br = type('obj', (object,), {
        'tag': 'br',
        'text': None,
        'getchildren': lambda self: [],
        'tail': None
    })()
    result = extract_text_array(mock_br)
    assert True in result
    
    # Test with block tag (div)
    mock_div = type('obj', (object,), {
        'tag': 'div',
        'text': 'text',
        'getchildren': lambda self: [],
        'tail': None
    })()
    result = extract_text_array(mock_div)
    assert None in result
    assert 'text' in result
    
    # Test with nested elements
    mock_child = type('obj', (object,), {
        'tag': 'b',
        'text': 'bold',
        'getchildren': lambda self: [],
        'tail': ' tail'
    })()
    mock_parent = type('obj', (object,), {
        'tag': 'p',
        'text': 'start ',
        'getchildren': lambda self: [mock_child],
        'tail': None
    })()
    result = extract_text_array(mock_parent)
    assert 'start ' in result
    assert 'bold' in result
    assert ' tail' in result
    
    # Test squash_artifical_nl parameter
    result_no_squash = extract_text_array(mock_parent, squash_artifical_nl=False)
    result_squash = extract_text_array(mock_parent, squash_artifical_nl=True)
    assert result_no_squash != result_squash
    
    # Test strip_artifical_nl parameter
    result_no_strip = extract_text_array(mock_parent, strip_artifical_nl=False)
    result_strip = extract_text_array(mock_parent, strip_artifical_nl=True)
    assert result_no_strip != result_strip


# LLM-generated content at query #22
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text node
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = 'Hello World'
    result = extract_text_array(dom)
    assert result == ['Hello World'], f"Expected ['Hello World'], got {result}"

    # Test with inline tag
    dom = Element('span')
    dom.text = 'inline text'
    result = extract_text_array(dom)
    assert result == ['inline text'], f"Expected ['inline text'], got {result}"

    # Test with separator tag (br)
    dom = Element('br')
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"

    # Test with block-level tag (div)
    dom = Element('div')
    dom.text = 'block text'
    result = extract_text_array(dom)
    assert result == ['block text'], f"Expected ['block text'], got {result}"

    # Test nested structure
    parent = Element('div')
    child = Element('span')
    child.text = 'child text'
    child.tail = ' tail text'
    parent.append(child)
    result = extract_text_array(parent)
    # Expect: [None, 'child text', ' tail text', None] after squash/strip
    # Starting with None for block-level parent, then child text, then tail, then None
    # After _squash_artifical_nl: [None, 'child text', ' tail text', None] (no change)
    # After _strip_artifical_nl: ['child text', ' tail text']
    assert result == ['child text', ' tail text'], f"Expected ['child text', ' tail text'], got {result}"

    # Test with multiple children
    parent = Element('div')
    child1 = Element('p')
    child1.text = 'first'
    child2 = Element('p')
    child2.text = 'second'
    parent.append(child1)
    parent.append(child2)
    result = extract_text_array(parent)
    # After squash/strip: ['first', 'second']
    assert result == ['first', 'second'], f"Expected ['first', 'second'], got {result}"

    # Test with empty element
    dom = Element('div')
    result = extract_text_array(dom)
    assert result == [], f"Expected [], got {result}"

    # Test with br separator in inline context
    parent = Element('span')
    br = Element('br')
    parent.append(br)
    result = extract_text_array(parent)
    # br is separator, so it becomes True
    # After squash: [True]
    # After strip: [True] (no strings to strip)
    assert result == [True], f"Expected [True], got {result}"

    # Test artificial newline squashing
    parent = Element('div')
    child1 = Element('span')
    child1.text = 'a'
    child2 = Element('span')
    child2.text = 'b'
    parent.append(child1)
    parent.append(child2)
    result = extract_text_array(parent)
    # Original: [None, 'a', None, 'b', None]
    # After _squash_artifical_nl: [None, 'a', None, 'b', None] (no consecutive Nones)
    # After _strip_artifical_nl: ['a', None, 'b'] (stripping leading/trailing Nones)
    assert result == ['a', None, 'b'], f"Expected ['a', None, 'b'], got {result}"

    # Test with callable tag (should return empty string)
    dom = Element('div')
    dom.tag = lambda: None  # Make tag callable
    result = extract_text_array(dom)
    assert result == '', f"Expected '', got {result}"


# LLM-generated content at query #23
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with separator (br)
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with block-level elements
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with nested elements
    doc = html.fromstring("<div><p>Text with <span>span</span> inside</p></div>")
    assert extract_text(doc) == "Text with span inside"
    
    # Test with whitespace normalization
    doc = html.fromstring("<p>  Lots   of   spaces  </p>")
    assert extract_text(doc) == "Lots of spaces"
    
    # Test with custom block_symbol
    doc = html.fromstring("<div>Block1</div><div>Block2</div>")
    assert extract_text(doc, block_symbol=' | ') == "Block1 | Block2"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello  "
    
    # Test with empty text
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with mixed inline and block elements
    doc = html.fromstring("<div><span>inline</span><p>block</p></div>")
    assert extract_text(doc) == "inline\nblock"
    
    # Test with multiple separators
    doc = html.fromstring("<p>Text<br/><br/>More text</p>")
    assert extract_text(doc) == "Text\n\nMore text"


# LLM-generated content at query #24
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello world</p>")
    assert extract_text_array(dom) == [None, "Hello world", None]
    
    # Test with inline tag
    dom = etree.fromstring("<span>Inline text</span>")
    result = extract_text_array(dom)
    assert "Inline text" in result
    
    # Test with separator tag
    dom = etree.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with nested elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, None, "Second", None, None]
    
    # Test with inline and block elements
    dom = etree.fromstring("<div><span>inline</span><p>block</p></div>")
    result = extract_text_array(dom)
    assert "inline" in result
    assert "block" in result
    
    # Test with squash_artifical_nl=False
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert len(result) > len(extract_text_array(dom))
    
    # Test with strip_artifical_nl=False
    dom = etree.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # Should have leading artificial newline
    
    # Test with empty element
    dom = etree.fromstring("<div></div>")
    assert extract_text_array(dom) == [None, None]
    
    # Test with element containing only whitespace
    dom = etree.fromstring("<p>   </p>")
    assert extract_text_array(dom) == [None, "   ", None]
    
    # Test with callable tag (should return empty string)
    class FakeElement:
        tag = lambda: None
    assert extract_text_array(FakeElement()) == ""


# LLM-generated content at query #25
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element
    dom = fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator (br)
    dom = fragment_fromstring("<span>Line1<br>Line2</span>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested inline elements
    dom = fragment_fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with whitespace squashing
    dom = fragment_fromstring("<p>  Multiple   spaces   </p>")
    assert extract_text(dom) == "Multiple spaces"
    
    # Test with mixed inline and block
    dom = fragment_fromstring("<div><h1>Title</h1><p>Content</p></div>")
    assert extract_text(dom) == "Title\nContent"
    
    # Test with empty element
    dom = fragment_fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with script tag (should be treated as inline)
    dom = fragment_fromstring("<div>Before<script>alert('test')</script>After</div>")
    assert extract_text(dom) == "Beforealert('test')After"
    
    # Test with custom block_symbol
    dom = fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=" | ") == "First | Second"
    
    # Test with custom sep_symbol
    dom = fragment_fromstring("<span>Line1<br>Line2</span>")
    assert extract_text(dom, sep_symbol=" | ") == "Line1 | Line2"
    
    # Test with squash_space=False
    dom = fragment_fromstring("<p>  Multiple   spaces   </p>")
    assert extract_text(dom, squash_space=False) == "  Multiple   spaces   "


# LLM-generated content at query #26
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml.html import fromstring
    dom = fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"]

    # Test with block tag
    dom = fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", None]

    # Test with separator tag (br)
    dom = fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]

    # Test with nested inline tags
    dom = fromstring("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]

    # Test with nested block tags
    dom = fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "Text", None, None]

    # Test with text and tail
    dom = fromstring("<div>Hello<span>World</span>Again</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", "World", "Again", None]

    # Test with multiple separators
    dom = fromstring("<div>Line1<br/>Line2</div>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None]

    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "A", None, None, None, "B", None, None]

    # Test with strip_artifical_nl=False
    dom = fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, None, "A", None, None]

    # Test with both squash and strip disabled
    dom = fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None, "A", None, None]

    # Test empty content
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]

    # Test with callable tag (should return empty string)
    class MockCallable:
        def __call__(self):
            pass
        tag = MockCallable()
    mock_dom = type('MockDom', (), {'tag': MockCallable(), 'text': None, 'getchildren': lambda: []})()
    result = extract_text_array(mock_dom)
    assert result == ""

    # Test with multiple children and text nodes
    dom = fromstring("<div>Start<p>Middle</p>End</div>")
    result = extract_text_array(dom)
    assert result == [None, "Start", None, "Middle", None, "End", None]


# LLM-generated content at query #27
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with inline elements
    dom = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "bold", " world"]
    
    # Test with separator (br)
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]
    
    # Test with block-level elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, "First", None, None, "Second", None]
    
    # Test with nested elements
    dom = etree.fromstring("<div><p>Text with <span>inline</span> content</p></div>")
    result = extract_text_array(dom)
    assert result == [None, "Text with ", "inline", " content", None]
    
    # Test with empty element
    dom = etree.fromstring("<p></p>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test squash_artifical_nl=True
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, "A", None, "B", None]
    
    # Test strip_artifical_nl=True
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["A", None, "B"]
    
    # Test both squash and strip
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["A", None, "B"]
    
    # Test with tail text
    dom = etree.fromstring("<p>Start <b>bold</b> middle <i>italic</i> end</p>")
    result = extract_text_array(dom)
    assert result == ["Start ", "bold", " middle ", "italic", " end"]
    
    # Test with callable tag (should return empty string)
    class CallableTag:
        def __call__(self):
            pass
    mock_dom = type('MockDom', (), {'tag': CallableTag()})()
    result = extract_text_array(mock_dom)
    assert result == ''


# LLM-generated content at query #28
#--------------------------

```python
def test_extract_text():
    # Test with simple inline content
    from lxml import html
    doc = html.fromstring("<p>Hello World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with nested inline tags
    doc = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with block elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with separator tags (br)
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with mixed inline and block
    doc = html.fromstring("<div><span>Hello</span><p>World</p></div>")
    assert extract_text(doc) == "Hello\nWorld"
    
    # Test with multiple whitespace
    doc = html.fromstring("<p>Hello    World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with newlines in text
    doc = html.fromstring("<p>Hello\nWorld</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with empty content
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with nested structure
    doc = html.fromstring("<div><p><b>Bold</b> and <i>italic</i></p></div>")
    assert extract_text(doc) == "Bold and italic"
    
    # Test with multiple block elements
    doc = html.fromstring("<div><h1>Title</h1><p>Paragraph</p></div>")
    assert extract_text(doc) == "Title\nParagraph"
    
    # Test custom block_symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=' | ') == "First | Second"
    
    # Test custom sep_symbol
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc, sep_symbol=' - ') == "Line1 - Line2"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>Hello    World</p>")
    assert extract_text(doc, squash_space=False) == "Hello    World"
    
    # Test with deep nesting
    doc = html.fromstring("<div><p><span><b>Deep</b></span></p></div>")
    assert extract_text(doc) == "Deep"
    
    # Test with list elements
    doc = html.fromstring("<ul><li>Item 1</li><li>Item 2</li></ul>")
    assert extract_text(doc) == "Item 1\nItem 2"


# LLM-generated content at query #29
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline text
    from lxml.html import fromstring
    dom = fromstring("<p>Hello world</p>")
    result = extract_text_array(dom)
    assert result == ["Hello world"]

    # Test with nested inline elements
    dom = fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "bold", " world"]

    # Test with block-level elements (should insert None for artificial newlines)
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"]

    # Test with separator elements (br)
    dom = fromstring("<p>Line1<br>Line2</p>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]

    # Test with empty element
    dom = fromstring("<p></p>")
    result = extract_text_array(dom)
    assert result == []

    # Test with only whitespace
    dom = fromstring("<p>   </p>")
    result = extract_text_array(dom)
    assert result == ["   "]

    # Test with nested block elements
    dom = fromstring("<div><div>Deep</div></div>")
    result = extract_text_array(dom)
    assert result == ["Deep"]

    # Test with mixed inline and block elements
    dom = fromstring("<div>Text <span>inline</span> <p>block</p></div>")
    result = extract_text_array(dom)
    assert result == ["Text ", "inline", " ", "block"]

    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result  # Should have artificial newlines preserved

    # Test with strip_artifical_nl=False
    dom = fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None  # Should have leading/trailing None

    # Test with callable tag (should return empty string)
    class CallableTag:
        def __call__(self):
            pass
    dom = fromstring("<div></div>")
    dom.tag = CallableTag()
    result = extract_text_array(dom)
    assert result == ""


# LLM-generated content at query #30
#--------------------------

```python
def test_extract_text():
    from lxml import html
    
    # Test basic text extraction
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with inline tags
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block tags
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separators
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested tags
    dom = html.fromstring("<div><p>Hello <b>World</b></p><p>Foo</p></div>")
    assert extract_text(dom) == "Hello World\nFoo"
    
    # Test with whitespace
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with multiple whitespace
    dom = html.fromstring("<p>Hello     World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with newlines in text
    dom = html.fromstring("<p>Hello\nWorld</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test empty text
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test custom block symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test custom separator symbol
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol='|') == "Line1|Line2"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "Hello World"  # still squashed because squash_space only affects artificial newlines
    
    # Test with multiple block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p><p>Third</p></div>")
    assert extract_text(dom) == "First\nSecond\nThird"
    
    # Test with mixed content
    dom = html.fromstring("<p>Hello <b>bold</b> and <i>italic</i> world</p>")
    assert extract_text(dom) == "Hello bold and italic world"
    
    # Test with nested block elements
    dom = html.fromstring("<div><div><p>Nested</p></div></div>")
    assert extract_text(dom) == "Nested"
    
    # Test with script tag (inline)
    dom = html.fromstring("<p>Hello <script>alert('test')</script> World</p>")
    assert extract_text(dom) == "Hello alert('test') World"  # script content is extracted
    
    # Test with br tags in sequence
    dom = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"  # multiple br are squashed
    
    # Test with leading/trailing whitespace
    dom = html.fromstring("<p>  Hello World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with whitespace around tags
    dom = html.fromstring("<p>Hello <b> World </b></p>")
    assert extract_text(dom) == "Hello World"  # whitespace is handled correctly
    
    # Test complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>First paragraph with <b>bold</b> text</p>
            <p>Second paragraph<br>with line break</p>
        </div>
    """)
    expected = "Title\nFirst paragraph with bold text\nSecond paragraph\nwith line break"
    assert extract_text(dom) == expected
    
    # Test with a tag that is not in INLINE_TAGS (block element)
    dom = html.fromstring("<article>Content</article>")
    assert extract_text(dom) == "Content"
    
    # Test with a tag that is in INLINE_TAGS
    dom = html.fromstring("<span>Inline</span>")
    assert extract_text(dom) == "Inline"
    
    # Test with nested inline tags
    dom = html.fromstring("<p><b><i>Bold and italic</i></b></p>")
    assert extract_text(dom) == "Bold and italic"


# LLM-generated content at query #31
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with separator element (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with block element (div)
    dom = html.fromstring("<div>Content</div>")
    result = extract_text_array(dom)
    assert result == ["Content"]
    
    # Test nested elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert None in result  # Should have artificial newlines
    
    # Test with text and tail
    dom = html.fromstring("<p>Text <b>bold</b> after</p>")
    result = extract_text_array(dom)
    assert "Text " in result
    assert "bold" in result
    assert " after" in result
    
    # Test squash_artifical_nl parameter
    dom = html.fromstring("<div><p>One</p><p>Two</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    # Should have multiple None values
    none_count = sum(1 for x in result if x is None)
    assert none_count >= 2
    
    # Test strip_artifical_nl parameter
    dom = html.fromstring("<div><p>Content</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result[0] == "Content"  # No leading None
    assert result[-1] == "Content"  # No trailing None
    
    # Test with empty element
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test with only whitespace text
    dom = html.fromstring("<div>   </div>")
    result = extract_text_array(dom)
    assert result == ["   "]
    
    # Test callable tag (should return empty string)
    dom = html.fromstring("<div>Test</div>")
    dom.tag = lambda: None
    result = extract_text_array(dom)
    assert result == ''


# LLM-generated content at query #32
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    class MockDom:
        tag = 'span'
        text = 'Hello World'
        def getchildren(self):
            return []
    assert extract_text_array(MockDom()) == ['Hello World']

    # Test with a block element (should add None for artificial newlines)
    class MockDomBlock:
        tag = 'div'
        text = 'Text'
        def getchildren(self):
            return []
    result = extract_text_array(MockDomBlock())
    assert result == [None, 'Text', None]

    # Test with separator element (br)
    class MockDomBr:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(MockDomBr()) == [True]

    # Test with nested elements
    class MockChild:
        tag = 'span'
        text = 'inner'
        tail = ' tail'
        def getchildren(self):
            return []
    
    class MockDomNested:
        tag = 'div'
        text = 'outer'
        def getchildren(self):
            return [MockChild()]
    
    result = extract_text_array(MockDomNested())
    assert None in result
    assert 'outer' in result
    assert 'inner' in result
    assert ' tail' in result

    # Test with squash_artifical_nl=False
    class MockDomSquash:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    result = extract_text_array(MockDomSquash(), squash_artifical_nl=False)
    assert result == [None, None]

    # Test with strip_artifical_nl=False
    result = extract_text_array(MockDomBlock(), strip_artifical_nl=False)
    assert result[0] is None
    assert result[-1] is None

    # Test callable tag returns empty string
    class MockDomCallable:
        tag = lambda: 'test'
        def getchildren(self):
            return []
    assert extract_text_array(MockDomCallable()) == ''

    # Test with multiple children and tails
    class MockChild1:
        tag = 'b'
        text = 'bold'
        tail = ' normal'
        def getchildren(self):
            return []
    
    class MockChild2:
        tag = 'br'
        text = None
        tail = ' after_br'
        def getchildren(self):
            return []
    
    class MockDomMultiple:
        tag = 'p'
        text = 'start '
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    
    result = extract_text_array(MockDomMultiple())
    assert 'start ' in result
    assert 'bold' in result
    assert ' normal' in result
    assert True in result  # br separator
    assert ' after_br' in result
    assert result[0] is None  # starting artificial newline
    assert result[-1] is None  # ending artificial newline


# LLM-generated content at query #33
#--------------------------

```python
def test_extract_text_array():
    # Test simple text node
    from html.parser import HTMLParser
    import xml.etree.ElementTree as ET
    
    # Test with a simple paragraph
    dom = ET.fromstring("<p>Hello world</p>")
    result = extract_text_array(dom)
    assert result == ['Hello world'], f"Expected ['Hello world'], got {result}"
    
    # Test with inline tags
    dom = ET.fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert result == ['Hello ', 'bold', ' world'], f"Expected ['Hello ', 'bold', ' world'], got {result}"
    
    # Test with separator tag (br)
    dom = ET.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert result == ['Line1', True, 'Line2'], f"Expected ['Line1', True, 'Line2'], got {result}"
    
    # Test with block-level tag (div)
    dom = ET.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    result = extract_text_array(dom)
    assert result == [None, 'Para1', None, None, 'Para2', None], f"Expected [None, 'Para1', None, None, 'Para2', None], got {result}"
    
    # Test with nested inline tags
    dom = ET.fromstring("<p>Text <span>span <b>bold</b> end</span> tail</p>")
    result = extract_text_array(dom)
    assert result == ['Text ', 'span ', 'bold', ' end', ' tail'], f"Expected ['Text ', 'span ', 'bold', ' end', ' tail'], got {result}"
    
    # Test with script tag (should return empty)
    dom = ET.fromstring("<script>alert('test')</script>")
    result = extract_text_array(dom)
    assert result == ['alert(\'test\')'], f"Expected ['alert(\\'test\\')'], got {result}"
    
    # Test with squash_artifical_nl=False
    dom = ET.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'Para1', None, None, 'Para2', None], f"Expected [None, 'Para1', None, None, 'Para2', None], got {result}"
    
    # Test with strip_artifical_nl=False
    dom = ET.fromstring("<div><p>Para1</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'Para1', None, None], f"Expected [None, 'Para1', None, None], got {result}"
    
    # Test with empty element
    dom = ET.fromstring("<p></p>")
    result = extract_text_array(dom)
    assert result == [], f"Expected [], got {result}"
    
    # Test with text before and after child
    dom = ET.fromstring("<p>Before <b>bold</b> after</p>")
    result = extract_text_array(dom)
    assert result == ['Before ', 'bold', ' after'], f"Expected ['Before ', 'bold', ' after'], got {result}"
    
    # Test with multiple separators
    dom = ET.fromstring("<p>Line1<br/>Line2<br/>Line3</p>")
    result = extract_text_array(dom)
    assert result == ['Line1', True, 'Line2', True, 'Line3'], f"Expected ['Line1', True, 'Line2', True, 'Line3'], got {result}"


# LLM-generated content at query #34
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    from xml.etree import ElementTree as ET
    html = "<p>Hello world</p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Hello world"
    
    # Test with nested inline tags
    html = "<p>Hello <b>bold</b> world</p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Hello bold world"
    
    # Test with block elements
    html = "<div><p>First</p><p>Second</p></div>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator tags (br)
    html = "<p>Line1<br/>Line2</p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with multiple whitespace
    html = "<p>Hello    world</p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Hello world"
    
    # Test with whitespace around tags
    html = "<p>  Hello  <b>  bold  </b>  world  </p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Hello bold world"
    
    # Test with empty content
    html = "<p></p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    html = "<p>   </p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == ""
    
    # Test with custom block symbol
    html = "<div><p>First</p><p>Second</p></div>"
    dom = ET.fromstring(html)
    assert extract_text(dom, block_symbol=" | ") == "First | Second"
    
    # Test with custom separator symbol
    html = "<p>Line1<br/>Line2</p>"
    dom = ET.fromstring(html)
    assert extract_text(dom, sep_symbol=" | ") == "Line1 | Line2"
    
    # Test with squash_space=False
    html = "<p>Hello world</p>"
    dom = ET.fromstring(html)
    result = extract_text(dom, squash_space=False)
    assert " " in result  # Should preserve original spacing
    
    # Test with complex nested structure
    html = """
    <div>
        <h1>Title</h1>
        <p>Paragraph with <b>bold</b> and <i>italic</i></p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </div>
    """
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with bold and italic" in result
    assert "Item 1" in result
    assert "Item 2" in result
    
    # Test with non-inline tags that are not block elements
    html = "<span>inline</span>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "inline"
    
    # Test with multiple consecutive block elements
    html = "<div><p>A</p><p>B</p><p>C</p></div>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "A\nB\nC"
    
    # Test with trailing whitespace
    html = "<p>Hello   </p>"
    dom = ET.fromstring(html)
    assert extract_text(dom) == "Hello"
```


# LLM-generated content at query #35
#--------------------------

```python
def test_extract_text_array():
    # Test with inline tag (span)
    dom_span = type('obj', (object,), {'tag': 'span', 'text': 'Hello', 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom_span)
    assert result == ['Hello'], f"Expected ['Hello'], got {result}"

    # Test with separator tag (br)
    dom_br = type('obj', (object,), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom_br)
    assert result == [True], f"Expected [True], got {result}"

    # Test with non-inline, non-separator tag (div)
    dom_div = type('obj', (object,), {'tag': 'div', 'text': 'Text', 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom_div)
    assert result == ['Text'], f"Expected ['Text'], got {result}"

    # Test with nested structure
    child = type('obj', (object,), {'tag': 'span', 'text': 'World', 'tail': '!', 'getchildren': lambda: []})()
    dom_parent = type('obj', (object,), {'tag': 'div', 'text': 'Hello ', 'tail': None, 'getchildren': lambda: [child]})()
    result = extract_text_array(dom_parent)
    assert result == ['Hello ', 'World', '!'], f"Expected ['Hello ', 'World', '!'], got {result}"

    # Test squash_artifical_nl=False
    dom_div2 = type('obj', (object,), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom_div2, squash_artifical_nl=False)
    assert result == [None, None], f"Expected [None, None], got {result}"

    # Test strip_artifical_nl=False
    result = extract_text_array(dom_div2, strip_artifical_nl=False)
    assert result == [None], f"Expected [None], got {result}"

    # Test with callable tag (should return empty string)
    dom_callable = type('obj', (object,), {'tag': lambda: None, 'text': None, 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom_callable)
    assert result == '', f"Expected '', got {result}"


# LLM-generated content at query #36
#--------------------------

```python
def test_extract_text():
    # Test with simple text
    dom = MockTag('p', text='Hello World')
    assert extract_text(dom) == 'Hello World'
    
    # Test with separator tag
    dom = MockTag('div', children=[
        MockTag('p', text='Line 1'),
        MockTag('br'),
        MockTag('p', text='Line 2')
    ])
    assert extract_text(dom) == 'Line 1\nLine 2'
    
    # Test with block tags
    dom = MockTag('div', children=[
        MockTag('p', text='Paragraph 1'),
        MockTag('p', text='Paragraph 2')
    ])
    assert extract_text(dom) == 'Paragraph 1\nParagraph 2'
    
    # Test with inline tags
    dom = MockTag('p', children=[
        MockTag('strong', text='bold'),
        MockTag('em', text='italic')
    ])
    assert extract_text(dom) == 'bolditalic'
    
    # Test with nested structure
    dom = MockTag('div', children=[
        MockTag('p', text='First', children=[
            MockTag('br'),
            MockTag('span', text='Second')
        ])
    ])
    assert extract_text(dom) == 'First\nSecond'
    
    # Test with whitespace in text
    dom = MockTag('p', text='  Hello   World  ')
    assert extract_text(dom, squash_space=True) == 'Hello World'
    
    # Test with custom symbols
    dom = MockTag('div', children=[
        MockTag('p', text='A'),
        MockTag('p', text='B')
    ])
    assert extract_text(dom, block_symbol=' | ') == 'A | B'
    
    # Test with separator symbol
    dom = MockTag('div', children=[
        MockTag('p', text='A'),
        MockTag('br'),
        MockTag('p', text='B')
    ])
    assert extract_text(dom, sep_symbol=' - ') == 'A - B'
    
    # Test with empty content
    dom = MockTag('div')
    assert extract_text(dom) == ''
    
    # Test with nested separators
    dom = MockTag('div', children=[
        MockTag('br'),
        MockTag('p', text='Text')
    ])
    assert extract_text(dom) == 'Text'
    
    # Test with trailing/leading whitespace
    dom = MockTag('p', text='  Hello  ')
    assert extract_text(dom, squash_space=True) == 'Hello'


# LLM-generated content at query #37
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction from a simple element
    from xml.etree.ElementTree import Element, SubElement, fromstring
    
    # Test 1: Simple paragraph
    p = fromstring("<p>Hello world</p>")
    assert extract_text(p) == "Hello world"
    
    # Test 2: Inline elements nested in block elements
    div = fromstring("<div><p>Hello <b>bold</b> world</p></div>")
    assert extract_text(div) == "Hello bold world"
    
    # Test 3: Multiple block elements with separators
    div = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(div) == "First\nSecond"
    
    # Test 4: Separator elements (br)
    div = fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(div) == "Line1\nLine2"
    
    # Test 5: Whitespace handling
    div = fromstring("<p>  Hello   world  </p>")
    assert extract_text(div) == "Hello world"
    
    # Test 6: Nested inline and block elements
    div = fromstring("<div><p>Text with <span>span</span> inside</p></div>")
    assert extract_text(div) == "Text with span inside"
    
    # Test 7: Empty elements
    div = fromstring("<div></div>")
    assert extract_text(div) == ""
    
    # Test 8: Multiple levels of nesting
    div = fromstring("<div><section><h1>Title</h1><p>Content</p></section></div>")
    assert extract_text(div) == "Title\nContent"
    
    # Test 9: Custom block_symbol
    p = fromstring("<p>Hello</p><p>World</p>")
    div = fromstring("<div>")
    div.append(p)
    div.append(p)
    assert extract_text(div, block_symbol=" | ") == "Hello | World"
    
    # Test 10: Custom sep_symbol
    p = fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(p, sep_symbol=" ") == "Line1 Line2"
    
    # Test 11: squash_space=False
    p = fromstring("<p>Hello   world</p>")
    assert extract_text(p, squash_space=False) == "Hello   world"
    
    # Test 12: Complex nested structure
    html = fromstring("""
        <div>
            <h1>Title</h1>
            <p>First paragraph with <a href='#'>link</a> inside</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(html)
    assert "Title" in result
    assert "First paragraph with link inside" in result
    assert "Item 1" in result
    assert "Item 2" in result


# LLM-generated content at query #38
#--------------------------

```python
def test_extract_text():
    # Test simple inline element
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<span>Hello</span>")
    assert extract_text(dom) == "Hello"
    
    # Test block element
    dom = fragment_fromstring("<div>Hello</div>")
    assert extract_text(dom) == "Hello"
    
    # Test separator element
    dom = fragment_fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test nested elements
    dom = fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with inline tags within block
    dom = fragment_fromstring("<p>Hello <b>world</b>!</p>")
    assert extract_text(dom) == "Hello world!"
    
    # Test whitespace squashing
    dom = fragment_fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test multiple separators
    dom = fragment_fromstring("<p>A<br><br>B</p>")
    assert extract_text(dom) == "A\n\nB"
    
    # Test empty element
    dom = fragment_fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test text only
    dom = fragment_fromstring("Plain text")
    assert extract_text(dom) == "Plain text"
    
    # Test complex nesting
    dom = fragment_fromstring(
        "<div><h1>Title</h1><p>Paragraph with <a href='#'>link</a></p></div>"
    )
    assert extract_text(dom) == "Title\nParagraph with link"
    
    # Test custom symbols
    dom = fragment_fromstring("<div><p>A</p><br><p>B</p></div>")
    assert extract_text(dom, block_symbol=' | ', sep_symbol=' - ') == "A |  -  B"
    
    # Test without squashing
    dom = fragment_fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "


# LLM-generated content at query #39
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml.html import fromstring
    dom = fromstring('<span>Hello World</span>')
    assert extract_text_array(dom) == ['Hello World']
    
    # Test with block element
    dom = fromstring('<div>Text</div>')
    result = extract_text_array(dom)
    assert result == ['Text']  # None values at start/end are stripped
    
    # Test with separator element
    dom = fromstring('<br/>')
    result = extract_text_array(dom)
    assert result == [True]  # True represents separator
    
    # Test with nested elements
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    result = extract_text_array(dom)
    assert result == ['First', 'Second']
    
    # Test with mixed inline and block elements
    dom = fromstring('<div>Hello <span>World</span>!</div>')
    result = extract_text_array(dom)
    assert result == ['Hello ', 'World', '!']
    
    # Test with nested br separator
    dom = fromstring('<div>Line1<br/>Line2</div>')
    result = extract_text_array(dom)
    assert result == ['Line1', True, 'Line2']
    
    # Test with whitespace text
    dom = fromstring('<p>  Hello   World  </p>')
    result = extract_text_array(dom)
    assert result == ['  Hello   World  ']
    
    # Test with empty element
    dom = fromstring('<div></div>')
    result = extract_text_array(dom)
    assert result == []
    
    # Test with callable tag (special case that returns empty string)
    class CallableTag:
        def __call__(self):
            pass
        tag = 'script'
    assert extract_text_array(CallableTag()) == ''
    
    # Test with squash_artifical_nl=False and strip_artifical_nl=False
    dom = fromstring('<div><p>Text</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result[0] is None  # Opening None for div
    assert result[1] is None  # Opening None for p
    assert result[2] == 'Text'
    assert result[3] is None  # Closing None for p
    assert result[4] is None  # Closing None for div
    
    # Test with only strip_artifical_nl=False
    dom = fromstring('<div><p>Text</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result[0] is None  # Opening None for div
    assert result[1] is None  # Opening None for p
    assert result[2] == 'Text'
    assert result[3] is None  # Closing None for p
    assert result[4] is None  # Closing None for div
    
    # Test with only squash_artifical_nl=False
    dom = fromstring('<div><p>Text</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result[0] is None  # Opening None for div
    assert result[1] is None  # Opening None for p
    assert result[2] == 'Text'
    assert len(result) == 3  # strip removed closing Nones
    
    # Test with complex nested structure
    html = '<div><h1>Title</h1><p>Paragraph with <strong>bold</strong> text</p></div>'
    dom = fromstring(html)
    result = extract_text_array(dom)
    assert result == ['Title', 'Paragraph with ', 'bold', ' text']
    
    # Test with multiple br separators
    dom = fromstring('<div>Line1<br/><br/>Line2</div>')
    result = extract_text_array(dom)
    assert result == ['Line1', True, True, 'Line2']
    
    # Test with tail text
    dom = fromstring('<div>Start<img src="test.png"/>End</div>')
    result = extract_text_array(dom)
    assert result == ['Start', 'End']


# LLM-generated content at query #40
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b>!</p>")
    assert extract_text(doc) == "Hello world!"
    
    # Test with block elements creating newlines
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with separator elements (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with nested elements
    doc = html.fromstring("<div><p>Paragraph</p><span>Span</span></div>")
    assert extract_text(doc) == "Paragraph\nSpan"
    
    # Test whitespace squashing
    doc = html.fromstring("<p>Hello    world</p>")
    assert extract_text(doc) == "Hello world"
    
    # Test strip leading/trailing whitespace
    doc = html.fromstring("<p>  Hello world  </p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with custom block_symbol
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc, sep_symbol=' ') == "Line1 Line2"
    
    # Test squash_space=False
    doc = html.fromstring("<p>  Hello  world  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello  world  "
    
    # Test empty document
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test deeply nested structure
    doc = html.fromstring("<div><p><b>Deep</b> <i>nesting</i></p></div>")
    assert extract_text(doc) == "Deep nesting"
    
    # Test multiple separators
    doc = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(doc) == "Line1\n\nLine2"
```


# LLM-generated content at query #41
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    dom = type('Mock', (), {'tag': 'span', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom)
    assert result == ['Hello'], f"Expected ['Hello'], got {result}"

    # Test with a separator element
    dom = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"

    # Test with a block element (not inline, not separator)
    dom = type('Mock', (), {'tag': 'div', 'text': 'Content', 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom)
    assert result == [None, 'Content', None], f"Expected [None, 'Content', None], got {result}"

    # Test with nested elements
    child = type('Mock', (), {'tag': 'span', 'text': 'World', 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': 'Hello ', 'tail': None, 'getchildren': lambda self: [child]})()
    result = extract_text_array(dom)
    assert result == [None, 'Hello ', 'World', None], f"Expected [None, 'Hello ', 'World', None], got {result}"

    # Test with callable tag (edge case)
    dom = type('Mock', (), {'tag': lambda: None, 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom)
    assert result == [], f"Expected [], got {result}"

    # Test with multiple children and tails
    child1 = type('Mock', (), {'tag': 'b', 'text': 'bold', 'tail': ' tail1', 'getchildren': lambda self: []})()
    child2 = type('Mock', (), {'tag': 'i', 'text': 'italic', 'tail': ' tail2', 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': 'Start ', 'tail': None, 'getchildren': lambda self: [child1, child2]})()
    result = extract_text_array(dom)
    assert result == [None, 'Start ', 'bold', ' tail1', 'italic', ' tail2', None], f"Expected [None, 'Start ', 'bold', ' tail1', 'italic', ' tail2', None], got {result}"

    # Test squash_artifical_nl=False
    dom = type('Mock', (), {'tag': 'div', 'text': 'Test', 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'Test', None], f"Expected [None, 'Test', None], got {result}"

    # Test strip_artifical_nl=False
    dom = type('Mock', (), {'tag': 'div', 'text': 'Test', 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'Test', None], f"Expected [None, 'Test', None], got {result}"

    # Test with separator element and text
    dom = type('Mock', (), {'tag': 'br', 'text': '\n', 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"

    # Test empty element
    dom = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom)
    assert result == [None, None], f"Expected [None, None], got {result}"


# LLM-generated content at query #42
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text_array(dom) == ["Hello World"]

    # Test with block element
    dom = html.fromstring("<div>Hello World</div>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]

    # Test with separator (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]

    # Test with nested elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"]

    # Test with inline inside block
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]

    # Test with tail text
    dom = html.fromstring("<p>Hello<br/>World</p>")
    result = extract_text_array(dom)
    assert result == ["Hello", True, "World"]

    # Test with None text
    dom = html.fromstring("<div><span></span></div>")
    result = extract_text_array(dom)
    assert result == []

    # Test squash_artifical_nl=False
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, "Text", None]

    # Test strip_artifical_nl=False
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == ["Text", None]

    # Test with callable tag (should return empty string)
    class MockDom:
        tag = lambda: None
    assert extract_text_array(MockDom()) == ""

    # Test complex nested structure
    dom = html.fromstring("<div><h1>Title</h1><p>Paragraph with <b>bold</b></p></div>")
    result = extract_text_array(dom)
    assert result == ["Title", "Paragraph with ", "bold"]


# LLM-generated content at query #43
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import etree
    dom = etree.fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]

    # Test with block tag
    dom = etree.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom)
    assert result == ["Text"]

    # Test with separator tag
    dom = etree.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]

    # Test with mixed inline and block tags
    dom = etree.fromstring("<div><span>Hello</span><p>World</p></div>")
    result = extract_text_array(dom)
    assert result == ["Hello", "World"]

    # Test with nested tags
    dom = etree.fromstring("<div><p><span>Nested</span> text</p></div>")
    result = extract_text_array(dom)
    assert result == ["Nested text"]

    # Test with text and tail
    dom = etree.fromstring("<div>Start<span>middle</span>End</div>")
    result = extract_text_array(dom)
    assert result == ["Start", "middle", "End"]

    # Test artificial newlines are squashed
    dom = etree.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == ["Para1", "Para2"]

    # Test artificial newlines are stripped
    dom = etree.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["Text"]

    # Test with empty element
    dom = etree.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []

    # Test with callable tag (should return empty string)
    class FakeTag:
        tag = lambda: None
    fake_dom = type('FakeDom', (), {'tag': lambda: None})()
    result = extract_text_array(fake_dom)
    assert result == ''

    # Test with multiple separators
    dom = etree.fromstring("<div><br/><br/></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [True, True]

    # Test with text containing whitespace
    dom = etree.fromstring("<div>  Hello   World  </div>")
    result = extract_text_array(dom)
    assert result == ["  Hello   World  "]


# LLM-generated content at query #44
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text node
    from lxml.html import fromstring
    dom = fromstring("<p>Hello World</p>")
    assert extract_text_array(dom) == ['Hello World']
    
    # Test with inline tags
    dom = fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert result == ['Hello ', 'bold', ' world']
    
    # Test with separator tag (br)
    dom = fromstring("<p>Line1<br>Line2</p>")
    result = extract_text_array(dom)
    assert True in result  # Contains separator marker
    
    # Test with block elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert None in result  # Contains artificial newlines
    
    # Test with nested structure
    dom = fromstring("<div><p>Text <span>inside</span> more</p></div>")
    result = extract_text_array(dom)
    assert 'Text ' in result
    assert 'inside' in result
    assert ' more' in result
    
    # Test empty element
    dom = fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test with tail text
    dom = fromstring("<p>Before<img src='test.jpg'>After</p>")
    result = extract_text_array(dom)
    assert 'Before' in result
    assert 'After' in result
    
    # Test squash_artifical_nl=False
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result.count(None) > 1  # Multiple None markers
    
    # Test strip_artifical_nl=False
    dom = fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None  # Starts or ends with None
    
    # Test with callable tag (should return empty string)
    class MockElement:
        tag = lambda: None
    assert extract_text_array(MockElement()) == ''


# LLM-generated content at query #45
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello World", None], f"Expected [None, 'Hello World', None], got {result}"
    
    # Test with inline element
    dom = etree.fromstring("<span>Inline text</span>")
    result = extract_text_array(dom)
    assert result == ["Inline text"], f"Expected ['Inline text'], got {result}"
    
    # Test with separator element (br)
    dom = etree.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"
    
    # Test with nested elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"], f"Expected ['First', 'Second'], got {result}"
    
    # Test with mixed inline and block elements
    dom = etree.fromstring("<div><span>inline</span><p>block</p></div>")
    result = extract_text_array(dom)
    assert None in result, f"Expected None in result, got {result}"
    assert "inline" in result, f"Expected 'inline' in result, got {result}"
    assert "block" in result, f"Expected 'block' in result, got {result}"
    
    # Test with element containing text and tail text
    dom = etree.fromstring("<p>Hello <b>world</b> again</p>")
    result = extract_text_array(dom)
    assert "Hello " in result, f"Expected 'Hello ' in result, got {result}"
    assert "world" in result, f"Expected 'world' in result, got {result}"
    assert " again" in result, f"Expected ' again' in result, got {result}"
    
    # Test with squash_artifical_nl=False
    dom = etree.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result[0] is None, f"Expected first element to be None, got {result[0]}"
    assert result[-1] is None, f"Expected last element to be None, got {result[-1]}"
    
    # Test with strip_artifical_nl=False
    dom = etree.fromstring("<p>Text</p>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert None in result, f"Expected None in result, got {result}"
    
    # Test with callable tag (should return empty string)
    from lxml import etree as etree2
    mock_dom = type('Mock', (), {'tag': lambda: None})()
    result = extract_text_array(mock_dom)
    assert result == '', f"Expected empty string, got {result}"
    
    # Test with empty element
    dom = etree.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [], f"Expected empty list, got {result}"


# LLM-generated content at query #46
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    from lxml.html import fromstring
    dom = fromstring("<span>Hello</span>")
    assert extract_text(dom) == "Hello"
    
    # Test with block tag
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator tag (br)
    dom = fromstring("<span>Line1<br/>Line2</span>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested inline and block elements
    dom = fromstring("<div><p>Hello <b>World</b></p></div>")
    assert extract_text(dom) == "Hello World"
    
    # Test with multiple block elements
    dom = fromstring("<div><p>Para1</p><p>Para2</p><p>Para3</p></div>")
    assert extract_text(dom) == "Para1\nPara2\nPara3"
    
    # Test with whitespace handling
    dom = fromstring("<div><p>  Hello   World  </p></div>")
    assert extract_text(dom) == "Hello World"
    
    # Test with custom block and separator symbols
    dom = fromstring("<div><p>First</p><br/><p>Second</p></div>")
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == "First|Second"
    
    # Test with squash_space=False
    dom = fromstring("<div><p>  Hello   World  </p></div>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test with empty content
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with nested inline elements
    dom = fromstring("<p><strong>Bold</strong> and <em>italic</em></p>")
    assert extract_text(dom) == "Bold and italic"
    
    # Test with script tag (should be empty)
    dom = fromstring("<div><script>alert('test')</script>Content</div>")
    assert extract_text(dom) == "Content"
    
    # Test with multiple br tags
    dom = fromstring("<div>Line1<br/><br/>Line2</div>")
    assert extract_text(dom) == "Line1\n\nLine2"
    
    # Test with mixed content
    dom = fromstring("<div>Start <p>Middle</p> End</div>")
    assert extract_text(dom) == "Start\nMiddle\nEnd"


# LLM-generated content at query #47
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello world</p>")
    assert extract_text_array(dom) == [None, "Hello world", None]
    
    # Test with inline element
    dom = etree.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with separator element
    dom = etree.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with nested elements
    dom = etree.fromstring("<p><b>Bold</b> text</p>")
    result = extract_text_array(dom)
    assert None in result
    assert "Bold" in result
    assert " text" in result
    
    # Test with multiple children
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result.count(None) >= 2  # Should have artificial newlines
    
    # Test with text and tail
    dom = etree.fromstring("<p>Start <b>bold</b> end</p>")
    result = extract_text_array(dom)
    assert "Start " in result
    assert "bold" in result
    assert " end" in result
    
    # Test empty element
    dom = etree.fromstring("<div></div>")
    assert extract_text_array(dom) == [None, None]
    
    # Test element with only text
    dom = etree.fromstring("<p>Only text</p>")
    assert extract_text_array(dom) == [None, "Only text", None]
    
    # Test nested inline elements
    dom = etree.fromstring("<p><span><b>Nested</b></span> text</p>")
    result = extract_text_array(dom)
    assert "Nested" in result
    assert " text" in result
    
    # Test with squash_artifical_nl=False
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result.count(None) >= 4  # Multiple None values before squashing
    
    # Test with strip_artifical_nl=False
    dom = etree.fromstring("<p>Hello</p>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # Should keep leading None
    assert result[-1] is None  # Should keep trailing None
    
    # Test callable tag returns empty string
    dom = etree.fromstring("<div>Test</div>")
    dom.tag = lambda: None  # Mock callable tag
    assert extract_text_array(dom) == ''
    
    # Test with complex nested structure
    html = """
    <div>
        <h1>Title</h1>
        <p>Paragraph with <a href="#">link</a> and <br/> break</p>
    </div>
    """
    dom = etree.fromstring(html)
    result = extract_text_array(dom)
    assert "Title" in result
    assert "Paragraph with " in result
    assert "link" in result
    assert " and " in result
    assert True in result  # Should contain separator from <br/>


# LLM-generated content at query #48
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>hello world</span>")
    assert extract_text_array(dom) == ["hello world"]
    
    # Test with block element (should add None markers)
    dom = html.fromstring("<div>text</div>")
    result = extract_text_array(dom)
    assert result == ["text"]  # None markers stripped by default
    
    # Test with separator element (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]  # True represents separator
    
    # Test with nested inline elements
    dom = html.fromstring("<span>hello <b>world</b></span>")
    result = extract_text_array(dom)
    assert result == ["hello ", "world"]
    
    # Test with block elements creating artificial newlines
    dom = html.fromstring("<div><p>first</p><p>second</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "first", None, None, "second", None, None]
    
    # Test with br separator
    dom = html.fromstring("<div>line1<br/>line2</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "line1", True, "line2", None]
    
    # Test squash_artifical_nl
    dom = html.fromstring("<div><p>a</p><p>b</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, "a", None, "b", None]
    
    # Test strip_artifical_nl
    dom = html.fromstring("<p>text</p>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["text"]
    
    # Test with callable tag (should return empty string)
    dom = html.fromstring("<div><function></function></div>")
    # Mock a callable tag
    from types import FunctionType
    class FakeElement:
        tag = lambda x: None
    assert extract_text_array(FakeElement()) == ''
    
    # Test with text and tail
    dom = html.fromstring("<div>before<span>inside</span>after</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "before", "inside", "after", None]
    
    # Test empty content
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test with multiple separators
    dom = html.fromstring("<div><br/><br/></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, True, True, None]


# LLM-generated content at query #49
#--------------------------

```python
def test_extract_text():
    # Test simple text extraction
    dom = SimpleElement('p', text='Hello World')
    assert extract_text(dom) == 'Hello World'
    
    # Test with inline tags
    dom = SimpleElement('p', children=[
        SimpleElement('strong', text='bold'),
        SimpleElement('em', text='italic')
    ], text='before')
    assert extract_text(dom) == 'before bold italic'
    
    # Test with separator tags
    dom = SimpleElement('div', children=[
        SimpleElement('br'),
        SimpleElement('span', text='text')
    ])
    result = extract_text(dom)
    assert '\n' in result
    assert 'text' in result
    
    # Test with block elements
    dom = SimpleElement('div', children=[
        SimpleElement('h1', text='Title'),
        SimpleElement('p', text='Paragraph')
    ])
    result = extract_text(dom)
    assert result.startswith('Title')
    assert 'Paragraph' in result
    assert '\n' in result
    
    # Test with custom symbols
    dom = SimpleElement('div', children=[
        SimpleElement('h1', text='Title'),
        SimpleElement('p', text='Paragraph')
    ])
    result = extract_text(dom, block_symbol=' | ', sep_symbol=' - ')
    assert ' | ' in result
    assert ' - ' in result
    
    # Test squash_space=False
    dom = SimpleElement('p', text='  Hello   World  ')
    result = extract_text(dom, squash_space=False)
    assert result == '  Hello   World  '
    
    # Test empty element
    dom = SimpleElement('div')
    assert extract_text(dom) == ''
    
    # Test nested inline tags
    dom = SimpleElement('p', children=[
        SimpleElement('a', text='link', children=[
            SimpleElement('strong', text='bold link')
        ])
    ])
    assert extract_text(dom) == 'link bold link'
    
    # Test multiple separators
    dom = SimpleElement('div', children=[
        SimpleElement('br'),
        SimpleElement('br'),
        SimpleElement('span', text='text')
    ])
    result = extract_text(dom)
    assert result.count('\n') == 1  # consecutive br should be squashed
    
    # Test whitespace handling
    dom = SimpleElement('p', text='Hello\n\nWorld')
    result = extract_text(dom)
    assert ' ' in result  # newlines should be converted to spaces
    assert '\n' not in result  # no newlines in inline element

class SimpleElement:
    """Helper class to simulate DOM elements for testing"""
    def __init__(self, tag, text=None, children=None):
        self.tag = tag
        self.text = text
        self.children = children or []
        
    def getchildren(self):
        return self.children
```


# LLM-generated content at query #50
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello world</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello world", None]
    
    # Test with inline tag
    dom = etree.fromstring("<span>inline text</span>")
    result = extract_text_array(dom)
    assert result == ["inline text"]
    
    # Test with separator tag
    dom = etree.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested inline tags
    dom = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "bold", " world", None]
    
    # Test with nested block tags
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, "Second", None, None]
    
    # Test with tail text
    dom = etree.fromstring("<p>Hello <b>bold</b>tail text</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "bold", "tail text", None]
    
    # Test with squash_artifical_nl=False
    dom = etree.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "Text", None, None]
    
    # Test with strip_artifical_nl=False
    dom = etree.fromstring("<p>Text</p>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Text", None]
    
    # Test with callable tag (edge case)
    class FakeDom:
        tag = lambda: None
    result = extract_text_array(FakeDom())
    assert result == ''
    
    # Test with empty element
    dom = etree.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test with multiple separators
    dom = etree.fromstring("<br/><br/>")
    result = extract_text_array(dom)
    assert result == [True, True]


# LLM-generated content at query #51
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    class MockElement:
        tag = 'span'
        text = 'Hello'
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text_array(MockElement())
    assert result == ['Hello']
    
    # Test with separator tag (br)
    class MockBrElement:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text_array(MockBrElement())
    assert result == [True]  # True represents separator
    
    # Test with block-level tag
    class MockDivElement:
        tag = 'div'
        text = 'Content'
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text_array(MockDivElement())
    expected = [None, 'Content', None]  # None represents artificial newline
    assert result == expected
    
    # Test with nested elements
    class MockChildElement:
        tag = 'span'
        text = 'Child'
        tail = ' tail'
        def getchildren(self):
            return []
    
    class MockParentElement:
        tag = 'div'
        text = 'Parent '
        tail = None
        def getchildren(self):
            return [MockChildElement()]
    
    result = extract_text_array(MockParentElement())
    expected = [None, 'Parent ', 'Child', ' tail', None]
    assert result == expected
    
    # Test with squash_artifical_nl=False
    result = extract_text_array(MockDivElement(), squash_artifical_nl=False)
    expected = [None, 'Content', None]
    assert result == expected
    
    # Test with strip_artifical_nl=False
    result = extract_text_array(MockDivElement(), strip_artifical_nl=False)
    expected = [None, 'Content', None]
    assert result == expected
    
    # Test empty element
    class MockEmptyElement:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text_array(MockEmptyElement())
    assert result == []
    
    # Test multiple children
    class MockChild1:
        tag = 'span'
        text = 'First'
        tail = ' '
        def getchildren(self):
            return []
    
    class MockChild2:
        tag = 'span'
        text = 'Second'
        tail = None
        def getchildren(self):
            return []
    
    class MockMultiChildElement:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    
    result = extract_text_array(MockMultiChildElement())
    expected = [None, 'First', ' ', 'Second', None]
    assert result == expected


# LLM-generated content at query #52
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction from a simple div
    from lxml.html import fromstring
    dom = fromstring("<div>Hello World</div>")
    assert extract_text(dom) == "Hello World"
    
    # Test text extraction with inline tags
    dom = fromstring("<div>Hello <b>bold</b> World</div>")
    assert extract_text(dom) == "Hello bold World"
    
    # Test separator tags (br)
    dom = fromstring("<div>Line1<br>Line2</div>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test block-level tags create newlines
    dom = fromstring("<div><p>Paragraph 1</p><p>Paragraph 2</p></div>")
    assert extract_text(dom) == "Paragraph 1\nParagraph 2"
    
    # Test nested structures
    dom = fromstring("<div><p>Text with <b>bold</b> and <i>italic</i></p></div>")
    assert extract_text(dom) == "Text with bold and italic"
    
    # Test whitespace squashing
    dom = fromstring("<div>  too   much   space  </div>")
    assert extract_text(dom) == "too much space"
    
    # Test empty elements
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test element with only whitespace
    dom = fromstring("<div>   </div>")
    assert extract_text(dom) == ""
    
    # Test multiple br tags
    dom = fromstring("<div>Text<br><br>More text</div>")
    assert extract_text(dom) == "Text\n\nMore text"
    
    # Test nested separators
    dom = fromstring("<div>Text<br><span>span</span></div>")
    assert extract_text(dom) == "Text\nspan"
    
    # Test custom block_symbol
    dom = fromstring("<div><p>Para1</p><p>Para2</p></div>")
    assert extract_text(dom, block_symbol="<br>") == "Para1<br>Para2"
    
    # Test custom sep_symbol
    dom = fromstring("<div>Line1<br>Line2</div>")
    assert extract_text(dom, sep_symbol="<br>") == "Line1<br>Line2"
    
    # Test with squash_space=False
    dom = fromstring("<div><p>  Para1  </p><p>  Para2  </p></div>")
    result = extract_text(dom, squash_space=False)
    assert "  " in result  # Should preserve some whitespace
    
    # Test with script tag (should not extract content)
    dom = fromstring("<div>Text<script>alert('test');</script>More</div>")
    assert extract_text(dom) == "TextMore"
    
    # Test complex nested structure
    dom = fromstring("""
        <div>
            <h1>Title</h1>
            <p>First <b>paragraph</b> with <a href="#">link</a></p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    expected = "Title\nFirst paragraph with link\nItem 1\nItem 2"
    assert extract_text(dom) == expected
    
    # Test with img tag (inline but self-closing)
    dom = fromstring("<div>Text<img src='test.jpg'>More text</div>")
    assert extract_text(dom) == "TextMore text"
    
    # Test with nested block elements
    dom = fromstring("<div><div><p>Deeply</p></div><p>nested</p></div>")
    assert extract_text(dom) == "Deeply\nnested"
```


# LLM-generated content at query #53
#--------------------------

```python
def test_extract_text():
    # Test simple inline element
    from lxml import html
    dom = html.fromstring('<span>Hello World</span>')
    assert extract_text(dom) == 'Hello World'
    
    # Test block element adds newlines
    dom = html.fromstring('<div>First</div><div>Second</div>')
    result = extract_text(dom)
    assert result == 'First\nSecond'
    
    # Test separator element (br)
    dom = html.fromstring('Line1<br>Line2')
    assert extract_text(dom) == 'Line1\nLine2'
    
    # Test nested structure
    dom = html.fromstring('<div><p>Paragraph <b>bold</b> text</p></div>')
    assert extract_text(dom) == 'Paragraph bold text'
    
    # Test whitespace squashing
    dom = html.fromstring('<p>Multiple    spaces   here</p>')
    assert extract_text(dom) == 'Multiple spaces here'
    
    # Test with custom block_symbol
    dom = html.fromstring('<div>A</div><div>B</div>')
    assert extract_text(dom, block_symbol='|') == 'A|B'
    
    # Test with custom sep_symbol
    dom = html.fromstring('A<br>B')
    assert extract_text(dom, sep_symbol='|') == 'A|B'
    
    # Test with squash_space=False
    dom = html.fromstring('<p>  spaced  text  </p>')
    assert extract_text(dom, squash_space=False) == '  spaced  text  '
    
    # Test empty content
    dom = html.fromstring('<div></div>')
    assert extract_text(dom) == ''
    
    # Test complex nested structure
    dom = html.fromstring('''
        <div>
            <h1>Title</h1>
            <p>Paragraph with <a href="#">link</a> and <br> break</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    ''')
    expected = 'Title\nParagraph with link and break\nItem 1\nItem 2'
    assert extract_text(dom) == expected
    
    # Test callable tag returns empty string
    class FakeElement:
        tag = lambda x: None
    fake_dom = FakeElement()
    assert extract_text(fake_dom) == ''


# LLM-generated content at query #54
#--------------------------

```python
def test_extract_text():
    # Test basic paragraph with inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b>!</p>")
    assert extract_text(doc) == "Hello world!"
    
    # Test separator tag (br)
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test block elements create newlines
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test nested structures
    doc = html.fromstring("<div><span>Text</span><b>Bold</b></div>")
    assert extract_text(doc) == "TextBold"
    
    # Test whitespace handling
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test empty text
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with custom separator symbols
    doc = html.fromstring("<p>First</p><p>Second</p>")
    assert extract_text(doc, block_symbol="|") == "First|Second"
    
    # Test with sep_symbol
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc, sep_symbol=" ") == "Line1 Line2"
    
    # Test squash_space=False
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello   World  "
    
    # Test complex nested structure
    doc = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> and <i>italic</i></p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(doc)
    assert "Title" in result
    assert "Paragraph" in result
    assert "bold" in result
    assert "italic" in result
    assert "Item 1" in result
    assert "Item 2" in result
    assert result.count('\n') >= 3  # Multiple block elements create newlines
    
    # Test inline tags don't create newlines
    doc = html.fromstring("<span>a</span><span>b</span>")
    assert extract_text(doc) == "ab"
    
    # Test leading/trailing whitespace removal
    doc = html.fromstring("<p>  text  </p>")
    assert extract_text(doc) == "text"


# LLM-generated content at query #55
#--------------------------

```python
def test_extract_text_array():
    # Test with None text and no children
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [None, None], f"Expected [None, None], got {result}"

    # Test with inline tag
    class MockElement:
        tag = 'span'
        text = 'hello'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['hello'], f"Expected ['hello'], got {result}"

    # Test with separator tag (br)
    class MockElement:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"

    # Test with nested elements
    class MockChild:
        tag = 'span'
        text = 'world'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = 'hello '
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [None, 'hello ', 'world', None], f"Expected [None, 'hello ', 'world', None], got {result}"

    # Test with tail text
    class MockChild:
        tag = 'span'
        text = 'inner'
        tail = ' tail'
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = 'start '
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [None, 'start ', 'inner', ' tail', None], f"Expected [None, 'start ', 'inner', ' tail', None], got {result}"

    # Test squash_artifical_nl parameter
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None], f"Expected [None, None], got {result}"

    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == [None], f"Expected [None], got {result}"

    # Test strip_artifical_nl parameter
    class MockElement:
        tag = 'div'
        text = 'text'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'text', None], f"Expected [None, 'text', None], got {result}"

    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['text'], f"Expected ['text'], got {result}"

    # Test callable tag returns empty string
    class MockElement:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == '', f"Expected '', got {result}"


# LLM-generated content at query #56
#--------------------------

```python
def test_extract_text():
    # Test basic inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test separator elements (br)
    doc = html.fromstring("Line1<br>Line2")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test block elements create newlines
    doc = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test nested elements
    doc = html.fromstring("<div><p>Paragraph <b>bold</b></p></div>")
    assert extract_text(doc) == "Paragraph bold"
    
    # Test custom block_symbol and sep_symbol
    doc = html.fromstring("<div>A</div><div>B<br>C</div>")
    assert extract_text(doc, block_symbol=' | ', sep_symbol=' - ') == "A | B - C"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(doc, squash_space=False) == "Hello world"
    
    # Test empty element
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test nested with text and tail
    doc = html.fromstring("<p>Text <b>bold</b> tail</p>")
    assert extract_text(doc) == "Text bold tail"
    
    # Test complex structure
    doc = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>First paragraph</p>
            <p>Second <span>inline</span> paragraph</p>
        </div>
    """)
    result = extract_text(doc)
    assert "Title" in result
    assert "First paragraph" in result
    assert "Second inline paragraph" in result
    assert result.count('\n') == 3  # Title, First paragraph, Second paragraph
    
    # Test whitespace normalization
    doc = html.fromstring("<p>  Multiple   spaces   </p>")
    assert extract_text(doc) == "Multiple spaces"
    
    # Test strip behavior
    doc = html.fromstring("  <p>Content</p>  ")
    assert extract_text(doc) == "Content"
    
    # Test with only separator elements
    doc = html.fromstring("<br><br>")
    assert extract_text(doc) == "\n\n"
    
    # Test mixed inline and block
    doc = html.fromstring("<span>inline</span><div>block</div>")
    assert extract_text(doc) == "inline\nblock"


# LLM-generated content at query #57
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml.html import fromstring
    dom = fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with block element
    dom = fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == ["Hello"]
    
    # Test with separator element (br)
    dom = fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested inline elements
    dom = fromstring("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]
    
    # Test with nested block elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, "First", None, None, "Second", None]
    
    # Test with mix of inline and block
    dom = fromstring("<div>Text <span>inline</span> more</div>")
    result = extract_text_array(dom)
    assert result == ["Text ", "inline", " more"]
    
    # Test with separator inside block
    dom = fromstring("<div>Line1<br>Line2</div>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]
    
    # Test with callable tag (should return empty string)
    dom = fromstring("<div>Text</div>")
    dom.tag = lambda: None  # Mock callable tag
    result = extract_text_array(dom)
    assert result == []
    
    # Test with None text
    dom = fromstring("<div><span></span></div>")
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with tail text
    dom = fromstring("<div><b>Bold</b> tail</div>")
    result = extract_text_array(dom)
    assert result == ["Bold", " tail"]
    
    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, "A", None, None, "B", None]
    
    # Test with strip_artifical_nl=False
    dom = fromstring("<div>Content</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Content", None]
    
    # Test with both squash and strip disabled
    dom = fromstring("<div>Content</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Content", None]
    
    # Test with nested separators
    dom = fromstring("<div><br><br></div>")
    result = extract_text_array(dom)
    assert result == [True, True]
    
    # Test with empty content
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test with complex nesting
    dom = fromstring("<div><p>Para <b>bold</b> and <i>italic</i></p><br><span>span</span></div>")
    result = extract_text_array(dom)
    assert result == [None, "Para ", "bold", " and ", "italic", None, True, "span"]


# LLM-generated content at query #58
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    class MockDom:
        tag = 'span'
        text = 'Hello'
        def getchildren(self):
            return []
    assert extract_text_array(MockDom()) == ['Hello']

    # Test with separator tag
    class MockBr:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(MockBr()) == [True]

    # Test with block tag
    class MockDiv:
        tag = 'div'
        text = 'Text'
        def getchildren(self):
            return []
    result = extract_text_array(MockDiv())
    assert result[0] is None  # starts with artificial newline
    assert 'Text' in result
    assert result[-1] is None  # ends with artificial newline

    # Test with nested elements
    class MockChild:
        tag = 'span'
        text = 'child'
        tail = None
        def getchildren(self):
            return []
    class MockParent:
        tag = 'div'
        text = 'parent '
        def getchildren(self):
            return [MockChild()]
    result = extract_text_array(MockParent())
    assert result[0] is None
    assert 'parent ' in result
    assert 'child' in result
    assert result[-1] is None

    # Test with tail text
    class MockTailChild:
        tag = 'b'
        text = 'bold'
        tail = ' tail'
        def getchildren(self):
            return []
    class MockWithTail:
        tag = 'p'
        text = 'Start '
        def getchildren(self):
            return [MockTailChild()]
    result = extract_text_array(MockWithTail())
    assert 'Start ' in result
    assert 'bold' in result
    assert ' tail' in result

    # Test with callable tag (returns empty string)
    class MockCallable:
        tag = lambda: None
        text = 'test'
        def getchildren(self):
            return []
    assert extract_text_array(MockCallable()) == ''

    # Test with squash_artifical_nl=False and strip_artifical_nl=False
    class MockBlock:
        tag = 'section'
        text = 'content'
        def getchildren(self):
            return []
    result = extract_text_array(MockBlock(), squash_artifical_nl=False, strip_artifical_nl=False)
    assert result.count(None) == 2  # both artificial newlines present

    # Test with separator in between inline tags
    class MockInlineWithBr:
        tag = 'span'
        text = 'text1'
        def getchildren(self):
            return [MockBr()]
    result = extract_text_array(MockInlineWithBr())
    assert 'text1' in result
    assert True in result  # separator present
    
    # Test empty element
    class MockEmpty:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    result = extract_text_array(MockEmpty())
    assert result[0] is None
    assert result[-1] is None
    assert len(result) == 2  # just the artificial newlines

    # Test with child having no text
    class MockNoTextChild:
        tag = 'span'
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockParentNoText:
        tag = 'div'
        text = 'parent'
        def getchildren(self):
            return [MockNoTextChild()]
    result = extract_text_array(MockParentNoText())
    assert 'parent' in result
    assert result.count(None) == 0  # no artificial newlines for inline child
    
    # Test multiple children
    class MockChild1:
        tag = 'i'
        text = 'italic'
        tail = ' '
        def getchildren(self):
            return []
    class MockChild2:
        tag = 'strong'
        text = 'bold'
        tail = None
        def getchildren(self):
            return []
    class MockMultiChild:
        tag = 'p'
        text = None
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    result = extract_text_array(MockMultiChild())
    assert 'italic' in result
    assert ' ' in result
    assert 'bold' in result
    assert result[0] is None  # starts with artificial newline for block tag
    assert result[-1] is None  # ends with artificial newline for block tag


# LLM-generated content at query #59
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml.html import fromstring
    dom = fromstring("<span>Hello World</span>")
    assert extract_text_array(dom) == ["Hello World"]
    
    # Test with block tag
    dom = fromstring("<div>Hello World</div>")
    result = extract_text_array(dom)
    assert result[0] is None  # artificial newline before content
    assert "Hello World" in result
    assert result[-1] is None  # artificial newline after content
    
    # Test with separator tag (br)
    dom = fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [True]  # separator marker
    
    # Test with nested tags
    dom = fromstring("<div><span>Hello</span><span>World</span></div>")
    result = extract_text_array(dom)
    assert None in result  # block tag markers
    assert "Hello" in result
    assert "World" in result
    
    # Test with text and tail
    dom = fromstring("<p>Hello <b>bold</b> text</p>")
    result = extract_text_array(dom)
    assert "Hello " in result
    assert "bold" in result
    assert " text" in result
    
    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert len([x for x in result if x is None]) >= 4  # multiple None markers
    
    # Test with strip_artifical_nl=False
    dom = fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # leading artificial newline preserved
    assert result[-1] is None  # trailing artificial newline preserved
    
    # Test with inline tag inside block tag
    dom = fromstring("<div><a href='#'>Link</a></div>")
    result = extract_text_array(dom)
    assert "Link" in result
    
    # Test with multiple nested tags
    dom = fromstring("<div><ul><li>Item 1</li><li>Item 2</li></ul></div>")
    result = extract_text_array(dom)
    assert "Item 1" in result
    assert "Item 2" in result
    
    # Test empty element
    dom = fromstring("<div></div>")
    assert extract_text_array(dom) == [None, None]  # empty with artificial newlines
    
    # Test with callable tag (should return empty string)
    class CallableTag:
        def __call__(self):
            pass
    dom = fromstring("<div>Test</div>")
    dom.tag = CallableTag()
    assert extract_text_array(dom) == ""


# LLM-generated content at query #60
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text_array(dom) == ["Hello World"]
    
    # Test with block element (should add artificial newlines)
    dom = html.fromstring("<div>Hello</div>")
    assert extract_text_array(dom) == [None, "Hello", None]
    
    # Test with separator element (br)
    dom = html.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with nested elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, "Second", None, None]
    
    # Test with inline elements inside block
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "World", None]
    
    # Test with text and tail
    dom = html.fromstring("<div>Start <span>Middle</span> End</div>")
    result = extract_text_array(dom)
    assert result == [None, "Start ", "Middle", " End", None]
    
    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "A", None, None, None, "B", None, None]
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Hello", None]
    
    # Test with callable tag (should return empty string)
    class MockElement:
        def __init__(self):
            self.tag = lambda: None
    mock_dom = MockElement()
    assert extract_text_array(mock_dom) == ""
    
    # Test with multiple separators
    dom = html.fromstring("<div>Line1<br/>Line2<br/>Line3</div>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", True, "Line3", None]
    
    # Test with empty element
    dom = html.fromstring("<div></div>")
    assert extract_text_array(dom) == [None, None]


# LLM-generated content at query #61
#--------------------------

```python
def test_extract_text():
    # Test with simple text
    dom = type('Node', (), {'tag': 'p', 'text': 'Hello', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(dom) == 'Hello'
    
    # Test with inline tags
    span = type('Node', (), {'tag': 'span', 'text': 'World', 'tail': None, 'getchildren': lambda: []})()
    p = type('Node', (), {'tag': 'p', 'text': 'Hello ', 'tail': None, 'getchildren': lambda: [span]})()
    assert extract_text(p) == 'Hello World'
    
    # Test with separator tag (br)
    br = type('Node', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    p = type('Node', (), {'tag': 'p', 'text': 'Line1', 'tail': None, 'getchildren': lambda: [br]})()
    assert extract_text(p) == 'Line1\nLine1'
    
    # Test with block-level tags
    div = type('Node', (), {'tag': 'div', 'text': 'Block1', 'tail': None, 'getchildren': lambda: []})()
    p = type('Node', (), {'tag': 'p', 'text': 'Block2', 'tail': None, 'getchildren': lambda: []})()
    container = type('Node', (), {'tag': 'body', 'text': None, 'tail': None, 'getchildren': lambda: [div, p]})()
    assert extract_text(container) == 'Block1\nBlock2'
    
    # Test with nested structure
    inner_span = type('Node', (), {'tag': 'span', 'text': 'inner', 'tail': ' after', 'getchildren': lambda: []})()
    outer_div = type('Node', (), {'tag': 'div', 'text': 'start ', 'tail': None, 'getchildren': lambda: [inner_span]})()
    assert extract_text(outer_div) == 'start inner after'
    
    # Test with whitespace squashing
    p = type('Node', (), {'tag': 'p', 'text': '  Hello   World  ', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(p) == 'Hello World'
    
    # Test with custom symbols
    p = type('Node', (), {'tag': 'p', 'text': 'Text', 'tail': None, 'getchildren': lambda: []})()
    div = type('Node', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda: [p]})()
    assert extract_text(div, block_symbol='|') == '|Text|'
    
    # Test with empty content
    empty = type('Node', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(empty) == ''
    
    # Test with squash_space=False
    p = type('Node', (), {'tag': 'p', 'text': '  Hello  ', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(p, squash_space=False) == '  Hello  '


# LLM-generated content at query #62
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml.html import fromstring
    dom = fromstring("<span>Hello World</span>")
    assert extract_text_array(dom) == ["Hello World"]

    # Test with block tag
    dom = fromstring("<div>Hello World</div>")
    assert extract_text_array(dom) == [None, "Hello World", None]

    # Test with separator tag
    dom = fromstring("<br>")
    assert extract_text_array(dom) == [True]

    # Test with nested tags
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, "First", None, "Second", None]

    # Test with inline tags inside block tags
    dom = fromstring("<div>Hello <b>World</b>!</div>")
    assert extract_text_array(dom) == [None, "Hello ", "World", "!", None]

    # Test with tail text
    dom = fromstring("<div><span>Hello</span> World</div>")
    assert extract_text_array(dom) == [None, "Hello", " World", None]

    # Test with multiple children
    dom = fromstring("<ul><li>Item1</li><li>Item2</li></ul>")
    result = extract_text_array(dom)
    assert result == [None, "Item1", None, "Item2", None]

    # Test with empty tag
    dom = fromstring("<div></div>")
    assert extract_text_array(dom) == [None, None]

    # Test with text only (no tags)
    dom = fromstring("Just text")
    assert extract_text_array(dom) == ["Just text"]

    # Test with nested inline tags
    dom = fromstring("<span>Hello <b>bold</b> world</span>")
    assert extract_text_array(dom) == ["Hello ", "bold", " world"]

    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>Test</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, "Test", None]

    # Test with strip_artifical_nl=False
    dom = fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Text", None]


# LLM-generated content at query #63
#--------------------------

```python
def test_extract_text():
    # Test simple inline tag
    from lxml.html import fromstring
    dom = fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(dom) == "Hello world"
    
    # Test separator tag (br)
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test block elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test nested block elements
    dom = fromstring("<div><p>Text</p><div><p>Nested</p></div></div>")
    assert extract_text(dom) == "Text\nNested"
    
    # Test text with whitespace
    dom = fromstring("<p>  Hello   world  </p>")
    assert extract_text(dom) == "Hello world"
    
    # Test empty element
    dom = fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test element with only whitespace
    dom = fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test multiple inline tags
    dom = fromstring("<p><span>Hello</span> <span>world</span></p>")
    assert extract_text(dom) == "Hello world"
    
    # Test custom block_symbol
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=" | ") == "First | Second"
    
    # Test custom sep_symbol
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=" | ") == "Line1 | Line2"
    
    # Test with squash_space=False
    dom = fromstring("<p>Hello <b>world</b></p>")
    result = extract_text(dom, squash_space=False)
    assert "Hello" in result and "world" in result
    
    # Test script tag should be ignored
    dom = fromstring("<div><script>var x = 1;</script><p>Content</p></div>")
    assert extract_text(dom) == "Content"
    
    # Test complex nested structure
    dom = fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <a href='#'>link</a> and <br> break</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph" in result
    assert "link" in result
    assert "break" in result
    assert "Item 1" in result
    assert "Item 2" in result
    
    # Test strip behavior
    dom = fromstring("<div>  <p>Text</p>  </div>")
    assert extract_text(dom) == "Text"


# LLM-generated content at query #64
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    dom = type('Mock', (), {'tag': 'span', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == 'Hello'
    
    # Test with separator tag (br)
    br_dom = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(br_dom, sep_symbol='\n') == ''
    
    # Test with block-level tag (div)
    div_dom = type('Mock', (), {'tag': 'div', 'text': 'Content', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(div_dom) == 'Content'
    
    # Test with nested inline elements
    child_span = type('Mock', (), {'tag': 'span', 'text': 'World', 'tail': None, 'getchildren': lambda self: []})()
    parent_div = type('Mock', (), {'tag': 'div', 'text': 'Hello ', 'tail': None, 'getchildren': lambda self: [child_span]})()
    assert extract_text(parent_div) == 'Hello World'
    
    # Test with whitespace handling
    whitespace_dom = type('Mock', (), {'tag': 'span', 'text': 'Hello   World', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(whitespace_dom) == 'Hello World'
    
    # Test with multiple br tags
    br1 = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    br2 = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: [br1, br2]})()
    assert extract_text(parent, sep_symbol='\n') == '\n\n'
    
    # Test with mixed inline and block elements
    inner_span = type('Mock', (), {'tag': 'span', 'text': 'inner', 'tail': ' after', 'getchildren': lambda self: []})()
    outer_div = type('Mock', (), {'tag': 'div', 'text': 'before ', 'tail': None, 'getchildren': lambda self: [inner_span]})()
    result = extract_text(outer_div)
    assert result == 'before inner after'
    
    # Test with empty element
    empty = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(empty) == ''


# LLM-generated content at query #65
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with block tag
    dom = html.fromstring("<div>Hello</div>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with separator tag
    dom = html.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with nested tags
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert None in result  # artificial newlines present
    assert "First" in result
    assert "Second" in result
    
    # Test with inline tags inside block
    dom = html.fromstring("<p>Hello <b>world</b>!</p>")
    result = extract_text_array(dom)
    assert "Hello " in result
    assert "world" in result
    assert "!" in result
    
    # Test with text and tail
    dom = html.fromstring("<p>Text <span>inner</span> tail</p>")
    result = extract_text_array(dom)
    assert "Text " in result
    assert "inner" in result
    assert " tail" in result
    
    # Test squash_artifical_nl=False
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    none_count = sum(1 for x in result if x is None)
    assert none_count >= 4  # Two for outer div, two for inner p tags
    
    # Test strip_artifical_nl=False
    dom = html.fromstring("<div>Content</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # Leading artificial newline
    assert result[-1] is None  # Trailing artificial newline
    
    # Test with callable tag (should return empty string)
    class FakeElement:
        tag = lambda: None
    dom = FakeElement()
    assert extract_text_array(dom) == ''
    
    # Test empty element
    dom = html.fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test with multiple separators
    dom = html.fromstring("<div><br/><br/></div>")
    result = extract_text_array(dom)
    assert result == [True, True]


# LLM-generated content at query #66
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction from a simple element
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"

    # Test inline tags don't create newlines
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"

    # Test block tags create newlines
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"

    # Test separator tags
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"

    # Test whitespace squashing
    dom = html.fromstring("<p>Hello    World</p>")
    assert extract_text(dom) == "Hello World"

    # Test nested inline elements
    dom = html.fromstring("<p>Hello <b><i>beautiful</i></b> World</p>")
    assert extract_text(dom) == "Hello beautiful World"

    # Test empty elements
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""

    # Test elements with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""

    # Test custom block_symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' ') == "First Second"

    # Test custom sep_symbol
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=' | ') == "Line1 | Line2"

    # Test squash_space=False
    dom = html.fromstring("<p>Hello    World</p>")
    assert extract_text(dom, squash_space=False) == "Hello    World"

    # Test script tags return empty
    dom = html.fromstring("<div><script>var x = 1;</script></div>")
    assert extract_text(dom) == ""

    # Test multiple nested block elements
    dom = html.fromstring("<div><section><article><p>Content</p></article></section></div>")
    assert extract_text(dom) == "Content"

    # Test mixed inline and block elements
    dom = html.fromstring("<div><p>Hello <span>beautiful</span></p><p>World</p></div>")
    assert extract_text(dom) == "Hello beautiful\nWorld"

    # Test leading/trailing whitespace stripping
    dom = html.fromstring("<p>  Hello World  </p>")
    assert extract_text(dom) == "Hello World"

    # Test newlines in text
    dom = html.fromstring("<p>Hello\nWorld</p>")
    assert extract_text(dom) == "Hello World"

    # Test tabs in text
    dom = html.fromstring("<p>Hello\tWorld</p>")
    assert extract_text(dom) == "Hello World"

    # Test complex nested structure
    dom = html.fromstring("""
        <div>
            <p>First <b>bold</b> text</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
            <p>Second paragraph</p>
        </div>
    """)
    result = extract_text(dom)
    assert result == "First bold text\nItem 1\nItem 2\nSecond paragraph"
```


# LLM-generated content at query #67
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>hello</span>")
    assert extract_text_array(dom) == ["hello"]
    
    # Test with a block element
    dom = html.fromstring("<div>hello</div>")
    assert extract_text_array(dom) == [None, "hello", None]
    
    # Test with a separator element
    dom = html.fromstring("<br>")
    assert extract_text_array(dom) == [True]
    
    # Test nested elements
    dom = html.fromstring("<div><p>text1</p><p>text2</p></div>")
    result = extract_text_array(dom)
    assert None in result
    assert "text1" in result
    assert "text2" in result
    
    # Test with squash_artifical_nl=True
    dom = html.fromstring("<div><p>text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result.count(None) == 1  # consecutive Nones should be squashed
    
    # Test with strip_artifical_nl=True
    dom = html.fromstring("<div><p>text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result[0] == "text"  # leading/trailing Nones should be stripped
    
    # Test with both squash and strip
    dom = html.fromstring("<div><p>text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["text"]
    
    # Test with text and tail
    dom = html.fromstring("<div>before<span>inside</span>after</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert "before" in result
    assert "inside" in result
    assert "after" in result
    
    # Test with empty element
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None] or result == []  # depending on squash/strip
    
    # Test with callable tag (should return empty string)
    mock_dom = type('Mock', (), {'tag': lambda: None})()
    assert extract_text_array(mock_dom) == ''


# LLM-generated content at query #68
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    doc = html.fromstring("<span>Hello</span>")
    assert extract_text_array(doc) == ["Hello"]
    
    # Test with block tag
    doc = html.fromstring("<div>Hello</div>")
    assert extract_text_array(doc) == [None, "Hello", None]
    
    # Test with separator tag
    doc = html.fromstring("<br>")
    assert extract_text_array(doc) == [True]
    
    # Test with nested tags
    doc = html.fromstring("<div><span>Hello</span> World</div>")
    result = extract_text_array(doc)
    assert result == [None, "Hello", " World", None]
    
    # Test with multiple children
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(doc)
    assert None in result
    assert "First" in result
    assert "Second" in result
    
    # Test with text in parent and children
    doc = html.fromstring("<div>Text <span>child</span> tail</div>")
    result = extract_text_array(doc)
    assert "Text " in result
    assert "child" in result
    assert " tail" in result
    
    # Test with squash_artifical_nl=True (default)
    doc = html.fromstring("<div></div>")
    result = extract_text_array(doc)
    assert result == []  # Squashed to nothing
    
    # Test with squash_artifical_nl=False
    doc = html.fromstring("<div></div>")
    result = extract_text_array(doc, squash_artifical_nl=False)
    assert result == [None, None]  # Two None values preserved
    
    # Test with strip_artifical_nl=True (default)
    doc = html.fromstring("<div>Hello</div>")
    result = extract_text_array(doc)
    assert result == ["Hello"]  # Leading/trailing None stripped
    
    # Test with strip_artifical_nl=False
    doc = html.fromstring("<div>Hello</div>")
    result = extract_text_array(doc, strip_artifical_nl=False)
    assert result == [None, "Hello", None]  # None values preserved
    
    # Test with callable tag (should return empty string)
    class FakeElement:
        tag = lambda: None
    assert extract_text_array(FakeElement()) == ''
    
    # Test with None text
    doc = html.fromstring("<div></div>")
    doc.text = None
    result = extract_text_array(doc)
    assert None in result or result == []


# LLM-generated content at query #69
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    from lxml import html
    doc = html.fromstring("<span>Hello World</span>")
    assert extract_text(doc) == "Hello World"
    
    # Test with block element
    doc = html.fromstring("<div>Hello</div><div>World</div>")
    assert extract_text(doc) == "Hello\nWorld"
    
    # Test with separator element (br)
    doc = html.fromstring("<p>Hello<br/>World</p>")
    assert extract_text(doc) == "Hello\nWorld"
    
    # Test with nested elements
    doc = html.fromstring("<div><p>Hello <b>World</b></p></div>")
    assert extract_text(doc) == "Hello World"
    
    # Test with whitespace squashing
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with multiple newlines (artificial)
    doc = html.fromstring("<div>Hello</div><div>World</div><div>Test</div>")
    assert extract_text(doc) == "Hello\nWorld\nTest"
    
    # Test with empty content
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with only whitespace
    doc = html.fromstring("<div>   </div>")
    assert extract_text(doc) == ""
    
    # Test with custom symbols
    doc = html.fromstring("<div>Hello</div><div>World</div>")
    assert extract_text(doc, block_symbol=' | ') == "Hello | World"
    
    # Test with sep_symbol
    doc = html.fromstring("<p>Hello<br/>World</p>")
    assert extract_text(doc, sep_symbol=' | ') == "Hello | World"
    
    # Test without squashing space
    doc = html.fromstring("<div>  Hello  World  </div>")
    assert extract_text(doc, squash_space=False) == "  Hello  World  "
    
    # Test with mixed inline/block elements
    doc = html.fromstring("<div><span>Hello</span><div>World</div></div>")
    assert extract_text(doc) == "Hello\nWorld"
    
    # Test with multiple separators
    doc = html.fromstring("<p>Line1<br/><br/>Line2</p>")
    assert extract_text(doc) == "Line1\n\nLine2"


# LLM-generated content at query #70
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tags
    from lxml.html import fromstring
    doc = fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with separator tag (br)
    doc = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with block tags creating newlines
    doc = fromstring("<div>First</div><div>Second</div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with nested tags
    doc = fromstring("<div><p>Para <span>one</span></p><p>Para two</p></div>")
    assert extract_text(doc) == "Para one\nPara two"
    
    # Test whitespace squashing
    doc = fromstring("<p>Hello     world</p>")
    assert extract_text(doc) == "Hello world"
    
    # Test whitespace with newlines
    doc = fromstring("<p>Hello\n    world</p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with custom block_symbol
    doc = fromstring("<div>A</div><div>B</div>")
    assert extract_text(doc, block_symbol='.') == "A.B"
    
    # Test with custom sep_symbol
    doc = fromstring("<p>A<br>B</p>")
    assert extract_text(doc, sep_symbol='|') == "A|B"
    
    # Test with no squashing
    doc = fromstring("<p>  Hello   world  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello   world  "
    
    # Test with image tag (inline)
    doc = fromstring("<p>Text <img src='test.jpg'> more text</p>")
    assert extract_text(doc) == "Text  more text"
    
    # Test empty text
    doc = fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test text only
    doc = fromstring("Simple text")
    assert extract_text(doc) == "Simple text"
    
    # Test nested block elements
    doc = fromstring("<body><h1>Title</h1><p>Content</p></body>")
    assert extract_text(doc) == "Title\nContent"
    
    # Test with script tag (inline but should be handled)
    doc = fromstring("<div><script>alert('test')</script>Text</div>")
    assert extract_text(doc) == "Text"


# LLM-generated content at query #71
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    dom = html.fromstring("<span>hello</span>")
    assert extract_text_array(dom) == ["hello"]
    
    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with block tag
    dom = html.fromstring("<div>text</div>")
    assert extract_text_array(dom) == ["text"]
    
    # Test with nested elements
    dom = html.fromstring("<div><span>hello</span> world</div>")
    result = extract_text_array(dom)
    assert "hello" in result
    assert " world" in result
    
    # Test with multiple children
    dom = html.fromstring("<p>first <b>bold</b> second</p>")
    result = extract_text_array(dom)
    assert result == ["first ", "bold", " second"]
    
    # Test with br separator
    dom = html.fromstring("<div>line1<br/>line2</div>")
    result = extract_text_array(dom)
    assert True in result  # br produces True
    
    # Test with callable tag (should return empty string)
    class CallableTag:
        def __call__(self):
            pass
    dom = type('obj', (object,), {'tag': CallableTag(), 'text': None, 'getchildren': lambda: []})()
    assert extract_text_array(dom) == ''
    
    # Test with None text
    dom = html.fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test squash_artifical_nl=False
    dom = html.fromstring("<div><p>text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result  # artificial newlines not squashed
    
    # Test strip_artifical_nl=False
    dom = html.fromstring("<div><p>text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None  # leading/trailing None preserved
    
    # Test complex nested structure
    dom = html.fromstring("<div><p>para1</p><br/><p>para2</p></div>")
    result = extract_text_array(dom)
    assert "para1" in result
    assert True in result
    assert "para2" in result


# LLM-generated content at query #72
#--------------------------

```python
def test_extract_text_array():
    # Test empty DOM
    class EmptyElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(EmptyElement()) == []

    # Test inline element
    class InlineElement:
        tag = 'span'
        text = 'hello'
        def getchildren(self):
            return []
    assert extract_text_array(InlineElement()) == ['hello']

    # Test block element (adds None for artificial newlines)
    class BlockElement:
        tag = 'div'
        text = 'hello'
        def getchildren(self):
            return []
    result = extract_text_array(BlockElement())
    assert None in result
    assert 'hello' in result

    # Test separator element
    class SeparatorElement:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(SeparatorElement()) == [True]

    # Test with children
    class ParentElement:
        tag = 'div'
        text = 'parent'
        def getchildren(self):
            child = InlineElement()
            child.tail = ' tail'
            return [child]
    result = extract_text_array(ParentElement())
    assert 'parent' in result
    assert 'hello' in result
    assert ' tail' in result

    # Test squash_artifical_nl=True
    class NestedBlock:
        tag = 'div'
        text = None
        def getchildren(self):
            child = BlockElement()
            child.tag = 'p'
            child.text = 'child'
            child.getchildren = lambda: []
            return [child]
    result = extract_text_array(NestedBlock(), squash_artifical_nl=True)
    # Should have squashed consecutive None values
    none_count = sum(1 for x in result if x is None)
    assert none_count <= 2  # at most 2 None (one for outer div, one for inner p)

    # Test strip_artifical_nl=True
    class LeadingTrailingBlock:
        tag = 'div'
        text = 'text'
        def getchildren(self):
            return []
    result = extract_text_array(LeadingTrailingBlock(), strip_artifical_nl=True)
    # Should not start or end with None
    if result:
        assert result[0] is not None
        assert result[-1] is not None

    # Test callable tag returns empty string
    class CallableTagElement:
        tag = lambda: None
        text = 'test'
        def getchildren(self):
            return []
    assert extract_text_array(CallableTagElement()) == ''


# LLM-generated content at query #73
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self.children = children or []
        
        def getchildren(self):
            return self.children
    
    # Test empty element
    elem = MockElement('div')
    assert extract_text(elem) == ''
    
    # Test simple text
    elem = MockElement('p', text='Hello World')
    assert extract_text(elem) == 'Hello World'
    
    # Test inline element (span)
    span = MockElement('span', text='inline')
    elem = MockElement('p', text='Some ', children=[span], tail=' text')
    assert extract_text(elem) == 'Some inline text'
    
    # Test separator element (br)
    br = MockElement('br')
    elem = MockElement('div', text='Line1', children=[br], tail='Line2')
    assert extract_text(elem, sep_symbol='\n') == 'Line1\nLine2'
    
    # Test block elements
    inner = MockElement('p', text='Inner')
    elem = MockElement('div', children=[inner])
    assert extract_text(elem, block_symbol='\n') == 'Inner'
    
    # Test nested blocks
    inner = MockElement('div', text='Nested')
    outer = MockElement('div', children=[inner])
    assert extract_text(outer, block_symbol='\n') == 'Nested'
    
    # Test whitespace squashing
    elem = MockElement('p', text='Hello   World')
    assert extract_text(elem) == 'Hello World'
    
    # Test multiple children
    child1 = MockElement('span', text='First')
    child2 = MockElement('span', text='Second')
    elem = MockElement('div', children=[child1, child2])
    assert extract_text(elem) == 'FirstSecond'
    
    # Test with block_symbol
    block1 = MockElement('div', text='Block1')
    block2 = MockElement('div', text='Block2')
    container = MockElement('div', children=[block1, block2])
    assert extract_text(container, block_symbol='\n') == 'Block1\nBlock2'
    
    # Test with custom symbols
    br = MockElement('br')
    elem = MockElement('div', text='A', children=[br], tail='B')
    assert extract_text(elem, sep_symbol='|') == 'A|B'
    
    # Test complex nested structure
    inner_span = MockElement('span', text='inner')
    outer_div = MockElement('div', text='Before ', children=[inner_span], tail=' After')
    assert extract_text(outer_div) == 'Before inner After'


# LLM-generated content at query #74
#--------------------------

```python
def test_extract_text_array():
    from lxml import etree
    
    # Test 1: Simple inline text
    dom = etree.fromstring("<p>Hello World</p>")
    assert extract_text_array(dom) == ["Hello World"]
    
    # Test 2: Nested inline elements
    dom = etree.fromstring("<p>Hello <b>bold</b> World</p>")
    assert extract_text_array(dom) == ["Hello ", "bold", " World"]
    
    # Test 3: Block elements with artificial newlines
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert None in result  # Should have artificial newlines
    assert result[0] == "First"
    assert result[-1] == "Second"
    
    # Test 4: Separator elements (br)
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert True in result  # br should be separator
    assert result[0] == "Line1"
    assert result[-1] == "Line2"
    
    # Test 5: Non-inline tag with children
    dom = etree.fromstring("<div><span>text</span></div>")
    result = extract_text_array(dom)
    assert result == ["text"]
    
    # Test 6: Empty element
    dom = etree.fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test 7: Mixed inline and block elements
    dom = etree.fromstring("<div><p>Para <b>bold</b></p><span>inline</span></div>")
    result = extract_text_array(dom)
    assert "Para " in result
    assert "bold" in result
    assert "inline" in result
    
    # Test 8: Text with tail
    dom = etree.fromstring("<p>Start<b>bold</b>End</p>")
    result = extract_text_array(dom)
    assert result == ["Start", "bold", "End"]
    
    # Test 9: Squash artificial newlines
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result.count(None) <= 1  # Should squash consecutive Nones
    
    # Test 10: Strip artificial newlines
    dom = etree.fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result[0] == "A"  # Leading/trailing Nones should be stripped
    
    # Test 11: Callable tag returns empty string
    dom = etree.fromstring("<div>text</div>")
    dom.tag = lambda: None
    assert extract_text_array(dom) == ''
    
    # Test 12: Multiple separators
    dom = etree.fromstring("<p>A<br/>B<br/>C</p>")
    result = extract_text_array(dom)
    assert result.count(True) == 2  # Two br elements
    
    # Test 13: Deep nesting
    dom = etree.fromstring("<div><p><span><b>deep</b></span></p></div>")
    result = extract_text_array(dom)
    assert result == ["deep"]


# LLM-generated content at query #75
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag (span)
    from html.parser import HTMLParser
    from lxml import html
    
    # Test empty element
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [], f"Expected empty list, got {result}"
    
    # Test with text only
    dom = html.fromstring("<div>Hello World</div>")
    result = extract_text_array(dom)
    assert result == ['Hello World'], f"Expected ['Hello World'], got {result}"
    
    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"
    
    # Test with inline tag (span)
    dom = html.fromstring("<span>text</span>")
    result = extract_text_array(dom)
    assert result == ['text'], f"Expected ['text'], got {result}"
    
    # Test with block tag (div)
    dom = html.fromstring("<div><p>text1</p><p>text2</p></div>")
    result = extract_text_array(dom)
    # Should have None (artificial newline) between block elements
    assert None in result, "Expected None (artificial newline) in result"
    assert 'text1' in result, "Expected 'text1' in result"
    assert 'text2' in result, "Expected 'text2' in result"
    
    # Test with nested structure
    dom = html.fromstring("<div><span>inner</span> tail</div>")
    result = extract_text_array(dom)
    assert 'inner' in result, "Expected 'inner' in result"
    assert ' tail' in result, "Expected ' tail' in result"
    
    # Test that strip_artifical_nl removes leading/trailing None
    dom = html.fromstring("<div><p>text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result[0] == 'text', f"Expected 'text' as first element, got {result}"
    
    # Test that squash_artifical_nl consolidates consecutive None
    dom = html.fromstring("<div><p></p><p></p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    none_count = sum(1 for x in result if x is None)
    assert none_count <= 1, f"Expected at most 1 None, got {none_count}"
    
    # Test with callable tag (edge case)
    class MockTag:
        def __call__(self):
            pass
    
    class MockElement:
        tag = MockTag()
        text = None
        def getchildren(self):
            return []
    
    result = extract_text_array(MockElement())
    assert result == [], f"Expected empty list for callable tag, got {result}"
    
    # Test with mixed content
    dom = html.fromstring("<div>before <span>inside</span> after</div>")
    result = extract_text_array(dom)
    assert 'before ' in result, "Expected 'before ' in result"
    assert 'inside' in result, "Expected 'inside' in result"
    assert ' after' in result, "Expected ' after' in result"


# LLM-generated content at query #76
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import etree
    dom = etree.fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"], f"Expected ['Hello'], got {result}"
    
    # Test with separator tag
    dom = etree.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"
    
    # Test with block tag
    dom = etree.fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result == [None, "Text", None], f"Expected [None, 'Text', None], got {result}"
    
    # Test with nested inline in block
    dom = etree.fromstring("<div><span>Hello</span> World</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", " World", None], f"Expected [None, 'Hello', ' World', None], got {result}"
    
    # Test with separator in block
    dom = etree.fromstring("<div>Line1<br/>Line2</div>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None], f"Expected [None, 'Line1', True, 'Line2', None], got {result}"
    
    # Test with block inside block
    dom = etree.fromstring("<div><p>Para</p>After</div>")
    result = extract_text_array(dom)
    assert result == [None, None, "Para", None, "After", None], f"Expected [None, None, 'Para', None, 'After', None], got {result}"
    
    # Test with squash_artifical_nl=True (default)
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == [None, "A", None, "B", None], f"Expected [None, 'A', None, 'B', None], got {result}"
    
    # Test with squash_artifical_nl=False
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert len(result) > 3, f"Expected more than 3 elements, got {result}"
    
    # Test with strip_artifical_nl=True (default)
    dom = etree.fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == ["Hello"], f"Expected ['Hello'], got {result}"
    
    # Test with strip_artifical_nl=False
    dom = etree.fromstring("<div>Hello</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert None in result or result[0] is None, f"Expected None at start, got {result}"
    
    # Test with callable tag
    dom = etree.fromstring("<div>Text</div>")
    dom.tag = lambda: None
    result = extract_text_array(dom)
    assert result == [], f"Expected [], got {result}"
    
    # Test empty element
    dom = etree.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [], f"Expected [], got {result}"
    
    # Test deep nesting
    dom = etree.fromstring("<div><span><b>Deep</b></span></div>")
    result = extract_text_array(dom)
    assert result == ["Deep"], f"Expected ['Deep'], got {result}"
    
    # Test text only
    dom = etree.fromstring("Simple text")
    result = extract_text_array(dom)
    assert result == ["Simple text"], f"Expected ['Simple text'], got {result}"


# LLM-generated content at query #77
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml.html import fromstring
    dom = fromstring("<span>hello</span>")
    assert extract_text_array(dom) == ["hello"]
    
    # Test with block tag (creates artificial newlines)
    dom = fromstring("<div>hello</div>")
    result = extract_text_array(dom)
    assert None in result  # artificial newline
    assert "hello" in result
    
    # Test with separator tag (br)
    dom = fromstring("<br/>")
    result = extract_text_array(dom)
    assert True in result  # separator marker
    
    # Test with nested elements
    dom = fromstring("<div><p>first</p><p>second</p></div>")
    result = extract_text_array(dom)
    assert result.count(None) >= 2  # artificial newlines around each paragraph
    
    # Test with text and tail
    dom = fromstring("<div>text1<span>inner</span>text2</div>")
    result = extract_text_array(dom)
    assert "text1" in result
    assert "inner" in result
    assert "text2" in result
    
    # Test with callable tag (should return empty string)
    dom = fromstring("<div>test</div>")
    dom.tag = lambda: None  # simulate callable tag
    assert extract_text_array(dom) == ''


# LLM-generated content at query #78
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]

    # Test with inline elements
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]

    # Test with separator element (br)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]

    # Test with block element inside
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, "First", None, None, "Second", None]

    # Test with nested inline elements
    dom = html.fromstring("<span>Text <em>emphasized</em> more text</span>")
    result = extract_text_array(dom)
    assert result == ["Text ", "emphasized", " more text"]

    # Test with callable tag (should return empty string)
    class MockCallable:
        def __call__(self):
            pass
    mock_dom = lambda: None
    mock_dom.tag = lambda: None
    mock_dom.text = None
    mock_dom.getchildren = lambda: []
    result = extract_text_array(mock_dom)
    assert result == []

    # Test squash_artifical_nl parameter
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ["A", None, "B"]

    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result  # Should have None values

    # Test strip_artifical_nl parameter
    dom = html.fromstring("<div><p>Content</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["Content"]

    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # Should start with None

    # Test with empty content
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []

    # Test with tail text
    dom = html.fromstring("<p>Before <b>bold</b> After</p>")
    result = extract_text_array(dom)
    assert result == ["Before ", "bold", " After"]


# LLM-generated content at query #79
#--------------------------

```python
def test_extract_text():
    # Test with simple inline tag
    from lxml.html import fromstring
    dom = fromstring('<span>hello world</span>')
    assert extract_text(dom) == 'hello world'
    
    # Test with block tag
    dom = fromstring('<div>hello</div>')
    assert extract_text(dom) == 'hello'
    
    # Test with separator tag
    dom = fromstring('<br>')
    assert extract_text(dom) == '\n'
    
    # Test with nested tags
    dom = fromstring('<div><p>first</p><p>second</p></div>')
    assert extract_text(dom) == 'first\nsecond'
    
    # Test with mixed inline and block
    dom = fromstring('<div><span>hello</span> <strong>world</strong></div>')
    assert extract_text(dom) == 'hello world'
    
    # Test with whitespace squashing
    dom = fromstring('<div>  hello   world  </div>')
    assert extract_text(dom) == 'hello world'
    
    # Test with multiple whitespace characters
    dom = fromstring('<div>hello\t\n\rworld</div>')
    assert extract_text(dom) == 'hello world'
    
    # Test empty content
    dom = fromstring('<div></div>')
    assert extract_text(dom) == ''
    
    # Test with custom block_symbol
    dom = fromstring('<div><p>first</p><p>second</p></div>')
    assert extract_text(dom, block_symbol=' | ') == 'first | second'
    
    # Test with custom sep_symbol
    dom = fromstring('<div>hello<br>world</div>')
    assert extract_text(dom, sep_symbol=' --- ') == 'hello --- world'
    
    # Test with squash_space=False
    dom = fromstring('<div>  hello   world  </div>')
    assert extract_text(dom, squash_space=False) == '  hello   world  '
    
    # Test with nested block elements
    dom = fromstring('<div><p>first</p><div><p>nested</p></div><p>last</p></div>')
    assert extract_text(dom) == 'first\nnested\nlast'
    
    # Test with inline elements inside block
    dom = fromstring('<p>This is <strong>important</strong> text</p>')
    assert extract_text(dom) == 'This is important text'
    
    # Test with script tag (treated as inline)
    dom = fromstring('<script>var x = 1;</script>')
    assert extract_text(dom) == 'var x = 1;'
    
    # Test with image tag
    dom = fromstring('<div>text <img src="test.jpg"/> tail</div>')
    assert extract_text(dom) == 'text tail'


# LLM-generated content at query #80
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text_array(dom) == ["Hello World"]
    
    # Test with block tag
    dom = html.fromstring("<div>Hello World</div>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    # Note: block tags add None around the text
    
    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert True in result  # Separators add True
    
    # Test with nested tags
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert "First" in result
    assert "Second" in result
    
    # Test with text and tail
    dom = html.fromstring("<div>Hello <b>World</b> Again</div>")
    result = extract_text_array(dom)
    assert "Hello " in result
    assert "World" in result
    assert " Again" in result
    
    # Test with squash_artifical_nl=True
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    # Should have fewer None values
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # First element should be None
    
    # Test with callable tag
    class MockDom:
        tag = lambda: None
    assert extract_text_array(MockDom()) == ''


# LLM-generated content at query #81
#--------------------------

```python
def test_extract_text_array():
    # Test with None tag (callable)
    class MockCallableTag:
        def __call__(self):
            pass
    mock_dom = type('MockDom', (), {'tag': MockCallableTag()})()
    assert extract_text_array(mock_dom) == ''
    
    # Test with separator tag (br)
    mock_dom = type('MockDom', (), {
        'tag': 'br',
        'text': None,
        'getchildren': lambda self: []
    })()
    result = extract_text_array(mock_dom)
    assert result == [True]
    
    # Test with inline tag
    mock_dom = type('MockDom', (), {
        'tag': 'span',
        'text': 'hello',
        'getchildren': lambda self: []
    })()
    result = extract_text_array(mock_dom)
    assert result == ['hello']
    
    # Test with block tag
    mock_dom = type('MockDom', (), {
        'tag': 'div',
        'text': 'hello',
        'getchildren': lambda self: []
    })()
    result = extract_text_array(mock_dom)
    assert result == [None, 'hello', None]
    
    # Test with children
    child = type('MockDom', (), {
        'tag': 'span',
        'text': 'child',
        'tail': None,
        'getchildren': lambda self: []
    })()
    parent = type('MockDom', (), {
        'tag': 'div',
        'text': 'parent ',
        'getchildren': lambda self: [child],
        'tail': None
    })()
    result = extract_text_array(parent)
    assert 'parent ' in result
    assert 'child' in result
    
    # Test squash_artifical_nl parameter
    mock_dom = type('MockDom', (), {
        'tag': 'div',
        'text': None,
        'getchildren': lambda self: []
    })()
    result_no_squash = extract_text_array(mock_dom, squash_artifical_nl=False)
    assert result_no_squash == [None, None]
    result_squash = extract_text_array(mock_dom, squash_artifical_nl=True)
    assert result_squash == [None]
    
    # Test strip_artifical_nl parameter
    mock_dom = type('MockDom', (), {
        'tag': 'div',
        'text': 'text',
        'getchildren': lambda self: []
    })()
    result_no_strip = extract_text_array(mock_dom, strip_artifical_nl=False)
    assert result_no_strip == [None, 'text', None]
    result_strip = extract_text_array(mock_dom, strip_artifical_nl=True)
    assert result_strip == ['text']


# LLM-generated content at query #82
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction from a simple element
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test extraction with inline tags
    dom = html.fromstring("<p>Hello <b>bold</b> World</p>")
    assert extract_text(dom) == "Hello bold World"
    
    # Test block-level elements create newlines
    dom = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test separator tags (br)
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test nested structures
    dom = html.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    assert extract_text(dom) == "Para1\nPara2"
    
    # Test with custom block_symbol
    dom = html.fromstring("<div>A</div><div>B</div>")
    assert extract_text(dom, block_symbol=" | ") == "A | B"
    
    # Test with custom sep_symbol
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol=" - ") == "Line1 - Line2"
    
    # Test squash_space=False
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test empty content
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test nested empty elements
    dom = html.fromstring("<div><p></p></div>")
    assert extract_text(dom) == ""
    
    # Test mixed content with whitespace
    dom = html.fromstring("<div>  Text  <span>  Span  </span>  More  </div>")
    assert extract_text(dom) == "Text Span More"
    
    # Test multiple br tags
    dom = html.fromstring("<p>A<br/><br/>B</p>")
    assert extract_text(dom) == "A\n\nB"
    
    # Test deep nesting
    dom = html.fromstring("<div><p><b><i>Deep</i></b></p></div>")
    assert extract_text(dom) == "Deep"
    
    # Test with script tag (should be ignored)
    dom = html.fromstring("<div>Text<script>alert('test')</script>More</div>")
    assert extract_text(dom) == "TextMore"


# LLM-generated content at query #83
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]

    # Test with inline tag
    dom = etree.fromstring("<p>Hello <b>World</b></p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]

    # Test with separator tag (br)
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]

    # Test with non-inline tag
    dom = etree.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom)
    assert result == ["Text"]

    # Test with nested inline tags
    dom = etree.fromstring("<p>Hello <i>beautiful <b>world</b></i></p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "beautiful ", "world"]

    # Test with empty element
    dom = etree.fromstring("<p></p>")
    result = extract_text_array(dom)
    assert result == []

    # Test with only whitespace
    dom = etree.fromstring("<p>   </p>")
    result = extract_text_array(dom)
    assert result == ["   "]

    # Test with callable tag (should return empty string)
    dom = etree.fromstring("<p>Text</p>")
    dom.tag = lambda: None
    result = extract_text_array(dom)
    assert result == ""

    # Test with multiple block-level elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"]

    # Test with tail text
    dom = etree.fromstring("<p>Hello<b>bold</b>world</p>")
    result = extract_text_array(dom)
    assert result == ["Hello", "bold", "world"]

    # Test with squash_artifical_nl=False and strip_artifical_nl=False
    dom = etree.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Text", None]

    # Test with strip_artifical_nl=False
    dom = etree.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == ["Text", None]

    # Test with multiple separators
    dom = etree.fromstring("<p>Line1<br/><br/>Line2</p>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, True, "Line2"]


# LLM-generated content at query #84
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<span>Hello World</span>")
    assert extract_text_array(dom) == ["Hello World"]
    
    # Test with non-inline element (creates artificial newlines)
    dom = fromstring("<div>Hello</div>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with separator element (br)
    dom = fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with nested structure
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"]
    
    # Test with inline elements inside block
    dom = fromstring("<div>Hello <b>World</b> again</div>")
    assert extract_text_array(dom) == ["Hello ", "World", " again"]
    
    # Test with separator between text
    dom = fromstring("<div>Line1<br/>Line2</div>")
    assert extract_text_array(dom) == ["Line1", True, "Line2"]
    
    # Test with nested separators
    dom = fromstring("<div><br/><br/></div>")
    result = extract_text_array(dom)
    assert result == [True, True]
    
    # Test with empty element
    dom = fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test with only whitespace
    dom = fromstring("<div>   </div>")
    assert extract_text_array(dom) == ["   "]
    
    # Test with nested non-inline elements
    dom = fromstring("<div><section><p>Text</p></section></div>")
    assert extract_text_array(dom) == ["Text"]
    
    # Test with inline tag that is not in INLINE_TAGS
    dom = fromstring("<div><custom>Text</custom></div>")
    assert extract_text_array(dom) == ["Text"]


# LLM-generated content at query #85
#--------------------------

```python
def test_extract_text_array():
    # Test 1: Simple text element
    from lxml import html
    dom = html.fromstring('<p>Hello World</p>')
    result = extract_text_array(dom)
    assert result == ['Hello World']
    
    # Test 2: Nested inline elements
    dom = html.fromstring('<p>Hello <b>bold</b> World</p>')
    result = extract_text_array(dom)
    assert result == ['Hello ', 'bold', ' World']
    
    # Test 3: Block element with nested content
    dom = html.fromstring('<div><p>First</p><p>Second</p></div>')
    result = extract_text_array(dom)
    expected = [None, 'First', None, None, 'Second', None, None]
    assert result == expected
    
    # Test 4: Separator element (br)
    dom = html.fromstring('<p>Line1<br/>Line2</p>')
    result = extract_text_array(dom)
    assert result == ['Line1', True, 'Line2']
    
    # Test 5: Empty element
    dom = html.fromstring('<p></p>')
    result = extract_text_array(dom)
    assert result == []
    
    # Test 6: Element with only tail text
    dom = html.fromstring('<div>Text</div>')
    result = extract_text_array(dom)
    assert result == [None, 'Text', None]
    
    # Test 7: Nested block elements
    dom = html.fromstring('<div><span>Inline</span></div>')
    result = extract_text_array(dom)
    expected = [None, 'Inline', None]
    assert result == expected
    
    # Test 8: Multiple children with mixed types
    dom = html.fromstring('<div>Start <b>bold</b> middle <i>italic</i> end</div>')
    result = extract_text_array(dom)
    assert result == [None, 'Start ', 'bold', ' middle ', 'italic', ' end', None]
    
    # Test 9: Element with text and children
    dom = html.fromstring('<p>Before <span>inside</span> After</p>')
    result = extract_text_array(dom)
    assert result == ['Before ', 'inside', ' After']
    
    # Test 10: Deeply nested structure
    dom = html.fromstring('<div><p><span>Text</span></p></div>')
    result = extract_text_array(dom)
    expected = [None, None, 'Text', None, None]
    assert result == expected
    
    # Test 11: Multiple consecutive separators
    dom = html.fromstring('<p>Text<br/><br/>More</p>')
    result = extract_text_array(dom)
    assert result == ['Text', True, True, 'More']
    
    # Test 12: Callable tag (should return empty)
    class MockElement:
        tag = lambda: None
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ''
    
    # Test 13: Whitespace handling
    dom = html.fromstring('<p>  Hello   World  </p>')
    result = extract_text_array(dom)
    assert result == ['  Hello   World  ']
    
    # Test 14: Mixed inline and block with separators
    dom = html.fromstring('<div><p>Para1</p><br/><p>Para2</p></div>')
    result = extract_text_array(dom)
    expected = [None, None, 'Para1', None, True, None, 'Para2', None, None]
    assert result == expected
    
    # Test 15: Element with no text content
    dom = html.fromstring('<div><b></b></div>')
    result = extract_text_array(dom)
    expected = [None, None]
    assert result == expected


# LLM-generated content at query #86
#--------------------------

```python
def test_extract_text():
    # Test empty element
    from lxml import etree
    dom = etree.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test simple text
    dom = etree.fromstring("<div>Hello World</div>")
    assert extract_text(dom) == "Hello World"
    
    # Test inline tags (should not add newlines)
    dom = etree.fromstring("<p>Hello <strong>World</strong></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test block-level tags (should add newlines)
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test separator tags (br)
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test nested inline tags
    dom = etree.fromstring("<p><span>Hello</span> <em>World</em></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test whitespace squashing
    dom = etree.fromstring("<p>Hello     World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test custom block_symbol parameter
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test custom sep_symbol parameter
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol='|') == "Line1|Line2"
    
    # Test squash_space=False
    dom = etree.fromstring("<div><p>Hello</p><p>World</p></div>")
    assert extract_text(dom, squash_space=False) == "\nHello\nWorld\n"
    
    # Test complex nested structure
    dom = etree.fromstring("""
        <div>
            <h1>Title</h1>
            <p>This is a <strong>paragraph</strong> with <em>emphasis</em>.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(dom)
    assert "Title" in result
    assert "This is a paragraph with emphasis." in result
    assert "Item 1" in result
    assert "Item 2" in result
    
    # Test with empty tags
    dom = etree.fromstring("<div><p></p><p>Content</p></div>")
    assert extract_text(dom) == "Content"
    
    # Test with leading/trailing whitespace
    dom = etree.fromstring("<div>  Hello World  </div>")
    assert extract_text(dom) == "Hello World"
    
    # Test multiple consecutive block elements
    dom = etree.fromstring("<div><p>First</p><div>Second</div><p>Third</p></div>")
    result = extract_text(dom)
    assert result.count('\n') == 2
    
    # Test with mixed inline and block elements
    dom = etree.fromstring("<p><strong>Important:</strong> <em>very</em> important</p>")
    assert extract_text(dom) == "Important: very important"


# LLM-generated content at query #87
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml import html
    doc = html.fromstring('<span>Hello World</span>')
    assert extract_text(doc) == 'Hello World'
    
    # Test with block element (should add newlines)
    doc = html.fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(doc) == 'First\nSecond'
    
    # Test with separator element (br)
    doc = html.fromstring('<p>Line1<br>Line2</p>')
    assert extract_text(doc) == 'Line1\nLine2'
    
    # Test with nested inline elements
    doc = html.fromstring('<p>Hello <b>World</b>!</p>')
    assert extract_text(doc) == 'Hello World!'
    
    # Test with whitespace squashing
    doc = html.fromstring('<p>   Multiple    spaces   </p>')
    assert extract_text(doc) == 'Multiple spaces'
    
    # Test with leading/trailing whitespace
    doc = html.fromstring('<p>   Text   </p>')
    assert extract_text(doc) == 'Text'
    
    # Test with empty content
    doc = html.fromstring('<p></p>')
    assert extract_text(doc) == ''
    
    # Test with only nested elements
    doc = html.fromstring('<div><span></span></div>')
    assert extract_text(doc) == ''
    
    # Test with custom block_symbol
    doc = html.fromstring('<div><p>A</p><p>B</p></div>')
    assert extract_text(doc, block_symbol=' ') == 'A B'
    
    # Test with custom sep_symbol
    doc = html.fromstring('<p>A<br>B</p>')
    assert extract_text(doc, sep_symbol=' ') == 'A B'
    
    # Test with squash_space=False
    doc = html.fromstring('<p>   Text   </p>')
    assert extract_text(doc, squash_space=False) == '   Text   '
    
    # Test with complex nested structure
    doc = html.fromstring('''
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> text</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    ''')
    result = extract_text(doc)
    assert 'Title' in result
    assert 'Paragraph with bold text' in result
    assert 'Item 1' in result
    assert 'Item 2' in result
```


# LLM-generated content at query #88
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring('<span>Hello <b>World</b></span>')
    assert extract_text(dom) == 'Hello World'

    # Test with block elements
    dom = fragment_fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom) == 'First\nSecond'

    # Test with separator elements (br)
    dom = fragment_fromstring('<p>Line1<br/>Line2</p>')
    assert extract_text(dom) == 'Line1\nLine2'

    # Test with whitespace squashing
    dom = fragment_fromstring('<p>  Hello   World  </p>')
    assert extract_text(dom) == 'Hello World'

    # Test with nested inline elements
    dom = fragment_fromstring('<div><span>A <b>B</b> <i>C</i></span></div>')
    assert extract_text(dom) == 'A B C'

    # Test empty content
    dom = fragment_fromstring('<div></div>')
    assert extract_text(dom) == ''

    # Test with custom block_symbol and sep_symbol
    dom = fragment_fromstring('<div><p>First</p><br/><p>Second</p></div>')
    assert extract_text(dom, block_symbol='|', sep_symbol='|') == 'First|Second'

    # Test with squash_space=False
    dom = fragment_fromstring('<p>  Hello   World  </p>')
    assert extract_text(dom, squash_space=False) == '  Hello   World  '

    # Test with mixed inline and block
    dom = fragment_fromstring('<div>Text <p>Block</p> More text</div>')
    result = extract_text(dom)
    assert 'Text' in result
    assert 'Block' in result
    assert 'More text' in result
    assert '\n' in result

    # Test with script tag (should be treated as inline but usually empty)
    dom = fragment_fromstring('<div><script>var x = 1;</script>Content</div>')
    assert extract_text(dom) == 'Content'

    # Test with multiple br tags
    dom = fragment_fromstring('<p>A<br/><br/>B</p>')
    assert extract_text(dom) == 'A\n\nB'


# LLM-generated content at query #89
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline tag
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring('<span>Hello World</span>')
    assert extract_text(dom) == 'Hello World'
    
    # Test with block-level tag
    dom = fragment_fromstring('<div>Hello<div>World</div></div>')
    assert extract_text(dom) == 'Hello\nWorld'
    
    # Test with separator tag
    dom = fragment_fromstring('<div>Line1<br>Line2</div>')
    assert extract_text(dom) == 'Line1\nLine2'
    
    # Test with multiple whitespace
    dom = fragment_fromstring('<div>Hello   World</div>')
    assert extract_text(dom) == 'Hello World'
    
    # Test with nested inline tags
    dom = fragment_fromstring('<div><b>Hello</b> <i>World</i></div>')
    assert extract_text(dom) == 'Hello World'
    
    # Test with custom block_symbol
    dom = fragment_fromstring('<div>Hello<div>World</div></div>')
    assert extract_text(dom, block_symbol='|') == 'Hello|World'
    
    # Test with custom sep_symbol
    dom = fragment_fromstring('<div>Line1<br>Line2</div>')
    assert extract_text(dom, sep_symbol='|') == 'Line1|Line2'
    
    # Test with squash_space=False
    dom = fragment_fromstring('<div>Hello   World</div>')
    assert extract_text(dom, squash_space=False) == 'Hello   World'
    
    # Test with empty tag
    dom = fragment_fromstring('<div></div>')
    assert extract_text(dom) == ''
    
    # Test with text only
    dom = fragment_fromstring('<p>Simple text</p>')
    assert extract_text(dom) == 'Simple text'
    
    # Test with multiple block levels
    dom = fragment_fromstring('<div><p>Para1</p><p>Para2</p></div>')
    assert extract_text(dom) == 'Para1\nPara2'
    
    # Test with mixed inline and block
    dom = fragment_fromstring('<div><span>Inline</span><div>Block</div></div>')
    assert extract_text(dom) == 'Inline\nBlock'
```


# LLM-generated content at query #90
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test 2: Block elements with newlines
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test 3: Separator tags (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test 4: Nested elements
    doc = html.fromstring("<div><p>Text with <b>bold</b> and <i>italic</i></p></div>")
    assert extract_text(doc) == "Text with bold and italic"
    
    # Test 5: Whitespace handling
    doc = html.fromstring("<p>  Multiple   spaces   </p>")
    assert extract_text(doc) == "Multiple spaces"
    
    # Test 6: Empty content
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test 7: Multiple block elements
    doc = html.fromstring("<div><h1>Title</h1><p>Content</p></div>")
    assert extract_text(doc) == "Title\nContent"
    
    # Test 8: Deep nesting
    doc = html.fromstring("<div><div><p><span>Deep</span></p></div></div>")
    assert extract_text(doc) == "Deep"
    
    # Test 9: Mixed inline and block
    doc = html.fromstring("<p>Start <b>middle</b> end</p>")
    assert extract_text(doc) == "Start middle end"
    
    # Test 10: Multiple br tags
    doc = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(doc) == "Line1\n\nLine2"
    
    # Test 11: With custom block_symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=' ') == "First Second"
    
    # Test 12: With custom sep_symbol
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc, sep_symbol=' ') == "Line1 Line2"
    
    # Test 13: squash_space=False
    doc = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello   world  "


# LLM-generated content at query #91
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml.html import fromstring
    dom = fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"], f"Expected ['Hello'], got {result}"

    # Test with separator tag (br)
    dom = fromstring("<br>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"

    # Test with block tag
    dom = fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result == [None, "Text", None], f"Expected [None, 'Text', None], got {result}"

    # Test with nested inline tags
    dom = fromstring("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"], f"Expected ['Hello ', 'World'], got {result}"

    # Test with nested block tags
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, "Second", None, None], \
        f"Expected [None, None, 'First', None, None, 'Second', None, None], got {result}"

    # Test with separator inside inline
    dom = fromstring("<span>Line1<br>Line2</span>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"], f"Expected ['Line1', True, 'Line2'], got {result}"

    # Test with text after child (tail)
    dom = fromstring("<div>Before <span>inside</span> After</div>")
    result = extract_text_array(dom)
    assert result == [None, "Before ", "inside", " After", None], \
        f"Expected [None, 'Before ', 'inside', ' After', None], got {result}"

    # Test with empty tag
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None], f"Expected [None, None], got {result}"

    # Test with callable tag (comment)
    dom = fromstring("<!-- comment -->")
    result = extract_text_array(dom)
    assert result == [''], f"Expected [''], got {result}"

    # Test squash_artifical_nl=False
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result, f"Expected None in result, got {result}"

    # Test strip_artifical_nl=False
    dom = fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None, f"Expected first element to be None, got {result}"

    # Test with inline tag containing block children
    dom = fromstring("<span><div>Nested</div></span>")
    result = extract_text_array(dom)
    assert result == [None, "Nested", None], f"Expected [None, 'Nested', None], got {result}"


# LLM-generated content at query #92
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    doc = html.fromstring("<span>Hello World</span>")
    result = extract_text_array(doc)
    assert result == ["Hello World"]
    
    # Test with separator element (br)
    doc = html.fromstring("<br/>")
    result = extract_text_array(doc)
    assert result == [True]
    
    # Test with block element (div)
    doc = html.fromstring("<div>Content</div>")
    result = extract_text_array(doc)
    assert result == [None, "Content", None]
    
    # Test with nested inline elements
    doc = html.fromstring("<p><b>Bold</b> and <i>italic</i></p>")
    result = extract_text_array(doc)
    assert result == [None, "Bold", " and ", "italic", None]
    
    # Test with separator between text
    doc = html.fromstring("<p>Line 1<br/>Line 2</p>")
    result = extract_text_array(doc)
    assert result == [None, "Line 1", True, "Line 2", None]
    
    # Test with text and tail
    doc = html.fromstring("<p>Before <b>bold</b> after</p>")
    result = extract_text_array(doc)
    assert result == [None, "Before ", "bold", " after", None]
    
    # Test with nested block elements
    doc = html.fromstring("<div><p>Para 1</p><p>Para 2</p></div>")
    result = extract_text_array(doc)
    assert result == [None, None, "Para 1", None, None, "Para 2", None, None]
    
    # Test with squash_artifical_nl=False
    doc = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(doc, squash_artifical_nl=False)
    assert result == [None, None, "Text", None, None]
    
    # Test with strip_artifical_nl=False
    doc = html.fromstring("<div>Content</div>")
    result = extract_text_array(doc, strip_artifical_nl=False)
    assert result == [None, "Content", None]
    
    # Test with both flags False
    doc = html.fromstring("<p>Text</p>")
    result = extract_text_array(doc, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Text", None]
    
    # Test with empty element
    doc = html.fromstring("<div></div>")
    result = extract_text_array(doc)
    assert result == []
    
    # Test with custom element not in INLINE_TAGS
    doc = html.fromstring("<custom>Text</custom>")
    result = extract_text_array(doc)
    assert result == [None, "Text", None]


# LLM-generated content at query #93
#--------------------------

```python
def test_extract_text():
    # Test with simple text node
    from lxml import etree
    dom = etree.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with nested inline elements
    dom = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with block elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator elements (br)
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with custom block symbol
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test with custom separator symbol
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol='|') == "Line1|Line2"
    
    # Test with squash_space=False
    dom = etree.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test with nested non-inline elements
    dom = etree.fromstring("<div><p>Text</p><span>More</span></div>")
    result = extract_text(dom)
    assert "Text" in result
    assert "More" in result
    
    # Test with empty element
    dom = etree.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = etree.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test with multiple separators
    dom = etree.fromstring("<p>A<br/><br/>B</p>")
    assert extract_text(dom) == "A\nB"
    
    # Test with inline elements that have attributes
    dom = etree.fromstring('<p>Hello <a href="#">link</a> world</p>')
    assert extract_text(dom) == "Hello link world"


# LLM-generated content at query #94
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with block tag (div)
    dom = html.fromstring("<div>Hello World</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello World", None]
    
    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested inline tags
    dom = html.fromstring("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]
    
    # Test with nested block tags
    dom = html.fromstring("<div><p>Paragraph</p></div>")
    result = extract_text_array(dom)
    assert None in result and "Paragraph" in result
    
    # Test with text and tail
    dom = html.fromstring("<div>Text1<span>Inner</span>Text2</div>")
    result = extract_text_array(dom)
    assert "Text1" in result
    assert "Inner" in result
    assert "Text2" in result
    
    # Test with empty element
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with callable tag (should return empty string)
    class FakeTag:
        tag = lambda: None
    dom.tag = lambda: None
    result = extract_text_array(dom)
    assert result == ''


# LLM-generated content at query #95
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml.html import fromstring
    dom = fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element (div)
    dom = fromstring("<div>Hello</div><div>World</div>")
    assert extract_text(dom) == "Hello\nWorld"
    
    # Test with inline elements inside block
    dom = fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with separator (br)
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested block elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with whitespace handling
    dom = fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with leading/trailing whitespace
    dom = fromstring("<div>  <span>Hello</span>  </div>")
    assert extract_text(dom) == "Hello"
    
    # Test with empty element
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with mixed content
    dom = fromstring("<div>Text <span>inline</span> more text</div>")
    assert extract_text(dom) == "Text inline more text"
    
    # Test with multiple nested block elements
    dom = fromstring("<div><p>Para1</p><p>Para2</p><p>Para3</p></div>")
    assert extract_text(dom) == "Para1\nPara2\nPara3"
    
    # Test with custom block_symbol
    dom = fromstring("<div>Hello</div><div>World</div>")
    assert extract_text(dom, block_symbol=" | ") == "Hello | World"
    
    # Test with custom sep_symbol
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=" | ") == "Line1 | Line2"
    
    # Test with squash_space=False
    dom = fromstring("<p>  Hello   World  </p>")
    result = extract_text(dom, squash_space=False)
    assert "  " in result  # whitespace preserved
    
    # Test with script tag (should be ignored)
    dom = fromstring("<div>Hello<script>alert('test')</script>World</div>")
    assert extract_text(dom) == "HelloWorld"
    
    # Test with nested inline tags
    dom = fromstring("<p><strong><em>Bold and italic</em></strong></p>")
    assert extract_text(dom) == "Bold and italic"
    
    # Test with multiple separators
    dom = fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with complex nested structure
    html = """
    <div>
        <h1>Title</h1>
        <p>Paragraph with <strong>bold</strong> text</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </div>
    """
    dom = fromstring(html)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with bold text" in result
    assert "Item 1" in result
    assert "Item 2" in result
    assert result.count("\n") >= 3  # multiple newlines for block elements
    
    # Test that trailing whitespace is stripped
    dom = fromstring("<div>  Content  </div>")
    assert extract_text(dom) == "Content"
    assert not extract_text(dom).startswith(" ")
    assert not extract_text(dom).endswith(" ")```


# LLM-generated content at query #96
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    class MockElement:
        tag = 'p'
        text = 'Hello World'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['Hello World']
    
    # Test with separator tag (br)
    class MockBr:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockBr()
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with non-inline tag
    class MockDiv:
        tag = 'div'
        text = 'Content'
        def getchildren(self):
            return []
    dom = MockDiv()
    result = extract_text_array(dom)
    assert result == ['Content']  # None stripped from both ends
    
    # Test with inline tag
    class MockSpan:
        tag = 'span'
        text = 'inline'
        def getchildren(self):
            return []
    dom = MockSpan()
    result = extract_text_array(dom)
    assert result == ['inline']
    
    # Test with nested elements
    class MockChild:
        tag = 'b'
        text = 'bold'
        tail = ' tail'
        def getchildren(self):
            return []
    
    class MockParent:
        tag = 'p'
        text = 'Start '
        def getchildren(self):
            return [MockChild()]
    dom = MockParent()
    result = extract_text_array(dom)
    assert result == ['Start ', 'bold', ' tail']
    
    # Test with multiple children and separators
    class MockChild1:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []
    
    class MockChild2:
        tag = 'span'
        text = 'text'
        tail = None
        def getchildren(self):
            return []
    
    class MockParent2:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    dom = MockParent2()
    result = extract_text_array(dom)
    assert result == [True, 'text']
    
    # Test with squash_artifical_nl=False
    class MockBlock:
        tag = 'div'
        text = 'A'
        def getchildren(self):
            return []
    dom = MockBlock()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'A', None]
    
    # Test with strip_artifical_nl=False
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == ['A', None]
    
    # Test with callable tag (should return empty string)
    class MockCallable:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = MockCallable()
    result = extract_text_array(dom)
    assert result == []


# LLM-generated content at query #97
#--------------------------

```python
def test_extract_text():
    # Test with simple text
    dom = type('MockDOM', (), {'tag': 'p', 'text': 'Hello world', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(dom) == 'Hello world'

    # Test with separator tag (br)
    br_dom = type('MockDOM', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(br_dom) == '\n'

    # Test with inline tag
    span_dom = type('MockDOM', (), {'tag': 'span', 'text': 'inline', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(span_dom) == 'inline'

    # Test with non-inline tag (block element)
    div_dom = type('MockDOM', (), {'tag': 'div', 'text': 'Block', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(div_dom) == 'Block'

    # Test with children
    child_dom = type('MockDOM', (), {'tag': 'span', 'text': 'child', 'tail': ' tail', 'getchildren': lambda: []})()
    parent_dom = type('MockDOM', (), {'tag': 'div', 'text': 'Parent ', 'tail': None, 'getchildren': lambda: [child_dom]})()
    assert extract_text(parent_dom) == 'Parent child tail'

    # Test with separator in children
    br_child = type('MockDOM', (), {'tag': 'br', 'text': None, 'tail': ' after br', 'getchildren': lambda: []})()
    parent_with_br = type('MockDOM', (), {'tag': 'div', 'text': 'Before ', 'tail': None, 'getchildren': lambda: [br_child]})()
    assert extract_text(parent_with_br) == 'Before after br'

    # Test squash whitespace
    dom_with_whitespace = type('MockDOM', (), {'tag': 'p', 'text': 'Hello   world', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(dom_with_whitespace) == 'Hello world'

    # Test with nested non-inline tags
    inner_div = type('MockDOM', (), {'tag': 'div', 'text': 'Inner ', 'tail': ' after', 'getchildren': lambda: []})()
    outer_div = type('MockDOM', (), {'tag': 'div', 'text': 'Outer ', 'tail': None, 'getchildren': lambda: [inner_div]})()
    assert extract_text(outer_div) == 'Outer Inner after'

    # Test with custom symbols
    custom_dom = type('MockDOM', (), {'tag': 'div', 'text': 'A', 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(custom_dom, block_symbol=' | ') == 'A'

    # Test empty DOM
    empty_dom = type('MockDOM', (), {'tag': 'p', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    assert extract_text(empty_dom) == ''


# LLM-generated content at query #98
#--------------------------

```python
def test_extract_text():
    # Test simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test block element with newline
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test separator element (br)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test nested inline elements
    dom = html.fromstring("<p>This is <strong>important</strong> text</p>")
    assert extract_text(dom) == "This is important text"
    
    # Test with whitespace squashing
    dom = html.fromstring("<p>  Multiple   spaces   </p>")
    assert extract_text(dom) == "Multiple spaces"
    
    # Test empty element
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test element with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <a href="#">link</a> inside</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with link inside" in result
    assert "Item 1" in result
    assert "Item 2" in result
    assert result.count('\n') >= 3
    
    # Test custom block_symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test custom sep_symbol
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol='|') == "Line1|Line2"
    
    # Test squash_space=False
    dom = html.fromstring("<p>  Hello   World  </p>")
    result = extract_text(dom, squash_space=False)
    assert "  " in result  # Should preserve some whitespace
    
    # Test with attributes
    dom = html.fromstring('<div class="test"><p>Content</p></div>')
    assert extract_text(dom) == "Content"
    
    # Test with comments (should be ignored)
    dom = html.fromstring("<div><!-- comment --><p>Text</p></div>")
    assert extract_text(dom) == "Text"
    
    # Test inline tags that should not add newlines
    dom = html.fromstring("<div><span>First</span><span>Second</span></div>")
    assert extract_text(dom) == "FirstSecond"
    
    # Test multiple separators
    dom = html.fromstring("<p>A<br><br>B</p>")
    assert extract_text(dom) == "A\nB"
```


# LLM-generated content at query #99
#--------------------------

```python
def test_extract_text_array():
    # Test with None tag (callable)
    class MockCallableTag:
        tag = lambda: None
    assert extract_text_array(MockCallableTag()) == ''
    
    # Test with simple inline tag
    class MockInlineDom:
        tag = 'span'
        text = 'Hello'
        def getchildren(self):
            return []
    result = extract_text_array(MockInlineDom())
    assert result == ['Hello']
    
    # Test with separator tag (br)
    class MockSeparatorDom:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    result = extract_text_array(MockSeparatorDom())
    assert result == [True]
    
    # Test with block-level tag (div)
    class MockBlockDom:
        tag = 'div'
        text = 'Text'
        def getchildren(self):
            return []
    result = extract_text_array(MockBlockDom())
    assert result == [None, 'Text', None]
    
    # Test with children
    class MockChild:
        tag = 'span'
        text = 'child'
        tail = None
        def getchildren(self):
            return []
    
    class MockParent:
        tag = 'div'
        text = 'Parent '
        def getchildren(self):
            return [MockChild()]
    result = extract_text_array(MockParent(), squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'Parent ', 'child', None]
    
    # Test squash_artifical_nl
    result = extract_text_array(MockParent(), squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, 'Parent ', 'child', None]  # Already squashed
    
    # Test strip_artifical_nl
    result = extract_text_array(MockParent(), squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ['Parent ', 'child']
    
    # Test with tail text
    class MockChildWithTail:
        tag = 'b'
        text = 'bold'
        tail = ' tail'
        def getchildren(self):
            return []
    
    class MockParentWithTail:
        tag = 'p'
        text = 'Start '
        def getchildren(self):
            return [MockChildWithTail()]
    result = extract_text_array(MockParentWithTail(), squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'Start ', 'bold', ' tail', None]


# LLM-generated content at query #100
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]
    
    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with block tag (div)
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", None]
    
    # Test with nested structure
    dom = html.fromstring("<div><p>Text <b>bold</b> and <i>italic</i></p></div>")
    result = extract_text_array(dom)
    assert None in result
    assert "Text " in result
    assert "bold" in result
    assert " and " in result
    assert "italic" in result
    
    # Test with text and tail
    dom = html.fromstring("<div>Start <span>middle</span> end</div>")
    result = extract_text_array(dom)
    assert "Start " in result
    assert "middle" in result
    assert " end" in result
    
    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "Text", None, None]
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Hello", None]
    
    # Test with callable tag (comment)
    from lxml import etree
    comment = etree.Comment("test")
    assert extract_text_array(comment) == ''
    
    # Test empty content
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with inline tags that should not get artificial newlines
    dom = html.fromstring("<a>Link</a>")
    result = extract_text_array(dom)
    assert result == ["Link"]
    
    # Test multiple br tags
    dom = html.fromstring("<br/><br/>")
    result = extract_text_array(dom)
    assert result == [True, True]


# LLM-generated content at query #101
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple text extraction
    from lxml import etree
    html = "<p>Hello world</p>"
    dom = etree.fromstring(html)
    assert extract_text(dom) == "Hello world"
    
    # Test 2: Text with nested inline elements
    html = "<p>Hello <b>bold</b> world</p>"
    dom = etree.fromstring(html)
    assert extract_text(dom) == "Hello bold world"
    
    # Test 3: Text with separator elements (br)
    html = "<p>Line1<br/>Line2</p>"
    dom = etree.fromstring(html)
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test 4: Text with block elements
    html = "<div><p>First</p><p>Second</p></div>"
    dom = etree.fromstring(html)
    assert extract_text(dom) == "First\nSecond"
    
    # Test 5: Custom block and separator symbols
    html = "<div><p>First</p><br/><p>Second</p></div>"
    dom = etree.fromstring(html)
    assert extract_text(dom, block_symbol='|', sep_symbol='-') == "First-Second"
    
    # Test 6: Nested structure
    html = "<div><p><b>Bold</b> and <i>italic</i></p></div>"
    dom = etree.fromstring(html)
    assert extract_text(dom) == "Bold and italic"
    
    # Test 7: Multiple separators
    html = "<p>A<br/><br/>B</p>"
    dom = etree.fromstring(html)
    assert extract_text(dom) == "A\nB"
    
    # Test 8: Text with whitespace
    html = "<p>  Hello   world  </p>"
    dom = etree.fromstring(html)
    assert extract_text(dom) == "Hello world"
    
    # Test 9: Empty element
    html = "<p></p>"
    dom = etree.fromstring(html)
    assert extract_text(dom) == ""
    
    # Test 10: Text with tail content
    html = "<div><p>Para</p>Tail</div>"
    dom = etree.fromstring(html)
    assert extract_text(dom) == "Para\nTail"
    
    # Test 11: Complex nested structure with separators
    html = "<div><p>First line</p><br/><p>Second <b>bold</b> line</p></div>"
    dom = etree.fromstring(html)
    assert extract_text(dom) == "First line\nSecond bold line"
    
    # Test 12: Deep nesting
    html = "<div><div><p>Deep</p></div></div>"
    dom = etree.fromstring(html)
    assert extract_text(dom) == "Deep"


# LLM-generated content at query #102
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"], f"Expected ['Hello'], got {result}"
    
    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"
    
    # Test with block element (div)
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result == [None, "Text", None], f"Expected [None, 'Text', None], got {result}"
    
    # Test with nested inline elements
    dom = html.fromstring("<p><b>Bold</b> and <i>italic</i></p>")
    result = extract_text_array(dom)
    assert result == [None, "Bold", " and ", "italic", None], f"Expected [None, 'Bold', ' and ', 'italic', None], got {result}"
    
    # Test with separator inside text
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None], f"Expected [None, 'Line1', True, 'Line2', None], got {result}"
    
    # Test with empty element
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None], f"Expected [None, None], got {result}"
    
    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "Text", None, None], f"Expected [None, None, 'Text', None, None], got {result}"
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Text", None], f"Expected [None, 'Text', None], got {result}"
    
    # Test with both squash and strip disabled
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Text", None], f"Expected [None, 'Text', None], got {result}"
    
    # Test with callable tag (should return empty string)
    class MockElement:
        tag = lambda: None
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == '', f"Expected empty string, got {result}"
    
    # Test with multiple children
    dom = html.fromstring("<div><span>A</span><span>B</span></div>")
    result = extract_text_array(dom)
    assert result == [None, "A", "B", None], f"Expected [None, 'A', 'B', None], got {result}"
    
    # Test with text and tail
    dom = html.fromstring("<div>Start<b>bold</b>End</div>")
    result = extract_text_array(dom)
    assert result == [None, "Start", "bold", "End", None], f"Expected [None, 'Start', 'bold', 'End', None], got {result}"


# LLM-generated content at query #103
#--------------------------

```python
def test_extract_text():
    # Test with simple text content
    from lxml.html import fromstring
    dom = fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with inline elements
    dom = fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator elements (br)
    dom = fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested elements
    dom = fromstring("<div><p>Hello <b>beautiful</b></p><p>World</p></div>")
    assert extract_text(dom) == "Hello beautiful\nWorld"
    
    # Test with custom block_symbol
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    dom = fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    dom = fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test with empty content
    dom = fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with text only child
    dom = fromstring("<div>Text only</div>")
    assert extract_text(dom) == "Text only"
    
    # Test with multiple inline elements
    dom = fromstring("<p><span>Hello</span> <span>World</span></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with deep nesting
    dom = fromstring("<div><div><p>Deep</p></div></div>")
    assert extract_text(dom) == "Deep"


# LLM-generated content at query #104
#--------------------------

```python
def test_extract_text():
    # Test with simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element
    dom = html.fromstring("<div>Hello</div>")
    assert extract_text(dom) == "Hello"
    
    # Test with nested elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator element (br)
    dom = html.fromstring("<span>Line1<br>Line2</span>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with multiple br elements
    dom = html.fromstring("<span>A<br><br>B</span>")
    assert extract_text(dom) == "A\n\nB"
    
    # Test with inline elements inside block
    dom = html.fromstring("<p>This is <strong>bold</strong> text</p>")
    assert extract_text(dom) == "This is bold text"
    
    # Test with whitespace squashing
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with nested blocks
    dom = html.fromstring("<div><p>First</p><div><p>Second</p></div></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test empty element
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test element with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test with custom symbols
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' | ', sep_symbol=' - ') == "First | Second"
    
    # Test with br separator using custom symbol
    dom = html.fromstring("<span>A<br>B</span>")
    assert extract_text(dom, sep_symbol=' - ') == "A - B"
    
    # Test with mixed inline and block elements
    dom = html.fromstring("<div><span>inline</span><p>block</p></div>")
    assert extract_text(dom) == "inline\nblock"
    
    # Test with nested inline elements
    dom = html.fromstring("<p><span><em>nested</em></span> text</p>")
    assert extract_text(dom) == "nested text"
    
    # Test with multiple levels of nesting
    dom = html.fromstring("<div><section><article><p>Deep content</p></article></section></div>")
    assert extract_text(dom) == "Deep content"


# LLM-generated content at query #105
#--------------------------

```python
def test_extract_text():
    # Test simple text element
    from lxml.html import fromstring
    html = "<p>Hello World</p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello World"
    
    # Test with nested inline elements
    html = "<p>Hello <b>World</b></p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello World"
    
    # Test with block elements (should add newlines)
    html = "<div><p>First</p><p>Second</p></div>"
    dom = fromstring(html)
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator (br tag)
    html = "<p>Line1<br>Line2</p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with whitespace normalization
    html = "<p>Hello    World</p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Hello World"
    
    # Test with empty elements
    html = "<p></p>"
    dom = fromstring(html)
    assert extract_text(dom) == ""
    
    # Test with multiple nested levels
    html = "<div><p>Text <span>inside</span></p></div>"
    dom = fromstring(html)
    assert extract_text(dom) == "Text inside"
    
    # Test with tail text
    html = "<p><b>Bold</b> and normal</p>"
    dom = fromstring(html)
    assert extract_text(dom) == "Bold and normal"


# LLM-generated content at query #106
#--------------------------

```python
def test_extract_text():
    # Test with simple text
    dom = type('Node', (), {'tag': 'p', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == 'Hello'
    
    # Test with separator tag
    br = type('Node', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Node', (), {
        'tag': 'p', 'text': 'Line1', 'tail': None,
        'getchildren': lambda self: [br]
    })()
    result = extract_text(dom)
    assert result == 'Line1\n'  # br adds newline separator
    
    # Test with inline tag
    span = type('Node', (), {'tag': 'span', 'text': 'world', 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Node', (), {
        'tag': 'p', 'text': 'Hello ', 'tail': None,
        'getchildren': lambda self: [span]
    })()
    assert extract_text(dom) == 'Hello world'
    
    # Test with block-level tags creating artificial newlines
    div1 = type('Node', (), {'tag': 'div', 'text': 'First', 'tail': None, 'getchildren': lambda self: []})()
    div2 = type('Node', (), {'tag': 'div', 'text': 'Second', 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Node', (), {
        'tag': 'body', 'text': None, 'tail': None,
        'getchildren': lambda self: [div1, div2]
    })()
    assert extract_text(dom) == 'First\nSecond'
    
    # Test with nested structure
    inner = type('Node', (), {'tag': 'span', 'text': 'inner', 'tail': None, 'getchildren': lambda self: []})()
    outer = type('Node', (), {
        'tag': 'div', 'text': 'Outer ', 'tail': None,
        'getchildren': lambda self: [inner]
    })()
    dom = type('Node', (), {
        'tag': 'body', 'text': None, 'tail': None,
        'getchildren': lambda self: [outer]
    })()
    assert extract_text(dom) == 'Outer inner'
    
    # Test with whitespace squashing
    dom = type('Node', (), {'tag': 'p', 'text': 'Hello   world', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == 'Hello world'
    
    # Test with tail text
    child = type('Node', (), {'tag': 'span', 'text': 'child', 'tail': ' tail', 'getchildren': lambda self: []})()
    dom = type('Node', (), {
        'tag': 'p', 'text': 'Parent ', 'tail': None,
        'getchildren': lambda self: [child]
    })()
    assert extract_text(dom) == 'Parent child tail'
    
    # Test empty document
    dom = type('Node', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == ''
    
    # Test with custom block_symbol
    dom = type('Node', (), {'tag': 'div', 'text': 'First', 'tail': None, 'getchildren': lambda self: []})()
    dom2 = type('Node', (), {'tag': 'div', 'text': 'Second', 'tail': None, 'getchildren': lambda self: []})()
    parent = type('Node', (), {
        'tag': 'body', 'text': None, 'tail': None,
        'getchildren': lambda self: [dom, dom2]
    })()
    assert extract_text(parent, block_symbol=' | ') == 'First | Second'
    
    # Test with custom sep_symbol
    br = type('Node', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Node', (), {
        'tag': 'p', 'text': 'A', 'tail': None,
        'getchildren': lambda self: [br]
    })()
    assert extract_text(dom, sep_symbol=' --- ') == 'A --- '
    
    # Test with squash_space=False
    dom = type('Node', (), {'tag': 'p', 'text': '  Hello  ', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom, squash_space=False) == '  Hello  '


# LLM-generated content at query #107
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with block elements creating newlines
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with separator elements (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with nested elements
    doc = html.fromstring("<div><span>Nested <b>bold</b> text</span></div>")
    assert extract_text(doc) == "Nested bold text"
    
    # Test with multiple spaces collapsed
    doc = html.fromstring("<p>Hello    world</p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with whitespace-only content
    doc = html.fromstring("<p>   </p>")
    assert extract_text(doc) == ""
    
    # Test with empty elements
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with custom block symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=" | ") == "First | Second"
    
    # Test with custom separator symbol
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc, sep_symbol=" - ") == "Line1 - Line2"
    
    # Test without squashing whitespace
    doc = html.fromstring("<p>Hello    world</p>")
    assert extract_text(doc, squash_space=False) == "Hello    world"
    
    # Test deep nesting
    doc = html.fromstring("<div><ul><li>Item 1</li><li>Item 2</li></ul></div>")
    result = extract_text(doc)
    assert "Item 1" in result
    assert "Item 2" in result
    assert result.count("\n") >= 1


# LLM-generated content at query #108
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction from simple element
    from lxml import etree
    dom = etree.HTML("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with inline elements
    dom = etree.HTML("<p>Hello <b>bold</b> World</p>")
    assert extract_text(dom) == "Hello bold World"
    
    # Test with block elements
    dom = etree.HTML("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separators (br)
    dom = etree.HTML("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested elements
    dom = etree.HTML("<div><span><b>Nested</b></span> text</div>")
    assert extract_text(dom) == "Nested text"
    
    # Test whitespace handling
    dom = etree.HTML("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test empty content
    dom = etree.HTML("<div></div>")
    assert extract_text(dom) == ""
    
    # Test custom block_symbol
    dom = etree.HTML("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' ') == "First Second"
    
    # Test custom sep_symbol
    dom = etree.HTML("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    dom = etree.HTML("<p>  Hello   World  </p>")
    result = extract_text(dom, squash_space=False)
    assert "  " in result or "   " in result  # Should preserve whitespace
    
    # Test complex nested structure
    dom = etree.HTML("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> text</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph" in result
    assert "bold" in result
    assert "Item 1" in result
    assert "Item 2" in result
    assert result.count('\n') >= 3  # Multiple newlines for block elements
    
    # Test with non-HTML tags
    dom = etree.fromstring("<custom>Test content</custom>")
    assert extract_text(dom) == "Test content"
    
    # Test with mixed inline and block
    dom = etree.HTML("<div>Text <span>inline</span> <p>block</p> more</div>")
    result = extract_text(dom)
    assert "Text" in result
    assert "inline" in result
    assert "block" in result
    assert "more" in result
    
    # Test leading/trailing whitespace removal
    dom = etree.HTML("<p>  Leading and trailing  </p>")
    assert extract_text(dom) == "Leading and trailing"
    
    # Test multiple consecutive block elements
    dom = etree.HTML("<div><p>First</p><p></p><p>Third</p></div>")
    assert extract_text(dom) == "First\n\nThird"


# LLM-generated content at query #109
#--------------------------

```python
def test_extract_text():
    # Test basic inline tags
    from lxml.html import fromstring
    
    # Test simple text
    dom = fromstring('<p>Hello World</p>')
    assert extract_text(dom) == 'Hello World'
    
    # Test with nested inline tags
    dom = fromstring('<p>Hello <b>bold</b> World</p>')
    assert extract_text(dom) == 'Hello bold World'
    
    # Test with separator tags (br)
    dom = fromstring('<p>Line1<br>Line2</p>')
    assert extract_text(dom) == 'Line1\nLine2'
    
    # Test multiple separators
    dom = fromstring('<p>Line1<br><br>Line2</p>')
    assert extract_text(dom) == 'Line1\n\nLine2'
    
    # Test block elements
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom) == 'First\nSecond'
    
    # Test nested block elements
    dom = fromstring('<div><div><p>Deep</p></div></div>')
    assert extract_text(dom) == 'Deep'
    
    # Test mixed content
    dom = fromstring('<div>Text before <p>Paragraph</p> Text after</div>')
    assert extract_text(dom) == 'Text before\nParagraph\nText after'
    
    # Test with whitespace
    dom = fromstring('<p>  Hello   World  </p>')
    assert extract_text(dom) == 'Hello World'
    
    # Test with multiple whitespace characters
    dom = fromstring('<p>Hello\t\tWorld\n\nTest</p>')
    assert extract_text(dom) == 'Hello World Test'
    
    # Test empty content
    dom = fromstring('<p></p>')
    assert extract_text(dom) == ''
    
    # Test with only whitespace
    dom = fromstring('<p>   </p>')
    assert extract_text(dom) == ''
    
    # Test with attributes (should not affect text)
    dom = fromstring('<p class="test">Hello</p>')
    assert extract_text(dom) == 'Hello'
    
    # Test with multiple block elements
    dom = fromstring('<div><h1>Title</h1><p>Content</p></div>')
    assert extract_text(dom) == 'Title\nContent'
    
    # Test with nested inline inside block
    dom = fromstring('<p>Hello <span>world</span>!</p>')
    assert extract_text(dom) == 'Hello world!'
    
    # Test with script tag (should be treated as inline)
    dom = fromstring('<div><script>alert("test")</script><p>Text</p></div>')
    result = extract_text(dom)
    assert 'alert' not in result
    assert 'Text' in result
    
    # Test with custom block_symbol
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    assert extract_text(dom, block_symbol=' ') == 'First Second'
    
    # Test with custom sep_symbol
    dom = fromstring('<p>Line1<br>Line2</p>')
    assert extract_text(dom, sep_symbol=' ') == 'Line1 Line2'
    
    # Test without squashing space
    dom = fromstring('<p>  Hello   World  </p>')
    assert extract_text(dom, squash_space=False) == '  Hello   World  '
    
    # Test with complex nested structure
    html = '''
    <div>
        <h1>Title</h1>
        <p>First paragraph with <b>bold</b> text</p>
        <p>Second paragraph<br>with line break</p>
    </div>
    '''
    dom = fromstring(html)
    result = extract_text(dom)
    assert 'Title' in result
    assert 'First paragraph with bold text' in result
    assert 'Second paragraph' in result
    assert 'with line break' in result
    
    # Test with empty tags
    dom = fromstring('<div><p></p><p>Content</p><p></p></div>')
    assert extract_text(dom) == 'Content'
    
    # Test with only separator tags
    dom = fromstring('<div><br></div>')
    assert extract_text(dom) == ''
    
    # Test with mixed inline and block tags
    dom = fromstring('<span>Inline</span><div>Block</div>')
    assert extract_text(dom) == 'Inline\nBlock'


# LLM-generated content at query #110
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []
        
        def getchildren(self):
            return self._children
    
    # Test 1: Simple text
    dom = MockElement('span', text='Hello World')
    assert extract_text(dom) == 'Hello World'
    
    # Test 2: Element with separator (br)
    dom = MockElement('div', children=[
        MockElement('span', text='Line 1'),
        MockElement('br', text=None),
        MockElement('span', text='Line 2')
    ])
    assert extract_text(dom) == 'Line 1\nLine 2'
    
    # Test 3: Nested inline elements
    dom = MockElement('p', children=[
        MockElement('strong', text='Bold'),
        MockElement('em', text='Italic')
    ])
    assert extract_text(dom) == 'BoldItalic'
    
    # Test 4: Block elements create newlines
    dom = MockElement('div', children=[
        MockElement('p', text='Paragraph 1'),
        MockElement('p', text='Paragraph 2')
    ])
    result = extract_text(dom)
    assert result == 'Paragraph 1\nParagraph 2'
    
    # Test 5: Squash whitespace
    dom = MockElement('span', text='Hello   World')
    assert extract_text(dom) == 'Hello World'
    
    # Test 6: Mixed content with tails
    dom = MockElement('p', children=[
        MockElement('a', text='Link', tail=' and more text')
    ])
    assert extract_text(dom) == 'Link and more text'
    
    # Test 7: Empty element
    dom = MockElement('div')
    assert extract_text(dom) == ''
    
    # Test 8: Code element (inline)
    dom = MockElement('code', text='print("hello")')
    assert extract_text(dom) == 'print("hello")'
    
    # Test 9: Multiple block elements with separators
    dom = MockElement('div', children=[
        MockElement('p', text='First'),
        MockElement('br'),
        MockElement('p', text='Second')
    ])
    result = extract_text(dom)
    assert result == 'First\nSecond'
    
    # Test 10: Deeply nested structure
    dom = MockElement('div', children=[
        MockElement('div', children=[
            MockElement('span', text='Nested')
        ])
    ])
    assert extract_text(dom) == 'Nested'
    
    # Test 11: Whitespace only content
    dom = MockElement('span', text='   ')
    assert extract_text(dom) == ''
    
    # Test 12: Multiple whitespace between elements
    dom = MockElement('p', children=[
        MockElement('span', text='Hello', tail='   '),
        MockElement('span', text='World')
    ])
    assert extract_text(dom) == 'Hello World'


# LLM-generated content at query #111
#--------------------------

```python
def test_extract_text():
    # Test with simple paragraph
    p = etree.fromstring("<p>Hello world</p>")
    assert extract_text(p) == "Hello world"
    
    # Test with inline tags
    p = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(p) == "Hello bold world"
    
    # Test with block tags (should add newlines)
    div = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(div) == "First\nSecond"
    
    # Test with separator tags (br)
    p = etree.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(p) == "Line1\nLine2"
    
    # Test with nested tags
    div = etree.fromstring("<div><p>Text with <span>span</span> inside</p></div>")
    assert extract_text(div) == "Text with span inside"
    
    # Test with multiple spaces
    p = etree.fromstring("<p>Hello    world</p>")
    assert extract_text(p) == "Hello world"
    
    # Test with whitespace and newlines
    p = etree.fromstring("<p>\n  Hello\n  world\n</p>")
    assert extract_text(p) == "Hello world"
    
    # Test with empty content
    p = etree.fromstring("<p></p>")
    assert extract_text(p) == ""
    
    # Test with nested separators
    div = etree.fromstring("<div><br/><br/></div>")
    assert extract_text(div) == "\n"
    
    # Test with custom block_symbol
    p = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(p, block_symbol=" | ") == "First | Second"
    
    # Test with custom sep_symbol
    p = etree.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(p, sep_symbol=" / ") == "Line1 / Line2"
    
    # Test with squash_space=False
    p = etree.fromstring("<p>Hello    world</p>")
    assert extract_text(p, squash_space=False) == "Hello    world"
    
    # Test with mixed inline and block elements
    div = etree.fromstring("<div><p>Para1</p>Some text<p>Para2</p></div>")
    assert extract_text(div) == "Para1\nSome text\nPara2"
    
    # Test with deeply nested structure
    div = etree.fromstring("<div><p><b><i>Nested</i></b></p></div>")
    assert extract_text(div) == "Nested"
    
    # Test with script tag (should be treated as inline)
    div = etree.fromstring("<div><script>alert('test')</script><p>Text</p></div>")
    assert extract_text(div) == "alert('test')Text"
    
    # Test with leading/trailing whitespace
    p = etree.fromstring("<p>  Hello  </p>")
    assert extract_text(p) == "Hello"
    
    # Test with multiple block elements
    div = etree.fromstring("<div><p>First</p><p>Second</p><p>Third</p></div>")
    assert extract_text(div) == "First\nSecond\nThird"
    
    # Test with None text in element
    p = etree.fromstring("<p><b></b>text</p>")
    assert extract_text(p) == "text"


# LLM-generated content at query #112
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"], f"Expected ['Hello World'], got {result}"

    # Test with a block-level tag
    dom = html.fromstring("<div>Hello World</div>")
    result = extract_text_array(dom)
    assert result == ["Hello World"], f"Expected ['Hello World'], got {result}"

    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"

    # Test with nested inline tags
    dom = html.fromstring("<span><b>Bold</b> and <i>italic</i></span>")
    result = extract_text_array(dom)
    assert result == ["Bold", " and ", "italic"], f"Expected ['Bold', ' and ', 'italic'], got {result}"

    # Test with block-level tag containing inline tags
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"], f"Expected ['First', 'Second'], got {result}"

    # Test with separator (br) between text
    dom = html.fromstring("Line1<br/>Line2")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"], f"Expected ['Line1', True, 'Line2'], got {result}"

    # Test with artificial newlines (None) for block tags
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Text", None], f"Expected [None, 'Text', None], got {result}"

    # Test squash artificial newlines
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, "A", None, "B"], f"Expected [None, 'A', None, 'B'], got {result}"

    # Test with nested block elements and strip artificial newlines
    dom = html.fromstring("<div><div><p>Content</p></div></div>")
    result = extract_text_array(dom)
    assert result == ["Content"], f"Expected ['Content'], got {result}"

    # Test with tail text
    dom = html.fromstring("<div><b>Bold</b> tail</div>")
    result = extract_text_array(dom)
    assert result == ["Bold", " tail"], f"Expected ['Bold', ' tail'], got {result}"

    # Test with multiple separators
    dom = html.fromstring("<br/><br/>")
    result = extract_text_array(dom)
    assert result == [True, True], f"Expected [True, True], got {result}"

    # Test empty element
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [], f"Expected [], got {result}"

    # Test callable tag (like comment)
    from lxml.html import HtmlElement
    comment = html.fromstring("<!-- comment -->")
    if hasattr(comment, 'tag') and callable(comment.tag):
        result = extract_text_array(comment)
        assert result == [], f"Expected [], got {result}"


# LLM-generated content at query #113
#--------------------------

```python
def test_extract_text():
    # Test simple inline element
    from lxml.html import fromstring
    dom = fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test block element with text
    dom = fromstring("<div>Hello World</div>")
    assert extract_text(dom) == "Hello World"
    
    # Test nested elements
    dom = fromstring("<div><p>First paragraph</p><p>Second paragraph</p></div>")
    assert extract_text(dom) == "First paragraph\nSecond paragraph"
    
    # Test separator element (br)
    dom = fromstring("<div>Line1<br>Line2</div>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test custom block symbol
    dom = fromstring("<div><p>Para1</p><p>Para2</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "Para1 | Para2"
    
    # Test custom separator symbol
    dom = fromstring("<div>Line1<br>Line2</div>")
    assert extract_text(dom, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test squash_space=False
    dom = fromstring("<div>  Hello   World  </div>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test multiple nested block elements
    dom = fromstring("<body><h1>Title</h1><p>Content</p></body>")
    assert extract_text(dom) == "Title\nContent"
    
    # Test empty element
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test element with only whitespace
    dom = fromstring("<div>   </div>")
    assert extract_text(dom) == ""
    
    # Test inline elements inside block elements
    dom = fromstring("<p>Hello <strong>World</strong></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test multiple separators
    dom = fromstring("<div>A<br><br>B</div>")
    assert extract_text(dom) == "A\n\nB"


# LLM-generated content at query #114
#--------------------------

```python
def test_extract_text_array():
    # Test with None tag (callable)
    mock_dom_callable = type('Mock', (), {'tag': lambda: None, 'text': None})()
    assert extract_text_array(mock_dom_callable) == ''
    
    # Test with separator tag
    mock_dom_br = type('Mock', (), {'tag': 'br', 'text': None, 'getchildren': lambda: []})()
    assert extract_text_array(mock_dom_br) == [True]
    
    # Test with inline tag
    mock_dom_span = type('Mock', (), {'tag': 'span', 'text': 'hello', 'getchildren': lambda: []})()
    assert extract_text_array(mock_dom_span) == ['hello']
    
    # Test with block tag
    mock_dom_div = type('Mock', (), {'tag': 'div', 'text': 'hello', 'getchildren': lambda: []})()
    result = extract_text_array(mock_dom_div)
    assert result[0] is None
    assert result[1] == 'hello'
    assert result[2] is None
    
    # Test with nested structure
    child = type('Mock', (), {'tag': 'span', 'text': 'world', 'tail': None, 'getchildren': lambda: []})()
    parent = type('Mock', (), {'tag': 'div', 'text': 'hello ', 'getchildren': lambda: [child], 'tail': None})()
    result = extract_text_array(parent)
    assert result[0] is None
    assert result[1] == 'hello '
    assert result[2] == 'world'
    assert result[3] is None
    
    # Test with squash_artifical_nl=False
    result = extract_text_array(mock_dom_div, squash_artifical_nl=False)
    assert len(result) == 3
    assert result[0] is None
    assert result[1] == 'hello'
    assert result[2] is None
    
    # Test with strip_artifical_nl=False
    result = extract_text_array(mock_dom_div, strip_artifical_nl=False)
    assert len(result) == 3
    assert result[0] is None
    assert result[1] == 'hello'
    assert result[2] is None
    
    # Test with both squash and strip disabled
    result = extract_text_array(mock_dom_div, squash_artifical_nl=False, strip_artifical_nl=False)
    assert len(result) == 3
    assert result[0] is None
    assert result[1] == 'hello'
    assert result[2] is None
    
    # Test with separator in nested structure
    child_br = type('Mock', (), {'tag': 'br', 'text': None, 'tail': 'after', 'getchildren': lambda: []})()
    parent_br = type('Mock', (), {'tag': 'div', 'text': 'before', 'getchildren': lambda: [child_br], 'tail': None})()
    result = extract_text_array(parent_br)
    assert result[0] is None
    assert result[1] == 'before'
    assert result[2] is True
    assert result[3] == 'after'
    assert result[4] is None
    
    # Test with multiple children
    child1 = type('Mock', (), {'tag': 'span', 'text': 'first', 'tail': ' ', 'getchildren': lambda: []})()
    child2 = type('Mock', (), {'tag': 'span', 'text': 'second', 'tail': None, 'getchildren': lambda: []})()
    parent_multi = type('Mock', (), {'tag': 'div', 'text': None, 'getchildren': lambda: [child1, child2], 'tail': None})()
    result = extract_text_array(parent_multi)
    assert result[0] is None
    assert result[1] == 'first'
    assert result[2] == ' '
    assert result[3] == 'second'
    assert result[4] is None
    
    # Test with empty text
    mock_dom_empty = type('Mock', (), {'tag': 'div', 'text': '', 'getchildren': lambda: []})()
    result = extract_text_array(mock_dom_empty)
    assert result[0] is None
    assert result[1] == ''
    assert result[2] is None


# LLM-generated content at query #115
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    from lxml import html
    doc = html.fromstring("<p>Hello World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with nested inline elements
    doc = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(doc) == "Hello bold world"
    
    # Test with block elements (should add newlines)
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with separator elements (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with nested block elements
    doc = html.fromstring("<div><p>Text</p></div>")
    assert extract_text(doc) == "Text"
    
    # Test whitespace squashing
    doc = html.fromstring("<p>Hello    World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with leading/trailing whitespace
    doc = html.fromstring("<p>  Hello World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with multiple whitespace characters (tabs, newlines)
    doc = html.fromstring("<p>Hello\t\nWorld</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test empty document
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with only text node
    doc = html.fromstring("Just text")
    assert extract_text(doc) == "Just text"
    
    # Test with attributes (should not affect text)
    doc = html.fromstring('<p class="test">Hello</p>')
    assert extract_text(doc) == "Hello"
    
    # Test custom block_symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=" | ") == "First | Second"
    
    # Test custom sep_symbol
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc, sep_symbol=" | ") == "Line1 | Line2"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello  World  </p>")
    result = extract_text(doc, squash_space=False)
    assert result == "  Hello  World  "
    
    # Test complex nested structure
    doc = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>First <b>paragraph</b></p>
            <p>Second paragraph<br>with break</p>
        </div>
    """)
    result = extract_text(doc)
    assert "Title" in result
    assert "First paragraph" in result
    assert "Second paragraph" in result
    assert "with break" in result
    
    # Test with script tags (should be excluded)
    doc = html.fromstring("<div>Text<script>alert('test');</script>More</div>")
    result = extract_text(doc)
    assert "alert" not in result
    assert "Text" in result
    assert "More" in result
    
    # Test with list elements
    doc = html.fromstring("<ul><li>Item 1</li><li>Item 2</li></ul>")
    result = extract_text(doc)
    assert "Item 1" in result
    assert "Item 2" in result
    assert result.count("\n") >= 1
    
    # Test with deeply nested structure
    doc = html.fromstring("<div><p><span><b>Deep</b> text</span></p></div>")
    assert extract_text(doc) == "Deep text"
```


# LLM-generated content at query #116
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test 2: Block tag with single child
    dom = html.fromstring("<div><p>Paragraph</p></div>")
    assert extract_text(dom) == "Paragraph"
    
    # Test 3: Multiple block tags
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test 4: Inline tags within block
    dom = html.fromstring("<p>This is <b>bold</b> text</p>")
    assert extract_text(dom) == "This is bold text"
    
    # Test 5: Nested inline tags
    dom = html.fromstring("<p><span>Nested <i>italic</i> text</span></p>")
    assert extract_text(dom) == "Nested italic text"
    
    # Test 6: Separator tag (br)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test 7: Multiple br tags
    dom = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(dom) == "Line1\n\nLine2"
    
    # Test 8: Mixed inline and block
    dom = html.fromstring("<div><h1>Title</h1><p>Paragraph with <a>link</a></p></div>")
    assert extract_text(dom) == "Title\nParagraph with link"
    
    # Test 9: Whitespace handling
    dom = html.fromstring("<p>  Spaces   around  </p>")
    assert extract_text(dom) == "Spaces around"
    
    # Test 10: Newlines in HTML
    dom = html.fromstring("<p>\n  Line1\n  Line2\n</p>")
    assert extract_text(dom) == "Line1 Line2"
    
    # Test 11: Empty content
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test 12: Empty with nested
    dom = html.fromstring("<div><p></p></div>")
    assert extract_text(dom) == ""
    
    # Test 13: Multiple block tags with inline
    dom = html.fromstring("<div><p>Para <b>1</b></p><p>Para 2</p></div>")
    assert extract_text(dom) == "Para 1\nPara 2"
    
    # Test 14: Custom block_symbol parameter
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "First | Second"
    
    # Test 15: Custom sep_symbol parameter
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test 16: squash_space=False
    dom = html.fromstring("<p>  Spaces   around  </p>")
    result = extract_text(dom, squash_space=False)
    assert "  " in result or "   " in result  # Should preserve some whitespace
    
    # Test 17: Nested block elements
    dom = html.fromstring("<div><section><p>Deep</p></section></div>")
    assert extract_text(dom) == "Deep"
    
    # Test 18: Multiple levels of nesting
    dom = html.fromstring("<div><ul><li>Item1</li><li>Item2</li></ul></div>")
    assert extract_text(dom) == "Item1\nItem2"
    
    # Test 19: Text in root element with children
    dom = html.fromstring("<p>Start <b>middle</b> end</p>")
    assert extract_text(dom) == "Start middle end"
    
    # Test 20: Multiple text nodes
    dom = html.fromstring("<p>Text1<b>Bold</b>Text2<i>Italic</i>Text3</p>")
    assert extract_text(dom) == "Text1BoldText2ItalicText3"
```


# LLM-generated content at query #117
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text node
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]

    # Test with inline tag
    dom = fragment_fromstring("<span>inline</span>")
    result = extract_text_array(dom)
    assert result == ["inline"]

    # Test with separator tag (br)
    dom = fragment_fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]

    # Test with nested inline tags
    dom = fragment_fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "bold", " world"]

    # Test with multiple block-level tags
    dom = fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, "First", None, None, "Second", None]

    # Test with br separator between text
    dom = fragment_fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]

    # Test with empty element
    dom = fragment_fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]

    # Test with squashed artificial newlines
    dom = fragment_fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == [None, "A", None, "B", None]

    # Test with stripped artificial newlines
    dom = fragment_fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["A"]

    # Test with both squash and strip
    dom = fragment_fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["A", None, "B"]


# LLM-generated content at query #118
#--------------------------

```python
def test_extract_text():
    # Test with inline elements (should not add newlines)
    from xml.etree.ElementTree import Element, SubElement, tostring
    from io import StringIO
    
    # Test 1: Simple text
    dom = Element('p')
    dom.text = 'Hello World'
    assert extract_text(dom) == 'Hello World'
    
    # Test 2: Inline elements (should be concatenated)
    dom = Element('p')
    dom.text = 'Hello '
    span = SubElement(dom, 'span')
    span.text = 'World'
    assert extract_text(dom) == 'Hello World'
    
    # Test 3: Block elements (should add newlines)
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = 'First paragraph'
    p2 = SubElement(dom, 'p')
    p2.text = 'Second paragraph'
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'
    
    # Test 4: Separator elements (br)
    dom = Element('p')
    dom.text = 'Line 1'
    br = SubElement(dom, 'br')
    br.tail = 'Line 2'
    assert extract_text(dom) == 'Line 1\nLine 2'
    
    # Test 5: Nested elements
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = 'Text with '
    strong = SubElement(p, 'strong')
    strong.text = 'bold'
    strong.tail = ' text'
    assert extract_text(dom) == 'Text with bold text'
    
    # Test 6: Whitespace handling
    dom = Element('p')
    dom.text = '  Multiple   spaces   '
    assert extract_text(dom) == 'Multiple spaces'
    
    # Test 7: Empty elements
    dom = Element('div')
    assert extract_text(dom) == ''
    
    # Test 8: Custom block_symbol
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = 'First'
    p2 = SubElement(dom, 'p')
    p2.text = 'Second'
    assert extract_text(dom, block_symbol=' | ') == 'First | Second'
    
    # Test 9: Custom sep_symbol
    dom = Element('p')
    dom.text = 'Line 1'
    br = SubElement(dom, 'br')
    br.tail = 'Line 2'
    assert extract_text(dom, sep_symbol=' -- ') == 'Line 1 -- Line 2'
    
    # Test 10: Squash space disabled
    dom = Element('p')
    dom.text = '  Multiple   spaces   '
    assert extract_text(dom, squash_space=False) == '  Multiple   spaces   '
    
    # Test 11: Mixed inline and block elements
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = 'Paragraph with '
    span = SubElement(p, 'span')
    span.text = 'inline'
    span.tail = ' text'
    assert extract_text(dom) == 'Paragraph with inline text'
    
    # Test 12: Multiple block elements with whitespace
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = '  First  '
    p2 = SubElement(dom, 'p')
    p2.text = '  Second  '
    assert extract_text(dom) == 'First\nSecond'
    
    # Test 13: Nested block elements
    dom = Element('div')
    outer = SubElement(dom, 'div')
    outer.text = 'Outer '
    inner = SubElement(outer, 'p')
    inner.text = 'Inner'
    inner.tail = ' text'
    assert extract_text(dom) == 'Outer\nInner text'
    
    # Test 14: Multiple consecutive separators
    dom = Element('p')
    dom.text = 'Start'
    br1 = SubElement(dom, 'br')
    br2 = SubElement(dom, 'br')
    br2.tail = 'End'
    assert extract_text(dom) == 'Start\n\nEnd'
    
    # Test 15: Non-inline tag as inline (should not add newlines)
    dom = Element('custom')
    dom.text = 'Custom text'
    assert extract_text(dom) == 'Custom text'


# LLM-generated content at query #119
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline tag
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"], f"Expected ['Hello World'], got {result}"
    
    # Test with a block tag
    dom = html.fromstring("<div>Hello World</div>")
    result = extract_text_array(dom)
    assert result == ["Hello World"], f"Expected ['Hello World'], got {result}"
    
    # Test with separator tag (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"
    
    # Test with nested inline tags
    dom = html.fromstring("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"], f"Expected ['Hello ', 'World'], got {result}"
    
    # Test with nested block tags
    dom = html.fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text_array(dom)
    assert result == ["Hello", None, "World"], f"Expected ['Hello', None, 'World'], got {result}"
    
    # Test with text and tail
    dom = html.fromstring("<div>Hello <b>bold</b> world</div>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "bold", " world"], f"Expected ['Hello ', 'bold', ' world'], got {result}"
    
    # Test with squash_artifical_nl=True
    dom = html.fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ["Hello", None, "World"], f"Expected ['Hello', None, 'World'], got {result}"
    
    # Test with strip_artifical_nl=True
    dom = html.fromstring("<p>Hello</p>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["Hello"], f"Expected ['Hello'], got {result}"
    
    # Test with both squashing and stripping
    dom = html.fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["Hello", None, "World"], f"Expected ['Hello', None, 'World'], got {result}"
    
    # Test with empty div
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [], f"Expected [], got {result}"
    
    # Test with only text
    dom = html.fromstring("Just text")
    result = extract_text_array(dom)
    assert result == ["Just text"], f"Expected ['Just text'], got {result}"


# LLM-generated content at query #120
#--------------------------

```python
def test_extract_text_array():
    from lxml import html
    
    # Test simple inline text
    dom = html.fromstring("<span>hello</span>")
    assert extract_text_array(dom) == ["hello"]
    
    # Test block-level element adds artificial newlines
    dom = html.fromstring("<div>hello</div>")
    assert extract_text_array(dom) == ["hello"]
    
    # Test separator element
    dom = html.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test nested elements
    dom = html.fromstring("<div><span>hello</span><span>world</span></div>")
    result = extract_text_array(dom)
    assert "hello" in result
    assert "world" in result
    
    # Test text in different parts of element
    dom = html.fromstring("<p>start <b>bold</b> end</p>")
    result = extract_text_array(dom)
    assert "start " in result
    assert "bold" in result
    assert " end" in result
    
    # Test artificial newlines for block elements
    dom = html.fromstring("<div><p>first</p><p>second</p></div>")
    result = extract_text_array(dom)
    assert result[0] == "first"
    assert result[1] is None  # artificial newline between paragraphs
    assert result[2] == "second"
    
    # Test separator element in middle
    dom = html.fromstring("<div>text<br/>more</div>")
    result = extract_text_array(dom)
    assert "text" in result
    assert True in result  # separator
    assert "more" in result
    
    # Test squash_artifical_nl parameter
    dom = html.fromstring("<div><p>a</p><p>b</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result.count(None) > 1  # Multiple None values when not squashed
    
    # Test strip_artifical_nl parameter
    dom = html.fromstring("<div>  text  </div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # Leading artificial newline preserved
    
    # Test with callable tag (should return empty string)
    class MockElement:
        tag = lambda: None
    mock = MockElement()
    assert extract_text_array(mock) == ""


# LLM-generated content at query #121
#--------------------------

```python
def test_extract_text_array():
    from html.parser import HTMLParser
    
    class MockElement:
        def __init__(self, tag, text=None, tail=None, children=None):
            self.tag = tag
            self.text = text
            self.tail = tail
            self._children = children or []
        
        def getchildren(self):
            return self._children
    
    # Test empty element
    elem = MockElement('div')
    assert extract_text_array(elem) == []
    
    # Test inline element with text
    elem = MockElement('span', text='hello')
    assert extract_text_array(elem) == ['hello']
    
    # Test block element with text
    elem = MockElement('div', text='hello')
    result = extract_text_array(elem)
    assert result[0] is None  # artificial newline before
    assert 'hello' in result
    assert result[-1] is None  # artificial newline after
    
    # Test separator element
    elem = MockElement('br')
    result = extract_text_array(elem)
    assert result == [True]  # separator marker
    
    # Test nested elements
    child = MockElement('span', text='world')
    parent = MockElement('div', text='hello ', children=[child], tail='!')
    result = extract_text_array(parent)
    assert None in result  # artificial newlines
    assert 'hello ' in result
    assert 'world' in result
    assert '!' in result
    
    # Test squash_artifical_nl parameter
    elem = MockElement('div', text='hello')
    result_with_squash = extract_text_array(elem, squash_artifical_nl=True)
    result_without_squash = extract_text_array(elem, squash_artifical_nl=False)
    assert len(result_with_squash) < len(result_without_squash)
    
    # Test strip_artifical_nl parameter
    elem = MockElement('div', text='hello')
    result_with_strip = extract_text_array(elem, strip_artifical_nl=True)
    result_without_strip = extract_text_array(elem, strip_artifical_nl=False)
    assert len(result_with_strip) < len(result_without_strip)
    
    # Test callable tag returns empty string
    class CallableTag:
        def __call__(self):
            pass
    
    elem = MockElement(CallableTag())
    assert extract_text_array(elem) == ''
    
    # Test multiple nested elements with separators
    inner = MockElement('br')
    outer = MockElement('div', children=[inner])
    result = extract_text_array(outer)
    assert True in result  # separator present
    
    # Test nested inline elements
    child = MockElement('strong', text='bold')
    parent = MockElement('span', text='some ', children=[child], tail=' text')
    result = extract_text_array(parent)
    assert 'some ' in result
    assert 'bold' in result
    assert ' text' in result
    # No artificial newlines for inline elements
    none_count = sum(1 for x in result if x is None)
    assert none_count == 0


# LLM-generated content at query #122
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    from lxml.html import fromstring
    dom = fromstring("<p>Hello world</p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with nested inline elements
    dom = fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with block elements creating newlines
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator elements (br)
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with custom block_symbol
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "A | B"
    
    # Test with custom sep_symbol
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    dom = fromstring("<p>  Hello   world  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   world  "
    
    # Test with empty content
    dom = fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with nested containers
    dom = fromstring("<div><span>Hello</span><div><span>World</span></div></div>")
    result = extract_text(dom)
    assert "Hello" in result
    assert "World" in result
    
    # Test with images (inline but no text)
    dom = fromstring("<p>Text <img src='test.png'> more text</p>")
    assert extract_text(dom) == "Text  more text"
    
    # Test complex nested structure
    dom = fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <b>bold</b> text</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with bold text" in result
    assert "Item 1" in result
    assert "Item 2" in result
    assert result.count('\n') >= 3  # Multiple newlines for block elements
```


# LLM-generated content at query #123
#--------------------------

```python
def test_extract_text_array():
    from lxml import etree
    
    # Test simple inline tag
    dom = etree.fromstring("<span>hello</span>")
    assert extract_text_array(dom) == ["hello"]
    
    # Test separator tag
    dom = etree.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test block tag
    dom = etree.fromstring("<div>text</div>")
    assert extract_text_array(dom) == ["text"]
    
    # Test nested structure
    dom = etree.fromstring("<div><p>para1</p><p>para2</p></div>")
    result = extract_text_array(dom)
    assert result == ["para1", None, "para2"]
    
    # Test with text and tail
    dom = etree.fromstring("<div>text1<span>inner</span>text2</div>")
    result = extract_text_array(dom)
    assert "text1" in result
    assert "inner" in result
    assert "text2" in result
    
    # Test with separator inside block
    dom = etree.fromstring("<div>line1<br/>line2</div>")
    result = extract_text_array(dom)
    assert result == ["line1", True, "line2"]
    
    # Test squash_artifical_nl parameter
    dom = etree.fromstring("<div><p>text</p></div>")
    result_no_squash = extract_text_array(dom, squash_artifical_nl=False)
    result_squash = extract_text_array(dom, squash_artifical_nl=True)
    assert None not in result_squash
    assert None in result_no_squash
    
    # Test strip_artifical_nl parameter
    dom = etree.fromstring("<div><p>text</p></div>")
    result_strip = extract_text_array(dom, strip_artifical_nl=True)
    result_no_strip = extract_text_array(dom, strip_artifical_nl=False)
    assert result_strip == ["text"]
    assert None in result_no_strip
    
    # Test nested block tags
    dom = etree.fromstring("<body><div><p>text</p></div></body>")
    result = extract_text_array(dom)
    assert result == ["text"]
    
    # Test callable tag returns empty string
    class CallableTag:
        def __call__(self):
            pass
    dom = type('Mock', (), {'tag': CallableTag(), 'text': None, 'getchildren': lambda: []})()
    assert extract_text_array(dom) == ''


# LLM-generated content at query #124
#--------------------------

```python
def test_extract_text_array():
    from html.parser import HTMLParser
    
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail
        
        def getchildren(self):
            return self.children
    
    # Test with simple inline element
    elem = MockElement('span', text='Hello')
    result = extract_text_array(elem)
    assert result == ['Hello']
    
    # Test with block element (should add None for artificial newlines)
    elem = MockElement('div', text='Hello')
    result = extract_text_array(elem)
    assert result == [None, 'Hello', None]
    
    # Test with separator element (br)
    elem = MockElement('br')
    result = extract_text_array(elem)
    assert result == [True]
    
    # Test with nested elements
    child = MockElement('strong', text='World')
    elem = MockElement('p', text='Hello ', children=[child])
    result = extract_text_array(elem)
    assert None in result
    assert 'Hello ' in result
    assert 'World' in result
    
    # Test that squash_artifical_nl works
    elem = MockElement('div', text='Line1')
    child = MockElement('div', text='Line2')
    elem.children = [child]
    result = extract_text_array(elem, squash_artifical_nl=True)
    # Should have squashed consecutive None values
    none_count = sum(1 for x in result if x is None)
    assert none_count <= 2  # Should have at most 2 Nones (start and end)
    
    # Test that strip_artifical_nl works
    elem = MockElement('div', text='Content')
    result = extract_text_array(elem, strip_artifical_nl=True)
    assert result[0] != None  # Should not start with None
    assert result[-1] != None  # Should not end with None
    
    # Test with callable tag (should return empty string)
    class CallableMock:
        def __call__(self):
            pass
    elem = MockElement(CallableMock())
    result = extract_text_array(elem)
    assert result == ['']


# LLM-generated content at query #125
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from xml.etree.ElementTree import Element, SubElement
    
    # Test 1: Simple paragraph with text
    p = Element('p')
    p.text = "Hello World"
    assert extract_text(p) == "Hello World"
    
    # Test 2: Paragraph with inline elements
    p = Element('p')
    p.text = "Hello "
    strong = SubElement(p, 'strong')
    strong.text = "World"
    assert extract_text(p) == "Hello World"
    
    # Test 3: Nested inline elements
    p = Element('p')
    p.text = "Hello "
    em = SubElement(p, 'em')
    em.text = "beautiful "
    strong = SubElement(em, 'strong')
    strong.text = "World"
    assert extract_text(p) == "Hello beautiful World"
    
    # Test 4: Block elements create newlines
    div = Element('div')
    p1 = SubElement(div, 'p')
    p1.text = "First"
    p2 = SubElement(div, 'p')
    p2.text = "Second"
    assert extract_text(div) == "First\nSecond"
    
    # Test 5: Separator elements (br)
    p = Element('p')
    p.text = "Line1"
    br = SubElement(p, 'br')
    br.tail = "Line2"
    assert extract_text(p) == "Line1\nLine2"
    
    # Test 6: Mixed content with block and inline
    div = Element('div')
    p = SubElement(div, 'p')
    p.text = "Hello "
    strong = SubElement(p, 'strong')
    strong.text = "World"
    p2 = SubElement(div, 'p')
    p2.text = "Second paragraph"
    assert extract_text(div) == "Hello World\nSecond paragraph"
    
    # Test 7: Whitespace handling
    p = Element('p')
    p.text = "  Hello   World  "
    assert extract_text(p) == "Hello World"
    
    # Test 8: Empty elements
    div = Element('div')
    p = SubElement(div, 'p')
    p.text = ""
    assert extract_text(div) == ""
    
    # Test 9: Deeply nested structure
    div = Element('div')
    section = SubElement(div, 'section')
    p = SubElement(section, 'p')
    p.text = "Deep text"
    assert extract_text(div) == "Deep text"
    
    # Test 10: Custom block_symbol
    p1 = Element('p')
    p1.text = "First"
    p2 = Element('p')
    p2.text = "Second"
    div = Element('div')
    div.append(p1)
    div.append(p2)
    assert extract_text(div, block_symbol='|') == "First|Second"
    
    # Test 11: Custom sep_symbol
    p = Element('p')
    p.text = "Line1"
    br = SubElement(p, 'br')
    br.tail = "Line2"
    assert extract_text(p, sep_symbol='<br>') == "Line1<br>Line2"
    
    # Test 12: Multiple separators
    p = Element('p')
    p.text = "Start"
    br1 = SubElement(p, 'br')
    br1.tail = ""
    br2 = SubElement(p, 'br')
    br2.tail = "End"
    assert extract_text(p) == "Start\n\nEnd" or extract_text(p) == "Start\nEnd"
    
    # Test 13: Leading/trailing whitespace
    p = Element('p')
    p.text = "  Text with spaces  "
    assert extract_text(p) == "Text with spaces"
    
    # Test 14: Nested block elements
    div = Element('div')
    outer = SubElement(div, 'div')
    inner = SubElement(outer, 'p')
    inner.text = "Nested"
    assert extract_text(div) == "Nested"
    
    # Test 15: Complex real-world example
    html = Element('div')
    h1 = SubElement(html, 'h1')
    h1.text = "Title"
    p = SubElement(html, 'p')
    p.text = "This is a "
    a = SubElement(p, 'a')
    a.text = "link"
    a.tail = " in a paragraph."
    assert extract_text(html) == "Title\nThis is a link in a paragraph."


# LLM-generated content at query #126
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text node
    from lxml.html import fromstring
    dom = fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == [None, 'Hello World', None]
    
    # Test with inline tag
    dom = fromstring("<span>Inline text</span>")
    result = extract_text_array(dom)
    assert result == ['Inline text']
    
    # Test with separator tag
    dom = fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested tags
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, 'First', None, None, None, 'Second', None, None]
    
    # Test with mixed inline and block elements
    dom = fromstring("<p>Text with <span>inline</span> and <br/> break</p>")
    result = extract_text_array(dom)
    assert result == [None, 'Text with ', 'inline', ' and ', True, ' break', None]
    
    # Test with text and tail
    dom = fromstring("<div>Start <b>bold</b> End</div>")
    result = extract_text_array(dom)
    assert result == [None, 'Start ', 'bold', ' End', None]
    
    # Test empty element
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with squash_artifical_nl=False
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result
    assert result.count(None) >= 4  # Multiple None values should exist
    
    # Test with strip_artifical_nl=False
    dom = fromstring("<p>Text</p>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'Text', None]


# LLM-generated content at query #127
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml.html import fromstring
    dom = fromstring('<span>Hello</span>')
    result = extract_text_array(dom)
    assert result == ['Hello']
    
    # Test with block tag (non-inline, non-separator)
    dom = fromstring('<div>Hello</div>')
    result = extract_text_array(dom)
    assert result == ['Hello']
    
    # Test with separator tag (br)
    dom = fromstring('<br>')
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested tags
    dom = fromstring('<div><span>Hello</span> World</div>')
    result = extract_text_array(dom)
    assert result == ['Hello', ' World']
    
    # Test with multiple block tags
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    result = extract_text_array(dom)
    assert result == ['First', 'Second']
    
    # Test with artificial newlines (None values)
    dom = fromstring('<div><span>Hello</span></div>')
    result = extract_text_array(dom)
    assert None not in result  # None values should be squashed/stripped
    
    # Test with squash_artifical_nl=False
    dom = fromstring('<div><span>Hello</span></div>')
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result[0] is None  # Should preserve artificial newlines
    
    # Test with strip_artifical_nl=False
    dom = fromstring('<div><span>Hello</span></div>')
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[-1] is None  # Should preserve trailing artificial newline
    
    # Test with callable tag (should return empty string)
    class CallableTag:
        def __call__(self):
            pass
    dom = fromstring('<div></div>')
    dom.tag = CallableTag()
    result = extract_text_array(dom)
    assert result == ''
    
    # Test with text and tail
    dom = fromstring('<div>Start<span>Middle</span>End</div>')
    result = extract_text_array(dom)
    assert 'Start' in result
    assert 'Middle' in result
    assert 'End' in result
    
    # Test with multiple children
    dom = fromstring('<ul><li>Item 1</li><li>Item 2</li></ul>')
    result = extract_text_array(dom)
    assert 'Item 1' in result
    assert 'Item 2' in result
    
    # Test with empty dom
    dom = fromstring('<div></div>')
    result = extract_text_array(dom)
    assert result == []


# LLM-generated content at query #128
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple text with inline tags
    class MockElement:
        def __init__(self, tag, text=None, children=None, tail=None):
            self.tag = tag
            self.text = text
            self.children = children or []
            self.tail = tail
        
        def getchildren(self):
            return self.children
    
    # Test basic text extraction
    dom = MockElement('p', text='Hello World')
    result = extract_text(dom)
    assert result == 'Hello World', f"Expected 'Hello World', got '{result}'"
    
    # Test with inline tags (should not add newlines)
    span = MockElement('span', text='inner')
    dom = MockElement('p', text='before ', children=[span], tail=' after')
    # Need to set tail on child for proper extraction
    span.tail = ' after'
    result = extract_text(dom)
    assert result == 'before inner after', f"Expected 'before inner after', got '{result}'"
    
    # Test with separator tag (br)
    br = MockElement('br')
    dom = MockElement('p', text='line1', children=[br], tail='line2')
    br.tail = 'line2'
    result = extract_text(dom, sep_symbol='\n')
    assert result == 'line1\nline2', f"Expected 'line1\\nline2', got '{result}'"
    
    # Test with block-level tags (div)
    inner_div = MockElement('div', text='inner')
    dom = MockElement('div', text='outer', children=[inner_div])
    result = extract_text(dom, block_symbol='\n')
    assert result == 'outer\ninner', f"Expected 'outer\\ninner', got '{result}'"
    
    # Test with nested inline tags
    strong = MockElement('strong', text='emphasized')
    dom = MockElement('p', text='This is ', children=[strong], tail=' text')
    strong.tail = ' text'
    result = extract_text(dom)
    assert result == 'This is emphasized text', f"Expected 'This is emphasized text', got '{result}'"
    
    # Test with multiple children
    span1 = MockElement('span', text='first')
    span2 = MockElement('span', text='second')
    dom = MockElement('div', children=[span1, span2])
    span1.tail = ' '
    span2.tail = None
    result = extract_text(dom)
    assert result == 'first second', f"Expected 'first second', got '{result}'"
    
    # Test with whitespace squashing
    dom = MockElement('p', text='  too   much   space  ')
    result = extract_text(dom)
    assert result == 'too much space', f"Expected 'too much space', got '{result}'"
    
    # Test with empty element
    dom = MockElement('p')
    result = extract_text(dom)
    assert result == '', f"Expected empty string, got '{result}'"
    
    # Test custom symbols
    dom = MockElement('div', text='hello')
    result = extract_text(dom, block_symbol='|', sep_symbol='|')
    assert result == 'hello', f"Expected 'hello', got '{result}'"
    
    # Test with callable tag (should return empty)
    class CallableTag:
        def __call__(self):
            pass
    
    dom = MockElement(CallableTag(), text='should not appear')
    result = extract_text(dom)
    assert result == '', f"Expected empty string, got '{result}'"
    
    # Test complex nested structure
    inner_span = MockElement('span', text='world')
    outer_div = MockElement('div', text='hello ', children=[inner_span], tail='!')
    inner_span.tail = '!'
    dom = MockElement('body', children=[outer_div])
    result = extract_text(dom)
    assert result == 'hello world!', f"Expected 'hello world!', got '{result}'"
```


