####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple inline text
    dom1 = type('Mock', (), {'tag': 'p', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom1) == 'Hello'

    # Test 2: Nested inline elements
    child = type('Mock', (), {'tag': 'strong', 'text': 'World', 'tail': '!', 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'p', 'text': 'Hello ', 'tail': None, 'getchildren': lambda self: [child]})()
    assert extract_text(parent) == 'Hello World!'

    # Test 3: Block element with newline
    child = type('Mock', (), {'tag': 'div', 'text': 'Content', 'tail': None, 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'body', 'text': None, 'tail': None, 'getchildren': lambda self: [child]})()
    assert extract_text(parent) == 'Content'

    # Test 4: Separator (br tag)
    dom = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == '\n'

    # Test 5: Multiple separators
    child1 = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    child2 = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: [child1, child2]})()
    assert extract_text(parent) == '\n\n'

    # Test 6: Whitespace collapsing
    dom = type('Mock', (), {'tag': 'p', 'text': 'Hello   World', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == 'Hello World'

    # Test 7: Leading/trailing whitespace stripped
    dom = type('Mock', (), {'tag': 'p', 'text': '  Hello  ', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == 'Hello'

    # Test 8: Custom symbols
    dom = type('Mock', (), {'tag': 'div', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: []})()
    dom2 = type('Mock', (), {'tag': 'p', 'text': 'World', 'tail': None, 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'body', 'text': None, 'tail': None, 'getchildren': lambda self: [dom, dom2]})()
    assert extract_text(parent, block_symbol='|') == 'Hello|World'

    # Test 9: Empty input
    dom = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text(dom) == ''

    # Test 10: Multiple blocks with separators
    child1 = type('Mock', () , {'tag': 'div', 'text': 'Line1', 'tail': None, 'getchildren': lambda self: []})()
    child2 = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    child3 = type('Mock', (), {'tag': 'div', 'text': 'Line2', 'tail': None, 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'body', 'text': None, 'tail': None, 'getchildren': lambda self: [child1, child2, child3]})()
    assert extract_text(parent) == 'Line1\nLine2'

    # Test 11: Nested block elements
    inner = type('Mock', (), {'tag': 'p', 'text': 'inner', 'tail': None, 'getchildren': lambda self: []})()
    outer = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: [inner]})()
    assert extract_text(outer) == 'inner'

    # Test 12: Multiple text nodes in one element
    child = type('Mock', (), {'tag': 'span', 'text': 'world', 'tail': '!', 'getchildren': lambda self: []})()
    parent = type('Mock', (), {'tag': 'p', 'text': 'Hello ', 'tail': None, 'getchildren': lambda self: [child]})()
    assert extract_text(parent) == 'Hello world!'


# LLM-generated content at query #2
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from xml.etree import ElementTree as ET
    html = '<p>Hello <b>world</b>!</p>'
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "Hello world!", f"Expected 'Hello world!', got '{result}'"

    # Test with separator element (br)
    html = '<p>Line1<br/>Line2</p>'
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "Line1\nLine2", f"Expected 'Line1\\nLine2', got '{result}'"

    # Test with block elements (div, p)
    html = '<div><p>First</p><p>Second</p></div>'
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "First\nSecond", f"Expected 'First\\nSecond', got '{result}'"

    # Test with whitespace squashing
    html = '<p>Hello   \n   world</p>'
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "Hello world", f"Expected 'Hello world', got '{result}'"

    # Test with nested inline elements
    html = '<span>Hello <em>beautiful</em> world</span>'
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "Hello beautiful world", f"Expected 'Hello beautiful world', got '{result}'"

    # Test with empty element
    html = '<div></div>'
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "", f"Expected empty string, got '{result}'"

    # Test with only text
    html = '<p>Just text</p>'
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "Just text", f"Expected 'Just text', got '{result}'"

    # Test with multiple br elements
    html = '<p>Line1<br/><br/>Line2</p>'
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "Line1\n\nLine2", f"Expected 'Line1\\n\\nLine2', got '{result}'"

    # Test with leading/trailing whitespace
    html = '  <p>  Hello  </p>  '
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "Hello", f"Expected 'Hello', got '{result}'"

    # Test with custom block_symbol
    html = '<div><p>First</p><p>Second</p></div>'
    dom = ET.fromstring(html)
    result = extract_text(dom, block_symbol=' ')
    assert result == "First Second", f"Expected 'First Second', got '{result}'"

    # Test with custom sep_symbol
    html = '<p>Line1<br/>Line2</p>'
    dom = ET.fromstring(html)
    result = extract_text(dom, sep_symbol=' ')
    assert result == "Line1 Line2", f"Expected 'Line1 Line2', got '{result}'"

    # Test with squash_space=False
    html = '<p>Hello   world</p>'
    dom = ET.fromstring(html)
    result = extract_text(dom, squash_space=False)
    assert result == "Hello   world", f"Expected 'Hello   world', got '{result}'"

    # Test with nested block elements
    html = '<div><div><p>Deep</p></div></div>'
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "Deep", f"Expected 'Deep', got '{result}'"

    # Test with mixed inline and block elements
    html = '<div>Hello <span>world</span><p>New paragraph</p></div>'
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "Hello world\nNew paragraph", f"Expected 'Hello world\\nNew paragraph', got '{result}'"

    # Test with tail text
    html = '<p>Hello<b>bold</b>world</p>'
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "Helloboldworld", f"Expected 'Helloboldworld', got '{result}'"

    # Test with multiple levels of nesting
    html = '<div><p>First <b>bold</b> text</p><p>Second <i>italic</i> text</p></div>'
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert result == "First bold text\nSecond italic text", f"Expected 'First bold text\\nSecond italic text', got '{result}'"
```


# LLM-generated content at query #3
#--------------------------

```python
def test_extract_text():
    # Test basic inline tags - should not add newlines
    from xml.etree import ElementTree as ET
    
    # Simple text node
    dom = ET.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Inline tags within block
    dom = ET.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Multiple inline tags
    dom = ET.fromstring("<p>Hello <b>bold</b> and <i>italic</i></p>")
    assert extract_text(dom) == "Hello bold and italic"
    
    # Separator tags (like <br>)
    dom = ET.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Multiple separators
    dom = ET.fromstring("<p>Line1<br/><br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Block tags should add newlines
    dom = ET.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Nested block tags
    dom = ET.fromstring("<div><div><p>Deep</p></div></div>")
    assert extract_text(dom) == "Deep"
    
    # Whitespace handling
    dom = ET.fromstring("<p>Hello   World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Leading/trailing whitespace
    dom = ET.fromstring("<p>  Hello World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Mixed content
    dom = ET.fromstring("<div><p>Hello <b>World</b></p><p>Second<br/>line</p></div>")
    assert extract_text(dom) == "Hello World\nSecond\nline"
    
    # Empty elements
    dom = ET.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Element with only whitespace
    dom = ET.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Custom block_symbol
    dom = ET.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=" - ") == "First - Second"
    
    # Custom sep_symbol
    dom = ET.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol=" | ") == "Line1 | Line2"
    
    # squash_space=False
    dom = ET.fromstring("<p>  Hello   World  </p>")
    result = extract_text(dom, squash_space=False)
    assert "  " in result  # Should preserve original whitespace
    
    # Complex nested structure
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
    dom = ET.fromstring(html)
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with bold text" in result
    assert "Item 1" in result
    assert "Item 2" in result
    assert result.count("\n") >= 4  # Should have multiple newlines
    
    # Nested inline tags
    dom = ET.fromstring("<p><b><i>Bold italic</i></b></p>")
    assert extract_text(dom) == "Bold italic"
    
    # Inline tag with tail text
    dom = ET.fromstring("<p><b>Bold</b> and <i>italic</i> text</p>")
    assert extract_text(dom) == "Bold and italic text"
    
    # Separator with no text
    dom = ET.fromstring("<p><br/></p>")
    assert extract_text(dom) == ""
    
    # Multiple block elements at same level
    dom = ET.fromstring("<div><p>A</p><p>B</p><p>C</p></div>")
    assert extract_text(dom) == "A\nB\nC"```


# LLM-generated content at query #4
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    dom = type('Mock', (), {'tag': 'span', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text_array(dom) == ['Hello']

    # Test with block element
    dom = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text_array(dom) == [None]

    # Test with separator element
    dom = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text_array(dom) == [True]

    # Test nested inline inside block
    child = type('Mock', (), {'tag': 'span', 'text': 'World', 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': 'Hello ', 'tail': None, 'getchildren': lambda self: [child]})()
    result = extract_text_array(dom)
    assert None in result
    assert 'Hello ' in result
    assert 'World' in result

    # Test with tail text
    child = type('Mock', (), {'tag': 'span', 'text': 'Hello', 'tail': ' World', 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: [child]})()
    result = extract_text_array(dom)
    assert 'Hello' in result
    assert ' World' in result

    # Test with callable tag (should return empty string)
    dom = type('Mock', (), {'tag': lambda: None, 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    assert extract_text_array(dom) == ''

    # Test with multiple children
    child1 = type('Mock', (), {'tag': 'span', 'text': 'First', 'tail': ' ', 'getchildren': lambda self: []})()
    child2 = type('Mock', (), {'tag': 'span', 'text': 'Second', 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: [child1, child2]})()
    result = extract_text_array(dom)
    assert 'First' in result
    assert 'Second' in result

    # Test consecutive None squashing
    child = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: [child]})()
    result = extract_text_array(dom)
    # Should have only one None instead of three (start, child div, end)
    none_count = sum(1 for x in result if x is None)
    assert none_count <= 1

    # Test None stripping at edges
    child = type('Mock', (), {'tag': 'span', 'text': 'Content', 'tail': None, 'getchildren': lambda self: []})()
    dom = type('Mock', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: [child]})()
    result = extract_text_array(dom)
    assert result[0] is not None
    assert result[-1] is not None


# LLM-generated content at query #5
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
    
    # Test with separator (br)
    dom = etree.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"
    
    # Test with nested elements
    dom = etree.fromstring("<p>Hello <b>World</b>!</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "World", "!", None], f"Expected [None, 'Hello ', 'World', '!', None], got {result}"
    
    # Test with nested inline elements
    dom = etree.fromstring("<span>Hello <b>World</b></span>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"], f"Expected ['Hello ', 'World'], got {result}"
    
    # Test with separator inside inline
    dom = etree.fromstring("<span>Line1<br/>Line2</span>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"], f"Expected ['Line1', True, 'Line2'], got {result}"
    
    # Test with callable tag (should return empty)
    class FakeElement:
        tag = lambda: None
    dom = FakeElement()
    result = extract_text_array(dom)
    assert result == "", f"Expected empty string, got {result}"
    
    # Test with squash_artifical_nl=False and strip_artifical_nl=False
    dom = etree.fromstring("<p>Text</p>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Text", None], f"Expected [None, 'Text', None], got {result}"
    
    # Test with multiple consecutive block elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, "First", None, "Second", None], f"Expected [None, 'First', None, 'Second', None], got {result}"
    
    # Test with empty element
    dom = etree.fromstring("<p></p>")
    result = extract_text_array(dom)
    assert result == [None, None], f"Expected [None, None], got {result}"
```


# LLM-generated content at query #6
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello world</p>")
    result = extract_text_array(dom)
    assert result == ["Hello world"]
    
    # Test with inline elements
    dom = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "bold", " world"]
    
    # Test with separator element
    dom = etree.fromstring("<div>Line1<br/>Line2</div>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None]
    
    # Test with nested block elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, None, "Second", None, None]
    
    # Test with squash_artifical_nl=False
    dom = etree.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "Text", None, None]
    
    # Test with strip_artifical_nl=False
    dom = etree.fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Text", None]
    
    # Test empty element
    dom = etree.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test element with only tail text
    dom = etree.fromstring("<div><br/>tail</div>")
    result = extract_text_array(dom)
    assert result == [True, "tail"]
    
    # Test callable tag (like comment)
    class MockElement:
        tag = lambda: None
    assert extract_text_array(MockElement()) == ""


# LLM-generated content at query #7
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with inline tags
    dom = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "bold", " world"]
    
    # Test with separator tags (br)
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]
    
    # Test with block-level tags (div)
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, "First", None, None, "Second", None]
    
    # Test with nested structure
    dom = etree.fromstring("<div>Text <span>inner</span> more</div>")
    result = extract_text_array(dom)
    assert result == [None, "Text ", "inner", " more", None]
    
    # Test with empty element
    dom = etree.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with callable tag (should return empty string)
    class MockDom:
        tag = lambda: None
    result = extract_text_array(MockDom())
    assert result == ""
    
    # Test with squash_artifical_nl=False
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result
    
    # Test with strip_artifical_nl=False
    dom = etree.fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None


# LLM-generated content at query #8
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction from a simple element
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with inline elements
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with separator element (br)
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with nested structure
    dom = html.fromstring("<div><p>Hello <b>World</b></p><p>Second line</p></div>")
    assert extract_text(dom) == "Hello World\nSecond line"
    
    # Test whitespace handling
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with multiple whitespace and newlines
    dom = html.fromstring("<p>Hello\n\n\nWorld</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test empty content
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test with inline tags that should not create newlines
    dom = html.fromstring("<p>Text <span>inside</span> span</p>")
    assert extract_text(dom) == "Text inside span"
    
    # Test custom block_symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "First | Second"
    
    # Test custom sep_symbol
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>First paragraph with <b>bold</b> text</p>
            <p>Second paragraph<br/>with line break</p>
        </div>
    """)
    expected = "Title\nFirst paragraph with bold text\nSecond paragraph\nwith line break"
    assert extract_text(dom) == expected
    
    # Test with non-inline tags that are not block (like script)
    dom = html.fromstring("<div>Hello<script>alert('test');</script>World</div>")
    result = extract_text(dom)
    assert "alert" not in result
    assert "Hello" in result
    assert "World" in result
```


# LLM-generated content at query #9
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    assert extract_text_array(dom) == ["Hello World"]
    
    # Test with a block element (should add None before and after)
    dom = html.fromstring("<div>Text</div>")
    assert extract_text_array(dom) == ["Text"]
    
    # Test with separator element (br)
    dom = html.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with nested elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"]
    
    # Test with inline element containing text and tail
    dom = html.fromstring("<p>Hello <b>World</b>!</p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World", "!"]
    
    # Test with separator in between text
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]
    
    # Test with multiple levels of nesting
    dom = html.fromstring("<div><ul><li>Item1</li><li>Item2</li></ul></div>")
    result = extract_text_array(dom)
    assert result == ["Item1", "Item2"]
    
    # Test with empty element
    dom = html.fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test with text only
    dom = html.fromstring("Just text")
    assert extract_text_array(dom) == ["Just text"]
    
    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result  # Should contain artificial newlines
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None  # Should have leading/trailing None


# LLM-generated content at query #10
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import etree
    dom = etree.fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"], f"Expected ['Hello'], got {result}"
    
    # Test with a block element (should add None for artificial newlines)
    dom = etree.fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == ["Hello"], f"Expected ['Hello'], got {result}"
    
    # Test with nested elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"], f"Expected ['First', 'Second'], got {result}"
    
    # Test with separator tag (br)
    dom = etree.fromstring("<span>Line1<br/>Line2</span>")
    result = extract_text_array(dom)
    assert True in result, "Expected True for br tag"
    assert result == ["Line1", True, "Line2"], f"Expected ['Line1', True, 'Line2'], got {result}"
    
    # Test with inline tags (should not add None)
    dom = etree.fromstring("<p><b>Bold</b> and <i>italic</i></p>")
    result = extract_text_array(dom)
    assert result == ["Bold", " and ", "italic"], f"Expected ['Bold', ' and ', 'italic'], got {result}"
    
    # Test with mixed content
    dom = etree.fromstring("<div>Start <span>middle</span> end</div>")
    result = extract_text_array(dom)
    assert result == ["Start ", "middle", " end"], f"Expected ['Start ', 'middle', ' end'], got {result}"
    
    # Test with tail text
    dom = etree.fromstring("<div><p>Para1</p>Text after</div>")
    result = extract_text_array(dom)
    assert result == ["Para1", "Text after"], f"Expected ['Para1', 'Text after'], got {result}"
    
    # Test with callable tag (should return empty string)
    class MockElement:
        def __init__(self):
            self.tag = lambda: None
    mock_dom = MockElement()
    result = extract_text_array(mock_dom)
    assert result == [], f"Expected [], got {result}"
    
    # Test with squash_artifical_nl=False
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result, "Expected None for artificial newlines"
    
    # Test with strip_artifical_nl=False
    dom = etree.fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None or result[-1] is None, "Expected leading/trailing None"
    
    # Test with both parameters False
    dom = etree.fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result[0] is None, "Expected leading None"
    assert result[-1] is None, "Expected trailing None"
    assert result == [None, "A", None], f"Expected [None, 'A', None], got {result}"
    
    # Test with nested separators
    dom = etree.fromstring("<div>Text<br/><br/>More text</div>")
    result = extract_text_array(dom)
    assert result.count(True) == 2, "Expected two True values for two br tags"


# LLM-generated content at query #11
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with inline elements
    dom = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with separator (br)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with nested elements
    dom = html.fromstring("<div><p>Text with <span>span</span> inside</p></div>")
    assert extract_text(dom) == "Text with span inside"
    
    # Test with whitespace handling
    dom = html.fromstring("<p>  Multiple   spaces   </p>")
    assert extract_text(dom) == "Multiple spaces"
    
    # Test with empty content
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test custom block_symbol
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "A | B"
    
    # Test custom sep_symbol
    dom = html.fromstring("<p>A<br>B</p>")
    assert extract_text(dom, sep_symbol=' --- ') == "A --- B"
    
    # Test squash_space=False
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test complex nested structure
    dom = html.fromstring("""
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
    
    # Test with multiple br tags
    dom = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(dom) == "Line1\n\nLine2"
    
    # Test with text in root element
    dom = html.fromstring("<p>Root text</p>")
    assert extract_text(dom) == "Root text"
    
    # Test with tail text
    dom = html.fromstring("<p>Before <b>bold</b> After</p>")
    assert extract_text(dom) == "Before bold After"
```


# LLM-generated content at query #12
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction from a simple element
    from lxml.html import fromstring
    dom = fromstring("<p>Hello world</p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with inline tags - should not add newlines
    dom = fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with block elements - should add newlines
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator tags (br)
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested elements
    dom = fromstring("<div><p>Text with <span>span</span> inside</p></div>")
    assert extract_text(dom) == "Text with span inside"
    
    # Test with whitespace squashing
    dom = fromstring("<p>  Hello    world  </p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with multiple spaces and newlines in HTML
    dom = fromstring("<p>Hello\n\n  world</p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with empty elements
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with elements containing only whitespace
    dom = fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test with custom block_symbol
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=" | ") == "First | Second"
    
    # Test with custom sep_symbol
    dom = fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=" | ") == "Line1 | Line2"
    
    # Test with squash_space=False
    dom = fromstring("<p>  Hello    world  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello    world  "
    
    # Test with complex nested structure
    html = """
    <div>
        <h1>Title</h1>
        <p>Paragraph with <a href="#">link</a> and <strong>bold</strong></p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </div>
    """
    dom = fromstring(html)
    expected = "Title\nParagraph with link and bold\nItem 1\nItem 2"
    assert extract_text(dom) == expected
    
    # Test with multiple br tags
    dom = fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(dom) == "Line1\n\nLine2"
    
    # Test with inline elements that are not in INLINE_TAGS
    dom = fromstring("<p>Text <custom>custom</custom> text</p>")
    assert extract_text(dom) == "Text custom text"
    
    # Test with tail text
    dom = fromstring("<p>Before <b>bold</b> After</p>")
    assert extract_text(dom) == "Before bold After"
```


# LLM-generated content at query #13
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple inline text
    from lxml import html
    dom = html.fromstring("<p>Hello world</p>")
    assert extract_text(dom) == "Hello world"
    
    # Test 2: Nested inline elements
    dom = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test 3: Block elements with separators
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test 4: Line break separator
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test 5: Multiple whitespace handling
    dom = html.fromstring("<p>Hello    world</p>")
    assert extract_text(dom) == "Hello world"
    
    # Test 6: Whitespace with newlines
    dom = html.fromstring("<p>Hello\n  world</p>")
    assert extract_text(dom) == "Hello world"
    
    # Test 7: Nested block elements
    dom = html.fromstring("<div><p>First</p><div><p>Second</p></div></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test 8: Mixed inline and block elements
    dom = html.fromstring("<div><p>Text with <b>bold</b> and <i>italic</i></p></div>")
    assert extract_text(dom) == "Text with bold and italic"
    
    # Test 9: Empty elements
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test 10: Element with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test 11: Custom separator symbols
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "First | Second"
    
    # Test 12: Custom separator for br
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test 13: No whitespace squashing
    dom = html.fromstring("<p>Hello    world</p>")
    assert extract_text(dom, squash_space=False) == "Hello    world"
    
    # Test 14: Complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <a href="#">link</a> text</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    expected = "Title\nParagraph with link text\nItem 1\nItem 2"
    assert extract_text(dom) == expected
    
    # Test 15: Multiple br tags
    dom = html.fromstring("<p>Line1<br/><br/>Line2</p>")
    assert extract_text(dom) == "Line1\n\nLine2"
    
    # Test 16: Leading and trailing whitespace
    dom = html.fromstring("  <p>Content</p>  ")
    assert extract_text(dom) == "Content"
    
    # Test 17: Inline elements that are not inline (should create newlines)
    dom = html.fromstring("<div><span>Inline</span><div>Block</div></div>")
    assert extract_text(dom) == "Inline\nBlock"
    
    # Test 18: Deep nesting
    dom = html.fromstring("<div><p><b><i>Deep</i></b></p></div>")
    assert extract_text(dom) == "Deep"
    
    # Test 19: Mixed separators and blocks
    dom = html.fromstring("<div><p>Text</p><br/><p>More</p></div>")
    assert extract_text(dom) == "Text\n\nMore"
    
    # Test 20: Script and style tags should be handled as inline
    dom = html.fromstring("<div><script>var x = 1;</script><p>Text</p></div>")
    assert extract_text(dom) == "Text"
```


# LLM-generated content at query #14
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello world</p>")
    assert extract_text_array(dom) == ["Hello world"]
    
    # Test with inline element
    dom = etree.fromstring("<p><span>Hello</span> world</p>")
    assert extract_text_array(dom) == [None, "Hello", " world", None]
    
    # Test with separator (br)
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert True in result  # Should contain separator marker
    
    # Test with nested inline elements
    dom = etree.fromstring("<p><strong>Bold</strong> and <em>italic</em></p>")
    result = extract_text_array(dom)
    assert result.count(None) == 2  # Two block-level markers
    
    # Test with empty element
    dom = etree.fromstring("<div></div>")
    assert extract_text_array(dom) == [None, None]
    
    # Test with text only
    dom = etree.fromstring("<div>Simple text</div>")
    assert extract_text_array(dom) == [None, "Simple text", None]
    
    # Test with multiple children
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, None, "Second", None, None]
    
    # Test with tail text
    dom = etree.fromstring("<p>Text before <b>bold</b> text after</p>")
    assert extract_text_array(dom) == [None, "Text before ", "bold", " text after", None]
    
    # Test with squashing artificial newlines
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    # Should have squashed consecutive None values
    none_count = sum(1 for x in result if x is None)
    assert none_count <= 3  # Squashed from original 4-5 to fewer
    
    # Test with stripping artificial newlines
    dom = etree.fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["A"]
    
    # Test with callable tag (should return empty string)
    dom = etree.fromstring("<div></div>")
    dom.tag = lambda: None  # Make tag callable
    assert extract_text_array(dom) == ''
    
    # Test with deep nesting
    dom = etree.fromstring("<div><p><span>Deep <b>nesting</b></span></p></div>")
    result = extract_text_array(dom)
    assert "Deep " in result
    assert "nesting" in result
    
    # Test with multiple separators
    dom = etree.fromstring("<div>Text<br/>More<br/>End</div>")
    result = extract_text_array(dom)
    true_count = sum(1 for x in result if x is True)
    assert true_count == 2  # Two br elements
    
    # Test without squashing
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in result  # Should have None markers
    assert result.count(None) == 4  # Two from each p tag
    
    # Test with attributes (should not affect)
    dom = etree.fromstring('<p class="test">Text</p>')
    assert extract_text_array(dom) == ["Text"]


# LLM-generated content at query #15
#--------------------------

```python
def test_extract_text():
    # Test basic inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with block elements (should add newlines)
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with separator elements (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with nested elements
    doc = html.fromstring("<div><p>Text <b>bold</b> and <i>italic</i></p></div>")
    assert extract_text(doc) == "Text bold and italic"
    
    # Test empty element
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test with whitespace
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with custom block symbol
    doc = html.fromstring("<div><p>A</p><p>B</p></div>")
    assert extract_text(doc, block_symbol=' ') == "A B"
    
    # Test with custom separator symbol
    doc = html.fromstring("<p>A<br>B</p>")
    assert extract_text(doc, sep_symbol=' ') == "A B"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello   World  "
    
    # Test with text after element
    doc = html.fromstring("<p>Text <b>bold</b> after</p>")
    assert extract_text(doc) == "Text bold after"
    
    # Test with multiple nested blocks
    doc = html.fromstring("<div><p>A</p><div><p>B</p><p>C</p></div><p>D</p></div>")
    assert extract_text(doc) == "A\nB\nC\nD"
    
    # Test with pre element (should preserve whitespace - currently not implemented in extract_text_array)
    # This tests the current behavior
    doc = html.fromstring("<pre>  Hello   World  </pre>")
    result = extract_text(doc)
    assert result == "Hello World"  # Current behavior squashes whitespace
    
    # Test with script/style elements
    doc = html.fromstring("<div><script>var x = 1;</script>Text</div>")
    assert extract_text(doc) == "Text"
    
    # Test with deeply nested structure
    doc = html.fromstring("<div><p><span><b>Deep</b></span></p></div>")
    assert extract_text(doc) == "Deep"
```


# LLM-generated content at query #16
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello World", None], f"Expected [None, 'Hello World', None] but got {result}"
    
    # Test with inline element
    dom = etree.fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"], f"Expected ['Hello'] but got {result}"
    
    # Test with separator (br)
    dom = etree.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True] but got {result}"
    
    # Test with nested elements
    dom = etree.fromstring("<p>Hello <b>World</b>!</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "World", "!", None], f"Expected [None, 'Hello ', 'World', '!', None] but got {result}"
    
    # Test with multiple inline elements
    dom = etree.fromstring("<p><span>Hello</span> <span>World</span></p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", " ", "World", None], f"Expected [None, 'Hello', ' ', 'World', None] but got {result}"
    
    # Test with nested block elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    expected = [None, None, "First", None, None, None, "Second", None, None]
    assert result == expected, f"Expected {expected} but got {result}"
    
    # Test with element having text and tail
    dom = etree.fromstring("<p>Hello <b>World</b> again</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "World", " again", None], f"Expected [None, 'Hello ', 'World', ' again', None] but got {result}"
    
    # Test with empty element
    dom = etree.fromstring("<p></p>")
    result = extract_text_array(dom)
    assert result == [None, None], f"Expected [None, None] but got {result}"
    
    # Test with element containing only whitespace
    dom = etree.fromstring("<p>   </p>")
    result = extract_text_array(dom)
    assert result == [None, "   ", None], f"Expected [None, '   ', None] but got {result}"
    
    # Test squash_artifical_nl=False
    dom = etree.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    expected = [None, None, "Text", None, None, None]
    assert result == expected, f"Expected {expected} but got {result}"
    
    # Test strip_artifical_nl=False
    dom = etree.fromstring("<p>Text</p>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Text", None], f"Expected [None, 'Text', None] but got {result}"
    
    # Test with callable tag (should return empty string)
    dom = etree.fromstring("<p>Text</p>")
    dom.tag = lambda: None  # Simulate callable tag
    result = extract_text_array(dom)
    assert result == "", f"Expected empty string but got {result}"


# LLM-generated content at query #17
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple text node
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 2: Nested inline elements
    dom = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test 3: Block elements create newlines
    dom = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test 4: Separator elements (br)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test 5: Multiple whitespace is squashed
    dom = html.fromstring("<p>Hello    World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 6: Whitespace around elements
    dom = html.fromstring("<p>  Hello <b>  bold  </b> world  </p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test 7: Nested block elements
    dom = html.fromstring("<div><p>Paragraph</p><p>Another</p></div>")
    assert extract_text(dom) == "Paragraph\nAnother"
    
    # Test 8: Custom separators
    dom = html.fromstring("<p>Hello<br>World</p>")
    assert extract_text(dom, sep_symbol=" | ") == "Hello | World"
    
    # Test 9: Custom block symbol
    dom = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text(dom, block_symbol=" | ") == "First | Second"
    
    # Test 10: Empty elements
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test 11: Deeply nested structure
    dom = html.fromstring("<div><p><span>Deep <b>text</b></span></p></div>")
    assert extract_text(dom) == "Deep text"
    
    # Test 12: Multiple separators
    dom = html.fromstring("<p>Line1<br><br>Line2</p>")
    assert extract_text(dom) == "Line1\n\nLine2"
    
    # Test 13: Text with newlines
    dom = html.fromstring("<p>Hello\nWorld</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 14: Inline elements don't create newlines
    dom = html.fromstring("<span>Hello</span><span>World</span>")
    assert extract_text(dom) == "HelloWorld"
    
    # Test 15: Mixed inline and block with text
    dom = html.fromstring("<p>Some text <a href='#'>link</a> more text</p>")
    assert extract_text(dom) == "Some text link more text"


# LLM-generated content at query #18
#--------------------------

```python
def test_extract_text():
    # Setup a simple DOM structure for testing
    from lxml import html
    
    # Test 1: Simple text extraction
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 2: Text with inline tags
    dom = html.fromstring("<p>Hello <strong>World</strong></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 3: Block level elements should create newlines
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test 4: Separator tags (br) should create newlines
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test 5: Nested inline elements
    dom = html.fromstring("<p><span>Text <em>emphasized</em> end</span></p>")
    assert extract_text(dom) == "Text emphasized end"
    
    # Test 6: Multiple whitespace should be squashed
    dom = html.fromstring("<p>Hello    World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 7: Leading/trailing whitespace should be stripped
    dom = html.fromstring("<p>   Hello World   </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 8: Empty content
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test 9: Mixed inline and block elements
    dom = html.fromstring("<div><h1>Title</h1><p>Content</p></div>")
    assert extract_text(dom) == "Title\nContent"
    
    # Test 10: Custom sep_symbol
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol='|') == "Line1|Line2"
    
    # Test 11: Custom block_symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test 12: Without squashing spaces
    dom = html.fromstring("<p>Hello    World</p>")
    assert extract_text(dom, squash_space=False) == "Hello    World"
    
    # Test 13: Complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>First <strong>paragraph</strong></p>
            <p>Second paragraph with <br> line break</p>
        </div>
    """)
    result = extract_text(dom)
    assert "Title" in result
    assert "First paragraph" in result
    assert "Second paragraph with" in result
    assert result.count('\n') >= 2  # At least two newlines
    
    # Test 14: Script tags should be excluded
    dom = html.fromstring("<p>Text <script>alert('test')</script> more</p>")
    assert extract_text(dom) == "Text  more"
    
    # Test 15: Multiple separator tags
    dom = html.fromstring("<p>Line1<br><br>Line2</p>")
    result = extract_text(dom)
    assert "Line1" in result
    assert "Line2" in result
    
    # Test 16: Text with special whitespace characters
    dom = html.fromstring("<p>Hello\u200BWorld</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 17: Deeply nested inline tags
    dom = html.fromstring("<p><span><em><strong>Deep</strong></em></span></p>")
    assert extract_text(dom) == "Deep"
    
    # Test 18: Mixed content with tail text
    dom = html.fromstring("<p>Start <b>bold</b> middle <i>italic</i> end</p>")
    assert extract_text(dom) == "Start bold middle italic end"
    
    # Test 19: Empty DOM
    dom = html.fromstring("<html></html>")
    assert extract_text(dom) == ""
    
    # Test 20: Block element with no text
    dom = html.fromstring("<div><p></p><p></p></div>")
    assert extract_text(dom) == ""
```


# LLM-generated content at query #19
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml import etree
    dom = etree.fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with separator tag (br)
    dom = etree.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with block element
    dom = etree.fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result == [None, "Text", None]
    
    # Test with nested elements
    dom = etree.fromstring("<div><span>Hello</span> <span>World</span></div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", " ", "World", None]
    
    # Test with separator in block
    dom = etree.fromstring("<div>Line1<br/>Line2</div>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None]
    
    # Test with empty element
    dom = etree.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with text only
    dom = etree.fromstring("<p>Simple text</p>")
    result = extract_text_array(dom)
    assert result == [None, "Simple text", None]
    
    # Test with multiple children
    dom = etree.fromstring("<div><b>Bold</b><i>Italic</i></div>")
    result = extract_text_array(dom)
    assert result == [None, "Bold", "Italic", None]
    
    # Test squash_artifical_nl=False
    dom = etree.fromstring("<div><p>Para1</p><p>Para2</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, "Para1", None, None, "Para2", None, None]
    
    # Test strip_artifical_nl=False
    dom = etree.fromstring("<div>Text</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Text", None]
    
    # Test with callable tag
    dom = etree.fromstring("<div><script>function()</script></div>")
    # script tag is callable, should return empty string
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with inline elements inside block
    dom = etree.fromstring("<div><span>inline</span> text</div>")
    result = extract_text_array(dom)
    assert result == [None, "inline", " text", None]
    
    # Test with multiple separators
    dom = etree.fromstring("<div><br/><br/></div>")
    result = extract_text_array(dom)
    assert result == [None, True, True, None]


# LLM-generated content at query #20
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from lxml import html
    doc = html.fromstring("<p>Hello <b>world</b></p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with separator element (br)
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with block elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with nested elements
    doc = html.fromstring("<div><p>Text with <span>span</span> inside</p></div>")
    assert extract_text(doc) == "Text with span inside"
    
    # Test with whitespace
    doc = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with multiple whitespace characters
    doc = html.fromstring("<p>Hello\t\tworld</p>")
    assert extract_text(doc) == "Hello world"
    
    # Test with empty element
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with only text content
    doc = html.fromstring("Just text")
    assert extract_text(doc) == "Just text"
    
    # Test with custom block_symbol
    doc = html.fromstring("<p>First</p><p>Second</p>")
    assert extract_text(doc, block_symbol=' | ') == "First | Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc, sep_symbol=' - ') == "Line1 - Line2"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello   world  "
    
    # Test with complex nested structure
    doc = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Paragraph with <a href="#">link</a> and <br/> break</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    """)
    result = extract_text(doc)
    assert "Title" in result
    assert "Paragraph with link and" in result
    assert "break" in result
    assert "Item 1" in result
    assert "Item 2" in result
    
    # Test with inline elements that shouldn't add newlines
    doc = html.fromstring("<span>inline</span> <strong>text</strong>")
    assert extract_text(doc) == "inline text"
    
    # Test with multiple br elements
    doc = html.fromstring("<p>Line1<br/><br/>Line2</p>")
    assert extract_text(doc) == "Line1\n\nLine2"```


# LLM-generated content at query #21
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text node
    from lxml import etree
    dom = etree.fromstring("<p>Hello world</p>")
    assert extract_text_array(dom) == [None, "Hello world", None]
    
    # Test with inline tag
    dom = etree.fromstring("<span>inline text</span>")
    assert extract_text_array(dom) == ["inline text"]
    
    # Test with separator tag
    dom = etree.fromstring("<br/>")
    assert extract_text_array(dom) == [True]
    
    # Test with nested inline tags
    dom = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(dom)
    assert None in result
    assert "Hello " in result
    assert "bold" in result
    assert " world" in result
    
    # Test with multiple levels of nesting
    dom = etree.fromstring("<div><p>First <span>nested</span> text</p></div>")
    result = extract_text_array(dom)
    assert len(result) > 0
    assert any(isinstance(x, str) for x in result)
    
    # Test with empty element
    dom = etree.fromstring("<div></div>")
    assert extract_text_array(dom) == [None, None]
    
    # Test with text and tail
    dom = etree.fromstring("<p>Text<em>emphasized</em>tail</p>")
    result = extract_text_array(dom)
    assert "Text" in result
    assert "emphasized" in result
    assert "tail" in result
    
    # Test with squash_artifical_nl=False
    dom = etree.fromstring("<p>Text</p>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result[0] is None  # opening artificial newline
    assert "Text" in result
    assert result[-1] is None  # closing artificial newline
    
    # Test with strip_artifical_nl=False
    dom = etree.fromstring("<p>Text</p>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # opening artificial newline not stripped
    assert "Text" in result
    assert result[-1] is None  # closing artificial newline not stripped
    
    # Test with callable tag
    class MockElement:
        tag = lambda: None
    assert extract_text_array(MockElement()) == ''


# LLM-generated content at query #22
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with nested inline elements
    dom = html.fromstring("<p>Hello <strong>World</strong></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator elements (br)
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with multiple consecutive br elements
    dom = html.fromstring("<p>Line1<br/><br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with whitespace squashing
    dom = html.fromstring("<p>  Hello    World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with mixed inline and block elements
    dom = html.fromstring("<div><h1>Title</h1><p>Paragraph</p></div>")
    assert extract_text(dom) == "Title\nParagraph"
    
    # Test with empty elements
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with nested block elements
    dom = html.fromstring("<div><section><p>Text</p></section></div>")
    assert extract_text(dom) == "Text"
    
    # Test with custom block_symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>  Hello  World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello  World  "
    
    # Test with complex nested structure
    dom = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>First <strong>bold</strong> text</p>
            <p>Second line<br/>with break</p>
        </div>
    """)
    expected = "Title\nFirst bold text\nSecond line\nwith break"
    assert extract_text(dom) == expected
    
    # Test with span elements (inline)
    dom = html.fromstring("<p><span>Hello</span> <span>World</span></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with text before and after child elements
    dom = html.fromstring("<p>Before <strong>middle</strong> After</p>")
    assert extract_text(dom) == "Before middle After"
    
    # Test with tail text
    dom = html.fromstring("<div><p>First</p>Between<p>Second</p></div>")
    assert extract_text(dom) == "First\nBetween\nSecond"


# LLM-generated content at query #23
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    from html import HTML
    dom = HTML().p("Hello world")
    assert extract_text(dom) == "Hello world"
    
    # Test with nested inline elements
    dom = HTML().p("Hello ", HTML().b("bold"), " world")
    assert extract_text(dom) == "Hello bold world"
    
    # Test with block elements
    dom = HTML().div(HTML().p("First"), HTML().p("Second"))
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator elements (br)
    dom = HTML().p("Line1", HTML().br(), "Line2")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with custom block_symbol
    dom = HTML().div(HTML().p("A"), HTML().p("B"))
    assert extract_text(dom, block_symbol=' ') == "A B"
    
    # Test with custom sep_symbol
    dom = HTML().p("A", HTML().br(), "B")
    assert extract_text(dom, sep_symbol=' ') == "A B"
    
    # Test with squash_space=False
    dom = HTML().p("  Hello   world  ")
    assert extract_text(dom, squash_space=False) == "Hello world"
    
    # Test with complex nested structure
    dom = HTML().div(
        HTML().h1("Title"),
        HTML().p("Paragraph with ", HTML().a("link", href="#"), " text"),
        HTML().br(),
        HTML().span("Inline span")
    )
    result = extract_text(dom)
    assert "Title" in result
    assert "Paragraph with link text" in result
    assert "Inline span" in result
    
    # Test with empty dom
    dom = HTML().div()
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = HTML().p("   ")
    assert extract_text(dom) == ""
    
    # Test with multiple consecutive block elements
    dom = HTML().div(HTML().p("A"), HTML().p("B"), HTML().p("C"))
    assert extract_text(dom) == "A\nB\nC"
    
    # Test with mix of inline, block and separator elements
    dom = HTML().div(
        HTML().span("Start"),
        HTML().br(),
        HTML().p("Block"),
        HTML().br(),
        HTML().span("End")
    )
    result = extract_text(dom)
    assert result.startswith("Start")
    assert "Block" in result
    assert "End" in result
    
    # Test with non-inline elements that are not block (like script, style)
    dom = HTML().div(HTML().script("alert('test')"), HTML().p("Content"))
    assert "alert" not in extract_text(dom)
    assert "Content" in extract_text(dom) or "Content" in extract_text(dom)
    
    # Test stripping leading and trailing whitespace
    dom = HTML().p("  Hello  ")
    assert extract_text(dom) == "Hello"
    
    # Test multiple spaces collapsed
    dom = HTML().p("Hello    World")
    assert extract_text(dom) == "Hello World"


# LLM-generated content at query #24
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple text extraction
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 2: Nested inline elements
    dom = html.fromstring("<p>Hello <b>bold</b> World</p>")
    assert extract_text(dom) == "Hello bold World"
    
    # Test 3: Multiple block elements
    dom = html.fromstring("<p>First</p><p>Second</p>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test 4: Separator elements (br)
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test 5: Nested block elements
    dom = html.fromstring("<div><p>Inside</p></div>")
    assert extract_text(dom) == "Inside"
    
    # Test 6: Whitespace handling
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 7: Multiple whitespace and newlines in source
    dom = html.fromstring("<p>Hello\n\nWorld</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 8: Empty elements
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test 9: Deeply nested structure
    dom = html.fromstring("<div><span><b>Deep</b></span></div>")
    assert extract_text(dom) == "Deep"
    
    # Test 10: Mixed inline and block
    dom = html.fromstring("<div><h1>Title</h1><p>Content with <a>link</a></p></div>")
    assert extract_text(dom) == "Title\nContent with link"
    
    # Test 11: Custom block_symbol
    dom = html.fromstring("<p>First</p><p>Second</p>")
    assert extract_text(dom, block_symbol='|') == "First|Second"
    
    # Test 12: Custom sep_symbol
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol='|') == "Line1|Line2"
    
    # Test 13: Disabling squash_space
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test 14: Multiple separators
    dom = html.fromstring("<p>One<br/>Two<br/>Three</p>")
    assert extract_text(dom) == "One\nTwo\nThree"
    
    # Test 15: Text in different inline tags
    dom = html.fromstring("<p><em>em</em> <strong>strong</strong> <code>code</code></p>")
    assert extract_text(dom) == "em strong code"
    
    # Test 16: Script tag should be treated as inline
    dom = html.fromstring("<p>Text <script>var x=1;</script> more</p>")
    assert extract_text(dom) == "Text var x=1; more"
    
    # Test 17: Multiple block elements with text
    dom = html.fromstring("<div><h1>Title</h1><p>Para</p><br/><span>Span</span></div>")
    assert extract_text(dom) == "Title\nPara\nSpan"
    
    # Test 18: Whitespace only content
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test 19: Complex nesting with multiple text nodes
    dom = html.fromstring("<div>Start <p>Middle <b>bold</b> text</p> End</div>")
    assert extract_text(dom) == "Start Middle bold text End"
    
    # Test 20: Consecutive block elements with no text
    dom = html.fromstring("<div><p></p><p></p></div>")
    assert extract_text(dom) == ""
```


# LLM-generated content at query #25
#--------------------------

```python
def test_extract_text():
    # Test with simple text
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with inline elements
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator elements
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested elements
    dom = html.fromstring("<div><p>Text with <b>bold</b> and <i>italic</i></p></div>")
    assert extract_text(dom) == "Text with bold and italic"
    
    # Test with whitespace
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with multiple whitespace characters
    dom = html.fromstring("<p>Hello\t\tWorld</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with empty element
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with only text
    dom = html.fromstring("Just text")
    assert extract_text(dom) == "Just text"
    
    # Test with custom block_symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "


# LLM-generated content at query #26
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    from lxml import html
    doc = html.fromstring("<p>Hello World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test inline elements
    doc = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(doc) == "Hello World"
    
    # Test block elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test separator elements (br)
    doc = html.fromstring("<p>First<br/>Second</p>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test nested elements
    doc = html.fromstring("<div><p>Hello <b>bold</b> world</p></div>")
    assert extract_text(doc) == "Hello bold world"
    
    # Test multiple block elements with text
    doc = html.fromstring("<div><p>One</p><p>Two</p><p>Three</p></div>")
    assert extract_text(doc) == "One\nTwo\nThree"
    
    # Test with whitespace
    doc = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with multiple whitespace and newlines
    doc = html.fromstring("<p>\n  Hello   \n  World  \n</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test empty element
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test element with only nested elements
    doc = html.fromstring("<div><p>Text</p></div>")
    assert extract_text(doc) == "Text"
    
    # Test mixed content
    doc = html.fromstring("<p>Start <b>middle</b> end</p>")
    assert extract_text(doc) == "Start middle end"
    
    # Test with custom block_symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=" ") == "First Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>First<br/>Second</p>")
    assert extract_text(doc, sep_symbol=" ") == "First Second"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Hello   World  </p>")
    result = extract_text(doc, squash_space=False)
    assert result == "  Hello   World  "
    
    # Test nested inline elements
    doc = html.fromstring("<p><b><i>Bold Italic</i></b></p>")
    assert extract_text(doc) == "Bold Italic"
    
    # Test elements with tails
    doc = html.fromstring("<p>Hello<b>bold</b>world</p>")
    assert extract_text(doc) == "Helloboldworld"
    
    # Test complex nested structure
    html_str = """
    <div>
        <h1>Title</h1>
        <p>First paragraph with <b>bold</b> text</p>
        <p>Second paragraph<br/>with line break</p>
    </div>
    """
    doc = html.fromstring(html_str)
    expected = "Title\nFirst paragraph with bold text\nSecond paragraph\nwith line break"
    assert extract_text(doc) == expected
    
    # Test with callable tag (should return empty string)
    class MockElement:
        tag = lambda: None
    mock = MockElement()
    assert extract_text(mock) == ""
```


# LLM-generated content at query #27
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple text node
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 2: Nested inline elements
    dom = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(dom) == "Hello bold world"
    
    # Test 3: Block elements creating newlines
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom)
    assert result == "First\nSecond"
    
    # Test 4: Separator elements (br)
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    result = extract_text(dom)
    assert result == "Line1\nLine2"
    
    # Test 5: Nested block elements
    dom = html.fromstring("<div><div><p>Deep</p></div></div>")
    assert extract_text(dom) == "Deep"
    
    # Test 6: Empty elements
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test 7: Whitespace handling
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 8: Multiple whitespace characters
    dom = html.fromstring("<p>Hello\t\nWorld</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 9: Mixed inline and block elements
    dom = html.fromstring("<div><span>Inline</span><p>Block</p></div>")
    result = extract_text(dom)
    assert result == "Inline\nBlock"
    
    # Test 10: Custom block and sep symbols
    dom = html.fromstring("<div><p>First</p><br><p>Second</p></div>")
    result = extract_text(dom, block_symbol=" | ", sep_symbol=" - ")
    assert result == "First - Second"
    
    # Test 11: Multiple nested block elements
    dom = html.fromstring("<div><p>First</p><div><p>Second</p></div><p>Third</p></div>")
    result = extract_text(dom)
    assert result == "First\nSecond\nThird"
    
    # Test 12: Elements with tail text
    dom = html.fromstring("<p>Hello <b>bold</b> after bold</p>")
    assert extract_text(dom) == "Hello bold after bold"
    
    # Test 13: Complex nested structure
    dom = html.fromstring("<div><p>Para <span>span <b>bold</b></span> end</p><p>Second para</p></div>")
    result = extract_text(dom)
    assert result == "Para span bold end\nSecond para"
    
    # Test 14: Multiple inline elements
    dom = html.fromstring("<p><i>italic</i> and <u>underline</u></p>")
    assert extract_text(dom) == "italic and underline"
    
    # Test 15: Leading/trailing whitespace in text
    dom = html.fromstring("<p>  Leading and trailing  </p>")
    assert extract_text(dom) == "Leading and trailing"
```


# LLM-generated content at query #28
#--------------------------

```python
def test_extract_text_array():
    # Test simple text node
    from lxml import html
    doc = html.fromstring("<p>Hello world</p>")
    result = extract_text_array(doc)
    assert result == [None, "Hello world", None]

    # Test inline element
    doc = html.fromstring("<span>inline</span>")
    result = extract_text_array(doc)
    assert result == ["inline"]

    # Test separator element
    doc = html.fromstring("<br/>")
    result = extract_text_array(doc)
    assert result == [True]

    # Test nested elements
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(doc)
    expected = [None, None, "First", None, None, None, "Second", None, None, None]
    assert result == expected

    # Test with tail text
    doc = html.fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text_array(doc)
    expected = [None, "Hello ", "bold", " world", None]
    assert result == expected

    # Test squash_artifical_nl=False
    doc = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(doc, squash_artifical_nl=False)
    assert result == [None, None, "Text", None, None]

    # Test strip_artifical_nl=False
    doc = html.fromstring("<p>Text</p>")
    result = extract_text_array(doc, strip_artifical_nl=False)
    assert result == [None, "Text", None]

    # Test with callable tag (should return empty string)
    doc = html.fromstring("<div></div>")
    doc.tag = lambda: None
    result = extract_text_array(doc)
    assert result == ""

    # Test empty element
    doc = html.fromstring("<div></div>")
    result = extract_text_array(doc)
    assert result == [None, None]

    # Test mixed inline and block elements
    doc = html.fromstring("<div><span>inline</span><p>block</p></div>")
    result = extract_text_array(doc)
    expected = [None, "inline", None, "block", None, None]
    assert result == expected

    # Test multiple separators
    doc = html.fromstring("<div><br/><br/></div>")
    result = extract_text_array(doc)
    expected = [None, True, True, None]
    assert result == expected


# LLM-generated content at query #29
#--------------------------

```python
def test_extract_text_array():
    # Test empty dom
    class EmptyDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(EmptyDom()) == []
    assert extract_text_array(EmptyDom(), squash_artifical_nl=False, strip_artifical_nl=False) == [None, None]

    # Test inline tag
    class InlineDom:
        tag = 'span'
        text = 'hello'
        def getchildren(self):
            return []
    assert extract_text_array(InlineDom()) == ['hello']

    # Test separator tag (br)
    class BrDom:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(BrDom()) == [True]

    # Test block tag
    class BlockDom:
        tag = 'div'
        text = 'text'
        def getchildren(self):
            return []
    assert extract_text_array(BlockDom()) == ['text']
    assert extract_text_array(BlockDom(), squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'text', None]

    # Test nested structure
    class ChildDom:
        tag = 'span'
        text = 'child'
        tail = ' tail'
        def getchildren(self):
            return []
    
    class ParentDom:
        tag = 'div'
        text = 'parent '
        def getchildren(self):
            return [ChildDom()]
    
    result = extract_text_array(ParentDom())
    assert result == ['parent ', 'child', ' tail']

    # Test multiple children
    class Child1:
        tag = 'span'
        text = 'first'
        tail = ' '
        def getchildren(self):
            return []
    
    class Child2:
        tag = 'b'
        text = 'second'
        tail = None
        def getchildren(self):
            return []
    
    class MultiDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return [Child1(), Child2()]
    
    result = extract_text_array(MultiDom())
    assert result == ['first', ' ', 'second']

    # Test squash_artifical_nl
    class NestedBlock:
        tag = 'div'
        text = 'outer'
        def getchildren(self):
            inner = type('InnerBlock', (), {'tag': 'p', 'text': 'inner', 'getchildren': lambda self: [], 'tail': None})()
            return [inner]
    
    result = extract_text_array(NestedBlock(), squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == ['outer', None, 'inner']

    # Test strip_artifical_nl
    result = extract_text_array(NestedBlock(), squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['outer', 'inner']

    # Test callable tag returns empty string
    class CallableDom:
        tag = lambda: None
        text = 'should not appear'
        def getchildren(self):
            return []
    assert extract_text_array(CallableDom()) == ''


# LLM-generated content at query #30
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline element
    from lxml.html import fromstring
    dom = fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]

    # Test with block element
    dom = fromstring("<div>Hello World</div>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]

    # Test with separator element (br)
    dom = fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]

    # Test with nested elements
    dom = fromstring("<div><span>Hello</span> <span>World</span></div>")
    result = extract_text_array(dom)
    assert result == ["Hello", " ", "World"]

    # Test with block element containing inline elements
    dom = fromstring("<p>Hello <b>World</b></p>")
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]

    # Test with multiple block elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", None, "Second"]

    # Test with separator between text
    dom = fromstring("Line1<br/>Line2")
    result = extract_text_array(dom)
    assert result == ["Line1", True, "Line2"]

    # Test with empty element
    dom = fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []

    # Test with text only
    dom = fromstring("Just text")
    result = extract_text_array(dom)
    assert result == ["Just text"]

    # Test with deep nesting
    dom = fromstring("<div><p><span>Deep</span></p><p><span>Nested</span></p></div>")
    result = extract_text_array(dom)
    assert result == ["Deep", None, "Nested"]

    # Test with None text and tail
    dom = fromstring("<div><b></b>Tail text</div>")
    result = extract_text_array(dom)
    assert result == ["Tail text"]

    # Test with callable tag (should return empty string)
    class MockCallable:
        tag = lambda: None
    
    mock_dom = MockCallable()
    result = extract_text_array(mock_dom)
    assert result == ['']


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
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
    assert extract_text_array(elem) == []
    
    # Test with text content
    elem = MockElement('span', text='hello')
    result = extract_text_array(elem)
    assert result == ['hello']
    
    # Test with separator tag (br)
    elem = MockElement('br')
    result = extract_text_array(elem)
    assert result == [True]
    
    # Test with block-level tag (div)
    elem = MockElement('div')
    result = extract_text_array(elem)
    assert result == [None, None]
    
    # Test nested structure
    child = MockElement('span', text='world', tail='!')
    parent = MockElement('div', text='hello ', children=[child])
    result = extract_text_array(parent)
    assert result == [None, 'hello ', 'world', '!', None]
    
    # Test with squash_artifical_nl=True
    elem = MockElement('div', text='hello')
    result = extract_text_array(elem, squash_artifical_nl=True)
    assert result == ['hello']
    
    # Test with strip_artifical_nl=True
    elem = MockElement('div')
    result = extract_text_array(elem, strip_artifical_nl=True)
    assert result == []
    
    # Test inline tag without text
    elem = MockElement('span')
    result = extract_text_array(elem)
    assert result == []
    
    # Test multiple children
    child1 = MockElement('b', text='bold')
    child2 = MockElement('i', text='italic', tail=' ')
    parent = MockElement('p', children=[child1, child2])
    result = extract_text_array(parent)
    assert result == [None, 'bold', 'italic', ' ', None]
    
    # Test callable tag
    class CallableTag:
        def __call__(self):
            pass
        def getchildren(self):
            return []
    
    elem = MockElement(CallableTag())
    assert extract_text_array(elem) == ''


# LLM-generated content at query #2
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline text
    from html.parser import HTMLParser
    import xml.etree.ElementTree as ET
    
    # Test 1: Simple text node
    dom = ET.fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello World", None], f"Expected [None, 'Hello World', None], got {result}"
    
    # Test 2: Inline tag (span)
    dom = ET.fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"], f"Expected ['Hello'], got {result}"
    
    # Test 3: Separator tag (br)
    dom = ET.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"
    
    # Test 4: Nested inline tags
    dom = ET.fromstring("<p>Hello <b>World</b>!</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "World", "!", None], f"Expected [None, 'Hello ', 'World', '!', None], got {result}"
    
    # Test 5: Nested block tags
    dom = ET.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    expected = [None, None, "First", None, None, None, "Second", None, None]
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test 6: Tag with tail text
    dom = ET.fromstring("<p>Hello<br/>World</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", True, "World", None], f"Expected [None, 'Hello', True, 'World', None], got {result}"
    
    # Test 7: Empty element
    dom = ET.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None], f"Expected [None, None], got {result}"
    
    # Test 8: Text only in tail
    dom = ET.fromstring("<div><span>Hello</span> World</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", " World", None], f"Expected [None, 'Hello', ' World', None], got {result}"
    
    # Test 9: Deeply nested structure
    dom = ET.fromstring("<div><p>Text with <b>bold</b> and <i>italic</i></p></div>")
    result = extract_text_array(dom)
    expected = [None, None, "Text with ", "bold", " and ", "italic", None, None]
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test 10: Multiple separators
    dom = ET.fromstring("<p>Line1<br/>Line2<br/>Line3</p>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", True, "Line3", None], f"Expected [None, 'Line1', True, 'Line2', True, 'Line3', None], got {result}"
    
    # Test 11: Custom tag (not in INLINE_TAGS)
    dom = ET.fromstring("<custom>Content</custom>")
    result = extract_text_array(dom)
    assert result == [None, "Content", None], f"Expected [None, 'Content', None], got {result}"
    
    # Test 12: Squash artificial newlines
    dom = ET.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == [None, "A", None, "B", None], f"Expected [None, 'A', None, 'B', None], got {result}"
    
    # Test 13: Strip artificial newlines
    dom = ET.fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["A"], f"Expected ['A'], got {result}"
    
    # Test 14: Both squash and strip
    dom = ET.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["A", None, "B"], f"Expected ['A', None, 'B'], got {result}"
    
    # Test 15: Callable tag (should return empty string)
    class MockCallable:
        def __call__(self):
            pass
    
    mock_dom = type('MockDom', (), {'tag': MockCallable()})()
    result = extract_text_array(mock_dom)
    assert result == '', f"Expected '', got {result}"


# LLM-generated content at query #3
#--------------------------

```python
def test_extract_text_array():
    # Test with simple inline tag
    from lxml import html
    doc = html.fromstring("<span>Hello World</span>")
    assert extract_text_array(doc) == ["Hello World"]
    
    # Test with block tag
    doc = html.fromstring("<div>Hello World</div>")
    assert extract_text_array(doc) == [None, "Hello World", None]
    
    # Test with separator
    doc = html.fromstring("<br>")
    assert extract_text_array(doc) == [True]
    
    # Test with nested tags
    doc = html.fromstring("<div><span>Hello</span> <span>World</span></div>")
    result = extract_text_array(doc)
    assert None in result
    assert "Hello" in result
    assert " " in result
    assert "World" in result
    
    # Test with text and tail
    doc = html.fromstring("<div>Start <span>middle</span> End</div>")
    result = extract_text_array(doc)
    assert "Start " in result
    assert "middle" in result
    assert " End" in result
    
    # Test with empty element
    doc = html.fromstring("<div></div>")
    assert extract_text_array(doc) == [None, None]
    
    # Test with callable tag (should return empty string)
    class FakeElement:
        tag = lambda: None
    assert extract_text_array(FakeElement()) == ""
    
    # Test with multiple separators
    doc = html.fromstring("<br><br>")
    result = extract_text_array(doc)
    assert result == [True, True]
    
    # Test with text only
    doc = html.fromstring("Just text")
    assert extract_text_array(doc) == ["Just text"]
    
    # Test with complex nesting
    doc = html.fromstring("<div><p>Para <b>bold</b> text</p><br/></div>")
    result = extract_text_array(doc)
    assert None in result
    assert "Para " in result
    assert "bold" in result
    assert " text" in result
    assert True in result
    
    # Test squash_artifical_nl parameter
    doc = html.fromstring("<div><p>Test</p></div>")
    result_no_squash = extract_text_array(doc, squash_artifical_nl=False)
    result_squash = extract_text_array(doc, squash_artifical_nl=True)
    assert result_no_squash.count(None) > result_squash.count(None) or result_no_squash == result_squash
    
    # Test strip_artifical_nl parameter
    doc = html.fromstring("<div>Content</div>")
    result_no_strip = extract_text_array(doc, strip_artifical_nl=False)
    result_strip = extract_text_array(doc, strip_artifical_nl=True)
    assert len(result_no_strip) >= len(result_strip)


# LLM-generated content at query #4
#--------------------------

```python
def test_extract_text():
    # Test with simple inline text
    from xml.etree.ElementTree import Element, SubElement, fromstring
    
    # Test 1: Simple text
    dom = fromstring("<p>Hello World</p>")
    result = extract_text(dom)
    assert result == "Hello World"
    
    # Test 2: Nested inline elements
    dom = fromstring("<p>Hello <b>World</b></p>")
    result = extract_text(dom)
    assert result == "Hello World"
    
    # Test 3: Block elements with separators
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom)
    assert result == "First\nSecond"
    
    # Test 4: br separator
    dom = fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text(dom)
    assert result == "Line1\nLine2"
    
    # Test 5: Whitespace handling
    dom = fromstring("<p>  Hello   World  </p>")
    result = extract_text(dom)
    assert result == "Hello World"
    
    # Test 6: Custom separator symbols
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text(dom, block_symbol='|', sep_symbol='-')
    assert result == "A|B"
    
    # Test 7: Disable space squashing
    dom = fromstring("<p>  Hello   World  </p>")
    result = extract_text(dom, squash_space=False)
    assert result == "  Hello   World  "
    
    # Test 8: Mixed inline and block elements
    dom = fromstring("<div><p>Hello <b>beautiful</b> world</p><p>Goodbye</p></div>")
    result = extract_text(dom)
    assert result == "Hello beautiful world\nGoodbye"
    
    # Test 9: Empty element
    dom = fromstring("<p></p>")
    result = extract_text(dom)
    assert result == ""
    
    # Test 10: Element with only whitespace
    dom = fromstring("<p>   </p>")
    result = extract_text(dom)
    assert result == ""
    
    # Test 11: Complex nested structure
    dom = fromstring("""
        <div>
            <p>First paragraph</p>
            <p>Second <span>paragraph</span></p>
            <br/>
            <p>Third</p>
        </div>
    """)
    result = extract_text(dom)
    assert result == "First paragraph\nSecond paragraph\n\nThird"
    
    # Test 12: Multiple br tags
    dom = fromstring("<p>A<br/>B<br/>C</p>")
    result = extract_text(dom)
    assert result == "A\nB\nC"
    
    # Test 13: Nested block elements
    dom = fromstring("<div><div><p>Deep</p></div><p>Shallow</p></div>")
    result = extract_text(dom)
    assert result == "Deep\nShallow"
    
    # Test 14: Preserve inline element content
    dom = fromstring("<p><a>Link</a> and <span>span</span></p>")
    result = extract_text(dom)
    assert result == "Link and span"
    
    # Test 15: Text after closing tags
    dom = fromstring("<p>Hello</p>World<p>After</p>")
    root = Element('root')
    root.append(fromstring("<p>Hello</p>"))
    root.append(fromstring("<p>World</p>"))
    root.append(fromstring("<p>After</p>"))
    result = extract_text(root)
    assert result == "Hello\nWorld\nAfter"
```


# LLM-generated content at query #5
#--------------------------

```python
def test_extract_text():
    # Test with a simple text node
    class MockTextNode:
        def __init__(self, text):
            self.text = text
            self.tag = 'span'
            self.tail = None
            self.getchildren = lambda: []
    
    node = MockTextNode('Hello World')
    assert extract_text(node) == 'Hello World'
    
    # Test with block-level elements (div)
    class MockDiv:
        def __init__(self, children=None):
            self.tag = 'div'
            self.text = None
            self.tail = None
            self.children = children or []
            self.getchildren = lambda: self.children
    
    class MockSpan:
        def __init__(self, text, tail=None):
            self.tag = 'span'
            self.text = text
            self.tail = tail
            self.getchildren = lambda: []
    
    # Test with nested block elements
    child1 = MockSpan('First')
    child2 = MockSpan('Second')
    parent = MockDiv([child1, child2])
    assert extract_text(parent) == 'First\nSecond'
    
    # Test with separator tags (br)
    class MockBr:
        def __init__(self):
            self.tag = 'br'
            self.text = None
            self.tail = None
            self.getchildren = lambda: []
    
    br = MockBr()
    span1 = MockSpan('Before')
    span2 = MockSpan('After')
    parent = MockDiv([span1, br, span2])
    assert extract_text(parent) == 'Before\nAfter'
    
    # Test with multiple whitespace
    span = MockSpan('  Hello   World  ')
    assert extract_text(span) == 'Hello World'
    
    # Test with nested block elements and text
    child = MockSpan('Child')
    parent = MockDiv([MockSpan('Parent '), child])
    assert extract_text(parent) == 'Parent\nChild'
    
    # Test with custom symbols
    assert extract_text(parent, block_symbol=' | ') == 'Parent | Child'
    assert extract_text(parent, sep_symbol=' | ') == 'Parent\nChild'  # separator not affected
    
    # Test with empty element
    empty = MockDiv([])
    assert extract_text(empty) == ''
    
    # Test with text only in tail
    span = MockSpan('', tail=' tail text')
    parent = MockDiv([span])
    assert extract_text(parent) == 'tail text'
    
    # Test with nested elements and tails
    child1 = MockSpan('child1', tail=' tail1')
    child2 = MockSpan('child2', tail=' tail2')
    parent = MockDiv([child1, child2])
    result = extract_text(parent)
    assert 'child1' in result
    assert 'child2' in result
    assert 'tail1' in result
    assert 'tail2' in result
```


# LLM-generated content at query #6
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple text without any tags
    class SimpleText:
        tag = 'span'
        text = 'Hello World'
        def getchildren(self):
            return []
    assert extract_text(SimpleText()) == 'Hello World'
    
    # Test 2: Nested inline tags
    class NestedInline:
        tag = 'span'
        text = 'Hello '
        def getchildren(self):
            child = SimpleText()
            child.text = 'World'
            return [child]
        @property
        def tail(self):
            return None
    assert extract_text(NestedInline()) == 'Hello World'
    
    # Test 3: Block elements should add newlines
    class BlockElement:
        tag = 'div'
        text = 'First'
        def getchildren(self):
            return []
        @property
        def tail(self):
            return None
    class TwoBlocks:
        tag = 'body'
        text = None
        def getchildren(self):
            block1 = BlockElement()
            block2 = BlockElement()
            block2.text = 'Second'
            return [block1, block2]
        @property
        def tail(self):
            return None
    assert extract_text(TwoBlocks()) == 'First\nSecond'
    
    # Test 4: Separator tags (br) should add sep_symbol
    class BrTag:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
        @property
        def tail(self):
            return None
    class TextWithBr:
        tag = 'span'
        text = 'Line1'
        def getchildren(self):
            return [BrTag()]
        @property
        def tail(self):
            return None
    assert extract_text(TextWithBr()) == 'Line1\n'
    
    # Test 5: Whitespace collapsing
    class WhitespaceText:
        tag = 'span'
        text = 'Hello    World'
        def getchildren(self):
            return []
    assert extract_text(WhitespaceText()) == 'Hello World'
    
    # Test 6: Empty text
    class EmptyTag:
        tag = 'div'
        text = ''
        def getchildren(self):
            return []
    assert extract_text(EmptyTag()) == ''
    
    # Test 7: None text
    class NoneText:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    assert extract_text(NoneText()) == ''
    
    # Test 8: Multiple nested levels
    class DeepNested:
        tag = 'div'
        text = None
        def getchildren(self):
            inner = type('Inner', (), {
                'tag': 'p',
                'text': 'Middle',
                'getchildren': lambda self: [],
                'tail': None
            })()
            return [inner]
        @property
        def tail(self):
            return None
    assert extract_text(DeepNested()) == 'Middle'
    
    # Test 9: Custom symbols
    class CustomSymbols:
        tag = 'div'
        text = 'A'
        def getchildren(self):
            br = BrTag()
            return [br]
        @property
        def tail(self):
            return None
    assert extract_text(CustomSymbols(), block_symbol='|', sep_symbol='-') == 'A-'
    
    # Test 10: Squash space disabled
    class NoSquash:
        tag = 'span'
        text = 'Hello    World'
        def getchildren(self):
            return []
    assert extract_text(NoSquash(), squash_space=False) == 'Hello    World'
```


# LLM-generated content at query #7
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from xml.etree.ElementTree import Element, SubElement
    from html import escape
    
    # Test 1: Simple text element
    div = Element('div')
    div.text = "Hello World"
    result = extract_text_array(div)
    assert result == [None, "Hello World", None], f"Expected [None, 'Hello World', None], got {result}"
    
    # Test 2: Inline element without separators
    span = Element('span')
    span.text = "Inline text"
    result = extract_text_array(span)
    assert result == ["Inline text"], f"Expected ['Inline text'], got {result}"
    
    # Test 3: Element with child
    parent = Element('div')
    parent.text = "Parent "
    child = SubElement(parent, 'span')
    child.text = "Child"
    child.tail = " tail"
    result = extract_text_array(parent)
    assert result == [None, "Parent ", "Child", " tail", None], f"Expected [None, 'Parent ', 'Child', ' tail', None], got {result}"
    
    # Test 4: Separator element (br)
    br = Element('br')
    result = extract_text_array(br)
    assert result == [True], f"Expected [True], got {result}"
    
    # Test 5: Nested inline elements
    p = Element('p')
    p.text = "Text "
    strong = SubElement(p, 'strong')
    strong.text = "bold"
    strong.tail = " more"
    result = extract_text_array(p)
    assert result == [None, "Text ", "bold", " more", None], f"Expected [None, 'Text ', 'bold', ' more', None], got {result}"
    
    # Test 6: Empty element
    empty = Element('div')
    result = extract_text_array(empty)
    assert result == [None, None], f"Expected [None, None], got {result}"
    
    # Test 7: Multiple children
    ul = Element('ul')
    li1 = SubElement(ul, 'li')
    li1.text = "First"
    li2 = SubElement(ul, 'li')
    li2.text = "Second"
    result = extract_text_array(ul)
    assert result == [None, "First", None, "Second", None], f"Expected [None, 'First', None, 'Second', None], got {result}"
    
    # Test 8: Squash artificial newlines
    div = Element('div')
    div.text = "A"
    sub = SubElement(div, 'div')
    sub.text = "B"
    sub.tail = "C"
    result = extract_text_array(div, squash_artifical_nl=True)
    assert result == [None, "A", None, "B", None, "C", None], f"Expected [None, 'A', None, 'B', None, 'C', None], got {result}"
    
    # Test 9: Strip artificial newlines
    div = Element('div')
    div.text = "Text"
    result = extract_text_array(div, strip_artifical_nl=True)
    assert result == ["Text"], f"Expected ['Text'], got {result}"
    
    # Test 10: Callable tag returns empty string
    class FakeTag:
        def __call__(self):
            pass
    
    fake = Element('fake')
    fake.tag = FakeTag()
    result = extract_text_array(fake)
    assert result == '', f"Expected '', got {result}"
    
    # Test 11: Deep nesting
    html = Element('html')
    body = SubElement(html, 'body')
    body.text = "Body "
    p = SubElement(body, 'p')
    p.text = "Paragraph "
    span = SubElement(p, 'span')
    span.text = "Span"
    span.tail = " end"
    p.tail = " after"
    body.tail = " after body"
    result = extract_text_array(html)
    assert len(result) > 0, "Should produce non-empty result"
    
    # Test 12: Multiple separators
    div = Element('div')
    br1 = SubElement(div, 'br')
    br1.tail = " text1 "
    br2 = SubElement(div, 'br')
    br2.tail = " text2"
    result = extract_text_array(div)
    assert True in result, "Should contain separator marker"
    assert result.count(True) == 2, f"Expected 2 separators, got {result.count(True)}"


# LLM-generated content at query #8
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello world</p>")
    assert extract_text_array(dom) == [None, "Hello world", None]

    # Test with inline element
    dom = etree.fromstring("<span>inline text</span>")
    assert extract_text_array(dom) == ["inline text"]

    # Test with separator element (br)
    dom = etree.fromstring("<br/>")
    assert extract_text_array(dom) == [True]

    # Test with nested inline elements
    dom = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text_array(dom) == [None, "Hello ", "bold", " world", None]

    # Test with separator inside block element
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text_array(dom) == [None, "Line1", True, "Line2", None]

    # Test with nested block elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert None in result  # Should have artificial newlines
    assert "First" in result
    assert "Second" in result

    # Test with text and tail text
    dom = etree.fromstring("<p>Start<b>bold</b>End</p>")
    assert extract_text_array(dom) == [None, "Start", "bold", "End", None]

    # Test with empty element
    dom = etree.fromstring("<p></p>")
    assert extract_text_array(dom) == [None, None]

    # Test with callable tag (should return empty string)
    class MockElement:
        tag = lambda x: None
    assert extract_text_array(MockElement()) == ""

    # Test with squash_artifical_nl=False
    dom = etree.fromstring("<p>Text</p>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, "Text", None]  # No squashing

    # Test with strip_artifical_nl=False
    dom = etree.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    # Should have artificial newlines at start and end
    assert result[0] is None or result[-1] is None


# LLM-generated content at query #9
#--------------------------

```python
def test_extract_text():
    # Test with simple text
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with nested inline elements
    dom = html.fromstring("<p>Hello <strong>World</strong></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with line break
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with multiple line breaks
    dom = html.fromstring("<p>Line1<br/><br/>Line2</p>")
    assert extract_text(dom) == "Line1\n\nLine2"
    
    # Test with whitespace squashing
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with nested block elements
    dom = html.fromstring("<div><p>First</p><div><p>Second</p></div></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with custom block symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' | ') == "First | Second"
    
    # Test with custom separator symbol
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(dom, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with empty element
    dom = html.fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test with mixed inline and block elements
    dom = html.fromstring("<div><span>Hello</span><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"
    
    # Test with multiple levels of nesting
    dom = html.fromstring("<div><p><strong>Hello</strong> <em>World</em></p></div>")
    assert extract_text(dom) == "Hello World"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>  Hello   World  </p>")
    assert extract_text(dom, squash_space=False) == "Hello   World"
    
    # Test with none value
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""


# LLM-generated content at query #10
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from xml.etree.ElementTree import Element, SubElement
    
    # Test 1: Simple text without any tags
    dom1 = Element('p')
    dom1.text = "Hello World"
    assert extract_text(dom1) == "Hello World"
    
    # Test 2: Text with inline tags
    dom2 = Element('p')
    dom2.text = "Hello "
    span = SubElement(dom2, 'span')
    span.text = "World"
    span.tail = "!"
    assert extract_text(dom2) == "Hello World!"
    
    # Test 3: Text with block elements
    dom3 = Element('div')
    p1 = SubElement(dom3, 'p')
    p1.text = "First paragraph"
    p2 = SubElement(dom3, 'p')
    p2.text = "Second paragraph"
    assert extract_text(dom3) == "First paragraph\nSecond paragraph"
    
    # Test 4: Text with separator (br)
    dom4 = Element('div')
    dom4.text = "Line 1"
    br = SubElement(dom4, 'br')
    br.tail = "Line 2"
    assert extract_text(dom4) == "Line 1\nLine 2"
    
    # Test 5: Nested elements
    dom5 = Element('div')
    outer = SubElement(dom5, 'div')
    inner = SubElement(outer, 'p')
    inner.text = "Nested text"
    assert extract_text(dom5) == "Nested text"
    
    # Test 6: Whitespace handling
    dom6 = Element('p')
    dom6.text = "Hello    World"
    assert extract_text(dom6) == "Hello World"
    
    # Test 7: Empty document
    dom7 = Element('div')
    assert extract_text(dom7) == ""
    
    # Test 8: Multiple spaces and newlines
    dom8 = Element('p')
    dom8.text = "Hello\n\n\nWorld"
    assert extract_text(dom8) == "Hello World"
    
    # Test 9: Custom block and separator symbols
    dom9 = Element('div')
    p1 = SubElement(dom9, 'p')
    p1.text = "First"
    p2 = SubElement(dom9, 'p')
    p2.text = "Second"
    assert extract_text(dom9, block_symbol=' | ') == "First | Second"
    
    # Test 10: Mixed content with multiple inline elements
    dom10 = Element('p')
    dom10.text = "The "
    strong = SubElement(dom10, 'strong')
    strong.text = "quick"
    strong.tail = " brown "
    em = SubElement(dom10, 'em')
    em.text = "fox"
    em.tail = " jumps"
    assert extract_text(dom10) == "The quick brown fox jumps"
    
    # Test 11: Element with children but no text in parent
    dom11 = Element('div')
    p = SubElement(dom11, 'p')
    p.text = "Content"
    assert extract_text(dom11) == "Content"
    
    # Test 12: Nested block elements
    dom12 = Element('div')
    outer = SubElement(dom12, 'div')
    outer.text = "Outer "
    inner = SubElement(outer, 'div')
    inner.text = "Inner"
    assert extract_text(dom12) == "Outer Inner"
    
    # Test 13: Multiple separators
    dom13 = Element('div')
    dom13.text = "Start"
    br1 = SubElement(dom13, 'br')
    br1.tail = "Middle"
    br2 = SubElement(dom13, 'br')
    br2.tail = "End"
    assert extract_text(dom13) == "Start\nMiddle\nEnd"
```


# LLM-generated content at query #11
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import etree
    dom = etree.fromstring("<p>Hello World</p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello World", None]
    
    # Test with inline element
    dom = etree.fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"]
    
    # Test with separator (br)
    dom = etree.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested inline elements
    dom = etree.fromstring("<p>Hello <b>World</b></p>")
    result = extract_text_array(dom)
    assert result == [None, "Hello ", "World", None]
    
    # Test with separator between text
    dom = etree.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None]
    
    # Test with empty element
    dom = etree.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]
    
    # Test with multiple nested block elements
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    expected = [None, None, "First", None, None, "Second", None, None]
    assert result == expected
    
    # Test squash_artifical_nl=False
    dom = etree.fromstring("<p>Text</p>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, "Text", None]
    
    # Test strip_artifical_nl=False
    dom = etree.fromstring("<p>Text</p>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Text", None]
    
    # Test with both squash and strip disabled
    dom = etree.fromstring("<p>Text</p>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Text", None]
    
    # Test with callable tag (should return empty string)
    class CallableTag:
        def __call__(self):
            pass
    dom = etree.fromstring("<p>Text</p>")
    dom.tag = CallableTag()
    result = extract_text_array(dom)
    assert result == ""
    
    # Test with nested elements and text/tail
    dom = etree.fromstring("<div>Start<p>Middle</p>End</div>")
    result = extract_text_array(dom)
    expected = [None, "Start", None, "Middle", None, "End", None]
    assert result == expected


# LLM-generated content at query #12
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    class MockElement:
        tag = 'p'
        text = 'Hello'
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text_array(MockElement())
    assert result == [None, 'Hello', None]
    
    # Test with inline element (no artificial newlines)
    class MockInlineElement:
        tag = 'span'
        text = 'World'
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text_array(MockInlineElement())
    assert result == ['World']
    
    # Test with separator element (br)
    class MockBrElement:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text_array(MockBrElement())
    assert result == [True]
    
    # Test with nested elements
    class MockChild:
        tag = 'span'
        text = 'inner'
        tail = ' after '
        def getchildren(self):
            return []
    
    class MockParent:
        tag = 'div'
        text = 'before '
        tail = None
        def getchildren(self):
            return [MockChild()]
    
    result = extract_text_array(MockParent())
    assert result == [None, 'before ', 'inner', ' after ', None]
    
    # Test with squash_artifical_nl=True (default)
    class MockSquash:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            child1 = type('obj', (object,), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
            child2 = type('obj', (object,), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
            return [child1, child2]
    
    result = extract_text_array(MockSquash(), squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, None]  # Squashed from [None, None, None, None]
    
    # Test with strip_artifical_nl=True (default)
    class MockStrip:
        tag = 'div'
        text = 'content'
        tail = None
        def getchildren(self):
            return []
    
    result = extract_text_array(MockStrip(), squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ['content']  # Stripped leading/trailing None
    
    # Test with both squash and strip disabled
    result = extract_text_array(MockStrip(), squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'content', None]


# LLM-generated content at query #13
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    from lxml import html
    dom = html.fromstring("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with nested inline elements
    dom = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block elements (should add newlines)
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator elements (br)
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
    
    # Test with multiple whitespace characters
    dom = html.fromstring("<p>Hello\t\nWorld</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test with whitespace stripping
    dom = html.fromstring("<p>  Hello World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test empty content
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test deeply nested structure
    dom = html.fromstring("<div><p><b>Hello</b> <i>World</i></p></div>")
    assert extract_text(dom) == "Hello World"
    
    # Test with multiple block elements and separators
    dom = html.fromstring("<div><p>A<br>B</p><p>C</p></div>")
    assert extract_text(dom) == "A\nB\nC"
    
    # Test with pre-like elements (not in INLINE_TAGS)
    dom = html.fromstring("<pre>Line1\nLine2</pre>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with mixed content including tail text
    dom = html.fromstring("<p>Hello <b>World</b> again</p>")
    assert extract_text(dom) == "Hello World again"


# LLM-generated content at query #14
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    result = extract_text_array(dom)
    assert result == ["Hello"]

    # Test with a block element
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", None]

    # Test with separator element
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]

    # Test with nested elements
    dom = html.fromstring("<div><span>Hello</span> World</div>")
    result = extract_text_array(dom)
    assert result == [None, "Hello", " World", None]

    # Test with multiple children
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == [None, None, "First", None, None, None, "Second", None, None]

    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, "Hello", None]

    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Hello", None]

    # Test with both flags False
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Hello", None]

    # Test with callable tag (returns empty string)
    class MockElement:
        tag = lambda: None
    result = extract_text_array(MockElement())
    assert result == ""

    # Test with empty element
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]

    # Test with text only
    dom = html.fromstring("Hello")
    result = extract_text_array(dom)
    assert result == ["Hello"]

    # Test with nested separators
    dom = html.fromstring("<div>Line1<br/>Line2</div>")
    result = extract_text_array(dom)
    assert result == [None, "Line1", True, "Line2", None]

    # Test with multiple separators in a row
    dom = html.fromstring("<div><br/><br/></div>")
    result = extract_text_array(dom)
    assert result == [None, True, True, None]


# LLM-generated content at query #15
#--------------------------

```python
def test_extract_text():
    # Test with simple text
    from lxml import html
    dom = html.fromstring("<p>Hello world</p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with inline tags
    dom = html.fromstring("<p>Hello <strong>world</strong></p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with block elements
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separators
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with nested tags
    dom = html.fromstring("<div><p><span>Text</span></p></div>")
    assert extract_text(dom) == "Text"
    
    # Test with whitespace
    dom = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(dom) == "Hello world"
    
    # Test with multiple whitespace and newlines
    dom = html.fromstring("<p>Hello\n\nworld</p>")
    assert extract_text(dom) == "Hello world"
    
    # Test empty element
    dom = html.fromstring("<p></p>")
    assert extract_text(dom) == ""
    
    # Test with only whitespace
    dom = html.fromstring("<p>   </p>")
    assert extract_text(dom) == ""
    
    # Test with custom block_symbol
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=' ') == "First Second"
    
    # Test with custom sep_symbol
    dom = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(dom, sep_symbol=' ') == "Line1 Line2"
    
    # Test with squash_space=False
    dom = html.fromstring("<p>  Hello   world  </p>")
    assert extract_text(dom, squash_space=False) == "  Hello   world  "
    
    # Test with mixed inline and block elements
    dom = html.fromstring("<div><p>Hello <strong>world</strong></p><p>Second</p></div>")
    assert extract_text(dom) == "Hello world\nSecond"
    
    # Test with script tag (should be ignored)
    dom = html.fromstring("<div><p>Hello</p><script>alert('test');</script><p>World</p></div>")
    assert extract_text(dom) == "Hello\nWorld"


# LLM-generated content at query #16
#--------------------------

```python
def test_extract_text():
    # Test with a simple inline element
    from lxml.html import fromstring
    dom = fromstring("<span>Hello World</span>")
    assert extract_text(dom) == "Hello World"
    
    # Test with block element
    dom = fromstring("<div>Hello</div>")
    assert extract_text(dom) == "Hello"
    
    # Test with nested elements
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test with separator (br)
    dom = fromstring("<span>Line1<br>Line2</span>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test with custom block_symbol
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(dom, block_symbol=" | ") == "First | Second"
    
    # Test with custom sep_symbol
    dom = fromstring("<span>Line1<br>Line2</span>")
    assert extract_text(dom, sep_symbol=" | ") == "Line1 | Line2"
    
    # Test with squash_space=False
    dom = fromstring("<div>  Hello   World  </div>")
    assert extract_text(dom, squash_space=False) == "  Hello   World  "
    
    # Test with mixed inline and block elements
    dom = fromstring("<div><span>Inline</span><p>Block</p></div>")
    assert extract_text(dom) == "Inline\nBlock"
    
    # Test with nested block elements
    dom = fromstring("<div><div><p>Deep</p></div></div>")
    assert extract_text(dom) == "Deep"
    
    # Test with empty element
    dom = fromstring("<div></div>")
    assert extract_text(dom) == ""
    
    # Test with text in parent and child
    dom = fromstring("<div>Parent text<p>Child text</p></div>")
    assert extract_text(dom) == "Parent text\nChild text"
    
    # Test with multiple separators
    dom = fromstring("<span>A<br>B<br>C</span>")
    assert extract_text(dom) == "A\nB\nC"
    
    # Test with mixed inline and separator
    dom = fromstring("<span>Text<br><b>Bold</b></span>")
    assert extract_text(dom) == "Text\nBold"
    
    # Test with consecutive block elements
    dom = fromstring("<div><p>First</p><p>Second</p><p>Third</p></div>")
    assert extract_text(dom) == "First\nSecond\nThird"
    
    # Test with whitespace normalization
    dom = fromstring("<div>  Multiple   spaces  </div>")
    assert extract_text(dom) == "Multiple spaces"
    
    # Test with newline in text
    dom = fromstring("<div>Line1\nLine2</div>")
    assert extract_text(dom) == "Line1 Line2"
```


# LLM-generated content at query #17
#--------------------------

```python
def test_extract_text():
    # Test basic text extraction
    from lxml import html
    doc = html.fromstring("<p>Hello World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test nested inline elements
    doc = html.fromstring("<p>Hello <b>World</b></p>")
    assert extract_text(doc) == "Hello World"
    
    # Test block elements add newlines
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test separator elements (br)
    doc = html.fromstring("<p>Line1<br>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test multiple whitespace gets squashed
    doc = html.fromstring("<p>Hello    World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test leading/trailing whitespace removed
    doc = html.fromstring("<p>  Hello World  </p>")
    assert extract_text(doc) == "Hello World"
    
    # Test empty elements
    doc = html.fromstring("<p></p>")
    assert extract_text(doc) == ""
    
    # Test deeply nested structure
    doc = html.fromstring("<div><p><b>Deep</b> text</p></div>")
    assert extract_text(doc) == "Deep text"
    
    # Test custom block_symbol
    doc = html.fromstring("<div><p>A</p><p>B</p></div>")
    assert extract_text(doc, block_symbol=' | ') == "A | B"
    
    # Test custom sep_symbol
    doc = html.fromstring("<p>A<br>B</p>")
    assert extract_text(doc, sep_symbol=' | ') == "A | B"
    
    # Test squash_space=False
    doc = html.fromstring("<p>  Hello  </p>")
    assert extract_text(doc, squash_space=False) == "  Hello  "
    
    # Test elements with tail text
    doc = html.fromstring("<p>Hello<b>bold</b>world</p>")
    assert extract_text(doc) == "Helloboldworld"
    
    # Test multiple block level nesting
    doc = html.fromstring("<div><p>First</p><span>Inline</span><p>Second</p></div>")
    assert extract_text(doc) == "First\nInline\nSecond"
    
    # Test with attributes (should not affect text extraction)
    doc = html.fromstring('<p class="test">Hello</p>')
    assert extract_text(doc) == "Hello"
    
    # Test mixed inline and block
    doc = html.fromstring("<div><span>Hi</span><p>There</p></div>")
    assert extract_text(doc) == "Hi\nThere"
```


# LLM-generated content at query #18
#--------------------------

```python
def test_extract_text():
    # Test 1: Simple text with no tags
    from minidom import parseString
    dom = parseString("<p>Hello World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 2: Nested inline tags
    dom = parseString("<p>Hello <b>World</b></p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 3: Block elements create newlines
    dom = parseString("<div>First</div><div>Second</div>")
    assert extract_text(dom) == "First\nSecond"
    
    # Test 4: Separator tags (br)
    dom = parseString("<p>Line1<br>Line2</p>")
    assert extract_text(dom) == "Line1\nLine2"
    
    # Test 5: Empty content
    dom = parseString("<p></p>")
    assert extract_text(dom) == ""
    
    # Test 6: Whitespace squashing
    dom = parseString("<p>  Hello    World  </p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 7: Multiple block elements with text
    dom = parseString("<div>First</div><div>Second</div><div>Third</div>")
    assert extract_text(dom) == "First\nSecond\nThird"
    
    # Test 8: Inline tags within block tags
    dom = parseString("<div><span>Hello</span> <span>World</span></div>")
    assert extract_text(dom) == "Hello World"
    
    # Test 9: Complex nested structure
    dom = parseString("<div><p>Paragraph <b>bold</b></p><p>Second <i>italic</i></p></div>")
    assert extract_text(dom) == "Paragraph bold\nSecond italic"
    
    # Test 10: Custom separators
    dom = parseString("<p>Hello<br>World</p>")
    assert extract_text(dom, sep_symbol=" | ") == "Hello | World"
    
    # Test 11: Custom block symbol
    dom = parseString("<div>First</div><div>Second</div>")
    assert extract_text(dom, block_symbol=" | ") == "First | Second"
    
    # Test 12: Whitespace with multiple spaces and newlines
    dom = parseString("<p>Hello   \n  World</p>")
    assert extract_text(dom) == "Hello World"
    
    # Test 13: Leading and trailing whitespace
    dom = parseString("  <p>Hello World</p>  ")
    assert extract_text(dom) == "Hello World"
    
    # Test 14: Nested block elements
    dom = parseString("<div><div>Nested</div></div>")
    assert extract_text(dom) == "Nested"
    
    # Test 15: Mixed inline and block elements
    dom = parseString("<div><b>Bold</b> and <i>italic</i></div><div>New block</div>")
    assert extract_text(dom) == "Bold and italic\nNew block"


# LLM-generated content at query #19
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text node
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<p>Hello World</p>")
    assert extract_text_array(dom) == ["Hello World"]
    
    # Test with inline tags
    dom = fromstring("<p>Hello <b>World</b></p>")
    assert extract_text_array(dom) == ["Hello ", "World"]
    
    # Test with non-inline tags (should add None separators)
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert None in result  # Should have artificial newlines
    
    # Test with separator tags (br)
    dom = fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert True in result  # br should be represented as True
    
    # Test with nested inline and block elements
    dom = fromstring("<div><span>Text</span><p>Para</p></div>")
    result = extract_text_array(dom)
    assert "Text" in result
    assert "Para" in result
    
    # Test with callable tag (should return empty string)
    dom = fromstring("<script>function()</script>")
    assert extract_text_array(dom) == ""
    
    # Test with tail text
    dom = fromstring("<p>Hello<b>bold</b>tail</p>")
    assert extract_text_array(dom) == ["Hello", "bold", "tail"]
    
    # Test squash_artifical_nl parameter
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    result_with_squash = extract_text_array(dom, squash_artifical_nl=True)
    result_without_squash = extract_text_array(dom, squash_artifical_nl=False)
    assert len(result_with_squash) <= len(result_without_squash)
    
    # Test strip_artifical_nl parameter
    dom = fromstring("<div><p>Content</p></div>")
    result_with_strip = extract_text_array(dom, strip_artifical_nl=True)
    result_without_strip = extract_text_array(dom, strip_artifical_nl=False)
    assert not result_with_strip or isinstance(result_with_strip[0], str)
    if result_without_strip:
        assert result_without_strip[0] is None  # Should start with None without stripping
    
    # Test with empty dom
    dom = fromstring("<div></div>")
    assert extract_text_array(dom) == []
    
    # Test with deeply nested structure
    dom = fromstring("<div><ul><li>Item1</li><li>Item2</li></ul></div>")
    result = extract_text_array(dom)
    assert "Item1" in result
    assert "Item2" in result
```


# LLM-generated content at query #20
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple text element
    from lxml.html import fromstring
    dom = fromstring('<p>Hello World</p>')
    result = extract_text_array(dom)
    assert result == ['Hello World']
    
    # Test with inline elements
    dom = fromstring('<p>Hello <b>bold</b> world</p>')
    result = extract_text_array(dom)
    assert result == ['Hello ', 'bold', ' world']
    
    # Test with block elements (should insert None)
    dom = fromstring('<div><p>First</p><p>Second</p></div>')
    result = extract_text_array(dom)
    assert None in result
    assert 'First' in result
    assert 'Second' in result
    
    # Test with separator elements (br)
    dom = fromstring('<p>Line1<br/>Line2</p>')
    result = extract_text_array(dom)
    assert True in result  # br separator
    
    # Test with nested inline elements
    dom = fromstring('<span>text <a href="#">link</a> text</span>')
    result = extract_text_array(dom)
    assert result == ['text ', 'link', ' text']
    
    # Test with None text content
    dom = fromstring('<div><br/></div>')
    result = extract_text_array(dom)
    assert True in result
    
    # Test squash_artifical_nl=True (default)
    dom = fromstring('<div><p>Test</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ['Test']
    
    # Test squash_artifical_nl=False
    dom = fromstring('<div><p>Test</p></div>')
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert None in result  # Should have artificial newlines
    
    # Test strip_artifical_nl=True (default)
    dom = fromstring('<div><p>Test</p></div>')
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['Test']
    
    # Test strip_artifical_nl=False
    dom = fromstring('<div><p>Test</p></div>')
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # Should start with None
    
    # Test with callable tag (should return empty string)
    class MockElement:
        tag = lambda: None
    mock_dom = MockElement()
    assert extract_text_array(mock_dom) == ''
    
    # Test with complex nested structure
    html = '<div><p>Hello <b>world</b></p><br/><p>Second <i>line</i></p></div>'
    dom = fromstring(html)
    result = extract_text_array(dom)
    assert 'Hello' in result
    assert 'world' in result
    assert 'Second' in result
    assert 'line' in result
    assert True in result  # br separator
    
    # Test that ordered list items are preserved
    dom = fromstring('<ol><li>Item 1</li><li>Item 2</li></ol>')
    result = extract_text_array(dom)
    assert 'Item 1' in result
    assert 'Item 2' in result
    
    # Test with empty element
    dom = fromstring('<p></p>')
    result = extract_text_array(dom)
    assert result == [] or result == ['']
    
    # Test with whitespace only
    dom = fromstring('<p>   </p>')
    result = extract_text_array(dom)
    assert result == ['   ']  # Whitespace should be preserved in array form
    
    # Test multiple separators
    dom = fromstring('<p>Text<br/><br/>More text</p>')
    result = extract_text_array(dom)
    assert result.count(True) == 2  # Two br separators


# LLM-generated content at query #21
#--------------------------

```python
def test_extract_text():
    # Test with simple text element
    from lxml import html
    doc = html.fromstring("<p>Hello World</p>")
    assert extract_text(doc) == "Hello World"
    
    # Test with inline elements - should not add newlines
    doc = html.fromstring("<p>Hello <b>bold</b> world</p>")
    assert extract_text(doc) == "Hello bold world"
    
    # Test with block elements - should add newlines
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc) == "First\nSecond"
    
    # Test with separator elements (br)
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc) == "Line1\nLine2"
    
    # Test with nested elements
    doc = html.fromstring("<div><p>Text with <b>bold</b> and <i>italic</i></p></div>")
    assert extract_text(doc) == "Text with bold and italic"
    
    # Test with whitespace normalization
    doc = html.fromstring("<p>  Multiple   spaces   </p>")
    assert extract_text(doc) == "Multiple spaces"
    
    # Test with empty element
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""
    
    # Test with custom block_symbol
    doc = html.fromstring("<div><p>First</p><p>Second</p></div>")
    assert extract_text(doc, block_symbol=' | ') == "First | Second"
    
    # Test with custom sep_symbol
    doc = html.fromstring("<p>Line1<br/>Line2</p>")
    assert extract_text(doc, sep_symbol=' | ') == "Line1 | Line2"
    
    # Test with squash_space=False
    doc = html.fromstring("<p>  Multiple   spaces   </p>")
    assert extract_text(doc, squash_space=False) == "  Multiple   spaces   "
    
    # Test with multiple block elements
    doc = html.fromstring("<div><h1>Title</h1><p>Content</p></div>")
    assert extract_text(doc) == "Title\nContent"
    
    # Test with nested block elements
    doc = html.fromstring("<div><div><p>Nested</p></div></div>")
    assert extract_text(doc) == "Nested"
```


# LLM-generated content at query #22
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text node
    dom = type('Node', (), {'tag': 'p', 'text': 'Hello', 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom)
    assert result == [None, 'Hello', None], f"Expected [None, 'Hello', None], got {result}"
    
    # Test with inline tag
    dom = type('Node', (), {'tag': 'span', 'text': 'world', 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom)
    assert result == ['world'], f"Expected ['world'], got {result}"
    
    # Test with separator tag (br)
    dom = type('Node', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom)
    assert result == [True], f"Expected [True], got {result}"
    
    # Test with nested structure
    child = type('Node', (), {'tag': 'b', 'text': 'bold', 'tail': ' text', 'getchildren': lambda: []})()
    parent = type('Node', (), {'tag': 'p', 'text': 'Some ', 'tail': None, 'getchildren': lambda: [child]})()
    result = extract_text_array(parent)
    assert result == [None, 'Some ', 'bold', ' text', None], f"Expected [None, 'Some ', 'bold', ' text', None], got {result}"
    
    # Test with callable tag (should return empty string)
    dom = type('Node', (), {'tag': lambda: None, 'text': None, 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom)
    assert result == '', f"Expected '', got {result}"
    
    # Test with squash_artifical_nl=True
    dom = type('Node', (), {'tag': 'p', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == [None], f"Expected [None], got {result}"
    
    # Test with strip_artifical_nl=True
    dom = type('Node', (), {'tag': 'div', 'text': 'content', 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['content'], f"Expected ['content'], got {result}"
    
    # Test with both squash and strip
    dom = type('Node', (), {'tag': 'div', 'text': 'content', 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['content'], f"Expected ['content'], got {result}"


# LLM-generated content at query #23
#--------------------------

```python
def test_extract_text_array():
    # Test empty dom
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(MockElement()) == [None, None]

    # Test inline tag without text
    class MockInline:
        tag = 'span'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(MockInline()) == []

    # Test separator tag
    class MockBr:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(MockBr()) == [True]

    # Test element with text
    class MockWithText:
        tag = 'div'
        text = 'hello'
        def getchildren(self):
            return []
    assert extract_text_array(MockWithText()) == [None, 'hello', None]

    # Test nested elements
    class MockChild:
        tag = 'span'
        text = 'world'
        tail = '!'
        def getchildren(self):
            return []
    
    class MockParent:
        tag = 'div'
        text = 'hello '
        def getchildren(self):
            return [MockChild()]
    
    assert extract_text_array(MockParent()) == [None, 'hello ', 'world', '!', None]

    # Test with squash_artifical_nl=True (default)
    class MockMultipleNl:
        tag = 'div'
        text = 'a'
        def getchildren(self):
            child = MockElement()
            child.tag = 'div'
            child.text = None
            child.getchildren = lambda: []
            return [child]
    result = extract_text_array(MockMultipleNl())
    # Should have squashed consecutive None values
    assert result.count(None) <= 1  # Should have at most one None

    # Test with squash_artifical_nl=False
    result_no_squash = extract_text_array(MockMultipleNl(), squash_artifical_nl=False)
    assert result_no_squash.count(None) >= 2  # Should have multiple None values

    # Test with strip_artifical_nl=True (default)
    class MockLeadingTrailingNl:
        tag = 'div'
        text = 'test'
        def getchildren(self):
            return []
    assert extract_text_array(MockLeadingTrailingNl()) == ['test']

    # Test with strip_artifical_nl=False
    result_no_strip = extract_text_array(MockLeadingTrailingNl(), strip_artifical_nl=False)
    assert result_no_strip[0] is None
    assert result_no_strip[-1] is None

    # Test callable tag returns empty string
    class MockCallableTag:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(MockCallableTag()) == ''


# LLM-generated content at query #24
#--------------------------

```python
def test_extract_text():
    # Test with simple inline elements
    from html import HTML
    doc = HTML()
    doc.html.body.p("Hello World")
    assert extract_text(doc) == "Hello World"

    # Test with block elements
    doc = HTML()
    doc.html.body.div("First paragraph")
    doc.html.body.div("Second paragraph")
    assert extract_text(doc) == "First paragraph\nSecond paragraph"

    # Test with separator (br)
    doc = HTML()
    doc.html.body.p("Line1")
    doc.html.body.p.br()
    doc.html.body.p("Line2")
    assert extract_text(doc) == "Line1\nLine2"

    # Test with nested inline elements
    doc = HTML()
    p = doc.html.body.p
    p.span("Hello ")
    p.strong("World")
    assert extract_text(doc) == "Hello World"

    # Test with whitespace squashing
    doc = HTML()
    doc.html.body.p("Hello    World")
    assert extract_text(doc) == "Hello World"

    # Test with custom block symbol
    doc = HTML()
    doc.html.body.div("First")
    doc.html.body.div("Second")
    assert extract_text(doc, block_symbol=" | ") == "First | Second"

    # Test with custom separator symbol
    doc = HTML()
    doc.html.body.p("Line1")
    doc.html.body.p.br()
    doc.html.body.p("Line2")
    assert extract_text(doc, sep_symbol=" | ") == "Line1 | Line2"

    # Test with squash_space=False
    doc = HTML()
    doc.html.body.p("Hello    World")
    result = extract_text(doc, squash_space=False)
    assert "    " in result  # whitespace preserved

    # Test with empty document
    doc = HTML()
    assert extract_text(doc) == ""

    # Test with pre tag (should preserve whitespace)
    doc = HTML()
    doc.html.body.pre("Hello\n    World")
    result = extract_text(doc)
    assert "Hello\n    World" in result

    # Test with nested block elements
    doc = HTML()
    outer = doc.html.body.div
    outer.div("Inner")
    outer.div("Another")
    assert extract_text(doc) == "Inner\nAnother"

    # Test with text before and after children
    doc = HTML()
    p = doc.html.body.p
    p("Start ")
    p.span("middle")
    p(" end")
    result = extract_text(doc)
    assert "Start middle end" in result

    # Test with multiple levels of nesting
    doc = HTML()
    doc.html.body.div.div.span("Deep")
    assert extract_text(doc) == "Deep"

    # Test with whitespace only content
    doc = HTML()
    doc.html.body.p("   ")
    assert extract_text(doc) == ""

    # Test with None text nodes
    doc = HTML()
    doc.html.body.div
    assert extract_text(doc) == ""

    # Test with complex mixing of inline and block
    doc = HTML()
    body = doc.html.body
    body.div("Block1")
    body.p.span("Inline")
    body.div("Block2")
    result = extract_text(doc)
    assert "Block1" in result
    assert "Inline" in result
    assert "Block2" in result
    assert result.count("\n") == 2
```


# LLM-generated content at query #25
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello World</span>")
    result = extract_text_array(dom)
    assert result == ["Hello World"]
    
    # Test with block element (should add None for artificial newlines)
    dom = html.fromstring("<div>Text</div>")
    result = extract_text_array(dom)
    assert result == ["Text"]
    
    # Test with separator element (br)
    dom = html.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested elements
    dom = html.fromstring("<div><span>Hello</span><span>World</span></div>")
    result = extract_text_array(dom)
    assert None in result  # should have artificial newlines
    
    # Test with text and tail
    dom = html.fromstring("<p>Hello <b>bold</b> text</p>")
    result = extract_text_array(dom)
    assert "Hello " in result
    assert "bold" in result
    assert " text" in result
    
    # Test with separator inside inline
    dom = html.fromstring("<span>Line1<br/>Line2</span>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert True in result  # should have separator
    
    # Test with callable tag (should return empty string)
    dom = html.fromstring("<div>Text</div>")
    dom.tag = lambda: None  # make tag callable
    result = extract_text_array(dom)
    assert result == ""
    
    # Test empty element
    dom = html.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test with multiple levels of nesting
    dom = html.fromstring("<div><p><span>Deep</span></p></div>")
    result = extract_text_array(dom)
    assert "Deep" in result
    assert None in result  # artificial newlines from block elements
    
    # Test squashing artificial newlines
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    # Should have consecutive Nones squashed
    none_count = result.count(None)
    assert none_count <= 2  # at most one before and one after
    
    # Test stripping artificial newlines
    dom = html.fromstring("<div><p>Text</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result[0] == "Text"  # should not start/end with None
    
    # Test with multiple children
    dom = html.fromstring("<ul><li>Item1</li><li>Item2</li></ul>")
    result = extract_text_array(dom)
    assert "Item1" in result
    assert "Item2" in result
    
    # Test with inline elements only (should not add artificial newlines)
    dom = html.fromstring("<span>Inline <em>emphasized</em> text</span>")
    result = extract_text_array(dom)
    assert None not in result  # no block elements, so no artificial newlines
    assert True not in result  # no separators
    assert "Inline " in result
    assert "emphasized" in result
    assert " text" in result


# LLM-generated content at query #26
#--------------------------

```python
def test_extract_text():
    # Test with empty DOM
    class EmptyDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    assert extract_text(EmptyDom()) == ''
    
    # Test with simple text
    class SimpleDom:
        tag = 'p'
        text = 'Hello World'
        def getchildren(self):
            return []
    assert extract_text(SimpleDom()) == 'Hello World'
    
    # Test with inline tags
    class InlineDom:
        tag = 'span'
        text = 'Hello '
        def getchildren(self):
            child = SimpleDom()
            child.tag = 'strong'
            child.text = 'World'
            child.tail = '!'
            return [child]
    assert extract_text(InlineDom()) == 'Hello World!'
    
    # Test with block tags (div)
    class BlockDom:
        tag = 'div'
        text = 'First '
        def getchildren(self):
            child = SimpleDom()
            child.tag = 'div'
            child.text = 'Second'
            return [child]
    assert extract_text(BlockDom()) == 'First Second'
    
    # Test with separator tags (br)
    class BrDom:
        tag = 'div'
        text = 'Line1'
        def getchildren(self):
            br = SimpleDom()
            br.tag = 'br'
            br.text = None
            return [br]
        tail = 'Line2'
    assert extract_text(BrDom()) == 'Line1\nLine2'
    
    # Test with multiple block levels
    class NestedBlockDom:
        tag = 'div'
        text = 'A'
        def getchildren(self):
            inner = BlockDom()
            inner.text = 'B'
            return [inner]
    assert extract_text(NestedBlockDom()) == 'A\nB'
    
    # Test with squash_space=False
    assert extract_text(SimpleDom(), squash_space=False) == 'Hello World'
    
    # Test with custom block_symbol
    assert extract_text(BlockDom(), block_symbol=' ') == 'First Second'
    
    # Test with custom sep_symbol
    assert extract_text(BrDom(), sep_symbol=' | ') == 'Line1 | Line2'
    
    # Test with whitespace squashing
    class WhitespaceDom:
        tag = 'p'
        text = 'Hello    World'
        def getchildren(self):
            return []
    assert extract_text(WhitespaceDom()) == 'Hello World'
    
    # Test with leading/trailing whitespace
    class LeadingTrailingDom:
        tag = 'p'
        text = '  Hello World  '
        def getchildren(self):
            return []
    assert extract_text(LeadingTrailingDom()) == 'Hello World'
    
    # Test with multiple children
    class MultiChildDom:
        tag = 'div'
        text = 'Start '
        def getchildren(self):
            child1 = SimpleDom()
            child1.tag = 'span'
            child1.text = 'Middle'
            child1.tail = ' End'
            return [child1]
    assert extract_text(MultiChildDom()) == 'Start Middle End'
    
    # Test with callable tag (edge case)
    class CallableTagDom:
        def tag(self):
            pass
        text = 'Should be empty'
        def getchildren(self):
            return []
    assert extract_text(CallableTagDom()) == ''
```


# LLM-generated content at query #27
#--------------------------

```python
def test_extract_text():
    # Test simple inline tag
    from lxml import html
    doc = html.fromstring("<span>hello</span>")
    assert extract_text(doc) == "hello"

    # Test block tag added newline
    doc = html.fromstring("<div>hello</div>")
    assert extract_text(doc) == "hello"

    # Test separator tag
    doc = html.fromstring("<br>")
    assert extract_text(doc) == ""  # br is separator, no text

    # Test nested inline inside block
    doc = html.fromstring("<div><span>hello</span></div>")
    assert extract_text(doc) == "hello"

    # Test multiple inline tags
    doc = html.fromstring("<span>hello</span><span>world</span>")
    assert extract_text(doc) == "helloworld"

    # Test block tags add newlines
    doc = html.fromstring("<div>hello</div><div>world</div>")
    assert extract_text(doc) == "hello\nworld"

    # Test separator (br) adds newline
    doc = html.fromstring("hello<br>world")
    assert extract_text(doc) == "hello\nworld"

    # Test whitespace squashing
    doc = html.fromstring("<span>hello   world</span>")
    assert extract_text(doc) == "hello world"

    # Test leading/trailing whitespace removed
    doc = html.fromstring("  hello world  ")
    assert extract_text(doc) == "hello world"

    # Test nested block tags
    doc = html.fromstring("<div><p>hello</p><p>world</p></div>")
    assert extract_text(doc) == "hello\nworld"

    # Test custom block_symbol
    doc = html.fromstring("<div>hello</div><div>world</div>")
    assert extract_text(doc, block_symbol=" | ") == "hello | world"

    # Test custom sep_symbol
    doc = html.fromstring("hello<br>world")
    assert extract_text(doc, sep_symbol=" - ") == "hello - world"

    # Test squash_space=False
    doc = html.fromstring("  hello   world  ")
    assert extract_text(doc, squash_space=False) == "  hello   world  "

    # Test empty document
    doc = html.fromstring("")
    assert extract_text(doc) == ""

    # Test tag with only text
    doc = html.fromstring("just text")
    assert extract_text(doc) == "just text"

    # Test mixed content
    doc = html.fromstring("<div>hello <b>world</b> foo</div>")
    assert extract_text(doc) == "hello world foo"

    # Test img tag (inline, no text)
    doc = html.fromstring("<div><img src='test.png'/>text</div>")
    assert extract_text(doc) == "text"

    # Test multiple newlines from blocks
    doc = html.fromstring("<div>a</div><div>b</div><div>c</div>")
    assert extract_text(doc) == "a\nb\nc"

    # Test text with tail
    doc = html.fromstring("<div>hello <span>world</span> foo</div>")
    assert extract_text(doc) == "hello world foo"

    # Test pre tag (not in INLINE_TAGS, but should preserve whitespace?)
    # Actually pre is not in INLINE_TAGS, so acts as block
    doc = html.fromstring("<pre>hello\nworld</pre>")
    assert extract_text(doc) == "hello\nworld"

    # Test textarea (inline tag with whitespace)
    doc = html.fromstring("<textarea>hello   world</textarea>")
    assert extract_text(doc) == "hello world"

    # Test input (self-closing inline)
    doc = html.fromstring("<input type='text'/>")
    assert extract_text(doc) == ""

    # Test complex nesting
    doc = html.fromstring("""
        <div>
            <h1>Title</h1>
            <p>Some <b>bold</b> text</p>
            <br/>
            <p>More text</p>
        </div>
    """)
    result = extract_text(doc)
    assert "Title" in result
    assert "Some bold text" in result
    assert "More text" in result
    assert result.count("\n") >= 2  # multiple newlines from blocks and br

    # Test None text
    doc = html.fromstring("<div></div>")
    assert extract_text(doc) == ""

    # Test only whitespace
    doc = html.fromstring("   ")
    assert extract_text(doc) == ""

    # Test unicode whitespace
    doc = html.fromstring("hello\u200Bworld")
    assert extract_text(doc) == "hello world"  # zero-width space is whitespace

    # Test multiple br separators
    doc = html.fromstring("a<br><br>b")
    assert extract_text(doc) == "a\n\nb"

    # Test block followed by inline
    doc = html.fromstring("<div>hello</div><span>world</span>")
    assert extract_text(doc) == "hello\nworld"

    # Test inline followed by block
    doc = html.fromstring("<span>hello</span><div>world</div>")
    assert extract_text(doc) == "hello\nworld"
```


# LLM-generated content at query #28
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text element
    from lxml import html
    dom = html.fromstring("<p>Hello world</p>")
    result = extract_text_array(dom)
    assert result == ['Hello world']
    
    # Test with inline tags
    dom = html.fromstring("<p>Hello <b>world</b></p>")
    result = extract_text_array(dom)
    assert result == ['Hello ', 'world']
    
    # Test with separator tag (br)
    dom = html.fromstring("<p>Line1<br/>Line2</p>")
    result = extract_text_array(dom)
    assert True in result  # br generates True
    
    # Test with block-level tags
    dom = html.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert None in result  # block tags generate None
    
    # Test with nested inline tags
    dom = html.fromstring("<p>Text <span>inside <em>emphasized</em></span> end</p>")
    result = extract_text_array(dom)
    assert result == ['Text ', 'inside ', 'emphasized', ' end']
    
    # Test with strip_artifical_nl=True (default)
    dom = html.fromstring("<div><p>Content</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['Content']  # leading/trailing None stripped
    
    # Test with strip_artifical_nl=False
    dom = html.fromstring("<div><p>Content</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result[0] is None  # leading None preserved
    assert result[-1] is None  # trailing None preserved
    
    # Test with squash_artifical_nl=True (default)
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=True)
    # Should have only one None between A and B
    none_count = sum(1 for x in result if x is None)
    assert none_count == 1
    
    # Test with squash_artifical_nl=False
    dom = html.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    # Should have multiple None between A and B
    none_count = sum(1 for x in result if x is None)
    assert none_count > 1
    
    # Test with callable tag (should return empty string)
    class FakeDom:
        tag = lambda: None
    result = extract_text_array(FakeDom())
    assert result == ''
    
    # Test with tail text
    dom = html.fromstring("<p>Start <b>bold</b> middle <i>italic</i> end</p>")
    result = extract_text_array(dom)
    assert result == ['Start ', 'bold', ' middle ', 'italic', ' end']
    
    # Test with empty element
    dom = html.fromstring("<p></p>")
    result = extract_text_array(dom)
    assert result == []
    
    # Test with only whitespace
    dom = html.fromstring("<p>   </p>")
    result = extract_text_array(dom)
    assert result == ['   ']
    
    # Test with mixed block and inline
    dom = html.fromstring("<div><h1>Title</h1><p>Para <a href='#'>link</a></p></div>")
    result = extract_text_array(dom)
    assert 'Title' in result
    assert 'Para ' in result
    assert 'link' in result
    assert None in result  # block-level separators
    
    # Test with multiple br tags
    dom = html.fromstring("<p>Line1<br/><br/>Line2</p>")
    result = extract_text_array(dom)
    true_count = sum(1 for x in result if x is True)
    assert true_count == 2  # two br tags
    
    # Test with nested block elements
    dom = html.fromstring("<div><section><article><p>Deep</p></article></section></div>")
    result = extract_text_array(dom)
    assert result == ['Deep']  # nested blocks collapsed
    
    # Test with text before and after child elements
    dom = html.fromstring("<p>Before <span>inside</span> After</p>")
    result = extract_text_array(dom)
    assert result == ['Before ', 'inside', ' After']


# LLM-generated content at query #29
#--------------------------

```python
def test_extract_text_array():
    # Test with a simple inline element
    from lxml import html
    dom = html.fromstring("<span>Hello</span>")
    assert extract_text_array(dom) == ["Hello"]

    # Test with a block element
    dom = html.fromstring("<div>Hello</div>")
    assert extract_text_array(dom) == [None, "Hello", None]

    # Test with a separator element
    dom = html.fromstring("<br/>")
    assert extract_text_array(dom) == [True]

    # Test with nested elements
    dom = html.fromstring("<div><span>Hello</span> World</div>")
    assert extract_text_array(dom) == [None, "Hello", " World", None]

    # Test with multiple block elements
    dom = html.fromstring("<div>First</div><div>Second</div>")
    assert extract_text_array(dom) == [None, "First", None, None, "Second", None]

    # Test with separator inside inline
    dom = html.fromstring("<span>Before<br/>After</span>")
    assert extract_text_array(dom) == ["Before", True, "After"]

    # Test with callable tag (should return empty string)
    class MockElement:
        tag = lambda: None
    assert extract_text_array(MockElement()) == ""

    # Test with None text and tail
    dom = html.fromstring("<div><span></span></div>")
    assert extract_text_array(dom) == [None, None, None]

    # Test with nested non-inline elements
    dom = html.fromstring("<div><p>Text</p></div>")
    assert extract_text_array(dom) == [None, None, "Text", None, None]

    # Test with squash_artifical_nl=False and strip_artifical_nl=False
    dom = html.fromstring("<div>Hello</div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Hello", None]

    # Test with multiple consecutive None values
    dom = html.fromstring("<div></div><div></div>")
    result = extract_text_array(dom)
    assert result == [None, None]  # Squashed consecutive Nones

    # Test with inline elements that should not add None
    dom = html.fromstring("<b>Bold</b><i>Italic</i>")
    assert extract_text_array(dom) == ["Bold", "Italic"]

    # Test with separators and inline elements
    dom = html.fromstring("<div>Line1<br/>Line2</div>")
    assert extract_text_array(dom) == [None, "Line1", True, "Line2", None]


# LLM-generated content at query #30
#--------------------------

```python
def test_extract_text_array():
    # Test with simple text node
    dom = type('Mock', (), {'tag': 'p', 'text': 'Hello', 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom)
    assert result == [None, 'Hello', None]
    
    # Test with inline tag
    dom = type('Mock', (), {'tag': 'span', 'text': 'world', 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom)
    assert result == ['world']
    
    # Test with separator tag
    dom = type('Mock', (), {'tag': 'br', 'text': None, 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom)
    assert result == [True]
    
    # Test with nested elements
    child_dom = type('Mock', (), {'tag': 'b', 'text': 'bold', 'tail': ' tail', 'getchildren': lambda: []})()
    parent_dom = type('Mock', (), {'tag': 'p', 'text': 'before ', 'tail': None, 'getchildren': lambda: [child_dom]})()
    result = extract_text_array(parent_dom)
    assert result == [None, 'before ', 'bold', ' tail', None]
    
    # Test with callable tag (returns empty string)
    dom = type('Mock', (), {'tag': lambda: None, 'text': None, 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom)
    assert result == ''
    
    # Test squash_artifical_nl
    dom = type('Mock', (), {'tag': 'div', 'text': 'a', 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ['a']
    
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'a', None]
    
    # Test strip_artifical_nl
    dom = type('Mock', (), {'tag': 'div', 'text': '  text  ', 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['  text  ']
    
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, '  text  ', None]


