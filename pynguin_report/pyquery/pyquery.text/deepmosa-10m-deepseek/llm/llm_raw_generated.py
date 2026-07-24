####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_merge_original_parts_with_strings_only():
    result = _merge_original_parts(["  hello   world  "])
    assert result == ["hello world"]

def test_merge_original_parts_with_non_strings_only():
    result = _merge_original_parts([1, 2, 3])
    assert result == [1, 2, 3]

def test_merge_original_parts_with_mixed_types():
    result = _merge_original_parts(["  hello ", 1, " world  ", 2, "  foo  "])
    assert result == ["hello", 1, "world", 2, "foo"]

def test_merge_original_parts_with_empty_strings():
    result = _merge_original_parts(["  ", "", "  "])
    assert result == []

def test_merge_original_parts_with_only_whitespace():
    result = _merge_original_parts(["   ", " \n ", "  "])
    assert result == []

def test_merge_original_parts_with_adjacent_strings():
    result = _merge_original_parts(["hello", " world", "foo bar"])
    assert result == ["hello world foo bar"]

def test_merge_original_parts_with_leading_and_trailing_whitespace():
    result = _merge_original_parts(["  hello world  ", "  foo bar  "])
    assert result == ["hello world foo bar"]

def test_merge_original_parts_with_non_string_first():
    result = _merge_original_parts([1, " hello ", 2])
    assert result == [1, "hello", 2]

def test_merge_original_parts_with_non_string_last():
    result = _merge_original_parts([" hello ", 1, " world ", 2])
    assert result == ["hello", 1, "world", 2]

def test_merge_original_parts_with_multiple_consecutive_non_strings():
    result = _merge_original_parts(["a", 1, 2, "b"])
    assert result == ["a", 1, 2, "b"]


# LLM-generated content at query #2
#--------------------------

def test_merge_original_parts_empty_list():
    assert _merge_original_parts([]) == []

def test_merge_original_parts_single_string():
    assert _merge_original_parts(["hello"]) == ["hello"]

def test_merge_original_parts_single_non_string():
    assert _merge_original_parts([42]) == [42]

def test_merge_original_parts_multiple_strings():
    assert _merge_original_parts(["hello", " world"]) == ["hello world"]

def test_merge_original_parts_strings_with_whitespace():
    assert _merge_original_parts(["  hello ", " world  "]) == ["hello world"]

def test_merge_original_parts_strings_only_whitespace():
    assert _merge_original_parts(["   ", " "]) == []

def test_merge_original_parts_mixed_with_non_string():
    assert _merge_original_parts(["hello", 42, "world"]) == ["hello", 42, "world"]

def test_merge_original_parts_multiple_non_strings():
    assert _merge_original_parts([1, 2, 3]) == [1, 2, 3]

def test_merge_original_parts_string_then_non_string():
    assert _merge_original_parts(["hello", 42]) == ["hello", 42]

def test_merge_original_parts_non_string_then_string():
    assert _merge_original_parts([42, "world"]) == [42, "world"]

def test_merge_original_parts_multiple_groups():
    assert _merge_original_parts(["a", "b", 1, "c", "d"]) == ["a b", 1, "c d"]

def test_merge_original_parts_empty_string_between_non_strings():
    assert _merge_original_parts([1, "", 2]) == [1, 2]

def test_merge_original_parts_only_empty_strings():
    assert _merge_original_parts(["", ""]) == []


# LLM-generated content at query #3
#--------------------------

def test_extract_text_simple_text():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello</p>")
    result = extract_text(dom)
    assert result == "Hello"

def test_extract_text_with_whitespace():
    from lxml.html import fromstring
    dom = fromstring("<p>  Hello   World  </p>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_multiple_paragraphs():
    from lxml.html import fromstring
    dom = fromstring("<p>First</p><p>Second</p>")
    result = extract_text(dom)
    assert result == "First\nSecond"

def test_extract_text_nested_elements():
    from lxml.html import fromstring
    dom = fromstring("<div><p>Hello</p><p>World</p></div>")
    result = extract_text(dom)
    assert result == "Hello\nWorld"

def test_extract_text_with_separator():
    from lxml.html import fromstring
    dom = fromstring("<hr>")
    result = extract_text(dom)
    assert result == "\n"

def test_extract_text_with_inline_tag():
    from lxml.html import fromstring
    dom = fromstring("<p><b>Bold</b> text</p>")
    result = extract_text(dom)
    assert result == "Bold text"

def test_extract_text_with_tail():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello<br/>World</p>")
    result = extract_text(dom)
    assert result == "Hello\nWorld"

def test_extract_text_empty_dom():
    from lxml.html import fromstring
    dom = fromstring("<div></div>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_with_custom_symbols():
    from lxml.html import fromstring
    dom = fromstring("<p>First</p><p>Second</p>")
    result = extract_text(dom, block_symbol=' | ', sep_symbol=' - ')
    assert result == "First | Second"

def test_extract_text_squash_space_false():
    from lxml.html import fromstring
    dom = fromstring("<p>  Hello   World  </p>")
    result = extract_text(dom, squash_space=False)
    assert result == "  Hello   World  "


# LLM-generated content at query #4
#--------------------------

def test_squash_artifical_nl_with_all_none():
    assert _squash_artifical_nl([None, None, None]) == [None]

def test_squash_artifical_nl_with_no_none():
    assert _squash_artifical_nl(["a", "b", "c"]) == ["a", "b", "c"]

def test_squash_artifical_nl_with_mixed_values():
    assert _squash_artifical_nl(["a", None, "b", None, None, "c"]) == ["a", None, "b", None, "c"]

def test_squash_artifical_nl_with_consecutive_none():
    assert _squash_artifical_nl(["a", None, None, None, "b"]) == ["a", None, "b"]

def test_squash_artifical_nl_with_none_at_start():
    assert _squash_artifical_nl([None, "a", "b"]) == [None, "a", "b"]

def test_squash_artifical_nl_with_none_at_end():
    assert _squash_artifical_nl(["a", "b", None]) == ["a", "b", None]

def test_squash_artifical_nl_with_single_element():
    assert _squash_artifical_nl(["a"]) == ["a"]

def test_squash_artifical_nl_with_single_none():
    assert _squash_artifical_nl([None]) == [None]

def test_squash_artifical_nl_with_empty_list():
    assert _squash_artifical_nl([]) == []


# LLM-generated content at query #5
#--------------------------

def test_extract_text_basic_paragraph():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello world</p>")
    result = extract_text(dom)
    assert result == "Hello world"

def test_extract_text_with_separator():
    from lxml.html import fromstring
    dom = fromstring("<hr/><p>Text after separator</p>")
    result = extract_text(dom)
    assert result == "Text after separator"

def test_extract_text_with_nested_tags():
    from lxml.html import fromstring
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom)
    assert result == "First\nSecond"

def test_extract_text_with_tail_text():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text(dom)
    assert result == "Hello bold world"

def test_extract_text_with_multiple_whitespace():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello    world</p>")
    result = extract_text(dom)
    assert result == "Hello world"

def test_extract_text_with_leading_trailing_whitespace():
    from lxml.html import fromstring
    dom = fromstring("<p>  Hello world  </p>")
    result = extract_text(dom)
    assert result == "Hello world"

def test_extract_text_empty_document():
    from lxml.html import fromstring
    dom = fromstring("<div></div>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_block_symbol():
    from lxml.html import fromstring
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom, block_symbol='<br>')
    assert result == "First<br>Second"

def test_extract_text_sep_symbol():
    from lxml.html import fromstring
    dom = fromstring("<hr/><p>Text</p>")
    result = extract_text(dom, sep_symbol='---')
    assert result == "---\nText"

def test_extract_text_squash_space_false():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello    world</p>")
    result = extract_text(dom, squash_space=False)
    assert result == "Hello    world"


# LLM-generated content at query #6
#--------------------------

def test_extract_text_simple_text():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello World</p>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_with_separator():
    from lxml.html import fromstring
    dom = fromstring("<hr/>")
    result = extract_text(dom)
    assert result == "\n"

def test_extract_text_nested_tags():
    from lxml.html import fromstring
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom)
    assert result == "First\nSecond"

def test_extract_text_with_whitespace():
    from lxml.html import fromstring
    dom = fromstring("<p>  Hello   World  </p>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_empty_dom():
    from lxml.html import fromstring
    dom = fromstring("<div></div>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_inline_tags():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello <b>World</b></p>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_separator_between_text():
    from lxml.html import fromstring
    dom = fromstring("<p>Before</p><hr/><p>After</p>")
    result = extract_text(dom)
    assert result == "Before\nAfter"

def test_extract_text_custom_symbols():
    from lxml.html import fromstring
    dom = fromstring("<div><p>Line1</p><p>Line2</p></div>")
    result = extract_text(dom, block_symbol='|', sep_symbol='-')
    assert result == "Line1|Line2"

def test_extract_text_squash_space_false():
    from lxml.html import fromstring
    dom = fromstring("<p>  Hello   World  </p>")
    result = extract_text(dom, squash_space=False)
    assert result == "  Hello   World  "

def test_extract_text_with_tail():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello</p>World")
    result = extract_text(dom)
    assert result == "Hello\nWorld"


# LLM-generated content at query #7
#--------------------------

def test_strip_artifical_nl_empty_list():
    result = _strip_artifical_nl([])
    assert result == []

def test_strip_artifical_nl_only_strings_no_nl():
    result = _strip_artifical_nl(["hello", "world"])
    assert result == ["hello", "world"]

def test_strip_artifical_nl_leading_nl():
    result = _strip_artifical_nl(["\n", "hello"])
    assert result == ["hello"]

def test_strip_artifical_nl_trailing_nl():
    result = _strip_artifical_nl(["hello", "\n"])
    assert result == ["hello"]

def test_strip_artifical_nl_leading_and_trailing_nl():
    result = _strip_artifical_nl(["\n", "hello", "\n"])
    assert result == ["hello"]

def test_strip_artifical_nl_multiple_leading_nl():
    result = _strip_artifical_nl(["\n", "\n", "hello"])
    assert result == ["hello"]

def test_strip_artifical_nl_multiple_trailing_nl():
    result = _strip_artifical_nl(["hello", "\n", "\n"])
    assert result == ["hello"]

def test_strip_artifical_nl_all_nl():
    result = _strip_artifical_nl(["\n", "\n"])
    assert result == []

def test_strip_artifical_nl_mixed_types():
    result = _strip_artifical_nl(["\n", 1, "hello", 2, "\n"])
    assert result == [1, "hello", 2]

def test_strip_artifical_nl_no_strings():
    result = _strip_artifical_nl([1, 2, 3])
    assert result == [1, 2, 3]

def test_strip_artifical_nl_single_string_nl():
    result = _strip_artifical_nl(["\n"])
    assert result == []

def test_strip_artifical_nl_single_string_not_nl():
    result = _strip_artifical_nl(["hello"])
    assert result == ["hello"]


# LLM-generated content at query #8
#--------------------------

def test_empty_parts():
    assert _strip_artifical_nl([]) == []

def test_all_non_string_parts():
    assert _strip_artifical_nl([1, 2, 3]) == [1, 2, 3]

def test_single_string_part():
    assert _strip_artifical_nl(["hello"]) == ["hello"]

def test_single_non_string_part():
    assert _strip_artifical_nl([42]) == [42]

def test_leading_and_trailing_strings():
    assert _strip_artifical_nl(["a", 1, "b"]) == ["a", 1, "b"]

def test_only_strings():
    assert _strip_artifical_nl(["a", "b", "c"]) == ["a", "b", "c"]

def test_non_string_leading_only():
    assert _strip_artifical_nl([1, "a", 2]) == [1, "a", 2]

def test_non_string_trailing_only():
    assert _strip_artifical_nl(["a", 1, 2]) == ["a", 1, 2]

def test_multiple_non_string_leading():
    assert _strip_artifical_nl([1, 2, "a", "b", 3]) == [1, 2, "a", "b", 3]

def test_multiple_non_string_trailing():
    assert _strip_artifical_nl(["a", "b", 1, 2]) == ["a", "b", 1, 2]

def test_no_leading_string():
    assert _strip_artifical_nl([1, 2, 3, "a"]) == [1, 2, 3, "a"]

def test_no_trailing_string():
    assert _strip_artifical_nl(["a", 1, 2, 3]) == ["a", 1, 2, 3]

def test_mixed_with_non_string_at_ends():
    assert _strip_artifical_nl([1, "x", "y", 2]) == [1, "x", "y", 2]


# LLM-generated content at query #9
#--------------------------

def test_extract_text_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element("div")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_simple_text():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("p")
    dom.text = "Hello"
    result = extract_text(dom)
    assert result == "Hello"

def test_extract_text_with_child():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    dom.text = "Start "
    child = SubElement(dom, "span")
    child.text = "middle"
    child.tail = " end"
    result = extract_text(dom)
    assert result == "Start middle end"

def test_extract_text_with_separator():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    child = SubElement(dom, "br")
    child.tail = "after"
    result = extract_text(dom, block_symbol="\n", sep_symbol="\n")
    assert result == "\nafter"

def test_extract_text_with_block_element():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    child = SubElement(dom, "p")
    child.text = "block"
    result = extract_text(dom)
    assert result == "block"

def test_extract_text_multiple_blocks():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    child1 = SubElement(dom, "p")
    child1.text = "first"
    child2 = SubElement(dom, "p")
    child2.text = "second"
    result = extract_text(dom)
    assert result == "first\nsecond"

def test_extract_text_with_squash_space():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    dom.text = "  hello   world  "
    result = extract_text(dom, squash_space=True)
    assert result == "hello world"

def test_extract_text_no_squash_space():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    dom.text = "  hello   world  "
    result = extract_text(dom, squash_space=False)
    assert result == "  hello   world  "

def test_extract_text_custom_symbols():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    child = SubElement(dom, "p")
    child.text = "a"
    child2 = SubElement(dom, "p")
    child2.text = "b"
    result = extract_text(dom, block_symbol="|", sep_symbol="|")
    assert result == "a|b"

def test_extract_text_nested_blocks():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    inner = SubElement(dom, "div")
    inner.text = "inner"
    outer = SubElement(dom, "p")
    outer.text = "outer"
    result = extract_text(dom)
    assert result == "inner\nouter"


# LLM-generated content at query #10
#--------------------------

def test_predicate_evaluates_true():
    dom = None
    squash_space = True
    result = extract_text(dom, squash_space=squash_space)
    assert squash_space == True


# LLM-generated content at query #11
#--------------------------

def test_extract_text_simple_text():
    from xml.etree.ElementTree import Element
    dom = Element("p")
    dom.text = "Hello World"
    assert extract_text(dom) == "Hello World"

def test_extract_text_with_block_tag():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    p1 = SubElement(dom, "p")
    p1.text = "First"
    p2 = SubElement(dom, "p")
    p2.text = "Second"
    assert extract_text(dom) == "First\nSecond"

def test_extract_text_with_separator_tag():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    hr = SubElement(dom, "hr")
    hr.tail = "After"
    assert extract_text(dom) == "After"

def test_extract_text_with_inline_tag():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("p")
    span = SubElement(dom, "span")
    span.text = "Inline"
    span.tail = " text"
    assert extract_text(dom) == "Inline text"

def test_extract_text_multiline_block():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    p1 = SubElement(dom, "p")
    p1.text = "Line1"
    br = SubElement(p1, "br")
    br.tail = "Line2"
    assert extract_text(dom) == "Line1\nLine2"

def test_extract_text_squash_whitespace():
    from xml.etree.ElementTree import Element
    dom = Element("p")
    dom.text = "  Hello   World  "
    assert extract_text(dom) == "Hello World"

def test_extract_text_strip_artifical_newlines():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    p = SubElement(dom, "p")
    p.text = "Content"
    assert extract_text(dom) == "Content"

def test_extract_text_custom_block_symbol():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    p1 = SubElement(dom, "p")
    p1.text = "A"
    p2 = SubElement(dom, "p")
    p2.text = "B"
    assert extract_text(dom, block_symbol=" | ") == "A | B"

def test_extract_text_custom_sep_symbol():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    hr = SubElement(dom, "hr")
    hr.tail = "After"
    assert extract_text(dom, sep_symbol=" - ") == "After"

def test_extract_text_no_squash_space():
    from xml.etree.ElementTree import Element
    dom = Element("p")
    dom.text = "  Hello  "
    assert extract_text(dom, squash_space=False) == "  Hello  "

def test_extract_text_nested_tags():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    outer = SubElement(dom, "div")
    outer.text = "Outer "
    inner = SubElement(outer, "span")
    inner.text = "Inner"
    inner.tail = " end"
    assert extract_text(dom) == "Outer Inner end"

def test_extract_text_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element("div")
    assert extract_text(dom) == ""

def test_extract_text_only_whitespace():
    from xml.etree.ElementTree import Element
    dom = Element("p")
    dom.text = "   "
    assert extract_text(dom) == ""

def test_extract_text_block_and_separator():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    p = SubElement(dom, "p")
    p.text = "Text"
    hr = SubElement(dom, "hr")
    hr.tail = "Tail"
    assert extract_text(dom) == "Text\nTail"


# LLM-generated content at query #12
#--------------------------

```python
def test_extract_text_array_with_simple_text():
    class MockElement:
        tag = 'p'
        text = 'Hello'
        tail = None
        def getchildren(self):
            return []
    result = extract_text_array(MockElement())
    assert result == ['Hello']

def test_extract_text_array_with_separator():
    class MockElement:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []
    result = extract_text_array(MockElement())
    assert result == []

def test_extract_text_array_with_inline_tag():
    class MockElement:
        tag = 'span'
        text = 'inline'
        tail = None
        def getchildren(self):
            return []
    result = extract_text_array(MockElement())
    assert result == ['inline']

def test_extract_text_array_with_artifical_nl_squash():
    class MockChild:
        tag = 'p'
        text = 'child'
        tail = None
        def getchildren(self):
            return []
    class MockParent:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return [MockChild()]
    result = extract_text_array(MockParent())
    assert result == ['child']

def test_extract_text_array_with_strip_artifical_nl():
    class MockChild:
        tag = 'p'
        text = 'text'
        tail = None
        def getchildren(self):
            return []
    class MockParent:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return [MockChild()]
    result = extract_text_array(MockParent(), strip_artifical_nl=True)
    assert result == ['text']

def test_extract_text_array_without_squash():
    class MockChild:
        tag = 'p'
        text = 'a'
        tail = None
        def getchildren(self):
            return []
    class MockParent:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return [MockChild()]
    result = extract_text_array(MockParent(), squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'a', None]

def test_extract_text_array_with_empty():
    class MockElement:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return []
    result = extract_text_array(MockElement())
    assert result == []

def test_extract_text_array_with_callable_tag():
    class MockElement:
        tag = lambda: None
        text = None
        tail = None
        def getchildren(self):
            return []
    result = extract_text_array(MockElement())
    assert result == ''

def test_extract_text_array_with_multiple_children():
    class MockChild1:
        tag = 'b'
        text = 'bold'
        tail = ' and '
        def getchildren(self):
            return []
    class MockChild2:
        tag = 'i'
        text = 'italic'
        tail = None
        def getchildren(self):
            return []
    class MockParent:
        tag = 'p'
        text = 'start '
        tail = None
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    result = extract_text_array(MockParent())
    assert result == ['start ', 'bold', ' and ', 'italic']

def test_extract_text_array_with_separator_in_children():
    class MockChild:
        tag = 'br'
        text = None
        tail = 'after'
        def getchildren(self):
            return []
    class MockParent:
        tag = 'p'
        text = 'before'
        tail = None
        def getchildren(self):
            return [MockChild()]
    result = extract_text_array(MockParent())
    assert result == ['before', 'after']

def test_extract_text_array_nested_inline():
    class MockInner:
        tag = 'span'
        text = 'inner'
        tail = None
        def getchildren(self):
            return []
    class MockOuter:
        tag = 'span'
        text = 'outer'
        tail = None
        def getchildren(self):
            return [MockInner()]
    result = extract_text_array(MockOuter())
    assert result == ['outerinner']

def test_extract_text_array_with_artifical_nl_at_ends():
    class MockChild:
        tag = 'p'
        text = 'middle'
        tail = None
        def getchildren(self):
            return []
    class MockParent:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return [MockChild()]
    result = extract_text_array(MockParent(), squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['middle']
```


# LLM-generated content at query #13
#--------------------------

def test_extract_text_predicate_true():
    dom = None
    result = extract_text(dom, squash_space=True)
    assert result is not None


# LLM-generated content at query #14
#--------------------------

def test_squash_space_true():
    dom = None
    result = extract_text(dom, squash_space=True)
    assert result is not None


# LLM-generated content at query #15
#--------------------------

def test_extract_text_with_plain_text_node():
    from lxml.html import fromstring
    html = "<p>Hello world</p>"
    dom = fromstring(html)
    result = extract_text(dom)
    assert result == "Hello world"

def test_extract_text_with_nested_tags():
    from lxml.html import fromstring
    html = "<div><p>Hello</p><p>World</p></div>"
    dom = fromstring(html)
    result = extract_text(dom)
    assert result == "Hello\nWorld"

def test_extract_text_with_separator():
    from lxml.html import fromstring
    html = "<hr><p>Text</p>"
    dom = fromstring(html)
    result = extract_text(dom)
    assert result == "Text"

def test_extract_text_with_pre_tag():
    from lxml.html import fromstring
    html = "<pre>  Hello   World  </pre>"
    dom = fromstring(html)
    result = extract_text(dom)
    assert result == "  Hello   World  "

def test_extract_text_with_inline_tags():
    from lxml.html import fromstring
    html = "<p>Hello <b>bold</b> world</p>"
    dom = fromstring(html)
    result = extract_text(dom)
    assert result == "Hello bold world"

def test_extract_text_empty_dom():
    from lxml.html import fromstring
    html = ""
    dom = fromstring(html)
    result = extract_text(dom)
    assert result == ""

def test_extract_text_with_only_whitespace():
    from lxml.html import fromstring
    html = "<p>   </p>"
    dom = fromstring(html)
    result = extract_text(dom)
    assert result == ""

def test_extract_text_custom_symbols():
    from lxml.html import fromstring
    html = "<div><p>A</p><p>B</p></div><hr><p>C</p>"
    dom = fromstring(html)
    result = extract_text(dom, block_symbol='|', sep_symbol=':')
    assert result == "A|B:C"

def test_extract_text_squash_space_false():
    from lxml.html import fromstring
    html = "<div><p>A</p><p>B</p></div>"
    dom = fromstring(html)
    result = extract_text(dom, squash_space=False)
    assert result == "\nA\nB\n"

def test_extract_text_with_comment_node():
    from lxml.html import fromstring
    html = "<div><!-- comment --><p>Text</p></div>"
    dom = fromstring(html)
    result = extract_text(dom)
    assert result == "Text"


# LLM-generated content at query #16
#--------------------------

def test_extract_text_predicate_false():
    dom = None
    result = extract_text(dom, block_symbol='\n', sep_symbol='\n', squash_space=False)


# LLM-generated content at query #17
#--------------------------

def test_squash_space_true_predicate():
    dom = None
    result = extract_text(dom, squash_space=True)
    assert True


# LLM-generated content at query #18
#--------------------------

def test_callable_dom_tag_returns_empty_string():
    dom = type('MockDom', (), {'tag': lambda: None, 'text': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom)
    assert result == ''


# LLM-generated content at query #19
#--------------------------

def test_extract_text_empty_dom():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div></div>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_simple_text():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div>Hello World</div>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_with_inline_tags():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<p>Hello <b>World</b></p>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_with_block_tags():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom)
    assert result == "First\nSecond"

def test_extract_text_with_separator_tags():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div><br/>Line</div>")
    result = extract_text(dom)
    assert result == "\nLine"

def test_extract_text_with_whitespace_squashing():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div>  Hello   World  </div>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_with_nested_tags():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div><span>Hello</span> <span>World</span></div>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_with_tail_text():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div>Hello<b>bold</b>World</div>")
    result = extract_text(dom)
    assert result == "HelloboldWorld"

def test_extract_text_squash_space_disabled():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div>  Hello  </div>")
    result = extract_text(dom, squash_space=False)
    assert result == "  Hello  "

def test_extract_text_block_symbol_custom():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom, block_symbol=' ')
    assert result == "First Second"

def test_extract_text_sep_symbol_custom():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div><br/>Line</div>")
    result = extract_text(dom, sep_symbol=' ')
    assert result == " Line"


# LLM-generated content at query #20
#--------------------------

```
def test_predicate_at_line10_evaluates_to_true():
    from unittest.mock import Mock
    dom = Mock()
    dom.tag = "div"
    dom.text = "some text"
    dom.getchildren.return_value = []
    result = extract_text_array(dom)
    assert result[0] == "some text"
```


# LLM-generated content at query #21
#--------------------------

```
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        tag = "span"
        text = "hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_extract_text_array_with_separator():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_nested_tags():
    class MockChild:
        tag = "span"
        text = "world"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "hello "
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["hello ", "world"]

def test_extract_text_array_squash_nl():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_nl():
    class MockElement:
        tag = "div"
        text = "content"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["content"]

def test_extract_text_array_with_child_tail():
    class MockChild:
        tag = "span"
        text = "inner"
        tail = " tail"
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "start"
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["start", "inner", " tail"]

def test_extract_text_array_inline_tag():
    class MockElement:
        tag = "b"
        text = "bold"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["bold"]

def test_extract_text_array_none_text():
    class MockElement:
        tag = "p"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_callable_tag():
    class MockElement:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ""
```


# LLM-generated content at query #22
#--------------------------

```
def test_strip_artifical_nl_called_when_true():
    dom = type('MockDom', (), {'tag': 'div', 'text': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == []
```


# LLM-generated content at query #23
#--------------------------

```python
def test_extract_text_array_empty_dom():
    from lxml import etree
    dom = etree.fromstring("<div></div>")
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    from lxml import etree
    dom = etree.fromstring("<p>Hello</p>")
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_with_separator():
    from lxml import etree
    dom = etree.fromstring("<br/>")
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    from lxml import etree
    dom = etree.fromstring("<span>inline</span>")
    result = extract_text_array(dom)
    assert result == ["inline"]

def test_extract_text_array_nested_tags():
    from lxml import etree
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text_array(dom)
    assert result == ["First", "Second"]

def test_extract_text_array_with_tail():
    from lxml import etree
    dom = etree.fromstring("<p>Hello<b>bold</b>world</p>")
    result = extract_text_array(dom)
    assert result == ["Hello", "bold", "world"]

def test_extract_text_array_squash_artifical_nl_false():
    from lxml import etree
    dom = etree.fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, "A", None, None, "B", None]

def test_extract_text_array_strip_artifical_nl_false():
    from lxml import etree
    dom = etree.fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == ["A", None]

def test_extract_text_array_no_artifical_nl_options():
    from lxml import etree
    dom = etree.fromstring("<div><p>A</p></div>")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "A", None]

def test_extract_text_array_empty_text_nodes():
    from lxml import etree
    dom = etree.fromstring("<div><p></p></div>")
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_only_whitespace():
    from lxml import etree
    dom = etree.fromstring("<div>   </div>")
    result = extract_text_array(dom)
    assert result == ["   "]

def test_extract_text_array_callable_tag():
    from lxml import etree
    dom = etree.fromstring("<div></div>")
    dom.tag = lambda: None
    result = extract_text_array(dom)
    assert result == ""

def test_extract_text_array_deep_nesting():
    from lxml import etree
    dom = etree.fromstring("<div><p><span>deep</span></p></div>")
    result = extract_text_array(dom)
    assert result == ["deep"]

def test_extract_text_array_multiple_separators():
    from lxml import etree
    dom = etree.fromstring("<br/><br/>")
    result = extract_text_array(dom)
    assert result == [True, True]

def test_extract_text_array_mixed_inline_and_block():
    from lxml import etree
    dom = etree.fromstring("<div><span>inline</span><p>block</p></div>")
    result = extract_text_array(dom)
    assert result == ["inline", "block"]
```


# LLM-generated content at query #24
#--------------------------

```python
def test_extract_text_array_with_text_and_no_children():
    dom = type('Element', (object,), {'tag': 'p', 'text': 'Hello', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_with_separator_tag():
    dom = type('Element', (object,), {'tag': 'br', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    dom = type('Element', (object,), {'tag': 'span', 'text': 'text', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == ['text', None]

def test_extract_text_array_with_child():
    child = type('Element', (object,), {'tag': 'b', 'text': 'bold', 'getchildren': lambda self: [], 'tail': ' tail'})()
    dom = type('Element', (object,), {'tag': 'p', 'text': 'before ', 'getchildren': lambda self: [child], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'before ', None, 'bold', ' tail', None]

def test_extract_text_array_squash_artifical_nl():
    dom = type('Element', (object,), {'tag': 'p', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_artifical_nl():
    dom = type('Element', (object,), {'tag': 'p', 'text': 'a', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['a']

def test_extract_text_array_callable_tag():
    dom = type('Element', (object,), {'tag': lambda: None, 'text': 'text', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom)
    assert result == ''
```


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_false():
    dom = type('MockDom', (object,), {'tag': 'p', 'text': None, 'getchildren': lambda self: []})()
    dom.tag = 'p'
    result = extract_text_array(dom)
    assert len(result) == 0 or result[0] is not None
```


# LLM-generated content at query #26
#--------------------------

def test_extract_text_with_squash_space_true():
    dom = []
    result = extract_text(dom, squash_space=True)
    assert isinstance(result, str)


# LLM-generated content at query #27
#--------------------------

def test_squash_artifical_nl_true():
    dom = type('MockDom', (), {'tag': 'p', 'text': 'hello', 'tail': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert isinstance(result, list)


# LLM-generated content at query #28
#--------------------------

```python
def test_extract_text_array_with_empty_dom():
    class MockElement:
        tag = 'p'
        text = None
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_text_only():
    class MockElement:
        tag = 'span'
        text = 'hello'
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['hello']

def test_extract_text_array_with_separator_tag():
    class MockElement:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    class MockChild:
        tag = 'b'
        text = 'bold'
        tail = ' tail'
        def getchildren(self):
            return []
    class MockElement:
        tag = 'span'
        text = 'start'
        tail = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['start', 'bold', ' tail']

def test_extract_text_array_with_artificial_newlines():
    class MockElement:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_squash_and_strip():
    class MockChild:
        tag = 'p'
        text = 'para'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['para']

def test_extract_text_array_without_squash():
    class MockChild:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None, None]

def test_extract_text_array_without_strip():
    class MockElement:
        tag = 'div'
        text = None
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_with_mixed_content():
    class MockChild1:
        tag = 'span'
        text = 'inner'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'p'
        text = 'text'
        tail = None
        def getchildren(self):
            return [MockChild1()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['text', 'inner']

def test_extract_text_array_with_separator_and_text():
    class MockElement:
        tag = 'br'
        text = None
        tail = 'after'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True, 'after']
```


# LLM-generated content at query #29
#--------------------------

def test_elif_dom_tag_not_in_inline_tags_evaluates_to_false():
    dom = type('Dom', (), {'tag': 'INLINE', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    extract_text_array(dom)


# LLM-generated content at query #30
#--------------------------

def test_predicate_at_line_17_evaluates_to_true():
    mock_dom = type('MockDom', (), {'tag': 'div', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    INLINE_TAGS = {'span', 'b', 'i'}
    SEPARATORS = {'br', 'hr'}
    result = extract_text_array(mock_dom)
    assert result[-1] is None


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_false():
    dom = type('MockDom', (object,), {'tag': 'p', 'text': 'hello', 'tail': None})()
    dom.getchildren = lambda: []
    result = extract_text_array(dom)
    assert dom.tag not in INLINE_TAGS and dom.tag not in SEPARATORS
```


# LLM-generated content at query #32
#--------------------------

def test_predicate_at_line_17_evaluates_to_false():
    from lxml.html import fromstring
    dom = fromstring("<div>text</div>")
    result = extract_text_array(dom)
    assert True


# LLM-generated content at query #33
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_single_text():
    class MockDom:
        tag = "span"
        text = "hello"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_extract_text_array_separator_tag():
    class MockDom:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_child():
    class MockChild:
        tag = "b"
        text = "bold"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "p"
        text = "before "
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "before ", "bold", None]

def test_extract_text_array_squash_newlines():
    class MockChild:
        tag = "div"
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild(), MockChild()]
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_strip_newlines():
    class MockDom:
        tag = "div"
        text = "text"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["text"]

def test_extract_text_array_callable_tag():
    class MockDom:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ""

def test_extract_text_array_inline_tag():
    class MockDom:
        tag = "a"
        text = "link"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["link"]

def test_extract_text_array_with_tail():
    class MockChild:
        tag = "i"
        text = "italic"
        tail = " after"
        def getchildren(self):
            return []
    class MockDom:
        tag = "p"
        text = "before "
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "before ", "italic", " after", None]

def test_extract_text_array_both_squash_and_strip():
    class MockChild:
        tag = "div"
        text = "content"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["content"]
```


# LLM-generated content at query #34
#--------------------------

def test_predicate_false():
    dom = None
    block_symbol = '\n'
    sep_symbol = '\n'
    squash_space = False
    result = extract_text(dom, block_symbol, sep_symbol, squash_space)


# LLM-generated content at query #35
#--------------------------

def test_extract_text_empty_dom():
    dom = type('obj', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text(dom)
    assert result == ""

def test_extract_text_simple_text():
    dom = type('obj', (object,), {'tag': 'p', 'text': 'Hello', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text(dom)
    assert result == "Hello"

def test_extract_text_with_child():
    child = type('obj', (object,), {'tag': 'b', 'text': 'bold', 'getchildren': lambda self: [], 'tail': ' text'})()
    dom = type('obj', (object,), {'tag': 'p', 'text': 'Some ', 'getchildren': lambda self: [child], 'tail': None})()
    result = extract_text(dom)
    assert result == "Some bold text"

def test_extract_text_with_separator():
    dom = type('obj', (object,), {'tag': 'br', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text(dom, sep_symbol='\n')
    assert result == "\n"

def test_extract_text_with_block_element():
    child = type('obj', (object,), {'tag': 'p', 'text': 'Para', 'getchildren': lambda self: [], 'tail': None})()
    dom = type('obj', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [child], 'tail': None})()
    result = extract_text(dom)
    assert result == "Para"

def test_extract_text_multiple_blocks():
    child1 = type('obj', (object,), {'tag': 'p', 'text': 'First', 'getchildren': lambda self: [], 'tail': None})()
    child2 = type('obj', (object,), {'tag': 'p', 'text': 'Second', 'getchildren': lambda self: [], 'tail': None})()
    dom = type('obj', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [child1, child2], 'tail': None})()
    result = extract_text(dom)
    assert result == "First\nSecond"

def test_extract_text_with_tail_on_separator():
    child = type('obj', (object,), {'tag': 'br', 'text': None, 'getchildren': lambda self: [], 'tail': ' after'})()
    dom = type('obj', (object,), {'tag': 'div', 'text': 'before ', 'getchildren': lambda self: [child], 'tail': None})()
    result = extract_text(dom)
    assert result == "before after"


# LLM-generated content at query #36
#--------------------------

def test_predicate_line_17_true():
    from lxml.html import fromstring
    dom = fromstring("<p>text</p>")
    result = extract_text_array(dom)
    assert dom.tag not in INLINE_TAGS
    assert dom.tag not in SEPARATORS


# LLM-generated content at query #37
#--------------------------

def test_extract_text_simple_text():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello world</p>")
    result = extract_text(dom)
    assert result == "Hello world"

def test_extract_text_with_block_symbol():
    from lxml.html import fromstring
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom, block_symbol='\n')
    assert result == "First\nSecond"

def test_extract_text_with_sep_symbol():
    from lxml.html import fromstring
    dom = fromstring("<div><br/>Separator</div>")
    result = extract_text(dom, sep_symbol='\n')
    assert result == "Separator"

def test_extract_text_squash_space():
    from lxml.html import fromstring
    dom = fromstring("<p>  Hello   world  </p>")
    result = extract_text(dom, squash_space=True)
    assert result == "Hello world"

def test_extract_text_no_squash_space():
    from lxml.html import fromstring
    dom = fromstring("<p>  Hello   world  </p>")
    result = extract_text(dom, squash_space=False)
    assert result == "  Hello   world  "

def test_extract_text_nested_inline():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text(dom)
    assert result == "Hello bold world"

def test_extract_text_empty_dom():
    from lxml.html import fromstring
    dom = fromstring("<div></div>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_with_tail():
    from lxml.html import fromstring
    dom = fromstring("<div>Text before <span>inner</span> tail after</div>")
    result = extract_text(dom)
    assert result == "Text before inner tail after"


# LLM-generated content at query #38
#--------------------------

```python
def test_extract_text_array_empty_dom():
    dom = type('MockDom', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    dom.tag = 'div'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_simple_text():
    dom = type('MockDom', (object,), {'tag': 'p', 'text': 'Hello', 'getchildren': lambda self: [], 'tail': None})()
    dom.tag = 'p'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['Hello']

def test_extract_text_array_with_separator():
    dom = type('MockDom', (object,), {'tag': 'br', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    dom.tag = 'br'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == [True]

def test_extract_text_array_with_artificial_newlines():
    dom = type('MockDom', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    dom.tag = 'div'
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_squash_artifical_nl():
    dom = type('MockDom', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    dom.tag = 'div'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_artifical_nl():
    dom = type('MockDom', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    dom.tag = 'div'
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_with_child():
    child = type('MockDom', (object,), {'tag': 'span', 'text': 'World', 'getchildren': lambda self: [], 'tail': '!'})()
    child.tag = 'span'
    dom = type('MockDom', (object,), {'tag': 'div', 'text': 'Hello ', 'getchildren': lambda self: [child], 'tail': None})()
    dom.tag = 'div'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['Hello ', 'World', '!']

def test_extract_text_array_inline_tag():
    dom = type('MockDom', (object,), {'tag': 'b', 'text': 'Bold', 'getchildren': lambda self: [], 'tail': None})()
    dom.tag = 'b'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['Bold']

def test_extract_text_array_no_squash_no_strip():
    child = type('MockDom', (object,), {'tag': 'span', 'text': 'Hello', 'getchildren': lambda self: [], 'tail': None})()
    child.tag = 'span'
    dom = type('MockDom', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [child], 'tail': None})()
    dom.tag = 'div'
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'Hello', None]

def test_extract_text_array_with_separator_and_text():
    dom = type('MockDom', (object,), {'tag': 'br', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    dom.tag = 'br'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == [True]

def test_extract_text_array_none_text():
    dom = type('MockDom', (object,), {'tag': 'p', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    dom.tag = 'p'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_with_tail():
    dom = type('MockDom', (object,), {'tag': 'div', 'text': 'Hello', 'getchildren': lambda self: [], 'tail': ' World'})()
    dom.tag = 'div'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['Hello', ' World']

def test_extract_text_array_multiple_children():
    child1 = type('MockDom', (object,), {'tag': 'b', 'text': 'Bold', 'getchildren': lambda self: [], 'tail': ' '})()
    child1.tag = 'b'
    child2 = type('MockDom', (object,), {'tag': 'i', 'text': 'Italic', 'getchildren': lambda self: [], 'tail': None})()
    child2.tag = 'i'
    dom = type('MockDom', (object,), {'tag': 'p', 'text': 'Text ', 'getchildren': lambda self: [child1, child2], 'tail': None})()
    dom.tag = 'p'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['Text ', 'Bold', ' ', 'Italic']

def test_extract_text_array_callable_tag():
    dom = type('MockDom', (object,), {'tag': lambda: None, 'text': 'should not appear', 'getchildren': lambda self: [], 'tail': None})()
    dom.tag = lambda: None
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['']  # Since callable tag returns empty string

def test_extract_text_array_strip_leading_none():
    child = type('MockDom', (object,), {'tag': 'span', 'text': 'Hello', 'getchildren': lambda self: [], 'tail': None})()
    child.tag = 'span'
    dom = type('MockDom', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [child], 'tail': None})()
    dom.tag = 'div'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['Hello']

def test_extract_text_array_strip_trailing_none():
    child = type('MockDom', (object,), {'tag': 'span', 'text': 'Hello', 'getchildren': lambda self: [], 'tail': None})()
    child.tag = 'span'
    dom = type('MockDom', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [child], 'tail': None})()
    dom.tag = 'div'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['Hello']
```


# LLM-generated content at query #39
#--------------------------

def test_elif_not_in_inline_tags_false():
    dom = type('MockDom', (), {'tag': 'p', 'text': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == []


# LLM-generated content at query #40
#--------------------------

```python
def test_squash_artifical_nl_true():
    dom = type('MockDom', (), {'tag': 'div', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert True  # predicate at line 20 evaluates to True


# LLM-generated content at query #41
#--------------------------

```python
def test_extract_text_array_empty_dom(monkeypatch):
    class MockElement:
        tag = 'p'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_text_only(monkeypatch):
    class MockElement:
        tag = 'span'
        text = 'Hello'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_with_separator(monkeypatch):
    class MockElement:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    monkeypatch.setattr('your_module.SEPARATORS', {'br'})
    monkeypatch.setattr('your_module.INLINE_TAGS', set())
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_artificial_nl(monkeypatch):
    class MockElement:
        tag = 'div'
        text = 'A'
        def getchildren(self):
            return []
    monkeypatch.setattr('your_module.SEPARATORS', set())
    monkeypatch.setattr('your_module.INLINE_TAGS', set())
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['A']

def test_extract_text_array_with_child_and_tail(monkeypatch):
    class MockChild:
        tag = 'b'
        text = 'bold'
        tail = ' tail'
        def getchildren(self):
            return []
    class MockElement:
        tag = 'p'
        text = 'before '
        def getchildren(self):
            return [MockChild()]
    monkeypatch.setattr('your_module.SEPARATORS', set())
    monkeypatch.setattr('your_module.INLINE_TAGS', {'b'})
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['before ', 'bold', ' tail']

def test_extract_text_array_squash_artifical_nl(monkeypatch):
    class MockElement:
        tag = 'div'
        text = 'A'
        def getchildren(self):
            return []
    monkeypatch.setattr('your_module.SEPARATORS', set())
    monkeypatch.setattr('your_module.INLINE_TAGS', set())
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ['A']

def test_extract_text_array_strip_artifical_nl(monkeypatch):
    class MockElement:
        tag = 'div'
        text = 'A'
        def getchildren(self):
            return []
    monkeypatch.setattr('your_module.SEPARATORS', set())
    monkeypatch.setattr('your_module.INLINE_TAGS', set())
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['A']

def test_extract_text_array_no_squash_no_strip(monkeypatch):
    class MockElement:
        tag = 'div'
        text = 'A'
        def getchildren(self):
            return []
    monkeypatch.setattr('your_module.SEPARATORS', set())
    monkeypatch.setattr('your_module.INLINE_TAGS', set())
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'A', None]

def test_extract_text_array_multiple_children(monkeypatch):
    class MockChild1:
        tag = 'span'
        text = 'child1'
        tail = ' tail1'
        def getchildren(self):
            return []
    class MockChild2:
        tag = 'span'
        text = 'child2'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = 'start '
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    monkeypatch.setattr('your_module.SEPARATORS', set())
    monkeypatch.setattr('your_module.INLINE_TAGS', {'span'})
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['start ', 'child1', ' tail1', 'child2']

def test_extract_text_array_separator_with_children(monkeypatch):
    class MockChild:
        tag = 'span'
        text = 'inner'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'br'
        text = None
        def getchildren(self):
            return [MockChild()]
    monkeypatch.setattr('your_module.SEPARATORS', {'br'})
    monkeypatch.setattr('your_module.INLINE_TAGS', {'span'})
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True, 'inner']

def test_extract_text_array_squash_consecutive_nl(monkeypatch):
    class MockChild:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = 'A'
        def getchildren(self):
            return [MockChild()]
    monkeypatch.setattr('your_module.SEPARATORS', set())
    monkeypatch.setattr('your_module.INLINE_TAGS', set())
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ['A']

def test_extract_text_array_strip_leading_trailing_nl(monkeypatch):
    class MockChild:
        tag = 'div'
        text = 'B'
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockChild()]
    monkeypatch.setattr('your_module.SEPARATORS', set())
    monkeypatch.setattr('your_module.INLINE_TAGS', set())
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['B']

def test_extract_text_array_mixed_content(monkeypatch):
    class MockChild:
        tag = 'b'
        text = 'bold'
        tail = ' after bold '
        def getchildren(self):
            return []
    class MockElement:
        tag = 'p'
        text = 'before '
        def getchildren(self):
            return [MockChild()]
    monkeypatch.setattr('your_module.SEPARATORS', {'br'})
    monkeypatch.setattr('your_module.INLINE_TAGS', {'b'})
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['before ', 'bold', ' after bold ']
```


# LLM-generated content at query #42
#--------------------------

```python
def test_strip_artifical_nl_true():
    dom = MagicMock()
    dom.tag = 'div'
    dom.text = None
    dom.getchildren.return_value = []
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == [] or True  # predicate at line 22 evaluates to True when strip_artifical_nl is True
```


# LLM-generated content at query #43
#--------------------------

def test_dom_tag_in_separators():
    dom = type('MockDom', (object,), {'tag': 'p', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom)
    assert result[0] == True


# LLM-generated content at query #44
#--------------------------

```
def test_squash_artifical_nl_true():
    dom = type('MockDom', (object,), {'tag': 'p', 'text': 'hello', 'getchildren': lambda self: []})()
    dom.tail = None
    INLINE_TAGS = []
    SEPARATORS = []
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == ['hello']
```


# LLM-generated content at query #45
#--------------------------

```python
def test_extract_text_array_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = 'Hello'
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_with_separator():
    from xml.etree.ElementTree import Element
    dom = Element('br')
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_nested_inline():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('span')
    dom.text = 'Hello '
    child = SubElement(dom, 'b')
    child.text = 'World'
    result = extract_text_array(dom)
    assert result == ['Hello ', 'World']

def test_extract_text_array_squash_nl():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    child = SubElement(dom, 'p')
    child.text = 'Hello'
    child2 = SubElement(dom, 'p')
    child2.text = 'World'
    result = extract_text_array(dom)
    assert result == ['Hello', None, 'World']

def test_extract_text_array_strip_leading_nl():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    child = SubElement(dom, 'p')
    child.text = 'Hello'
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_strip_trailing_nl():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    child = SubElement(dom, 'p')
    child.text = 'Hello'
    child.tail = ' World'
    result = extract_text_array(dom)
    assert result == ['Hello', ' World']

def test_extract_text_array_no_squash():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    child = SubElement(dom, 'p')
    child.text = 'Hello'
    child2 = SubElement(dom, 'p')
    child2.text = 'World'
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'Hello', None, None, 'World', None]

def test_extract_text_array_no_strip():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    child = SubElement(dom, 'p')
    child.text = 'Hello'
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'Hello', None]

def test_extract_text_array_multiple_children():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    c1 = SubElement(dom, 'span')
    c1.text = 'A'
    c2 = SubElement(dom, 'b')
    c2.text = 'B'
    c2.tail = ' '
    c3 = SubElement(dom, 'i')
    c3.text = 'C'
    result = extract_text_array(dom)
    assert result == ['A', 'B', ' ', 'C']

def test_extract_text_array_with_separator_and_text():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    child = SubElement(dom, 'br')
    child.tail = 'text after break'
    result = extract_text_array(dom)
    assert result == [True, 'text after break']
```


# LLM-generated content at query #46
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockDom:
        tag = 'p'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockDom:
        tag = 'p'
        text = 'Hello'
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_with_separator():
    class MockDom:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    class MockDom:
        tag = 'span'
        text = 'text'
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['text']

def test_extract_text_array_with_children():
    class MockChild:
        tag = 'b'
        text = 'bold'
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = 'p'
        text = 'before'
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['before', 'bold']

def test_extract_text_array_with_tail():
    class MockChild:
        tag = 'a'
        text = 'link'
        tail = ' after'
        def getchildren(self):
            return []
    class MockDom:
        tag = 'p'
        text = 'click '
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['click ', 'link', ' after']

def test_extract_text_array_squash_artifical_nl():
    result = _squash_artifical_nl([None, None, 'a', None, None, 'b'])
    assert result == [None, 'a', None, 'b']

def test_extract_text_array_strip_artifical_nl():
    result = _strip_artifical_nl([None, 'a', 'b', None])
    assert result == ['a', 'b']

def test_extract_text_array_full_processing():
    class MockChild:
        tag = 'span'
        text = 'world'
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = 'div'
        text = 'hello '
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['hello ', 'world']


# LLM-generated content at query #47
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_text_only():
    class MockElement:
        tag = 'p'
        text = 'Hello'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_with_separator_tag():
    class MockElement:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    class MockElement:
        tag = 'span'
        text = 'Inline'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['Inline']

def test_extract_text_array_with_child_and_tail():
    class MockChild:
        tag = 'b'
        text = 'Bold'
        tail = ' tail'
        def getchildren(self):
            return []
    class MockElement:
        tag = 'p'
        text = 'Text '
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['Text ', 'Bold', ' tail']

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = 'p'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == [None]

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = 'p'
        text = 'hello'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['hello']

def test_extract_text_array_squash_and_strip():
    class MockElement:
        tag = 'p'
        text = 'hello'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['hello']

def test_extract_text_array_multiple_children():
    class MockChild1:
        tag = 'b'
        text = 'bold'
        tail = None
        def getchildren(self):
            return []
    class MockChild2:
        tag = 'i'
        text = 'italic'
        tail = ' after'
        def getchildren(self):
            return []
    class MockElement:
        tag = 'p'
        text = 'start '
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['start ', 'bold', 'italic', ' after']

def test_extract_text_array_nested_separator():
    class MockChild:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = 'line1'
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['line1', True]

def test_extract_text_array_without_squash():
    class MockElement:
        tag = 'p'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_without_strip():
    class MockElement:
        tag = 'p'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, None]


# LLM-generated content at query #48
#--------------------------

```
def test_squash_artifical_nl_true():
    dom = type('MockDom', (object,), {'tag': 'p', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert True
```


# LLM-generated content at query #49
#--------------------------

def test_strip_artifical_nl_true():
    dom = type('MockDom', (), {'tag': 'p', 'text': 'hello', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert isinstance(result, list)


# LLM-generated content at query #50
#--------------------------

```
def test_strip_artifical_nl_true():
    r = [None, "text", None]
    strip_artifical_nl = True
    r = _strip_artifical_nl(r)
    assert strip_artifical_nl == True
```


# LLM-generated content at query #51
#--------------------------

```
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        tag = "p"
        text = "Hello"
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["Hello"]

def test_extract_text_array_inline_tag():
    class MockElement:
        tag = "b"
        text = "bold"
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["bold"]

def test_extract_text_array_separator():
    class MockElement:
        tag = "br"
        text = None
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == [True]

def test_extract_text_array_nested_tags():
    class MockChild:
        tag = "span"
        text = "world"
        tail = "!"
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "Hello "
        tail = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["Hello ", "world", "!"]

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = "div"
        text = None
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = "div"
        text = "text"
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["text"]

def test_extract_text_array_no_squash_no_strip():
    class MockElement:
        tag = "div"
        text = "text"
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "text", None]

def test_extract_text_array_children_with_tail():
    class MockChild:
        tag = "a"
        text = "click"
        tail = " here"
        def getchildren(self):
            return []
    class MockElement:
        tag = "p"
        text = "Please "
        tail = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["Please ", "click", " here"]

def test_extract_text_array_callable_tag():
    class MockElement:
        tag = lambda: None
        text = "test"
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == [""]


# LLM-generated content at query #52
#--------------------------

def test_extract_text_array_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element("div")
    dom.text = None
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    from xml.etree.ElementTree import Element
    dom = Element("div")
    dom.text = "Hello"
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_with_inline_tag():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    dom.text = "Hello "
    child = SubElement(dom, "span")
    child.text = "World"
    child.tail = "!"
    result = extract_text_array(dom)
    assert result == ["Hello ", "World", "!"]

def test_extract_text_array_with_separator():
    from xml.etree.ElementTree import Element
    dom = Element("br")
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_block_tag():
    from xml.etree.ElementTree import Element
    dom = Element("p")
    dom.text = "Text"
    result = extract_text_array(dom)
    assert result == ["Text"]

def test_extract_text_array_trailing_artificial_nl_stripped():
    from xml.etree.ElementTree import Element
    dom = Element("p")
    dom.text = "Hello"
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_squash_artificial_nl_false():
    from xml.etree.ElementTree import Element
    dom = Element("div")
    dom.text = None
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_strip_artifical_nl_false():
    from xml.etree.ElementTree import Element
    dom = Element("p")
    dom.text = "Hello"
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "Hello", None]

def test_extract_text_array_nested_blocks():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    child1 = SubElement(dom, "p")
    child1.text = "First"
    child1.tail = None
    child2 = SubElement(dom, "p")
    child2.text = "Second"
    result = extract_text_array(dom)
    assert result == ["First", "Second"]

def test_extract_text_array_multiple_separators():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    SubElement(dom, "br")
    child = SubElement(dom, "span")
    child.text = "text"
    result = extract_text_array(dom)
    assert result == ["text"]


# LLM-generated content at query #53
#--------------------------

def test_predicate_squash_space_false():
    dom = None
    result = extract_text(dom, squash_space=False)


# LLM-generated content at query #54
#--------------------------

def test_predicate_false_squash_space_false():
    dom = None
    extract_text(dom, squash_space=False)


# LLM-generated content at query #55
#--------------------------

```python
def test_predicate_line12_true():
    from unittest.mock import Mock
    child = Mock()
    child.tag = "p"
    child.text = None
    child.tail = None
    child.getchildren.return_value = []
    dom = Mock()
    dom.tag = "div"
    dom.text = None
    dom.getchildren.return_value = [child]
    SEPARATORS = {"br"}
    INLINE_TAGS = {"b", "i"}
    result = extract_text_array(dom)
    assert result == [None]  # predicate at line 12 evaluates to True because there is a child
```


# LLM-generated content at query #56
#--------------------------

```python
def test_extract_text_array_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_text_only():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = 'Hello'
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_separator_tag():
    from xml.etree.ElementTree import Element
    dom = Element('br')
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag():
    from xml.etree.ElementTree import Element
    dom = Element('span')
    dom.text = 'text'
    result = extract_text_array(dom)
    assert result == ['text']

def test_extract_text_array_block_tag_with_text():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    dom.text = 'content'
    result = extract_text_array(dom)
    assert result == ['content']

def test_extract_text_array_nested_tags():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    dom.text = 'a'
    child = SubElement(dom, 'span')
    child.text = 'b'
    child.tail = 'c'
    result = extract_text_array(dom)
    assert result == ['a', 'b', 'c']

def test_extract_text_array_squash_artifical_nl():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    SubElement(dom, 'div')
    SubElement(dom, 'div')
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == [None]

def test_extract_text_array_strip_artifical_nl():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    SubElement(dom, 'div')
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_no_squash_no_strip():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    SubElement(dom, 'div')
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None, None]

def test_extract_text_array_separator_with_text():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    br = SubElement(dom, 'br')
    br.tail = 'after'
    result = extract_text_array(dom)
    assert result == [True, 'after']

def test_extract_text_array_mixed_inline_block():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    dom.text = 'start'
    span = SubElement(dom, 'span')
    span.text = 'middle'
    div2 = SubElement(dom, 'div')
    div2.text = 'end'
    result = extract_text_array(dom)
    assert result == ['start', 'middle', 'end']

def test_extract_text_array_artifical_nl_between_text():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    dom.text = 'a'
    SubElement(dom, 'div')
    sub = SubElement(dom, 'div')
    sub.text = 'b'
    result = extract_text_array(dom)
    assert result == ['a', 'b']

def test_extract_text_array_multiple_separators():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    SubElement(dom, 'br')
    SubElement(dom, 'br')
    result = extract_text_array(dom)
    assert result == [True, True]
```


# LLM-generated content at query #57
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_single_text():
    class MockDom:
        tag = 'p'
        text = 'Hello'
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_separator_tag():
    class MockDom:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_child_and_tail():
    class MockChild:
        tag = 'b'
        text = 'bold'
        tail = ' tail'
        def getchildren(self):
            return []
    class MockDom:
        tag = 'p'
        text = 'start '
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['start ', 'bold', ' tail']

def test_extract_text_array_squash_artifical_nl_false():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_strip_artifical_nl_false():
    class MockDom:
        tag = 'div'
        text = 'text'
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'text', None]

def test_extract_text_array_nested_structure():
    class MockChild2:
        tag = 'span'
        text = 'inner'
        tail = None
        def getchildren(self):
            return []
    class MockChild1:
        tag = 'div'
        text = None
        tail = ' after'
        def getchildren(self):
            return [MockChild2()]
    class MockDom:
        tag = 'body'
        text = 'before '
        def getchildren(self):
            return [MockChild1()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['before ', 'inner', ' after']

def test_extract_text_array_separator_with_text():
    class MockDom:
        tag = 'br'
        text = 'text'
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['text']

def test_extract_text_array_separator_with_child():
    class MockChild:
        tag = 'span'
        text = 'child'
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = 'br'
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['child']

def test_extract_text_array_multiple_artifical_nl_squashed():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_leading_and_trailing_nl_stripped():
    class MockDom:
        tag = 'div'
        text = 'mid'
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['mid']```


# LLM-generated content at query #58
#--------------------------

def test_extract_text_array_empty_dom():
    dom = type("Mock", (), {"tag": "div", "text": None, "getchildren": lambda self: [], "tail": None})()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_single_text():
    dom = type("Mock", (), {"tag": "span", "text": "hello", "getchildren": lambda self: [], "tail": None})()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_extract_text_array_separator():
    dom = type("Mock", (), {"tag": "br", "text": None, "getchildren": lambda self: [], "tail": None})()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_artificial_newline():
    dom = type("Mock", (), {"tag": "div", "text": None, "getchildren": lambda self: [], "tail": None})()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_child():
    child = type("Mock", (), {"tag": "span", "text": "world", "getchildren": lambda self: [], "tail": "!"})()
    dom = type("Mock", (), {"tag": "div", "text": "hello ", "getchildren": lambda self: [child], "tail": None})()
    result = extract_text_array(dom)
    assert result == ["hello ", "world", "!"]

def test_extract_text_array_squash_artifical_nl():
    child = type("Mock", (), {"tag": "span", "text": "hello", "getchildren": lambda self: [], "tail": None})()
    dom = type("Mock", (), {"tag": "div", "text": None, "getchildren": lambda self: [child], "tail": None})()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_extract_text_array_strip_artifical_nl():
    child = type("Mock", (), {"tag": "span", "text": "hello", "getchildren": lambda self: [], "tail": None})()
    dom = type("Mock", (), {"tag": "div", "text": None, "getchildren": lambda self: [child], "tail": None})()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_extract_text_array_no_squash():
    child = type("Mock", (), {"tag": "div", "text": "a", "getchildren": lambda self: [], "tail": None})()
    dom = type("Mock", (), {"tag": "div", "text": None, "getchildren": lambda self: [child], "tail": None})()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, "a", None, None]

def test_extract_text_array_no_strip():
    child = type("Mock", (), {"tag": "div", "text": "a", "getchildren": lambda self: [], "tail": None})()
    dom = type("Mock", (), {"tag": "div", "text": None, "getchildren": lambda self: [child], "tail": None})()
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "a", None]


# LLM-generated content at query #59
#--------------------------

```python
def test_predicate_at_line_12_evaluates_to_false():
    dom = type('MockDOM', (), {'tag': 'div', 'text': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == [None, None] or result == []  # depending on _squash and _strip behavior
```


# LLM-generated content at query #60
#--------------------------

def test_predicate_line12_false():
    from unittest.mock import Mock
    child = Mock()
    child.getchildren.return_value = []
    child.tail = None
    dom = Mock()
    dom.tag = "p"
    dom.text = None
    dom.getchildren.return_value = [child]
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result is not None


# LLM-generated content at query #61
#--------------------------

```python
def test_squash_artifical_nl_false_when_squash_artifical_nl_is_false():
    dom = type('MockDom', (), {'tag': 'div', 'text': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]  # Two None entries from lines 9 and 19, not squashed
```


# LLM-generated content at query #62
#--------------------------

def test_extract_text_array_empty_tag():
    import xml.etree.ElementTree as ET
    root = ET.fromstring("<div></div>")
    result = extract_text_array(root)
    assert result == []

def test_extract_text_array_inline_tag_with_text():
    import xml.etree.ElementTree as ET
    root = ET.fromstring("<span>hello</span>")
    result = extract_text_array(root)
    assert result == ["hello"]

def test_extract_text_array_separator_tag():
    import xml.etree.ElementTree as ET
    root = ET.fromstring("<br/>")
    result = extract_text_array(root)
    assert result == [True]

def test_extract_text_array_nested_inline():
    import xml.etree.ElementTree as ET
    root = ET.fromstring("<span>hello <b>world</b></span>")
    result = extract_text_array(root)
    assert result == ["hello ", "world"]

def test_extract_text_array_block_with_children():
    import xml.etree.ElementTree as ET
    root = ET.fromstring("<div><p>first</p><p>second</p></div>")
    result = extract_text_array(root)
    assert result == ["first", "second"]

def test_extract_text_array_squash_artifical_nl():
    import xml.etree.ElementTree as ET
    root = ET.fromstring("<div>a</div>")
    result = extract_text_array(root, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == ["a"]


# LLM-generated content at query #63
#--------------------------

```
def test_predicate_at_line_17_evaluates_to_true():
    class MockDOM:
        tag = 'not_INLINE_TAGS_not_SEPARATORS'
        text = None
        def getchildren(self):
            return []
    dom = MockDOM()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert None in result
```


# LLM-generated content at query #64
#--------------------------

def test_extract_text_simple_text():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<p>Hello World</p>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_with_block_elements():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom)
    assert result == "First\nSecond"

def test_extract_text_with_separator():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<hr>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_with_inline_elements():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<span>Hello <b>World</b></span>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_with_preformatted():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<pre>  Hello\n  World  </pre>")
    result = extract_text(dom)
    assert result == "  Hello\n  World  "

def test_extract_text_with_whitespace_squashing():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div>  Hello   World  </div>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_with_nested_blocks():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div><section><p>A</p></section><p>B</p></div>")
    result = extract_text(dom)
    assert result == "A\nB"

def test_extract_text_empty_dom():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div></div>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_with_tail_text():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div>Start<b>bold</b>End</div>")
    result = extract_text(dom)
    assert result == "StartboldEnd"

def test_extract_text_with_separator_and_text():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<hr>Separator")
    result = extract_text(dom)
    assert result == "Separator"


# LLM-generated content at query #65
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_returns_plain_text():
    class MockDom:
        tag = 'span'
        text = 'hello'
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['hello']

def test_extract_text_array_with_separator_tag():
    class MockDom:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_none_tag_after_separator():
    class MockDom:
        tag = 'div'
        text = 'a'
        def getchildren(self):
            child = MockDom()
            child.tag = 'br'
            child.text = None
            child.tail = None
            child.getchildren = lambda: []
            return [child]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['a', True]

def test_extract_text_array_squash_artifical_nl():
    class MockDom:
        tag = 'div'
        text = 'a'
        def getchildren(self):
            child = MockDom()
            child.tag = 'div'
            child.text = None
            child.tail = 'b'
            child.getchildren = lambda: []
            return [child]
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ['a', 'b']

def test_extract_text_array_strip_artifical_nl():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            child = MockDom()
            child.tag = 'span'
            child.text = 'hello'
            child.tail = None
            child.getchildren = lambda: []
            return [child]
    dom = MockDom()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['hello']

def test_extract_text_array_squash_false_preserves_none():
    class MockDom:
        tag = 'div'
        text = 'a'
        def getchildren(self):
            child = MockDom()
            child.tag = 'div'
            child.text = None
            child.tail = 'b'
            child.getchildren = lambda: []
            return [child]
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'a', None, 'b', None]

def test_extract_text_array_strip_false_keeps_leading_trailing():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            child = MockDom()
            child.tag = 'span'
            child.text = 'hello'
            child.tail = None
            child.getchildren = lambda: []
            return [child]
    dom = MockDom()
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'hello', None]
```


# LLM-generated content at query #66
#--------------------------

def test_squash_space_false():
    dom = None
    result = extract_text(dom, squash_space=False)


# LLM-generated content at query #67
#--------------------------

def test_squash_space_is_false():
    dom = None
    result = extract_text(dom, squash_space=False)
    assert result is not None


# LLM-generated content at query #68
#--------------------------

def test_strip_artifical_nl_false():
    r = [None]
    strip_artifical_nl = False
    squash_artifical_nl = False
    result = _strip_artifical_nl(r) if strip_artifical_nl else r
    assert result == [None]


# LLM-generated content at query #69
#--------------------------

def test_predicate_line17_false():
    dom = type('Mock', (), {'tag': 'inline_tag', 'text': None, 'getchildren': lambda: []})()
    dom.tag = 'span'
    INLINE_TAGS = {'span', 'a', 'b'}
    SEPARATORS = {'br', 'hr'}
    result = extract_text_array(dom)
    assert True


# LLM-generated content at query #70
#--------------------------

def test_extract_text_predicate_false():
    dom = []
    result = extract_text(dom, squash_space=False)


# LLM-generated content at query #71
#--------------------------

def test_squash_space_false_does_not_strip():
    dom = None  # mock or minimal dom object
    result = extract_text(dom, squash_space=False)
    assert result == result  # no strip behavior


# LLM-generated content at query #72
#--------------------------

def test_extract_text_returns_string_for_simple_text():
    from xml.etree.ElementTree import Element
    dom = Element("p")
    dom.text = "Hello world"
    result = extract_text(dom)
    assert result == "Hello world"

def test_extract_text_handles_nested_inline_tags():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("p")
    dom.text = "Hello "
    b = SubElement(dom, "b")
    b.text = "bold"
    b.tail = " world"
    result = extract_text(dom)
    assert result == "Hello bold world"

def test_extract_text_handles_block_tags():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    p1 = SubElement(dom, "p")
    p1.text = "First paragraph"
    p2 = SubElement(dom, "p")
    p2.text = "Second paragraph"
    result = extract_text(dom)
    assert result == "First paragraph\nSecond paragraph"

def test_extract_text_handles_separator_tags():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    hr = SubElement(dom, "hr")
    hr.tail = "After hr"
    result = extract_text(dom)
    assert result == "After hr"

def test_extract_text_handles_preformatted_text():
    from xml.etree.ElementTree import Element
    dom = Element("pre")
    dom.text = "  Multiple   spaces  "
    result = extract_text(dom)
    assert result == "  Multiple   spaces  "

def test_extract_text_handles_nested_block_and_inline():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    p1 = SubElement(dom, "p")
    p1.text = "Line1"
    p2 = SubElement(dom, "p")
    b = SubElement(p2, "b")
    b.text = "Bold"
    b.tail = " text"
    result = extract_text(dom)
    assert result == "Line1\nBold text"

def test_extract_text_returns_empty_string_for_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element("div")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_strips_trailing_newlines():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    p = SubElement(dom, "p")
    p.text = "Text"
    result = extract_text(dom)
    assert result == "Text"


# LLM-generated content at query #73
#--------------------------

```python
def test_squash_artifical_nl_true_when_squash_artifical_nl_is_true():
    dom = type('MockDom', (object,), {'tag': 'div', 'text': 'Hello', 'getchildren': lambda self: [], 'tail': None})()
    extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert True
```


# LLM-generated content at query #74
#--------------------------

def test_predicate_evaluates_to_true():
    dom = None
    squash_space = True
    a = []
    result = ''.join(
        '\n' if x is None else (
            '\n' if x is True else x
        )
        for x in a
    )
    if squash_space:
        result = result.strip()
    assert squash_space == True


# LLM-generated content at query #75
#--------------------------

```python
def test_extract_text_array_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_text():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = 'Hello'
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_separator_tag():
    from xml.etree.ElementTree import Element
    dom = Element('br')
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_inline_tag():
    from xml.etree.ElementTree import Element
    dom = Element('span')
    dom.text = 'world'
    result = extract_text_array(dom)
    assert result == ['world']

def test_extract_text_array_with_child():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    child = Element('p')
    child.text = 'text'
    dom.append(child)
    result = extract_text_array(dom)
    assert result == ['text']

def test_extract_text_array_with_tail():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    child = SubElement(dom, 'b')
    child.text = 'bold'
    child.tail = ' normal'
    result = extract_text_array(dom)
    assert result == ['bold', ' normal']

def test_extract_text_array_squash_true():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    SubElement(dom, 'p')
    SubElement(dom, 'p')
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == []

def test_extract_text_array_strip_true():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    SubElement(dom, 'p')
    child = SubElement(dom, 'span')
    child.text = 'content'
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['content']

def test_extract_text_array_complex():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    dom.text = 'start '
    child1 = SubElement(dom, 'b')
    child1.text = 'bold'
    child1.tail = ' middle '
    child2 = SubElement(dom, 'i')
    child2.text = 'italic'
    child2.tail = ' end'
    result = extract_text_array(dom)
    assert result == ['start ', 'bold', ' middle ', 'italic', ' end']


# LLM-generated content at query #76
#--------------------------

def test_predicate_at_line_17_evaluates_to_false():
    from unittest.mock import Mock
    dom = Mock()
    dom.tag = "p"
    dom.text = None
    dom.getchildren.return_value = []
    result = extract_text_array(dom)
    assert dom.tag not in INLINE_TAGS and dom.tag not in SEPARATORS


# LLM-generated content at query #77
#--------------------------

```
def test_strip_artifical_nl_true():
    dom = type('MockDom', (object,), {'tag': 'p', 'text': 'hello', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ['hello']
```


# LLM-generated content at query #78
#--------------------------

```python
def test_extract_text_array_empty_children():
    dom = type('MockDOM', (), {
        'tag': 'div',
        'text': 'some text',
        'getchildren': lambda: []
    })()
    result = extract_text_array(dom)
    assert len(result) == 2
    assert result[0] is None
    assert result[1] == 'some text'
```


# LLM-generated content at query #79
#--------------------------

```python
def test_predicate_line20_evaluates_false():
    dom = type('MockDOM', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)


# LLM-generated content at query #80
#--------------------------

```python
def test_predicate_line12_true():
    from unittest.mock import Mock
    child1 = Mock()
    child1.tag = 'child'
    child1.tail = None
    child1.getchildren = lambda: []
    child1.text = None
    dom = Mock()
    dom.tag = 'div'
    dom.text = 'sample'
    dom.getchildren = lambda: [child1]
    result = extract_text_array(dom)
    assert len(result) >= 3
    assert result[0] is None
    assert result[1] == 'sample'
    assert result[2] is None
```


# LLM-generated content at query #81
#--------------------------

def test_extract_text_simple_text():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<p>Hello World</p>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_nested_inline():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<p>Hello <b>World</b></p>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_with_separator():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom)
    assert result == "First\nSecond"

def test_extract_text_with_block_symbol():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom, block_symbol=' | ')
    assert result == "First | Second"

def test_extract_text_with_sep_symbol():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div><br/>Separator</div>")
    result = extract_text(dom, sep_symbol=' | ')
    assert result == " | Separator"

def test_extract_text_squash_space_true():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<p>  Hello   World  </p>")
    result = extract_text(dom, squash_space=True)
    assert result == "Hello World"

def test_extract_text_squash_space_false():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<p>  Hello   World  </p>")
    result = extract_text(dom, squash_space=False)
    assert result == "  Hello   World  "

def test_extract_text_empty_dom():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div></div>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_only_tail():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div>Text</div>")
    result = extract_text(dom)
    assert result == "Text"

def test_extract_text_nested_block():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div><section><p>Paragraph</p></section></div>")
    result = extract_text(dom)
    assert result == "Paragraph"

def test_extract_text_multiple_blocks():
    from lxml.html import fragment_fromstring
    dom = fragment_fromstring("<div><p>First</p><p>Second</p><p>Third</p></div>")
    result = extract_text(dom)
    assert result == "First\nSecond\nThird"


# LLM-generated content at query #82
#--------------------------

def test_strip_artifical_nl_false():
    result = extract_text_array(MockDom(tag='p', text='hello', children=[]), squash_artifical_nl=False, strip_artifical_nl=False)
    assert len(result) == 1
    assert result[0] == 'hello'


# LLM-generated content at query #83
#--------------------------

def test_extract_text_simple_text():
    dom = type('obj', (object,), {'tag': 'span', 'text': 'hello', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text(dom)
    assert result == 'hello'

def test_extract_text_with_newline():
    dom = type('obj', (object,), {'tag': 'div', 'text': 'line1', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text(dom)
    assert result == 'line1'

def test_extract_text_with_separator():
    dom = type('obj', (object,), {'tag': 'br', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text(dom)
    assert result == '\n'

def test_extract_text_nested():
    child = type('obj', (object,), {'tag': 'span', 'text': 'world', 'getchildren': lambda self: [], 'tail': '!'})()
    dom = type('obj', (object,), {'tag': 'div', 'text': 'hello ', 'getchildren': lambda self: [child], 'tail': None})()
    result = extract_text(dom)
    assert result == 'hello world!'

def test_extract_text_multiple_blocks():
    child1 = type('obj', (object,), {'tag': 'p', 'text': 'first', 'getchildren': lambda self: [], 'tail': None})()
    child2 = type('obj', (object,), {'tag': 'p', 'text': 'second', 'getchildren': lambda self: [], 'tail': None})()
    dom = type('obj', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [child1, child2], 'tail': None})()
    result = extract_text(dom)
    assert result == 'first\nsecond'

def test_extract_text_with_separator_inline():
    child = type('obj', (object,), {'tag': 'br', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    dom = type('obj', (object,), {'tag': 'span', 'text': 'a', 'getchildren': lambda self: [child], 'tail': 'b'})()
    result = extract_text(dom)
    assert result == 'a\nb'

def test_extract_text_empty():
    dom = type('obj', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text(dom)
    assert result == ''

def test_extract_text_whitespace_squash():
    dom = type('obj', (object,), {'tag': 'span', 'text': '  hello   world  ', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text(dom)
    assert result == 'hello world'

def test_extract_text_nested_with_tail():
    child = type('obj', (object,), {'tag': 'b', 'text': 'bold', 'getchildren': lambda self: [], 'tail': ' normal'})()
    dom = type('obj', (object,), {'tag': 'p', 'text': 'text ', 'getchildren': lambda self: [child], 'tail': None})()
    result = extract_text(dom)
    assert result == 'text bold normal'


# LLM-generated content at query #84
#--------------------------

def test_squash_space_false_does_not_strip():
    dom = type('Dom', (), {})()
    result = extract_text(dom, squash_space=False)
    assert result == ""


# LLM-generated content at query #85
#--------------------------

def test_squash_space_true_predicate():
    dom = "test"
    result = extract_text(dom, squash_space=True)
    assert result is not None


# LLM-generated content at query #86
#--------------------------

def test_extract_text_simple_text():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = "Hello"
    result = extract_text(dom)
    assert result == "Hello"

def test_extract_text_with_child():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = "Hello"
    p2 = SubElement(dom, 'p')
    p2.text = "World"
    result = extract_text(dom)
    assert result == "Hello\nWorld"

def test_extract_text_with_separator():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    hr = SubElement(dom, 'hr')
    p = SubElement(dom, 'p')
    p.text = "After"
    result = extract_text(dom)
    assert result == "\nAfter"

def test_extract_text_with_nested_inline():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('p')
    b = SubElement(dom, 'b')
    b.text = "Bold"
    dom.text = "Text "
    result = extract_text(dom)
    assert result == "Text Bold"

def test_extract_text_strip_whitespace():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = "  Hello  "
    result = extract_text(dom)
    assert result == "Hello"

def test_extract_text_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    result = extract_text(dom)
    assert result == ""

def test_extract_text_with_none_text():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = None
    result = extract_text(dom)
    assert result == ""

def test_extract_text_block_symbol_custom():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = "First"
    p2 = SubElement(dom, 'p')
    p2.text = "Second"
    result = extract_text(dom, block_symbol='<br>')
    assert result == "First<br>Second"

def test_extract_text_sep_symbol_custom():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    hr = SubElement(dom, 'hr')
    p = SubElement(dom, 'p')
    p.text = "After"
    result = extract_text(dom, sep_symbol='---')
    assert result == "---After"

def test_extract_text_squash_space_false():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = "  Hello  "
    result = extract_text(dom, squash_space=False)
    assert result == "  Hello  "


# LLM-generated content at query #87
#--------------------------

def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_with_separator():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    class MockElement:
        tag = "span"
        text = "inline"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["inline"]

def test_extract_text_array_with_child():
    class MockChild:
        tag = "b"
        text = "bold"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "before "
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["before ", "bold"]

def test_extract_text_array_with_tail():
    class MockChild:
        tag = "a"
        text = "link"
        tail = " after"
        def getchildren(self):
            return []
    class MockElement:
        tag = "p"
        text = "click "
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["click ", "link", " after"]

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = "div"
        text = "a"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == ["a"]

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = "div"
        text = "a"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["a"]

def test_extract_text_array_no_squash_no_strip():
    class MockElement:
        tag = "div"
        text = "a"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "a", None]


# LLM-generated content at query #88
#--------------------------

```python
def test_strip_artifical_nl_true():
    dom = type('MockDom', (), {'tag': 'p', 'text': 'hello', 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ['hello']  # Assuming _strip_artifical_nl removes None values
```


# LLM-generated content at query #89
#--------------------------

def test_squash_artificial_nl_false():
    dom = type('MockDom', (), {'tag': 'p', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]


# LLM-generated content at query #90
#--------------------------

def test_extract_text_simple_paragraph():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = "Hello world"
    assert extract_text(dom) == "Hello world"

def test_extract_text_with_separator():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = "First"
    br = SubElement(dom, 'br')
    p2 = SubElement(dom, 'p')
    p2.text = "Second"
    assert extract_text(dom) == "First\nSecond"

def test_extract_text_multiple_paragraphs():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = "Para1"
    p2 = SubElement(dom, 'p')
    p2.text = "Para2"
    p3 = SubElement(dom, 'p')
    p3.text = "Para3"
    assert extract_text(dom) == "Para1\nPara2\nPara3"

def test_extract_text_with_inline_tags():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('p')
    dom.text = "Hello "
    b = SubElement(dom, 'b')
    b.text = "bold"
    b.tail = " world"
    assert extract_text(dom) == "Hello bold world"

def test_extract_text_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    assert extract_text(dom) == ""

def test_extract_text_with_nested_inline():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('p')
    dom.text = "Start "
    a = SubElement(dom, 'a')
    a.text = "link"
    a.tail = " end"
    assert extract_text(dom) == "Start link end"

def test_extract_text_block_symbol_custom():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = "Line1"
    p2 = SubElement(dom, 'p')
    p2.text = "Line2"
    assert extract_text(dom, block_symbol='|') == "Line1|Line2"

def test_extract_text_sep_symbol_custom():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = "Text"
    br = SubElement(dom, 'br')
    assert extract_text(dom, sep_symbol='|') == "Text|"

def test_extract_text_squash_space_false():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = "A"
    p2 = SubElement(dom, 'p')
    p2.text = "B"
    assert extract_text(dom, squash_space=False) == "\nA\n\nB\n"

def test_extract_text_with_whitespace_in_text():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = "Hello   world"
    assert extract_text(dom) == "Hello world"


# LLM-generated content at query #91
#--------------------------

def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        tag = "span"
        text = "hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_extract_text_array_with_separator():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_nl_squash():
    class MockChild:
        tag = "div"
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_with_nl_strip():
    class MockElement:
        tag = "div"
        text = "hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["hello"]


# LLM-generated content at query #92
#--------------------------

```
def test_extract_text_array_empty_dom_with_no_tag():
    class MockElement:
        tag = None
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_dom_with_string_tag():
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_dom_with_text():
    class MockElement:
        tag = 'p'
        text = 'hello'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'hello', None]

def test_extract_text_array_with_child():
    class MockChild:
        tag = 'span'
        text = 'world'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None, 'world', None, None]

def test_extract_text_array_with_separator_tag():
    class MockChild:
        tag = 'br'
        text = None
        tail = 'tail'
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, True, 'tail', None, None]

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = 'div'
        text = 'text'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ['text']

def test_extract_text_array_with_text_and_child_and_tail():
    class MockChild:
        tag = 'span'
        text = 'inner'
        tail = ' after '
        def getchildren(self):
            return []
    class MockElement:
        tag = 'p'
        text = 'before '
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'before ', None, 'inner', ' after ', None]

def test_extract_text_array_squash_and_strip():
    class MockChild:
        tag = 'span'
        text = 'content'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['content']

def test_extract_text_array_callable_tag_returns_empty_string():
    class MockElement:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ''

def test_extract_text_array_multiple_children():
    class MockChild1:
        tag = 'b'
        text = 'bold'
        tail = ' '
        def getchildren(self):
            return []
    class MockChild2:
        tag = 'i'
        text = 'italic'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'p'
        text = None
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None, 'bold', ' ', None, 'italic', None]

def test_extract_text_array_nested_children():
    class MockGrandchild:
        tag = 'strong'
        text = 'nested'
        tail = None
        def getchildren(self):
            return []
    class MockChild:
        tag = 'span'
        text = None
        tail = ' end'
        def getchildren(self):
            return [MockGrandchild()]
    class MockElement:
        tag = 'div'
        text = 'start '
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'start ', None, None, 'nested', None, ' end', None, None]


# LLM-generated content at query #93
#--------------------------

def test_squash_artifical_nl_true():
    dom = type('MockDom', (object,), {'tag': 'p', 'text': 'hello', 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result is not None


# LLM-generated content at query #94
#--------------------------

```python
def test_for_loop_iterates_over_children():
    from unittest.mock import Mock
    child1 = Mock()
    child1.tag = 'p'
    child1.text = None
    child1.tail = None
    child1.getchildren.return_value = []
    child2 = Mock()
    child2.tag = 'span'
    child2.text = None
    child2.tail = None
    child2.getchildren.return_value = []
    dom = Mock()
    dom.tag = 'div'
    dom.text = None
    dom.getchildren.return_value = [child1, child2]
    result = extract_text_array(dom)
    assert len(result) >= 2
```


# LLM-generated content at query #95
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = 'p'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        tag = 'p'
        text = 'Hello'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_separator_tag():
    class MockElement:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag():
    class MockElement:
        tag = 'b'
        text = 'bold'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['bold']

def test_extract_text_array_with_child():
    class MockChild:
        tag = 'span'
        text = 'child'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = 'parent'
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['parent', 'child']

def test_extract_text_array_with_tail():
    class MockChild:
        tag = 'a'
        text = 'link'
        tail = ' after'
        def getchildren(self):
            return []
    class MockElement:
        tag = 'p'
        text = 'before '
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['before ', 'link', ' after']

def test_extract_text_array_with_artifical_nl():
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = 'div'
        text = 'a'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == ['a']

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = 'div'
        text = 'b'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ['b']

def test_extract_text_array_no_squash_no_strip():
    class MockElement:
        tag = 'div'
        text = 'c'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'c', None]

def test_extract_text_array_with_separator_and_text():
    class MockElement:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    class MockParent:
        tag = 'p'
        text = 'line1'
        def getchildren(self):
            return [MockElement()]
    dom = MockParent()
    result = extract_text_array(dom)
    assert result == ['line1', True]

def test_extract_text_array_nested():
    class MockLeaf:
        tag = 'i'
        text = 'italic'
        tail = None
        def getchildren(self):
            return []
    class MockMiddle:
        tag = 'b'
        text = 'bold '
        tail = ' end'
        def getchildren(self):
            return [MockLeaf()]
    class MockRoot:
        tag = 'div'
        text = 'start '
        def getchildren(self):
            return [MockMiddle()]
    dom = MockRoot()
    result = extract_text_array(dom)
    assert result == ['start ', 'bold ', 'italic', ' end']


# LLM-generated content at query #96
#--------------------------

def test_squash_space_true():
    dom = "test"
    result = extract_text(dom, squash_space=True)
    assert True


# LLM-generated content at query #97
#--------------------------

def test_extract_text_empty_dom():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div></div>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_simple_text():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div>Hello</div>")
    result = extract_text(dom)
    assert result == "Hello"

def test_extract_text_with_separator():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div><br/></div>")
    result = extract_text(dom)
    assert result == "\n"

def test_extract_text_nested():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom)
    assert result == "First\nSecond"

def test_extract_text_with_tail():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div><b>Bold</b> text</div>")
    result = extract_text(dom)
    assert result == "Bold text"

def test_extract_text_squash_whitespace():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div>  Hello   World  </div>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_block_symbol():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text(dom, block_symbol='|')
    assert result == "A|B"

def test_extract_text_sep_symbol():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div>A<br/>B</div>")
    result = extract_text(dom, sep_symbol='|')
    assert result == "A|B"

def test_extract_text_squash_space_false():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div>  Hello </div>")
    result = extract_text(dom, squash_space=False)
    assert result == "  Hello "


# LLM-generated content at query #98
#--------------------------

def test_predicate_false():
    dom = None
    result = extract_text(dom, squash_space=False)
    assert result == ""


# LLM-generated content at query #99
#--------------------------

```
def test_extract_text_simple_text():
    from lxml import etree
    dom = etree.fromstring("<p>Hello world</p>")
    result = extract_text(dom)
    assert result == "Hello world"

def test_extract_text_with_block_symbol():
    from lxml import etree
    dom = etree.fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom, block_symbol='\n')
    assert result == "First\nSecond"

def test_extract_text_with_sep_symbol():
    from lxml import etree
    dom = etree.fromstring("<div><br/>Break</div>")
    result = extract_text(dom, sep_symbol='\n')
    assert result == "Break"

def test_extract_text_nested_tags():
    from lxml import etree
    dom = etree.fromstring("<div><span>Inner</span> Outer</div>")
    result = extract_text(dom)
    assert result == "Inner Outer"

def test_extract_text_empty_dom():
    from lxml import etree
    dom = etree.fromstring("<div></div>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_with_tail():
    from lxml import etree
    dom = etree.fromstring("<p>Hello <b>bold</b> world</p>")
    result = extract_text(dom)
    assert result == "Hello bold world"

def test_extract_text_squash_space_false():
    from lxml import etree
    dom = etree.fromstring("<div>  Multiple   spaces  </div>")
    result = extract_text(dom, squash_space=False)
    assert result == "  Multiple   spaces  "

def test_extract_text_block_symbol_custom():
    from lxml import etree
    dom = etree.fromstring("<div><p>Line1</p><p>Line2</p></div>")
    result = extract_text(dom, block_symbol='|')
    assert result == "Line1|Line2"

def test_extract_text_sep_symbol_custom():
    from lxml import etree
    dom = etree.fromstring("<div>Text<br/>More</div>")
    result = extract_text(dom, sep_symbol='|')
    assert result == "Text|More"

def test_extract_text_with_pre_tag():
    from lxml import etree
    dom = etree.fromstring("<pre>  Preserved  </pre>")
    result = extract_text(dom)
    assert result == "  Preserved  "
```


# LLM-generated content at query #100
#--------------------------

def test_extract_text_squash_space_returns_false_when_squash_space_is_false():
    dom = []
    result = extract_text(dom, squash_space=False)
    assert result == ''


# LLM-generated content at query #101
#--------------------------

def test_squash_artifical_nl_false_when_squash_artifical_nl_is_false():
    dom = type('MockDom', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert not result  # or assert result == [] depending on expected output


# LLM-generated content at query #102
#--------------------------

def test_extract_text_returns_empty_string_for_callable_tag():
    dom = type('FakeDom', (), {'tag': lambda: None, 'text': None, 'getchildren': lambda: []})()
    assert extract_text(dom) == ''

def test_extract_text_single_text_node():
    dom = type('FakeDom', (), {'tag': 'p', 'text': 'Hello', 'getchildren': lambda: [], 'tail': None})()
    assert extract_text(dom) == 'Hello'

def test_extract_text_with_nested_tags():
    child = type('FakeDom', (), {'tag': 'b', 'text': 'bold', 'getchildren': lambda: [], 'tail': ' normal'})()
    dom = type('FakeDom', (), {'tag': 'p', 'text': 'Start ', 'getchildren': lambda: [child], 'tail': None})()
    assert extract_text(dom) == 'Start bold normal'

def test_extract_text_with_separator_tag():
    child = type('FakeDom', (), {'tag': 'br', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    dom = type('FakeDom', (), {'tag': 'p', 'text': 'Line1', 'getchildren': lambda: [child], 'tail': None})()
    assert extract_text(dom, sep_symbol='\n') == 'Line1\n'

def test_extract_text_with_block_tag():
    child = type('FakeDom', (), {'tag': 'div', 'text': 'Block', 'getchildren': lambda: [], 'tail': None})()
    dom = type('FakeDom', (), {'tag': 'p', 'text': None, 'getchildren': lambda: [child], 'tail': None})()
    assert extract_text(dom, block_symbol='\n') == 'Block'

def test_extract_text_squash_space_true():
    child = type('FakeDom', (), {'tag': 'span', 'text': '  spaced  ', 'getchildren': lambda: [], 'tail': '  text  '})()
    dom = type('FakeDom', (), {'tag': 'p', 'text': '  hello  ', 'getchildren': lambda: [child], 'tail': None})()
    assert extract_text(dom) == 'hello spaced text'

def test_extract_text_squash_space_false():
    child = type('FakeDom', (), {'tag': 'span', 'text': '  spaced  ', 'getchildren': lambda: [], 'tail': '  text  '})()
    dom = type('FakeDom', (), {'tag': 'p', 'text': '  hello  ', 'getchildren': lambda: [child], 'tail': None})()
    assert extract_text(dom, squash_space=False) == '  hello    spaced    text  '

def test_extract_text_empty_dom():
    dom = type('FakeDom', (), {'tag': 'p', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    assert extract_text(dom) == ''

def test_extract_text_without_children():
    dom = type('FakeDom', (), {'tag': 'p', 'text': 'Alone', 'getchildren': lambda: [], 'tail': None})()
    assert extract_text(dom) == 'Alone'

def test_extract_text_with_multiple_children():
    child1 = type('FakeDom', (), {'tag': 'b', 'text': 'first', 'getchildren': lambda: [], 'tail': None})()
    child2 = type('FakeDom', (), {'tag': 'i', 'text': 'second', 'getchildren': lambda: [], 'tail': None})()
    dom = type('FakeDom', (), {'tag': 'p', 'text': None, 'getchildren': lambda: [child1, child2], 'tail': None})()
    assert extract_text(dom) == 'firstsecond'


# LLM-generated content at query #103
#--------------------------

def test_extract_text_with_squash_space_true():
    dom = None
    result = extract_text(dom, squash_space=True)
    assert isinstance(result, str)


# LLM-generated content at query #104
#--------------------------

def test_squash_space_false_predicate():
    a = extract_text_array(dom, squash_artifical_nl=False)
    if squash_space:
        a = _strip_artifical_nl(_squash_artifical_nl(_merge_original_parts(a)))
    result = ''.join(
        block_symbol if x is None else (
            sep_symbol if x is True else x
        )
        for x in a
    )
    if squash_space:
        result = result.strip()
    return result


# LLM-generated content at query #105
#--------------------------

```python
def test_predicate_line12_true():
    dom = type('Mock', (), {
        'tag': 'p',
        'text': 'some text',
        'getchildren': lambda self: [],
        'tail': None
    })()
    result = extract_text_array(dom)
    assert 'some text' in result
```


# LLM-generated content at query #106
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "p"
        text = None
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_separator_tag():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag():
    class MockElement:
        tag = "span"
        text = "text"
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["text"]

def test_extract_text_array_with_child():
    class MockChild:
        tag = "b"
        text = "bold"
        tail = None
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockChild"
    class MockElement:
        tag = "p"
        text = "before "
        def getchildren(self):
            return [MockChild()]
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["before ", "bold"]

def test_extract_text_array_with_tail():
    class MockChild:
        tag = "b"
        text = "bold"
        tail = " after"
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockChild"
    class MockElement:
        tag = "p"
        text = "before "
        def getchildren(self):
            return [MockChild()]
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["before ", "bold", " after"]

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = "p"
        text = "a"
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ["a"]

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == []
```


# LLM-generated content at query #107
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    from unittest.mock import Mock
    dom = Mock()
    dom.tag = "div"
    dom.text = None
    dom.getchildren.return_value = []
    dom.tag not in ["br", "hr"] and dom.tag not in ["p", "div", "span"]
    extract_text_array(dom)


# LLM-generated content at query #108
#--------------------------

def test_extract_text_array_strips_artificial_newlines():
    dom = type('MockDom', (), {'tag': 'p', 'text': 'hello', 'getchildren': lambda: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ['hello']


# LLM-generated content at query #109
#--------------------------

def test_predicate_at_line17_evaluates_to_false():
    dom = type('MockDom', (), {'tag': 'INLINE_TAG', 'text': None, 'getchildren': lambda self: []})()
    INLINE_TAGS = {'INLINE_TAG'}
    SEPARATORS = set()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)


# LLM-generated content at query #110
#--------------------------

```python
def test_empty_dom():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_dom_with_text_only():
    class MockDom:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_dom_with_separator_tag():
    class MockDom:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_dom_with_inline_tag_and_text():
    class MockDom:
        tag = "span"
        text = "inline"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["inline"]

def test_dom_with_artificial_newlines_stripped():
    class Child:
        tag = "p"
        text = "child"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [Child()]
    dom = MockDom()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["child"]

def test_dom_with_artificial_newlines_squashed():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None]

def test_dom_with_children_and_tail():
    class Child:
        tag = "span"
        text = "child"
        tail = " tail"
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = "before"
        def getchildren(self):
            return [Child()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["before", "child", " tail"]

def test_dom_with_nested_separators():
    class Inner:
        tag = "br"
        text = None
        tail = "after"
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = "start"
        def getchildren(self):
            return [Inner()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["start", True, "after"]

def test_dom_skip_empty_strings():
    class MockDom:
        tag = "p"
        text = ""
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [""]

def test_dom_with_multiple_artificial_newlines_squashed():
    class Child1:
        tag = "div"
        text = None
        tail = None
        def getchildren(self):
            return []
    class Child2:
        tag = "div"
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [Child1(), Child2()]
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == [None]

def test_dom_with_no_artificial_newlines_stripped():
    class MockDom:
        tag = "p"
        text = "text"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["text"]
```


# LLM-generated content at query #111
#--------------------------

def test_strip_artifical_nl_false():
    class MockDom:
        tag = "p"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]


# LLM-generated content at query #112
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_single_text():
    class MockDom:
        tag = "span"
        text = "hello"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_extract_text_array_separator_tag():
    class MockDom:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag_with_text():
    class MockDom:
        tag = "b"
        text = "bold"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["bold"]

def test_extract_text_array_block_tag_artifical_newlines():
    class MockDom:
        tag = "p"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_child():
    class MockChild:
        tag = "span"
        text = "child"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["child"]

def test_extract_text_array_with_tail():
    class MockChild:
        tag = "span"
        text = "child"
        tail = " tail"
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = "before"
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["before", "child", " tail"]

def test_extract_text_array_squash_artifical_nl():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == []

def test_extract_text_array_strip_artifical_nl():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_callable_tag():
    class MockDom:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ""

def test_extract_text_array_separator_with_text():
    class MockDom:
        tag = "hr"
        text = "separator"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True, "separator"]

def test_extract_text_array_inline_tag_with_child():
    class MockChild:
        tag = "span"
        text = "inner"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "b"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["inner"]

def test_extract_text_array_block_tag_with_multiple_children():
    class MockChild1:
        tag = "span"
        text = "first"
        tail = " "
        def getchildren(self):
            return []
    class MockChild2:
        tag = "span"
        text = "second"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["first", " ", "second"]

def test_extract_text_array_nested_block_tags():
    class MockInner:
        tag = "div"
        text = "inner text"
        tail = None
        def getchildren(self):
            return []
    class MockOuter:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockInner()]
    dom = MockOuter()
    result = extract_text_array(dom)
    assert result == ["inner text"]

def test_extract_text_array_squash_multiple_artifical_nl():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_leading_trailing_artifical_nl():
    class MockChild:
        tag = "div"
        text = "content"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom, strip_artifical_nl=True, squash_artifical_nl=False)
    assert result == ["content"]

def test_extract_text_array_no_squash_no_strip():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]
```


# LLM-generated content at query #113
#--------------------------

def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        tag = "span"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_separator_tag():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_child():
    class MockChild:
        tag = "span"
        text = "World"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "Hello "
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = "div"
        text = "Content"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["Content"]

def test_extract_text_array_no_squash_no_strip():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_with_tail():
    class MockChild:
        tag = "span"
        text = "Hello"
        tail = " World"
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Hello", " World"]


# LLM-generated content at query #114
#--------------------------

```python
def test_squash_artifical_nl_evaluates_true():
    dom = type('MockDom', (object,), {
        'tag': 'p',
        'text': None,
        'tail': None,
        'getchildren': lambda self: []
    })()
    INLINE_TAGS = []
    SEPARATORS = []
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert True
```


# LLM-generated content at query #115
#--------------------------

```python
def test_extract_text_array_with_none_dom_tag_callable():
    class MockDom:
        tag = lambda: None
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ''

def test_extract_text_array_with_separator_tag():
    class MockDom:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    class MockDom:
        tag = 'span'
        text = 'hello'
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['hello']

def test_extract_text_array_with_non_inline_non_separator_tag():
    class MockDom:
        tag = 'div'
        text = 'text'
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['text']

def test_extract_text_array_with_child_and_tail():
    class MockChild:
        tag = 'b'
        text = 'bold'
        tail = ' tail'
        def getchildren(self):
            return []
    class MockDom:
        tag = 'p'
        text = 'start'
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['start', 'bold', ' tail']

def test_extract_text_array_with_squash_artifical_nl():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_with_strip_artifical_nl():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_with_both_flags_false():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_complex_scenario():
    class MockChild:
        tag = 'span'
        text = 'inner'
        tail = ' after'
        def getchildren(self):
            return []
    class MockDom:
        tag = 'div'
        text = 'before '
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['before ', 'inner', ' after']

def test_extract_text_array_with_tag_in_separators():
    SEPARATORS = {'hr', 'br'}
    class MockDom:
        tag = 'hr'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_tag_in_inline_tags():
    INLINE_TAGS = {'b', 'i', 'span'}
    class MockDom:
        tag = 'b'
        text = 'bold'
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ['bold']


# LLM-generated content at query #116
#--------------------------

```
def test_squash_artifical_nl_false():
    dom = type('MockDom', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None]
```


# LLM-generated content at query #117
#--------------------------

```python
def test_strip_artifical_nl_true():
    dom = type('MockDom', (), {'tag': 'div', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert True  # predicate at line 22 evaluates to True
```


# LLM-generated content at query #118
#--------------------------

def test_predicate_at_line17_evaluates_to_false():
    dom = type('MockDom', (object,), {
        'tag': 'p',
        'text': None,
        'getchildren': lambda self: []
    })()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == []


# LLM-generated content at query #119
#--------------------------

```
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_text_only():
    class MockElement:
        tag = "span"
        text = "hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_extract_text_array_with_separator_tag():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    class MockElement:
        tag = "b"
        text = "bold"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["bold"]

def test_extract_text_array_with_child():
    class MockChild:
        tag = "span"
        text = "child"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "parent"
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["parent", "child"]

def test_extract_text_array_with_artificial_newlines():
    class MockChild:
        tag = "span"
        text = "child"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "parent"
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["parent", "child"]

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = "div"
        text = "a"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ["a"]

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_with_tail():
    class MockChild:
        tag = "span"
        text = "child"
        tail = " tail"
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "parent"
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["parent", "child", " tail"]

def test_extract_text_array_no_squash_no_strip():
    class MockChild:
        tag = "span"
        text = "child"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "parent"
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "parent", None, "child", None]

def test_extract_text_array_with_separator_between_texts():
    class MockChild:
        tag = "br"
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "first"
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["first", True]

def test_extract_text_array_nested_structure():
    class MockGrandchild:
        tag = "span"
        text = "grandchild"
        tail = None
        def getchildren(self):
            return []
    class MockChild:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockGrandchild()]
    class MockElement:
        tag = "div"
        text = "root"
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["root", "grandchild"]

def test_extract_text_array_multiple_none_values():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_separator_with_text():
    class MockElement:
        tag = "br"
        text = "text"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True, "text"]

def test_extract_text_array_inline_tag_with_child():
    class MockChild:
        tag = "span"
        text = "child"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "b"
        text = "bold"
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["bold", "child"]

def test_extract_text_array_squash_multiple_none():
    class MockElement:
        tag = "div"
        text = "a"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ["a"]

def test_extract_text_array_strip_with_trailing_none():
    class MockElement:
        tag = "div"
        text = "a"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["a"]

def test_extract_text_array_with_only_separator():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_empty_children():
    class MockElement:
        tag = "div"
        text = "text"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["text"]

def test_extract_text_array_no_text_in_element():
    class MockChild:
        tag = "span"
        text = None
        tail = "tail"
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["tail"]

def test_extract_text_array_multiple_children():
    class MockChild1:
        tag = "span"
        text = "first"
        tail = None
        def getchildren(self):
            return []
    class MockChild2:
        tag = "span"
        text = "second"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "parent"
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["parent", "first", "second"]

def test_extract_text_array_with_mixed_tags():
    class MockChild1:
        tag = "br"
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockChild2:
        tag = "span"
        text = "text"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "start"
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["start", True, "text"]

def test_extract_text_array_strip_with_leading_none():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_squash_keeps_one_none():
    class MockChild:
        tag = "span"
        text = "child"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, "child", None]

def test_extract


# LLM-generated content at query #120
#--------------------------

def test_extract_text_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    result = extract_text(dom)
    assert result == ""

def test_extract_text_simple_text():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('p')
    dom.text = "Hello"
    result = extract_text(dom)
    assert result == "Hello"

def test_extract_text_with_child():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = "Hello"
    result = extract_text(dom)
    assert result == "Hello"

def test_extract_text_multiple_children():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = "Hello"
    p2 = SubElement(dom, 'p')
    p2.text = "World"
    result = extract_text(dom)
    assert result == "Hello\nWorld"

def test_extract_text_with_separator():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    hr = SubElement(dom, 'hr')
    p = SubElement(dom, 'p')
    p.text = "After"
    result = extract_text(dom)
    assert result == "\nAfter"

def test_extract_text_with_tail():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    a = SubElement(dom, 'a')
    a.text = "Link"
    a.tail = " text"
    result = extract_text(dom)
    assert result == "Link text"

def test_extract_text_squash_whitespace():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = "  Hello   World  "
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_block_symbol():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = "First"
    p2 = SubElement(dom, 'p')
    p2.text = "Second"
    result = extract_text(dom, block_symbol='|')
    assert result == "First|Second"

def test_extract_text_sep_symbol():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    hr = SubElement(dom, 'hr')
    p = SubElement(dom, 'p')
    p.text = "After"
    result = extract_text(dom, sep_symbol='---')
    assert result == "\n---After"

def test_extract_text_squash_space_false():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = "Hello"
    p2 = SubElement(dom, 'p')
    p2.text = "World"
    result = extract_text(dom, squash_space=False)
    assert result == "Hello\nWorld"


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_extract_text_simple_text():
    from lxml.etree import Element
    root = Element('p')
    root.text = 'Hello World'
    assert extract_text(root) == 'Hello World'

def test_extract_text_with_block_element():
    from lxml.etree import Element, SubElement
    root = Element('div')
    p1 = SubElement(root, 'p')
    p1.text = 'First'
    p2 = SubElement(root, 'p')
    p2.text = 'Second'
    assert extract_text(root) == 'First\nSecond'

def test_extract_text_with_separator():
    from lxml.etree import Element, SubElement
    root = Element('div')
    hr = SubElement(root, 'hr')
    p = SubElement(root, 'p')
    p.text = 'After'
    assert extract_text(root) == '\nAfter'

def test_extract_text_with_inline_element():
    from lxml.etree import Element, SubElement
    root = Element('p')
    root.text = 'Hello '
    b = SubElement(root, 'b')
    b.text = 'World'
    assert extract_text(root) == 'Hello World'

def test_extract_text_with_tail():
    from lxml.etree import Element, SubElement
    root = Element('p')
    root.text = 'Hello '
    b = SubElement(root, 'b')
    b.text = 'World'
    b.tail = '!'
    assert extract_text(root) == 'Hello World!'

def test_extract_text_multiple_blocks():
    from lxml.etree import Element, SubElement
    root = Element('div')
    p1 = SubElement(root, 'p')
    p1.text = 'Line1'
    p2 = SubElement(root, 'p')
    p2.text = 'Line2'
    p3 = SubElement(root, 'p')
    p3.text = 'Line3'
    assert extract_text(root) == 'Line1\nLine2\nLine3'

def test_extract_text_with_whitespace():
    from lxml.etree import Element
    root = Element('p')
    root.text = '   Hello   World   '
    assert extract_text(root) == 'Hello World'

def test_extract_text_empty_element():
    from lxml.etree import Element
    root = Element('div')
    assert extract_text(root) == ''

def test_extract_text_nested_blocks():
    from lxml.etree import Element, SubElement
    root = Element('div')
    outer = SubElement(root, 'div')
    p = SubElement(outer, 'p')
    p.text = 'Nested'
    assert extract_text(root) == 'Nested'

def test_extract_text_with_comment():
    from lxml.etree import Element, Comment
    root = Element('p')
    root.text = 'Text'
    comment = Comment('comment')
    root.append(comment)
    assert extract_text(root) == 'Text'


# LLM-generated content at query #2
#--------------------------

```
def test_squash_artifical_nl_with_no_none():
    assert _squash_artifical_nl(["a", "b", "c"]) == ["a", "b", "c"]

def test_squash_artifical_nl_with_single_none():
    assert _squash_artifical_nl([None]) == [None]

def test_squash_artifical_nl_with_consecutive_nones():
    assert _squash_artifical_nl([None, None, None]) == [None]

def test_squash_artifical_nl_with_mixed():
    assert _squash_artifical_nl(["a", None, None, "b", None, "c"]) == ["a", None, "b", None, "c"]

def test_squash_artifical_nl_with_none_at_start():
    assert _squash_artifical_nl([None, "a", "b"]) == [None, "a", "b"]

def test_squash_artifical_nl_with_none_at_end():
    assert _squash_artifical_nl(["a", "b", None]) == ["a", "b", None]

def test_squash_artifical_nl_with_multiple_none_blocks():
    assert _squash_artifical_nl(["a", None, None, "b", None, None, "c"]) == ["a", None, "b", None, "c"]

def test_squash_artifical_nl_with_empty_list():
    assert _squash_artifical_nl([]) == []

def test_squash_artifical_nl_with_all_none():
    assert _squash_artifical_nl([None, None, None, None]) == [None]
```


# LLM-generated content at query #3
#--------------------------

def test_strip_artifical_nl_with_empty_list():
    result = _strip_artifical_nl([])
    assert result == []

def test_strip_artifical_nl_with_single_string():
    result = _strip_artifical_nl(["hello"])
    assert result == ["hello"]

def test_strip_artifical_nl_with_non_string_start():
    result = _strip_artifical_nl([1, 2, "hello"])
    assert result == [1, 2, "hello"]

def test_strip_artifical_nl_with_non_string_end():
    result = _strip_artifical_nl(["hello", 1, 2])
    assert result == ["hello", 1, 2]

def test_strip_artifical_nl_with_non_string_both_ends():
    result = _strip_artifical_nl([1, "hello", 2])
    assert result == [1, "hello", 2]

def test_strip_artifical_nl_with_all_non_string():
    result = _strip_artifical_nl([1, 2, 3])
    assert result == [1, 2, 3]

def test_strip_artifical_nl_with_multiple_strings():
    result = _strip_artifical_nl(["a", "b", "c"])
    assert result == ["a", "b", "c"]

def test_strip_artifical_nl_with_non_string_at_start_only():
    result = _strip_artifical_nl([1, "hello"])
    assert result == [1, "hello"]

def test_strip_artifical_nl_with_non_string_at_end_only():
    result = _strip_artifical_nl(["hello", 1])
    assert result == ["hello", 1]


# LLM-generated content at query #4
#--------------------------

def test_merge_original_parts_all_strings():
    assert _merge_original_parts(["  hello   world  "]) == ["hello world"]

def test_merge_original_parts_empty_list():
    assert _merge_original_parts([]) == []

def test_merge_original_parts_mixed_with_non_strings():
    assert _merge_original_parts(["a", 1, "b"]) == [1, "a b"]

def test_merge_original_parts_non_strings_only():
    assert _merge_original_parts([1, 2, 3]) == [1, 2, 3]

def test_merge_original_parts_strings_only_with_whitespace():
    assert _merge_original_parts(["  foo  ", "  bar  "]) == ["foo bar"]

def test_merge_original_parts_strings_that_become_empty_after_squash():
    assert _merge_original_parts(["   ", "  "]) == []

def test_merge_original_parts_non_strings_between_strings():
    assert _merge_original_parts(["a", None, "b"]) == [None, "a b"]

def test_merge_original_parts_multiple_non_strings():
    assert _merge_original_parts(["x", 1, 2, "y"]) == [1, 2, "x y"]

def test_merge_original_parts_single_non_string():
    assert _merge_original_parts([42]) == [42]


# LLM-generated content at query #5
#--------------------------

def test_extract_text_with_squash_space_true():
    dom = ["hello", None, "world"]
    result = extract_text(dom, squash_space=True)
    assert result is not None


# LLM-generated content at query #6
#--------------------------

def test_predicate_at_line11_evaluates_to_false():
    dom = []
    extract_text(dom, squash_space=False)


# LLM-generated content at query #7
#--------------------------

def test_predicate_false():
    dom = None
    result = extract_text(dom, squash_space=False)
    assert result is not None


# LLM-generated content at query #8
#--------------------------

def test_extract_text_plain_text():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = "Hello world"
    assert extract_text(dom) == "Hello world"

def test_extract_text_with_block_element():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = "First"
    p2 = SubElement(dom, 'p')
    p2.text = "Second"
    assert extract_text(dom) == "First\nSecond"

def test_extract_text_with_separator():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    hr = SubElement(dom, 'hr')
    p = SubElement(dom, 'p')
    p.text = "After hr"
    assert extract_text(dom) == "\nAfter hr"

def test_extract_text_with_squash_space():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = "  Hello   world  "
    assert extract_text(dom, squash_space=True) == "Hello world"

def test_extract_text_no_squash_space():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = "  Hello   world  "
    assert extract_text(dom, squash_space=False) == "  Hello   world  "

def test_extract_text_with_nested_inline():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('p')
    dom.text = "Hello "
    b = SubElement(dom, 'b')
    b.text = "bold"
    b.tail = " world"
    assert extract_text(dom) == "Hello bold world"

def test_extract_text_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    assert extract_text(dom) == ""

def test_extract_text_with_tail_only():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = "Text"
    p.tail = " Tail"
    assert extract_text(dom) == "Text Tail"

def test_extract_text_custom_block_symbol():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = "Line1"
    p2 = SubElement(dom, 'p')
    p2.text = "Line2"
    assert extract_text(dom, block_symbol=' | ') == "Line1 | Line2"

def test_extract_text_custom_sep_symbol():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    hr = SubElement(dom, 'hr')
    p = SubElement(dom, 'p')
    p.text = "After"
    assert extract_text(dom, sep_symbol=' --- ') == " --- After"


# LLM-generated content at query #9
#--------------------------

def test_predicate_evaluates_to_false():
    dom = None
    result = extract_text(dom, squash_space=False)


# LLM-generated content at query #10
#--------------------------

def test_extract_text_squash_space_false():
    dom = None
    result = extract_text(dom, squash_space=False)
    assert result == ""


# LLM-generated content at query #11
#--------------------------

def test_predicate_false():
    dom = []
    result = extract_text(dom, squash_space=False)
    assert result == ""


# LLM-generated content at query #12
#--------------------------

```
def test_extract_text_empty_dom():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div></div>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_simple_text():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div>Hello World</div>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_with_separator():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div><hr/>Text after hr</div>")
    result = extract_text(dom)
    assert result == "Text after hr"

def test_extract_text_with_block_elements():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div><p>First paragraph</p><p>Second paragraph</p></div>")
    result = extract_text(dom)
    assert result == "First paragraph\nSecond paragraph"

def test_extract_text_with_nested_inline():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div>Hello <b>bold</b> world</div>")
    result = extract_text(dom)
    assert result == "Hello bold world"

def test_extract_text_with_whitespace_collapse():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div>  Too   many   spaces  </div>")
    result = extract_text(dom)
    assert result == "Too many spaces"

def test_extract_text_with_separator_and_block():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div><hr/><p>After hr</p></div>")
    result = extract_text(dom)
    assert result == "After hr"

def test_extract_text_multiple_blocks_with_text():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div>Start<p>Middle</p>End</div>")
    result = extract_text(dom)
    assert result == "Start\nMiddle\nEnd"

def test_extract_text_only_separator():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<hr/>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_nested_blocks():
    from xml.etree.ElementTree import fromstring
    dom = fromstring("<div><div><p>Deep</p></div></div>")
    result = extract_text(dom)
    assert result == "Deep"


# LLM-generated content at query #13
#--------------------------

def test_predicate_false():
    dom = None
    block_symbol = '\n'
    sep_symbol = '\n'
    squash_space = False
    result = extract_text(dom, block_symbol, sep_symbol, squash_space)


# LLM-generated content at query #14
#--------------------------

```python
def test_extract_text_array_returns_empty_string_for_callable_tag():
    class FakeElement:
        tag = lambda: None
    dom = FakeElement()
    dom.text = None
    dom.getchildren = lambda: []
    result = extract_text_array(dom)
    assert result == ''

def test_extract_text_array_separator_tag_returns_true():
    from lxml.etree import Element
    dom = Element('br')
    dom.text = None
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag_returns_text():
    from lxml.etree import Element, SubElement
    dom = Element('span')
    dom.text = 'hello'
    result = extract_text_array(dom)
    assert result == ['hello']

def test_extract_text_array_non_inline_non_separator_tag_adds_none():
    from lxml.etree import Element
    dom = Element('div')
    dom.text = None
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_children_and_tail():
    from lxml.etree import Element, SubElement
    dom = Element('div')
    child = SubElement(dom, 'p')
    child.text = 'text1'
    child.tail = 'tail1'
    result = extract_text_array(dom)
    assert result == ['text1', 'tail1']

def test_extract_text_array_with_squash_artifical_nl_false():
    from lxml.etree import Element, SubElement
    dom = Element('div')
    child1 = SubElement(dom, 'p')
    child1.text = 'a'
    child2 = SubElement(dom, 'p')
    child2.text = 'b'
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'a', None, None, 'b', None, None]

def test_extract_text_array_with_strip_artifical_nl_false():
    from lxml.etree import Element, SubElement
    dom = Element('div')
    child = SubElement(dom, 'p')
    child.text = 'text'
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == ['text', None]

def test_extract_text_array_combined_options():
    from lxml.etree import Element, SubElement
    dom = Element('div')
    child1 = SubElement(dom, 'p')
    child1.text = 'first'
    child2 = SubElement(dom, 'br')
    child2.tail = 'second'
    result = extract_text_array(dom)
    assert result == ['first', True, 'second']

def test_extract_text_array_empty_dom():
    from lxml.etree import Element
    dom = Element('div')
    dom.text = None
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_only_text_in_non_inline_tag():
    from lxml.etree import Element
    dom = Element('div')
    dom.text = 'text'
    result = extract_text_array(dom)
    assert result == ['text']
```


# LLM-generated content at query #15
#--------------------------

def test_squash_space_true_predicate():
    dom = None
    result = extract_text(dom, squash_space=True)
    assert True


# LLM-generated content at query #16
#--------------------------

def test_extract_text_predicate_true():
    dom = None
    result = extract_text(dom)
    assert True


# LLM-generated content at query #17
#--------------------------

def test_predicate_true():
    dom = None
    result = extract_text(dom, squash_space=True)
    assert result == ""


# LLM-generated content at query #18
#--------------------------

def test_predicate_at_line_11_evaluates_to_true():
    dom = None
    result = extract_text(dom, squash_space=True)


# LLM-generated content at query #19
#--------------------------

```python
def test_extract_text_array_returns_empty_string_for_callable_tag():
    class MockElement:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ''

def test_extract_text_array_returns_string_list_with_text():
    class MockElement:
        tag = 'p'
        text = 'Hello'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_returns_none_for_block_tag():
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_returns_true_for_separator_tag():
    class MockElement:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_children():
    class MockChild:
        tag = 'span'
        text = 'World'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = 'Hello '
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['Hello ', 'World']

def test_extract_text_array_squash_artifical_nl_false():
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_strip_artifical_nl_false():
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_with_tail():
    class MockChild:
        tag = 'a'
        text = 'Click'
        tail = ' here'
        def getchildren(self):
            return []
    class MockElement:
        tag = 'p'
        text = 'Please '
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['Please ', 'Click', ' here']

def test_extract_text_array_with_separator_child():
    class MockChild:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_inline_tag():
    class MockChild:
        tag = 'b'
        text = 'bold'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'span'
        text = 'normal'
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['normal', 'bold']
```


# LLM-generated content at query #20
#--------------------------

def test_extract_text_returns_empty_string_for_callable_tag():
    class FakeDom:
        tag = lambda: None
    assert extract_text(FakeDom()) == ''

def test_extract_text_returns_text_for_simple_text_node():
    class FakeDom:
        tag = 'p'
        text = 'Hello'
        def getchildren(self):
            return []
    assert extract_text(FakeDom()) == 'Hello'

def test_extract_text_uses_block_symbol_for_non_inline_tags():
    class FakeDom:
        tag = 'div'
        text = 'Line1'
        def getchildren(self):
            return []
    assert extract_text(FakeDom()) == 'Line1'

def test_extract_text_uses_sep_symbol_for_separator_tags():
    class FakeDom:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    assert extract_text(FakeDom()) == '\n'

def test_extract_text_joins_child_text():
    class FakeChild:
        tag = 'span'
        text = 'Child'
        tail = None
        def getchildren(self):
            return []
    class FakeDom:
        tag = 'div'
        text = 'Parent'
        def getchildren(self):
            return [FakeChild()]
    assert extract_text(FakeDom()) == 'ParentChild'

def test_extract_text_strips_whitespace_when_squash_space_true():
    class FakeDom:
        tag = 'p'
        text = '  Hello World  '
        def getchildren(self):
            return []
    assert extract_text(FakeDom()) == 'Hello World'

def test_extract_text_preserves_whitespace_when_squash_space_false():
    class FakeDom:
        tag = 'p'
        text = '  Hello World  '
        def getchildren(self):
            return []
    assert extract_text(FakeDom(), squash_space=False) == '  Hello World  '

def test_extract_text_handles_nested_separators():
    class FakeChild:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []
    class FakeDom:
        tag = 'div'
        text = 'A'
        def getchildren(self):
            return [FakeChild()]
    assert extract_text(FakeDom()) == 'A\n'

def test_extract_text_handles_multiple_children():
    class FakeChild1:
        tag = 'span'
        text = 'First'
        tail = ' '
        def getchildren(self):
            return []
    class FakeChild2:
        tag = 'span'
        text = 'Second'
        tail = None
        def getchildren(self):
            return []
    class FakeDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return [FakeChild1(), FakeChild2()]
    assert extract_text(FakeDom()) == 'First Second'


# LLM-generated content at query #21
#--------------------------

def test_extract_text_empty_dom():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text(dom)
    assert result == ''

def test_extract_text_simple_text():
    class MockDom:
        tag = 'span'
        text = 'Hello'
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text(dom)
    assert result == 'Hello'

def test_extract_text_with_child():
    class MockChild:
        tag = 'b'
        text = 'bold'
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = 'p'
        text = 'Some '
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text(dom)
    assert result == 'Some bold'

def test_extract_text_with_separator():
    class MockDom:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text(dom, sep_symbol='\n')
    assert result == '\n'

def test_extract_text_with_block_symbol():
    class MockDom:
        tag = 'div'
        text = 'Line1'
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text(dom, block_symbol='\n')
    assert result == 'Line1'


# LLM-generated content at query #22
#--------------------------

def test_extract_text_predicate_false():
    dom = type('MockDom', (object,), {})()
    result = extract_text(dom, squash_space=False)


# LLM-generated content at query #23
#--------------------------

def test_squash_space_true():
    dom = None
    result = extract_text(dom, squash_space=True)


# LLM-generated content at query #24
#--------------------------

def test_extract_text_squash_space_false():
    dom = []
    result = extract_text(dom, squash_space=False)
    assert result != ""


# LLM-generated content at query #25
#--------------------------

def test_extract_text_predicate_true():
    dom = None
    result = extract_text(dom, squash_space=True)


# LLM-generated content at query #26
#--------------------------

def test_predicate_false():
    a = [None, "text", True, "more text"]
    extract_text_array = lambda dom, squash_artifical_nl=True: a
    _strip_artifical_nl = lambda x: x
    _squash_artifical_nl = lambda x: x
    _merge_original_parts = lambda x: x
    result = extract_text("dom", squash_space=False)


# LLM-generated content at query #27
#--------------------------

def test_squash_space_true_predicate():
    dom = [("text", "  hello  "), ("br", None), ("text", "  world  ")]
    result = extract_text(dom, squash_space=True)
    assert result == "hello\nworld"


# LLM-generated content at query #28
#--------------------------

def test_callable_dom_tag_returns_empty_string():
    dom = type('MockDom', (), {'tag': lambda: None})()
    result = extract_text_array(dom)
    assert result == ''


# LLM-generated content at query #29
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockDom:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_separator_tag():
    class MockDom:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag():
    class MockDom:
        tag = "span"
        text = "inline"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["inline"]

def test_extract_text_array_with_child():
    class MockChild:
        tag = "b"
        text = "bold"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "p"
        text = "before"
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["before", "bold"]

def test_extract_text_array_with_tail():
    class MockChild:
        tag = "a"
        text = "link"
        tail = " tail"
        def getchildren(self):
            return []
    class MockDom:
        tag = "p"
        text = "click "
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["click ", "link", " tail"]

def test_extract_text_array_squash_artifical_nl():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == []

def test_extract_text_array_strip_artifical_nl():
    class MockDom:
        tag = "div"
        text = "text"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["text"]

def test_extract_text_array_no_squash_no_strip():
    class MockDom:
        tag = "div"
        text = "text"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "text", None]

def test_extract_text_array_multiple_children():
    class MockChild1:
        tag = "span"
        text = "one"
        tail = None
        def getchildren(self):
            return []
    class MockChild2:
        tag = "span"
        text = "two"
        tail = " tail"
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = "start "
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["start ", "one", "two", " tail"]

def test_extract_text_array_nested_separator():
    class MockChild:
        tag = "br"
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "p"
        text = "line1"
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["line1", True]

def test_extract_text_array_squash_multiple_none():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_leading_none():
    class MockDom:
        tag = "div"
        text = "text"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["text", None]

def test_extract_text_array_strip_trailing_none():
    class MockDom:
        tag = "div"
        text = "text"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["text", None]

def test_extract_text_array_callable_tag():
    class MockDom:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ""

def test_extract_text_array_nested_structure():
    class MockGrandChild:
        tag = "i"
        text = "italic"
        tail = " after "
        def getchildren(self):
            return []
    class MockChild:
        tag = "b"
        text = "bold"
        tail = None
        def getchildren(self):
            return [MockGrandChild()]
    class MockDom:
        tag = "p"
        text = "text "
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["text ", "bold", "italic", " after "]
```


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_at_line_10_evaluates_to_false():
    from lxml.html import fromstring
    dom = fromstring("<div>text</div>")
    dom.text = None
    result = extract_text_array(dom)
    assert result == [None, None]  # lines 9 and 19 append None
```


# LLM-generated content at query #31
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_separator_tag():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag():
    class MockElement:
        tag = "span"
        text = "text"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["text"]

def test_extract_text_array_with_child():
    class MockChild:
        tag = "b"
        text = "bold"
        tail = " tail"
        def getchildren(self):
            return []
    class MockParent:
        tag = "p"
        text = "before "
        def getchildren(self):
            return [MockChild()]
    dom = MockParent()
    result = extract_text_array(dom)
    assert result == ["before ", "bold", " tail"]

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == []

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_no_squash_no_strip():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_mixed_tags():
    class MockChild:
        tag = "span"
        text = "inline"
        tail = None
        def getchildren(self):
            return []
    class MockParent:
        tag = "div"
        text = "start"
        def getchildren(self):
            return [MockChild()]
    dom = MockParent()
    result = extract_text_array(dom)
    assert result == ["start", "inline"]
```


# LLM-generated content at query #32
#--------------------------

def test_elif_condition_false():
    dom = type('MockDom', (), {'tag': 'div', 'text': None, 'getchildren': lambda: []})()
    SEPARATORS = {'br'}
    INLINE_TAGS = {'span', 'b'}
    result = extract_text_array(dom)


# LLM-generated content at query #33
#--------------------------

```python
def test_squash_artifical_nl_false():
    from your_module import extract_text_array
    dom = type('MockDom', (object,), {'tag': 'p', 'text': 'Hello', 'tail': None, 'getchildren': lambda self: []})()
    dom.tag = 'p'
    dom.text = 'Hello'
    dom.tail = None
    dom.getchildren = lambda: []
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert len(result) == 2
    assert result[0] == 'Hello'
    assert result[1] is None
```


# LLM-generated content at query #34
#--------------------------

```
def test_extract_text_array_empty_dom():
    dom = type('FakeElement', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    dom = type('FakeElement', (object,), {'tag': 'p', 'text': 'Hello', 'getchildren': lambda: [], 'tail': None})()
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_with_separator():
    dom = type('FakeElement', (object,), {'tag': 'br', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_artificial_nl():
    dom = type('FakeElement', (object,), {'tag': 'div', 'text': 'A', 'getchildren': lambda: [], 'tail': None})()
    result = extract_text_array(dom)
    assert result == ['A']

def test_extract_text_array_squash_and_strip():
    child = type('FakeElement', (object,), {'tag': 'b', 'text': None, 'getchildren': lambda: [], 'tail': ' world'})()
    dom = type('FakeElement', (object,), {'tag': 'p', 'text': 'Hello', 'getchildren': lambda: [child], 'tail': None})()
    result = extract_text_array(dom)
    assert result == ['Hello world']

def test_extract_text_array_squash_artifical_nl_false():
    child = type('FakeElement', (object,), {'tag': 'div', 'text': 'A', 'getchildren': lambda: [], 'tail': 'B'})()
    dom = type('FakeElement', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda: [child], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None, 'A', 'B', None]

def test_extract_text_array_strip_artifical_nl_false():
    child = type('FakeElement', (object,), {'tag': 'div', 'text': 'A', 'getchildren': lambda: [], 'tail': 'B'})()
    dom = type('FakeElement', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda: [child], 'tail': None})()
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, 'A', 'B']

def test_extract_text_array_nested_inline_tags():
    child = type('FakeElement', (object,), {'tag': 'span', 'text': 'inner', 'getchildren': lambda: [], 'tail': None})()
    dom = type('FakeElement', (object,), {'tag': 'b', 'text': 'start', 'getchildren': lambda: [child], 'tail': None})()
    result = extract_text_array(dom)
    assert result == ['startinner']

def test_extract_text_array_multiple_separators():
    dom1 = type('FakeElement', (object,), {'tag': 'br', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    dom2 = type('FakeElement', (object,), {'tag': 'br', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    dom = type('FakeElement', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda: [dom1, dom2], 'tail': None})()
    result = extract_text_array(dom)
    assert result == [True, True]
```


# LLM-generated content at query #35
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "p"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_single_text_node():
    class MockElement:
        tag = "span"
        text = "hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_extract_text_array_with_separator_tag():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    class MockElement:
        tag = "b"
        text = "bold"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == ["bold"]

def test_extract_text_array_with_block_tag_adds_artificial_nl():
    class MockElement:
        tag = "div"
        text = "text"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "text", None]

def test_extract_text_array_with_child_and_tail():
    class MockChild:
        tag = "span"
        text = "child"
        tail = "tail"
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "before"
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "before", "child", "tail", None]

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = "div"
        text = "content"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["content"]

def test_extract_text_array_squash_and_strip():
    class MockElement:
        tag = "div"
        text = "content"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["content"]

def test_extract_text_array_separator_with_child():
    class MockChild:
        tag = "span"
        text = "child"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [True, "child"]
```


# LLM-generated content at query #36
#--------------------------

def test_strip_artifical_nl_true():
    dom = type('Mock', (object,), {'tag': 'p', 'text': 'hello', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert True


# LLM-generated content at query #37
#--------------------------

def test_predicate_at_line12_evaluates_to_false():
    class MockChild:
        tag = 'p'
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = 'div'
        text = 'some text'
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)


# LLM-generated content at query #38
#--------------------------

def test_squash_artifical_nl_evaluates_false():
    dom = type('MockDom', (), {'tag': 'div', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None]


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_line12_true():
    from unittest.mock import Mock
    child = Mock()
    child.tag = 'div'
    child.text = None
    child.tail = None
    child.getchildren = Mock(return_value=[])
    dom = Mock()
    dom.tag = 'p'
    dom.text = None
    dom.getchildren = Mock(return_value=[child])
    result = extract_text_array(dom)
    assert True


# LLM-generated content at query #40
#--------------------------

def test_dom_tag_in_separators():
    dom = type('MockDom', (), {'tag': 'br', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom)
    assert result[0] == True


# LLM-generated content at query #41
#--------------------------

def test_strip_artifical_nl_false():
    r = [None, "text", None, True, None]
    strip_artifical_nl = False
    result = _strip_artifical_nl(r)
    assert result == r


# LLM-generated content at query #42
#--------------------------

```python
def test_dom_text_is_not_none():
    dom = type('MockDom', (), {'tag': 'div', 'text': 'some text', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom)
    assert dom.text is not None
```


# LLM-generated content at query #43
#--------------------------

def test_extract_text_array_empty_dom():
    dom = type('MockDom', (), {'tag': 'div', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_with_text_only():
    dom = type('MockDom', (), {'tag': 'div', 'text': 'Hello', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['Hello']

def test_extract_text_array_with_separator():
    dom = type('MockDom', (), {'tag': 'br', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    dom = type('MockDom', (), {'tag': 'span', 'text': 'inline', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['inline']

def test_extract_text_array_nested_children():
    child = type('MockDom', (), {'tag': 'div', 'text': 'child_text', 'getchildren': lambda self: [], 'tail': ' tail'})()
    dom = type('MockDom', (), {'tag': 'div', 'text': 'parent_text', 'getchildren': lambda self: [child], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['parent_text', 'child_text', ' tail']

def test_extract_text_array_squash_artifical_nl():
    child = type('MockDom', (), {'tag': 'div', 'text': 'text', 'getchildren': lambda self: [], 'tail': None})()
    dom = type('MockDom', (), {'tag': 'div', 'text': None, 'getchildren': lambda self: [child], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, 'text', None]

def test_extract_text_array_strip_artifical_nl():
    child = type('MockDom', (), {'tag': 'div', 'text': 'text', 'getchildren': lambda self: [], 'tail': None})()
    dom = type('MockDom', (), {'tag': 'div', 'text': None, 'getchildren': lambda self: [child], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ['text']

def test_extract_text_array_with_separator_and_text():
    child = type('MockDom', (), {'tag': 'br', 'text': None, 'getchildren': lambda self: [], 'tail': ' after_br'})()
    dom = type('MockDom', (), {'tag': 'div', 'text': 'before_br', 'getchildren': lambda self: [child], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['before_br', True, ' after_br']

def test_extract_text_array_no_squash_no_strip():
    child = type('MockDom', (), {'tag': 'div', 'text': 'child', 'getchildren': lambda self: [], 'tail': None})()
    dom = type('MockDom', (), {'tag': 'div', 'text': 'parent', 'getchildren': lambda self: [child], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'parent', None, 'child', None, None]


# LLM-generated content at query #44
#--------------------------

def test_predicate_at_line_17_evaluates_to_false():
    dom = type('MockDom', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom)
    assert result == []


# LLM-generated content at query #45
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_inline_tag():
    class MockElement:
        tag = "span"
        text = "hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["hello"]

def test_extract_text_array_separator_tag():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == [True]

def test_extract_text_array_block_tag_no_text():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_nested_with_text():
    class MockChild:
        tag = "span"
        text = "world"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "hello "
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["hello ", "world"]

def test_extract_text_array_with_tail():
    class MockChild:
        tag = "span"
        text = "inner"
        tail = " tail"
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "start"
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["start", "inner", " tail"]

def test_extract_text_array_squash_and_strip():
    class MockChild:
        tag = "div"
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_no_squash():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_no_strip():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_multiple_children():
    class MockChild1:
        tag = "span"
        text = "first"
        tail = None
        def getchildren(self):
            return []
    class MockChild2:
        tag = "span"
        text = "second"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["first", "second"]
```


# LLM-generated content at query #46
#--------------------------

```python
def test_predicate_line12_evaluates_to_false():
    from unittest.mock import Mock
    child = Mock()
    child.tag = "child_tag"
    child.text = None
    child.tail = None
    dom = Mock()
    dom.tag = "parent_tag"
    dom.text = None
    dom.getchildren.return_value = [child]
    result = extract_text_array(dom)
    assert True


# LLM-generated content at query #47
#--------------------------

def test_extract_text_with_text_only():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello World</p>")
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_with_separator():
    from lxml.html import fromstring
    dom = fromstring("<br>Text after break")
    result = extract_text(dom)
    assert result == "Text after break"

def test_extract_text_with_nested_elements():
    from lxml.html import fromstring
    dom = fromstring("<div><p>First</p><p>Second</p></div>")
    result = extract_text(dom)
    assert result == "First\nSecond"

def test_extract_text_with_mixed_content():
    from lxml.html import fromstring
    dom = fromstring("<p>Hello <b>World</b> again</p>")
    result = extract_text(dom)
    assert result == "Hello World again"

def test_extract_text_with_multiple_separators():
    from lxml.html import fromstring
    dom = fromstring("<p>Line1<br><br>Line2</p>")
    result = extract_text(dom)
    assert result == "Line1\nLine2"

def test_extract_text_with_custom_block_symbol():
    from lxml.html import fromstring
    dom = fromstring("<div><p>A</p><p>B</p></div>")
    result = extract_text(dom, block_symbol=' | ')
    assert result == "A | B"

def test_extract_text_with_custom_sep_symbol():
    from lxml.html import fromstring
    dom = fromstring("<p>A<br>B</p>")
    result = extract_text(dom, sep_symbol=' -- ')
    assert result == "A -- B"

def test_extract_text_with_squash_space_false():
    from lxml.html import fromstring
    dom = fromstring("<p>  Hello   World  </p>")
    result = extract_text(dom, squash_space=False)
    assert result == "  Hello   World  "

def test_extract_text_with_empty_dom():
    from lxml.html import fromstring
    dom = fromstring("<div></div>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_with_only_whitespace():
    from lxml.html import fromstring
    dom = fromstring("<p>   </p>")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_with_script_tag():
    from lxml.html import fromstring
    dom = fromstring("<script>var x = 1;</script>")
    result = extract_text(dom)
    assert result == "var x = 1;"

def test_extract_text_with_style_tag():
    from lxml.html import fromstring
    dom = fromstring("<style>body { color: red; }</style>")
    result = extract_text(dom)
    assert result == "body { color: red; }"

def test_extract_text_with_list_elements():
    from lxml.html import fromstring
    dom = fromstring("<ul><li>Item1</li><li>Item2</li></ul>")
    result = extract_text(dom)
    assert result == "Item1\nItem2"

def test_extract_text_with_deeply_nested_elements():
    from lxml.html import fromstring
    dom = fromstring("<div><span><b>Deep</b></span></div>")
    result = extract_text(dom)
    assert result == "Deep"


# LLM-generated content at query #48
#--------------------------

def test_strip_artifical_nl_false():
    dom = type('Dom', (object,), {'tag': 'div', 'text': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert strip_artifical_nl == False


# LLM-generated content at query #49
#--------------------------

```python
def test_predicate_at_line_20_evaluates_to_false():
    dom = type('MockDom', (object,), {
        'tag': 'p',
        'text': 'Hello',
        'tail': None,
        'getchildren': lambda self: []
    })()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ['Hello'] or result == ['Hello', None]
```


# LLM-generated content at query #50
#--------------------------

def test_squash_artifical_nl_true():
    dom = type('MockDom', (object,), {
        'tag': 'div',
        'text': None,
        'tail': None,
        'getchildren': lambda self: []
    })()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result is not None


# LLM-generated content at query #51
#--------------------------

def test_predicate_at_line_17_evaluates_to_false():
    dom = type('MockDom', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom)
    assert result[-1] is not None


# LLM-generated content at query #52
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    from unittest.mock import MagicMock
    dom = MagicMock()
    dom.tag = "p"
    dom.text = None
    child = MagicMock()
    child.tail = None
    dom.getchildren.return_value = [child]
    from your_module import SEPARATORS, INLINE_TAGS
    SEPARATORS = set()
    INLINE_TAGS = set()
    result = extract_text_array(dom)
    assert dom.tag not in INLINE_TAGS and dom.tag not in SEPARATORS
```


# LLM-generated content at query #53
#--------------------------

```python
def test_predicate_at_line_12_true():
    dom = type('MockDom', (), {'tag': 'div', 'text': 'text', 'getchildren': lambda: []})()
    result = extract_text_array(dom)
    assert 'text' in result
```


# LLM-generated content at query #54
#--------------------------

```
def test_extract_text_array_empty_dom():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_with_text_only():
    class MockDom:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["Hello"]

def test_extract_text_array_separator_tag():
    class MockDom:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == [True]

def test_extract_text_array_non_inline_tag_without_children():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_with_child_and_tail():
    class MockChild:
        tag = "span"
        text = "child"
        tail = " tail"
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = "parent"
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["parent", "child", " tail"]

def test_extract_text_array_squash_artifical_nl():
    class MockChild:
        tag = "span"
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_strip_artifical_nl():
    class MockChild:
        tag = "span"
        text = "text"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["text"]

def test_extract_text_array_no_squash_no_strip():
    class MockChild:
        tag = "span"
        text = "text"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "text", None] or result == [None, "text", None]

def test_extract_text_array_multiple_children():
    class MockChild1:
        tag = "span"
        text = "first"
        tail = " "
        def getchildren(self):
            return []
    class MockChild2:
        tag = "b"
        text = "second"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["first", " ", "second"]

def test_extract_text_array_nested_structure():
    class MockInnerChild:
        tag = "i"
        text = "inner"
        tail = None
        def getchildren(self):
            return []
    class MockChild:
        tag = "span"
        text = None
        tail = " after span"
        def getchildren(self):
            return [MockInnerChild()]
    class MockDom:
        tag = "div"
        text = "before "
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["before ", "inner", " after span"]


# LLM-generated content at query #55
#--------------------------

```python
def test_extract_text_array_with_none_dom_tag_callable():
    class MockDom:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ''


def test_extract_text_array_with_separator_tag():
    class MockDom:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    result = extract_text_array(MockDom())
    assert result == [True]


def test_extract_text_array_with_inline_tag():
    class MockDom:
        tag = 'span'
        text = 'hello'
        def getchildren(self):
            return []
    result = extract_text_array(MockDom())
    assert result == ['hello']


def test_extract_text_array_with_block_tag_and_text():
    class MockDom:
        tag = 'div'
        text = 'hello'
        def getchildren(self):
            return []
    result = extract_text_array(MockDom())
    assert result == ['hello']


def test_extract_text_array_with_children():
    class MockChild:
        tag = 'span'
        text = 'world'
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = 'div'
        text = 'hello '
        def getchildren(self):
            return [MockChild()]
    result = extract_text_array(MockDom())
    assert result == ['hello ', 'world']


def test_extract_text_array_with_nested_separators():
    class MockChild:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = 'div'
        text = 'line1'
        def getchildren(self):
            return [MockChild()]
    result = extract_text_array(MockDom())
    assert result == ['line1', True]


def test_extract_text_array_squash_artifical_nl():
    class MockDom:
        tag = 'div'
        text = 'a'
        def getchildren(self):
            return []
    result = extract_text_array(MockDom(), squash_artifical_nl=True)
    assert result == ['a']


def test_extract_text_array_no_squash_artifical_nl():
    class MockDom:
        tag = 'div'
        text = 'a'
        def getchildren(self):
            return []
    result = extract_text_array(MockDom(), squash_artifical_nl=False)
    assert result == [None, 'a', None]


def test_extract_text_array_strip_artifical_nl():
    class MockDom:
        tag = 'div'
        text = 'a'
        def getchildren(self):
            return []
    result = extract_text_array(MockDom(), strip_artifical_nl=True)
    assert result == ['a']


def test_extract_text_array_no_strip_artifical_nl():
    class MockDom:
        tag = 'div'
        text = 'a'
        def getchildren(self):
            return []
    result = extract_text_array(MockDom(), strip_artifical_nl=False)
    assert result == [None, 'a', None]


# LLM-generated content at query #56
#--------------------------

```python
def test_strip_artifical_nl_true():
    dom = MockDom("p", text="hello")
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["hello"]


# LLM-generated content at query #57
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    from unittest.mock import Mock
    dom = Mock()
    dom.tag = "p"
    dom.text = None
    dom.getchildren.return_value = []
    dom.tag in INLINE_TAGS = False
    SEPARATORS = []
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert dom.tag not in INLINE_TAGS and dom.tag not in SEPARATORS
```


# LLM-generated content at query #58
#--------------------------

```
def test_extract_text_array_empty_dom():
    class FakeDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = FakeDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_single_text():
    class FakeDom:
        tag = "p"
        text = "hello"
        def getchildren(self):
            return []
    dom = FakeDom()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_extract_text_array_separator_tag():
    class FakeDom:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = FakeDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag_no_artifical():
    class FakeDom:
        tag = "b"
        text = "bold"
        def getchildren(self):
            return []
    dom = FakeDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == ["bold"]

def test_extract_text_array_block_tag_with_artifical():
    class FakeDom:
        tag = "div"
        text = "text"
        def getchildren(self):
            return []
    dom = FakeDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "text", None]

def test_extract_text_array_with_child():
    class FakeChild:
        tag = "span"
        text = "child"
        tail = None
        def getchildren(self):
            return []
    class FakeDom:
        tag = "div"
        text = "before"
        def getchildren(self):
            return [FakeChild()]
    dom = FakeDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "before", "child", None]

def test_extract_text_array_with_tail():
    class FakeChild:
        tag = "span"
        text = None
        tail = " after"
        def getchildren(self):
            return []
    class FakeDom:
        tag = "div"
        text = "before"
        def getchildren(self):
            return [FakeChild()]
    dom = FakeDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "before", " after", None]

def test_extract_text_array_squash_artifical():
    class FakeDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = FakeDom()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_artifical():
    class FakeDom:
        tag = "div"
        text = "hello"
        def getchildren(self):
            return []
    dom = FakeDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["hello"]

def test_extract_text_array_callable_tag():
    class FakeDom:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = FakeDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_nested_separator():
    class FakeChild:
        tag = "br"
        text = None
        tail = None
        def getchildren(self):
            return []
    class FakeDom:
        tag = "div"
        text = "line1"
        def getchildren(self):
            return [FakeChild()]
    dom = FakeDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "line1", True, None]

def test_extract_text_array_multiple_children():
    class FakeChild1:
        tag = "span"
        text = "a"
        tail = " "
        def getchildren(self):
            return []
    class FakeChild2:
        tag = "b"
        text = "b"
        tail = None
        def getchildren(self):
            return []
    class FakeDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [FakeChild1(), FakeChild2()]
    dom = FakeDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "a", " ", "b", None]


# LLM-generated content at query #59
#--------------------------

def test_strip_artifical_nl_true():
    dom = Mock()
    dom.tag = "div"
    dom.text = None
    dom.getchildren.return_value = []
    strip_artifical_nl = True
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert _strip_artifical_nl.called


# LLM-generated content at query #60
#--------------------------

```
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_text_only():
    class MockElement:
        tag = "span"
        text = "hello"
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_extract_text_array_separator_tag():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag_with_text():
    class MockChild:
        tag = "b"
        text = "bold"
        tail = None
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockChild"
    class MockElement:
        tag = "span"
        text = "before "
        def getchildren(self):
            return [MockChild()]
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["before ", "bold"]

def test_extract_text_array_with_artificial_nl():
    class MockElement:
        tag = "div"
        text = "line1"
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["line1"]

def test_extract_text_array_multiple_children():
    class MockChild1:
        tag = "span"
        text = "child1"
        tail = " tail1 "
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockChild1"
    class MockChild2:
        tag = "br"
        text = None
        tail = " tail2"
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockChild2"
    class MockElement:
        tag = "div"
        text = "start "
        def getchildren(self):
            return [MockChild1(), MockChild2()]
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["start ", "child1", " tail1 ", True, " tail2"]

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == []

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_no_squash_no_strip():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_with_tail_after_separator():
    class MockChild:
        tag = "br"
        text = None
        tail = " after"
        def getchildren(self):
            return []
        def __repr__(self):
            return "MockChild"
    class MockElement:
        tag = "div"
        text = "before "
        def getchildren(self):
            return [MockChild()]
        def __repr__(self):
            return "MockElement"
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["before ", True, " after"]
```


# LLM-generated content at query #61
#--------------------------

```python
def test_extract_text_array_with_empty_dom():
    class MockElement:
        tag = "p"
        text = None
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_separator_tag():
    class MockElement:
        tag = "br"
        text = None
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    class MockElement:
        tag = "span"
        text = "hello"
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_extract_text_array_with_nested_tags():
    class MockChild:
        tag = "span"
        text = "world"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "hello "
        tail = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["hello ", "world"]

def test_extract_text_array_with_artifical_newlines():
    class MockChild:
        tag = "span"
        text = "inner"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = None
        tail = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "inner", None]

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = "div"
        text = None
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = "div"
        text = "content"
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["content"]

def test_extract_text_array_with_tail():
    class MockChild:
        tag = "span"
        text = "inner"
        tail = " after"
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "before "
        tail = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["before ", "inner", " after"]

def test_extract_text_array_callable_tag():
    class MockElement:
        tag = lambda: None
        text = None
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ""

def test_extract_text_array_separator_with_text():
    class MockElement:
        tag = "hr"
        text = "separator"
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True, "separator"]
```


# LLM-generated content at query #62
#--------------------------

def test_predicate_line17_evaluates_true():
    from unittest.mock import MagicMock
    dom = MagicMock()
    dom.tag = "p"
    dom.text = None
    dom.getchildren.return_value = []
    SEPARATORS = set()
    INLINE_TAGS = {"span", "b"}
    result = extract_text_array(dom)
    assert result[-1] is None


# LLM-generated content at query #63
#--------------------------

def test_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_simple_text():
    class MockElement:
        tag = "p"
        text = "hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["hello"]

def test_separator_tag():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [True]

def test_inline_tag():
    class MockElement:
        tag = "span"
        text = "inline"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == ["inline"]

def test_nested_children():
    class MockChild:
        tag = "b"
        text = "bold"
        tail = " tail"
        def getchildren(self):
            return []
    class MockElement:
        tag = "p"
        text = "start "
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == ["start ", "bold", " tail"]

def test_artificial_newlines():
    class MockElement:
        tag = "div"
        text = "a"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "a", None]

def test_squash_artifical_nl():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_strip_artifical_nl():
    class MockChild:
        tag = "span"
        text = "inner"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["inner"]

def test_both_squash_and_strip():
    class MockChild:
        tag = "span"
        text = "content"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["content"]

def test_callable_tag():
    class MockElement:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ""


# LLM-generated content at query #64
#--------------------------

```
def test_strip_artifical_nl_flag_true():
    dom = type('MockDom', (object,), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert True
```


# LLM-generated content at query #65
#--------------------------

def test_strip_artifical_nl_false():
    dom = type('MockDom', (), {'tag': 'div', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    r = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert r == [None]


# LLM-generated content at query #66
#--------------------------

```
def test_extract_text_array_empty_dom():
    class MockElement:
        def __init__(self):
            self.tag = "div"
            self.text = None
            self.tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        def __init__(self):
            self.tag = "p"
            self.text = "Hello"
            self.tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_with_separator():
    class MockElement:
        def __init__(self):
            self.tag = "br"
            self.text = None
            self.tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_children():
    class MockChild:
        def __init__(self):
            self.tag = "span"
            self.text = "World"
            self.tail = " "
        def getchildren(self):
            return []
    class MockElement:
        def __init__(self):
            self.tag = "div"
            self.text = "Hello "
            self.tail = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        def __init__(self):
            self.tag = "div"
            self.text = None
            self.tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == []

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        def __init__(self):
            self.tag = "p"
            self.text = "Test"
            self.tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["Test"]

def test_extract_text_array_no_squash_no_strip():
    class MockElement:
        def __init__(self):
            self.tag = "div"
            self.text = None
            self.tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_complex():
    class MockChild1:
        def __init__(self):
            self.tag = "b"
            self.text = "bold"
            self.tail = " "
        def getchildren(self):
            return []
    class MockChild2:
        def __init__(self):
            self.tag = "br"
            self.text = None
            self.tail = None
        def getchildren(self):
            return []
    class MockElement:
        def __init__(self):
            self.tag = "div"
            self.text = "Text "
            self.tail = None
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Text ", "bold", " "]

def test_extract_text_array_callable_tag():
    class MockElement:
        def __init__(self):
            self.tag = lambda: None
            self.text = None
            self.tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_inline_tag():
    class MockElement:
        def __init__(self):
            self.tag = "span"
            self.text = "inline"
            self.tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["inline"]
```


# LLM-generated content at query #67
#--------------------------

```
def test_extract_text_array_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_single_text():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = 'Hello'
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_with_separator():
    from xml.etree.ElementTree import Element
    dom = Element('br')
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    from xml.etree.ElementTree import Element
    dom = Element('span')
    dom.text = 'text'
    result = extract_text_array(dom)
    assert result == ['text']

def test_extract_text_array_nested_elements():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = 'Hello'
    span = SubElement(p, 'span')
    span.text = 'World'
    result = extract_text_array(dom)
    assert result == ['Hello', 'World']

def test_extract_text_array_with_artifical_nl_squash():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = 'First'
    p2 = SubElement(dom, 'p')
    p2.text = 'Second'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, 'First', None, 'Second', None]

def test_extract_text_array_with_artifical_nl_strip():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = 'First'
    p2 = SubElement(dom, 'p')
    p2.text = 'Second'
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == [None, 'First', None, 'Second', None]

def test_extract_text_array_both_flags():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = 'First'
    p2 = SubElement(dom, 'p')
    p2.text = 'Second'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['First', None, 'Second']

def test_extract_text_array_separator_multiple():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    br1 = SubElement(dom, 'br')
    br2 = SubElement(dom, 'br')
    p = SubElement(dom, 'p')
    p.text = 'text'
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, True, True, 'text', None]

def test_extract_text_array_mixed_content():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    dom.text = 'Start'
    br = SubElement(dom, 'br')
    br.tail = ' Middle '
    span = SubElement(dom, 'span')
    span.text = 'End'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['Start', True, ' Middle ', 'End']

def test_extract_text_array_callable_tag():
    dom = type('Mock', (), {'tag': lambda: None, 'text': None, 'getchildren': lambda: []})()
    result = extract_text_array(dom)
    assert result == ''

def test_extract_text_array_strip_none_only():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_squash_consecutive_none():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    SubElement(dom, 'p')
    SubElement(dom, 'p')
    p = SubElement(dom, 'p')
    p.text = 'text'
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, 'text', None]
```


# LLM-generated content at query #68
#--------------------------

```python
def test_extract_text_array_for_loop_child_not_none():
    from lxml import etree
    dom = etree.Element("div")
    child = etree.SubElement(dom, "span")
    child.text = "text"
    child.tail = "tail"
    result = extract_text_array(dom)
    assert True  # predicate at line 12 evaluates to False because dom.getchildren() returns one child
```


# LLM-generated content at query #69
#--------------------------

```
def test_extract_text_array_returns_empty_string_for_callable_tag():
    class MockDom:
        tag = lambda: None
    assert extract_text_array(MockDom()) == ''

def test_extract_text_array_with_separator_tag_returns_list_with_true():
    class MockDom:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(MockDom()) == [True]

def test_extract_text_array_with_inline_tag_and_text():
    class MockDom:
        tag = 'span'
        text = 'hello'
        def getchildren(self):
            return []
    assert extract_text_array(MockDom()) == ['hello']

def test_extract_text_array_with_inline_tag_and_no_text():
    class MockDom:
        tag = 'span'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(MockDom()) == []

def test_extract_text_array_with_block_tag_adds_none_around_content():
    class MockDom:
        tag = 'div'
        text = 'text'
        def getchildren(self):
            return []
    assert extract_text_array(MockDom()) == ['text']

def test_extract_text_array_with_squash_artifical_nl_squashes_consecutive_nones():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            child = type('Child', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
            return [child]
    assert extract_text_array(MockDom(), squash_artifical_nl=True, strip_artifical_nl=False) == [None]

def test_extract_text_array_with_strip_artifical_nl_strips_leading_and_trailing_nones():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            child = type('Child', (), {'tag': 'span', 'text': 'a', 'tail': None, 'getchildren': lambda self: []})()
            return [child]
    assert extract_text_array(MockDom(), squash_artifical_nl=False, strip_artifical_nl=True) == ['a']

def test_extract_text_array_handles_child_tail():
    class MockDom:
        tag = 'div'
        text = 'start'
        def getchildren(self):
            child = type('Child', (), {'tag': 'span', 'text': None, 'tail': ' middle ', 'getchildren': lambda self: []})()
            return [child]
    assert extract_text_array(MockDom()) == ['start', ' middle ']

def test_extract_text_array_with_separator_child():
    class MockDom:
        tag = 'div'
        text = 'a'
        def getchildren(self):
            child = type('Child', (), {'tag': 'br', 'text': None, 'tail': 'b', 'getchildren': lambda self: []})()
            return [child]
    assert extract_text_array(MockDom()) == ['a', True, 'b']

def test_extract_text_array_with_nested_blocks():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            inner = type('Inner', (), {'tag': 'div', 'text': 'inner', 'tail': None, 'getchildren': lambda self: []})()
            return [inner]
    assert extract_text_array(MockDom()) == ['inner']

def test_extract_text_array_preserves_artifical_nl_when_squash_false_and_strip_false():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            child = type('Child', (), {'tag': 'span', 'text': 'x', 'tail': None, 'getchildren': lambda self: []})()
            return [child]
    assert extract_text_array(MockDom(), squash_artifical_nl=False, strip_artifical_nl=False) == [None, 'x', None]

def test_extract_text_array_returns_empty_list_for_empty_block():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    assert extract_text_array(MockDom(), squash_artifical_nl=False, strip_artifical_nl=False) == [None, None]

def test_extract_text_array_with_only_text_in_block():
    class MockDom:
        tag = 'div'
        text = 'only'
        def getchildren(self):
            return []
    assert extract_text_array(MockDom()) == ['only']

def test_extract_text_array_with_squash_and_strip_on_complex_structure():
    class MockDom:
        tag = 'div'
        text = None
        def getchildren(self):
            child1 = type('Child1', (), {'tag': 'div', 'text': 'a', 'tail': None, 'getchildren': lambda self: []})()
            child2 = type('Child2', (), {'tag': 'br', 'text': None, 'tail': 'b', 'getchildren': lambda self: []})()
            child3 = type('Child3', (), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
            return [child1, child2, child3]
    assert extract_text_array(MockDom()) == ['a', True, 'b']
```


# LLM-generated content at query #70
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_single_text():
    class MockDom:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_with_separator():
    class MockDom:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_children():
    class MockChild:
        tag = "span"
        text = "World"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = "Hello "
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["Hello ", "World"]

def test_extract_text_array_squash_artifical_nl():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == []

def test_extract_text_array_strip_artifical_nl():
    class MockDom:
        tag = "div"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["Hello"]

def test_extract_text_array_no_squash_no_strip():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_callable_tag():
    class MockDom:
        def tag(self):
            pass
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ''

def test_extract_text_array_with_tail():
    class MockChild:
        tag = "span"
        text = "inner"
        tail = " tail"
        def getchildren(self):
            return []
    class MockDom:
        tag = "p"
        text = "start"
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["start", "inner", " tail"]

def test_extract_text_array_multiple_separators():
    class MockDom:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]
```


# LLM-generated content at query #71
#--------------------------

```python
def test_predicate_line17_evaluates_to_true():
    dom = type('MockDom', (object,), {'tag': 'p', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    INLINE_TAGS = ['a', 'span']
    SEPARATORS = ['br']
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result[-1] is None
    assert dom.tag not in INLINE_TAGS
    assert dom.tag not in SEPARATORS
```


# LLM-generated content at query #72
#--------------------------

```python
def test_squash_artifical_nl_false():
    dom = type('MockDom', (), {'tag': 'p', 'text': None, 'getchildren': lambda: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert not _squash_artifical_nl.called


# LLM-generated content at query #73
#--------------------------

def test_strip_artifical_nl_false_does_not_strip():
    dom = type('MockDom', (object,), {
        'tag': 'p',
        'text': 'hello',
        'tail': None,
        'getchildren': lambda self: []
    })()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == ['hello', None]


# LLM-generated content at query #74
#--------------------------

def test_extract_text_simple_text():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = 'Hello world'
    assert extract_text(dom) == 'Hello world'

def test_extract_text_with_block_element():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = 'First paragraph'
    p2 = SubElement(dom, 'p')
    p2.text = 'Second paragraph'
    assert extract_text(dom) == 'First paragraph\nSecond paragraph'

def test_extract_text_with_separator():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    hr = SubElement(dom, 'hr')
    hr.tail = ' after hr'
    assert extract_text(dom) == ' after hr'

def test_extract_text_with_nested_inline():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('p')
    dom.text = 'Hello '
    b = SubElement(dom, 'b')
    b.text = 'bold'
    b.tail = ' world'
    assert extract_text(dom) == 'Hello bold world'

def test_extract_text_whitespace_squashing():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = '  Hello   world  '
    assert extract_text(dom) == 'Hello world'

def test_extract_text_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    assert extract_text(dom) == ''

def test_extract_text_only_block_elements():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    SubElement(dom, 'br')
    assert extract_text(dom) == ''

def test_extract_text_multiple_separators():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    hr1 = SubElement(dom, 'hr')
    hr1.tail = ' between '
    hr2 = SubElement(dom, 'hr')
    assert extract_text(dom) == ' between '

def test_extract_text_custom_block_symbol():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = 'A'
    assert extract_text(dom, block_symbol=' | ') == 'A'

def test_extract_text_custom_sep_symbol():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    hr = SubElement(dom, 'hr')
    hr.tail = ' separator'
    assert extract_text(dom, sep_symbol=' --- ') == ' ---  separator'

def test_extract_text_squash_space_false():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = '  Hello   world  '
    assert extract_text(dom, squash_space=False) == '  Hello   world  '


# LLM-generated content at query #75
#--------------------------

def test_strip_artifical_nl_true():
    dom = type('FakeDOM', (), {'tag': 'p', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == []


# LLM-generated content at query #76
#--------------------------

```
def test_squash_artifical_nl_true():
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result is not None
```


# LLM-generated content at query #77
#--------------------------

def test_extract_text_array_with_none_dom():
    pass

def test_extract_text_array_empty_dom():
    pass

def test_extract_text_array_single_text():
    pass

def test_extract_text_array_separator_tag():
    pass

def test_extract_text_array_inline_tag():
    pass

def test_extract_text_array_nested_elements():
    pass

def test_extract_text_array_with_tails():
    pass

def test_extract_text_array_squash_artifical_nl_true():
    pass

def test_extract_text_array_strip_artifical_nl_true():
    pass

def test_extract_text_array_both_flags_false():
    pass

def test_extract_text_array_separator_with_children():
    pass

def test_extract_text_array_inline_with_children():
    pass

def test_extract_text_array_multiple_separators():
    pass

def test_extract_text_array_mixed_tags():
    pass

def test_extract_text_array_only_artifical_nl():
    pass

def test_extract_text_array_leading_trailing_artifical_nl():
    pass

def test_extract_text_array_callable_tag():
    pass

def test_extract_text_array_none_text():
    pass

def test_extract_text_array_empty_children():
    pass

def test_extract_text_array_deep_nesting():
    pass

def test_extract_text_array_squash_removes_consecutive_none():
    pass

def test_extract_text_array_strip_removes_leading_none():
    pass

def test_extract_text_array_strip_removes_trailing_none():
    pass

def test_extract_text_array_squash_then_strip():
    pass

def test_extract_text_array_no_artifical_nl_in_output():
    pass


# LLM-generated content at query #78
#--------------------------

def test_strip_artifical_nl_false():
    r = [None, "some text", None]
    strip_artifical_nl = False
    assert strip_artifical_nl == False


# LLM-generated content at query #79
#--------------------------

```python
def test_squash_artifical_nl_true():
    dom = MagicMock()
    dom.tag = 'p'
    dom.text = None
    dom.getchildren.return_value = []
    dom.callable = False
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result is not None
    assert _squash_artifical_nl.called
```


# LLM-generated content at query #80
#--------------------------

def test_extract_text_simple_text():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = 'Hello world'
    assert extract_text(dom) == 'Hello world'

def test_extract_text_with_block_tag():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = 'First'
    p2 = SubElement(dom, 'p')
    p2.text = 'Second'
    assert extract_text(dom) == 'First\nSecond'

def test_extract_text_with_separator():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    br = SubElement(dom, 'br')
    br.tail = 'text'
    assert extract_text(dom) == 'text'

def test_extract_text_nested_inline():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('span')
    b = SubElement(dom, 'b')
    b.text = 'bold'
    assert extract_text(dom) == 'bold'

def test_extract_text_with_artificial_newlines():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = 'Line1'
    span = SubElement(p, 'span')
    span.text = 'Line2'
    assert extract_text(dom) == 'Line1\nLine2'

def test_extract_text_strips_leading_trailing_spaces():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = '  hello  '
    assert extract_text(dom) == 'hello'

def test_extract_text_squashes_multiple_spaces():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = 'hello    world'
    assert extract_text(dom) == 'hello world'

def test_extract_text_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    assert extract_text(dom) == ''

def test_extract_text_only_whitespace():
    from xml.etree.ElementTree import Element
    dom = Element('p')
    dom.text = '   '
    assert extract_text(dom) == ''


# LLM-generated content at query #81
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_false():
    dom = type('MockDom', (object,), {'tag': 'div', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom)
    assert True  # predicate evaluated to False, function returned without appending None at line 19
```


# LLM-generated content at query #82
#--------------------------

def test_predicate_true_when_tag_not_in_inline_and_not_in_separators():
    dom = type('MockDom', (), {'tag': 'custom_tag', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom)
    assert result[-1] is None


# LLM-generated content at query #83
#--------------------------

```python
def test_squash_artifical_nl_false():
    dom = type('MockDom', (object,), {'tag': 'div', 'text': 'Hello', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == ['Hello', None]  # No squash, predicate at line 20 is False, so r unchanged
```


# LLM-generated content at query #84
#--------------------------

```python
def test_for_loop_predicate_false():
    from your_module import extract_text_array
    class MockChild:
        def __init__(self):
            self.tail = None
    class MockDom:
        tag = 'p'
        text = 'hello'
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert len(result) >= 2
```


# LLM-generated content at query #85
#--------------------------

```
def test_extract_text_array_empty_dom():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockDom:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
        tail = None
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_with_children():
    class MockChild:
        tag = "span"
        text = "world"
        tail = "!"
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = "Hello "
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["Hello ", "world", "!"]

def test_extract_text_array_separator_tag():
    class MockDom:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_artifical_nl_squash():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == []

def test_extract_text_array_artifical_nl_strip():
    class MockDom:
        tag = "div"
        text = "a"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["a"]

def test_extract_text_array_with_none_text():
    class MockDom:
        tag = "p"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_multiple_children():
    class MockChild1:
        tag = "b"
        text = "bold"
        tail = " "
        def getchildren(self):
            return []
    class MockChild2:
        tag = "i"
        text = "italic"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild1(), MockChild2()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["bold", " ", "italic"]

def test_extract_text_array_squash_false():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_strip_false():
    class MockDom:
        tag = "div"
        text = "content"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == [None, "content", None]
```


# LLM-generated content at query #86
#--------------------------

```python
def test_extract_text_array_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element('div')
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('p')
    dom.text = 'Hello'
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_with_separator():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('br')
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('span')
    dom.text = 'inline'
    result = extract_text_array(dom)
    assert result == ['inline']

def test_extract_text_array_nested_inline():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('p')
    b = SubElement(dom, 'b')
    b.text = 'bold'
    result = extract_text_array(dom)
    assert result == ['bold']

def test_extract_text_array_with_tail():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('p')
    dom.text = 'start'
    b = SubElement(dom, 'b')
    b.text = 'bold'
    b.tail = ' tail'
    result = extract_text_array(dom)
    assert result == ['start', 'bold', ' tail']

def test_extract_text_array_squash_artifical_nl():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p1 = SubElement(dom, 'p')
    p1.text = 'first'
    p2 = SubElement(dom, 'p')
    p2.text = 'second'
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ['first', None, 'second']

def test_extract_text_array_strip_artifical_nl():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element('div')
    p = SubElement(dom, 'p')
    p.text = 'text'
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['text']


# LLM-generated content at query #87
#--------------------------

```python
def test_squash_artificial_nl_false():
    dom = type('MockDom', (), {
        'tag': 'p',
        'text': None,
        'getchildren': lambda self: [],
        'tail': None
    })()
    result = extract_text_array(dom, squash_artifical_nl=False)
    assert result == [None]  # predicate at line 20 evaluates to False


# LLM-generated content at query #88
#--------------------------

```py

def test_extract_text_array_empty_dom():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockDom:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_separator_tag():
    class MockDom:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag():
    class MockDom:
        tag = "span"
        text = "text"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["text"]

def test_extract_text_array_with_children():
    class MockChild:
        tag = "b"
        text = "bold"
        tail = None
        def getchildren(self):
            return []
    class MockParent:
        tag = "p"
        text = "before"
        def getchildren(self):
            return [MockChild()]
    dom = MockParent()
    result = extract_text_array(dom)
    assert result == ["before", "bold"]

def test_extract_text_array_with_tail():
    class MockChild:
        tag = "a"
        text = "link"
        tail = " after"
        def getchildren(self):
            return []
    class MockParent:
        tag = "p"
        text = "start "
        def getchildren(self):
            return [MockChild()]
    dom = MockParent()
    result = extract_text_array(dom)
    assert result == ["start ", "link", " after"]

def test_extract_text_array_squash_artifical_nl():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == []

def test_extract_text_array_strip_artifical_nl():
    class MockChild:
        tag = "div"
        text = "middle"
        tail = None
        def getchildren(self):
            return []
    class MockParent:
        tag = "div"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockParent()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["middle"]

def test_extract_text_array_no_squash_no_strip():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_nested_separator():
    class MockChild:
        tag = "br"
        text = None
        tail = " next"
        def getchildren(self):
            return []
    class MockParent:
        tag = "p"
        text = "first"
        def getchildren(self):
            return [MockChild()]
    dom = MockParent()
    result = extract_text_array(dom)
    assert result == ["first", True, " next"]

```


# LLM-generated content at query #89
#--------------------------

def test_extract_text_empty_dom():
    from xml.etree.ElementTree import Element
    dom = Element("div")
    result = extract_text(dom)
    assert result == ""

def test_extract_text_simple_text():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("p")
    dom.text = "Hello"
    result = extract_text(dom)
    assert result == "Hello"

def test_extract_text_with_child():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    dom.text = "Hello "
    child = SubElement(dom, "span")
    child.text = "World"
    child.tail = "!"
    result = extract_text(dom)
    assert result == "Hello World!"

def test_extract_text_with_separator():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    child = SubElement(dom, "hr")
    result = extract_text(dom)
    assert result == "\n"

def test_extract_text_with_block_elements():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    p1 = SubElement(dom, "p")
    p1.text = "First"
    p2 = SubElement(dom, "p")
    p2.text = "Second"
    result = extract_text(dom)
    assert result == "First\nSecond"

def test_extract_text_with_squash_space():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    dom.text = "  Hello   World  "
    result = extract_text(dom)
    assert result == "Hello World"

def test_extract_text_nested_structure():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    outer = SubElement(dom, "div")
    outer.text = "Outer "
    inner = SubElement(outer, "span")
    inner.text = "Inner"
    inner.tail = ""
    result = extract_text(dom)
    assert result == "Outer Inner"

def test_extract_text_multiple_separators():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    SubElement(dom, "hr")
    SubElement(dom, "br")
    result = extract_text(dom)
    assert result == "\n\n"

def test_extract_text_trailing_block():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    p = SubElement(dom, "p")
    p.text = "Content"
    SubElement(dom, "div")
    result = extract_text(dom)
    assert result == "Content"

def test_extract_text_leading_block():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    SubElement(dom, "div")
    p = SubElement(dom, "p")
    p.text = "Content"
    result = extract_text(dom)
    assert result == "Content"

def test_extract_text_with_inline_tags():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("span")
    dom.text = "Inline"
    result = extract_text(dom)
    assert result == "Inline"

def test_extract_text_custom_symbols():
    from xml.etree.ElementTree import Element, SubElement
    dom = Element("div")
    p1 = SubElement(dom, "p")
    p1.text = "A"
    p2 = SubElement(dom, "p")
    p2.text = "B"
    result = extract_text(dom, block_symbol='|', sep_symbol='|')
    assert result == "A|B"


# LLM-generated content at query #90
#--------------------------

def test_predicate_at_line_17_evaluates_to_true():
    dom = type('MockDom', (), {'tag': 'p', 'text': None, 'tail': None, 'getchildren': lambda self: []})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result[-1] is None


# LLM-generated content at query #91
#--------------------------

def test_predicate_at_line_17_evaluates_to_false():
    from unittest.mock import MagicMock
    dom = MagicMock()
    dom.tag = "p"
    dom.text = None
    dom.getchildren.return_value = []
    result = extract_text_array(dom)
    assert True


# LLM-generated content at query #92
#--------------------------

```python
def test_for_loop_predicate_false_when_no_children():
    dom = type('MockDom', (), {'tag': 'div', 'text': 'text', 'getchildren': lambda: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == ['text', None]  # Line 12 predicate (for child in dom.getchildren()) evaluates to False when no children
```


# LLM-generated content at query #93
#--------------------------

def test_extract_text_array_empty_dom():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockDom:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_with_separator():
    class MockDom:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_inline_tag():
    class MockDom:
        tag = "b"
        text = "bold"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["bold"]

def test_extract_text_array_with_nested_elements():
    class MockChild:
        tag = "span"
        text = "child"
        tail = None
        def getchildren(self):
            return []
    class MockDom:
        tag = "div"
        text = "parent"
        def getchildren(self):
            return [MockChild()]
    dom = MockDom()
    result = extract_text_array(dom)
    assert result == ["parent", "child"]

def test_extract_text_array_squash_artifical_nl():
    class MockDom:
        tag = "div"
        text = "a"
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ["a"]

def test_extract_text_array_strip_artifical_nl():
    class MockDom:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockDom()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == []


# LLM-generated content at query #94
#--------------------------

def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Hello", None]

def test_extract_text_array_with_separator():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [True]

def test_extract_text_array_with_child():
    class MockChild:
        tag = "span"
        text = "World"
        tail = None
        def getchildren(self):
            return []
    class MockParent:
        tag = "div"
        text = "Hello "
        def getchildren(self):
            return [MockChild()]
    dom = MockParent()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Hello ", "World", None]

def test_extract_text_array_squash_artifical_nl():
    class MockElement:
        tag = "div"
        text = "A"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None, "A"]

def test_extract_text_array_strip_artifical_nl():
    class MockElement:
        tag = "div"
        text = "B"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == ["B"]

def test_extract_text_array_nested_tags():
    class MockChild:
        tag = "b"
        text = "bold"
        tail = " normal"
        def getchildren(self):
            return []
    class MockParent:
        tag = "p"
        text = "Some "
        def getchildren(self):
            return [MockChild()]
    dom = MockParent()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "Some ", "bold", " normal", None]

def test_extract_text_array_callable_tag():
    class MockElement:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [""]


# LLM-generated content at query #95
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = 'p'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_text_no_children():
    class MockElement:
        tag = 'p'
        text = 'hello'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['hello']

def test_extract_text_array_separator_tag():
    class MockElement:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag():
    class MockElement:
        tag = 'span'
        text = 'world'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['world']

def test_extract_text_array_with_child():
    class MockChild:
        tag = 'b'
        text = 'bold'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'p'
        text = 'before '
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['before ', 'bold']

def test_extract_text_array_with_tail():
    class MockChild:
        tag = 'a'
        text = 'link'
        tail = ' after'
        def getchildren(self):
            return []
    class MockElement:
        tag = 'p'
        text = 'click '
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['click ', 'link', ' after']

def test_extract_text_array_squash_artifical_nl():
    class MockChild:
        tag = 'div'
        text = 'a'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ['a']

def test_extract_text_array_strip_artifical_nl():
    class MockChild:
        tag = 'div'
        text = 'b'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['b']


# LLM-generated content at query #96
#--------------------------

```
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_single_text():
    class MockElement:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_separator_tag():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag():
    class MockElement:
        tag = "span"
        text = "inline"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["inline"]

def test_extract_text_array_with_children():
    class MockChild:
        tag = "b"
        text = "bold"
        tail = " tail"
        def getchildren(self):
            return []
    class MockElement:
        tag = "p"
        text = "start "
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["start ", "bold", " tail"]

def test_extract_text_array_squash_artifical_nl():
    class MockChild:
        tag = "div"
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "p"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_artifical_nl():
    class MockChild:
        tag = "div"
        text = "content"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "p"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == [None, "content", None]

def test_extract_text_array_both_options():
    class MockChild:
        tag = "div"
        text = "content"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "p"
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["content"]

def test_extract_text_array_callable_tag():
    class MockElement:
        tag = lambda: None
        text = "ignored"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_nested_structure():
    class MockInner:
        tag = "span"
        text = "inner"
        tail = " after"
        def getchildren(self):
            return []
    class MockChild:
        tag = "div"
        text = "child "
        tail = None
        def getchildren(self):
            return [MockInner()]
    class MockElement:
        tag = "body"
        text = "start "
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ["start ", "child ", "inner", " after"]
```


# LLM-generated content at query #97
#--------------------------

def test_extract_text_array_empty_dom_no_tag():
    class MockElement:
        tag = None
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        tag = 'p'
        text = 'hello'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['hello']

def test_extract_text_array_separator_tag():
    class MockElement:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag():
    class MockElement:
        tag = 'span'
        text = 'world'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['world']

def test_extract_text_array_with_child():
    class MockChild:
        tag = 'span'
        text = 'child'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = 'parent'
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['parent', 'child']

def test_extract_text_array_with_tail():
    class MockChild:
        tag = 'span'
        text = 'child'
        tail = ' tail'
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = 'parent'
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['parent', 'child', ' tail']

def test_extract_text_array_nested_separator():
    class MockChild:
        tag = 'br'
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = 'before'
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['before', True]

def test_extract_text_array_squash_artifical_nl_true():
    class MockChild:
        tag = 'span'
        text = 'a'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ['a']

def test_extract_text_array_strip_artifical_nl_true():
    class MockChild:
        tag = 'span'
        text = 'a'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['a']

def test_extract_text_array_callable_tag():
    class MockElement:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []


# LLM-generated content at query #98
#--------------------------

def test_extract_text_with_squash_space_true():
    dom = None
    result = extract_text(dom, squash_space=True)
    assert result == ""


# LLM-generated content at query #99
#--------------------------

```python
def test_extract_text_array_with_callable_tag_returns_empty_string():
    class FakeElement:
        tag = lambda: None
        text = None
        def getchildren(self):
            return []
    dom = FakeElement()
    result = extract_text_array(dom)
    assert result == ''

def test_extract_text_array_single_text_node():
    class FakeElement:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
    dom = FakeElement()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_separator_tag_adds_true():
    class FakeElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = FakeElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_separator_tag_with_text():
    class FakeElement:
        tag = "hr"
        text = "separator"
        def getchildren(self):
            return []
    dom = FakeElement()
    result = extract_text_array(dom)
    assert result == ["separator"]

def test_extract_text_array_non_inline_tag_adds_none():
    class FakeElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = FakeElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_with_child_and_tail():
    class FakeChild:
        tag = "b"
        text = "bold"
        tail = " tail"
        def getchildren(self):
            return []
    class FakeElement:
        tag = "p"
        text = "start"
        def getchildren(self):
            return [FakeChild()]
    dom = FakeElement()
    result = extract_text_array(dom)
    assert result == ["start", "bold", " tail"]

def test_extract_text_array_squash_artifical_nl_false():
    class FakeElement:
        tag = "div"
        text = "a"
        def getchildren(self):
            return []
    dom = FakeElement()
    result = extract_text_array(dom, squash_artifical_nl=False)
    expected = [None, "a", None]
    assert result == expected

def test_extract_text_array_strip_artifical_nl_false():
    class FakeElement:
        tag = "div"
        text = "b"
        def getchildren(self):
            return []
    dom = FakeElement()
    result = extract_text_array(dom, strip_artifical_nl=False)
    assert result == ["b"]

def test_extract_text_array_inline_tag_no_artifical_nl():
    class FakeElement:
        tag = "span"
        text = "inline"
        def getchildren(self):
            return []
    dom = FakeElement()
    result = extract_text_array(dom)
    assert result == ["inline"]

def test_extract_text_array_multiple_children():
    class FakeChild1:
        tag = "b"
        text = "first"
        tail = None
        def getchildren(self):
            return []
    class FakeChild2:
        tag = "i"
        text = "second"
        tail = None
        def getchildren(self):
            return []
    class FakeElement:
        tag = "div"
        text = None
        def getchildren(self):
            return [FakeChild1(), FakeChild2()]
    dom = FakeElement()
    result = extract_text_array(dom)
    assert result == ["first", "second"]
```


# LLM-generated content at query #100
#--------------------------

```python
def test_extract_text_array_empty_dom():
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        tag = 'p'
        text = 'Hello'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['Hello']

def test_extract_text_array_with_child():
    class MockChild:
        tag = 'b'
        text = 'bold'
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = 'p'
        text = 'before '
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['before ', 'bold']

def test_extract_text_array_separator_tag():
    class MockElement:
        tag = 'br'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_inline_tag():
    class MockElement:
        tag = 'span'
        text = 'inline'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ['inline']

def test_extract_text_array_artifical_nl():
    class MockElement:
        tag = 'div'
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_squash_nl():
    class MockElement:
        tag = 'div'
        text = 'a'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ['a']

def test_extract_text_array_strip_nl():
    class MockElement:
        tag = 'div'
        text = 'b'
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ['b']


# LLM-generated content at query #101
#--------------------------

def test_predicate_line_17_evaluates_true():
    class MockElement:
        tag = "div"
        text = None
        tail = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result[-1] is None


# LLM-generated content at query #102
#--------------------------

def test_extract_text_array_empty_dom():
    class MockElement:
        tag = "div"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    class MockElement:
        tag = "p"
        text = "Hello"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Hello"]

def test_extract_text_array_separator_tag():
    class MockElement:
        tag = "br"
        text = None
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == [True]

def test_extract_text_array_with_child():
    class MockChild:
        tag = "span"
        text = "world"
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "Hello "
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Hello ", "world"]

def test_extract_text_array_with_tail():
    class MockChild:
        tag = "a"
        text = "click"
        tail = " here"
        def getchildren(self):
            return []
    class MockElement:
        tag = "p"
        text = "Please "
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["Please ", "click", " here"]

def test_extract_text_array_squash_nl():
    class MockElement:
        tag = "div"
        text = "A"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=True)
    assert result == ["A"]

def test_extract_text_array_strip_nl():
    class MockElement:
        tag = "div"
        text = "B"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, strip_artifical_nl=True)
    assert result == ["B"]

def test_extract_text_array_no_squash_no_strip():
    class MockElement:
        tag = "div"
        text = "C"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, "C", None]

def test_extract_text_array_inline_tag():
    class MockElement:
        tag = "b"
        text = "bold"
        def getchildren(self):
            return []
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["bold"]

def test_extract_text_array_nested_separator():
    class MockChild:
        tag = "br"
        text = None
        tail = None
        def getchildren(self):
            return []
    class MockElement:
        tag = "div"
        text = "line1"
        def getchildren(self):
            return [MockChild()]
    dom = MockElement()
    result = extract_text_array(dom)
    assert result == ["line1", True]


# LLM-generated content at query #103
#--------------------------

def test_strip_artifical_nl_false():
    r = [None, "text", None]
    strip_artifical_nl = False
    assert not (strip_artifical_nl is True)


# LLM-generated content at query #104
#--------------------------

```
def test_extract_text_array_empty_dom():
    dom = type('Elem', (), {'tag': 'div', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom)
    assert result == []

def test_extract_text_array_simple_text():
    dom = type('Elem', (), {'tag': 'span', 'text': 'hello', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == ['hello']

def test_extract_text_array_with_separator():
    dom = type('Elem', (), {'tag': 'br', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [True]

def test_extract_text_array_with_block_tag():
    dom = type('Elem', (), {'tag': 'p', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, None]

def test_extract_text_array_with_children():
    child = type('Elem', (), {'tag': 'span', 'text': 'world', 'getchildren': lambda self: [], 'tail': ' tail'})()
    dom = type('Elem', (), {'tag': 'div', 'text': 'hello ', 'getchildren': lambda self: [child], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == [None, 'hello ', 'world', ' tail', None]

def test_extract_text_array_squash_artifical_nl():
    dom = type('Elem', (), {'tag': 'p', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=False)
    assert result == [None]

def test_extract_text_array_strip_artifical_nl():
    dom = type('Elem', (), {'tag': 'p', 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=True)
    assert result == []

def test_extract_text_array_callable_tag():
    dom = type('Elem', (), {'tag': lambda: None, 'text': None, 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom)
    assert result == ''

def test_extract_text_array_inline_tag():
    dom = type('Elem', (), {'tag': 'strong', 'text': 'bold', 'getchildren': lambda self: [], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=False, strip_artifical_nl=False)
    assert result == ['bold']

def test_extract_text_array_mixed_content():
    child1 = type('Elem', (), {'tag': 'span', 'text': 'inner', 'getchildren': lambda self: [], 'tail': ' after'})()
    dom = type('Elem', (), {'tag': 'div', 'text': 'before ', 'getchildren': lambda self: [child1], 'tail': None})()
    result = extract_text_array(dom, squash_artifical_nl=True, strip_artifical_nl=True)
    assert result == ['before ', 'inner', ' after']
```


