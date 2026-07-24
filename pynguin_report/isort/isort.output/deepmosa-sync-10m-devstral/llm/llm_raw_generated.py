####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_empty_output():
    assert _ensure_newline_before_comment([]) == []

def test_single_comment_line():
    assert _ensure_newline_before_comment(["# comment"]) == ["# comment"]

def test_single_non_comment_line():
    assert _ensure_newline_before_comment(["code"]) == ["code"]

def test_comment_after_code():
    assert _ensure_newline_before_comment(["code", "# comment"]) == ["code", "", "# comment"]

def test_comment_after_empty_line():
    assert _ensure_newline_before_comment(["", "# comment"]) == ["", "# comment"]

def test_comment_after_comment():
    assert _ensure_newline_before_comment(["# comment1", "# comment2"]) == ["# comment1", "# comment2"]

def test_multiple_comments_with_code():
    assert _ensure_newline_before_comment(["code1", "# comment1", "code2", "# comment2"]) == ["code1", "", "# comment1", "code2", "", "# comment2"]

def test_no_newline_needed():
    assert _ensure_newline_before_comment(["# comment1", "", "code", "# comment2"]) == ["# comment1", "", "code", "", "# comment2"]

def test_none_line_handling():
    assert _ensure_newline_before_comment([None, "# comment"]) == [None, "", "# comment"]


# LLM-generated content at query #2
#--------------------------

```python
def test_with_straight_imports_combine_straight_imports_no_as_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"os": [], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["os", "sys"], "standard", [], "import")
    assert result == ["import os, sys"]

def test_with_straight_imports_combine_straight_imports_with_inline_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"os": [], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["comment1"], "sys": ["comment2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["os", "sys"], "standard", [], "import")
    assert result == ["import os, sys  # comment1 comment2"]

def test_with_straight_imports_combine_straight_imports_with_above_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"os": [], "sys": []}}},
        categorized_comments={"above": {"straight": {"os": ["# above comment"]}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["os", "sys"], "standard", [], "import")
    assert result == ["# above comment", "import os, sys"]

def test_with_straight_imports_combine_straight_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["os_path"]}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["os"], "standard", [], "import")
    assert result == ["import os as os_path"]

def test_with_straight_imports_no_combine_straight_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"os": [], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["os", "sys"], "standard", [], "import")
    assert result == ["import os", "import sys"]

def test_with_straight_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"os": [], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["os", "sys"], "standard", ["sys"], "import")
    assert result == ["import os"]

def test_with_straight_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["comment"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False, ignore_comments=True)
    result = _with_straight_imports(parsed, config, ["os"], "standard", [], "import")
    assert result == ["import os"]

def test_with_straight_imports_custom_comment_prefix():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["comment"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False, comment_prefix=" # ")
    result = _with_straight_imports(parsed, config, ["os"], "standard", [], "import")
    assert result == ["import os # comment"]


# LLM-generated content at query #3
#--------------------------

```python
def test_output_as_string_with_empty_lines():
    lines = ["Hello", "", "World", ""]
    line_separator = "\n"
    result = _output_as_string(lines, line_separator)
    assert result == "Hello\n\nWorld\n"

def test_output_as_string_without_empty_lines():
    lines = ["Hello", "World"]
    line_separator = "\n"
    result = _output_as_string(lines, line_separator)
    assert result == "Hello\nWorld\n"

def test_output_as_string_with_only_empty_lines():
    lines = ["", "", ""]
    line_separator = "\n"
    result = _output_as_string(lines, line_separator)
    assert result == "\n"

def test_output_as_string_with_custom_separator():
    lines = ["Hello", "World"]
    line_separator = " | "
    result = _output_as_string(lines, line_separator)
    assert result == "Hello | World | "


# LLM-generated content at query #4
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment1", "comment2"]}},
        as_map={"from": {"os.path": ["os.path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path, sys  # comment1; comment2", "from os import os.path as ospath"]


# LLM-generated content at query #5
#--------------------------

```python
def test_with_star_comments_returns_comments_with_star_comment():
    parsed = parse.ParsedContent(categorized_comments={"nested": {module: {"*": "star_comment"}}})
    module = "test_module"
    comments = ["comment1", "comment2"]
    assert _with_star_comments(parsed, module, comments) == ["comment1", "comment2", "star_comment"]

def test_with_star_comments_returns_comments_without_star_comment():
    parsed = parse.ParsedContent(categorized_comments={"nested": {module: {}}})
    module = "test_module"
    comments = ["comment1", "comment2"]
    assert _with_star_comments(parsed, module, comments) == ["comment1", "comment2"]

def test_with_star_comments_returns_comments_with_empty_nested():
    parsed = parse.ParsedContent(categorized_comments={"nested": {}})
    module = "test_module"
    comments = ["comment1", "comment2"]
    assert _with_star_comments(parsed, module, comments) == ["comment1", "comment2"]


# LLM-generated content at query #6
#--------------------------

```python
def test_with_from_imports_basic_case():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {"os": ("# comment",)},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False, comment_prefix="# ")
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # comment"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", ["os.sys"], "import")
    assert result == ["from os import path"]

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path as ospath"]

def test_with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"os": {"*": "# star comment"}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path", "from os import sys"]

def test_with_from_imports_combine_as_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*", "path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True, combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *", "from os import path as ospath"]

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {"os": ("# comment",)},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]

def test_with_from_imports_with_above_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {"os": ["# above comment"]}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["# above comment", "from os import path"]

def test_with_from_imports_with_nested_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"os": {"path": "# nested comment"}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # nested comment"]


# LLM-generated content at query #7
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        imports={},
        categorized_comments={},
        as_map={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "print('hello')"

def test_sorted_imports_single_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os\n"

def test_sorted_imports_multiple_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"os": ["# comment"]},
        },
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os  # comment\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": {"path"}}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os as path\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(remove_imports=["os"])
    assert sorted_imports(parsed, config) == "import sys\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(combine_straight_imports=True)
    assert sorted_imports(parsed, config) == "import os, sys\n"

def test_sorted_imports_with_no_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "FUTURE": {"straight": {"__future__": set()}, "from": {}},
            "STDLIB": {"straight": {"os": set()}, "from": {}},
        },
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["FUTURE", "STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(no_sections=True)
    assert sorted_imports(parsed, config) == "from __future__ import absolute_import\nimport os\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(force_sort_within_sections=True)
    assert sorted_imports(parsed, config) == "import os\nimport sys\n"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "FUTURE": {"straight": {"__future__": set()}, "from": {}},
            "STDLIB": {"straight": {"os": set()}, "from": {}},
        },
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["FUTURE", "STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(lines_between_sections=2)
    assert sorted_imports(parsed, config) == "from __future__ import absolute_import\n\n\nimport os\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(import_headings={"stdlib": "Standard Library"})
    assert sorted_imports(parsed, config) == "# Standard Library\nimport os\n"

def test_sorted_imports_with_ensure_newline_before_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(ensure_newline_before_comments=True)
    assert sorted_imports(parsed, config) == "import os\n"

def test_sorted_imports_with_formatting_function():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(formatting_function=lambda x, y, z: x.upper())
    assert sorted_imports(parsed, config) == "IMPORT OS\n"

def test_sorted_imports_with_lines_before_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator


# LLM-generated content at query #8
#--------------------------

```python
def test_star_comment_is_none():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {"module": {}}}
    module = "module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == comments


# LLM-generated content at query #9
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", ["module.import1"], "import")
    assert result == ["from module import import2"]

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={
            "from": {"module": ["comment1", "comment2"]},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False, comment_prefix="# ")
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2 # comment1; comment2"]

def test_with_from_imports_combine_as_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"module.import1": ["import1 as alias1"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1 as alias1, import2"]

def test_with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["*", "import1"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"module": {"*": ["star comment"]}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import * # star comment", "from module import import1"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1", "from module import import2"]

def test_with_from_imports_no_inline_sort():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import2", "import1"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(no_inline_sort=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import2, import1"]

def test_with_from_imports_only_sections():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(only_sections=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2"]


# LLM-generated content at query #10
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        imports={},
        categorized_comments={},
        as_map={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

def test_sorted_imports_single_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\n"

def test_sorted_imports_multiple_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# comment above"]}}, "straight": {"os": ["# inline comment"]}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "# comment above\nimport os  # inline comment\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["os_path"]}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os as os_path\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_no_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}, "THIRDPARTY": {"straight": {"django": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB", "THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import django\nimport os\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"sys": set(), "os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(import_headings={"stdlib": "Standard Library Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Standard Library Imports\nimport os\n"


# LLM-generated content at query #11
#--------------------------

```python
def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        import_index=-1,
        lines_without_imports=["print('hello')"],
        line_separator="\n",
        imports={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')"


# LLM-generated content at query #12
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2"]

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={
            "from": {"module": ["comment1", "comment2"]},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False, comment_prefix="#")
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2  # comment1; comment2"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", ["module.import1"], "import")
    assert result == ["from module import import2"]

def test_with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "*"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"module": {"*": ["star_comment"]}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import *  # star_comment", "from module import import1"]

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"module.import1": ["import1 as alias1"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1 as alias1"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1", "from module import import2"]

def test_with_from_imports_force_grid_wrap():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2", "import3"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_grid_wrap=2, line_length=20)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == [
        "from module import (\n    import1,\n    import2,\n    import3,\n)"
    ]

def test_with_from_imports_split_on_trailing_comma():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas={"module"},
    )
    config = Config(split_on_trailing_comma=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == [
        "from module import (\n    import1,\n    import2,\n)"
    ]


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) or config.only_sections


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_evaluates_to_false():
    parsed = parse.ParsedContent()
    config = Config()
    from_modules = []
    section = ""
    remove_imports = []
    import_type = ""

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert result == []


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_1():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_evaluates_to_false():
    config = Config()
    config.no_inline_sort = True
    config.force_single_line = False
    config.only_sections = True

    assert not (
        not config.no_inline_sort
        or (config.force_single_line and "module" not in config.single_line_exclusions)
    )


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_1():
    parsed = parse.ParsedContent()
    config = Config()
    from_modules = ["module1"]
    section = "section1"
    remove_imports = []
    import_type = "import"

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert result == []


# LLM-generated content at query #18
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path, sys"]


# LLM-generated content at query #19
#--------------------------

```python
def test_sorted_imports_predicate_false():
    parsed = parse.ParsedContent(
        import_index=-1,
        lines_without_imports=["print('Hello, World!')"],
        line_separator="\n",
        imports={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1
    )
    config = DEFAULT_CONFIG
    extension = "py"
    import_type = "import"
    assert sorted_imports(parsed, config, extension, import_type) == "print('Hello, World!')"


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) or config.only_sections


# LLM-generated content at query #21
#--------------------------

```python
def test_with_straight_imports_predicate_false():
    parsed = parse.ParsedContent(
        as_map={"straight": {"module1": ["alias1"]}},
        imports={"STDLIB": {"straight": {"module1": ["import1"]}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config(combine_straight_imports=True)
    straight_modules = ["module1"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == []


# LLM-generated content at query #22
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="standard",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path"]

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {"os": ["# comment"]},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="standard",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path  # comment"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="standard",
        remove_imports=["os.sys"],
        import_type="import",
    )
    assert result == ["from os import path"]

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="standard",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path", "from os import path as ospath"]

def test_with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["*"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {"os": {"*": ["# all"]}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="standard",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import *  # all"]

def test_with_from_imports_with_above_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {"os": ["# above comment"]}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="standard",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["# above comment", "from os import path"]

def test_with_from_imports_with_nested_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"os": {"path": ["# nested comment"]}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="standard",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path  # nested comment"]

def test_with_from_imports_with_force_single_line():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="standard",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path", "from os import sys"]

def test_with_from_imports_with_combine_as_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="standard",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path as ospath"]

def test_with_from_imports_with_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {"os": ["# comment"]},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="standard",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path"]


# LLM-generated content at query #23
#--------------------------

```python
def test_as_imports_predicate_with_straight_as_imports():
    parsed = parse.ParsedContent(
        as_map={"straight": {"module1": ["alias1"], "module2": ["alias2"]}},
        imports={"section": {"straight": {"module1": [], "module2": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config()
    straight_modules = ["module1", "module2"]
    section = "section"
    remove_imports = []
    import_type = "import"

    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == []


# LLM-generated content at query #24
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        imports={},
        categorized_comments={},
        as_map={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {"os": ["# comment above"]}}, "straight": {"os": ["# inline comment"]}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "# comment above\nimport os  # inline comment\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["os_module"]}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os as os_module\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"from": {}}, "from": {"os": {}}},
        as_map={"from": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from os import path\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"FUTURE": {"straight": {"__future__": set()}}, "THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["FUTURE", "THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "import __future__\n\n\nimport os\n"


# LLM-generated content at query #25
#--------------------------

```python
def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
        sections=[],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"


# LLM-generated content at query #26
#--------------------------

```python
def test_as_imports_predicate():
    parsed = parse.ParsedContent(
        as_map={"straight": {"module1": ["alias1"], "module2": []}},
        imports={"section": {"straight": {"module1": [], "module2": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    straight_modules = ["module1", "module2"]
    assert any(module in parsed.as_map["straight"] for module in straight_modules)


# LLM-generated content at query #27
#--------------------------

```python
def test_sorted_imports_empty_parsed_content():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        imports={},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=[],
        place_in_section={},
    )
    result = sorted_imports(parsed)
    assert result == "\n"

def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        imports={},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=[],
        place_in_section={},
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')\n"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": [("import os", "os")]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        place_in_section={},
    )
    result = sorted_imports(parsed)
    assert result == "import os\n\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": [("import os", "os")], "sys": [("import sys", "sys")]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        place_in_section={},
    )
    result = sorted_imports(parsed)
    assert result == "import os\nimport sys\n\n"

def test_sorted_imports_with_as_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": [("import os", "os")]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {"os": ["os_path"]}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        place_in_section={},
    )
    result = sorted_imports(parsed)
    assert result == "import os\nimport os as os_path\n\n"

def test_sorted_imports_with_from_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {}, "from": {"os": [("from os import path", "os")]}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        place_in_section={},
    )
    result = sorted_imports(parsed)
    assert result == "from os import path\n\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": [("import os", "os")]}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# OS module"]}, "from": {}}, "straight": {"os": ["# For path operations"]}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        place_in_section={},
    )
    result = sorted_imports(parsed)
    assert result == "# OS module\nimport os  # For path operations\n\n"

def test_sorted_imports_combine_straight_imports():
    config = Config(combine_straight_imports=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": [("import os", "os")], "sys": [("import sys", "sys")]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        place_in_section={},
    )
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n\n"

def test_sorted_imports_with_section_heading():
    config = Config(import_headings={"stdlib": "Standard Library"})
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": [("import os", "os")]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        place_in_section={},
    )
    result = sorted_imports(parsed, config)
    assert result == "# Standard Library\nimport os\n\n"

def test_sorted_imports_with_remove_imports():
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": [("import os", "os")], "sys": [("import sys", "sys")]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        place_in_section={},
    )
    result = sorted_imports(parsed, config)
    assert result == "import sys\n\n"

def test_sorted_imports_with_force_sort_within_sections():
    config = Config(force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"sys": [("import sys", "sys")], "os": [("import os", "os")]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        place_in_section={},
    )
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\n"


# LLM-generated content at query #28
#--------------------------

```python
def test_with_straight_imports_predicate_false():
    parsed = parse.ParsedContent(
        as_map={"straight": {"module1": ["alias1"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        imports={"section": {"straight": {"module1": []}}}
    )
    config = Config(combine_straight_imports=True)
    straight_modules = ["module1"]
    section = "section"
    remove_imports = []
    import_type = "import"

    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert not any(module in parsed.as_map["straight"] for module in straight_modules)


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_evaluates_to_false():
    parsed = parse.ParsedContent()
    config = Config()
    from_modules = []
    section = ""
    remove_imports = []
    import_type = ""

    assert not _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_evaluates_to_false():
    config = Config(
        no_inline_sort=True,
        force_single_line=False,
        single_line_exclusions=[],
        only_sections=True
    )
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and "module" not in config.single_line_exclusions)
    )


# LLM-generated content at query #31
#--------------------------

```python
def test_sorted_imports_with_no_import_index():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')"


# LLM-generated content at query #32
#--------------------------

```python
def test_sorted_imports_returns_string():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"FUTURE": {"straight": {}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed)
    assert isinstance(result, str)


# LLM-generated content at query #33
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #34
#--------------------------

```python
def test_sorted_imports_predicate():
    assert sorted_imports is not None


# LLM-generated content at query #35
#--------------------------

```python
def test_sorted_imports_predicate():
    assert sorted_imports is not None


# LLM-generated content at query #36
#--------------------------

```python
def test_with_from_imports_basic_case():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment1", "comment2"]}},
        as_map={"from": {"os.path": ["path as p"], "os.sys": ["sys as s"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(
        no_inline_sort=False,
        force_single_line=False,
        single_line_exclusions=[],
        only_sections=False,
        combine_as_imports=True,
        combine_star=False,
        ignore_comments=False,
        comment_prefix="#",
        force_grid_wrap=0,
        line_length=88,
        split_on_trailing_comma=False,
        multi_line_output=wrap.Modes.GRID,
    )
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == [
        "from os import path, sys  # comment1; comment2",
        "from os import path as p",
        "from os import sys as s",
    ]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(
        no_inline_sort=False,
        force_single_line=False,
        single_line_exclusions=[],
        only_sections=False,
        combine_as_imports=False,
        combine_star=False,
        ignore_comments=False,
        comment_prefix="#",
        force_grid_wrap=0,
        line_length=88,
        split_on_trailing_comma=False,
        multi_line_output=wrap.Modes.GRID,
    )
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=["os.sys"],
        import_type="import",
    )
    assert result == ["from os import path  # comment"]

def test_with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"from": {"os": ["star comment"]}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(
        no_inline_sort=False,
        force_single_line=False,
        single_line_exclusions=[],
        only_sections=False,
        combine_as_imports=False,
        combine_star=True,
        ignore_comments=False,
        comment_prefix="#",
        force_grid_wrap=0,
        line_length=88,
        split_on_trailing_comma=False,
        multi_line_output=wrap.Modes.GRID,
    )
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(
        no_inline_sort=False,
        force_single_line=True,
        single_line_exclusions=[],
        only_sections=False,
        combine_as_imports=False,
        combine_star=False,
        ignore_comments=False,
        comment_prefix="#",
        force_grid_wrap=0,
        line_length=88,
        split_on_trailing_comma=False,
        multi_line_output=wrap.Modes.GRID,
    )
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == [
        "from os import path  # comment",
        "from os import sys",
    ]

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(
        no_inline_sort=False,
        force_single_line=False,
        single_line_exclusions=[],
        only_sections=False,
        combine_as_imports=False,
        combine_star=False,
        ignore_comments=True,
        comment_prefix="#",
        force_grid_wrap=0,
        line_length=88,
        split_on_trailing_comma=False,
        multi_line_output=wrap.Modes.GRID,
    )
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path"]

def test_with_from_imports_nested_comment():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {"os": []},
            "nested": {"os": {"path": "nested comment"}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(
        no_inline_sort=False,
        force_single_line=False,
        single_line_exclusions=[],
        only_sections=False,
        combine_as_imports=False,
        combine_star=False,
        ignore_comments=False,
        comment_prefix="#",
        force_grid_wrap=0,
        line_length=88,
        split_on_trailing_comma=False,
        multi_line_output=wrap.Modes.GRID,
    )
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path  # nested comment"]

def test_with_from_imports_combine_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.path": ["path as p"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(
        no_inline_sort=False,
        force_single_line=False,
        single_line_exclusions=[],
        only_sections=False,
        combine_as_imports=True,
        combine_star=False,
        ignore_comments=False,
        comment_prefix="#",
        force_grid_wrap=0,
        line_length=88,
        split_on_trailing_comma=False,
        multi_line_output=wrap.Modes.GRID,
    )
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == [
        "from os import path, path as p  # comment",
    ]

def test_with_from_imports_only_sections():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.path": ["path as p"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(
        no_inline_sort=False,
        force_single_line=False,
        single_line_exclusions=[],
        only_sections=True,
        combine_as_imports=True,
        combine_star=False,
        ignore_comments=False,
        comment_prefix="#",
        force_grid_wrap=0,
        line_length=88,
        split_on_trailing_comma=False,
        multi_line_output=wrap.Modes.GRID,
    )
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == [
        "from os import path, path as p  # comment",
    ]

def test_with_from_imports_noqa_comment():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {"os": []},
            "nested": {"os": {"path": "noqa: F401"}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(
        no_inline_sort=False


# LLM-generated content at query #37
#--------------------------

```python
def test__with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["path", "sys"],
                }
            }
        },
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed, config, ["os"], "THIRDPARTY", [], "import"
    )
    assert result == ["from os import path, sys"]

def test__with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["path", "sys"],
                }
            }
        },
        categorized_comments={
            "from": {"os": ("# comment1", "# comment2")},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed, config, ["os"], "THIRDPARTY", [], "import"
    )
    assert result == ["from os import path, sys  # comment1; comment2"]

def test__with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["path", "sys"],
                }
            }
        },
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed, config, ["os"], "THIRDPARTY", ["os.path"], "import"
    )
    assert result == ["from os import sys"]

def test__with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["path"],
                }
            }
        },
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed, config, ["os"], "THIRDPARTY", [], "import"
    )
    assert result == ["from os import path as ospath"]

def test__with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["*"],
                }
            }
        },
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"os": {"*": "# star comment"}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed, config, ["os"], "THIRDPARTY", [], "import"
    )
    assert result == ["from os import *  # star comment"]

def test__with_from_imports_with_force_single_line():
    parsed = parse.ParsedContent(
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["path", "sys"],
                }
            }
        },
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(
        parsed, config, ["os"], "THIRDPARTY", [], "import"
    )
    assert result == ["from os import path", "from os import sys"]

def test__with_from_imports_with_combine_as_imports():
    parsed = parse.ParsedContent(
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["path"],
                }
            }
        },
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(
        parsed, config, ["os"], "THIRDPARTY", [], "import"
    )
    assert result == ["from os import path as ospath"]

def test__with_from_imports_with_ignore_comments():
    parsed = parse.ParsedContent(
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["path", "sys"],
                }
            }
        },
        categorized_comments={
            "from": {"os": ("# comment1", "# comment2")},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(
        parsed, config, ["os"], "THIRDPARTY", [], "import"
    )
    assert result == ["from os import path, sys"]

def test__with_from_imports_with_comment_prefix():
    parsed = parse.ParsedContent(
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["path", "sys"],
                }
            }
        },
        categorized_comments={
            "from": {"os": ("# comment1", "# comment2")},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(comment_prefix="# ")
    result = _with_from_imports(
        parsed, config, ["os"], "THIRDPARTY", [], "import"
    )
    assert result == ["from os import path, sys  # comment1; comment2"]

def test__with_from_imports_with_above_comments():
    parsed = parse.ParsedContent(
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["path", "sys"],
                }
            }
        },
        categorized_comments={
            "from": {},
            "above": {"from": {"os": ["# above comment"]}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed, config, ["os"], "THIRDPARTY", [], "import"
    )
    assert result == ["# above comment", "from os import path, sys"]


# LLM-generated content at query #38
#--------------------------

```python
def test_with_straight_imports_predicate_false():
    parsed = parse.ParsedContent(
        as_map={"straight": {"module1": ["alias1"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        imports={"STDLIB": {"straight": {"module1": []}}}
    )
    config = Config(combine_straight_imports=True)
    straight_modules = ["module1"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == []


# LLM-generated content at query #39
#--------------------------

```python
def test_with_straight_imports_no_modules():
    parsed = parse.ParsedContent(
        imports={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, [], "straight", [], "import") == []

def test_with_straight_imports_combine_no_comments():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["sys"], "os": ["os"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "straight", [], "import") == ["import sys, os"]

def test_with_straight_imports_combine_inline_comments():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["sys"], "os": ["os"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {"sys": ["comment1"], "os": ["comment2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "straight", [], "import") == ["import sys, os  # comment1 comment2"]

def test_with_straight_imports_combine_above_comments():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["sys"], "os": ["os"]}},
        categorized_comments={"above": {"straight": {"sys": ["# above comment"]}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "straight", [], "import") == ["# above comment", "import sys, os"]

def test_with_straight_imports_as_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["sys"], "os": ["os"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"sys": ["sys_alias"]}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "straight", [], "import") == ["import os", "import sys as sys_alias"]

def test_with_straight_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["sys"], "os": ["os"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "straight", ["sys"], "import") == ["import os"]

def test_with_straight_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["sys"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {"sys": ["comment"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False, ignore_comments=True)
    assert _with_straight_imports(parsed, config, ["sys"], "straight", [], "import") == ["import sys"]

def test_with_straight_imports_custom_comment_prefix():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["sys"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {"sys": ["comment"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False, comment_prefix=" # ")
    assert _with_straight_imports(parsed, config, ["sys"], "straight", [], "import") == ["import sys # comment"]


# LLM-generated content at query #40
#--------------------------

```python
def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        original_line_count=1,
        import_index=-1,
        line_separator="\n",
        imports={},
        sections=[],
        place_imports={},
        import_placements={}
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_sorted_imports_empty_parsed():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        sections=[],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == ""

def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        sections=[],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\n"

def test_sorted_imports_single_from_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={"THIRDPARTY": {"from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from os import path\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_multiple_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={"THIRDPARTY": {"from": {"os": {"path": set()}, "sys": {"argv": set()}}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from os import path\nfrom sys import argv\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {"os": ["# comment above"]}}, "straight": {"os": ["# inline comment"]}},
        as_map={"straight": {}},
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "# comment above\nimport os  # inline comment\n"

def test_sorted_imports_with_as_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["osp"]}},
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os as osp\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={"THIRDPARTY": {"straight": {"sys": set(), "os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n"


# LLM-generated content at query #2
#--------------------------

```python
def test_sorted_imports_with_empty_parsed_content():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        imports={},
        categorized_comments={},
        as_map={},
        sections=[],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\n"

def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        imports={},
        categorized_comments={},
        as_map={},
        sections=[],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

def test_sorted_imports_with_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n"

def test_sorted_imports_with_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\n"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from os import path\n\n"

def test_sorted_imports_with_combined_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["op_sys"]}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os as op_sys\n\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# comment above"]}}, "straight": {"os": ["# inline comment"]}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "# comment above\nimport os  # inline comment\n\n"

def test_sorted_imports_with_section_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"stdlib": "Standard Library"})
    result = sorted_imports(parsed, config)
    assert result == "# Standard Library\nimport os\n\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n\n"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}, "THIRDPARTY": {"straight": {"django": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB", "THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\nimport django\n\n"

def test_sorted_imports_with_ensure_newline_before_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["# comment"]}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n# comment\n\n"


# LLM-generated content at query #3
#--------------------------

```python
def test_sorted_imports_empty_parsed_content():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        imports={},
        as_map={},
        categorized_comments={},
        place_imports={},
        import_placements={},
        sections=[],
        forced_separate=[],
    )
    result = sorted_imports(parsed)
    assert result == "\n"

def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        imports={},
        as_map={},
        categorized_comments={},
        place_imports={},
        import_placements={},
        sections=[],
        forced_separate=[],
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')\n"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    result = sorted_imports(parsed)
    assert result == "import os\n\n"

def test_sorted_imports_single_from_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {}, "from": {"os": [("path", None)]}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    result = sorted_imports(parsed)
    assert result == "from os import path\n\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"sys": ["sys"], "os": ["os"]}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    result = sorted_imports(parsed)
    assert result == "import os, sys\n\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {"os": ["# Comment above"]}}, "straight": {"os": ["# Inline comment"]}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    result = sorted_imports(parsed)
    assert result == "# Comment above\nimport os  # Inline comment\n\n"

def test_sorted_imports_with_as_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        as_map={"straight": {"os": ["os_path"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    result = sorted_imports(parsed)
    assert result == "import os as os_path\n\n"

def test_sorted_imports_combine_straight_imports():
    config = Config(combine_straight_imports=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"sys": ["sys"], "os": ["os"]}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n\n"

def test_sorted_imports_remove_imports():
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"sys": ["sys"], "os": ["os"]}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    result = sorted_imports(parsed, config)
    assert result == "import sys\n\n"

def test_sorted_imports_with_section_heading():
    config = Config(import_headings={"stdlib": "Standard Library"})
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    result = sorted_imports(parsed, config)
    assert result == "# Standard Library\nimport os\n\n"

def test_sorted_imports_with_lines_between_sections():
    config = Config(lines_between_sections=2)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={
            "FUTURE": {"straight": {"__future__": ["__future__"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os"]}, "from": {}},
        },
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["FUTURE", "STDLIB"],
        forced_separate=[],
    )
    result = sorted_imports(parsed, config)
    assert result == "import __future__\n\n\n\nimport os\n\n"


# LLM-generated content at query #4
#--------------------------

```python
def test_empty_input():
    assert _ensure_newline_before_comment([]) == []

def test_single_comment_line():
    assert _ensure_newline_before_comment(["# comment"]) == ["# comment"]

def test_comment_after_empty_line():
    assert _ensure_newline_before_comment(["", "# comment"]) == ["", "# comment"]

def test_comment_after_non_empty_line():
    assert _ensure_newline_before_comment(["code", "# comment"]) == ["code", "", "# comment"]

def test_multiple_comments():
    assert _ensure_newline_before_comment(["# comment1", "# comment2"]) == ["# comment1", "# comment2"]

def test_comment_after_code_with_existing_newline():
    assert _ensure_newline_before_comment(["code", "", "# comment"]) == ["code", "", "# comment"]

def test_mixed_lines():
    assert _ensure_newline_before_comment(["code1", "# comment1", "code2", "# comment2"]) == ["code1", "", "# comment1", "code2", "", "# comment2"]

def test_no_comments():
    assert _ensure_newline_before_comment(["line1", "line2", "line3"]) == ["line1", "line2", "line3"]

def test_comment_at_start():
    assert _ensure_newline_before_comment(["# comment", "code"]) == ["# comment", "code"]

def test_comment_at_end():
    assert _ensure_newline_before_comment(["code", "# comment"]) == ["code", "", "# comment"]

def test_multiple_empty_lines_before_comment():
    assert _ensure_newline_before_comment(["code", "", "", "# comment"]) == ["code", "", "", "# comment"]


# LLM-generated content at query #5
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        imports={},
        categorized_comments={},
        as_map={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    result = sorted_imports(parsed)
    assert result == "import os\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    result = sorted_imports(parsed)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={
            "above": {"straight": {"os": ["# Comment above os"]}},
            "straight": {"os": ["# Inline comment for os"]},
        },
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    result = sorted_imports(parsed)
    assert result == "# Comment above os\nimport os  # Inline comment for os\n"

def test_sorted_imports_combine_straight_imports():
    config = Config(combine_straight_imports=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": {"ospath"}}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    result = sorted_imports(parsed)
    assert result == "import os as ospath\n"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"from": {}}, "from": {}},
        as_map={"from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    result = sorted_imports(parsed, import_type="from")
    assert result == "from os import path\n"

def test_sorted_imports_with_section_heading():
    config = Config(import_headings={"stdlib": "Standard Library"})
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    result = sorted_imports(parsed, config=config)
    assert result == "# Standard Library\nimport os\n"

def test_sorted_imports_with_force_sort_within_sections():
    config = Config(force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"sys": set(), "os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_remove_imports():
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import sys\n"

def test_sorted_imports_with_lines_between_sections():
    config = Config(lines_between_sections=2)
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={
            "FUTURE": {"straight": {"__future__": set()}},
            "STDLIB": {"straight": {"os": set()}},
        },
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["FUTURE", "STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import __future__\n\n\nimport os\n"


# LLM-generated content at query #6
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment1", "comment2"]}},
        line_separator="\n",
        as_map={"from": {}},
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path, sys  # comment1; comment2"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment1"]}},
        line_separator="\n",
        as_map={"from": {}},
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", ["os.path"], "import")
    assert result == ["from os import sys  # comment1"]

def test_with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"nested": {"os": {"*": ["star comment"]}}},
        line_separator="\n",
        as_map={"from": {}},
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment1"]}},
        line_separator="\n",
        as_map={"from": {"os.path": ["path as ospath"]}},
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path as ospath  # comment1"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment1"]}},
        line_separator="\n",
        as_map={"from": {}},
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path", "from os import sys"]

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment1"]}},
        line_separator="\n",
        as_map={"from": {}},
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]


# LLM-generated content at query #7
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        imports={},
        categorized_comments={},
        as_map={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

def test_sorted_imports_single_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\n"

def test_sorted_imports_multiple_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# comment above"]}}, "straight": {"os": ["# inline comment"]}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\n# comment above\nimport os  # inline comment\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["os_path"]}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nimport os as os_path\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "\nimport sys\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os, sys\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"sys": set(), "os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\nimport sys\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "\n# Third Party Imports\nimport os\n"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "FUTURE": {"straight": set(), "from": {"__future__": {"print_function"}}},
            "THIRDPARTY": {"straight": {"os": set()}, "from": {}},
        },
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["FUTURE", "THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(lines_between_sections=1)
    result = sorted_imports(parsed, config)
    assert result == "\nfrom __future__ import print_function\n\nimport os\n"

def test_sorted_imports_with_ensure_newline_before_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["# comment"]}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os  # comment\n"

def test_sorted_imports_with_lines_after_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\n\n\nprint('hello')"

def test_sorted_imports_with_formatting_function():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight


# LLM-generated content at query #8
#--------------------------

```python
def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        import_index=-1,
        lines_without_imports=["print('Hello, world!')"],
        line_separator="\n",
        imports={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed)
    assert result == "print('Hello, world!')"


# LLM-generated content at query #9
#--------------------------

```python
def test_with_star_comments_when_star_comment_exists():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {module: {"*": "star_comment"}}}
    module = "test_module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2", "star_comment"]

def test_with_star_comments_when_star_comment_does_not_exist():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {module: {}}}
    module = "test_module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2"]

def test_with_star_comments_when_module_does_not_exist():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {}}
    module = "test_module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2"]


# LLM-generated content at query #10
#--------------------------

```python
def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        original_line_count=1,
        import_index=-1,
        line_separator="\n",
        imports={},
        sections=[],
        place_imports={},
        import_placements={}
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"


# LLM-generated content at query #11
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        imports={},
        as_map={},
        categorized_comments={},
        place_imports={},
        import_placements={},
        sections=[],
        original_line_count=1,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "print('hello')"

def test_sorted_imports_with_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os\n\nprint('hello')"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config(force_sort_within_sections=True)
    assert sorted_imports(parsed, config) == "import os\nimport sys\n\nprint('hello')"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config(remove_imports=["os"])
    assert sorted_imports(parsed, config) == "print('hello')"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config(combine_straight_imports=True)
    assert sorted_imports(parsed, config) == "import os, sys\n\nprint('hello')"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config(import_headings={"stdlib": "Standard Library"})
    assert sorted_imports(parsed, config) == "# Standard Library\nimport os\n\nprint('hello')"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}, "THIRDPARTY": {"straight": {"numpy": set()}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB", "THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(lines_between_sections=2)
    assert sorted_imports(parsed, config) == "import os\n\n\nimport numpy\n\nprint('hello')"

def test_sorted_imports_with_ensure_newline_before_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config(ensure_newline_before_comments=True)
    assert sorted_imports(parsed, config) == "import os\n\nprint('hello')"

def test_sorted_imports_with_formatting_function():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config(formatting_function=lambda x, y, z: x.upper())
    assert sorted_imports(parsed, config) == "IMPORT OS\n\nPRINT('HELLO')"

def test_sorted_imports_with_lines_before_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config(lines_before_imports=2)
    assert sorted_imports(parsed, config) == "\n\nimport os\n\nprint('hello')"

def test_sorted_imports_with_lines_after_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config(lines_after_imports=2)
    assert sorted_imports(parsed, config) == "import os\n\n\nprint('hello')"

def test_sorted_imports_with_place_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={"STDLIB": ["import sys"]},
        import_placements={"print('hello')": "STDLIB"},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os\n\nprint('hello')\nimport sys\n"


# LLM-generated content at query #12
#--------------------------

```python
def test_with_straight_imports_empty_modules():
    parsed = parse.ParsedContent(
        imports={"straight": {}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, [], "straight", [], "import") == []

def test_with_straight_imports_combine_no_comments():
    parsed = parse.ParsedContent(
        imports={"straight": {"a": [], "b": []}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["a", "b"], "straight", [], "import") == ["import a, b"]

def test_with_straight_imports_combine_inline_comments():
    parsed = parse.ParsedContent(
        imports={"straight": {"a": [], "b": []}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {"a": ["comment1"], "b": ["comment2"]}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["a", "b"], "straight", [], "import") == ["import a, b  # comment1 comment2"]

def test_with_straight_imports_combine_above_comments():
    parsed = parse.ParsedContent(
        imports={"straight": {"a": [], "b": []}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {"a": ["above1"], "b": ["above2"]}}, "straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["a", "b"], "straight", [], "import") == ["above1", "above2", "import a, b"]

def test_with_straight_imports_no_combine():
    parsed = parse.ParsedContent(
        imports={"straight": {"a": [], "b": []}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config(combine_straight_imports=False)
    assert _with_straight_imports(parsed, config, ["a", "b"], "straight", [], "import") == ["import a", "import b"]

def test_with_straight_imports_as_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"a": [], "b": []}},
        as_map={"straight": {"a": ["alias1"], "b": ["alias2"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["a", "b"], "straight", [], "import") == ["import a as alias1", "import b as alias2"]

def test_with_straight_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"a": [], "b": []}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["a", "b"], "straight", ["a"], "import") == ["import b"]

def test_with_straight_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"straight": {"a": []}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {"a": ["comment1"]}},
    )
    config = Config(combine_straight_imports=False, ignore_comments=True)
    assert _with_straight_imports(parsed, config, ["a"], "straight", [], "import") == ["import a"]

def test_with_straight_imports_comment_prefix():
    parsed = parse.ParsedContent(
        imports={"straight": {"a": []}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {"a": ["comment1"]}},
    )
    config = Config(combine_straight_imports=False, comment_prefix=" # ")
    assert _with_straight_imports(parsed, config, ["a"], "straight", [], "import") == ["import a # comment1"]


# LLM-generated content at query #13
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {"os.path": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ("comment",)}},
        as_map={"from": {"os.path": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # comment"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {"os.path": [], "os.sys": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", ["os.sys"], "import")
    assert result == ["from os import path"]

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path as ospath"]

def test_with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"nested": {"os": {"*": "star comment"}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {"os.path": [], "os.sys": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path", "from os import sys"]

def test_with_from_imports_with_nested_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"nested": {"os": {"path": "nested comment"}}},
        as_map={"from": {"os.path": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # nested comment"]

def test_with_from_imports_with_above_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"above": {"from": {"os": ["above comment"]}}},
        as_map={"from": {"os.path": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["above comment", "from os import path"]

def test_with_from_imports_with_straight_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"straight": {"os.path": ["straight comment"]}},
        as_map={"from": {"os.path": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # straight comment"]

def test_with_from_imports_with_noqa_comment():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"nested": {"os": {"path": "noqa"}}},
        as_map={"from": {"os.path": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False, multi_line_output=wrap.Modes.HANGING_INDENT)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # noqa"]


# LLM-generated content at query #14
#--------------------------

```python
def test_sorted_imports_empty_parsed_content():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        sections=[],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\n"

def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        sections=[],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": []}, "from": {}}, "straight": {"os": []}, "from": {}},
        as_map={"straight": {"os": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n"

def test_sorted_imports_single_from_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"straight": {}, "from": {"os": []}}, "straight": {}, "from": {"os": []}},
        as_map={"straight": {}, "from": {"os": []}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from os import path\n\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": [], "sys": []}, "from": {}}, "straight": {"os": [], "sys": []}, "from": {}},
        as_map={"straight": {"os": [], "sys": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\n"

def test_sorted_imports_multiple_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}, "sys": {"argv": set()}}}},
        categorized_comments={"above": {"straight": {}, "from": {"os": [], "sys": []}}, "straight": {}, "from": {"os": [], "sys": []}},
        as_map={"straight": {}, "from": {"os": [], "sys": []}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from os import path\nfrom sys import argv\n\n"

def test_sorted_imports_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": [], "sys": []}, "from": {}}, "straight": {"os": [], "sys": []}, "from": {}},
        as_map={"straight": {"os": [], "sys": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# Comment above"]}, "from": {}}, "straight": {"os": ["# Inline comment"]}, "from": {}},
        as_map={"straight": {"os": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "# Comment above\nimport os  # Inline comment\n\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": []}, "from": {}}, "straight": {"os": []}, "from": {}},
        as_map={"straight": {"os": ["osp"]}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport os as osp\n\n"

def test_sorted_imports_with_section_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": []}, "from": {}}, "straight": {"os": []}, "from": {}},
        as_map={"straight": {"os": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"stdlib": "Standard Library"})
    result = sorted_imports(parsed, config)
    assert result == "# Standard Library\nimport os\n\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": [], "sys": []}, "from": {}}, "straight": {"os": [], "sys": []}, "from": {}},
        as_map={"straight": {"os": [], "sys": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\n"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}, "THIRDPARTY": {"straight": {"django": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": [], "django":


# LLM-generated content at query #15
#--------------------------

```python
def test_as_imports_predicate():
    parsed = parse.ParsedContent(
        as_map={"straight": {"module1": ["alias1"], "module2": []}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        imports={"section": {"straight": {"module1": [], "module2": []}}}
    )
    straight_modules = ["module1", "module2"]
    assert any(module in parsed.as_map["straight"] for module in straight_modules)


# LLM-generated content at query #16
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        as_map={"from": {"os.path": ["os.path as osp"]}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        as_map={"from": {}},
        categorized_comments={
            "from": {"os": ["# comment"]},
            "above": {"from": {}},
            "nested": {},
        },
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # comment"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        as_map={"from": {}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", ["os.sys"], "import")
    assert result == ["from os import path"]

def test_with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        as_map={"from": {}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"os": {"*": ["# star comment"]}},
        },
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        as_map={"from": {"os.path": ["path as osp"]}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path as osp"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        as_map={"from": {}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path", "from os import sys"]

def test_with_from_imports_multiline():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        as_map={"from": {}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(line_length=20)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import (\n    path,\n    sys,\n)"]

def test_with_from_imports_no_inline_sort():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["sys", "path"]}}},
        as_map={"from": {}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(no_inline_sort=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import sys, path"]

def test_with_from_imports_only_sections():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        as_map={"from": {}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(only_sections=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]


# LLM-generated content at query #17
#--------------------------

```python
def test_sorted_imports_basic_case():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[""],
        line_separator="\n",
        imports={
            "STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {}},
            "FIRSTPARTY": {"straight": {}, "from": {}},
            "LOCALFOLDER": {"straight": {}, "from": {}},
        },
        sections=["STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[""],
        line_separator="\n",
        imports={
            "STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {}},
            "FIRSTPARTY": {"straight": {}, "from": {}},
            "LOCALFOLDER": {"straight": {}, "from": {}},
        },
        sections=["STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[""],
        line_separator="\n",
        imports={
            "STDLIB": {"straight": {}, "from": {"os": {"path": set()}}},
            "THIRDPARTY": {"straight": {}, "from": {}},
            "FIRSTPARTY": {"straight": {}, "from": {}},
            "LOCALFOLDER": {"straight": {}, "from": {}},
        },
        sections=["STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from os import path\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[""],
        line_separator="\n",
        imports={
            "STDLIB": {"straight": {"os": set()}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {}},
            "FIRSTPARTY": {"straight": {}, "from": {}},
            "LOCALFOLDER": {"straight": {}, "from": {}},
        },
        sections=["STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"],
        categorized_comments={"above": {"straight": {"os": ["# OS import"]}}, "straight": {"os": ["# OS module"]}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "# OS import\nimport os  # OS module\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[""],
        line_separator="\n",
        imports={
            "STDLIB": {"straight": {"os": set()}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {}},
            "FIRSTPARTY": {"straight": {}, "from": {}},
            "LOCALFOLDER": {"straight": {}, "from": {}},
        },
        sections=["STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["osp"]}},
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport os as osp\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[""],
        line_separator="\n",
        imports={
            "STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {}},
            "FIRSTPARTY": {"straight": {}, "from": {}},
            "LOCALFOLDER": {"straight": {}, "from": {}},
        },
        sections=["STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(remove_imports=["sys"])
    result = sorted_imports(parsed, config)
    assert result == "import os\n"

def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        import_index=-1,
        lines_without_imports=["print('hello')"],
        line_separator="\n",
        imports={
            "STDLIB": {"straight": {}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {}},
            "FIRSTPARTY": {"straight": {}, "from": {}},
            "LOCALFOLDER": {"straight": {}, "from": {}},
        },
        sections=["STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[""],
        line_separator="\n",
        imports={
            "STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {}},
            "FIRSTPARTY": {"straight": {}, "from": {}},
            "LOCALFOLDER": {"straight": {}, "from": {}},
        },
        sections=["STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[""],
        line_separator="\n",
        imports={
            "STDLIB": {"straight": {"os": set()}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {}},
            "FIRSTPARTY": {"straight": {}, "from": {}},
            "LOCALFOLDER": {"straight": {}, "from": {}},
        },
        sections=["STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(import


# LLM-generated content at query #18
#--------------------------

```python
def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=[],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"


# LLM-generated content at query #19
#--------------------------

```python
def test_with_straight_imports_combined_straight_imports():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": [], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": [], "sys": []}},
    )
    config = Config(combine_straight_imports=True, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == ["import os, sys"]

def test_with_straight_imports_combined_straight_imports_with_inline_comments():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": [], "sys": []}}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"os": ["comment1"], "sys": ["comment2"]},
        },
    )
    config = Config(combine_straight_imports=True, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == ["import os, sys  # comment1 comment2"]

def test_with_straight_imports_combined_straight_imports_with_above_comments():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": [], "sys": []}}},
        categorized_comments={
            "above": {"straight": {"os": ["# comment1"], "sys": ["# comment2"]}},
            "straight": {"os": [], "sys": []},
        },
    )
    config = Config(combine_straight_imports=True, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == ["# comment1", "# comment2", "import os, sys"]

def test_with_straight_imports_combined_straight_imports_with_as_imports():
    parsed = parse.ParsedContent(
        as_map={"straight": {"os": ["os_path"]}},
        imports={"STDLIB": {"straight": {"os": ["os_path"]}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": []}},
    )
    config = Config(combine_straight_imports=True, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == ["import os as os_path"]

def test_with_straight_imports_combined_straight_imports_removed():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": [], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": [], "sys": []}},
    )
    config = Config(combine_straight_imports=True, ignore_comments=True, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == ["import os, sys"]

def test_with_straight_imports_combined_straight_imports_empty():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config(combine_straight_imports=True, ignore_comments=False, comment_prefix="# ")
    straight_modules = []
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == []

def test_with_straight_imports_combined_straight_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": [], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": [], "sys": []}},
    )
    config = Config(combine_straight_imports=False, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = ["os"]
    import_type = "import"
    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == ["import sys"]


# LLM-generated content at query #20
#--------------------------

```python
def test_sorted_imports_predicate():
    assert sorted_imports is not None


# LLM-generated content at query #21
#--------------------------

```python
def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        import_index=-1,
        lines_without_imports=["print('hello')"],
        line_separator="\n",
        imports={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')"


# LLM-generated content at query #22
#--------------------------

```python
def test_with_star_comments():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {"module": {"*": "star comment"}}}
    result = _with_star_comments(parsed, "module", ["comment1"])
    assert result == ["comment1", "star comment"]

def test_with_star_comments_no_star():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {"module": {}}}
    result = _with_star_comments(parsed, "module", ["comment1"])
    assert result == ["comment1"]

def test_with_from_imports_empty():
    parsed = parse.ParsedContent()
    config = Config()
    result = _with_from_imports(parsed, config, [], "section", [], "import_type")
    assert result == []

def test_with_from_imports_single_module():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"from": {"module": ["import1"]}}}
    parsed.as_map = {"from": {}}
    parsed.categorized_comments = {"from": {}, "above": {"from": {}}, "nested": {}}
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1"]

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"from": {"module": ["import1"]}}}
    parsed.as_map = {"from": {}}
    parsed.categorized_comments = {"from": {"module": ["comment1"]}, "above": {"from": {}}, "nested": {}}
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1  # comment1"]

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"from": {"module": ["import1"]}}}
    parsed.as_map = {"from": {"module.import1": ["import1 as alias1"]}}
    parsed.categorized_comments = {"from": {}, "above": {"from": {}}, "nested": {}}
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1 as alias1"]

def test_with_from_imports_with_star():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"from": {"module": ["*"]}}}
    parsed.as_map = {"from": {}}
    parsed.categorized_comments = {"nested": {"module": {"*": "star comment"}}, "from": {}, "above": {"from": {}}}
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import *  # star comment"]

def test_with_from_imports_with_remove_imports():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"from": {"module": ["import1", "import2"]}}}
    parsed.as_map = {"from": {}}
    parsed.categorized_comments = {"from": {}, "above": {"from": {}}, "nested": {}}
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", ["module.import1"], "import")
    assert result == ["from module import import2"]


# LLM-generated content at query #23
#--------------------------

```python
def test_sorted_imports_predicate():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[""],
        line_separator="\n",
        imports={},
        place_imports={},
        import_placements={},
        sections=[],
        original_line_count=1
    )
    config = Config()
    assert sorted_imports(parsed, config) == "\n"


# LLM-generated content at query #24
#--------------------------

```python
def test_sorted_imports_empty_parsed_content():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        imports={},
        categorized_comments={},
        as_map={},
        sections=[],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "\n"

def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        imports={},
        categorized_comments={},
        as_map={},
        sections=[],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "print('hello')\n"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "import os\n\n"

def test_sorted_imports_single_from_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {}, "from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "from os import path\n\n"

def test_sorted_imports_multiple_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"os": {"path": set()}},
            }
        },
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "from os import path\n\nimport os\nimport sys\n\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={
            "above": {"straight": {"os": ["# comment above"]}},
            "straight": {"os": ["# inline comment"]},
        },
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "# comment above\nimport os  # inline comment\n\n"

def test_sorted_imports_with_as_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["ospath"]}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "import os as ospath\n\n"

def test_sorted_imports_with_removed_imports():
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed, config) == "import sys\n\n"

def test_sorted_imports_with_combine_straight_imports():
    config = Config(combine_straight_imports=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed, config) == "import os, sys\n\n"

def test_sorted_imports_with_import_headings():
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed, config) == "# Third Party Imports\nimport os\n\n"

def test_sorted_imports_with_import_footers():
    config = Config(import_footers={"thirdparty": "End of Third Party Imports"})
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed, config) == "import os\n\n# End of Third Party Imports\n"

def test_sorted_imports_with_lines_between_sections():
    config = Config(lines_between_sections=2)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={
            "FUTURE": {"straight": {"__future__": set()}, "from": {}},
            "THIRDPARTY": {"straight": {"os": set()}, "from": {}},
        },
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["FUTURE", "THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed, config) == "import __future__\n\n\n\nimport os\n\n"

def test_sorted_imports_with_force_sort_within_sections():
    config = Config(force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"sys": set(), "os": set()},
                "from": {"os": {"path": set()}},
            }
        },
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed, config) == "from os import path\nimport os\nimport sys\n\n"

def test_sorted_imports_with_lines_after_imports():
    config = Config(lines_after_imports=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        original


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_1():
    parsed = parse.ParsedContent()
    config = Config()
    from_modules = ["module1", "module2"]
    section = "section1"
    remove_imports = []
    import_type = "import"

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)

    assert isinstance(result, list)


# LLM-generated content at query #26
#--------------------------

```python
def test_sorted_imports_when_import_index_is_not_negative_one():
    parsed = parse.ParsedContent(import_index=0, lines_without_imports=[], line_separator="\n")
    config = DEFAULT_CONFIG
    extension = "py"
    import_type = "import"
    result = sorted_imports(parsed, config, extension, import_type)
    assert result == "\n"


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_1_evaluates_to_false():
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    )


# LLM-generated content at query #28
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path, sys"]

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={
            "from": {"os": ["comment1", "comment2"]},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path, sys  # comment1; comment2"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", ["os.path"], "import")
    assert result == ["from os import sys"]

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path as ospath"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path", "from os import sys"]

def test_with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"os": {"*": ["star comment"]}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_multiline_reformat():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys", "other"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(line_length=20)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert len(result) > 1


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #30
#--------------------------

```python
def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')"


# LLM-generated content at query #31
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        imports={},
        categorized_comments={},
        as_map={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')"

def test_sorted_imports_single_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed)
    assert result == "\nimport os"

def test_sorted_imports_multiple_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed)
    assert result == "\nimport os\nimport sys"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["# comment"]}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed)
    assert result == "\nimport os  # comment"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["path"]}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed)
    assert result == "\nimport os as path"

def test_sorted_imports_with_force_sort_within_sections():
    config = Config(force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"sys": set(), "os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\nimport sys"

def test_sorted_imports_with_combine_straight_imports():
    config = Config(combine_straight_imports=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed, config)
    assert result == "\nimport os, sys"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"from": {}}, "from": {}},
        as_map={"from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed, import_type="from")
    assert result == "\nfrom os import path"

def test_sorted_imports_with_star_imports():
    config = Config(star_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"from": {"os": {"*": set(), "path": set()}}}},
        categorized_comments={"above": {"from": {}}, "from": {}},
        as_map={"from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed, config, import_type="from")
    assert result == "\nfrom os import *\nfrom os import path"

def test_sorted_imports_with_import_headings():
    config = Config(import_headings={"stdlib": "Standard Library"})
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed, config)
    assert result == "\n# Standard Library\nimport os"

def test_sorted_imports_with_import_footers():
    config = Config(import_footers={"stdlib": "End of Standard Library"})
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\n\n# End of Standard Library"

def test_sorted_imports_with_ensure_newline_before_comments():
    config = Config(ensure_newline_before_comments=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed, config)
    assert result == "\nimport os"

def test_sorted_imports_with_lines_after_imports():
    config = Config(lines_after_imports=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\n\n\nprint('hello')"

def test_sorted_imports_with_lines_before_imports():
    config = Config(lines_before_imports=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    result = sorted_imports(parsed, config)
    assert result == "\


# LLM-generated content at query #32
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment1", "comment2"]}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path, sys  # comment1; comment2"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment1"]}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=["os.path"],
        import_type="import",
    )
    assert result == ["from os import sys  # comment1"]

def test_with_from_imports_star_comment():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"nested": {"os": {"*": "star comment"}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment1"]}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path as ospath  # comment1"]

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment1"]}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path"]

def test_with_from_imports_combine_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment1"]}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path as ospath  # comment1"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment1"]}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path", "from os import sys  # comment1"]

def test_with_from_imports_above_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {"os": ["comment1"]},
            "above": {"from": {"os": ["above comment"]}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["above comment", "from os import path  # comment1"]

def test_with_from_imports_nested_comment():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"nested": {"os": {"path": "nested comment"}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path  # nested comment"]

def test_with_from_imports_noqa_comment():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"nested": {"os": {"path": "noqa: F401"}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(multi_line_output=wrap.Modes.HANGING_INDENT)
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path  # noqa: F401"]


# LLM-generated content at query #33
#--------------------------

```python
def test_sorted_imports_with_no_import_index():
    parsed = parse.ParsedContent(
        import_index=-1,
        lines_without_imports=["print('hello')"],
        line_separator="\n",
        imports={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')"


# LLM-generated content at query #34
#--------------------------

```python
def test_sorted_imports_predicate_false():
    parsed = parse.ParsedContent(
        import_index=-1,
        lines_without_imports=["line1", "line2"],
        line_separator="\n",
        imports={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=2
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "line1\nline2"


# LLM-generated content at query #35
#--------------------------

```python
def test_sorted_imports_with_empty_parsed_content():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=-1,
        original_line_count=0,
        sections=[],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    assert sorted_imports(parsed) == "\n"

def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=-1,
        original_line_count=1,
        sections=[],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    assert sorted_imports(parsed) == "print('hello')\n"

def test_sorted_imports_with_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": []}, "from": {}}, "straight": {"os": []}, "from": {}},
        as_map={"straight": {"os": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(combine_straight_imports=False)
    assert sorted_imports(parsed, config) == "import os\n\n"

def test_sorted_imports_with_combined_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": ["os"], "sys": ["sys"]}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": [], "sys": []}, "from": {}}, "straight": {"os": [], "sys": []}, "from": {}},
        as_map={"straight": {"os": [], "sys": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(combine_straight_imports=True)
    assert sorted_imports(parsed, config) == "import os, sys\n\n"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {}, "from": {"os": ["path"]}}},
        categorized_comments={"above": {"straight": {}, "from": {"os": []}}, "straight": {}, "from": {"os": []}},
        as_map={"straight": {}, "from": {}},
        import_index=0,
        original_line_count=1,
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    assert sorted_imports(parsed) == "from os import path\n\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# Comment above"]}, "from": {}}, "straight": {"os": ["# Inline comment"]}, "from": {}},
        as_map={"straight": {"os": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(ignore_comments=False)
    assert sorted_imports(parsed, config) == "# Comment above\nimport os  # Inline comment\n\n"

def test_sorted_imports_with_removed_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": ["os"], "sys": ["sys"]}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": [], "sys": []}, "from": {}}, "straight": {"os": [], "sys": []}, "from": {}},
        as_map={"straight": {"os": [], "sys": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(remove_imports=["os"])
    assert sorted_imports(parsed, config) == "import sys\n\n"

def test_sorted_imports_with_custom_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"CUSTOM": {"straight": {"custom": ["custom"]}, "from": {}}},
        categorized_comments={"above": {"straight": {"custom": []}, "from": {}}, "straight": {"custom": []}, "from": {}},
        as_map={"straight": {"custom": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        sections=["CUSTOM"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(import_headings={"custom": "Custom Imports"})
    assert sorted_imports(parsed, config) == "# Custom Imports\nimport custom\n\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"sys": ["sys"], "os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {"sys": [], "os": []}, "from": {}}, "straight": {"sys": [], "os": []}, "from": {}},
        as_map={"straight": {"sys": [], "os": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(force_sort_within_sections=True)
    assert sorted_imports(parsed, config) == "import os\nimport sys\n\n"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}, "THIRDPARTY": {"straight": {"requests": ["requests"]}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": [], "requests": []}, "from": {}}, "straight": {"os": [], "requests": []}, "from": {}},
        as_map={"straight": {"os": [], "requests": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        sections=["STDLIB", "THIRDPARTY"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    assert sorted_imports(parsed, config) == "import os\n\n\nimport requests\n\n"

def test_sorted_imports_with_formatting_function():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": []}, "from": {}}, "straight": {"os": []}, "from": {}},
        as_map={"straight": {"os": []}, "from": {}},
        import_index=0,
        original_line_count=1,
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(formatting_function=lambda x, y, z: x.upper())
    assert sorted_imports(parsed, config) == "IMPORT OS\n\n"

def test_sorted_imports_with_place_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["# Placeholder"],
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": []}, "from": {}}, "straight": {"os": []}, "from": {}},
        as_map


